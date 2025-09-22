# 12_rational_identities.py
# --------------------------------------------------------------------------------
# Identity-first PTM pipeline + exact (rational) transform identities and LLL reduction
#
# What’s new vs your last script(s):
#   • Exact rational nullspace for transform identities (small denominators)
#   • Integer lattice basis from the rational nullspace
#   • LLL lattice reduction to get short/sparse-ish integer identities
#   • CSV export of both float-based and rational/LLL-based identities
#   • Keeps your PM/PTM LP with OEP + OEM + α_T intact
#   • NEW: Explicit detection/logging of prep+transform == another prep (split-prep paradox)
#
# Dependencies: numpy, scipy.optimize.linprog (already used). No external LLL lib; we implement LLL here.

import numpy as np
from itertools import combinations, product, permutations
from fractions import Fraction
from scipy.optimize import linprog
import csv, os

# =============================================================================
# --- Linear-identity utilities (float SVD helpers you already used) ----------
# =============================================================================

def generating_state_identities(states, tol=1e-10):
    S = np.asarray(states)
    U, s, Vt = np.linalg.svd(S, full_matrices=True)
    rank = (s > tol).sum()
    return Vt[rank:, :]  # rows α with S @ α = 0

def generating_effect_identities(effects, tol=1e-10):
    E = np.asarray(effects)
    U, s, Vt = np.linalg.svd(E, full_matrices=True)
    rank = (s > tol).sum()
    return Vt[rank:, :]

def generating_transform_identities_float(Ts, tol=1e-10):
    Tflat = np.column_stack([np.asarray(T).reshape(-1, 1) for T in Ts])  # (dim^2) x T
    U, s, Vt = np.linalg.svd(Tflat, full_matrices=True)
    rank = (s > tol).sum()
    return Vt[rank:, :]  # rows α s.t. sum_t α_t T_t = 0

def _normalize_vec(a, zero_tol=1e-12):
    a = np.array(a, float)
    a[np.abs(a) < zero_tol] = 0.0
    m = np.max(np.abs(a)) if a.size else 0.0
    if m > 0:
        a = a / m
        for x in a:
            if abs(x) > 0:
                if x < 0: a = -a
                break
    return a

def _alpha_residual_matrix(Ts, alpha):
    R = sum(float(alpha[t]) * Ts[t] for t in range(len(Ts)))
    return R, float(np.max(np.abs(R)))

# =============================================================================
# --- Exact rational nullspace (Gaussian elimination over Fractions) ----------
# =============================================================================

def as_fraction_matrix(Ts, max_den=1):
    """
    Stack transforms into Tflat (dim^2 x T), and convert each entry to Fraction with limit_denominator(max_den).
    For Clifford/stabilizer maps entries are integers in {-1,0,1}, so max_den=1 suffices (exact).
    """
    Tflat = np.column_stack([np.asarray(T).reshape(-1, 1) for T in Ts])
    m, n = Tflat.shape
    A = [[Fraction(Tflat[i, j]).limit_denominator(max_den) for j in range(n)] for i in range(m)]
    return A  # list of lists of Fraction

def rref_fraction(A):
    """
    Reduced row echelon form over Fractions. Returns (R, pivots), where:
      - R is the RREF matrix (list of lists of Fraction)
      - pivots is a list of pivot column indices (in increasing order)
    """
    A = [row[:] for row in A]
    m = len(A); n = len(A[0]) if m else 0
    i = 0
    pivots = []
    for j in range(n):
        pivot = None
        for r in range(i, m):
            if A[r][j] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        A[i], A[pivot] = A[pivot], A[i]
        piv = A[i][j]
        A[i] = [x / piv for x in A[i]]
        for r in range(m):
            if r != i and A[r][j] != 0:
                fac = A[r][j]
                A[r] = [A[r][c] - fac * A[i][c] for c in range(n)]
        pivots.append(j)
        i += 1
        if i == m:
            break
    return A, pivots

def rational_nullspace_fraction(A):
    """
    Compute a basis of the rational nullspace of A (m x n) over Fractions using RREF.
    Returns a list of basis vectors (each a list of Fractions of length n).
    """
    m = len(A); n = len(A[0]) if m else 0
    R, pivots = rref_fraction(A)
    pivset = set(pivots)
    free_cols = [j for j in range(n) if j not in pivset]
    basis = []
    for f in free_cols:
        v = [Fraction(0) for _ in range(n)]
        v[f] = Fraction(1)
        for i_row, j_piv in enumerate(pivots):
            s = Fraction(0)
            for j in range(n):
                if j == j_piv:
                    continue
                if R[i_row][j] != 0 and v[j] != 0:
                    s += R[i_row][j] * v[j]
            v[j_piv] = -s
        basis.append(v)
    return basis  # list of Fraction vectors

def integer_kernel_basis_from_rational(basis_frac):
    """
    Given a rational nullspace basis (list of vectors of Fractions), convert each to a primitive integer vector
    by multiplying by the LCM of denominators and dividing by the GCD of numerators.
    Returns a list of integer numpy arrays (1D), forming a Z-basis of the integer solution lattice.
    """
    int_basis = []
    for v in basis_frac:
        dens = [f.denominator for f in v]
        lcm = 1
        for d in dens:
            a, b = lcm, d
            while b:
                a, b = b, a % b
            gcd = a
            lcm = (lcm // gcd) * d
        ints = [int(f.numerator * (lcm // f.denominator)) for f in v]
        g = 0
        for x in ints:
            g = abs(x) if g == 0 else gcd_int(g, x)
        if g > 0:
            ints = [x // g for x in ints]
        for x in ints:
            if x != 0:
                if x < 0:
                    ints = [-y for y in ints]
                break
        int_basis.append(np.array(ints, dtype=int))
    return int_basis

def gcd_int(a, b):
    a = abs(int(a)); b = abs(int(b))
    while b:
        a, b = b, a % b
    return a

# =============================================================================
# --- LLL lattice reduction (integer basis -> shorter/sparser vectors) --------
# =============================================================================

def lll_reduce(B, delta=0.75):
    """
    LLL reduction for integer lattice basis.
    Input: B with columns as basis vectors (shape n x k)
    Output: LLL-reduced basis (same shape)
    """
    B = B.astype(np.int64).copy()
    n, k = B.shape
    Bf = B.astype(np.float64)

    def gram_schmidt(Bf):
        n, k = Bf.shape
        U = np.zeros_like(Bf)
        mu = np.zeros((k, k))
        norm2 = np.zeros(k)
        for i in range(k):
            vi = Bf[:, i].copy()
            for j in range(i):
                mu[i, j] = np.dot(vi, U[:, j]) / norm2[j] if norm2[j] != 0 else 0.0
                vi -= mu[i, j] * U[:, j]
            U[:, i] = vi
            norm2[i] = np.dot(vi, vi)
        return U, mu, norm2

    def size_reduce(i, j, B, mu):
        q = int(round(mu[i, j]))
        if q != 0:
            B[:, i] -= q * B[:, j]
            mu[i, :j+1] -= q * mu[j, :j+1]
            mu[i, j] -= q

    U, mu, norm2 = gram_schmidt(Bf)
    i = 1
    while i < k:
        for j in range(i-1, -1, -1):
            size_reduce(i, j, B, mu)
        Bf = B.astype(np.float64)
        U, mu, norm2 = gram_schmidt(Bf)
        if norm2[i] >= (delta - mu[i, i-1]**2) * norm2[i-1]:
            i += 1
        else:
            B[:, [i, i-1]] = B[:, [i-1, i]]
            i = max(i-1, 1)
    return B

# =============================================================================
# --- Build “nice” transform identities via rational nullspace + LLL ----------
# =============================================================================

def nice_transform_identities(Ts, max_den=1, delta=0.75, top_k=None):
    """
    Compute transform identities α with exact arithmetic and reduce them to "nice" small-integer forms.
      1) Build A = Tflat (dim^2 x T) as Fractions (limit_denominator(max_den)).
      2) Rational nullspace basis via exact RREF.
      3) Convert to integer basis of the kernel lattice.
      4) LLL-reduce that integer basis to get short vectors.
      5) Normalize vectors to primitive integer form.
    Returns a list of integer α-vectors (numpy arrays), sorted by (support, L1, max|coeff|).
    """
    A_frac = as_fraction_matrix(Ts, max_den=max_den)
    basis_frac = rational_nullspace_fraction(A_frac)
    if not basis_frac:
        return []
    int_basis = integer_kernel_basis_from_rational(basis_frac)
    if not int_basis:
        return []
    B = np.stack(int_basis, axis=1)
    B_red = lll_reduce(B, delta=delta)
    cand = [B_red[:, i].copy() for i in range(B_red.shape[1])]
    cand += [b.copy() for b in int_basis]
    normed = []
    seen = set()
    for v in cand:
        g = 0
        for x in v: g = abs(x) if g==0 else gcd_int(g, x)
        if g > 1: v = v // g
        for x in v:
            if x != 0:
                if x < 0: v = -v
                break
        key = tuple(v.tolist())
        if key not in seen and np.any(v != 0):
            seen.add(key)
            normed.append(v)
    def score(v):
        supp = int(np.count_nonzero(v))
        l1 = int(np.sum(np.abs(v)))
        lmax = int(np.max(np.abs(v)))
        return (supp, l1, lmax)
    normed.sort(key=score)
    if top_k is not None:
        normed = normed[:top_k]
    return normed

# =============================================================================
# --- Vertex enumerators, PM & PTM LPs with OEP+OEM+α (kept) ------------------
# =============================================================================

def vertices_source_simplex_with_equalities(N_states, Aeq_extra=None, beq_extra=None):
    N = N_states
    Aeq = [np.ones(N)]; beq = [1.0]
    if Aeq_extra is not None:
        Aeq_extra = np.atleast_2d(np.asarray(Aeq_extra, float))
        beq_extra = np.zeros(Aeq_extra.shape[0]) if beq_extra is None else np.atleast_1d(np.asarray(beq_extra, float))
        Aeq += [row for row in Aeq_extra]; beq += beq_extra.tolist()
    Aeq = np.array(Aeq, float); beq = np.array(beq, float)
    rank = np.linalg.matrix_rank(Aeq)
    verts = []
    for supp in combinations(range(N), rank):
        As = Aeq[:, supp]
        if np.linalg.matrix_rank(As) < rank: continue
        try:
            xs = np.linalg.lstsq(As, beq, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        x = np.zeros(N); x[list(supp)] = xs
        if np.all(xs >= -1e-10) and np.allclose(Aeq @ x, beq, atol=1e-8):
            x = np.maximum(x, 0.0); s = x.sum()
            if s > 0: x /= s
            if np.allclose(Aeq @ x, beq, atol=1e-8) and not any(np.allclose(x, v, atol=1e-8) for v in verts):
                verts.append(x)
    return np.array(verts)

def vertices_measurement_blocks_with_equalities(groups, Aeq_extra=None, beq_extra=None):
    block_offsets, L = [], 0
    for g in groups: block_offsets.append(L); L += g
    det_vertices = []
    for picks in product(*[range(g) for g in groups]):
        v = np.zeros(L)
        for b,pick in enumerate(picks):
            v[block_offsets[b] + pick] = 1.0
        det_vertices.append(v)
    det_vertices = np.array(det_vertices)
    if Aeq_extra is None: return det_vertices
    Aeq_extra = np.asarray(Aeq_extra, float)
    beq_extra = np.zeros(Aeq_extra.shape[0]) if beq_extra is None else np.asarray(beq_extra, float)
    keep = [v for v in det_vertices if np.allclose(Aeq_extra @ v, beq_extra, atol=1e-8)]
    return np.array(keep)

def pm_feasibility_lp(Phi_vertices, p_data_pm, OEP_eq=None):
    Phi = np.asarray(Phi_vertices); Kprime, K_eff = Phi.shape
    p = np.asarray(p_data_pm); assert p.shape[0] == K_eff
    N_states = p.shape[1]
    C = 0 if OEP_eq is None else np.asarray(OEP_eq).shape[0]
    def idx(s,kp): return s*Kprime + kp
    n = N_states*Kprime
    Aeq, beq = [], []
    for s in range(N_states):
        row = np.zeros(n); row[slice(s*Kprime,(s+1)*Kprime)] = 1.0
        Aeq.append(row); beq.append(1.0)
    if C > 0:
        A = np.asarray(OEP_eq, float)
        for c in range(C):
            for kp in range(Kprime):
                row = np.zeros(n)
                for s in range(N_states): row[idx(s,kp)] = A[c, s]
                Aeq.append(row); beq.append(0.0)
    for s in range(N_states):
        for k in range(K_eff):
            row = np.zeros(n)
            for kp in range(Kprime): row[idx(s,kp)] = Phi[kp, k]
            Aeq.append(row); beq.append(p[k, s])
    Aeq = np.vstack(Aeq); beq = np.array(beq, float)
    bounds = [(0,None)]*n; c = np.zeros(n)
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    return {"feasible": res.success}

def build_ptm_lp_matrices(Phi, Psi, p_data, alphas_T=None, A_prep=None, A_meas=None):
    Phi = np.asarray(Phi); Kprime, K_eff = Phi.shape
    Psi = np.asarray(Psi); K, N_states = Psi.shape
    _, _, T = p_data.shape
    def idx(t,kp,kappa): return t*Kprime*K + kp*K + kappa
    n = T*Kprime*K
    Aeq, beq = [], []
    for t in range(T):
        row = np.zeros(n)
        for kp in range(Kprime):
            for kappa in range(K): row[idx(t,kp,kappa)] = 1.0
        Aeq.append(row); beq.append(1.0)
    if T >= 2:
        t0 = 0
        for kappa in range(K):
            for t in range(1, T):
                row = np.zeros(n)
                for kp in range(Kprime):
                    row[idx(t,kp,kappa)] += 1.0
                    row[idx(t0,kp,kappa)] -= 1.0
                Aeq.append(row); beq.append(0.0)
    if alphas_T is not None and np.size(alphas_T)>0:
        Aalpha = np.atleast_2d(np.asarray(alphas_T, float))
        for c in range(Aalpha.shape[0]):
            a = Aalpha[c]
            for kp in range(Kprime):
                for kappa in range(K):
                    row = np.zeros(n)
                    for t in range(T): row[idx(t,kp,kappa)] = a[t]
                    Aeq.append(row); beq.append(0.0)
    if A_prep is not None and np.size(A_prep)>0:
        A_prep = np.atleast_2d(np.asarray(A_prep, float))
        alpha_map = np.array([A_prep @ Psi[kappa, :].reshape(-1,1) for kappa in range(K)]).squeeze(-1)
        for t in range(T):
            for kp in range(Kprime):
                for c in range(A_prep.shape[0]):
                    row = np.zeros(n)
                    for kappa in range(K):
                        row[idx(t,kp,kappa)] = alpha_map[kappa, c]
                    Aeq.append(row); beq.append(0.0)
    if A_meas is not None and np.size(A_meas)>0:
        A_meas = np.atleast_2d(np.asarray(A_meas, float))
        beta_map = np.array([A_meas @ Phi[kp, :].reshape(-1,1) for kp in range(Kprime)]).squeeze(-1)
        for t in range(T):
            for kappa in range(K):
                for c in range(A_meas.shape[0]):
                    row = np.zeros(n)
                    for kp in range(Kprime):
                        row[idx(t,kp,kappa)] = beta_map[kp, c]
                    Aeq.append(row); beq.append(0.0)
    for t in range(T):
        for s in range(N_states):
            for k in range(K_eff):
                row = np.zeros(n)
                for kp in range(Kprime):
                    for kappa in range(K):
                        row[idx(t,kp,kappa)] = Phi[kp, k] * Psi[kappa, s]
                Aeq.append(row); beq.append(p_data[k, s, t])
    Aeq = np.vstack(Aeq) if Aeq else np.zeros((0, n))
    beq = np.array(beq, float)
    bounds = [(0, None)] * n
    return Aeq, beq, bounds

def ptm_feasibility_lp(Phi, Psi, p_data, alphas_T=None, A_prep=None, A_meas=None):
    Aeq, beq, bounds = build_ptm_lp_matrices(Phi, Psi, p_data, alphas_T=alphas_T, A_prep=A_prep, A_meas=A_meas)
    c = np.zeros(Aeq.shape[1])
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    return {"feasible": res.success}

def ptm_min_slack(Phi, Psi, p_data, alphas_T=None, A_prep=None, A_meas=None):
    Aeq, beq, bx = build_ptm_lp_matrices(Phi, Psi, p_data, alphas_T=alphas_T, A_prep=A_prep, A_meas=A_meas)
    m, n = Aeq.shape
    A_ub = np.vstack([np.hstack([ Aeq, -np.ones((m,1)) ]),
                      np.hstack([-Aeq, -np.ones((m,1)) ])])
    b_ub = np.hstack([beq, -beq])
    bounds = bx + [(0, None)]
    c = np.zeros(n+1); c[-1] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return float(res.fun) if res.success and res.x is not None else np.inf

# =============================================================================
# --- Scenarios (stabilizer states/effects; Baldi-4 and Clifford-24 transforms)
# =============================================================================

def make_stabilizer_states_effects():
    # 6 stabilizer states as columns ω = (1, x, y, z)
    bloch = {"Z+":(0,0,1),"Z-":(0,0,-1),"X+":(1,0,0),"X-":(-1,0,0),"Y+":(0,1,0),"Y-":(0,-1,0)}
    S = np.column_stack([[1.0,x,y,z] for (_, (x,y,z)) in bloch.items()])  # 4 x 6
    # effects: blocks (Z+,Z-)|(X+,X-)|(Y+,Y-)
    def e_pm(nx,ny,nz,sgn): return 0.5*np.array([1.0, sgn*nx, sgn*ny, sgn*nz], float)
    E = np.column_stack([
        e_pm(0,0,1,+1), e_pm(0,0,1,-1),
        e_pm(1,0,0,+1), e_pm(1,0,0,-1),
        e_pm(0,1,0,+1), e_pm(0,1,0,-1),
    ])  # 4 x 6
    groups = (2,2,2)
    return S, E, groups

def _stabilizer_state_names_in_order():
    # Must match the insertion order used in make_stabilizer_states_effects
    return ["Z+","Z-","X+","X-","Y+","Y-"]

def make_affine_from_R(R):
    T = np.eye(4)
    T[1:4,1:4] = R
    return T

def Rz(theta):
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)

def make_stabilizer_transforms_IZSSinv():
    Z = Rz(np.pi)
    Sp = Rz(+np.pi/2)
    Sm = Rz(-np.pi/2)
    Rs = [np.eye(3), Z, Sp, Sm]
    return [make_affine_from_R(R) for R in Rs]

def make_clifford24_transforms():
    """
    24 single-qubit Clifford rotations on Bloch sphere: signed permutation 3x3 matrices with det=+1.
    """
    Rs = []
    for p in permutations(range(3)):
        P = np.zeros((3,3))
        for i,j in enumerate(p): P[i,j] = 1.0
        for sx in (+1,-1):
            for sy in (+1,-1):
                for sz in (+1,-1):
                    Sg = np.diag([sx,sy,sz])
                    R = Sg @ P
                    if np.linalg.det(R) > 0.5:  # det +1
                        Rs.append(R.astype(float))
    uniq = []
    for R in Rs:
        if not any(np.allclose(R, Q) for Q in uniq):
            uniq.append(R)
    assert len(uniq) == 24, f"expected 24, got {len(uniq)}"
    return [make_affine_from_R(R) for R in uniq]

def compute_data(states, effects, transforms):
    _, Ns = states.shape
    _, Ke = effects.shape
    Tn = len(transforms)
    p = np.zeros((Ke, Ns, Tn))
    for t, T in enumerate(transforms):
        W = T @ states
        p[:, :, t] = effects.T @ W
    return p

# =============================================================================
# --- NEW: Prep+Transform == Prep detection & logging -------------------------
# =============================================================================

def find_identity_index(Ts, tol=1e-12):
    I = np.eye(4)
    for i, T in enumerate(Ts):
        if np.allclose(T, I, atol=tol):
            return i
    return -1

def detect_prep_transform_equalities(S, Ts, tol=1e-12):
    """
    Return list of triples (s, t, s2, state_residual_inf) such that T_t @ S[:, s] == S[:, s2] within tol
    """
    S = np.asarray(S); Ts = list(Ts)
    Ns = S.shape[1]
    out = []
    for s in range(Ns):
        v = S[:, s]
        for t, T in enumerate(Ts):
            w = T @ v
            # find matching state index s2
            diffs = np.max(np.abs(S - w.reshape(-1,1)), axis=0)
            hits = np.where(diffs <= tol)[0]
            for s2 in hits:
                res = float(np.max(np.abs(S[:, s2] - w)))
                out.append((s, t, s2, res))
    # deduplicate identical entries
    seen = set(); clean = []
    for s, t, s2, res in out:
        key = (s, t, s2)
        if key not in seen:
            seen.add(key); clean.append((s, t, s2, res))
    return clean

def verify_equalities_in_data(p_ptm, identity_idx, equalities, tol=1e-10):
    """
    For each (s,t,s2), compute max_k |p(k|s2, I) - p(k|s,t)|. If identity_idx < 0, returns np.nan for dev.
    Returns list of (s,t,s2, max_prob_dev).
    """
    Ke, Ns, Tn = p_ptm.shape
    out = []
    for (s, t, s2, _) in equalities:
        if 0 <= identity_idx < Tn:
            dev = float(np.max(np.abs(p_ptm[:, s2, identity_idx] - p_ptm[:, s, t])))
        else:
            dev = float('nan')
        out.append((s, t, s2, dev))
    return out

def transform_names_for_scenario(scenario, Ts):
    if scenario == "baldi4":
        # order: I, Z, S, S^-1 in our constructor
        return ["I","Z","S","S^-1"]
    # generic labels
    return [f"T{j}" for j in range(len(Ts))]

# =============================================================================
# --- CSV helpers --------------------------------------------------------------
# =============================================================================

def save_identities_csv(csv_path, records, append=False):
    fields = ["kind","scenario","T_count","index","coeffs","coeffs_rational",
              "residual_inf","tol","holds","note"]
    mode = "a" if (append and os.path.exists(csv_path)) else "w"
    with open(csv_path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == "w": w.writeheader()
        for rec in records:
            out = dict(rec)
            out["coeffs"] = "[" + ", ".join(str(x) for x in rec["coeffs"]) + "]"
            out["coeffs_rational"] = "[" + ", ".join(rec["coeffs_rational"]) + "]"
            w.writerow(out)

def save_equalities_csv(csv_path, records, append=False):
    """
    Save prep+transform==prep equalities.
    Each record: {
      "scenario","t_idx","t_name","s_idx","s_name","s2_idx","s2_name",
      "state_residual_inf","prob_residual_inf","note"
    }
    """
    fields = ["scenario","t_idx","t_name","s_idx","s_name","s2_idx","s2_name",
              "state_residual_inf","prob_residual_inf","note"]
    mode = "a" if (append and os.path.exists(csv_path)) else "w"
    with open(csv_path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == "w": w.writeheader()
        for rec in records:
            w.writerow(rec)

# =============================================================================
# --- Identity-first runner with rational/LLL add-on + paradox logging ---------
# =============================================================================

def run_identity_pipeline_with_nice(scenario="baldi4", tol=1e-12, csv_all="identities_found.csv",
                                    csv_nice="identities_nice.csv", csv_eq="prep_transform_equalities.csv",
                                    print_top=8, print_eq_top=20):
    S, E, groups = make_stabilizer_states_effects()
    if scenario == "baldi4":
        Ts = make_stabilizer_transforms_IZSSinv()
    elif scenario == "clifford24":
        Ts = make_clifford24_transforms()
    else:
        raise ValueError("scenario must be 'baldi4' or 'clifford24'")

    # Names for pretty prints
    state_names = _stabilizer_state_names_in_order()
    t_names = transform_names_for_scenario(scenario, Ts)

    # Vertices
    Phi_vertices = vertices_measurement_blocks_with_equalities(list(groups))
    Psi_vertices = vertices_source_simplex_with_equalities(S.shape[1])

    # Data
    p_ptm = compute_data(S, E, Ts)
    p_pm  = p_ptm.mean(axis=2)

    # Float SVD identities (for reference)
    recs_all = []

    A_T_float = generating_transform_identities_float(Ts)
    A_T_verified = []
    for j, row in enumerate(A_T_float):
        _, res = _alpha_residual_matrix(Ts, row)
        a = _normalize_vec(row)
        recs_all.append({"kind":"transform(float-SVD)","scenario":scenario,"T_count":len(Ts),
                         "index":j,"coeffs":[f"{x:.12g}" for x in a],"coeffs_rational":[f"{Fraction(x).limit_denominator()}" for x in a],
                         "residual_inf":f"{res:.3e}","tol":tol,"holds":(res<=tol),"note":"auto"})
        if res <= tol:
            A_T_verified.append(a)

    # Prep & Meas identities (float-based)
    A_P = generating_state_identities(S); A_Pv=[]
    for j,row in enumerate(A_P):
        res = float(np.max(np.abs(S @ row)))
        a = _normalize_vec(row)
        recs_all.append({"kind":"prep","scenario":scenario,"T_count":len(Ts),
                         "index":j,"coeffs":[f"{x:.12g}" for x in a],"coeffs_rational":[f"{Fraction(x).limit_denominator()}" for x in a],
                         "residual_inf":f"{res:.3e}","tol":tol,"holds":(res<=tol),"note":"auto"})
        if res <= tol: A_Pv.append(a)
    A_M = generating_effect_identities(E); A_Mv=[]
    for j,row in enumerate(A_M):
        res = float(np.max(np.abs(E @ row)))
        a = _normalize_vec(row)
        recs_all.append({"kind":"meas","scenario":scenario,"T_count":len(Ts),
                         "index":j,"coeffs":[f"{x:.12g}" for x in a],"coeffs_rational":[f"{Fraction(x).limit_denominator()}" for x in a],
                         "residual_inf":f"{res:.3e}","tol":tol,"holds":(res<=tol),"note":"auto"})
        if res <= tol: A_Mv.append(a)

    save_identities_csv(csv_all, recs_all, append=False)
    print(f"[{scenario}] (float) identities saved to '{csv_all}' — "
          f"{len(A_T_verified)} transform, {len(A_Pv)} prep, {len(A_Mv)} meas verified.")

    # === “nice” transform identities via rational nullspace + LLL ===
    nice_int_vecs = nice_transform_identities(Ts, max_den=1, delta=0.75, top_k=None)
    recs_nice = []
    for idx, v in enumerate(nice_int_vecs):
        alpha = v.astype(float)
        _, res = _alpha_residual_matrix(Ts, alpha)
        rat = [str(int(x)) for x in v.tolist()]
        recs_nice.append({"kind":"transform(nice-LLL)","scenario":scenario,"T_count":len(Ts),
                          "index":idx,"coeffs":[int(x) for x in v.tolist()],"coeffs_rational":rat,
                          "residual_inf":f"{res:.3e}","tol":tol,"holds":(res<=tol),"note":"LLL-reduced"})
    save_identities_csv(csv_nice, recs_nice, append=False)
    print(f"[{scenario}] (nice) {len(nice_int_vecs)} LLL-reduced integer transform identities saved to '{csv_nice}'.")

    top = nice_int_vecs[:print_top]
    if top:
        print(f"[{scenario}] a few 'nice' integer transform identities (sorted by support, L1, max):")
        for i, v in enumerate(top, 1):
            supp = int(np.count_nonzero(v))
            l1 = int(np.sum(np.abs(v)))
            lmax = int(np.max(np.abs(v)))
            nz = np.nonzero(v)[0].tolist()
            print(f"  {i:2d}) support={supp:2d}, L1={l1:2d}, max={lmax:2d} | α = {v.tolist()} | nonzeros @ {nz}")

    # === NEW: Detect and verify prep+transform==prep equalities ==============
    equalities = detect_prep_transform_equalities(S, Ts, tol=1e-12)
    t_id = find_identity_index(Ts, tol=1e-12)
    verified = verify_equalities_in_data(p_ptm, t_id, equalities, tol=1e-10)

    # Save equality records
    eq_records = []
    for (s, t, s2, state_res) , (_, _, _, prob_res) in zip(equalities, verified):
        eq_records.append({
            "scenario": scenario,
            "t_idx": t,
            "t_name": t_names[t] if 0 <= t < len(t_names) else f"T{t}",
            "s_idx": s,  "s_name": state_names[s] if 0 <= s < len(state_names) else f"s{s}",
            "s2_idx": s2, "s2_name": state_names[s2] if 0 <= s2 < len(state_names) else f"s{s2}",
            "state_residual_inf": f"{state_res:.3e}",
            "prob_residual_inf": f"{prob_res:.3e}" if np.isfinite(prob_res) else "NA(no I)",
            "note": "prep+transform==prep (within tol)"
        })
    save_equalities_csv(csv_eq, eq_records, append=False)
    print(f"[{scenario}] found {len(equalities)} prep+transform==prep equalities; saved to '{csv_eq}'.")
    if equalities:
        show = min(print_eq_top, len(equalities))
        print(f"[{scenario}] a few equalities (showing {show}):  (s) --{ 'T' }--> (s2) | t-name | state-res | prob-res")
        for i in range(show):
            s, t, s2, state_res = equalities[i]
            _, _, _, prob_res = verified[i]
            sN  = state_names[s]  if s  < len(state_names) else f"s{s}"
            s2N = state_names[s2] if s2 < len(state_names) else f"s{s2}"
            tN  = t_names[t] if t < len(t_names) else f"T{t}"
            prob_str = f"{prob_res:.3e}" if np.isfinite(prob_res) else "NA(no I)"
            print(f"  {i+1:2d}) {sN:>2} --{tN:>4}--> {s2N:>2} | {tN:<6} | {state_res:.3e} | {prob_str}")

    # PM vs PTM with ALL verified identities (float)
    OEP_eq = np.vstack(A_Pv) if A_Pv else None
    pm_ok  = pm_feasibility_lp(Phi_vertices, p_pm, OEP_eq=OEP_eq)["feasible"]

    alphas_T = np.vstack(A_T_verified) if A_T_verified else None
    A_prep   = np.vstack(A_Pv) if A_Pv else None
    A_meas   = np.vstack(A_Mv) if A_Mv else None

    ptm_ok = ptm_feasibility_lp(Phi_vertices, Psi_vertices, p_ptm, alphas_T=alphas_T, A_prep=A_prep, A_meas=A_meas)["feasible"]
    delta  = ptm_min_slack(Phi_vertices, Psi_vertices, p_ptm, alphas_T=alphas_T, A_prep=A_prep, A_meas=A_meas)

    print(f"[{scenario}] PM feasible? {pm_ok} | PTM feasible? {ptm_ok} | δ*={delta:.4e}")
    if pm_ok and (not ptm_ok):
        # Split-prep paradox banner with a couple of concrete equalities
        print("\n=== SPLIT-PREP PARADOX DETECTED ===")
        print("PM admits a noncontextual model, but PTM does not;")
        if t_id >= 0:
            print("below are explicit equalities with matching operational data: p(k|s2, I) = p(k|s, t)")
        else:
            print("identity transform not found — cannot verify data equality against I, but state equalities hold.")
        for i, (s, t, s2, _) in enumerate(equalities[:min(10, len(equalities))], 1):
            sN  = state_names[s]  if s  < len(state_names) else f"s{s}"
            s2N = state_names[s2] if s2 < len(state_names) else f"s{s2}"
            tN  = t_names[t] if t < len(t_names) else f"T{t}"
            pr  = [r for (_,_,_,r) in verified if _ == _]  # placeholder to avoid unused var warning
            # fetch matching prob dev
            dev = [d for (ss,tt,ss2,d) in verified if (ss,tt,ss2)==(s,t,s2)]
            dev_str = f"{dev[0]:.3e}" if dev and np.isfinite(dev[0]) else "NA"
            print(f"  {i:2d}) {sN} --{tN}--> {s2N}   | max_k |p(k|{s2N}, I) - p(k|{sN}, {tN})| = {dev_str}")
        print("====================================\n")

# =============================================================================
# --- Main --------------------------------------------------------------------
# =============================================================================

def main():
    # Classic 4-transform stabilizer set (Baldi): one clean identity expected
    run_identity_pipeline_with_nice(scenario="baldi4",  tol=1e-12,
                                    csv_all="identities_found.csv",
                                    csv_nice="identities_nice.csv",
                                    csv_eq="prep_transform_equalities.csv",
                                    print_top=8,
                                    print_eq_top=20)
    print("\n" + "-"*80 + "\n")
    # Full 24 single-qubit Cliffords: many transform identities; LLL yields nicer integer ones
    run_identity_pipeline_with_nice(scenario="clifford24", tol=1e-12,
                                    csv_all="identities_found4.csv",
                                    csv_nice="identities_nice4.csv",
                                    csv_eq="prep_transform_equalities4.csv",
                                    print_top=12,
                                    print_eq_top=30)

if __name__ == "__main__":
    main()
