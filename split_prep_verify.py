# 13_split_prep.py
# --------------------------------------------------------------------------------
# Focused test of the concrete split-prep equality:
#   X+ --Z--> X-  (i.e., T_Z @ ω_{X+} = ω_{X-})
# within the Baldi-4 transform scenario {I, Z, S, S^-1}.
#
# What this script does:
#   • Builds stabilizer states/effects and Baldi-4 transforms
#   • Verifies state-level equality T_Z ω_{X+} = ω_{X-}
#   • Verifies probability-table equality p(k|X-, I) = p(k|X+, Z) for all 6 effects
#   • Generates prep/effect/transform identities (linear, via SVD)
#   • Runs PM and PTM feasibility LPs (with OEP + OEM + α_T)
#   • Prints a clear “SPLIT-PREP PARADOX” confirmation if PM feasible & PTM infeasible
#
# Dependencies: numpy, scipy (optimize.linprog). No CSV output here—pure console test.
# --------------------------------------------------------------------------------

import numpy as np
from itertools import combinations, product, permutations
from scipy.optimize import linprog

# -------------------------
# Stabilizer scenario
# -------------------------

def make_stabilizer_states_effects():
    # 6 stabilizer states as columns ω = (1, x, y, z)
    bloch = {"Z+":(0,0,1),"Z-":(0,0,-1),"X+":(1,0,0),"X-":(-1,0,0),"Y+":(0,1,0),"Y-":(0,-1,0)}
    names = list(bloch.keys())  # ["Z+","Z-","X+","X-","Y+","Y-"]
    S = np.column_stack([[1.0,x,y,z] for (_, (x,y,z)) in bloch.items()])  # 4 x 6

    # effects: three binary measurements Z/X/Y as ± projectors
    def e_pm(nx,ny,nz,sgn): return 0.5*np.array([1.0, sgn*nx, sgn*ny, sgn*nz], float)
    E = np.column_stack([
        e_pm(0,0,1,+1), e_pm(0,0,1,-1),  # Z+ , Z-
        e_pm(1,0,0,+1), e_pm(1,0,0,-1),  # X+ , X-
        e_pm(0,1,0,+1), e_pm(0,1,0,-1),  # Y+ , Y-
    ])  # 4 x 6
    groups = (2,2,2)
    return S, E, groups, names

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
    Ts = [make_affine_from_R(R) for R in Rs]
    t_names = ["I","Z","S","S^-1"]
    return Ts, t_names

# -------------------------
# Data
# -------------------------

def compute_data(states, effects, transforms):
    _, Ns = states.shape
    _, Ke = effects.shape
    Tn = len(transforms)
    p = np.zeros((Ke, Ns, Tn))
    for t, T in enumerate(transforms):
        W = T @ states
        p[:, :, t] = effects.T @ W
    return p

# -------------------------
# Vertex enumerators
# -------------------------

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
        for b, pick in enumerate(picks):
            v[block_offsets[b] + pick] = 1.0
        det_vertices.append(v)
    det_vertices = np.array(det_vertices)
    if Aeq_extra is None: return det_vertices
    Aeq_extra = np.asarray(Aeq_extra, float)
    beq_extra = np.zeros(Aeq_extra.shape[0]) if beq_extra is None else np.asarray(beq_extra, float)
    keep = [v for v in det_vertices if np.allclose(Aeq_extra @ v, beq_extra, atol=1e-8)]
    return np.array(keep)

# -------------------------
# Identities (SVD)
# -------------------------

def generating_state_identities(states, tol=1e-10):
    S = np.asarray(states)
    U, s, Vt = np.linalg.svd(S, full_matrices=True)
    rank = (s > tol).sum()
    return Vt[rank:, :]

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

# -------------------------
# PM & PTM LPs (with OEP + OEM + α_T)
# -------------------------

def pm_feasibility_lp(Phi_vertices, p_data_pm, OEP_eq=None):
    Phi = np.asarray(Phi_vertices); Kprime, K_eff = Phi.shape
    p = np.asarray(p_data_pm); assert p.shape[0] == K_eff
    N_states = p.shape[1]
    C = 0 if OEP_eq is None else np.asarray(OEP_eq).shape[0]
    def idx(s,kp): return s*Kprime + kp
    n = N_states*Kprime
    Aeq, beq = [], []

    # normalization per s
    for s in range(N_states):
        row = np.zeros(n); row[slice(s*Kprime,(s+1)*Kprime)] = 1.0
        Aeq.append(row); beq.append(1.0)

    # OEP per kp
    if C > 0:
        A = np.asarray(OEP_eq, float)
        for c in range(C):
            for kp in range(Kprime):
                row = np.zeros(n)
                for s in range(N_states): row[idx(s,kp)] = A[c, s]
                Aeq.append(row); beq.append(0.0)

    # data consistency
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
    """
    Variables: μ_t(kp,κ) ≥ 0  (mixing over measurement/source vertices at each t).
    Constraints:
      • (F1b) normalization per t
      • (F1c) independence across t per κ
      • (F1d) transform identities α_T over t
      • OEP inside PTM using A_prep over κ (for each t,kp)
      • OEM inside PTM using A_meas over kp (for each t,κ)
      • (F1e) data consistency
    """
    Phi = np.asarray(Phi); Kprime, K_eff = Phi.shape
    Psi = np.asarray(Psi); K, N_states = Psi.shape
    _, _, T = p_data.shape

    def idx(t,kp,kappa): return t*Kprime*K + kp*K + kappa
    n = T*Kprime*K
    Aeq, beq = [], []

    # (F1b) normalization per t
    for t in range(T):
        row = np.zeros(n)
        for kp in range(Kprime):
            for kappa in range(K): row[idx(t,kp,kappa)] = 1.0
        Aeq.append(row); beq.append(1.0)

    # (F1c) independence across t per κ
    if T >= 2:
        t0 = 0
        for kappa in range(K):
            for t in range(1, T):
                row = np.zeros(n)
                for kp in range(Kprime):
                    row[idx(t,kp,kappa)] += 1.0
                    row[idx(t0,kp,kappa)] -= 1.0
                Aeq.append(row); beq.append(0.0)

    # (F1d) transform identities
    if alphas_T is not None and np.size(alphas_T) > 0:
        Aalpha = np.atleast_2d(np.asarray(alphas_T, float))
        for c in range(Aalpha.shape[0]):
            a = Aalpha[c]
            for kp in range(Kprime):
                for kappa in range(K):
                    row = np.zeros(n)
                    for t in range(T): row[idx(t,kp,kappa)] = a[t]
                    Aeq.append(row); beq.append(0.0)

    # OEP inside PTM (prep identities)
    if A_prep is not None and np.size(A_prep) > 0:
        A_prep = np.atleast_2d(np.asarray(A_prep, float))
        alpha_map = np.array([A_prep @ Psi[kappa, :].reshape(-1,1) for kappa in range(K)]).squeeze(-1)  # K x C_P
        for t in range(T):
            for kp in range(Kprime):
                for c in range(A_prep.shape[0]):
                    row = np.zeros(n)
                    for kappa in range(K):
                        row[idx(t,kp,kappa)] = alpha_map[kappa, c]
                    Aeq.append(row); beq.append(0.0)

    # OEM inside PTM (measurement identities)
    if A_meas is not None and np.size(A_meas) > 0:
        A_meas = np.atleast_2d(np.asarray(A_meas, float))
        beta_map = np.array([A_meas @ Phi[kp, :].reshape(-1,1) for kp in range(Kprime)]).squeeze(-1)  # Kprime x C_M
        for t in range(T):
            for kappa in range(K):
                for c in range(A_meas.shape[0]):
                    row = np.zeros(n)
                    for kp in range(Kprime):
                        row[idx(t,kp,kappa)] = beta_map[kp, c]
                    Aeq.append(row); beq.append(0.0)

    # (F1e) data consistency
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

# -------------------------
# Split-prep verification helpers
# -------------------------

def find_identity_index(Ts, tol=1e-12):
    I = np.eye(4)
    for i, T in enumerate(Ts):
        if np.allclose(T, I, atol=tol):
            return i
    return -1

def verify_state_equality(S, s_idx, Ts, t_idx, s2_idx, tol=1e-12):
    w = Ts[t_idx] @ S[:, s_idx]
    return float(np.max(np.abs(w - S[:, s2_idx])))

def verify_probability_equality(p_ptm, s_idx, t_idx, s2_idx, tI_idx, tol=1e-12):
    if tI_idx < 0:
        return np.nan
    dev = float(np.max(np.abs(p_ptm[:, s2_idx, tI_idx] - p_ptm[:, s_idx, t_idx])))
    return dev

# -------------------------
# Main test
# -------------------------

def main():
    tol = 1e-12

    # Build scenario and data
    S, E, groups, state_names = make_stabilizer_states_effects()
    Ts, t_names = make_stabilizer_transforms_IZSSinv()
    p_ptm = compute_data(S, E, Ts)
    p_pm  = p_ptm.mean(axis=2)

    # Vertices for LPs
    Phi_vertices = vertices_measurement_blocks_with_equalities(list(groups))
    Psi_vertices = vertices_source_simplex_with_equalities(S.shape[1])

    # Generate identities (float SVD) and keep only those with tiny residuals
    A_T = generating_transform_identities_float(Ts)
    A_Tv = []
    for row in A_T:
        R = sum(float(row[t]) * Ts[t] for t in range(len(Ts)))
        res = float(np.max(np.abs(R)))
        if res <= tol:
            a = _normalize_vec(row)
            A_Tv.append(a)
    alphas_T = np.vstack(A_Tv) if A_Tv else None

    A_P = generating_state_identities(S)
    A_Pv = []
    for row in A_P:
        if float(np.max(np.abs(S @ row))) <= tol:
            A_Pv.append(_normalize_vec(row))
    A_M = generating_effect_identities(E)
    A_Mv = []
    for row in A_M:
        if float(np.max(np.abs(E @ row))) <= tol:
            A_Mv.append(_normalize_vec(row))

    OEP_eq = np.vstack(A_Pv) if A_Pv else None
    A_prep = np.vstack(A_Pv) if A_Pv else None
    A_meas = np.vstack(A_Mv) if A_Mv else None

    # --- Our concrete split-prep equality: X+ --Z--> X- ---
    # indices under our ordering: ["Z+","Z-","X+","X-","Y+","Y-"]
    s_idx  = 2  # X+
    s2_idx = 3  # X-
    t_idx  = 1  # Z
    tI_idx = find_identity_index(Ts, tol=1e-12)

    state_dev = verify_state_equality(S, s_idx, Ts, t_idx, s2_idx, tol=1e-12)
    prob_dev  = verify_probability_equality(p_ptm, s_idx, t_idx, s2_idx, tI_idx, tol=1e-12)

    print("\n--- Verifying concrete split-prep equality ---")
    print(f"Target: {state_names[s_idx]} --{t_names[t_idx]}--> {state_names[s2_idx]}")
    print(f"State residual ||T_t ω_s - ω_{state_names[s2_idx]}||_∞ = {state_dev:.3e}")
    if np.isfinite(prob_dev):
        print(f"Max_k |p(k|{state_names[s2_idx]}, I) - p(k|{state_names[s_idx]}, {t_names[t_idx]})| = {prob_dev:.3e}")
    else:
        print("Identity transform not found; probability equality vs I cannot be checked.")

    # --- Run PM and PTM feasibility LPs ---
    print("\n--- Running PM/PTM LPs (with OEP + OEM + α_T) ---")
    pm_ok  = pm_feasibility_lp(Phi_vertices, p_pm, OEP_eq=OEP_eq)["feasible"]
    ptm_ok = ptm_feasibility_lp(Phi_vertices, Psi_vertices, p_ptm,
                                alphas_T=alphas_T, A_prep=A_prep, A_meas=A_meas)["feasible"]
    delta  = ptm_min_slack(Phi_vertices, Psi_vertices, p_ptm,
                           alphas_T=alphas_T, A_prep=A_prep, A_meas=A_meas)

    print(f"PM feasible? {pm_ok} | PTM feasible? {ptm_ok} | δ*={delta:.4e}")

    # --- Report paradox if present ---
    if pm_ok and (not ptm_ok):
        print("\n=== SPLIT-PREP PARADOX CONFIRMED ===")
        if np.isfinite(prob_dev):
            print(f"{state_names[s_idx]} --{t_names[t_idx]}--> {state_names[s2_idx]}  with p-table match vs identity:")
            print(f"  max_k |p(k|{state_names[s2_idx]}, I) - p(k|{state_names[s_idx]}, {t_names[t_idx]})| = {prob_dev:.3e}")
        else:
            print(f"{state_names[s_idx]} --{t_names[t_idx]}--> {state_names[s2_idx]} (state-level equality holds; identity not in Ts)")
        print("PM admits a noncontextual model, but PTM does not once transforms+identities are enforced.")
        print("======================================\n")
    else:
        print("\n(No flip detected in this run.)\n")

if __name__ == "__main__":
    main()
