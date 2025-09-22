# split_prep_paradox
Code designed for finding and verifying split-prep paradoxes

Cases in which a direct preparation P2 is operationally indistinguishable from a preparation P1 followed by a transformation T (so that the final statistics coincide), yet adding the transformational layer renders a noncontextual model impossible even though the underlying PM statistics remain noncontextual.


## Dependencies
- Python ≥ 3.8  
- [NumPy](https://numpy.org/)  
- [SciPy](https://scipy.org/) (`scipy.optimize.linprog`)  
- Standard library: `fractions`, `csv`, `os`, `itertools`

Install dependencies via pip:
```bash
pip install numpy scipy

```
## Running the Script

Clone or place `split_prep_detector.py` in your project directory, then run:

```bash
python split_prep_detector.py

```
## What Happens When You Run It

The pipeline is executed for:

- **Baldi-4** (4 stabilizer transforms, includes identity).
- **Clifford-24** (all 24 single-qubit Cliffords).

Results are printed to the console:

- Number of float-based prep/meas/transform identities found.
- A few sample "nice" LLL-reduced integer transform identities.
- A few detected `prep + transform == prep` equalities.
- PM vs PTM feasibility checks and δ* slack value.
- An explicit **SPLIT-PREP PARADOX** warning if detected.

Several CSV files are created for later analysis:

- `identities_found.csv` – all float-based identities (Baldi-4).
- `identities_nice.csv` – rational/LLL-reduced transform identities (Baldi-4).
- `prep_transform_equalities.csv` – prep+transform=prep equalities (Baldi-4).
- `identities_found4.csv`, `identities_nice4.csv`, `prep_transform_equalities4.csv` – same outputs for Clifford-24.
