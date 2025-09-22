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
