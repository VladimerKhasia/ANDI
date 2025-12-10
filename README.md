<div align="center">

# ANDI: Arithmetic Normalization / Decorrelated Inertia Learning


</div>

<p align="center">
  <strong>Implementation of the paper "ANDI: Arithmetic Normalization / Decorrelated Inertia Learning"</strong>
</p>

[![ResearchGate](https://img.shields.io/badge/ResearchGate-View_Paper-00CCBB?style=flat&logo=ResearchGate&logoColor=white)](https://www.researchgate.net/publication/398447768_ANDI_Arithmetic_Normalization_Decorrelated_Inertia)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17886314.svg)](https://doi.org/10.5281/zenodo.17886314)



## Abstract

Deep learning optimization typically necessitates a trade-off between memory efficiency (first-
order methods like SGD/AdamW) and convergence speed (second-order structural methods like
K-FAC/Shampoo). While structural optimizers offer superior generalization by capturing parameter
correlations, they incur prohibitive O(N 3) compute costs or significant memory overheads. We
introduce ANDI (Arithmetic Normalization / Decorrelated Inertia), a first-order optimizer that
approximates structural preconditioning in linear O(N ) time with O(1) additional memory overhead.
ANDI achieves this via three mechanisms: (1) Prime Topology Mixing, formally a fixed permutation
operator designed to disrupt local grid correlations; (2) One-Shot Arithmetic Equilibration, a linear-
time approximation of Sinkhorn-Knopp matrix balancing that stabilizes feature variance; and (3)
Hypotenuse Energy Regularization, a smooth, differentiable gradient scaling mechanism that enforces
a lower bound on update energy to navigate saddle points. We present two variants: ANDI-
Direct (Self-Equilibration) and ANDI-Lateral (Lateral Inhibition). Empirical results across MLPs,
CNNs, and Transformers (NanoGPT) demonstrate that ANDI matches the convergence trajectory
of adaptive methods while maintaining the memory footprint of standard SGD. 

---

Latest version of the paper is `ANDI_v3.pdf`

### Experiment Manifest

To reproduce the results, run `ANDI.py` or for more convinience turn it into Jupyter Notebook `ANDI.ipynb`. 

---

## Installation

1. For quick experimentation with Jupyter Notebook

```bash
pip install torch torchvision numpy matplotlib requests
```
2. Or clone the repository to your local machine and install the required dependencies using pip:

```bash
pip install -r requirements.txt
# Once the dependencies are installed, you can execute the script using:
python ANDI.py
```


## Citation

If you utilize this code or the concepts presented in **ANDI** for your research, please cite the following paper:

```bibtex
@misc{khasia2025andi_zenodo,
  author       = {Khasia, Vladimer},
  title        = {ANDI: Arithmetic Normalization / Decorrelated Inertia},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17886314},
  url          = {https://doi.org/10.5281/zenodo.17886314},
  note         = {Preprint}
}
```

```bibtex
@misc{khasia2025andi,
  author       = {Khasia, Vladimer},
  title        = {ANDI: Arithmetic Normalization / Decorrelated Inertia},
  year         = {2025},
  publisher    = {ResearchGate},
  doi          = {10.13140/RG.2.2.28381.47841},
  url          = {https://www.researchgate.net/publication/398447768_ANDI_Arithmetic_Normalization_Decorrelated_Inertia},
  note         = {Preprint}
}
```

<!--
```bibtex
@article{khasia2025andi,
  title={ANDI: Arithmetic Normalization / Decorrelated Inertia Learning},
  author={Khasia, Vladimer},
  journal={arXiv preprint	???????????????},
  year={2025}
}
```
-->




