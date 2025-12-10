<div align="center">

# ANDI: Arithmetic Normalization / Decorrelated Inertia Learning


</div>

<p align="center">
  <strong>Implementation of the paper "ANDI: Arithmetic Normalization / Decorrelated Inertia Learning"</strong>
</p>

[![ResearchGate](https://img.shields.io/badge/ResearchGate-View_Paper-00CCBB?style=flat&logo=ResearchGate&logoColor=white)](https://www.researchgate.net/publication/398447768_ANDI_Arithmetic_Normalization_Decorrelated_Inertia)

<!-- [![arXiv](https://img.shields.io/badge/arXiv-????.?????-b31b1b.svg)](????????????????????????) --> 

## Abstract

Modern neural network optimizers face a trilemma: they must balance Adaptivity (Adam), Structural Regularization (Shampoo, MUON), and Computational Efficiency (SGD). While second-order and structural methods offer superior generalization, they typically incur prohibitive $O(N^3)$ compute costs or massive memory overheads. We introduce **ANDI**, a first-order optimizer that approximates structural preconditioning in linear $O(N)$ time with $O(1)$ memory overhead. ANDI achieves this via a three-step mechanism: (1) Prime Topology Mixing to break local grid correlations; (2) One-Shot Arithmetic Equilibration to stabilize feature variance via broadcasted normalization; and (3) Hypotenuse Energy Regularization to automatically navigate saddle points. We present two variants: **ANDI-Direct**, which performs self-equilibration, and **ANDI-Lateral**, which employs a lateral inhibition mechanism to enforce spatial competition. Empirical results across MLPs, CNNs, and RNNs demonstrate that ANDI matches the convergence speed of adaptive methods and the generalization of structural methods, while maintaining the memory footprint of standard SGD.

---

Paper versions are attached as PDF files `andi.pdf`, `andi_v2.pdf`

### Experiment Manifest

To reproduce the results, run `ANDI.py` or for more convinience turn it into Jupyter Notebook `ANDI.ipynb`. You can do the same for alternative algorithm `ANDI_Lateral.py`.

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




