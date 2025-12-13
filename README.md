<div align="center">

# ANDI: Adaptive Norm-Distribution Interface


</div>

<p align="center">
  <strong>Implementation of the paper "ANDI: Adaptive Norm-Distribution Interface"</strong>
</p>

<!--
[![ResearchGate](https://img.shields.io/badge/ResearchGate-View_Paper-00CCBB?style=flat&logo=ResearchGate&logoColor=white)](https://www.researchgate.net/publication/398447768_ANDI_Arithmetic_Normalization_Decorrelated_Inertia)
-->

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17923146.svg)](https://doi.org/10.5281/zenodo.17923146)



## Abstract

The optimization of deep neural networks is currently dominated by two paradigms: coordinate-
wise adaptive methods (e.g., AdamW), which ignore parameter correlations, and higher-order struc-
tural methods (e.g., K-FAC, Muon), which enforce geometric constraints but suffer from super-linear
computational complexity. We introduce the Adaptive Norm-Distribution Interface (ANDI),
a first-order optimizer that bridges this gap via structured preconditioning. ANDI applies an element-
wise equilibration transformation derived from the additive equilibration of row and column norms,
effectively approximating matrix balancing without iterative solvers or singular value decomposi-
tion. We prove that ANDI strictly maintains descent directions and provides an implicit trust region
bounded by the gradient energy. Empirically, ANDI matches the convergence of spectral methods on
ResNet-9 (CIFAR-10) while maintaining the O(N ) computational profile of AdamW. Furthermore,
on Transformer-based causal language modeling (NanoGPT), ANDI outperforms both diagonal and
spectral baselines, suggesting that additive norm-equilibration serves as a superior inductive bias for attention-based architectures.

---

### Experiment Manifest

To reproduce the results, run `ANDI.py` or for more convenience turn it into Jupyter Notebook `ANDI.ipynb`. 

---

## Installation

1. For quick experimentation with Jupyter Notebook

```bash
pip install torch torchvision matplotlib numpy requests
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
  title        = {ANDI: Adaptive Norm-Distribution Interface},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17923146},
  url          = {https://doi.org/10.5281/zenodo.17923146},
  note         = {Preprint}
}
```

<!--
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
-->

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





