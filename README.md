<div align="center">

# ANDI: Adaptive Norm-Distribution Interface


</div>

<p align="center">
  <strong>Implementation of the paper "ANDI: Adaptive Norm-Distribution Interface"</strong>
</p>

<!--
[![ResearchGate](https://img.shields.io/badge/ResearchGate-View_Paper-00CCBB?style=flat&logo=ResearchGate&logoColor=white)](https://www.researchgate.net/publication/398447768_ANDI_Arithmetic_Normalization_Decorrelated_Inertia)
-->

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18087892.svg)](https://doi.org/10.5281/zenodo.18087892)



## Abstract

The optimization of deep neural networks is currently dominated by two paradigms: coordinate-
wise adaptive methods (e.g., AdamW), which ignore parameter correlations, and higher-order struc-
tural methods (e.g., K-FAC, Muon), which enforce geometric constraints but suffer from super-linear
computational complexity. We introduce the Adaptive Norm-Distribution Interface (ANDI),
a first-order optimizer that bridges this gap via structured preconditioning. ANDI applies an element-
wise equilibration transformation derived from the additive equilibration of row and column norms,
effectively approximating matrix balancing without iterative solvers or singular value decomposi-
tion. We prove that ANDI strictly maintains descent directions and provides an implicit trust
region bounded by the gradient energy. Empirically, ANDI matches the convergence of spectral
methods on ResNet-9 (CIFAR-10) while maintaining the O(N ) computational profile of AdamW.
Furthermore, on Transformer-based causal language modeling (NanoGPT), ANDI outperforms both
diagonal and spectral baselines, suggesting that additive norm-equilibration serves as a superior in-
ductive bias for attention-based architectures. Finally, we demonstrate scalability to the 8-billion
parameter regime by fine-tuning Llama-3, where ANDI exhibits rapid convergence within the
constrained optimization subspaces of Low-Rank Adaptation (LoRA).

---

### Experiment Manifest

To reproduce the results, navigate to the `experiments/` folder. The notebooks correspond directly to the figures in the paper:

| Notebook | Objective | Paper Figure |
| :--- | :--- | :--- |
| **`ANDI.py`** | **small-scale.** Train Autoencoder ResNet GPT. | **Fig. 1** |
| **`ANDI_finetune.py`** | **large-scale.** Fine-tune LLAMA 3 8B 4bit with LoRA. | **Fig. 2** |

---

## Installation

1. For quick experimentation with Jupyter Notebook or Google Colab turn `.py` files into `.ipynb`. Colab is good choice if you do not have GPU.

<!-- ```bash
# 1. Install Unsloth (and PyTorch)
pip install unsloth

# 2. Install specific library versions
pip install "transformers==4.56.2" --no-deps
pip install "trl==0.22.2" --no-deps

# 3. Install Data Science utilities (most likely you do not need this in Google Colab) 
pip install torchvision numpy requests pandas matplotlib seaborn datasets # torch 
``` -->
2. Or clone the repository to your local machine and install the required dependencies using pip:

```bash
# cd ANDI
pip install -r requirements.txt
# Once the dependencies are installed, you can execute the script using:
python experiments/ANDI.py 
python experiments/ANDI_finetune.py # YOU NEED GPU TO RUN THIS ! ! ! 
```


## Citation

If you utilize this code or the concepts presented in **ANDI** for your research, please cite the following paper:

```bibtex
@misc{khasia2025andi_zenodo,
  author       = {Khasia, Vladimer},
  title        = {ANDI: Adaptive Norm-Distribution Interface},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18087892},
  url          = {https://doi.org/10.5281/zenodo.18087892},
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






