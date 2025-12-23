#@title complete experiment
"""
================================================================================
EXPERIMENTAL SETUP & PAPER REFERENCES
Based on: "Benchmarking Optimizers for Large Language Model Pretraining" (Semenov et al., 2025)
================================================================================

1. OPTIMIZER SELECTION: D-MUON (Dispersed Muon) vs. AdamW
   - We utilize D-Muon instead of the original Muon.
   - Reference: Section A.3, Figures 4 & 8.
   - Findings: The paper demonstrates that original Muon suffers in long training runs
     because it lacks weight decay for 2D matrices. D-Muon applies decoupled weight decay
     to all parameters and uses an AdamW backend for 1D vectors, yielding consistent
     state-of-the-art performance (Takeaway 3 & 10).

2. REPORTING STANDARDS (Added based on paper recommendations)
   A. Validation Loss (The True Metric):
      - Reference: Figure 1 & Takeaway 12.
      - Why: Training loss can be misleading due to overfitting. The paper emphasizes
        reporting Final Validation Loss to determine the true ranking of optimizers.

   B. Wall-Clock Time:
      - Reference: Figure 18 & Appendix D.3.
      - Finding: Step count is insufficient. Optimizers like SOAP have high overhead
        per step. We now report Wall-Clock time per 100 iterations or total run to
        measure computational throughput.

   C. Gradient Norm Evolution:
      - Reference: Figures 12, 27, 28, 29.
      - Finding: The shape of the gradient norm is a proxy for training stability.
        Sign-based methods (Lion) and SF-AdamW exhibit distinct "bump" shapes.
        AdamW usually shows a gradual increase or flat trajectory. Spikes indicate instability.

   D. Ablation on Batch Size (128 vs 256):
      - Reference: Figures 5 & 6 (Small vs Large Batches).
      - Finding: Some optimizers (Signum, Lion, Muon) benefit significantly from
        larger batch sizes (Takeaway 1 & 2), while others (AdamW) are more consistent
        but may be outperformed at scale.

3. HYPERPARAMETER CHOICES & TUNING
   A. Newton-Schulz Iterations (D-Muon):
      - Setting: 5 steps.
      - Reference: Figure 37.
      - Finding: Increasing iterations beyond 5 yields no performance gain but
        increases wall-clock time.

   B. Momentum (D-Muon):
      - Setting: 0.95 (Nesterov).
      - Reference: Tables 40 & 41 (Hyperparameter Tuning for 720M models).
      - Finding: 0.95 consistently outperformed 0.9 or 0.99 for Muon-based methods.

   C. Schedulers:
      - AdamW: Cosine Scheduler.
      - D-Muon: Warmup-Stable-Decay (WSD).
      - Reference: Figure 11 (a vs c) and Takeaway 6.
      - Finding: Muon exhibits a unique preference for WSD (Linear Decay also works but
        WSD is more robust), whereas AdamW performs best with standard Cosine.

4. PROXY ADAPTATION & SCALING LAWS
   - Learning Rates: Robust search performed [AdamW: 1e-3~2e-4, D-Muon: 0.05~0.01].
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import requests
import time
import pandas as pd
from collections import defaultdict

# ==========================================
# 0. CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
TRAIN_STEPS = 2500 # Default steps

print(f"Running Flexible Benchmark on: {DEVICE}")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# ==========================================
# 1. OPTIMIZERS (ANDI & DMuon & Helpers)
# ==========================================

class ANDI(optim.Optimizer):
    """
    ANDI (Adaptive Normalized Direction w/ Identity)
    """
    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.9, nesterov=True, min_dim=17):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, min_dim=min_dim)
        super().__init__(params, defaults)
        self.epsilon_scalar = torch.tensor(1.0, device="cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            mom = group['momentum']
            nest = group['nesterov']
            min_dim = group['min_dim']

            for p in group['params']:
                if p.grad is None: continue
                g = p.grad

                # 1. Weight Decay
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                # 2. Check Dimensions
                is_matrix = g.ndim > 1
                if is_matrix:
                    original_shape = g.shape
                    # Flatten to 2D
                    g_mat = g.view(g.shape[0], -1)
                    rows, cols = g_mat.shape
                    is_large_enough = (rows >= min_dim and cols >= min_dim)
                else:
                    is_large_enough = False

                if is_large_enough:
                    # === ANDI LOGIC ===
                    # Add epsilon in-place for speed
                    r_norm = g_mat.norm(dim=1, keepdim=True).add_(1e-8)
                    c_norm = g_mat.norm(dim=0, keepdim=True).add_(1e-8)

                    g_white = g_mat / (r_norm + c_norm)

                    # Scale (Preserve Magnitude)
                    g_norm = g.norm().add_(1e-8)
                    target = torch.hypot(g_norm, self.epsilon_scalar)

                    g_white_norm = g_white.norm().add_(1e-8)
                    scale = target / g_white_norm

                    g_final = g_white.mul_(scale).view(original_shape)
                else:
                    # Fallback for vectors/small dims
                    g_norm = g.norm().add_(1e-8)
                    target = torch.hypot(g_norm, self.epsilon_scalar)
                    g_final = g.mul(target / g_norm)

                # 3. Momentum
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = g_final.clone()

                buf = state['momentum_buffer']
                buf.mul_(mom).add_(g_final)

                update = g_final.add(buf, alpha=mom) if nest else buf
                p.data.add_(update, alpha=-lr)

        return loss

# class ANDI(optim.Optimizer):
#     def __init__(self, params, lr=0.02, momentum=0.9, nesterov=True, weight_decay=0.0):
#         defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
#         super().__init__(params, defaults)
#         self.epsilon_scalar = torch.tensor(1.0, device=DEVICE)

#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad(): loss = closure()

#         for group in self.param_groups:
#             lr = group['lr']
#             mom = group['momentum']
#             nest = group['nesterov']
#             wd = group['weight_decay']

#             for p in group['params']:
#                 if p.grad is None: continue
#                 g = p.grad
#                 if wd != 0: p.data.mul_(1 - lr * wd)

#                 if g.ndim > 1:
#                     original_shape = g.shape
#                     g_mat = g.view(g.shape[0], -1)
#                     rows, cols = g_mat.shape
#                     if rows > 16 and cols > 16:
#                         # Improved stability epsilon
#                         r_norm = g_mat.norm(dim=1, keepdim=True).add_(1e-8)
#                         c_norm = g_mat.norm(dim=0, keepdim=True).add_(1e-8)
#                         g_white = g_mat / (r_norm + c_norm)
#                         g_norm = g.norm().add_(1e-8)
#                         target = torch.hypot(g_norm, self.epsilon_scalar)
#                         g_white_norm = g_white.norm().add_(1e-8)
#                         scale = target / g_white_norm
#                         g_final = g_white.mul_(scale).view(original_shape)
#                     else:
#                         g_norm = g.norm().add_(1e-8)
#                         target = torch.hypot(g_norm, self.epsilon_scalar)
#                         g_final = g.mul(target / g_norm)
#                 else:
#                     g_norm = g.norm().add_(1e-8)
#                     target = torch.hypot(g_norm, self.epsilon_scalar)
#                     g_final = g.mul(target / g_norm)

#                 state = self.state[p]
#                 if 'momentum_buffer' not in state: state['momentum_buffer'] = g_final.clone()
#                 buf = state['momentum_buffer']
#                 buf.mul_(mom).add_(g_final)
#                 update = g_final.add(buf, alpha=mom) if nest else buf
#                 p.data.add_(update, alpha=-lr)
#         return loss


def newton_schulz_5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() if G.is_cuda and G.dtype == torch.float32 else G
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1): X = X.T; transposed = True
    else: transposed = False
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed: X = X.T
    return X.to(G.dtype)

# class DMuon(optim.Optimizer):
#     def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01, ns_steps=5, adamw_betas=(0.8, 0.999)):
#         defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, adamw_betas=adamw_betas)
#         super().__init__(params, defaults)
#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad(): loss = closure()
#         for group in self.param_groups:
#             lr, wd, mom = group['lr'], group['weight_decay'], group['momentum']
#             ns_steps, (beta1, beta2) = group['ns_steps'], group['adamw_betas']
#             for p in group['params']:
#                 if p.grad is None: continue
#                 g = p.grad
#                 if wd != 0: p.data.mul_(1 - lr * wd)
#                 if p.ndim == 2 and p.size(0) > 32 and p.size(1) > 32:
#                     state = self.state[p]
#                     if 'momentum_buffer' not in state: state['momentum_buffer'] = torch.zeros_like(p)
#                     buf = state['momentum_buffer']
#                     buf.mul_(mom).add_(g)
#                     g_orth = newton_schulz_5(buf, steps=ns_steps)
#                     scale_factor = max(1, p.size(0) / p.size(1)) ** 0.5
#                     p.data.add_(g_orth, alpha=-lr * scale_factor)
#                 else:
#                     state = self.state[p]
#                     if 'step' not in state:
#                         state['step'] = 0; state['exp_avg'] = torch.zeros_like(p); state['exp_avg_sq'] = torch.zeros_like(p)
#                     state['step'] += 1
#                     exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                     exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
#                     exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
#                     denom = exp_avg_sq.sqrt().add_(1e-8)
#                     bias_cor1 = 1 - beta1 ** state['step']
#                     bias_cor2 = 1 - beta2 ** state['step']
#                     p.data.addcdiv_(exp_avg, denom, value=-(lr * math.sqrt(bias_cor2) / bias_cor1))
#         return loss

class DMuon(optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01, ns_steps=5, adamw_betas=(0.8, 0.999)):
        """
        D-Muon Optimizer (Dispersed Muon).

        Reference: "Benchmarking Optimizers for Large Language Model Pretraining" (Semenov et al., 2025)

        Mechanism:
        1. Decoupled Weight Decay is applied to ALL parameters.
        2. Muon (Newton-Schulz) is used for tensors >= 2D (Linear & Conv2d weights).
        3. AdamW is used as a backend for 1D tensors (Biases, LayerNorms).
        """
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps, adamw_betas=adamw_betas)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            mom = group['momentum']
            ns_steps = group['ns_steps']
            beta1, beta2 = group['adamw_betas']

            for p in group['params']:
                if p.grad is None: continue
                g = p.grad

                # 1. Decoupled Weight Decay (Universal)
                # D-Muon applies this to everything before the optimizer split.
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                # 2. Condition Check: Is this a Matrix/Tensor suitable for Muon?
                # - Must be at least 2D.
                # - Must be large enough (>32 in primary dims) to justify Newton-Schulz cost.
                use_muon = False

                # Check 1: Large Linear Layers [Out, In]
                if p.ndim == 2 and p.size(0) > 32 and p.size(1) > 32:
                    use_muon = True

                # Check 2: Large Conv2d Layers [Out, In, kH, kW]
                # We check p.size(0) (Out Channels) > 32 to ensure we don't process tiny 1x1 convs excessively.
                elif p.ndim == 4 and p.size(0) > 32:
                    use_muon = True

                if use_muon:
                    # ====================================================
                    # MUON BRANCH (Spectral Update)
                    # ====================================================
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p)

                    buf = state['momentum_buffer']
                    # Apply Momentum
                    buf.mul_(mom).add_(g)

                    # Matricize: Flatten to [Out_Channels, Input_Features]
                    # If p is 4D (Conv2d), view as [N, C*H*W]
                    if p.ndim == 4:
                        buf_flat = buf.view(p.size(0), -1)
                    else:
                        buf_flat = buf

                    # Newton-Schulz Iteration (Orthogonalization)
                    # Note: newton_schulz_5 handles internal bfloat16 casting/normalization
                    g_orth_flat = newton_schulz_5(buf_flat, steps=ns_steps)

                    # Scaling Factor
                    # Muon scales updates based on the ratio of output/input dimensions
                    # to keep variance consistent across layer shapes.
                    rows = buf_flat.size(0)
                    cols = buf_flat.size(1)
                    scale_factor = max(1, rows / cols) ** 0.5

                    # Reshape back to original 4D (if necessary)
                    if p.ndim == 4:
                        g_orth = g_orth_flat.view_as(p)
                    else:
                        g_orth = g_orth_flat

                    # Final Update
                    p.data.add_(g_orth, alpha=-lr * scale_factor)

                else:
                    # ====================================================
                    # ADAMW BRANCH (Backend for vectors/small tensors)
                    # ====================================================
                    state = self.state[p]
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    # Standard AdamW update logic
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    denom = exp_avg_sq.sqrt().add_(1e-8)

                    bias_cor1 = 1 - beta1 ** state['step']
                    bias_cor2 = 1 - beta2 ** state['step']

                    # AdamW step size
                    step_size = lr * math.sqrt(bias_cor2) / bias_cor1

                    # Note: We use addcdiv_. Weight decay was already applied at the top.
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

# ==========================================
# 2. SCHEDULERS
# ==========================================

def get_wsd_schedule(optimizer, warmup_steps, training_steps, cooldown_fraction=0.2):
    cooldown_steps = int(training_steps * cooldown_fraction)
    stable_steps = training_steps - warmup_steps - cooldown_steps
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        if current_step < (warmup_steps + stable_steps): return 1.0
        return max(0.0, 1.0 - (current_step - (warmup_steps + stable_steps)) / float(max(1, cooldown_steps)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_cosine_schedule(optimizer, warmup_steps, training_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, training_steps - warmup_steps))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==========================================
# 3. MODELS
# ==========================================

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, block_size=64, n_embd=128, n_head=4, n_layer=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[nn.Sequential(
            nn.LayerNorm(n_embd), nn.MultiheadAttention(n_embd, n_head, batch_first=True),
            nn.LayerNorm(n_embd), nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd))
        ) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        mask = torch.triu(torch.ones(T, T, device=idx.device)*float('-inf'), diagonal=1)
        for b in self.blocks:
            x = x + b[1](b[0](x), b[0](x), b[0](x), attn_mask=mask, is_causal=True)[0]
            x = x + b[3](b[2](x))
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        def cb(in_c, out_c, pool=False):
            l = [nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
            if pool: l.append(nn.MaxPool2d(2))
            return nn.Sequential(*l)
        self.net = nn.Sequential(cb(3,64), cb(64,128,True), cb(128,128), cb(128,128),
                                 cb(128,256,True), cb(256,512,True), cb(512,512), cb(512,512),
                                 nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_classes))
    def forward(self, x): return self.net(x)

class DeepAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 256), nn.Tanh(), nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 16), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 256), nn.Tanh(), nn.Linear(256, 784), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x.view(x.size(0), -1)))

# ==========================================
# 4. ENGINE & UTILS
# ==========================================

def configure_optimizer_groups(model, weight_decay):
    decay, no_decay = set(), set()
    whitelist = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention)
    blacklist = (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if pn.endswith('bias'): no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist): decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist): no_decay.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter = decay & no_decay
    union = decay | no_decay
    for pn in param_dict.keys():
        if pn not in union: no_decay.add(pn)
    return [{"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}]

def get_loaders(task, batch_size):
    if task == "AUTOENCODER":
        tf = transforms.Compose([transforms.ToTensor()])
        train_ds = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=tf)
        val_ds = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=tf)
        train_ds = torch.utils.data.Subset(train_ds, range(20000))
        val_ds = torch.utils.data.Subset(val_ds, range(2000))
        return (torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False))
    elif task == "RESNET":
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        tf_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(*stats)])
        tf_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        train_ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=tf_train)
        val_ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=tf_val)
        return (torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False))
    return None, None

def prepare_shakespeare():
    path = os.path.join(DATA_DIR, 'input.txt')
    if not os.path.exists(path):
        try: r = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'); open(path, 'w').write(r.text)
        except: open(path, 'w').write("Dummy " * 10000)
    with open(path, 'r', encoding='utf-8') as f: text = f.read()
    chars = sorted(list(set(text))); stoi = { ch:i for i,ch in enumerate(chars) }
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)

@torch.no_grad()
def evaluate(model, task_name, val_loader, gpt_val_data, batch_size, block_size=64):
    model.eval()
    losses = []
    # IMPROVEMENT: Increased eval steps from 10 to 50 for more robust "Best LR" selection
    EVAL_STEPS = 50

    if task_name == "GPT":
        for _ in range(EVAL_STEPS):
            ix = torch.randint(len(gpt_val_data) - block_size, (batch_size,))
            x = torch.stack([gpt_val_data[i:i+block_size] for i in ix]).to(DEVICE)
            y = torch.stack([gpt_val_data[i+1:i+block_size+1] for i in ix]).to(DEVICE)
            _, loss = model(x, y)
            losses.append(loss.item())
    else:
        iter_loader = iter(val_loader)
        for _ in range(EVAL_STEPS):
            try: batch = next(iter_loader)
            except:
                iter_loader = iter(val_loader)
                batch = next(iter_loader)

            if task_name == "AUTOENCODER":
                img = batch[0].to(DEVICE)
                loss = F.mse_loss(model(img), img.view(img.size(0), -1))
            elif task_name == "RESNET":
                img, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)
                loss = F.cross_entropy(model(img), targets)
            losses.append(loss.item())
    model.train()
    return np.mean(losses)

def train_engine(task_name, model_fn, optimizer_cls, opt_kwargs, seeds, batch_size=128, steps=TRAIN_STEPS):
    agg_train, agg_val, agg_grad, agg_time = [], [], [], []
    gpt_train, gpt_val, gpt_vocab = (None, None, 0)
    train_loader, val_loader = (None, None)
    block_size = 64

    if task_name == "GPT": gpt_train, gpt_val, gpt_vocab = prepare_shakespeare()
    else: train_loader, val_loader = get_loaders(task_name, batch_size)

    for seed in seeds:
        seed_everything(seed)
        model = model_fn(gpt_vocab, block_size=block_size).to(DEVICE) if task_name=="GPT" else model_fn().to(DEVICE)

        wd_val = opt_kwargs.get("weight_decay", 0.0)
        init_kwargs = {k: v for k, v in opt_kwargs.items() if k != "weight_decay"}
        param_groups = configure_optimizer_groups(model, wd_val)
        opt = optimizer_cls(param_groups, **init_kwargs)

        warmup = int(steps * 0.1)
        sched = get_wsd_schedule(opt, warmup, steps) if optimizer_cls in [ANDI, DMuon] else get_cosine_schedule(opt, warmup, steps)
        ## if we want to use cosine for ANDI comment out above line and use this: 
        # sched = get_wsd_schedule(opt, warmup, steps) if optimizer_cls == DMuon else get_cosine_schedule(opt, warmup, steps)
        iter_loader = iter(train_loader) if train_loader else None
        model.train()
        r_t, r_v, r_g = [], [], []
        start = time.time()

        for step in range(steps):
            if task_name == "GPT":
                ix = torch.randint(len(gpt_train) - block_size, (batch_size,))
                x = torch.stack([gpt_train[i:i+block_size] for i in ix]).to(DEVICE)
                y = torch.stack([gpt_train[i+1:i+block_size+1] for i in ix]).to(DEVICE)
                _, loss = model(x, y)
            elif task_name == "AUTOENCODER":
                try: batch = next(iter_loader)
                except: iter_loader = iter(train_loader); batch = next(iter_loader)
                img = batch[0].to(DEVICE)
                loss = F.mse_loss(model(img), img.view(img.size(0), -1))
            else:
                try: batch = next(iter_loader)
                except: iter_loader = iter(train_loader); batch = next(iter_loader)
                img, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)
                loss = F.cross_entropy(model(img), targets)

            opt.zero_grad(); loss.backward();
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            opt.step(); sched.step()

            if step % 50 == 0:
                r_t.append(loss.item()); r_g.append(norm.item())
                r_v.append(evaluate(model, task_name, val_loader, gpt_val, batch_size, block_size))

        agg_time.append(time.time() - start)
        agg_train.append(r_t); agg_val.append(r_v); agg_grad.append(r_g)

    return (np.mean(agg_train, axis=0), np.mean(agg_val, axis=0), np.mean(agg_grad, axis=0), np.mean(agg_time))

# ==========================================
# 5. FLEXIBLE PLOTTING (IMPROVED)
# ==========================================

def plot_results(results):
    """
    Improved plotting with legends for all columns.
    """
    tasks = list(results.keys())
    if not tasks:
        print("No results to plot.")
        return

    n_cols = len(tasks)
    n_rows = 3

    # Handle the 'single task' case where axes is 1D
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 12))

    # Ensure axes is always addressable as [row, col]
    if n_cols == 1:
        axis_getter = lambda r, c: axes[r]
    else:
        axis_getter = lambda r, c: axes[r, c]

    # Style mapping to ensure consistent colors
    styles = {
        "AdamW":  {"color": "blue",   "style": ":",  "alpha": 0.7},
        "D-Muon": {"color": "orange", "style": "--", "alpha": 0.8},
        "ANDI":   {"color": "red",    "style": "-",  "alpha": 1.0, "width": 2}
    }

    for i, task in enumerate(tasks):
        opt_results = results[task]

        ax_t = axis_getter(0, i)
        ax_v = axis_getter(1, i)
        ax_g = axis_getter(2, i)

        for opt_name, data in opt_results.items():
            # Data unpack: (best_score, (train, val, grad), lr, time)
            best_score, curves, best_lr, wall_time = data
            train_curve, val_curve, grad_curve = curves

            steps = np.arange(len(train_curve)) * 50

            # Style fallback
            s = styles.get(opt_name, {"color": "black", "style": "-", "alpha": 0.5})
            lw = s.get("width", 1.5)

            # Label includes Best LR
            lbl = f"{opt_name} (lr={best_lr})"

            ax_t.plot(steps, train_curve, color=s["color"], linestyle=s["style"], alpha=s["alpha"], linewidth=lw, label=lbl)
            ax_v.plot(steps, val_curve, color=s["color"], linestyle=s["style"], alpha=s["alpha"], linewidth=lw, label=lbl)
            ax_g.plot(steps, grad_curve, color=s["color"], linestyle=s["style"], alpha=s["alpha"], linewidth=lw, label=lbl)

        # Decorations
        ax_t.set_title(f"{task}\nTraining Loss")
        ax_v.set_title(f"{task}\nValidation Loss")
        ax_g.set_title(f"{task}\nGradient Norm L2")

        for ax in [ax_t, ax_v, ax_g]:
            ax.grid(True, alpha=0.3)
            # IMPROVEMENT: Add legend to every task column, not just the first one
            ax.legend(fontsize='small', loc='best', fancybox=True, framealpha=0.5)

            if task == "AUTOENCODER" and ax != ax_g:
                ax.set_yscale("log")

    plt.tight_layout()
    fname = "benchmark_flexible.png"
    plt.savefig(fname)
    print(f"Plots saved to {fname}")
    plt.show()

# ==========================================
# 6. FLEXIBLE RUNNER (WITH VERIFICATION)
# ==========================================

def run_benchmarks(tasks_to_run=None, quick_debug=False, lrs_andi=None, lrs_muon=None, lrs_adam=None):
    """
    Args:
        tasks_to_run (list): List of strings ["GPT", "RESNET", "AUTOENCODER"]. Default: All.
        quick_debug (bool): If True, runs 1 seed, 1 LR, fewer steps.
    """
    if tasks_to_run is None:
        tasks_to_run = ["GPT", "AUTOENCODER", "RESNET"]

    if quick_debug:
        print("\n>>> QUICK DEBUG MODE ACTIVATED <<<")
        seeds = [42]
        steps = 200
        bs = 64
        lrs_andi = [0.05] if not lrs_andi else lrs_andi
        lrs_muon = [0.02] if not lrs_muon else lrs_muon
        lrs_adam = [1e-3] if not lrs_adam else lrs_adam
    else:
        print("\n>>> FULL BENCHMARK MODE <<<")
        seeds = [42, 1337]
        steps = TRAIN_STEPS
        bs = 128
        lrs_andi = [0.09, 0.05, 0.009] if not lrs_andi else lrs_andi
        lrs_muon = [0.05, 0.02, 0.01] if not lrs_muon else lrs_muon
        lrs_adam = [1e-3, 5e-4, 2e-4] if not lrs_adam else lrs_adam

    results = {}
    hyperparam_logs = []

    model_map = {
        "AUTOENCODER": DeepAutoencoder,
        "GPT": NanoGPT,
        "RESNET": ResNet9
    }

    for task in tasks_to_run:
        print(f"\n--- TASK: {task} ---")
        m = model_map[task]
        results[task] = {}

        # Helper to run search
        def run_search(opt_name, opt_cls, lrs, extra_kwargs, sched_type):
            best = (float('inf'), None, None, None) # score, curves, lr, time
            print(f"  Searching {opt_name}...")

            for lr in lrs:
                kwargs = {"lr": lr, **extra_kwargs}
                t, v, g, tm = train_engine(task, m, opt_cls, kwargs, seeds, batch_size=bs, steps=steps)

                # Score based on last 5 eval steps
                score = np.mean(v[-5:])

                # SANITY CHECK: Print the result of this specific LR
                print(f"    -> [LR: {lr}] Final Val Loss: {score:.4f} (Time: {tm:.1f}s)")

                # Check for NaNs or Inf
                if np.isnan(score) or np.isinf(score):
                    print(f"       [WARNING] Run diverged/crashed. Skipping.")
                    continue

                if score < best[0]:
                    best = (score, (t, v, g), lr, tm)

            # Store result
            if best[2] is None:
                print(f"    [FAILURE] All runs for {opt_name} failed/diverged.")
            else:
                print(f"    => BEST {opt_name}: LR {best[2]} (Val: {best[0]:.4f})")

            results[task][opt_name] = best
            hyperparam_logs.append({
                "Task": task, "Opt": opt_name,
                "LR": best[2], "Val": best[0], "Time": best[3]
            })

        # 1. ANDI
        run_search("ANDI", ANDI, lrs_andi, {"weight_decay": 0.01}, "wsd")

        # 2. DMuon
        run_search("D-Muon", DMuon, lrs_muon, {"momentum": 0.95, "weight_decay": 0.01, "ns_steps": 5}, "wsd")

        # 3. AdamW
        run_search("AdamW", optim.AdamW, lrs_adam, {"weight_decay": 0.01}, "cosine")

    # Final Report
    print("\nSUMMARY:")
    df_logs = pd.DataFrame(hyperparam_logs)
    if not df_logs.empty:
        print(df_logs.to_string())
    else:
        print("No valid results found.")

    # Flexible Plotting
    plot_results(results)

if __name__ == "__main__":
    # 1. Run EVERYTHING (Standard)
    run_benchmarks()

    # 2. Run ONLY selected e.g. GPT (Standard)
    # run_benchmarks(tasks_to_run=["GPT"])        

    # 3. Run EVERYTHING but QUICK check (Code sanity)
    # run_benchmarks(quick_debug=True)

    # 4. Run ONLY GPT, QUICK check (Code sanity)
    # run_benchmarks(tasks_to_run=["GPT"], quick_debug=True)

############################################################
# SUMMARY:
#           Task     Opt     LR       Val        Time
# 0          GPT    ANDI  0.090  1.535093  108.158942
# 1          GPT  D-Muon  0.020  1.554517  126.902201
# 2          GPT   AdamW  0.001  1.577025  103.147051
# 3  AUTOENCODER    ANDI  0.090  0.015622   73.564764
# 4  AUTOENCODER  D-Muon  0.010  0.011245   80.671418
# 5  AUTOENCODER   AdamW  0.001  0.019965   72.198882
# 6       RESNET    ANDI  0.009  0.357959  254.689102
# 7       RESNET  D-Muon  0.010  0.276949  321.450085
# 8       RESNET   AdamW  0.001  0.337882  245.441207