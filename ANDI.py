#Complete Benchmark: ANDI 
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
from dataclasses import dataclass

# ==========================================
# 0. CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# Steps sufficient to see convergence/escape
TRAIN_STEPS = 2500
EVAL_INTERVAL = 50
BATCH_SIZE = 128
SEEDS = [42, 1337]

print(f"Running Comprehensive ANDI Benchmark on: {DEVICE}")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 1. OPTIMIZERS
# ==========================================

class ANDI(optim.Optimizer):
    """
    (Row + Col Norms) without prime stride mixing.
    """
    def __init__(self, params, lr=0.02, momentum=0.9, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
          with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr, mom, nest = group['lr'], group['momentum'], group['nesterov']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad

                if g.ndim > 1:
                    original_shape = g.shape
                    g_mat = g.reshape(g.shape[0], -1)
                    rows, cols = g_mat.shape

                    if rows > 16 and cols > 16:
                        # Self-Equilibration
                        r_norm = g_mat.norm(dim=1, keepdim=True) + 1e-8
                        c_norm = g_mat.norm(dim=0, keepdim=True) + 1e-8
                        g_white = g_mat / (r_norm + c_norm)
                        g_white = g_white.view(original_shape)

                        # Scale
                        in_norm = g.norm() + 1e-8
                        target = torch.hypot(in_norm, torch.tensor(1.0, device=DEVICE))
                        g_final = g_white * (target / (g_white.norm() + 1e-8))
                    else:
                        target = torch.hypot(g.norm(), torch.tensor(1.0, device=DEVICE))
                        g_final = g * (target / (g.norm() + 1e-8))
                else:
                    target = torch.hypot(g.norm(), torch.tensor(1.0, device=DEVICE))
                    g_final = g * (target / (g.norm() + 1e-8))

                state = self.state[p]
                if 'momentum_buffer' not in state: state['momentum_buffer'] = g_final.clone()
                buf = state['momentum_buffer']
                buf.mul_(mom).add_(g_final)
                update = g_final.add(buf, alpha=mom) if nest else buf
                p.data.add_(update, alpha=-lr)
        return loss

# --- Baselines ---
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

class Muon(optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.9, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
          with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                if g.ndim > 1:
                    g_mat = g.reshape(g.shape[0], -1)
                    if g_mat.size(0) > 32 and g_mat.size(1) > 32:
                        g_orth = newton_schulz_5(g_mat, steps=group['ns_steps'])
                        scale_factor = max(1, g_mat.size(0)/g_mat.size(1))**0.5
                        g_final = g_orth.view(g.shape) * scale_factor
                    else: g_final = g
                else: g_final = g
                state = self.state[p]
                if 'momentum_buffer' not in state: state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(g_final)
                update = g_final.add(buf, alpha=group['momentum']) if group['nesterov'] else buf
                p.data.add_(update, alpha=-group['lr'])
        return loss

# ==========================================
# 2. MODELS & DATA
# ==========================================
class DeepAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 256), nn.Tanh(), nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 16), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 256), nn.Tanh(), nn.Linear(256, 784), nn.Sigmoid())
    def forward(self, x):
        return self.decoder(self.encoder(x.view(x.size(0), -1)))

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

def get_loaders(task):
    if task == "SADDLE":
        tf = transforms.Compose([transforms.ToTensor()])
        ds = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=tf)
        return torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True), None
    elif task == "CIFAR":
        tf = transforms.Compose([transforms.RandomCrop(32,4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.49,0.48,0.44),(0.2,0.19,0.2))])
        ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=tf)
        return torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True), None
    elif task == "GPT":
        return None, None

# ==========================================
# 3. ENGINE
# ==========================================

def train_engine(task_name, model_fn, optimizer_cls, opt_kwargs, seeds):
    losses_matrix = []

    if task_name == "GPT":
        path = os.path.join(DATA_DIR, 'ts.txt')
        if not os.path.exists(path):
            try:
                r = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
                with open(path, 'w') as f: f.write(r.text)
            except Exception as e:
                print(f"Failed to download data: {e}. Creating dummy.")
                with open(path, 'w') as f: f.write("dummy " * 10000)

        text = open(path, 'r').read(); chars = sorted(list(set(text))); stoi = {c:i for i,c in enumerate(chars)}
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        vocab_size = len(chars)
        def get_batch():
            ix = torch.randint(len(data) - 64, (BATCH_SIZE,))
            x = torch.stack([data[i:i+64] for i in ix])
            y = torch.stack([data[i+1:i+64+1] for i in ix])
            return x.to(DEVICE), y.to(DEVICE)
        loader = None; gpt_loader = get_batch
    else:
        loader, _ = get_loaders(task_name)
        vocab_size = 0; gpt_loader = None

    for seed in seeds:
        seed_everything(seed)
        model = model_fn(vocab_size).to(DEVICE) if task_name == "GPT" else model_fn().to(DEVICE)
        opt = optimizer_cls(model.parameters(), **opt_kwargs)
        seed_losses = []
        if task_name != "GPT": iter_loader = iter(loader)

        for i in range(TRAIN_STEPS):
            if task_name == "GPT":
                x, y = gpt_loader()
                _, loss = model(x, y)
            else:
                try: batch = next(iter_loader)
                except: iter_loader = iter(loader); batch = next(iter_loader)
                x = batch[0].to(DEVICE)
                if task_name == "SADDLE": loss = F.mse_loss(model(x), x.view(x.size(0), -1))
                else: loss = F.cross_entropy(model(x), batch[1].to(DEVICE))

            opt.zero_grad(); loss.backward(); opt.step()
            if i % EVAL_INTERVAL == 0: seed_losses.append(loss.item())
        losses_matrix.append(seed_losses)
    return np.mean(losses_matrix, axis=0)

def search_lr(task, model_fn, opt_cls, space, name):
    print(f" > Search {name} {space}...")
    best_loss = float('inf'); best_curve = None; best_lr = space[0]

    for lr in space:
        curve = train_engine(task, model_fn, opt_cls, {"lr": lr}, [SEEDS[0]])
        final_val = curve[-1]
        if not np.isnan(final_val) and final_val < best_loss:
            best_loss = final_val; best_curve = curve; best_lr = lr

    print(f"   * Best {name}: {best_lr}")
    return train_engine(task, model_fn, opt_cls, {"lr": best_lr}, SEEDS)

# ==========================================
# 4. EXECUTION & PLOTTING
# ==========================================

def run_experiment():
    tasks = ["SADDLE", "CIFAR", "GPT"]
    results = {}

    search_space = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

    for task in tasks:
        print(f"\n--- {task} ---")
        results[task] = {}
        if task == "SADDLE": m = DeepAutoencoder
        elif task == "CIFAR": m = ResNet9
        elif task == "GPT": m = NanoGPT

        results[task]["AdamW"] = train_engine(task, m, optim.AdamW, {"lr": 3e-4 if task=="GPT" else 1e-3}, SEEDS)
        results[task]["Muon"] = train_engine(task, m, Muon, {"lr": 0.02 if task=="GPT" else 0.05}, SEEDS)

        results[task]["ANDI"] = search_lr(task, m, ANDI, search_space, "ANDI")

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, task in enumerate(tasks):
        ax = axes[i]
        steps = np.arange(len(results[task]["AdamW"])) * EVAL_INTERVAL

        ax.plot(steps, results[task]["AdamW"], 'b:', label="AdamW", alpha=0.5)
        ax.plot(steps, results[task]["Muon"], color='orange', linestyle='-.', label="Muon", alpha=0.5)
        ax.plot(steps, results[task]["ANDI"], 'g-', label="ANDI", linewidth=2)

        ax.set_title(f"{task}")
        ax.grid(True, alpha=0.3)

        if task == "GPT":
            all_curves = [results[task][k] for k in results[task].keys()]
            flat = [v for curve in all_curves for v in curve[1:] if not np.isnan(v)]
            if flat: ax.set_ylim(min(flat)*0.95, max(flat)*1.05)
        elif task == "SADDLE":
            ax.set_yscale("log")

        ax.set_xlabel("Steps")
        if i == 0: ax.set_ylabel("Loss")
        ax.legend()

    plt.tight_layout()
    plt.savefig("andi_clean_benchmark.png")
    plt.show()

if __name__ == "__main__":
    run_experiment()