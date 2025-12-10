# ANDI - Lateral

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import requests

# ==========================================
# 0. GLOBAL SETUP
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Running ANDI Benchmark on: {DEVICE}")

def seed_everything(seed=422):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 1. OPTIMIZER: ANDI (Fixed & Structural)
# ==========================================

class ANDI(optim.Optimizer):
    """
    ANDI: Arithmetic Normalization / Decorrelated Inertia
    
    A linear-time O(N) structural optimizer.
    
    Core Mechanisms:
    1. Prime Topology Mixing (Fixed): 
       We calculate energy statistics on the ORIGINAL grid, but apply them 
       to a SHIFTED (Prime Rolled) grid. This forces features to equilibrate 
       against distant neighbors, breaking local grid-lock artifacts.
       
    2. One-Shot Arithmetic Equilibration:
       Approximates Matrix Balancing using Arithmetic Mean (R+C) instead of 
       Geometric Mean. This provides a stable, conservative lower bound 
       on the divisor.
       
    3. Hypotenuse Energy Scaling:
       Soft-clipping mechanism that boosts weak signals (Saddle Points) 
       while preserving strong signals (Convex Valleys).
       Target = sqrt(||G||^2 + 1)
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
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                
                original_shape = g.shape
                
                # We apply logic only to tensors with sufficient dimensions (Matrices/Convs)
                # 1D Vectors (Biases) skip the structural phase.
                if g.ndim > 1:
                    # Flatten to 2D (Out, In)
                    g_mat = g.reshape(g.shape[0], -1)
                    rows, cols = g_mat.shape
                    
                    if rows > 16 and cols > 16:
                        # PHASE 1: ARITHMETIC EQUILIBRATION (The "Anchor")
                        # =================================================
                        # Calculate the Norms (Marginals) of the ORIGINAL grid.
                        
                        # Row Norms (Filter Magnitude)
                        r_marginals = g_mat.norm(dim=1, keepdim=True) + 1e-8
                        # Col Norms (Feature Activation Magnitude)
                        c_marginals = g_mat.norm(dim=0, keepdim=True) + 1e-8
                        
                        # The Arithmetic Denominator (Broadcasted implicit matrix)
                        # Denom represents the "Capacity" of the current grid location.
                        denom = r_marginals + c_marginals
                        
                        # =================================================
                        # PHASE 2: PRIME TOPOLOGY MIXING (The "Twist")
                        # =================================================
                        # We shift the gradient columns by a Prime Number (7).
                        # We do NOT shift the denominator.
                        
                        prime_shift = 7
                        g_shifted = torch.roll(g_mat, shifts=prime_shift, dims=-1)
                        
                        # MISALIGNMENT STRATEGY (Lateral Inhibition):
                        # We divide the SHIFTED gradient by the UNSHIFTED denominator.
                        # This normalizes Feature[k] using the Capacity of Position[k+7].
                        # If Position[k+7] is "loud" (high norm), it suppresses Feature[k].
                        g_white = g_shifted / denom
                        
                        # Restore Topology: Unroll back to original positions.
                        g_white = torch.roll(g_white, shifts=-prime_shift, dims=-1)
                        
                        # =================================================
                        # PHASE 3: HYPOTENUSE ENERGY SCALING (The "Gain")
                        # =================================================
                        # The whitening step changes the global energy.
                        # We re-project it onto a "Safe Manifold".
                        
                        in_norm = g.norm() + 1e-8
                        curr_norm = g_white.norm() + 1e-8
                        
                        # Target: If Input is small (<1), boost to ~1. 
                        #         If Input is large (>>1), keep it large.
                        target_norm = torch.hypot(in_norm, torch.tensor(1.0, device=g.device))
                        
                        ratio = target_norm / curr_norm
                        g_final = g_white * ratio
                        g_final = g_final.view(original_shape)
                        
                    else:
                        # Fallback for small layers: Simple Hypotenuse Boost
                        in_norm = g.norm() + 1e-8
                        target_norm = torch.hypot(in_norm, torch.tensor(1.0, device=g.device))
                        g_final = g * (target_norm / in_norm)
                else:
                    # Fallback for 1D vectors: Simple Hypotenuse Boost
                    in_norm = g.norm() + 1e-8
                    target_norm = torch.hypot(in_norm, torch.tensor(1.0, device=g.device))
                    g_final = g * (target_norm / in_norm)

                # =================================================
                # PHASE 4: MOMENTUM & UPDATE
                # =================================================
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.clone(g_final).detach()
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g_final)
                
                update = g_final.add(buf, alpha=momentum) if nesterov else buf
                p.data.add_(update, alpha=-lr)
                
        return loss

# ==========================================
# 2. BASELINES (Newton-Schulz / MUON)
# ==========================================

def newton_schulz(G, steps=5):
    """
    Standard Iterative Whitening (Used in MUON/Shampoo).
    Scales O(N^3) or O(Iter * N^2).
    """
    f_norm = torch.norm(G, p='fro') + 1e-8
    X = G.div(f_norm)
    rows, cols = X.shape
    if cols > rows:
        X = X.t()
        eye = torch.eye(rows, device=G.device)
        transposed = True
    else:
        eye = torch.eye(cols, device=G.device)
        transposed = False 
    
    # Iterative refinement
    for _ in range(steps):
        A = torch.mm(X.t(), X)
        X = 0.5 * torch.mm(X, 3 * eye - A)
        
    if transposed: X = X.t()
    return X * f_norm

class StandardMUON(optim.Optimizer):
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
                original_shape = g.shape
                
                # Flatten >2D tensors
                if g.ndim > 2: g = g.flatten(1)
                
                # Apply Newton-Schulz if matrix is large enough
                if g.ndim == 2 and g.size(0) > 10 and g.size(1) > 10:
                    g_orth = newton_schulz(g, group['ns_steps'])
                else: 
                    g_orth = g
                
                g_orth = g_orth.view(original_shape)
                
                state = self.state[p]
                if 'momentum_buffer' not in state: 
                    state['momentum_buffer'] = torch.clone(g_orth).detach()
                
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(g_orth)
                update = g_orth.add(buf, alpha=group['momentum']) if group['nesterov'] else buf
                p.data.add_(update, alpha=-group['lr'])
        return loss

# ==========================================
# 3. EXPERIMENT SUITE
# ==========================================

class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28*28, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(self.flatten(x))

class SaddleAutoencoder(nn.Module):
    """Deep narrow autoencoder designed to cause vanishing gradients."""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128), nn.Tanh(), 
            nn.Linear(128, 32), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128), nn.Tanh(), 
            nn.Linear(128, 28*28), nn.Sigmoid()
        )
    def forward(self, x): return self.decoder(self.encoder(self.flatten(x)))

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), 
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.classifier(self.features(x))

class CharLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, vocab_size)
    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        return self.fc(out) 

def get_dataloader(task_name):
    print(f"Loading data for: {task_name}...")
    if task_name == "Sanity Check":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dset = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        dset = torch.utils.data.Subset(dset, range(2000))
        return DataLoader(dset, batch_size=64, shuffle=True)
    elif task_name == "Saddle Point":
        transform = transforms.ToTensor() 
        dset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        dset = torch.utils.data.Subset(dset, range(2000))
        return DataLoader(dset, batch_size=64, shuffle=True)
    elif task_name == "Real World":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        dset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
        dset = torch.utils.data.Subset(dset, range(2000)) 
        return DataLoader(dset, batch_size=32, shuffle=True)
    elif task_name == "Recurrent":
        path = os.path.join(DATA_DIR, 'tinyshakespeare.txt')
        if not os.path.exists(path):
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            try:
                with open(path, 'w') as f: f.write(requests.get(url).text)
            except:
                with open(path, 'w') as f: f.write("dummy " * 1000)
        with open(path, 'r') as f: text = f.read()
        chars = sorted(list(set(text)))
        stoi = { ch:i for i,ch in enumerate(chars) }
        data_ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        seq_len = 64
        num_samples = 1000 
        X, Y = [], []
        for i in range(num_samples):
            idx = random.randint(0, len(data_ids) - seq_len - 1)
            X.append(data_ids[idx:idx+seq_len])
            Y.append(data_ids[idx+1:idx+seq_len+1])
        dset = TensorDataset(torch.stack(X), torch.stack(Y))
        return DataLoader(dset, batch_size=32, shuffle=True), len(chars)
    return None

def run_training_step(model, batch, task_type, device):
    x, y = batch
    x, y = x.to(device), y.to(device)
    if task_type == "Reconstruction":
        recon = model(x)
        loss = F.mse_loss(recon, x.view(x.size(0), -1))
        return loss
    elif task_type == "Language":
        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
        return loss
    else: 
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        return loss

def execute_experiment(task_name, model_class, task_type, tuning_steps=40, full_steps=250, seeds=[42]):
    print(f"\n>>> STARTING EXPERIMENT: {task_name}")
    if task_name == "Recurrent":
        loader, vocab_size = get_dataloader(task_name)
        model_args = (vocab_size,)
    else:
        loader = get_dataloader(task_name)
        model_args = ()
    
    adam_lrs = [1e-3, 3e-4]
    muon_lrs = [0.05, 0.02]
    andi_lrs = [0.05, 0.02] # ANDI can handle high LRs like MUON

    optimizers_to_test = ["Adam", "StandardMUON", "ANDI"]
    best_configs = {}
    
    print("  Phase 1: Tuning Learning Rates...")
    for opt_name in optimizers_to_test:
        if opt_name == "Adam": lrs = adam_lrs
        elif opt_name == "StandardMUON": lrs = muon_lrs
        else: lrs = andi_lrs
        
        best_loss = float('inf')
        best_lr = lrs[0]
        
        for lr in lrs:
            seed_everything(42)
            model = model_class(*model_args).to(DEVICE)
            
            if opt_name == "Adam": opt = optim.Adam(model.parameters(), lr=lr)
            elif opt_name == "StandardMUON": opt = StandardMUON(model.parameters(), lr=lr)
            else: opt = ANDI(model.parameters(), lr=lr)
            
            iter_loader = iter(loader)
            total_loss = 0
            for i in range(tuning_steps):
                try: batch = next(iter_loader)
                except StopIteration: 
                    iter_loader = iter(loader)
                    batch = next(iter_loader)
                opt.zero_grad()
                loss = run_training_step(model, batch, task_type, DEVICE)
                loss.backward()
                # Clip RNN for Adam/MUON fairness, but ANDI handles it natively
                if task_name == "Recurrent": torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / tuning_steps
            if avg_loss < best_loss and not math.isnan(avg_loss):
                best_loss = avg_loss
                best_lr = lr
        best_configs[opt_name] = best_lr
        print(f"    {opt_name} Best LR: {best_lr} (Loss: {best_loss:.4f})")

    print("  Phase 2: Verification Run...")
    results = {k: [] for k in optimizers_to_test}
    for opt_name, lr in best_configs.items():
        seed_curves = []
        for seed in seeds:
            seed_everything(seed)
            model = model_class(*model_args).to(DEVICE)
            
            if opt_name == "Adam": opt = optim.Adam(model.parameters(), lr=lr)
            elif opt_name == "StandardMUON": opt = StandardMUON(model.parameters(), lr=lr)
            else: opt = ANDI(model.parameters(), lr=lr)
            
            losses = []
            iter_loader = iter(loader)
            for i in range(full_steps):
                try: batch = next(iter_loader)
                except StopIteration: 
                    iter_loader = iter(loader)
                    batch = next(iter_loader)
                opt.zero_grad()
                loss = run_training_step(model, batch, task_type, DEVICE)
                loss.backward()
                if task_name == "Recurrent": torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if i % 10 == 0:
                    val = loss.item()
                    losses.append(val if not math.isnan(val) else 10.0)
            seed_curves.append(losses)
        results[opt_name] = np.array(seed_curves)
    return results, best_configs

def run_suite():
    experiments = [
        {"name": "Sanity Check", "model": BaselineMLP, "type": "Classification"},
        {"name": "Saddle Point", "model": SaddleAutoencoder, "type": "Reconstruction"},
        {"name": "Real World", "model": SimpleCNN, "type": "Classification"},
        {"name": "Recurrent", "model": CharLSTM, "type": "Language"}
    ]
    all_results = {}
    for exp in experiments:
        res, configs = execute_experiment(exp["name"], exp["model"], exp["type"])
        all_results[exp["name"]] = (res, configs)

    print("\n>>> PLOTTING RESULTS...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (name, (data, configs)) in enumerate(all_results.items()):
        ax = axes[idx]
        for opt_name, curves in data.items():
            mean = np.mean(curves, axis=0)
            x = np.arange(len(mean)) * 10
            
            # Styling
            if opt_name == "ANDI": 
                ax.plot(x, mean, label=f"{opt_name} (Ours)", linewidth=2.5, color='green')
            else:
                ax.plot(x, mean, label=opt_name, linewidth=1.5, alpha=0.8)
                
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_ylabel("Loss")
        ax.set_xlabel("Steps")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if name in ["Sanity Check", "Saddle Point"]: ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_suite()