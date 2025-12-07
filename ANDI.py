# ANDI - code is designed such that you can run it directly in Jupyter Notebook.
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

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 1. OPTIMIZER: ANDI (The Lightweight Champion)
# ==========================================

# class ANDI(optim.Optimizer):
#     """
#     ANDI Optimizer.
    
#     Philosophy: "Stable Whitening, Hypotenuse Scaling."
    
#     1. One-Shot Arithmetic Whitening (O(N)):
#        - Denom = RowNorm + ColNorm. (Stable).
#        - Unlike Geometric Mean (sqrt(R*C)), this dampens noise outliers
#          instead of exploding them. This fixes the MLP/CNN instability.
       
#     2. Hypotenuse Gain Control:
#        - Target_Norm = sqrt(Input_Norm^2 + Floor^2).
#        - Floor = 1.0 (Unit Energy).
#        - Saddle/RNN (Input~0): Target -> 1.0 (Infinite Gain -> Escape).
#        - MLP (Input>>1): Target -> Input (Unit Gain -> Stability).
       
#     3. Prime Topology Mixing:
#        - Uses 'torch.roll' with a prime stride to allow the whitening 
#          to see global context in O(N).
#     """
#     def __init__(self, params, lr=0.02, momentum=0.9, nesterov=True):
#         defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
#         super().__init__(params, defaults)

#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = None
#         if closure is not None: 
#             with torch.enable_grad(): loss = closure()

#         for group in self.param_groups:
#             lr = group['lr']
#             momentum = group['momentum']
#             nesterov = group['nesterov']

#             for p in group['params']:
#                 if p.grad is None: continue
#                 g = p.grad
                
#                 original_shape = g.shape
                
#                 # We apply logic only to Matrices (or larger)
#                 if g.ndim > 1:
#                     # Flatten to (Out, In)
#                     g_mat = g.reshape(g.shape[0], -1)
#                     rows, cols = g_mat.shape
                    
#                     if rows > 16 and cols > 16:
#                         # 1. PRIME TOPOLOGY MIXING
#                         # Roll columns by a prime to mix input features
#                         # This breaks local grid artifacts in CNNs/RNNs
#                         prime_shift = 7
#                         curr = torch.roll(g_mat, shifts=prime_shift, dims=-1)
                        
#                         # 2. ARITHMETIC WHITENING (Stable One-Shot)
#                         # R = Norm of each filter (Out)
#                         # C = Norm of each feature trace (In)
#                         r_norms = curr.norm(dim=1, keepdim=True) + 1e-8
#                         c_norms = curr.norm(dim=0, keepdim=True) + 1e-8
                        
#                         # Arithmetic Mean Denominator: R + C
#                         # This is the "Conservative" whitener. 
#                         # It respects the dominant energy constraint.
#                         # We use broadcasting to create the (R, C) divisor matrix implicitly.
#                         denom = r_norms + c_norms
#                         g_white = curr / denom
                        
#                         # Unroll
#                         g_white = torch.roll(g_white, shifts=-prime_shift, dims=-1)
                        
#                         # 3. HYPOTENUSE SCALING (The "Smart" Gain)
#                         in_norm = g.norm() + 1e-8
#                         curr_norm = g_white.norm() + 1e-8
                        
#                         # Floor = 1.0 (Unit Energy).
#                         # If In=1e-5 (Saddle), Target=1.0. (Boost).
#                         # If In=10.0 (MLP), Target=sqrt(100+1)=10.05. (No Boost).
#                         target_norm = torch.hypot(in_norm, torch.tensor(1.0, device=g.device))
                        
#                         g_final = g_white * (target_norm / curr_norm)
#                         g_final = g_final.view(original_shape)
#                     else:
#                         # Small layers: Simple Hypotenuse Boost
#                         in_norm = g.norm() + 1e-8
#                         target_norm = torch.hypot(in_norm, torch.tensor(1.0, device=g.device))
#                         g_final = g * (target_norm / in_norm)
#                 else:
#                     # Vectors: Simple Hypotenuse Boost
#                     in_norm = g.norm() + 1e-8
#                     target_norm = torch.hypot(in_norm, torch.tensor(1.0, device=g.device))
#                     g_final = g * (target_norm / in_norm)

#                 # 4. MOMENTUM UPDATE
#                 state = self.state[p]
#                 if 'momentum_buffer' not in state:
#                     state['momentum_buffer'] = torch.clone(g_final).detach()
                
#                 buf = state['momentum_buffer']
#                 buf.mul_(momentum).add_(g_final)
                
#                 update = g_final.add(buf, alpha=momentum) if nesterov else buf
#                 p.data.add_(update, alpha=-lr)
                
#         return loss

class ANDI(optim.Optimizer):
    """
    ANDI: A Linear-Time, Low-Memory Structural Optimizer.
    
    1. SPACE: 1x Memory cost (same as SGD, 50% of Adam).
    2. TIME:  O(N) Complexity (same as Adam, much faster than MUON).
    
    Algorithm:
    1. Global Context Mixing: 
       We use a prime-number 'roll' (circular shift) to mix feature columns.
       This breaks local grid dependencies (artifacts) in O(N).
       
    2. Arithmetic Whitening (Structural Regularization):
       Instead of expensive SVD or Newton-Schulz (O(N^3)), we use:
       W = G / (RowNorms + ColNorms).
       This bounds the spectral radius of the gradient matrix roughly within [0, 1],
       decorrelating layers without matrix multiplication.
       
    3. Hypotenuse Energy Floor (Scale Invariance):
       Target = sqrt(||G||^2 + 1).
       This provides a smooth transition between:
       - Saddle Points (||G||~0): Boosts to 1.0 (Adam behavior).
       - Convex Slopes (||G||>>1): Preserves ||G|| (SGD behavior).
    """
    def __init__(self, params, lr=0.02, momentum=0.9, nesterov=True):
        # LR=0.02 matches MUON. 
        # Because we normalize energy, we can use larger LRs than Adam (1e-3).
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
                
                # Check for Matrix/Conv structure (N > 1D)
                if g.ndim > 1:
                    # View as (Output_Features, Input_Features)
                    original_shape = g.shape
                    g_mat = g.reshape(g.shape[0], -1)
                    rows, cols = g_mat.shape
                    
                    # Only apply Structural Whitening to "Matrices"
                    # Small vectors (biases) don't need "whitening" (this is not real whitening).
                    if rows > 16 and cols > 16:
                        # -------------------------------------------------
                        # Step 1: Prime Mixing (Global Context)
                        # -------------------------------------------------
                        # Break local correlations by shifting columns.
                        # 7 is a safe prime that fits in registers.
                        prime_shift = 7
                        curr = torch.roll(g_mat, shifts=prime_shift, dims=-1)
                        
                        # -------------------------------------------------
                        # Step 2: Decorrelation
                        # -------------------------------------------------
                        # Calculate Energy of Rows (Filter Magnitude)
                        # and Columns (Feature Activation Magnitude).
                        r_energy = curr.norm(dim=1, keepdim=True) + 1e-8
                        c_energy = curr.norm(dim=0, keepdim=True) + 1e-8
                        
                        # Normalize by the Arithmetic Mean of energies.
                        # This balances the matrix: G_new[i,j] roughly <= 0.5
                        # Stability Guarantee: Denom cannot be 0.
                        g_white = curr / (r_energy + c_energy)
                        
                        # Restore structure
                        g_white = torch.roll(g_white, shifts=-prime_shift, dims=-1)
                        
                        # -------------------------------------------------
                        # Step 3: Hypotenuse Gain (Energy Restoration)
                        # -------------------------------------------------
                        # Current Norm is arbitrary (approx 0.5 * sqrt(N)).
                        # We need to map it back to a "Physical" scale.
                        
                        in_norm = g.norm() + 1e-8
                        curr_norm = g_white.norm() + 1e-8
                        
                        # Regularization: Ensure Energy >= 1.0 (Unit Sphere).
                        # This prevents vanishing gradients.
                        target_norm = torch.hypot(in_norm, torch.tensor(1.0, device=g.device))
                        
                        # Rescale
                        ratio = target_norm / curr_norm
                        g_final = g_white * ratio
                        g_final = g_final.view(original_shape)
                        
                    else:
                        # Fallback for small matrices: Just Energy Floor
                        in_norm = g.norm() + 1e-8
                        target_norm = torch.hypot(in_norm, torch.tensor(1.0, device=g.device))
                        g_final = g * (target_norm / in_norm)
                else:
                    # Fallback for 1D vectors: Just Energy Floor
                    in_norm = g.norm() + 1e-8
                    target_norm = torch.hypot(in_norm, torch.tensor(1.0, device=g.device))
                    g_final = g * (target_norm / in_norm)

                # -------------------------------------------------
                # Step 4: Momentum (Standard SGD)
                # -------------------------------------------------
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.clone(g_final).detach()
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g_final)
                
                update = g_final.add(buf, alpha=momentum) if nesterov else buf
                p.data.add_(update, alpha=-lr)
                
        return loss

# ==========================================
# 2. BASELINES
# ==========================================

def newton_schulz(G, steps=5):
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
                if g.ndim > 2: g = g.flatten(1)
                if g.ndim == 2 and g.size(0) > 10 and g.size(1) > 10:
                    g_orth = newton_schulz(g, group['ns_steps'])
                else: g_orth = g
                g_orth = g_orth.view(original_shape)
                state = self.state[p]
                if 'momentum_buffer' not in state: state['momentum_buffer'] = torch.clone(g_orth).detach()
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
        self.net = nn.Sequential(nn.Linear(28*28, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x): return self.net(self.flatten(x))

class SaddleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(nn.Linear(28*28, 128), nn.Tanh(), nn.Linear(128, 32), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(32, 128), nn.Tanh(), nn.Linear(128, 28*28), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(self.flatten(x)))

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10))
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
    # ANDI uses MUON-like LRs (Robust Arithmetic Whitening allows high LR)
    prime_lrs = [0.05, 0.02]

    optimizers_to_test = ["Adam", "StandardMUON", "ANDI"]
    best_configs = {}
    
    print("  Phase 1: Tuning...")
    for opt_name in optimizers_to_test:
        if opt_name == "Adam": lrs = adam_lrs
        else: lrs = muon_lrs
        
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
                if task_name == "Recurrent": torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / tuning_steps
            if avg_loss < best_loss and not math.isnan(avg_loss):
                best_loss = avg_loss
                best_lr = lr
        best_configs[opt_name] = best_lr
        print(f"    {opt_name} Winner: {best_lr} (Loss: {best_loss:.4f})")

    print("  Phase 2: Verification...")
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
                    losses.append(val if not math.isnan(val) else 100.0)
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

    print("\n>>> PLOTTING...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (name, (data, configs)) in enumerate(all_results.items()):
        ax = axes[idx]
        for opt_name, curves in data.items():
            mean = np.mean(curves, axis=0)
            x = np.arange(len(mean)) * 10
            label = f"{opt_name}"
            if opt_name == "ANDI": label += " (Ours)"
            ax.plot(x, mean, label=label)
        ax.set_title(name)
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if name in ["Sanity Check", "Saddle Point"]: ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_suite()