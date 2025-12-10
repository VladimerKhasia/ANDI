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
import time

# ==========================================
# 0. GLOBAL CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Running ANDI Comprehensive Benchmark on: {DEVICE}")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 1. OPTIMIZERS
# ==========================================

class ANDI_Direct(optim.Optimizer):
    """
    ANDI-Direct (Self-Equilibration).
    Normalizes the gradient by the marginals of the *shifted* topology.
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
                
                # Structural Processing for Matrices (Conv2d/Linear)
                if g.ndim > 1:
                    original_shape = g.shape
                    g_mat = g.reshape(g.shape[0], -1)
                    rows, cols = g_mat.shape
                    
                    if rows > 16 and cols > 16:
                        # 1. Prime Mixing
                        prime_shift = 7
                        g_mix = torch.roll(g_mat, shifts=prime_shift, dims=-1)
                        
                        # 2. Self-Equilibration (Norms of the MIXED view)
                        r_norm = g_mix.norm(dim=1, keepdim=True) + 1e-8
                        c_norm = g_mix.norm(dim=0, keepdim=True) + 1e-8
                        g_white = g_mix / (r_norm + c_norm)
                        
                        # Restore
                        g_white = torch.roll(g_white, shifts=-prime_shift, dims=-1)
                        
                        # 3. Soft-Unit Scaling (Hypotenuse)
                        in_norm = g.norm() + 1e-8
                        target = torch.hypot(in_norm, torch.tensor(1.0, device=DEVICE))
                        g_final = g_white * (target / (g_white.norm() + 1e-8))
                        g_final = g_final.view(original_shape)
                    else:
                        # Fallback
                        target = torch.hypot(g.norm(), torch.tensor(1.0, device=DEVICE))
                        g_final = g * (target / (g.norm() + 1e-8))
                else:
                    target = torch.hypot(g.norm(), torch.tensor(1.0, device=DEVICE))
                    g_final = g * (target / (g.norm() + 1e-8))

                # Momentum
                state = self.state[p]
                if 'momentum_buffer' not in state: state['momentum_buffer'] = g_final.clone()
                buf = state['momentum_buffer']
                buf.mul_(mom).add_(g_final)
                update = g_final.add(buf, alpha=mom) if nest else buf
                p.data.add_(update, alpha=-lr)
        return loss

class ANDI_Lateral(optim.Optimizer):
    """
    ANDI-Lateral (Cross-Gating).
    Normalizes the *shifted* gradient by the marginals of the *original* topology.
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
                        # 1. Calculate Anchors (Original Topology)
                        r_anchor = g_mat.norm(dim=1, keepdim=True) + 1e-8
                        c_anchor = g_mat.norm(dim=0, keepdim=True) + 1e-8
                        denom = r_anchor + c_anchor
                        
                        # 2. Prime Mixing & Lateral Inhibition
                        prime_shift = 7
                        g_shift = torch.roll(g_mat, shifts=prime_shift, dims=-1)
                        
                        # Cross-Gate: Shifted Signal / Original Capacity
                        g_white = g_shift / denom
                        
                        # Restore
                        g_white = torch.roll(g_white, shifts=-prime_shift, dims=-1)
                        
                        # 3. Soft-Unit Scaling
                        in_norm = g.norm() + 1e-8
                        target = torch.hypot(in_norm, torch.tensor(1.0, device=DEVICE))
                        g_final = g_white * (target / (g_white.norm() + 1e-8))
                        g_final = g_final.view(original_shape)
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

# --- MUON Helper ---
def newton_schulz(G, steps=5):
    f_norm = torch.norm(G, p='fro') + 1e-8
    X = G.div(f_norm)
    rows, cols = X.shape
    if cols > rows:
        X = X.t(); transposed = True
        eye = torch.eye(rows, device=G.device)
    else:
        transposed = False
        eye = torch.eye(cols, device=G.device)
    
    for _ in range(steps):
        A = torch.mm(X.t(), X)
        X = 0.5 * torch.mm(X, 3 * eye - A)
    
    if transposed: X = X.t()
    return X * f_norm

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
                    if g_mat.size(0) > 10 and g_mat.size(1) > 10:
                        g_orth = newton_schulz(g_mat, group['ns_steps'])
                        g_orth = g_orth.view(g.shape)
                    else: g_orth = g
                else: g_orth = g
                
                state = self.state[p]
                if 'momentum_buffer' not in state: state['momentum_buffer'] = g_orth.clone()
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(g_orth)
                update = g_orth.add(buf, alpha=group['momentum']) if group['nesterov'] else buf
                p.data.add_(update, alpha=-group['lr'])
        return loss

# ==========================================
# 2. MODELS (Academic Standard)
# ==========================================

# A. MLP (Sanity Check)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x)

# B. Autoencoder (Saddle Point)
class DeepAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Deep and narrow to force vanishing gradients
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.Tanh(), nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 16), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 256), nn.Tanh(), nn.Linear(256, 784), nn.Sigmoid()
        )
    def forward(self, x): return self.decoder(self.encoder(self.flatten(x)))

# C. ResNet-9 (Real World / CNN) - Faster than ResNet18, standard benchmark
class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        def conv_block(in_c, out_c, pool=False):
            layers = [nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU()]
            if pool: layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)
        
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_classes))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)

# D. NanoGPT (Transformer / LLM) - The "Publication Ticket"
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, block_size=64, n_embd=128, n_head=4, n_layer=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.LayerNorm(n_embd),
                CausalSelfAttention(n_embd, n_head, block_size),
                nn.LayerNorm(n_embd),
                nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))
            ) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ==========================================
# 3. DATA & EXPERIMENT ENGINE
# ==========================================

def get_data(task):
    # We use 'in' to match the descriptive task names
    if "MLP" in task or "AutoEnc" in task:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        ds = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=tf)
        ds = torch.utils.data.Subset(ds, range(5000)) 
        return DataLoader(ds, batch_size=128, shuffle=True), None
    
    elif "ResNet" in task:
        tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=tf)
        ds = torch.utils.data.Subset(ds, range(5000))
        return DataLoader(ds, batch_size=64, shuffle=True), None
    
    elif "GPT" in task:
        path = os.path.join(DATA_DIR, 'tinyshakespeare.txt')
        if not os.path.exists(path):
            try:
                r = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
                with open(path, 'w') as f: f.write(r.text)
            except:
                # Fallback if no internet
                with open(path, 'w') as f: f.write("dummy " * 10000)
                
        with open(path, 'r') as f: text = f.read()
        chars = sorted(list(set(text)))
        stoi = {ch:i for i,ch in enumerate(chars)}
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        
        block_size = 64
        batch_size = 32
        def get_batch():
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([data[i:i+block_size] for i in ix])
            y = torch.stack([data[i+1:i+block_size+1] for i in ix])
            return x.to(DEVICE), y.to(DEVICE)
        return get_batch, len(chars)
    
    print(f"Error: Task '{task}' not recognized in get_data")
    return None, None

def run_experiment(task_name, model_cls, steps=300, seeds=[42, 43]):
    print(f"\n>>> Task: {task_name}")
    
    # 1. Setup Data
    loader, vocab_size = get_data(task_name)
    if loader is None: return {} # Safety check
    
    # 2. Define Optimizers to Test
    configs = [
        ("AdamW", optim.AdamW, [1e-3, 3e-4]),
        ("Muon", Muon, [0.02, 0.05]),
        ("ANDI-Direct", ANDI_Direct, [0.02, 0.05]),
        ("ANDI-Lateral", ANDI_Lateral, [0.02, 0.05])
    ]
    
    results = {}
    
    for opt_name, opt_cls, lrs in configs:
        best_loss = float('inf')
        best_lr = lrs[0]
        
        # A. Tuning Phase
        print(f"  Tuning {opt_name}...", end="")
        for lr in lrs:
            seed_everything(42)
            # Check substring for GPT
            if "GPT" in task_name: model = model_cls(vocab_size).to(DEVICE)
            else: model = model_cls().to(DEVICE)
            
            opt = opt_cls(model.parameters(), lr=lr)
            
            temp_loss = 0
            # Create iterator safely
            if "GPT" not in task_name: iter_loader = iter(loader)

            for i in range(50):
                if "GPT" in task_name:
                    x, y = loader()
                    _, loss = model(x, y)
                else:
                    try: batch = next(iter_loader)
                    except: iter_loader = iter(loader); batch = next(iter_loader)
                    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    
                    if "AutoEnc" in task_name:
                        recon = model(x)
                        loss = F.mse_loss(recon, x.view(x.size(0), -1))
                    else:
                        pred = model(x)
                        loss = F.cross_entropy(pred, y)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                temp_loss += loss.item()
            
            avg_loss = temp_loss / 50
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_lr = lr
        print(f" Best LR: {best_lr}")

        # B. Verification Phase
        curves = []
        for seed in seeds:
            seed_everything(seed)
            if "GPT" in task_name: model = model_cls(vocab_size).to(DEVICE)
            else: model = model_cls().to(DEVICE)
            
            opt = opt_cls(model.parameters(), lr=best_lr)
            
            seed_losses = []
            if "GPT" not in task_name: iter_loader = iter(loader)
            
            for i in range(steps):
                if "GPT" in task_name:
                    x, y = loader()
                    _, loss = model(x, y)
                else:
                    try: batch = next(iter_loader)
                    except: iter_loader = iter(loader); batch = next(iter_loader)
                    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    
                    if "AutoEnc" in task_name:
                        recon = model(x)
                        loss = F.mse_loss(recon, x.view(x.size(0), -1))
                    else:
                        pred = model(x)
                        loss = F.cross_entropy(pred, y)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                if i % 10 == 0:
                    seed_losses.append(loss.item())
            curves.append(seed_losses)
        
        results[opt_name] = np.mean(curves, axis=0)
        
    return results
    
# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    tasks = [
        ("Sanity Check (MLP)", MLP),
        ("Saddle Point (AutoEnc)", DeepAutoencoder),
        ("Real World (ResNet9)", ResNet9),
        ("LLM (NanoGPT)", NanoGPT)
    ]
    
    all_data = {}
    for name, cls in tasks:
        all_data[name] = run_experiment(name, cls, steps=300)
        
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (task, data) in enumerate(all_data.items()):
        ax = axes[i]
        for opt, curve in data.items():
            x = np.arange(len(curve)) * 10
            
            # Styling
            if "ANDI" in opt:
                style = '-' if "Lateral" in opt else '--'
                width = 2.5
                alpha = 1.0
            elif "AdamW" in opt:
                style = ':'
                width = 2.0
                alpha = 0.8
            else:
                style = '-.'
                width = 1.5
                alpha = 0.7
                
            ax.plot(x, curve, label=opt, linestyle=style, linewidth=width, alpha=alpha)
            
        ax.set_title(task)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if "Saddle" in task or "MLP" in task: ax.set_yscale('log')
        
    plt.tight_layout()
    plt.savefig("andi_benchmark_results.png")
    plt.show()
    print("Done! Results saved to andi_benchmark_results.png")