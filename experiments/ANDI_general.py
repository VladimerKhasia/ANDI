# Code makes the algorithm correctly applicable not only for full fine-tuning but for LoRA as well.
# Complete Benchmark: AdamW vs Muon vs ANDI (Unsloth Implementation)
# Link:  https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama

# import os, re
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth
# else:
#     # Do this only in Colab notebooks! Otherwise use pip install unsloth
#     import torch; v = re.match(r"[0-9]{1,}\.[0-9]{1,}", str(torch.__version__)).group(0)
#     xformers = "xformers==" + ("0.0.33.post1" if v=="2.9" else "0.0.32.post2" if v=="2.8" else "0.0.29.post3")
#     !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
#     !pip install --no-deps unsloth
# !pip install transformers==4.56.2
# !pip install --no-deps trl==0.22.2

import os
import sys
import gc
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import torch.optim as optim
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template

# ==========================================
# 1. OPTIMIZERS (Strict Float32)
# ==========================================

class ANDI(optim.Optimizer):
    """
    ANDI (Adaptive Normalized Direction w/ Identity)
    Generalized for LoRA and Full Training.
    
    Args:
        min_dim (int): The minimum dimension size required to trigger the 
                       Self-Equilibration logic. 
                       If you use LoRA r < 4 than use min_dim=r
    """
    # CHANGE 1: Added weight_decay=0.0 to __init__
    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.9, nesterov=True, min_dim=4):
        # CHANGE 2: Added weight_decay to defaults
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, min_dim=min_dim)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay'] # CHANGE 3a: Retrieve WD
            mom = group['momentum']
            nest = group['nesterov']
            min_dim = group['min_dim'] 
            
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad

                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                # Check dimensions to see if we apply ANDI or Fallback
                is_large_enough = False
                if g.ndim > 1:
                    original_shape = g.shape
                    g_mat = g.reshape(g.shape[0], -1)
                    rows, cols = g_mat.shape
                    
                    if rows >= min_dim and cols >= min_dim:
                        is_large_enough = True

                if is_large_enough:
                    # === PRIMARY ALGORITHM (Self-Equilibration) ===
                    # 1. Calculate Norms
                    r_norm = g_mat.norm(dim=1, keepdim=True) + 1e-8
                    c_norm = g_mat.norm(dim=0, keepdim=True) + 1e-8
                    
                    # 2. Equilibrate
                    g_white = g_mat / (r_norm + c_norm)
                    g_white = g_white.view(original_shape)

                    # 3.  Scale (Preserve Magnitude, floor at 1.0)
                    in_norm = g.norm() + 1e-8
                    target = torch.hypot(in_norm, torch.tensor(1.0, device=g.device))
                    
                    # 4. Apply 
                    g_final = g_white * (target / (g_white.norm() + 1e-8))
                else:
                    target = torch.hypot(g.norm(), torch.tensor(1.0, device=g.device))
                    g_final = g * (target / (g.norm() + 1e-8))

                # === MOMENTUM & UPDATE STEP ===
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = g_final.clone()
                
                buf = state['momentum_buffer']
                buf.mul_(mom).add_(g_final)
                
                update = g_final.add(buf, alpha=mom) if nest else buf
                p.data.add_(update, alpha=-lr)
                
        return loss

class Muon(optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
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
            nest = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None: continue

                g = p.grad.float()

                if wd > 0:
                    p.data.add_(p.data, alpha=-lr * wd)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(mom).add_(g)
                update_g = g.add(buf, alpha=mom) if nest else buf

                if update_g.ndim == 2 and update_g.size(0) > 10 and update_g.size(1) > 10:
                    g_ortho = self.zeropower_via_newtonschulz5(update_g, steps=ns_steps)
                    row, col = update_g.size()
                    scale_factor = 0.2 * max(row, col)
                    p.data.add_(g_ortho, alpha=-lr * scale_factor)
                else:
                    p.data.add_(update_g, alpha=-lr)
        return loss

    def zeropower_via_newtonschulz5(self, G, steps=5, eps=1e-7):
        assert len(G.shape) == 2
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G.float()
        X /= (X.norm() + eps)
        if G.size(0) > G.size(1): X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = A @ X
            X = a * X + b * B + c * A @ B
        if G.size(0) > G.size(1): X = X.T
        return X

# ==========================================
# 2. EXPERIMENTAL SETUP
# ==========================================

MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
SEED = 3407
BATCH_SIZE = 6  #2  Unsloth NB
GRAD_ACCUM = 2  #4  Unsloth NB
MAX_STEPS = 300 #60 Unsloth NB

# 1. Dataset Prep
dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
dataset = dataset.train_test_split(test_size=200, seed=SEED)

def force_cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def run_experiment(run_name, optimizer_class, lr, custom_wd):
    print(f"\n> RUN: {run_name} | LR: {lr} | WD: {custom_wd}")

    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # 2. Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = SEED,
    )

    # === 3. PRECISION ===
    # We must explicitly cast trainable params to Float32.
    # This replaces the job that "adamw_8bit" usually does internally.
    # Without this, BF16 rounding will kill small Weight Decay updates.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # 4. This applies the Llama-3 special tokens correctly
    tokenizer = get_chat_template(tokenizer, chat_template = "llama-3")

    def format_func(examples):
        texts = []
        for i, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            # Create the message structure expected by Unsloth/Llama-3
            messages = [
                {"role": "user", "content": f"{i}\n{input_text}"},
                {"role": "assistant", "content": output},
            ]
            # Apply template (handles BOS/EOS/Special Tokens)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    train_dataset = dataset["train"].map(format_func, batched=True)
    eval_dataset = dataset["test"].map(format_func, batched=True)

    # 5. Setup Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_class is None:
        custom_opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=custom_wd)
    else:
        custom_opt = optimizer_class(trainable_params, lr=lr, weight_decay=custom_wd)

    # 6. Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        packing = False,
        optimizers = (custom_opt, None),
        args = SFTConfig(
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps = 20, # Unsloth default 5 increased to 20 because  default 60 steps were increased to 300
            max_steps = MAX_STEPS,
            learning_rate = lr,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "sgd",
            weight_decay = 0.0,
            lr_scheduler_type = "cosine",  #"linear", # Unsloth default "linear" changed to "cosine" -> to help AdamW
            seed = SEED,
            output_dir = f"outputs_{run_name.replace(' ', '_')}",
            report_to = "none",
            eval_strategy = "steps",
            eval_steps = 20,
        ),
    )

    trainer_stats = trainer.train()

    loss_history = []
    eval_history = []
    for log in trainer.state.log_history:
        if "loss" in log and "eval_loss" not in log:
            loss_history.append({"step": log["step"], "loss": log["loss"], "run_name": run_name})
        elif "eval_loss" in log:
            eval_history.append({"step": log["step"], "eval_loss": log["eval_loss"], "run_name": run_name})

    del model, tokenizer, trainer, custom_opt
    force_cleanup()

    return loss_history, eval_history

# ==========================================
# 3. EXECUTE BENCHMARK
# ==========================================
force_cleanup()
all_train = []
all_eval = []

# Baseline: AdamW (WD=0.01)
# Note: Unsloth uses 0.001, but we use 0.01 here to match comunity defaults for all optimizers.
t, e = run_experiment("AdamW (WD=0.01)", None, 1e-3, 0.01)     # community default: PyTorch LR 1e-3 WD 0.01        #unsloth LR default 2e-4 but try 5e-4, 3e-4 -> all worsen results for fp32   
all_train.extend(t); all_eval.extend(e)

# Muon (WD=0.1)
t, e = run_experiment("Muon (WD=0.1)", Muon, 5e-3, 0.1)        # community defult: PyTorch
all_train.extend(t); all_eval.extend(e)

# ANDI (WD=0.1)
t, e = run_experiment("ANDI (WD=0.1)", ANDI, 5e-2, 0.1)        # we set WD=0.1 just for comparability with MUON as ANDI does not depend much on WD's etc.     
all_train.extend(t); all_eval.extend(e)

# ==========================================
# 4. PLOTTING
# ==========================================
df_train = pd.DataFrame(all_train)
df_eval = pd.DataFrame(all_eval)
df_train['loss_smooth'] = df_train.groupby('run_name')['loss'].transform(lambda x: x.ewm(alpha=0.1).mean())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
sns.lineplot(data=df_train, x="step", y="loss_smooth", hue="run_name", ax=ax1, linewidth=2, palette="viridis")
ax1.set_title("Training Loss (Smoothed)")
if not df_eval.empty:
    sns.lineplot(data=df_eval, x="step", y="eval_loss", hue="run_name", ax=ax2, linewidth=2, marker="o", palette="viridis")
    ax2.set_title("Validation Loss")
plt.tight_layout()
plt.show()