wandb_log = True
wandb_project = "owt-10k"
wandb_run_name = "gpt2-MoE-16experts(973M)-top2-aux-0.02"

# these make the total batch size be ~0.5M
# 3 batch size * 1024 block size * 20 gradaccum * 8 GPUs = 491,520
batch_size = 3
block_size = 1024
gradient_accumulation_steps = 20 * 8

# this makes total number of tokens be 1B
max_iters = 10000
lr_decay_iters = 10000
warmup_iters = 100

# model
use_moe = True
num_experts = 16
top_k = 2
expert_load_balance_loss_coef = 0.02

# eval stuff
eval_interval = 200
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
