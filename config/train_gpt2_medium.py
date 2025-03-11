# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = "owt-10k"
wandb_run_name = "gpt2-medium(353M)"

# these make the total batch size be ~0.5M
# 5 batch size * 1024 block size * 12 gradaccum * 8 GPUs = 491,520
batch_size = 5
block_size = 1024
gradient_accumulation_steps = 12 * 8

# model parameters
n_layer = 24
n_head = 16
n_embd = 1024

# this makes total number of tokens be 5B
max_iters = 10000
lr_decay_iters = 10000
warmup_iters = 100

# eval stuff
eval_interval = 200
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
