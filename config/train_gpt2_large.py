# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = "owt-10k"
wandb_run_name = "gpt2-large(774M)"

# these make the total batch size be ~0.5M
# 2 batch size * 1024 block size * 30 gradaccum * 8 GPUs = 491,520
batch_size = 2
block_size = 1024
gradient_accumulation_steps = 30 * 8

# model parameters
n_layer = 36
n_head = 20
n_embd = 1280

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
