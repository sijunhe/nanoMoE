"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model import GPT, MLP, CausalSelfAttention, LayerNorm


class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        self.dropout = nn.Dropout(config.dropout)
        self.gate = nn.Linear(self.n_embd, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])

    def forward(self, x):
        batch_size, sequence_length, hidden_dim = x.shape
        # reshape x into 2-dimensional tensor since batch_size and sequence_lengh doesn't matter in FFN, unlike attention
        x = x.view(-1, hidden_dim)  # (batch * seq, num_experts)
        router_logits = self.gate(x)  # (batch * seq, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        # routing_weights (batch * seq, top_k)
        # selected_experts (batch * seq, top_k)
        if self.top_k != 1:
            # re-normalize the top-k routing weights
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # cast back to the input dtype
        routing_weights = routing_weights.to(x.dtype)

        # init an empty tensor to accumulate the output of each expert
        total_output_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)
        # (batch * seq, top_k, num_experts) => (num_experts, top_k, batch * seq)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            top_k_idx, token_idx = torch.where(expert_mask[expert_idx])
            # expert_mask[expert_idx][top_k_idx][token_idx] is routed the expert_idx-th turned on
            # top_k_idx:  0 ... top_k
            # token_idx: token 0 ... batch_size * seq_length

            # current state: all token inputs related to this expert
            # Add None in order to broadcast
            current_state = x[None, token_idx].reshape(-1, hidden_dim)
            # forward the expert and then multiply by the weights
            current_hidden_states = (
                expert_layer(current_state)
                * routing_weights[token_idx, top_k_idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            total_output_states.index_add_(
                0, token_idx, current_hidden_states.to(x.dtype)
            )

        total_output_states = total_output_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

        return total_output_states, router_logits


class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.moe_ffn = MoEBlock(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        moe_output, routing_weights = self.moe_ffn(self.ln_2(x))
        x = x + moe_output
        return x, routing_weights


@dataclass
class MoEGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    num_experts: int = 4  # number of total experts in the MoE Layer
    top_k: int = 1  # nomber of active experts in the MoE Layer
    expert_load_balance_loss_coef: float = 0.02
    gradient_checkpointing: bool = False


def expert_load_balance_loss(
    router_logits, num_experts: int, top_k: int
) -> torch.Tensor:
    # roting_weights: a tuple of logits, each with the shape of [batch_size * seq_length, num_experts]

    compute_device = router_logits[0].device
    concatenated_router_logits = torch.cat(
        [layer_gate.to(compute_device) for layer_gate in router_logits], dim=0
    )

    routing_weights = torch.nn.functional.softmax(concatenated_router_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))

    return overall_loss * num_experts


class MoEGPT(GPT):
    def __init__(self, config):
        nn.Module.__init__(self)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([MoELayer(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        router_logits_all_layers = None
        for block in self.transformer.h:
            if self.config.gradient_checkpointing and self.training:
                x, router_logits = checkpoint(block.__call__, x)
            else:
                x, router_logits = block(x)
            if router_logits_all_layers is None:
                router_logits_all_layers = (router_logits,)
            else:
                router_logits_all_layers = (router_logits,) + router_logits_all_layers
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            aux_loss = expert_load_balance_loss(
                router_logits=router_logits_all_layers,
                num_experts=self.config.num_experts,
                top_k=self.config.top_k,
            )
            # only add aux loss for training, val loss is next-token-prediction loss only
            if self.training:
                loss += self.config.expert_load_balance_loss_coef * aux_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None
            aux_loss = None

        return logits, loss, aux_loss
