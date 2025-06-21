from typing import Optional, List
import torch
from torch import logit, nn
from torch.nn import functional as F
from dataclasses import dataclass

from .gpt import Transformer, ModelArgs, RMSNorm

SEQUENCES = {
    "fwd-rev": 2,
}


@dataclass
class NHeadedGPTConfig(ModelArgs):
    output_heads: int = 1
    sequences: str = "fwd-rev"


class MultiNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, output_heads: int = 1):
        super().__init__()
        self.norms = nn.ModuleList([RMSNorm(dim, eps=eps) for _ in range(output_heads)])

    def forward(self, x: torch.Tensor):
        # Apply each norm to the input tensor
        return torch.stack([norm(x) for norm in self.norms], dim=1)


class MultiHead(nn.Module):
    def __init__(self, dim: int, vocab_size: int, bias=False, output_heads: int = 1):
        super().__init__()
        self.output_heads = output_heads
        self.heads = nn.ModuleList(
            [nn.Linear(dim, vocab_size, bias=bias) for _ in range(output_heads)]
        )

    def forward(self, x: torch.Tensor):
        return torch.stack(
            [self.heads[i](x[:, i]) for i in range(self.output_heads)], dim=1
        )


class NHeadedGPT(Transformer):
    def __init__(self, config: NHeadedGPTConfig):
        super().__init__(config)
        self.output_heads = config.output_heads
        self.sequence_type = config.sequences

        assert (
            self.sequence_type in SEQUENCES
        ), f"Invalid sequence type: {self.sequence_type}, choose from {list(SEQUENCES.keys())}"
        assert (
            self.output_heads == SEQUENCES[self.sequence_type]
        ), "Mismatch between output heads and sequence type"

        self.norm = MultiNorm(
            config.dim, eps=config.norm_eps, output_heads=self.output_heads
        )
        self.output = MultiHead(
            config.dim,
            config.vocab_size,
            bias=False,
            output_heads=self.output_heads,
        )
        self.nhead_initialize_weights()

    def nhead_initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        for mod in self.output.heads:
            nn.init.constant_(mod.weight, 0)
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        self.output.heads[1] = self.output.heads[0]  # share weights for fwd-rev
        self.norm.norms[0] = self.norm.norms[1]  # share norms for fwd-rev

    def build_seq_dirs(self, idx: torch.Tensor):
        """
        Builds the seq directions for the input tensor idx.
        This should only be used during training.

        Args:
            idx (torch.Tensor): Input tensor of shape (bs, seq_len)
        """
        if self.sequence_type == "fwd-rev":
            # For fwd-rev, we flip and stack the forawrd and reversed sequences
            idx_fwd = idx
            idx_rev = torch.flip(idx_fwd, dims=[-1])
            idx = torch.stack((idx_fwd, idx_rev), dim=1)

        return idx

    def reverse_seq_dirs(self, logits: torch.Tensor):
        """
        Aligns the logits of the various sequence directions.

        Args:
            logits (torch.Tensor): Logits of shape (bs, sequence_dirs, seq_len, vocab_size)

        Returns:
            logits (torch.Tensor): Logits of shape (bs, sequence_dirs, seq_len, vocab_size)
        """
        if self.sequence_type == "fwd-rev":
            # slice the reversed logits and flip them (b, sd, sl, d) -> (b, sl, d)
            logits[:, 1] = torch.flip(logits[:, 1], dims=[1])
        return logits

    def forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        if idx is not None and cond_idx is not None:  # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[
                :, : self.cls_token_num
            ]
            token_embeddings = self.tok_embeddings(
                idx
            )  # idx shape: (bs, sequence_dirs, seq_len) -> (bs, sequence_dirs, seq_len, dim)
            token_embeddings = token_embeddings.sum(dim=1)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
        else:
            if cond_idx is not None:  # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[
                    :, : self.cls_token_num
                ]
            else:  # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(idx)

            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis

        if self.training:
            freqs_cis = self.freqs_cis[: token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]

        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)

        # output layers
        h = self.norm(h)
        logits = self.output(h).float()
        if self.training:
            if self.output_heads > 1:
                logits = logits[:, :, self.cls_token_num - 1 :].contiguous()
            else:
                logits = logits[:, self.cls_token_num - 1 :].contiguous()
        logits = self.reverse_seq_dirs(logits)

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            # This branch is for validation
            loss_all = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            valid_all = valid[:, None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            # This branch is for training

            # why do we sum over sequence directions?
            # because log(p(x_i)) = log(p(x_i | x_{<i})) + log(p(x_i | x_{>i}))
            # product of probabilities is equivalent to summing the log probabilities
            poe_logits = logits.sum(dim=1)  # sum over sequence directions
            loss = F.cross_entropy(
                poe_logits.view(-1, poe_logits.size(-1)), targets.view(-1)
            )

        return logits, loss


### class-conditional
def MultiHeadGPT_XXXL(**kwargs):
    return NHeadedGPT(
        NHeadedGPTConfig(n_layer=48, n_head=40, dim=2560, **kwargs)
    )  # 3.9B


def MultiHeadGPT_XXL(**kwargs):
    return NHeadedGPT(
        NHeadedGPTConfig(n_layer=48, n_head=24, dim=1536, **kwargs)
    )  # 1.4B


def MultiHeadGPT_XL(**kwargs):
    return NHeadedGPT(
        NHeadedGPTConfig(n_layer=36, n_head=20, dim=1280, **kwargs)
    )  # 775M


def MultiHeadGPT_L(**kwargs):
    return NHeadedGPT(
        NHeadedGPTConfig(n_layer=24, n_head=16, dim=1024, **kwargs)
    )  # 343M


def MultiHeadGPT_B(**kwargs):
    return NHeadedGPT(
        NHeadedGPTConfig(n_layer=12, n_head=12, dim=768, **kwargs)
    )  # 111M


MultiGPT_models = {
    "MultiGPT-B": MultiHeadGPT_B,
    "MultiGPT-L": MultiHeadGPT_L,
    "MultiGPT-XL": MultiHeadGPT_XL,
    "MultiGPT-XXL": MultiHeadGPT_XXL,
    "MultiGPT-XXXL": MultiHeadGPT_XXXL,
}


if __name__ == "__main__":
    latent_size = 256 // 16
    MultiHeadGPT_B(
        vocab_size=16384,
        block_size=latent_size**2,
        num_classes=1000,
        cls_token_num=1,
        model_type="c2i",
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        drop_path_rate=0.0,
        token_dropout_p=0.0,
        output_heads=2,
    )
