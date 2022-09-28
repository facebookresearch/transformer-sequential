import argparse
import math
import warnings
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_seq import (FeedForwardLayer, MultiHeadSeqAttention,
                                    TransformerOutput)
from models.utils import pos_emb, skew, unskew, parse_layerwise


class VARIANT(IntEnum):
    # staircase transformers
    VARIANT_ONE = 1


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def add_args(parser):
    parser.add_argument(
        "--staircase-size",
        type=int,
        default=64,
        help="number of tokens in each transformer forward",
    )
    parser.add_argument(
        "--max-staircase-size-forward",
        type=int,
        default=63,
        help="max number of fresh tokens considered",
    )
    parser.add_argument(
        "--fix-staircase-size-forward",
        type=int,
        default=-1,
        help="max number of fresh tokens considered",
    )
    parser.add_argument(
        "--validation-staircase-size-forward",
        type=int,
        default=32,
        help="max number of fresh tokens considered during validation",
    )
    parser.add_argument(
        "--staircase-split-query",
        action="store_true",
        default=False,
        help="query and key, value are not the same size",
    )
    parser.add_argument(
        "--staircase-module-fixed-length",
        action="store_true",
        default=False,
        help="init h_prev with 0s to ensure the transformer module has fixed forward length",
    )

    parser.add_argument("--staircase-variant", type=int,
                        default=1, help="1, 2, 3, 4")


class StaircaseSeqAttention(nn.Module):
    """
    Sequential self-attention layer.

    Each position only attends to its previous L positions (doesn't include the current
    position) using relative position embeddings.
    """

    def __init__(self, args):
        super(StaircaseSeqAttention, self).__init__()
        self.args = args
        if args.attn_drop < 0:
            self.dropout = nn.Dropout(args.dropout)
        else:
            self.dropout = nn.Dropout(args.attn_drop)

        self.key_pe, self.val_pe = None, None
        if args.attn_key_mode != "context":
            self.key_pe = pos_emb(args, (1, args.head_dim, args.attn_lim))

    def forward(self, query, key, value):
        # query = B x M x H
        # key, value = B x (M+L) x H
        # mask_causal M * L
        mask_causal = query.new_zeros(
            key.size(1), key.size(1)).fill_(float("-inf"))
        mask_causal = mask_causal.triu(diagonal=1)
        mask_causal = mask_causal[-query.size(1):, ]
        aux_loss = 0

        key_pe, val_pe = self.key_pe, self.val_pe

        attn = 0
        if self.args.attn_key_mode in ["both", "context"]:
            # compute attention from context
            # B x M  x L (src)
            attn = torch.matmul(query, key.transpose(-1, -2))

        if self.args.attn_key_mode in ["both", "position"]:
            # compute the effect of position embedding
            # cut key_pe to be the size of L
            L_size = attn.size(-1)
            attn_pos = torch.matmul(query, key_pe[:, :, -L_size:])  # B x M x L
            attn_pos = skew(attn_pos, 0)  # B x M x (N + L)
            attn_pos = attn_pos[:, :, -L_size - 1: -1]  # B x M x L
            attn = attn + attn_pos
        attn = attn + mask_causal

        if self.args.pers > 0:
            attn, out_mem = self.pers_mem(query, attn)
        else:
            attn = attn / math.sqrt(self.args.head_dim)  # B x M X L
            attn = F.softmax(attn, dim=-1)

        if self.args.vis:
            self.attn_map_snapshot = attn.detach().cpu()
        attn = self.dropout(attn)  # B x M X L

        out = 0

        # attn_cont = skew(attn, 0)  # B x M X (L+M)
        # out = out + torch.matmul(attn_cont, value)  # B x M x H
        out = out + torch.matmul(attn, value)  # B x S x H

        if self.args.pers > 0:
            out = out + out_mem

        return out, aux_loss


class StaircaseMultiHeadSeqAttention(MultiHeadSeqAttention):
    def __init__(self, args):
        super(StaircaseMultiHeadSeqAttention, self).__init__(args)
        self.args = args
        self.attn = StaircaseSeqAttention(args)


class TransformerModLayer(nn.Module):
    def __init__(self, args, layer_ind):
        super(TransformerModLayer, self).__init__()
        self.args = args
        self.attn = StaircaseMultiHeadSeqAttention(args)
        self.ff = FeedForwardLayer(args)
        self.norm1 = nn.LayerNorm(args.hid_sz)
        self.norm2 = nn.LayerNorm(args.hid_sz)

    def attention(self, query, key, value):
        return self.attn(query, key, value)[0]

    def forward(self, h, context, **kargs):
        # h = B x S x H
        if self.args.pre_norm:
            # add layer norm on context as well
            context = self.norm1(context)
            attn_out = self.attention(self.norm1(h), context, context)
        else:
            attn_out = self.attention(h, context, context)

        # FF
        if self.args.pre_norm:
            h2 = h + attn_out  # B x S x H
            ff_out = self.ff(self.norm2(h2))
            out = h2 + ff_out  # B x S x H
        else:
            h2 = self.norm1(h + attn_out)  # B x S x H
            ff_out = self.ff(h2)
            out = self.norm2(h2 + ff_out)  # B x S x H

        return out

    def get_cache_size(self):
        return 0


class TransformerMod(nn.Module):
    def __init__(self, args):
        super(TransformerMod, self).__init__()
        self.args = args
        self.build_layers()
        for l in range(1, len(self.layers)):
            self.layers[l].attn.attn.key_pe = self.layers[0].attn.attn.key_pe
            self.layers[l].attn.attn.val_pe = self.layers[0].attn.attn.val_pe

    def build_layers(self):
        self.layers = nn.ModuleList()
        if self.args.share_layers:
            self.layers.append(TransformerModLayer(self.args, 0))
        else:
            for l in range(self.args.nlayers):
                args_copy = parse_layerwise(self.args, l)
                self.layers.append(TransformerModLayer(args_copy, l))

    def get_layer(self, layer_ind):
        if self.args.share_layers:
            return self.layers[0]
        else:
            return self.layers[layer_ind]

    def forward(self, h, context):
        # h : B x S x H
        for l in range(self.args.nlayers):
            # only forward size get updated from the context as well
            h = self.get_layer(l)(h, context)  # B x S x H
            forward_size = h.size(1)
            if h.size(1) == context.size(1):
                # self-attention
                context = h
            else:
                context = torch.cat([context[:, :-forward_size, :], h], dim=1)
        return h


class StaircaseModel(nn.Module):
    def __init__(self, args):
        super(StaircaseModel, self).__init__()
        self.args = args
        self.transformer = TransformerMod(args)
        self.fix_staircase_size_forward = self.args.fix_staircase_size_forward
        self.staircase_size = self.args.staircase_size
        self.mem_size = self.args.mem_sz
        self.hidden_size = self.args.hid_sz
        self.variant = VARIANT(self.args.staircase_variant)

        assert self.variant < VARIANT.VARIANT_THREE
        assert self.mem_size % self.fix_staircase_size_forward == 0
        assert self.staircase_size % self.fix_staircase_size_forward == 0
        self.validation_staircase_size_forward = (
            self.args.validation_staircase_size_forward
        )

        self.in_emb = nn.Embedding(args.vocab_sz, args.hid_sz)

        self.out = TransformerOutput(args)
        if args.tied:
            self.out.out_emb.weight = self.in_emb.weight

        if args.emb_drop > 0:
            self.emb_dropout = nn.Dropout(args.emb_drop)
        if args.out_drop > 0:
            self.out_dropout = nn.Dropout(args.out_drop)
        if self.args.pre_norm:
            self.out_norm = nn.LayerNorm(args.hid_sz)

    def init_hid_cache(self, batch_sz):
        # creates a cache of # steps
        # 256 / 64 = 4
        # cache size
        # 192
        # 128
        # 64
        # 0
        steps = self.staircase_size // self.fix_staircase_size_forward
        hid = []
        for i in range(steps):
            cache_size = self.staircase_size - \
                (i + 1) * self.fix_staircase_size_forward
            if cache_size > 0:
                hid.append(
                    [
                        torch.zeros((batch_sz, cache_size, self.hidden_size)).to(
                            self.args.device
                        )
                        for i in range(self.args.nlayers)
                    ]
                )
        return hid

    def get_cache(self, h_prev, idx):
        if idx >= len(h_prev):
            return [None]
        return h_prev[idx]

    def assemble_context(self, cache, prev_outputs, new_tokens):
        context = [cache, prev_outputs, new_tokens]
        context = [i for i in context if i is not None]
        context = torch.cat(context, dim=1)
        return context

    def assemble_query(self, prev_outputs, new_tokens):
        query = [prev_outputs, new_tokens]
        query = [i for i in query if i is not None]
        query = torch.cat(query, dim=1)
        return query

    def get_new_tokens(self, h, start_idx, end_idx):
        if end_idx > h.size(1):
            return None
        return h[:, start_idx:end_idx, :]

    def forward(self, x, h_prev, target=None, **kargs):
        # input h B x M
        # assume h_prev [B, staircase_size], and will init with 0s
        # create output placeholder
        # output [B, mem_size, hidden_size]
        hid_after_embed = self.in_emb(x)  # B x M x H
        if self.args.emb_drop > 0:
            hid_after_embed = self.emb_dropout(hid_after_embed)
        # no cache between forwards

        output = hid_after_embed.new_zeros(
            (hid_after_embed.size(0), self.mem_size, self.hidden_size)
        )
        start_idx = 0
        # generate scheduling for staircases, randomness for training purpose
        total_steps = (
            self.mem_size + self.staircase_size
        ) // self.fix_staircase_size_forward - 1
        prev_output = None
        for step_idx in range(total_steps):
            end_idx = start_idx + self.fix_staircase_size_forward
            cache = self.get_cache(h_prev, step_idx)
            new_tokens = self.get_new_tokens(
                hid_after_embed, start_idx, end_idx)
            # should put into cache:
            cache_id = step_idx - (
                total_steps
                - (self.staircase_size // self.fix_staircase_size_forward)
                + 1
            )
            # consumed all forward steps
            # when should we move prev_output forwards
            if step_idx >= self.staircase_size // self.fix_staircase_size_forward:
                prev_output = prev_output[:,
                                          self.fix_staircase_size_forward:, :]
            # input to the first layer of the model
            if self.variant == VARIANT.VARIANT_ONE:
                context = self.assemble_context(
                    cache[0], prev_output, new_tokens)
                h = self.assemble_query(prev_output, new_tokens)
                assert context.size(1) <= self.staircase_size
                # forward into layers
                cache_for_next = []
                for layer in range(self.args.nlayers):
                    # only forward size get updated from the context as well
                    h = self.transformer.get_layer(layer)(h, context)
                    if h.size(1) == context.size(1):
                        # self-attention
                        context = h
                    elif layer + 1 < self.args.nlayers:
                        context = torch.cat([cache[layer + 1], h], dim=1)
                    # put into cache
                    if cache_id >= 0:
                        cache_for_next.append(h)
                # the output from the last layer
                prev_output = h
                # put into cache
                if len(cache_for_next) > 0:
                    h_prev[cache_id] = cache_for_next
            else:
                raise RuntimeError("Variant 3 and 4 is in another file!")

            start_idx = end_idx
            # put into output
            if step_idx - self.staircase_size // self.fix_staircase_size_forward >= -1:
                offset = (
                    step_idx
                    - self.staircase_size // self.fix_staircase_size_forward
                    + 1
                )
                if self.variant == VARIANT.VARIANT_ONE:
                    output[
                        :,
                        offset
                        * self.fix_staircase_size_forward: offset
                        * self.fix_staircase_size_forward
                        + prev_output.size(1),
                        :,
                    ] = prev_output
        out = output
        if self.args.pre_norm:
            out = self.out_norm(out)
        if self.args.out_drop > 0:
            out = self.out_dropout(out)
        out = self.out(out, target)
        # TODO: need to separate the output for prediction from the output for
        # feeding to the next transformer step.
        return out, h_prev, 0.0
