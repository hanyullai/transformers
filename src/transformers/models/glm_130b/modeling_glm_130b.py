import math
import copy
import torch
import torch.nn.functional as F

from .position_embedding import RotaryEmbedding, apply_rotary_pos_emb_index

from ...utils import (
    requires_backends,
)

from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from .configuration_glm_130b import GLM130BConfig

from ...modeling_utils import PreTrainedModel

from torch import nn
from torch.nn.utils import skip_init

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm
    class LayerNorm(FusedLayerNorm):
        def __init__(self, *args, pb_relax=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.pb_relax = pb_relax

        def forward(self, x):
            if not self.pb_relax:
                return super().forward(x)
            return super().forward(x / (x.abs().max().detach() / 8))
except ModuleNotFoundError:
    print('Please install apex to use fused_layer_norm, fall back to torch.nn.LayerNorm')
    from torch.nn import LayerNorm



def get_masks_and_position_ids(seq, mask_position, context_length, gmask=False):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., : context_length - 1] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    position_ids = torch.arange(len(seq), dtype=torch.long, device=tokens.device)
    if not gmask:
        position_ids[context_length - 1 :] = mask_position

    position_ids = position_ids.unsqueeze(0)

    return tokens, attention_mask, position_ids

def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        hidden_size_per_partition,
        scaling_attention_score=True,
        mems=None,
        **kwargs
    ):

        mem = mems[kwargs["layer_id"]] if mems is not None else None

        # seqlen, batch, num_attention_heads, hidden_size_per_attention_head
        seq_len, b, nh, hidden_size = key_layer.shape

        # b, seqlen, stack, head, hidden
        cache_kv = (
            torch.stack((key_layer, value_layer))
            .permute(2, 1, 0, 3, 4)
            .detach()
            .contiguous()
            .view(b, seq_len, nh * hidden_size * 2)
        )
        kwargs["output_this_layer"]["mem_kv"] = cache_kv

        if mem is not None:  # the first time, mem is None
            # might change batch_size
            # b, seqlen, stack, head, hidden -> stack, seqlen, b, head, hidden
            mem = mem.expand(b, -1, -1).reshape(b, mem.shape[1], 2, nh, hidden_size).permute(2, 1, 0, 3, 4)
            memk, memv = mem[0], mem[1]
            key_layer = torch.cat((memk, key_layer), dim=0)
            value_layer = torch.cat((memv, value_layer), dim=0)

        query_key_layer_scaling_coeff = float(kwargs["layer_id"] + 1)
        if scaling_attention_score:
            query_layer = query_layer / (math.sqrt(hidden_size) * query_key_layer_scaling_coeff)

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=1.0,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        if self.scale_mask_softmax:
            self.scale_mask_softmax.scale = query_key_layer_scaling_coeff
            attention_probs = self.scale_mask_softmax(attention_scores, attention_mask.contiguous())
        else:
            if not (attention_mask.shape[-2] == 1 and (attention_mask > 0).all()):
                # if auto-regressive, skip
                attention_scores.masked_fill_(attention_mask, -10000.0)

            attention_scores = attention_scores.float()
            attention_scores = attention_scores * query_key_layer_scaling_coeff

            attention_probs = F.softmax(attention_scores, dim=-1)

            attention_probs = attention_probs.half()


        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                 layer_id, hidden_size_per_attention_head=None, bias=True,
                 params_dtype=torch.float):
        super(SelfAttention, self).__init__()
        
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.hidden_size_per_partition = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_heads_per_partition = num_attention_heads
        self.rotary_emb = RotaryEmbedding(
            self.hidden_size // self.num_attention_heads,
            base=10000,
            precision=torch.half,
            learnable=False,
        )

        try:
            from apex.transformer.functional import FusedScaleMaskSoftmax
            from apex.transformer.enums import AttnMaskType
            self.scale_mask_softmax = FusedScaleMaskSoftmax(
                input_in_fp16=True,
                input_in_bf16=False,
                attn_mask_type=AttnMaskType.padding,
                scaled_masked_softmax_fusion=True,
                mask_func=self.attention_mask_func,
                softmax_in_fp32=True,
                scale=1,
            )
        except ModuleNotFoundError:
            print(
                "Please install apex to use FusedScaleMaskSoftmax, otherwise the inference efficiency will be greatly reduced"
            )
        
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head

        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head


        # Strided linear layer.
        self.query_key_value = skip_init(torch.nn.Linear,
            hidden_size,
            3 * self.inner_hidden_size,
            bias=bias,
            dtype=params_dtype,
        )

        self.dense = skip_init(torch.nn.Linear,
            self.inner_hidden_size,
            hidden_size,
            bias=bias,
            dtype=params_dtype,
        )
    
    @staticmethod
    def attention_mask_func(attention_scores, attention_mask):
        attention_scores.masked_fill_(attention_mask, -10000.0)
        return attention_scores

    def split_tensor_along_last_dim(self, tensor, num_partitions,
                                contiguous_split_chunks=False):
        """Split a tensor along its last dimension.
        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                    in memory.
        """
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def forward(self, hidden_states, mask, *args, **kw_args):
        """
        hidden_states: [seq_len, batch, hidden_size]
        mask: [(1, 1), seq_len, seq_len]
        """
        
        # [seq_len, batch, 3 * hidden_size]
        torch.set_printoptions(precision=50)
        print(hidden_states.device, self.query_key_value.weight.device, self.query_key_value.bias.device)
        print(hidden_states.float().abs().mean())
        print(self.query_key_value.weight.float().abs().mean(), self.query_key_value.bias.float().abs().mean())
        mixed_raw_layer = self.query_key_value(hidden_states)

        # [seq_len, batch, 3 * hidden_size] --> [seq_len, batch, num_attention_heads, 3 * hidden_size_per_attention_head]
        new_tensor_shape = mixed_raw_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)

        # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
        (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_raw_layer, 3)

        kw_args["position_ids"] = kw_args["position_ids"].transpose(0, 1)

        cos, sin = self.rotary_emb(value_layer, seq_len=kw_args["position_ids"].max() + 1)

        # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
        query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, kw_args["position_ids"])

        # [seq_len, batch, hidden_size]
        context_layer = attention_fn(self, query_layer, key_layer, value_layer, mask, self.hidden_size_per_partition, **kw_args)

        output = self.dense(context_layer)

        return output

class GEGLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_fn = F.gelu

    def forward(self, x):
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        return x1 * self.activation_fn(x2)

class GLU(torch.nn.Module):
    def __init__(self, hidden_size, inner_hidden_size=None,
        layer_id=None, bias=True, activation_func=GEGLU(), params_dtype=torch.float):
        super(GLU, self).__init__()
        self.layer_id = layer_id
        self.activation_func = activation_func

        # Project to 4h.
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = skip_init(torch.nn.Linear,
            self.hidden_size,
            2 * self.inner_hidden_size,
            bias=bias,
            dtype=params_dtype,
        )
        # Project back to h.
        self.dense_4h_to_h = skip_init(torch.nn.Linear,
            self.inner_hidden_size,
            self.hidden_size,
            bias=bias,
            dtype=params_dtype,
        )
        

    def forward(self, hidden_states, **kw_args):
        """
        hidden_states: [seq_len, batch, hidden_size]
        """
        
        # [seq_len, batch, 2 * inner_hidden_size]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)

        intermediate_parallel = self.activation_func(intermediate_parallel)

        output = self.dense_4h_to_h(intermediate_parallel)

        return output


class GLM130BBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        layernorm_epsilon,
        layer_id,
        inner_hidden_size=None,
        hidden_size_per_attention_head=None,
        layernorm=LayerNorm,
        use_bias=True,
        params_dtype=torch.float,
        num_layers=70
    ):
        super(GLM130BBlock, self).__init__()
        # Set output layer initialization if not provided.

        self.layer_id = layer_id

        # Layernorm on the input data.
        self.input_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            bias=use_bias,
            params_dtype=params_dtype,
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        self.num_layers = num_layers

        # GLU
        self.glu = GLU(
            hidden_size,
            inner_hidden_size=inner_hidden_size,
            bias=use_bias,
            layer_id=layer_id,
            params_dtype=params_dtype,
        )

    def forward(self, hidden_states, mask, *args, **kw_args):
        """
        hidden_states: [seq_len, batch, hidden_size]
        mask: [(1, 1), seq_len, seq_len]
        """

        # Layer norm at the begining of the transformer layer.
        # [seq_len, batch, hidden_size]
        attention_input = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output = self.attention(attention_input, mask, **kw_args)

        # Residual connection.
        alpha = (2 * self.num_layers) ** 0.5
        hidden_states = attention_input * alpha + attention_output

        mlp_input = self.post_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = self.glu(mlp_input, **kw_args)

        # Second residual connection.
        output = mlp_input * alpha + mlp_output

        return output
    

class GLM130BPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    is_parallelizable = True
    supports_gradient_checkpointing = False
    config_class = GLM130BConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLM130BBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        return 



class GLM130BModel(GLM130BPreTrainedModel):
    def __init__(self, config: GLM130BConfig):
        requires_backends(self, "apex")
        super().__init__(config)

        # recording parameters
        self.max_sequence_length = config.max_sequence_length
        self.hidden_size = config.hidden_size
        self.params_dtype = torch.half
        self.num_attention_heads = config.num_attention_heads
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.layernorm_epsilon = config.layernorm_epsilon
        self.inner_hidden_size = config.inner_hidden_size
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads

        self.word_embeddings = skip_init(torch.nn.Embedding,
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_size, 
            dtype=self.params_dtype
        )
            
        def get_layer(layer_id):
            return GLM130BBlock(
                self.hidden_size,
                self.num_attention_heads,
                self.layernorm_epsilon,
                layer_id,
                inner_hidden_size=self.inner_hidden_size,
                hidden_size_per_attention_head=self.hidden_size_per_attention_head,
                layernorm=LayerNorm,
                use_bias=True,
                params_dtype=self.params_dtype,
            )

        self.layers = torch.nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(self.num_layers)]
        )

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(self.hidden_size, eps=self.layernorm_epsilon)

        self.final_word_embeddings = skip_init(torch.nn.Embedding,
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_size, 
            dtype=self.params_dtype
        )


    def get_input_embeddings(self):
        return self.word_embeddings


    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings


    def forward(self, input_ids, position_ids, attention_mask, *,
                output_hidden_states=False, **kw_args):
        # sanity check
        assert len(input_ids.shape) >= 2
        batch_size, query_length = input_ids.shape[:2]

        if attention_mask is None:
            attention_mask = torch.ones(1, 1, device=input_ids.device).type_as(
                next(self.parameters())
            )  # None means full attention
        assert len(attention_mask.shape) == 2 or \
            len(attention_mask.shape) == 4 and attention_mask.shape[1] == 1

        # initial output_cross_layer might be generated by word/position_embedding_forward
        output_cross_layer = {}

        # [seq_len, batch, hidden_size]
        hidden_states = self.word_embeddings(input_ids).transpose(0, 1)

        output_per_layers = []

        output_this_layer = []

        print(self.final_word_embeddings.weight.float().abs().mean())
        print(self.word_embeddings.weight.float().abs().mean())

        for i, layer in enumerate(self.layers):

            print(layer.attention.query_key_value.weight.float().abs().mean())
            print('l', layer.attention.query_key_value.bias.float().abs().mean())
            print('h', hidden_states.float().abs().mean())

            output_this_layer_obj, output_cross_layer_obj = {}, {}

            layer_ret = layer(hidden_states, attention_mask,
                layer_id=torch.tensor(i),
                **kw_args,
                position_ids=position_ids,
                **output_cross_layer,
                output_this_layer=output_this_layer_obj, output_cross_layer=output_cross_layer_obj)


            if isinstance(layer_ret, tuple):
                layer_ret = layer_ret[0]

            hidden_states, output_this_layer, output_cross_layer = layer_ret, output_this_layer_obj, output_cross_layer_obj

            if output_hidden_states:
                output_this_layer['hidden_states'] = hidden_states
            output_per_layers.append(output_this_layer)

        # Final layer norm.
        logits = self.final_layernorm(hidden_states)

        logits = F.linear(logits, self.final_word_embeddings.weight).transpose(0, 1).contiguous()

        outputs = [logits]
        outputs.extend(output_per_layers)

        return outputs