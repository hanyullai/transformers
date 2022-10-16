import math
import copy
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from torch.nn import CrossEntropyLoss
from torch.nn.utils import skip_init

from ...utils import (
    requires_backends,
)

from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_glm_130b import GLM130BConfig

from ...modeling_utils import PreTrainedModel

from torch import nn

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

from torch.nn import LayerNorm

class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq.half()
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        pass


    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions

@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
               F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k

def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        hidden_size_per_partition,
        layer_id,
        layer_past=None,
        scaling_attention_score=True,
        use_cache=False,
    ):
        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key, key_layer), dim=0)
            value_layer = torch.cat((past_value, value_layer), dim=0)
        
        # seqlen, batch, num_attention_heads, hidden_size_per_attention_head
        seq_len, b, nh, hidden_size = key_layer.shape


        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        query_key_layer_scaling_coeff = float(layer_id + 1)
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
            if not (attention_mask == 0).all():
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

        outputs = (context_layer, present, attention_probs)

        return outputs


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
        self.query_key_value = skip_init(
            torch.nn.Linear,
            hidden_size,
            3 * self.inner_hidden_size,
            bias=bias,
            dtype=params_dtype,
        )

        self.dense = skip_init(
            torch.nn.Linear,
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

    def forward(
        self, 
        hidden_states: torch.Tensor,
        position_ids,
        attention_mask: torch.Tensor, 
        layer_id,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        """
        hidden_states: [seq_len, batch, hidden_size]
        attention_mask: [(1, 1), seq_len, seq_len]
        """
        
        # [seq_len, batch, 3 * hidden_size]
        mixed_raw_layer = self.query_key_value(hidden_states)

        # [seq_len, batch, 3 * hidden_size] --> [seq_len, batch, num_attention_heads, 3 * hidden_size_per_attention_head]
        new_tensor_shape = mixed_raw_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)

        # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
        (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_raw_layer, 3)

        position_ids = position_ids.transpose(0, 1)

        cos, sin = self.rotary_emb(value_layer, seq_len=position_ids.max() + 1)

        # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
        query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, position_ids)

        # [seq_len, batch, hidden_size]
        context_layer, present, attention_probs = attention_fn(
            self=self, 
            query_layer=query_layer, 
            key_layer=key_layer, 
            value_layer=value_layer, 
            attention_mask=attention_mask, 
            hidden_size_per_partition=self.hidden_size_per_partition, 
            layer_id=layer_id, 
            layer_past=layer_past, 
            use_cache=use_cache
        )

        output = self.dense(context_layer)

        outputs = (output, present)

        if output_attentions:
            outputs += (attention_probs,)

        return outputs # output, present, attention_probs

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
        self.dense_h_to_4h = skip_init(
            torch.nn.Linear,
            self.hidden_size,
            2 * self.inner_hidden_size,
            bias=bias,
            dtype=params_dtype,
        )
        # Project back to h.
        self.dense_4h_to_h = skip_init(
            torch.nn.Linear,
            self.inner_hidden_size,
            self.hidden_size,
            bias=bias,
            dtype=params_dtype,
        )
        

    def forward(self, hidden_states):
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

    def forward(
        self, 
        hidden_states: torch.Tensor,
        position_ids,
        attention_mask: torch.Tensor,
        layer_id,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        """
        hidden_states: [seq_len, batch, hidden_size]
        attention_mask: [(1, 1), seq_len, seq_len]
        """

        # Layer norm at the begining of the transformer layer.
        # [seq_len, batch, hidden_size]
        attention_input = self.input_layernorm(hidden_states)

        # Self attention.
        attention_outputs = self.attention(
            attention_input, 
            position_ids, 
            attention_mask=attention_mask, 
            layer_id=layer_id, 
            layer_past=layer_past, 
            use_cache=use_cache, 
            output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]

        # Residual connection.
        alpha = (2 * self.num_layers) ** 0.5
        hidden_states = attention_input * alpha + attention_output

        mlp_input = self.post_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = self.glu(mlp_input)

        # Second residual connection.
        output = mlp_input * alpha + mlp_output

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs # hidden_states, present, attentions
    

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

        self.word_embeddings = skip_init(
            torch.nn.Embedding,
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


    def get_input_embeddings(self):
        return self.word_embeddings


    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def get_masks(self, context_length, device):

        attention_mask = torch.ones((1, context_length, context_length), device=device)
        attention_mask.tril_()
        attention_mask[..., :context_length - 1] = 1
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

        return attention_mask

    def get_position_ids(self, mask_position, context_length, device, gmask=False):

        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1:] = mask_position

        position_ids = position_ids.unsqueeze(0)

        return position_ids

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

            MASK, gMASK = 150000, 150001
            mask_token = MASK if MASK in input_ids else gMASK
            use_gmask = False if MASK in input_ids else gMASK
            seq = input_ids[0].tolist()
            
            mask_position = seq.index(mask_token)

            if attention_mask is None:
                attention_mask = self.get_masks(
                    context_length=len(seq),
                    device=input_ids.device
                )

            if position_ids is None:
                position_ids = self.get_position_ids(
                    mask_position=mask_position,
                    context_length=len(seq),
                    device=input_ids.device,
                    gmask=use_gmask
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # [seq_len, batch, hidden_size]
        hidden_states = inputs_embeds.transpose(0, 1)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[0]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.zeros(1, 1, device=input_ids.device).bool()
            
        else:
            attention_mask = attention_mask.to(input_ids.device)

        for i, layer in enumerate(self.layers):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_ret = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                layer_id=torch.tensor(i),
                layer_past=past_key_values[i],
                use_cache=use_cache,
                output_attentions=output_attentions
            )

            hidden_states = layer_ret[0]

            if use_cache:
                presents = presents + (layer_ret[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_ret[2 if use_cache else 1],)

        # Final layer norm.
        hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class GLM130BForConditionalGeneration(GLM130BPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.hidden_size = config.hidden_size
        # self.params_dtype = torch.half
        # self.vocab_size = config.vocab_size

        self.transformer = GLM130BModel(config)

        self.lm_head = skip_init(
            nn.Linear,
            config.hidden_size, 
            config.vocab_size, 
            bias=False,
            dtype=torch.half
        )

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_masks_and_position_ids(self, mask_position, context_length, device, gmask=False):

        attention_mask = torch.ones((1, context_length, context_length), device=device)
        attention_mask.tril_()
        attention_mask[..., :context_length - 1] = 1
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1:] = mask_position

        position_ids = position_ids.unsqueeze(0)

        return attention_mask, position_ids

    def prepare_inputs_for_generation(
        self, 
        input_ids: torch.LongTensor,
        past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict:

        MASK, gMASK = 150000, 150001
        mask_token = MASK if MASK in input_ids else gMASK
        use_gmask = False if MASK in input_ids else gMASK
        seq = input_ids[0].tolist()
        mask_position = seq.index(mask_token)

        if mask_token not in seq:
            raise ValueError("You have to add either [MASK] or [gMASK] in your input")

        # only last token for input_ids if past is not None
        if past:
            last_token = input_ids[:, -1].unsqueeze(-1)
            position_ids = torch.tensor([[mask_position]], dtype=torch.long, device=input_ids.device)

            return {
                "input_ids": last_token,
                "past_key_values": past,
                "position_ids": position_ids,
            }

        else:

            attention_mask, position_ids = self.get_masks_and_position_ids(
                mask_position=mask_position,
                context_length=len(seq),
                device=input_ids.device,
                gmask=use_gmask
            )

            return {
                "input_ids": input_ids,
                "past_key_values": past,
                "position_ids": position_ids, 
                "attention_mask": attention_mask
            }

    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states).permute(1, 0, 2).contiguous()

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )

    @torch.no_grad()
    def generate(
        self,
        **kwargs,
    ):
        MASK, gMASK = 150000, 150001
        bos, eos = 150004, 150005

        while True:
            output_ids = super().generate(**kwargs)
            
            output_seq = output_ids[0].tolist()
            mask_token = MASK if MASK in output_seq else gMASK
            mask_position = output_seq.index(mask_token)
            bos_position = output_seq.index(bos)
            if eos in output_seq:
                eos_position = output_seq.index(eos)
            else:
                eos_position = len(output_seq)
                
            return_seq = output_seq[:mask_position] + output_seq[bos_position+1:eos_position] + output_seq[mask_position+1:bos_position]

            if mask_token in return_seq: 
                kwargs['input_ids'] = torch.tensor([return_seq + [bos]], dtype=torch.long, device=kwargs['input_ids'].device)
            else:
                break
            
        return torch.tensor([return_seq], dtype=torch.long, device=kwargs['input_ids'].device)

