# coding=utf-8
# Copyright 2022 THUDM and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLM-130B model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)

GLM_130B_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "THUDM/GLM-130B": "https://huggingface.co/shunxing1234/GLM/resolve/main/config.json",
}


class GLM130BConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~GLM130BModel`].
    It is used to instantiate an GLM model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the GLM [shunxing1234/GLM-base-cased](https://huggingface.co/shunxing1234/GLM-base-cased) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the GLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~GLMModel`] or
            [`~TFGLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~GLMModel`] or
            [`~TFGLMModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import GLMModel, GLMConfig

    >>> # Initializing a GLM shunxing1234/GLM-base-cased style configuration
    >>> configuration = GLMConfig()

    >>> # Initializing a model from the shunxing1234/GLM-base-cased style configuration
    >>> model = GLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "glm_130b"
    attribute_map = {
        "num_hidden_layers": "num_layers"
    }

    def __init__(
        self,
        vocab_size=150528,
        hidden_size=12288,
        num_layers=70,
        num_attention_heads=96,
        layernorm_epsilon=1e-5,
        use_cache=False,
        bos_token_id=150004,
        eos_token_id=150005,
        max_sequence_length=2048,
        inner_hidden_size=32768,
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.layernorm_epsilon = layernorm_epsilon
        self.inner_hidden_size = inner_hidden_size
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

