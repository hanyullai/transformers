# coding=utf-8
# Copyright 2022 shunxing1234 and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for GLM-130B."""
import sys
import unicodedata
from typing import List, Optional
import os
import collections
import re

from ...tokenization_utils import PreTrainedTokenizer, _is_punctuation
from ...utils import logging
from .icetk_glm_130b import _IceTokenizer
from functools import lru_cache

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "ice_text.model"
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "THUDM/GLM-130B": "https://huggingface.co/shunxing1234/GLM/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "THUDM/GLM-130B": 2048,
}


class GLM130BTokenizer(PreTrainedTokenizer):
    """
    Construct a GLM-130B tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids"]

    def __init__(
            self,
            vocab_file='test',
            do_lower_case=True,
            max_len=None,
            bos_token='[sop]',
            eos_token='[eos]',
            pad_token='[pad]',
            mask_token='[MASK]',
            gMASK_token='[gMASK]',
            **kwargs
    ):
        super().__init__(bos_token=bos_token, eos_token=eos_token, max_len=max_len,
                         pad_token=pad_token, mask_token=mask_token,
                         gMASK_token=gMASK_token, **kwargs)

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.gMASK_token = gMASK_token
        self.icetokenizer = _IceTokenizer()


    @property
    def vocab_size(self):
        return self.icetokenizer.vocab_size

    def get_vocab(self):
        return self.icetokenizer.vocab

    def _tokenize(self, text):
        mask_pattern = r"\[g?MASK\]"
        text_list = re.split(mask_pattern, text)
        pattern_list = re.compile(mask_pattern).findall(text)
        split_ids = []
        split_tokens = []
        for i in range(len(pattern_list)):
            pattern = pattern_list[i]
            sub_text = text_list[i]
            split_ids.extend(self.icetokenizer.tokenize(sub_text))
            split_ids.append(self.icetokenizer.get_command(pattern))

        split_ids.extend(self.icetokenizer.tokenize(text_list[-1]))
        split_tokens = [self.icetokenizer.IdToToken(idx) for idx in split_ids]

        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.icetokenizer.TokenToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.icetokenizer.IdToToken(index)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is not None:
            token_ids_0 += token_ids_1
        mask_ids = self.icetokenizer.get_command(self.mask_token)
        gmask_ids = self.icetokenizer.get_command(self.gMASK_token)
        if mask_ids not in token_ids_0 and gmask_ids not in token_ids_0:
            token_ids_0 += [gmask_ids]

        if token_ids_0[-1] != mask_ids and token_ids_0[-1] != gmask_ids:
            token_ids_0 += [self.icetokenizer.get_command(self.eos_token)]

        return token_ids_0

