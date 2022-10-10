# -*- encoding: utf-8 -*-

import os
import sys
import math
import random
from typing import List, Tuple, Union

import torch

import sentencepiece as spm
from . import sentencepiece_model_pb2 as model

class SPTokenizer:
    def __init__(self, model_path):
        self.proto = model.ModelProto()
        with open(model_path, 'rb') as fin:
            proto_str = fin.read()
            self.proto.ParseFromString(proto_str)
        self.refresh()
        
    def refresh(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_proto=self.proto.SerializeToString())
        self.num_tokens = self.sp.vocab_size()

    def add_special_tokens(self, tokens):
        for token in tokens:
            new_token = model.ModelProto().SentencePiece()
            new_token.piece = token
            new_token.score = 0
            self.proto.pieces.append(new_token)
        self.refresh()

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: List[int]):
        return self.sp.DecodeIds(ids)

    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)

    def convert_tokens_to_ids(self, tokens):
        return [self.sp.PieceToId(token) for token in tokens]

    def convert_token_to_id(self, token):
        return self.sp.PieceToId(token)

    def convert_id_to_token(self, idx):
        return self.sp.IdToPiece(idx)
    
    def __len__(self):
        return self.num_tokens

class TextTokenizer(SPTokenizer):
    def __init__(self, path='~/ice_text.model', device='cuda', fp16=True):
        self.configure(path, device, fp16)

        fp = os.path.join(path)
        super().__init__(fp)
        
    def configure(self, path=None, device=None, fp16=None):
        if path is not None:
            self.path = os.path.expanduser(path)
        if device is not None:
            self.device = device
        if fp16 is not None:
            self.fp16 = fp16
    
