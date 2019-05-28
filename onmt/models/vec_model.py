""" Onmt NMT Model base class definition """
import json
import requests

from nltk.tokenize import sent_tokenize
import numpy as np
import torch.nn as nn
import torch

class VecModel(nn.Module):

    def __init__(self, encoder, decoder, multigpu=False, use_src_directly_for_dec=False):
        self.multigpu = multigpu
        super(VecModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_src_directly_for_dec = use_src_directly_for_dec

    def forward(self, i, src, prev_prediction, covered_prediction=None, source_vector=None, target_id=0):

        enc_final, memory_bank, lengths = self.encoder(src, None)
        if i >=0:
            if covered_prediction is None :
                decoder_outputs, _ = \
                    self.decoder(i, src, memory_bank, prev_prediction, enc_final[0], source_vector)
            else:
                decoder_outputs, _ = \
                    self.decoder(i, src, memory_bank, prev_prediction, enc_final[0], covered_prediction, source_vector, target_id)
            return decoder_outputs
        else:
            src_sizes = src.size()
            prev_prediction = None
            scores = []
            for i in range(src_sizes[1]):
                if covered_prediction is None :
                    decoder_outputs, score = \
                        self.decoder(i, src, memory_bank, prev_prediction,enc_final[0], source_vector)
                else:
                    decoder_outputs, score = \
                        self.decoder(i, src, memory_bank, prev_prediction, enc_final[0], covered_prediction,
                                     source_vector, target_id)
                    covered_prediction += decoder_outputs.detach()
                prev_prediction = decoder_outputs
                scores.append(score)
            return prev_prediction, scores, covered_prediction


