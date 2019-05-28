"""
Vector differentiator:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt


class LayerNormAVG(nn.Module):
    """
        Layer Normalization class inspired by Transformer normalization, but here we normalize to given average
        to preserve magnitue of USE
    """

    def __init__(self, features, desired_avg, eps=1e-6):
        super(LayerNormAVG, self).__init__()
        self.desiredAVG = desired_avg
        self.eps = eps
        self.size = features

    def forward(self, x):
        to_norm = torch.sqrt(self.desiredAVG * self.size / torch.sum(x ** 2))
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        ret = (x - mean) / (std + self.eps)
        return to_norm * ret

#cnn8
class VecdiffDecoder(nn.Module):

    def __init__(self, d_model, input_size, dropout=0):
        super(VecdiffDecoder, self).__init__()

        self.b_concat_modifier = 1 #=1 - just take source embedding,  =3 - contactencate only backwards rnn layer to B, if 4 - cvoncatenate both layers
        self.I_concat_modifier = 2

        # Basic attributes.
        self.decoder_type = 'vecdiff'
        self.hidden_size = d_model

        self.Winsrc = nn.Linear(input_size, d_model)
        self.Wfg = nn.Linear(input_size * self.I_concat_modifier + 2 , d_model)
        self.Wig = nn.Linear(input_size * self.I_concat_modifier + 2, d_model)

        self.Wogb = nn.Linear(input_size * self.b_concat_modifier, d_model)
        self.WogA = nn.Linear(input_size, d_model)
        self.WogD = nn.Linear(input_size, d_model)

        self.Wctb = nn.Linear(input_size* self.b_concat_modifier, d_model)
        self.WcteI = nn.Linear(input_size * self.I_concat_modifier, d_model)
        self.WcteII = nn.Linear(input_size, d_model)

        self.final = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

        self.input_size = input_size
        self.d_model = d_model
        self.lnorm = LayerNormAVG(d_model, 0.002)

    def forward(self, i, src, enc_memory, prev_prediction, final_encoder_h, source_vector=None):
        """
        i - current position in src
        src - input source vectors
        enc_memory - encoder memory - both forward and backward
        prev_prediction - previous prediction of current model
        """
        if prev_prediction is None:
            if src[0,i].is_cuda:
                prev_prediction = torch.nn.Tanh()(self.Winsrc(torch.nn.Tanh()(source_vector).cuda() ))
            else:
                prev_prediction = torch.nn.Tanh()(self.Winsrc(torch.nn.Tanh()(source_vector)))
            prev_prediction = self.lnorm(prev_prediction)
        A = prev_prediction
        if not A.is_cuda and src[0,i].is_cuda:
            A = A.cuda()

        B = src[:, i] #[0

        D = A - B

        prev_cur_angle = F.cosine_similarity(A, B, dim=(len(A.size()) - 1))
        prev_total_angle = F.cosine_similarity(A, source_vector, dim=(len(A.size()) - 1))

        I = torch.cat(( enc_memory[i, :, :], prev_cur_angle.unsqueeze(1) , prev_total_angle.unsqueeze(1) ),1) #enc_memory[i][0][0:self.input_size] # check src_size x batch x 1024

        fg = torch.sigmoid( self.Wfg(I))
        ig = torch.sigmoid(self.Wig(I))

        og = torch.sigmoid( ( self.Wogb(B) + self.WogA(A) )) #*delta
        # print("og " + str(og[0:5]))
        ct = fg * A + ig * torch.nn.Tanh()(self.Wctb(B) )
        ht = og * torch.sigmoid(ct)
        score = torch.sigmoid(self.final(ht))
        #print("score " + str(score.item()))
        out = A - (D * score)
        return out, score
