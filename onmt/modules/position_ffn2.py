"""
Position feed-forward network from "Attention is All You Need" + 2nd input for parametrizing updates
"""

import torch.nn as nn
import torch
import onmt


class PositionwiseFeedForward2(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward2, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_11 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = onmt.modules.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, query):
        """
        Layer definition.

        Args:
            input: [ batch_size, input_len, model_dim ]


        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.layer_norm( torch.mm(self.w_11(query), self.w_1( x.transpose(0,1) )) )
        output = self.dropout_2(self.w_2(inter))
        return output + x
