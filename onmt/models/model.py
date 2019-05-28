""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder, multigpu=False, use_src_directly_for_dec=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_src_directly_for_dec = use_src_directly_for_dec

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        if len(tgt.size())==3:
            #if tgt.size()[2]>1:
            tgt = tgt[:-1]  # exclude last target from inputs
        elif len(tgt.size())==2:
            tgt = torch.unsqueeze(tgt, 0)
        if self.encoder is not None:
            enc_final, memory_bank, lengths = self.encoder(src, lengths)
            mem_sizes = memory_bank.size()
            enc_state = \
                self.decoder.init_decoder_state(src, memory_bank, enc_final)
            if self.use_src_directly_for_dec: # and :
                if len(src.size())<3:
                    enc_state.input_feed = torch.unsqueeze(src,0)
                enc_state.input_feed = torch.unsqueeze(enc_final[0][-1,:,:],0) #last hidden layer of top bi-lstm
                #memory_bank = torch.transpose(memory_bank,0,1)

            decoder_outputs, dec_state, attns = \
                self.decoder(tgt, memory_bank,
                             enc_state if dec_state is None
                             else dec_state,
                             memory_lengths=lengths)
        else:
            # enc_state = \
            #     self.decoder.init_decoder_state(src, None, None)
            decoder_outputs, dec_state, attns = \
                self.decoder(src, memory_lengths=lengths)

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state