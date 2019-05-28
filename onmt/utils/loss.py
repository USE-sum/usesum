"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
import onmt.inputters as inputters
from onmt.modules.sparse_losses import SparsemaxLoss
from math import isnan


def build_loss_compute(model, tgt_vocab, opt, train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength, focal_gamma=opt.focal_gamma)
    elif opt.model_type=="vector":
        sequential_target = False
        if opt.decoder_type=="vecdif_multi":
            sequential_target=True
        compute = AcosLoss(model.generator, tgt_vocab, model.decoder.hidden_size, device, sequential_target=sequential_target) #model.generator
    else:
        compute = NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing if train else 0.0)
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[inputters.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        to_compare = batch.src[0, :1, :]
        shard_state["to_compare"] = to_compare
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def monolithic_compute_loss_multivec(self, batch, output):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        stats = None
        i = 0
        for o in output:
            range_ = (i, i+1)
            shard_state = self._make_shard_state(batch, o, range_, None)
            to_compare = batch.src[:, i, :] # to compare makes no point in validation.
            shard_state["to_compare"] = to_compare
            _, batch_stats = self._compute_loss(batch, **shard_state)
            if stats is None:
                stats = batch_stats
            else:
                stats.update(batch_stats)
            i+=1

        return stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization, to_compare=None):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number
          to_compare (vector) - sources used for current prediction - used only in vecdiff

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = onmt.utils.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns)
        for shard in shards(shard_state, shard_size):
            if to_compare is not None:
                shard["to_compare"]=to_compare
            loss, stats = self._compute_loss(batch, **shard)
            #try:
            loss.div(float(normalization)).backward()
            # except Exception as e:
            #     print("PROBLEM "+str(e))
            batch_stats.update(stats)
        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def _stats_vec(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        # equal = scores.eq(target).sum().item()
        # pred = scores.max(1)[1]
        # non_padding = target.ne(self.padding_idx)
        # num_correct = pred.eq(target) \
        #                   .masked_select(non_padding) \
        #                   .sum() \
        #                   .item()
        # num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), 1 ,1 ) # equal, target.size()[1])

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class AcosLoss(LossComputeBase):
    """
    arcus cosine loss
    """
    def __init__(self, generator, tgt_vocab, output_size, device, sequential_target=False):
        super(AcosLoss, self).__init__(generator, tgt_vocab)
        self.zero_vec = torch.zeros(1,output_size, device=device)
        self.filled_vec = torch.zeros(1, output_size, device=device).fill_(0.0001)
        #self.prev_vec = torch.zeros(1,output_size, device=device)
        self.prev_distance = None # torch.zeros(1, 1, device=device)
        self.sequential_target=sequential_target
        self.lrelu = nn.LeakyReLU(0.01)


    def _compute_loss(self, batch, output, target, to_compare):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        if self.generator is not None:
            output = torch.squeeze(output, dim=0)
            output = self.generator(output)
        while len(output.size()) < len(target.size()):
            output = output.unsqueeze(0)

        v1 = F.cosine_similarity(output, target, dim=(len(target.size())-1) ) #torch.abs()

        v2 = torch.acos(v1)
        vstat = v2.clone()

        if self.prev_distance is None:
            self.prev_distance = torch.ones_like(v2) *1.5

        if v2.size()[0]> self.prev_distance.size()[0]: # in such case,
            v2 = v2[:self.prev_distance.size()[0]]
        elif v2.size()[0]< self.prev_distance.size()[0]: # in such case,
            self.prev_distance = self.prev_distance[:v2.size()[0]]

        v3 = v2 - self.prev_distance[:v2.size()[0]] # v2/10 + F.relu remove relu ?
        if self.sequential_target:
            optimal_improvement = torch.abs(F.cosine_similarity(to_compare, target, dim=(len(target.size()) - 1)))
            optimal_improvement = torch.acos(optimal_improvement)
            if v2.size()[0] > optimal_improvement.size()[0]:  # in such case,
                v2 = v2[:optimal_improvement.size()[0]]
            elif v2.size()[0] < optimal_improvement.size()[0]:  # in such case,
                optimal_improvement = optimal_improvement[:v2.size()[0]]
            if v2.size()[0] != optimal_improvement.size()[0]:
                print("v2 "+str(v2.size))
                print("optimal_improvement " + str(optimal_improvement.size))
            v3a = v2 - optimal_improvement
            v4 = v3a + F.relu(v3)
        else:
            v4 = v3
        #print(str(v2)+" \n v3="+str(v3)+" \n v3a="+str(v3a)+"   \n v4="+str(v4)+"\n sum= "+str(v4.sum())+" \n\n" )
        self.prev_distance = v2.detach()

        #print("targe " + str(target[0,0:5]) + "   outout= " + str(output[0,0:5]) + " loss = " + str(v2.item())+"  final loss = "+str(v3))
        stats = self._stats_vec(vstat.sum()/vstat.size()[0], output, target)
        return v4.sum(), stats

    def _make_shard_state(self, batch, output, range_, attns=None):
        if self.sequential_target:
            return {
                "output": output,
                "target": batch.tgt[:,range_[0]: range_[1],:].squeeze(1),
            }
        return {
            "output": output,
            "target": batch.tgt[range_[0]: range_[1]],
        }

class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        self.vector = not isinstance(generator[1], nn.Sigmoid)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, len(tgt_vocab), ignore_index=self.padding_idx
            )
        elif self.sparse:
            self.criterion = SparsemaxLoss(
                ignore_index=self.padding_idx, size_average=False
            )
        elif self.vector:
            self.criterion = SparsemaxLoss(
                ignore_index=self.padding_idx, size_average=False
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, output, target):
        bottled_output = self._bottle(output)
        if self.sparse:
            # for sparsemax loss, the loss function operates on the raw output
            # vector, not a probability vector. Hence it's only necessary to
            # apply the first part of the generator here.
            scores = self.generator[0](bottled_output)
        else:
            scores = self.generator(bottled_output)
        gtruth = target.view(-1)

        loss = self.criterion(scores, gtruth)
        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
