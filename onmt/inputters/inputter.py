# -*- coding: utf-8 -*-
"""
    Defining general functions for inputters
"""
import glob
import os

from collections import Counter, defaultdict, OrderedDict
from itertools import count
import json
import requests

from nltk.tokenize import sent_tokenize
import numpy as np
import torch
import torchtext.data
import torchtext.vocab

from onmt.inputters.dataset_base import UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD
from onmt.inputters.text_dataset import TextDataset
from onmt.inputters.image_dataset import ImageDataset
from onmt.inputters.audio_dataset import AudioDataset
from onmt.inputters.vector_dataset import VectorDataset
from onmt.utils.logging import logger
import re

import gc


def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = _getstate
torchtext.vocab.Vocab.__setstate__ = _setstate


def get_fields(data_type, n_src_features, n_tgt_features):
    """
    Args:
        data_type: type of the source input. Options are [text|img|audio].
        n_src_features: the number of source features to
            create `torchtext.data.Field` for.
        n_tgt_features: the number of target features to
            create `torchtext.data.Field` for.

    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """
    if data_type == 'text':
        return TextDataset.get_fields(n_src_features, n_tgt_features)
    elif data_type == 'img':
        return ImageDataset.get_fields(n_src_features, n_tgt_features)
    elif data_type == 'audio':
        return AudioDataset.get_fields(n_src_features, n_tgt_features)
    elif data_type == 'vector':
        return VectorDataset.get_fields(n_src_features, n_tgt_features)
    else:
        raise ValueError("Data type not implemented")


def load_fields_from_vocab(vocab, data_type="text"):
    """
    Load Field objects from `vocab.pt` file.
    """
    vocab = dict(vocab)
    n_src_features = len(collect_features(vocab, 'src'))
    n_tgt_features = len(collect_features(vocab, 'tgt'))
    fields = get_fields(data_type, n_src_features, n_tgt_features)
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields


def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            f.vocab.stoi = f.vocab.stoi
            vocab.append((k, f.vocab))
    return vocab


def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                 specials=[UNK_WORD, PAD_WORD,
                                           BOS_WORD, EOS_WORD],
                                 max_size=vocab_size)


def get_num_features(data_type, corpus_file, side):
    """
    Args:
        data_type (str): type of the source input.
            Options are [text|img|audio].
        corpus_file (str): file path to get the features.
        side (str): for source or for target.

    Returns:
        number of features on `side`.
    """
    assert side in ["src", "tgt"]

    if data_type == 'text':
        return TextDataset.get_num_features(corpus_file, side)
    elif data_type == 'img':
        return ImageDataset.get_num_features(corpus_file, side)
    elif data_type == 'audio':
        return AudioDataset.get_num_features(corpus_file, side)
    elif data_type == 'vector':
        return VectorDataset.get_num_features(corpus_file, side)
    else:
        raise ValueError("Data type not implemented")


def make_features(batch, side, data_type='text'):
    """
    Args:
        batch (Tensor): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input.
            Options are [text|img|audio].
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    assert side in ['src', 'tgt']
    if isinstance(batch.__dict__[side], tuple):
        data = batch.__dict__[side][0]
    else:
        data = batch.__dict__[side]

    feat_start = side + "_feat_"
    keys = sorted([k for k in batch.__dict__ if feat_start in k])
    features = [batch.__dict__[k] for k in keys]
    levels = [data] + features

    if data_type == 'text' : #or data_type == 'vector'
        return torch.cat([level.unsqueeze(2) for level in levels], 2)
    else:
        return levels[0]


def collect_features(fields, side="src"):
    """
    Collect features from Field object.
    """
    assert side in ["src", "tgt"]
    feats = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feats.append(key)
    return feats


def collect_feature_vocabs(fields, side):
    """
    Collect feature Vocab objects from Field object.
    """
    assert side in ['src', 'tgt']
    feature_vocabs = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feature_vocabs.append(fields[key].vocab)
    return feature_vocabs


def  build_dataset(fields, data_type, src_data_iter=None, src_path=None,
                  src_dir=None, tgt_data_iter=None, tgt_path=None,
                  src_seq_length=0, tgt_seq_length=0,
                  src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                  dynamic_dict=True, sample_rate=0,
                  window_size=0, window_stride=0, window=None,
                  normalize_audio=True, use_filter_pred=True,
                  image_channel_size=3):
    """
    Build src/tgt examples iterator from corpus files, also extract
    number of features.
    """

    def _make_examples_nfeats_tpl(data_type, src_data_iter, src_path, src_dir,
                                  src_seq_length_trunc, sample_rate,
                                  window_size, window_stride,
                                  window, normalize_audio,
                                  image_channel_size=3):
        """
        Process the corpus into (example_dict iterator, num_feats) tuple
        on source side for different 'data_type'.
        """

        if data_type == 'text':
            src_examples_iter, num_src_feats = \
                TextDataset.make_text_examples_nfeats_tpl(
                    src_data_iter, src_path, src_seq_length_trunc, "src")

        elif data_type=="vector":
            src_examples_iter, num_src_feats = \
                VectorDataset.make_vectors_examples_nfeats_tpl(src_data_iter,
                    src_path, src_seq_length_trunc, "src")
            # src_examples_iter, num_src_feats = \
            #     TextDataset.make_text_examples_nfeats_tpl(
            #         src_data_iter, src_path, src_seq_length_trunc, "src")

        elif data_type == 'img':
            src_examples_iter, num_src_feats = \
                ImageDataset.make_image_examples_nfeats_tpl(
                    src_data_iter, src_path, src_dir, image_channel_size)

        elif data_type == 'audio':
            if src_data_iter:
                raise ValueError("""Data iterator for AudioDataset isn't
                                    implemented""")

            if src_path is None:
                raise ValueError("AudioDataset requires a non None path")
            src_examples_iter, num_src_feats = \
                AudioDataset.make_audio_examples_nfeats_tpl(
                    src_path, src_dir, sample_rate,
                    window_size, window_stride, window,
                    normalize_audio)

        return src_examples_iter, num_src_feats

    src_examples_iter, num_src_feats = \
        _make_examples_nfeats_tpl(data_type, src_data_iter, src_path, src_dir,
                                  src_seq_length_trunc, sample_rate,
                                  window_size, window_stride,
                                  window, normalize_audio,
                                  image_channel_size=image_channel_size)

    if data_type != 'vector':
        # For all data types, the tgt side corpus is in form of text.
        tgt_examples_iter, num_tgt_feats = \
            TextDataset.make_text_examples_nfeats_tpl(
                tgt_data_iter, tgt_path, tgt_seq_length_trunc, "tgt")
    else:
        tgt_examples_iter, num_tgt_feats = \
            VectorDataset.make_vectors_examples_nfeats_tpl(
                tgt_data_iter, tgt_path, tgt_seq_length_trunc, "tgt")

    if data_type == 'text':
        dataset = TextDataset(fields, src_examples_iter, tgt_examples_iter,
                              num_src_feats, num_tgt_feats,
                              src_seq_length=src_seq_length,
                              tgt_seq_length=tgt_seq_length,
                              dynamic_dict=dynamic_dict,
                              use_filter_pred=use_filter_pred)

    elif data_type == 'img':
        dataset = ImageDataset(fields, src_examples_iter, tgt_examples_iter,
                               num_src_feats, num_tgt_feats,
                               tgt_seq_length=tgt_seq_length,
                               use_filter_pred=use_filter_pred,
                               image_channel_size=image_channel_size)

    elif data_type == 'vector':
        dataset = VectorDataset(fields, src_examples_iter, tgt_examples_iter,
                               num_src_feats, 0,
                               tgt_seq_length=tgt_seq_length)

    elif data_type == 'audio':
        dataset = AudioDataset(fields, src_examples_iter, tgt_examples_iter,
                               tgt_seq_length=tgt_seq_length,
                               use_filter_pred=use_filter_pred)

    return dataset


def _build_field_vocab(field, counter, vocab_extend=[], **kwargs):
    spec_list = [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]
    spec_list = spec_list + list(vocab_extend)
    specials = list(OrderedDict.fromkeys(
        tok for tok in spec_list
        if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_path, src_vocab_size, src_words_min_frequency,
                tgt_vocab_path, tgt_vocab_size, tgt_words_min_frequency,
                src_vocab_extend_path, tgt_vocab_extend_path):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
        data_type: "text", "img" or "audio"?
        share_vocab(bool): share source and target vocabulary?
        src_vocab_path(string): Path to src vocabulary file.
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_path(string): Path to tgt vocabulary file.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        Dict of Fields
    """
    counter = {}

    # Prop src from field to get lower memory using when training with image
    if data_type == 'img' or data_type == 'audio' :
        fields.pop("src")
    #if data_type == 'vector':
        #fields.pop("tgt")


    for k in fields:
        counter[k] = Counter()

    # Load vocabulary
    src_vocab = load_vocabulary(src_vocab_path, tag="source")
    tgt_vocab = load_vocabulary(tgt_vocab_path, tag="target")

    src_vocab_extend = load_vocabulary(src_vocab_extend_path, tag="source")
    if src_vocab_extend is None:
        src_vocab_extend = []
    tgt_vocab_extend = load_vocabulary(tgt_vocab_extend_path, tag="target")
    if tgt_vocab_extend is None:
        tgt_vocab_extend = []

    for index, path in enumerate(train_dataset_files):
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for ex in dataset.examples:
            for k in fields:
                val = getattr(ex, k, None)
                if not fields[k].sequential:
                    continue
                elif k == 'src':
                    if src_vocab:
                        val = [item for item in val if item in src_vocab]
                    elif len(src_vocab_extend)>0:
                        val = [item for item in val if item not in src_vocab_extend]
                elif k == 'tgt':
                    if tgt_vocab:
                        val = [item for item in val if item in tgt_vocab]
                    elif len(tgt_vocab_extend)>0:
                        val = [item for item in val if item not in tgt_vocab_extend]
                counter[k].update(val)

        # Drop the none-using from memory but keep the last
        if (index < len(train_dataset_files) - 1):
            dataset.examples = None
            gc.collect()
            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    _build_field_vocab(fields["tgt"], counter["tgt"],
                       vocab_extend=tgt_vocab_extend,
                       max_size=tgt_vocab_size,
                       min_freq=tgt_words_min_frequency)
    logger.info(" * tgt vocab size: %d." % len(fields["tgt"].vocab))

    # All datasets have same num of n_tgt_features,
    # getting the last one is OK.
    for j in range(dataset.n_tgt_feats):
        key = "tgt_feat_" + str(j)
        _build_field_vocab(fields[key], counter[key])
        logger.info(" * %s vocab size: %d." % (key,
                                               len(fields[key].vocab)))

    if data_type == 'text' or data_type == 'vector': #or data_type == 'vector'
        _build_field_vocab(fields["src"], counter["src"],
                           vocab_extend=src_vocab_extend,
                           max_size=src_vocab_size,
                           min_freq=src_words_min_frequency)
        logger.info(" * src vocab size: %d." % len(fields["src"].vocab))

        # All datasets have same num of n_src_features,
        # getting the last one is OK.
        for j in range(dataset.n_src_feats):
            key = "src_feat_" + str(j)
            _build_field_vocab(fields[key], counter[key])
            logger.info(" * %s vocab size: %d." %
                        (key, len(fields[key].vocab)))

        # Merge the input and output vocabularies.
        if share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            logger.info(" * merging src and tgt vocab...")
            merged_vocab = merge_vocabs(
                [fields["src"].vocab, fields["tgt"].vocab],
                vocab_size=src_vocab_size)
            fields["src"].vocab = merged_vocab
            fields["tgt"].vocab = merged_vocab

    return fields

def build_vocab_vec(fields):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
        data_type: "text", "img" or "audio"?

    Returns:
        Dict of Fields
    """

    counter = {fields["tgt"].unk_token:1, fields["tgt"].pad_token:1, fields["tgt"].init_token:1,
                 fields["tgt"].eos_token:1}
    _build_field_vocab(fields["tgt"], counter)
    logger.info(" * tgt vocab size: %d." % len(fields["tgt"].vocab))

    counter = {fields["src"].unk_token: 1, fields["src"].pad_token: 1, fields["src"].init_token: 1,
               fields["src"].eos_token: 1}
    _build_field_vocab(fields["src"], counter)
    logger.info(" * src vocab size: %d." % len(fields["src"].vocab))

    return fields


def load_vocabulary(vocabulary_path, tag=""):
    """
    Loads a vocabulary from the given path.
    :param vocabulary_path: path to load vocabulary from
    :param tag: tag for vocabulary (only used for logging)
    :return: vocabulary or None if path is null
    """
    vocabulary = None
    if vocabulary_path:
        vocabulary = set([])
        logger.info("Loading {} vocabulary from {}".format(tag,
                                                           vocabulary_path))

        if not os.path.exists(vocabulary_path):
            raise RuntimeError(
                "{} vocabulary not found at {}!".format(tag, vocabulary_path))
        else:
            with open(vocabulary_path) as f:
                for line in f:
                    if len(line.strip()) == 0:
                        continue
                    word = line.strip().split()[0]
                    vocabulary.add(word)
    return vocabulary


def _pad_batches(batches, eos_np=None ):
    max_len_src = 0
    max_len_tgt = 0
    for b in batches:  # sources are sorted but target not necessarily
        if not isinstance(b.tgt, tuple) and b.tgt.shape[0] > max_len_tgt :
                max_len_tgt = b.tgt.shape[0]
        if not isinstance(b.src, tuple)  and b.src.shape[0] > max_len_src:
                max_len_src = b.src.shape[0]
    #print("will equalize src to "+str(max_len_src) +" and tgt to "+str(max_len_tgt)+"   "+str(lb.tgt.shape)+"   "+str(lb.tgt))
    for b in batches:
        if eos_np is not None:
            b.eos_np = eos_np
        if max_len_src > 0:
            cur_len = b.src.shape[0]
            if eos_np is None:
                eos_np = np.expand_dims(b.src[len(b.src)-1], axis=0)
            to_add = np.repeat(eos_np, max_len_src - cur_len, axis=0)
            new_src = np.append(b.src, to_add, axis=0)
            b.src = new_src
        if max_len_tgt>0:
            cur_len = b.tgt.shape[0]
            if eos_np is None:
                eos_np = b.tgt[len(b.tgt)-1]
            if max_len_tgt - cur_len>0:
                to_add = np.repeat(eos_np, max_len_tgt - cur_len, axis=0)
                new_tgt = np.append(b.tgt, to_add, axis=0)
                b.tgt = new_tgt

class OrderedIterator(torchtext.data.Iterator):
    """ Ordered Iterator Class """

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=False, shuffle=None, sort=None,
                 sort_within_batch=None, equalize_sizes=False):
        super(OrderedIterator,self).__init__(dataset, batch_size, sort_key, device,
                 batch_size_fn, train,
                 repeat, shuffle, sort, sort_within_batch)
        self.equalize_sizes = equalize_sizes

    def create_batches(self):
        """ Create batches """
        if self.train:
            def _pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)

                    for b in random_shuffler(list(p_batch) ):
                        if self.equalize_sizes:
                            for bb in b:
                                if not isinstance(bb.src, (np.ndarray, np.generic)):
                                    bb.src = np.array(bb.src)
                                if not isinstance(bb.tgt, (np.ndarray, np.generic)):
                                    bb.tgt = np.array(bb.tgt)
                            _pad_batches(b)
                        yield b

            self.batches = _pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                if self.equalize_sizes:
                    for bb in b:
                        if not isinstance(bb.src, (np.ndarray, np.generic)):
                            bb.src = np.array(bb.src)
                        if not isinstance(bb.tgt, (np.ndarray, np.generic)):
                            bb.tgt = np.array(bb.tgt)
                    _pad_batches(b)
                self.batches.append(sorted(b, key=self.sort_key))

class USEIterator(torchtext.data.Iterator):
    """ Ordered Iterator Class """

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=False, shuffle=None, sort=None,
                 sort_within_batch=None, use_port="", force_target_split=False ):
        super(USEIterator,self).__init__(dataset, batch_size, sort_key, device,
                 batch_size_fn, train,
                 repeat, shuffle, sort, sort_within_batch)
        self.use_port = use_port
        self.force_target_split = force_target_split
        self.eos_seq= None
        self.eos_np = None
        self.sos_np = None
        self.memory = {}
        if force_target_split:
            EOS_SEQUENCE="end of sequence"
            headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
            url = 'http://localhost:' + self.use_port + "/v1/models/universal_encoder:predict"
            data = {"instances": [EOS_SEQUENCE]}
            jsonized = json.dumps(data)
            r = requests.post(url, data=jsonized, headers=headers)
            ret_list = json.loads(r.text)["predictions"]
            self.eos_seq = ret_list[0]
            self.eos_np = np.expand_dims(np.array(ret_list[0]), axis=0)

    def use_vectorize(self, txt, tokenize=True, add_padd_end=False):
        txt = re.sub(r"\.\s*\.", ".", re.sub(r"^.{0,30}-lrb-[\w\W\s]{1,15}-rrb-", "",
                                             txt.rstrip().replace("</t> <t>", ".").rstrip().replace("<t>",
                                                                                                         "").replace(
                                                        "</t>", "")))

        if tokenize or self.force_target_split:
            sentences = sent_tokenize(txt)

        else:
            sentences = [txt]

        sentences = [re.sub(r"\.\s*\.", ".", re.sub(r"^.{0,30}-lrb-[\w\W\s]{1,15}-rrb-", "",
                                          x.rstrip().replace("</t> <t>", ".").rstrip().replace("<t>", "").replace(
                                              "</t>", ""))) for x in sentences]
        sentences = [x for x in sentences if len(x) > 0]
        joined = " ".join(sentences)
        if joined in self.memory:
            return self.memory[joined]
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        url = 'http://localhost:' + self.use_port + "/v1/models/universal_encoder:predict"

        data = {"instances": sentences}
        jsonized = json.dumps(data)
        r = requests.post(url, data=jsonized, headers=headers)
        ret_list = json.loads(r.text)["predictions"]
        if add_padd_end and self.eos_seq is not None:
            ret_list.append(self.eos_seq)
        src = np.array([ret_list])
        if len(self.memory)< 100000:
            self.memory[joined] = src
        return src

    def create_batches(self):
        """ Create batches """
        i=0
        for d in self.data():
            if isinstance(d.src, tuple):
                if len(d.src)>0 and isinstance(d.src[0], str):
                    d_str = " ".join(tup for tup in d.src)
                    d.src = d_str
                elif self.eos_np is not None:
                    d.src = d.src + (self.eos_np[0],)

            if isinstance(d.tgt, tuple):
                if len(d.tgt) > 0 and isinstance(d.tgt[0], str):
                    d_tgt = " ".join(tup for tup in d.tgt)
                    d.tgt= d_tgt
            if isinstance(d.src, str):
                # src_txt = " ".join(d.src)
                vectors = self.use_vectorize(d.src, True, True)
                d.src = vectors[0]

            if isinstance(d.tgt, str):
                # tgt_txt = " ".join(d.tgt)
                vectors = self.use_vectorize(d.tgt, False, True)
                d.tgt = vectors[0]
            i +=1


        if self.train:
            def _pool(data, random_shuffler):

                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch) ):

                        _pad_batches(b, self.eos_np)
                        yield b

            self.batches = _pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                if self.eos_np is not None:
                    _pad_batches(b, self.eos_np)
                self.batches.append(sorted(b, key=self.sort_key))


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train, use_port="", force_target_split=False, equalize_sizes=False):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train
        self.use_port = use_port
        self.force_target_split=force_target_split
        self.equalize_sizes= equalize_sizes

        self.cur_iter = self._next_dataset_iterator(datasets, force_target_split)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter, self.force_target_split)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter[0])

    def _next_dataset_iterator(self, dataset_iter, force_target_split=False):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset.examples = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        if self.use_port=="":
            return OrderedIterator(
                dataset=self.cur_dataset, batch_size=self.batch_size,
                batch_size_fn=self.batch_size_fn,
                device=self.device, train=self.is_train,
                sort=False, sort_within_batch=False,
                repeat=False, shuffle=False, equalize_sizes=self.equalize_sizes)
        else:
            return USEIterator(
                dataset=self.cur_dataset, batch_size=self.batch_size,
                batch_size_fn=self.batch_size_fn,
                device=self.device, train=self.is_train,
                sort=False, sort_within_batch=False,
                repeat=False, shuffle=False, use_port=self.use_port,force_target_split=force_target_split )


def build_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    if is_train and opt.batch_type == "tokens":
        def batch_size_fn(new, count, sofar):
            """
            In token batching scheme, the number of sequences is limited
            such that the total number of src/tgt tokens (including padding)
            in a batch <= batch_size
            """
            # Maintains the longest src and tgt length in the current batch
            global max_src_in_batch, max_tgt_in_batch
            # Reset current longest length at a new batch (count=1)
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            # Src: <bos> w1 ... wN <eos>
            max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
            # Tgt: w1 ... wN <eos>
            max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)
    else:
        batch_size_fn = None

    if opt.gpu_ranks:
        device = "cuda"
    else:
        device = "cpu"

    equalize_sizes = False
    if opt.decoder_type=="vecdif": #dont use batch > 1 for vecdif for now
        equalize_sizes=True
    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train, opt.use_port, force_target_split=opt.force_target_split,
                           equalize_sizes=equalize_sizes)


def lazily_load_dataset(corpus_type, opt):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def _load_fields(dataset, data_type, opt, checkpoint):
    if checkpoint is not None:
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = load_fields_from_vocab(
            checkpoint['vocab'], data_type)
    else:
        fields = load_fields_from_vocab(
            torch.load(opt.data + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    if data_type=="vector":
        fields["tgt"].postprocessing=None
    if data_type == 'text':
        if opt.use_port!="":
            for (k, f) in fields.items():
                if k =="src" or k == "tgt":
                    f.use_vocab=False
                    f.dtype = torch.float
                    f.sequential = False
                    f.include_lengths = False
        logger.info(' * vocabulary size. source = %d; target = %d' %
                    (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        logger.info(' * vocabulary size. target = %d' %
                    (len(fields['tgt'].vocab)))

    return fields


def _collect_report_features(fields):
    src_features = collect_features(fields, side='src')
    tgt_features = collect_features(fields, side='tgt')

    return src_features, tgt_features
