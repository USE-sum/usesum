#!/usr/bin/env python
"""
    Training on a single process
"""
from __future__ import division

import argparse
import os
import random
import torch

import onmt.opts as opts

from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields, _collect_report_features, load_fields_from_vocab


def calculate_missing(stoi1, stoi2):
    missing = 0
    shared = 0
    for src2 in stoi2:
        # print(src2)
        if src2 not in stoi1:
            print("MISSING " + src2)
            missing += 1
        else:
            shared += 1
    print("missing: " + str(missing) + "   shared = " + str(shared))

def main(opt):

    vocab1 = load_fields_from_vocab(
        torch.load(opt.data + '.vocab.pt'), "text")

    vocab2 = load_fields_from_vocab(
        torch.load(opt.data2 + '.vocab.pt'), "text")
    calculate_missing(vocab1["src"].vocab.stoi, vocab2["src"].vocab.stoi)
    calculate_missing(vocab1["tgt"].vocab.stoi, vocab2["tgt"].vocab.stoi)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='compare_vocabs.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.vocab_compare_opts(parser)

    opt = parser.parse_args()
    main(opt)
