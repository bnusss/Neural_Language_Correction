# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import codecs
import jieba
from collections import defaultdict
from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "<pad>"
_SOS = "<sos>"
_EOS = "<eos>"
_UNK = "<unk>"
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_ZH_WORD_SPLIT = re.compile(u"[\u4e00-\u9fa5]")
_DIGIT_RE = re.compile(r"\d")


# word tokenizer: split the sentence into a list of tokens.
# sentence is `unicode object`
def word_tokenizer(sentence):
    return jieba.lcut(sentence)


# char tokenizer: split the sentence into a list of tokens.
# sentence is `unicode object`
def char_tokenizer(sentence):
    return list(sentence.strip())


# create vocabulary from train&dev data
def create_vocabulary(vocabulary_path, data_paths, max_vocabulary_size, tokenizer=None, normalize_digits=False):
    if not os.path.exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = defaultdict(int)
        for path in data_paths:
            with codecs.open(path, mode="r", encoding="utf-8") as fr:
                counter = 0
                for line in fr:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  processing line %d" % (counter,))
                    tokens = tokenizer(line) if tokenizer else word_tokenizer(line)
                    for w in tokens:
                        word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
                        vocab[word] += 1
                        
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with codecs.open(vocabulary_path, mode="w", encoding="utf-8") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with codecs.open(vocabulary_path, mode="r", encoding="utf-8") as fr:
            rev_vocab.extend(fr.readlines())
        rev_vocab = [line.strip("\n") for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=False):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = word_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
    if not os.path.exists(target_path):
        print("Tokenizing data in %s" % (data_path,))
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with codecs.open(data_path, mode="r", encoding="utf-8") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                for idx, line in enumerate(data_file):
                    if idx+1 % 100000 == 0:
                        print("  tokenizing line %d" % idx)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer, normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def get_data_path(data_dir, data_type="train"):
    data_path = os.path.join(data_dir, data_type)
    print("[%s] x data path is:%s" % (data_type.upper(), data_path + ".x.txt"))
    print("[%s] y data path is:%s" % (data_type.upper(), data_path + ".y.txt"))
    if os.path.exists(data_path + ".x.txt") and os.path.exists(data_path + ".y.txt"):
        return data_path
    else:
        print("Please provider [%s] x&y data" % (data_type.upper()))
        sys.exit()


def prepare_nlc_data(data_dir, max_vocabulary_size, tokenizer=char_tokenizer, other_dev_path=None):
    # Get nlc data to the specified directory.
    train_path = get_data_path(data_dir, "train")
    dev_path   = get_data_path(data_dir, "valid")
    
    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "vocab.dat")
    create_vocabulary(vocab_path, [train_path + ".y.txt", train_path + ".x.txt"], max_vocabulary_size, tokenizer)

    # Create token ids for the training data.
    y_train_ids_path = train_path + ".ids.y"
    x_train_ids_path = train_path + ".ids.x"
    data_to_token_ids(train_path + ".y.txt", y_train_ids_path, vocab_path, tokenizer)
    data_to_token_ids(train_path + ".x.txt", x_train_ids_path, vocab_path, tokenizer)

    # Create token ids for the development data.
    y_dev_ids_path = dev_path + ".ids.y"
    x_dev_ids_path = dev_path + ".ids.x"
    data_to_token_ids(dev_path + ".y.txt", y_dev_ids_path, vocab_path, tokenizer)
    data_to_token_ids(dev_path + ".x.txt", x_dev_ids_path, vocab_path, tokenizer)

    return (x_train_ids_path, y_train_ids_path, x_dev_ids_path, y_dev_ids_path, vocab_path)
