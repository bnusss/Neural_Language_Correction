# -*- coding: utf-8 -*-
# Copyright 2016 Stanford University
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

import nlc_data

import random
import pinyin
import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


def pair_iter(fnamex, fnamey, batch_size, num_layers, sort_and_shuffle=True):
    fdx, fdy = open(fnamex), open(fnamey)
    batches = []

    while True:
        if len(batches) == 0:
            fill_batch(batches, fdx, fdy, batch_size, sort_and_shuffle=sort_and_shuffle)
        if len(batches) == 0:
            break

        x_tokens, y_tokens = batches.pop(0)
        y_tokens = add_sos_eos(y_tokens)
        
        x_padded, y_padded = padded(x_tokens, num_layers), padded(y_tokens, 1)
        
        source_tokens = np.array(x_padded).T
        source_mask = (source_tokens != nlc_data.PAD_ID).astype(np.int32)
        
        target_tokens = np.array(y_padded).T
        target_mask = (target_tokens != nlc_data.PAD_ID).astype(np.int32)

        yield (source_tokens, source_mask, target_tokens, target_mask)


def fill_batch(batches, fdx, fdy, batch_size, sort_and_shuffle=True):

    def tokenize(string):
        return [int(s) for s in string.split()]

    line_pairs = []
    linex, liney = fdx.readline(), fdy.readline()

    while linex and liney:
        x_tokens, y_tokens = tokenize(linex), tokenize(liney)

        if len(x_tokens) < FLAGS.max_seq_len and len(y_tokens) < FLAGS.max_seq_len:
            line_pairs.append((x_tokens, y_tokens))
        if len(line_pairs) == batch_size * 16:
            break
        linex, liney = fdx.readline(), fdy.readline()

    if sort_and_shuffle:
        line_pairs = sorted(line_pairs, key=lambda e: len(e[0]))

    for batch_start in xrange(0, len(line_pairs), batch_size):
        x_batch, y_batch = list(zip(*line_pairs[batch_start:batch_start+batch_size]))
        batches.append((x_batch, y_batch))

    if sort_and_shuffle:
        random.shuffle(batches)


def add_sos_eos(tokens):
    return list(map(lambda token_list: [nlc_data.SOS_ID] + token_list + [nlc_data.EOS_ID], tokens))


def padded(tokens, depth):
    maxlen = max(list(map(lambda x: len(x), tokens)))
    align = pow(2, depth - 1)
    padlen = maxlen + (align - maxlen) % align
    return list(map(lambda token_list: token_list + [nlc_data.PAD_ID] * (padlen - len(token_list)), tokens))


def get_tokenizer(flags):
    if flags.tokenizer.lower() == 'char':
        return nlc_data.char_tokenizer
    elif flags.tokenizer.lower() == 'word':
        return nlc_data.basic_tokenizer
    else:
        raise
    return tokenizer


def getPinYin(s):
    return pinyin.get(s, format='numerical', delimiter=' ')


def levenshtein(s1, s2):
    '''编辑距离'''
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def levenshtein2(s1, s2):
    if len(s1) == 0:
        return len(s2)
    elif len(s2) == 0:
        return len(s1)
    elif len(s1) == 0 and len(s2) == 0:
        return 0
    else:
        return min(levenshtein2(s1[:-1], s2)+1,
                   levenshtein2(s1, s2[:-1])+1,
                   levenshtein2(s1[:-1], s2[:-1])+(1 if s1[-1] != s2[-1] else 0))


def levenshtein3(s1, s2):
    row, col = len(s1)+1, len(s2)+1
    state_matrix = [[0]*col for i in range(row)] # row*col zero matrix
    # initial left&top border
    for i in range(row):
        state_matrix[i][0] = i
    for j in range(col):
        state_matrix[0][j] = j
    # begin match
    cost = 0
    for i in range(1, row):
        s1_ = s1[i-1]
        for j in range(1, col):
            s2_ = s2[j-1]
            cost = (1 if s1_ != s2_ else 0)
            state_matrix[i][j] = min(state_matrix[i-1][j]+1,
                                     state_matrix[i][j-1]+1, 
                                     state_matrix[i-1][j-1]+cost)

    return state_matrix[row-1][col-1]


def evaluate(sentx, senty):
    '''评估'''
    sent_max_len   = max(len(list(sentx)), len(list(senty)))
    sentx_pinyin = getPinYin(sentx)
    senty_pinyin = getPinYin(senty)
    pinyin_max_len = max(len(sentx_pinyin), len(senty_pinyin))

    pinyin_distance = levenshtein(sentx_pinyin, senty_pinyin)
    sent_distance   = levenshtein(sentx, senty)

    diff_ratio = 0.2*pinyin_distance/pinyin_max_len + 0.8*sent_distance/sent_max_len

    return diff_ratio


def strQ2B(ustring):
    '''全角转半角'''
    rstring = ''
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += unichr(inside_code)
    return rstring

def strB2Q(ustring):
    '''半角转全角'''
    rstring = ''
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 32:                                 #半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:        #半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += unichr(inside_code)
    return rstring
