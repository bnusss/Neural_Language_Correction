# -*- coding:utf-8 -*-

import codecs
import pickle
from util import *
from decode import *


load_vocab()
load_model()

revise_ratios  = []
correct_ratios = []
error_ratios   = []
with codecs.open(FLAGS.data_dir+'/'+FLAGS.tokenizer.lower()+'/valid.x.txt', encoding='utf-8') as fr1:
    with codecs.open(FLAGS.data_dir+'/'+FLAGS.tokenizer.lower()+'/valid.y.txt', encoding='utf-8') as fr2:
        for sent, sentt in zip(fr1, fr2):
            correct_sent = sentt.strip()
            original_sent = sent.strip()
            revised_sent = decode(original_sent)

            error_ratio  = evaluate(original_sent, correct_sent)
            error_ratios.append(error_ratio)
            revise_ratio = evaluate(original_sent, revised_sent)
            revise_ratios.append(revise_ratio)
            correct_ratio = evaluate(revised_sent, correct_sent)
            correct_ratios.append(correct_ratio)

corpus_error_ratios = []
with codecs.open('./data_dir/data/char/train.x.txt', encoding='utf-8') as fr1:
    with codecs.open('./data_dir/data/char/train.y.txt', encoding='utf-8') as fr2:
        for sent1, sent2 in zip(fr1, fr2):
	    error_ratio = evaluate(sent1, sent2)
	    corpus_error_ratios.append(error_ratio)

try:
    fp = open('ratios.pkl', 'wb')
    pickle.dump([corpus_error_ratios, error_ratios, correct_ratios, revise_ratios], fp)
    fp.close()
except Exception as e:
    raise e
finally:
    print("average corpus error ratio: {}".format(sum(corpus_error_ratios)*1.0/len(corpus_error_ratios)))
    print("average error ratio: {}".format(sum(error_ratios)*1.0/len(error_ratios)))
    print("average correct ratio: {}".format(sum(correct_ratios)*1.0/len(correct_ratios)))
    print("average revise ratio: {}".format(sum(revise_ratios)*1.0/len(revise_ratios)))
