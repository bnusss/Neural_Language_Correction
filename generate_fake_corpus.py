#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import pinyin
import codecs
import cPickle as pickle
from random import random
from collections import Counter
from collections import defaultdict


zh_char_p = re.compile(ur'[\u4e00-\u9fa5]')
pinyin_p   = re.compile(ur'ng$|n$|^n|^l')

def count_thing():
    proun_dict = defaultdict(set)
    cnt = Counter()

    fpath = '/home/gl/Documents/zh_tsinghua.nolowercase'
    with codecs.open(fpath, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr):
            if idx % 10000 == 0:
                print 'processed {} lines.'.format(idx)
            chars = zh_char_p.findall(line)
            pys = [pinyin.get(char, format='numerical') for char in chars]
            for py, char in zip(pys, chars):
                proun_dict[py].add(char)
            cnt.update(chars)

    # adjust the proun_dict by char freq
    proun_dict2 = {}
    for proun, chars in proun_dict.iteritems():
        chars = list(chars)
        chars = sorted(chars, key=lambda x: cnt[x], reverse=True)
        proun_dict2[proun] = chars


    print len(proun_dict2)
    with open('proun_dict.pkl', 'wb') as fw:
        pickle.dump(proun_dict2, fw)

    with open('char_count.pkl', 'wb') as fw:
        cnt = cnt.most_common()
        pickle.dump(cnt, fw)

# generate fake corpus
# random change pinyin or change to same proun char
def generate():
    proun_dict = None
    if not os.path.exists('proun_dict.pkl'):
        count_thing()
    proun_dict = pickle.load(open('proun_dict.pkl', 'rb'))

    p1 = 0.2  # prob to change to same proun char
    p2 = 0.02 # prob to change pinyin
    fpath = '/home/gl/Documents/zh_tsinghua.nolowercase'
    with codecs.open(fpath, 'r', encoding='utf-8') as fr:
        new_lines = []
        for idx, line in enumerate(fr):
            if idx % 10000 == 0:
                print 'processed {} lines.'.format(idx)
            new_chars = []
            for char in line:
                if zh_char_p.match(char):
                    p1_ = random()
                    p2_ = random()
                    py = pinyin.get(char, format='numerical')
                    if p1_ <= p1:
                        char = proun_dict[py][0] # we just change to the most freq char
                    elif p2_ <= p2:
                        if py.startswith('l'):
                            py = 'n' + py[1:]
                        elif py.startswith('n'):
                            py = 'l' + py[1:]
                        elif py.startswith('nen'):
                            py = 'neng' if py[-2] != 'g' else 'nen'
                        elif py[:-1].endswith('in'):
                            py = 'ing' if py[-2] != 'g' else 'in'
                        else:
                            pass
                        if proun_dict.has_key(py):
                            char = proun_dict[py][0] # we just change to the most freq char
                new_chars.append(char)
            new_line = ''.join(new_chars)
            new_lines.append(new_line.encode('utf-8'))
        with open('/home/gl/Documents/zh_tsinghua.nolowercase.fake', 'w') as fw:
            fw.writelines(new_lines)

if __name__ == '__main__':
    generate()
