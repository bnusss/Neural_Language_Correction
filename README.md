This project ports our original code in Theano to Tensorflow. Still under development.

# Introduction

Implementation of Neural Language Correction (http://arxiv.org/abs/1603.09727) on Tensorflow.

# Training

To train character level model (default):


    $ python train.py --data_dir /dir/to/train/and/valid/data (train.x.txt&train.y.txt / valid.x.txt&valid.y.txt) --train_dir /dir/to/save/train_data

To train word level model:

    $ python train.py --tokenizer WORD --data_dir /dir/to/train/and/valid/data (train.x.txt&train.y.txt / valid.x.txt&valid.y.txt)

# Interactive Decoding

    $ python decode.py --data_dir /dir/to/train/and/valid/data (train.x.txt&train.y.txt / valid.x.txt&valid.y.txt) --train_dir /dir/to/save/train_data

# Tensorflow Dependency

- Tensorflow 1.1.0

# Other implementations

- Chainer implementation by @sotetsuk: [https://github.com/sotetsuk/neural-language-correction](https://github.com/sotetsuk/neural-language-correction)
