This project ports our original code in Theano to Tensorflow. Still under development.

# Introduction

Implementation of Neural Language Correction (http://arxiv.org/abs/1603.09727) on Tensorflow.
But this is a chinese version!!!

# Training

To train character level model (for chinese, we'd better only use this mode <chinese character>):

    $ python train.py --data_dir /dir/to/train_valid/data (train.x.txt&train.y.txt / valid.x.txt&valid.y.txt) --train_dir /dir/to/save/train_data

**NOTICE:** "x" means orignal error sentences, "y" means revised version sentences

# Interactive Decoding

    $ python decode.py --data_dir /dir/to/vocab_test/data (vocab.dat&test.x.txt) --train_dir /dir/to/save/train_data

# Tensorflow Dependency

- Tensorflow 1.1.0

# Other implementations

- Chainer implementation by @sotetsuk: [https://github.com/sotetsuk/neural-language-correction](https://github.com/sotetsuk/neural-language-correction)
