#!/usr/bin/python
# ---------------------------------------
# File Name : main.py
# Creation Date : 22-01-2018
# Last Modified : Mon Jan 22 11:11:11 2018
# Created By : wdd
# ---------------------------------------
# -*- coding: utf-8 -*-
import argparse
import sys
import os
import random
import torch
import numpy as np
from glo import Option, Global, get_logger, experiment, VarDict
from util import fit, ModelCheckpoint, save, save_model, load, load_model
from data import Data, conll_reader, text_reader, write_conll, NaryTree

try:
    import cPickle as pickle
except:
    import pickle

_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument('--task', default='permute', type=str, action='store',
                         help='The tasks:\n train: Train the model')
_arg_parser.add_argument('--src', default='', type=str, action='store',
                         help='The file of the source language (.conllu format)')
_arg_parser.add_argument('--tgt', default='', type=str, action='store',
                         help='The file of the target language (.conllu format)')
_arg_parser.add_argument('--output', default='', type=str, action='store', help='The output file')
_arg_parser.add_argument('--model', default='', type=str, action='store', help='Path for the model file')
_arg_parser.add_argument('--init', default='zero', type=str, action='store',
                         help='Initialization for permutation parameters')
_arg_parser.add_argument('--pretrain', default='', type=str, action='store', help='Path for pretrained model')
_arg_parser.add_argument('--lm_smoother', default='add0.1', type=str, action='store', help='')
_arg_parser.add_argument('--feature', default='A', type=str, action='store',
                         help='The feature setup, default using relation-pos pair')
_arg_parser.add_argument('--optim', default='adam', type=str, action='store', help='The optimizer')
_arg_parser.add_argument('--lr', default=0.01, type=float, action='store', help='Initial learning rate')
_arg_parser.add_argument('--beta', default=0.2, type=float, action='store', help='')
_arg_parser.add_argument('--alpha_st', default=0.2, type=float, action='store', help='The alpha_1 value')
_arg_parser.add_argument('--alpha_ts', default=1, type=float, action='store', help='The alpha_2 value')
_arg_parser.add_argument('--prm_batch', default=300, type=int, action='store',
                         help='The batchsize for computing features (for parallelism)')
_arg_parser.add_argument('--batch_size', default=500, type=int, action='store', help='The batchsize for training')
_arg_parser.add_argument('--mx_itr', default=10, type=int, action='store', help='Max number of batch updates per epoch')
_arg_parser.add_argument('--mx_sent', default=10000, type=int, action='store',
                         help='Max number of sentences used for training (only for training self-permutation model)')
_arg_parser.add_argument('--mx_epoch', default=10, type=int, action='store', help='Max number of epoches')
_arg_parser.add_argument('--mx_dep_train', default=5, type=int, action='store', help='Max number of nodes for training')
_arg_parser.add_argument('--mx_dep', default=7, type=int, action='store', help='Max number of nodes for test')
_arg_parser.add_argument('--mx_len', default=40, type=int, action='store',
                         help='Max length for the sentence for training')
_arg_parser.add_argument('--continue_train', default=1, type=int, action='store',
                         help='Reload the model and continue training')
_arg_parser.add_argument('--check_freq', default=1, type=int, action='store',
                         help='Save training model every X epoches')
_arg_parser.add_argument('--verbose', default=1, type=int, action='store', help='Verbosity')
_arg_parser.add_argument('--seed', default=1, type=int, action='store', help='Random seed')
_args = _arg_parser.parse_args()

logger = get_logger()


def _config(args_dict=vars(_args)):
    return '\n'.join('{}:{}'.format(key, val) for (key, val) in args_dict.items())


def _start():
    logger.info('Python Version:\n' + sys.version)
    logger.info('Numpy Version:%s' % np.__version__)
    logger.info('PyTorch Version:%s' % torch.__version__)
    random.seed(_args.seed)
    np.random.seed(_args.seed)
    Option.add(vars(_args))
    Option.format = 'conllu'
    logger.info('Configuration:\n' + str(Option))
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


def _finish():
    logger.info('<DONE>')


def _callbacks(model=_args.model):
    return [ModelCheckpoint(model,
                            verbose=Option.verbose, continue_train=Option.continue_train,
                            nb_epoch=Option.mx_epoch, check_freq=Option.check_freq)]


def _build_feature_vocab(fn):
    from perm import CollectFeatureVocab, PermModel
    collect_vocab = CollectFeatureVocab()
    logger.info('Building Feature vocabulary from...:%s' % fn)
    treebank = Data(fn, conll_reader)
    valid = 0
    Global.mx_dep = Option.mx_dep_train
    for tt, tree in enumerate(map(NaryTree.conll2tree, treebank)):
        if PermModel.valid_tree(tree):
            valid += 1
            tree.postOrderTraversal(collect_vocab)
    logger.info('Trees processed:%d/%d' % (valid, tt + 1))
    return collect_vocab.vocab


@experiment(before=[_start], after=[_finish])
def self_model():
    feature_vocab = _build_feature_vocab(Option.src)
    from perm import TreeParamPermModel as PM
    pm = PM(feature_vocab, Option)
    logger.info('[Train] Loading the source language...:%s' % Option.src)
    Global.mx_dep = Option.mx_dep_train
    data_train = Data(Option.src, conll_reader, valid=lambda x: len(x) < Option.mx_len)
    logger.info('Training...')
    fit(pm, {'data': data_train, 'mx_sent': Option.mx_sent, 'batch_size': Option.batch_size},
        nb_epoch=Option.mx_epoch, callbacks=_callbacks(model=''))
    save(fn=Option.model, model={'model': pm.state_dict(), 'feature_vocab': feature_vocab})


def _load_lm(prfx, lm_data):
    from lm import BiGramLanguageModel as LM
    logger.info('[%s] Training LM...' % prfx)
    lm = LM(smoother_name=Option.lm_smoother, train_kwargs={'data': lm_data})
    logger.info('[%s] Number of POS types: %i' % (prfx, lm.vocab_size))
    return lm


@experiment(before=[_start], after=[_finish])
def permute():
    if not Option.model:
        logger.info('[Warning] No model file will be saved since --model isn\'t specified!')
    logger.info('[Target] Loading the target language...:%s' % Option.tgt)
    tgt_data = Data(Option.tgt, conll_reader if Option.tgt.endswith('.conllu') else text_reader)
    lm_tgt = _load_lm('Target', tgt_data)
    from perm import UParamPermModel as PM
    if Option.pretrain:
        logger.info('Loading the pretrained model...:%s' % Option.pretrain)
        ptr_pm = load(Option.pretrain)
        feature_vocab = ptr_pm['feature_vocab']
        pm = PM(feature_vocab, Option, lm_tgt, smoother=lm_tgt.smoother, lambdap=lm_tgt.lambdap,
                vocab_size=lm_tgt.vocab_size)
        pm.load_state_dict(ptr_pm['model'])
    else:
        feature_vocab = _build_feature_vocab(Option.src)
        pm = PM(feature_vocab, Option, lm_tgt, smoother=lm_tgt.smoother, lambdap=lm_tgt.lambdap,
                vocab_size=lm_tgt.vocab_size)

    logger.info('[Train_before] Loading the source language...:%s' % Option.src)
    Global.mx_dep = Option.mx_dep_train
    Global.feature_vocab = feature_vocab
    data_train = Data(Option.src, conll_reader, valid=lambda x: len(x) < Option.mx_len)
    lm_src_train = _load_lm('Train', pm.filter(data_train))
    logger.info('[Train_before] Objective: %f' % pm.skewD(lm_tgt, lm_src_train))
    logger.info('Training...')
    fit(pm, {'data': data_train, 'mx_itr': Option.mx_itr, 'batch_size': Option.batch_size},
        nb_epoch=Option.mx_epoch, callbacks=_callbacks())
    save_model(Option.model, Option.to_dict(), Global.to_dict())

    Global.mx_dep = Option.mx_dep
    logger.info('[Test_before] Loading the source language...:%s' % Option.src)
    data_test = Data(Option.src, conll_reader)
    lm_data_test = _load_lm('Test_before', data_test)
    logger.info('[Test_before] Skew-Divergence: %f' % pm.skewD(lm_tgt, lm_data_test))
    logger.info('[Test] Permuting&writing...:%s' % Option.output)
    write_conll(Option.output, pm.permute(data_test))
    lm_data_test_out = _load_lm('Test_after', Data(Option.output, conll_reader))
    logger.info('[Test_after] Skew-Divergence: %f' % pm.skewD(lm_tgt, lm_data_test_out))


def _load_model_data(model_path, info=''):
    model_prfx = model_path[:model_path.find('.weight')]
    args, metadata = map(VarDict, load_model(model_prfx))
    logger.info('Model%s Configuration:\n' % info + str(args))
    from perm import UParamPermModel as PM
    pm = PM(metadata.feature_vocab, args, None, smoother=None, lambdap=None, vocab_size=None)
    logger.info('Initialize with pre-trained weights:%s' % model_path)
    pm.load(model_path)
    return pm, args, metadata


@experiment(before=[_start], after=[_finish])
def test():
    pm, args, metadata = _load_model_data(Option.model)
    Global.mx_dep = Option.mx_dep
    for test, test_out in zip(Option.test, Option.test_out):
        logger.info('[Test] Loading the source language...:%s' % test)
        data_test = Data(test, conll_reader)
        logger.info('[Test] Permuting&writing...:%s' % test_out)
        write_conll(test_out, pm.permute(data_test))


if __name__ == "__main__": locals()[_args.task]()
