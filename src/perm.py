#!/usr/bin/python
# --------------------------------------- 
# File Name : perm.py
# Creation Date : 23-01-2018
# Last Modified : Tue Jan 23 23:40:09 2018
# Created By : wdd 
# ---------------------------------------
import os
import shutil
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import *
from data import Data, TokenEntry, NaryTree
from typing import List, Dict
from glo import get_logger, Option, Global
from util import MAX_INT, get_swap
from data import CONST_BOS, CONST_EOS, CONST_TKN
from abc import ABC, abstractmethod
from torch import optim
from itertools import cycle
import numpy as np
import random
import sys

# from memory_profiler import profile
logger = get_logger()
if 'GPU' not in os.environ or int(os.environ['GPU']) == 0:
    logger.info('Using CPU')
    use_gpu = False
else:
    logger.info('Using GPU')
    use_gpu = True

get_data = (lambda x: x.data.cpu()) if use_gpu else (lambda x: x.data)

activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu}

VERB = 'V'
NOUN = 'N'


def norm(w):
    uprob = torch.exp(w - torch.max(w))
    return uprob / torch.sum(uprob)


def Variable(inner):
    return torch.autograd.Variable(inner.cuda() if use_gpu else inner)


def MyTensor(inner):
    return torch.Tensor(inner)


def Parameter(shape=None, init=xavier_uniform):
    if hasattr(init, 'shape'):
        assert not shape
        return nn.Parameter(MyTensor(init))
    shape = (shape, 1) if type(shape) == int else shape
    return nn.Parameter(init(MyTensor(shape)))


def scalar(f):
    if type(f) == int:
        return Variable(torch.LongTensor([f]))
    return Variable(torch.FloatTensor([float(f)]))


def cat(l, dimension=-1):
    valid_l = l
    if dimension < 0:
        dimension += len(valid_l[0].size())
    return torch.cat(valid_l, dimension)


def get_optim(opt, parameters):
    if opt == 'sgd':
        return optim.SGD(parameters, lr=Option.lr)
    elif opt == 'adam':
        return optim.Adam(parameters, lr=Option.lr)
    elif opt == 'adadelta':
        return optim.Adadelta(parameters, lr=Option.lr)
    elif opt == 'adagrad':
        return optim.Adagrad(parameters, lr=Option.lr)
    elif opt == 'adamax':
        return optim.Adamax(parameters, lr=Option.lr)
    elif opt == 'asgd':
        return optim.ASGD(parameters, lr=Option.lr)
    elif opt == 'rmsprop':
        return optim.RMSprop(parameters, lr=Option.lr)
    elif opt == 'rprop':
        return optim.Rprop(parameters, lr=Option.lr)


perm_list_map = dict(map(lambda x: (x, get_swap(x)), range(1, 8)))


class PermModel(ABC):
    def __init__(self):
        ...

    @staticmethod
    def valid_tree(tree: NaryTree) -> bool:
        return tree and tree.mx_chd <= Global.mx_dep

    @abstractmethod
    def permute_tree(self, tree: NaryTree) -> NaryTree:
        ...

    def _iter_valid_trees(self, treebank: Data) -> NaryTree:
        valid, tt = 0, 0
        for tree in map(NaryTree.conll2tree, treebank):
            tt += 1
            if self.__class__.valid_tree(tree):
                valid += 1
                yield tree
        logger.info('Trees processed:%d/%d' % (valid, tt))

    def filter(self, treebank: Data) -> List[TokenEntry]:
        tt, valid = 0, 0
        for tree in self._iter_valid_trees(treebank):
            cnt_tkn = NaryTree.CountTokens()
            tree.postOrderTraversal(cnt_tkn)
            tt += cnt_tkn.count
            cnt_valid = NaryTree.CountValid()
            tree.postOrderTraversal(cnt_valid)
            valid += cnt_valid.count
            yield NaryTree.tree2conll(tree)
        logger.info('Nodes processed:%d/%d,%.2f' % (valid, tt, float(valid) / tt))

    def permute(self, treebank: Data) -> List[TokenEntry]:
        for tree in self._iter_valid_trees(treebank):
            yield NaryTree.tree2conll(self.permute_tree(tree))


class ParamModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def _step(self, list_loss):
        self.loss(list_loss).backward()
        self.trainer.step()
        self.trainer.zero_grad()

    def train(self, data: Data, mx_sent: int = MAX_INT, batch_size: int = 1) -> float:
        ...

    def save(self, fn):
        if not fn: return
        tmp = fn + '.tmp'
        torch.save(self.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        m = torch.load(fn) if use_gpu else torch.load(fn, map_location=lambda storage, loc: storage)
        self.load_state_dict(m)

    def loss(self, list_loss):
        ...


class TreeParamModel(ParamModel):
    def __init__(self):
        ParamModel.__init__(self)

    def train(self, data: Data, mx_sent: int = MAX_INT, batch_size: int = 1) -> float:
        shuffledData = list(self._iter_valid_trees(data))
        random.shuffle(shuffledData)
        nsent, ntkn, tt_n_loss, list_e_loss = 0, 0, 0., []
        loss = 0
        start = time.time()
        for tree in shuffledData:
            if nsent == mx_sent: break
            if len(list_e_loss) == batch_size:
                loss = -tt_n_loss / nsent
                self._step(list_e_loss)
                logger.info('Processed sentence number:%i, LL:%f, Time:%f' %
                            (nsent, loss, time.time() - start))
                list_e_loss = []
                start = time.time()
            e_loss, n_loss = self.forward(tree)
            list_e_loss += [e_loss]
            tt_n_loss += n_loss
            ntkn += len(tree) + 1
            nsent += 1
        if list_e_loss:
            loss = -tt_n_loss / nsent
            self._step(list_e_loss)
            logger.info('Processed sentence number:%i, LL:%f, Time:%f' %
                        (nsent, loss, time.time() - start))
        return loss


class UParamModel(ParamModel):
    def __init__(self):
        ParamModel.__init__(self)

    def train(self, data: Data, mx_itr: int = MAX_INT, batch_size: int = 1) -> float:
        shuffledData = list(self._iter_valid_trees(data))
        random.shuffle(shuffledData)
        n_sent, n_itr, batch = 0, 0, []
        start = time.time()
        for tree in cycle(shuffledData):
            if n_itr == mx_itr: break
            if len(batch) == batch_size:
                e_loss, (info, n_loss) = self.forward(batch)
                self._step(e_loss)
                n_itr += 1
                batch.clear()
                logger.info('Processed sentence number:%i, %s:%f, Time:%f' %
                            (n_sent, info, n_loss, time.time() - start))
                start = time.time()
            batch += [tree]
            n_sent += 1
        return n_loss


class CollectFeatureVocab(NaryTree.NodeFunc):
    def __init__(self):
        NaryTree.NodeFunc.__init__(self)
        self.t = set()
        self.tt = set()
        self.r = set()
        self.rr = set()
        self.tr = set()
        self.trtr = set()

    def _add_bi_feature(self, c1, c2):
        if isinstance(c1, str):
            c1_tag, c1_relation = c1, c1
        else:
            c1_tag, c1_relation = c1.tag, c1.relation
        if isinstance(c2, str):
            c2_tag, c2_relation = c2, c2
        else:
            c2_tag, c2_relation = c2.tag, c2.relation
        self.tt.add((c1_tag, c2_tag))
        self.rr.add((c1_relation, c2_relation))
        self.trtr.add((c1_tag, c1_relation, c2_tag, c2_relation))

    def _add_uni_feature(self, c: NaryTree):
        self.t.add(c.tag)
        self.r.add(c.relation)
        self.tr.add((c.tag, c.relation))

    def _init_feature(self, children: List[NaryTree], head_idx):
        n = len(children)
        self._add_bi_feature(CONST_BOS, children[0])
        self._add_bi_feature(children[-1], CONST_EOS)
        for i in range(n):
            if i > 0:
                self._add_bi_feature(children[i - 1], children[i])
            if i < head_idx:
                self._add_uni_feature(children[i])
            for j in range(i + 1, n):
                if i != head_idx and j != head_idx:
                    self._add_bi_feature(children[i], children[j])
                if i < head_idx and j > head_idx:
                    self._add_bi_feature(children[i], children[j])
                if i > head_idx and j > head_idx:
                    self._add_bi_feature(children[i], children[j])

    def _update_feature(self, children: List[NaryTree], head_idx, i, j):
        n = len(children)
        left_ch = CONST_BOS if not i else children[i - 1]
        right_ch = CONST_EOS if j == n - 1 else children[j + 1]
        self._add_bi_feature(left_ch, children[j])
        self._add_bi_feature(children[i], right_ch)
        self._add_bi_feature(children[j], children[i])
        if i == head_idx:
            self._add_uni_feature(children[j])
            for k in range(j + 1, n):
                self._add_bi_feature(children[j], children[k])
        if j == head_idx:
            for k in range(j + 1, n):
                self._add_bi_feature(children[i], children[k])
            for k in range(i):
                self._add_bi_feature(children[k], children[i])
        if i != head_idx and j != head_idx:
            self._add_bi_feature(children[j], children[i])
            if i > head_idx and j > head_idx:
                self._add_bi_feature(children[j], children[i])

    def __call__(self, tree: NaryTree):
        if tree.children and (tree.tag == 'VERB' or tree.tag == 'NOUN' or tree.tag == 'PRON' or tree.tag == 'PROPN'):
            nchildren, children = len(tree.children), tree.children
            for i, child in enumerate(children):
                if child.entry:
                    head_idx = i
            swap_list = perm_list_map[nchildren]
            self._init_feature(children, head_idx)
            for i, j in swap_list:
                self._update_feature(children, head_idx, i, j)
                children[i], children[j] = children[j], children[i]
                if head_idx == i:
                    head_idx = j
                elif head_idx == j:
                    head_idx = i

    @property
    def vocab(self):
        return {
            't': {t: i for i, t in enumerate(self.t)},
            'tt': {t: i for i, t in enumerate(self.tt)},
            'r': {t: i for i, t in enumerate(self.r)},
            'rr': {t: i for i, t in enumerate(self.rr)},
            'tr': {t: i for i, t in enumerate(self.tr)},
            'trtr': {t: i for i, t in enumerate(self.trtr)}
        }

    def __str__(self):
        return 't: %i\n' % len(self.t) + \
               'tt: %i\n' % len(self.tt) + \
               'r: %i\n' % len(self.r) + \
               'rr: %i\n' % len(self.rr) + \
               'tr: %i\n' % len(self.tr) + \
               'trtr: %i' % len(self.trtr)


class PermProxy(NaryTree.NodeFunc):
    def __init__(self, model, option, feature_vocab):
        NaryTree.NodeFunc.__init__(self)
        self.option = option
        self.feature_vocab = feature_vocab
        self.model = model
        self.feature_type = ['_', '_L', '_m', '_r', '_A']
        for prfx in [VERB, NOUN]:
            ulen, blen = 0, 0
            if 'T' in option.feature:
                ulen += len(feature_vocab['t'])
                blen += len(feature_vocab['tt'])
            if 'R' in self.option.feature:
                ulen += len(feature_vocab['r'])
                blen += len(feature_vocab['rr'])
            if 'A' in self.option.feature:
                ulen += len(feature_vocab['tr'])
                blen += len(feature_vocab['trtr'])
            init = {'init': np.zeros((ulen + 4 * blen, 1))} if option.init == 'zero' \
                else {'init': np.random.normal(np.zeros((ulen + 4 * blen, 1)))}
            self.model._reg_param(prfx, Parameter(**init))
        logger.info('Number of parameters: %i' % (2 * (ulen + 4 * blen)))

    def _add_bi_feature(self, prfx: str, c1, c2, feat_map: Dict, val=1.):
        feature_group = feat_map.setdefault(prfx, {})
        if isinstance(c1, str):
            c1_tag, c1_relation = c1, c1
        else:
            c1_tag, c1_relation = c1.tag, c1.relation
        if isinstance(c2, str):
            c2_tag, c2_relation = c2, c2
        else:
            c2_tag, c2_relation = c2.tag, c2.relation
        if 'T' in self.option.feature:
            if (c1_tag, c2_tag) in self.feature_vocab['tt']:
                feature_group['tt'][self.feature_vocab['tt'][c1_tag, c2_tag]] += val
        if 'R' in self.option.feature:
            if (c1_relation, c2_relation) in self.feature_vocab['rr']:
                feature_group['rr'][self.feature_vocab['rr'][c1_relation, c2_relation]] += val
        if 'A' in self.option.feature:
            if (c1_tag, c1_relation, c2_tag, c2_relation) in self.feature_vocab['trtr']:
                feature_group['trtr'][self.feature_vocab['trtr'][c1_tag, c1_relation, c2_tag, c2_relation]] += val

    def _add_uni_feature(self, prfx: str, c: NaryTree, feat_map: Dict, val=1.):
        feature_group = feat_map.setdefault(prfx, {})
        if 'T' in self.option.feature:
            if c.tag in self.feature_vocab['t']:
                feature_group['t'][self.feature_vocab['t'][c.tag]] += val
        if 'R' in self.option.feature:
            if c.relation in self.feature_vocab['r']:
                feature_group['r'][self.feature_vocab['r'][c.relation]] += val
        if 'A' in self.option.feature:
            if (c.tag, c.relation) in self.feature_vocab['tr']:
                feature_group['tr'][self.feature_vocab['tr'][c.tag, c.relation]] += val

    def _init_feature(self, children: List[NaryTree], head_idx):
        feat_map, n = {}, len(children)
        for f_type in self.feature_type:
            pair = [('T', 't'), ('R', 'r'), ('A', 'tr')] if f_type == '_' else [('T', 'tt'), ('R', 'rr'), ('A', 'trtr')]
            feat_group = feat_map.setdefault(self.prfx + f_type, {})
            for f, t in filter(lambda x: x[0] in self.option.feature, pair):
                feat_group[t] = np.zeros(len(self.feature_vocab[t]), dtype=float)
        self._add_bi_feature(self.prfx + '_A', CONST_BOS, children[0], feat_map)
        self._add_bi_feature(self.prfx + '_A', children[-1], CONST_EOS, feat_map)
        for i in range(n):
            if i:
                self._add_bi_feature(self.prfx + '_A', children[i - 1], children[i], feat_map)
            if i < head_idx:
                self._add_uni_feature(self.prfx + '_', children[i], feat_map)
            for j in range(i + 1, n):
                if i != head_idx and j != head_idx:
                    self._add_bi_feature(self.prfx + '_L', children[i], children[j], feat_map)
                if i < head_idx and j > head_idx:
                    self._add_bi_feature(self.prfx + '_m', children[i], children[j], feat_map)
                if i > head_idx and j > head_idx:
                    self._add_bi_feature(self.prfx + '_r', children[i], children[j], feat_map)
        return feat_map

    def _update_feature(self, children: List[NaryTree], head_idx, i, j, feat_map):
        n = len(children)
        left_ch = CONST_BOS if not i else children[i - 1]
        right_ch = CONST_EOS if j == n - 1 else children[j + 1]
        self._add_bi_feature(self.prfx + '_A', left_ch, children[i], feat_map, -1.)
        self._add_bi_feature(self.prfx + '_A', left_ch, children[j], feat_map)
        self._add_bi_feature(self.prfx + '_A', children[j], right_ch, feat_map, -1.)
        self._add_bi_feature(self.prfx + '_A', children[i], right_ch, feat_map)
        self._add_bi_feature(self.prfx + '_A', children[i], children[j], feat_map, -1.)
        self._add_bi_feature(self.prfx + '_A', children[j], children[i], feat_map)
        if i == head_idx:
            self._add_uni_feature(self.prfx + '_', children[j], feat_map)
            for k in range(j + 1, n):
                self._add_bi_feature(self.prfx + '_m', children[j], children[k], feat_map)
                self._add_bi_feature(self.prfx + '_r', children[j], children[k], feat_map, -1.)
            for k in range(i):
                self._add_bi_feature(self.prfx + '_m', children[k], children[j], feat_map, -1.)
        if j == head_idx:
            self._add_uni_feature(self.prfx + '_', children[i], feat_map, -1.)
            for k in range(j + 1, n):
                self._add_bi_feature(self.prfx + '_r', children[i], children[k], feat_map)
                self._add_bi_feature(self.prfx + '_m', children[i], children[k], feat_map, -1.)
            for k in range(i):
                self._add_bi_feature(self.prfx + '_m', children[k], children[i], feat_map)
        if i != head_idx and j != head_idx:
            self._add_bi_feature(self.prfx + '_L', children[i], children[j], feat_map, -1.)
            self._add_bi_feature(self.prfx + '_L', children[j], children[i], feat_map)
            if i > head_idx and j > head_idx:
                self._add_bi_feature(self.prfx + '_r', children[j], children[i], feat_map)
                self._add_bi_feature(self.prfx + '_r', children[i], children[j], feat_map, -1.)

    def _get_vec(self, feature_pack):
        feature_vec = []
        for feature_type, feature_group in feature_pack.items():
            feature_vec += list(feature_group.values())
        return np.concatenate(feature_vec)

    def _get_lin_score(self, feature_pack):
        weights = self.model.__getattr__(self.prfx)
        feature_vec = []
        for feature_type, feature_group in feature_pack.items():
            feature_vec += list(feature_group.values())
        return torch.mm(Variable(MyTensor([np.concatenate(feature_vec)])), weights)

    def viz_feature(self, feature_pack):
        val = 0
        for feature_type, feature_group in feature_pack.items():
            print('-----------------------------------')
            print(feature_type)
            weights = self.model.__getattr__(feature_type)
            s = 0
            for gram, vec in feature_group.items():
                print(gram, ':')
                for f, id in self.feature_vocab[gram].items():
                    if vec[id] != 0.:
                        val += vec[id] * weights[s + id].data.numpy()
                        print(f, vec[id], weights[s + id].data.numpy()[0])
                s += len(self.feature_vocab[gram])
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        return val

    def __call__(self, tree: NaryTree, w2span):
        if tree.children:
            nchildren = len(tree.children)
            if len(tree.children) < 2:
                return
            if tree.tag == 'VERB':
                self.prfx = VERB
            elif tree.tag == 'NOUN' or tree.tag == 'PRON' or tree.tag == 'PROPN':
                self.prfx = NOUN
            else:
                return
            swap_list = perm_list_map[nchildren]
            order, orders = list(range(nchildren)), []
            for i, j in swap_list:
                order[i], order[j] = order[j], order[i]
                orders += [list(order)]
            prob = norm(w2span[tree.start, tree.end])
            prob = prob.cpu() if use_gpu else prob
            selected_order = random.choices(orders, weights=prob.data.numpy(), k=1)[0]
            new_children = [None] * nchildren
            for i, o in enumerate(selected_order):
                new_children[i] = tree.children[o]
            tree.children = new_children

    def iter_features(self, tree: NaryTree):
        if tree.children and len(tree.children) > 1 and (
                tree.tag == 'VERB' or tree.tag == 'NOUN' or tree.tag == 'PRON' or tree.tag == 'PROPN'):
            nchildren = len(tree.children)
            if tree.tag == 'VERB':
                self.prfx = VERB
            elif tree.tag == 'NOUN' or tree.tag == 'PRON' or tree.tag == 'PROPN':
                self.prfx = NOUN
            swap_list = perm_list_map[nchildren]
            children = tree.children
            for i, child in enumerate(children):
                if child.entry:
                    head_idx = i
            feature_map = self._init_feature(children, head_idx)
            for i, j in swap_list:
                self._update_feature(children, head_idx, i, j, feature_map)
                children[i], children[j] = children[j], children[i]
                if head_idx == i:
                    head_idx = j
                elif head_idx == j:
                    head_idx = i
                yield (tree.start, tree.end, self.prfx, self._get_vec(feature_map))
        if tree.children:
            for child in tree.children:
                for itm in self.iter_features(child):
                    yield itm

    def collect_perm_probs(self, tree: NaryTree):
        Nbatch, Nw, Ni = [], [], 0
        Nweights = self.model.__getattr__(NOUN)
        Vbatch, Vw, Vi = [], [], 0
        Vweights = self.model.__getattr__(VERB)
        span_map = {}
        for s, t, prfx, vec in self.iter_features(tree):
            if prfx == NOUN:
                Ni += 1
                Nbatch += [vec]
                if (s, t) not in span_map:
                    span_map[s, t] = [NOUN, Ni - 1, None]
                span_map[s, t][2] = Ni
                if len(Nbatch) == self.option.prm_batch:
                    Nw += [torch.mm(Variable(MyTensor(Nbatch)), Nweights)]
                    Nbatch.clear()
            if prfx == VERB:
                Vi += 1
                Vbatch += [vec]
                if (s, t) not in span_map:
                    span_map[s, t] = [VERB, Vi - 1, None]
                span_map[s, t][2] = Vi
                if len(Vbatch) == self.option.prm_batch:
                    Vw += [torch.mm(Variable(MyTensor(Vbatch)), Vweights)]
                    Vbatch.clear()
        if len(Nbatch):
            Nw += [torch.mm(Variable(MyTensor(Nbatch)), Nweights)]
        if len(Vbatch):
            Vw += [torch.mm(Variable(MyTensor(Vbatch)), Vweights)]
        if len(Nw):
            Nw = torch.cat(Nw)
        if len(Vw):
            Vw = torch.cat(Vw)

        ret = {}
        for (s, t), (prfx, i, j) in span_map.items():
            if prfx == NOUN:
                ret[s, t] = Nw[i:j]
            elif prfx == VERB:
                ret[s, t] = Vw[i:j]
        return ret

    def permute(self, tree: NaryTree):
        w2span = self.collect_perm_probs(tree)
        tree.postOrderTraversal(lambda x: self(x, w2span))
        tree.postPorc()
        return tree


class BiGramECount(NaryTree.NodeFunc):

    def __init__(self, model_proxy, smoother, lambdap, vocab_size, option):
        NaryTree.NodeFunc.__init__(self)
        self._ec = {}
        self._C = {}
        self._L = {}
        self._R = {}
        self.model_proxy = model_proxy
        self.smoother = smoother
        self.lambdap = lambdap
        self.vocab_size = vocab_size
        self.option = option

    @staticmethod
    def _update_dict(to_dict, from_dict):
        for k, val in from_dict.items():
            to_dict[k] = to_dict.get(k, 0.) + val

    def __call__(self, tree: NaryTree):
        if tree.entry:
            self._C.setdefault((tree.start, tree.end), {})
            self._L.setdefault((tree.start, tree.end), {})[tree.tag] = scalar(1.)
            self._R.setdefault((tree.start, tree.end), {})[tree.tag] = scalar(1.)
            return
        else:
            nchildren = len(tree.children)
            self.model_proxy.prfx = None
            if tree.tag == 'VERB':
                self.model_proxy.prfx = VERB
            elif tree.tag == 'NOUN' or tree.tag == 'PRON' or tree.tag == 'PROPN':
                self.model_proxy.prfx = NOUN
            C, L, R, children = {}, {}, {}, tree.children
            for i, child in enumerate(children):
                if child.entry:
                    head_idx = i
                BiGramECount._update_dict(C, self._C[child.start, child.end])
            root = children[head_idx].entry.parent_id == 0
            if self.model_proxy.prfx and nchildren > 1:
                swap_list = perm_list_map[nchildren]
                fs = norm(self.span2w[tree.start, tree.end])
                order, order_ec, last_non, Z = list(range(nchildren)), {}, {}, []
                s = -1
                for t in order + [nchildren]:
                    last_non[s, t] = -1
                    s = t

                Z_ = scalar(0.)

                def count(s, t):
                    last_idx = last_non.pop((s, t))
                    incr = Z_ if last_idx == -1 else Z_ - Z[last_idx]
                    order_ec[s, t] = order_ec.get((s, t), scalar(0.)) + incr

                def update_lnon(s, t, idx):
                    assert (s, t) not in last_non
                    last_non[s, t] = idx

                for idx, (w, (i, j)) in enumerate(zip(fs, swap_list)):
                    o_i, o_j = order[i], order[j]
                    o_im1, o_jp1 = order[i - 1] if i > 0 else -1, order[j + 1] if j < nchildren - 1 else nchildren
                    count(o_i, o_j)
                    update_lnon(o_j, o_i, idx - 1)
                    count(o_im1, o_i)
                    update_lnon(o_im1, o_j, idx - 1)
                    count(o_j, o_jp1)
                    update_lnon(o_i, o_jp1, idx - 1)
                    order[i], order[j] = order[j], order[i]
                    Z += [Z_ + w]
                    Z_ = Z[-1]
                s = -1
                for t in order + [nchildren]:
                    count(s, t)
                    s = t
                for (i, j), ec in order_ec.items():
                    if i == -1:
                        for l, lv in self._L[children[j].start, children[j].end].items():
                            L[l] = L.get(l, scalar(0.)) + ec * lv
                        continue
                    if j == nchildren:
                        for r, rv in self._R[children[i].start, children[i].end].items():
                            R[r] = R.get(r, scalar(0.)) + ec * rv
                        continue
                    c_i, c_j = children[i], children[j]
                    for l, lv in self._R[c_i.start, c_i.end].items():
                        for r, rv in self._L[c_j.start, c_j.end].items():
                            C[l, r] = C.get((l, r), scalar(0.)) + ec * lv * rv
            else:
                for i in range(nchildren - 1):
                    c_i, c_j = children[i], children[i + 1]
                    for l, lv in self._R[c_i.start, c_i.end].items():
                        for r, rv in self._L[c_j.start, c_j.end].items():
                            C[l, r] = C.get((l, r), scalar(0.)) + lv * rv
                L = self._L[children[0].start, children[0].end]
                R = self._R[children[-1].start, children[-1].end]
            self._C[tree.start, tree.end] = C
            self._L[tree.start, tree.end] = L
            self._R[tree.start, tree.end] = R
            if root:
                for t, v in L.items():
                    C[CONST_BOS, t] = L[t]
                for t, v in R.items():
                    C[t, CONST_EOS] = R[t]
                self._ec = C

    def log_prob(self, x, y):
        '''Computes a smoothed estimate of the bigram probability p(y | x)
           according to the language model.'''
        if self.smoother == 'UNIFORM':
            return scalar(-math.log(self.vocab_size))
        elif self.smoother == 'ADDL':
            return torch.log(self.bi_ec.get((x, y), scalar(0.)) + scalar(self.lambdap)) - \
                   scalar(math.log(self.uni_c.get(x, 0.) + self.lambdap * self.vocab_size))
        elif self.smoother.startswith('BACKOFF_ADDL'):
            py = math.log(self.uni_c.get(y, 0.) + self.lambdap) - \
                 math.log(self.uni_c[CONST_TKN] + self.lambdap * self.vocab_size)
            pxy = torch.log(
                self.bi_ec.get((x, y), scalar(0.)) + scalar(self.lambdap * self.vocab_size * math.exp(py))) - \
                  scalar(math.log(self.uni_c.get(x, 0.) + self.lambdap * self.vocab_size))
            return pxy
        else:
            sys.exit('%s has some weird value' % self.smoother)

    def train(self, batch: List[NaryTree]):
        self.uni_c, self.bi_ec = {}, {}
        for tree in batch:
            collector = NaryTree.FringeCollector()
            tree.postOrderTraversal(collector)
            for t in [CONST_BOS, CONST_EOS] + list(map(lambda x: x.tag, collector.leaves)):
                self.uni_c[t] = self.uni_c.get(t, 0.) + 1.
            self.uni_c[CONST_TKN] = self.uni_c.get(CONST_TKN, 0.) + len(collector.leaves) + 1
            self.span2w = self.model_proxy.collect_perm_probs(tree)
            tree.postOrderTraversal(self)
            for st, ec_st in self._ec.items():
                self.bi_ec[st] = self.bi_ec.get(st, scalar(0.)) + ec_st
            self.clear()

    def clear(self):
        self._C.clear()
        self._L.clear()
        self._R.clear()
        self._ec.clear()


class PermECount(NaryTree.NodeFunc):

    def __init__(self, model_proxy, option):
        NaryTree.NodeFunc.__init__(self)
        self.model_proxy = model_proxy
        self.option = option
        self._ec = []

    def __call__(self, tree: NaryTree):
        if tree.children:
            nchildren = len(tree.children)
            self.model_proxy.prfx = None
            if tree.tag == 'VERB':
                self.model_proxy.prfx = VERB
            elif tree.tag == 'NOUN' or tree.tag == 'PRON' or tree.tag == 'PROPN':
                self.model_proxy.prfx = NOUN
            else:
                return
            if self.model_proxy.prfx and nchildren > 1:
                ws = self.span2w[tree.start, tree.end]
                uprob = ws - torch.max(ws)
                self._ec += [uprob[-1] - torch.log(torch.sum(torch.exp(uprob)))]

    def loss(self):
        neg_llh = -torch.sum(torch.cat(self._ec)) if self._ec else scalar(0.)
        cpu_neg_llh = neg_llh.cpu() if use_gpu else neg_llh
        return neg_llh, cpu_neg_llh.data.numpy()

    def clear(self):
        self._ec.clear()


class UParamPermModel(UParamModel, PermModel):

    def __init__(self, feature_vocab, option, test_lm, **kwargs):
        PermModel.__init__(self)
        ParamModel.__init__(self)
        self.feature_vocab = feature_vocab
        self.feature_types = {}
        self.option = option
        self.test_lm = test_lm
        self.model_proxy = PermProxy(self, option, feature_vocab)
        self.trainer = get_optim(option.optim, self.parameters())
        self.train_proxy = BiGramECount(self.model_proxy, kwargs['smoother'], kwargs['lambdap'], kwargs['vocab_size'],
                                        option)
        if use_gpu:
            self.cuda()

    def _reg_param(self, name, value):
        self.__setattr__(name, value)
        self.feature_types[name] = value.data.numpy().shape

    def loss(self, e_loss):
        """
        Multiply by zero to make sure all the parameters have gradients
        """
        ret = e_loss
        for feature_type, shape in self.feature_types.items():
            weights = self.__getattr__(feature_type)
            ret += torch.mm(Variable(MyTensor(np.zeros(shape).transpose())), weights)[0, 0]
        return ret

    def skewD(self, lm1, lm2):

        def skD(m1, m2, alpha):
            ret = 0.
            for st, c_st in m1.tokens.items():
                if type(st) == tuple:
                    log_r = m1.log_prob(*st)
                    log_q = m2.log_prob(*st)
                    ret += c_st * (log_r - np.log(alpha * np.exp(log_q) + (1.0 - alpha) * np.exp(log_r)))
            return ret / m1.tokens[CONST_TKN]

        alpha_st = self.option.alpha_st
        alpha_ts = self.option.alpha_ts
        beta = self.option.beta
        return (1 - beta) * skD(lm1, lm2, alpha_st) + beta * skD(lm2, lm1, alpha_ts)

    def _skew_obj(self, batch: List[NaryTree]):
        self.train_proxy.train(batch)
        # D(q||r) =sum_y q(y)(log q(y) − log r(y))
        # s_a(q, r) = D(r||aq + (1 − a)r)
        #           = sum_y r(y)(log r(y) − log aq(y)+(1-a)r(y))
        alpha_st = self.option.alpha_st
        sk_st = scalar(0.)
        for st, c_st in self.test_lm.tokens.items():
            if type(st) == tuple:
                log_r = self.test_lm.log_prob(*st)
                log_q = self.train_proxy.log_prob(*st)
                sk_st += scalar(c_st * log_r) \
                         - scalar(c_st) * torch.log(
                    scalar(alpha_st) * torch.exp(log_q)
                    + scalar((1.0 - alpha_st) * np.exp(log_r))
                )
        sk_st *= scalar(self.train_proxy.uni_c[CONST_TKN] / self.test_lm.tokens[CONST_TKN])
        alpha_ts = self.option.alpha_ts
        sk_ts = scalar(0.)
        for st, e_st in self.train_proxy.bi_ec.items():
            log_r = self.train_proxy.log_prob(*st)
            log_q = self.test_lm.log_prob(*st)
            sk_ts += e_st * (log_r - torch.log(scalar(alpha_ts * np.exp(log_q))
                                               + scalar(1.0 - alpha_ts) * torch.exp(log_r)))
        beta = self.option.beta
        loss = scalar(1. - beta) * sk_st + scalar(beta) * sk_ts
        cpu_loss = loss.cpu() if use_gpu else loss
        np_loss = cpu_loss.data.numpy()
        return loss, ('OBJ', np_loss / self.train_proxy.uni_c[CONST_TKN])

    def forward(self, batch: List[NaryTree]):
        return self._skew_obj(batch)

    def permute_tree(self, tree: NaryTree) -> NaryTree:
        return self.model_proxy.permute(tree)


class TreeParamPermModel(TreeParamModel, PermModel):

    def __init__(self, feature_vocab, option):
        PermModel.__init__(self)
        TreeParamModel.__init__(self)
        self.option = option
        if not feature_vocab:
            logger.info('WARNING: Empty model!')
            return
        self.feature_vocab = feature_vocab
        self.feature_types = {}
        self.model_proxy = PermProxy(self, option, feature_vocab)
        self.trainer = get_optim(option.optim, self.parameters())
        self.train_proxy = PermECount(self.model_proxy, option)
        if use_gpu:
            self.cuda()

    def _reg_param(self, name, value):
        self.__setattr__(name, value)
        self.feature_types[name] = value.data.numpy().shape

    def loss(self, list_loss):
        """
        Multiply by zero to make sure all the parameters have gradients
        """
        ret = torch.sum(cat(list_loss))
        for feature_type, shape in self.feature_types.items():
            weights = self.__getattr__(feature_type)
            ret += torch.mm(Variable(MyTensor(np.zeros(shape).transpose())), weights)[0, 0]
        return ret

    def forward(self, tree: NaryTree):
        self.train_proxy.span2w = self.model_proxy.collect_perm_probs(tree)
        tree.postOrderTraversal(self.train_proxy)
        e_loss, n_loss = self.train_proxy.loss()
        self.train_proxy.clear()
        return e_loss, n_loss

    def permute_tree(self, tree: NaryTree) -> NaryTree:
        return self.model_proxy.permute(tree)
