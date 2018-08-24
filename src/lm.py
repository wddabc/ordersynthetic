#!/usr/bin/python
# --------------------------------------- 
# File Name : model.py
# Creation Date : 22-01-2018
# Last Modified : Mon Jan 22 12:45:42 2018
# Created By : wdd 
# ---------------------------------------
import re
import sys
import math
import numpy as np
from glo import get_logger
from data import Data, TokenEntry, CONST_BOS, CONST_EOS, CONST_UNK, CONST_TKN
from typing import Callable, List, Dict, Iterable
from util import MAX_INT
from abc import ABC, abstractmethod

logger = get_logger()


class LanguageModel(ABC):
    def __init__(self, train_kwargs: Dict):
        self.vocab = set()
        if train_kwargs:
            self.train(**train_kwargs)

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def log_likelihood(self):
        ...


class BiGramLanguageModel(LanguageModel):
    def __init__(self,
                 smoother_name: str = 'backoff_add-1',
                 thr: int = 0,
                 train_kwargs: Dict = None
                 ):
        self.set_smoother(smoother_name)
        self.thr = thr
        LanguageModel.__init__(self, train_kwargs)

    def set_smoother(self, arg):
        '''Sets smoother type and lambda from a string passed in by the user on the
           command line.'''

        r = re.compile('^(.*?)-?([0-9.]*)$')
        m = r.match(arg)

        if not m.lastindex:
            sys.exit('Smoother regular expression failed for %s' % arg)
        else:
            smoother_name = m.group(1)
            if m.lastindex >= 2 and len(m.group(2)):
                lambda_arg = m.group(2)
                self.lambdap = float(lambda_arg)
            else:
                self.lambdap = None

        if smoother_name.lower() == 'uniform':
            self.smoother = 'UNIFORM'
        elif smoother_name.lower() == 'add':
            self.smoother = 'ADDL'
        elif smoother_name.lower() == 'backoff_add':
            self.smoother = 'BACKOFF_ADDL'
        else:
            assert False, 'Can\'t recognize smoother name %s' % smoother_name
        logger.info('smoother=%s, lambda=%.2f' % (self.smoother, self.lambdap))

        if self.lambdap is None and self.smoother.find('ADDL') != -1:
            assert False, 'You must include a non-negative lambda value in smoother name %s' % arg

    def _count(self, x, y=None):
        if y:
            self.tokens[x, y] = self.tokens.get((x, y), 0) + 1.
            self.tokens[y] = self.tokens.get(y, 0) + 1.
            self.tokens[CONST_TKN] = self.tokens.get(CONST_TKN, 0) + 1.  # the zero-gram
        else:
            assert x == CONST_BOS
            self.tokens[x] = self.tokens.get(x, 0) + 1.

    def _build_vocab(self, data, mx_sent: int = MAX_INT,
                     valid: Callable[[List[TokenEntry]], bool] = lambda x: True):
        # Read the training corpus and collect any information that will be
        # needed by prob later on.
        # Note: In a real system, you wouldn't do this work every time you
        # ran the testing program.  You'd do it only once and save the
        # trained model to disk in some format.
        vocab = {}
        for i_sent, sent in enumerate(filter(valid, data)):
            assert type(sent) == list
            if i_sent == mx_sent: break
            vocab[CONST_BOS] = vocab.get(CONST_BOS, 0.) + 1.
            for i, ent in enumerate(sent[1:]):
                x = ent.cpos
                vocab[x] = vocab.get(x, 0.) + 1.
            vocab[CONST_EOS] = vocab.get(CONST_EOS, 0.) + 1.
        self.vocab.update(k for k, v in vocab.items() if v >= self.thr)
        self.vocab.add(CONST_UNK)

    def log_prob(self, x, y):
        '''Computes a smoothed estimate of the bigram probability p(y | x)
           according to the language model.'''
        if self.smoother == 'UNIFORM':
            return -math.log(self.vocab_size)
        elif self.smoother == 'ADDL':
            return math.log(self.tokens.get((x, y), 0) + self.lambdap) - \
                   math.log(self.tokens.get(x, 0) + self.lambdap * self.vocab_size)
        elif self.smoother.startswith('BACKOFF_ADDL'):
            py = math.log(self.tokens.get(y, 0) + self.lambdap) - \
                 math.log(self.tokens[CONST_TKN] + self.lambdap * self.vocab_size)
            pxy = math.log(self.tokens.get((x, y), 0) + self.lambdap * self.vocab_size * math.exp(py)) - \
                  math.log(self.tokens.get(x, 0) + self.lambdap * self.vocab_size)
            return pxy
        else:
            sys.exit('%s has some weird value' % self.smoother)

    def prob(self, x, y):
        return math.exp(self.log_prob(x, y))

    def train(self,
              data,
              mx_sent: int = MAX_INT,
              valid: Callable[[List[TokenEntry]], bool] = lambda x: True
              ):
        # While training, we'll keep track of all the bigram types
        # accumulate the type and token counts into the global hash tables.
        data = list(data)
        self._build_vocab(data, mx_sent, valid)
        self.tokens, x, nsent = {}, CONST_BOS, 0
        for sent in filter(valid, data):
            if nsent == mx_sent:
                break
            x = CONST_BOS
            self._count(x)
            for i, ent in enumerate(sent[1:]):
                y = ent.cpos
                if y not in self.vocab:
                    y = CONST_UNK
                self._count(x, y)
                x = y
            self._count(x, CONST_EOS)  # count EOS 'end of sequence' token after the final context
            nsent += 1
        self.vocab_size = len(self.vocab)
        logger.info('-->Number of tokens:%d' % self.tokens[CONST_TKN])
        logger.info('-->Number of sentences:%d' % nsent)

    def log_likelihood(self,
                       data: Iterable[List[TokenEntry]],
                       mx_sent: int = MAX_INT
                       ):
        logger.info('Computing Log-Likelihood...')
        logprob, ntkn, x, nsent = 0.0, 0, CONST_BOS, 0
        for sent in data:
            if nsent == mx_sent:
                break
            x = CONST_BOS
            for ent in sent[1:]:
                y = ent.cpos
                if y not in self.vocab:
                    y = CONST_UNK
                logprob += self.log_prob(x, y)
                x = y
                ntkn += 1
            logprob += self.log_prob(x, CONST_EOS)
            ntkn += 1
            nsent += 1
        logger.info('-->Number of tokens:%d' % ntkn)
        logger.info('-->Number of sentences:%d' % nsent)
        logger.info('-->Log-Likelihood:%.2f' % logprob)
        return logprob, ntkn

    def log_likelihood1(self,
                        test_count
                        ):
        logger.info('Computing Log-Likelihood...')
        neg_llh = 0.
        for st, c_st in test_count.items():
            if type(st) == tuple:
                neg_llh += -c_st * self.log_prob(*st)
        print(neg_llh, test_count[CONST_TKN], np.exp(neg_llh / test_count[CONST_TKN]))

    def test(self,
             data: Iterable[List[TokenEntry]],
             mx_sent: int = MAX_INT,
             ):
        logprob, ntkn = self.log_likelihood(data, mx_sent)
        return ntkn, logprob, np.exp(-logprob / ntkn)

    def perplexity(self,
                   data: Iterable[List[TokenEntry]],
                   mx_sent: int = MAX_INT,
                   ):
        logprob, ntkn = self.log_likelihood(data, mx_sent)
        return np.exp(-logprob / ntkn)
