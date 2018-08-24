import os
import time
import shutil
import math
import random
import json
import numpy as np
from glo import Global, get_logger
from typing import Any, Tuple, List

try:
    import cPickle as pickle
except:
    import pickle

logger = get_logger()

MAX_INT = 0xffffffff
MIN_FLOAT = -float('inf')


def save(fn, model: Any, verbose: bool = True):
    if not fn: return
    tmp = fn + '.tmp'
    if fn.endswith('.json'):
        o = 'w'
        saver = json
    else:
        o = 'wb'
        saver = pickle
    with open(tmp, o) as outfile:
        if verbose:
            logger.info('Saving to:%s' % fn)
        saver.dump(model, outfile)
    if os.path.exists(tmp):
        shutil.move(tmp, fn)


def load(fn, verbose: bool = True):
    if fn.endswith('.json'):
        i = 'r'
        loader = json
    else:
        i = 'rb'
        loader = pickle
    with open(fn, i) as f:
        if verbose:
            logger.info('Loading from:%s' % fn)
        model = loader.load(f)
    return model


def save_model(model_prfx: str, args: Any, metadata: Any):
    if not model_prfx: return
    logger.info('Saving metadata...')
    save(fn=model_prfx + '.args.json', model=args)
    save(fn=model_prfx + '.metadata.pkl', model=metadata)


def load_model(model_prfx: str, items: Tuple = ('.args.json', '.metadata.pkl')):
    ret = []
    for item in items:
        fn = model_prfx + item
        model = load(fn)
        ret += [model]
    return ret


class Callback(object):
    def __init__(self):
        pass

    def _set_params(self, params):
        self.params = params

    def _set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass


class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto'):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            logger.warn('EarlyStopping mode %s is unknown, '
                        'fallback to auto mode.' % (self.mode),
                        RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs={}):
        self.wait = 0  # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            logger.warn('Early stopping requires %s available!' %
                        (self.monitor), RuntimeWarning)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

    def on_train_end(self, logs={}):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))


class ModelCheckpoint(Callback):
    @staticmethod
    def _exists_model(model_path):
        return os.path.isfile(model_path)

    @staticmethod
    def _rm_model(model_path):
        return os.remove(model_path)

    def __init__(self, filepath, eval_func_map={}, save_best_only=True, verbose=0, continue_train=True, check_freq=1,
                 nb_epoch=MAX_INT):
        super(ModelCheckpoint, self).__init__()
        self._status_path = filepath + '.train_status.json'
        self._continue_train = continue_train
        self.verbose = verbose
        self._check_freq = check_freq
        self._epoch_begin = 0
        self._nb_epoch = nb_epoch
        self._eval_func_map = eval_func_map
        self._status = {}
        self._save_best_only = save_best_only
        monitor, mode = '', 'auto'
        for key in eval_func_map:
            if key.startswith('*'):
                monitor, val = key[1:], eval_func_map.pop(key)
                eval_func_map[monitor] = val
                mode = val[1]
                assert type(mode) is str
                break
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath + '.weights.e{epoch_continue}' if filepath else filepath

        if mode not in ['auto', 'min', 'max']:
            logger.warn('ModelCheckpoint mode %s is unknown, '
                        'fallback to auto mode.' % (mode),
                        RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def _stop_train(self, logs={}):
        if math.isnan(logs['loss']):
            logger.info('Nan dectected! Stop training')
            return True
        if logs['epoch_continue'] == self._nb_epoch:
            return True

    def print_eval(self, epoch, last_model_path, logs):
        header = '*' if ModelCheckpoint._exists_model(last_model_path) else '-'
        if self.verbose > 0:
            header += 'epoch=%i,loss=%.6f,time=%i' % (epoch, logs.get('loss'), logs.get('time'))
            if self.monitor:
                header += ',%s=%.2f(best_%i:%.2f)' % (self.monitor, logs.get(self.monitor), self.best_epoch, self.best)
            res_evals = ','.join(
                '%s=%.2f' % (eval_name, func(self.model, logs)) for eval_name, (func, freq) in
                self._eval_func_map.items()
                if eval_name != self.monitor and not (epoch + 1) % freq)
            if len(res_evals):
                header += ',' + res_evals
            logger.info(header)
        return header

    def on_train_begin(self, logs={}):
        self.best_epoch = 0
        Global.counter = 0
        if self._continue_train and os.path.isfile(self._status_path):
            last_status = load(self._status_path, verbose=False)
            last_model_path = last_status['last_model_path']
            self._epoch_begin = last_status['last_itr']
            rand_status = last_status['last_random']
            random.setstate((rand_status[0], tuple(rand_status[1]), rand_status[2]))
            Global.fpointer = last_status['last_counter']
            logger.info('Initialize with pre-trained weights:%s' % last_model_path)
            self.model.load(last_model_path)
            if self._continue_train > 1:
                self._status = last_status
                if 'last_' + self.monitor in last_status:
                    logs[self.monitor] = last_status['last_' + self.monitor]
                    if 'best_itr' in last_status:
                        self.best_epoch = last_status['best_itr']
                        self.best = last_status['best_val']
                        self.best_model_path = last_status['best_model_path']
                        self.best_info = last_status['best_info']
            logger.info('Begin with epoch:%i' % self._epoch_begin)
            if self._epoch_begin >= self._nb_epoch:
                logger.info('Model already trained for epoch:%i' % self._epoch_begin)
                self.model.stop_training = True
        else:
            logs['loss'], logs['time'] = -1, 0
            Global.fpointer = 0
            self.on_epoch_end(0, logs)

    def on_epoch_end(self, epoch, logs={}):
        epoch += self._epoch_begin
        logs['epoch_continue'] = epoch
        if self._stop_train(logs):
            self.model.stop_training = True
        last_model_path = self.filepath.format(epoch=epoch, **logs)
        if ModelCheckpoint._exists_model(last_model_path):
            ModelCheckpoint._rm_model(last_model_path)
        is_best = False
        if self.monitor:
            current = self._eval_func_map[self.monitor][0](self.model, logs)
            logs[self.monitor] = current
            if self.monitor_op(current, self.best):  # this is the best model
                self.best = current
                if self._save_best_only:
                    if 'best_model_path' in self._status:
                        ModelCheckpoint._rm_model(self._status['best_model_path'])
                self.model.save(last_model_path)
                is_best = True
                self.best_epoch = epoch
                self._status['best_itr'] = epoch
                self._status['best_val'] = self.best
                self._status['best_model_path'] = last_model_path
            elif not (epoch + 1) % self._check_freq:
                self.model.save(last_model_path)
        elif not (epoch + 1) % self._check_freq:
            self.model.save(last_model_path)
        header = self.print_eval(epoch, last_model_path, logs)
        if is_best:
            self.best_info = header
            self._status['best_info'] = header
        if ModelCheckpoint._exists_model(last_model_path):
            self._status['last_' + self.monitor] = logs.get(self.monitor)
            self._status['last_model_path'] = last_model_path
            self._status['last_itr'] = epoch
            self._status['last_random'] = random.getstate()
            self._status['last_counter'] = Global.counter
            save(self._status_path, self._status, verbose=False)

    def on_train_end(self, logs={}):
        if self.monitor:
            logger.info('Best:' + self.best_info)


def fit(model, data_kwargs, nb_epoch=100, callbacks=[]):
    for cbk in callbacks: cbk.model = model
    model.stop_training = False
    logs = {}
    for cbk in callbacks: cbk.on_train_begin(logs)
    for epoch in range(1, nb_epoch + 1):
        if model.stop_training: break
        ts = time.time()
        logs['loss'] = model.train(**data_kwargs)
        logs['time'] = time.time() - ts
        for cbk in callbacks: cbk.on_epoch_end(epoch, logs)
    for cbk in callbacks: cbk.on_train_end(logs)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def show_parameters(module):
    for p in module._parameters:
        print(p)
    for child in module.children():
        print('----------------', child, '----------------')
        show_parameters(child)


def _sjt_perm(n: int, p: List[int], pi: List[int], d: List[int], swap_list: List[Tuple[int, int]]) -> None:
    if n >= len(p):
        return
    _sjt_perm(n + 1, p, pi, d, swap_list);
    for i in range(n):
        swap_list += [(np.min([pi[n], pi[n] + d[n]]), np.max([pi[n], pi[n] + d[n]]))]
        z = p[pi[n] + d[n]]
        p[pi[n]] = z
        p[pi[n] + d[n]] = n
        pi[z] = pi[n]
        pi[n] = pi[n] + d[n]
        _sjt_perm(n + 1, p, pi, d, swap_list)
    d[n] = -d[n]


def get_swap(n: int) -> List[Tuple[int, int]]:
    if n == 1:
        return [(0, 0)]
    p = list(range(n))  # permutation
    pi = list(range(n))  # inverse permutation
    d = [-1] * n  # direction = +1 or -1
    list_sjt_swap = []
    _sjt_perm(0, p, pi, d, list_sjt_swap)
    list_sjt_swap += [(0, 1)]
    return list_sjt_swap
