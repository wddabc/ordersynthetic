#!/usr/bin/python
# --------------------------------------- 
# File Name : data.py
# Creation Date : 22-01-2018
# Last Modified : Mon Jan 22 12:45:35 2018
# Created By : wdd 
# ---------------------------------------
import re
import os
import shutil
from glo import get_logger, Option
from typing import IO, List, Callable, Optional, Tuple
from abc import ABC, abstractmethod

logger = get_logger()

CONST_ROOT = 'ROOT'
CONST_BOS = '_BOS_'
CONST_EOS = '_EOS_'
CONST_UNK = 'X'
CONST_TKN = '_ALL_'
INIT_POSTAG = dict((p, i) for i, p in enumerate([CONST_BOS, CONST_EOS, CONST_UNK]))
INIT_DEPREL = dict((p, i) for i, p in enumerate([CONST_UNK]))
UD_POSTAG = [CONST_BOS, CONST_EOS, CONST_UNK, "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
             "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB"]
UD_DEPREL = [CONST_UNK, CONST_ROOT, "root", "acl", "advcl", "advmod", "amod", "appos", "aux", "auxpass",
             "case",
             "cc", "ccomp", "compound", "conj", "cop", "csubj", "csubjpass", "dep", "det", "discourse", "dislocated",
             "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "name", "neg", "nmod", "nsubj",
             "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "vocative", "xcomp"]
DEFAULT_POSTAG_DICT = dict((t, i) for i, t in enumerate(UD_POSTAG))
DEFAULT_DEPREL_DICT = dict((t, i) for i, t in enumerate(UD_DEPREL))
numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");


# 1 ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes.
# 2 FORM: Word form or punctuation symbol.
# 3 LEMMA: Lemma or stem of word form.
# 4 UPOSTAG: Universal part-of-speech tag.
# 5 XPOSTAG: Language-specific part-of-speech tag; underscore if not available.
# 6 FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
# 7 HEAD: Head of the current word, which is either a value of ID or zero (0).
# 8 DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
# 9 DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
# 10 MISC: Any other annotation.


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


class TokenEntry(object):
    def __init__(self, id, form, lemma, cpos, pos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos.upper()
        self.pos = cpos.upper() if Option.format == 'conllu' else pos.upper()
        self.parent_id = parent_id
        self.relation = relation.split(':')[0] if relation else relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                  str(self.parent_id), self.relation, self.deps,
                  self.misc]
        return '\t'.join(['_' if v is None else v for v in values])

    def to_tuple(self):
        return (
            self.id, self.form, self.norm, self.cpos, self.pos, self.parent_id, self.relation, self.lemma, self.feats,
            self.deps, self.misc, self.pred_parent_id, self.pred_relation)

    @staticmethod
    def new_entry(id, form, norm, cpos, pos, parent_id, relation, lemma, feats,
                  deps, misc, pred_parent_id, pred_relation):
        ret = TokenEntry(id, form, lemma, pos, cpos)
        ret.id = id
        ret.form = form
        ret.norm = norm
        ret.cpos = cpos
        ret.pos = pos
        ret.parent_id = parent_id
        ret.relation = relation
        ret.lemma = lemma
        ret.feats = feats
        ret.deps = deps
        ret.misc = misc
        ret.pred_parent_id = pred_parent_id
        ret.pred_relation = pred_relation
        return ret

    def copy(self):
        return TokenEntry.new_entry(self.id,
                                    self.form,
                                    self.norm,
                                    self.cpos,
                                    self.pos,
                                    self.parent_id,
                                    self.relation,
                                    self.lemma,
                                    self.feats,
                                    self.deps,
                                    self.misc,
                                    self.pred_parent_id,
                                    self.pred_relation)


def conll_reader(fh: IO) -> List[TokenEntry]:
    root = TokenEntry(0, CONST_ROOT, CONST_ROOT, CONST_ROOT, CONST_ROOT, CONST_ROOT, -1, CONST_ROOT, CONST_ROOT,
                      CONST_ROOT)
    tokens = [root]
    cnt = 0
    for line in fh:
        tok = re.compile('\t').split(line.strip())

        if not tok or line.strip() == '':
            if len(tokens) > 1:
                cnt += 1
                yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                continue
            else:
                tokens.append(TokenEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5],
                                         int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9]))
    if len(tokens) > 1:
        cnt += 1
        yield tokens


def text_reader(fh: IO) -> List[TokenEntry]:
    for line in fh:
        yield [TokenEntry(0, CONST_ROOT, CONST_ROOT, CONST_ROOT, CONST_ROOT, CONST_ROOT, -1, CONST_ROOT, CONST_ROOT,
                          CONST_ROOT)] + \
              list(map(lambda x: TokenEntry(0, 'None', 'None', x, 'None', 'None', -1, 'None', 'None', 'None'),
                       line.strip().split()))


def write_conll(fn, data):
    logger.info('Writing to:%s' % fn)
    tmp = fn + '.tmp'
    i_sent, tkns = -1, 0
    with open(tmp, 'w', encoding='utf-8') as f:
        for i_sent, sent in enumerate(data):
            for ent in sent[1:]:
                tkns += 1
                print(str(ent), file=f)
            print('', file=f)
    if os.path.exists(tmp):
        shutil.move(tmp, fn)
    logger.info('-->Number of tokens:%d' % tkns)
    logger.info('-->Number of sentences:%d' % (i_sent + 1))


class Data(object):
    def __init__(self, fname: str, reader: Callable[[IO], List[TokenEntry]], valid=lambda x: True):
        self._fname = fname
        self._reader = reader
        self._valid = valid

    def __iter__(self) -> List[TokenEntry]:
        tt, t_tkn, valid, valid_tkn = 0, 0, 0, 0
        with open(self._fname, 'r', encoding='utf-8') as f:
            for data in self._reader(f):
                tt += 1
                t_tkn += len(data) - 1
                if self._valid(data):
                    valid += 1
                    valid_tkn += len(data) - 1
                    yield data
        logger.info('[Data]Sentences processed:%d/%d' % (valid, tt))
        logger.info('[Data]Tokens processed:%d/%d' % (valid_tkn, t_tkn))

    @property
    def fname(self) -> str:
        return self._fname


class NaryTree(object):
    def __init__(self, start: int, end: int, tag: str, relation: str, entry: TokenEntry,
                 children: List[Optional['NaryTree']] = None):
        self.start = start
        self.end = end
        self.tag = tag
        self.relation = relation
        self.entry = entry
        self.children = children

    def __len__(self):
        return self.end - self.start

    class NodeFunc(ABC):
        def __init__(self):
            ...

        @abstractmethod
        def __call__(self, node: Optional['NaryTree']):
            ...

    class CountTokens(NodeFunc):
        def __init__(self):
            NaryTree.NodeFunc.__init__(self)
            self.count = 0

        def __call__(self, node: Optional['NaryTree']):
            if not node.children:
                self.count += 1

    class CountValid(NodeFunc):
        def __init__(self):
            NaryTree.NodeFunc.__init__(self)
            self.count = 0

        def __call__(self, node: Optional['NaryTree']):
            if node.children and \
                    (len(node.children) == 1 or \
                     (node.tag == 'VERB' or node.tag == 'NOUN' or node.tag == 'PRON' or node.tag == 'PROPN')):
                self.count += 1

    class CountMaxChildren(NodeFunc):
        def __init__(self):
            NaryTree.NodeFunc.__init__(self)
            self.count = 0

        def __call__(self, node: Optional['NaryTree']):
            if node.children and len(node.children) > self.count:
                self.count = len(node.children)

    class UpdateSpan(NodeFunc):
        def __init__(self):
            NaryTree.NodeFunc.__init__(self)

        def __call__(self, node: Optional['NaryTree']):
            if node.children:
                node.start = node.children[0].start
                node.end = node.children[-1].end

    class FringeCollector(NodeFunc):
        def __init__(self):
            NaryTree.NodeFunc.__init__(self)
            self.leaves = []

        def __call__(self, node: Optional['NaryTree']):
            if node.entry:
                self.leaves += [node]

    def preOrderTraversal(self, func: NodeFunc) -> None:
        func(self)
        if self.children:
            for child in self.children:
                child.preOrderTraversal(func)

    def postOrderTraversal(self, func: NodeFunc) -> None:
        if self.children:
            for child in self.children:
                child.postOrderTraversal(func)
        func(self)

    @staticmethod
    def conll2tree(list_tkn: List[TokenEntry]) -> Optional['NaryTree']:
        list_node, list_head = [], []
        for i, tkn in enumerate(list_tkn):
            if not i: continue
            list_head += [tkn.parent_id - 1]
            assert len(list_head) == tkn.id
            node = NaryTree(tkn.id - 1, tkn.id, tkn.pos, tkn.relation, None,
                            [NaryTree(tkn.id - 1, tkn.id, tkn.pos, tkn.relation, tkn)])
            list_node += [node]
        # Check Nonprojectivity
        for idx1, head1 in enumerate(list_head):
            for idx2, head2 in enumerate(list_head):
                if idx1 > head1 and head1 != head2:
                    if (idx1 > head2 and idx1 < idx2 and head1 < head2) \
                            or (idx1 < head2 and idx1 > idx2 and head1 < idx2):
                        return None
                if idx1 < head1 and head1 != head2:
                    if (head1 > head2 and head1 < idx2 and idx1 < head2) \
                            or (head1 < head2 and head1 > idx2 and idx1 < idx2):
                        return None
        root = None
        for i, (node, head_idx) in enumerate(zip(list_node, list_head)):
            if head_idx == -1:
                root = node
                continue
            head = list_node[head_idx]
            if i < head_idx:
                head.children.insert(len(head.children) - 1, node)
            else:
                head.children += [node]
        root.postOrderTraversal(NaryTree.UpdateSpan())
        cnt_mx = NaryTree.CountMaxChildren()
        root.postOrderTraversal(cnt_mx)
        root.mx_chd = cnt_mx.count
        return root

    @staticmethod
    def genDependency(tree: Optional['NaryTree'], leaves: List) -> Tuple[Optional['NaryTree'], int]:
        if tree.entry:
            tup = [tree, tree.start]
            leaves[tree.start] = tup
            return tup
        head_idx, list_head = -1, []
        for child in tree.children:
            _, head_idx_child = NaryTree.genDependency(child, leaves)
            if child.entry:
                head_idx = head_idx_child
            list_head += [head_idx_child]
        for i in list_head:
            leaves[i][1] = head_idx
        return leaves[head_idx]

    def _updateSpan(self):
        collector = NaryTree.FringeCollector()
        self.postOrderTraversal(collector)
        for i, leaf in enumerate(collector.leaves):
            leaf.start, leaf.end = i, i + 1
            leaf.entry.id = i + 1
        self.postOrderTraversal(NaryTree.UpdateSpan())

    def _updateDependency(self):
        leaves = [None] * self.end
        NaryTree.genDependency(self, leaves)[1] = -1
        for leaf, head_idx in leaves:
            leaf.entry.parent_id = head_idx + 1

    def deepCopy(self):
        if self.entry:
            return NaryTree(self.start, self.end, self.tag, self.relation, self.entry.copy())
        else:
            return NaryTree(self.start, self.end, self.tag, self.relation, None,
                            children=list(map(lambda x: x.deepCopy(), self.children)))

    def postPorc(self):
        self._updateSpan()
        self._updateDependency()

    @staticmethod
    def tree2conll(tree: Optional['NaryTree']) -> List[TokenEntry]:
        collector = NaryTree.FringeCollector()
        tree.postOrderTraversal(collector)
        return [TokenEntry(0, CONST_ROOT, CONST_ROOT, CONST_ROOT, CONST_ROOT, CONST_ROOT, -1, CONST_ROOT,
                           CONST_ROOT, CONST_ROOT)] + list(map(lambda x: x.entry, collector.leaves))

    def __str__(self):
        return '\n'.join(list(map(str, NaryTree.tree2conll(self)[1:])))


def itr_file_list(input, pattern):
    pattern = re.compile(pattern)
    for root, dir, files in os.walk(input):
        for fn in files:
            abs_fn = os.path.join(root, fn)
            folder_pattern = os.path.join(os.path.basename(root), fn)
            if pattern.match(folder_pattern):
                yield os.path.normpath(abs_fn)
