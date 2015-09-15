#encoding=utf-8
from __future__ import absolute_import
import jieba
import jieba.posseg
import os
from operator import itemgetter
from .textrank import textrank
try:
    from .analyzer import ChineseAnalyzer
except ImportError:
    pass

_curpath = os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
abs_path = os.path.join(_curpath, "idf.txt")

STOP_WORDS = set((
    "the","of","is","and","to","in","that","we","for","an","are",
    "by","be","as","on","with","can","if","from","which","you","it",
    "this","then","at","have","all","not","one","has","or","that"
))


class IDFLoader:
    def __init__(self):
        self.path = ""
        self.idf_freq = {}
        self.median_idf = 0.0

    def set_new_path(self, new_idf_path):
        if self.path != new_idf_path:
            content = open(new_idf_path, 'rb').read().decode('utf-8')
            idf_freq = {}
            lines = content.rstrip('\n').split('\n')
            for line in lines:
                word, freq = line.split(' ')
                idf_freq[word] = float(freq)
            median_idf = sorted(idf_freq.values())[len(idf_freq)//2]
            self.idf_freq = idf_freq
            self.median_idf = median_idf
            self.path = new_idf_path

    def get_idf(self):
        return self.idf_freq, self.median_idf

idf_loader = IDFLoader()
idf_loader.set_new_path(abs_path)


def set_idf_path(idf_path):
    new_abs_path = os.path.normpath(os.path.join(os.getcwd(), idf_path))
    if not os.path.exists(new_abs_path):
        raise Exception("jieba: path does not exist: " + new_abs_path)
    idf_loader.set_new_path(new_abs_path)


def set_stop_words(stop_words_path):
    global STOP_WORDS
    abs_path = os.path.normpath(os.path.join(os.getcwd(), stop_words_path))
    if not os.path.exists(abs_path):
        raise Exception("jieba: path does not exist: " + abs_path)
    content = open(abs_path,'rb').read().decode('utf-8')
    lines = content.replace("\r", "").split('\n')
    for line in lines:
        STOP_WORDS.add(line)


def extract_tags(sentence, top_k=20, with_weight=False, allow_pos=None):
    """
    Extract keywords from sentence using TF-IDF algorithm
    :param sentence:
    :param top_k: return how many top keywords. `None` for all possible words
    :param with_weight: if True, return a list of (word, weight);if False, return a list of words
    :param allow_pos: the allowed POS list eg. ['ns', 'n', 'vn', 'v','nr']
                if the POS of w is not in this list,it will be filtered
    :return:
    """
    global STOP_WORDS, idf_loader

    idf_freq, median_idf = idf_loader.get_idf()

    if allow_pos:
        allow_pos = frozenset(allow_pos)
        words = jieba.posseg.cut(sentence)
    else:
        words = jieba.cut(sentence)
    freq = {}
    for w in words:
        if allow_pos:
            if w.flag not in allow_pos:
                continue
            else:
                w = w.word
        if len(w.strip()) < 2 or w.lower() in STOP_WORDS:
            continue
        freq[w] = freq.get(w, 0.0) + 1.0
    total = sum(freq.values())
    for k in freq:
        freq[k] *= idf_freq.get(k, median_idf) / total

    if with_weight:
        tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
    else:
        tags = sorted(freq, key=freq.__getitem__, reverse=True)
    if top_k:
        return tags[:top_k]
    else:
        return tags
