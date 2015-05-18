#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Last modified: linpeng.ding(DDYDLP@gmail.com)

import re
import jieba
import jieba.posseg as posse
import jieba.analyse as analyse

word_dict = {u'每月': ["每月"]}
jieba.load_userdict_from_file('userdict.txt')
jieba.load_userdict_from_dict(word_dict)

re_han_default = re.compile("([\u4E00-\u9FA5a-zA-Z0-9+#&\._]+)", re.U)
re_skip_default = re.compile("(\r\n|\s)", re.U)
re_han_cut_all = re.compile("([\u4E00-\u9FA5]+)", re.U)
re_skip_cut_all = re.compile("[^a-zA-Z0-9+#\n]", re.U)


def segment_all_words(texts):
    word_info = jieba.cut(texts, cut_all=True)
    for item in word_info:
        print item


def segment_hmm_words(texts):
    word_info = jieba.cut(texts, hmm=True)
    for item in word_info:
        print item


def segment_word(word):
    """
    文本切词排序
    :param word:
    :return:
    """
    seg_list = posse.cut(''.join(analyse.extract_tags(word, max(len(word) / 7, 5))))
    key_words = jieba.cut_for_search(''.join(set(map(lambda x: ''
                                     if x.flag[0] in ['q', 'd', 'p', 'u'] else x.word, seg_list))))
    for item in key_words:
        print item

    return list(key_words)


def segment_text_desc(word):
    """
    文本描述切词
    :param word:
    :return:
    """
    key_words = jieba.tokenize(word)

    for item in key_words:
        print item
    return list(key_words)


def segment_user_input(word):
    """
    用户输入切词
    :param word:
    :return:
    """
    # lexical = ['n', 'v', 'a', 't']
    # seg_words = posse.cut(word)
    # key_words = map(lambda x: x.word, sorted(filter(lambda x: x.flag[0] in lexical, seg_words),
    #                                          key=lambda x: lexical.index(x.flag[0])))
    key_words = analyse.extract_tags(word, max(len(word) / 7, 5))
    if not key_words:
        return [word.decode('utf-8')]
    return key_words


def segment_cut_all_word(word):
    """
    用户输入切词（全模式）
    :param word:
    :return:
    """
    seg_words = jieba.cut(word, cut_all=True)
    for item in seg_words:
        print item
    return list(seg_words)


if __name__ == "__main__":
    """
    贪心算法遍历每一个字，每个字刻画组成词的子节点结构
    全模式：阿/狸/是/一名/来自/清华/清华大学/清华大学美术学院/华大/大学/大学美术/美术/美术学
           /美术学院/术学/学院/的/学生/创造/造出/造出来/出来/的/虚构/小狐/狐狸
        遍历每个字，并输出与之关联的词，用字典储存子节点

    精确模式：阿狸/，/是/一名/来自/清华大学美术学院/的/学生/创造/出来/的/虚构/小/狐狸/。/
        最大长度子节点切分，通过维特比算法切除每个字前后优先成词的顺序，以及可能成词的新词

    搜索模式：阿狸/，/是/一名/来自/清华/华大/大学/美术/术学/学院/美术学/清华大学美术学院/
            /的/学生/创造/出来/的/虚构/小/狐狸/。/
        最大长度子节点切分，每个词内进行最小长度切分

    关键字提取：清华大学美术学院/阿狸/虚构/狐狸/一名/创造/学生/来自/出来
              根据tf/idf方法，进行词排序
    """
    words = u"工信处女干事每月经过下属科室都要亲口交代交换机等技术性器件的安装工作, 科学术"
    # segment_all_words(words)
    # segment_text_desc(words)
    segment_hmm_words(words)
    # segment_cut_all_word(words)
    # segment_text_desc(words)
    # segment_user_input(words)