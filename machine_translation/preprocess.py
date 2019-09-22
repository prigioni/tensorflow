#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2019-8-22 9:45 
# @Author : lauqasim
# @File : preprocess.py 
# @Software: PyCharm
import warnings
warnings.filterwarnings("ignore")
import tqdm
import pickle


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def text_to_int(sentence, map_dict, max_length=20, is_target=False):
    """
    对文本句子进行数字编码
    sentence: 一个完整的句子，str类型
    map_dict: 单词到数字的映射，dict
    max_length: 句子的最大长度
    is_target: 是否为目标语句。在这里要区分目标句子与源句子，因为对于目标句子（即翻译后的句子）我们需要在句子最后增加<EOS>
    """

    # 用<PAD>填充整个序列
    text_to_idx = []
    # unk index
    unk_idx = map_dict.get("<UNK>")
    pad_idx = map_dict.get("<PAD>")
    eos_idx = map_dict.get("<EOS>")

    # 如果输入源文本
    if not is_target:
        for word in sentence.lower().split():
            text_to_idx.append(map_dict.get(word, unk_idx))

    # 否则，对于输出目标文本需要做<EOS>的填充最后
    else:
        for word in sentence.lower().split():
            text_to_idx.append(map_dict.get(word, unk_idx))
        text_to_idx.append(eos_idx)

    # 如果超长需要截断
    if len(text_to_idx) > max_length:
        return text_to_idx[:max_length]
    # 如果不够则增加<PAD>
    else:
        text_to_idx = text_to_idx + [pad_idx] * (max_length - len(text_to_idx))
        return text_to_idx


def preprocess_and_save_data(source_path, target_path):
    source_text = load_data(source_path)
    target_text = load_data(target_path)

    # 构造英文词典
    source_vocab = list(set(source_text.lower().split()))
    # 构造法文词典
    target_vocab = list(set(target_text.lower().split()))

    # 特殊字符
    SOURCE_CODES = ['<PAD>', '<UNK>']
    TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']  # 在target中，需要增加<GO>与<EOS>特殊字符

    # 构造英文映射字典
    source_vocab_to_int = {word: idx for idx, word in enumerate(SOURCE_CODES + source_vocab)}
    source_int_to_vocab = {idx: word for idx, word in enumerate(SOURCE_CODES + source_vocab)}

    # 构造法语映射词典
    target_vocab_to_int = {word: idx for idx, word in enumerate(TARGET_CODES + target_vocab)}
    target_int_to_vocab = {idx: word for idx, word in enumerate(TARGET_CODES + target_vocab)}

    # 对源句子进行转换 Tx = 20
    source_text_to_int = []

    for sentence in tqdm.tqdm(source_text.split("\n")):
        source_text_to_int.append(text_to_int(sentence, source_vocab_to_int, 20,
                                              is_target=False))
    # 对目标句子进行转换  Ty = 25
    target_text_to_int = []

    for sentence in tqdm.tqdm(target_text.split("\n")):
        target_text_to_int.append(text_to_int(sentence, target_vocab_to_int, 25,
                                              is_target=True))

    # Save data for later use
    pickle.dump((
        (source_text_to_int, target_text_to_int),
        (source_vocab_to_int, target_vocab_to_int),
        (source_int_to_vocab, target_int_to_vocab)), open('data/preprocess.p', 'wb'))


source_path = 'data/small_vocab_en.txt'
target_path = 'data/small_vocab_fr.txt'
preprocess_and_save_data(source_path, target_path)