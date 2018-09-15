import os
import random
import json
from tqdm import tqdm
import jieba
import numpy as np
import unicodedata
# from udicOpenData.dictionary import *

  # "0": "__PAD__",
  # "1": "__GO__",
  # "2": "__EOS__",
  # "3": "__UNK__",
  # "4": "__NUM__",
  # "5": "__ENG__",
  # "6": "__PUN__",

padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3
jieba.load_userdict('../NameDict_Ch_v2')

class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []

#load資料，並利用word2id將sequence轉成word id像這樣[1,2,3,4,5,6]
def loadDataset(intputFile, word2idFile, Number=None):
    print("load data......")
    data = []
    sentenceCount = 0

    with open(word2idFile,'r') as w2idFile:
        w2id = json.load(w2idFile) #w2id = dict
        with open(intputFile, 'r') as infile:
            # print("load complete......")
            for line in tqdm(infile.readlines()):
                line=line.strip('\n')
                sentenceCount += 1
                sentenceTemp = []
                for word in line.split(" "):
                    if word == '\n':
                        continue
                    try:
                        sentenceTemp.append(w2id[word])
                    except Exception as e:
                        print(" ")
                        print(e)
                        print("Error: "+line)
                # print(sentenceTemp)
                data.append(sentenceTemp)
                if sentenceCount == Number:
                    break

    return data

def add_noise(sentence, prob):
        # First, omit some words
        sentence = sentence.copy()
        omit_prob = prob
        num_omissions = int(omit_prob * len(sentence))
        inds_to_omit = np.random.permutation(len(sentence))[:num_omissions]
        for i in inds_to_omit:
            sentence[i] = unknownToken
        # Second, swap some of adjacent words
        swap_prob = prob
        num_swaps = int(swap_prob * (len(sentence) - 1))
        inds_to_swap = np.random.permutation(len(sentence) - 1)[:num_swaps]
        for i in inds_to_swap:
            sentence[i], sentence[i+1] = sentence[i+1], sentence[i]
        # print("3",[sentence])
        # print(sentence)
        return sentence

def change_target(sentence, id2word, target_word2id):
    changed_sentence = []
    for i in sentence:
        # changed_sentence.append(target_word2id[id2word[str(i)]])
        changed_sentence.append(target_word2id.get(id2word[str(i)], unknownToken))
    # print("origin_target", sentence)
    # print("changed_sentence", changed_sentence)
    return changed_sentence

def createBatch(samples, prob, id2word, target_word2id, only_wuxia=False, noise=False):
    batch = Batch()
    if noise:
        inputSentences = [add_noise(sample, prob) for sample in samples]  
    else:
        inputSentences = samples

    if only_wuxia:
        targetsSentences = [change_target(sample, id2word, target_word2id) for sample in samples]
    else:
        targetsSentences = samples     

    # print("1",inputSentences)
    # print("2",targetsSentences)

    batch.encoder_inputs_length = [len(inputSentence) for inputSentence in inputSentences]
    batch.decoder_targets_length = [len(targetsSentence)+1 for targetsSentence in targetsSentences]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for sample in inputSentences:
        source = list(reversed(sample))
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source)

    for sample in targetsSentences:
        target = sample + [eosToken]
        pad = [padToken] * (max_target_length - len(target))
        batch.decoder_targets.append(target + pad)

    return batch


def getBatches(data, batch_size, prob, id2word, target_word2id, only_wuxia=False, noise=False):
    random.shuffle(data)
    data_len = len(data)

    for i in range(0, data_len, batch_size):
        batch = createBatch(data[i:min(i + batch_size, data_len)], prob, id2word, target_word2id, only_wuxia, noise)
        yield batch


def sentence2enco(sentence, word2id, prob, id2word, target_word2id, only_wuxia=False, noise=False):
    '''
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，现将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 处理之后的数据，可直接feed进模型进行预测
    '''
    # print(sentence)
    sentence = sentence.strip('\n')
    sentence = unicodedata.normalize('NFKC', sentence)
    print(sentence)
    if sentence == '':
        return None
    #分词
    tokens = list(jieba.cut(sentence, cut_all=False))
    # if len(tokens) > 20:
    #     return None
    print(tokens)
    #将每个单词转化为id
    wordIds = []
    for token in tokens:
        wordIds.append(word2id.get(token, unknownToken))
    #调用createBatch构造batch
    print("wordIds:", wordIds)
    batch = createBatch([wordIds], prob, id2word, target_word2id, only_wuxia, noise)
    return batch


