from torch.utils.data import Dataset
import tqdm
import json
import torch
import random
import numpy as np
from sklearn.utils import shuffle
import re
import pandas as pd

class SentimentDataset(Dataset):
    def __init__(self, corpus_path, word2idx, max_seq_len, data_regularization=False):
        self.data_regularization = data_regularization
        self.word2idx = word2idx
        # define max length
        self.max_seq_len = max_seq_len
        # directory of corpus dataset
        self.corpus_path = corpus_path
        # define special symbols
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4
        self.num_index = 5
        self.lines = []
        # 加载语料
        globals = {
            'nan': self.unk_index
        }
        with open(corpus_path, "r", encoding="utf-8") as f:

            self.lines = [eval(line, globals) for line in tqdm.tqdm(f, desc="Loading Dataset")]
            # 打乱顺序
            # self.lines = shuffle(self.lines)
            # 获取数据长度(条数)
            self.corpus_length = len(self.lines)

    def __len__(self):
        return self.corpus_length

    def __getitem__(self, item):
        # text1,text2,label = self.lines[item]['text1'],self.lines[item]['text2'],self.lines[item]['label']
        # 预测
        text1, text2, label = self.lines[item]['text1'], self.lines[item]['text2'],0

        text_input1 = [self.word2idx.get(char,self.unk_index) for char in str(text1)]

        text_input2 = [self.word2idx.get(char,self.unk_index) for char in str(text2)]
        # add cls and sep 和截断

        text_input1 = ([self.cls_index] + text_input1)[:self.max_seq_len +1]
        text_input2 = (text_input2 + [self.sep_index])[:self.max_seq_len +1]
        text_final = text_input1 + [self.sep_index] + text_input2 + [self.sep_index]
        text_final = text_final[:self.max_seq_len]
        output = {
            'text_input':torch.tensor(text_final),
            'label':torch.tensor([label])
        }

        return output