import configparser
import os
import json
import tqdm
from dataset.twosentences import SentimentDataset
from models.samemeaning import *
from torch.utils.data import DataLoader
import torch
import pandas as pd

class Meaning_Analysis:
    def __init__(self, max_seq_len,
                 batch_size,
                 with_cuda=True, # 是否使用GPU, 如未找到GPU, 则自动切换CPU
                 ):
        config_ = configparser.ConfigParser()
        config_.read("./config/meaning_model_config.ini")
        self.config = config_["DEFAULT"]
        self.vocab_size = int(self.config["vocab_size"])
        self.batch_size = batch_size

        # 加载字典
        with open(self.config["word2idx_path_for_apply"], "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        # 判断是否有可用GPU
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        # 允许的最大序列长度
        self.max_seq_len = max_seq_len
        # 定义模型超参数
        bertconfig = BertConfig(vocab_size=self.vocab_size)
        # 初始化BERT情感分析模型
        self.bert_model = Bert_meaning_Analysis(config=bertconfig)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 开去evaluation模型, 关闭模型内部的dropout层
        self.bert_model.eval()

        # 初始化位置编码
        self.hidden_dim = bertconfig.hidden_size
        self.positional_enc = self.init_positional_encoding()
        # 扩展位置编码的维度, 留出batch维度,
        # 即positional_enc: [batch_size, embedding_dimension]
        self.positional_enc = torch.unsqueeze(self.positional_enc, dim=0)

        # 加载BERT预训练模型
        self.load_model(self.bert_model, dir_path=self.config["state_dict_dir_for_apply"])

        test_dataset = SentimentDataset(corpus_path=self.config["res_corpus_path"],
                                        word2idx=self.word2idx,
                                        max_seq_len=self.max_seq_len,
                                        data_regularization=False
                                        )
        test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          num_workers=0,
                                          collate_fn=lambda x: x)
        self.data_iter = tqdm.tqdm(enumerate(test_dataloader),
                              desc="预测",
                              total=len(test_dataloader)
                              )

    def init_positional_encoding(self):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.max_seq_len)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
        # 归一化
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc


    def load_model(self, model, dir_path="./output"):
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        checkpoint = torch.load(checkpoint_dir)
        # 情感分析模型刚开始训练的时候, 需要载入预训练的BERT,
        # 这是我们不载入模型原本用于训练Next Sentence的pooler
        # 而是重新初始化了一个
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} loaded!".format(checkpoint_dir))

    def padding(self, output_dic_lis):
        """动态padding, 以当前mini batch内最大的句长进行补齐长度"""
        text_input = [i["text_input"] for i in output_dic_lis]
        # text_input2 = [i["text_input2"] for i in output_dic_lis]
        text_input1 = torch.nn.utils.rnn.pad_sequence(text_input, batch_first=True)
        # text_input2 = torch.nn.utils.rnn.pad_sequence(text_input2, batch_first=True)
        label = torch.cat([i["label"] for i in output_dic_lis])
        return {"text_input": text_input1,
                # "text_input2": text_input2,
                "label": label}

    def __call__(self, batch_size=1):
        res = []
        with torch.no_grad():
            for i, data in self.data_iter:
                # padding

                data = self.padding(data)
                # 将数据发送到计算设备
                data = {key: value.to(self.device) for key, value in data.items()}
                # 根据padding之后文本序列的长度截取相应长度的位置编码,
                # 并发送到计算设备
                # print('长度',data["text_input"].size()[-1])
                positional_enc = self.positional_enc[:, :data["text_input"].size()[-1], :].to(self.device)
                # 正向传播, 得到预测结果
                predictions = self.bert_model.forward(text=data["text_input"],
                                                      position=positional_enc
                                                            )
                res.append(predictions)

        return res


    def find_most_recent_state_dict(self, dir_path):
        """
        :param dir_path: 存储所有模型文件的目录
        :return: 返回最新的模型文件路径, 按模型名称最后一位数进行排序
        """
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
        dic_lis = [i for i in dic_lis if "model" in i]
        # print(dic_lis)
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]


if __name__ == '__main__':
    model = Meaning_Analysis(max_seq_len=20, batch_size=1)
    res = model()
    pre = []
    for i in res:
        pre.append(float(i[0].cpu().detach().numpy()[0]))

    pre = pd.DataFrame(data=pre)
    pre.to_csv('./prediction_result/result.tsv',encoding='utf-8')
