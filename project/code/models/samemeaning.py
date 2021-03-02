from torch import nn
from models.bert_model import *
import numpy as np
# CLS对应的一条向量
class Bert_meaning_Analysis(nn.Module):
    def __init__(self,config):
        super(Bert_meaning_Analysis, self).__init__()
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size,1024)
        self.dense = nn.Linear(1024,1)
        self.activation = nn.Sigmoid()

    def get_loss(self,predictions,labels):
        # 将预测和标记的维度展平, 防止出现维度不一致
        # print('预测的：',predictions,type(predictions))
        # print('标签:',labels,type(labels))
        predictions = predictions.view(-1)
        labels = labels.float().view(-1)
        epsilon = 1e-8
        # 交叉熵
        loss = - labels * torch.log(predictions + epsilon) - (torch.tensor(1.0) - labels) * torch.log(torch.tensor(1.0) - predictions + epsilon)
        # 求均值, 并返回可以反传的loss
        # loss为一个实数
        loss = torch.mean(loss)
        return loss

    def forward(self,text,position,lables=None):
        hidden_layers,_ = self.bert(text,position,output_all_encoded_layers=True)
        # hidden_layers2,_ = self.bert(text2,position2,output_all_encoded_layers=True)
        sequence_output = hidden_layers[-1]
        # sequence_output2 = hidden_layers1[-1]
        cls_info = sequence_output[:,0]
        # cls_info2 = sequence_output2[:, 0]

        #计算cosine相似度

        predictions = self.linear(cls_info)
        predictions = self.dense(predictions)
        # predictions2 = self.dense(cls_info2)
        predictions = self.activation(predictions)
        # predictions2 = self.activation(predictions2)
        # predictions = self.cosine_similarity(predictions1,predictions2)
        # predictions = torch.from_numpy(predictions).to("cuda:0")
        if lables is not None:
            loss = self.get_loss(predictions,lables)
            return predictions,loss
        else:
            return predictions

    def cosine_similarity(self,x, y, norm=False):
        """ 计算两个向量x和y的余弦相似度 """
        assert len(x) == len(y), "len(x) != len(y)"
        zero_list = [0] * len(x)
        if x == zero_list or y == zero_list:
            return float(1) if x == y else float(0)

        # method 1
        x = x.data.cpu().numpy()
        y = y.data.cpu().numpy()
        res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        # 越接近1 越不一样 所以加一个1-
        return 1 -(0.5 * cos + 0.5) if norm else cos  # 归一化到[0, 1]区间内

