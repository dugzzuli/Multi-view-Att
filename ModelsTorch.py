import torch
from torch import nn, optim

from torch.utils.data import DataLoader
import numpy as np
from torch import nn,optim
from Database import load_data, load_3sources, myDataset, acc_val
from torchsummary import summary

class ViewAttention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(ViewAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)


        betaZ=(beta * z)
        m,n,z=betaZ.size()
        zArr=[]
        for i in range(n):
            zArr.append(betaZ[:,i,:])
        return torch.cat(zArr,1),beta
        # return (beta * z).sum(1),beta

class ViewEncoder(nn.Module):
    def __init__(self,fea_dim,layers=[128,1024,64]):
        super(ViewEncoder, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(fea_dim, layers[0]),
                                     nn.LeakyReLU(0.2,True),
                                     nn.Linear(layers[0], layers[1]),
                                     nn.LeakyReLU(0.2,True),
                                     nn.Linear(layers[1], layers[2]),
                                     )

    def forward(self, x):
        encode = self.encoder(x)
        return encode

class ViewDecoder(nn.Module):
    def __init__(self,fea_dim,layers=[128,1024,64]):
        super(ViewDecoder, self).__init__()
        self.decoder = nn.Sequential(
                                     nn.Linear(layers[2]*2, layers[1]),
                                     nn.LeakyReLU(0.2,True),
                                     nn.Linear(layers[1], layers[0]),
                                     nn.LeakyReLU(0.2,True),
                                     nn.Linear(layers[0], fea_dim)
                                    )
    def forward(self, output):
        decode = self.decoder(output)
        return decode

class HANEncoderLayer(nn.Module):
    def __init__(self,view_num,in_size,out_size,layers):
        super(HANEncoderLayer, self).__init__()

        self.gat_layers = nn.ModuleList()
        for i in range(view_num):
            self.gat_layers.append(ViewEncoder(in_size[i],layers))
        self.view_attention = ViewAttention(in_size=out_size)
        self.view_num = view_num
    def forward(self, gs):
        view_embeddings = []
        for i, g in enumerate(gs):
            view_embeddings.append(self.gat_layers[i](g))
        view_embeddings = torch.stack(view_embeddings, dim=1)  # (N, M, D * K)
        return self.view_attention(view_embeddings)

class HANDecoderLayer(nn.Module):
    def __init__(self,view_num,in_size,layers): #in_size 代表 输入样本的维度
        super(HANDecoderLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(view_num):
            self.gat_layers.append(ViewDecoder(in_size[i],layers))

        self.view_num = view_num
    def forward(self, gs):
        view_att=[]
        for i in range(self.view_num):
            view_att.append(self.gat_layers[i](gs))
        return view_att


class HAN(nn.Module):
    def __init__(self, view_num, in_size, hidden_size,layers):
        super(HAN, self).__init__()
        self.encoderLayer=HANEncoderLayer(view_num, in_size, hidden_size,layers) #默认添加一个
        self.decoderLayer=HANDecoderLayer(view_num,in_size,layers)

    def forward(self, g):
        h,beta = self.encoderLayer(g)
        att_con=self.decoderLayer(h)
        return h,att_con,beta

# 自定义损失函数
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.sum(torch.pow((x - y), 2),1))























