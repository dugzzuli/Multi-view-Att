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

        return (beta * z).sum(1),beta

class ViewEncoder(nn.Module):
    def __init__(self,fea_dim,layers=[512,128,1024,64]):
        super(ViewEncoder, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(fea_dim, layers[0]),
                                     nn.ReLU(True),
                                     nn.Linear(layers[0], layers[1]),
                                     nn.ReLU(True),
                                     nn.Linear(layers[1], layers[2]),
                                     nn.ReLU(True),
                                     nn.Linear(layers[2], layers[3]))

    def forward(self, x):
        encode = self.encoder(x)
        return encode

class ViewDecoder(nn.Module):
    def __init__(self,fea_dim,layers=[512,128,1024,64]):
        super(ViewDecoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(layers[3], layers[2]),
                                     nn.ReLU(True),
                                     nn.Linear(layers[2], layers[1]),
                                     nn.ReLU(True),
                                     nn.Linear(layers[1], layers[0]),
                                     nn.ReLU(True),
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

if __name__ == "__main__":

    data,target=load_3sources()
    clustering = len(np.unique(target))
    batch_size = len(target)

    lr = 1e-4
    weight_decay = 1e-5
    epoches = 500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layers = [1024, 512, 128, 32]

    dims = [np.shape(d)[1] for d in data]

    model = HAN(len(dims), dims, layers[-1], layers=layers)

    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        print('using...cuda')
        model.cuda()

    train_data = myDataset(data[0])
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)

    train_data1 = myDataset(data[1])
    train_loader1 = DataLoader(train_data1, shuffle=False, batch_size=batch_size, drop_last=True)

    train_data2= myDataset(data[2])
    train_loader2 = DataLoader(train_data2, shuffle=False, batch_size=batch_size, drop_last=True)


    for epoch in range(epoches):
        if epoch in [epoches * 0.25, epoches * 0.5]:
            for param_group in optimizier.param_groups:
                param_group['lr'] *= 0.1
        for single,single1,single2 in zip(train_loader,train_loader1,train_loader2):
            h,output,beta=model([single.float().cuda(),single1.float().cuda(),single2.float().cuda()])
            criterion = nn.MSELoss()
            loss=0
            # for i,g in enumerate(output):
            loss+=criterion(output[0],single.float().cuda())
            loss += criterion(output[1], single1.float().cuda())
            loss += criterion(output[2], single2.float().cuda())

            optimizier.zero_grad()
            loss.backward()
            optimizier.step()

        if(epoch%10==0):
            for param_group in optimizier.param_groups:
                print(param_group['lr'])
        print("epoch=", epoch, loss.data.float())



    # model.eval()
    print(h.detach().cpu().numpy().shape)

    low_repre=h.detach().cpu().numpy()

    from sklearn import metrics
    from sklearn.cluster import KMeans

    cluster = KMeans(n_clusters=6, random_state=0,init='k-means++').fit(low_repre)

    nmi = metrics.normalized_mutual_info_score(cluster.labels_, np.reshape(target, -1))
    print("nmi",nmi)
    ac = acc_val(cluster.labels_, np.reshape(target, -1))
    print("ac:",ac)






























