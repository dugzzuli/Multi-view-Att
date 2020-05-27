import torch
from torch import nn, optim

from torch.utils.data import DataLoader
import numpy as np
from torch import nn,optim
from Database import load_data, load_3sources, myDataset, acc_val,MyDataset2Viewer,load_BBCSport
from torchsummary import summary

from ModelsTorch import *
import matplotlib.pyplot as plt




if __name__ == "__main__":

    data,target=load_BBCSport()
    clustering=len(np.unique(target))
    batch_size=np.shape(target)[1]


    lr = 1e-4


    weight_decay = 1e-5

    epoches = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    layers = [256, 64, 16]

    dims = [np.shape(d)[1] for d in data]

    model = HAN(len(dims), dims, layers[-1], layers=layers)

    criterion = My_loss()

    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizier = optim.Adam(model.parameters(), lr=lr)

    if torch.cuda.is_available():
        print('using...cuda')
        model.cuda()
    print(model)

    train_data = MyDataset2Viewer(data[0],data[1],target)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)

    loss_dict = []
    for epoch in range(epoches):
        if epoch in [epoches * 0.25, epoches * 0.5]:
            for param_group in optimizier.param_groups:
                param_group['lr'] *= 0.1


        for single,single1,l in train_loader:
            h,output,beta=model([single.float().cuda(),single1.float().cuda()])

            loss=0
            # for i,g in enumerate(output):
            loss+=criterion(output[0],single.float().cuda())
            loss += criterion(output[1], single1.float().cuda())

            l1_regularization, l2_regularization = 0,0  # 定义L1及L2正则化损失

            for param in model.parameters():
                l1_regularization += torch.norm(param, 1)  # L1正则化
                l2_regularization += torch.norm(param, 2)  # L2 正则化

            loss += weight_decay*l1_regularization
            loss += weight_decay*l2_regularization


            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
        loss_dict.append(loss.item())

        if(epoch%10==0):
            for param_group in optimizier.param_groups:
                print(param_group['lr'])
            print("epoch=", epoch, loss.data.float())



    # model.eval()
    print(h.detach().cpu().numpy().shape)

    low_repre=h.detach().cpu().numpy()

    from sklearn import metrics
    from sklearn.cluster import KMeans

    cluster = KMeans(n_clusters=clustering, random_state=0,init='k-means++').fit(low_repre)

    nmi = metrics.normalized_mutual_info_score(cluster.labels_, np.reshape(target, -1))
    print("nmi",nmi)
    ac = acc_val(cluster.labels_, np.reshape(target, -1))
    print("ac:",ac)

    # 画loss在迭代过程中的变化情况
    plt.plot(loss_dict, label='loss for every epoch')
    plt.legend()
    plt.show()






























