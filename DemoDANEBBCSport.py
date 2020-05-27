import torch
from torch import nn, optim

from torch.utils.data import DataLoader
import numpy as np
from torch import nn,optim
from Database import load_data, load_3sources, myDataset, acc_val,MyDataset2Viewer,load_BBCSport
from torchsummary import summary
from AEDemo.DANEModel import  AutoEncoder,DANEMV
import matplotlib.pyplot as plt
from ModelsTorch import My_loss



if __name__ == "__main__":

    data,target=load_BBCSport()
    clustering=len(np.unique(target))
    batch_size=np.shape(target)[1]
    lr = 1e-4
    weight_decay = 1e-5
    epoches = 1000
    dims = [np.shape(d)[1] for d in data]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = MyDataset2Viewer(data[0], data[1], target)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)

    model=DANEMV(dims)
    if torch.cuda.is_available():
        print('using...cuda')
        model.cuda()
    # print(model)
    criterion = My_loss()

    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_dict=[]
    for epoch in range(epoches):
        if epoch in [epoches * 0.25, epoches * 0.5]:
            for param_group in optimizier.param_groups:
                param_group['lr'] *= 0.1

        for single, single1, l in train_loader:
            encodeList,decodeList = model([single.float().cuda(), single1.float().cuda()])

            loss=0
            # for decoderS in decodeList:
            loss+=criterion(decodeList[0],single.float().cuda())
            loss+=criterion(decodeList[1], single1.float().cuda())
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
        if(epoch%10==0):
            print("epoch=", epoch, loss.data.float())

        loss_dict.append(loss.item())


    concatH=torch.cat(encodeList,1)
    low_repre = concatH.detach().cpu().numpy()

    from sklearn import metrics
    from sklearn.cluster import KMeans

    cluster = KMeans(n_clusters=clustering, random_state=0, init='k-means++').fit(low_repre)

    nmi = metrics.normalized_mutual_info_score(cluster.labels_, np.reshape(target, -1))
    print("nmi", nmi)
    ac = acc_val(cluster.labels_, np.reshape(target, -1))
    print("ac:", ac)

    # 画loss在迭代过程中的变化情况
    plt.plot(loss_dict, label='loss for every epoch')
    plt.legend()
    plt.show()

    print(model)








































