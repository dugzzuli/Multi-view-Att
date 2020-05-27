import torch
from torch import nn, optim

from torch.utils.data import DataLoader
import numpy as np
from torch import nn,optim
from Database import load_data, load_3sources, myDataset, acc_val,load_BBC
from torchsummary import summary
from ModelsTorch import *

if __name__ == "__main__":

    data,target=load_BBC()
    clustering = len(np.unique(target))
    batch_size = len(target)
    lr = 1e-4
    weight_decay = 1e-5
    epoches = 500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    layers = [500, 500, 2000, 32]

    dims = [np.shape(d)[1] for d in data]

    model=HAN(len(dims),dims,layers[-1],layers=layers)

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

    train_data3 = myDataset(data[3])
    train_loader3 = DataLoader(train_data3, shuffle=False, batch_size=batch_size, drop_last=True)


    for epoch in range(epoches):
        if epoch in [epoches * 0.25, epoches * 0.5]:
            for param_group in optimizier.param_groups:
                param_group['lr'] *= 0.1
        for single,single1,single2,single3 in zip(train_loader,train_loader1,train_loader2,train_loader3):
            h,output,beta=model([single.float().cuda(),single1.float().cuda(),single2.float().cuda(),single3.float().cuda()])
            criterion = nn.MSELoss()
            loss=0
            # for i,g in enumerate(output):
            loss+=criterion(output[0],single.float().cuda())
            loss += criterion(output[1], single1.float().cuda())
            loss += criterion(output[2], single2.float().cuda())
            loss += criterion(output[3], single3.float().cuda())

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

    cluster = KMeans(n_clusters=5, random_state=0,init='k-means++').fit(low_repre)

    nmi = metrics.normalized_mutual_info_score(cluster.labels_, np.reshape(target, -1))
    print("nmi",nmi)
    ac = acc_val(cluster.labels_, np.reshape(target, -1))
    print("ac:",ac)






























