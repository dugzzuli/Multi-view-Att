import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from torch import nn,optim

from Database import load_data,load_3sources



class AutoEncoder(nn.Module):
    def __init__(self,input_dim):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(input_dim, 256),
                                     nn.LeakyReLU(0.2,True),
                                     nn.Linear(256, 64),
                                     nn.LeakyReLU(0.2, True),
                                     nn.Linear(64, 16))


        self.decoder = nn.Sequential(nn.Linear(16, 64),
                                     nn.LeakyReLU(0.2,True),
                                     nn.Linear(64, 256),
                                     nn.LeakyReLU(0.2,True),
                                     nn.Linear(256, input_dim)
                                    )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class DANEMV(nn.Module):
    def __init__(self,dims):
        super(DANEMV, self).__init__()

        self.viewsList=nn.ModuleList()
        for dim in dims:
            self.viewsList.append(AutoEncoder(dim))
    def forward(self, gs):
        encodeList=[]
        decodeList = []
        for i, g in enumerate(gs):
            encode, decode=self.viewsList[i](g)
            encodeList.append(encode)
            decodeList.append(decode)
        return encodeList,decodeList

if __name__ == "__main__":
    # data,target=load_data()

    data,target=load_3sources()
    print(np.shape(data))
    print(np.shape(target))

    batch_size=128
    lr=1e-3
    weight_decay=1e-5
    epoches=500

    model=()

    criterion=nn.MSELoss()

    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        print("===")
        model.cuda()

    train_loader = DataLoader(data, shuffle=True, batch_size=batch_size, drop_last=True)

    for epoch in range(epoches):
        if epoch in [epoches * 0.25, epoches * 0.5]:
            for param_group in optimizier.param_groups:
                param_group['lr'] *= 0.1

        for single in train_loader:
            # print(single)

            _, output = model(single.float().cuda())
            loss = criterion(output,single.float().cuda())

            # backward
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
        print("epoch=", epoch, loss.data.float())




        for param_group in optimizier.param_groups:
            print(param_group['lr'])
















