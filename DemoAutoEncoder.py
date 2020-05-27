import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from torch import nn,optim
from Database import load_data,load_3sources


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(13, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 3))


        self.decoder = nn.Sequential(nn.Linear(3, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 13)
                                    )
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


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
















