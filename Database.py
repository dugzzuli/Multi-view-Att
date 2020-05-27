import numpy as np
from sklearn import datasets
import scipy.io as sio
import torch
from torch.utils.data import DataLoader,Dataset

from sklearn import metrics

def acc_val(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def load_data():
    iris = datasets.load_boston()

    print(iris.data.shape)

    print(iris.target.shape)
    return iris.data,iris.target

def load_3sources():
    dataMat=sio.loadmat('Dataset/3sources.mat')
    data=dataMat['data']
    target=dataMat['truelabel']
    dataR=[]
    for i in range(data.size):
        dataR.append(np.transpose(data[0, i]))
    return dataR,target[0,0]

def load_BBCSport():
    dataMat=sio.loadmat('Dataset/BBCSport.mat')
    data=dataMat['data']
    target=dataMat['truelabel']
    dataR=[]
    for i in range(data.size):
        dataR.append(np.transpose(data[0, i]))
    return dataR,target[0,0]

def load_BBC():
    dataMat=sio.loadmat('Dataset/BBC.mat')
    data=dataMat['data']
    target=dataMat['truelabel']
    dataR=[]
    for i in range(data.size):
        dataR.append(np.transpose(data[0, i]))
    return dataR,target[0,0]

class myDataset(Dataset):
    def __init__(self, dataSource):
        self.dataSource = dataSource

    def __getitem__(self,index):
        element = self.dataSource[index]
        return element
    def __len__(self):
        return len(self.dataSource)


class MyDataset2Viewer(Dataset):
    def __init__(self, data1,data2,labels):
        self.data1 = data1
        self.data2 = data2
        self.labels = np.transpose(labels)

    def __getitem__(self, index):
        d1,d2,labels = self.data1[index],self.data2[index],self.labels[index]
        return d1,d2,labels

    def __len__(self):
        return len(self.data1)

if __name__ == "__main__":
    data,target=load_3sources()

    train_data=MyDataset2Viewer(data[0],data[1],target)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=13, drop_last=True)

    for d in train_loader:
        print(d.float().size())


    # datall=zip(data[0],data[1])
    #
    # train_data = myDataset(datall)
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=13, drop_last=True)
    #
    # for single in train_loader:
    #     print(np.shape(single))

    #
    # train_data = myDataset(data[0])
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=13, drop_last=True)
    #
    # train_data1 = myDataset(data[1])
    # train_loader1 = DataLoader(train_data1, shuffle=False, batch_size=13, drop_last=True)
    #
    # for single,single1 in zip(train_loader,train_loader1):
    #     print(np.shape(single),np.shape(single1))
    #

