import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import Func
import NetDefine
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv(r'D:\codeProject\PycharmProject\DNN_Infection\Dataset\test.csv',
                     names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

    X1 = df['preg']
    X2 = df['plas']
    X3 = df['pres']
    X4 = df['skin']
    X5 = df['test']
    X6 = df['mass']
    X7 = df['pedi']
    X8 = df['age']

    X1 = Func.Normalization(X1)
    X2 = Func.Normalization(X2)
    X3 = Func.Normalization(X3)
    X4 = Func.Normalization(X4)
    X5 = Func.Normalization(X5)
    X6 = Func.Normalization(X6)
    X7 = Func.Normalization(X7)
    X8 = Func.Normalization(X8)

    # print(X1, X2, X3, X4, X5, X6, X7, X8)

    X_df = [X1, X2, X3, X4, X5, X6, X7, X8]
    X = np.array(X_df).T
    X = X.tolist()
    X = torch.FloatTensor(X)
    # print(X)

    Y1 = df['class']

    # Y1 = Func.Normalization(Y1)

    Y_df = [Y1]
    Y = np.array(Y_df).T
    Y = Y.tolist()
    Y = torch.FloatTensor(Y)

    # print(Y)
    netD = NetDefine.TwoLayerNet(8, 10, 1)
    state_dict = torch.load('model.pt')
    netD.load_state_dict(state_dict['model'])
    input = X
    output = netD(input)
    # print(output)

    result = output.data.numpy()

    for i in range(len(result)):
        if result[i]<-0.5:
            result[i] = 0
        else:result[i] = 1

    Func.Calculation(result, Y1)

    x = np.linspace(0,153,154)

    plt.figure()
    plt.scatter(x, Y, label='class', s=10)
    plt.scatter(x, result , label='class_pre', s=10)
    plt.legend()

    plt.show()