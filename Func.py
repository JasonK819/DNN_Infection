import numpy as np

def Normalization(nparray):
    item_max = np.max(nparray)
    item_min = np.min(nparray)

    for i in range(len(nparray)):
        nparray[i] = (nparray[i]-item_min)/(item_max-item_min)

    return nparray

def Calculation(result, test):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(result)):
        if result[i] == 1 and test[i] == 1:
            TP += 1
        if result[i] == 1 and test[i] == 0:
            FP += 1
        if result[i] == 0 and test[i] == 1:
            FN += 1
        if result[i] == 0 and test[i] == 0:
            TN += 1

    print('ACC:', (TP+TN) / len(result))
    print('PPV:', TP / (TP + FP))
    print('TPR:', TP / (TP + FN))
    print('TNR:', TN / (TN + FP))