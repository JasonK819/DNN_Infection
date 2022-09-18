import numpy as np
import pandas as pd

data = np.loadtxt("Pima.csv",encoding="utf-8", delimiter=",")

np_Data_train = np.split(data, [614,], 0)[0]
np_Data_test = np.split(data, [614,], 0)[1]
# print(np_Data_train)

pd_data_train = pd.DataFrame(np_Data_train, columns=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])
pd_data_test = pd.DataFrame(np_Data_test, columns=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])
# print(pd_data_train)
# print(pd_data_test)

#保存于工程Dataset目录下
pd_data_train.to_csv(r'D:\codeProject\PycharmProject\DNN_Infection\Dataset\train.csv', header=0, index=0)
pd_data_test.to_csv(r'D:\codeProject\PycharmProject\DNN_Infection\Dataset\test.csv', header=0, index=0)

