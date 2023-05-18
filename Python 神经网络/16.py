# _*_ coding:utf-8 _*_
#input("enter input:\n")
training_data_file=open("mnist_dataset/mnist_train_100.csv","r")


import pandas
import matplotlib.pyplot as plt

df=pandas.read_csv('mnist_dataset/mnist_train_100.csv',header=None)

print(df.head(6))


df.info()

row=2
data=df.iloc[row]
label=data[0]
img=data[1:].values.reshape(28,28)
plt.title("label="+str(label))
plt.imshow(img,interpolation='none',cmap='Blues')
plt.show()


























