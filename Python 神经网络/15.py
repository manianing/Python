# _*_ coding:utf-8 _*_
#input("enter input:\n")
training_data_file=open("mnist_dataset/mnist_train_100.csv","r")

import torch##
import torch.nn as nn##
from torch.utils.data import  Dataset
import pandas
import matplotlib.pyplot as plt



class MnistDataset(Dataset):
    def __init__(self,csv_file):
        self.data_df=pandas.read_csv(csv_file,header=None)
        pass
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self,index):
        label=self.data_df.iloc[index,0]
        target = torch.zeros((10))
        target[label]=1.0
        image_values = torch.FloatTensor(self.data_df.iloc[index,1:].values) / 255.0
        return label,image_values,target
        pass
    def plot_image(self,index):
        data=self.data_df.iloc[index]
        img=data[1:].values.reshape(28,28)
        plt.title("label="+str(data[0]))
        plt.imshow(img,interpolation='none',cmap='Blues')
        plt.show()
        pass
    pass

mnist_dataset=MnistDataset('mnist_dataset/mnist_train_100.csv')    
    
mnist_dataset.plot_image(2)    


class Classfier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(784,200),
            nn.Sigmoid(),
            nn.Linear(200,10),
            nn.Sigmoid()
             )
        #self.model=nn.Sequential(
        #    nn.Linear(784,200),
        #    nn.LeakyReLU(0.02),
         #   nn.Linear(200,10),
         #   nn.LeakyReLU(0.02)
         #  )
        
        self.loss_function=nn.MSELoss()
        #self.loss_function=nn.BCELoss()
        #self.optimiser=torch.optim.SGD(self.parameters(),lr=0.01)
        self.optimiser=torch.optim.Adam(self.parameters(),lr=0.01)
        self.counter=0
        self.progress=[]
        pass
    def forward(self,inputs):
        return self.model(inputs)
    def train(self,inputs,targets):
        outputs=self.forward(inputs)
        loss=self.loss_function(outputs,targets)
        self.counter+=1
        if(self.counter%10==0):
            self.progress.append(loss.item())
            pass
        if(self.counter%20==0):
            print("counter=",self.counter)
            pass
        
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass
    def plot_progress(self):
        df=pandas.DataFrame(self.progress,columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
        pass
    pass


C=Classfier()
epochs=4
for i in range(epochs):
    print('training epoch',i+1,"of",epochs)
    for label, image_data_tensor, target_tensor in mnist_dataset:
        C.train(image_data_tensor,target_tensor)
        pass
    pass
C.plot_progress()


mnist_test_dataset=MnistDataset('mnist_dataset/mnist_test_10.csv')  
record=0
mnist_test_dataset.plot_image(record)


image_data=mnist_test_dataset[record][1]
output=C.forward(image_data)
pandas.DataFrame(output.detach().numpy()).plot(kind='bar', legend=False, ylim=(0,1))
plt.show()
 
score=0
items=0
for label, image_data_tensor, target_tensor in mnist_test_dataset:
    answer = C.forward(image_data_tensor).detach().numpy()
    if (answer.argmax() == label):
        score += 1
        pass
    items+=1
    pass
print(score, items, score/items)
        














