import numpy


from typing_extensions import Self
from numpy import put
class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learninggrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        
        self.lr=learninggrate
        pass
    def train(self):
        pass
    def query(self):
        pass

input_nodes=3
hidden_nodes=3
output_nodes=3
learning_rate=0.3


n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
n.query(self,[1.0,0.5,-1.5])

#a=numpy.random.rand(3,3)
#print(a)
#self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
#self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))



