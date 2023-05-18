import numpy
import scipy.special
import matplotlib.pyplot 


class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learninggrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learninggrate
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function=lambda x:scipy.special.expit(x)
      
    pass
    
    def train(self,inputs_list,target_list):
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(target_list,ndmin=2).T
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        output_errors=targets-final_outputs
        hidden_errors=numpy.dot(self.who.T,output_errors)
        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))
        
        pass
    
    def query(self,inputs_list):
        inputs=numpy.array(inputs_list,ndmin=2).T
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        return final_outputs
               
        #pass
    
input_nodes=784
hidden_nodes=100
output_nodes=10
learning_rate=0.2

#n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
#data_file=open("mnist_dataset/mnist_test_10.csv","r")
training_data_file=open("mnist_dataset/mnist_train_100.csv","r")
training_data_list=training_data_file.readlines()
training_data_file.close()
#%matplotlib inlines
epochs=10
for e in range(epochs):
    for record in training_data_list:
        all_values=record.split(',')
        inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets=numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs, targets)
    pass
pass


test_data_file=open("mnist_dataset/mnist_test_10.csv","r")
#data_file=open("mnist_dataset/mnist_train_100.csv","r")
test_data_list=test_data_file.readlines()
test_data_file.close()
scorecard=[]
for record in test_data_list:
    all_values=record.split(',')
    correct_label=int(all_values[0])
    print(correct_label,"correct_label")
    inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    outputs=n.query(inputs)
    label=numpy.argmax(outputs)
    print(label,"network answer")
    if(label==correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass
pass

print(scorecard)

scorecard_array=numpy.asarray(scorecard)
print("performance=",scorecard_array.sum()/scorecard_array.size)



test_data_file=open("mnist_dataset/mnist_test_10.csv","r")
#data_file=open("mnist_dataset/mnist_train_100.csv","r")
test_data_list=test_data_file.readlines()
test_data_file.close()

all_value=test_data_list[3].split(',')
print(all_value[0])
image_array=numpy.asfarray(all_value[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()

print(n.query((numpy.asfarray(all_value[1:])/255.0*0.99)+0.01))



import imageio
import glob
import matplotlib.pyplot
from glob import glob
import numpy as np
import imageio.v2 as imageio

image_path="my_own_images/2828_my_own_?.png"
print(image_path)
records=glob(image_path)

our_own_dataset = []

for record in records:
    print(record) 
    image_array=imageio.imread(record,mode='L') 
    image_data = 255.0 - image_array.reshape(784)
    image_data = (image_data / 255.0 * 0.99) + 0.01
    label = int(record[-5:-4])
    record = np.append(label, image_data)
    our_own_dataset.append(record)
pass
    
#print(our_own_dataset[4])


for our_own_data in our_own_dataset:
    image_input = our_own_data[1:]
    output =n.query(image_input)
    image_zero = our_own_data[0]
    label = np.argmax(output)
    print(image_zero, label)
    if label == image_zero:
        print("success")
    else:
        print("fail")
pass

        
        




