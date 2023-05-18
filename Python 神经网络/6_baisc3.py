import numpy
import scipy.special
import matplotlib.pyplot 
import scipy.ndimage

class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learninggrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learninggrate
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function=lambda x:scipy.special.expit(x)
        self.inverse_activation_function=lambda x:scipy.special.logit(x)
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
    def backquery(self,targets_list):
        final_outputs=numpy.array(targets_list,ndmin=2).T
        final_inputs=self.inverse_activation_function(final_outputs)
        hidden_outputs=numpy.dot(self.who.T,final_inputs)
        hidden_outputs-=numpy.min(hidden_outputs)
        hidden_outputs/=numpy.max(hidden_outputs)
        hidden_outputs*=0.98
        hidden_outputs+=0.01
        hidden_inputs=self.inverse_activation_function(hidden_outputs)
        inputs=numpy.dot(self.wih.T,hidden_inputs)
        inputs-=numpy.min(inputs)
        inputs/=numpy.max(inputs)
        inputs*=0.98
        inputs+=0.01
        return inputs
        
    
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
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)
        
        
        
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


label=7
targets=numpy.zeros(output_nodes)+0.01
targets[label]=0.99
print(targets)
image_data=n.backquery(targets)
matplotlib.pyplot.imshow(image_data.reshape(28,28),cmap='Greys', interpolation='None')
matplotlib.pyplot.show()
