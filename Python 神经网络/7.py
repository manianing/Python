#data_file=open("C:/Users/ma/eclipse-workspace/P4_21_01/src/1/mnist_dataset/mnist_train_100.csv","r")
from commonmark import inlines
data_file=open("mnist_dataset/mnist_train_100.csv","r")
data_list=data_file.readlines()
data_file.close()
len(data_list)
print(data_list[2])
all_value=data_list[2].split(',')
print(all_value[0])
import numpy
import matplotlib.pyplot 
#%matplotlib inlines
all_values=data_list[0].split(',')
image_array=numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()
onodes=10
targets=numpy.zeros(onodes)+0.01
targets[int(all_values[0])]=0.99
print(targets)

#scale_input=(numpy.asfarray(all_values[1:])/255*0.99)+0.01
#print(scale_input)