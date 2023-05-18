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
    
print(our_own_dataset[4])








