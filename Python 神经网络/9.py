
import imageio.v2 as imageio

import numpy

#img_data=255-img_array.reshape(784)
#img_data=(img_data/255.0*0.99)+0.01


img1=imageio.imread("my_own_images/2828_my_own_6.png",mode='L') 
img_data1=255-img1.reshape(784) 
img_data=(img_data1/255.0*0.99)+0.01
data2=numpy.array(img_data).T

print(img_data1)
print(data2)


stream=open("1.txt",mode='w')

for record in img_data:

    stream.write(str(record))
pass
stream.close()

import numpy as np
np.savetxt("2.txt",img_data,fmt='%.2f',delimiter=" ",newline=',')
np.savetxt("2.csv",img1,fmt='%.2f',delimiter=",")

np.savetxt("3.txt",data2,fmt='%.2f',delimiter=",",newline=',')

import cv2
path2="my_own_images/2828_my_own_6.png"
img2=cv2.imread("my_own_images/2828_my_own_6.png")
inf=img2.shape
print(inf[2])


import numpy as np
 
# 定义要保存的数据
data = [[1, 2], [3, 4]]
 
# 使用np.savetxt函数保存数据
np.savetxt('data.txt', data, fmt='%d', delimiter=',')
print(data)

