# _*_ coding:utf-8 _*_
#input("enter input:\n")

import numpy
import scipy.special
import matplotlib.pyplot 

wih=numpy.random.normal(0.0,pow(100,-0.5),(100,784))
who=numpy.random.normal(0.0,pow(10,-0.5),(10,100))
print(wih)
print(who)


w1=numpy.random.normal(0,pow(10,-0.5),[1,10])
print(w1)

w2=pow(10,-0.5)
print(w2)


w3=numpy.random.normal(0,1)
print(w3)

import random
for i in range(5):
    print(random.random())
    print(random.uniform(10,20))
    print(random.randint(100,200))
    print(random.randrange(100,1000,2))





from numpy import random
 
x = random.normal(loc=0, scale=2, size=(2, 3))
 
print(x)

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
 
sns.distplot(random.normal(size=1000), hist=False)
 
plt.show()

import numpy as np
mu, sigma = 0, 0.1      # 均值和标准差
s = np.random.normal(mu, sigma, 1000)
abs(mu - np.mean(s)) < 0.01
abs(sigma - np.std(s, ddof=1)) < 0.01
import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, normed=True, color='b')
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import math


def func_normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)


mean1, sigma1 = 2,0.5
x1 = np.linspace(mean1 - 6*sigma1, mean1 + 6*sigma1, 100)

mean2, sigma2 = 2,1
x2 = np.linspace(mean2 - 6*sigma2, mean2 + 6*sigma2, 100)

mean3, sigma3 = 3,1
x3 = np.linspace(mean3 - 6*sigma3, mean3 + 6*sigma3, 100)

y1 = func_normal_distribution(x1, mean1, sigma1)
y2 = func_normal_distribution(x2, mean2, sigma2)
y3 = func_normal_distribution(x3, mean3, sigma3)

plt.plot(x1, y1, 'r', label='m=2,sig=0.5')
plt.plot(x2, y2, 'g', label='m=2,sig=1')
plt.plot(x3, y3, 'b', label='m=3,sig=1')
plt.legend()
plt.grid()
plt.show()
