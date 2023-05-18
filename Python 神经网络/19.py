# _*_ coding:utf-8 _*_
#input("enter input:\n")

import numpy
import scipy.special
import matplotlib.pyplot 
import matplotlib.pyplot as plt

wih=numpy.random.normal(0.0,pow(100,-0.5),(100,784))
who=numpy.random.normal(0.0,pow(10,-0.5),(10,100))
#print(wih)
#print(who)
#plt.hist(wih, bins=100)
#plt.show()


a=scipy.special.expit(0)

print(a)



function=lambda x:scipy.special.expit(x)
print(function(0))




