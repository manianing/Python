from functools import total_ordering

s="""hello
world"""
print(s)


a=2+2
print(a)

s1="hello"
s2='hello'
print(s1+s2)
print(s[0])
print(s[-1])
print(s[0:4])


s3="hello world"
s4=s3.split( )
print(s4)



s5="hello,world"
s6=s5.split(",")
print(s6)

print(len(s5))


a=[1,2,3,4,5,'hello',6.0]
print(a*2)
print(a+a)
print(a[-1])
print(len(a))

a.append("world")
print(a)

s={1,3,4,5}
print(s)
print(len(s))
s.add(1)
print(s)
s.add(2)
print(s)
s.add(8)
print(s)

b={2,3,5}
print(s&b)
print(s|b)
print(s-b)
print(s^b)


dic={1:"dog",2:'cat'}
print(dic)
print(dic[1])
dic[3]="pig"
print(dic)
print(dic.keys())
print(dic.values())
print(dic.items())


import numpy
from numpy import array, arange
array1=array([1,2,3,4])
print(array1)
print(array1*2)

from matplotlib.pyplot  import plot
import matplotlib.pyplot as pt
pt.plot(array1,array1+2)
pt.show()


line='1 2 3 4 5'
fields=line.split()
print(fields)

total=0
for field in fields:
    total+=int(field)
print(total)


numbers=[int(field) for field in fields]
numbers
print(sum(numbers))


w=sum([int(field) for field in line.split()])
print(w)

fc=open('1.txt',"w")
fc.write("1,2,3,4\n")
fc.write('2,3,4\n')
fc.close()

fo=open('1.txt','r')
data=[]
for record in fo:
    data.append([int(field) for field in record.split(',')])
fo.close()
print(data)

fo=open('1.txt','r')
for line in fo.readlines():
    data.append([int(field) for field in line.split(',')])
fo.close()
print(data)


for row in data:
    print(row)

def poly(x,a,b,c):
    y=a*x**2+b*x+c
    return y

x=1
print(poly(x,1,2,3))

x=array([1,2,3])
print(poly(x,1,2,3))



def poly2(x,a=1,b=2,c=3):
    y=a*x**2+b*x+c
    return y

x=arange(10)

x=array([0,1,2,3,4])
poly2(x)
print(poly2(x))


import  os
print(os.getpid())

print(os.sep)


class Person(object):
    def __init__(self,first,last,age):
        self.first=first
        self.last=last
        self.age=age
    def full_Name(self):
        return print(self.first+" "+self.last)
    
p1=Person("ma","nian",30)
p1.full_Name()


p1.animal=dic
print(p1.animal)





print(type(p1))

import sys
print(sys.maxsize)


abs(-21)
round(21.6)
print( max(2,4,5))
print(min(3,4,6))


print(1e-2)
print(0xff)
print(0o10)
print(0b111)
