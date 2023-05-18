from _ast import In
from commonmark import inlines
from test.test_configparser import InlineCommentStrippingTestCase
from pyglet.text.document import InlineElement


print("hello world ")

message="Good morning "

name="ada lover "
print(name.title())
print(name.upper( ))
print(name.lower())

first_name="ada"
last_name="love"
full_name=first_name+" "+last_name
print(full_name)

2*3
x=10
print(x-5)

list( range(10))
for n  in range(10):
    print(n)
    pass
print("done")

for n in range(10):
    print("the square of",n,"is",n*n)
    pass
print("done")
### the following prints out the cube of 2
print(2**3)

def avg(x,y):
    print("first input is",x)
    print("second input is",y)
    a=(x+y)/2.0
    print("average is ",a)
    return a

avg(25, 5)

import numpy
a=numpy.zeros([3,2])
print(a)
a[0,0]=1
a[0,1]=2
a[1,0]=9
a[2,1]=12
print(a)

print(a[0,1])
v=a[1,0]
print(v)

class Dog:
    def bark(self):
        print("woof")
        pass
pass

dog=Dog()
dog.bark()


