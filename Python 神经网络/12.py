s='hello world'
print(s)
s2="Python"
print(s2)
print(s+s2)
print(s2*3)
print(len(s))
line="12 3 4 5 6 7 8 9 \n 0 0 0 0"
number=line.split( )
print(number)

s=""
t=s.join(number)
print(t)


s=":"
t=s.join(number)
print(t)


s3="hello world"
s4=s3.replace('hello', 'python')
print(s4)

print(s4.upper())
print(s4.lower())

print(dir(s4))

a1=str(1.1+2.2)
a2=repr(1.1+2.2)

a3=hex(255)
a4=oct(255)
a5=bin(255)

print(a1+a2+a3+a4+a5)

r1='{} {} {}'.format('a','b','c')
print(r1)
r2='{0} {u} {0}'.format('a',u=15,color='c')
print(r2)
import math
r3='{0:10} {1:10d} {2:10.2f}'.format('add',15,2.5)
print(r3)



