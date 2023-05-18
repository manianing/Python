# _*_ coding:utf-8 _*_

#input("enter input:\n")
import sys;x='woerd'; sys.stdout.write(x+'\n')
from _ast import Is
x='a'
y='b'
print(x)
print(y)

print(x,end=' ')
print(y,end=' ')

print( x, y)


num=100 # reference data
weight=100
name="ShowAi"
print(num)
print(weight)
print(name)

x=y=w=10
print(x,y,w)

a,b,c,=1,2,"withing"
print(a,b,c)

num1=1
num2=10

del num1,num2

a1=20
b=0.2
pi=3.24j

S="helloAi"
print(S[1:3])
print(S*2)
print(S[1:6:2])
print(S[-3:-1])

t=['s','h','o','w','M','e','A','i']
print(t[1: ])

t2=['123','show']

print(t)
print(t2)
print(t+t2)

t3=('good','list','hello')
t4=("ing","study")
print(t4+t3)

t[2]=200
print(t)

dit={}
dit['one']="this is one"
dit[2]="this is two"
dit[3]={'name':'show','code':3466,'dept':200}
dit3={1:'show',2:3466,3:200}
tinydict = {'name': 'ShowMeAI','code':3456, 'dept': 'AI'}
print(dit['one'])
print(dit[2])
print(dit3)
print(dit3.values())
print(dit3.keys())
print(tinydict) # 输出完整的字典
print(tinydict.keys()) # 输出所有键
print(tinydict.values()) # 输出所有值



a1=10
a2=3
c=a1+a2
print(c)
c=a1-a2
print(c)
c=a1*a2
print(c)
c=a1/a2
print(c)
c=a1%a2
print(c)
c=a1**a2
print(c)

if a1==a2:
    print("a1 ="  "a2")
else:
    print("a1!=" "a2")
a=20
b=20 

if(a is b):
    print("aisb")
else:
    print("aisnot b")

if(a is not b):
    print("a is not b")
else:
    print("a is b")


print(id(a))
print(id(b))

list=[1,20,30,40]
if(a in list):
    print("in")
c=10+10+0
print(id(c))    
    
    
m=3
if(m>10):
    print("10")
elif(m>5): 
    print("5")
elif(m>0):
    print("0")
       
    
    
    
    