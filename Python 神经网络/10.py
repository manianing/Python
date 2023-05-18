import numpy
import scipy.special
import matplotlib.pyplot 
import time



train_data_file=open("mnist_dataset/2.txt","r")
train_data_list=train_data_file.readlines()
train_data_file.close()

for record in train_data_list:
    all_values=record.split(",")
    #time.sleep(3)
    print(all_values)

    
pass



records = ['4月01日 17:00 IG PK RA','4月02日 17:00 苏州LGN PK SN','4月03日 17:00 FPX PK RA','4月04日 17:00 SN PK 西安WE','4月05日 17:00 北京JDG PK FPX','4月06日 17:00 SN PK TES'] #作者：千锋python https://www.bilibili.com/read/cv11546089 出处：bilibili
records2 = ['2,3,4,5,6','3,2,3,3','4','5','6','7']
stream=open("1.txt",mode='w')

for record in records:
    print(record)
    stream.write(record+'\n')
    stream.write("god"+'\n')
pass
stream.close()


stream=open("1.txt",mode='r')
content=stream.read()
print(content)
stream.close()


stream=None
try:
    stream=open("1.txt","r")
    content=stream.read()
    print(content)
except:
    print("ng file not find")
finally:
    if stream!=None:
        stream.close()


with open("1.txt","r") as stream:
    content=stream.read()
    print(content)
    print(stream.closed)
    
    
try:
    with open("1.txt","r") as stream:
        content=stream.read(5)
        print(content)
        stream.seek(3,0)
        content=stream.read()
        print(content)
except:
    print("error")
    print(stream.closed)
    
    
    
    
    
    
    
'''
stream.close()
stream=open("1.txt",mode='w')
stream.write("god")
'''


