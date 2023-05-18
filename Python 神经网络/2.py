 #coding:gbk
 # __init__()方法的前后应是两个“_” !
class Dog:
   
    def __init__(self,petname,temp):
        self.name=petname;
        self.temperature=temp;
    
    def status(self):
        print("dog name is",self.name)
        print("dog temperature is",self.temperature)
        pass
    def setTemperature(self,temp):
        self.temperature=temp;
        pass
    def bark(self):
        print("woof")
        pass
    
    pass

lassie = Dog("lassie",37)
lassie.status()

lassie.setTemperature(40)
lassie.status()
