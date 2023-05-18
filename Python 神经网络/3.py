######################数据结构部分#############################################
import numpy as np
import matplotlib.pyplot  as  plt

#    %matplotlib inline

class Node(object):
    def __init__(self, inbound_nodes = []):
        self.inbound_nodes = inbound_nodes
        self.value = None
        self.outbound_nodes = []
         
        self.gradients = {}
         
        for node in inbound_nodes:
             node.outbound_nodes.append(self)
             
    def forward(self):
         raise NotImplementedError
         
    def backward(self):
         raise NotImplementedError
         
         
class Input(Node):
     def __init__(self):
         Node.__init__(self)
         
     def forward(self):
         pass
         
     def backward(self):
         self.gradients = {self : 0}
         for n in self.outbound_nodes:
             self.gradients[self] += n.gradients[self]
             
 ##################################################################################
class Linear(Node):
     def __init__(self, X, W, b):
         Node.__init__(self, [X, W, b])
         
     def forward(self):
         X = self.inbound_nodes[0].value
         W = self.inbound_nodes[1].value
         b = self.inbound_nodes[2].value
         self.value = np.dot(X, W) + b 
         
     def backward(self):
         self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes }
         for n in self.outbound_nodes:
             grad_cost = n.gradients[self]
             self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
             self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
             self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis = 0, keepdims = False)
             
 ###################################################################################
class Sigmoid(Node):
     def __init__(self, node):
         Node.__init__(self, [node])
         
     def _sigmoid(self, x):
         return 1. / (1. +  np.exp(-x))    #exp() 方法返回x的指数,e的x次幂
         
     def forward(self):
         input_value = self.inbound_nodes[0].value
         self.value = self._sigmoid(input_value)
         
     def backward(self):
         self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
         for n in self.outbound_nodes:
             grad_cost = n.gradients[self]
             sigmoid = self.value 
             self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost
             
 
class MSE(Node):
     def __init__(self, y, a):
         Node.__init__(self, [y, a])
         
         
     def forward(self):
         y = self.inbound_nodes[0].value.reshape(-1, 1)
         a = self.inbound_nodes[1].value.reshape(-1, 1)
         
         self.m = self.inbound_nodes[0].value.shape[0]
         self.diff = y - a 
         self.value = np.mean(self.diff**2)
         
         
     def backward(self):
         self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff 
         self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff
         
         
         
 ##########################计算图部分#############################################
def topological_sort(feed_dict):
     input_nodes = [n for n in feed_dict.keys()]
     G = {}
     nodes = [n for n in input_nodes]
     while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in' : set(), 'out' : set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in' : set(), 'out' : set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)
            
     L = []
     S = set(input_nodes)
     while len(S) > 0 :
        n = S.pop()
        if isinstance(n, Input):
            n.value = feed_dict[n]
        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if len(G[m]['in'])  == 0 :
                S.add(m)
     return L 
        
            
    
#######################使用方法##############################################
#首先由图的定义执行顺序
 #graph  = topological_sort(feed_dict)
def forward_and_backward(graph):
    for n in graph :
        n.forward()
        
    for n in graph[:: -1]:
        n.backward()
        
#对各个模块进行正向计算和反向求导
 #forward_and_backward(graph)

#########################介绍梯度下降################
def sgd_update(trainables, learning_rate = 1e-2):
    for t in trainables :
        t.value = t.value - learning_rate * t.gradients[t]
        
###########使用这个模型#################################
    from sklearn.utils import resample
        from sklearn import datasets

#   %matplotlib inline

        data = datasets.load_iris()
        X_  = data.data 
        y_  = data.target  
        y_[y_ == 2] = 1             # 0 for virginica, 1 for not virginica
    print(X_.shape, y_.shape)   # out (150,4) (150,)

########################用写的模块来定义这个神经网络#########################

np.random.seed(0)
n_features = X_.shape[1]
n_class = 1
n_hidden = 3

X, y = Input(), Input()
W1, b1  = Input(), Input()
W2, b2  = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
t1 = Sigmoid(l2)
cost = MSE(y, t1)


###########训练模型###########################################
#随即初始化参数值
W1_0 = np.random.random(X_.shape[1] * n_hidden).reshape([X_.shape[1], n_hidden])
W2_0 = np.random.random(n_hidden * n_class).reshape([n_hidden, n_class])
b1_0 = np.random.random(n_hidden)
b2_0 = np.random.random(n_class)

#将输入值带入算子
feed_dict = {
    X: X_, y: y_,
    W1:W1_0, b1: b1_0,
    W2:W2_0, b2: b2_0
}

#训练参数
#这里训练100轮(eprochs),每轮抽4个样本(batch_size),训练150/4次(steps_per_eproch),学习率 0.1
epochs = 100
m = X_.shape[0]
batch_size = 4
steps_per_eproch = m // batch_size
lr = 0.1

graph = topological_sort(feed_dict)
trainables = [W1, b1,W2, b2]

l_Mat_W1 = [W1_0]
l_Mat_W2 = [W2_0]

l_loss = []
for i in range(epochs):
    loss = 0
    for j in range(steps_per_eproch):
        X_batch, y_batch = resample(X_, y_, n_samples = batch_size)
        X.value = X_batch
        y.value = y_batch
        
        forward_and_backward(graph)
        sgd_update(trainables, lr)
        loss += graph[-1].value
        
    l_loss.append(loss)
    if i % 10 ==9 :
        print("Eproch %d, Loss = %1.5f" % (i, loss))
        
        
#图形化显示
plt.plot(l_loss)
plt.title("Cross Entropy value")
plt.xlabel("Eproch")
plt.ylabel("Loss")
plt.show()


##########最后用模型预测所有的数据的情况
X.value = X_
y.value = y_
for n in  graph:
    n.forward()
    
    
plt.plot(graph[-2].value.ravel())
plt.title("predict for all 150 Iris data")
plt.xlabel("Sample ID")
plt.ylabel("Probability for not a virginica")
plt.show()