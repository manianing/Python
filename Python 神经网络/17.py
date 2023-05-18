# _*_ coding:utf-8 _*_
#input("enter input:\n")
import numpy
import scipy.special
import matplotlib.pyplot 
import random

def blockedgauss(mu,sigma):
    while True:
        numb = random.gauss(mu,sigma)
        if (numb > 0 and numb < 1):
            break
    return numb

a=blockedgauss(0, 2)
print(a)


import matplotlib.pyplot as plt
import scipy.stats as stats

lower, upper = -1, 1
mu, sigma = 0, 2
X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
N = stats.norm(loc=mu, scale=sigma)

fig, ax = plt.subplots(2, sharex=True)
ax[0].hist(X.rvs(10000), normed=True)
ax[1].hist(N.rvs(10000), normed=True)
plt.show()



import scipy.stats
lower = 0
upper = 1
mu = 0.5
sigma = 0.1
N = 10

samples = scipy.stats.truncnorm.rvs(
          (lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=N)
print(samples)


import numpy as np
import matplotlib.pyplot as plt

val_min = 0
val_max = 1
variation = (val_max - val_min)/2
std_dev = variation/3
mean = (val_max + val_min)/2
dist_normal = np.random.normal(mean, std_dev,  100)
print('Normal distribution\n\tMin: 0:.2f, Max: 1:.2f'
      .format(dist_normal.min(), dist_normal.max()))
plt.hist(dist_normal, bins=30)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

val_min = 1400
val_max = 2800
variation = (val_max - val_min)/2
std_dev = variation/3
mean = (val_max + val_min)/2
fig, ax = plt.subplots(3, 3)
plt.suptitle("Histogram examples by Davidson Lima (github.com/davidsonlima)", 
             fontweight='bold')
i = 0
j = 0
pos = 1
while (i < 3):
    while (j < 3):
        dist_normal = np.random.normal(mean, std_dev,  100)
        max_min = 'Min: 0:.2f, Max: 1:.2f'.format(dist_normal.min(), dist_normal.max())
        ax[i, j].hist(dist_normal, bins=30, label='Dist' + str(pos))
        ax[i, j].set_title('Normal distribution ' + str(pos))
        ax[i, j].legend()
        ax[i, j].text(mean, 0, max_min, horizontalalignment='center', color='white')
        print('Normal distribution 0\n\tMin: 1:.2f, Max: 2:.2f'
              .format(pos, dist_normal.min(), dist_normal.max()))
        j += 1
        pos += 1
    j = 0
    i += 1
plt.show()



