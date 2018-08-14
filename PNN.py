import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

datatrain = np.genfromtxt('data_train_PNN.txt')#load data train from txt file
datatest = np.genfromtxt('data_test_PNN.txt')#load data test from txt file

#=======================================create the model, you can ignore this below till line 17
ax.scatter(datatrain[:,0],datatrain[:,1],datatrain[:,2])
ax.set_xlabel('Attribute 1')
ax.set_ylabel('Attribute 2')
ax.set_zlabel('Attribute 3')
plt.show()
#==============================================================================

smooth = 0.001#smooth value
def fungsiG (xn1,xn2,xn3,x1,x2,x3):#g(x) function
    return np.exp(-1*((((xn1-x1)**2+(xn2-x2)**2+(xn3-x3)**2)/2*smooth**2)))

def fungsiF (x1,x2,x3,data):#f(x) function
    count1 = 0
    count2 = 0
    count3 = 0
    sumG1 = 0
    sumG2 = 0
    sumG3 = 0
    f1 = 0
    f2 = 0
    f3 = 0
    for i in range(len(data)):
        if data[i][3] == 0.0:
            count1 = count1 + 1
            sumG1 += fungsiG(x1,x2,x3,data[i][0],data[i][1],data[i][2])
        if data[i][3] == 1.0:
            count2 = count2 + 1
            sumG2 += fungsiG(x1,x2,x3,data[i][0],data[i][1],data[i][2])
        if data[i][3] == 2.0:
            count3 = count3 + 1
            sumG3 += fungsiG(x1,x2,x3,data[i][0],data[i][1],data[i][2])
    f1 = sumG1/count1
    f2 = sumG2/count2
    f3 = sumG3/count3
    return f1,f2,f3
    
def classify(f1,f2,f3):#classification function
    if f1 > f2 and f1 > f3:
        return 0.0
    elif f2 > f1 and f2 > f3:
        return 1.0
    else:
        return 2.0

count = 0
hasilpercobaan = []
while count < 10:#try ten times, you can modify this just change the while condition
    train,test = train_test_split(datatrain, test_size=0.33)#choose 33% random data from data train for data test
    hasil = []    
    for i in range(len(test)):
        f1,f2,f3 = fungsiF(test[i][0],test[i][1],test[i][2],train)
        hasil.append(classify(f1,f2,f3))
    
    benar = 0            
    for i in range(len(hasil)):
        if hasil[i] == test[i][3]:
            benar += 1
            
    jum = len(hasil)
    count += 1
    hasilpercobaan.append(benar/jum*100)

print("Accuracy : ",np.mean(hasilpercobaan),"%")#show the accuracy
#the accuracy can be different on every run, because the data choose randomly on every run too

result = datatest
for i in range(len(datatest)): #do classification on data test
    f1,f2,f3 = fungsiF(datatest[i][0],datatest[i][1],datatest[i][2],datatrain)
    result[i][3]=(classify(f1,f2,f3))

print("Classification result to data test : ")
print(result)#show the classification result