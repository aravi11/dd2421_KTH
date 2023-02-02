import monkdata as m
import dtree as dt
import numpy as np
import drawtree_qt5 as qt5

#Initialize the dataset

monk_train = [m.monk1, m.monk2, m.monk3]

monk_test = [m.monk1test, m.monk2test, m.monk3test]

entropy_values = [0, 0, 0]

### Assignemt 1: Get Entropy of the MONK training dataset
print('********** Assignment 1 **********')
for iter in range(3):
  entropy_values[iter] = dt.entropy(monk_train[iter])
  print('Monk{} - {}'.format(iter, entropy_values[iter]))


### Assignemt 3: Get Expected Information Gain of the MONK training dataset

def get_info_gain(): 
  print('********** Assignment 3 **********')
  for i in range(3): #To iterate through the monk datasets
    for attribute in range(6): # To iterate through each attribute in the dataset
        avg_gain = dt.averageGain(monk_train[i], m.attributes[attribute])
        print('{} - {} : {}'.format('monk_' + str(i+1),m.attributes[attribute], avg_gain ))    

get_info_gain()


### Assignemt 5: Get error set of the MONK train and test dataset
print('********** Assignment 5 **********')

for i in range(3):
  d_tree = dt.buildTree(monk_train[i], m.attributes)
  print('{}: {}'.format( 'monk_'+ str(i+1),1 - dt.check(d_tree, monk_train[i])))
  print('{}: {}'.format( 'monk_'+ str(i+1)+'test',1 - dt.check(d_tree, monk_test[i])))  

qt5.drawTree(dt.buildTree (m.monk2,m.attributes))

#a = dt.allPruned(d_tree)
#print(str(a))