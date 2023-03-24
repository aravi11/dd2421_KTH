import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split

import pandas as pd
nameToLabel = {"Dragspel": 0.0, "Serpent": 2.0, "Nyckelharpa": 1.0}
labelToName = {0.0: "Dragspel", 2.0: "Serpent", 1.0: "Nyckelharpa"}

def getDataFromCSV(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocessTrainingData(train):
    print(list(train.columns.values))
    print('\nTraining size before cleaning : {}'.format(len(train)))

    train.y = train.y.replace(['ragspel', 'yckelharpa', 'erpent'], ['Dragspel' , 'Nyckelharpa', 'Serpent'])
    print(train.y.unique())

    for iter in train.y.unique():
        if( (iter != 'Serpent') and  (iter != 'Dragspel') and (iter != 'Nyckelharpa')):
            train = train[train.y != iter]

    # Filling row with '?' with mean 
    train.x1 = train.x1.replace(['?'], 1.13075) # Filled nan and other values with mean


    #Convert all object dtypes in dataframe to float 
    train['x1'] = pd.to_numeric(train['x1'])
    train['x2'] = pd.to_numeric(train['x2'])
    train['x3'] = pd.to_numeric(train['x3'])
    train['x4'] = pd.to_numeric(train['x4'])


    x1_mean = train.x1.mean()
    x2_mean = train.x2.mean()
    x3_mean = train.x3.mean()
    x4_mean = train.x4.mean()
    x7_mean = train.x7.mean()
    x8_mean = train.x8.mean()
    x9_mean = train.x9.mean()
    x10_mean = train.x10.mean()
    x13_mean = train.x13.mean()
    train = train.drop('x5', axis=1)


    train = train.fillna(9999999999.0) #Filling all nan values with a random number

    print(train.y.unique())
    print(train.dtypes)
    print('\nTraining size after cleaning : {}'.format(len(train)))

    train.y.replace('', np.nan)
    train.dropna(subset=["y"])
    print('List of unique class types : {}'.format(train.y.unique()))

    # Replace all nan values (filled with random number) with the mean
    train.x1 = train.x1.replace([ 9999999999.0], x1_mean) # Filled nan and other values with mean and std dev
    train.x2 = train.x2.replace([ 9999999999.0], x2_mean) # Filled nan and other values with mean and std dev
    train.x3 = train.x3.replace([ 9999999999.0], x3_mean) # Filled nan and other values with mean and std dev
    train.x4 = train.x4.replace([ 9999999999.0], x4_mean) # Filled nan and other values with mean and std dev
    train.x7 = train.x7.replace([ 9999999999.0], x7_mean) # Filled nan and other values with mean and std dev
    train.x8 = train.x8.replace([ 9999999999.0], x8_mean) # Filled nan and other values with mean and std dev
    train.x9 = train.x9.replace([ 9999999999.0], x9_mean) # Filled nan and other values with mean and std dev
    train.x10 = train.x10.replace([ 9999999999.0], x10_mean) # Filled nan and other values with mean and std dev
    train.x13 = train.x13.replace([ 9999999999.0], x13_mean) # Filled nan and other values with mean and std dev


    train.x6 = train.x6.replace(['Brinnelvägen 8','Entrée', 'KTH Biblioteket', 'Lindstedsvägen 24', 'Slussen', 'Östra stationen', 'Ostra stationen', 9999999999.0 ], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0])

    x6_train_uniques = train.x6.unique()
    print(x6_train_uniques)


    train.x11 = train.x11.replace(['False', 'F', 'Flase', 'True', 'tru', 'Tru', 9999999999.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    x11_train_uniques = train.x11.unique()
    train.x12 = train.x12.replace(['False', 'F', 'Flase', 'True', 'tru', 'Tru', 9999999999.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    x12_train_uniques = train.x12.unique()


    print(x11_train_uniques)
    print(x12_train_uniques)

    train.y = train.y.replace(['Dragspel', 'Nyckelharpa', 'Serpent'], [0.0, 1.0, 2.0])

    print(train.y.unique())

    x = train.drop('y',axis =1)
    x = np.asarray(x)
    #print(x)
    y = train.y 
    y = np.asarray(y)
    #print(y)

    return x,y

def preprocessEvalData(eval):
    print(list(eval.columns.values))
    print('Eval size before cleaning nan : {}'.format(len(eval)))

    #print('Eval size after cleaning nan : {}'.format(len(eval)))
    eval = eval.fillna(9999999999.0)

    eval = eval.drop('x5', axis=1)

    eval.x6 = eval.x6.replace(['Brinnelvägen 8','Entrée', 'KTH Biblioteket', 'Lindstedsvägen 24', 'Slussen', 'Östra stationen', 9999999999.0 ], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    x6_eval_uniques = eval.x6.unique()
    print(x6_eval_uniques)

    eval.x11 = eval.x11.replace([False, True], [0.0, 1.0])
    eval.x12 = eval.x12.replace([False, True], [0.0, 1.0])

    eval = np.asarray(eval)

    return eval


def splitIntoTrainingAndValidation(X, y, valFraction=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=valFraction)
    return X_train, X_test, y_train, y_test

def countEntriesPerLabel(y):
    classes = np.unique(y)
    for label in classes:
        entries = np.where(y == label)
        print("Number of entries for class: " + str(label))
        print(entries[0].shape)

def getAccuracy():
    f = open("eval_predictions.txt", "r")
    x = f.readlines()

    f2 = f = open("EvaluationGT-6.csv", "r")
    y = f2.readlines()

    matches = 0

    for index in range(len(x)):
        if x[index] == y[index]:
            matches += 1

    print(f"{(matches / len(x)) * 100} %")