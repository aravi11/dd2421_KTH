from data_preprocessing import *
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split


def trainTreeWithWholeTrainSet(model, X, y):
    model.fit(X, y)
    return model

def printPredictions(predictions):
    file = open("final_xg_boost_eval_predictions.txt", "w")
    for entry in predictions:
        file.write(labelToName[entry] + "\n")
    file.close()


def main():

    trainData = getDataFromCSV('./TrainOnMe-6.csv')
    #Removed the follwing lines from Traning Data as they had empty cols or outliers
    #740 Dragspel,0.63800,-1074.00949,
    
    X,y = preprocessTrainingData(trainData)

    countEntriesPerLabel(y)
    
    #model = xgb.XGBClassifier(learning_rate=0.2, n_estimators=50, max_depth=4, gamma=0.5)
    xgb_model = xgb.XGBClassifier() 
    optimization_dict = {'max_depth': [2,4,6],'n_estimators': [50,100,200]}

    model = GridSearchCV(xgb_model, optimization_dict, scoring='accuracy', verbose=1)


    evalData = getDataFromCSV('./EvaluateOnMe-6.csv')
    
    evalSet = preprocessEvalData(evalData)

    model = trainTreeWithWholeTrainSet(model, X, y)
    testPredictions = model.predict(X)
    predictions = model.predict(evalSet)
    printPredictions(predictions)

    getAccuracy()

main()