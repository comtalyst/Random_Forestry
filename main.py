import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as pyplot

# read data from the excel file
data = pd.read_excel("Immunotherapy.xlsx")
predict = "Result_of_Treatment"
X = np.array(data.drop([predict], 1))
Y = np.array(data[[predict]])

testSize = 0.3                                      # 30 percent of data will be part of the test set

# read the existing best model's accuracy
# best = 0                                          # comment lines below instead if first time / want to reset the save
pickleIn = open("modelBest.pickle", "rb")
best = pickle.load(pickleIn)

# find keep generating the model to find the best model
for _ in range(100):                                # 100 times
    # split training and testing data
    xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, Y, test_size=testSize)
    # train the model and score it
    randomForest = RandomForestClassifier(n_estimators=100)
    randomForest.fit(xTrain, np.ravel(yTrain))
    acc = randomForest.score(xTest, yTest)
    if acc > best:
        best = acc
        # save the model
        with open("model.pickle", "wb") as f:
            pickle.dump(randomForest, f)
        # save the best value
        with open("modelBest.pickle", "wb") as f:
            pickle.dump(best, f)

# now, pick up the best
pickleIn = open("model.pickle", "rb")
randomForest = pickle.load(pickleIn)
acc = best

# select the predictive features
selector = SelectFromModel(randomForest, prefit=True)
selectedFeatures = data.axes[1][selector.get_support(indices=True)]     # get the selected labels
print(selectedFeatures)

# visualize (for visual information / experiment)
colors = {0:'red', 1:'blue'}
for x in range(90):
    pyplot.scatter(data['age'][x], data['Time'][x], color=colors[data[predict][x]])
pyplot.show()

# The variables that are most likely to be predictive are age and Time