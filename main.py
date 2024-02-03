import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from mlxtend.plotting import plot_decision_regions

iris_data = datasets.load_iris()

#print(iris_data.data.shape)
#print(iris_data.target.shape)

features = iris_data.data
target = iris_data.target

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = svm.SVC()
