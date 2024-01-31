import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

base_path = "E:/OneDrive - TUNI.fi/Tampere University (MSc in CS-DS)/Year 2/Period 4/DATA.ML.200 Pattern Recognition and Machine Learning/Online Programming Test 1/data/"

X_train = np.loadtxt(base_path+"X_train.dat")
y_train= np.loadtxt(base_path+"Y_train.dat")
X_test= np.loadtxt(base_path+"X_test.dat")
y_test = np.loadtxt(base_path+"Y_test.dat")

#1nn 
knn = KNeighborsClassifier(n_neighbors = 1)

# encoding continious values
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# calculate MAE
y_test = label_encoder.fit_transform(y_test)
error = mae(y_test, y_pred)
  
# display
print("Mean absolute error : " + str(error))