

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error as mae


base_path = "E:/OneDrive - TUNI.fi/Tampere University (MSc in CS-DS)/Year 2/Period 4/DATA.ML.200 Pattern Recognition and Machine Learning/Online Programming Test 1/data/"




X_train = np.loadtxt(base_path+"X_train.dat")
y_train= np.loadtxt(base_path+"Y_train.dat")
X_test= np.loadtxt(base_path+"X_test.dat")
y_test = np.loadtxt(base_path+"Y_test.dat")




# create linear regression object
reg = linear_model.LinearRegression()
  
# train the model using the training sets
reg.fit(X_train, y_train)



# prediction
y_pred = reg.predict(X_test)


# calculate MAE
error = mae(y_test, y_pred)
  
# display
print("Mean absolute error : " + str(error))





