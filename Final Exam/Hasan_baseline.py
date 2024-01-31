# importing necessry libraries
import numpy as np
import matplotlib.pyplot as plt

# load data
X_train = np.loadtxt( 'E:\OneDrive - TUNI.fi\Tampere University (MSc in CS-DS)\Year 2\Period 4\DATA.ML.200 Pattern Recognition and Machine Learning\Final Exam\X_train.dat' )
X_test = np.loadtxt('E:\OneDrive - TUNI.fi\Tampere University (MSc in CS-DS)\Year 2\Period 4\DATA.ML.200 Pattern Recognition and Machine Learning\Final Exam\X_test.dat')
y_train = np.loadtxt( 'E:\OneDrive - TUNI.fi\Tampere University (MSc in CS-DS)\Year 2\Period 4\DATA.ML.200 Pattern Recognition and Machine Learning\Final Exam\y_train.dat' )
y_test = np.loadtxt('E:\OneDrive - TUNI.fi\Tampere University (MSc in CS-DS)\Year 2\Period 4\DATA.ML.200 Pattern Recognition and Machine Learning\Final Exam\y_test.dat')

# Calculate the mean of the target variable (y)
y_pred = np.mean(y_train)

# Create an array of the same length as the test data, filled with the mean value
y_pred_baseline = np.full_like(y_test, y_pred)

# Calculate the mean squared error between the baseline and the true target values
mse_baseline = np.mean((y_pred_baseline - y_test) ** 2)

# Print the mean squared error of the baseline prediction
print("Mean squared error (baseline): {:.2f}".format(mse_baseline))
