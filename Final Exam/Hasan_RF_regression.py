import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# load data
X_train = np.loadtxt( 'E:\OneDrive - TUNI.fi\Tampere University (MSc in CS-DS)\Year 2\Period 4\DATA.ML.200 Pattern Recognition and Machine Learning\Final Exam\X_train.dat' )
X_test = np.loadtxt('E:\OneDrive - TUNI.fi\Tampere University (MSc in CS-DS)\Year 2\Period 4\DATA.ML.200 Pattern Recognition and Machine Learning\Final Exam\X_test.dat')
y_train = np.loadtxt( 'E:\OneDrive - TUNI.fi\Tampere University (MSc in CS-DS)\Year 2\Period 4\DATA.ML.200 Pattern Recognition and Machine Learning\Final Exam\y_train.dat')
y_test = np.loadtxt('E:\OneDrive - TUNI.fi\Tampere University (MSc in CS-DS)\Year 2\Period 4\DATA.ML.200 Pattern Recognition and Machine Learning\Final Exam\y_test.dat')

# Baseline:  
# Calculate the mean of the target variable (y)
y_pred = np.mean(y_train)

# Create an array of the same length as the test data, filled with the mean value
y_pred_baseline = np.full_like(y_test, y_pred)

# Baseline end...


# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor()

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
predictions = rf_model.predict(X_test)

# Calculate the mean absolute error of the predictions
mae_rf = mean_absolute_error(y_test, predictions)

# Print the MAE of the predictions
print("RF : Mean Absolute Error: ", mae_rf)


# Calculate the mean absolute error of the baseline predictions
mae_bl = mean_absolute_error(y_test, y_pred_baseline)

# Print the MAE of the predictions
print("Baseline : Mean Absolute Error: ", mae_bl)