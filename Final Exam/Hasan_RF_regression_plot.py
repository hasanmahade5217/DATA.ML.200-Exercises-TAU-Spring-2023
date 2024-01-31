import numpy as np
import matplotlib.pyplot as plt
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

depths = [1,2,3,4,5,6,7,8,9,10,11,12]
mae = []
for i in range(len(depths)):
    # Initialize the Random Forest Regressor model
    rf_model = RandomForestRegressor(max_depth=i+1)

    # Fit the model to the training data
    rf_model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = rf_model.predict(X_test)

    # Calculate the mean absolute error of the predictions
    mae_rf = mean_absolute_error(y_test, predictions)
    mae.append(mae_rf)
    print(mae_rf)
    
# standard deviation    
std = np.std(mae)   


# plot
plt.plot(depths,mae)
plt.xlabel("Max Depths")
plt.ylabel("MAE")
plt.fill_between(depths, mae-std,mae+std,
                alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', linewidth=0) 