# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# Let's then generate 5 data points from two normally distributed classes.

np.random.seed(13)
x_h = np.random.normal(1.1,0.3,5)
x_e = np.random.normal(1.9,0.4,5)
plt.plot(x_h,np.zeros([5,1]),'co', label="hobbit")
plt.plot(x_e,np.zeros([5,1]),'mo', label="elf")
plt.title('Training samples from two different classes')
plt.legend()
plt.xlabel('height [m]')
plt.show()

# Let's assign each point suitable output (𝑦∈0,1)

y_h = np.zeros(x_h.shape)
y_h[:] = 0.0
y_e = np.zeros(x_e.shape)
y_e[:] = +1.0
plt.plot(x_h,y_h,'co', label="hobbit")
plt.plot(x_e,y_e,'mo', label="elf")
plt.title('Training samples from two different classes c1 ja c2')
plt.legend()
plt.xlabel('height [m]')
plt.ylabel('y [class id]')
plt.show()

# Let's put all training data points to the same vectors

x_tr = np.concatenate((x_h,x_e))
y_tr = np.concatenate((y_h,y_e))
print(f'The size of x is {x_tr.size}')
print(f'The size of y is {y_tr.size}')

def grdiant_discent(intial_w0, intial_w1,epochs, lr):
    
    # Compute MSE heat map for different a and b
    w0_t = intial_w0
    w1_t = intial_w1
    num_of_epochs = epochs
    learning_rate = lr
    
    MSE_list =[]

    for e in range(num_of_epochs):
        y = expit(w1_t*x_tr+w0_t)
    
        # gradiant calculation
        gd_w1 = -np.sum(2*x_tr*(y_tr-y)*y*(1-y))
        gd_w0 = -np.sum(2*(y_tr-y)*y*(1-y))
    
        # updating weights
        w1_t = w1_t-learning_rate*gd_w1
        w0_t = w0_t-learning_rate*gd_w0
        
        y_pred = expit(w1_t*x_tr+w0_t)
        MSE = np.sum((y_tr-y_pred)**2)/(len(y_tr))
        MSE_list.append(MSE)
        
    
        # Plot after every 20th epoch
        if np.mod(e,20) == 0 or e == 1: 
            plt.title(f'Epoch={e} w0={w0_t:.2f} w1={w1_t:.2f} MSE={MSE:.2f}')
            plt.plot(x_h,y_h,'co', label="hobbit")
            plt.plot(x_e,y_e,'mo', label="elf")
            x = np.linspace(0.0,+5.0,50)
            plt.plot(x,expit(w1_t*x+w0_t),'b-',label='y=logsig(w1x+w0)')
            plt.plot([0.5, 5.0],[0.5,0.5],'k--',label='y=0 (class boundary)')
            plt.xlabel('height [m]')
            plt.legend()
            plt.show()
            
    
    return MSE_list

# List to sotre MSE values for different Learning Rates
MSE_lr = []

# list of different learning rates
learning_rate =[0.1, 0.5, 0.7, 0.9]
epochs = 100
w0 = 0
w1 = 0

# Examining different learning rates
for lr in learning_rate:
    temp_MSE = grdiant_discent(w0,w1,epochs,lr)
    MSE_lr.append(temp_MSE) 
    
x_epochs = list(range(0,epochs))
plt.title("Analysis of different learning rates")    
plt.plot(x_epochs, MSE_lr[0], 'b',label='lr = 0.1')
plt.plot(x_epochs, MSE_lr[1], 'g',label='lr = 0.5')
plt.plot(x_epochs, MSE_lr[2], 'r',label='lr = 0.7')
plt.plot(x_epochs, MSE_lr[3], 'y',label='lr = 0.9')
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# from the analysis of different learing rates we can see that
# learing rates  = 0.5, 0.7 and 0.9 convrges to the MSE value  = 0.7
# However, for the learning rate = 0.7 and 0.9 MSE value varies a lot. 
# On the other hand, 0.5 learning rate shows more stable values. 
# So I prefer to use learing rate 0.5. 

grdiant_discent(w0,w1,epochs,lr= 0.5)

    