# import necessary libraries
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Create environment
env = gym.make("Taxi-v3")
env.reset()
env.render()

# check state and action size
state_size = env.observation_space.n
print("State size: ", state_size)

action_size = env.action_space.n
print("Action size: ", action_size)

# Play with the environment

"""
Actions
There are 6 discrete deterministic actions:
- 0: move south
- 1: move north
- 2: move east
- 3: move west
- 4: pickup passenger
- 5: drop off passenger
"""
# The code below is commented so that everything else runs smoothly. 
# if you need to check that the environment is working perfectly or not
# then just uncomment this section and run again
"""
done = False
env.reset()
env.render()
while not done:
    action = int(input('0/south 1/north 2/east 3/west 4/pickup 5/drop:'))
    new_state, reward, done, info = env.step(action)
    time.sleep(1.0) 
    print(f'S_t+1={new_state}, r_t+1={reward}, done={done}')
    env.render()

"""    
    
 # evaluation policy    
def eval_policy(env_, pi_, gamma_, t_max_, episodes_):
    env_.reset()

    v_pi_rep = np.empty(episodes_)
    for e in range(episodes_):
        s_t = env.reset()
        v_pi = 0
        for t in range(t_max_):
            a_t = pi_[s_t]
            s_t, r_t, done, info = env_.step(a_t) 
            v_pi += gamma_**t*r_t
            if done:
                break
        v_pi_rep[e] = v_pi
        env.close()
        
    return np.mean(v_pi_rep), np.min(v_pi_rep), np.max(v_pi_rep), np.std(v_pi_rep)


qtable = np.zeros((500,6)) # Taxi v3
episodes = 1500 # num of training episodes
interactions = 666 # max num of interactions per episode
epsilon = 0.99 # e-greedy
alpha = 0.5 # learning rate 1.
gamma = 0.9 # reward decay rate
hist = [] # evaluation history

# Main Q-learning loop
for episode in range(episodes):
    
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for interact in range(interactions):
        # exploitation vs. exploratin by e-greedy sampling of actions
        if np.random.uniform(0, 1) > epsilon:
            action = np.argmax(qtable[state,:])
        else:
            action = np.random.randint(0,6)

        # Observe
        new_state, reward, done, info = env.step(action)

        # Update Q-table
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
                
        # Our new state is state
        state = new_state
        
        # Check if terminated
        if done == True: 
            break
    
    if episode % 10 == 0 or episode == 1:
        pi = np.argmax(qtable, axis=1)
        val_mean, val_min, val_max, val_std = eval_policy(env, pi, gamma, interactions, episodes)
        hist.append([episode, val_mean,val_min,val_max,val_std])

env.reset()


hist = np.array(hist)
print(hist.shape)

plt.plot(hist[:,0],hist[:,1])
plt.show()


hist = np.array(hist)
print(hist.shape)

plt.plot(hist[:,0],hist[:,1])
# Zero-clipped
#plt.fill_between(hist[:,0], np.maximum(hist[:,1]-hist[:,4],np.zeros(hist.shape[0])),hist[:,1]+hist[:,4],
plt.fill_between(hist[:,0], hist[:,1]-hist[:,4],hist[:,1]+hist[:,4],
                alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', linewidth=0)

plt.xlabel("Episode")
plt.ylabel("Mean of the expected return")
plt.grid()
plt.show()


# Model with the input size of number of states and 
# output size of number of actions... 

model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=(state_size,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(action_size, activation='linear')
])

loss_fn= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
model.summary()

# one hot encoding the states (inputs)
x =  np.zeros((state_size, state_size))
for i in range(state_size):
    x[i][i] = 1
    
#x

# picking the best actions from the qtable (option 1)
best_action=np.zeros(state_size)
for i in range(state_size):
    best_action[i] = np.argmax(qtable[i])

y = best_action
#y

# Taining the model
history = model.fit(x,y, epochs=100, steps_per_epoch=500)

# loss plot
plt.plot(history.history['loss'], color='r', label='loss')

# accuracy plot
plt.plot(history.history['accuracy'], color = 'g', label='accuracy')
plt.legend()