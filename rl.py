import numpy as np
import pickle 
import tensorflow as tf
import random
## Part 1 - Linear Model Training using SGD
# This part can be skipped unless you want to understand the details of how the linear model is being trained using Stochastic Gradient Descent. 
# A starting point can be found here : https://medium.com/deeplearningschool/2-1-linear-regression-f782ada81a53
# However there are many online ressources on the topic. 


class NLinearModels(object):
    def __init__(self, num_states , num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._q_s_a = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()


        

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 150, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 150, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc1, 100, activation=tf.nn.relu)
        fc4 = tf.layers.dropout(
    fc3,
    rate=0.9,
    noise_shape=None,
    seed=None,
    training=False,
    name=None
)
        self._logits = tf.layers.dense(fc3, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, sess, state):
        return sess.run(self._logits, feed_dict={self._states:
                                                 state.reshape(1, self._num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})




class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)

## Part 2 - Experience Replay
## This part has to be read and understood in order to code the main.py file. 

class ExperienceReplay(object):
    """
    During gameplay all the experiences < s, a, r, s’ > are stored in a replay memory. 
    In training, batches of randomly drawn experiences are used to generate the input and target for training.
    """

    def __init__(self, sess, model, max_memory=100, discount=.9, max_eps = 1,min_eps = 0):
        """
        Setup
        max_memory: the maximum number of experiences we want to store
        memory: a list of experiences
        discount: the discount factor for future experience
        
        In the memory the information whether the game ended at the experience is stored seperately in a nested array
        [...
        [experience, game_over]
        [experience, game_over]
        ...]
        """
        self.sess = sess
        self.model = model
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.eps = max_eps
        self.decay = 0.05
        self.memory = Memory(max_memory)
        self.discount = discount

    def remember(self, experience, game_over):
        #Save an experience to memory
        self.memory.add_sample([experience, game_over])
        

    def get_batch(self, model, batch_size=10):
        
        #How many experiences do we have?
        len_memory = self.memory._max_memory
        
        #Calculate the number of actions that can possibly be taken in the game
        num_actions = 4
        
        #Dimensions of the game field
        env_dim = list(self.memory._samples[0][0][0].shape)
        env_dim[0] = min(len_memory, batch_size)
        
        
        #We want to return an input and target vector with inputs from an observed state...
        inputs = np.zeros(env_dim)
        #...and the target r + gamma * max Q(s’,a’)
        #Note that our target is a matrix, with possible fields not only for the action taken but also
        #for the other possible actions. The actions not take the same value as the prediction to not affect them
        Q = np.zeros((inputs.shape[0], num_actions))
        #We draw experiences to learn from randomly
        batch = self.memory.sample(self.model._batch_size)

        states = np.array([val[0][0] for val in batch])
        next_states = np.array([(np.zeros((1,29,41,2))
                             if val[1] == True else val[0][3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self.model.predict_batch(states.reshape((32,-1)), self.sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self.model.predict_batch(next_states.reshape((32,-1)), self.sess)
        
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if b[1] == True :
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + self.discount * np.amax(q_s_a_d[i])
            inputs[i:i+1] = state
            Q[i] = current_q
        
        return inputs, Q


    def load(self):
        self.memory = pickle.load(open("save_rl/memory.pkl","rb"))
    def save(self):
        pickle.dump(self.memory,open("save_rl/memory.pkl","wb"))
