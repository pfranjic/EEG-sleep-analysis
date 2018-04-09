import numpy as np
from collections import Counter, defaultdict
from itertools import count

class HMM:
    # Discrete Hidden Markov Model
    def __init__(self, nr_states, nr_groups):
        self.nr_states = nr_states
        self.nr_groups = nr_groups
        self.pi = np.zeros(nr_states)
        self.A = np.zeros((self.nr_states, self.nr_states))
        self.B = np.zeros((self.nr_states, self.nr_groups))

    def train(self, sleep_stages_train, epoch_codes_train):
        # the hidden states are observed so the Baum-Welch 
        # training algorithm to find out pi, A and B isn't necessary
        unique, counts = np.unique(sleep_stages_train, return_counts=True)
        nr_epochs = sleep_stages_train.shape[0]
        self.pi = np.array(counts) / nr_epochs

        self.A = np.zeros((self.nr_states, self.nr_states))
        for (x,y), c in Counter(zip(sleep_stages_train, sleep_stages_train[1:])).items():
            self.A[x][y] = c
        # self.A = np.random.randint(100, size=(self.nr_states, self.nr_states))
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        self.B = np.zeros((self.nr_states, self.nr_groups))
        for (x,y),c in Counter(zip(sleep_stages_train, epoch_codes_train)).items():
            self.B[x,y] = c
        self.B = self.B / self.B.sum(axis=1, keepdims=True)

    def get_state_sequence(self, x):
        # Viterbi algorithm
        # according to https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
        T = x.shape[0]
        delta = np.zeros((T, self.nr_states))
        psi = np.zeros((T, self.nr_states))
        delta[0] = np.log(self.pi) + np.log(self.B[:,x[0]])
        for t in range(1,T):
            for j in range(self.nr_states):
                delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j])) + np.log(self.B[j, x[t]])
                psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states
