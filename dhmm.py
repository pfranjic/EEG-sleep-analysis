from __future__ import division
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter, defaultdict
from itertools import count
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

def read_annotations_from_file(filename, sleep_stages_dict):
    with open(filename) as f:
        content = f.readlines()    
    nr_states = len(sleep_stages_dict)
    sleep_stages = []
    for index in range(3,len(content)):
        tokens = content[index].split()
        sleep_stages = sleep_stages + [sleep_stages_dict[tokens[7]]]*int((int(tokens[9])/30))
    sleep_stages = np.array(sleep_stages)
    return sleep_stages

def load_epochs_from_file(filename, epoch_length, fs):
    # fs: sampling frequency
    f = pyedflib.EdfReader(filename)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbuf = f.readSignal(0)
    L = epoch_length * fs # signal length
    epochs = np.reshape(sigbuf, (-1, L))
    return epochs

def extract_features_from_epochs(epochs, fs, epoch_length):
    N = epochs.shape[0]
    L = epoch_length * fs # signal length
    f = np.linspace(0, L-1, L) * fs/L
    delta1,delta2,theta1,theta2,alpha1,alpha2,beta1,beta2 = 0,4,4,8,8,13,13,30
    all_indices = np.where((f <= beta2))    
    delta_indices = np.where((f >= delta1) & (f <= delta2))
    theta_indices = np.where((f >= theta1) & (f <= theta2))
    alpha_indices = np.where((f >= alpha1) & (f <= alpha2))
    beta_indices = np.where((f >= beta1) & (f <= beta2))
    nr_features = 6 # number of features to be calculated
    features = np.zeros((N,nr_features))
    # calculation of delta, theta, alpha and beta band power ratios
    for index in range(0,N):    
        epoch = epochs[index, :]
        Y = abs(np.fft.fft(epoch))
        mean_total_power = np.mean(Y[all_indices])
        features[index,:] = (mean_total_power, np.mean(f[all_indices]*Y[all_indices]) / mean_total_power, 
        np.mean(Y[delta_indices]) / mean_total_power, np.mean(Y[theta_indices]) / mean_total_power, 
        np.mean(Y[alpha_indices]) / mean_total_power, np.mean(Y[beta_indices]) / mean_total_power)
    return preprocessing.scale(features)

def features_to_codebook(features, nr_groups):
    N = features.shape[0]
    nr_features = features.shape[1]
    # vector quantization
    ## codebook generation
    minimums = np.min(features, 0)
    maximums = np.max(features, 0)
    min_max = zip(minimums,maximums)
    intervals = []
    
    for index in range(nr_features):
        intervals.append(ranges(min_max[index][0], min_max[index][1], nr_groups))
    ### initialize random codebook
    codebook = np.array([[random.uniform(min_max[column][0], min_max[column][1]) 
        for column in range(nr_features)] for row in range(nr_groups)])
    epoch_codes = np.zeros(N, dtype=np.int)
    epoch_codes_prev = np.zeros(N, dtype=np.int)

    while True:
        for index_epoch in range(N):
            distances = np.zeros(nr_groups)
            for index_codebook in range(nr_groups):
                distances[index_codebook] = np.linalg.norm(codebook[index_codebook, :]-features[index_epoch, :])
            epoch_codes[index_epoch] = np.argmin(distances)
        if np.array_equal(epoch_codes_prev, epoch_codes):
            break
        epoch_codes_prev = np.copy(epoch_codes)
        # calculate new center vectors
        for code in np.unique(epoch_codes):
            code_indices = np.where(epoch_codes == code)      
            grouped_vectors = np.squeeze(features[code_indices,:])
            codebook[code] = np.mean(grouped_vectors, axis=0)
    return codebook, epoch_codes

def ranges(start, end, nb):
    step = (end-start) / nb
    return [(start+step*i, start+step*(i+1)) for i in range(nb)]

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
        for (x,y), c in Counter(zip(sleep_stages_train, sleep_stages_train[1:])).iteritems():
            self.A[x][y] = c
        # self.A = np.random.randint(100, size=(self.nr_states, self.nr_states))
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        self.B = np.zeros((self.nr_states, self.nr_groups))
        for (x,y),c in Counter(zip(sleep_stages_train, epoch_codes_train)).iteritems():
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
    
if __name__ == '__main__':
    sleep_stages_dict = {'Sleep_stage_W':5, 'Sleep_stage_1':3, 'Sleep_stage_2':2, 'Sleep_stage_3':1,
        'Sleep_stage_4':0, 'Sleep_stage_R':4, 'Movement_time':6}
    sleep_stages = read_annotations_from_file("annotations.txt", sleep_stages_dict)
    nr_states = len(np.unique(sleep_stages))
    # annotations contain long sequences of the awake state at the beginning and the end - those are removed
    actual_sleep_epochs_indices = np.where(sleep_stages != sleep_stages_dict['Sleep_stage_W'])
    sleep_start_index = actual_sleep_epochs_indices[0][0]
    sleep_end_index = actual_sleep_epochs_indices[0][-1]
    sleep_stages = sleep_stages[sleep_start_index:sleep_end_index]
    
    epochs = load_epochs_from_file("SC4001E0-PSG.edf", epoch_length = 30, fs = 100)
    epochs = epochs[sleep_start_index:sleep_end_index,:]

    features = extract_features_from_epochs(epochs, epoch_length = 30, fs = 100)
   
    nr_groups = 20
    codebook, epoch_codes = features_to_codebook(features, nr_groups)

    sleep_stages_train, sleep_stages_test = np.split(sleep_stages, [int(0.8*sleep_stages.shape[0])])         
    epoch_codes_train, epoch_codes_test = np.split(epoch_codes, [int(0.8*epoch_codes.shape[0])]) 
        
    
    hmm = HMM(nr_states, nr_groups)
    hmm.train(sleep_stages_train, epoch_codes_train)
    x=hmm.get_state_sequence(epoch_codes_test)

    print sleep_stages_test
    print x
    print accuracy_score(sleep_stages_test, x)

