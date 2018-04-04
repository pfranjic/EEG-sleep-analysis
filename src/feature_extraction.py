import numpy as np
from sklearn import preprocessing
import random

def extract_features_from_epochs(epochs, fs, epoch_length):
    print("Extracting features...")
    N = epochs.shape[0]
    L = epoch_length * fs # signal length
    f = np.linspace(0, L-1, L) * fs/L
    delta1, delta2, theta1, theta2, alpha1, alpha2, beta1, beta2 = 0, 4, 4, 8, 8, 13, 13, 30
    all_indices = np.where((f <= beta2))    
    delta_indices = np.where((f >= delta1) & (f <= delta2))
    theta_indices = np.where((f >= theta1) & (f <= theta2))
    alpha_indices = np.where((f >= alpha1) & (f <= alpha2))
    beta_indices = np.where((f >= beta1) & (f <= beta2))
    nr_features = 6 # number of features to be calculated
    features = np.zeros((N, nr_features))
    # calculation of delta, theta, alpha and beta band power ratios
    for index in range(N):    
        epoch = epochs[index, :]
        Y = abs(np.fft.fft(epoch))
        mean_total_power = np.mean(Y[all_indices])
        features[index,:] = (mean_total_power, np.mean(f[all_indices] * Y[all_indices]) / mean_total_power, 
        np.mean(Y[delta_indices]) / mean_total_power, np.mean(Y[theta_indices]) / mean_total_power, 
        np.mean(Y[alpha_indices]) / mean_total_power, np.mean(Y[beta_indices]) / mean_total_power)
    return preprocessing.scale(features)

def features_to_codebook(features, nr_groups):
    print("Discretizing features...")
    N = features.shape[0]
    nr_features = features.shape[1]
    # vector quantization
    ## codebook generation
    minimums = np.min(features, 0)
    maximums = np.max(features, 0)
    min_max = list(zip(minimums,maximums))
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
                distances[index_codebook] = np.linalg.norm(codebook[index_codebook, :] - features[index_epoch, :])
            epoch_codes[index_epoch] = np.argmin(distances)
        if np.array_equal(epoch_codes_prev, epoch_codes):
            break
        epoch_codes_prev = np.copy(epoch_codes)
        # calculate new center vectors
        for code in np.unique(epoch_codes):
            code_indices = np.where(epoch_codes == code)      
            grouped_vectors = np.squeeze(features[code_indices, :])
            codebook[code] = np.mean(grouped_vectors, axis=0)
    return codebook, epoch_codes

def ranges(start, end, nb):
    step = (end - start) / nb
    return [(start + step * i, start + step * (i + 1)) for i in range(nb)]
