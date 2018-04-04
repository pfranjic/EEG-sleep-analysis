import numpy as np
import pyedflib

def read_annotations_from_file(filename, sleep_stages_dict):
    print("Reading annotations...")
    with open(filename) as f:
        content = f.readlines()    
    nr_states = len(sleep_stages_dict)
    sleep_stages = []
    for index in range(3, len(content)):
        tokens = content[index].split()
        sleep_stages = sleep_stages + [sleep_stages_dict[tokens[7]]]*int((int(tokens[9]) / 30))
    sleep_stages = np.array(sleep_stages)
    return sleep_stages

def load_epochs_from_file(filename, epoch_length, fs):
    print("Loading epochs...")
    # fs: sampling frequency
    f = pyedflib.EdfReader(filename)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbuf = f.readSignal(0)
    L = epoch_length * fs # signal length
    epochs = np.reshape(sigbuf, (-1, L))
    return epochs
