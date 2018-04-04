import sys
import numpy as np
from sklearn.metrics import accuracy_score
from hmm import HMM
from file_reading import read_annotations_from_file
from file_reading import load_epochs_from_file
from feature_extraction import features_to_codebook
from feature_extraction import extract_features_from_epochs
import warnings

# Main
# run: python src/dhmm.py data/SC4001E0-PSG.edf data/annotations.txt
# ====
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    """ Process the commandline arguments. Two arguments are expected: The .edf file path
    and the annotations .txt file path.
    """
    if len(sys.argv) == 3:
        edf_file = sys.argv[1]
        annotations_file = sys.argv[2]

        sleep_stages_dict = {'Sleep_stage_W':5, 'Sleep_stage_1':3, 'Sleep_stage_2':2, 'Sleep_stage_3':1,
            'Sleep_stage_4':0, 'Sleep_stage_R':4, 'Movement_time':6}
        sleep_stages = read_annotations_from_file(annotations_file, sleep_stages_dict)
        nr_states = len(np.unique(sleep_stages))
        # annotations contain long sequences of the awake state at the beginning and the end - those are removed
        actual_sleep_epochs_indices = np.where(sleep_stages != sleep_stages_dict['Sleep_stage_W'])
        sleep_start_index = actual_sleep_epochs_indices[0][0]
        sleep_end_index = actual_sleep_epochs_indices[0][-1]
        sleep_stages = sleep_stages[sleep_start_index:sleep_end_index]
        
        epochs = load_epochs_from_file(edf_file, epoch_length = 30, fs = 100)
        epochs = epochs[sleep_start_index:sleep_end_index,:]

        features = extract_features_from_epochs(epochs, epoch_length = 30, fs = 100)
       
        nr_groups = 20 # number of discrete features groups
        codebook, epoch_codes = features_to_codebook(features, nr_groups)
        
        training_percentage = 0.8 # % of data used for training the model
        sleep_stages_train, sleep_stages_test = np.split(sleep_stages, [int(training_percentage * sleep_stages.shape[0])])         
        epoch_codes_train, epoch_codes_test = np.split(epoch_codes, [int(training_percentage * epoch_codes.shape[0])]) 
            
        hmm = HMM(nr_states, nr_groups)
        hmm.train(sleep_stages_train, epoch_codes_train)
        x=hmm.get_state_sequence(epoch_codes_test)

        sleep_stages_reverse = {y:x for x,y in sleep_stages_dict.items()}
        actual_phases = list(map(lambda phase: sleep_stages_reverse[phase], sleep_stages_test))
        predicted_phases = list(map(lambda phase: sleep_stages_reverse[phase], x))
        print("Actual sleep phases paired with predicted sleep phases:")
        for actual, predicted in zip(actual_phases, predicted_phases):
            print(actual, predicted)
        print("Accuracy:", accuracy_score(sleep_stages_test, x))

    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: ./dhmm.py {edf-file} {annotation-file}"]))

  

