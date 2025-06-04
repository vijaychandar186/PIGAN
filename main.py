import sys
import os
from experiments.run import run_experiment
from utils.constants import SUBTYPE_FLAGS, select_subtype

def main():
    """Main function to run experiments for multiple models and a selected subtype."""
    subtype_options = ['H1N1', 'H3N2', 'H5N1', 'COV19']
    subtype = subtype_options[3]  # COV19
    subtype_flag = select_subtype(subtype)
    
    data_paths = {
        0: './raw/H1N1_cluster/',
        1: './raw/H3N2_cluster/',
        2: './raw/H5N1_cluster/',
        3: './processed/COV19/'
    }
    dataset_paths = {
        0: './processed/H1N1_drop_duplicates/triplet_cluster',
        1: './processed/H3N2/triplet_cluster',
        2: './processed/H5N1/triplet_cluster',
        3: './processed/COV19/triplet_cluster'
    }
    
    model_options = ['transformer', 'lstm', 'gru', 'rnn', 'attention', 'da-rnn', 'svm', 'random forest', 'logistic regression', 'mutagan']
    models_to_run = ['transformer','mutagan']
    
    for model in models_to_run:
        print(f"\nExperimental results with model {model} on subtype {subtype}:")
        run_experiment(model, subtype, data_paths[subtype_flag], dataset_paths[subtype_flag])

if __name__ == '__main__':
    main()