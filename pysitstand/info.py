from os.path import abspath
import os

listpath = abspath(os.getcwd()).split('/')

k = len(listpath)
for i, path in enumerate(listpath):
    if path == 'pysitstand':
        k=i
        break

# CONSTANT VARIABLES
PATH = '/'.join(listpath[:k])+'/pysitstand'
DATASET_PATH = '/'.join(listpath[:k])+'/pysitstand/raw_data'

CH_TYPES = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eog', 'eog',]

CH_NAMES = ['FCz', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4', 'POz', 'VEOG', 'HEOG']
