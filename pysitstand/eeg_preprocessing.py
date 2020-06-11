from pysitstand.eeg import EEG
from pysitstand.utils import reshape2Dto3D, butter_bandpass_filter, highpass_filter, notch_filter, randomString
from pysitstand.info import DATASET_PATH, PATH
from pysitstand.info import CH_NAMES, CH_TYPES

import numpy as np
import os
import mne
from mne.io import concatenate_raws
from mne import Epochs
from mne.preprocessing import create_eog_epochs, ICA
import scipy.io
from datetime import datetime
import os

try:
    import matlab.engine
except:
    pass  
# for saving image
# import matplotlib
# matplotlib.use('Agg')
# import sys
# try:
#     from PyQt5.QtWidgets import QApplication
#     from PyQt5 import QtCore
#     os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
#     qapp = QApplication(sys.argv)
#     qapp.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
# except:
#     pass

# Perform bandpass-filter on each trial.
def peform_butter_bandpass_filter(data, lowcut, highcut, fs, order):
    if len(data.shape) == 3:
        data_finished = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
        for i in range(data.shape[0]):
            data_finished[i,:,:] = butter_bandpass_filter(data[i,:,:], lowcut, highcut, fs, order)
        return data_finished
    else:
        print("Error dimesion") 

def peform_highpass_filter(data, lowcut, sfreq, order):
    if len(data.shape) == 3:
        data_finished = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
        for i in range(data.shape[0]):
            data_finished[i,:,:] = highpass_filter(data[i,:,:], lowcut, sfreq, order)
        return data_finished
    else:
        print("Error dimesion") 

def peform_notch_filter(data, f0, fs, Q):
    if len(data.shape) == 3:
        data_finished = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
        for i in range(data.shape[0]):
            data_finished[i,:,:] = notch_filter(data[i,:,:], f0, fs, Q)
        return data_finished
    else:
        print("Error dimesion") 

def rASR(data, sfreq, new_sfreq):
    '''
    data shape 2D: (chanels, 15sec*250Hz*5trial)
    '''
    # Create an info object.
    ch_types = CH_TYPES
    ch_names = CH_NAMES
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    long = 18000 # 15 s
    raws = [mne.io.RawArray(data[i][:,:long] , info) for i in range(len(data))]
    raw = concatenate_raws(raws)
    raw.set_montage("standard_1020")

    if new_sfreq is not None and new_sfreq!=sfreq:
        raw_resampled = raw.copy().resample(new_sfreq, npad='auto')
        rawdata = raw_resampled.get_data()
    else:
        rawdata = raw.get_data()
    rawdata = rawdata[:11,:]
    eng = matlab.engine.start_matlab()
    eng.addpath(PATH, '-begin');  

    # gen filename
    now = datetime.now()
    timestamp = str(datetime.timestamp(now))
    file_name = DATASET_PATH+'/'+randomString(3)+timestamp+'.mat'
    chanlocs = PATH+'/eeglab2019_0/eeg_chan11.locs'
    scipy.io.savemat(file_name, {'data':rawdata})
    raw_corrected = eng.rASR(file_name, chanlocs, PATH)
    os.remove(file_name)
    data = reshape2Dto3D(np.array(raw_corrected), trials=5)
    return data

def ica(data, sfreq, new_sfreq, save_name=None, threshold=2):

    if save_name is not None:
        for directory in ['ica','eog_score','eog_avg','raw_EEG','corrected_EEG','montage','new_raw']:
            if not os.path.exists(directory):
                os.makedirs(directory)
    # Create a dummy mne.io.RawArray object
    ch_types = CH_TYPES
    ch_names = CH_NAMES
    # Create an info object.
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    long = 18000 # 15 s
    raws = [mne.io.RawArray(data[i][:,:long] , info) for i in range(len(data))]
    raw = concatenate_raws(raws)
    raw.set_montage("standard_1020")

    if new_sfreq is not None and new_sfreq!=sfreq:
        raw_resampled = raw.copy().resample(new_sfreq, npad='auto')
        raw_tmp = raw_resampled.copy()
    else:
        raw_tmp = raw.copy()
    ica_obj = ICA(random_state=None)
    ica_obj.fit(raw_tmp)

    n_max_eog = 3  # use max 3 components
    eog_epochs = create_eog_epochs(raw_tmp)
    eog_epochs.decimate(5).apply_baseline((None, None))

    try:
        eog_inds, scores_eog = ica_obj.find_bads_eog(eog_epochs,threshold=threshold)
        print('Found %d EOG component(s)' % (len(eog_inds),))

        #remove EOG from EEG
        ica_obj.exclude += eog_inds
    except:
        pass
    raw_corrected = raw_resampled.copy()
    ica_obj.apply(raw_corrected)
    print(ica_obj)

    # save fig and data
    if save_name is not None:
        ica_obj.plot_sources(raw_tmp, show = False).savefig('ica/'+save_name+'_ica.png') 
        try:
            ica_obj.plot_scores(scores_eog, exclude=eog_inds, title='EOG scores',show = False).savefig('eog_score/'+save_name+'_eog_score.png') 
        except:
            pass
        ica_obj.plot_sources(eog_epochs.average(), title='EOG average',show = False).savefig('eog_avg/'+save_name+'_eog_avg.png')
        raw.plot(show = False,scalings=dict(eeg=50, eog=150)).savefig('raw_EEG/'+save_name+'_raw_EEG.png')
        raw_corrected.plot(show = False,scalings=dict(eeg=50, eog=150)).savefig('corrected_EEG/'+save_name+'_corrected_EEG.png')
        ica_obj.plot_components(inst=raw_tmp,show = False)[0].savefig('montage/'+save_name+'_montage.png')
        print('======================================')
        print(raw_corrected.get_data().shape)
        raw_corrected.save('new_raw/'+save_name+'_raw.fif', overwrite=True)
    return reshape2Dto3D(raw_corrected.get_data(), trials=5)

def preprocessing(data, filter_medthod, sfreq):
    for key, value in filter_medthod.items():
        if key == 'butter_bandpass_filter':
            data = peform_butter_bandpass_filter(data, value['lowcut'], value['highcut'], sfreq, value['order'])
            print('butter_bandpass_filter:', value)
        elif key == 'notch_filter':
            data = peform_notch_filter(data, value['f0'], sfreq, 25) 
            print('notch_filter:', value)
        elif key == 'highpass_filter':
            data = peform_highpass_filter(data, value['highcut'], sfreq, value['order'])
            print('highpass_filter:', value)
        elif key == 'ica':
            data = ica(data, sfreq, value['new_sfreq'], value['save_name'], value['threshold'])
            sfreq = value['new_sfreq'] # after resample
            print('ica:', value)
        elif key == 'rASR':
            data = rASR(data, sfreq, value['new_sfreq'])
            print('rASR', value)
    return data

def apply_eeg_preprocessing(subject_name=None, session='mi', task='sit', filter_medthod=None):
    '''Collect data from each run to preprocess data such as filtering and calulate ICA, then remove EOG signals from data

    Parameters
    ----------
    subject_name: ex. 'S04'
    filter_medthod: dict

    Usage
    ----------
    # filter params
    new_sfreq = 250 # for downsampling before applying ica
    notch = {'f0': 50}
    bandpass = {'lowcut': 1, 'highcut': 40, 'order': filter_order}
    ica = {'new_sfreq': new_sfreq, 'save_name': None, 'threshold': 2}

    # it will perform preprocessing from this order
    filter_medthod = {'notch_filter': notch, 
                    'butter_bandpass_filter': bandpass,
                    'ica': ica}

    # apply filter and ICA 
    eeg = apply_eeg_preprocessing(subject_name='S01', session='mi', task='sit', filter_medthod=filter_medthod)
    '''

    subject_path = DATASET_PATH+'/'+subject_name+'_EEG/'
    sfreq = 1200
    new_sfreq = sfreq
    if filter_medthod is not None:
        for key in filter_medthod.keys():
            if key=='ica' or key=='rASR':
                new_sfreq = filter_medthod[key]['new_sfreq'] if filter_medthod[key]['new_sfreq'] is not None else sfreq
    # subject_path = 'pysitstand/raw_data/'+subject_name+'_EEG/'
    runs, trials, channels, datapoint = 3, 5, 11, 15*new_sfreq
    processed_data = np.zeros((runs, trials, channels, datapoint)) # 3 runs, 5 trials, 11 eeg channels, 14 sec*250 Hz
    # for each run. Normally, each subject has 3 runs
    for i in range(runs):
        p_name = subject_path + subject_name+ '_EEG_' + str(i+1)+'.csv'
        eeg = EEG(p_name, 1200)
        raw_array = eeg.read_CSV()
        if session == 'me': # ME 
            arr_sit, arr_stand = eeg.collect_data_allphase(3, raw_array)
            if task == 'sit':
                del arr_stand
                data = arr_sit
            elif task == 'stand':
                del arr_sit
                data = arr_stand
        elif session == 'mi': # MI 
            if task == 'sit':
                data = eeg.collect_data_allphase(7, raw_array)
            elif task == 'stand':
                data = eeg.collect_data_allphase(6, raw_array)

        tmp = preprocessing(data=data, filter_medthod=filter_medthod, sfreq=sfreq)
        processed_data[i] = tmp[:,:channels,:] # drop EOG channels
    processed_data = processed_data.reshape(-1, channels, datapoint) # reshape 4D to 3D
    return processed_data

# The time-locked EEG and EMG for choosing MRCP which prior to voluntary movement(onset) (2.5 seconds: before 1.5 secs and after 1 sec)
def picking_mrcp_from_onset(eeg_data, onset, sfreq = 250):
    before = int(1.5*sfreq)
    after = int(1*sfreq)
    mrcp_duration = 2.5
    mrcp_data = np.zeros((eeg_data.shape[0], eeg_data.shape[1], int(mrcp_duration*sfreq)))     
    for trial in range(eeg_data.shape[0]):
        movement_onset = onset[trial]
        mrcp_data[trial,:,:] = eeg_data[trial,:,int(movement_onset-before):int(movement_onset+after)]
    return mrcp_data
