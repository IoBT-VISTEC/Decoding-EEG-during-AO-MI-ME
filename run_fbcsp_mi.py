import numpy as np
import os
import sys
import csv
from sklearn.model_selection import KFold

from pysitstand.model import fbcsp
from pysitstand.utils import sliding_window, sliding_window2
from pysitstand.eeg_preprocessing import apply_eeg_preprocessing
"""
Binary classification model.
We apply FBCSP-SVM (9 subbands from 4-40 Hz) on the subject-dependent scheme (leave a single trial for testing) for EEG-based MI classification.
x sec window size with y% step (0.1 means overlap 90%)

# How to run

>> python run_fbcsp_mi.py <window_size> <step> <filter order> <performing task> <prediction motel>

>> python run_fbcsp_mi.py 2 0.1 4 stand R_vs_AO rASR
>> python run_fbcsp_mi.py 2 0.1 4 sit R_vs_AO rASR
>> python run_fbcsp_mi.py 2 0.1 4 stand AO_vs_MI rASR
>> python run_fbcsp_mi.py 2 0.1 4 sit AO_vs_MI rASR

>> python run_fbcsp_mi.py 2 0.1 4 stand AO_vs_MI ICA && python run_fbcsp_mi.py 2 0.1 4 stand R_vs_AO ICA
>> python run_fbcsp_mi.py 2 0.1 4 sit AO_vs_MI ICA && python run_fbcsp_mi.py 2 0.1 4 sit R_vs_AO ICA
"""

def load_data(subject, task, prediction_model, artifact_remover, filter_order, window_size, step, sfreq):
    #load data the preprocessing

    # filter params
    notch = {'f0': 50}
    bandpass = {'lowcut': 1, 'highcut': 40, 'order': filter_order}
    ica = {'new_sfreq': sfreq, 'save_name': None, 'threshold': 2}
    rASR = {'new_sfreq': sfreq}

    # it will perform preprocessing from this order
    if artifact_remover == 'ICA':
        filter_medthod = {'notch_filter': notch, 
                    'butter_bandpass_filter': bandpass,
                    'ica': ica}
    elif artifact_remover == 'rASR':
        filter_medthod = {'notch_filter': notch, 
                    'butter_bandpass_filter': bandpass,
                    'rASR': rASR}

    
    # apply filter and ICA 
    data = apply_eeg_preprocessing(subject_name=subject, session='mi', task=task, filter_medthod=filter_medthod)

    # data : 15 sec 
    # define data
    R_class = data[:,:,int(2*sfreq):int(6*sfreq)]          # rest 2 to 6 s
    AO_class = data[:,:,int(6*sfreq):int(10*sfreq)] 
    MI_class = data[:,:,int(11*sfreq):int(15*sfreq)]       # beep at 11s

    len_data_point = R_class.shape[-1]
    num_windows = int(((len_data_point-win_len_point)/(win_len_point*step))+1)

    # define class
    if prediction_model == 'R_vs_AO':
        # sliding windows
        R_class_slided = np.zeros([15, num_windows, 11, window_size*sfreq])
        AO_class_slided = np.zeros([15, num_windows, 11, window_size*sfreq])
        for i, (R,AO) in enumerate(zip(R_class, AO_class)):
            R_class_slided[i,:,:,:] = np.copy(sliding_window2(np.array([R]), win_sec_len=window_size, step=step, sfreq=sfreq))
            AO_class_slided[i,:,:,:] = np.copy(sliding_window2(np.array([AO]), win_sec_len=window_size, step=step, sfreq=sfreq))
        X0 = np.copy(R_class_slided)
        X1 = np.copy(AO_class_slided)
    elif prediction_model == 'AO_vs_MI':
        # sliding windows
        AO_class_slided = np.zeros([15, num_windows, 11, window_size*sfreq])
        MI_class_slided = np.zeros([15, num_windows, 11, window_size*sfreq])
        for i, (AO,MI) in enumerate(zip(AO_class, MI_class)):
            AO_class_slided[i,:,:,:] = np.copy(sliding_window2(np.array([AO]), win_sec_len=window_size, step=step, sfreq=sfreq))
            MI_class_slided[i,:,:,:] = np.copy(sliding_window2(np.array([MI]), win_sec_len=window_size, step=step, sfreq=sfreq))
        X0 = np.copy(AO_class_slided)
        X1 = np.copy(MI_class_slided)
            
    del data, R_class, AO_class, MI_class

    y0 = np.zeros([X0.shape[0], X0.shape[1]])
    y1 = np.ones([X1.shape[0], X1.shape[1]])
    assert len(X0) == len(y0)
    assert len(X1) == len(y1)
    return X0, y0, X1, y1

if __name__ == "__main__":

    window_size = int(sys.argv[1]) # 1,2,3 sec.
    step = float(sys.argv[2]) # 0.1 --> overlap(90%)
    filter_order = int(sys.argv[3]) # 2 order of all fillter
    task = sys.argv[4] # stand, sit
    prediction_model = sys.argv[5] # R_vs_AO, AO_vs_MI
    artifact_remover = sys.argv[6] # ICA, rASR
    sfreq = 250 # new sampling rate [max = 1200 Hz]
    win_len_point = int(window_size*sfreq)

    for x in sys.argv:
        print("Argument: ", x)
    
    subjects = [ 'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08']

    if task == 'stand':
        save_name = 'sit_to_stand_mi'
    elif task == 'sit':
        save_name = 'stand_to_sit_mi'

    if prediction_model == 'R_vs_AO':
        save_path = 'MI-v2-'+artifact_remover+'-FBCSP-cv-'+str(window_size)+'s_'+task+'_'+prediction_model+'_filter_order_'+str(filter_order)
    elif prediction_model == 'AO_vs_MI':
        save_path = 'MI-v2-'+artifact_remover+'-FBCSP-cv-'+str(window_size)+'s_'+task+'_'+prediction_model+'_filter_order_'+str(filter_order)

    header = [ 'fold', 'accuracy', 
                '0.0 f1-score', '1.0 f1-score', 'average f1-score',
                '0.0 recall', '1.0 recall', 'average recall',
                '0.0 precision', '1.0 precision', 'average precision',
                'sensitivity', 'specificity'
            ]

    sum_value_all_subjects = []
    for subject in subjects:
        from joblib import dump, load
        print('===================='+subject+'==========================')

        for directory in [save_path, save_path+'/model', save_path+'/y_slice_wise']:
            if not os.path.exists(directory):
                os.makedirs(directory)

        #load data the preprocessing
        X0, y0, X1, y1 = load_data(subject=subject, task=task, 
                                    prediction_model=prediction_model, 
                                    artifact_remover=artifact_remover,
                                    filter_order=filter_order, 
                                    window_size=window_size, 
                                    step=step, 
                                    sfreq=sfreq)

        with open(save_path+'/'+save_path+'_result.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([str(subject)])
            writer.writerow(header)

        kf = KFold(n_splits=15, shuffle=False) # Define the split - into 15 folds 
        print(kf)
        accuracy_sum, precision_0_sum, recall_0_sum, f1_0_sum, precision_1_sum, recall_1_sum, f1_1_sum, precision_sum, recall_sum, f1_sum = [], [], [], [], [], [], [], [], [], []
        sen_sum, spec_sum = [], []
        predict_result = []
        X_csp_com = []
        for index_fold, (train_idx, test_idx) in enumerate(kf.split(X0)):
            print("=============fold {:02d}==============".format(index_fold))
            print('fold: {}, train_index: {}, test_index: {}'.format(index_fold, train_idx, test_idx))

            X0_train, X1_train = X0[train_idx], X1[train_idx]
            y0_train, y1_train = y0[train_idx], y1[train_idx]
            X0_test, X1_test = X0[test_idx], X1[test_idx]
            y0_test, y1_test = y0[test_idx], y1[test_idx]

            X_train = np.concatenate((X0_train.reshape(-1, X0_train.shape[-2], X0_train.shape[-1]), 
                        X1[train_idx].reshape(-1, X1_train.shape[-2], X1_train.shape[-1])), axis=0)
            y_train = np.concatenate((y0_train.reshape(-1), y1_train.reshape(-1)), axis=0)

            X_test = np.concatenate((X0_test.reshape(-1, X0_test.shape[-2], X0_test.shape[-1]), 
                        X1[test_idx].reshape(-1, X1_test.shape[-2], X1_test.shape[-1])), axis=0)
            y_test = np.concatenate((y0_test.reshape(-1), y1_test.reshape(-1)), axis=0)
            
            print("Dimesion of training set is: {} and label is: {}".format (X_train.shape, y_train.shape))
            print("Dimesion of testing set is: {} and label is: {}".format( X_test.shape, y_test.shape))
        
            # classification
            accuracy, report, sen, spec, X_test_csp, y_true, y_pred, classifier = fbcsp(X_train=X_train, y_train=y_train,
                                                                                        X_test=X_test, y_test=y_test, 
                                                                                        filter_order=filter_order, session='mi')
            dump(classifier, save_path+'/model/'+subject+save_name+'_'+str(index_fold+1).zfill(2)+'.gz') 
            
            # saving
            precision_0 = report['0.0']['precision']
            recall_0 = report['0.0']['recall']
            f1_0 = report['0.0']['f1-score']

            precision_1 = report['1.0']['precision']
            recall_1 = report['1.0']['recall']
            f1_1 = report['1.0']['f1-score']
            
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1 = report['weighted avg']['f1-score']

            accuracy_sum.append(accuracy)

            precision_0_sum.append(precision_0)
            recall_0_sum.append(recall_0)
            f1_0_sum.append(f1_0)

            precision_1_sum.append(precision_1)
            recall_1_sum.append(recall_1)
            f1_1_sum.append(f1_1)

            precision_sum.append(precision)
            recall_sum.append(recall)
            f1_sum.append(f1)
            sen_sum.append(sen)
            spec_sum.append(spec)

            row = [index_fold+1, accuracy,
                f1_0, f1_1, f1,
                recall_0, recall_1, recall,
                precision_0, precision_1, precision,
                sen, spec]


            predict_result.append([y_true, y_pred])
            X_csp_com.append(X_test_csp)

            with open(save_path+'/'+save_path+'_result.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)

            print(subject, 'save DONE!!!!')
            print('***************************************')
            print('***************************************')
            print('***************************************')
            print('***************************************')

        mean_value = [np.mean(accuracy_sum),
        np.mean(f1_0_sum), np.mean(f1_1_sum), np.mean(f1_sum),
        np.mean(recall_0_sum), np.mean(recall_1_sum), np.mean(recall_sum),
        np.mean(precision_0_sum), np.mean(precision_1_sum), np.mean(precision_sum),
        np.mean(sen_sum), np.mean(spec_sum)]

        sum_value_all_subjects.append(mean_value)

        np.savez(save_path+'/y_slice_wise/'+subject+save_name+'.npz', x = np.array(X_csp_com), y = np.array(predict_result))

        with open(save_path+'/'+save_path+'_result.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['mean', mean_value[0], 
            mean_value[1], mean_value[2], mean_value[3], 
            mean_value[4], mean_value[5], mean_value[6], 
            mean_value[7], mean_value[8], mean_value[9],
            mean_value[10], mean_value[11]])
            writer.writerow([])
    
    mean_all = np.mean(sum_value_all_subjects, axis=0)
    print(mean_all)

    with open(save_path+'/'+save_path+'_result.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['accuracy', 
                    '0.0 f1-score', '1.0 f1-score', 'average f1-score',
                    '0.0 recall', '1.0 recall', 'average recall',
                    '0.0 precision', '1.0 precision', 'average precision',
                    'sensitivity', 'specificity'
                    ])
        writer.writerows(sum_value_all_subjects)
        writer.writerow(mean_all)