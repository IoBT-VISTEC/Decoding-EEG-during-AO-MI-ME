import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import butter, filtfilt, iirnotch
from statistics import mean 
from scipy import signal
from pysitstand.utils  import *
from pysitstand.info  import DATASET_PATH
# Data description
smp_freq = 250
data_len = 16
used_data_lean = 15
data_points = int(smp_freq*data_len)
used_data_points = int(data_points - (1*smp_freq)) # we have removed the first second out from resting state period
num_chs = 8 
num_trial = 5
num_run = 3
num_persons = 8
filter_order = 2

def readHeader(raw_data):
    phase = raw_data[1]
    scenario = raw_data[3]
    subjectno = raw_data[5]
    gender = raw_data[7]
    age = raw_data[9]
    s_type = raw_data[11]
    count  = raw_data[13]
    
    if scenario == 1:
        s_type = 0 # sit = 0 
    elif scenario == 2:
        s_type = 1 # stand = 1
    elif scenario == 3:
        if count%2==1:
            s_type = 3
        else:
            s_type = 2
    elif scenario == 4:
        s_type = 4
    elif scenario == 5:
        s_type = 5
    elif scenario == 6:
        s_type = 6
    elif scenario == 7:
        s_type = 7
    return scenario, count, phase, subjectno, gender, age, s_type

def reshapeIndex(index_value):
    tmp_a = []
    tmp_b = []
    reshape_index = []
    for i in range(len(index_value)):
        for j in range(len(index_value[i])):
            tmp_a.append(index_value[i][j])
            if index_value[i][j,3] == 5: # last phase
                tmp_b.append(np.array(tmp_a))
                tmp_a.clear()
        reshape_index.append(np.array(tmp_b))
        tmp_b.clear()
    reshape_index = np.array(reshape_index)
    return reshape_index

def split_scenario(scenario, data, index):
    return data[scenario-1], index[scenario-1] # list

def split_count(count, data, index):
    return data[count-1], index[count-1] # array

def split_phase(phase, data, index):
    if index[0,1] <= 2: # scenario 1 and 2
        if phase == 1 and index[0,3] == 1:
            return data[:,:int(index[1,0])]
        elif phase == 3 and index[1,3] == 3:
            return data[:,int(index[1,0]):int(index[2,0])]
        elif phase == 5 and index[2,3] == 5:
            return data[:,int(index[2,0]):]
    elif index[0,1] >= 3: # scenario 3,4,5,6,7
        if phase == 1 and index[0,3] == 1:
            return data[:,:int(index[1,0])]
        elif phase == 2 and index[1,3] == 2:
            return data[:,int(index[1,0]):int(index[2,0])]
        elif phase == 3 and index[2,3] == 3:
            return data[:,int(index[2,0]):int(index[3,0])]
        elif phase == 5 and index[3,3] == 5:
            return data[:,int(index[3,0]):]

def getData(raw_data):
    tmp = []
    re_shape = []
    isStart = False
    trial = 1
    j = 0
    key_index = []
    index_value = []
    data = []
    f = []
    for i in raw_data:
        if raw_data[i][0] == "phase" and not raw_data[i][1] == 0:      # start trial
            isStart = True
            scenario, count, phase, subjectno, gender, age, s_type = readHeader(raw_data[i])
            key_index.append(np.array([j, scenario, count, phase, s_type]).astype(np.int))
            if phase == 1 and not count == 1:
                j = 0
            # print('j=', j, i, scenario, count, phase," :: ",subjectno, gender, age, s_type)
        elif raw_data[i][0] == "phase" and raw_data[i][1] == 0:        # end of count (scenario)
            re_shape.append(np.array(tmp).T)
            data.append(list(re_shape)) # 
            re_shape.clear() #
            tmp.clear()
            trial = 1
            j = 0
            index_value.append(np.array(key_index))
            key_index.clear()
            # print("end of scenario:",scenario," : ",i)
        elif(isStart):      # append data to each trials
            if count == trial:
                tmp.append(raw_data[i][1:9].astype(np.float))
                j = j+1
            else:
                re_shape.append(np.array(tmp).T)
                tmp.clear()
                tmp.append(raw_data[i][1:9].astype(np.float))
                trial = trial+1
                j = j+1
    # key_index = np.array(key_index)  # [j, scenario, count, phase, s_type]
    index_value = reshapeIndex(np.array(index_value))
    return data, index_value
def remove_very_large_amp(data):
    for i in range(len(data)):
        for j in range(data[i].shape[-1]):
            if j > 10 and abs(data[i][j]-data[i][j-1]) > 900:
                data[i][j] = mean(np.concatenate((data[i][j-5:j], data[i][j+1]), axis=None))
                #data[i][j] = mean(data[i][j-5:j])
            elif j <= 10 and abs(data[i][j]-data[i][j+1]) > 900:
                data[i][j] = mean(data[i][j+1:j+6])
    return data

def preprocessing(data):
#     #data = standardize(data)
    data = remove_very_large_amp(data)
    data = notch_filter(data, 50, 250, 30)
    data = butter_bandpass_filter(data, 15, 124, 250, order=filter_order)
    return data

def processed_ME_EMG_each_run(data, index_raw, num_scenario):
    sensor_used = [1,2,3,4,6,7] #ignore 0, 5
    arr_data, index = split_scenario(num_scenario, data, index_raw)
    sit_arr = []
    stand_arr = []
    for i in range(10):
        tmp = preprocessing(arr_data[i][:,:])
        if i%2==0:
            sit_arr.append(tmp) #1200
        else:
            stand_arr.append(tmp)

    X_sit = np.zeros((num_trial, num_chs-2, used_data_points))
    X_stand = np.zeros((num_trial, num_chs-2, used_data_points))
    for i in range(num_trial):
        for id_sen, sensor in enumerate(sensor_used):
            X_sit[i,id_sen,:] = sit_arr[i][sensor,-used_data_points:]
            X_stand[i,id_sen,:] = stand_arr[i][sensor,-used_data_points:]
    
    return  X_sit , X_stand

def processed_MI_EMG_each_run(data, index_raw, num_scenario):
    data_used, index = split_scenario(num_scenario, data, index_raw)
    sensor_used = [1,2,3,4,6,7]
    data_arr = np.zeros((num_trial, num_chs-2, used_data_points))
    for i in range(5):
        tmp_data = preprocessing(data_used[i][:,-used_data_points:])
        data_arr[i,:,:] = tmp_data[sensor_used]
    
    return data_arr

# 1st step in pre-precesing on EMG data for ME session
def perform_precessing(subject_name): # should be in ['S01','S02,'S03,'S04','S05','S06','S07','S08']
    EMG_real_sit = np.zeros((num_run, num_trial, num_chs-2, used_data_points))
    EMG_real_stand = np.zeros((num_run, num_trial, num_chs-2, used_data_points))
    for index_run, run in enumerate(range(1, 4)):
        print("Current person_"+ subject_name+ "_Current round_", run)
        subject_path = DATASET_PATH+'/'+subject_name+'_EMG/'
        file_name = subject_path + subject_name+ '_EMG_' + str(run).zfill(2)+'.csv'
        raw_data = pd.read_csv(file_name , header=None, index_col=None)
        raw_data = raw_data.T
        arr_raw, index_raw = getData(raw_data)
        real_sit, real_stand = processed_ME_EMG_each_run(arr_raw, index_raw, num_scenario=3)
        EMG_real_sit[index_run, :, :, :] = real_sit
        EMG_real_stand[index_run, :, :, :] = real_stand
        
        final_EMG_real_sit = EMG_real_sit.reshape(-1,num_chs-2,used_data_points)
        final_EMG_real_stand = EMG_real_stand.reshape(-1,num_chs-2,used_data_points)
    return final_EMG_real_sit, final_EMG_real_stand 

# Next step we have carried out the 2nd pre-processing step
def tkeo(a):

	"""
	Calculates the TKEO of a given recording by using four samples.
	See Deburchgrave et al., 2008

	Arguments:
	a 			--- 1D numpy array.

	Returns:
	1D numpy array containing the tkeo per sample
	"""

	# Create two temporary arrays of equal length, shifted 1 sample to the right
	# and left and squared:
	
	l = 1
	p = 2
	q = 0
	s = 3
	
	aTkeo = a[l:-p]*a[p:-l]-a[q:-s]*a[s:]

	return aTkeo

def rectify_and_lowpass_filter(data, lowcut, fs, order):
    
    # process EMG signal: rectify
    data_rectified = abs(data)
    
    # create lowpass filter and apply to rectified signal to get EMG envelope 
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass')
    data_envelope = filtfilt(b, a, data_rectified)
    return data_envelope

# 2nd step in pre-precesing on EMG data for ME session
def final_perform_preprocessing(data_sit, data_stand):
    # After performing TKEO, the data poits will be removed the length of 3 points
    EMG_seats_envelope = np.zeros((data_sit.shape[0], data_sit.shape[1], data_sit.shape[2]-3)) 
    EMG_stand_envelope = np.zeros((data_stand.shape[0], data_stand.shape[1], data_stand.shape[2]-3))

    for i in range(data_sit.shape[0]):
        for j in range(data_sit.shape[1]):
            data_tkeo_seats = tkeo(data_sit[i,j,:])
            data_tkeo_stand = tkeo(data_stand[i,j,:])
            EMG_seats_envelope[i,j,:] = rectify_and_lowpass_filter(data_tkeo_seats, 3, smp_freq, order=filter_order)
            EMG_stand_envelope[i,j,:] = rectify_and_lowpass_filter(data_tkeo_stand, 3, smp_freq, order=filter_order)

    return EMG_seats_envelope, EMG_stand_envelope

# We created the detective onset for actual movements
def detective_onset(data, h):
    # selected the time interval from 6 s to 9 s as a quet or resting state (This period supposed that there were no any actual movements.)
    ref_signals = data[int(6*smp_freq):int(9*smp_freq)]
    mean_ref = ref_signals.mean()
    std_ref = ref_signals.std()
    T = mean_ref + (h*std_ref)
    # FIND onset
    list_index, list_val = [], []
    for index, val in enumerate(data[int(10*smp_freq):]): # start exploring detective onset since the 9-seconds.
        if val >= T: # there are movements in this period
            list_index.append(index)
            list_val.append(1)
        else: # there are no movements in this period
            list_index.append(index)
            list_val.append(0)
    # Make sure that the picked onset is correct. 
    for id_list, val_the in enumerate(list_val):
        if val_the == 1 and list_val[id_list+1:id_list+5] == [1,1,1,1] and list_val[id_list-4:id_list] == [0,0,0,0]:
#             print("Movement_onset_is", list_index[id_list]+(9*smp_freq))
            return list_index[id_list]+(10*smp_freq)
        
def apply_detective_onset(data_sit, data_stand, theshold=10):        
    # Default theshold=10
    EMG_onset_seats = np.zeros((data_sit.shape[0], data_sit.shape[1]))
    EMG_onset_stand = np.zeros((data_stand.shape[0], data_stand.shape[1]))
    for id_trials in range(data_sit.shape[0]):
        for id_ch in range(data_sit.shape[1]):
            EMG_onset_seats[id_trials, id_ch] = detective_onset(data_sit[id_trials, id_ch, :], theshold)
            EMG_onset_stand[id_trials, id_ch] = detective_onset(data_stand[id_trials, id_ch, :], theshold)
    # We calculated the mean of detective onsets for all sensors, including ignoring the sensor that cannot find the detective onset.
    EMG_onset_seats_mean = mean_without_nan(EMG_onset_seats)
    EMG_onset_stand_mean = mean_without_nan(EMG_onset_stand) 
    return EMG_onset_seats_mean, EMG_onset_stand_mean
 