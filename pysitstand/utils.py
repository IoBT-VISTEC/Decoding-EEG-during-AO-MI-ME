# Preprocessing
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.metrics import confusion_matrix
import random
import string

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def highpass_filter(data, highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='high')
    y = filtfilt(b, a, data)
    return y

def notch_filter(data, f0, fs, Q): # Q = Quality factor
    w0 = f0/(fs/2)
    b, a = iirnotch(w0, Q)
    y = filtfilt(b, a, data)
    return y

def sliding_window2(data, win_sec_len, step, sfreq): 
    if len(data.shape) == 3:
        len_data_point = data.shape[2]
        win_len_point = int(win_sec_len*sfreq)
        number_window = int(((len_data_point-win_len_point)/(win_len_point*step))+1)
        data_slided = np.ones((data.shape[0], data.shape[1], number_window, win_len_point))
        for i in range(data.shape[0]):#number of sample
            for j in range(data.shape[1]): #number of channel
                for idx in range(number_window): #num of slice 
                    start_pos = int(idx * win_len_point * step) 
                    stop_pos = int(start_pos + win_len_point) 
                    data_slided[i, j, idx, :] = data[i, j, start_pos:stop_pos]
        data_swap = np.swapaxes(data_slided,1,2)
        print("DONE!!!", "Dimension of data is:", data_swap.reshape(-1, data_swap.shape[2], data_swap.shape[3]).shape)
        return data_swap.reshape(-1, data_swap.shape[2], data_swap. shape[3])
    else:
        print("Wrong dimension")

def sliding_window(data, win_sec_len, step, sfreq):
    if len(data.shape) == 3:
        len_data = data.shape[2]
        step = step 
        win_len = int(win_sec_len*sfreq)
        num_win = int(((len_data - win_len)/(win_len * step))+1) # A number of sliding window
        data_slid = np.zeros((data.shape[0], num_win, data.shape[1], win_len))

        for sample in range(data.shape[0]):             # number of samples
            for idx_win in range(num_win):              # number of slices
                for channel in range(data.shape[1]):    # number of channels
                    start_pos = int(idx_win * win_len * step)
                    stop_pos  = int(start_pos + win_len)
                    data_slid[sample, idx_win, channel, :] = data[sample, channel, start_pos:stop_pos]
        data_slid = data_slid.reshape(-1, data.shape[1], win_len)
        return data_slid
    else:
        print("Wrong dimension")
        
def sen_spec(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    total=sum(sum(cm))
    accuracy=(cm[0,0]+cm[1,1])/total
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    return sensitivity, specificity

def mean_without_nan(data):
    data_mean = np.zeros((data.shape[0]))
    if len(data.shape) ==  2:
        for i in range(data.shape[0]):
            data_used = data[i, :]
            data_isnan_mean = data_used[~np.isnan(data_used)].mean()
            data_mean[i] = data_isnan_mean
        return data_mean
    else:
        print("Error dimesion") 

def reshape2Dto3D(data, trials):
    channels, datapoint = data.shape
    timepoint = int(datapoint//trials)
    y = np.zeros((channels,trials,timepoint))
    for i in range(len(data)):
        y[i] = data[i].reshape(trials, timepoint)
    return np.swapaxes(y, 0, 1)

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))
