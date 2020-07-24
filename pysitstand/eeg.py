import pandas as pd
import numpy as np
import glob
import os

class EEG:
    """ EEG extract and preprocessing 
    ****** If you want to edit, see ICA_plot and preprocessing function ********
    
    Parameters
    ----------
    csv_file_name : EEG .csv file name
    sampling_rate : sampling rate (Hz) of EEG raw data (defult is 1200 Hz)

    Returns
    ----------
    corrected_EEG/ : figures of EEG time-domain after remove EOG
    eog_avg/ : EOG average figures
    eog_score/ : EOG score of ICA components
    ica/ : ICA components of EOG
    montage/ : ICA components montages
    new_raw/ : EEG mne epoch raw .fif file (data was re)
                After run this file the data was filtered between 1-40 Hz, 
                resampled to 250 Hz and removed EOG components
                Each trial is 15s long
    raw_EEG/ : EEG figures before remove EOG signals
    
    """
    def __init__(self, csv_file_name, sampling_rate=1200):
        self.sampling_rate = sampling_rate
        self.csv_file_name = csv_file_name

    def read_CSV(self):
        """Read EEG raw data in block format (CSV file) into numpy array

        Parameters
        ----------
        csv_file_name : .csv file name of EEG raw data in block format

        Returns
        ----------
        raw_array :  EEG raw data in numpy array in block format

        """
        all_files = sorted(glob.glob(self.csv_file_name), key=os.path.getmtime)
        li = []
        for filename in all_files:
            df = pd.read_csv(filename, header=None)
            df.drop(df.index[0], inplace=True)
            print(df.shape)
            li.append(df)
        raw = pd.concat(li)
        raw_array = np.array(raw)
        return  raw_array

    def split_scenario(self, scenario, raw_array):
        """Split scenario of EEG raw data

        Parameters
        ----------
        scenario : scenario number
                    1: resting while sit
                    2: resting while stand
                    3: physical sit and stand
                    4: trying to stand
                    5: trying to sit
                    6: imagining to stand
                    7: imagining to sit
        raw_array : EEG raw data in numpy array in block format

        Returns
        ----------
        raw_scenario_data :  EEG raw data in numpy array in block format of that scenario 
        """
        scenario_data = []
        count_data = []
        tmp = []
        scenario_tmp = 0
        raw_scenario_data = []
        for i in range(len(raw_array)):
            if ( i%14  == 0): #HEADER       
                #print(raw_array[i])        
                time_stamp = raw_array[i,0]
                phase_tmp = raw_array[i,1]    #timing
                scenario_tmp = raw_array[i,2] #class
                subject_no = raw_array[i,3]
                gender = raw_array[i,4]
                age = raw_array[i,5]
                s_type = raw_array[i,6]    #sit/stand
                count_tmp = raw_array[i,7]   #trial
                #print(time_stamp, phase, scenario, subject_no, gender, age, s_type, count_tmp)
                if scenario == scenario_tmp:
                    raw_scenario_data.append(raw_array[i].tolist())
            elif scenario == scenario_tmp: #DATA
                raw_scenario_data.append(raw_array[i].tolist())

        raw_scenario_data = np.array(raw_scenario_data)
        return raw_scenario_data

    def split_count(self, count, raw_array):
        """Split count of EEG raw data

        Parameters
        ----------
        count : count number
        Note 
        scenatio 1 and 2 have 1 count (trial) 
        scenario 3 has 10 counts (trials), odd indicates sit to stand tasks and even indicates stand to sit tasks
        scenario 4-7 have 5 counts (trials)

        raw_array : EEG raw data in numpy array in block format

        Returns
        ----------
        raw_count_data :  EEG raw data in numpy array in block format of that count 
        """
        scenario_data = []
        count_data = []
        tmp = []
        raw_count_data = []
        for i in range(len(raw_array)):
            if ( i%14  == 0): #HEADER  
                time_stamp = raw_array[i,0]
                phase_tmp = raw_array[i,1]    #timing
                scenario_tmp = raw_array[i,2] #class
                subject_no = raw_array[i,3]
                gender = raw_array[i,4]
                age = raw_array[i,5]
                s_type = raw_array[i,6]    #sit/stand
                count_tmp = float(raw_array[i,7])   #trial
                #print(time_stamp, phase, scenario, subject_no, gender, age, s_type, count_tmp)
                if count == count_tmp:
                    raw_count_data.append(raw_array[i].tolist())
            elif count == count_tmp: #DATA
                raw_count_data.append(raw_array[i].tolist())        

        raw_count_data = np.array(raw_count_data)
        return raw_count_data

    def split_phase(self, phase, raw_array):
        """Split phase of EEG raw data

        Parameters
        ----------
        phase : phase number
        Note 
        phase 1 indicates resting
        phase 2 indicates video stimulation
        phase 3 indicates resting 1s after video
        phase 5 indicates performing the tasks

        raw_array : EEG raw data in numpy array in block format

        Returns
        ----------
        raw_phase_data :  EEG raw data in numpy array in block format of that phase 
        """
        scenario_data = []
        count_data = []
        tmp = []
        raw_phase_data = []

        for i in range(len(raw_array)):
            if ( i%14  == 0): #HEADER  
                time_stamp = raw_array[i,0]
                phase_tmp = float(raw_array[i,1])    #timing
                scenario_tmp = raw_array[i,2] #class
                subject_no = raw_array[i,3]
                gender = raw_array[i,4]
                age = raw_array[i,5]
                s_type = raw_array[i,6]    #sit/stand
                count_tmp = float(raw_array[i,7])   #trial
                #print(time_stamp, phase_tmp, scenario_tmp, subject_no, gender, age, s_type, count_tmp)
                if phase == phase_tmp:
                    raw_phase_data.append(raw_array[i].tolist())
            elif phase == phase_tmp: #DATA
                raw_phase_data.append(raw_array[i].tolist())        

        raw_phase_data = np.array(raw_phase_data)

        return raw_phase_data

    def extract_data(self, raw_array):
        """Extract EEG raw data in block format into time-domain format

        Parameters
        raw_array : EEG raw data in numpy array in block format

        Returns
        ----------
        f_array :  EEG time-domain raw data in numpy array 
        """
        tmp = []
        re_shape = []

        for i in range(len(raw_array)):   
            if ( i%14  == 0):      
                time_stamp = raw_array[i,0]
                phase = raw_array[i,1]    #timing
                scenario = raw_array[i,2] #class
                subject_no = raw_array[i,3]
                gender = raw_array[i,4]
                age = raw_array[i,5]
                s_type = raw_array[i,6]    #sit/stand
                count = raw_array[i,7]     #trial
                #print(time_stamp, phase, scenario, subject_no, gender, age, s_type, count)
            else:
                tmp.append(raw_array[i].tolist())
            if (i%14 == 13):
                tmp  = np.array(tmp).T
                re_shape.extend(tmp)
                tmp = []
        re_shape = np.array(re_shape).T
        t = np.arange(0, 1504.0, 1)
        s = 1 + np.sin(2*np.pi*t)

        f_array = re_shape.astype(np.float)
        return f_array

    def collect_data_allphase(self, scenario, raw_array):
        """Collect data all phase of senario number

        Parameters
        ----------
        scenario: scenario number
        raw_array : EEG raw data in numpy array in block format

        Returns
        ----------
        arr_sit, arr_stand :  list of EEG time-domain raw data of sit and stand tasks in scenario 3
        arr : list of of EEG time-domain raw data in other scenatio
        """
        data = self.split_scenario(scenario, raw_array)
        arr_sit = []
        arr_stand = []
        arr = []

        if scenario == 3:
            for i in range(1,11):
                if i%2==0: #sit
                    arr_sit.append(self.extract_data(self.split_count(i,data))[:,-18000:])
                else: #stand
                    arr_stand.append(self.extract_data(self.split_count(i,data))[:,-18000:])
            return np.array(arr_sit), np.array(arr_stand)
        elif scenario>3 and scenario<=7:
            for i in range(1,6):
                arr.append(self.extract_data(self.split_count(i,data))[:,-18000:])
            return np.array(arr)
        elif scenario <3:
            return self.extract_data(data)[:,:]
