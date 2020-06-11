function out = rASR(file_name, chanlocs, PATH)

    p1 = genpath(strcat(PATH, '/eeglab2019_0'));
    addpath(p1)

    eeglab;
    
    eeg_data = load(file_name).data
    EEGin = pop_importdata('dataformat', 'array', 'data', eeg_data, 'srate', 250, 'xmin', 0, 'nbchan', 11, 'chanlocs', chanlocs);

    arg_flatline = 5;
    arg_highpass = [0.25 0.75];
    arg_channel = -1;
    arg_noisy = -1;
    arg_burst = 5;
    arg_window = -1;
    EEGout = clean_rawdata(EEGin, arg_flatline, arg_highpass, arg_channel, arg_noisy, arg_burst, arg_window);

    out = EEGout.data;