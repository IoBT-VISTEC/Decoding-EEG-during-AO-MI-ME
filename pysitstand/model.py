import numpy as np
from sklearn.svm import SVC
from mne.decoding import CSP
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from pysitstand.utils import sen_spec, butter_bandpass_filter

def fbcsp(X_train, y_train, X_test, y_test, filter_order=2, session='mi'):
    
    '''
    X_train, X_test: EEG data, 3D numpy array (#windows, #channels #timepoint)
    y_train, y_test: labels, 1D numpy array (#windows)
    '''

    if session == 'mi':
        filters = [[4, 8], [8, 12], [12, 16], 
                [16, 20], [20, 24], [24, 28], 
                [28, 32], [32, 36], [36, 40]]
    elif session == 'me':
        filters = [[0.1, 0.5], [0.5, 1], [1, 1.5], 
                   [1.5, 2], [2, 2.5], [2.5, 3]]

    tuned_parameters = [{'kernel': ['rbf'], 
                     'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.01, 0.1, 10, 25, 50, 100, 1000],
                     'class_weight': ['balanced']},
                    {'kernel': ['sigmoid'], 
                     'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.01, 0.1, 10, 25, 50, 100, 1000], 
                     'class_weight': ['balanced']},
                    {'kernel': ['linear'], 
                     'gamma' : ['auto'], 
                     'C':[0.001, 0.01, 0.1, 10, 25, 50, 100, 1000],
                     'class_weight': ['balanced']}]
  
    n_components = 3   
    n_features = 9

    n_fbank = len(filters)   
    
    # csp = CSP(n_components=n_components, norm_trace=False)
    X_train_fbcsp = np.zeros([X_train.shape[0], n_fbank, n_components])
    X_test_fbcsp = np.zeros((X_test.shape[0], n_fbank, n_components))

    fbcsp = {} # dict
    for idx, (f1,f2) in enumerate(filters, start=0):        
        X_train_fb = butter_bandpass_filter(X_train, f1, f2, fs=250, order=filter_order)
        X_test_fb = butter_bandpass_filter(X_test, f1, f2, fs=250, order=filter_order)
        csp = CSP(n_components=n_components, norm_trace=False)
        X_train_fbcsp[:, idx, :] = csp.fit_transform(X_train_fb, y_train) 
        fbcsp[(f1,f2)] = csp
        for n_sample in range(X_test_fb.shape[0]):
            csp_test = X_test_fb[n_sample, :, :].reshape(1, X_test_fb.shape[1], X_test_fb.shape[2])
            X_test_fbcsp[n_sample, idx, :] = csp.transform(csp_test)

    nsamples, nx, ny = X_train_fbcsp.shape
    X_train_fbcsp = X_train_fbcsp.reshape((nsamples, nx*ny))

    nsamples, nx, ny = X_test_fbcsp.shape
    X_test_fbcsp = X_test_fbcsp.reshape((nsamples, nx*ny))
    
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_train_fbcsp = selector.fit_transform(X_train_fbcsp, y_train)
    X_test_fbcsp = selector.transform(X_test_fbcsp)        

    print("Dimesion of training set is: {} and label is: {}".format(X_train_fbcsp.shape, y_train.shape))
    print("Dimesion of testing set is: {} and label is: {}".format(X_test_fbcsp.shape, y_test.shape))

    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    clf = GridSearchCV(SVC(), tuned_parameters, cv=inner_cv, scoring = 'accuracy', iid=True)
    clf.fit(X_train_fbcsp , y_train)
    #Clasifying with an optimal parameter set
    Optimal_params = clf.best_params_
    print(Optimal_params)
    classifier = SVC(**Optimal_params)
    classifier.fit(X_train_fbcsp , y_train)
    y_true, y_pred = y_test, classifier.predict(X_test_fbcsp)
    svm_acc = classifier.score(X_test_fbcsp, y_test)
    sen, spec = sen_spec(y_true, y_pred)
    print('X_test CSP shape:',X_test_fbcsp.shape)
    print("Classification accuracy:",svm_acc)
    print(classification_report(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)

    model = {'filters': filters,'fbcsp': fbcsp,'SelectKBest': selector, 'classifier': classifier}

    return svm_acc, report, sen, spec, X_test_fbcsp, y_true, y_pred, model
