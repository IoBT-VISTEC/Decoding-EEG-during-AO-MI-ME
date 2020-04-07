# Decoding EEG Rhythms During Action Observation, Motor Imagery and Execution for Standing and Sitting

## Abstract 
 **Event-related desynchronization and synchronization (ERD/S) and movement-related cortical potential (MRCP) play an important role in brain-computer interfaces (BCI) for lower limb rehabilitation, particularly in standing and sitting. However, little is known about the differences in the cortical activation between standing and sitting, especially how the brain's intention modulates the pre-movement sensorimotor rhythm as they do for switching movements. In this study, we aim to investigate the decoding of continuous EEG rhythms during action observation (AO), motor imagery (MI), and motor execution (ME) for standing and sitting. We developed a behavioral task in which participants were instructed to perform both AO and MI/ME in regard to the actions of sit-to-stand and stand-to-sit. Our results demonstrated that the ERD was prominent during AO, whereas ERS was typical during MI at the alpha band across the sensorimotor area. A combination of the filter bank common spatial pattern (FBCSP) and support vector machine (SVM) for classification was used for both offline and pseudo-online analysis. The offline analysis indicated the classification of AO and MI providing the highest mean accuracy at 82.73±2.38 in stand-to-sit transition. The results were acceptable in comparison to the original FBCSP study of right hand and right foot activation classifications. By applying the pseudo-online analysis, we demonstrated the possibility of decoding neural intentions from the integration of both AO and MI. These observations led us to the promising aspect of using our developed tasks to build future exoskeleton-based rehabilitation systems.**
 
## Data Description
![protocol](fig/timeline.png)
<p align="center"> 
<b>Fig. 1</b> Timeline of each experimental trial. The four states displayed include resting (0--4 s), AO (4--8 s), idle (8--9 s), and task performing, either MI or ME (9--13 s). 
</p>

## Experimental protocol

To investigate the feasibility of decoding the MI and MRCP signals during the intended movement executions with continuous EEG recordings, the entire experimental procedure composed of two sessions: MI and ME. Each session consisted of 3 runs (5 trials each), incorporating a total of 30 trials. The protocol began with a sitting posture, followed by 5 repeated trials of sit-to-stand and stand-to-sit tasks alternatively. Figure 1 displays the sequence of four states in each trial: R, AO, idle, and task performing states (MI or ME). During the R state, a black screen was displayed on the monitor for 5 seconds (s). The participants were instructed to remain relaxed and motionless. To avoid the ambiguity of the instructions, a video stimulus showing the sit-to-stand or stand-to-sit video task lasted for 4 to 5 s was presented to guide the participants in the AO state. The participants were instructed to perform the given task following an audio cue (beep) within 4 s. In the ME, the participants were to complete a succession of self-paced voluntary movement executions. Whereas in the MI, the participants were to commence motion imagining immediately after the audio cue.

![EEG and EOG setup](fig/EEG-electrodes.001.png)
<p align="center"> 
<b>Fig. 2</b> The channel configuration of the International 10-20 system (11 EEG and 2 EOG recording electrodes). The left panel corresponding location of each electrode; The right panel indicates the indexing. 
</p>

### EEG and EOG signals

* A _g.USBamp RESEARCH_ was used to recored EEG and EOG signals as displyed in Figure 2.
* The sampling rate was set at 1200 Hz.
* **EEG**: 11 electrodes were placed on *FCz*, *C3*, *Cz*, *C4*, *CP3*, *CPz*, *CP4*, *P3*, *Pz*, *P4*, and *POz*
* **EOG**: 2 electrodes were placed on under (*VEOG*) and next (*HEOG*) to the outer canthus of the right eye
* The impedance of both EEG and EOG signals was maintained at below 10 *k*Ω throughout the experiment

![EMG setup](fig/EMG_data_description_new.001.jpeg)
<p align="center"> 
<b>Fig. 2</b> The channel configuration of the 6 EMG recording electrodes, which shows the indexing corresponding location of each electrode. 
</p>

### EMG signals
* An _OpenBCI_ was used to recorded EMG signals.
* The sampling rate was set at 250 Hz.
* 6 electrodes were placed on rectus femoris (*RF*), tibialis anticus (*TA*), and  gastrocnemius lateralis (*GL*) of two lower limbs
* There was only recording EMGs on ME session
* The raw EMG of each sit-to-stand/stand-to-sit transition was formed in a dimension of participants￼*runs*￼trials*￼channels*￼time points (8￼*3*￼5*￼6*￼3500).


