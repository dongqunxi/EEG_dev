import mne
from scot.eegtopo.eegpos3d import positions as eeg_locations
from scot.utils import cuthill_mckee
from scipy import signal
from scot.eegtopo.topoplot import Topoplot
from scot import plotting
import matplotlib.pyplot as plt
import scot
from scot.matfiles import loadmat
import numpy as np
import glob
from scipy.stats.mstats import zscore

#Set random seed for repeatable results
np.random.seed(42)

EEG_dirs = '/home/uais_common/EEG_dev/static'
factor = 1
fs = 1000 / factor
morder = 100
comp = 4
p_v = 0.05
rep = 1000
#Load filtered abnormal data
fn_list1 = glob.glob(EEG_dirs + '/filter/addi/*.mat')
ori_data = []
for fn_name in fn_list1:
    mat_data = loadmat(fn_name)
    b = mat_data['b']
    new_b = scot.datatools.cut_segments(b[:, :240000], np.arange(10000, 250000, 10000), -10000, 0)
    ori_data += list(new_b)
    
#Load filtered normal data 
#normal_dirs = '/home/uais/EEG_dev/normal_male'
fn_list2 = glob.glob(EEG_dirs + '/filter/normal/*.mat')
for fn_name in fn_list2:
    mat_data = loadmat(fn_name)
    b = mat_data['b']
    new_b = scot.datatools.cut_segments(b[:, :240000], np.arange(10000, 250000, 10000), -10000, 0)
    ori_data += list(new_b)
    

do_fil_tog = False
if do_fil_tog:
    ori_data = np.array(ori_data)
    if factor > 1:
        ori_data = signal.decimate(ori_data, q=factor, axis=-1) 
    h_data = np.hstack(ori_data)
    z_data = zscore(h_data, axis=-1)
    s_data = np.split(z_data, 20, axis=-1)
    data = np.array(s_data)
do_fil_sep = True
if do_fil_sep:
    ori_data = np.array(ori_data)
    if factor > 1:
        ori_data = signal.decimate(ori_data, q=factor, axis=-1) 
    ori1 = ori_data[:240]
    h_data = np.hstack(ori1)
    z_data = zscore(h_data, axis=-1)
    s_data = np.split(z_data, 240, axis=-1)
    data1 = np.array(s_data)
    class1 = []
    for i in range(data1.shape[0]):
        class1.append('addi')
    ori2 = ori_data[240:]
    h_data = np.hstack(ori2)
    z_data = zscore(h_data, axis=-1)
    s_data = np.split(z_data, 240, axis=-1)
    data2 = np.array(s_data)
    class2 = []
    for j in range(data2.shape[0]):
        class2.append('norm')
    data = np.vstack((data1, data2))
    
#classes = ['addi', 'addi', 'addi', 'addi', 'addi', 'addi', 'addi', 'addi', 'addi', 'addi', 
#         'norm', 'norm', 'norm', 'norm', 'norm', 'norm', 'norm', 'norm', 'norm', 'norm']
labels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
            'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1','FC2',
            'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz',
            'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4',
            'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6',
            'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9',
            'FT10', 'Fpz', 'CPz']
locs = [[v for v in eeg_locations[l].vector] for l in labels]
# Construct workspace
ws = scot.Workspace({'model_order': morder}, reducedim=comp, fs=fs, locations=locs)
# Fit filtered data into the workspace and perform cspvarica
#data = abs(data)
classes = class1 + class2
ws.set_data(data, classes)
#ws.do_cspvarica()
ws.do_mvarica(varfit='class')
p = ws.var_.test_whiteness(morder)
print('Whiteness:', p)
print ws.var_.is_stable()

# Prepare unflitered data
#Load filtered abnormal data
fn_list1 = glob.glob(EEG_dirs + '/unfilter/addi/*.mat')
ori_data = []
for fn_name in fn_list1:
    mat_data = loadmat(fn_name)
    b = mat_data['b']
    new_b = scot.datatools.cut_segments(b[:, :240000], np.arange(10000, 250000, 10000), -10000, 0)
    ori_data += list(new_b)
    
#Load filtered normal data 
#normal_dirs = '/home/uais/EEG_dev/normal_male'
fn_list2 = glob.glob(EEG_dirs + '/unfilter/normal/*.mat')
for fn_name in fn_list2:
    mat_data = loadmat(fn_name)
    b = mat_data['b'] 
    new_b = scot.datatools.cut_segments(b[:, :240000], np.arange(10000, 250000, 10000), -10000, 0)
    ori_data += list(new_b)

#data zescoring
do_unfil_tog = False
if do_unfil_tog:
    ori_data = np.array(ori_data)
    h_data = np.hstack(ori_data)
    z_data = zscore(h_data, axis=-1)
    s_data = np.split(z_data, 20, axis=-1)
    data = np.array(s_data)
    
do_unfil_sep = True
if do_unfil_sep:
    ori_data = np.array(ori_data)
    if factor > 1:
        ori_data = signal.decimate(ori_data, q=factor, axis=-1) 
    ori1 = ori_data[:240]
    h_data = np.hstack(ori1)
    z_data = zscore(h_data, axis=-1)
    s_data = np.split(z_data, 240, axis=-1)
    data1 = np.array(s_data)
    class1 = []
    ori2 = ori_data[240:]
    h_data = np.hstack(ori2)
    z_data = zscore(h_data, axis=-1)
    s_data = np.split(z_data, 240, axis=-1)
    data2 = np.array(s_data)
    data = np.vstack((data1, data2))

ws.set_data(data, classes)
ws.plot_f_range = [1, 45]
ws.plot_diagonal = 'topo'
ws.plot_outside_topo = False

p = ws.var_.test_whiteness(morder)
print('Whiteness:', p)
#############################################
ws.set_used_labels(['addi'])
ws.fit_var()
print ws.var_.is_stable()
#-----------------------------
# Configure plotting options


fig1 = ws.plot_connectivity_topos()
ws.get_connectivity('GPDC', fig1)
ws.get_surrogate_connectivity('GPDC', plot=fig1, repeats=1000)
fig1.suptitle('heroin abstinent')

ws.set_used_labels(['norm'])
ws.fit_var()
print ws.var_.is_stable()
#ws.do_mvarica(varfit='class')
#-----------------------------
# Configure plotting options
fig2 = ws.plot_connectivity_topos()
ws.get_connectivity('GPDC', fig2)
ws.get_surrogate_connectivity('GPDC', plot=fig2, repeats=1000)
fig2.suptitle('non-addicted')
ws.show_plots()


#ws.do_mvarica(varfit='class')
#p = ws.var_.test_whiteness(morder)
#print('Whiteness:', p)
fig = ws.plot_connectivity_topos()
p, s, _ = ws.compare_conditions(['addi'], ['norm'], 'GPDC', alpha=p_v, repeats=rep,
                                plot=fig)

ws.show_plots()

