from mne.io import Raw
import mne
from scot.eegtopo.eegpos3d import positions as eeg_locations
from scot.utils import cuthill_mckee
from scot.eegtopo.topoplot import Topoplot
from scot import plotting
import matplotlib.pyplot as plt
import scot
import numpy as np
EEG_dirs = '/home/uais/EEG_dev/d151'
fn_raw = EEG_dirs + '/sample_raw.fif'
fn_new_raw = fn_raw[:fn_raw.rfind('_raw.fif')] + '_new-raw.fif'
# Set random seed for repeatable results
np.random.seed(42)

cl_data = False
# Drop noisy channels and cut the interest time period
if cl_data:
    start, stop = 100, 200
    raw = Raw(fn_raw, preload=True)
    raw.info['bads'] += ['ECG']
    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    raw.save(fn_new_raw, tmin=start, tmax=stop, picks=picks,
            overwrite=True)

apl_scot = True
if apl_scot:
    raw_new = Raw(fn_new_raw)
    data, times = raw_new[:]
    ds = np.array([data[:, :50000],data[:, 50000:-1]])
    fs = int(raw_new.info['sfreq'])
    classes = ['abnorm', 'norm']
    labels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
              'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1','FC2',
              'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz',
               'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4',
               'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6',
               'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9',
               'FT10', 'Fpz', 'CPz']
    locs = [[v for v in eeg_locations[l].vector] for l in labels]
    ws = scot.Workspace({'model_order': 40}, reducedim=15, fs=fs, locations=locs)
    ws.set_data(ds, classes)
    ws.do_cspvarica()
    p = ws.var_.test_whiteness(50)
    print('Whiteness:', p)
    
    freq = np.linspace(0, fs, ws.nfft_)
    alpha, beta = {}, {}
    for c in np.unique(classes):
        ws.set_used_labels([c])
        ws.fit_var()
        con = ws.get_connectivity('ffDTF')
        alpha[c] = np.mean(con[:, :, np.logical_and(8 < freq, freq < 12)], axis=2)
    
    # Prepare topography plots
    topo = Topoplot()
    topo.set_locations(locs)
    mixmaps = plotting.prepare_topoplots(topo, ws.mixing_)
    
    # Force diagonal (self-connectivity) to 0
    np.fill_diagonal(alpha['abnorm'], 0)
    np.fill_diagonal(alpha['norm'], 0)
    
    order = None
    for cls in ['abnorm', 'norm']:
        np.fill_diagonal(alpha[cls], 0)
    
        w = alpha[cls]
        m = alpha[cls] > 4
    
        # use same ordering of components for each class
        if not order:
            order = cuthill_mckee(m)
    
        # fixed color, but alpha varies with connectivity strength
        r = np.ones(w.shape)
        g = np.zeros(w.shape)
        b = np.zeros(w.shape)
        a = (alpha[cls]-4) / max(np.max(alpha['abnorm']-4), np.max(alpha['norm']-4))
        c = np.dstack([r, g, b, a])
    
        plotting.plot_circular(colors=c, widths=w, mask=m, topo=topo,
                            topomaps=mixmaps, order=order)
        plt.title(cls)
    
    plotting.show_plots()
