from suite2p import default_ops
import h5py
from datetime import datetime

mov_path = '/mnt/scratch/deepinterpolation/gcamp_8m_predict_result_poisson.h5'

my_ops = {
    # general
    'diameter': 10,
    'do_bidiphase': True,
    'save_mat': False,
    'save_NWB': False,
    'tau': 1.0,
    # 'preclassify': 0., # apply classifier before signal extraction with a prob of 0.3
    'combined': False,
    'h5py_key': 'data',
    
    'h5py': ''
    
    # registration
    'do_registration': False, # 2 forces re-registration
    'keep_movie_raw': True, # must be true for 2 step reg
    'two_step_registration': True,
    'nimg_init': 2000, # subsampled frames for finding reference image
    'batch_size': 500, #2000, # number of frames per batch, default=500
    'align_by_chan': 1, # 1-based, use 2 for tdT
    
    # non rigid registration settings
    'nonrigid': False, # whether to use nonrigid registration
    
    # cell extraction
    'denoise': False,
    'threshold_scaling': 2.0, # adjust the automatically determined threshold by this scalar multiplier, was 1. (WH) # 0.6 for low signal, default 5
    'sparse_mode': False,
    'max_iterations': 50, # usualy stops at threshold scaling, default 20
    'high_pass': 100,  # running mean subtraction with window of size 'high_pass' (use low values for 1P), default 100
    # 'classifier_path': 'c:/users/will/code/suite2p/suite2p/classifiers/classifier_8m.npy',
    

    # etc for bypass run script
    'save_path': '/mnt/scratch/deepinterpolation/la_suite2p'
}

ops = {**default_ops(), **my_ops}
ops['date_proc'] = datetime.now()


print('***Loading movie***')
with h5py.File(mov_path, 'r') as f:
    mov = f['data'][:,:256,:]
print('done.')

n_frame, Lx, Ly = mov.shape

ops['Ly'] = Lx
ops['Lx'] = Ly