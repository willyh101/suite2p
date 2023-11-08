import h5py
import time
import os

import numpy as np

from suite2p import classification, detection, default_ops, extraction, registration

mov_path = '/mnt/scratch/deepinterpolation/gcamp_8m_predict_result_poisson.h5'

my_ops = {
    # general
    'diameter': 12,
    'do_bidiphase': False,
    'save_mat': False,
    'save_NWB': False,
    'tau': 1,
    # 'preclassify': 0., # apply classifier before signal extraction with a prob of 0.3
    'combined': False,
    'nplanes': 1,
    'nchannels': 1,
    'fs': 60,
    
    # registration
    'do_registration': False, # 2 forces re-registration
    
    
    # cell extraction
    'denoise': False,
    # 'threshold_scaling': 1.0, # adjust the automatically determined threshold by this scalar multiplier, was 1. (WH) # 0.6 for low signal, default 5
    'sparse_mode': False,
    # 'max_iterations': 50, # usualy stops at threshold scaling, default 20
    # 'high_pass': 100,  # running mean subtraction with window of size 'high_pass' (use low values for 1P), default 100
    # 'classifier_path': 'c:/users/will/code/suite2p/suite2p/classifiers/classifier_8m.npy',
    
    # etc for bypass run script
    'save_path': '/mnt/scratch/deepinterpolation/la_suite2p',
    'save_path0': '/mnt/scratch/deepinterpolation/la_suite2p'
}

ops = {**default_ops(), **my_ops}

ops['save_folder'] = 'suite2p'
save_folder = os.path.join(ops['save_path0'], ops['save_folder'])
os.makedirs(save_folder, exist_ok=True)

print('***Loading movie***')
with h5py.File(mov_path, 'r') as f:
    mov = f['data'][:,:256,:]
print('done.')


n_frame, Lx, Ly = mov.shape

ops['Ly'] = Lx
ops['Lx'] = Ly
ops['yrange'] = np.array([0,ops['Ly']])
ops['xrange'] = np.array([0,ops['Lx']])

ops['meanImg'] = mov[:2000,...].mean(axis=0)
meanImgE = registration.register.compute_enhanced_mean_image(ops['meanImg'].astype(np.float32), ops)
ops['meanImgE'] = meanImgE


# Select file for classification
ops_classfile = ops.get('classifier_path')
builtin_classfile = classification.builtin_classfile
user_classfile = classification.user_classfile
if ops_classfile:
    print(f'applying classifier {str(ops_classfile)}')
    classfile = ops_classfile
elif ops['use_builtin_classifier'] or not user_classfile.is_file():
    print(f'applying builtin classifier at {str(builtin_classfile)}')
    classfile = builtin_classfile
else:
    print(f'applying default {str(user_classfile)}')
    classfile = user_classfile

######## CELL DETECTION ##############
t11=time.time()
print('----------- ROI DETECTION')
ops, stat = detection.detection_wrapper(mov, ops=ops, classfile=classfile)
tx= time.time()-t11
print(f'----------- Total {tx:.2f} sec.' )

######## ROI EXTRACTION ##############
t11=time.time()
print('----------- EXTRACTION')
stat, F, Fneu, F_chan2, Fneu_chan2 = extraction.extraction_wrapper(stat, mov, ops=ops)

tx= time.time()-t11
print(f'----------- Total {tx:.2f} sec.' )

######## ROI CLASSIFICATION ##############
t11=time.time()
print('----------- CLASSIFICATION')
if len(stat):
    iscell = classification.classify(stat=stat, classfile=classfile)
else:
    iscell = np.zeros((0, 2))
print(f'----------- Total {tx:.2f} sec.' )

spks = np.zeros_like(F)


np.save(os.path.join(save_folder, 'stat.npy'), stat)
np.save(os.path.join(save_folder, 'F.npy'), F)
np.save(os.path.join(save_folder, 'Fneu.npy'), Fneu)
np.save(os.path.join(save_folder, 'iscell.npy'), iscell)
np.save(os.path.join(save_folder, 'spks.npy'), spks)
np.save(os.path.join(save_folder, 'ops.npy'), ops)