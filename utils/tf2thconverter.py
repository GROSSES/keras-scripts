# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 03:11:20 2016

@author: gpu2
"""



from keras import backend as K
from keras.utils.np_utils import convert_kernel
import res_net50 # this is just for an example, comment if you want to use this script
import h5py

model = res_net50.get_resnet50() # load the weight
model.load_weights('tf_resnet50.h5') # load tf/ date
for layer in model.layers:
   if layer.__class__.name in ['Convolution2D', 'Convolution1D']:
      original_w = K.get_value(layer.W)
      converted_w = convert_kernel(original_w)
      K.set_value(layer.W, converted_w)
model.save_weights('th_resnet50.h5') # save weights

"""

f_th = h5py.File('thresnet50.h5','w')
f_tf = h5py.File('resnet50.h5','r')
print f_tf.keys()
for k in f_tf.keys():
    grp = f_th.create_group(k)
    if k[:3]=='res' or k[:4]=='conv':
        grp.create_dataset('weights',data=convert_kernel(f_tf[k]['weights'][:]))

    else:
        grp.create_dataset('weights',data=f_tf[k]['weights'][:])
    
    grp.create_dataset('bias',data=f_tf[k]['bias'][:])
f_th.close()
f_tf.close()
      
"""