'''
Plot some images as RGB from the Swisstopo dataset
loop over the classes to produce a 5x5 images tiles per class
'''
import os, glob, random, pickle
import rasterio as rio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms
import torch

class RescaleQuantile99(object):
    '''
        Rescale image color between 0 and 1 from min and max value of each band
        torch.floatTensor as input and output.
    '''
    def __call__(self, img):
        if img.shape ==(5,200,200):
            out=np.float32(img)
            out[0,:,:] = img[0,:,:]/ ( np.quantile(img[0,:,:],0.99) +1e-3)
            out[1,:,:] = img[1,:,:]/ ( np.quantile(img[1,:,:],0.99) +1e-3)
            out[2,:,:] = img[2,:,:]/ ( np.quantile(img[2,:,:],0.99) +1e-3)
            out[3,:,:] = img[3,:,:]/ ( np.quantile(img[3,:,:],0.99) +1e-3)
            out[4,:,:] = (img[4:,:] -475  )/(3242. -475)
            out[out>1]=1
            out[out<0]=0
            return out
        else :
            print('Rescale failed,  original image returned') 
            return img  
class ToTensorNP(object):
    '''
        Converts PIL images to tensor via a NumPy array.
        Needed because torchvision.transforms.ToTensor() requires
        unsigned 8-bit integer images as inputs.
    '''
    def __call__(self, img):
        out = torch.from_numpy(np.array(img))
        if out.dim() == 2:              # one-band image (WxH)
            out = out.unsqueeze(0)      # 1xWxH
        return out

# 1. Load paths, dictionnary, data
#-----------------------------------
data_path = '/home/valerie/data/refsurface/'
dico_path = '/home/valerie/Python/landuse/Dictionnaries/'

# Load labels from the dictionnary
os.chdir(dico_path)
with open('dict_NOLU46_all.p', 'rb') as fp:
    RELI2class = pickle.load(fp)

# Load dictionnary of class to text label
with open('NOLU_class2txt.p', 'rb') as fp:
    class2txt = pickle.load(fp)

# Read all tiff files
os.chdir(data_path)
tif = glob.glob('*.tif')
tif.sort()
N = len (tif)

# Define tranformations to apply : 
transforms_data = transforms.Compose([ RescaleQuantile99(), ToTensorNP()])       
klass = list(class2txt.keys()) # list of all classes
for k in klass:
    print(k, class2txt[ k  ])

#2. loop over the classes to produce a 5x5 images tiles
#------------------------------------------------------
klass= [303]

for k in klass:         # loop over all the classes
    lab = class2txt[ k  ]
    forest = []
    sel = []
    
    for i in range(N): # loop over all images 
        if RELI2class[ int( tif[i][:-4]  ) ] == k :
            forest += [tif[i]] # select images from the class k of interest

    print(k,lab, len(forest))
    if  len(forest) <25: 
        #print('\tRemoved:',k,lab, len(forest))
        sel = forest
    else:
        sel = random.sample(forest, k=25)
   
    idx =0

    for image in sel: # loop over the images of the selected class

        tmp = rio.open (data_path + image).read() # read image as np array
        tmp = transforms_data(tmp)
        tmp = np.float32(tmp)
        tmp = tmp[1:4,:,:]  # select RGB channels
        tmp = tmp*255
        #tmp.shape
        tmp = np.moveaxis(tmp, [0, 1, 2], [2,0,1])
        tmp = np.uint8(tmp)
        #x = plt.subplot(5,5,idx)
        #plt.subplots_adjust(wspace = .001,hspace = .001)
        x = plt.figure()
        x = plt.imshow(tmp)
        x = plt.axis('off')
        idx +=1
        fn = '/home/valerie/Python/landuse/Reporting/'+ lab+'_' +str(idx) +'.png'
        plt.savefig(fn,dpi=300,bbox_inches='tight')
        print(fn)

