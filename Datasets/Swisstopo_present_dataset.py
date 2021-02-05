''''
Loop over the classes to produce a images tiles

'''

import os, glob, random, pickle
import rasterio as rio
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms
import torch
class RescaleQuantile98(object):
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

def get_img(path):
    tmp = rio.open (path).read() # read image as np array
    tmp = RescaleQuantile98()(tmp)*255
    tmp = tmp[1:4,:,:]  # select RGB channels
    tmp = np.moveaxis(tmp, [0, 1, 2], [2,0,1])
    tmp = np.uint8(tmp)
    return tmp

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

klass = list(class2txt.keys()) # list of all classes

# Read all tiff files
os.chdir(data_path)
tif = glob.glob('*.tif')
tif.sort()
N = len (tif) 

plot= 141
fig = plt.figure(figsize=(12, 3))
fig.subplots_adjust(left=0.01, right=0.99)

serie=0
extent=(-3, 4, -4, 3)
#2. loop over the classes to produce a images tiles
#------------------------------------------------------
for ii,k in enumerate( klass):         # loop over all the classes
    lab = class2txt[ k  ]
    lab = lab.replace('_',' ')
    print(ii,k,lab)
    
    interest = []
    sel = []
    
    for i in range(N): # loop over all images 
        if RELI2class[ int( tif[i][:-4]  ) ] == k : # compare the category of the image with the target category
            interest += [tif[i]] # select images from the class k of interest
    
    if  len(interest) < 100 : 
        continue
    sel=random.sample(interest, k=9)
    print(ii,lab, len(interest))
    grid = ImageGrid(fig, plot,  # similar to subplot(141)
                    nrows_ncols=(3, 3),
                    axes_pad=0.05,
                    label_mode=0,
                    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])   

    for idx, ax in enumerate (grid):
        #print(idx,ax)
        im = ax.axis('off')
        if idx==0 :
            ax.set_title(lab, fontsize=8, loc='left', color = "k")
        img= get_img(data_path + sel[idx])
        im = ax.imshow(img, extent=extent, interpolation="nearest")
        im = ax.axis('off')     
        
    plot+=1
    if plot<144 : 
        continue
    elif ii == 44:
        print('-'*25)
        fn = '/home/valerie/Python/landuse/Images/reporting/x3serie10.png'
        plt.savefig(fn,dpi=300,bbox_inches='tight')
        fig = plt.figure(figsize=(4, 3))
        fig.subplots_adjust(left=0.01, right=0.99)

    else : 
        plot=141
        serie+=1
        print('-'*25)
        fn = '/home/valerie/Python/landuse/Images/reporting/x3serie'+str(serie)+'.png'
        plt.savefig(fn,dpi=300,bbox_inches='tight')
        fig = plt.figure(figsize=(12, 3))
        fig.subplots_adjust(left=0.01, right=0.99)

# ------------Paste images together------------
if True:
    from PIL import Image
    #Read the images
    src = '/home/valerie/Python/landuse/Images/reporting/x3serie' 
        
    im1 = Image.open(src+'1.png')
    im2 = Image.open(src+'2.png')
    im3 = Image.open(src+'3.png')
    im4 = Image.open(src+'4.png')
    im5 = Image.open(src+'5.png')
    im6 = Image.open(src+'6.png')
    im7 = Image.open(src+'7.png')
    im8 = Image.open(src+'8.png')
    im9 = Image.open(src+'9.png')
    img10 =Image.open(src+'10.png')

    #resize, first image
    l,h= im1.size
    new_image = Image.new('RGB',(l, 4*h), (255,255,255))
    new_image.paste(im1,(0,0))
    new_image.paste(im2,(0,h))
    new_image.paste(im3,(0,2*h))
    new_image.paste(im4,(0,3*h))
  
    new_image2 = Image.new('RGB',(l, 4*h), (255,255,255))
    new_image2.paste(im5,(0,0))
    new_image2.paste(im6,(0,1*h))
    new_image2.paste(im7,(0,2*h))
    new_image2.paste(im8,(0,3*h))
    
    new_image3 = Image.new('RGB',(l, 2*h), (255,255,255))
    new_image3.paste(im9,(0,0))
    new_image3.paste(img10,(0,h))

    new_image.save(src + "tot.jpg","JPEG")
    new_image2.save(src + "tot2.jpg","JPEG")
    new_image3.save(src+ "tot3.jpg","JPEG")
