import numpy as np
import pandas as pd
import torch, torchvision, os,glob, pickle, time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,models
from PIL import Image
import rasterio as rio
import torchvision.transforms.functional as TF
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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

   
class RescaleQuantile98(object):
    '''
        Rescale image color between 0 and 1 from min and max value of each band
        torch.floatTensor as input and output.
    '''
    def __call__(self, img):
        if img.shape ==(5,200,200):
            out=np.float32(img)
            out[0,:,:] = img[0,:,:]/ ( np.quantile(img[0,:,:],0.98) +1e-3)
            out[1,:,:] = img[1,:,:]/ ( np.quantile(img[1,:,:],0.98) +1e-3)
            out[2,:,:] = img[2,:,:]/ ( np.quantile(img[2,:,:],0.98) +1e-3)
            out[3,:,:] = img[3,:,:]/ ( np.quantile(img[3,:,:],0.98) +1e-3)
            out[4,:,:] = (img[4:,:] -475  )/(3242. -475)
            out[out>1]=1
            out[out<0]=0
            return out
        else :
            print('Rescale failed,  original image returned') 
            return img  
  

class SwisstopoDataset(Dataset):
    '''
        Load the dataset from path into tensor.

        Import raster names from csv files. Raster names are <tileID>.tif  i.e. 12345678.tif
        Import dictionnary matching tileID-class: i.e. 12345678 : 101
        Transform class id with label encoder: 101 : 0
        Apply the different transforms.
        Return the dataset as tensors couple (image-label).
    '''
    def __init__(self, list_path, dico_path, data_path,transforms_data=None):
        ''' 
            Initialization 
        '''       
        self.list_path = list_path  # path to csv file containing the name of training,val and test images
        self.data_path = data_path  # path to the image folder
        self.transforms = transforms_data    # data augmentation transforms

        # Load image list from csv
        df = pd.read_csv(self.list_path)
        self.data_lst = df.values.tolist()
        self.data_lst = [''.join(ele) for ele in self.data_lst] 
        
        # If no data transforms is given, use this one:
        if self.transforms is None:
            means = ( 0.5615, 0.4772, 0.5387, 0.5958, 0.3816)
            stds  = ( 0.2391, 0.2320, 0.2238, 0.2076, 0.1552)
            self.transforms =  transforms.Compose([      
                                    RescaleQuantile98(),       
                                    ToTensorNP(),
                                    transforms.Normalize(means, stds),
                                ])
             
        # load dictionnary : file name - class 
        with open(dico_path, 'rb') as fp:
            self.tileID2class = pickle.load(fp) 
        # all the classes :
        if dico_path == '/home/valerie/Python/landuse/Dictionnaries/dict_NOLU46_clean.p':
            self.classes =  [101, 103, 105, 106, 107, 121, 122,
                             123, 162, 163, 201, 202, 221, 222,
                             242, 301, 304, 401, 402, 421, 423]
        else:
            self.classes = [101, 103, 105, 106, 107, 108, 121, 122,
                            123, 146, 147, 162, 163, 201, 202, 221, 
                            222, 223, 241, 242, 243, 301, 303, 304, 
                            401, 402, 421, 423]
    
    def __getitem__(self, idx):
        ''' 
            Generates one sample of data
        ''' 
        if torch.is_tensor(idx):
            idx = idx.tolist()
                
        # Labels normalized with LabelEncoder 
        le = LabelEncoder()
        le.fit(list( self.classes))
    
        #Produces the data label
        tileID= int ( self.data_lst[idx][:-4] ) # tile id from file name
        # Save class id from the dictionnary, use the label encoder
        classe = le.transform([ self.tileID2class[tileID] ]).item()

        # Load the raster from path as np array with dtype uint16
        tmp = rio.open (self.data_path + self.data_lst[idx] ).read()
        tmp = np.float32(tmp)   #stored as np array with float32 type            
        tensor = self.transforms(tmp)# apply data transforms

        return (tensor,classe)
    
    def __len__(self):
        ''' Denotes the total number of samples'''
        return len(self.data_lst)

class MyRandomHorizontalFlip(object):
    '''
        Random horizontal flip on np array
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self,  img ):        
        if torch.rand(1) < self.p:
            #print('horizontal flip')     
            return img[:,:,::-1]
        return img

class MyRandomVerticalFlip(object):
    '''
        Random vertical flip on np array
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self,  img ):     
        if torch.rand(1) < self.p:
            #print('vertical flip')   
            return img[:,::-1,:]
        return img

class MyRandomRotation90(object):
    '''
        Random rotation by 90 degree clockwise
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self,  img ):        
        if torch.rand(1) < self.p:
           #print('rotation 90degree')    
            return np.rot90(img, k=1,axes=(2,1))
        return img

class MyRandomJittering(object):
    '''
        Random color jittering
        Input & output are tensors

        Brigthness (+ Saturation + contrast) 
            Can be any non negative number. 0 gives a black image, 
            1 gives the original image while 2 increases the 
            brightness by a factor of 2.
        hue_factor is the amount of shift in H channel and 
            must be in the interval [-0.5, 0.5].
        
        For the original image without any transformation, 
        set all arguments to 0.0
    '''
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue

    def __call__(self,  img):
        out =  img
        # we only apply jittering on channel 1,2,3 (RGB)
        img = img [1:4,:,:]              
    
        # choose factor from uniform distribution 
        brightness_factor = np.random.uniform(max(0, 1 -   self.contrast  ), 1 +  self.contrast)
        contrast_factor   = np.random.uniform(max(0, 1 -   self.contrast  ), 1 +  self.contrast)
        saturation_factor = np.random.uniform(max(0, 1 -   self.saturation), 1 +  self.saturation)
        hue_factor        = np.random.uniform(max(-0.5, -  self.hue ), min(0.5,  self.hue) )
        #print('color jit :bright', brightness_factor,'contrast',contrast_factor,'sat',saturation_factor,'hue',hue_factor)

        pil = transforms.ToPILImage()(img) # convert to PILImage
        pil = TF.adjust_brightness(pil, brightness_factor)
        pil = TF.adjust_contrast(pil, contrast_factor)
        pil = TF.adjust_saturation(pil, saturation_factor)
        pil = TF.adjust_hue(pil, hue_factor)
        img = transforms.ToTensor()(pil) # convert back to tensor
        out [1:4,:,:] = img
        return out


def plt_training( loss,accuracy,params,fp):
    '''
        plot training curve for model: accuracy and loss over train and validation datasets
    
    '''
    if  False :
        fp = '/home/valerie/Python/landuse/TrainedModels/state_sCBL_0999.pth'
        checkpoint = torch.load(fp)
        loss = checkpoint['loss']    
        accuracy = checkpoint['accuracy']
        params = checkpoint['params']
    
    t_loss = loss[::2]
    v_loss = loss[1::2]
    t_acc = accuracy[::2]
    v_acc = accuracy[1::2]
    epoch =list (range(len(t_loss))) 
    
    # plot
    fig, (ax1,ax2) = plt.subplots(2,1)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Resnet50 training')
    plt.figtext(0.2,0.9,params)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim((0,1))
    ax1.plot(epoch,t_acc,label='training accuracy')
    ax1.plot(epoch,v_acc, label ='validation accuracy')
    ax1.tick_params(axis='y')
    ax1.legend()

    ax2.set_ylabel('Losses')
    ax2.set_xlabel('Epochs')
    ax2.set_ylim((0,max(v_loss)))
    ax2.plot(epoch,t_loss, label= 'training loss')
    ax2.plot(epoch,v_loss, label = 'validation loss')
    ax2.tick_params(axis='y')
    ax2.legend()
    plt.show()
    fn = ('Images/training_plots/plot_' + fp[49:-4] +'.png')
    os.chdir('/home/valerie/Python/landuse/')
    plt.savefig(fn)
    return

def replace_bn(module, name):
    '''
    Recursively put instance norm instead of batch
    module = model to start code.
    name = model name as a string
    FROM : https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/11
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            #print('replaced: ', name, attr_str)
            new_bn = torch.nn.InstanceNorm2d(target_attr.num_features, affine=False, track_running_stats=False)
            setattr(module, attr_str, new_bn)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():      
        replace_bn(immediate_child_module, name) 

def plot_tensor():
    ''' Checking that data transformation and normalization do not mess the datasets
    '''
    # Prepare for the Swisstopo dataset
    dico_path = '/home/valerie/Python/landuse/Dictionnaries/dict_NOLU46_all.p'
    list_path = '/home/valerie/Python/landuse/tile_list/train.csv'
    data_path = '/home/valerie/data/refsurface/'

    # Define the transformations to perform on input images
    transforms_data = transforms.Compose([
                        MyRandomHorizontalFlip(0.5),
                        MyRandomVerticalFlip(0.5),
                        MyRandomRotation90(0.5),
                        RescaleQuantile98(),                
                        ToTensorNP(),
                        MyRandomJittering( brightness=0.3, contrast=0.3,   
                                            saturation=0.3, hue=0.05)    # only on RGB channels
                        ])
    idx = np.random.randint(10000)             
    # plot 9 random RGB images  from 5D tensor (5,200,200):  
    for k in range(1,10):
          
        img,_ = SwisstopoDataset(list_path, dico_path, data_path,transforms_data).__getitem__(idx)
        tmp = img.numpy()
        tmp = tmp[(1,2,3),:,:] # Select RGB channels
        tmp = tmp*255
        tmp = np.moveaxis(tmp, [0, 1, 2], [2,0,1])
        tmp = np.uint8(tmp)
        x = plt.subplot(3,3,k)
        x = plt.imshow(tmp)
        x = plt.axis('off')
    
    #Save image
    fn = ('/home/valerie/Python/landuse/Images/reporting/I.png')
    plt.savefig(fn)

    return

def Compute_mean_std_on_Tensor()  :
    '''
        Compute mean and standard deviation for a sample of the dataset
        Mean for each band [0.5614521  0.47717252 0.5386521  0.5957687  0.38164657] 
        standard deviation  [0.23908012 0.23200981 0.22376862 0.20764491 0.1551916 ] 
    '''
    # Prepare for the Swisstopo dataset
    dico_path = '/home/valerie/Python/landuse/Dictionnaries/dict_NOLU46_all.p'
    list_path = '/home/valerie/Python/landuse/tile_list/val.csv'
    data_path = '/home/valerie/data/refsurface/'

    #The following parameters for normlization give means =0, std =1:
    means = ( 0.5615, 0.4772, 0.5387, 0.5958, 0.3816)
    stds  = ( 0.2391, 0.2320, 0.2238, 0.2076, 0.1552)

    # Load dataset, data transforms and dataloader
    transforms_data = transforms.Compose([ 
                        RescaleQuantile98(),       
                        ToTensorNP(),
                        transforms.Normalize(means, stds)
                        ])
    train_set = SwisstopoDataset(list_path, dico_path, data_path,transforms_data)
    loader = DataLoader(train_set, batch_size=300)
    
    #Initialize parameters
    nimages = 0
    mean,std,quantile02, quantile99, quantile50 = 0.,0.,0.,0.,0.
    maxi = torch.zeros(5)
    mini = torch.ones([5])*5000
    for batch, _ in loader:
        print(round(nimages/len(train_set)*100,2),'%')
        
        nimages += batch.size(0)  # Update total number of images      
        
        batch =np.swapaxes(batch,1,0) # Rearrange batch to be the shape of [ C, B, W , H]
        # Rearrange batch to be the shape of [ C, B* W * H]      
        batch = batch.reshape(batch.size(0), batch.size(1)* batch.size(2)* batch.size(3))
             
        # Compute mean and std here
        mean += batch.mean(1)   
        std += batch.std(1)

        # Find min and max in each band
        maxi = torch.cat((maxi, batch.max(1).values),0)
        mini = torch.cat((mini, batch.min(1).values),0)
        quantile02 += np.quantile(batch,0.01,axis=1)
        quantile99 += np.quantile(batch,0.99,axis=1)
        quantile50 += np.quantile(batch,0.50,axis=1)

    maxi = torch.reshape(maxi,(len(loader)+1,5))
    max_p_col = maxi.max(0).values
    
    mini = torch.reshape(mini,(len(loader)+1,5))
    mini_p_col = mini.min(0).values

    # Final step
    mean /= len(loader)
    std /= len(loader)
    quantile99/=len(loader)
    quantile02/=len(loader)
    quantile50/=len(loader)
    
    print('\n Mean for each band',np.array(mean),'\n')
    print('\n Standard deviation ',np.array(std),'\n')
    print('\n Max values :',np.array(max_p_col),'\n')
    print('\n Min values :',np.array(mini_p_col),'\n')
    print('\n Quantile 99% :',np.array(quantile99),'\n')
    print('\n Quantile 1% :',np.array(quantile02),'\n')
    print('\n Mediane :',np.array(quantile50),'\n')

    return

def MyResnet50():
    '''
        Load resnet 50 with adaptations for Swisstopo dataset
    '''
    # Load pretrained Resnet50 layer
    model_ft = models.resnet.resnet50(pretrained=True)

    # Add a dropout layer and change output size
    num_fc = model_ft.fc.in_features
    model_ft.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),            # add dropout with 20% probability
        torch.nn.Linear(num_fc, 28)       # new output layer, mapping to 28 classes
    )

    # Change the first layer (conv1) to add a 4th and a 5th channel.
    weight_conv1 = model_ft.conv1.weight.clone()
    model_ft.conv1 = torch.nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # weight from Red channel(0) are copied for the new channels ( DEM and NIR)
    with torch.no_grad():
        model_ft.conv1.weight[:, 1:4] = weight_conv1        # RGB channels
        model_ft.conv1.weight[:, 0] = weight_conv1[:, 0]    # NIR 
        model_ft.conv1.weight[:, 4] = weight_conv1[:, 0]    # DEM

    # Replace the batch normalisation layers with instance normalisation
    # replace_bn(model_ft, 'model_ft') : better without

    return model_ft

def get_samples_per_class(set_list,dico_path):
    '''
        Return an array with the number of samples per class
        present in list (path to csv file) with the labels from the
        dictionnary (dico_path)
    '''    
    if False :
        set_list = '/home/valerie/Python/landuse/tile_list/train.csv'
        dico_path ='/home/valerie/Python/landuse/Dictionnaries/dict_NOLU46_all.p'

    # Read dictionnary tile id to labels :
    with open(dico_path , 'rb') as fp:
        tileID2class = pickle.load(fp)
    # read data from file : get tile id
    df =  pd.read_csv( set_list, delimiter = ",", header =0)
    df = df.values.tolist()
    labels,samples_per_cls = [],[]
    # get all labels in dataset
    for img_name in df:
        labels+=[tileID2class[int(img_name[0][:-4])]]
    # extract all classes present in dataset :
    classes_all=np.unique(labels)
    print('The dataset contains',len(classes_all),'classes and',len(df),'images in total.')

    for classe in classes_all : # loop over each class
        lst=[]
        for img in df: # loop over each image
            if tileID2class[int(img[0][:-4])] ==classe:
                lst+=[img_name] 
        samples_per_cls+=[len(lst)]
        #print('class:',classe,'with',len(lst),'samples')    

    return samples_per_cls