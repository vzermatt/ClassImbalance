'''
    Train Resnet 50 model on the dataset
'''

import argparse
import torch
from torch.utils.data import  DataLoader
from torchvision import transforms, models
import time
import copy
import util
import customLoss
import torch.nn.functional as F

# Argument parser
parser = argparse.ArgumentParser('Train ResNet50')
parser.add_argument('--lr', default = 1e-5, type = float )  
parser.add_argument('--ep', default = 100, help = "number of epochs", type =int)
parser.add_argument('--bs', default = 128, help = "size of the batch", type =int)
parser.add_argument('--wd', default = 1e-1, help = " weight decay ", type = float )
parser.add_argument('--step', default = 40, help = " learning rate step size ", type = int )
parser.add_argument('--info', default = 'MODEL_NAME', help = "model name and parameters", type = str )
args = parser.parse_args()

# set random seed
seed = 23452
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

# 1. Load train and validation  set
#------------------------------
# Locate data and labels for training and validation
data_path = '/home/valerie/data/refsurface/'
dico_path = '/home/valerie/Python/landuse/Dictionnaries/dict_NOLU46_clean.p'   # select the  dictionnary !
train_list = '/home/valerie/Python/landuse/tile_list/cleanv2_train.csv'         # clean train!
val_list   ='/home/valerie/Python/landuse/tile_list/cleanv2_val.csv'            # clean val !

# Define the transformations to perform on input images
transforms_data = transforms.Compose([
                    util.MyRandomHorizontalFlip(0.5),
                    util.MyRandomVerticalFlip(0.5),
                    util.MyRandomRotation90(0.5),
                    util.RescaleQuantile98(),              
                    util.ToTensorNP(),
                    util.MyRandomJittering( brightness=0.3, contrast=0.3, 
                                            saturation=0.3, hue=0.05)    # only on RGB channels
                    ])

transforms_data_val = transforms.Compose([
                    util.RescaleQuantile98(),              
                    util.ToTensorNP()
                    ])

# Define dataset and dataloaders
train_set = util.SwisstopoDataset(train_list, dico_path, data_path, transforms_data)
val_set   = util.SwisstopoDataset(val_list,   dico_path, data_path, transforms_data_val)

train_loader = DataLoader(train_set, batch_size= args.bs , shuffle=True, num_workers=1)
val_loader   = DataLoader(val_set,   batch_size= args.bs , shuffle=True, num_workers=1)
# Some info about the dataset :
train_nb = len(train_set)
val_nb   = len(val_set)

samples_per_cls = util.get_samples_per_class(train_list,dico_path)
no_of_classes = len(samples_per_cls)

# 2. Define model
#-----------------------------
# Load pretrained Resnet50 layer
model_ft = models.resnet.resnet50(pretrained=True)

# Add a dropout layer and change output size
num_fc = model_ft.fc.in_features
model_ft.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),            # add dropout with 50% probability
    torch.nn.Linear(num_fc, no_of_classes)       # new output layer
)

# Change the first layer (conv1) to add a 4th and a 5th channel.
weight_conv1 = model_ft.conv1.weight.clone()
model_ft.conv1 = torch.nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)

# weight from Red channel(0) are copied for the new channels ( DEM and NIR)
with torch.no_grad():
    model_ft.conv1.weight[:, 1:4] = weight_conv1        # RGB channels
    model_ft.conv1.weight[:, 0] = weight_conv1[:, 0]    # NIR 
    model_ft.conv1.weight[:, 4] = weight_conv1[:, 0]    # DEM

# TWO PHASES TRAINING PART 
#_______________________________________
#src ='/home/valerie/Python/landuse/TrainedModels/state_clean_sCBL_099_patience30.pth'
#checkpoint = torch.load(src)
#model_dict = checkpoint['model']
#model_ft.load_state_dict(model_dict) # load the model state dictionnary 
#_______________________________________

# Train on GPU
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
print(device)
# 3. Define the loss function  & optimizer
#----------------------------------------------
lossf ='CBL'

if lossf == 'CEL': # Softmax loss
    criterion_ft = torch.nn.CrossEntropyLoss()

elif lossf == 'EQL': # Equalization Loss
    criterion_ft = customLoss.SoftmaxEQL(lambda_ = 1000, ignore_prob =0.9)

elif lossf =='inverse_freq':
    # Soft max loss with inverse or inverse square root  frequency reweighting
    class_weights = customLoss.Compute_frequency_weights(freq='simple', samples_per_cls = samples_per_cls)
    class_weights = class_weights.to(device)
    criterion_ft = torch.nn.CrossEntropyLoss(weight=class_weights)    

elif lossf == 'CBL': # Class Balanced Loss
    # parameters needed for the class balanced loss
    # beta  :  class blanced loss 
    # gamma : focal loss weight
    beta, gamma = 0.99,1
    loss_type = 'softmax'
    class_weights = customLoss.Compute_CB_weights(samples_per_cls, no_of_classes, beta)
    class_weights = class_weights.to(device)

    if loss_type =='softmax':
        criterion_ft = torch.nn.CrossEntropyLoss(weight=class_weights)

    elif loss_type == 'focal':
        criterion_ft = customLoss.CB_loss # adapt the train loop

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=args.lr, eps=1e-08, 
                                betas=(0.9, 0.999), weight_decay= args.wd )

# Decay learning rate by a factor of 0.1 every x epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR( optimizer_ft , step_size=args.step, gamma=1)
t_acc =[]
t_loss =[]
   
filename = '/home/valerie/Python/landuse/TrainedModels/state_'+args.info+'.pth'

# 6. Train the network and save it
# --------------------------------
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    
    print ( "\n\nModel training with following parameters :",
            "\nNumber of epochs:", num_epochs,
            "\nNumber of workers:", train_loader.num_workers,
            "\nBatch size :", train_loader.batch_size,
            "\nInitial learning rate:",args.lr,
            "\nWeight decay:", args.wd,
            '\nLoss function: ', lossf, '\nmodel name:',args.info,
            '\n'
            )
    # Initialization
    best_model_wts = copy.deepcopy(model.state_dict()) # initialization of state_dict
    best_acc = 0.0
    patience = 0  
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}, Patience {}'.format(epoch, num_epochs - 1, patience))
        time_elapsed = time.time() - since
        print('Duration {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
        print('-' * 10)
        if patience >= 30 :#  implementation of early stopping with xx epochs patience 
            break
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in (train_loader if (phase == 'train') else val_loader) :
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()# zero the parameter gradients

                # forward: track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels )
                    #loss = customLoss.CB_loss(labels, outputs, samples_per_cls,  no_of_classes, loss_type,  beta, gamma)   # loss for CBL                

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            dataset_size = (train_nb if phase=='train'else val_nb)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            aux = torch.Tensor.cpu(epoch_acc)
            t_acc.append(aux.tolist())
            t_loss.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model if it is the best and save its weights
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                patience =0 
            elif phase == 'val' and epoch_acc < best_acc : # early stopping
                patience +=1

            # save the model at each epochs    
            if phase == 'val': 
                state = {
                        'model': model.state_dict(),
                        'loss': t_loss,
                        'accuracy': t_acc,
                        'params': args
                        }
                torch.save(state, filename)
        print()

    # Single execution at the end of the training
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    state = {
        'model': model.state_dict(),
        'loss': t_loss,
        'accuracy': t_acc,
        'params': args
        }
    torch.save(state, filename)

    return model, state

# 7. Train and save the model

model_tr, state_tr = train_model(model_ft, criterion_ft, 
                                        optimizer_ft, 
                                        exp_lr_scheduler, 
                                        num_epochs=args.ep)

