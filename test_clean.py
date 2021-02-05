'''
    Evaluates the model performance after the training
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
import torch, time, torchvision, sklearn, util, pickle
from torchvision import transforms, models


# 1. load test set
#------------------
# Locate data and labels for training and validation 
src ='/home/valerie/Python/landuse/TrainedModels/state_cleanv2_focal.pth'
checkpoint = torch.load(src)
model_dict = checkpoint['model']
params = checkpoint['params']
data_path = '/home/valerie/data/refsurface/'
dico_path = '/home/valerie/Python/landuse/Dictionnaries/dict_NOLU46_clean.p' #clean dictionnary
test_list = '/home/valerie/Python/landuse/tile_list/cleanv2_test.csv' # cleanv2 test set
label_txt_path = '/home/valerie/Python/landuse/Dictionnaries/NOLU_class2txt.p'

# Define the transformations to perform on input images
transforms_data = transforms.Compose([
                    util.RescaleQuantile98(),              
                    util.ToTensorNP(),
                    ])

# Define dataset and dataloaders
test_set = util.SwisstopoDataset(test_list, dico_path, data_path, transforms_data)
test_nb = len(test_set)
test_loader = torch.utils.data.DataLoader(test_set, batch_size= 1 , shuffle=True, num_workers=1)

# Load labels names from the label id to text dictionnary
with open(label_txt_path, 'rb') as fp:
    class2txt = pickle.load(fp)
            
# CLEAN Labels normalized with LabelEncoder 
# We removed : 146,147, 108, 303 (unsuitable class), 241,243 (fusion with 242),223(fusion with 222)
class21 = classes =  [101, 103, 105, 106, 107, 121, 122,
                     123, 162, 163, 201, 202, 221, 222,
                     242, 301, 304, 401, 402, 421, 423]

le = LabelEncoder()
le.fit(list( class21))

# 2. Load trained Model
#-----------------------
samples_per_cls = util.get_samples_per_class(test_list,dico_path)
no_of_classes = len(samples_per_cls)
#no_of_classes = 28

# 2. Define model
#-----------------------------
# Load pretrained Resnet50 layer
model_test = models.resnet.resnet50(pretrained=True)
# Add a dropout layer and change output size
num_fc = model_test.fc.in_features
model_test.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),            # add dropout with 20% probability
    torch.nn.Linear(num_fc, no_of_classes)       # new output layer
)      
# Change the first layer (conv1) to add a 4th and a 5th channel.
weight_conv1 = model_test.conv1.weight.clone()
model_test.conv1 = torch.nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)

# weight from Red channel(0) are copied for the new channels ( DEM and NIR)
with torch.no_grad():
    model_test.conv1.weight[:, 1:4] = weight_conv1        # RGB channels
    model_test.conv1.weight[:, 0] = weight_conv1[:, 0]    # NIR 
    model_test.conv1.weight[:, 4] = weight_conv1[:, 0]    # DEM

model_test.load_state_dict(model_dict) # load the model state dictionnary 
model_test.eval()
# send model to GPU 
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"
model_test = model_test.to(device)

# Running model on test set
print ( "\n Model testing\n",'-'*15,'\n',params)
running_corrects = 0
all_preds =[]
all_labels =[]
model_test.eval()

# 3. Test the model over the test dataset
#----------------------------------------
# Iterate over data.
idx =0
for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model_test(inputs)
    _,preds = torch.max(outputs, 1)
    all_preds.append(torch.Tensor.item(preds))
    all_labels.append(torch.Tensor.item(labels))
    running_corrects += torch.sum(preds == labels.data)
    idx+=1
    if idx%500 ==0:
        print(round(idx/len(test_set)*100,0),'%')

# Evaluating results
epoch_acc = running_corrects.double() / test_nb
print('Test Phase Overall Accuracy: {:.4f}'.format( epoch_acc))


all_labels = le.inverse_transform(all_labels)
all_preds = le.inverse_transform(all_preds)
classes =list(np.unique(all_labels))
cm_label=[]
for cls in classes:
    tmp = class2txt[cls]
    cm_label+=[tmp.replace('_', ' ')]

# from label id to label text
all_txtlabels =[]
all_txtpreds =[]

for el in list( le.inverse_transform(all_labels)):
    all_txtlabels.append(class2txt[el] )
for el in list( le.inverse_transform(all_preds)):
    all_txtpreds.append(class2txt[el] )

# extract all classes name in a list
classes =list(np.unique(all_labels))

# 3. Create confusion matrix
#-----------------------------
if False:
    fp = '/home/valerie/Python/landuse/Images/testing/predictions_cleanv2_baseline21.txt'
    src=fp
    x= pd.read_csv(fp)
    all_txtlabels = list(x['labels'] )
    all_txtpreds  = list(x['predictions'])
    classes =list(np.unique(all_labels))

# produce nice label without '_'
klasses =[]
for el in classes: 
    klasses +=[el.replace('_', ' ')]
#compute CM
cm = confusion_matrix( all_labels, all_preds )
df_cm = DataFrame(cm, index = [i for i in cm_label],   
        columns = [i for i in cm_label       ] )
#paramters for CM plot
plt.close
sn.set(font_scale=0.4) #label size
ax= sn.heatmap(df_cm, annot=True,   
            fmt="d", #.1f
            annot_kws={"size": 4}, #font size
            robust =True,
            cmap="YlGnBu",
            cbar =False, 
            square = True
            )
ax.set(title="Confusion matrix for the clean Baseline model" ,
    xlabel="Predicted Label",
    ylabel="True Label",)

#fn = '/home/valerie/Python/landuse/Images/confusion_matrix/cm_BASELINE_accl.png'
fn = '/home/valerie/Python/landuse/Images/confusion_matrix/cm_New_clean_baseline21.png'
plt.savefig(fn,dpi=200,bbox_inches='tight')
print('Confusion matrix saved : ', fn)

# Compute quick statistics with sklearn
report = sklearn.metrics.classification_report(all_txtlabels,all_txtpreds, target_names=classes,zero_division=0)
print(report)

# 5. Compute average accuracies per class group :rare-common-frequent  :
#------------------------------------------------------------------
rare = ['Industrial_and_commercial_areas_>_1_ha', 'Residential_areas_(blocks_of_flats)', 
        'Public_buildings_and_surroundings', 'Agricultural_buildings_and_surroundings', 
        'Unspecified_buildings_and_surroundings', 'Motorways', 'Parking_areas', 'Construction_sites', 
        'Unexploited_urban_areas', 'Sports_facilities', 'Golf_courses', 'Orchards', 
        'Arable_land_in_general', 'Alpine_meadows_in_general', 'Alpine_sheep_grazing_pastures_in_general', 
        'Lumbering_areas', 'Damaged_forest', 'Lakes', 'Rivers_streams', 'Alpine_sports_facilities']
common = ['Residential_areas_(one_and_two-family_houses)', 'Roads', 'Semi-natural_grassland_in_general', 'Farm_pastures_in_general']
freq = ['Vineyards', 'Alpine_pastures_in_general', 'Forest', 'Unused']

precision = [0]*len(classes)
recall = [0]*len(classes)
rare_acc = []
common_acc =[]
freq_acc =[]

print('no \tprecision recall  class size\t label\n','-'*50)
for k,klass in enumerate(classes):

    if cm.sum(0)[k]==0 :
        precision[k] =0.
    else :
        precision[k] = round (  cm[k,k] / cm.sum(0)[k]  ,2)   # TP / (TP + FP)
    recall[k]    = round (  cm[k,k] / cm.sum(1)[k]  ,2)  # TP / (TP + FN)

    # assign each class to a group and give it its precision
    if klass in rare:
        rare_acc+=[precision[k]]
    elif klass in common :
        common_acc+=[precision[k]]
    elif klass in freq :
        freq_acc+=[precision[k]]
    print( k,'\t', precision[k], '\t',recall[k],'\t',cm.sum(1)[k], '\t', classes[k])
    

kappa = round( cohen_kappa_score(all_labels,all_preds) ,3)
f1score = round ( sklearn.metrics.f1_score(all_txtlabels,all_txtpreds, average = 'macro'   ), 3) 

print('-'*50,'\nKappa value :\t\t',kappa)
print('Overall accuracy: \t {:.3f} '.format( epoch_acc))
print('Average precision:\t', round( np.nanmean(precision),3))
print('Average recall:\t', round( np.nanmean(recall),3))
print('F1-Score:\t\t',f1score)
print('Rare class accuracy:\t', round ( np.nanmean(rare_acc) ,3) )
print('Common class accuracy:\t', round (  np.nanmean(common_acc),3))
print('Frequent class accuracy:', round (  np.nanmean(freq_acc),3))

# Printing results to a txt file
#----------------------------------
file_name = '/home/valerie/Python/landuse/Images/testing/results_'+ src[49:-4]+'.txt'
fd = open(file_name,'w')
fd.write('Testing results for model: ' + src +'\n')
fd.write( str(params) + '\n')
fd.write('-'*50+'\nKappa value :\t\t\t'+ str(kappa )+ '\n')
fd.write('Overall accuracy: \t\t'   + str(round(epoch_acc.item(),3)) + '\n')
fd.write('Average precision:\t\t'   + str(round( np.nanmean(precision),4))+ '\n')
fd.write('Average recall:\t\t\t'    + str( round( np.nanmean(recall),3)) + '\n')
fd.write('F1-Score:\t\t\t\t'        + str( f1score)+ '\n')
fd.write('Rare class accuracy:\t'   + str(round ( np.nanmean(rare_acc) ,3) ) + '\n' )
fd.write('Common class accuracy:\t' + str(round (  np.nanmean(common_acc),3) )+ '\n')
fd.write('Frequent class accuracy:' + str(round (  np.nanmean(freq_acc),3))+ '\n')
fd.write('-'*50 + '\n no \tprecision recall  class size\t label\n' +'-'*50 + '\n')

for idx in range(k+1) : 
    fd.write(str(idx)+','+ str(precision[idx])+ ','+str(recall[idx])+','+str(cm.sum(1)[idx]) + ','+ classes[idx] + '\n') 
fd.close()

#Printing all test label and predictions to txt files
#----------------------------------------------------
file_name = '/home/valerie/Python/landuse/Images/testing/predictions_'+ src[49:-4]+'.txt'
fd = open(file_name,'w')
fd.write('labels,predictions\n') 
for idx in range(len(all_labels)) : 
    fd.write(str(all_txtlabels[idx])+','+ str(all_txtpreds[idx])+'\n') 
fd.close()
