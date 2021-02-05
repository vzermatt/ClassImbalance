import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
import  time, sklearn,  pickle



# 1. load FHNW prediction and grund truth
#------------------------------------------
src ='/home/valerie/data/FHNW_predictions.csv'
label_txt_path = '/home/valerie/Python/landuse/Dictionnaries/NOLU_class2txt.p'

df= pd.read_csv(src,sep=',')

all_labels =df['gndtruth'].to_list()    # 35 gt labels
all_preds =df['prediction'].to_list()   # 25 pred labels

# Load dictionnary for matching class id with labels
with open(label_txt_path, 'rb') as fp:
    class2txt = pickle.load(fp)


# CLEAN classes in our model : REMOVED 146,303, 147,108
class_all = [101, 103, 105, 106, 107,  121, 122, 
                123,  162, 163, 201, 202, 221,
                222,  242,  301,  304, 
                401, 402, 421, 423]

# Replace the class grouped :
count242,count222=0,0
for idx, smpl in enumerate(all_labels):
    if smpl == 241 or smpl==243:
        all_labels[idx]=242
        count242 +=1
    elif smpl == 223 :
        all_labels[idx]=222
        count222+=1

for idx, smpl in enumerate(all_preds):
    #print(idx,smpl)
    if smpl == 241 or smpl==243:
        all_preds[idx]=242
    elif smpl == 223 :
        all_preds[idx]=222


# Enumerate classes absent from our dataset
for idx, k in  enumerate(np.unique([all_preds,all_labels])):
    if not(k in class_all):
        print(idx,'\t',k,class2txt[k])

# Select only classes present in our model:
new_labels, new_preds=[],[]

for k in range(len(all_labels)):

    if all_labels[k] in class_all :
        if  all_preds[k] in class_all  :
            new_labels += [all_labels[k]]
            new_preds  += [all_preds[k]]
        else:
            print('Not in dataset : pred',all_preds[k])
    else:
        print('Not in dataset :label ',all_labels[k])
print(len(new_labels),len(new_preds))

all_labels=new_labels
all_preds=new_preds

# extract all classes name in a list
classes =list(np.unique([all_labels,all_preds]))

# compute overall accuracy
running_corrects=0
for k in range(len(all_labels)):    
    if all_labels[k]==all_preds[k]:
        running_corrects +=1

class_all2txt=[]
for x in class_all:
    tmp=class2txt[x]
    tmp =tmp.replace('_',' ')
    class_all2txt+=[tmp]

overall_acc =  running_corrects/len(all_labels)
classes = class_all2txt
# 2. Create confusion matrix
#-----------------------------
cm = confusion_matrix( all_labels, all_preds )
df_cm = DataFrame(cm, index = [i for i in class_all2txt],   
        columns = [i for i in class_all2txt ]
        )

x = plt.figure()
sn.set(font_scale=0.5) #label size
ax= sn.heatmap(df_cm, annot=True,   
            fmt='d', 
            annot_kws={"size": 4}, #font size
            robust =True,
            cmap="YlGnBu",
            cbar =False, 
            square = True
            )
ax.set(title="Heatmap FHNWby ADELE" ,
    xlabel="Predicted Label",
    ylabel="True Label")

fn = '/home/valerie/Python/landuse/Images/confusion_matrix/cm_FHNW_adele_nbCLEAN_2801.png'
plt.savefig(fn,dpi=200,bbox_inches='tight')
print('Confusion matrix saved : ', fn)

# 3. Compute quick statistics with sklearn
#-------------------------------------------
report = sklearn.metrics.classification_report(all_labels,all_preds, target_names=classes,zero_division=0)
print(report)

# 4. Compute average accuracies per class group :rare-common-frequent  :
#-----------------------------------------------------------------------

# define class groups : 
rare = ['Industrial and commercial areas > 1 ha', 'Residential areas (blocks of flats)', 
        'Public buildings and surroundings', 'Agricultural buildings and surroundings', 
        'Unspecified buildings and surroundings', 'Motorways', 'Parking areas',
        'Construction sites', 'Unexploited urban areas', 'Sports facilities', 'Golf courses', 
        'Orchards', 'Arable land in general', 'Alpine meadows in general', 
        'Alpine sheep grazing pastures in general',  'Lumbering areas', 'Damaged forest', 'Lakes', 
        'Rivers streams', 'Alpine sports facilities']
common = ['Residential areas (one and two-family houses)', 'Roads',             
        'Semi-natural grassland in general', 'Farm pastures in general']
freq   = ['Vineyards', 'Alpine pastures in general', 'Forest', 'Unused']


# initilizes statistics :
precision = [0]*len(classes)
recall = [0]*len(classes)
rare_acc = []
common_acc =[]
freq_acc =[]
others_acc=[]



# 5. print results for each class
# ------------------------------
print('no \tprecision recall  class size\t label\n','-'*50)
for k,klass in enumerate(classes):

    if cm.sum(0)[k]==0 :
        precision[k] =0.
    else :
        precision[k] = round (  cm[k,k] / cm.sum(0)[k]  ,2)   # TP / (TP + FP)
    if cm.sum(1)[k]==0 :
        recall[k]=0.
    else:
        recall[k]    = round (  cm[k,k] / cm.sum(1)[k]  ,2)  # TP / (TP + FN)

    # assign each class to a group and give it its precision
    if klass in rare:
        rare_acc+=[precision[k]]
    elif klass in common :
        common_acc+=[precision[k]]
    elif klass in freq :
        freq_acc+=[precision[k]]
    elif klass in others :
        others_acc+=[precision[k]]
        print(classes[k],cm.sum(1)[k])
    else:
        print('WARNING unknow class')
    print( k,'\t', precision[k], '\t',recall[k],'\t',cm.sum(1)[k], '\t', classes[k])
    
# 6. Print global results
#-------------------------
kappa = round( cohen_kappa_score(all_labels,all_preds) ,3)
f1score = round ( sklearn.metrics.f1_score(all_labels,all_preds, average = 'macro'   ), 3) 

print('-'*50,'\nKappa value :\t\t',kappa)
print('Overall accuracy: \t {:.3f} '.format( overall_acc))
print('Average precision:\t', round( np.nanmean(precision),3))
print('Average recall:\t', round( np.nanmean(recall),3))
print('F1-Score:\t\t',f1score)
print('Rare class accuracy:\t', round ( np.nanmean(rare_acc) ,3) )
print('Common class accuracy:\t', round (  np.nanmean(common_acc),3))
print('Frequent class accuracy:', round (  np.nanmean(freq_acc),3))
#print('Others class accuracy:\t', round (  np.nanmean(others_acc),3))

# 7. Printing results to a txt file
#----------------------------------
if True :
    file_name = '/home/valerie/Python/landuse/Images/testing/results_FHNWCLEAN_2801.txt'
    fd = open(file_name,'w')
    fd.write('Testing results for model: ' + src +'\n')
    fd.write( 'FHNW\n')
    fd.write('-'*50+'\nKappa value :\t\t\t'+ str(kappa )+ '\n')
    fd.write('Overall accuracy: \t\t'   + str(round(overall_acc,3)) + '\n')
    fd.write('Average precision:\t\t'   + str(round( np.nanmean(precision),4))+ '\n')
    fd.write('Average recall:\t\t\t'    + str( round( np.nanmean(recall),3)) + '\n')
    fd.write('F1-Score:\t\t\t\t'        + str( f1score)+ '\n')
    fd.write('Rare class accuracy:\t'   + str(round ( np.nanmean(rare_acc) ,3) ) + '\n' )
    fd.write('Common class accuracy:\t' + str(round (  np.nanmean(common_acc),3) )+ '\n')
    fd.write('Frequent class accuracy:' + str(round (  np.nanmean(freq_acc),3))+ '\n')
    fd.write('-'*50 + '\n no \tprecision recall  class size\t label\n' +'-'*50 + '\n')

    for idx in range(k+1) : 
        fd.write(str(idx)+','+ str(precision[idx])+ ','+str(recall[idx])+','
            +str(cm.sum(1)[idx]) + ','+ classes[idx] + '\n') 
    fd.close()




