'''
    Produce the training set for comparison with ADELE
    Read the csv files with ADELE predictions
    Exclude samples points from the test set
    PRoduce the train, validation and test sets for our models
'''

import numpy as np
import pandas as pd
import os, pickle, glob

def print_class_distr(train,val,test):    
    x = pd.read_csv( train, delimiter = ",", header =0).values.tolist()
    y = pd.read_csv( val, delimiter = ",", header =0).values.tolist()
    z = pd.read_csv( test, delimiter = ",", header =0).values.tolist()
    print(train,'-'*30)
    
    class_all = [101, 103, 105, 106, 107, 108, 121, 122, 
                123, 146, 147, 162, 163, 201, 202, 221,
                222, 223, 241, 242, 243, 301, 303, 304, 
                401, 402, 421, 423]
    for classe in class_all : 
        tr_lst,val_lst, test_lst=[],[],[]
        # Find all the image corresponding to a class
        for img_name in x:
            if tileID2class[int(img_name[0][:-4])] ==classe:
                tr_lst+=[img_name]
        #print(len(tr_lst),',')
        for img_name in y:
            if tileID2class[int(img_name[0][:-4])] ==classe:
                val_lst+=[img_name]
        for img_name in z:
            if tileID2class[int(img_name[0][:-4])] ==classe:
                test_lst+=[img_name]
        nb = len(tr_lst) + len(val_lst) + len(test_lst)
        print(classe,'\t',nb,'\t',len(tr_lst),'\t',len(val_lst),'\t',len(test_lst))
    return


# 1. Load Dictionnary and path to data
# --------------------------------------------------------
src ='/home/valerie/data/FHNW_predictions.csv'# path to FHNW predictions 
dest = '/home/valerie/Python/landuse/tile_list/'# path to destination for data list
datapath = '/home/valerie/data/refsurface/'# data path 

# Dictionnary contains pairs: tile id (RELI)-> class label
dico_path = '/home/valerie/Python/landuse/Dictionnaries/'
with open(dico_path+'dict_NOLU46_all.p', 'rb') as fp:
    tileID2class = pickle.load(fp)

# 2. Produce the test set from the FHNW predictions:
#-----------------------------------------------
# Read prediction data from FHNW :
df   = pd.read_csv(src,sep=',')
RELI =df['RELI'].to_list()    #these locations will be used for testing
RELI =list(np.unique(RELI)) # Remove duplicated predictions

# 3. Description of data distribution in FHNW :
#-----------------------------------------------
class_all = np.unique(list(tileID2class.values()))

for classe in class_all : 
    lst=[]
    # Find all the image corresponding to a class
    for img_name in RELI:
        if tileID2class[int(img_name)] ==classe:
            lst+=[img_name]
    print(classe,len(lst))


# 4. list of additional classes in fhnw, absent from my model to be removed :
additional= [102,124,141,144,145,161,203,422]
'''1     102 Industrial_and_commercial_areas_<_1_ha
10       124 Railway_surfaces
11       141 Energy_supply_plants
12       144 Dumps
13       145 Quarries_mines
16       161 Public_parks
21       203 Horticulture
34       422 Avalanche_and_rockfall_protection_structures
--------------------------------------------------------
in total 14 pictures
'''
# produce the list for test set :
test_fn=[]
for img_name in RELI:
    if tileID2class[img_name] in additional : 
        continue # do not keep image in additional classes
    else:
        test_fn+=[str(img_name)+'.tif']

Nreli=len(RELI)     # about 10 % of the total dataset
nb_test= len(test_fn)
print('Nreli,nb_test, Nreli-nb_test:')
Nreli,nb_test, Nreli-nb_test

os.chdir(dest)
if True:
    z = pd.DataFrame(test_fn)
    z.to_csv('fhnw_test.csv',index=False,sep=';',header=False)
print_class_distr(dest+'fhnw_test.csv',dest+'fhnw_test.csv',dest+'fhnw_test.csv')

# 3. Produce the train and validation from all the other available data
#-----------------------------------------------------------------------
#STEPS : 0: list all samples,
#  1: remove data used in test set, about 10% 
#  2: select 10% for validation with stratified sampling
#  3. use all the rest for training


# 0. Read all tiff stored in the data folder
os.chdir(datapath)
raster =glob.glob('*.tif') # the raster file name is <RELI>.tif

# 1. Start by removing images used for testing by FHNW:
# -----------------------  --------------------------------
new_lst=[]
for el in raster:
    if el in test_fn : 
        continue
    else : 
        new_lst+= [el]
# Dictionnary contains pairs: tile id (RELI)-> class label
dico_path = '/home/valerie/Python/landuse/Dictionnaries/'
with open(dico_path+'dict_NOLU46_all.p', 'rb') as fp:
    tileID2class = pickle.load(fp)

print('Original Raster dataset size:',len(raster),' test set size:',len(test_fn),'\nExpected size of bew list:', len(raster)- len(test_fn),'\nactual size of new list:',len(new_lst))

# Create list of  classes
# -----------------------  
class_all = np.unique(list(tileID2class.values()))
print('class total\t train\t validation \t label')
print('-'*85)
train_fn, val_fn, removed =[],[],[]
too_small =  [102, 104, 124, 125, 141, 142, 143, 144, 145, 161, 164, 165, 166, 203, 302, 403, 422, 424] 

# loop over each class 
for classe in class_all : 

    lst = []

    # store in lst all the image corresponding to the current class
    for img_name in new_lst:
        if tileID2class[int(img_name[:-4])] == classe:
            lst+=[img_name]
    nb = len(lst)   # number of samples for the current class

    if classe in additional: # Remove classes im additionnal
        print(classe,'is in additionalm, with ',nb )
        removed +=lst        
        continue
    elif classe in too_small:
        print(classe,'is too small, with ',nb )
        removed +=lst        
        continue
    # define the number of data to select for the current class
    val_nb  = int( 0.1* nb)     #  2: select 10% for validation with stratified sampling
    train_nb = nb - val_nb      #  3. use all the rest for training
      
    print(classe,nb,train_nb,val_nb)

    # Randomize the list order
    lst= np.random.permutation(lst)
    lst= np.random.permutation(lst)
    lst= list(lst)

    #Attribute images to each dataset
    train_fn.extend(lst[:train_nb])
    val_fn.extend(lst[train_nb:])
print('-'*85)

# 4. check that it works properly
# -----------------------  
tot = len(train_fn)+len(val_fn)+len(removed)

if len(new_lst) != (tot): 
    print('missing data',len(train_fn),len(val_fn),tot,len(new_lst))
else : print('ok')

print('\n\nTotal: len train',len(train_fn), '\tlen val :', len(val_fn),'\tlen test set',len(test_fn),'len removed',len(removed))

print('\nIn total in raster', len(raster),'total above :',tot+len(test_fn),'\n\nAll right !')   

# 5 Writing to csv file
# -----------------------  
os.chdir(dest)
train_fn= np.random.permutation(train_fn)
train_fn= list(train_fn)
x = pd.DataFrame(train_fn)
x.to_csv('fhnw_train.csv',index=False,sep=';',header=False)

val_fn= np.random.permutation(val_fn)
val_fn= list(val_fn)
y = pd.DataFrame(val_fn)
y.to_csv('fhnw_val.csv',index=False,sep=';',header=False)

z = pd.DataFrame(test_fn)

print_class_distr(dest+'fhnw_train.csv',dest+'fhnw_val.csv',dest+'fhnw_test.csv')

# Check that dataset do not overlap :
print( set(  x[0]).intersection(y[0])  )
print( set(  z[0]).intersection(y[0])  )
print( set(  x[0]).intersection(z[0])  )

# 6. Produce the train and test set with clean classes
#-----------------------------------------------------

# we start again : clean test set
RELI =list(np.unique(RELI)) # Remove duplicated predictions

# 3. Description of data distribution in FHNW :
#-----------------------------------------------
class_all = np.unique(list(tileID2class.values()))

# NEW CLEAN Dictionnary 
dico_path = '/home/valerie/Python/landuse/Dictionnaries/'
with open(dico_path+'dict_NOLU46_clean.p', 'rb') as fp:
    tileID2class = pickle.load(fp)

# 1. list of 21 classes to keep  :
class_all = [101, 103, 105, 106, 107,  121, 122, 
                123, 162, 163, 201, 202, 221,
                222, 242, 301, 304, 
                401, 402, 421, 423]

# 2.describe data distribution in test set
tot=0
for classe in class_all : 
    lst=[]
    for img_name in RELI:
        if tileID2class[int(img_name)] ==classe:
            lst+=[img_name]
    print(classe,len(lst))
    tot+=len(lst)

# 3. produce the list for test set :
test_fn=[]
removed=0
for img_name in RELI:
    if tileID2class[img_name] in class_all :        
        test_fn+=[str(img_name)+'.tif']
    else :
        removed +=1
print(len(RELI),tot,len(test_fn),removed)   
dest = '/home/valerie/Python/landuse/tile_list/'     
z = pd.DataFrame(test_fn)
z.to_csv(dest+'fhnw_clean_test.csv',index=False,sep=';',header=False)

# 4. produce the train and validation set based on fhnw train and fhnw test
# Dictionnary contains pairs: class id-class label(fulltext)

# USE CLEAN Dictionnary 
dico_path = '/home/valerie/Python/landuse/Dictionnaries/'
with open(dico_path+'dict_NOLU46_clean.p', 'rb') as fp:
    tileID2class = pickle.load(fp)

fhnw_train = str(dest) + 'fhnw_train.csv'
fhnw_val = str(dest) + 'fhnw_val.csv'

def My_aux_function(input_fn,output_fn):

    df =  pd.read_csv( input_fn, delimiter = ",", header =0)  
    df = df.values.tolist()
    outputlist=[]
    removed=0

    # Define the labels that we will ignore in the clean dataset:
    ignore = [146,108,303,147]
    outputlist=[]

    for img_name in df:   # loop over the image to find all those belonging to a class
        if tileID2class[int(img_name[0][:-4])] in ignore:
            removed+=1
            continue
        else:
            outputlist+=[img_name]

    print('*'*50,'\nElement in output file:',len(outputlist),'Element removed:',removed)
    print('Total:',removed+len(outputlist),'\t\t Input size: ',len(df))

    x = pd.DataFrame(outputlist)
    x.to_csv(output_fn,index=False,sep=';',header=False)

My_aux_function(fhnw_train,dest+'fhnw_clean_train.csv')

My_aux_function(fhnw_val,dest+'fhnw_clean_val.csv')
print_class_distr(dest+'fhnw_clean_train.csv',dest+'fhnw_clean_val.csv',dest+'fhnw_clean_test.csv')