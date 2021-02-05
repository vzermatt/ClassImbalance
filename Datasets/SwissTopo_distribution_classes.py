''''
produce txt file contains the files names for the train, validation and test set

'''

import numpy as np
import pandas as pd
import os, pickle, glob
os.chdir('/home/valerie/Python/landuse/')

src  = '/home/valerie/data/refsurface/'
dest = '/home/valerie/Python/landuse/tile_list/'
dico_path = '/home/valerie/Python/landuse/Dictionnaries/'

# 1. Load Dictionnary and path to data
# --------------------------------------------------------
# Dictionnary contains pairs: tile id- class id
os.chdir(dico_path)
with open('dict_NOLU46_all.p', 'rb') as fp:
    tileID2class = pickle.load(fp)

# Dictionnary contains pairs: class id-class label(fulltext)
with open('NOLU_class2txt.p', 'rb') as fp:
    class2txt = pickle.load(fp)

# Read all tiff stored in the dataset
os.chdir(src)
raster =glob.glob('*.tif') # the raster file name is <tileID>.tif
N = len(raster)

# 2. Datasets with stratified random sampling 
#-------------------------------------------
# 60% for training
# 30% for testing
# 10% for validation

# Create list of  classes
class_all = np.unique(list(tileID2class.values()))
print('class total\t train\t validation test\t\t label')
print('-'*85)
removed=[]
removed_nb =0
train_fn=[]
val_fn=[]
test_fn=[]
lstc =[]
classc=[]

for classe in class_all : 
    lst = []

    # Find all the image corresponding to a class
    for img_name in raster:
        if tileID2class[int(img_name[:-4])] ==classe:
            lst+=[img_name]

    nb = len(lst)
    

    if nb <100:  # Ignore class with less than 100 samples :
        removed+=[classe]
        removed_nb +=nb
        continue # no sampling, skip the rest of the loop
    print(classe,class2txt[classe],nb)
    # define the number of data to select for the current class
    train_nb= int(0.6* nb)
    val_nb  = int( 0.1* nb)
    test_nb = nb - train_nb - val_nb
      
    print(classe,train_nb)
    classc += [classe]
    lstc+= [train_nb]  
    # Randomize the list order
    lst= np.random.permutation(lst)
    lst= np.random.permutation(lst)
    lst= list(lst)

    #Attribute files to each dataset
    train_fn.extend(lst[0:train_nb])
    val_fn.extend(lst[train_nb:(train_nb+val_nb)])
    test_fn.extend(lst[(train_nb+val_nb):nb])

    #print(classe,'\t',nb,'\t',train_nb,'\t',val_nb,'\t',test_nb,'\t',class2txt[classe])
    # Matching totals
    if (train_nb+val_nb + test_nb ) != nb : print('problem with',classe,'class: uncomplete sampling')

print('-'*85)
# check for consistency 
tot = len(train_fn)+len(val_fn) +len(test_fn)
if N != (tot+removed_nb): print('missing data')

print('Total:\t',tot,'\t',len(train_fn),'\t',len(val_fn),'\t',len(test_fn))
print('\nThe following',len(removed),'rare classes have been removed(< 100 samples):\n', 
        removed,'\nIn total', removed_nb,'images removed over',N,'.\n\n')


''''        

# 3. Writing to csv file
# -----------------------  
os.chdir(dest)
x = pd.DataFrame(train_fn)
x.to_csv('train.csv',index=False,sep=';',header=False)
y = pd.DataFrame(val_fn)
y.to_csv('val.csv',index=False,sep=';',header=False)
z = pd.DataFrame(test_fn)
z.to_csv('test.csv',index=False,sep=';',header=False)

# Check that dataset do not overlap :
print( set(  x[0]).intersection(y[0])  )
print( set(  z[0]).intersection(y[0])  )
print( set(  x[0]).intersection(z[0])  )

'''

# 4. Produce a balanced dataset with 100 image for each class
# # ---------------------------------------------------------
# 60 for training
# 10 for validation
def Produce_samples_list(input_fn, max_samples_per_cls, output_fn):
    path = str(dest) + input_fn +'.csv'
    df =  pd.read_csv( path, delimiter = ",", header =0)
    df = df.values.tolist()
    print('\n Number of samples in input file:',len(df),'\n','*'*50)
    outputlist=[]
    removed=0
    # Find all the image corresponding to a class
    for classe in class_all : 
        lst = []
        nb=0
        for img_name in df:   # loop over the image to find all those belonging to a class
            if tileID2class[int(img_name[0][:-4])] ==classe:
                lst+=[img_name]
        nb = len(lst)
        if nb ==0:
           continue 
        elif nb> max_samples_per_cls:
           list_size = max_samples_per_cls
        else :
            list_size = nb
        print(list_size,'\t', nb,'\t',classe,class2txt[classe])
       # Randomize the list order
        lst= np.random.permutation(lst)
        lst= np.random.permutation(lst)
        lst= list(lst)
        outputlist.extend(lst[0: list_size])
        removed += nb - list_size
    print('*'*50,'\nElement in output file:',len(outputlist),'Element removed:',removed)
    print('Total:',removed+len(outputlist),'\t\t Input size: ',len(df))
    x = pd.DataFrame(outputlist)
    x.to_csv(output_fn,index=False,sep=';',header=False)
    return outputlist


def print_class_distr(train,val,test):    
    x = pd.read_csv( train, delimiter = ",", header =0).values.tolist()
    y = pd.read_csv( val, delimiter = ",", header =0).values.tolist()
    z = pd.read_csv( test, delimiter = ",", header =0).values.tolist()
    print(train,'-'*30)
    
    class_all = [101, 103, 105, 106, 107, 108, 121, 122, 
                123, 146, 147, 162, 163, 201, 202, 221,
                222, 223, 241, 242, 243, 301, 303, 304, 
                401, 402, 421, 423]
    print('classe\t nb \t len(tr_lst) \t len(val_lst) \t len(test_lst)')
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
        print(classe,'\t',nb,'\t',len(tr_lst),'\t',len(val_lst),'\t',len(test_lst),class2txt[classe])
    return

dest = '/home/valerie/Python/landuse/tile_list/'
dico_path = '/home/valerie/Python/landuse/Dictionnaries/'
os.chdir(dico_path)
# Dictionnary contains pairs: tile id- class id
with open('dict_NOLU46_all.p', 'rb') as fp:
    tileID2class = pickle.load(fp)

# Dictionnary contains pairs: class id-class label(fulltext)
with open('NOLU_class2txt.p', 'rb') as fp:
    class2txt = pickle.load(fp)

# Dictionnary contains pairs: class id-class label(fulltext)
dest = '/home/valerie/Python/landuse/tile_list/'
class_all = np.unique(list(tileID2class.values()))
no_of_class = len(class_all)

_ = Produce_samples_list('train', 60, str(dest +'100_train.csv'))
_ = Produce_samples_list('train', 600, str(dest +'1000_train.csv'))
_ = Produce_samples_list('val', 10, str(dest +'100_val.csv'))
_ = Produce_samples_list('val', 100, str(dest +'1000_val.csv'))

# Use clean dictionnary now 
with open('dict_NOLU46_clean.p', 'rb') as fp:
    tileID2class = pickle.load(fp)

_ = Produce_samples_list('cleanv2_train', 60, str(dest +'cleanv2_100_train.csv'))
_ = Produce_samples_list('cleanv2_train', 600, str(dest +'cleanv2_1000_train.csv'))
_ = Produce_samples_list('cleanv2_val', 10, str(dest +'cleanv2_100_val.csv'))
_ = Produce_samples_list('cleanv2_val', 100, str(dest +'cleanv2_1000_val.csv'))


# Verfiy what written :
# with normal dictionnary :  
dico_path = '/home/valerie/Python/landuse/Dictionnaries/'
os.chdir(dico_path)

with open('dict_NOLU46_all.p', 'rb') as fp:
    tileID2class = pickle.load(fp)

    
# Print class description
print_class_distr(dest +'train.csv',dest +'val.csv',dest +'test.csv')
print_class_distr(dest +'100_train.csv',dest +'100_val.csv',dest +'test.csv')
print_class_distr(dest +'1000_train.csv',dest +'1000_val.csv',dest +'test.csv')

# use clean dictionnary
os.chdir(dico_path)
with open('dict_NOLU46_clean.p', 'rb') as fp:
    tileID2class = pickle.load(fp)

print_class_distr(dest +'cleanv2_train.csv',dest +'cleanv2_val.csv',dest +'cleanv2_test.csv')
print_class_distr(dest +'cleanv2_100_train.csv',dest +'cleanv2_100_val.csv',dest +'cleanv2_test.csv')
print_class_distr(dest +'cleanv2_1000_train.csv',dest +'cleanv2_1000_val.csv',dest +'cleanv2_test.csv')


# 10. Define rare - common - frequent classes 
#-------------------------------------------

# Create list of  classes
class_all = np.unique(list(tileID2class.values()))
print('class total\t train\t validation test\t\t label')
print('-'*85)
removed=[]
removed_nb =0
rare =[]
common=[]
freq=[]
rare_nb =0
common_nb=0
freq_nb=0

for classe in class_all : 
    lst = []
    # Find all the image corresponding to a class
    for img_name in raster:
        if tileID2class[int(img_name[:-4])] ==classe:
            lst+=[img_name]
   
    nb = len(lst)
    if nb <100:  # Ignore class with less than 100 samples :
        removed+=[classe]
        removed_nb +=nb
        #continue # no sampling, skip the rest of the loop
    elif nb <1000:
        print(class2txt[classe],nb)
        rare += [classe]
        rare_nb +=nb
        #continue
    elif nb <3000:
        print(class2txt[classe],nb)
        common+= [classe]
        common_nb +=nb
        #continue
    else :
        print(class2txt[classe],nb)
        freq +=[classe]
        freq_nb +=nb


rare_txt = []
common_txt=[]
freq_txt = []
for k in rare :
    rare_txt+=[class2txt[k]]
for k in common :
    common_txt+=[class2txt[k]]
for k in freq :
    freq_txt+=[class2txt[k]]
    
