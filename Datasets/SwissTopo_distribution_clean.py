'''
    produce a train test and validation set with clean labels
    = removes 4 classes from the train.txt val.txt text.txt lists
    Also read txt file and produce class statistics ( number of samples per class for each ltxt file)
'''


import numpy as np
import pandas as pd
import pickle

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

# load existing train-test-val dataset:
filep = '/home/valerie/Python/landuse/tile_list/'
sets = [ 'train','val','test'] # choose the data list  
dico_path = '/home/valerie/Python/landuse/Dictionnaries/dict_NOLU46_all.p'

# Define the labels that we will ignore in the clean dataset:
ignore = [146,108,303,147]
#ignore = [146,108,303,147,223,241,243]

# Load dictionnary with pairs tile ID- tile class
with open(dico_path , 'rb') as fp:
    tileID2class = pickle.load(fp)

if False: # write new clean dataset

    for label_path in sets:
        path = str(filep) + label_path +'.csv'
        df =  pd.read_csv( path, delimiter = ",", header =0)
        df = df.values.tolist()
        #print(len(df))
        sel, removed = [],[] 
        
        # Find all the image corresponding to the class to remove class
        for img_name in df:

            if tileID2class[int(img_name[0][:-4])] in ignore:
                removed+=[img_name]
                #print(tileID2class[int(img_name[0][:-4])])
            else:
                sel+=[img_name]
        # Check that all img have been correctly treated :
        print('\n\nFor the labels in ',label_path[-10:],'We had initially ',len(df),'samples\n, We removed ',len(removed), 'samples from the following confusing classes',ignore,'\nWe have ',len(sel),'sample left in the set.\n check sum :',len(removed)+len(sel),'-'*50)
        if True: # write the new dataset
            new_fn = str(filep)+'cleanv2_'+ str(label_path) +'.csv'
            x = pd.DataFrame(sel)
            x.to_csv(new_fn,index=False,sep=';',header=False)


dico_path = '/home/valerie/Python/landuse/Dictionnaries/dict_NOLU46_clean.p'
# Load dictionnary with pairs tile ID- tile class
with open(dico_path , 'rb') as fp:
    tileID2class = pickle.load(fp)

sets = [ 'train','val','test','100_train','100_val','test'] # choose the data list  
# now read dataset & classes :
xp = str(filep)+'cleanv2_'+ str(sets[0]) +'.csv'
yp = str(filep)+'cleanv2_'+ str(sets[1]) +'.csv'
zp = str(filep)+'cleanv2_'+ str(sets[2]) +'.csv'
print_class_distr(xp,yp,zp)
xp = str(filep)+'cleanv2_'+ str(sets[3]) +'.csv'
yp = str(filep)+'cleanv2_'+ str(sets[4]) +'.csv'
zp = str(filep)+'cleanv2_'+ str(sets[5]) +'.csv'
print_class_distr(xp,yp,zp)


dico_path = '/home/valerie/Python/landuse/Dictionnaries/dict_NOLU46_all.p'
with open(dico_path , 'rb') as fp:
    tileID2class = pickle.load(fp)

xp = str(filep)+ str(sets[0]) +'.csv'
yp = str(filep)+ str(sets[1]) +'.csv'
zp = str(filep)+ str(sets[2]) +'.csv'
print_class_distr(xp,yp,zp)
xp = str(filep)+ str(sets[3]) +'.csv'
yp = str(filep)+ str(sets[4]) +'.csv'
zp = str(filep)+ str(sets[5]) +'.csv'
print_class_distr(xp,yp,zp)

# now read for FHNW dataset & classes :
xp = str(filep)+'fhnw_train.csv'
yp = str(filep)+'fhnw_val.csv'
zp = str(filep)+'fhnw_test.csv'
print_class_distr(xp,yp,zp)


# now read for clean FHNW dataset & classes :
xp = str(filep)+'fhnw_clean_train.csv'
yp = str(filep)+'fhnw_clean_val.csv'
zp = str(filep)+'fhnw_clean_test.csv'
print_class_distr(xp,yp,zp)