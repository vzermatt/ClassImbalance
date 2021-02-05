'''
    Produce the dictionnary that knows the class of an image based on its RELI
    Special dictionnary with fusionned classes for Alpine pasture ( clean dataset)
    and semi-natural grass land
'''
# Load  labels  and create dictionnary
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
import pickle , os 


label_path = "/home/valerie/Documents/QGIS/tile_AOI/NOLU18_46_AOI.csv"
label_df = read_csv(label_path, delimiter = ",", header =0)
id= list(label_df['RELI'])
lab = list( label_df['LU18_46'])
#-------------------------------------------
# Fusion of several classes into one 
#-------------------------------------------

N= len(id) # total numer of samples
# count the number of label 
for k in [241,242,243,222,223]:
    print('count for',k,':',lab.count(k))

# replace lab 
for k in range(N):
    if lab[k]==241 or lab[k]==243:
        lab[k]=242
    elif lab[k]==223:
        lab[k]=222

# count the number of new label 
for k in [241,242,243,222,223]:
    print('count for',k,':',lab.count(k))

# define a dictionnary for each tile id (the RELI number, formed by its coordinates) and its class
dico = {id[k]:lab[k] for k in range(len(id))}

# Save the dictionnary
os.chdir('/home/valerie/Python/landuse/Dictionnaries')
with open('dict_NOLU46_clean.p', 'wb') as fp:
    pickle.dump(dico, fp)

# Load dictionnary
with open('dict_NOLU46_clean.p', 'rb') as fp:
    data = pickle.load(fp)


