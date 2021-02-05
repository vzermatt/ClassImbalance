import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# function to compare results from two classifiers
# -------------------------------------------------
def Read_result_from_txt(fn):
    '''
    Read classification results for each class from testing report
    '''
    first_record = False
    idx,precision,recall,class_size,label = [],[],[],[],[]

    fd = open(fn, 'r')
    for ligne in fd:
        if ligne.rstrip()[0] == '0':
            first_record = True
        if first_record == True : 
            ligne = ligne.replace('\n','',1)
            #ligne = ligne.replace('_',' ')
            a,b,c,d,e =ligne.split(',')
            idx+=[int(a)]
            precision +=[float(b)]
            recall +=[float(c)]
            class_size+=[int(d)]
            label +=[e]
    fd.close()
    return idx,precision,recall,class_size,label

def difference( current):
    difference = np.array(current) - np.array(bl_precision_r)
    difference = list( np.round(difference,2))
    return difference

def select_group(liste, group='common'):
    rare = ['Industrial_and_commercial_areas_>_1_ha', 'Residential_areas_(blocks_of_flats)',      'Public_buildings_and_surroundings', 'Agricultural_buildings_and_surroundings',         'Unspecified_buildings_and_surroundings', 'Motorways', 'Parking_areas',         'Construction_sites', 'Unexploited_urban_areas', 'Sports_facilities', 'Golf_courses', 'Orchards', 'Arable_land_in_general', 'Alpine_meadows_in_general', 'Alpine_sheep_grazing_pastures_in_general', 'Lumbering_areas', 'Damaged_forest', 'Lakes', 'Rivers_streams', 'Alpine_sports_facilities']
    common = ['Residential_areas_(one_and_two-family_houses)', 'Roads','Semi-natural_grassland_in_general', 'Farm_pastures_in_general']
    freq = ['Vineyards', 'Alpine_pastures_in_general', 'Forest', 'Unused']

    if group=='rare' :
        groupe =rare
    elif group=='common':
        groupe=common 
    else :
        groupe=freq
    select_precision=[]
    select_labels = groupe

    for idx,el in enumerate(liste):
        #print(el,idx,labels[idx])
        if labels[idx] in groupe:
            #print(labels[idx],'is in groupe',group)
            select_precision += [el]

    return select_precision, select_labels

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 3 points vertical offset
                    textcoords="offset points",
                    fontsize = 'xx-small',
                    ha='center', va='bottom')


#Give paths to report files : 
baseline = '/home/valerie/Python/landuse/Images/testing/results_13Dec2020_20h20.txt'  
f0 = '/home/valerie/Python/landuse/Images/testing/results_focalCBL_099_2.txt'
f1 = '/home/valerie/Python/landuse/Images/testing/results_16Dec2020_17h55.txt'
f2 = '/home/valerie/Python/landuse/Images/testing/results_17Dec2020_23h16.txt'
f3 = '/home/valerie/Python/landuse/Images/testing/results_18Dec2020_18h26.txt'
f4 = '/home/valerie/Python/landuse/Images/testing/results_19Dec2020_08h38.txt'
f5 = '/home/valerie/Python/landuse/Images/testing/results_20Dec2020_20h28.txt'
f6 = '/home/valerie/Python/landuse/Images/testing/results_21Dec2020_22h51.txt'
f7 = '/home/valerie/Python/landuse/Images/testing/results_EQL_600_095.txt'

# Collect precision for each class from txt file
_ , bl_precision, _ ,_ , labels = Read_result_from_txt(baseline)
_ , fCBL2, _ ,_ , _ = Read_result_from_txt(f0)
_ , eql_09_600, _ ,_ , _ = Read_result_from_txt(f1)
_ , eql_075_600, _ ,_ , _ = Read_result_from_txt(f2)
_ , eql_09_300, _ ,_ , _ = Read_result_from_txt(f3)
_ , eql_075_300, _ ,_ , _ = Read_result_from_txt(f4)
_ , eql_075_300v2, _ ,_ , _ = Read_result_from_txt(f5)
_ , eql_050_300, _ ,_ , _ = Read_result_from_txt(f6)
_ , eql_095_600, _ ,_ , _ = Read_result_from_txt(f7)

# Select grouping (rare common or frequent classes)
sel_group = 'rare'
fCBL2_r,_           = select_group(fCBL2, group=sel_group)
eql_095_600,_       = select_group(eql_095_600, group=sel_group)
eql_09_600_r,labels_r = select_group(eql_09_600, group=sel_group)
eql_075_600_r,_    = select_group(eql_075_600, group=sel_group)
eql_09_300_r,_     = select_group(eql_09_300, group=sel_group)
eql_075_300_r,_    = select_group(eql_075_300, group=sel_group)
eql_075_300v2_r,_  = select_group(eql_075_300v2, group=sel_group)
eql_050_300_r,_    = select_group(eql_050_300, group=sel_group)
bl_precision_r,_   = select_group(bl_precision, group=sel_group)
labels = labels_r

# Plotting parameters
x = np.arange(len( labels ))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
#rectsref = ax.bar(x , bl_precision_r, width, label='Baseline precision(absolute)')
rects0 = ax.bar(x +0.1, difference (fCBL2_r), width, label='fCBL2_r')
'''rects1 = ax.bar(x +0.1, difference (eql_09_600_r), width, label='EQL 0.9 600')
rects3 = ax.bar(x +0.2 , difference (eql_09_300_r), width, label='EQL 0.9 300')
rects2 = ax.bar(x +0.3, difference (eql_075_600_r), width, label='EQL 075 600')
rects4 = ax.bar(x +0.4, difference (eql_075_300_r), width, label='EQL 075 300')
rects5 = ax.bar(x +0.5 , difference (eql_075_300v2_r), width, label='EQL 075 300v2')
rects6 = ax.bar(x +0.6 , difference (eql_050_300_r), width, label='EQL 050 300')
rects7 = ax.bar(x +0.7 , difference (eql_095_600), width, label='EQL 0.95 600')'''
ax.axhline(y=0, color='k', lw=0.5)

# Add some text for label, title and custom x-axis tick label, etc.
ax.set_ylabel('Precision difference with baseline')
ax.set_title(sel_group+' classes scores compared to baseline model')
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize = 'x-small',rotation = 45, ha="right")
ax.set_ylim(-1, 1)

ax.legend(fontsize='xx-small')

#autolabel(rects1)
#autolabel(rects2)
fig.tight_layout()

dest = '/home/valerie/Python/landuse/Images/testing/baseline_vs_eql_n111'+sel_group+'.png'
plt.savefig(dest,dpi=200,bbox_inches='tight')
