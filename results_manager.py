# results compaction

import matplotlib, os, glob,torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def Print_results_to_file(file_names,model_names, output_name):
    df =pd.DataFrame([])
    if len(file_names)!=len(model_names): print('Problem with file length names or model name')

    for idx, path in enumerate(file_names ):
        print(idx,path)
        a,b,c,d,e = Read_result_from_txt(src+path+'.txt')
        f1_score = 2* np.array(b)*np.array(c) / (np.array(b) + np.array(c))
        f1_score=np.nan_to_num(f1_score)
        print(f1_score)

        df[str(model_names[idx]+'_f1score')]= f1_score
        df[str(model_names[idx]+'_precision')]= b 
        df[str(model_names[idx]+'_recall')]= c

    df[str(model_names[idx]+'lab')]= e
    df[str(model_names[idx]+'class_size')]= d
    df.to_csv(output_name)

def ListAllModelAndParameters(print_to_file=False) :
    # produce list of all models with training parameters
    if print_to_file==True:
        file_name = '/home/valerie/Python/landuse/Images/results_summary_models.txt'
        fd = open(file_name,'w')
        fd.write('hello\n')
        fd.close()  
    os.chdir('/home/valerie/Python/landuse/TrainedModels/')
    state_dict =glob.glob('*.pth')
    for src in state_dict :
        checkpoint = torch.load(src)
        params = checkpoint['params']
        accuracy= checkpoint['accuracy']
        v_acc = accuracy[1::2]
        print(src,params,'Number of epochs:',len(accuracy)/2, 'Best validation accuracy',max(v_acc))
        if print_to_file==True:
            fd = open(file_name,'a+')
            fd.write(src+'\t'+ str(params)+'\tNumber of epochs:'+str (len(accuracy)/2)+ '\tBest validation accuracy'+str(max(v_acc))+'\n\n')
            fd.close()       
    return

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


# List of files for the main experiment :  
src = '/home/valerie/Python/landuse/Images/testing/results_'
baseline = '13Dec2020_20h20'  #Baseline
und100 = '100_1901'  #undersampling
und1000= '1000_1901'
twoPh = 'twophases_1000_1901'
sqrtIFW = 'sq_inv_freq'
IFW =  '14Dec2020_21h16'   #14Dec2020_21h16,pth
EQL = '19Dec2020_08h38' #19DEc2020_08h38 eql 075 300
sCBL= 'sCBL_099'
fCBL = 'focalCBL_099_1'

file_names = [baseline,    und100,  und1000, twoPh,  IFW,    sqrtIFW,EQL, sCBL,    fCBL]
model_names= ['baseline','und100','und1000','twoPh','IFW','sqrtIFW','EQL','sCBL','fCBL']
# Saving all results into on clean csv file
fsave = '/home/valerie/Python/landuse/Reporting/overall_metrics_2601.txt'

Print_results_to_file(file_names,model_names, fsave)

#---------------------------------------------------------------
# List of files for the clean experiment : 
src = '/home/valerie/Python/landuse/Images/testing/results_'
baseline = 'cleanv2_baseline21'
und100 = 'cleanv2_100_1901'
und1000= 'cleanv2_1000_1901'
twoPh = 'cleanv2_twophases_1000_1901'
IFW = 'cleanv2_inverse_freq'
sqrtIFW = 'clean_sqrt_inv_freq'
EQL = 'cleanv2_EQL_1000_09'
sCBL= 'cleanv2_CBL_099'
fCBL = 'cleanv2_focalCBL_099_1'

file_names = [baseline,und100,und1000,twoPh,IFW, sqrtIFW,EQL,sCBL,fCBL]
model_names= ['baseline','und100','und1000','twoPh','IFW','sqrtIFW','EQL','sCBL','fCBL']
# Saving all results into on clean csv file
fsave = '/home/valerie/Python/landuse/Reporting/cleanv2_metrics_2601.txt'
Print_results_to_file(file_names,model_names, fsave)

#------------------------------------------------
# list of files for the EQL experiment on full dataset:
src = '/home/valerie/Python/landuse/Images/testing/results_'
EQL_600_09 = '16Dec2020_17h55'
EQL_600_075 = '17Dec2020_23h16'
EQL_300_090 = '18Dec2020_18h26'
EQL_300_075 = '19Dec2020_08h38'
EQL_300_075v2 = '20Dec2020_20h28'
EQL_300_050 = '21Dec2020_22h51'
EQL_600_095 = 'EQL_600_095'
EQL_600_095v2 = 'EQL_600_095v2'
EQL_1800_09 = 'EQL_1800_09'

file_names = [EQL_600_09, EQL_600_075,EQL_300_090,EQL_300_075 ,EQL_300_075v2,EQL_300_050,EQL_600_095, EQL_600_095v2, EQL_1800_09 ]
model_names = ['EQL_600_09','EQL_600_075','EQL_300_090','EQL_300_075 ','EQL_300_075v2','EQL_300_050','EQL_600_095', 'EQL_600_095v2', 'EQL_1800_09' ]
fsave = '/home/valerie/Python/landuse/Reporting/EQL_metrics_2601.txt'
Print_results_to_file(file_names,model_names, fsave)

#------------------------------------------------
# list of files for the EQL experiment on full dataset:
src = '/home/valerie/Python/landuse/Images/testing/results_'
sCBL_099 = 'sCBL_099'
sCBL_0999= 'sCBL_0999'
sCBL_09999= 'sCBL_09999_1901x'
fCBL_099_05 ='focalCBL_099_05'
fCBL_099_1 ='focalCBL_099_1'
fCBL_099_2 ='focalCBL_099_2'
fCBL_099_3 ='focalCBL_099_3'

file_names=[sCBL_099, sCBL_0999, sCBL_09999, fCBL_099_05, fCBL_099_1, fCBL_099_2, fCBL_099_3 ]
model_names = ['sCBL_099','sCBL_0999','sCBL_09999','fCBL_099_05','fCBL_099_1','fCBL_099_2','fCBL_099_3 ']
fsave = '/home/valerie/Python/landuse/Reporting/CBL_metrics_2601.txt'
Print_results_to_file(file_names,model_names, fsave)

#------------------------------------------------
# new function to extract means per model
def Read_global_perf_from_txt(fn):
    '''
    Read classification results for each class from testing report
    '''
    first_record = False
    idx,precision,recall,class_size,label = [],[],[],[],[]

    fd = open(fn, 'r')
    for ligne in fd:
        if ligne.rstrip()[0] == 'K':
            first_record = True      
        
        if first_record == True : 
            ligne = ligne.replace('\n','',1)
            #ligne = ligne.replace('_',' ')
            a,b,c,d,e =ligne.split(',')
            idx+=[int(a)]
            oa +=[float(b)]
            arr +=[float(c)]

    fd.close()
    return idx,precision,recall,class_size,label