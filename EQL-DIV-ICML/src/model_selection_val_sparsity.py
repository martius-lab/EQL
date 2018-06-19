import os, sys
import stat
import numpy as np
from operator import itemgetter
'''
expects a file with one row per network and columns reporting the parameters and sparsity and performance
First line should be the column names, #C col1 col2 col3...
then one additional comments line:  # extrapolation datasets etc
A sample file is in example_parameter_scan_result.txt

These are the typical columns is the file.
['k', 'iter', 'layers', 'epochs', 'nodes', 'lr', 'L1', 'L2', 'shortcut', 'batchsize', 'regstart', 'regend', 
'id','dataset', 'gradient', 'numactive', 'bestnumactive', 'bestepoch','dups', 'inserts', 'runtime', 'extrapol1', 'extrapol2', 'extrapol3',
 'extrapolbest1', 'extrapolbest2', 'extrapolbest3', 'valerror', 'valerrorbest', 'testerror']
 
'''
def select_instance(file):
    value_dict = {}
    with open(file ,'r') as file:
            k = 0
            lines = file.readlines()
            keys = lines[0].split()[1:]
            extrapolL = [x for x in keys if ("extrapol" in x and not "best" in x)]
            for key in keys:
                    nums = []
                    for l in lines[2:]: # to remove #the line containing "#extra datasets"
                        nums.append(l.split()[k])
                    k += 1
                    value_dict[key] = nums
                    #print key , value_dict[key]
    lines = 0
    e = []
    e_mean = []
    e_var = []
    for i in range(len(value_dict["id"])):
            value_dict["id"][int(i)] = int(value_dict["id"][int(i)])
            value_dict["nodes"][int(i)] = int(value_dict["nodes"][int(i)])
            value_dict["numactive"][int(i)] = float(value_dict["numactive"][int(i)])
            value_dict["iter"][int(i)] = int(value_dict["iter"][int(i)])
            value_dict["valerror"][int(i)] = float(value_dict["valerror"][int(i)])
            # value_dict["valextrapol"][int(i)] = float(value_dict["valextrapol"][int(i)])
            for k in extrapolL:
                    value_dict[k][int(i)] = float(value_dict[k][int(i)])
            lines += 1
    print "lines: ", lines
    
    active_ = []
    validation_ = []
    id_ = []
    extrapol_ = []
    #extrapol_val = []

    for i in range(lines):
        validation_.append(value_dict["valerror"][i])
        active_.append(value_dict["numactive"][i])
        if "extrapol2" in value_dict:
            extrapol_.append(value_dict["extrapol2"][i])
        id_.append(value_dict["id"][i])
        
        #extrapol_val.append(value_dict["valextrapol"][i])
    active = np.asarray(active_)
    active = (active-np.min(active))/(np.max(active)-np.min(active)) # normalize
    validation = np.asarray(validation_)    
    validation = (validation-np.min(validation))/(np.max(validation)-np.min(validation)) # normalize

    norm_score = np.sqrt(active**2 + validation**2)
    # only for information
    
    if len(extrapol_) > 0:
        best_extrapol = sorted(zip(id_, extrapol_), key=itemgetter(1))[0]
        print (" best extrapolating model: (only for information):", best_extrapol)
        

    score = zip(list(norm_score), id_, active_, validation_, extrapol_)
    score.sort(key = itemgetter(0))
    best_instance = score[0]
    print ("selected instance model: score: {} id: {} #active: {}\t val-error: {}\t extra-pol2-error: {}".format(*best_instance))

    # (best_instance[3], score)
    return dict(zip(['score','id', 'num_active', 'valerror', 'extrapol2'], best_instance))
