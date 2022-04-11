import bilby
import numpy as np
import csv
import pandas as pd
import os
import glob

sub_dir_name_list=['3G', 'pow', 'turn_on_pow', 'U', '2G', 'G', '2G_cut', 'lognorm','logu','gamma','sst','2G_cut_min','2G_min']

dir_path=os.path.abspath( os.path.join(os.getcwd()) )
dir_name=os.path.basename(dir_path)

evidence={}
for i in range(len(sub_dir_name_list)):
    try:
        os.path.exists(dir_path+'/'+sub_dir_name_list[i]+"/hy_outdir/*.json")
        fnames= glob.glob(dir_path+'/'+sub_dir_name_list[i]+"/hy_outdir/*.json")
        fname=fnames[0]
        re=bilby.result.read_in_result(filename=fname)
        evidence[sub_dir_name_list[i]]=re.log_evidence
    except:
        evidence[sub_dir_name_list[i]]=0
print(evidence)
evidence=pd.DataFrame.from_dict(evidence,orient='index')

evidence.to_csv('{}_evidence.csv'.format(dir_name),header=0,index=sub_dir_name_list)
