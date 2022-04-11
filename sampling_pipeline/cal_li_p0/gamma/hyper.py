import numpy as np
from scipy.special import erf
from scipy.stats import beta as beta_dist
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
import bilby
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from bilby.core.prior import Uniform
from bilby.core.sampler import run_sampler
from bilby.core.prior import LogUniform
from bilby.hyper.likelihood import HyperparameterLikelihood
from scipy import interpolate
from scipy import integrate
import random
import argparse
import os
import glob
import warnings
warnings.filterwarnings("ignore")
#####################
outdir='hy_outdir'
current_path=os.path.abspath( os.path.join(os.getcwd()) )
parent_path=os.path.abspath( os.path.join(os.getcwd(),'..') )
p_parent_path=os.path.abspath( os.path.join(os.getcwd(),'../..') )
parent_dir_name=os.path.basename(parent_path)
current_dir_name=os.path.basename(current_path)
data_exp=np.loadtxt(p_parent_path+'/{}.txt'.format(parent_dir_name))
import sys
sys.path.append(p_parent_path)
import utilizes as utilize
npool=8
NN=48
nlive=400
mx=5000
#################################
data_df=list()
ns_m=data_exp
for i in range(NN):
    re=pd.DataFrame(ns_m[i*5000:(i+1)*5000],columns=['mu'])
    data_df.append(re)
samples = data_df

def run_prior(dataset):
    return 1/(2.5-0.5)

hyper_prior=eval('utilize.hyper_prior_{}'.format(current_dir_name) )

hp_likelihood = HyperparameterLikelihood(
    posteriors=samples, hyper_prior=hyper_prior,
    sampling_prior=run_prior, log_evidences=0, max_samples=mx)

hp_priors =eval('utilize.hp_priors_{}'.format(current_dir_name) ) 

# And run sampler
result = run_sampler(
    likelihood=hp_likelihood, priors=hp_priors, sampler='dynesty', nlive=nlive,
    use_ratio=False, outdir=outdir, label='u_hype_{}'.format(np.random.randint(1,88888)),
    verbose=True, clean=True,npool=npool)
result.plot_corner()

