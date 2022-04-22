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
##############################
#uniform model
def hyper_prior_U(dataset,mlo,mup):
    return (( dataset['mu'] >= mlo) & (dataset['mu'] <= mup))/np.abs(mup-mlo)

hp_priors_U = dict(mlo=Uniform(0.9, 2.9, 'mlo',r'$\rm{m^l}$'),
                mup=Uniform(1.9, 2.9, 'mup',r'$\rm{m^u}$'))

#logu
def hyper_prior_logu(dataset,mlo,mup):
    return (( dataset['mu'] >= mlo) & (dataset['mu'] <= mup))/( dataset['mu'] * np.log(mup / mlo) )

hp_priors_logu = dict(mlo=LogUniform(0.9, 2.9, 'mlo',r'$\rm{m^l}$'),
                mup=LogUniform(1.9, 2.9, 'mup',r'$\rm{m^u}$'))

#turn_on_pow
def hyper_prior_turn_on_pow(dataset, alpha, mmin, mmax, delta_m):
    parameters = dict(
        alpha=alpha, mmin=mmin, mmax=mmax, delta_m=delta_m)
    pow_norm = norm_ppow(parameters)
    probability = turn_on_pow(dataset['mu'], parameters, pow_norm)
    return probability

def turn_on_pow(ms, parameters, pow_norm):
    al, mn, mx, delta_m = extract_mass_parameters(parameters)
    p_pow = ppow(ms, parameters) / pow_norm
    return  p_pow 

def ppow(ms, parameters):
    """1d unnormalised powerlaw mass probability with smoothed low-mass end"""
    al, mn, mx, delta_m = extract_mass_parameters(parameters)
    return ms**(-al) * window(ms, mn, mx, delta_m)

def norm_ppow(parameters):
    """normalise ppow, requires m1s, an array of m values, and dm, the spacing of
    that array"""
    return np.trapz(ppow(m1s, parameters), m1s)

def window(ms, mn, mx, delta_m):
    """Apply a one sided window between mmin and mmin+dm to the mass pdf.

    The upper cut off is a step function,
    the lower cutoff is a logistic rise over delta_m solar masses.

    See T&T18 Eq

    """
    dM = mx - mn
    delta_m /= dM
    # some versions of numpy can't deal with pandas columns indexing an array
    ms_arr = np.array(ms)
    sel_p = (ms_arr >= mn) & (ms_arr <= (mn + delta_m * dM))
    ms_p = ms_arr[sel_p] - mn
    Zp = np.nan_to_num(2 * delta_m * (1 / (2 * ms_p / dM) +
                       1 / (2 * ms_p / dM - 2 * delta_m)))
    window = np.ones_like(ms)
    window[(ms_arr < mn) | (ms_arr > mx)] = 0
    window[sel_p] = 1 / (np.exp(Zp) + 1)
    return window

def extract_mass_parameters(parameters):
    """extract the parameters of the mass distribution hyperparameters used in
    T&T18 from either a list or dictionary."""
    if isinstance(parameters, list):
        return parameters
    elif isinstance(parameters, dict):
        keys = ['alpha', 'mmin', 'mmax', 'delta_m']
        return [parameters[key] for key in keys]

# set up arrays for numerical normalisation
m1s = np.linspace(0.9, 2.9, 100)
dm = m1s[1] - m1s[0]

hp_priors_turn_on_pow= dict(alpha=Uniform(-5, 25, 'alpha', '$\\alpha$'),
                 mmin=Uniform(0.9, 2.9, 'mmin', '$mmin$'),
                mmax=Uniform(1.9, 2.9, 'mmax', '$mmax$'),
                delta_m=Uniform(0.01, 1, 'delta', '$\\delta$'))

#pow
def hyper_prior_pow(dataset,mlo,mup,beta):
    beta=-1*beta
    return (( dataset['mu'] >= mlo) & (dataset['mu'] <= mup))*((1+beta)/(mup**(1+beta)-mlo**(1+beta)))*dataset['mu']**beta

hp_priors_pow = dict(mlo=Uniform(0.9, 2.9, 'mlo',r'$\rm{m^l}$'),
                mup=Uniform(1.9, 2.9, 'mup',r'$\rm{m^u}$'),
                beta=Uniform(-5, 25, 'beta','$\\beta$'))

#lognorm
def hyper_prior_lognorm(dataset, s_mu, s_sigma):
    return np.exp(- (np.log(dataset['mu']) - s_mu)**2 / (2 * s_sigma**2)) /\
        (2 * np.pi * s_sigma**2)**0.5/(dataset['mu'])

hp_priors_lognorm = dict(s_mu=LogUniform(0.9, 2.9, 's_mu', '$\mu$'),
                 s_sigma=LogUniform(0.01, 2, 's_sigma', '$\sigma$'))

#G
def hyper_prior_G(dataset, mu, sigma):
    mup=2.9
    mlo=0.9
    normalisingTerm = 0.5 * ( erf((mu-mlo)/(np.sqrt(2) * sigma)) -  erf((mu-mup)/(np.sqrt(2) * sigma)) )
    return (np.exp(- (dataset['mu'] - mu)**2 / (2 * sigma**2)) /\
        (2 * np.pi * sigma**2)**0.5)/normalisingTerm 
hp_priors_G = dict(mu=Uniform(0.9, 2.9, 's_mu', '$\mu$'),
                 sigma=Uniform(0.01, 2, 's_sigma', '$\sigma$'))

#3G
def hyper_prior_3G(dataset, mu1, sigma1,mu2,sigma2,alpha,mu3,sigma3,beta):
    mup=2.9
    mlo=0.9
    normalisingTerm1 = 0.5 * ( erf((mu1-mlo)/(np.sqrt(2) * sigma1)) -  erf((mu1-mup)/(np.sqrt(2) * sigma1)) )
    normalisingTerm2 = 0.5 * ( erf((mu2-mlo)/(np.sqrt(2) * sigma2)) -  erf((mu2-mup)/(np.sqrt(2) * sigma2)) )
    normalisingTerm3 = 0.5 * ( erf((mu3-mlo)/(np.sqrt(2) * sigma3)) -  erf((mu3-mup)/(np.sqrt(2) * sigma3)) )
    if mu1 < mu2 and mu3>mu2  and alpha+beta<=1:
        return ((alpha*(np.exp(- (dataset['mu'] - mu1)**2 / (2 * sigma1**2)) /(2 * np.pi * sigma1**2)**0.5))/normalisingTerm1)\
        +((beta*(np.exp(- (dataset['mu'] - mu2)**2 / (2 * sigma2**2)) /(2 * np.pi * sigma2**2)**0.5))/normalisingTerm2)\
        +(((1-alpha-beta)*(np.exp(- (dataset['mu'] - mu3)**2 / (2 * sigma3**2)) /(2 * np.pi * sigma3**2)**0.5))/normalisingTerm3)
    else:
        return 0
hp_priors_3G = dict(mu1=Uniform(0.9, 2.9, 'mu1', '$\mu_1$'),
                 sigma1=Uniform(0.01, 2, 'sigma1', '$\sigma_1$'),
                mu2=Uniform(0.9, 2.9, 'mu2', '$\mu_2$'),
                sigma2=Uniform(0.01, 2, 'sigma2', '$\sigma_2$'),
                alpha=Uniform(0.01, 1, 'alpha', '$\\alpha$'),
                mu3=Uniform(0.9, 2.9, 'mu3', '$\mu_3$'),
                sigma3=Uniform(0.01, 2, 'sigma3', '$\sigma_3$'),
                beta=Uniform(0.01, 1, 'beta', '$\\beta$'))

                
#2G 
def hyper_prior_2G(dataset, mu1, sigma1,mu2,sigma2,alpha):
    mup=2.9
    mlo=0.9
    normalisingTerm1 = 0.5 * ( erf((mu1-mlo)/(np.sqrt(2) * sigma1)) -  erf((mu1-mup)/(np.sqrt(2) * sigma1)) )
    normalisingTerm2 = 0.5 * ( erf((mu2-mlo)/(np.sqrt(2) * sigma2)) -  erf((mu2-mup)/(np.sqrt(2) * sigma2)) )
    return ((mu1 < mu2)  & ( dataset['mu'] >= mlo) & (dataset['mu'] <= mup)) *\
        ( (( alpha*(np.exp(- (dataset['mu'] - mu1)**2 / (2 * sigma1**2)) /(2 * np.pi * sigma1**2)**0.5)) /normalisingTerm1) +\
        (1-alpha)*( ((np.exp(- (dataset['mu'] - mu2)**2 / (2 * sigma2**2)) /(2 * np.pi * sigma2**2)**0.5) ) / normalisingTerm2) )
hp_priors_2G = dict(mu1=Uniform(0.9, 2.9, 'mu1', '$\mu_1$'),
                 sigma1=Uniform(0.01, 2, 'sigma1', '$\sigma_1$'),
                mu2=Uniform(0.9, 2.9, 'mu2', '$\mu_2$'),
                sigma2=Uniform(0.01, 2, 'sigma2', '$\sigma_2$'),
                alpha=Uniform(0.01, 1, 'alpha', '$\\alpha$'))

#2G cut 
def hyper_prior_2G_cut(dataset, mu1, sigma1,mu2,sigma2,alpha,mup):
    mlo=0.9
    normalisingTerm1 = 0.5 * ( erf((mu1-mlo)/(np.sqrt(2) * sigma1)) -  erf((mu1-mup)/(np.sqrt(2) * sigma1)) )
    normalisingTerm2 = 0.5 * ( erf((mu2-mlo)/(np.sqrt(2) * sigma2)) -  erf((mu2-mup)/(np.sqrt(2) * sigma2)) )
    return ((mu1 < mu2)  & ( dataset['mu'] >= mlo) & (dataset['mu'] <= mup)) *\
        ( (( alpha*(np.exp(- (dataset['mu'] - mu1)**2 / (2 * sigma1**2)) /(2 * np.pi * sigma1**2)**0.5)) /normalisingTerm1) +\
        (1-alpha)*( ((np.exp(- (dataset['mu'] - mu2)**2 / (2 * sigma2**2)) /(2 * np.pi * sigma2**2)**0.5) ) / normalisingTerm2) )
hp_priors_2G_cut = dict(mu1=Uniform(0.9, 2.9, 'mu1', '$\mu_1$'),
                 sigma1=Uniform(0.01, 2, 'sigma1', '$\sigma_1$'),
                mu2=Uniform(0.9, 2.9, 'mu2', '$\mu_2$'),
                sigma2=Uniform(0.01, 2, 'sigma2', '$\sigma_2$'),
                alpha=Uniform(0.01, 1, 'alpha', '$\\alpha$'),
                mup=Uniform(1.9, 2.9, 'mup',r'$\rm{m^u}$') )

#SST
from scipy.special import beta
def hyper_prior_sst(dataset, mu,sigma,nu,tau):
        c = 2 * nu * ((1 + nu ** 2) *
                                beta(0.5, tau / 2) *
                                tau ** 0.5) ** -1
        m = ((2 * tau ** 0.5) * (nu - nu ** -1)) / (
                (tau - 1) * beta(0.5, 0.5 * tau))
        s2 = ((tau / (tau - 2)) * (
                nu ** 2 + nu ** -2 - 1) - m ** 2)
        mu_0 = mu - (sigma * m / np.sqrt(s2))
        sigma_0 = sigma / np.sqrt(s2)
        z = (dataset['mu'] - mu_0) / sigma_0
        p = np.where(dataset['mu'] < mu_0,
                     (c / sigma_0) * (1 + ((nu ** 2) * (z ** 2)) / tau) ** (
                             -(tau + 1) / 2),
                     (c / sigma_0) * (1 + (z ** 2) / ((nu ** 2) * tau)) ** (
                             -(tau + 1) / 2))
        return p

hp_priors_sst = dict(mu=Uniform(0.9, 2.9, 'mlo',r'$\rm{m^l}$'),
                sigma=Uniform(0.01, 2, 'sigma',r'$\rm{m^u}$'),
                nu=Uniform(0,8,'nu'),
                   tau=Uniform(2.001,20,'tau') )
#gamma
from scipy.special import beta
from scipy.special import gamma
def hyper_prior_gamma(dataset, k,theta):
    return (1 / (gamma(k)*theta**k)) * dataset['mu']**(k-1) *np.exp(-dataset['mu']/theta)

hp_priors_gamma = dict(k=Uniform(0, 80, 'k',r'$k$'),
                theta=Uniform(0.01, 0.1, 'theta',r'$\theta$') )

#2G cut min
def hyper_prior_2G_cut_min(dataset, mu1, sigma1,mu2,sigma2,alpha,mup,mlo):
    normalisingTerm1 = 0.5 * ( erf((mu1-mlo)/(np.sqrt(2) * sigma1)) -  erf((mu1-mup)/(np.sqrt(2) * sigma1)) )
    normalisingTerm2 = 0.5 * ( erf((mu2-mlo)/(np.sqrt(2) * sigma2)) -  erf((mu2-mup)/(np.sqrt(2) * sigma2)) )
    return ((mu1 < mu2)  & ( dataset['mu'] >= mlo) & (dataset['mu'] <= mup)) *\
        ( (( alpha*(np.exp(- (dataset['mu'] - mu1)**2 / (2 * sigma1**2)) /(2 * np.pi * sigma1**2)**0.5)) /normalisingTerm1) +\
        (1-alpha)*( ((np.exp(- (dataset['mu'] - mu2)**2 / (2 * sigma2**2)) /(2 * np.pi * sigma2**2)**0.5) ) / normalisingTerm2) )
hp_priors_2G_cut_min = dict(mu1=Uniform(0.9, 2.9, 'mu1', '$\mu_1$'),
                 sigma1=Uniform(0.01, 2, 'sigma1', '$\sigma_1$'),
                mu2=Uniform(0.9, 2.9, 'mu2', '$\mu_2$'),
                sigma2=Uniform(0.01, 2, 'sigma2', '$\sigma_2$'),
                alpha=Uniform(0.01, 1, 'alpha', '$\\alpha$'),
                mup=Uniform(1.9, 2.9, 'mup',r'$\rm{m^u}$'),
                mlo=Uniform(0.9, 1.5, 'mlo',r'$\rm{m^l}$') )

#2G min
def hyper_prior_2G_min(dataset, mu1, sigma1,mu2,sigma2,alpha,mlo):
    mup=2.9
    normalisingTerm1 = 0.5 * ( erf((mu1-mlo)/(np.sqrt(2) * sigma1)) -  erf((mu1-mup)/(np.sqrt(2) * sigma1)) )
    normalisingTerm2 = 0.5 * ( erf((mu2-mlo)/(np.sqrt(2) * sigma2)) -  erf((mu2-mup)/(np.sqrt(2) * sigma2)) )
    return ((mu1 < mu2)  & ( dataset['mu'] >= mlo) & (dataset['mu'] <= mup)) *\
        ( (( alpha*(np.exp(- (dataset['mu'] - mu1)**2 / (2 * sigma1**2)) /(2 * np.pi * sigma1**2)**0.5)) /normalisingTerm1) +\
        (1-alpha)*( ((np.exp(- (dataset['mu'] - mu2)**2 / (2 * sigma2**2)) /(2 * np.pi * sigma2**2)**0.5) ) / normalisingTerm2) )
hp_priors_2G_min = dict(mu1=Uniform(0.9, 2.9, 'mu1', '$\mu_1$'),
                 sigma1=Uniform(0.01, 2, 'sigma1', '$\sigma_1$'),
                mu2=Uniform(0.9, 2.9, 'mu2', '$\mu_2$'),
                sigma2=Uniform(0.01, 2, 'sigma2', '$\sigma_2$'),
                alpha=Uniform(0.01, 1, 'alpha', '$\\alpha$'),
                mlo=Uniform(0.9, 1.5, 'mlo',r'$\rm{m^l}$') )

