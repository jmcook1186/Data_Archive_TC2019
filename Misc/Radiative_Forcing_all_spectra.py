#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:48:35 2017

@author: joe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from scipy import stats

################ IMPORT CSVs FOR ALBEDO AND HCRF DATA #########################
########## DEFINE SITES TO INCLUDE IN EACH IMPURITY LOADING CLASS #############

WL = np.arange(350, 2500, 1)

alb_master = pd.read_csv('/home/joe/Code/Albedo_master.csv')

HAsites = ['13_7_SB2', '13_7_SB4',
           '14_7_S5', '14_7_SB1', '14_7_SB4', '14_7_SB5', '14_7_SB10',
           '15_7_SB3',
           '21_7_S3',
           '21_7_SB1', '21_7_SB6', '21_7_SB7',
           '22_7_SB4', '22_7_SB5', '22_7_S3', '22_7_S5',
           '23_7_SB3', '23_7_SB4', '23_7_SB5', '23_7_S3', '23_7_S5',
           '24_7_SB2', '24_7_S1',
           '25_7_S1']

LAsites = ['13_7_S2', '13_7_S5', '13_7_SB1',
           '14_7_S2', '14_7_S3', '14_7_SB2', '14_7_SB3', '14_7_SB7', '14_7_SB9',
           '15_7_S2', '15_7_S3', '15_7_SB4',
           '20_7_SB1', '20_7_SB3',
           '21_7_S1', '21_7_S5', '21_7_SB2', '21_7_SB4',
           '22_7_SB1', '22_7_SB2', '22_7_SB3', '22_7_S1',
           '23_7_S1', '23_7_S2',
           '24_7_SB2', '24_7_S2',
           '25_7_S2', '25_7_S4', '25_7_S5']

Snowsites = ['13_7_S4',
             '14_7_S4', '14_7_SB6', '14_7_SB8',
             '17_7_SB1', '17_7_SB2']  # ,
# '20_7_SB4']

CIsites = ['13_7_S1', '13_7_S3', '13_7_SB3', '13_7_SB5',
           '14_7_S1',
           '15_7_S1', '15_7_S4', '15_7_SB1', '15_7_SB2', '15_7_SB5',
           '20_7_SB2',
           '21_7_S2', '21_7_S4', '21_7_SB3', '21_7_SB5', '21_7_SB8',
           '22_7_S2', '22_7_S4',
           '23_7_SB1', '23_7_SB2',
           '23_7_S4',
           '25_7_S3']

########## CREATE EMPTY DATAFRAMES AND LISTS TO POPULATE LATER ON #############
###############################################################################

HA_alb = pd.DataFrame()
LA_alb = pd.DataFrame()
Snow_alb = pd.DataFrame()
CI_alb = pd.DataFrame()
HA_RF_list = []
LA_RF_list = []

WL = np.arange(350, 2500, 1)
PsynthEfficiency = 0.05  # Efficiency of photosynthesis in ice algae, assumed 5%
#### Loop through each group and pull the albedo spectra into dataframes ######
########### also calculate mean spectrum from all sites #######################


for i in HAsites:
    HA_alb[i] = alb_master[i]
    HA_mean = HA_alb.mean(axis=1)
    HA_max = HA_alb.max(axis=1)
    HA_min = HA_alb.min(axis=1)
    HA_std = HA_alb.std(axis=1)

for ii in LAsites:
    LA_alb[ii] = alb_master[ii]
    LA_mean = LA_alb.mean(axis=1)
    LA_max = LA_alb.max(axis=1)
    LA_min = LA_alb.min(axis=1)

for iii in Snowsites:
    Snow_alb[iii] = alb_master[iii]
    Snow_mean = Snow_alb.mean(axis=1)
    Snow_max = Snow_alb.max(axis=1)
    Snow_min = Snow_alb.min(axis=1)

for iv in CIsites:
    CI_alb[iv] = alb_master[iv]
    CI_mean = CI_alb.mean(axis=1)
    CI_max = CI_alb.max(axis=1)
    CI_min = CI_alb.min(axis=1)

# optional plotting of mean albedo spectra
plt.figure(1)
plt.plot(WL, HA_mean, 'g', label='Hbio')
plt.plot(WL, LA_mean, 'r', label='Lbio')
plt.plot(WL, CI_mean, 'b', label='CI')
plt.plot(WL, Snow_mean, 'k', label='SN')
plt.ylabel('Albedo')
plt.xlabel('Wavelength(nm)')
plt.xlim(350, 2200)
plt.ylim(0, 1)


###############################################################################
######################### FUNCTION DEFINITIONS ################################


###################### incoming irradiance spectra ##########################
# function imports irradiance from csv file, appends to dataframe, interpolates
# to 1nm resolution, outputs to new dataframe column and returns dataframe.
# Data comes from solar irradiance simulator and was produced hourly for the
# 2017 Black and Bloom field site for 22/7/2017

def irradiance():
    incomingDF = pd.DataFrame()  # dataframe for incoming irradiance data
    incomingDF2 = pd.DataFrame()  # output dataframe
    path = '/home/joe/Code/S6/'  # path where csv files are held
    file = 'Incoming_'  # filename -number and extension
    hours = ['03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13',
             '14', '15', '16', '17', '18', '19', '20', '21']  # clock hours of modelled irradiance
    for i in hours:
        p = pd.read_csv(str(path + file + i + '.csv'))
        p2 = np.squeeze(np.array(p))
        incomingDF[i] = p2  # loop hours and add each spectrum to dataframe

    for i in hours:
        temp = np.array(incomingDF[i][0:215])  # limit to waveband of interest
        temp = temp.ravel()  # ravel to standardise shape of DF (=1D)
        x = np.arange(temp.size)  # x values for interpolation
        y = temp  # y values for interpolation
        new_length = 2150  # number of values in newly interpolated dataset
        new_x = np.linspace(x.min(), x.max(), new_length)  # function fitting
        new_y = sci.interpolate.interp1d(x, y, kind='cubic')(new_x)  # interpolation
        incoming = new_y  # assignment of spectrum at new resolution
        incomingDF2[i] = incoming  # populate output dataframe

    return hours, incomingDF2


############################ IRF ##############################################
# function takes incoming irradiance spectra and multiplies it by the difference
# between clean ice and algal ice spectra for Hbio and Lbio surfaces
# The spectra used are the mean albedo forneach surface class from the
# July 2017 Black and Bloom Project Field camp. measurements made by J Cook and
# A Tedstone. The function IRF can be used to generate hourly biological IRFs
# when called in a loop or a single IRF for one time point.


def IRF(CI, HA, LA, incoming):
    HA_RF_temp = (CI - HA)  # diffeence between clean ice and algal ice albedo
    LA_RF_temp = (CI - LA)

    HA_RF = [(a * b) * (1 - PsynthEfficiency) for a, b in
             zip(HA_RF_temp, incoming)]  # mutiply by incoming irradiance and account for
    LA_RF = [(a * b) * (1 - PsynthEfficiency) for a, b in
             zip(LA_RF_temp, incoming)]  # photosynthetic energy losses (won't heat ice)
    HA_RF_total = np.round(np.sum(HA_RF), 0)  # total rounded to nearest whole number
    LA_RF_total = np.round(np.sum(LA_RF), 0)
    HA_RF_list.append(HA_RF_total)  # append wavelength-integrated IRF to list
    LA_RF_list.append(LA_RF_total)

    return HA_RF, LA_RF, HA_RF_total, LA_RF_total, HA_RF_list, LA_RF_list


def IRF_daily(HA_RF_DF, LA_RF_DF):  # requires a dataframe to have been produced of hourly RFs using IRF function
    HA_RF_temp = (HA_RF_DF.sum(axis=1)) * 3600  # sum of IRF per hour, x 3600 (3600 seconds per hour)
    HA_RF_daily = HA_RF_temp.sum(axis=0)  # sum of all hourly IRFs over entire day

    LA_RF_temp = (LA_RF_DF.sum(axis=1)) * 3600
    LA_RF_daily = LA_RF_temp.sum(axis=0)

    return HA_RF_daily, LA_RF_daily


def melt(HA_RF_daily, LA_RF_daily):
    HA_daily_melt = HA_RF_daily / (336 * 1e4)  # divide daily radiative forcing in J by the latent heat of melting
    LA_daily_melt = LA_RF_daily / (336 * 1e4)  # multiplied by 10e4 to convert from m2 to cm2
    return HA_daily_melt, LA_daily_melt


##############################################################################
############### DRIVE FUNCTIONS AND and plot #################################
##############################################################################


hours, incomingDF2 = irradiance()  # call irradiance function to retrieve hours
# and incoming irradiance dataframe

# set up new DFs to store hourly radiative forcings generated in loop
HA_RF_DF = pd.DataFrame()
LA_RF_DF = pd.DataFrame()

HA_result = []
LA_result = []
# call IRF function in loop to retrieve IRFs per hour

for i in HA_alb.columns:
    HA = HA_alb.loc[:][i]
    for ii in CI_alb.columns:
        CI = CI_alb.loc[:][ii]
        for iii in LA_alb.columns:
            LA = LA_alb.loc[:][iii]


            for i in hours:

                incoming = incomingDF2[i]
                HA_RF, LA_RF, HA_RF_total, LA_RF_total, HA_RF_list, LA_RF_list = IRF(CI, HA, LA,
                                                                                     incoming)  # call function

                HA_RF_DF[i] = HA_RF  # append columns to DF for each hourly IRF
                LA_RF_DF[i] = LA_RF


                # Call IRF_daily function to retrieve IRF integrated over entire day
                HA_RF_daily, LA_RF_daily = IRF_daily(HA_RF_DF, LA_RF_DF)
                print('Hbio daily RF in kJ = ', HA_RF_daily / 1000, 'Lbio daily RF in kJ = ', LA_RF_daily / 1000)

                # Call melt function to retrieve melt in cm w.e. over the enire day
                HA_daily_melt, LA_daily_melt = melt(HA_RF_daily, LA_RF_daily)  # function call
                print('Hbio melt in cm w.e. = ', HA_daily_melt, 'Lbio melt in cm w.e. = ', LA_daily_melt)

                HA_result.append(HA_daily_melt)
                LA_result.append(LA_daily_melt)

# list of daily melt rates to array
HA_result = np.array(HA_result)
LA_result =np.array(LA_result)

# remove any where CI < HA or LA
HA_result = HA_result[HA_result>0]
LA_result = LA_result[LA_result>0]

# mean of daily melt rates
HA_result_mean = np.mean(HA_result)
LA_result_mean = np.mean(LA_result)

# standard error for daily melt rates
HA_result_se = stats.sem(HA_result)
LA_result_se = stats.sem(LA_result)

# print results
print("\n Hbio daily melt = {} mm w.e. +/- {} (s.e.)".format(HA_result_mean,HA_result_se))
print("\n Lbio daily melt = {} mm w.e. +/- {} (s.e.)".format(LA_result_mean,LA_result_se))