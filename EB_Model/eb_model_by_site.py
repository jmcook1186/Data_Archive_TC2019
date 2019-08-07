#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:39:55 2018

@author: joe
"""

import numpy as np
import pandas as pd
import ebmodel as ebm


def runit(alb):

	total_list = []
	for i in np.arange(0,len(met_data),1):
	##############################################################################
	## Input Data, as per first row of Brock and Arnold (2000) spreadsheet

		lat = 67.0666
		lon = -49.38
		lon_ref = 0
		summertime = 0
		slope = 1.
		aspect = 90.
		elevation = 1020.
		albedo = alb
		roughness = 0.005
		met_elevation = 1020.
		lapse = 0.0065
		day = 202
		time = met_data.loc[i]['Time']
		inswrad = met_data.loc[i]['Radiation']
		airtemp = met_data.loc[i]['AirTemp']
		windspd = met_data.loc[i]['Windspeed']
		avp = 900
		##############################################################################


		SWR,LWR,SHF,LHF = ebm.calculate_seb(
			lat, lon, lon_ref, day, time, summertime,
			slope, aspect, elevation, met_elevation, lapse,
			inswrad, avp, airtemp, windspd, albedo, roughness)

		sw_melt, lw_melt, shf_melt, lhf_melt, total = ebm.calculate_melt(
			SWR,LWR,SHF,LHF, windspd, airtemp)

		total_list.append(total)
		daily_total = np.sum(total_list)

		print(alb,'   ',daily_total)

	return daily_total


# Set up empty lists
HA_melt_list=[]
LA_melt_list = []
CI_melt_list = []
HA_result = []
LA_result = []


# Read in met data and BBAs
met_data = pd.read_csv('/home/joe/Code/EB_Model/met_data.csv')
BBA_list = pd.read_csv('/home/joe/Code/EB_Model/BBA_and_Class.csv')

# filter BBAs by surface class
HA_filter = BBA_list['Class']=='HA'
LA_filter = BBA_list['Class']=='LA'
CI_filter = BBA_list['Class']=='CI'

HA_BBA = BBA_list[HA_filter]
LA_BBA = BBA_list[LA_filter]
CI_BBA = BBA_list[CI_filter]


# run function for each albedo value in each class and append to relevant list

for alb in HA_BBA['BBA']:
	daily_total = runit(alb)
	HA_melt_list.append(daily_total)


for alb in LA_BBA['BBA']:
	daily_total = runit(alb)
	LA_melt_list.append(daily_total)

for alb in CI_BBA['BBA']:
	daily_total = runit(alb)
	CI_melt_list.append(daily_total)

HA_melt_mean = np.mean(HA_melt_list)
HA_melt_std = np.std(HA_melt_list)
LA_melt_mean = np.mean(LA_melt_list)
LA_melt_std = np.std(LA_melt_list)
CI_melt_mean = np.mean(CI_melt_list)
CI_melt_std = np.std(CI_melt_list)


# subtract every bio melt from every clean melt and append to list
for i in HA_melt_list:
	for ii in CI_melt_list:
		if i > ii:
			HA_result.append(i - ii)

for i in LA_melt_list:
	for ii in CI_melt_list:
		if i > ii:
			LA_result.append(i-ii)

# calculate average and std
HA_result_mean = np.mean(HA_result)
LA_result_mean = np.mean(LA_result)
HA_result_std = np.std(HA_result)
LA_result_std = np.std(LA_result)
CI_result_mean = np.mean(CI_melt_list)
CI_result_std = np.std(CI_melt_list)

# calculate error using propagation of error equation
HA_result_error = np.sqrt((HA_result_std**2)+(CI_result_std**2))
LA_result_error = np.sqrt((LA_result_std**2)+(CI_result_std**2))

# calculate % melt attributed to presence of alage in algal sites
melt_attributed_to_HA = HA_result_mean / HA_melt_mean
melt_attributed_to_LA = LA_result_mean / LA_melt_mean

# calculate error using propagation of error equation
melt_attributed_to_HA_error = np.sqrt(((HA_result_std**2)/HA_result_mean)/((HA_melt_std**2)/HA_melt_mean))
melt_attributed_to_LA_error = np.sqrt(((LA_result_std**2)/LA_result_mean)/((LA_melt_std**2)/LA_melt_mean))

# print results
print("% Melt attributed to Hbio = {} +/- {}".format(melt_attributed_to_HA*100, melt_attributed_to_HA_error))
print("% Melt attributed to Lbio = {} +/- {}".format(melt_attributed_to_LA*100, melt_attributed_to_LA_error))



