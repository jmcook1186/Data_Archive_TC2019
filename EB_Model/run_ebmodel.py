#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:39:55 2018

@author: joe
"""

import numpy as np
import ebmodel as ebm

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

	print('daily_total = {}'.format(np.sum(total_list)))






