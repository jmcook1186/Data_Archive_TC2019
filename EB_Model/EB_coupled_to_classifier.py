import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ebmodel as ebm
import multiprocessing as mp
import ebmodel as ebm

melt_list = []
n_cpus = mp.cpu_count()
ds = xr.open_dataset('/home/joe/Code/IceSurfClassifiers/Sentinel_Outputs/2017_Outputs/T22WEV_Classification_and_Albedo_Data_2017.nc', chunks = {'x':1000, 'y':1000})
lenx,leny = (200,200)#ds.albedo.shape

albedo = np.ravel(np.array(ds.albedo.values[3000:3200,3000:3200]))
albedo_chunks = np.array_split(albedo,n_cpus)

##############################################################################
## Input Data, as per first row of Brock and Arnold (2000) spreadsheet
##############################################################################

def runit(alb):

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
    time = 1200
    inswrad = 571
    avp = 900
    airtemp = 5.612
    windspd = 3.531

    SWR,LWR,SHF,LHF = ebm.calculate_seb(lat, lon, lon_ref, day, time, summertime, slope, aspect, elevation,
                                        met_elevation, lapse, inswrad, avp, airtemp, windspd, albedo, roughness)

    sw_melt, lw_melt, shf_melt, lhf_melt, total = ebm.calculate_melt(
        SWR,LWR,SHF,LHF, windspd, airtemp)

    return total


with mp.Pool(processes=n_cpus) as pool:

    # starts the sub-processes without blocking
    # pass the chunk to each worker process
    result = pool.map(runit,albedo)

result = np.reshape(np.array(result),[lenx,leny])

