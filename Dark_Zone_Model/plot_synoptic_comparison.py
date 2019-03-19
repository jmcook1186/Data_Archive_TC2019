"""
Plot 2016 vs 2017 dark ice comparison for Cook surface classification paper

Script based on paper_darkice_repo/plot_annual_duration_basemap.py and plot_annual_cumulative_common_dark.py
"""


import matplotlib.pyplot as plt
from matplotlib import rcParams
import xarray as xr
import numpy as np
import plotmap
import pyproj
import mar_raster
import georaster
from matplotlib import dates
import pandas as pd
from matplotlib import cm
import datetime as dt

rcParams['font.sans-serif'] = 'Arial'
rcParams['axes.unicode_minus'] = False
rcParams['legend.fontsize'] = 6
rcParams['xtick.labelsize'] = 6
rcParams['xtick.direction'] = 'in'
rcParams['xtick.major.pad'] = 3
rcParams['figure.titlesize'] = 6

dark2017 = xr.open_dataset('/scratch/physical_controls/MOD09GA.006.onset.2017.bare60.dark45.JJA.win7.b3.d3.nc')
darkall = xr.open_dataset('/scratch/physical_controls/MOD09GA.006.onset.2000-2016.bare60.dark45.JJA.win7.b3.d3.nc')
onset = xr.concat([darkall.sel(TIME='2016'), dark2017], dim='TIME')



mask_dark = np.flipud(georaster.SingleBandRaster('/scratch/physical_controls/mask_dark_common_area_613m.tif').r)

# June=152
# Also in date_reduction_modis_1d.py - make sure identical in both places!
as_perc = (100. / ((243-152)-onset.bad_dur)) * onset.dark_dur
toplot = as_perc \
	.sel(TIME=slice('2016','2017')) \
	.where(onset.dark_dur > 5) \
	.where(mask_dark == 1)

years = np.arange(2016, 2018)
n = 0

fig_extent = (-52, -48, 65, 70)
lon_0 = -40
ocean_kws = dict(fc='#C6DBEF', ec='none', alpha=1, zorder=199)
land_kws = dict(fc='#F6E8C3', ec='none', alpha=1, zorder=200)
ice_kws = dict(fc='white', ec='none', alpha=1, zorder=201)

# Load in land and ice dataframes, just once
shps_loader = plotmap.Map(extent=fig_extent, lon_0=lon_0)
df_ocean = shps_loader.load_polygons(shp_file='/scratch/L0data/NaturalEarth/ne_50m_ocean/ne_50m_ocean', 
	label='ocean')
df_land = shps_loader.load_polygons(shp_file='/scratch/L0data/NaturalEarth/ne_50m_land/ne_50m_land', 
	label='land')
df_ice = shps_loader.load_polygons(shp_file='/scratch/L0data/NaturalEarth/ne_50m_glaciated_areas/ne_50m_glaciated_areas', 
	label='ice')

marproj = mar_raster.create_proj4(ds_fn='/scratch/MARv3.6.2-7.5km-v2-ERA/ICE.2016.01-08.q13.nc')
data_ll_geo = marproj(as_perc.X.min(), as_perc.Y.min(), inverse=True)
data_ur_geo = marproj(as_perc.X.max(), as_perc.Y.max(), inverse=True)
data_ll = shps_loader.map(data_ll_geo[0], data_ll_geo[1])
data_ur = shps_loader.map(data_ur_geo[0], data_ur_geo[1])
data_extent = (data_ll[0], data_ur[0], data_ll[1], data_ur[1])

shps_loader = None
plt.close()

def facet(fig, ax, data, title_label, label_grid, imshow_kws, de_km_fmt):

	facet_map = plotmap.Map(extent=fig_extent, lon_0=lon_0, fig=fig, ax=ax)

	# Draw basic underlays
	facet_map.plot_polygons(df=df_ocean, plot_kws=ocean_kws)
	facet_map.plot_polygons(df=df_land, plot_kws=land_kws)
	facet_map.plot_polygons(df=df_ice, plot_kws=ice_kws)
	facet_map.map.drawmapboundary(fill_color='#C6DBEF', linewidth=0)

	# Data
	facet_map.im = facet_map.ax.imshow(data, extent=data_extent, zorder=300, **imshow_kws)

	# Facet title
	facet_map.ax.annotate(title_label,fontsize=6, fontweight='bold', xy=(0.05, 0.96), xycoords='axes fraction',
           horizontalalignment='left', verticalalignment='top',zorder=300)

	# Facet extent
	if de_km_fmt is not False:
		facet_map.ax.annotate(de_km_fmt,fontsize=6, xy=(0.96, 0.04), xycoords='axes fraction',
        	   horizontalalignment='right', verticalalignment='bottom',zorder=3000,
        	   bbox=dict(boxstyle='square,pad=0.2', fc='#e3e3e3', ec='none'))
	
	# Ticks/graticules
	if label_grid:
		facet_map.geo_ticks(3, 2, rotate_parallels=True, linewidth=0.5, 
			color='#737373', fontsize=6)
	else:
		facet_map.geo_ticks(3, 2, rotate_parallels=True, linewidth=0.5, 
			color='#737373',
			mlabels=[0,0,0,0], plabels=[0,0,0,0])
	
	for axis in ['top','bottom','left','right']:
		facet_map.ax.spines[axis].set_linewidth(0.5)
		facet_map.ax.spines[axis].set_edgecolor('gray')

	return facet_map



# Construct facet plot, subplot-by-subplot
fig = plt.figure(figsize=(3.1, 4))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(7, 2)


imshow_kws = dict(cmap='Reds', vmin=0, vmax=100, interpolation='none')

# 2016
#ax = plt.subplot(2, 2, 1)
ax = plt.subplot(gs[0:5,0])
de_km = (np.sum(np.where(toplot.sel(TIME='2016') >= 5, 1, 0)) * (614.523 * 613.923)) / 1000000
de_km_fmt = '{:.0f}'.format(de_km) + ' km$^2$'
f = facet(fig, ax, np.flipud(toplot.sel(TIME='2016').squeeze().values), 'A', 
	True, imshow_kws, de_km_fmt)
f.ax.annotate('2016',fontsize=6, fontweight='bold', xy=(0.94, 0.96), xycoords='axes fraction',
       horizontalalignment='right', verticalalignment='top',zorder=3000)

# 2017
#ax = plt.subplot(2, 2, 2)
ax = plt.subplot(gs[0:5,1])
de_km = (np.sum(np.where(toplot.sel(TIME='2017') >= 5, 1, 0)) * (614.523 * 613.923)) / 1000000
de_km_fmt = '{:.0f}'.format(de_km) + ' km$^2$'
f = facet(fig, ax, np.flipud(toplot.sel(TIME='2017').squeeze().values), 'B', 
	True, imshow_kws, de_km_fmt)
f.ax.annotate('2017',fontsize=6, fontweight='bold', xy=(0.94, 0.96), xycoords='axes fraction',
       horizontalalignment='right', verticalalignment='top',zorder=3000)




mar_mask_dark = georaster.SingleBandRaster('/scratch/physical_controls/mask_dark_common_area_7.5km.tif')
mar_mask_dark.r = np.flipud(mar_mask_dark.r)
mar_path = '/scratch/MARv3.8_EIN_7.5km/ICE.201*nc'
x_slice = slice(-586678,-254545)
y_slice = slice(-949047,69833)

## Copied from analyse_snow.py
SHSN2_all = mar_raster.open_mfxr(mar_path,
	dim='TIME', transform_func=lambda ds: ds.SHSN2.sel(X=x_slice, 
		Y=y_slice))
SHSN2_all_1617 = SHSN2_all.sel(TIME=slice('2016','2017'))
snow_above_ice = SHSN2_all.sel(SECTOR1_1=1.0).where(mar_mask_dark.r > 0).where((SHSN2_all['TIME.month'] >= 4) & (SHSN2_all['TIME.month'] < 9)).mean(dim=('X', 'Y'))
snow_above_ice = snow_above_ice.sel(TIME=slice('2016', '2017')).to_pandas()


# Snowline clearing markers (t_B)
bare_doy_med_masked_common = onset.bare.where(mask_dark > 0).median(dim=['X','Y'])
bare_doy_q25_masked_common = onset.bare.where(mask_dark > 0).quantile(0.25, dim=['X','Y'])
bare_doy_q75_masked_common = onset.bare.where(mask_dark > 0).quantile(0.75, dim=['X','Y'])
# Remove the window duration from the date to get the true snowline retreat date
bare_doy_med_masked_common -= 7 ##CHECK
bare_doy_q25_masked_common -= 7 
bare_doy_q75_masked_common -= 7 



## 2016 SNOW
#ax = plt.subplot(2, 2, 3)
ax = plt.subplot(gs[5:7,0])
plt.plot(snow_above_ice['2016'].index, snow_above_ice['2016'], color='#1D91C0', lw=2)
plt.ylim(-0.05, 1.5)
plt.grid('off')
ax.yaxis.tick_left()
plt.yticks([0, 0.5, 1, 1.5], [0, 0.5, 1, 1.5])
ax.tick_params(axis='y', direction='out')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='x', bottom='on', top='off')
plt.xlabel('2016')
ax.annotate('C',fontsize=6, fontweight='bold', xy=(0.04, 0.96), xycoords='axes fraction',
	   horizontalalignment='left', verticalalignment='top',zorder=3000)


## 2017 SNOW
#ax = plt.subplot(2, 2, 4)
ax = plt.subplot(gs[5:7,1])
plt.plot(snow_above_ice['2017'].index, snow_above_ice['2017'], color='#1D91C0', lw=2)
plt.ylim(-0.05, 1.5)
plt.grid('off')
ax.yaxis.tick_left()
plt.yticks([0, 0.5, 1, 1.5], [0, 0.5, 1, 1.5])
ax.tick_params(axis='y', direction='out')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='x', bottom='on', top='off')
plt.yticks([])
ax.spines['left'].set_visible(False)
plt.tick_params(axis='y', left='off')
plt.xlabel('2017')

plt.yticks([])
ax.spines['left'].set_visible(False)
plt.tick_params(axis='y', left='off', right='off')
ax.annotate('D',fontsize=6, fontweight='bold', xy=(0.04, 0.96), xycoords='axes fraction',
	   horizontalalignment='left', verticalalignment='top',zorder=3000)




## 2016 DARK ICE
csum_store = []
#ax = plt.subplot(2, 2, 3)
ax = plt.subplot(gs[5:7,0])
ax2 = ax.twinx()
year = onset.sel(TIME='2016').squeeze().TIME
y = pd.to_datetime(year.values).strftime('%Y')

## Windowed dataset
data = onset.sel(TIME=str(y)).dark.where(mask_dark > 0).where(onset.dark < 240)
data = data.values[~np.isnan(data.values)]
print('%s : %s px' %(y, len(data.flatten())))
# Generate histogram
hist, bin_edges = np.histogram(data.flatten(), range=(152, 239), bins=239-152)
# Convert pixels to area in kms
hist = hist * 0.377
bin_dates = [dt.datetime.strptime('%s %s' %(y, int(b)), '%Y %j') for b in bin_edges]
# Accumulate
csum = np.cumsum(hist)

plt.plot(bin_dates[:-1], csum, '-', color='#CB181D', lw=2, alpha=0.8)

# Save data to csv
csum_pd16 = pd.Series(csum, index=bin_dates[:-1])


clear = int(bare_doy_med_masked_common.sel(TIME=str(y)).values[0])
clear_date = dt.datetime.strptime('%s %s' %(y, clear), '%Y %j')
clearq25 = int(bare_doy_q25_masked_common.sel(TIME=str(y)).values[0])
clearq25_date = dt.datetime.strptime('%s %s' %(y, clearq25), '%Y %j')
clearq75 = int(bare_doy_q75_masked_common.sel(TIME=str(y)).values[0])
clearq75_date = dt.datetime.strptime('%s %s' %(y, clearq75), '%Y %j')

plt.plot(clear_date, 4000, '|', color='gray', markersize=4)
plt.plot([clearq25_date, clearq75_date], [4000, 4000], '-', color='gray', linewidth=0.6)
plt.ylim(-500, 12000)
ax2.xaxis.set_major_locator(dates.MonthLocator())
ax2.xaxis.set_major_formatter(dates.DateFormatter('%m'))   
plt.grid('off')

ax2.yaxis.tick_right()
plt.yticks([0, 4000, 8000, 12000], [0, 4, 8, 12])
ax2.tick_params(axis='y', direction='out')

ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.tick_params(direction='out')
ax.tick_params(direction='out')
plt.tick_params(axis='x', bottom='on', top='off')

plt.yticks([])
ax2.spines['right'].set_visible(False)
plt.tick_params(axis='y', left='off', right='off')



## 2017 DARK ICE
csum_store = []
#ax = plt.subplot(2, 2, 4)
ax = plt.subplot(gs[5:7,1])
ax2 = ax.twinx()
year = onset.sel(TIME='2017').squeeze().TIME
y = pd.to_datetime(year.values).strftime('%Y')

## Windowed dataset
data = onset.sel(TIME=str(y)).dark.where(mask_dark > 0).where(onset.dark < 240)
data = data.values[~np.isnan(data.values)]
print('%s : %s px' %(y, len(data.flatten())))
# Generate histogram
hist, bin_edges = np.histogram(data.flatten(), range=(152, 239), bins=239-152)
# Convert pixels to area in kms
hist = hist * 0.377
bin_dates = [dt.datetime.strptime('%s %s' %(y, int(b)), '%Y %j') for b in bin_edges]
# Accumulate
csum = np.cumsum(hist)

plt.plot(bin_dates[:-1], csum, '-', color='#CB181D', lw=2, alpha=0.8)

# Save data to csv
csum_pd17 = pd.Series(csum, index=bin_dates[:-1])

clear = int(bare_doy_med_masked_common.sel(TIME=str(y)).values[0])
clear_date = dt.datetime.strptime('%s %s' %(y, clear), '%Y %j')
clearq25 = int(bare_doy_q25_masked_common.sel(TIME=str(y)).values[0])
clearq25_date = dt.datetime.strptime('%s %s' %(y, clearq25), '%Y %j')
clearq75 = int(bare_doy_q75_masked_common.sel(TIME=str(y)).values[0])
clearq75_date = dt.datetime.strptime('%s %s' %(y, clearq75), '%Y %j')

plt.plot(clear_date, 4000, '|', color='gray', markersize=4)
plt.plot([clearq25_date, clearq75_date], [4000, 4000], '-', color='gray', linewidth=0.6)
plt.ylim(-500, 12000)
ax2.xaxis.set_major_locator(dates.MonthLocator())
ax2.xaxis.set_major_formatter(dates.DateFormatter('%m'))   
plt.grid('off')

ax2.yaxis.tick_right()
plt.yticks([0, 4000, 8000, 12000], [0, 4, 8, 12])
ax2.tick_params(axis='y', direction='out')

ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.tick_params(direction='out')
ax.tick_params(direction='out')
plt.tick_params(axis='x', bottom='on', top='off')


fig.text(0.51, 0.02, 'Month of Year', ha='center', va='center', color='black') 
fig.text(0.06, 0.21, 'Mean Snow Depth (m)', ha='center', va='center', 
	color='#1D91C0', rotation='vertical') 
fig.text(0.93, 0.21, 'Cumulative Dark Ice Extent \n(x 10$^3$ km$^2$)', ha='center', 
	va='center', color='#CB181D', rotation='vertical') 

# Add colorbar for the duration plots
# left, bottom, width, height
cb_ax = fig.add_axes([0.25, 0.88, 0.5, 0.025])
cbar = plt.colorbar(f.im, cax=cb_ax, orientation='horizontal', 
	ticks=(0, 20, 40, 60, 80, 100), drawedges=False)
cbar.set_label('Dark % of cloud-free JJA observations', fontsize=6)
cbar.outline.set_visible(False)
cbar.ax.xaxis.set_ticks_position('top')
plt.subplots_adjust(hspace=0.4, left=0.15, right=0.82, top=0.83)

#plt.savefig('/home/at15963/Dropbox/work/papers/cook_bioalbedo/darkice_2016_2017.pdf')
#plt.savefig('/home/at15963/Dropbox/work/papers/cook_bioalbedo/darkice_2016_2017.png', dpi=600)



## Export data underlying figure

# Dark ice extent-duration
onset.attrs['proj4'] = '+proj=sterea +lat_0=70.5 +lon_0=-40 +k=1 +datum=WGS84 +units=m'
onset.to_netcdf('/home/at15963/Dropbox/work/papers/cook_bioalbedo/data_outputs/darkice_extent_duration_2016_2017.nc')

# Mean snow depth above ice
snow_above_ice.name = 'Mean snow above ice in common area (m)'
snow_above_ice.to_csv('/home/at15963/Dropbox/work/papers/cook_bioalbedo/data_outputs/snow_above_ice.csv',
	header=True)

# Raw MAR data
mar = mar_raster.open_xr('/scratch/MARv3.8_EIN_7.5km/ICE.2016.01-12.q39.nc')
coords = {'SECTOR1_1':mar.SECTOR1_1, 'TIME':SHSN2_all_1617.TIME, 'Y':mar.Y, 'X':mar.X}
save_snow = xr.Dataset({'LAT':mar.LAT, 'LON':mar.LON, 'MSK':mar.MSK, 'SH':mar.SH, 
	'SHSN2':SHSN2_all_1617.load()}, coords=coords) 
	#'X':mar.X, 'Y':mar.Y, 
save_snow.attrs['institute'] = mar.attrs['institute']
save_snow.attrs['model'] = mar.attrs['model']
save_snow.to_netcdf('/home/at15963/Dropbox/work/papers/cook_bioalbedo/data_outputs/MARv3.8_EIN_7.5km_SHSN2.nc')

csum_pd16.index.name = 'TIME'
csum_pd16.name = 'Cumulative dark ice extent in common area (km^2)'
csum_pd16.to_csv('/home/at15963/Dropbox/work/papers/cook_bioalbedo/data_outputs/darkice_cumulative_2016.csv',
	header=True)
csum_pd17.index.name = 'TIME'
csum_pd17.name = 'Cumulative dark ice extent in common area (km^2)'
csum_pd17.to_csv('/home/at15963/Dropbox/work/papers/cook_bioalbedo/data_outputs/darkice_cumulative_2017.csv',
	header=True)