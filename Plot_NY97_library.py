#!/global/homes/i/indah/.conda/envs/climate_py39/bin/python

from copy import deepcopy
import matplotlib.gridspec as gridspec
from cartopy.util import add_cyclic_point
import xarray as xr
import numpy as np
import matplotlib.pyplot as PP
import Ngl
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmasher as cmr
from scipy.ndimage import gaussian_filter as gf
import pyinterp.backends.xarray
import pyinterp.fill
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
import matplotlib
import matplotlib as mpl
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)
global xc,yc
# xc,yc = -121.8781201, 39.38145027
xc,yc = 237.7779906860763-360, 40.059411677627395

#----------------------------------------------------------------------------------------------------------
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

#----------------------------------------------------------------------------------------------------------
def plot_z_dIVT(data, data2, titles, resolutions):
    source_proj = ccrs.PlateCarree()
    target_proj = ccrs.PlateCarree(central_longitude=-170)

    ivt_levs = np.arange(-300,350,50)
    z_levs = np.linspace(900,1600,8)

    fig = PP.figure(figsize=(10,7), linewidth=1, edgecolor="black")
    spec = gridspec.GridSpec(ncols=22, nrows=3, figure=fig)

    axs00 = fig.add_subplot(spec[0, 0:8], projection=target_proj)
    axs10 = fig.add_subplot(spec[1, 0:8], projection=target_proj)
    axs20 = fig.add_subplot(spec[2, 0:8], projection=target_proj)

    axs01 = fig.add_subplot(spec[0, 8:13], projection=target_proj)
    axs11 = fig.add_subplot(spec[1, 8:13], projection=target_proj)
    axs21 = fig.add_subplot(spec[2, 8:13], projection=target_proj)

    axs02 = fig.add_subplot(spec[0, 13:22], projection=target_proj)
    axs12 = fig.add_subplot(spec[1, 13:22], projection=target_proj)
    axs22 = fig.add_subplot(spec[2, 13:22], projection=target_proj)

    axs = np.array([axs00,axs10,axs20])
    for i,ax,res in zip(np.arange(len(resolutions)),axs,resolutions):
        cf = ax.contourf(data[i].lon, data[i].lat, data[i], levels=ivt_levs, \
                         transform=source_proj, cmap=bwr_cmap, extend="both")
        custimize_ax(ax=ax, proj=source_proj, ylab=titles[i], xlab=None, \
                     extent=[-126.5,-115.5,31.5,43.5], states=True, fontsize=7)

    cb0 = fig.colorbar(cf, ax=axs, orientation="vertical", aspect=50, shrink=0.90)
    cb0.ax.tick_params(labelsize=7, labelrotation=90)
    cb0.ax.set_yticklabels(ivt_levs.astype("int") ,va='center')
    axs[0].set_title(r"IVT [kg m$^{-1}$s$^{-1}$]", loc="right", pad=0.01, \
                     fontdict={"fontsize":8, "fontweight":"normal"})

    axs = np.array([axs01,axs11,axs21])
    for i,ax,res in zip(np.arange(len(resolutions)),axs,resolutions):
        cf = ax.contourf(data2[i].lon, data2[-1].lat, data2[-1].sel(lev=850), levels=z_levs, \
                         transform=source_proj, cmap=mpl.cm.Greys, alpha=0.5)
        c = ax.contour  (data2[-1].lon, data2[-1].lat, data2[-1].sel(lev=850), levels=z_levs, \
                         transform=source_proj, colors="black", alpha=0.8, linewidths=0.75)
        c = ax.contour  (data2[i].lon, data2[i].lat, data2[i].sel(lev=850), levels=z_levs, \
                         transform=source_proj, cmap=mpl.cm.RdYlBu, linewidths=2.5)
        custimize_ax(ax=ax, proj=source_proj, yrlab=titles[i], xlab=None, \
                     extent=[-128.5,-115.5,31.5,43.5], states=True, fontsize=8)

    axs[0].set_title(r"z [m]", loc="right", \
                     fontdict={"fontsize":8, "fontweight":"normal"})

    axs = np.array([axs02,axs12,axs22])
    for i,ax,res in zip(np.arange(len(resolutions)),axs,resolutions):
        cf = ax.contourf(data2[i].lon, data2[-1].lat, gf(data2[-1].sel(lev=850),sigma=2), levels=z_levs, \
                         transform=source_proj, cmap=mpl.cm.Greys, alpha=0.5)
        c = ax.contour  (data2[-1].lon, data2[-1].lat, data2[-1].sel(lev=850), levels=z_levs, \
                         transform=source_proj, colors="black", alpha=0.8, linewidths=0.75)
        c = ax.contour  (data2[i].lon, data2[i].lat, data2[i].sel(lev=850), levels=z_levs, \
                         transform=source_proj, cmap=mpl.cm.RdYlBu, linewidths=2.5)
        custimize_ax(ax=ax, proj=source_proj, extent=[180, 250, 11.5, 63.5], states=True, fontsize=8)

    cb = fig.colorbar(cf, ax=axs, orientation="vertical", aspect=40, shrink=0.9)
    cb.add_lines(c)
    for l in cb.lines:
        l.set_linewidth(5)
    cb.ax.tick_params(labelsize=7, labelrotation=90)
    cb.ax.set_yticklabels(z_levs.astype("int"),va='center')
    axs[0].set_title(r"z [m]", loc="right", \
                     fontdict={"fontsize":8, "fontweight":"normal"})

    return fig

#----------------------------------------------------------------------------------------------------------
def plot_ivt(DataSets, DataSets2, resolutions, ylabels):

    source_proj = ccrs.PlateCarree()
    target_proj = ccrs.PlateCarree(central_longitude=-170)

    ivt_levs = np.linspace(0,1200,7)
    z_levs = np.linspace(900,1600,8)

    fig,axs = PP.subplots(4,1,figsize=(3.5,8), subplot_kw={"projection":target_proj})

    for i,ax,res in zip(np.arange(len(resolutions)),axs,resolutions):

        ds  =  DataSets[i].isel(time=1)
        ds2 = DataSets2[i].isel(time=1)

        cf = ax.contourf(ds.lon, ds.lat, ds.IVT.sel(res=res), levels=ivt_levs, \
                         transform=source_proj, extend="max", cmap=sat_cmap)
        cs = ax.contour(ds2.lon, ds2.lat, ds2.Z.sel(res=res, lev=850).rename({"lon":"x", "lat":"y"}).rio.write_crs("epsg:4326", inplace=True).rio.interpolate_na(), \
                       levels=z_levs, transform=source_proj, \
                       linestyles=":", linewidths=1.5, \
                       colors="black", alpha=0.9999)
        cl = ax.clabel(cs,inline=1,fontsize=6)

        custimize_ax(ax=ax, proj=source_proj, ylab=ylabels[i], xlab=None)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.68])
    cb = fig.colorbar(cf, cax=cbar_ax, orientation="vertical", )
    cb.ax.tick_params(labelsize=10)
    cb.set_label(r"IVT [kg m$^{-1}$s$^{-1}$]", weight='bold', fontsize=10)

    return fig, str(ds.time.values).split(":")[0]

#----------------------------------------------------------------------------------------------------------
def plot_ivt_times(DataSets, DataSets2, resolutions, ylabels, start_dates):
    
    source_proj = ccrs.PlateCarree()
    target_proj = ccrs.PlateCarree(central_longitude=-170)

    ivt_levs = np.linspace(0,1200,7)
    z_levs = np.linspace(800,1600,9)

    fig,axs = PP.subplots(4,4,figsize=(8,5), subplot_kw={"projection":target_proj})

    for j in range(4):
        for i,res in zip(np.arange(len(resolutions)),resolutions):

            ds  =  DataSets[i].isel(time=j)
            ds2 = DataSets2[i].isel(time=j)

            ax = axs[i,j]
            ds = DataSets[i].isel(time=j)

            cf = ax.contourf(ds.lon, ds.lat, ds.IVT.sel(res=res), levels=ivt_levs, \
                             transform=source_proj, extend="max", cmap=sat_cmap)

            cs = ax.contour(ds2.lon, ds2.lat, ds2.Z.sel(res=res, lev=850).rename({"lon":"x", "lat":"y"}).rio.write_crs("epsg:4326", inplace=True).rio.interpolate_na(), \
                       levels=z_levs, transform=source_proj, \
                       linestyles=":", linewidths=1.5, \
                       colors="black", alpha=0.90)
            cl = ax.clabel(cs,inline=1,fontsize=5)

            if j==0:
                custimize_ax(ax=ax, proj=source_proj, ylab=ylabels[i], xlab=None, fontsize=6)
            else:
                custimize_ax(ax=ax, proj=source_proj,fontsize=6)
            if i==0: ax.set_title(start_dates[j], fontsize=6)

    fig.subplots_adjust(right=0.97)
    cbar_ax = fig.add_axes([0.99, 0.15, 0.012, 0.68])
    cb = fig.colorbar(cf, cax=cbar_ax, orientation="vertical", )
    cb.ax.tick_params(labelsize=10)
    cb.set_label(r"IVT [kg m$^{-1}$s$^{-1}$]", weight='bold', fontsize=10)

    return fig

#----------------------------------------------------------------------------------------------------------
def smooth_cmap(rgb, method="cubic", s=2):

    from scipy.interpolate import splev, splrep
    from scipy.interpolate import CubicSpline
    import numpy as np
    
    new = []
    
    if method=="smooth":
        for i in range(3):
            y = rgb_colours[:,i]
            x = np.arange(len(y))
            xp = np.linspace(x[0], x[-1], 266)

            spl = splrep(x, y, s=2)
            yp = splev(xp, spl)[5:-5]
            yp = np.clip(yp,0,1)
            new.append(yp)
    elif method=="cubic":
        for i in range(3):
            y = rgb_colours[:,i]
            x = np.arange(len(y))
            xp = np.linspace(x[0], x[-1], 266)

            cs = CubicSpline(x, y)
            yp = cs(xp)[5:-5]
            yp = np.clip(yp,0,1)
            new.append(yp)
    return np.vstack(new).T

#----------------------------------------------------------------------------------------------------------
img = Image.open('./Satellite.png')
data = img.load()

# Loop through pixels and extract rgb value
rgb_colours = []
for i in range(img.size[1]):
    rgb = [x/255 for x in data[0, i]]  # scale values 0-1
    rgb_colours.append(rgb)

rgb_colours = np.array(rgb_colours)[::-1,:4]
rgb_colours = smooth_cmap(rgb_colours, s=3, method="smooth")
sat_cmap = mpl.colors.ListedColormap(rgb_colours, name="satellite")


#----------------------------------------------------------------------------------------------------------
img = Image.open('./BW_BWR.png')
data = img.load()

# Loop through pixels and extract rgb value
rgb_colours = []
for i in range(img.size[1]):
    rgb = [x/255 for x in data[0, i]]  # scale values 0-1
    rgb_colours.append(rgb)

rgb_colours = np.array(rgb_colours)[::-1,:4]
rgb_colours = smooth_cmap(rgb_colours, s=3, method="smooth")
bwr_cmap = mpl.colors.ListedColormap(rgb_colours, name="bw_bwr")

#----------------------------------------------------------------------------------------------------------
def sigma_to_pressure(ds, var=None):
    """ Convert hybrid to pressure
        single time
    """
    # attrs = ds[var].attrs
    attrs = ds[var].attrs

    #  Do the interpolation.
    intyp = 1                             # 1=linear, 2=log, 3=log-log
    kxtrp = False                         # True=extrapolate (when the output pressure level is outside of the range of psrf)
    lev = np.array([1000,975,950,925,900,875,850,825,800,775,750,700,650,600,550,\
                    500,450,400,350,300,250,225,200,\
                    175,150,125,100,70,50,30,20,10,7,5,3,2,1])#[::-1]

    # hyam =      ds["hyam"].values[0]
    # hybm =      ds["hybm"].values[0]
    hyam =      ds["hyai"].values[0]
    hybm =      ds["hybi"].values[0]
    psrf =      ds["PS"].values
    P0mb =      1000
    lats =      ds["lat"].values
    lons =      ds["lon"].values

    varnew = Ngl.vinth2p(ds[var].values,hyam,hybm,lev,psrf,intyp,P0mb,1,kxtrp)
    varnew[varnew==1e30] = np.NaN
    
    newds = ds.drop_dims(["lev","ilev","lat","lon"])
    newds = newds.assign_coords({"lev":lev, "lat":lats, "lon":lons})
    dims = newds.dims
    coords = newds.coords

    # Create new xarray Dataset
    darray_p = xr.DataArray(varnew[:], \
                          dims = dims, \
                          coords = coords, \
                          name = var, \
                          attrs = attrs, \
                         )
    return darray_p.to_dataset()

#----------------------------------------------------------------------------------------------------------
def custimize_ax(ax=None, proj=None, ylab=None, xlab=None, yrlab=None, fontsize=8, extent=[-200,-100,10,65],\
                 states=False):
    
    ax.text(-0.01, 0.5, ylab, va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=fontsize)
    ax.text(0.5, -0.1, xlab, va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=fontsize)
    ax.text(1.15, 0.5, yrlab, va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=fontsize)
    ax.coastlines()
    ax.set_extent(extent, crs=proj)
    if states:
        ax.add_feature(cfeature.STATES,linewidth=0.5,edgecolor="black")
        ax.add_feature(cfeature.BORDERS,linewidth=0.5,edgecolor="black")
    
    return

#----------------------------------------------------------------------------------------------------------
def sample_sierra_jet(ds,res):

    topo = xr.open_dataset("data/topo_wus30x{}.nc".format(res)).sel(lon=slice(234-360, 244-360),lat=slice(34, 42))
    topo = topo.assign_coords(lon=topo.lon+360)
    topo = topo.z
    topo.values[np.where(topo<0)]=0.
    topo = topo.to_dataset()
    ds = ds.sel(lon=slice(234, 244),lat=slice(34, 42))
    
    if res>=8:
        U = pyinterp.backends.xarray.Grid3D(ds.U, geodetic=False)
        U = pyinterp.fill.gauss_seidel(U)[1]
        U = np.rollaxis(np.rollaxis(U, axis=1),axis=-1)
        V = pyinterp.backends.xarray.Grid3D(ds.V, geodetic=False)
        V = pyinterp.fill.gauss_seidel(V)[1]
        V = np.rollaxis(np.rollaxis(V, axis=1),axis=-1)
        ds.U.values = U
        ds.V.values = V
    
    dp = 1/res
    N = 40

    theta=0.6333548955340397
    x0 = xc - 3.0*np.cos(theta)
    x1 = xc + 2.0*np.cos(theta)
    y0 = yc - 3.0*np.sin(theta)
    y1 = yc + 2.0*np.sin(theta)
    dx,dy = y1-y0,x1-x0
    r = dy/dx
    NN = 1/dp*((x1-x0)**2+(y1-y0)**2)**0.5
    NN = int(np.floor(NN))
    
    # ACROSS
    x = xr.DataArray(np.linspace(x0, x1, NN)+360, dims="topo")
    y = xr.DataArray(np.linspace(y0, y1, NN), dims="topo")
    
    dx = np.diff(x.values)[0]
    dy = np.diff(y.values)[0]
    nleft = np.where(x<=xc+360)[0][-1]+1
    x0,x1 = xc-dx*nleft,xc+dx*(NN-nleft-1)
    y0,y1 = yc-dy*nleft,yc+dy*(NN-nleft-1)
    x = xr.DataArray(np.linspace(x0, x1, NN)+360, dims="topo")
    y = xr.DataArray(np.linspace(y0, y1, NN), dims="topo")
    
    topo_across = topo.sel(lat=y, lon=x, method="nearest")
    topo_across = topo_across.drop_vars(["lon","lat"])
    topo_across = topo_across.assign(xt=x, yt=y)
    topo_across =  1e5*np.exp(-1*topo_across.z/8000)
    topo_across.name = "H"
    xacross = np.linspace(x0, x1, N)+360
    yacross = np.linspace(y0, y1, N)
    lat_across = xr.DataArray(yacross, dims="points")
    lon_across = xr.DataArray(xacross, dims="points")
    across =  ds.sel(lat=lat_across, lon=lon_across, method="nearest")
    across = across.assign(lon=lon_across, lat=lat_across)
    across = across.assign(H=topo_across)
    across = across.assign(xt=x)    
    across = across.assign(yt=y)
    
    # ALONG
    x0 = xc - 4.5*np.cos(theta+np.pi/2)
    x1 = xc + 2.0*np.cos(theta+np.pi/2)
    y0 = yc - 4.5*np.sin(theta+np.pi/2)
    y1 = yc + 2.0*np.sin(theta+np.pi/2)

    x = xr.DataArray(np.linspace(x0, x1, NN)+360, dims="topo")
    y = xr.DataArray(np.linspace(y0, y1, NN), dims="topo")
        
    dx = np.diff(x.values)[0]
    dy = np.diff(y.values)[0]
    nleft = np.where(x>=xc+360)[0][-1]+1
    x0,x1 = xc-dx*nleft,xc+dx*(NN-nleft-1)
    y0,y1 = yc-dy*nleft,yc+dy*(NN-nleft-1)
    x = xr.DataArray(np.linspace(x0, x1, NN)+360, dims="topo")
    y = xr.DataArray(np.linspace(y0, y1, NN), dims="topo")
    
    topo_along = topo.sel(lat=y, lon=x, method="nearest")
    topo_along = topo_along.drop_vars(["lon","lat"])
    topo_along = topo_along.assign(xt=x, yt=y)
    topo_along  =  1e5*np.exp( -1*topo_along.z/8000)
    topo_along.name = "H"    
    
    xalong = np.linspace(x0, x1, N)+360
    yalong = np.linspace(y0, y1, N)
    lat_along = xr.DataArray(yalong, dims="points")
    lon_along = xr.DataArray(xalong, dims="points")
    along =  ds.sel(lat=lat_along, lon=lon_along, method="nearest")
    along = along.assign(lon=lon_along, lat=lat_along)
    along = along.assign(xt=x)    
    along = along.assign(yt=y)    
    along = along.assign(H=topo_along)

    along.U.values  = -1*(along.U.values*np.cos(theta) - along.V.values*np.sin(theta)                   )  
    along.V.values  =  1*(along.U.values*np.sin(theta) + along.V.values*np.cos(theta)                   ) 
    across.U.values =  1*(across.U.values*np.cos(-theta+np.pi/2) - across.V.values*np.sin(-theta+np.pi/2) )
    across.V.values = -1*(across.U.values*np.sin(-theta+np.pi/2) + across.V.values*np.cos(-theta+np.pi/2) )
    
    newlev = np.arange(600,1000,10)
    along = along.interp(lev=newlev, method="cubic")
    across = across.interp(lev=newlev, method="cubic")
    
    return along,across

#----------------------------------------------------------------------------------------------------------
def create_AA(ERA5_p, E3SM_p, time=None):

    Along,Across=[],[]
    along,across = along,across=sample_sierra_jet(ERA5_p.sel(res="ERA5").isel(time=time).sel(lev=slice(600,1000)), res=4)
    Along.append(along)
    Across.append(across)
    along,across = along,across=sample_sierra_jet(E3SM_p.sel(res="ne0wus30x8").isel(time=time).sel(lev=slice(600,1000)), res=8)
    Along.append(along)
    Across.append(across)
    along,across = along,across=sample_sierra_jet(E3SM_p.sel(res="ne0wus30x16").isel(time=time).sel(lev=slice(600,1000)), res=16)
    Along.append(along)
    Across.append(across)
    along,across = along,across=sample_sierra_jet(E3SM_p.sel(res="ne0wus30x32").isel(time=time).sel(lev=slice(600,1000)), res=32)
    Along.append(along)
    Across.append(across)
    Resolutions = ["ERA5 ", "RRM-E3SM (14 km)", "RRM-E3SM (7 km)", "RRM-E3SM (3.5 km)"]
    
    return Along,Across,Resolutions

#----------------------------------------------------------------------------------------------------------
def plot_profile(Along, Across, Resolutions, time=None, max=False, fs=10):
    
    j1 = 29
    # colors = ("#C1CDCD", "#35B2EE", "#2C9ACD", "#1B688B")
    colors = ("#C1CDCD", "#B4DAE9", "#03FA08", "#1B688B")
    xlims =  (         [-13,26],            [-3,38]  )
    xticks = (np.arange(-10,30,5), np.arange(-0,40,5))
    
    fig,axs = PP.subplots(1,2,figsize=(5.0, 3.0))

    LW = 2
    lw = 0.25
    #===========================================================================================================================
    #Along
    xlabels = [r"C-D wind [m/s]", r"A-B wind [m/s]", r"u [m/s]", r"v [m/s] (S-N)"]
    ax = axs[0]
    for i in range(4):
        # test = np.abs(Along[i].U.sel(lev=950))
        # j1 = test.argmax().values
        ax.plot(Along[i].U.isel(points=j1), Along[i].lev, color=colors[i], lw=LW, label="{}".format(Resolutions[i]), alpha=0.9)    
    ax.text(0.96, 1.05, '{}'.format(str(Along[i].time.values).split(":")[0]), \
            transform=ax.transAxes, horizontalalignment='center', size=fs)
    ax = axs[1]
    for i in range(4):
        ax.plot(Along[i].V.isel(points=j1), Along[i].lev, color=colors[i], lw=LW, label="{}".format(Resolutions[i]))
    #===========================================================================================================================
    #Plot parameters
    for i,ax in enumerate(axs):
        ax.invert_yaxis()
        ax.tick_params(labelsize=fs)
        ax.set_xlabel(xlabels[i], fontdict={"weight":"normal", "fontsize":fs})
        ax.set_xticks(xticks[i])
        ax.set_xticklabels(xticks[i], fontdict={"weight":"normal", "fontsize":fs})
        ax.set_xlim(xlims[i])
        if i==0:    
            ax.text(-0.22, 0.5, r"p [hPa]", fontdict={"weight":"normal", "fontsize":fs}, \
                    ha="center", va="center", transform=ax.transAxes, rotation=90)
        if i==1:
            ax.legend(fontsize=fs-1, loc='upper right', framealpha=0.9, \
                      bbox_to_anchor=(0.20, 0.99), handlelength=0.75, handleheight=1.5)
            ax.yaxis.tick_right()
            ax.set_yticklabels([])
    fig.show()
    
    return fig

#----------------------------------------------------------------------------------------------------------
def plot_transect_location(Along, Across, Resolutions, time=None, max=False):
    
    if max:
        j1,j2 = 33,26
    else:
        j1,j2 = 27,24
        
    lats = Along[0].lat.values
    lats,lats=np.meshgrid(lats,lats)
    
    colors = sat_cmap(np.linspace(0,1,40))
    
    fig,ax = PP.subplots()
    CS3 = ax.contourf(lats, cmap=sat_cmap)
    # colors = bwr_cmap(np.linspace(0,1,40))
    # CS3 = PP.contourf(lats, cmap=bwr_cmap)
    PP.close(fig)
    
    xlims =           [-18,33]  
    xticks = np.arange(-15,35,10)    
    xlabels = r"u [m/s] (S-N)"
    fig,axs = PP.subplots(1,4,figsize=(8.5, 3.0))

    #===========================================================================================================================
    for i,ax in enumerate(axs):
        for j in range(len(Along[i].points)):
            ax.plot(Along[i].U.isel(points=j), Along[i].lev, color=colors[j], label=Along[i].lat.isel(points=j).values, lw=0.85)
            # ax.plot(((Along[i].V**2+Along[i].U**2)**0.5).isel(points=j), Along[i].lev, color=colors[j], label=along.lat.isel(points=j).values)
    #===========================================================================================================================
    #Plot parameters
    for i,ax in enumerate(axs):
        ax.invert_yaxis()
        ax.tick_params(labelsize=7)
        ax.text(0, 1.03, Resolutions[i], fontdict={"weight":"normal", "fontsize":"7"}, \
                    ha="left", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabels, fontdict={"weight":"normal", "fontsize":"8"})
        ax.set_xticks(xticks)
        ax.set_xlim(xlims)
        # ax.set_ylim(1000,600)
        if i==0:    
            ax.text(-0.26, 0.5, r"p [hPa]", fontdict={"weight":"normal", "fontsize":"7"}, \
                    ha="center", va="center", transform=ax.transAxes, rotation=90)
        else:
            ax.set_yticklabels([])
    # fig.tight_layout()
    cbar_ax = fig.add_axes([0.915, 0.15, 0.013, 0.7])
    cb = fig.colorbar(CS3, cax=cbar_ax)
    cb.set_label(r"u [m/s] (S-N)", fontsize=7.5)
    fig.show()
    
    return fig

#----------------------------------------------------------------------------------------------------------
def plot_transect(along, across,res=None, time=0, lev=950):
        
    f = xr.open_dataset("data/topo_wus30x{}.nc".format(res)).sel(lon=slice(232-360, 245-360), lat=slice(32, 44))
    
    if res==4:  RES,tit = "ERA5","ERA5"
    if res==8:  RES,tit = "ne0wus30x8","RRM-E3SM(14 km)"
    if res==16: RES,tit = "ne0wus30x16","RRM-E3SM(7 km)"
    if res==32: RES,tit = "ne0wus30x32","RRM-E3SM(3.5 km)"
    
    if res==4:
        ds = xr.open_dataset("data/era5_p.nc").isel(time=time).sel(lev=lev,lon=slice(228, 255), lat=slice(28, 48), res=RES)
    else:        
        ds = xr.open_dataset("data/e3sm_p.nc").isel(time=time).sel(lev=lev,lon=slice(228, 255), lat=slice(28, 48), res=RES)
    u = ds.U.values
    v = ds.V.values
    lon = ds.lon.values
    lat = ds.lat.values
    
    source_proj = ccrs.PlateCarree()
    target_proj = ccrs.PlateCarree(central_longitude=-170)
    
    cmap = cmr.get_sub_cmap('gist_earth', 0.05, 1.00)
    cm0 = cmr.get_sub_cmap('RdYlBu_r', 0.0, 1.0)
    cm1 = cmr.get_sub_cmap('RdYlBu_r', 0.0, 1.0)
    # cm1 = sat
    # cm0 = sat

    u_levs0 = np.arange(0,30,5)
    v_levs0 = np.arange(-10,8,2)
    norm0 = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=6)

    u_levs1 = np.arange(-10,25,5)
    v_levs1 = np.arange(0,24,4)
    norm1 = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=25)
    
    fig = PP.figure(figsize=(10,2.2))
    axs = [fig.add_subplot(1,3, 1),  fig.add_subplot(1,3, 2), fig.add_subplot(1,3, 3, projection=target_proj)]
    ax0,ax1,ax2 = axs
    
    ax=ax0
    cf = ax.contourf(across.lon, across.lev, across.V, levels=v_levs0, cmap=cm0, norm=norm0, extend="both")
    ax.fill_between(across.xt, y1=np.ones_like(across.xt)*1000, y2=across.H/100, color="black", zorder=101)
    # cs = ax.contour(across.lon, across.lev, across.V, levels=v_levs0, colors="black", linewidths=0.8)
    cs = ax.contour(across.lon, across.lev, -across.U, levels=u_levs0, colors="black", linewidths=0.8)
    cl = ax.clabel(cs,inline=1, fontsize=8)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r"v [m/s] (S-N)", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    ax.plot(np.ones_like(across.lev)*xc+360, across.lev, color="white", linestyle=":", alpha=0.65)
    ax.plot(across.lon, np.ones_like(across.lon)*950, color="white", linestyle=":", alpha=0.65)
    ax.set_ylabel("p [hPa]", labelpad=1.5, fontdict={"weight":"normal", "size":7})    
    ax.text(-0.25, 0.5, tit, fontsize=8, rotation="vertical", va='center', transform=ax.transAxes)
    ax.tick_params(axis='y', rotation=90)
    ax.set_xlabel("Lon", fontdict={"weight":"normal", "size":8})
    ax.set_title("Sierra-perpendicular", fontsize=8, loc="left")
    
    ax=ax1
    cf = ax.contourf(along.lat,along.lev, along.U, norm=norm1, cmap=cm1, extend="both", levels=u_levs1)#, levels=u_levs0, cmap=cm0, extend=None), levels=u_levs1, cmap=cm1, extend=None)
    ax.fill_between(along.yt, y1=np.ones_like(along.yt)*1000, y2=along.H/100, color="black", zorder=101)
    cs = ax.contour(along.lat, along.lev, along.V, levels=v_levs1, colors="black", linewidths=0.8)
    cl = ax.clabel(cs, inline=1, fontsize=8)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r"u [m/s] (S-N)", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    ax.plot(np.ones_like(along.lev)*yc, along.lev, color="white", linestyle=":", alpha=0.65)
    ax.plot(along.lat, np.ones_like(along.lat)*950, color="white", linestyle=":", alpha=0.65)
    ax.set_xlabel("Lat", fontdict={"weight":"normal", "size":8})
    ax.set_title("Sierra-parallel", fontsize=8, loc="left")
    ax.set_yticklabels([])
    ax.set_ylabel(None)
    
    for ax in (ax0,ax1):
        ax.tick_params(labelsize=7)
        ax.set_ylim(600,1000)
        ax.invert_yaxis()

    levels=np.arange(0,3750,50)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    ax=ax2
    xskip,yskip=4,4
    lon_along,lat_along,lon_across,lat_across = along.lon,along.lat,across.lon,across.lat
    LON,LAT = np.meshgrid(lon,lat)
    
    x,y = LON[::xskip,::yskip],LAT[::xskip,::yskip]
    u,v = u[::xskip,::yskip],v[::xskip,::yskip]
    
    cf = ax.pcolormesh(f.lon, f.lat, f.z, transform=source_proj, cmap=cmap, norm=norm)#, vmax=3300)#, levels=np.linspace(0,3500,10), )
    cb = fig.colorbar(cf, ax=ax)
    cb.ax.tick_params(labelsize=8)
    # ax.text(0.65, 1.04, r"Topography [m]", fontsize=7, transform=ax.transAxes)
    cb.set_label(r"Topography [m]", fontsize=8)
    ax.coastlines(edgecolor="white")
    ax.add_feature(cfeature.STATES)
    ax.set_extent([232, 244, 34, 42])
    ax.plot( lon_along,  lat_along, transform=source_proj, lw=1, color="cyan")
    ax.plot(lon_across, lat_across, transform=source_proj, lw=1, color="cyan")
    q = ax.quiver(x, y, u, v, transform=source_proj, color="white", angles="uv", scale=250)
    qk = ax.quiverkey(q, 0.50, 1.06, 10, r'$\vec{u}_{950}$ [10 m s$^{-1}$]',
                      labelpos='W', transform=ccrs.PlateCarree(),
                      color='k', fontproperties={"size":8})
    fig.tight_layout()
    
    return fig

#----------------------------------------------------------------------------------------------------------
def plot_transect_inFig(along, across, J, fig, res=None, time=0, lev=950, fs=10, qkey=False):
        
    f = xr.open_dataset("data/topo_wus30x{}.nc".format(res)).sel(lon=slice(232-360, 245-360), lat=slice(32, 44))
    
    if res==4:  RES,tit = "ERA5","ERA5"
    if res==8:  RES,tit = "ne0wus30x8","RRM-E3SM(14 km)"
    if res==16: RES,tit = "ne0wus30x16","RRM-E3SM(7 km)"
    if res==32: RES,tit = "ne0wus30x32","RRM-E3SM(3.5 km)"
    
    if res==4:
        ds = xr.open_dataset("data/era5_p.nc").isel(time=time).sel(lev=lev,lon=slice(228, 255), lat=slice(28, 48), res=RES)
    else:        
        ds = xr.open_dataset("data/e3sm_p.nc").isel(time=time).sel(lev=lev,lon=slice(228, 255), lat=slice(28, 48), res=RES)
    u = ds.U.values
    v = ds.V.values
    lon = ds.lon.values
    lat = ds.lat.values
    
    source_proj = ccrs.PlateCarree()
    target_proj = ccrs.PlateCarree(central_longitude=-170)
    
    cmap = cmr.get_sub_cmap('gist_earth', 0.05, 1.00)
    cm0 = cmr.get_sub_cmap('RdYlBu_r', 0.0, 1.0)
    cm1 = cmr.get_sub_cmap('RdYlBu_r', 0.0, 1.0)

    u_levs0 = np.arange(0,30,5)
    v_levs0 = np.arange(-10,8,2)
    norm0 = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=6)

    u_levs1 = np.arange(-10,25,5)
    v_levs1 = np.arange(0,24,4)
    norm1 = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=25)
    
    axs = [fig.add_subplot(4,3, int(3*J + 1)),  fig.add_subplot(4,3, int(3*J + 2)), fig.add_subplot(4,3, int(3*J + 3), projection=target_proj)]
    ax0,ax1,ax2 = axs
    
    ax=ax0
    cf = ax.contourf(across.lon, across.lev, across.V, levels=v_levs0, cmap=cm0, norm=norm0, extend="both")
    ax.fill_between(across.xt, y1=np.ones_like(across.xt)*1000, y2=across.H/100, color="black", zorder=98)
    # cs = ax.contour(across.lon, across.lev, across.V, levels=v_levs0, colors="black", linewidths=0.8)
    cs = ax.contour(across.lon, across.lev, -across.U, levels=u_levs0, colors="black", linewidths=0.8)
    cl = ax.clabel(cs,inline=1, fontsize=fs)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r"C-D wind [m/s]", fontsize=fs)
    cb.ax.tick_params(labelsize=fs)
    ax.plot(np.ones_like(across.lev)*xc+360, across.lev, color="white", linestyle=":", alpha=0.75, zorder=99)
    ax.plot(across.lon, np.ones_like(across.lon)*950, color="white", linestyle=":", alpha=0.75, zorder=99)
    ax.set_ylabel("p [hPa]", labelpad=1.5, fontdict={"weight":"normal", "size":fs})    
    ax.text(-0.25, 0.5, tit, fontsize=fs, rotation="vertical", va='center', transform=ax.transAxes)
    ax.tick_params(axis='y', rotation=90)
    
    ax=ax1
    cf = ax.contourf(along.lat,along.lev, along.U, norm=norm1, cmap=cm1, extend="both", levels=u_levs1)#, levels=u_levs0, cmap=cm0, extend=None), levels=u_levs1, cmap=cm1, extend=None)
    ax.fill_between(along.yt, y1=np.ones_like(along.yt)*1000, y2=along.H/100, color="black", zorder=98)
    cs = ax.contour(along.lat, along.lev, along.V, levels=v_levs1, colors="black", linewidths=0.8)
    cl = ax.clabel(cs, inline=1, fontsize=fs)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r"A-B wind [m/s]", fontsize=fs)
    cb.ax.tick_params(labelsize=fs)
    ax.plot(np.ones_like(along.lev)*yc, along.lev, color="white", linestyle=":", alpha=0.75, zorder=99)
    ax.plot(along.lat, np.ones_like(along.lat)*950, color="white", linestyle=":", alpha=0.75, zorder=99)
    ax.set_yticklabels([])
    ax.set_ylabel(None)
    
    for ax in (ax0,ax1):
        ax.tick_params(labelsize=fs)
        ax.set_ylim(600,1000)
        ax.invert_yaxis()

    levels=np.arange(0,3750,50)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    ax=ax2
    xskip,yskip=4,4
    lon_along,lat_along,lon_across,lat_across = along.lon,along.lat,across.lon,across.lat
    LON,LAT = np.meshgrid(lon,lat)
    
    x,y = LON[::xskip,::yskip],LAT[::xskip,::yskip]
    u,v = u[::xskip,::yskip],v[::xskip,::yskip]
    
    cf = ax.pcolormesh(f.lon, f.lat, f.z, transform=source_proj, cmap=cmap, norm=norm)#, vmax=3300)#, levels=np.linspace(0,3500,10), )
    
    cb = fig.colorbar(cf, ax=ax)
    cb.ax.tick_params(labelsize=fs)
    cb.set_label(r"Topography [m]", fontsize=fs)
    ax.coastlines(edgecolor="white")
    ax.add_feature(cfeature.STATES)
    
    ax.set_extent([232, 244, 34, 42])
    ax.set_yticks([34, 36, 38, 40, 42], crs=ccrs.PlateCarree())
    ax.set_xticks(np.array([234, 236, 238, 240, 242]), crs=ccrs.PlateCarree())

    ax.plot( lon_along,  lat_along, transform=source_proj, lw=1.5, color="cyan")
    ax.plot(lon_across, lat_across, transform=source_proj, lw=1.5, color="cyan")
    ax.text( lon_across[0]-0.5,   lat_across[0]-0.5,  "A", transform=source_proj, size=fs, weight="bold", color="cyan")
    ax.text( lon_across[-1],  lat_across[-1], "B", transform=source_proj, size=fs, weight="bold", color="cyan")
    ax.text( lon_along[0],   lat_along[0]-0.5,  "C", transform=source_proj, size=fs, weight="bold", color="cyan")
    ax.text( lon_along[-1]-0.5,  lat_along[-1]+0.1, "D", transform=source_proj, size=fs, weight="bold", color="cyan")
    q = ax.quiver(x, y, u, v, transform=source_proj, color="white", angles="uv", scale=250)
    
    if qkey:
        qk = ax.quiverkey(q, 0.50, 1.06, 10, r'$\vec{u}_{950}$ [10 m s$^{-1}$]',
                          labelpos='W', transform=ccrs.PlateCarree(),
                          color='k', fontproperties={"size":fs})
    return axs

#----------------------------------------------------------------------------------------------------------
def plot_along_across_transect(ERA5_p, E3SM_p, time=0, fs=10, figsize=(14,9)):
    
    along0,across0 = sample_sierra_jet(ERA5_p.sel(res="ERA5").isel(time=time).sel(lev=slice(600,1000)), res=4)
    along1,across1 = sample_sierra_jet(E3SM_p.sel(res="ne0wus30x8").isel(time=time).sel(lev=slice(600,1000)), res=8)
    along2,across2 = sample_sierra_jet(E3SM_p.sel(res="ne0wus30x16").isel(time=time).sel(lev=slice(600,1000)), res=16)
    along3,across3 = sample_sierra_jet(E3SM_p.sel(res="ne0wus30x32").isel(time=time).sel(lev=slice(600,1000)), res=32)
    
    fig = PP.figure(figsize=figsize)
    
    axs0 = plot_transect_inFig(along0, across0, 0, fig, res=4,  time=time, fs=fs, qkey=True)
    axs1 = plot_transect_inFig(along1, across1, 1, fig, res=8,  time=time, fs=fs)
    axs2 = plot_transect_inFig(along2, across2, 2, fig, res=16, time=time, fs=fs)
    axs3 = plot_transect_inFig(along3, across3, 3, fig, res=32, time=time, fs=fs)
        
    for ax in [axs0[0],axs1[0],axs2[0],axs3[0]]:
        ax.set_yticks([900,800,700,600])    
        
    ax = axs0[0]
    ax.set_title("Sierra-perpendicular", fontsize=fs, loc="center")

    ax = axs0[1]
    ax.set_title("Sierra-parallel", fontsize=fs, loc="center")
    
    ax = axs3[-1]
    ax.set_xticks(np.array([234, 238, 242]), crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.array([234, 238, 242]))
    fig.tight_layout()
    
    for ax in (axs0[-1], axs1[-1], axs2[-1], axs3[-1]): 
        lon_formatter = LongitudeFormatter(degree_symbol=" ")
        lat_formatter = LatitudeFormatter(degree_symbol=" ")
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        
    for ax in np.concatenate([axs0,axs1,axs2]):
        ax.set_xlabel("")
        ax.set_xticklabels([])
        
    for ax in (axs0[0], axs1[0], axs2[0], axs3[0],):
        ax.set_title("A", fontweight="bold", fontsize=fs, loc="left", y=0.975)
        ax.set_title("B", fontweight="bold", fontsize=fs, loc="right", y=0.975)
        ax.set_xlabel("")
    ax = axs3[0]
    lon_formatter = LongitudeFormatter(degree_symbol=" ")
    ax.xaxis.set_major_formatter(lon_formatter)
        
    for ax in (axs0[1], axs1[1], axs2[1], axs3[1]): 
        ax.set_xlabel("")
        ax.set_title("C", fontweight="bold", fontsize=fs, loc="left", y=0.975)
        ax.set_title("D", fontweight="bold", fontsize=fs, loc="right", y=0.975)
    ax = axs3[1]
    ax.set_xticks([37,38,39,40,41], crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter(degree_symbol=" ")
    ax.xaxis.set_major_formatter(lat_formatter)    
        
        
    return fig