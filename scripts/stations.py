import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths


def converter(geoCoord: str):
    delimiter = geoCoord.find("°")
    if delimiter == -1:
        delimiter = geoCoord.find("о")
    if delimiter == -1:
        delimiter = geoCoord.find("o")
    return float(geoCoord[:delimiter]) + float(geoCoord[delimiter + 1:delimiter + 3]) / 60


def loadStations() -> pd.DataFrame:
    stations = pd.read_csv(paths.DATA_PATH_SROK8C + "../catalog.csv")

    NUMBER_OF_STATIONS = len(stations)

    locs = np.empty((NUMBER_OF_STATIONS, 2))
    for stationNumber in range(NUMBER_OF_STATIONS):
        locs[stationNumber, 0] = converter(stations["Широта"][stationNumber])
        locs[stationNumber, 1] = converter(stations["Долгота"][stationNumber])

    stations['Широта'] = locs[:, 0]
    stations['Долгота'] = locs[:, 1]

    return stations


def plotRihmiWdcStations():
    stations = loadStations()

    FIG_PATH = "../figures/"
    fig = plt.figure(figsize=(16, 9), layout='tight')
    lambert_crs = ccrs.LambertConformal(central_longitude=105, standard_parallels=(38.5, 38.5))
    ax = fig.add_subplot(1, 1, 1, projection=lambert_crs)
    ax.set_extent([45, 155, 30, 75], crs=ccrs.Geodetic())
    for i in range(len(stations)):
        plt.plot(float(stations['Долгота'][i]), float(stations['Широта'][i]), transform=ccrs.PlateCarree(), marker='.',
            color='red')

    resol = '50m'  # use data at this scale
    bodr = cf.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol,
                                  facecolor='none', alpha=0.7)
    land = cf.NaturalEarthFeature('physical', 'land', scale=resol, edgecolor='k', facecolor=cf.COLORS['land'])
    ocean = cf.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cf.COLORS['water'])
    lakes = cf.NaturalEarthFeature('physical', 'lakes', scale=resol, edgecolor='b', facecolor=cf.COLORS['water'])
    rivers = cf.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resol, edgecolor='b', facecolor='none')

    ax.add_feature(land, facecolor='beige')
    ax.add_feature(ocean, linewidth=0.2)
    ax.add_feature(lakes, linewidth=0.2)
    ax.add_feature(rivers, linewidth=0.2)
    ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)

    gl = ax.gridlines(xlocs=np.arange(20, 180 + 1, 20), ylocs=np.arange(20, 90 + 1, 10), draw_labels=True, dms=True,
        x_inline=False, y_inline=False, alpha=0.5, color='k')
    gl.rotate_labels = False

    ax.set_title("Станции, включенные в набор данных «Основные метеорологические параметры (сроки)»\n", fontsize=16)
    fig.savefig(FIG_PATH + "rihmi_wdc_stations.png", dpi=250)


if __name__ == "__main__":
    plotRihmiWdcStations()
