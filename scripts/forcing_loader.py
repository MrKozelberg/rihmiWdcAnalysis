import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import linregress
import datetime as dt

import paths
from constants import DT_1, DT_0, FILL_VALUE_FLOAT
from scripts.physics_util import airHumFnc, waterPartialPrsFnc, sunEarthDistanceFnc, declinationFnc, czFnc, \
    optAirMassFnc, satWaterPartialPrsFnc
from scripts.stations import loadStations

warnings.filterwarnings('ignore')

TSI = pd.read_csv(paths.TSI_PATH_OUT) #loadTSI()
STATIONS = loadStations()
HOURS_OF_MEASUREMENTS_PER_DAY = 8
INTERVAL_BTW_MEASUREMENTS_IN_SEC = 3 * 3600
NUMBER_OF_HOURS = HOURS_OF_MEASUREMENTS_PER_DAY * (DT_1 - DT_0).days

VALID_QUALITIES = [0, 1, 2]


def datetimeFromLine(line) -> dt.datetime:
    return dt.datetime(int(line[6:10]), int(line[11:13]), int(line[14:16]), int(line[17:19]))


def localTimeFromLine(line) -> int:
    return int(line[37:39])


def cloudinessFromLine(line) -> float:
    code = line[53:55]
    quality = int(line[56:57])

    res = FILL_VALUE_FLOAT

    CLDNSS_11 = 0.05  # 11
    CLDNSS_12 = 0.95  # 12
    CLDNSS_13 = FILL_VALUE_FLOAT  # 13

    if quality in VALID_QUALITIES:
        if code == "11":
            res = CLDNSS_11
        elif code == "12":
            res = CLDNSS_12
        elif code == "13":
            res = CLDNSS_13
        else:
            res = float(code) / 10.0

    return res


def windFromLine(line) -> float:
    code = line[118:120]
    quality = int(line[121:122])
    idGT = line[123:124]

    res = FILL_VALUE_FLOAT
    if quality in VALID_QUALITIES and idGT == "0":
        res = float(code)

    return res


def precipTotFromLine(line) -> float:
    code = line[132:138]
    quality = int(line[139:140])
    res = FILL_VALUE_FLOAT
    if quality in VALID_QUALITIES:
        res = float(code)
    return res


def airTempFromLine(line) -> float:
    code = line[181:186]
    quality = int(line[187:188])
    res = FILL_VALUE_FLOAT
    if quality in VALID_QUALITIES:
        res = float(code)
    return res


def waterPartialPrsFromLine(line) -> float:
    code = line[231:236]
    quality = int(line[237:238])
    prec = line[239:240]
    res = FILL_VALUE_FLOAT
    if quality in VALID_QUALITIES:
        res = float(code)
        if prec == "1":
            res = np.round(res, 1)
        if prec == "2":
            res = np.round(res, 2)
        res *= 100
    return res


def relHumFromLine(line) -> float:
    code = line[241:244]
    quality = int(line[245:246])
    res = FILL_VALUE_FLOAT
    if quality in VALID_QUALITIES:
        res = float(code)
    return res


def srfPrsFromLine(line) -> float:
    code = line[267:273]
    quality = int(line[274:275])
    res = FILL_VALUE_FLOAT
    if quality in VALID_QUALITIES:
        res = float(code)
        res = np.round(res, 1) * 100.0
    return res


def calcAirHum(theWaterPartialPrs, theRelHum, theSrfPrs, theAirTemp) -> float:
    T_0 = 273.16

    res = FILL_VALUE_FLOAT

    if ~np.isnan(theSrfPrs):
        if ~np.isnan(theWaterPartialPrs):
            res = airHumFnc(theWaterPartialPrs, theSrfPrs)
        elif ~np.isnan(theRelHum) and ~np.isnan(theAirTemp):
            theWaterPartialPrs = waterPartialPrsFnc(theRelHum, theAirTemp + T_0)
            res = airHumFnc(theWaterPartialPrs, theSrfPrs)

    return res


def calcRsds(theCloudiness, theDatetime, theLocalTime, theTSIVal, lat, theSrfPrs):
    dayOfYear = 1 + (theDatetime - dt.datetime(theDatetime.year, 1, 1)).days

    sunEarthDistance = sunEarthDistanceFnc(dayOfYear)
    declination = declinationFnc(dayOfYear)
    cz = czFnc(lat, declination, theLocalTime)
    optAirMass = optAirMassFnc(cz, theSrfPrs)

    transmittance2 = 0.8

    directClr = max(0.0, cz * theTSIVal / sunEarthDistance ** 2 / (
            1 + optAirMass * (1 - transmittance2 ** 2) / 2 * transmittance2 ** 2))
    direct = directClr * (1.0 - theCloudiness)

    diffusiveClr = max(0.0, 0.38 * cz * (theTSIVal - directClr))
    diffusive = diffusiveClr * (1.0 - theCloudiness) + 0.34 * theCloudiness * (directClr + diffusiveClr)

    res = direct + diffusive

    return res


def calcRlds(theAirTemp, theRelHum, theSrfPrs, theCloudiness, z):
    SIGMA = 5.6697e-8  # W / m^2 / K^4
    DT_DZ = -0.0065  # K/m
    T0 = 273.16  # K

    airTempPrime = theAirTemp - DT_DZ * z
    waterPartialPrsPrime = (theRelHum / 100) * satWaterPartialPrsFnc(theAirTemp + T0)
    # Marks, D., Dozier, J. A clear-sky longwave radiation model
    # for remote alpine areas. Arch. Met. Geoph. Biokl. B. 27,
    # 159–187 (1979). https://doi.org/10.1007/BF02243741
    deltaA = 1.24 * (theSrfPrs / 1013.25) * ((waterPartialPrsPrime / 100) / (airTempPrime + T0)) ** 0.143

    if theAirTemp >= T0:
        B = 1 + 0.17 * theCloudiness ** 2
    else:
        B = 1 + 0.4 * theCloudiness * (theRelHum / 100)

    return B * deltaA * SIGMA * theAirTemp ** 4


def calcPrecipRate(precipTot):
    res = np.full(len(precipTot), FILL_VALUE_FLOAT)
    i0 = 0
    i1 = i0 + 1
    # durations = []
    for i, aa in enumerate(precipTot[1:]):
        if np.isnan(aa):  # not (aa != FILL_VALUE_FLOAT):
            i1 += 1
        else:
            # durations += [i1 - i0]
            if i1 - i0 < 2 * HOURS_OF_MEASUREMENTS_PER_DAY:
                res[i0:i1] = aa / (i1 - i0)
            else:
                res[i1-2*HOURS_OF_MEASUREMENTS_PER_DAY:i1] = aa / (2*HOURS_OF_MEASUREMENTS_PER_DAY)
            i0 = i
            i1 = i0 + 1
    # import matplotlib.pyplot as plt
    # plt.hist(durations, 1000, density=True, log=True); plt.show()
    return res / INTERVAL_BTW_MEASUREMENTS_IN_SEC


def IQRFilter(df: pd.DataFrame):
    types = df.dtypes
    cols = df.columns
    for i, col in enumerate(cols):
        if str(types.iloc[i]) != 'float64':
            continue
        q25 = df[col][~np.isnan(df[col])].quantile(0.25)
        q75 = df[col][~np.isnan(df[col])].quantile(0.75)
        iqr = q75 - q25
        cutoff = 1.5 * iqr
        lower, upper = q25 - cutoff, q75 + cutoff
        df[col][df[col] < lower] = FILL_VALUE_FLOAT
        df[col][df[col] > upper] = FILL_VALUE_FLOAT


def fillGaps(x):
    gaps = np.isnan(x)

    res = linregress(np.arange(NUMBER_OF_HOURS)[~gaps], x[~gaps])
    x_lin = res.intercept + res.slope * np.arange(NUMBER_OF_HOURS)

    dates = [DT_0 + dt.timedelta(seconds=INTERVAL_BTW_MEASUREMENTS_IN_SEC*h) for h in range(NUMBER_OF_HOURS)]

    x_seas = np.zeros(366)
    x_max = np.full(366, FILL_VALUE_FLOAT)
    x_min = np.full(366, FILL_VALUE_FLOAT)
    count_seas = np.zeros(366)
    for h in np.arange(NUMBER_OF_HOURS)[~gaps]:
        day_number = (dates[h] - dt.datetime(dates[h].year,1,1)).days
        x_seas[day_number] += (x[h] - x_lin[h])
        if np.isnan(x_max[day_number]):
            x_max[day_number] = x[h]
        if np.isnan(x_min[day_number]):
            x_min[day_number] = x[h]
        x_max[day_number] = np.maximum(x_max[day_number], x[h])
        x_min[day_number] = np.minimum(x_min[day_number], x[h])
        count_seas[day_number] += 1
    x_seas[count_seas > 0] /= count_seas[count_seas > 0]
    x_seas = interp1d(np.arange(366)[count_seas > 0], x_seas[count_seas > 0])(np.arange(366))
    x_max = interp1d(np.arange(366)[~np.isnan(x_max)], x_max[~np.isnan(x_max)])(np.arange(366))
    x_min = interp1d(np.arange(366)[~np.isnan(x_min)], x_min[~np.isnan(x_min)])(np.arange(366))


    x_daily = np.zeros(HOURS_OF_MEASUREMENTS_PER_DAY)
    count_daily = np.zeros(HOURS_OF_MEASUREMENTS_PER_DAY)
    for h in np.arange(NUMBER_OF_HOURS)[~gaps]:
        day_number = (dates[h] - dt.datetime(dates[h].year,1,1)).days
        hour_number = int(dates[h].hour / 24 * HOURS_OF_MEASUREMENTS_PER_DAY)
        x_daily[hour_number] += (x[h] - x_lin[h] - x_seas[day_number])
        count_daily[hour_number] += 1
    x_daily[count_daily > 0] /= count_daily[count_daily > 0]
    x_daily = interp1d(
        np.arange(HOURS_OF_MEASUREMENTS_PER_DAY)[count_daily > 0],
        x_daily[count_daily > 0]
    )(np.arange(HOURS_OF_MEASUREMENTS_PER_DAY))

    for h in np.arange(NUMBER_OF_HOURS)[gaps]:
        day_number = (dates[h] - dt.datetime(dates[h].year, 1, 1)).days
        hour_number = int(dates[h].hour / 24 * HOURS_OF_MEASUREMENTS_PER_DAY)
        x[h] = x_lin[h] + x_seas[day_number] + x_daily[hour_number]
        x[h] = np.maximum(x_min[day_number], x[h])
        x[h] = np.minimum(x_max[day_number], x[h])


def loadForcing(stationNumber: int) -> pd.DataFrame:
    stationID = STATIONS['Индекс ВМО'][stationNumber]
    lat = STATIONS['Широта'][stationNumber]
    z = STATIONS['Высота'][stationNumber]

    datetimeGMT = np.empty(NUMBER_OF_HOURS, dtype=dt.datetime)
    rsds = np.full(NUMBER_OF_HOURS, FILL_VALUE_FLOAT)
    rlds = np.full(NUMBER_OF_HOURS, FILL_VALUE_FLOAT)
    wind = np.full(NUMBER_OF_HOURS, FILL_VALUE_FLOAT)
    precipTot = np.full(NUMBER_OF_HOURS, FILL_VALUE_FLOAT)
    airTemp = np.full(NUMBER_OF_HOURS, FILL_VALUE_FLOAT)
    airHum = np.full(NUMBER_OF_HOURS, FILL_VALUE_FLOAT)
    srfPrs = np.full(NUMBER_OF_HOURS, FILL_VALUE_FLOAT)

    hourNumber = -1

    with open(paths.DATA_PATH_SROK8C + f"{stationID}.dat") as file:
        for line in file:
            hourNumber += 1

            theDatetimeGMT = datetimeFromLine(line)
            if theDatetimeGMT < DT_0:
                hourNumber = -1
                continue
            if theDatetimeGMT >= DT_1:
                break
            datetimeGMT[hourNumber] = theDatetimeGMT

            theLocalTime = localTimeFromLine(line)

            theCloudiness = cloudinessFromLine(line)

            theWind = windFromLine(line)
            wind[hourNumber] = theWind

            thePrecipTot = precipTotFromLine(line)
            precipTot[hourNumber] = thePrecipTot

            theAirTemp = airTempFromLine(line)
            airTemp[hourNumber] = theAirTemp

            theWaterPartialPrs = waterPartialPrsFromLine(line)

            theRelHum = relHumFromLine(line)

            theSrfPrs = srfPrsFromLine(line)
            srfPrs[hourNumber] = theSrfPrs

            theAirHum = calcAirHum(theWaterPartialPrs, theRelHum, theSrfPrs, theAirTemp)
            airHum[hourNumber] = theAirHum

            dayNumber = hourNumber // HOURS_OF_MEASUREMENTS_PER_DAY
            theTSIVal = float(TSI["TSI [W/m^2]"][dayNumber])
            theRsds = calcRsds(theCloudiness, theDatetimeGMT, theLocalTime, theTSIVal, lat, theSrfPrs)
            rsds[hourNumber] = theRsds

            theRlds = calcRlds(theAirTemp, theRelHum, theSrfPrs, theCloudiness, z)
            rlds[hourNumber] = theRlds

    precipRate = calcPrecipRate(precipTot)

    forcingDF = pd.DataFrame(data={"GMT datetime": datetimeGMT, "Downward short-wave radiation (W/m^2)": rsds,
        "Downward long-wave radiation (W/m^2)": rlds, "Wind speed (m/s)": wind, "Precipitation rate (mm/s)": precipRate,
        "Air temperature (°C)": airTemp, "Specific air humidity (kg/kg)": airHum, "Surface pressure (Pa)": srfPrs})

    # Remove outliers
    IQRFilter(forcingDF[forcingDF.columns[1:]])

    # Fill gaps
    gapsFraction = {"Downward short-wave radiation (W/m^2)": 0.0,
        "Downward long-wave radiation (W/m^2)": 0.0, "Wind speed (m/s)": 0.0, "Precipitation rate (mm/s)": 0.0,
        "Air temperature (°C)": 0.0, "Specific air humidity (kg/kg)": 0.0, "Surface pressure (Pa)": 0.0}
    for col in forcingDF.columns[1:]:
        gapsFraction[col] = sum(np.isnan(forcingDF[col])) / NUMBER_OF_HOURS
        forcingDF[col] = forcingDF[col].interpolate(limit=4, method='nearest')
        fillGaps(forcingDF[col])

    return forcingDF, gapsFraction

if __name__ == "__main__":

    # load forcing data for the first station
    a, b = loadForcing(0)
    print(b)
    # print(a)
    #
    # # Show the loaded fields
    # import seaborn as sns
    import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(1, len(a.columns[1:]), figsize=((3 + 2) * len(a.columns[1:]), 3))
    # for i, col in enumerate(a.columns[1:]):
    #     _ = sns.kdeplot(
    #         a[~np.isnan(a[col])],
    #         x=col,
    #         ax=ax[i],
    #         clip=(a[~np.isnan(a[col])][col].min(), a[~np.isnan(a[col])][col].max())
    #     )
    # plt.subplots_adjust(wspace=0.3)
    #
    # plt.show()

    # i = 0
    # for col in a.columns[1:]:
    #     i += 1
    #     plt.plot(a['GMT datetime'], ~np.isnan(a[col])[:] * i, alpha=0.5, label=col)
    # plt.show()

    for col in a.columns[1:]:
        plt.plot(a['GMT datetime'], a[col], alpha=0.5, label=col)
        plt.legend()
        plt.show()



