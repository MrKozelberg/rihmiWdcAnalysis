import numpy as np


def airHumFnc(theWaterPartialPrs, theSrfPrs):
    EPSILON = 287.0597 / 461.5250
    return EPSILON * theWaterPartialPrs / (theSrfPrs - (1.0 - EPSILON) * theWaterPartialPrs)


def satWaterPartialPrsFnc(temp):
    E_0 = 611.21
    T_0 = 273.16
    if temp >= T_0:
        return E_0 * np.exp(17.502 * (temp - T_0) / (temp - 32.19))
    else:
        return E_0 * np.exp(22.587 * (temp - T_0) / (temp + 0.7))


def waterPartialPrsFnc(relHum, temp):
    return (relHum / 100) * satWaterPartialPrsFnc(temp)


def sunEarthDistanceFnc(dayOfYear):
    return 1.0 - 0.01672 * np.cos((dayOfYear - 4) / 365.256363 * 2 * np.pi)


def declinationFnc(dayOfYear):
    return np.arcsin(np.sin(-23.44 / 180 * np.pi) * np.cos(
        2 * np.pi / 365.24 * (dayOfYear + 10) + 2 * 0.0167 * np.sin(2 * np.pi / 365.24 * (dayOfYear - 2))))


def czFnc(lat, declination, theLocalTime):
    return (np.sin(lat / 180 * np.pi) * np.sin(declination) + np.cos(lat / 180 * np.pi) * np.cos(declination) * np.cos(
        np.pi / 12.0 * theLocalTime))


def optAirMassFnc(cz, theSrfPrs):
    return 2.0016 * (theSrfPrs / 100) / (1013.25 * (cz + np.sqrt(cz ** 2 + 0.003147)))
