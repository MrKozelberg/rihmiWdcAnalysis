import datetime
import numpy as np
import pandas as pd

import constants
import paths


def fromFloatingPointYearToDT(floatYear: float) -> datetime.datetime:
    year = int(floatYear)
    yearSize = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365
    day = int((floatYear - int(floatYear)) * yearSize)
    return datetime.datetime(year, 1, 1) + datetime.timedelta(days=day - 1)


def loadTSI() -> pd.DataFrame:
    """
    TOTAL SOLAR IRRADIANCE
    
    We use dataset from the URL: https://www.sidc.be/observations/space-based-timelines/tsi
    
    1) S. Dewitte, J. Cornelis and M. Meftah. Centennial Total Solar Irradiance Variation. Remote Sens. 2022, 14, 1072. https://doi.org/10.3390/rs14051072
    
    2) S. Dewitte and S. Nevens, "The Total Solar Irradiance Climate Data Record", 2016 ApJ 830 25, http://iopscience.iop.org/article/10.3847/0004-637X/830/1/25

    :return: dataframe of daily TSI
    """

    numberOfDays = (constants.DT_1 - constants.DT_0).days

    from dateutil.relativedelta import relativedelta
    dates = np.array([constants.DT_0 + relativedelta(days=i) for i in range(numberOfDays)])

    tsiDF = pd.read_csv(paths.TSI_PATH, delimiter=' ', skiprows=21, header=None, encoding='unicode_escape',
                        names=['Floating point year', "TSI in W/m²", "Julian date"])

    tsiDates = np.array([fromFloatingPointYearToDT(fltYear) for fltYear in tsiDF['Floating point year'].to_numpy()])
    dailyTSI = np.full(numberOfDays, constants.FILL_VALUE_FLOAT)
    for dateNumber, date in enumerate(dates):
        if sum(tsiDates == date) == 1:
            dailyTSI[dateNumber] = float(tsiDF["TSI in W/m²"][tsiDates == date])
        elif sum(tsiDates == date) > 1:
            dailyTSI[dateNumber] = float(tsiDF["TSI in W/m²"][tsiDates == date].mean())

    data = {"GMT date": dates, "TSI [W/m^2]": dailyTSI}

    result = pd.DataFrame(data=data)
    result["TSI [W/m^2]"] = result["TSI [W/m^2]"].interpolate()

    return result

if __name__ == "__main__":
    tsiDF = loadTSI()
    tsiDF.to_csv(paths.TSI_PATH_OUT)