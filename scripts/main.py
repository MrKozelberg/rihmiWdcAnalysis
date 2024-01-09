from scripts.forcing_loader import loadForcing, STATIONS

NUMBER_OF_STATIONS = len(STATIONS)

if __name__ == "__main__":
    for stationNumber in range(5): #NUMBER_OF_STATIONS):
        print(STATIONS[stationNumber:stationNumber+1].transpose())
        forcingDF, gapsFraction = loadForcing(stationNumber)
        print(gapsFraction)
        if gapsFraction[max(gapsFraction)] > 0.3:
            print("Data from this station has too much gaps")
        print(" ")