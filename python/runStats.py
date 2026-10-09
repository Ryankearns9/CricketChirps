import csv
import pandas as pd
import matplotlib.pyplot as plt

#User Inputs
csvFileName = "testCSV.csv"

#Begin Code

#Read CSV
df = pd.read_csv(csvFileName)


speciesNames = ['Backross Pennylvanicus', 'Pennsylvanicus',         'Backcross Penn',
                 'Firmus',                'Unknown',                'Backross Firmus',
              'F2 Hybrid',                'F1 Hybrid']

TargetColumns = ['BurstDuration (s)', 'BurstPower (dB)','Burst Freq (kHz)','numPulses',"Time between bursts"]


#Sort out bad detects
powerMin = 95 #dB
minPulses = 4

powerFilter = df[TargetColumns[1]] >= powerMin
pulseFilter = df[TargetColumns[3]] >= minPulses
filteredDF = df[powerFilter & pulseFilter]

#Calculate time between bursts
df["Time between bursts"] = df.groupby('FileName')['BurstStart (s)'].diff()
filteredDF["Time between bursts"] = filteredDF.groupby('FileName')['BurstStart (s)'].diff()

#Check Pulses in Bursts
for targetCol in TargetColumns:
    ax = df.boxplot(by ='Species', column =[targetCol], grid = False)
    fig = ax.get_figure()
    fig.set_size_inches(32, 18)
    fig.savefig(f"../Figures/Stats/unfiltered_{targetCol}.png", bbox_inches='tight', dpi=300)

    ax = filteredDF.boxplot(by ='Species', column =[targetCol], grid = False)
    fig = ax.get_figure()
    fig.set_size_inches(32, 18)
    fig.savefig(f"../Figures/Stats/filtered_{targetCol}.png", bbox_inches='tight', dpi=300)


