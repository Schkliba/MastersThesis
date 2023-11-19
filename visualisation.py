from deap import tools
import seaborn as sns
import pandas as pd

def logbook2pandas(logbook:tools.Logbook, stats):
    df = pd.DataFrame()
    for stat in stats:
        df[stat] = logbook.select(stat)
    return df

