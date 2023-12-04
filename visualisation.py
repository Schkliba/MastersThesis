from deap import tools
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statconf
def logbook2pandas(logbook:tools.Logbook):
    df = pd.DataFrame()
    for stat in statconf.STAT_LIST:
        df[stat] = np.array(logbook.select(stat))
    return df

def single_run_chart(df:pd.DataFrame):
    sns.lineplot(data=df,x="gen", y="avg")
    plt.show()