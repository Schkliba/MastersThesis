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

def novelty_logbook2pandas(logbook:tools.Logbook):
    df = pd.DataFrame()
    df.index = np.array(logbook.chapters["fitness"].select("gen"))
    df.index.name = "gen"
    for stat in statconf.STAT_LIST:
        df["fit_" + stat] = np.array(logbook.chapters["fitness"].select(stat))
        df["nov_" + stat] = np.array(logbook.chapters["novelty"].select(stat))
    return df

def single_novelty_run_chart(df:pd.DataFrame):
    fig, axs = plt.subplots(2,1)
    sns.lineplot(data=df, x="gen", y='fit_avg', ax=axs[0])
    sns.lineplot(data=df, x="gen", y='nov_avg', ax=axs[1])
    plt.show()

def single_run_chart(df:pd.DataFrame):
    sns.lineplot(data=df,x="gen", y="avg")
    plt.show()