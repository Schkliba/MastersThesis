from deap import tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statconf

def logbook2pandas(logbook:tools.Logbook):
    df = pd.DataFrame()
    df["gen"] = np.array(logbook.select("gen"))
    df.index.name = "gen"
    for stat in statconf.STAT_LIST:
        df[stat] = logbook.select(stat)
    return df

def novelty_logbook2pandas(logbook:tools.Logbook):
    df = pd.DataFrame()
    df["gen"] = np.array(logbook.chapters["fitness"].select("gen"))
    df.index.name = "gen"
    for stat in statconf.STAT_LIST:
        df["fit_" + stat] = np.array(logbook.chapters["fitness"].select(stat))
        df["nov_" + stat] = np.array(logbook.chapters["novelty"].select(stat))
    return df

def single_novelty_run_chart(df:pd.DataFrame):
    fig, axs = plt.subplots(2,1)
    fitness = df.melt(id_vars=["gen"], value_vars=["fit_avg", "fit_min", "fit_max"], var_name="type", value_name="fitness")
    novelty = df.melt(id_vars=["gen"], value_vars=["nov_avg", "nov_min", "nov_max"],var_name="type", value_name="novelty")
    sns.lineplot(data=fitness, x="gen", y="fitness", hue="type", ax=axs[0])
    sns.lineplot(data=novelty, x="gen", y="novelty", hue="type",  ax=axs[1])
    plt.show()

def single_run_chart(df:pd.DataFrame):
    fitness = df.melt(id_vars=["gen"], value_vars=["avg", "min", "max"], var_name="type", value_name="fitness")
    sns.lineplot(data=fitness, x="gen", y="fitness", hue="type")
    plt.show()