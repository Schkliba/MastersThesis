
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import json
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist
from evaluation_utils_module import  rename, select_minimal_examples_old
from constants import LAMBDA_CONTS, DIFF_CONTS, ENIVROMENTS
from functools import lru_cache
from itertools import combinations
import math

def load_gen_direct_data(en, cont, alg, name):
    path = f"./Data/generation_ex/{en}/{cont}/{alg}/{name}"
    json_files = list(Path(path).glob("*.json"))
    print(f"Found {len(json_files)} files!")
    with open(json_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_final_direct_data(en, cont, alg, name):
    path = f"./Data/final/{en}/{cont}/{alg}/{name}"
    json_files = list(Path(path).glob("*.json"))
    print(f"Found {len(json_files)} files!")
    with open(json_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_grid_direct_data(en, cont, alg, name):
    path = f"./Data/grid_search/{name}/{en}/{cont}/{alg}"
    json_files = list(Path(path).glob("*.json"))
    print(f"Found {len(json_files)} files!")
    with open(json_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def gen_extract_values(file_content):
    df = pd.DataFrame(file_content["fitness"]).reset_index(names="seed").melt(
        id_vars="seed", 
        var_name="ng", 
        value_name="population"
    )
    df["ng"] = df["ng"].astype(int)
    df["max_fit"] = df["population"].apply(lambda x: np.max(list(map(lambda y: y[0], x))))
    df["avg_fit"] = df["population"].apply(lambda x: np.mean(list(map(lambda y: y[0], x))))
    df["median_fit"] = df["population"].apply(lambda x: np.median(list(map(lambda y: y[0], x))))
    df["min_fit"] = df["population"].apply(lambda x: np.min(list(map(lambda y: y[0], x))))
    df["std_fit"] = df["population"].apply(lambda x: np.std(list(map(lambda y: y[0], x))))

    df["div"] = df["population"].apply(lambda x: np.mean(pdist(list(map(lambda y: y[1], x)))))
    return df



def final_extract_values(file_content):
    df = pd.DataFrame(file_content["fitness"]).reset_index(names="seed").melt(
        var_name="seed", 
        value_name="population"
    )
    df["max_fit"] = df["population"].apply(lambda x: np.max(list(map(lambda y: y[0], x))))
    df["avg_fit"] = df["population"].apply(lambda x: np.mean(list(map(lambda y: y[0], x))))
    df["median_fit"] = df["population"].apply(lambda x: np.median(list(map(lambda y: y[0], x))))
    df["min_fit"] = df["population"].apply(lambda x: np.min(list(map(lambda y: y[0], x))))
    df["std_fit"] = df["population"].apply(lambda x: np.std(list(map(lambda y: y[0], x))))
    df["div"] = df["population"].apply(lambda x: np.mean(pdist(list(map(lambda y: y[1], x)))))

    return df

def get_protocol_table2(cont, protocol):
    origin, index = select_minimal_examples_old(protocol["origin"])
    df_template = {
        cont:{
            "origin":rename(origin[0]),
            "final": protocol["final"]
        }
    }
    if cont == "novelty_limit":
        print(protocol["origin"])
        print(protocol["final"])
    df = pd.concat(
        {
            row: pd.Series({
                (lvl1, lvl2): value
                for lvl1, sub in vals.items()
                for lvl2, value in sub.items()
            })
            for row, vals in df_template.items()
        },
        axis=1
    ).T

    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df

def get_protocol_table(en, alg, name):
    big_table = pd.DataFrame()
    cts = LAMBDA_CONTS if alg == "lambda" else DIFF_CONTS
    for cont in cts:
        protocol = load_grid_direct_data(en, cont, alg, name)
        table = get_protocol_table2(cont, protocol)
        big_table = pd.concat([big_table, table])

    return big_table.sort_index(axis=1)


def plot_generations(en, alg, name, attr:str):
    cts = LAMBDA_CONTS if alg == "lambda" else DIFF_CONTS
    df = pd.DataFrame()
    for cont in cts:
        content = load_gen_direct_data(en=en, alg=alg, cont=cont, name=name)
        temp = gen_extract_values(content)
        temp["cont"] = cont
        df = pd.concat([df, temp], axis=0)
    df = df.reset_index(drop=True)
    def trim_25(df):
        q1 = df[attr].quantile(0.25)
        q3 = df[attr].quantile(0.75)
        return df[(df[attr] >= q1) & (df[attr] <= q3)]

    dff = df[["seed","ng", attr, "cont"]]
    filtered = dff.groupby(["ng", "cont"], group_keys=False).apply(trim_25)
    return sns.lineplot(data=filtered, x="ng", y=attr, hue="cont")  

@lru_cache(maxsize=None)
def get_final_seed_mapped_value(en, cont, alg, name):
    file_content = load_final_direct_data(en, cont , alg, name)
    df = pd.DataFrame(file_content["fitness"]).melt(
            var_name="seed", 
            value_name="population"
    )
    df["fitnesses"] = df["population"].apply(lambda x: x[0][0])
    df["behaviors"] = df["population"].apply(lambda x: x[1])
    df = df.drop("population", axis=1)

    def behavior_diversity(x):
        X = np.array([np.asarray(v) for v in x], dtype=float)
        X = np.vstack(X)
        return pdist(X).mean()

    return df.groupby("seed").agg({"fitnesses": np.max, "behaviors": lambda x: behavior_diversity(x) })

def friedman_test_total_containers(names, attribute):
    big_table = pd.DataFrame()
    cts = LAMBDA_CONTS
    conts = list(cts.keys())
    for cont in conts:
        values = pd.DataFrame()
        for alg in ["lambda", "diff"]:
            for en in ENIVROMENTS:
                values_en = get_final_seed_mapped_value(en, cont, alg, names[en]).copy()
                values_en.index = values_en.index + str(en) + str(alg)
                values = pd.concat([values, values_en])
        table = values[attribute]
        table.name = cont
        big_table = pd.concat([big_table, table], axis=1)
    return friedmanchisquare(*[big_table[col] for col in big_table.columns])

def friedman_test_per_environment_containers(en, names, attribute):
    big_table = pd.DataFrame()
    cts = LAMBDA_CONTS
    conts = list(cts.keys())
    for cont in conts:
        values = pd.DataFrame()
        for alg in ["lambda", "diff"]:
            values_en = get_final_seed_mapped_value(en, cont, alg, names[en]).copy()
            values_en.index = values_en.index + str(en) + str(alg)
            values = pd.concat([values, values_en])
        table = values[attribute]
        table.name = cont
        big_table = pd.concat([big_table, table], axis=1)
    return friedmanchisquare(*[big_table[col] for col in big_table.columns])

def hodges_lehmann_paired(x, y):
    d = np.asarray(x) - np.asarray(y)
    walsh = [(d[i] + d[j]) / 2
              for i in range(len(d))
              for j in range(i, len(d))]
    return np.median(walsh)

def wilcoxon_test_per_environment_containers(en,names, attribute):
    cts = LAMBDA_CONTS #if alg == "lambda" else DIFF_CONTS
    big_table = pd.DataFrame()
    conts = list(cts.keys())
    for cont in conts:
        values = pd.DataFrame()
        for alg in ["lambda", "diff"]:
            values_en = get_final_seed_mapped_value(en, cont, alg, names[en]).copy()
            values_en.index = values_en.index + str(en) + str(alg)
            values = pd.concat([values, values_en])
        table = values[attribute]
        table.name = cont
        big_table = pd.concat([big_table, table], axis=1)
    tests = []
    for duelants in combinations(conts, 2):
        selection1 = big_table[duelants[0]]
        selection2 = big_table[duelants[1]]
        stats, p = wilcoxon(selection1, selection2)
        direction = hodges_lehmann_paired(selection1, selection2)
        tests.append((duelants[0], duelants[1], stats, p, direction))

    return pd.DataFrame(tests, columns=["contestant1", "contestant2", "wilcoxon_T", "p", "direction"])

def wilcoxon_test_per_env_alg_containers(en,alg, names, attribute):
    cts = LAMBDA_CONTS #if alg == "lambda" else DIFF_CONTS
    big_table = pd.DataFrame()
    conts = list(cts.keys())
    for cont in conts:
        values = get_final_seed_mapped_value(en, cont, alg, names[en]).copy()
        values.index = values.index + str(en) + str(alg)
        table = values[attribute]
        table.name = cont
        big_table = pd.concat([big_table, table], axis=1)
    tests = []
    for duelants in combinations(conts, 2):
        selection1 = big_table[duelants[0]]
        selection2 = big_table[duelants[1]]
        stats, p = wilcoxon(selection1, selection2)
        direction = hodges_lehmann_paired(selection1, selection2)
        tests.append((duelants[0], duelants[1], stats, p, direction))

    return pd.DataFrame(tests, columns=["contestant1", "contestant2", "wilcoxon_T", "p", "direction"])

def wilcoxon_test_per_environment_algorithms(en, names, attribute):
    cts = LAMBDA_CONTS #if alg == "lambda" else DIFF_CONTS
    big_table = pd.DataFrame()
    conts = list(cts.keys())
    for alg in ["lambda", "diff"]:
        values = pd.DataFrame()
        for cont in conts:
            values_en = get_final_seed_mapped_value(en, cont, alg, names[en]).copy()
            values_en.index = values_en.index + str(en) + str(cont)
            values = pd.concat([values, values_en])
        table = values[attribute]
        table.name = alg
        big_table = pd.concat([big_table, table], axis=1)
    stats, p = wilcoxon(big_table["lambda"], big_table["diff"])
    direction = hodges_lehmann_paired(big_table["lambda"],  big_table["diff"])
    print(direction)
    return stats, p, direction

def wilcoxon_test_total_algorithms(names, attribute):
    cts = LAMBDA_CONTS #if alg == "lambda" else DIFF_CONTS
    big_table = pd.DataFrame()
    conts = list(cts.keys())
    for alg in ["lambda", "diff"]:
        values = pd.DataFrame()
        for cont in conts:
            for en in ENIVROMENTS:
                values_en = get_final_seed_mapped_value(en, cont, alg, names[en]).copy()
                values_en.index = values_en.index + str(en) + str(cont)
                values = pd.concat([values, values_en])
        table = values[attribute]
        table.name = alg
        big_table = pd.concat([big_table, table], axis=1)   
    stats, p = wilcoxon(big_table["lambda"], big_table["diff"])
    return stats, p

def wilcoxon_test_total_containers(names, attribute):
    cts = LAMBDA_CONTS #if alg == "lambda" else DIFF_CONTS
    big_table = pd.DataFrame()
    conts = list(cts.keys())
    for cont in conts:
        values = pd.DataFrame()
        for alg in ["lambda", "diff"]:
            for en in ENIVROMENTS:
                values_en = get_final_seed_mapped_value(en, cont, alg, names[en]).copy()
                values_en.index = values_en.index + str(en) + str(alg)
                values = pd.concat([values, values_en])
        table = values[attribute]
        table.name = cont
        big_table = pd.concat([big_table, table], axis=1)
    tests = []
    for duelants in combinations(conts, 2):
        selection1 = big_table[duelants[0]]
        selection2 = big_table[duelants[1]]
        stats, p = wilcoxon(selection1, selection2)
        tests.append((duelants[0], duelants[1], stats, p))
    return pd.DataFrame(tests, columns=["contestant1", "contestant2", "wilcoxon_T", "p"])

def apply_corrections(wilcoxon_df:pd.DataFrame):
    new = wilcoxon_df.copy()
    pvals = new["p"]
    reject, p_adj, _, _ = multipletests(pvals, method="holm")
    new["holm_p"] = p_adj
    new["holm_reject"] = reject
    reject, p_adj, _, _ = multipletests(
        pvals,
        method="simes-hochberg"
    )
    new["hoch_p"] = p_adj
    new["hoch_reject"] = reject
    return new

def friedman_test_intergroup_total(alg, names, attribute, groups):
    big_table = pd.DataFrame()
    for g in groups:
        group_values = pd.DataFrame()
        for cont in groups[g]:
            values = pd.DataFrame()
            for en in ENIVROMENTS:
                values_en = get_final_seed_mapped_value(en, cont, alg, names[en]).copy()
                values_en.index = values_en.index + str(en)
                values = pd.concat([values, values_en])
            table = values[attribute]
            table.name = cont
            group_values[cont] = values
        table = group_values.median(axis=1)
        table.name = g
    big_table = pd.concat([big_table, table], axis=1)
    return friedmanchisquare(*[big_table[col] for col in big_table.columns])


def friedman_test_intragroup_total(alg, names, attribute, group):
    big_table = pd.DataFrame()
    for cont in group:
        values = pd.DataFrame()
        for en in ENIVROMENTS:
            values_en = get_final_seed_mapped_value(en, cont, alg, names[en]).copy()
            values_en.index = values_en.index + str(en)
            values = pd.concat([values, values_en])
        table = values[attribute]
        table.name = cont
    big_table = pd.concat([big_table, table], axis=1)
    return friedmanchisquare(*[big_table[col] for col in big_table.columns])

@lru_cache(maxsize=None)
def get_final_seed_mapped_value(en, cont, alg, name):
    file_content = load_final_direct_data(en, cont , alg, name)
    df = pd.DataFrame(file_content["fitness"]).melt(
            var_name="seed", 
            value_name="population"
    )
    df["fitnesses"] = df["population"].apply(lambda x: x[0][0])
    df["behaviors"] = df["population"].apply(lambda x: x[1])
    df = df.drop("population", axis=1)

    def behavior_diversity(x):
        X = np.array([np.asarray(v) for v in x], dtype=float)
        X = np.vstack(X)
        return pdist(X).mean()

    return df.groupby("seed").agg({"fitnesses": np.max, "behaviors": lambda x: behavior_diversity(x) })

