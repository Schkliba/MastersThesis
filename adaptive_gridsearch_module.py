from evaluation_utils_module import task_job, rename, select_minimal_exaples
import itertools
import concurrent.futures
import constants as Cs
import os
import json
import datetime
import pickle
import optuna
import numpy as np
from scipy.spatial.distance import pdist
import logging
from concurrent.futures import Future


SEEDS = [101,102,103]

def evaluation_of_setup(en, run_name, alg, container, out_path, ev_ng, **kwargs):
    #we do not deviate the sigma
    # first we evaluate the current setup

    stat_futures = {}
    args = rename(kwargs)
    args["ng"] = ev_ng
    dirpath = os.path.join(os.path.realpath(out_path), container,alg, run_name)

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        print("Launching " + alg + "on Enviroment " + str(en) + f" with {args}")
        for s in SEEDS:
            future = executor.submit(task_job, container=container, alg=alg, en=en, args=args, s=s, out_path=dirpath + "/stat")
            stat_futures[future] = s

    stats = {}
    pops = {}
    maxes = []
    diversities = []
    for future in concurrent.futures.as_completed(stat_futures):
        s = stat_futures[future]
        result = future.result()
        if "fitness" not in stats:
            stats["fitness"] = {}
        stats["fitness"][s] = result[0]
        behaviors = list(map(lambda x:x[1], result[0]))
        fitnesses = list(map(lambda x:x[0], result[0]))
        maxes.append(np.max(fitnesses))
        diversity = pdist(np.array(behaviors)).mean()
        diversities.append(diversity)
        pops[s]= result[1] 
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{ts}_{container}_"+ "|".join(f"{k}: {v}" for k, v in args.items())
    stats["environment"] = en
    stats["algorithm"] = alg
    stats["container"] = container
    stats["arguments"] = args
    json_path = os.path.join(dirpath, filename +".json")

    with open(json_path, "w") as json_file:
        json.dump(stats,json_file)
        print("Finished "+ filename)
    return np.mean(maxes), np.mean(diversity)


def argument_combination_genration(selected, D_args):
    deltas = []
    contraints_global = []
    for i, a in enumerate(D_args):
        delta = D_args[a][0]
        constraints = D_args[a][1]
        deltas.append(delta)
        contraints_global.append(constraints)

    generated_arguments = []

    for delta in itertools.product(*deltas):
        new = selected.copy()
        keep = True
        for i, a in enumerate(D_args):
            new[a] = selected[a] + delta[i]
            passed = (len(contraints_global[i]) == 0 or
            contraints_global[i][0] is None or
            contraints_global[i][0] < new[a] or
            contraints_global[i][1] is None or
            contraints_global[i][1] > new[a])
            keep = keep and passed

        if not keep: continue
    

        generated_arguments.append(new)
    return generated_arguments

def process_generated_arguments(run_name, en, container, alg, generated_arguments, selected_fitnes, selected_diversity, visited, outpath):
    visited_new = visited.copy()
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        arg_futures={}        
        for args in generated_arguments:
            toupled = tuple(sorted(args.items()))
            if toupled in visited:
                future = Future().set_result(visited[toupled])

            future = executor.submit(
                evaluation_of_setup, 
                run_name=run_name,
                en=en, 
                alg=alg,
                ev_ng=20, 
                container=container,
                out_path=outpath,
                **args)
            arg_futures[future] = args
        max_fitness = selected_fitnes
        max_diversity = selected_diversity
        return_candidate = None
        for future in concurrent.futures.as_completed(arg_futures):
            args = arg_futures[future]
            toupled = tuple(sorted(args.items()))
            fitness, diversity  = future.result()
            visited_new[toupled] = (fitness, diversity)
            if fitness > max_fitness:
                print("We should have changed who's the best!")
                max_fitness = fitness
                return_candidate = args
                max_diversity = diversity
            elif fitness == max_fitness:
                if diversity > max_diversity:
                    print("We should have changed based on diversity!")
                    max_fitness = fitness
                    return_candidate = args
                    max_diversity = diversity
                
        return max_fitness, max_diversity, return_candidate, visited_new
    

def adaptive_lambda_grid_search(run_name, en,container, hops, starting_position, starting_fitness, dl, dm, dcr, dmr, outpath):
    visited = dict()
    visited[tuple(sorted(starting_position.items()))] = (starting_fitness, 0)
    selected = starting_position
    selected_fitness = starting_fitness
    selected_diversity = 0
    D_cr = [dcr, 0, -dcr]
    D_mr = [dmr, 0, -dmr]
    D_m = [dm, 0, -dm]
    D_l = [dl, 0, -dl]
    one_success = False
    for i in range(hops):
        generated_arguments = argument_combination_genration(
            selected=selected, 
            D_args={"cr":[D_cr, (0,1)], "l":[D_l, (0,None)]})
        # we have finised our 
        if generated_arguments == []: return selected, selected_fitness

        selection_fitness, selection_diversity, new_selected, new_visited = process_generated_arguments(
            run_name=run_name,
            en=en, 
            container=container, 
            alg="lambda",
            visited=visited, 
            generated_arguments=generated_arguments, 
            selected_fitnes=selected_fitness,
            selected_diversity=selected_diversity,
            outpath=outpath
        )
        if new_selected is not None:
            one_success = True
            if tuple(sorted(new_selected.items())) in visited:
                print("Cycle found! Terminating")
                return selected, new_visited
            selected = new_selected
            selected_diversity = selection_diversity
            selected_fitness = selection_fitness
        visited = new_visited
        generated_arguments = argument_combination_genration(
            selected=selected, 
            D_args={"mr":[D_mr, (0,1)], "m":[D_m, (0,None)]})
        # we have finised our 
        if generated_arguments == []: return selected, selected_fitness
        selection_fitness, selection_diversity, new_selected, new_visited = process_generated_arguments(
            run_name=run_name,
            en=en, 
            container=container, 
            alg="lambda", 
            generated_arguments=generated_arguments, 
            selected_fitnes=selected_fitness,
            selected_diversity=selected_diversity,
            visited=visited,
            outpath=outpath
        )
        if new_selected is None:
            visited = new_visited
        else:
            one_success = True
            if tuple(sorted(new_selected.items())) in visited:
                print("Cycle found! Terminating")
                return selected, new_visited
            visited = new_visited
            selected = new_selected
            selected_diversity = selection_diversity
            selected_fitness = selection_fitness
        
        if not one_success:
            return selected, visited
    return selected, visited



def adaptive_diff_grid_search(
        run_name,
        en,
        container, 
        hops, 
        starting_position, 
        starting_fitness, 
        dl, 
        dcr, 
        dmr, 
        outpath
    ):
    visited = dict()
    visited[tuple(sorted(starting_position.items()))] = (starting_fitness, 0)
    selected = starting_position
    selected_fitness = starting_fitness
    selected_diversity = 0
    D_cr = [dcr, 0, -dcr]
    D_mr = [dmr, 0, -dmr]
    D_l = [dl, 0, -dl]
    one_success = False
    for i in range(hops):

        generated_arguments = argument_combination_genration(
            selected=selected, 
            D_args={"l":[D_l, (0,None)]})
        # we have finised our 
        if generated_arguments == []: return selected, selected_fitness

        selection_fitness, selection_diversity, new_selected, new_visited = process_generated_arguments(
            run_name=run_name,
            en=en, 
            container=container, 
            alg="diff",
            visited=visited, 
            generated_arguments=generated_arguments, 
            selected_fitnes=selected_fitness,
            selected_diversity=selected_diversity,
            outpath=outpath
        )
        if new_selected is not None:
            one_success = True
            if tuple(sorted(new_selected.items())) in visited:
                print("Cycle found! Terminating")
                return selected, new_visited
            selected = new_selected
            selected_diversity = selection_diversity
            selected_fitness = selection_fitness
        visited = new_visited


        generated_arguments = argument_combination_genration(
            selected=selected, 
            D_args={"mr":[D_mr, (0,1)], "cr":[D_cr, (0,None)]})
        # we have finised our 
        if generated_arguments == []: return selected, selected_fitness
        selection_fitness, selection_diversity, new_selected, new_visited = process_generated_arguments(
            run_name=run_name,
            en=en, 
            container=container, 
            alg="diff",
            visited=visited, 
            generated_arguments=generated_arguments, 
            selected_fitnes=selected_fitness,
            selected_diversity=selected_diversity,
            outpath=outpath
        )
        if new_selected is None:
            visited = new_visited
        else:
            one_success = True
            if tuple(sorted(new_selected.items())) in visited:
                print("Cycle found! Terminating")
                return selected, new_visited
            visited = new_visited
            selected = new_selected
            selected_diversity = selection_diversity
            selected_fitness = selection_fitness
        
        if not one_success:
            return selected, visited
        
    return selected, visited

def adaptive_grid_search(en, alg, run_name, container, hops = 3, out_path="./Data/grid_search"):
    with open("relevant_studies.json", "r") as f:
        relevant_study_names = json.load(f)
    storage = f"sqlite:///Data/optuna/{en}/{container}/{alg}.db"
    study_name = relevant_study_names[container][alg][en]

    new_study = optuna.load_study(study_name=study_name,storage=storage)
    most_promising, mi = select_minimal_exaples([t.params for t in new_study.best_trials])
    selected_trial = new_study.best_trials[mi[0]]
    start =datetime.datetime.now()
    
    if alg=="lambda":
        if en == "lunarlander":
            dl = 15
            dm = 15
            dcr = 0.1
            dmr = 0.05
        else: #cartpole
            dl = 10
            dm = 10
            dcr = 0.1
            dmr = 0.05
        selected, visited = adaptive_lambda_grid_search(
            run_name=run_name,
            en=en,
            container=container, 
            hops=hops, 
            starting_position=rename(most_promising[0]),
            starting_fitness=selected_trial.value, 
            dl=dl, 
            dm=dm, 
            dcr=dcr, 
            dmr=dmr,
            outpath=out_path+f"/{en}"
        )
    elif alg =="diff":
        if en == "lunarlander":
            dl = 10
            dcr = 0.1
            dmr = 0.1
        else: #cartpole
            dl = 5
            dcr = 0.1
            dmr = 0.1
        selected, visited = adaptive_diff_grid_search(
            run_name=run_name,
            en=en,
            container=container, 
            hops=hops, 
            starting_position=rename(most_promising[0]),
            starting_fitness=selected_trial.value, 
            dl=dl,  
            dcr=dcr, 
            dmr=dmr,
            outpath=out_path+f"/{en}"
        )
    end =datetime.datetime.now()
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{ts}_{container}_{en}_protocol.json"
    dirpath = os.path.join(os.path.realpath(out_path),en, container,alg)

    protocol = {"start":start.strftime("%Y-%m-%d_%H-%M-%S"), "end": end.strftime("%Y-%m-%d_%H-%M-%S"), "origin": most_promising, "final":selected, "visited":[dict(k) for k in visited]}
    json_path = os.path.join(dirpath, filename + ".json")
    with open(json_path, "w") as json_file:
        json.dump(protocol,json_file)


    

        
    
    


        
    
    

  