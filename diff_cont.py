from libs import constants as consts
import visualisation
import libs.agent_infra as ai
import datetime
import argparse
import numpy as np
import os

from deap import base
from deap import creator

parser = argparse.ArgumentParser()

parser.add_argument("-g", "--generations", help="Number of generations", type=int, default=10)
parser.add_argument("-cr", "--cross_rate", help="Rate of crossing individuals", type=float, default=0.6)
parser.add_argument("-cn", "--container", help="Container type", choices=["fitness", "novelty", "add_novelty", "archiving", "fit_archiving", "novelty_archiving"], default="fitness")
parser.add_argument("-en", "--environment", help="Enviroment type", choices=["cartpole", "lunarlander", "waterworld"], default="cartpole")

parser.add_argument("-fw", "--fit_weight", help="Initial weight given to the fitnes", type=float, default=0.2)
parser.add_argument("-rd", "--decay", help="Decay of the weight", type=float, default=0.2)

parser.add_argument("-lm", "--limit", help="Limit for limit_archiving", type=int, default=0)

parser.add_argument("-ab", "--archive_batch", help="Batch of the archive", type=int, default=1)
parser.add_argument("-at", "--archive_period", help="Period of archiving", type=int, default=2)


parser.add_argument("-mr", "--mutation_rate", help="Rate of mutation", type=float, default=0.9)
parser.add_argument("-p", "--pop", help="Base population", type=int, default=7)
parser.add_argument("-s", "--seed", help="Seed of the random generator", type=int, default=42)
parser.add_argument("-e", "--episodes", help="Seed of the random generator", type=int, default=5)
parser.add_argument("-pt", "--out_path", help="Path to store output data", type=str, default="./Data/Junk/")
parser.add_argument("-exp", "--experiment", help="indication if it's experiment", action="store_true", default=False)


def main(args:argparse.Namespace):
    ng, l, cr, mr, seed = args.generations, args.pop, args.cross_rate, args.mutation_rate, args.seed
    container = args.container
    environment = args.environment
    episodes = args.episodes
    out_path = args.out_path
    if container == "add_novelty":
        fit_weight = args.fit_weight
    else:
        fit_weight=None
    df, pop = argumented_function(
        env=environment,
        container=container,
        ng=ng, l=l, cr=cr, mr=mr, episodes=episodes, decay=args.decay,
        archive_batch=args.archive_batch, archiving_period=args.archive_period,
        limit = args.limit,
        fitness_weight=fit_weight,
        seed=seed,
        out_path=out_path
    )
    if args.container == "fitness" or args.container == "fit_archiving":
        visual_chart = visualisation.single_run_chart
    else:
        visual_chart = visualisation.single_novelty_run_chart

    if not args.experiment:
        visual_chart(df)
    return df, pop

def argumented_function(
        env:str, 
        container:str,  
        ng:int, 
        l:int, 
        mr:float, 
        cr:float, 
        episodes:int = 3, 
        seed:int = 42,
        archiving_period = 2,
        archive_batch = 1,
        fitness_weight=0.2,
        decay=2,
        limit=0,
        out_path = "./Data/Junk/"):
    
    cont_cls = consts.DIFF_CONTS[container]
    enviroment = consts.ENIVROMENTS[env](False)
    replay_env = consts.ENIVROMENTS[env](True)
    rp = lambda x: replay_env.evalutation(x, seed, 1)

    toolbox = base.Toolbox()
    toolbox.register("triOp", ai.tri_op, alpha=mr)
    toolbox.register("recombine", ai.recombine, prob_filter=cr)

    if container == "fitness" or container == "fit_archiving" or container == "elite_archiving":
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual",  enviroment.get_individual_base(), fitness=creator.FitnessMax, behaviour=None)
        toolbox.register("evaluate", enviroment.evalutation, seed=seed, episodes=episodes)
        visual_conv = visualisation.logbook2pandas
        #visual_chart = visualisation.single_run_chart
    elif container == "add_novelty":
        creator.create("FitnessNovelty", base.Fitness, weights=(1.0, ))
        creator.create("FitnessTrue", base.Fitness, weights=(1.0, )) 
        creator.create(
                    "Individual", 
                    enviroment.get_individual_base(), 
                    fitness=creator.FitnessNovelty, 
                    fitness2=creator.FitnessTrue,
                    behaviour=None
        )
        toolbox.register("evaluate", enviroment.evalutation_b, seed=seed, episodes=episodes)
        visual_conv = visualisation.novelty_logbook2pandas
        #visual_chart = visualisation.single_novelty_run_chart
    else:
        creator.create("FitnessNovelty", base.Fitness, weights=(1.0, ))
        creator.create("FitnessTrue", base.Fitness, weights=(1.0, )) 
        creator.create(
                        "Individual", 
                        enviroment.get_individual_base(), 
                        fitness=creator.FitnessNovelty , 
                        fitness2=creator.FitnessTrue, 
                        behaviour=None
        )
        toolbox.register("evaluate", enviroment.evalutation_b, seed=seed, episodes=episodes)
        visual_conv = visualisation.novelty_logbook2pandas
        #visual_chart = visualisation.single_novelty_run_chart

    enviroment.prepare_toolbox(toolbox, creator.Individual)
    if container == "fit_archiving" or container == "novelty_archving" or container=="elite_archiving":
        alg = cont_cls(
            l,toolbox, seed, ng, creator, 
            archiving_period=archiving_period, 
            store_batch=archive_batch
        )
    elif container == "novelty_limit":
        alg = cont_cls(
            l,toolbox, seed, ng, creator, 
            archiving_period=archiving_period, 
            store_batch=archive_batch,
            archive_limit = limit

        )
    elif container == "add_novelty" or container == "sub_novelty" :
        alg = cont_cls(
            l,toolbox, seed, ng, creator,
            fitness_limits = consts.FITNESS_LIMIT[env], 
            novelty_limits = consts.NOVELTY_LIMIT[env],
            fit_w = fitness_weight, 
            decay=decay
        )
    else:
        alg = cont_cls(
            l,toolbox, seed, ng, creator
        )
    alg.replay_f = rp
    alg.run()
    df = visual_conv(alg.logbook)
    dirpath = os.path.join(os.path.realpath(out_path), container,"diff")
    filepath = "%s,%s,g%i,e%i,pop%i,mr%.4f,s%i.out" % (
                                                str(datetime.datetime.utcnow()),
                                                env, 
                                                ng,
                                                episodes,
                                                l,
                                                mr,
                                                seed
                                            )
    path = os.path.join(dirpath,filepath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    df.to_csv(path)

    return df, alg.final_pop

if __name__ == "__main__":
    args = parser.parse_args()
    main(args=args)
