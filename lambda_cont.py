import constants as consts
import visualisation
import libs.agent_infra as ai
import datetime
import statconf
import argparse
import numpy as np
import os

from deap import base
from deap import creator
from deap import algorithms

parser = argparse.ArgumentParser()
parser.add_argument("-en", "--enviroment", help="Enviroment that is being tested", choices=["cartpole", "lunarlander", "waterworld"], default="cartpole")

parser.add_argument("-g", "--generations", help="Number of generations", type=int, default=10)
parser.add_argument("-cm", "--cross_method", help="Method of crossing individuals", choices=["mean", "uniform"], default="mean")
parser.add_argument("-cn", "--container", help="Container type", choices=["fitness", "novelty", "fit_archiving", "archiving"], default="fitness")
parser.add_argument("-cr", "--cross_rate", help="Rate of crossing individuals", type=float, default=0.5)
parser.add_argument("-cru", "--cross_uni", help="Method of crossing individuals", type=float, default=0.4)
parser.add_argument("-mr", "--mutation_rate", help="Rate of mutation", type=float, default=0.01)
parser.add_argument("-ms", "--mutation_sigma", help="Sigma of mutation", type=float, default=1)
parser.add_argument("-p", "--lambdan", help="Base population", type=int, default=30)
parser.add_argument("-m", "--mu", help="Offspring pool", type=int, default=30)
parser.add_argument("-s", "--seed", help="Seed of the random generator", type=int, default=42)
parser.add_argument("-pa", "--out_path", help="Path to store output data", type=str, default="./Data/Junk/")
parser.add_argument("-exp", "--experiment", help="insication if it's experiment", action="store_true", default=False)
parser.add_argument("-e", "--episodes", help="Seed of the random generator", type=int, default=5)
parser.add_argument("-fw", "--fit_weight", help="Initial weight given to the fitnes", type=float, default=0.2)




def main(args: argparse.Namespace):
    ng, l, m, cr, mr, seed = args.generations, args.lambdan, args.mu, args.cross_rate, args.mutation_rate, args.seed
    container = args.container
    cross_method = args.cross_method
    environment = args.environment
    episodes = args.episodes
    cross_uni = args.cross_uni
    mutation_sigma = args.mutation_sigma
    out_path = args.out_path

    df, pop= argumented_function(
        enviroment=environment,
        cross_method=cross_method,
        container=container,
        ng=ng, l=l, m=m, cr=cr, mr=mr, episodes=episodes, cross_uni=cross_uni, mutation_sigma=mutation_sigma,
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
        cross_method:str, 
        container:str,  
        ng:int, 
        l:int, 
        m:int, 
        mr:float, 
        cr:float, 
        mutation_sigma:float = 1.0, 
        cross_uni=0.5, 
        episodes:int = 3, 
        seed:int = 42,
        archiving_period = 2,
        archive_batch = 1,
        fitness_weight = 0.2,
        out_path = "./Data/Junk/"
    ):
    replay_env = consts.ENIVROMENTS[env](True)
    enviroment = consts.ENIVROMENTS[env](False)
    rp = lambda x: replay_env.evalutation(x, seed, 1)

    toolbox = base.Toolbox()
    if cross_method == "mean":
        toolbox.register("mate", ai.center_cross)
    else:
        toolbox.register("mate", ai.uniform_cross, prob_filter=cross_uni)
    toolbox.register("mutate", ai.mutation_func, sigma=mutation_sigma)

    if container == "fitness" or container == "fit_archiving":
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual",  enviroment.get_individual_base(), fitness=creator.FitnessMax)
        toolbox.register("evaluate", enviroment.evalutation, seed=seed, episodes=episodes)
        visual_conv = visualisation.logbook2pandas
    elif container == "add_novelty":
        creator.create("FitnessNovelty", base.Fitness, weights=(1.0, ))
        creator.create("FitnessTrue", base.Fitness, weights=(1.0, ))
 
        creator.create(
            "Individual", 
            enviroment.get_individual_base(), 
            fitness=creator.FitnessNovelty, 
            fitness2=creator.FitnessTrue,
            behavior=None
        )
        
        toolbox.register("evaluate", enviroment.evalutation_b, seed=seed, episodes=episodes)
        visual_conv = visualisation.novelty_logbook2pandas

    else:
        creator.create("FitnessNovelty", base.Fitness, weights=(1.0, ))
        creator.create("FitnessTrue", base.Fitness, weights=(1.0, )) 
        creator.create(
            "Individual", 
            enviroment.get_individual_base(), 
            fitness=creator.FitnessNovelty, 
            fitness2=creator.FitnessTrue,
            behavior=None
        )
        toolbox.register("evaluate", enviroment.evalutation_b, seed=seed, episodes=episodes)
        visual_conv = visualisation.novelty_logbook2pandas

    enviroment.prepare_toolbox(toolbox, creator.Individual)
    cont_cls = consts.LAMBDA_CONTS[container]
    if container == "fit_archiving":
        cont_cls :consts.containers.LambdaArchivingContainer
        alg = cont_cls(
            l, 
            m, 
            mr, 
            cr, 
            seed, 
            ng,
            toolbox,
            creator, 
            archiving_period=archiving_period, 
            store_batch=archive_batch
        )
    elif container == "add_novelty":
        alg = cont_cls(
            l, 
            m, 
            mr, 
            cr, 
            seed, 
            ng,
            toolbox,
            creator,  
            fit_w = fitness_weight
        )
    else:
        alg = cont_cls(l, m, mr, cr, seed, ng,toolbox,creator)

    alg.replay_f = rp
    alg.run()
    df = visual_conv(alg.logbook)
    dirpath = os.path.join(os.path.realpath(out_path), container,"lambda/"+cross_method)
    filepath = "%s,%s,g%i,e%i,m%i,l%i,s%i.out" % (
                                                str(datetime.datetime.utcnow()),
                                                env, 
                                                ng,
                                                episodes,
                                                m,
                                                l,
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