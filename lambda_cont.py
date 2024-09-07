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


parser = argparse.ArgumentParser()
parser.add_argument("-en", "--enviroment", help="Enviroment that is being tested", choices=["cartpole", "lunarlander", "waterworld"], default="cartpole")

parser.add_argument("-g", "--generations", help="Number of generations", type=int, default=10)
parser.add_argument("-cm", "--cross_method", help="Method of crossing individuals", choices=["mean", "uniform"], default="mean")
parser.add_argument("-cn", "--container", help="Container type", choices=["fitness", "novelty", "fit_archiving", "archiving"], default="fitness")
parser.add_argument("-cr", "--cross_rate", help="Rate of crossing individuals", type=float, default=0.5)
parser.add_argument("-cru", "--cross_uni", help="Method of crossing individuals", type=float, default=0.4)
parser.add_argument("-mr", "--mutation_rate", help="Rate of mutation", type=float, default=0.01)
parser.add_argument("-ms", "--mutation_sigma", help="Sigma of mutation", type=float, default=1)
parser.add_argument("-l", "--lambdan", help="Base population", type=int, default=30)
parser.add_argument("-m", "--mu", help="Offspring pool", type=int, default=30)
parser.add_argument("-s", "--seed", help="Seed of the random generator", type=int, default=42)
parser.add_argument("-p", "--out_path", help="Path to store output data", type=str, default="./Data/Junk/")
parser.add_argument("-exp", "--experiment", help="insication if it's experiment", action="store_true", default=False)
parser.add_argument("-e", "--episodes", help="Seed of the random generator", type=int, default=5)




def main(args: argparse.Namespace):
    ng, l, m, cr, mr, seed = args.generations, args.lambdan, args.mu, args.cross_rate, args.mutation_rate, args.seed
    enviroment = consts.ENIVROMENTS[args.enviroment](False)
    replay_env = consts.ENIVROMENTS[args.enviroment](True)

    cont_cls = consts.LAMBDA_CONTS[args.container]

    rp = lambda x: replay_env.evalutation(x, seed, 1)

    toolbox = base.Toolbox()
    if args.cross_method == "mean":
            toolbox.register("mate", ai.center_cross)
    else:
        toolbox.register("mate", ai.uniform_cross, prob_filter=args.cross_uni)
    toolbox.register("mutate", ai.mutation_func, sigma=args.mutation_sigma)

    if args.container == "fitness" or args.container == "fit_archiving":
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual",  enviroment.get_individual_base(), fitness=creator.FitnessMax)
        toolbox.register("evaluate", enviroment.evalutation, seed=seed, episodes=args.episodes)
        visual_conv = visualisation.logbook2pandas
        visual_chart = visualisation.single_run_chart
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
        toolbox.register("evaluate", cartpole.evalutation_b, seed=seed, episodes=args.episodes)
        visual_conv = visualisation.novelty_logbook2pandas
        visual_chart = visualisation.single_novelty_run_chart

    enviroment.prepare_toolbox(toolbox, creator.Individual)

    alg = cont_cls(l, m, mr, cr, seed, ng,toolbox,creator)

    alg.replay_f = rp
    alg.run()
    df = visual_conv(alg.logbook)

    dirpath = os.path.join(os.path.realpath(args.out_path), args.container,"lambda/"+args.cross_method)
    filepath = "%s,g%i,e%i,m%i,l%i,s%i.out" % (
                                                str(datetime.datetime.utcnow()), 
                                                args.generations,
                                                args.episodes,
                                                args.mu,
                                                args.lambdan,
                                                args.seed
                                            )
    path = os.path.join(dirpath,filepath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    df.to_csv(path)
    if not args.experiment:
        visual_chart(df)
    return df, alg.final_pop

if __name__ == "__main__":
    args = parser.parse_args()
    main(args=args)