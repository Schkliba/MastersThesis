import containers
import cartpole
import visualisation
import datetime
import statconf
import argparse
import numpy as np
import os

from deap import base
from deap import creator

EVAL_EPISODES = 5

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--generations", help="Number of generations", type=int, default=10)
parser.add_argument("-cm", "--cross_method", help="Method of crossing individuals", choices=["mean", "uniform"], default="mean")
parser.add_argument("-cn", "--container", help="Container type", choices=["fitness", "novelty"], default="fitness")
parser.add_argument("-cr", "--cross_rate", help="Rate of crossing individuals", type=float, default=0.5)
parser.add_argument("-cru", "--cross_uni", help="Method of crossing individuals", type=float, default=0.4)
parser.add_argument("-mr", "--mutation_rate", help="Rate of mutation", type=float, default=0.01)
parser.add_argument("-ms", "--mutation_sigma", help="Sigma of mutation", type=float, default=1)
parser.add_argument("-l", "--lambdan", help="Base population", type=int, default=30)
parser.add_argument("-m", "--mu", help="Offspring pool", type=int, default=30)
parser.add_argument("-s", "--seed", help="Seed of the random generator", type=int, default=42)
parser.add_argument("-p", "--out_path", help="Path to store output data", type=str, default="./Data/Junk/")


def main(args: argparse.Namespace):
    ng, l, m, cr, mr, seed = args.generations, args.lambdan, args.mu, args.cross_rate, args.mutation_rate, args.seed
    rp = lambda x: cartpole.evalutation(x, seed, 1, True)
    toolbox = base.Toolbox()
    if args.cross_method == "mean":
            toolbox.register("mate", cartpole.mean_cross)
    else:
        toolbox.register("mate", cartpole.uniform_cross, prob_filter=args.cross_uni)
    toolbox.register("mutate", cartpole.mutation_func, sigma=args.mutation_sigma)

    if args.container == "fitness":
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessMax)
        toolbox.register("evaluate", cartpole.evalutation, seed=seed, episodes=EVAL_EPISODES)
        alg = containers.LambdaAlgContainer(l, m, mr, cr, seed, ng,toolbox,creator)
        visual_conv = visualisation.logbook2pandas
        visual_chart = visualisation.single_run_chart

    if args.container == "novelty":
        creator.create("FitnessNovelty", base.Fitness, weights=(1.0, ))
        creator.create("FitnessTrue", base.Fitness, weights=(1.0, )) 
        creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessNovelty , fitness2=creator.FitnessTrue)
        toolbox.register("evaluate", cartpole.evalutation_b, seed=seed, episodes=EVAL_EPISODES)
        alg = containers.LambdaNoveltyAlg(l,m,mr,cr,seed,ng, toolbox, creator)
        visual_conv = visualisation.novelty_logbook2pandas
        visual_chart = visualisation.single_novelty_run_chart

    alg.replay_f = rp
    alg.run()
    df = visual_conv(alg.logbook)
    dirpath = os.path.join(os.path.realpath(args.out_path), args.container,"lambda/"+args.cross_method)
    path = os.path.join(dirpath,str(datetime.datetime.utcnow())+".out")
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    df.to_csv(path)
    visual_chart(df)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args=args)