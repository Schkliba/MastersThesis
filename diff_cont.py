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

EVAL_EPISODES = 3

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--generations", help="Number of generations", type=int, default=10)
parser.add_argument("-cr", "--cross_rate", help="Rate of crossing individuals", type=float, default=0.6)
parser.add_argument("-cn", "--container", help="Container type", choices=["fitness", "novelty", "archiving", "fit_archiving"], default="fitness")
parser.add_argument("-mr", "--mutation_rate", help="Rate of mutation", type=float, default=0.9)
parser.add_argument("-p", "--pop", help="Base population", type=int, default=7)
parser.add_argument("-s", "--seed", help="Seed of the random generator", type=int, default=42)
parser.add_argument("-pt", "--out_path", help="Path to store output data", type=str, default="./Data/Junk/")


def main(args:argparse.Namespace):
    ng, l, cr, mr, seed = args.generations, args.pop, args.cross_rate, args.mutation_rate, args.seed
    rp = lambda x: cartpole.evalutation(x, seed, 1, True)
    toolbox = base.Toolbox()
    toolbox.register("triOp", cartpole.tri_op, alpha=mr)
    toolbox.register("recombine", cartpole.recombine, prob_filter=cr)

    if args.container == "fitness":
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessMax)
        toolbox.register("evaluate", cartpole.evalutation, seed=seed, episodes=EVAL_EPISODES)
        alg = containers.DiffAlgContainer(l,toolbox, seed, ng,creator)
        visual_conv = visualisation.logbook2pandas
        visual_chart = visualisation.single_run_chart

    if args.container == "novelty":
        creator.create("FitnessNovelty", base.Fitness, weights=(1.0, ))
        creator.create("FitnessTrue", base.Fitness, weights=(1.0, )) 
        creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessNovelty , fitness2=creator.FitnessTrue)

        toolbox.register("evaluate", cartpole.evalutation_b, seed=seed, episodes=EVAL_EPISODES)
        alg = containers.DiffNoveltyContainer(l,toolbox, seed, ng,creator)
        visual_conv = visualisation.novelty_logbook2pandas
        visual_chart = visualisation.single_novelty_run_chart

    if args.container == "fit_archiving":
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessMax)
        toolbox.register("evaluate", cartpole.evalutation, seed=seed, episodes=EVAL_EPISODES)
        alg = containers.DiffArchivingContainer(l,toolbox, seed, ng,creator)
        visual_conv = visualisation.logbook2pandas
        visual_chart = visualisation.single_run_chart

    if args.container == "archiving":
        creator.create("FitnessNovelty", base.Fitness, weights=(1.0, ))
        creator.create("FitnessTrue", base.Fitness, weights=(1.0, )) 
        creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessNovelty , fitness2=creator.FitnessTrue, behaviour=None)

        toolbox.register("evaluate", cartpole.evalutation_b, seed=seed, episodes=EVAL_EPISODES)

        alg = containers.DiffNoveltyArchivingContainer(l,toolbox, seed, ng,creator)
        visual_conv = visualisation.novelty_logbook2pandas
        visual_chart = visualisation.single_novelty_run_chart


    alg.replay_f = rp
    alg.run()
    df = visual_conv(alg.logbook)
    dirpath = os.path.join(os.path.realpath(args.out_path), args.container,"diff")
    path = os.path.join(dirpath,str(datetime.datetime.utcnow())+".out")
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    df.to_csv(path)
    visual_chart(df)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args=args)
