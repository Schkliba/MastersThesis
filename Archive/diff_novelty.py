import containers
import cartpole
import visualisation
import datetime
import statconf
import argparse
import numpy as np

from deap import base
from deap import creator

EVAL_EPISODES = 10

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--generations", help="Number of generations", type=int, default=10)
parser.add_argument("-cr", "--cross_rate", help="Rate of crossing individuals", type=float, default=0.6)
parser.add_argument("-mr", "--mutation_rate", help="Rate of mutation", type=float, default=0.9)
parser.add_argument("-p", "--pop", help="Base population", type=int, default=7)
parser.add_argument("-s", "--seed", help="Seed of the random generator", type=int, default=42)
parser.add_argument("-p", "--out_path", help="Path to store output data", type=str, default="./Data/Junk/")


args = parser.parse_args()

creator.create("FitnessNovelty", base.Fitness, weights=(1.0, ))
#creator.create("Behavior", base.Fitness, weights=(1.0, 1.0)) #end position & angle
creator.create("FitnessTrue", base.Fitness, weights=(1.0, )) 

creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessNovelty , fitness2=creator.FitnessTrue)#, behaviour=creator.Behavior)
ng, l, cr, mr, seed = args.generations, args.pop, args.cross_rate, args.mutation_rate, args.seed
rp = lambda x: cartpole.evalutation(x, seed, 1, True)

toolbox = base.Toolbox()
toolbox.register("evaluate", cartpole.evalutation_b, seed=seed, episodes=EVAL_EPISODES)
toolbox.register("triOp", cartpole.cartdiff, alpha=mr)
toolbox.register("recombine", cartpole.cartrecomb, prob_filter=cr)


alg = containers.DiffNoveltyContainer(l,toolbox, seed, ng,creator)
alg.replay_f = rp
alg.run()
df = visualisation.novelty_logbook2pandas(alg.logbook)
df.to_csv(args.out_path + "diff_nov_"+str(datetime.datetime.utcnow())+".out")
visualisation.single_novelty_run_chart(df)
