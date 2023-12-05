import containers as con
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
parser.add_argument("-cr", "--cross_rate", help="Rate of crossing individuals", type=float, default=0.5)
parser.add_argument("-mr", "--mutation_rate", help="Rate of mutation", type=float, default=0.01)
parser.add_argument("-l", "--lambdan", help="Base population", type=int, default=30)
parser.add_argument("-m", "--mu", help="Offspring pool", type=int, default=30)
parser.add_argument("-s", "--seed", help="Seed of the random generator", type=int, default=42)

args = parser.parse_args()

ng, l, m, cr, mr, seed = args.generations, args.lambdan, args.mu, args.cross_rate, args.mutation_rate, args.seed
rp = lambda x: cartpole.evalutation(x, seed, 1, True)

creator.create("FitnessNovelty", base.Fitness, weights=(1.0, ))
#creator.create("Behavior", base.Fitness, weights=(1.0, 1.0)) #end position & angle
creator.create("FitnessTrue", base.Fitness, weights=(1.0, )) 

creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessNovelty , fitness2=creator.FitnessTrue)#, behaviour=creator.Behavior)

toolbox = base.Toolbox()
toolbox.register("evaluate", cartpole.evalutation_b, seed=seed, episodes=EVAL_EPISODES)
toolbox.register("mate", cartpole.cartover)
toolbox.register("mutate", cartpole.mutcartion, sigma=1)

alg = con.LambdaNoveltyAlg(l,m,mr,cr,seed,ng, toolbox, creator)
alg.replay_f = rp
alg.run()
df = visualisation.logbook2pandas(alg.logbook)
df.to_csv("./Data/lambda_nov_"+str(datetime.datetime.utcnow())+".out")
visualisation.single_run_chart(df)