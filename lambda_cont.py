import containers
import cartpole
import numpy as np
from deap import base
from deap import creator

EVAL_EPISODES = 10

paramters =[
    (10, 30, 30, 0.4, 0.01, 42),
    (10,30, 30, 0.4, 0.05, 42),
    (10,30, 30, 0.4, 0.1, 42),
    (10,30, 30, 0.4, 0.15, 42)
]
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessMax)

for p in paramters:
    ng, l, m, cr, mr, seed = p
    rp = lambda x: cartpole.evalutation(x, seed, 1, True)
    toolbox = base.Toolbox()
    toolbox.register("evaluate", cartpole.evalutation, seed=seed, episodes=EVAL_EPISODES)


    toolbox.register("mate", cartpole.cartover)
    toolbox.register("mutate", cartpole.mutcartion, sigma=1)
    alg = containers.LambdaAlgContainer(l, m, mr, cr, seed, ng,toolbox,creator,rp)
    alg.run()
    print(alg.logbook)
