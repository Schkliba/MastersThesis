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

parser.add_argument("-g", "--generations", help="Number of generations", type=int, default=10)
parser.add_argument("-cr", "--cross_rate", help="Rate of crossing individuals", type=float, default=0.6)
parser.add_argument("-cn", "--container", help="Container type", choices=["fitness", "novelty", "archiving", "fit_archiving"], default="fitness")
parser.add_argument("-en", "--enviroment", help="Enviroment type", choices=["cartpole", "lunarlander", "waterworld"], default="cartpole")

parser.add_argument("-mr", "--mutation_rate", help="Rate of mutation", type=float, default=0.9)
parser.add_argument("-p", "--pop", help="Base population", type=int, default=7)
parser.add_argument("-s", "--seed", help="Seed of the random generator", type=int, default=42)
parser.add_argument("-e", "--episodes", help="Seed of the random generator", type=int, default=5)
parser.add_argument("-pt", "--out_path", help="Path to store output data", type=str, default="./Data/Junk/")
parser.add_argument("-exp", "--experiment", help="insication if it's experiment", action="store_true", default=False)


def main(args:argparse.Namespace):
    ng, l, cr, mr, seed = args.generations, args.pop, args.cross_rate, args.mutation_rate, args.seed
    enviroment = consts.ENIVROMENTS[args.enviroment](False)
    replay_env = consts.ENIVROMENTS[args.enviroment](True)
    cont_cls = consts.DIFF_CONTS[args.container]
    rp = lambda x: replay_env.evalutation(x, seed, 1)

    toolbox = base.Toolbox()
    toolbox.register("triOp", ai.tri_op, alpha=mr)
    toolbox.register("recombine", ai.recombine, prob_filter=cr)

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
                        fitness=creator.FitnessNovelty , 
                        fitness2=creator.FitnessTrue, 
                        behaviour=None
        )
        toolbox.register("evaluate", enviroment.evalutation_b, seed=seed, episodes=args.episodes)
        visual_conv = visualisation.novelty_logbook2pandas
        visual_chart = visualisation.single_novelty_run_chart

    enviroment.prepare_toolbox(toolbox, creator.Individual)

    alg = cont_cls(l,toolbox, seed, ng,creator)

    alg.replay_f = rp
    alg.run()
    df = visual_conv(alg.logbook)
    dirpath = os.path.join(os.path.realpath(args.out_path), args.container,"diff")
    filepath = "%s,g%i,e%i,pop%i,mr%.4f,s%i.out" % (
                                                str(datetime.datetime.utcnow()), 
                                                args.generations,
                                                args.episodes,
                                                args.pop,
                                                args.mutation_rate,
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
