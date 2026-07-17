from libs import cartpole, lunarlander, containers
import diff_cont
import lambda_cont
import numpy as np
from collections import namedtuple

Bounds = namedtuple("Bounds", ["max", "min"])
ENIVROMENTS = {
    "cartpole": cartpole.CartpoleEvaluator,
    "lunarlander": lunarlander.LunarLanderEvaluator,
}

LAMBDA_CONTS = {
    "fitness": containers.LambdaAlgContainer,
    "novelty": containers.LambdaNoveltyAlg,
    "add_novelty": containers.LambdaAddNoveltyContainer,
    "sub_novelty": containers.LambdaSubNoveltyContainer,
    "fit_archiving": containers.LambdaArchivingContainer,
    "elite_archiving": containers.LambdaArchivingEliteContainer,
    #"archiving": containers.LambdaArchivingNoveltyContainer,
    "novelty_archiving": containers.LambdaArchivingNoveltyContainer,
    "novelty_limit": containers.LambdaArchivingLimitNoveltyContainer
    
}
    
DIFF_CONTS = {
    "fitness": containers.DiffAlgContainer,
    "novelty": containers.DiffNoveltyContainer,
    "add_novelty": containers.DiffAdditionNoveltyContainer,
    "sub_novelty": containers.DiffSubNoveltyContainer,
    "fit_archiving": containers.DiffArchivingContainer,
    "elite_archiving": containers.DiffArchivingEliteContainer,
    #"archiving": containers.DiffArchivingNoveltyContainer,
    "novelty_archiving": containers.DiffArchivingNoveltyContainer,
    "novelty_limit": containers.DiffArchivingLimitNoveltyContainer,
    

}

ALG_MAPPING = {
    "diff": diff_cont, 
    "lambda": lambda_cont,
}

FITNESS_LIMIT = {
    "lunarlander":Bounds(
        max = 200,
        min = -1000
    ),
    "cartpole":Bounds(
        max = 500,
        min = 0
    )
}

NOVELTY_LIMIT = {
    "lunarlander":Bounds(
        max = 2*np.sqrt(2),
        min = 0
    ),
    "cartpole":Bounds(
        max=2*np.sqrt(2),
        min=0
    )
}

EXAMPLE_MAPPING  = {
    "lambda":"l",
    "pop":"l",
    "mu": "m",
    "start_fit_w":"fitness_weight",
    "mutation_rate":"mr",
    "cross_rate":"cr",
    "sigma": "mutation_sigma",
    "cross":"cross_uni",
    "crossmethod":"cross_method"
}