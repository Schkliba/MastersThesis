import cartpole
import lunarlander
import containers
import diff_cont
import lambda_cont

ENIVROMENTS = {
    "cartpole": cartpole.CartpoleEvaluator,
    "lunarlander": lunarlander.LunarLanderEvaluator,
}

LAMBDA_CONTS = {
    "fitness": containers.LambdaAlgContainer,
    "fit_archiving": containers.LambdaArchivingContainer,
    "archiving": containers.LambdaArchivingNoveltyContainer,
    "novelty": containers.LambdaNoveltyAlg,
    "add_novelty": containers.LambdaAddNoveltyContainer,
    "novelty_archiving": containers.LambdaArchivingNoveltyContainer
}
    
DIFF_CONTS = {
    "fitness": containers.DiffAlgContainer,
    "fit_archiving": containers.DiffArchivingContainer,
    "archiving": containers.DiffArchivingNoveltyContainer,
    "novelty": containers.DiffNoveltyContainer,
    "add_novelty": containers.DiffAdditionNoveltyContainer,
    "novelty_archiving": containers.DiffArchivingNoveltyContainer
}

ALG_MAPPING = {
    "diff": diff_cont, 
    "lambda": lambda_cont,
}

