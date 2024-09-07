import cartpole
import lunarlander
import containers

ENIVROMENTS = {
    "cartpole": cartpole.CartpoleEvaluator,
    "lunarlander": lunarlander.LunarLanderEvaluator,
}

LAMBDA_CONTS = {
    "fitness": containers.LambdaAlgContainer,
    "fit_archiving": containers.LambdaArchivingContainer,
    "archiving": containers.LambdaArchivingNoveltyContainer,
    "novelty": containers.LambdaNoveltyAlg
}
    
DIFF_CONTS = {
    "fitness": containers.DiffAlgContainer,
    "fit_archiving": containers.DiffArchivingContainer,
    "archiving": containers.DiffArchivingNoveltyContainer,
    "novelty": containers.LambdaNoveltyAlg
}