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
    "elite_archiving":containers.LambdaArchivingEliteContainer,
    "archiving": containers.LambdaArchivingNoveltyContainer,
    "novelty": containers.LambdaNoveltyAlg,
    "add_novelty": containers.LambdaAddNoveltyContainer,
    "novelty_archiving": containers.LambdaArchivingNoveltyContainer,
    "novelty_limit": containers.LambdaArchivingLimitNoveltyContainer,
    "sub_novelty": containers.LambdaSubNoveltyContainer
}
    
DIFF_CONTS = {
    "fitness": containers.DiffAlgContainer,
    "fit_archiving": containers.DiffArchivingContainer,
    "elite_archiving":containers.DiffArchivingEliteContainer,
    "archiving": containers.DiffArchivingNoveltyContainer,
    "novelty": containers.DiffNoveltyContainer,
    "add_novelty": containers.DiffAdditionNoveltyContainer,
    "novelty_archiving": containers.DiffArchivingNoveltyContainer,
    "novelty_limit": containers.DiffArchivingLimitNoveltyContainer,
    "sub_novelty":containers.DiffSubNoveltyContainer

}

ALG_MAPPING = {
    "diff": diff_cont, 
    "lambda": lambda_cont,
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