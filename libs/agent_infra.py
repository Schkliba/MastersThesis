import keras
import numpy as np
import copy 

class Evaluator:
    def __init__(self):
        self.enviroment = None
        self.behavior_space_f = lambda x: x

    @staticmethod
    def gen_pop(pop_size, ind_f):
        return [ind_f() for i in range(pop_size)]

    def evalutation_b(self, individual: keras.Model, seed:int, episodes:int) -> float:
        """
        Returns the average score achieved on the given number of episodes and normalised behaviour.
        """
        # Create the environment
        self.enviroment.reset(seed=seed)

        # Evaluate the episodes
        total_score = 0
        for episode in range(episodes):
            observation, score, done = self.enviroment.reset()[0], 0, False
            while not done:
                prediction = individual(observation[np.newaxis])[0].numpy()
                action = np.argmax(prediction)

                observation, reward, terminated, truncated, info = self.enviroment.step(action)
                score += reward
                done = terminated or truncated

            total_score += score
        return total_score / episodes, [observation[0]/4.8, observation[2]/0.418]

    def evalutation(self, individual: keras.Model, seed:int, episodes:int) -> float:
        fit, b = self.evalutation_b(individual, seed, episodes)
        return fit,

class Player(keras.Model):
    def __deepcopy__(self, memo):
            new = self.__class__(self)
            new.mutable_layer = copy.deepcopy(self.mutable_layer, memo)
            return new
            
    def __call__(self, inputs):
        self.d1:keras.layers.Dense
        x = self.d1(inputs)
        x = self.mutable_layer(x)
        return self.d_out(x)
    
    def get_agent_weights(self):
        return self.mutable_layer.get_weights()

    def set_agent_weights(self, weights):
        self.mutable_layer.set_weights(weights)

#Cross overs
def switcheroo(ind1, ind2):
    return ind2, ind1 

def center_cross(ind1:Player, ind2:Player):
    u = ind1.get_agent_weights()
    v = ind2.get_agent_weights()
    f = [(u[i] + v[i])/2 for i in range(len(u)) ]
    ind1.set_agent_weights(f)
    ind2.set_agent_weights(f)
    return ind1, ind2

def uniform_cross(ind1:Player, ind2:Player, prob_filter):
    u = ind1.get_agent_weights()
    v = ind2.get_agent_weights()
    newind1 = []
    newind2 = []
    for i, mat in enumerate(u):
        cpoints = np.random.random(mat.shape) < prob_filter
        n = np.where(cpoints, mat, v[i])
        newind1.append(n)
        n = np.where(cpoints, v[i], mat)
        newind2.append(n)
    ind1.set_agent_weights(newind1)
    ind2.set_agent_weights(newind2)
    return ind1, ind2


# Mutations
def mutation_func(individual:Player, sigma):
    ws = individual.get_agent_weights()
    rand = [w + np.random.normal(0, sigma, w.shape) for w in ws]
    individual.set_agent_weights(rand)
    return individual,

# Defferentials
def tri_op(base:Player, diff1:Player, diff2:Player, alpha:float):
    a = base.get_agent_weights()
    b = diff1.get_agent_weights()
    c = diff2.get_agent_weights()
    n = [(a[i] + alpha * (b[i] - c[i])) for i in range(len(a))]
    #print(n)
    return n

def recombine(ind:Player, mats, prob_filter):
    newind = []
    og = ind.get_agent_weights()
    for i, mat in enumerate(mats):
        cpoints = np.random.random(mat.shape) < prob_filter
        n = np.where(cpoints, mat, og[i])
        newind.append(n)
    ind.set_agent_weights(newind)
    return ind

