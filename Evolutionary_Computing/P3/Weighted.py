from numpy.random import rand, randint
import numpy as np
import matplotlib.pyplot as plt
import math
import random

#--------------------------------------------
def objective_function_01(I):

    x1 = I[0]
    x2 = I[1]
    Objective_min = (1 + math.cos(2 * 3.14 * x1 * x2)) * (math.exp(-(abs(x1) + abs(x2)) / 2))
    Objective_max = 1 / (1 + Objective_min)

    return Objective_max


def objective_function_02(I):

    x1 = I[0]
    x2 = I[1]
    Objective_min = (1 + math.tan(2 * 3.14 * x1 * x2)) * (math.exp(-(abs(x1) + abs(x2)) / 2))
    Objective_max = 1 / (1 + Objective_min)

    return Objective_max

#-------------------------------------------
def crossover(chromosome, crossover_rate):
    offspring = [] #child

    for i in range(0, int(len(chromosome) / 2)):# N/2 --> N = pop

        p1 = chromosome[2*i]  # parent 1
        p2 = chromosome[(2*i)+1]  # parent 2

        if rand() < crossover_rate:

            # Create Children
            c1 = p1[0:1] + p2[1:2]
            c2 = p2[0:1] + p1[1:2]

            offspring.append(c1)
            offspring.append(c2)

        else:  # Transfer state
            offspring.append(p1)
            offspring.append(p2)

        return offspring

#-----------------------------------------------
def mutation(pop,mutation_rate):
    offspring = []
    mu, sigma = 0, 0.01

    for i in range(int(len(pop))):
        p1 = pop[i] # parent

        if rand() < mutation_rate:

            c = np.random.randint(0, len(p1))
            c1 = p1
            p_01 = random.uniform(mu, sigma)
            c1[c] = c1[c] + p_01
            offspring.append(c1)

        else:
            offspring.append(p1)

    return offspring

#--------------------------------------------
def selection(pop, archive, fitness_, p):
    next_generation = []

    fitness_min = min(fitness_)

    sum_ = 0
    for i in fitness_:
        sum_ += i - fitness_min

    prob = []
    for i in fitness_:
        p = ((i - fitness_min) / sum_)
        prob.append(p)

    best_index = np.argmax(fitness_)  # get index of best fitness
    next_generation.append(pop[best_index])  # keep the best

    index = list(range(len(pop)))

    prob_array = np.array(prob)
    index_selected = np.random.choice(index, size=int(len(pop) - p), replace=False, p=prob_array)

    # add from active pop to next gen
    c = 0
    for i in range(len(pop) - 1):
        next_generation.append(pop[index_selected[c]])  # generate next generation
        c += 1

    # add from archive to next gen
    if len(archive) > 2:
        c = 0
        random_ = np.random.randint(1, len(archive)-1, size=(p,))
        for i in range(p):
            next_generation.append(archive[random_[c]])
            c += 1

    return next_generation

#----------------------------------------------
#Parameters of the real genetic algorithm
bounds = [[-4, 2], [-1.5, 1]]
iteration = 50 # T = 50
m = 2 #Variable number
pop_size = 50
crossover_rate = 0.3 # Pc
pm_high = 0.01
pm_low = 0.001
k = 2
p = 3 # a parameter for select(N-p)


#Initial population
pop = []
for i in range(0, pop_size):
    pop_01 = []
    p1 = bounds[0][0] + (bounds[0][1] - bounds[0][0]) * random.uniform(0, 1)
    p2 = bounds[1][0] + (bounds[1][1] - bounds[1][0]) * random.uniform(0, 1)
    pop_01.append(p1)
    pop_01.append(p2)
    pop.append(pop_01)


#main program
best_fitness = []
avg_fitness = []

r = [random.random() for i in range(0, k)]
wi = []
for i in range(0, k):
    wi.append(r[i] / sum(r))


for t in range(iteration):
    offspring = crossover(pop,crossover_rate)
    pm = pm_high - ((pm_high - pm_low) * (t/iteration)) #calcute mutation rate

    offspring = mutation(offspring,pm)


    for s in offspring:
        pop.append(s)

    # --------------------------------------
    real_chromosome = pop
    fitness_01 = [objective_function_01(d) for d in real_chromosome] # fitness value
    fitness_02 = [objective_function_02(d) for d in real_chromosome]

    fitness_ = []
    s = 0
    for j in real_chromosome:
        f1 = wi[0] * objective_function_01(j)
        f2 = wi[1] * objective_function_02(j)
        s += f1
        s += f2
        fitness_.append(s)

    # --------------------------------------
    Archive = []
    if fitness_01.index(max(fitness_01)) == fitness_02.index(max(fitness_02)):
        Archive.append(fitness_01.index(max(fitness_01)))

    # --------------------------------------
    pop = selection(pop, Archive, fitness_, p)
    # avg_fitness.append(np.mean(fitness))

print('Done!')

#-------Plot----------------------------
# fig = plt.figure()
# plt.plot(avg_fitness)
# fig.suptitle('Average fitness over Generations')
# plt.xlabel('Generation')
# plt.ylabel('Fitness Value')
# plt.show()
