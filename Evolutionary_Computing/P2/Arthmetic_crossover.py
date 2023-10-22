from numpy.random import rand, randint
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy import stats

#--------------------------------------------
def objective_function(I):

    x1 = I[0]
    x2 = I[1]
    Objective_min = (1 + math.cos(2 * 3.14 * x1 * x2)) * (math.exp(-(abs(x1) + abs(x2)) / 2))
    Objective_max = 1 / (1 + Objective_min)

    return Objective_max

#-------------------------------------------
def Arthmetic_crossover(chromosome, crossover_rate):
    offspring = [] #child

    for i in range(0, int(len(chromosome) / 2)):# N/2 --> N = pop

        p1 = chromosome[2*i]  # parent 1
        p2 = chromosome[(2*i)+1]  # parent 2

        lamda_01 = float(input('Enter lamda_01:'))
        lamda_02 = float(input('Enter lamda_02:'))

        while True:

            if lamda_01 + lamda_02 == 1:
                break
            else:
                lamda_01 = float(input('Enter lamda_01:'))
                lamda_02 = float(input('Enter lamda_02:'))


        if rand() < crossover_rate:

            # Create Children
            c1 = (lamda_01 * p1) + (lamda_02 * p2)
            c2 = (lambda_01 * p2) + (lamda_02 * p1)

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
def selection(pop, fitness, pop_size):

    next_generation = []
    best_index = np.argmax(fitness) #get index of best fitness
    next_generation.append(pop[best_index]) #keep the best

    P = [f / sum(fitness) for f in fitness] #selection prob
    index = list(range(len(pop)))


    #|pop| = 20 and pop_size = 10
    index_selected = np.random.choice(index, size=pop_size-1, replace=False, p=P) #Selected best chromosome = Roulette wheel selection


    c = 0
    for i in range(pop_size-1):
        next_generation.append(pop[index_selected[c]]) #generate next generation
        c += 1

    return next_generation

#----------------------------------------------
#Parameters of the real genetic algorithm
bounds = [[-4, 2], [-1.5, 1]]
iteration = 2
m = 2 #Variable number
pop_size = 50
crossover_rate = 0.3 # Pc
mutation_rate = 0.2 # Pm

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
best, best_eval = 0, objective_function(pop[0])


for gen in range(iteration):
    offspring = Arthmetic_crossover(pop,crossover_rate)
    offspring = mutation(offspring,mutation_rate)


    for s in offspring:
        pop.append(s)


    real_chromosome=pop
    fitness = [objective_function(d) for d in real_chromosome] # fitness value

    index = np.argmax(fitness)
    current_best = pop[index]

    best_fitness.append(1/max(fitness)-1)
    pop = selection(pop,fitness,pop_size)