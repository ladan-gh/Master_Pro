from numpy.random import rand, randint
import numpy as np
import matplotlib.pyplot as plt
import math
import random

#---------------------------------------------
def ackley(x):#multimodal function = fitness
    n = len(x)
    sum_sq = sum([xi**2 for xi in x])
    cos_sum = sum([math.cos(2 * math.pi * xi) for xi in x])

    return -20 * math.exp(-0.2 * math.sqrt(sum_sq / n)) - math.exp(cos_sum / n) + 20 + math.e

#-----------------------------------
def distance(pop):# M function

    list_ = []
    for i in pop:
        list_.append(distance_function(i))

    x = np.sum(list_)
    return x

#------------------------------------
def distance_between_two_chromosome(ch1, ch2):
    d1 = (ch1[0] - ch2[0]) ** 2
    d2 = (ch1[1] - ch2[1]) ** 2

    dist = d1 + d2
    return dist

#-----------------------------------
def distance_function(chromosome_01): # Sh()
    sharing_redius = 0.2

    for j in pop:
        dist_ = distance_between_two_chromosome(chromosome_01, j)

        if dist_ < sharing_redius:
            return (1 - (dist_ / sharing_redius))
        else:
            return 0

#-----------------------------------
def shared_merit(chromosome, pop_):

    result = ackley(chromosome) / distance(pop_)
    return result

#-------------------------------------------
def crossover(chromosome_fit, crossover_rate):
    offspring = [] #child

    chromosome = [t[0] for t in chromosome_fit]
    fit = [t[1] for t in chromosome_fit]

    for i in range(0, int(len(chromosome_fit) / 2)):# N/2 --> N = pop

        p1 = chromosome[2*i]  # parent 1
        p2 = chromosome[(2*i)+1]  # parent 2

        f1 = fit[2*i]
        f2 = fit[(2*i)+1]

        if rand() < crossover_rate:

            # Create Children
            c1 = p1[0:1] + p2[1:2]
            c2 = p2[0:1] + p1[1:2]

            f_child1 = ackley(c1)
            f_child2 = ackley(c2)

            if f_child1 > f1:
                if f_child1 > f2:
                    offspring.append(c1)

                else:
                    offspring.append(p1)
                    offspring.append(p2)


            elif f_child2 > f1:
                if f_child2 > f2:
                    offspring.append(c2)

                else:
                    offspring.append(p1)
                    offspring.append(p2)

        else:  # Transfer state
            offspring.append(p1)
            offspring.append(p2)

        return offspring

#-----------------------------------------------
def mutation(pop,mutation_rate):# Because this code for real genetic
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
iteration = 50
m = 2 #Variable number
pop_size = 50
crossover_rate = 0.3 # Pc
mutation_rate = 0.2 # Pm


#Initial population
pop = []
for i in range(0, pop_size):# page 57 in book
    pop_01 = []
    p1 = bounds[0][0] + (bounds[0][1] - bounds[0][0]) * random.uniform(0, 1)
    p2 = bounds[1][0] + (bounds[1][1] - bounds[1][0]) * random.uniform(0, 1)
    pop_01.append(p1)
    pop_01.append(p2)
    pop.append(pop_01)


#main program
best_fitness = []
avg_fitness = []
# best, best_eval = 0, objective_function(pop[0])


for gen in range(iteration):

    fitness_ = [ackley(p) for p in pop]# fitness value

    index = np.argmax(fitness_)
    current_best = pop[index]

    best_fitness.append(1 / max(fitness_) - 1)
    pop = selection(pop, fitness_, pop_size)

    shared_merit_list = [shared_merit(p, pop) for p in pop]

    pop_fit = list(zip(pop, shared_merit_list))

    offspring = crossover(pop_fit, crossover_rate)
    offspring = mutation(offspring, mutation_rate)

    for s in offspring:
        pop.append(s)
