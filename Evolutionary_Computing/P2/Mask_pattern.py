from collections import Counter
from numpy.random import rand, randint
import numpy as np
import matplotlib.pyplot as plt
import math

#------------------------------------------
def objective_function(I):

    x1 = I[0]
    x2 = I[1]
    Objective_min = (1 + math.cos(2 * 3.14 * x1 * x2)) * (math.exp(-(abs(x1) + abs(x2)) / 2))
    Objective_max = 1 / (1 + Objective_min)

    return Objective_max

#---------------------------------------
#the rest of the python code can be kept the same
def crossover(pop, mask, crossover_rate):
    offspring = [] #child
    Pf = 0.3

    for i in range(0, int(len(pop) / 2)):  # N/2 --> N = pop

        p1 = pop[2 * i - 1]  # parent 1
        p2 = pop[2 * i]  # parent 2

        if rand() < crossover_rate:

            for j in range(0, len(p1[0])):

                if p1[0][j] == p2[0][j]:
                    # if p1[0][j] == mask[j] or mask[j] == 'X':
                    if p1[0][j] == mask[j] or mask[j] == 2:
                        offspring.append(p1[0][j])

                elif p1[0][j] != p2[0][j]:
                    # if mask[j] == 'X':
                    # print('j is: ', j)
                    # print('mask is: ', mask[j])
                    if mask[j] == 2:
                        Ps = (p1[1] / (p1[1] + p2[1]))

                        if Ps > 0.5:
                            offspring.append(p1[0][j])

                        else:
                            offspring.append(p2[0][j])


                elif p1[0][j] != p2[0][j]:
                    if mask[j] == 0 or mask[j] == 1:

                        if rand() > Pf:
                            offspring.append(mask[j])

                        else:
                            Ps = (p1[1] / (p1[1] + p2[1]))
                            if Ps > 0.5:
                                offspring.append(p1[0][j])

                            else:
                                offspring.append(p2[0][j])


                elif p1[0][j] == p2[0][j]:
                    if mask[j] != p1[0][j]:
                        if rand() > Pf:
                            offspring.append(mask[j])

                        else:
                            offspring.append(p1[0][j])


    return offspring

#--------------------------------------
def mutation(pop, mutation_rate):
    offspring = []

    for i in range(len(pop)):
        p1 = pop[i] #parent

        if rand() < mutation_rate:
            cp = randint(0, len(p1))
            c1 = p1
            if c1[cp] == 1:
                c1[cp] = 0 #flip
            else:
                c1[cp] = 1

            offspring.append(c1)

        else:#Transfer state
            offspring.append(p1)

    return offspring

#-------------------------------------------------------
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

#-------------------------------------------------------
def decoding(bounds, bits, chromosome):
    real_chromosome = []
    for i in range(len(bounds)):
        st, end = i * bits, (i * bits) + bits #extract the chromosome
        sub = chromosome[st:end] #type of sub is List
        chars = ''.join([str(s) for s in sub]) #convert to chars
        integer = int(chars, 2) #convert to integer
        real_value = bounds[i][0] + (integer / ((2**bits) - 1)) * (bounds[i][1] - bounds[i][0])

        real_chromosome.append(real_value)

    return real_chromosome

#-----------------------------------------------------
def create_mask(x, y):
    # x = min()
    # y = max() #positive

    print(x)
    print(y)

    # print(len(x))
    # print(len(y))

    mask = []
    for i in range(len(x)):
        if x[i] == y[i]:
            # mask.append('X') #don't care
            mask.append(2)
        else:
            mask.append(y[i])

    return mask

#-----------------------------------------------
def majority(lst):

        max_count = 0
        most_common_bit = []

        for i in range(0, len(lst)):#member
            count_0 = 0
            count_1 = 0
            for k in range(0, len(lst[i])):#bit = 16
                # print(lst[i])

                for j in range(0, len(lst)):#member
                    # print(lst[j])
                    # print(lst[j][k])

                    if lst[j][k] == 0:
                        count_0 += 1
                    else:
                        count_1 += 1

                if count_0 > count_1:
                    if count_0 > max_count:
                        max_count = count_0
                        most_common_bit.append(0)
                else:
                    if count_1 > max_count:
                        max_count = count_1
                        most_common_bit.append(1)

        return most_common_bit
        # print(most_common_bit)

#------------------------------------------------------
#Parameters of the binary genetic algorithm
bounds = [[-4, 2], [-1.5, 1]]
iteration = 4
bits = 8
pop_size = 15
crossover_rate = 0.9 # Pc
mutation_rate = 0.005 # Pm

#Initial population
pop = [randint(0, 2, bits * len(bounds)).tolist() for _ in  range(pop_size)] #number of chromosome


#main program
best_fitness = []
avg_fitness = []
best, best_eval = 0, objective_function(decoding(bounds, bits, pop[0]))

# best_so_far = []

for gen in range(1, iteration+1):

    real_chromosome = [decoding(bounds, bits, p) for p in pop]

    fitness = [objective_function(d) for d in real_chromosome] #fitness value


    index = np.argmax(fitness)
    current_best = pop[index]

    best_fitness.append(1 / max(fitness) - 1)
    pop = selection(pop, fitness, pop_size)

    pop_fit_list = list(zip(pop, fitness))
    sorted_list = sorted(pop_fit_list, key = lambda x: x[1]) #ascending


    chromosome_selected = int(pop_size / 4)
    fitness_sorted_min = sorted_list[0:chromosome_selected]
    chromosome_selected = -chromosome_selected
    fitness_sorted_max = sorted_list[chromosome_selected:]

    population_max = [t[0] for t in fitness_sorted_max] #t = (10101, 0.2)
    population_min = [t[0] for t in fitness_sorted_min]

    positive_mask = majority(population_max)
    negative_mask = majority(population_min)

    mask_ = create_mask(negative_mask, positive_mask)


    offspring = crossover(pop_fit_list, mask_, crossover_rate)
    offspring = mutation(offspring, mutation_rate)

    for s in offspring:
        pop.append(s)