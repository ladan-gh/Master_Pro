from numpy.random import rand, randint
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
import random

#------------------------------------------
def objective_function(I):

    x1 = I[0]
    x2 = I[1]
    Objective_min = (1 + math.cos(2 * 3.14 * x1 * x2)) * (math.exp(-(abs(x1) + abs(x2)) / 2))
    Objective_max = 1 / (1 + Objective_min)

    return Objective_max

#---------------------------------------
#the rest of the python code can be kept the same
def crossover_mask(pop, mask, crossover_rate):
    offspring = [] #child
    Pf = 0.3

    for i in range(0, int(len(pop) / 2)):  # N/2 --> N = pop

        p1 = pop[2 * i - 1]  # parent 1
        p2 = pop[2 * i]  # parent 2

        if rand() < crossover_rate:

            for j in range(0, len(p1[0])):

                if p1[0][j] == p2[0][j]:
                    if p1[0][j] == mask[j] or mask[j] == 'X':
                        offspring.append(p1[0][j])
                        # offspring[j] = p1[0][j]


                elif p1[0][j] != p2[0][j]:
                    if mask[j] == 'X':
                        Ps = (p1[1] / (p1[1] + p2[1]))

                        if Ps > 0.5:
                            offspring.append(p1[0][j])
                            # offspring[j] = p1[0][j]

                        else:
                            offspring.append(p2[0][j])
                            # offspring[j] = p2[0][j]


                elif p1[0][j] != p2[0][j]:
                    if mask[j] == 0 or mask[j] == 1:

                        if rand() > Pf:
                            offspring.append(mask[j])
                            # offspring[j] = mask[j]

                        else:
                            Ps = (p1[1] / (p1[1] + p2[1]))
                            if Ps > 0.5:
                                offspring.append(p1[0][j])
                                # offspring[j] = p1[0][j]

                            else:
                                offspring.append(p2[0][j])
                                # offspring[j] = p2[0][j]


                elif p1[0][j] == p2[0][j]:
                    if mask[j] != p1[0][j]:
                        if rand() > Pf:
                            # offspring.append(mask[j])
                            offspring[j] = mask[j]

                        else:
                            # offspring.append(p1[0][j])
                            offspring[j] = p1[0][j]

    return offspring

#---------------------------------------
#the rest of the python code can be kept the same
def crossover(pop, crossover_rate):
    offspring = [] #child

    for i in range(0, int(len(pop) / 2)): # N/2 --> N = pop

        p1 = pop[2*i-1].copy() #parent 1
        p2 = pop[2 * i].copy() #parent 2

        if rand() < crossover_rate:
            cp = randint(1, len(p1)-1, size=1) # one random cutting points

            #Create Children
            c1 = p1[:cp[0]] + p2[cp[0]:]
            c2 = p2[:cp[0]] + p1[cp[0]:]

            offspring.append(c1)
            offspring.append(c2)

        else: #Transfer state
            offspring.append(p1)
            offspring.append(p2)

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
def Roulette_wheel_selection(pop, fitness, pop_size):

    next_generation = []
    best_index = np.argmax(fitness) #get index of best fitness
    next_generation.append(pop[best_index]) #keep the best

    P = [f / sum(fitness) for f in fitness] #selection prob

    index = list(range(len(pop)))
    # index = [0, 1, 2, 3, ...., 9]

    #|pop| = 20 and pop_size = 10
    index_selected = np.random.choice(index, size=pop_size-1, replace=False, p=P) #Selected best chromosome = Roulette wheel selection

    for i in range(pop_size-1):
        next_generation.append(pop[index_selected[i]]) #generate next generation

    return next_generation

#-----------------------------------------
def sus_selection(population, fitness, num_parents):
    # Calculate selection probability
    selection_prob = fitness / np.sum(fitness)

    # Calculate cumulative probability
    cum_prob = np.cumsum(selection_prob)

    # Calculate the distance between the pointers
    dist = 1.0 / num_parents

    # Generate starting point
    start_point = np.random.uniform(low=0, high=dist)

    # Generate pointers
    pointers = np.arange(start_point, 1.0, step=dist)

    # Initialize parent indices
    parent_indices = np.zeros(num_parents, dtype=int)

    # Select parents
    i, j = 0, 0
    while i < num_parents:
        if pointers[i] <= cum_prob[j]:
            parent_indices[i] = j
            i += 1
        else:
            j += 1

    # Return selected parents
    return population[parent_indices]
#------------------------------------------------------
def Boltzman(pop, fitness, pop_size, T):

    next_generation = []
    best_index = np.argmax(fitness)  # get index of best fitness
    next_generation.append(pop[best_index])  # keep the best

    fit_scale = np.exp(fitness_array / T)

    fitness_array = np.array(fitness)
    P = pop_size * (fit_scale / np.sum(np.exp(fitness_array / T)))#نرخ انتظار

    index = list(range(len(pop)))

    # |pop| = 20 and pop_size = 10
    index_selected = np.random.choice(index, size=pop_size - 1, replace=False, p=P)

    c = 0
    for i in range(pop_size - 1):
        next_generation.append(pop[index_selected[c]])  # generate next generation
        c += 1

    return next_generation

#-------------------------------------------------------
def Ranked(pop, sorted_index, pop_size):

    next_generation = []
    best_index = np.argmax(fitness)  # get index of best fitness
    next_generation.append(pop[best_index])  # keep the best

    while True:

        print('pop_size is: ', pop_size)
        q0 = float(input('Enter value for q between 0 and 1/pop_size :'))  # 0
        q = float(input('Enter value for q0 between 1/pop_size and 2/pop_size :'))  # 2/N

        if q + q0 == 2 / pop_size:
            break

    # q0 = 0
    # q = 2 / pop_size

    P = []
    index_ = np.array(sorted_index)

    for i in range(0, len(sorted_index)):
        P.append((q - (q-q0)) *  ((index_[i] - 1) / (pop_size - 1)))# selection prob

    index = list(range(len(pop)))# index = [0,1,2,....,N-1]

    # |pop| = 20 and pop_size = 10
    index_selected = np.random.choice(index, size=pop_size - 1, replace = False, p=P)#[0,2,1,3,....]

    c = 0
    for i in range(pop_size - 1):
        next_generation.append(pop[index_selected[c]])  # generate next generation
        c += 1

    return next_generation

#------------------------------------------------------
def Tornoment(pop, pop_size):
    # k = 2 (Binary)
    p = []
    # pop = [(pop1, fit1), (pop2, fit2), .....]

    for i in range(0, pop_size):

        cp = random.randint(1, pop_size-1)
        k = pop[cp]
        p1 = k[0]
        fitness1 = k[1]


        cp = random.randint(1, pop_size-1)
        k = pop[cp]
        p2 = k[0]
        fitness2 = k[1]

        if rand() < (1 / (1 + np.exp(-(fitness1 - fitness2) / T))):
            p.append(p1)
        else:
            p.append(p2)

    return p
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
#------------------------------------------------------
def create_mask(x, y):
    # x = min()
    # y = max()

    mask = []
    for i in range(len(x)):
        if x[i] == y[i]:
            mask.append('X') #don't care
        else:
            mask.append(y[i])

    return mask

#-----------------------------------------------
def majority(binary):
    integer_value = []

    for i in range(len(binary)):
        integer_value.append(''.join(str(bit) for bit in binary[i]))

    # count the number of occurrences of each binary value
    binary_counts = Counter(integer_value)

    # get the binary value with the highest count
    majority_vote = binary_counts.most_common(1)[0][0]

    return majority_vote

#-----------------------------------------------------
#Parameters of the binary genetic algorithm
bounds = [[-4, 2], [-1.5, 1]]
iteration = 2
bits = 8
pop_size = 5
crossover_rate = 0.9 # Pc
mutation_rate = 0.005 # Pm

#Initial population
pop = [randint(0, 2, bits * len(bounds)).tolist() for _ in  range(pop_size)] #number of chromosome


#main program
best_fitness = []
avg_fitness = []
# best, best_eval = 0, objective_function(decoding(bounds, bits, pop[0]))
T = 0.5
# best_so_far = []

for gen in range(1, iteration+1):

    real_chromosome = [decoding(bounds, bits, p) for p in pop]

    fitness = [objective_function(d) for d in real_chromosome] #fitness value

    index = np.argmax(fitness)
    current_best = pop[index]

    best_fitness.append(1 / max(fitness) - 1)
    # pop = Roulette_wheel_selection(pop, fitness, pop_size)

    # pop = sus_selection(pop, fitness, pop_size)


    # pop = Boltzman(pop, fitness, pop_size, T)
    # T -= 0.01


    # pop_fit_list = list(zip(pop, fitness))
    # sorted_list = sorted(pop_fit_list, key=lambda x: x[1])
    # index_ = [i for i in range(1, len(sorted_list)+1)]
    # pop_fit_index = list(zip(pop, fitness, index_))
    # pop = Ranked(pop, index_, pop_size)


    pop_fit_list = list(zip(pop, fitness))
    pop = Tornoment(pop_fit_list, pop_size)


    # pop_fit_list = list(zip(pop, fitness))
    # sorted_list = sorted(pop_fit_list, key=lambda x: x[1])
    #
    #
    # chromosome_selected = int(pop_size / 4)
    # fitness_sorted_min = sorted_list[0:chromosome_selected]
    # chromosome_selected = -chromosome_selected
    # fitness_sorted_max = sorted_list[chromosome_selected:]
    #
    # population_max = [t[0] for t in fitness_sorted_max]
    # population_min = [t[0] for t in fitness_sorted_min]
    #
    # positive_mask = majority(population_max)
    # negative_mask = majority(population_min)
    #
    # mask_ = create_mask(negative_mask, positive_mask)
    # offspring = crossover_mask(pop_fit_list, mask_, crossover_rate)


    # offspring = crossover(pop, crossover_rate)
    offspring = mutation(offspring, mutation_rate)

    for s in offspring:
        pop.append(s)
