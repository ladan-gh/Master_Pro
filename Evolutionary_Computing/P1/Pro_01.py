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
        p1 = pop[i].copy() #parent

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

    # print(index_selected)
    # print(index_selected[0])
    # print(pop)
    # print(pop[index_selected[0]])

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

#------------------------------------------------------
#Parameters of the binary genetic algorithm
bounds = [[-4, 2], [-1.5, 1]]
iteration = 50
bits = 8
pop_size = 10
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
    offspring = crossover(pop, crossover_rate)
    offspring = mutation(offspring, mutation_rate)


    for s in offspring:
        pop.append(s)

    real_chromosome = [decoding(bounds, bits, p) for p in pop]

    fitness = [objective_function(d) for d in real_chromosome] #fitness value

    for i in range(0, pop_size):
        if fitness[i] > best_eval:
            best, best_eval = pop[i], fitness[i]
            print(">%d, new best f(%s) = %f" % (gen, real_chromosome[i], fitness[i]))
            # best_so_far.append(best_eval)


    avg_fitness.append(np.mean(fitness))

    index = np.argmax(fitness)
    current_best = pop[index]

    best_fitness.append(1 / max(fitness) - 1)
    pop = selection(pop, fitness, pop_size)


# for i in range(0, len(best_so_far)):
#     if i == 0:
#         print(">%d, best_eval = %f" % (1, best_so_far[i]))
#     else:
#         print(">%d, best_eval = %f" % (i, best_so_far[i]))

print('Done!')

#-------Plot----------------------------
fig = plt.figure()
plt.plot(avg_fitness)
fig.suptitle('Average fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.show()

#---------------------------------------
# fig = plt.figure()
# plt.plot(best_fitness)
# fig.suptitle('Evolution of the best Chromosome')
# plt.xlabel('Iteration')
# plt.ylabel('Objective function value')
# plt.show()

# print('Min objective function value:', min(best_fitness))
# print('Optimal solution:', decoding(bounds, bits, current_best))