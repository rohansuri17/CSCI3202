import GAinspector
#import numpy as np
from utils import *

import random

def randomGenome(length):
    """
    :param length:
    :return: string, random binary digit
    """
    """Your Code Here"""
    genome = ""
    for i in range(0,length):
    	genome += str(random.randint(0,1))
    return genome	
    #raiseNotDefined()

def makePopulation(size, length):
    """
    :param size - of population:
    :param length - of genome
    :return: list of length size containing genomes of length length
    """

    """Your Code Here"""
    population = []
    for i in range(0,size):
    	population.append(randomGenome(length))
    return population	

    #raiseNotDefined()


def fitness(genome):
    """
    :param genome: 
    :return: the fitness value of a genome
    """
    fitness_val = 0
    for i in range(0,len(genome)):
    	fitness_val+=int(genome[i])
    return fitness_val	


    



    raiseNotDefined()

def evaluateFitness(population):
    """
    :param population: 
    :return: a pair of values: the average fitness of the population as a whole and the fitness of the best individual in the population.
    """
    max = -1
    sum = 0
    for i in range(0,len(population)):
    	sum += fitness(population[i])
    	if(fitness(population[i])>max):
    		max = fitness(population[i])
    avg = sum/len(population)

    return [avg,max]
    	


    raiseNotDefined()



def crossover(genome1, genome2):
    """
    :param genome1:
    :param genome2:
    :return: two new genomes produced by crossing over the given genomes at a random crossover point.
    """
    crossover_point = random.randint(1,len(genome1)-1)
    gene1 = genome1[:crossover_point]+genome2[crossover_point:]
    gene2 = genome2[:crossover_point]+genome1[crossover_point:]

    return gene2,gene1
    raiseNotDefined()





def mutate(genome, mutationRate):
    """
    :param genome:
    :param mutationRate:
    :return: a new mutated version of the given genome.
    """
    new_genome = ""
    num_genomes_to_change = round(len(genome) * mutationRate)
    mutate_gene = genome[0: num_genomes_to_change]
    #new_genome = ""
    for x in mutate_gene:
    	if(x=="1"):
    		x = "0"
    	else: 
    		x = "1"	
    	new_genome+=x	

    

    res = new_genome + genome[num_genomes_to_change: len(genome)]
    #print(new_genome)		
    return res		
    			
    	
    raiseNotDefined()

def selectPair(population):
    """

    :param population:
    :return: two genomes from the given population using fitness-proportionate selection. This function should use weightedChoice, which we wrote in class, as a helper function.
    """
    weightsArray = []
    for x in population:
    	weightsArray.append(fitness(x))
    genomeA = weightedChoice(population,weightsArray)
    genomeB = weightedChoice(population,weightsArray)

    return genomeA,genomeB	

    raiseNotDefined()

def performCrossover(crossoverRate):
    maxVal = 100
    chanceVal = random.randint(1, maxVal)

    return chanceVal <= crossoverRate * maxVal # 70 

def runGA(populationSize, crossoverRate, mutationRate, logFile=""):
    """

    :param populationSize: :param crossoverRate: :param mutationRate: :param logFile: :return: xt file in which to
    store the data generated by the GA, for plotting purposes. When the GA terminates, this function should return
    the generation at which the string of all ones was found.is the main GA program, which takes the population size,
    crossover rate (pc), and mutation rate (pm) as parameters. The optional logFile parameter is a string specifying
    the name of a te
    """
    '''
    genomeLength = 20
    bestGeneration = -1
    population = makePopulation(populationSize, genomeLength)
    currentGeneration = 0

    saveRun = 1
    saveFile = open(logFile, 'a')
    saveFile.write("----new run----\n")
    print("Population Size ", populationSize)
    print("Genome Length ", genomeLength)

    while (bestGeneration == -1):
        newGen = []
        highest, avg = evaluateFitness(population)

        if (highest == genomeLength):
            bestGeneration = currentGeneration

        #popSum = sum(list(map(lambda x: fitness(x), population)))
        print("Generation ", currentGeneration, ": average fitness ", avg, ", best fitness ", highest)

        saveFile.write(str(
            "pop " + str(populationSize) + " genLen " + str(genomeLength) + " gen " + str(currentGeneration) + " avg " + str(
                avg) + " best " + str(highest)) + "\n")

        for i in range(int(populationSize / 2)):

            newGenomeA, newGenomeB = selectPair(population)

            if (performCrossover(crossoverRate)):
                newGenomeA, newGenomeB = crossover(newGenomeA, newGenomeB)

            newGenomeA = mutate(newGenomeA, mutationRate)
            newGenomeB = mutate(newGenomeB, mutationRate)

            newGen.append(newGenomeA)
            newGen.append(newGenomeB)

        population = newGen

        currentGeneration = currentGeneration + 1
    if (saveRun == 1):
        saveFile.close()
    return bestGeneration
    '''
    genomeLength = 20
    pop = makePopulation(populationSize,genomeLength)
    curr_gen = 0
    best_gen = -1
    fr = open(logFile,'w')
    print("Population size: ", )

    while(best_gen==-1 and curr_gen < 51):
        #population,y = sortByFitness(pop)
        next_gen = []
        avg,best = evaluateFitness(pop)
        
        if(best == genomeLength):
            best_gen = curr_gen

        

        #print("Generation#: ", curr_gen," average fitness: ", avg, " best fitness ", best )
        #print("average fitness", avg)
        print(avg)

        #if(x % 10 == 0):
            #fr.write(str("Generation#: " + str(x) + " average fitness: " + str(avg) + " best fitness " + str(best)))
        for j in range(int(populationSize/2)):
            gene1, gene2 = selectPair(pop)
            r = random.random()
            if(r<=crossoverRate):
                gene1, gene2 = crossover(gene1,gene2)

            gene1 = mutate(gene1,mutationRate)
            gene2 = mutate(gene2,mutationRate)

            next_gen.append(gene1)
            next_gen.append(gene2)

        pop = next_gen
        curr_gen = curr_gen + 1
    fr.close()  
    return curr_gen-1




    raiseNotDefined()





if __name__ == '__main__':
    #Testing Code
    print("Test Suite")
    GAinspector.test(randomGenome)
    GAinspector.test(makePopulation)
    GAinspector.test(fitness)
    GAinspector.test(evaluateFitness)
    GAinspector.test(crossover)
    GAinspector.test(mutate)
    GAinspector.test(selectPair)
    sum_best_gens = 0
    maxi = -1
    mini =99999

    for i in range(0,50):
        x = runGA(100, 0.7, 0.9, "run1.txt")
        print(x)
        sum_best_gens += x
        if x > maxi:
            maxi = x
        if x<mini:
            mini = x
    avg = sum_best_gens/50
    print("AVG", avg)
    print("Max", maxi)
    print("Min", mini)


    