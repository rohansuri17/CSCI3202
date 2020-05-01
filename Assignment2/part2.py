import robby
#import numpy as np
from utils import *
import random
POSSIBLE_ACTIONS = ["MoveNorth", "MoveSouth", "MoveEast", "MoveWest", "StayPut", "PickUpCan", "MoveRandom"]
rw = robby.World(10, 10)
rw.graphicsOff()


def sortByFitness(genomes):
    tuples = [(fitness(g), g) for g in genomes]
    tuples.sort()
    sortedFitnessValues = [f for (f, g) in tuples]
    sortedGenomes = [g for (f, g) in tuples]
    return sortedGenomes, sortedFitnessValues


def randomGenome(length):
    """
    :param length:
    :return: string, random integers between 0 and 6 inclusive
    """

    """Your Code Here"""
    genome = ""
    for i in range(0,length):
        genome += str(random.randint(0,6))
    return genome   



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
    raiseNotDefined()

def fitness(genome, steps=200, init=0.50):
    """

    :param genome: to test
    :param steps: number of steps in the cleaning session
    :param init: amount of cans
    :return:
    """
    reward = 0 
    sum = 0 
    for j in range(0,1):

        
        if len(genome) != 243:
            raise Exception("strategy is not a string of length 243")
        for char in genome:
            if char not in "0123456":
                raise Exception("strategy contains a bad character: '%s'" % char)
        if type(steps) is not int or steps < 1:
            raise Exception("steps must be an integer > 0")
        if type(init) is str:
            # init is a config file
            rw.load(init)
        elif type(init) in [int, float] and 0 <= init <= 1:
            # init is a can density
            rw.goto(0, 0)
            rw.distributeCans(init)
        else:
            raise Exception("invalid initial configuration")    
        for i in range(0,steps):
            #print("A"+genome[i])
            #sum+=1
            if genome[i]=="0":
                reward +=rw.performAction("MoveNorth")
                #print(reward)
            elif genome[i]=="1":
                reward +=rw.performAction("MoveSouth")                            
                #print(reward)  
            elif genome[i] =="2":
                reward +=rw.performAction("MoveEast")
                #print(reward)
            elif genome[i]=="3":
                reward +=rw.performAction("MoveWest")
                #print(reward)
            elif genome[i]=="4":
                reward +=rw.performAction("MoveRandom")
                #print(reward)
            elif genome[i]=="5":
                reward +=rw.performAction("StayPut")
                #print(reward)
            else:
                reward +=rw.performAction("PickUpCan")
                #print(reward)

        #avg = reward/25
        #print(sum)
    
    return reward        


        
            

    raiseNotDefined()

def evaluateFitness(population):
    """
    :param population:
    :return: a pair of values: the average fitness of the population as a whole and the fitness of the best individual
    in the population.
    """
    max = -999999
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
        x=str(random.randint(0,6))
        new_genome+=x   

    

    res = new_genome + genome[num_genomes_to_change: len(genome)]
    #print(new_genome)      
    return res      
    raiseNotDefined()

def selectPair(population):
    """

    :param population:
    :return: two genomes from the given population using fitness-proportionate selection.
    This function should use RankSelection,
    """
    '''
    fitnessArray = []
    weightedArray = []
    #a=0
    for x in population:
        fitnessArray.append(str(fitness(x)))

    sortByFitness(fitnessArray)
    for j in range(0,len(fitnessArray)):
        #a+=i
        weightedArray.append(j)   




    genomeA = weightedChoice(population,weightedArray)
    genomeB = weightedChoice(population,weightedArray)
    return genomeA,genomeB
    '''

    #sortByFitness(population)    
    weightsArray = range(1,len(population)+1)
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
    genomeLength = 243
    pop = makePopulation(populationSize,genomeLength)
    num_generations = 300
    #fitness_best_strategy = 200
    best_strategy = 0
    fr = open(logFile,'w')
    #print()

    for x in range(num_generations):
        next_gen = []
        population,y = sortByFitness(pop)
        avg,best = evaluateFitness(pop)

        

        #print("Generation#: ", x," average fitness: ", avg, " best fitness ", y[populationSize-1] )
        print(y[populationSize-1])
        if(x % 10 == 0):
            fr.write(str("Generation#: " + str(x) + " average fitness: " + str(avg) + " best fitness " + str(y[populationSize-1])) + "Genome: " + str(population[populationSize-1])+ "\n")

            #rw.demo(population[populationSize-1],steps = 200, init = 0.5)
        
            #best_strategy = fitness(population[populationSize-1])
            #print("Best strategy:", best_strategy)
        #if(y[populationSize-1] >= fitness_best_strategy):
            #break;
        if(y[populationSize-1] > best_strategy):
            best_strategy = int(population[populationSize-1])

            
            
        for j in range(int(populationSize/2)):
            gene1, gene2 = selectPair(population)
            r = random.random()
            if(r<=crossoverRate):
                gene1, gene2 = crossover(gene1,gene2)  
            gene1 = mutate(gene1,mutationRate)
            gene2 = mutate(gene2,mutationRate)

            next_gen.append(gene1)
            next_gen.append(gene2)
        pop = next_gen    


    fr.close()  
    return str(best_strategy)



    '''
    genomeLength = 243
    bestGeneration = -1
    population = makePopulation(populationSize, genomeLength)
    #population = sortByFitness(population)
    currentGeneration = 0

    saveRun = 1
    saveFile = open(logFile, 'a')
    saveFile.write("----new run----\n")
    print("Population Size ", populationSize)
    print("Genome Length ", 243)

    while (currentGeneration < 300):
        newGen = []
        population,x = sortByFitness(population)
        avg, highest = evaluateFitness(population)

        if (currentGeneration % 10 ==0):
            bestGeneration = highest

        #popSum = sum(list(map(lambda x: fitness(x), population)))
        print("Generation ", currentGeneration, ": average fitness ", avg, ", best fitness ", highest)

        saveFile.write(str(
            "pop " + str(populationSize) + " genLen " + str(243) + " gen " + str(currentGeneration) + " avg " + str(
                avg) + " best " + str(highest)) + "\n")

        for i in range(int(populationSize / 2)):

            newGenomeA, newGenomeB = selectPair(population)

            if (performCrossover(crossoverRate)):
                gene1, gene2 = crossover(newGenomeA, newGenomeB)

            gene1 = mutate(gene1, mutationRate)
            gene2 = mutate(gene2, mutationRate)

            newGen.append(gene1)
            newGen.append(gene2)

        population = newGen

        currentGeneration = currentGeneration + 1
    if (saveRun == 1):
        saveFile.close()    
    return bestGeneration
    '''



    raiseNotDefined()


def test_FitnessFunction():
    f = fitness(rw.strategyM)
    #rw.demo(rw.strategyM,steps=200,init=0.50)
    #print("Fitness for StrategyM : {0}".format(f))
    #f2 = fitness(runGA(100, 1.0, 0.005,"GAoutput.txt"),steps = 200, init=0.50)
    #y = runGA(100,1.0,0.005,"GAoutput.txt")
    #print(y)
    #print("Fitness for my Strategy : {0}".format(f2)))
    x = runGA(100,1, 0.5,"GAoutput.txt")
    print(x)
    print(fitness(x))
    #print(fitness("132022415356040606306402240066610546241366334611261610125024660016205124300666125201606630333246462260606102330146631124213054641345456222313410123355504626020236662604600565163601524445012364633524601360113423634034121353363354105354025401212"))






test_FitnessFunction()
'''
x = runGA(100, 1.0, 0.005,"GAoutput.txt") #pop 100 crossover 1 mutation 0.005
x1 = fitness(runGA(100, 1.0, 0.005,"GAoutput.txt"),steps = 200, init = 0.50)
y = runGA(1000, 1.0, 0.005,"GAoutput.txt")#pop 1000
y1 = fitnss(runGA(1000, 1.0, 0.005,"GAoutput.txt"),steps = 200, init = 0.50)
a = runGA(100, 0.1, 0.005,"GAoutput.txt")#cross 0.1
a1 = fitness(runGA(100, 0.1, 0.005,"GAoutput.txt"),steps = 200, init = 0.50)#cross 0.1)
b = runGA(100, 1.0, 0.5,"GAoutput.txt")#mutation 0.5
b1 = fitness(runGA(100, 1.0, 0.5,"GAoutput.txt"), steps = 200, init = 0.50)
c = runGA(100, 1.0, 0.005,"GAoutput.txt")#generations 3
c1 = fitness(runGA(100, 1.0, 0.005,"GAoutput.txt"), steps = 200, init =0.50)
print(x,y, a, b ,c, "\n")
print(x1, y1, a1, b1, c1)
'''


'''
or i in range(0,50):
        x = 
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
'''
