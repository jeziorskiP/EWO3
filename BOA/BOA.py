import random
import numpy as np 
import math
import matplotlib.pyplot as plt 

import statistics
import matplotlib.pyplot as plt 

times = 30
number = 1

a = 0.2
p = 0.9
c = 0.001

"""
lower = -50
upper = 50
fun = 1
d = 2

"""

"""
lower = -10
upper = 10
fun = 2
d = 2
#min x = pi, f(x...) = -1
target = -1
"""


"""
#sphere
lower = -10  
upper = 10
fun = 3
d = 20
#min x = 0 f(x...) = 0
target = 0
"""


"""
lower = -10  
upper = 10
fun = 4
d = 10
#min x = 0 f(x...) = 0
target = 0
"""



"""
lower = -10  
upper = 10
fun = 5
d = 10
#min x = 0 f(x...) = 0
target = 0
"""

"""
n_iterations = 1000#int(input("Inform the number of iterations: "))
target_error = 0.001#float(input("Inform the target error: "))
n_particles = 30#int(input("Inform the number of particles: "))
"""



katalog = "./test"+ str(number) +"/"

filename = "init" +  str(number) + ".txt"

file1 = open(filename,"r") 
readed = file1.read().split('\n')
print(readed)
W = float( readed[0] )
c1 = float( readed[1] )
sg = int( readed[2] )
pm = float(readed[3])

lower = int( readed[4] )
upper = int( readed[5] )
fun = int( readed[6] )
d = int( readed[7] )
#min x = 0 f(x...) = 0
target = float( readed[8] )

n_iterations = int( readed[9] )
target_error = float( readed[10] )
n_particles = int( readed[11] )




print("parametry:")
print("W ", W, " c1 ", c1, " sg ", sg, " pm ", pm, " lower ", lower, " upper ", upper, " fun ", fun )



class Particle():
    def __init__(self):
        self.position = np.array( [ (-1) ** (bool(random.getrandbits(1))) * random.random()*upper for i in range(d)        ] )

        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = np.array([0 for i in range(d)])
        self.exemplar = np.array( [ (-1) ** (bool(random.getrandbits(1))) * random.random()*upper for i in range(d)        ] )
        self.count = 0
        self.tournament = 0
        self.fragrance = 0.0
        self.fitnessValue = 0.0

    def __str__(self):
        print("Pozycja ", self.position, " pbest: ", self.pbest_position, " ILOSC: ", len(self.position), " count: ", self.count, " tour: ", self.tournament," F: ",self.fragrance, " Fitness: ",self.fitnessValue )
    
    def move(self):
        self.position = self.position + self.velocity


class Space():

    def __init__(self, target, target_error, n_particles):
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_position = np.array([random.random()*upper for i in range(d) ])

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()
   
    def fitness1(self, particle, way):
        if(fun == 1):
            if(way == "normal"):
                return particle.position[0] ** 2 + particle.position[1] ** 2 + 1
            elif(way =="exemplar"):
                return particle.exemplar[0] ** 2 + particle.exemplar[1] ** 2 + 1
            else:
                print("ERROR")
                
        elif(fun == 2):
            if(way == "normal"):
                return -math.cos( particle.position[0] )*math.cos( particle.position[1] )*math.exp( -(particle.position[0] - math.pi)**2  - (particle.position[1] - math.pi)**2 )
            elif(way =="exemplar"):
                return -math.cos( particle.exemplar[0] )*math.cos( particle.exemplar[1] )*math.exp( -(particle.exemplar[0] - math.pi)**2  - (particle.exemplar[1] - math.pi)**2 )
        
        elif(fun == 3):
            if(way == "normal"):
                result = 0.0
                for i in range(d):
                    result += particle.position[i]**2
                    
                return result
                
            elif(way =="exemplar"):
                result = 0.0
                for i in range(d):
                    result += particle.exemplar[i]**2
                    
                return result
                
            else:
                print("ERROR")

        elif(fun == 4):
            if(way == "normal"):
                result = 0.0
                for i in range(d):
                    result += (particle.position[i] - i+1)**2
                    
                return result
                
            elif(way =="exemplar"):
                result = 0.0
                for i in range(d):
                    result += (particle.exemplar[i] - i+1)**2
                    
                return result
                
            else:
                print("ERROR")
        elif(fun == 5):
            if(way == "normal"):
                resultSum = 0.0
                resultMulti = 1.0
                for i in range(d):
                    resultSum += (abs(particle.position[i])**2)
                    resultMulti = resultMulti * (abs(particle.position[i])**2)
                    
                return resultSum + resultMulti
                
            elif(way =="exemplar"):
                resultSum = 0.0
                resultMulti = 1.0
                for i in range(d):
                    resultSum += (abs(particle.exemplar[i])**2)
                    resultMulti = resultMulti * (abs(particle.exemplar[i])**2)
                    
                return resultSum + resultMulti
                
            else:
                print("ERROR")
            
            
        else:
            print("ERROR")
            
        #tablica
    def fitnessOffSpring(self, particle):
        if(fun == 1):
            return particle[0] ** 2 + particle[1] ** 2 + 1
        elif(fun == 2):
            return math.cos( particle[0] )*math.cos( particle[1] )*math.exp( -(particle[0] - math.pi)**2  - (particle[1]-math.pi)**2 )
        elif(fun == 3):
            result = 0.0
            for i in range(d):
                result += particle[i]**2
            return result
        elif(fun == 4):
            result = 0.0
            for i in range(d):
                result += (particle[i] - i+1)**2
            return result
        elif(fun == 5):
            resultSum = 0.0
            resultMulti = 1.0
            for i in range(d):
                resultSum += (abs(particle[i])**2)
                resultMulti = resultMulti * (abs(particle[i])**2)
                
            return resultSum + resultMulti
        else:
            print("error")
        

    def calcFragrance(self):
        for particle in self.particles:
            p = pow(particle.fitnessValue, a)
            particle.fragrance = c*p
    
    def calcFitnessValue(self):
        for particle in self.particles:
            particle.fitnessValue = self.fitness1(particle, "normal")
    
    def cal(self, input):
        result = 0
        for i in range(d):
            result = input[i]**2



    def fitnessX(self, input):
        return math.cos( input[0] )*math.cos( input[1] )*math.exp( -(input[0] - math.pi)**2  - (input[1]-math.pi)**2 )



    def set_pbest(self):
        for particle in self.particles:
            fitness_cadidate = self.fitness1(particle, "normal")
            if(particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position
            

    def set_gbest(self):
        for particle in self.particles:
            best_fitness_cadidate = self.fitness1(particle, "normal")
            if(self.gbest_value > best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position
                
    def set_gbest_BOA(self):
        for particle in self.particles:
            if self.gbest_value > particle.fitnessValue:
                self.gbest_value = particle.fitnessValue
                self.gbest_position = particle.position
                self.gbest_frag = particle.fragrance
                print("zmieniam best w swarm0")

    def moveEq2(self, particle):
        r = random.random() 
        r2 = np.asarray(particle.exemplar) * r*r

        particle.position = particle.position + (r2 - particle.position) * particle.fragrance
        
    def moveEq3(self, particle):
        r = random.random()/2
        max = len(self.particles) -1
        end = 0
        while end == 0:
            end = 1
            pos1 = random.randint(0,max)
            pos2 = random.randint(0,max)
            if(pos1 != pos2):
                end = 1

        particle.position = particle.position  + (r*r * self.particles[pos1].position - self.particles[pos2].position ) * particle.fragrance



        
    def Work(self):
        for particle in self.particles:
            
            OffSpring = []
            #crossingOver
            for dim in range(d):            #from 0 to d
                randomParticleIndex = random.randint(0, len(self.particles)-1)   #from 0 to n_particles-1

                randomParticle = self.particles[randomParticleIndex]
                
                if( self.fitness1(particle, "normal") < self.fitness1(randomParticle, "normal")):
                    newVal = random.random() * particle.pbest_position[dim] + (1- random.random() ) * self.gbest_position[dim]
                    OffSpring.append(newVal)
                else:
                    OffSpring.append(randomParticle.position[dim])

            #end for
            
            #Mutation
            for dim in range(d):
                rand = random.random()
                if(rand < pm):
                    OffSpring[dim] = random.uniform(lower, upper)
            #end for
            
            #Selection
            if(self.fitnessOffSpring(OffSpring) < self.fitness1(particle, "exemplar") ):
                particle.exemplar = OffSpring
                
            else:
                particle.count +=1
            #end EXEMPLAR 
            
            if(particle.count > sg):
                particle.count = 0
                particle.tournament += 1
                #print("ZMIENIAM NA ZERO!!!!!!")
                particle.exemplar  = self.tournament()
                #particle.exemplar  = self.gbest_position
            
            #Update
  
            r = random.random()
            if(r < p):            #wiekszosc ma robic to dobre
                #r2 = np.asarray(particle.exemplar) * r*r
                #particle.position = particle.position + (r2 - particle.position) * particle.fragrance
                self.moveEq2(particle)
            else:
                self.moveEq3(particle)
            
            
    def tournament(self):
        indexArray = ([ i for i in range(n_particles)])
        count = int(n_particles*50/100)
        if(count <3):
            count = 3
        
        cols = count
        rows = 2

        list = [[0]*cols for _ in [0]*rows]
        for x in range(count):
            randIndex = random.randint(0,n_particles-1)
            list[0][x] = randIndex
            list[1][x] = self.fitness1(self.particles[randIndex], "normal")
            #print( self.fitness1(self.particles[randIndex], "normal") )
            #print("I: ", self.particles[randIndex].fragrance)
            
        
        wartosc = np.amin(list, axis=1)

        result = np.where(list == wartosc[1]  )

        #print("Index", result[1][0])            # nr kolumny w ktorej jest najmniejsza wartosc
        
        winner = ( list[0][  result[1][0] ] )

        particleIndex = winner

        return self.particles[particleIndex].position
        
    def BOAPrepare(self):
        self.calcFitnessValue()
        self.calcFragrance()
    
    def AvgPerIter(self):
        result = 0.0
        cnt = 0
        for particle in self.particles:
            result += particle.pbest_value
            cnt +=1
    
        return result/cnt        
        

    def AvgFitnessValuePerIter(self):
        result = 0.0
        cnt = 0
        for particle in self.particles:
            result += self.fitness1(particle, "normal")
            cnt +=1
    
        return result/cnt
    
    
    def plotAvg(self, time, array1, array2, array3):
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        X = []
        Y1 = []
        Y2 = []
        Y3 = []
        
        for a in range(0, len(array1)):   
            Y1.append(array1[a])
            Y2.append(array2[a])
            Y3.append(array3[a])
            X.append(a)

        axes[0].plot(X, Y1, label = "line1") 
        axes[0].plot(X, Y2, label = "line2") 
        axes[0].plot(X, Y3, label = "line3") 
        # naming the x axis 
        axes[0].set_xlabel('iteration') 
        # naming the y axis 
        axes[0].set_ylabel('value') 
        # giving a title to my graph 
        axes[0].set_title('') 
        # function to show the plot      



        axes[1].plot(X, Y1, label = "Line1") 
        axes[1].plot(X, Y2, label = "Line2") 
        axes[1].plot(X, Y3, label = "Line3") 
        # naming the x axis 
        axes[1].set_xlabel('iteration') 
        # naming the y axis 
        axes[1].set_ylabel('value') 
        # giving a title to my graph 
        axes[1].set_title('Skala logarytmiczna') 
        axes[1].set_yscale('log')
        # function to show the plot 
        
        
        plt.legend()
        plt.show()   
        
        fig.savefig( katalog+str(time) + ".jpg")
    
        plt.close('all')
    
    
    
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
# średnia ze średnich najlepszych dostosowan - 
values = []


for time in range(times):

    msg = ""
    msg += "Przebieg nr: "+ str(time) + "\n"

        
    search_space = Space(target, target_error, n_particles)
    particles_vector = [Particle() for _ in range(search_space.n_particles)]
    search_space.particles = particles_vector
    search_space.print_particles()


    avgPerIter = []
    avgFitnessValuePerIter = []
    gBestFitness = []


    iteration = 0
    while(iteration < n_iterations):
        print("Iter: ",iteration)
        
        search_space.BOAPrepare()
        search_space.set_pbest()    
        search_space.set_gbest()

        if(abs(search_space.gbest_value - search_space.target) <= search_space.target_error):
            break
            
        search_space.Work()
        
        avgPerIter.append(search_space.AvgPerIter())
        avgFitnessValuePerIter.append(search_space.AvgFitnessValuePerIter() )
        gBestFitness.append( search_space.gbest_value )
        
        msg += "Iteracja nr " +  str(iteration) + " Najlepsze dopasowanie: " + str(search_space.gbest_value) + "\n"
        
        iteration += 1
        #search_space.print_particles()
        
        print("best vallue: ",search_space.gbest_value)

    search_space.plotAvg(time,avgPerIter,avgFitnessValuePerIter, gBestFitness )
    
    msg += "-------------------------------------------\n"
    msg += ("Najlepsze dopasowanie:  " + str(search_space.gbest_value  ) )
    
    
    print(msg,  file=open(katalog+str(time)+".txt", 'w'))
    
    values.append(search_space.gbest_value)
    
    print("The best solution is: ", search_space.gbest_position, " in n_iterations: ", iteration)
    print()
    print("best vallue: ",search_space.gbest_value)

msgF = "Mean " + str((statistics.mean(values))) + "\n"+ "Min " + str((min(values)  )) + "\n"+ "Max " + str((max(values))) + "\n"+  "Median " + str((statistics.median(values) ))  + "\n"+  "Std " + str( np.std(values) ) + "\n"+  "Closest " + str( search_space.find_nearest(values, search_space.target) )  + "\n"
            
print("Mean ",statistics.mean(values)  )
print("Min ",min(values)  )
print("Max ",max(values)  )
print("Median ",statistics.median(values) )
print("Std  ",np.std(values)  )
print("Closest  ", search_space.find_nearest(values, search_space.target)  )




print(msgF,  file=open(katalog+"Wyniki.txt", 'w'))

