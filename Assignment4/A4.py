import numpy as np
import utils
import pandas as pd
from sklearn.neural_network import MLPClassifier




#Perfect Instances
five =  [0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0]
two = [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0]
patterns = [five,two]

def loadGeneratedData():
	gdf = pd.read_csv("rosu7115-TrainingData.csv")
	return gdf
	#print("TODO")


def distort_input(instance, percent_distortion):

    #percent distortion should be a float from 0-1
    #Should return a distorted version of the instance, relative to distoriton Rate
    new_inst = []
    changed = []
    num_instances_to_change = round(len(instance)*percent_distortion)
    for i in range(0,int(num_instances_to_change)):
    	changed.append(instance[i])
    	if(instance[i]==0):
    		changed[i] = 1
    	elif(instance[i]==1):
    		changed[i] = 0

    	#changed.append(instance[i])
    #changed = instance[0: num_instances_to_change]
    for j in range(int(num_instances_to_change), len(instance)):
    	changed.append(instance[j])
    return changed


    print("TODO")
    #utils.raiseNotDefined()


class HopfieldNetwork:
	def __init__(self, size):
		self.h = np.zeros([size,size])

	def addSinglePattern(self, p):
		#Update the hopfield matrix using the passed pattern
		for i in range(0,len(self.h)):
			for j in range(0,len(self.h)):
				if(i==j):
					self.h[i][j]=0
				else:
					self.h[i][j] += ((2*p[i])-1) * ((2*p[j])-1)

		#print("TODO")
		#utils.raiseNotDefined()

	def fit(self, patterns):
		# for each pattern
		# Use your addSinglePattern function to learn the final h matrix
		for x in patterns:
			self.addSinglePattern(x)
		return self.h
		#print("TODO")
		#utils.raiseNotDefined()

	def retrieve(self, inputPattern):
		#Use your trained hopfield network to retrieve and return a pattern based on the
		#input pattern.
		#HopfieldNotes.pdf on canvas is a good reference for asynchronous updating which
		#has generally better convergence properties than synchronous updating.
		changed = True
		y = -999
		x = np.random.choice(25, 25, replace = False)
		iterator = 0
		while(changed==True):
			iterator = 0
			for i in range(0,25):
				y = np.dot(inputPattern, self.h[:,[x[i]]])
				if(y>=0):
					y = 1
				elif(y<0):
					y = 0
				if(inputPattern[x[i]] == y):
					iterator += 0
	
				else:
					iterator += 1
					inputPattern[x[i]] = y
			if(iterator>0):
				changed = True
			else:
				changed = False
			iterator = 0

		return inputPattern



		print("TODO")
		#utils.raiseNotDefined()

	def classify(self, inputPattern):
		#Classify should consider the input and classify as either, five or two
		#You will call your retrieve function passing the input
		#Compare the returned pattern to the 'perfect' instances
		#return a string classification 'five', 'two' or 'unknown'
		r = self.retrieve(inputPattern)
		#print(r)
		#print(two)
		if(np.array_equal(r,two)):
			return 'two'
		elif(np.array_equal(r,five)):
			return 'five'
		else:
			return 'unknown'
		#print("TODO")
		#utils.raiseNotDefined()




if __name__ == "__main__":
	hopfieldNet = HopfieldNetwork(25)

	#utils.visualize(five)
	#utils.visualize(two)
	#hopfieldNet.addSinglePattern(two)
	#hopfieldNet.addSinglePattern(five)
	#hopfieldNet.fit(patterns)
	#g = loadGeneratedData()
	#MP = MLPClassifier(max_iter = 10000)
	#MP.fit(patterns,["five","two"])
	#a = []
	#for index,row in g.iterrows():
		#a.append(row[0:25])
	#arr = np.array(a)
	#print(arr)
	#print(MP.predict(a))
	#print(distort_input(two,1))

	#df1 = pd.read_csv("rosu7115-TrainingData.csv")
	#df2 = pd.read_csv("NewInput.csv")
	df_final = pd.read_csv("final.csv")
	#print(df_final)
	z = []
	t = []
	for x, y in df_final.iterrows():
		z.append(y[0:25])
		t.append(y[25])

	zz = np.array(z)
	tt = np.array(t)

	#for index,row in df.iterrows():


	#print(hopfieldNet.classify(two))

	#x = []
	#b = loadGeneratedData()
	#c = []
	#n = []
	#for x, y in b.iterrows():
	#	c.append(y[0:25])
	#	n.append(y[25])
	#d = np.array(c)
	#f = np.array(n)
	#print(f)
	
	MP1 = MLPClassifier(max_iter=1000, hidden_layer_sizes = 10000)
	MP1.fit(patterns,["five","two"])
	#MP1.predict(d)
	hopfieldNet.fit(patterns)
	#for i in range(0,8):
		#print(hopfieldNet.classify(d[i]))
		#print(MP1.predict(distort_input(d[i],0.5)))
	e = []
	
	for j in np.arange(0.0,0.51,0.01):
		accuracy = 0
		accuracy2 = 0
		e = []
		#print("Distortion", j)
		MP1 = MLPClassifier(max_iter=1000)
		MP1.fit(patterns,["five","two"])
		#MP1.predict(d)
		hopfieldNet.fit(patterns)
		for i in range(0,8):
			#print(i)
			o = distort_input(zz[i],j)
			#print("HOPFIELD", hopfieldNet.classify(o))
			#if(f[i]==hopfieldNet.classify(o)):
				#accuracy+=1
			e.append(o)
		for i in range(0,len(e)):
			if(MP1.predict(e)[i]==tt[i]):
				accuracy2+=1
		#print(accuracy/8)
		print(accuracy2/8)
		#print("MLP", MP1.predict(e))
		#print(hopfieldNet.classify(e))

		#e.append(distort_input)
		#print(MP1.predict(distort_input(d,j)))
			#print(hopfieldNet.classify(distort_input(c[i],j)))



	
	#for i in np.arange(0,0.5,0.01):
		#distort_input(z,i)


		#print(i)
		#print(distort_input(z,i))


