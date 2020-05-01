from __future__ import print_function
import keras
import utils
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from pathlib import Path
from sklearn.model_selection import train_test_split
import glob
from datetime import datetime
#print datetime.datetime.now()



import numpy as np

from PIL import Image

"""def getVGGFeatures(img, layerName):
		base_model = VGG16(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)
	img = img.resize((224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	internalFeatures = model.predict(x)

	return internalFeatures
"""
def getVGGFeatures(fileList, layerName):
	#Initial Model Setup
	base_model = VGG16(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)
	
	#Confirm number of files passed is what was expected
	rArray = []
	print ("Number of Files Passed:")
	print(len(fileList))

	for iPath in fileList:
		#Time Printing for Debug, you can comment this out if you wish
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print("Current Time =", current_time)
		try:
			#Read Image
			img = image.load_img(iPath)
			#Update user as to which image is being processed
			#print("Getting Features " +iPath)
			#Get image ready for VGG16
			img = img.resize((224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			#Generate Features
			internalFeatures = model.predict(x)
			rArray.append((iPath, internalFeatures))			
		except:
			print ("Failed "+ iPath)
	return rArray

def cropImage(image, x1, y1, x2, y2):

	#utils.raiseNotDefined()

	cropped_image = image.crop((x1,y1,x2,y2))
	return cropped_image

def standardizeImage(image, x, y):
	#utils.raiseNotDefined()
	resized_image = image.resize((x,y))
	return resized_image

def preProcessImages(images):
	#arr = np.array()
	fr = open("downloadedFiles.txt")
	arr = []
	img_arr = []
	for line in fr:
		img_arr.append(line.strip().split('\t')[7])
		arr.append(line.strip().split('\t')[5])
	#print(img_arr)
	path = r'/Users/rohansuri17/Downloads/Assignment5_SkeletonCode_v2/uncropped/'
	path2 = r'/Users/rohansuri17/Downloads/Assignment5_SkeletonCode_v2/uncropped2/'

	#img = Image.open(path + "vartan2.jpg")
	#img.show()
	#print(len(img_arr))

	for i in range(0,len(img_arr)):
		try:
			img = Image.open(path+img_arr[i])
		except:
			print("rip")
		a1 = arr[i].split(',')
		#print(arr[i])
		print(a1)
		cropped_image = cropImage(img,int(a1[0]),int(a1[1]),int(a1[2]),int(a1[3]))
		#cropped_image.show()
		cropped_image = standardizeImage(cropped_image,60,60)
		cropped_image.save(path2+img_arr[i])



	#print(arr)
	#for
	#for i in os.listdir(path):
		#images.append(os.path.join(path,i))
	#print(images)
	#for i in fr, os.listdir(path):
		#try:
			#img = Image.open(os.path.join(path,i))
			#cropped_image = cropImage(img,line.strip().split('\t')[4])
		#except IOError:
			#print("Error opening image")
		#img.show()
		#print(i)
	#for i in range(0,len(images)):
		#try:
			#img = Image.open(Path(images[i]))
			#cropped_image = cropImage(img,arr[i])
		#except IOError:
			#print("Error opening image")
		#img.show()
		#print(i)


		#new_img = Image.open(images[i])
		#crp= cropImage(new_img,1,2,3,4)
		#new_img.save("/Users/rohansuri17/Downloads/Assignment5_SkeletonCode_v2/uncropped2")
	#utils.raiseNotDefined()

def visualizeWeight():
	#You can change these parameters if you need to
	utils.raiseNotDefined()

def trainFaceClassifier(preProcessedImages, labels):
	#utils.raiseNotDefined()
	#img_rows, img_cols = 60, 60
	#x_train, y_train, x_validate, y_validate, x_test, y_test = [],[],[],[],[],[]
	#arr = []
	#path = r'/Users/rohansuri17/Downloads/Assignment5_SkeletonCode_v2/uncropped2/'
	#preProcessedImages = np.array(Imag)
	preProcessedImages = preProcessedImages.astype('float32')
	preProcessedImages = preProcessedImages/255
	print(preProcessedImages.shape)
	print(labels.shape)

	x_train, x_test, y_train, y_test = train_test_split(preProcessedImages,labels,test_size = 0.15)

	
	x_train, x_val, y_train, y_val = train_test_split(preProcessedImages,labels ,test_size = 0.1)
	#x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size = 0.1)


	Y_train = keras.utils.to_categorical(y_train, 6)
	Y_val = keras.utils.to_categorical(y_val, 6)
	Y_test = keras.utils.to_categorical(y_test, 6)


	model = Sequential()
	model.add(Dense(1000, input_shape=(3600,)))
	model.add(Activation('sigmoid'))                            

	model.add(Dense(6))
	model.add(Activation('softmax'))


	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


	# training the model and saving metrics in history
	history = model.fit(x_train, Y_train,
	          batch_size=128, epochs=20,
	          verbose=2,
	          validation_data=(x_val, Y_val))


	score = model.evaluate(x_test, Y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

		


	#x = np.array(Image.open(path+preProcessedImages[0]))
	#print(x)
	#for i in range(0,len(preProcessedImages)):
		#try:
			#np.append(x,Image.open(path+preProcessedImages[i]))
		#except IOError:
			#print("IMG not found... moving on")
	#print(x)

	#np_x_train = np.array([],[])
	#np_y_train = np.array([])
	#np_x_validate = np.array([])
	#np_y_validate = np.array([])
	#np_x_test = np.array([])
	#np_y_test = np.array([])
	#x = []
	#print(np_x_train.shape)

	#for i in range(0,len(preProcessedImages)):
		#if(i<len(preProcessedImages)/3):
			#try:
				#x = np.array(Image.open(path+preProcessedImages[i]).convert('L'))
				
			#except:
				#print("meh")
			#print(x)
			#np_x_train = np_x_train.flatten().reshape(3000,1)
			#x = x.flatten().reshape(x.shape[0],1)
			#np_x_train = np.asarray(np_x_train)
			#print(np_x_train.shape())
			#np_x_train.reshape((0,0))
			#print(np_x_train.shape)
			#print(x.shape)
			#x = x.flatten()#.reshape(3600,1)
			#print(x.shape)
			#print(np_x_train.shape)
			#np_x_train = np.hstack((np_x_train,x))

		

			#np_y_train = np.concatenate((np_y_train,labels[i]))
			#print(np_y_train)
		#elif(i>len(preProcessedImages)/3 and i< (2*len(preProcessedImages)/3)):
			#print("hi")
		#	try:
		#		y = np.array(Image.open(path+preProcessedImages[i]).convert('L'))
		#	except:
		#		print("meh")
		#	np_x_validate = np.concatenate((np_x_validate, x[i]))
		#	np_y_validate = np.concatenate((np_y_validate, labels[i]))
		#else:
		#	try:
		#		z = np.array(Image.open(path+preProcessedImages[i]).convert('L'))
		#	except:
		#		print("meh")
		#	np_x_test = np.concatenate((np_x_test, x[i]))
		#	np_y_test = np.concatenate((np_y_test, labels[i]))
	#print("X_train shape", np_x_train.shape)
	#print("y_train shape", np_y_train.shape)

	#print("X_test shape", np_x_test.shape)
	#print("y_test shape", np_y_test.shape)


def trainFaceClassifier_VGG(extractedFeatures, labels):
	#utils.raiseNotDefined()
	path2 = r'/Users/rohansuri17/Downloads/Assignment5_SkeletonCode_v2/uncropped2/'
	fr = open("downloadedFiles.txt")
	filelist = []
	for image in os.listdir(path2):
		filelist.append(os.path.join(path2,image))
		#print(filelist)
	img_arr = []
	for line in fr:
		#print("hi")
		img_arr.append(path2+line.strip().split('\t')[7])
	print(img_arr)
	model2 = getVGGFeatures(img_arr, 'block4_pool')

	model3 = []

	for i in range(0,657):
		for j in range(1,2):
			model3.append(model2[i][j])
	print(model3)


	img = []
	np_model2 = np.asarray(model3)
	#np_model2 = np_model2.reshape(np_model2.shape[1:])
	#for i in range(0,len(np_model2)):
	#np_model2=np_model2.flatten()
		#img.append()

	print(np_model2.shape)
	print("here it is :(")
	#np_model2 = np_model2.astype('float32')
	#np_model2 = np.delete(np_model2,2,0)



	
	x_train, x_test, y_train, y_test = train_test_split(np_model2,labels,test_size = 0.15)

	x_train, x_val, y_train, y_val = train_test_split(np_model2,labels,test_size = 0.1)




	Y_train = keras.utils.to_categorical(y_train, 6)
	Y_val = keras.utils.to_categorical(y_val, 6)
	Y_test = keras.utils.to_categorical(y_test, 6)
	

	model = Sequential()
	model.add(Dense(1000, input_shape=(1,14,14,512)))
	model.add(Activation('relu'))                            

	model.add(Dense(6))
	model.add(Activation('softmax'))

	
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


	# training the model and saving metrics in history
	print(x_val)
	print(Y_val)
	np_x_val = np.asarray(x_val)
	np_Y_val = np.asarray(Y_val)

	history = model.fit(x_train, Y_train,
	          batch_size=128, epochs=20,
	          verbose=2,
	          validation_data=(np_x_val, np_Y_val))

	
	score = model.evaluate(x_test, Y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	

	



if __name__ == '__main__':
	print("Your Program Here")
	#img = Image.open("bracco0.jpg")

	#cropped_img = cropImage(img,861,971,2107,2217)

	#cropped_img.show()

	#fr = open("downloadedFiles.txt")
	#for line in fr:
		#print(line.strip().split('\t')[5])
	img_arr = []
	arr = []
	labels = []
	l = np.array([])
	#preProcessImages(arr)
	path2 = r'/Users/rohansuri17/Downloads/Assignment5_SkeletonCode_v2/uncropped2/'

	fr = open("downloadedFiles.txt")
	label_as_num = -1
	for line in fr:
		#print("hi")
		img_arr.append(line.strip().split('\t')[7])
		arr.append(line.strip().split('\t')[5])
		labels.append(line.strip().split('\t')[0])
		#if("bracco" in fr):
			#label_as_num = 0
		#if("gilpin" in fr):
			#label_as_num = 1
		#if("harmon" in fr):
			#label_as_num = 2
		#if("butler" in fr):
			#label_as_num = 3
		#if("radcliffe" in fr):
			#label_as_num = 4 
		#if("vartan" in fr):
			#label_as_num = 5
		#l = np.append(l,label_as_num)
	#print(labels)
	#labels = ["Lorraine","Peri","Angie", "Gerard", "Daniel","Michael"]
	#preProcessImages(1)
	#img_arr_2 = np.array(img_arr)

	fr.close()

	fr2 = open("downloadedFiles.txt") 
	for lines in fr2:
		if("bracco" in lines):
			label_as_num = 0
		if("gilpin" in lines):
			label_as_num = 1
		if("harmon" in lines):
			label_as_num = 2
		if("butler" in lines):
			label_as_num = 3
		if("radcliffe" in lines):
			label_as_num = 4 
		if("vartan" in lines):
			label_as_num = 5
		l = np.append(l,label_as_num)
	#print(l)
	#print(l.shape)
	#print(labels)

	
		
	img = []
	images = np.array(img)
	#print(img)
	for i in range(0,len(img_arr)):
		try:
			x = np.array(Image.open(path2+img_arr[i]).convert('L'))
		except:
			print("Nope")
		#print(i)
		x = x.flatten()
		#print(x)
		#images = np.delete(images,0,0)
		#images = np.append(images, x)
		img.append(x)
		#print(img.shape)

		#images = np.vstack((images, x))
		#print(images)
	#print(img.shape)
	#img2= np.asarray(img)
	for i in range(0,len(img)):
		images = np.vstack((img,x))

	images = np.delete(images,2,0)

	#print(images)







	#l.reshape(658,1)
	trainFaceClassifier(images,l)
	#trainFaceClassifier_VGG(1,l)





	#trainFaceClassifier()
