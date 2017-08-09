
#========================================================== PART II =====================================================================
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy import misc
import numpy as np
import time
import os

#Loading dataset 1
x1 = []; y1 = []						
os.chdir("data1")
arr = os.listdir()
for a in arr:
	os.chdir(a)
	images = os.listdir()
	for i in images:
		img = misc.imread(i, flatten = True).flatten()
		x1.append(img)
		y1.append(a)
	os.chdir("../")
	
x1 = np.array(x1); y1 = np.array(y1)
os.chdir("../")

#Loading dataset 2
x2 = []; y2 = []
os.chdir("data2")
arr = os.listdir()
for a in arr:
	os.chdir(a)
	images = os.listdir()
	for i in images:
		img = misc.imread(i, flatten = True).flatten()
		x2.append(img)
		y2.append(a)
	os.chdir("../")
	
x2 = np.array(x2); y2 = np.array(y2)
os.chdir("../")


#------------------------------------------------------(a)-----------------------------------------------------------------


size1 = x1.shape 
size2 = x2.shape

avg_face1 = np.array([np.sum(x1[:,i]) for i in range(size1[1])]) / size1[0]
avg_face2 = np.array([np.sum(x2[:,i]) for i in range(size2[1])]) / size2[0]

misc.imshow(avg_face1.reshape(50,37))
misc.imshow(avg_face2.reshape(112,92))


#------------------------------------------------------(b)-----------------------------------------------------------------


x1 = np.array([x1[i] - avg_face1 for i in range(size1[0])]) 
x2 = np.array([x2[i] - avg_face2 for i in range(size2[0])])

u1, d1, v1 = np.linalg.svd(x1)
u2, d2, v2 = np.linalg.svd(x2)

pc1 = v1[:50]
pc2 = v2[:50]

#------------------------------------------------------(c)-----------------------------------------------------------------


eigen_face1 = np.array([pc1[i] - np.min(pc1[i]) for i in range(50)])
eigen_face2 = np.array([pc2[i] - np.min(pc2[i]) for i in range(50)])

eigen_face1 = np.array([eigen_face1[i] * (255 / np.max(eigen_face1[i])) for i in range(50)])
eigen_face2 = np.array([eigen_face2[i] * (255 / np.max(eigen_face2[i])) for i in range(50)])

print("Top 5 eigen faces for Dataset 1:")
for i in range(5):
	misc.imshow(eigen_face1[i].reshape(50,37))

print("Top 5 eigen faces for Dataset 2:")
for i in range(5):
	misc.imshow(eigen_face2[i].reshape(112,92))


#------------------------------------------------------(d)-----------------------------------------------------------------


x1 = np.array([x1[i] + avg_face1 for i in range(size1[0])]) 
x2 = np.array([x2[i] + avg_face2 for i in range(size2[0])])

projected_x1 = np.dot(x1, pc1.T)
for i in range(50):
	projected_x1[:,i] = (projected_x1[:,i] - np.mean(projected_x1[:,i])) / np.std(projected_x1[:,i])

projected_x2 = np.dot(x2, pc2.T)
for i in range(50):
	projected_x2[:,i] = (projected_x2[:,i] - np.mean(projected_x2[:,i])) / np.std(projected_x2[:,i])


#------------------------------------------------------(e)-----------------------------------------------------------------


print("Dataset 1: ")
start_time = time.time()
clf = SVC(kernel = 'linear')
scores = cross_val_score(clf, x1, y1, cv = 10)
print(scores)
print(np.mean(scores))
print("Using original set of attributes --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
scores = cross_val_score(clf, projected_x1, y1, cv = 10)
print(scores)
print(np.mean(scores))
print("Using projected set of attributes --- %s seconds ---" % (time.time() - start_time))

print("Dataset 2: ")
start_time = time.time()
scores = cross_val_score(clf, x2, y2, cv = 10)
print(scores)
print(np.mean(scores))
print("Using original set of attributes --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
scores = cross_val_score(clf, projected_x2, y2, cv = 10)
print(scores)
print(np.mean(scores))
print("Using projected set of attributes --- %s seconds ---" % (time.time() - start_time))


#------------------------------------------------------(f)-----------------------------------------------------------------

projected_x1 = np.dot(x1, pc1.T)
projected_x2 = np.dot(x2, pc2.T)

choice = 0
while(choice != 3):
	print("1. Get a projected image from Dataset 1 \n 2. Get a projected image from Dataset 2 \n 3. Exit")
	choice = int(input())
	if(choice == 1):
		print("Enter the face ID: ")
		id = int(input())
		misc.imshow(x1[id].reshape(50,37))
		misc.imshow((np.dot(projected_x1[id], pc1) + avg_face1).reshape(50,37))
	elif(choice == 2):
		print("Enter the face ID: ")
		id = int(input())
		misc.imshow(x2[id].reshape(112,92))
		misc.imshow((np.dot(projected_x2[id], pc2) + avg_face2).reshape(112,92))

print("Program Completed!")