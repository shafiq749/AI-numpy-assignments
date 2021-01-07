#Import the numpy package under the name np
import numpy as np
#Create a null vector of size 10
null_vector =  np.zeros(10)
print(null_vector)
#Create a vector with values ranging from 10 to 49
value_vector = np.arange(10,49)
print(value_vector)
#Find the shape of previous array in question 3
print(np.shape(value_vector))
#Print the type of the previous array in question 3
print(type(value_vector))
#Print the numpy version and the configuration
print(np.__version__)
print(np.show_config())
#Print the dimension of the array in question 3
print(np.ndim(value_vector))
#Create a boolean array with all the True values
bolarray = np.ones((2,2),dtype=bool)
print(bolarray)
#Create a two dimensional array
d2arr = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,22,33,44,55]])
print(np.ndim(d2arr))
#Create a three dimensional array
d3arr= np.ones((2,3,4))
print(np.ndim(d3arr))
#Reverse a vector (first element becomes last)
revarr = np.arange(10,20)
print(revarr[::-1])
#Create a null vector of size 10 but the fifth value which is 1
null_vector2 =  np.zeros(10)
null_vector2[5]=1
print(null_vector2[5])
#Create a 3x3 identity matrix
matrix3x3 = np.identity(3)
print(matrix3x3)
#arr = np.array([1, 2, 3, 4, 5])
#Convert the data type of the given array from int to float
arr = np.array([1, 2, 3, 4, 5])
print(arr.dtype)
ar2 = arr.astype('float64')
print(ar2.dtype)
#Multiply arr1 with arr2
arr1 = np.array([[1., 2., 3.],[4., 5., 6.]]) 
print(np.ndim(arr1))
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
res = np.multiply(arr1,arr2)
print(res)
#Make an array by comparing both the arrays provided above
comparearr = arr1 == arr2
print(comparearr)
#Extract all odd numbers from arr with values(0-9)
arr = np.array([0,1,2,3,4,5,6,7,8,9,10])
print(arr[arr%2==1])
#Replace all odd numbers to -1 from previous array
arr[arr%2==1]=-1
print(arr)
#Replace the values of indexes 5,6,7 and 8 to 12
arr = np.arange(10)
arr[5:9]=12
print(arr)
#Create a 2d array with 1 on the border and 0 inside
arr = np.ones((10,10))
arr[1:-1,1:-1]=0
print(arr)
#Replace the value 5 to 12
arr2d = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
arr2d[1][1]=12
print(arr2d)
#Convert all the values of 1st array to 64
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0]=64
print(arr3d)
#Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it
arr2r = np.arange(0,9).reshape(3,3)
print(np.ndim(arr2r))
print(arr2r[0,0:])  
#Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it
arr2r = np.array([[0,1,2,3,4],[5,6,7,8,9]])
print(arr2r[1][1])
#Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows
arrtwo = np.arange(0,9).reshape(3,3)
print(arrtwo)
print(arrtwo[0:2,2])
#Create a 10x10 array with random values and find the minimum and maximum values
randarray = np.random.randn(10,10)
print(np.min(randarray))
print(np.max(randarray))
#27.Find the common items between a and b
a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a,b))
#Find the positions where elements of a and b match
a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.in1d(a,b))
#Find all the values from array data where the values from array names are not equal to Will
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
print(data[names!='Will'])
#Find all the values from array data where the values from array names are not equal to Will and Joe
valuepass = (names!='Will') & (names!='Joe')
print(data[valuepass])
arry2 = np.arange(1,16).reshape(5,3)
print(arry2)
#Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16
arry2 = np.arange(1,17).reshape(2,2,4)
print(arry2)
#Swap axes of the array you created in Question 32
print(np.transpose(arry2))
#Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0
size10 = np.arange(10)
print(size10)
size10sqrt = np.sqrt(size10)
print(size10sqrt)
size10sqrt[size10sqrt<0.5]=0
print(size10sqrt)
#Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays
maxarr = np.random.randn(12)
print(maxarr)
maxarr1 = np.random.randn(12)
print(maxarr1)
mx = np.maximum(maxarr,maxarr1)
print(mx)
#Find the unique names and sort them out!
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
print(np.unique(names))
#From array a remove all items present in array b
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
print(np.setdiff1d(a, b))
#Following is the input NumPy array delete column two and insert following new column in its place.
sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
print(sampleArray)
newColumn = np.array([[10,10,10]])
sampleArray = np.delete(sampleArray, 1,1)
print(sampleArray)
sampleArray = np.insert(sampleArray , 1,newColumn , 1)
print(sampleArray)
#Find the dot product of the above two matrix
x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
print(np.dot(x,y))
#Generate a matrix of 20 random values and find its cumulative sum
matris = np.random.randn(20).reshape(5,4)
print(matris.cumsum())