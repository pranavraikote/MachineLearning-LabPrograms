#Import modules
import math

#Sigmoid Function
def sigmoid(x):
	y= 1/(1+math.exp(-x))
	return y

#Define Inputs and Target for XOR Gate
#Input1
x1=[0,0,1,1]		
#Input2
x2=[0,1,0,1]		
#Target
t=[0,1,1,0]  		

# Initialize random weights and biases
# Hidden layer First Perceptron
b1=-0.3
w11=0.2
w21= 0.1
# Hidden Layer Second Perceptron
b2=0.5
w12=-0.4
w22=0.3
# Output Layer Perceptron
b3=-0.4
w13=-0.3
w23=0.2

#Initilize few values
error=0
iteration=0
train=True
print("weight are:")
print("w11 : %4.2f w12: %4.2f w21: %4.2f w22: %4.2f w13: %4.2f  w23: %4.2f  \n" %(w11,w12,w21,w22,w13,w23))


# Training 
while(train):

	for i in range(len(x1)):

		#Input for each Perceptron of Hidden Layer
		z_in1=b1+x1[i]*w11+x2[i]*w21
		z_in2=b2+x1[i]*w12+x2[i]*w22

		#Computing Activation Function Output
		z1=round(sigmoid(z_in1),5)
		z2=round(sigmoid(z_in2),5)

		#Output Layer Forward Pass
		y_in=b3+z1*w13+z2*w23
		y=round(sigmoid(y_in),5)

		#Error Computation
		del_k=round((t[i]-y)*y*(1-y),5)
		error=del_k
        
		#Back Propagation
		#Weight Update for Output Layer
		w13=round(w13+del_k*z1,5)
		w23=round(w23+del_k*z2,5)
		b3=round(b3+del_k,5)

		#Error calculation for Hidden Layer
		del_1=del_k*w13*z1*(1-z1)
		del_2=del_k*w23*z2*(1-z2)

		#Update Weight and Biases
		b1=round(b1+del_1,5)
		w11=round(w11+del_1*x1[i],5)
		w12=round(w12+del_2*x1[i],5)

		b2=round(b2+del_2,4)
		w21=round(w21+del_1*x2[i],5)
		w22=round(w22+del_2*x2[i],5)

		iteration=iteration+1

	if(iteration==100000):
		train=False

#Sample code for printing the Output Predictions	    
for i in range(len(x1)):
	
	#Input for each Perceptron of Hidden Layer
	z_in1=b1+x1[i]*w11+x2[i]*w21
	z_in2=b2+x1[i]*w12+x2[i]*w22

	#Computing Activation Function Output
	z1=round(sigmoid(z_in1),5)
	z2=round(sigmoid(z_in2),5)

	#Output layer Forward Pass
	y_in=b3+z1*w13+z2*w23
	y=round(sigmoid(y_in),5)
	print("predicted output ",y)