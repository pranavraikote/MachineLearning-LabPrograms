#Import modules
import csv
num_attributes = 6
a = []

#Read the dataset file
csvfile = open('FindS.csv', 'r')
reader = csv.reader(csvfile)

for row in reader:
    a.append(row)
print(row)

#Print initial Hypothesis
print("\nThe initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes
print(hypothesis)

for j in range(0,num_attributes):
    hypothesis[j] = a[0][j]
    
print("\nFind S: Finding a Maximally Specific Hypothesis\n")
for i in range(0,len(a)):
    if a[i][num_attributes]=='Yes':
        for j in range(0,num_attributes):
            if a[i][j]!=hypothesis[j]:
                hypothesis[j]='?'
            else:
                hypothesis[j]= a[i][j] 
        print("For Training Instance No:{0}  the  hypothesis  is ".format(i),hypothesis)
print("\n The Maximally Specific Hypothesis for a given Training Examples :\n")
print(hypothesis)
