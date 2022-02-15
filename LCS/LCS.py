
# input protein sequences from txt files
protein_a_file=open("d:/Protein_A.txt",'r')
protein_b_file=open("d:/Protein_B.txt",'r')
protein_a=protein_a_file.read().replace('\n','')
protein_b=protein_b_file.read().replace('\n','')

# lengths of each sequence
len1 = len(protein_a)
len2 = len(protein_b)

# generate the blank LCS matrix and LCS index number for printing
LCS_matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
LCS=[]


# fill the columns in the LCS matrix
for i in range(0, len1):
    for j in range(0, len2):
        # input 0 to the first row and column
        if (i==0) or (j==0):
             LCS_matrix[i][j]=0
        # if Protein A and B are same, input (left-upper side value +1) to current position
        if(protein_a[i]==protein_b[j]):
            LCS_matrix[i][j]=LCS_matrix[i-1][j-1]+1
        else:
            # if Protein A and B are different, input (bigger value +1) to current position
            if(LCS_matrix[i-1][j]>LCS_matrix[i][j-1]):
                LCS_matrix[i][j]=LCS_matrix[i-1][j]
            else:
                LCS_matrix[i][j]=LCS_matrix[i][j-1]


# generate LCS index for printing
i=len1-1
j=len2-1
# start the last position (the last right-bottom value)
while LCS_matrix[i][j]!=0:
    # if left value is as same as current, move to left
    if(LCS_matrix[i][j]==LCS_matrix[i][j-1]):
        j=j-1
    # if upper value is as same as current, move to upper
    elif (LCS_matrix[i][j]==LCS_matrix[i-1][j]):
        i=i-1
    # if both values are different, move left-upper
    elif (LCS_matrix[i][j]-1==LCS_matrix[i-1][j-1]):
        # input current index to LCS
        LCS.append(i)
        i=i-1
        j=j-1



print("Protein A length : ",len1)
print("Protein B length : ",len2)
print("The Longest Common Sequence : ")
# for printing LCS
m=1
for k in range(0, len(LCS)):
    # print index
    if(m%80==1):
        print('{0:4d}'.format(m),' ',end='')
    # print LCS value
    print(protein_a[LCS[len(LCS)-1-k]],end='')

    # print index
    if(m%80==0):
        print(' ',m)
    m = m + 1

print("\n\nThe Longest Common Sequence(length) : ", LCS_matrix[len1-1][len2-1])
