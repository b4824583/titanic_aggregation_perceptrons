import pandas as pd
import numpy as np
# 繪圖相關套件
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib.gridspec as gridspec
import seaborn as sns
from IPython.display import display
plt.style.use( 'ggplot' ) 
import math
def sigmoid(x):
    return 1/(1+math.exp(-x))
# 定義用來統計欄位缺漏值總數的函數
def Missing_Counts( Data ) : 
    missing = Data.isnull().sum()  # 計算欄位中缺漏值的數量 
    missing = missing[ missing>0 ]
    missing.sort_values( inplace=True ) 
    
    Missing_Count = pd.DataFrame( { 'ColumnName':missing.index, 'MissingCount':missing.values } )  # Convert Series to DataFrame
    Missing_Count[ 'Percentage(%)' ] = Missing_Count['MissingCount'].apply( lambda x:round(x/Data.shape[0]*100,2) )
    return  Missing_Count


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
submit = pd.read_csv('gender_submission.csv')
print(df_train.shape)
# exit()
# print( 'train :' )
# display( Missing_Counts(df_train) )

# print( 'test :' )
# display( Missing_Counts(df_test) )
df_data=df_train.append(df_test)
# plt.figure( figsize=(10,5) )
# df_data['Survived'].value_counts().plot( kind='pie', colors=['lightcoral','skyblue'], autopct='%1.2f%%' )
# plt.title( 'Survival' )  # 圖標題
# plt.ylabel( '' )
# plt.show()

# selected_cols = ['Sex','Pclass','Embarked','SibSp','Parch']

# plt.figure( figsize=(10,len(selected_cols)*5) )
# gs = gridspec.GridSpec(len(selected_cols),1)    
# for i, col in enumerate( df_data[selected_cols] ) :        
#     ax = plt.subplot( gs[i] )
#     sns.countplot( df_data[col], hue=df_data.Survived, palette=['lightcoral','skyblue'] )
#     ax.set_yticklabels([])
#     ax.set_ylabel( 'Counts' )
#     ax.legend( loc=1 )   # upper right:1 ; upper left:2
#     for p in ax.patches:
#         ax.annotate( '{:,}'.format(p.get_height()), (p.get_x(), p.get_height()+1.5) )
# plt.show()

# print(df_data["Sex"].shape)






# print(df_train["Sex"])
# print(df_train["Pclass"])
# print(df_train["Embarked"])
# print(df_train["SibSp"])
# print(df_train["Parch"])
# exit()
df_train['Embarked'].fillna( 'S', inplace=True )
df_train['Fare'].fillna( df_train.Fare.median(), inplace=True )

#### 
#[0] sex
#[1] age
#[2] pclase
#[3] fare
#[4] family
#[5] Embarked
input_data=np.zeros((891,7))
input_data[:,6]=1.0
# print(input_data[0])
for i in range(891):
    if(df_train["Sex"][i]=="male"):
        input_data[i][0]=1
    else:
        input_data[i][0]=0
    if(df_train["Age"][i]>=16):
        input_data[i][1]=1
    else:
        input_data[i][1]=0
    
    input_data[i][2]=df_train["Pclass"][i]

    input_data[i][4]=df_train["SibSp"][i]+df_train["Parch"][i]
    # if(df_train[""])
    if(df_train["Embarked"][i]=="S"):
        input_data[i][5]=1
    elif(df_train["Embarked"][i]=="C"):
        input_data[i][5]=2
    elif(df_train["Embarked"][i]=="Q"):
        input_data[i][5]=3
    input_data[i][3]=df_train["Fare"][i]
    # print(input_data[i])
weight=np.random.rand(7)
output=np.zeros((891,1),dtype=float)
# single_weight=weight
# single_input=input_data[0]
# value=weight.T.dot(single_input)
# print(value)
# exit()
# output=sigmoid(value)
# print()
# input_data[0]
# exit()
# df_answer=pd.read_csv('titanic/gender_submission.csv', sep=',')



# for i in range(df_train.shape[0]):
#     # print(df_train["Survived"][i])
#     single_input=input_data[i]
#     value=weight.T.dot(single_input)
#     value=sigmoid(value)
#     loss=(df_train["Survived"][i]-value)**2
#     print(loss)
    # weight[0]=weight[0]-# Graident descent

###

#min regression
CEE=0.0
learning_rate=0.001

CEE_sum=0.0
for epoch in range(150):
    print("epoch:"+"\t"+str(epoch))
    for i in range(df_train.shape[0]):

        if(df_train["Survived"][i]==1):
            # print("No "+str(i)+":\t \t live")
            survived=1
        else:
            # print("No "+str(i)+":\t \t dead")
            survived=-1
        for j in range(weight.shape[0]):

            CEE=math.log(1+math.exp((weight[j]*input_data[i][j]*survived)*-1))+CEE
        CEE=CEE/weight.shape[0]
        predicted_value=sigmoid(weight.T.dot(input_data[i]))
        if(predicted_value>=0.5):
            # print("predicte:\t live")
            pass

        else:
            # print("predict:\t dead")
            pass

        # print("predicted value:\t"+str(predicted_value))

        # print("CEE:\t \t"+str(round(CEE,4)))
        # print("---------------------------\n")
        for j in range(weight.shape[0]):
            if(df_train["Survived"][i]==1):
                survived=1
            else:
                survived=-1
            # CEE=math.log(1+math.exp((weight[j]*input_data[i][j]*survived)*-1))+CEE
            weight[j]=weight[j]-(sigmoid((-1)*weight.T.dot(input_data[i])*survived)*(-1*survived*input_data[i][j]))*learning_rate
        # print(str(weight)+"\n")
    print("-------------------------------------------------------------------------")
    correct=0
    percent=0.0
    for i in range(df_train.shape[0]):
        if(df_train["Survived"][i]==1):
            survived=1
        else:
            survived=-1
        for j in range(weight.shape[0]):
            CEE=math.log(1+math.exp((weight[j]*input_data[i][j]*survived)*-1))+CEE
        CEE=CEE/weight.shape[0]
        predicted_value=sigmoid(weight.T.dot(input_data[i]))
        if(predicted_value>=0.5):
            # print("predicte:\t live")
            pass
        else:
            # print("predict:\t dead")
            pass
        if(survived==1 and predicted_value>=0.5):
            correct+=1
        elif(survived==-1 and predicted_value<0.5):
            correct+=1
    print("correct:"+str(correct)+"/"+str(df_train.shape[0]))
    percent=correct/df_train.shape[0]
    print("percent:"+str(round(percent,4)))
#-----------------------------------------
#test
#-----------------------------------------

        
    # for i in range(df_train.shape[0]):
    #     for j in range(weight.shape[0]):
    #         weight[j]=weight[j]-(sigmoid((-1)*weight.T.dot(input_data[i])*survived)*(-1*survived*input_data[i][j]))*learning_rate

    # for i in range()
    # print(CEE)

df_test = pd.read_csv('test.csv')
submit = pd.read_csv('gender_submission.csv')

print("-------------------------------------------------------------------------")
correct=0
percent=0.0
answer=np.zeros((df_test.shape[0],2),dtype=int)

for i in range(df_test.shape[0]):

    if(df_test["Sex"][i]=="male"):
        input_data[i][0]=1
    else:
        input_data[i][0]=0
    if(df_test["Age"][i]>=16):
        input_data[i][1]=1
    else:
        input_data[i][1]=0
    
    input_data[i][2]=df_test["Pclass"][i]
    input_data[i][3]=df_test["Fare"][i]
    input_data[i][4]=df_test["SibSp"][i]+df_test["Parch"][i]
    # if(df_train[""])
    if(df_test["Embarked"][i]=="S"):
        input_data[i][5]=1
    elif(df_test["Embarked"][i]=="C"):
        input_data[i][5]=2
    elif(df_test["Embarked"][i]=="Q"):
        input_data[i][5]=3



    if(submit["Survived"][i]==1):
        whether_survived=1
    else:
        whether_survived=-1

    predicted_value=sigmoid(weight.T.dot(input_data[i]))
    answer[i][0]=df_test["PassengerId"][i]
    if(predicted_value>=0.5):
        answer[i][1]=1
        # print("predicte:\t live")
        pass
    else:
        answer[i][1]=0
        # print("predict:\t dead")
        pass
    if(whether_survived==1 and predicted_value>=0.5):
        correct+=1
    elif(whether_survived==-1 and predicted_value<0.5):
        correct+=1
print("###############")
print("------test-----")
print("###############")
answer_csv={"PassengerId":answer[:,0],"Survived":answer[:,1]}

answer_csv_df = pd.DataFrame(answer_csv)
answer_csv_df.to_csv("kaggel_upload.csv",index=False)
print("correct:"+str(correct)+"/"+str(df_test.shape[0]))
percent=correct/df_test.shape[0]
print("percent:"+str(round(percent,4)))