#final
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
le=preprocessing.LabelEncoder()

from sklearn.tree import DecisionTreeClassifier

def predict_colleges(percentile, caste, course):

    df1=pd.read_csv("C:\\Users\\ombha\\Downloads\\minal_25_10_23.csv")
    df=df1.copy()
    len(df)
    #unique values for printing purpose only
    cous=np.unique(df['Course'])
    colg=np.unique(df['College'])
    cast=np.unique(df['Caste'])

    df['College'] = le.fit_transform(df['College'].astype(str))
    df['Course'] = le.fit_transform(df['Course'])
    df['Caste'] = le.fit_transform(df['Caste'])

    #label encoding
    codeclg=[]
    for i in range(len(colg)):
        codeclg.append(i)

    # print("codeclg",codeclg)

    codecous=[]
    for i in range(len(cous)):
        codecous.append(i)

    # print("codecous",codecous)

    codecast=[]
    for i in range(len(cast)):
        codecast.append(i)
    # print("codecast",codecast)

    # print("\ncolleges:\n")
    le.fit(colg)
    encoded = dict(zip(le.classes_, le.transform(le.classes_)))
    # for i in encoded.items():
    #     print(i)
    
    '''print("Choose Course and Caste from following Code:")
    print("\nCourse:")
    le.fit(cous)
    encoded = dict(zip(le.classes_, le.transform(le.classes_)))
    for i in encoded.items():
        print(i)

    print("\nCaste:")
    le.fit(cast)
    encoded = dict(zip(le.classes_, le.transform(le.classes_)))
    for i in encoded.items():
        print(i)'''

    # sorted_df = df1.sort_values(by=["Rank"], ascending=True)

    df['College']=df['College'].replace(colg,codeclg)
    bak_college=np.array(df['College'])
    df.head()

    df['Course']=df['Course'].replace(cous,codecous)
    bak_course=np.array(df['Course'])
    df.head()

    df['Caste']=df['Caste'].replace(cast,codecast)
    bak_caste=np.array(df['Caste'])
    df.head()

    # def userip():
    #     col=df.columns.tolist()[1:4]
    #     # print(col)
    #     usrip=[]
    #     for i in col:
    #         print("==================================================")
    #         usrip.append(eval(input(i+": ")))

    #     return usrip

    # print("You may have change to get entrance in: \n")

    # usrip=userip()
    # # accuracy_scores = []

    flist=[]
    n=10
    while(n!=0):

        X = df.drop(columns=["College"])
        y=df['College']
        X = X.values
        y = y.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
        clfdt = DecisionTreeClassifier()
        clfdt.fit(X_train,y_train)
        #prediction for accuracy
        preddt=clfdt.predict(X_test)
        scrdt=clfdt.score(X_test,y_test)
        scrdt=eval("%0.5f"%scrdt)*100
        # accuracy_scores.append(scrdt)
        # accuracy = accuracy_score(y_test, preddt)

        df.head(2)

        userpreddt=clfdt.predict([[course, caste, percentile]])
        predclgcode=int(userpreddt)

        #   print(colg[codeclg.index(userpreddt[0])],end="\n")
        flist.append(colg[codeclg.index(userpreddt[0])])

        indexCollege = df[ df['College'] == predclgcode].index
        df.drop(indexCollege , inplace=True)
        df.head(15)
        # df=df.Rank
        n=n-1
    return flist

# print("Algorithm Score: ",scrdt)
# print("Accuracy on the test set: {:.2f}%".format(accuracy * 100))

# plt.plot(range(1, 16), accuracy_scores)
# plt.xlabel("Iteration")
# plt.ylabel("Accuracy")
# plt.title("Accuracy Over Iterations")
# plt.show()

# print("user input you typed is",usrip)

'''percentile = float(input("Enter percentile: "))
caste = int(input("Enter caste: "))
course = int(input("Enter course: "))

[percentile, caste, course] = [percentile, caste, course]

flist=predict_colleges(percentile, caste, course)
for element in flist:
    print(element)'''

# Calculate the mean accuracy score
# mean_accuracy = np.mean(accuracy_scores)
# print("Mean Accuracy:", mean_accuracy)