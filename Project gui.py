import warnings
warnings.filterwarnings("ignore")
from tkinter import *
import tkinter.messagebox
from tkinter import filedialog
import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
import pandas_ml as pdml
from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc,roc_curve
root=Tk()
root.geometry("800x600")
root.title("Credit Card Fraud Detection")
var=StringVar()
choice=StringVar()
choice.set("-select-")
d={'ACC':[],'TPR':[]}
aucarr={'auc':[]}
def logistic_regression():
    print("------------------------LOGISTIC REGRESSION-----------------------")
    df = pd.read_csv(var.get(), low_memory=False)
    df = df.sample(frac=1).reset_index(drop=True)
    frauds = df.loc[df['Class'] == 1]
    non_frauds = df.loc[df['Class'] == 0]
    print("\n")
    print("We have", len(frauds), "fraud data points and", len(non_frauds), "nonfraudulent data points.\n")
    X = df.iloc[:,:-1]
    y = df['Class']
    print("X and y sizes, respectively:", len(X), len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
    '''print("\nTrain and test sizes, respectively:", len(X_train), len(y_train), "|", len(X_test), len(y_test))
    print("Total number of frauds:", len(y.loc[df['Class'] == 1]))
    print("Number of frauds on y_test:", len(y_test.loc[df['Class'] == 1]))
    print("Number of frauds on y_train:", len(y_train.loc[df['Class'] == 1]))'''
    logistic = linear_model.LogisticRegression(C=1e5)
    logistic.fit(X_train, y_train)
    print("\nScore: ", logistic.score(X_test, y_test))
    y_predicted = np.array(logistic.predict(X_test))
    y_right = np.array(y_test)
    confusion_matrix = ConfusionMatrix(y_right, y_predicted)
    print("\n\nConfusion matrix:\n%s" % confusion_matrix)
    #confusion_matrix.plot(normalized=True)
    T = Text(root, height=60, width=60)
    T.pack(pady=20,side=BOTTOM, fill=Y)
    for l in confusion_matrix.stats():
        T.insert(END,[l,confusion_matrix.stats()[l]])
        T.insert(END,"\n")
    d['ACC'].append(confusion_matrix.stats()['ACC']*100)
    d['TPR'].append(confusion_matrix.stats()['TPR']*100)
    fpr,tpr,thresholds=roc_curve(y_right, y_predicted)
    aucarr['auc'].append(auc(fpr,tpr))
    #plt.show()
def logistic_reg_smote():
    l=1
    if(l==1):
        print("------------------------LOGISTIC REGRESSION WITH SMOTE-----------------------")
        df = pd.read_csv(var.get(), low_memory=False)
        df = df.sample(frac=1).reset_index(drop=True)
        frauds = df.loc[df['Class'] == 1]
        non_frauds = df.loc[df['Class'] == 0]
        print("\nWe have", len(frauds), "fraud data points and", len(non_frauds), "nonfraudulent data points.")
        X = df.iloc[:,:-1]
        y = df['Class']

        print("X and y sizes, respectively:", len(X), len(y))
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
        print("Train and test sizes, respectively:", len(X_train), len(y_train), "|", len(X_test), len(y_test))
        print("Total number of frauds:", len(y.loc[df['Class'] == 1]))
        print("Number of frauds on y_test:", len(y_test.loc[df['Class'] == 1]))
        print("Number of frauds on y_train:", len(y_train.loc[df['Class'] == 1]))
        df2 = pdml.ModelFrame(X_train, target=y_train)
        sampler = df2.imbalance.over_sampling.SMOTE()
        sampled = df2.fit_sample(sampler)
        print("\nSize of training set after over sampling:", len(sampled))
        X_train_sampled = sampled.iloc[:,1:]
        y_train_sampled = sampled['Class']


        logistic = linear_model.LogisticRegression(C=1e5)
        logistic.fit(X_train_sampled, y_train_sampled)
        print("Score: ", logistic.score(X_test, y_test))
        y_predicted1 = np.array(logistic.predict(X_test))
        y_right1 = np.array(y_test)

        confusion_matrix1 = ConfusionMatrix(y_right1, y_predicted1)
        print("\n\nConfusion matrix:\n%s" % confusion_matrix1)
        #confusion_matrix1.plot(normalized=True)
        T = Text(root, height=60, width=60)
        T.pack(pady=20,side=BOTTOM, fill=Y)
        for l in confusion_matrix1.stats():
            T.insert(END,[l,confusion_matrix1.stats()[l]])
            T.insert(END,"\n")
        d['ACC'].append(confusion_matrix1.stats()['ACC']*100)
        d['TPR'].append(confusion_matrix1.stats()['TPR']*100)
        fpr,tpr,thresholds=roc_curve(y_right1, y_predicted1)
        aucarr['auc'].append(auc(fpr,tpr))
        
        #plt.show()
def random_forest():
    l=1
    if(l==1):
        print("------------------------RANDOM FOREST-----------------------")
        df = pd.read_csv(var.get(), low_memory=False)
        df = df.sample(frac=1).reset_index(drop=True)
        frauds = df.loc[df['Class'] == 1]
        non_frauds = df.loc[df['Class'] == 0]
        print("\nWe have", len(frauds), "fraud data points and", len(non_frauds), "nonfraudulent data points.")
        X = df.iloc[:,:-1]
        y = df['Class']

        print("X and y sizes, respectively:", len(X), len(y))
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
        print("Train and test sizes, respectively:", len(X_train), len(y_train), "|", len(X_test), len(y_test))
        print("Total number of frauds:", len(y.loc[df['Class'] == 1]))
        print("Number of frauds on y_test:", len(y_test.loc[df['Class'] == 1]))
        print("Number of frauds on y_train:", len(y_train.loc[df['Class'] == 1]))
        clf= RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_predicted1 =np.array(clf.predict(X_test))
        y_right1=np.array(y_test)
        confusion_matrix1=ConfusionMatrix(y_right1,y_predicted1)
        print("\n\nConfusion matrix:\n%s" % confusion_matrix1)
        #confusion_matrix1.plot(normalized=True)
        T = Text(root, height=60, width=60)
        T.pack(pady=20,side=BOTTOM, fill=Y)
        for l in confusion_matrix1.stats():
            T.insert(END,[l,confusion_matrix1.stats()[l]])
            T.insert(END,"\n")
        d['ACC'].append(confusion_matrix1.stats()['ACC']*100)
        d['TPR'].append(confusion_matrix1.stats()['TPR']*100)
        fpr,tpr,thresholds=roc_curve(y_right1, y_predicted1)
        aucarr['auc'].append(auc(fpr,tpr))

        #plt.show()
def random_for_smote():
    l=1
    if(l==1):
        print("------------------------RANDOM FOREST WITH SMOTE-----------------------")
        df = pd.read_csv(var.get(), low_memory=False)
        df = df.sample(frac=1).reset_index(drop=True)
        frauds = df.loc[df['Class'] == 1]
        non_frauds = df.loc[df['Class'] == 0]
        print("We have", len(frauds), "fraud data points and", len(non_frauds), "nonfraudulent data points.\n")
        X = df.iloc[:,:-1]
        y = df['Class']
        print("X and y sizes, respectively:", len(X), len(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
        print("\nTrain and test sizes, respectively:", len(X_train), len(y_train), "|", len(X_test), len(y_test))
        print("Total number of frauds:", len(y.loc[df['Class'] == 1]))
        print("Number of frauds on y_test:", len(y_test.loc[df['Class'] == 1]))
        print("Number of frauds on y_train:", len(y_train.loc[df['Class'] == 1]))
        df2 = pdml.ModelFrame(X_train, target=y_train)
        sampler = df2.imbalance.over_sampling.SMOTE()
        sampled = df2.fit_sample(sampler)
        print("\nSize of training set after over sampling:", len(sampled))
        X_train_sampled = sampled.iloc[:,1:]
        y_train_sampled = sampled['Class']
        
        clf= RandomForestClassifier()
        clf.fit(X_train_sampled, y_train_sampled)
        y_predicted1 =np.array(clf.predict(X_test))
        y_right1=np.array(y_test)
        confusion_matrix1=ConfusionMatrix(y_right1,y_predicted1)
        print("\n\nConfusion matrix:\n%s" % confusion_matrix1)
        #confusion_matrix1.plot(normalized=True)
        T = Text(root, height=60, width=60)
        T.pack(pady=20,side=BOTTOM, fill=Y)
        for l in confusion_matrix1.stats():
            T.insert(END,[l,confusion_matrix1.stats()[l]])
            T.insert(END,"\n")
        d['ACC'].append(confusion_matrix1.stats()['ACC']*100)
        d['TPR'].append(confusion_matrix1.stats()['TPR']*100)
        fpr,tpr,thresholds=roc_curve(y_right1, y_predicted1)
        aucarr['auc'].append(auc(fpr,tpr))

        #plt.show()
def choose():
    tempdir = filedialog.askopenfilename(parent=root, initialdir= "C:/Users/Kaushik/Desktop/", title='Please select a directory')
    var.set(tempdir)
    if(len(var.get())>0):
        mEntry.insert(0,var)
def run():
    if(len(var.get())==0):
        tkinter.messagebox.showinfo(title="Dialog Box", message="Cannot upload empty file!")
    elif(not(var.get()).endswith('csv')):
        tkinter.messagebox.showinfo(title="Dialog Box", message="Unsupported format of the file\n The file should be csv")
    if((choice.get())=='-select-'):
            tkinter.messagebox.showinfo(title="Dialog Box", message="Please select the Algorithm")
    elif((var.get()).endswith('csv') and (choice.get())=='Kmeans'):
        print("------------------------"+str(choice.get())+"-----------------------")
        print("\n")
        df = pd.read_csv(var.get(), low_memory=False)
        #print(df.head())
        X = df.iloc[:,:-1]
        y = df['Class']
        X_scaled = scale(X)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size = 0.33, random_state=500)
        kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
        kmeans.fit(X_train)
        h = .01 
        x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
        y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
        plt.plot(X_reduced[:, 0], X_reduced[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
        plt.title('K-means clustering on the credit card fraud dataset\n'
          'Centroids are marked with white cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        predictions = kmeans.predict(X_test)
        pred_fraud = np.where(predictions == 1)[0]
        real_fraud = np.where(y_test == 1)[0]
        false_pos = len(np.setdiff1d(pred_fraud, real_fraud))
        pred_good = np.where(predictions == 0)[0]
        real_good = np.where(y_test == 0)[0]
        false_neg = len(np.setdiff1d(pred_good, real_good))
        false_neg_rate = false_neg/(false_pos+false_neg)
        accuracy = (len(X_test) - (false_neg + false_pos)) / len(X_test)
        print("Accuracy:\n", accuracy)
        print("False negative rate (with respect to misclassifications): ", false_neg_rate)
        print("False negative rate (with respect to all the data): ", false_neg / len(predictions))
        print("False negatives, false positives, mispredictions:", false_neg, false_pos, false_neg + false_pos)
        print("Total test data points:", len(X_test))
        plt.show()
    elif((var.get()).endswith('csv') and (choice.get())=='Logistic Regression'):
        logistic_regression()
 
    elif((var.get()).endswith('csv') and (choice.get())=='Logistic Regression with SMOTE'):
        logistic_reg_smote()
    elif((var.get()).endswith('csv') and (choice.get())=='Random Forest'):
        random_forest()
    elif((var.get()).endswith('csv') and (choice.get())=='Random Forest with SMOTE'):        
        random_for_smote()
def compare():
    print("************************************COMPARISON*****************************************************")
    if(len(var.get())==0):
        tkinter.messagebox.showinfo(title="Dialog Box", message="Cannot upload empty file!")
    elif(not(var.get()).endswith('csv')):
        tkinter.messagebox.showinfo(title="Dialog Box", message="Unsupported format of the file\n The file should be csv")
    d['ACC']=[]
    d['TPR']=[]
    aucarr['auc']=[]
    logistic_regression()
    logistic_reg_smote()
    random_forest()
    random_for_smote()
    objects = ('Logistic Regression','Logistic with SMOTE','Random forest','Random forest with SMOTE')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, aucarr['auc'], align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Area Under Curve')
    plt.xlabel('Models')
    plt.title('Algorithms Comparisons')
    plt.show()

frame=Frame(root)
frame.pack()
button=Button(frame,text="Choose File",command=choose)
button.pack(padx=50,pady=50,side=LEFT)
choices = ['Kmeans','Logistic Regression','Random Forest','Random Forest with SMOTE','Logistic Regression with SMOTE']
popupMenu = OptionMenu(frame,choice, *choices)
popupMenu.pack(side=LEFT)
slogan=Button(frame,text="Performance Measures",command=run)
slogan.pack(padx=50,pady=50,side=LEFT)
button1=Button(frame,text="Compare",command=compare)
button1.pack(padx=50,pady=50,side=LEFT)
mEntry= Entry(root,width=60,textvariable=var).pack()

root.mainloop()
