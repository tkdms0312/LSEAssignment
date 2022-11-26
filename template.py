#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/tkdms0312/LSEAssignment.git

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

pipe = make_pipeline(
    StandardScaler(),
    RandomForestClassifier()
    )

svm_pipe = make_pipeline(
	StandardScaler(),
	SVC()
    )

def load_dataset(dataset_path):
	return pd.read_csv(dataset_path)


def dataset_stat(dataset_df):
    
    #groupby 안쓰고 class 별 데이터 개수 출력한 방식
    #entCnt = 0;
    #cnt0 = 0
    #cnt1 = 0
    #while entCnt < dataset_df.shape[0]:
        #if dataset_df.target[entCnt] == 0: 
            #cnt0 += 1
        #elif dataset_df.target[entCnt] == 1:
            #cnt1 += 1
        #entCnt += 1           
        
    return dataset_df.shape[1]-1,dataset_df.groupby("target").size()[0], dataset_df.groupby("target").size()[1]

def split_dataset(dataset_df, testset_size):
	X = dataset_df.drop(columns="target", axis=1)
	Y = dataset_df["target"]
	x_Train, x_Test, y_Train, y_Test = train_test_split(X, Y, test_size=testset_size)
    #x = dataset_df.drop(columns="target", axis=1)
	#x = dataset_df_data
	#y = dataset_df["target"]
	#y = dataset_df.target
	#x_train, x_test, y_train, y_test = train_test_split(x, y, testset_size, random_state=1)
	return x_Train, x_Test, y_Train, y_Test
    
def decision_tree_train_test(x_train, x_test, y_train, y_test):
	dt_cls = DecisionTreeClassifier() #decision tree 모듈
	dt_cls.fit(x_train, y_train) #decision tree 학습
    
	y_pred = dt_cls.predict(x_test)
	y_true = y_test
	a = accuracy_score(y_test, dt_cls.predict(x_test))
	p = precision_score(y_true, y_pred)
	r = recall_score(y_true, y_pred)
	return a, p, r

def random_forest_train_test(x_train, x_test, y_train, y_test):
	rf_cls = RandomForestClassifier() #랜덤포레스트 모듈
	rf_cls.fit(x_train, y_train) #랜덤포레스트 학습
    
	y_pred = rf_cls.predict(x_test)
	y_true = y_test
	a = accuracy_score(rf_cls.predict(x_test), y_test)
	p = precision_score(y_true, y_pred)
	r = recall_score(y_true, y_pred)
	return a, p, r

def svm_train_test(x_train, x_test, y_train, y_test):
	svm_pipe.fit(x_train, y_train)
    
	y_pred = svm_pipe.predict(x_test)
	y_true = y_test
	a = accuracy_score(y_test, svm_pipe.predict(x_test))
	p = precision_score(y_true, y_pred)
	r = recall_score(y_true, y_pred)
	return a, p, r

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)


if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)
    
    
	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
