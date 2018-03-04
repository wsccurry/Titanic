import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import classification_report

#从网上读取数据
titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
X=titanic[['pclass','age','sex']]  #取pclass,age,sex这三个特征量
y=titanic['survived']
X['age'].fillna(X['age'].median(),inplace=True) #补全age,值为age的中位数

#采用交叉验证,按3:1分割训练集和数据集,随机种子设为30
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=30)

vec=DictVectorizer(sparse=False)
X_train.info()
#将Dataframe类型先转换为dict类型,再转换为ndarray,其中类别型的单独剥离出来自成一个特征，此时由原来三个特征变为六个特征
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.fit_transform(X_test.to_dict(orient='record'))

#使用单一决策树模型的结果
dtc=tree.DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_pred=dtc.predict(X_test)
print('使用单一决策树模型的准确率:%s',dtc.score(X_test,y_test))
print(classification_report(dtc_y_pred,y_test))

#使用随机森林模型的结果
rfc=ensemble.RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred=rfc.predict(X_test)
print('使用随机森林模型的准确率:%s',rfc.score(X_test,y_test))
print(classification_report(rfc_y_pred,y_test))

#使用SVM模型的结果
from sklearn import svm
svc=svm.SVC(gamma=0.001,C=100)
svc.fit(X_train,y_train)
svc_y_pred=svc.predict(X_test)
print('使用SVM模型的准确率:%s',svc.score(X_test,y_test))
print(classification_report(svc_y_pred,y_test))

#使用逻辑回归模型的结果
from sklearn import linear_model
lr=linear_model.LogisticRegression()
lr.fit(X_train,y_train)
lr_y_pred=lr.predict(X_test)
print('使用逻辑回归模型的准确率:%s',lr.score(X_test,y_test))
print(classification_report(lr_y_pred,y_test))