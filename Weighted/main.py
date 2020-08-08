import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt

MyFontSize=20
data = pd.read_csv("../data.csv", sep=",")
X = np.array(data[["s"]])



predict = "sk1"

y = np.array(data[[predict]])
#transformer1 = FunctionTransformer(np.abs, validate=False)
#transformer2 = FunctionTransformer(np.log, validate=False)
#y0 = transformer1.fit_transform(y_raw)
#y = transformer2.fit_transform(y0)

#x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2, random_state=0)

n=len(y)
ntest=int(20/100*n)
y_train = y[0:n-ntest]
y_test = y[n-ntest:n]
x_train = X[0:n-ntest,:]
x_test = X[n-ntest:n,:]
linear = linear_model.LinearRegression()
sample_weight = np.exp(y_train)
sample_weight = np.reshape(sample_weight,len(sample_weight))

#k_fold
n_splits = 10
ntrain=len(y_train)
increment=int(ntrain/n_splits)
allindex = [*range(0 , ntrain)]
accmean = 0
numerator = 0
accArray = []
for i in range(n_splits):
    print(i)
    foldindex = allindex.copy()
    del foldindex[i*increment : (i+1)*increment]
    y_train_fold = y_train[foldindex]
    x_train_fold = x_train[foldindex]
    sample_weight_fold = sample_weight[foldindex]
    model=linear.fit(x_train_fold, y_train_fold, sample_weight_fold)
    predictions = linear.predict(x_test)
    error = y_test - predictions
    sse = (np.transpose(error) @ error)[0,0]
    numerator = numerator + sse
    #model=linear.fit(x_train, y_train, sample_weight)
    acc = linear.score(x_test, y_test)
    accArray.append(acc)
    accmean = accmean + acc
    print(acc)
    #print('Coefficient: ', linear.coef_)
    #print('Intercept: ', linear.intercept_)



numerator = numerator/n_splits
print("numerator:")
print(numerator)

variance = np.var(y_test) #ddof=0
print("variance:")
print(variance)

denumerator = variance * len(y_test)
print("denumerator:")
print(denumerator)

R2 = 1 - numerator/denumerator
print("R^2:")
print(R2)

print("accArray")
print(accArray)

print("accArray.median")
print(np.median(accArray))

model=linear.fit(x_train, y_train, sample_weight)
print("mean val:")
accmean = accmean/n_splits
print(accmean)
predictions = linear.predict(X)

iX = np.arange(1,len(X)+1)

plt.figure()
plt.scatter(iX, -np.exp(y))
plt.plot(iX, -np.exp(predictions), "k--", label="Fit")
plt.xlabel('Index of training', fontsize=MyFontSize)
plt.ylabel('ss\u03C3', fontsize=MyFontSize)
plt.xticks(fontsize=MyFontSize)
plt.yticks(fontsize=MyFontSize)
#plt.title("Exponential Fit")
#plt.show()
plt.savefig('sss.eps', format='eps')







predict = "sk2"

y = np.array(data[[predict]])
#transformer1 = FunctionTransformer(np.abs, validate=False)
#transformer2 = FunctionTransformer(np.log, validate=False)
#y0 = transformer1.fit_transform(y_raw)
#y = transformer2.fit_transform(y0)

#x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2, random_state=0)

n=len(y)
ntest=int(20/100*n)
y_train = y[0:n-ntest]
y_test = y[n-ntest:n]
x_train = X[0:n-ntest,:]
x_test = X[n-ntest:n,:]
linear = linear_model.LinearRegression()
sample_weight = np.exp(y_train)
sample_weight = np.reshape(sample_weight,len(sample_weight))

#k_fold
n_splits = 10
ntrain=len(y_train)
increment=int(ntrain/n_splits)
allindex = [*range(0 , ntrain)]
accmean = 0
numerator = 0
accArray = []
for i in range(n_splits):
    print(i)
    foldindex = allindex.copy()
    del foldindex[i*increment : (i+1)*increment]
    y_train_fold = y_train[foldindex]
    x_train_fold = x_train[foldindex]
    sample_weight_fold = sample_weight[foldindex]
    model=linear.fit(x_train_fold, y_train_fold, sample_weight_fold)
    predictions = linear.predict(x_test)
    error = y_test - predictions
    sse = (np.transpose(error) @ error)[0,0]
    numerator = numerator + sse
    #model=linear.fit(x_train, y_train, sample_weight)
    acc = linear.score(x_test, y_test)
    accArray.append(acc)
    accmean = accmean + acc
    print(acc)
    #print('Coefficient: ', linear.coef_)
    #print('Intercept: ', linear.intercept_)



numerator = numerator/n_splits
print("numerator:")
print(numerator)

variance = np.var(y_test) #ddof=0
print("variance:")
print(variance)

denumerator = variance * len(y_test)
print("denumerator:")
print(denumerator)

R2 = 1 - numerator/denumerator
print("R^2:")
print(R2)

print("accArray")
print(accArray)

print("accArray.median")
print(np.median(accArray))

model=linear.fit(x_train, y_train, sample_weight)
print("mean val:")
accmean = accmean/n_splits
print(accmean)
predictions = linear.predict(X)

iX = np.arange(1,len(X)+1)

plt.figure()
plt.scatter(iX, np.exp(y))
plt.plot(iX, np.exp(predictions), "k--", label="Fit")
plt.xlabel('Index of training', fontsize=MyFontSize)
plt.ylabel('sp\u03C3', fontsize=MyFontSize)
plt.xticks(fontsize=MyFontSize)
plt.yticks(fontsize=MyFontSize)
#plt.title("Exponential Fit")
#plt.show()
plt.savefig('sps.eps', format='eps')











predict = "sk3"

y = np.array(data[[predict]])
#transformer1 = FunctionTransformer(np.abs, validate=False)
#transformer2 = FunctionTransformer(np.log, validate=False)
#y0 = transformer1.fit_transform(y_raw)
#y = transformer2.fit_transform(y0)

#x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2, random_state=0)

n=len(y)
ntest=int(20/100*n)
y_train = y[0:n-ntest]
y_test = y[n-ntest:n]
x_train = X[0:n-ntest,:]
x_test = X[n-ntest:n,:]
linear = linear_model.LinearRegression()
sample_weight = np.exp(y_train)
sample_weight = np.reshape(sample_weight,len(sample_weight))

#k_fold
n_splits = 10
ntrain=len(y_train)
increment=int(ntrain/n_splits)
allindex = [*range(0 , ntrain)]
accmean = 0
numerator = 0
accArray = []
for i in range(n_splits):
    print(i)
    foldindex = allindex.copy()
    del foldindex[i*increment : (i+1)*increment]
    y_train_fold = y_train[foldindex]
    x_train_fold = x_train[foldindex]
    sample_weight_fold = sample_weight[foldindex]
    model=linear.fit(x_train_fold, y_train_fold, sample_weight_fold)
    predictions = linear.predict(x_test)
    error = y_test - predictions
    sse = (np.transpose(error) @ error)[0,0]
    numerator = numerator + sse
    #model=linear.fit(x_train, y_train, sample_weight)
    acc = linear.score(x_test, y_test)
    accArray.append(acc)
    accmean = accmean + acc
    print(acc)
    #print('Coefficient: ', linear.coef_)
    #print('Intercept: ', linear.intercept_)



numerator = numerator/n_splits
print("numerator:")
print(numerator)

variance = np.var(y_test) #ddof=0
print("variance:")
print(variance)

denumerator = variance * len(y_test)
print("denumerator:")
print(denumerator)

R2 = 1 - numerator/denumerator
print("R^2:")
print(R2)

print("accArray")
print(accArray)

print("accArray.median")
print(np.median(accArray))

model=linear.fit(x_train, y_train, sample_weight)
print("mean val:")
accmean = accmean/n_splits
print(accmean)
predictions = linear.predict(X)

iX = np.arange(1,len(X)+1)

plt.figure()
plt.scatter(iX, np.exp(y))
plt.plot(iX, np.exp(predictions), "k--", label="Fit")
plt.xlabel('Index of training', fontsize=MyFontSize)
plt.ylabel('pp\u03C3', fontsize=MyFontSize)
plt.xticks(fontsize=MyFontSize)
plt.yticks(fontsize=MyFontSize)
#plt.title("Exponential Fit")
#plt.show()
plt.savefig('pps.eps', format='eps')















predict = "sk4"

y = np.array(data[[predict]])
#transformer1 = FunctionTransformer(np.abs, validate=False)
#transformer2 = FunctionTransformer(np.log, validate=False)
#y0 = transformer1.fit_transform(y_raw)
#y = transformer2.fit_transform(y0)

#x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2, random_state=0)

n=len(y)
ntest=int(20/100*n)
y_train = y[0:n-ntest]
y_test = y[n-ntest:n]
x_train = X[0:n-ntest,:]
x_test = X[n-ntest:n,:]
linear = linear_model.LinearRegression()
sample_weight = np.exp(y_train)
sample_weight = np.reshape(sample_weight,len(sample_weight))

#k_fold
n_splits = 10
ntrain=len(y_train)
increment=int(ntrain/n_splits)
allindex = [*range(0 , ntrain)]
accmean = 0
numerator = 0
accArray = []
for i in range(n_splits):
    print(i)
    foldindex = allindex.copy()
    del foldindex[i*increment : (i+1)*increment]
    y_train_fold = y_train[foldindex]
    x_train_fold = x_train[foldindex]
    sample_weight_fold = sample_weight[foldindex]
    model=linear.fit(x_train_fold, y_train_fold, sample_weight_fold)
    predictions = linear.predict(x_test)
    error = y_test - predictions
    sse = (np.transpose(error) @ error)[0,0]
    numerator = numerator + sse
    #model=linear.fit(x_train, y_train, sample_weight)
    acc = linear.score(x_test, y_test)
    accArray.append(acc)
    accmean = accmean + acc
    print(acc)
    #print('Coefficient: ', linear.coef_)
    #print('Intercept: ', linear.intercept_)



numerator = numerator/n_splits
print("numerator:")
print(numerator)

variance = np.var(y_test) #ddof=0
print("variance:")
print(variance)

denumerator = variance * len(y_test)
print("denumerator:")
print(denumerator)

R2 = 1 - numerator/denumerator
print("R^2:")
print(R2)

print("accArray")
print(accArray)

print("accArray.median")
print(np.median(accArray))

model=linear.fit(x_train, y_train, sample_weight)
print("mean val:")
accmean = accmean/n_splits
print(accmean)
predictions = linear.predict(X)

iX = np.arange(1,len(X)+1)

plt.figure()
plt.scatter(iX, -np.exp(y))
plt.plot(iX, -np.exp(predictions), "k--", label="Fit")
plt.xlabel('Index of training', fontsize=MyFontSize)
plt.ylabel('pp\u03C0', fontsize=MyFontSize)
plt.xticks(fontsize=MyFontSize)
plt.yticks(fontsize=MyFontSize)
#plt.title("Exponential Fit")
#plt.show()
plt.savefig('ppp.eps', format='eps')




