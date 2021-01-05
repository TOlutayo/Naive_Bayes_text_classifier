# ECSE Mini-project 2

The accompanying notebook contains the custom built Naive bayes class and our best performing model, a Logistic Regression classifier along with their K-cross validation performance. This document gives an explanation of the functions and methods used and how to succesfully implement the notebook.
## Usage
Mount your google drive to gain access to the training data
```python
from google.colab import drive
drive.mount('/content/gdrive')
```
```python
pd_train_data = pd.read_csv('/content/gdrive/My Drive/ECSE 551 Project 2/train.csv')
pd_test_data = pd.read_csv('/content/gdrive/My Drive/ECSE 551 Project 2/test.csv')
```

Execute the data pre-processing section. There should not be any errors but if there are, ensure that you have read the training data from the correct directory.

## Naive Bayes Class
The Naive Bayes class consists of four methods. 
The fit method given by:
```python
#create Naivebayes class
def fit(self,X,y):
 grouped = []
 for i in np.unique(y):
     temp = []
     for j,k in zip(X,y):
        if k==i:
            temp.append(j)
     grouped.append(temp)
                
 self.prior_prob = np.array([np.log(len(i)/len(y)) for i in grouped])
            
 word_occurs = np.array([np.array(i).sum(axis=0) for i in grouped])+1
        
 no_of_doc = np.array([[np.array(i).shape[0] for i in grouped]])+2
                

 self.cond_prob = word_occurs/no_of_doc.T
          
```

This function calculates the conditional probabilities of the features conditioned on a given class and stores them in array of 8 arrays. Each array is itself an array of the conditional probabilities of features for each class obtained from the input training data and labels. The inputs to this function are the training data and labels.

The next method of the class is the predict method which makes predictions based on processed new input data and outputs the predicted labels. It's given by
```python
 def predict(self, X):
      self.labs = []
      for i in X:
        result = []
        temp_yes = np.log(self.cond_prob)*i 
        temp_no = np.array(1-i)*np.log(1-self.cond_prob) 
        temp = temp_yes + temp_no
        for j in temp:
          result.append(np.array([j]).sum(axis=1))
        result = np.array(result).T+self.prior_prob
        self.labs.append(np.argmax(result))
```
It adds up all of the conditional probabilities for each class and selects the class with the highest probability. Note that log sums are being used in order to avoid underflow.

The next method of the class is the accu_eval method that computes the accuracy of the model based on the predicted values and the actual values. It is given by 
```python
def accu_eval(self,y):
      self.acc = 0
      for i in range(len(y)):
        if self.labs[i]==y[i]:
          self.acc+=1
      print("The accuracy over the test data is",self.acc/len(y)*100,"%")
      return self.acc/len(y)
```
The last method of the class is the write_to_csv function that outputs the predicted labels in a csv format. It's given by:
```python
def write_to_csv(self):
      #return csv file with labels
      self.labs = le.inverse_transform(self.labs)
      dt = {'subreddit':self.labs}
      df = pd.DataFrame(dt)
      df.to_csv('out.csv',columns=['subreddit'],index_label=['id'])
```
This method was used to obtain the csv output file for Kaggle submissions of the predicted labels for the test data.
## Testing the NB model
In order to test the accuracy of the NB model, a K-fold cross validation metric was used. The functions are built based on the Kfold library provided in SciKitLearn. 
```python
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
#this loop will run n_splits number of times
def K_Fold_Eval(n_splits, model, X_train, y_train):
  kf = KFold(n_splits, shuffle = True)
  accs = []

  for train_ind, val_ind in kf.split(X_train):
    kf_x_train, kf_x_val = X_train[train_ind], X_train[val_ind]
    kf_y_train, kf_y_val = y_train[train_ind], y_train[val_ind]

    fitted = model.fit(kf_x_train, kf_y_train)
    predictions = fitted.predict(kf_x_val)
    acc = accuracy_score(kf_y_val, predictions)
    print(acc)
    accs.append(acc)
  
  av_acc = sum(accs)/len(accs)

  return av_acc
  
def K_Fold_Eval_NB(n_splits, model, X_train, y_train):
  kf = KFold(n_splits = n_splits, shuffle = True)
  accs = []

  for train_ind, val_ind in kf.split(X_train):
    kf_x_train, kf_x_val = X_train[train_ind], X_train[val_ind]
    kf_y_train, kf_y_val = y_train[train_ind], y_train[val_ind]

    #fitted = model.fit(kf_x_train, kf_y_train)
    model.fit(kf_x_train, kf_y_train)
    model.predict(kf_x_val)
    acc = model.accu_eval(kf_y_val)
    accs.append(acc)
  
  av_acc = sum(accs)/len(accs)

  return av_acc
```
As you can see in the code snippett above, there are two K-cross validation functions. K_Fold_Eval_NB and K_Fold_Eval. The former is used to evaluate the accuracy of the custom built NB model while the latter is used to evaluate the accuracy of the inbuilt SciKitLearn model classes. 

We used a 10-fold cross validation to obtain the accuracy of the NB model. Results are discussed extensively in the report.
```python
k_fold_NB = K_Fold_Eval_NB(10, Naivebayes(), train_data, train_labels)
print('K-fold Accuracy for Naivebayes', k_fold_NB)
```

## Logistic Regression Model
This was the model that obtained the highest K-cross validation accuracy and also our highest score on Kaggle. The model was built using the in built logistic regression model provided by SciKitLearn. The model initialization and results obtained are discussed extensively in the report. The model is given as:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter = 500)
logreg = model.fit(X_train, y_train)

predictions = logreg.predict(X_test)
acc = accuracy_score(y_test, predictions)
print('Logistic Regression Accuracy:', acc)

test_predictions = logreg.predict(test_data)

write_to_csv_sklearn(test_predictions, 'logreg')
```
And the K-cross Validation of the LR model is given as:
```python
from sklearn.linear_model import LogisticRegression
av_acc_log = K_Fold_Eval(10, LogisticRegression(max_iter = 500), train_data, train_labels)
print('av acc logreg', av_acc_log)
```

## Running the Model on Test Data
In order to run the model on the test data, read the test data from the correct directory. In our case
```python
test_data = pd.read_csv('/content/gdrive/My Drive/ECSE 551 Project 2/test.csv')
test_data = test_data['body'].tolist()
test_data = vectorizer.transform(test_data)
test_data = test_data.toarray()
print(test_data.shape)

td = Naivebayes()
td.fit(train_data, train_labels)
td.predict(test_data)
```
and then output the results to a csv file 
```python
td.write_to_csv()
```
## Running the models
To run the logistic  regression model in the python notebook, you need to uncomment blocks 4 and blocks 6 and comment out block 5. To run the naive bayes model comment out blocks 4 and 6.
Please note that due to the random shuffle in K-fold cross validation, the results will not be perfectly replicable.

## Contact
If there are any issues with running the notebok please conatct toluwaleke.olutayo@mail.mcgill.ca and I will reply immediately.

## License
[MIT](https://choosealicense.com/licenses/mit/)