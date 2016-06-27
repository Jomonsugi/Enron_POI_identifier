#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
from sklearn.naive_bayes import GaussianNB

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 
                 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
print ''
print 'Feature list:'
print features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### removing outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

### Creating new features
### Storing to my_dataset for easy export below
my_dataset = data_dict

for i in my_dataset.values():
    i['salary_eso_ratio'] = 0
    if i['exercised_stock_options'] != 'NaN' and i['salary'] != 'NaN':
       i['salary_eso_ratio'] = float(i['exercised_stock_options'])/float(i['salary'])
features_list.append('salary_eso_ratio')

for i in my_dataset.values():
    i['salary_bonus_ratio'] = 0
    if i['bonus'] != 'NaN' and i['salary'] != 'NaN':
       i['salary_bonus_ratio'] = float(i['bonus'])/float(i['salary'])
features_list.append('salary_bonus_ratio')

for i in my_dataset.values():
    i['salary_tsv_ratio'] = 0
    if i['total_stock_value'] != 'NaN' and i['salary'] != 'NaN':
       i['salary_tsv_ratio'] = float(i['total_stock_value'])/float(i['salary'])
features_list.append('salary_tsv_ratio')

for i in my_dataset.values():
    i['total_stock_vs_exercised_stock_ratio'] = 0
    if i['total_stock_value'] != 'NaN' and i['exercised_stock_options'] != 'NaN':
       i['total_stock_vs_exercised_stock_ratio'] = float(i['exercised_stock_options'])/float(i['total_stock_value'])
features_list.append('total_stock_vs_exercised_stock_ratio')


### these ratio features are hurting both my accuracy and recall,
### even when not selected by kbest but not sure why...so I am commenting them out
'''
for person in my_dataset.values():
    person['to_poi_message_ratio'] = 0
    person['from_poi_message_ratio'] = 0
    if float(person['from_messages']) > 0:
        person['to_poi_message_ratio'] = float(person['from_this_person_to_poi'])/float(person['from_messages'])
    if float(person['to_messages']) > 0:
        person['from_poi_message_ratio'] = float(person['from_poi_to_this_person'])/float(person['to_messages'])
features_list.extend(['to_poi_message_ratio', 'from_poi_message_ratio'])
'''

print ''
print 'Updated feature list:'
print features_list

### Extract features and labels from dataset for local testing
### These steps prepare the data for 'StratifiedShuffleSplit' and all methods of validation
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
#scaler = preprocessing.MinMaxScaler()
kbest = SelectKBest(f_classif)
clf_DTC = DecisionTreeClassifier(random_state = 42)
NB = GaussianNB()


#t0 = time()
#pipeline =  Pipeline(steps=[("kbest", kbest), ("DTC", clf_DTC)])

#decided on this pipeline:
pipeline =  Pipeline(steps=[("kbest", kbest), ("NB", NB)])

#pipeline =  Pipeline(steps=[('scaling', scaler),("kbest", kbest), ("DTC", clf_DTC)])
#print "pipeline time:", round(time()-t0, 3), "s"

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
parameters = {'kbest__k': [1,2,3,4,5,6,7,8,9,10]}
cv = StratifiedShuffleSplit(labels, 10, random_state = 42)

#t0 = time()
gs = GridSearchCV(pipeline, parameters, cv = cv)
#print "GridSearchCV time:", round(time()-t0, 3), "s"

t0 = time()
gs.fit(features, labels)
print "fit time:", round(time()-t0, 3), "s"

#t0 = time()
clf = gs.best_estimator_
#print "estimation time:", round(time()-t0, 3), "s"

X_new = gs.best_estimator_.named_steps['kbest']
# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in X_new.scores_ ]
# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  X_new.pvalues_ ]
# Get SelectKBest feature names, whose indices are stored in 'X_new.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in X_new.get_support(indices=True)]

# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

# Print
print ' '
print 'Selected Features, Scores, P-Values'
print features_selected_tuple


features_selected_bool = gs.best_estimator_.named_steps['kbest'].get_support()
features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]
print ''
print 'Features selected by SelectKBest:'
print features_selected_list

'''
features_selected=[features_list[i+1] for i in clf.named_steps['kbest'].get_support(indices=True)]
importances = clf.named_steps['DTC'].feature_importances_
indices = np.argsort(importances)[::-1]
print ''
print 'Feature Ranking: '
for i in range(len(features_selected)):
    print "feature no. {}: {} ({})".format(i+1,features_selected[indices[i]],importances[indices[i]])
'''
    
# import the test_classifier from tester.py
from tester import test_classifier
print ' '
# use test_classifier to evaluate the model
# selected by GridSearchCV
print "Tester Classification report:" 
test_classifier(clf, my_dataset, features_list)
print ' '

    
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
