import pandas as pd
import numpy as np

#visualizations
import matplotlib.pyplot as plt

#Evaluating models
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

#logistic regression
from sklearn.linear_model import LogisticRegression

#Esemble boosting methods
from sklearn.ensemble import  GradientBoostingClassifier

#XGboost

import seaborn as sns
sns.set_style('darkgrid')

#Random Forest
from sklearn.ensemble import RandomForestClassifier

#grid search/ cross validation
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import sqlite3

class result(object):
    """preds is a dictionary whose keys are 'train_preds' and 'test_preds' and the corresponding value is 
        the prediction for X_train and X_test
        
        params is a dictionary of parameters used in the model
    
    """
    def __init__(self, params=None, preds={'train_preds':None, 'test_preds':None}):
        self.params = params#parameter used, dictionary
        self.preds = preds#[train_preds, test_preds]
        self.data = None
        self.note = None
        self.metrics = None # to save the output of save_metrics
    def print_metrics(self):
        '''Print evaluation scores for the train and the test sets'''
        print('Train set\n')
        print("Precision Score: {}".format(precision_score(y_train_resampled, self.preds['train_preds'])))
        print("Recall Score: {}".format(recall_score(y_train_resampled, self.preds['train_preds'])))
        print("Accuracy Score: {}".format(accuracy_score(y_train_resampled, self.preds['train_preds'])))
        print("F1 Score: {}".format(f1_score(y_train_resampled, self.preds['train_preds'])))
        print("Roc_Auc_Score: {}".format(roc_auc_score(y_train_resampled, self.preds['train_preds'])))
        print('\n')
        
        print('Test set\n')
        print("Precision Score: {}".format(precision_score(Y_test, self.preds['test_preds'])))
        print("Recall Score: {}".format(recall_score(Y_test, self.preds['test_preds'])))
        print("Accuracy Score: {}".format(accuracy_score(Y_test, self.preds['test_preds'])))
        print("F1 Score: {}".format(f1_score(Y_test, self.preds['test_preds'])))
        print("Roc_Auc_Score: {}".format(roc_auc_score(Y_test, self.preds['test_preds'])))
        print('\n')
        
    def save_metrics(self):
        ''' Save the evaluation scores as a dictionary. '''
        scores ={}
        scores['Train set']={
            "Precision Score":precision_score(y_train_resampled, self.preds['train_preds']),
            "Recall Score":recall_score(y_train_resampled, self.preds['train_preds']),
            "Accuracy Score":accuracy_score(y_train_resampled, self.preds['train_preds']),
            "F1 Score":f1_score(y_train_resampled, self.preds['train_preds']),
            "Roc_Auc_Score":roc_auc_score(y_train_resampled, self.preds['train_preds']),
        }

        scores['Test set']={
            "Precision Score":precision_score(Y_test, self.preds['test_preds']),
            "Recall Score":recall_score(Y_test, self.preds['test_preds']),
            "Accuracy Score":accuracy_score(Y_test, self.preds['test_preds']),
            "F1 Score":f1_score(Y_test, self.preds['test_preds']),
            "Roc_Auc_Score":roc_auc_score(Y_test, self.preds['test_preds']),
            
        }
        return scores


def Saving_results(result_obj,#result objec 
                   md_obj,#model object such as LogisticRegression
                   result_name,#The name of the model such as logistic regression to save it in train_scores/test_scores df
                  FtImp = True
                  ):
    
    md_obj.fit(scaled_X_train,y_train_resampled)#fit model
    train_preds = md_obj.predict(scaled_X_train)#y_train_prediction
    test_preds = md_obj.predict(scaled_X_test)#y_test_prediction
    
    result_obj.preds = {'train_preds':train_preds, 'test_preds':test_preds}#predictions grouped as a dictionary saved 
    result_obj.params = md_obj.__dict__#hyperparameters saved 
    result_obj.metrics = result_obj.save_metrics()#evaluation scores saved as a dictionary 
    #result_obj.print_metrics() #print evaluation scores 
    Results[result_name] = result_obj#the result obj saved in Results
    print('{} is saved in Results table.'.format(result_name))
    
    
    #evaluation scores saved in the dataframes
    train_scores.loc[result_name] = result_obj.metrics['Train set'] 
    test_scores.loc[result_name] = result_obj.metrics['Test set']
    
    #updating the dataframes of evalution scores
    Results['train_scores'] = train_scores
    Results['test_scores'] = test_scores
    print('train_scores dataframe is updated.\n')
    print(train_scores)
    print('\n')
    print('test_scores dataframe is updated.\n')
    print(test_scores)
    if FtImp:
        Results[result_name] = pd.Series(md_obj.feature_importances_, X_train.columns).sort_values(ascending=False)
    else :
            LogReg_FtImp = pd.DataFrame()
            LogReg_FtImp['feature']=np.array(X_train.columns)
            LogReg_FtImp['importance']= result_obj.params['coef_'][0]
            LogReg_FtImp['importance_abs'] = abs(LogReg_FtImp['importance'])#absolute values of coefficients to rank features' influences
            LogReg_FtImp.sort_values(by=['importance_abs'],ascending=False, inplace=True)
            LogReg_FtImp.reset_index(inplace=True)
            LogReg_FtImp.drop(['index'],axis=1, inplace=True)
            Results[result_name] = LogReg_FtImp
    return md_obj

def hyperparameter_tuning():
    '''This function returns the best prarmeter for each function'''

    # Grid search is used to find the optimal hyperparameters of a model which results in the most 'accurate' prediction.
    logreg = GridSearchCV(LogisticRegression(fit_intercept=False, solver='sag'),
                          param_grid = {'C': [0.001, 0.01, 0.1, 1]},
                          cv = 5)
    logreg.fit(scaled_X_train,y_train_resampled)
    
    RNDforest = GridSearchCV(RandomForestClassifier(),
                          param_grid= {'n_estimators':[50,100], 'criterion':['gini','entropy'], 'max_depth':[1, 3, 5, 7]},
                         cv = 3)
    RNDforest.fit(scaled_X_train,y_train_resampled)
    
    GR_Boost = GridSearchCV(GradientBoostingClassifier(),
                            param_grid= {'max_depth':[5, 7, 9], 'n_estimators':[80]},
                            cv = 3)
    GR_Boost.fit(scaled_X_train,y_train_resampled)
    
    return logreg.best_params_ , RNDforest.best_params_, GR_Boost.best_params_

def model_train(logreg_params, RNDforest_params, GR_Boost_params):
    ''' The function to train the model'''
    logreg = LogisticRegression(fit_intercept=False, C=logreg_params['C'], solver='sag')
    
    RNDforest = RandomForestClassifier(n_estimators = RNDforest_params['n_estimators'], criterion = RNDforest_params['criterion'], max_depth = RNDforest_params['max_depth'])
    
    GR_Boost = GradientBoostingClassifier(max_depth = GR_Boost_params['max_depth'], n_estimators = GR_Boost_params['n_estimators'])                        
    
    Log_reg = result()
    rndforest = result()
    gradBoost = result()

    logreg = Saving_results(Log_reg,logreg,'Logistic Regression', FtImp = False)
    RNDforest = Saving_results(rndforest,RNDforest,'Random Forest')
    GR_Boost = Saving_results(gradBoost,GR_Boost,'Gradient Boosting')
    return logreg, RNDforest, GR_Boost

if __name__ == "__main__":

    con = sqlite3.connect('final_dataset.db') # Opening a connections t othe SQLite database file

    train = pd.read_sql('select * from train', con) # read the train data from the train table.
    test = pd.read_sql('select * from test', con) # read the test data from the train table.
    
    x = train[[i for i in list(train.columns) if i not in ['campaign_id', 'id', 'customer_id', 'coupon_id','redemption_status']]]
    y = train['redemption_status']
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=0)   
    test = test[[i for i in list(test.columns) if i not in ['campaign_id', 'id', 'customer_id', 'coupon_id']]]
    smote = SMOTE()
    # Resampling the data since the data is imbalance 
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, Y_train) 

    # Normalizing the data
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train_resampled)
    scaled_X_test = scaler.transform(X_test)
    scaled_X_train = pd.DataFrame(scaled_X_train, columns = X_train.columns)
    scaled_X_test = pd.DataFrame(scaled_X_test, columns = X_train.columns)
    scaled_test = scaler.transform(test)

    Results ={}
    #Dataframes to save evaluating scores of different models
    train_scores =pd.DataFrame(columns = ['Precision Score', 'Recall Score', 'Accuracy Score', 'F1 Score', 'Roc_Auc_Score'])
    test_scores =pd.DataFrame(columns = ['Precision Score', 'Recall Score', 'Accuracy Score', 'F1 Score', 'Roc_Auc_Score'])

    logreg_params, RNDforest_params, GR_Boost_params = hyperparameter_tuning() 
    lr, rf, gbc = model_train(logreg_params, RNDforest_params, GR_Boost_params)

    train_scores.to_csv('train_scores.csv')
    test_scores.to_csv('test_scores.csv')

    test_pred = gbc.predict(scaled_test)
    print('\nPrediction on test data using Gradient Boosting')
    print("Redemption count : ", list(test_pred).count(1))
    print("Non-Redemption count : ", list(test_pred).count(0))
    # Saving the important features of each model

    print('\n********IMPORTANT FEATURES********')
    LogReg_FtImp = Results['Logistic Regression']
    # printing the top 10 important features.
    print('\nLogistic Regression')
    print(LogReg_FtImp[:10])
    rndforest_feat_imp  = pd.DataFrame(Results['Random Forest']).reset_index()
    rndforest_feat_imp .columns =['Feature', 'Score']
    print('\nRandom Forest')
    print(rndforest_feat_imp[:10])
    gradBoost_feat_imp = pd.DataFrame(Results['Gradient Boosting']).reset_index()
    gradBoost_feat_imp.columns =['Feature', 'Score']
    print('\nGradient Boosting')
    print(gradBoost_feat_imp[:10])
    

