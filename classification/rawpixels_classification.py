import os
import time
import math
from pathlib import Path
from pprint import pprint
import time
import sys
import argparse
cwd = os.getcwd()
from utils.functions import *
from utils.plotting_functions import *
import sklearn
import pandas as pd
import numpy as np
from sklearn import preprocessing
pd.options.mode.chained_assignment = None
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mannwhitneyu
import logging
import matplotlib.pyplot as plt 

from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Temporarily filter out the UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)




def rawpixels_classification(classifier_name, classification_type):
    if classification_type == "binary":
            malicious_families = ["Adposhel", "Agent", "Allaple", "Amonetize", "Androm", "Autorun", "BrowseFox","Dinwod", "Elex", "Expiro", "Fasong", "HackKMS", "Hlux", "Injector", "InstallCore", "MultiPlug", "Neoreklami", "Neshta", "Regrun", "Sality", "Snarasite", "Stantinko", "VBA", "VBKrypt", "Vilsel"] # list of malicious familie
    elif classification_type == "multiclass":
            malicious_families = ["Any"]

    def get_train_function(classification_type):
        if classification_type == "binary":
            return train_test_binary
        elif classification_type == "multiclass":
            return train_test_multiclass
        else:
            raise ValueError("Invalid train_type")

    train_test = get_train_function(classification_type)

    def get_active_function(classification_type):
        if classification_type == "binary":
            return active_learning_binary_raw
        elif classification_type == "multiclass":
            return active_learning_multiclass_raw
        else:
            raise ValueError("Invalid train_type")

    active_learning = get_active_function(classification_type)

    def get_assign_function(classification_type):
        if classification_type == "binary":
            return assign_confidence_binary
        elif classification_type == "multiclass":
            return assign_confidence_multiclass
        else:
            raise ValueError("Invalid train_type")

    assign_confidence = get_assign_function(classification_type)
    
    F_holdout = 0.20   # FUTURE SET. Percentage of samples (taken from each individual set, malicious AND benign) to put in F.I

    n, k = 2,3 # number of runs of CEF-SSL
    Cm_list = [1, 2, 5]   # COST OF LABELLING. 1=malicious and benign samples have same cost; 2=malicious samples have twice the cost of benign samples; 5=malicious samples have five times the cose of benign samples



    Lm_list = [50, 100] # LABELLING BUDGET (in malicious samples). The script will choose the provided number of malicious samples, and then pick the benign samples according to the Cm value
    active_budget = [0.5,0.5] # either 'regular' or a custom integer
    activeLearning_trials = 2 # how many times each "active learning" model is retrained on new randomly chosen samples (falling in the confidence thresholds)
    al_confidences = ['high', 'mid', 'low']
    high_confidence_window = 0.2 # samples whose confidence is ABOVE (100-high_confidence_window/2) are considered to be of "high confidence"
    low_confidence_window = 0.2 # samples whose confidence is BELOW (low_confidence_window/2) are considered to be of "high confidence"
    # choose the basic classification algorithm here. 
    ## IMPORTANT: to make it work with the implemented SsL techniques, the classifier must support the "predict_proba" method!
   

    ###### LOAD DATASET

    #features = pd.Index(['feature1', 'feature2', '...']) # list of features
    label_name = 'Label' # name of the "label" column
    malware_dataset = "./data/malware_dataset.csv"
    benign_dataset = "./data/benign_dataset.csv" #path of dataset file
    if classification_type == "binary":
            malicious_families = ["Adposhel", "Agent", "Allaple", "Amonetize", "Androm", "Autorun", "BrowseFox","Dinwod", "Elex", "Expiro", "Fasong", "HackKMS", "Hlux", "Injector", "InstallCore", "MultiPlug", "Neoreklami", "Neshta", "Regrun", "Sality", "Snarasite", "Stantinko", "VBA", "VBKrypt", "Vilsel"] # list of malicious familie
    elif classification_type == "multiclass":
            malicious_families = ["Any"]
    #df = pd.read_csv(malicious_dataset, index_col=0) # load dataset into a dataframe. NOTE: the "load_dataset" method is just an abstraction and must be provided by the user
    #malicious_families = ["Adposhel", "Agent", "Allaple", "Amonetize", "Androm", "Autorun", "BrowseFox","Dinwod", "Elex", "Expiro", "Fasong", "HackKMS", "Hlux", "Injector", "InstallCore", "MultiPlug", "Neoreklami", "Neshta", "Regrun", "Sality", "Snarasite", "Stantinko", "VBA", "VBKrypt", "Vilsel"] # list of malicious familie
    labels = ["Adposhel", "Agent", "Allaple", "Amonetize", "Androm", "Autorun", "BrowseFox","Dinwod", "Elex", "Expiro", "Fasong", "HackKMS", "Hlux", "Injector", "InstallCore", "MultiPlug", "Neoreklami", "Neshta", "Regrun", "Sality", "Snarasite", "Stantinko", "VBA", "VBKrypt", "Vilsel", "Other"]
    
    benign_df = convert(pd.read_csv(benign_dataset,sep=";"))
    
    malicious_df2 = convert(pd.read_csv(malware_dataset, sep=";" ))
    

    #for fam in labels:
     #   filtered_df = malicious_df2[malicious_df2['Label'] == fam]
      #  print(filtered_df)
    algorithms = ['rf','gb','svm','knn']

    if classifier_name == 'all':
        selected_algorithms = algorithms
    else:
        selected_algorithms = [classifier_name]
   
    
    class_label_mapping = {
    "Other": 0,
    "Adposhel": 1,
    "Agent": 2,
    "Allaple": 3,
    "Amonetize": 4,
    "Androm": 5,
    "Autorun": 6,
    "BrowseFox": 7,
    "Dinwod": 8,
    "Elex": 9,
    "Expiro": 10,
    "Fasong": 11,
    "HackKMS": 12,
    "Hlux": 13,
    "Injector": 14,
    "InstallCore": 15,
    "MultiPlug": 16,
    "Neoreklami": 17,
    "Neshta": 18,
    "Regrun": 19,
    "Sality": 20,
    "Snarasite": 21,
    "Stantinko": 22,
    "VBA": 23,
    "VBKrypt": 24,
    "Vilsel": 25,
}
    benign_df['Label'] = benign_df['Label'].map(class_label_mapping)
    malicious_df2['Label'] = malicious_df2['Label'].map(class_label_mapping)
    #print(benign_df)
    algorithm_results = {algorithm: {'UpperBound': {'F1': [], 'Time': []},
                                 'LowerBound': {'F1': [], 'Time': []},
                                 'Base': {'F1': [], 'Time': []},
                                 'High': {'F1': [], 'Time': []},
                                 'HighHigh': {'F1': [], 'Time': []},
                                 'ActiveLow': {'F1': [], 'Time': []},
                                 'ActiveMid': {'F1': [], 'Time': []},
                                 'ActiveHigh': {'F1': [], 'Time': []},
                                 'PseudoLow': {'F1': [], 'Time': []},
                                 'PseudoMid': {'F1': [], 'Time': []},
                                 'PseudoHigh': {'F1': [], 'Time': []}} for algorithm in selected_algorithms}


    #print("C:", arr.ndim)########## BEGIN EVALUATION ##########
    executions_total = n * k * len(Cm_list) * len(Lm_list) * len(malicious_families)
    print("\nCEF-SsL performs ({}*{}) runs for each combination above.\nBecause there are {} malicious families, the total amount of executions of CEF-SsL are: {}".format(n, k, len(malicious_families), executions_total))
    # the script is executed for each malicious family (in the case a dataset contains multiple "attacks"; otherwise, it will be run just once)
    for index, family in enumerate(malicious_families):
        if classification_type == 'binary':
            numerical_value = class_label_mapping.get(family)
            
            malicious_df = malicious_df2[malicious_df2["Label"] == numerical_value]
        else:
            malicious_df = malicious_df2.copy()
        begin = time.time()
        print("Benign:{}\tMalicious:{}\t({})".format(len(benign_df), len(malicious_df), family))
        for Cm in Cm_list:
           
            for Lm in Lm_list:
                
                for reiterate in range(k):
                    # We create a new F
                    print("reiterate: {} on {}".format(reiterate, k))
                    malicious_df['holdout'] = (np.random.uniform(0,1,len(malicious_df)) <= F_holdout)
                    benign_df['holdout'] = (np.random.uniform(0,1,len(benign_df)) <= F_holdout)
                    malicious_F_df, malicious_UL_df = malicious_df[malicious_df['holdout']==True], malicious_df[malicious_df['holdout']==False]
                    benign_F_df, benign_UL_df = benign_df[benign_df['holdout']==True], benign_df[benign_df['holdout']==False]
                    
                    F_df = pd.concat([malicious_F_df, benign_F_df])
                    
                    F_labels = F_df[label_name]
                    print("Size of F (test data): {} ben {} mal ({} tot)".format(len(benign_F_df), len(malicious_F_df), len(F_df)))

                    UL_df = pd.concat([malicious_UL_df, benign_UL_df])
                    
                    UL_labels = np.array(UL_df["Label"])  
                    
                    for run in range(n):
                        print("\trun {} of {} (for reiterate: {} on {})".format(run, n, reiterate, k))
                        Lb = int(Lm * Cm)
                        Lcost = Lb + Lm
                        if classification_type == 'binary':
                            malicious_L_df = malicious_UL_df.sample(n=Lm)
                        else:
                            sampled_dfs = []
                            for label in range(1, 26):
                              label_df = malicious_UL_df[malicious_UL_df['Label'] == label]
                              if len(label_df) >= Lm:
                                 sampled_df = label_df.sample(n=Lm)
                                 sampled_dfs.append(sampled_df)
                            malicious_L_df = pd.concat(sampled_dfs)
                        
                        
                        benign_L_df = benign_UL_df.sample(n=Lb)
                       
                        malicious_U_df = malicious_UL_df.drop(malicious_L_df.index)
                        benign_U_df = benign_UL_df.drop(benign_L_df.index)
                    
                       


                        #for active learning

                        if classification_type == 'binary':
                            malicious_support_L_df = malicious_L_df.sample(n=int(Lm * active_budget[0]))
                        else:
                            sampled_dfs = []
                            for label in range(1, 26):
                                label_df = malicious_L_df[malicious_L_df['Label'] == label]
                                if len(label_df) >= Lm:
                                    sampled_df = label_df.sample(n=int(Lm * active_budget[0]))
                                    sampled_dfs.append(sampled_df)
                            malicious_support_L_df = pd.concat(sampled_dfs)
                        benign_support_L_df = benign_L_df.sample(n=int(Lb * active_budget[1]))
                        support_L_df = pd.concat([malicious_support_L_df, benign_support_L_df])
                        
                        support_L_labels = np.array(support_L_df["Label"])
                        

                        # regenerating U
                        
                        malicious_U_df_active = malicious_UL_df.drop(malicious_support_L_df.index)
                        benign_U_df_active = benign_UL_df.drop(benign_support_L_df.index)
                        support_U_df = pd.concat([malicious_U_df_active, benign_U_df_active]) 

                    
        
                    
                        L_df = pd.concat([malicious_L_df, benign_L_df])
                        #print(L_df)
                       

                        L_labels = np.array(L_df[label_name])   
                        U_df = pd.concat([malicious_U_df, benign_U_df])

                        U_labels = np.array(U_df[label_name])  



                        X_UL = np.stack(UL_df["Image"].values)
                        
                        X_U = np.stack(U_df["Image"].values)
                        X_F = np.stack(F_df["Image"].values)
                        X_L = np.stack(L_df["Image"].values)
                        prova = UL_df["Image"].tolist()
                        nsamples, nx, ny = X_UL.shape
                        X_UL = X_UL.reshape((nsamples,nx*ny))
                    
                        nsamples, nx, ny = X_U.shape
                        X_U = X_U.reshape((nsamples,nx*ny))
                        nsamples, nx, ny = X_F.shape
                        X_F = X_F.reshape((nsamples,nx*ny))
    
                        nsamples, nx, ny = X_L.shape
                        X_L = X_L.reshape((nsamples,nx*ny))


                        for algorithm in selected_algorithms:
                            if algorithm == 'rf':
                                base_clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=2, min_samples_split=2, 
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, 
                                min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=-2, 
                                random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
                            elif algorithm == 'gb':
                                base_clf = HistGradientBoostingClassifier(max_depth=3, max_iter=80, learning_rate=0.01, random_state=42, min_samples_leaf=1, early_stopping=True)
                            elif algorithm == 'svm':
                                base_clf = svm.SVC(kernel='rbf', probability=True)
                            elif algorithm == 'knn':
                                base_clf = KNeighborsClassifier(n_neighbors=20, p=2)
                       

                        
                        
                            print("\t\t\tSL (UpperBound). Large L: {}b {}m\t"
                                .format(len(benign_L_df)+len(benign_U_df), len(malicious_L_df)+len(malicious_U_df), len(UL_df)))
                            SL_clf, SL_precision, SL_recall, SL_fscore, SL_trainTime, SL_testTime, SL_Fpredictions = train_test(train_df=X_UL, train_labels=UL_labels, 
                                                                                                test_df=X_F, test_labels=F_labels, messages=1, base_clf=base_clf)
                            print("F1: {:.5f} PRECISION: {:.5f}  RECALL: {:.5f} TrainTime: {:.3f}s".format(SL_fscore, SL_precision, SL_recall, SL_trainTime))
                            #print("F1:", SL_fscore)
                            algorithm_results[algorithm]['UpperBound']['F1'].append(SL_fscore)
                            algorithm_results[algorithm]['UpperBound']['Time'].append(SL_trainTime)
                            print("\t\t\tsl (LowerBound). L: {}b {}m\t"
                                .format(len(benign_L_df), len(malicious_L_df), len(L_df)), )
                            sl_clf, sl_precision, sl_recall, sl_fscore, sl_trainTime, sl_testTime, sl_Fpredictions = train_test(train_df=X_L, train_labels=L_labels, 
                                                                                                test_df=X_F, test_labels=F_labels, messages=1, base_clf=base_clf)
                            print(" F1: {:.5f} PRECISION: {:.5f}  RECALL: {:.5f}  TrainTime: {:.3f}s".format(sl_fscore, sl_precision, sl_recall, sl_trainTime))
                      
                            algorithm_results[algorithm]['LowerBound']['F1'].append(sl_fscore)
                            algorithm_results[algorithm]['LowerBound']['Time'].append(sl_trainTime)
                    
                            F_df["SL_predictions"] = SL_Fpredictions
                            F_df["sl_predictions"] = sl_Fpredictions
                      
                            #########################
                            ##### VANILLA Pseudo Labelling
                            ## PREPARE    
                            start_time = time.time()
                            
                            
                            sl_Upredictions = sl_clf.predict(X_U)
                        

                            sl_Uprobabilities =  sl_clf.predict_proba(X_U)
                            sl_predictTime = time.time() - start_time
                        
                            U_df['sl_prediction'] = sl_Upredictions
                            L_df['sl_prediction'] = L_labels # dummy column (the pseudo-labelling models will use this column as 'label')
                            if classification_type == 'binary':
                                    U_df['sl_probability'] = (np.hsplit(sl_Uprobabilities,2))[0]
                            else: 
                                    U_df['sl_probability'] = np.array(sl_Uprobabilities).tolist()
                                
                            
                           

                            # Assign confidence based on (predicted) probability and split dataframe
                           
                            U_df, U_high_df, U_mid_df, U_low_df = assign_confidence(U_df, high_confidence_window, low_confidence_window, 
                                                                                                probability_column_name = 'sl_probability', confidence_column_name = 'sl_confidence', debug=False, split=True)
                    

                            
                                

                            pseudoAll_L_df = pd.concat([U_df, L_df])
                            pseudoAll_L_labels = pseudoAll_L_df['sl_prediction']
                           
                            ps = np.stack(pseudoAll_L_df['Image'].values)
                          
                            
        
                            
                            nsamples, nx, ny = ps.shape
                            ps = ps.reshape((nsamples,nx*ny))
                            ## TRAIN and TEST
                            print("\t\t\tBaseline ssl (all pseudo labels). L: {}b {}m\t"
                                .format(len(benign_L_df), len(malicious_L_df)) )
                            ssl_clf, ssl_precision, ssl_recall, ssl_fscore, ssl_trainTime, ssl_testTime, ssl_Fpredictions = train_test(train_df=ps, train_labels=pseudoAll_L_labels, 
                                                                                                test_df=X_F, test_labels=F_labels, messages=1, base_clf=base_clf)
                            ssl_trainTime = ssl_trainTime + sl_predictTime + sl_trainTime
                            print("F1: {:.5f} PRECISION: {:.5f}  RECALL: {:.5f}  TrainTime: {:.3f}s".format(ssl_fscore, ssl_precision, ssl_recall, ssl_trainTime))
                            algorithm_results[algorithm]['Base']['F1'].append(ssl_fscore)
                            algorithm_results[algorithm]['Base']['Time'].append(ssl_trainTime)
                            F_df["ssl_predictions"] = ssl_Fpredictions


                          
                            
                            
                            pseudoHigh_L_df = pd.concat([U_high_df, L_df])
                           
                            psh = np.stack(pseudoHigh_L_df['Image'].values) 
                            nsamples, nx, ny = psh.shape
                            psh = psh.reshape((nsamples,nx*ny))

                            pseudoHigh_L_labels = pseudoHigh_L_df['sl_prediction']
                            print("\t\t\tpssl (high confidence pseudo labels). L: {}b {}m\t".
                                format(len(benign_L_df), len(malicious_L_df)))
                            pssl_clf, pssl_precision, pssl_recall, pssl_fscore, pssl_trainTime, pssl_testTime, pssl_Fpredictions = train_test(train_df=psh, train_labels=pseudoHigh_L_labels, 
                                                                                                test_df=X_F, test_labels=F_labels, messages=1, base_clf=base_clf)
                            pssl_trainTime = pssl_trainTime + sl_predictTime + sl_trainTime
                            print("F1: {:.5f} PRECISION: {:.5f}  RECALL: {:.5f}  TrainTime: {:.3f}s".format(pssl_fscore, pssl_precision, pssl_recall, pssl_trainTime))
                            algorithm_results[algorithm]['High']['F1'].append(pssl_fscore)
                            algorithm_results[algorithm]['High']['Time'].append(pssl_trainTime)
                            

                            F_df["pssl_predictions"] = pssl_Fpredictions


                           
                            
                            U_midlow_df = pd.concat([U_mid_df, U_low_df])
                          

                            U_midlow_df2 = np.stack(U_midlow_df['Image'].values)
                           
                            nsamples, nx, ny = U_midlow_df2.shape
                            U_midlow_df2 = U_midlow_df2.reshape(nsamples,nx*ny)
                            
                            start_time = time.time()
                            ssl_Umlpredictions, ssl_Umlprobabilities = ssl_clf.predict(U_midlow_df2), ssl_clf.predict_proba(U_midlow_df2)
                            ssl_predictTime = time.time() - start_time
                            U_midlow_df['ssl_prediction'] = ssl_Umlpredictions
                            pseudoHigh_L_df['ssl_prediction'] = pseudoHigh_L_df['sl_prediction'] # dummy column (the retrained pseudo-labelling models will use this column as 'label')
                            if classification_type == 'binary':
                                    U_midlow_df['ssl_probability'] = (np.hsplit(ssl_Umlprobabilities,2))[0]
                            else: 
                                    U_midlow_df['ssl_probability'] = np.array(ssl_Umlprobabilities).tolist()
                                
                       

                            U_midlow_df, U_midlow_high_df, U_midlow_mid_df, U_midlow_low_df = assign_confidence(U_midlow_df, high_confidence_window, low_confidence_window, 
                                                                                                probability_column_name = 'ssl_probability', confidence_column_name = 'ssl_confidence', debug=False, split=True)
                            pseudoHigh_high_L_df = pd.concat([pseudoHigh_L_df, U_midlow_high_df])
                      
                            pseudoHigh_high_L_labels = pseudoHigh_high_L_df['ssl_prediction']
                            pshh = np.stack(pseudoHigh_high_L_df['Image'].values)                    
                            nsamples, nx, ny = pshh.shape
                            pshh = pshh.reshape((nsamples,nx*ny))
                            print("\t\t\trpssl (high confidence pseudo labels, twice). L: {}b {}m\t".
                                format(len(benign_L_df), len(malicious_L_df)))
                            rpssl_clf, rpssl_precision, rpssl_recall, rpssl_fscore, rpssl_trainTime, rpssl_testTime, rpssl_Fpredictions = train_test(train_df=pshh, train_labels=pseudoHigh_high_L_labels, 
                                                                                                test_df=X_F, test_labels=F_labels, messages=1, base_clf=base_clf)
                            rpssl_trainTime = rpssl_trainTime + pssl_trainTime
                            print("F1: {:.5f} PRECISION: {:.5f}  RECALL: {:.5f}  TrainTime: {:.3f}s".format(rpssl_fscore, rpssl_precision, rpssl_recall, rpssl_trainTime))
                            
                            algorithm_results[algorithm]['HighHigh']['F1'].append(rpssl_fscore)
                            algorithm_results[algorithm]['HighHigh']['Time'].append(rpssl_trainTime)
                        
                            ############ ACTIVE LEARNING
                            #### This is for the support model to use for active learning

                            #print("ACTIVE BUDGET", Lm)
                            #print(active_budget[0])# generating support L
                            #malicious_support_L_df = malicious_L_df.sample(n=int(Lm * active_budget[0]))
                            #benign_support_L_df = benign_L_df.sample(n=int(Lb * active_budget[1]))
                            #support_L_df = pd.concat([malicious_support_L_df, benign_support_L_df])
                            #support_L_labels = support_L_df[label_name]
                            #support_L_df = convert_from_csv(support_L_df)
                        
                            #support_L_labels = np.array(support_L_df["Label"])
                            support_L_df2 = np.stack(support_L_df["Image"].values)
                            nsamples, nx, ny = support_L_df2.shape
                            support_L_df2 = support_L_df2.reshape(nsamples,nx*ny)

                            # regenerating U
                        # malicious_UL_df = malicious_df[malicious_df['holdout']==False]
                            #benign_UL_df = benign_df[benign_df['holdout']==False]
                            #malicious_U_df = malicious_UL_df.drop(malicious_support_L_df.index)
                            #benign_U_df = benign_UL_df.drop(benign_support_L_df.index)
                            #support_U_df = pd.concat([malicious_U_df, benign_U_df])
                            #support_U_labels = support_U_df[label_name]
                            #support_U_df = convert_from_csv(support_U_df)
                            support_U_labels = np.array(support_U_df["Label"])
                            support_U_df2 = np.stack(support_U_df['Image'].values)

                            nsamples, nx, ny = support_U_df2.shape
                            support_U_df2 = support_U_df2.reshape(nsamples,nx*ny)
                        
                            # training support sl and predicting labels on the (new) U
                            support_sl_clf, support_sl_precision, support_sl_recall, support_sl_fscore, support_sl_trainTime, support_sl_testTime, support_sl_Fpredictions = train_test(train_df=support_L_df2, train_labels=support_L_labels, 
                                                                                                test_df=X_F, test_labels=F_labels,  messages=0, base_clf=base_clf)
                                             

                            start_time = time.time()
                            F_df["support_sl_predictions"] = support_sl_Fpredictions


                            support_sl_Upredictions, support_sl_Uprobabilities = support_sl_clf.predict(support_U_df2), support_sl_clf.predict_proba(support_U_df2)
                            support_sl_predictTime = time.time() - start_time
                            support_U_df['support_sl_prediction'] = support_sl_Upredictions
                            support_L_df['support_sl_prediction'] = support_L_df[label_name] # dummy column (the pseudo-labelling models will use this column as 'label')
                            if classification_type == 'binary':
                                    support_U_df['support_sl_probability'] = (np.hsplit(support_sl_Uprobabilities,2))[0]
                            else: 
                                    support_U_df['support_sl_probability'] = np.array(support_sl_Uprobabilities).tolist()
                            
                          

                            # Assign confidence based on (predicted) probability and split dataframe
                            support_U_df, support_U_high_df, support_U_mid_df, support_U_low_df = assign_confidence(support_U_df, high_confidence_window, low_confidence_window, 
                                                                                                probability_column_name = 'support_sl_probability', confidence_column_name = 'support_sl_confidence', debug=False, split=True)
                            
                            
                            # Train and Test - VANILLA Active Learning
                            leftout_budget = int(Lm * (1-active_budget[0])) + int(Lb * (1-active_budget[1]))
                            

                            if 'high' in al_confidences:
                                print("\t\t\tahssl (active-high confidence)... L: {}b {}m + {} leftout...\t".
                                format(len(benign_support_L_df), len(malicious_support_L_df), leftout_budget))
                                ahssl_precision, ahssl_recall, ahssl_fscore, ahssl_trainTime, al_high_Fpredictions = active_learning(confidences_df=support_U_high_df, 
                                                                                                                baseTrain_df=support_L_df, 
                                                                                                                validation_df=F_df, 
                                                                                                                base_clf=base_clf,
                                                                                                                trials=activeLearning_trials, ssl=False, samples=leftout_budget, label_name = label_name, messages=1)
                                ahssl_trainTime = ahssl_trainTime + support_sl_predictTime + support_sl_trainTime
                                print("F1: {:.5f} PRECISION: {:.5f}  RECALL: {:.5f}  TrainTime: {:.3f}s".format(ahssl_fscore, ahssl_precision, ahssl_recall, ahssl_trainTime))
                                algorithm_results[algorithm]['ActiveHigh']['F1'].append(ahssl_fscore)
                                algorithm_results[algorithm]['ActiveHigh']['Time'].append(ahssl_trainTime)
                            
                            if 'mid' in al_confidences:
                                print("\t\t\taossl (active-mid confidence)... L: {}b {}m + {} leftout...\t".
                                format(len(benign_support_L_df), len(malicious_support_L_df), leftout_budget))
                                aossl_precision, aossl_recall, aossl_fscore, aossl_trainTime, al_mid_Fpredictions = active_learning(confidences_df=support_U_mid_df, 
                                                                                                                baseTrain_df=support_L_df, 
                                                                                                                validation_df=F_df, 
                                                                                                                    base_clf=base_clf,
                                                                                                                trials=activeLearning_trials, ssl=False, samples=leftout_budget, label_name = label_name, messages=1)
                                aossl_trainTime = aossl_trainTime + support_sl_predictTime + support_sl_trainTime
                                print("F1: {:.5f} PRECISION: {:.5f}  RECALL: {:.5f}  TrainTime: {:.3f}s".format(aossl_fscore, aossl_precision, aossl_recall, aossl_trainTime))
                                algorithm_results[algorithm]['ActiveMid']['F1'].append(aossl_fscore)
                                algorithm_results[algorithm]['ActiveMid']['Time'].append(aossl_trainTime)
                            
                            if 'low' in al_confidences:
                                print("\t\t\talssl (active-low confidence)... L: {}b {}m + {} leftout...\t".
                                format(len(benign_support_L_df), len(malicious_support_L_df), leftout_budget))
                                alssl_precision, alssl_recall, alssl_fscore, alssl_trainTime, al_low_Fpredictions= active_learning(confidences_df=support_U_low_df, 
                                                                                                                baseTrain_df=support_L_df, 
                                                                                                                validation_df=F_df, 
                                                                                                                base_clf=base_clf,
                                                                                                                trials=activeLearning_trials, ssl=False, samples=leftout_budget, label_name = label_name, messages=1)
                                alssl_trainTime = alssl_trainTime + support_sl_predictTime + support_sl_trainTime
                                print("F1: {:.5f} PRECISION: {:.5f}  RECALL: {:.5f} TrainTime: {:.3f}s".format(alssl_fscore, alssl_precision, alssl_recall, alssl_trainTime))
                                algorithm_results[algorithm]['ActiveLow']['F1'].append(alssl_fscore)
                                algorithm_results[algorithm]['ActiveLow']['Time'].append(alssl_trainTime)
                            
                                F_df["al_mid_Fpredictions"] = al_mid_Fpredictions
                                F_df["al_low_Fpredictions"] = al_low_Fpredictions
                                F_df["al_high_Fpredictions"] = al_high_Fpredictions


                            
                            ############
                            #### PREPARE: Pseudo Labelling + Active Learning
                            
                            ## First, train the support pseudo-labelling model using the labels with high-confidence
                            support_L_df['support_sl_prediction'] = support_L_df[label_name]
                            pseudoHigh_support_L_df = pd.concat([support_U_high_df, support_L_df])
                            pseudoHigh_support_L_labels = pseudoHigh_support_L_df['support_sl_prediction']
                                         
                            pshs = np.stack(pseudoHigh_support_L_df['Image'].values)

                            nsamples, nx, ny = pshs.shape
                            pshs = pshs.reshape(nsamples,nx*ny)


                            
                            print("\t\t\tPseudo-Active Learning: the base size of the training set is: {}".format(len(pseudoHigh_support_L_df)))
                            support_pssl_clf, support_pssl_precision, support_pssl_recall, support_pssl_fscore, support_pssl_trainTime, support_pssl_testTime, support_pssl_Fpredictions = train_test(train_df=pshs, train_labels=pseudoHigh_support_L_labels, 
                                                                                                test_df=X_F, test_labels=F_labels,  messages=0, base_clf=base_clf)
                            support_pssl_trainTime = support_pssl_trainTime + support_sl_trainTime + support_sl_predictTime
                            
                            ## Then, use such support pseudo-labelling to predict the confidence of the remaining samples in U
                            support_U_midlow_df = pd.concat([support_U_mid_df, support_U_low_df])
                            start_time = time.time()
                            suml = np.stack(support_U_midlow_df["Image"].values)
                            nsamples,nx,ny = suml.shape
                            suml = suml.reshape(nsamples,nx*ny)
                            support_pssl_Umlpredictions, support_pssl_Umlprobabilities = support_pssl_clf.predict(suml), support_pssl_clf.predict_proba(suml)
                            support_pssl_predictTime = time.time() - start_time
                            support_U_midlow_df['support_ssl_prediction'] = support_pssl_Umlpredictions
                            pseudoHigh_support_L_df['support_ssl_prediction'] = pseudoHigh_support_L_df['support_sl_prediction'] # dummy column (the retrained pseudo-labelling models will use this column as 'label')
                            if classification_type == 'binary':
                                    support_U_midlow_df['support_ssl_probability'] = (np.hsplit(support_pssl_Umlprobabilities,2))[0]
                            else: 
                                    support_U_midlow_df['support_ssl_probability'] = np.array(support_pssl_Umlprobabilities).tolist()
                                    
                           
                            support_U_midlow_df, support_U_midlow_high_df, support_U_midlow_mid_df, support_U_midlow_low_df = assign_confidence(support_U_midlow_df, high_confidence_window, low_confidence_window,
                                                                                                                                                probability_column_name = 'support_ssl_probability', confidence_column_name = 'support_ssl_confidence', debug=False, split=True)
                           
                            ## TRAIN and TEST - Pesudo-Active Learning
                            if 'high' in al_confidences:
                                print("\t\t\tpahssl (pseudoActive-high confidence)... L: {}b {}m + {} leftout...\t".
                                format(len(benign_support_L_df), len(malicious_support_L_df), leftout_budget))
                                pahssl_precision, pahssl_recall, pahssl_fscore, pahssl_trainTime, p_high_Fpredictions = active_learning(confidences_df=support_U_midlow_high_df, 
                                                                                                                baseTrain_df=pseudoHigh_support_L_df, 
                                                                                                                validation_df=F_df, 
                                                                                                                base_clf=base_clf,
                                                                                                                trials=activeLearning_trials, ssl=True, samples=leftout_budget, label_name = label_name, messages=1)
                                pahssl_trainTime = pahssl_trainTime + support_pssl_trainTime + support_pssl_predictTime
                                print("F1: {:.5f} PRECISION: {:.5f}  RECALL: {:.5f}  TrainTime: {:.3f}s".format(pahssl_fscore, pahssl_precision, pahssl_recall, pahssl_trainTime))
                                algorithm_results[algorithm]['PseudoHigh']['F1'].append(pahssl_fscore)
                                algorithm_results[algorithm]['PseudoHigh']['Time'].append(pahssl_trainTime)
                            
                            if 'mid' in al_confidences:
                                print("\t\t\tpahssl (pseudoActive-mid confidence)... L: {}b {}m + {} leftout...\t".
                                format(len(benign_support_L_df), len(malicious_support_L_df), leftout_budget))
                                paossl_precision, paossl_recall, paossl_fscore, paossl_trainTime, p_mid_Fpredictions = active_learning(confidences_df=support_U_midlow_mid_df, 
                                                                                                                baseTrain_df=pseudoHigh_support_L_df, 
                                                                                                                validation_df=F_df, 
                                                                                                                base_clf=base_clf,
                                                                                                                trials=activeLearning_trials, ssl=True, samples=leftout_budget, label_name = label_name, messages=1)
                                paossl_trainTime = paossl_trainTime + support_pssl_trainTime + support_pssl_predictTime
                                print("F1: {:.5f} PRECISION: {:.5f}  RECALL: {:.5f} TrainTime: {:.3f}s".format(paossl_fscore, paossl_precision, paossl_recall, paossl_trainTime))
                                algorithm_results[algorithm]['PseudoMid']['F1'].append(paossl_fscore)
                                algorithm_results[algorithm]['PseudoMid']['Time'].append(paossl_trainTime)
                            
                            if 'low' in al_confidences:
                                print("\t\t\tpahssl (pseudoActive-low confidence)... L: {}b {}m + {} leftout...\t".
                                format(len(benign_support_L_df), len(malicious_support_L_df), leftout_budget))
                                palssl_precision, palssl_recall, palssl_fscore, palssl_trainTime, p_low_Fpredictions = active_learning(confidences_df=support_U_midlow_low_df, 
                                                                                                                baseTrain_df=pseudoHigh_support_L_df, 
                                                                                                                validation_df=F_df, 
                                                                                                                    base_clf=base_clf,
                                                                                                                trials=activeLearning_trials, ssl=True, samples=leftout_budget, label_name = label_name, messages=1)
                                palssl_trainTime = palssl_trainTime + support_pssl_trainTime + support_pssl_predictTime
                                print("F1: {:.5f} PRECISION: {:.5f}  RECALL: {:.5f} TrainTime: {:.3f}s".format(palssl_fscore, palssl_precision, palssl_recall, palssl_trainTime))
                                algorithm_results[algorithm]['PseudoLow']['F1'].append(palssl_fscore)
                                algorithm_results[algorithm]['PseudoLow']['Time'].append(palssl_trainTime)
                                F_df["p_mid_Fpredictions"] = p_mid_Fpredictions
                                F_df["p_low_Fpredictions"] = p_low_Fpredictions
                                F_df["p_high_Fpredictions"] = p_high_Fpredictions


                        
    runtime = time.time() - begin
    print("Runtime total: {:.5f}".format(runtime))

    if classification_type== "binary":
        wilcoxon_subdirectory = "results/output_images_binary/wilcoxon_test"
        f_score_time_subdirectory = "results/output_images_binary/f1_score_execution_time"
    else:
        wilcoxon_subdirectory = "results/output_images_multiclass/wilcoxon_test"
        f_score_time_subdirectory = "results/output_images_multiclass/f1_score_execution_time"

    os.makedirs(wilcoxon_subdirectory, exist_ok=True)
    os.makedirs(f_score_time_subdirectory, exist_ok=True)

    

    for algorithm, cases in algorithm_results.items():
        avgF1 = []
        avgTime = []
        for case, results in cases.items():
            total_F1_algorithm_case = sum(results['F1'])
            total_execution_time_case = sum(results['Time'])
            average_F1_algorithm_case = total_F1_algorithm_case / executions_total
            average_execution_time_case = total_execution_time_case / executions_total
            print(f"Average F1 score for {algorithm} ({case}): {average_F1_algorithm_case}")
            print(f"Average execution time for {algorithm} ({case}): {average_execution_time_case} seconds")
            avgF1.append(average_F1_algorithm_case)
            avgTime.append(average_execution_time_case)
        plot_average(avgF1, avgTime)
        filename_avg = f"f1_score_execution_time_{algorithm}_raw.png"
        f_score_time_path = os.path.join(f_score_time_subdirectory, filename_avg)
        plt.savefig(f_score_time_path)
        plt.close()
        f1_base = sum(algorithm_results[algorithm]["Base"]['F1'])
        f1_high = sum(algorithm_results[algorithm]['High']['F1'])
        f1_highhigh = sum(algorithm_results[algorithm]['HighHigh']['F1'])
        best_pseudo_value = max(f1_base, f1_high, f1_highhigh)
        f1_activelow = sum(algorithm_results[algorithm]['ActiveLow']['F1'])
        f1_activemid = sum(algorithm_results[algorithm]['ActiveMid']['F1'])
        f1_activehigh = sum(algorithm_results[algorithm]['ActiveHigh']['F1'])
        f1_pseudolow = sum(algorithm_results[algorithm]['PseudoLow']['F1'])
        f1_pseudomid = sum(algorithm_results[algorithm]['PseudoMid']['F1'])
        f1_pseudohigh = sum(algorithm_results[algorithm]['PseudoHigh']['F1'])
        best_active_value = max(f1_activelow, f1_activemid, f1_activehigh, f1_pseudolow, f1_pseudomid, f1_pseudohigh)
        if best_pseudo_value == f1_base:
            best_pseudo = 'baselinessl'
            best_pseudo_scores = algorithm_results[algorithm]['Base']['F1']
        elif best_pseudo_value == f1_high:
            best_pseudo = 'pseudohigh'
            best_pseudo_scores = algorithm_results[algorithm]['High']['F1']
        elif best_pseudo_value == f1_highhigh:
            best_pseudo = 'pseudohigh2'
            best_pseudo_scores = algorithm_results[algorithm]['HighHigh']['F1']
                
        
        
        
        if best_active_value == f1_activelow:
            best_active = 'activelow'
            bestactive_scores = algorithm_results[algorithm]['ActiveLow']['F1']
        elif best_active_value == f1_activehigh:
            best_active = 'activehigh'
            bestactive_scores = algorithm_results[algorithm]['ActiveHigh']['F1']
        elif best_active_value == f1_activemid:
            best_active = 'activemid'
            bestactive_scores = algorithm_results[algorithm]['ActiveMid']['F1']
        elif best_active_value == f1_pseudomid:
            best_active = 'pseudoactivemid'
            bestactive_scores = algorithm_results[algorithm]['PseudoMid']['F1']
        elif best_active_value == f1_pseudolow:
            best_active = 'pseudoactivelow'
            bestactive_scores = algorithm_results[algorithm]['PseudoLow']['F1']
        elif best_active_value == f1_pseudohigh:
            best_active = 'pseudoactivehigh'
            bestactive_scores = algorithm_results[algorithm]['PseudoHigh']['F1']    

        n1 = len(algorithm_results[algorithm]["LowerBound"]['F1'])
        n2 = len(best_pseudo_scores)
        n3 = len(bestactive_scores)
        statistic, p_value_pseudo = mannwhitneyu(algorithm_results[algorithm]["LowerBound"]['F1'], best_pseudo_scores)
        #print("Best pseuod scores", best_pseudo_scores)
        

        # Calculate expected mean and standard deviation of the U-statistic
        
        U2 = n1*n2 - statistic
        U = min(statistic,U2)
        expected_mean_U = n1 * n2 / 2
        expected_std_U = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)

        # Calculate the z-value
        z_value_pseudo = (statistic - expected_mean_U +0.5) / expected_std_U
        print(f"BASELINE VS PURE SSL {algorithm} :")
        print(f"Wilcoxon Test Statistic: {statistic}")
        print(f"P-value: {p_value_pseudo}")
        print(f"z-value: {z_value_pseudo}")

        statistic, p_value_active = mannwhitneyu(algorithm_results[algorithm]["LowerBound"]['F1'], bestactive_scores)
        U2 = n1*n3 - statistic
        U = min(statistic,U2)
        expected_mean_U = n1 * n3 / 2
        expected_std_U = np.sqrt((n1 * n3 * (n1 + n3 + 1)) / 12)

        # Calculate the z-value
        z_value_active = (statistic - expected_mean_U +0.5) / expected_std_U
        print(f"BASELINE VS ACTIVE LEARNING {algorithm} : ")
        print(f"Wilcoxon Test Statistic: {statistic}")
        print(f"P-value: {p_value_active}")
        print(f"z-value: {z_value_active}")

        plot_wilcoxon(best_pseudo, best_active, p_value_pseudo, z_value_pseudo, p_value_active, z_value_active)
        filename_wilcoxon = f"wilcoxon_test_{algorithm}_raw.png"
        wilcoxon_path = os.path.join(wilcoxon_subdirectory, filename_wilcoxon)
        #plt.figure(figsize=(8, 6))
        plt.savefig(wilcoxon_path)
        plt.close()

        

                

    

        
        

    ########## END EVALUATION ##########

