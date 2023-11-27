from sklearn.ensemble import *
from sklearn.metrics import *
import time
import numpy as np
import pandas as pd
import copy
import cv2
from sklearn.decomposition import PCA


def load_and_preprocess_image(image_path):
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise Exception(f"Failed to load image from {image_path}")

        # Resize the image to the desired dimensions
        image = cv2.resize(image, (128, 128))

        return image
    except Exception as e:
        print(f"Error loading or preprocessing image: {e}")
        return None  # Return None to indicate an error



def convert(df):
    df2 = pd.DataFrame(columns=['Image', 'Label'])
    
    for im in range(len(df)):
        path = df["Image"].iloc[im]
        
        try:
            img = load_and_preprocess_image(path)  # Call the load_and_preprocess_image function
        except Exception as e:
            print("Problem: %s", str(e))
            continue
        
        if img is not None:
            label = df["Label"].iloc[im]
            new_row = pd.DataFrame({'Image': [img], 'Label': [label]})
            df2 = pd.concat([df2,new_row], ignore_index=True)
       
        else:
            print("Error: %s", path)
    
    return df2




def train_test_binary(train_df, train_labels, test_df, test_labels, messages=0, base_clf=None):
    malware_label = next(label for label in train_labels if label != 0)
    print("Malware label", malware_label)
    start_time = time.time()
    if base_clf==None:
        clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
                 min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-2, 
                 random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
    else:
        clf = copy.deepcopy(base_clf)
    if messages == 1:
        print("Train size: {}---".format(len(train_df)), end="")
    clf.fit(train_df, train_labels)
    train_time = time.time() - start_time
    start_time = time.time()
    predictions = clf.predict(test_df)
    test_time = time.time() - start_time
    precision, recall, fscore, support = precision_recall_fscore_support(test_labels, predictions, beta=1.0, 
                                    labels=[0, malware_label],  average=None, sample_weight=None, zero_division='warn')
    if messages==2:
        print("\tF1-score: {:.5f}".format(fscore[0]), end="")
    return clf, precision, recall, fscore[0], train_time, test_time, predictions

def assign_confidence_binary(df, high_confidence_window, low_confidence_window, probability_column_name = 'Probability', confidence_column_name = 'Confidence', debug=True, split=True):
    df_high = None
    df_mid = None
    df_low = None
    mid_confidence_lowerBound, mid_confidence_upperBound, high_confidence_upperBound, high_confidence_lowerBound = find_confidence_ranges(high_confidence_window, low_confidence_window, debug=debug)
    #print(mid_confidence_lowerBound,mid_confidence_upperBound, high_confidence_lowerBound, high_confidence_upperBound)
    df.loc[(df[probability_column_name] > mid_confidence_lowerBound) & (df[probability_column_name] <= mid_confidence_upperBound), confidence_column_name] = 'low'
    df.loc[(df[probability_column_name] > high_confidence_lowerBound) & (df[probability_column_name] <= mid_confidence_lowerBound), confidence_column_name] = 'mid'
    df.loc[(df[probability_column_name] > mid_confidence_upperBound) & (df[probability_column_name] <= high_confidence_upperBound), confidence_column_name] = 'mid'
    df.loc[(df[probability_column_name] >= 0) & (df[probability_column_name] <= high_confidence_lowerBound), confidence_column_name] = 'high'
    df.loc[(df[probability_column_name] > high_confidence_upperBound) & (df[probability_column_name] <= 1), confidence_column_name] = 'high'
    if split:
        df_high = df.loc[(df[confidence_column_name] == 'high')]
        df_mid = df.loc[(df[confidence_column_name] == 'mid')]
        df_low = df.loc[(df[confidence_column_name] == 'low')]
    if debug:
        print("Check for null values: {}".format(df[confidence_column_name].isnull().values.any()))
        print("Size of high-mid-low: {}\t{}\t{}".format(len(df_high), len(df_mid), len(df_low)))
    
    print("\nSamples with high confidence:", len(df_high))
    print("Samples with mid confidence:", len(df_mid))
    print("Samples with low confidence:", len(df_low))
    print("\n")
    return df, df_high, df_mid, df_low

def train_test_multiclass(train_df, train_labels, test_df, test_labels, messages=0, base_clf=None):
    start_time = time.time()
    labels = list(set(train_labels))
    
    
    if base_clf==None:
        clf = RandomForestClassifier(n_estimators=100,n_classes=26,  criterion='gini', max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
                 min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-2, 
                 random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
    else:
        clf = copy.deepcopy(base_clf)
    if messages == 1:
        print("Train size: {}---".format(len(train_df)), end="")
        
   
    clf.fit(train_df, train_labels)
    train_time = time.time() - start_time
    start_time = time.time()

    predictions = clf.predict(test_df)
    test_time = time.time() - start_time
    precision, recall, fscore, support = precision_recall_fscore_support(test_labels, predictions, beta=1.0, 
                                labels=labels,  average="weighted", sample_weight=None, zero_division="warn")
    if messages==2:
        print("\tF1-score: {:.5f}".format(fscore[0]), end="")
    #print(fscore)
    return clf, precision, recall, fscore, train_time, test_time, predictions

def assign_confidence_multiclass(df, high_confidence_window, low_confidence_window, probability_column_name = 'Probability', confidence_column_name = 'Confidence', debug=True, split=True):
    df_high = None
    df_mid = None
    df_low = None
    mid_confidence_lowerBound, mid_confidence_upperBound, high_confidence_upperBound, high_confidence_lowerBound = find_confidence_ranges(high_confidence_window, low_confidence_window, debug=debug)
    #print(df[probability_column_name])
    df['overall_confidence'] = df[probability_column_name].apply(lambda x: max(x))
    

   
    def map_values(val):
      
           if val > mid_confidence_lowerBound and val <= mid_confidence_upperBound:
              return 'low'#result.append("low")
           elif val > high_confidence_lowerBound and val <= mid_confidence_lowerBound: 
              return 'mid' #sult.append("mid")
           elif val > mid_confidence_upperBound and val <= high_confidence_upperBound:
              return 'mid' #sult.append("mid")
           elif val >= 0 and val<= high_confidence_lowerBound:
              return 'high' #sult.append("high")
           elif val > high_confidence_upperBound and  val <= 1:  # Handle other cases if needed
              return 'high' #sult.append("high")
           #return result
    
    df[confidence_column_name] = df['overall_confidence'].apply(map_values)#all_confidences = []

    if split:
        #df_mid = df[df[confidence_column_name].apply(lambda x: 'mid' in x)]
        df_high = df.loc[(df[confidence_column_name] == 'high')]
        df_mid = df.loc[(df[confidence_column_name] == 'mid')]
        df_low = df.loc[(df[confidence_column_name] == 'low')]
    if debug:
        print("Check for null values: {}".format(df[confidence_column_name].isnull().values.any()))
        print("Size of high-mid-low: {}\t{}\t{}".format(len(df_high), len(df_mid), len(df_low)))
    return df, df_high, df_mid, df_low


def find_confidence_ranges(high_confidence_window, low_confidence_window, debug=False):
    if (high_confidence_window + low_confidence_window) > 1:
        print("Error! Wrong confidence windows!")
        return False
    mid_confidence_window = 1 - (high_confidence_window + low_confidence_window)
    high_confidence_upperBound = 1-(high_confidence_window/2)
    high_confidence_lowerBound = 0+(high_confidence_window/2)
    mid_confidence_upperBound = high_confidence_upperBound - (mid_confidence_window/2)
    mid_confidence_lowerBound = high_confidence_lowerBound + (mid_confidence_window/2)
    low_confidence_upperBound = mid_confidence_upperBound - (low_confidence_window/2)
    low_confidence_lowerBound = mid_confidence_lowerBound + (low_confidence_window/2)
    if debug==True:
        print("Low Confidence: from {:.3f} < x < {:.3f}".format(mid_confidence_lowerBound, mid_confidence_upperBound))
        print("Mid Confidence: from {:.3f} < x < {:.3f} or {:.3f} < x < {:.3f}".format(high_confidence_lowerBound, mid_confidence_lowerBound, mid_confidence_upperBound, high_confidence_upperBound))
        print("High Confidence: from 0 < x < {:.3f} or {:.3f} < x < 1".format(high_confidence_lowerBound, high_confidence_upperBound))
    return mid_confidence_lowerBound, mid_confidence_upperBound, high_confidence_upperBound, high_confidence_lowerBound

def initialize_emptyResults():
    precisions = np.array([])
    recalls = np.array([])
    fscores = np.array([])
    times = np.array([])
    return precisions, recalls, fscores, times

def add_results(precisions, recalls, fscores, times, precision_temp, recall_temp, fscore_temp, time_temp):
    precisions = np.append(precisions, precision_temp)
    recalls = np.append(recalls, recall_temp)
    fscores = np.append(fscores, fscore_temp)
    times = np.append(times, time_temp)
    return precisions, recalls, fscores, times

def setRelabeling(samples, df):
    if samples > len(df):
        return len(df)
    return samples

def active_learning_binary(confidences_df, baseTrain_df, validation_df, features, trials=5, ssl=True, samples=50, label_name = 'label', base_clf=None, messages=0):
    al_precisions, al_recalls, al_fscores, al_times = initialize_emptyResults()
    messages = messages
    for i in range(trials):
        #choose samples
        relabeling = setRelabeling(samples, confidences_df)
        active_df = confidences_df.sample(n=relabeling)
        if ssl:
            active_df['support_ssl_prediction'] = active_df[label_name]
        activeTrain_df = pd.concat([active_df, baseTrain_df])
        if ssl:
            activeTrain_labels = activeTrain_df['support_ssl_prediction']
        else:
            activeTrain_labels = activeTrain_df[label_name]
        validation_labels = validation_df[label_name]
        al_clf, al_precision, al_recall, al_fscore, al_trainTime, al_testTime, al_Fpredictions = train_test_binary(train_df=activeTrain_df.iloc[:, :features], train_labels=activeTrain_labels, test_df=validation_df.iloc[:, :features], test_labels=validation_labels, messages=messages, base_clf=base_clf)
        al_precisions, al_recalls, al_fscores, al_times = add_results(al_precisions, al_recalls, al_fscores, al_times, 
                                                                         al_precision, al_recall, al_fscore, al_trainTime)
        messages = 0 #print debug messages only once per batch of trials
    al_precision = al_precisions.mean()
    al_recall = al_recalls.mean()
    al_fscore = al_fscores.mean()
    al_trainTime = al_times.mean()    
    if messages>0:
        print("\tF1-score (avg): {:.5f}".format(al_fscore))
    return al_precision, al_recall, al_fscore, al_trainTime

def active_learning_multiclass(confidences_df, baseTrain_df, validation_df, features, trials=2, ssl=True, samples=50, label_name = 'Label', base_clf=None, messages=0):
    al_precisions, al_recalls, al_fscores, al_times = initialize_emptyResults()
    messages = messages
    
    for i in range(trials):
        #choose samples
        relabeling = setRelabeling(samples, confidences_df)
        active_df = confidences_df.sample(n=relabeling)
        if ssl:
            active_df['support_ssl_prediction'] = active_df[label_name]
        activeTrain_df = pd.concat([active_df, baseTrain_df])
        if ssl:
            activeTrain_labels = activeTrain_df['support_ssl_prediction']
        else:
            activeTrain_labels = np.array(activeTrain_df[label_name])
        validation_labels = np.array(validation_df[label_name])

       
        

        al_clf, al_precision, al_recall, al_fscore, al_trainTime, al_testTime, al_Fpredictions = train_test_multiclass(train_df=activeTrain_df.iloc[:, :features], train_labels=activeTrain_labels, test_df=validation_df.iloc[:, :features], test_labels=validation_labels, messages=messages, base_clf=base_clf)
        al_precisions, al_recalls, al_fscores, al_times = add_results(al_precisions, al_recalls, al_fscores, al_times, 
                                                                         al_precision, al_recall, al_fscore, al_trainTime)
        messages = 0 #print debug messages only once per batch of trials
    al_precision = al_precisions.mean()
    al_recall = al_recalls.mean()
    al_fscore = al_fscores.mean()
    al_trainTime = al_times.mean()    
    if messages>0:
        print("\tF1-score (avg): {:.5f}".format(al_fscore))
    return al_precision, al_recall, al_fscore, al_trainTime

def active_learning_binary_raw(confidences_df, baseTrain_df, validation_df, trials=5, ssl=True, samples=50, label_name = 'label', base_clf=None, messages=0):
    al_precisions, al_recalls, al_fscores, al_times = initialize_emptyResults()
    messages = messages
    pca = PCA(n_components=50)
    for i in range(trials):
        #choose samples
        relabeling = setRelabeling(samples, confidences_df)
        active_df = confidences_df.sample(n=relabeling)
        if ssl:
            active_df['support_ssl_prediction'] = active_df[label_name]
        activeTrain_df = pd.concat([active_df, baseTrain_df])
        if ssl:
            activeTrain_labels = activeTrain_df['support_ssl_prediction']
        else:
            activeTrain_labels = activeTrain_df[label_name]
        validation_labels = validation_df[label_name]
        train = np.stack(activeTrain_df["Image"].values)
        nsamples, nx, ny = train.shape
        train = train.reshape((nsamples,nx*ny))
        pca.fit(train)

                        # Transform your data into the PCA space
        train = pca.transform(train)
        val = np.stack(validation_df["Image"].values)
        nsamples, nx, ny = val.shape
        val = val.reshape((nsamples,nx*ny))
        pca.fit(val)

                        # Transform your data into the PCA space
        val = pca.transform(val)
        al_clf, al_precision, al_recall, al_fscore, al_trainTime, al_testTime, al_Fpredictions = train_test_binary(train_df=train, train_labels=activeTrain_labels, test_df=val, test_labels=validation_labels, messages=messages, base_clf=base_clf)
        al_precisions, al_recalls, al_fscores, al_times = add_results(al_precisions, al_recalls, al_fscores, al_times, 
                                                                         al_precision, al_recall, al_fscore, al_trainTime)
        messages = 0 #print debug messages only once per batch of trials
    al_precision = al_precisions.mean()
    al_recall = al_recalls.mean()
    al_fscore = al_fscores.mean()
    al_trainTime = al_times.mean()    
    if messages>0:
        print("\tF1-score (avg): {:.5f}".format(al_fscore))
    return al_precision, al_recall, al_fscore, al_trainTime, al_Fpredictions

def active_learning_multiclass_raw(confidences_df, baseTrain_df, validation_df, trials=5, ssl=True, samples=50, label_name = 'label', base_clf=None, messages=0):
    al_precisions, al_recalls, al_fscores, al_times = initialize_emptyResults()
    messages = messages
    pca = PCA(n_components=50)
    for i in range(trials):
        #choose samples
        relabeling = setRelabeling(samples, confidences_df)
        active_df = confidences_df.sample(n=relabeling)
        if ssl:
            active_df['support_ssl_prediction'] = active_df[label_name]
        activeTrain_df = pd.concat([active_df, baseTrain_df])
        if ssl:
            activeTrain_labels = activeTrain_df['support_ssl_prediction']
        else:
            activeTrain_labels = activeTrain_df[label_name]
        validation_labels = validation_df[label_name]
        train = np.stack(activeTrain_df["Image"].values)
        nsamples, nx, ny = train.shape
        train = train.reshape((nsamples,nx*ny))
        pca.fit(train)

                        # Transform your data into the PCA space
        train = pca.transform(train)
        val = np.stack(validation_df["Image"].values)
        nsamples, nx, ny = val.shape
        val = val.reshape((nsamples,nx*ny))
        pca.fit(val)

                        # Transform your data into the PCA space
        val = pca.transform(val)
        al_clf, al_precision, al_recall, al_fscore, al_trainTime, al_testTime, al_Fpredictions = train_test_multiclass(train_df=train, train_labels=activeTrain_labels, test_df=val, test_labels=validation_labels, messages=messages, base_clf=base_clf)
        al_precisions, al_recalls, al_fscores, al_times = add_results(al_precisions, al_recalls, al_fscores, al_times, 
                                                                         al_precision, al_recall, al_fscore, al_trainTime)
        messages = 0 #print debug messages only once per batch of trials
    al_precision = al_precisions.mean()
    al_recall = al_recalls.mean()
    al_fscore = al_fscores.mean()
    al_trainTime = al_times.mean()    
    if messages>0:
        print("\tF1-score (avg): {:.5f}".format(al_fscore))
    return al_precision, al_recall, al_fscore, al_trainTime, al_Fpredictions