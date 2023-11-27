# main_script.py
from classification.features_classification import *
from classification.rawpixels_classification import *
import argparse

def main(classifier_name, feature_name, classification_type):
    if feature_name == "raw":
        rawpixels_classification(classifier_name,classification_type)
    else:
        features_classification(classifier_name,feature_name, classification_type)
    

if __name__ == "__main__":
    # Get user input for classifier, feature, and classification type
    parser = argparse.ArgumentParser(description="CEF-SsL-MCC")
    parser.add_argument("--classification",
                        type=str,
                        choices=["binary", "multiclass"],
                        help="Type of classification: binary or multiclass"
                        )
    parser.add_argument("--features",
                        type=str,
                        #nargs="+",
                        choices=["glcm",
                                 "lbp",
                                 "hog",
                                 "raw",
                                 "combined"
                                 ],
                        help="Type of features to use"
                        )
    parser.add_argument("--classifier",
                        type =str,
                        #nargs="+",
                        choices=["rf",
                                 "gb",
                                 "knn",
                                 "svm",
                                 "all"
                                 ],
                        help="Type of classifier"
                        )
    args = parser.parse_args()
    classifier_name = args.classifier
    feature_name = args.features
    classification_type = args.classification
    
    main(classifier_name, feature_name, classification_type)
