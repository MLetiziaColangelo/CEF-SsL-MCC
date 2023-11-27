# CEF-SsL-MCC

## Description
The CEF-SsL-MCC framework is designed for malware family classification. It provides functionalities for multi-class classification using various classifiers and feature extraction methods. The project contains the following folders:

- **data**: this directory contains the dataset Malevis in CSV format, that associates each image path to the family label;
- **utils**: this folder contains functions used throughout the project for both multi-class and binary classification scenarios; 
- **results**: this directory images for both f-scores and execution time averages and wilcoxon test for each model, considering direct multi-class classification and the utilisation of multiple binary classifier for multi-class classification;
- **main**: this directory contains the main functions for handling both malware classification with feature extraction and with raw pixels directly as input.


## Installation
This program is compatible with all operating systems and requires Python 3.10 or later. 

To install this project, follow these steps:
1. Clone the repository: 
```
git clone https://github.com/MLetiziaColangelo/CEF-SsL-MCC.git
```
2. Navigate to the project directory: 
```
cd CEF-SsL-MCC
```
3. Install dependencies: 
```
pip install numpy scipy scikit-learn scikit-image imbalanced-learn pandas opencv-python matplotlib pytorch
```

## Usage
Before executing the framework, configure it by specifying parameters:

- classification_type: Specify direct multi-class classification or multiple binary classifiers using `--classification` with the following options `multiclass` or `binary`.
-  classifier_name: Choose a classifier: `--classifier` with `all`, `svm`, `knn`, `rf`, or `gb`.
- feature_name: Select the feature extraction method: `--features` with `lbp`, `hog`, `glcm`, `combined` or `raw`.

To use this project use command-line options to set these parameters:

```
python main.py --classification multiclass --features lbp --classifier rf
```

- `multiclass`: Run the program for direct multi-class classification.
- `lbp`: Utilize Local Binary Pattern as the feature extraction method.
- `rf`: Use the Random Forest classifier.


