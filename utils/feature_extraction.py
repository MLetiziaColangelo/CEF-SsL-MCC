from sklearn.ensemble import *
from sklearn.metrics import *
from utils.functions import *
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_features_from_dataframe(df, feature):
    feature_list = []
    scaler = StandardScaler()
    for image_path in df["Image"]:
        image = load_and_preprocess_image(image_path) 
             
        
        if image is not None:           
           if feature == 'lbp':           
            features = extract_lbp_features(image)
           elif feature == 'hog':
            features = extract_hog_features(image)
           elif feature == 'glcm':
            features = extract_glcm_features(image)

        feature_list.append(features)
    if feature == 'hog':
       hog_features_scaled = scaler.fit_transform(feature_list)
       feature_list = extract_pca_features(hog_features_scaled, 50)

    return feature_list
        

# Function to extract LBP features from an image
def extract_lbp_features(image):
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize

    return hist

# Function to extract HOG features from an image
def extract_hog_features(image):
   hog_features= hog(image, pixels_per_cell = (8,8), cells_per_block = (2,2), orientations = 4, block_norm = 'L2-Hys')
   return hog_features

# Function that apply PCA dimensionality reduction to HOG features
def extract_pca_features(hog_features, n_components):
   pca = PCA(n_components=n_components)
   hog_features_pca = pca.fit_transform(hog_features)
   return hog_features_pca
    
# Function to extract GLCM features from an image
def extract_glcm_features(image):
    properties = ['Energy', 'Correlation', 'Dissimilarity','Homogeneity', 'Contrast']
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm_features = []
    glcm_feature_list = []
    

# Loop through the list of GLCM properties and create the specific feature names


    for angle in angles:      
            glcm_ben = graycomatrix(image, [1], [angle], symmetric=True, normed=True)

            properties = {
            'Energy': graycoprops(glcm_ben, 'energy')[0],
            'Correlation': graycoprops(glcm_ben, 'correlation')[0],
            'Dissimilarity': graycoprops(glcm_ben, 'dissimilarity')[0],
            'Homogeneity': graycoprops(glcm_ben, 'homogeneity')[0],
            'Contrast': graycoprops(glcm_ben, 'contrast')[0]
            }

            glcm_features.append(properties)
            
    
    flattened_list  = [value[0] for entry in glcm_features for value in entry.values()]
    

         
    return flattened_list
