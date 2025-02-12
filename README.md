# Project 3 - Heart Attack Outcome   
 
## Table of Contents   

- [Project Description](#project-description)   
- [Installation](#installation)   
- [Usage](#usage)   
- [Data Processing](#data-processing)   
- [File Structure](#file-structure)   
- [Problems Encountered](#problems-encountered)   
- [References](#references)   
- [Team Members](#team-members)   
- [License](#license)   

## PROJECT DESCRIPTION   
 As members of the Myocardio Minds, in partnership with St. Algorithm's Cardiac Institute (SACI), bring awareness to heart attack disease and survivability.  To do this, we created models to predict heart attack probability.  We then built a GUI for concerned users to leverage our models.  Finally, we made the models multilingual (English/Spanish/German).   

## INSTALLATION   

1. Ensure you have Python 3.10 or higher installed.   
2. Clone this repository: `git clone https://github.com/Pete1001/heart_attack_outcome.git`   

## USAGE   

1. Run the notebook: `Prediction_using_utility.ipynb`    
2. Run the notebook: `feature_importances.ipynb`   
3. Finally, run the notebook: `heart-disease-risk-gui_corrected.ipynb`     

## DATA PROCESSING:   

1. Data Collection: Heart disease risk dataset location: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset   
Mortality from heart attack dataset location: https://www.kaggle.com/datasets/asgharalikhan/mortality-rate-heart-patient-pakistan-hospital/   

2. Data Exploration: .csv files were read in and explored through the columns and info() methods.    

3. Data Cleanup:   
    - Encoding applied used OrdinalEncoder, OneHotEncoder and LabelEncoder        
    - Data was scaled using StandardScaler   
    - Performed train-test split   
    
4. Data Optimization:      
    - Create two Neural Network Models using Keras and TensorFlow to predict heart attack risk and outcome   
    - Create a utilities.py to efficiently call functions   
    - Important features were identified in order to create meaningful user questionnaires   

## FILE STRUCTURE: 

Code language: Python (python)   
Project2_Bank_Customer_Churn/   
├─Prediction_using_utility.ipynb  
├─feature_importances.ipynb    
├─heart-disease-risk-gui_corrected.ipynb       
├─utilities.py        
├─README.md   
└─Resources/   
     -FIC.Full CSV.csv      
     -MyocardioMinds.png   
     -StAlgorithmsCardiacInstitute.png   
     -heart_attack_model.h5   
     -heart_disease_health_indicators_BRFSS2015.csv   

## PROBLEMS ENCOUNTERED:   
-Our first attempt to develop and train an app, that predicts a heart attack outcome, we used sourced data https://www.kaggle.com/datasets/ankushpanday1/heart-attack-risk-predictions (heart_attack_predictions.csv).   

-This dataset proved to be unreliable and we could not get any model to make a prediction with accuracy being extremely low; thus, we investigated other datasets.  This dataset was synthetically created and dynamically balanced and proved to be unusable.   

Originally, we had discussed creating a second questionnaire, along with its own GUI.  That second questionnaire would predict one’s risk of heart disease.  However, the (above described) dataset accuracy complication plus GUI creation complications, led to insufficient time to complete the second questionnaire.    

During code testing we noticed some inconsistencies with the logical prediction.  We determined that this is due to biased data, because the data had more mortalities (0) than survivors (1).      

## REFERENCES   
1. https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset    

2. https://www.kaggle.com/datasets/asgharalikhan/mortality-rate-heart-patient-pakistan-hospital/    

3. https://www.kaggle.com/datasets/nareshbhat/health-care-data-set-on-heart-attack-possibility   

## TEAM MEMBERS   
1. Pete Link   
2. Ronak Dsouza  
3. Rebecca Carr  
4. Steve Vierling   

## LICENSE   
This project is licensed under the Myocardio Minds      
