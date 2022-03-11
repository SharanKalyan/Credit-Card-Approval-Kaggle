# Credit-Card-Approval-Kaggle

The source of data : https://www.kaggle.com/

The data is about approving Credit cards for the customers based on some parameters like age, gender, income, realty, education, family status etc. 
The data is highly imballanced with only less than 1% of the data belonging to one class (0) and the other 99% belonging to the other class. (1) 

I have Analysed the data and visualized it. While analysis, I was able to infer the most important features that affected the approval of Credit cards and i've creaetd some extra features that help classify between the 2 classes.

There are 3 Notebooks in this project.

1. "1_Data_Cleansing_Notebook.ipynb" - This notebook contains the codes for cleaning the data and creating the "clean_data.csv" file, which will be used in the        analysis and visualizations.
2. "2_Visualizations_Analysis.ipynb" notebook contains the visualizations and analysis aspects of the data and another file "clean_data_for_ml.csv" has been            created which will be used in the ML algorithms!.  
3. "3_Machine_Learning.ipynb" notebook contains the codes for the Machine learning aspects of the project.

Despite the data being highly imbalanced, I was able to achive an F1 score of 0.97 for the 0 class and 1.0 for the 1 class. The Analysis I have made helped me identify the most important feature and remove the unwanted features. 


Accuracy : 0.9997

Confusion Matrix :

    [[19    1]
     [0  3750]]

                    precision    recall  f1-score   support

               0       1.00      0.95      0.97        20
               1       1.00      1.00      1.00      3750

        accuracy                           1.00      3770
       macro avg       1.00      0.97      0.99      3770
    weighted avg       1.00      1.00      1.00      3770

Here's the ROC Curve! 

![image](https://user-images.githubusercontent.com/20862520/157607031-435b9004-4e97-4b57-a2cf-0d486576dd65.png)
