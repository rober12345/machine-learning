# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:06:30 2024
EDA project  -- Data Preprocessing Project Tutorial
Step 1: Loading the dataset
You can download the dataset directly from Kaggle.com or from 
the following link: https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv. Store the raw data in the ./data/raw folder.

Step 2: Perform a complete EDA
This step is vital to ensure that we keep the variables
 that are strictly necessary and eliminate those that are not 
 relevant or do not provide information. 
 Use the example Notebook we worked on and adapt it to this use case.

Be sure to conveniently divide the data set into train and test as we have seen in the lesson.

Step 3: Save the processed dataset
After EDA you can save the data in the ./data/processed folder. Make sure to add the data folder in the .gitignore. The data as well as the models should not be uploaded to git
@author: rober ugalde


C:\Users\rober ugalde\Documents\GitHub\machine-learning-python-template


https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv



columns : id,name,host_id,host_name,neighbourhood_group,
neighbourhood,latitude,longitude,room_type,price,
minimum_nights,number_of_reviews,last_review,
reviews_per_month,calculated_host_listings_count,
availability_365
"""    
    
    

import pandas as pd
train_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
test_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
test_survived_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
test_data["host_id"] = test_survived_data["host_id"]

total_data = pd.concat([train_data, test_data]).reset_index(inplace = False)
total_data.drop(columns = ["index"], inplace = True)
total_data.head()





import matplotlib.pyplot as plt 
import seaborn as sns


fig, axis = plt.subplots(2, 2, figsize = (10, 7), gridspec_kw={'height_ratios': [6, 1]})

# Creating a multiple figure with histograms and box plots
sns.histplot(ax = axis[0, 0], data = total_data, x = "availability_365").set(xlabel = None)
sns.boxplot(ax = axis[1, 0], data = total_data, x = "availability_365")
sns.histplot(ax = axis[0, 1], data = total_data, x = "host_id").set(xlabel = None, ylabel = None)
sns.boxplot(ax = axis[1, 1], data = total_data, x = "host_id")

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()




fig, axis = plt.subplots(2, 2, figsize = (10, 7))

# Create a multiple scatter diagram
sns.regplot(ax = axis[0, 0], data = total_data, x = "Fare", y = "Survived")
sns.heatmap(total_data[["Survived", "Fare"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)
sns.regplot(ax = axis[0, 1], data = total_data, x = "Age", y = "Survived").set(ylabel=None)
sns.heatmap(total_data[["Survived", "Age"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()






fig, axis = plt.subplots(2, 1, figsize = (5, 7))

# Create a multiple scatter diagram
sns.regplot(ax = axis[0], data = total_data, x = "Age", y = "Fare")
sns.heatmap(total_data[["Fare", "Age"]].corr(), annot = True, fmt = ".2f", ax = axis[1])

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()




fig, axis = plt.subplots(2, 3, figsize = (15, 7))

sns.countplot(ax = axis[0, 0], data = total_data, x = "Sex", hue = "Survived")
sns.countplot(ax = axis[0, 1], data = total_data, x = "Pclass", hue = "Survived").set(ylabel = None)
sns.countplot(ax = axis[0, 2], data = total_data, x = "Embarked", hue = "Survived").set(ylabel = None)
sns.countplot(ax = axis[1, 0], data = total_data, x = "SibSp", hue = "Survived")
sns.countplot(ax = axis[1, 1], data = total_data, x = "Parch", hue = "Survived").set(ylabel = None)

plt.tight_layout()
fig.delaxes(axis[1, 2])

plt.show()





fig, axis = plt.subplots(figsize = (10, 5), ncols = 2)

sns.barplot(ax = axis[0], data = total_data, x = "Sex", y = "Survived", hue = "Pclass")
sns.barplot(ax = axis[1], data = total_data, x = "Embarked", y = "Survived", hue = "Pclass").set(ylabel = None)

plt.tight_layout()

plt.show()



___

Correlation analysis
The goal of correlation analysis with categorical-categorical data is to uncover patterns and dependencies between variables, aiding in understanding how they interact within a dataset. This analysis is fundamental in various fields including social sciences, marketing research, and epidemiology, where categorical data often represent key attributes of interest.

This analysis aims to determine whether and how the categories of one variable are related to the categories of another.


total_data["Sex_n"] = pd.factorize(total_data["Sex"])[0]
total_data["Embarked_n"] = pd.factorize(total_data["Embarked"])[0]

fig, axis = plt.subplots(figsize = (10, 6))

sns.heatmap(total_data[["Sex_n", "Pclass", "Embarked_n", "SibSp", "Parch", "Survived"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()



______

Numerical-categorical analysis (complete)
This is the most detailed analysis we can carry out. To do this, we simply have to calculate the correlations between the variables, since this is the best indication of the relationships. Thus, once we have verified that there is a relationship, we can go deeper into the study. Another element that can be very helpful is to obtain the two-by-two relationships between all the data in the dataset. This is, in part, redundant because there are many things that we have already calculated, so it is optional.


fig, axis = plt.subplots(figsize = (10, 7))

sns.heatmap(total_data[["Age", "Fare", "Sex_n", "Pclass", "Embarked_n", "SibSp", "Parch", "Survived"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()


In the first graph, we see that as age 
increases, the presence of first class 
tickets becomes more noticeable, and as 
age decreases, third class tickets become 
more present, reinforcing the negative 
relationship between the observed variables. 
The second graph also reinforces what was
 observed, as better class tickets should 
 be more expensive.



fig, axis = plt.subplots(figsize = (10, 5), ncols = 2)

sns.regplot(ax = axis[0], data = total_data, x = "Age", y = "Pclass")
sns.regplot(ax = axis[1], data = total_data, x = "Fare", y = "Pclass").set(ylabel = None, ylim = (0.9, 3.1))

plt.tight_layout()

plt.show()



Once the correlation has been calculated, 
we can draw the pairplot (this is an optional 
                          step):
    
    
sns.pairplot(data = total_data)
    
    
    
    
    