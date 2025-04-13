# ml

This repository contains various machine learning projects for practicing and learning different models and techniques. The focus is on both **regression** and **classification** models, using real-world datasets and providing examples for practical applications.

---

## **Resources**  

- **[YouTube: Build Your First Machine Learning Model in Python](https://www.youtube.com/watch?v=29ZQ3TDGgRQ)**  
A tutorial on how to create your first machine learning model using Python. In this tutorial, **Linear Regression** is used.

- **[Kaggle: Datasets](https://www.kaggle.com/datasets)**  
  Kaggle provides a large collection of datasets for various machine learning tasks, including both regression and classification problems.

---

## **Model Types**

Machine learning models can be divided into two broad categories based on the nature of the target variable (`y`):

### **Regression Model (used here)**  
- **Purpose**: A regression model is used when the target variable (`y`) is **quantitative** — i.e., it represents continuous numerical values.  
  - **Example**: Predicting the price of a house, the temperature tomorrow, or a person’s salary based on various features.  
  - **Algorithms**:  
    - Linear Regression
    - Decision Trees (for regression)
    - Random Forest (for regression)
    - Support Vector Machines (SVR)
    - Neural Networks (for regression tasks)
  
- **Used in this repo**: In this project, we are using **Linear Regression** to predict continuous target values. For example, in the Boston Housing dataset, we predict the median house value (`medv`) based on features like crime rate, average number of rooms, and property tax rate.

### **Classification Model**  
- **Purpose**: A classification model is used when the target variable (`y`) is **categorical** — i.e., it consists of discrete class labels.  
  - **Example**: Predicting whether an email is spam or not (binary classification), or predicting the type of animal in a picture (multi-class classification).  
  - **Algorithms**:  
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Decision Trees (for classification)
    - Random Forest (for classification)
    - Support Vector Machines (SVM)
    - Neural Networks (for classification tasks)

- **Common Use Cases**:  
  - **Spam detection** (binary classification: spam vs. non-spam)
  - **Disease diagnosis** (binary or multi-class classification: different diseases or conditions)
  - **Image classification** (multi-class classification: different categories of objects)

---

## **Example Model: Linear Regression for Boston Housing Dataset**

In this repository, we are focusing on a **regression** problem, where the goal is to predict a continuous target value. The Boston Housing dataset is used to predict the median value of houses in various Boston suburbs based on features such as the crime rate, the average number of rooms, and the proximity to highways.

- **Target Variable (`y`)**: The median value of owner-occupied homes (`medv`), measured in **thousands of dollars**.
- **Features (`X`)**: Various factors influencing home prices, such as **crime rate (`crim`)**, **average number of rooms (`rm`)**, **property tax rate (`tax`)**, and other socioeconomic variables.
- **Year**: The data was collected in **1980**.

### Implementation
The code and full implementation of the model can be found in the [**predict_house_prices_lr.ipynb**](predict_house_prices_lr.ipynb) file in this repository.

## Comparison: scikit-learn vs PyTorch

| Feature                          | **scikit-learn**                                      | **PyTorch**                                      |
|----------------------------------|------------------------------------------------------|-------------------------------------------------|
| **Primary Use Case**             | Traditional machine learning (ML) models (e.g., linear regression, classification, clustering) | Deep learning, neural networks, custom models   |
| **Abstraction Level**            | High-level, user-friendly API for quick model building | Low-level, more flexible but requires more code |
| **Ease of Use**                  | Very easy to use, especially for standard ML tasks    | Steeper learning curve, more flexibility        |
| **Models Available**             | Linear regression, decision trees, SVMs, KNN, etc.    | Neural networks, deep learning models (CNN, RNN, etc.) |
| **GPU Support**                  | No GPU support                                       | Full GPU support with CUDA for fast computation |
| **Data Handling**                | Works well with small to medium-sized datasets       | Works well with large datasets, especially with deep learning |
| **Training Loop**                | Built-in, simple (e.g., `model.fit()`)                | You need to write the training loop yourself (more control) |
| **Performance**                   | Optimized for traditional ML tasks, not suited for large-scale deep learning | Highly optimized for training deep learning models |
| **Flexibility**                  | Less flexible (mostly for standard ML algorithms)    | Very flexible, allows custom models, layers, etc. |
| **Preprocessing Tools**          | Extensive tools for data preprocessing (scaling, encoding, etc.) | Basic data manipulation; requires custom preprocessing |
| **Deployment**                   | Easy to deploy for smaller models                    | Designed for production with deep learning models, includes tools like TorchServe |
| **Integration with Other Libraries** | Integrates well with pandas, NumPy, and matplotlib   | Works well with NumPy, pandas, and other deep learning tools like TensorBoard |


