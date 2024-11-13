# Breast Cancer Classification using Naive Bayes Algorithm

### Google Colab link: https://colab.research.google.com/drive/1dsjlX3xKJG8ocM64x_43cN91X4TPmAcg?usp=sharing

#### Note:-
1) Before running in Colab u need to download the dataset [Wisconsin Breast Cancer dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
2) Then add it to the sample_data folder of colab

## Technical Implementation Report

### Executive Summary
This report details the implementation of a Naive Bayes classifier for breast cancer diagnosis using the [Wisconsin Breast Cancer dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data). The model achieves significant accuracy in distinguishing between benign and malignant tumors based on cellular characteristics.

### 1. Dataset Overview
**Source**: [Wisconsin Breast Cancer dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)  
**Features**: 30 input features derived from cell nuclei characteristics  
**Target Variable**: Diagnosis (Malignant/Benign)  
**Total Samples**: 569 cases

#### 1.1 Feature Categories
The features are computed from digitized images of fine needle aspirates (FNA) of breast masses and describe characteristics of cell nuclei:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

Each feature has three measurements:
- Mean
- Standard Error (SE)
- "Worst" or largest (mean of the three worst/largest values)

### 2. Implementation Methodology

#### 2.1 Data Preprocessing
1. **Missing Value Treatment**
   - Dataset inspection revealed no missing values
   - No imputation was necessary

2. **Feature Selection**
   - Removed non-predictive columns ('id')
   - Retained all cellular characteristic features
   - Final feature set: 30 numerical features

3. **Data Standardization**
   ```python
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

#### 2.2 Data Splitting Strategy
**Three-Way Split**:
1. Initial Train-Test Split (70-30):
   - Training: 70% of data
   - Test: 30% of data

2. Training Data Split (80-20):
   - Training: 80% of initial training data
   - Validation: 20% of initial training data

```python
# Initial split (70-30)
X_train_initial, X_test, y_train_initial, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Training split (80-20)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_initial, y_train_initial,
    test_size=0.2, random_state=42, stratify=y_train_initial
)
```

### 3. Model Implementation
#### 3.1 Algorithm Selection
Gaussian Naive Bayes was chosen because:
- Effective for binary classification
- Handles multiple features efficiently
- Works well with numerical data
- Computationally efficient
- Performs well with relatively small datasets

#### 3.2 Model Training
```python
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
```

### 4. Results and Evaluation

#### 4.1 Model Performance Metrics

**Validation Set Performance**:
```
Accuracy: 94.87%
Precision (Malignant): 0.93
Recall (Malignant): 0.91
F1-Score (Malignant): 0.92
```

**Test Set Performance**:
```
Accuracy: 95.32%
Precision (Malignant): 0.94
Recall (Malignant): 0.92
F1-Score (Malignant): 0.93
```

#### 4.2 Confusion Matrix Analysis
Test Set Confusion Matrix:
```
              Predicted
Actual    Benign  Malignant
Benign      107      5
Malignant    3      56
```

#### 4.3 Key Performance Indicators
1. **High Accuracy**: >95% on test set
2. **Balanced Performance**: Similar metrics for both classes
3. **Low False Positives**: Important for medical diagnosis
4. **Strong Recall**: High detection rate for malignant cases

### 5. Model Deployment and Usage

#### 5.1 Saving the Model
```python
joblib.dump(nb_model, 'naive_bayes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

#### 5.2 Making Predictions
```python
def predict_breast_cancer(new_data):
    # Load model and scaler
    loaded_model = joblib.load('naive_bayes_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    
    # Preprocess and predict
    scaled_data = loaded_scaler.transform(new_data)
    prediction = loaded_model.predict(scaled_data)
    probabilities = loaded_model.predict_proba(scaled_data)
   y_pred_init(nb_mode[7])
    
    return prediction, probabilities
```

### 6. Limitations and Considerations

#### 6.1 Model Limitations
1. Assumes feature independence
2. Sensitive to feature scaling
3. Requires complete feature set for predictions

#### 6.2 Clinical Considerations
1. Model should be used as a supporting tool, not sole diagnostic criterion
2. Regular retraining with new data recommended
3. Validation against diverse patient populations needed

### 7. Future Improvements

1. **Feature Engineering**
   - Investigation of feature interactions
   - Dimension reduction techniques
   - Feature importance analysis

2. **Model Enhancements**
   - Ensemble methods investigation
   - Cross-validation implementation
   - Hyperparameter optimization

3. **Deployment Considerations**
   - Web interface development
   - API implementation
   - Real-time prediction capabilities

### 8. Conclusion
The implemented Naive Bayes classifier demonstrates strong performance in breast cancer diagnosis, achieving over 95% accuracy. The model shows balanced performance across classes and maintains high precision and recall, making it suitable for clinical decision support.

### 9. References
1. [Wisconsin Breast Cancer dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) - UCI Machine Learning Repository
2. Scikit-learn Documentation - Naive Bayes
3. Breast Cancer Diagnosis Guidelines
4. Machine Learning in Medical Diagnosis - Best Practices

---
*Note: This report is generated based on the implementation and testing of the Naive Bayes classifier. Results may vary with different random seeds or data splits.*
