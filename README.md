# Traffic Congestion Hotspot Prediction

## 1. Project Overview

Traffic Congestion Hotspot Prediction is a Machine Learning-based project designed to analyze traffic congestion patterns and identify highly congested areas and roads in Bangalore.

The project uses historical traffic data to classify congestion into three categories:

- Low
- Medium
- High

## 2. Problem Statement

Traffic congestion is a major problem in urban areas like Bangalore. It causes travel delays, increases fuel consumption, and affects daily transportation.

The objective of this project is to analyze historical traffic information, identify congestion hotspots, and develop a Machine Learning-based congestion classification system.

## 3. Project Objectives

- Analyze Bangalore traffic data.
- Perform data cleaning and preprocessing.
- Explore traffic patterns using EDA.
- Implement multiple Machine Learning algorithms.
- Compare model performance.
- Identify highly congested roads and areas.
- Develop an interactive prediction application.

## 4. Dataset Information

Dataset: Bangalore Traffic Dataset

Source: Kaggle

The dataset contains traffic-related information such as location, road details, weather conditions, roadwork activities, and congestion information.

Dataset download instructions will be provided in this repository.

## 5. Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Git and GitHub

## 6. Machine Learning Algorithms

Four Machine Learning algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. XGBoost Classifier

## 7. Model Evaluation

The algorithms were evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Confusion Matrix

### Best Performing Model

Logistic Regression

Accuracy: 80.96%

Macro F1-score: 73.70%

These are the results obtained during the current experiment.

## 8. Project Structure

```text
Traffic_Congestion_Hotspot_Prediction/
│
├── app.py
├── Traffic_Congestion.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── best_traffic_model.pkl
│   ├── traffic_preprocessor.pkl
│   ├── label_encoder.pkl
│   └── feature_columns.pkl
│
├── results/
│   ├── graphs/
│   └── hotspots/
│
└── data/
```

## 9. Installation and Execution

### Step 1: Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### Step 2: Open Project Folder

```bash
cd Traffic_Congestion_Hotspot_Prediction
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

Download the dataset from the Kaggle source mentioned in this repository and place it in the project directory using the filename expected by the application.

### Step 5: Run Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 10. Project Results

The project includes:

- Traffic congestion analysis.
- Area-wise hotspot identification.
- Road-wise hotspot ranking.
- Machine Learning model comparison.
- Interactive congestion classification application.

## 11. Future Improvements

- Real-time traffic data integration.
- Geographical hotspot visualization.
- Future traffic congestion forecasting.
- Integration with live traffic APIs.

## 12. Author

Developed as an academic Machine Learning project.