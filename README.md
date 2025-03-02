# 🩺 Lung Cancer Prediction  

This project is a **Machine Learning-based Lung Cancer Prediction App** built using **Streamlit**. It predicts lung cancer risk based on user inputs and visualizes probability distributions.  

## 🚀 Features  
✅ **User Input via Sidebar** (Sliders & Dropdowns)  
✅ **CSV File Upload** for bulk predictions  
✅ **Prediction Output with Probability Graph**  
✅ **Comparison of Machine Learning Models**  
✅ **Hyperparameter Tuning for Best Accuracy**  

## 📂 Dataset  
- **Source:** Kaggle  
- **Format:** CSV  
- **Preprocessing:** Handled missing values & class imbalance using **SMOTE**  

## 🛠️ Tech Stack  
- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)  
- **Machine Learning Models:**  
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Random Forest  
  - Gradient Boosting  
  - AdaBoost  
- **Streamlit** for UI  

## 🏗️ Model Training Workflow  
1. **Data Preprocessing & EDA:** Cleaned data, applied feature selection & SMOTE  
2. **Model Building:** Trained multiple models on preprocessed data  
3. **Comparison of Models:** Evaluated based on accuracy & training time  
4. **Hyperparameter Tuning:** Used **GridSearchCV** on top 3 models  
5. **Final Model Selection:** Picked **Logistic Regression** & saved it using Pickle  
6. **Converted Jupyter Code to .py Script**  
7. **Built Streamlit App for Deployment**  

## ⏳ Model Performance  
### **Before Hyperparameter Tuning**  
| Model | Accuracy | Training Time (seconds) |  
|--------|------------|-------------------|  
| **Logistic Regression** | 92.86% | 0.0042 |  
| **KNN** | 94.64% | 0.0010 |  
| **Random Forest** | 89.29% | 0.0543 |  
| **Gradient Boosting** | 89.29% | 0.0451 |  
| **AdaBoost** | 92.86% | 0.0303 |  

### **After Hyperparameter Tuning**  
| Model | Best Accuracy |  
|--------|--------------|  
| **Logistic Regression** | **95.63%** |  
| **KNN** | **95.11%** |  
| **AdaBoost** | **95.37%** |  

## 📊 Visualizations  
- **Probability Bar Graph** (Lung cancer risk %)  

## 📁 Project Structure  
```
Lung_Cancer_Prediction
│── 📁 venv                      # Virtual environment  
│── 📁 data_preprocessing_EDA    # Data cleaning & feature selection  
│── 📁 model_building            # Training ML models  
│── 📁 model_comparison          # Accuracy & time-based evaluation  
│── 📁 hyperparameter_tuning     # Optimizing top 3 models  
│── 📁 final_accuracy            # Best model results  
│── 📁 jupyter_to_py             # Converted Jupyter code to .py   
│── 📜 app.py                    # Streamlit app for predictions and Streamlit UI script  
│── 📜 requirements.txt          # Required libraries  
│── 📜 README.md                 # Project documentation  
│── 📁 data
│   ├── lung_cancer_data.csv     # Dataset used  
│── 📁 models
│   ├── logistic_regression.pkl  # Saved ML model  
│── 📁 images
│   ├── prediction_bar_chart.png # Probability graph  
```

## 🔮 Future Enhancements  
✅ Upgrade ML models to **Deep Learning (CNN, LSTMs, etc.)**  
✅ Integrate **Image Processing** for X-ray scans  
✅ Use **ChatGPT or Google API** for health recommendations  

## ▶️ How to Run  
1. Clone the repo:  
   ```bash
   https://github.com/sprashasya/Lung_cancer_streamlit.git
   ```  
2. Create a virtual environment:  
   ```bash
   python -m venv venv  
   source venv/bin/activate  # Mac/Linux  
   venv\Scripts\activate  # Windows  
   ```  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt  
   ```  
4. Run the app:  
   ```bash
   streamlit run app.py  
   ```  
