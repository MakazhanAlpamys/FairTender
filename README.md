# Suspicious Government Procurement Analysis System

## Project Description

This system is an interpretable machine learning model designed to detect suspicious government procurements based on various transaction and supplier characteristics. The project includes both model training scripts and a Streamlit web application for convenient procurement analysis through a user interface.

## System Requirements

### Python Versions:
- **Recommended version:** Python 3.8-3.10
- **Not supported:** Python 3.11+ (due to incompatibility with some dependencies)

### Operating Systems:
- Windows
- Linux
- macOS

## System Features

- **Base Model:** Training and saving a model for detecting suspicious procurements
- **Web Interface:** User interface based on Streamlit
- **Bulk Analysis:** Upload CSV files with procurement data for mass analysis
- **Manual Check:** Check individual procurements by entering parameters with instant results
- **Visualizations:** Charts showing feature importance, model metrics, and result distribution
- **Average Statistics:** Comparison of average indicators for suspicious and normal procurements
- **Interactive Regional Map:** Visualization of suspicious procurement distribution across Kazakhstan regions with the ability to analyze various metrics

## Project Structure

- `suspicious_purchases_model.py` - main model script
- `app_streamlit.py` - Streamlit web application
- `analyze_results.py` - script for analyzing model results
- `final_training_data.csv` - training dataset
- `final_test_data_user_input.csv` - test dataset for predictions
- `predictions.csv` - file with prediction results
- `model.pkl` - saved model
- `encoders.pkl` - saved encoders for categorical variables
- `confusion_matrix.png` - visualization of model confusion matrix
- `feature_importance.png` - visualization of feature importance by XGBoost
- `shap_importance.png` - visualization of SHAP feature importance
- `shap_summary.png` - visualization of SHAP feature influence distribution
- `probability_distribution.png` - visualization of prediction probability distribution
- `requirements.txt` - project dependencies file
- `uploads/` - directory for user-uploaded files
- `static/` - directory for static resources (logo, etc.)

## Features Used

1. **Procurement Characteristics:**
   - `category` - procurement category
   - `region` - procurement region
   - `price` - total contract value
   - `avg_price_category_region` - average contract value in the category and region
   - `price_per_unit` - price per unit of goods/services
   - `days_to_tender` - number of days until tender

2. **Supplier Characteristics:**
   - `supplier_id` - supplier identifier
   - `supplier_name` - supplier name
   - `supplier_win_count` - number of contracts won by supplier
   - `supplier_years_active` - number of years supplier has been active
   - `supplier_total_contracts` - total number of supplier contracts
   - `supplier_avg_contract_value` - average value of supplier contracts

## Target Variable

- `is_suspicious` - indicator of procurement suspiciousness (1 - suspicious, 0 - not suspicious)

## Technologies Used

- **Streamlit:** Framework for creating web applications for machine learning and data analysis
- **XGBoost:** Machine learning algorithm for classification
- **SHAP:** Library for explaining machine learning model predictions
- **Pandas, NumPy:** Libraries for data processing
- **Matplotlib, Seaborn:** Libraries for data visualization
- **Scikit-learn:** Machine learning library

## Web Interface Features

### 1. Home Page
Provides general project information, displays model metrics, and describes features used.

### 2. File Upload
Upload CSV files for bulk procurement analysis. Displays statistics, identified suspicious procurements, and probability distribution.

### 3. Manual Check
Interface for entering individual procurement parameters with instant analysis results and prediction explanation using SHAP.

### 4. Visualizations
Various charts and metrics to evaluate model performance:
- Model quality metrics
- Feature importance (SHAP)
- Prediction probability distribution

### 5. Regional Map
Interactive map of Kazakhstan visualizing suspicious procurements by region:
- Display of various metrics (count, percentage, sum of suspicious procurements)
- Suspiciousness heat map
- Table with detailed regional data
- Comparative regional charts

## Installation and Setup

### 1. Install Python 3.10 (recommended)

Download and install Python 3.10 from the [official website](https://www.python.org/downloads/release/python-3108/).

For Windows:
- Select "Windows installer (64-bit)"
- Check "Add Python to PATH" during installation

### 2. Create a Virtual Environment

#### For Windows:

```bash
# Check available Python versions:
py -0

# Create virtual environment with Python 3.10:
py -3.10 -m venv venv_py310

# Activate virtual environment:
venv_py310\Scripts\activate
```

#### For Linux/macOS:

```bash
# Create virtual environment:
python3.10 -m venv venv_py310

# Activate virtual environment:
source venv_py310/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required dependencies:
pip install -r requirements.txt
```

### 4. Run the Project

#### Run Streamlit Web Application:

```bash
streamlit run app_streamlit.py
```
After launching the web application, open your web browser and navigate to: http://localhost:8501

#### Run Base Model for Training:

```bash
python suspicious_purchases_model.py
```

#### Run Results Analysis:

```bash
python analyze_results.py
```

### Alternative Installation for Python 3.11+

If you have Python 3.11 or newer installed, you can update the requirements.txt file with the following versions (however, full functionality is not guaranteed):

```
streamlit==1.28.0
pandas==2.1.0
numpy==1.24.0
matplotlib==3.7.3
seaborn==0.12.2
scikit-learn==1.3.0
xgboost==2.0.0
shap==0.43.0
plotly==5.18.0
pillow==10.0.0
joblib==1.3.2
scipy==1.11.3
```

## Model Metrics

- **Accuracy:** 96.1%
- **Precision:** 98.5%
- **Recall:** 77.4%
- **F1-score:** 86.7%

## Input Data Format

The system accepts CSV files with the following columns:
- `category`: Procurement category
- `region`: Procurement region
- `price`: Contract value
- `avg_price_category_region`: Average value by category and region
- `supplier_id`: Supplier identifier
- `supplier_name`: Supplier company name
- `supplier_win_count`: Number of contracts won
- `days_to_tender`: Days until tender
- `price_per_unit`: Price per unit
- `supplier_years_active`: Supplier years of activity
- `supplier_total_contracts`: Total number of contracts
- `supplier_avg_contract_value`: Average contract value

## Results and Interpretation

After running the model, the following will be created:
- `predictions.csv` file with predictions on the test dataset
- Visualizations for analyzing model performance

The model uses XGBoost with SHAP (SHapley Additive exPlanations) to ensure interpretability of results. SHAP allows understanding which features are most important for deciding procurement suspiciousness and how specific feature values influence model predictions.

## Troubleshooting

### Dependency Installation Issues:
- If you have Python 3.11+, follow the alternative installation instructions
- If errors occur with SHAP or XGBoost, ensure you're using Python 3.10 or lower
- NumPy issues often arise from Python version incompatibility

### Streamlit Launch Issues:
- Ensure all dependencies are installed correctly
- Verify the correct virtual environment is activated

## Notes

The project is developed for Python 3.8-3.10 and uses compatible versions of all libraries. If installation or launch issues occur, it's recommended to create a new virtual environment with Python 3.10.
