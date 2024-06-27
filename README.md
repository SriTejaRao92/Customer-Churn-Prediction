# Customer Churn Prediction

This project predicts customer churn using historical data.

## Project Description
The goal of this project is to predict whether a customer will churn (i.e., leave the service) based on historical data. Various machine learning models are used to build the predictive model.

## Project Structure
- `data/`: Contains the raw data file `telco_customer_churn.csv`.
- `notebooks/`: Contains the Jupyter Notebook `churn_analysis.ipynb` with the complete analysis.
- `src/`: Contains Python scripts for data preprocessing, exploratory data analysis, feature engineering, and modeling.
- `README.md`: Project documentation.
- `requirements.txt`: List of dependencies.

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer_churn_prediction.git
   cd customer_churn_prediction
2. Install dependencies:
   pip install -r requirements.txt
   
## Usage
1. Run the data preprocessing script:
   python src/data_preprocessing.py
2. Run the exploratory data analysis script:
   python src/eda.py
3. Run the feature engineering script:
   python src/feature_engineering.py
4. Run the modeling script:
   python src/modeling.py
   
## Features
1.Data Preprocessing: Handling missing values, encoding categorical variables, normalizing numerical features.
2.Exploratory Data Analysis: Visualizations of data distributions and correlations.
3.Feature Engineering: Creating new features such as tenure categories.
4.Modeling: Building and evaluating machine learning models (Logistic Regression, Random Forest, XGBoost).

## Results and Visualizations
Logistic Regression: (0.795260663507109, 0.6375266524520256, 0.5329768270944741, 0.5805825242718446, 0.7116143012167012)
Random Forest: (0.785781990521327, 0.6252873563218391, 0.48484848484848486, 0.5461847389558233, 0.6898096523661404)
XGBoost: (0.7691943127962085, 0.5734126984126984, 0.5151515151515151, 0.5427230046948357, 0.6881761449224328)

## Contributing
Contributions are welcome! Please create a pull request or open an issue for any improvements or bug fixes.

## License
This project is licensed under the MIT License.

## Contact Information
For any questions or feedback, please contact SriTejaRao Gade at sritejarao1@gmail.com
