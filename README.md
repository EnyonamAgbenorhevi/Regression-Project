# Regression-Project
# CONTEXT
In this project, we aim to perform time series forecasting to predict store sales by utilizing data from Corporation Favorita, a prominent grocery retailer in Ecuador. The primary objective is to build a highly accurate model capable of predicting unit sales for different items across various Favorita stores.
# PROCEDURE
This document provides a comprehensive and detailed account of the step-by-step procedures followed to successfully achieve the project's goals at each stage. The following steps were carefully executed in order to fulfill the project's objectives with precision.
# STEPS
1. Data collection: we gathered time series sales data from multiple sources, including a SQL Server database containing three main tables (table1, table2, and table3). Additionally, we accessed relevant information from CSV files stored in designated zip files and OneDrive. The dataset comprises essential details such as store_nbr, family, sales, onpromotion, test.csv, transaction.csv, sample_submission.csv, stores.csv, oil.csv, and holidays_events.csv. These diverse data sources provide valuable information required for our analysis and forecasting tasks.

   
2. Data Loading: After collecting the data, we proceed to load it into our code and perform necessary transformations to make it suitable for analysis. To connect to the SQL Server database and retrieve data from the specified table, we utilize the pyodbc package. Simultaneously, the pandas library is employed to read data from the CSV files. By combining the data obtained from both SQL and CSV sources, we create a unified and comprehensive dataset that will be used for further analysis and modeling.

 
3. Data Evaluation (EDA): In order to gain valuable insights from the dataset, we conduct exploratory data analysis. This involves several crucial steps, such as summarizing the data, identifying and handling duplicates, and addressing any missing values. Additionally, we leverage visual analyses using tools like SARIMA (Seasonal Autoregressive Integrated Moving Average) and ADF (Augmented Dickey-Fuller) tests to detect patterns and trends within the dataset.
To perform these tasks, we make use of essential libraries like pandas and numpy for data manipulation and organization. For data visualization, we rely on matplotlib and seaborn, which provide powerful and intuitive plotting capabilities.
By undertaking exploratory data analysis with these tools, we can uncover key patterns and valuable information from the dataset, ultimately enabling us to make informed decisions and develop a successful forecasting model for our project.


4. Data Processing and Engineering: The dataset undergoes a series of data processing steps to ensure its cleanliness and suitability for further analysis. This includes handling missing values to ensure that no critical information is omitted. Additionally, categorical variables are transformed into a suitable format for analysis, and new features may be generated to enhance the dataset's predictive power.To achieve these data processing tasks, we leverage techniques available in the pandas library. Through these operations, we aim to create a refined and well-structured dataset, which will serve as the foundation for our subsequent analysis and modeling efforts.
   
5. Answering Questions with Visualizations: To explore and answer essential inquiries related to time series data, we employ informative visualizations. Leveraging the powerful capabilities of the matplotlib and seaborn libraries, we generate meaningful plots and charts that effectively illustrate the relationships between variables and the time series data.These visualizations play a crucial role in uncovering patterns, trends, and correlations within the dataset. They provide valuable insights and aid in making data-driven decisions during the analysis and modeling stages. By presenting the information in a visually appealing manner, we enhance our understanding of the time series data and ensure its effective communication to stakeholders and team members.
   
6. Train and Evaluate Models: This project involves training and evaluating various machine learning models, namely ARIMA, SARIMA, Decision Tree, Random Forest, XGBoost Regressor, and CatBoost Regressor. To assess the performance of these models, we employ evaluation metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Root Mean Log Squared Error (RMLSE). These metrics provide valuable insights into the accuracy and effectiveness of each model in predicting time series sales data.By comparing the results from different models and datasets, we can identify the most suitable approach for our time series forecasting task, ensuring the development of a robust and reliable predictive model.

7. Evaluate Chosen Model: To further enhance the performance of selected models, we utilize GridSearchCV for hyperparameter tuning. This process helps us find the best combination of hyperparameters that optimizes the models' performance. Once the model is fine-tuned, we use it to make predictions on the time series sales data. This refined model is expected to provide more accurate and reliable predictions, thereby improving the overall forecasting accuracy of the project. The process of hyperparameter tuning through GridSearchCV allows us to maximize the potential of the selected model and deliver more precise results for our time series forecasting task.

8. Future Predictions: After training and validating the time series model, it becomes capable of making predictions on new and unseen data. This valuable capability empowers businesses to forecast various time-dependent outcomes and take proactive measures based on these predictions.By deploying the model in production, it can continuously monitor incoming data and predict future events or trends. This real-time forecasting ability allows businesses to stay ahead of changes and make informed decisions to optimize their operations and strategies.The predictive power of the model serves as a valuable tool for businesses to anticipate market demands, manage inventory, plan resources efficiently, and make well-informed decisions to achieve their goals. With the model in place, businesses gain a competitive edge and can respond quickly to changing market conditions, ensuring long-term success and growth.



# Installation
pyodbc
 catboost
 python-dotenv
 pandas
 numpy
 matplotlib
 seaborn
 scipy
 pmdarima

# Packages
import pyodbc
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 from statsmodels.tsa.seasonal import seasonal_decompose
 from statsmodels.tsa.stattools import adfuller
 from sklearn.model_selection import train_test_split
 from dotenv import dotenv_values
 from scipy import stats
 from statsmodels.tsa.arima.model import ARIMA
 from statsmodels.tsa.statespace.sarimax import SARIMAX
 from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
 from xgboost import XGBRegressor
 from pmdarima.arima import auto_arima
 from sklearn.model_selection import GridSearch
 from catboost import CatBoostRegressor
 from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
 import warnings
 warnings.filterwarnings("ignore")

 ## Authors and Acknowledgement
 The table provided presents the initial contributors of the project, along with their corresponding Github IDs and the articles they have written to document their unique perspectives on the project.

## Conclusion
In summary, this project revolves around addressing a time series forecasting challenge. Through the application of time-dependent data and advanced modeling techniques, we have achieved accurate predictions and gained valuable insights from the temporal patterns present in the dataset. The utilization of time series analysis empowers us to make well-informed decisions and strategically plan for the future, setting the stage for successful outcomes and effective foresight in various domains.

## License
 The MIT-LICENSE.txt is a popular open-source software license utilized for the distribution and sharing of software, code, and various creative works. It provides a permissive and flexible approach to licensing, making it widely adopted in the open-source community.


