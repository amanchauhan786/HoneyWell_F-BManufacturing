# HoneyWell_F-BManufacturing
Predictive Quality Control for F&B Manufacturing
A Brewery Case Study for the Honeywell Campus Connect Hackathon
This project is an end-to-end machine learning system designed to predict quality anomalies in a brewery's manufacturing process in real-time. It uses a Random Forest model to provide an actionable Quality Alert and is demonstrated through a fully interactive web dashboard built with Streamlit.

🚀 Live Dashboard Demo
You can access and interact with the live dashboard here:
https://fbmanufacturing.streamlit.app/

(Insert a screenshot or GIF of your running Streamlit dashboard here)
Note: You will need to upload this screenshot to an img folder in your repository and update the link.

📋 Project Overview
This project tackles the critical challenge of maintaining consistent product quality in the Food & Beverage industry. By analyzing in-process sensor data, our system can identify potential quality issues before a batch is finished, allowing for proactive intervention.

Key Features:
Data-Driven Anomaly Detection: Classifies each production batch as either "Normal" or "Anomaly" based on its process parameters.

Model Bake-Off: A comparative analysis was conducted between four models (Logistic Regression, Random Forest, XGBoost, LightGBM) to select the best performer.

Explainable AI (XAI): Uses SHAP (SHapley Additive exPlanations) to explain why the model made a specific prediction, turning it from a "black box" into a practical decision-support tool.

Interactive Dashboard: A user-friendly web interface to explore predictions and understand the underlying data.

🛠️ Tech Stack
Language: Python

Data Analysis & Modeling: Pandas, Scikit-learn, Imbalanced-learn (SMOTE), XGBoost, LightGBM

Explainability: SHAP

Dashboard & Deployment: Streamlit, ngrok

Development Environment: Google Colab, PyCharm / VS Code

⚙️ Setup and Installation
To run this project locally, follow these steps:

Clone the repository:

git clone https://github.com/amanchauhan786/HoneyWell_F-BManufacturing.git
cd HoneyWell_F-BManufacturing


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install the required libraries:

pip install -r requirements.txt


▶️ How to Run the Dashboard
Once the setup is complete, you can launch the Streamlit dashboard from your terminal:

streamlit run app/streamlit_app.py


Your web browser will open a new tab with the interactive application.

🔗 Important Links & Resources
Live Dashboard: https://fbmanufacturing.streamlit.app/

Google Colab Notebook: https://colab.research.google.com/drive/19Gnsta9qQuZd3zjNXV8C9FBAsfNoOFbX

Video Demo: https://drive.google.com/file/d/1YkbhyUPzWdRO7ZKV9DZL-raHcVcF53GZ/view

Primary Dataset: https://www.kaggle.com/datasets/aqmarh11/brewery-operations-and-market-analysis-dataset

LinkedIn Profile:

Aman Chauhan: [Your LinkedIn Profile URL]
