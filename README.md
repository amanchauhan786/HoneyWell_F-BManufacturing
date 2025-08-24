# 🍺 Predictive Quality Control for F&B Manufacturing
### A Brewery Case Study for the Honeywell Campus Connect Hackathon

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fbmanufacturing.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-orange)
![XAI](https://img.shields.io/badge/Explainable-AI-green)

This project is an end-to-end machine learning system designed to predict quality anomalies in a brewery's manufacturing process in real-time. It uses a **Random Forest model** to provide actionable Quality Alerts and is demonstrated through a fully **interactive web dashboard** built with Streamlit.

## 🚀 Live Dashboard Demo

You can access and interact with the live dashboard here:  
**👉 [https://fbmanufacturing.streamlit.app/](https://fbmanufacturing.streamlit.app/)**

## 📋 Project Overview

This project tackles the critical challenge of maintaining consistent product quality in the Food & Beverage industry. By analyzing in-process sensor data, our system can identify potential quality issues **before a batch is finished**, allowing for proactive intervention.

### Key Features:

- **🔍 Data-Driven Anomaly Detection**: Classifies each production batch as either "Normal" or "Anomaly" based on its process parameters
- **📊 Model Bake-Off**: Comparative analysis between four models (Logistic Regression, Random Forest, XGBoost, LightGBM) to select the best performer
- **🤖 Explainable AI (XAI)**: Uses SHAP (SHapley Additive exPlanations) to explain why the model made a specific prediction
- **🎛️ Interactive Dashboard**: User-friendly web interface to explore predictions and understand the underlying data

## 🛠️ Tech Stack

- **Language**: Python
- **Data Analysis & Modeling**: Pandas, Scikit-learn, Imbalanced-learn (SMOTE), XGBoost, LightGBM
- **Explainability**: SHAP
- **Dashboard & Deployment**: Streamlit, ngrok
- **Development Environment**: Google Colab, PyCharm / VS Code

## ⚙️ Setup and Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amanchauhan786/HoneyWell_F-BManufacturing.git
   cd HoneyWell_F-BManufacturing
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ How to Run the Dashboard

Once the setup is complete, you can launch the Streamlit dashboard from your terminal:

```bash
streamlit run app/streamlit_app.py
```

Your web browser will open a new tab with the interactive application.

## 📁 Project Structure

```
HoneyWell_F-BManufacturing/
├── app/
│   └── streamlit_app.py          # Main dashboard application
├── data/
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed data
├── models/                       # Trained model files
├── notebooks/                    # Jupyter/Colab notebooks
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## 🔗 Important Links & Resources

- **📊 Live Dashboard**: [https://fbmanufacturing.streamlit.app/](https://fbmanufacturing.streamlit.app/)
- **📓 Google Colab Notebook**: [https://colab.research.google.com/drive/19Gnsta9qQuZd3zjNXV8C9FBAsfNoOFbX](https://colab.research.google.com/drive/19Gnsta9qQuZd3zjNXV8C9FBAsfNoOFbX)
- **🎥 Video Demo**: [https://drive.google.com/file/d/1YkbhyUPzWdRO7ZKV9DZL-raHcVcF53GZ/view](https://drive.google.com/file/d/1YkbhyUPzWdRO7ZKV9DZL-raHcVcF53GZ/view)
- **📦 Primary Dataset**: [https://www.kaggle.com/datasets/aqmarh11/brewery-operations-and-market-analysis-dataset](https://www.kaggle.com/datasets/aqmarh11/brewery-operations-and-market-analysis-dataset)

## 👨‍💻 Author

**Aman Chauhan**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/aman-chauhan-128552256/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/amanchauhan786)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Honeywell for organizing the Campus Connect Hackathon
- Kaggle community for providing the brewery operations dataset
- Streamlit team for the excellent dashboard framework
- SHAP developers for the model explainability tools

---

⭐ **Star this repo if you found it helpful!** ⭐
