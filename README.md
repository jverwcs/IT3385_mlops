### IT3385 MLOps — Model Prediction App

A Streamlit web app for training and serving ML models with a simple, student-friendly MLOps workflow (data → train → evaluate → predict → monitor).

Live app: https://modelpredictionmlops.streamlit.app

#### 1. Project overview

This app lets a user:
1. load a pre-trained model from /models or train a new one,
2. run predictions (single/batch) and download results

#### 2. Folder Structure
it3385_mlops/  
├─ apps/                      # Streamlit front-end pages  
│  ├─ Homepage.py             # Main entry point in Streamlit Cloud  
│  └─ ...                     # Other pages/components  
├─ datasets/                  # Given Datasets
├─ models/                    # Saved models / artifacts  
├─ notebooks/                 # Jupyter exploration  
├─ temp/                      # Scratch / intermediate files 
├─ logs.log                   # Simple app log  
├─ requirements.txt           # Python dependencies  
└─ runtime.txt                # (Local hint) preferred Python, e.g., python-3.10.14  

#### 3. User Guide
1. Open the app: visit the live URL above.
2. Homepage: read the overview and navigation.
3. Data: input some data, or upload a CSV for batch.
4. Predict: select a dataset, run inference, preview and download predictions.
> Tip: For best results, keep your CSV headers clean (no spaces/special chars) and ensure target/feature types are correct.

#### 4. Run locally (Developer Setup)
1) Clone  
> git clone https://github.com/jverwcs/it3385_mlops  
> cd it3385_mlops
2) Create a new virtual environment with python == 3.10
3) Install requirements.txt
4) Run the app with Streamlit
> Streamlit run apps/Homepage.py
5) Deploy on Streamlit Cloud

#### 5. Troubleshooting (common errors)
1) When Deploying the app make sure the setting for python is 3.10
2) If some dependencies cannot be uploaded, try upgrading pip or change to a compatitable version

### **Disclaimer: This is a school project, please do not create any unnecessary issues.**
