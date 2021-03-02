FROM python:3.7

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY cefr_predictor/models/xgboost.joblib cefr_predictor/models/
COPY cefr_predictor/preprocessing.py cefr_predictor/inference.py cefr_predictor/
COPY CEFR_Predictor.py .

EXPOSE 8080

CMD streamlit run --server.port 8080 CEFR_Predictor.py