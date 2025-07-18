import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle

# Load the saved XGBoost model
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Expresso Telecom Customer Churn Prediction")
st.markdown("Enter customer details to predict if they are likely to churn.")

# User input features
def user_input_features():
    REGION = st.selectbox('REGION', ["DAKAR", "THIES", "DIOURBEL", "KAOLACK", "SAINT-LOUIS", "FATICK", "TAMBACOUNDA", "ZIGUINCHOR", "LOUGA"])
    TENURE = st.selectbox('TENURE', ["1-2 year", "2-3 years", "3-4 years", "4-5 years", "> 5 years"])
    MONTANT = st.number_input('MONTANT', min_value=0.0)
    FREQUENCE_RECH = st.number_input('FREQUENCE_RECH', min_value=0.0)
    REVENUE = st.number_input('REVENUE', min_value=0.0)
    ARPU_SEGMENT = st.number_input('ARPU_SEGMENT', min_value=0.0)
    FREQUENCE = st.number_input('FREQUENCE', min_value=0.0)
    DATA_VOLUME = st.number_input('DATA_VOLUME', min_value=0.0)
    ON_NET = st.number_input('ON_NET', min_value=0.0)
    ORANGE = st.number_input('ORANGE', min_value=0.0)
    MRG = st.selectbox('MRG', ["OUI", "NON"])
    REGULARITY = st.slider('REGULARITY', 0, 30, 10)
    TOP_PACK = st.selectbox('TOP_PACK', ["TOP_PACK_1", "TOP_PACK_2", "TOP_PACK_3", "TOP_PACK_4", "TOP_PACK_5"])
    FREQ_TOP_PACK = st.number_input('FREQ_TOP_PACK', min_value=0.0)

    data = {
        'REGION': REGION,
        'TENURE': TENURE,
        'MONTANT': MONTANT,
        'FREQUENCE_RECH': FREQUENCE_RECH,
        'REVENUE': REVENUE,
        'ARPU_SEGMENT': ARPU_SEGMENT,
        'FREQUENCE': FREQUENCE,
        'DATA_VOLUME': DATA_VOLUME,
        'ON_NET': ON_NET,
        'ORANGE': ORANGE,
        'MRG': MRG,
        'REGULARITY': REGULARITY,
        'TOP_PACK': TOP_PACK,
        'FREQ_TOP_PACK': FREQ_TOP_PACK
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# One-hot encode like the training
input_encoded = pd.get_dummies(input_df)

# Align with training columns
model_features = model.get_booster().feature_names
missing_cols = set(model_features) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0
input_encoded = input_encoded[model_features]

# Prediction
if st.button('Predict Churn'):
    prediction = model.predict(input_encoded)
    result = 'Churn' if prediction[0] == 1 else 'Not Churn'
    st.subheader(f'Prediction: {result}')
