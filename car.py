import streamlit as st

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

import joblib

#========================================
#PAGE SETUP
#========================================

st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.title("🚗 Car Price Prediction Dashboard")

#========================================

#STEP 1: LOAD DATA

#========================================

st.sidebar.header("📂 Upload or Load Data")

uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

else:

    df = pd.read_csv("car.csv")   # use your provided file

st.success("✅ Dataset loaded successfully!")

st.write("### 📋 Dataset Preview")

st.dataframe(df.head())

df.columns = df.columns.str.strip()

#========================================

#STEP 2: DATA CLEANING

#========================================

st.header("🧹 Data Cleaning")

#Clean year: numeric only, drop invalid, int type

df['year'] = pd.to_numeric(df['year'], errors='coerce')

df = df.dropna(subset=['year'])

df['year'] = df['year'].astype(int)

#'Price': Remove 'Ask for Price', commas, convert

df = df[df['Price'].str.lower() != 'ask for price']

df['Price'] = df['Price'].str.replace(',', '', regex=False).astype(int)

#'kms_driven': remove text, commas, spaces, convert

df['kms_driven'] = (

df['kms_driven']

.str.replace('kms', '', case=False, regex=False)

.str.replace(',', '', regex=False)

.str.strip()

)

df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')

df['kms_driven'] = df['kms_driven'].fillna(df['kms_driven'].median()).astype(int)

#Fill missing fuel_type with mode

df['fuel_type'] = df['fuel_type'].fillna(df['fuel_type'].mode()[0])

#Keep first 3 words in name

df['name'] = df['name'].apply(lambda x: ' '.join(str(x).split()[:3]))

#Cap extremely high prices

mean_price = df.loc[df['Price'] <= 5_000_000, 'Price'].mean()

df.loc[df['Price'] > 5_000_000, 'Price'] = int(mean_price)

df['Price'] = df['Price'].astype(int)

st.write("### 🧹 Cleaned Data")

st.dataframe(df.head())

#========================================

#STEP 3: VISUALIZATION

#========================================

st.header("📊 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Distribution")

fig, ax = plt.subplots()

sns.histplot(df['Price'], kde=True, color='lightgreen', ax=ax)

st.pyplot(fig)

with col2:
    st.subheader("Feature Correlation Heatmap")

numeric_df = df.select_dtypes(include=['number'])

corr = numeric_df.corr(numeric_only=True)

fig, ax = plt.subplots()

sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)

st.pyplot(fig)

#========================================
#STEP 4: MODEL TRAINING
#========================================

st.header("🤖 Model Training")

features = ['name', 'company', 'year', 'kms_driven', 'fuel_type']

target = 'Price'

df = df.dropna(subset=features + [target])

X = df[features]

y = df[target]

categorical_cols = ['name', 'company', 'fuel_type']

numeric_cols = ['year', 'kms_driven']

column_transform = ColumnTransformer([

("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),

("num", StandardScaler(), numeric_cols)

])

model = Pipeline([

('transformer', column_transform),

('regressor', RandomForestRegressor(random_state=42))

])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.success("✅ Model trained successfully!")

st.write(f"R² Score: {r2:.3f}")

st.write(f"MAE: {mae:.2f}")

st.write(f"RMSE: {rmse:.2f}")

joblib.dump(model, "car_price_rf_model.pkl")

#========================================
#STEP 5: CAR PRICE PREDICTION
#========================================

st.header("🔮 Car Price Prediction Tool")

col1, col2, col3 = st.columns(3)

with col1:

    selected_company = st.selectbox("Car Company", sorted(df['company'].unique()))

year = st.slider("Year of Manufacture", min_value=2005, max_value=2024, value=2019)

with col2:
   filtered_names = df[df['company']==selected_company]['name'].unique()

selected_name = st.selectbox("Car Model", sorted(filtered_names))

kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=60000, step=500)

with col3:
   fuel_type = st.selectbox("Fuel Type", sorted(df['fuel_type'].unique()))

if st.button("🚀 Predict Car Price"):
    loaded_model = joblib.load("car_price_rf_model.pkl")

    input_data = pd.DataFrame({
        
        'name': [selected_name],
        'company': [selected_company],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
    })
    pred_price = loaded_model.predict(input_data)[0]
    st.success(f"Predicted Car Price: ₹{pred_price:,.2f}")


st.markdown("---")

st.caption("Built with ❤ by Hanee and krishna | Streamlit + Scikit-learn")

