import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
#import pickle

# Load the trained model
joblib_file = "xgboost_model.pkl" 
scaler_file = "scaler.pkl" 
model = joblib.load(joblib_file)
scaler = joblib.load(scaler_file)

# Sidebar button styles
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        width: 250px;
    }
    .stButton > button {
        width: 100%;
        margin-bottom: 10px;
        background-color: #008CBA; /* Blue background */
        color: white; /* White text */
        border-radius: 5px; /* Rounded corners */
        border: none;
        padding: 10px 20px;
        text-align: center;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #005f6b; /* Darker blue on hover */
    }                
    </style>
    """, unsafe_allow_html=True)

# Load the dataset
#df_diabetes = pd.read_csv('diabetes2015.csv')
df0 = pd.read_csv('credit_risk_dataset.csv')
df=df0.sample(n=3000, random_state=42)
df['loan_status'] = df['loan_status'].replace({1: 'Default', 0: 'Non-default'})
#df2=df.sample(n=3000, random_state=42)
dummy_columns = pd.Index(['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
       'person_home_ownership_OTHER', 'person_home_ownership_OWN',
       'person_home_ownership_RENT', 'loan_intent_EDUCATION',
       'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
       'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'loan_grade_B',
       'loan_grade_C', 'loan_grade_D', 'loan_grade_E', 'loan_grade_F',
       'loan_grade_G', 'cb_person_default_on_file_Y'])

numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']#
category_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
category_cols_label = ['loan_status']+category_cols
selected_features_cn = numerical_cols+category_cols
reduced_cols = ['person_age', 'loan_percent_income', 'loan_int_rate', 'loan_status']
all_variables = df.columns.tolist()

# Sidebar menu
st.sidebar.title("Credit Risk Prediction")
menu = st.sidebar.radio("Select a page", ["Credit Risk Prediction", "Static Visualization", "Dynamic Visualization"])

# Data Visualization Page
if menu == "Dynamic Visualization":
    st.title("Dynamic Visualization")
    
    # # Default values
    # default_x = 'person_income'
    # default_y = 'loan_amnt'
    # default_hue = 'loan_status'
    

    # Dropdown for graph type selection
    graph_type = st.selectbox("Select Graph Type", ["Bar Plot", "Scatter Plot", "Box Plot", "Histogram", "Line Plot"])

    # Dynamically update the options for x and y based on selected graph type
    if graph_type == "Bar Plot":
        x_var = st.selectbox("Select X variable", category_cols_label, index=category_cols_label.index("loan_intent"))
        y_var = None  # y is not needed for bar plots
        hue_var = st.selectbox("Select Hue (optional)", [None] + category_cols_label, index=category_cols_label.index("loan_status")+1)

    elif graph_type in ["Scatter Plot", "Line Plot"]:
        x_var = st.selectbox("Select X variable", numerical_cols, index=numerical_cols.index("person_age"))
        y_var = st.selectbox("Select Y variable", numerical_cols, index=numerical_cols.index("loan_percent_income"))
        hue_var = st.selectbox("Select Hue (optional)", [None] + category_cols_label, index=category_cols_label.index("loan_status")+1)

    elif graph_type == "Box Plot":
        x_var = st.selectbox("Select X variable", category_cols_label, index=category_cols_label.index("loan_intent"))
        y_var = st.selectbox("Select Y variable", numerical_cols, index=numerical_cols.index("loan_amnt"))
        hue_var = st.selectbox("Select Hue (optional)", [None] + category_cols_label, index=category_cols_label.index("loan_status")+1)

    elif graph_type == "Histogram":
        x_var = st.selectbox("Select X variable", numerical_cols, index=numerical_cols.index("loan_amnt"))
        y_var = None  # y is not needed for histograms
        hue_var = st.selectbox("Select Hue (optional)", [None] + category_cols_label, index=category_cols_label.index("loan_status")+1)

    # Plotting the selected graph
    if st.button("Generate Plot"):
        plt.figure(figsize=(10, 6))
        
        if graph_type == "Bar Plot":
            sns.countplot(x=x_var, hue=hue_var, data=df)
        elif graph_type == "Scatter Plot":
            sns.scatterplot(x=x_var, y=y_var, hue=hue_var, data=df)
        elif graph_type == "Box Plot":
            sns.boxplot(x=x_var, y=y_var, hue=hue_var, data=df)
        elif graph_type == "Histogram":
            sns.histplot(x=x_var, hue=hue_var, data=df, kde=True)
        elif graph_type == "Line Plot":
            sns.lineplot(x=x_var, y=y_var, hue=hue_var, data=df)
        
        st.pyplot(plt)

# Diabetes Prediction Page
elif menu == "Credit Risk Prediction":
    st.title("Credit Risk Prediction")
    st.header("User Input Parameters")

    def user_input_features():
        # Input fields
        col1, col2 = st.columns(2)
        with col1:
            person_age = st.number_input("Person Age", min_value=18, max_value=100, placeholder='18-100')
        with col2:
            person_income = st.number_input('Person Income', placeholder='>=4000', min_value=4000, max_value=7000000, value=10000)

        col3, col4 = st.columns(2)
        with col3:
            person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        with col4:
            person_emp_length = st.number_input("Employment Length (in years)", min_value=0, max_value=50, placeholder="0-50", value=4)
            #gender = st.radio('Gender', ('Male', 'Female'), horizontal=True)

        col5, col6 = st.columns(2)
        with col5:
            loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        with col6:
            loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        
        col7, col8 = st.columns(2)
        with col7:
            loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=35000, value=1000)
        with col8:
            loan_int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=25.0, value=10.0, format="%.2f")
        
        col9, col10 = st.columns(2)
        with col9:    
            cb_person_default_on_file = st.radio('Historical Default on File', ('No', 'Yes'), horizontal=True)
            # Convert radio button selection to True/False
            cb_person_default_on_file = True if cb_person_default_on_file == "Yes" else False
        with col10:
            cb_person_cred_hist_length = st.number_input("Credit History Length (in years)", min_value=2, max_value=30, value=10)

        # Data dictionary
        data = {
            'person_age': person_age,
            'person_income': person_income,
            'person_home_ownership': person_home_ownership,
            'person_emp_length': person_emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'cb_person_default_on_file': cb_person_default_on_file,
            'cb_person_cred_hist_length': cb_person_cred_hist_length
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # Predict button
    if st.button('Predict (XGBoost)'):

        input_df['loan_percent_income'] = float(input_df['loan_amnt'][0]) / float(input_df['person_income'][0])
        predicting_data = input_df
        predicting_data[numerical_cols] = scaler.transform(predicting_data[numerical_cols])
        predicting_data = pd.get_dummies(predicting_data)
        # Align the new data to the original training columns
        predicting_data = predicting_data.reindex(columns=dummy_columns, fill_value=False)
        predicted_rs = model.predict(predicting_data)
        rs_text = 'Default' if predicted_rs == 1 else 'Non-Default'
        st.markdown(f"## {rs_text}")
elif menu == "Static Visualization":
    st.title("Static Visualization")  
    # Example static plots
    # st.subheader("Loan Status Distribution")
    # plt.figure(figsize=(10, 6))
    # sns.countplot(x='loan_status', data=df)
    # st.pyplot(plt)

    # st.subheader("Loan Amount Distribution by Home Ownership")
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(x='person_home_ownership', y='loan_amnt', data=df)
    # st.pyplot(plt)
    
    # st.subheader("Interest Rate Distribution")
    # plt.figure(figsize=(10, 6))
    # sns.histplot(x='loan_int_rate', data=df, kde=True)
    # st.pyplot(plt)   
    # Visualize categorical coluumns
    st.subheader("Categorical Columns Visualization")
    for col in category_cols:
        st.write(f"Distribution of {col}")
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col, hue="loan_status")
        plt.xticks(rotation=45)
        st.pyplot(plt) 
    # Visualize distributions of numerical features
    st.subheader("Numerical Columns Visualization")
    st.write("Histograms of Numerical Features")
    df[numerical_cols].hist(bins=20, figsize=(14, 10))
    st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure and show it in Streamlit

    # Correlation matrix
    st.write("Correlation Matrix")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[['loan_status']+numerical_cols].replace({'Default':1, 'Non-default':0}).corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

    @st.cache_data
    def generate_pairplot(data):
        sns.pairplot(data, hue='loan_status')
        return plt.gcf()
    
    # Pairplot
    st.write("Pairplot of Numerical Features")
    pairplot_figure = generate_pairplot(df[reduced_cols])
    st.pyplot(pairplot_figure)    
