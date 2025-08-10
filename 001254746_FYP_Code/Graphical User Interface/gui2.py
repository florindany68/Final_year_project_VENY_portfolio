import joblib
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
import os
from openai import OpenAI
import base64
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np



logo_sidebar = '/home/danny/Desktop/001254746_FYP_Code/Graphical User Interface/VENY AI(6).png'
logo_veny = '/home/danny/Desktop/001254746_FYP_Code/Graphical User Interface/VENY AI.svg'

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image(logo_veny, width=300)

st.logo(logo_veny, size='large',icon_image=logo_sidebar)
st.title("Hi, I'm Veny! Your AI assistant for investing")
st.markdown(" I'm your assistant to help you enter into the vast world of investing. Choose your level your risk to"
            "get the most appropriate companies for you. "
            "Low Risk: This portfolio focuses on stable, established companies—often called 'blue-chip' stocks—and may "
            "include dividend-paying stocks. "
            "It aims for slow but steady growth with less volatility, making it suitable for conservative investors."
            "Medium Risk: A balanced portfolio that includes a mix of stable companies and some growth stocks. It aims to grow your investment over time"
            "while managing potential losses. This is a good choice if you're seeking moderate growth and can tolerate some ups and downs."
            "High Risk: This portfolio is heavily weighted toward smaller companies with high potential. "
            "It can offer high returns but also bigger swings in value. Ideal if you have a long time horizon and are comfortable taking more risk.""")



dataset_path = "/home/danny/Desktop/001254746_FYP_Code/data/processed_file.csv"
model_path = "/home/danny/Desktop/001254746_FYP_Code/saved_models/RandomForest_modelMax.pkl"
scaler_path = "/home/danny/Desktop/001254746_FYP_Code/saved_models/feature_scalerRF_Max.pkl"
id_columns  = ['Industry', 'Sector', 'Company','Symbol']
exclude_columns = id_columns + ['Risk Category', 'Risk Score']
LSTM_model_path = "/home/danny/Desktop/001254746_FYP_Code/saved_models/LSTM_model_final.keras"
prediction_model = load_model(LSTM_model_path)
lstm_scaler = "/home/danny/Desktop/001254746_FYP_Code/saved_models/LSTM_model_final_scaler.pkl"
scaler_lstm = joblib.load(lstm_scaler)


def fetching_data(ticker, start_date='2010-12-31', end_date='2024-12-31'):
    data = yf.download(ticker, start=start_date, end=end_date)
    df = pd.DataFrame(data['Close'].copy())
    return df

def predict_future_return(prediction_model, ticker, scaler_lstm, end_date='2024-12-31', sequence_length=252):
    # Fetch historical stock data
    df_stock = fetching_data(ticker, start_date='2010-12-31', end_date=end_date)

    train_data_partition = int(len(df_stock) * 0.8)
    memory_start_idx = max(train_data_partition - sequence_length, 0)
    start_from = df_stock.index[memory_start_idx]
    df_test = df_stock[df_stock.index >= start_from]
    test_data = df_test.values.reshape(-1, 1)


    if len(test_data) <= sequence_length:
        raise ValueError(f"Not enough data to create sequences for {ticker}!")

    test_data_scaled = scaler_lstm.transform(test_data)

    # Predict first test day
    first_sequence = test_data_scaled[:sequence_length]
    first_sequence = first_sequence.reshape(1, sequence_length, 1)
    first_test_day_prediction_scaled = prediction_model.predict(first_sequence)
    first_test_day_prediction = scaler_lstm.inverse_transform(first_test_day_prediction_scaled)[0][0]

    # Predict last test day
    last_sequence = test_data_scaled[-sequence_length:]
    last_sequence = last_sequence.reshape(1, sequence_length, 1)
    last_test_day_prediction_scaled = prediction_model.predict(last_sequence)
    last_test_day_prediction = scaler_lstm.inverse_transform(last_test_day_prediction_scaled)[0][0]

    # Calculate predicted return
    predicted_return = ((last_test_day_prediction - first_test_day_prediction) / first_test_day_prediction) * 100

    return {
        'first_predicted_price': first_test_day_prediction,
        'last_predicted_price': last_test_day_prediction,
        'predicted_return_pct': predicted_return,
    }


def lime_explanations_company(dataset_df, model, scaler, features, selected_index):
    X_data = dataset_df[features]
    X_scaled = scaler.transform(X_data)

    explainer = LimeTabularExplainer(
        training_data=X_scaled,
        feature_names=features,  # This parameter takes the features from the model
        class_names=['Low', 'Medium', 'High'],  # This parameter takes the target variables of the model
        mode='classification',
        discretize_continuous=True
    )
    iloc_index = dataset_df.index.get_loc(selected_index)
    instance = X_scaled[iloc_index]

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=len(features),
    )

    company_pick = dataset_df.loc[selected_index, 'Company']
    plt.figure(figsize=(12, 8))
    explanation.as_pyplot_figure()
    plt.title(f"LIME Explanation for {company_pick}")
    plt.tight_layout()

    # Save the figure
    lime_file = f"lime_explanation_{company_pick}.png"
    plt.savefig(lime_file)
    plt.close()

    return lime_file

def open_api_explanaitions_specific_company(lime_file, specific_index):
    os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    def encode_image(lime_file):
        with open(lime_file, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    base64_image = encode_image(lime_file)

    prompt = f"""You are a financial advisor and the Lime graph (your pick for the user) is your reason for picking the stock
    Explain, in simple day to day terms, why { specific_index } is a great pick for the user level of risk. """
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system",
             "content": "You are a financial advisor and you explain me what are the most important features based on the level of risk in very beginner terms"},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content

def shap_explanations(X_new, model, features):
    explainer = shap.Explainer(model, X_new)
    shap_values = explainer(X_new, check_additivity=False)

    plt.figure(figsize=(20, 20))
    shap.summary_plot(shap_values,
                      X_new,
                      feature_names=features,
                      max_display=20,
                      plot_type="bar",
                      class_names=['Low', 'Medium', 'High'],
                      show=False)
    plt.tight_layout()
    plt.title(f'SHAP Feature Importance Summary ')
    plt.legend(labels=['Low', 'Medium', 'High'])
    explicatie_shap = 'shap_summary_bar_RFoverall.png'
    plt.savefig(explicatie_shap)
    plt.close()
    return explicatie_shap

def open_api_explanaitions_overall_model(image_path):
    os.environ[
        "OPENAI_API_KEY"]

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    base64_image = encode_image(image_path)

    prompt = f"""You are given a model output, you are a financial advisor and explain why did I got these companies, what indicators
    (features) mattered most for my predictions. Try to explain these in day to day terms, because the user never invested before and make it very 
    clear to them. The user does not have the graph displayed, tell them what you think, rather than telling them you read from a graph."""
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system",
             "content": "You are a financial advisor and you explain me what are the most important features based on the level of risk in very beginner terms"},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content

if 'lime_file' not in st.session_state:
    st.session_state.lime_file = None
if 'portfolio_shown' not in st.session_state:
    st.session_state.portfolio_shown = False
if 'prediction_done' not in st.session_state:
    st.session_state['prediction_done'] = False
if 'dataset_df' not in st.session_state:
    st.session_state['dataset_df'] = None

if st.button("Start the prediction"):
    with st.spinner('Loading data...'):
        dataset_df = pd.read_csv(dataset_path)
        st.success(f'Dataset loaded. Shape: {dataset_df.shape}')

    with st.spinner('Loading model...'):
        model = joblib.load(model_path)

    with st.spinner('Loading scaler...'):
        scaler = joblib.load(scaler_path)
    with st.spinner('Making predictions...'):
        features = [c for c in dataset_df if c not in exclude_columns]
        st.session_state.features = features
        X_new = scaler.transform(dataset_df[features])
        y_pred = model.predict(X_new)
        dataset_df["Predicted Category"] = y_pred

        st.session_state.model =model
        st.session_state.scaler = scaler
        st.session_state.dataset_df = dataset_df
        st.session_state.prediction_done = True
        st.success(f"Prediction Complete  with an {accuracy_score(dataset_df['Risk Category'], y_pred):} accuracy")

if st.session_state.prediction_done:
    option = st.selectbox(
        "Pick your style of investments",
        ['Low', 'Medium', 'High']
        )
    st.write("You chose:", option)

    if st.button("Show your portfolio") or st.session_state.portfolio_shown:
        st.session_state.portfolio_shown = True
        user_companies = st.session_state.dataset_df[st.session_state.dataset_df['Predicted Category'] == option]

        if len(user_companies) > 0:
           st.write(f"Here are the the most appropriate companies that {option} risk:")
           if "Risk Score" in user_companies.columns:
                user_companies = user_companies.sort_values(by=['Risk Score'])

           user_generated_portfolio = user_companies[id_columns + ['Predicted Category']].head(10)
           st.dataframe(user_generated_portfolio)



           initial_investment = st.number_input("Enter your initial investment: ",
                                                min_value=1000, max_value=1000000, step=1000)



           company_indices = user_generated_portfolio.index.tolist()
           company_names = user_generated_portfolio['Company'].tolist()
           ticker_company = user_generated_portfolio['Symbol'].tolist()
           company_options = [f"{idx}: {name}" for idx, name in zip(company_indices, company_names)]

           st.subheader("Projected Future Returns for Your Portfolio")

           portfolio_returns = []



           for ticker in ticker_company:
                   progress_bar = st.progress(0)
                   try:
                       company_name = company_names[ticker_company.index(ticker)]
                       result = predict_future_return(prediction_model, ticker, scaler_lstm, sequence_length=252)
                       portfolio_returns.append({
                           'Company': company_name,
                           'Ticker': ticker,
                           'Predicted First Price': result['first_predicted_price'],
                           'Predicted Last Price': result['last_predicted_price'],
                           'Predicted Return (%)': result['predicted_return_pct']
                       })
                   except Exception as e:
                       st.warning(f"There was an error while predicting {ticker}. {e}")
                   progress_bar.empty()


           if portfolio_returns:
               returns_df = pd.DataFrame(portfolio_returns)
               stock_allocation = initial_investment /len(returns_df)
               returns_df['Allocation ($)'] = stock_allocation
               returns_df["Expected Value ($)"] = returns_df["Allocation ($)"] *(1+returns_df["Predicted Return (%)"]/100)
               returns_df["Profit/Loss ($)"] = returns_df["Expected Value ($)"] - returns_df["Allocation ($)"]

               user_portfolio_display = returns_df.copy()
               for col in ['Predicted First Price', 'Predicted Last Price', 'Predicted Return (%)',
                           'Allocation ($)', 'Expected Value ($)', 'Profit/Loss ($)']:
                   user_portfolio_display[col] = user_portfolio_display[col].apply(lambda x: f"{x:.2f}")

               st.dataframe(user_portfolio_display)

               total_expected_value = returns_df['Expected Value ($)'].sum()
               total_profit_loss = returns_df['Profit/Loss ($)'].sum()
               full_portfolio_return = (total_expected_value / initial_investment - 1) * 100

               st.subheader("Portfolio Summary")
               col1, col2, col3 = st.columns(3)
               with col1:
                   st.metric("Initial Investment", f"${initial_investment:,.2f}")
               with col2:
                   st.metric("Expected Value", f"${total_expected_value:,.2f}")
               with col3:
                   st.metric("Expected Return", f"{full_portfolio_return:.2f}%",
                             delta=f"${total_profit_loss:,.2f}")



           else:
               st.warning("No future return predictions available.")


           selected_company_option = st.selectbox(
               "Select a company for detailed explanation:",
               company_options,
           )

           if st.button("Show LIME explanation", key='mor_lime'):
               with st.spinner("Generating LIME explanation..."):
                selected_index = int(selected_company_option.split(':')[0])
                lime_file = lime_explanations_company(
                    dataset_df=st.session_state.dataset_df,
                    model=st.session_state.model,
                    scaler=st.session_state.scaler,
                    features=st.session_state.features,
                    selected_index=selected_index,
                   )
                st.image(lime_file)
                with st.spinner("Generating predictions..."):
                    explicatie_user2 = open_api_explanaitions_specific_company(lime_file, selected_index)
                    st.subheader(f"Overall Explanation for {selected_company_option} :")
                    st.write(explicatie_user2)

           scaler = st.session_state.scaler
           model = st.session_state.model
           features = st.session_state.features
           X_all = st.session_state.dataset_df[features]
           X_selected = scaler.transform(X_all)
           shap_graph = shap_explanations(X_selected, model, features)
           st.image(shap_graph)

           with st.spinner("I am thinking..."):
               explicatie_user = open_api_explanaitions_overall_model(shap_graph)
               st.subheader("Overall Explanation")
               st.write(explicatie_user)
        else:
           st.warning("No companies found with the selected risk level")


