import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# =========================
# Load Model
# =========================
xgb_model = joblib.load("model/xgboost_churn_model.pkl")

# =========================
# SHAP Helper
# =========================
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# =========================
# Preprocessing Pipeline
# =========================
def preprocess_data(structured_df, reviews_df=None):
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    import re, string, emoji
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    # --- 1: RFM Features ---
    structured_df['Recency'] = structured_df['days_since_last_purchase']
    structured_df['Frequency'] = structured_df['order_frequency']
    structured_df['Monetary'] = structured_df['avg_order_value']

    # --- 1b: Frequency Risk (added back to match training) ---
    structured_df['Frequency_Risk'] = 1 / (structured_df['Frequency'] + 1e-6)

    # --- 2: Churn Risk Index ---
    structured_df['Loyalty_Risk_Factor'] = structured_df['loyalty_program'].apply(lambda x: 0 if x == 'Yes' else 1)
    structured_df['Recency_Rank'] = structured_df['Recency'].rank(method='first', ascending=False)
    structured_df['Frequency_Rank'] = structured_df['Frequency'].rank(method='first', ascending=True)
    structured_df['Churn_Risk_Index'] = (
        structured_df['Recency_Rank'] + structured_df['Frequency_Rank'] +
        structured_df['Loyalty_Risk_Factor'] * (structured_df['Recency_Rank'].max() + structured_df['Frequency_Rank'].max())
    )

    # --- 3: City Tier ---
    urban_cities = ['Ikeja', 'Lekki', 'Victoria Island', 'Yaba', 'Surulere', 'Maitama', 'Garki', 'Wuse']
    suburban_cities = ['Rumuokoro', 'Mile 1', 'Trans-Amadi', 'D-line', 'Port Harcourt GRA', 'Utako', 'Asokoro']
    def classify_city_tier(city):
        if city in urban_cities:
            return 'Urban'
        elif city in suburban_cities:
            return 'Suburban'
        else:
            return 'Other'
    structured_df['City_Tier'] = structured_df['city'].apply(classify_city_tier)

    # --- 4: Spend-to-Income ---
    income_mapping = {'Low': 1, 'Middle': 2, 'High': 3}
    structured_df['Income_Proxy'] = structured_df['income_level'].map(income_mapping)
    structured_df['Spend_to_Income_Ratio'] = structured_df['avg_order_value'] / (structured_df['Income_Proxy'] + 1e-6)
    structured_df['Spend_to_Income_Ratio_Quartile'] = pd.qcut(
        structured_df['Spend_to_Income_Ratio'],
        q=4,
        labels=['Low_Ratio', 'Medium_Low_Ratio', 'Medium_High_Ratio', 'High_Ratio']
    )

    # --- 5: Age Range ---
    age_bins = [0, 18, 25, 35, 45, 55, 65, structured_df['age'].max() + 1]
    age_labels = ['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+']
    structured_df['Age_Range'] = pd.cut(structured_df['age'], bins=age_bins, labels=age_labels, right=False)

    # --- 6: Ordinal Encoding ---
    ordinal_features = ['income_level', 'Age_Range', 'Spend_to_Income_Ratio_Quartile']
    income_categories = [['Low', 'Middle', 'High']]
    age_range_categories = [structured_df['Age_Range'].dropna().unique().categories.tolist()]
    age_range_categories[0].sort()
    spend_ratio_categories = [['Low_Ratio', 'Medium_Low_Ratio', 'Medium_High_Ratio', 'High_Ratio']]
    ordinal_encoder = OrdinalEncoder(
        categories=income_categories + age_range_categories + spend_ratio_categories,
        handle_unknown='use_encoded_value', unknown_value=-1
    )
    df_churn_ordinal_encoded = structured_df[ordinal_features].copy()
    df_churn_ordinal_encoded[ordinal_features] = ordinal_encoder.fit_transform(df_churn_ordinal_encoded[ordinal_features])

    # --- 7: One-Hot Encoding ---
    categorical_cols = structured_df.select_dtypes(include='object').columns.tolist()
    extra_categorical = ['City_Tier', 'Spend_to_Income_Ratio_Quartile', 'Age_Range']
    all_categorical_features = list(set(categorical_cols + extra_categorical))
    nominal_features = [col for col in all_categorical_features if col not in ordinal_features + ['customer_id', 'churn']]
    from sklearn.preprocessing import OneHotEncoder
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    nominal_encoded_data = one_hot_encoder.fit_transform(structured_df[nominal_features])
    nominal_encoded_df = pd.DataFrame(
        nominal_encoded_data,
        columns=one_hot_encoder.get_feature_names_out(nominal_features),
        index=structured_df.index
    )

    # --- 8: Numeric Features ---
    numeric_cols = structured_df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'churn']
    numeric_subset = structured_df[['customer_id'] + numeric_cols].copy()
    numeric_subset = numeric_subset.rename(columns={'customer_id': 'CustomerID'})

    # --- 9: Merge Structured Data ---
    df_structured_encoded = pd.concat([numeric_subset, df_churn_ordinal_encoded, nominal_encoded_df], axis=1)

    # --- 10: Sentiment Merge ---
    if reviews_df is not None and "review_text" in reviews_df.columns:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        def clean_text(text):
            if pd.isna(text): return ""
            text = text.lower()
            text = emoji.demojize(text, delimiters=(" ", " "))
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\s+', ' ', text.strip())
            return text
        reviews_df['cleaned_review_text'] = reviews_df['review_text'].apply(clean_text)
        reviews_df['vader_compound_score'] = reviews_df['cleaned_review_text'].apply(
            lambda x: analyzer.polarity_scores(x)['compound']
        )
        sentiment_agg = reviews_df.groupby('customer_id')['vader_compound_score'].mean().reset_index()
        sentiment_agg = sentiment_agg.rename(columns={'vader_compound_score': 'average_sentiment_score'})
        df_structured_encoded = df_structured_encoded.merge(
            sentiment_agg, left_on="CustomerID", right_on="customer_id", how="left"
        ).drop("customer_id", axis=1)
    else:
        df_structured_encoded['average_sentiment_score'] = 0

    return df_structured_encoded

# =========================
# Prediction
# =========================
def predict_churn(preprocessed_df):
    if "CustomerID" in preprocessed_df.columns:
        customer_ids = preprocessed_df["CustomerID"].values
    else:
        customer_ids = np.arange(len(preprocessed_df))

    feature_df = preprocessed_df.drop(
        columns=[c for c in ["CustomerID", "churn"] if c in preprocessed_df.columns],
        errors="ignore"
    )

    # 🔑 Align features with training
    expected_features = xgb_model.get_booster().feature_names
    feature_df = feature_df.reindex(columns=expected_features, fill_value=0)

    preds = xgb_model.predict(feature_df)
    probs = xgb_model.predict_proba(feature_df)[:, 1]

    return pd.DataFrame({
        "CustomerID": customer_ids,
        "Churn_Probability": probs,
        "Churn_Prediction": preds
    })

# =========================
# Global SHAP
# =========================
def explain_with_shap(preprocessed_df, sample_size=100):
    feature_df = preprocessed_df.drop(
        columns=[c for c in ["CustomerID", "churn"] if c in preprocessed_df.columns],
        errors="ignore"
    )
    expected_features = xgb_model.get_booster().feature_names
    feature_df = feature_df.reindex(columns=expected_features, fill_value=0)

    sample = feature_df.sample(min(sample_size, len(feature_df)), random_state=42)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(sample)

    st.subheader(" SHAP Feature Importance (Global)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    st.pyplot(fig)

    st.subheader(" SHAP Summary Plot (Beeswarm)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample, show=False)
    st.pyplot(fig)

# =========================
# Per-Customer SHAP + Recommendations
# =========================
def explain_single_customer(preprocessed_df, customer_id):
    feature_df = preprocessed_df.drop(
        columns=[c for c in ["CustomerID", "churn"] if c in preprocessed_df.columns],
        errors="ignore"
    )
    expected_features = xgb_model.get_booster().feature_names
    feature_df = feature_df.reindex(columns=expected_features, fill_value=0)

    row_index = preprocessed_df[preprocessed_df["CustomerID"] == customer_id].index[0]
    row_features = feature_df.loc[[row_index]]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(row_features)

    st.subheader(f" SHAP Explanation for Customer {customer_id}")
    st_shap(shap.plots.force(shap_values[0]), height=300)

    contribs = pd.DataFrame({
        "Feature": row_features.columns,
        "SHAP_Value": shap_values.values[0]
    }).sort_values(by="SHAP_Value", key=abs, ascending=False)

    st.subheader(" Feature Contributions (Top Drivers)")
    st.dataframe(contribs.head(10))

    st.subheader(" Recommended Actions")
    top_features = contribs.head(3)
    for _, row in top_features.iterrows():
        if row["Feature"].lower().startswith("recency"):
            st.write("- Customer inactive for a while → Send re-engagement offers or reminders.")
        elif row["Feature"].lower().startswith("frequency"):
            st.write("- Low purchase frequency → Offer loyalty discounts or bundles.")
        elif "sentiment" in row["Feature"].lower():
            st.write("- Negative sentiment detected → Prioritize customer support outreach.")
        elif "income" in row["Feature"].lower():
            st.write("- Consider tailored pricing or installment plans for lower-income customers.")
        else:
            st.write(f"- Monitor impact of **{row['Feature']}** on churn and design targeted outreach.")

# =========================
# Segment-Level Insights
# =========================
def show_segment_insights(results, processed_df):
    st.subheader(" Segment-Level Churn Insights")
    merged = processed_df.merge(results, on="CustomerID")
    segment_cols = ["City_Tier", "income_level", "Age_Range", "platform"]
    for col in segment_cols:
        if col in merged.columns:
            st.markdown(f"**Churn Probability by {col}**")
            segment_stats = merged.groupby(col)["Churn_Probability"].mean().sort_values(ascending=False)
            st.bar_chart(segment_stats)

# =========================
# Top At-Risk Customers
# =========================
def show_top_risk_customers(results, top_n=10):
    st.subheader(f" Top {top_n} At-Risk Customers")
    top_customers = results.sort_values("Churn_Probability", ascending=False).head(top_n)
    st.dataframe(top_customers)
    csv = top_customers.to_csv(index=False).encode('utf-8')
    st.download_button(
        f"Download Top {top_n} At-Risk Customers",
        data=csv,
        file_name=f"top_{top_n}_at_risk_customers.csv",
        mime="text/csv"
    )

# =========================
# Generate PDF Report
# =========================
def generate_pdf_report(results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(" Customer Churn Prediction Report", styles['Title']))
    story.append(Spacer(1, 12))

    avg_churn = results["Churn_Probability"].mean()
    story.append(Paragraph(f"Average Predicted Churn Probability: {avg_churn:.2f}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Table of Top 10 At-Risk Customers
    top_customers = results.sort_values("Churn_Probability", ascending=False).head(10)
    data = [list(top_customers.columns)] + top_customers.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
    ]))
    story.append(table)

    doc.build(story)
    buffer.seek(0)
    return buffer

# =========================
# Streamlit UI
# =========================
st.title("Customer Churn Prediction & Insights Dashboard")
st.markdown("Upload customer structured data (required) and reviews (optional). The model predicts churn probability, explains feature importance using SHAP, and provides actionable insights.")

structured_file = st.file_uploader("Upload Structured Data CSV", type="csv")
reviews_file = st.file_uploader("Upload Reviews Data CSV (Optional)", type="csv")

if structured_file:
    structured_df = pd.read_csv(structured_file)
    reviews_df = pd.read_csv(reviews_file) if reviews_file else None
    st.subheader(" Data Preview")
    st.dataframe(structured_df.head())

    if st.button("Run Prediction"):
        processed_df = preprocess_data(structured_df, reviews_df)
        results = predict_churn(processed_df)
        st.session_state['processed_df'] = processed_df
        st.session_state['results'] = results

    if 'results' in st.session_state:
        st.subheader("🔮 Prediction Results")
        st.dataframe(st.session_state['results'].head(20))
        csv = st.session_state['results'].to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", data=csv, file_name="churn_predictions.csv", mime="text/csv")

        # Insights
        show_segment_insights(st.session_state['results'], st.session_state['processed_df'])
        show_top_risk_customers(st.session_state['results'], top_n=10)

        # SHAP Explanations
        explain_with_shap(st.session_state['processed_df'])
        st.subheader(" Explain Individual Customer Prediction")
        customer_list = st.session_state['processed_df']["CustomerID"].unique().tolist()
        selected_customer = st.selectbox("Select CustomerID to Explain", customer_list)
        if st.button("Explain This Customer"):
            explain_single_customer(st.session_state['processed_df'], selected_customer)

        # PDF Report Download
        pdf_buffer = generate_pdf_report(st.session_state['results'])
        st.download_button(" Download PDF Report", data=pdf_buffer, file_name="churn_report.pdf", mime="application/pdf")
else:
    st.warning("Please upload the structured customer data CSV to proceed.")
