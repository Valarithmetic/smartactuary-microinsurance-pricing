import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Microinsurance Risk Classification",
    layout="wide"
)

st.title(
    "Machine Learning Risk Classification Framework for Life Microinsurance"
)

st.markdown("""
A predictive underwriting model developed using
Random Forest classification to support risk assessment.
""")

st.header("Executive Dashboard")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Accuracy","90%")
col2.metric("Risk Factors","6")
col3.metric("Risk Classes","2")
col4.metric("Model","Random Forest")

st.header("Project Overview")

st.write("""
This project explores the use of machine learning
for risk classification within life microinsurance.
""")

st.header("Population Characteristics")

image = Image.open(
    "assets/descriptive_analysis.png"
)

st.image(
    image,
    use_container_width=True
)

st.header("Feature Importance")

image = Image.open(
    "assets/feature_importance.png"
)

st.image(
    image,
    use_container_width=True
)

st.header("Classification Performance")

image = Image.open(
    "assets/confusion_matrix.png"
)

st.image(
    image,
    use_container_width=True
)

st.header("Business Applications")

st.markdown("""
- Underwriting Support
- Risk Segmentation
- Product Pricing
- Product Development
""")
