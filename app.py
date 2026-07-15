import streamlit as st

st.set_page_config(
    page_title="Life Microinsurance Pricing Platform",
    page_icon="📊",
    layout="wide"
)

st.title("Life Microinsurance Pricing Platform")

st.markdown("""
A machine learning-driven actuarial pricing framework designed to improve affordability and accessibility of life microinsurance products for informal sector workers.
""")

st.divider()

st.header("Executive Summary")

st.write("""
This project demonstrates how actuarial pricing and machine learning can be combined to create fair and sustainable microinsurance products.

A Random Forest model was used to classify policyholders into risk categories before premium estimation.
""")
