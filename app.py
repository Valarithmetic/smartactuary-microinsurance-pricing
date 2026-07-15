import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

logo = Image.open("assets/logo.png")

st.set_page_config(
    page_title="Smart Actuary — Micro-Insurance Risk Classification",
    page_icon=logo,
    layout="wide",
)

# =========================================================
# SMART ACTUARY CASE STUDY TEMPLATE — shared styling
# Reuse this CSS block + component functions for every
# future case study app (Cox regression, Gompertz pricing,
# surrender model, etc.) to keep a consistent identity.
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500&family=JetBrains+Mono:wght@500&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #10233A; }
    h1, h2, h3 { font-family: 'Poppins', sans-serif; color: #F4EFE6 !important; }
    p, span, label, li, .stMarkdown, div[data-testid="stMarkdownContainer"] p { color: #C7D0DC !important; }
    hr { border-color: #26405F !important; }
    footer {visibility: hidden;}

    /* Hero */
    .sa-hero {
        background: radial-gradient(circle at 20% 20%, #1B3352 0%, #0C1B2E 70%);
        border-radius: 18px; padding: 64px 48px 52px; margin-bottom: 8px;
        border: 1px solid #26405F;
    }
    .sa-hero .tag { font-family:'JetBrains Mono'; font-size:12px; letter-spacing:3px; color:#D9A441 !important; text-transform:uppercase; }
    .sa-hero h1 { font-size:40px; font-weight:800; color:#F4EFE6 !important; margin:14px 0 12px; max-width:760px; line-height:1.15; }
    .sa-hero .subtags span {
        display:inline-block; font-family:'JetBrains Mono'; font-size:11px; color:#AEB9C6 !important;
        border:1px solid #33507A; border-radius:999px; padding:5px 14px; margin:4px 8px 4px 0;
    }

    /* KPI row */
    .sa-kpi { background:#F4EFE6; border-radius:12px; padding:20px 18px; text-align:center; border-top:4px solid #D9A441; }
    .sa-kpi .num { font-family:'JetBrains Mono'; font-size:32px; font-weight:600; color:#10233A !important; display:block; }
    .sa-kpi .lbl { font-size:12px; color:#5C6670 !important; text-transform:uppercase; letter-spacing:0.5px; }

    /* Section panel */
    .sa-eyebrow { font-family:'JetBrains Mono'; font-size:11px; letter-spacing:2px; color:#D9A441 !important; text-transform:uppercase; }
    .sa-h2 { font-size:24px; font-weight:700; color:#F4EFE6 !important; margin:6px 0 18px; }
    .sa-panel { background:#F4EFE6; border-radius:12px; padding:26px 30px; }
    .sa-panel p, .sa-panel li, .sa-panel span, .sa-panel strong { color:#10233A !important; }

    /* Result cards */
    .sa-card { background:#F4EFE6; border-radius:12px; padding:20px 22px; height:100%; border-top:4px solid #D9A441; }
    .sa-card h4 { color:#10233A !important; margin:0 0 4px; font-family:'Poppins'; font-size:13px; text-transform:uppercase; letter-spacing:0.5px; }
    .sa-card p, .sa-card span { color:#10233A !important; }
    .sa-card .big { font-family:'JetBrains Mono'; font-size:26px; font-weight:600; color:#10233A !important; }
    .risk-high { border-top-color:#b3541e; }
    .risk-low { border-top-color:#2f7a4f; }

    .stButton>button { background-color:#D9A441; color:#412402; font-weight:600; border-radius:8px; border:none; padding:10px 24px; width:100%; }
    .stButton>button:hover { background-color:#c4913a; color:#412402; }
</style>
""", unsafe_allow_html=True)


def sa_section_start(eyebrow, title):
    st.markdown(f'<span class="sa-eyebrow">{eyebrow}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="sa-h2">{title}</div>', unsafe_allow_html=True)


def sa_section_end():
    st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================
hcol1, hcol2 = st.columns([0.09, 0.91])
with hcol1:
    st.image(logo, width=64)
with hcol2:
    st.markdown('<span class="sa-eyebrow" style="padding-top:10px; display:inline-block;">Smart Actuary · Case Study</span>', unsafe_allow_html=True)

st.markdown("""
<div class="sa-hero">
    <h1>Machine Learning Risk Classification for Life Micro-Insurance</h1>
    <div class="subtags">
        <span>Predictive Underwriting</span>
        <span>Random Forest Classification</span>
        <span>Risk Analytics</span>
    </div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
for col, num, lbl in zip(
    [k1, k2, k3, k4],
    ["95%", "6", "2", "Random Forest"],
    ["Classification accuracy", "Predictive variables", "Risk categories", "Model"]
):
    col.markdown(f'<div class="sa-kpi"><span class="num">{num}</span><span class="lbl">{lbl}</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# THE CHALLENGE
# =========================================================
sa_section_start("The problem", "The Challenge")
st.markdown("""
Millions of informal-sector workers remain underserved by traditional insurance. Income is
irregular, formal records are sparse, and flat, one-size-fits-all pricing either overprices
low-risk applicants or underprices high-risk ones — leaving a significant share of the population
without affordable coverage.
""")
sa_section_end()

# =========================================================
# OUR SOLUTION
# =========================================================
sa_section_start("The approach", "Our Solution")
st.markdown("""
Smart Actuary developed a machine learning framework capable of classifying policyholders into
high- and low-risk categories using demographic, socioeconomic, and health indicators — assigning
each applicant a risk band before a premium is calculated, rather than pricing everyone the same.
""")
sa_section_end()

# =========================================================
# THE DATA
# =========================================================
sa_section_start("The dataset", "The Data")
st.markdown("""
The model was trained on **100 policyholder records** across six variables — age, gender,
education status, number of dependants, estimated income, and chronic illness history — with
risk level (High / Low) as the target.
""")
st.image(Image.open("assets/descriptive_analysis.png"), use_container_width=True)
st.markdown("""
**Key findings**

- Age was fairly evenly spread across the working-age population, roughly 18 to 65.
- Estimated income was heavily right-skewed — most applicants earned under KES 20,000/month.
- Chronic illness history was present in roughly 4 in 10 applicants.
- The dataset was close to balanced across both risk classes, avoiding a bias toward either label.
""")
sa_section_end()

# =========================================================
# KEY RISK DRIVERS
# =========================================================
sa_section_start("How the model thinks", "Key Risk Drivers")
st.image(Image.open("assets/feature_importance.png"), use_container_width=True)
st.markdown("""
Chronic illness history contributed over 40% of the model's predictive power — the strongest
single determinant of underwriting decisions — followed by estimated income and age. Gender had
the least influence, a useful check that the model isn't leaning on a factor it shouldn't be.
""")
sa_section_end()

# =========================================================
# MODEL PERFORMANCE
# =========================================================
sa_section_start("The results", "Model Performance")
col_img, col_stats = st.columns([1.3, 1])
with col_img:
    st.image(Image.open("assets/confusion_matrix.png"), use_container_width=True)
with col_stats:
    st.markdown("""
**95% overall accuracy**

- High-risk correctly classified: 12 / 13
- Low-risk correctly classified: 7 / 7

The model's one error was conservative — a high-risk applicant classified as low-risk never
happened in testing; the single misclassification ran the safer direction.
""")
sa_section_end()

# =========================================================
# BUSINESS IMPACT
# =========================================================
sa_section_start("What it enables", "Business Impact")
st.markdown("""
- Faster, more consistent underwriting decisions
- Improved risk segmentation ahead of pricing
- Better-supported, individualized premium pricing
- Expanded financial inclusion for underserved populations
""")
sa_section_end()

# =========================================================
# WHY THIS MATTERS
# =========================================================
sa_section_start("The bigger picture", "Why This Matters")
st.markdown("""
Traditional underwriting often depends on manual judgment and limited historical information.
Machine learning introduces consistency and scalability into that process — helping insurers
improve operational efficiency while supporting broader, fairer access to insurance for
populations that conventional models tend to leave out.
""")
sa_section_end()

# =========================================================
# TRY IT YOURSELF
# =========================================================
sa_section_start("Interactive demo", "Try It Yourself")
st.markdown(
    "A live, simplified version of the model — trained on synthetic data reflecting the same "
    "relationships as the original study, for demonstration only."
)

@st.cache_resource
def train_demo_model():
    np.random.seed(42)
    n = 800
    age_ = np.random.randint(18, 65, n)
    income_ = np.random.randint(3000, 60000, n)
    dependents_ = np.random.randint(0, 6, n)
    chronic_ = np.random.binomial(1, 0.18, n)
    edu_ = np.random.randint(0, 17, n)
    risk_score = (0.03*age_ - 0.00006*income_ + 0.25*dependents_ + 1.8*chronic_ - 0.05*edu_ + np.random.normal(0, 0.6, n))
    risk_band = (risk_score > np.median(risk_score)).astype(int)
    X = pd.DataFrame({"age": age_, "income": income_, "dependents": dependents_, "chronic_illness": chronic_, "education_years": edu_})
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X, risk_band)
    return model

demo_model = train_demo_model()

ic1, ic2, ic3 = st.columns(3)
with ic1:
    age_in = st.slider("Age", 18, 65, 30)
    income_in = st.number_input("Monthly income (KES)", 1000, 100000, 15000, step=500)
with ic2:
    dependents_in = st.slider("Dependants", 0, 8, 1)
    education_in = st.slider("Years of education", 0, 17, 8)
with ic3:
    chronic_in = st.radio("Chronic illness history?", ["No", "Yes"])
    st.write("")
    run = st.button("Calculate risk band & premium")

if run:
    chronic_val = 1 if chronic_in == "Yes" else 0
    row = pd.DataFrame([{"age": age_in, "income": income_in, "dependents": dependents_in, "chronic_illness": chronic_val, "education_years": education_in}])
    pred = demo_model.predict(row)[0]
    proba = demo_model.predict_proba(row)[0][pred]
    band = "High risk" if pred == 1 else "Low risk"
    band_class = "risk-high" if pred == 1 else "risk-low"
    premium = 150 if pred == 1 else 100
    r1, r2 = st.columns(2)
    r1.markdown(f'<div class="sa-card {band_class}"><h4>Risk classification</h4><div class="big">{band}</div><p>{proba*100:.1f}% model confidence</p></div>', unsafe_allow_html=True)
    r2.markdown(f'<div class="sa-card"><h4>Recommended daily premium</h4><div class="big">KES {premium}</div><p>Derived from risk classification.</p></div>', unsafe_allow_html=True)
sa_section_end()

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="text-align:center; padding: 24px 0;">
<span class="sa-eyebrow">Built by Smart Σ Actuary</span><br>
<span style="color:#C7D0DC;">Actuarial intelligence · Risk analytics · Predictive modeling</span><br><br>
<a href="https://mary-aringo-portfolio.netlify.app" style="color:#D9A441; font-weight:600;">Portfolio</a>
&nbsp;&nbsp;|&nbsp;&nbsp;
<a href="https://github.com/Valarithmetic" style="color:#D9A441; font-weight:600;">GitHub</a>
&nbsp;&nbsp;|&nbsp;&nbsp;
<a href="https://smartactuary.co.ke/#projects" style="color:#D9A441; font-weight:600;">More case studies</a>
&nbsp;&nbsp;|&nbsp;&nbsp;
<a href="https://smartactuary.co.ke/#contact" style="color:#D9A441; font-weight:600;">Contact</a>
</div>
""", unsafe_allow_html=True)
