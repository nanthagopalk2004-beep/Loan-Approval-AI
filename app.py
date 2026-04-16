import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# Page config
st.set_page_config(page_title="CredFlow AI - Loan Finance", page_icon="💳", layout="wide")

# Load model and explainer
clf = joblib.load("loan_approval_model.pkl")
try:
    bg_data = joblib.load("shap_background.pkl")
except FileNotFoundError:
    bg_data = None

# Encoding dictionaries
GENDER_MAP = {"Female": 0, "Male": 1}
EDUCATION_MAP = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}
HOME_OWNERSHIP_MAP = {"Rent": 0, "Own": 1, "Mortgage": 2, "Other": 3}
LOAN_INTENT_MAP = {"Personal": 0, "Education": 1, "Medical": 2, "Venture": 3, "Home Improvement": 4, "Debt Consolidation": 5}
DEFAULT_MAP = {"No": 0, "Yes": 1}

FEATURE_EXPLANATIONS = {
    "credit_score": {"positive": "your strong credit history", "negative": "your current credit history"},
    "loan_percent_income": {"positive": "a manageable loan amount relative to your income", "negative": "a high loan amount compared to your income"},
    "person_income": {"positive": "your income level", "negative": "your current income level"},
    "person_home_ownership": {"positive": "stable housing status", "negative": "housing stability considerations"},
    "previous_loan_defaults_on_file": {"positive": "a good repayment record", "negative": "past repayment history"},
    "person_emp_exp": {"positive": "consistent employment experience", "negative": "limited employment experience"}
}

IMPROVEMENT_SUGGESTIONS = {
    "credit_score": "Maintaining timely repayments and reducing outstanding balances may improve future outcomes.",
    "loan_percent_income": "Reducing the loan amount or increasing income can improve eligibility.",
    "person_income": "A higher verified income may improve future decisions.",
    "person_home_ownership": "Long-term housing stability may positively influence future applications.",
    "previous_loan_defaults_on_file": "Consistent repayment behavior over time can improve eligibility."
}

# Professional CSS for Pylon-style design
PYLON_CSS = """
:root {
    --primary: #1e3a5f;
    --secondary: #0066cc;
    --accent: #ff7a00;
    --light: #f5f7fa;
    --muted: #6b7280;
    --border: #e5e7eb;
    --success: #10b981;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #f9fafb; font-family: 'Segoe UI', Arial, sans-serif; color: #1f2937; }
h1, h2, h3, h4, h5, h6 { color: var(--primary); font-weight: 700; }

.stApp { background: #f9fafb; }
.main { background: #f9fafb; padding: 0; }

/* Header */
.header-top { background: var(--primary); color: white; padding: 20px 0; text-align: center; }
.logo { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
.tagline { font-size: 14px; opacity: 0.9; }

/* Navigation */
.nav-links { display: flex; justify-content: center; gap: 24px; padding: 16px 0; background: white; border-bottom: 1px solid var(--border); margin-bottom: 32px; flex-wrap: wrap; }
.nav-links a { text-decoration: none; color: var(--primary); font-weight: 600; font-size: 14px; transition: color 0.3s; }
.nav-links a:hover { color: var(--accent); }
.nav-links a.active { color: var(--secondary); border-bottom: 3px solid var(--secondary); padding-bottom: 8px; }

/* Hero Section */
.hero { 
    background: linear-gradient(135deg, var(--primary) 0%, #2a4d7a 100%);
    color: white; padding: 80px 40px; text-align: center; margin-bottom: 48px;
}
.hero h1 { font-size: 48px; margin-bottom: 16px; }
.hero p { font-size: 18px; opacity: 1; margin-bottom: 32px; max-width: 600px; margin-left: auto; margin-right: auto; text-shadow: 0 2px 6px rgba(0,0,0,0.25); color: white !important; }
.hero-cta { 
    background: var(--accent); color: white; padding: 14px 32px; border-radius: 8px;
    font-weight: 600; text-decoration: none; display: inline-block; transition: all 0.3s;
}
.hero-cta:hover { background: #ff6b00; transform: translateY(-2px); box-shadow: 0 8px 20px rgba(255, 122, 0, 0.3); }

/* Hero Stats */
.hero-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 24px; margin-top: 40px; }
.stat-box { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px; backdrop-filter: blur(10px); color: white; }
.stat-number { font-size: 32px; font-weight: 700; color: white; }
.stat-label { font-size: 13px; opacity: 0.9; margin-top: 8px; color: white; }

/* Service Cards */
.services-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 24px; margin-bottom: 48px; }
.service-card { 
    background: white; padding: 32px 24px; border-radius: 12px; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.08); transition: all 0.3s;
    border-top: 4px solid var(--primary);
}
.service-card:hover { 
    transform: translateY(-8px); 
    box-shadow: 0 12px 28px rgba(0,0,0,0.12);
    border-top-color: var(--accent);
}
.service-icon { font-size: 36px; margin-bottom: 16px; }
.service-card h3 { margin-bottom: 12px; font-size: 18px; }
.service-card p { color: var(--muted); font-size: 14px; line-height: 1.6; }

/* Testimonial Cards */
.testimonials-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; margin-bottom: 48px; }
.testimonial-card { 
    background: white; padding: 28px; border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border-left: 4px solid var(--accent);
}
.testimonial-text { color: var(--muted); font-size: 14px; font-style: italic; margin-bottom: 16px; line-height: 1.6; }
.testimonial-author { font-weight: 600; color: var(--primary); font-size: 14px; }
.testimonial-role { color: var(--accent); font-size: 12px; }
.testimonial-stars { color: #fbbf24; font-size: 13px; margin-bottom: 8px; }

/* Process Steps */
.process-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 24px; margin-bottom: 48px; }
.step-box { text-align: center; position: relative; }
.step-number { 
    display: inline-flex; align-items: center; justify-content: center;
    width: 50px; height: 50px; background: var(--secondary); color: white;
    border-radius: 50%; font-weight: 700; font-size: 20px; margin-bottom: 16px;
}
.step-box h4 { margin-bottom: 8px; font-size: 16px; color: var(--primary); }
.step-box p { color: var(--muted); font-size: 13px; }

/* Calculator Section */
.calculator-box { background: white; padding: 32px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 48px; color: var(--primary); }
.calculator-box h2 { margin-bottom: 24px; color: var(--primary); }
.calc-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
.calc-result { background: var(--light); padding: 20px; border-radius: 8px; border-left: 4px solid var(--secondary); }
.calc-result-value { font-size: 28px; font-weight: 700; color: var(--secondary); }
.calc-result-label { color: var(--muted); font-size: 12px; margin-top: 4px; }

/* Stats Section */
.stats-section { background: var(--primary); color: white; padding: 60px 40px; margin: 48px 0; border-radius: 12px; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 32px; }
.stat-box-large { text-align: center; }
.stat-box-large .number { font-size: 40px; font-weight: 700; }
.stat-box-large .label { font-size: 14px; opacity: 0.85; margin-top: 8px; }

/* Section Headers */
.section-header { text-align: center; margin-bottom: 48px; }
.section-header h2 { font-size: 36px; margin-bottom: 12px; color: var(--primary); }
.section-header p { color: var(--muted); font-size: 16px; max-width: 500px; margin: 0 auto; }

/* Apply Form */
.apply-form { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 48px; color: var(--primary); }
.form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 24px; }
.apply-form h2 { color: var(--primary); }
.apply-form label { color: var(--primary); font-weight: 600; }

/* Footer */
.footer { background: var(--primary); color: white; padding: 60px 40px; margin-top: 80px; }
.footer-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 40px; margin-bottom: 40px; }
.footer-section h4 { margin-bottom: 16px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
.footer-links { display: flex; flex-direction: column; gap: 8px; }
.footer-links a { color: rgba(255,255,255,0.7); text-decoration: none; font-size: 13px; transition: color 0.3s; }
.footer-links a:hover { color: var(--accent); }
.footer-bottom { border-top: 1px solid rgba(255,255,255,0.1); padding-top: 24px; text-align: center; font-size: 13px; opacity: 0.7; }

/* Responsive */
@media (max-width: 768px) {
    .hero { padding: 40px 20px; }
    .hero h1 { font-size: 32px; }
    .nav-links { gap: 12px; }
    .services-grid, .testimonials-grid, .process-steps { grid-template-columns: 1fr; }
}

/* Fix text colors on light backgrounds */
.stMarkdown, .stMetric, .stSelectbox, .stNumberInput { color: var(--primary) !important; }
.stMarkdown p, .stMarkdown li { color: var(--primary) !important; }
.stForm { background: transparent !important; box-shadow: none !important; }

/* Ensure hero paragraph stays white even if Streamlit's .stMarkdown rules apply */
.stMarkdown .hero p, .hero p { color: white !important; }

/* Global button styling to give nav and CTAs a modern look */
.stButton>button {
    background: white !important;
    color: var(--primary) !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    margin: 0 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 6px 18px rgba(16,24,40,0.08) !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    transition: all 0.18s ease !important;
}
.stButton>button:hover {
    background: var(--secondary) !important;
    color: white !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 24px rgba(16,24,40,0.12) !important;
}
.stButton>button:focus {
    outline: 2px solid rgba(0,102,204,0.12) !important;
}

/* Make hero CTA more prominent (it uses a Streamlit button) */
.stButton>button[key="hero_cta"] {
    background: var(--accent) !important;
    color: white !important;
}

/* Make forms and calculator area transparent and interactive to avoid blocking scroll/clicks */
.apply-form, .calculator-box {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}
.apply-form *, .calculator-box * {
    pointer-events: auto !important;
}
.stApp, .main { overflow: auto !important; }
"""

def generate_customer_message(shap_df: pd.DataFrame, prediction: int) -> str:
    positives, negatives, suggestions = [], [], []
    for _, row in shap_df.head(3).iterrows():
        f = row["feature"]
        v = row["shap_value"]
        if f not in FEATURE_EXPLANATIONS:
            continue
        if v > 0:
            positives.append(FEATURE_EXPLANATIONS[f]["positive"])
        else:
            negatives.append(FEATURE_EXPLANATIONS[f]["negative"])
            if f in IMPROVEMENT_SUGGESTIONS:
                suggestions.append(IMPROVEMENT_SUGGESTIONS[f])
    if prediction == 1:
        header = "✅ Loan Decision: Approved"
        summary = "Your loan application was approved after reviewing multiple financial factors."
    else:
        header = "❌ Loan Decision: Not Approved"
        summary = "Your loan application could not be approved at this time."
    msg = f"{header}\n\n{summary}\n\n"
    if positives:
        msg += "### Positive factors\n"
        for p in positives:
            msg += f"- {p.capitalize()}\n"
    if negatives:
        msg += "\n### Factors that limited the decision\n"
        for n in negatives:
            msg += f"- {n.capitalize()}\n"
    if suggestions:
        msg += "\n### How you may improve\n"
        for s in set(suggestions):
            msg += f"- {s}\n"
    return msg

def calculate_recommended_loan(income: float, credit_score: float, employment_exp: float, dti_ratio: float) -> float:
    base_multiplier = 0.25
    credit_factor = 1.2 if credit_score >= 750 else (1.0 if credit_score >= 650 else 0.7)
    stability_factor = 1.1 if employment_exp >= 5 else (1.0 if employment_exp >= 2 else 0.8)
    dti_factor = 1.2 if dti_ratio < 0.2 else (1.0 if dti_ratio < 0.35 else 0.7)
    final_multiplier = base_multiplier * credit_factor * stability_factor * dti_factor
    return round(income * final_multiplier, 2)

# Session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Apply CSS
st.markdown(f"<style>{PYLON_CSS}</style>", unsafe_allow_html=True)

# Header
st.markdown("""
<div class='header-top'>
    <div class='logo'>🏦 CredFlow AI</div>
    <div class='tagline'>Building A Brighter Financial Future</div>
</div>
""", unsafe_allow_html=True)

# Navigation (use Streamlit buttons for reliable interaction)
def render_nav():
    nav_items = ["Home", "Apply", "Loan Calculator", "About"]
    # Create outer columns so the nav group sits centered on the page.
    # Middle column will contain the actual nav buttons, keeping them grouped together.
    outer = st.columns([1, 3, 1])
    with outer[1]:
        btn_cols = st.columns(len(nav_items))
        for i, item in enumerate(nav_items):
            # use compact buttons and stable keys so styling/behavior remains consistent
            if btn_cols[i].button(item, key=f"nav_{item}", use_container_width=False):
                st.session_state.current_page = item

render_nav()

# Page routing with buttons in sidebar
st.sidebar.markdown("### Navigation")
if st.sidebar.button("🏠 Home", use_container_width=True):
    st.session_state.current_page = "Home"
if st.sidebar.button("📝 Apply for Loan", use_container_width=True):
    st.session_state.current_page = "Apply"
if st.sidebar.button("🧮 Loan Calculator", use_container_width=True):
    st.session_state.current_page = "Loan Calculator"
if st.sidebar.button("ℹ️ About Us", use_container_width=True):
    st.session_state.current_page = "About"

# ===== HOME PAGE =====
if st.session_state.current_page == "Home":
    # Hero
    st.markdown("""
    <div class='hero'>
        <h1>Building A Brighter Financial Future</h1>
        <p>Simple & Secure Payment Process. Get your loan approved in minutes with our AI-powered platform.</p>
        <div class='hero-stats'>
            <div class='stat-box'><div class='stat-number'>45%</div><div class='stat-label'>Faster Approval</div></div>
            <div class='stat-box'><div class='stat-number'>99%</div><div class='stat-label'>Secure Platform</div></div>
            <div class='stat-box'><div class='stat-number'>24h</div><div class='stat-label'>Quick Turnaround</div></div>
            <div class='stat-box'><div class='stat-number'>50K+</div><div class='stat-label'>Happy Customers</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Apply Now →", key="hero_cta", use_container_width=False):
        st.session_state.current_page = "Apply"
        st.experimental_rerun()
    
    st.markdown("\n\n")
    
    # Services Section
    st.markdown("""
    <div class='section-header'>
        <h2>All Loan Services</h2>
        <p>We offer a variety of loan products tailored to meet your financial needs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    services = [
        ("💰", "Personal Loan", "Quick and flexible personal loans for any purpose with competitive rates."),
        ("🏠", "Home Loan", "Affordable home loans with flexible repayment options to own your dream home."),
        ("🚗", "Auto Loan", "Finance your vehicle with low interest rates and hassle-free approval."),
        ("📚", "Education Loan", "Support your studies with education loans featuring favorable terms."),
        ("💼", "Business Loan", "Growth opportunities for small and large businesses with quick disbursal."),
        ("💍", "Wedding Loan", "Make your special day memorable with our dedicated wedding loan products."),
    ]
    
    cols = st.columns(3)
    for idx, (icon, title, desc) in enumerate(services):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class='service-card'>
                <div class='service-icon'>{icon}</div>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("\n\n")
    
    # Trust Stats
    st.markdown("""
    <div class='stats-section'>
        <h2 style='text-align: center; color: white; margin-bottom: 40px;'>Why Choose Us?</h2>
        <div class='stats-grid'>
            <div class='stat-box-large'>
                <div class='number'>98%</div>
                <div class='label'>Approval Rate</div>
            </div>
            <div class='stat-box-large'>
                <div class='number'>₹50Cr</div>
                <div class='label'>Disbursed</div>
            </div>
            <div class='stat-box-large'>
                <div class='number'>50K+</div>
                <div class='label'>Happy Customers</div>
            </div>
            <div class='stat-box-large'>
                <div class='number'>100+</div>
                <div class='label'>Staff Members</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("\n\n")
    
    # How It Works
    st.markdown("""
    <div class='section-header'>
        <h2>How It Works</h2>
        <p>Our simple 4-step process gets you approved quickly.</p>
    </div>
    """, unsafe_allow_html=True)
    
    steps = [
        ("1", "Submit Details", "Fill in your basic information and loan requirements."),
        ("2", "AI Assessment", "Our AI analyzes your eligibility instantly."),
        ("3", "Verification", "Quick document verification process."),
        ("4", "Disbursal", "Get approved and receive funds via bank transfer."),
    ]
    
    cols = st.columns(4)
    for idx, (num, title, desc) in enumerate(steps):
        with cols[idx]:
            st.markdown(f"""
            <div class='step-box'>
                <div class='step-number'>{num}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("\n\n")
        
    st.markdown("""
    <div class='section-header'>
        <h2>Resources & Guidance</h2>
        <p>Helpful tips and guides to improve your loan eligibility and financial health.</p>
    </div>
    """, unsafe_allow_html=True)

    resources = [
        ("Improve Your Credit", "Simple steps to raise your credit score over time."),
        ("Choose The Right Loan", "Understand which loan product suits your needs."),
        ("Document Checklist", "Prepare documents to speed up verification."),
        ("Contact Support", "We're here to help — reach out anytime."),
    ]
    cols = st.columns(4)
    for idx, (title, desc) in enumerate(resources):
        with cols[idx % 4]:
            st.markdown(f"""
            <div class='service-card'>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ===== APPLY PAGE =====
elif st.session_state.current_page == "Apply":
    st.markdown("""
    <div class='section-header'>
        <h2>Apply for a Loan</h2>
        <p>Fill in your details below and get instant AI-powered loan eligibility assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    # Helpful note to avoid large empty boxes when Streamlit layout recalculates
    st.info("Please fill in all required fields in the form below. If a field is missing, check the model requirements.")

    st.markdown("""<div class='apply-form'>""", unsafe_allow_html=True)
    
    with st.form("loan_application"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 100, 30)
            gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
            education = st.selectbox("Education Level", list(EDUCATION_MAP.keys()))
            income = st.number_input("Annual Income (₹)", 0, 1_000_000, 50_000, step=1000, format="%d")
            emp_exp = st.number_input("Years Employed", 0, 50, 5)
        
        with col2:
            home = st.selectbox("Home Ownership", list(HOME_OWNERSHIP_MAP.keys()))
            loan_amount = st.number_input("Requested Loan (₹)", 500, 500_000, 10_000, step=500, format="%d")
            intent = st.selectbox("Loan Purpose", list(LOAN_INTENT_MAP.keys()))
            interest = st.number_input("Interest Rate (%)", 0.0, 40.0, 10.0, step=0.1)
            credit_score = st.number_input("Credit Score", 300, 900, 650)
        
        loan_pct = st.slider("Loan as % of Income", 0.0, 1.0, 0.2, step=0.01)
        credit_len = st.number_input("Credit History Length (years)", 0, 50, 5)
        prev_default = st.selectbox("Previous Loan Default", list(DEFAULT_MAP.keys()))
        
        submitted = st.form_submit_button("Submit Application →", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submitted:
        dti_ratio = loan_amount / income if income else 0.0
        input_dict = {}
        for feat in clf.feature_names_in_:
            if feat == "person_age":
                input_dict[feat] = age
            elif feat == "person_gender":
                input_dict[feat] = GENDER_MAP[gender]
            elif feat == "person_education":
                input_dict[feat] = EDUCATION_MAP[education]
            elif feat == "person_income":
                input_dict[feat] = income
            elif feat == "person_emp_exp":
                input_dict[feat] = emp_exp
            elif feat == "person_home_ownership":
                input_dict[feat] = HOME_OWNERSHIP_MAP[home]
            elif feat == "loan_amnt":
                input_dict[feat] = loan_amount
            elif feat == "loan_intent":
                input_dict[feat] = LOAN_INTENT_MAP[intent]
            elif feat == "loan_int_rate":
                input_dict[feat] = interest
            elif feat == "loan_percent_income":
                input_dict[feat] = loan_pct
            elif feat == "cb_person_cred_hist_length":
                input_dict[feat] = credit_len
            elif feat == "credit_score":
                input_dict[feat] = credit_score
            elif feat == "previous_loan_defaults_on_file":
                input_dict[feat] = DEFAULT_MAP[prev_default]
            elif feat == "dti_ratio":
                input_dict[feat] = dti_ratio
            else:
                input_dict[feat] = 0
        
        X_user = pd.DataFrame([input_dict])
        
        explainer = shap.TreeExplainer(clf, data=bg_data) if bg_data is not None else shap.TreeExplainer(clf)
        pred = clf.predict(X_user)[0]
        prob = clf.predict_proba(X_user)[0, 1]
        recommended_amount = calculate_recommended_loan(income, credit_score, emp_exp, dti_ratio)
        
        shap_values = explainer.shap_values(X_user)
        if isinstance(shap_values, list):
            shap_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_arr = shap_values
        shap_arr = np.array(shap_arr).ravel()
        ncols = len(X_user.columns)
        if shap_arr.size < ncols:
            shap_arr = np.concatenate([shap_arr, np.zeros(ncols - shap_arr.size)])
        elif shap_arr.size > ncols:
            shap_arr = shap_arr[:ncols]
        
        shap_df = pd.DataFrame({"feature": X_user.columns, "shap_value": shap_arr}).sort_values("shap_value", key=abs, ascending=False)
        
        st.markdown(generate_customer_message(shap_df, pred))
        st.metric("Approval Probability", f"{prob:.1%}")
        
        if pred == 1:
            st.success(f"💰 Recommended Loan Amount: ₹{recommended_amount:,.0f}")
        else:
            st.info(f"Try a lower amount like ₹{recommended_amount:,.0f} for better chances.")
        
        with st.expander("📊 Feature Impact Analysis"):
            st.bar_chart(shap_df.set_index("feature").shap_value)

# ===== LOAN CALCULATOR PAGE =====
elif st.session_state.current_page == "Loan Calculator":
    st.markdown("""
    <div class='section-header'>
        <h2>Loan Calculator</h2>
        <p>Calculate your estimated monthly payment and total interest.</p>
    </div>
    """, unsafe_allow_html=True)

    # Short description to occupy space instead of an empty white box
    st.write("Use the calculator below to estimate your monthly EMI and total interest. Adjust amount, rate and tenure to see changes.")

    st.markdown("""<div class='calculator-box'>""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amount = st.number_input("Loan Amount (₹)", 10000, 10000000, 500000, step=10000)
    with col2:
        interest_rate = st.number_input("Annual Interest Rate (%)", 0.1, 30.0, 10.0, step=0.1)
    with col3:
        loan_tenure = st.number_input("Loan Tenure (Months)", 6, 360, 60, step=6)
    
    # EMI Calculation
    monthly_rate = interest_rate / 100 / 12
    if monthly_rate == 0:
        emi = loan_amount / loan_tenure
    else:
        emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** loan_tenure) / ((1 + monthly_rate) ** loan_tenure - 1)
    
    total_amount = emi * loan_tenure
    total_interest = total_amount - loan_amount
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='calc-result'>
            <div class='calc-result-value'>₹{emi:,.0f}</div>
            <div class='calc-result-label'>Monthly EMI</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='calc-result'>
            <div class='calc-result-value'>₹{total_interest:,.0f}</div>
            <div class='calc-result-label'>Total Interest</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='calc-result'>
            <div class='calc-result-value'>₹{total_amount:,.0f}</div>
            <div class='calc-result-label'>Total Amount</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("\n\n")
    
    # Amortization Table
    st.subheader("Amortization Schedule")
    schedule = []
    balance = loan_amount
    for month in range(1, min(loan_tenure + 1, 61)):
        interest = balance * monthly_rate
        principal = emi - interest
        balance -= principal
        schedule.append({"Month": month, "EMI": emi, "Principal": principal, "Interest": interest, "Balance": max(0, balance)})
    
    schedule_df = pd.DataFrame(schedule)
    st.dataframe(schedule_df.style.format({"EMI": "₹{:,.0f}", "Principal": "₹{:,.0f}", "Interest": "₹{:,.0f}", "Balance": "₹{:,.0f}"}), use_container_width=True)

# ===== ABOUT PAGE =====
elif st.session_state.current_page == "About":
    st.markdown("""
    <div class='section-header'>
        <h2>About CredFlow AI</h2>
        <p>We are India's leading AI-powered loan platform.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## Our Mission
    
    To make loans accessible, transparent, and affordable for every Indian by leveraging cutting-edge AI technology.
    
    ## Why Choose CredFlow AI?
    
    - **Fast Approvals**: Get approved in minutes, not days
    - **Fair Pricing**: Competitive rates based on AI assessment
    - **Transparent Process**: No hidden charges or surprises
    - **24/7 Support**: Round-the-clock customer assistance
    - **Secure Platform**: Bank-grade security for your data
    
    ## Our Commitment
    
    We believe in empowering individuals and businesses with financial freedom through innovative solutions.
    """)
    
    st.markdown("\n\n")
    
    # Contact Info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📞 Phone\n+91 9087705080")
    with col2:
        st.markdown("### 📧 Email\nsmmusthaba04@gmail.com")
    with col3:
        st.markdown("### 🏢 Address\nTamil Nadu, India")

# Footer
st.markdown("""
<div class='footer'>
    <div class='footer-grid'>
        <div class='footer-section'>
            <h4>Company</h4>
            <div class='footer-links'>
                <a href='#'>About Us</a>
                <a href='#'>Careers</a>
                <a href='#'>Blog</a>
                <a href='#'>Contact</a>
            </div>
        </div>
        <div class='footer-section'>
            <h4>Services</h4>
            <div class='footer-links'>
                <a href='#'>Personal Loan</a>
                <a href='#'>Home Loan</a>
                <a href='#'>Auto Loan</a>
                <a href='#'>Business Loan</a>
            </div>
        </div>
        <div class='footer-section'>
            <h4>Legal</h4>
            <div class='footer-links'>
                <a href='#'>Privacy Policy</a>
                <a href='#'>Terms & Conditions</a>
                <a href='#'>Disclaimer</a>
            </div>
        </div>
        <div class='footer-section'>
            <h4>Connect</h4>
            <div class='footer-links'>
                <a href='#'>Facebook</a>
                <a href='#'>Twitter</a>
                <a href='#'>LinkedIn</a>
                <a href='#'>Instagram</a>
            </div>
        </div>
    </div>
    <div class='footer-bottom'>
        <p>© 2026 Pylon Finance. All rights reserved. | Simple • Transparent • Secure</p>
    </div>
</div>
""", unsafe_allow_html=True)
