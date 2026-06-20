import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="🏃 Aerofit Treadmill Customer Profiler",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Aerofit themed: Vibrant Orange-Red #FF4B2B & Sleek Charcoal)
st.markdown("""
<style>
    .reportview-container {
        background: #111111;
    }
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #FF4B2B;
        font-size: 38px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .accent-header {
        color: #f5f5f1;
        font-size: 16px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #1f1f1f;
        border-left: 5px solid #FF4B2B;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-card-grey {
        background-color: #1f1f1f;
        border-left: 5px solid #564d4d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-val {
        color: #f5f5f1;
        font-size: 28px;
        font-weight: 800;
    }
    .metric-lbl {
        color: #b3b3b3;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
</style>
""", unsafe_allowed_html=True)

# Data Loading (Cached for performance)
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'aerofit_treadmill.csv')
    return pd.read_csv(csv_path)

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading aerofit_treadmill.csv: {e}")
    st.stop()

# ── Sidebar Filter Setup ──────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/Running_icon.svg", width=60)
st.sidebar.title("🏃 Filter Customer Data")

# 1. Product Filter
products = sorted(df["Product"].unique())
selected_products = st.sidebar.multiselect("Treadmill Models", products, default=products)

# 2. Gender Filter
genders = sorted(df["Gender"].unique())
selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)

# 3. Marital Status Filter
marital_states = sorted(df["MaritalStatus"].unique())
selected_marital = st.sidebar.multiselect("Marital Status", marital_states, default=marital_states)

# 4. Income Slider
min_income = float(df["Income"].min())
max_income = float(df["Income"].max())
selected_income_range = st.sidebar.slider(
    "Annual Income ($)", 
    min_income, 
    max_income, 
    (min_income, max_income)
)

# Apply filters
df_filtered = df.copy()

if selected_products:
    df_filtered = df_filtered[df_filtered["Product"].isin(selected_products)]
if selected_genders:
    df_filtered = df_filtered[df_filtered["Gender"].isin(selected_genders)]
if selected_marital:
    df_filtered = df_filtered[df_filtered["MaritalStatus"].isin(selected_marital)]

df_filtered = df_filtered[
    (df_filtered["Income"] >= selected_income_range[0]) & 
    (df_filtered["Income"] <= selected_income_range[1])
]

# Page Header
st.markdown("<div class='main-header'>🏃 Aerofit Treadmill Customer Profiler</div>", unsafe_allowed_html=True)
st.markdown("<div class='accent-header'>Interactive segmentation analysis matching customer profiles to treadmill products</div>", unsafe_allowed_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sales = len(df_filtered)
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-val'>{total_sales:,}</div>
        <div class='metric-lbl'>Total Sales Volume</div>
    </div>
    """, unsafe_allowed_html=True)

with col2:
    avg_age = df_filtered["Age"].mean() if not df_filtered.empty else 0
    st.markdown(f"""
    <div class='metric-card-grey'>
        <div class='metric-val'>{avg_age:.1f} yrs</div>
        <div class='metric-lbl'>Average Customer Age</div>
    </div>
    """, unsafe_allowed_html=True)

with col3:
    avg_income = df_filtered["Income"].mean() if not df_filtered.empty else 0
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-val'>${avg_income:,.0f}</div>
        <div class='metric-lbl'>Average Annual Income</div>
    </div>
    """, unsafe_allowed_html=True)

with col4:
    avg_fitness = df_filtered["Fitness"].mean() if not df_filtered.empty else 0
    st.markdown(f"""
    <div class='metric-card-grey'>
        <div class='metric-val'>{avg_fitness:.1f} / 5.0</div>
        <div class='metric-lbl'>Avg Self-Rated Fitness</div>
    </div>
    """, unsafe_allowed_html=True)

st.markdown("<br>", unsafe_allowed_html=True)

# Tabs setup
tab1, tab2, tab3 = st.tabs(["📊 Customer Segments", "🧮 Dynamic Probability Calculator", "📂 Data Explorer & Recommendations"])

# ── TAB 1: Visual Insights ────────────────────────────────────────────────
with tab1:
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("🎯 Customer Demographics (Income vs Age clusters)")
        if not df_filtered.empty:
            fig_scatter = px.scatter(
                df_filtered,
                x="Age",
                y="Income",
                color="Product",
                size="Fitness",
                hover_data=["Education", "Miles", "Usage"],
                color_discrete_sequence=["#FF4B2B", "#564d4d", "#8CAAE6"],
                template="plotly_dark",
                labels={"Income": "Annual Income ($)"}
            )
            fig_scatter.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No customer data matches current filters.")
            
    with row1_col2:
        st.subheader("💰 Income Distribution by Treadmill Model")
        if not df_filtered.empty:
            fig_box = px.box(
                df_filtered,
                x="Product",
                y="Income",
                color="Product",
                color_discrete_sequence=["#FF4B2B", "#564d4d", "#8CAAE6"],
                template="plotly_dark",
                labels={"Income": "Annual Income ($)"}
            )
            fig_box.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No customer data matches current filters.")

    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.subheader("🏋️ Weekly Usage Patterns")
        if not df_filtered.empty:
            fig_hist = px.histogram(
                df_filtered,
                x="Usage",
                color="Product",
                barmode="group",
                color_discrete_sequence=["#FF4B2B", "#564d4d", "#8CAAE6"],
                template="plotly_dark",
                labels={"Usage": "Planned Runs/Week"}
            )
            fig_hist.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No data available.")
            
    with row2_col2:
        st.subheader("🔥 Self-Rated Fitness vs. Product Selection")
        if not df_filtered.empty:
            fitness_summary = df_filtered.groupby(["Fitness", "Product"]).size().reset_index(name="count")
            fig_bar = px.bar(
                fitness_summary,
                x="Fitness",
                y="count",
                color="Product",
                barmode="stack",
                color_discrete_sequence=["#FF4B2B", "#564d4d", "#8CAAE6"],
                template="plotly_dark"
            )
            fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No data available.")

# ── TAB 2: Dynamic Probability Calculator ──────────────────────────────────
with tab2:
    st.subheader("🧮 Interactive Conditional Probabilities Calculator")
    st.markdown("""
    This utility dynamically calculates the **conditional probability** of purchasing a specific treadmill model,
    given the customer demographics selected. This matches the Bayesian/Conditional Probability model used in the case study report.
    """)
    
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        prob_gender = st.selectbox("Customer Gender", ["All"] + list(df["Gender"].unique()), key="aer_prob_gen")
    with col_prob2:
        prob_marital = st.selectbox("Customer Marital Status", ["All"] + list(df["MaritalStatus"].unique()), key="aer_prob_mar")
        
    sub_df = df.copy()
    if prob_gender != "All":
        sub_df = sub_df[sub_df["Gender"] == prob_gender]
    if prob_marital != "All":
        sub_df = sub_df[sub_df["MaritalStatus"] == prob_marital]
        
    total_matching = len(sub_df)
    
    if total_matching > 0:
        counts = sub_df["Product"].value_counts()
        prob_rows = []
        for prod in ["KP281", "KP481", "KP781"]:
            cnt = counts.get(prod, 0)
            prob_rows.append({
                "Treadmill Model": prod,
                "Purchase Volume": cnt,
                "Conditional Probability (P(Model | Demo))": f"{(cnt / total_matching):.4%}",
                "Probability Decimal": cnt / total_matching
            })
            
        prob_res_df = pd.DataFrame(prob_rows)
        
        # Display table
        st.dataframe(prob_res_df[["Treadmill Model", "Purchase Volume", "Conditional Probability (P(Model | Demo))"]], use_container_width=True)
        
        # Render a simple plotly pie chart of probabilities
        fig_prob_pie = px.pie(
            prob_res_df,
            values="Probability Decimal",
            names="Treadmill Model",
            hole=0.4,
            color="Treadmill Model",
            color_discrete_map={"KP281": "#FF4B2B", "KP481": "#564d4d", "KP781": "#8CAAE6"},
            template="plotly_dark",
            title=f"Probability Distribution for Segment: Gender={prob_gender}, Marital Status={prob_marital}"
        )
        fig_prob_pie.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_prob_pie, use_container_width=True)
        
    else:
        st.warning("No data points found matching the selected filters.")

# ── TAB 3: Data Explorer & Recommendations ────────────────────────────────
with tab3:
    # Strategic Insights Expander
    st.markdown("### 💡 Demographics Targeting & Strategic Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.markdown("""
        #### Key Findings
        1. **KP281 (Entry-Level)**: Standard product purchased by budget-conscious users of both genders. Income ranges are generally below $50k.
        2. **KP481 (Mid-Range)**: Moderately fitness-oriented users. Purchases are distributed evenly, but users run fewer miles compared to KP781.
        3. **KP781 (Premium)**: Highly concentrated segment. Purchased almost exclusively by customers with high incomes (>$75,000), self-rated fitness of 5, and weekly usage of 4+ times.
        """)
        
    with rec_col2:
        st.markdown("""
        #### Actionable Recommendations
        - **Premium Model Marketing (KP781)**: Target corporate executives and high-income athletic segments. Advertise features like advanced metrics tracking and structural durability.
        - **Budget Promotion (KP281)**: Launch seasonal discounts or EMI plans targeting younger cohorts and students.
        - **Cross-selling**: Target mid-level (KP481) users with upgrading benefits as their fitness rating improves from 3 to 4.
        """)
        
    st.markdown("---")
    st.markdown("#### 📂 Customer Dataset Explorer")
    st.write(f"Displaying **{len(df_filtered):,}** customer records matching filters.")
    
    st.dataframe(df_filtered, use_container_width=True)
    
    # Download Button
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="aerofit_filtered_customers.csv",
        mime="text/csv",
    )
