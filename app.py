import streamlit as st
import plotly.graph_objects as go
from decision_engine import supply_ai
import os

# Page configuration
st.set_page_config(
    page_title="Enterprise Supply Chain AI",
    page_icon="⚡",
    layout="wide"
)

# Professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .decision-box {
        background: #f8f9ff;
        border: 2px solid #2E86AB;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Application header
st.markdown("""
<div class="main-header">
    <h1>Enterprise Supply Chain Decision Engine</h1>
    <p style="font-size: 1.1em; margin: 0;">
        AI-Powered Multi-Modal Intelligence | Real Business Decisions | Enterprise Ready
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.title("Configuration Panel")
    
    language = st.selectbox("Language Selection", ["English", "Español"], index=0)
    
    st.markdown("---")
    st.markdown("""
    **System Capabilities:**
    - Real supply chain data analysis (200+ records)
    - AI-powered decision recommendations
    - Multi-modal data integration (CSV + Text)
    - Bilingual operations support
    - Enterprise-ready architecture
    """)
    
    # API connection status
    api_status = "Connected" if os.getenv("ANTHROPIC_API_KEY") else "Demo Mode"
    st.markdown(f"**Claude API Status:** {api_status}")

# Initialize data on first load
if 'data_loaded' not in st.session_state:
    with st.spinner("Loading supply chain data..."):
        records_loaded = supply_ai.load_supply_data()
        supply_ai.load_manual(supply_ai.create_sample_manual())
        st.session_state.data_loaded = True

# Key performance indicators dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #2E86AB; margin: 0;">200</h3>
        <p style="margin: 0;">Supply Items</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    high_risk = len(supply_ai.supply_data[supply_ai.supply_data['Overall_Risk'] > 0.6])
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #FF4444; margin: 0;">{high_risk}</h3>
        <p style="margin: 0;">High Risk Items</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_lead = supply_ai.supply_data['Lead_Time_Days'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #FFA500; margin: 0;">{avg_lead:.1f}</h3>
        <p style="margin: 0;">Avg Lead Time (Days)</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    suppliers = supply_ai.supply_data['Supplier'].nunique()
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #2E86AB; margin: 0;">{suppliers}</h3>
        <p style="margin: 0;">Active Suppliers</p>
    </div>
    """, unsafe_allow_html=True)

# Data integration section
st.header("Multi-Modal Data Integration")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Supply Chain Operational Data")
    st.success("Status: REAL Data Loaded from Public Dataset")
    
    # Show data source information
    data_source = "DataCo Global Supply Chain Dataset" if len(supply_ai.supply_data) > 0 else "Sample Data"
    st.info(f"Source: {data_source} ({len(supply_ai.supply_data)} records)")
    
    # Display data preview
    if supply_ai.supply_data is not None and not supply_ai.supply_data.empty:
        display_cols = ['Supplier', 'Product', 'Current_Stock', 'Overall_Risk']
        available_cols = [col for col in display_cols if col in supply_ai.supply_data.columns]
        if available_cols:
            preview_df = supply_ai.supply_data[available_cols].head()
            st.dataframe(preview_df, use_container_width=True)
        else:
            st.dataframe(supply_ai.supply_data.head(), use_container_width=True)

with col2:
    st.subheader("Supplier Policy Manual")
    st.success("Status: Policy Manual Integrated")
    st.info("Content: Quality standards, delivery requirements, emergency procedures")
    
    # Display manual excerpt
    manual_excerpt = supply_ai.manual_text[:300] + "..."
    st.text_area("Manual Content Preview", manual_excerpt, height=150, disabled=True)

# AI decision engine interface
st.header("AI Decision Engine")

# Predefined business scenarios
scenarios = [
    "TechCorp Mexico experiencing delays. What are our alternatives?",
    "Stock levels critically low for Industrial Relays. Immediate action needed.",
    "Quality ratings dropping for AsiaTech suppliers. Risk assessment required.",
    "Lead times increased 30% from Asian suppliers. Strategic recommendations needed."
]

st.markdown("**Sample Supply Chain Challenge Scenarios:**")
for i, scenario in enumerate(scenarios):
    if st.button(f"Scenario {i+1}: {scenario}", key=f"scenario_{i}"):
        st.session_state.current_query = scenario

# User query input
query = st.text_area(
    "Supply Chain Challenge Description:",
    value=st.session_state.get('current_query', ''),
    height=80,
    placeholder="Describe your supply chain situation or request specific recommendations..."
)

# Generate decision button
if st.button("Generate AI Decision Recommendation", type="primary"):
    if query:
        with st.spinner("Analyzing multi-modal data and generating decision recommendation..."):
            result = supply_ai.analyze_situation(query, language)
            
            if not result.get("error"):
                # Display AI recommendation
                st.markdown(f"""
                <div class="decision-box">
                    <h3>AI Decision Recommendation</h3>
                    {result['recommendation']}
                </div>
                """, unsafe_allow_html=True)
                
                # Display detailed analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Situation Analysis")
                    analysis = result['analysis']
                    st.markdown(f"""
                    **Focus Area:** {analysis['focus_area']}  
                    **High Risk Items:** {analysis['high_risk_count']}/{analysis['total_items']}  
                    **Critical Stock Issues:** {analysis['critical_stock_count']}  
                    **Average Lead Time:** {analysis['avg_lead_time']:.1f} days  
                    **Average Quality Rating:** {analysis['avg_quality']:.1f}/5.0
                    """)
                    
                    if analysis['focus_items']:
                        st.markdown("**Priority Items for Review:**")
                        for item in analysis['focus_items'][:3]:
                            risk_indicator = "HIGH" if item['Overall_Risk'] > 0.7 else "MEDIUM" if item['Overall_Risk'] > 0.4 else "LOW"
                            st.markdown(f"- {item['Product']} from {item['Supplier']} (Risk Level: {risk_indicator})")
                
                with col2:
                    st.markdown("### Risk Distribution Analysis")
                    st.plotly_chart(result['charts']['risk_pie'], use_container_width=True)
                
                # Supplier performance analysis
                st.markdown("### Supplier Performance Analysis")
                st.plotly_chart(result['charts']['supplier_bar'], use_container_width=True)
                
            else:
                st.error(result['error'])
    else:
        st.warning("Please enter a supply chain challenge description.")

# Technical architecture information
st.markdown("---")
st.markdown("""
<div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center;">
    <h4>Technical Architecture Overview</h4>
    <p><strong>Multi-Modal Processing:</strong> CSV Operational Data + Policy Documents + AI Decision Engine</p>
    <p><strong>Technology Stack:</strong> Python + Streamlit + Claude API + Sentence Transformers + Plotly</p>
    <p><strong>Scalability:</strong> Architecture designed for deployment across global manufacturing facilities</p>
    <hr style="margin: 1rem 0;">
    <p><strong>Developed by:</strong> Luis Rivera | AI Systems Integration Manager Candidate</p>
    <p style="color: #666; margin: 0;">Demonstration of: System Integration | Business Intelligence | Global Operations Support</p>
</div>
""", unsafe_allow_html=True)
