import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats


# ---------- Configuration ----------
DEFAULT_COUNTRIES = ["Germany", "United States", "China", "India", "Australia"]
FORECAST_YEARS = 10
ROLLING_WINDOW = 5
PARIS_TARGET_REDUCTION = 45  # 45% reduction by 2030 from 2010 levels


# ---------- Page config ----------
st.set_page_config(
    page_title="CO₂ Emissions Intelligence Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-blue: #003D7A;
        --secondary-blue: #0076C0;
        --accent-teal: #00A3A1;
    }
    
    /* Executive summary banner */
    .exec-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
    }
    
    /* Insight boxes */
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Headers */
    h1 {
        color: var(--primary-blue);
        font-weight: 700;
    }
    
    h3 {
        color: var(--secondary-blue);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Data Loading ----------
@st.cache_data
def load_data():
    """Load and preprocess CO₂ emissions data with enhanced features."""
    df = pd.read_csv("co2_emissions.csv")
    
    df = df.rename(columns={
        "Entity": "Country",
        "Year": "Year",
        "Annual CO₂ emissions (per capita)": "CO2"
    })
    
    df = df[["Country", "Year", "CO2"]]
    df_filtered = df[df["Country"].isin(DEFAULT_COUNTRIES)].copy()
    df_filtered = df_filtered.sort_values(["Country", "Year"])
    
    # Calculate year-over-year percentage change
    df_filtered["CO2_change_pct"] = (
        df_filtered.groupby("Country")["CO2"].pct_change() * 100
    )
    
    return df_filtered


# ---------- Advanced Calculation Functions ----------
def calculate_cagr(df, first_year, last_year):
    """Calculate compound annual growth rate."""
    first_value = df[df["Year"] == first_year]["CO2"].mean()
    last_value = df[df["Year"] == last_year]["CO2"].mean()
    
    if first_value > 0 and last_year > first_year:
        return ((last_value / first_value) ** (1 / (last_year - first_year)) - 1) * 100
    return 0


def calculate_yoy_change(df):
    """Calculate year-over-year percentage change."""
    latest = df.sort_values("Year").groupby("Country").tail(1)
    prev = df.sort_values("Year").groupby("Country").nth(-2)
    
    if len(prev) > 0 and prev["CO2"].mean() != 0:
        return ((latest["CO2"].mean() - prev["CO2"].mean()) / prev["CO2"].mean()) * 100
    return 0


def calculate_paris_alignment(df_country, baseline_year=2010):
    """
    Calculate alignment with Paris Agreement targets.
    Target: 45% reduction by 2030 from 2010 levels.
    """
    if baseline_year not in df_country["Year"].values:
        return None
    
    baseline = df_country[df_country["Year"] == baseline_year]["CO2"].mean()
    latest_year = df_country["Year"].max()
    latest = df_country[df_country["Year"] == latest_year]["CO2"].mean()
    
    actual_reduction = ((baseline - latest) / baseline) * 100
    target_reduction = PARIS_TARGET_REDUCTION
    gap = target_reduction - actual_reduction
    
    return {
        "baseline": baseline,
        "latest": latest,
        "actual_reduction": actual_reduction,
        "target_reduction": target_reduction,
        "gap": gap,
        "on_track": gap < 10  # Within 10% of target
    }


def calculate_emission_velocity(df_country):
    """
    Calculate the rate of change in emissions (acceleration/deceleration).
    """
    df_sorted = df_country.sort_values("Year")
    # Second derivative of emissions over time
    co2_values = df_sorted["CO2"].values
    if len(co2_values) < 3:
        return 0
    
    # Calculate acceleration using finite differences
    first_diff = np.diff(co2_values)
    second_diff = np.diff(first_diff)
    return np.mean(second_diff)


def generate_forecast(df_country, country, years_ahead=10):
    """Generate forecast with confidence intervals using linear regression."""
    df_c = df_country[df_country["Country"] == country].dropna(subset=["CO2"])
    
    if len(df_c) < 2:
        return None
    
    X = df_c["Year"].values.reshape(-1, 1)
    y = df_c["CO2"].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_year = df_c["Year"].max()
    future_years = np.arange(last_year + 1, last_year + years_ahead + 1)
    future_preds = model.predict(future_years.reshape(-1, 1))
    
    # Calculate confidence interval (simplified)
    residuals = y - model.predict(X)
    std_error = np.std(residuals)
    confidence_interval = 1.96 * std_error  # 95% CI
    
    return pd.DataFrame({
        "Year": future_years,
        "CO2": future_preds,
        "Lower_CI": future_preds - confidence_interval,
        "Upper_CI": future_preds + confidence_interval,
        "Country": f"{country} (forecast)"
    })


def calculate_scenario(df_country, reduction_rate, target_year=2050):
    """
    Calculate future emissions under different reduction scenarios.
    reduction_rate: annual percentage reduction (negative = reduction, positive = growth)
    """
    latest_year = df_country["Year"].max()
    latest_value = df_country[df_country["Year"] == latest_year]["CO2"].mean()
    
    years = np.arange(latest_year + 1, target_year + 1)
    scenario_values = []
    
    for i, year in enumerate(years):
        value = latest_value * ((1 + reduction_rate/100) ** (i + 1))
        scenario_values.append(value)
    
    return pd.DataFrame({
        "Year": years,
        "CO2": scenario_values
    })


# ---------- Visualization Functions ----------
def create_waterfall_chart(df_country):
    """Create waterfall chart showing cumulative emission changes."""
    yearly_avg = df_country.groupby("Year")["CO2"].mean()
    changes = yearly_avg.diff().dropna()
    
    fig = go.Figure(go.Waterfall(
        name="Emission Changes",
        orientation="v",
        x=changes.index.astype(str),
        y=changes.values,
        text=[f"{val:+.2f}t" for val in changes.values],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#ff6b6b"}},
        decreasing={"marker": {"color": "#51cf66"}},
        totals={"marker": {"color": "#667eea"}}
    ))
    
    fig.update_layout(
        title="Year-over-Year CO₂ Emission Changes",
        xaxis_title="Year",
        yaxis_title="Change in CO₂ (tonnes per capita)",
        template="plotly_white",
        height=400
    )
    
    return fig


def create_decomposition_chart(df_country):
    """
    Create trend decomposition showing trend, seasonality, and residuals.
    (Simplified version - in production would use statsmodels)
    """
    df_sorted = df_country.sort_values("Year")
    
    # Simple moving average as trend
    df_sorted["Trend"] = df_sorted.groupby("Country")["CO2"].transform(
        lambda x: x.rolling(window=5, center=True).mean()
    )
    
    # Residuals
    df_sorted["Residual"] = df_sorted["CO2"] - df_sorted["Trend"]
    
    fig = go.Figure()
    
    for country in df_sorted["Country"].unique():
        df_c = df_sorted[df_sorted["Country"] == country]
        
        fig.add_trace(go.Scatter(
            x=df_c["Year"],
            y=df_c["CO2"],
            name=f"{country} - Actual",
            mode="lines+markers"
        ))
        
        fig.add_trace(go.Scatter(
            x=df_c["Year"],
            y=df_c["Trend"],
            name=f"{country} - Trend",
            mode="lines",
            line=dict(dash="dash")
        ))
    
    fig.update_layout(
        title="Emissions: Actual vs Long-term Trend",
        xaxis_title="Year",
        yaxis_title="CO₂ (tonnes per capita)",
        template="plotly_white",
        height=500
    )
    
    return fig


def create_scenario_comparison(df_country, countries_selected):
    """Create scenario comparison visualization."""
    fig = go.Figure()
    
    # Add historical data
    for country in countries_selected:
        df_c = df_country[df_country["Country"] == country]
        fig.add_trace(go.Scatter(
            x=df_c["Year"],
            y=df_c["CO2"],
            name=f"{country} - Historical",
            mode="lines",
            line=dict(width=2)
        ))
    
    # Add scenarios
    latest_year = df_country["Year"].max()
    avg_latest = df_country[df_country["Year"] == latest_year]["CO2"].mean()
    
    scenarios = {
        "Business as Usual (0%)": 0,
        "Moderate Action (-2%/yr)": -2,
        "Paris Aligned (-7%/yr)": -7,
        "Aggressive (-10%/yr)": -10
    }
    
    colors = ["gray", "orange", "green", "darkgreen"]
    
    for (name, rate), color in zip(scenarios.items(), colors):
        scenario_df = calculate_scenario(df_country, rate, 2050)
        fig.add_trace(go.Scatter(
            x=scenario_df["Year"],
            y=scenario_df["CO2"],
            name=name,
            mode="lines",
            line=dict(dash="dash", color=color, width=2)
        ))
    
    fig.update_layout(
        title="Emission Scenarios to 2050",
        xaxis_title="Year",
        yaxis_title="CO₂ (tonnes per capita)",
        template="plotly_white",
        height=500,
        hovermode="x unified"
    )
    
    return fig


# ---------- UI Components ----------
def render_executive_summary(df_country, countries_selected, year_range):
    """Render executive summary banner."""
    st.markdown("""
    <div class="exec-summary">
        <h2 style='margin: 0; color: white;'>Executive Summary</h2>
        <p style='margin-top: 0.5rem; color: white; font-size: 16px;'>
            Comprehensive climate intelligence dashboard analyzing global CO₂ emission trends,
            policy alignment, and future scenarios.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(df):
    """Render sidebar controls with enhanced options."""
    st.sidebar.title("🌍 CO₂ Intelligence")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 📊 Data Selection")
    countries_selected = st.sidebar.multiselect(
        "Select countries",
        sorted(df["Country"].unique()),
        default=["Germany"],
        help="Choose countries to analyze"
    )
    
    min_year = int(df["Year"].min())
    max_year = int(df["Year"].max())
    
    year_range = st.sidebar.slider(
        "Select year range",
        min_year,
        max_year,
        (min_year, max_year),
        help="Adjust time period for analysis"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔮 Analysis Options")
    
    show_forecast = st.sidebar.checkbox(
        "Show 10-year forecast", 
        value=False,
        help="Display ML-based emission predictions"
    )
    
    show_confidence = st.sidebar.checkbox(
        "Show confidence intervals",
        value=True,
        help="Display uncertainty ranges in forecasts"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Benchmarks")
    
    compare_to_paris = st.sidebar.checkbox(
        "Compare to Paris Agreement",
        value=True,
        help="Show alignment with climate targets"
    )
    
    return countries_selected, year_range, show_forecast, show_confidence, compare_to_paris


def render_key_metrics(df_country, countries_selected, year_range, compare_to_paris):
    """Render enhanced key metrics with insights."""
    if df_country.empty:
        return
    
    # Calculate all metrics
    latest = df_country.sort_values("Year").groupby("Country").tail(1)
    avg_latest = latest["CO2"].mean()
    yoy_change = calculate_yoy_change(df_country)
    cagr = calculate_cagr(df_country, year_range[0], year_range[1])
    velocity = calculate_emission_velocity(df_country)
    
    # Paris alignment
    paris_data = None
    if compare_to_paris:
        paris_data = calculate_paris_alignment(df_country)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Latest Emissions",
        f"{avg_latest:.2f} t",
        delta=f"{yoy_change:+.1f}% YoY",
        delta_color="inverse"
    )
    
    col2.metric(
        "CAGR",
        f"{cagr:+.2f}%",
        delta="per year",
        delta_color="inverse" if cagr < 0 else "normal"
    )
    
    col3.metric(
        "Emission Velocity",
        f"{velocity:.3f}",
        delta="acceleration" if velocity > 0 else "deceleration",
        delta_color="inverse" if velocity > 0 else "normal"
    )
    
    if paris_data:
        col4.metric(
            "Paris Alignment",
            "✅ On Track" if paris_data["on_track"] else "❌ Off Track",
            delta=f"{paris_data['gap']:.1f}% gap",
            delta_color="inverse"
        )
    else:
        peak_year = int(df_country.loc[df_country["CO2"].idxmax(), "Year"])
        col4.metric("Peak Year", peak_year)
    
    return cagr, paris_data


def render_insights(cagr, paris_data, df_country):
    """Render key insights and recommendations."""
    st.markdown("### 🎯 Key Insights")
    
    insights = []
    
    # CAGR insight
    if cagr > 0:
        insights.append(f"⚠️ Emissions are **increasing** at {cagr:.2f}% annually—urgent action needed to reverse this trend.")
    else:
        insights.append(f"✅ Emissions are **decreasing** at {abs(cagr):.2f}% annually—progress toward climate goals.")
    
    # Paris alignment insight
    if paris_data:
        if paris_data["on_track"]:
            insights.append(f"✅ Current trajectory is **aligned** with Paris Agreement targets.")
        else:
            insights.append(f"❌ Currently **{paris_data['gap']:.1f}% behind** Paris Agreement targets. Additional reductions of {paris_data['gap']:.1f}% needed.")
    
    # Velocity insight
    velocity = calculate_emission_velocity(df_country)
    if abs(velocity) < 0.01:
        insights.append("📊 Emission rates are **stabilizing**—consistent trajectory observed.")
    elif velocity > 0:
        insights.append("📈 Emissions are **accelerating**—rate of increase is growing.")
    else:
        insights.append("📉 Emissions are **decelerating**—rate of decrease is improving.")
    
    for insight in insights:
        st.markdown(f"""
        <div class="insight-box">
            {insight}
        </div>
        """, unsafe_allow_html=True)


def render_overview_tab(df_country, countries_selected, year_range, compare_to_paris):
    """Render Overview tab with executive insights."""
    render_executive_summary(df_country, countries_selected, year_range)
    
    if df_country.empty:
        st.warning("⚠️ No data available for this selection. Please adjust your filters.")
        return
    
    st.subheader(f"Analysis: {', '.join(countries_selected)} ({year_range[0]}–{year_range[1]})")
    
    # Key metrics
    cagr, paris_data = render_key_metrics(df_country, countries_selected, year_range, compare_to_paris)
    
    st.markdown("---")
    
    # Insights
    render_insights(cagr, paris_data, df_country)
    
    st.markdown("---")
    
    # Rankings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Latest Rankings")
        latest = df_country.sort_values("Year").groupby("Country").tail(1)
        latest_rank = latest.sort_values("CO2", ascending=False).reset_index(drop=True)
        latest_rank.index = latest_rank.index + 1
        st.dataframe(
            latest_rank[["Country", "CO2"]].rename(columns={"CO2": "CO₂ (t/capita)"}),
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### 📊 Performance Summary")
        summary_data = []
        for country in countries_selected:
            df_c = df_country[df_country["Country"] == country]
            country_cagr = calculate_cagr(df_c, year_range[0], year_range[1])
            summary_data.append({
                "Country": country,
                "CAGR": f"{country_cagr:+.2f}%",
                "Trend": "📈" if country_cagr > 0 else "📉"
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)


def render_trends_tab(df_country, countries_selected, show_forecast, show_confidence):
    """Render Trends tab with advanced visualizations."""
    if df_country.empty:
        st.warning("⚠️ No data available for this selection.")
        return
    
    st.markdown("### 📈 Historical Trends & Decomposition")
    
    # Decomposition chart
    fig_decomp = create_decomposition_chart(df_country)
    st.plotly_chart(fig_decomp, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 💹 Waterfall Analysis")
    
    # Waterfall chart
    fig_waterfall = create_waterfall_chart(df_country)
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Forecast section
    if show_forecast:
        st.markdown("---")
        st.markdown("### 🔮 10-Year Forecast")
        
        fig = go.Figure()
        
        # Add historical data
        for country in countries_selected:
            df_c = df_country[df_country["Country"] == country]
            fig.add_trace(go.Scatter(
                x=df_c["Year"],
                y=df_c["CO2"],
                name=f"{country} - Historical",
                mode="lines+markers",
                line=dict(width=2)
            ))
        
        # Add forecasts
        for country in countries_selected:
            forecast_df = generate_forecast(df_country, country, FORECAST_YEARS)
            if forecast_df is not None:
                fig.add_trace(go.Scatter(
                    x=forecast_df["Year"],
                    y=forecast_df["CO2"],
                    name=f"{country} - Forecast",
                    mode="lines",
                    line=dict(dash="dash", width=2)
                ))
                
                if show_confidence:
                    fig.add_trace(go.Scatter(
                        x=forecast_df["Year"],
                        y=forecast_df["Upper_CI"],
                        fill=None,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast_df["Year"],
                        y=forecast_df["Lower_CI"],
                        fill='tonexty',
                        mode="lines",
                        line=dict(width=0),
                        name=f"{country} - 95% CI",
                        fillcolor='rgba(100, 100, 100, 0.2)'
                    ))
        
        fig.update_layout(
            title="Historical Data with Predictive Forecast",
            xaxis_title="Year",
            yaxis_title="CO₂ (tonnes per capita)",
            template="plotly_white",
            height=500,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_scenario_tab(df_country, countries_selected):
    """Render Scenario Analysis tab."""
    if df_country.empty:
        st.warning("⚠️ No data available for this selection.")
        return
    
    st.markdown("### 🎯 Scenario Planning: Path to 2050")
    
    st.markdown("""
    Explore different policy scenarios and their impact on future emissions.
    Adjust the annual reduction rate to see how different strategies affect outcomes.
    """)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### Scenario Parameters")
        
        custom_reduction = st.slider(
            "Annual reduction rate (%)",
            -15.0, 5.0, -7.0, 0.5,
            help="Negative = reduction, Positive = growth"
        )
        
        st.markdown("---")
        st.markdown("#### Reference Scenarios")
        st.markdown("""
        - **Business as Usual**: 0%/year
        - **Moderate Action**: -2%/year
        - **Paris Aligned**: -7%/year
        - **Aggressive**: -10%/year
        """)
    
    with col2:
        # Scenario comparison chart
        fig = create_scenario_comparison(df_country, countries_selected)
        st.plotly_chart(fig, use_container_width=True)
    
    # Impact summary
    st.markdown("---")
    st.markdown("### 📊 Scenario Impact Summary")
    
    latest_year = df_country["Year"].max()
    latest_value = df_country[df_country["Year"] == latest_year]["CO2"].mean()
    
    target_year = 2050
    years_ahead = target_year - latest_year
    
    scenarios_impact = {
        "Business as Usual": 0,
        "Moderate Action": -2,
        "Paris Aligned": -7,
        "Aggressive": -10,
        f"Custom ({custom_reduction:+.1f}%)": custom_reduction
    }
    
    impact_data = []
    for name, rate in scenarios_impact.items():
        future_value = latest_value * ((1 + rate/100) ** years_ahead)
        total_reduction = ((latest_value - future_value) / latest_value) * 100
        impact_data.append({
            "Scenario": name,
            f"{target_year} Emissions": f"{future_value:.2f} t",
            "Total Reduction": f"{total_reduction:.1f}%",
            "Status": "✅" if total_reduction > 40 else "⚠️"
        })
    
    st.dataframe(pd.DataFrame(impact_data), use_container_width=True)


def render_methodology():
    """Render detailed methodology."""
    with st.expander("📚 Methodology & Data Sources"):
        st.markdown("""
        ### Data Sources
        
        - **CO₂ Emissions**: Our World in Data (OWID) - Per capita annual emissions
        - **Time Period**: Historical data from 1750 to present
        - **Coverage**: Global dataset with 200+ countries
        
        ### Calculations
        
        **CAGR (Compound Annual Growth Rate)**
        ```
        CAGR = (Ending Value / Beginning Value)^(1/n) - 1
        where n = number of years
        ```
        
        **Emission Velocity (Acceleration)**
        - Measures the rate of change in emission growth/decline
        - Calculated using second derivative of emission values
        - Positive = accelerating emissions, Negative = decelerating
        
        **Paris Agreement Alignment**
        - Baseline: 2010 emission levels
        - Target: 45% reduction by 2030
        - Gap: Difference between actual and required reduction
        
        **Forecasting Method**
        - Linear regression on historical data
        - 95% confidence intervals using residual standard error
        - 10-year projection horizon
        
        ### Scenario Parameters
        
        - **Business as Usual**: Current trajectory (0% annual change)
        - **Moderate Action**: -2% annual reduction
        - **Paris Aligned**: -7% annual reduction (aligned with 1.5°C target)
        - **Aggressive**: -10% annual reduction (well below 2°C target)
        
        ### Limitations
        
        - Forecasts assume linear trends; actual emissions may vary
        - Does not account for policy changes or technological breakthroughs
        - Per capita data doesn't reflect absolute national emissions
        - Historical data quality varies by country and time period
        """)


# ---------- Main Application ----------
def main():
    """Main application entry point."""
    # Load data
    df_filtered = load_data()
    
    # Render sidebar
    result = render_sidebar(df_filtered)
    countries_selected, year_range, show_forecast, show_confidence, compare_to_paris = result
    
    # Validate selection
    if len(countries_selected) == 0:
        st.warning("⚠️ Please select at least one country from the sidebar.")
        st.stop()
    
    # Filter data
    df_country = df_filtered[
        (df_filtered["Country"].isin(countries_selected)) &
        (df_filtered["Year"].between(year_range[0], year_range[1]))
    ]
    
    # Main title
    st.title("🌍 Global CO₂ Emissions Intelligence Platform")
    st.markdown("*Advanced analytics for climate action and policy insights*")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Trends & Forecasts",
        "🎯 Scenario Analysis",
        "📄 Raw Data",
        "🗺️ Global Map"
    ])
    
    with tab1:
        render_overview_tab(df_country, countries_selected, year_range, compare_to_paris)
        render_methodology()
    
    with tab2:
        render_trends_tab(df_country, countries_selected, show_forecast, show_confidence)
    
    with tab3:
        render_scenario_tab(df_country, countries_selected)
    
    with tab4:
        st.markdown("### 📄 Dataset Explorer")
        st.dataframe(df_country.reset_index(drop=True), use_container_width=True)
        
        csv = df_country.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Filtered Data (CSV)",
            csv,
            "co2_filtered_data.csv",
            "text/csv"
        )
    
    with tab5:
        st.markdown("### 🗺️ Global Emissions Map")
        
        if not df_country.empty:
            latest_global = df_filtered.sort_values("Year").groupby("Country").tail(1)
            
            fig_map = px.choropleth(
                latest_global,
                locations="Country",
                locationmode="country names",
                color="CO2",
                color_continuous_scale="Reds",
                title="Latest CO₂ Emissions Per Capita by Country",
                hover_data={"CO2": ":.2f"}
            )
            fig_map.update_layout(height=600)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("⚠️ No data available to display the map.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>CO₂ Emissions Intelligence Platform</strong></p>
        <p>Built with Streamlit • Data from Our World in Data</p>
        <p style='font-size: 12px;'>For educational and analytical purposes</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()