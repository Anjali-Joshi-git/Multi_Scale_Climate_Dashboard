import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Multi-Scale Climate Dashboard",
    page_icon="🌍",
    layout="wide"
)

class ScientificForecaster:
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.model = None
        self.optimal_window = 15
        self.max_reliable_horizon = 15

        # Validate data before initializing
        self._validate_data()
        self.initialize_model()

    def _validate_data(self):
        """Validate that data has required columns"""
        required_cols = ['year', 'LandAverageTemperature', 'Seasonally Adjusted CO2 (ppm)']
        missing_cols = [col for col in required_cols if col not in self.historical_data.columns]
        
        if missing_cols:
            st.warning(f"Missing columns in data: {missing_cols}. Using fallback data.")
            # Create fallback data
            years = list(range(1971, 2016))
            self.historical_data = pd.DataFrame({
                'year': years,
                'LandAverageTemperature': [8.5 + 0.03 * (year-1971) for year in years],
                'Seasonally Adjusted CO2 (ppm)': [320 + 1.7 * (year-1971) for year in years]
            })
    
    def initialize_model(self):
        """Initialize with most recent 15-year window"""
        self.historical_data = self.historical_data.sort_values('year')
        latest_year = self.historical_data['year'].max()
        train_start = latest_year - self.optimal_window + 1
        
        training_data = self.historical_data[
            self.historical_data['year'].between(train_start, latest_year)
        ]
        
        X_train = training_data[['Seasonally Adjusted CO2 (ppm)']].values
        y_train = training_data['LandAverageTemperature'].values
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        self.training_period = f"{train_start}-{latest_year}"
        self.last_update = latest_year
        self.current_co2 = float(training_data[training_data['year'] == latest_year]['Seasonally Adjusted CO2 (ppm)'].iloc[0])
        self.current_temp = float(training_data[training_data['year'] == latest_year]['LandAverageTemperature'].iloc[0])
        self.sensitivity = float(self.model.coef_[0] * 100)
        self.reliable_until = latest_year + self.max_reliable_horizon
    
    def get_prediction_confidence(self, target_year):
        """Calculate confidence level based on prediction horizon"""
        years_ahead = target_year - self.last_update
        
        if years_ahead <= 5:
            return "high", "✅", "High confidence (1-5 years)"
        elif years_ahead <= 15:
            return "medium", "🟡", "Medium confidence (6-15 years)"
        else:
            return "low", "🔴", "Low confidence (16+ years)"
    
    def generate_co2_scenarios(self, target_year):
        """Generate DIFFERENT CO₂ scenarios for each emissions pathway"""
        years_ahead = target_year - self.last_update
        
        scenarios = {
            'High Emissions': {
                'co2': self.current_co2 + years_ahead * 3.0,
                'color': '#DC2626',
                'description': 'Rapid fossil fuel growth (+3.0 ppm/year)'
            },
            'Current Trend': {
                'co2': self.current_co2 + years_ahead * 2.5,
                'color': '#EA580C',
                'description': 'Current policies (+2.5 ppm/year)'
            },
            'Climate Action': {
                'co2': self.current_co2 + years_ahead * 1.5,
                'color': '#16A34A', 
                'description': 'Moderate action (+1.5 ppm/year)'
            },
            'Paris Agreement': {
                'co2': self.current_co2 + years_ahead * 1.0,
                'color': '#2563EB',
                'description': 'Strong action (+1.0 ppm/year)'
            }
        }
        return scenarios
    
    def predict_temperature(self, target_year, co2_level):
        prediction = float(self.model.predict([[float(co2_level)]])[0])
        years_ahead = target_year - self.last_update
        
        base_uncertainty = 0.17
        time_penalty = 0.02 * years_ahead
        uncertainty = base_uncertainty + time_penalty
        
        confidence_level, emoji, description = self.get_prediction_confidence(target_year)
        
        return {
            'temperature': round(prediction, 2),
            'uncertainty': round(uncertainty, 2),
            'confidence_low': round(prediction - uncertainty, 2),
            'confidence_high': round(prediction + uncertainty, 2),
            'warming': round(prediction - self.current_temp, 2),
            'confidence_level': confidence_level,
            'confidence_emoji': emoji,
            'confidence_description': description,
            'years_ahead': years_ahead
        }

@st.cache_data
def load_data():
    """Load climate data for all three scales with robust error handling"""
    data = {}
    import os
    
    # COMPREHENSIVE FILE DIAGNOSTICS
    st.sidebar.header("🔍 COMPREHENSIVE FILE DEBUG")
    
    # Check all data files
    data_files = [
        'modern_data_1971_2015.csv',
        'country_warming_rates.csv', 
        'city_warming_rates.csv',
        'vulnerability_results.csv'
    ]
    
    for file in data_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        st.sidebar.write(f"{status} {file}")
        
        if exists:
            try:
                # Show file info
                file_stats = os.stat(file)
                st.sidebar.write(f"   Size: {file_stats.st_size} bytes")
                
                # Show first few rows and columns
                if file.endswith('.csv'):
                    df_sample = pd.read_csv(file, nrows=3)
                    st.sidebar.write(f"   Columns: {list(df_sample.columns)}")
                    st.sidebar.write(f"   Sample data:")
                    st.sidebar.dataframe(df_sample)
            except Exception as e:
                st.sidebar.write(f"   Error reading: {e}")
    
    # Level 1: Global data - ENHANCED DEBUGGING
    try:
        global_data = pd.read_csv('modern_data_1971_2015.csv')
        
        # SHOW EXACT COLUMN NAMES
        st.sidebar.header("🔍 GLOBAL DATA - RAW COLUMNS")
        st.sidebar.write("ACTUAL COLUMNS:", list(global_data.columns))
        st.sidebar.write("FIRST 3 ROWS:")
        st.sidebar.dataframe(global_data.head(3))
        
        # Enhanced column mapping with more variations
        column_mapping = {}
        
        # Detect year column - expanded list
        year_candidates = ['Year', 'year', 'YEAR', 'Time', 'time', 'DATE', 'date', 
                          'yr', 'Yr', 'annual', 'Annual', 'decade', 'Decade']
        for candidate in year_candidates:
            if candidate in global_data.columns:
                column_mapping[candidate] = 'year'
                st.sidebar.info(f"Found year column: {candidate} -> year")
                break
        
        # Detect temperature column - expanded list
        temp_candidates = ['LandAverageTemperature', 'Temperature', 'temperature', 'temp', 'TEMP',
                          'Land Average Temperature', 'Global Temperature', 'global_temperature',
                          'temp_global', 'global_temp', 'mean_temp', 'Mean_Temp', 'anomaly', 'Anomaly']
        for candidate in temp_candidates:
            if candidate in global_data.columns:
                column_mapping[candidate] = 'LandAverageTemperature'
                st.sidebar.info(f"Found temperature column: {candidate} -> LandAverageTemperature")
                break
        
        # Detect CO2 column - expanded list
        co2_candidates = ['Seasonally Adjusted CO2 (ppm)', 'CO2', 'co2', 'carbon_dioxide',
                         'Carbon Dioxide', 'CO2_ppm', 'co2_ppm', 'CO2 (ppm)', 'co2_ppm',
                         'carbon', 'Carbon', 'co2_concentration', 'CO2_concentration']
        for candidate in co2_candidates:
            if candidate in global_data.columns:
                column_mapping[candidate] = 'Seasonally Adjusted CO2 (ppm)'
                st.sidebar.info(f"Found CO2 column: {candidate} -> Seasonally Adjusted CO2 (ppm)")
                break
        
        # Apply renaming if we found matches
        if column_mapping:
            global_data = global_data.rename(columns=column_mapping)
            st.success(f"🔧 Auto-renamed columns: {column_mapping}")
        else:
            st.sidebar.error("❌ No column matches found!")
            st.sidebar.write("Available columns:", list(global_data.columns))
        
        # Check if we have the required columns after renaming
        required_columns = ['year', 'LandAverageTemperature', 'Seasonally Adjusted CO2 (ppm)']
        missing_columns = [col for col in required_columns if col not in global_data.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.warning("Using sample data for global analysis")
            
            # Create fallback data
            years = list(range(1971, 2016))
            global_data = pd.DataFrame({
                'year': years,
                'LandAverageTemperature': [8.5 + 0.03 * (year-1971) for year in years],
                'Seasonally Adjusted CO2 (ppm)': [320 + 1.7 * (year-1971) for year in years]
            })
        else:
            st.success("✅ Loaded global climate data")
        
        data['global'] = global_data
        
    except FileNotFoundError:
        st.warning("📝 Using sample global data")
        years = list(range(1971, 2016))
        temps = [8.5 + 0.03 * (year-1971) + np.random.normal(0, 0.15) for year in years]
        co2 = [320 + 1.7 * (year-1971) for year in years]
        data['global'] = pd.DataFrame({
            'year': years,
            'LandAverageTemperature': temps,
            'Seasonally Adjusted CO2 (ppm)': co2
        })
    
    # Level 2: Country data - ENHANCED WITH COLUMN VALIDATION
    try:
        country_data = pd.read_csv('country_warming_rates.csv')
        
        # DEBUG: Show country data structure
        st.sidebar.header("🔍 COUNTRY DATA - RAW COLUMNS")
        st.sidebar.write("ACTUAL COLUMNS:", list(country_data.columns))
        st.sidebar.write("FIRST 3 ROWS:")
        st.sidebar.dataframe(country_data.head(3))
        
        # Auto-detect and map country column names
        country_column_mapping = {}
        
        # Detect country name column
        country_candidates = ['country', 'Country', 'COUNTRY', 'nation', 'Nation', 
                             'country_name', 'Country_Name', 'location', 'Location']
        for candidate in country_candidates:
            if candidate in country_data.columns:
                country_column_mapping[candidate] = 'country'
                st.sidebar.info(f"Found country column: {candidate} -> country")
                break
        
        # Detect warming rate column
        warming_candidates = ['warming_rate_c_per_decade', 'warming_rate', 'Warming Rate',
                             'warming', 'Warming', 'rate', 'Rate', 'trend', 'Trend',
                             'slope', 'Slope', 'change_rate', 'Change_Rate']
        for candidate in warming_candidates:
            if candidate in country_data.columns:
                country_column_mapping[candidate] = 'warming_rate_c_per_decade'
                st.sidebar.info(f"Found warming rate column: {candidate} -> warming_rate_c_per_decade")
                break
        
        # Apply renaming
        if country_column_mapping:
            country_data = country_data.rename(columns=country_column_mapping)
            st.info(f"🔧 Auto-renamed country columns: {country_column_mapping}")
        
        # Ensure required columns exist
        if 'country' not in country_data.columns:
            # If no country column found, use the first column as country names
            first_col = country_data.columns[0]
            country_data = country_data.rename(columns={first_col: 'country'})
            st.warning(f"🔧 Using first column '{first_col}' as country names")
        
        if 'warming_rate_c_per_decade' not in country_data.columns:
            # Try to find numeric columns that could be warming rates
            numeric_cols = country_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                country_data = country_data.rename(columns={numeric_cols[0]: 'warming_rate_c_per_decade'})
                st.warning(f"🔧 Using numeric column '{numeric_cols[0]}' as warming rate")
            else:
                # Add default warming rate
                country_data['warming_rate_c_per_decade'] = 0.3
                st.warning("🔧 Added default warming rate column")
        
        st.success("✅ Loaded country-level data")
        data['country'] = country_data
        
    except FileNotFoundError:
        st.warning("📝 Using sample country data")
        countries = ['Turkmenistan', 'Mongolia', 'Kazakhstan', 'Russia', 'Iran', 'Canada']
        warming_rates = [0.331, 0.328, 0.318, 0.317, 0.307, 0.303]
        data['country'] = pd.DataFrame({
            'country': countries,
            'warming_rate_c_per_decade': warming_rates
        })
    
    # Level 3: Urban data
    try:
        urban_data = pd.read_csv('city_warming_rates.csv')
        st.success("✅ Loaded urban-level data")
        data['urban'] = urban_data
        
    except FileNotFoundError:
        st.warning("📝 Using sample urban data")
        cities = ['Mashhad', 'Harbin', 'Changchun', 'Moscow', 'Shenyang', 'Kiev']
        countries = ['Iran', 'China', 'China', 'Russia', 'China', 'Ukraine']
        urban_rates = [0.164, 0.160, 0.155, 0.151, 0.143, 0.135]
        data['urban'] = pd.DataFrame({
            'city': cities,
            'country': countries,
            'warming_rate_c_per_decade': urban_rates,
            'warming_intensity': ['Extreme', 'Extreme', 'Extreme', 'Extreme', 'Extreme', 'Extreme']
        })
    
    return data

@st.cache_data  
def load_vulnerability_data():
    """Load vulnerability results with fallback"""
    try:
        vulnerability_df = pd.read_csv('vulnerability_results.csv')
        st.success("✅ Loaded pre-calculated vulnerability scores")
        return vulnerability_df
    except FileNotFoundError:
        st.info("📝 Using sample vulnerability data")
        
        # Create vulnerability data that matches the countries in our main dataset
        try:
            country_data = pd.read_csv('country_warming_rates.csv')
            countries = country_data['country'].unique().tolist()
        except:
            countries = ['Turkmenistan', 'Mongolia', 'Kazakhstan', 'Russia', 'Iran', 'Canada']
        
        vulnerability_scores = np.random.uniform(0.3, 0.9, len(countries))
        
        vulnerability_df = pd.DataFrame({
            'country': countries,
            'vulnerability_score': vulnerability_scores,
        })
        
        # Create categories based on scores
        vulnerability_df['vulnerability_category'] = pd.cut(
            vulnerability_df['vulnerability_score'],
            bins=[0, 0.4, 0.6, 0.8, 1],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return vulnerability_df

def show_global_analysis(global_data):
    """Level 1: Global Climate Analysis - FIXED VERSION"""
    st.header("🌍 Global Climate Patterns")
    
    try:
        # DEBUG: Show what data we actually have
        with st.expander("🔍 Debug Global Data"):
            st.write("Columns in global data:", list(global_data.columns))
            st.write("First few rows:")
            st.dataframe(global_data.head())
            st.write("Data types:")
            st.write(global_data.dtypes)
        
        # Check for required columns and rename if needed
        required_columns = ['year', 'LandAverageTemperature', 'Seasonally Adjusted CO2 (ppm)']
        
        # Create column mapping for common variations
        column_mapping = {}
        if 'Year' in global_data.columns and 'year' not in global_data.columns:
            column_mapping['Year'] = 'year'
        if 'Land Average Temperature' in global_data.columns:
            column_mapping['Land Average Temperature'] = 'LandAverageTemperature'
        if 'CO2' in global_data.columns or 'co2' in global_data.columns:
            co2_col = 'CO2' if 'CO2' in global_data.columns else 'co2'
            column_mapping[co2_col] = 'Seasonally Adjusted CO2 (ppm)'
        
        # Apply renaming if needed
        if column_mapping:
            global_data = global_data.rename(columns=column_mapping)
            st.info(f"🔧 Renamed columns: {column_mapping}")
        
        # Check if we have the required columns now
        missing_columns = [col for col in required_columns if col not in global_data.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.warning("Using fallback data for global analysis")
            
            # Create fallback data
            years = list(range(1971, 2016))
            global_data = pd.DataFrame({
                'year': years,
                'LandAverageTemperature': [8.5 + 0.03 * (year-1971) for year in years],
                'Seasonally Adjusted CO2 (ppm)': [320 + 1.7 * (year-1971) for year in years]
            })
        
        # Now proceed with the analysis
        forecaster = ScientificForecaster(global_data)
        
        # Create two columns for layout
        col_main, col_sidebar = st.columns([3, 1])
        
        with col_sidebar:
            # Sidebar with projection settings
            st.sidebar.title("🎯 Projection Settings")
            
            st.sidebar.markdown(f"""
            **Model Status:**
            - Training: {forecaster.training_period}
            - Current CO₂: {forecaster.current_co2:.1f} ppm
            - Current temp: {forecaster.current_temp:.2f}°C
            - Reliable until: **{forecaster.reliable_until}**
            """)
            
            # Moved confidence indicator up
            confidence_level, emoji, description = forecaster.get_prediction_confidence(2030)
            st.sidebar.markdown(f"**Default Confidence:** {emoji} {description}")
        
        with col_main:
            # Model overview
            st.subheader("🔬 Model Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Period", forecaster.training_period)
            with col2:
                st.metric("Climate Sensitivity", f"{forecaster.sensitivity:.2f}°C/100ppm")
            with col3:
                st.metric("Current CO₂", f"{forecaster.current_co2:.1f} ppm")
            with col4:
                st.metric("Current Temp", f"{forecaster.current_temp:.2f}°C")
            
            # TARGET YEAR SLIDER
            st.subheader("🎯 Projection Settings")
            
            col_slider1, col_slider2 = st.columns([2, 1])
            
            with col_slider1:
                target_year = st.slider(
                    "Select Target Year for Projections",
                    min_value=2016,
                    max_value=2035,
                    value=2030,
                    step=1,
                    help="Choose the target year for climate projections"
                )
            
            with col_slider2:
                # Dynamic confidence indicator
                confidence_level, emoji, description = forecaster.get_prediction_confidence(target_year)
                years_ahead = target_year - forecaster.last_update
                
                st.metric(
                    "Prediction Confidence",
                    f"{emoji} {confidence_level.title()}",
                    f"{years_ahead} years ahead"
                )
            
            # Projections
            st.subheader("🔮 Global Projections")
            
            scenarios = forecaster.generate_co2_scenarios(target_year)
            projections = {}
            
            for scenario_name, scenario_data in scenarios.items():
                prediction = forecaster.predict_temperature(target_year, scenario_data['co2'])
                projections[scenario_name] = {
                    **prediction,
                    'co2': scenario_data['co2'],
                    'color': scenario_data['color'],
                    'description': scenario_data['description']
                }
            
            # Display projections
            cols = st.columns(4)
            for i, (scenario_name, data) in enumerate(projections.items()):
                with cols[i]:
                    st.markdown(f"<h4 style='color: {data['color']};'>{scenario_name}</h4>", unsafe_allow_html=True)
                    st.metric(
                        "Projected Temperature",
                        f"{data['temperature']}°C",
                        f"{data['warming']:+.2f}°C"
                    )
                    st.metric("CO₂ Concentration", f"{data['co2']:.1f} ppm")
                    st.caption(data['description'])

            # CO₂ Scenario Comparison Chart
            st.subheader("📈 CO₂ Scenario Comparison")
            
            # Create data for chart
            scenario_names = list(projections.keys())
            co2_levels = [projections[name]['co2'] for name in scenario_names]
            temperatures = [projections[name]['temperature'] for name in scenario_names]
            colors = [projections[name]['color'] for name in scenario_names]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=scenario_names,
                y=co2_levels,
                name='CO₂ Concentration',
                marker_color=colors,
                hovertemplate='%{x}<br>CO₂: %{y:.1f} ppm<extra></extra>',
                yaxis='y1'
            ))
            
            fig.add_trace(go.Scatter(
                x=scenario_names,
                y=temperatures,
                name='Temperature',
                mode='markers+lines',
                line=dict(color='black', width=3),
                marker=dict(size=10, symbol='diamond'),
                hovertemplate='%{x}<br>Temperature: %{y:.2f}°C<extra></extra>',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=f"CO₂ and Temperature Projections for {target_year}",
                xaxis_title="Scenario",
                yaxis=dict(title="CO₂ (ppm)", side='left'),
                yaxis2=dict(title="Temperature (°C)", side='right', overlaying='y'),
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Historical trends chart
            st.subheader("📊 Historical Trends")
            fig = px.line(
                global_data, 
                x='year', 
                y='LandAverageTemperature',
                title='Global Average Temperature Trend (1971-2015)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in global analysis: {e}")
        # Show more debug info
        with st.expander("🔧 Technical Details"):
            st.write("Global data columns:", list(global_data.columns))
            st.write("Global data shape:", global_data.shape)
            st.write("Exception details:", str(e))

def show_country_analysis_with_vulnerability(country_data, vulnerability_df):
    """Country analysis with vulnerability scoring"""
    st.header("🇺🇳 Country-Level Climate Analysis")
    
    try:
        # Display raw data structure
        with st.expander("🔍 Raw Data Structure"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Country Data")
                st.write("Columns:", list(country_data.columns))
                st.write("Sample data:")
                st.dataframe(country_data.head(5))
                st.write(f"Total countries: {len(country_data)}")
            
            with col2:
                st.subheader("Vulnerability Data")
                st.write("Columns:", list(vulnerability_df.columns))
                st.write("Sample data:")
                st.dataframe(vulnerability_df.head(5))
                st.write(f"Total countries: {len(vulnerability_df)}")
        
        # Safe merge
        merged_data = country_data.merge(vulnerability_df, on='country', how='left', suffixes=('', '_vuln'))
        
        st.success(f"✅ Successfully loaded data for {len(merged_data)} countries")
        
        # Show key statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_warming = merged_data['warming_rate_c_per_decade'].mean()
            st.metric("Global Average Warming", f"{avg_warming:.3f}°C/decade")
        with col2:
            max_warming = merged_data['warming_rate_c_per_decade'].max()
            st.metric("Fastest Warming", f"{max_warming:.3f}°C/decade")
        with col3:
            min_warming = merged_data['warming_rate_c_per_decade'].min()
            st.metric("Slowest Warming", f"{min_warming:.3f}°C/decade")
        with col4:
            high_quality = len(merged_data[merged_data['r_squared'] > 0.5])
            st.metric("High Quality Data", f"{high_quality} countries")
        
        # Country selector
        selected_country = st.selectbox(
            "Select Country",
            options=merged_data['country'].unique(),
            key="country_select"
        )
        
        # Tabs for organized analysis
        tab1, tab2, tab3 = st.tabs(["📊 Warming Analysis", "🛡️ Vulnerability Assessment", "📈 Data Quality"])
        
        with tab1:
            st.subheader("🌡️ Country Warming Rates")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Show top warming countries
                top_countries = merged_data.nlargest(15, 'warming_rate_c_per_decade')
                fig = px.bar(
                    top_countries,
                    x='warming_rate_c_per_decade',
                    y='country',
                    orientation='h',
                    title='Top 15 Fastest-Warming Countries',
                    color='warming_rate_c_per_decade',
                    color_continuous_scale='Reds',
                    labels={'warming_rate_c_per_decade': 'Warming Rate (°C/decade)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                country_info = merged_data[merged_data['country'] == selected_country].iloc[0]
                
                # Display warming rate
                warming_rate = country_info['warming_rate_c_per_decade']
                st.metric(
                    f"{selected_country} Warming Rate",
                    f"{warming_rate:.3f}°C/decade"
                )
                
                # Show comparison to average
                avg_warming = merged_data['warming_rate_c_per_decade'].mean()
                comparison = warming_rate - avg_warming
                st.metric(
                    "vs Global Average",
                    f"{comparison:+.3f}°C/decade"
                )
                
                # Show ranking
                ranking = (merged_data['warming_rate_c_per_decade'] > warming_rate).sum() + 1
                st.metric("Global Ranking", f"#{ranking}")
        
        with tab2:
            st.subheader("🛡️ Climate Vulnerability Assessment")
            
            if 'vulnerability_score' in merged_data.columns:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Vulnerability distribution
                    if 'vulnerability_category' in merged_data.columns:
                        vuln_counts = merged_data['vulnerability_category'].value_counts()
                        fig = px.bar(
                            x=vuln_counts.index,
                            y=vuln_counts.values,
                            color=vuln_counts.index,
                            color_discrete_map={
                                'Low': '#2ecc71',
                                'Medium': '#f39c12', 
                                'High': '#e74c3c',
                                'Critical': '#8b0000'
                            },
                            title='Climate Vulnerability Distribution',
                            labels={'x': 'Vulnerability Category', 'y': 'Number of Countries'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Top vulnerable countries
                    top_vulnerable = merged_data.nlargest(10, 'vulnerability_score')
                    fig2 = px.bar(
                        top_vulnerable,
                        x='vulnerability_score',
                        y='country',
                        orientation='h',
                        color='vulnerability_category',
                        color_discrete_map={
                            'Low': '#2ecc71',
                            'Medium': '#f39c12', 
                            'High': '#e74c3c',
                            'Critical': '#8b0000'
                        },
                        title='Top 10 Most Vulnerable Countries',
                        labels={'vulnerability_score': 'Vulnerability Score'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    country_info = merged_data[merged_data['country'] == selected_country].iloc[0]
                    
                    if 'vulnerability_score' in country_info:
                        st.metric("Vulnerability Score", f"{country_info['vulnerability_score']:.2f}")
                    
                    if 'vulnerability_category' in country_info:
                        st.metric("Risk Category", country_info['vulnerability_category'])
                    
                    # Show risk assessment
                    if 'vulnerability_category' in country_info:
                        category = country_info['vulnerability_category']
                        if category in ['High', 'Critical']:
                            st.error("🚨 High climate vulnerability detected")
                            st.write("**Priority for adaptation funding**")
                        elif category == 'Medium':
                            st.warning("⚠️ Moderate climate vulnerability")
                            st.write("**Monitor and plan adaptation**")
                        else:
                            st.success("✅ Lower climate vulnerability")
                            st.write("**Focus on mitigation**")
            
            else:
                st.info("ℹ️ Vulnerability data not available")
        
        with tab3:
            st.subheader("📈 Data Quality Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # R-squared distribution
                if 'r_squared' in merged_data.columns:
                    fig = px.histogram(
                        merged_data,
                        x='r_squared',
                        title='R-squared Distribution (Data Quality)',
                        labels={'r_squared': 'R-squared Value'},
                        color_discrete_sequence=['#3498db']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Data points distribution
                if 'data_points' in merged_data.columns:
                    fig = px.histogram(
                        merged_data,
                        x='data_points',
                        title='Data Points Distribution',
                        labels={'data_points': 'Number of Data Points'},
                        color_discrete_sequence=['#2ecc71']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error in country analysis: {e}")

def show_urban_analysis(urban_data):
    """Level 3: Urban-Level Analysis"""
    st.header("🏙️ Urban Climate Analysis")
    
    try:
        # Display raw data structure
        with st.expander("🔍 Raw Urban Data Structure"):
            st.write("Columns:", list(urban_data.columns))
            st.write("Sample data:")
            st.dataframe(urban_data.head(5))
            st.write(f"Total cities: {len(urban_data)}")
            if 'warming_rate_c_per_decade' in urban_data.columns:
                st.write(f"Warming range: {urban_data['warming_rate_c_per_decade'].min():.3f} to {urban_data['warming_rate_c_per_decade'].max():.3f}°C/decade")
        
        # Urban warming intensity distribution
        if 'warming_intensity' in urban_data.columns:
            st.subheader("🏙️ Urban Warming Intensity Distribution")
            intensity_counts = urban_data['warming_intensity'].value_counts()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    x=intensity_counts.index,
                    y=intensity_counts.values,
                    color=intensity_counts.index,
                    color_discrete_map={
                        'Very Slow': '#2ecc71',
                        'Slow': '#f39c12',
                        'Moderate': '#e67e22',
                        'Fast': '#e74c3c',
                        'Extreme': '#8b0000'
                    },
                    title='Urban Warming Intensity Distribution',
                    labels={'x': 'Warming Intensity', 'y': 'Number of Cities'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 Urban Warming Stats")
                total_cities = len(urban_data)
                extreme_cities = len(urban_data[urban_data['warming_intensity'] == 'Extreme'])
                st.metric("Total Cities Analyzed", total_cities)
                st.metric("Extreme Warming Cities", extreme_cities)
                
                if 'warming_rate_c_per_decade' in urban_data.columns:
                    st.metric(
                        "Fastest Warming City", 
                        f"{urban_data['warming_rate_c_per_decade'].max():.3f}°C/decade"
                    )
        
        # Top urban hotspots
        if 'warming_rate_c_per_decade' in urban_data.columns:
            st.subheader("🔥 Urban Warming Hotspots")
            top_urban = urban_data.nlargest(10, 'warming_rate_c_per_decade')
            
            fig = px.bar(
                top_urban,
                x='warming_rate_c_per_decade',
                y='city',
                orientation='h',
                color='warming_rate_c_per_decade',
                color_continuous_scale='Reds',
                title='Top 10 Fastest-Warming Cities',
                hover_data=['country'] if 'country' in urban_data.columns else None
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Geographic patterns
        st.subheader("🌍 Urban Geographic Patterns")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Northeast China Cluster:**")
            st.write("Harbin, Changchun, Shenyang")
            st.write("Rapid industrial warming")
        
        with col2:
            st.write("**Middle East Hotspots:**")
            st.write("Mashhad, Baghdad")
            st.write("Arid climate amplification")
        
        with col3:
            st.write("**Northern Cities:**")
            st.write("Moscow, Montreal, Toronto")
            st.write("Cold climate sensitivity")
            
    except Exception as e:
        st.error(f"Error in urban analysis: {e}")

def show_cross_scale_comparison(global_data, country_data, urban_data):
    """Cross-scale comparisons - FIXED VERSION"""
    st.header("📊 Multi-Scale Comparison")
    
    try:
        # Display raw data structures
        with st.expander("🔍 Raw Data Structures"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🌍 Global Data")
                st.write("Columns:", list(global_data.columns))
                st.write(f"Records: {len(global_data)}")
                st.dataframe(global_data.head(3))
            
            with col2:
                st.subheader("🇺🇳 Country Data")
                st.write("Columns:", list(country_data.columns))
                st.write(f"Countries: {len(country_data)}")
                st.dataframe(country_data[['country', 'warming_rate_c_per_decade', 'r_squared']].head(3) if 'country' in country_data.columns else country_data.head(3))
            
            with col3:
                st.subheader("🏙️ Urban Data")
                st.write("Columns:", list(urban_data.columns))
                st.write(f"Cities: {len(urban_data)}")
                st.dataframe(urban_data.head(3))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🌍 Global Scale")
            if 'warming_rate_c_per_decade' in country_data.columns:
                global_avg = country_data['warming_rate_c_per_decade'].mean()
                st.metric("Average Warming Rate", f"{global_avg:.3f}°C/decade")
            else:
                st.metric("Average Warming Rate", "N/A")
            st.write("**Primary Driver:** CO₂ concentrations")
            st.write("**Pattern:** Uniform global trend")
            st.write("**Data Source:** Global temperature records")
        
        with col2:
            st.subheader("🇺🇳 Country Scale")
            if 'warming_rate_c_per_decade' in country_data.columns:
                country_min = country_data['warming_rate_c_per_decade'].min()
                country_max = country_data['warming_rate_c_per_decade'].max()
                st.metric(
                    "Range of Warming", 
                    f"{country_min:.3f} - {country_max:.3f}°C/decade"
                )
                st.metric("Countries Analyzed", f"{len(country_data)}")
            else:
                st.metric("Range of Warming", "N/A")
            st.write("**Key Factors:** Geography, latitude, elevation")
            st.write("**Pattern:** Regional variations")
            st.write("**Data Source:** National temperature datasets")
        
        with col3:
            st.subheader("🏙️ Urban Scale")
            if 'warming_rate_c_per_decade' in urban_data.columns:
                urban_avg = urban_data['warming_rate_c_per_decade'].mean()
                urban_min = urban_data['warming_rate_c_per_decade'].min()
                urban_max = urban_data['warming_rate_c_per_decade'].max()
                st.metric("Urban Average", f"{urban_avg:.3f}°C/decade")
                st.metric("Urban Range", f"{urban_min:.3f} - {urban_max:.3f}°C/decade")
            else:
                st.metric("Urban Average", "N/A")
            st.write("**Key Factors:** Urban heat island, population density")
            st.write("**Pattern:** Local amplification")
            st.write("**Data Source:** City weather station data")
        
        # Scale comparison chart
        if all(col in df.columns for df, col in [(country_data, 'warming_rate_c_per_decade'), 
                                               (urban_data, 'warming_rate_c_per_decade')]):
            st.subheader("📈 Warming Rates Across Scales")
            
            scales_data = pd.DataFrame({
                'scale': ['Global Average', 'Country Average', 'Urban Average', 'Fastest Country', 'Fastest City'],
                'warming_rate': [
                    country_data['warming_rate_c_per_decade'].mean(),
                    country_data['warming_rate_c_per_decade'].mean(),
                    urban_data['warming_rate_c_per_decade'].mean(),
                    country_data['warming_rate_c_per_decade'].max(),
                    urban_data['warming_rate_c_per_decade'].max()
                ],
                'type': ['Background', 'Regional', 'Local', 'Extreme', 'Extreme']
            })
            
            fig = px.bar(
                scales_data,
                x='scale',
                y='warming_rate',
                color='type',
                color_discrete_map={
                    'Background': '#2ecc71',
                    'Regional': '#f39c12',
                    'Local': '#e74c3c',
                    'Extreme': '#8b0000'
                },
                title='Warming Rates Across Spatial Scales',
                labels={'warming_rate': 'Warming Rate (°C/decade)', 'scale': 'Spatial Scale'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("💡 Key Multi-Scale Insights")
        
        if all(col in df.columns for df, col in [(country_data, 'warming_rate_c_per_decade'), 
                                               (urban_data, 'warming_rate_c_per_decade')]):
            country_avg = country_data['warming_rate_c_per_decade'].mean()
            urban_avg = urban_data['warming_rate_c_per_decade'].mean()
            country_max = country_data['warming_rate_c_per_decade'].max()
            urban_max = urban_data['warming_rate_c_per_decade'].max()
            
            insights = [
                f"**Scale Matters:** Country average ({country_avg:.3f}°C/decade) vs Urban average ({urban_avg:.3f}°C/decade)",
                f"**Hotspot Identification:** Fastest country ({country_max:.3f}°C/decade) vs Fastest city ({urban_max:.3f}°C/decade)",
                "**Urban-Rural Difference:** Cities show different warming patterns than surrounding regions",
                "**Policy Implications:** Different adaptation strategies needed at each scale"
            ]
        else:
            insights = [
                "**Scale Matters:** Warming rates vary across global, country, and urban scales",
                "**Hotspot Identification:** Different patterns emerge at different spatial resolutions",
                "**Urban-Rural Difference:** Cities often show distinct warming patterns",
                "**Policy Implications:** Tailored strategies needed for each scale"
            ]
        
        for insight in insights:
            st.write(f"• {insight}")
            
    except Exception as e:
        st.error(f"Error in cross-scale comparison: {e}")

def main():
    # DEBUG: Show what's being loaded
    st.sidebar.header("🔍 Data Loading Debug")
    
    # Load all data
    data = load_data()
    
    # Show what columns we actually got
    st.sidebar.write("**Global Data Columns:**", list(data['global'].columns))
    st.sidebar.write("**Country Data Columns:**", list(data['country'].columns))
    st.sidebar.write("**Urban Data Columns:**", list(data['urban'].columns))
    
    st.markdown("""
<h1 style='text-align: center; font-family: "Arial", sans-serif; font-size: 48px; color: #2c3e50;'>
🌍 Multi-Scale Climate Dashboard
</h1>
""", unsafe_allow_html=True)
    
    # Load vulnerability data
    vulnerability_results = load_vulnerability_data()
    
    # Data overview
    with st.sidebar.expander("📊 Data Overview"):
        st.write(f"**Global Data:** {len(data['global'])} records")
        st.write(f"**Countries:** {len(data['country'])} countries") 
        st.write(f"**Cities:** {len(data['urban'])} cities")
        if 'year' in data['global'].columns:
            st.write(f"**Time Period:** {data['global']['year'].min()}-{data['global']['year'].max()}")
    
    # Navigation
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.radio(
        "Select Analysis Scale:",
        ["Global Overview", "Country Analysis", "Urban Insights", "Cross-Scale Comparisons"]
    )
    
    # Display selected page
    try:
        if page == "Global Overview":
            show_global_analysis(data['global'])
        elif page == "Country Analysis":
            show_country_analysis_with_vulnerability(data['country'], vulnerability_results)
        elif page == "Urban Insights":
            show_urban_analysis(data['urban'])
        elif page == "Cross-Scale Comparisons":
            show_cross_scale_comparison(data['global'], data['country'], data['urban'])
    except Exception as e:
        st.error(f"Error loading page: {e}")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Data Sources")
    st.sidebar.markdown("""
    - **Global:** CO₂ and temperature records
    - **Country:** National temperature datasets  
    - **Urban:** City weather station data
    """)

if __name__ == "__main__":
    main()
