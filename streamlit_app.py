import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="CO₂ Emission Estimator", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Emission Predictor",
    "Recommendation",
    "Compare Models",
    "About"
])
# HOME PAGE
if page == "Home":
    st.title("Carbon Emission Intelligence System")
    st.image("https://media.greenmatters.com/brand-img/ytZ-T9sp7/0x0/what-emissions-do-cars-produce4-1604596615333.jpg")

    st.markdown("""
    ---
    ### **Have you ever wondered...**

    - How much **CO₂ your logistics fleet** emits daily?
    - Whether switching from a **truck to a bike** could make a real environmental difference?
    - If a **machine learning model** could predict and reduce emissions better than traditional rules?

    ---
    ### **The Problem**

    With rising pollution and stricter sustainability targets, companies are under immense pressure to **reduce their carbon footprint**. Yet, **estimating CO₂ emissions accurately** and identifying eco-friendly alternatives remain a complex task—until now.

    ---
    ### **Our Solution: Carbon Emission Intelligence System**

    This project uses:
    - Real-world logistics data
    - **Machine Learning (Random Forest, XGBoost, Linear Regression)**
    - **Rule-based logic** as a baseline
    - **K-Means clustering** to group emission behaviors

    It offers:
    - Accurate **CO₂ emission predictions**
    - Smart **vehicle recommendations** for lower emissions
    - **Model comparison** dashboard
    - **Cluster insights** to understand vehicle usage patterns

    ---
    ###  Key Features

    - Predict carbon emissions from logistics vehicles
    - Recommend cleaner transportation modes
    - Compare ML models vs rule-based methods
    - Visualize emission clusters and performance

    ---
    """)

    # Layout of features (two columns)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(" **ML Models Used:**")
        st.markdown("- Random Forest")
        st.markdown("- XGBoost")
        st.markdown("- Linear Regression")

    with col2:
        st.markdown(" **Clustering Insight:**")
        st.markdown("- KMeans Algorithm")
        st.markdown("- PCA Visualization")
        st.markdown("- Group: High vs Low Emitters")

    st.info("Use the sidebar to explore pages like **Emission Predictor**, **Recommendation**, **Compare Models**, and **About**.")
    st.success("“The greatest threat to our planet is the belief that someone else will save it.” — Robert Swan")
    col1, col2 = st.columns(2)
    with col1:
      st.image("images/output.png", caption="VISUALIZATION")
    with col2:
      st.image("images/output2.png", caption="VISUALIZATION")

# Emission Predictor Page
if page == "Emission Predictor":
    st.title("Emission Predictor")
    st.markdown("Predict **CO₂ emissions** using different ML models with smart logistics input.")
    # Load Models
    rf_model = joblib.load("Notebook/rf_model_pipeline.pkl")
    xgb_model = joblib.load("Notebook/xgboost_pipeline.pkl")
    lr_model = joblib.load("Notebook/linearregression_pipeline.pkl")
    # Company → Vehicle → Fuel Mappings
    company_vehicle_fuel_map = {
        "Delhivery": {
            "Bike": ["Electric", "Petrol"],
            "Auto": ["Petrol", "Electric", "LNG"],
            "Van": ["Diesel", "Electric"],
            "Light Truck": ["Diesel", "CNG"],
            "Medium Truck": ["LNG"]
        },
        "Ecom Express": {
            "Bike": ["Electric", "Petrol"],
            "Auto": ["CNG", "Electric"],
            "Van": ["Diesel", "CNG"],
            "Light Truck": ["Diesel"]
        },
        "Shadowfax": {
            "Bike": ["Electric", "Petrol"],
            "Van": ["Diesel"],
            "Light Truck": ["Diesel", "CNG"]
        },
        "XpressBees": {
            "Bike": ["Electric", "Petrol"],
            "Van": ["Diesel", "CNG", "Electric"],
            "Light Truck": ["Diesel", "CNG"]
        },
        "LoadShare": {
            "Bike": ["Electric", "Petrol"],
            "Auto": ["Petrol", "Electric"]
        },
        "Bluedart": {
            "Bike": ["Electric", "Petrol"],
            "Auto": ["Petrol", "Electric"],
            "Van": ["Diesel", "Electric", "Hybrid"],
            "Light Truck": ["Diesel"],
            "Cargo Plane": ["Jet Fuel"]
        },
        "DTDC": {
            "Van": ["Diesel"],
            "Light Truck": ["Diesel"]
        },
        "Flipkart": {
            "Van": ["Diesel", "Electric"],
            "Light Truck": ["Diesel", "LNG", "Electric"],
            "Medium Truck": ["LNG", "Diesel"]
        },
        "Amazon": {
            "Van": ["Diesel", "Electric"],
            "Light Truck": ["Diesel", "Electric"],
            "Cargo Plane": ["Jet Fuel"]
        },
        "VRL Logistics": {
            "Medium Truck": ["Diesel", "CNG"],
            "Heavy Truck": ["Diesel"]
        },
        "Gati": {
            "Medium Truck": ["Diesel", "CNG", "Electric"],
            "Heavy Truck": ["Diesel"]
        },
        "Safeexpress": {
            "Medium Truck": ["Diesel"]
        },
        "TCL Freight": {
            "Medium Truck": ["Diesel"],
            "Heavy Truck": ["Diesel"]
        },
        "Rivigo": {
            "Heavy Truck": ["Diesel"]
        },
        "CONCOR": {
            "Cargo Train": ["Diesel", "Electric"]
        },
        "Gateway Rail Freight": {
            "Cargo Train": ["Diesel", "Electric"]
        },
        "Adani logistics": {
            "Cargo Train": ["Diesel", "Electric"]
        },
        "SpiceXpress": {
            "Cargo Plane": ["Jet Fuel"]
        }
    }

    # Vehicle to Mode Map
    mode_map = {
        "Bike": "Road", "Auto": "Road", "Van": "Road",
        "Light Truck": "Road", "Medium Truck": "Road", "Heavy Truck": "Road",
        "Cargo Train": "Rail", "Cargo Plane": "Air"
    }
    # User Inputs
    company = st.selectbox("Logistics Partner", list(company_vehicle_fuel_map.keys()))
    valid_vehicles = list(company_vehicle_fuel_map[company].keys())
    vehicle_type = st.selectbox("Vehicle Type", valid_vehicles)

    valid_fuels = company_vehicle_fuel_map[company][vehicle_type]
    fuel_type = st.selectbox("Fuel Type", valid_fuels)

    mode = mode_map.get(vehicle_type, "Road")
    st.text(f"Auto-selected Mode: {mode}")

    distance = st.number_input("Distance (in km)", min_value=1.0)
    vehicle_age = st.slider("Vehicle Age (in years)", 0, 100)
    vehicle_capacity = st.number_input("Vehicle Capacity (in kg)", min_value=1.0)
    load_capacity = st.number_input("Load Carried (in kg)", min_value=1.0)
    no_of_stops = st.slider("Number of Stops", 0, 300)
    avg_speed = st.number_input("Average Speed (km/h)", min_value=1.0)
    traffic_condition = st.selectbox("Traffic Condition", ["Low", "Moderate", "Heavy"])
    engine_norm_type = st.selectbox("Engine Norm Type", ["BS-III", "BS-IV", "BS-VI", "Electric", "Turbofan", "Turboprop", "Jet"])

    model_choice = st.radio("Choose Prediction Model", ["Random Forest", "XGBoost", "Linear Regression"])
    # Derived Feature Calculator
    def calculate_derived_features(fuel_type, distance, vehicle_capacity, load_capacity, no_of_stops, avg_speed):
        fuel_type_lower = fuel_type.lower()
        fe_dict = {
            'diesel': 20,
            'petrol': 45,
            'lng': 4,
            'cng': 15,
            'hybrid': 0.06,
            'electric': 0.035,
            'jet fuel': 0.3
        }

        adjusted_fuel_eff = fe_dict.get(fuel_type_lower, 15)

        if fuel_type_lower == 'electric':
            fuel_per_km = 0
            energy_kwh_per_km = adjusted_fuel_eff
            total_energy = energy_kwh_per_km * distance
        elif fuel_type_lower == 'hybrid':
            fuel_per_km = 1 / 45  # petrol component
            energy_kwh_per_km = 0.06
            total_energy = energy_kwh_per_km * distance
        else:
            fuel_per_km = 1 / adjusted_fuel_eff
            energy_kwh_per_km = 0
            total_energy = 0

        return {
            "adjusted_fuel_efficiency_in_km_litre": adjusted_fuel_eff,
            "fuel_per_km": fuel_per_km,
            "adjusted_energy_kwh_per_km": energy_kwh_per_km,
            "total_energy_kwh": total_energy,
            "speed_per_stop": avg_speed / (no_of_stops + 1),
            "load_utilization": round((load_capacity / vehicle_capacity) * 100, 2),
            "load_factor": load_capacity / vehicle_capacity
        }
    # Prediction
    if st.button("Predict CO₂ Emission"):
        derived = calculate_derived_features(
            fuel_type, distance, vehicle_capacity, load_capacity, no_of_stops, avg_speed
        )

        input_df = pd.DataFrame([{
            "logistics_partner": company,
            "vehicle_type": vehicle_type,
            "mode": mode,
            "distance_in_km_per_route": distance,
            "vehicle_age_in_years": vehicle_age,
            "vehicle_capacity": vehicle_capacity,
            "load_capacity": load_capacity,
            "no_of_stop": no_of_stops,
            "fuel_type": fuel_type,
            "average_speed_in_km_per_hr": avg_speed,
            "traffic_condition": traffic_condition,
            "engine_norm_type": engine_norm_type,
            **derived
        }])

        model = {
            "Random Forest": rf_model,
            "XGBoost": xgb_model,
            "Linear Regression": lr_model
        }.get(model_choice, rf_model)

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted CO₂ Emission: **{prediction:.2f} kg**")

        # Store in session state for recommendation page
        st.session_state['last_input'] = {
            "input_df": input_df, 
            'logistics_partner': company,
            'distance': distance,
            'mode' : mode,
            'vehicle_type': vehicle_type,
            'fuel_type': fuel_type,
            'load_capacity': load_capacity,
            'vehicle_capacity': vehicle_capacity,
            'no_of_stop' : no_of_stops,
            'average_speed_in_km_per_hr' : avg_speed,
            'traffic_condition' : traffic_condition,
            'vehicle_age_in_years' : vehicle_age,
            "engine_norm_type": engine_norm_type,
            "model_used": model_choice,
            'prediction': prediction
        }

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        
        
        
if page == "Recommendation":
  

    st.title("Emission Reduction Recommendations")
    st.markdown("Based on your previous input, here are **intelligent suggestions** to help reduce CO₂ emissions.")

    if 'last_input' not in st.session_state:
        st.warning("Please first predict emissions in the **'Emission Estimator'** page.")
        st.stop()

    # Load models
    rf_model = joblib.load("Notebook/rf_model_pipeline.pkl")
    xgb_model = joblib.load("Notebook/xgboost_pipeline.pkl")
    lr_model = joblib.load("Notebook/linearregression_pipeline.pkl")

    model_map = {
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "Linear Regression": lr_model
    }

    # Retrieve session data
    user_input = st.session_state['last_input']
    base_input = pd.DataFrame(user_input["input_df"]) if isinstance(user_input["input_df"], dict) else user_input["input_df"]
    actual_prediction = user_input["prediction"]
    distance = user_input["distance"]
    load_capacity = user_input["load_capacity"]
    vehicle_capacity = user_input["vehicle_capacity"]
    vehicle_type = user_input["vehicle_type"]
    fuel_type = user_input["fuel_type"]
    model_used = user_input.get("model_used", "Random Forest")

    model = model_map.get(model_used, rf_model)
    load_utilization = round((load_capacity / vehicle_capacity) * 100, 2)
    # Trip Summary
    st.subheader("Summary of Your Trip")
    st.markdown(f"""
    - **Vehicle Type**: {vehicle_type}  
    - **Fuel Type**: {fuel_type}  
    - **Distance**: {distance} km  
    - **Load Utilization**: {load_utilization}%  
    - **Predicted CO₂ Emission**: {actual_prediction:.2f} kg  
    """)
    # Alternative Suggestions
    st.subheader("Alternative Low-Emission Options")

    all_combinations = [
        ("Bike", "Electric"), ("Bike", "Petrol"),
        ("Auto", "Petrol"), ("Auto", "Electric"), ("Auto", "LNG"),
        ("Van", "Diesel"), ("Van", "Electric"), ("Van", "CNG"), ("Van", "Hybrid"),
        ("Light Truck", "Diesel"), ("Light Truck", "CNG"), ("Light Truck", "Electric"), ("Light Truck", "LNG"),
        ("Medium Truck", "Diesel"), ("Medium Truck", "CNG"), ("Medium Truck", "Electric"), ("Medium Truck", "LNG"),
        ("Heavy Truck", "Diesel"), ("Heavy Truck", "CNG"),
        ("Cargo Train", "Electric"), ("Cargo Train", "Diesel"),
        ("Cargo Plane", "Jet Fuel")
    ]

    results = []
    for vt, ft in all_combinations:
        if load_capacity > 100 and vt in ["Bike"]:
            continue
        if load_capacity > 300 and vt in ["Auto"]:
            continue
        if load_capacity > 2000 and vt in ["Van"]:
            continue
        if vt == "Cargo Plane" and distance < 500:
            continue
        if vt == "Bike" and distance > 150:
            continue
        
        modified_input = base_input.copy()
        modified_input.loc[:, "vehicle_type"] = vt
        modified_input.loc[:, "fuel_type"] = ft
        modified_input.loc[:, "mode"] = {"Cargo Plane": "Air", "Cargo Train": "Rail"}.get(vt, "Road")
        modified_input.loc[:, "engine_norm_type"] = {
            "Electric": "Electric",
            "Hybrid": "Hybrid",
            "Jet Fuel": "Jet"
        }.get(ft, "BS-VI")

        try:
            pred = model.predict(modified_input)[0]
            results.append((vt, ft, pred))
        except Exception as e:
            continue
        
    results.sort(key=lambda x: x[2])
    top_recommendations = results[:3]

    if top_recommendations:
        for vt, ft, pred in top_recommendations:
            st.markdown(f"Try **{vt}** using **{ft}** → Estimated Emission: **{pred:.2f} kg**")
    else:
        st.info("No better configurations found for this load/distance.")

    # Emission Level Feedback
    st.subheader(" Emission Level Assessment")
    if actual_prediction < 10:
        st.success(" **Low Emission Trip** – Great job!")
    elif actual_prediction < 30:
        st.info(" **Moderate Emission Trip** – Good, but could be better.")
    else:
        st.error(" **High Emission Trip** – Strongly consider switching to alternatives.")



if page == "Compare Models":
    st.title("Compare ML Models")
    st.markdown("Analyze performance of different models for CO₂ emission prediction.")
    # Load Test Data & Models
    df = pd.read_csv("dataset/carbon_emission_final_engineered.csv")
    feature_cols = [
        'logistics_partner', 'vehicle_type', 'mode',
        'distance_in_km_per_route', 'vehicle_age_in_years', 'vehicle_capacity_in_kg', 'load_capacity_in_kg',
        'no_of_stop', 'fuel_type', 'average_speed_in_km_per_hr', 'traffic_condition', 'engine_norm_type',
        'adjusted_fuel_efficiency_in_km_litre', 'fuel_per_km', 'adjusted_energy_kwh_per_km',
        'total_energy_kwh', 'speed_per_stop', 'load_utilization', 'load_factor'
    ]

    df = df.dropna(subset=feature_cols + ['c02_emission_kg'])

    X = df[feature_cols]
    y = df['c02_emission_kg']
    # Load Models
    rf_model = joblib.load("Notebook/rf_model_pipeline.pkl")
    xgb_model = joblib.load("Notebook/xgboost_pipeline.pkl")
    lr_model = joblib.load("Notebook/linearregression_pipeline.pkl")
    # Evaluate Models
    def evaluate_model(model, X, y):
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return mae, rmse, r2

    results = []

    for name, model in [("Random Forest", rf_model), ("XGBoost", xgb_model), ("Linear Regression", lr_model)]:
        mae, rmse, r2 = evaluate_model(model, X, y)
        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R² Score": round(r2, 4)
        })

    result_df = pd.DataFrame(results)
    # Show Results Table
    st.subheader("Model Performance Metrics")
    st.dataframe(result_df.set_index("Model"))
    # Plot Comparison
    st.subheader("Visual Comparison")

    fig, ax = plt.subplots(figsize=(8, 5))
    result_df_melted = result_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(data=result_df_melted, x="Metric", y="Score", hue="Model")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.grid(True)
    st.pyplot(fig)

    