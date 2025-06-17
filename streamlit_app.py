## data url: https://www.kaggle.com/datasets/therohithanand/used-car-price-prediction?resource=download

## Step 00 - Import of the packages
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
#from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from datetime import datetime

st.set_page_config(
    page_title="Used-Car Explorer",
    layout="centered",
    page_icon="🏎️",
)


## Step 01 - Setup
st.sidebar.title("🚗Used-Car Data Exploration Tool🚙")
page = st.sidebar.selectbox("Select Page",["Introduction 🚘","Visualization 📊", "Prediction 🔮"])


#st.video("video.mp4")

#st.image("house2.png")


def encode_data(df):
    df_encoded = df.copy()
    le = LabelEncoder()
    cols_to_encode = [
        "fuel_type", "brand", "transmission", "color",
        "service_history", "accidents_reported", "insurance_valid"
    ]
    for col in cols_to_encode:
        if col in df_encoded.columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded

st.write("   ")
st.write("   ")
st.write("   ")
df = pd.read_csv("usedCar.csv")
df["car_age"] = datetime.now().year - df["make_year"]
df_encoded = encode_data(df)

## Step 02 - Load dataset
if page == "Introduction 🚘":

    # Title
    st.title("🏁 Welcome to the Used-Car Explorer App")
    st.image("Used-Vehicles-banner.png", use_container_width=True)
    st.markdown(
        '''
        Welcome to the ultimate dashboard for **Used-Car Data Exploration**!  
        Navigate through the sidebar to:
        - 📊 Explore car data & price trends
        - 🔮 Evaluate custom car examples using Linear Regression
        - 📑 Generate automated EDA reports

        **Dataset Overview:** 10,000 used car listings with detailed specs.
        https://www.kaggle.com/datasets/therohithanand/used-car-price-prediction?resource=download

        ---
        '''
    )


    st.subheader("🚗 Sample Data")
    st.dataframe(df.head(10))

    st.success("Use the left sidebar to start exploring or forecasting!")

    st.caption("© 2025 Used-Car Explorer | Yazn & Ann-Mei")

elif page == "Visualization 📊":
    from sklearn.preprocessing import LabelEncoder
    ## Step 03 - Data Viz
    st.title("📊 Car Data Visualization")

    with st.sidebar:
        st.header("🔧 Filters")

        # Brand filter
        all_brands = sorted(df["brand"].unique())
        brands = st.multiselect("Brand", options=all_brands, default=all_brands)

        # Fuel type filter
        all_fuels = sorted(df["fuel_type"].unique())
        fuels = st.multiselect("Fuel type", options=all_fuels, default=all_fuels)

        # Transmission filter
        all_trans = sorted(df["transmission"].unique())
        transmissions = st.multiselect("Transmission", options=all_trans, default=all_trans)

        # Year range filter
        min_year, max_year = int(df["make_year"].min()), int(df["make_year"].max())
        year_range = st.slider("Manufacture year", min_year, max_year, (min_year, max_year))

        # Accident range filter
        max_acc = int(df["accidents_reported"].max())
        acc_range = st.slider("Accidents reported ≤", 0, max_acc, max_acc)

        st.markdown("---")
        if st.button("🔄 Reset filters"):
            st.experimental_rerun()

    # Apply filters
    mask = (
        df["brand"].isin(brands)
        & df["fuel_type"].isin(fuels)
        & df["transmission"].isin(transmissions)
        & df["make_year"].between(year_range[0], year_range[1])
        & (df["accidents_reported"] <= acc_range)
    )
    filtered = df.loc[mask]

    # ---------------------------
    # Metrics row
    # ---------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Average price", f"${filtered['price_usd'].mean():,.0f}")
    col2.metric("Avg. mileage (km/ℓ)", f"{filtered['mileage_kmpl'].mean():.1f}")
    col3.metric("Total cars", f"{len(filtered):,}")

    st.markdown("---")

    # ---------------------------
    # Tabs for visualisations
    # ---------------------------
    price_tab, mileage_tab, engine_tab, accident_tab, age_tab, corr_tab = st.tabs([
        "💰 Price by Brand",
        "⛽ Mileage vs Price",
        "⚙️ Engine vs Price",
        "🚧 Accidents Distribution",
        "📈 Age vs Price",
        "📊 Correlation Heatmap",
    ])

    # 1. Price by Brand
    with price_tab:
        st.subheader("Average Price by Brand")
        avg_price = (
            filtered.groupby("brand")["price_usd"].mean().reset_index().sort_values("price_usd", ascending=False)
        )
        bar = (
            alt.Chart(avg_price, height=400)
            .mark_bar()
            .encode(
                x=alt.X("brand:N", sort="-y", title="Brand"),
                y=alt.Y("price_usd:Q", title="Average price (USD)"),
                tooltip=["brand", alt.Tooltip("price_usd", format=",.0f")],
            )
        )
        st.altair_chart(bar, use_container_width=True)

    # 2. Mileage vs Price
    with mileage_tab:
        st.subheader("Mileage vs Price (colored by fuel type)")
        scatter = (
            alt.Chart(filtered, height=400)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("mileage_kmpl:Q", title="Mileage (km/ℓ)"),
                y=alt.Y("price_usd:Q", title="Price (USD)"),
                color=alt.Color("fuel_type:N", title="Fuel"),
                tooltip=["brand", "fuel_type", "mileage_kmpl", "price_usd"],
            )
        )
        st.altair_chart(scatter.interactive(), use_container_width=True)

    # 3. Engine vs Price
    with engine_tab:
        st.subheader("Engine size vs Price")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(data=filtered, x="engine_cc", y="price_usd", hue="transmission", inner="quartile", ax=ax)
        ax.set_title("Price distribution by Engine Size")
        st.pyplot(fig)

    # 4. Accidents distribution
    with accident_tab:
        st.subheader("Distribution of Reported Accidents")
        hist = (
            alt.Chart(filtered, height=400)
            .mark_bar()
            .encode(
                alt.X("accidents_reported:Q", bin=alt.Bin(maxbins=30), title="Accidents reported"),
                alt.Y("count():Q", title="Number of cars"),
                tooltip=[alt.Tooltip("count():Q", format=",")],
            )
        )
        st.altair_chart(hist, use_container_width=True)

    # 5. Car age vs Price
    with age_tab:
        st.subheader("Car Age vs Price")
        filtered["age_group"] = pd.cut(filtered["car_age"], bins=[0, 3, 6, 10, 15, 25], labels=["0-3", "4-6", "7-10", "11-15", "16+"])
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(data=filtered, x="age_group", y="price_usd", inner="quartile", ax=ax)
        ax.set_title("Price distribution by Car Age Group")
        st.pyplot(fig)

    # 6. Correlation heatmap
    with corr_tab:
        st.subheader("Correlation Heatmap (numeric columns)")
        numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
        corr_matrix = filtered[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig, use_container_width=True)

    # ---------------------------
    # Download button
    # ---------------------------
    with st.expander("⬇️ Download filtered data as CSV"):
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="filtered_used_cars.csv", mime="text/csv")

    st.subheader("Categorical Distribution Pie Charts")
    cat_cols = ["fuel_type", "transmission", "service_history", "insurance_valid"]
    for col in cat_cols:
        st.markdown(f"### {col.replace('_', ' ').title()}")
        pie_data = filtered[col].value_counts().reset_index()
        pie_data.columns = [col, "count"]

        fig, ax = plt.subplots()
        ax.pie(pie_data["count"], labels=pie_data[col], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    st.caption("© 2025 Used-Car Explorer | Yazn & Ann-Mei")

elif page == "Prediction 🔮":
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn import metrics
    import matplotlib.pyplot as plt
    st.title("🔮 Predict Used Car Prices")

    # Drop NA
    df_encoded = df_encoded.dropna()

    # Feature selection
    features = [col for col in df_encoded.columns if col != 'price_usd']
    target = 'price_usd'

    # Sidebar inputs
    st.sidebar.header("🔧 Prediction Settings")
    selected_features = st.sidebar.multiselect("Select features", features, default=features)
    future_years = st.sidebar.slider("Predict how many years into the future?", 1, 10, 5)

    if selected_features:

        # Update X to use future_age
        X = df_encoded[selected_features].copy()

        y = df_encoded[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        st.subheader("📊 Model Evaluation")
        st.write("- Mean Absolute Error (MAE):", round(metrics.mean_absolute_error(y_test, predictions), 2))
        st.write("- Mean Squared Error (MSE):", round(metrics.mean_squared_error(y_test, predictions), 2))
        st.write("- R² Score:", round(metrics.r2_score(y_test, predictions), 3))

        # Scatter plot
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        ax.set_xlabel("Actual Prices")
        ax.set_ylabel("Predicted Prices")
        ax.set_title("Actual vs Predicted Prices")
        st.pyplot(fig)
    else:
        st.warning("Please select at least one feature to continue.")
    



    st.subheader("🔮 Predict the Price of a Custom Car")

    df_model = df.dropna()
    le_dict = {}
    for col in ["fuel_type", "brand", "transmission", "color", "service_history", "insurance_valid"]:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le

    df_model["age"] = 2025 - df_model["make_year"]
    features = ["mileage_kmpl", "engine_cc", "owner_count", "accidents_reported", "age"] + list(le_dict.keys())
    X = df_model[features]
    y = df_model["price_usd"]

    model = LinearRegression()
    model.fit(X, y)

    with st.form("custom_input"):
        st.write("### Enter Car Specifications")
        mileage = st.number_input("Mileage (km/ℓ)", min_value=0.0, value=15.0)
        engine = st.number_input("Engine size (cc)", min_value=600, value=1500)
        owners = st.number_input("Number of previous owners", min_value=0, value=1)
        accidents = st.number_input("Number of accidents reported", min_value=0, value=0)
        age = st.number_input("Car age (years)", min_value=0, value=5)

        inputs = []
        for col in le_dict:
            options = list(le_dict[col].classes_)
            selected = st.selectbox(col.replace('_', ' ').title(), options)
            inputs.append(le_dict[col].transform([selected])[0])

        if st.form_submit_button("Predict Price"):
            custom_features = np.array([[mileage, engine, owners, accidents, age] + inputs])
            price_pred = model.predict(custom_features)[0]
            st.success(f"Estimated price: ${price_pred:,.2f}")

    st.caption("© 2025 Used-Car Explorer | Yazn & Ann-Mei")
