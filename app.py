import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import requests
import io
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
import base64
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore only convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def encode_gif(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

undiscovered_gif = encode_gif("undiscovered.gif")
astronaut_gif = encode_gif("astronaut.gif")
habitable_gif = encode_gif("habitable.gif")
spaceship_gif = encode_gif("spaceshipineed.gif")

st.set_page_config(
    page_title="EXO'25",
    layout="wide",            # 👈 Enables wide mode
    initial_sidebar_state="auto"
)

# Load local background image and encode to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    

def set_pixel_theme():
    encoded_image = get_base64_image("whatineed2.png")  # Update path if needed

    custom_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Press Start 2P', monospace !important;
        color: #ffa72c !important;
        text-align: left !important;
    }}

    .stApp {{
        background-image: 
            linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
            url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    h1, h2, h3, h4, h5, h6, p, div, span, label {{
        font-family: 'Press Start 2P', monospace !important;
        color: #ffa72c !important;
        text-shadow: 0 0 5px #ffcc70;
    }}

    .stSelectbox, .stButton > button, .stSlider, .stTextInput {{
        background-color: #1a1a2e !important;
        color: #ffa72c !important;
        border: 2px solid #ffa72c !important;
        font-family: 'Press Start 2P', monospace !important;
        box-shadow: 0 0 15px #CC5500;
        padding: 10px;
        border-radius: 12px;
        transition: all 0.3s ease-in-out;
    }}

    .stSelectbox:hover, .stButton > button:hover {{
        background-color: #CC5500 !important;
        color: #1a1a2e !important;
        transform: scale(1.05);
        box-shadow: 0 0 25px #ffa72c;
    }}

    .stPlotlyChart {{
        border-radius: 15px;
        box-shadow: 0 0 25px #CC5500;
        background-color: rgba(10, 10, 25, 0.9);
        padding: 1.5rem;
    }}

    header[data-testid="stHeader"] {{
        background-color: rgba(10, 10, 25, 0.95);
        border-bottom: 2px solid #ffa72c;
    }}

    section[data-testid="stSidebar"] {{
        background-color: rgba(10, 10, 25, 0.92);
        border-right: 2px solid #ffa72c;
    }}

    section[data-testid="stSidebar"] .css-1d391kg {{
        color: #ffa72c !important;
    }}

    .css-1aumxhk, .css-1kyxreq, .css-10trblm {{
        font-family: 'Press Start 2P', monospace !important;
    }}

    /* Typewriter intro */
    #intro-typewriter {{
        width: 100%;
        font-family: 'Press Start 2P', monospace;
        white-space: nowrap;
        overflow: hidden;
        border-right: 2px solid #ffa72c;
        font-size: 16px;
        animation: typing 4s steps(40, end), blink 0.8s step-end infinite alternate;
    }}

    @keyframes typing {{
        from {{ width: 0; }}
        to {{ width: 100%; }}
    }}

    @keyframes blink {{
        50% {{ border-color: transparent; }}
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# 🎨 Activate the pixel art theme
set_pixel_theme()

# Load NASA Exoplanet dataset
@st.cache_data
def load_nasa_data():
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv"
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.text), low_memory=False)
    return df

@st.cache_data
def load_phl_data():
    path = "hwc_table_all.csv" 
    phl_df = pd.read_csv(path)
    return phl_df

def merge_and_reduce_nasa_phl(nasa_df, phl_df):
    # Normalize planet names for matching
    nasa_df['pl_name_clean'] = nasa_df['pl_name'].str.lower().str.replace(" ", "").str.strip()
    phl_df['Name_clean'] = phl_df['Name'].str.lower().str.replace(" ", "").str.strip()

    # Assign habitability labels
    habitable_set = set(phl_df['Name_clean'])
    nasa_df['pl_habitable'] = nasa_df['pl_name_clean'].apply(lambda x: 1 if x in habitable_set else 0)

    # Drop helper column
    nasa_df.drop(columns=['pl_name_clean'], inplace=True)

    # Reduce dataset size: keep all habitable, sample 6% of non-habitable
    habitable_df = nasa_df[nasa_df['pl_habitable'] == 1]
    non_habitable_df = nasa_df[nasa_df['pl_habitable'] == 0]
    non_habitable_sampled = non_habitable_df.sample(frac=0.06, random_state=42)

    reduced_df = pd.concat([habitable_df, non_habitable_sampled], ignore_index=True)

    return reduced_df

@st.cache_data
def preprocess_data(df_raw, threshold=0.4):
    df = df_raw.copy()

    # Drop high-missing columns
    missing_ratio = df.isnull().mean()
    df = df.loc[:, missing_ratio <= threshold]

    # Separate numerical/categorical
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Impute numerics
    num_imputer = SimpleImputer(strategy="mean")
    df_num = pd.DataFrame(num_imputer.fit_transform(df[num_cols]), columns=num_cols, index=df.index)

    # Scale numerics
    df_num = pd.DataFrame(RobustScaler().fit_transform(df_num), columns=num_cols, index=df.index)

    # Impute categoricals
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df_cat = pd.DataFrame(cat_imputer.fit_transform(df[cat_cols]), columns=cat_cols, index=df.index)

    # Recombine
    df_clean = pd.concat([df_num, df_cat], axis=1)

    # Reattach 'pl_name' from original df for matching
    if 'pl_name' in df_raw.columns:
        df_clean['pl_name'] = df_raw['pl_name'].values

    return df_clean

page = st.sidebar.selectbox("Navigate", ["Usage","EDA", "Model Fitting", "EXO’25 Game"])

# Setup GIF placeholder
gif_placeholder = st.empty()

with open("gifineed-ezgif.com-resize.gif", "rb") as f:
    gif_data = base64.b64encode(f.read()).decode()

# Display the GIF while loading
gif_placeholder.markdown(
    f"""
    <div style='text-align: center; margin-top: 3rem;'>
        <img src="data:image/gif;base64,{gif_data}" width="400">
        <p style='color: #ffa72c; font-family: "Press Start 2P", monospace; margin-top: 1rem;'>Scanning the Galaxy...</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 🔄 Load the actual data (this takes time)
df = load_nasa_data()
phl_df = load_phl_data()
gif_placeholder.empty()

df_reduced = merge_and_reduce_nasa_phl(df, phl_df)
df_clean = preprocess_data(df_reduced)

column_names = df_clean.columns.tolist()

# Save to a text file
with open("column_names.txt", "w") as f:
    for col in column_names:
        f.write(f"{col}\n")

y = df_clean["pl_habitable"]
X_latest = df_clean.drop(columns=["pl_habitable", "habitability_score"], errors="ignore")
burnt_orange_colors = ["#CC5500", "#FF7F11", "#FF9F1C", "#D16002", "#A63A00", "#E2711D"]
color_discrete_sequence=burnt_orange_colors
color_continuous_scale=burnt_orange_colors

if page == "Usage":
    st.title("📘 How to Use the EXO’25 App")

    st.markdown("""
    Welcome to **EXO’25**, your space lab to explore the habitability of exoplanets using advanced machine learning and a retro-themed interactive RL game! 🚀✨

    ---

    ### 🧭 Navigation Overview
    The app has four pages accessible from the **left sidebar**:

    1. **📊 EDA (Exploratory Data Analysis)**  
       - Visualize trends across key planet and star features  
       - Explore relationships with plots like:
         - Radius vs Galactic Longitude
         - Correlation Heatmaps
         - Radar Plots and Violin Distributions

    2. **🧠 Model Fitting**  
       - Train and evaluate **three machine learning models**:
         - CatBoost
         - Histogram-based Gradient Boosting
         - Logistic Regression
       - Enable optional **feature selection** using mutual information
       - Models are trained with **increased complexity** for better performance
       - View classification report + confusion matrix

    3. **🎮 EXO’25 Game**  
       - RL-based interactive pixel game  
       - Play as a space explorer or let the **agent** prioritize planet visits  
       - Choose your ML model to power the agent's habitability decisions  
       - Track **score, rewards**, and **planet visit logs**


    ---

    ### 🧬 Dataset & Preprocessing
    - Combined **NASA Exoplanet Archive** and **PHL Habitable Catalog**
    - Normalized planet names to match labels
    - Sampled 6% of non-habitable planets to reduce class imbalance
    - Dropped features with >40% missing values
    - Imputed & scaled numeric and categorical columns

    ---

    ### 🧪 ML Models Used
    - **CatBoostClassifier**: 1000 iterations, depth 10, learning rate 0.05  
    - **HistGradientBoosting**: 500 iterations, max depth 10, regularization 1.0  
    - **LogisticRegression**: Max iterations 2000, regularized with C=0.1  
    - Optional checkbox lets user train on **top 50 features only**  

    ---

    ### 🎮 Gameplay Logic
    - 50 randomly selected exoplanets with predicted habitability
    - Budget to explore 15 planets
    - Rewards: +10 for habitable, -1 for non-habitable
    - Powered by Q-learning with a dynamically updated Q-table
    - Choose model to drive the agent’s behavior

    ---

    ### 🪄 Tips for Best Experience
    - Use **desktop + wide mode** for full visual effect  
    - Try the **"Use Top Features"** checkbox before training  
    - Toggle between manual or agent-driven gameplay  
    - Keep your curiosity cosmic 🌌

    """)

    
elif page == "EDA":
    st.title("📊 Exoplanet EDA")

    numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_clean.select_dtypes(exclude=np.number).columns.tolist()

    eda_option = st.selectbox(
        "Choose a visualization",
        [
            "Target Balance",
            "Correlation Heatmap",
            "Habitable vs Non-habitable: Radius vs Longitude",
            "Box: Stellar Temperature by Habitability",
            "Violin: Stellar Mass by Habitability",
            "Radar Plot: Earth-like Features",
            "2D Density: Radius vs Mass",
            "Facet Plot: Radius by Discovery Method",
            "Bubble: Stellar Mass vs Radius vs Temp",
            "Pairwise Scatter (Top 5)"
        ]
    )

    if eda_option == "Target Balance":
        fig = px.pie(
                df_clean,
                names="pl_habitable",
                title="Habitable vs Non-Habitable Planets",
                color_discrete_sequence=burnt_orange_colors
                )

        st.plotly_chart(fig, use_container_width=True)

    elif eda_option == "Correlation Heatmap":
        corr = df_clean[numeric_cols].corr()
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            annotation_text=corr.round(2).values,
            colorscale=burnt_orange_colors, showscale=True
        )
        fig.update_layout(title="Correlation Heatmap", height=700)
        st.plotly_chart(fig, use_container_width=True)

    elif eda_option == "Habitable vs Non-habitable: Radius vs Longitude":
        if "pl_rade" in df_clean.columns and "glon" in df_clean.columns:
            fig = px.scatter(
                df_clean, x="pl_rade", y="glon", color="pl_habitable",
                color_discrete_sequence=burnt_orange_colors,
                title="Habitability by Radius vs Galactic Longitude",
                labels={"glon": "Galactic Longitude", "pl_rade": "Planet Radius"}
            )
            st.plotly_chart(fig, use_container_width=True)

    elif eda_option == "Box: Stellar Temperature by Habitability" and "st_teff" in df_clean.columns:
        fig = px.box(df_clean, x="pl_habitable", y="st_teff", points="all", title="Stellar Temperature by Habitability", color_discrete_sequence=burnt_orange_colors)
        st.plotly_chart(fig, use_container_width=True)

    elif eda_option == "Violin: Stellar Mass by Habitability" and "st_mass" in df_clean.columns:
        fig = px.violin(df_clean, y="st_mass", x="pl_habitable", box=True, points="all", title="Stellar Mass by Habitability", color_discrete_sequence=burnt_orange_colors)
        st.plotly_chart(fig, use_container_width=True)

    elif eda_option == "Radar Plot: Earth-like Features":
        radar_cols = ['pl_rade', 'st_mass', 'st_teff', 'st_rad']
        radar_cols = [col for col in radar_cols if col in df_clean.columns]
        if radar_cols:
            hab_avg = df_clean[df_clean["pl_habitable"] == 1][radar_cols].mean()
            nonhab_avg = df_clean[df_clean["pl_habitable"] == 0][radar_cols].mean()
            df_radar = pd.DataFrame({
                "feature": radar_cols,
                "Habitable": hab_avg.values,
                "Non-Habitable": nonhab_avg.values
            })
            df_radar = df_radar.melt(id_vars="feature", var_name="Type", value_name="Value")
            fig = px.line_polar(df_radar, r="Value", theta="feature", color="Type", line_close=True, title="Radar: Earth-like Features", color_discrete_sequence=burnt_orange_colors)
            st.plotly_chart(fig, use_container_width=True)

    elif eda_option == "2D Density: Radius vs Mass":
        if all(x in df_clean.columns for x in ["pl_rade", "st_mass"]):
            fig = px.density_contour(df_clean, x="pl_rade", y="st_mass", color="pl_habitable",
                                     title="2D Density: Radius vs Stellar Mass", color_discrete_sequence=burnt_orange_colors)
            st.plotly_chart(fig, use_container_width=True)

    elif eda_option == "Facet Plot: Radius by Discovery Method":
        if "pl_rade" in df_clean.columns and "discoverymethod" in df_clean.columns:
            fig = px.histogram(df_clean, x="pl_rade", facet_col="discoverymethod",
                               color="pl_habitable", title="Planet Radius by Discovery Method", color_discrete_sequence=burnt_orange_colors)
            st.plotly_chart(fig, use_container_width=True)

    elif eda_option == "Bubble: Stellar Mass vs Radius vs Temp":
        if all(x in df_clean.columns for x in ["st_mass", "pl_rade", "st_teff"]):
            fig = px.scatter(df_clean, x="st_mass", y="pl_rade",
                             color="pl_habitable", hover_name="pl_name",
                             title="Bubble Chart: Stellar Mass vs Radius vs Temperature", color_discrete_sequence=burnt_orange_colors)
            st.plotly_chart(fig, use_container_width=True)

    elif eda_option == "Pairwise Scatter (Top 5)":
        top5 = df_clean[numeric_cols].iloc[:, 5:10]
        fig = px.scatter_matrix(top5, dimensions=top5.columns, color=df_clean["pl_habitable"],
                                title="Pairwise Scatter Matrix (Top 5 Numeric Features)", color_discrete_sequence=burnt_orange_colors)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Fitting":
    st.title("🤖 Model Selection & Training")
        
    st.markdown("### Choose your model(Just 🐻 with me)")
    col1, col2 = st.columns([2, 2])
    with col1:
        model_choice = st.selectbox("Select a model", ["LogisticRegression", "CatBoost", "HistGradientBoosting"])
    with col2:
        use_top_features = st.checkbox("Use Top Features Only (Auto)", value=False)

    if st.button("Train Selected Model"):
        loading_placeholder = st.empty()
        with open("thinking.gif", "rb") as f:
            gif_data = base64.b64encode(f.read()).decode()

        loading_placeholder.markdown(
            f"""
            <div style='text-align: center; margin-top: 3rem;'>
                <img src="data:image/gif;base64,{gif_data}" width="300">
                <p style='color: #ffa72c; font-family: "Press Start 2P", monospace; margin-top: 1rem;'>🚀 Training Your Model...</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Handle categoricals (CatBoost supports strings; others need dummies)
        if model_choice == "CatBoost":
            X_input = X_latest.copy()
        else:
            cat_cols = X_latest.select_dtypes(exclude=np.number).columns.tolist()
            X_input = pd.get_dummies(X_latest, columns=cat_cols, drop_first=True)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_input, y_encoded, test_size=0.2)

        # 🧠 Feature selection (optional)
        if use_top_features:
            from sklearn.feature_selection import mutual_info_classif
            if model_choice == "CatBoost":
                X_numeric = X_train.select_dtypes(include=np.number)
            else:
                X_numeric = pd.DataFrame(X_train, columns=X_input.columns).select_dtypes(include=np.number)

            top_k = 50
            mi = mutual_info_classif(X_numeric, y_train)
            top_indices = np.argsort(mi)[::-1][:top_k]
            top_features = X_numeric.columns[top_indices]

            X_train = X_train[top_features]
            X_test = X_test[top_features]

        if model_choice == "CatBoost":
            cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            model = CatBoostClassifier(
                iterations=1000,
                depth=10,
                learning_rate=0.05,
                l2_leaf_reg=5,
                verbose=0
            )
            model.fit(X_train, y_train, cat_features=cat_cols)
            y_pred = model.predict(X_test)

        elif model_choice == "HistGradientBoosting":
            model = HistGradientBoostingClassifier(
                max_iter=500,
                max_depth=10,
                learning_rate=0.05,
                l2_regularization=1.0
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_choice == "LogisticRegression":
            model = LogisticRegression(
                max_iter=1000,
                C=0.1,
                penalty="l2",
                solver="lbfgs"
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)


        loading_placeholder.empty()
        # Report
        st.markdown("### 📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix", labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig_cm, use_container_width=True)

elif page == "EXO’25 Game":

    with open("trial2.html", "r", encoding="utf-8") as f:
        html_template = f.read()

    # Replace GIF paths with base64 versions
    html_final = html_template \
        .replace("undiscovered.gif", f"data:image/gif;base64,{undiscovered_gif}") \
        .replace("astronaut.gif", f"data:image/gif;base64,{astronaut_gif}") \
        .replace("habitable.gif", f"data:image/gif;base64,{habitable_gif}") \
        .replace("spaceshipineed.gif", f"data:image/gif;base64,{spaceship_gif}")

    components.html(html_final, height=1000, scrolling=False)


