import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------
# 1. Load the dataset
# -------------------
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "your_health_data.csv")

df = pd.read_csv(file_path)

# -------------------
# 2. Streamlit App Title
# -------------------
st.set_page_config(page_title="Health Dashboard", layout="wide")
st.title("📊 Health Data Dashboard")
st.write("Analysis of multiple health conditions from a single dataset.")

# -------------------
# 3. Show raw data
# -------------------
with st.expander("View Raw Data"):
    st.dataframe(df)

# -------------------
# 4. Basic dataset info
# -------------------
st.subheader("Dataset Overview")
st.write(f"**Number of records:** {df.shape[0]}")
st.write(f"**Number of columns:** {df.shape[1]}")
st.write("**Columns:**", ", ".join(df.columns))

# -------------------
# 5. Sidebar Filters
# -------------------
st.sidebar.header("Filters")
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = df.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

selected_column = st.sidebar.selectbox("Select a numeric column for histogram:", numeric_columns)
selected_category = None
if categorical_columns:
    selected_category = st.sidebar.selectbox("Select a category column for filtering:", categorical_columns)
    unique_values = df[selected_category].dropna().unique().tolist()
    selected_value = st.sidebar.selectbox("Select value to filter:", unique_values)
    df_filtered = df[df[selected_category] == selected_value]
else:
    df_filtered = df

# -------------------
# 6. Histogram
# -------------------
st.subheader(f"Histogram of {selected_column}")
fig, ax = plt.subplots()
ax.hist(df_filtered[selected_column].dropna(), bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel(selected_column)
ax.set_ylabel("Count")
st.pyplot(fig)

# -------------------
# 7. Correlation Heatmap
# -------------------
st.subheader("Correlation Heatmap")
if len(numeric_columns) > 1:
    corr = df[numeric_columns].corr()
    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im)
    st.pyplot(fig)
else:
    st.write("Not enough numeric columns for correlation heatmap.")

# -------------------
# 8. Summary Statistics
# -------------------
st.subheader("Summary Statistics")
st.write(df.describe())











# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# df = pd.read_csv('data/your_health_data.csv') 

# # Health disease prevention tips
# health_prevention_tips = {
#     "Diabetes": [
#         "Maintain a balanced diet focusing on low sugars.",
#         "Engage in regular physical activity.",
#         "Monitor blood sugar levels regularly.",
#         "Stay hydrated and manage stress levels.",
#         "Consult a healthcare provider for regular check-ups."
#     ],
#     "Hypertension": [
#         "Limit salt intake and eat more fruits and vegetables.",
#         "Stay active and maintain a healthy weight.",
#         "Reduce alcohol consumption and quit smoking.",
#         "Monitor blood pressure regularly.",
#         "Manage stress through relaxation techniques."
#     ],
#     "Heart Disease": [
#         "Eat heart-healthy foods like whole grains.",
#         "Maintain a healthy weight and exercise regularly.",
#         "Avoid tobacco and limit unhealthy fats.",
#         "Schedule regular check-ups and screenings.",
#         "Stay informed about heart health risks."
#     ],
#     "Obesity": [
#         "Eat a balanced diet rich in fruits and vegetables.",
#         "Engage in regular physical activities.",
#         "Monitor portion sizes and avoid sugary drinks.",
#         "Seek support from healthcare professionals.",
#         "Be mindful of emotional eating triggers."
#     ],
#     "Respiratory Diseases": [
#         "Avoid exposure to pollutants and smoke.",
#         "Stay physically active and maintain a healthy diet.",
#         "Get vaccinated against flu and pneumonia.",
#         "Practice good hygiene and avoid infections.",
#         "Consult a doctor when experiencing symptoms."
#     ]
# }

# # Set visual theme
# sns.set(style="whitegrid")
# st.set_page_config(page_title="Health Disease Analyzer", layout="wide")
# st.title("🩺 Health Disease Analysis Dashboard")
# # Background color toggle
# theme = st.selectbox("🎨 Choose Theme", ["Light", "Dark"])

# # Define custom CSS for theme
# if theme == "Dark":
#     st.markdown("""
#         <style>
#         body {
#             background-color: #1e1e1e;
#             color: white;
#         }
#         .stApp {
#             background-color: #1e1e1e;
#             color: white;
#         }
#         </style>
#     """, unsafe_allow_html=True)
# else:
#     st.markdown("""
#         <style>
#         body {
#             background-color: #f5f5f5;
#             color: black;
#         }
#         .stApp {
#             background-color: #f5f5f5;
#             color: black;
#         }
#         </style>
#     """, unsafe_allow_html=True)

# # Base path for the data files
# base_path = "C:/Users/HP/Desktop/health_disease_analysis/data"

# # Sidebar menu
# section = st.sidebar.selectbox("Select Analysis Section", [
#     "Overview Dashboard",
#     "Diabetes",
#     "Hypertension",
#     "Heart Disease",
#     "Obesity",
#     "Respiratory Diseases",
#     "📊 ML Prediction"
# ])
# st.sidebar.markdown("## 🛡 Health Prevention Guide")
# selected_disease = st.sidebar.selectbox("Select a Health Disease", list(health_prevention_tips.keys()))

# # ------------------------ 1. DIABETES ------------------------
# if section == "Diabetes":
#     df = pd.read_csv(f"{base_path}/diabetes_data.csv")
#     df.columns = [c.strip().replace(" ", "_") for c in df.columns]
#     summary = df.groupby("Region")["Cases"].sum().sort_values(ascending=False).head(10)

#     st.subheader("🍭 Top 10 Regions with Highest Diabetes Cases")
#     summary_df = summary.reset_index()
#     summary_df.index = summary_df.index + 1
#     st.dataframe(summary_df)

#     # Visualization
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(x=summary.values, y=summary.index, palette="Blues_r", ax=ax)
#     for i, v in enumerate(summary.values):
#         ax.text(v + 100, i, str(v), color='black', va='center', fontweight='bold')
#     ax.set_xlabel("Cases of Diabetes")
#     ax.set_ylabel("Region")
#     ax.set_title("Top 10 Regions with Highest Diabetes Cases")
#     st.pyplot(fig)

#     # Trend
#     trend = df.groupby("Year")["Cases"].sum().reset_index()
#     st.subheader("📈 Diabetes Trend Over the Years")
#     fig2, ax2 = plt.subplots(figsize=(10, 6))
#     sns.lineplot(data=trend, x="Year", y="Cases", marker="o", color="orange", ax=ax2)
#     ax2.set_xlabel("Year")
#     ax2.set_ylabel("Total Diabetes Cases")
#     ax2.set_title("Trend of Diabetes Cases Over Years")
#     st.pyplot(fig2)

# # ------------------------ 2. HYPERTENSION ------------------------
# elif section == "Hypertension":
#     df = pd.read_csv(f"{base_path}/hypertension_data.csv")
#     df.columns = [c.strip().replace(" ", "_") for c in df.columns]
#     summary = df.groupby("Region")["Cases"].sum().sort_values(ascending=False).head(10)

#     st.subheader("📈 Top 10 Regions with Highest Hypertension Cases")
#     summary_df = summary.reset_index()
#     summary_df.index = summary_df.index + 1
#     st.dataframe(summary_df)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(x=summary.values, y=summary.index, palette="Greens", ax=ax)
#     for i, v in enumerate(summary.values):
#         ax.text(v + 50, i, str(v), color='black', va='center', fontweight='bold')
#     ax.set_title("Top 10 Regions with Highest Hypertension Cases")
#     st.pyplot(fig)

#     # Trend
#     trend = df.groupby("Year")["Cases"].sum().reset_index()
#     st.subheader("📈 Trend of Hypertension Cases Over the Years")
#     fig2, ax2 = plt.subplots(figsize=(10, 5))
#     sns.lineplot(data=trend, x="Year", y="Cases", marker="o", color="green", ax=ax2)
#     ax2.set_title("Hypertension Cases Over the Years")
#     ax2.set_xlabel("Year")
#     ax2.set_ylabel("Number of Cases")
#     st.pyplot(fig2)

# # ------------------------ 3. HEART DISEASE ------------------------
# elif section == "Heart Disease":
#     df = pd.read_csv(f"{base_path}/heart_disease_data.csv")
#     df.columns = [c.strip().replace(" ", "_") for c in df.columns]
#     summary = df.groupby("Region")["Cases"].sum().sort_values(ascending=False).head(10)

#     st.subheader("❤️ Top 10 Regions with Highest Heart Disease Cases")
#     summary_df = summary.reset_index()
#     summary_df.index = summary_df.index + 1
#     st.dataframe(summary_df)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(x=summary.values, y=summary.index, palette="Reds", ax=ax)
#     ax.set_title("Top 10 Regions with Highest Heart Disease Cases")
#     st.pyplot(fig)

#     # Trend
#     trend = df.groupby("Year")["Cases"].sum().reset_index()
#     st.subheader("📈 Trend of Heart Disease Cases Over the Years")
#     fig2, ax2 = plt.subplots(figsize=(10, 5))
#     sns.lineplot(data=trend, x="Year", y="Cases", marker="o", color="red", ax=ax2)
#     ax2.set_xlabel("Year")
#     ax2.set_ylabel("Number of Cases")
#     st.pyplot(fig2)

# # ------------------------ 4. OBESITY ------------------------
# elif section == "Obesity":
#     df = pd.read_csv(f"{base_path}/obesity_data.csv")
#     df.columns = [c.strip().replace(" ", "_") for c in df.columns]
#     summary = df.groupby("Region")["Cases"].sum().sort_values(ascending=False).head(10)

#     st.subheader("🏋️‍♀️ Top 10 Regions with Highest Obesity Cases")
#     summary_df = summary.reset_index()
#     summary_df.index = summary_df.index + 1
#     st.dataframe(summary_df)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(x=summary.values, y=summary.index, palette="Purples", ax=ax)
#     ax.set_title("Top 10 Regions with Highest Obesity Cases")
#     st.pyplot(fig)

#     # Trend
#     trend = df.groupby("Year")["Cases"].sum().reset_index()
#     st.subheader("📈 Trend of Obesity Cases Over the Years")
#     fig2, ax2 = plt.subplots(figsize=(10, 5))
#     sns.lineplot(data=trend, x="Year", y="Cases", marker="o", color="purple", ax=ax2)
#     ax2.set_xlabel("Year")
#     ax2.set_ylabel("Number of Cases")
#     st.pyplot(fig2)

# # ------------------------ 5. RESPIRATORY DISEASES ------------------------
# elif section == "Respiratory Diseases":
#     df = pd.read_csv(f"{base_path}/respiratory_diseases_data.csv")
#     df.columns = [c.strip().replace(" ", "_") for c in df.columns]
#     summary = df.groupby("Region")["Cases"].sum().sort_values(ascending=False).head(10)

#     st.subheader("💨 Top 10 Regions with Highest Respiratory Disease Cases")
#     summary_df = summary.reset_index()
#     summary_df.index = summary_df.index + 1
#     st.dataframe(summary_df)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(x=summary.values, y=summary.index, palette="Blues", ax=ax)
#     ax.set_title("Top 10 Regions with Highest Respiratory Disease Cases")
#     st.pyplot(fig)

#     # Trend
#     trend = df.groupby("Year")["Cases"].sum().reset_index()
#     st.subheader("📈 Trend of Respiratory Disease Cases Over the Years")
#     fig2, ax2 = plt.subplots(figsize=(10, 5))
#     sns.lineplot(data=trend, x="Year", y="Cases", marker="o", color="blue", ax=ax2)
#     ax2.set_xlabel("Year")
#     ax2.set_ylabel("Number of Cases")
#     st.pyplot(fig2)

# #------------------------ML Prediction------------------
# elif section == "📊 ML Prediction":
#     from sklearn.linear_model import LinearRegression
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import r2_score

#     st.subheader("📊 Predict Future Health Disease Cases with Machine Learning")

#     health_options = {
#         "Diabetes": {
#             "file": "diabetes_data.csv",
#             "column": "Cases"
#         },
#         "Hypertension": {
#             "file": "hypertension_data.csv",
#             "column": "Cases"
#         },
#         "Heart Disease": {
#             "file": "heart_disease_data.csv",
#             "column": "Cases"
#         },
#         "Obesity": {
#             "file": "obesity_data.csv",
#             "column": "Cases"
#         },
#         "Respiratory Diseases": {
#             "file": "respiratory_diseases_data.csv",
#             "column": "Cases"
#         }
#     }

#     selected_health = st.selectbox("Select Health Disease Category", list(health_options.keys()))
#     file_path = f"{base_path}/{health_options[selected_health]['file']}"
#     target_column = health_options[selected_health]['column']

#     df = pd.read_csv(file_path)
#     df.columns = [c.strip().replace(" ", "_") for c in df.columns]
#     df_grouped = df.groupby("Year")[target_column].sum().reset_index()

#     X = df_grouped[["Year"]]
#     y = df_grouped[target_column]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     future_year = st.slider("Select Future Year for Prediction", 2025, 2035, 2025)
#     prediction = model.predict([[future_year]])[0]

#     st.markdown(f"### 📌 Predicted {selected_health} cases in {future_year}: {int(prediction):,}")

#     y_pred = model.predict(X_test)
#     r2 = r2_score(y_test, y_pred)
#     st.markdown(f"*Model R² Score*: {r2:.3f}")

#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.lineplot(data=df_grouped, x="Year", y=target_column, label="Historical", marker="o", ax=ax)
#     ax.scatter(future_year, prediction, color="red", label=f"Prediction ({future_year})", s=100)
#     ax.set_title(f"{selected_health} Trend & Future Prediction")
#     ax.set_xlabel("Year")
#     ax.set_ylabel("Number of Cases")
#     ax.legend()
#     st.pyplot(fig)

# # ------------------------ OVERVIEW ------------------------
# else:
#     df_diabetes = pd.read_csv(f"{base_path}/diabetes_data.csv")
#     df_hypertension = pd.read_csv(f"{base_path}/hypertension_data.csv")
#     df_heart = pd.read_csv(f"{base_path}/heart_disease_data.csv")
#     df_obesity = pd.read_csv(f"{base_path}/obesity_data.csv")
#     df_respiratory = pd.read_csv(f"{base_path}/respiratory_diseases_data.csv")

#     totals = {
#         "Diabetes": df_diabetes["Cases"].sum(),
#         "Hypertension": df_hypertension["Cases"].sum(),
#         "Heart Disease": df_heart["Cases"].sum(),
#         "Obesity": df_obesity["Cases"].sum(),
#         "Respiratory Diseases": df_respiratory["Cases"].sum()
#     }

#     st.subheader("📊 Overview of Health Diseases")
#     total_df = pd.DataFrame(list(totals.items()), columns=["Health Disease", "Total Cases"])
#     total_df.index = total_df.index + 1
#     st.dataframe(total_df)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(x="Total Cases", y="Health Disease", data=total_df, palette="Set2", ax=ax)
#     for index, value in enumerate(total_df["Total Cases"]):
#         ax.text(value + 1000, index, f"{int(value):,}", va='center', fontsize=9)
#     ax.set_title("Overall Health Disease Comparison")
#     st.pyplot(fig)

# # 🧠 Prevention Tips Section
# st.markdown("## 🧠 Health Tips & Prevention")
# if selected_disease:
#     with st.expander(f"🛡 How to Prevent: {selected_disease}"):
#         for tip in health_prevention_tips[selected_disease]:
#             st.markdown(f"- {tip}")
# st.markdown("### 🚨 Health Resources")
# st.markdown("- 📞 *Health Helpline: 104*")
# st.markdown("- 🏥 *Local Hospital Contact: Check local listings*")

# if selected_disease:
#     tips_text = f"Health Prevention Tips for {selected_disease}\n\n"
#     tips_text += "\n".join(f"- {tip}" for tip in health_prevention_tips[selected_disease])

#     st.download_button(
#         label="📥 Download Health Tips",
#         data=tips_text,
#         file_name=f"{selected_disease}_prevention_tips.txt",
#         mime="text/plain"
#     )

# # ------------------------ FOOTER ------------------------
# st.markdown("""
#     <div style='text-align: center; color: grey; font-size: 14px;'>
#         🩺 Health Data Analyzer Project | 2025<br>
#         Built using Python, Pandas, and Streamlit | Health Data Set
#     </div>
# """, unsafe_allow_html=True)









# # import streamlit as st
# # import pandas as pd
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # # === 1. Load the combined dataset ===
# # data_path = "C:\Users\megha\OneDrive\Desktop\health__dashboard _project\data\your_health_data.csv"  # Make sure the file exists in this path
# # df = pd.read_csv(data_path)


# # # Set up the page
# # st.set_page_config(page_title="Health Issues Dashboard", layout="wide")

# # # Title
# # st.title("📊 Health Issues Dashboard")
# # st.markdown("Analyze health issue trends by region, year, and disease.")

# # # === 1. Load the combined dataset ===
# # data_path = "data/your_dataset.csv"  # Change this if your file is named differently
# # df = pd.read_csv(data_path)

# # # Clean column names: remove leading/trailing whitespace
# # df.columns = df.columns.str.strip()

# # # Optional: Rename columns if your CSV uses different names
# # df.rename(columns={
# #     "Area": "Region",
# #     "Disease_Name": "Disease",
# #     "Number_of_Cases": "Cases"
# # }, inplace=True)

# # # Ensure required columns exist
# # required_columns = {"Year", "Region", "Disease", "Cases"}
# # if not required_columns.issubset(df.columns):
# #     st.error(f"Dataset must contain the following columns: {required_columns}")
# #     st.stop()

# # # Convert 'Year' to integer if needed
# # df["Year"] = df["Year"].astype(str).str.strip()
# # df["Year"] = pd.to_numeric(df["Year"], errors='coerce').astype("Int64")

# # # === 2. Sidebar filters ===
# # st.sidebar.header("🔍 Filter Data")
# # selected_year = st.sidebar.selectbox("Select Year", sorted(df["Year"].dropna().unique()))
# # selected_region = st.sidebar.selectbox("Select Region", sorted(df["Region"].dropna().unique()))
# # selected_disease = st.sidebar.selectbox("Select Disease", sorted(df["Disease"].dropna().unique()))

# # # === 3. Filtered Data ===
# # filtered_df = df[
# #     (df["Year"] == selected_year) &
# #     (df["Region"] == selected_region) &
# #     (df["Disease"] == selected_disease)
# # ]

# # # === 4. Display filtered results ===
# # st.subheader(f"📄 Data for {selected_disease} in {selected_region} ({selected_year})")
# # if filtered_df.empty:
# #     st.warning("No data available for the selected filters.")
# # else:
# #     st.dataframe(filtered_df, use_container_width=True)

# # # === 5. Visualization: Trend over years for selected disease in selected region ===
# # st.subheader(f"📈 {selected_disease} Trend Over Years in {selected_region}")

# # trend_data = df[
# #     (df["Region"] == selected_region) &
# #     (df["Disease"] == selected_disease)
# # ].groupby("Year")["Cases"].sum().reset_index()

# # fig, ax = plt.subplots(figsize=(10, 5))
# # sns.lineplot(data=trend_data, x="Year", y="Cases", marker="o", ax=ax)
# # ax.set_title(f"{selected_disease} Cases Trend in {selected_region}")
# # ax.set_xlabel("Year")
# # ax.set_ylabel("Cases")
# # st.pyplot(fig)
