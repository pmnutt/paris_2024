import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Function to count medals by type
def count_medals(medal_type, medals):
    return medals[medals['medal_type'] ==
                  medal_type]['country_name'].value_counts()


# Load datasets
@st.cache_resource
def load_data():
    gdp_countries = pd.read_csv('Dataset/gdp_countries.csv')
    medals = pd.read_csv('Dataset/olympic_medals.csv')
    noc_regions = pd.read_csv('Dataset/noc_regions.csv')
    population_by_country = pd.read_csv(
        'Dataset/population_by_country_2020.csv')
    tokyo_2021 = pd.read_csv('Dataset/Tokyo 2021 dataset.csv')
    return gdp_countries, medals, noc_regions, population_by_country, tokyo_2021


gdp_countries, medals, noc_regions, population_by_country, tokyo_2021 = load_data(
)

# Data Cleaning and Preprocessing
gdp_countries.drop_duplicates(inplace=True)
medals.drop_duplicates(inplace=True)
noc_regions.drop_duplicates(inplace=True)
population_by_country.drop_duplicates(inplace=True)
tokyo_2021.drop_duplicates(inplace=True)

# Create a DataFrame with counts for each medal type
hist_sum_score = pd.DataFrame({
    'gold': count_medals('GOLD', medals),
    'silver': count_medals('SILVER', medals),
    'bronze': count_medals('BRONZE', medals)
})
st.header("Olympic Insigth and Prediction for Paris 2024")
# Calculate total medals
hist_sum_score['total'] = hist_sum_score['gold'] + hist_sum_score[
    'silver'] + hist_sum_score['bronze']
hist_sum_score.dropna(inplace=True)
st.write("Total medals in Historical") 
hist_sum_score.sort_values(by='total', ascending=False)


# Reset index and rename columns
hist_sum_score = hist_sum_score.reset_index().rename(columns={
    'index': 'country_name'
}).sort_values(by='total', ascending=False)

st.bar_chart(hist_sum_score['total'], height=400,)
st.write("Total medals in Historical")
st.markdown("The main idea is to predict the total medals in Paris 2024, based on the historical data. The data is cleaned and preprocessed to ensure that the model can accurately predict the total medals. I use the GDP and Population data to predict the total medals. The model is trained on historical data and then tested on the latest data")
st.write(hist_sum_score)
# Merging with other datasets
hist_sum_score = hist_sum_score.merge(gdp_countries,on='country_name',how='right')
hist_sum_score = hist_sum_score.merge(tokyo_2021,on='country_name',how='right')
hist_sum_score = hist_sum_score.merge(population_by_country,on='country_name',how='right')
hist_sum_score = hist_sum_score.merge(medals,on='country_name',how='right')

# Calculate new features
number_sport = hist_sum_score['discipline_title'].nunique()
hist_sum_score[
    'GDP_per_capita'] = hist_sum_score['gdp'] / hist_sum_score['population']
hist_sum_score[
    'average_medals_per_sport'] = hist_sum_score['total'] / number_sport

hist_sum_score.drop_duplicates(inplace=True)
hist_sum_score.dropna(inplace=True)

# Display the dataset

st.markdown("---")
st.write("This dataset contains the number of medals won by each country in the Olympics since 1984. The dataset also contains information about the country's GDP per capita,")
col1, col2 = st.columns(2)
with col1:
    st.header("GDP per capita")
    st.scatter_chart(gdp_countries,x = 'gdp',y = 'country_name',height=400)
with col2:
    st.header("Population")
    st.bar_chart(population_by_country,y = 'population',x = 'country_name',height=400)
st.write("## Merged Data")
st.write(hist_sum_score)

# Define features and target
features = ['gdp', 'average_medals_per_sport', 'population', 'GDP_per_capita']
target = 'Total'

X = hist_sum_score[features]
y = hist_sum_score[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# If there are issues with shapes, stop execution
if X.shape[0] == 0 or y.shape[0] == 0:
    st.error(
        "The feature matrix X or target vector y is empty. Check your data processing steps."
    )
else:
    param_grid = {
        'n_estimators': [int(x) for x in range(100, 1001, 100)],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [int(x) for x in range(10, 101, 10)],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Random Forest
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_grid,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)
    rf_random.fit(X_train, y_train)
    best_params = rf_random.best_params_
    st.write(f"Best parameters: {best_params}")

    best_rf = RandomForestRegressor(**best_params)
    best_rf.fit(X_train, y_train)
    y_pred_best_rf = best_rf.predict(X_test)

    # Evaluation Metrics
    def evaluate_model(y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        st.write(f"{model_name} Performance:")
        st.write(f"MAE: {mae:.3f}")
        st.write(f"MSE: {mse:.3f}")
        st.write(f"RMSE: {rmse:.3f}")
        st.write(f"R2 Score: {r2:.3f}")
        st.write("\n")

    evaluate_model(y_test, y_pred_best_rf, 'Tuned Random Forest')

    # Assuming you have the Paris 2024 dataset prepared similarly
    paris_2024_data = hist_sum_score.copy(
    )  # Adjust this line if the dataset has a different name

    # Predict using the best model
    paris_2024_predictions = best_rf.predict(paris_2024_data[features])

    # Add predictions to the dataset
    paris_2024_data['predicted_medals'] = paris_2024_predictions
    st.write("## Paris 2024 Predictions")
    st.write("In conclusion, the best model based on historical data is Tuned Random Forest. The predictions for Paris 2024")
    st.write(f"RMSE: {rmse:.3f} & R2 Score: {r2:.3f}, That shown the data is not accurate from the historical data")
    st.write(paris_2024_data.head())

    # Visualization of actual vs predicted values
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred_best_rf, alpha=0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
            '--r',
            linewidth=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted Medal Counts')
    st.pyplot(fig)

    # Interactive plot with Plotly
    fig = px.bar(
        paris_2024_data.sort_values(by='predicted_medals', ascending=False),
        x='country_name',
        y='predicted_medals',
        title='Predicted Medal Counts for Paris 2024 Olympics',
        labels={
            'predicted_medals': 'Predicted Medals',
            'country_name': 'Country'
        },
    )
    fig.update_layout(xaxis_title='Country',
                      yaxis_title='Predicted Medals',
                      xaxis=dict(tickangle=-90))
    st.plotly_chart(fig)
