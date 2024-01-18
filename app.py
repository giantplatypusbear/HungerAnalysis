import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from dash import dash_table
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
import numpy as np


#creating dash
app = dash.Dash(__name__)
app.title = "World Hunger Dashboard"

#importing data
hungerindex = pd.read_csv("C:/Users/Rithi/OneDrive/Documents/WorldHunger/data/prevalence-of-undernourishment.csv")

country_data = []
for country, data in hungerindex.groupby('Entity'):
    if len(data) > 1:  # Only consider countries with data for more than one year
        country_data.append(data)

# Calculate the correlation for each country
correlation_data = []
for data in country_data:
    correlation = data['Year'].corr(data['Prevalence of undernourishment (% of population)'])
    correlation_data.append({'Country': data['Entity'].iloc[0], 'Correlation': correlation})

correlation_df = pd.DataFrame(correlation_data)
positive_correlation_df = correlation_df[correlation_df["Correlation"] > 0]
negative_correlation_df = correlation_df[correlation_df["Correlation"] < 0]

data_2001_to_2019 = hungerindex[hungerindex["Year"].between(2001, 2019)]
unique_countries = hungerindex["Entity"].unique()

# Create an empty DataFrame for binary columns
country_columns = pd.DataFrame()

# Iterate over unique countries
for country in unique_countries:
    country_columns[country] = (data_2001_to_2019["Entity"] == country).astype(int)

# Concatenate the binary columns to the main DataFrame
data_2001_to_2019 = pd.concat([data_2001_to_2019, country_columns], axis=1)

years_2020_to_2025 = pd.DataFrame({'Year': list(range(2020, 2026))})


for country in unique_countries:
    data_2001_to_2019[country] = (data_2001_to_2019["Entity"] == country).astype(int)

merged_data = pd.merge(data_2001_to_2019, hungerindex, on=["Year", "Entity", "Prevalence of undernourishment (% of population)"], how="inner")


# Linear Regression model
models = {}  #models for each country


for entity in merged_data['Entity'].unique():
    
    model = LinearRegression()

    country_data = data_2001_to_2019[data_2001_to_2019["Entity"] == entity]

    X_country = country_data[['Year'] + list(unique_countries)]
    y_country = country_data["Prevalence of undernourishment (% of population)"]

    model.fit(X_country, y_country)

    models[entity] = model

# Initialize an empty DataFrame for storing predictions
data_2020_to_2025 = pd.DataFrame(columns=["Entity", "Year", "Predicted"])


if not data_2020_to_2025.empty:
    # Iterate over each unique entity
    for entity in merged_data['Entity'].unique():
        
        years = list(range(2020, 2026))

        # Predict values for the entity for the years 2020 to 2025 using the corresponding model
        model = models[entity]
        X_predict = pd.DataFrame({'Year': years, **{c: (c == entity) for c in unique_countries}})
        predictions = model.predict(X_predict)

        # Create a DataFrame for the entity's predictions
        entity_data = pd.DataFrame({
            "Entity": [entity] * len(years),
            "Year": years,
            "Predicted": predictions
        })

        # Append the entity's data to the overall data_2020_to_2025 DataFrame
        data_2020_to_2025 = pd.concat([data_2020_to_2025, entity_data], ignore_index=True)




app.layout = html.Div([
    html.H1("World Hunger Analysis Dashboard", style={
        'background-color': '#4BC7DF',
        'color': 'white',
        'padding': '10px',
        'font-family': 'Tahoma, Geneva, sans-serif',
    }),
    html.Div([
        html.H2("Information", style={'font-family': 'Tahoma, Geneva, sans-serif'}),
        dcc.Markdown("""
Prevalence of undernourishments is the percentage of the population whose habitual food consumption is insufficient to provide the dietary energy levels that are required to maintain a normal active and healthy life. Data showing as 2.5 may signify a prevalence of undernourishment below 2.5%.

Limitations and exceptions: From a policy and program standpoint, this measure has its limits. First, food insecurity exists even where food availability is not a problem because of inadequate access of poor households to food. Second, food insecurity is an individual or household phenomenon, and the average food available to each person, even corrected for possible effects of low income, is not a good predictor of food insecurity among the population. And third, nutrition security is determined not only by food security but also by the quality of care of mothers and children and the quality of the household's health environment (Smith and Haddad 2000).

Statistical concept and methodology: Data on undernourishment are from the Food and Agriculture Organization (FAO) of the United Nations and measure food deprivation based on average food available for human consumption per person, the level of inequality in access to food, and the minimum calories required for an average person.

        """)
    ], style={'display': 'inline-block', 'width': '30%', 'vertical-align': 'top', 'margin-top': '20px'}),
    html.Div([
        dcc.Graph(id="heatmap"),
        dcc.Slider(id="year-slider", min=hungerindex["Year"].min(), max=hungerindex["Year"].max(), value=hungerindex["Year"].min(), marks={str(year): str(year) for year in hungerindex["Year"].unique()}),
        dcc.Graph(id="trends"),
        dcc.Dropdown(id="country-dropdown", options=[{"label": country, "value": country} for country in hungerindex["Entity"].unique()]),
        dcc.RadioItems(id="date-toggle", options=[
            {'label': '2001 to 2019', 'value': '2001_to_2019'},
            {'label': '2020 to 2025 Predicted', 'value': '2020_to_2025_predicted'}
        ], value='2001_to_2019'),
    ], style={'display': 'inline-block', 'width': '69%', 'vertical-align': 'top'}),  
      

html.Div([
        html.Div([
            html.H2("Positive Correlations"),
            dash_table.DataTable(
                id='positive-correlation-table',
                columns=[{'name': col, 'id': col} for col in positive_correlation_df.columns],
                data=positive_correlation_df.to_dict('records'),
                page_size=10 
            )
        ], style={'width': '49%', 'display': 'inline-block','margin-right': '1%'}),

       
        html.Div([
            html.H2("Negative Correlations"),
            dash_table.DataTable(
                id='negative-correlation-table',
                columns=[{'name': col, 'id': col} for col in negative_correlation_df.columns],
                data=negative_correlation_df.to_dict('records'),
                page_size=10  
            )
        ], style={'width': '49%', 'display': 'inline-block','margin-left': '1%'}),
    ])
])

@app.callback(
    Output("heatmap", "figure"),
    Input("year-slider", "value")
)
def update_heatmap(selected_year):
    
    filtered_data = hungerindex[hungerindex["Year"] == selected_year]

   #choromap
    heatmap_fig = px.choropleth(
        filtered_data,
        locations="Entity",  # Country names
        locationmode="country names",
        color="Prevalence of undernourishment (% of population)",  
        hover_name="Entity",
        title=f"Prevalence of undernourishment (% of population) in {selected_year}"
    )

    return heatmap_fig


@app.callback(
    Output("trends", "figure"),
    [Input("country-dropdown", "value"), Input("date-toggle", "value")]
)
def update_charts(selected_country, selected_date):
    title = ""
    y_column = "Prevalence of undernourishment (% of population)"
    
    if selected_date == "2001_to_2019":
        filtered_data = data_2001_to_2019[data_2001_to_2019["Entity"] == selected_country]
        title = "Trends in Hunger for {}".format(selected_country)
    elif selected_date == "2020_to_2025_predicted":
        filtered_data = data_2020_to_2025[data_2020_to_2025["Entity"] == selected_country]
        title = "Trends in Hunger for {} (2020 to 2025 Predicted)".format(selected_country)
        y_column = "Predicted"

    if filtered_data.empty:
        title = "No data available for the selected country and date range"

    # Create a line chart for trends analysis
    trends_fig = px.line(
        filtered_data,
        x="Year",  # X-axis: Year
        y=y_column,  # Y-axis
        title=title,
    )

    return trends_fig




if __name__ == "__main__":
    app.run_server(debug=True)
