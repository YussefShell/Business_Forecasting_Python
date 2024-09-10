
# **Adidas Quarterly Sales Forecasting**

This project aims to analyze Adidas' quarterly sales revenue data using time series analysis techniques. The goal is to build a predictive model that captures the trend, seasonality, and fluctuations in the revenue, and forecast future sales.

The project employs the **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors)** model to predict future revenue, and it includes key insights derived from visualizations and statistical analysis.

## **Project Overview**

1. **Data Visualization**: We first visualized Adidas' quarterly revenue to understand the patterns of growth and seasonality.
2. **Seasonal Decomposition**: We broke down the data into three components—trend, seasonality, and residuals—using multiplicative decomposition.
3. **Modeling**: We implemented a SARIMAX model, configured with both non-seasonal and seasonal parameters, to capture both short-term and long-term dependencies in the data.
4. **Forecasting**: The trained SARIMAX model was used to predict future sales revenue for eight quarters (two years).
5. **Evaluation**: Residuals were analyzed to check for randomness and ensure that the model accurately captures systematic patterns.

## **Installation and Requirements**

Before running the code, install the following dependencies:
```bash
pip install pandas
pip install matplotlib
pip install statsmodels
pip install plotly
```

## **Data**

The dataset consists of quarterly sales revenue for Adidas, which can be found in the file `adidas-quarterly-sales.csv`. The data includes two columns:
- `Time Period`: Quarters from 2000Q1 to 2021Q4.
- `Revenue`: Quarterly revenue in millions of dollars.

## **Code Explanation**

The analysis was carried out using the following steps in `Python`:

1. **Loading the Data**:
    ```python
    data = pd.read_csv("adidas-quarterly-sales.csv")
    print(data)
    ```
    The data is loaded using `pandas` and printed for a preliminary view.

2. **Visualization of Quarterly Sales**:
    ```python
    figure = px.line(data, x="Time Period", y="Revenue", title='Quarterly Sales Revenue of Adidas in Millions')
    figure.show()
    ```
    A line chart is generated using `Plotly` to visualize the time series of Adidas' quarterly revenue.

3. **Seasonal Decomposition**:
    ```python
    result = seasonal_decompose(data["Revenue"], model='multiplicative', period=30)
    fig = result.plot()
    fig.set_size_inches(15, 10)
    ```
    We used the `seasonal_decompose` function to break down the revenue into trend, seasonal, and residual components. The decomposition helps to understand the inherent patterns in the data.

4. **Autocorrelation and Partial Autocorrelation**:
    ```python
    pd.plotting.autocorrelation_plot(data["Revenue"])
    plot_pacf(data["Revenue"], lags=20)
    ```
    We analyzed the correlations between the data points to identify lag dependencies. The autocorrelation and partial autocorrelation plots helped in selecting the appropriate ARIMA parameters.

5. **SARIMAX Model**:
    ```python
    model = sm.tsa.statespace.SARIMAX(data['Revenue'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fitted = model.fit()
    print(model_fitted.summary())
    ```
    The **SARIMAX** model was defined with non-seasonal and seasonal parameters. The model was trained on the historical revenue data to capture both trend and seasonality.

6. **Forecasting**:
    ```python
    predictions = model_fitted.predict(start=len(data), end=len(data)+7)
    plt.figure(figsize=(15, 10))
    plt.plot(data["Revenue"], label="Training Data")
    plt.plot(predictions, label="Predictions", color='red')
    plt.legend()
    plt.show()
    ```
    The model was used to predict the revenue for the next eight quarters (Q1 2022 to Q4 2024). The predicted values are plotted against the actual training data for comparison.

## Results 

### 1. Visualization of Quarterly Revenue
The time series plot of Adidas' quarterly sales revenue clearly highlights the company's growth over time. Key observations:
- **Seasonality**: Clear seasonal patterns with periodic peaks and troughs.
- **Trend**: A strong upward trend in revenue, indicating overall business growth.
  
  **Key Observations**:
  - The revenue steadily increases over time, with a notable dip around 2020, likely reflecting the impact of the COVID-19 pandemic.
  - After the dip, the revenue rebounds significantly, showing that the company recovers relatively quickly from downturns.

### 2. Seasonal Decomposition
Using multiplicative decomposition, the revenue data was broken down into three key components: trend, seasonality, and residual.

- **Trend Component**:
  - Shows a long-term upward movement in revenue.
  - The sharp drop in revenue around 2020 is highlighted, followed by a strong recovery.
  
- **Seasonality Component**:
  - A repeating cycle where Q4 consistently shows higher revenue, likely due to holiday sales.
  - Q1 typically shows a decline.
  - The amplitude of seasonal effects remains consistent across the years.

- **Residual Component**:
  - Residuals represent random noise in the data after removing the trend and seasonal effects.
  - There are some spikes in the residuals during the 2020 dip, which the model does not fully capture.

### 3. Autocorrelation and Partial Autocorrelation Analysis

- **Autocorrelation (ACF) Plot**:
  - Significant autocorrelation at early lags, particularly at lag 1, meaning revenue from one quarter is highly correlated with the previous quarter.
  - The presence of seasonal spikes at lag 4 and lag 8 suggests quarterly seasonality.

- **Partial Autocorrelation (PACF) Plot**:
  - Significant values at lag 1 and lag 4 confirm the autoregressive relationship in the data.
  - The PACF findings suggest using an AR(1) model with seasonal components is appropriate.

### 4. SARIMAX Model Results

The SARIMAX model was applied to the data with the following parameters: ARIMA (1,1,1) and Seasonal ARIMA (1,1,1,12). The model incorporates both the trend and seasonal effects into the prediction.

- **Model Summary**:
  - **AR(1)**: Coefficient of 0.7176, statistically significant, indicating that past revenue has a strong influence on current revenue.
  - **MA(1)**: Negative coefficient (-0.9981) suggests a short-term corrective effect, though borderline significant (p-value ~0.067).
  - **Seasonal AR(1) and MA(1)**: Not statistically significant, but including them improves model performance.
  - **Sigma2**: Indicates some uncertainty in the predictions.

- **Model Performance**:
  - **AIC**: 1106.564, a lower value indicates a better fit.
  - **BIC**: 1118.152, indicating a balance between model complexity and fit.
  - **Log Likelihood**: -548.282, indicating a reasonable fit to the data.

### 5. Forecasting Future Sales

The model predicts the next 8 quarters (2 years) of revenue, revealing continued growth.

- **Predicted Growth**: Revenue is expected to rise, with consistent seasonal fluctuations.
- **Seasonal Variation**: As expected, there are dips and spikes corresponding to seasonal patterns.
- **Recovery**: Post-COVID revenue shows strong recovery, aligning with historical trends.

- **Key Insights**:
  - Revenue is forecasted to reach approximately **6,105 million USD by Q4 2024**, up from **5,514 million USD in Q4 2021**.
  - The forecast indicates a consistent rise in revenue, with the lowest predicted value at **5,421 million USD in Q2 2023**.

### 6. Residual Analysis

The residuals were analyzed to ensure the model captures systematic patterns in the data.

- **Key Findings**:
  - Residuals are randomly distributed, indicating the model successfully accounts for trend and seasonality.
  - Occasional spikes in the residuals during periods of sudden revenue changes, such as the COVID-19 pandemic dip, suggest the model struggles with unexpected events.

## **Conclusion**

The SARIMAX model provides a robust fit for Adidas' quarterly sales revenue data, capturing both the long-term trend and seasonal fluctuations. The predictions suggest positive growth for Adidas over the next two years, with revenue expected to rise consistently despite seasonal dips. While the model struggles with sudden shocks (e.g., pandemic-related dips), it remains useful for strategic planning.
