
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

## **Results**

1. **Quarterly Sales Visualization**:
    - The visualized sales data show a strong upward trend in revenue over the years, with clear seasonal patterns (e.g., high sales in Q4 due to holiday shopping).
    - Revenue dips significantly during the COVID-19 pandemic, but the company shows a sharp recovery afterward.

2. **Seasonal Decomposition**:
    - **Trend**: A long-term positive growth trend is observed, with a dip during the pandemic.
    - **Seasonality**: The data shows quarterly seasonal effects, with Q4 consistently being the strongest quarter.
    - **Residual**: The residuals are close to zero for most periods, indicating the model captures the major trends and seasonal effects.

3. **Autocorrelation and Partial Autocorrelation**:
    - The ACF plot shows strong autocorrelation at lag 1, confirming the influence of previous quarter revenues on the current quarter.
    - The PACF plot further supports the autoregressive relationship, showing significance at lag 1 and lag 4, indicating both short-term and seasonal dependencies.

4. **SARIMAX Model**:
    - The SARIMAX model achieved a good fit, capturing both the short-term AR(1) relationship and the seasonal AR and MA terms.
    - The model's summary indicates that the **AR(1)** term is highly significant, with a coefficient of **0.7176**. The **MA(1)** term, while not as strongly significant, is still useful in capturing short-term noise in the data.
    - The model has an **AIC of 1106.564** and **BIC of 1118.152**, suggesting a reasonable balance between model complexity and goodness-of-fit.

5. **Forecasting**:
    - The predicted revenue for the next eight quarters shows continued growth, with revenue expected to reach **6105 million USD** by Q4 2024.
    - Seasonal fluctuations are reflected in the forecast, showing expected increases in Q4 each year.
    - The model struggled slightly with the sudden drop caused by the pandemic, but overall, it performs well in capturing both long-term trends and seasonal effects.

## **Conclusion**

The SARIMAX model provides a robust fit for Adidas' quarterly sales revenue data, capturing both the long-term trend and seasonal fluctuations. The predictions suggest positive growth for Adidas over the next two years, with revenue expected to rise consistently despite seasonal dips. While the model struggles with sudden shocks (e.g., pandemic-related dips), it remains useful for strategic planning.
