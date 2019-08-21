# Forecasting_Energy_use
Predictive IoT Modeling - Forecasting Energy use

This IoT project aims to create predictive models for forecasting energy consumption of appliances. Data were collected using temperature and humidity sensors of a wireless network, the weather forecast of an airport station and the use of energy by light fixtures.
The drawing project is not customized and offers better removal conditions. The data set was collected for a period of 10 minutes for about 5 months. The temperature and humidity conditions of the house were monitored with a ZigBee wireless sensor network. Each node without frequency and humidity transmission around 3 min. Then an average of the data was calculated for the 10 minute phases.
Energy data were recorded every 10 minutes with m bus power meters. Time from the nearest weather station to the airport was transferred from a Trusted Prediction (rp5.ru) dataset and merged with experimental datasets using a data column and time. Oral programs were not included in the data set for regression model testing and non-predictive attribute filters (parameters).
The project was built with the power driver database. RandomForest was used for an attribute selection and SVM, Regression
Multilinear logistics for the predictive model.
R language was used.
