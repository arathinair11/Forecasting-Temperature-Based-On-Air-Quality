import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.holtwinters as ets
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy.linalg as la
from scipy import signal
from tqdm import tqdm
from scipy.stats import chi2
from toolbox import statstoolsACF_PACF, cal_rolling_mean_var, KPSS_Test, ADF_Cal, diff_seasonal_calc, \
    createGPAC,zero_pole_print, Q_pred, create_samples,calc_Q_Score,plot_prediction_method_axis,average_train,\
    naive_train,ses_train,drift_train,average_test,naive_test,drift_test,ses_test,\
    inverse_diff

'''
Load the dataset
'''
print(20*'= ' + 'Load Dataset' + 20*' =')

df = pd.read_excel("AirQualityUCI.xlsx", engine="openpyxl")
print(df.head())


'''
Pre-processing dataset
'''
print(20*'= ' + 'Pre-processing' + 20*' =')

df.replace(to_replace = -200, value =np.nan,inplace=True)
df.fillna(df.mean(numeric_only = True),inplace=True)
df.drop(['Unnamed: 15',"Unnamed: 16"], axis=1, inplace=True)
print(df.describe())
print("Null Values:",df.isnull().sum().sum())

'''
Plot of the dependent variable versus time.
'''
print(20*'= ' + 'Plot Dependent Variable' + 20*' =')

temp = df["T"].astype(float)
df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),format='%Y-%m-%d %H:%M:%S',errors='coerce')
date = df["DateTime"]

plt.figure()
plt.plot(date,temp, label = 'Temperature')
plt.xlabel('Month')
plt.ylabel('Temperature in Celsius')
plt.title('Temperature increase over time')
plt.legend()
plt.show()

'''
ACF/PACF of the dependent variable.
'''
print(20*'= ' + 'ACF/PACF: Dependent Variable' + 20*' =')

def acf(series):
    n = len(series)
    data = np.asarray(series)
    mean = np.mean(data)
    var = np.sum((data - mean) ** 2)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / var
        return round(acf_lag, 3)

    x = np.arange(n)
    acf_coef = []
    for i in x:
        temp = r(i)
        acf_coef.append(temp)
    return acf_coef

def plot_autocorr(data, lags):
    r = data[:lags]
    r1 = np.array(r[::-1])
    R = np.concatenate((r1, r[1:]), axis=0)
    x = np.arange((-1*int(len(r))+1), int(len(r1)))

    plt.figure(figsize=(20,5))
    plt.stem(x, R)
    (markers, stemlines, baseline) = plt.stem(x, R)
    plt.setp(markers, markerfacecolor='r')
    n = len(data)
    m = 1.96 / np.sqrt(n)

    plt.axhspan(-m, m, alpha=0.2, color='blue')
    plt.xlabel('Lags', fontsize=20)
    plt.ylabel('Magnitude', fontsize=20)

plotdata = acf(temp)
plot_autocorr(plotdata, lags = 100)
plt.title('Autocorrelation function of Temperature', fontsize = 25)
plt.show()

statstoolsACF_PACF(temp,100,'Temperature')

'''
Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient
'''
print(20*'= ' + 'Correlation Matrix' + 20*' =')

corr = df.corr()
fig,ax = plt.subplots(figsize=(12, 10))
ax = sns.heatmap(corr, annot=True,cmap=plt.cm.Reds)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize = 9)
plt.title('Correlation Matrix', fontsize = 20)
plt.show()

'''
Split the dataset into train set (80%) and test set (20%)
'''
df["Temperature"] = df["T"]
X = df.loc[:, ~df.columns.isin(["Date","Time","DateTime","T","Temperature"])]
Y = df["T"]
X = sm.add_constant(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle=False, test_size=0.2)


'''
Stationarity: Check for a need to make the dependent variable stationary.If the dependent variable is not stationary
Perform ACF/PACF analysis for stationarity. Perform ADF-test & kpss-test and plot the rolling mean and variance for the raw data and the transformed data.
'''
print(20*'= ' + 'Stationarity' + 20*' =')

cal_rolling_mean_var(temp)

ADF_Cal(temp)

KPSS_Test(temp)

df['Order_Diff_1'] = temp.diff()
Order_Diff_1 = df['Order_Diff_1']
cal_rolling_mean_var(Order_Diff_1)
ADF_Cal(Order_Diff_1.dropna())
KPSS_Test(Order_Diff_1.dropna())

plotdata = acf(Order_Diff_1.dropna())
statstoolsACF_PACF(plotdata,50,'Temp_Order_Diff_1')

# Seasonally adjusting the data (24 hours), Tested(6,12,168 hours)
temp_seasdiff = diff_seasonal_calc(temp, 24)
statstoolsACF_PACF(temp_seasdiff, lags=40, title_str=f'Temperature: {24} hours')
cal_rolling_mean_var(temp_seasdiff)
ADF_Cal(df['Temperature'].dropna())
KPSS_Test(df['Temperature'].dropna())

'''
Time series Decomposition: Approximate the trend and the seasonality and plot the detrended and the seasonally adjusted data set.
Find the out the strength of the trend and seasonality.
'''
print(20*'= ' + 'Time series Decomposition' + 20*' =')

stl_df = df[["Temperature","DateTime"]].copy()
stl_df.replace(to_replace = "NaT", value =np.nan,inplace=True)
stl_df.dropna(inplace = True)
stl_df.set_index("DateTime", inplace = True)
stl = STL(stl_df).fit()
plt.legend(loc="best")
plt.xlabel("Year")
plt.ylabel("Temperature")
stl.plot()
plt.show()

t = stl.trend
s = stl.seasonal
r = stl.resid


strength_trend = np.max([0, 1-(r.var()/(t+r).var())])
print(f'\n The strength of trend for this data set is {strength_trend:0.3f}')

strength_seasonal = np.max([0, 1-(r.var()/(s+r).var())])
print(f' The strength of seasonality for this data set is {strength_seasonal:0.3f}')

adjusted_seasonal = stl_df["Temperature"] - s #Adjust for seasonality

adjusted_trend = stl_df["Temperature"] - t #Adjust for trend

plt.figure()
stl_df["Temperature"].plot(label='Original set')
adjusted_seasonal.plot(label='Seasonally Adjusted Data')
plt.title(f'Seasonally Adjusted and Original Data\n Strength of Seasonality:{strength_seasonal:0.3f}', fontsize=14,)
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.legend(loc='best')
plt.show()

plt.figure()
stl_df["Temperature"].plot(label='Original set')
adjusted_trend.plot(label='De-Trended Data')
plt.title(f'De-Trended and Original Data\n Strength of Trend:{strength_trend:0.3f}', fontsize=14,)
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.legend(loc='best')
plt.show()

'''
Holt-Winters method: Using the Holt-Winters method try to find the best fit using the train dataset and make a prediction using the test set.
'''
print(20*'= ' + 'Holt-Winter Linear Forecasting Method' + 20*' =')

holtt = ets.ExponentialSmoothing(Y_train, trend=None, damped_trend=False, seasonal='mul',seasonal_periods=24).fit()

pred_y = holtt.forecast(steps=len(Y_test.values))
pred_y = pd.DataFrame(pred_y).set_index(Y_test.index)

# Calculate error
error = Y_test + 1 - pred_y[0]
error_2 = error ** 2

plt.figure()
plt.plot(Y_train.index,Y_train, label = 'Train Data')
plt.plot(Y_test.index,Y_test, label='Test Data')
plt.plot(pred_y, label='Holt-Winter Linear Forecast', alpha = 0.9)
plt.title(f"Plot for Holt-Winter Linear Forecast Method \n MSE: {error_2.mean():0.2f}")
plt.legend(loc ="best")
plt.show()


predictions = holtt.predict(start=1, end=Y_train.shape[0])
residuals = Y_train.values - predictions
print(f'Holt- Winter Q: {calc_Q_Score(residuals.values, Y_train.values, lags=24, print_out=False):0.2f}')
print(f'Holt- Winter RSME:{np.sqrt(error_2.mean()):0.2f}')

'''
Feature selection: Explains how the feature selection was performed and whether the collinearity exits not.
Backward stepwise regression along with SVD and condition number is needed.
'''
print(20*'= ' + 'Feature selection' + 20*' =')

df_svd_X = X_train.copy()
df_svd_y = Y_train.copy()
s,d,v = la.svd(df_svd_X)

print(f"\nSingularValues:\n{d}\n")
print(f"Condition Number\n{la.cond(df_svd_X)}\n")

model = sm.OLS(df_svd_y,df_svd_X).fit()
#print(model.summary())
#R-squared:  0.934
#Adj. R-squared: 0.934
#AIC:3.215e+04
#BIC:3.224e+04

df_svd_X.drop(["NMHC(GT)"], axis = 1, inplace = True)
model = sm.OLS(df_svd_y,df_svd_X).fit()
#print(model.summary())
#R-squared:  0.934
#Adj. R-squared: 0.934
#AIC:3.215e+04
#BIC:3.223e+04

df_svd_X.drop(["PT08.S1(CO)"], axis = 1, inplace = True)
model = sm.OLS(df_svd_y,df_svd_X).fit()
#print(model.summary())
#R-squared:  0.934
#Adj. R-squared: 0.934
#AIC:3.215e+04
#BIC:3.222e+04

df_svd_X.drop(["PT08.S3(NOx)"], axis = 1, inplace = True)
model = sm.OLS(df_svd_y,df_svd_X).fit()
#print(model.summary())
#R-squared:  0.934
#Adj. R-squared: 0.934
#AIC:3.215e+04
#BIC:3.221e+04

df_svd_X.drop(["NOx(GT)"], axis = 1, inplace = True)
model = sm.OLS(df_svd_y,df_svd_X).fit()
#print(model.summary())
#R-squared:  0.934
#Adj. R-squared: 0.934
#AIC:3.215e+04
#BIC:3.221e+04

df_svd_X.drop(["const"], axis = 1, inplace = True)
model = sm.OLS(df_svd_y,df_svd_X).fit()
#print(model.summary())

df_svd_X.drop(["PT08.S2(NMHC)"], axis = 1, inplace = True)
model = sm.OLS(df_svd_y,df_svd_X).fit()
#print(model.summary())

df_svd_X.drop(["PT08.S4(NO2)"], axis = 1, inplace = True)
model = sm.OLS(df_svd_y,df_svd_X).fit()
#print(model.summary())

df_svd_X.drop(["CO(GT)"], axis = 1, inplace = True)
model = sm.OLS(df_svd_y,df_svd_X).fit()
#print(model.summary())

df_svd_X.drop(["PT08.S5(O3)"], axis = 1, inplace = True)
model = sm.OLS(df_svd_y,df_svd_X).fit()
print(model.summary())


s,d,v = la.svd(df_svd_X)
print(f"\nSingularValues:\n{d}\n")
print(f"Condition Number\n{la.cond(df_svd_X)}\n")

df_model = df[["DateTime","Temperature","C6H6(GT)","NO2(GT)","RH","AH"]].copy()


'''
Base-models: average, naïve, drift, simple and exponential smoothing.
You need to perform an h-step prediction based on the base models and compare the SARIMA model performance with the base model predication.
'''
print(20*'= ' + 'Base-models' + 20*' =')

method_dict = {average_test: 'Average', naive_test: 'Naive', drift_test: 'Drift', ses_test: 'SES'}

for key in method_dict:
    avg = key(Y_train, Y_test)
    average_pred = avg[0]
    error = avg[1]
    error_2 = avg[2]

    plot_prediction_method_axis(Y_train, Y_test, average_pred, error_2, method_str=method_dict[key] + ' Method')

print(f'Average Q: {calc_Q_Score(average_train(Y_train.values)[1], Y_train.values, lags=24, print_out=False):0.2f}')
print(f'Naive Q: {calc_Q_Score(naive_train(Y_train.values)[1], Y_train.values, lags=24, print_out=False):0.2f}')
print(f'Drift Q: {calc_Q_Score(drift_train(Y_train.values)[1], Y_train.values, lags=24, print_out=False):0.2f}')
print(f'SES Q: {calc_Q_Score(ses_train(Y_train.values)[1], Y_train.values, lags=24, print_out=False):0.2f}')


'''
Develop the multiple linear regression model that represent the dataset. Check the accuracy of the developed model.
1.	Perform one-step ahead prediction and compare the performance versus the test set.
2.	Hypothesis tests analysis: F-test, t-test.
3.	AIC, BIC, RMSE, R-squared and Adjusted R-squared
4.	ACF of residuals.
5.	Q-value
6.	Variance and mean of the residuals.

'''
print(20*'= ' + 'Multiple Linear Regression Model' + 20*' =')

X = df_model.loc[:, ~df_model.columns.isin(["DateTime","T","Temperature"])]
Y = df_model["Temperature"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle=False, test_size=0.2)

model = sm.OLS(Y_train,X_train).fit()
pred_y = model.predict(X_test)
forecast_errors = Y_test-pred_y
error_2 = forecast_errors**2

print(model.summary())

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

pred_diff = len(Y_test) - len(pred_y)

ax.plot( Y_train, linewidth=1, label="Training Set")
ax.plot(Y_test, linewidth=1, label="Testing Set")
ax.plot(pred_y, linewidth=1, label="Forecast", alpha=0.8)
ax.set_xlabel('Time (Hourly)')
ax.set_ylabel('Temperature')
ax.set_title(f'Temperature Predictions using OLS Multiple Linear Regression \n MSE: {error_2.mean()}', fontsize=14)
ax.legend()
plt.show()

print(f'Temperature Predictions using OLS Multiple Linear Regression \n RMSE: {np.sqrt(error_2.mean())}')

model = sm.OLS(Y_train,X_train).fit()
pred_y = model.predict(X_train)
pred_errors = Y_train-pred_y
res_error_2 = pred_errors**2

plot_autocorr(acf(pred_errors.values), lags = 60)
plt.title('Autocorrelation function of residuals', fontsize = 12)
plt.show()

RSME = math.sqrt(np.mean(pred_errors.values ** 2))

print(f"RSME : {RSME:0.3f}")

Q_pred(Y_train,pred_errors,10,4)

var_for =  int(np.var(forecast_errors))
mean_for =  int(np.mean(forecast_errors))
print("The variance of the residuals are: ", var_for)
print("The mean of the residuals are: ", mean_for)

'''
ARMA and ARIMA and SARIMA model order determination: Develop an ARMA, ARIMA and SARIMA model that represent the dataset.
a. Preliminary model development procedures and results. (ARMA model order determination). Pick at least two orders using GPAC table.
b. Should include discussion of the autocorrelation function and the GPAC. Include a plot of the autocorrelation function and the GPAC table within this section).
c. Include the GPAC table in your report and highlight the estimated order.
'''
print(20*'= ' + 'Order Determination' + 20*' =')

df_model = df_model.set_index('DateTime')
X = df_model.loc[:, ~df_model.columns.isin(["DateTime""Temperature"])]
Y = diff_seasonal_calc(df_model["Temperature"],24)
X = X.iloc[24:, ]
Y.index = X.index
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle=False, test_size=0.2)

statstoolsACF_PACF(Y_train,lags = 50, title_str='')

createGPAC(Y_train.values,'',7,7)

'''
ARMA(1,0) model
'''
print(20*'= ' + 'ARMA(1,0)' + 20*' =')

na=1
nb=0

model = statsmodels.tsa.arima.model.ARIMA(Y_train, order=(na,0,nb), freq='H').fit()
na_params=model.params[0:na]*-1
nb_params=model.params[na::]

# 1-step prediction
model_pred = model.predict(start=1, end=Y_train.shape[0])
residuals= model.resid
print(model.summary())


# Plots the Training set
plt.figure(figsize=(8,6))
plt.plot(Y_train, label='Training Set')
plt.plot(model_pred, label='Predicted Values', alpha=0.9)
plt.title('ARMA(1,0) Model\n Prediction of Training Set')
plt.xlabel('Time (Hourly)')
plt.ylabel('Temperature')
plt.legend()
plt.show()

plot_autocorr(acf(residuals.values), lags= 100)
statstoolsACF_PACF(residuals, lags=100, title_str='ARMA(1,0) Residuals\n')

con_intervals=model.conf_int().drop(['const','sigma2'])

if na!=0:
    na_vals=model.arparams*-1
else:
    na_vals=0
na_con=con_intervals[0:na].values*-1

if nb!=0:
    nb_vals=model.maparams
else:
    nb_vals=0
nb_con = con_intervals[na::]

print(f'\n============\nConfidence Interval Results:\n============\n')
print('Na coeffs\n----------')
if len(na_con)==0:
    print('None')
else:
    [print(f'{na_vals[i]:0.3f}: {na_con[i][0]:0.3f} to {na_con[i][1]:0.3f}') for i in range(len(na_con))]

print('\nNb coeffs\n----------')
if len(nb_con)==0:
    print('None')
else:
    [print(f'{nb_vals[i]:0.3f}: {nb_con[i][0]:0.3f} to {nb_con[i][1]:0.3f}') for i in range(len(nb_con))]

# Generate the standard deviation

summary = model.summary().as_text()

# Extract the number of items used to make the STE
observations = int(summary[summary.find('No. Observations:') + len('No. Observations:'):summary.find('Model:')].strip())

# Extract the STE for each variable
summary_params = summary[summary.find('const'):summary.find('sigma2')].split('\n')
summary_params = summary_params[1:-1]  # Remove the constant and sigma2 rows

# Extract the STE
collector = []
for param in range(len(summary_params)):
    collector.append([i for i in summary_params[param].split(' ') if i != ''])

collector = pd.DataFrame(collector)
collector = collector.filter([0, 1, 2])

# Convert STE to STD
collector[2] = collector[2].astype(float) * np.sqrt(observations)

# Print the Standard Deviations
na_std = collector.iloc[0:na, :]
nb_std = collector.iloc[na::, :]

print(f'\n============\nStandard Deviation Results:\n============\n')
print('Na coeffs\n----------')
if na_std.shape[0] == 0:
    print('None')
else:
    [print(f'{float(na_std.iloc[i, 1]) * -1:0.3f} STD: {na_std.iloc[i, 2]:0.3f}') for i in range(na_std.shape[0])]

print('\nNb coeffs\n----------')
if nb_std.shape[0] == 0:
    print('None')
else:
    [print(f'{float(nb_std.iloc[i, 1]) * -1:0.3f} STD: {nb_std.iloc[i, 2]:0.3f}') for i in range(nb_std.shape[0])]

# Chi-squared Test

N = len(Y_train)
DOF = 24 - na - nb
alpha = 0.01
QCrit = chi2.ppf(1 - alpha, DOF)
Q, p = sm.stats.acorr_ljungbox(residuals, lags=[24])
if Q < QCrit:
    print('The Residual errors can be considered as random')
else:
    print('The Residual errors are NOT RANDOM for this model\n')
print('The Critical value: ', QCrit)
print('The Q value: ', Q)
print('The P value: ', p)
print(f'\nDegrees of Freedom: {DOF}')

# Display poles and zeros (roots of numerator and roots of denominator)
print('\n Zero-Pole Cancellation\n')
params = model.params[1:-1]
poly_y = params[0:na].values
poly_e = params[na::].values

try:
    zeros = np.poly(poly_e)[1::]
    zero_pole_print(zeros)
except:
    print('(1-0)')

print('----------------')
try:
    poles = np.poly(poly_y)[1::]
    zero_pole_print(poles)
except:
    print('(1-0)')

# Display the estimated variance of error.
params = model.params[1:-1]
na_params = np.array([1] + list(params[0:na].values))
nb_params = np.array([1] + list(params[na::].values))

if na == 0:
    na_params = np.zeros(nb_params.shape)
    na_params[0] = 1

if na_params.shape[0] < nb_params.shape[0]:
    na_params = np.pad(na_params, (0, nb_params.shape[0] - na_params.shape[0]), 'constant')

if nb == 0:
    nb_params = np.zeros(na_params.shape)
    nb_params[0] = 1

if nb_params.shape[0] < na_params.shape[0]:
    nb_params = np.pad(nb_params, (0, na_params.shape[0] - nb_params.shape[0]), 'constant')

# Construct the system in reverse to get the error instead of the y
sys = (na_params, nb_params, 1)

# Generate a white noise set using these params
wn = create_samples(residuals.shape[0], wn_mean=0, wn_var=1)

# Process the system
_, e_dlsim = signal.dlsim(sys, wn)

print(f' The Estimated Variance of the Error is {np.var(e_dlsim):0.3f}')

# Check for Bias
print(f'The Mean of the Residuals is {np.mean(residuals):0.2f}')

# Covarience heatmap of all features
cov = model.cov_params()
cov = cov.drop(['const', 'sigma2'], axis=0)
cov = cov.drop(['const', 'sigma2'], axis=1)

fig, ax = plt.subplots(figsize=[6, 6])
sns.heatmap(cov, center=0, cmap='vlag', annot=True, fmt='0.6f', ax=ax)
plt.title("Covariance Matrix of the Estimated Parameters\n")
plt.show()


# One step ahead Prediction
model_loop = statsmodels.tsa.arima.model.ARIMA(Y_train, order=(na, 0, nb), freq='H').fit()

predictions = []
for i in tqdm(range(len(Y_test))):
    predictions.append(model_loop.forecast(steps=1))
    model_loop = model_loop.append(np.array(Y_test[i:i + 1]))

predictions_series=pd.concat(predictions)
predictions_series.to_csv('One-step-ahead-Prediction-ARMA(1,0).csv')

# Plot the one step ahead - Testing Set Only
plt.figure(figsize=(8,6))
plt.plot(Y_test, label='True Values')
plt.plot(predictions_series, label='Forecast Values', alpha=0.9)
plt.title('ARMA(1,0) Predicted Parameters Model\n One Step Ahead Testing Set')
plt.xlabel('Time (Hourly)')
plt.ylabel('Traffic Volume')
plt.legend()
plt.show()


# Plot the one step ahead - Full Data Set
plt.figure(figsize=(8,6))
plt.plot(Y_train, label='Training Set')
plt.plot(Y_test, label='Testing Set')
plt.plot(predictions_series, label='Forecast', alpha=0.9)
plt.title(f'ARMA(1,0) Predicted Parameters Model\n One Step Ahead Forecasting\nMSE: {model.mse:0.2f}')
plt.xlabel('Time (Hourly)')
plt.ylabel('Traffic Volume')
plt.legend()
plt.show()
print(f'ARMA(1,0) Predicted Parameters Model\n One Step Ahead Forecasting \n RMSE: {np.sqrt(model.mse)}')

forecast_error = predictions_series.values - Y_test.values
print(f' The Estimated Variance of the Residual Error is {np.var(residuals):0.3f}')
print(f' The Estimated Variance of the Forecast Error is {np.var(forecast_error):0.3f}')
print(f' The Estimated Variance ratio is {np.var(residuals)/np.var(forecast_error):0.2f}')

'''
ARMA(3,2) model
'''
print(20*'= ' + 'ARMA(3,2)' + 20*' =')

na = 3
nb = 2

model = statsmodels.tsa.arima.model.ARIMA(Y_train, order=(na, 0, nb), freq='H').fit()
na_params = model.params[0:na] * -1
nb_params = model.params[na::]
print(model.summary())
# 1-step prediction
model_pred = model.predict(start=1, end=Y_train.shape[0])
residuals = model.resid

# Plots the Training set
plt.figure(figsize=(8, 6))
plt.plot(Y_train, label='Training Set')
plt.plot(model_pred, label='Predicted Values', alpha=0.9)
plt.title('ARMA(3,2) Model\n Prediction of Training Set')
plt.xlabel('Time (Hourly)')
plt.ylabel('Temperature')
plt.legend()
plt.show()

plot_autocorr(acf(residuals.values), lags= 100)
statstoolsACF_PACF(residuals, lags=100, title_str='ARMA(3,2) Residuals\n')

# Generate the Confidence intervals for the Parameters

con_intervals = model.conf_int().drop(['const', 'sigma2'])

if na != 0:
    na_vals = model.arparams * -1
else:
    na_vals = 0
na_con = con_intervals[0:na].values * -1

if nb != 0:
    nb_vals = model.maparams
else:
    nb_vals = 0
nb_con = con_intervals[na::].values

print(f'\n============\nConfidence Interval Results:\n============\n')
print('Na coeffs\n----------')
if len(na_con) == 0:
    print('None')
else:
    [print(f'{na_vals[i]:0.3f}: {na_con[i][0]:0.3f} to {na_con[i][1]:0.3f}') for i in range(len(na_con))]

print('\nNb coeffs\n----------')
if len(nb_con) == 0:
    print('None')
else:
    [print(f'{nb_vals[i]:0.3f}: {nb_con[i][0]:0.3f} to {nb_con[i][1]:0.3f}') for i in range(len(nb_con))]

# Generate the standard deviation
summary = model.summary().as_text()

# Extract the number of items used to make the STE
observations = int(summary[summary.find('No. Observations:') + len('No. Observations:'):summary.find('Model:')].strip())

# Extract the STE for each variable
summary_params = summary[summary.find('const'):summary.find('sigma2')].split('\n')
summary_params = summary_params[1:-1]  # Remove the constant and sigma2 rows

# Extract the STE
collector = []
for param in range(len(summary_params)):
    collector.append([i for i in summary_params[param].split(' ') if i != ''])

collector = pd.DataFrame(collector)
collector = collector.filter([0, 1, 2])

# Convert STE to STD
collector[2] = collector[2].astype(float) * np.sqrt(observations)

# Print the Standard Deviations

na_std = collector.iloc[0:na, :]
nb_std = collector.iloc[na::, :]

print(f'\n============\nStandard Deviation Results:\n============\n')
print('Na coeffs\n----------')
if na_std.shape[0] == 0:
    print('None')
else:
    [print(f'{float(na_std.iloc[i, 1]) * -1:0.3f} STD: {na_std.iloc[i, 2]:0.3f}') for i in range(na_std.shape[0])]

print('\nNb coeffs\n----------')
if nb_std.shape[0] == 0:
    print('None')
else:
    [print(f'{float(nb_std.iloc[i, 1]) * -1:0.3f} STD: {nb_std.iloc[i, 2]:0.3f}') for i in range(nb_std.shape[0])]

# Chi-squared Test
N = len(Y_train)
DOF = 24 - na - nb
alpha = 0.01
QCrit = chi2.ppf(1 - alpha, DOF)
Q, p = sm.stats.acorr_ljungbox(residuals, lags=[24])
if Q < QCrit:
    print('The Residual errors can be considered as random')
else:
    print('The Residual errors are NOT RANDOM for this model\n')
print('The Critical value: ', QCrit)
print('The Q value: ', Q)
print('The P value: ', p)
print(f'\nDegrees of Freedom: {DOF}')

# Display poles and zeros (roots of numerator and roots of denominator)
print('\n Zero-Pole Cancellation\n')
params = model.params[1:-1]
poly_y = params[0:na].values
poly_e = params[na::].values

try:
    zeros = np.poly(poly_e)[1::]
    zero_pole_print(zeros)
except:
    print('(1-0)')

print('----------------')
try:
    poles = np.poly(poly_y)[1::]
    zero_pole_print(poles)
except:
    print('(1-0)')

# Display the estimated variance of error.
params = model.params[1:-1]
na_params = np.array([1] + list(params[0:na].values))
nb_params = np.array([1] + list(params[na::].values))

if na == 0:
    na_params = np.zeros(nb_params.shape)
    na_params[0] = 1

if na_params.shape[0] < nb_params.shape[0]:
    na_params = np.pad(na_params, (0, nb_params.shape[0] - na_params.shape[0]), 'constant')

if nb == 0:
    nb_params = np.zeros(na_params.shape)
    nb_params[0] = 1

if nb_params.shape[0] < na_params.shape[0]:
    nb_params = np.pad(nb_params, (0, na_params.shape[0] - nb_params.shape[0]), 'constant')

# Construct the system in reverse to get the error instead of the y
sys = (na_params, nb_params, 1)

# Generate a white noise set using these params
wn = create_samples(residuals.shape[0], wn_mean=0, wn_var=1)

# Process the system
_, e_dlsim = signal.dlsim(sys, wn)

print(f' The Estimated Variance of the Error is {np.var(e_dlsim):0.3f}')

# Check for Bias
print(f'The Mean of the Residuals is {np.mean(residuals):0.2f}')

# Covarience heatmap of all features
cov = model.cov_params()
cov = cov.drop(['const', 'sigma2'], axis=0)
cov = cov.drop(['const', 'sigma2'], axis=1)

fig, ax = plt.subplots(figsize=[9, 8])
sns.heatmap(cov, center=0, cmap='vlag', annot=True, fmt='0.4f', ax=ax)
plt.title("Covariance Matrix of the Estimated Parameters\n")
plt.show()

# One step ahead Prediction
model_loop = statsmodels.tsa.arima.model.ARIMA(Y_train, order=(na, 0, nb), freq='H').fit()

predictions = []
for i in tqdm(range(len(Y_test))):
    predictions.append(model_loop.forecast(steps=1))
    model_loop = model_loop.append(np.array(Y_test[i:i + 1]))

predictions_series=pd.concat(predictions)
predictions_series.to_csv('One-step-ahead-Prediction-ARMA(3,2).csv')

# Plot the one step ahead - Testing Data Set Only
plt.figure(figsize=(8, 6))
plt.plot(Y_test, label='True Values')
plt.plot(predictions_series, label='Forecast Values', alpha=0.9)
plt.title('ARMA(3,2) Predicted Parameters Model\n One Step Ahead Testing Set')
plt.xlabel('Time (Hourly)')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Plot the one step ahead -  Full Data Set
plt.figure(figsize=(8, 6))
plt.plot(Y_train, label='Training Set')
plt.plot(Y_test, label='Testing Set')
plt.plot(predictions_series, label='Forecast', alpha=0.9)
plt.title(f'ARMA(3,2) Predicted Parameters Model\n One Step Ahead Forecasting\nMSE: {model.mse:0.2f}')
plt.xlabel('Time (Hourly)')
plt.ylabel('Temperature')
plt.legend()
plt.show()
print(f'ARMA(3,2)Predicted Parameters Model\n One Step Ahead Forecasting \n RMSE: {np.sqrt(model.mse)}')

forecast_error = predictions_series.values - Y_test.values
print(f' The Estimated Variance of the Residual Error is {np.var(residuals):0.3f}')
print(f' The Estimated Variance of the Forecast Error is {np.var(forecast_error):0.3f}')
print(f' The Estimated Variance ratio is {np.var(residuals) / np.var(forecast_error):0.2f}')

'''
SARIMA(2,0,0)(2,0,0)24 model
'''
print(20*'= ' + 'SARIMA(2,0,0)(2,0,0)24 ' + 20*' =')

na = 2
nb = 0

model = statsmodels.tsa.arima.model.ARIMA(Y_train, order=(na, 0, nb,), seasonal_order=(2, 0, 0, 24), freq='H').fit()
na_params = model.params[0:na] * -1
nb_params = model.params[na::]
print(model.summary())

model_pred = model.predict(start=1, end=Y_train.shape[0])
residuals = model.resid


plt.figure(figsize=(8, 6))
plt.plot(Y_train, label='Training Set')
plt.plot(model_pred, label='Predicted Values', alpha=0.9)
plt.title('SARIMA(1,0,0)(1,0,0)24  Model\n Prediction of Training Set')
plt.xlabel('Time (Hourly)')
plt.ylabel('Temperature')
plt.legend()
plt.show()

plot_autocorr(acf(residuals.values), lags= 100)
statstoolsACF_PACF(residuals, lags=100, title_str='SARIMA(2,0,0)(2,0,0)24 Residuals\n')

# Generate the Confidence intervals for the Parameters
con_intervals = model.conf_int().drop(['const', 'sigma2'])

if na != 0:
    na_vals = model.arparams * -1
else:
    na_vals = 0
na_con = con_intervals[0:na].values * -1

seasonal_na_vals = model.seasonalarparams * -1
seasonal_na_con = con_intervals[na::].values * -1

print(f'\n============\nConfidence Interval Results:\n============\n')
print('Na coeffs\n----------')
if len(na_con) == 0:
    print('None')
else:
    [print(f'{na_vals[i]:0.3f}: {na_con[i][0]:0.3f} to {na_con[i][1]:0.3f}') for i in range(len(na_con))]

print('\nSeasonal Na coeffs\n----------')
[print(f'{seasonal_na_vals[i]:0.3f}: {seasonal_na_con[i][0]:0.3f} to {seasonal_na_con[i][1]:0.3f}') for i in
 range(len(na_con))]

print('\nNb coeffs\n----------')
print('None')

# Generate the standard deviation
summary = model.summary().as_text()

# Extract the number of items used to make the STE
observations = int(summary[summary.find('No. Observations:') + len('No. Observations:'):summary.find('Model:')].strip())

# Extract the STE for each variable
summary_params = summary[summary.find('const'):summary.find('sigma2')].split('\n')
summary_params = summary_params[1:-1]  # Remove the constant and sigma2 rows

# Extract the STE
collector = []
for param in range(len(summary_params)):
    collector.append([i for i in summary_params[param].split(' ') if i != ''])

collector = pd.DataFrame(collector)
collector = collector.filter([0, 1, 2])

# Convert STE to STD
collector[2] = collector[2].astype(float) * np.sqrt(observations)

# Print the Standard Deviations

na_std = collector.iloc[0:na, :]
seasonal_na_std = collector.iloc[na::, :]

print(f'\n============\nStandard Deviation Results:\n============\n')
print('Na coeffs\n----------')
if na_std.shape[0] == 0:
    print('None')
else:
    [print(f'{float(na_std.iloc[i, 1]) * -1:0.3f} STD: {na_std.iloc[i, 2]:0.3f}') for i in range(na_std.shape[0])]
print('\nSeasonal Na coeffs\n----------')
[print(f'{float(seasonal_na_std.iloc[i, 1]) * -1:0.3f} STD: {seasonal_na_std.iloc[i, 2]:0.3f}') for i in
 range(na_std.shape[0])]

print('\nNb coeffs\n----------')
print('None')

# Chi-squared Test

N = len(Y_train)
DOF = 24 - na - nb
alpha = 0.01
QCrit = chi2.ppf(1 - alpha, DOF)
Q, p = sm.stats.acorr_ljungbox(residuals, lags=[24])
if Q < QCrit:
    print('The Residual errors can be considered as random')
else:
    print('The Residual errors are NOT RANDOM for this model\n')
print('The Critical value: ', QCrit)
print('The Q value: ', Q)
print('The P value: ', p)
print(f'\nDegrees of Freedom: {DOF}')

# Display poles and zeros (roots of numerator and roots of denominator)
print('\n Zero-Pole Cancellation\n')
params = model.params[1:-1]
poly_y = params[0:na + 2].values
poly_e = params[na + 2::].values

try:
    zeros = np.poly(poly_e)[1::]
    zero_pole_print(zeros)
except:
    print('(1-0)')

print('----------------')
try:
    poles = np.poly(poly_y)[1::]
    zero_pole_print(poles)
except:
    print('(1-0)')

# Display the estimated variance of error.
params = model.params[1:-1]
na_params = np.array([1] + list(params[0:na].values) + ([0] * 23) + [params[na]])
nb_params = np.array([1] + list(params[na::].values))

if na == 0:
    na_params = np.zeros(nb_params.shape)
    na_params[0] = 1

if na_params.shape[0] < nb_params.shape[0]:
    na_params = np.pad(na_params, (0, nb_params.shape[0] - na_params.shape[0]), 'constant')

if nb == 0:
    nb_params = np.zeros(na_params.shape)
    nb_params[0] = 1

if nb_params.shape[0] < na_params.shape[0]:
    nb_params = np.pad(nb_params, (0, na_params.shape[0] - nb_params.shape[0]), 'constant')

# Construct the system in reverse to get the error instead of the y
sys = (na_params, nb_params, 1)

# Generate a white noise set using these params
wn = create_samples(residuals.shape[0], wn_mean=0, wn_var=1)

# Process the system
_, e_dlsim = signal.dlsim(sys, wn)

print(f' The Estimated Variance of the Error is {np.var(e_dlsim):0.3f}')

# Check for Bias
print(f'The Mean of the Residuals is {np.mean(residuals):0.2f}')

# Covarience heatmap of all features
cov = model.cov_params()
cov = cov.drop(['const', 'sigma2'], axis=0)
cov = cov.drop(['const', 'sigma2'], axis=1)

fig, ax = plt.subplots(figsize=[6, 6])
sns.heatmap(cov, center=0, cmap='vlag', annot=True, fmt='0.6f', ax=ax)
plt.title("Covariance Matrix of the Estimated Parameters\n")
plt.show()

na = 1
nb = 0

# One step ahead Prediction

model_loop = statsmodels.tsa.arima.model.ARIMA(Y_train, order=(na, 0, nb), seasonal_order=(1, 0, 0, 24),freq='H').fit()

predictions = []
for i in tqdm(range(len(Y_test))):
    predictions.append(model_loop.forecast(steps=1))
    model_loop = model_loop.append(np.array(Y_test[i:i + 1]))

predictions_series=pd.concat(predictions)
predictions_series.to_csv('One-step-ahead-Prediction-SARIMA(2,0,0).csv')

# Plot the one step ahead - Testing Data set
plt.figure(figsize=(8, 6))
plt.plot(Y_test, label='True Values')
plt.plot(predictions_series, label='Forecast Values', alpha=0.9)
plt.title('SARIMA(2,0,0)(2,0,0)24 Predicted Parameters Model\n One Step Ahead Testing Set')
plt.xlabel('Time (Hourly)')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Plot the one step ahead - Full Dataset
plt.figure(figsize=(8, 6))
plt.plot(Y_train, label='Training Set')
plt.plot(Y_test, label='Testing Set')
plt.plot(predictions_series, label='Forecast', alpha=0.9)
plt.title(f'SARIMA(2,0,0)(2,0,0)24 Predicted Parameters Model\n One Step Ahead Forecasting\nMSE: {model.mse:0.2f}')
plt.xlabel('Time (Hourly)')
plt.ylabel('Temperature')
plt.legend()
plt.show()

print(f'SARIMA(2,0,0)(2,0,0)24Predicted Parameters Model\n One Step Ahead Forecasting \n RMSE: {np.sqrt(model.mse)}')

# Calculate the forecast error
forecast_error = predictions_series.values - Y_test.values
print(f' The Estimated Variance of the Residual Error is {np.var(residuals):0.3f}')
print(f' The Estimated Variance of the Forecast Error is {np.var(forecast_error):0.3f}')
print(f' The Estimated Variance ratio is {np.var(residuals) / np.var(forecast_error):0.2f}')

'''
Final Model SARIMA(2,0,0)(2,0,0)24
'''
print(20*'= ' + 'Final Model SARIMA(2,0,0)(2,0,0)24' + 20*' =')

na = 2
nb = 0

model = statsmodels.tsa.arima.model.ARIMA(Y_train, order=(na, 0, nb), seasonal_order=(2, 0, 0, 24),freq='H').fit()
na_params = model.params[0:na] * -1
nb_params = model.params[na::]


# 1-step prediction Using the statsmodel model for comparison
model_pred = model.predict(start=1, end=Y_train.shape[0])
residuals = model.resid

print(model.summary())


# Plots the Training set using the statsmodel model
plt.figure(figsize=(8, 6))
plt.plot(Y_train, label='Training Set')
plt.plot(model_pred, label='Predicted Values', alpha=0.9)
plt.title('SARIMA(2,0,0)(2,0,0)24 Model\n Prediction of Training Set')
plt.xlabel('Time (Hourly)')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Autocorrelation
plot_autocorr(acf(residuals), lags = 24)

statstoolsACF_PACF(residuals, lags=24, title_str='SARIMA(2,0,0)(2,0,0)24 Residuals\n')

#model_pred_test = model.predict(start=1, end=len(Y_test))

# Manual 1-step Prediction
values = pd.Series(np.zeros(Y_train.shape[0]))

# The training dataset, but re-indexed for ease of access
y_predict = Y_train.copy()
y_predict.index = [i for i in range(0, y_predict.shape[0])]


# Now, incrementally make predictions
for i in range(1, y_predict.shape[0]):
    values[i] = (-1.0609 * y_predict.iloc[i-1] + 0.1301 * y_predict.iloc[i-2] +
                 0.6019 * y_predict.iloc[i-24] + 0.3002 * y_predict.iloc[i-48])

values.index = Y_train.index
one_step = pd.Series(values.iloc[1::])
residuals = Y_train.iloc[1::] - one_step

# Plot the Training set
plt.figure(figsize=(8, 6))
plt.plot(Y_train, label='Training Set')
plt.plot(model_pred, label='Model Predicted Values', alpha=0.9)
plt.plot(one_step, label='Forecast Function Predicted Values', alpha=0.9)
plt.title('SARIMA(2,0,0)(2,0,0)24  Model\n Prediction of Training Set')
plt.xlabel('Time (Hourly)')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# SARIMA(2,0,0)(2,0,0)24 model - H-step Forecast Model

def h_step_prediction(train_in, test_in, steps=50, return_val=False):

    furthest_back = 1

    varience_pred = pd.Series(dtype='float64')

    values = pd.Series(np.zeros((steps + furthest_back + 1)))

    values.index = [i for i in range(-1 * furthest_back, steps + 1)]

    for i in range(0, 1):
        values[i - 1] = train_in.iloc[i]

    # Now, incrementally make predictions
    for i in range(1, steps + 1):
            values[i] = (-1.0609 * y_predict.iloc[i-1] + 0.1301 * y_predict.iloc[i-2] +
                 0.6019 * y_predict.iloc[i-24] + 0.3002 * y_predict.iloc[i-48])

    pred_val = pd.Series(values[furthest_back + 1::])


    varience_pred = varience_pred.append(pred_val)
    varience_pred.index = test_in.index[:steps]

    if return_val == True:
        return varience_pred

    else:

        plt.figure(figsize=(8, 6))
        plt.plot(test_in[:steps], label='Test Values')
        plt.plot(test_in.index[:steps], pred_val[:steps], label='Forecast Values',alpha=0.9)
        plt.xlabel('Time (Hourly)')
        plt.ylabel('Temperature')
        plt.title(f'{steps} Step Prediction \nForecasted Values vs True Values')
        plt.legend()
        plt.show()

# Plot 50 step Prediction
h_step_prediction(Y_train, Y_test, steps=50)

# Plot full testing set
h_step_prediction(Y_train, Y_test, steps=Y_test.shape[0])

