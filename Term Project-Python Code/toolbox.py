import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from statsmodels.stats.diagnostic import acorr_ljungbox
import seaborn as sns
from scipy.signal import dlsim
import math


def cal_rolling_mean_var(x):
    x = pd.DataFrame(x)
    title = str(x.columns.values)
    rol_mean = []
    rol_var = []
    for i in range(len(x)):
        mean = x.head(i).mean()
        rol_mean.append(mean)
        var = (x.head(i).std()) ** 2
        rol_var.append(var)
    plt.figure()
    plt.subplot(2,1,1)
    plt.subplots_adjust(hspace = 0.5)
    plt.plot(rol_mean, label = 'Rolling Mean', color = 'darkslateblue')
    plt.xlabel('Samples')
    plt.ylabel('Mean')
    plt.title('Rolling Mean of ' + title[2:-2],loc = 'left',fontsize = 10, fontweight="bold")
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(rol_var, label = 'Rolling Variance',color = 'darkolivegreen')
    plt.xlabel('Samples')
    plt.ylabel('Variance')
    plt.title('Rolling Variance of ' + title[2:-2],loc = 'left',fontsize = 10, fontweight="bold")
    plt.legend()
    plt.grid()
    plt.show()


def KPSS_Test(timeseries):
    print ('Results of KPSS Test for ' + str(timeseries.name) +':')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)


def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic for "+ str(x.name) +": %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def statstoolsACF_PACF(system_in, lags=20, title_str=''):
    acf = sm.tsa.stattools.acf(system_in, nlags=lags, fft=False)
    pacf = sm.tsa.stattools.pacf(system_in, nlags=lags)

    plt.figure()
    plt.subplot(211)
    plot_acf(system_in, ax=plt.gca(), lags=lags)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.subplot(212)
    plot_pacf(system_in, ax=plt.gca(), lags=lags)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.suptitle(f'ACF and PACF:{title_str}')
    plt.tight_layout()
    plt.show()

def diff_seasonal_calc(df, interval=1):
    diff = []
    for i in range(interval, len(df)):
      value = df[i] - df[i - interval]
      diff.append(value)

    diff=pd.DataFrame(diff)
    return diff


def auto_corr_cal(array_in, k):

    if k == 0:
        return 1

    elif k >= len(array_in):
        raise ValueError('K exceeds maximum value.')

    else:

        t = k + 1 - 1  # -1 for 0 indexing

        y_t = array_in[t::]

        y_t_minusk = array_in[0:-k]

        y_mean = array_in.mean()

        y_t = y_t - y_mean
        y_t_minusk = y_t_minusk - y_mean

        numerator = sum(y_t * y_t_minusk)
        denom = sum((array_in - y_mean) ** 2)

        return numerator / denom

def calc_Q_Score(residuals, train_in, lags=5, print_out=False):

    num_samples = len(train_in)
    auto_corr = []

    residuals = residuals[~np.isnan(residuals)]

    for i in range(lags):
        auto_corr.append(auto_corr_cal(residuals, i))

    summed = np.sum(np.array(auto_corr)[1:] ** 2)
    q_score = num_samples * summed

    if print_out != False:
        print(f' The Q Score is {q_score: 0.3f}\n')
    return q_score


def Cal_GPAC(acf, cols, rows):
    GPAC_table = np.zeros((cols, rows))
    mid = int(len(acf) / 2)
    for j in range(rows):
        for k in range(1, cols + 1):
            num = np.zeros((k, k))
            den = np.zeros((k, k))

            acf_counter = mid + j

            for c in range(k):
                k_counter = 0
                for r in range(k):
                    den[r, c] = acf[acf_counter + k_counter]
                    k_counter += 1
                acf_counter -= 1

            num[:, :-1] = den[:, :-1]

            acf_counter = mid + j
            for r in range(k):
                num[r, -1] = acf[acf_counter + 1]
                acf_counter += 1

            num_det = np.linalg.det(num)
            den_det = np.linalg.det(den)

            gpac_value = num_det / den_det
            GPAC_table[j, k-1] = gpac_value
        xticks = np.arange(1,k+1,1)

    plt.subplots(figsize=(15,10))
    ax = sns.heatmap(GPAC_table, vmin=-1, vmax=1, center=0, square=True, cmap='magma', annot=True, fmt='.3f')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xticklabels(xticks, horizontalalignment='center')
    ax.set(xlabel='k', ylabel='j')
    plt.title('Generalized Partial AutoCorrelation (GPAC) Table')
    plt.show()


def createGPACTable(auto_corr_in, j_val=12, k_val=12):
    j_val += 1
    k_val += 1

    big_out = np.zeros([j_val, k_val])

    for k in range(1, k_val):
        for j in range(0, j_val):

            full_array = []

            # Generate the top row
            ry_array = [i for i in range((j - k + 1),
                                         j + 1)]  # Generates the array for the top row of the denominator, but backwards
            ry_array = ry_array[::-1]  # Reverse to proper order
            ry_array = np.array(ry_array)

            # Generate the rest based off that row
            for i in range(0, k):
                full_array.append(ry_array + i)

            # Concat to a single array
            full_array = np.array(full_array)

            # Make the nuerator array
            numer_array = full_array.copy()

            # The fiinal column of the numerator is different. Generate it here
            numer_col = [i for i in range(j + 1, j + k + 1)]

            # Replace the final col
            numer_array[:, -1] = numer_col

            # Take the absolute values  - all negative Ry are equivalent to Positive Ry
            numer_array = abs(numer_array) * 1.0
            denom_array = abs(full_array) * 1.0

            # Get the respective numbers from the autocor array
            numer_array2 = numer_array.copy()

            for i in range(numer_array2.shape[0]):
                for n in range(numer_array2.shape[1]):
                    numer_array2[i, n] = auto_corr_in[int(numer_array2[i, n])]

            denom_array2 = denom_array.copy()

            for i in range(denom_array2.shape[0]):
                for n in range(denom_array2.shape[1]):
                    denom_array2[i, n] = auto_corr_in[int(denom_array2[i, n])]

            val = np.linalg.det(numer_array2) / np.linalg.det(denom_array2)

            big_out[j, k] = val
            big_out_display = pd.DataFrame(big_out)

    big_out_display = pd.DataFrame(big_out)

    return big_out_display.iloc[:, 1:]


def plotGPAC(gpac_table, equation_string='', decimals=1):

    num_format = '0.' + str(decimals) + 'f'
    if equation_string != '':
        title_str = "GPAC Table\n" + equation_string
    else:
        title_str = "GPAC Table"

    fig, ax = plt.subplots(figsize=[9, 7])
    sns.heatmap(gpac_table, vmin=-1, vmax=1, cmap='BrBG', annot=True, fmt=num_format, ax=ax)
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=True, bottom=True, top=True, labeltop=True)
    plt.yticks(rotation=0)
    plt.xlabel('k', fontsize=15)
    plt.ylabel('j', fontsize=15)
    plt.title(title_str, fontsize=15)
    plt.show()


def createGPAC(system_in, equation_string='', j_val=12, k_val=12):
    autocor_system = sm.tsa.stattools.acf(system_in, nlags=100, fft=False)
    gpac_table_plot = createGPACTable(autocor_system, j_val, k_val)
    plotGPAC(gpac_table_plot, equation_string)


def zero_pole_print(poly_in):
    collect=[]
    for i in poly_in:
        if i<0:
            collect.append(str(f"(1+{i*-1:0.4f})"))
        else:
            collect.append(str(f"(1-{i:0.4f})"))
    str_out=''
    for i in collect:
        str_out+=i
    print(str_out)

def Q_pred(data, residuals, deg,lags = 10):
    from scipy.stats import chi2
    N = len(data)
    DOF = N - deg
    alpha = 0.01
    QCrit = chi2.ppf(1 - alpha, DOF)
    Q, p = sm.stats.acorr_ljungbox(residuals, lags=[lags])
    if Q < QCrit:
        print('The Residual errors can be considered as random')
    else:
        print('The Residual errors are NOT RANDOM for this model\n')
    print('The Critical value: ', QCrit)
    print('The Q value: ', Q)
    print('The P value: ', p)


def create_samples(n_samples, wn_mean=0, wn_var=1):
    wn_std = np.sqrt(wn_var)
    np.random.seed(42)
    e = np.random.normal(wn_mean, wn_std, size=n_samples)
    return e


def average_train(train_in):
    if type(train_in) == list:
        train_in = np.array(train_in)

    # Capture predicted values
    pred_y = []

    # Loop through and get the predicted values by averaging
    for i in range(len(train_in)):
        pred_y.append(np.mean(train_in[0:i]))

    # Calculate error
    error = train_in - pred_y
    error_2 = error ** 2

    return pred_y, error, error_2



def drift_train(train_in):

    if type(train_in) == list:
        train_in = np.array(train_in)

    # Capture predicted values
    pred_y = [np.nan]

    # Loop through and get the predicted values
    for i in range(2, len(train_in)):
        fullT = train_in[0:i]
        val = fullT[-1] + 1 * ((fullT[-1] - fullT[0]) / (len(fullT) - 1))

        pred_y.append(val)

    # Calculate error
    error = np.array(train_in[1::]) - np.array(pred_y)
    error_2 = error ** 2

    return pred_y, error, error_2


def average_test(train_in, test_in):

    if type(train_in) == list:
        train_in = np.array(train_in)

    if type(test_in) == list:
        train_in = np.array(test_in)

    # Take mean of training set, this will be the value for all testing points
    train_mean = np.mean(train_in)
    pred_y = np.ones([len(test_in)]) * train_mean

    # Calculate error
    error = test_in - pred_y
    error_2 = error ** 2

    return pred_y, error, error_2


def naive_train(train_in):

    if type(train_in) == list:
        train_in = np.array(train_in)

    # Capture predicted values
    pred_y = []

    # Loop through and get the predicted values by averaging
    for i in range(1, len(train_in)):
        pred_y.append(train_in[i - 1])

    # Calculate error
    error = train_in[1::] - pred_y
    error_2 = error ** 2

    return pred_y, error, error_2


def naive_test(train_in, test_in):

    if type(train_in) == list:
        train_in = np.array(train_in)

    if type(test_in) == list:
        test_in = np.array(test_in)

    # Take mean of training set, this will be the value for all testing points
    pred_y = np.ones([len(test_in)]) * train_in.iloc[-1]

    # Calculate error
    error = test_in - pred_y
    error_2 = error ** 2

    return pred_y, error, error_2


def drift_test(train_in, test_in):

    if type(train_in) == list:
        train_in = np.array(train_in)

    if type(test_in) == list:
        test_in = np.array(test_in)

    pred_y = []

    # Loop through the testing values
    for i in range(0, len(test_in)):
        val = train_in.iloc[-1] + (i + 1) * ((train_in.iloc[-1] - train_in.iloc[0]) / (len(train_in) - 1))
        pred_y.append(val)

    # Calculate error
    error = test_in - pred_y
    error_2 = error ** 2

    return pred_y, error, error_2


def ses_train(train_in, alpha=0.5):

    if type(train_in) == list:
        train_in = np.array(train_in)

    # Capture predicted values -pre-load with 1st value
    pred_y = [train_in[0]]

    # Loop through and get the predicted values
    for i in range(1, len(train_in)):
        val = alpha * train_in[i] + (1 - alpha) * pred_y[-1]
        pred_y.append(val)

    # Calculate error
    error = np.array(train_in[1::]) - np.array(pred_y[0:-1])
    error_2 = error ** 2

    return pred_y, error, error_2


def ses_test(train_in, test_in, alpha=0.5):

    if type(train_in) == list:
        train_in = np.array(train_in)

    if type(test_in) == list:
        test_in = np.array(test_in)

    # Get the prediction for the final value in the training set
    final_pred = ses_train(train_in, alpha=0.5)[0][-1]

    val = alpha * test_in.iloc[-1] + (1 - alpha) * final_pred

    # Take mean of training set, this will be the value for all testing points
    pred_y = np.ones([len(test_in)]) * val

    # Calculate error
    error = test_in - pred_y
    error_2 = error ** 2

    return pred_y, error, error_2

def plot_prediction_method_axis(train_in, test_in, pred_in, error2, method_str):

    fig, ax = plt.subplots()
    ax.plot(train_in.index, train_in, label='Train')
    ax.plot(test_in.index,test_in, label='Test')
    ax.plot(test_in.index,pred_in, label='Forecast', alpha=0.9)
    plt.legend(loc='best')
    plt.title(f'Temperature Predictions using {method_str}\n RMSE: {error2.mean() :0.2f}', fontsize=14)
    plt.xlabel('Time (Hourly)')
    plt.ylabel('Temperature')
    plt.show()
    return

def inverse_diff(y,z_hat,interval =1):
    y_new = np.zeros(len(y))
    for i in range(1,len(z_hat)):
        y_new[i] = z_hat.iloc[i-interval] +y.iloc[i-interval]
    y_new = y_new[1:]
    return y_new
