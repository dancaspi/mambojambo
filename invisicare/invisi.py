import pandas as pd
from fbprophet.plot import add_changepoints_to_plot
from fbprophet import Prophet
import os
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
data_folder="invisi_data"
df_all = pd.read_csv(os.path.join(data_folder,'invisi_iucs_calls_cdr.csv'))

client_imsi = 425030040831554
#client_phone_number = '050-5339478'

'''
library(prophet)
library(Slade)
library(ggplot2)
df = read.csv("invisi_iucs_calls_cdr.csv")
dfa = df[df[,"imei"] == df[99976,"imei"],]
dfa = dfa[dfa[,"calltype"] %in% c(2081,2091),]
dfa = dfa[dfa[,"transactiontype"] %in% c(86),]

x = dfa[,"duration"]
xd = aggregate(dfa[,"duration"],by=list(dfa[,"start_date"]),mean)[,"x"]

aggd = aggregate(dfa[,"duration"],by=list(dfa[,"start_date"]),mean)
colnames(aggd) <- c("ds","y")
m <- prophet(aggd)
future <- make_future_dataframe(m, periods = 30)
forecast <- predict(m, future)
plot(m, forecast) + add_changepoints_to_plot(m)
'''

df = df_all.copy()


df = df[(df.imsi==client_imsi)]
df = df[(df.calltype == 2081) | (df.calltype==2091)]
df = df[df.transactiontype == 86]




'minutes talked'
'times talked'
'time talked incoming'
'time talked outgoing'


'''
@param history: entire time series data for the customer
@param daysBackAlert: is alert in past $daysBackAlert days 
@param daysBackTrend: 
@param: ts: two columns, time in column named start_date and the monitored value in a column named val

@return:
'''

def f(ts,ci=0.99, trendTh=0.01,outlier_pctg=3, daysBackAlert=1, daysBackTrend=365):

    xd = ts.copy()
    xd = xd.reset_index(drop=False)
    xd = xd.rename(index=str, columns={"start_date": "ds", "val": "y"})

    bottom_th = np.percentile(xd.y, outlier_pctg)
    top_th = np.percentile(xd.y, 100 - outlier_pctg)
    xd = xd[(xd.y > bottom_th) & (xd.y < top_th)]

    m = Prophet(interval_width=ci)
    m.fit(xd)
    future = m.make_future_dataframe(periods=0)  ## make dataframe for the future along with the past
    forecast = m.predict(future)
    #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    #fig1 = m.plot(forecast)
    #a = add_changepoints_to_plot(fig1.gca(), m, forecast, threshold=0.001)
    #fig1.show()


    ret = []
    trendAlerts = m.changepoints[np.abs(np.nanmean(m.params['delta'], axis=0)) >= trendTh]
    deltas = m.params['delta'][0][np.abs(np.nanmean(m.params['delta'], axis=0)) >= trendTh]
    for alert,delta in zip(trendAlerts,deltas):
        if pd.to_datetime(alert)> pd.to_datetime(pd.to_datetime(dt.date.today()) - pd.Timedelta(days=daysBackTrend)):
            ret += dict(
                alertType="trend",
                alertStartDate=alert,
                direction=delta
            )
    ts.start_date=pd.to_datetime(ts.start_date)

    xx = pd.merge(left_on = 'start_date',right_on='ds',left=ts,right=forecast)

    alerts = xx.tail(daysBackAlert).apply(
        lambda x: (x['val'] > x['yhat_upper']) or ((x['val'] < x['yhat_lower'])),axis=1)

    ret += list(xx.tail(daysBackAlert)[alerts].apply(
        lambda x:
        dict(
        alertType="abrupt",
        alertStartDate=str(x['start_date']),
        direction=x['val'] - x['yhat_upper'] if x['val'] > x['yhat_upper'] else x['val'] - x['yhat_lower']
        )
        ,axis=1
    ))

    return ret



