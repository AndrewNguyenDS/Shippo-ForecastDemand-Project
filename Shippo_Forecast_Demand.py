#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import datetime
import math
import os


# In[ ]:



print("start run script forecast_today.py")
print(datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S"))
## Create engine and connect to server
engine = create_engine("postgresql+psycopg2://shippo_bi:kfwZh2vxfntQtfx4@shippo-production.csgydtxz27xw.ap-southeast-1.rds.amazonaws.com:5432/shippo_production")
connection = engine.connect()
## Query data from database
data = pd.read_sql_query('SELECT ("createdTime" + interval \'7 hours\') as "createdTime", "deliverFromProvinceId" as "Location" FROM "DeliveryOrder" WHERE ("deliverFromProvinceId" =9 OR "deliverFromProvinceId" =80) and DATE("createdTime" + interval \'7 hours\') < DATE(now() + interval \'7 hours\') and DATE("createdTime" + interval \'7 hours\') > DATE(\'1/1/2018\')' , connection)

users_df = pd.read_sql_query("SELECT id,Date(\"firstUsingService\" + interval '7 hours') as to_date FROM public.\"Users\"where realm='customer'", connection)



users_df.to_date = pd.to_datetime(users_df.to_date)


users_df.head()



salesman = pd.read_excel("./ShippoSales.xlsx")


# In[ ]:


# # END OF LOADING DATA
# ## PREPROCESSING DATA BELOW



## Rule Chia ca 
def rule(time):
    hour = time.hour
    if hour < 6:
        return (0,0)
    elif 6<= hour < 11:
        return (0,1)
    elif 11<= hour < 16:
        return (0,2)
    elif 16 <= hour < 19:
        return (0,3)
    else:
        return (1, 0)

## Get active users
def get_active_users_last_month(date):
    active_order = users_df[np.logical_and(users_df.to_date > date - pd.DateOffset(days = 7), users_df.to_date < date)]
    active_user = active_order.customerId.nunique()
    return active_user

def active_users(row):
    row["active_user"] = get_active_users_last_month(row['ngay_giao'])
    return row

## Filter holidays
def holiday(row):
    if row['ngay_giao'] in holidays:
        row['demand'] = np.nan
    for i in range(6):
        row['before_holiday'] = 0
        if row['ngay_giao'] + pd.DateOffset(days = i+1) in holidays and row['ngay_giao'] + pd.DateOffset(days = i+1) not in sundays:
            row['before_holiday'] = 7-i
            break
    return row

def saturday(row):
    if row.ngay_giao in sats and row.ca_giao > 1:
        row['demand'] = np.nan
    return row
## Get # new users last week
def get_new_users_last_month(date):
    temp = users_df[np.logical_and(users_df.to_date > date - pd.DateOffset(days = 7), users_df.to_date < date)]
    new_users = temp.shape[0]
    return new_users
def new_users(row):
    row["new_users"] = get_new_users_last_month(row['ngay_giao'])
    return row
    

## Handle Missing data:
def get_date(d):
    return pd.to_datetime("{}/{}/{}".format(d[:2], d[2:4], d[4:8]))
def subtract_week(d):
    date = get_date(d)
    date = date - pd.DateOffset(days = 7)
    temp = date.strftime("%m%d%Y") + d[-1]
    return temp
def subtract_month(d):
    date = get_date(d)
    date = date - pd.DateOffset(months = 1)
    temp = date.strftime("%m%d%Y") + d[-1]
    return temp
def subtract_day(d):
    date= get_date(d)
    date = date - pd.DateOffset(days=1)
    temp = date.strftime("%m%d%Y") + d[-1]
    return temp
# def estimate(x, i):
#     a = 1.00804
#     b = 6.7392487
#     res = a*x +b
#     if i == 0:
#         return float(res)
#     else:
#         for e in range (i):
#             res = a*res + b
#         return float(res)

def fill_na_custom(row):
    if math.isnan(row.demand):
        row["demand"] = row["fill_data"]
    return row


# In[ ]:


# ## GENERATE HOLIDAYS:


tet2016 = [datetime.date(2017,1,22) + datetime.timedelta(days=x) for x in range(0, 22)]
tet2017 = [datetime.date(2018,2,10) + datetime.timedelta(days=x) for x in range(0, 21)]
nghile2017 = [datetime.date(2016, 12, 31), datetime.date(2017, 1, 2), datetime.date(2017, 1, 1), datetime.date(2017,4, 16),            datetime.date(2017, 4, 29), datetime.date(2017, 4, 30), datetime.date(2017, 5, 1), datetime.date(2017, 5, 2), datetime.date(2017, 9, 2),             datetime.date(2017, 9, 3), datetime.date(2017, 9, 4)]
nghile2018 = [datetime.date(2017, 12, 30),datetime.date(2017, 12, 31), datetime.date(2018, 1, 1),             datetime.date(2018, 4, 25), datetime.date(2018, 4, 28), datetime.date(2018, 4, 29),             datetime.date(2018, 4, 30), datetime.date(2018, 5, 1), datetime.date(2018, 9, 1), datetime.date(2018, 9, 2), datetime.date(2018, 9, 3)]
nghile2019 = [datetime.date(2018,1,31), datetime.date(2019,1,1), datetime.date(2019,1,30), datetime.date(2019,1,31), datetime.date(2019,2,1),datetime.date(2019,2,2), datetime.date(2019,2,3), datetime.date(2019,2,4),
              datetime.date(2019,2,5), datetime.date(2019,2,6), datetime.date(2019,2,7), datetime.date(2019,2,8), datetime.date(2019,2,9), datetime.date(2019,2,10), datetime.date(2019,2,11), datetime.date(2019,2,12), datetime.date(2019,2,13),
              datetime.date(2019,2,14), datetime.date(2019,2,15), datetime.date(2019,2,16)]

sundays = []
def allsundays(year):
   d = datetime.date(year, 1, 1)                    # January 1st
   d += datetime.timedelta(days = 6 - d.weekday())  # First Sunday
   while d.year == year:
      yield d
      d += datetime.timedelta(days = 7)
    
def allsats(year):
   d = datetime.date(year, 1, 1)                    # January 1st
   d += datetime.timedelta(days = 5 - d.weekday())  # First Sunday
   while d.year == year:
      yield d
      d += datetime.timedelta(days = 7)
sundays = [d for d in allsundays(2016)] + [d for d in allsundays(2017)] + [d for d in allsundays(2018)]
holidays = tet2016 + tet2017 + nghile2017 + nghile2018 + nghile2019
holidays = [np.datetime64(i) for i in holidays]
sundays = [np.datetime64(i) for i in sundays]
sats = [d for d in allsats(2016)] + [d for d in allsats(2017)] + [d for d in allsats(2018)]
sats = [np.datetime64(i) for i in sats]


# In[ ]:


## Preprocessing data
## Convert to date time type
data.createdTime = pd.to_datetime(data.createdTime)
## Ap dung Rule chia ca:
result = np.array([rule(xi) for xi in data.createdTime])
## Them cot ngay_giao
data["ngay_giao"] = data.createdTime.values.astype("datetime64[D]")+ result[:,0].astype("timedelta64[D]")
## Them cot ca_giao
data["ca_giao"] = result[:,1]
# data = data[data.ca_giao != 0]
# data = data[data.ngay_giao > pd.to_datetime("2018/01/01")]

HN_sales = salesman.iloc[0].to_dict()
HCM_sales = salesman.iloc[1].to_dict()
data_hn = data[data.Location == 9]
data_hcm = data[data.Location== 80]
data_hn["temp"] = data_hn.createdTime.dt.year.astype(str) + "-" +data_hn.createdTime.dt.month.astype(str) 
data_hcm["temp"] = data_hcm.createdTime.dt.year.astype(str) + "-" +data_hcm.createdTime.dt.month.astype(str)
data_hn["sales"] = data_hn.temp.map(HN_sales).astype(float)
data_hcm["sales"] = data_hcm.temp.map(HCM_sales).astype(float)
del data_hn['Location']
del data_hn['temp']
del data_hcm['temp']
del data_hcm['Location']


# In[ ]:


## Create data for training model:
model_data = []
demand_dicts = []
for df in [data_hn, data_hcm]:
    modelling_data = pd.DataFrame()
    modelling_data = df.groupby(by = ["ngay_giao","ca_giao"]).agg({'createdTime':'count','sales':'mean'})
    modelling_data.columns = ['demand',"sales"]
    modelling_data = modelling_data.reset_index()
    modelling_data = modelling_data.apply(holiday, axis = 1)
#     modelling_data = modelling_data.apply(saturday, axis = 1)
    modelling_data = modelling_data.dropna()
    modelling_data["ngay_ca"] = modelling_data.ngay_giao.dt.strftime("%m%d%Y") + modelling_data.ca_giao.astype(str)
    modelling_data = modelling_data.apply(new_users, axis = 1)

    demand_dict = dict(zip(modelling_data.ngay_ca, modelling_data.demand))
    demand_dicts.append(demand_dict)
    modelling_data = modelling_data.set_index("ngay_ca")

    demand_lag_week = pd.DataFrame()
    demand_lag_week["ngay_giao"] = modelling_data.ngay_giao - pd.DateOffset(days = 7)
    demand_lag_week["ca_giao"] = modelling_data.ca_giao
    demand_lag_week["ngay_ca_1"]= demand_lag_week.ngay_giao.dt.strftime("%m%d%Y") + demand_lag_week.ca_giao.astype(str)
    demand_lag_week["demand"] = demand_lag_week.ngay_ca_1.map(demand_dict)

    demand_lag_2week = pd.DataFrame()
    demand_lag_2week["ngay_giao"] = modelling_data.ngay_giao - pd.DateOffset(days = 14)
    demand_lag_2week["ca_giao"] = modelling_data.ca_giao
    demand_lag_2week["ngay_ca_1"]= demand_lag_2week.ngay_giao.dt.strftime("%m%d%Y") + demand_lag_2week.ca_giao.astype(str)
    demand_lag_2week["demand"] = demand_lag_2week.ngay_ca_1.map(demand_dict)
    
    demand_lag_3week = pd.DataFrame()
    demand_lag_3week["ngay_giao"] = modelling_data.ngay_giao - pd.DateOffset(days = 21)
    demand_lag_3week["ca_giao"] = modelling_data.ca_giao
    demand_lag_3week["ngay_ca_1"]= demand_lag_3week.ngay_giao.dt.strftime("%m%d%Y") + demand_lag_3week.ca_giao.astype(str)
    demand_lag_3week["demand"] = demand_lag_3week.ngay_ca_1.map(demand_dict)

    demand_lag_month = pd.DataFrame()
    demand_lag_month["ngay_giao"] = modelling_data.ngay_giao - pd.DateOffset(months = 1)
    demand_lag_month["ca_giao"] = modelling_data.ca_giao
    demand_lag_month["ngay_ca_1"] = demand_lag_month.ngay_giao.dt.strftime("%m%d%Y") + demand_lag_month.ca_giao.astype(str)
    demand_lag_month["demand"] = demand_lag_month.ngay_ca_1.map(demand_dict)

    demand_lag_day = pd.DataFrame()
    demand_lag_day["ngay_giao"] = modelling_data.ngay_giao - pd.DateOffset(days = 1)
    demand_lag_day["ca_giao"] = modelling_data.ca_giao
    demand_lag_day["ngay_ca_1"] = demand_lag_day.ngay_giao.dt.strftime("%m%d%Y") + demand_lag_day.ca_giao.astype(str)
    demand_lag_day["demand"] = demand_lag_day.ngay_ca_1.map(demand_dict)
    
    
    
    demand_lag_ca1 = pd.DataFrame()
    demand_lag_ca1["ngay_giao"] = modelling_data.ngay_giao
    demand_lag_ca1["ca_giao"] = modelling_data.ca_giao - 1
    demand_lag_ca1["ngay_ca_1"] = demand_lag_ca1.ngay_giao.dt.strftime("%m%d%Y") + demand_lag_ca1.ca_giao.astype(str)
    demand_lag_ca1["demand"] = demand_lag_ca1.ngay_ca_1.map(demand_dict)
    ## FILL NA FOR LAG WEEK
    thresh_hold = np.min(modelling_data.ngay_giao)
    fill_data =[]
    null_demand_lag_week = demand_lag_week[(demand_lag_week.demand).isnull()]
    for index,row in null_demand_lag_week.iterrows():
        d = row["ngay_ca_1"]
        d = subtract_week(d)
        available_data = np.nan
        while get_date(d) > thresh_hold and math.isnan(available_data):
            i = 0
            try:
                available_data = modelling_data.demand.loc[d]
            except:
                available_data = np.nan
            d = subtract_week(d)
            i = i+1
        fill_data.append(available_data)    
    null_demand_lag_week['fill_data'] = fill_data
    fill_data = null_demand_lag_week['fill_data']
    fill_data = fill_data.reset_index()
    demand_lag_week = demand_lag_week.reset_index()
    demand_lag_week = pd.merge(demand_lag_week, fill_data, how="left", on="ngay_ca")
    demand_lag_week = demand_lag_week.set_index("ngay_ca")
    demand_lag_week = demand_lag_week.apply(fill_na_custom, axis = 1)

    ## FILL NA FOR LAG MONTH
    thresh_hold = np.min(modelling_data.ngay_giao)
    fill_data =[]
    null_demand_lag_month = demand_lag_month[(demand_lag_month.demand).isnull()]
    for index,row in null_demand_lag_month.iterrows():
        d = row["ngay_ca_1"]
        d = subtract_month(d)
        available_data = np.nan
        while get_date(d) > thresh_hold and math.isnan(available_data):
            try:
                available_data = modelling_data.demand.loc[d]
            except:
                available_data = np.nan
            d = subtract_month(d)
        fill_data.append(available_data)    
    null_demand_lag_month['fill_data'] = fill_data
    fill_data = null_demand_lag_month['fill_data']
    fill_data = fill_data.reset_index()
    demand_lag_month = demand_lag_month.reset_index()
    demand_lag_month = pd.merge(demand_lag_month, fill_data, how="left", on="ngay_ca")
    demand_lag_month = demand_lag_month.set_index("ngay_ca")
    demand_lag_month = demand_lag_month.apply(fill_na_custom, axis = 1)

    ## Fill NA for lag day:
    thresh_hold = np.min(modelling_data.ngay_giao)
    fill_data =[]
    null_demand_lag_day = demand_lag_day[(demand_lag_day.demand).isnull()]
    for index,row in null_demand_lag_day.iterrows():
        d = row["ngay_ca_1"]
        d = subtract_day(d)
        available_data = np.nan
        while get_date(d) > thresh_hold and math.isnan(available_data):
            try:
                available_data = modelling_data.demand.loc[d]
            except:
                available_data = np.nan
            d = subtract_day(d)
        fill_data.append(available_data)    
    null_demand_lag_day['fill_data'] = fill_data
    fill_data = null_demand_lag_day['fill_data']
    fill_data = fill_data.reset_index()
    demand_lag_day = demand_lag_day.reset_index()
    demand_lag_day = pd.merge(demand_lag_day, fill_data, how="left", on="ngay_ca")
    demand_lag_day = demand_lag_day.set_index("ngay_ca")
    demand_lag_day = demand_lag_day.apply(fill_na_custom, axis = 1)
    modelling_data["demand_lag_week"] = demand_lag_week['demand']
    modelling_data["demand_lag_2week"] = demand_lag_2week['demand']
    modelling_data["demand_lag_3week"] = demand_lag_3week['demand']
    modelling_data['ma_week'] = np.nanmean(modelling_data[['demand_lag_week', 'demand_lag_2week', 'demand_lag_3week']], axis = 1)
    modelling_data["demand_lag_month"] = demand_lag_month['demand']
    modelling_data["demand_lag_day"] = demand_lag_day['demand']
    modelling_data['demand_lag_ca1'] = demand_lag_ca1['demand']
    modelling_data['numDays'] = (modelling_data.ngay_giao -  pd.to_datetime('1/1/2018'))/np.timedelta64(1, 'D') 
    modelling_data['dow'] = modelling_data.ngay_giao.dt.dayofweek
    model_data.append(modelling_data)


# In[ ]:


from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

## làm mô hình cho ca 1 riêng cho kết quả tốt hơn
## Lam mo hinh cho tung ngay trong tuan
model_dict = {}
locs = ['HN',"HCM"]
for i in range(len(model_data)):
    model_data_ = model_data[i]
    for e in range(4):
        for j in range(7):
            temp = model_data_[model_data_.dow == j]
            X_y = temp[['ma_week',"numDays","demand_lag_week", "demand_lag_month","before_holiday","new_users",'sales','demand']]
            X_y = X_y.dropna()
            X_ = X_y[["numDays","demand_lag_week", "demand_lag_month","before_holiday","new_users",'sales']]
            y_ = X_y['demand']
            X = X_[X_.index.str[-1]=='{}'.format(e)]
            y = y_[y_.index.str[-1]=='{}'.format(e)]
            performance = 10000
            mod = 0
            X_1, X_test, y_1, y_test = train_test_split(X, y, test_size = 0.2)
            X_train, X_valid, y_train, y_valid = train_test_split(X_1, y_1, test_size = 0.2)
            alphas = [0.01, 0.1, 0.3, 0.5, 0.8]
            for alpha in alphas:
                model = linear_model.Lasso(alpha = alpha, normalize=True)
                model.fit(X_train, y_train)
                per = model.score(X_valid, y_valid)
                if per < performance:
                    performance = per
                    mod = model
                    # print(alpha)
            performance = model.score(X_test, y_test)
            model_dict['{0}_{1}_{2}'.format(locs[i], e, j)] = [mod, performance]

model_dict


# In[ ]:


def holiday_date(date):
    if date in holidays:
        return 0
    for i in range(6): 
        res = 0
        if date + np.timedelta64(i,"D") in holidays and date + np.timedelta64(i, "D") not in sundays:
            res= 7-i
            break
    return res

result = pd.read_csv("shippo_forecast_week.csv", header = None)
result[0] = pd.to_datetime(result[0])
result = result[np.logical_and(result[0]<=pd.Timestamp.today() - pd.DateOffset(n=1), result[0]>pd.Timestamp.today() - pd.DateOffset(n=10))]
result.to_csv("shippo_forecast_week.csv", header=False, index=False)
# os.remove('shippo_forecast_week.csv')


# In[ ]:


for m in range(13):    
    current_date = np.datetime64(datetime.datetime.now().strftime("%Y-%m-%d")) + np.timedelta64(m)
    numDays = (pd.to_datetime(current_date) -  pd.to_datetime('1/1/2018'))/np.timedelta64(1, 'D') 
    last_week = np.datetime64(pd.to_datetime(current_date) - pd.DateOffset(days = 7),'D')
    last_2week = np.datetime64(pd.to_datetime(current_date) - pd.DateOffset(days = 14),'D')
    last_3week = np.datetime64(pd.to_datetime(current_date) - pd.DateOffset(days = 21),'D')
    last_month = np.datetime64(pd.to_datetime(current_date) - pd.DateOffset(months = 1),'D')
    dow = pd.to_datetime(current_date).weekday()
    hol = holiday_date(np.datetime64(pd.to_datetime(current_date)))
    new_user = get_new_users_last_month(pd.to_datetime(current_date))
    with open("shippo_forecast_week.csv",'a+') as file:
        for i in range(2):
            temp = pd.to_datetime(current_date)
            se = salesman.iloc[i]['{}-{}'.format(temp.year, temp.month)]
            for j in range(4):
                lag_week = np.nan
                lag_month = np.nan
                temp_month = last_month
                temp_week = last_week
                thresh_hold = np.min(model_data[i].ngay_giao)
                lag_1week = demand_dicts[i].get(pd.to_datetime(last_week).strftime("%m%d%Y") + '{}'.format(j), np.nan)
                lag_2week = demand_dicts[i].get(pd.to_datetime(last_2week).strftime("%m%d%Y") + '{}'.format(j), np.nan)
                lag_3week = demand_dicts[i].get(pd.to_datetime(last_3week).strftime("%m%d%Y") + '{}'.format(j), np.nan)
                ma_week = np.nanmean([lag_1week, lag_2week, lag_3week])
                while pd.to_datetime(temp_week) > thresh_hold and math.isnan(lag_week):
                    try:
                        lag_week = demand_dicts[i][pd.to_datetime(temp_week).strftime("%m%d%Y") + '{}'.format(j)]
                    except:
                        lag_week = np.nan
                    temp_week = np.datetime64(pd.to_datetime(temp_week) - pd.DateOffset(days = 7),'D')
                while pd.to_datetime(temp_month) > thresh_hold and math.isnan(lag_month):
                    try:
                        lag_month = demand_dicts[i][pd.to_datetime(temp_month).strftime("%m%d%Y") + '{}'.format(j)]
                    except:
                        lag_month = np.nan
                    temp_month = np.datetime64(pd.to_datetime(temp_month) - pd.DateOffset(months = 1),'D')

                model = model_dict['{}_{}_{}'.format(locs[i], j, dow)][0]
                inp = np.array([ numDays, lag_week, lag_month, hol, new_user, se]).reshape(1, -1)
                try:
                    res = model.predict(inp)
                except:
                    res = "Not enough information"
                with open("shippo_forecast_week.csv",'a+') as file:
                    try:
                        file.write("{}, {} ca {}, {}, {}\n".format(current_date, locs[i], j, float(res),  model_dict['{}_{}_{}'.format(locs[i], j, dow)][1]))
                    except:
                        pass


# In[ ]:


lastNightSql = 'Select "deliverFromProvinceId", count(id) from public."DeliveryOrder"where "createdTime" + interval \'7 hours\'> Date(now() + interval \'7 hours\' - interval \'1 days\') + interval \'19 hours\'and "createdTime" + interval \'7 hours\'< Date(now() + interval \'7 hours\') + interval \'6 hours\'group by "deliverFromProvinceId"'

actualLastNight = pd.read_sql_query(lastNightSql, connection)
actualLastNight = actualLastNight.replace({9: " HN ca 0", 80: " HCM ca 0"})
actualLastNight['Date'] =  pd.to_datetime(np.datetime64(pd.Timestamp.today().date()))
forecast_week = pd.read_csv('shippo_forecast_week.csv', header = None)
forecast_week[0] = pd.to_datetime(forecast_week[0])
temp = forecast_week.merge(actualLastNight, how='left', left_on=[0, 1],right_on = ['Date', 'deliverFromProvinceId'])

def addActual(row):
    if row[0] == np.datetime64(pd.Timestamp.today().date()):
        if row[1].endswith('0'):
            row['adjusted'] = row['count']
        else:
            row['adjusted'] = row[2]
    else:
        row['adjusted'] = row[2]
    return row
temp = temp.apply(addActual, axis = 1)
temp = temp[[0,1,'adjusted',3]]
temp.to_csv("shippo_forecast_week.csv", header=False, index=False)

