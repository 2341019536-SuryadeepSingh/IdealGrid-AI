from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parent
DEMAND_FILE=ROOT/'data/raw/hourlyLoadDataIndia.xlsx'
SUPPLY_FILE=ROOT/'data/raw/Daily_Power_Gen_Source_march_23.csv'
MODEL_DIR=ROOT/'models'; MODEL_DIR.mkdir(exist_ok=True)
REGIONS=['Northern','Western','Eastern','Southern','North-Eastern']
FEATURES=['lag_24','lag_168','hour','sin_month','cos_month']

# The source generation dataset is daily energy generation (MU/day), not hourly MW.
# Convert daily MU to an average MW equivalent before comparing it with hourly demand.
# 1 MU = 1,000,000 kWh, so average MW over one day = MU * 1,000,000 / 24 / 1,000.
# Therefore: average MW = MU * 1000 / 24.
DAILY_MU_TO_AVG_MW = 1000.0 / 24.0

def prepare_data():
    d=pd.read_excel(DEMAND_FILE)
    d['datetime']=pd.to_datetime(d['datetime'])
    d=d.sort_values('datetime').set_index('datetime')
    d=d.rename(columns={'Northen Region Hourly Demand':'Northern Region Hourly Demand'})
    d=d.loc['2019-01-01':'2023-12-31 23:00:00']

    s=pd.read_csv(SUPPLY_FILE)
    s['date']=pd.to_datetime(s['date'])
    s=s[s['source'].astype(str).str.strip().str.lower().eq('total')].drop_duplicates('date')
    s=s.set_index('date').sort_index()
    s=s[['NR','WR','SR','ER','NER']].rename(columns={
        'NR':'Northern Region Supply', 'WR':'Western Region Supply',
        'SR':'Southern Region Supply', 'ER':'Eastern Region Supply',
        'NER':'North-Eastern Region Supply'
    })

    # Convert daily generation (MU/day) to average MW equivalent, then carry the
    # daily value across the hours of that day. This is an analytical average,
    # not an instantaneous/dispatchable hourly supply measurement.
    s=s * DAILY_MU_TO_AVG_MW
    supply=s.resample('h').ffill()
    df=d.join(supply,how='inner').dropna()
    df.to_csv(ROOT/'data/processed/grid_hourly.csv',index_label='datetime')
    return df

def main():
    df=prepare_data(); metrics={}
    for region in REGIONS:
        dc=f'{region} Region Hourly Demand'; sc=f'{region} Region Supply'
        r=df[[dc,sc]].copy()
        r['hour']=r.index.hour; r['month']=r.index.month
        r['sin_month']=np.sin(2*np.pi*r['month']/12); r['cos_month']=np.cos(2*np.pi*r['month']/12)
        r['lag_24']=r[dc].shift(24); r['lag_168']=r[dc].shift(168)
        r['gap']=r[sc]-r[dc]; r['utilization']=r[dc]/r[sc].replace(0,np.nan)
        r=r.replace([np.inf,-np.inf],np.nan).dropna()
        split=int(len(r)*0.8); train,test=r.iloc[:split],r.iloc[split:]
        model=XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=6,objective='reg:squarederror',random_state=42,n_jobs=-1)
        model.fit(train[FEATURES],train[dc]); pred=model.predict(test[FEATURES])
        y=test[dc].to_numpy(); mask=np.abs(y)>1e-9
        metrics[region]={
            'MAE_MW':float(mean_absolute_error(test[dc],pred)),
            'RMSE_MW':float(np.sqrt(mean_squared_error(test[dc],pred))),
            'R2':float(r2_score(test[dc],pred)),
            'MAPE_pct':float(np.mean(np.abs((y[mask]-pred[mask])/y[mask]))*100),
            'train_rows':len(train),'test_rows':len(test)
        }
        train_pred=model.predict(train[FEATURES])
        ax=pd.DataFrame({
            'utilization':train['utilization'].to_numpy(),
            'residual':train[dc].to_numpy()-train_pred,
            'gap':train['gap'].to_numpy()
        })
        iso=IsolationForest(n_estimators=200,contamination=0.05,random_state=42).fit(ax)
        scaler=MinMaxScaler().fit(ax)
        joblib.dump(model,MODEL_DIR/f'{region}_model.pkl')
        joblib.dump(iso,MODEL_DIR/f'{region}_isolation.pkl')
        joblib.dump(scaler,MODEL_DIR/f'{region}_scaler.pkl')
    (MODEL_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
