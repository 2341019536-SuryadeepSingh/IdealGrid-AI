from pathlib import Path
import numpy as np, pandas as pd, joblib
ROOT=Path(__file__).resolve().parents[1]; MODEL_DIR=ROOT/'models'
REGIONS=['Northern','Western','Eastern','Southern','North-Eastern']
FEATURES=['lag_24','lag_168','hour','sin_month','cos_month']

def make_features(lag_24,lag_168,timestamp):
    ts=pd.Timestamp(timestamp)
    return pd.DataFrame([{
        'lag_24':float(lag_24),
        'lag_168':float(lag_168),
        'hour':ts.hour,
        'sin_month':np.sin(2*np.pi*ts.month/12),
        'cos_month':np.cos(2*np.pi*ts.month/12)
    }],columns=FEATURES)

def predict(region,lag_24,lag_168,timestamp):
    model=joblib.load(MODEL_DIR/f'{region}_model.pkl')
    return float(model.predict(make_features(lag_24,lag_168,timestamp))[0])

def analyze_region(region,lag_24,lag_168,avg_generation_mw,timestamp,actual=None):
    predicted=predict(region,lag_24,lag_168,timestamp)
    avg_generation_mw=float(avg_generation_mw)
    gap=avg_generation_mw-predicted
    coverage=predicted/avg_generation_mw if avg_generation_mw>0 else np.nan

    if gap<0:
        status='CRITICAL'
        rec='Forecast demand is above the daily-generation average equivalent. Review reserve capacity and demand-response options.'
    elif coverage>=.90:
        status='WARNING'
        rec='Demand is close to the generation average equivalent. Maintain reserve margin and monitor the region closely.'
    else:
        status='HEALTHY'
        rec='Forecast demand is below the daily-generation average equivalent for the selected hour.'

    out={
        'region':region,
        'predicted_demand':predicted,
        'avg_generation_mw':avg_generation_mw,
        'gap':gap,
        'coverage':coverage*100,
        'status':status,
        'recommendation':rec
    }

    if actual is not None and str(actual).strip()!='':
        actual=float(actual)
        residual=actual-predicted
        x=pd.DataFrame([{'utilization':coverage,'residual':residual,'gap':gap}])
        iso=joblib.load(MODEL_DIR/f'{region}_isolation.pkl')
        scaler=joblib.load(MODEL_DIR/f'{region}_scaler.pkl')
        anomaly=bool(iso.predict(x)[0]==-1)
        z=scaler.transform(x)[0]
        score=(1-z[0])*.35+abs(z[1])*.30+abs(z[2])*.20+anomaly*.15
        out.update({
            'actual_demand':actual,
            'residual':residual,
            'anomaly':anomaly,
            'inefficiency_score':float(np.clip(score,0,1)*100)
        })
    return out

def optimize_transfers(results):
    surplus=[{'region':r['region'],'amount':float(r['gap'])} for r in results if r['gap']>0]
    deficit=[{'region':r['region'],'amount':float(-r['gap'])} for r in results if r['gap']<0]
    surplus.sort(key=lambda x:x['amount'],reverse=True); deficit.sort(key=lambda x:x['amount'],reverse=True)
    transfers=[]; i=j=0
    while i<len(surplus) and j<len(deficit):
        amount=min(surplus[i]['amount'],deficit[j]['amount'])
        if amount>0:
            transfers.append({'from':surplus[i]['region'],'to':deficit[j]['region'],'mw':float(amount)})
        surplus[i]['amount']-=amount; deficit[j]['amount']-=amount
        if surplus[i]['amount']<=1e-6: i+=1
        if deficit[j]['amount']<=1e-6: j+=1
    return transfers,sum(x['amount'] for x in surplus[i:]),sum(x['amount'] for x in deficit[j:])
