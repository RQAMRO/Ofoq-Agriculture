# ملف: pipeline.py
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("🚀 بدء تشغيل خط تجميع بيانات أفق (Ofoq) تلقائياً...")

# 1. حساب التواريخ ديناميكياً (آخر 122 يوم)
end_date = datetime.now()
start_date = end_date - timedelta(days=122)
start_str = start_date.strftime('%Y%m%d')
end_str = end_date.strftime('%Y%m%d')

# 2. جلب البيانات من NASA POWER
url = "https://power.larc.nasa.gov/api/temporal/daily/point"
params = {
    'parameters': 'T2M_MAX,T2M_MIN,RH2M,WS10M,T2M,GWETTOP,GWETROOT,PRECTOTCORR',
    'community': 'RE',
    'longitude': 35.39,
    'latitude': 32.2438,
    'start': start_str,
    'end': end_str,
    'format': 'JSON'
}

try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    new_data = pd.DataFrame(data['properties']['parameter']).reset_index().rename(columns={'index': 'DATE'})
    new_data['DATE'] = pd.to_datetime(new_data['DATE'])
    new_data['SEQ'] = range(1, len(new_data) + 1)
    new_data['YEAR'] = new_data['DATE'].dt.year
    new_data['DOY'] = new_data['DATE'].dt.dayofyear
    print(f"✅ تم جلب {len(new_data)} يوم من ناسا.")
except Exception as e:
    print(f"❌ فشل جلب البيانات: {e}")
    new_data = None

# 3. دالة التنظيف والمعادلات
def clean_and_engineer(df):
    df = df.copy()
    df = df.replace(-999, np.nan)
    numeric_cols = ['T2M_MAX', 'T2M_MIN', 'RH2M', 'WS10M', 'T2M', 'GWETTOP', 'GWETROOT', 'PRECTOTCORR']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'SEQ' not in df.columns:
        df['SEQ'] = range(1, len(df) + 1)
        
    df['Diurnal_Temp_Range'] = df['T2M_MAX'] - df['T2M_MIN']
    df['Diurnal_Verification'] = np.where(df['Diurnal_Temp_Range'] > 0, 'VALID', 'INVALID')
    df['Cumulative_Rainfall'] = df['PRECTOTCORR'].cumsum()
    df['Rain_Verification'] = np.where(df['PRECTOTCORR'] >= 0, 'VALID', 'INVALID')
    
    def get_soil_status(gwetroot):
        if pd.isna(gwetroot): return "Unknown"
        elif gwetroot < 0.19: return "Very dry"
        elif gwetroot < 0.47: return "Ideal for agriculture"
        else: return "Saturated"
    df['Soil_Status'] = df['GWETROOT'].apply(get_soil_status)
    
    def get_recommendation(row):
        try: month = pd.to_datetime(row['DATE']).month
        except: month = 11
        soil = row['Soil_Status']
        tmax = row['T2M_MAX']
        if soil == "Saturated": return "🚫 التربة مشبعة - إيقاف الري فوراً"
        elif tmax > 35 and soil == "Very dry": return "⚠️ إجهاد حراري - تأجيل الزراعة"
        elif soil == "Ideal for agriculture":
            return "✅ ظروف مثالية - ازرع البقوليات" if month in [10, 11, 12, 1, 2] else "🌱 ظروف جيدة - متابعة المحاصيل"
        else: return "⏳ انتظر - الظروف غير مناسبة"
    df['Recommendation'] = df.apply(get_recommendation, axis=1)
    
    def get_fertilizer_savings(row):
        rec = str(row['Recommendation'])
        soil = row['Soil_Status']
        has_legumes = any(word in rec for word in ['بقوليات', 'فول', 'حمص', 'عدس'])
        return 0.35 if (has_legumes and soil == 'Ideal for agriculture') else 0.0
    df['Fertilizer_Savings_%'] = df.apply(get_fertilizer_savings, axis=1)
    return df

# 4. المعالجة والدمج
if new_data is not None:
    merged_df = clean_and_engineer(new_data)
else:
    print("⚠️ لا توجد بيانات جديدة، يتم إيقاف التشغيل.")
    exit()

# 5. التنبؤ بـ 7 أيام قادمة
last_14_days = merged_df.tail(14).copy()
avg_tmax, avg_tmin, avg_rain = last_14_days['T2M_MAX'].mean(), last_14_days['T2M_MIN'].mean(), last_14_days['PRECTOTCORR'].mean()
avg_gwetroot, avg_rh, avg_wind = last_14_days['GWETROOT'].mean(), last_14_days['RH2M'].mean(), last_14_days['WS10M'].mean()
recent_avg, older_avg = last_14_days['T2M_MAX'].tail(7).mean(), last_14_days['T2M_MAX'].head(7).mean()
trend_factor = recent_avg / older_avg if older_avg > 0 else 1.0

forecasts = []
last_date = last_14_days['DATE'].max()
last_seq = int(last_14_days['SEQ'].max())
last_cum_rain = last_14_days['Cumulative_Rainfall'].iloc[-1]

for day in range(1, 8):
    future_date = last_date + timedelta(days=day)
    noise = np.random.uniform(0.95, 1.05)
    forecast_tmax = round(avg_tmax * trend_factor * noise, 2)
    forecast_tmin = round(avg_tmin * trend_factor * noise, 2)
    forecast_rain = round(max(0, avg_rain * noise), 2)
    
    et_daily = 0.0023 * ((forecast_tmax + forecast_tmin)/2 + 17.8) * np.sqrt(max(forecast_tmax - forecast_tmin, 1)) * 15
    forecast_gwetroot = round(max(0.10, min(0.60, avg_gwetroot + (forecast_rain/100) - (et_daily/100))), 3)
    last_cum_rain += forecast_rain
    
    if forecast_gwetroot >= 0.47: soil_status = "Saturated"
    elif forecast_gwetroot >= 0.19: soil_status = "Ideal for agriculture"
    else: soil_status = "Very dry"
    
    month = future_date.month
    if soil_status == "Saturated": rec = "🚫 التربة مشبعة - إيقاف الري فوراً"
    elif forecast_tmax > 35 and soil_status == "Very dry": rec = "⚠️ إجهاد حراري - تأجيل الزراعة"
    elif soil_status == "Ideal for agriculture": rec = "✅ ظروف مثالية - ازرع البقوليات" if month in [10, 11, 12, 1, 2] else "🌱 ظروف جيدة - متابعة المحاصيل"
    else: rec = "⏳ انتظر - الظروف غير مناسبة"
    
    forecasts.append({
        'SEQ': last_seq + day, 'YEAR': future_date.year, 'DATE': future_date,
        'DOY': future_date.timetuple().tm_yday, 'T2M_MAX': forecast_tmax, 'T2M_MIN': forecast_tmin,
        'RH2M': round(avg_rh * noise, 2), 'WS10M': round(avg_wind * noise, 2),
        'T2M': round((forecast_tmax + forecast_tmin)/2, 2), 'GWETTOP': round(avg_gwetroot * 0.2, 3),
        'GWETROOT': forecast_gwetroot, 'PRECTOTCORR': forecast_rain,
        'Diurnal_Temp_Range': round(forecast_tmax - forecast_tmin, 2),
        'Diurnal_Verification': 'FORECAST', 'Soil_Status': soil_status,
        'Rain_Verification': 'FORECAST', 'Cumulative_Rainfall': round(last_cum_rain, 2),
        'Recommendation': rec,
        'Fertilizer_Savings_%': 0.35 if ('بقوليات' in rec and soil_status == 'Ideal for agriculture') else 0.0,
        'is_forecast': True
    })

forecast_df = pd.DataFrame(forecasts)
merged_df['is_forecast'] = False
full_df = pd.concat([merged_df, forecast_df], ignore_index=True)
print(f"✅ تم دمج {len(merged_df)} يوم تاريخي + {len(forecast_df)} يوم توقع.")

# 6. التصدير والحفظ في مجلد data
os.makedirs('data', exist_ok=True)

# حفظ JSON
export_df = full_df.copy()
export_df['DATE'] = export_df['DATE'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
json_data = {
    "metadata": {
        "location": "Al-Nasariya, Nablus, Palestine",
        "coordinates": {"latitude": 32.2438, "longitude": 35.39},
        "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "source": "NASA POWER + Ofoq Framework",
        "total_days": len(export_df)
    },
    "daily_data": export_df.to_dict(orient='records')
}
with open('data/ofoq_data.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

# حفظ Excel
full_df.to_excel('data/ofoq_data.xlsx', index=False, engine='openpyxl')
print("💾 تم حفظ البيانات المحدثة في مجلد data/")
