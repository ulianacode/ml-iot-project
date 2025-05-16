import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import geopandas as gpd
from shapely.geometry import Point

df = pd.read_csv('dtp201804-1.csv', encoding='cp1251', sep=';')

df['crash_date'] = df['crash_date'].astype(str)
df['crash_time'] = df['crash_time'].astype(str)
df['crash_datetime'] = pd.to_datetime(df['crash_date'] + ' ' + df['crash_time'])
df['hour'] = df['crash_datetime'].dt.hour
df['day_of_week'] = df['crash_datetime'].dt.dayofweek
df['month'] = df['crash_datetime'].dt.month
le = LabelEncoder()
df['road_type_encoded'] = le.fit_transform(df['road_type'])
df['crash_type_encoded'] = le.fit_transform(df['crash_type_name'])
df['crash_reason_encoded'] = le.fit_transform(df['crash_reason'])

df['is_dangerous'] = np.where(df['fatalities_amount'] > 0, 1, 0)

features = ['road_type_encoded', 'crash_type_encoded', 'crash_reason_encoded',
           'hour', 'day_of_week', 'month', 'vehicles_amount',
           'participants_amount', 'latitude', 'longitude']

X = df[features]
y = df['is_dangerous']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

importances = model.feature_importances_
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.4f}")

geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

df['lat_rounded'] = df['latitude'].round(3)
df['lon_rounded'] = df['longitude'].round(3)

danger_zones = df.groupby(['road_name', 'lat_rounded', 'lon_rounded']).agg({
    'is_dangerous': 'sum',
    'fatalities_amount': 'sum',
    'victims_amount': 'sum',
    'vehicles_amount': 'mean'
}).reset_index()

danger_zones = danger_zones.sort_values('is_dangerous', ascending=False)