import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('dtp201804-1.csv', encoding='cp1251', sep=';')

# Фильтрация Московской области
df_moscow = df[df['reg_name'].str.contains('Московская')].copy()

# Кодирование категориальных признаков
le = LabelEncoder()
df_moscow.loc[:, 'road_type_encoded'] = le.fit_transform(df_moscow['road_type'])
df_moscow.loc[:, 'crash_type_encoded'] = le.fit_transform(df_moscow['crash_type_name'])
df_moscow.loc[:, 'crash_reason_encoded'] = le.fit_transform(df_moscow['crash_reason'])

# Целевая переменная
df_moscow.loc[:, 'is_dangerous'] = (df_moscow['fatalities_amount'] > 0).astype(int)

# Выбор признаков
features = [
    'road_type_encoded', 'crash_type_encoded', 'crash_reason_encoded',
    'vehicles_amount', 'participants_amount', 'latitude', 'longitude'
]
X = df_moscow[features]
y = df_moscow['is_dangerous']

# Масштабирование и разделение данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))