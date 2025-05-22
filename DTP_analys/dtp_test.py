import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample
import folium

class MyRandomForestClassifier:
    def __init__(self, n_estimators=50, max_features='sqrt', max_depth=5):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.trees = []
        self.features_indices = []
        self.oob_indices = []

    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_features

    def _balanced_bootstrap(self, X, y):
        X_0 = X[y == 0]
        y_0 = y[y == 0]
        X_1 = X[y == 1]
        y_1 = y[y == 1]

        n_samples = min(len(y_0), len(y_1))

        X_0_boot, y_0_boot = resample(X_0, y_0, n_samples=n_samples, replace=True)
        X_1_boot, y_1_boot = resample(X_1, y_1, n_samples=n_samples, replace=True)

        X_boot = np.vstack([X_0_boot, X_1_boot])
        y_boot = np.concatenate([y_0_boot, y_1_boot])

        idx = np.random.permutation(len(y_boot))
        return X_boot[idx], y_boot[idx]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)

        self.trees = []
        self.features_indices = []
        self.oob_indices = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._balanced_bootstrap(X, y)

            features_idx = np.random.choice(n_features, max_features, replace=False)

            # Вычисляем OOB
            all_indices = set(range(len(X)))
            sample_indices = set([tuple(x) for x in X_sample])
            oob_idx = [i for i in range(len(X)) if tuple(X[i]) not in sample_indices]

            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=5)
            tree.fit(X_sample[:, features_idx], y_sample)

            self.trees.append(tree)
            self.features_indices.append(features_idx)
            self.oob_indices.append(oob_idx)

    def predict(self, X):
        tree_preds = []
        for tree, feat_idx in zip(self.trees, self.features_indices):
            preds = tree.predict(X[:, feat_idx])
            tree_preds.append(preds)

        tree_preds = np.array(tree_preds)
        majority_votes = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)
        return majority_votes.astype(int)

    def oob_score(self, X, y):
        n_samples = X.shape[0]
        oob_votes = [[] for _ in range(n_samples)]

        for tree, feat_idx, oob_idx in zip(self.trees, self.features_indices, self.oob_indices):
            for i in oob_idx:
                pred = tree.predict(X[i, feat_idx].reshape(1, -1))[0]
                oob_votes[i].append(pred)

        final_preds = []
        true_y = []

        for i in range(n_samples):
            if oob_votes[i]:
                vote = Counter(oob_votes[i]).most_common(1)[0][0]
                final_preds.append(vote)
                true_y.append(y[i])

        if not final_preds:
            return 0.0

        return np.mean(np.array(final_preds) == np.array(true_y))

def visualize_tree(model, feature_names, tree_index=0):
    plt.figure(figsize=(20, 10))
    plot_tree(model.trees[tree_index],
              feature_names=[feature_names[i] for i in model.features_indices[tree_index]],
              filled=True,
              rounded=True,
              class_names=['Not Dangerous', 'Dangerous'])
    plt.title(f"Визуализация дерева №{tree_index + 1}")
    plt.show()

# Загрузка и предварительная обработка данных
df = pd.read_csv('dtp201804-1.csv', encoding='cp1251', sep=';')
df_moscow = df[df['reg_name'].str.contains('Московская')].copy()

# Кодирование категориальных признаков
le = LabelEncoder()
df_moscow['road_type_encoded'] = le.fit_transform(df_moscow['road_type'])
df_moscow['crash_type_encoded'] = le.fit_transform(df_moscow['crash_type_name'])
df_moscow['crash_reason_encoded'] = le.fit_transform(df_moscow['crash_reason'])

# Группировка данных по участкам дорог
df_moscow['lat_rounded'] = df_moscow['latitude'].round(2)
df_moscow['lon_rounded'] = df_moscow['longitude'].round(2)
road_segments = df_moscow.groupby(['road_type_encoded', 'lat_rounded', 'lon_rounded'])

# Создание DataFrame с агрегированными данными по участкам дорог
road_stats = road_segments.agg({
    'fatalities_amount': 'sum',
    'participants_amount': 'sum',
    'vehicles_amount': 'sum',
    'crash_type_encoded': lambda x: Counter(x).most_common(1)[0][0],
    'crash_reason_encoded': lambda x: Counter(x).most_common(1)[0][0]
}).reset_index()

# Добавление количества ДТП на участке
road_stats['crash_count'] = road_segments.size().values

# Определение опасности дороги
road_stats['is_dangerous_road'] = (
    (road_stats['fatalities_amount'] > 0) |
    (road_stats['crash_count'] > 3)
).astype(int)

# Подготовка признаков и целевой переменной
features_road = [
    'road_type_encoded', 'crash_type_encoded', 'crash_reason_encoded',
    'vehicles_amount', 'participants_amount', 'crash_count', 'lat_rounded', 'lon_rounded'
]

X_road = road_stats[features_road].values
y_road = road_stats['is_dangerous_road'].values


scaler = StandardScaler()
X_road_scaled = scaler.fit_transform(X_road)
X_train_road, X_test_road, y_train_road, y_test_road = train_test_split(
    X_road_scaled, y_road, test_size=0.3, random_state=42
)


forest_road = MyRandomForestClassifier(n_estimators=50, max_depth=5)
forest_road.fit(X_train_road, y_train_road)
y_pred_road = forest_road.predict(X_test_road)


print("=== Classification Report (Road Danger) ===")
print(classification_report(y_test_road, y_pred_road, zero_division=1))

print("=== OOB Score ===")
print("OOB Accuracy:", forest_road.oob_score(X_train_road, y_train_road))

visualize_tree(forest_road, features_road)

# Визуализация опасных участков на карте
dangerous_roads = road_stats[road_stats['is_dangerous_road'] == 1]
moscow_center = [df_moscow['latitude'].mean(), df_moscow['longitude'].mean()]

map_roads = folium.Map(location=moscow_center, zoom_start=11)
for _, row in dangerous_roads.iterrows():
    folium.CircleMarker(
        location=[row['lat_rounded'], row['lon_rounded']],
        radius=5 + row['crash_count']/2,
        color='red',
        fill=True,
        popup=f"""
        Участок дороги:
        Тип: {row['road_type_encoded']}
        ДТП: {row['crash_count']}
        Погибших: {row['fatalities_amount']}
        Участников: {row['participants_amount']}
        """
    ).add_to(map_roads)

map_roads.save("dangerous_roads_moscow.html")
print("Карта опасных участков сохранена как 'dangerous_roads_moscow.html'")
