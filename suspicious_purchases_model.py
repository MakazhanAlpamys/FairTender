#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import shap

# Настройки для отображения русских символов
plt.rcParams['font.family'] = 'DejaVu Sans'

# Загрузка данных
print("Загрузка обучающих данных...")
train_data = pd.read_csv('final_training_data.csv')

# Загрузка тестового набора данных для прогнозирования
print("Загрузка тестового набора данных...")
test_data = pd.read_csv('final_test_data_user_input.csv')
print(f"Размер тестового набора данных: {test_data.shape}")

# Предварительный анализ данных
print(f"\nРазмер обучающего набора данных: {train_data.shape}")
print("\nРаспределение целевой переменной:")
print(train_data['is_suspicious'].value_counts())
print("\nПримеры первых 5 записей:")
print(train_data.head())

# Создаем словари для категориальных признаков, включая все возможные значения из обоих наборов данных
print("\nПодготовка кодирования категориальных признаков...")
categorical_features = ['category', 'region', 'supplier_id', 'supplier_name']
encoders = {}

for feature in categorical_features:
    # Объединяем уникальные значения из обучающего и тестового наборов
    all_values = pd.concat([train_data[feature], test_data[feature]], axis=0).unique()
    
    # Создаем и обучаем кодировщик для всех уникальных значений
    le = LabelEncoder().fit(all_values.astype(str))
    encoders[feature] = le
    
    # Применяем кодировщик к обучающим данным
    train_data[feature] = le.transform(train_data[feature].astype(str))
    
    # Применяем тот же кодировщик к тестовым данным
    test_data[feature] = le.transform(test_data[feature].astype(str))

# Подготовка данных
# Разделение признаков и целевой переменной
X = train_data.drop('is_suspicious', axis=1)
y = train_data['is_suspicious']

# Определение числовых признаков
numerical_features = [col for col in X.columns if col not in categorical_features]

# Разделение на обучающий и тестовый наборы
print("\nРазделение данных на обучающую и тестовую выборки...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Хранение оригинальных данных для SHAP (в кодированном виде)
X_test_original = X_test.copy()

# Обучение модели XGBoost
print("\nОбучение модели XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric='logloss'
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Оценка модели
print("\nОценка модели на тестовом наборе...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Точность (accuracy): {accuracy:.4f}")
print(f"Точность (precision): {precision:.4f}")
print(f"Полнота (recall): {recall:.4f}")
print(f"F1-мера: {f1:.4f}")

# Построение матрицы ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Не подозрительный', 'Подозрительный'],
            yticklabels=['Не подозрительный', 'Подозрительный'])
plt.xlabel('Предсказанные значения')
plt.ylabel('Фактические значения')
plt.title('Матрица ошибок')
plt.savefig('confusion_matrix.png')

# Анализ важности признаков
print("\nАнализ важности признаков...")
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), X_train.columns[sorted_idx])
plt.title('Важность признаков (Feature Importance)')
plt.savefig('feature_importance.png')

# SHAP анализ для интерпретации
print("\nПроведение SHAP-анализа...")
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Создание графика SHAP для объяснения модели
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP важность признаков')
plt.savefig('shap_importance.png')

plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP распределение влияния признаков')
plt.savefig('shap_summary.png')

# Применение модели к тестовому набору и сохранение предсказаний
print("\nПрогнозирование на тестовом наборе...")
test_predictions = model.predict(test_data)
test_probabilities = model.predict_proba(test_data)[:, 1]

# Создание датафрейма с результатами
results = pd.DataFrame({
    'id': test_data.index,
    'is_suspicious_pred': test_predictions,
    'is_suspicious_prob': test_probabilities
})

# Сохранение результатов
print("\nСохранение результатов прогнозирования...")
results.to_csv('predictions.csv', index=False)

print("\nАнализ завершен. Результаты сохранены в файле predictions.csv")
print("Визуализации сохранены в файлах confusion_matrix.png, feature_importance.png, shap_importance.png и shap_summary.png") 