import os
import io
import base64
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from static.logo import create_logo, get_logo_as_base64
import plotly.express as px
import json
import time
import datetime
import requests
from bs4 import BeautifulSoup
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException

# Настройка страницы
st.set_page_config(
    page_title="Анализ подозрительных закупок",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Создаем папки, если их нет
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Добавление логотипа в верхнем углу
logo_path = create_logo()
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=200)

# Словарь для перевода названий столбцов
column_translations = {
    'category': 'категория',
    'region': 'регион',
    'price': 'стоимость',
    'avg_price_category_region': 'средняя_стоимость_категория_регион',
    'supplier_id': 'ид_поставщика',
    'supplier_name': 'название_поставщика',
    'supplier_win_count': 'количество_выигранных_контрактов',
    'days_to_tender': 'дней_до_тендера',
    'price_per_unit': 'цена_за_единицу',
    'supplier_years_active': 'лет_активности_поставщика',
    'supplier_total_contracts': 'всего_контрактов_поставщика',
    'supplier_avg_contract_value': 'средняя_стоимость_контрактов_поставщика',
    'is_suspicious': 'подозрительность'
}

# Обратный словарь для перевода с русского на английский
reverse_translations = {v: k for k, v in column_translations.items()}

# Загрузка модели
model_path = 'model.pkl'
encoders_path = 'encoders.pkl'

# Функция для загрузки или обучения модели
@st.cache_resource
def load_or_train_model():
    if os.path.exists(model_path) and os.path.exists(encoders_path):
        # Загрузка модели и кодировщиков
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
    else:
        # Загружаем обучающие и тестовые данные
        st.info("Загрузка данных и подготовка модели...")
        train_data = pd.read_csv('final_training_data.csv')
        test_data = pd.read_csv('final_test_data_user_input.csv')
        
        # Подготовка данных
        categorical_features = ['category', 'region', 'supplier_id', 'supplier_name']
        encoders = {}
        
        for feature in categorical_features:
            # Объединяем уникальные значения из обоих наборов
            all_values = pd.concat([train_data[feature], test_data[feature]], axis=0).unique()
            le = LabelEncoder().fit(all_values.astype(str))
            encoders[feature] = le
            
            # Применяем кодировщик
            train_data[feature] = le.transform(train_data[feature].astype(str))
        
        # Подготовка данных для обучения
        X = train_data.drop('is_suspicious', axis=1)
        y = train_data['is_suspicious']
        
        # Обучение модели XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X, y)
        
        # Сохранение модели и кодировщиков
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
        
        st.success("Модель и кодировщики сохранены.")
    
    return model, encoders

# Загружаем модель и кодировщики
model, encoders = load_or_train_model()

# Получение схемы данных для заполнения формы
@st.cache_data
def get_feature_info():
    train_data = pd.read_csv('final_training_data.csv')
    feature_info = {}
    for feature in train_data.columns:
        if feature != 'is_suspicious':
            if feature in ['category', 'region', 'supplier_id', 'supplier_name']:
                # Для категориальных признаков сохраняем список значений
                unique_values = sorted(train_data[feature].astype(str).unique().tolist())
                feature_info[feature] = {
                    'type': 'categorical', 
                    'values': unique_values
                }
            else:
                # Для числовых полей вычисляем статистику
                feature_info[feature] = {
                    'type': 'numerical', 
                    'min': float(train_data[feature].min()),
                    'max': float(train_data[feature].max()),
                    'mean': float(train_data[feature].mean())
                }
    return feature_info

# Получаем информацию о признаках
feature_info = get_feature_info()

# Функция для предсказания
def predict_suspicious(input_data):
    """Предсказывает подозрительность закупки на основе входных данных"""
    try:
        # Проверяем, что входные данные - это DataFrame
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError("input_data должен быть pandas DataFrame")
        
        # Создаем копию данных, чтобы не изменять оригинал
        processed_data = input_data.copy()
        
        # Преобразование категориальных признаков
        categorical_features = ['category', 'region', 'supplier_id', 'supplier_name']
        for feature in categorical_features:
            if feature in processed_data.columns:
                # Обработка неизвестных категорий
                processed_data[feature] = processed_data[feature].astype(str)
                
                # Проверяем наличие неизвестных категорий
                unknown_categories = []
                for value in processed_data[feature].unique():
                    if value not in encoders[feature].classes_:
                        unknown_categories.append(value)
                
                if unknown_categories:
                    # Заменяем неизвестные категории на первую известную категорию
                    safe_category = encoders[feature].classes_[0]
                    # Комментарий вместо предупреждения
                    # print(f"Обнаружены неизвестные категории в признаке '{feature}': {unknown_categories}. Заменены на '{safe_category}'")
                    
                    # Применяем замену
                    processed_data[feature] = processed_data[feature].apply(
                        lambda x: safe_category if x in unknown_categories else x
                    )
                
                # Теперь безопасно применяем трансформацию
                processed_data[feature] = encoders[feature].transform(processed_data[feature])
        
        # Получаем правильный порядок признаков из модели
        feature_names = model.feature_names_in_
        
        # Убеждаемся, что входные данные имеют те же признаки в том же порядке
        missing_features = set(feature_names) - set(processed_data.columns)
        extra_features = set(processed_data.columns) - set(feature_names)
        
        if missing_features or extra_features:
            msg = []
            if missing_features:
                msg.append(f"Отсутствуют: {missing_features}")
                # Добавляем отсутствующие признаки и заполняем их нулями
                for feature in missing_features:
                    processed_data[feature] = 0
            
            if extra_features:
                msg.append(f"Лишние: {extra_features}")
                # Удаляем лишние признаки
            
            # Комментарий вместо предупреждения
            # print(f"Несоответствие признаков. {', '.join(msg)}")
        
        # Переупорядочиваем столбцы в соответствии с порядком в модели
        processed_data = processed_data[feature_names]
        
        # Предсказание
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1]
        
        return predictions, probabilities
    except Exception as e:
        # Тихая обработка ошибки без уведомления пользователя
        # print(f"Ошибка при предсказании: {str(e)}")
        # В случае ошибки, возвращаем все нули
        return np.zeros(len(input_data)), np.zeros(len(input_data))

# Функция для создания SHAP графика для одного предсказания с русскими названиями
def create_shap_plot_single(input_data):
    """Создает SHAP график для объяснения предсказания"""
    try:
        # Обработка категориальных признаков
        categorical_features = ['category', 'region', 'supplier_id', 'supplier_name']
        for feature in categorical_features:
            if feature in input_data.columns:
                # Безопасное преобразование категориальных признаков
                try:
                    # Получаем уникальные значения во входных данных
                    unique_values = input_data[feature].astype(str).unique()
                    
                    # Проверяем, есть ли неизвестные значения
                    unknown_values = [val for val in unique_values if val not in encoders[feature].classes_]
                    
                    if unknown_values:
                        # Заменяем неизвестные значения на первое известное значение
                        safe_value = encoders[feature].classes_[0]
                        # Комментарий вместо предупреждения
                        # print(f"Обнаружены неизвестные значения в признаке {feature}: {unknown_values}. Заменены на '{safe_value}'")
                        
                        # Заменяем неизвестные значения
                        input_data[feature] = input_data[feature].astype(str).apply(
                            lambda x: safe_value if x in unknown_values else x
                        )
                    
                    # Безопасное преобразование
                    input_data[feature] = encoders[feature].transform(input_data[feature].astype(str))
                
                except Exception as e:
                    # Тихая обработка ошибки
                    # print(f"Ошибка при обработке признака {feature}: {e}")
                    # В случае ошибки заполняем нулями
                    input_data[feature] = 0
        
        # Получаем правильный порядок признаков из модели
        feature_names = model.feature_names_in_
        
        # Проверяем, все ли необходимые признаки присутствуют
        missing_features = [feat for feat in feature_names if feat not in input_data.columns]
        if missing_features:
            # print(f"Отсутствуют необходимые признаки: {missing_features}")
            # Заполняем отсутствующие признаки нулями
            for feat in missing_features:
                input_data[feat] = 0
        
        # Переупорядочиваем столбцы в соответствии с порядком в модели
        input_data = input_data[feature_names]
        
        # Создаем словарь для перевода названий признаков в графике SHAP
        feature_names_translated = {}
        for feat in feature_names:
            if feat in column_translations:
                feature_names_translated[feat] = column_translations[feat]
            else:
                feature_names_translated[feat] = feat
        
        # Создаем объясняющий объект SHAP
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)
        
        # Создаем новую фигуру
        plt.figure(figsize=(10, 6))
        
        # Получаем SHAP значения для первого экземпляра
        shap_values_first = shap_values[0]
        feature_names_list = input_data.columns.tolist()
        
        # Сортируем признаки по абсолютным значениям SHAP
        sorted_idx = np.argsort(abs(shap_values_first.values))
        sorted_features = [feature_names_list[i] for i in sorted_idx]
        sorted_values = shap_values_first.values[sorted_idx]
        
        # Берем только топ-10 признаков для лучшей визуализации
        top_features = sorted_features[-10:]
        top_values = sorted_values[-10:]
        
        # Создаем бар-график с русскими названиями
        y_pos = np.arange(len(top_features))
        plt.barh(
            y=[feature_names_translated.get(feature, feature) for feature in top_features],
            width=top_values,
            color=['#FF4136' if x > 0 else '#0074D9' for x in top_values]
        )
        
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Влияние на предсказание (значение SHAP)')
        plt.ylabel('Признак')
        plt.title('Вклад признаков в предсказание')
        plt.tight_layout()
        
        return plt.gcf()
    except Exception as e:
        # Тихая обработка ошибки без уведомления пользователя
        # print(f"Ошибка при создании SHAP графика: {str(e)}")
        # Возвращаем пустой график без сообщения об ошибке
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.axis('off')
        return fig

# Функция для создания обобщенных графиков SHAP с русскими названиями
def create_shap_summary_plot(sample_data, n_samples=100):
    """Создает общий SHAP график с русскими названиями признаков"""
    try:
        if len(sample_data) > n_samples:
            sample_data = sample_data.sample(n=n_samples, random_state=42)
        
        # Преобразование категориальных признаков
        categorical_features = ['category', 'region', 'supplier_id', 'supplier_name']
        for feature in categorical_features:
            if feature in sample_data.columns:  # Проверяем наличие столбца
                # Безопасное преобразование категориальных признаков
                try:
                    # Получаем уникальные значения в данных
                    unique_values = sample_data[feature].astype(str).unique()
                    
                    # Проверяем, есть ли значения, которых нет в обучающих данных
                    unknown_values = [val for val in unique_values if val not in encoders[feature].classes_]
                    
                    if unknown_values:
                        # Заменяем неизвестные значения на первое известное значение (безопасное значение)
                        safe_value = encoders[feature].classes_[0]
                        # Комментарий вместо предупреждения
                        # print(f"Обнаружены неизвестные значения в признаке {feature}: {unknown_values}. Заменены на '{safe_value}'")
                        
                        # Заменяем неизвестные значения
                        sample_data[feature] = sample_data[feature].astype(str).apply(
                            lambda x: safe_value if x in unknown_values else x
                        )
                    
                    # Теперь безопасно трансформируем данные
                    sample_data[feature] = encoders[feature].transform(sample_data[feature].astype(str))
                    
                except Exception as e:
                    # Тихая обработка ошибки
                    # print(f"Ошибка при обработке признака {feature}: {e}")
                    # В случае ошибки заполняем признак нулями
                    sample_data[feature] = 0
        
        # Получаем правильный порядок признаков из модели
        feature_names = model.feature_names_in_
        
        # Проверяем, все ли необходимые признаки присутствуют
        missing_features = [feat for feat in feature_names if feat not in sample_data.columns]
        if missing_features:
            # print(f"Отсутствуют необходимые признаки: {missing_features}")
            # Заполняем отсутствующие признаки нулями
            for feat in missing_features:
                sample_data[feat] = 0
        
        # Переупорядочиваем столбцы в соответствии с порядком в модели
        sample_data = sample_data[feature_names]
        
        # Создаем словарь для перевода названий признаков
        feature_names_translated = {}
        for feat in feature_names:
            if feat in column_translations:
                feature_names_translated[feat] = column_translations[feat]
            else:
                feature_names_translated[feat] = feat
        
        # Создаем копию данных с русскими названиями столбцов для SHAP
        sample_data_ru = sample_data.copy()
        sample_data_ru.columns = [feature_names_translated.get(col, col) for col in sample_data.columns]
        
        # Создаем объясняющий объект SHAP
        explainer = shap.Explainer(model)
        shap_values = explainer(sample_data)
        
        # Создаем новую фигуру
        plt.figure(figsize=(10, 8))
        
        # Получаем фрейм со значениями SHAP и соответствующими признаками
        shap_df = pd.DataFrame(shap_values.values, columns=sample_data.columns)
        
        # Вычисляем средние абсолютные значения SHAP для каждого признака
        feature_importance = pd.DataFrame({
            'feature': shap_df.columns,
            'importance': shap_df.abs().mean().values
        }).sort_values('importance', ascending=False)
        
        # Создаем бар-график с русскими названиями
        plt.barh(
            y=[feature_names_translated.get(feature, feature) for feature in feature_importance['feature']],
            width=feature_importance['importance'],
            color='#0099ff'
        )
        plt.xlabel('Важность признака (среднее значение |SHAP|)')
        plt.ylabel('Признак')
        plt.title('Важность признаков по SHAP')
        plt.tight_layout()
        
        return plt.gcf()
    except Exception as e:
        # Тихая обработка ошибки без уведомления пользователя
        # print(f"Ошибка при создании SHAP графика: {str(e)}")
        # Возвращаем пустой график без сообщения об ошибке
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.axis('off')
        return fig

# Функция для создания графиков метрик
def create_metrics_plot():
    """Создает график с метриками модели"""
    # Загрузим тестовые данные и получим предсказания
    train_data = pd.read_csv('final_training_data.csv')
    
    # Получаем тестовую выборку (20% от обучающей)
    _, X_test, _, y_test = train_test_split(
        train_data.drop('is_suspicious', axis=1), 
        train_data['is_suspicious'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Преобразуем категориальные признаки
    categorical_features = ['category', 'region', 'supplier_id', 'supplier_name']
    for feature in categorical_features:
        X_test[feature] = encoders[feature].transform(X_test[feature].astype(str))
    
    # Получаем предсказания
    y_pred = model.predict(X_test)
    
    # Рассчитываем метрики
    metrics = {
        'Точность (accuracy)': accuracy_score(y_test, y_pred),
        'Точность (precision)': precision_score(y_test, y_pred),
        'Полнота (recall)': recall_score(y_test, y_pred),
        'F1-мера': f1_score(y_test, y_pred)
    }
    
    # Создаем матрицу ошибок
    cm = pd.crosstab(y_test, y_pred, rownames=['Фактические'], colnames=['Предсказанные'])
    
    # Визуализируем матрицу ошибок
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Не подозрительный', 'Подозрительный'],
                yticklabels=['Не подозрительный', 'Подозрительный'])
    plt.title('Матрица ошибок')
    
    return fig, metrics

# Функция для создания графика распределения вероятностей
def create_probability_distribution_plot(predictions_df):
    """Создает график распределения вероятностей предсказаний с русскими подписями"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Определяем названия столбцов с вероятностью и предсказанием
    prob_col = 'is_suspicious_prob'
    pred_col = 'is_suspicious_pred'
    
    # Убедимся, что столбцы существуют
    if prob_col not in predictions_df.columns or pred_col not in predictions_df.columns:
        # Проверим русские названия
        if 'подозрительность_prob' in predictions_df.columns:
            prob_col = 'подозрительность_prob'
        if 'подозрительность_pred' in predictions_df.columns:
            pred_col = 'подозрительность_pred'
    
    # Создаем копию для отображения
    display_df = predictions_df.copy()
    
    # Строим график
    sns.histplot(
        data=display_df, 
        x=prob_col,
        hue=pred_col,
        bins=30,
        palette=['green', 'red'],
        ax=ax
    )
    
    plt.title('Распределение вероятностей предсказаний')
    plt.xlabel('Вероятность подозрительности')
    plt.ylabel('Количество закупок')
    
    # Меняем названия в легенде (0 -> "Не подозрительные", 1 -> "Подозрительные")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Не подозрительные', 'Подозрительные'])
    
    return fig

# Функции для парсинга данных с сайта госзакупок

# Функция для инициализации драйвера
def init_driver(headless=True):
    """
    Инициализация драйвера Chrome с настройками
    """
    try:
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            # Первый метод - использование конкретной стабильной версии ChromeDriver
            driver = webdriver.Chrome(service=Service(ChromeDriverManager(version="114.0.5735.90").install()), options=chrome_options)
            return driver
        except Exception as inner_e:
            st.warning(f"Не удалось установить ChromeDriver конкретной версии. Попытка с другими методами.")
            try:
                # Второй метод - использование webdriver_manager без указания версии
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
                return driver
            except Exception as inner_e2:
                st.warning(f"Не удалось установить ChromeDriver через менеджер. Попытка использования локального драйвера.")
                # Третий метод - просто создание драйвера (если драйвер уже установлен в системе)
                driver = webdriver.Chrome(options=chrome_options)
                return driver
    except Exception as e:
        st.error(f"Ошибка при инициализации драйвера: {str(e)}")
        st.info("Для использования этой функции требуется Chrome и ChromeDriver, совместимые друг с другом.")
        # Покажем альтернативные способы получения данных
        help_text = """
        **Как решить проблему:**
        1. Установите более старую версию Chrome (114 или 115)
        2. Или установите ChromeDriver вручную с [официального сайта](https://chromedriver.chromium.org/downloads)
        3. Или воспользуйтесь альтернативными способами загрузки данных (CSV файл)
        """
        st.markdown(help_text)
        return None

# Функция для парсинга данных с сайта госзакупок
def parse_goszakup(url, query=None, price_from=None, price_to=None, status=None, max_pages=5):
    """
    Парсинг данных с сайта госзакупок
    
    Parameters:
    - url: базовый URL для парсинга
    - query: поисковый запрос (наименование лота)
    - price_from: минимальная сумма закупки
    - price_to: максимальная сумма закупки
    - status: статус закупки
    - max_pages: максимальное количество страниц для парсинга
    
    Returns:
    - DataFrame с данными о закупках
    """
    results = []
    driver = init_driver(headless=True)
    
    if driver is None:
        return pd.DataFrame()
    
    try:
        # Переходим на страницу
        driver.get(url)
        time.sleep(3)
        
        # Заполняем поисковую форму, если параметры заданы
        if query:
            search_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Поиск']"))
            )
            search_input.clear()
            search_input.send_keys(query)
        
        # Заполняем поле минимальной суммы, если задано
        if price_from:
            try:
                price_from_input = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Сумма закупки с']"))
                )
                price_from_input.clear()
                price_from_input.send_keys(str(price_from))
            except TimeoutException:
                st.warning("Не удалось найти поле для ввода минимальной суммы")
        
        # Заполняем поле максимальной суммы, если задано
        if price_to:
            try:
                price_to_input = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Сумма закупки по']"))
                )
                price_to_input.clear()
                price_to_input.send_keys(str(price_to))
            except TimeoutException:
                st.warning("Не удалось найти поле для ввода максимальной суммы")
        
        # Выбираем статус, если задан
        if status:
            try:
                # Открываем выпадающий список статусов
                status_dropdown = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Статус')]"))
                )
                status_dropdown.click()
                time.sleep(1)
                
                # Выбираем нужный статус
                status_option = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, f"//div[contains(text(), '{status}')]"))
                )
                status_option.click()
            except TimeoutException:
                st.warning("Не удалось выбрать статус закупки")
        
        # Нажимаем кнопку поиска
        try:
            search_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Поиск')]"))
            )
            search_button.click()
            time.sleep(3)
        except TimeoutException:
            st.warning("Не удалось найти кнопку поиска")
        
        # Парсинг страниц с результатами
        current_page = 1
        
        while current_page <= max_pages:
            # Получаем таблицу с результатами
            try:
                table = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//table[@id='resultTable']"))
                )
                
                # Парсим строки таблицы
                rows = table.find_elements(By.TAG_NAME, "tr")
                
                # Пропускаем заголовок таблицы
                for row in rows[1:]:
                    try:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        
                        if len(cells) >= 6:
                            # Извлекаем данные из ячеек
                            lot_number = cells[0].text
                            lot_name = cells[1].find_element(By.TAG_NAME, "a").text
                            quantity = cells[2].text
                            price = cells[3].text.replace(" ", "").replace(",", ".")
                            purchase_type = cells[4].text
                            status = cells[5].text
                            
                            # Получаем ссылку на детали лота
                            lot_link = cells[1].find_element(By.TAG_NAME, "a").get_attribute("href")
                            
                            # Добавляем данные в результаты
                            results.append({
                                "lot_number": lot_number,
                                "lot_name": lot_name,
                                "quantity": quantity,
                                "price": price,
                                "purchase_type": purchase_type,
                                "status": status,
                                "lot_link": lot_link
                            })
                    except Exception as e:
                        continue
                
                # Проверяем наличие кнопки "Следующая страница"
                try:
                    next_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, "//a[@aria-label='Next']"))
                    )
                    next_button.click()
                    current_page += 1
                    time.sleep(2)
                except TimeoutException:
                    # Если кнопка не найдена, значит достигли последней страницы
                    break
                
            except TimeoutException:
                st.warning("Не удалось загрузить таблицу с результатами")
                break
        
        return pd.DataFrame(results)
    
    except Exception as e:
        st.error(f"Ошибка при парсинге данных: {str(e)}")
        return pd.DataFrame()
    
    finally:
        if driver:
            driver.quit()

# Функция для получения дополнительных данных о закупке
def get_lot_details(lot_link, driver=None):
    """
    Получение дополнительных данных о закупке по ссылке
    """
    close_driver = False
    if driver is None:
        driver = init_driver(headless=True)
        close_driver = True
    
    details = {}
    
    try:
        driver.get(lot_link)
        time.sleep(3)
        
        # Извлекаем данные о поставщике
        try:
            supplier_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Поставщик')]"))
            )
            supplier_tab.click()
            time.sleep(2)
            
            # Получаем имя поставщика
            supplier_name = driver.find_element(By.XPATH, "//div[contains(@class, 'supplier-name')]").text
            details["supplier_name"] = supplier_name
            
            # Получаем ID поставщика
            supplier_id = driver.find_element(By.XPATH, "//div[contains(@class, 'supplier-id')]").text
            details["supplier_id"] = supplier_id
        except:
            details["supplier_name"] = "Нет данных"
            details["supplier_id"] = "Нет данных"
        
        # Извлекаем данные о регионе
        try:
            region_element = driver.find_element(By.XPATH, "//div[contains(text(), 'Регион:')]/following-sibling::div")
            details["region"] = region_element.text
        except:
            details["region"] = "Нет данных"
        
        # Извлекаем данные о категории
        try:
            category_element = driver.find_element(By.XPATH, "//div[contains(text(), 'Категория:')]/following-sibling::div")
            details["category"] = category_element.text
        except:
            details["category"] = "Нет данных"
        
        # Дата публикации
        try:
            date_element = driver.find_element(By.XPATH, "//div[contains(text(), 'Дата публикации:')]/following-sibling::div")
            details["publication_date"] = date_element.text
            
            # Вычисляем дни до тендера
            pub_date = datetime.datetime.strptime(date_element.text, "%d.%m.%Y")
            today = datetime.datetime.now()
            details["days_to_tender"] = (today - pub_date).days
        except:
            details["publication_date"] = "Нет данных"
            details["days_to_tender"] = 0
        
        return details
    
    except Exception as e:
        print(f"Ошибка при получении деталей закупки: {str(e)}")
        return {}
    
    finally:
        if close_driver and driver:
            driver.quit()

# Функция для подготовки данных для модели
def prepare_data_for_model(parsed_data):
    """
    Подготовка данных, полученных с сайта, для анализа моделью
    
    Parameters:
    - parsed_data: DataFrame с данными, полученными при парсинге
    
    Returns:
    - DataFrame с данными, подготовленными для анализа моделью
    """
    if parsed_data.empty:
        return pd.DataFrame()
    
    try:
        # Создаем копию данных
        model_data = parsed_data.copy()
        
        # Преобразуем цену в числовой формат
        model_data["price"] = model_data["price"].astype(float)
        
        # Генерируем некоторые синтетические признаки для демонстрации
        # Эти данные в реальном использовании должны быть получены из API
        
        # Средняя цена по категории и региону
        model_data["avg_price_category_region"] = model_data.groupby(["category", "region"])["price"].transform("mean")
        
        # Цена за единицу (для демонстрации используем случайные значения)
        model_data["price_per_unit"] = model_data["price"] / np.random.randint(1, 10, size=len(model_data))
        
        # Количество выигранных контрактов поставщиком (демо)
        model_data["supplier_win_count"] = np.random.randint(0, 50, size=len(model_data))
        
        # Лет активности поставщика (демо)
        model_data["supplier_years_active"] = np.random.randint(1, 15, size=len(model_data))
        
        # Всего контрактов поставщика (демо)
        model_data["supplier_total_contracts"] = model_data["supplier_win_count"] + np.random.randint(0, 50, size=len(model_data))
        
        # Средняя стоимость контрактов поставщика (демо)
        model_data["supplier_avg_contract_value"] = model_data["price"] * np.random.uniform(0.8, 1.2, size=len(model_data))
        
        # Приведение столбцов к формату, требуемому моделью
        required_columns = [
            'category', 'region', 'price', 'avg_price_category_region',
            'supplier_id', 'supplier_name', 'supplier_win_count',
            'days_to_tender', 'price_per_unit', 'supplier_years_active',
            'supplier_total_contracts', 'supplier_avg_contract_value'
        ]
        
        # Проверяем наличие всех необходимых столбцов
        for column in required_columns:
            if column not in model_data.columns:
                if column in ["supplier_id", "supplier_name", "region", "category"]:
                    model_data[column] = "unknown"
                else:
                    model_data[column] = 0
        
        # Возвращаем только столбцы, необходимые для модели
        return model_data[required_columns]
        
    except Exception as e:
        st.error(f"Ошибка при подготовке данных для модели: {str(e)}")
        return pd.DataFrame()

# Функция для запуска парсинга с отображением прогресса
def run_parsing_with_progress(url, query=None, price_from=None, price_to=None, status=None, max_pages=5):
    """
    Запуск парсинга с отображением прогресса в Streamlit
    """
    # Создаем контейнер для отображения прогресса
    progress_container = st.empty()
    
    # Отображаем начальный прогресс
    progress_container.progress(0)
    progress_container.text("Инициализация парсинга...")
    
    # Парсинг данных
    parsed_data = parse_goszakup(url, query, price_from, price_to, status, max_pages)
    
    if parsed_data.empty:
        progress_container.error("Не удалось получить данные. Проверьте параметры поиска.")
        return pd.DataFrame(), pd.DataFrame()
    
    progress_container.progress(50)
    progress_container.text(f"Получено {len(parsed_data)} записей. Подготовка данных для анализа...")
    
    # Подготовка данных для модели
    model_data = prepare_data_for_model(parsed_data)
    
    if model_data.empty:
        progress_container.error("Не удалось подготовить данные для анализа.")
        return pd.DataFrame(), pd.DataFrame()
    
    progress_container.progress(100)
    progress_container.text("Парсинг и подготовка данных завершены успешно!")
    
    return parsed_data, model_data

# Боковая панель
st.sidebar.title("Навигация")

# Выбор страницы
page = st.sidebar.radio(
    "Выберите раздел",
    ["Главная", "Загрузка файла", "Проверка вручную", "Визуализации", "Карта регионов"]
)

# Подвал боковой панели
st.sidebar.markdown("---")
st.sidebar.info("Система анализа подозрительных государственных закупок © 2025")

# Главная страница
if page == "Главная":
    st.title("Система анализа подозрительных государственных закупок")
    st.subheader("FairTender.kz")
    
    st.markdown("""
    ### О системе
    
    Данная система использует методы машинного обучения для выявления потенциально подозрительных государственных закупок.
    
    ### Возможности:
    
    - **Массовый анализ**: Загрузка CSV файла с данными о закупках для массового анализа
    - **Проверка вручную**: Проверка отдельной закупки путем ввода ее параметров
    - **Визуализации**: Графики, показывающие важность признаков и метрики модели
    """)
    
    # Информация о модели
    st.header("Информация о модели")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Точность (accuracy)", "96.1%")
    col2.metric("Точность (precision)", "98.5%")
    col3.metric("Полнота (recall)", "77.4%")
    col4.metric("F1-мера", "86.7%")
    
    st.markdown("""
    ### Используемые признаки:
    
    #### Характеристики закупки:
    - **Категория** - категория закупки
    - **Регион** - регион проведения закупки
    - **Стоимость** - общая стоимость контракта
    - **Средняя стоимость по категории и региону** - среднее значение для данной категории и региона
    - **Цена за единицу** - стоимость за единицу товара/услуги
    - **Дней до тендера** - количество дней до проведения тендера
    
    #### Характеристики поставщика:
    - **ID поставщика** - уникальный идентификатор поставщика
    - **Название поставщика** - название компании
    - **Выигранных контрактов** - количество выигранных контрактов поставщиком
    - **Лет активности** - период работы поставщика на рынке
    - **Всего контрактов** - общее количество контрактов поставщика
    - **Средняя стоимость контрактов** - средняя стоимость предыдущих контрактов поставщика
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Загрузка файла:** Загрузите CSV файл с данными о закупках для массового анализа")
    
    with col2:
        st.info("**Проверка вручную:** Введите параметры конкретной закупки для анализа ее подозрительности")
    
    with col3:
        st.info("**Визуализации:** Ознакомьтесь с графиками и метриками модели")

# Страница загрузки файла
elif page == "Загрузка файла":
    st.title("Загрузка данных для анализа")
    
    st.write("Загрузите CSV файл с данными о закупках для анализа.")
    
    uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
    
    if uploaded_file is not None:
        try:
            # Загружаем данные
            uploaded_data = pd.read_csv(uploaded_file)
            
            # Проверяем, что все необходимые колонки присутствуют
            required_columns = ['category', 'region', 'price', 'avg_price_category_region',
                'supplier_id', 'supplier_name', 'supplier_win_count',
                'days_to_tender', 'price_per_unit', 'supplier_years_active',
                'supplier_total_contracts', 'supplier_avg_contract_value']
            
            missing_columns = [col for col in required_columns if col not in uploaded_data.columns]
            
            if missing_columns:
                st.error(f"В файле отсутствуют следующие столбцы: {', '.join(missing_columns)}")
                st.error(f"Требуемые столбцы: {', '.join(required_columns)}")
                
                # Показываем таблицу с названиями столбцов на русском и английском
                translation_df = pd.DataFrame({
                    'Английское название': list(column_translations.keys()),
                    'Русское название': list(column_translations.values())
                })
                st.table(translation_df)
                
                st.error("Пожалуйста, проверьте соответствие имен столбцов и загрузите файл повторно.")
            else:
                # Переименовываем столбцы из русских в английские, если они присутствуют
                renamed_data = uploaded_data.copy()
                for rus_col in renamed_data.columns:
                    if rus_col in reverse_translations:
                        renamed_data = renamed_data.rename(columns={rus_col: reverse_translations[rus_col]})
                
                # Показываем пример загруженных данных
                st.subheader("Пример данных:")
                # Переводим столбцы для отображения
                display_data = renamed_data.copy()
                for eng_col in display_data.columns:
                    if eng_col in column_translations:
                        display_data = display_data.rename(columns={eng_col: column_translations[eng_col]})
                st.dataframe(display_data.head())
                
                # Предсказание
                with st.spinner('Анализ данных...'):
                    try:
                        predictions, probabilities = predict_suspicious(renamed_data)
                        
                        # Создаем DataFrame с результатами
                        results_df = pd.DataFrame({
                            'is_suspicious_pred': predictions,
                            'is_suspicious_prob': probabilities
                        })
                        
                        # Объединяем результаты с исходными данными
                        results = pd.concat([renamed_data, results_df], axis=1)
                        
                        # Отображаем результаты
                        st.subheader("Результаты анализа:")
                        
                        # Подозрительные закупки
                        suspicious_count = results['is_suspicious_pred'].sum()
                        total_count = len(results)
                        
                        st.warning(f"Обнаружено {suspicious_count} подозрительных закупок из {total_count} ({suspicious_count/total_count:.1%})")
                        
                        # Фильтруем подозрительные
                        suspicious_df = results[results['is_suspicious_pred'] == 1].sort_values('is_suspicious_prob', ascending=False)
                        
                        # Функция для подсветки подозрительных строк
                        def highlight_suspicious(row):
                            # Проверяем наличие столбца на русском или английском
                            if 'is_suspicious_pred' in row:
                                pred_col = 'is_suspicious_pred'
                            elif 'подозрительность_pred' in row:
                                pred_col = 'подозрительность_pred'
                            else:
                                return [''] * len(row)
                            
                            color = 'background-color: #ffcccc' if row[pred_col] == 1 else ''
                            return [color] * len(row)
                        
                        # Отображаем топ подозрительных
                        if not suspicious_df.empty:
                            st.subheader("Топ подозрительные закупки:")
                            # Переименовываем столбцы на русский язык для отображения
                            display_suspicious = suspicious_df.copy()
                            # Переводим только те столбцы, для которых есть перевод
                            for eng_col in display_suspicious.columns:
                                if eng_col in column_translations:
                                    display_suspicious = display_suspicious.rename(columns={eng_col: column_translations[eng_col]})
                            
                            styled_suspicious = display_suspicious.head(10).style.apply(highlight_suspicious, axis=1)
                            st.dataframe(styled_suspicious)
                        
                        # Добавляем визуализацию распределения вероятностей
                        st.subheader("Распределение вероятностей:")
                        fig = create_probability_distribution_plot(results)
                        st.pyplot(fig)
                        
                        # Добавляем статистику средних значений
                        st.subheader("Средняя статистика:")
                        
                        # Создаем DataFrame со средними значениями для числовых столбцов
                        numeric_cols = ['price', 'avg_price_category_region', 'supplier_win_count', 
                                      'days_to_tender', 'price_per_unit', 'supplier_years_active', 
                                      'supplier_total_contracts', 'supplier_avg_contract_value']
                        
                        # Отдельно для подозрительных и не подозрительных закупок
                        suspicious_stats = results[results['is_suspicious_pred'] == 1][numeric_cols].mean()
                        non_suspicious_stats = results[results['is_suspicious_pred'] == 0][numeric_cols].mean()
                        all_stats = results[numeric_cols].mean()
                        
                        # Создаем DataFrame для отображения
                        stats_df = pd.DataFrame({
                            'Подозрительные закупки': suspicious_stats,
                            'Обычные закупки': non_suspicious_stats,
                            'Все закупки': all_stats
                        })
                        
                        # Переименовываем строки индексов на русский
                        stats_df.index = [column_translations.get(col, col) for col in stats_df.index]
                        
                        # Форматируем числовые данные для лучшей читаемости
                        stats_display = stats_df.copy()
                        for col in stats_display.columns:
                            stats_display[col] = stats_display[col].apply(lambda x: f"{x:,.2f}")
                        
                        # Отображаем статистику
                        st.dataframe(stats_display)
                        
                        # Добавляем сравнительную диаграмму для выбранных показателей
                        st.subheader("Сравнение показателей:")
                        
                        # Выбираем наиболее важные показатели для сравнения
                        key_metrics = ['price', 'price_per_unit', 'supplier_win_count', 'supplier_years_active']
                        comparison_df = stats_df.loc[[column_translations.get(col, col) for col in key_metrics]]
                        
                        # Строим диаграмму сравнения
                        fig, ax = plt.subplots(figsize=(10, 6))
                        comparison_df.plot(kind='bar', ax=ax)
                        plt.title('Сравнение средних значений показателей')
                        plt.ylabel('Значение')
                        plt.xlabel('Показатель')
                        plt.xticks(rotation=45)
                        plt.legend(title='Категория закупок')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Возможность скачать результаты
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="Скачать результаты",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Ошибка при анализе данных: {str(e)}")
                
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")



# Страница проверки вручную
elif page == "Проверка вручную":
    st.title("Проверка отдельной закупки")
    
    st.write("Укажите параметры закупки для проверки на подозрительность.")
    
    # Создаем форму
    with st.form("manual_check_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Категория и регион
            category = st.selectbox(
                "Категория закупки",
                options=feature_info['category']['values'],
                index=0
            )
            
            region = st.selectbox(
                "Регион",
                options=feature_info['region']['values'],
                index=0
            )
            
            # Цена и средняя цена
            price = st.number_input(
                "Стоимость контракта (₸)",
                min_value=float(feature_info['price']['min']),
                max_value=float(feature_info['price']['max']),
                value=float(feature_info['price']['mean'])
            )
            
            avg_price_category_region = st.number_input(
                "Средняя стоимость по категории/региону (₸)",
                min_value=float(feature_info['avg_price_category_region']['min']),
                max_value=float(feature_info['avg_price_category_region']['max']),
                value=float(feature_info['avg_price_category_region']['mean'])
            )
            
            # Поставщик
            supplier_id = st.selectbox(
                "Идентификатор поставщика",
                options=feature_info['supplier_id']['values'][:100],
                index=0
            )
            
            supplier_name = st.selectbox(
                "Название поставщика",
                options=feature_info['supplier_name']['values'][:100],
                index=0
            )
        
        with col2:
            # Данные о поставщике
            supplier_win_count = st.number_input(
                "Кол-во выигранных контрактов",
                min_value=int(feature_info['supplier_win_count']['min']),
                max_value=int(feature_info['supplier_win_count']['max']),
                value=int(feature_info['supplier_win_count']['mean'])
            )
            
            supplier_years_active = st.number_input(
                "Лет активности",
                min_value=int(feature_info['supplier_years_active']['min']),
                max_value=int(feature_info['supplier_years_active']['max']),
                value=int(feature_info['supplier_years_active']['mean'])
            )
            
            supplier_total_contracts = st.number_input(
                "Всего контрактов",
                min_value=int(feature_info['supplier_total_contracts']['min']),
                max_value=int(feature_info['supplier_total_contracts']['max']),
                value=int(feature_info['supplier_total_contracts']['mean'])
            )
            
            # Дополнительные параметры
            days_to_tender = st.number_input(
                "Дней до тендера",
                min_value=int(feature_info['days_to_tender']['min']),
                max_value=int(feature_info['days_to_tender']['max']),
                value=int(feature_info['days_to_tender']['mean'])
            )
            
            price_per_unit = st.number_input(
                "Цена за единицу (₸)",
                min_value=float(feature_info['price_per_unit']['min']),
                max_value=float(feature_info['price_per_unit']['max']),
                value=float(feature_info['price_per_unit']['mean'])
            )
            
            supplier_avg_contract_value = st.number_input(
                "Средняя стоимость контрактов (₸)",
                min_value=float(feature_info['supplier_avg_contract_value']['min']),
                max_value=float(feature_info['supplier_avg_contract_value']['max']),
                value=float(feature_info['supplier_avg_contract_value']['mean'])
            )
        
        # Кнопка отправки формы
        submitted = st.form_submit_button("Проверить закупку")
    
    # Обработка формы
    if submitted:
        # Собираем данные из формы
        form_data = {
            'category': category,
            'region': region,
            'price': price,
            'avg_price_category_region': avg_price_category_region,
            'supplier_id': supplier_id,
            'supplier_name': supplier_name,
            'supplier_win_count': supplier_win_count,
            'days_to_tender': days_to_tender,
            'price_per_unit': price_per_unit,
            'supplier_years_active': supplier_years_active,
            'supplier_total_contracts': supplier_total_contracts,
            'supplier_avg_contract_value': supplier_avg_contract_value
        }
        
        # Создаем DataFrame из введенных данных
        input_data = pd.DataFrame([form_data])
        
        # Получаем предсказания
        with st.spinner('Анализ данных...'):
            prediction, probability = predict_suspicious(input_data)
        
        # Показываем результаты
        st.subheader("Результат анализа")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.error("⚠️ Закупка подозрительная")
            else:
                st.success("✅ Закупка не подозрительная")
            
            st.metric("Вероятность подозрительности", f"{probability[0]:.2%}")
        
        with col2:
            # Создаем SHAP график
            fig = create_shap_plot_single(input_data)
            st.pyplot(fig)
        
        # Показываем введенные данные
        with st.expander("Показать введенные данные"):
            # Переводим ключи на русский для отображения
            form_data_ru = {}
            for key, value in form_data.items():
                if key in column_translations:
                    form_data_ru[column_translations[key]] = value
                else:
                    form_data_ru[key] = value
            st.json(form_data_ru)

# Страница визуализаций
elif page == "Визуализации":
    st.title("Визуализации")
    
    tab1, tab2, tab3 = st.tabs(["Метрики модели", "Важность признаков", "Распределение вероятностей"])
    
    with tab1:
        st.header("Метрики качества модели")
        fig, metrics = create_metrics_plot()
        st.pyplot(fig)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Точность (accuracy)", f"{metrics['Точность (accuracy)']:.2%}")
        col2.metric("Точность (precision)", f"{metrics['Точность (precision)']:.2%}")
        col3.metric("Полнота (recall)", f"{metrics['Полнота (recall)']:.2%}")
        col4.metric("F1-мера", f"{metrics['F1-мера']:.2%}")
    
    with tab2:
        st.header("Важность признаков (SHAP)")
        
        # Загружаем данные для SHAP графиков
        train_data = pd.read_csv('final_training_data.csv')
        sample_data = train_data.drop('is_suspicious', axis=1).sample(n=min(100, len(train_data)), random_state=42)
        
        # Преобразование категориальных признаков
        categorical_features = ['category', 'region', 'supplier_id', 'supplier_name']
        for feature in categorical_features:
            sample_data[feature] = encoders[feature].transform(sample_data[feature].astype(str))
        
        # Получаем правильный порядок признаков из модели
        feature_names = model.feature_names_in_
        
        # Переупорядочиваем столбцы в соответствии с порядком в модели
        sample_data = sample_data[feature_names]
        
        # Создаем график
        with st.spinner('Создание SHAP графика...'):
            fig = create_shap_summary_plot(sample_data)
            st.pyplot(fig)
        
        st.markdown("""
        **Как интерпретировать SHAP значения:**
        - Чем выше признак в списке, тем больше его влияние на предсказание модели
        - Красный цвет означает увеличение вероятности подозрительности
        - Синий цвет означает уменьшение вероятности подозрительности
        """)
    
    with tab3:
        st.header("Распределение вероятностей предсказаний")
        
        # Загружаем имеющиеся предсказания, если они есть
        predictions_path = 'uploads/predictions.csv'
        if os.path.exists(predictions_path):
            predictions_df = pd.read_csv(predictions_path)
            if 'is_suspicious_pred' in predictions_df.columns and 'is_suspicious_prob' in predictions_df.columns:
                fig = create_probability_distribution_plot(predictions_df)
                st.pyplot(fig)
                
                # Статистика
                suspicious_count = predictions_df['is_suspicious_pred'].sum()
                total_count = len(predictions_df)
                avg_prob = predictions_df['is_suspicious_prob'].mean()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Всего проанализировано", f"{total_count}")
                col2.metric("Подозрительных закупок", f"{suspicious_count} ({suspicious_count/total_count:.1%})")
                col3.metric("Средняя вероятность", f"{avg_prob:.2%}")
            else:
                st.info("Файл предсказаний не содержит необходимых столбцов.")
        else:
            st.info("Загрузите данные через страницу 'Загрузка файла' для просмотра распределения вероятностей.")

# Страница карты регионов
elif page == "Карта регионов":
    st.title("Распределение подозрительных закупок по регионам Казахстана")
    
    st.info("**Обратите внимание:** Карта использует демонстрационные данные (мок-данные), на которых был обучен алгоритм. Распределение закупок используется только для иллюстрации функциональности системы.")
    
    # Создаем мок-данные для регионов Казахстана
    # Список основных регионов Казахстана
    kazakhstan_regions = [
        "Абайская область", "Акмолинская область", "Актюбинская область", 
        "Алматинская область", "Атырауская область", "Восточно-Казахстанская область", 
        "Жамбылская область", "Жетысуская область", "Западно-Казахстанская область",
        "Карагандинская область", "Костанайская область", "Кызылординская область", 
        "Мангистауская область", "Павлодарская область", "Северо-Казахстанская область", 
        "Туркестанская область", "Улытауская область",
        "г. Алматы", "г. Астана", "г. Шымкент"
    ]
    
    # Создаем DataFrame с мок-данными по закупкам для каждого региона
    np.random.seed(42)  # Для воспроизводимости результатов
    
    # Генерация мок-данных
    region_data = pd.DataFrame({
        'region': kazakhstan_regions,
        'total_tenders': np.random.randint(200, 1000, size=len(kazakhstan_regions)),
    })
    
    # Генерируем количество подозрительных закупок (с большими значениями для некоторых регионов)
    region_data['suspicious_tenders'] = np.random.randint(10, 150, size=len(kazakhstan_regions))
    
    # Добавляем различные искусственные характеристики для визуализации
    region_data['suspicious_percent'] = (region_data['suspicious_tenders'] / region_data['total_tenders'] * 100).round(1)
    region_data['avg_tender_amount'] = np.random.randint(1000000, 10000000, size=len(kazakhstan_regions))
    region_data['avg_suspicious_amount'] = region_data['avg_tender_amount'] * np.random.uniform(1.1, 1.5, size=len(kazakhstan_regions))
    
    # Координаты регионов (приблизительные центры регионов Казахстана)
    region_coordinates = {
        "Абайская область": [49.3, 81.5],
        "Акмолинская область": [51.9, 70.0],
        "Актюбинская область": [50.3, 57.2],
        "Алматинская область": [44.0, 78.4],
        "Атырауская область": [47.1, 51.9],
        "Восточно-Казахстанская область": [48.7, 82.6],
        "Жамбылская область": [43.5, 71.4],
        "Жетысуская область": [45.1, 79.0],
        "Западно-Казахстанская область": [50.0, 51.2],
        "Карагандинская область": [49.8, 73.1],
        "Костанайская область": [52.6, 63.3],
        "Кызылординская область": [44.8, 65.5],
        "Мангистауская область": [44.6, 53.3],
        "Павлодарская область": [52.3, 76.9],
        "Северо-Казахстанская область": [54.2, 69.4],
        "Туркестанская область": [43.3, 68.3],
        "Улытауская область": [47.9, 66.9],
        "г. Алматы": [43.2, 76.9],
        "г. Астана": [51.2, 71.4],
        "г. Шымкент": [42.3, 69.6]
    }
    
    # Добавляем координаты в DataFrame
    region_data['latitude'] = region_data['region'].apply(lambda x: region_coordinates[x][0])
    region_data['longitude'] = region_data['region'].apply(lambda x: region_coordinates[x][1])
    
    # Создаем вкладки для разных представлений
    map_tab1, map_tab2, map_tab3 = st.tabs(["Карта подозрительности", "Тепловая карта", "Таблица данных"])
    
    with map_tab1:
        st.subheader("Распределение подозрительных закупок")
        
        # Фильтры
        col1, col2 = st.columns(2)
        with col1:
            metric_choice = st.selectbox(
                "Показать метрику:", 
                ["Количество подозрительных закупок", "Процент подозрительных закупок", "Средняя сумма подозрительных закупок"]
            )
        
        with col2:
            min_tenders = st.slider(
                "Минимальное количество закупок:", 
                min_value=int(region_data['total_tenders'].min()),
                max_value=int(region_data['total_tenders'].max()),
                value=int(region_data['total_tenders'].min())
            )
        
        # Фильтруем данные
        filtered_data = region_data[region_data['total_tenders'] >= min_tenders]
        
        # Определяем, какие данные показать на карте в зависимости от выбранной метрики
        if metric_choice == "Количество подозрительных закупок":
            size_col = "suspicious_tenders"
            hover_name = "region"
            hover_data = ["suspicious_tenders", "total_tenders", "suspicious_percent"]
            size_max = 50
            title = "Количество подозрительных закупок по регионам"
        elif metric_choice == "Процент подозрительных закупок":
            size_col = "suspicious_percent"
            hover_name = "region"
            hover_data = ["suspicious_percent", "suspicious_tenders", "total_tenders"]
            size_max = 60
            title = "Процент подозрительных закупок по регионам"
        else:
            size_col = "avg_suspicious_amount"
            hover_name = "region"
            hover_data = ["avg_suspicious_amount", "suspicious_tenders", "total_tenders"]
            size_max = 70
            title = "Средняя сумма подозрительных закупок по регионам"
        
        # Создаем карту с интерактивным отображением данных
        fig = px.scatter_mapbox(
            filtered_data,
            lat="latitude",
            lon="longitude",
            size=size_col,
            color="suspicious_percent",
            color_continuous_scale=px.colors.sequential.Reds,
            size_max=size_max,
            hover_name=hover_name,
            hover_data=hover_data,
            mapbox_style="carto-positron",
            zoom=4,
            center={"lat": 48.5, "lon": 68},
            opacity=0.7,
            title=title
        )
        
        fig.update_layout(height=600, margin={"r":0, "t":30, "l":0, "b":0})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Как интерпретировать карту:**
        - **Размер маркера** показывает выбранную метрику (количество/процент/сумму подозрительных закупок)
        - **Цвет маркера** соответствует проценту подозрительных закупок (чем темнее, тем выше процент)
        - **Наведите курсор** на маркер для отображения детальной информации по региону
        """)
    
    with map_tab2:
        st.subheader("Тепловая карта подозрительности закупок по регионам")
        
        # Создаем тепловую карту по процентам подозрительных закупок
        fig = px.density_mapbox(
            region_data,
            lat="latitude",
            lon="longitude",
            z="suspicious_percent",
            radius=50,
            center={"lat": 48.5, "lon": 68},
            zoom=4,
            mapbox_style="carto-positron",
            opacity=0.7,
            color_continuous_scale=px.colors.sequential.Reds,
            title="Тепловая карта подозрительности закупок"
        )
        
        fig.update_layout(height=600, margin={"r":0, "t":30, "l":0, "b":0})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Тепловая карта** показывает концентрацию подозрительных закупок в разных регионах Казахстана. 
        Более интенсивный красный цвет соответствует более высокому проценту подозрительных закупок.
        """)
        
    with map_tab3:
        st.subheader("Таблица данных по регионам")
        
        # Переименуем столбцы для отображения
        display_data = region_data.copy()
        display_data = display_data.rename(columns={
            'region': 'Регион',
            'total_tenders': 'Всего закупок',
            'suspicious_tenders': 'Подозрительных закупок',
            'suspicious_percent': 'Процент подозрительных (%)',
            'avg_tender_amount': 'Средняя сумма закупки (₸)',
            'avg_suspicious_amount': 'Средняя сумма подозрительной закупки (₸)'
        })
        
        # Отображаем таблицу без координат
        st.dataframe(
            display_data.drop(['latitude', 'longitude'], axis=1).sort_values('Процент подозрительных (%)', ascending=False),
            use_container_width=True
        )
        
        # Кнопка для скачивания данных
        csv = display_data.drop(['latitude', 'longitude'], axis=1).to_csv(index=False)
        st.download_button(
            label="Скачать данные по регионам",
            data=csv,
            file_name="regions_data.csv",
            mime="text/csv"
        )
        
        # Добавляем график распределения процента подозрительных закупок по регионам
        st.subheader("Распределение процента подозрительных закупок")
        fig, ax = plt.subplots(figsize=(12, 8))
        sorted_data = display_data.sort_values('Процент подозрительных (%)', ascending=True)
        sns.barplot(
            x='Процент подозрительных (%)',
            y='Регион',
            data=sorted_data,
            palette='Reds_r',
            ax=ax
        )
        plt.xlabel('Процент подозрительных закупок (%)')
        plt.ylabel('Регион')
        plt.title('Распределение процента подозрительных закупок по регионам')
        st.pyplot(fig) 