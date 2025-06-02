"""
Модуль для тестирования класса DataPreprocessor из app.preprocessing
Проверяет корректность предобработки данных, включая:
- Инициализацию препроцессора
- Проверку обязательных признаков
- Обработку категориальных признаков
- Обработку числовых признаков
- Обработку пропущенных значений
- Кодирование конкретных значений
"""

import pytest
import pandas as pd
import numpy as np
from app.preprocessing import DataPreprocessor

# Фикстура для создания экземпляра препроцессора
@pytest.fixture(scope="module")
def preprocessor():
    """
    Фикстура, создающая экземпляр DataPreprocessor для всех тестов модуля.
    Загружает артефакты из файла 'artifacts/preprocessor.pkl'
    """
    return DataPreprocessor('artifacts/preprocessor.pkl')

def test_preprocessor_initialization(preprocessor):
    """
    Тестирует инициализацию препроцессора.
    Проверяет, что все необходимые атрибуты загружены корректно.
    """
    # Проверка загрузки списков признаков
    assert preprocessor.cat_columns is not None
    assert isinstance(preprocessor.cat_columns, list)
    assert len(preprocessor.cat_columns) > 0
    
    assert preprocessor.num_columns is not None
    assert isinstance(preprocessor.num_columns, list)
    
    # Проверка загрузки используемых признаков
    assert preprocessor.used_features is not None
    assert len(preprocessor.used_features) > 0
    
    # Проверка загрузки компонентов обработки
    assert hasattr(preprocessor, 'ordinal_encoder')
    assert hasattr(preprocessor, 'simple_imputer_before_ord')
    assert hasattr(preprocessor, 'simple_imputer_after_ord')

def test_feature_check(preprocessor):
    """
    Тестирует проверку обязательных признаков.
    Убеждается, что препроцессор корректно выявляет отсутствие обязательных признаков.
    """
    # Создаем данные с одним признаком (неполный набор)
    sample_data = pd.DataFrame({'Age': [30]})
    
    # Проверяем, что преобразование вызывает исключение
    with pytest.raises(ValueError) as excinfo:
        preprocessor.transform(sample_data)
    
    # Проверяем текст сообщения об ошибке
    assert "Отсутствуют обязательные признаки" in str(excinfo.value)

def test_categorical_processing(preprocessor):
    """
    Тестирует обработку категориальных признаков.
    Проверяет преобразование значений и кодирование.
    """
    # Создаем полный набор признаков со значениями по умолчанию
    sample_data = pd.DataFrame({
        col: ['0'] * 2  # Для категориальных признаков
        if col in preprocessor.cat_columns 
        else [0] * 2    # Для числовых признаков
        for col in preprocessor.used_features
    })
    
    # Устанавливаем тестовые значения для конкретных признаков
    sample_data['Gender'] = ['Male', 'Female']
    sample_data['Stress Level'] = ['2', '3']
    
    # Выполняем преобразование
    processed = preprocessor.transform(sample_data)
    
    # Проверяем кодирование признака Gender
    # Male должен кодироваться как 0, Female как 1
    assert processed['Gender'].tolist() == [0, 1]
    
    # Проверяем, что Stress Level преобразован в числовой формат
    assert processed['Stress Level'].dtype in [np.int64, np.float64]
    
    # Проверяем отсутствие пропущенных значений после преобразований
    assert not processed[preprocessor.cat_columns].isnull().any().any()

def test_numerical_processing(preprocessor):
    """
    Тестирует обработку числовых признаков.
    Проверяет корректность преобразований и работу скалера (если используется).
    """
    # Создаем полный набор признаков со значениями по умолчанию
    sample_data = pd.DataFrame({
        col: ['0']  # Для категориальных признаков
        if col in preprocessor.cat_columns 
        else [0]    # Для числовых признаков
        for col in preprocessor.used_features
    })
    
    # Устанавливаем тестовые значения для числовых признаков
    sample_data['Age'] = [25, 30, 35]  # Три строки данных
    sample_data['BMI'] = [22.1, 24.5, 26.8]
    
    # Дублируем категориальные признаки для трех строк
    for col in preprocessor.cat_columns:
        sample_data[col] = ['0'] * 3
    
    # Выполняем преобразование
    processed = preprocessor.transform(sample_data)
    
    # Проверяем наличие числовых признаков в результате
    assert 'Age' in processed.columns
    assert 'BMI' in processed.columns
    
    # Проверяем обработку в зависимости от наличия скалера
    if preprocessor.num_scaler:
        # Для масштабированных признаков проверяем диапазон [0, 1]
        assert processed['Age'].min() >= 0
        assert processed['Age'].max() <= 1
        assert processed['BMI'].min() >= 0
        assert processed['BMI'].max() <= 1
    else:
        # Без скалера значения должны остаться неизменными
        assert processed['Age'].tolist() == [25, 30, 35]
        assert processed['BMI'].tolist() == [22.1, 24.5, 26.8]

def test_missing_values_handling(preprocessor):
    """
    Тестирует обработку пропущенных значений.
    Проверяет, что после преобразований нет пропущенных значений.
    """
    # Создаем данные с пропусками во всех признаках
    sample_data = pd.DataFrame({
        col: [None] * 2  # Создаем пропуски
        for col in preprocessor.used_features
    })
    
    # Выполняем преобразование
    processed = preprocessor.transform(sample_data)
    
    # Проверяем отсутствие пропущенных значений после обработки
    assert not processed.isnull().any().any()
    
    # Проверяем, что все значения заполнены
    for col in processed.columns:
        assert processed[col].notnull().all()

@pytest.mark.parametrize("gender_value,expected_code", [
    ('Male', 0),     # Стандартное значение
    ('Female', 1),   # Стандартное значение
    ('male', 0),     # Проверка регистронезависимости (если реализовано)
    ('female', 1),   # Проверка регистронезависимости
    ('M', 0),        # Альтернативное обозначение
    ('F', 1),        # Альтернативное обозначение
    ('Other', -1),   # Неизвестное значение (если обрабатывается)
    ('', -1),        # Пустое значение
    (None, -1)       # Пропущенное значение
])
def test_gender_encoding(preprocessor, gender_value, expected_code):
    """
    Параметризованный тест для кодирования признака Gender.
    Проверяет различные варианты значений и их преобразование.
    """
    # Создаем базовый набор данных
    sample_data = pd.DataFrame({
        col: ['0'] if col in preprocessor.cat_columns else [0]
        for col in preprocessor.used_features
    })
    
    # Устанавливаем тестовое значение для Gender
    sample_data['Gender'] = [gender_value]
    
    # Выполняем преобразование
    processed = preprocessor.transform(sample_data)
    
    # Проверяем результат кодирования
    assert processed['Gender'].iloc[0] == expected_code

def test_data_type_conversion(preprocessor):
    """
    Тестирует преобразование типов данных.
    Проверяет корректность конвертации указанных колонок.
    """
    # Создаем тестовые данные
    sample_data = pd.DataFrame({
        'Stress Level': [1, 2],  # Целочисленное значение
        'Family History': [0, 1],
        'Diabetes': [1, 0],
        'Alcohol Consumption': [3, 4],
        'Gender': ['Male', 'Female'],
        'Diet': ['Healthy', 'Unhealthy']
    })
    
    # Добавляем остальные признаки со значениями по умолчанию
    for col in preprocessor.used_features:
        if col not in sample_data.columns:
            if col in preprocessor.cat_columns:
                sample_data[col] = ['0'] * 2
            else:
                sample_data[col] = [0] * 2
    
    # Выполняем преобразование
    processed = preprocessor.transform(sample_data)
    
    # Проверяем преобразование в строковый тип
    # (преобразование должно происходить внутри препроцессора)
    assert processed['Stress Level'].dtype in [np.int64, np.float64]
    
    # Проверяем преобразование признака Diet
    assert isinstance(processed['Diet'].iloc[0], (int, float)) or \
           processed['Diet'].dtype in [np.int64, np.float64]

def test_smoke_test(preprocessor):
    """
    Дымовой тест - проверяет, что препроцессор работает
    с минимально валидным набором данных.
    """
    # Создаем минимальный валидный набор данных
    sample_data = pd.DataFrame({
        col: ['0'] * 2 if col in preprocessor.cat_columns else [0] * 2
        for col in preprocessor.used_features
    })
    
    # Выполняем преобразование
    processed = preprocessor.transform(sample_data)
    
    # Проверяем базовые условия
    assert not processed.empty
    assert processed.shape[0] == 2
    assert processed.shape[1] == len(preprocessor.used_features)
    assert not processed.isnull().any().any()

def test_performance_with_large_data(preprocessor):
    """
    Тест производительности - проверяет обработку большого объема данных.
    """
    # Создаем большой набор данных (1000 строк)
    sample_data = pd.DataFrame({
        col: np.random.choice(['0', '1'], 1000) 
        if col in preprocessor.cat_columns 
        else np.random.rand(1000)
        for col in preprocessor.used_features
    })
    
    # Выполняем преобразование и замеряем время
    processed = preprocessor.transform(sample_data)
    
    # Проверяем результат
    assert processed.shape[0] == 1000
    assert processed.shape[1] == len(preprocessor.used_features)
