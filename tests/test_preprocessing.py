"""
Модуль для тестирования класса DataPreprocessor из app.preprocessing
Проверяет корректность обработки данных перед передачей в модель
"""

import pytest
import pandas as pd
import numpy as np
from app.preprocessing import DataPreprocessor

# Фикстура для инициализации препроцессора один раз для всех тестов
@pytest.fixture(scope="module")
def preprocessor():
    # Загрузка препроцессора из артефакта
    return DataPreprocessor('artifacts/preprocessor.pkl')

def test_preprocessor_initialization(preprocessor):
    """
    Тестирование инициализации препроцессора.
    
    Цели:
    1. Проверить загрузку конфигурации
    2. Убедиться в наличии всех необходимых атрибутов
    3. Проверить корректность типов данных параметров
    """
    # Проверка категориальных признаков
    assert preprocessor.cat_columns is not None
    assert isinstance(preprocessor.cat_columns, list)
    assert len(preprocessor.cat_columns) > 0
    
    # Проверка числовых признаков
    assert preprocessor.num_columns is not None
    assert isinstance(preprocessor.num_columns, list)
    
    # Проверка используемых фичей
    assert preprocessor.used_features is not None
    assert len(preprocessor.used_features) > 0
    
    # Проверка наличия компонентов обработки
    assert hasattr(preprocessor, 'ordinal_encoder')  # Кодировщик категорий
    assert hasattr(preprocessor, 'simple_imputer_before_ord')  # Импьютер для пропусков
    assert hasattr(preprocessor, 'simple_imputer_after_ord')  # Импьютер после кодирования

def test_feature_check(preprocessor):
    """
    Тест проверки обязательных признаков.
    
    Цели:
    1. Проверить реакцию на отсутствие обязательных фичей
    2. Убедиться в корректности сообщения об ошибке
    """
    # Создание датафрейма с неполным набором фичей
    sample_data = pd.DataFrame({'Age': [30]})
    
    # Проверка генерации исключения при трансформации
    with pytest.raises(ValueError) as excinfo:
        preprocessor.transform(sample_data)
    
    # Проверка содержания сообщения об ошибке
    assert "Отсутствуют обязательные признаки" in str(excinfo.value)

def test_categorical_processing(preprocessor):
    """
    Тестирование обработки категориальных признаков.
    
    Цели:
    1. Проверить кодирование категорий
    2. Убедиться в обработке пропусков
    3. Проверить преобразование типов данных
    """
    # Создание тестовых данных с категориальными фичами
    sample_data = pd.DataFrame({
        col: ['0'] * 2 if col in preprocessor.cat_columns else [0] * 2
        for col in preprocessor.used_features
    })
    
    # Добавление конкретных значений для проверки
    sample_data['Gender'] = ['Male', 'Female']
    sample_data['Stress Level'] = ['2', '3']
    
    # Трансформация данных
    processed = preprocessor.transform(sample_data)
    
    # Проверка кодирования гендера
    assert processed['Gender'].tolist() == [0, 1]
    # Проверка преобразования типа для Stress Level
    assert processed['Stress Level'].dtype in [np.int64, np.float64]
    # Проверка отсутствия пропусков в категориальных фичах
    assert not processed[preprocessor.cat_columns].isnull().any().any()

def test_numerical_processing(preprocessor):
    """
    Тестирование обработки числовых признаков.
    
    Цели:
    1. Проверить масштабирование (если используется)
    2. Убедиться в сохранении фичей
    3. Проверить обработку без изменений (если скейлер отключен)
    """
    # Создание тестовых данных с числовыми фичами
    sample_data = pd.DataFrame({
        col: ['0'] * 3 if col in preprocessor.cat_columns else [0] * 3
        for col in preprocessor.used_features
    })
    
    # Добавление конкретных числовых значений
    sample_data['Age'] = [25, 30, 35]
    sample_data['BMI'] = [22.1, 24.5, 26.8]
    
    processed = preprocessor.transform(sample_data)
    
    # Проверка наличия фичей после преобразования
    assert 'Age' in processed.columns
    assert 'BMI' in processed.columns
    
    # Ветвление проверки в зависимости от использования скейлера
    if preprocessor.num_scaler:
        # Проверка масштабирования в диапазон [0, 1]
        assert processed['Age'].min() >= 0
        assert processed['Age'].max() <= 1
        assert processed['BMI'].min() >= 0
        assert processed['BMI'].max() <= 1
    else:
        # Проверка сохранения оригинальных значений
        assert processed['Age'].tolist() == [25, 30, 35]
        assert processed['BMI'].tolist() == [22.1, 24.5, 26.8]

def test_missing_values_handling(preprocessor):
    """
    Тестирование обработки пропущенных значений.
    
    Цели:
    1. Проверить удаление строк с пропусками
    2. Убедиться в корректности импьютинга
    """
    # Создание данных с пропусками
    sample_data = pd.DataFrame({
        col: [None, 'valid'] if col in preprocessor.cat_columns else [None, 0]
        for col in preprocessor.used_features
    })
    
    processed = preprocessor.transform(sample_data)
    
    # Проверка отсутствия пропусков после обработки
    assert not processed.isnull().any().any()
    # Проверка удаления строки с пропусками
    assert len(processed) == 1

# Параметризация для проверки разных вариантов гендера
@pytest.mark.parametrize("gender_value,expected_code", [
    ('Male', 0),
    ('Female', 1),
    ('male', 0),    # Проверка нижнего регистра
    ('female', 1),  # Проверка нижнего регистра
    ('M', 0),       # Проверка сокращенной формы
    ('F', 1),       # Проверка сокращенной формы
])
def test_gender_encoding(preprocessor, gender_value, expected_code):
    """
    Параметризованный тест кодирования признака 'Gender'.
    
    Цели:
    1. Проверить обработку разных форматов ввода
    2. Убедиться в консистентности кодирования
    """
    # Создание датафрейма с одной строкой
    sample_data = pd.DataFrame({
        col: ['0'] for col in preprocessor.used_features
    }).head(1)
    
    # Установка тестового значения гендера
    sample_data['Gender'] = [gender_value]
    
    # Проверка соответствия кодировки ожидаемому значению
    processed = preprocessor.transform(sample_data)
    assert processed['Gender'].iloc[0] == expected_code

def test_data_type_conversion(preprocessor):
    """
    Тестирование преобразования типов данных.
    
    Цели:
    1. Проверить конвертацию категорий в числовой формат
    2. Убедиться в корректности типов после преобразования
    """
    # Создание данных смешанных типов
    sample_data = pd.DataFrame({
        'Stress Level': [1, 2],
        'Family History': [0, 1],
        'Diabetes': [1, 0],
        'Alcohol Consumption': [3, 4],
        'Gender': ['Male', 'Female'],
        'Diet': ['Healthy', 'Unhealthy']
    })
    
    # Добавление недостающих фичей со значениями по умолчанию
    for col in preprocessor.used_features:
        if col not in sample_data.columns:
            if col in preprocessor.cat_columns:
                sample_data[col] = ['0'] * 2
            else:
                sample_data[col] = [0] * 2
    
    processed = preprocessor.transform(sample_data)
    
    # Проверка числовых типов после преобразования
    assert processed['Stress Level'].dtype in [np.int64, np.float64]
    assert processed['Diet'].dtype in [np.int64, np.float64]

def test_smoke_test(preprocessor):
    """
    Дымовой тест (smoke test) базовой функциональности.
    
    Цели:
    1. Проверить работу препроцессора на минимальных данных
    2. Убедиться в отсутствии критических ошибок
    """
    # Создание минимального валидного датасета
    sample_data = pd.DataFrame({
        col: ['0'] * 2 if col in preprocessor.cat_columns else [0] * 2
        for col in preprocessor.used_features
    })
    
    processed = preprocessor.transform(sample_data)
    
    # Проверка базовых свойств результата
    assert not processed.empty
    assert processed.shape[0] == 2
    assert processed.shape[1] == len(preprocessor.used_features)
    assert not processed.isnull().any().any()

def test_performance_with_large_data(preprocessor):
    """
    Тест производительности на большом наборе данных.
    
    Цели:
    1. Проверить обработку 1000 записей
    2. Убедиться в отсутствии ошибок памяти
    3. Проверить соответствие размерности вывода
    """
    # Генерация синтетических данных (1000 строк)
    sample_data = pd.DataFrame({
        col: np.random.choice(['0', '1'], 1000) 
        if col in preprocessor.cat_columns 
        else np.random.rand(1000)
        for col in preprocessor.used_features
    })
    
    processed = preprocessor.transform(sample_data)
    
    # Проверка сохранения структуры данных
    assert processed.shape[0] == 1000
    assert processed.shape[1] == len(preprocessor.used_features)
