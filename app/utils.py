import joblib
import pandas as pd
import os

def load_data(file_path):
    """Загружает данные из CSV-файла"""
    return pd.read_csv(file_path)

def save_artifacts(model, preprocessor, cat_columns, num_columns, threshold, 
                   model_path, preprocessor_path):
    """
    Сохраняет артефакты модели и предобработки
    :param model: обученная модель
    :param preprocessor: ColumnTransformer
    :param cat_columns: список категориальных признаков
    :param num_columns: список числовых признаков
    :param threshold: оптимальный порог классификации
    :param model_path: путь для сохранения модели
    :param preprocessor_path: путь для сохранения артефактов предобработки
    """
    # Сохранение модели
    model.save_model(model_path)

    # Извлекаем пайплайн для категориальных признаков
    cat_pipeline = preprocessor.named_transformers_['ord'] 
    
    # Подготовка и сохранение артефактов предобработки
    artifacts = {
        'ordinal_encoder': cat_pipeline.named_steps['ord'],
        'simple_imputer_before_ord': cat_pipeline.named_steps['simpleImputer_before_ord'],
        'simple_imputer_after_ord': cat_pipeline.named_steps['simpleImputer_after_ord'],
        'cat_columns': cat_columns,
        'num_columns': num_columns,
        'used_features': cat_columns + num_columns,
        'default_threshold': threshold
    }
    
    # Создание директории, если необходимо
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    joblib.dump(artifacts, preprocessor_path)
