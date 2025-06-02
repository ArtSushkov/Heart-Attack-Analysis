from catboost import CatBoostClassifier
import pandas as pd
import joblib

class ModelWrapper:
    def __init__(self, model_path='artifacts/catboost_model.cbm', preprocessor_path='artifacts/preprocessor.pkl'):
        """
        Инициализация обертки для модели
        :param model_path: путь к файлу модели
        :param preprocessor_path: путь к артефактам предобработки
        """
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        
        # Загружаем артефакты предобработки для получения порога
        artifacts = joblib.load(preprocessor_path)
        self.default_threshold = artifacts['default_threshold']
    
    def predict_proba(self, data):
        """
        Возвращает вероятности класса 1
        :param data: DataFrame с обработанными данными
        :return: массив вероятностей
        """
        return self.model.predict_proba(data)[:, 1]
    
    def predict(self, data, threshold=None):
        """
        Возвращает бинарные предсказания с возможностью настройки порога
        :param data: DataFrame с обработанными данными
        :param threshold: порог классификации (None для порога по умолчанию)
        :return: массив бинарных предсказаний
        """
        if threshold is None:
            threshold = self.default_threshold
        
        probabilities = self.predict_proba(data)
        return (probabilities >= threshold).astype(int)
    
    def predict_to_csv(self, data, ids, output_path='predictions.csv', threshold=None):
        """
        Сохраняет предсказания в CSV-файл с выбором типа предсказания
        :param data: DataFrame с обработанными данными
        :param ids: массив идентификаторов
        :param output_path: путь для сохранения файла
        :param threshold: порог для бинарных предсказаний
        (По умолчанию порог выставляется 0.85 для достижения специфичности близкой к 0.85)
        """
        if threshold is None:
            # Используем оптимальный порог по умолчанию
            predictions = self.predict(data)
        else:
            # Используем указанный порог
            predictions = self.predict(data, threshold=threshold)
    
        result = pd.DataFrame({'id': ids, 'prediction': predictions})
        result.to_csv(output_path, index=False)
        return result
