import joblib
import pandas as pd
import numpy as np 

class DataPreprocessor:
    def __init__(self, artifacts_path='artifacts/preprocessor.pkl'):
        """
        Инициализация предобработчика данных
        :param artifacts_path: путь к файлу с артефактами предобработки
        """
        artifacts = joblib.load(artifacts_path)
        self.ordinal_encoder = artifacts['ordinal_encoder']
        self.simple_imputer_before_ord = artifacts['simple_imputer_before_ord']
        self.simple_imputer_after_ord = artifacts['simple_imputer_after_ord']
        self.cat_columns = artifacts['cat_columns']
        self.num_columns = artifacts['num_columns']
        self.used_features = artifacts['used_features']

        # Если в артефактах есть Scaler, загружаем его
        if 'num_scaler' in artifacts:
            self.num_scaler = artifacts['num_scaler']
        else:
            self.num_scaler = None

    def transform(self, df):
        """
        Выполняет полную предобработку данных
        :param df: DataFrame с исходными данными
        :return: DataFrame с обработанными данными
        """
        # 1. Проверка наличия необходимых признаков
        self._check_features(df)
        
        # 2. Копирование данных для обработки
        df_processed = df[self.used_features].copy()
        
        # 3. Удаление строк с пропусками в ЛЮБОМ столбце
        df_processed = self._remove_rows_with_missing_values(df_processed)
        
        # 4. Преобразование типов данных
        df_processed = self._convert_dtypes(df_processed)
        
        # 5. Обработка категориальных признаков
        df_processed[self.cat_columns] = self.simple_imputer_before_ord.transform(df_processed[self.cat_columns])
        df_processed[self.cat_columns] = self.ordinal_encoder.transform(df_processed[self.cat_columns])
        df_processed[self.cat_columns] = self.simple_imputer_after_ord.transform(df_processed[self.cat_columns])
        
        # 6. Обработка числовых признаков (если есть скалер)
        if self.num_scaler is not None and self.num_columns:
            df_processed[self.num_columns] = self.num_scaler.transform(df_processed[self.num_columns])

        return df_processed

    def _remove_rows_with_missing_values(self, df):
        """
        Удаляет строки с пропущенными значениями в любом столбце
        :param df: DataFrame для обработки
        :return: Очищенный DataFrame
        """
        # Заменяем бесконечности на NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Удаляем строки с пропусками
        initial_count = len(df)
        df_cleaned = df.dropna()
        removed_count = initial_count - len(df_cleaned)
        
        # Проверка, остались ли данные
        if df_cleaned.empty:
            raise ValueError("После удаления строк с пропусками данные пусты!")
        
        print(f"Удалено строк с пропусками: {removed_count}/{initial_count}")
        return df_cleaned

    def _check_features(self, df):
        """Проверяет наличие всех необходимых признаков"""
        missing = [col for col in self.used_features if col not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют обязательные признаки: {missing}")

    def _convert_dtypes(self, df):
        """Выполняет преобразование типов данных"""
        # Колонки, которые нужно преобразовать в int -> str
        int_to_str = ['Stress Level', 'Family History', 'Diabetes', 'Alcohol Consumption']
        for col in int_to_str:
            if col in df.columns:
                # Безопасное преобразование в целое число
                try:
                    df[col] = df[col].astype(int).astype(str)
                except Exception as e:
                    # Для проблемных значений используем строку 'error'
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-999).astype(int).astype(str)
        
        # Обработка колонки Gender
        if 'Gender' in df.columns:
            # Заменяем пропуски на специальное значение
            df['Gender'] = df['Gender'].fillna('Unknown')
            df['Gender'] = df['Gender'].map({
                'Male': '0', 
                'Female': '1',
                'Unknown': '-999'  # Специальное значение для пропусков
            })
        
        # Обработка колонки Diet
        if 'Diet' in df.columns:
            df['Diet'] = df['Diet'].fillna('Unknown').astype(str)
            
        return df
