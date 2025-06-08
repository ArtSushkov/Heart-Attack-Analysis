from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
import pandas as pd
from app.preprocessing import DataPreprocessor
from app.model import ModelWrapper
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import os

app = FastAPI()

# Инициализация компонентов
preprocessor = DataPreprocessor('artifacts/preprocessor.pkl')
model = ModelWrapper('artifacts/catboost_model.cbm', 
                    'artifacts/preprocessor.pkl')

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = None):
    try:
        # Проверка формата файла
        if not file.filename.endswith('.csv'):
            raise HTTPException(400, detail="Неверный формат файла. Поддерживаются только CSV-файлы.")
        
        # Загрузка данных
        df = pd.read_csv(file.file)
        
        # Проверка наличия обязательных колонок
        if 'id' not in df.columns:
            raise HTTPException(400, detail="CSV должен содержать колонку 'id'")
        
        # Сохранение ID
        ids = df['id']
        
        # Предобработка и предсказание
        processed_data, mask = preprocessor.transform(df.drop('id', axis=1))
        predictions = model.predict(processed_data, threshold=threshold)
        
        # Фильтруем ID по маске валидных строк
        valid_ids = ids[mask]
        
        # Формирование структурированного ответа
        results = []
        for patient_id, prediction in zip(valid_ids, predictions):
            results.append({
                "patient_id": int(patient_id),
                "prediction": int(prediction),
                "risk_level": "высокий" if prediction == 1 else "низкий",
                "explanation": "Рекомендуется консультация кардиолога" if prediction == 1 
                              else "Риск в пределах нормы"
            })
        
        return JSONResponse(content={
            "status": "success",
            "predictions_count": len(results),
            "high_risk_count": sum(1 for p in predictions if p == 1),
            "predictions": results
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": str(e)
            }
        )

@app.post("/predict_to_csv")
async def predict_to_csv(
	file: UploadFile = File(...),
	threshold: float = None,
	background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        # Проверка формата файла
        if not file.filename.endswith('.csv'):
            raise HTTPException(400, detail="Неверный формат файла. Поддерживаются только CSV-файлы.")
        
        # Загрузка данных
        df = pd.read_csv(file.file)
        
        # Проверка наличия обязательных колонок
        if 'id' not in df.columns:
            raise HTTPException(400, detail="CSV должен содержать колонку 'id'")
        
        # Сохранение ID
        ids = df['id']
        
        # Предобработка и предсказание
        processed_data, mask = preprocessor.transform(df.drop('id', axis=1))
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp_path = tmp.name
        
        # Сохраняем предсказания в файл
        model.predict_to_csv(processed_data, ids[mask], output_path=tmp_path, threshold=threshold)

	# Добавляем задачу на удаление файла после отправки
        background_tasks.add_task(lambda: os.unlink(tmp_path))
        
        # Возвращаем файл как ответ
        return FileResponse(
            tmp_path,
            media_type='text/csv',
            filename="predictions.csv"
        )
    
    except Exception as e:
        # Удаляем временный файл при ошибке
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": str(e)
            }
        )

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "model_loaded": True,
        "service": "Heart Attack Risk Prediction API",
        "version": "1.0.0"
    }
