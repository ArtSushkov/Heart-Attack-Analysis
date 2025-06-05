from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from app.preprocessing import DataPreprocessor
from app.model import ModelWrapper
from fastapi.responses import JSONResponse

app = FastAPI()

# Инициализация компонентов
preprocessor = DataPreprocessor('artifacts/preprocessor.pkl')
model = ModelWrapper('artifacts/catboost_model.cbm', 
                    'artifacts/preprocessor.pkl')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
        predictions = model.predict(processed_data)
        
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

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "model_loaded": True,
        "service": "Heart Attack Risk Prediction API",
        "version": "1.0.0"
    }
