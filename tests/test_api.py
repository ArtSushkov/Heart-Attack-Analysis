"""
Модуль тестирования API для FastAPI-приложения
Использует pytest и TestClient из FastAPI для проверки конечных точек
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

# Инициализация тестового клиента для FastAPI приложения
client = TestClient(app)

def test_health_check():
    """
    Тестирование эндпоинта проверки работоспособности (/health).
    
    Цели:
    1. Проверить доступность эндпоинта
    2. Убедиться в корректности формата ответа
    3. Подтвердить статус сервиса 'OK'
    """
    response = client.get("/health")
    # Проверка HTTP статус-кода
    assert response.status_code == 200
    # Проверка структуры и значения JSON ответа
    assert response.json()["status"] == "OK"

def test_predict_endpoint():
    """
    Тестирование основного predict эндпоинта (/predict).
    
    Цели:
    1. Проверить обработку CSV-файла
    2. Убедиться в корректности формата ответа
    3. Проверить наличие обязательных полей в ответе
    4. Подтвердить правильность типов данных
    """
    # Использование тестового файла из директории tests/
    with open("tests/test_sample.csv", "rb") as f:
        # Отправка POST-запроса с файлом
        response = client.post(
            "/predict",
            files={"file": ("test_sample.csv", f, "text/csv")}
        )
    
    # Проверка успешного выполнения запроса
    assert response.status_code == 200
    data = response.json()
    
    # Проверка наличия ключевых полей в ответе
    assert "predictions" in data
    # Проверка типа данных predictions
    assert isinstance(data["predictions"], list)
    # Подтверждение наличия хотя бы одного предсказания
    assert len(data["predictions"]) > 0
    
    # Проверка дополнительных аналитических полей
    assert "high_risk_count" in data  # Количество высокорисковых случаев
    assert "predictions_count" in data  # Общее количество предсказаний
    assert "status" in data  # Статус операции
