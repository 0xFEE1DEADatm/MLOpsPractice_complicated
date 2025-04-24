# MLOps Practice Project

Этот проект демонстрирует развёртывание ML-модели с использованием инструментов MLOps: MLflow, Docker и FastAPI. Он включает тренировку модели, логирование экспериментов, API-сервер для инференса и документацию Swagger.

---
## Быстрый старт

1.
   git clone https://github.com/yourname/mlflow-project.git
   cd mlflow-project

2. 
    docker-compose up --build

## Установка зависимостей
   
    pip install -r requirements.txt

## Обучение модели и логирование в MLflow
    python log_and_train.py

    MLflow UI: http://localhost:5000

## Запуск сервера предсказаний
    python -m uvicorn serve_model:app --reload
    
Swagger UI доступен по адресу http://127.0.0.1:8000/docs

**Примеры тела запроса для /predict**

POST /predict:
    {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
    }
    
Ответ:
    {
    "prediction": 0
    }


POST /predict:
    {
    "sepal_length": 5.8,
    "sepal_width": 2.7,
    "petal_length": 5.1,
    "petal_width": 1.9
    }
    
Ответ:
    {
    "prediction": 2
    }


POST /predict:
    {
    "sepal_length": "invalid",
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
    }
    
Ответ:
    {
    "detail": [
        {
        "type": "float_parsing",
        "loc": [
            "body",
            "sepal_length"
        ],
        "msg": "Input should be a valid number, unable to parse string as a number",
        "input": "invalid"
        }
    ]
    }
