from src.data_processing import load_data, preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model

# Cargar y preprocesar los datos
data = load_data('data/iris.csv')
X_train, X_test, y_train, y_test = preprocess_data(data)

# Entrenar el modelo
model = train_model(X_train, y_train)

# Evaluar el modelo
accuracy, report = evaluate_model(model, X_test, y_test)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
