import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('wordnet')

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv("twitter_training.csv")

# Eliminar filas con valores NaN en la columna 'texto'
df = df.dropna(subset=['texto'])

# Limpieza de datos y preprocesamiento del texto
def preprocess_text(text):
    # Asegurarse de que el texto sea una cadena
    text = str(text)
    # Eliminar caracteres especiales y puntuación
    text = re.sub(r'[^\w\s]', '', text)
    # Convertir a minúsculas
    text = text.lower()
    # Tokenización
    tokens = word_tokenize(text)
    # Eliminación de palabras vacías
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lematización
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Aplicar preprocesamiento a la columna 'texto'
df['texto_preprocesado'] = df['texto'].apply(preprocess_text)

# Crear el vectorizador TF-IDF
vectorizer = TfidfVectorizer()
# Aplicar el vectorizador a los datos preprocesados
X = vectorizer.fit_transform(df['texto_preprocesado'])
y = df['sentimientos']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo de regresión logística
model = LogisticRegression(solver='saga')
model.fit(X_train, y_train)

# Función para predecir el sentimiento de un texto ingresado por el usuario
def predict_sentiment(text):
    # Preprocesar el texto
    preprocessed_text = preprocess_text(text)
    # Vectorizar el texto
    text_vectorized = vectorizer.transform([preprocessed_text])
    # Predecir el sentimiento
    sentiment = model.predict(text_vectorized)
    return sentiment[0]

# Texto de ejemplo para probar el modelo
sample_text = "I hate this place, it's disgusting!"

# Predecir el sentimiento del texto de ejemplo
predicted_sentiment = predict_sentiment(sample_text)
print(sample_text, "-" "Sentimiento predicho para este texto es:", predicted_sentiment)

