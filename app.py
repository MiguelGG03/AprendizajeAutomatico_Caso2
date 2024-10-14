import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Cargar el modelo entrenado
MODEL_PATH = 'models/model.h5'
model = load_model(MODEL_PATH)

# Preprocesar la imagen antes de la predicción
def preprocess_image(image):
    image = image.resize((128, 128))  # Ajustar el tamaño según tu modelo
    image = np.array(image) / 255.0  # Normalizar la imagen
    image = np.expand_dims(image, axis=0)  # Añadir una dimensión extra (batch_size, altura, anchura, canales)
    return image

# Función para hacer la predicción
def predict_image(image):
    image = preprocess_image(image)  # Preprocesar la imagen
    prediction = model.predict(image)  # Realizar la predicción
    predicted_class = np.argmax(prediction, axis=1)[0]  # Obtener la clase predicha
    benigno, maligno = prediction[0]  # Obtener la probabilidad de cada clase

    benigno = benigno * 100
    maligno = maligno * 100

    # Redondear a 2 decimales
    benigno = round(benigno, 2)
    maligno = round(maligno, 2)

    # Devolver la clase más probable y las probabilidades formateadas
    return f'La clase más probable es {"Benigno" if predicted_class == 0 else "Maligno"}\nBenigno : {benigno:.2f}% ; Maligno : {maligno:.2f}%'

# Crear la interfaz en Gradio
iface = gr.Interface(
    fn=predict_image,  # Función que realiza la predicción
    inputs=gr.Image(type="pil"),  # Entrada: imagen en formato PIL
    outputs="text",  # Salida: texto con la clase predicha
    title="Clasificador de Imágenes con CNN",
    description="Sube una imagen para obtener una predicción."
)

# Lanzar la interfaz
iface.launch()
