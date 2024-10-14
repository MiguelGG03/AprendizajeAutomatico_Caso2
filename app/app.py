import lightning as L
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64

# Cargar el modelo de TensorFlow entrenado
MODEL_PATH = 'models/model.h5'
model = load_model(MODEL_PATH)

# Preprocesamiento de la imagen para TensorFlow
def preprocess_image(image):
    image = image.resize((224, 224))  # Ajustar el tamaño según tu modelo
    image = np.array(image) / 255.0  # Normalizar la imagen
    image = np.expand_dims(image, axis=0)  # Expandir dimensiones para adaptarlo al modelo
    return image

# Clase LightningApp que maneja la interfaz y el modelo
class ImagePredictionApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.prediction = None
        self.image_data = None

    def predict(self, image):
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return predicted_class

    def render(self):
        # Si la imagen ha sido subida y predicción realizada
        if self.prediction:
            return f"""
            <h2>Resultado de la predicción: {self.prediction}</h2>
            <img src="data:image/png;base64,{self.image_data}" style="max-width: 400px;" />
            """
        else:
            return """
            <h2>Sube una imagen para predecir</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Predecir</button>
            </form>
            """

    def handle_post(self, request):
        # Procesar el archivo subido
        file = request.files['file']
        image = Image.open(file.stream)

        # Convertir la imagen a base64 para mostrarla en el frontend
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        self.image_data = base64.b64encode(buffered.getvalue()).decode()

        # Realizar la predicción
        self.prediction = self.predict(image)

if __name__=='__main__':
    # Inicializar la aplicación Lightning con la clase creada
    app = L.LightningApp(ImagePredictionApp())