# app.py

from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
import openai
import speech_recognition as sr
import pyttsx3
from pytube import YouTube
import requests
from bs4 import BeautifulSoup
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os
from googletrans import Translator
import langdetect
from textblob import TextBlob
import PyPDF2
import sympy as sp
import base64
import wikipedia
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from geopy.distance import geodesic
import morse3 as morse
from newsapi import NewsApiClient
from PIL import ImageDraw, ImageFont

app = Flask(__name__)

# Configuración de la API de OpenAI
openai.api_key = 'TU_API_KEY_DE_OPENAI'

# Configuración de la API de NewsAPI
newsapi = NewsApiClient(api_key='TU_API_KEY_DE_NEWSAPI')

# Modelo de visión por computadora
vision_model = models.resnet50(pretrained=True)
vision_model.eval()

# Configuración del reconocimiento de voz
recognizer = sr.Recognizer()

# Configuración de síntesis de voz
engine = pyttsx3.init()

# Configuración del traductor
translator = Translator()

# Función para generar respuestas con GPT
def generar_respuesta(texto):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=texto,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Función para procesar imágenes en formato base64 y detectar objetos
def procesar_imagen_base64(imagen_base64):
    imagen_bytes = base64.b64decode(imagen_base64)
    imagen = Image.open(io.BytesIO(imagen_bytes))
    return procesar_imagen(imagen)

def procesar_imagen(imagen):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(imagen)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = vision_model(input_batch)

    _, pred = torch.max(output, 1)
    return pred.item()

# Función para reconocer voz y convertir a texto
def reconocer_voz(audio_file):
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "No se pudo entender el audio."
    except sr.RequestError:
        return "Error con el servicio de reconocimiento de voz."

# Función para convertir texto a voz
def texto_a_voz(texto):
    engine.say(texto)
    engine.runAndWait()

# Función para obtener información de un creador en YouTube
def obtener_info_youtube(url):
    yt = YouTube(url)
    info = {
        "title": yt.title,
        "author": yt.author,
        "views": yt.views
    }
    return info

# Función para hacer web scraping y aprender de sitios web
def aprender_de_web(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

# Función para generar imágenes (Placeholder, se necesita implementar con una API real)
def generar_imagen(descripcion):
    return "https://fakeimage.com/generated_image.png"  # Sustituye con la lógica real para generar imágenes

# Función para detectar el idioma de un texto
def detectar_idioma(texto):
    idioma = langdetect.detect(texto)
    return idioma

# Función para traducir texto a un idioma especificado (por defecto al inglés)
def traducir_texto(texto, dest="en"):
    traduccion = translator.translate(texto, dest=dest)
    return traduccion.text

# Función para resumir textos largos usando OpenAI
def resumir_texto(texto):
    prompt = f"Resume el siguiente texto: {texto}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Función para analizar el sentimiento de un texto
def analizar_sentimiento(texto):
    blob = TextBlob(texto)
    sentimiento = blob.sentiment
    return f"Polaridad: {sentimiento.polarity}, Subjetividad: {sentimiento.subjectivity}"

# Función para convertir PDF a texto
def convertir_pdf_a_texto(pdf_file):
    reader = PyPDF2.PdfFileReader(pdf_file)
    texto = ""
    for page in range(reader.numPages):
        texto += reader.getPage(page).extract_text()
    return texto

# Función para realizar cálculos matemáticos
def realizar_calculo(expresion):
    try:
        resultado = sp.sympify(expresion)
        return str(resultado)
    except sp.SympifyError:
        return "Expresión matemática no válida."

# Función para obtener información meteorológica
def obtener_clima(ciudad):
    api_key = 'TU_API_KEY_DE_CLIMA'
    url = f"http://api.openweathermap.org/data/2.5/weather?q={ciudad}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    if data["cod"] == 200:
        main = data["main"]
        temperatura = main["temp"]
        clima = data["weather"][0]["description"]
        return f"El clima en {ciudad} es {clima} con una temperatura de {temperatura}°C."
    else:
        return "Ciudad no encontrada."

# Función para buscar información en Wikipedia
def buscar_wikipedia(consulta):
    resumen = wikipedia.summary(consulta, sentences=3, auto_suggest=False)
    return resumen

# Función para enviar un correo electrónico
def enviar_correo(destinatario, asunto, mensaje):
    remitente = 'TU_CORREO@gmail.com'
    password = 'TU_CONTRASEÑA'

    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = asunto

    msg.attach(MIMEText(mensaje, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(remitente, password)
        text = msg.as_string()
        server.sendmail(remitente, destinatario, text)
        server.quit()
        return "Correo enviado con éxito."
    except Exception as e:
        return f"Fallo al enviar correo: {str(e)}"

# Función para buscar noticias
def buscar_noticias(tema):
    top_headlines = newsapi.get_top_headlines(q=tema, language='es', country='co')
    if top_headlines['totalResults'] > 0:
        noticias = top_headlines['articles']
        respuesta = ""
        for noticia in noticias[:3]:  # Limitar a las 3 primeras noticias
            respuesta += f"Título: {noticia['title']}\nDescripción: {noticia['description']}\nURL: {noticia['url']}\n\n"
        return respuesta
    else:
        return "No se encontraron noticias sobre este tema."

# Función para convertir texto a código Morse
def convertir_a_morse(texto):
    morse_code = morse.Morse(texto)
    return str(morse_code)

# Función para calcular la distancia entre dos coordenadas geográficas
def calcular_distancia(coord1, coord2):
    distancia = geodesic(coord1, coord2).kilometers
    return f"La distancia entre los puntos es de {distancia:.2f} kilómetros."

# Función para crear stickers
def crear_sticker(texto, color_fondo="white", color_texto="black"):
    ancho, alto = 400, 200
    imagen = Image.new('RGB', (ancho, alto), color_fondo)
    dibujador = ImageDraw.Draw(imagen)
    
    # Selecciona una fuente para el texto
    fuente = ImageFont.load_default()
    
    # Calcula el tamaño del texto y su posición
    ancho_texto, alto_texto = dibujador.textsize(texto, font=fuente)
    x = (ancho - ancho_texto) / 2
    y = (alto - alto_texto) / 2
    
    # Dibuja el texto en la imagen
    dibujador.text((x, y), texto, font=fuente, fill=color_texto)
    
    # Guarda la imagen
    sticker_path = 'sticker.png'
    imagen.save(sticker_path)
    return sticker_path

# Endpoint principal para manejar las solicitudes de WhatsApp
@app.route("/whatsapp", methods=['POST'])
def whatsapp():
    msg = request.form.get('Body')
    resp = MessagingResponse()

    if 'youtube.com' in msg or 'youtu.be' in msg:
        info = obtener_info_youtube(msg)
        respuesta = f"El video '{info['title']}' fue creado por {info['author']} y tiene {info['views']} vistas."
        resp.message(respuesta)
    
    elif 'crea una imagen' in msg:
        descripcion = msg.replace('crea una imagen', '').strip()
        imagen_url = generar_imagen(descripcion)
        resp.message(f"Aquí tienes la imagen que pediste: {imagen_url}")
    
    elif 'reproduce este audio' in msg:
        audio_url = obtener_url_audio(msg)  # Necesitarás implementar esta función
        texto = reconocer_voz(audio_url)
        resp.message(f"El texto del audio es: {texto}")
    
    elif 'aprende de' in msg:
        url = msg.replace('aprende de', '').strip()
        texto = aprender_de_web(url)
        resp.message(f"He aprendido lo siguiente del sitio web: {texto[:200]}...")  # Limitamos la cantidad de texto
    
    elif 'detecta el idioma' in msg:
        texto = msg.replace('detecta el idioma', '').strip()
        idioma = detectar_idioma(texto)
        resp.message(f"El idioma detectado es: {idioma}")
    
    elif 'traduce' in msg:
        texto = msg.replace('traduce', '').strip()
        traduccion = traducir_texto(texto)
        resp.message(f"Traducción: {traduccion}")
    
    elif 'resume' in msg:
        texto = msg.replace('resume', '').strip()
        resumen = resumir_texto(texto)
        resp.message(f"Resumen: {resumen}")

    elif 'analiza el sentimiento' in msg:
        texto = msg.replace('analiza el sentimiento', '').strip()
        sentimiento = analizar_sentimiento(texto)
        resp.message(f"Sentimiento: {sentimiento}")
    
    elif 'convierte PDF' in msg:
        # Necesitarás implementar lógica para cargar el PDF. Por ahora, se simula con un archivo existente.
        pdf_path = "ruta/a/tu/archivo.pdf"  # Simula el archivo PDF
        texto = convertir_pdf_a_texto(pdf_path)
        resp.message(f"Texto del PDF: {texto[:200]}...")  # Limitamos la cantidad de texto
    
    elif 'calcula' in msg:
        expresion = msg.replace('calcula', '').strip()
        resultado = realizar_calculo(expresion)
        resp.message(f"Resultado: {resultado}")
    
    elif 'clima en' in msg:
        ciudad = msg.replace('clima en', '').strip()
        clima = obtener_clima(ciudad)
        resp.message(clima)
    
    elif 'busca en Wikipedia' in msg:
        consulta = msg.replace('busca en Wikipedia', '').strip()
        resumen = buscar_wikipedia(consulta)
        resp.message(f"Información de Wikipedia: {resumen}")
    
    elif 'envía un correo' in msg:
        partes = msg.replace('envía un correo', '').strip().split(';')
        destinatario = partes[0].strip()
        asunto = partes[1].strip()
        mensaje = partes[2].strip()
        resultado = enviar_correo(destinatario, asunto, mensaje)
        resp.message(resultado)
    
    elif 'procesa imagen' in msg:
        imagen_base64 = msg.replace('procesa imagen', '').strip()
        resultado = procesar_imagen_base64(imagen_base64)
        resp.message(f"Resultado del procesamiento de imagen: {resultado}")

    elif 'busca noticias de' in msg:
        tema = msg.replace('busca noticias de', '').strip()
        noticias = buscar_noticias(tema)
        resp.message(f"Noticias recientes sobre {tema}:\n{noticias}")
    
    elif 'convierte a morse' in msg:
        texto = msg.replace('convierte a morse', '').strip()
        morse_code = convertir_a_morse(texto)
        resp.message(f"Código Morse: {morse_code}")

    elif 'calcula distancia entre' in msg:
        coords = msg.replace('calcula distancia entre', '').strip().split(';')
        coord1 = tuple(map(float, coords[0].split(',')))
        coord2 = tuple(map(float, coords[1].split(',')))
        distancia = calcular_distancia(coord1, coord2)
        resp.message(distancia)

    elif 'crea un sticker' in msg:
        texto = msg.replace('crea un sticker', '').strip()
        sticker_path = crear_sticker(texto)
        with open(sticker_path, 'rb') as f:
            resp.message('Aquí tienes tu sticker:', media_url=f'static/{sticker_path}')
    
    else:
        respuesta = generar_respuesta(msg)
        resp.message(respuesta)

    return str(resp)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
