import openai
import speech_recognition as sr
import pyttsx3
from pytube import YouTube
import requests
from bs4 import BeautifulSoup
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
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

# Función principal de prueba
def main():
    print("¡Bienvenido al asistente de consola!")
    while True:
        print("\nOpciones:")
        print("1. Obtener info de YouTube")
        print("2. Crear una imagen")
        print("3. Reconocer voz")
        print("4. Aprender de un sitio web")
        print("5. Detectar idioma")
        print("6. Traducir texto")
        print("7. Resumir texto")
        print("8. Analizar sentimiento")
        print("9. Convertir PDF a texto")
        print("10. Realizar cálculo matemático")
        print("11. Obtener clima")
        print("12. Buscar en Wikipedia")
        print("13. Enviar correo")
        print("14. Procesar imagen")
        print("15. Buscar noticias")
        print("16. Convertir a Morse")
        print("17. Calcular distancia")
        print("18. Crear un sticker")
        print("0. Salir")
        
        opcion = input("Selecciona una opción: ")
        
        if opcion == "0":
            break
        
        elif opcion == "1":
            url = input("Ingresa la URL del video de YouTube: ")
            info = obtener_info_youtube(url)
            print(f"El video '{info['title']}' fue creado por {info['author']} y tiene {info['views']} vistas.")
        
        elif opcion == "2":
            descripcion = input("Ingresa la descripción de la imagen: ")
            imagen_url = generar_imagen(descripcion)
            print(f"Aquí tienes la imagen que pediste: {imagen_url}")
        
        elif opcion == "3":
            audio_file = input("Ingresa la ruta del archivo de audio: ")
            texto = reconocer_voz(audio_file)
            print(f"El texto del audio es: {texto}")
        
        elif opcion == "4":
            url = input("Ingresa la URL del sitio web: ")
            texto = aprender_de_web(url)
            print(f"He aprendido lo siguiente del sitio web: {texto[:200]}...")  # Limitamos la cantidad de texto
        
        elif opcion == "5":
            texto = input("Ingresa el texto para detectar el idioma: ")
            idioma = detectar_idioma(texto)
            print(f"El idioma detectado es: {idioma}")
        
        elif opcion == "6":
            texto = input("Ingresa el texto a traducir: ")
            traduccion = traducir_texto(texto)
            print(f"Traducción: {traduccion}")
        
        elif opcion == "7":
            texto = input("Ingresa el texto a resumir: ")
            resumen = resumir_texto(texto)
            print(f"Resumen: {resumen}")

        elif opcion == "8":
            texto = input("Ingresa el texto para analizar el sentimiento: ")
            sentimiento = analizar_sentimiento(texto)
            print(f"Sentimiento: {sentimiento}")
        
        elif opcion == "9":
            pdf_file = input("Ingresa la ruta del archivo PDF: ")
            texto = convertir_pdf_a_texto(pdf_file)
            print(f"Texto del PDF: {texto[:200]}...")  # Limitamos la cantidad de texto
        
        elif opcion == "10":
            expresion = input("Ingresa la expresión matemática a calcular: ")
            resultado = realizar_calculo(expresion)
            print(f"Resultado: {resultado}")
        
        elif opcion == "11":
            ciudad = input("Ingresa la ciudad para obtener el clima: ")
            clima = obtener_clima(ciudad)
            print(clima)
        
        elif opcion == "12":
            consulta = input("Ingresa la consulta para buscar en Wikipedia: ")
            resumen = buscar_wikipedia(consulta)
            print(f"Información de Wikipedia: {resumen}")
        
        elif opcion == "13":
            destinatario = input("Ingresa el destinatario del correo: ")
            asunto = input("Ingresa el asunto del correo: ")
            mensaje = input("Ingresa el mensaje del correo: ")
            resultado = enviar_correo(destinatario, asunto, mensaje)
            print(resultado)
        
        elif opcion == "14":
            imagen_base64 = input("Ingresa la imagen en base64 para procesar: ")
            resultado = procesar_imagen_base64(imagen_base64)
            print(f"Resultado del procesamiento de imagen: {resultado}")

        elif opcion == "15":
            tema = input("Ingresa el tema para buscar noticias: ")
            noticias = buscar_noticias(tema)
            print(f"Noticias recientes sobre {tema}:\n{noticias}")
        
        elif opcion == "16":
            texto = input("Ingresa el texto para convertir a Morse: ")
            morse_code = convertir_a_morse(texto)
            print(f"Código Morse: {morse_code}")

        elif opcion == "17":
            coord1 = input("Ingresa la primera coordenada (latitud,longitud): ")
            coord2 = input("Ingresa la segunda coordenada (latitud,longitud): ")
            coord1 = tuple(map(float, coord1.split(',')))
            coord2 = tuple(map(float, coord2.split(',')))
            distancia = calcular_distancia(coord1, coord2)
            print(distancia)

        elif opcion == "18":
            texto = input("Ingresa el texto para crear el sticker: ")
            sticker_path = crear_sticker(texto)
            print(f"Sticker creado y guardado en: {sticker_path}")

        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()

