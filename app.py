import cv2
import os
import utiles
from flask import Flask, flash, redirect, render_template, request, Response, jsonify
import numpy as np
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import tensorflow as tf
from utiles import processesing
from utiles import percentage
from keras.preprocessing import image
model = load_model('model-facemask.h5')
FRAMES_VIDEO = 20.0
RESOLUCION_VIDEO = (640, 480)
# Marca de agua
# https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
UBICACION_FECHA_HORA = (0, 15)
FUENTE_FECHA_Y_HORA = cv2.FONT_HERSHEY_PLAIN
ESCALA_FUENTE = 1
COLOR_FECHA_HORA = (255, 255, 255)
GROSOR_TEXTO = 1
TIPO_LINEA_TEXTO = cv2.LINE_AA
MODEL_PATH = 'models/model.h5'
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
fourcc = cv2.VideoWriter_fourcc(*'XVID')
archivo_video = None
grabando = False
camara = cv2.VideoCapture(0)

app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def root():
    return render_template("index.html")


@app.route('/aboutus')
def plot():
    return render_template('about.html')


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


def images(img):
    image_read = []
    image1 = image.load_img(img)
    image2 = image.img_to_array(image1)
    image3 = cv2.resize(image2, (64, 64))
    image_read.append(image3)
    img_array = np.asarray(image_read)
    return img_array


def agregar_fecha_hora_frame(frame):
    cv2.putText(frame, utiles.fecha_y_hora(), UBICACION_FECHA_HORA, FUENTE_FECHA_Y_HORA,
                ESCALA_FUENTE, COLOR_FECHA_HORA, GROSOR_TEXTO, TIPO_LINEA_TEXTO, color=cv2.COLOR_YUV2RGBA_NV12)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generador_frames():
    while True:
        ok, imagen = obtener_frame_camara()
        if not ok:
            break
        else:
            # Regresar la imagen en modo de respuesta HTTP
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + imagen + b"\r\n"


def obtener_frame_camara():
    ok, frame = camara.read()
    if not ok:
        return False, None
    # agregar_fecha_hora_frame(frame)
    # Escribir en el vídeo en caso de que se esté grabando
    if grabando and archivo_video is not None:
        archivo_video.write(frame)
    # Codificar la imagen como JPG
    _, bufer = cv2.imencode(".jpg", frame)
    imagen = bufer.tobytes()

    return True, imagen


@app.route("/streaming_camara")
def streaming_camara():
    return Response(generador_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/predict", methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        img = request.files['ima'].read()
        print(img)
        npimg = np.fromstring(img, np.uint8)
# convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        cv2.imwrite("images/output.png", img)

        image3 = cv2.resize(img, (64, 64))
        image = np.expand_dims(image3, axis=0)

        imgarray = image
        print(imgarray)
        u = model.predict(imgarray)
        print("MODEL RESPONSE: ", u)
        pre = processesing(u)

        perc = percentage(u, pre)

        if pre == 0:
            print(0)
            response = "Mask ON! You are Safe"
            return render_template("result.html", predict=response, percent=str(perc)+"A%")

        if pre == 1:
            print(1)
            response = "Mask OFF! Please wear the Mask"
            return render_template("result.html", predict=response, percent=str(perc)+" B%")

        if pre == 2:
            print(2)
            response = "Mask OFF! Please wear the Mask"
            return render_template("result.html", predict=response, percent=str(perc)+" C%")

        if pre == 3:
            print(3)
            response = "Mask OFF! Please wear the Mask"
            return render_template("result.html", predict=response, percent=str(perc)+" D%")

    if request.method == 'GET':
        return render_template("upload.html")


if __name__ == '__main__':
    app.run(debug=True)
