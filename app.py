import cv2
import utiles
from flask import Flask, render_template, Response, jsonify
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

fourcc = cv2.VideoWriter_fourcc(*'XVID')
archivo_video = None
grabando = False

camara = cv2.VideoCapture(0)

#el app pricipal
app = Flask(__name__)


@app.route("/")
def root():
    return render_template("index.html")

@app.route('/aboutus')
def plot():
    return render_template('about.html')
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

#@app.route('/predcit')
#def predict():
def agregar_fecha_hora_frame(frame):
    cv2.putText(frame, utiles.fecha_y_hora(), UBICACION_FECHA_HORA, FUENTE_FECHA_Y_HORA,
                ESCALA_FUENTE, COLOR_FECHA_HORA, GROSOR_TEXTO, TIPO_LINEA_TEXTO,color=cv2.COLOR_YUV2RGBA_NV12)



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
    #agregar_fecha_hora_frame(frame)
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

if __name__=='__main__':
    app.run(debug=True)