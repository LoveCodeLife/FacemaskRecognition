from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

@app.route("/")
def root():
    return render_template("index.html")


@app.route("/streaming_camara")
def streaming_camara():
    return Response(generador_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')