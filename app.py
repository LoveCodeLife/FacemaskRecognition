from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

@app.route("/")
def root():
    return render_template("index.html")
