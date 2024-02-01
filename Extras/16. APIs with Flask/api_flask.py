# Importing Libraries
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "<html><body>Testing Flask API. Endpoint Acess: /prettytext or /evennumber</body></html>"

@app.route("/prettytext")
def endpoint_prettytext():
    sent_text = request.args.get("text")
    decor_text = prettytext(sent_text)
    return f"<html><body>{decor_text}</body></html>"

if __name__ == "__main__":
    app.run(debug=True)

