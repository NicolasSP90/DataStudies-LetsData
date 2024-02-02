# Importing Libraries
from flask import Flask, request, jsonify
from functionalities import prettytext, even_number

app = Flask(__name__)

@app.route("/")
def home():
    return "<html><body>Testing Flask API. Endpoint Acess: /prettytext or /evennumber</body></html>"

@app.route("/prettytext")
def endpoint_prettytext():
    #http://127.0.0.1:5000/prettytext?text=[insert the text here]
    sent_text = request.args.get("text")
    decor_text = prettytext(sent_text)
    return f"<html><body>{decor_text}</body></html>"

if __name__ == "__main__":
    app.run(debug=True)


