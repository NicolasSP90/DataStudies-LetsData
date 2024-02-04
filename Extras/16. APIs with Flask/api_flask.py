# Importing Libraries
from flask import Flask, request, jsonify
from functionalities import prettytext, even_number, hasdiabetes

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

@app.route("/even_number", methods = ["POST"])
def endpoint_even_number():
    # Using the form_api.html to get the number
    sent_numb = request.form.get("number")
    even_answer = even_number(int(sent_numb))
    return f"<html><body>{even_answer}</body></html>"

@app.route("/hasdiabetes", methods = ["POST"])
def endpoint_hasdiabetes():
    # Using postman API with http://127.0.0.1:5000/hasdiabetes
    content = request.get_json()
    age = content["age"]
    weight = content["weight"]
    glucose = content["glucose"]

    diabetes = hasdiabetes(age, weight, glucose)

    return jsonify(
        age = age,
        weight = weight,
        glucose = glucose,
        has_diabetes = diabetes
    )



if __name__ == "__main__":
    app.run(debug=True)


