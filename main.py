from flask import Flask, render_template, request
import pickle

# Load ML model and vectorizer
model = pickle.load(open('spam_detection.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    message = ""
    
    if request.method == "POST":
        # Check if this is the Clear button
        if "clear" in request.form:
            message = ""
            result = None
        else:
            # Get the message from textarea
            message = request.form["message"]
            message_vector = cv.transform([message])
            prediction = model.predict(message_vector)
            result = prediction[0]  # "Spam" or "Not Spam"
    
    return render_template("index.html", result=result, message=message)

if __name__ == "__main__":
    app.run(debug=True)
