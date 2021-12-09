from flask import Flask, render_template, request, redirect, url_for
from caption_generator import generator


app = Flask(__name__)

x = 43534


@app.route("/")
# @app.route("/index")
# def index():
#     return render_template("index.html", x = x)

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/model")
def model():
    return render_template("model.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/download_picture", methods=["POST"])
def download_picture():
    if request.method == "POST":
        file = request.files['file1']
        # print(type(file))
        # print(file)
        # generator(file)
        name = "Saif"
    
    return render_template("home.html", name=name)



if __name__ == "__main__":
    app.run(debug=True)
