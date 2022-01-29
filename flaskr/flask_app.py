from flask import Flask, render_template,request
import sys
import os

sys.path.append('../model')
sys.path.append('../data')

from recommend_tags import recommend_tags

path = os.getcwd()
app = Flask(__name__)
infer = recommend_tags('../model/model.pth')

@app.route("/")
def home():
  return render_template("index.html")

@app.route("/", methods=['GET'])
def upload_data_get():
  
  return render_template("index.html")

@app.route("/", methods=['POST'])
def upload_data():
  question = request.form['question-input']
  question = question.strip(" ")
  recommended_tags = infer.get_tag(question)
  return render_template("index.html", recommended_tags=recommended_tags)
  
if __name__ == "__main__":
  # try:
  #   port = int(sys.argv[1])
  # except Exception as e:
  #   port = 80
  app.run(debug = True)