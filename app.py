import os
import io
from flask import Flask,jsonify,request,redirect,url_for,send_file
from base64 import encodebytes
from PIL import Image
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

SAVE_IMAGE = "sourceScan.jpg"

# rename to targetScan.jpg later, after integration with ML model
SEND_IMAGE = "sourceScan.jpg"

@app.route('/',methods=["POST"])
@cross_origin()
def upload_file():
  print(request.files)
  print(request.form)
  file=request.files['file'].read();
  
  with open(SAVE_IMAGE, "wb") as binary_file:
    binary_file.write(file)
  
  # ----CONVERT THE IMAGE HERE-----


  # encode the image in base64
  pil_img = Image.open(SEND_IMAGE, mode='r') # reads the PIL image
  byte_arr = io.BytesIO()
  pil_img.save(byte_arr, format='jpeg') # convert the PIL image to byte array
  encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64

  
  data={
    "source":"CT",
    "target":"MRI",
    "image":encoded_img
  }


  return jsonify(data)
