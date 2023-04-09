from numpy import asarray
import imageio
from keras.utils import img_to_array
from keras.utils import load_img
from numpy import load
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot

import os
import io
from flask import Flask,jsonify,request,redirect,url_for,send_file
from base64 import encodebytes
from PIL import Image
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

SAVE_IMAGE = 'ct40.png'

# rename to targetScan.jpg later, after integration with ML model
SEND_IMAGE = "targetscan.png"

@app.route('/',methods=["POST"])
@cross_origin()
def upload_file():
  print(request.files)
  print(request.form)
  file=request.files['file'].read();
  
  with open(SAVE_IMAGE, "wb") as binary_file:
    binary_file.write(file)
  
  # ----CONVERT THE IMAGE HERE-----

def Resnet_Block(n_filters, input_layer):

  init = RandomNormal(stddev=0.02)

  Gen = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
  Gen = InstanceNormalization(axis=-1)(Gen)

  Gen = Activation('relu')(Gen)

  Gen = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(Gen)
  Gen = InstanceNormalization(axis=-1)(Gen)

  Gen = Concatenate()([Gen, input_layer])

  return Gen

def Define_Generator(ImageShape = (256,256,3), n_resnet_times = 9):

  init = RandomNormal(stddev=0.02)
  Input_Image = Input(shape = ImageShape)

  Gen = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(Input_Image)
  Gen = InstanceNormalization(axis=-1)(Gen)
  Gen = Activation('relu')(Gen)

  Gen = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(Gen)
  Gen = InstanceNormalization(axis=-1)(Gen)
  Gen = Activation('relu')(Gen)

  Gen = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(Gen)
  Gen = InstanceNormalization(axis=-1)(Gen)
  Gen = Activation('relu')(Gen)

  for ix in range(n_resnet_times):
    Gen = Resnet_Block(256, Gen)
  
  Gen = Conv2DTranspose(128, (3,3), strides = (2,2), padding= 'same', kernel_initializer=init)(Gen)
  Gen = InstanceNormalization(axis=-1)(Gen)
  Gen = Activation('relu')(Gen)

  Gen = Conv2DTranspose(64, (3,3), strides = (2,2), padding= 'same', kernel_initializer=init)(Gen)
  Gen = InstanceNormalization(axis=-1)(Gen)
  Gen = Activation('relu')(Gen)

  Gen = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(Gen)
  Gen = InstanceNormalization(axis=-1)(Gen)
  out_image = Activation('tanh')(Gen)
  # define model
  model = Model(Input_Image, out_image)
  return model

G_ModelA2B = Define_Generator()
G_ModelA2B.load_weights(r'C:\Users\tejas\Desktop\CT2MRI\BackendCode\cycleGAN-backend\G_Model_A2B_000050.h5')
G_ModelB2A = Define_Generator()
G_ModelB2A.load_weights(r'C:\Users\tejas\Desktop\CT2MRI\BackendCode\cycleGAN-backend\G_Model_B2A_000050.h5')

def Load_Image(Path, size=(256,256)):
  Data = list()
  Img = load_img(Path, target_size = size)

  Img = img_to_array(Img)

  Data.append(Img)
  
  return asarray(Data)


def ConvertScan(G_ModelA2B, G_ModelB2A, ImagePath, NameImg, size=(256,256)):

  Imagee = Load_Image(ImagePath)

  if NameImg == "CT":
    
    ImageOut = G_ModelA2B.predict(Imagee)
    ImageOut = (ImageOut + 1) / 2.0
    ImageOut = ImageOut.reshape(256,256,3)
    imageio.imwrite(SEND_IMAGE, ImageOut)
  else:
    
    ImageOut = G_ModelB2A.predict(Image)
    ImageOut = (ImageOut + 1) / 2.0
    ImageOut = ImageOut.reshape(256,256,3)
    imageio.imwrite(SEND_IMAGE, ImageOut)

print("Running -->>")
ConvertScan(G_ModelA2B,G_ModelB2A,SAVE_IMAGE,NameImg="CT")
  # encode the image in base64
  #pil_img = Image.open(SEND_IMAGE, mode='r') # reads the PIL image
  #byte_arr = io.BytesIO()
  #pil_img.save(byte_arr, format='jpeg') # convert the PIL image to byte array
  #encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64

  
  #data={
    #"source":"CT",
    #"target":"MRI",
    #"image":encoded_img
 #}


  #return jsonify(data)
