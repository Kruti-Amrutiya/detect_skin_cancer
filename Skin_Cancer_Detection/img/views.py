from django.shortcuts import render,redirect
from .form import ImageForm
from .models import Image
from django.conf import settings
from tensorflow import keras
from keras.models import load_model
import cv2
import tensorflow as tf
import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image
np.random.seed(123)
import logging
from django.conf import settings

fmt = getattr(settings, 'LOG_FORMAT', None)
lvl = getattr(settings, 'LOG_LEVEL', logging.DEBUG)

logging.basicConfig(format=fmt, level=lvl)
logging.debug("Logging started on %s for %s" % (logging.root.name, logging.getLevelName(lvl)))
global img_url 
img_url = ''
def index(request):
    logging.debug("Enter In Index")
    if request.method == "POST":
        form=ImageForm(data=request.POST,files=request.FILES)
        logging.debug("File:{}".format(request.FILES['image']))
        img_url = request.FILES['image']
        if img_url is not None:
            mod = tf.keras.models.load_model(r'C:\Users\DELL\projects\testing\skin_cancer_mnist_model.h5')
            #image_path = r"C:\Users\DELL\projects\skin_lesion_project\media\images\ISIC_0024308.jpg"
            classes = {0:'Actinic keratoses',1:'Basal cell carcinoma',2:'Benign keratosis-like lesions',3:'Dermatofibroma',4:'Melanocytic nevi',
            5:'Melanoma',6:'Vascular lesions'}

            def detect_skin_lesions(url):
                print("Enter in main function")
                img = mpimg.imread(url)
                img = cv2.resize(img,(75,100))
                prediction_output = mod.predict_classes(img.reshape([-1,75,100,3]))
                predicted_class = classes[prediction_output[0]]
                return predicted_class
                print("Exit from main function")


            url = r"C:\Users\DELL\projects\testing\media\images\{}".format(img_url)
            logging.debug(url)
            logging.debug("Main")
            skin_lesion_type = detect_skin_lesions(url)
            logging.debug("Skin Lesion type:",skin_lesion_type)


        if form.is_valid():
            form.save()
            obj=form.instance
            logging.debug("obj : {}".format(obj))
            return render(request,"index.html",{"obj":obj})  
    else:
        form=ImageForm()    
    img=Image.objects.all()
    logging.debug("File2:{}".format(request.FILES))
    #logging.debug("File img:{}".format(request.FILES['image']))
    logging.debug("img:{}".format(img))
    # url = r"C:\Users\DELL\projects\testing\media\images\{}".format(img_url)
    # logging.debug(url)
    return render(request,"index.html",{"img":img,"form":form})

