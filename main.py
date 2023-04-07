from typing import Union

from fastapi import FastAPI
import joblib
import pandas

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import *
from sklearn import tree
from sklearn import ensemble
from sklearn.svm import SVC
from sklearn import metrics

tabletModel = joblib.load(open("Tablet.sav", 'rb'))
featuresScalerTablet = joblib.load(open("scalerFeaturesTablet.skl",'rb'))
targetScalerTablet = joblib.load(open("scalerTargetTablet.skl",'rb'))

smartphoneModel = joblib.load(open("Smartphone.sav", 'rb'))
featuresScalerSmartphone = joblib.load(open("scalerFeaturesSmartphone.skl",'rb'))
targetScalerSmartphone = joblib.load(open("scalerTargetSmartphone.skl",'rb'))

desktopModel = joblib.load(open("Desktop.sav", 'rb'))
featuresScalerDesktop = joblib.load(open("scalerFeaturesDesktop.skl",'rb'))
targetScalerDesktop = joblib.load(open("scalerTargetDesktop.skl",'rb'))

laptopModel = joblib.load(open("Laptop.sav", 'rb'))
featuresScalerLaptop = joblib.load(open("scalerFeaturesLaptop.skl",'rb'))
targetScalerLaptop = joblib.load(open("scalerTargetLaptop.skl",'rb'))

def prepareForModelTablet(screenSize,storage,ram,resolution,yearOfLaunch,megapixels,model):
    models = ["iPad","iPad Pro","iPad Air","iPad mini","Galaxy Tab"]
    modelsVector = [name==model for name in models]
    modelsVector.append(1-sum(modelsVector))
    return [screenSize,storage,ram,resolution,yearOfLaunch,megapixels]+modelsVector


def prepareForModelSmartphone(screenSize,storage,ram,megapixels,resolution,bandwith,yearOfLaunch,model):
    models = ["Xiaomi","Sony","Redmi","iPhone","Huawei","Google","Galaxy"]
    modelsVector = [name==model for name in models]
    modelsVector.append(1-sum(modelsVector))
    return [screenSize,storage,ram,megapixels,resolution,bandwith,yearOfLaunch]+modelsVector

def prepareForModelDesktop(hddStorage,sddStorage,ram,yearOfLaunch,brand,graphicsModel,cpuModel):
    brands = ["Dell","Lenovo","HP","Fujitsu","Mars","Acer"]
    graphicsBrands = ["Intel","AMD","Nvidia"]
    cpuBrands = ["Xeon","Pentium","Core i3","Core i5","Core i7"]
    
    brandsVector = [name==brand for name in brands]
    brandsVector.append(1-sum(brandsVector))
    
    graphicsVector = [name==graphicsModel for name in graphicsBrands]
    
    cpuVector = [name==cpuModel for name in cpuBrands]
    cpuVector.append(1-sum(cpuVector))
    return [hddStorage,sddStorage,ram,yearOfLaunch]+brandsVector+graphicsVector+cpuVector

def prepareForModelLaptop(screenSize,storage,ram,sddStorage,resolution,yearOfLaunch,brand,graphicsModel,cpuModel):
    brands = ["Thinkpad","Probook","Elitebook","Latitude","Lifebook","Autre"]
    graphicsBrands = ["Intel","AMD","Nvidia"]
    cpuBrands = ["Xeon","Pentium","Core i3","Core i5","Core i7"]
    
    brandsVector = [name==brand for name in brands]
    brandsVector.append(1-sum(brandsVector))
    
    graphicsVector = [name==graphicsModel for name in graphicsBrands]
    
    cpuVector = [name==cpuModel for name in cpuBrands]
    cpuVector.append(1-sum(cpuVector))
    print([screenSize,storage,ram,sddStorage,resolution,yearOfLaunch]+brandsVector+graphicsVector+cpuVector)
    return [screenSize,storage,ram,sddStorage,resolution,yearOfLaunch]+brandsVector+graphicsVector+cpuVector

app = FastAPI()

@app.get("/tablet/")
def read_root(screenSize,storage,ram,resolution,yearOfLaunch,megapixels,model):
    return {"price":targetScalerTablet.inverse_transform([tabletModel.predict(featuresScalerTablet.transform([prepareForModelTablet(screenSize,storage,ram,resolution,yearOfLaunch,megapixels,model)]))])[0][0]}

@app.get("/smartphone/")
def read_root(screenSize,storage,ram,megapixels,resolution,bandwith,yearOfLaunch,model):
    return {"price":targetScalerSmartphone.inverse_transform([smartphoneModel.predict(featuresScalerSmartphone.transform([prepareForModelSmartphone(screenSize,storage,ram,megapixels,resolution,bandwith,yearOfLaunch,model)]))])[0][0]}

@app.get("/desktop/")
def read_root(hddStorage,sddStorage,ram,yearOfLaunch,brand,graphicsModel,cpuModel):
    return {"price":targetScalerDesktop.inverse_transform([desktopModel.predict(featuresScalerDesktop.transform([prepareForModelDesktop(hddStorage,sddStorage,ram,yearOfLaunch,brand,graphicsModel,cpuModel)]))])[0][0]}

@app.get("/laptop/")
def read_root(screenSize,hddStorage,ram,sddStorage,resolution,yearOfLaunch,brand,graphicsModel,cpuModel):
    return {"price":targetScalerLaptop.inverse_transform([laptopModel.predict(featuresScalerLaptop.transform([prepareForModelLaptop(screenSize,hddStorage,ram,sddStorage,resolution,yearOfLaunch,brand,graphicsModel,cpuModel)]))])[0][0]}

