import joblib
import pandas
import uvicorn,os
import numpy
from sklearn.linear_model import *
from scipy.spatial import distance
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

desktopData = pandas.read_excel('./Dataset/Desktop.xlsx')
laptopData = pandas.read_excel('./Dataset/Laptop.xlsx')
tabletData = pandas.read_excel('./Dataset/Tablet.xlsx')
smartphoneData = pandas.read_excel('./Dataset/Smartphone.xlsx')

tabletModel = joblib.load(open("./Model/Models/MinPrice/Tablet.sav", 'rb'))
featuresScalerTablet = joblib.load(open("./Model/scalers/MinPrice/scalerFeaturesTablet.skl",'rb'))
targetScalerTablet = joblib.load(open("./Model/scalers/MinPrice/scalerTargetTablet.skl",'rb'))

smartphoneModel = joblib.load(open("./Model/Models/MinPrice/Smartphone.sav", 'rb'))
featuresScalerSmartphone = joblib.load(open("./Model/scalers/MinPrice/scalerFeaturesSmartphone.skl",'rb'))
targetScalerSmartphone = joblib.load(open("./Model/scalers/MinPrice/scalerTargetSmartphone.skl",'rb'))

desktopModel = joblib.load(open("./Model/Models/MinPrice/Desktop.sav", 'rb'))
featuresScalerDesktop = joblib.load(open("./Model/scalers/MinPrice/scalerFeaturesDesktop.skl",'rb'))
targetScalerDesktop = joblib.load(open("./Model/scalers/MinPrice/scalerTargetDesktop.skl",'rb'))

laptopModel = joblib.load(open("./Model/Models/MinPrice/Laptop.sav", 'rb'))
featuresScalerLaptop = joblib.load(open("./Model/scalers/MinPrice/scalerFeaturesLaptop.skl",'rb'))
targetScalerLaptop = joblib.load(open("./Model/scalers/MinPrice/scalerTargetLaptop.skl",'rb'))

tabletModelMaxPrice = joblib.load(open("./Model/Models/MaxPrice/Tablet.sav", 'rb'))
featuresScalerTabletMaxPrice = joblib.load(open("./Model/scalers/MaxPrice/scalerFeaturesTablet.skl",'rb'))
targetScalerTabletMaxPrice = joblib.load(open("./Model/scalers/MaxPrice/scalerTargetTablet.skl",'rb'))

smartphoneModelMaxPrice = joblib.load(open("./Model/Models/MaxPrice/Smartphone.sav", 'rb'))
featuresScalerSmartphoneMaxPrice = joblib.load(open("./Model/scalers/MaxPrice/scalerFeaturesSmartphone.skl",'rb'))
targetScalerSmartphoneMaxPrice = joblib.load(open("./Model/scalers/MaxPrice/scalerTargetSmartphone.skl",'rb'))

desktopModelMaxPrice = joblib.load(open("./Model/Models/MaxPrice/Desktop.sav", 'rb'))
featuresScalerDesktopMaxPrice = joblib.load(open("./Model/scalers/MaxPrice/scalerFeaturesDesktop.skl",'rb'))
targetScalerDesktopMaxPrice = joblib.load(open("./Model/scalers/MaxPrice/scalerTargetDesktop.skl",'rb'))

laptopModelMaxPrice = joblib.load(open("./Model/Models/MaxPrice/Laptop.sav", 'rb'))
featuresScalerLaptopMaxPrice = joblib.load(open("./Model/scalers/MaxPrice/scalerFeaturesLaptop.skl",'rb'))
targetScalerLaptopMaxPrice = joblib.load(open("./Model/scalers/MaxPrice/scalerTargetLaptop.skl",'rb'))

desktopFeaturesScaled = featuresScalerDesktopMaxPrice.transform(desktopData[list(desktopData)[0:len(list(desktopData))-1]])
laptopFeaturesScaled = featuresScalerLaptopMaxPrice.transform(laptopData[list(laptopData)[0:len(list(laptopData))-1]])
smartphoneFeaturesScaled = featuresScalerSmartphoneMaxPrice.transform(smartphoneData[list(smartphoneData)[0:len(list(smartphoneData))-1]])
tabletFeaturesScaled = featuresScalerTabletMaxPrice.transform(tabletData[list(tabletData)[0:len(list(tabletData))-1]])

desktopTarget = desktopData[list(desktopData)[len(list(desktopData))-1]]
laptopTarget = laptopData[list(laptopData)[len(list(laptopData))-1]]
smartphoneTarget = smartphoneData[list(smartphoneData)[len(list(smartphoneData))-1]]
tabletTarget = tabletData[list(tabletData)[len(list(tabletData))-1]]

def evaluatePrediction(devicesScaled,featuresScaled, targetNotScaled):
    values = [(devicesScaled[index],targetNotScaled[index]) for index in range(0,len(devicesScaled))]
    distances = [(distance.euclidean(featuresScaled[0],value[0]),value[1]) for value in values]
    distances.sort()
    return min(100,(1-((numpy.std([a[1] for a in distances][1:21]))/(sum([a[1] for a in distances][1:21])/20))/3)*100)


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
    brandsVector, graphicsVector, cpuVector = prepareVectors(brands, graphicsBrands, cpuBrands,brand,graphicsModel,cpuModel)
    return [hddStorage,sddStorage,ram,yearOfLaunch]+brandsVector+graphicsVector+cpuVector

def prepareForModelLaptop(screenSize,storage,ram,sddStorage,resolution,yearOfLaunch,brand,graphicsModel,cpuModel):
    brands = ["Thinkpad","Probook","Elitebook","Latitude","Lifebook","Autre"]
    graphicsBrands = ["Intel","AMD","Nvidia"]
    cpuBrands = ["Xeon","Pentium","Core i3","Core i5","Core i7"]
    brandsVector, graphicsVector, cpuVector = prepareVectors(brands, graphicsBrands, cpuBrands,brand,graphicsModel,cpuModel)
    return [screenSize,storage,ram,sddStorage,resolution,yearOfLaunch]+brandsVector+graphicsVector+cpuVector

def prepareVectors(brands, graphicsBrands, cpuBrands, brand, graphicsModel, cpuModel):
    brandsVector = [name==brand for name in brands]
    brandsVector.append(1-sum(brandsVector))
    graphicsVector = [name==graphicsModel for name in graphicsBrands]
    cpuVector = [name==cpuModel for name in cpuBrands]
    cpuVector.append(1-sum(cpuVector))
    return brandsVector,graphicsVector,cpuVector

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Device(BaseModel):
    name: str
    description: str
    minPrice: float
    maxPrice: float
    
    def toDict(self):
        return{"name":self.name,"description":self.description,"minPrice":self.minPrice,"maxPrice":self.maxPrice}

class DesktopComputer(Device):
    """A desktop computer (for section browse of the web application, TODO)"""
    hddCapacity: int
    sddCapacity: int
    RAM: int
    brand: str
    graphics: str
    cpu: str
    yearBought: int
    
    def toDict(self):
        deviceDict = super().toDict()
        deviceDict["hddCapacity"]=self.hddCapacity
        deviceDict["sddCapacity"]=self.sddCapacity
        deviceDict["RAM"]=self.RAM
        deviceDict["brand"]=self.brand
        deviceDict["graphics"]=self.graphics
        deviceDict["cpu"]=self.cpu
        deviceDict["yearBought"]=self.yearBought
        return deviceDict

desktopComputers=[]

class Response(BaseModel):
    MaxPrice:float = Field(..., example=245)
    MinPrice:float = Field(..., example=205)
    Confidence:float = Field(..., example=91)

@app.post("/devices/desktop/")
async def createDesktop(desktopComputer: DesktopComputer):
    """Creates a new desktop computer (for section browse of the web application, TODO)"""
    desktopComputers.append(desktopComputer)
    return {"id":len(desktopComputers)-1}

@app.get("/devices/desktop/")
def getDesktop(identifier:int):
    """Gets a desktop computer by its id (for section browse of the web application, TODO)"""
    return desktopComputers[identifier].toDict()

@app.get("/devices/desktop/get")
def getDesktop(start:int,howMany:int=20):
    """Gets a list of desktop computers (for section browse of the web application, TODO)"""
    someDesktopComputers={}
    for index in range(start,min(len(desktopComputers),start+min(20,howMany))):
        someDesktopComputers[str(index)] = desktopComputers[index]
    return {"devices":someDesktopComputers}

@app.get("/")
def root():
    return {"message": "Use /docs to see the documentation"}

@app.get("/tablet/", response_model=Response)
def read_root(screenSize,storage,ram,resolution,yearOfLaunch,megapixels,model):
    """Predicts the price of a tablet based on its features"""
    features = [prepareForModelTablet(screenSize,storage,ram,resolution,yearOfLaunch,megapixels,model)]
    minPrice = targetScalerTablet.inverse_transform([tabletModel.predict(featuresScalerTablet.transform(features))])[0][0]
    maxPrice = targetScalerTabletMaxPrice.inverse_transform([tabletModelMaxPrice.predict(featuresScalerTabletMaxPrice.transform(features))])[0][0]
    confidence = evaluatePrediction(tabletFeaturesScaled,featuresScalerTabletMaxPrice.transform(features),tabletTarget)
    if  maxPrice>minPrice:#We want the min price to be the lowest. Because of the way the model is trained, sometimes the max price is lower than the min price
        return Response(MinPrice=minPrice, MaxPrice=maxPrice, Confidence=confidence)
    else:
        return Response(MinPrice=maxPrice, MaxPrice=minPrice, Confidence=confidence)

@app.get("/smartphone/", response_model=Response)
def read_root(screenSize,storage,ram,megapixels,resolution,bandwith,yearOfLaunch,model):
    """Predicts the price of a smartphone based on its features"""
    features = [prepareForModelSmartphone(screenSize,storage,ram,megapixels,resolution,bandwith,yearOfLaunch,model)]
    minPrice = targetScalerSmartphone.inverse_transform([smartphoneModel.predict(featuresScalerSmartphone.transform(features))])[0][0]
    maxPrice = targetScalerSmartphoneMaxPrice.inverse_transform([smartphoneModelMaxPrice.predict(featuresScalerSmartphoneMaxPrice.transform(features))])[0][0]
    confidence = evaluatePrediction(smartphoneFeaturesScaled,featuresScalerSmartphoneMaxPrice.transform(features),smartphoneTarget)
    if  maxPrice>minPrice:#We want the min price to be the lowest. Because of the way the model is trained, sometimes the max price is lower than the min price
        return Response(MinPrice=minPrice, MaxPrice=maxPrice, Confidence=confidence)
    else:
        return Response(MinPrice=maxPrice, MaxPrice=minPrice, Confidence=confidence)

@app.get("/desktop/", response_model=Response)
def read_root(hddStorage,sddStorage,ram,yearOfLaunch,brand,graphicsModel,cpuModel):
    """Predicts the price of a desktop computer based on its features"""
    features = [prepareForModelDesktop(hddStorage,sddStorage,ram,yearOfLaunch,brand,graphicsModel,cpuModel)]
    minPrice = targetScalerDesktop.inverse_transform([desktopModel.predict(featuresScalerDesktop.transform(features))])[0][0]
    maxPrice = targetScalerDesktopMaxPrice.inverse_transform([desktopModelMaxPrice.predict(featuresScalerDesktopMaxPrice.transform(features))])[0][0]
    confidence = evaluatePrediction(desktopFeaturesScaled,featuresScalerDesktopMaxPrice.transform(features),desktopTarget)
    if  maxPrice>minPrice:#We want the min price to be the lowest. Because of the way the model is trained, sometimes the max price is lower than the min price
        return Response(MinPrice=minPrice, MaxPrice=maxPrice, Confidence=confidence)
    else:
        return Response(MinPrice=maxPrice, MaxPrice=minPrice, Confidence=confidence)

@app.get("/laptop/", response_model=Response)
def read_root(screenSize,hddStorage,ram,sddStorage,resolution,yearOfLaunch,brand,graphicsModel,cpuModel):
    """Predicts the price of a laptop based on its features"""
    features = [prepareForModelLaptop(screenSize,hddStorage,ram,sddStorage,resolution,yearOfLaunch,brand,graphicsModel,cpuModel)]
    minPrice = targetScalerLaptop.inverse_transform([laptopModel.predict(featuresScalerLaptop.transform(features))])[0][0]
    maxPrice = targetScalerLaptopMaxPrice.inverse_transform([laptopModelMaxPrice.predict(featuresScalerLaptopMaxPrice.transform(features))])[0][0]
    confidence = evaluatePrediction(laptopFeaturesScaled,featuresScalerLaptopMaxPrice.transform(features),laptopTarget)
    if  maxPrice>minPrice:#We want the min price to be the lowest. Because of the way the model is trained, sometimes the max price is lower than the min price
        return Response(MinPrice=minPrice, MaxPrice=maxPrice, Confidence=confidence)
    else:
        return Response(MinPrice=maxPrice, MaxPrice=minPrice, Confidence=confidence)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
