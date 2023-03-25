import pandas,time


df = pandas.read_excel("CleanData/Tablet.ods", engine="odf")

def tratarNanMax(valor):
    try:
        return int(valor.split(',')[0])
    except:
        return 100000
    
def tratarNanMin(valor):
    try:
        return int(valor.split(',')[0])
    except:
        return 0

processorIndexs = []
processors = ['iPad','iPad Pro','iPad Air','iPad mini','Galaxy Tab','']
columnsToAppend = [[],[],[],[],[],[]]

for index, row in df.iterrows():
    firstToMatch = 0
    for p in processors:
        if p in row['Serie']:
            break
        firstToMatch+=1
    count = 0
    for _ in processors:
        if count == firstToMatch:
            columnsToAppend[count].append(1)
        else:
            columnsToAppend[count].append(0)
        count+=1

for p in range(0,len(processors)):
    if p == len(processors)-1:
        df['Other'] = columnsToAppend[p]
    else:
        df[processors[p]] = columnsToAppend[p]

minCol=[]
maxCol=[]
for index, row in df.iterrows():
    minCol.append(min(tratarNanMax(row['Correcto']),(min(tratarNanMax(row['Muy bueno']),tratarNanMax(row['Excelente'])))))
    maxCol.append(max(tratarNanMin(row['Correcto']),(max(tratarNanMin(row['Muy bueno']),tratarNanMin(row['Excelente'])))))

df['MinPrice']=minCol
df['MaxPrice']=maxCol

almacenamiento = []
ram = []
resolucion = []
megapix = []
for index, row in df.iterrows():
    almacenamiento.append(int(row['Almacenamiento'].split(' ')[0]))
    ram.append(float(row['Memoria RAM'].split(' ')[0]))
    try:
        resolucion.append(max(int(row['Resolución'].split('x')[0]),int(row['Resolución'].split('x')[1])))
    except:
        resolucion.append(max(int(row['Resolución'].split(' x ')[0]),int(row['Resolución'].split(' x ')[1])))
        
    try:
        megapix.append(int(row['Definición de la cámara trasera'].split(' ')[0]))
    except:
        megapix.append(int(row['Megapíxeles']))

df['Almacenamiento'] = almacenamiento
df['Memoria RAM'] = ram
df['Resolución']=resolucion
df['Definición de la cámara trasera']=megapix


df = df.drop(columns=['Correcto','Muy bueno','Excelente','Color','Operador','Modelo','Definición de la cámara frontal','Marca','Serie'])

df.to_excel("CleanData/TabletTest.ods", index=False)
