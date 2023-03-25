import pandas,time


df = pandas.read_excel("CleanData/Laptop.ods", engine="odf")


brands = {}
identifier = 0
for brand in df['Serie'].values:
    x = (''.join(str(brand).split(' ')[0])).lower()
    if x not in brands:
        brands[x] = identifier
        identifier+=1

for key,value in brands.items():
    modified = [(''.join(str(brand).split(' ')[0]).lower()) for brand in df['Serie'].values]
    df[key] = [int(x==key) for x in modified]
    

brands = {}
identifier = 0
for brand in df['Tarjeta gráfica'].values:
    x = (''.join(str(brand).split(' ')[0])).lower()
    if x not in brands:
        brands[x] = identifier
        identifier+=1


for key,value in brands.items():
    modified = [(''.join(str(brand).split(' ')[0]).lower()) for brand in df['Tarjeta gráfica'].values]
    df[key] = [int(x==key) for x in modified]


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
processors = ['Xeon','Pentium','Core i3','Core i5','Core i7','']
columnsToAppend = [[],[],[],[],[],[],[]]

for index, row in df.iterrows():
    firstToMatch = 0
    for p in processors:
        if p in row['Procesador']:
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
        df['Other Processor'] = columnsToAppend[p]
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
velocidad = []
so=[]
resolucion = []
for index, row in df.iterrows():
    almacenamiento.append(int(row['Almacenamiento'].split(' ')[0]))
    ram.append(int(row['Memoria RAM'].split(' ')[0]))
    velocidad.append(float(row['Velocidad del procesador'].split(' ')[0].replace(',','.')))
    so.append(int("Windows" in row['Sistema operativo']))
    resolucion.append(max(int(row['Resolución'].split(' x ')[0]),int(row['Resolución'].split(' x ')[1])))

df['Almacenamiento'] = almacenamiento
df['Memoria RAM'] = ram
df['Velocidad del procesador'] = velocidad
df['Sistema operativo'] =so
df['Resolución']=resolucion


df = df.drop(columns=['Correcto','Muy bueno','Excelente','Tarjeta gráfica','Modelo','Procesador','Serie','Marca'])

df.to_excel("CleanData/LaptopTest.ods", index=False)
