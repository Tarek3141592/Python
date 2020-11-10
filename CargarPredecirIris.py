from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np

from tensorflow.keras.models import model_from_json

#Cargar json y crear el modelo
json_file = open('iris_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#Cargar los pesos dentro del modelo cargado
loaded_model.load_weights("iris_model.h5")
print ("El modelo se cargo desde el disco")

#Evaluar el modelo cargado en un test de datos:
loaded_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metricts = ['accuracy'])
 
""" al parecer no tiene la suficiente capacidad para hacer multrprocesos """
#No se ha podido cargar el score
#score = loaded_model.evaluate(4.6, 3.1, verbose = 1)

#print (f'Test accuracy: {score[1]}')


r = loaded_model.predict(np.array([[5.1, 3.5, 1.4, 0.2]]))
print(r)
print(np.argmax(r))