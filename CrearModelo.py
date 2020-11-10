#importando el dataset de scikt-Learn y otro paquetes utilizados
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

#Vamos a imprtar nuestro cosas de Keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.models import model_from_json

#Crearemos una semilla para la reproducción
seed = 40
np.random.seed(seed)

#Importamos el dataset de Iris
iris = load_iris()

#Y los vectores de las características y etiquetas de estos:
x = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

#Se crea una salidas rapidas
y = keras.utils.to_categorical(y)

#Se crean variables globales
n_features = len(feature_names)
n_classes = names.shape[0]

#Este arreglo de "Elements" funciona como un indice dentro del dataset de iris
elements_to_display = [20, 80, 120]
for element in elements_to_display:
	print(f"Element {x[element]}th:")
	print(f" - features: {x[element]}")
	print(f" - Target: {y[element]}")
	print(f" - Species: {names[element % 3]}")
	print()

#Secciona el dataset en un conjunto entrenado y probado
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = seed)

def iris_model(input_dim, output_dim, init_nodes = 4, name = 'model'):
	"""Modelo FF-MLP para el problema de clasificacion de Iris"""
	
	#Crear el modelo
	model = Sequential(name = name)
	model.add(Dense(init_nodes, input_dim = input_dim, activation = 'relu'))
	model.add(Dense(2 * init_nodes, activation = 'relu'))
	model.add(Dense(3 * init_nodes, activation = 'relu'))
	model.add(Dense(output_dim, activation = 'softmax'))
	
	#Compilar el modelo
	model.compile(loss = 'categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])
	
	return model

model = iris_model(n_features, n_classes)
model.summary()

#Hperparametros
epochs = 250
batch = 8

#Linea del modelo
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), verbose = True, epochs = epochs, batch_size = batch)

#Evaluacion final del modelo
scores = model.evaluate(x_test, y_test, verbose = False)
#El comentario de abajo realiza una impresion dentro de cada una de las epocas
#print(f'Test accuracy: {scores[1]}')

def plot_loss(history):
	plt.style.use("ggplot")
	plt.figure(figsize = (8, 4))
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title("Perdidas del modelo entrenado")
	plt.xlabel("epoch #")
	plt.ylabel("Loss")
	plt.legend(['Train', 'Test'], loc = 'upper left')
	plt.show()

def plot_accuracy(history):
	plt.style.use("ggplot")
	plt.figure(figsize = (8, 4))
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.xlabel(["Epoch #"])
	plt.ylabel("Accuracy")
	plt.legend(['Train', 'Test'], loc = 'upper left')
	plt.show()

plot_loss(history)
plot_accuracy(history)

#Serializar el modelo a JSON
model_json = model.to_json()
with open("iris_model.json", "w") as json_file:
	json_file.write(model_json)

#Serializar pesos a HDF5 (es necesario h5py):
model.save_weights("iris_model.h5")
print("El modelo se guardo en el disco")

"""
################################################################
#Este arreglo de "Elements" funciona como un indice dentro del dataset de iris
#Este arreglo de "Elements" funciona como un indice dentro del dataset de iris
elements_to_display = [20, 80, 120]
for element in elements_to_display:
    prediction_vector = model.predict(np.array([x[element]]))
    print(f"Element {element}th:")
    print(f"  - Features: {x[element]}")
    print(f"  - Target: {y[element]}")
    print(f"  - Scpecies: {names[np.argmax(y[element])]}")
    print(f"  - Predicted species: {names[np.argmax(prediction_vector)]}")
    print()
"""