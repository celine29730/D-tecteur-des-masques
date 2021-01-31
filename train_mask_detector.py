# USAGE
# python train_mask_detector.py --dataset dataset

# importation des bibliothèques necessaires
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construire l'analyseur d'arguments et analyser les arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# initialiser le taux d'apprentissage initial, le nombre d'époques à former,
# et taille du lot

INIT_LR = 1e-4
EPOCHS = 20
#une epoque contient plusieurs iterations (iteration = nb d'images / Batch)
#le nombre d'térationcoreespond à une époque.
BS = 32
#BS est choisi de façon aléatoire par l'utilisateur (4, 8, 16, 32 ). Plus BS est grand, plus le modèle sera précis.

# récupérez la liste des images dans notre répertoire de jeux de données, puis initialisez
# la liste des données (c'est-à-dire des images) et des images de classe

print("[INFO] loading images...")

imagePaths = list(paths.list_images(args["dataset"]))
print(len(imagePaths))
data = []
labels = []

# boucle sur les chemins de l'image

for imagePath in imagePaths:
	# extraire le label de classe du nom de fichier
	label = imagePath.split(os.path.sep)[-2]

	# charger l'image d'entrée (224x224) et la prétraiter

	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

# mettre à jour respectivement les listes de données et d'étiquettes
	data.append(image)
	labels.append(label)

# convertir les données et les étiquettes en tableaux NumPy
data = np.array(data, dtype="float32")
labels = np.array(labels)

# effectuer un encodage sur les étiquettes

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partitionner les données en fractionnements d'entraînement et de test en utilisant 80% de
# les données pour la formation et les 20% restants pour les tests

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

print("============================================================================================================================\n\n")
print(len(trainX))
# construire le générateur d'images d'entraînement pour l'augmentation des données

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# chargez le réseau MobileNetV2, en vous assurant que les ensembles de couches FC de
# laisser derrière soi

baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construire la tête du modèle qui sera placée au-dessus du
# le modèle de base

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# placez le modèle FC de la tête sur le modèle de base (cela deviendra
# le modèle réel que nous allons former)

model = Model(inputs=baseModel.input, outputs=headModel)

# boucle sur toutes les couches du modèle de base et les fige pour qu'elles
# * not * être mis à jour lors du premier processus de formation

for layer in baseModel.layers:
	layer.trainable = False

# compilation du modele
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# former le chef de réseau
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# faire des prédictions sur l'ensemble de test

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# pour chaque image de l'ensemble de test, nous devons trouver l'index du
# étiquette avec la plus grande probabilité prédite correspondante


predIdxs = np.argmax(predIdxs, axis=1)

# afficher un rapport de classification formaté

print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# sérialiser le modèle sur le disque
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# tracer la perte d'entraînement et la précision
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])