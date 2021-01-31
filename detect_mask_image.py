# USAGE
# python detect_mask_image.py --image examples/example_01.png

# importation des bibliothèques
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# construire l'analyseur d'arguments et analyser les arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
print(args)

# charger notre modèle de détecteur de visage sérialisé
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# charger le modèle de détecteur de masque facial
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

# ON charge l'image d'entrée à partir du disque, clonez-la et récupérez l'image spatiale
# dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# construction d'un blob à partir de l'image.
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# faire passer le blob à travers le réseau et obtenir les détections de visage
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

# boucle sur les détections
for i in range(0, detections.shape[2]):
	# extraire la confiance (c.-à-d. la probabilité) associée à
	# la détection

	confidence = detections[0, 0, i, 2]

	# filtrer les détections faibles en s'assurant que la confiance
	# supérieur à la confiance minimale


	if confidence > args["confidence"]:
		# calculer les coordonnées (x, y) de la boîte englobante pour
		# L'object


		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# assurez-vous que les cadres de délimitation correspondent aux dimensions du
		# cadre


		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# extraire le ROI du visage, le convertir du canal BGR au canal RVB
		# ordering, redimensionnez-le à 224x224 et prétraitez-le


		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# passez le visage à travers le modèle pour déterminer si le visage
		# a un masque ou pas


		(mask, withoutMask) = model.predict(face)[0]

		# déterminer le libellé de la classe et la couleur que nous utiliserons pour dessiner
		# le cadre de sélection et le texte


		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# inclure la probabilité dans le modele
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# affiche le libellé et le rectangle de la boîte englobante sur la sortie
		# Cadre


		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# afficher l'image de sortie
cv2.imshow("Output", image)
cv2.waitKey(0)