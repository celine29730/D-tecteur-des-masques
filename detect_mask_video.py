# USAGE
# python detect_mask_video.py

# importation des bibliothèques
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# saisir les dimensions du cadre, puis construire un blob à partir de cela


	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# faire passer le blob à travers le réseau et obtenir les détections de visage
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialiser notre liste de visages, leurs emplacements correspondants,
	# et la liste des prédictions de notre réseau de masques faciaux
	faces = []
	locs = []
	preds = []

	# boucle sur les détections
	for i in range(0, detections.shape[2]):
		# extraire la confiance (c'est-à-dire la probabilité) associée à la détection
		confidence = detections[0, 0, i, 2]

# filtrer les détections faibles en s'assurant que la confiance
# supérieur à la confiance minimale

		if confidence > args["confidence"]:
			# calculer les coordonnées (x, y) de la boîte englobante pour l'objet

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# assurez-vous que les cadres de délimitation correspondent aux dimensions du cadre
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extraire le ROI du visage, le convertir du canal BGR au canal RVB
			# redimensionnez-le à 224x224 et prétraitez-le


			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# ajouter le visage et les cadres de délimitation à leurs listes

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# faire des prédictions uniquement si au moins un visage a été détecté
	if len(faces) > 0:
		# pour une inférence plus rapide, nous ferons des prédictions par lots sur * tous *
		# visages en même temps plutôt que des prédictions un par un
		# dans la boucle `for` ci-dessus
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# renvoie un 2-tuple des emplacements de visage et de leur emplacements correspondants
	return (locs, preds)

# construire l'analyseur d'arguments et analyser les arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# charger notre modèle de détecteur de visage sérialisé à partir du disque
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# charger le modèle de détecteur de masque facial à partir du disque
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialisez le flux vidéo et laissez le capteur de la caméra se réchauffer
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# boucle sur les images du flux vidéo
while True:
	# # saisissez l'image du flux vidéo fileté et redimensionnez-la
	# pour avoir une largeur maximale de 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# détecter les visages dans le cadre et déterminer s'ils portent un
	# masque facial ou non
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# boucle sur les emplacements des visages détectés et leurs
	# Emplacements
	for (box, pred) in zip(locs, preds):
		# déballer le cadre de sélection et les prédictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# déterminer le libellé de la classe et la couleur que nous utiliserons pour dessiner
		# le cadre de sélection et le texte
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		#insertion smiley
		smiley = "\smiley_content.jpg" if mask > withoutMask else "\smiley_mecontent.png"
		path = 'image' + smiley
		# Lire une image en mode par défaut 
		image = cv2.imread(path) 
		# Nom de la fenêtre dans laquelle l'image est affichée 
		window_name = 'Image'
		# font 
		font = cv2.FONT_HERSHEY_SIMPLEX 
		# org 
		org = (50, 50) 
		# fontScale 
		fontScale = 1
		# Blue color in BGR 
		color_smiley = (255, 0, 0) 
		# Épaisseur de ligne de 2 px 
		thickness = 2
		# Utilisation de la méthode cv2.putText () 
		image = cv2.putText(image, '', org, font,  
						fontScale, color_smiley, thickness, cv2.LINE_AA) 

		# inclure la probabilité dans le label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# afficher le label et le rectangle de la boîte englobante sur la sortie
		# Cadre
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


	# afficher la trame de sortie
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# Affichage de l'image
Icône de validation par la communauté 
	cv2.imshow(window_name, image) 
	# si la touche `q` a été enfoncée, sortir de la boucle
	if key == ord("q"):
		break

# faire un peu de nettoyage
cv2.destroyAllWindows()
vs.stop()