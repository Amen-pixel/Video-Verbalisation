# =====================================================================
# PROTOTYPE DE DÉMONSTRATION – Vidéo-verbalisation (YOLO26 + ByteTrack)
# =====================================================================
import os
import cv2
import requests
import pytesseract
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from statistics import median, mean
from ultralytics import YOLO

# Patch NumPy pour les anciennes librairies
for attr, cible in [("long", np.int64), ("ulong", np.uint64), ("int_", np.int64), ("float_", np.float64)]:
    if not hasattr(np, attr):
        setattr(np, attr, cible)

# =====================================================================
# 1. RÉGLAGES
# =====================================================================
VIDEO = "trafic_bénin.mp4"
URL_API = "http://127.0.0.1:8008/api/passages"
DEBUT_VIDEO = datetime.now()      # ou l'heure réelle du tournage : datetime(2026, 11, 23, 8, 0, 0)

# Limites de vitesse par type de route (modifie-les selon la route filmée)
LIMITES = {"agglomeration": 50, "route_secondaire": 80, "route_nationale": 100, "voie_rapide": 120}
TYPE_ROUTE = "agglomeration"
SEUIL_KMH = float(LIMITES[TYPE_ROUTE])

VITESSE_MAX_PLAUSIBLE = 150.0     # au-dessus = erreur de suivi, ignoré
PIXELS_PAR_METRE = 25.0           # À RÉGLER (remets ta valeur validée)
LIGNE_Y_RATIO = 0.6               # ligne de contrôle (0 = haut de l'image, 1 = bas)

# --- Précision maximale (lent) ---
MODELE = "yolo26x.pt"             # le plus précis ; si erreur : "yolo11x.pt"
IMGSZ = 960                       # grande résolution = petits véhicules et véhicules lointains détectés
CONF = 0.15                       # seuil bas = on rate moins de véhicules
SAUT = 1                          # toutes les images analysées

AFFICHER_FENETRE = False          # False = plus rapide, tu regardes demo_annotee.mp4 à la fin
ENVOYER_CONFORMES = True          # les véhicules en règle comptent dans les stats (sans photo)
MARGE = 20
DOSSIER = "captures"
CLASSES = {2: "Voiture", 3: "Moto", 5: "Bus", 7: "Camion"}
ROUGE, VERT, BLEU, JAUNE = (0, 0, 255), (0, 200, 0), (255, 170, 0), (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

os.makedirs(DOSSIER, exist_ok=True)
with open("custom_bytetrack.yaml", "w") as f:
    f.write("tracker_type: bytetrack\ntrack_high_thresh: 0.25\ntrack_low_thresh: 0.05\n"
            "new_track_thresh: 0.25\ntrack_buffer: 90\nmatch_thresh: 0.8\nfuse_score: True\n")

model = YOLO(MODELE)


# =====================================================================
# 2. OUTILS : OCR PLAQUE + DESSIN
# =====================================================================
def lire_plaque(img):
    try:
        gris = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 11, 17, 17)
        bw = cv2.threshold(gris, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cfg = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        plaque = "".join(c for c in pytesseract.image_to_string(bw, config=cfg) if c.isalnum()).upper()
        return plaque if len(plaque) >= 3 else "INCONNUE"
    except Exception:
        return "INCONNUE"


def cadre(img, box, texte, couleur, epaisseur):
    """Cadre + étiquette sur fond plein, lisible même en vidéo."""
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), couleur, epaisseur, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(texte, FONT, 0.7, 2)
    ty = max(y1, th + 12)
    cv2.rectangle(img, (x1, ty - th - 12), (x1 + tw + 10, ty), couleur, -1)
    cv2.putText(img, texte, (x1 + 5, ty - 6), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def entete(img, texte):
    """Bandeau d'infos + légende des couleurs."""
    zone = img[0:150, 0:min(img.shape[1], 900)]
    zone[:] = cv2.addWeighted(zone, 0.35, np.zeros_like(zone), 0.65, 0)
    cv2.putText(img, texte, (15, 32), FONT, 0.75, JAUNE, 2, cv2.LINE_AA)
    for i, (c, t) in enumerate([(VERT, "CONFORME (en regle apres la ligne)"),
                                (ROUGE, "INFRACTION"), (BLEU, "EN COURS (pas encore controle)")]):
        cv2.rectangle(img, (15, 52 + i * 30), (35, 72 + i * 30), c, -1)
        cv2.putText(img, t, (45, 70 + i * 30), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


# =====================================================================
# 3. BOUCLE PRINCIPALE
# =====================================================================
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Impossible d'ouvrir la vidéo : {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
fenetre = "PROTOTYPE - Video-verbalisation"
if AFFICHER_FENETRE:
    cv2.namedWindow(fenetre, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(fenetre, 1280, 720)

historique, recents, dernier_y = {}, {}, {}
vus, controles = set(), set()
verbalises, conformes = {}, {}             # id -> vitesse retenue
sortie, n = None, 0
print(f"Analyse de {VIDEO} ({total} images, {fps:.0f} FPS) avec {MODELE} – Ctrl+C pour arrêter proprement")

try:
    while cap.grab():
        n += 1
        if n % SAUT:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        h, w = frame.shape[:2]
        ligne_y = int(h * LIGNE_Y_RATIO)
        horodatage = (DEBUT_VIDEO + timedelta(seconds=n / fps)).strftime("%Y-%m-%d %H:%M:%S")
        if sortie is None:
            sortie = cv2.VideoWriter("demo_annotee.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps / SAUT, (w, h))

        res = model.track(frame, persist=True, tracker="custom_bytetrack.yaml",
                          classes=list(CLASSES), imgsz=IMGSZ, conf=CONF, verbose=False)[0]

        vehicules, decisions = [], []
        if res.boxes is not None and res.boxes.id is not None:
            boites = res.boxes.xyxy.cpu().numpy().astype(int)
            ids = res.boxes.id.cpu().numpy().astype(int)
            classes = res.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), tid, cls in zip(boites, ids, classes):
                nom, cx = CLASSES[cls], (x1 + x2) // 2
                vus.add(tid)

                # --- Vitesse lissée, mesurée sur le point bas du véhicule ---
                hist = historique.setdefault(tid, deque(maxlen=20))
                hist.append((cx, y2, n))
                vitesse = 0.0
                if len(hist) >= 4:
                    xa, ya, na = hist[0]
                    dist_m = (((cx - xa) ** 2 + (y2 - ya) ** 2) ** 0.5) / PIXELS_PAR_METRE
                    vitesse = dist_m / ((n - na) / fps) * 3.6
                if 0 < vitesse <= VITESSE_MAX_PLAUSIBLE:
                    recents.setdefault(tid, deque(maxlen=8)).append(vitesse)
                vehicules.append((tid, nom, vitesse, (x1, y1, x2, y2)))

                # --- CONTRÔLE : une seule décision, au franchissement de la ligne ---
                py, dernier_y[tid] = dernier_y.get(tid), y2
                if (py is not None and tid not in controles and py != y2
                        and (py - ligne_y) * (y2 - ligne_y) <= 0):
                    v = recents.get(tid)
                    if v and len(v) >= 2:
                        controles.add(tid)
                        vit = float(median(v))
                        infraction = bool(vit > SEUIL_KMH)
                        plaque = ""
                        if infraction:
                            verbalises[tid] = vit
                            plaque = lire_plaque(frame[max(0, y1 - MARGE):min(h, y2 + MARGE),
                                                       max(0, x1 - MARGE):min(w, x2 + MARGE)])
                            print(f"INFRACTION #{tid} : {nom} à {vit:.0f} km/h | Plaque : {plaque}")
                        else:
                            conformes[tid] = vit
                        decisions.append((tid, nom, vit, float(mean(v)), infraction, plaque))

        # --- DESSIN : tous les véhicules, avec leur statut ---
        for tid, nom, vitesse, box in vehicules:
            if tid in verbalises:
                cadre(frame, box, f"INFRACTION #{tid} | {verbalises[tid]:.0f} km/h", ROUGE, 5)
            elif tid in conformes:
                cadre(frame, box, f"OK #{tid} | {conformes[tid]:.0f} km/h", VERT, 4)
            else:
                cadre(frame, box, f"{nom} #{tid}", BLEU, 2)
        cv2.line(frame, (0, ligne_y), (w, ligne_y), JAUNE, 2, cv2.LINE_AA)
        entete(frame, f"PROTOTYPE | {horodatage} | {TYPE_ROUTE} {SEUIL_KMH:.0f} km/h | "
                      f"Controles: {len(controles)} | Infractions: {len(verbalises)}")

        # --- CAPTURE : image complète au moment du franchissement (verts + rouges visibles) ---
        for tid, nom, vit, moy, infraction, plaque in decisions:
            nom_image = ""
            if infraction:
                nom_image = f"infraction_id_{tid}.jpg"
                cv2.imwrite(os.path.join(DOSSIER, nom_image), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if infraction or ENVOYER_CONFORMES:
                try:
                    requests.post(URL_API, timeout=1, json={
                        "date_heure": horodatage, "track_id": int(tid), "type_vehicule": nom,
                        "vitesse_kmh": round(vit, 1), "vitesse_moy": round(moy, 1),
                        "infraction": infraction, "limite_kmh": SEUIL_KMH,
                        "plaque": plaque, "image_name": nom_image})
                except Exception as e:
                    print(f"API injoignable : {e}")

        sortie.write(frame)
        if n % 50 == 0:
            print(f"  {n}/{total} images ({100 * n / total:.0f} %) | contrôlés : {len(controles)} | infractions : {len(verbalises)}")
        if AFFICHER_FENETRE:
            cv2.imshow(fenetre, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
except KeyboardInterrupt:
    print("Arrêt demandé – finalisation de la vidéo...")

cap.release()
if sortie is not None:
    sortie.release()
cv2.destroyAllWindows()

print(f"\nTerminé : {len(vus)} véhicules suivis | {len(controles)} contrôlés à la ligne "
      f"({len(conformes)} conformes, {len(verbalises)} infractions).")
if len(vus) > len(controles):
    print(f"{len(vus) - len(controles)} pistes n'ont pas franchi la ligne (hors trajectoire, fausses détections "
          f"courtes ou trop peu de mesures) : ajuste LIGNE_Y_RATIO si ce sont de vrais véhicules.")
if len(controles) >= 10 and len(verbalises) / len(controles) > 0.6:
    print("ATTENTION : plus de 60 % d'infractions -> PIXELS_PAR_METRE est sûrement mal calibré.")
print("Vidéo finale : demo_annotee.mp4")
