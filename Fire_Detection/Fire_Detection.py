import cv2
from ultralytics import YOLO
import pygame
import time

# Modeli yükle (eğittiğin modeli path olarak yaz)
model = YOLO("runs/detect/fire-detection-v95/weights/best.pt")  # <- kendi model yolunu yaz

# Alarm dosyasını tanımla
ALARM_SOUND = "public-domain-beep-sound-100267.mp3"  # aynı klasörde olmalı veya tam yol vermelisin

# Ses sistemi başlat
pygame.mixer.init()

def play_alert_sound():
    pygame.mixer.music.load(ALARM_SOUND)
    pygame.mixer.music.play()

# Kamera başlat
cap = cv2.VideoCapture(0)  # 0 = varsayılan kamera

fire_detected = False
last_alarm_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Model ile tahmin yap
    results = model.predict(frame, imgsz=640, conf=0.5)

    # Tespitleri çiz
    annotated_frame = results[0].plot()

    # Ateş varsa alarm çal
    if results[0].boxes:
        if not fire_detected or time.time() - last_alarm_time > 10:
            print("🔥 Yangın tespit edildi!")
            play_alert_sound()
            fire_detected = True
            last_alarm_time = time.time()
    else:
        fire_detected = False

    # Görüntüyü göster
    cv2.imshow("Fire Detection", annotated_frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
