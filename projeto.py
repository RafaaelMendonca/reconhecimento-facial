import cv2

carrega_algoritmo = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:
  camera, frame = webcam.read()
  video_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = carrega_algoritmo.detectMultiScale(video_cinza, scaleFactor=1.2, minNeighbors=5)

  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # mostra o vídeo com as faces detectadas
    cv2.imshow('Faces Detectadas', frame)
  # para interromper a execução
  if cv2.waitKey(1) == ord('q'):
    break

# Libera a câmera e fecha janelas
webcam.release()
cv2.destroyAllWindows()
