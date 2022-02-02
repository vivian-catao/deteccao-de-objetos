import cv2

obj_img = cv2.imread("maispessoas.jpg") #, cv2.IMREAD_COLOR 
obj_img_cinza = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)

detector_facial = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
deteccoes = detector_facial.detectMultiScale(obj_img_cinza, scaleFactor=1.2, minNeighbors = 5 )
print(deteccoes)
print (len(deteccoes))

for x, y,l,h in deteccoes:
    #print(x,y,l,h)
    cv2.rectangle(obj_img,(x,y),(x+l,y+h),(0,255,0),5)



cv2.namedWindow('Imagem')
cv2.imshow('Imagem', obj_img)
cv2.waitKey()
