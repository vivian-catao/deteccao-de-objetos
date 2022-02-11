import cv2

obj_img = cv2.imread("test_10.png") #, cv2.IMREAD_COLOR 
obj_img_cinza = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)


classificador = cv2.CascadeClassifier('first_cascade.xml') 
deteccoes = classificador.detectMultiScale(obj_img_cinza, scaleFactor=1.1, minNeighbors = 5 )
print(deteccoes)
print (len(deteccoes))

for x, y,l,h in deteccoes:
    #print(x,y,l,h) #imprimir coordenadas
    cv2.rectangle(obj_img,(x,y),(x+l,y+h),(0,255,0),2)



cv2.namedWindow('Imagem')
cv2.imshow('Imagem', obj_img)
cv2.waitKey()
