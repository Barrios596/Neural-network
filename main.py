import cv2
import numpy as np
import image_loader as il
import red_neuronal as rn
from tkinter import messagebox
from tkinter import *

#outputs para mostrar el resultado
outputs = [
    'Signo de interrogación',
    'Sad Face',
    'Huevo',
    'Circulo',
    'Smiley Face',
    'Cuadrado',
    'Arbol',
    'Triangulo',
    'Casa',
    'Mickey Mouse'
]

drawing=False
mode=True
imagen=[]


def show_message(title, message):
    window = Tk()
    window.eval('tk::PlaceWindow %s center' % window.winfo_toplevel())
    window.withdraw()
    
    messagebox.showinfo(title, message)

    window.deiconify()
    window.destroy()
    window.quit()

def interactive_drawing(event,x,y,flags,param):
    global ix,iy,drawing, mode

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(img,(ix,iy),(x,y),(255,255,255),10)
                ix=x
                iy=y
                imagen.append((int(ix*28/1000),int(iy*28/1000)))
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.circle(img,(x,y),1,(0,0,255),-1)
    return x,y


print('\nPor favor espere.\nCargando las imágenes de entrenamiento...\n\n')
data = il.loadImages()

print('Imágenes cargadas.\nEntrenando a la red neural...\n\n')
red = rn.Red(np.array([28*28,64,10]))
red.descenso_gradiente(data, 8, 10)

print('Red entrenada.\nA continuación dibuje la imagen y presione la tecla enter cuando termine.\n')
img = np.zeros((1000,1000,3), np.uint8)
cv2.namedWindow('Canvas')
cv2.setMouseCallback('Canvas',interactive_drawing)

while (1):

    while(1):
        cv2.imshow('Canvas',img)

        k = cv2.waitKey(1) & 0xFF
        if k==13:
            break
        
        if k == 27:
            cv2.destroyAllWindows()
            exit()

    nueva = list(set(imagen))
    matrizImagen = np.zeros((28,28))

    for x,y in nueva:
        matrizImagen[y][x] = 1
    
    result = red.feedforward(matrizImagen.flatten().reshape(784,1))
    title = "Resultado"
    message = "La imagen es: " + str(outputs[np.argmax(result)]) + " con un " + str(np.amax(result) * 100) + " % de seguridad" 
    show_message(title, message)

    imagen = []
    matrizImagen = np.zeros((28,28))
    img = np.zeros((1000,1000,3), np.uint8)