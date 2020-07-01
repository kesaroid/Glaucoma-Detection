import os
from tkinter import *
from tkinter import messagebox

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas
from keras.models import load_model

model = load_model('f1.h5')
test_path = r'test'
spath = ''  # TODO: replace me with a reasonable default


def get_filenames():
    global test_path
    return os.listdir(test_path)


def curselect(_event):
    global spath
    index = t1.curselection()[0]
    spath = t1.get(index)
    return spath


def autoroi(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=5)

    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = img[y:y + h, x:x + w]

    return roi


def prediction():
    img = cv2.imread(os.path.join('test', spath))
    img = autoroi(img)
    img = cv2.resize(img, (256, 256))
    img = np.reshape(img, [1, 256, 256, 3])

    prob = model.predict(img)
    Class = prob.argmax(axis=-1)

    return Class


def run():
    Class = prediction()
    if Class == 0:
        messagebox.showinfo('Prediction', 'You have been diagnosed with Glaucoma')
    else:
        messagebox.showinfo('Prediction', 'Congratulations! You are Healthy')


def run_all():
    x = os.listdir(test_path)
    y = []
    affected = 0

    for i in x:
        img = cv2.imread(os.path.join('test', i))
        img = autoroi(img)
        img = cv2.resize(img, (256, 256))
        img = np.reshape(img, [1, 256, 256, 3])

        prob = model.predict(img)
        Class = prob.argmax(axis=-1)
        y.append(Class[0])
        if Class == 1:
            affected += 1

    df = pandas.DataFrame(data=y, index=x, columns=['output'])
    df.to_csv('output.csv', sep=',')


def ROI():
    img = cv2.imread(os.path.join('test', spath))
    roi = autoroi(img)
    cv2.imshow('Region of Interest', roi)


def preview():
    img = cv2.imread(os.path.join('test', spath))
    cv2.imshow('Image', img)


def graph():
    total = len(os.listdir(test_path))
    affected = pandas.read_csv('output.csv')
    affected = affected['output'].sum()

    healthy = total - affected

    piey = ['Glaucomatous', 'Healthy']
    piex = [affected, healthy]

    plt.axis('equal')
    plt.pie(piex, labels=piey, radius=1.5, autopct='%0.1f%%', explode=[0.2, 0])
    plt.show()


# Frontend GUI


window = Tk()
window.title('Glaucoma Detection')
window.geometry('1000x550')
window.configure(background='grey')

l1 = Label(window, text='Test Image', font=('Arial', 20), padx=10, bg='grey')
l1.grid(row=0, column=0)

b1 = Button(window, text='Run', font=('Arial', 20), command=run)
b1.grid(row=1, column=3)

b2 = Button(window, text='Preview', font=('Arial', 20), command=preview)
b2.grid(row=1, column=2, rowspan=2, padx=10)

b2 = Button(window, text='ROI', font=('Arial', 20), command=ROI)
b2.grid(row=2, column=2, rowspan=3, padx=10)

b3 = Button(window, text='Run all', font=('Arial', 20), command=run_all)
b3.grid(row=2, column=3)

b4 = Button(window, text='Graph', font=('Arial', 20), command=graph)
b4.grid(row=3, column=3)

t1 = Listbox(window, height=20, width=60, selectmode=SINGLE, font=('Arial', 15), justify=CENTER)
t1.grid(row=1, column=0, rowspan=3, padx=10)
for filename in get_filenames():
    t1.insert(END, filename)
t1.bind('<<ListboxSelect>>', curselect)

sb1 = Scrollbar(window)
sb1.grid(row=1, column=1, rowspan=4)

t1.configure(yscrollcommand=sb1.set)
sb1.configure(command=t1.yview)

window.mainloop()
