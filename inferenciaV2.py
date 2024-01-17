# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:11:40 2022

@author: dguti
"""

from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import time
import numpy as np

# modelo, etiqeutas e imagen de ejemplo
model_file = "xception_quant.tflite"
label_file = "cards_dataset_labels.txt"
image_file = "rey.jpg"

#%%

# Inicializar el interprete de tensorflow lite
interpreter = edgetpu.make_interpreter(model_file)
# reserva de los tensores
interpreter.allocate_tensors()

# Ajustamos datos entrada
size = common.input_size(interpreter) # datos entrada
image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)
img_array = np.array(image)
norm_im = (img_array/127.5)-1
norm_im = np.clip(norm_im,-1,1)

# Ejecutar inferencia
common.set_input(interpreter, norm_im)
#medimos tiempo:
inicio=time.time()
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)
fin = time.time()

#%% Comprobamos el resultados
labels = dataset.read_label_file(label_file)
for c in classes:
  print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
  
#imprimimos el tiempo que ha tardado la ejecucion:
tpo_trans = fin-inicio
print(f'El programa tardo {tpo_trans} segundos en ejecutarse')
  
