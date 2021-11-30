import PIL.Image
from PIL import Image, ImageDraw
from math import ceil
import numpy as np
from io import BytesIO
import IPython.display

def imshow(a, format='png', jpeg_fallback=True):
  a = np.asarray(a, dtype=np.uint8)
  str_file = BytesIO()
  PIL.Image.fromarray(a).save(str_file, format)
  im_data = str_file.getvalue()
  try:
    disp = IPython.display.display(IPython.display.Image(im_data))
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print ('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format)
      return imshow(a, format='jpeg')
    else:
      raise
  return disp
def createImageGrid(images, scale=0.25, rows=1):
   w,h = images[0].size
   w = int(w*scale)
   h = int(h*scale)
   height = rows*h
   cols = ceil(len(images) / rows)
   width = cols*w
   canvas = PIL.Image.new('RGBA', (width,height), 'white')
   for i,img in enumerate(images):
     img = img.resize((w,h), PIL.Image.ANTIALIAS)
     canvas.paste(img, (w*(i % cols), h*(i // cols)))
   return canvas

def show_four_results(result_dir="/validation_stimuli",start_no=0, coeff = 1.0):
    images =[]
    for n in range(4):
        images+=[PIL.Image.open(result_dir+'/images/'+str(n+start_no)+sufix) for sufix in ['neg'+str(coeff)+'.png','neu.png','pos'+str(coeff)+'.png']]
    imshow(createImageGrid(images,0.4,4))