
import struct
import sys
from codecs import decode


class PotIO:

  train_filename = "1.0train-GB1.pot"
  test_fileName = "1.0test-GB1.pot"
  dev_filename = "001.pot"

  def __init__(self):
    self.readFiles()



  def readFiles(self):
    byte_order = sys.byteorder
    test_data = []
    #train_file = open(self.train_filename,'r')

    with open(self.dev_filename, "rb") as f:
      while True:
        b_this_sample_size = f.read(2)

        print("b this sample size",b_this_sample_size)
        if b_this_sample_size == b'':
          break
        us_this_sample_size = self.h2ud(b_this_sample_size)
        print("this sample byte size:",us_this_sample_size)

        this_data = f.read(us_this_sample_size)

        tag_code = this_data[:2]
        tag_code = hex()
        print("tag_code:",tag_code.hex(),"parsed",tag_code.decode("GBK"))

        b_stroke_number = this_data[4:6]
        print(b_stroke_number)
        us_stroke_number = self.h2ud(b_stroke_number)
        print("stroke number",us_stroke_number)

        strokes = []
        stroke = []
        for i in range(0,us_stroke_number,4):
          decoded = []
          next = [this_data[6+i:6+i+2],this_data[6+i+2:6+i+4]]
          decoded = [(int(next[0][0]),int(next[0][1])),(int(next[1][0]),int(next[1][1]))]

          if decoded == [(255,255),(0,0)]: # end of stroke
            strokes.append(stroke)
            stroke = []
          else:
            stroke.append(decoded)
          if decoded == [(255, 255), (255, 255)]:
            break
        print(len(strokes[0]))
        for i in strokes:
          print(i)

        tag_code = 0
        exit()



  def h2ud(self,bs):
    '''reverse and converts hex to unsigned int'''
    bs = str(bs.hex())
    val = 0
    dict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a':10,'b':11,'c':12,'d':13,'e':14,'f':15}
    for i in range(len(bs)):
      val += dict[bs[i]] * (16 ** i)
    return val


class Sample:

  def __init__(self,tag_code,stroke_number,stroke_data):

    return




  def construct_image(self,stroke_number, stroke_data):
    return