import struct
import matplotlib.pyplot as plt
import os
from time import time
import multiprocessing as mp
import matplotlib
from math import cosh
from copy import deepcopy
import threading
matplotlib.use("MacOSX")


opt_file_dir = 'optFilesByTag'
img_file_dir = 'imgsByTag'
train_filename = "1.0train-GB1.pot"
test_fileName = "1.0test-GB1.pot"
data_upper_bound = 5

class PotIO:
  ###OPT FILE RELATED FUNCTIONS ARE NOT IMPLEMENTED YET
  ## Call readFiles() to input sets

  # {'tag_code':[sample_1,sample_2,...,]} , sample_n = [strokes_1,strokes_2,...], stroke_n = [v_1,v_2,...]
  # tag_data_dic is the data set itself, organized according to tag_code
  # tag_data_dict['0x21'] should give you all the samples of !
  # tag_data_dict['0x21'][0] gives you a sample (all the strokes) of one sample
  # tag_data_dict['0x21'][0][0] gives you one stroke
  # tag_data_dict['0x21'][0][0][0] gives you one vector from that stroke

  train_dict= {}
  test_dict = {}

  def __init__(self):
    print("PotIO object instantiated, call readFiles() to input train and test set\n")
    return

  def getTrainTest(self):
    return self.train_dict, self.test_dict


  def findThatBigImage(self):
    ''' find an image that has size larger than certain limit'''
    xylimit = 830
    print(21)
    for key in self.train_dict.keys():
      for sample in self.train_dict[key]:
        for stroke in sample:
          for v in stroke:
            if max(v[1],v[0]) > xylimit:
              for stroke in sample:
                plt.plot([x[0] for x in stroke],[x[1] for x in stroke])
              plt.show()


  def readFiles(self):
    # read train and test data
    print("Reading files...")
    def readFile(filename):
      global data_buffer
      characters = []
      '''read file, create internal representation of binary file data in ints'''
      print("Decoding",filename, "stroke files...\n"
            "Data will be normalized\n"
            "Redundant points in stroke will be removed\n")
      start = time()

      with open(filename, "rb") as f:
        while True:
          sample_size = f.read(2)
          if sample_size == b'':
            break

          sample_size = struct.unpack("<H", sample_size)[0]
          dword_code = f.read(2)
          if dword_code[0] != 0:
            dword_code = bytes((dword_code[1], dword_code[0]))
          tag_code = struct.unpack(">H", dword_code)[0]
          f.read(2)
          # try:
          tag = struct.pack('>H', tag_code).decode("gb2312")[0]
          # except:
          #   print("rip")
          #   f.read(sample_size - 2)
          #   continue
          tag_code = hex(tag_code)
          stroke_number = struct.unpack("<H", f.read(2))[0]

          strokes_samples = []
          stroke_samples = []
          next = b'\x00'
          while next != (b'\xff\xff', b'\xff\xff'):
            next = (f.read(2), f.read(2))
            if next == (b'\xff\xff', b'\x00\x00'):
              strokes_samples.append(stroke_samples)
              stroke_samples = []
            else:
              stroke_samples.append(((struct.unpack("<H", next[0])[0], struct.unpack("<H", next[1])[0])))

          sample = Sample(tag_code, tag, stroke_number, strokes_samples)
          sample.shrinkPixels()
          sample.normalize(128)
          sample.removeRedundantPoints()
          characters.append(sample)

      print("Stroke file decoded in ", "%.3f" % (time() - start), "seconds.\n")
      return characters

    def organizeByTag(characters):
      tag_dict = {}
      '''transform the internal representation of the data to a dictionary organized by tag'''
      print("Sorting data according to tags...\n")
      start = time()
      for char in characters:
        if char.tag_code not in tag_dict.keys():
          tag_dict[char.tag_code] = [char.stroke_data]
        else:
          tag_dict[char.tag_code].append(char.stroke_data)
      print("Data sorted in", "%.3f" % (time() - start), "seconds.\n")
      return tag_dict

    train_chars = readFile(train_filename)
    print("read",len(train_chars),"number of train characters.\n")
    test_chars = readFile(test_fileName)
    print("read",len(test_chars),"number of test characters.\n")
    self.train_dict = organizeByTag(train_chars)
    self.test_dict = organizeByTag(test_chars)
    print("train set character size:",len(self.train_dict.keys()))
    print("test set character size:",len(self.test_dict.keys()))

    train_chars = None
    test_chars = None
    print("Finished reading files, call self.shrinkDics(size) to reduce the number of classes")
    print("Call RNN.buildInternalRepresentationsFromDic(IO.train_dict, IO.test_dict) to load the input to RNN")


  def makeOptFile(self):
    '''legacy - put each character and its samples to files'''

    # overview: tag $sample$sample
    # for sample: #strokes#strokes
    # for strokes: *stroke*stroke*stroke
    # for stroke: !x,y!x,y

    def makeOneOptFile(add_ons,tag_dict):
      if not os.path.exists(opt_file_dir + add_ons):
        os.mkdir(opt_file_dir + add_ons)
        print("Optimized tag file directory made.\n")

      if len(os.listdir(opt_file_dir + add_ons)) == 0:
        print("Writing optimized tag files to optimized tag file directory...\n")
        start = time()
        for tagcode in tag_dict.keys():
          content = tagcode
          f = open(opt_file_dir + add_ons + '/' + tagcode, 'w')
          for sample in tag_dict[tagcode]:
            content += '$'
            for stroke in sample:
              content += '#'
              for v in stroke:
                content += '!'
                try:
                  content += str(v[0]) + ',' + str(v[1])
                except Exception as e:
                  print(e, tagcode, sample)
          f.write(content)
          f.close()
        print("Optimized tag files wrote in", "%.3f" % (time() - start), "seconds.\n")

    makeOneOptFile("Train",self.train_dict)
    makeOneOptFile("Test",self.test_dict)


  def shrinkDics(self,new_size):
    '''shrink class size down to certain size'''
    new_keys = list(self.train_dict.keys())[:new_size]
    def shrinkDic(dic,keys):
      new_dic = {}
      for key in keys:
        new_dic[key] = dic[key]
      return new_dic
    self.train_dict = shrinkDic(self.train_dict,new_keys)
    self.test_dict = shrinkDic(self.test_dict,new_keys)



class Sample:

  def __init__(self,tag_code,tag,stroke_number,stroke_data):
    self.tag_code= tag_code
    self.tag = tag
    self.stroke_number =stroke_number
    self.stroke_data = stroke_data # strokes make up the character
    return

  def show(self):
    '''plots the character using matplotlib'''
    for stroke in self.stroke_data:
      plt.plot([p[0] for p in stroke],[p[1] for p in stroke])
    plt.show()

  def shrinkPixels(self):
    '''normalize the pixel values to a minimum of 0,
    eg. (1234,2345) -> (34,45) so that the character has minimum coordinates of (0,_),(_,0)'''
    minx = self.stroke_data[0][0][0]
    maxy = 0
    for stroke in self.stroke_data:
      for v in stroke:
        minx = min(minx,v[0])
        maxy = max(maxy,v[1])

    for strokes in self.stroke_data:
      for s in range(len(strokes)):
        strokes[s] = (strokes[s][0] - minx,maxy - strokes[s][1])

  def normalize(self,upper_bound):
    bounds = [self.stroke_data[0][0][0],self.stroke_data[0][0][1]]
    for stroke in self.stroke_data:
      for v in stroke:
        bounds = [max(bounds[0],v[0]),max(bounds[1],v[1])]
    bound = max(bounds)
    for stroke in self.stroke_data:
      for i in range(len(stroke)):
        stroke[i] = (stroke[i][0]/bound * upper_bound, stroke[i][1]/bound * upper_bound)

  def removeRedundantPoints(self):
    new_stroke_data = []
    for stroke in self.stroke_data:
      new_stroke = [stroke[0]]
      # add the stroke if it only contains one point
      if len(stroke) == 1:
        new_stroke_data.append(new_stroke)
        continue
      if (new_stroke[-1][1] - stroke[1][1]) == 0: last_cos = 100
      else:last_cos = - cosh((new_stroke[-1][0] - stroke[1][0]) / (new_stroke[-1][1] - stroke[1][1]))

      for i in range(1,len(stroke)-1):
        # save the stroke if it represents a large euclidean change
        dx = new_stroke[-1][0] - stroke[i][0]
        dy = new_stroke[-1][1] - stroke[i][1]
        if dy == 0: this_cos = 100
        else: this_cos = cosh(dx/dy)
        if dx**2 + dy **2 > 100 or abs(this_cos - last_cos) > 0.2:
          new_stroke.append(stroke[i])
        last_cos = this_cos
      new_stroke.append(stroke[-1])
      for i in range(len(stroke)):
        new_v = (int(stroke[i][0]), int(stroke[i][1]))
        stroke[i] = new_v
      new_stroke_data.append(new_stroke)
    self.stroke_data = new_stroke_data
