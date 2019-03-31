import struct
import matplotlib.pyplot as plt
import os
from time import time
import shutil
from parallel import Parallel
import multiprocessing as mp
import matplotlib

import threading
matplotlib.use("MacOSX")


opt_file_dir = 'optFilesByTag'
img_file_dir = 'imgsByTag'
train_filename = "1.0train-GB1.pot"
test_fileName = "1.0test-GB1.pot"


class PotIO:

  # {'tag_code':[sample_1,sample_2,...,]} , sample_n = [strokes_1,strokes_2,...], stroke_n = [v_1,v_2,...]
  # tag_data_dic is the data set itself, organized according to tag_code
  # tag_data_dict['0x21'] should give you all the samples of !
  # tag_data_dict['0x21'][0] gives you a sample (all the strokes) of one sample
  # tag_data_dict['0x21'][0][0] gives you one stroke
  # tag_data_dict['0x21'][0][0][0] gives you one vector from that stroke

  train_dict= {}
  test_dict = {}

  def __init__(self):
    print("PotIO object instantiated, call readFiles() to input train and test set")
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

    def readFile(filename):
      global data_buffer
      characters = []
      '''read file, create internal representation of binary file data in ints'''
      print("Decoding stroke files...")
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
          try:
            tag = struct.pack('>H', tag_code).decode("gb2312")[0]
          except:
            print("rip")
            f.read(sample_size - 2)
            continue
          tag_code = hex(tag_code)
          stroke_number = struct.unpack("<H", f.read(2))[0]

          strokes_samples = []
          stroke_samples = []
          current_stroke_number = 0
          next = b'\x00'
          while next != (b'\xff\xff', b'\xff\xff'):
            next = (f.read(2), f.read(2))
            if next == (b'\xff\xff', b'\x00\x00'):
              strokes_samples.append(stroke_samples)
              stroke_samples = []
              current_stroke_number += 1
            else:
              stroke_samples.append(((struct.unpack("<H", next[0])[0], struct.unpack("<H", next[1])[0])))
              current_stroke_number = 0

          sample = Sample(tag_code, tag, stroke_number, strokes_samples)
          sample.shrinkPixels()
          characters.append(sample)
      print("Stroke file decoded in ", "%.3f" % (time() - start), "seconds.\n")
      return characters

    def organizeByTag(characters):
      tag_dict = {}
      '''transform the internal representation of the data to a dictionary organized by tag'''
      print("Sorting data according to tags...")
      start = time()
      for char in characters:
        if char.tag_code not in tag_dict.keys():
          tag_dict[char.tag_code] = [char.stroke_data]
        else:
          tag_dict[char.tag_code].append(char.stroke_data)
      print("Data sorted in", "%.3f" % (time() - start), "seconds.\n")
      return tag_dict

    train_chars = readFile(train_filename)
    test_chars = readFile(test_fileName)
    self.train_dict = organizeByTag(train_chars)
    self.test_dict = organizeByTag(test_chars)
    print("train set stroke size:",len(self.train_dict.keys()))
    print("test set stroke size:",len(self.test_dict.keys()))

    train_chars = None
    test_chars = None






  def makeOptFile(self):
    '''put each character and its samples to files, prepare multi-thread reading
    or partial loading of the data sets'''

    # overview: tag $sample$sample
    # for sample: #strokes#strokes
    # for strokes: *stroke*stroke*stroke
    # for stroke: !x,y!x,y

    def makeOneOptFile(add_ons,tag_dict):
      if not os.path.exists(opt_file_dir + add_ons):
        os.mkdir(opt_file_dir + add_ons)
        print("Optimized tag file directory made.")

      if len(os.listdir(opt_file_dir + add_ons)) == 0:
        print("Writing optimized tag files to optimized tag file directory...")
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



  def readOptFiles(self):


    def readFileBundle(dir):
      manager = mp.Manager()
      dic = manager.dict()
      dic["broken"] = []
      filelist = os.listdir(dir)
      processes = []
      cpu_count = os.cpu_count()
      list_size = len(filelist)
      print("size of tags:",list_size)
      batch_size = list_size//cpu_count + 1
      end_point = 0
      for i in range(0,len(filelist),batch_size):
        filenames = filelist[i:i + batch_size]
        end_point = i + batch_size
        processes.append(mp.Process(target=readOneOptFile, args=(dir+"/",filenames,dic)))
      processes.append(mp.Process(target=readOneOptFile, args=(dir+"/",filelist[end_point:list_size],dic)))
      print(len(processes))
      for p in processes[:2]:
        p.start()
      for p in processes[:2]:
        p.join()







    def readOneOptFile(dir,filenames,d):
      for filename in filenames:
        f = open(dir + filename, 'r')
        data = f.read()
        temp = data.split('$')

        tag_code = temp[0]
        sample_arr = []

        for sample in temp[1:]:
          stroke_arr = []
          for stroke in sample.split('#')[1:]:
            v_arr = []
            for v in stroke.split('!')[1:]:
              v_arr.append([int(x) for x in v.split(',')])
            stroke_arr.append(v_arr)
          sample_arr.append(stroke_arr)
        try:
          d[filename] = sample_arr
        except:

          print('filename is ',filename + '\n')
          # d["broken"].append(filename)

    self.test_dict = readFileBundle(opt_file_dir + "Test")
    #self.train_dict = readFileBundle(opt_file_dir + "Train")

    #return train_dict,test_dict



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
    for strokes in self.stroke_data:
      for stroke in strokes:
        minx = min(minx,stroke[0])
        maxy = max(maxy,stroke[1])

    for strokes in self.stroke_data:
      for s in range(len(strokes)):
        strokes[s] = (strokes[s][0] - minx,maxy - strokes[s][1])