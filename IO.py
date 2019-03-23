
import struct
import matplotlib.pyplot as plt
import os
from time import time
import shutil
#from parallel import Parallel
import multiprocessing as mp
import matplotlib
matplotlib.use("MacOSX")

class PotIO:

  train_filename = "1.0train-GB1.pot"
  test_fileName = "1.0test-GB1.pot"

  tag_data_dict = {} # {'tag_code':[sample_1,sample_2,...,]} , sample_n = [strokes_1,strokes_2,...], stroke_n = [v_1,v_2,...]
  # tag_data_dic is the data set itself, organized according to tag_code
  # tag_data_dict['0x21'] should give you all the samples of !
  # tag_data_dict['0x21'][0] gives you a sample (all the strokes) of one sample
  # tag_data_dict['0x21'][0][0] gives you one stroke
  # tag_data_dict['0x21'][0][0][0] gives you one vector from that stroke

  opt_file_dir = 'optFilesByTag'
  img_file_dir = 'imgsByTag'

  def __init__(self):
    self.characters = self.readFiles()
    self.organizeByTag()
    self.makeCharFile()
    #self.writeTagDicToPNG() ##This one is super slow



  def findThatBigImage(self):
    ''' find an image that has size larger than certain limit'''
    xylimit = 830
    for key in self.tag_data_dict.keys():
      for sample in self.tag_data_dict[key]:
        for stroke in sample:
          for v in stroke:
            if max(v[1],v[0]) > xylimit:
              for stroke in sample:
                plt.plot([x[0] for x in stroke],[x[1] for x in stroke])
              plt.show()



  def getSample(self,index):
    '''legacy, get a sample from sample pool'''
    return self.characters[index]




  def readFiles(self):
    '''read file, create internal representation of binary file data in ints'''
    print("decoding stroke files...")
    start = time()
    characters = []

    position = 0
    with open(self.test_fileName, "rb") as f:

      while True:
        position += 1
        sample_size = f.read(2)
        if sample_size == b'':
          break

        sample_size = struct.unpack("<H", sample_size)[0]
        # print("sample size:",sample_size)


        dword_code = f.read(2)
        if dword_code[0] != 0:
          dword_code = bytes((dword_code[1],dword_code[0]))

        tag_code = struct.unpack(">H", dword_code)[0]
        f.read(2) # next two hex are meaningless
        # print(hex(tag_code))
        try:  tag = struct.pack('>H', tag_code).decode("gb2312")[0]
        except:
          print("rip")
          f.read(sample_size - 2)
          continue
        tag_code = hex(tag_code)

        # print("tag code:",tag_code+',',"tag:",tag)

        stroke_number = struct.unpack("<H",f.read(2))[0]
        # print("stroke number:",stroke_number)

        strokes_samples= []
        stroke_samples = []
        current_stroke_number = 0
        next = b'\x00'
        while next != (b'\xff\xff', b'\xff\xff'):
          next = (f.read(2),f.read(2))
          if next == (b'\xff\xff', b'\x00\x00'):
            # print("stroke end")
            strokes_samples.append(stroke_samples)
            # print(strokes_samples)
            stroke_samples = []
            current_stroke_number += 1
          else:
            stroke_samples.append(((struct.unpack("<H",next[0])[0],struct.unpack("<H",next[1])[0])))
            current_stroke_number = 0

        sample = Sample(tag_code,tag,stroke_number,strokes_samples)
        sample.shrinkPixels()
        characters.append(sample)
    print("stroke file decoded in ", "%.3f"%(time() - start),"seconds.\n")
    return characters

  def organizeByTag(self):
    '''transform the internal representation of the data to a dictionary organized by tag'''
    print("Sorting data according to tags...")
    start = time()
    for char in self.characters:
      if char.tag_code not in self.tag_data_dict.keys():
        self.tag_data_dict[char.tag_code] = [char.stroke_data]
      else:
        self.tag_data_dict[char.tag_code].append(char.stroke_data)
    print("data sorted in", "%.3f" % (time() - start),"seconds.\n")

    #self.characters = None # Delete characters to save RAM


  def makeCharFile(self):
    '''put each character and its samples to files, prepare multi-thread reading
    or partial loading of the data sets'''

    # overview: tag $sample$sample
    # for sample: #strokes#strokes
    # for strokes: *stroke*stroke*stroke
    # for stroke: !x,y!x,y

    if not os.path.exists(self.opt_file_dir):
      os.mkdir(self.opt_file_dir)
      print("Optimized tag file directory made.")

    if len(os.listdir(self.opt_file_dir)) == 0:
      print("writing optimized tag files to optimized tag file directory...")
      start = time()
      for tagcode in self.tag_data_dict.keys():
        content = tagcode
        f = open(self.opt_file_dir+'/'+tagcode,'w')
        for sample in self.tag_data_dict[tagcode]:
          content += '$'
          for stroke in sample:
            content += '#'
            for v in stroke:
              content += '!'
              try:
                content += str(v[0]) + ',' + str(v[1])
              except Exception as e:
                print(e,tagcode,sample)
        f.write(content)
        f.close()
      print("Optimized tag files wrote in", "%.3f" % (time() - start),"seconds.\n" )

  def readOptFiles(self):
    '''read the optimized files'''
    dic = {}
    filelist = []
    for filename in filelist:

      f = open(filename, 'r')
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
      dic[filename] = sample_arr

  def writeTagDicToPNG(self):
    try: shutil.rmtree(self.img_file_dir)
    except: print("fail")
    os.mkdir(self.img_file_dir)

    print("Generating PNG files...")
    start = time()

    key_count = len(self.tag_data_dict.keys())
    thread_count = mp.cpu_count()
    key_list = list(self.tag_data_dict.keys())
    slice_size = key_count//thread_count
    divided_key_list = []
    print("Distributing",key_count,"image data packages to",thread_count, "threads")
    for i in range(thread_count-1):
      divided_key_list.append(key_list[i * slice_size: (i+1) * slice_size])
    divided_key_list.append(key_list[(thread_count-1) * slice_size:])

    parallel = Parallel()
    processes = [mp.Process(target=parallel.writeTagDicToPNG, args=(self.img_file_dir,self.tag_data_dict,keys)) for keys in divided_key_list]
    for p in processes:
      p.start()
    for p in processes:
      p.join()

    print("image files generated in", "%.3f" % (time() - start),"seconds.\n")


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