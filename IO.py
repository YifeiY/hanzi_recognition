
import struct
import sys
from codecs import decode
import matplotlib.pyplot as plt
import os

class PotIO:

  train_filename = "1.0train-GB1.pot"
  test_fileName = "1.0test-GB1.pot"
  dev_filename = "001.pot"
  dev_characters = None
  tag_data_dict = {} # {'tag_code':[sample_1,sample_2,...,]} , sample_n = [strokes_1,strokes_2,...], strokes_n = [v_1,v_2,...]
  opt_file_dir = 'optFilesByTag'
  def __init__(self):
    self.characters = self.readFiles()
    self.organizeByTag()
    self.makeCharFile()


  def getSample(self,index):
    return self.characters[index]

  def readFiles(self):
    byte_order = sys.byteorder
    print("reading files, please wait...")
    test_data = []
    #train_file = open(self.train_filename,'r')
    characters = []

    position = 0
    with open(self.dev_filename, "rb") as f:

      while True:
        print("{:,}".format(position))
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
    return characters

  def organizeByTag(self):
    for char in self.characters:
      if char.tag_code not in self.tag_data_dict.keys():
        self.tag_data_dict[char.tag_code] = [[char.stroke_data]]
      else:
        self.tag_data_dict[char.tag_code].append(char.stroke_data)
    #self.characters = None # Delete characters to save RAM


  def makeCharFile(self):
    # overview: tag $sample$sample
    # for sample: #strokes#strokes
    # for strokes: *stroke*stroke*stroke
    # for stroke: !x,y!x,y

    if not os.path.exists(self.opt_file_dir):
      os.mkdir(self.opt_file_dir)
      print("making optimized tag file directory")

    if len(os.listdir(self.opt_file_dir)) == 0:
      print("writing optimized tag files to optimized tag file directory")
      for tagcode in self.tag_data_dict.keys():
        content = tagcode
        f = open(self.opt_file_dir+'/'+tagcode,'w')
        for sample in self.tag_data_dict[tagcode]:
          content += '$'
          for strokes in sample:
            content += '#'
            for stroke in strokes:
              content += '*'
              for v in stroke:
                content += '!'
                try:
                  content += str(v[0]) + ',' + str(v[1])
                except:
                  print("isjaodfim",v)
        f.write(content)
        f.close()
      print("finished writing optimized tag files")



class Sample:

  def __init__(self,tag_code,tag,stroke_number,stroke_data):
    self.tag_code= tag_code
    self.tag = tag
    self.stroke_number =stroke_number
    self.stroke_data = stroke_data # strokes make up the character
    return


  def show(self):
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


  def construct_image(self,stroke_number, stroke_data):
    return