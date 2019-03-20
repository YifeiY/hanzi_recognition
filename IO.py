
import struct
import sys
from codecs import decode


class PotIO:

  train_filename = "1.0train-GB1.pot"
  test_fileName = "1.0test-GB1.pot"
  dev_filename = "001.pot"
  dev_characters = None
  def __init__(self):
    self.characters = self.readFiles()


  def getSample(self,index):
    return self.characters[index]

  def readFiles(self):
    byte_order = sys.byteorder
    print("reading files, please wait...")
    test_data = []
    #train_file = open(self.train_filename,'r')
    characters = []

    position = 0
    with open(self.train_filename, "rb") as f:

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
        characters.append(Sample(tag_code,tag,stroke_number,strokes_samples))
    return characters






class Sample:

  def __init__(self,tag_code,tag,stroke_number,stroke_data):
    self.tag_code= tag_code
    self.tag = tag
    self.stroke_number =stroke_number
    self.stroke_data = stroke_data
    return






  def construct_image(self,stroke_number, stroke_data):
    return