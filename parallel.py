import os
import shutil
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib
matplotlib.use('agg')

class Parallel:
  def __init__(self):
    self.buffer = []

  def writeTagDicToPNG(self,img_file_dir,tag_data_dict,keys):
    print("Generating img files in parallel...")
    leftlim = 0
    rightlim = 500
    maxx = 300
    maxy = 300
    for key in keys:
      print(key)
      char = tag_data_dict[key]
      for i in range(len(char)):
        for stroke in char[i]:
          axes = plt.gca()
          axes.set_xlim([leftlim, rightlim])
          axes.set_ylim([leftlim, rightlim])
          plt.plot([x[0] for x in stroke], [x[1] for x in stroke])
        plt.savefig(img_file_dir + '/' + key + '-' + str(i) + '.png')
        plt.clf()