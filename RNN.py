import numpy as np
import tensorflow as tf
from time import time
tf.set_random_seed(1)
from keras.layers import GRU
from keras.models import Sequential
from matplotlib import pyplot as plt

class RNN():
  train_set = []
  test_set = []
  test_labels = []
  train_labels = []

  # dic['tag'] = [ Sample1: [stroke1: [TVs]]]

  def __init__(self):
    return

  def buildRNN(self):
      return

  def buildInternalRepresentationsFromDic(self,train_dic,test_dic):

    def buildInternalRepresentation(dic):
      labels = []
      samples = []
      print("number of keys in dic:",len(dic.keys()))
      for key in dic.keys():
        for sample in dic[key]:
          labels.append(key)
          new_strokes = []
          for stroke in sample:
            # [x,y,dx,dy,pen down, pen up]
            stroke_rep = []
            if len(stroke) == 1:
              stroke_rep = [[stroke[0][0],stroke[0][1],0,0,1,1]]
            else:
              for i in range(0,len(stroke)-1):
                c = stroke[i]
                n = stroke[i + 1]
                stroke_rep.append([int(c[0]),int(c[1]),int(n[0]-c[0]),int(n[1]-c[1]),0,0])
              # pen down stroke
              stroke_rep[0] = [int(stroke_rep[0][0]),int(stroke_rep[0][1]),int(stroke_rep[0][2]),int(stroke_rep[0][3]),1,stroke_rep[0][5]]
              # pen up stroke
              stroke_rep[-1] = [int(stroke_rep[-1][0]),int(stroke_rep[-1][1]),int(stroke_rep[-1][2]),int(stroke_rep[-1][3]),stroke_rep[-1][4],1]
            new_strokes += stroke_rep
          samples.append(new_strokes)
      return samples,labels

    self.train_set,self.train_labels = buildInternalRepresentation(train_dic)
    self.test_set,self.test_labels = buildInternalRepresentation(test_dic)

  def saveInternalRepresentationFiles(self):
    start = time()
    print('saving np files...')
    np.save('trainset',self.train_set)
    np.save('trainlabels',self.train_labels)
    np.save('testset', self.test_set)
    np.save('testlabel', self.test_labels)
    print("4 np files saved in",time() - start,"seconds")

  def loadInternalRepresentationFiles(self):
    start = time()
    print('reading np files...')
    self.train_set = np.load("trainset.npy")
    self.train_labels = np.load('trainlabels.npy')
    self.test_set = np.load('testset.npy')
    self.test_labels = np.load('testlabel.npy')
    print("4 np files read in",time() - start,"seconds")

  def show(self,i):
    for stroke in self.train_set[i]:
      if stroke [-2] == 1:
        new_stroke = []
      if stroke [-1] == 1:
        new_stroke.append([stroke[0],stroke[1]])
        plt.plot([x[0] for x in new_stroke],[x[1] for x in new_stroke])
      else:
        new_stroke.append([stroke[0],stroke[1]])
    plt.show()

