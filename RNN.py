import numpy as np
from time import time
from keras.layers import GRU,Bidirectional,Dense,Dropout,AveragePooling1D,Flatten
from keras.models import Sequential
from keras.initializers import Constant
from matplotlib import pyplot as plt


# feature dimension, GRU stack depth, Dense, output classes

data_fixed_length = 100
number_of_classes = 100 # this is the size of the shrunk dictionary.keys() size in PotIO class in IO.py
config = [6,100,200,number_of_classes]

epochs = 60
batch_size = 1000

class RNN():
  train_set = []
  test_set = []
  test_labels = []
  train_labels = []

  # dic['tag'] = [ Sample1: [stroke1: [TVs]]]

  def __init__(self):
    return

  def exec(self):

    self.loadInternalRepresentationFiles()
    print('transforming data')
    self.augumentDataSets()
    self.toNpArrs()
    model = self.buildRNN()
    print('Starting training')
    model.fit(self.train_set,self.train_labels,validation_data=(self.test_set,self.test_labels),batch_size=batch_size,epochs = epochs)


  def buildRNN(self):
    input_n = config[0]
    stack_depth = config[1]
    dense_n = config[2]
    classes = config[3]

    print('Building model')
    model = Sequential()
    model.add(Bidirectional(GRU(1,input_shape=(data_fixed_length,6),return_sequences=True,bias_initializer=Constant(value=5))))
    for i in range(stack_depth-2):
      model.add(Bidirectional(GRU(1,return_sequences=True,bias_initializer=Constant(value=5))))
    model.add(Bidirectional(GRU(1, return_sequences=True,bias_initializer=Constant(value=5)),merge_mode='sum'))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(number_of_classes))
    print('compiling model')
    model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
    return model

  def toNpArrs(self):
    def toNpArr(s):
      new = []
      for i in s:
        temp = []
        for j in i:
          temp.append(np.asarray(j))
        new.append(np.asarray(temp))
      return np.array(new)
    self.train_set = toNpArr(self.train_set)
    self.test_set = toNpArr(self.test_set)



  def augumentDataSets(self):
    def augumentDataSet(dataset):
      for i in range(len(dataset)):
        if len(dataset[i]) > data_fixed_length:
          dataset[i] = dataset[i][:data_fixed_length]
        else:
          dataset[i] = dataset[i] + [[-1] * 6] * (data_fixed_length - len(dataset[i]))
    augumentDataSet(self.train_set)
    augumentDataSet(self.test_set)


  def buildInternalRepresentationsFromDic(self,train_dic,test_dic):
    def buildInternalRepresentation(dic):
      labels = []
      samples = [[0]]
      n_keys = len(dic.keys())
      config[-1] = n_keys
      print("number of keys in dic:",n_keys)

      for key in dic.keys():
        for sample in dic[key]:
          labels.append(key)
          new_strokes = [[]]
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
          samples += [new_strokes[1:]]
      return samples[1:],labels

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
    self.convertLabelsToKeys()
    self.train_labels = np.reshape(np.array(self.train_labels),(len(self.train_labels),1))
    self.test_labels = np.reshape(np.array(self.test_labels),(len(self.test_labels),1))



  def convertLabelsToKeys(self):
    def defineDict(labels):
      label_dic = {}
      current_index = 0
      for l in labels:
        if l not in label_dic.keys():
          label_dic[l] = current_index
          current_index += 1
      return label_dic,{v: k for k, v in label_dic.items()}
    self.l2k,self.k2l = defineDict(self.test_labels)

    self.train_labels = [self.l2k[i] for i in self.train_labels]
    self.test_labels = [self.l2k[i] for i in self.test_labels]



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


rnn = RNN()
rnn.exec()


