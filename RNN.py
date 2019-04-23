import numpy as np
from time import time
from keras.layers import GRU,Bidirectional,Dense,Dropout,AveragePooling1D,Flatten
from keras.models import Sequential,load_model
from keras.initializers import Constant
from matplotlib import pyplot as plt
# feature dimension, GRU stack depth, Dense, output classes

data_fixed_length = 100 # this is the fixed length of vectors in a character
number_of_classes = 3740 # this is the size of the shrunk dictionary.keys() size in PotIO class in IO.py

epochs = 10
batch_size = 256

##net 6: 6-> [100,300,500] -> 100 -> 50
class RNN():
  train_set = []
  test_set = []
  test_labels = []
  train_labels = []

  # dic['tag'] = [ Sample1: [stroke1: [TVs]]]

  def __init__(self):
    print("initiated RNN object, call ")
    return

  def exec(self):

    self.loadInternalRepresentationFiles()
    print('transforming data')
    self.augumentDataSets()
    self.toNpArrs()

    self.model = self.buildRNN()
    print('Starting training')
    self.history = self.model.fit(self.train_set,self.train_labels,validation_data=(self.test_set,self.test_labels),
              batch_size=batch_size,epochs = epochs,verbose= 1)
    self.model.save("RNNmodel.h5")

    #self.plotHistory()
  
    
  def buildRNN(self):
    print('Building model')
    model = Sequential()
    model.add(Bidirectional(GRU(500, return_sequences=True),merge_mode='sum'))
    model.add(Bidirectional(GRU(300, return_sequences=True),merge_mode='sum'))
    model.add(Flatten())
    model.add(Dense(number_of_classes,activation='softmax'))
    print('compiling model')
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
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
          dataset[i] = dataset[i] + [[0] * 6] * (data_fixed_length - len(dataset[i]))
    augumentDataSet(self.train_set)
    augumentDataSet(self.test_set)


  def buildInternalRepresentationsFromDic(self,train_dic,test_dic):
    '''this build representation file from files built by IO.py'''
    print("building dictionary from  IO class")
    def buildInternalRepresentation(dic):
      labels = []
      samples = [[0]]
      n_keys = len(dic.keys())
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
    print("internal representation has been built, call self.saveInternalRepresentationFiles() to save these representation")
    print("self.loadInternalRepresentationFiles() will load the saved files to RNN ")

  def saveInternalRepresentationFiles(self):
    start = time()
    print('saving representation files...')
    np.save('trainset',self.train_set)
    np.save('trainlabels',self.train_labels)
    np.save('testset', self.test_set)
    np.save('testlabel', self.test_labels)
    print("4 representation files saved in",time() - start,"seconds")

  def loadInternalRepresentationFiles(self):
    start = time()
    print('reading representation files...')
    self.train_set = np.load("trainset.npy")
    self.train_labels = np.load('trainlabels.npy')
    self.test_set = np.load('testset.npy')
    self.test_labels = np.load('testlabel.npy')
    print("4 np files read in",time() - start,"seconds")
    self.convertLabelsToKeys()
    self.train_labels = np.reshape(np.array(self.train_labels),(len(self.train_labels),1))
    self.test_labels = np.reshape(np.array(self.test_labels),(len(self.test_labels),1))
    print("representation files have been loaded")
    print("call self.exec() to start training.")
    print("model configuration can be modified in buildRNN()")




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
    def toClassArray(labels):
      new_labels = [[]]
      for l in labels:
        new_i = [0] * number_of_classes
        new_i[l] = 1
        new_labels.append(new_i)
      return new_labels[1:]

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

  def plotHistory(self):
    # list all data in history
    history = self.history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

def continueTraining(batch_size, n_epoch):
  rnn = RNN()
  rnn.loadInternalRepresentationFiles()
  print('transforming data')
  rnn.augumentDataSets()
  rnn.toNpArrs()
  print('continue training')
  model = load_model("RNNmodel.h5")
  rnn.history = model.fit(rnn.train_set, rnn.train_labels, validation_data=(rnn.test_set, rnn.test_labels),
            batch_size=batch_size, epochs=n_epoch, verbose=1,callbacks=[PlotLossesKeras()])
  model.save("RNNmodel.h5")    


rnn = RNN()
rnn.exec()