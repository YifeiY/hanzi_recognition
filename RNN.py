import numpy as np
import tensorflow as tf

tf.set_random_seed(1)


class RNN():
  trainset = {}
  testset = {}

  # dic['tag'] = [ Sample1: [stroke1: [TVs]]]

  def __init__(self):
    return

  def buildInternalRepresentationsFromDic(self,train_dic,test_dic):

    def buildInternalRepresentation(dic):
      labels = []
      samples = []
      iter = 1
      for key in dic.keys():
        iter += 1
        print(iter)
        for sample in dic[key]:
          labels.append(int(key, 16))
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
            stroke_rep[-1] = [int(stroke_rep[0][0]),int(stroke_rep[0][1]),int(stroke_rep[0][2]),int(stroke_rep[0][3]),stroke_rep[0][4],1]
            new_strokes += stroke_rep
          samples.append(new_strokes)
      print(np.array(samples).shape)
      print(np.array(labels).shape)
      return tf.data.Dataset.from_tensor_slices((np.array(samples),np.array(labels)))

    self.train_set = buildInternalRepresentation(train_dic)
    self.testset = buildInternalRepresentation(test_dic)