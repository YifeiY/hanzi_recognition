from IO import PotIO
from IO import Sample
from RNN import RNN
# io = PotIO()
# io.readFiles()
# rnn = RNN()
# rnn.buildInternalRepresentationsFromDic(io.train_dict,io.test_dict)

class Test():
  def __init__(self):
    io = PotIO()
    rnn = RNN()

    io.readFiles()
    rnn.buildInternalRepresentationsFromDic(io.train_dict, io.test_dict)


  def execute(self):
    return


Test()