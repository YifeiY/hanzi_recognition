from IO import PotIO
from IO import Sample

io = PotIO()
io.readOptFiles()
print(len(io.test_dict.keys()))

