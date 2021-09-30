import FADS.FeatureSelector as fs
from FADS.input import get_dset
from matplotlib.pyplot import show

dset = get_dset("/home/shussain/Simulated_data/13072021/0.5/sim_10/PS/output")
ln = fs.LassoNet(dset, (500, 100,))
print(ln.top_n(100))
ln.manhattan()
show()