import seaborn as sns
import matplotlib.pyplot as plt

cp = sns.color_palette("Set1")

fig = plt.figure()
leg = plt.figure(figsize=(3,2))
lines=[plt.plot([], color=cp[0], lw=10, mec='black')[0], plt.plot([], color=cp[1], lw=10)[0]]
leg.legend(lines, ["p value thresholding", "LassoNet"],loc='center')
leg.savefig("leg.jpeg")

leg.show()
