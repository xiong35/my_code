import matplotlib.pyplot as plt
import numpy as np
# demo1
""" x = np.linspace(0,10,30)
plt.plot(x,np.sin(x),'--') """

# demo2
""" rng = np.random.RandomState(7)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
size = 2000 * rng.rand(100)
plt.scatter(x,y,c = colors,s = size,alpha = 0.25,cmap ='viridis')
plt.colorbar() """

# demo3
""" x = np.linspace(0,10,50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)
plt.errorbar(x,y,xerr = 0.2,yerr = dy,fmt ='.k') """

# demo4
""" np.random.seed(7)
data = np.random.randn(1000)
plt.hist(data,bins = 30,histtype='stepfilled',density=True) """

# demo5
""" x1 = np.random.normal(0,0.8,1000)
x2 = np.random.normal(-2,1,1000)
x3 = np.random.normal(3,2,1000)

kwargs = dict(alpha=0.25,bins = 30,density=True)
plt.hist(x1,**kwargs)
plt.hist(x2,**kwargs)
plt.hist(x3,**kwargs) """

# demo6
""" mean = [0, 0]
cov = [[1, 2], [1, 1]]
x,y = np.random.multivariate_normal(mean,cov,2000).T
# plt.hist2d(x,y,bins = 300)
plt.hexbin(x,y,gridsize=20) """


"""
demo7
 """

x = np.linspace(0, 10, 100)
y = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y, "--", label="sin(x)")
plt.plot(x, y2, "-bo", c = 'dimgray',label="coS(x)")
plt.ylim(-1.5, 1.5)
plt.xlabel("variable x")
plt.ylabel("variable y")
plt.title("sin")
# if you were wrong, vsc will automatically tall you
# what fit the func
plt.grid(alpha=0.25)
# axis vertical/horizontal line:
plt.axvline(x=np.pi/2, ls='-.', c='r', alpha=0.25)
plt.axhspan(ymin=-0.5, ymax=0.5, facecolor='b', alpha=0.25)
# illustration
plt.text(np.pi, 0, 'sin(x)', weight='bold', color='#ff8c00')
plt.annotate('maximum', xy=(np.pi/2, 1),
             xytext=(np.pi/2+2,.5 ), weight='bold',
             c='dimgray', arrowprops=dict(arrowstyle='fancy',
                                          connectionstyle='angle3', color='gray'))
plt.legend(loc='upper center',frameon=True,ncol=2)



plt.show()
