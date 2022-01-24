import numpy as np
import pylab as pl

x = []  # Make an array of x values
y = []  # Make an array of y values for each x value

for i in range(-128,127):
    x.append(i)

for j in range(-128,127):
    temp  = j *(2**(1 - abs((j/128))))
    y.append(temp)
# print('y',y)

# pl.xlim(-128, 127)# set axis limits
# pl.ylim(-128, 127)
pl.axis([-128, 127,-128, 127])

pl.title('S-model Curve Function ',fontsize=20)# give plot a title
pl.xlabel('Input Value',fontsize=20)# make axis labels
pl.ylabel('Output Value',fontsize=20)




pl.plot(x, y,color='red')  # use pylab to plot x and y
pl.show()  # show the plot on the screen