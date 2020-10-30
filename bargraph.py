import numpy as np
import matplotlib.pyplot as plt

'''x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15]
y1 = [28.22, 18.13,32.30, 17.46, 23.28, 28.78, 42.90, 15.19, 34.05, 44.61, 24.31, 21.93, 49.73, 9.06, 50.32]
y2 = [31.52, 25.97, 39.46, 21.67, 24.61, 32.98, 41.88, 19.55, 39.26, 52.25, 37.21, 25.77, 50.06, 13.93, 51.33]
#labels = ['Frogs', 'Hogs', 'Bogs', 'Slogs']
labels = ['Adirondack', 'ArtL', 'Jadeplant', 'Motorcycle', 'MotorcycleE', 'Piano', 'PianoL', 'Pipes', 'Playroom', 'Playtable', 'PlaytableP', 'Recycle', 'Shelves', 'Teddy', 'Vintage']


#bx[index_plot].plot(x[::-1],y[::-1], marker=positions[paths]['marker'], markersize=positions[paths]['markersize'],linestyle='dotted', c=positions[paths]['color'], label=paths)

plt.plot(x, y1, marker='o', linestyle='dotted', color='blue', markersize='2', label="17x17 Adaptive")
plt.plot(x, y2, marker='o', linestyle='dotted', color='red', markersize='2', label="17x17")
# You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(x, labels, rotation='vertical')
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.03)
plt.legend(ncol=1, loc='lower right')  
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
plt.show()

exit()

names = ['Adirondack', 'ArtL', 'Jadeplant', 'Motorcycle']
values = [1, 10, 100, 10]

fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(111)
plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
#plt.suptitle('Categorical Plotting')
#ax.set_xticklabels( ('Adirondack', 'ArtL', 'Jadeplant', 'Motorcycle' ), rotation=45 )
plt.show()'''


N = 15
xvals = range(N)
ind = np.arange(N)  # the x locations for the groups
ind = (ind)
print(ind)
width = 0.10       # the width of the bars
margin = 0.00

fig = plt.figure()
#plt.grid(color="")
#plt.xlim(-0.05, 0.3)
plt.xlim([min(xvals) - 0.08, max(xvals) + 0.64])
#plt.subplots_adjust(wspace=0,hspace=0)
#plt.xlim(left=-0.05)
#plt.xlim(right=0.3)
ax = fig.add_subplot(111)
#plt.rc('grid', linestyle="--", color='#f2f2f2')


#bx = fig.add_subplot(111)



yvals = [29.59, 23.64, 37.11, 18.36, 20.97, 30.67, 40.82, 17.68, 37.08, 48.48, 30.24, 23.95, 49.70, 12.84, 49.84]
rects1 = ax.bar(ind, yvals, width, color='white', hatch="////", align="center", edgecolor='black')

zvals = [27.34, 17.97, 32.13, 16.79, 22.57, 28.25, 42.79, 15.16, 33.59, 44.45, 24.17, 21.46, 49.75, 9.06, 49.97]
rects2 = ax.bar(ind+width+margin, zvals, width, color='#d9d9d9', align="center", edgecolor='black')

kvals = [24.52, 49.12, 38.49, 18.33, 41.16, 38.30, 81.26, 19.63, 33.78, 41.96, 24.04, 28.57, 47.90, 15.06, 51.75]
rects3 = ax.bar(ind+width*2+margin*2, kvals, width, color='white', hatch="--", align="center", edgecolor='black')
#gvals = [28.22, 18.13,32.30, 17.46, 23.28, 28.78, 42.90, 15.19, 34.05, 44.61, 24.31, 21.93, 49.73, 9.06, 50.32]
#rects4 = ax.bar(ind+width*3+margin*3, gvals, width, color='gray', align="center", edgecolor='black')

#dvals = [27.06,16.05,23.2,19.30,24.65, 29.44,39.77,15.85,28.91,43.75,23.92,24.36,47.58,10.18,38.60]
#rects5 = ax.bar(ind+width*4+margin*4, dvals, width, color='white', hatch="oooo", align="center", edgecolor='black')

#svals = [4.88,14.34,30.17,6.92,9.75,5.56,14.99,14.37,15.74,10.85,7.83,3.00,5.68,5.69,20.27] #INV
#rects6 = bx.bar(ind+width*4+margin*4, svals, width, color='white', hatch="oooo", align="center", edgecolor='black')

#ax.set_ylabel('Bad Pixels',linespacing=3.2)
ax.set_xticks(ind+width+0.10)
ax.set_xticklabels( ('Adirondack', 'ArtL', 'Jadeplant', 'Motorcycle', 'MotorcycleE', 'Piano', 'PianoL', 'Pipes', 'Playroom', 'Playtable', 'PlaytableP', 'Recycle', 'Shelves', 'Teddy', 'Vintage' ), rotation=45 )
#ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax.set_alpha(0.5)
ax.legend( (rects1[0], rects2[0], rects3[0] ), ( '7x7', '15x15', 'Multi' ) )

#ax.plot(x, y, label = r"This is \textbf{line 1}")
#ax.plot(x, z, label = r"This is \textit{line 2}")
'''ylabel = ax.yaxis.get_label()
ylabel.set_rotation_mode('anchor')
ylabel.set_va('baseline')
ylabel.set_ha('center')
ylabel.set_rotation('vertical')'''
ax.set_axisbelow(True)
plt.grid('grid', linestyle='dashed', color='#d9d9d9')




def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2.+0.03, h+0.5, '%d'%int(h),
                ha='center', va='bottom', rotation=90)

def autolabel(svals, rects1):
    for sval, rect1 in zip(svals, rects1):
        h1 = rect1.get_height()
        #h = rect.get_height()
        #print(h1)
        ax.text(rect1.get_x()+rect1.get_width()/2-0.02, h1+0.5, 'INV: %.2f%%'%(sval), rotation=90, fontsize=9, verticalalignment='bottom')

#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)
#autolabel(rects4)
#autolabel(svals,rects5)

plt.show()
