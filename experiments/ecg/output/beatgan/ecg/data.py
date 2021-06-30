import matplotlib.pyplot as plt

f = open('val_info.txt', 'r')
line_num=1
line=f.readline()

x=[]

while line:
    tmp=line.find(':')
    inp=float(line[tmp+1:tmp+7])
    print(inp)
    x.append(inp)
    line = f.readline()

f.close()

plt.plot(x)
plt.show()
