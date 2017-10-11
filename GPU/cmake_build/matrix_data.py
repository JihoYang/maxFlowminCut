import matplotlib.pyplot as plt
import pylab
f_start = open("start_edge.txt" , "r")
start = f_start.read()
f_start.close()

f_end = open("end_edge.txt" , "r")
end = f_end.read()
f_end.close()

start = start.split("\n")
end = end.split("\n")

start_int = []
end_int = []

for v in start[1:-1]:
    start_int.append(int(v))

for v in end[1:-1]:
    end_int.append(int(v))

edges = []

for e in range(len(start_int)):
    edges.append(e)

file_name = start[0].split("/")[-1]

pylab.xscale("log")
pylab.yscale("log")

pylab.plot(edges, start_int, 'r.', label='Start Edge')
pylab.plot(edges, end_int, 'b.', label='End Edge')

pylab.title(file_name, fontsize=15)
pylab.xlabel('Edges(log)', fontsize=15)
pylab.ylabel('Vertices(log)', fontsize=15)
pylab.legend(loc='best', fontsize=15)
pylab.show()






