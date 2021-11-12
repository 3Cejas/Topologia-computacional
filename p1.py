import math
from typing import no_type_check
from numpy.lib.function_base import append
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
import numpy as np

class CS:
    def __init__(self):
        self.index = []
        self.orden = []

    #Devuelve la dimensión del complejo símplice
    def dimension(self):
        max = 0
        for i in self.index:
            if (len(i) > max):
                max = len(i)
        return max - 1
    
    #def caras(self):
        num = 0
        for i in self.index:
            num = num + 1
        return num
    
    def caras(self):
        res = []
        for lista in self.index:
            res.append(lista[0])
            for i in reversed(range(len(lista[0]))):
                for j in combinaciones(lista[0], i):
                    res.append(tuple(j))
        res = [x for x in res if x]
        result = []
        for elem in res:
            if elem not in result:
                result.append(elem)
        return sorted(result, key=len)

    #def carasDim(self,dimensionDada):
        #Dado un k-simplice y una l-dimension: k+1!/((l+1!)(k-l!))
        n=0
        for i in self.index:
            if (len(i) == dimensionDada):
                n = n+1
        return n
        #return math.factorial(self.caras() - 1)/(math.factorial(dimensionDada)*math.factorial(self.caras()-dimensionDada -1))
    
    def carasDim(self,dimensionDada):
        #Dado un k-simplice y una l-dimension: k+1!/((l+1!)(k-l!))
        caras = []
        for i in self.caras():
            if (len(i) == dimensionDada + 1):
                caras.append(i)
        return caras

    def estrella(self, elem):
        caras = self.caras
        cocaras = []
        for i in caras:
            for j in i:
                if(j == elem):
                    cocaras.append(i)

        return cocaras
    
    def estrella_cerrada(self, elem):
        cocaras = self.estrella(elem)
        caras = self.caras()
        sum = cocaras + caras
        nueva_sum = []
        for elem in sum:
            if elem not in nueva_sum:
                nueva_sum.append(elem)
        k = nueva_sum
        return sorted(k, key=len)

    def link(self, elem):
        estrella_cerrada =self.estrella_cerrada(elem)
        estrella = self.estrella(elem)
        res=[]
        for lst in estrella_cerrada:
            if lst not in estrella:
                res.append(lst)
        return res

    def cEuler(self):
        i = 0
        caracteristica = 0
        while i <= self.dimension():
            cont = 0
            for j in self.carasDim(i):
                cont = cont + 1
            caracteristica = caracteristica + cont*(-1)**i
            i = i + 1
        return caracteristica

    def nConexo(self):
        n = 0
        if self.dimension() == -1:
            return 0
        elif self.carasDim(1) == []:
            for k in self.caras():
                n=n+1
            return n
        n = 0
        aristas =self.carasDim(1)
        vertices = [item for t in (self.carasDim(0)) for item in t]
        actual = vertices[0]
        adyacencia = dict.fromkeys(vertices,[])
        print("inicio",adyacencia)
        indice = 0
        for vertice in vertices:
            almacenador = []
            for tuples in aristas:
                print("mi vértice actual es:", vertice, "la arista actual es:", tuples)

                if vertice == tuples[0]:
                    almacenador.append(tuples[1])
                if vertice == tuples[1]:
                    almacenador.append(tuples[0])
            adyacencia[vertice] = almacenador
        
        visited = set()
        c_conexas = []
        print("Following is the Depth-First Search")
        print(adyacencia)
        for vertice in vertices:
            dfs(visited, adyacencia, vertice)
            if not(list(visited) in c_conexas): 
                c_conexas.append(list(visited))
                n = n+1
        print("como")
        return n

    def añadir(self, lista):
        return self.index.append(lista)

    def insert(self, lista, numero):
        zipped = list(zip(lista,[numero]))
        self.añadir(zipped[0])
        self.orden.append(zipped[0])
        self.orden = sorted(self.orden, key=lambda tup: tup[1])

        self.orden = sorted(self.orden, key=lambda tup: len(tup[0]))
        return 

    def filtration(self, numero):
        return list(filter(lambda fil: menor_igual_que_peso(fil[1], numero),self.orden))


def menor_igual_que_peso(numero, peso):
    return numero <= peso

def dfs(visited, graph, node):  #function for dfs 
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)
def potencia(c):
    """Calcula y devuelve el conjunto potencia del 
       conjunto c.
    """
    if len(c) == 0:
        return [[]]
    r = potencia(c[:-1])
    return r + [s + [c[-1]] for s in r]

def combinaciones(c, n):
    """Calcula y devuelve una lista con todas las
       combinaciones posibles que se pueden hacer
       con los elementos contenidos en c tomando n
       elementos a la vez.
    """
    return (s for s in potencia(c) if len(s) == n)

def AlphaComplex(points):
    tri = (Delaunay(points)).simplices
    final = []
    for item in tri:
        x =[]
        for elem in item:
            x.append(str(elem))
        final.append(tuple(x))
    aristas=[]
    for triangulo in tri:
        for i in combinaciones(triangulo, 2):
            aristas.append(i)
    result = []
    for elem in aristas:
        if elem not in result:
            result.append(elem)
  
    finissima = []
    for aris in result:
        if aris[0] > aris[1]:
            temp = aris[0]
            aris[0] = aris[1]
            aris[1] = temp
        if aris not in finissima:
            finissima.append(tuple(aris))
    sin_rep = []
    for aris in finissima:
        if aris not in sin_rep:
            sin_rep.append(tuple(aris))

    ver = flat_list = [item for sublist in tri for item in sublist]
    verint = []
    for elem in ver:
        verint.append(int(elem))
    alpha = []
    for elem in verint:
        if elem not in alpha:
            alpha.append(elem)
    final = []
    puntos_sin_tupla = sorted(alpha)
    for elem in alpha:
        elem = [elem]
        final.append(tuple(elem))
    puntos = final
    final = final +sin_rep
    triana = []
    for elem in tri:
        triana.append(tuple(elem))
    alpha = final
    alpha =  alpha +triana
    print(alpha)
    dist_alpha(points, puntos,sin_rep)
    puntos = sorted(puntos)
    dicc_puntos_coord = dict(zip(puntos_sin_tupla, points))
    #Hallamos los baricentros
    baricentros = {}
    radios = {}
    for triangulo in triana:
        p1 = dicc_puntos_coord [triangulo[0]]
        p2 = dicc_puntos_coord [triangulo[1]]
        p3 = dicc_puntos_coord [triangulo[2]]

        x=(p1[0]+p2[0]+p3[0])/3
        y=(p1[1]+p2[1]+p3[1])/3
        baricentros[triangulo] = [x,y]
        radios [triangulo] = circleRadius(p1,p2,p3)
        radioli = radios.items()
    
    print

    return alpha

def circleRadius(b, c, d):
  temp = c[0]**2 + c[1]**2
  bc = (b[0]**2 + b[1]**2 - temp) / 2
  cd = (temp - d[0]**2 - d[1]**2) / 2
  det = (b[0] - c[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - c[1])

  if abs(det) < 1.0e-10:
    return None

  # Center of circle
  cx = (bc*(c[1] - d[1]) - cd*(b[1] - c[1])) / det
  cy = ((b[0] - c[0]) * cd - (c[0] - d[0]) * bc) / det

  radius = ((cx - b[0])**2 + (cy - b[1])**2)**.5

  return radius
def dist_alpha(points, puntos, aristas):
    puntos = sorted(puntos)
    print("puntos", puntos)
    print(points[0])
    distancias = []
    
    for arista in aristas:
        punto1 = [arista[0]]
        punto2 = [arista[1]]
        p1 = points[puntos.index(tuple(punto1))]
        p2 = points[puntos.index(tuple(punto2))]
        
        distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
        distancias.append(distance)
    print(aristas)
    print(distancias)
    dic_aris_dis= dict(zip(aristas, distancias))
    result = sublevel(dic_aris_dis,0.26)
    print(result)
    return result

def sublevel(dic_aris_dis, radio):
    filtro= {}
    for aris in dic_aris_dis:
        valor = dic_aris_dis[aris]
        if valor < radio:
            filtro[aris] = valor
    return filtro

''' 
simplice_prueba = CS()
simplice_prueba.añadir(["1"])
simplice_prueba.añadir(["2"])
simplice_prueba.añadir(["3"])
simplice_prueba.añadir(["4"])
simplice_prueba.añadir(["1", "2"])
simplice_prueba.añadir(["1", "3"])
simplice_prueba.añadir(["1", "4"])
simplice_prueba.añadir(["3", "4"])
simplice_prueba.añadir(["2", "3"])
simplice_prueba.añadir(["1", "2", "3"])
simplice_prueba.añadir(["1", "3", "4"])
simplice_prueba2 = CS()
simplice_prueba3 = CS()
simplice_prueba3.añadir(["1"])
simplice_prueba3.añadir(["2"])
simplice_prueba3.añadir(["3"])
simplice_prueba3.añadir(["4"])
'''
#
#simplice_prueba.añadir([(1)])
#simplice_prueba.añadir([(2)])
#simplice_prueba.añadir([(3)])
#simplice_prueba.añadir([(4)])
#simplice_prueba.añadir([(1, 2)])
#simplice_prueba.añadir([(1, 3)])
#simplice_prueba.añadir([(1, 4)])
#simplice_prueba.añadir([(3, 4)])
#simplice_prueba.añadir([(2, 3)])
#simplice_prueba.añadir([(1, 2, 3)])
#simplice_prueba.añadir([(1, 3, 4)])
#
#print(simplice_prueba.index)
#print(simplice_prueba.dimension())
#print(simplice_prueba.caras())
#print(simplice_prueba.carasDim(0))
#print(simplice_prueba.carasDim(1))
'''print(simplice_prueba.carasDim(2))
print(simplice_prueba.carasDim(3))

print(simplice_prueba.estrella("1"))
print(simplice_prueba.estrella_cerrada("1"))
print(simplice_prueba.link("1"))
print(simplice_prueba.cEuler())
print(simplice_prueba.nConexo())
print(simplice_prueba2.nConexo())
print(simplice_prueba3.nConexo())
'''

'''
simplice_prueba4 = CS()
simplice_prueba4.añadir([("0",)])
simplice_prueba4.añadir([("1",)])
simplice_prueba4.añadir([("2", "3")])
simplice_prueba4.añadir([("4", "5")])
simplice_prueba4.añadir([("5", "6")])
simplice_prueba4.añadir([("4", "6")])
simplice_prueba4.añadir([("6", "7", "8", "9")])
#print(simplice_prueba4.caras())
print(simplice_prueba4.carasDim(1))
print(simplice_prueba4.nConexo()) 
simplice_prueba5 = CS()
simplice_prueba5.añadir([("0", "1")])
simplice_prueba5.añadir([("1", "2", "3", "4")])
simplice_prueba5.añadir([("4", "5")])
simplice_prueba5.añadir([("5", "6")])
simplice_prueba5.añadir([("4", "6")])
simplice_prueba5.añadir([("6", "7", "8")])
simplice_prueba5.añadir([("8", "9")])

simplice_prueba6 = CS()
simplice_prueba6.añadir([("0", "1")])

#print(simplice_prueba5.caras())
'''

simplice_prueba1 = CS()
simplice_prueba1.insert([("0", "1")], 2.0)
simplice_prueba1.insert([("0",)], 2.0)
simplice_prueba1.insert([("1",)], 10.0)
simplice_prueba1.insert([("1", "2")], 10.0)

#print(simplice_prueba1.filtration(10.0))



points=np.array([(0.38021546727456423, 0.46419202339598786), (0.7951628297672293, 0.49263630135869474), (0.566623772375203, 0.038325621649018426), (0.3369306814864865, 0.7103735061134965), (0.08272837815822842, 0.2263273314352896), (0.5180166301873989, 0.6271769943824689), (0.33691411899985035, 0.8402045183219995), (0.33244488399729255, 0.4524636520475205), (0.11778991601260325, 0.6657734204021165), (0.9384303415747769, 0.2313873874340855)])     
plt.plot(points[:,0],points[:,1],'ko')
#plt.show()

vor=Voronoi(points)
fig = voronoi_plot_2d(vor,show_vertices=False,line_width=2, line_colors='blue')
plt.plot(points[:,0],points[:,1],'ko')
#plt.show()

tri = Delaunay(points)
fig = voronoi_plot_2d(vor,show_vertices=False,line_width=2, line_colors='blue' )
c=np.ones(len(points))
cmap = matplotlib.colors.ListedColormap("limegreen")
plt.tripcolor(points[:,0],points[:,1],tri.simplices, c, edgecolor="k", lw=2, cmap=cmap)
plt.plot(points[:,0], points[:,1], 'ko')
#plt.show()

alpha=AlphaComplex(points)

