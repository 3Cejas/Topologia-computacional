import copy
import math
from os import remove
from queue import Empty
from typing import no_type_check
from numpy.lib.function_base import append
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Polygon
from itertools import combinations
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
    
    def caras(self):
        maximales = self.index
        res = []
        for simplice in maximales:
            res.append(tuple(simplice))
            for i in range(len(simplice)):
                combinaciones = list(combinations(simplice, i))
                res = res + combinaciones
        resultado =[]
        for elem in res:
            if len(elem) != 0 and elem not in resultado:
                resultado.append(elem)
        for l in resultado:
                for elem in resultado:
                    if(set(elem).issubset(set(l)) and len(elem) == len(l) and l != elem):
                        resultado.remove(elem)
        return sorted(resultado, key=len)

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
        caras = self.caras()
        cocaras = []
        for i in caras:
            if(all(elem1 in i  for elem1 in elem)):
                cocaras.append(i)

        return cocaras
    
    def estrella_cerrada(self, elem):
        estrella_c = self.estrella(elem)
        for i in estrella_c:
            for j in range(len(i)):
                combinaciones = list(combinations(i, j))
                for h in combinaciones:
                    if(h not in estrella_c and len(h) != 0):
                        estrella_c.append(h)
        
        return sorted(estrella_c, key=len)

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
        #print("inicio",adyacencia)
        indice = 0
        for vertice in vertices:
            almacenador = []
            for tuples in aristas:
                #print("mi vértice actual es:", vertice, "la arista actual es:", tuples)

                if vertice == tuples[0]:
                    almacenador.append(tuples[1])
                if vertice == tuples[1]:
                    almacenador.append(tuples[0])
            adyacencia[vertice] = almacenador
        
        visited = set()
        c_conexas = []
        #print("Following is the Depth-First Search")
        #print(adyacencia)
        for vertice in vertices:
            dfs(visited, adyacencia, vertice)
            if not(list(visited) in c_conexas): 
                c_conexas.append(list(visited))
                n = n+1
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
    def boundary_matrix(self,p):
        x = self.carasDim(p)
        y = self.carasDim(p-1)
        matriz = np.zeros((len(y), len(x)))
        i = 0 #fila
        j = 0 #columna
        for cara in x:
            combinaciones = list(combinations(cara, p))
            resultado =[]
            for elem in combinaciones:
                if len(elem) != 0 and elem not in resultado:
                    resultado.append(elem)
            for fila in y:
                if fila in resultado:
                    matriz[i][j] = 1
                i +=1
            j +=1
            i = 0
        return matriz

    def forma_smith(self, p):
        M_delta = self.boundary_matrix(p)
        x = self.carasDim(p)
        y = self.carasDim(p-1)
        i = 0 #fila
        j = 0 #columna
        for i in range(min(len(y), len(x))):
            j=i            
            for m in range(len(y)):
                for n in range (len(y)):
                    if ((M_delta[m][:] == M_delta[n][:]).all() and n > m and m >(i-1)):
                        M_delta[n][:] = (M_delta[n][:] + M_delta[n][:])%2
            if (i < min(len(y), len(x)) and M_delta[i][j] != 1):
                for n in range(len(x)):
                    if(n >= j):
                        for m in range(len(y)):
                            if(M_delta[m][n] == 1 and M_delta[i][j] != 1 and m > i):
                                M_delta[[m, i], : ] = M_delta[[i, m],: ]
                            if(M_delta[i][j] != 1):
                                M_delta[:,[n, j ]] = M_delta[:,[j, n] ]

            if (i < min(len(y), len(x)) and M_delta[i][j] == 1):
                for m in range(len(y)):
                    if (M_delta[m][j] == 1 and m > i):
                        M_delta[m][:] = (M_delta[m][:] + M_delta[i][:])%2     
                for n in range(len(x)):
                    if (M_delta[i][n] == 1 and n > j):
                        M_delta[i][n] = (M_delta[i][n] + M_delta[i][j])%2
        return M_delta
        
    def n_betti(self, p):
        N = self.forma_smith(p)
        x = self.carasDim(p)
        y = self.carasDim(p-1)
        contador = 0
        for i in range(min(len(y), len(x))):
            if(N[i][i] == 1):
                contador +=1
        dim_Z = len(x) -contador 
        N = self.forma_smith(p+1)
        x = self.carasDim(p+1)
        y = self.carasDim(p)
        dim_B = 0
        for i in range(min(len(y), len(x))):
            if(N[i][i] == 1):
                dim_B+=1
        return (dim_Z - dim_B)

    def incremental(self):
        b_0 = b_1 = 0
        añadidos = []
        simplice = self.caras()
        aristas = 0
        vertices_activos = []
        caras_actuales = 0
        it = 0
        list_it = []
        for s in simplice:
            if(len(s)== 1): 
                b_0 +=1
            if(len(s) == 2):
                añadidos.append(s)
                if s[0] not in vertices_activos:
                    vertices_activos.append(s[0])
                if s[1] not in vertices_activos:
                    vertices_activos.append(s[1])
                aristas +=1
                if((aristas-len(vertices_activos)+1)>caras_actuales):
                    b_1 +=((aristas-len(vertices_activos)+1)-caras_actuales)
                    caras_actuales = (aristas-len(vertices_activos)+1)
                else:
                    b_0 -=1
            if(len(s) >= 3):
                b_1 -=1
            list_it.append([b_0, b_1])
            it +=1
        print("b_0: ", b_0, "\nb_1: ", b_1)
        return [b_0, b_1, list_it]

    def general_boundary_matrix(self):
        simplices = self.caras()
        matriz = np.zeros((len(simplices), len(simplices)))
        for s in simplices:
            for l in simplices:
                if(set(l).issubset(set(s)) and len(l) == len(s)-1):
                    matriz[simplices.index(l)][simplices.index(s)] = 1
        return matriz
    # def n_ch(self):
    #     ps = self.dimension()
    #     for p in range(ps):
    #         for list_it in self.n_betti(p)[2]:
    #             for i in range(len(list_it)):
    #                 for j in range(len(list_it)):
    #                     if(j > i):
    #                         viven = (list_it[j-1][p] - list_it[i][p] - x ) - ( x- x)
    #                         nacen =

    def alg_matricial(self):
        matriz = self.general_boundary_matrix()
        caras = self.caras()
        dgm_0 = []
        dgm_1 = []
        flaglow = 0
        for j in range(len(matriz)):
            if(low(matriz, j) != -1):
                for j_0 in reversed(range(j)):
                    if low(matriz, j_0) == low(matriz, j) and j != j_0:
                        matriz[:,j] = (matriz[:,j] + matriz[:,j_0])%2
                if(len(caras[j]) == 2 and low(matriz, j) != -1):
                    dgm_0.append(tuple((0,caras[j])))
                if(len(caras[j])> 2 and low(matriz, j) != -1):
                    dgm_1.append(tuple((caras[low(matriz,j)],caras[j])))
        print("hola\n",matriz)
        for j in range(len(matriz)):
            if(low(matriz, j) == -1 and len(caras[j]) == 1):
                for i in range(len(matriz)):
                    if(low(matriz,i) == j):
                        if( tuple((0, math.inf)) not in dgm_0):
                            dgm_0.append(tuple((0, math.inf)))
            if(low(matriz, j) == -1 and len(caras[j]) == 2):
                flaglow = 0
                for i in range(len(matriz)):
                    if(low(matriz,i) == j):
                          flaglow = 1 
                if(flaglow == 0):
                    if(tuple((caras[low(matriz,j)], math.inf)) not in dgm_1):
                            print("caraaas")
                            print(caras)
                            print(caras[j])
                            print(j)
                            print(low(matriz,i))
                            dgm_1.append(tuple((caras[low(matriz,j)], math.inf)))

        print(dgm_0)
        print(dgm_1)
        return [matriz, dgm_0, dgm_1]

def low(matriz, j):
    red_flag = 0
    for fila in reversed(range(len(matriz))):
        
        if(matriz[fila][j] == 1 and red_flag == 0):
            red_flag = 1
            return fila
    return -1
def combinaciones(c, n):
    # Calcula y devuelve una lista con todas las
    # combinaciones posibles que se pueden hacer
    # con los elementos contenidos en c tomando n
    # elementos a la vez.

    return (s for s in potencia(c) if len(s) == n)

def menor_igual_que_peso(numero, peso):
    return numero <= peso

def dfs(visited, graph, node):  #function for dfs 
    if node not in visited:
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)


def potencia(c):
    # Calcula y devuelve el conjunto potencia del 
    # conjunto c.
    
    if len(c) == 0:
        return [[]]
    r = potencia(c[:-1])
    return r + [s + [c[-1]] for s in r]

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
    final_r = []
    puntos_sin_tupla = sorted(alpha)
    for elem in alpha:
        elem = [elem]
        elem_r = [tuple(elem), 0]
        final.append(tuple(elem))
        final_r.append(tuple(elem_r))
    puntos_r = final_r
    puntos = final
    final = final +sin_rep
    triana = []
    for elem in tri:
        triana.append(tuple(elem))
    alpha = final
    alpha =  alpha +triana
    aris_peso=list(dis_aris(points, puntos,sin_rep).items())
    puntos = sorted(puntos)
    tris_peso = dis_tris(puntos_sin_tupla, points, triana)
    union = []
    union.extend(puntos_r)
    union.extend(aris_peso)
    union.extend(tris_peso)
    orden = sorted(union, key=lambda tup: tup[1])
    return orden

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

def dis_tris(puntos_sin_tupla, points, triana):
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
    tris_peso = list(radios.items())

    return tris_peso
def dis_aris(points, puntos, aristas):
    puntos = sorted(puntos)
    distancias = []
    for arista in aristas:
        punto1 = [arista[0]]
        punto2 = [arista[1]]
        p1 = points[puntos.index(tuple(punto1))]
        p2 = points[puntos.index(tuple(punto2))]
        distance = math.dist(p1,p2)
        distancias.append(distance*0.5)
      
    dic_aris_dis= dict(zip(aristas, distancias))

    return dic_aris_dis

def sublevel(alpha, radio):
    alpha_filtrado = []
    for elem in alpha:
        if elem[1] <= radio:
            alpha_filtrado.append(elem[0])
    return alpha_filtrado

def DelaunayVoronoi(points):
    alpha=AlphaComplex(points)
    alpha_pesos = []
    for elem in alpha:
        alpha_pesos.append(elem[1])
    fig = voronoi_plot_2d(vor,show_vertices=False,line_width=2, line_colors='blue')
    plt.plot(points[:,0],points[:,1],'ko')
    cmap = matplotlib.colors.ListedColormap("limegreen")
    c=np.ones(len(points))
    plt.tripcolor(points[:,0],points[:,1],tri.simplices, c, edgecolor="k", lw=2,cmap=cmap)
    plt.show()

def plotSublevel(points, alpha, radio):
    sublevel_sin_puntos = []
    for elem in sublevel(alpha, 0.26):
        if len(elem) > 1:
            sublevel_sin_puntos.append(elem)

    voronoi_plot = voronoi_plot_2d(vor,show_vertices=False,line_width=2, line_colors='blue')
    plt.plot(points[:,0],points[:,1],'ko')
    for elem in sublevel_sin_puntos:
        if (len(elem) == 3):
            p = Polygon([[points[elem[0]][0], points[elem[0]][1]], [points[elem[1]][0], points[elem[1]][1]], [points[elem[2]][0], points[elem[2]][1]] ], color= "limegreen", closed=False)
            ax = plt.gca()
            ax.add_patch(p)
        else:
            plt.plot([points[elem[0]][0], points[elem[1]][0]], [points[elem[0]][1], points[elem[1]][1]], 'black')
        plt.pause(0.5)
    plt.show()

def puntos_persistencia(points):
    alpha = sorted(AlphaComplex(points), key=lambda tup: len(tup[0]))
    simplice_alpha = CS()
    for elem in alpha:
        for num in elem[0]:
            num = str(num)
        simplice_alpha.añadir(elem[0])
    #print(simplice_alpha.index)
    res_inc = simplice_alpha.alg_matricial()
    print(res_inc[0])
    dgm_0 = res_inc[1]
    dgm_1 = res_inc[2]
    p_dgm0=[]
    p_infinito = []
    for elem in dgm_0:
        if(elem[1] == math.inf):
            p_infinito.append(elem[0])
        p2=-1
        encontrado = 0
        for elem1 in alpha:
            if (elem[1]==elem1[0] and encontrado == 0):
                p2=elem1[1]
                p_dgm0.append([0,p2])
                encontrado = 1
            
    p_dgm1=[]
    for elem in dgm_1:
        p1=p2=-1
        encontrado1 = 0
        encontrado2 = 0
        for elem1 in alpha:
            if (elem[0]==elem1[0]):
                p1=elem1[1]
                encontrado1 = 1
            if (elem[1]==elem1[0]):
                p2=elem1[1]
                encontrado2 = 1
            if (p1 > 0 and p2 > 0 and encontrado1==1 and encontrado2 ==1):
                p_dgm1.append([p1,p2])
                encontrado1 = 0
                encontrado2 = 0
            if(elem[1] == math.inf):
                if(elem[0] == elem1[0]):
                    p_infinito.append(elem1[1])
       
    return[p_dgm0, p_dgm1, p_infinito]

def diagrama_persistencia(lista):
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    print(lista[2])
    max = 0
    for x in lista:
        for y in x:
            if(x != lista[2]):
                if(y[1] >= max):
                    max = y[1]
    max = max + max/3
    for x in lista[0]:
        plt.scatter(x[0], x[1],c = 'b')
    for x in lista[1]:
        plt.scatter(x[0], x[1], c ='r')
    for x in lista[2]:
        if(x == 0):
            plt.scatter(x, max, c ='b')
        if(x != 0):
            plt.scatter(x, max, c ='r')

    plt.plot([0, max], [0, max], linestyle='--', c="gray")
    plt.plot([0, max], [max, max], linestyle='--', c = "gray")
    plt.show()

simplice_prueba = CS()

def test_tetraedro():
    #TETRAEDRO
    print("TEST TETRAEDRO")
    simplice_tetraedro = CS()
    simplice_tetraedro.añadir(["0", "1", "2", "3"])

    # print(simplice_tetraedro.n_betti(0))
    # print(simplice_tetraedro.n_betti(1))
    # print(simplice_tetraedro.n_betti(2))
    # print(simplice_tetraedro.n_betti(3))
    # print(simplice_tetraedro.general_boundary_matrix())
    #print(alg_matricial(simplice_tetraedro.general_boundary_matrix()))

#test_tetraedro()
simplice_alma = CS()
simplice_alma.añadir(["0", "1", "3"])
simplice_alma.añadir(["1", "3", "4"])
simplice_alma.añadir(["1", "2", "4"])
simplice_alma.añadir(["3", "4", "5"])
# print(simplice_alma.general_boundary_matrix())
#print(simplice_alma.n_betti(1))
#simplice_alma.alg_matricial()

#BORDE DEL TETRAEDRO
# simplice_tetraedro_borde = CS()
# for caras in simplice_tetraedro.carasDim(2):
#      simplice_tetraedro_borde.añadir(caras)
# simplice_tetraedro_borde.añadir(["0", "1"])
# simplice_tetraedro_borde.añadir(["0", "2"])
# simplice_tetraedro_borde.añadir(["0", "3"])
# simplice_tetraedro_borde.añadir(["1", "2"])
# simplice_tetraedro_borde.añadir(["1", "3"])
# simplice_tetraedro_borde.añadir(["2", "3"])

# print(simplice_tetraedro_borde.n_betti(0))
# print(simplice_tetraedro_borde.n_betti(1))
# print(simplice_tetraedro_borde.n_betti(2))


# El toro con las dos triangulaciones.
simplice_toro1 = CS()
simplice_toro1.añadir(["1", "2", "4"])
simplice_toro1.añadir(["2", "4", "5"])
simplice_toro1.añadir(["2", "3", "5"])
simplice_toro1.añadir(["3", "5", "6"])
simplice_toro1.añadir(["1", "3", "6"])
simplice_toro1.añadir(["1", "4", "6"])
simplice_toro1.añadir(["4", "5", "7"])
simplice_toro1.añadir(["5", "7", "8"])
simplice_toro1.añadir(["5", "6", "8"])
simplice_toro1.añadir(["6", "8", "9"])
simplice_toro1.añadir(["4", "6", "9"])
simplice_toro1.añadir(["4", "7", "9"])
simplice_toro1.añadir(["1", "7", "8"])
simplice_toro1.añadir(["1", "2", "8"])
simplice_toro1.añadir(["2", "8", "9"])
simplice_toro1.añadir(["2", "3", "9"])
simplice_toro1.añadir(["3", "7", "9"])
simplice_toro1.añadir(["1", "3", "7"])

# simplice_toro1.incremental()
# print(simplice_toro1.n_betti(0))
# print(simplice_toro1.n_betti(1))



simplice_toro2 = CS()
# for caras in simplice_toro1.carasDim(1):
#     simplice_toro2.añadir(caras)

simplice_toro2.añadir(["0", "1", "5"])
simplice_toro2.añadir(["0", "3", "5"])
simplice_toro2.añadir(["1", "2", "5"])
simplice_toro2.añadir(["2", "5", "6"])
simplice_toro2.añadir(["0", "2", "6"])
simplice_toro2.añadir(["0", "3", "6"])
simplice_toro2.añadir(["3", "4", "5"])
simplice_toro2.añadir(["4", "5", "7"])
simplice_toro2.añadir(["5", "6", "7"])
simplice_toro2.añadir(["6", "7", "8"])
simplice_toro2.añadir(["3", "6", "8"])
simplice_toro2.añadir(["3", "4", "8"])
simplice_toro2.añadir(["0", "4", "7"])
simplice_toro2.añadir(["0", "1", "7"])
simplice_toro2.añadir(["1", "7", "8"])
simplice_toro2.añadir(["1", "2", "8"])
simplice_toro2.añadir(["0", "4", "8"])
simplice_toro2.añadir(["0", "2", "8"])

# print(simplice_toro2.n_betti(0))
# print(simplice_toro2.n_betti(1))
# print(simplice_toro2.n_betti(2))


# El plano proyectivo.
simplice_planoproy = CS()
simplice_planoproy.añadir(["1", "2", "6"])
simplice_planoproy.añadir(["2", "3", "4"])
simplice_planoproy.añadir(["1", "3", "4"])
simplice_planoproy.añadir(["1", "2", "5"])
simplice_planoproy.añadir(["2", "3", "5"])
simplice_planoproy.añadir(["1", "3", "6"])
simplice_planoproy.añadir(["2", "4", "6"])
simplice_planoproy.añadir(["1", "4", "5"])
simplice_planoproy.añadir(["3", "5", "6"])
simplice_planoproy.añadir(["4", "5", "6"])

# print(simplice_planoproy.n_betti(0))
# print(simplice_planoproy.n_betti(1))
# print(simplice_planoproy.n_betti(2))

# La botella de Klein.
simplice_klein = CS()
simplice_klein.añadir(["0", "1", "5"])
simplice_klein.añadir(["0", "3", "5"])
simplice_klein.añadir(["1", "2", "5"])
simplice_klein.añadir(["2", "5", "6"])
simplice_klein.añadir(["0", "2", "6"])
simplice_klein.añadir(["0", "4", "6"])
simplice_klein.añadir(["3", "4", "5"])
simplice_klein.añadir(["4", "5", "7"])
simplice_klein.añadir(["5", "6", "7"])
simplice_klein.añadir(["6", "7", "8"])
simplice_klein.añadir(["4", "6", "8"])
simplice_klein.añadir(["3", "4", "8"])
simplice_klein.añadir(["0", "4", "7"])
simplice_klein.añadir(["0", "1", "7"])
simplice_klein.añadir(["1", "7", "8"])
simplice_klein.añadir(["1", "2", "8"])
simplice_klein.añadir(["0", "2", "8"])
simplice_klein.añadir(["0", "3", "8"])

# print(simplice_klein.n_betti(0))
# print(simplice_klein.n_betti(1))
# print(simplice_klein.n_betti(2))


# El anillo.

simplice_anillo = CS()

simplice_anillo.añadir(["1", "2", "4"])
simplice_anillo.añadir(["1", "3", "6"])
simplice_anillo.añadir(["1", "4", "6"])
simplice_anillo.añadir(["2", "3", "5"])
simplice_anillo.añadir(["2", "4", "5"])
simplice_anillo.añadir(["3", "5", "6"])

# print(simplice_anillo.n_betti(0))
# print(simplice_anillo.n_betti(1))
# print(simplice_anillo.n_betti(2))

# El sombrero del asno.

simplice_asno = CS()



# Del complejo simplicial de la transparencia 4 del documento Homología Simplicial II.

simplice_hs2t4 = CS()
simplice_hs2t4.añadir(["0", "1"])
simplice_hs2t4.añadir(["1", "2", "3", "4"])
simplice_hs2t4.añadir(["4", "5"])
simplice_hs2t4.añadir(["5", "6"])
simplice_hs2t4.añadir(["4", "6"])
simplice_hs2t4.añadir(["6", "7", "8"])
simplice_hs2t4.añadir(["8", "9"])

# print(simplice_hs2t4.n_betti(0))
# print(simplice_hs2t4.n_betti(1))
# print(simplice_hs2t4.n_betti(2))
# print(simplice_hs2t4.n_betti(3))

# Del doble toro.


# De algunos alfa complejos.

simplice_alg = CS()
simplice_alg.añadir(["0", "1", "2"])
simplice_alg.añadir(["0", "2", "3"]) 
simplice_alg.añadir(["0", "8", "9"])
simplice_alg.añadir(["7", "8", "9"])
simplice_alg.añadir(["6", "7", "8"])
simplice_alg.añadir(["5", "6", "8"])
simplice_alg.añadir(["3", "4"])
simplice_alg.añadir(["3", "5"])
simplice_alg.añadir(["4", "5"])
#print(simplice_alg.caras())
# simplice_alg.incremental()
# print(simplice_alg.n_betti(0))
# print(simplice_alg.n_betti(1))


# simplice_prueba.añadir(["0", "1"])
# simplice_prueba.añadir(["1", "2", "3", "4"])
# simplice_prueba.añadir(["4", "5"])
# simplice_prueba.añadir(["5", "6"])
# simplice_prueba.añadir(["4", "6"])
# simplice_prueba.añadir(["6", "7", "8"])
# simplice_prueba.añadir(["8", "9"])


# print(simplice_prueba.index)
# print(simplice_prueba.caras())
# print(simplice_prueba.carasDim(1))
# print(simplice_prueba.estrella(("2",)))
# print(simplice_prueba.nConexo())

# print(simplice_prueba.forma_smith(2))

# print(simplice_prueba.n_betti(0))
# print(simplice_prueba.n_betti(1))
# print(simplice_prueba.n_betti(2))
# print(simplice_prueba.n_betti(3))


# simplice_prueba2 = CS()
# simplice_prueba3 = CS()
# simplice_prueba3.añadir(["1"])
# simplice_prueba3.añadir(["2"])
# simplice_prueba3.añadir(["3"])
# simplice_prueba3.añadir(["4"])



# simplice_prueba.añadir([(1, 2, 3)])
# simplice_prueba.añadir([(1, 3, 4)])

# print(simplice_prueba.index)
# print(simplice_prueba.dimension())
# print(simplice_prueba.caras())
# print(simplice_prueba.carasDim(0))
# print(simplice_prueba.carasDim(1))

# print(simplice_prueba.carasDim(2))
# print(simplice_prueba.carasDim(3))
# print(simplice_prueba.estrella("1"))
# print(simplice_prueba.estrella_cerrada("1"))
# print(simplice_prueba.link("1"))
# print(simplice_prueba.cEuler())
# print(simplice_prueba.nConexo())

# print(simplice_prueba2.nConexo())
# print(simplice_prueba3.nConexo())



# simplice_prueba4 = CS()
# simplice_prueba4.añadir([("0",)])
# simplice_prueba4.añadir([("1",)])
# simplice_prueba4.añadir([("2", "3")])
# simplice_prueba4.añadir([("4", "5")])
# simplice_prueba4.añadir([("5", "6")])
# simplice_prueba4.añadir([("4", "6")])
# simplice_prueba4.añadir([("6", "7", "8", "9")])
# print(simplice_prueba4.caras())
# print(simplice_prueba4.carasDim(1))
# print(simplice_prueba4.nConexo()) 
# simplice_prueba6 = CS()
# simplice_prueba6.añadir([("0", "1")])


# simplice_pruebapeso = CS()

# simplice_pruebapeso.insert([("0", "1")], 2.0)
# simplice_pruebapeso.insert([("0",)], 2.0)
# simplice_pruebapeso.insert([("1",)], 10.0)
# simplice_pruebapeso.insert([("1", "2")], 10.0)

# print(simplice_pruebapeso.filtration(9.0))

simplice_alg1 = CS()
simplice_alg1.añadir(["0"])
simplice_alg1.añadir(["1"]) 
simplice_alg1.añadir(["2", "3"])
simplice_alg1.añadir(["4", "5",])
simplice_alg1.añadir(["5", "6"])
simplice_alg1.añadir(["4", "6"])
simplice_alg1.añadir(["3", "4"])
simplice_alg1.añadir(["6", "7", "8"])
#print(simplice_alg.caras())
# simplice_alg1.incremental()
# print(simplice_alg1.n_betti(0))
# print(simplice_alg1.n_betti(1))

points=np.array([(0.38021546727456423, 0.46419202339598786), (0.7951628297672293, 0.49263630135869474), (0.566623772375203, 0.038325621649018426), (0.3369306814864865, 0.7103735061134965), (0.08272837815822842, 0.2263273314352896), (0.5180166301873989, 0.6271769943824689), (0.33691411899985035, 0.8402045183219995), (0.33244488399729255, 0.4524636520475205), (0.11778991601260325, 0.6657734204021165), (0.9384303415747769, 0.2313873874340855)])     


vor=Voronoi(points)

tri = Delaunay(points)
#DelaunayVoronoi(points)
#AlphaComplex(points)


diagrama_persistencia(puntos_persistencia(points))
#plotSublevel(points, AlphaComplex(points), 0.26 )
