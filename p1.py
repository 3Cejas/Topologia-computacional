import math
from typing import no_type_check

class CS:
    def __init__(self):
        self.index = []
    
    def añadir(self, lista):
        return self.index.append(lista)

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
        return self.index

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
        for i in self.index:
            if (len(i) == dimensionDada + 1):
                caras.append(i)
        return caras

    def estrella(self, elem):
        caras = self.index
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

        n = 1
        for j in self.carasDim(1):
            #Falta hacer aquí el algoritmo de visitación de vértices
            n=1
        return n
    
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
print(simplice_prueba.dimension())
#print(simplice_prueba.caras())
print(simplice_prueba.carasDim(0))
print(simplice_prueba.carasDim(1))
print(simplice_prueba.carasDim(2))
print(simplice_prueba.carasDim(3))

print(simplice_prueba.estrella("1"))
print(simplice_prueba.estrella_cerrada("1"))
print(simplice_prueba.link("1"))
print(simplice_prueba.cEuler())
print(simplice_prueba.nConexo())
print(simplice_prueba2.nConexo())
print(simplice_prueba3.nConexo())

