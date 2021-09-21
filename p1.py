import math

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
        return max
    
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
            if (len(i) == dimensionDada):
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
simplice_prueba.añadir(["1", "4", "3"])

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
#print(simplice_prueba.carasDim(1))
#print(simplice_prueba.carasDim(2))
#print(simplice_prueba.carasDim(3))

print(simplice_prueba.estrella("1"))
print(simplice_prueba.estrella_cerrada("1"))
print(simplice_prueba.link("1"))
