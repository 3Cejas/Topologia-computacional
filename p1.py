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
    
    def caras(self):
        num = 0
        for i in self.index:
            num = num + 1
        return num

    def carasDim(self,dimensionDada):
        #Dado un k-simplice y una l-dimension: k+1!/((l+1!)(k-l!))
        n=0
        for i in self.index:
            if (len(i) == dimensionDada):
                n = n+1
        return n
        #return math.factorial(self.caras() - 1)/(math.factorial(dimensionDada)*math.factorial(self.caras()-dimensionDada -1))
        
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
print(simplice_prueba.dimension())
print(simplice_prueba.caras())
print(simplice_prueba.carasDim(2))
