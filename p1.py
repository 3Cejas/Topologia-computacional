
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
        numero = self.dimension
        return 2^(numero+1)-1
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
