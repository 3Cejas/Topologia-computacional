
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
simplice_prueba = CS()
simplice_prueba.añadir(["1"])
simplice_prueba.añadir(["2"])
simplice_prueba.añadir(["1", "2", "3", "4"])
print(simplice_prueba.dimension())