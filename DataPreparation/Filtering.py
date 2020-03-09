
import pandas as pd

class Filtering():
    
    def __init__(self, dataFile, out):
            
        self.dataFile = dataFile
        self.out = out
    
    def applyLimiar(self):
        print("Filtering step \n")
        
        vagas = pd.read_csv(self.dataFile, encoding = 'cp1252')
#        print("Vagas")
#        print(vagas.shape)

        # Selecting IT job offers
        limiar = 0.5
        vagas_ti = vagas[vagas.TI > limiar]
        vagas_ti = vagas_ti[["descricao","titulo"]]
        vagas_ti.shape

        vagas = vagas.to_csv(self.out + "vagas_ti.csv")
        
        print("\n Filtering step done!")