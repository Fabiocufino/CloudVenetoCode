Ciao Andrea, ho finito di farte l'analisi sul dataset di radioattività e ti annuncio i risultati finali.

# Feature Creation:
Ho costruito la tabella delle feature ($\Delta t$, $\Delta r$, $E_p$, $E_d$,$R_{prompt}$, $R_{delayed}$, $Label$ (che è 1 se è un vero IBD, 0 se invece è un evento di BKG qualsiasi (IBD uncorrelated e Radioactivity). Ho poi aggiunto anche $Source$ che contiene proprio questa informazione: 1 se la coppia di eventi (quindi una riga nella tabella delle features) proviene da IBD oppure 0 se invece la coppia proviene dal dataset di radioattività.


# Manual Cut Algorithm:
Leggo la tabella delle features creata precedentemente e costruisco 3 dict differenti: 
	
- all_feature -> contiene tutta la tabella di features,
- features_BKG -> contiene tutti gli eventi di BKG (quindi creata con all_feature_df['Label'] == 0)
- features_IBD -> contiene tutti gli eventi di IBD veri (quindi creata con all_feature_df['Label'] == 0)
	
Costruisco la funzione "selection" con i cut presi dal paper che mi hai inviato e faccio 3 diverse "selezioni"per i 3 dict trovati:

### 1. selection(features_IBD):
Quindi è una selezione di tutti gli eventi di IBD veri.


Venongono selezionati come IBD  1435115.0 Su un tot di 1468385\
Efficiency:  97.73424544652799\
Purity:  100.0


### 2. selection(features_BKG[features_BKG["Source"] == 0]):
Quindi è una ricerca di falsi eventi IBD nel dataset di SOLA RADIOATTIVITÀ.


Vengono selezionati 26.0 fake IBD su un totale di 993457\
Efficiency:  99.9973828761587\
Purity:  100.0

### 3. selection(all_feature):
Efficiency:  97.73424544652799\
Purity:  99.99742187587098


# ML_algorithm:
Ho creato 3 dict allo stesso modo di  Manual Cut Algorithm, ovvero ho all_feature, features_BKG e features_IBD. 

## BDT:
Performa benissimo, no overfittings, no underfittings. LA curva Log loss sembra perfetta. 

### 1. BDT su Veri IBD:
Quindi è '(X_true_IBD_df = all_feature[(all_feature["Label"] == 1) & (all_feature["Source"] == 1)])'.


Venongono selezionati come IBD  1468350 Su un tot di 1468385\
Efficiency:  99.99761642893384\
Purity:  100.0


### 2. BDT su radioactivity BKG:
Quindi è una ricerca di falsi eventi IBD nel dataset di SOLA RADIOATTIVITÀ, ovvero 'X_rad_df = all_feature[(all_feature["Label"] == 0) & (all_feature["Source"] == 0)])'


Vengono selezionati 23 fake IBD su 993457\
Efficiency:  99.99768485198655\
Purity:  100.0


Direi un successo. Sto finendo di runnare l'ottimizzazione di XGBoost e una rete neurale con PyTorch, ma meglio di così non credo si passa fare. Sto finendo di ultimare anche la model interpretability.