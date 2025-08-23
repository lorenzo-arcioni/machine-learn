# Introduzione all'Ottimizzazione non Lineare

## Perché serve?
Quando lavoriamo con problemi di **regressione**, abbiamo notato che i dati **raramente** seguono una distribuzione lineare.  
La maggior parte dei fenomeni reali è **non lineare**, e quindi i nostri modelli devono essere in grado di rappresentare questa complessità.

In generale, capita molto spesso di trovarsi davanti a una **funzione di perdita** il cui gradiente dipende dai parametri del modello in maniera **non lineare**.  
Un esempio classico è la **regressione logistica**.

👉 Abbiamo visto che:
- Solo per funzioni **convesse** possiamo trovare una **soluzione in forma chiusa** (ponendo il gradiente $= 0$).
- Le funzioni non lineari **non sono sempre** non convesse (ad esempio le funzioni quadratiche sono non lineari ma restano convesse).
- Tuttavia, **molte funzioni non lineari sono non convesse**: in questi casi **non esistono soluzioni in forma chiusa**, e dobbiamo ricorrere all’**ottimizzazione numerica non lineare**.


## 🌱 Metodi di Primo Ordine
La famiglia di algoritmi più semplice e diffusa sono i **metodi di primo ordine**, che utilizzano **solo il gradiente** (derivata prima) della funzione obiettivo.  

Il più noto è la **[[Discesa del Gradiente]] (Gradient Descent, GD)**:
- algoritmo iterativo che muove i parametri nella direzione opposta al gradiente,  
- permette di avvicinarsi progressivamente a un **minimo locale**,  
- garantisce il **minimo globale** solo se la funzione è convessa.  


## 🌳 Oltre il Primo Ordine
Oltre ai metodi di primo ordine, esistono algoritmi più sofisticati che sfruttano anche informazioni di ordine superiore:  

- **Metodi di secondo ordine** → utilizzano anche le derivate seconde (Hessiana), come il metodo di Newton o quasi-Newton.  
- **Tecniche ibride e varianti moderne** → come i metodi adattivi (Adam, RMSProp, Adagrad), o strategie di ottimizzazione vincolata.  


## 📂 In questa sezione
Questa nota funge da **introduzione generale**.  
Nelle note collegate approfondiamo le diverse tecniche di ottimizzazione:

- 👉 [Gradient Descent (GD)](Gradient_Descent.md)  
- 👉 [Metodi di Secondo Ordine](Second_Order_Methods.md)  
- 👉 [Ottimizzazioni Adattive](Adaptive_Optimizers.md)  

---

✍️ **In sintesi**:  
L’ottimizzazione non lineare è il cuore dell’apprendimento automatico.  
Che si tratti di un metodo semplice come il GD o di algoritmi più avanzati, tutti condividono lo stesso obiettivo: **trovare parametri che minimizzino la funzione di perdita**.  
