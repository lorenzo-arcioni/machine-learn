# Correlazione

La **Correlazione** è una misura statistica che quantifica la forza e la direzione della relazione lineare tra due variabili. È uno strumento fondamentale nell'analisi dei dati per comprendere come le variabili si comportano congiuntamente e per identificare potenziali relazioni predittive.

## **1. Definizione e Concetti Fondamentali**

La correlazione misura il grado in cui due variabili tendono a variare insieme. È importante distinguere tra:

- **Correlazione**: Misura l'associazione tra variabili, ma non implica causalità.
- **Causalità**: Indica che una variabile influenza direttamente l'altra.

> **Nota importante**: "Correlazione non implica causalità" - due variabili possono essere correlate senza che una causi l'altra, ad esempio a causa di una terza variabile confondente.

### **1.1. Covarianza**

Prima di definire la correlazione, introduciamo la **covarianza**, che misura come due variabili variano congiuntamente rispetto alle loro medie.

Per due variabili aleatorie $X$ e $Y$, la covarianza è definita come:

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$

Per un campione di $n$ osservazioni $\{(x_i, y_i)\}_{i=1}^n$, la covarianza campionaria è:

$$
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})
$$

Dove:
- $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ è la media campionaria di $X$
- $\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$ è la media campionaria di $Y$

**Interpretazione:**
- Se $\text{Cov}(X, Y) > 0$: le variabili tendono a crescere insieme (relazione positiva)
- Se $\text{Cov}(X, Y) < 0$: quando una cresce, l'altra tende a decrescere (relazione negativa)
- Se $\text{Cov}(X, Y) = 0$: le variabili non hanno una relazione lineare

**Limite della covarianza:** Il valore della covarianza dipende dalle unità di misura delle variabili, rendendo difficile interpretarne la forza della relazione.

## **2. Coefficiente di Correlazione di Pearson**

Il **Coefficiente di Correlazione di Pearson** (o correlazione lineare) risolve il problema della dipendenza dalle unità di misura normalizzando la covarianza.

### **2.1. Definizione**

Per due variabili aleatorie $X$ e $Y$, il coefficiente di correlazione di Pearson è:

$$
\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

Dove:
- $\sigma_X = \sqrt{\mathbb{V}[X]}$ è la deviazione standard di $X$
- $\sigma_Y = \sqrt{\mathbb{V}[Y]}$ è la deviazione standard di $Y$

Per un campione, il coefficiente di correlazione campionario è:

$$
r_{X,Y} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}
$$

### **2.2. Proprietà**

1. **Valori limitati**: $-1 \leq r_{X,Y} \leq 1$
2. **Simmetria**: $r_{X,Y} = r_{Y,X}$
3. **Invarianza per trasformazioni lineari**: Se $Y' = aY + b$ con $a > 0$, allora $r_{X,Y'} = r_{X,Y}$
4. **Indipendenza**: Se $X$ e $Y$ sono indipendenti, allora $r_{X,Y} = 0$ (ma non viceversa)

### **2.3. Interpretazione**

- $r = 1$: **correlazione positiva perfetta** (relazione lineare crescente)
- $r = -1$: **correlazione negativa perfetta** (relazione lineare decrescente)
- $r = 0$: **assenza di correlazione lineare** (non implica assenza di relazione non lineare)
- $|r| \in [0.7, 1]$: correlazione forte
- $|r| \in [0.4, 0.7]$: correlazione moderata
- $|r| \in [0, 0.4]$: correlazione debole

### **2.4. Dimostrazione: $-1 \leq r \leq 1$**

**Teorema:** Per ogni coppia di variabili $X$ e $Y$, vale $-1 \leq r_{X,Y} \leq 1$.

**Dimostrazione:**

Consideriamo le variabili standardizzate:

$$
Z_X = \frac{X - \mathbb{E}[X]}{\sigma_X}, \quad Z_Y = \frac{Y - \mathbb{E}[Y]}{\sigma_Y}
$$

Allora $\mathbb{E}[Z_X] = \mathbb{E}[Z_Y] = 0$ e $\mathbb{V}[Z_X] = \mathbb{V}[Z_Y] = 1$.

Per ogni $t \in \mathbb{R}$, consideriamo la variabile aleatoria $W = Z_X + tZ_Y$. Poiché la varianza è sempre non negativa:

$$
\mathbb{V}[W] = \mathbb{V}[Z_X + tZ_Y] \geq 0
$$

Sviluppando:

$$
\mathbb{V}[Z_X + tZ_Y] = \mathbb{V}[Z_X] + t^2\mathbb{V}[Z_Y] + 2t\text{Cov}(Z_X, Z_Y)
$$

$$
= 1 + t^2 + 2t\rho_{X,Y} \geq 0
$$

Questa è una disuguaglianza valida per ogni $t \in \mathbb{R}$. Il discriminante della parabola deve essere non positivo:

$$
\Delta = 4\rho_{X,Y}^2 - 4 \leq 0
$$

$$
\rho_{X,Y}^2 \leq 1
$$

Quindi $-1 \leq \rho_{X,Y} \leq 1$. $\square$

## **3. Forme Alternative e Matriciali**

### **3.1. Forma Matriciale**

Data una matrice di dati $\mathbf{X} \in \mathbb{R}^{n \times p}$ con $n$ osservazioni e $p$ variabili, la **matrice di correlazione** $\mathbf{R} \in \mathbb{R}^{p \times p}$ ha elementi:

$$
R_{ij} = r_{X_i, X_j}
$$

Dove $X_i$ e $X_j$ sono le colonne $i$-esima e $j$-esima di $\mathbf{X}$.

La matrice di correlazione può essere calcolata come:

$$
\mathbf{R} = \mathbf{D}^{-1/2} \mathbf{S} \mathbf{D}^{-1/2}
$$

Dove:
- $\mathbf{S}$ è la matrice di covarianza
- $\mathbf{D} = \text{diag}(S_{11}, S_{22}, \ldots, S_{pp})$ è la matrice diagonale delle varianze

### **3.2. Formula Computazionale**

Una forma alternativa utile per il calcolo è:

$$
r_{X,Y} = \frac{n\sum x_i y_i - \sum x_i \sum y_i}{\sqrt{n\sum x_i^2 - (\sum x_i)^2} \sqrt{n\sum y_i^2 - (\sum y_i)^2}}
$$

Questa formula evita il calcolo esplicito delle medie ed è più efficiente computazionalmente.

## **4. Correlazione di Rango**

Quando i dati non seguono una distribuzione normale o contengono outlier, è preferibile utilizzare misure di correlazione non parametriche basate sui ranghi.

### **4.1. Correlazione di Spearman**

Il **coefficiente di correlazione di Spearman** $\rho_s$ è il coefficiente di Pearson applicato ai ranghi delle osservazioni:

$$
\rho_s = r_{\text{rank}(X), \text{rank}(Y)}
$$

Se non ci sono ranghi ripetuti, esiste una formula semplificata:

$$
\rho_s = 1 - \frac{6\sum_{i=1}^{n} d_i^2}{n(n^2-1)}
$$

Dove $d_i$ è la differenza tra i ranghi di $x_i$ e $y_i$.

**Caratteristiche:**
- Misura la correlazione monotona (non necessariamente lineare)
- Robusto agli outlier
- Appropriato per dati ordinali

### **4.2. Correlazione di Kendall**

Il **coefficiente tau di Kendall** ($\tau$) misura la concordanza tra le coppie di osservazioni:

$$
\tau = \frac{n_c - n_d}{\frac{1}{2}n(n-1)}
$$

Dove:
- $n_c$ è il numero di coppie concordanti
- $n_d$ è il numero di coppie discordanti

Una coppia $(x_i, y_i)$ e $(x_j, y_j)$ è:
- **Concordante** se $(x_i - x_j)(y_i - y_j) > 0$
- **Discordante** se $(x_i - x_j)(y_i - y_j) < 0$

## **5. Test di Significatività**

### **5.1. Test per la Correlazione di Pearson**

Per testare l'ipotesi nulla $H_0: \rho = 0$ contro $H_1: \rho \neq 0$, si utilizza la statistica:

$$
t = r\sqrt{\frac{n-2}{1-r^2}}
$$

Che segue una distribuzione $t$ di Student con $n-2$ gradi di libertà.

**Procedura:**
1. Calcolare la statistica $t$
2. Confrontare con il valore critico $t_{\alpha/2, n-2}$
3. Se $|t| > t_{\alpha/2, n-2}$, rifiutare $H_0$

### **5.2. Intervallo di Confidenza**

Per costruire un intervallo di confidenza per $\rho$, si utilizza la **trasformazione di Fisher**:

$$
z = \frac{1}{2}\ln\left(\frac{1+r}{1-r}\right) = \text{arctanh}(r)
$$

La statistica $z$ è approssimativamente normale con:

$$
z \sim \mathcal{N}\left(\frac{1}{2}\ln\left(\frac{1+\rho}{1-\rho}\right), \frac{1}{n-3}\right)
$$

L'intervallo di confidenza al $(1-\alpha)\%$ per $\rho$ è:

$$
\left[\tanh\left(z - z_{\alpha/2}\sqrt{\frac{1}{n-3}}\right), \tanh\left(z + z_{\alpha/2}\sqrt{\frac{1}{n-3}}\right)\right]
$$

## **6. Correlazione Parziale**

La **correlazione parziale** misura la correlazione tra due variabili $X$ e $Y$ controllando l'effetto di una o più variabili addizionali $Z$.

### **6.1. Definizione**

La correlazione parziale tra $X$ e $Y$ dato $Z$ è:

$$
r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ}r_{YZ}}{\sqrt{1-r_{XZ}^2}\sqrt{1-r_{YZ}^2}}
$$

**Interpretazione:**
- Misura la correlazione "pura" tra $X$ e $Y$ rimuovendo l'effetto lineare di $Z$
- Utile per identificare correlazioni spurie dovute a variabili confondenti

### **6.2. Esempio**

Consideriamo:
- $X$: consumo di gelato
- $Y$: numero di annegamenti
- $Z$: temperatura

Potremmo osservare una forte correlazione tra $X$ e $Y$ ($r_{XY} > 0$), ma questa potrebbe essere spurie causata da $Z$. La correlazione parziale $r_{XY \cdot Z}$ potrebbe essere vicina a zero, indicando che la temperatura è la vera causa comune.

## **7. Correlazione e Regressione**

Esiste una stretta relazione tra correlazione e regressione lineare semplice.

### **7.1. Relazione con il Coefficiente di Determinazione**

Nella regressione lineare semplice $y = w_0 + w_1 x + \epsilon$, il coefficiente di determinazione $R^2$ è uguale al quadrato del coefficiente di correlazione:

$$
R^2 = r_{X,Y}^2
$$

**Dimostrazione:**

Il coefficiente di determinazione è definito come:

$$
R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
$$

Nella regressione semplice, può essere dimostrato che:

$$
R^2 = \frac{[\sum(x_i - \bar{x})(y_i - \bar{y})]^2}{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2} = r_{X,Y}^2
$$

Questo segue direttamente dalla definizione di $r_{X,Y}$. $\square$

### **7.2. Relazione con il Coefficiente Angolare**

Il coefficiente angolare della regressione lineare può essere espresso in termini di correlazione:

$$
w_1 = r_{X,Y} \frac{\sigma_Y}{\sigma_X}
$$

Questo mostra che:
- Il segno di $w_1$ è determinato dal segno di $r_{X,Y}$
- La pendenza dipende sia dalla correlazione che dal rapporto delle deviazioni standard

## **8. Limitazioni e Precauzioni**

### **8.1. Correlazione misura solo relazioni lineari**

Il coefficiente di Pearson cattura solo relazioni lineari. Relazioni non lineari (quadratiche, esponenziali, ecc.) possono avere $r \approx 0$ anche se esiste una forte dipendenza.

**Esempio:** Per la relazione $y = x^2$ con $x \in [-1, 1]$, si ha $r_{X,Y} = 0$ ma esiste una dipendenza funzionale perfetta.

### **8.2. Outlier e Valori Influenti**

La correlazione di Pearson è sensibile agli outlier, che possono:
- Aumentare artificialmente la correlazione
- Mascherare correlazioni reali
- Invertire il segno della correlazione

**Soluzione:** Utilizzare correlazioni di rango (Spearman, Kendall) o identificare e trattare gli outlier.

### **8.3. Correlazione Spuria**

Due variabili possono essere correlate senza alcuna relazione causale diretta, a causa di:
- **Variabile confondente**: Una terza variabile influenza entrambe
- **Causalità inversa**: La direzione causale è opposta a quella ipotizzata
- **Coincidenza**: Correlazione puramente casuale

### **8.4. Dimensione del Campione**

Con campioni piccoli:
- La correlazione può essere significativa anche se debole
- La stima è meno affidabile
- È necessario verificare la significatività statistica

## **9. Applicazioni Pratiche**

### **9.1. Analisi Esplorativa dei Dati**

- Identificazione di variabili correlate prima della modellazione
- Costruzione di matrici di correlazione per dataset multivariati
- Visualizzazione tramite heatmap o scatter plot

### **9.2. Feature Selection**

- Rimozione di variabili altamente correlate (multicollinearità)
- Selezione delle feature più correlate con la variabile target
- Identificazione di gruppi di variabili ridondanti

### **9.3. Validazione di Ipotesi**

- Verifica di relazioni teoriche tra variabili
- Test di ipotesi scientifiche
- Analisi di serie temporali (autocorrelazione)

### **9.4. Diagnostica di Modelli**

- Analisi dei residui nella regressione
- Verifica dell'indipendenza degli errori
- Identificazione di pattern non catturati dal modello

## **10. Conclusioni**

La correlazione è uno strumento fondamentale per:
- Quantificare relazioni lineari tra variabili
- Guidare l'analisi esplorativa e la modellazione
- Identificare potenziali relazioni causali (che richiedono ulteriori indagini)

**Punti chiave da ricordare:**
- Correlazione ≠ Causalità
- Il coefficiente di Pearson misura solo relazioni lineari
- Utilizzare misure alternative (Spearman, Kendall) per dati non parametrici
- Verificare sempre la significatività statistica
- Considerare la correlazione parziale per controllare variabili confondenti
- Esplorare visivamente i dati prima di interpretare la correlazione

**Risorse aggiuntive:**
- *Statistical Methods* - Snedecor & Cochran
- *The Elements of Statistical Learning* - Hastie, Tibshirani, Friedman
- *Introduction to Statistical Learning* - James, Witten, Hastie, Tibshirani
