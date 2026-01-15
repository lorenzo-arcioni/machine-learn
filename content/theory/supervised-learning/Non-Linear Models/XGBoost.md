# XGBoost: Un Sistema Scalabile di Tree Boosting

## Indice
1. [Introduzione](#introduzione)
2. [Fondamenti del Tree Boosting](#fondamenti)
3. [Obiettivo Regolarizzato](#obiettivo)
4. [Algoritmi di Split Finding](#split-finding)
5. [Design del Sistema](#system-design)
6. [Ottimizzazioni Avanzate](#ottimizzazioni)
7. [Risultati Sperimentali](#risultati)

---

## 1. Introduzione {#introduzione}

### Cos'è XGBoost?

XGBoost (eXtreme Gradient Boosting) è un sistema di machine learning altamente efficiente e scalabile basato sul **gradient tree boosting**. Sviluppato da Tianqi Chen e Carlos Guestrin all'Università di Washington, è diventato lo strumento di riferimento per competizioni di machine learning e applicazioni industriali.

### Perché XGBoost è importante?

Nel 2015, **17 delle 29 soluzioni vincenti** su Kaggle hanno utilizzato XGBoost. Questa predominanza dimostra l'efficacia del sistema in una vasta gamma di problemi:
- Classificazione di eventi in fisica delle alte energie
- Previsione di vendite
- Click-through rate prediction
- Classificazione di malware
- Rilevamento del movimento
- E molti altri

### Principali Innovazioni

XGBoost introduce quattro innovazioni fondamentali:

1. **Algoritmo sparsity-aware**: gestisce in modo efficiente dati sparsi (missing values, encoding one-hot)
2. **Weighted quantile sketch**: permette l'apprendimento approssimato con garanzie teoriche
3. **Struttura a blocchi cache-aware**: ottimizza l'accesso alla memoria
4. **Out-of-core computation**: gestisce dataset che non entrano in memoria

---

## 2. Fondamenti del Tree Boosting {#fondamenti}

### Cos'è il Gradient Boosting?

Il gradient boosting è una tecnica di **ensemble learning** che combina più modelli deboli (tipicamente alberi di decisione) per creare un modello forte. L'idea chiave è aggiungere iterativamente nuovi modelli che correggono gli errori dei modelli precedenti.

### Il Modello Ensemble

Dato un dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}$ con $n$ esempi e $m$ features ($|\mathcal{D}| = n$, $\mathbf{x}_i \in \mathbb{R}^m$, $y_i \in \mathbb{R}$), XGBoost usa un modello ensemble di $K$ alberi:

$$\hat{y}_i = \phi(\mathbf{x}_i) = \sum_{k=1}^K f_k(\mathbf{x}_i), \quad f_k \in \mathcal{F}$$

dove $\mathcal{F} = \{f(\mathbf{x}) = w_{q(\mathbf{x})}\}$ è lo spazio degli alberi di regressione (CART - Classification And Regression Trees).

#### Componenti di un Albero

Ogni funzione $f_k$ è definita da:
- **Struttura $q$**: mappa un esempio a un indice di foglia ($q: \mathbb{R}^m \rightarrow T$)
- **Pesi delle foglie $w$**: vettore di $T$ valori reali ($w \in \mathbb{R}^T$)
- $T$ è il numero di foglie nell'albero

**Differenza con alberi di decisione classici**: A differenza degli alberi di decisione che assegnano categorie, gli alberi di regressione assegnano un **punteggio continuo** $w_i$ a ogni foglia.

---

## 3. Obiettivo Regolarizzato {#obiettivo}

### Funzione Obiettivo

XGBoost minimizza un obiettivo **regolarizzato** che bilancia accuratezza e complessità del modello:

$$\mathcal{L}(\phi) = \sum_{i} l(\hat{y}_i, y_i) + \sum_{k} \Omega(f_k)$$

dove:
- $l$ è una **funzione di loss differenziabile e convessa** (es. errore quadratico, log-loss)
- $\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$ è il **termine di regolarizzazione**

#### Interpretazione del Termine di Regolarizzazione

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$$

- $\gamma T$: penalizza il **numero di foglie** (favorisce alberi più semplici)
- $\frac{1}{2}\lambda \|w\|^2$: penalizza **pesi grandi** (regolarizzazione L2, simile a Ridge)

**Perché la regolarizzazione è importante?**
- Previene l'overfitting
- Favorisce modelli più semplici e interpretabili
- Migliora la generalizzazione

Quando $\lambda = \gamma = 0$, l'obiettivo si riduce al gradient boosting tradizionale.

### Apprendimento Additivo

Poiché ottimizzare direttamente $K$ alberi è intrattabile, XGBoost usa un approccio **greedy additivo**:

1. Inizia con $\hat{y}_i^{(0)} = 0$
2. A ogni iterazione $t$, aggiungi un albero $f_t$ che minimizza:

$$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$$

### Approssimazione di Taylor del Secondo Ordine

Per ottimizzare velocemente l'obiettivo, XGBoost usa l'**espansione di Taylor** del secondo ordine:

$$\mathcal{L}^{(t)} \simeq \sum_{i=1}^n \left[l(y_i, \hat{y}^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2}h_i f_t^2(\mathbf{x}_i)\right] + \Omega(f_t)$$

dove:
- $g_i = \frac{\partial l(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}$ è il **gradiente primo**
- $h_i = \frac{\partial^2 l(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$ è il **gradiente secondo** (Hessiana)

**Perché il secondo ordine?**
- Convergenza più rapida (come il metodo di Newton vs. gradient descent)
- Migliore approssimazione locale della funzione di loss
- Maggiore stabilità numerica

Rimuovendo i termini costanti:

$$\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^n \left[g_i f_t(\mathbf{x}_i) + \frac{1}{2}h_i f_t^2(\mathbf{x}_i)\right] + \Omega(f_t)$$

### Calcolo dei Pesi Ottimali

Definiamo $I_j = \{i | q(\mathbf{x}_i) = j\}$ come l'insieme di esempi assegnati alla foglia $j$.

Riscriviamo l'obiettivo raggruppando per foglie:

$$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^T \left[\left(\sum_{i \in I_j} g_i\right)w_j + \frac{1}{2}\left(\sum_{i \in I_j} h_i + \lambda\right)w_j^2\right] + \gamma T$$

Questa è una **funzione quadratica** in $w_j$. Per ogni foglia $j$, il peso ottimale è:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

**Interpretazione**:
- Numeratore: somma dei gradienti (direzione di "correzione")
- Denominatore: somma delle Hessiane + regolarizzazione (scala di "confidenza")

Sostituendo $w_j^*$ nell'obiettivo, otteniamo il **punteggio di qualità** della struttura:

$$\tilde{\mathcal{L}}^{(t)}(q) = -\frac{1}{2}\sum_{j=1}^T \frac{\left(\sum_{i \in I_j} g_i\right)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T$$

Questo punteggio misura quanto è "buona" una particolare struttura di albero $q$.

### Valutazione degli Split

Per decidere dove dividere un nodo, calcoliamo il **guadagno** dello split:

Sia $I = I_L \cup I_R$ l'insieme di esempi prima dello split, e $I_L$, $I_R$ gli insiemi dopo. Il guadagno è:

$$\mathcal{L}_{split} = \frac{1}{2}\left[\frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left(\sum_{i \in I_R} g_i\right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda}\right] - \gamma$$

**Interpretazione**:
- Primi due termini: qualità dei due nodi figli
- Terzo termine: qualità del nodo padre
- $-\gamma$: penalità per aver aggiunto una foglia

Uno split è vantaggioso se $\mathcal{L}_{split} > 0$.

### Tecniche Aggiuntive di Regolarizzazione

#### 1. Shrinkage (Learning Rate)

Dopo ogni iterazione, i pesi dei nuovi alberi sono scalati di un fattore $\eta$ (tipicamente 0.01-0.3):

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(\mathbf{x}_i)$$

**Effetto**: riduce l'influenza di ogni singolo albero, lasciando spazio agli alberi futuri per miglioramenti incrementali.

#### 2. Column Subsampling

A ogni iterazione (o split), si seleziona casualmente una frazione delle features (tipicamente 50-80%).

**Vantaggi**:
- Previene l'overfitting (simile a Random Forest)
- Accelera il calcolo
- Secondo il feedback degli utenti, è più efficace del row subsampling

---

## 4. Algoritmi di Split Finding {#split-finding}

### 4.1 Exact Greedy Algorithm

L'algoritmo **exact greedy** enumera tutti i possibili split per ogni feature:

**Procedimento**:
1. Per ogni feature $k = 1, \ldots, m$:
   - Ordina gli esempi per il valore della feature $k$
   - Scansiona linearmente per calcolare le statistiche di gradiente cumulative
   - Valuta tutti i possibili split usando la formula del guadagno

**Complessità**: $O(n \cdot m \cdot \log n)$ per albero (dominato dall'ordinamento)

**Vantaggi**:
- Trova sempre lo split ottimale
- Semplice da implementare

**Svantaggi**:
- Non scalabile per dati enormi
- Problematico quando i dati non entrano in memoria

### 4.2 Approximate Algorithm

Per dataset grandi, XGBoost propone un algoritmo **approssimato** basato su quantili:

**Idea chiave**: Invece di considerare tutti i possibili valori di split, proponi un insieme di $l$ **candidati** $S_k = \{s_{k1}, s_{k2}, \ldots, s_{kl}\}$ per ogni feature $k$.

**Varianti**:

1. **Global proposal**: i candidati sono proposti una sola volta all'inizio e usati per tutti i livelli dell'albero
2. **Local proposal**: i candidati sono ri-proposti dopo ogni split (più costoso ma più accurato per alberi profondi)

**Procedimento**:
1. Proponi candidati basati sui percentili della distribuzione
2. Mappa i valori continui in bucket definiti dai candidati
3. Aggrega le statistiche per bucket
4. Trova il miglior split tra i candidati

**Parametro chiave**: $\epsilon$ (precisione dell'approssimazione)
- Approssimativamente $1/\epsilon$ candidati per feature
- $\epsilon = 0.1$ → circa 10 bucket per feature
- $\epsilon$ più piccolo → più accurato ma più costoso

### 4.3 Weighted Quantile Sketch

**Problema**: Come selezionare i candidati in modo che rappresentino bene i dati?

XGBoost usa i **quantili pesati** invece dei semplici percentili.

#### Perché Pesati?

Riscriviamo l'obiettivo approssimato come:

$$\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^n \frac{1}{2}h_i\left(f_t(\mathbf{x}_i) - \frac{g_i}{h_i}\right)^2 + \Omega(f_t) + \text{const}$$

Questa è una **weighted squared loss** con:
- Etichette: $g_i / h_i$
- Pesi: $h_i$

Quindi gli esempi con Hessiana maggiore dovrebbero avere più influenza nella selezione dei candidati.

#### Rank Function

Definiamo la **rank function pesata** per la feature $k$:

$$r_k(z) = \frac{1}{\sum_{(x,h) \in \mathcal{D}_k} h} \sum_{(x,h) \in \mathcal{D}_k, x < z} h$$

dove $\mathcal{D}_k = \{(x_{1k}, h_1), (x_{2k}, h_2), \ldots, (x_{nk}, h_n)\}$.

**Obiettivo**: Trovare candidati $\{s_{k1}, s_{k2}, \ldots, s_{kl}\}$ tali che:

$$|r_k(s_{k,j}) - r_k(s_{k,j+1})| < \epsilon$$

con $s_{k1} = \min_i x_{ik}$ e $s_{kl} = \max_i x_{ik}$.

**Innovazione**: XGBoost introduce un nuovo algoritmo distribuibile con garanzie teoriche per calcolare questi quantili pesati (descritto nell'appendix del paper).

### 4.4 Sparsity-Aware Split Finding

**Problema**: I dati reali sono spesso **sparsi** a causa di:
- Missing values
- Zero frequenti nelle statistiche
- Feature engineering (es. one-hot encoding)

**Soluzione**: Algoritmo che impara una **direzione di default** per ogni split.

#### Come Funziona

Per ogni split candidato:
1. Assegna gli esempi **non-missing** ai nodi sinistro/destro normalmente
2. Prova **due scenari**:
   - Esempi missing vanno a **sinistra**
   - Esempi missing vanno a **destra**
3. Scegli la direzione che massimizza il guadagno

**Vantaggi**:
- Complessità **lineare** nel numero di valori non-missing: $O(\|x\|_0)$ invece di $O(n)$
- Gestisce naturalmente tutti i pattern di sparsità
- 50x più veloce su dati sparsi (come dimostrato su Allstate dataset)

**Dettaglio tecnico**: Durante la scansione, consideriamo solo $I_k = \{i \in I | x_{ik} \neq \text{missing}\}$.

---

## 5. Design del Sistema {#system-design}

### 5.1 Column Block Structure

**Problema**: L'ordinamento dei dati è l'operazione più costosa nel tree learning.

**Soluzione**: Struttura a **blocchi** con dati pre-ordinati.

#### Caratteristiche

- Dati memorizzati in blocchi in-memory
- Formato **Compressed Sparse Column (CSC)**
- Ogni colonna ordinata per valore della feature
- Ordinamento fatto **una sola volta** prima del training

#### Vantaggi

1. **Exact greedy**: un singolo blocco con scan lineare
2. **Approximate**: più blocchi distribuibili o su disco
3. **Parallelizzazione**: ogni colonna può essere processata in parallelo
4. **Column subsampling**: facile selezionare subset di colonne

#### Complessità Temporale

**Senza block structure**:
- Exact greedy: $O(Kd\|\mathbf{x}\|_0 \log n)$ dove:
  - $K$ = numero di alberi
  - $d$ = profondità massima
  - $\|\mathbf{x}\|_0$ = numero di entry non-missing

**Con block structure**:
- Exact greedy: $O(Kd\|\mathbf{x}\|_0 + \|\mathbf{x}\|_0 \log n)$
- Approximate: $O(Kd\|\mathbf{x}\|_0 + \|\mathbf{x}\|_0 \log B)$ dove $B$ = max righe per blocco

**Risparmio**: fattore $\log n$ (o $\log q$ per approximate), significativo per $n$ grande.

### 5.2 Cache-Aware Access

**Problema**: La struttura a blocchi richiede accessi alla memoria **non contigui** per le statistiche di gradiente.

#### Pattern di Accesso

Durante lo split finding:
1. Le features sono accedute in ordine (dal blocco ordinato)
2. Gli indici di riga sono **sparsi** e non sequenziali
3. Le statistiche di gradiente ($g_i$, $h_i$) sono accedute per indice di riga
4. Questo crea **cache miss** frequenti

**Dipendenza read/write immediata**:
```
Per ogni valore in colonna ordinata:
    Leggi indice riga i (non contiguo)
    Fetch g_i, h_i (cache miss!)
    Accumula statistiche (scrittura)
```

#### Soluzione: Prefetching

**Algoritmo cache-aware**:
1. Alloca un **buffer interno** per ogni thread
2. Fetch delle statistiche di gradiente in **mini-batch**
3. Accumula su batch invece che elemento per elemento

**Effetto**: trasforma la dipendenza read/write da immediata a più lunga, riducendo l'overhead.

**Risultati**: 2x più veloce su dataset grandi (10M+ esempi).

#### Scelta della Block Size

Per l'algoritmo approssimato, la dimensione del blocco è critica:

- **Troppo piccola**: parallelizzazione inefficiente
- **Troppo grande**: cache miss (le statistiche non entrano in cache)

**Scelta ottimale**: $2^{16}$ (65,536) esempi per blocco
- Bilancia cache e parallelizzazione
- Confermato empiricamente su diversi dataset

### 5.3 Out-of-Core Computation

**Obiettivo**: Processare dataset che non entrano in RAM.

#### Strategia Base

1. Dividi i dati in **multipli blocchi**
2. Memorizza ogni blocco su disco
3. Usa un **thread indipendente** per prefetch dei blocchi in memoria
4. Computa mentre leggi da disco (overlapping I/O e computation)

**Problema**: Il disk I/O domina il tempo di calcolo.

#### Ottimizzazione 1: Block Compression

**Tecnica**:
- Comprimi ogni blocco per colonne
- Decomprimi on-the-fly con thread indipendente durante il caricamento

**Compressione utilizzata**:
- Feature values: algoritmo general-purpose
- Row indices: offset a 16-bit ($2^{16}$ esempi per blocco)

**Risultati**:
- **26-29% compression ratio** sui dataset testati
- Trade-off: computation (decompressione) vs. disk reading

#### Ottimizzazione 2: Block Sharding

**Tecnica**:
- Distribuisci i dati su **multipli dischi** in modo alternato
- Thread di prefetch dedicato per ogni disco
- Thread di training legge alternativamente da ogni buffer

**Vantaggi**:
- Aumenta il **throughput** di lettura
- Parallelizza l'I/O su più dispositivi

**Risultati combinati**:
- Compression: 3x speedup
- Sharding (2 dischi): ulteriori 2x speedup
- **6x più veloce** dell'approccio base

---

## 6. Ottimizzazioni Avanzate {#ottimizzazioni}

### 6.1 Confronto con Altri Sistemi

XGBoost è l'unico sistema che combina tutte le seguenti capacità:

| Feature | XGBoost | pGBRT | Spark MLLib | H2O | scikit-learn | R gbm |
|---------|---------|-------|-------------|-----|--------------|-------|
| Exact greedy | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Approximate global | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ |
| Approximate local | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Out-of-core | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Sparsity-aware | ✓ | ✗ | Partial | Partial | ✗ | Partial |
| Parallel | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |

### 6.2 Integrazione e Portabilità

XGBoost è disponibile in molteplici ecosistemi:

**Linguaggi**:
- Python (con integrazione scikit-learn)
- R
- Julia
- Java/Scala

**Piattaforme distribuite**:
- Hadoop
- Apache Spark
- Apache Flink
- MPI
- Sun Grid Engine

**Libreria base**: rabit per operazioni allreduce distribuite

---

## 7. Risultati Sperimentali {#risultati}

### 7.1 Dataset Utilizzati

#### Allstate Insurance (10M)
- **Task**: Classificazione probabilità di claim assicurativo
- **Features**: 4,227 (principalmente sparse da one-hot encoding)
- **Uso**: Valutare sparsity-aware algorithm

#### Higgs Boson (10M)
- **Task**: Classificazione eventi fisici
- **Features**: 28 (21 cinetiche + 7 derivate)
- **Uso**: Confronti con baseline, cache evaluation

#### Yahoo LTRC (473K)
- **Task**: Learning to rank
- **Features**: 700
- **Query**: ~20K con ~22 documenti ciascuna
- **Uso**: Confronto con pGBRT

#### Criteo (1.7B)
- **Task**: Click-through rate prediction
- **Features**: 67 (13 integer + 26 CTR stats + 26 counts)
- **Dimensione**: > 1 TB in formato LibSVM
- **Uso**: Scalabilità distribuita e out-of-core

### 7.2 Risultati di Classificazione

**Higgs-1M (500 alberi)**:

| Metodo | Tempo per albero (sec) | Test AUC |
|--------|------------------------|----------|
| XGBoost | 0.684 | 0.8304 |
| XGBoost (colsample=0.5) | 0.640 | 0.8245 |
| scikit-learn | 28.51 | 0.8302 |
| R gbm | 1.032 | 0.6224 |

**Osservazioni**:
- XGBoost è **10x più veloce** di scikit-learn con accuratezza simile
- Column subsampling migliora velocità con minima perdita di accuratezza
- R gbm è più veloce ma molto meno accurato (usa greedy one-side)

### 7.3 Learning to Rank

**Yahoo LTRC (500 alberi)**:

| Metodo | Tempo per albero (sec) | NDCG@10 |
|--------|------------------------|---------|
| XGBoost | 0.826 | 0.7892 |
| XGBoost (colsample=0.5) | 0.506 | 0.7913 |
| pGBRT | 2.576 | 0.7915 |

**Osservazioni**:
- XGBoost exact greedy batte pGBRT approximate in velocità
- Column subsampling **migliora** leggermente l'accuratezza (previene overfitting)

### 7.4 Impatto Sparsity-Aware

**Allstate-10K**:
- Sparsity-aware algorithm: **50x più veloce** della versione naive
- Conferma l'importanza critica dell'ottimizzazione per dati sparsi

### 7.5 Out-of-Core Performance

**Criteo (subsets crescenti su singola macchina)**:

| Metodo | 100M | 200M | 400M | 1.7B |
|--------|------|------|------|------|
| Basic | OK | OK | Slow | OOM |
| +Compression | 3x faster | 3x faster | OK | OK |
| +Sharding (2 disks) | 6x faster | 6x faster | 2x faster | OK |

**Macchina**: AWS c3.8xlarge (32 vcore, 2x320GB SSD, 60GB RAM)

**Osservazioni**:
- Compression da sola: 3x speedup
- Sharding aggiunge ulteriore 2x
- Gestisce 1.7B esempi su singola macchina desktop

### 7.6 Performance Distribuita

**Criteo completo (32 nodi EC2 m3.2xlarge, 10 iterazioni)**:

#### End-to-end time (incluso data loading):
- **Spark MLLib**: Out of memory a 400M
- **H2O**: Molto lento nel loading, out of memory a 800M
- **XGBoost**: Scala linearmente fino a 1.7B

#### Per-iteration time:
- XGBoost è **10x più veloce di Spark** per iterazione
- XGBoost è **2.2x più veloce di H2O** per iterazione

#### Scaling con numero di macchine (dataset completo 1.7B):
- **4 macchine**: Gestisce l'intero dataset
- **8 macchine**: ~1.8x speedup
- **16 macchine**: ~3.5x speedup
- **32 macchine**: ~6.5x speedup

Scaling leggermente super-lineare grazie a maggiore file cache disponibile.

---

## 8. Dettagli Implementativi Importanti

### 8.1 Calcolo delle Statistiche di Gradiente

Per ogni loss function, dobbiamo calcolare:

**Loss quadratica** $l(y, \hat{y}) = (y - \hat{y})^2$:
- $g_i = -2(y_i - \hat{y}_i^{(t-1)})$
- $h_i = 2$

**Log-loss** (classificazione binaria) $l(y, \hat{y}) = y\log(1 + e^{-\hat{y}}) + (1-y)\log(1 + e^{\hat{y}})$:
- $g_i = p_i - y_i$ dove $p_i = 1/(1 + e^{-\hat{y}_i^{(t-1)}})$
- $h_i = p_i(1 - p_i)$

### 8.2 Gestione di Missing Values

L'algoritmo sparsity-aware gestisce tre scenari:

1. **Valore realmente mancante**: impara direzione ottimale
2. **Zero in sparse matrix**: trattato come missing, impara direzione
3. **Valore specifico dell'utente**: può essere configurato

### 8.3 Parallelizzazione

**Level di parallelizzazione**:
1. **Feature-level**: ogni feature processata da thread diverso
2. **Tree-level**: multipli alberi costruiti in parallelo (meno comune)
3. **Data-level**: dati partizionati tra nodi (distributed)

**Sincronizzazione**: AllReduce per aggregare statistiche di gradiente

---

## 9. Confronto con Random Forest

| Aspetto | XGBoost | Random Forest |
|---------|---------|---------------|
| **Costruzione alberi** | Sequenziale (boosting) | Parallela (bagging) |
| **Dipendenza** | Ogni albero dipende dai precedenti | Alberi indipendenti |
| **Profondità** | Tipicamente limitata (3-10) | Alberi completi |
| **Prediction** | Somma pesata | Media semplice |
| **Overfitting** | Controllato con regolarizzazione | Controllato con randomness |
| **Interpretabilità** | Difficile (ensemble sequenziale) | Media (aggregazione) |
| **Performance** | Generalmente migliore su strutturati | Buona baseline |

---

## 10. Best Practices e Tuning

### 10.1 Parametri Chiave

**Struttura albero**:
- `max_depth`: 3-10 (default 6)
- `min_child_weight`: sum of instance weight needed in a child
- `gamma`: minimum loss reduction required for split

**Regolarizzazione**:
- `lambda`: L2 reg on weights (default 1)
- `alpha`: L1 reg on weights (default 0)
- `eta`: learning rate 0.01-0.3 (default 0.3)

**Sampling**:
- `subsample`: row sampling ratio 0.5-1.0
- `colsample_bytree`: column sampling per tree 0.5-1.0
- `colsample_bylevel`: column sampling per level

### 10.2 Strategia di Tuning

**Step 1**: Fissa parametri conservativi
- `max_depth`: 6
- `eta`: 0.3
- `subsample`, `colsample_bytree`: 0.8

**Step 2**: Ottimizza struttura albero
- Varia `max_depth` (3-10)
- Varia `min_child_weight` (1-6)
- Usa cross-validation

**Step 3**: Aggiungi regolarizzazione
- Aumenta `lambda` se overfitting
- Aggiungi `alpha` per feature selection

**Step 4**: Riduci learning rate e aumenta alberi
- Riduci `eta` a 0.01-0.1
- Aumenta numero alberi di conseguenza
- Early stopping per evitare overfitting

### 10.3 Segni di Overfitting

- Train error molto < validation error
- Validation error aumenta mentre train error diminuisce
- Performance degrada su test set

**Soluzioni**:
1. Aumenta `lambda`, `alpha`
2. Riduci `max_depth`
3. Aumenta `min_child_weight`
4. Usa `subsample`, `colsample_bytree` < 1.0
5. Riduci `eta` e usa early stopping

---

## 11. Vantaggi e Limitazioni

### 11.1 Vantaggi di XGBoost

**Performance**:
- State-of-the-art su dati tabulari
- Velocità superiore ai competitor
- Scala a miliardi di esempi

**Flessibilità**:
- Supporta custom loss functions
- Gestisce missing values nativamente
- Funziona con dati sparsi

**Robustezza**:
- Regolarizzazione built-in
- Meno propenso a overfitting
- Gestisce outliers bene

**Praticità**:
- Pochi hyperparameter da tunare
- Feature importance automatica
- Integrazione in molti ecosistemi

### 11.2 Limitazioni

**Interpretabilità**:
- Modelli complessi difficili da interpretare
- Ensemble di centinaia di alberi
- Non lineare e non parametrico

**Dati non strutturati**:
- Non ottimale per immagini, testo, audio
- Deep learning preferibile per questi domini
- Richiede feature engineering manuale

**Memoria**:
- Richiede tutto il dataset in memoria (o out-of-core)
- Alberi memorizzati interamente
- Può essere memory-intensive

**Training time**:
- Più lento di linear models
- Sequenziale per natura (boosting)
- Non parallelizzabile quanto Random Forest

**Extrapolation**:
- Non estrapolano bene fuori dal range di training
- Previsioni limitate ai valori visti
- Problematico per serie temporali con trend

---

## 12. Applicazioni Pratiche

### 12.1 Quando Usare XGBoost

**Ideale per**:
- Dati tabulari strutturati
- Classification e regression
- Ranking problems
- Competizioni Kaggle
- Feature importance analysis
- Dataset medi-grandi (1K - 100M+ rows)

**Esempi di successo**:
- Click-through rate prediction (advertising)
- Credit scoring (finanza)
- Fraud detection (sicurezza)
- Customer churn prediction (business)
- Medical diagnosis (healthcare)
- Energy consumption forecasting

### 12.2 Quando NON Usare XGBoost

**Alternative migliori**:
- **Immagini/Video**: CNN (Convolutional Neural Networks)
- **Testo/NLP**: Transformers (BERT, GPT)
- **Audio**: RNN, WaveNet
- **Dati piccoli (< 1000 rows)**: Linear models, SVM
- **Real-time inference critico**: Linear models, decision trees singoli
- **Interpretabilità critica**: Linear models, GAMs, single decision trees

---

## 13. Innovazioni Tecniche Dettagliate

### 13.1 Weighted Quantile Sketch

**Problema formale**:
Dato un multi-set $\mathcal{D}_k = \{(x_{1k}, h_1), (x_{2k}, h_2), \ldots, (x_{nk}, h_n)\}$, definire:

$r_k(z) = \frac{1}{\sum_{(x,h) \in \mathcal{D}_k} h} \sum_{(x,h) \in \mathcal{D}_k, x < z} h$

**Obiettivo**: Trovare $\{s_{k1}, s_{k2}, \ldots, s_{kl}\}$ tale che:

$|r_k(s_{k,j}) - r_k(s_{k,j+1})| < \epsilon$

**Proprietà chiave**:
- **Merge operation**: Combina due summary con errore $\max(\epsilon_1, \epsilon_2)$
- **Prune operation**: Riduce elementi mantenendo garanzie di errore
- **Distribuibile**: Può essere calcolato in parallelo

**Algoritmo**:
1. Costruisci summary locale per ogni partizione
2. Merge dei summary con garanzie di errore
3. Prune per ridurre dimensione mantenendo precisione

### 13.2 Column Block - Dettagli Implementativi

**Struttura dati**:
```
Block = {
    feature_columns: [Column_1, Column_2, ..., Column_m]
    row_indices: sorted per ogni colonna
    gradient_stats: [g_1, g_2, ..., g_n], [h_1, h_2, ..., h_n]
}
```

**Processo di costruzione**:
1. Per ogni feature $k$:
   - Crea coppie $(valore_{ik}, indice_i)$
   - Ordina per valore
   - Memorizza in formato compresso
2. Un solo ordinamento iniziale
3. Riutilizzato per tutte le iterazioni

**Accesso durante split finding**:
```
Per ogni feature k:
    Per ogni valore v in column k (già ordinato):
        i = get_row_index(v)
        g_L += g[i]
        h_L += h[i]
        Calcola gain per split
```

### 13.3 Gestione Missing Values - Algoritmo Completo

**Procedura per ogni split candidato**:

1. **Raccogli statistiche non-missing**:
   $G_{present} = \sum_{i: x_{ik} \neq missing} g_i$
   $H_{present} = \sum_{i: x_{ik} \neq missing} h_i$

2. **Scenario A - Missing vanno a sinistra**:
   - Scansiona valori in ordine crescente
   - Per ogni threshold $t$:
     - $G_L = G_{missing} + \sum_{i: x_{ik} < t} g_i$
     - $G_R = \sum_{i: x_{ik} \geq t} g_i$
     - Calcola gain

3. **Scenario B - Missing vanno a destra**:
   - Scansiona valori in ordine decrescente
   - Per ogni threshold $t$:
     - $G_R = G_{missing} + \sum_{i: x_{ik} \geq t} g_i$
     - $G_L = \sum_{i: x_{ik} < t} g_i$
     - Calcola gain

4. **Scegli direzione ottimale**: quella che massimizza il gain

**Complessità**: $O(|I_k|)$ dove $I_k$ sono gli esempi non-missing (invece di $O(|I|)$)

---

## 14. Matematica Avanzata

### 14.1 Derivazione Completa dell'Obiettivo

Partiamo dalla funzione obiettivo generale:

$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$

**Espansione di Taylor al secondo ordine** intorno a $\hat{y}_i^{(t-1)}$:

$l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) \approx l(y_i, \hat{y}_i^{(t-1)}) + \frac{\partial l}{\partial \hat{y}_i^{(t-1)}} f_t(\mathbf{x}_i) + \frac{1}{2} \frac{\partial^2 l}{\partial (\hat{y}_i^{(t-1)})^2} f_t^2(\mathbf{x}_i)$

Definendo:
- $g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}$
- $h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$

Otteniamo:

$\mathcal{L}^{(t)} \approx \sum_{i=1}^n [l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i)] + \Omega(f_t)$

Rimuovendo il termine costante $l(y_i, \hat{y}_i^{(t-1)})$:

$\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^n [g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i)] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$

**Raggruppando per foglie** con $I_j = \{i | q(\mathbf{x}_i) = j\}$ e $f_t(\mathbf{x}_i) = w_{q(\mathbf{x}_i)}$:

$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^T \left[\left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2}\left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2\right] + \gamma T$

**Minimizzazione rispetto a $w_j$**:

$\frac{\partial \tilde{\mathcal{L}}^{(t)}}{\partial w_j} = \sum_{i \in I_j} g_i + \left(\sum_{i \in I_j} h_i + \lambda\right) w_j = 0$

$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$

**Sostituendo nell'obiettivo**:

$\tilde{\mathcal{L}}^{(t)}(q) = -\frac{1}{2} \sum_{j=1}^T \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T$

### 14.2 Interpretazione Geometrica

Il termine di regolarizzazione può essere visto come:

$\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$

**Interpretazione bayesiana**:
- Prior gaussiano sui pesi: $w_j \sim \mathcal{N}(0, 1/\lambda)$
- Penalità sul numero di foglie: preference per alberi semplici
- MAP estimation invece di MLE

**Interpretazione MDL** (Minimum Description Length):
- $\gamma T$: costo di codifica della struttura
- $\frac{1}{2}\lambda \|w\|^2$: costo di codifica dei pesi
- Minimizzare $\mathcal{L}$ = minimizzare lunghezza descrizione

---

## 15. Confronto con Altre Tecniche di Boosting

### 15.1 AdaBoost vs XGBoost

| Aspetto | AdaBoost | XGBoost |
|---------|----------|---------|
| **Loss** | Exponential loss | Generic differentiable loss |
| **Ottimizzazione** | Peso esempi | Gradient descent in function space |
| **Weak learners** | Tipicamente stumps | Regression trees con depth |
| **Regolarizzazione** | Implicita | Esplicita (L1, L2, gamma) |
| **Robustezza** | Sensibile a outliers | Più robusto |
| **Flessibilità** | Limitata | Alta (custom objectives) |

### 15.2 Gradient Boosting (sklearn) vs XGBoost

| Aspetto | sklearn GBM | XGBoost |
|---------|-------------|---------|
| **Regolarizzazione** | Solo max_depth, min_samples | L1, L2, gamma, alpha |
| **Secondo ordine** | No | Sì (Newton boosting) |
| **Parallelizzazione** | No | Sì |
| **Missing values** | No (errore) | Sì (learn direction) |
| **Sparsità** | Ignora | Ottimizzato |
| **Cache optimization** | No | Sì |
| **Out-of-core** | No | Sì |
| **Velocità** | 1x | 10-40x |

### 15.3 LightGBM vs XGBoost

**LightGBM** (Microsoft, 2017):

**Vantaggi di LightGBM**:
- **Leaf-wise** growth invece di level-wise (più veloce)
- **GOSS** (Gradient-based One-Side Sampling)
- **EFB** (Exclusive Feature Bundling)
- Più veloce su dataset molto grandi
- Usa meno memoria

**Vantaggi di XGBoost**:
- Più maturo e testato
- Level-wise growth più conservativo (meno overfitting)
- Migliore supporto distribuito
- Più opzioni di regolarizzazione

**Quando usare quale**:
- **XGBoost**: Dataset medio-grandi, bisogno di stabilità
- **LightGBM**: Dataset enormi (> 10M rows), bisogno di velocità

### 15.4 CatBoost vs XGBoost

**CatBoost** (Yandex, 2017):

**Vantaggi di CatBoost**:
- **Categorical features** gestite nativamente (senza encoding)
- **Ordered boosting** riduce overfitting
- **Symmetric trees** più veloci in inference
- Meno tuning richiesto (buoni defaults)

**Vantaggi di XGBoost**:
- Più veloce su dati non categorici
- Migliore per dati sparsi
- Ecosistema più maturo
- Più flessibile

---

## 16. Feature Importance in XGBoost

### 16.1 Metriche di Importanza

**1. Gain (default)**:
- Media del guadagno di loss quando feature è usata per split
- Formula: $\frac{1}{K} \sum_{k=1}^K \sum_{\text{splits on feature}} \mathcal{L}_{split}$
- **Pro**: Misura impatto reale sulla loss
- **Contro**: Bias verso features con molti possibili split

**2. Weight (Frequency)**:
- Numero di volte che feature appare negli split
- **Pro**: Semplice, intuitivo
- **Contro**: Non considera qualità degli split

**3. Cover**:
- Media del numero di esempi affetti da split sulla feature
- Formula: $\frac{1}{K} \sum_{k=1}^K \sum_{\text{splits on feature}} |I|$
- **Pro**: Considera distribuzione dei dati
- **Contro**: Bias verso features comuni

### 16.2 SHAP Values per XGBoost

**TreeSHAP** (Lundberg & Lee, 2017):

Calcola contributi Shapley per ogni feature:

$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f_{S \cup \{j\}}(x_S) - f_S(x_S)]$

**Vantaggi**:
- Garanzie teoriche (unico metodo con proprietà desiderabili)
- Additive: $\sum_j \phi_j(x) = f(x) - E[f(X)]$
- Consistenza: se feature contribuisce di più, ha SHAP maggiore

**Implementazione efficiente** per trees:
- Complexity $O(TLD^2)$ invece di $O(TL2^M)$
  - $T$ = numero alberi
  - $L$ = max numero foglie
  - $D$ = max profondità
  - $M$ = numero features

---

## 17. Ottimizzazioni Pratiche

### 17.1 Early Stopping

**Strategia**:
1. Dividi dati in train/validation
2. Monitora metrica su validation set
3. Ferma training se metrica non migliora per $n$ rounds

**Implementazione**:
```
Parametri:
- early_stopping_rounds: numero iterazioni senza miglioramento
- eval_metric: metrica da monitorare
- eval_set: validation set

Processo:
For each iteration t:
    Construisci albero f_t su train
    Valuta su eval_set
    Se metrica non migliora per n rounds:
        STOP e return best iteration
```

**Benefici**:
- Previene overfitting automaticamente
- Riduce tempo training
- Trova numero ottimale alberi

### 17.2 Cross-Validation

**CV distribuito in XGBoost**:
- Ogni fold processato in parallelo
- Stratified CV per classification
- Supporto per custom folds

**Utilizzo ottimale**:
1. CV per hyperparameter tuning
2. Train finale su tutti i dati
3. Early stopping su separate validation

### 17.3 Monotonic Constraints

**Problema**: In alcuni domini, relazioni devono essere monotone
- Es: Credit scoring - income ↑ → credit score ↑
- Es: Insurance - age ↑ → premium ↑

**Soluzione in XGBoost**:
```
Parametro monotone_constraints:
- +1: relazione crescente
- -1: relazione decrescente
-  0: nessun constraint
```

**Implementazione**:
- Durante split finding, considera solo split che rispettano constraint
- Garantisce monotonia globale del modello

---

## 18. Casi d'Uso Avanzati

### 18.1 Custom Objective Functions

XGBoost permette di definire loss custom fornendo:

**Gradient**:
$g_i = \frac{\partial l(y_i, \hat{y}_i)}{\partial \hat{y}_i}$

**Hessian**:
$h_i = \frac{\partial^2 l(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}$

**Esempio - Huber Loss** (robusto a outliers):

$l(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}$

$g = \begin{cases}
\hat{y} - y & \text{if } |y - \hat{y}| \leq \delta \\
\delta \cdot \text{sign}(\hat{y} - y) & \text{otherwise}
\end{cases}$

$h = \begin{cases}
1 & \text{if } |y - \hat{y}| \leq \delta \\
0 & \text{otherwise}
\end{cases}$

### 18.2 Multi-output Regression

Per predire $K$ output simultaneamente:

**Approccio 1 - Modelli separati**:
- Train $K$ modelli XGBoost indipendenti
- **Pro**: Semplice, parallelizzabile
- **Contro**: Ignora correlazioni tra output

**Approccio 2 - Multi-task learning**:
- Custom objective che considera tutti $K$ output
- **Pro**: Sfrutta correlazioni
- **Contro**: Più complesso, richiede tuning

### 18.3 Imbalanced Classification

**Problema**: Classi sbilanciate (es. fraud: 0.1% positivi)

**Soluzioni**:

**1. Scale_pos_weight**:
$\text{scale\_pos\_weight} = \frac{\text{count}(y=0)}{\text{count}(y=1)}$
- Aumenta peso degli esempi positivi

**2. Custom weighted loss**:
- Penalizza errori su classe minoritaria di più

**3. Threshold tuning**:
- XGBoost produce probabilità
- Ottimizza threshold per F1, precision, recall

**4. Focal Loss**:
$FL(p_t) = -(1-p_t)^\gamma \log(p_t)$
- Focalizza su esempi difficili
- $\gamma$ controlla focus (tipicamente 2)

---

## 19. Limitazioni Teoriche e Pratiche

### 19.1 Limiti Teorici

**1. Capacità di Approssimazione**:
- Alberi CART possono approssimare funzioni continue (teorema di approssimazione universale)
- MA richiedono numero esponenziale di foglie per alcune funzioni
- Particolarmente inefficienti per funzioni molto smooth

**2. Sample Complexity**:
- Numero di esempi necessari cresce con complessità della funzione target
- Per funzioni con interazioni di alto ordine, serve $O(2^d)$ esempi
- Non efficiente quanto metodi parametrici quando assumptions sono corretti

**3. Generalizzazione**:
- Bound di generalizzazione dipendono da:
  - Numero alberi $K$
  - Profondità $d$
  - Numero esempi $n$
- Trade-off bias-variance esplicito

### 19.2 Limiti Pratici

**1. Tempo di Training**:
- $O(nmd)$ per iterazione nel caso base
- Con $K$ alberi: $O(Knmd)$
- Non scala bene con numero di features $m$

**2. Consumo Memoria**:
- Block structure richiede $O(nm)$ memoria
- Gradient stats: $O(n)$ per iterazione
- Alberi: $O(KT)$ dove $T$ = foglie medie
- Total: $O(nm + Kn + KT)$

**3. Hyperparameter Sensitivity**:
- Molti parametri da tunare
- Interazioni complesse tra parametri
- Richiede expertise per tuning ottimale

**4. Interpretabilità**:
- Ensemble di centinaia di alberi
- Impossibile visualizzare completamente
- SHAP values aiutano ma sono computazionalmente costosi

---

## 20. Conclusioni e Direzioni Future

### 20.1 Contributi Chiave di XGBoost

1. **Sistema end-to-end** che combina algoritmi e ottimizzazioni di sistema
2. **Scalabilità** senza precedenti (billions di esempi, single machine)
3. **Innovazioni algoritmiche**: sparsity-aware, weighted quantile sketch
4. **Innovazioni di sistema**: cache-aware, out-of-core, block structure
5. **Impatto pratico**: dominanza in competizioni ML e industry adoption

### 20.2 Lezioni Apprese

**Design di Sistemi ML**:
- **Co-design** algoritmo-sistema è cruciale
- Ottimizzazioni hardware (cache, I/O) fanno differenza enorme
- Scalabilità richiede attenzione a ogni livello dello stack

**Machine Learning Pratico**:
- Regolarizzazione è essenziale per generalizzazione
- Gestione sparsità e missing values deve essere first-class
- Flessibilità (custom objectives) abilita nuove applicazioni

### 20.3 Sviluppi Post-Paper

**LightGBM** (2017):
- Leaf-wise growth
- Histogram-based learning
- Più veloce su dataset enormi

**CatBoost** (2017):
- Ordered boosting
- Categorical features native
- Symmetric trees

**XGBoost Improvements**:
- GPU acceleration
- Federated learning support
- Categorical features support (2022)
- Distributed training migliorato

### 20.4 Quando XGBoost Rimane la Scelta Migliore

**Nel 2024-2026**:
- **Dati tabulari strutturati**: ancora SOTA
- **Interpretabilità importante**: con SHAP
- **Risorse limitate**: efficiente su singola macchina
- **Production stability**: maturo e testato
- **Integrazione ecosistema**: supporto ovunque

**Alternative emergenti**:
- **TabNet, FT-Transformer**: neural networks per tabular data
- **AutoML**: automated feature engineering + XGBoost
- **Deep learning**: quando dati non-strutturati o huge scale

### 20.5 Direzioni di Ricerca

**Aperte**:
1. **Theoretical understanding**: perché boosting funziona così bene?
2. **Automated tuning**: come ridurre necessità di expertise?
3. **Neural-symbolic hybrid**: combinare trees e neural nets?
4. **Causal inference**: usare trees per causal discovery?
5. **Online learning**: incremental boosting efficiente?

---

## Appendice: Formule di Riferimento Rapido

### Loss Functions Comuni

**Regression**:
- Square loss: $l(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$
  - $g = \hat{y} - y$, $h = 1$
  
**Binary Classification**:
- Logistic loss: $l(y, \hat{y}) = y\log(1 + e^{-\hat{y}}) + (1-y)\log(1 + e^{\hat{y}})$
  - $p = 1/(1 + e^{-\hat{y}})$
  - $g = p - y$, $h = p(1-p)$

**Multi-class** (softmax):
- $l = -\sum_k y_k \log(p_k)$ dove $p_k = \frac{e^{\hat{y}_k}}{\sum_j e^{\hat{y}_j}}$
  - $g_k = p_k - y_k$
  - $h_k = p_k(1 - p_k)$

### Formule Chiave

**Peso ottimale foglia**:
$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$

**Score struttura**:
$\text{Score}(q) = -\frac{1}{2}\sum_{j=1}^T \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T$

**Gain split**:
$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}\right] - \gamma$

dove $G = \sum_i g_i$, $H = \sum_i h_i$ per il rispettivo set.

---

## Riferimenti e Risorse

### Paper Originale
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.

### Risorse Online
- Documentazione ufficiale: https://xgboost.readthedocs.io/
- Repository GitHub: https://github.com/dmlc/xgboost
- Tutorial: https://xgboost.readthedocs.io/en/latest/tutorials/

### Paper Correlati
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine
- Friedman, J. H. (2002). Stochastic gradient boosting
- Breiman, L. (2001). Random forests
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree
- Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features

---

**Fine della Nota Completa su XGBoost**
