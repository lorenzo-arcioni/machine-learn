# CatBoost: Unbiased Boosting with Categorical Features

## Indice

1. [Introduzione](#introduzione)
2. [Background Teorico](#background-teorico)
3. [Il Problema delle Feature Categoriche](#il-problema-delle-feature-categoriche)
4. [Target Statistics](#target-statistics)
5. [Prediction Shift e Ordered Boosting](#prediction-shift-e-ordered-boosting)
6. [Implementazione Pratica](#implementazione-pratica)
7. [Risultati Sperimentali](#risultati-sperimentali)
8. [Conclusioni](#conclusioni)

---

## Introduzione

**CatBoost** (abbreviazione di "Categorical Boosting") è una libreria open-source per il gradient boosting sviluppata da Yandex che introduce due innovazioni algoritmiche fondamentali:

1. **Ordered Boosting**: un'alternativa basata su permutazioni all'algoritmo classico di gradient boosting
2. **Algoritmo innovativo per il processing delle feature categoriche**

Entrambe queste tecniche sono state create per combattere il **prediction shift**, un particolare tipo di target leakage presente in tutte le implementazioni esistenti di gradient boosting.

### Motivazione

Il gradient boosting è una delle tecniche di machine learning più potenti e viene utilizzato con successo in moltissimi ambiti applicativi:
- Ricerca web
- Sistemi di raccomandazione
- Previsioni meteorologiche
- Problemi con feature eterogenee, dati rumorosi e dipendenze complesse

Nonostante il suo successo, **tutti gli algoritmi esistenti soffrono di un problema statistico fondamentale** che porta a un bias sistematico nelle predizioni.

---

## Background Teorico

### Il Problema di Apprendimento

Consideriamo un dataset $\mathcal{D} = \{(\mathbf{x}_k, y_k)\}_{k=1}^n$ dove:
- $\mathbf{x}_k = (x_k^1, \ldots, x_k^m)$ è un vettore casuale di $m$ **feature**
- $y_k \in \mathbb{R}$ è il **target** (può essere binario o una risposta numerica)
- Gli esempi sono indipendenti e identicamente distribuiti secondo una distribuzione sconosciuta $P(\cdot, \cdot)$

**Obiettivo**: Addestrare una funzione $F: \mathbb{R}^m \to \mathbb{R}$ che minimizza la perdita attesa:

$$\mathcal{L}(F) := \mathbb{E} L(y, F(\mathbf{x}))$$

dove:
- $L(\cdot, \cdot)$ è una funzione di perdita liscia
- $(\mathbf{x}, y)$ è un esempio di test campionato da $P$ indipendentemente dal training set

### Gradient Boosting: Principio di Base

Il gradient boosting costruisce iterativamente una sequenza di **approssimazioni** $F^t: \mathbb{R}^m \to \mathbb{R}$, per $t = 0, 1, \ldots$ in modo greedy.

Ad ogni step, la nuova approssimazione è ottenuta in modo additivo:

$$F^t = F^{t-1} + \alpha h^t$$

dove:
- $\alpha$ è lo **step size** (learning rate)
- $h^t: \mathbb{R}^m \to \mathbb{R}$ è il **base predictor** (predittore di base)

Il base predictor viene scelto da una famiglia di funzioni $\mathcal{H}$ per minimizzare la perdita attesa:

$$h^t = \arg\min_{h \in \mathcal{H}} \mathcal{L}(F^{t-1} + h) = \arg\min_{h \in \mathcal{H}} \mathbb{E} L(y, F^{t-1}(\mathbf{x}) + h(\mathbf{x}))$$

#### Gradient Step

In pratica, questo problema di minimizzazione viene affrontato con il **metodo di Newton** o con uno **step di gradiente negativo**. Nel caso dello step di gradiente, $h^t$ viene scelto in modo che $h^t(\mathbf{x})$ approssimi $-g^t(\mathbf{x}, y)$, dove:

$$g^t(\mathbf{x}, y) := \frac{\partial L(y, s)}{\partial s}\bigg|_{s=F^{t-1}(\mathbf{x})}$$

Tipicamente si usa l'approssimazione ai minimi quadrati:

$$h^t = \arg\min_{h \in \mathcal{H}} \mathbb{E}\left(-g^t(\mathbf{x}, y) - h(\mathbf{x})\right)^2$$

**Punto chiave**: Questa è la formulazione teorica. In pratica, l'aspettazione è sconosciuta e viene approssimata usando il training set $\mathcal{D}$.

### Decision Trees come Base Predictors

CatBoost utilizza **alberi di decisione binari** come base predictors. Un albero di decisione è un modello costruito attraverso una partizione ricorsiva dello spazio delle feature $\mathbb{R}^m$ in diverse regioni disgiunte (nodi dell'albero).

#### Struttura di un Albero

La partizione avviene secondo i valori di alcuni **attributi di splitting** $a$, che sono tipicamente variabili binarie del tipo:

$$a = \mathbb{1}_{\{x^k > t\}}$$

dove:
- $x^k$ è una feature numerica o binaria
- $t$ è una **soglia** (threshold)
- Nel caso di feature binarie, $t = 0.5$

**Perché split binari?** 
- Split non binari (es. basati su tutti i valori di una feature categorica) porterebbero a:
  - Alberi poco profondi (incapaci di catturare dipendenze complesse), oppure
  - Alberi molto complessi con numero esponenziale di nodi terminali (con statistiche target più deboli in ciascuno)
- La complessità dell'albero ha un effetto cruciale sull'accuratezza: alberi meno complessi sono meno soggetti a overfitting

#### Rappresentazione Matematica

Ogni foglia (regione finale) dell'albero è associata a un valore che stima la risposta $y$. Un albero $h$ può essere scritto come:

$$h(\mathbf{x}) = \sum_{j=1}^J b_j \mathbb{1}_{\{\mathbf{x} \in R_j\}}$$

dove:
- $R_j$ sono le regioni disgiunte corrispondenti alle foglie
- $b_j$ sono i valori assegnati a ciascuna foglia
- $J$ è il numero totale di foglie

Nel gradient boosting, gli attributi di split e i valori delle foglie sono tipicamente scelti secondo il criterio dei minimi quadrati, poiché l'albero è costruito per approssimare il gradiente negativo.

---

## Il Problema delle Feature Categoriche

### Cosa sono le Feature Categoriche?

Una **feature categorica** è una variabile con un insieme discreto di valori chiamati **categorie**, che non sono confrontabili tra loro. Esempi tipici:
- User ID
- Regione geografica
- Tipo di prodotto
- Publisher
- Categoria di annuncio

Queste feature sono fondamentali in molte applicazioni pratiche, specialmente nel click prediction (previsione dei click su annunci pubblicitari).

### Approcci Esistenti

#### 1. One-Hot Encoding

La tecnica più popolare consiste nel **one-hot encoding**: per ogni categoria si aggiunge una nuova feature binaria che indica se quella categoria è presente.

**Problema**: Per feature con **alta cardinalità** (molte categorie possibili, come "user ID"), questo approccio porta a un numero impraticabile di nuove feature.

**Soluzione comune**: Raggruppare le categorie in un numero limitato di cluster e poi applicare one-hot encoding.

#### 2. Target Statistics (TS)

Un metodo popolare consiste nel raggruppare le categorie in base alle **target statistics**: si stima il valore atteso del target in ogni categoria.

Micci-Barreca (2001) propose di considerare le TS come nuove feature numeriche invece dell'one-hot encoding.

**Vantaggio importante**: Tra tutte le possibili partizioni delle categorie in due insiemi, uno split ottimale sui dati di training (in termini di logloss, Gini index, MSE) può essere trovato tra le soglie sulla feature numerica TS.

#### 3. Gradient Statistics (LightGBM)

LightGBM converte le feature categoriche in **gradient statistics** ad ogni step del gradient boosting.

**Problemi**:
1. Aumento drammatico del tempo di computazione (calcola le statistiche per ogni valore categorico ad ogni step)
2. Aumento del consumo di memoria (deve memorizzare quale categoria appartiene a quale nodo per ogni split)
3. Per risolvere questi problemi, LightGBM raggruppa le categorie meno frequenti in un cluster unico, perdendo parte dell'informazione

### Conclusione sull'Approccio

**Le Target Statistics come feature numeriche sembrano essere il metodo più efficiente** per gestire le feature categoriche con minima perdita di informazione. Richiedono di calcolare e memorizzare solo un numero per categoria.

Le TS sono ampiamente utilizzate, ad esempio nel click prediction (click-through rates), dove feature categoriche come user, region, ad, publisher giocano un ruolo cruciale.

---

## Target Statistics

### Definizione

Come discusso, un modo efficace ed efficiente per gestire una feature categorica $i$ è sostituire la categoria $x_k^i$ del $k$-esimo esempio di training con **una** feature numerica uguale a una **target statistic** $\hat{x}_k^i$.

Comunemente, questa stima il target atteso $y$ condizionato alla categoria:

$$\hat{x}_k^i \approx \mathbb{E}(y \mid x^i = x_k^i)$$

### Greedy TS: L'Approccio Diretto (e Problematico)

Un approccio diretto consiste nello stimare $\mathbb{E}(y \mid x^i = x_k^i)$ come la media di $y$ sugli esempi di training con la stessa categoria $x_k^i$.

Per ridurre il rumore per categorie a bassa frequenza, si applica uno **smoothing** con un prior $p$:

$$\hat{x}_k^i = \frac{\sum_{j=1}^n \mathbb{1}_{\{x_j^i = x_k^i\}} \cdot y_j + a \cdot p}{\sum_{j=1}^n \mathbb{1}_{\{x_j^i = x_k^i\}} + a}$$

dove:
- $a > 0$ è un parametro di smoothing
- $p$ è tipicamente il valore medio del target nel dataset
- Il numeratore somma tutti i target degli esempi con la stessa categoria, più un termine di prior
- Il denominatore conta quanti esempi hanno quella categoria, più il peso del prior

**Spiegazione intuitiva dello smoothing**: 
- Se una categoria appare poche volte, la media dei target potrebbe essere molto rumorosa
- Aggiungendo $a \cdot p$ al numeratore e $a$ al denominatore, "mescoliamo" la stima empirica con il prior $p$
- Quando il numero di osservazioni è piccolo, la stima si avvicina a $p$
- Quando il numero di osservazioni è grande, la stima si avvicina alla media empirica

### Il Problema del Target Leakage

Il problema di questo approccio **greedy** è il **target leakage**: la feature $\hat{x}_k^i$ è calcolata usando $y_k$, il target di $\mathbf{x}_k$ stesso!

Questo porta a un **conditional shift**: la distribuzione di $\hat{x}^i \mid y$ differisce tra esempi di training e test.

#### Esempio Estremo

Consideriamo un caso estremo per illustrare quanto questo possa influenzare drasticamente l'errore di generalizzazione:

**Setup**:
- La $i$-esima feature è categorica
- Tutti i suoi valori sono unici (ogni esempio ha una categoria diversa)
- Per ogni categoria $A$: $P(y=1 \mid x^i = A) = 0.5$ (classificazione binaria)

**Sul training set**:
- $\hat{x}_k^i = \frac{y_k + ap}{1 + a}$
- È sufficiente uno split con soglia $t = \frac{0.5 + ap}{1 + a}$ per classificare perfettamente tutti gli esempi di training

**Sul test set**:
- Per tutti gli esempi test, la greedy TS vale $p$
- Il modello predice $0$ per tutti se $p < t$, oppure $1$ se $p \geq t$
- **Accuracy = 0.5** in entrambi i casi!

### Proprietà Desiderata P1

Formuliamo la seguente proprietà desiderata per le TS:

**P1**: $\mathbb{E}(\hat{x}^i \mid y = v) = \mathbb{E}(\hat{x}_k^i \mid y_k = v)$

dove $(\mathbf{x}_k, y_k)$ è il $k$-esimo esempio di training.

Nell'esempio sopra: 
- $\mathbb{E}(\hat{x}_k^i \mid y_k) = \frac{y_k + ap}{1 + a}$ (dipende da $y_k$!)
- $\mathbb{E}(\hat{x}^i \mid y) = p$ (costante)
- Le due distribuzioni sono diverse → violazione di P1

### Approcci per Evitare il Conditional Shift

L'idea generale è calcolare la TS per $\mathbf{x}_k$ su un sottoinsieme di esempi $\mathcal{D}_k \subset \mathcal{D} \setminus \{\mathbf{x}_k\}$ che **esclude** $\mathbf{x}_k$:

$$\hat{x}_k^i = \frac{\sum_{\mathbf{x}_j \in \mathcal{D}_k} \mathbb{1}_{\{x_j^i = x_k^i\}} \cdot y_j + a \cdot p}{\sum_{\mathbf{x}_j \in \mathcal{D}_k} \mathbb{1}_{\{x_j^i = x_k^i\}} + a}$$

#### Holdout TS

Un modo è partizionare il training set in due parti: $\mathcal{D} = \hat{\mathcal{D}}_0 \sqcup \hat{\mathcal{D}}_1$

- Si usa $\mathcal{D}_k = \hat{\mathcal{D}}_0$ per calcolare le TS
- Si usa $\hat{\mathcal{D}}_1$ per il training

**Vantaggi**: Questo approccio holdout soddisfa P1.

**Svantaggi**: Riduce significativamente la quantità di dati usati sia per il training che per il calcolo delle TS. Viola la seguente proprietà desiderata:

**P2**: *Uso efficace di tutti i dati di training sia per calcolare le TS che per apprendere il modello*

#### Leave-One-Out TS

A prima vista, la tecnica **leave-one-out** potrebbe sembrare funzionare bene:
- Per esempi di training $\mathbf{x}_k$: si prende $\mathcal{D}_k = \mathcal{D} \setminus \{\mathbf{x}_k\}$
- Per esempi di test: si prende $\mathcal{D}_k = \mathcal{D}$

**Sorprendentemente, questo NON previene il target leakage!**

**Controesempio**: Consideriamo una feature categorica costante: $x_k^i = A$ per tutti gli esempi.

Sia $n^+$ il numero di esempi con $y = 1$. Allora:

$$\hat{x}_k^i = \frac{n^+ - y_k + a \cdot p}{n - 1 + a}$$

Si può classificare perfettamente il training set con uno split alla soglia:

$$t = \frac{n^+ - 0.5 + a \cdot p}{n - 1 + a}$$

Infatti:
- Se $y_k = 1$: $\hat{x}_k^i = \frac{n^+ - 1 + ap}{n - 1 + a} < t$
- Se $y_k = 0$: $\hat{x}_k^i = \frac{n^+ + ap}{n - 1 + a} > t$

### Ordered TS: La Soluzione di CatBoost

CatBoost usa una strategia più efficace basata sul **principio di ordinamento** (ordering principle), idea centrale dell'intero paper.

**Ispirazione**: Algoritmi di online learning che ricevono esempi di training sequenzialmente nel tempo. In questi algoritmi, i valori delle TS per ogni esempio dipendono solo dalla "storia" osservata fino a quel momento.

**Adattamento all'offline setting**:

1. Introduciamo un "tempo" artificiale: una **permutazione casuale** $\sigma$ degli esempi di training
2. Per ogni esempio, usiamo tutta la "storia" disponibile per calcolare la sua TS:
   - $\mathcal{D}_k = \{\mathbf{x}_j : \sigma(j) < \sigma(k)\}$ per un esempio di training
   - $\mathcal{D}_k = \mathcal{D}$ per un esempio di test

**Proprietà delle Ordered TS**:
- ✓ Soddisfa P1 (nessun conditional shift)
- ✓ Soddisfa P2 (usa tutti i dati per il training)
- **Problema**: Se usiamo una sola permutazione, gli esempi iniziali hanno TS con varianza molto più alta degli esempi successivi

**Soluzione**: CatBoost usa **diverse permutazioni** per diversi step del gradient boosting (dettagli in Sezione Implementazione).

---

## Prediction Shift e Ordered Boosting

### Identificazione del Problema

In questa sezione riveliamo il problema del **prediction shift** nel gradient boosting, che non era stato precedentemente riconosciuto né affrontato.

Come nel caso delle TS, il prediction shift è causato da un particolare tipo di target leakage.

### Il Problema nella Pratica

Ricordiamo che nella teoria (Sezione Background) il gradient step è definito come:

$$h^t = \arg\min_{h \in \mathcal{H}} \mathbb{E}\left(-g^t(\mathbf{x}, y) - h(\mathbf{x})\right)^2$$

**In pratica**, l'aspettazione è sconosciuta e viene approssimata usando lo stesso dataset $\mathcal{D}$:

$$h^t = \arg\min_{h \in \mathcal{H}} \frac{1}{n} \sum_{k=1}^n \left(-g^t(\mathbf{x}_k, y_k) - h(\mathbf{x}_k)\right)^2$$

dove ricordiamo che:

$$g^t(\mathbf{x}_k, y_k) = \frac{\partial L(y_k, s)}{\partial s}\bigg|_{s=F^{t-1}(\mathbf{x}_k)}$$

### La Catena di Shift

Descriviamo la seguente catena di problemi:

1. **Shift dei gradienti**: La distribuzione condizionale del gradiente $g^t(\mathbf{x}_k, y_k) \mid \mathbf{x}_k$ (considerando la casualità di $\mathcal{D} \setminus \{\mathbf{x}_k\}$) è shifted rispetto alla distribuzione su un esempio test $g^t(\mathbf{x}, y) \mid \mathbf{x}$

2. **Bias del base predictor**: Il base predictor $h^t$ definito dall'equazione pratica è biased rispetto alla soluzione dell'equazione teorica

3. **Impatto sulla generalizzazione**: Questo influenza infine la capacità di generalizzazione del modello addestrato $F^t$

### Causa: Target Leakage

Questi problemi sono causati da **target leakage**. I gradienti usati ad ogni step sono stimati usando i valori target degli stessi data point su cui è stato costruito il modello corrente $F^{t-1}$.

**Il problema fondamentale**: La distribuzione condizionale $F^{t-1}(\mathbf{x}_k) \mid \mathbf{x}_k$ per un esempio di training $\mathbf{x}_k$ è shifted, in generale, rispetto alla distribuzione $F^{t-1}(\mathbf{x}) \mid \mathbf{x}$ per un esempio test $\mathbf{x}$.

Questo è chiamato **prediction shift**.

### Letteratura Esistente

Il problema è stato menzionato in precedenza ma mai formalmente definito:

#### Iterated Bagging (Breiman, 2001)

Basato sulla stima out-of-bag, costruisce un weak learner bagged ad ogni iterazione basandosi su stime residue "out-of-bag".

**Problema**: Come dimostrato formalmente dagli autori nell'Appendice, queste stime residue sono ancora shifted!

Inoltre, lo schema di bagging aumenta il tempo di apprendimento di un fattore pari al numero di bucket di dati.

#### Subsampling (Friedman, 2002)

Proposto il subsampling del dataset ad ogni iterazione.

**Problema**: Affronta il problema in modo molto più euristico e solo lo allevia, non lo risolve.

### Analisi Formale: Caso Semplificato

Analizziamo formalmente il prediction shift in un caso semplice:
- Task di regressione con funzione di perdita quadratica: $L(y, \hat{y}) = (y - \hat{y})^2$

In questo caso, il gradiente negativo $-g^t(\mathbf{x}_k, y_k)$ può essere sostituito dalla **funzione residuo**:

$$r^{t-1}(\mathbf{x}_k, y_k) := y_k - F^{t-1}(\mathbf{x}_k)$$

(abbiamo rimosso il moltiplicatore 2, che non è rilevante per l'analisi)

#### Setup del Teorema

Assumiamo:
- $m = 2$ feature: $x^1, x^2$
- Entrambe sono variabili di Bernoulli i.i.d. con $p = 1/2$
- $y = f^*(\mathbf{x}) = c_1 x^1 + c_2 x^2$ (dipendenza lineare vera)
- Facciamo $N = 2$ step di gradient boosting con decision stumps (alberi di profondità 1)
- Step size $\alpha = 1$
- Otteniamo un modello $F = F^2 = h^1 + h^2$
- Assumiamo (senza perdita di generalità) che $h^1$ sia basato su $x^1$ e $h^2$ su $x^2$

### Teorema 1 (Risultato Principale)

**Parte 1**: Se due campioni indipendenti $\mathcal{D}_1$ e $\mathcal{D}_2$ di dimensione $n$ sono usati per stimare $h^1$ e $h^2$ rispettivamente, allora:

$$\mathbb{E}_{\mathcal{D}_1, \mathcal{D}_2} F^2(\mathbf{x}) = f^*(\mathbf{x}) + O(1/2^n)$$

per ogni $\mathbf{x} \in \{0,1\}^2$.

**Interpretazione**: Il modello addestrato è una stima **unbiased** della vera dipendenza $y = f^*(\mathbf{x})$ (fino a un termine esponenzialmente piccolo, che si verifica per ragioni tecniche).

**Parte 2**: Se lo stesso dataset $\mathcal{D} = \mathcal{D}_1 = \mathcal{D}_2$ è usato per entrambi $h^1$ e $h^2$, allora:

$$\mathbb{E}_{\mathcal{D}} F^2(\mathbf{x}) = f^*(\mathbf{x}) - \frac{1}{n-1} c_2 \left(x^2 - \frac{1}{2}\right) + O(1/2^n)$$

**Interpretazione**: Soffriamo di un **bias** pari a $-\frac{1}{n-1} c_2(x^2 - 1/2)$, che è:
- Inversamente proporzionale alla dimensione dei dati $n$
- Dipendente dalla relazione vera $f^*$ (proporzionale a $c_2$)

#### Sketch della Dimostrazione (Parte 2)

Denotiamo con $\xi_{st}$, $s,t \in \{0,1\}$, il numero di esempi $(\mathbf{x}_k, y_k) \in \mathcal{D}$ con $\mathbf{x}_k = (s,t)$.

**Step 1 - Primo stump**: 

$$h^1(s,t) = c_1 s + \frac{c_2 \xi_{s1}}{\xi_{s0} + \xi_{s1}}$$

Il valore atteso su un **esempio test** $\mathbf{x}$ è:

$$\mathbb{E}(h^1(\mathbf{x})) = c_1 x^1 + \frac{c_2}{2}$$

Il valore atteso su un **esempio di training** $\mathbf{x}_k$ è diverso:

$$\mathbb{E}(h^1(\mathbf{x}_k)) = \left(c_1 x^1 + \frac{c_2}{2}\right) - c_2 \left(\frac{2x^2 - 1}{n}\right) + O(2^{-n})$$

**Questo è il prediction shift di $h^1$!**

**Step 2 - Conseguenza sul secondo stump**: Come conseguenza di questo shift, il valore atteso di $h^2$ è:

$$\mathbb{E}(h^2(\mathbf{x})) = c_2\left(x^2 - \frac{1}{2}\right)\left(1 - \frac{1}{n-1}\right) + O(2^{-n})$$

su un esempio test $\mathbf{x}$.

**Step 3 - Risultato finale**:

$$\mathbb{E}(h^1(\mathbf{x}) + h^2(\mathbf{x})) = f^*(\mathbf{x}) - \frac{1}{n-1} c_2\left(x^2 - \frac{1}{2}\right) + O(1/2^n)$$

### Connessione con le Greedy TS

Ricordiamo che le greedy TS $\hat{x}^i$ possono essere considerate come un semplice modello statistico che predice il target $y$.

Soffrono di un problema simile: conditional shift di $\hat{x}_k^i \mid y_k$, causato dal target leakage (usare $y_k$ per calcolare $\hat{x}_k^i$).

---

## Ordered Boosting

### Idea di Base

Proponiamo un algoritmo di boosting che non soffre del problema di prediction shift.

**Scenario ideale** (con dati illimitati): Ad ogni step del boosting, campioniamo un nuovo dataset $\mathcal{D}_t$ indipendentemente e otteniamo residui unshifted applicando il modello corrente ai nuovi esempi di training.

**Problema pratico**: I dati etichettati sono limitati!

### L'Approccio con Modelli Multipli

Assumiamo di voler imparare un modello con $I$ alberi.

**Osservazione chiave**: Per rendere il residuo $r^{I-1}(\mathbf{x}_k, y_k)$ unshifted, abbiamo bisogno che $F^{I-1}$ sia addestrato **senza** l'esempio $\mathbf{x}_k$.

**Problema apparente**: Poiché abbiamo bisogno di residui unbiased per tutti gli esempi di training, nessun esempio può essere usato per training $F^{I-1}$, il che sembrerebbe rendere impossibile il processo di training.

**Soluzione**: È possibile mantenere un insieme di modelli che differiscono per gli esempi usati nel loro training. Poi, per calcolare il residuo su un esempio, usiamo un modello addestrato senza di esso.

### Applicazione del Principio di Ordinamento

Per costruire tale insieme di modelli, usiamo il **principio di ordinamento** (già applicato alle TS).

**Algoritmo concettuale**:
1. Prendiamo una permutazione casuale $\sigma$ degli esempi di training
2. Manteniamo $n$ diversi **modelli di supporto** $M_1, \ldots, M_n$ tali che il modello $M_i$ è appreso usando solo i primi $i$ esempi nella permutazione
3. Ad ogni step, per ottenere il residuo per il $j$-esimo esempio, usiamo il modello $M_{j-1}$

### Algoritmo 1: Ordered Boosting (Versione Concettuale)

**Input**: $\{(\mathbf{x}_k, y_k)\}_{k=1}^n$, $I$ (numero di iterazioni)

1. $\sigma \leftarrow$ permutazione casuale di $[1,n]$
2. $M_i \leftarrow 0$ per $i = 1, \ldots, n$
3. **Per** $t \leftarrow 1$ **a** $I$:
   - **Per** $i \leftarrow 1$ **a** $n$:
     - $r_i \leftarrow y_i - M_{\sigma(i)-1}(\mathbf{x}_i)$ (calcola residuo usando modello precedente)
   - **Per** $i \leftarrow 1$ **a** $n$:
     - $\Delta M \leftarrow \text{LearnModel}((\mathbf{x}_j, r_j) : \sigma(j) \leq i)$
     - $M_i \leftarrow M_i + \Delta M$ (aggiorna modello)
4. **Return** $M_n$

**Problema pratico**: Questo algoritmo non è fattibile nella maggior parte dei task pratici a causa della necessità di addestrare $n$ modelli diversi, il che aumenta la complessità e i requisiti di memoria di un fattore $n$.

**Soluzione**: In CatBoost, è implementata una modifica efficiente di questo algoritmo basata su GBDT con alberi di decisione come base predictors.

### Ordered Boosting con Feature Categoriche

Abbiamo proposto di usare permutazioni casuali per:
- $\sigma_{cat}$: per il calcolo delle TS
- $\sigma_{boost}$: per l'ordered boosting

**Domanda cruciale**: Quando le combiniamo in un unico algoritmo, queste due permutazioni devono essere in qualche modo dipendenti?

**Risposta**: Sì! Devono coincidere: $\sigma_{cat} = \sigma_{boost}$

**Perché?** Se le permutazioni fossero diverse, esisterebbero esempi $\mathbf{x}_i$ e $\mathbf{x}_j$ tali che:
- $\sigma_{boost}(i) < \sigma_{boost}(j)$, ma
- $\sigma_{cat}(i) > \sigma_{cat}(j)$

In questo caso:
- Il modello $M_{\sigma_{boost}(j)}$ è addestrato usando, in particolare, le TS dell'esempio $\mathbf{x}_i$
- Ma queste TS sono calcolate usando $y_j$ (il target di $\mathbf{x}_j$)
- Questo può causare uno shift nella predizione $M_{\sigma_{boost}(j)}(\mathbf{x}_j)$

**Garanzia teorica**: Impostando $\sigma_{cat} = \sigma_{boost}$, garantiamo che il target $y_i$ non sia usato per addestrare $M_i$ (né per il calcolo delle TS, né per la stima del gradiente).

---

## Implementazione Pratica

CatBoost ha due modalità di boosting:

1. **Plain**: L'algoritmo GBDT standard con TS ordinate incorporate
2. **Ordered**: Una modifica efficiente dell'Algoritmo 1 (Ordered Boosting)

### Permutazioni Multiple

All'inizio, CatBoost genera $s+1$ permutazioni casuali indipendenti del dataset di training:

- **$\sigma_1, \ldots, \sigma_s$**: Usate per la valutazione degli split che definiscono le strutture degli alberi (nodi interni)
- **$\sigma_0$**: Usata per scegliere i valori delle foglie $b_j$ degli alberi ottenuti

**Perché multiple permutazioni?**

Per esempi con "storia" breve in una data permutazione:
- Le TS hanno alta varianza
- Le predizioni usate dall'ordered boosting ($M_{\sigma(i)-1}(\mathbf{x}_i)$) hanno alta varianza

Usare solo una permutazione aumenterebbe la varianza delle predizioni finali del modello.

**Più permutazioni** permettono di ridurre questo effetto, come confermato dagli esperimenti.

### Oblivious Decision Trees

CatBoost usa come base predictors gli **oblivious decision trees** (anche chiamati decision tables).

**Definizione**: Un albero si dice "oblivious" se lo stesso criterio di splitting è usato attraverso un intero livello dell'albero.

**Proprietà**:
- Alberi bilanciati
- Meno inclini all'overfitting
- Permettono di accelerare significativamente l'esecuzione al tempo di test

### Procedura di Costruzione di un Albero

#### Modalità Ordered

Durante il processo di apprendimento, manteniamo i **modelli di supporto** $M_{r,j}$, dove:
- $M_{r,j}(i)$ è la predizione corrente per il $i$-esimo esempio
- Basata sui primi $j$ esempi nella permutazione $\sigma_r$

**Ad ogni iterazione $t$**:

1. **Selezione permutazione**: Campioniamo una permutazione casuale $\sigma_r$ da $\{\sigma_1, \ldots, \sigma_s\}$

2. **Calcolo TS**: Per le feature categoriche, tutte le TS sono calcolate secondo questa permutazione

3. **Calcolo gradienti**: Basandoci su $M_{r,j}(i)$, calcoliamo i gradienti corrispondenti:
   $g_{r,j}(i) = \frac{\partial L(y_i, s)}{\partial s}\bigg|_{s=M_{r,j}(i)}$

4. **Costruzione albero**: Durante la costruzione dell'albero, approssimiamo il gradiente $G$ in termini di similarità del coseno $\cos(\cdot, \cdot)$, dove per ogni esempio $i$ prendiamo il gradiente $g_{r,\sigma(i)-1}(i)$ (basato solo sugli esempi precedenti in $\sigma_r$)

5. **Valutazione split candidati**: Il valore della foglia $\Delta(i)$ per l'esempio $i$ è ottenuto individualmente facendo la media dei gradienti $g_{r,\sigma_r(i)-1}$ degli esempi precedenti $p$ che si trovano nella stessa foglia $\text{leaf}_r(i)$ a cui appartiene l'esempio $i$

6. **Struttura comune**: Quando la struttura dell'albero $T_t$ (sequenza di attributi di splitting) è costruita, la usiamo per boostare **tutti** i modelli $M_{r',j}$. Una struttura comune è usata per tutti i modelli, ma l'albero è aggiunto a diversi $M_{r',j}$ con diversi insiemi di valori delle foglie

#### Modalità Plain

Funziona similmente a una procedura GBDT standard, ma:
- Se le feature categoriche sono presenti, mantiene $s$ modelli di supporto $M_r$ corrispondenti alle TS basate su $\sigma_1, \ldots, \sigma_s$

### Scelta dei Valori delle Foglie

Dati tutti gli alberi costruiti, i valori delle foglie del modello finale $F$ sono calcolati dalla procedura standard di gradient boosting, ugualmente per entrambe le modalità:

1. Gli esempi di training $i$ sono associati alle foglie $\text{leaf}_0(i)$
2. Si usa la permutazione $\sigma_0$ per calcolare le TS
3. Quando il modello finale $F$ è applicato a un nuovo esempio al tempo di test, si usano TS calcolate sull'intero training data

### Trick di Complessità

Nell'implementazione pratica, si usa un trick importante che riduce significativamente la complessità computazionale.

**Invece di** $O(s \cdot n^2)$ valori $M_{r,j}(i)$, memorizziamo e aggiorniamo solo i valori:

$M'_{r,j}(i) := M_{r,2^j}(i)$

per:
- $j = 1, \ldots, \lceil \log_2 n \rceil$
- Tutti gli $i$ con $\sigma_r(i) \leq 2^{j+1}$

**Riduzione**: Questo riduce il numero di predizioni di supporto mantenute a $O(s \cdot n)$.

### Complessità Computazionale

La complessità computazionale per iterazione è mostrata nella seguente tabella:

| Procedura | Complessità |
|-----------|-------------|
| CalcGradient | $O(s \cdot n)$ |
| Build $T$ | $O(\|C\| \cdot n)$ |
| Calc values $b_j^t$ | $O(n)$ |
| Update $M$ | $O(s \cdot n)$ |
| Calc ordered TS | $O(N_{TS,t} \cdot n)$ |

dove:
- $C$ è l'insieme di split candidati da considerare all'iterazione data
- $N_{TS,t}$ è il numero di TS da calcolare all'iterazione $t$

**Conclusione**: L'implementazione di ordered boosting con alberi di decisione ha la **stessa complessità asintotica** del GBDT standard con TS ordinate.

Rispetto ad altri tipi di TS, le ordered TS rallentano di un fattore $s$ le procedure: CalcGradient, aggiornamento modelli $M$, e calcolo delle TS.

### Combinazioni di Feature

Un dettaglio importante di CatBoost è l'uso di **combinazioni di feature categoriche** come feature categoriche aggiuntive.

**Motivazione**: Catturano dipendenze di ordine superiore, come l'informazione congiunta di user ID e topic dell'annuncio nel task di ad click prediction.

**Problema**: Il numero di combinazioni possibili cresce esponenzialmente con il numero di feature categoriche nel dataset.

**Soluzione greedy di CatBoost**: Per ogni split di un albero, CatBoost combina (concatena):
- Tutte le feature categoriche (e loro combinazioni) già usate per split precedenti nell'albero corrente
- Con tutte le feature categoriche nel dataset

Le combinazioni sono convertite in TS al volo.

### Bayesian Bootstrap

Prima di addestrare un albero, si assegna un peso $w_i = a_i^t$ a ogni esempio $i$, dove $a_i^t$ sono generati secondo la procedura di Bayesian bootstrap.

Questi pesi sono usati come moltiplicatori per i gradienti quando si calcolano:
- $\Delta(i)$
- Le componenti del vettore $\Delta - G$ per definire $\text{loss}(T_c)$

**Motivazione**: Anche se il subsampling da solo non può evitare completamente il problema di prediction shift, si è dimostrato efficace nella pratica.

### Gestione degli Esempi Iniziali

Per esempi $i$ con valori piccoli di $\sigma_r(i)$, la varianza di $g_{r,\sigma_r(i)-1}(i)$ può essere alta.

**Soluzione**: Si scartano i $\Delta(i)$ dall'inizio della permutazione quando si calcola la loss. Specificamente, si eliminano le componenti corrispondenti dei vettori $G$ e $\Delta$ quando si calcola la similarità del coseno.

---

## Risultati Sperimentali

### Setup Sperimentale

Gli esperimenti confrontano CatBoost con le librerie open-source più popolari:
- **XGBoost**
- **LightGBM**

Su vari task di machine learning ben noti.

#### Preprocessing

Per XGBoost, LightGBM e il raw setting di CatBoost:
- Le feature categoriche sono preprocessate calcolando ordered TS basate su una permutazione casuale degli esempi del training set
- I valori risultanti delle TS sono considerati feature numeriche

#### Tuning dei Parametri

Si usa l'algoritmo di ottimizzazione sequenziale **Tree Parzen Estimator** (implementato nella libreria Hyperopt) con 50 step, minimizzando la logloss.

#### Train-Test Split

- 80% training set
- 20% test set
- Cross-validazione a 5 fold per il tuning dei parametri

### Dataset Utilizzati

Gli esperimenti sono stati condotti su 9 dataset:

1. **Adult** (48,842 istanze, 15 feature): Previsione se una persona guadagna più di 50K all'anno
2. **Amazon** (32,769 istanze, 10 feature): Kaggle Amazon Employee challenge
3. **Click Prediction** (399,482 istanze, 12 feature): Previsione click su annunci (KDD Cup 2012)
4. **Epsilon** (400,000 istanze, 2000 feature): PASCAL Challenge 2008
5. **KDD Appetency** (50,000 istanze, 231 feature)
6. **KDD Churn** (50,000 istanze, 231 feature)
7. **KDD Internet** (10,108 istanze, 69 feature): Versione binarizzata
8. **KDD Upselling** (50,000 istanze, 231 feature)
9. **Kick Prediction** (72,983 istanze, 36 feature): Kaggle "Don't Get Kicked!" challenge

### Risultati: Confronto con Baseline

CatBoost (modalità Ordered) supera gli altri algoritmi su **tutti** i dataset considerati.

**Significatività statistica**: Eccetto tre dataset (Appetency, Churn e Upselling), i miglioramenti sono statisticamente significativi con p-value $\ll 0.01$ (paired one-tailed t-test).

#### Risultati Dettagliati (Logloss / Zero-one loss)

| Dataset | CatBoost | LightGBM | XGBoost |
|---------|----------|----------|---------|
| Adult | **0.270 / 0.127** | +2.4% / +1.9% | +2.2% / +1.0% |
| Amazon | **0.139 / 0.044** | +17% / +21% | +17% / +21% |
| Click | **0.392 / 0.156** | +1.2% / +1.2% | +1.2% / +1.2% |
| Epsilon | **0.265 / 0.109** | +1.5% / +4.1% | +11% / +12% |
| Appetency | **0.072 / 0.018** | +0.4% / +0.2% | +0.4% / +0.7% |
| Churn | **0.232 / 0.072** | +0.1% / +0.6% | +0.5% / +1.6% |
| Internet | **0.209 / 0.094** | +6.8% / +8.6% | +7.9% / +8.0% |
| Upselling | **0.166 / 0.049** | +0.3% / +0.1% | +0.04% / +0.3% |
| Kick | **0.286 / 0.095** | +3.5% / +4.4% | +3.2% / +4.1% |

I valori nelle colonne LightGBM e XGBoost rappresentano l'aumento percentuale relativo rispetto a CatBoost.

**Miglioramenti particolarmente significativi**:
- Amazon: +17% rispetto ai baseline
- Internet: +6.8%-7.9% rispetto ai baseline
- Epsilon: +11% rispetto a XGBoost

### Confronto tra Modalità Ordered e Plain

#### Risultati Generali

La modalità Ordered è particolarmente utile su **dataset piccoli**.

Il beneficio maggiore dall'Ordered si osserva su:
- **Adult**: 40K esempi di training
- **Internet**: < 40K esempi di training

#### Risultati Dettagliati (Plain vs Ordered)

| Dataset | Plain Logloss | Plain Zero-one | Variazione |
|---------|---------------|----------------|------------|
| Adult | 0.272 | 0.127 | +1.1% / -0.1% |
| Amazon | 0.139 | 0.044 | -0.6% / -1.5% |
| Click | 0.392 | 0.156 | -0.05% / +0.19% |
| Epsilon | 0.266 | 0.110 | +0.6% / +0.9% |
| Appetency | 0.072 | 0.018 | +0.5% / +1.5% |
| Churn | 0.232 | 0.072 | -0.06% / -0.17% |
| Internet | 0.217 | 0.099 | +3.9% / +5.4% |
| Upselling | 0.166 | 0.049 | +0.1% / +0.4% |
| Kick | 0.285 | 0.095 | -0.2% / -0.1% |

#### Esperimento con Dataset Filtrati

Per validare l'ipotesi che il bias sia più grande per dataset più piccoli:
- Si addestra CatBoost in modalità Ordered e Plain su dataset filtrati casualmente
- Si confrontano le loss ottenute

**Risultato**: Per dataset più piccoli, la performance relativa della modalità Plain peggiora, confermando l'ipotesi che il bias (secondo il Teorema 1) sia più grande per dataset più piccoli.

### Analisi delle Target Statistics

Confronto di diverse TS introdotte nella sezione corrispondente, tutte implementate come opzioni di CatBoost in modalità Ordered:

#### Risultati (Variazione relativa rispetto a Ordered TS)

| Dataset | Greedy | Holdout | Leave-one-out |
|---------|--------|---------|---------------|
| Adult | +1.1% / +0.8% | +2.1% / +2.0% | +5.5% / +3.7% |
| Amazon | +40% / +32% | +8.3% / +8.3% | +4.5% / +5.6% |
| Click | +13% / +6.7% | +1.5% / +0.5% | +2.7% / +0.9% |
| Appetency | +24% / +0.7% | +1.6% / -0.5% | +8.5% / +0.7% |
| Churn | +12% / +2.1% | +0.9% / +1.3% | +1.6% / +1.8% |
| Internet | +33% / +22% | +2.6% / +1.8% | +27% / +19% |
| Upselling | +57% / +50% | +1.6% / +0.9% | +3.9% / +2.9% |
| Kick | +22% / +28% | +1.3% / +0.32% | +3.7% / +3.3% |

**Osservazioni**:

1. **Ordered TS** superano significativamente tutti gli altri approcci
2. **Holdout TS** è il migliore tra i baseline per la maggior parte dei dataset (non soffre di conditional shift - P1), ma è peggiore di ordered TS (meno uso efficace dei dati - P2)
3. **Leave-one-out** è solitamente migliore di greedy TS, ma può essere molto peggiore su alcuni dataset (es. Adult)
4. **Greedy TS** soffre sia delle categorie a bassa frequenza che di quelle ad alta frequenza

### Combinazioni di Feature

Cambiare il numero $c_{max}$ di feature consentite per le combinazioni:
- Da 1 a 2: **miglioramento medio di 1.86%** (fino a 11.3%)
- Da 1 a 3: **miglioramento di 2.04%**
- Ulteriore aumento di $c_{max}$: non influenza significativamente la performance

### Numero di Permutazioni

Effetto del numero $s$ di permutazioni sulla performance:
- $s = 3$: diminuzione media di logloss di **0.19%** rispetto a $s = 1$
- $s = 9$: diminuzione media di logloss di **0.38%** rispetto a $s = 1$

**Conclusione**: Aumentare $s$ diminuisce leggermente la logloss, ma i guadagni diminuiscono.

### Tempi di Esecuzione

Confronto dei tempi su dataset Epsilon:

| Algoritmo | Tempo per albero |
|-----------|------------------|
| **CatBoost Plain** | **1.1 s** |
| CatBoost Ordered | 1.9 s |
| XGBoost | 3.9 s |
| **LightGBM** | **1.1 s** |

**Osservazioni**:
- CatBoost Plain e LightGBM sono i più veloci
- Ordered mode è circa **1.7 volte più lento** (come atteso)
- CatBoost ha anche un'implementazione GPU altamente efficiente (dettagli su GitHub)

---

## Conclusioni

### Contributi Principali

1. **Identificazione e analisi del problema di prediction shift** presente in tutte le implementazioni esistenti di gradient boosting

2. **Proposta di una soluzione generale**: ordered boosting con ordered TS

3. **Implementazione pratica**: CatBoost, una nuova libreria di gradient boosting

4. **Risultati empirici**: CatBoost supera i principali pacchetti GBDT e porta a nuovi risultati state-of-the-art su benchmark comuni

### Innovazioni Algoritmiche

#### Ordered Target Statistics

- Evitano il conditional shift (proprietà P1)
- Usano efficientemente tutti i dati di training (proprietà P2)
- Basate sul principio di ordinamento (permutazioni casuali)
- Superiori a greedy TS, holdout TS, e leave-one-out TS

#### Ordered Boosting

- Elimina il prediction shift causato da target leakage
- Mantiene modelli di supporto multipli
- Usa permutazioni casuali per simulare un "tempo artificiale"
- Implementazione efficiente con complessità $O(s \cdot n)$ invece di $O(n^2)$

#### Gestione delle Feature Categoriche

- Conversione diretta a target statistics numeriche
- Combinazioni di feature per catturare interazioni di ordine superiore
- Efficiente per feature ad alta cardinalità

### Vantaggi Pratici

1. **Performance superiore**: Miglioramenti significativi su tutti i dataset testati
2. **Robustezza**: Particolarmente efficace su dataset piccoli
3. **Efficienza**: Complessità computazionale comparabile ai metodi standard
4. **Flessibilità**: Due modalità (Plain e Ordered) per diversi scenari
5. **Gestione nativa delle categoriche**: Nessun preprocessing manuale necessario

### Aspetti Teorici Rilevanti

Il paper fornisce:
- **Analisi formale** del prediction shift (Teorema 1)
- **Dimostrazione matematica** del bias introdotto dal riutilizzo dei dati
- **Garanzie teoriche** che ordered boosting elimina questo bias
- **Spiegazione** del perché permutazioni identiche $\sigma_{cat} = \sigma_{boost}$ sono necessarie

### Limitazioni e Considerazioni

1. **Ordered mode**: Circa 1.7 volte più lento di Plain mode
2. **Permutazioni multiple**: Richiedono più memoria ma riducono la varianza
3. **Trade-off**: Bilanciamento tra accuratezza e velocità

### Impatto

CatBoost rappresenta un avanzamento significativo nel gradient boosting:
- Risolve problemi teorici fondamentali precedentemente non riconosciuti
- Fornisce risultati empirici superiori
- È open-source e ampiamente utilizzato nella pratica

L'identificazione e risoluzione del prediction shift è un contributo importante per la comunità di machine learning, con implicazioni che vanno oltre CatBoost stesso.

---

## Riferimenti e Approfondimenti

### Concetti Chiave da Ricordare

1. **Target Leakage**: Quando informazioni sul target "fuoriescono" nel processo di training, causando overfitting
2. **Conditional Shift**: Quando $P(X|Y)$ differisce tra training e test
3. **Prediction Shift**: Quando $P(F(\mathbf{x})|\mathbf{x})$ differisce tra training e test
4. **Ordering Principle**: Usare permutazioni casuali per simulare un ordine temporale
5. **Oblivious Trees**: Alberi che usano lo stesso split criterion su ogni livello

### Perché CatBoost Funziona

La combinazione di:
- Ordered boosting (elimina prediction shift)
- Ordered target statistics (elimina conditional shift)
- Feature combinations (cattura interazioni complesse)
- Oblivious trees (riduce overfitting, accelera inferenza)
- Permutazioni multiple (riduce varianza)

porta a un algoritmo che è sia teoricamente fondato che empiricamente superiore.

### Applicazioni Pratiche

CatBoost è particolarmente adatto per:
- **Click prediction**: Feature categoriche come user ID, region, ad
- **Recommendation systems**: User-item interactions
- **Fraud detection**: Transazioni con molte feature categoriche
- **Ranking problems**: Web search, e-commerce
- **Qualsiasi problema con feature categoriche ad alta cardinalità**
