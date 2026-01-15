# Metriche di Valutazione per Classificazione in Machine Learning

## Indice

1. [Fondamenti Teorici](#1-fondamenti-teorici)
2. [Matrice di Confusione](#2-matrice-di-confusione)
3. [Teoria delle Decisioni e Loss Functions](#3-teoria-delle-decisioni-e-loss-functions)
4. [Metriche Fondamentali](#4-metriche-fondamentali)
5. [Curve ROC e Analisi delle Performance](#5-curve-roc-e-analisi-delle-performance)
6. [Curve Precision-Recall](#6-curve-precision-recall)
7. [Metriche Avanzate e Robuste](#7-metriche-avanzate-e-robuste)
8. [Valutazione Probabilistica e Calibrazione](#8-valutazione-probabilistica-e-calibrazione)
9. [Classificazione Multi-Classe](#9-classificazione-multi-classe)
10. [Guida Pratica alla Scelta delle Metriche](#10-guida-pratica-alla-scelta-delle-metriche)

## 1. Fondamenti Teorici

### 1.1 Introduzione

La valutazione di modelli di classificazione è un problema fondamentale nel machine learning. Non esiste una singola metrica universale: la scelta dipende dal problema specifico, dalla distribuzione dei dati, e dai costi associati ai diversi tipi di errore.

Questo documento presenta una trattazione rigorosa e completa delle principali metriche di valutazione, partendo dai fondamenti della teoria delle decisioni bayesiane fino alle applicazioni pratiche.

### 1.2 Il Framework della Teoria delle Decisioni Bayesiane

Nel contesto della teoria delle decisioni, un problema di classificazione può essere formalizzato come un **gioco contro la natura**:

1. **La natura** sceglie uno stato (label) $y \in \mathcal{Y}$, sconosciuto a noi
2. **La natura** genera un'osservazione $x \in \mathcal{X}$, che possiamo osservare
3. **Noi** scegliamo un'azione $a$ da uno spazio di azioni $\mathcal{A}$
4. **Incorriamo** in una perdita $L(y, a)$ che misura la discrepanza tra stato reale e azione scelta

#### Objective: Decision Rule Ottimale

L'obiettivo è trovare una **decision rule** (o **policy**) $\delta: \mathcal{X} \rightarrow \mathcal{A}$ che minimizzi la perdita attesa:

$$\delta^*(x) = \arg\min_{a \in \mathcal{A}} \mathbb{E}_{p(y|x)}[L(y, a)]$$

Nell'approccio **bayesiano**, dopo aver osservato $x$, l'azione ottimale è quella che minimizza la **perdita attesa a posteriori** (posterior expected loss):

$$\rho(a|x) = \mathbb{E}_{p(y|x)}[L(y, a)] = \sum_{y \in \mathcal{Y}} L(y, a) \cdot p(y|x)$$

Quindi, il **Bayes estimator** (o **Bayes decision rule**) è:

$$\delta^*(x) = \arg\min_{a \in \mathcal{A}} \rho(a|x) = \arg\min_{a \in \mathcal{A}} \sum_{y \in \mathcal{Y}} L(y, a) \cdot p(y|x)$$

**Interpretazione intuitive**: Il Bayes estimator ci dice: "Data l'osservazione $x$, scegli l'azione che minimizza la perdita media che ti aspetti di subire, pesando ogni possibile stato reale $y$ per la sua probabilità a posteriori $p(y|x)$."

#### Principio di Utilità Attesa Massima

In economia, è più comune parlare di **funzione di utilità** $U(y, a) = -L(y, a)$. Il problema diventa:

$$\delta^*(x) = \arg\max_{a \in \mathcal{A}} \mathbb{E}_{p(y|x)}[U(y, a)]$$

Questo è il **principio di utilità attesa massima**, che costituisce la base del comportamento razionale in condizioni di incertezza.

### 1.3 Rischio e Generalizzazione

Il **rischio** (o rischio atteso) di una decision rule $\delta$ è la perdita media sulla distribuzione dei dati:

$$R(\delta) = \mathbb{E}_{(X,Y) \sim p(x,y)}[L(Y, \delta(X))] = \int_{\mathcal{X}} \int_{\mathcal{Y}} L(y, \delta(x)) \, p(x,y) \, dy \, dx$$

Dobbiamo distinguere tre concetti fondamentali:

**Rischio Empirico** (Training Risk):
$$\hat{R}_{\text{train}}(\delta) = \frac{1}{n} \sum_{i=1}^{n} L(y_i, \delta(x_i))$$

È la perdita media calcolata sul training set. Tende a **sottostimare** il vero rischio (overfitting).

**Rischio di Generalizzazione** (True Risk):
$$R_{\text{true}}(\delta) = \mathbb{E}_{(X,Y) \sim p_{\text{true}}(x,y)}[L(Y, \delta(X))]$$

È il vero rischio sulla distribuzione sottostante (sconosciuta). È quello che vogliamo davvero minimizzare.

**Rischio Empirico su Test Set**:
$$\hat{R}_{\text{test}}(\delta) = \frac{1}{m} \sum_{j=1}^{m} L(y_j^{\text{test}}, \delta(x_j^{\text{test}}))$$

È una stima non distorta di $R_{\text{true}}(\delta)$ se il test set è indipendente dal training.

**Principio Fondamentale**: Minimizziamo $\hat{R}_{\text{train}}(\delta)$ durante il training, ma valutiamo su $\hat{R}_{\text{test}}(\delta)$ per stimare $R_{\text{true}}(\delta)$.

## 2. Matrice di Confusione

### 2.1 Definizione e Struttura

La **matrice di confusione** (confusion matrix) è la struttura fondamentale per calcolare tutte le metriche di classificazione binaria. Essa organizza le predizioni in base alla classe reale e alla classe predetta.

Per un problema di classificazione binaria, con:
- $y = 1$: classe **positiva**
- $y = 0$: classe **negativa**

La matrice di confusione ha questa struttura:

|                        | **Predetto Positivo** ($\hat{y}=1$) | **Predetto Negativo** ($\hat{y}=0$) | **Totale** |
|------------------------|-------------------------------------|-------------------------------------|------------|
| **Reale Positivo** ($y=1$) | **TP** (True Positive)              | **FN** (False Negative)             | $P = TP + FN$ |
| **Reale Negativo** ($y=0$) | **FP** (False Positive)             | **TN** (True Negative)              | $N = TN + FP$ |
| **Totale**                 | $P^* = TP + FP$                     | $N^* = TN + FN$                     | $n = P + N$ |

### 2.2 Definizioni Rigorose

Dato un dataset di $n$ esempi $\{(x_i, y_i)\}_{i=1}^n$ e un classificatore che produce predizioni $\hat{y}_i$, definiamo:

**True Positive (TP)**:
$$TP = |\{i : y_i = 1 \land \hat{y}_i = 1\}|$$
Numero di istanze positive correttamente classificate come positive.

**True Negative (TN)**:
$$TN = |\{i : y_i = 0 \land \hat{y}_i = 0\}|$$
Numero di istanze negative correttamente classificate come negative.

**False Positive (FP)**:
$$FP = |\{i : y_i = 0 \land \hat{y}_i = 1\}|$$
Numero di istanze negative erroneamente classificate come positive (**Errore di Tipo I**).

**False Negative (FN)**:
$$FN = |\{i : y_i = 1 \land \hat{y}_i = 0\}|$$
Numero di istanze positive erroneamente classificate come negative (**Errore di Tipo II**).

**Mnemonico**: La prima lettera (T/F) indica se la predizione è corretta (True) o errata (False). La seconda lettera (P/N) indica cosa ha predetto il classificatore.

### 2.3 Relazioni Fondamentali

Dalla matrice di confusione derivano identità fondamentali:

**Totale esempi**:
$$n = TP + TN + FP + FN$$

**Esempi positivi reali**:
$$P = TP + FN$$

**Esempi negativi reali**:
$$N = TN + FP$$

**Esempi predetti come positivi**:
$$P^* = TP + FP$$

**Esempi predetti come negativi**:
$$N^* = TN + FN$$

**Prevalenza** (proporzione di positivi):
$$\pi = \frac{P}{n} = \frac{TP + FN}{n}$$

### 2.4 Interpretazione Probabilistica

Possiamo interpretare i conteggi della matrice di confusione in termini probabilistici. Definiamo:

**Ipotesi**:
- $H_0$: L'istanza appartiene alla classe negativa ($y=0$)
- $H_1$: L'istanza appartiene alla classe positiva ($y=1$)

**Decisioni**:
- $D_0$: Classificare come negativo ($\hat{y}=0$)
- $D_1$: Classificare come positivo ($\hat{y}=1$)

Allora possiamo scrivere le probabilità condizionate:

**True Positive Rate (TPR)**:
$$\text{TPR} = P(D_1 | H_1) = P(\hat{y}=1 | y=1) = \frac{TP}{P}$$
Probabilità di classificare correttamente un positivo.

**False Positive Rate (FPR)**:
$$\text{FPR} = P(D_1 | H_0) = P(\hat{y}=1 | y=0) = \frac{FP}{N}$$
Probabilità di classificare erroneamente un negativo come positivo (Errore di Tipo I).

**True Negative Rate (TNR)**:
$$\text{TNR} = P(D_0 | H_0) = P(\hat{y}=0 | y=0) = \frac{TN}{N}$$
Probabilità di classificare correttamente un negativo.

**False Negative Rate (FNR)**:
$$\text{FNR} = P(D_0 | H_1) = P(\hat{y}=0 | y=1) = \frac{FN}{P}$$
Probabilità di classificare erroneamente un positivo come negativo (Errore di Tipo II).

**Relazioni complementari**:
$$\text{TPR} + \text{FNR} = 1$$
$$\text{TNR} + \text{FPR} = 1$$

### 2.5 Esempio: Rilevamento di Malattie della Tiroide

Consideriamo il problema di rilevare malattie della tiroide usando un dataset con 3428 pazienti nel test set, di cui 250 hanno una malattia tiroidea.

**Prima configurazione** (soglia di default $\tau = 0.5$):

|                    | Predetto Normale | Predetto Malato | Totale |
|--------------------|------------------|-----------------|--------|
| Realmente Normale  | 3177             | 1               | 3178   |
| Realmente Malato   | 237              | 13              | 250    |
| Totale             | 3414             | 14              | 3428   |

Analisi:
- **Accuracy**: $(3177 + 13) / 3428 = 93.1\%$ (sembra buona!)
- **Recall**: $13 / 250 = 5.2\%$ (pessimo! Perdiamo il 95% dei malati)
- **Precision**: $13 / 14 = 92.9\%$ (alta, ma poche predizioni positive)

Questo classificatore è **praticamente inutile**: un modello "dummy" che predice sempre "normale" otterrebbe accuracy del $92.7\%$, quasi identica!

**Seconda configurazione** (soglia abbassata a $\tau = 0.15$):

|                    | Predetto Normale | Predetto Malato | Totale |
|--------------------|------------------|-----------------|--------|
| Realmente Normale  | 3067             | 111             | 3178   |
| Realmente Malato   | 165              | 85              | 250    |
| Totale             | 3232             | 196             | 3428   |

Analisi:
- **Accuracy**: $(3067 + 85) / 3428 = 91.9\%$ (leggermente diminuita)
- **Recall**: $85 / 250 = 34\%$ (migliorato significativamente!)
- **Precision**: $85 / 196 = 43.4\%$ (diminuita, ma accettabile)

**Conclusione**: Il secondo modello è probabilmente più utile in pratica, nonostante l'accuracy leggermente inferiore. Questo esempio illustra perché l'accuracy da sola è insufficiente per problemi sbilanciati.

## 3. Teoria delle Decisioni e Loss Functions

### 3.1 Loss Functions e Bayes Estimators

La scelta della **loss function** $L(y, a)$ determina quale azione è ottimale. Diverse loss functions portano a diversi estimatori ottimali.

#### 3.1.1 0-1 Loss e Stima MAP

La **0-1 loss** è la più semplice e naturale:

$$L_{0-1}(y, a) = \mathbb{I}(y \neq a) = \begin{cases} 0 & \text{se } a = y \\ 1 & \text{se } a \neq y \end{cases}$$

**Interpretazione**: Penalizziamo ugualmente tutti gli errori, senza distinzione di tipo.

**Teorema 3.1** (Bayes Estimator per 0-1 Loss):
*La 0-1 loss è minimizzata dalla stima MAP (Maximum A Posteriori).*

**Dimostrazione**:

La perdita attesa a posteriori per l'azione $a$ è:

$$\rho(a|x) = \sum_{y \in \mathcal{Y}} L_{0-1}(y,a) \cdot p(y|x) = \sum_{y \neq a} p(y|x) = 1 - p(a|x)$$

Per minimizzare $\rho(a|x)$, dobbiamo massimizzare $p(a|x)$:

$$\delta^*(x) = \arg\min_a \rho(a|x) = \arg\max_a p(a|x) = \arg\max_{y \in \mathcal{Y}} p(y|x)$$

che è esattamente la **stima MAP**. $\square$

**Corollario**: Per classificazione binaria con 0-1 loss, la regola ottimale è:

$$\hat{y} = \begin{cases} 1 & \text{se } p(y=1|x) > 0.5 \\ 0 & \text{altrimenti} \end{cases}$$

#### 3.1.2 Loss Asimmetrica e Costi Differenziati

Nella pratica, i diversi tipi di errore hanno spesso costi diversi. Rappresentiamo questo con una **matrice di loss**:

|            | $\hat{y}=1$ | $\hat{y}=0$ |
|------------|------------|------------|
| $y=1$      | $0$        | $L_{FN}$   |
| $y=0$      | $L_{FP}$   | $0$        |

dove:
- $L_{FN}$: costo di un False Negative (mancata rilevazione)
- $L_{FP}$: costo di un False Positive (falso allarme)

**Teorema 3.2** (Regola di Decisione Ottimale con Costi Asimmetrici):
*Sotto la matrice di loss asimmetrica, dovremmo classificare come positivo se e solo se:*

$$\frac{p(y=1|x)}{p(y=0|x)} > \frac{L_{FP}}{L_{FN}}$$

**Dimostrazione**:

Le perdite attese per le due azioni sono:

$$\rho(\hat{y}=0|x) = L_{FN} \cdot p(y=1|x) + 0 \cdot p(y=0|x) = L_{FN} \cdot p(y=1|x)$$

$$\rho(\hat{y}=1|x) = 0 \cdot p(y=1|x) + L_{FP} \cdot p(y=0|x) = L_{FP} \cdot p(y=0|x)$$

Scegliamo $\hat{y}=1$ quando $\rho(\hat{y}=1|x) < \rho(\hat{y}=0|x)$:

$$L_{FP} \cdot p(y=0|x) < L_{FN} \cdot p(y=1|x)$$

Dividendo entrambi i lati per $p(y=0|x)$ e $L_{FN}$:

$$\frac{L_{FP}}{L_{FN}} < \frac{p(y=1|x)}{p(y=0|x)}$$

$\square$

**Corollario (Soglia Ottimale)**:
Se $L_{FN} = c \cdot L_{FP}$ con $c > 0$, la regola diventa: classificare come positivo se $p(y=1|x) > \tau^*$ dove:

$$\tau^* = \frac{1}{1 + c} = \frac{L_{FP}}{L_{FP} + L_{FN}}$$

**Esempi numerici**:

1. **Screening medico**: $L_{FN} = 100$, $L_{FP} = 1$ (la mancata diagnosi è 100 volte più grave)
   $$\tau^* = \frac{1}{101} \approx 0.01$$
   Soglia molto bassa → massimizziamo il recall.

2. **Anti-spam**: $L_{FN} = 1$, $L_{FP} = 10$ (eliminare email legittima è 10 volte peggio)
   $$\tau^* = \frac{10}{11} \approx 0.91$$
   Soglia alta → massimizziamo la precision.

#### 3.1.3 Reject Option

In applicazioni ad alto rischio (medicina, finanza), può essere preferibile **rifiutare** di classificare esempi incerti piuttosto che rischiare errori gravi.

Formalizziamo l'azione di rifiuto come $a = \text{reject}$ con costo $\lambda_r$, mentre gli errori di classificazione hanno costo $\lambda_s$ (substitution error).

**Teorema 3.3** (Regola con Reject Option):
*L'azione ottimale è:*

$$\delta^*(x) = \begin{cases}
\arg\max_c p(y=c|x) & \text{se } \max_c p(y=c|x) \geq 1 - \frac{\lambda_r}{\lambda_s} \\
\text{reject} & \text{altrimenti}
\end{cases}$$

**Dimostrazione** (sketch):

Il costo atteso per classificare nella classe $c$ è:

$$\rho(\hat{y}=c|x) = \lambda_s \cdot P(\text{errore}|x) = \lambda_s \cdot (1 - p(y=c|x))$$

Il costo per rifiutare è costante: $\rho(\text{reject}|x) = \lambda_r$.

Conviene classificare se:

$$\lambda_s \cdot (1 - p(y=c|x)) < \lambda_r$$

$$p(y=c|x) > 1 - \frac{\lambda_r}{\lambda_s}$$

Scegliamo la classe con probabilità massima solo se supera questa soglia. $\square$

**Esempio**: Se $\lambda_s = 10$ (errore costa 10) e $\lambda_r = 2$ (rifiuto costa 2):
$$\text{Soglia} = 1 - \frac{2}{10} = 0.8$$

Rifiutiamo di classificare se $\max_c p(y=c|x) < 0.8$.

#### 3.1.4 Quadratic Loss e Posterior Mean

Per problemi di regressione o quando lavoriamo con probabilità, la **quadratic loss** (o squared error) è naturale:

$$L_2(y, a) = (y - a)^2$$

**Teorema 3.4** (Bayes Estimator per Quadratic Loss):
*La $\ell_2$ loss è minimizzata dalla media a posteriori.*

**Dimostrazione**:

La perdita attesa a posteriori è:

$$\rho(a|x) = \mathbb{E}[(y-a)^2|x] = \mathbb{E}[y^2|x] - 2a\mathbb{E}[y|x] + a^2$$

Deriviamo rispetto ad $a$ e poniamo uguale a zero:

$$\frac{\partial \rho(a|x)}{\partial a} = -2\mathbb{E}[y|x] + 2a = 0$$

$$\Rightarrow a^* = \mathbb{E}[y|x] = \int y \, p(y|x) \, dy$$

Questa è la **stima MMSE (Minimum Mean Squared Error)**. $\square$

**Applicazione in regressione**: Per un modello lineare $p(y|x,w) = \mathcal{N}(y|w^Tx, \sigma^2)$, l'estimatore MMSE è:

$$\hat{y}(x) = \mathbb{E}[y|x, \mathcal{D}] = x^T \mathbb{E}[w|\mathcal{D}]$$

Basta usare la media a posteriori dei parametri.

#### 3.1.5 Absolute Loss e Posterior Median

La **absolute loss** (o $\ell_1$ loss) è più robusta agli outlier:

$$L_1(y, a) = |y - a|$$

**Teorema 3.5** (Bayes Estimator per Absolute Loss):
*La $\ell_1$ loss è minimizzata dalla mediana a posteriori.*

**Dimostrazione**:

La perdita attesa è:

$$\rho(a|x) = \int |y-a| p(y|x) dy = \int_{-\infty}^{a} (a-y) p(y|x) dy + \int_{a}^{\infty} (y-a) p(y|x) dy$$

Deriviamo rispetto ad $a$. Usando la regola di Leibniz:

$$\frac{\partial \rho(a|x)}{\partial a} = \int_{-\infty}^{a} p(y|x) dy - \int_{a}^{\infty} p(y|x) dy$$

$$= P(y \leq a|x) - P(y > a|x)$$

Ponendo uguale a zero:

$$P(y \leq a|x) = P(y > a|x) = \frac{1}{2}$$

che è la definizione di **mediana**. $\square$

**Perché la $\ell_1$ è più robusta?** La $\ell_2$ loss penalizza quadraticamente le deviazioni, quindi un singolo outlier molto distante può dominare la loss. La $\ell_1$ penalizza linearmente, riducendo l'influenza degli outlier.

## 4. Metriche Fondamentali

### 4.1 Accuracy (Accuratezza)

L'**accuracy** è la metrica più semplice e intuitiva: misura la proporzione di predizioni corrette.

**Definizione**:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{TP + TN}{n}$$

**Interpretazione probabilistica**:
$$\text{Accuracy} = P(\hat{y} = y)$$
È la probabilità che una predizione casuale sia corretta.

**Proprietà**:
- **Range**: $[0, 1]$, dove $1$ indica predizioni perfette
- **Simmetrica** rispetto alle classi
- **Uguale peso** a errori positivi e negativi

**Limitazione critica: Dataset Sbilanciati**

Consideriamo un problema di fraud detection dove solo l'1% delle transazioni è fraudolenta ($\pi = 0.01$).

Un classificatore "dummy" che **predice sempre negativo** ottiene:
$$\text{Accuracy}_{\text{dummy}} = \frac{0 + 0.99n}{n} = 0.99$$

Questo sembra eccellente, ma il modello è completamente inutile! Non rileva nessuna frode.

**Teorema 4.1** (Lower Bound su Accuracy per Classificatore Dummy):
*Un classificatore che predice sempre la classe maggioritaria ottiene accuracy pari alla prevalenza della classe maggioritaria:*

$$\text{Accuracy}_{\text{majority}} = \max(\pi, 1-\pi)$$

**Conclusione**: L'accuracy è **inadeguata per dataset sbilanciati**. Dobbiamo usare metriche che distinguano tra i diversi tipi di errore.

### 4.2 Precision (Precisione)

La **precision** misura l'affidabilità delle predizioni positive.

**Definizione**:
$$\text{Precision} = \frac{TP}{TP + FP} = \frac{TP}{P^*}$$

**Interpretazione probabilistica**:
$$\text{Precision} = P(y=1|\hat{y}=1)$$
"Tra tutti i casi che ho predetto come positivi, qual è la probabilità che siano realmente positivi?"

**Interpretazione intuitiva**: "Quando il modello dice 'positivo', quanto possiamo fidarci?"

**Quando è critica**: Scenari dove i **falsi positivi sono costosi**:

1. **Spam detection**: Classificare email legittime come spam può far perdere comunicazioni importanti
2. **Diagnosi mediche aggressive**: Prescrivere chemioterapia a pazienti sani
3. **Raccomandazioni**: Raccomandare prodotti irrilevanti irrita l'utente
4. **Allerte di sicurezza**: Troppi falsi allarmi causano "alarm fatigue"

**Complemento - False Discovery Rate (FDR)**:
$$\text{FDR} = 1 - \text{Precision} = \frac{FP}{TP+FP}$$
Proporzione di "scoperte" che sono in realtà false.

**Dipendenza dalla Prevalenza**

La precision dipende fortemente dalla prevalenza $\pi = P(y=1)$. Usando il teorema di Bayes:

$$\text{Precision} = P(y=1|\hat{y}=1) = \frac{P(\hat{y}=1|y=1) \cdot P(y=1)}{P(\hat{y}=1)}$$

$$= \frac{\text{TPR} \cdot \pi}{\text{TPR} \cdot \pi + \text{FPR} \cdot (1-\pi)}$$

**Esempio**: Con TPR = 0.9, FPR = 0.1:
- Se $\pi = 0.5$: Precision = $\frac{0.9 \cdot 0.5}{0.9 \cdot 0.5 + 0.1 \cdot 0.5} = 0.9$
- Se $\pi = 0.01$: Precision = $\frac{0.9 \cdot 0.01}{0.9 \cdot 0.01 + 0.1 \cdot 0.99} \approx 0.08$

Con prevalenza bassa, anche un FPR modesto degrada drasticamente la precision!

### 4.3 Recall (Sensibilità, True Positive Rate)

Il **recall** misura la capacità di identificare i positivi.

**Definizione**:
$$\text{Recall} = \text{TPR} = \text{Sensitivity} = \frac{TP}{TP + FN} = \frac{TP}{P}$$

**Interpretazione probabilistica**:
$$\text{Recall} = P(\hat{y}=1|y=1)$$
"Tra tutti i casi realmente positivi, quale proporzione riesco a identificare?"

**Interpretazione intuitiva**: "Quanto è completa la mia rilevazione dei positivi?"

**Quando è critico**: Scenari dove i **falsi negativi sono costosi**:

1. **Screening medico**: Non diagnosticare un tumore è potenzialmente fatale
2. **Rilevamento frodi**: Non bloccare una transazione fraudolenta causa perdite economiche
3. **Sistemi di sicurezza**: Non rilevare un'intrusione compromette la sicurezza
4. **Information retrieval**: Non trovare documenti rilevanti limita l'utilità del sistema

**Complemento - False Negative Rate (FNR)**:
$$\text{FNR} = \text{Miss Rate} = 1 - \text{Recall} = \frac{FN}{TP+FN}$$
Proporzione di positivi che "perdiamo" (manchiamo di rilevare).

**Indipendenza dalla Prevalenza**

A differenza della precision, il recall **non dipende dalla prevalenza** perché è condizionato sulla classe reale:

$\text{Recall} = P(\hat{y}=1|y=1)$

Questa è una probabilità condizionata su $y=1$, che dipende solo da $p(x|y=1)$ e dalla soglia di decisione, non da $P(y=1)$.

### 4.4 Specificity (True Negative Rate)

La **specificity** è il "recall per la classe negativa".

**Definizione**:
$\text{Specificity} = \text{TNR} = \frac{TN}{TN + FP} = \frac{TN}{N}$

**Interpretazione probabilistica**:
$\text{Specificity} = P(\hat{y}=0|y=0)$
"Tra tutti i casi realmente negativi, quale proporzione riesco a identificare correttamente?"

**Relazione con FPR**:
$\text{FPR} = 1 - \text{Specificity} = \frac{FP}{FP + TN} = P(\hat{y}=1|y=0)$

Il FPR è la probabilità di **falso allarme** (Errore di Tipo I nel testing d'ipotesi).

**Importanza in medicina**: In test diagnostici, specificity alta significa pochi falsi positivi, riducendo ansia ingiustificata e procedure invasive non necessarie.

### 4.5 Trade-off Precision vs Recall

Precision e recall sono tipicamente in **trade-off**: migliorare una tende a peggiorare l'altra.

**Intuizione del trade-off**:

Consideriamo un classificatore probabilistico che produce $p(y=1|x)$ e una soglia $\tau$:

$\hat{y} = \begin{cases} 1 & \text{se } p(y=1|x) > \tau \\ 0 & \text{altrimenti} \end{cases}$

**Abbassando la soglia** $\tau$ (classifichiamo più casi come positivi):
- ✅ **Recall aumenta**: Catturiamo più veri positivi
- ❌ **Precision diminuisce**: Includiamo anche più falsi positivi

**Alzando la soglia** $\tau$ (siamo più selettivi):
- ✅ **Precision aumenta**: Solo predizioni molto confidenti
- ❌ **Recall diminuisce**: Perdiamo alcuni veri positivi "border-line"

**Analisi formale**:

Al variare di $\tau$:

$\tau \to 0: \quad \begin{cases} \text{Recall} \to 1 \\ \text{Precision} \to \pi \end{cases} \quad \text{(tutto positivo)}$

$\tau \to 1: \quad \begin{cases} \text{Recall} \to 0 \\ \text{Precision} \to 1 \end{cases} \quad \text{(tutto negativo)}$

**Esempio numerico**:

Dataset: 100 positivi, 900 negativi. Modello produce score da 0 a 1.

| Soglia $\tau$ | TP | FP | FN | Precision | Recall |
|---------------|----|----|----|-----------| -------|
| 0.9 | 10 | 5 | 90 | 0.67 | 0.10 |
| 0.7 | 40 | 50 | 60 | 0.44 | 0.40 |
| 0.5 | 70 | 200 | 30 | 0.26 | 0.70 |
| 0.3 | 90 | 500 | 10 | 0.15 | 0.90 |

Osserviamo chiaramente il trade-off: recall alta → precision bassa, e viceversa.

### 4.6 F-Scores: Armonizzare Precision e Recall

#### 4.6.1 F1-Score: Media Armonica

L'**F1-score** combina precision e recall in una singola metrica bilanciata.

**Definizione**:
$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$

**Perché la media armonica?**

La media armonica di due numeri $a$ e $b$ è:
$H(a,b) = \frac{2ab}{a+b}$

È più **severa** della media aritmetica quando i valori sono sbilanciati:
$H(a,b) \leq G(a,b) \leq A(a,b)$
dove $G$ è la media geometrica e $A$ la media aritmetica.

**Esempio illustrativo**:

| Precision | Recall | Media Aritmetica | F1 (Media Armonica) |
|-----------|--------|------------------|---------------------|
| 0.9 | 0.9 | 0.90 | 0.90 |
| 0.9 | 0.5 | 0.70 | 0.64 |
| 0.9 | 0.1 | 0.50 | 0.18 |
| 0.5 | 0.5 | 0.50 | 0.50 |

L'F1 **penalizza fortemente** sistemi con una metrica molto bassa, anche se l'altra è alta.

**Proprietà matematiche**:

1. **Range**: $F_1 \in [0, 1]$
2. **Massimo**: $F_1 = 1$ se e solo se $\text{Precision} = \text{Recall} = 1$
3. **Simmetria**: $F_1(P, R) = F_1(R, P)$
4. **Monotonia**: $F_1$ cresce se aumentiamo sia $P$ che $R$

**Derivazione alternativa**:

Possiamo scrivere:
$F_1 = \frac{1}{\frac{1}{2}\left(\frac{1}{P} + \frac{1}{R}\right)}$

L'F1 è la media armonica perché è l'inverso della media aritmetica degli inversi.

#### 4.6.2 F-Beta Score: Peso Asimmetrico

Il **F-beta score** generalizza l'F1 permettendo di pesare diversamente recall e precision.

**Definizione**:
$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$

**Interpretazione del parametro $\beta$**:

- $\beta < 1$: **Precision pesa di più** (enfasi su evitare falsi positivi)
- $\beta = 1$: **Peso uguale** (F1-score standard)
- $\beta > 1$: **Recall pesa di più** (enfasi su catturare tutti i positivi)

**Valori comuni**:

**$F_{0.5}$** (Precision vale il doppio):
$F_{0.5} = 1.25 \cdot \frac{P \cdot R}{0.25 \cdot P + R}$
Uso: Spam detection, dove falsi positivi sono molto costosi.

**$F_2$** (Recall vale il doppio):
$F_2 = 5 \cdot \frac{P \cdot R}{4 \cdot P + R}$
Uso: Screening medico, dove falsi negativi sono molto costosi.

**Derivazione del peso di $\beta$**:

Riscriviamo l'F-beta in forma estesa:
$F_\beta = \frac{(1+\beta^2) \cdot TP}{(1+\beta^2) \cdot TP + \beta^2 \cdot FN + FP}$

Notiamo che:
- I falsi negativi (FN) sono pesati per $\beta^2$
- I falsi positivi (FP) sono pesati per $1$

Quindi $\beta^2$ è il **rapporto di importanza** tra recall e precision:
$\beta^2 = \frac{\text{Importanza del Recall}}{\text{Importanza della Precision}}$

**Esempio numerico**:

Consideriamo tre modelli con diverse caratteristiche:

| Modello | Precision | Recall | F1 | F0.5 | F2 |
|---------|-----------|--------|----|----- |----|
| A | 0.90 | 0.50 | 0.64 | 0.75 | 0.56 |
| B | 0.50 | 0.90 | 0.64 | 0.56 | 0.75 |
| C | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 |

- **Modello A**: Alta precision, basso recall → F0.5 lo premia
- **Modello B**: Bassa precision, alto recall → F2 lo premia  
- **Modello C**: Bilanciato → performance costante su tutti gli F-score

**Scelta di $\beta$ in base al dominio**:

1. **Medicina (screening)**: $\beta = 2$ o superiore (priorità su recall)
2. **Spam filtering**: $\beta = 0.5$ (priorità su precision)
3. **Information retrieval**: $\beta = 1$ (bilanciamento)
4. **Fraud detection**: $\beta = 1.5$ - $2$ (leggermente sbilanciato verso recall)

## 5. Curve ROC e Analisi delle Performance

### 5.1 Introduzione alle Curve ROC

La **curva ROC** (Receiver Operating Characteristic) è uno strumento fondamentale per valutare classificatori binari indipendentemente dalla scelta della soglia.

**Contesto storico**: Le curve ROC furono sviluppate durante la Seconda Guerra Mondiale per analizzare segnali radar. Il nome "Receiver Operating Characteristic" deriva proprio dall'analisi dei ricevitori radio.

### 5.2 Costruzione della Curva ROC

**Definizione formale**:

Data una famiglia di classificatori parametrizzati da una soglia $\tau \in [0,1]$:
$\hat{y}(\tau) = \mathbb{I}(p(y=1|x) > \tau)$

La curva ROC è il grafico dei punti:
$\text{ROC}(\tau) = \big(\text{FPR}(\tau), \text{TPR}(\tau)\big)$

al variare di $\tau$ da 0 a 1.

**Coordinate**:
- **Asse X**: False Positive Rate = $\frac{FP}{N}$
- **Asse Y**: True Positive Rate = $\frac{TP}{P}$

**Algoritmo di costruzione**:

```
Input: Score s_i e label y_i per i = 1,...,n

1. Ordina gli esempi per score decrescente: s_1 ≥ s_2 ≥ ... ≥ s_n
2. Inizializza: TP = 0, FP = 0
3. Per ogni soglia τ (prendendo s_i come soglie):
   a. Se y_i = 1: TP++
   b. Se y_i = 0: FP++
   c. Calcola: TPR = TP/P, FPR = FP/N
   d. Aggiungi punto (FPR, TPR) alla curva
4. Collega i punti per formare la curva
```

### 5.3 Interpretazione e Punti Notevoli

**Punti estremi**:

$(0, 0)$: **Origine** - Soglia $\tau = 1$ → tutto classificato come negativo
- TP = 0, FP = 0
- Classifier "always negative"

$(1, 1)$: **Angolo in alto a destra** - Soglia $\tau = 0$ → tutto classificato come positivo
- TP = P, FP = N
- Classifier "always positive"

$(0, 1)$: **Angolo in alto a sinistra** - Classificatore perfetto
- TP = P, FP = 0
- Nessun errore

**Diagonale principale** $y = x$: Classificatore casuale
- Prediction random con $P(\hat{y}=1) = p$
- In media: $\text{TPR} = p$, $\text{FPR} = p$

**Interpretazione geometrica**:

- **Curve vicine all'angolo (0,1)**: Ottimo classificatore
  - Alto TPR con basso FPR
  - Buona separazione tra classi

- **Curve vicine alla diagonale**: Classificatore scarso
  - Nessun potere discriminante
  - Simile a indovinare casualmente

- **Curve sotto la diagonale**: Classificatore "invertito"
  - Performance peggiore del caso
  - Invertire le predizioni migliorerebbe il modello!

**Proprietà di monotonia**:

La curva ROC è **monotona crescente**: muovendoci lungo la curva da sinistra a destra (abbassando $\tau$), sia TPR che FPR aumentano (o restano costanti).

**Teorema 5.1** (Monotonia della Curva ROC):
*Per un classificatore con score $s(x)$, se $\tau_1 < \tau_2$, allora:*
$\text{FPR}(\tau_1) \geq \text{FPR}(\tau_2) \quad \text{e} \quad \text{TPR}(\tau_1) \geq \text{TPR}(\tau_2)$

**Dimostrazione**: Abbassando la soglia, classifichiamo più esempi come positivi, quindi sia TP che FP possono solo aumentare (o restare costanti). $\square$

### 5.4 Area Under the Curve (AUC-ROC)

L'**AUC** (Area Under the ROC Curve) è una metrica scalare che riassume la performance complessiva del classificatore.

**Definizione matematica**:
$\text{AUC} = \int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))$

dove $t$ varia lungo la curva (parametro di soglia).

**Range**: $\text{AUC} \in [0, 1]$

**Interpretazione dei valori**:

- $\text{AUC} = 1.0$: **Perfetto** - Separazione completa tra classi
- $\text{AUC} = 0.9$: **Eccellente** - Ottima discriminazione
- $\text{AUC} = 0.8$: **Buono** - Buona discriminazione
- $\text{AUC} = 0.7$: **Accettabile** - Discriminazione discreta
- $\text{AUC} = 0.5$: **Nullo** - Nessun potere discriminante (casuale)
- $\text{AUC} < 0.5$: **Invertito** - Performance peggiore del caso

**Teorema 5.2** (Interpretazione Probabilistica dell'AUC):
*L'AUC è la probabilità che un esempio positivo casuale abbia score maggiore di un esempio negativo casuale:*

$\text{AUC} = P(s(X_+) > s(X_-))$

dove $X_+ \sim p(x|y=1)$ e $X_- \sim p(x|y=0)$.

**Dimostrazione** (sketch):

Consideriamo tutti i possibili confronti tra un esempio positivo e uno negativo. Per ogni soglia $\tau$, contiamo:
- Quante coppie $(x_+, x_-)$ hanno $s(x_+) > \tau$ e $s(x_-) \leq \tau$

La curva ROC traccia esattamente questa proporzione. L'integrale accumula tutti questi confronti, dando la frazione totale di coppie ordinate correttamente.

Formalmente, l'AUC può essere calcolata come:
$\text{AUC} = \frac{1}{P \cdot N} \sum_{i: y_i=1} \sum_{j: y_j=0} \mathbb{I}(s_i > s_j)$

dove $P$ è il numero di positivi e $N$ di negativi. Questo è esattamente una stima di $P(s(X_+) > s(X_-))$. $\square$

**Corollario**: L'AUC equivale alla statistica U del test di Mann-Whitney-Wilcoxon:
$\text{AUC} = \frac{U}{P \cdot N}$

dove $U$ è la statistica U di Mann-Whitney.

**Calcolo pratico dell'AUC**:

**Metodo 1** (Regola del trapezio):
$\text{AUC} \approx \sum_{i=1}^{n-1} \frac{1}{2}(\text{TPR}_i + \text{TPR}_{i+1}) \cdot (\text{FPR}_{i+1} - \text{FPR}_i)$

**Metodo 2** (Conteggio di coppie concordanti):
$\text{AUC} = \frac{\#\{(i,j): y_i=1, y_j=0, s_i > s_j\}}{P \cdot N}$

### 5.5 Proprietà Fondamentali dell'AUC

**Proprietà 5.1** (Invarianza alla Scala):
L'AUC dipende solo dall'**ordinamento** degli score, non dai valori assoluti.

Se applichiamo una trasformazione monotona crescente $f$ agli score:
$\text{AUC}(f(s)) = \text{AUC}(s)$

**Implicazione**: Possiamo confrontare modelli che producono score su scale diverse (e.g., probabilità vs logit vs distance).

**Proprietà 5.2** (Robustezza allo Sbilanciamento):
L'AUC **non dipende dalla prevalenza** della classe positiva.

Se cambiamo la distribuzione di classe nel test set, l'AUC rimane invariata (a patto che $p(x|y)$ non cambi).

**Dimostrazione**: TPR e FPR sono entrambi condizionati su $y$:
$\text{TPR} = P(\hat{y}=1|y=1), \quad \text{FPR} = P(\hat{y}=1|y=0)$

Questi dipendono solo da $p(x|y=1)$, $p(x|y=0)$ e dalla soglia, non da $P(y)$. $\square$

**Proprietà 5.3** (Interpretazione come Ranking Metric):
L'AUC misura la qualità del **ranking** prodotto dal classificatore:
- Un buon ranking mette esempi positivi in cima
- AUC alta → la maggior parte dei positivi è rankata sopra i negativi

**Limitazioni dell'AUC**:

1. **Non fornisce informazioni sulla calibrazione**: Due modelli con stesso AUC possono avere probabilità molto diverse

2. **Aggregazione su tutte le soglie**: Può mascherare performance scarse in regioni critiche

3. **Ottimizza per ranking globale**: Può non essere ottimale se ci interessa solo una specifica regione operativa (e.g., basso FPR)

4. **Sensibilità ridotta**: Cambiamenti in regioni di bassa densità hanno stesso peso di regioni ad alta densità

### 5.6 Operating Points e Trade-offs

**Operating point**: Un punto specifico sulla curva ROC corrispondente a una soglia $\tau$.

**Scelta dell'operating point**:

La curva ROC mostra tutti i possibili trade-off, ma dobbiamo scegliere un punto operativo specifico basato su:

1. **Costi asimmetrici**: Se $c = L_{FN}/L_{FP}$, cerchiamo il punto che minimizza:
   $\text{Cost}(\tau) = c \cdot \text{FNR}(\tau) + \text{FPR}(\tau)$

2. **Vincoli operativi**: 
   - "FPR deve essere $\leq 0.05$" → scegli il punto con FPR massimo 0.05 e TPR massimo
   - "Recall deve essere $\geq 0.9$" → scegli il punto con TPR minimo 0.9 e FPR minimo

3. **Youden's index**: Massimizza la distanza dalla diagonale:
   $J = \text{TPR} - \text{FPR} = \text{Sensitivity} + \text{Specificity} - 1$
   Equivale a massimizzare l'informedness.

### 5.7 Equal Error Rate (EER)

L'**Equal Error Rate** è il punto sulla curva ROC dove:
$\text{FPR}(\tau^*) = \text{FNR}(\tau^*) = \text{EER}$

Equivalentemente, dove:
$\text{FPR}(\tau^*) = 1 - \text{TPR}(\tau^*)$

**Interpretazione geometrica**: Intersezione della curva ROC con la linea $y = 1 - x$.

**Proprietà**:
- Bilanciamento naturale tra i due tipi di errore
- Utile quando non abbiamo informazioni sui costi relativi
- EER basso indica performance migliore

**Calcolo**: Cercare la soglia dove $|\text{FPR} - \text{FNR}|$ è minimo.

### 5.8 Confronto tra Modelli con ROC

**Dominanza**: Il modello A **domina** il modello B se:
$\text{TPR}_A(\tau) \geq \text{TPR}_B(\tau) \quad \forall \text{FPR}(\tau)$

In altre parole, la curva ROC di A è sempre sopra (o coincide con) quella di B.

Se A domina B, allora certamente $\text{AUC}_A \geq \text{AUC}_B$.

**Curve che si intersecano**: Se le curve ROC si intersecano, nessun modello domina l'altro. La scelta dipende dalla regione operativa:
- Se operiamo a basso FPR (alta specificità), scegliamo il modello migliore in quella regione
- Se operiamo ad alto TPR (alta sensibilità), scegliamo il modello migliore in quella regione

**Esempio**:
- Modello A: Migliore per FPR < 0.1 (applicazioni dove FP sono molto costosi)
- Modello B: Migliore per FPR > 0.1 (applicazioni dove vogliamo alto recall)

## 6. Curve Precision-Recall

### 6.1 Motivazione per Dataset Sbilanciati

Quando la classe positiva è **rara** (e.g., $P(y=1) \ll 0.5$), la curva ROC può essere **poco informativa**:

**Problema con ROC per classi rare**:

1. Il numero di negativi $N$ è molto grande
2. Anche un piccolo FPR corrisponde a **molti falsi positivi** in termini assoluti
3. La maggior parte della curva ROC è compressa vicino all'origine
4. Variazioni importanti nella precision sono mascherate

**Esempio numerico**:

Dataset: 10,000 esempi, 100 positivi (1%), 9,900 negativi.

Due classificatori:
- **Modello A**: TPR = 0.90, FPR = 0.02
- **Modello B**: TPR = 0.90, FPR = 0.05

Sulla curva ROC sembrano molto simili (stessa TPR, FPR simili).

Ma calcoliamo la precision:

$\text{Precision}_A = \frac{TP}{TP + FP} = \frac{90}{90 + (0.02 \times 9900)} = \frac{90}{288} \approx 0.31$

$\text{Precision}_B = \frac{90}{90 + (0.05 \times 9900)} = \frac{90}{585} \approx 0.15$

La precision di B è **metà** di quella di A! Ma questo non è evidente nella curva ROC.

**Soluzione**: La curva **Precision-Recall** focalizza l'attenzione sui positivi, rendendola più informativa per dataset sbilanciati.

### 6.2 Definizione della Curva PR

**Definizione**: La curva Precision-Recall plotta:
$\text{PR}(\tau) = \big(\text{Recall}(\tau), \text{Precision}(\tau)\big)$

al variare della soglia $\tau$.

**Coordinate**:
- **Asse X**: Recall = $\frac{TP}{P}$
- **Asse Y**: Precision = $\frac{TP}{TP + FP}$

**Costruzione**:

```
Input: Score s_i e label y_i per i = 1,...,n

1. Ordina per score decrescente: s_1 ≥ s_2 ≥ ... ≥ s_n
2. Inizializza: TP = 0, FP = 0
3. Per ogni soglia τ:
   a. Aggiorna TP e FP
   b. Calcola: Recall = TP/P, Precision = TP/(TP+FP)
   c. Aggiungi punto (Recall, Precision)
```

### 6.3 Interpretazione e Comportamento

**Punti notevoli**:

**Alta soglia** ($\tau \to 1$):
- Poche predizioni positive (solo le più confidenti)
- Recall basso, Precision alta
- Punto in basso a destra della curva

**Bassa soglia** ($\tau \to 0$):
- Molte predizioni positive
- Recall alto, Precision bassa (≈ prevalenza)
- Punto in alto a sinistra della curva

**Baseline casuale**:

Un classificatore casuale che predice positivo con probabilità $p$ ottiene:
$\text{Precision}_{\text{random}} = \frac{P}{n} = \pi$

indipendentemente da $p$ (in media). Quindi la baseline è una **linea orizzontale** a $y = \pi$.

**Interpretazione**: Una curva PR buona deve stare **sopra** questa baseline.

**Forma tipica**: La curva PR tende a decrescere muovendosi da sinistra a destra (aumentando recall). Questo riflette il trade-off precision-recall.

### 6.4 Average Precision (AP)

L'**Average Precision** riassume la curva PR in un singolo numero.

**Definizione** (interpolata):
$\text{AP} = \sum_{k=1}^{n} (R_k - R_{k-1}) \cdot P_k$

dove $(P_k, R_k)$ sono precision e recall al $k$-esimo elemento rankat, ordinati per recall crescente.

**Interpretazione**: Approssimazione dell'area sotto la curva PR, pesando ogni livello di recall per quanto è "grande" (quanto recall guadagniamo).

**Definizione alternativa** (usata in PASCAL VOC):
$\text{AP} = \sum_{k=1}^{n} (R_k - R_{k-1}) \cdot P_{\text{interp}}(R_k)$

dove:
$P_{\text{interp}}(R_k) = \max_{R' \geq R_k} P(R')$

Questo usa la precision **interpolata** (massima raggiungibile per recall $\geq R_k$), rendendo la curva monotona.

**Relazione con Ranking**:

$\text{AP} = \frac{1}{P} \sum_{k=1}^{n} P(k) \cdot \text{rel}(k)$

dove:
- $P(k)$ = precision at rank $k$
- $\text{rel}(k) = 1$ se l'item al rank $k$ è positivo, 0 altrimenti
- $P$ = numero totale di positivi

**Interpretazione**: L'AP è la precision media su tutte le posizioni dove troviamo un positivo.

### 6.5 Precision@K e Recall@K

In information retrieval e ranking systems, spesso ci interessano solo i top-K risultati.

**Precision@K**:
$P@K = \frac{|\{i \in \text{top-}K : y_i = 1\}|}{K}$

Frazione di positivi tra i primi K elementi rankati.

**Recall@K**:
$R@K = \frac{|\{i \in \text{top-}K : y_i = 1\}|}{P}$

Frazione di tutti i positivi catturati nei primi K elementi.

**Average Precision@K**:
$AP@K = \frac{1}{\min(m, K)} \sum_{k=1}^{K} P(k) \cdot \text{rel}(k)$

dove $m$ è il numero di positivi nel dataset.

**Uso tipico**: 
- Motori di ricerca: P@10 (prime 10 ricerche)
- Sistemi di raccomandazione: P@20 (prime 20 raccomandazioni)
- Object detection: mAP@IoU (mean Average Precision a diversi IoU threshold)

### 6.6 Confronto AUC-ROC vs AUC-PR

**Differenze fondamentali**:

| Aspetto | AUC-ROC | AUC-PR |
|---------|---------|--------|
| **Assi** | TPR vs FPR | Precision vs Recall |
| **Focus** | Bilanciamento tra positivi e negativi | Solo classe positiva |
| **Baseline** | Diagonale (0.5) | Orizzontale (prevalenza $\pi$) |
| **Dipendenza da $\pi$** | Invariante | Dipendente |
| **Dataset bilanciati** | Ottimo | Equivalente |
| **Dataset sbilanciati** | Può essere misleading | Più informativa |
| **Interpretazione** | Ranking globale | Rilevanza dei positivi |

**Teorema 6.1** (Sensibilità al Prior):
*Dato uno shift nella prevalenza da $\pi_{\text{train}}$ a $\pi_{\text{test}}$:*
- *L'AUC-ROC rimane invariante*
- *L'AUC-PR cambia proporzionalmente*

**Dimostrazione**:

ROC usa metriche condizionate su $y$:
$\text{TPR} = P(\hat{y}=1|y=1), \quad \text{FPR} = P(\hat{y}=1|y=0)$

Queste dipendono solo da $p(x|y)$, non da $P(y)$.

PR usa Precision che dipende esplicitamente dal prior:
$\text{Precision} = P(y=1|\hat{y}=1) = \frac{P(\hat{y}=1|y=1) \cdot P(y=1)}{P(\hat{y}=1)}$

Per Bayes:
$\text{Precision} = \frac{\text{TPR} \cdot \pi}{\text{TPR} \cdot \pi + \text{FPR} \cdot (1-\pi)}$

Cambiando $\pi$, la precision cambia, quindi cambia AUC-PR. $\square$

**Implicazione pratica**: Se il test set ha prevalenza diversa dal training, AUC-PR sarà diversa anche con stesso modello. Questo rende AUC-PR più "onesta" per dataset molto sbilanciati.

**Quando usare quale**:

- **ROC-AUC**: 
  - Dataset bilanciati o moderatamente sbilanciati
  - Interessa ranking generale
  - Vogliamo confrontare modelli indipendentemente dalla prevalenza
  
- **PR-AUC**:
  - Dataset fortemente sbilanciati ($\pi < 0.1$ o $\pi > 0.9$)
  - Focus sulla classe positiva rara
  - Information retrieval e detection tasks

## 7. Metriche Avanzate e Robuste

### 7.1 Matthews Correlation Coefficient (MCC)

Il **Matthews Correlation Coefficient** è considerato una delle metriche più bilanciate e robuste per classificazione binaria.

**Definizione**:
$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$

**Derivazione**: L'MCC è il **coefficiente di correlazione di Pearson** $\phi$ tra le variabili binarie $y$ (label reale) e $\hat{y}$ (predizione).

Per due variabili binarie, il coefficiente $\phi$ è:
$\phi = \frac{n_{11}n_{00} - n_{10}n_{01}}{\sqrt{n_{1\cdot}n_{0\cdot}n_{\cdot1}n_{\cdot0}}}$

dove $n_{ij}$ è la frequenza congiunta di $y=i$ e $\hat{y}=j$. Sostituendo con la notazione della confusion matrix:
- $n_{11} = TP$ (entrambi 1)
- $n_{00} = TN$ (entrambi 0)
- $n_{10} = FN$ ($y=1$, $\hat{y}=0$)
- $n_{01} = FP$ ($y=0$, $\hat{y}=1$)
- $n_{1\cdot} = TP + FN$ (totale reali positivi)
- $n_{0\cdot} = TN + FP$ (totale reali negativi)
- $n_{\cdot1} = TP + FP$ (totale predetti positivi)
- $n_{\cdot0} = TN + FN$ (totale predetti negativi)

Otteniamo esattamente la formula dell'MCC.

**Range e Interpretazione**:

$\text{MCC} \in [-1, +1]$

- $\text{MCC} = +1$: **Perfetto** - Predizione completamente corretta
- $\text{MCC} = 0$: **Casuale** - Performance non migliore del caso
- $\text{MCC} = -1$: **Inverso perfetto** - Predizioni completamente sbagliate (ma consistentemente)

**Interpretazione come correlazione**:
- MCC positivo: Associazione positiva tra predizioni e realtà
- MCC vicino a 0: Nessuna associazione (predizioni random)
- MCC negativo: Associazione negativa (predizioni sistematicamente inverse)

**Proprietà Fondamentali**:

**Proprietà 7.1** (Simmetria):
$\text{MCC}(y, \hat{y}) = \text{MCC}(\neg y, \neg \hat{y})$

L'MCC è invariante rispetto allo scambio di classe (chiamare "positivo" quello che prima era "negativo").

**Dimostrazione**: Sotto lo scambio $y \leftrightarrow (1-y)$ e $\hat{y} \leftrightarrow (1-\hat{y})$:
- $TP \leftrightarrow TN$
- $FP \leftrightarrow FN$

Il numeratore diventa: $TN \cdot TP - FN \cdot FP = TP \cdot TN - FP \cdot FN$ (invariato).

Il denominatore è simmetrico per costruzione. $\square$

**Proprietà 7.2** (Robustezza allo Sbilanciamento):
L'MCC **non favorisce la classe maggioritaria** ed è considerato la metrica più affidabile per dataset sbilanciati.

**Confronto con Accuracy su Dataset Sbilanciato**:

Esempio: 95 negativi, 5 positivi.

**Classificatore Dummy** (sempre negativo):
- TP = 0, TN = 95, FP = 0, FN = 5
- Accuracy = $95/100 = 0.95$ (sembra ottimo!)
- MCC = $\frac{0 \cdot 95 - 0 \cdot 5}{\sqrt{0 \cdot 5 \cdot 95 \cdot 100}} = \frac{0}{0}$ (indefinito, o 0)

**Classificatore Bilanciato**:
- TP = 4, TN = 90, FP = 5, FN = 1
- Accuracy = $94/100 = 0.94$ (leggermente peggio)
- MCC = $\frac{4 \cdot 90 - 5 \cdot 1}{\sqrt{9 \cdot 5 \cdot 95 \cdot 91}} \approx 0.60$ (molto meglio!)

L'MCC riconosce correttamente che il secondo classificatore è superiore.

**Relazione con altre metriche**:

L'MCC può essere espresso in termini di TPR, TNR, PPV (Precision), NPV:

$\text{MCC} = \sqrt{\text{TPR} \cdot \text{TNR} \cdot \text{PPV} \cdot \text{NPV}} - \sqrt{\text{FNR} \cdot \text{FPR} \cdot \text{FOR} \cdot \text{FDR}}$

dove FOR = False Omission Rate, FDR = False Discovery Rate.

**Nota computazionale**: Quando uno qualsiasi dei termini nel denominatore è zero, l'MCC è indefinito (divisione per zero). In pratica, si assegna MCC = 0 in questi casi.

### 7.2 Cohen's Kappa ($\kappa$)

Il **Cohen's Kappa** misura l'accordo tra predizioni e realtà, **corretto per l'accordo casuale**.

**Definizione**:
$\kappa = \frac{p_o - p_e}{1 - p_e}$

dove:
- $p_o = \frac{TP + TN}{n}$ è l'**accuratezza osservata**
- $p_e$ è l'**accuratezza attesa per caso**

**Calcolo di $p_e$** (Accordo Casuale Atteso):

Se $y$ e $\hat{y}$ fossero **indipendenti** ma con le stesse distribuzioni marginali:

$p_e = P(y = \hat{y}|\text{indipendenza})$

$= P(y=1) \cdot P(\hat{y}=1) + P(y=0) \cdot P(\hat{y}=0)$

$= \frac{TP + FN}{n} \cdot \frac{TP + FP}{n} + \frac{TN + FP}{n} \cdot \frac{TN + FN}{n}$

$= \frac{(TP+FN)(TP+FP) + (TN+FP)(TN+FN)}{n^2}$

**Interpretazione**:

$\kappa = \frac{\text{Accordo Osservato} - \text{Accordo Casuale}}{1 - \text{Accordo Casuale}}$

- Numeratore: Quanto l'accordo osservato supera il caso
- Denominatore: Massimo miglioramento possibile rispetto al caso

**Range**:

$\kappa \in [-1, 1]$

ma tipicamente $\kappa \in [0, 1]$ per classificatori ragionevoli.

**Scala di Landis e Koch** (interpretazione classica):

| Kappa | Forza dell'Accordo |
|-------|-------------------|
| $< 0$ | Peggiore del caso |
| $0.00 - 0.20$ | Lieve |
| $0.21 - 0.40$ | Discreto |
| $0.41 - 0.60$ | Moderato |
| $0.61 - 0.80$ | Sostanziale |
| $0.81 - 1.00$ | Quasi perfetto |

**Esempio di calcolo**:

Dataset: 100 esempi, 60 positivi, 40 negativi.
Modello: TP = 50, TN = 30, FP = 10, FN = 10.

$p_o = \frac{50 + 30}{100} = 0.80$

$p_e = \frac{60 \cdot 60}{100^2} + \frac{40 \cdot 40}{100^2} = \frac{3600 + 1600}{10000} = 0.52$

$\kappa = \frac{0.80 - 0.52}{1 - 0.52} = \frac{0.28}{0.48} \approx 0.58$

Interpretazione: Accordo **moderato** (secondo Landis e Koch).

**Relazione con MCC**:

Per problemi binari, MCC e Kappa sono correlati ma **non identici**. In generale:
- MCC è preferito per la sua interpretazione come correlazione
- MCC ha migliori proprietà matematiche
- Kappa è più usato in ambito medico/statistico per inter-rater agreement

**Differenza chiave**: Kappa usa le distribuzioni marginali empiriche per calcolare $p_e$, mentre MCC è una pura misura di correlazione.

### 7.3 Balanced Accuracy

La **balanced accuracy** è particolarmente utile per dataset sbilanciati, dando peso uguale a ciascuna classe.

**Definizione**:
$\text{Balanced Accuracy} = \frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right) = \frac{\text{TPR} + \text{TNR}}{2}$

**Equivalente**:
$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$

**Motivazione**: L'accuracy standard può essere dominata dalla classe maggioritaria. La balanced accuracy:
- Calcola accuracy per ciascuna classe separatamente
- Fa la media (non pesata) delle due

**Esempio illustrativo**:

Dataset: 950 negativi, 50 positivi.

**Classificatore A** (sempre negativo):
- Accuracy = $950/1000 = 0.95$
- Balanced Accuracy = $\frac{0 + 1}{2} = 0.50$ ← Rivela che è casuale!

**Classificatore B**:
- TP = 40, TN = 900, FP = 50, FN = 10
- Accuracy = $940/1000 = 0.94$
- Balanced Accuracy = $\frac{40/50 + 900/950}{2} = \frac{0.8 + 0.947}{2} \approx 0.87$

La balanced accuracy rivela correttamente che B è molto migliore di A, anche se l'accuracy semplice è simile.

**Generalizzazione Multi-Classe**:

$\text{Balanced Accuracy} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FN_c}$

dove $C$ è il numero di classi.

**Proprietà**:
- Range: $[0, 1]$
- Balanced Accuracy = 0.5 per classificatore casuale (binario)
- Non favorisce alcuna classe
- Più interpretabile dell'MCC per utenti non tecnici

**Confronto con Macro-F1**:

Entrambe danno peso uguale alle classi, ma:
- Balanced Accuracy: Media di recall per classe
- Macro-F1: Media di F1 per classe (combina precision e recall)

### 7.4 Informedness e Markedness

Due metriche meno note ma teoricamente importanti.

**Informedness** (Bookmaker Informedness):
$\text{Informedness} = \text{TPR} + \text{TNR} - 1 = \text{Sensitivity} + \text{Specificity} - 1$

**Interpretazione**: 
- Quanto il classificatore è più informato del caso?
- Probabilità di decisione informata vs casuale
- Range: $[-1, 1]$ dove 0 = casuale

**Relazione**: 
$\text{Informedness} = 2 \cdot \text{Balanced Accuracy} - 1$

**Markedness**:
$\text{Markedness} = \text{PPV} + \text{NPV} - 1 = \text{Precision} + \text{NPV} - 1$

dove NPV (Negative Predictive Value) = $\frac{TN}{TN+FN}$.

**Interpretazione**: Quanto sono "marcate" (affidabili) le predizioni?

**Teorema 7.1** (Relazione MCC con Informedness e Markedness):
$\text{MCC} = \sqrt{\text{Informedness} \times \text{Markedness}}$

(quando tutti i termini sono definiti e non negativi)

**Dimostrazione** (sketch):
$\text{Informedness} = \frac{TP}{P} + \frac{TN}{N} - 1$

$\text{Markedness} = \frac{TP}{P^*} + \frac{TN}{N^*} - 1$

Espandendo e semplificando usando le identità della confusion matrix, si ottiene che il loro prodotto geometrico è correlato a MCC². $\square$

## 8. Valutazione Probabilistica e Calibrazione

### 8.1 Introduzione

Molti classificatori producono **probabilità** $p(y=1|x)$ anziché solo label binari. È importante valutare:
1. **Discriminazione**: Il modello separa bene le classi? (ROC, PR)
2. **Calibrazione**: Le probabilità predette riflettono le vere frequenze?

Un modello può avere ottima discriminazione (AUC alta) ma pessima calibrazione.

**Esempio**: Un modello che produce sempre $p=0.9$ per positivi e $p=0.1$ per negativi ha:
- Perfetta discriminazione (AUC = 1)
- Pessima calibrazione (le probabilità non riflettono l'incertezza reale)

### 8.2 Log Loss (Cross-Entropy Loss)

La **log loss** valuta la qualità delle probabilità predette.

**Definizione** (Binaria):
$\mathcal{L}_{\text{log}} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$

dove $p_i = P(y_i=1|x_i)$ è la probabilità predetta.

**Derivazione**: La log loss è l'**entropia incrociata** tra distribuzione empirica e predetta:

$H(q, p) = -\mathbb{E}_{y \sim q}[\log p(y|x)]$

Per label binari deterministici: $q(y=1|x) = y$ (0 o 1):
$H = -y \log p - (1-y) \log(1-p)$

**Proprietà**:

1. **Range**: $[0, +\infty)$ dove 0 indica probabilità perfette
2. **Penalizzazione logaritmica**: Predizioni confidenti ma sbagliate sono penalizzate esponenzialmente
3. **Proper scoring rule**: Minimizzata dalle vere probabilità

**Proper Scoring Rule**: Una metrica è "proper" se è ottimizzata predicendo le vere probabilità:
$\mathbb{E}_{y \sim p^*}[S(y, p)] \geq \mathbb{E}_{y \sim p^*}[S(y, q)] \quad \forall q$

con uguaglianza solo se $q = p^*$.

**Esempi di penalizzazione**:

| Vera Classe | Probabilità Predetta | Log Loss |
|-------------|---------------------|----------|
| $y=1$ | $p=0.99$ | $-\log(0.99) \approx 0.01$ |
| $y=1$ | $p=0.9$ | $-\log(0.9) \approx 0.11$ |
| $y=1$ | $p=0.5$ | $-\log(0.5) \approx 0.69$ |
| $y=1$ | $p=0.1$ | $-\log(0.1) \approx 2.30$ |
| $y=1$ | $p=0.01$ | $-\log(0.01) \approx 4.61$ |

Notare la **penalizzazione esponenziale**: predire $p=0.01$ quando $y=1$ costa 460 volte più che predire $p=0.99$!

**Collegamento con Maximum Likelihood**:

Minimizzare la log loss è equivalente a massimizzare la log-likelihood:
$\arg\min_\theta \mathcal{L}_{\text{log}} = \arg\max_\theta \sum_{i=1}^{n} \log p(y_i|x_i, \theta)$

Questo è il principio del **Maximum Likelihood Estimation (MLE)**.

**Multi-Classe**:
$\mathcal{L}_{\text{log}} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \log(p_{ic})$

dove $y_{ic} = 1$ se $y_i = c$, altrimenti 0 (one-hot encoding).

### 8.3 Brier Score

Il **Brier score** misura l'errore quadratico delle probabilità.

**Definizione** (Binaria):
$\text{BS} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2$

dove $y_i \in \{0, 1\}$.

**Derivazione**: È semplicemente il **Mean Squared Error (MSE)** tra probabilità predette e label binari.

**Range**: $[0, 1]$ dove 0 indica probabilità perfette.

**Decomposizione di Murphy**:

Il Brier score può essere decomposto in tre componenti interpretabili:

$\text{BS} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$

dove:

**Uncertainty** (varianza intrinseca):
$\text{Uncertainty} = \bar{y}(1 - \bar{y})$
dove $\bar{y}$ è la prevalenza. Non controllabile dal modello.

**Resolution** (capacità di separare):
$\text{Resolution} = \frac{1}{n} \sum_{k=1}^{K} n_k(\bar{y}_k - \bar{y})^2$
Quanto bene il modello separa esempi con diverse probabilità vere. Vogliamo massimizzarlo.

**Reliability** (calibrazione):
$\text{Reliability} = \frac{1}{n} \sum_{k=1}^{K} n_k(\bar{y}_k - \bar{p}_k)^2$
Deviazione tra probabilità predette e frequenze osservate. Vogliamo minimizzarlo.

**Interpretazione**: 
- Un buon modello ha **alta resolution** (separa bene) e **bassa reliability** (ben calibrato)
- BS basso indica entrambe le proprietà

**Confronto Log Loss vs Brier Score**:

| Aspetto | Log Loss | Brier Score |
|---------|----------|-------------|
| **Penalizzazione** | Logaritmica (severa) | Quadratica (moderata) |
| **Range** | $[0, \infty)$ | $[0, 1]$ |
| **Proper scoring** | Sì | Sì |
| **Interpretabilità** | Meno intuitiva | Più intuitiva (MSE) |
| **Sensibilità a errori** | Molto alta | Moderata |
| **Decomponibile** | No (direttamente) | Sì (Murphy) |

**Quando usare quale**:
- **Log Loss**: Training di modelli (gradient-based), quando predizioni molto sbagliate devono essere evitate
- **Brier Score**: Valutazione finale, quando vogliamo interpretabilità e decomposizione

### 8.4 Calibrazione delle Probabilità

Un modello è **ben calibrato** (o **affidabile**) se le probabilità predette riflettono le vere frequenze:

**Definizione Formale**:
$P(y=1 | p(y=1|x) = q) = q \quad \forall q \in [0,1]$

**Interpretazione**: "Tra tutti gli esempi a cui assegno probabilità $q$, una frazione $q$ dovrebbe essere effettivamente positiva."

**Esempio**:
- Se predico $p=0.7$ per 100 esempi, circa 70 dovrebbero essere realmente positivi
- Se predico $p=0.3$ per 50 esempi, circa 15 dovrebbero essere realmente positivi

#### 8.4.1 Reliability Diagram (Calibration Plot)

Il **reliability diagram** visualizza la calibrazione.

**Procedura**:

1. **Binning**: Dividi le predizioni in $B$ bin basati su $p_i$ (e.g., $B=10$ bin di ampiezza 0.1)

2. **Per ogni bin $b$**:
   - Calcola **probabilità media predetta**: $\bar{p}_b = \frac{1}{|B_b|} \sum_{i \in B_b} p_i$
   - Calcola **frazione empirica di positivi**: $\bar{y}_b = \frac{1}{|B_b|} \sum_{i \in B_b} y_i$

3. **Plot**: $\bar{y}_b$ (asse Y) vs $\bar{p}_b$ (asse X)

**Interpretazione**:

- **Diagonale perfetta** ($\bar{y}_b = \bar{p}_b$ per ogni bin): Calibrazione perfetta
- **Sopra la diagonale**: Modello **sotto-confidente** (predice probabilità troppo basse)
- **Sotto la diagonale**: Modello **sovra-confidente** (predice probabilità troppo alte)
- **Forma a S**: Modello sovra-confidente alle estremità, sotto-confidente al centro

**Esempio**:

Bin $[0.8, 0.9]$:
- $\bar{p}_b = 0.85$ (probabilità media predetta)
- $\bar{y}_b = 0.95$ (frazione reale di positivi)
- Interpretazione: Il modello è sotto-confidente in questa regione

#### 8.4.2 Expected Calibration Error (ECE)

L'**ECE** quantifica numericamente la deviazione dalla calibrazione perfetta.

**Definizione**:
$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} |\bar{y}_b - \bar{p}_b|$

dove:
- $B$ = numero di bin
- $|B_b|$ = numero di esempi nel bin $b$
- $\bar{y}_b$ = frazione empirica di positivi nel bin
- $\bar{p}_b$ = probabilità media predetta nel bin

**Interpretazione**: Media pesata della deviazione assoluta dalla calibrazione perfetta.

**Proprietà**:
- Range: $[0, 1]$
- ECE = 0 indica calibrazione perfetta
- Usa errore **assoluto** (più robusto del quadratico)

**Maximum Calibration Error (MCE)**:
$\text{MCE} = \max_{b=1,\ldots,B} |\bar{y}_b - \bar{p}_b|$

Misura la **peggiore** deviazione locale dalla calibrazione.

**Scelta del numero di bin**: Tipicamente $B \in \{10, 15, 20\}$. Troppo pochi → scarsa risoluzione. Troppi → bin con pochi esempi (stime instabili).

#### 8.4.3 Metodi di Calibrazione

Se un modello ha buona discriminazione ma scarsa calibrazione, possiamo **post-processare** le probabilità.

**Platt Scaling** (Regressione Logistica):

Applica una trasformazione logistica agli score:
$p_{\text{calib}}(y=1|x) = \frac{1}{1 + e^{-(a \cdot s(x) + b)}}$

dove:
- $s(x)$ è lo score non calibrato del modello
- $a, b$ sono parametri appresi su un **validation set**

**Procedura**:
1. Genera score $s_i$ per il validation set
2. Fit regressione logistica: $y_i \sim \text{Logistic}(a \cdot s_i + b)$
3. Applica la trasformazione agli score futuri

**Quando usare**: Funziona bene quando la relazione score-probabilità è monotona e approssimativamente sigmoidale (comune per SVM, Naive Bayes).

**Isotonic Regression**:

Apprende una funzione **monotona crescente** non-parametrica $f: \mathbb{R} \to [0,1]$:
$p_{\text{calib}}(y=1|x) = f(s(x))$

**Procedura**:
1. Ordina validation set per score crescente
2. Trova la funzione a gradini monotona che minimizza MSE con i label
3. Applica $f$ agli score futuri

**Quando usare**: Più flessibile di Platt, funziona per relazioni non-sigmoidali. Richiede più dati per evitare overfitting.

**Temperature Scaling** (per Neural Networks):

Scala i **logit** con un parametro temperatura $T$:
$p_i^{\text{calib}} = \frac{e^{z_i/T}}{\sum_{j=1}^{C} e^{z_j/T}}$

dove $z_i$ sono i logit (output pre-softmax).

**Effetto di $T$**:
- $T > 1$: **"Smoothing"** → probabilità meno confidenti (più disperse)
- $T < 1$: **"Sharpening"** → probabilità più confidenti (più concentrate)
- $T = 1$: Nessun cambiamento

**Apprendimento**: Trova $T$ che minimizza log loss sul validation set (tipicamente con grid search o gradient descent).

**Vantaggi**: Mantiene l'ordinamento relativo delle classi, singolo parametro globale, preserva accuracy.

**Confronto metodi**:

| Metodo | Parametri | Flessibilità | Requisiti Dati | Uso Tipico |
|--------|-----------|--------------|----------------|------------|
| Platt | 2 | Bassa (sigmoid) | Moderati | SVM, Naive Bayes |
| Isotonic | Molti (piecewise) | Alta | Molti | Alberi, ensemble |
| Temperature | 1 | Bassa (scala) | Pochi | Neural networks |

### 8.5 Decisioni Ottimali con Costi Asimmetrici

#### 8.5.1 Framework del Rischio Bayesiano

Abbiamo visto nella Sezione 3 che la decision rule ottimale minimizza il rischio atteso. Approfondiamo ora come utilizzare questo framework in pratica.

**Rischio Atteso** per soglia $\tau$:
$$R(\tau) = L_{FN} \cdot \text{FNR}(\tau) \cdot \pi + L_{FP} \cdot \text{FPR}(\tau) \cdot (1-\pi)$$

dove $\pi = P(Y=1)$ è la prevalenza.

**Teorema 8.1** (Soglia Ottimale per Costi Asimmetrici):
*Data loss matrix con costi $L_{FP}$ e $L_{FN}$, la soglia ottimale è:*

$$\tau^* = \frac{L_{FP} \cdot (1-\pi)}{L_{FP} \cdot (1-\pi) + L_{FN} \cdot \pi}$$

**Dimostrazione**:

Dal Teorema 3.2, classifichiamo come positivo quando:
$$\frac{p(y=1|x)}{p(y=0|x)} > \frac{L_{FP}}{L_{FN}}$$

Riscrivendo in termini di $p(y=1|x) = p$:
$$\frac{p}{1-p} > \frac{L_{FP}}{L_{FN}}$$

Risolvendo per $p$:
$$p > \frac{L_{FP}}{L_{FP} + L_{FN}}$$

Questa è la soglia ottimale quando $\pi = 0.5$. Per prevalenza arbitraria, la soglia diventa:
$$\tau^* = \frac{L_{FP} \cdot (1-\pi)}{L_{FP} \cdot (1-\pi) + L_{FN} \cdot \pi}$$

$\square$

**Casi speciali**:

1. **Costi uguali** ($L_{FP} = L_{FN} = 1$):
   $$\tau^* = \frac{1-\pi}{1-\pi+\pi} = 1-\pi$$
   
2. **Prevalenza bilanciata** ($\pi = 0.5$):
   $$\tau^* = \frac{L_{FP}}{L_{FP} + L_{FN}}$$

**Esempio pratico**:

Screening medico: $L_{FN} = 1000$ (vita a rischio), $L_{FP} = 1$ (test aggiuntivo), $\pi = 0.01$.

$$\tau^* = \frac{1 \cdot 0.99}{1 \cdot 0.99 + 1000 \cdot 0.01} = \frac{0.99}{10.99} \approx 0.09$$

Soglia molto bassa → massimizziamo la sensibilità, accettando molti falsi positivi.

### 8.5.2 Cost-Sensitive Learning

Invece di ottimizzare la soglia post-hoc, possiamo integrare i costi **durante il training**.

**Weighted Loss**:
$$\mathcal{L}_{\text{weighted}} = -\frac{1}{n}\sum_{i=1}^n \left[w_1 \cdot y_i\log(p_i) + w_0 \cdot (1-y_i)\log(1-p_i)\right]$$

dove i pesi sono proporzionali ai costi:
$$w_1 = L_{FN}, \quad w_0 = L_{FP}$$

**Class Rebalancing**:
Alternativamente, possiamo ri-pesare le classi per compensare lo sbilanciamento:
$$w_c = \frac{n}{C \cdot n_c}$$
dove $C$ è il numero di classi e $n_c$ il numero di esempi della classe $c$.

## 9. Classificazione Multi-Classe

### 9.1 Estensione della Matrice di Confusione

Per $C$ classi, la matrice di confusione è $C \times C$:

$$\text{CM}[i,j] = \text{numero di esempi con classe reale } i \text{ predetti come } j$$

**Diagonale**: Predizioni corrette
**Fuori diagonale**: Errori

### 9.2 Metriche Per-Classe

Per ogni classe $c$, definiamo:

**Precision per classe $c$**:
$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c} = \frac{\text{CM}[c,c]}{\sum_i \text{CM}[i,c]}$$

**Recall per classe $c$**:
$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c} = \frac{\text{CM}[c,c]}{\sum_j \text{CM}[c,j]}$$

**F1 per classe $c$**:
$$F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### 9.3 Aggregazione: Macro vs Micro vs Weighted

**Macro-Average** (media semplice):
$$\text{Macro-Precision} = \frac{1}{C}\sum_{c=1}^C \text{Precision}_c$$

**Interpretazione**: Ogni classe ha peso uguale, indipendentemente dalla sua frequenza.
**Uso**: Dataset bilanciati, tutte le classi sono ugualmente importanti.

**Micro-Average** (aggregazione globale):
$$\text{Micro-Precision} = \frac{\sum_{c=1}^C TP_c}{\sum_{c=1}^C (TP_c + FP_c)}$$

**Interpretazione**: Ogni esempio ha peso uguale.
**Uso**: Dataset sbilanciati, classi maggioritarie sono più importanti.

**Weighted-Average** (pesata per frequenza):
$$\text{Weighted-Precision} = \sum_{c=1}^C \frac{n_c}{n} \cdot \text{Precision}_c$$

**Interpretazione**: Peso proporzionale alla dimensione della classe.
**Uso**: Compromesso tra macro e micro.

**Esempio**:

3 classi: A (100 esempi), B (50 esempi), C (10 esempi)

| Classe | Precision | Recall |
|--------|-----------|--------|
| A | 0.90 | 0.85 |
| B | 0.80 | 0.75 |
| C | 0.50 | 0.40 |

- **Macro-Precision** = $(0.90 + 0.80 + 0.50)/3 = 0.73$
- **Micro-Precision** = $(90 + 40 + 5)/(100 + 50 + 10) = 0.84$
- **Weighted-Precision** = $0.90 \cdot \frac{100}{160} + 0.80 \cdot \frac{50}{160} + 0.50 \cdot \frac{10}{160} = 0.85$

### 9.4 One-vs-Rest e One-vs-One

**One-vs-Rest (OvR)**:
- Per ogni classe $c$, creiamo un problema binario: classe $c$ vs tutte le altre
- Calcoliamo metriche binarie per ciascun problema
- Aggreghiamo con macro/micro/weighted

**One-vs-One (OvO)**:
- Per ogni coppia di classi $(c_i, c_j)$, creiamo un classificatore binario
- Totale: $\binom{C}{2} = \frac{C(C-1)}{2}$ classificatori
- Utile per SVM multi-classe

### 9.5 Matthews Correlation Coefficient Multi-Classe

L'MCC può essere esteso al caso multi-classe:

$$\text{MCC} = \frac{\sum_{k,l,m} C_{kk}C_{lm} - C_{kl}C_{mk}}{\sqrt{\sum_k\left(\sum_l C_{kl}\right)\left(\sum_{k'\neq k}\sum_{l'}C_{k'l'}\right)} \cdot \sqrt{\sum_k\left(\sum_l C_{lk}\right)\left(\sum_{k'\neq k}\sum_{l'}C_{l'k'}\right)}}$$

dove $C$ è la matrice di confusione.

**Interpretazione**: Generalizzazione del coefficiente di correlazione al caso multi-classe.

**Range**: $[-1, +1]$ come nel caso binario.

## 10. Guida Pratica alla Scelta delle Metriche

### 10.1 Albero Decisionale

```
Dataset bilanciato?
├─ Sì
│  ├─ Interessa solo accuracy? → Accuracy, Balanced Accuracy
│  └─ Serve probabilità? → Log Loss, Brier Score, Calibration
│
└─ No (sbilanciato)
   ├─ Qual è la classe di interesse?
   │  ├─ Classe rara (positiva)
   │  │  ├─ FN molto costosi? → Recall, F2, PR-AUC
   │  │  ├─ FP molto costosi? → Precision, F0.5
   │  │  └─ Bilanciamento? → F1, MCC
   │  │
   │  └─ Entrambe le classi importanti → Balanced Accuracy, MCC, Cohen's Kappa
   │
   └─ Serve valutazione threshold-independent? → ROC-AUC (se moderatamente sbilanciato), PR-AUC (se molto sbilanciato)
```

### 10.2 Raccomandazioni per Dominio

**Medicina (Screening)**:
- **Primarie**: Recall, Sensitivity, F2
- **Secondarie**: Specificity, PR-AUC
- **Perché**: FN (mancata diagnosi) sono critici

**Medicina (Diagnostica Definitiva)**:
- **Primarie**: Balanced Accuracy, MCC, F1
- **Secondarie**: Specificity, PPV
- **Perché**: Bilanciamento tra evitare trattamenti inutili e non perdere malati

**Fraud Detection**:
- **Primarie**: Precision@K, PR-AUC, F1.5
- **Secondarie**: Recall, ROC-AUC
- **Perché**: FN costosi (perdite economiche), ma serve precision ragionevole

**Spam Filtering**:
- **Primarie**: Precision, F0.5
- **Secondarie**: FPR, Specificity
- **Perché**: FP (email legittime in spam) sono inaccettabili

**Information Retrieval**:
- **Primarie**: MAP (Mean Average Precision), NDCG, Precision@K
- **Secondarie**: Recall@K, F1
- **Perché**: Focus su top-K risultati e qualità del ranking

**Computer Vision (Classification)**:
- **Bilanciato**: Top-1 Accuracy, Top-5 Accuracy
- **Sbilanciato**: Macro-F1, Per-class metrics
- **Perché**: Dipende dal numero e bilanciamento delle classi

**Sentiment Analysis / NLP**:
- **Primarie**: Macro-F1, Weighted-F1
- **Secondarie**: Per-class Precision/Recall, Confusion Matrix
- **Perché**: Classi spesso sbilanciate, tutte le sentiment importanti

### 10.3 Checklist di Valutazione

Prima di scegliere le metriche, rispondi a:

1. **Dataset è bilanciato?**
   - [ ] Sì (Accuracy OK)
   - [ ] No (evitare Accuracy)

2. **Costi asimmetrici?**
   - [ ] FP più costosi → enfatizza Precision
   - [ ] FN più costosi → enfatizza Recall
   - [ ] Bilanciati → F1, MCC

3. **Soglia fissa o variabile?**
   - [ ] Fissa → metriche a soglia fissata (Precision, Recall, F1)
   - [ ] Variabile → curve (ROC, PR)

4. **Serve calibrazione?**
   - [ ] Sì → Log Loss, Brier Score, ECE, Reliability Diagram
   - [ ] No → solo discriminazione

5. **Multi-classe?**
   - [ ] Macro (classi ugualmente importanti)
   - [ ] Micro (esempi ugualmente importanti)
   - [ ] Weighted (compromesso)

### 10.4 Metriche da Riportare Sempre

**Minimo indispensabile**:
1. Matrice di confusione (visualizzazione completa)
2. Almeno 2 metriche complementari (e.g., Precision + Recall, o F1 + MCC)
3. Curva appropriata (ROC o PR) con AUC

**Report completo**:
1. Confusion matrix
2. Precision, Recall, F1
3. ROC curve + AUC-ROC
4. PR curve + AUC-PR (se sbilanciato)
5. MCC o Cohen's Kappa
6. Calibration plot + ECE (se probabilistico)
7. Per-class metrics (se multi-classe)

### 10.5 Errori Comuni da Evitare

**❌ Usare solo Accuracy su dataset sbilanciato**
- Un modello dummy può avere accuracy alta

**❌ Ignorare la calibrazione**
- AUC alta non implica probabilità ben calibrate

**❌ Ottimizzare solo una metrica**
- Trade-off impliciti possono nascondere problemi

**❌ Non considerare i costi reali**
- FP e FN raramente hanno stesso costo

**❌ Confrontare modelli con metriche diverse**
- Usare stesse metriche per confronti fair

**❌ Dimenticare intervalli di confidenza**
- Report puntuale senza incertezza è fuorviante

**❌ Usare test set per tuning**
- Porta a overfitting ottimistico

**✅ Best Practices**:
1. Sempre riportare confusion matrix
2. Usare multiple metriche complementari
3. Considerare i costi del dominio applicativo
4. Validare calibrazione se si usano probabilità
5. Report con confidence intervals (bootstrap o cross-validation)
6. Mantenere test set completamente holdout

## Riferimenti e Risorse

**Paper fondamentali**:
- Provost, F., Fawcett, T. (2001). "Robust Classification for Imprecise Environments"
- Davis, J., Goadrich, M. (2006). "The Relationship Between Precision-Recall and ROC Curves"
- Chicco, D., Jurman, G. (2020). "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation"

**Libri consigliati**:
- Murphy, K. P. (2022). "Probabilistic Machine Learning: An Introduction"
- Hastie, T., Tibshirani, R., Friedman, J. (2009). "The Elements of Statistical Learning"
- Bishop, C. M. (2006). "Pattern Recognition and Machine Learning"

**Strumenti software**:
- `sklearn.metrics` (Python): Implementazione completa
- `ROCR` (R): Visualizzazione ROC/PR
- `calibration` (Python): Post-processing calibration
