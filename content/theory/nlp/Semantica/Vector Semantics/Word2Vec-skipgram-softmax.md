# Skip-gram con Softmax

Il modello **Skip-gram** di *word2vec* con softmax è una tecnica di apprendimento non supervisionato usata per generare vettori densi (embedding) che rappresentano parole in uno spazio continuo a dimensione $D$.

Vediamo nel dettaglio tutti i passaggi e le componenti del modello.

## Parametri da apprendere in Skip-Gram con Softmax

Nel modello Skip-Gram di word2vec, l'obiettivo principale è imparare rappresentazioni dense (embedding) delle parole che catturino il loro significato in relazione al contesto in cui appaiono. Per fare ciò, dobbiamo definire e apprendere dei parametri, che rappresentano queste strutture vettoriali.

### Definizione dei parametri

Sia $V$ il vocabolario di parole del modello, e sia $D$ la dimensione dello spazio di embedding, cioè il numero di componenti o caratteristiche usate per rappresentare ciascuna parola come un vettore numerico continuo. Ogni dimensione può essere interpretata come un "tema" o una caratteristica latente che cattura aspetti semantici o sintattici della parola.

Indichiamo con:

$$\large
\bm{\theta} =
\begin{bmatrix}
\bm{\theta}_W \\[0.3em] \hline \\[-0.9em]
\bm{\theta}_C
\end{bmatrix}
\quad\text{con}\quad
\bm{\theta}_W \in \mathbb{R}^{|V| \times D},\quad
\bm{\theta}_C \in \mathbb{R}^{|V| \times D}
$$

l'insieme dei parametri del modello, suddiviso in due matrici principali:

- **$\bm{\theta}_W$** (matrice degli embedding delle parole centro):
 
    - **Dimensione:** $|V| \times D$
    - Ogni riga di $\bm{\theta}_W$ è un vettore che rappresenta una parola specifica **nel ruolo di parola centrale** all’interno di una finestra di contesto. Questo significa che il vettore codifica le proprietà della parola quando è il punto focale della previsione del modello.
    - Il vettore di embedding in $\bm{\theta}_W$ viene usato dal modello per cercare di predire le parole di contesto che la circondano: ad esempio, dato un vettore centrale, il modello calcola la probabilità di ogni parola nel vocabolario come possibile parola di contesto.
    - Questa rappresentazione è fondamentale perché permette al modello di apprendere relazioni tra parole basate sulle co-occorrenze: parole con significati simili o usi simili tendono ad avere vettori vicini nello spazio degli embedding.
    - È importante notare che la stessa parola avrà vettori distinti in $\bm{\theta}_W$ e in $\bm{\theta}_C$, poiché il suo ruolo nel modello cambia (centro vs contesto). Questo permette una rappresentazione più ricca e flessibile del linguaggio.


- **$\bm{\theta}_C$** (matrice degli embedding delle parole contesto):

    - **Dimensione:** $|V| \times D$
    - Ogni riga di $\bm{\theta}_C$ è un vettore che rappresenta una parola **quando essa agisce come contesto** di una parola centrale. In altre parole, questi vettori sono usati per modellare le parole che circondano la parola centrale nella finestra di contesto.
    - La funzione di $\bm{\theta}_C$ è catturare le proprietà semantiche e sintattiche delle parole nel loro ruolo di contesto, cioè come "indizi" o segnali che aiutano a prevedere la parola centrale.
    - Ad esempio, la parola "delicious" avrà un embedding in $\bm{\theta}_C$ che riflette il suo uso frequente vicino a parole legate al cibo, mentre la stessa parola avrà un embedding differente in $\bm{\theta}_W$ quando appare come parola centrale.
    - Questa doppia rappresentazione consente al modello di distinguere come una parola si comporta quando è il fulcro della previsione (centro) rispetto a quando è un "supporto" per predire altre parole (contesto).
    - Grazie a $\bm{\theta}_C$, il modello impara a riconoscere quali parole di contesto sono più probabili dati i vettori delle parole centrali, migliorando così la capacità di rappresentare le relazioni semantiche tra parole.

Questa suddivisione di parametri consente al modello di catturare dinamiche diverse, come il significato di una parola quando appare come centro o quando appare come contesto nella finestra di contesto.

<img src="/images/tikz/a24d8ddc52dc38d45e29d9fd9070e6a1.svg" style="display: block; width: 100%; height: auto; max-height: 600px;" class="tikz-svg" />

### Perché due matrici distinte?

- **Ruoli diversi**: 

  - $\bm{\theta}_W$: embedding quando la parola è **centro** (target da cui si predice).  
  - $\bm{\theta}_C$: embedding quando la parola è **contesto** (segnale per la previsione).

- **Esempio** (“Il **gatto** nero dorme…”):

  - “gatto” → $\bm{\theta}_W$ cattura come “gatto” governa il contesto (“nero”, “dorme”).  
  - “nero”, “dorme” → $\bm{\theta}_C$ catturano come questi agiscono da indizi per “gatto”.

### ⚠️ Problemi con un singolo embedding

1. **Ruolo funzionale perso**
  
   - Ogni parola può comparire sia come **centrale** sia come **di contesto**.
   - Esempio:
  
     - “**book**” come centrale (es. *"I read a book about history."*) → predice parole come *read*, *history*.
     - “**book**” come contesto (es. *"She put the book on the table."*) → aiuta a predire *put*, *table*.
  
   - Se usiamo **un solo embedding**, non distinguiamo questi ruoli → perdiamo informazione funzionale importante.

2. **Relazioni asimmetriche non modellate**

   - Il significato delle relazioni cambia a seconda della direzione:

     - “**eat**” → “**food**” = tipico: il verbo suggerisce l’oggetto (cosa si mangia).
     - “**food**” → “**eat**” = più debole: “food” potrebbe comparire in molti altri contesti (buy, cook, smell…).
  
   - Se usiamo lo stesso embedding per “food” in entrambi i ruoli, non possiamo catturare questa asimmetria.
   - Due matrici permettono:

     - $\theta_W$(eat) → embedding ottimizzato per predire cibo.
     - $\theta_C$(food) → embedding ottimizzato per essere predetto da verbi come *eat*.

3. **Embedding meno precisi**

   - Un solo embedding deve essere "tuttofare" → media tra ruoli e significati.
   - Risultato: vettori **più confusi, meno specializzati**, e performance peggiori in downstream tasks.
   - Due matrici aiutano a ottenere rappresentazioni **più informative e discriminative**.

### Numero totale di parametri

Il numero complessivo di parametri del modello è dato dalla somma degli elementi di entrambe le matrici:

$$
2 \cdot |V| \times D
$$

Ovvero:

- $|V| \times D$ parametri per gli embedding come centro,
- $|V| \times D$ parametri per gli embedding come contesto.

### Visualizzazione intuitiva

Immagina il vocabolario come una lista di parole:

| Indice | Parola      | Embedding Centro ($\bm{\theta}_W$) | Embedding Contesto ($\bm{\theta}_C$) |
|--------|-------------|-----------------------------------------|--------------------------------------------|
| 1      | "lemon"     | vettore in $\mathbb{R}^D$              | vettore in $\mathbb{R}^D$                 |
| 2      | "tablespoon"| vettore in $\mathbb{R}^D$              | vettore in $\mathbb{R}^D$                 |
| ...    | ...         | ...                                     | ...                                        |
| \|V\|    | "jam"       | vettore in $\mathbb{R}^D$              | vettore in $\mathbb{R}^D$                 |

- Quando "tablespoon" è parola centro, useremo la riga 2 di $\bm{\theta}_W$.
- Quando "tablespoon" è nel contesto, useremo la riga 2 di $\bm{\theta}_C$.

### Perché sono vettori?

Rappresentare le parole come vettori in uno spazio continuo di dimensione $D$ consente al modello di apprendere relazioni semantiche e sintattiche tra parole, ad esempio:

- Parole con significati simili tendono ad avere vettori vicini nello spazio,
- Relazioni di analogia possono essere rappresentate come vettori differenza, es. vettore("re") - vettore("uomo") + vettore("donna") ≈ vettore("regina").

### Riassumendo:

- $\bm{\theta}_W$ e $\bm{\theta}_C$ sono matrici di embedding distinte per parola centro e contesto.
- Entrambe hanno dimensione $|V| \times D$.
- Complessivamente abbiamo $2 \cdot |V| \times D$ parametri da imparare.
- Questo doppio embedding è la chiave per modellare le relazioni tra parole in un modo più ricco e flessibile.

Questa struttura di parametri sarà la base su cui il modello Skip-Gram costruirà la sua funzione di probabilità e la sua funzione di perdita durante l'addestramento.

## Il concetto di self-supervision nello Skip-gram

Il training si basa su un grande corpus di testo, ad esempio:  
`... lemon, a tablespoon of apricot jam, a pinch ...`

Il modello considera una finestra di contesto di ampiezza $m$ (ad esempio $m=2$) centrata sulla parola al tempo $t$:

- La parola centrale è $w_t$, nel nostro esempio "apricot".
- Le parole del contesto sono quelle all’interno della finestra di dimensione $2m$ intorno a $w_t$:
  - $w_{t-2}$, $w_{t-1}$ a sinistra,
  - $w_{t+1}$, $w_{t+2}$ a destra.

|   | ~~lemon~~ | ~~a~~ | [tablespoon | of | **apricot** | jam | a] | ~~pinch~~ |
|:-:|:---------:|:-----:|:-----------:|:--:|:-----------:|:---:|:--:|:--------:|
|   |           |       |  $w_{t-2}$  | $w_{t-1}$ | **$w_t$** | $w_{t+1}$ | $w_{t+2}$ |          |

È detto **self-supervision** perché non usa etichette esterne, ma sfrutta il contesto delle parole all’interno del testo come se fosse un’etichetta. 😃

Il modello **Skip-gram** classico (e anche il CBOW) non cattura la posizione precisa delle parole nel contesto rispetto alla parola centrale, cioè non distingue se una parola del contesto sta a sinistra o a destra, o a quale distanza esatta.

## Obiettivo del modello

Vogliamo modellare la probabilità congiunta di osservare le parole di contesto data la parola centrale $w_t$, ossia:

$$\mathbb P(w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2} \mid w_t; \bm{\theta}) $$

Per semplicità si assume una **forte indipendenza condizionata** tra le parole di contesto dato il centro:

$$ \mathbb P(w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2} \mid w_t; \bm{\theta}) \approx \prod_{j=-m, j \neq 0}^{m} \mathbb P(w_{t+j} \mid w_t; \bm{\theta})$$

Questo significa che ogni parola di contesto è indipendente dalle altre data la parola centrale.

## Come si calcola $\mathbb P(w_{t+j}\mid w_t)$?

Dato un centro $w_t$, vogliamo predire la parola di contesto $w_{t+j}$. Questa probabilità è modellata come una distribuzione categorica su tutto il vocabolario $V$.

1. Prendiamo l'embedding della parola centro: se $i$ è l'indice di $w_t$ in $\bm{\theta}_W$, consideriamo il vettore riga $\bm{\theta}_W^i$ (di dimensione $1 \times |D|$).
2. Calcoliamo i punteggi (logits) per tutte le parole del vocabolario come prodotto scalare tra ogni vettore di contesto in $\bm{\theta}_C$ e l'embedding del centro:

   $$
   \underbrace{\mathbf{z}_i}_{|V|\times 1}=\overbrace{\underbrace{\bm{\theta}_C}_{|V|\times D}}^{\text{as context}}\cdot\overbrace{\underbrace{{\bm{\theta}_{W}^i}^T}_{D\times 1}}^{\text{as center}}
   $$

   dove $\mathbf{z}$ è un vettore di dimensione $|V|$, con ogni elemento che rappresenta la similarità (dot product) tra la parola centro e una possibile parola di contesto.

3. Applichiamo la funzione **softmax** ai logits per ottenere una distribuzione di probabilità:

  $$
  \mathbf{p}_i = \text{softmax}(\mathbf{z}_i) = \begin{bmatrix}
  p_1 \\
  \\
  \vdots \\
  \\
  p_{|V|}
  \\[0.45em]
  \end{bmatrix}= \begin{bmatrix}
  \mathbb P(w_{t+j} = \text{`apple`} | w_t = \text{`apricot`}) \\
  \\
  \vdots \\
  \\
  \mathbb P(w_{t+j} = \text{`zucchini`} | w_t = \text{`apricot`})
  \end{bmatrix}
  =  
  \Large\begin{bmatrix}
  \frac{e^{z_1}}{\sum_{i=1}^{|V|} e^{z_{i}}} \\
  \\
  \vdots \\
  \\
  \frac{e^{z_{|V|}}}{\sum_{i=1}^{|V|} e^{z_{i}}}
  \end{bmatrix}
  $$

Così otteniamo la probabilità di ogni parola del vocabolario come contesto dato il centro $w_t$.

**Remark.** L'indice della parola $w_t$ nella matrice $\bm{\theta}_W$ è $i$.

## Massimizzazione della likelihood su tutta la finestra

Per ogni parola centrale $w_t$, la probabilità congiunta di osservare tutte le parole di contesto nella finestra è:

$$
\prod_{j=-m, j \neq 0}^{m} \mathbb P(w_{t+j} | w_t; \bm{\theta})
$$

Il nostro obiettivo è trovare i parametri $\bm{\theta}$ che massimizzano la likelihood su tutto il corpus, ossia:

$$
\bm{\theta}^* = \arg\max_{\bm{\theta}} \prod_{t=1}^T \prod_{j=-m, j \neq 0}^m \mathbb P(w_{t+j} | w_t; \bm{\theta})
$$

## Funzione di perdita (loss) derivata dalla likelihood

L’obiettivo dell'addestramento è massimizzare la **likelihood** dei dati osservati, ovvero la probabilità di osservare le parole di contesto dato il centro, su tutto il corpus:

$$
L(\bm{\theta}) = \prod_{t=1}^T \prod_{j=-m, j \neq 0}^{m} \mathbb P(w_{t+j} \mid w_t; \bm{\theta})
$$

Lavorare direttamente con la likelihood può essere numericamente instabile, quindi passiamo al **logaritmo della likelihood** (log-likelihood), che è una trasformazione monotona e rende il prodotto una somma:

$$
\log L(\bm{\theta}) = \sum_{t=1}^T \sum_{j=-m, j \neq 0}^{m} \log \mathbb P(w_{t+j} \mid w_t; \bm{\theta})
$$

Il nostro obiettivo è quindi **massimizzare** questa log-likelihood:

$$
\bm{\theta}^* = \arg\max_{\bm{\theta}} \log L(\bm{\theta})
$$

In pratica, però, gli algoritmi di ottimizzazione numerica (come la discesa del gradiente) lavorano meglio se formuliamo il problema come **minimizzazione**. Per questo motivo, definiamo la **funzione di perdita** come l'opposto della log-likelihood:

$$
\mathcal{L}(\bm{\theta}) = - \sum_{t=1}^T \sum_{j=-m, j \neq 0}^{m} \log \mathbb P(w_{t+j} \mid w_t; \bm{\theta})
$$

Così facendo, possiamo minimizzare la funzione $\mathcal {L}$ per ottenere i parametri $\bm{\theta}^*$ che massimizzano la log-likelihood.

Possiamo ora esplicitare $\mathbb P(w_{t+j} \mid w_t)$ usando la softmax, come visto in precedenza. Supponiamo che:
- $\mathbf u_{w_t}$ sia l'embedding della parola centrale $w_t$, quindi la riga corrispondente a $w_t$ della matrice $\bm{\theta}_W$
- $\mathbf v_{w_{t+j}}$ sia l'embedding della parola di contesto $w_{t+j}$, quindi la riga corrispondente a $w_{t+j}$ della matrice $\bm{\theta}_C$

Allora la probabilità predetta dal modello è:

$$
\mathbb P(w_{t+j} \mid w_t)
= \frac{
    \exp\!\bigl(\mathbf{v}_{\,w_{t+j}}^\top \,\mathbf{u}_{\,w_t}\bigr)
  }{
    \displaystyle \sum_{w' \in V}
      \exp\!\bigl(\mathbf{v}_{\,w'}^\top \,\mathbf{u}_{\,w_t}\bigr)
  }
$$

Sostituendo nella funzione di perdita otteniamo:

$$
\mathcal{L}(\bm{\theta})
= - \sum_{t=1}^{T} \sum_{\substack{j=-m \\ j \neq 0}}^{m}
    \log
    \frac{
      \exp\!\bigl(\mathbf{v}_{\,w_{t+j}}^\top \,\mathbf{u}_{\,w_t}\bigr)
    }{
      \displaystyle \sum_{w' \in V}
        \exp\!\bigl(\mathbf{v}_{\,w'}^\top \,\mathbf{u}_{\,w_t}\bigr)
    }
$$

Applicando le proprietà del logaritmo, la loss per una singola coppia $(w_t, w_{t+j})$ diventa:

$$
\mathcal{L}(w_{t+j}, w_t; \bm{\theta}) = - \log
    \frac{
      \exp\!\bigl(\mathbf{v}_{\,w_{t+j}}^\top \,\mathbf{u}_{\,w_t}\bigr)
    }{
      \displaystyle \sum_{w' \in V}
        \exp\!\bigl(\mathbf{v}_{\,w'}^\top \,\mathbf{u}_{\,w_t}\bigr)
    }
$$

che si può riscrivere come:

$$
\mathcal{L}(w_{t+j}, w_t; \bm{\theta})
= -\,\underbrace{\mathbf{v}_{\,w_{t+j}}^\top \,\mathbf{u}_{\,w_t}}_\text{Similarità contesto-parola}
  \;+\;
  \underbrace{\log
  \sum_{w' \in V}
    \exp\!\bigl(\mathbf{v}_{\,w'}^\top \,\mathbf{u}_{\,w_t}\bigr)}_\text{Similarità di tutti gli altri contesti con la stessa parola}
$$

Questa formula evidenzia il trade-off tra massimizzare la similarità centro-contesto della parola corretta e normalizzare le probabilità su tutto il vocabolario.

Infine, la **loss media** su tutto il corpus è:

$$
\mathcal{L}(\bm{\theta})
= -\frac{1}{T}
  \sum_{t=1}^{T} \sum_{\substack{j=-m \\ j \neq 0}}^{m}
    \log \mathbb P(w_{t+j} \mid w_t; \bm{\theta})
= -\frac{1}{T}
  \sum_{t=1}^{T} \sum_{\substack{j=-m \\ j \neq 0}}^{m}
    \log
    \frac{
      \exp\!\bigl(\mathbf{v}_{\,w_{t+j}}^\top \,\mathbf{u}_{\,w_t}\bigr)
    }{
      \displaystyle \sum_{w' \in V}
        \exp\!\bigl(\mathbf{v}_{\,w'}^\top \,\mathbf{u}_{\,w_t}\bigr)
    }
$$

che coincide con la cross-entropy fra la distribuzione softmax predetta e la distribuzione one-hot vera.

## Ottimizzazione tramite SGD

🧠 *Prima di continuare, vedi la nota dedicata sul funzionamento dello SGD: [[Discesa del Gradiente]].*

L’addestramento del modello Skip-gram con softmax consiste nell’ottimizzare i parametri $\bm{\theta} = \begin{bmatrix} \bm{\theta}_W \\ \bm{\theta}_C \end{bmatrix}$ per massimizzare la probabilità delle parole di contesto osservate, dato ciascun centro $w_t$ nel corpus.

L’obiettivo è **minimizzare la loss media** dei dati, ovvero la somma della log-probabilità dei contesti osservati dato ogni parola centrale, moltiplicata per $-\frac{1}{T}$. Formalmente:

$$
\mathcal{L}(\bm{\theta}) = -\frac{1}{T} \sum_{t=1}^T \sum_{\substack{j = -m \\ j \ne 0}}^m \log \mathbb{P}(w_{t+j} \mid w_t; \bm{\theta})
$$

dove:

- $T$ è il numero totale di parole nel corpus,
- $m$ è l'ampiezza della finestra di contesto,
- $\mathbb{P}(w_{t+j} \mid w_t; \bm{\theta})$ è la probabilità (softmax) di osservare $w_{t+j}$ dato il centro $w_t$, definita come:

$$
\mathbb{P}(w_{t+j} \mid w_t; \bm{\theta}) = \frac{\exp\left( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} \right)}{\sum_{k=1}^{|V|} \exp\left( \mathbf v_{w_k} \cdot \mathbf u_{w_t} \right)}
$$

con:

- $w_t$: parola centrale (indice $c$),
- $w_{t+j}$: parola di contesto (indice $o$),
- $\mathbf u_{w_t} \in \mathbb{R}^D$: vettore embedding della parola centro $w_t$,
- $\mathbf v_{w_{t+j}} \in \mathbb{R}^D$: vettore embedding della parola contesto $w_{t+j}$.

### Come si ottimizza?

Poiché la somma al denominatore del softmax scorre su tutto il vocabolario ($|V|$ è molto grande), il calcolo diretto è troppo costoso. Tuttavia, per ora assumiamo di usare il **softmax esatto**, per chiarezza.

Il modello viene ottimizzato tramite **Stochastic Gradient Descent (SGD)**, cioè:

1. Si considera una coppia $(w_t, w_{t+j})$ (parola centro + parola di contesto osservata),
2. Si calcola la **loss negativa log-likelihood** per quella coppia:

$$
\mathcal{L}(w_{t+j}, w_t; \bm{\theta}) = -\log \mathbb{P}(w_{t+j} \mid w_t; \bm{\theta}) = -\log \frac{\exp\left( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} \right)}{\sum_{k=1}^{|V|} \exp\left( \mathbf v_{w_k} \cdot \mathbf u_{w_t} \right)}
$$

1. Si calcola il **gradiente** della loss rispetto a $\bm{\theta}$,
2. Si aggiorna $\bm{\theta}$ secondo la regola standard dello SGD:

$$
\bm{\theta} \leftarrow \bm{\theta} - \eta \cdot \nabla_{\bm{\theta}}\mathcal{L}(w_{t+j}, w_t; \bm{\theta})
$$

dove $\eta$ è il learning rate.

### Calcolo del gradiente

Calcoliamo ora il gradiente della funzione di loss rispetto ai vettori di embedding coinvolti, assumendo sempre l’uso del softmax esatto.

Fissiamo una singola coppia $(w_t, w_{t+j})$, cioè una parola centrale e una parola di contesto. La loss associata a questa coppia è:

$$
\underbrace{\mathcal{L}_{(t,j)}}_{\mathcal{L}(w_{t+j}, w_t; \bm{\theta})} = -\log \mathbb{P}(w_{t+j} \mid w_t; \bm{\theta})
= -\log \left( \frac{\exp\left( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} \right)}{\sum_{k=1}^{|V|} \exp\left( \mathbf v_{w_k} \cdot \mathbf u_{w_t} \right)} \right)
$$

Dove:

- $\mathbf u_{w_t} \in \mathbb{R}^D$: vettore della parola **centro** (da $\bm{\theta}_W$),
- $\mathbf v_{w_k} \in \mathbb{R}^D$: vettori delle parole **contesto** (da $\bm{\theta}_C$),
- $|V|$: dimensione del vocabolario.

#### Gradiente rispetto al vettore della parola centro $\mathbf u_{w_t}$

Vogliamo calcolare il gradiente della loss rispetto al vettore centro $\mathbf u_{w_t}$ per la coppia $(w_t, w_{t+j})$:

$$
\nabla_{\mathbf u_{w_t}} \mathcal{L}_{(t,j)} =
- \nabla_{\mathbf u_{w_t}} \left( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t}
- \log \sum_{k=1}^{|V|} \exp\left( \mathbf v_{w_k} \cdot \mathbf u_{w_t} \right) \right)
$$

1. **Derivata del primo termine** (prodotto scalare):

  $$
  \nabla_{\mathbf u_{w_t}} \left( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} \right)
  = \mathbf v_{w_{t+j}}
  $$

  Motivo: la derivata di un prodotto scalare $\mathbf a^\top \mathbf x$ rispetto a $\mathbf x$ è $\mathbf a$.

2. **Derivata del secondo termine** (log-somma-esponenziali + chain rule):

  $$
  \nabla_{\mathbf u_{w_t}} \left( \log \sum_{k=1}^{|V|} \exp( \mathbf v_{w_k} \cdot \mathbf u_{w_t} ) \right)
  $$

  **Passo 1** – Applichiamo la derivata del logaritmo:

  $$
  \nabla_{\mathbf u_{w_t}} \log f(\mathbf u_{w_t}) = \frac{1}{f(\mathbf u_{w_t})} \cdot \nabla_{\mathbf u_{w_t}} f(\mathbf u_{w_t})
  $$

  Dove $f(\mathbf u_{w_t}) = \sum_{k=1}^{|V|} \exp( \mathbf v_{w_k} \cdot \mathbf u_{w_t} )$

  **Passo 2** – Derivata della somma:

  $$
  \nabla_{\mathbf u_{w_t}} \sum_{k=1}^{|V|} \exp( \mathbf v_{w_k} \cdot \mathbf u_{w_t} ) = \sum_{k=1}^{|V|} \exp( \mathbf v_{w_k} \cdot \mathbf u_{w_t} ) \cdot \mathbf v_{w_k}
  $$

  **Passo 3** – Mettiamo tutto insieme:

  $$
  \nabla_{\mathbf u_{w_t}} \log \sum_{k=1}^{|V|} \exp( \mathbf v_{w_k} \cdot \mathbf u_{w_t} )
  = \frac{ \sum_{k=1}^{|V|} \exp( \mathbf v_{w_k} \cdot \mathbf u_{w_t} ) \cdot \mathbf v_{w_k} }
  { \sum_{k=1}^{|V|} \exp( \mathbf v_{w_k} \cdot \mathbf u_{w_t} ) }
  = \sum_{k=1}^{|V|} \mathbb{P}(w_k \mid w_t) \cdot \mathbf v_{w_k}
  $$

  dove:

  $$
  \mathbb{P}(w_k \mid w_t) = \frac{\exp(\mathbf v_{w_k} \cdot \mathbf u_{w_t})}{\sum_{j=1}^{|V|} \exp(\mathbf v_{w_j} \cdot \mathbf u_{w_t})}
  $$

  **Combinazione** dei due termini:

  $$
  \nabla_{\mathbf u_{w_t}} \mathcal{L}_{(t,j)}
  = - \left( \mathbf v_{w_{t+j}} - \sum_{k=1}^{|V|} \mathbb{P}(w_k \mid w_t) \cdot \mathbf v_{w_k} \right)
  $$


#### Gradiente rispetto al vettore contesto corretto $\mathbf v_{w_{k}}$ con $k = t + j$

Calcoliamo:

$$
\nabla_{\mathbf v_{w_{t+j}}} \mathcal{L}_{(t,j)} =
- \nabla_{\mathbf v_{w_{t+j}}} \left( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t}
- \log \sum_{i=1}^{|V|} \exp( \mathbf v_{w_i} \cdot \mathbf u_{w_t} ) \right)
$$


1. **Derivata del primo termine**:

$$
\nabla_{\mathbf v_{w_{t+j}}} \left( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} \right)
= \mathbf u_{w_t}
$$


2. **Derivata del secondo termine**:

Solo il termine $k = t+j$ dipende da $\mathbf v_{w_{t+j}}$, ma deriviamo comunque la somma intera, trattando ogni termine:

$$
\nabla_{\mathbf v_{w_{t+j}}} \log \left( \sum_{k=1}^{|V|} \exp( \mathbf v_{w_k} \cdot \mathbf u_{w_t} ) \right)
= \frac{1}{\sum_{k=1}^{|V|} \exp( \mathbf v_{w_k} \cdot \mathbf u_{w_t} )} \cdot 
\nabla_{\mathbf v_{w_{t+j}}} \left( \sum_{k=1}^{|V|} \exp( \mathbf v_{w_k} \cdot \mathbf u_{w_t} ) \right)
$$

Solo il termine $k = t+j$ sopravvive:

$$
\nabla_{\mathbf v_{w_{t+j}}} \left( \sum_{k=1}^{|V|} \exp( \mathbf v_{w_k} \cdot \mathbf u_{w_t} ) \right) 
= \nabla_{\mathbf v_{w_{t+j}}} \left( \exp( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} ) \right)
= \exp( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} ) \cdot \mathbf u_{w_t}.
$$

Mettendo quindi tutto insieme otteniamo:

$$
\nabla_{\mathbf v_{w_{t+j}}} \log \left( \sum_{k=1}^{|V|} \exp( \mathbf v_{w_k} \cdot \mathbf u_{w_t} ) \right)
=
\frac{ \exp( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} ) \cdot \mathbf u_{w_t} }
{ \sum_{k=1}^{|V|} \exp( \mathbf v_{w_k} \cdot \mathbf u_{w_t} ) }
= \mathbb{P}(w_{t+j} \mid w_t) \cdot \mathbf u_{w_t}
$$

**Combinazione** dei due termini:

$$
\nabla_{\mathbf v_{w_{t+j}}} \mathcal{L}_{(t,j)}
= - \left( \mathbf u_{w_t} - \mathbb{P}(w_{t+j} \mid w_t) \cdot \mathbf u_{w_t} \right)
= \left( \mathbb{P}(w_{t+j} \mid w_t) - 1 \right) \cdot \mathbf u_{w_t}
$$

#### Gradiente rispetto agli altri vettori contesto $\mathbf v_{w_k}$ con $k \ne t+j$

Sia la loss per la coppia $(w_t, w_{t+j})$:

$$
\mathcal{L}_{(t,j)}
= -\Bigl(\mathbf v_{w_{t+j}}\!\cdot\!\mathbf u_{w_t}\Bigr)
  + \log \sum_{i=1}^{|V|} \exp\!\bigl(\mathbf v_{w_i}\!\cdot\!\mathbf u_{w_t}\bigr).
$$

Vogliamo calcolare 
$\nabla_{\mathbf v_{w_k}} \mathcal{L}_{(t,j)}$
per un indice $k\neq t+j$.

1. **Derivata del primo termine**  
   
   Il **primo termine** dipende **solo** da $\mathbf v_{w_{t+j}}$, non da $\mathbf v_{w_k}$ quando $k\ne t+j$.  
   
   $$
   \nabla_{\mathbf v_{w_k}}
   \bigl(\mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t}\bigr)
   = 0
   \quad\text{per }k \ne t+j.
   $$

2. **Derivata del secondo termine**  

   Il **secondo termine** è
   $$
   F(\mathbf v_{w_i})
   = \log \sum_{i=1}^{|V|} \exp\!\bigl(\mathbf v_{w_i} \cdot \mathbf u_{w_t}\bigr).
   $$
   
   - **Passo 2.1**: applichiamo la derivata del logaritmo:
     $$
     \nabla_{\mathbf v_{w_k}}\,F
     = \frac{1}{\displaystyle \sum_{i=1}^{|V|} \exp(\mathbf v_{w_i}\!\cdot\!\mathbf u_{w_t})}
       \;\nabla_{\mathbf v_{w_k}}
       \sum_{i=1}^{|V|} \exp(\mathbf v_{w_i}\!\cdot\!\mathbf u_{w_t}).
     $$
   
   - **Passo 2.2**: derivata della somma di esponenziali. In questa somma, ogni termine indice $i$ è
     $\exp(\mathbf v_{w_i}\!\cdot\!\mathbf u_{w_t})$. Solo quando $i=k$ l’esponenziale dipende da $\mathbf v_{w_k}$.  
     
     $$
     \nabla_{\mathbf v_{w_k}}
     \sum_{i=1}^{|V|} \exp(\mathbf v_{w_i}\!\cdot\!\mathbf u_{w_t})
     = \nabla_{\mathbf v_{w_k}}
       \exp(\mathbf v_{w_k}\!\cdot\!\mathbf u_{w_t})
     = \exp(\mathbf v_{w_k}\!\cdot\!\mathbf u_{w_t}) \;\mathbf u_{w_t}.
     $$
   
   - **Passo 2.3**: sostituiamo nella regola del log:
     $$
     \nabla_{\mathbf v_{w_k}}\,F
     = \frac{\exp(\mathbf v_{w_k}\!\cdot\!\mathbf u_{w_t}) \;\mathbf u_{w_t}}
            {\displaystyle \sum_{i=1}^{|V|} \exp(\mathbf v_{w_i}\!\cdot\!\mathbf u_{w_t})}
     = \mathbb{P}(w_k \mid w_t)\;\mathbf u_{w_t}.
     $$

3. **Combinazione dei termini**  

   Sommando le due derivazioni (primo termine zero + secondo termine):

   $$
   \nabla_{\mathbf v_{w_k}} \mathcal{L}_{(t,j)}
   = 0 + \mathbb{P}(w_k \mid w_t)\;\mathbf u_{w_t}
   = \mathbb{P}(w_k \mid w_t)\;\mathbf u_{w_t}.
   $$


#### Riassunto aggiornamenti

Per ogni coppia $(w_t, w_{t+j})$, aggiorniamo:

- Il vettore **centro** $\mathbf u_{w_t}$ secondo:

  $$
  \mathbf u_{w_t} \leftarrow \mathbf u_{w_t} - \eta \cdot \nabla_{\mathbf u_{w_t}} \mathcal{L}_{(t,j)}
  $$

- Il vettore **contesto corretto** $\mathbf v_{w_{t+j}}$ secondo:

  $$
  \mathbf v_{w_{t+j}} \leftarrow \mathbf v_{w_{t+j}} - \eta \cdot \nabla_{\mathbf v_{w_{t+j}}} \mathcal{L}_{(t,j)}
  $$

- Gli altri vettori **contesto** $\mathbf v_{w_k}$ con $k \ne t+j$, opzionalmente:

  $$
  \mathbf v_{w_k} \leftarrow \mathbf v_{w_k} - \eta \cdot \nabla_{\mathbf v_{w_k}} \mathcal{L}_{(t,j)}
  $$

In pratica, si usa **Negative Sampling** per evitare l'aggiornamento su tutto il vocabolario.

### Negative Sampling

L’obiettivo del **Negative Sampling** è approssimare in modo efficiente la funzione di perdita originale basata sulla softmax, che richiede una somma su tutto il vocabolario $|V|$ — troppo costosa per vocabolari grandi.

Invece di calcolare la probabilità normalizzata per tutte le parole, si trasforma il problema in una **serie di classificazioni binarie**.

#### Strategia

- La coppia **positiva** $(w_t, w_{t+j})$ (parola centrale e parola di contesto reale) è trattata come un esempio **positivo**, con **target = 1**.
- Si campionano $K$ parole **negative** $w_1', \dots, w_K'$ da una distribuzione rumorosa (noise distribution), e si trattano come esempi **negativi**, con **target = 0**.

#### Notazione

- $\mathbf u_{w_t}$: vettore embedding della parola centrale (input)
- $\mathbf v_{w_{t+j}}$: embedding della parola di contesto positiva (output)
- $\mathbf v_{w_k'}$: embedding delle parole negative
- $\sigma(x) = \frac{1}{1 + e^{-x}}$: funzione sigmoide

#### Loss per una singola coppia $(w_t, w_{t+j})$ e $K$ parole negative:

$$
\mathcal{L}_{\text{NS}}^{(t,j)} =
- \log \sigma( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} )
- \sum_{k=1}^K \log \sigma( - \mathbf v_{w_k'} \cdot \mathbf u_{w_t} )
$$

- Il primo termine spinge $\mathbf u_{w_t}$ e $\mathbf v_{w_{t+j}}$ ad avere un **prodotto scalare alto**, quindi un’alta probabilità.
- Il secondo termine penalizza $\mathbf u_{w_t}$ e i vettori negativi $\mathbf v_{w_k'}$ se il loro prodotto scalare è troppo alto.

### Calcolo dei Gradienti

La loss per una singola coppia $(w_t, w_{t+j})$ e $K$ parole negative è:

$$
\mathcal{L}_{\text{NS}}^{(t,j)} =
- \log \sigma( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} )
- \sum_{k=1}^K \log \sigma( - \mathbf v_{w_k'} \cdot \mathbf u_{w_t} )
$$

Dove:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$
- La derivata della sigmoide: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

#### 1. Derivata rispetto a $\mathbf u_{w_t}$ (embedding della parola centrale)

Partiamo dalla derivata della loss rispetto a $\mathbf u_{w_t}$:

$$
\nabla_{\mathbf u_{w_t}} \mathcal{L}_{\text{NS}}^{(t,j)} =
\frac{\partial}{\partial \mathbf u_{w_t}} \left(
- \log \sigma( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} )
- \sum_{k=1}^K \log \sigma( - \mathbf v_{w_k'} \cdot \mathbf u_{w_t} )
\right)
$$

##### Primo termine (positivo):

$$
\frac{\partial}{\partial \mathbf u_{w_t}} \left[ - \log \sigma( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} ) \right]
$$

Applichiamo la chain rule:

1. $x = \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t}$  
2. $\frac{d}{dx}[-\log \sigma(x)] = - \frac{\sigma'(x)}{\sigma(x)} = - (1 - \sigma(x))$

3. $\frac{\partial x}{\partial \mathbf u_{w_t}} = \mathbf v_{w_{t+j}}$

Quindi:

$$
= - (1 - \sigma( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} )) \cdot \mathbf v_{w_{t+j}} 
= ( \sigma( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} ) - 1 ) \cdot \mathbf v_{w_{t+j}}
$$

##### Secondo termine (negativi):

Ogni termine nella somma:

$$
\frac{\partial}{\partial \mathbf u_{w_t}} \left[ - \log \sigma( - \mathbf v_{w_k'} \cdot \mathbf u_{w_t} ) \right]
$$

Applichiamo la regola della catena:

1. $x = - \mathbf v_{w_k'} \cdot \mathbf u_{w_t}$  
2. $\sigma(x)' = \sigma(x)(1 - \sigma(x))$  
3. $\frac{d}{dx}[-\log \sigma(x)] = - (1 - \sigma(x))$

Ma attenzione: deriviamo rispetto a $\mathbf u_{w_t}$, quindi:

$$
\frac{\partial}{\partial \mathbf u_{w_t}} \left[ - \log \sigma( - \mathbf v_{w_k'} \cdot \mathbf u_{w_t} ) \right]
= \sigma( \mathbf v_{w_k'} \cdot \mathbf u_{w_t} ) \cdot \mathbf v_{w_k'}
$$

Perché:

- $\sigma(-x) = 1 - \sigma(x)$
- $\frac{d}{dx}[-\log(1 - \sigma(x))] = \sigma(x)$

Sommiamo su $k$:

$$
\sum_{k=1}^K \sigma( \mathbf v_{w_k'} \cdot \mathbf u_{w_t} ) \cdot \mathbf v_{w_k'}
$$

##### Totale:

$$
\nabla_{\mathbf u_{w_t}} \mathcal{L}_{\text{NS}}^{(t,j)} =
( \sigma( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} ) - 1 ) \cdot \mathbf v_{w_{t+j}} +
\sum_{k=1}^K \sigma( \mathbf v_{w_k'} \cdot \mathbf u_{w_t} ) \cdot \mathbf v_{w_k'}
$$

#### 2. Derivata rispetto a $\mathbf v_{w_{t+j}}$ (embedding del contesto positivo)

$$
\frac{\partial}{\partial \mathbf v_{w_{t+j}}} \left[ - \log \sigma( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} ) \right]
$$

Stesso ragionamento:

1. $x = \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t}$
2. $\frac{d}{dx}[-\log \sigma(x)] = - (1 - \sigma(x))$

Derivata rispetto a $\mathbf v_{w_{t+j}}$:

$$
= ( \sigma( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} ) - 1 ) \cdot \mathbf u_{w_t}
$$

Nella somma non compare l'embedding del contesto positivo. Quindi la derivata rispetto a $\mathbf v_{w_{t+j}}$ è nulla.

#### 3. Derivata rispetto ad ogni $\mathbf v_{w_k'}$ (embedding delle parole negative)

$$
\frac{\partial}{\partial \mathbf v_{w_k'}} \left[ - \log \sigma( - \mathbf v_{w_k'} \cdot \mathbf u_{w_t} ) \right]
$$

Come sopra:

- $\sigma(-x)' = - \sigma(x)(1 - \sigma(x))$
- $- \log \sigma(-x) = - \log (1 - \sigma(x))$

Quindi:

$$
= \sigma( \mathbf v_{w_k'} \cdot \mathbf u_{w_t} ) \cdot \mathbf u_{w_t}
$$


#### Riepilogo dei Gradienti

- **Parola centrale $\mathbf u_{w_t}$**:

$$
\nabla_{\mathbf u_{w_t}} \mathcal{L}_{\text{NS}}^{(t,j)}=
( \sigma( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} ) - 1 ) \cdot \mathbf v_{w_{t+j}} 
+ \sum_{k=1}^K \sigma( \mathbf v_{w_k'} \cdot \mathbf u_{w_t} ) \cdot \mathbf v_{w_k'}
$$

- **Parola positiva $\mathbf v_{w_{t+j}}$**:

$$
\nabla_{\mathbf v_{w_{t+j}}} \mathcal{L}_{\text{NS}}^{(t,j)} =
( \sigma( \mathbf v_{w_{t+j}} \cdot \mathbf u_{w_t} ) - 1 ) \cdot \mathbf u_{w_t}
$$

- **Parola negativa $\mathbf v_{w_k'}$**:

$$
\nabla_{\mathbf v_{w_k'}} \mathcal{L}_{\text{NS}}^{(t,j)} =
\sigma( \mathbf v_{w_k'} \cdot \mathbf u_{w_t} ) \cdot \mathbf u_{w_t}
$$

### Intuizione Finale

- Il gradiente rispetto alla parola **positiva** cerca di **avvicinare** gli embedding $\mathbf u_{w_t}$ e $\mathbf v_{w_{t+j}}$.
- I gradienti rispetto alle parole **negative** cercano di **allontanare** $\mathbf u_{w_t}$ da $\mathbf v_{w_k'}$, se sono troppo simili.
- In questo modo, la rete impara a **distinguerle** — creando spazi semantici utili per rappresentare significato e contesto.

### Perché il Negative Sampling funziona?

In un training tradizionale con softmax, ogni parola nel vocabolario è considerata in ogni update: inefficiente e inutile, perché la maggior parte delle parole **non sono rilevanti** nel contesto dato.

Il negative sampling funziona bene **anche campionando solo poche parole negative** perché:

- La maggior parte delle parole nel vocabolario **non appaiono nel contesto locale**. È sufficiente penalizzarne alcune per rappresentare questo "mare di parole irrilevanti".
- L’aggiornamento stocastico su $K$ parole negative scelte a caso **approssima il gradiente medio** su tutte le parole negative.
- Il modello impara a **differenziare le parole "giuste" da quelle "sbagliate"**, non a predire ogni parola nel vocabolario.
- Inoltre, campionando le negative da una distribuzione “disturbata” (es. proporzionale a $P(w)^{3/4}$), si aumenta l’efficacia dei campioni più informativi.

➡️ **In sintesi**: invece di imparare su tutto il vocabolario, impariamo da un campione ben scelto. L'efficienza migliora enormemente senza perdita significativa in qualità. Perché, in effetti, ad ogni iterazione ci interessa molto di più la relazione tra parola **centro** e **contesto** che quella tra parola **centro** e parole **non-contesto**.

### Negative Sampling vs Softmax: differenze chiave

Con il **Negative Sampling** non si calcola più una vera distribuzione di probabilità normalizzata su tutto il vocabolario, come avviene con la softmax classica.

- **Softmax classico:**  
  Calcola la probabilità che una parola sia nel contesto dato il centro, considerando *tutte* le parole del vocabolario. Questo è costoso ma produce una distribuzione completa.

- **Negative Sampling:**  
  Trasforma il problema in una serie di classificazioni binarie:  
  - Le coppie (parola centro, parola contesto reale) sono esempi positivi.  
  - Le coppie con parole negative campionate casualmente sono esempi negativi.  

Il modello impara a distinguere parole di contesto “vere” da parole “false”, ma non produce una distribuzione completa su tutte le parole.

Con Negative Sampling, il modello cerca invece una funzione che spinge gli embeddings di coppie (centro, positivo) a essere simili (dot product alto) e quelli (centro, negativi) a essere dissimili (dot product basso).

**In pratica:**  
Negative Sampling ottimizza l’efficienza concentrandosi solo su alcune parole negative per update, ma non fornisce probabilità normalizzate su tutto il vocabolario come la softmax.

Se servono probabilità vere, si usano softmax o sue varianti (hierarchical softmax), ma a costo computazionale maggiore.

### Effetto dell’ottimizzazione

Iterando su molte coppie $(w_t, w_{t+j})$ osservate dal corpus, il modello:

- rafforza le associazioni tra centri e contesti frequenti (es. “eat” → “food”),
- indebolisce associazioni tra parole che non co-occorrono.

Alla convergenza, gli embedding $\bm{\theta}_W$ e $\bm{\theta}_C$ riflettono **strutture semantiche** e **sintattiche** apprese dai dati: parole con significati simili finiscono in regioni vicine dello spazio vettoriale.

## Conclusioni

Il modello **Skip-gram con softmax** rappresenta un approccio fondamentale nell'ambito dell'apprendimento non supervisionato per la rappresentazione distribuita delle parole. Utilizzando due matrici distinte — una per le parole *centro* e una per le parole *contesto* — il modello riesce a catturare in modo più preciso le relazioni semantiche e sintattiche nel linguaggio naturale.

Questa separazione consente di modellare efficacemente le **asimmetrie** e i **ruoli funzionali** delle parole, migliorando la qualità degli embedding e le prestazioni in numerosi compiti downstream come il POS tagging, il parsing o il semantic similarity.

La formulazione probabilistica basata su **softmax** permette di interpretare le previsioni come distribuzioni categoriali su tutto il vocabolario, sebbene a un costo computazionale elevato. Questo ha motivato lo sviluppo di tecniche più efficienti come il **Negative Sampling** e la **Hierarchical Softmax**, che estendono il framework Skip-gram per corpus di grandi dimensioni.

### Risorse utili e approfondimenti

- Dan Jurafsky & James H. Martin, *Speech and Language Processing*, 3rd Edition (draft):  
  https://web.stanford.edu/~jurafsky/slp3/  

- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space*  
  [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)

- Goldberg & Levy (2014), *word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method*  
  [https://arxiv.org/abs/1402.3722](https://arxiv.org/abs/1402.3722)

- TensorFlow Tutorial: *Word2Vec Skip-gram*  
  [https://www.tensorflow.org/tutorials/text/word2vec](https://www.tensorflow.org/tutorials/text/word2vec)

- Chris McCormick, *Word2Vec Tutorial* (con codice e spiegazioni passo-passo)  
  [https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

- Blog di Jay Alammar, *The Illustrated Word2Vec*  
  [https://jalammar.github.io/illustrated-word2vec/](https://jalammar.github.io/illustrated-word2vec/)

Questa panoramica costituisce la base concettuale per affrontare estensioni più sofisticate e ottimizzazioni del modello, fondamentali per lavorare con corpus molto ampi o con vocabolari di grandi dimensioni.

