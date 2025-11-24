# Swin Transformer: Shifted Window Transformers

## Introduzione

Lo **Swin Transformer** (Shifted Window Transformer) è un'architettura di deep learning progettata per elaborare immagini in modo efficiente. A differenza dei Vision Transformer tradizionali che applicano l'attenzione su tutti i pixel dell'immagine contemporaneamente (operazione molto costosa computazionalmente), Swin Transformer utilizza una strategia intelligente basata su **finestre locali** che vengono spostate tra i layer per catturare sia informazioni locali che globali.

Immaginate di osservare un'immagine attraverso piccole finestre: prima guardate regioni locali, poi spostate leggermente le finestre per vedere connessioni tra diverse regioni. Questa è l'essenza di Swin Transformer.

In generale, Swin Transformer trasforma un'immagine in una sequenza di token (come in NLP) e le passa attraverso un'architettura di deep learning, in modo da ottenere rappresentazioni gerarchiche di immagini in un embedding 1D.

## Architettura Generale

L'architettura di Swin Transformer si compone di quattro fasi principali, ognuna con una risoluzione spaziale progressivamente ridotta (simile alle CNN), creando una **gerarchia di rappresentazioni**.

<img src="https://user-images.githubusercontent.com/24825165/121768619-038e6d80-cb9a-11eb-8cb7-daa827e7772b.png" alt="Immagine di un Transformer" style="display: block; margin-left: auto; margin-right: auto;">

### Input e Parametri

Consideriamo un'immagine di input:

$$
\mathbf{X}_{\text{input}} \in \mathbb{R}^{B \times 3 \times H_0 \times W_0}
$$

dove:
- $B$ = dimensione del batch (numero di immagini elaborate insieme)
- $3$ = numero di canali (RGB)
- $H_0 = W_0 = 224$ (tipicamente, dimensione immagine standard)

## Fase 1: Patch Partition and Embedding

### Suddivisione in Patch

Il primo passo consiste nel suddividere l'immagine in patch non sovrapposte. Ogni patch viene trattata come un "token" (simile alle parole in NLP).

**Parametri:**
- Dimensione patch: $P = 4$ (tipicamente)
- Dimensione embedding: $C = 96$ (tipicamente)

**Operazione:**

L'immagine viene suddivisa usando una convoluzione 2D:

$$
\text{Conv2D}: \mathbb{R}^{B \times 3 \times 224 \times 224} \rightarrow \mathbb{R}^{B \times 96 \times 56 \times 56}
$$

Parametri della convoluzione:
- Kernel size: $4 \times 4 \implies K = 4$
- Stride: $S = 4$
- Output channels: $C_{out} = 96$


Questo produce:

$$
H_1 = \frac{H_0}{P} = \frac{224}{4} = 56
$$

$$
W_1 = \frac{W_0}{P} = \frac{224}{4} = 56
$$

La convoluzione 2D che genera gli embedding è definita come:

$$
Y_{c_{\text{out}}, h, w} =
\sum_{c_{\text{in}}=1}^{C_{\text{in}}}
\sum_{i=0}^{K-1} \sum_{j=0}^{K-1}
W_{c_{\text{out}}, c_{\text{in}}, i, j} \cdot
X_{c_{\text{in}},\, h \cdot S + i,\, w \cdot S + j}
+ b_{c_{\text{out}}} = \sum_{d=1}^{3} \sum_{r=0}^{3} \sum_{s=0}^{3}
W_{c_{out},d,r,s} \cdot X_{d,\, h\cdot4 + r,\, w\cdot4 + s} + b_{c_{out}}
$$

Dove:

- $c_{out} \in \{1, \ldots, C_{out}\}$, $h \in \{1, \ldots, H_1\}$, $w \in \{1, \ldots, W_1\}$
- $W \in \mathbb{R}^{96 \times 3 \times 4 \times 4}$ dove ogni $W_c$ è un **filtro** che rileva un pattern visivo locale (bordi, texture, colori…)
- I valori di $W$ **sono parametri appresi** durante il training mediante backpropagation

Il numero totale di patch è:

$$
N = H_1 \times W_1 = 56 \times 56 = 3136
$$

**Reshape per elaborazione:**

Il tensore viene riorganizzato da formato spaziale a sequenza:

$$
\mathbf{X}_{\text{embed}} \in \mathbb{R}^{B \times 96 \times 56 \times 56} \rightarrow \mathbb{R}^{B \times 3136 \times 96}
$$

Questo permette di elaborare i patch come sequenze di token. È quindi più comodo per applicare LayerNorm / dropout / pos embedding.

**Normalizzazione (opzionale):**

Se `patch_norm=True`, si applica [[Layer Normalization]]:

$$
\mathbf{X}_{\text{norm}} = \text{LayerNorm}(\mathbf{X}_{\text{embed}})
$$

con $\mathbf{X}_{\text{norm}} \in \mathbb{R}^{B \times 3136 \times 96}$

### [[Absolute Position Embedding]] (opzionale)

Se il parametro `ape=True` è attivato, viene aggiunto un embedding posizionale assoluto:

$$
\mathbf{X}_{\text{pos}} = \mathbf{X}_{\text{norm}} + \mathbf{E}_{\text{abs}}
$$

dove $\mathbf{E}_{\text{abs}} \in \mathbb{R}^{1 \times 3136 \times 96}$ è un parametro apprendibile che codifica la posizione di ogni patch nell'immagine.

Dimensione finale dopo dropout:

$$
\mathbf{X}_0 = \text{Dropout}(\mathbf{X}_{\text{pos}}) \in \mathbb{R}^{B \times 3136 \times 96}
$$

**Remark:** Il Dropout ovviamente viene appplicato solamente durante la fase di training.

## Stage 1: Primo Livello della Gerarchia

Lo Stage 1 processa le patch con la massima risoluzione spaziale.

### Parametri dello Stage 1

- Risoluzione input: $H_1 \times W_1 = 56 \times 56 = 3136$
- Dimensione canali: $C_1 = 96$
- Numero di blocchi: $\text{depth}_1 = 2$
- Numero di head di attenzione: $\text{heads}_1 = 3$
- Dimensione finestra: $M = 7$

### Swin Transformer Block

Ogni stage contiene una sequenza di **Swin Transformer Blocks**. Ogni blocco alterna tra:
1. **W-MSA** (Window-based Multi-head Self Attention)
2. **SW-MSA** (Shifted Window-based Multi-head Self Attention)

<img src="https://www.researchgate.net/publication/375432550/figure/fig3/AS:11431281212044866@1702536170969/Two-consecutive-Swin-transformer-blocks.tif" alt="Swin Transformer Block" style="display: block; margin-left: auto; margin-right: auto; height: 500px;">

## W-MSA: Window-based Multi-head Self Attention

### Concetto Base

Invece di calcolare l'attenzione su tutte le $N = 3136$ patch (operazione $O(N^2)$), W-MSA divide l'immagine in **finestre non sovrapposte** di dimensione $M \times M$ e calcola l'attenzione **localmente** all'interno di ogni finestra.

### Partizionamento in Finestre

**Input al blocco:**

$$
\mathbf{X} \in \mathbb{R}^{B \times (H \times W) \times C}
$$

Per lo Stage 1: $H = W = 56$, $C = 96$

**Reshape spaziale:**

Dato che per calcolare l'attenzione si lavora su finestre di dimensione $M \times M$, l'immagine viene riorganizzata da sequenza di patch a "immagine" di patch. In questo modo, è possibile calcolare l'attenzione localmente.

$$
\mathbf{X} \rightarrow \mathbb{R}^{B \times H \times W \times C} = \mathbb{R}^{B \times 56 \times 56 \times 96}
$$

**Partizionamento:**

L'immagine viene divisa in finestre di dimensione $M \times M$ (con $M = 7$):

$$
\text{num\_windows} = \frac{H}{M} \times \frac{W}{M} = \frac{56}{7} \times \frac{56}{7} = 8 \times 8 = 64
$$

Ogni finestra contiene:

$$
M^2 = 7 \times 7 = 49 \text{ patch}
$$

**Reshape per l'attenzione:**

Dopo aver suddiviso l'immagine in finestre per il **Window-based Multi-Head Self Attention (W-MSA)**, otteniamo:


$$
\mathbf{X}_{\text{windows}} \in \mathbb{R}^{(B \times 64) \times 49 \times 96}
$$

**Perché `(B × 64)`?**  

- Ogni finestra è trattata come un piccolo "mini-batch" indipendente.  
- Appiattendo batch e finestre in una singola dimensione `(B × 64)`, possiamo **calcolare l'attenzione su tutte le finestre in parallelo** senza fare loop immagine per immagine.  
- In pratica: tutti gli esempi del batch e tutte le finestre diventano un unico batch “virtuale” per l’attenzione.

### Layer Normalization

Prima dell'attenzione, si applica normalizzazione:

$$
\mathbf{X}_{\text{norm}} = \text{LayerNorm}(\mathbf{X}_{\text{windows}})
$$

$$
\mathbf{X}_{\text{norm}} \in \mathbb{R}^{(B \times 64) \times 49 \times 96}
$$

### Multi-Head Self Attention

**Numero di head:** $3$ (per Stage 1)

**Dimensione per head:**

$$
d_h = \frac{C}{\text{heads}} = \frac{96}{3} = 32
$$

**Proiezioni QKV:**

Le query, key e value vengono generate con una proiezione lineare:

$$
[\mathbf{Q} \;|\; \mathbf{K} \;|\; \mathbf{V}]
 = \mathbf{X}_{\text{norm}} \mathbf{W}_{qkv}
$$

dove $\mathbf{W}_{qkv} \in \mathbb{R}^{96 \times 288} (288 = 3 \times 96$ per $\mathbf Q, \mathbf K, \mathbf V)$

Risultato:

$$
[\mathbf{Q} \;|\; \mathbf{K} \;|\; \mathbf{V}]
 \in \mathbb{R}^{(B \times 64) \times 49 \times 288}
$$

L'ultima dimensione è una concatenazione di $\mathbf Q, \mathbf K, \mathbf V$:

$$
\mathbf{Q} \in \mathbb{R}^{(B \times 64) \times 49 \times 96} , \mathbf{K} \in \mathbb{R}^{(B \times 64) \times 49 \times 96} , \mathbf{V} \in \mathbb{R}^{(B \times 64) \times 49 \times 96}].
$$

**Separazione e Reshape:**

$$
\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{(B \times 64) \times 49 \times 96}
$$

Reshape per multi-head ($\text{heads} = 3$):

$$
\mathbf{Q} \rightarrow \mathbb{R}^{(B \times 64) \times 3 \times 49 \times 32}
$$

$$
\mathbf{K} \rightarrow \mathbb{R}^{(B \times 64) \times 3 \times 49 \times 32}
$$

$$
\mathbf{V} \rightarrow \mathbb{R}^{(B \times 64) \times 3 \times 49 \times 32}
$$

In pratica, dividiamo i 96 canali in 32 canali per ogni head.

### Calcolo dell'Attenzione
**Scaled Dot-Product Attention:**
$$
\mathbf{A} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_h}} + \mathbf{B}
$$
dove:
- $\mathbf{Q} \mathbf{K}^T \in \mathbb{R}^{(B \times 64) \times 3 \times 49 \times 49}$ (matrice di attenzione)
- Ci dice, per ogni finestra e per ogni immagine, quanto ogni patch è legata a tutte le altre.
- $\sqrt{d_h} = \sqrt{32} \approx 5.66$ (fattore di scala)
- $\mathbf{B} \in \mathbb{R}^{3 \times 49 \times 49}$ (relative position bias)

### Relative Position Bias
Il tensore $\mathbf{B} \in \mathbb{R}^{3 \times 49 \times 49}$ è ottenuto a partire dalla **tabella dei bias**.
Swin Transformer usa un **bias posizionale relativo** apprendibile invece di embedding posizionali assoluti per ogni token.

**Tabella dei bias:**
Consideriamo la head $q$. Siano $(y_i, x_i)$ e $(y_j, x_j)$ le coordinate del patch $p_i$ (query) e $p_j$ (key) rispettivamente (rispetto alla finestra $w$), dove $y$ indica la riga e $x$ la colonna. Sia $B_{\text{table}} \in \mathbb{R}^{(2M-1) \times (2M-1)}$ la tabella dei bias relativa all'head $q$.

La tabella dei bias contiene un valore di bias per ogni coppia di $(\Delta y, \Delta x)$. Quindi, per ogni coppia di patch $(y_i, x_i)$ e $(y_j, x_j)$, il valore di $B_{\text{table}}$ corrispondente all'offset $(y_j - y_i, x_j - x_i)$ viene recuperato dalla tabella.

#### Esempio
Assumiamo per semplicità $M = 2$.
$$
w = \begin{bmatrix}
p_{0, 0} & p_{0, 1} \\
p_{1, 0} & p_{1, 1}
\end{bmatrix}
$$

La $B_{\text{table}}$ ha dimensione $(2M-1) \times (2M-1)$, quindi $B_{\text{table}} \in \mathbb{R}^{3 \times 3}$.

$$
B_{\text{table}} =
\begin{array}{c|ccc}
\Delta y \backslash \Delta x & -1 & 0 & +1 \\
\hline
-1 & 0.1 & 0.2 & 0.3 \\
0 & 0.4 & 0.5 & 0.6 \\
+1 & 0.7 & 0.8 & 0.9 \\
\end{array}
\quad
\text{con } \Delta y, \Delta x \in \{-1, 0, +1\}
$$

Questa è la tabella dei bias relativa all'head $q$ appresa durante il training. Questa tabella ci sta dicendo che abbiamo un bias per ogni posizione relativa tra due patch. In particolare, abbiamo un bias per tutte le posizioni relative $(\Delta y, \Delta x)$ della key rispetto alla query:
- **Stessa posizione:** $(0, 0)$
- **Key più a destra:**
  - Direttamente a destra: $(0, +1)$
  - In alto a destra: $(-1, +1)$
  - In basso a destra: $(+1, +1)$
- **Key più a sinistra:**
  - Direttamente a sinistra: $(0, -1)$
  - In alto a sinistra: $(-1, -1)$
  - In basso a sinistra: $(+1, -1)$
- **Key sulla stessa colonna:**
  - Sopra: $(-1, 0)$
  - Sotto: $(+1, 0)$

Consideriamo le coordinate di due patch: query $(y_i, x_i) = (0, 0)$ e key $(y_j, x_j) = (1, 1)$:
$$
(y_j - y_i, x_j - x_i) = (1 - 0, 1 - 0) = (+1, +1)
$$

quindi la key $p_{1, 1}$ è in basso a destra rispetto alla query $p_{0, 0}$. E così per tutte le combinazioni di patch $(y_i, x_i)$ e $(y_j, x_j)$.

Quindi, nel calcolo dell'attenzione per ogni coppia di patch, in totale avremo $M^2 \times M^2 = 4 \times 4 = 16$ coppie. Ogni coppia avrà un bias associato in base alla loro posizione relativa.

Assumendo che la patch in prima posizione sia la patch query e che la seconda sia la patch key $(Q, K)$:
- **Stessa posizione:** $(p_{0, 0}, p_{0, 0})$, $(p_{0, 1}, p_{0, 1})$, $(p_{1, 0}, p_{1, 0})$, $(p_{1, 1}, p_{1, 1})$ → offset $(0, 0)$
- **Key a destra:** $(p_{0, 0}, p_{0, 1})$, $(p_{1, 0}, p_{1, 1})$ → offset $(0, +1)$
- **Key a sinistra:** $(p_{0, 1}, p_{0, 0})$, $(p_{1, 1}, p_{1, 0})$ → offset $(0, -1)$
- **Key sotto:** $(p_{0, 0}, p_{1, 0})$, $(p_{0, 1}, p_{1, 1})$ → offset $(+1, 0)$
- **Key sopra:** $(p_{1, 0}, p_{0, 0})$, $(p_{1, 1}, p_{0, 1})$ → offset $(-1, 0)$
- **Key in basso a destra:** $(p_{0, 0}, p_{1, 1})$ → offset $(+1, +1)$
- **Key in alto a sinistra:** $(p_{1, 1}, p_{0, 0})$ → offset $(-1, -1)$
- **Key in basso a sinistra:** $(p_{0, 1}, p_{1, 0})$ → offset $(+1, -1)$
- **Key in alto a destra:** $(p_{1, 0}, p_{0, 1})$ → offset $(-1, +1)$

Nell'immagine in basso è presente uno schema a colori che renderà sicuramente meglio l'idea molto semplice alla base di questo approccio.

<img src="">

#### Costruzione della matrice di bias estesa

Per ottenere la matrice dei bias con dimensioni $M^2 \times M^2$ (nel nostro esempio $4 \times 4$), che è quella che verrà poi utilizzata nel calcolo dell'attenzione, si procede come segue:

1. Per ogni coppia di patch $(p_i, p_j)$ nella finestra, con coordinate $(y_i, x_i)$ e $(y_j, x_j)$
2. Si calcola l'offset relativo: $(\Delta y, \Delta x) = (y_j - y_i, x_j - x_i)$
3. Si recupera il valore corrispondente dalla tabella $B_{\text{table}}[\Delta y, \Delta x]$
4. Si inserisce questo valore nella posizione $(i, j)$ della matrice estesa

Per il nostro esempio con $M = 2$, la matrice dei bias ${B} \in \mathbb{R}^{4 \times 4}$ sarà:

$$
{B} = \begin{bmatrix}
0.5 & 0.6 & 0.8 & 0.9 \\
0.4 & 0.5 & 0.7 & 0.8 \\
0.2 & 0.3 & 0.5 & 0.6 \\
0.1 & 0.2 & 0.4 & 0.5
\end{bmatrix}
$$

dove ogni riga corrisponde a una patch query e ogni colonna a una patch key. Ad esempio:
- L'elemento $(0, 3)$ corrisponde alla coppia $(p_{0,0}, p_{1,1})$ con offset $(+1, +1)$, quindi ha valore $0.9$
- L'elemento $(0, 1)$ corrisponde alla coppia $(p_{0,0}, p_{0,1})$ con offset $(0, +1)$, quindi ha valore $0.6$
- L'elemento $(3, 0)$ corrisponde alla coppia $(p_{1,1}, p_{0,0})$ con offset $(-1, -1)$, quindi ha valore $0.1$

Nel caso generale con $M = 7$ (finestre $7 \times 7$ con 49 patch), avremo $B_{\text{table}} \in \mathbb{R}^{13 \times 13}$ e ${B} \in \mathbb{R}^{49 \times 49}$ per ciascuna delle 3 heads.

Questo meccanismo permette al modello di apprendere quanto sia importante la posizione relativa tra patch durante l'attenzione, rendendo il bias condiviso per tutte le coppie di patch con la stessa distanza relativa.

### Applicazione Softmax

Dopo aver calcolato i punteggi di attenzione grezzi e aggiunto il relative position bias, otteniamo la matrice $\mathbf{A}$. Ora dobbiamo normalizzare questi punteggi per ottenere delle probabilità che indichino quanto ogni patch dovrebbe "prestare attenzione" alle altre patch nella finestra.

$$
\mathbf{A}_{\text{norm}} = \text{Softmax}(\mathbf{A})
$$

$$
\mathbf{A}_{\text{norm}} \in \mathbb{R}^{(B \times 64) \times 3 \times 49 \times 49}
$$

**Cosa fa la Softmax:**

La funzione softmax viene applicata **lungo l'ultima dimensione** (le 49 colonne, corrispondenti alle patch key). Per ogni patch query (riga), la softmax trasforma i punteggi grezzi in una distribuzione di probabilità che somma a 1:

$$
\text{Softmax}(a_i) = \frac{e^{a_i}}{\sum_{j=1}^{49} e^{a_j}}
$$

**Significato:**
- Ogni riga di $\mathbf{A}_{\text{norm}}$ ora rappresenta una distribuzione di probabilità
- I valori sono compresi tra 0 e 1
- La somma di ogni riga è esattamente 1
- Valori più alti indicano che la patch query dovrebbe "prestare più attenzione" a quella specifica patch key

**Perché è necessaria:**
La softmax serve a:
1. **Normalizzare** i punteggi in un range interpretabile (0-1)
2. **Amplificare** le differenze tra punteggi alti e bassi (grazie alla funzione esponenziale)
3. **Creare una distribuzione di probabilità** che può essere usata come sistema di pesi per combinare i values

### Applicazione ai Values

Ora che abbiamo i pesi di attenzione normalizzati, possiamo usarli per aggregare le informazioni dai values:

$$
\mathbf{O} = \mathbf{A}_{\text{norm}} \mathbf{V}
$$

$$
\mathbf{O} \in \mathbb{R}^{(B \times 64) \times 3 \times 49 \times 32}
$$

**Operazione matriciale:**

Stiamo moltiplicando:
- $\mathbf{A}_{\text{norm}} \in \mathbb{R}^{(B \times 64) \times 3 \times 49 \times 49}$ (pesi di attenzione)
- $\mathbf{V} \in \mathbb{R}^{(B \times 64) \times 3 \times 49 \times 32}$ (values)

Risultato: $\mathbf{O} \in \mathbb{R}^{(B \times 64) \times 3 \times 49 \times 32}$

**Cosa succede concretamente:**

Per ogni head e per ogni patch query (49 patch totali):
1. Prendiamo la riga corrispondente di $\mathbf{A}_{\text{norm}}$ (i suoi pesi di attenzione verso tutte le 49 patch)
2. Usiamo questi pesi per fare una **media ponderata** di tutti i 49 values
3. Il risultato è un nuovo vettore di dimensione 32 (dimensione per head) che rappresenta l'informazione aggregata

**Interpretazione:**

Ogni patch nel risultato $\mathbf{O}$ è ora una combinazione pesata di tutte le patch della finestra. I pesi sono determinati dall'attenzione:
- Se la patch query $i$ ha alta attenzione verso la patch $j$, il value della patch $j$ contribuirà maggiormente al risultato finale della patch $i$
- Questo permette a ogni patch di "raccogliere" informazioni contestuali dalle patch vicine in base alla loro rilevanza.

### Concatenazione delle Head

Le 3 head hanno processato l'informazione in parallelo, ciascuna con la propria prospettiva (parametri $\mathbf{W}_{qkv}$ diversi). Ora dobbiamo ricombinare i loro output:

$$
\mathbf{O} \rightarrow \mathbb{R}^{(B \times 64) \times 49 \times 96}
$$

**Operazione di reshape:**

Passiamo da:
- $\mathbb{R}^{(B \times 64) \times 3 \times 49 \times 32}$

A:
- $\mathbb{R}^{(B \times 64) \times 49 \times 96}$

**Come funziona:**

Per ogni patch (delle 49), concateniamo gli output delle 3 head:
- Head 1: vettore di 32 dimensioni
- Head 2: vettore di 32 dimensioni  
- Head 3: vettore di 32 dimensioni
- **Concatenazione:** vettore di $32 + 32 + 32 = 96$ dimensioni

**Motivazione del multi-head:**

Ogni head ha imparato a catturare aspetti diversi delle relazioni tra patch:
- Una head potrebbe specializzarsi in relazioni spaziali locali
- Un'altra potrebbe catturare pattern di texture
- Una terza potrebbe rilevare strutture globali

Concatenando i loro output, otteniamo una rappresentazione ricca che combina tutte queste prospettive.

### Proiezione Finale

L'output concatenato delle head viene trasformato attraverso un'ultima proiezione lineare:

$
\mathbf{O}_{\text{proj}} = \mathbf{O} \mathbf{W}_{\text{proj}}
$

dove $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{96 \times 96}$

**Perché serve questa proiezione finale?**

A questo punto del processo, abbiamo concatenato gli output delle 3 head ottenendo per ogni patch un vettore di 96 dimensioni. Tuttavia, questa concatenazione è semplicemente un "affiancamento" dei risultati delle diverse head, senza alcuna interazione tra loro. Ogni blocco di 32 dimensioni proviene da una head specifica e rimane isolato dagli altri.

La proiezione finale è essenziale per **integrare e miscelare** le informazioni che le diverse head hanno estratto in modo indipendente. Pensiamo a questa operazione come a un "strato di fusione" che permette al modello di imparare come combinare al meglio le diverse prospettive catturate dalle head.

**Il concetto di mixing delle informazioni:**

Consideriamo un esempio concreto. Supponiamo che durante il training, il modello abbia scoperto che:
- La **head 1** è brava a identificare bordi verticali
- La **head 2** è specializzata nel riconoscere texture
- La **head 3** cattura relazioni spaziali a lungo raggio

Dopo la concatenazione, abbiamo tutte queste informazioni presenti nel vettore da 96 dimensioni, ma sono separate in tre blocchi distinti. La proiezione finale, attraverso la matrice $\mathbf{W}_{\text{proj}}$, permette di creare nuove feature che sono **combinazioni** di queste informazioni. 

Per esempio, potrebbe imparare che per riconoscere un particolare oggetto serve combinare informazioni dai bordi verticali (head 1) con informazioni sulla texture (head 2), creando una nuova feature che rappresenta questa combinazione. Questo non sarebbe possibile con la semplice concatenazione.

**Apprendimento di rappresentazioni più ricche:**

La matrice $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{96 \times 96}$ contiene parametri apprendibili. Durante il training con backpropagation, il modello impara quali combinazioni delle feature estratte dalle head sono più utili per il task finale (classificazione, object detection, etc.). 

In pratica, ogni riga di $\mathbf{W}_{\text{proj}}$ definisce come costruire una nuova dimensione dell'output combinando linearmente tutte le 96 dimensioni dell'input concatenato. Questo significa che ogni dimensione dell'output può dipendere da qualsiasi dimensione di qualsiasi head, permettendo interazioni complesse.

**Preparazione per la residual connection:**

Un altro aspetto fondamentale è che questa proiezione trasforma l'output del meccanismo di attenzione in uno spazio di rappresentazione che è compatibile con l'input originale $\mathbf{X}$. Ricordiamo che alla fine di questo blocco dovremo sommare questo output all'input originale (residual connection):

$$
\mathbf{X}_{\text{attn}} = \mathbf{X} + \text{DropPath}(\mathbf{O}_{\text{proj}})
$$

Per poter effettuare questa somma in modo significativo, l'output deve trovarsi nello stesso spazio di rappresentazione dell'input. La proiezione finale garantisce questa compatibilità, mappando le feature elaborate dall'attenzione in uno spazio dove possono essere integrate con le feature originali.

**Controllo della capacità del modello:**

Inoltre, questa proiezione aggiunge un ulteriore strato di parametri apprendibili al modello. Mentre il numero di parametri nella proiezione QKV è determinato dalla necessità di creare query, key e value, la proiezione finale offre al modello una capacità aggiuntiva di apprendere trasformazioni complesse. Questo è particolarmente importante perché il meccanismo di attenzione da solo potrebbe non essere sufficiente a catturare tutte le relazioni necessarie.

**Dettagli dimensionali:**

Osserviamo più nel dettaglio cosa succede numericamente:

Input: $\mathbf{O} \in \mathbb{R}^{(B \times 64) \times 49 \times 96}$

Moltiplicazione per $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{96 \times 96}$

Output: $\mathbf{O}_{\text{proj}} \in \mathbb{R}^{(B \times 64) \times 49 \times 96}$

Per ogni patch (una delle 49 in ogni finestra), prendiamo il suo vettore di 96 dimensioni e lo moltiplichiamo per la matrice $\mathbf{W}_{\text{proj}}$. Questo produce un nuovo vettore di 96 dimensioni dove ciascuna componente è una combinazione lineare pesata di tutte le 96 componenti originali.

La dimensionalità rimane invariata (96 → 96), ma la **qualità** e il **contenuto informativo** delle feature sono stati arricchiti attraverso questa trasformazione apprendibile. Non è un semplice passaggio di dati, ma un vero e proprio step di elaborazione che il modello ottimizza durante il training per massimizzare le performance sul task specifico.

### Merge delle Finestre

Finora abbiamo lavorato su finestre separate. Ora dobbiamo ricostruire l'immagine completa ricomponendo tutte le finestre:

$$
\mathbf{O}_{\text{proj}} \in \mathbb{R}^{(B \times 64) \times 49 \times 96} \rightarrow \mathbb{R}^{B \times 56 \times 56 \times 96} \rightarrow \mathbb{R}^{B \times 3136 \times 96}
$$

**Processo di ricostruzione:**

**Passo 1: Da finestre piatte a finestre spaziali**

Da $\mathbb{R}^{(B \times 64) \times 49 \times 96}$ a $\mathbb{R}^{B \times 64 \times 49 \times 96}$

Separiamo la dimensione batch dalle finestre.

**Passo 2: Reshape delle patch in ogni finestra**

Ogni finestra contiene 49 patch disposte in una griglia $7 \times 7$:

$\mathbb{R}^{B \times 64 \times 49 \times 96} \rightarrow \mathbb{R}^{B \times 64 \times 7 \times 7 \times 96}$

Ora ogni finestra ha una struttura spaziale bidimensionale.

**Passo 3: Ricomposizione della griglia di finestre**

Le 64 finestre erano disposte in una griglia $8 \times 8$ (ricordiamo che $8 \times 8 = 64$ finestre):

$\mathbb{R}^{B \times 64 \times 7 \times 7 \times 96} \rightarrow \mathbb{R}^{B \times 8 \times 8 \times 7 \times 7 \times 96}$

**Passo 4: Riorganizzazione in feature map**

Riorganizziamo le dimensioni per ottenere un'unica feature map continua:

$\mathbb{R}^{B \times 8 \times 8 \times 7 \times 7 \times 96} \rightarrow \mathbb{R}^{B \times (8 \times 7) \times (8 \times 7) \times 96} = \mathbb{R}^{B \times 56 \times 56 \times 96}$

Ogni finestra $7 \times 7$ viene posizionata nella sua posizione originale nella griglia $8 \times 8$, ricreando la feature map completa $56 \times 56$.

**Passo 5: Flatten finale**

Per compatibilità con le operazioni successive, appiattiamo le dimensioni spaziali:

$\mathbb{R}^{B \times 56 \times 56 \times 96} \rightarrow \mathbb{R}^{B \times 3136 \times 96}$

dove $3136 = 56 \times 56$ è il numero totale di patch nell'immagine.

**Significato del merge:**

Questo processo è l'operazione inversa del windowing iniziale:
- Abbiamo diviso l'immagine in finestre per applicare l'attenzione locale in modo efficiente
- Ora ricomponiamo le finestre per recuperare la struttura spaziale completa dell'immagine
- Ogni patch mantiene le informazioni aggregate dalla sua finestra locale
- La struttura gerarchica è preservata per i layer successivi.

### Residual Connection

Infine, abbiamo una connessione residuale:

$$
\mathbf{X}_{\text{attn}} = \mathbf{X} + \text{DropPath}(\mathbf{O}_{\text{proj}})
$$

$$
\mathbf{X}_{\text{attn}} \in \mathbb{R}^{B \times 3136 \times 96}
$$

Da notare che la $\text{DropPath}$ viene applicata solo durante il processo di training del modello. Questo significa che durante l'inferenza, avremo semplicemente:

$$
\mathbf{X}_{\text{attn}} = \mathbf{X} + \mathbf{O}_{\text{proj}}.
$$

## Feed-Forward Network (MLP)

Dopo l'attenzione, ogni blocco include una rete feed-forward:

**Layer Norm:**

$$
\mathbf{X}_{\text{norm2}} = \text{LayerNorm}(\mathbf{X}_{\text{attn}})
$$

**MLP:**

$$
\mathbf{X}_{\text{mlp}} = \text{GELU}(\mathbf{X}_{\text{norm2}} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2
$$

dove:
- $\mathbf{W}_1 \in \mathbb{R}^{96 \times 384}$ (espansione con ratio 4)
- $\mathbf{b}_1 \in \mathbb{R}^{384}$
- $\mathbf{W}_2 \in \mathbb{R}^{384 \times 96}$ (proiezione al numero di canali originale)

**Dimensioni intermediate:**

$$
\mathbf{X}_{\text{hidden}} \in \mathbb{R}^{B \times 3136 \times 384}
$$

$$
\mathbf{X}_{\text{mlp}} \in \mathbb{R}^{B \times 3136 \times 96}
$$

**Residual Connection:**

$$
\mathbf{X}_{\text{out}} = \mathbf{X}_{\text{attn}} + \text{DropPath}(\mathbf{X}_{\text{mlp}})
$$

$$
\mathbf{X}_{\text{out}} \in \mathbb{R}^{B \times 3136 \times 96}
$$

## SW-MSA: Shifted Window Multi-head Self Attention

Nel Swin Transformer, l’idea è di applicare l’attenzione non su tutta l’immagine (troppo costoso), ma su piccole finestre locali — ad esempio blocchi di $7×7$ pixel.

**👉 Problema:**
Se ogni finestra è indipendente, i pixel in finestre diverse non comunicano mai.
Quindi, un pixel nell’angolo di una finestra non “vede” nulla fuori da quella finestra.

**👉 Soluzione:**
Il secondo blocco di ogni stage (swin transformer block) sposta le finestre di metà finestra ($M/2$) in entrambe le direzioni.
Così, le nuove finestre contengono parti di quelle vecchie, permettendo scambio di informazioni tra regioni diverse.

### Input Processing

Prima di applicare lo shift, l'output del primo blocco viene normalizzato:

$$
\mathbf{X}_{\text{norm1}} = \text{LayerNorm}(\mathbf{X}_{\text{out}})
$$

$$
\mathbf{X}_{\text{norm1}} \in \mathbb{R}^{B \times 3136 \times 96}
$$

**Reshape spaziale per lo shift:**

$$
\mathbf{X}_{\text{norm1}} \rightarrow \mathbb{R}^{B \times 56 \times 56 \times 96}
$$

### Cyclic Shift

Prima di tutto calcoliamo il fattore di **shift**. Ricordiamo che $M$ rappresenta la dimensione della finestra e consideriamo la parte intera della divisione per $2$.

**Shift Amount:**

$$
s = \lfloor \frac{M}{2} \rfloor = \lfloor \frac{7}{2} \rfloor = 3
$$

**Operazione di Shift:**

L'immagine viene shiftata ciclicamente di $s$ pixel sia in altezza che in larghezza:

$$
\mathbf{X}_{\text{shifted}}[i, j] = \mathbf{X}_{\text{norm1}}[(i - s) \mod H, (j - s) \mod W]
$$

In notazione tensoriale usando `torch.roll`:

$$
\mathbf{X}_{\text{shifted}} = \text{roll}(\mathbf{X}_{\text{norm1}}, \text{shifts}=(-3, -3), \text{dims}=(1, 2))
$$

**Dimensioni:**

$$
\mathbf{X}_{\text{norm1}} \in \mathbb{R}^{B \times 56 \times 56 \times 96} \rightarrow \mathbf{X}_{\text{shifted}} \in \mathbb{R}^{B \times 56 \times 56 \times 96}
$$

### Partizionamento e Mascheramento

Dopo lo shift, l'immagine viene partizionata in finestre come in W-MSA. Tuttavia, alcune finestre ora contengono regioni che non erano adiacenti nell'immagine originale.

**Attention Mask:**

Una maschera viene applicata per impedire l'attenzione tra regioni non adiacenti:

$$
\mathbf{A}_{\text{masked}} = \mathbf{A} + \mathbf{M}_{\text{mask}}
$$

dove $\mathbf{M}_{\text{mask}}$ contiene $-100$ per coppie di posizioni che non dovrebbero interagire e $0$ altrove.

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*uEzhdlu2GppknurQZ7k82A.png" alt="Immagine di un Transformer" style="display: block; margin-left: auto; margin-right: auto;">

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*ptVDLGO3fLPkjLNCNZHs-Q.png" alt="Immagine di un Transformer" style="display: block; margin-left: auto; margin-right: auto;">

**Calcolo dell'attenzione con maschera:**

$$
\mathbf{A}_{\text{norm}} = \text{Softmax}(\mathbf{A}_{\text{masked}})
$$

### Reverse Cyclic Shift

Dopo l'attenzione, lo shift viene invertito:

$$
\mathbf{X}_{\text{unshifted}} = \text{roll}(\mathbf{X}_{\text{out}}, \text{shifts}=(3, 3), \text{dims}=(1, 2))
$$

L'output ha la stessa forma dell'input:

$$
\mathbf{X}_{\text{unshifted}} \in \mathbb{R}^{B \times 3136 \times 96}
$$

## Patch Merging

Alla fine di ogni stage (eccetto l'ultimo), un layer di **Patch Merging** riduce la risoluzione spaziale e aumenta il numero di canali.

### Downsampling Stage 1 → Stage 2

**Input:**

$$
\mathbf{X}_1 \in \mathbb{R}^{B \times 3136 \times 96}
$$

**Reshape spaziale:**

$$
\mathbf{X}_1 \rightarrow \mathbb{R}^{B \times 56 \times 56 \times 96}
$$

**Sampling 2×2:**

L'immagine viene campionata prendendo pixel alternati per creare 4 sottoimmagini:

$$
\mathbf{X}_0 = \mathbf{X}_1[:, 0::2, 0::2, :] \in \mathbb{R}^{B \times 28 \times 28 \times 96}
$$

$$
\mathbf{X}_1 = \mathbf{X}_1[:, 1::2, 0::2, :] \in \mathbb{R}^{B \times 28 \times 28 \times 96}
$$

$$
\mathbf{X}_2 = \mathbf{X}_1[:, 0::2, 1::2, :] \in \mathbb{R}^{B \times 28 \times 28 \times 96}
$$

$$
\mathbf{X}_3 = \mathbf{X}_1[:, 1::2, 1::2, :] \in \mathbb{R}^{B \times 28 \times 28 \times 96}
$$

**Concatenazione:**

$$
\mathbf{X}_{\text{concat}} = \text{Concat}([\mathbf{X}_0, \mathbf{X}_1, \mathbf{X}_2, \mathbf{X}_3], \text{dim}=-1)
$$

$$
\mathbf{X}_{\text{concat}} \in \mathbb{R}^{B \times 28 \times 28 \times 384}
$$

**Flatten:**

$$
\mathbf{X}_{\text{concat}} \rightarrow \mathbb{R}^{B \times 784 \times 384}
$$

**Layer Norm:**

$$
\mathbf{X}_{\text{norm}} = \text{LayerNorm}(\mathbf{X}_{\text{concat}})
$$

**Proiezione lineare:**

$$
\mathbf{X}_2 = \mathbf{X}_{\text{norm}} \mathbf{W}_{\text{reduction}}
$$

dove $\mathbf{W}_{\text{reduction}} \in \mathbb{R}^{384 \times 192}$

**Output:**

$$
\mathbf{X}_2 \in \mathbb{R}^{B \times 784 \times 192}
$$

Ora abbiamo:
- Risoluzione: $28 \times 28$ (dimezzata)
- Canali: $192$ (raddoppiati)
- Token: $784 = 28 \times 28$

## Stage 2: Secondo Livello della Gerarchia

### Parametri Stage 2

- Risoluzione: $H_2 \times W_2 = 28 \times 28$
- Canali: $C_2 = 192$
- Numero di blocchi: $\text{depth}_2 = 2$
- Numero di head: $\text{heads}_2 = 6$
- Dimensione finestra: $M = 7$

### Calcoli

**Numero di finestre:**

$$
\text{num\_windows} = \frac{28}{7} \times \frac{28}{7} = 4 \times 4 = 16
$$

**Dimensione per head:**

$$
d_h = \frac{192}{6} = 32
$$

Il processing è identico allo Stage 1, ma con dimensioni diverse. Ogni finestra contiene ancora $7 \times 7 = 49$ patch.

## Stage 3: Terzo Livello della Gerarchia

### Patch Merging Stage 2 → Stage 3

**Input:**

$$
\mathbf{X}_2 \in \mathbb{R}^{B \times 784 \times 192}
$$

Dopo patch merging:

$$
\mathbf{X}_3 \in \mathbb{R}^{B \times 196 \times 384}
$$

dove $196 = 14 \times 14$

### Parametri Stage 3

- Risoluzione: $H_3 \times W_3 = 14 \times 14$
- Canali: $C_3 = 384$
- Numero di blocchi: $\text{depth}_3 = 6$
- Numero di head: $\text{heads}_3 = 12$
- Dimensione finestra: $M = 7$

**Numero di finestre:**

$$
\text{num\_windows} = \frac{14}{7} \times \frac{14}{7} = 2 \times 2 = 4
$$

**Dimensione per head:**

$$
d_h = \frac{384}{12} = 32
$$

## Stage 4: Quarto Livello della Gerarchia

### Patch Merging Stage 3 → Stage 4

**Input:**

$$
\mathbf{X}_3 \in \mathbb{R}^{B \times 196 \times 384}
$$

Dopo patch merging:

$$
\mathbf{X}_4 \in \mathbb{R}^{B \times 49 \times 768}
$$

dove $49 = 7 \times 7$

### Parametri Stage 4

- Risoluzione: $H_4 \times W_4 = 7 \times 7$
- Canali: $C_4 = 768$
- Numero di blocchi: $\text{depth}_4 = 2$
- Numero di head: $\text{heads}_4 = 24$
- Dimensione finestra: $M = 7$

**Numero di finestre:**

$$
\text{num\_windows} = \frac{7}{7} \times \frac{7}{7} = 1 \times 1 = 1
$$

Con una sola finestra, W-MSA e SW-MSA sono equivalenti (non c'è shifting).

**Dimensione per head:**

$$
d_h = \frac{768}{24} = 32
$$

## Classification Head

Dopo tutti gli stage, l'output viene processato per la classificazione.

### Layer Norm Finale

$$
\mathbf{X}_{\text{norm}} = \text{LayerNorm}(\mathbf{X}_4)
$$

$$
\mathbf{X}_{\text{norm}} \in \mathbb{R}^{B \times 49 \times 768}
$$

### Global Average Pooling

$$
\mathbf{X}_{\text{pool}} = \text{AvgPool}(\mathbf{X}_{\text{norm}}^T)
$$

Trasponendo: $\mathbf{X}_{\text{norm}}^T \in \mathbb{R}^{B \times 768 \times 49}$

Dopo pooling:

$$
\mathbf{X}_{\text{pool}} \in \mathbb{R}^{B \times 768 \times 1} \rightarrow \mathbb{R}^{B \times 768}
$$

### Linear Classifier

$$
\mathbf{y} = \mathbf{X}_{\text{pool}} \mathbf{W}_{\text{head}} + \mathbf{b}_{\text{head}}
$$

dove:
- $\mathbf{W}_{\text{head}} \in \mathbb{R}^{768 \times K}$
- $K$ = numero di classi (es. 1000 per ImageNet)

**Output finale:**

$$
\mathbf{y} \in \mathbb{R}^{B \times K}
$$

Questo vettore contiene i logit per ogni classe.

## Riepilogo delle Dimensioni

| Stage | Risoluzione | Canali | Token | Blocchi | Head | Head Dim |
|-------|-------------|--------|-------|---------|------|----------|
| Input | 224×224 | 3 | - | - | - | - |
| Embed | 56×56 | 96 | 3136 | - | - | - |
| 1 | 56×56 | 96 | 3136 | 2 | 3 | 32 |
| 2 | 28×28 | 192 | 784 | 2 | 6 | 32 |
| 3 | 14×14 | 384 | 196 | 6 | 12 | 32 |
| 4 | 7×7 | 768 | 49 | 2 | 24 | 32 |
| Output | - | 768 | 1 | - | - | - |
| Logits | - | K | - | - | - | - |

## Complessità Computazionale

### Self-Attention Standard (Global)

Per $N$ token con dimensione $C$:

$$
\text{Complexity}_{\text{global}} = O(N^2 \cdot C)
$$

Per Stage 1 con $N = 3136$:

$$
\text{Complexity}_{\text{global}} = O(3136^2 \cdot 96) \approx O(9.4 \times 10^8)
$$

### Window-based Self-Attention

Con finestre di dimensione $M \times M$:

$$
\text{Complexity}_{\text{window}} = O\left(\frac{N}{M^2} \cdot (M^2)^2 \cdot C\right) = O(N \cdot M^2 \cdot C)
$$

Per Stage 1 con $M = 7$:

$$
\text{Complexity}_{\text{window}} = O(3136 \cdot 49 \cdot 96) \approx O(1.5 \times 10^7)
$$

**Riduzione della complessità:**

$$
\text{Speedup} = \frac{N^2 \cdot C}{N \cdot M^2 \cdot C} = \frac{N}{M^2} = \frac{3136}{49} = 64\times
$$

La window-based attention è **64 volte più efficiente** per lo Stage 1!

## Swin Transformer V2: Miglioramenti e Differenze

Swin Transformer V2 introduce diverse modifiche per migliorare stabilità, scalabilità e prestazioni.

### 1. Scaled Cosine Attention

**Swin V1** usa dot-product attention standard:

$$
\mathbf{A} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_h}}
$$

**Swin V2** usa cosine attention con temperatura apprendibile:

$$
\mathbf{A} = \tau \cdot \frac{\mathbf{Q}_{\text{norm}} \mathbf{K}_{\text{norm}}^T}{\|\mathbf{Q}_{\text{norm}}\| \|\mathbf{K}_{\text{norm}}\|}
$$

dove:
- Normalizzazione: $\mathbf{Q}_{\text{norm}} = \frac{\mathbf{Q}}{\|\mathbf{Q}\|_2}$, $\mathbf{K}_{\text{norm}} = \frac{\mathbf{K}}{\|\mathbf{K}\|_2}$
- $\tau = \log(\text{scale})$ è un parametro apprendibile per head
- $\text{scale}$ è limitato: $\text{scale} \leq \frac{1}{0.01} = 100$

**Vantaggi:**
- Maggiore stabilità durante il training
- Gradiente più uniforme
- Migliore convergenza per modelli grandi

### 2. Continuous Relative Position Bias

**Swin V1** usa una tabella discreta:

$$
\mathbf{B} \in \mathbb{R}^{(2M-1) \times (2M-1) \times h}
$$

**Swin V2** usa una **MLP continua** per generare i bias:

$$
\mathbf{B} = \text{MLP}(\Delta \mathbf{p})
$$

dove $\Delta \mathbf{p}$ sono le coordinate relative normalizzate.

**Architettura MLP:**

$$
\mathbf{B} = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \Delta \mathbf{p} + \mathbf{b}_1)
$$

con:
- $\mathbf{W}_1 \in \mathbb{R}^{512 \times 2}$ (proietta coordinate 2D a 512 dim)
- $\mathbf{W}_2 \in \mathbb{R}^{h \times 512}$ (proietta a numero di head)

**Normalizzazione delle coordinate:**

$$
\Delta p_x = \frac{i - j}{M - 1}, \quad \Delta p_y = \frac{k - l}{M - 1}
$$

Trasformazione log:

$$
\Delta \hat{p} = \text{sign}(\Delta p) \cdot \log_2(|\Delta p| + 1) / \log_2(8)
$$

Questa trasformazione mappa le coordinate in $[-1, 1]$ con maggiore risoluzione vicino allo zero.

**Post-processing del bias:**

$
\mathbf{B}_{\text{final}} = 16 \cdot \sigma(\mathbf{B})
$

dove $\sigma$ è la funzione sigmoid. Questo scala i bias in un range controllato $[0, 16]$.

**Vantaggi:**
- Transferibilità tra diverse risoluzioni di finestra
- Interpolazione continua delle posizioni
- Migliore generalizzazione

### 3. Log-spaced Continuous Position Bias

Le coordinate relative vengono trasformate in scala logaritmica prima di essere processate dalla MLP:

$
\hat{x} = \text{sign}(x) \cdot \log(1 + |x|) / \log(8)
$

Questo permette una rappresentazione più uniforme di distanze diverse.

### 4. Rimozione della Normalizzazione Pre-Attenzione

**Swin V1:**

$
\mathbf{X}_{\text{attn}} = \mathbf{X} + \text{Attn}(\text{LN}(\mathbf{X}))
$

$
\mathbf{X}_{\text{out}} = \mathbf{X}_{\text{attn}} + \text{MLP}(\text{LN}(\mathbf{X}_{\text{attn}}))
$

**Swin V2:** usa post-normalization per migliorare la stabilità

$
\mathbf{X}_{\text{attn}} = \text{LN}(\mathbf{X} + \text{Attn}(\mathbf{X}))
$

$
\mathbf{X}_{\text{out}} = \text{LN}(\mathbf{X}_{\text{attn}} + \text{MLP}(\mathbf{X}_{\text{attn}}))
$

Tuttavia, guardando il codice fornito, Swin V2 mantiene ancora la pre-normalization ma con alcune modifiche ai parametri di inizializzazione.

### 5. Scaled Cosine Attention - Dettagli Implementativi

**Query e Key Bias:**

In Swin V2, il QKV bias viene modificato:

$
\text{qkv\_bias} = [\mathbf{q}_{\text{bias}}, \mathbf{0}, \mathbf{v}_{\text{bias}}]
$

Il bias per le key è zero, mentre query e value hanno bias apprendibili.

**Calcolo attention:**

$
\mathbf{A} = \text{normalize}(\mathbf{Q}) \cdot \text{normalize}(\mathbf{K})^T
$

dove la normalizzazione è:

$
\text{normalize}(\mathbf{X}) = \frac{\mathbf{X}}{\|\mathbf{X}\|_2 + \epsilon}
$

**Scaling con temperatura:**

$
\tau = \exp(\log(\text{scale}))
$

con $\text{scale}$ limitato a $\max = \log(1/0.01) = \log(100) \approx 4.6$

$
\mathbf{A}_{\text{scaled}} = \tau \cdot \mathbf{A}
$

### 6. Modifiche al Patch Merging

**Swin V1:**

$
\text{Norm} \rightarrow \text{Linear}
$

**Swin V2:**

$
\text{Linear} \rightarrow \text{Norm}
$

L'ordine è invertito:

$
\mathbf{X}_{\text{reduced}} = \mathbf{W}_{\text{reduction}} \cdot \mathbf{X}_{\text{concat}}
$

$
\mathbf{X}_{\text{out}} = \text{LayerNorm}(\mathbf{X}_{\text{reduced}})
$

## Fused Window Process: Ottimizzazione

Il codice fornito mostra una implementazione ottimizzata delle operazioni di window partition e cyclic shift.

### Standard Implementation

**Roll + Window Partition (Python):**

```python
shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
x_windows = window_partition(shifted_x, window_size)
```

Questo richiede due operazioni separate:
1. Cyclic shift (roll)
2. Window partition (reshape + permute)

### Fused Implementation

Il kernel fuso combina entrambe le operazioni in un'unica operazione CUDA ottimizzata:

```python
x_windows = WindowProcess.apply(x, B, H, W, C, -shift_size, window_size)
```

**Forward Pass:**

$
\mathbf{X} \in \mathbb{R}^{B \times H \times W \times C} \xrightarrow{\text{fused}} \mathbf{X}_{\text{windows}} \in \mathbb{R}^{(B \cdot n_w) \times M \times M \times C}
$

in un'unica operazione kernel.

**Vantaggi:**
- Riduzione accessi alla memoria
- Eliminazione di tensori intermedi
- Migliore utilizzo della cache
- Speedup 2-3× rispetto all'implementazione standard

### Reverse Process

Analogamente per l'operazione inversa:

```python
x = WindowProcessReverse.apply(attn_windows, B, H, W, C, shift_size, window_size)
```

Combina:
1. Window merge
2. Reverse cyclic shift

$
\mathbf{X}_{\text{windows}} \in \mathbb{R}^{(B \cdot n_w) \times M \times M \times C} \xrightarrow{\text{fused}} \mathbf{X} \in \mathbb{R}^{B \times H \times W \times C}
$

## Training Details

### Loss Function

Per classificazione con $K$ classi:

$
\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{k=1}^{K} y_{ik} \log(\text{softmax}(\hat{y}_{ik}))
$

Con label smoothing ($\alpha = 0.1$):

$
y'_{ik} = (1 - \alpha) \cdot y_{ik} + \frac{\alpha}{K}
$

$
\mathcal{L}_{\text{smooth}} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{k=1}^{K} y'_{ik} \log(\text{softmax}(\hat{y}_{ik}))
$

### Data Augmentation

**Mixup:** combina due immagini

$
\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1 - \lambda) \mathbf{x}_j
$

$
\tilde{\mathbf{y}} = \lambda \mathbf{y}_i + (1 - \lambda) \mathbf{y}_j
$

dove $\lambda \sim \text{Beta}(\alpha_{\text{mixup}}, \alpha_{\text{mixup}})$ con $\alpha_{\text{mixup}} = 0.8$

**CutMix:** sostituisce una regione dell'immagine

$
\mathbf{M} \in \{0, 1\}^{H \times W}
$

$
\tilde{\mathbf{x}} = \mathbf{M} \odot \mathbf{x}_i + (1 - \mathbf{M}) \odot \mathbf{x}_j
$

### Optimizer

**AdamW** con:
- Learning rate base: $\eta_{\text{base}} = 5 \times 10^{-4}$
- Weight decay: $\lambda = 0.05$
- $\beta_1 = 0.9$, $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$

**Linear scaling rule:**

$
\eta = \eta_{\text{base}} \times \frac{B \times N_{\text{gpu}}}{512}
$

### Learning Rate Schedule

**Cosine decay** con warmup:

**Warmup phase** (primi 20 epochs):

$
\eta(t) = \eta_{\text{warmup}} + \frac{\eta - \eta_{\text{warmup}}}{T_{\text{warmup}}} \cdot t
$

dove $\eta_{\text{warmup}} = 5 \times 10^{-7}$

**Cosine decay phase:**

$
\eta(t) = \eta_{\text{min}} + \frac{\eta - \eta_{\text{min}}}{2} \left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}} \pi\right)\right)
$

dove $\eta_{\text{min}} = 5 \times 10^{-6}$, $T_{\text{max}} = 300$ epochs

### Stochastic Depth

**Drop path** con probabilità crescente per layer profondi:

$
p_l = p_{\text{max}} \cdot \frac{l}{L}
$

dove:
- $p_{\text{max}} = 0.1$ (drop path rate massimo)
- $l$ = indice del layer
- $L$ = numero totale di layer

**Applicazione:**

$
\mathbf{X}_{\text{out}} = \mathbf{X} + \text{Bernoulli}(1 - p_l) \cdot \frac{\text{Layer}(\mathbf{X})}{1 - p_l}
$

## Varianti di Swin Transformer

### Swin-T (Tiny)

| Parametro | Valore |
|-----------|--------|
| Embed dim | 96 |
| Depths | [2, 2, 6, 2] |
| Num heads | [3, 6, 12, 24] |
| Window size | 7 |
| Parametri | ~29M |

### Swin-S (Small)

| Parametro | Valore |
|-----------|--------|
| Embed dim | 96 |
| Depths | [2, 2, 18, 2] |
| Num heads | [3, 6, 12, 24] |
| Window size | 7 |
| Parametri | ~50M |

### Swin-B (Base)

| Parametro | Valore |
|-----------|--------|
| Embed dim | 128 |
| Depths | [2, 2, 18, 2] |
| Num heads | [4, 8, 16, 32] |
| Window size | 7 |
| Parametri | ~88M |

### Swin-L (Large)

| Parametro | Valore |
|-----------|--------|
| Embed dim | 192 |
| Depths | [2, 2, 18, 2] |
| Num heads | [6, 12, 24, 48] |
| Window size | 7 |
| Parametri | ~197M |

## Applicazioni

### Image Classification

Output diretto dal classification head:

$
\mathbf{y} \in \mathbb{R}^{B \times K}
$

### Object Detection

Swin Transformer può essere usato come backbone in framework come:
- **Mask R-CNN**
- **Cascade Mask R-CNN**
- **HTC** (Hybrid Task Cascade)

Le feature maps dei diversi stage vengono usate:

$
\{\mathbf{F}_1, \mathbf{F}_2, \mathbf{F}_3, \mathbf{F}_4\}
$

con risoluzioni:

$
\left\{\frac{H}{4} \times \frac{W}{4}, \frac{H}{8} \times \frac{W}{8}, \frac{H}{16} \times \frac{W}{16}, \frac{H}{32} \times \frac{W}{32}\right\}
$

### Semantic Segmentation

Usato in **UperNet** per segmentazione:

Le feature gerarchiche vengono combinate con:
- **FPN** (Feature Pyramid Network)
- **PPM** (Pyramid Pooling Module)

Output finale:

$
\mathbf{S} \in \mathbb{R}^{B \times K_{\text{seg}} \times H \times W}
$

dove $K_{\text{seg}}$ è il numero di classi di segmentazione.

## Vantaggi di Swin Transformer

### 1. Efficienza Computazionale

La window-based attention riduce la complessità da quadratica a lineare rispetto alla risoluzione:

$
O(N^2) \rightarrow O(N)
$

### 2. Hierarchical Representation

Come le CNN, Swin costruisce rappresentazioni gerarchiche che sono utili per task dense come detection e segmentation.

### 3. Flessibilità

Può processare immagini di diverse dimensioni (con alcuni aggiustamenti) grazie alla struttura a finestre.

### 4. State-of-the-Art Performance

Su ImageNet-1K:
- Swin-B: 83.5% top-1 accuracy
- Swin-L: 84.5% top-1 accuracy

Su COCO object detection:
- Swin-L: 58.7 box AP (Cascade Mask R-CNN)

## Limitazioni e Considerazioni

### 1. Window Size Trade-off

- Window piccole: meno recettività globale, più efficienza
- Window grandi: maggiore recettività, meno efficienza

Il valore $M = 7$ è un compromesso empirico.

### 2. Shifted Windows Overhead

Lo shift ciclico e il mascheramento aggiungono overhead computazionale, anche se recuperato dall'efficienza locale.

### 3. Memory Requirements

Nonostante l'efficienza, i modelli grandi (Swin-L) richiedono ancora molta memoria:

$
\text{Memory} \propto B \cdot H \cdot W \cdot C \cdot L
$

### 4. Fixed Window Size

Il window size fisso può non essere ottimale per tutte le scale di oggetti nell'immagine.

## Conclusioni

Swin Transformer rappresenta un breakthrough nell'applicazione dei Transformer alla computer vision, combinando:

1. **Efficienza** delle window-based attention
2. **Flessibilità** delle shifted windows per catturare dipendenze cross-window
3. **Struttura gerarchica** simile alle CNN per task multi-scala
4. **Performance state-of-the-art** su molteplici benchmark

Le dimensioni dei tensori attraverso la rete mostrano una progressione logica:

$
\mathbb{R}^{B \times 3 \times 224 \times 224} \rightarrow \mathbb{R}^{B \times 3136 \times 96} \rightarrow \mathbb{R}^{B \times 784 \times 192} \rightarrow \mathbb{R}^{B \times 196 \times 384} \rightarrow \mathbb{R}^{B \times 49 \times 768} \rightarrow \mathbb{R}^{B \times K}
$

Ogni transizione dimezza la risoluzione spaziale e raddoppia i canali, creando rappresentazioni sempre più astratte e semantiche dell'input, mentre la window-based attention mantiene la complessità computazionale gestibile anche ad alte risoluzioni.
