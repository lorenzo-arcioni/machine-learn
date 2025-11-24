# CNN14 (PANN): Architettura per Audio Classification

## Introduzione

La CNN14 è un'architettura di rete neurale convoluzionale facente parte della famiglia PANN (Pretrained Audio Neural Networks), progettata specificamente per l'analisi e la classificazione di segnali audio. Il nome "CNN14" deriva dal fatto che contiene 14 layer pesati (6 ConvBlock × 2 conv ciascuno + 2 layer fully connected).

L'architettura trasforma un segnale audio grezzo in un embedding denso ad alta dimensionalità, catturando caratteristiche acustiche complesse attraverso una gerarchia di rappresentazioni sempre più astratte.

## Panoramica del Pipeline

Il processo di elaborazione si articola in diverse fasi:

1. **Estrazione dello Spettrogramma** - conversione da dominio temporale a tempo-frequenza
2. **Estrazione Log-Mel** - rappresentazione percettivamente motivata
3. **Normalizzazione Batch** - standardizzazione delle feature
4. **Data Augmentation** (solo training) - SpecAugmentation e Mixup
5. **Elaborazione Convoluzionale** - 6 blocchi convoluzionali gerarchici
6. **Pooling Globale** - aggregazione temporale
7. **Classificazione** - layer fully connected

## 1. Input: Forma d'Onda Audio

L'input alla rete è un segnale audio campionato digitalmente:

$$
\mathbf{s} \in \mathbb{R}^{B \times N_{samples}}
$$

dove:
- $B$ = dimensione del batch
- $N_{samples}$ = numero di campioni audio (tipicamente $480000$ per 10 secondi a 48 kHz)

**Intuizione**: Il segnale audio è una sequenza di valori che rappresentano l'ampiezza della pressione sonora nel tempo. È la rappresentazione più grezza del suono.

## 2. Estrazione dello Spettrogramma

### 2.1 Short-Time Fourier Transform (STFT)

La STFT applica la trasformata di Fourier a finestre temporali sovrapposte del segnale:

```python{visible}
self.spectrogram_extractor = Spectrogram(
    n_fft=window_size,      # dimensione della FFT (es. 2048)
    hop_length=hop_size,    # spostamento tra finestre (es. 320)
    win_length=window_size, # lunghezza della finestra
    window='hann',          # finestra di Hann
    center=True,            # padding centrato
    pad_mode='reflect',     # riflessione ai bordi
    freeze_parameters=True  # parametri non addestrabili
)
```

**Matematicamente**:

$$
X[k, n] = \sum_{m=0}^{N-1} x[m + nH] \cdot w[m] \cdot e^{-j2\pi km/N}
$$

dove:
- $k$ = bin di frequenza ($0 \leq k < N/2 + 1$)
- $n$ = frame temporale
- $H$ = hop size
- $w[m]$ = finestra di Hann
- $N$ = dimensione della FFT

**Output dello spettrogramma di potenza**:

$$
\mathbf{S} = |X|^2 \in \mathbb{R}^{B \times 1 \times T_{frames} \times F_{bins}}
$$

Con $N_{fft} = 2048$ e hop size $= 320$:
- $T_{frames} = \lfloor 480000 / 320 \rfloor + 1 = 1501$
- $F_{bins} = 2048/2 + 1 = 1025$

**Intuizione**: Lo spettrogramma decompone il segnale audio nelle sue componenti di frequenza nel tempo. Ogni colonna verticale rappresenta lo "spettro di frequenze" presente in una piccola finestra temporale. È come guardare quali note musicali vengono suonate in ogni istante.

### 2.2 Finestra di Hann

La finestra di Hann riduce le discontinuità ai bordi di ogni frame:

$$
w[m] = 0.5 \left(1 - \cos\left(\frac{2\pi m}{N-1}\right)\right)
$$

**Intuizione**: Senza finestra, il taglio brusco del segnale crea "artefatti spettrali" (frequenze spurie). La finestra di Hann "sfuma" dolcemente il segnale ai bordi, riducendo questi artefatti.

## 3. Log-Mel Spectrogram

### 3.1 Mel Filter Bank

Il banco di filtri Mel converte le frequenze lineari in scala Mel, che approssima meglio la percezione umana delle frequenze:

```python{visible}
self.logmel_extractor = LogmelFilterBank(
    sr=sample_rate,
    n_fft=window_size,
    n_mels=mel_bins,     # numero di bande Mel (es. 64)
    fmin=fmin,           # frequenza minima (es. 50 Hz)
    fmax=fmax,           # frequenza massima (es. 14000 Hz)
    ref=1.0,
    amin=1e-10,
    top_db=None,
    freeze_parameters=True
)
```

**Scala Mel**: Conversione da Hz a Mel:

$$
\text{Mel}(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right)
$$

**Applicazione dei filtri triangolari**:

$$
\mathbf{M}_{mel}[b, 1, t, m] = \sum_{k=0}^{F_{bins}-1} \mathbf{S}[b, 1, t, k] \cdot H_m[k]
$$

dove $H_m[k]$ sono i filtri triangolari nella scala Mel.

**Conversione in log-scale**:

$$
\mathbf{M}_{log}[b, 1, t, m] = 10 \log_{10}(\max(\mathbf{M}_{mel}[b, 1, t, m], \text{amin}))
$$

con $\text{amin} = 10^{-10}$ per evitare $\log(0)$.

**Output**:

$$
\mathbf{M}_{log} \in \mathbb{R}^{B \times 1 \times 1501 \times 64}
$$

**Intuizione**: 
- L'orecchio umano non percepisce le frequenze linearmente: la differenza tra 100 e 200 Hz è molto più evidente della differenza tra 10100 e 10200 Hz
- La scala Mel riflette questa non-linearità percettiva
- I filtri triangolari aggregano bin di frequenza vicini, riducendo la dimensionalità e catturando "bande di frequenza" percettivamente rilevanti
- La scala logaritmica comprime la gamma dinamica, rendendo sia suoni deboli che forti discriminabili dalla rete

## 4. Normalizzazione Batch (bn0)

### 4.1 Trasposizione per Batch Norm

```python{visible}
x = x.transpose(1, 3)  # (B, 1, 1501, 64) → (B, 64, 1501, 1)
x = self.bn0(x)
x = x.transpose(1, 3)  # (B, 64, 1501, 1) → (B, 1, 1501, 64)
```

**Perché questa trasposizione?**

`BatchNorm2d(64)` normalizza lungo la dimensione dei canali (dim=1) e si aspetta 
**64 canali in input**. Il log-Mel spectrogram ha forma (B, 1, 1501, 64), con 
1 canale e 64 bande Mel sull'ultima dimensione.

La trasposizione **temporanea** sposta le 64 bande Mel nella posizione dei canali, 
permettendo a BatchNorm2d di normalizzare ogni banda Mel indipendentemente. 
Dopo la normalizzazione, si ritrasporta alla forma originale.

**Nota tecnica**: `bn0` è definito come `BatchNorm2d(64)`, quindi normalizza 
esattamente 64 "canali", che corrispondono alle nostre 64 bande Mel dopo la 
trasposizione.

### 4.2 Operazione di Batch Normalization

Per ogni banda Mel $m$ (ora un "canale"):

$$
\hat{x}[b, m, t, f] = \gamma_m \frac{x[b, m, t, f] - \mu_m}{\sqrt{\sigma_m^2 + \epsilon}} + \beta_m
$$

dove:
- $\mu_m$ = media su tutto il batch e posizioni spazio-temporali per la banda $m$
- $\sigma_m^2$ = varianza corrispondente
- $\gamma_m, \beta_m$ = parametri apprendibili (scaling e shift)
- $\epsilon = 10^{-5}$ = costante per stabilità numerica

**Calcolo della media**:

$$
\mu_m = \frac{1}{B \cdot T_{frames} \cdot 1} \sum_{b=1}^{B} \sum_{t=1}^{T_{frames}} x[b, m, t, 0]
$$

**Intuizione**: 
- Diverse bande di frequenza possono avere energie molto diverse (es. i bassi sono tipicamente più forti degli alti)
- La batch normalization standardizza ogni banda indipendentemente, con media 0 e varianza 1
- Questo aiuta il training permettendo a tutte le bande di contribuire equamente al gradiente
- I parametri $\gamma$ e $\beta$ permettono alla rete di "reimparare" la scala ottimale per ogni banda

## 5. Data Augmentation (Solo Training)

### 5.1 SpecAugmentation

```python{visible}
self.spec_augmenter = SpecAugmentation(
    time_drop_width=64,    # larghezza maschere temporali
    time_stripes_num=2,    # numero maschere temporali
    freq_drop_width=8,     # larghezza maschere frequenziali
    freq_stripes_num=2     # numero maschere frequenziali
)
```

**Time masking**: Azzera 2 strisce verticali casuali di larghezza massima 64 frame

$$
\mathbf{M}[b, 1, t, m] = 0 \quad \text{per } t \in [t_0, t_0 + w_t]
$$

dove $t_0$ è casuale e $w_t \leq 64$.

**Frequency masking**: Azzera 2 strisce orizzontali casuali di larghezza massima 8 bande

$$
\mathbf{M}[b, 1, t, m] = 0 \quad \text{per } m \in [m_0, m_0 + w_m]
$$

**Intuizione**: 
- Simula occlusioni e distorsioni reali (es. rumori transienti, perdita di banda)
- Forza la rete a non dipendere da specifiche regioni tempo-frequenza
- Migliora la robustezza e generalizzazione
- È come "nascondere" parti dello spettrogramma per insegnare alla rete a riconoscere i suoni anche con informazioni incomplete

### 5.2 Mixup

```python{visible}
if self.training and mixup_lambda is not None:
    x = do_mixup(x, mixup_lambda)
```

**Operazione**:

$$
\tilde{x} = \lambda x_i + (1-\lambda) x_j
$$

$$
\tilde{y} = \lambda y_i + (1-\lambda) y_j
$$

dove $\lambda \sim \text{Beta}(\alpha, \alpha)$ (tipicamente $\alpha = 1$, distribuzione uniforme).

**Intuizione**:
- Mixa linearmente due esempi e le loro etichette
- Crea esempi "ibridi" che forzano la rete a imparare interpolazioni smooth tra classi
- Riduce overfitting e migliora calibrazione delle probabilità
- È come sovrapporre due suoni e chiedere alla rete di riconoscere entrambi proporzionalmente

## 6. Blocchi Convoluzionali Gerarchici

Prima di procedere con i singoli blocchi, è importante comprendere la struttura del `ConvBlock`, che è l'unità fondamentale ripetuta nell'architettura CNN14. Ogni `ConvBlock` è composto da:

1. **Prima convoluzione 2D** con kernel $3 \times 3$, stride $(1,1)$, padding $(1,1)$
2. **Batch Normalization** sulla prima convoluzione
3. **Attivazione ReLU in-place** ($\text{ReLU}$)
4. **Seconda convoluzione 2D** con kernel $3 \times 3$, stride $(1,1)$, padding $(1,1)$
5. **Batch Normalization** sulla seconda convoluzione
6. **Attivazione ReLU in-place**
7. **Pooling operation** (average, max, o avg+max)

### 6.1 Struttura del ConvBlock

Ogni `ConvBlock` implementa il pattern:

```
Input → Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → Pooling → Output
```

**Implementazione matematica**:

Matematicamente, per un `ConvBlock` che trasforma $C_{in}$ canali in $C_{out}$ canali:

$$
\begin{aligned}
\mathbf{h}_1 &= \text{Conv2d}_{C_{in} \rightarrow C_{out}}(\mathbf{x}, \text{kernel}=3, \text{padding}=1, \text{stride}=1) \\
\mathbf{h}_1 &= \text{ReLU}(\text{BN}(\mathbf{h}_1)) \\
\mathbf{h}_2 &= \text{Conv2d}_{C_{out} \rightarrow C_{out}}(\mathbf{h}_1, \text{kernel}=3, \text{padding}=1, \text{stride}=1) \\
\mathbf{h}_2 &= \text{ReLU}(\text{BN}(\mathbf{h}_2)) \\
\mathbf{x}_{out} &= \text{Pool}(\mathbf{h}_2, \text{pool\_size})
\end{aligned}
$$

dove i parametri sono: 

- $\mathbf{x}$: input di dimensione $(C_{in}, H_{in}, W_{in})$
- $C_{in}$: numero di canali dell'input
- $C_{out}$: numero di canali dell'output
- pool_size: dimensione del pooling
- pool_type: tipo di pooling ('avg', 'max', 'avg+max')

**Convoluzione 2D con padding=1**:

Per un kernel $3 \times 3$, il padding di 1 pixel preserva le dimensioni spaziali:

$$
H_{out} = H_{in}, \quad W_{out} = W_{in}
$$

La convoluzione calcola:

$$
y[c_{out}, h, w] = \sum_{c_{in}} \sum_{i=-1}^{1} \sum_{j=-1}^{1} x[c_{in}, h+i, w+j] \cdot K[c_{out}, c_{in}, i+1, j+1]
$$

**Nota**: il bias $+ b[c_{out}]$ è omesso nelle convoluzioni perché la BatchNorm successiva include già un parametro di shift (β), rendendo il bias della conv ridondante. Questo è uno standard nelle architetture moderne.

**Batch Normalization dopo Conv**:

$$
\hat{h}[b, c, h, w] = \gamma_c \frac{h[b, c, h, w] - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c
$$

Normalizza le attivazioni per canale, stabilizzando il training.

**[[ReLU (Rectified Linear Unit)]]**:

$$
\text{ReLU}(x) = \max(0, x)
$$

**Intuizione di ReLU**:
- Introduce non-linearità: senza ReLU, stacking di conv sarebbe equivalente a una singola conv
- "Accende" i neuroni solo per attivazioni positive
- Crea rappresentazioni sparse (molti zeri)
- Permette alla rete di imparare funzioni complesse e gerarchiche

**Average Pooling 2D**:

$$
y[c, h', w'] = \frac{1}{k_h \cdot k_w} \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} x[c, h' \cdot s_h + i, w' \cdot s_w + j]
$$

Con pool_size $(2,2)$ e stride $(2,2)$ (default):

$$
H_{out} = \lfloor H_{in} / 2 \rfloor, \quad W_{out} = \lfloor W_{in} / 2 \rfloor
$$

dove:

- $k_h$ e $k_w$ sono le dimensioni del kernel
- $s_h$ e $s_w$ sono le dimensioni dello stride
- $y$ ha dimensioni $(C_{out}, H_{out}, W_{out})$

**Max Pooling 2D**:

$$
y[c, h', w'] = \max_{i=0}^{k_h-1} \max_{j=0}^{k_w-1} x[c, h' \cdot s_h + i, w' \cdot s_w + j]
$$

Con pool_size $(2,2)$ e stride $(2,2)$ (default):

$$
H_{out} = \lfloor H_{in} / 2 \rfloor, \quad W_{out} = \lfloor W_{in} / 2 \rfloor
$$

dove:

- $k_h$ e $k_w$ sono le dimensioni del kernel
- $s_h$ e $s_w$ sono le dimensioni dello stride
- $y$ ha dimensioni $(C_{out}, H_{out}, W_{out})$

**Avg Pooling 2D + Max Pooling 2D**

Combinazione di Average Pooling 2D e Max Pooling 2D. Da notare che entrambi devono avere lo stesso pool_size e stride.

$$
y[c, h', w'] = \text{AvgPooling\_2D}(x, pool\_size, stride) + \text{MaxPooling\_2D}(x, pool\_size, stride)
$$

**Intuizione del pooling**:
- Riduce le dimensioni spaziali, creando rappresentazioni più compatte
- Introduce invarianza locale: piccoli spostamenti nell'input non cambiano drasticamente l'output
- Aumenta il "campo recettivo": neuroni nei layer successivi "vedono" porzioni sempre più grandi dell'input originale
- Average pooling preserva informazione globale sulla presenza di feature (vs max pooling che preserva solo la più forte)

**Riassumendo:**
La convoluzione 2D con padding $(1,1)$ preserva le dimensioni spaziali:

$$
\text{Se } \mathbf{x}_{in} \in \mathbb{R}^{B \times C_{in} \times H \times W} \text{, dopo Conv2D+BatchNorm+ReLU: } \mathbf{h} \in \mathbb{R}^{B \times C_{out} \times H \times W}
$$

Il pooling con pool_size $(2,2)$ dimezza le dimensioni spaziali:

$$
\text{Se } \mathbf{h} \in \mathbb{R}^{B \times C \times H \times W} \text{, dopo pool: } \mathbf{x}_{out} \in \mathbb{R}^{B \times C \times \lfloor H/2 \rfloor \times \lfloor W/2 \rfloor}.
$$

### 6.2 Progressione dei ConvBlock

**ConvBlock 1**: $1 \rightarrow 64$ canali

Il primo blocco convoluzionale espande il canale singolo del Mel spectrogram a 64 feature maps:

```python{visible}
x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
x = F.dropout(x, p=0.2, training=self.training)
```

$$
\mathbf{x}_1 \in \mathbb{R}^{B \times 1 \times 1501 \times 64} \rightarrow \boxed{\text{ConvBlock1}} \rightarrow \mathbf{x}_2 \in \mathbb{R}^{B \times 64 \times 750 \times 32}
$$

- Input: log-Mel spectrogram (1 canale)
- Output: 64 feature maps
- Riduzione: $1501 \rightarrow 750$ frame, $64 \rightarrow 32$ bande Mel

**Dropout con p=0.2:**

$$
\mathbf{x}_2 = \text{Dropout}_{0.2}(\mathbf{x}_2)
$$

Durante il training, ogni elemento ha probabilità 0.2 di essere azzerato e i valori rimanenti vengono scalati di $1/0.8 = 1.25$ per mantenere l'aspettativa.

**Intuizione**: Il primo layer impara feature di basso livello come edge detector, rivelatori di energia locale, pattern di texture semplici.

**ConvBlock 2**: $64 \rightarrow 128$ canali

```python{visible}
x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
x = F.dropout(x, p=0.2, training=self.training)
```

$$
\mathbf{x}_2 \in \mathbb{R}^{B \times 64 \times 750 \times 32} \rightarrow \boxed{\text{ConvBlock2}} \rightarrow \mathbf{x}_3 \in \mathbb{R}^{B \times 128 \times 375 \times 16}
$$

$$
\mathbf{x}_3 = \text{Dropout}_{0.2}(\mathbf{x}_3)
$$

**Intuizione**: Combina feature di basso livello in pattern più complessi (es. sequenze di armoniche, transizioni temporali specifiche).

**ConvBlock 3**: $128 \rightarrow 256$ canali

```python{visible}
x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
x = F.dropout(x, p=0.2, training=self.training)
```

$$
\mathbf{x}_3 \in \mathbb{R}^{B \times 128 \times 375 \times 16} \rightarrow \boxed{\text{ConvBlock3}} \rightarrow \mathbf{x}_4 \in \mathbb{R}^{B \times 256 \times 187 \times 8}
$$

$$
\mathbf{x}_4 = \text{Dropout}_{0.2}(\mathbf{x}_4)
$$

**Intuizione**: Inizia a riconoscere "parti di suoni" (es. inizio di una nota, parte di una parola, pattern ritmici).

**ConvBlock 4**: $256 \rightarrow 512$ canali

```python{visible}
x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
x = F.dropout(x, p=0.2, training=self.training)
```

$$
\mathbf{x}_4 \in \mathbb{R}^{B \times 256 \times 187 \times 8} \rightarrow \boxed{\text{ConvBlock4}} \rightarrow \mathbf{x}_5 \in \mathbb{R}^{B \times 512 \times 93 \times 4}
$$

$$
\mathbf{x}_5 = \text{Dropout}_{0.2}(\mathbf{x}_5)
$$

**Intuizione**: Riconosce strutture acustiche di medio livello (es. frasi musicali brevi, fonemi, eventi sonori specifici).

**ConvBlock 5**: $512 \rightarrow 1024$ canali

```python{visible}
x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
x = F.dropout(x, p=0.2, training=self.training)
```

$$
\mathbf{x}_5 \in \mathbb{R}^{B \times 512 \times 93 \times 4} \rightarrow \boxed{\text{ConvBlock5}} \rightarrow \mathbf{x}_6 \in \mathbb{R}^{B \times 1024 \times 46 \times 2}
$$

$$
\mathbf{x}_6 = \text{Dropout}_{0.2}(\mathbf{x}_6)
$$

**Intuizione**: Cattura pattern di alto livello e context (es. stile musicale, tipo di speaker, ambiente acustico).

**ConvBlock 6**: $1024 \rightarrow 2048$ canali (pooling 1×1)

```python{visible}
x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
x = F.dropout(x, p=0.2, training=self.training)
```

$$
\mathbf{x}_6 \in \mathbb{R}^{B \times 1024 \times 46 \times 2} \rightarrow \boxed{\text{ConvBlock6}} \rightarrow \mathbf{x}_7 \in \mathbb{R}^{B \times 2048 \times 46 \times 2}
$$

$$
\mathbf{x}_7 = \text{Dropout}_{0.2}(\mathbf{x}_7)
$$

**Nota cruciale**: Il pooling $(1,1)$ **non riduce** le dimensioni spaziali. Serve solo per:
- Aumentare la capacità di rappresentazione (2048 canali)
- Applicare dropout per regolarizzazione
- Mantenere coerenza architetturale

**Intuizione**: Questo layer finale di feature extraction crea una rappresentazione molto ricca e astratta, con 2048 dimensioni che codificano caratteristiche semantiche di alto livello del contenuto audio.

## 7. Pooling Temporale e Globale

### 7.1 Frequency Pooling


Dopo ConvBlock6, abbiamo $\mathbf{x}_7 \in \mathbb{R}^{B \times 2048 \times 46 \times 2}$.

**Operazione**:

$$
\mathbf{x}_{freq}[b, c, t] = \frac{1}{2}\sum_{f=0}^{1} \mathbf{x}_7[b, c, t, f] = \frac{\mathbf{x}_7[b, c, t, 0] + \mathbf{x}_7[b, c, t, 1]}{2}
$$

```python{visible}
x = torch.mean(x, dim=3)
```

**Output**:

$$
\mathbf{x}_{freq} \in \mathbb{R}^{B \times 2048 \times 46}
$$

**Intuizione**:
- A questo punto, le 2 "bande di frequenza" rimaste sono altamente astratte (non più frequenze fisiche)
- Fare la media collassa completamente l'informazione frequenziale
- Otteniamo una sequenza temporale pura: 46 frame, ciascuno con 2048 feature
- È come avere una "storia" del suono in 46 "capitoli", dove ogni capitolo ha 2048 caratteristiche

### 7.2 Global Temporal Pooling (Max + Average)

**Max Pooling temporale**:

```python{visible}
(x1, _) = torch.max(x, dim=2)
```

$$
\mathbf{x}_{max}[b, c] = \max_{t=1}^{46} \mathbf{x}_{freq}[b, c, t]
$$

**Output**: $\mathbf{x}_{max} \in \mathbb{R}^{B \times 2048}$

**Intuizione di Max Pooling**:
- Seleziona l'attivazione **massima** di ogni feature attraverso tutto il clip audio
- Cattura le caratteristiche **più prominenti** o **eventi più forti**
- È invariante alla posizione temporale: non importa *quando* compare un evento, conta solo che ci sia
- Ottimo per rilevare presenza/assenza di pattern specifici (es. "c'è un clacson da qualche parte?")

**Average Pooling temporale**:

```python{visible}
x2 = torch.mean(x, dim=2)
```

$$
\mathbf{x}_{avg}[b, c] = \frac{1}{46}\sum_{t=1}^{46} \mathbf{x}_{freq}[b, c, t]
$$

**Output**: $\mathbf{x}_{avg} \in \mathbb{R}^{B \times 2048}$

**Intuizione di Average Pooling**:
- Calcola l'attivazione **media** di ogni feature attraverso tutto il clip
- Cattura la **distribuzione globale** e le caratteristiche **persistenti**
- Sensibile alla durata e frequenza di occorrenza di pattern
- Ottimo per texture e caratteristiche diffuse (es. "questa musica è prevalentemente ritmica?")

**Combinazione (somma element-wise)**:

```python{visible}
x = x1 + x2
```

$$
\mathbf{x}_{global}[b, c] = \mathbf{x}_{max}[b, c] + \mathbf{x}_{avg}[b, c]
$$

**Output**: $\mathbf{x}_{global} \in \mathbb{R}^{B \times 2048}$

**Intuizione della combinazione**:
- Unisce due "punti di vista" complementari sullo stesso audio
- Max cattura eventi salienti e picchi
- Average cattura statistiche globali e background
- La somma permette alla rete di pesare entrambi gli aspetti
- È come avere sia il "momento clou" che il "riassunto generale" di ogni caratteristica

Questo è l'**embedding audio finale**: un vettore di 2048 dimensioni che riassume tutto il contenuto acustico del clip.

## 8. Layer Fully Connected e Classificazione

### 8.1 Dropout Pre-FC

$$
\mathbf{x}_{global} = \text{Dropout}_{0.5}(\mathbf{x}_{global})
$$

```python{visible}
x = F.dropout(x, p=0.5, training=self.training)
```

**Dropout aggressivo al 50%**: spegne casualmente metà dei neuroni dell'embedding.

**Intuizione**: 
- Prima dei layer densi, il dropout deve essere più forte (50% vs 20% nei conv)
- I fully connected layer hanno molti più parametri e sono più soggetti a overfitting
- Questo è il dropout più importante per la regolarizzazione

### 8.2 Primo Layer Fully Connected (fc1)

```python{visible}
self.fc1 = nn.Linear(2048, 2048, bias=True)

x = F.relu_(self.fc1(x)) # (B, 2048)
```

**Operazione**:

**Operazione**:

$$
\mathbf{h}_{fc1} = \text{ReLU}(\mathbf{x}_{global} \mathbf{W}_1^\top + \mathbf{b}_1)
$$

dove $\mathbf{W}_1 \in \mathbb{R}^{2048 \times 2048}$, $\mathbf{b}_1 \in \mathbb{R}^{2048}$.

**Dettaglio del linear layer**:

$$
\mathbf{h}_{fc1}[b, j] = \text{ReLU} \left( \sum_{i=1}^{2048} \mathbf{x}_{global}[b, i] \cdot \mathbf{W}_1[j, i] + \mathbf{b}_1[j] \right), \quad b = 1, \dots, B, \; j = 1, \dots, 2048
$$

**Intuizione**:
- Proietta l'embedding in un nuovo spazio 2048D
- Permette interazioni non-lineari tra tutte le 2048 feature
- ReLU introduce non-linearità critica
- Questo layer "ragiona" sull'embedding: combina e ricombina le feature per creare rappresentazioni ancora più astratte
- È come un layer di "integrazione semantica"

**Output**: $\mathbf{h}_{fc1} \in \mathbb{R}^{B \times 2048}$

### 8.3 Secondo Dropout

```python{visible}
self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

embedding = F.dropout(x, p=0.5, training=self.training) # (B, 2048)
clipwise_output = torch.sigmoid(self.fc_audioset(x)) # (B, classes_num)
```

**Dettaglio importante**: Il secondo dropout (50%) viene applicato per creare 
l'`embedding` che viene restituito, ma **NON** viene applicato a $\mathbf{x}_{global}$ prima della 
classificazione. Infatti, `fc_audioset` opera su $\mathbf{x}_{global}$ (output di fc1+ReLU senza 
il secondo dropout) e non su $embedding$.

Questo significa che:
- **Embedding restituito**: Più regolarizzato (con dropout)
- **Classificazione**: Usa tutte le attivazioni (senza dropout)

Questa scelta permette di avere un embedding più robusto per task downstream 
mantenendo la massima informazione per la classificazione principale.

### 8.4 Layer di Classificazione (fc_audioset)

```python{visible}
clipwise_output = torch.sigmoid(self.fc_audioset(x))
```

**Operazione**:

$$
\mathbf{z} = \mathbf{h}_{fc1} \mathbf{W}_2^\top + \mathbf{b}_2
$$

dove $\mathbf{W}_2 \in \mathbb{R}^{C \times 2048}$, $\mathbf{b}_2 \in \mathbb{R}^{C}$, $C$ = numero di classi.

**Sigmoid activation** (multi-label classification):

$$
\mathbf{p}[b, c] = \sigma(\mathbf{z}[b, c]) = \frac{1}{1 + e^{-\mathbf{z}[b, c]}}
$$

**Output**: $\mathbf{p} \in \mathbb{R}^{B \times C}$, con $p[b, c] \in (0, 1)$

**Intuizione**:
- Proietta l'embedding nelle dimensioni delle classi target
- Sigmoid converte logits in probabilità indipendenti per ogni classe
- **Multi-label**: ogni classe può essere presente/assente indipendentemente (es. un audio può contenere sia "musica" che "parlato")
- Ogni $p[b, c]$ rappresenta la confidenza che la classe $c$ sia presente nel sample $b$
- Training: Binary Cross-Entropy Loss su ogni classe

## 9. Output della Rete

```python{visible}
output_dict = {
    'clipwise_output': clipwise_output,  # (B, classes_num)
    'embedding': embedding                # (B, 2048)
}
```

**clipwise_output**: Probabilità per ciascuna classe audio (es. probabilità che l'audio contenga "pianoforte", "voce umana", "traffico", ecc.)

**embedding**: Rappresentazione ad alta dimensionalità dell'audio, utile per:
- Transfer learning
- Similarity search
- Clustering
- Downstream tasks

## 10. Architettura Complessiva: Visualizzazione del Flusso

<img src="/images/tikz/cde8894f2cc2931a043bac786c150b9b.svg" style="display: block; width: 100%; height: auto; max-height: 600px;" class="tikz-svg" />
