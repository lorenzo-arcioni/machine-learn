# HTS-AT: Hierarchical Token-Semantic Audio Transformer

## Introduzione

HTS-AT (Hierarchical Token-Semantic Audio Transformer) è un'architettura basata su **[[Swin Transformer]]** progettata specificamente per la classificazione e il rilevamento di eventi sonori. A differenza delle architetture CNN tradizionali, HTS-AT utilizza meccanismi di **self-attention** gerarchici per catturare dipendenze a lungo raggio negli spettrogrammi audio.

Il nome deriva da:
- **Hierarchical**: elaborazione multi-scala attraverso layer successivi
- **Token-Semantic**: rappresentazione basata su token (patch) con significato semantico
- **Audio Transformer**: architettura transformer adattata per l'audio

L'architettura elabora l'audio attraverso una pipeline che:
1. Converte il segnale audio in rappresentazione tempo-frequenza (log-mel spectrogram)
2. Divide lo spettrogramma in patch non sovrapposte (tokenizzazione)
3. Elabora le patch attraverso blocchi Swin Transformer gerarchici
4. Produce output di classificazione sia frame-wise che clip-wise

## Panoramica del Pipeline

1. **Estrazione dello Spettrogramma** - STFT per conversione tempo-frequenza
2. **Estrazione Log-Mel** - rappresentazione percettiva con mel-scale
3. **Normalizzazione Batch** - standardizzazione delle feature mel
4. **Data Augmentation** (solo training) - SpecAugmentation e Mixup
5. **Reshape a Immagine 2D** - preparazione per patch embedding
6. **Patch Embedding** - tokenizzazione dello spettrogramma
7. **Position Embedding** (opzionale) - encoding posizionale assoluto
8. **Blocchi Swin Transformer** - elaborazione gerarchica con attention
9. **Pooling e Classificazione** - aggregazione e predizione

<img src="https://ar5iv.labs.arxiv.org/html/2202.00874/assets/x1.png" alt="Panoramica del Pipeline" style="display: block; margin-left: auto; margin-right: auto;">

## 1. Input: Forma d'Onda Audio

L'input è un segnale audio campionato digitalmente:

$$
\mathbf{s} \in \mathbb{R}^{B \times N_{samples}}
$$

dove:
- $B$ = dimensione del batch
- $N_{samples}$ = numero di campioni audio (tipicamente variabile, es. 10 secondi a 32 kHz = 320000 campioni)

Il segnale $\mathbf{s}$ rappresenta l’**onda sonora campionata nel tempo**, contenente tutte le informazioni acustiche della clip, incluse frequenze, intensità e dinamiche temporali. A questo livello, il modello non ha ancora estratto caratteristiche significative: ogni campione è semplicemente un valore numerico che descrive l’ampiezza del segnale in un dato istante.  

Prima di essere passato al successivo step (feature extraction), $\mathbf{s}$ può subire alcune operazioni preliminari come:  
- **Normalizzazione** per garantire che l’ampiezza rientri in un intervallo standard (ad esempio [-1, 1]),  
- **Rimozione del DC offset** per centrare la waveform intorno a zero,  
- **Eventuale trimming o padding** per uniformare la durata delle clip all’interno del batch.  

Questo passaggio è cruciale perché definisce la **rappresentazione numerica di base** su cui le successive trasformazioni (come trasformate di Fourier, mel-spectrogrammi o convoluzioni) potranno estrarre caratteristiche semantiche e acustiche rilevanti per l’allenamento contrastivo.

## 2. Estrazione dello Spettrogramma

### 2.1 Short-Time Fourier Transform (STFT)

```python
self.spectrogram_extractor = Spectrogram(
    n_fft=config.window_size,      # es. 1024
    hop_length=config.hop_size,     # es. 320
    win_length=config.window_size,
    window='hann',
    center=True,
    pad_mode='reflect',
    freeze_parameters=True
)
```

**Matematicamente**:

$$
\mathbf{X}_{STFT} = \text{STFT}(\mathbf{s}) \implies \mathbf{X}_{STFT}[b, m, k] = \sum_{n=0}^{N-1} \mathbf{s}[b, n + mH - \underbrace{\frac{N}{2}}_\text{center=True}] \cdot w[n] \cdot e^{-j 2 \pi k n/N_{fft}}
$$

Le dimensioni del tensore risultante sono:

$$
\mathbf{X_{STFT}} \in \mathbb{C}^{B \times C \times T_{frames} \times F_{bins}}
$$

Con:

- Dimensione della finestra $N=1024$
- Numero di punti della FFT $N_{fft} = 1024$
- $C$ numero di canali (1 per il segnale audio mono e 2 per il segnale audio stereo)
- hop size $= 320$ (configurazione tipica)
- $T_{frames} = \lfloor N_{samples} / 320 \rfloor + 1$
- $F_{bins} = N_{fft}/2 + 1 = 1024/2 + 1 = 513$

Se $N_{samples} < N_{fft}$, si effettua zero-padding della finestra prima della FFT per aumentare la risoluzione in frequenza.

**Output dello spettrogramma di potenza**:

Poiché la STFT produce valori complessi, si calcola il **power spectrogram** per ottenere una rappresentazione reale e positiva:

Calcolo del modulo quadrato per ogni elemento complesso:

$$
\mathbf{S} = |\mathbf{X}_{STFT}|^2 = \text{Re}(\mathbf{X}_{STFT})^2 + \text{Im}(\mathbf{X}_{STFT})^2 \in \mathbb{R}^{B \times 1 \times T_{frames} \times F_{bins}}
$$

Dove per ogni elemento:

$$
\mathbf{S}[b, 1, t, f] = (\text{Re}(\mathbf{X}_{STFT}[b, 1, t, f]))^2 + (\text{Im}(\mathbf{X}_{STFT}[b, 1, t, f]))^2
$$

Questa operazione element-wise converte i valori complessi della STFT in valori reali positivi che rappresentano l'energia del segnale.

Le dimensioni rimangono le stesse.


Con $N_{fft} = 1024$ e hop size $= 320$ (configurazione tipica):
- $T_{frames} = \lfloor N_{samples} / 320 \rfloor + 1$
- $F_{bins} = 1024/2 + 1 = 513$

**Intuizione**: Decompone il segnale nelle sue componenti di frequenza nel tempo, creando una rappresentazione 2D tempo-frequenza.

## 3. Log-Mel Spectrogram

### 3.1 Mel Filter Bank

```python
self.logmel_extractor = LogmelFilterBank(
    sr=config.sample_rate,          # es. 32000 Hz
    n_fft=config.window_size,       # es. 1024
    n_mels=config.mel_bins,         # es. 64
    fmin=config.fmin,               # es. 50 Hz
    fmax=config.fmax,               # es. 14000 Hz
    ref=1.0,
    amin=1e-10,
    top_db=None,
    freeze_parameters=True
)
```

**Scala Mel**:
La scala Mel è una trasformazione non lineare che approssima la percezione umana delle frequenze:

$$
\text{Mel}(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right)
$$

**Applicazione filtri triangolari**:

A partire dallo spettrogramma $\mathbf{S} \in \mathbb{R}^{B \times 1 \times T_{frames} \times F_{bins}}$, si applica una banca di filtri triangolari $H_m[k]$ dove ogni filtro $m$ aggrega i bin di frequenza lineari $k$ per ottenere bande di frequenza percettivamente significative:

$$
\mathbf{M}_{mel} = \mathbf{S}\mathbf{H}^\top \implies \mathbf{M}_{mel}[b, 1, t, m] = \sum_{k=0}^{F_{bins}-1} \mathbf{S}[b, 1, t, k] \cdot H_m[k]
$$

- $H_m[k]$ è la risposta del filtro triangolare $m$-esimo,
- $\mathbf{H} \in \mathbb{R}^{M_{mel} \times F_{bins}}$ è la matrice completa dei filtri Mel ($M_{mel}=64$),
- La somma avviene lungo la dimensione delle frequenze,
- $b = 0, \dots, B-1$, $t = 0, \dots, T_{frames}-1$, $m = 0, \dots, M_{mel}-1$.

**Conversione in log-scale**:

$$
\mathbf{M}_{log}[b, 1, t, m] = 10 \log_{10}(\max(\mathbf{M}_{mel}[b, 1, t, m], \epsilon))
$$

dove $\epsilon = 10^{-10}$ previene logaritmi di zero.

**Output**:

$$
\mathbf{M}_{log} \in \mathbb{R}^{B \times 1 \times T_{frames} \times 64}
$$

**Intuizione**: 
- La scala Mel riflette la percezione non-lineare delle frequenze da parte dell'orecchio umano
- I filtri triangolari aggregano bin di frequenza adiacenti in bande percettivamente significative
- La scala logaritmica comprime la gamma dinamica

## 4. Normalizzazione Batch (bn0)

### 4.1 Trasposizione per Batch Norm

```python
x = x.transpose(1, 3)  # (B, 1, T, 64) → (B, 64, T, 1)
x = self.bn0(x)
x = x.transpose(1, 3)  # (B, 64, T, 1) → (B, 1, T, 64)
```

Dove `self.bn0 = nn.BatchNorm2d(config.mel_bins)` con `config.mel_bins = 64`.

**Perché questa trasposizione?**

`BatchNorm2d(64)` normalizza lungo la dimensione dei canali (dim=1). Il log-Mel ha forma $(B, 1, T, 64)$ con 1 canale e 64 bande Mel sull'ultima dimensione. La trasposizione sposta le 64 bande Mel nella posizione dei canali per permettere la normalizzazione indipendente di ogni banda.

### 4.2 Operazione di Batch Normalization

Per ogni banda Mel $m$:

$$
\hat{x}[b, m, t, f] = \gamma_m \frac{x[b, m, t, f] - \mu_m}{\sqrt{\sigma_m^2 + \epsilon}} + \beta_m
$$

dove:
- $\mu_m$ = media su batch e posizioni spazio-temporali per la banda $m$
- $\sigma_m^2$ = varianza corrispondente
- $\gamma_m, \beta_m$ = parametri apprendibili
- $\epsilon = 10^{-5}$ = costante per stabilità numerica

**Intuizione**: Standardizza ogni banda di frequenza indipendentemente, permettendo a tutte le bande di contribuire equamente al training.

## 5. Data Augmentation (Solo Training)

### 5.1 SpecAugmentation

```python
self.spec_augmenter = SpecAugmentation(
    time_drop_width=64,
    time_stripes_num=2,
    freq_drop_width=8,
    freq_stripes_num=2
)
```

**Time masking**: Azzera 2 strisce verticali casuali di larghezza massima 64 frame:

$$
\mathbf{M}[b, 1, t, m] = 0 \quad \text{per } t \in [t_0, t_0 + w_t], \quad w_t \leq 64
$$

**Frequency masking**: Azzera 2 strisce orizzontali casuali di larghezza massima 8 bande:

$$
\mathbf{M}[b, 1, t, m] = 0 \quad \text{per } m \in [m_0, m_0 + w_m], \quad w_m \leq 8
$$

**Intuizione**: Simula occlusioni e distorsioni reali, forzando la rete a non dipendere da specifiche regioni tempo-frequenza.

### 5.2 Mixup

```python
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

dove $\lambda \sim \text{Beta}(\alpha, \alpha)$.

**Intuizione**: Crea esempi ibridi che migliorano la generalizzazione e riducono l'overfitting.

## 6. Reshape a Immagine 2D

Prima del patch embedding, lo spettrogramma viene trasformato in un formato "immagine" con dimensioni specifiche.

### 6.1 Parametri di Configurazione

```python
self.spec_size = 256              # dimensione target dell'immagine
self.freq_ratio = self.spec_size // config.mel_bins  # es. 256 // 64 = 4
```

Questo significa:
- **Target height**: $256$ pixel
- **Target width**: $256$ pixel  
- **Freq ratio**: $4$ (numero di ripetizioni delle bande Mel in verticale)

### 6.2 Funzione reshape_wav2img

```python
def reshape_wav2img(self, x):
    B, C, T, F = x.shape  # es. (B, 1, T_frames, 64)
    target_T = int(self.spec_size * self.freq_ratio)  # 256 * 4 = 1024
    target_F = self.spec_size // self.freq_ratio      # 256 // 4 = 64
```

**Step 1: Interpolazione temporale** (se necessario)

Se $T < 1024$:

$$
x = \text{interpolate}(x, (1024, F), \text{mode}=\text{bicubic})
$$

**Step 2: Interpolazione frequenziale** (se necessario)

Se $F < 64$ (in realtà già 64, quindi salta):

$$
x = \text{interpolate}(x, (T, 64), \text{mode}=\text{bicubic})
$$

**Step 3: Reshape con freq_ratio**

```python
x = x.permute(0,1,3,2).contiguous()  # (B, 1, 64, 1024)
x = x.reshape(B, C, F, self.freq_ratio, T // self.freq_ratio)
# (B, 1, 64, 4, 256)
x = x.permute(0,1,3,2,4).contiguous()  # (B, 1, 4, 64, 256)
x = x.reshape(B, C, self.freq_ratio * F, T // self.freq_ratio)
# (B, 1, 256, 256)
```

**Risultato finale**:

$$
\mathbf{x}_{img} \in \mathbb{R}^{B \times 1 \times 256 \times 256}
$$

**Intuizione**: 
- Lo spettrogramma viene riorganizzato in un'immagine quadrata $256 \times 256$
- Le 64 bande Mel vengono "espanse" verticalmente a 256 pixel (4 ripetizioni)
- La dimensione temporale viene compressa da 1024 a 256 frame
- Questo formato è ideale per l'elaborazione tramite patch embedding

## 7. Patch Embedding

Il patch embedding converte l'immagine $256 \times 256$ in una sequenza di token (patch).

### 7.1 Configurazione

```python
self.patch_size = 4         # dimensione della patch
self.patch_stride = (4, 4)  # stride della patch (non sovrapposto)

self.patch_embed = PatchEmbed(
    img_size=self.spec_size,      # 256
    patch_size=self.patch_size,   # 4
    in_chans=self.in_chans,       # 1
    embed_dim=self.embed_dim,     # es. 96
    norm_layer=self.norm_layer,
    patch_stride=patch_stride     # (4, 4)
)
```

### 7.2 Struttura del PatchEmbed

Il modulo `PatchEmbed` contiene:

1. **Conv2D per proiezione**:

```python
padding = ((patch_size[0] - patch_stride[0]) // 2, 
           (patch_size[1] - patch_stride[1]) // 2)
# Con patch_size=4 e stride=4: padding = (0, 0)

self.proj = nn.Conv2d(
    in_chans,      # 1
    embed_dim,     # 96
    kernel_size=patch_size,      # (4, 4)
    stride=patch_stride,         # (4, 4)
    padding=padding              # (0, 0)
)
```

2. **Normalizzazione** (opzionale):

```python
self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
```

### 7.3 Operazione di Patch Embedding

**Input**: $\mathbf{x}_{img} \in \mathbb{R}^{B \times 1 \times 256 \times 256}$

**Proiezione Conv2D**:

La convoluzione con kernel $4 \times 4$ e stride $(4, 4)$ divide l'immagine in patch non sovrapposte:

$$
\mathbf{x}_{conv} = \text{Conv2d}(\mathbf{x}_{img})
$$

**Dimensioni output della convoluzione**:

$$
H_{out} = \frac{256 - 4 + 2 \cdot 0}{4} + 1 = 64
$$

$$
W_{out} = \frac{256 - 4 + 2 \cdot 0}{4} + 1 = 64
$$

Quindi: $\mathbf{x}_{conv} \in \mathbb{R}^{B \times 96 \times 64 \times 64}$

**Flatten e Transpose**:

```python
x = x.flatten(2).transpose(1, 2)  # (B, 96, 64, 64) → (B, 96, 4096) → (B, 4096, 96)
```

$$
\mathbf{x}_{patches} \in \mathbb{R}^{B \times 4096 \times 96}
$$

dove $4096 = 64 \times 64$ è il numero totale di patch.

**Normalizzazione** (se presente):

$$
\mathbf{x}_{embed} = \text{LayerNorm}(\mathbf{x}_{patches})
$$

**Output finale del patch embedding**:

$$
\mathbf{x}_{embed} \in \mathbb{R}^{B \times 4096 \times 96}
$$

**Intuizione**:
- Ogni patch $4 \times 4$ dello spettrogramma originale diventa un token di dimensione 96
- L'immagine $256 \times 256$ viene divisa in $64 \times 64 = 4096$ patch
- Ogni token rappresenta una regione locale tempo-frequenza dello spettrogramma
- Questo è analogo alla tokenizzazione del testo in NLP, ma applicato a regioni spaziali

### 7.4 Grid Size e Patches Resolution

```python
self.grid_size = (img_size[0] // patch_stride[0], 
                  img_size[1] // patch_stride[1])
# self.grid_size = (256 // 4, 256 // 4) = (64, 64)

self.patches_resolution = self.grid_size  # (64, 64)
self.num_patches = 64 * 64 = 4096
```

Questi valori saranno usati dai layer successivi per operazioni che richiedono la struttura spaziale 2D.

## 8. Position Embedding (Opzionale)

### 8.1 Absolute Position Embedding

```python
if self.ape:  # Absolute Position Embedding
    self.absolute_pos_embed = nn.Parameter(
        torch.zeros(1, num_patches, self.embed_dim)
    )
    trunc_normal_(self.absolute_pos_embed, std=.02)
```

Se abilitato (`ape=True`):

$$
\mathbf{x}_{pos} = \mathbf{x}_{embed} + \mathbf{P}_{abs}
$$

dove $\mathbf{P}_{abs} \in \mathbb{R}^{1 \times 4096 \times 96}$ è un parametro apprendibile.

**Intuizione**: 
- I transformer sono permutation-invariant (non hanno nozione di ordine)
- Il position embedding aggiunge informazione sulla posizione spaziale di ogni patch
- Questo aiuta il modello a distinguere patch in diverse posizioni dello spettrogramma
- In HTS-AT, l'uso è opzionale perché i Swin Transformer usano **relative position bias** intrinseco

### 8.2 Position Dropout
Applicato sia durante il training che durante l'inferenza.

```python
self.pos_drop = nn.Dropout(p=self.drop_rate)
x = self.pos_drop(x)
```

Applica dropout dopo il position embedding per regolarizzazione.

## 9. Stochastic Depth (Drop Path)

Prima di entrare nei layer, viene definito uno schedule di stochastic depth:

```python
dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
```

**Esempio con configurazione tipica**:
- `depths = [2, 2, 6, 2]` → totale blocchi = $2 + 2 + 6 + 2 = 12$
- `drop_path_rate = 0.1`
- `dpr = [0.000, 0.009, 0.018, 0.027, 0.036, 0.045, 0.055, 0.064, 0.073, 0.082, 0.091, 0.100]`

**Intuizione**: La probabilità di drop aumenta linearmente con la profondità del layer. Questo migliora la regolarizzazione e facilita il training di reti profonde.

Quindi, durante training: Drop Path attivo con probabilità crescente nei layer più profondi. Mentre durante inferenza: Drop Path *completamente disabilitato*.

## 10. Architettura Gerarchica: BasicLayer

HTS-AT è organizzato in **4 stage gerarchici** (configurazione tipica: `depths=[2, 2, 6, 2]`):

```python
self.num_layers = len(self.depths)  # 4
self.layers = nn.ModuleList()

for i_layer in range(self.num_layers):
    layer = BasicLayer(
        dim=int(self.embed_dim * 2 ** i_layer),
        input_resolution=(patches_resolution[0] // (2 ** i_layer),
                          patches_resolution[1] // (2 ** i_layer)),
        depth=self.depths[i_layer],
        num_heads=self.num_heads[i_layer],
        window_size=self.window_size,
        mlp_ratio=self.mlp_ratio,
        qkv_bias=self.qkv_bias,
        qk_scale=self.qk_scale,
        drop=self.drop_rate,
        attn_drop=self.attn_drop_rate,
        drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
        norm_layer=self.norm_layer,
        downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
        use_checkpoint=use_checkpoint,
        norm_before_mlp=self.norm_before_mlp
    )
    self.layers.append(layer)
```

### 10.1 Progressione Gerarchica

Con `embed_dim=96` e `num_heads=[4, 8, 16, 32]`:

| Stage | Input Dim | Num Blocks | Num Heads | Input Resolution | Output Dim | Output Resolution |
|-------|-----------|------------|-----------|------------------|------------|-------------------|
| 1 | 96 | 2 | 4 | (64, 64) | 192 | (32, 32) |
| 2 | 192 | 2 | 8 | (32, 32) | 384 | (16, 16) |
| 3 | 384 | 6 | 16 | (16, 16) | 768 | (8, 8) |
| 4 | 768 | 2 | 32 | (8, 8) | 768 | (8, 8) |

**Note importanti**:
- Ogni stage (tranne l'ultimo) **raddoppia** il numero di canali
- Ogni stage (tranne l'ultimo) **dimezza** la risoluzione spaziale
- Il numero di attention heads aumenta proporzionalmente ai canali
- L'ultimo stage non fa downsampling

**Intuizione**:
- **Elaborazione multi-scala**: ogni stage cattura pattern a diversa granularità
- **Stage 1**: feature di basso livello (texture locali, transizioni frequenziali)
- **Stage 2-3**: feature di medio livello (pattern temporali, strutture armoniche)
- **Stage 4**: feature di alto livello (rappresentazioni semantiche globali)

## 11. BasicLayer: Struttura di uno Stage

Ogni `BasicLayer` contiene:
1. Una sequenza di `SwinTransformerBlock` (es. 2, 2, 6, o 2 blocchi)
2. Un layer di `PatchMerging` (tranne l'ultimo stage)

```python
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, ...):
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(...)
            for i in range(depth)
        ])
        
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, ...)
        else:
            self.downsample = None
```

### 11.1 Forward Pass di BasicLayer

```python
def forward(self, x):
    attns = []
    for blk in self.blocks:
        x, attn = blk(x)
        if not self.training:
            attns.append(attn.unsqueeze(0))
    
    if self.downsample is not None:
        x = self.downsample(x)
    
    if not self.training:
        attn = torch.cat(attns, dim=0)
        attn = torch.mean(attn, dim=0)
    
    return x, attn
```

**Flusso**:
1. Passa attraverso tutti i blocchi Swin Transformer
2. (Opzionale) Salva le attention maps per visualizzazione
3. Applica PatchMerging per downsampling
4. Restituisce output e attention maps aggregate

## 12. SwinTransformerBlock: Il Cuore dell'Architettura

<img src="../../../../../images/sw-mha.png" alt="Immagine di un SwinTransformerBlock" style="display: block; margin-left: auto; margin-right: auto;">

Il `SwinTransformerBlock` è l'unità fondamentale di elaborazione, che implementa:
1. **Window-based Multi-head Self-Attention (W-MSA)** o **Shifted Window MSA (SW-MSA)**
2. **Multi-Layer Perceptron (MLP)**
3. **Residual connections**
4. **Layer Normalization**

### 12.1 Struttura del SwinTransformerBlock

```python
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, 
                 shift_size=0, mlp_ratio=4., ...):
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size  # es. 8
        self.shift_size = shift_size    # 0 per W-MSA, window_size//2 per SW-MSA
        self.mlp_ratio = mlp_ratio      # 4.0
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), 
            num_heads=num_heads, ...
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)  # o BatchNorm1d se norm_before_mlp='bn'
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, ...)
```

**Parametri chiave**:
- `window_size`: dimensione della finestra locale (tipicamente 8)
- `shift_size`: offset per shifted window (0 o `window_size//2`)
- `mlp_ratio`: fattore di espansione del MLP (tipicamente 4)

### 12.2 Alternanza W-MSA e SW-MSA

All'interno di ogni `BasicLayer`, i blocchi alternano tra W-MSA e SW-MSA:

```python
self.blocks = nn.ModuleList([
    SwinTransformerBlock(
        ...,
        shift_size=0 if (i % 2 == 0) else window_size // 2,
        ...
    )
    for i in range(depth)
])
```

- **Blocco pari** (`i=0, 2, 4, ...`): `shift_size=0` → **W-MSA** (window attention senza shift)
- **Blocco dispari** (`i=1, 3, 5, ...`): `shift_size=window_size//2` → **SW-MSA** (shifted window attention)

**Intuizione**: L'alternanza permette di catturare sia interazioni locali che cross-window, superando la limitazione delle finestre isolate.

### 12.3 Forward Pass del SwinTransformerBlock

```python
def forward(self, x):
    H, W = self.input_resolution  # es. (64, 64) nel primo stage
    B, L, C = x.shape             # es. (B, 4096, 96)
    # L deve essere = H * W
    
    shortcut = x
    x = self.norm1(x)
    x = x.view(B, H, W, C)  # reshape a formato spaziale 2D
    
    # Cyclic shift (solo per SW-MSA)
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), 
                               dims=(1, 2))
    else:
        shifted_x = x
    
    # Partition windows
    x_windows = window_partition(shifted_x, self.window_size)
    # (num_windows*B, window_size, window_size, C)
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
    # (num_windows*B, window_size*window_size, C)
    
    # W-MSA/SW-MSA
    attn_windows, attn = self.attn(x_windows, mask=self.attn_mask)
    
    # Merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)
    
    # Reverse cyclic shift (solo per SW-MSA)
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), 
                       dims=(1, 2))
    else:
        x = shifted_x
    
    x = x.view(B, H * W, C)
    
    # Residual connection + DropPath
    x = shortcut + self.drop_path(x)
    
    # FFN (MLP)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    
    return x, attn
```

**Flusso dettagliato**:
1. **Layer Norm 1** + reshape a 2D
2. **Cyclic shift** (solo SW-MSA)
3. **Window partition** (divisione in finestre)
4. **Window Attention**
5. **Window merge** (ricostruzione)
6. **Reverse shift** (solo SW-MSA)
7. **Residual connection 1** con DropPath
8. **Layer Norm 2** + **MLP**
9. **Residual connection 2** con DropPath

### 12.3 Forward Pass (concettuale)

Flusso chiave (concettuale, senza codice):
1. LayerNorm su x_in
2. (eventuale) cyclic shift per SW-MSA
3. partition in finestre
4. window-attention su ogni finestra
5. merge delle finestre e reverse shift
6. residual connection + DropPath (output dell'attenzione)
7. LayerNorm + MLP
8. residual connection + DropPath (output finale del blocco)

### 12.4 Formula semplificata (versione leggibile)

Versione con dettagli di normalizzazione e DropPath:

$$
\boxed{%
x_{\text{out}} = x_{\text{in}} \;+\; \text{DropPath}\big(\text{SW-MSA}(\text{LN}(x_{\text{in}}))\big)
\;+\; \text{DropPath}\big(\text{MLP}(\text{LN}(x_{\text{in}} + \text{SW-MSA}(\text{LN}(x_{\text{in}}))))\big)
}
$$

Versione ancora più sintetica e intuitiva:

$$
\boxed{%
x_{\text{out}} = x_{\text{in}} \;+\; \text{Attn}(x_{\text{in}}) \;+\; \text{MLP}\big(x_{\text{in}} + \text{Attn}(x_{\text{in}})\big)
}
$$

**Dove:**
- $\text{LN}$ = Layer Normalization
- $\text{SW-MSA}$ = Shifted Window Multi-Head Self-Attention (o W-MSA quando non c'è shift)
- $\text{Attn}(x_{\text{in}})$ è una abbreviazione per $\text{SW-MSA}(\text{LN}(x_{\text{in}}))$
- $\text{DropPath}(\cdot)$ = stochastic depth (residual dropout)
- $\text{MLP}(\cdot)$ = feed-forward (due linear + attivazione)

Queste formule riassumono il comportamento operativo del blocco: l'output è la somma residua dell'input più l'aggiornamento apportato dall'attenzione locale e dal successivo MLP di raffinamento.



## 13. Window Partition e Reverse

### 13.1 Window Partition

Divide l'input 2D in finestre non sovrapposte:

```python
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: int (es. 8)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, 
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows
```

**Esempio con H=64, W=64, window_size=8**:

Input: $(B, 64, 64, C)$

Step 1: Reshape
$$
x \rightarrow (B, 8, 8, 8, 8, C)
$$

Step 2: Permute per raggruppare le finestre
$$
x \rightarrow (B, 8, 8, 8, 8, C) \text{ (posizioni: 0,1,3,2,4,5)}
$$

Step 3: View per collassare le finestre
$
x \rightarrow (B \cdot 64, 8, 8, C)
$

dove $64 = (64/8) \times (64/8) = 8 \times 8$ è il numero di finestre.

**Output**: $(num\_windows \cdot B, window\_size, window\_size, C)$ con $num\_windows = 64$

**Intuizione**: L'immagine viene divisa in una griglia di finestre $8 \times 8$, ognuna contenente $8 \times 8 = 64$ token. L'attention sarà calcolata indipendentemente all'interno di ogni finestra.

### 13.2 Window Reverse

Ricostruisce l'immagine 2D dalle finestre:

```python
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: int
        H, W: altezza e larghezza dell'immagine originale
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, 
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x
```

Inverte esattamente le operazioni di `window_partition`.

## 14. WindowAttention: Self-Attention Locale

Il modulo `WindowAttention` implementa la **multi-head self-attention** all'interno di ogni finestra, con l'aggiunta cruciale del **relative position bias**.

### 14.1 Struttura del WindowAttention

```python
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, 
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        self.dim = dim                      # es. 96
        self.window_size = window_size      # (8, 8)
        self.num_heads = num_heads          # es. 4
        head_dim = dim // num_heads         # 96 // 4 = 24
        self.scale = qk_scale or head_dim ** -0.5  # 1/sqrt(24) ≈ 0.204
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), 
                        num_heads)
        )  # (15*15, num_heads) = (225, 4)
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize relative position bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
```

### 14.2 Relative Position Bias: Calcolo degli Indici

Il relative position bias è un'innovazione chiave di Swin Transformer. Invece di usare position embedding assoluti, usa bias **relativi** che dipendono dalla distanza tra token.

**Inizializzazione degli indici** (fatto una volta nel costruttore):

```python
coords_h = torch.arange(self.window_size[0])  # [0, 1, 2, ..., 7]
coords_w = torch.arange(self.window_size[1])  # [0, 1, 2, ..., 7]
coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, 8, 8)
coords_flatten = torch.flatten(coords, 1)  # (2, 64)

# Calcola coordinate relative per ogni coppia di posizioni
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
# (2, 64, 64)

relative_coords = relative_coords.permute(1, 2, 0).contiguous()
# (64, 64, 2)

# Shift per rendere le coordinate positive
relative_coords[:, :, 0] += self.window_size[0] - 1  # +7
relative_coords[:, :, 1] += self.window_size[1] - 1  # +7

# Conversione a indice 1D
relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # *15
relative_position_index = relative_coords.sum(-1)  # (64, 64)

self.register_buffer("relative_position_index", relative_position_index)
```

**Intuizione del calcolo**:
- Per ogni coppia di posizioni $(i, j)$ nella finestra $8 \times 8$, calcola la loro posizione relativa
- La posizione relativa varia da $-7$ a $+7$ in entrambe le dimensioni
- Dopo lo shift: varia da $0$ a $14$ (totale 15 valori)
- L'indice 1D varia da $0$ a $224$ (totale $15 \times 15 = 225$ possibili offset)
- `relative_position_index[i, j]` contiene l'indice nella tabella dei bias per la coppia $(i,j)$

### 14.3 Forward Pass del WindowAttention

```python
def forward(self, x, mask=None):
    """
    Args:
        x: (num_windows*B, N, C) dove N = window_size * window_size = 64
        mask: (num_windows, N, N) o None
    """
    B_, N, C = x.shape  # es. (64*B, 64, 96)
    
    # Step 1: Compute Q, K, V
    qkv = self.qkv(x)  # (B_, N, 3*C) = (64*B, 64, 288)
    qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
    # (64*B, 64, 3, 4, 24)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    # (3, 64*B, 4, 64, 24)
    
    q, k, v = qkv[0], qkv[1], qkv[2]
    # Ciascuno: (64*B, 4, 64, 24)
    
    # Step 2: Scaled dot-product attention
    q = q * self.scale  # Scale by 1/sqrt(d_k)
    attn = (q @ k.transpose(-2, -1))  # (64*B, 4, 64, 64)
```

**Matematica della self-attention**:

Per ogni head $h$:

$
\text{Attention}(Q_h, K_h, V_h) = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}} + B\right) V_h
$

dove:
- $Q_h, K_h, V_h \in \mathbb{R}^{N \times d_k}$ con $N=64$, $d_k=24$
- $B \in \mathbb{R}^{N \times N}$ è il relative position bias
- $\sqrt{d_k} = \sqrt{24} \approx 4.899$

**Step 3: Aggiungi Relative Position Bias**:

```python
    # Retrieve relative position bias
    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)
    ].view(self.window_size[0] * self.window_size[1], 
           self.window_size[0] * self.window_size[1], -1)
    # (64, 64, num_heads) = (64, 64, 4)
    
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    # (4, 64, 64)
    
    attn = attn + relative_position_bias.unsqueeze(0)
    # (64*B, 4, 64, 64) + (1, 4, 64, 64)
```

**Intuizione del relative position bias**:
- Ogni coppia di posizioni $(i,j)$ ha un bias apprendibile diverso per ogni head
- Il bias dipende solo dalla **distanza relativa**, non dalla posizione assoluta
- Questo permette alla rete di apprendere preferenze per interazioni a diverse distanze
- Ad esempio: un head potrebbe imparare a dare più peso a token vicini, un altro a token lontani

**Step 4: Maschera per Shifted Window (opzionale)**:

```python
    if mask is not None:
        nW = mask.shape[0]  # numero di finestre
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + \
               mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)
```

La maschera è usata solo in SW-MSA per impedire l'attenzione tra token di finestre originali diverse (dopo lo shift).

**Step 5: Applica Attention e Proietta**:

```python
    attn = self.attn_drop(attn)
    
    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    # (64*B, 4, 64, 24) @ (64*B, 4, 64, 24) → (64*B, 4, 64, 24)
    # transpose → (64*B, 64, 4, 24)
    # reshape → (64*B, 64, 96)
    
    x = self.proj(x)      # Linear projection
    x = self.proj_drop(x)
    
    return x, attn
```

**Output**: 
- $x \in \mathbb{R}^{B' \times 64 \times 96}$ dove $B' = num\_windows \cdot B$
- $attn \in \mathbb{R}^{B' \times 4 \times 64 \times 64}$ (attention weights)

**Intuizione complessiva del WindowAttention**:
- Ogni finestra di $8 \times 8 = 64$ token viene elaborata indipendentemente
- La multi-head attention permette di catturare diversi tipi di relazioni
- Il relative position bias aggiunge informazione spaziale senza position embedding globali
- Questo design è molto più efficiente della global attention ($O(N^2)$ vs $O(n \cdot (N/n)^2)$ dove $n$ è il numero di finestre)

## 15. Shifted Window Attention (SW-MSA)

Il problema del W-MSA è che le finestre sono isolate: non c'è comunicazione tra finestre diverse. La **Shifted Window Attention** risolve questo problema.

### 15.1 Cyclic Shift

Per i blocchi con `shift_size = window_size // 2 = 4`:

```python
if self.shift_size > 0:
    shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), 
                           dims=(1, 2))
```

**Operazione**:

$
\text{shifted\_x}[b, h, w, c] = x[b, (h + 4) \mod H, (w + 4) \mod W, c]
$

**Visualizzazione dello shift**:

```
Prima dello shift:         Dopo shift di (-4, -4):
┌─────┬─────┐             ┌─────┬─────┐
│  A  │  B  │             │  D  │  C  │
│     │     │             │     │     │
├─────┼─────┤      →      ├─────┼─────┤
│  C  │  D  │             │  B  │  A  │
│     │     │             │     │     │
└─────┴─────┘             └─────┴─────┘
```

Ogni lettera rappresenta una finestra $8 \times 8$.

**Intuizione dello shift**:
- Dopo lo shift, le nuove finestre contengono parti di **4 finestre originali diverse**
- Questo permette la comunicazione cross-window
- Il cyclic shift è computazionalmente gratuito (solo reindexing)

### 15.2 Attention Mask per SW-MSA

Il problema: dopo lo shift, ogni nuova finestra contiene token da finestre originali diverse. Vogliamo permettere attenzione solo tra token della **stessa finestra originale**.

**Creazione della maschera**:

```python
if self.shift_size > 0:
    H, W = self.input_resolution
    img_mask = torch.zeros((1, H, W, 1))  # (1, 64, 64, 1)
    
    h_slices = (slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None))
    w_slices = (slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None))
    
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    
    # img_mask ora contiene 9 regioni numerate da 0 a 8
```

**Visualizzazione della maschera** (con window_size=8, shift_size=4, per una griglia 64x64):

```
┌───────┬───┬───────┬───┐
│   0   │ 1 │   2   │ 3 │
│       │   │       │   │
├───────┼───┼───────┼───┤
│   4   │ 5 │   6   │ 7 │
├───────┼───┼───────┼───┤
│   8   │ 9 │  10   │11 │
│       │   │       │   │
├───────┼───┼───────┼───┤
│  12   │13 │  14   │15 │
└───────┴───┴───────┴───┘
```

Ogni numero identifica una regione della finestra originale.

**Conversione a maschera di attenzione**:

```python
    mask_windows = window_partition(img_mask, self.window_size)
    # (num_windows, window_size*window_size, 1)
    mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
    # (num_windows, 64)
    
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    # (num_windows, 64, 64)
    
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
```

**Risultato**:
- $\text{attn\_mask}[w, i, j] = 0$ se i token $i$ e $j$ provengono dalla stessa regione originale
- $\text{attn\_mask}[w, i, j] = -100$ altrimenti

Quando questo viene aggiunto ai logits di attenzione prima del softmax, i valori $-100$ diventano effettivamente $0$ dopo il softmax (mascheratura dell'attenzione).

### 15.3 Reverse Cyclic Shift

Dopo l'attenzione, lo shift viene invertito:

```python
if self.shift_size > 0:
    x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), 
                   dims=(1, 2))
```

Questo riporta i token alle loro posizioni originali.

**Intuizione complessiva di SW-MSA**:
- Lo shift permette la comunicazione tra finestre diverse
- La maschera garantisce che ogni token attenda solo a token della stessa finestra originale
- L'alternanza W-MSA/SW-MSA crea connessioni sia locali che cross-window
- Questo design mantiene l'efficienza computazionale di W-MSA ma supera la limitazione delle finestre isolate

## 16. Multi-Layer Perceptron (MLP)

Dopo ogni attention block, c'è un MLP per elaborazione non-lineare:

```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
```

**Forward pass**:

```python
def forward(self, x):
    x = self.fc1(x)       # (B, N, C) → (B, N, 4*C)
    x = self.act(x)       # GELU activation
    x = self.drop(x)
    x = self.fc2(x)       # (B, N, 4*C) → (B, N, C)
    x = self.drop(x)
    return x
```

**Con mlp_ratio=4 e dim=96**:

$
\mathbf{x} \in \mathbb{R}^{B \times N \times 96} \xrightarrow{\text{fc1}} \mathbb{R}^{B \times N \times 384} \xrightarrow{\text{GELU}} \mathbb{R}^{B \times N \times 384} \xrightarrow{\text{fc2}} \mathbb{R}^{B \times N \times 96}
$

**GELU (Gaussian Error Linear Unit)**:

$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$

dove $\Phi(x)$ è la CDF della distribuzione normale standard.

**Intuizione**:
- Il MLP espande le feature a 4x la dimensione originale (bottleneck invertito)
- GELU introduce non-linearità smooth (simile a ReLU ma più morbido)
- Dropout per regolarizzazione
- Ogni token viene elaborato indipendentemente (elaborazione point-wise)

## 17. PatchMerging: Downsampling Gerarchico

Al termine di ogni stage (tranne l'ultimo), `PatchMerging` riduce la risoluzione spaziale e aumenta i canali.

### 17.1 Struttura del PatchMerging

```python
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        self.input_resolution = input_resolution  # (H, W)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
```

### 17.2 Forward Pass del PatchMerging

```python
def forward(self, x):
    """
    x: (B, H*W, C)
    """
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even"
    
    x = x.view(B, H, W, C)
    
    # Estrai 4 sottogriglie con stride 2
    x0 = x[:, 0::2, 0::2, :]  # top-left: (B, H/2, W/2, C)
    x1 = x[:, 1::2, 0::2, :]  # bottom-left: (B, H/2, W/2, C)
    x2 = x[:, 0::2, 1::2, :]  # top-right: (B, H/2, W/2, C)
    x3 = x[:, 1::2, 1::2, :]  # bottom-right: (B, H/2, W/2, C)
    
    x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*C)
    x = x.view(B, -1, 4 * C)             # (B, H/2*W/2, 4*C)
    
    x = self.norm(x)
    x = self.reduction(x)                # (B, H/2*W/2, 2*C)
    
    return x
```

**Visualizzazione**:

```
Input (H, W, C):           Output (H/2, W/2, 2*C):
┌───┬───┬───┬───┐         ┌───────┬───────┐
│ 0 │ 2 │ 0 │ 2 │         │  0123 │  0123 │
├───┼───┼───┼───┤   →     │concat │concat │
│ 1 │ 3 │ 1 │ 3 │         ├───────┼───────┤
├───┼───┼───┼───┤         │  0123 │  0123 │
│ 0 │ 2 │ 0 │ 2 │         │concat │concat │
├───┼───┼───┼───┤         └───────┴───────┘
│ 1 │ 3 │ 1 │ 3 │
└───┴───┴───┴───┘
```

Ogni gruppo di 4 patch (0,1,2,3) viene concatenato e poi proiettato.

**Matematica**:

Input: $\mathbf{x} \in \mathbb{R}^{B \times HW \times C}$

Dopo concatenazione: $\mathbf{x}_{cat} \in \mathbb{R}^{B \times \frac{HW}{4} \times 4C}$

Dopo proiezione lineare:

$
\mathbf{x}_{out} = \text{LayerNorm}(\mathbf{x}_{cat}) \cdot \mathbf{W}_{reduction}
$

dove $\mathbf{W}_{reduction} \in \mathbb{R}^{4C \times 2C}$.

Output: $\mathbf{x}_{out} \in \mathbb{R}^{B \times \frac{HW}{4} \times 2C}$

**Esempio con Stage 1 → Stage 2**:
- Input: $(B, 4096, 96)$ con risoluzione $(64, 64)$
- Dopo PatchMerging: $(B, 1024, 192)$ con risoluzione $(32, 32)$

**Intuizione**:
- Riduce risoluzione di 2x in entrambe le dimensioni (downsampling 4x totale)
- Raddoppia il numero di canali (da $C$ a $2C$)
- Simile al pooling nelle CNN ma apprendibile
- Permette elaborazione gerarchica: feature più astratte a risoluzioni più basse

## 18. Elaborazione Completa attraverso i 4 Stage

<img src="../../../../../images/HTS-AT-swin.png" alt="Elaborazione attraverso i 4 Stage" style="display: block; margin-left: auto; margin-right: auto;">

Riassumiamo il flusso attraverso l'intera architettura con un esempio concreto.

**Configurazione**: `embed_dim=96`, `depths=[2,2,6,2]`, `num_heads=[4,8,16,32]`, `window_size=8`

### Stage 1: Feature di Basso Livello

**Input**: $(B, 4096, 96)$ con risoluzione $(64, 64)$

**Blocco 1** (W-MSA):
- Window partition: $64 \times 64 \rightarrow 64$ finestre $8 \times 8$
- WindowAttention con 4 heads
- MLP con espansione 4x
- Output: $(B, 4096, 96)$

**Blocco 2** (SW-MSA):
- Cyclic shift di $(-4, -4)$
- Window partition con maschera
- WindowAttention con 4 heads
- Reverse shift
- MLP
- Output: $(B, 4096, 96)$

**PatchMerging**:
- Output: $(B, 1024, 192)$ con risoluzione $(32, 32)$

**Intuizione Stage 1**: Cattura pattern locali e texture di base dello spettrogramma.

### Stage 2: Feature di Medio-Basso Livello

**Input**: $(B, 1024, 192)$ con risoluzione $(32, 32)$

**Blocco 1** (W-MSA):
- $32 \times 32 \rightarrow 16$ finestre $8 \times 8$
- WindowAttention con 8 heads
- Output: $(B, 1024, 192)$

**Blocco 2** (SW-MSA):
- Shift + attention + reverse
- WindowAttention con 8 heads
- Output: $(B, 1024, 192)$

**PatchMerging**:
- Output: $(B, 256, 384)$ con risoluzione $(16, 16)$

**Intuizione Stage 2**: Inizia a catturare relazioni temporali e pattern armonici.

### Stage 3: Feature di Alto Livello

**Input**: $(B, 256, 384)$ con risoluzione $(16, 16)$

**6 Blocchi alternati** (W-MSA, SW-MSA, W-MSA, SW-MSA, W-MSA, SW-MSA):
- $16 \times 16 \rightarrow 4$ finestre $8 \times 8$
- WindowAttention con 16 heads
- Output dopo 6 blocchi: $(B, 256, 384)$

**PatchMerging**:
- Output: $(B, 64, 768)$ con risoluzione $(8, 8)$

**Intuizione Stage 3**: Stage più profondo che cattura dipendenze semantiche complesse e pattern a lungo raggio.

### Stage 4: Feature Semantiche Globali

**Input**: $(B, 64, 768)$ con risoluzione $(8, 8)$

**Blocco 1** (W-MSA):
- $8 \times 8 \rightarrow 1$ finestra $8 \times 8$
- WindowAttention con 32 heads (global attention!)
- Output: $(B, 64, 768)$

**Blocco 2** (SW-MSA):
- Con risoluzione $8 \times 8$ e window_size $8$, lo shift non ha effetto pratico
- Output: $(B, 64, 768)$

**Nessun PatchMerging** (ultimo stage)

**Output finale dopo normalization**: $(B, 64, 768)$

**Intuizione Stage 4**: Con solo 1 finestra, questo stage fa effettivamente **global attention** su tutta la rappresentazione compressa, catturando il context semantico completo.

## 19. Classificazione: TSCAM (Token-Semantic Convolutional Attention Map)

Dopo i 4 stage, HTS-AT usa un approccio innovativo chiamato **TSCAM** per la classificazione.

### 19.1 Configurazione TSCAM

```python
if self.config.enable_tscam:
    SF = self.spec_size // (2 ** (len(self.depths) - 1)) // self.patch_stride[0] // self.freq_ratio
    # SF = 256 // 8 // 4 // 4 = 2
    
    self.tscam_conv = nn.Conv2d(
        in_channels=self.num_features,     # 768
        out_channels=self.num_classes,     # es. 527
        kernel_size=(SF, 3),               # (2, 3)
        padding=(0, 1)
    )
    self.head = nn.Linear(num_classes, num_classes)
```

### 19.2 Forward Pass con TSCAM

**Input**: $\mathbf{x} \in \mathbb{R}^{B \times 64 \times 768}$ dopo l'ultimo stage

**Step 1: Layer Normalization**:

```python
x = self.norm(x)  # LayerNorm(768)
B, N, C = x.shape  # (B, 64, 768)
```

**Step 2: Reshape a formato spaziale 2D**:

Ricordiamo che `frames_num` è il numero di frame temporali originali dello spettrogramma (prima del reshape_wav2img). Con configurazione tipica:

```python
SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
# frames_num / 8 / 4
ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
# frames_num / 8 / 4

x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
# (B, 768, SF, ST)
```

Con `frames_num = 1024` (dopo interpolazione):
- $SF = 1024 / 8 / 4 = 32$
- $ST = 1024 / 8 / 4 = 32$

Quindi: $\mathbf{x} \in \mathbb{R}^{B \times 768 \times 32 \times 32}$

**Nota**: Questo non corrisponde alla risoluzione $(8, 8)$ dell'ultimo stage! Il reshape utilizza `frames_num` (numero di frame temporali dello spettrogramma originale dopo interpolazione) per ricostruire la struttura spazio-temporale originale, non la risoluzione ridotta dei token dopo i stage.

**Step 3: Group 2D CNN (Frequency Grouping)**:

Il codice raggruppa le feature lungo la dimensione di frequenza usando `freq_ratio`:

```python
B, C, F, T = x.shape  # (B, 768, 32, 32)
c_freq_bin = F // self.freq_ratio  # 32 // 4 = 8

x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
# (B, 768, 4, 8, 32)

x = x.permute(0, 1, 3, 2, 4).contiguous()
# (B, 768, 8, 4, 32)

x = x.reshape(B, C, c_freq_bin, -1)
# (B, 768, 8, 128)
```

**Intuizione del grouping**:
- Raggruppa le 32 bande di frequenza in 8 "meta-bande" di 4 bande ciascuna
- Riorganizza la dimensione temporale: da $(4, 32)$ a $(128)$
- Risultato: $(B, 768, 8, 128)$ dove 8 è il numero di bande frequenziali aggregate e 128 è la dimensione temporale estesa

**Step 4: Latent Output (Embedding)**:

```python
latent_output = self.avgpool(torch.flatten(x, 2))
# flatten(x, 2): (B, 768, 8, 128) → (B, 768, 1024)
# avgpool: (B, 768, 1024) → (B, 768, 1)
latent_output = torch.flatten(latent_output, 1)
# (B, 768)
```

Questo è l'**embedding** che verrà restituito come rappresentazione densa dell'audio.

**Step 5: TSCAM Convolution**:

```python
x = self.tscam_conv(x)  # (B, 768, 8, 128) → (B, 527, 6, 128)
```

La convoluzione con kernel $(2, 3)$ e padding $(0, 1)$:

$
F_{out} = \frac{8 - 2 + 2 \cdot 0}{1} + 1 = 7 \quad \text{(ma viene troncato a 6)}
$

$
T_{out} = \frac{128 - 3 + 2 \cdot 1}{1} + 1 = 128
$

Output: $(B, 527, 6, 128)$ dove 527 è il numero di classi (AudioSet).

**Step 6: Flatten e Interpolazione Temporale**:

```python
x = torch.flatten(x, 2)  # (B, 527, 6, 128) → (B, 527, 768)

# Interpolazione temporale per frame-wise prediction
fpx = interpolate(torch.sigmoid(x).permute(0, 2, 1).contiguous(), 
                  8 * self.patch_stride[1])
# Interpola da 768 frame a 8 * 4 = 32 frame (o altro target)
# Output: (B, target_frames, 527)
```

**Intuizione dell'interpolazione**:
- L'output viene interpolato alla risoluzione temporale desiderata per le predizioni frame-wise
- `8 * self.patch_stride[1]` calcola il numero di frame target basato sulla configurazione
- Questo permette di avere predizioni temporali dettagliate per il rilevamento di eventi

**Step 7: Pooling Temporale per Clip-wise Output**:

```python
x = self.avgpool(x)  # (B, 527, 768) → (B, 527, 1)
x = torch.flatten(x, 1)  # (B, 527)
```

**Step 8: Output Finale**:

```python
output_dict = {
    'framewise_output': fpx,              # (B, target_frames, 527)
    'clipwise_output': torch.sigmoid(x),  # (B, 527)
    'latent_output': latent_output        # (B, 768)
}
```

**Intuizione complessiva del TSCAM**:
- **Token**: rappresentazioni locali tempo-frequenza (dai Swin Transformer blocks)
- **Semantic**: la convoluzione aggrega semanticamente le informazioni
- **Convolutional Attention Map**: crea mappe di attenzione per ogni classe
- Il design permette sia predizioni clip-wise (intero audio) che frame-wise (temporalmente risolte)
- Questo è essenziale per task di sound event detection dove serve sapere *quando* avviene un evento

### 19.3 Alternative: Classificazione Senza TSCAM

Se `enable_tscam=False`:

```python
x = self.norm(x)  # (B, 64, 768)
x = self.avgpool(x.transpose(1, 2))  # (B, 768, 1)
x = torch.flatten(x, 1)  # (B, 768)

if self.num_classes > 0:
    x = self.head(x)  # Linear(768 → 527)
    
output_dict = {
    'clipwise_output': torch.sigmoid(x)  # (B, 527)
}
```

Questo è un approccio più semplice: global average pooling + layer fully connected.

## 20. Gestione di Audio di Lunghezza Variabile

HTS-AT implementa diverse strategie per gestire audio di lunghezza variabile, specialmente durante l'inferenza.

### 20.1 Modalità Repeat

Con `config.enable_repeat_mode=True`:

**Training**: Selezione casuale di una posizione e ripetizione:

```python
cur_pos = random.randint(0, (self.freq_ratio - 1) * self.spec_size - 1)
x = self.repeat_wat2img(x, cur_pos)
```

**Inference**: Elaborazione multipla e media:

```python
output_dicts = []
for cur_pos in range(0, (self.freq_ratio - 1) * self.spec_size + 1, self.spec_size):
    tx = x.clone()
    tx = self.repeat_wat2img(tx, cur_pos)
    output_dicts.append(self.forward_features(tx))

# Media delle predizioni
clipwise_output = sum(d["clipwise_output"] for d in output_dicts) / len(output_dicts)
framewise_output = sum(d["framewise_output"] for d in output_dicts) / len(output_dicts)
```

**Intuizione**: Elabora l'audio da diverse "prospettive temporali" e fa la media per robustezza.

### 20.2 Modalità Crop (Default)

Per audio più lunghi del target:

**Training**: Crop casuale:

```python
if x.shape[2] > self.freq_ratio * self.spec_size:
    x = self.crop_wav(x, crop_size=self.freq_ratio * self.spec_size)
```

**Inference**: Crop sovrapposti con media:

```python
overlap_size = (x.shape[2] - 1) // 4
crop_size = (x.shape[2] - 1) // 2

for cur_pos in range(0, x.shape[2] - crop_size - 1, overlap_size):
    tx = self.crop_wav(x, crop_size=crop_size, spe_pos=cur_pos)
    tx = self.reshape_wav2img(tx)
    output_dicts.append(self.forward_features(tx))

# Media delle predizioni
clipwise_output = sum(...) / len(output_dicts)
framewise_output = sum(...) / len(output_dicts)
```

**Intuizione**: Per audio lunghi, elabora segmenti sovrapposti e fa ensemble delle predizioni.

### 20.3 Modalità Infer

Con `infer_mode=True`:

```python
frame_num = x.shape[2]
target_T = int(self.spec_size * self.freq_ratio)
repeat_ratio = math.floor(target_T / frame_num)
x = x.repeat(repeats=(1, 1, repeat_ratio, 1))
x = self.reshape_wav2img(x)
output_dict = self.forward_features(x)
```

**Intuizione**: Per audio corti, ripete i frame per raggiungere la lunghezza target.

## 21. Architettura Completa: Visualizzazione del Flusso

```
Input Audio (B, N_samples)
    ↓
STFT → Power Spectrogram (B, 1, T_frames, 513)
    ↓
Mel Filterbank → Log-Mel (B, 1, T_frames, 64)
    ↓
BatchNorm2d(64) [con transpose]
    ↓
SpecAugmentation + Mixup [training only]
    ↓
Reshape to Image (B, 1, 256, 256)
    ↓
╔══════════════════════════════════════════════════╗
║             PATCH EMBEDDING                      ║
║  Conv2d(1→96, kernel=4, stride=4)               ║
║  Output: (B, 4096, 96)  [64×64 patches]         ║
╚══════════════════════════════════════════════════╝
    ↓
Position Embedding [optional] + Dropout
    ↓
╔══════════════════════════════════════════════════╗
║              STAGE 1: (64×64, 96)                ║
║  ┌────────────────────────────────────────┐     ║
║  │ SwinBlock 1 (W-MSA, 4 heads)           │     ║
║  │   • Window Partition (8×8 windows)     │     ║
║  │   • WindowAttention + Rel. Pos. Bias   │     ║
║  │   • MLP (96 → 384 → 96)                │     ║
║  └────────────────────────────────────────┘     ║
║  ┌────────────────────────────────────────┐     ║
║  │ SwinBlock 2 (SW-MSA, 4 heads)          │     ║
║  │   • Cyclic Shift (-4, -4)              │     ║
║  │   • Masked WindowAttention             │     ║
║  │   • Reverse Shift                      │     ║
║  │   • MLP (96 → 384 → 96)                │     ║
║  └────────────────────────────────────────┘     ║
║  PatchMerging: (B, 4096, 96) → (B, 1024, 192)   ║
╚══════════════════════════════════════════════════╝
    ↓
╔══════════════════════════════════════════════════╗
║              STAGE 2: (32×32, 192)               ║
║  2× SwinBlocks (W-MSA, SW-MSA) 8 heads          ║
║  PatchMerging: (B, 1024, 192) → (B, 256, 384)   ║
╚══════════════════════════════════════════════════╝
    ↓
╔══════════════════════════════════════════════════╗
║              STAGE 3: (16×16, 384)               ║
║  6× SwinBlocks alternati, 16 heads              ║
║  PatchMerging: (B, 256, 384) → (B, 64, 768)     ║
╚══════════════════════════════════════════════════╝
    ↓
╔══════════════════════════════════════════════════╗
║              STAGE 4: (8×8, 768)                 ║
║  2× SwinBlocks (global attention) 32 heads      ║
║  No PatchMerging                                 ║
║  Output: (B, 64, 768)                            ║
╚══════════════════════════════════════════════════╝
    ↓
LayerNorm(768)
    ↓
╔══════════════════════════════════════════════════╗
║              TSCAM HEAD                          ║
║  Reshape → (B, 768, 8, 128)                     ║
║  Conv2d(768→527, kernel=(2,3))                  ║
║  ├─→ AvgPool → Clipwise (B, 527)                ║
║  └─→ Interpolate → Framewise (B, T, 527)       ║
╚══════════════════════════════════════════════════╝
    ↓
Output: {
  'clipwise_output': (B, 527),    # predizioni per clip
  'framewise_output': (B, T, 527), # predizioni temporali
  'latent_output': (B, 768)        # embedding
}
```

## 22. Differenze Chiave tra HTS-AT e CNN14

| Aspetto | CNN14 | HTS-AT |
|---------|-------|---------|
| **Architettura base** | Convolutional Neural Network | Vision Transformer (Swin) |
| **Receptive field** | Locale, aumenta gradualmente | Global tramite self-attention |
| **Inductive bias** | Forte (località, traslazione) | Debole (appreso dai dati) |
| **Complessità** | $O(N)$ | $O(N)$ grazie a windowed attention |
| **Parametri** | ~80M | Variabile (tipicamente 30-90M) |
| **Feature extraction** | Convoluzione gerarchica | Self-attention gerarchica |
| **Position encoding** | Implicito (convoluzione) | Explicit (relative position bias) |
| **Long-range dependencies** | Limitato | Eccellente |
| **Output** | Solo clip-wise | Clip-wise + frame-wise |

## 23. Punti di Forza di HTS-AT

### 23.1 Attention Mechanism

**Vantaggio**: Cattura dipendenze a lungo raggio nello spettrogramma.

**Esempio pratico**: In musica, può collegare note distanti che fanno parte della stessa melodia. Nelle registrazioni ambientali, può associare eventi sonori correlati ma temporalmente separati.

### 23.2 Hierarchical Processing

**Vantaggio**: Elaborazione multi-scala naturale attraverso i 4 stage.

- Stage 1-2: Texture e pattern locali
- Stage 3: Strutture temporali complesse
- Stage 4: Context semantico globale

### 23.3 TSCAM per Frame-wise Prediction

**Vantaggio**: Output temporalmente risolto per sound event detection.

**Utilizzo**: Non solo "questo audio contiene un cane che abbaia", ma "il cane abbaia dal secondo 2.5 al secondo 4.8".

### 23.4 Relative Position Bias

**Vantaggio**: Encoding posizionale più flessibile rispetto agli absolute position embedding.

**Beneficio**: Il modello impara preferenze per distanze relative (es. "guarda i token a distanza 3") piuttosto che posizioni assolute.

### 23.5 Efficient Windowed Attention

**Vantaggio**: Complessità lineare rispetto alla lunghezza della sequenza.

- Global self-attention: $O(N^2)$ dove $N$ è il numero di token
- Windowed attention: $O(N \cdot M)$ dove $M$ è la dimensione della finestra (costante)

Con $N = 4096$ e $M = 64$, il risparmio computazionale è enorme.

## 24. Considerazioni Pratiche

### 24.1 Requisiti Computazionali

- **Memoria GPU**: Richiede più memoria di CNN14 per il training (attention matrices)
- **Velocità di inferenza**: Comparabile a CNN14 per audio singoli, più lenta per batch grandi
- **Training time**: Generalmente più lungo per convergenza

### 24.2 Transfer Learning

HTS-AT può utilizzare **checkpoint pre-trained** da Swin Transformer per computer vision:

```python
# I pesi dei Swin Transformer blocks possono essere inizializzati
# da modelli pre-trained su ImageNet
```

Questo accelera significativamente il training per task audio-specifici.

### 24.3 Data Augmentation

La combinazione di SpecAugmentation e Mixup è cruciale:

- **SpecAugmentation**: Previene overfitting su specifiche regioni tempo-frequenza
- **Mixup**: Migliora generalizzazione e calibrazione delle probabilità

### 24.4 Hyperparameter Tuning

Parametri critici:
- `window_size`: Trade-off tra local vs global attention (tipicamente 8)
- `depths`: Profondità di ogni stage (tipicamente [2,2,6,2])
- `embed_dim`: Dimensione base degli embedding (tipicamente 96)
- `num_heads`: Numero di attention heads per stage (tipicamente [4,8,16,32])

## 25. Conclusioni

HTS-AT rappresenta un'evoluzione significativa nell'elaborazione audio tramite deep learning:

1. **Self-attention gerarchica** permette di catturare dipendenze complesse a diverse scale temporali
2. **Windowed attention** mantiene l'efficienza computazionale
3. **TSCAM** fornisce output temporalmente risolti essenziali per event detection
4. **Relative position bias** offre encoding posizionale flessibile
5. **Architettura modulare** facilita transfer learning e customizzazione

L'architettura è particolarmente efficace per:
- Sound event detection (localizzazione temporale)
- Audio tagging multi-label
- Music information retrieval
- Environmental sound classification

La complessità aggiuntiva rispetto a CNN14 è giustificata quando il task richiede:
- Comprensione di context a lungo raggio
- Localizzazione temporale precisa degli eventi
- Handling di relazioni complesse nello spettrogramma

**Trade-off finale**: HTS-AT offre maggiore espressività e flessibilità a costo di maggiore complessità computazionale e necessità di più dati per il training.
