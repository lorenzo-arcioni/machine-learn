# Drop Path (Stochastic Depth)

## Introduzione

**Drop Path**, noto anche come **Stochastic Depth**, è una tecnica di regolarizzazione utilizzata nelle reti neurali profonde, particolarmente efficace nei modelli basati su architetture residuali come ResNet, Vision Transformer (ViT) e Swin Transformer.

A differenza del classico **Dropout** che disattiva casualmente singoli neuroni durante il training, Drop Path disattiva interi **percorsi residuali** (residual paths) o **layer interi**, rendendo la rete più robusta e migliorando la generalizzazione.

## Motivazione

### Il Problema delle Reti Profonde

Nelle architetture molto profonde con connessioni residuali, ogni layer contribuisce incrementalmente alla trasformazione dell'input. Tuttavia, durante il training si possono verificare alcuni problemi:

1. **Co-adaptation**: I layer possono diventare eccessivamente dipendenti l'uno dall'altro
2. **Redundancy**: Alcuni layer potrebbero imparare trasformazioni ridondanti
3. **Overfitting**: La capacità totale della rete può portare a memorizzazione invece che generalizzazione
4. **Gradient flow**: In reti molto profonde, alcuni layer potrebbero ricevere gradienti molto piccoli

### La Soluzione: Stochastic Depth

Drop Path affronta questi problemi rendendo la rete **stocasticamente più shallow** durante il training, forzando ogni layer a imparare rappresentazioni utili anche quando alcuni layer precedenti o successivi sono assenti.

## Formulazione Matematica

### Architettura Residuale Base

Consideriamo un blocco residuale standard:

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathbf{W}_l)
$$

dove:
- $\mathbf{x}_l$ è l'input al layer $l$
- $\mathcal{F}(\cdot)$ è una trasformazione (es. convoluzione, attention, MLP)
- $\mathbf{W}_l$ sono i parametri apprendibili del layer $l$
- $\mathbf{x}_{l+1}$ è l'output che viene passato al layer successivo

### Drop Path Applicato

Con Drop Path, la formula diventa:

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathbf{b}_l \cdot \mathcal{F}(\mathbf{x}_l, \mathbf{W}_l)
$$

dove $\mathbf{b}_l$ è una **variabile di Bernoulli**:

$$
\mathbf{b}_l \sim \text{Bernoulli}(p_l)
$$

con:
- $p_l$: probabilità di **mantenere** (keep probability) il path al layer $l$
- $1 - p_l$: probabilità di **droppare** il path (drop probability)

**Comportamento:**
- Se $\mathbf{b}_l = 1$: il path è attivo, la trasformazione viene applicata
- Se $\mathbf{b}_l = 0$: il path è droppato, passa solo l'identità $\mathbf{x}_l$

### Scaling During Training

Durante il training, per mantenere l'aspettativa del valore costante, spesso si applica uno **scaling factor**:

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \frac{\mathbf{b}_l}{p_l} \cdot \mathcal{F}(\mathbf{x}_l, \mathbf{W}_l)
$$

Questo garantisce che:

$$
\mathbb{E}[\mathbf{x}_{l+1}] = \mathbf{x}_l + \mathbb{E}\left[\frac{\mathbf{b}_l}{p_l}\right] \cdot \mathcal{F}(\mathbf{x}_l, \mathbf{W}_l) = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathbf{W}_l)
$$

poiché:

$$
\mathbb{E}\left[\frac{\mathbf{b}_l}{p_l}\right] = \frac{1}{p_l} \cdot p_l = 1
$$

### Inference (Test Time)

Durante l'inferenza, **non si applica drop path**. La rete usa tutti i layer deterministicamente:

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathbf{W}_l)
$$

Lo scaling applicato durante il training assicura che i valori attesi siano consistenti tra training e inference.

## Linear Decay Schedule

Una strategia comune è variare la drop probability in funzione della profondità del layer, usando un **linear decay schedule**:

$$
p_l = 1 - \frac{l}{L} \cdot \text{drop\_rate}
$$

dove:
- $l \in \{0, 1, \ldots, L-1\}$ è l'indice del layer
- $L$ è il numero totale di layer
- $\text{drop\_rate} \in [0, 1]$ è il tasso massimo di drop (per l'ultimo layer)

**Intuizione:**
- **Layer iniziali** (vicini all'input): drop probability bassa → quasi sempre attivi
- **Layer finali** (vicini all'output): drop probability alta → più frequentemente droppati

Questo riflette l'idea che i layer iniziali estraggono feature fondamentali, mentre i layer finali affinano rappresentazioni più specifiche.

### Esempio Numerico

Supponiamo:
- $L = 12$ layer totali
- $\text{drop\_rate} = 0.3$

Le probabilità di drop per ciascun layer saranno:

| Layer $l$ | Drop Probability $1 - p_l$ | Keep Probability $p_l$ |
|-----------|---------------------------|------------------------|
| 0         | $0.0 \cdot 0.3 = 0.000$   | $1.000$               |
| 1         | $\frac{1}{12} \cdot 0.3 = 0.025$ | $0.975$        |
| 2         | $\frac{2}{12} \cdot 0.3 = 0.050$ | $0.950$        |
| ...       | ...                        | ...                   |
| 11        | $\frac{11}{12} \cdot 0.3 = 0.275$ | $0.725$       |

## Implementazione in PyTorch

```python
import torch
import torch.nn as nn

class DropPath(nn.Module):
    """
    Drop Path (Stochastic Depth) per blocchi residuali.
    
    Args:
        drop_prob (float): Probabilità di droppare il path (0.0 = no drop)
        scale_by_keep (bool): Se True, scala per 1/keep_prob durante training
    """
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        # Durante inference o se drop_prob = 0, ritorna input invariato
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        # Calcola keep probability
        keep_prob = 1 - self.drop_prob
        
        # Crea maschera di Bernoulli con shape broadcast-compatibile
        # Shape: (batch_size, 1, 1, ..., 1) per broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Bernoulli: 0 o 1
        
        # Applica drop path con scaling
        if self.scale_by_keep:
            output = x.div(keep_prob) * random_tensor
        else:
            output = x * random_tensor
        
        return output
    
    def extra_repr(self):
        return f'drop_prob={self.drop_prob}'


def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    """
    Versione funzionale di Drop Path.
    """
    if drop_prob == 0.0 or not training:
        return x
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    
    if scale_by_keep:
        output = x.div(keep_prob) * random_tensor
    else:
        output = x * random_tensor
    
    return output


# Esempio di utilizzo in un blocco residuale
class ResidualBlock(nn.Module):
    def __init__(self, dim, drop_path_prob=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.transform = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path_prob)
    
    def forward(self, x):
        # Connessione residuale con drop path
        return x + self.drop_path(self.transform(self.norm(x)))


# Costruzione di una rete con linear decay schedule
class DeepNetwork(nn.Module):
    def __init__(self, dim=512, depth=12, drop_path_rate=0.3):
        super().__init__()
        
        # Linear decay: crescente dalla prima all'ultima layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            ResidualBlock(dim, drop_path_prob=dpr[i])
            for i in range(depth)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
```

## Differenze con Dropout Classico

| Aspetto | Dropout | Drop Path |
|---------|---------|-----------|
| **Granularità** | Singoli neuroni/canali | Interi layer/blocchi |
| **Applicazione** | All'interno di un layer | Sulla residual connection |
| **Posizione** | Dopo attivazioni | Sul path residuale |
| **Effetto sulla profondità** | Nessuno | Riduce la profondità effettiva |
| **Uso tipico** | MLP, FC layers | Architetture residuali profonde |

## Vantaggi di Drop Path

1. **Regolarizzazione efficace**: Riduce l'overfitting in reti molto profonde
2. **Training più veloce**: La rete si allena più velocemente poiché in media è meno profonda
3. **Ensemble implicito**: Durante il training si allenano esponenzialmente molte sotto-reti
4. **Migliore generalizzazione**: Forza ogni layer a contribuire in modo significativo
5. **Riduzione della co-adaptation**: I layer non possono dipendere rigidamente l'uno dall'altro

## Analisi Teorica

### Numero di Reti Possibili

Con $L$ layer e drop path, il numero di configurazioni possibili durante il training è:

$$
2^L
$$

Ad esempio, con $L = 24$ layer (come in ViT-Base), otteniamo $2^{24} \approx 16.7$ milioni di sotto-reti diverse!

### Profondità Attesa

La profondità attesa della rete durante il training è:

$$
\mathbb{E}[\text{depth}] = \sum_{l=1}^{L} p_l
$$

Con linear decay schedule:

$$
\mathbb{E}[\text{depth}] = \sum_{l=1}^{L} \left(1 - \frac{l-1}{L-1} \cdot \text{drop\_rate}\right) = L \cdot \left(1 - \frac{\text{drop\_rate}}{2}\right)
$$

**Esempio**: Con $L = 24$ e $\text{drop\_rate} = 0.3$:

$$
\mathbb{E}[\text{depth}] = 24 \cdot (1 - 0.15) = 20.4 \text{ layer}
$$

In media, la rete si comporta come se avesse circa 20-21 layer invece di 24.

## Applicazioni Pratiche

### Vision Transformer (ViT)

Nei Vision Transformer, Drop Path viene applicato dopo il blocco di attention e dopo l'MLP:

$$
\begin{align}
\mathbf{x}' &= \mathbf{x} + \text{DropPath}(\text{Attention}(\text{LN}(\mathbf{x}))) \\
\mathbf{x}'' &= \mathbf{x}' + \text{DropPath}(\text{MLP}(\text{LN}(\mathbf{x}')))
\end{align}
$$

### Swin Transformer

Nel Swin Transformer, Drop Path viene applicato dopo ogni sotto-blocco:

$$
\begin{align}
\mathbf{x}_{l,1} &= \mathbf{x}_l + \text{DropPath}(\text{W-MSA}(\text{LN}(\mathbf{x}_l))) \\
\mathbf{x}_{l,2} &= \mathbf{x}_{l,1} + \text{DropPath}(\text{MLP}(\text{LN}(\mathbf{x}_{l,1}))) \\
\mathbf{x}_{l+1,1} &= \mathbf{x}_{l,2} + \text{DropPath}(\text{SW-MSA}(\text{LN}(\mathbf{x}_{l,2}))) \\
\mathbf{x}_{l+1} &= \mathbf{x}_{l+1,1} + \text{DropPath}(\text{MLP}(\text{LN}(\mathbf{x}_{l+1,1})))
\end{align}
$$

### ConvNeXt e Architetture CNN

Drop Path è altrettanto efficace nelle CNN moderne:

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \text{DropPath}(\text{Conv}(\text{x}_l))
$$

## Iperparametri e Tuning

### Scelta del Drop Rate

Valori tipici per `drop_path_rate`:
- **Piccoli modelli** (< 12 layer): $0.1 - 0.2$
- **Modelli medi** (12-24 layer): $0.2 - 0.3$
- **Modelli grandi** (> 24 layer): $0.3 - 0.5$

### Strategie Alternative

Oltre al linear decay, esistono altre strategie:

**1. Uniform Drop Rate:**
$$
p_l = 1 - \text{drop\_rate} \quad \forall l
$$

**2. Quadratic Decay:**
$$
p_l = 1 - \left(\frac{l}{L}\right)^2 \cdot \text{drop\_rate}
$$

**3. Exponential Decay:**
$$
p_l = 1 - \left(1 - e^{-\lambda l}\right) \cdot \text{drop\_rate}
$$

## Risultati Empirici

Studi hanno dimostrato che Drop Path:
- **Migliora l'accuratezza** del 1-2% su ImageNet per modelli profondi
- **Riduce l'overfitting** significativamente con dataset piccoli
- **Accelera la convergenza** del 10-20% in termini di epoche necessarie
- **Scala bene** con la profondità del modello

## Conclusioni

Drop Path è una tecnica di regolarizzazione fondamentale per architetture residuali profonde moderne. La sua capacità di:
- Allenare ensemble impliciti di sotto-reti
- Ridurre la profondità effettiva durante il training
- Forzare ogni layer a contribuire significativamente

...la rende uno strumento essenziale nel toolkit del deep learning moderno, specialmente per Vision Transformer e architetture CNN profonde.

## Riferimenti

1. **Huang et al. (2016)**: "Deep Networks with Stochastic Depth" - Paper originale
2. **Dosovitskiy et al. (2021)**: "An Image is Worth 16x16 Words" - ViT paper
3. **Liu et al. (2021)**: "Swin Transformer" - Uso in Swin
4. **Touvron et al. (2021)**: "Training data-efficient image transformers" - DeiT paper
