# Residual Connections (Connessioni Residue)

## Introduzione

Le **residual connections** (o **skip connections**) sono uno dei concetti più influenti del deep learning moderno. Introdotte nel paper seminale **ResNet** (He et al., 2015), hanno rivoluzionato la capacità di addestrare reti neurali estremamente profonde, risolvendo il problema della **degradazione** che affliggeva le architetture precedenti.

## Formulazione Base

### Schema Generale

In un blocco con residual connection, l'output è definito come:

$$
\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x}, \{\mathbf{W}_i\})
$$

dove:
- $\mathbf{x}$ è l'input al blocco
- $\mathcal{F}(\mathbf{x}, \{\mathbf{W}_i\})$ è una trasformazione parametrica (possibilmente complessa)
- $\{\mathbf{W}_i\}$ sono i parametri apprendibili
- $\mathbf{y}$ è l'output del blocco

**Componenti:**
1. **Identity path** (percorso identità): $\mathbf{x}$ passa inalterato
2. **Residual path** (percorso residuo): $\mathcal{F}(\mathbf{x})$ rappresenta la trasformazione appresa
3. **Addition** (somma): combinazione elemento per elemento dei due percorsi

### Nel Contesto dei Transformer

Nei Transformer (ViT, Swin Transformer, BERT, ecc.), lo schema diventa:

$$
\mathbf{X}_{\text{out}} = \mathbf{X} + \text{DropPath}(\mathcal{F}(\text{LayerNorm}(\mathbf{X})))
$$

dove tipicamente $\mathcal{F}$ rappresenta:
- **Self-Attention** (nel blocco di attenzione)
- **Feed-Forward Network/MLP** (nel blocco MLP)

## Contesto: Blocco di Attenzione nel Swin Transformer

### Schema Completo del Blocco

Nel Swin Transformer, un blocco di attenzione completo segue questa struttura:

$$
\begin{align}
\mathbf{X}_{\text{norm}} &= \text{LayerNorm}(\mathbf{X}) \\
\mathbf{Q}, \mathbf{K}, \mathbf{V} &= \mathbf{X}_{\text{norm}} \mathbf{W}_{qkv} \\
\mathbf{A} &= \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{B}\right) \\
\mathbf{O} &= \mathbf{A}\mathbf{V} \\
\mathbf{O}_{\text{proj}} &= \mathbf{O}\mathbf{W}_{\text{proj}} \\
\mathbf{X}_{\text{attn}} &= \mathbf{X} + \text{DropPath}(\mathbf{O}_{\text{proj}})
\end{align}
$$

L'ultima equazione è la **residual connection**.

### Dimensionalità

Per concretezza, consideriamo lo Stage 1 del Swin Transformer:

$$
\begin{align}
\mathbf{X} &\in \mathbb{R}^{B \times 3136 \times 96} \quad \text{(input)} \\
\mathbf{O}_{\text{proj}} &\in \mathbb{R}^{B \times 3136 \times 96} \quad \text{(output attenzione)} \\
\mathbf{X}_{\text{attn}} &\in \mathbb{R}^{B \times 3136 \times 96} \quad \text{(output blocco)}
\end{align}
$$

dove:
- $B$ = batch size
- $3136 = 56 \times 56$ = numero di patch
- $96$ = dimensione dei canali

**Requisito fondamentale:** $\mathbf{X}$ e $\mathbf{O}_{\text{proj}}$ devono avere **esattamente le stesse dimensioni** per poter essere sommati elemento per elemento.

## Significato della Somma

### Interpretazione Algebrica

La somma:

$$
\mathbf{X}_{\text{attn}} = \mathbf{X} + \mathbf{O}_{\text{proj}}
$$

è una **somma elemento per elemento** (element-wise addition):

$$
[\mathbf{X}_{\text{attn}}]_{b,i,c} = [\mathbf{X}]_{b,i,c} + [\mathbf{O}_{\text{proj}}]_{b,i,c}
$$

per ogni:
- $b \in \{1, \ldots, B\}$ (esempio nel batch)
- $i \in \{1, \ldots, 3136\}$ (patch)
- $c \in \{1, \ldots, 96\}$ (canale)

### Interpretazione Concettuale

Questa non è una somma casuale, ma ha un profondo significato:

$$
\underbrace{\mathbf{X}_{\text{attn}}}_{\text{Nuova rappresentazione}} = \underbrace{\mathbf{X}}_{\text{Informazione originale}} + \underbrace{\mathbf{O}_{\text{proj}}}_{\text{Informazione contestuale}}
$$

**In parole:**
- $\mathbf{X}$: ciò che il modello già conosce (feature originali dei patch)
- $\mathbf{O}_{\text{proj}}$: ciò che il modello apprende attraverso l'attenzione (interazioni tra patch)
- $\mathbf{X}_{\text{attn}}$: feature arricchite che combinano conoscenza locale e contestuale

## Intuizione Visiva

### Esempio Concreto

Consideriamo un singolo patch $i$ in un'immagine:

**Prima dell'attenzione:**
$$
\mathbf{x}_i \in \mathbb{R}^{96} \quad \text{(feature locali del patch)}
$$

Questo vettore contiene informazioni su:
- Colore locale
- Texture
- Bordi
- Pattern elementari

**Dopo l'attenzione:**
$$
\mathbf{o}_{i,\text{proj}} \in \mathbb{R}^{96} \quad \text{(informazioni contestuali)}
$$

Questo vettore contiene informazioni su:
- Come il patch $i$ si relaziona con altri patch
- Quali patch sono semanticamente simili
- Relazioni spaziali globali
- Pattern di alto livello

**Dopo la residual connection:**
$$
\mathbf{x}_{i,\text{attn}} = \mathbf{x}_i + \mathbf{o}_{i,\text{proj}}
$$

Il patch ora contiene:
- **Informazioni locali** (da $\mathbf{x}_i$): "Io sono un pixel rosso con un bordo"
- **Informazioni contestuali** (da $\mathbf{o}_{i,\text{proj}}$): "Faccio parte di un oggetto più grande, circondato da altri patch simili"

### Analogia

Pensa alla residual connection come a:

> **Base knowledge + New insights = Enhanced understanding**

È come leggere un libro:
- $\mathbf{X}$: ciò che già sapevi prima di leggere
- $\mathbf{O}_{\text{proj}}$: nuove informazioni dal libro
- $\mathbf{X}_{\text{attn}}$: la tua conoscenza arricchita dopo la lettura

**Importante:** Non stai *sostituendo* la vecchia conoscenza, ma la stai *arricchendo*.

## Motivazione Profonda: Stabilità del Training

### Il Problema delle Reti Profonde

Prima delle residual connections, addestrare reti molto profonde era problematico:

**Degradation Problem:**
- Aggiungere più layer **dovrebbe** migliorare le prestazioni (più capacità di apprendimento)
- Nella pratica, oltre una certa profondità, le prestazioni **peggioravano**
- Questo non era dovuto a overfitting, ma a difficoltà di ottimizzazione

**Vanishing/Exploding Gradients:**

Consideriamo una rete profonda senza residual connections:

$$
\mathbf{y}_L = f_L(f_{L-1}(\ldots f_2(f_1(\mathbf{x})) \ldots))
$$

Il gradiente rispetto all'input è:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}_L} \prod_{l=1}^{L} \frac{\partial f_l}{\partial f_{l-1}}
$$

Questa **catena di prodotti** può causare:
- **Vanishing gradients:** se $\|\frac{\partial f_l}{\partial f_{l-1}}\| < 1$ → i gradienti tendono a zero
- **Exploding gradients:** se $\|\frac{\partial f_l}{\partial f_{l-1}}\| > 1$ → i gradienti esplodono

**Conseguenza:** I layer iniziali non imparano, o imparano molto lentamente.

### La Soluzione: Residual Connections

Con residual connections, la propagazione in avanti diventa:

$$
\mathbf{y}_l = \mathbf{y}_{l-1} + \mathcal{F}_l(\mathbf{y}_{l-1})
$$

Espandendo ricorsivamente:

$$
\mathbf{y}_L = \mathbf{y}_0 + \sum_{l=1}^{L} \mathcal{F}_l(\mathbf{y}_{l-1})
$$

### Analisi del Gradiente

Calcoliamo il gradiente rispetto a un layer intermedio $\mathbf{y}_l$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{y}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}_L} \frac{\partial \mathbf{y}_L}{\partial \mathbf{y}_l}
$$

Grazie alla struttura residuale:

$$
\mathbf{y}_L = \mathbf{y}_l + \sum_{i=l+1}^{L} \mathcal{F}_i(\mathbf{y}_{i-1})
$$

Quindi:

$$
\frac{\partial \mathbf{y}_L}{\partial \mathbf{y}_l} = \mathbf{I} + \frac{\partial}{\partial \mathbf{y}_l} \left(\sum_{i=l+1}^{L} \mathcal{F}_i(\mathbf{y}_{i-1})\right)
$$

dove $\mathbf{I}$ è la **matrice identità**.

**Risultato finale:**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{y}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}_L} \left(\mathbf{I} + \frac{\partial}{\partial \mathbf{y}_l} \left(\sum_{i=l+1}^{L} \mathcal{F}_i\right)\right)
$$

### Perché Questo è Cruciale

**Il termine $\mathbf{I}$ (identità) garantisce:**

1. **Flusso diretto del gradiente:** Anche se tutti i $\mathcal{F}_i$ hanno gradienti che vaniscono, il gradiente fluisce comunque attraverso l'identità
2. **Niente vanishing gradients:** Il gradiente non può mai andare completamente a zero
3. **Shortcut verso layer profondi:** L'informazione può "saltare" layer intermedi problematici

### Confronto Matematico

**Senza residual connection:**
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{y}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}_L} \prod_{l=1}^{L} \frac{\partial \mathcal{F}_l}{\partial \mathbf{y}_{l-1}}
$$

Se un solo $\frac{\partial \mathcal{F}_l}{\partial \mathbf{y}_{l-1}}$ è piccolo → il gradiente svanisce.

**Con residual connection:**
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{y}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}_L} \left(\mathbf{I} + \text{termini aggiuntivi}\right)
$$

Anche se i termini aggiuntivi vaniscono, $\mathbf{I}$ garantisce un gradiente minimo.

## Apprendimento di Funzioni Residue

### Intuizione della Residual Function

Le residual connections cambiano cosa il modello deve imparare:

**Senza residual connection:**
$$
\mathcal{F}(\mathbf{x}) \approx \mathbf{y}_{\text{target}}
$$

Il modello deve imparare la **trasformazione completa** da input a output.

**Con residual connection:**
$$
\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})
$$
$$
\mathcal{F}(\mathbf{x}) \approx \mathbf{y}_{\text{target}} - \mathbf{x}
$$

Il modello deve imparare solo la **differenza** (residuo) tra input e output.

### Perché Questo è Più Facile

**Scenario 1 - Mappatura Identità:**

Se l'output ideale è $\mathbf{y} \approx \mathbf{x}$ (nessuna trasformazione necessaria):

- **Senza residual:** $\mathcal{F}$ deve imparare $\mathcal{F}(\mathbf{x}) = \mathbf{x}$ (difficile!)
- **Con residual:** $\mathcal{F}$ deve solo imparare $\mathcal{F}(\mathbf{x}) = \mathbf{0}$ (facile! Basta settare i pesi a zero)

**Scenario 2 - Piccole Correzioni:**

Se l'output ideale è $\mathbf{y} \approx \mathbf{x} + \epsilon$ (piccola modifica):

- **Senza residual:** $\mathcal{F}$ deve rappresentare l'intera funzione complessa
- **Con residual:** $\mathcal{F}$ deve solo imparare il piccolo residuo $\epsilon$

### Esempio Numerico

Supponiamo che un layer debba trasformare:

$$
\mathbf{x} = [1.0, 2.0, 3.0] \rightarrow \mathbf{y}_{\text{target}} = [1.1, 2.05, 3.2]
$$

**Senza residual connection:**
$$
\mathcal{F}(\mathbf{x}) = [1.1, 2.05, 3.2]
$$

**Con residual connection:**
$$
\mathcal{F}(\mathbf{x}) = \mathbf{y}_{\text{target}} - \mathbf{x} = [0.1, 0.05, 0.2]
$$

È molto più facile apprendere piccole correzioni che funzioni complete!

## Compatibilità Dimensionale

### Requisito Fondamentale

Per poter sommare $\mathbf{x}$ e $\mathcal{F}(\mathbf{x})$, devono avere **esattamente le stesse dimensioni**:

$$
\mathbf{x} \in \mathbb{R}^{d} \quad \Rightarrow \quad \mathcal{F}(\mathbf{x}) \in \mathbb{R}^{d}
$$

### Proiezione Finale nel Meccanismo di Attenzione

Nel Swin Transformer, dopo la multi-head attention:

1. **Dopo concatenazione heads:**
   $$
   \mathbf{O} \in \mathbb{R}^{B \times N \times C}
   $$

2. **Proiezione finale:**
   $$
   \mathbf{O}_{\text{proj}} = \mathbf{O} \mathbf{W}_{\text{proj}}
   $$
   dove $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{C \times C}$

3. **Risultato:**
   $$
   \mathbf{O}_{\text{proj}} \in \mathbb{R}^{B \times N \times C}
   $$

**Scopo di $\mathbf{W}_{\text{proj}}$:**
- Miscelare informazioni delle diverse head
- **Garantire compatibilità dimensionale** con $\mathbf{X}$ per la residual connection

### Quando le Dimensioni Cambiano

In alcuni casi (es. downsampling tra stage), le dimensioni cambiano:

$$
\mathbf{x} \in \mathbb{R}^{H \times W \times C_{\text{in}}} \quad \rightarrow \quad \mathbf{y} \in \mathbb{R}^{H' \times W' \times C_{\text{out}}}
$$

dove $H' < H$, $W' < W$, $C_{\text{out}} > C_{\text{in}}$.

**Soluzione - Projection Shortcut:**

$$
\mathbf{y} = \mathbf{W}_s \mathbf{x} + \mathcal{F}(\mathbf{x})
$$

dove $\mathbf{W}_s$ è una proiezione lineare che adatta le dimensioni:

$$
\mathbf{W}_s: \mathbb{R}^{H \times W \times C_{\text{in}}} \rightarrow \mathbb{R}^{H' \times W' \times C_{\text{out}}}
$$

Tipicamente implementata con:
- Convoluzione $1 \times 1$ (per cambiare canali)
- Strided convolution o pooling (per ridurre risoluzione spaziale)

## Vantaggi delle Residual Connections

| Aspetto | Descrizione | Beneficio Matematico |
|---------|-------------|---------------------|
| **Stabilità del Training** | Mantiene il flusso del gradiente | $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ contiene sempre termine $\mathbf{I}$ |
| **Riutilizzo delle Feature** | Informazione originale preservata | $\mathbf{y} = \mathbf{x} + \Delta\mathbf{x}$ invece di $\mathbf{y} = f(\mathbf{x})$ |
| **Facilita Apprendimento** | Impara correzioni, non trasformazioni complete | $\mathcal{F}(\mathbf{x}) \approx \mathbf{0}$ invece di $\mathcal{F}(\mathbf{x}) \approx \mathbf{x}$ |
| **Ensemble Implicito** | Ogni percorso è una sotto-rete | $2^L$ combinazioni di percorsi possibili |
| **Profondità Arbitraria** | Permette reti con centinaia di layer | Gradiente non svanisce |
| **Convergenza Più Veloce** | Inizializzazione vicina all'identità | Meno epoche necessarie |

## Implementazione Pratica

### Codice Base

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Blocco residuale generico"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.transform = nn.Linear(dim, dim)
    
    def forward(self, x):
        # Residual connection
        return x + self.transform(self.norm(x))
```

### Con Drop Path

```python
class ResidualBlockWithDropPath(nn.Module):
    """Blocco residuale con Drop Path"""
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.transform = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x):
        # x + DropPath(F(x))
        return x + self.drop_path(self.transform(self.norm(x)))
```

### Attention Block Completo

```python
class SwinTransformerBlock(nn.Module):
    """Blocco Swin Transformer con residual connections"""
    def __init__(self, dim, num_heads, window_size=7, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=4.0)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x):
        # Prima residual connection (attenzione)
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        
        # Seconda residual connection (MLP)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        
        return x
```

### Projection Shortcut per Dimensioni Diverse

```python
class ResidualBlockWithProjection(nn.Module):
    """Blocco residuale con proiezione per dimensioni diverse"""
    def __init__(self, dim_in, dim_out, downsample=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.transform = nn.Linear(dim_in, dim_out)
        
        # Projection shortcut se le dimensioni cambiano
        if dim_in != dim_out or downsample:
            self.shortcut = nn.Linear(dim_in, dim_out)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        # Ws * x + F(x)
        return self.shortcut(x) + self.transform(self.norm(x))
```

## Varianti delle Residual Connections

### Pre-Activation Residual

Nel design originale di ResNet:

$$
\mathbf{y} = \text{ReLU}(\mathbf{x} + \mathcal{F}(\mathbf{x}))
$$

Nel design "pre-activation" (He et al., 2016):

$$
\mathbf{y} = \mathbf{x} + \mathcal{F}(\text{ReLU}(\text{BN}(\mathbf{x})))
$$

**Vantaggio:** Flusso del gradiente ancora più pulito.

### Post-LayerNorm (Transformer Standard)

Schema originale del Transformer (Vaswani et al., 2017):

$$
\mathbf{y} = \text{LayerNorm}(\mathbf{x} + \mathcal{F}(\mathbf{x}))
$$

### Pre-LayerNorm (Moderno)

Schema usato in modelli moderni (GPT, Swin, ViT):

$$
\mathbf{y} = \mathbf{x} + \mathcal{F}(\text{LayerNorm}(\mathbf{x}))
$$

**Vantaggi:**
- Training più stabile
- Gradienti più puliti
- No need for learning rate warmup

### Weighted Residual Connections

In alcuni modelli (es. ReZero, SkipInit):

$$
\mathbf{y} = \mathbf{x} + \alpha \cdot \mathcal{F}(\mathbf{x})
$$

dove $\alpha$ è un parametro apprendibile inizializzato a 0 o vicino a 0.

**Idea:** Inizialmente la rete è quasi identità, poi gradualmente impara.

## Analisi Teorica

### Numero di Percorsi

In una rete con $L$ blocchi residuali, ogni blocco offre due percorsi:
- Identity path
- Residual path

Il numero totale di percorsi dall'input all'output è:

$$
2^L
$$

**Esempio:** Con $L = 50$ (ResNet-50), abbiamo $2^{50} \approx 10^{15}$ percorsi!

### Interpretazione come Ensemble

La rete può essere vista come un **ensemble implicito** di $2^L$ reti più shallow che condividono parametri.

Durante il training, diverse combinazioni di percorsi vengono attivate casualmente (specialmente con Drop Path), creando un effetto di ensemble.

### Lunghezza Effettiva dei Percorsi

La lunghezza media dei percorsi in una rete con $L$ blocchi è:

$$
\mathbb{E}[\text{path length}] = \frac{L}{2}
$$

Questo spiega perché le residual networks si comportano come reti più shallow.

## Applicazioni e Risultati Empirici

### ImageNet Classification

**ResNet vs Plain Networks:**
- Plain-34: 28.54% top-1 error
- ResNet-34: 25.03% top-1 error (↓3.5%)
- ResNet-152: 21.43% top-1 error

**Con più layer, performance migliora** (contrariamente alle plain networks).

### Vision Transformer

ViT-Large (24 layer) con residual connections:
- Converge in ~300 epoche
- Accuracy: 87.76% su ImageNet

Senza residual connections:
- Training instabile
- Non converge

### Swin Transformer

Swin-Base con residual connections + Drop Path:
- 83.5% top-1 accuracy su ImageNet
- Training stabile anche con 24 stage

## Conclusioni

Le **residual connections** sono una delle innovazioni più importanti del deep learning moderno. Permettono:

1. **Training di reti arbitrariamente profonde** risolvendo il vanishing gradient problem
2. **Apprendimento più efficiente** focalizzandosi su correzioni residue
3. **Maggiore stabilità** garantendo un flusso costante del gradiente
4. **Ensemble implicito** creando esponenzialmente molti percorsi
5. **Riutilizzo delle feature** preservando informazione originale

Nella pratica, sono diventate uno **standard de facto** in quasi tutte le architetture moderne, dai CNN (ResNet, EfficientNet) ai Transformer (BERT, GPT, ViT, Swin).

## Riferimenti

1. **He et al. (2015)**: "Deep Residual Learning for Image Recognition" - Paper originale ResNet
2. **He et al. (2016)**: "Identity Mappings in Deep Residual Networks" - Pre-activation design
3. **Vaswani et al. (2017)**: "Attention Is All You Need" - Residual in Transformer
4. **Dosovitskiy et al. (2020)**: "An Image is Worth 16x16 Words" - ViT
5. **Liu et al. (2021)**: "Swin Transformer" - Hierarchical vision transformer
6. **Veit et al. (2016)**: "Residual Networks Behave Like Ensembles" - Analisi teorica
