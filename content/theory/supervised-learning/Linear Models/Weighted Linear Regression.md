# Regressione Ponderata (Weighted Regression)

La **Regressione Ponderata** (Weighted Regression) è un'estensione della [[Regressione Lineare]] che permette di assegnare **pesi differenti** alle osservazioni durante il processo di stima dei parametri. A differenza della regressione lineare ordinaria dove tutte le osservazioni contribuiscono ugualmente alla funzione di costo, la regressione ponderata riflette l'**importanza relativa** o l'**affidabilità** di ciascun punto dati.

## **1. Motivazione e Contesto**

### **1.1. Problemi della Regressione Lineare Ordinaria**

Nella regressione lineare classica, l'assunzione di **omoschedasticità** (varianza costante degli errori) e **uguale affidabilità** di tutte le osservazioni non sempre riflette la realtà. Situazioni comuni dove queste assunzioni sono violate includono:

1. **Eteroschedasticità**: La varianza dell'errore varia tra le osservazioni
2. **Affidabilità variabile**: Alcune misurazioni sono intrinsecamente più precise di altre
3. **Dimensione del campione variabile**: Osservazioni rappresentano gruppi di dimensioni diverse
4. **Outliers noti**: Alcuni punti sono chiaramente anomali ma non eliminabili
5. **Precisione strumentale**: Diversi strumenti di misura hanno accuratezze differenti

### **1.2. Filosofia della Ponderazione**

La regressione ponderata risolve questi problemi assegnando un **peso** $w_i$ a ciascuna osservazione $i$:

- **Peso alto** ($w_i$ grande): Osservazione più affidabile/importante → maggiore influenza sulla stima
- **Peso basso** ($w_i$ piccolo): Osservazione meno affidabile → minore influenza sulla stima  
- **Peso zero** ($w_i = 0$): Osservazione ignorata completamente

## **2. Formulazione Matematica**

### **2.1. Problema di Ottimizzazione**

Dato un dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i, w_i)\}_{i=1}^n$ dove $w_i > 0$ sono i pesi, il problema di ottimizzazione diventa:

$$
\min_{\mathbf{w}} \sum_{i=1}^{n} w_i \left( y_i - \mathbf{x}_i^T \boldsymbol{\theta} \right)^2
$$

Dove:
- $\boldsymbol{\theta} \in \mathbb{R}^{d+1}$ è il vettore dei parametri (incluso il bias)
- $\mathbf{x}_i \in \mathbb{R}^{d+1}$ è il vettore delle features (con 1 aggiunto per il bias)
- $w_i > 0$ è il peso dell'osservazione $i$
- $y_i$ è la variabile target

### **2.2. Interpretazione Probabilistica**

La regressione ponderata equivale alla **Maximum Likelihood Estimation** quando gli errori seguono distribuzioni normali con **varianze eterogenee**:

$$
\epsilon_i \sim \mathcal{N}(0, \sigma_i^2)
$$

Il peso ottimale è inversamente proporzionale alla varianza:
$$
w_i = \frac{1}{\sigma_i^2}
$$

Questo significa che osservazioni con **varianza più bassa** (più precise) ricevono **peso maggiore**.

### **2.3. Formulazione Matriciale**

In notazione matriciale, il problema diventa:

$$
\min_{\boldsymbol{\theta}} (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^T \mathbf{W} (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})
$$

Dove:
- $\mathbf{X} \in \mathbb{R}^{n \times (d+1)}$ è la matrice di design
- $\mathbf{y} \in \mathbb{R}^{n \times 1}$ è il vettore target  
- $\mathbf{W} \in \mathbb{R}^{n \times n}$ è la matrice diagonale dei pesi:

$$
\mathbf{W} = \begin{bmatrix}
w_1 & 0 & \cdots & 0 \\
0 & w_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & w_n
\end{bmatrix}
$$

## **3. Soluzione Analitica: Weighted Least Squares (WLS)**

### **3.1. Derivazione della Soluzione**

La funzione di costo pesata è:
$$
J(\boldsymbol{\theta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^T \mathbf{W} (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})
$$

Espandendo:
$$
J(\boldsymbol{\theta}) = \mathbf{y}^T \mathbf{W} \mathbf{y} - 2\mathbf{y}^T \mathbf{W} \mathbf{X} \boldsymbol{\theta} + \boldsymbol{\theta}^T \mathbf{X}^T \mathbf{W} \mathbf{X} \boldsymbol{\theta}
$$

### **3.2. Calcolo del Gradiente**

$$
\frac{\partial J}{\partial \boldsymbol{\theta}} = -2\mathbf{X}^T \mathbf{W} \mathbf{y} + 2\mathbf{X}^T \mathbf{W} \mathbf{X} \boldsymbol{\theta}
$$

### **3.3. Soluzione Ottimale**

Ponendo il gradiente uguale a zero:
$$
\mathbf{X}^T \mathbf{W} \mathbf{X} \boldsymbol{\theta} = \mathbf{X}^T \mathbf{W} \mathbf{y}
$$

La soluzione in forma chiusa è:
$$
\boldsymbol{\theta}^* = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{y}
$$

### **3.4. Confronto con OLS**

La formula è identica a quella della [[Regressione Lineare]], ma con l'aggiunta della matrice dei pesi $\mathbf{W}$:

| Metodo | Formula |
|--------|---------|
| **OLS** | $\boldsymbol{\theta}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ |
| **WLS** | $\boldsymbol{\theta}^* = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{y}$ |

Quando $\mathbf{W} = \mathbf{I}$ (tutti i pesi uguali a 1), WLS si riduce a OLS.

## **4. Proprietà Statistiche**

### **4.1. Stimatore BLUE**

Sotto le assunzioni di Gauss-Markov modificate per la regressione ponderata, WLS produce lo stimatore **Best Linear Unbiased Estimator (BLUE)**:

1. **Linearità**: $\boldsymbol{\theta}^*$ è una combinazione lineare di $\mathbf{y}$
2. **Non distorsione**: $\mathbb{E}[\boldsymbol{\theta}^*] = \boldsymbol{\theta}_{\text{vero}}$  
3. **Minima varianza**: Tra tutti gli stimatori lineari non distorti

### **4.2. Matrice di Varianza-Covarianza**

La matrice di varianza-covarianza dello stimatore WLS è:
$$
\text{Var}(\boldsymbol{\theta}^*) = \sigma^2 (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1}
$$

Gli errori standard dei coefficienti sono:
$$
SE(\theta_j) = \sigma \sqrt{[(\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1}]_{jj}}
$$

### **4.3. Stima della Varianza Residua**

$$
\hat{\sigma}^2 = \frac{\sum_{i=1}^n w_i (y_i - \hat{y}_i)^2}{n - p - 1}
$$

dove $p$ è il numero di predittori (escluso il bias).

## **5. Scelta dei Pesi**

### **5.1. Pesi Basati sulla Varianza**

Quando la varianza dell'errore è nota o stimabile:
$$
w_i = \frac{1}{\sigma_i^2}
$$

**Esempi:**
- Misurazioni con precisione nota
- Dati aggregati con numerosità diverse per gruppo
- Strumenti con accuratezza calibrata

### **5.2. Pesi Basati sulla Dimensione del Campione**

Per dati aggregati rappresentanti gruppi di dimensioni $n_i$:
$$
w_i = n_i
$$

Questo assume che la varianza dell'errore sia inversamente proporzionale alla dimensione del gruppo.

### **5.3. Pesi per Gestire Outliers**

#### **5.3.1. Pesi di Huber**
$$
w_i = \begin{cases}
1 & \text{se } |r_i| \leq k \\
\frac{k}{|r_i|} & \text{se } |r_i| > k
\end{cases}
$$

dove $r_i$ è il residuo standardizzato e $k$ è una soglia (tipicamente $k = 1.345$).

#### **5.3.2. Pesi di Tukey (Bisquare)**
$$
w_i = \begin{cases}
\left(1 - \left(\frac{r_i}{k}\right)^2\right)^2 & \text{se } |r_i| \leq k \\
0 & \text{se } |r_i| > k
\end{cases}
$$

### **5.4. Pesi Adattivi: Weighted Iteratively Reweighted Least Squares (IRLS)**

Un approccio iterativo per stimare automaticamente i pesi:

1. **Inizializzazione**: Eseguire OLS per ottenere $\hat{\boldsymbol{\theta}}^{(0)}$
2. **Per $t = 1, 2, \ldots$ fino a convergenza:**
   - Calcolare residui: $r_i^{(t-1)} = y_i - \mathbf{x}_i^T \hat{\boldsymbol{\theta}}^{(t-1)}$
   - Aggiornare pesi: $w_i^{(t)} = f(r_i^{(t-1)})$ usando una funzione peso robusta
   - Ricalcolare: $\hat{\boldsymbol{\theta}}^{(t)} = (\mathbf{X}^T \mathbf{W}^{(t)} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W}^{(t)} \mathbf{y}$

## **6. Implementazione Pratica**

### **6.1. Implementazione Base**

```python {visible}
import torch

def weighted_least_squares(X, y, weights):
    """
    Weighted Least Squares implementation
    Args:
        X: design matrix (n, d+1) with bias column
        y: target vector (n, 1)
        weights: weight vector (n,)
    Returns:
        theta: parameter estimates
        sigma2: residual variance estimate
    """
    n, d = X.shape
    
    # Ensure correct dimensions
    if y.dim() == 1:
        y = y.unsqueeze(1)
    if weights.dim() == 1:
        weights = weights.unsqueeze(1)
    
    # Create weight matrix
    W = torch.diag(weights.squeeze())
    
    # WLS solution: θ = (X^T W X)^(-1) X^T W y
    XTW = X.T @ W
    XTWX = XTW @ X
    theta = torch.inverse(XTWX) @ XTW @ y
    
    # Residuals and variance estimate
    y_pred = X @ theta
    residuals = y - y_pred
    weighted_residuals = weights * residuals**2
    sigma2 = weighted_residuals.sum() / (n - d)
    
    # Covariance matrix
    cov_matrix = sigma2 * torch.inverse(XTWX)
    
    return theta, sigma2, cov_matrix
```

### **6.2. Implementazione IRLS per Robustezza**

```python {visible}
def irls_robust_regression(X, y, max_iter=10, tol=1e-6, c=1.345):
    """
    Iteratively Reweighted Least Squares with Huber weights
    """
    n, d = X.shape
    
    # Initialize with OLS
    theta = torch.inverse(X.T @ X) @ X.T @ y
    
    for iteration in range(max_iter):
        # Calculate residuals
        residuals = y - X @ theta
        
        # Calculate robust scale (MAD-based)
        mad = torch.median(torch.abs(residuals - torch.median(residuals)))
        scale = 1.4826 * mad  # Convert MAD to standard deviation estimate
        
        # Standardized residuals
        std_residuals = residuals / scale
        
        # Huber weights
        weights = torch.ones_like(std_residuals)
        mask = torch.abs(std_residuals) > c
        weights[mask] = c / torch.abs(std_residuals[mask])
        
        # WLS step
        W = torch.diag(weights.squeeze())
        XTW = X.T @ W
        XTWX = XTW @ X
        theta_new = torch.inverse(XTWX) @ XTW @ y
        
        # Check convergence
        if torch.norm(theta_new - theta) < tol:
            break
            
        theta = theta_new
    
    return theta, weights
```

## **7. Diagnostica e Valutazione**

### **7.1. Metriche Ponderate**

Le metriche di performance devono essere adattate per riflettere la ponderazione:

#### **Mean Weighted Squared Error (MWSE)**
$$
MWSE = \frac{\sum_{i=1}^n w_i (y_i - \hat{y}_i)^2}{\sum_{i=1}^n w_i}
$$

#### **Weighted R-squared**
$$
R_w^2 = 1 - \frac{\sum_{i=1}^n w_i (y_i - \hat{y}_i)^2}{\sum_{i=1}^n w_i (y_i - \bar{y}_w)^2}
$$

dove $\bar{y}_w = \frac{\sum_{i=1}^n w_i y_i}{\sum_{i=1}^n w_i}$ è la media ponderata.

### **7.2. Analisi dei Residui Ponderati**

I residui standardizzati ponderati sono:
$$
r_{i,\text{std}} = \frac{r_i}{\hat{\sigma} \sqrt{1 - h_{ii}}} \sqrt{w_i}
$$

dove $h_{ii}$ è l'elemento diagonale della hat matrix: $\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W}$.

### **7.3. Test di Significatività**

I test statistici (test t per i coefficienti, test F globale) rimangono validi usando gli errori standard corretti dalla matrice di varianza-covarianza ponderata.

## **8. Confronto con Metodi Correlati**

### **8.1. Regressione Ponderata vs [[Regressione Lineare]]**

| Aspetto | OLS | WLS |
|---------|-----|-----|
| **Assunzioni sui pesi** | Tutti uguali ($w_i = 1$) | Possono variare |
| **Trattamento eteroschedasticità** | Inadeguato | Appropriato |
| **Robustezza agli outliers** | Bassa | Configurabile |
| **Complessità implementativa** | Bassa | Moderata |
| **Efficienza computazionale** | Alta | Leggermente inferiore |

### **8.2. Regressione Ponderata vs [[Locally Weighted Linear Regression]]**

| Aspetto | Weighted Regression | LWLR |
|---------|-------------------|------|
| **Natura dei pesi** | Fissi per dataset | Adattivi per ogni query |
| **Numero di modelli** | Uno globale | Uno per ogni predizione |
| **Capacità non-lineare** | Nessuna | Elevata |
| **Costo computazionale** | $O(d^3)$ una volta | $O(d^3)$ per query |
| **Interpretabilità** | Alta | Moderata |
| **Memoria richiesta** | Bassa | Alta (tutto il training set) |

### **8.3. Posizionamento nel Continuum**

```
OLS → Weighted Regression → LWLR → Kernel Regression
 ↑           ↑                ↑            ↑
Rigido    Pesi fissi    Pesi adattivi  Completamente
                                       non-parametrico
```

La regressione ponderata occupa una posizione intermedia, mantenendo la semplicità parametrica della regressione lineare mentre introducendo flessibilità nella gestione dell'affidabilità dei dati.

## **9. Applicazioni Pratiche**

### **9.1. Econometria**

- **Studi panel**: Pesi basati sulla precisione delle stime per paese/anno
- **Sondaggi**: Pesi per correggere bias di campionamento
- **Dati aggregati**: Pesi proporzionali alla popolazione rappresentata

### **9.2. Ingegneria**

- **Calibrazione strumenti**: Pesi basati sulla precisione di misura
- **Controllo qualità**: Pesi per riflettere l'importanza di specifiche caratteristiche
- **Modellazione fisica**: Pesi per dati da esperimenti con condizioni diverse

### **9.3. Medicina e Biologia**

- **Meta-analisi**: Pesi basati sulla dimensione degli studi
- **Dati clinici**: Pesi per pazienti con follow-up di durata diversa  
- **Farmacocinetrica**: Pesi per concentrazioni misurate a tempi diversi

### **9.4. Machine Learning**

- **Class imbalancing**: Pesi inversamente proporzionali alla frequenza delle classi
- **Active learning**: Pesi maggiori per campioni informativi
- **Transfer learning**: Pesi per bilanciare domini sorgente e target

## **10. Limitazioni e Considerazioni**

### **10.1. Scelta dei Pesi**

La performance della regressione ponderata dipende criticamente dalla scelta appropriata dei pesi:

- **Pesi ottimali**: Richiedono conoscenza della vera varianza degli errori
- **Pesi errati**: Possono peggiorare le prestazioni rispetto a OLS
- **Soggettività**: Spesso la scelta dei pesi contiene elementi arbitrari

### **10.2. Problemi Numerici**

- **Condizionamento**: $\mathbf{X}^T \mathbf{W} \mathbf{X}$ può essere mal condizionata se alcuni pesi sono molto grandi
- **Stabilità**: Pesi vicini a zero possono causare instabilità numerica
- **Scalabilità**: La costruzione della matrice $\mathbf{W}$ può essere costosa per $n$ grande

### **10.3. Interpretazione**

- I coefficienti WLS non sono direttamente confrontabili con quelli OLS
- L'interpretazione deve considerare l'effetto della ponderazione
- Le metriche di performance standard (R², MSE) richiedono adattamento

## **11. Estensioni e Varianti**

### **11.1. Weighted Ridge Regression**

Combinazione di ponderazione e regolarizzazione L2:
$$
\min_{\boldsymbol{\theta}} \sum_{i=1}^n w_i (y_i - \mathbf{x}_i^T \boldsymbol{\theta})^2 + \lambda \|\boldsymbol{\theta}\|_2^2
$$

Soluzione:
$$
\boldsymbol{\theta}^* = (\mathbf{X}^T \mathbf{W} \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{y}
$$

### **11.2. Weighted Elastic Net**

Estende elastic net con ponderazione:
$$
\min_{\boldsymbol{\theta}} \sum_{i=1}^n w_i (y_i - \mathbf{x}_i^T \boldsymbol{\theta})^2 + \lambda_1 \|\boldsymbol{\theta}\|_1 + \lambda_2 \|\boldsymbol{\theta}\|_2^2
$$

### **11.3. Generalized Least Squares (GLS)**

Estensione a matrici di covarianza complete (non necessariamente diagonali):
$$
\boldsymbol{\theta}^* = (\mathbf{X}^T \boldsymbol{\Omega}^{-1} \mathbf{X})^{-1} \mathbf{X}^T \boldsymbol{\Omega}^{-1} \mathbf{y}
$$

dove $\boldsymbol{\Omega}$ è la matrice di covarianza completa degli errori.

## **12. Conclusioni**

La regressione ponderata rappresenta un'estensione naturale e potente della [[Regressione Lineare]] che mantiene la semplicità concettuale e computazionale del modello lineare incorporando flessibilità nella gestione dell'affidabilità e importanza dei dati.

### **12.1. Punti di Forza**

- **Semplicità**: Modifica minimale della regressione lineare standard
- **Flessibilità**: Adatta a molteplici scenari pratici
- **Fondamento teorico**: Stimatore BLUE sotto assunzioni appropriate
- **Implementazione**: Richiede modifiche minime agli algoritmi esistenti

### **12.2. Quando Utilizzarla**

La regressione ponderata è particolarmente indicata quando:
- Le osservazioni hanno affidabilità/precisione nota e variabile
- Si vuole gestire l'eteroschedasticità senza perdere la semplicità lineare
- È necessario un controllo esplicito sull'influenza di specifiche osservazioni  
- Il dataset contiene outliers identificabili ma non eliminabili

### **12.3. Posizionamento nel Panorama ML**

Collocandosi tra la rigidità della regressione lineare ordinaria e la flessibilità della [[Locally Weighted Linear Regression]], la regressione ponderata offre un equilibrio ottimale per molte applicazioni pratiche dove è necessario mantenere interpretabilità e semplicità computazionale pur adattandosi alle caratteristiche variabili dei dati.
