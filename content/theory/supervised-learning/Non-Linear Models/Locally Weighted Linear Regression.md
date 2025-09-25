# Local Weighted Linear Regression (LWLR)

La **Local Weighted Linear Regression** (LWLR), nota anche come **Locally Weighted Regression** o **LOESS** (Locally Estimated Scatterplot Smoothing), è un'estensione non parametrica della regressione lineare che adatta un modello lineare localmente per ogni punto di query. A differenza della regressione lineare classica che trova un unico modello globale, LWLR costruisce un modello diverso per ogni previsione, dando più peso ai punti di training vicini al punto di query.

## **1. Motivazione e Intuizione**

Nella regressione lineare tradizionale, tutti i punti del training set contribuiscono ugualmente alla determinazione dei parametri del modello. Tuttavia, in molte applicazioni reali, la relazione tra variabili può variare localmente. LWLR risolve questo problema:

- **Adattandosi localmente** alla struttura dei dati
- **Pesando maggiormente** i punti vicini al punto di query
- **Riducendo l'influenza** dei punti lontani
- **Catturando pattern non lineari** attraverso approssimazioni lineari locali

### **1.1. Esempio Intuitivo**
Consideriamo una relazione non lineare tra temperatura e vendite di gelato. Un modello lineare globale potrebbe non catturare bene le variazioni stagionali, mentre LWLR può adattarsi localmente: in estate darà più peso ai dati estivi vicini, in inverno ai dati invernali, catturando così meglio la variabilità locale.

## **2. Formulazione Matematica**

### **2.1. Caso Univariato (Forma Vettoriale)**

Consideriamo un dataset di training $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^m$ dove $x_i \in \mathbb{R}$ e $y_i \in \mathbb{R}$.

Per un punto di query $x_q$, LWLR risolve il seguente problema di ottimizzazione pesato:

$$
\min_{\theta_0, \theta_1} \sum_{i=1}^{m} w_i(x_q) \left( y_i - \theta_0 - \theta_1 x_i \right)^2
$$

Dove:
- $\theta_0$ è l'intercetta (bias) del modello locale
- $\theta_1$ è il coefficiente angolare del modello locale
- $w_i(x_q)$ è il peso assegnato al punto $i$-esimo in funzione della sua distanza da $x_q$

### **2.2. Funzione Peso (Kernel)**

Il peso $w_i(x_q)$ è tipicamente definito usando un kernel gaussiano:

$$
w_i(x_q) = \exp\left(-\frac{(x_i - x_q)^2}{2\tau^2}\right)
$$

Dove:
- $\tau > 0$ è il **bandwidth parameter** che controlla la "larghezza" della finestra locale
- $\tau$ piccolo → finestra stretta → modello più "wiggly" (alta varianza, basso bias)
- $\tau$ grande → finestra larga → modello più smooth (bassa varianza, alto bias)

### **2.3. Formulazione Matriciale Generale**

Per il caso multivariato con dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^m$ dove $\mathbf{x}_i \in \mathbb{R}^d$ e $y_i \in \mathbb{R}$.

#### **Notazione Matriciale**
- $\mathbf{X} \in \mathbb{R}^{m \times (d+1)}$ è la matrice di design con bias:
  $$
  \mathbf{X} = \begin{bmatrix}
  1 & x_{1,1} & x_{1,2} & \cdots & x_{1,d} \\
  1 & x_{2,1} & x_{2,2} & \cdots & x_{2,d} \\
  \vdots & \vdots & \vdots & \ddots & \vdots \\
  1 & x_{m,1} & x_{m,2} & \cdots & x_{m,d}
  \end{bmatrix}
  $$

- $\mathbf{y} \in \mathbb{R}^{m \times 1}$ è il vettore delle variabili target:
  $$
  \mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_m \end{bmatrix}
  $$

- $\boldsymbol{\theta} \in \mathbb{R}^{(d+1) \times 1}$ è il vettore dei parametri locali:
  $$
  \boldsymbol{\theta} = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_d \end{bmatrix}
  $$

- $\mathbf{W}(\mathbf{x}_q) \in \mathbb{R}^{m \times m}$ è la matrice diagonale dei pesi:
  $$
  \mathbf{W}(\mathbf{x}_q) = \begin{bmatrix}
  w_1(\mathbf{x}_q) & 0 & \cdots & 0 \\
  0 & w_2(\mathbf{x}_q) & \cdots & 0 \\
  \vdots & \vdots & \ddots & \vdots \\
  0 & 0 & \cdots & w_m(\mathbf{x}_q)
  \end{bmatrix}
  $$

#### **Funzione Peso Multivariata**
$$
w_i(\mathbf{x}_q) = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_q\|_2^2}{2\tau^2}\right)
$$

#### **Problema di Ottimizzazione**
$$
\min_{\boldsymbol{\theta}} \sum_{i=1}^{m} w_i(\mathbf{x}_q) \left( y_i - \mathbf{x}_i^T \boldsymbol{\theta} \right)^2
$$

In forma matriciale:
$$
\min_{\boldsymbol{\theta}} (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^T \mathbf{W}(\mathbf{x}_q) (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})
$$

## **3. Soluzione Analitica (Weighted Least Squares)**

### **3.1. Derivazione della Soluzione Ottimale**

La funzione obiettivo da minimizzare è:
$$
J(\boldsymbol{\theta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^T \mathbf{W}(\mathbf{x}_q) (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})
$$

Espandendo:
$$
J(\boldsymbol{\theta}) = \mathbf{y}^T \mathbf{W} \mathbf{y} - 2\mathbf{y}^T \mathbf{W} \mathbf{X} \boldsymbol{\theta} + \boldsymbol{\theta}^T \mathbf{X}^T \mathbf{W} \mathbf{X} \boldsymbol{\theta}
$$

### **3.2. Calcolo del Gradiente**

Il gradiente rispetto a $\boldsymbol{\theta}$ è:
$$
\frac{\partial J}{\partial \boldsymbol{\theta}} = -2\mathbf{X}^T \mathbf{W} \mathbf{y} + 2\mathbf{X}^T \mathbf{W} \mathbf{X} \boldsymbol{\theta}
$$

**Derivazione dettagliata:**
- $\frac{\partial}{\partial \boldsymbol{\theta}}(\mathbf{y}^T \mathbf{W} \mathbf{y}) = \mathbf{0}$ (non dipende da $\boldsymbol{\theta}$)
- $\frac{\partial}{\partial \boldsymbol{\theta}}(-2\mathbf{y}^T \mathbf{W} \mathbf{X} \boldsymbol{\theta}) = -2\mathbf{X}^T \mathbf{W} \mathbf{y}$
- $\frac{\partial}{\partial \boldsymbol{\theta}}(\boldsymbol{\theta}^T \mathbf{X}^T \mathbf{W} \mathbf{X} \boldsymbol{\theta}) = 2\mathbf{X}^T \mathbf{W} \mathbf{X} \boldsymbol{\theta}$

### **3.3. Soluzione Ottimale**

Ponendo il gradiente uguale a zero:
$$
-2\mathbf{X}^T \mathbf{W} \mathbf{y} + 2\mathbf{X}^T \mathbf{W} \mathbf{X} \boldsymbol{\theta} = \mathbf{0}
$$

Risolvendo per $\boldsymbol{\theta}$:
$$
\boldsymbol{\theta}^*(\mathbf{x}_q) = (\mathbf{X}^T \mathbf{W}(\mathbf{x}_q) \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W}(\mathbf{x}_q) \mathbf{y}
$$

### **3.4. Predizione**

La predizione per il punto di query $\mathbf{x}_q$ è:
$$
\hat{y}_q = \mathbf{x}_q^T \boldsymbol{\theta}^*(\mathbf{x}_q)
$$

Sostituendo la soluzione ottimale:
$$
\hat{y}_q = \mathbf{x}_q^T (\mathbf{X}^T \mathbf{W}(\mathbf{x}_q) \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W}(\mathbf{x}_q) \mathbf{y}
$$

### **3.5. Implementazione in Forma Chiusa**

L'implementazione della soluzione analitica richiede alcuni passaggi fondamentali:

#### **Algoritmo:**
1. **Preparazione dei dati:**
   - Assicurarsi che $\mathbf{y}$ sia un vettore colonna $(m \times 1)$
   - Aggiungere colonna di 1's a $\mathbf{X}$ per il bias: $\mathbf{X}_{\text{aug}} \in \mathbb{R}^{m \times (d+1)}$
   - Estendere il punto query: $\mathbf{x}_{q,\text{aug}} \in \mathbb{R}^{1 \times (d+1)}$

2. **Calcolo dei pesi:**
   - Calcolare le distanze: $\text{diff}_i = \mathbf{x}_i - \mathbf{x}_q$ per $i = 1,\ldots,m$
   - Calcolare i pesi: $w_i = \exp\left(-\frac{\|\text{diff}_i\|^2}{2\tau^2}\right)$
   - Costruire la matrice diagonale: $\mathbf{W} = \text{diag}(w_1, w_2, \ldots, w_m)$

3. **Risoluzione del sistema:**
   - Calcolare $\mathbf{X}^T \mathbf{W}$
   - Calcolare $\mathbf{X}^T \mathbf{W} \mathbf{X}$
   - Risolvere: $\boldsymbol{\theta}^* = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{y}$

4. **Predizione:**
   - $\hat{y}_q = \mathbf{x}_{q,\text{aug}}^T \boldsymbol{\theta}^*$

#### **Implementazione:**

```python
def lwlr_CF(x_q, X, y, t=0.5):
    """
    Locally Weighted Linear Regression in forma chiusa
    Args:
        x_q: punto query (1, d)
        X: training data (m, d)  
        y: training labels (m,) o (m, 1)
        t: bandwidth parameter
    Returns:
        y_pred: predizione per x_q
        theta_weights: coefficienti delle features  
        theta_bias: intercetta
    """
    m, d = X.shape
    
    # Assicuriamoci che y sia un vettore colonna
    if y.dim() == 1:
        y = y.unsqueeze(1)  # (m, 1)
    
    # Aggiungi colonna di 1 per il bias
    X_aug = torch.cat([X, torch.ones(m, 1)], dim=1)  # (m, d+1)
    x_q_aug = torch.cat([x_q, torch.ones(1, 1)], dim=1)  # (1, d+1)
    
    # Calcolo dei pesi
    diff = X - x_q  # (m, d)
    weights = torch.exp(-diff.norm(dim=1)**2 / (2 * t**2))  # (m,)
    W = torch.diag(weights)  # (m, m)
    
    # Soluzione in forma chiusa: θ = (X^T W X)^(-1) X^T W y
    XTW = X_aug.T @ W  # (d+1, m)
    XTWX = XTW @ X_aug  # (d+1, d+1)
    theta = torch.inverse(XTWX) @ XTW @ y  # (d+1, 1)
    
    # Predizione
    y_pred = (x_q_aug @ theta).item()
    
    return y_pred, theta[:-1], theta[-1].item()
```

#### **Considerazioni Numeriche:**

1. **Stabilità dell'inversione:** La matrice $\mathbf{X}^T \mathbf{W} \mathbf{X}$ può essere mal condizionata se:
   - I punti sono quasi collineari nell'intorno locale
   - Alcuni pesi sono molto piccoli (vicini a zero)
   - Il bandwidth $\tau$ è troppo piccolo

2. **Alternative numericamente stabili:**
   - Usare la **decomposizione SVD**: $\boldsymbol{\theta}^* = \mathbf{V} \boldsymbol{\Sigma}^{-1} \mathbf{U}^T \mathbf{W} \mathbf{y}$
   - Usare **pseudo-inversa di Moore-Penrose**: $\boldsymbol{\theta}^* = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{\dagger} \mathbf{X}^T \mathbf{W} \mathbf{y}$
   - Aggiungere **regolarizzazione Ridge**: $(\mathbf{X}^T \mathbf{W} \mathbf{X} + \lambda \mathbf{I})^{-1}$

### **3.6. Vantaggi e Svantaggi della Forma Chiusa**

#### **Vantaggi:**
- **Convergenza garantita** in una sola iterazione
- **Soluzione esatta** (modulo errori numerici)
- **Deterministica** - risultati riproducibili
- **Veloce** per piccoli dataset

#### **Svantaggi:**
- **Complessità computazionale** $O(d^3)$ per l'inversione della matrice
- **Instabilità numerica** quando $\mathbf{X}^T \mathbf{W} \mathbf{X}$ è mal condizionata
- **Memoria** richiesta per memorizzare la matrice dei pesi
- **Non scalabile** per dataset molto grandi

## **4. Implementazioni con Gradient Descent**

Quando la soluzione analitica è computazionalmente proibitiva o numericamente instabile, si può utilizzare il gradient descent per ottimizzare i parametri.

### **4.1. Stochastic Gradient Descent (SGD)**

Nel SGD, i parametri vengono aggiornati per ogni singolo punto di training.

#### **Algoritmo SGD per LWLR:**

1. **Inizializzazione:** $\boldsymbol{\theta}^{(0)} = \mathbf{0}, b^{(0)} = 0$
2. **Per ogni epoca $t = 1, \ldots, T$:**
   3. **Per ogni campione $i = 1, \ldots, m$:**
      - Calcola peso: $w_i = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_q\|^2}{2\tau^2}\right)$
      - Calcola predizione: $\hat{y}_i = \mathbf{x}_i^T \boldsymbol{\theta} + b$
      - Calcola errore: $e_i = y_i - \hat{y}_i$
      - Calcola loss pesata: $L_i = \frac{1}{2} w_i e_i^2$
      - **Aggiorna parametri:**
        - $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot w_i \cdot e_i \cdot \mathbf{x}_i$
        - $b \leftarrow b + \alpha \cdot w_i \cdot e_i$

#### **Derivazione dei Gradienti SGD:**

Per un singolo campione $i$, la loss pesata è:
$$
L_i = \frac{1}{2} w_i (y_i - \mathbf{x}_i^T \boldsymbol{\theta} - b)^2
$$

I gradienti sono:
$$
\frac{\partial L_i}{\partial \boldsymbol{\theta}} = -w_i (y_i - \mathbf{x}_i^T \boldsymbol{\theta} - b) \mathbf{x}_i = -w_i e_i \mathbf{x}_i
$$

$$
\frac{\partial L_i}{\partial b} = -w_i (y_i - \mathbf{x}_i^T \boldsymbol{\theta} - b) = -w_i e_i
$$

Gli aggiornamenti dei parametri sono quindi:
$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha \frac{\partial L_i}{\partial \boldsymbol{\theta}} = \boldsymbol{\theta} + \alpha w_i e_i \mathbf{x}_i
$$

$$
b \leftarrow b - \alpha \frac{\partial L_i}{\partial b} = b + \alpha w_i e_i
$$

#### **Implementazione SGD:**

```python
def lwlr_SGD(x_q, X, y, t=0.5, lr=0.01, epochs=50):
    """
    Locally Weighted Linear Regression usando SGD
    Args:
        x_q: punto query (1, d)
        X: training data (m, d)
        y: training labels (m, 1)
        t: bandwidth parameter
        lr: learning rate
        epochs: numero di epochs
    """
    m, d = X.shape
    theta = torch.zeros(d, 1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    
    # Assicuriamoci che y sia della forma corretta
    if y.dim() == 1:
        y = y.unsqueeze(1)
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(m):
            xi = X[i:i+1]  # Mantieni dimensione (1, d)
            yi = y[i:i+1]  # Mantieni dimensione (1, 1)
            
            # Calcolo del peso
            diff = xi - x_q  # (1, d)
            wi = torch.exp(- diff.norm()**2 / (2 * t ** 2))
            
            # Predizione
            y_pred = xi @ theta + bias  # (1, 1)
            
            # Loss pesata
            loss = 0.5 * wi * (yi - y_pred) ** 2
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            with torch.no_grad():
                if theta.grad is not None:
                    theta -= lr * theta.grad
                if bias.grad is not None:
                    bias -= lr * bias.grad
            
            # Zero gradients
            if theta.grad is not None:
                theta.grad.zero_()
            if bias.grad is not None:
                bias.grad.zero_()
    
    # Predizione finale
    with torch.no_grad():
        y_pred = (x_q @ theta + bias).item()
    
    return y_pred, theta.detach(), bias.item()
```

### **4.2. Batch Gradient Descent (BGD)**

Nel BGD, i parametri vengono aggiornati utilizzando tutti i punti di training simultaneamente.

#### **Algoritmo BGD per LWLR:**

1. **Inizializzazione:** $\boldsymbol{\theta}^{(0)} = \mathbf{0}, b^{(0)} = 0$
2. **Per ogni epoca $t = 1, \ldots, T$:**
   - Calcola tutti i pesi: $w_i = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_q\|^2}{2\tau^2}\right), \forall i$
   - Calcola tutte le predizioni: $\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\theta} + b\mathbf{1}$
   - Calcola errori: $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$
   - Calcola loss pesata: $L = \frac{1}{2m} \sum_{i=1}^m w_i e_i^2$
   - **Aggiorna parametri:**
     - $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \frac{\alpha}{m} \sum_{i=1}^m w_i e_i \mathbf{x}_i$
     - $b \leftarrow b + \frac{\alpha}{m} \sum_{i=1}^m w_i e_i$

#### **Derivazione dei Gradienti BGD:**

La loss totale pesata è:
$$
L = \frac{1}{2m} \sum_{i=1}^m w_i (y_i - \mathbf{x}_i^T \boldsymbol{\theta} - b)^2
$$

I gradienti sono:
$$
\frac{\partial L}{\partial \boldsymbol{\theta}} = -\frac{1}{m} \sum_{i=1}^m w_i (y_i - \mathbf{x}_i^T \boldsymbol{\theta} - b) \mathbf{x}_i = -\frac{1}{m} \sum_{i=1}^m w_i e_i \mathbf{x}_i
$$

$$
\frac{\partial L}{\partial b} = -\frac{1}{m} \sum_{i=1}^m w_i (y_i - \mathbf{x}_i^T \boldsymbol{\theta} - b) = -\frac{1}{m} \sum_{i=1}^m w_i e_i
$$

In forma matriciale:
$$
\frac{\partial L}{\partial \boldsymbol{\theta}} = -\frac{1}{m} \mathbf{X}^T \mathbf{W} \mathbf{e}
$$

$$
\frac{\partial L}{\partial b} = -\frac{1}{m} \mathbf{1}^T \mathbf{W} \mathbf{e}
$$

Dove $\mathbf{W} = \text{diag}(w_1, w_2, \ldots, w_m)$ e $\mathbf{e} = \mathbf{y} - \mathbf{X}\boldsymbol{\theta} - b\mathbf{1}$.

#### **Implementazione BGD:**

```python
def lwlr_BGD(x_q, X, y, t=0.5, lr=0.01, epochs=100):
    """
    Locally Weighted Linear Regression con Batch Gradient Descent
    Args:
        x_q: punto query (1, d)
        X: training data (m, d)
        y: training labels (m,) o (m, 1)
        t: bandwidth parameter
        lr: learning rate
        epochs: numero di epoche
    """
    m, d = X.shape
    theta = torch.zeros(d, 1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    
    # Assicuriamoci che y sia colonna
    if y.dim() == 1:
        y = y.unsqueeze(1)  # (m,1)
    
    for epoch in range(epochs):
        # Differenze rispetto al punto query
        diff = X - x_q  # (m,d)
        wi = torch.exp(- diff.norm(dim=1)**2 / (2 * t**2))  # (m,)
        
        # Predizioni su tutti i dati
        y_pred = X @ theta + bias  # (m,1)
        
        # Loss pesata globale
        residuals = y - y_pred  # (m,1)
        loss = (wi.unsqueeze(1) * residuals**2).mean() / 2
        
        # Backward
        loss.backward()
        
        # Aggiornamento parametri
        with torch.no_grad():
            if theta.grad is not None:
                theta -= lr * theta.grad
            if bias.grad is not None:
                bias -= lr * bias.grad
        
        # Azzera gradienti
        if theta.grad is not None:
            theta.grad.zero_()
        if bias.grad is not None:
            bias.grad.zero_()
    
    # Predizione finale sul punto query
    with torch.no_grad():
        y_pred_q = (x_q @ theta + bias).item()
    
    return y_pred_q, theta.detach(), bias.item()
```

## **5. Proprietà e Caratteristiche di LWLR**

### **5.1. Vantaggi**

1. **Non parametrico**: Non assume una forma specifica per la funzione target
2. **Flessibilità locale**: Si adatta alle caratteristiche locali dei dati
3. **Robustezza**: Meno sensibile agli outliers globali
4. **Interpretabilità**: Ogni previsione ha un modello lineare locale interpretabile

### **5.2. Svantaggi**

1. **Computazione costosa**: Richiede riaddestramento per ogni query
2. **Maledizione della dimensionalità**: Performance degrada in alta dimensionalità
3. **Scelta del bandwidth**: Richiede tuning del parametro $\tau$
4. **Memoria**: Deve memorizzare tutto il training set

### **5.3. Bias-Variance Tradeoff**

Il parametro $\tau$ (bandwidth) controlla il tradeoff bias-varianza:

- **$\tau$ piccolo** (finestra stretta):
  - **Alto bias**: Il modello potrebbe non catturare il pattern locale
  - **Bassa varianza**: Più sensibile ai cambiamenti nei dati
  - **Overfitting**: Modello molto "wiggly"

- **$\tau$ grande** (finestra larga):
  - **Basso bias**: Si avvicina alla regressione lineare globale
  - **Alta varianza**: Più stabile rispetto ai cambiamenti nei dati
  - **Underfitting**: Modello troppo smooth

### **5.4. Scelta Ottimale del Bandwidth**

La scelta di $\tau$ può essere effettuata attraverso:

1. **Cross-validation**: Minimizzare l'errore di validazione
2. **Leave-one-out CV**: Particolarmente efficiente per LWLR
3. **Regola del pollice**: $\tau \approx \frac{\text{range dei dati}}{5}$
4. **Grid search**: Testare diversi valori e scegliere il migliore

## **6. Estensioni e Varianti**

### **6.1. Kernel Alternativi**

Oltre al kernel gaussiano, si possono utilizzare:

1. **Tricube Kernel**:
   $$
   w(u) = \begin{cases}
   (1 - |u|^3)^3 & \text{se } |u| \leq 1 \\
   0 & \text{altrimenti}
   \end{cases}
   $$

2. **Epanechnikov Kernel**:
   $$
   w(u) = \begin{cases}
   \frac{3}{4}(1 - u^2) & \text{se } |u| \leq 1 \\
   0 & \text{altrimenti}
   \end{cases}
   $$

### **6.2. Bandwidth Adattivo**

Il bandwidth può variare localmente:
$$
\tau_i = \tau_0 \cdot k\text{-th nearest neighbor distance}
$$

### **6.3. LWLR con Regolarizzazione**

Si può aggiungere regolarizzazione Ridge:
$$
\min_{\boldsymbol{\theta}} (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^T \mathbf{W}(\mathbf{x}_q) (\mathbf{y} - \mathbf{X}\boldsymbol{\theta}) + \lambda \|\boldsymbol{\theta}\|_2^2
$$

La soluzione diventa:
$
\boldsymbol{\theta}^* = (\mathbf{X}^T \mathbf{W} \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{y}
$

## **7. Confronto dei Metodi**

### **7.1. Confronto SGD vs BGD vs Forma Chiusa**

| Aspetto | Forma Chiusa | SGD | BGD |
|---------|--------------|-----|-----|
| **Convergenza** | Istantanea | Stocastica | Deterministica |
| **Velocità per epoca** | N/A | Più veloce | Più lenta |
| **Memoria** | Alta ($O(m^2)$ per $\mathbf{W}$) | Minore | Intermedia |
| **Parallelizzazione** | Limitata | Difficile | Facile |
| **Stabilità numerica** | Dipende da cond($\mathbf{X}^T\mathbf{W}\mathbf{X}$) | Robusta | Intermedia |
| **Controllo ottimizzazione** | Nessuno | Massimo | Intermedio |
| **Riproducibilità** | Perfetta | Richiede seed | Perfetta |
| **Scalabilità** | $O(d^3)$ | $O(mde)$ | $O(mde)$ |

### **7.2. Quando Usare Ogni Metodo**

**Forma Chiusa:**
- Dataset piccolo-medio (< 1000 punti)
- Matrice ben condizionata
- Serve la soluzione esatta
- Riproducibilità è critica

**SGD:**
- Dataset grande
- Problemi di condizionamento numerico
- Serve controllo fine dell'ottimizzazione
- Memoria limitata

**BGD:**
- Compromesso tra forma chiusa e SGD
- Convergenza stabile prioritaria
- Dataset di medie dimensioni

## **8. Esempio Comparativo Pratico**

Per illustrare le differenze tra le varie implementazioni di LWLR e confrontarle con la regressione lineare classica, consideriamo un esempio con dati sintetici che presentano pattern non lineari.

### **8.1. Dataset Sintetico**

Generiamo un dataset con una relazione non lineare:

```python
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)
data_points = 50
X = torch.linspace(-5, 5, data_points).unsqueeze(1)  # (50, 1)

# Trend lineare + componente sinusoidale
y_true = 3.5 * X.squeeze() + 1.5
amplitude = 4.0
frequency = 2.0
y = y_true + amplitude * torch.sin(frequency * X.squeeze())  # Pattern non lineare
```

### **8.2. Confronto dei Metodi**

Confrontiamo quattro approcci:

1. **Regressione Lineare Classica (forma chiusa)**: Trova un'unica retta per tutti i dati
2. **Regressione Lineare Classica (SGD)**: Stessa retta ma ottenuta iterativamente
3. **LWLR (forma chiusa)**: Adattamento locale ottimale
4. **LWLR (SGD)**: Adattamento locale iterativo

```python
# Dataset di test
X_test = torch.linspace(-5, 5, 25).unsqueeze(1)
y_true_test = 3.5 * X_test.squeeze() + 1.5
y_test = y_true_test + amplitude * torch.sin(frequency * X_test.squeeze())

# Predizioni LWLR (forma chiusa)
y_preds_lwlr_cf = []
for x_q in X_test:
    y_pred, _, _ = lwlr_CF(x_q.unsqueeze(0), X, y, t=0.5)
    y_preds_lwlr_cf.append(y_pred)

# Predizioni LWLR (SGD) 
y_preds_lwlr_sgd = []
for x_q in X_test:
    y_pred, _, _ = lwlr_SGD(x_q.unsqueeze(0), X, y, t=0.5, lr=0.1, epochs=25)
    y_preds_lwlr_sgd.append(y_pred)
```

### **8.3. Visualizzazione dei Risultati**

Un tipico confronto mostra che:

- **Regressione Lineare**: Produce una singola retta che rappresenta il trend medio globale, ma non cattura la variabilità locale
- **LWLR (CF)**: Si adatta perfettamente ai pattern locali, seguendo la curvatura dei dati
- **LWLR (SGD)**: Produce risultati molto simili alla forma chiusa, ma con leggere variazioni dovute alla natura stocastica dell'ottimizzazione

### **8.4. Analisi dell'Errore**

```python
# Calcolo MSE per ciascun metodo
mse_linear = torch.mean((y_pred_linear - y_test)**2)
mse_lwlr_cf = torch.mean((torch.tensor(y_preds_lwlr_cf) - y_test)**2) 
mse_lwlr_sgd = torch.mean((torch.tensor(y_preds_lwlr_sgd) - y_test)**2)

print(f"MSE Linear Regression: {mse_linear:.4f}")
print(f"MSE LWLR (Closed Form): {mse_lwlr_cf:.4f}")  
print(f"MSE LWLR (SGD): {mse_lwlr_sgd:.4f}")
```

**Risultati tipici:**
- **Linear Regression**: MSE più alto, non cattura la non linearità
- **LWLR (CF)**: MSE più basso, adattamento ottimale
- **LWLR (SGD)**: MSE simile alla forma chiusa, dipende da learning rate ed epoche

### **8.5. Effetto del Bandwidth**

Il parametro $\tau$ ha un impatto significativo sulle prestazioni:

```python
# Test con diversi valori di bandwidth
bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0]
mse_values = []

for tau in bandwidths:
    y_preds = []
    for x_q in X_test:
        y_pred, _, _ = lwlr_CF(x_q.unsqueeze(0), X, y, t=tau)
        y_preds.append(y_pred)
    
    mse = torch.mean((torch.tensor(y_preds) - y_test)**2)
    mse_values.append(mse.item())
    
# Il bandwidth ottimale minimizza l'MSE sul test set
optimal_tau = bandwidths[torch.argmin(torch.tensor(mse_values))]
```

**Osservazioni:**
- **$\tau$ troppo piccolo** ($< 0.5$): Overfitting, modello troppo "nervoso"
- **$\tau$ troppo grande** ($> 2.0$): Underfitting, si avvicina alla regressione lineare
- **$\tau$ ottimale** ($\approx 0.5-1.0$): Bilancia bias e varianza

### **8.6. Considerazioni Pratiche**

1. **Tempo di calcolo**: La forma chiusa è più veloce per singole predizioni, ma SGD può essere parallelizzato
2. **Stabilità numerica**: SGD è più robusto quando la matrice è mal condizionata  
3. **Controllo fine**: SGD permette di monitorare la convergenza e implementare early stopping
4. **Riproducibilità**: La forma chiusa è deterministica, SGD richiede seed fissato

## **9. Complessità Computazionale**

### **9.1. Analisi della Complessità**

**Soluzione Analitica (Forma Chiusa):**
- **Calcolo pesi**: $O(md)$ 
- **Costruzione matrice $\mathbf{W}$**: $O(m^2)$
- **Prodotti matriciali**: $O(md^2 + d^3)$
- **Inversione matrice**: $O(d^3)$
- **Complessità totale per query**: $O(m^2 + md^2 + d^3)$
- **Spazio**: $O(m^2 + md)$

**SGD:**
- **Complessità per epoca**: $O(md)$
- **Complessità totale**: $O(mdE)$ dove $E$ è il numero di epoche
- **Spazio**: $O(md)$

**BGD:**
- **Complessità per epoca**: $O(md)$ 
- **Complessità totale**: $O(mdE)$ dove $E$ è il numero di epoche
- **Spazio**: $O(md)$

### **9.2. Scalabilità**

- **Forma chiusa**: Non scala bene per $m$ o $d$ grandi
- **SGD/BGD**: Scalano linearmente con $m$ e $d$

## **10. Applicazioni Pratiche**

LWLR è particolarmente utile in:

1. **Time series forecasting**: Pattern temporali che cambiano localmente
2. **Robotics**: Controllo adattivo e apprendimento di traiettorie
3. **Computer vision**: Interpolazione di immagini, tracking
4. **Bioinformatica**: Analisi di dati genomici con pattern locali
5. **Economia**: Modelli che si adattano a regimi economici diversi
6. **Meteorologia**: Previsioni che si adattano alle condizioni locali

## **11. Limitazioni e Considerazioni**

### **11.1. Maledizione della Dimensionalità**

In alta dimensionalità ($d > 10-20$):
- I punti diventano equidistanti
- Tutti i pesi tendono a essere simili
- Il concetto di "località" perde significato
- Performance si degrada rapidamente

### **11.2. Problemi Numerici**

1. **Matrice singolare**: Quando $\mathbf{X}^T\mathbf{W}\mathbf{X}$ non è invertibile
2. **Condizionamento numerico**: Alto numero di condizione causa instabilità
3. **Underflow**: Pesi estremamente piccoli possono causare problemi

### **11.3. Soluzioni**

- **Regolarizzazione**: Aggiungere $\lambda \mathbf{I}$ alla matrice
- **SVD**: Usare decomposizione SVD invece dell'inversione diretta
- **Thresholding**: Impostare soglia minima per i pesi

## **12. Conclusioni**

LWLR rappresenta un potente compromesso tra la semplicità della regressione lineare e la flessibilità dei metodi non parametrici. La sua capacità di adattarsi localmente ai dati lo rende particolarmente adatto per problemi dove la relazione target varia spazialmente o temporalmente.

### **12.1. Linee Guida per la Scelta del Metodo**

**Usa la forma chiusa quando:**
- Dataset piccolo-medio (< 1000 punti)
- Dimensionalità bassa ($d < 20$)
- Matrice ben condizionata
- Serve la soluzione esatta
- Riproducibilità è critica

**Usa SGD quando:**
- Dataset grande ($m > 10000$)
- Problemi di condizionamento numerico
- Memoria limitata
- Serve controllo fine dell'ottimizzazione
- Possibilità di parallelizzazione

**Usa BGD quando:**
- Serve compromesso tra forma chiusa e SGD
- Convergenza stabile è prioritaria
- Dataset di medie dimensioni
- Si vuole evitare il rumore del SGD

### **12.2. Considerazioni Finali**

Le tre implementazioni offrono diversi vantaggi:
- **Forma chiusa**: Soluzione esatta e veloce per problemi piccoli
- **SGD**: Scalabilità e robustezza numerica
- **BGD**: Stabilità e controllo della convergenza

La scelta dipende dalle caratteristiche del problema: dimensionalità dei dati, dimensione del dataset, requisiti di accuratezza e risorse computazionali disponibili.

LWLR rimane uno strumento fondamentale nell'arsenal del machine learning, particolarmente efficace quando i pattern nei dati variano localmente e si necessita di un approccio che bilanci interpretabilità e flessibilità.
