# Differenziazione Automatica - Automatic Differentiation

## Definizione e Concetti Fondamentali

La **differenziazione automatica** (Automatic Differentiation, AD) è una tecnica computazionale per calcolare derivate di funzioni definite attraverso programmi informatici con **precisione numerica esatta** (fino alla precisione della macchina).

**Definizione formale**: Data una funzione $f: \mathbb{R}^n \to \mathbb{R}^m$ implementata come sequenza di operazioni elementari, l'AD calcola il gradiente $\nabla f$ o la matrice Jacobiana $J_f$ senza approssimazioni numeriche.

### Differenze con Altri Metodi

| Metodo | Precisione | Costo Computazionale | Limitazioni |
|--------|------------|---------------------|-------------|
| **Differenze Finite** | $O(h)$ o $O(h^2)$ | $O(n) \cdot \text{costo}(f)$ | Errori di troncamento e cancellazione |
| **Differenziazione Simbolica** | Esatta | Esplosione espressioni | Non pratica per funzioni complesse |
| **Differenziazione Automatica** | Esatta (precisione macchina) | $O(1) - O(n) \cdot \text{costo}(f)$ | Richiede implementazione speciale |

## Teorema Fondamentale dell'AD

**Teorema**: Ogni programma che calcola una funzione numerica può essere decomposto in una sequenza di operazioni elementari. Se conosciamo le derivate di queste operazioni elementari, possiamo calcolare la derivata dell'intera funzione applicando ripetutamente la **[[Chain Rule]]**.

**Formalizzazione**: Se $f = f_m \circ f_{m-1} \circ \cdots \circ f_1$, allora:
$$\frac{df}{dx} = \frac{df_m}{dx_{m-1}} \cdot \frac{df_{m-1}}{dx_{m-2}} \cdot \ldots \cdot \frac{df_1}{dx}$$

## Due Modalità: Forward e Reverse

### Forward Mode (Tangent Linear Mode)

Il **forward mode** calcola derivate propagando **perturbazioni infinitesimali** dalla variabile indipendente verso l'output.

**Principio**: Ogni variabile $v$ è rappresentata come coppia $(v, \dot{v})$ dove $\dot{v} = \frac{\partial v}{\partial x}$.

**Regole di propagazione**:
- $c + v = (c + v_0, \dot{v})$ (costante)
- $u + v = (u_0 + v_0, \dot{u} + \dot{v})$ (addizione)
- $u \cdot v = (u_0 \cdot v_0, \dot{u} \cdot v_0 + u_0 \cdot \dot{v})$ (prodotto)
- $\sin(v) = (\sin(v_0), \cos(v_0) \cdot \dot{v})$ (funzione trascendente)

**Esempio**: Calcolare $\frac{d}{dx}[x^2 + \sin(x)]$ in $x = \pi/4$

$$

\begin{align*}
\textbf{Inizializzazione:} \quad
x   &= \left(\tfrac{\pi}{4}, 1\right) 
      && \text{(valore e derivata seme).}\\[2ex]

\textbf{Primo passo:} \quad
v_{1} &= x \cdot x \\[0.5ex]
      &= \left(\tfrac{\pi}{4}, 1\right) \cdot \left(\tfrac{\pi}{4}, 1\right) \\[0.5ex]
      &= \Bigl(\tfrac{\pi}{4}\cdot\tfrac{\pi}{4}, \;\tfrac{\pi}{4}\cdot 1 + \tfrac{\pi}{4}\cdot 1\Bigr) \\[0.5ex]
      &= \left(\tfrac{\pi^{2}}{16}, \tfrac{\pi}{2}\right). \\[2ex]

\textbf{Secondo passo:} \quad
v_{2} &= \sin(x) \\[0.5ex]
      &= \left(\sin\!\left(\tfrac{\pi}{4}\right), \;\cos\!\left(\tfrac{\pi}{4}\right)\cdot 1\right) \\[0.5ex]
      &= \left(\tfrac{\sqrt{2}}{2}, \tfrac{\sqrt{2}}{2}\right). \\[2ex]

\textbf{Combinazione finale:} \quad
f     &= v_{1} + v_{2} \\[0.5ex]
      &= \left(\tfrac{\pi^{2}}{16} + \tfrac{\sqrt{2}}{2}, \;\tfrac{\pi}{2} + \tfrac{\sqrt{2}}{2}\right). \\[2ex]

\textbf{Risultato:} \quad
f'\!\left(\tfrac{\pi}{4}\right) &= \tfrac{\pi}{2} + \tfrac{\sqrt{2}}{2}.
\end{align*}


$$

### Reverse Mode (Backpropagation)

Il **reverse mode** calcola derivate propagando **moltiplicatori di Lagrange** dall'output verso le variabili indipendenti.

**Principio**: 
1. **Forward pass**: Calcola tutti i valori intermedi
2. **Backward pass**: Propaga le derivate usando la chain rule

**Notazione**: $\bar{v} = \frac{\partial f}{\partial v}$ (derivata parziale di $f$ rispetto alla variabile intermedia $v$)

**Regole di backpropagation**:
- Se $w = u + v$, allora $\bar{u} += \bar{w}$ e $\bar{v} += \bar{w}$
- Se $w = u \cdot v$, allora $\bar{u} += v \cdot \bar{w}$ e $\bar{v} += u \cdot \bar{w}$
- Se $w = \sin(u)$, allora $\bar{u} += \cos(u) \cdot \bar{w}$
**Esempio**: Calcolare $\frac{d}{dx}[x^2 + \sin(x)]$ in $x = \pi/4$  

#### Forward pass
$$
x = \tfrac{\pi}{4}, 
\qquad
v_{1} = x^2 = \tfrac{\pi^2}{16}, 
\qquad
v_{2} = \sin(x) = \tfrac{\sqrt{2}}{2},
\qquad
f = v_{1} + v_{2} = \tfrac{\pi^2}{16} + \tfrac{\sqrt{2}}{2}.
$$

---

#### Backward pass

- Inizializzazione sull’output:
$$
\bar{f} = 1
$$

- Per la somma $f = v_{1} + v_{2}$:
$$
\bar{v}_{1} += \bar{f} = 1,
\qquad
\bar{v}_{2} += \bar{f} = 1.
$$

- Per il quadrato $v_{1} = x^2$:
$$
\bar{x} += \frac{\partial v_{1}}{\partial x} \cdot \bar{v}_{1} 
= (2x) \cdot 1 
= 2 \cdot \tfrac{\pi}{4} = \tfrac{\pi}{2}.
$$

- Per il seno $v_{2} = \sin(x)$:
$$
\bar{x} += \frac{\partial v_{2}}{\partial x} \cdot \bar{v}_{2} 
= \cos(x) \cdot 1 
= \cos\!\left(\tfrac{\pi}{4}\right) 
= \tfrac{\sqrt{2}}{2}.
$$

---

#### Risultato
$$
\frac{df}{dx}\Big|_{x=\pi/4} = \bar{x} = \tfrac{\pi}{2} + \tfrac{\sqrt{2}}{2}.
$$


### Confronto Computazionale

**Forward Mode**: 
- Costo: $O(n)$ per calcolare $\nabla f$ quando $f: \mathbb{R}^n \to \mathbb{R}$
- Ottimale quando $n$ piccolo (poche variabili input)

**Reverse Mode**: 
- Costo: $O(m)$ per calcolare $\nabla f$ quando $f: \mathbb{R}^n \to \mathbb{R}^m$
- Ottimale quando $m$ piccolo (pochi output)
- **Machine Learning**: $m = 1$ (loss function) → reverse mode dominante

## Implementazione Forward Mode

```python
import numpy as np
import math

class Dual:
    """Numero duale per forward mode AD"""
    def __init__(self, value, derivative=0.0):
        self.val = value
        self.der = derivative
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Dual(self.val + other, self.der)
        return Dual(self.val + other.val, self.der + other.der)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Dual(self.val * other, self.der * other)
        # Regola del prodotto: (uv)' = u'v + uv'
        return Dual(self.val * other.val, 
                   self.der * other.val + self.val * other.der)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            # (u^n)' = n·u^(n-1)·u'
            return Dual(self.val ** other, 
                       other * (self.val ** (other-1)) * self.der)
        raise NotImplementedError("Dual^Dual non implementato")
    
    def sin(self):
        return Dual(math.sin(self.val), math.cos(self.val) * self.der)
    
    def cos(self):
        return Dual(math.cos(self.val), -math.sin(self.val) * self.der)
    
    def exp(self):
        exp_val = math.exp(self.val)
        return Dual(exp_val, exp_val * self.der)
    
    def __repr__(self):
        return f"Dual({self.val:.6f}, {self.der:.6f})"

# Funzioni helper
def sin(x):
    return x.sin() if isinstance(x, Dual) else math.sin(x)

def cos(x):
    return x.cos() if isinstance(x, Dual) else math.cos(x)

def exp(x):
    return x.exp() if isinstance(x, Dual) else math.exp(x)

# Esempio di utilizzo
def f(x):
    """f(x) = x² + sin(x) + exp(x)"""
    return x**2 + sin(x) + exp(x)

# Calcolo della derivata in x = 1.0
x = Dual(1.0, 1.0)  # seme: df/dx
result = f(x)
print(f"f(1) = {result.val:.6f}")
print(f"f'(1) = {result.der:.6f}")

# Verifica analitica: f'(x) = 2x + cos(x) + exp(x)
analytical = 2*1.0 + math.cos(1.0) + math.exp(1.0)
print(f"Analitico: f'(1) = {analytical:.6f}")
```

## Implementazione Reverse Mode

```python
import numpy as np

class Variable:
    """Variabile per reverse mode AD"""
    
    # Contatore globale per ID univoci
    _id_counter = 0
    
    def __init__(self, value, name=None):
        self.value = np.array(value) if not isinstance(value, np.ndarray) else value
        self.grad = None
        self.backward_fn = None
        self.children = []
        self.name = name
        
        # ID univoco per debug
        Variable._id_counter += 1
        self.id = Variable._id_counter
    
    def backward(self, gradient=None):
        """Backpropagation"""
        if gradient is None:
            gradient = np.ones_like(self.value)
        
        if self.grad is None:
            self.grad = np.zeros_like(self.value)
        
        self.grad += gradient
        
        # Propaga ai figli
        if self.backward_fn is not None:
            self.backward_fn(gradient)
    
    def zero_grad(self):
        """Reset gradienti"""
        self.grad = None
    
    def __add__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)
        
        result = Variable(self.value + other.value)
        
        def backward_fn(grad):
            self.backward(grad)
            other.backward(grad)
        
        result.backward_fn = backward_fn
        result.children = [self, other]
        return result
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)
        
        result = Variable(self.value * other.value)
        
        def backward_fn(grad):
            # ∂(uv)/∂u = v, ∂(uv)/∂v = u
            self.backward(grad * other.value)
            other.backward(grad * self.value)
        
        result.backward_fn = backward_fn
        result.children = [self, other]
        return result
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        
        result = Variable(self.value ** other)
        
        def backward_fn(grad):
            # ∂(u^n)/∂u = n·u^(n-1)
            self.backward(grad * other * (self.value ** (other - 1)))
        
        result.backward_fn = backward_fn
        result.children = [self]
        return result
    
    def sin(self):
        result = Variable(np.sin(self.value))
        
        def backward_fn(grad):
            # ∂sin(u)/∂u = cos(u)
            self.backward(grad * np.cos(self.value))
        
        result.backward_fn = backward_fn
        result.children = [self]
        return result
    
    def exp(self):
        exp_val = np.exp(self.value)
        result = Variable(exp_val)
        
        def backward_fn(grad):
            # ∂exp(u)/∂u = exp(u)
            self.backward(grad * exp_val)
        
        result.backward_fn = backward_fn
        result.children = [self]
        return result
    
    def __repr__(self):
        return f"Variable({self.value}, grad={self.grad}, id={self.id})"

# Funzioni helper
def sin(x):
    return x.sin()

def exp(x):
    return x.exp()

# Esempio: f(x,y) = x²y + sin(x) + exp(y)
def f_multivariate(x, y):
    return x**2 * y + sin(x) + exp(y)

# Calcolo gradienti
x = Variable(2.0, name='x')
y = Variable(1.0, name='y')

result = f_multivariate(x, y)
result.backward()

print(f"f(2,1) = {result.value:.6f}")
print(f"∂f/∂x = {x.grad:.6f}")
print(f"∂f/∂y = {y.grad:.6f}")

# Verifica analitica
# ∂f/∂x = 2xy + cos(x), ∂f/∂y = x² + exp(y)
analytical_dx = 2*2.0*1.0 + np.cos(2.0)  # = 4 + cos(2)
analytical_dy = 2.0**2 + np.exp(1.0)     # = 4 + e
print(f"Analitico: ∂f/∂x = {analytical_dx:.6f}")
print(f"Analitico: ∂f/∂y = {analytical_dy:.6f}")
```

## Computational Graph e Topological Sort

Il **computational graph** è un grafo diretto aciclico (DAG) che rappresenta le operazioni:

- **Nodi**: Variabili e operazioni
- **Archi**: Dipendenze tra operazioni

Per il reverse mode, è essenziale attraversare il grafo in **ordine topologico inverso**.

<img src="https://upload.wikimedia.org/wikipedia/commons/5/5d/Computational_graph.svg" alt="Computational Graph" style="width: 50%; height: 50%;">

## Analisi della Complessità

### Forward Mode
- **Spazio**: $O(n)$ per memorizzare derivate parziali
- **Tempo**: $O(n \cdot \text{costo}(f))$ per calcolare gradiente completo
- **Ottimale per**: $n \ll m$ (poche variabili input, molti output)

### Reverse Mode
- **Spazio**: $O(\text{grafo})$ per memorizzare computational graph
- **Tempo**: $O(m + \text{costo}(f))$ per calcolare gradiente
- **Ottimale per**: $m \ll n$ (molte variabili input, pochi output)

### Esempio Comparativo: Timing

```python
import time
import numpy as np

def timing_comparison():
    """Confronto prestazioni Forward vs Reverse mode"""
    
    # Funzione di test: f(x) = sum(x[i]² * sin(x[i]))
    def test_function_forward(x_dual):
        result = Dual(0.0, 0.0)
        for xi in x_dual:
            result = result + xi**2 * sin(xi)
        return result
    
    def test_function_reverse(x_vars):
        result = Variable(0.0)
        for xi in x_vars:
            result = result + xi**2 * sin(xi)
        return result
    
    dimensions = [10, 50, 100, 500]
    results = {'forward': [], 'reverse': []}
    
    for n in dimensions:
        x_val = np.random.randn(n)
        
        # Forward mode timing
        start = time.time()
        for i in range(n):  # Calcola ogni componente del gradiente
            x_dual = [Dual(x_val[j], 1.0 if j == i else 0.0) for j in range(n)]
            result = test_function_forward(x_dual)
        forward_time = time.time() - start
        
        # Reverse mode timing
        start = time.time()
        x_vars = [Variable(x_val[i]) for i in range(n)]
        result = test_function_reverse(x_vars)
        result.backward()
        reverse_time = time.time() - start
        
        results['forward'].append(forward_time)
        results['reverse'].append(reverse_time)
        
        print(f"n={n:3d}: Forward={forward_time:.4f}s, Reverse={reverse_time:.4f}s")
    
    # Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.loglog(dimensions, results['forward'], 'bo-', label='Forward Mode', linewidth=2)
    plt.loglog(dimensions, results['reverse'], 'ro-', label='Reverse Mode', linewidth=2)
    plt.xlabel('Numero di variabili (n)')
    plt.ylabel('Tempo (secondi)')
    plt.title('Confronto Prestazioni: Forward vs Reverse Mode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Esegui solo se si vuole testare le prestazioni
# timing_comparison()
```

## Teoremi di Complessità

### Teorema 1 (Ottimalità Forward Mode)
Per funzioni $f: \mathbb{R}^n \to \mathbb{R}^m$ con $n \leq m$, il forward mode calcola la matrice Jacobiana completa in tempo $O(n \cdot \text{costo}(f))$, che è ottimale.

### Teorema 2 (Ottimalità Reverse Mode)  
Per funzioni $f: \mathbb{R}^n \to \mathbb{R}^m$ con $m \leq n$, il reverse mode calcola la matrice Jacobiana completa in tempo $O(m \cdot \text{costo}(f))$, che è ottimale.

### Corollario (Machine Learning)
Nel deep learning, tipicamente $m = 1$ (loss function), quindi il reverse mode è asintoticamente superiore: $O(\text{costo}(f))$ vs $O(n \cdot \text{costo}(f))$.

## Differenziazione di Ordine Superiore

L'AD può essere esteso per calcolare **derivate di ordine superiore**:

### Forward-on-Forward (Hessiano)
Per calcolare $\frac{\partial^2 f}{\partial x_i \partial x_j}$:

```python
class HyperDual:
    """Numero iper-duale per derivate seconde"""
    def __init__(self, f, f_x=0, f_y=0, f_xy=0):
        self.f = f      # f(x,y)
        self.f_x = f_x  # ∂f/∂x
        self.f_y = f_y  # ∂f/∂y
        self.f_xy = f_xy # ∂²f/∂x∂y
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return HyperDual(self.f + other, self.f_x, self.f_y, self.f_xy)
        return HyperDual(
            self.f + other.f,
            self.f_x + other.f_x,
            self.f_y + other.f_y,
            self.f_xy + other.f_xy
        )
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return HyperDual(
                self.f * other,
                self.f_x * other,
                self.f_y * other,
                self.f_xy * other
            )
        
        # (uv)'' = u''v + 2u'v' + uv''
        return HyperDual(
            self.f * other.f,
            self.f_x * other.f + self.f * other.f_x,
            self.f_y * other.f + self.f * other.f_y,
            self.f_xy * other.f + self.f_x * other.f_y + 
            self.f_y * other.f_x + self.f * other.f_xy
        )

# Esempio: Hessiano di f(x,y) = x²y
x = HyperDual(2.0, 1.0, 0.0, 0.0)  # seme per ∂/∂x
y = HyperDual(3.0, 0.0, 1.0, 0.0)  # seme per ∂/∂y

result = x * x * y  # f(x,y) = x²y

print(f"f(2,3) = {result.f}")
print(f"∂f/∂x = {result.f_x}")    # 2xy = 12
print(f"∂f/∂y = {result.f_y}")    # x² = 4  
print(f"∂²f/∂x∂y = {result.f_xy}") # 2x = 4
```

## Applicazioni e Estensioni

### 1. Ottimizzazione
L'AD è fondamentale per algoritmi di ottimizzazione che richiedono gradienti esatti:
- **Newton-Raphson**: Richiede Hessiano
- **BFGS**: Beneficia di gradienti precisi
- **Constraint Optimization**: Jacobiani di vincoli

### 2. Risoluzione di EDO/EDP
Per sistemi differenziali con parametri:
$$\frac{dy}{dt} = f(y, \theta, t)$$

L'AD calcola $\frac{\partial y}{\partial \theta}$ (sensibilità ai parametri).

### 3. Probabilità e Statistica
- **Maximum Likelihood**: Gradiente della log-likelihood
- **Bayesian Inference**: MCMC con gradienti (HMC, NUTS)
- **Variational Inference**: Ottimizzazione di bound variazionali

### 4. Fisica Computazionale
- **Simulazioni Molecular Dynamics**: Forze da potenziali
- **Finite Element Methods**: Assemblaggio automatico di matrici
- **Optimal Control**: Condizioni di ottimalità

## Limitazioni e Considerazioni

### 1. Memoria
Il reverse mode richiede memorizzazione dell'intero computational graph:
- **Checkpointing**: Trade-off tempo/memoria
- **Recompute**: Ricalcola invece di memorizzare

### 2. Funzioni Non-Differenziabili
```python
def non_smooth_function(x):
    # |x| non è differenziabile in x=0
    return abs(x.value) if hasattr(x, 'value') else abs(x)
```

L'AD può dare risultati inconsistenti per funzioni non differenziabili.

### 3. Controllo di Flusso
```python
def conditional_function(x):
    if x > 0:
        return x**2
    else:
        return -x**3
```

Il controllo di flusso data-dependent complica l'AD.

## Note Storiche e Sviluppi

### Origini (1960s-1970s)
- **Wengert (1964)**: Prima formulazione dell'AD
- **Rall (1981)**: Teoria matematica rigorosa

### Rinascimento (1980s-1990s)  
- **ADIFOR, ADIC**: Primi compilatori AD
- **Griewank**: Teoria della complessità

### Era Moderna (2000s-present)
- **Machine Learning**: Esplosione dell'interesse
- **JAX, Autograd**: Librerie moderne
- **Differentiable Programming**: Nuovo paradigma

---

## Tags
#automatic-differentiation #calcolo #gradiente #machine-learning #ottimizzazione #computational-graph

## Collegamenti
- [[Chain Rule]]
- [[Gradiente e Ottimizzazione]]
- [[Backpropagation]]
- [[Computational Graphs]]
- [[Numerical Methods]]
