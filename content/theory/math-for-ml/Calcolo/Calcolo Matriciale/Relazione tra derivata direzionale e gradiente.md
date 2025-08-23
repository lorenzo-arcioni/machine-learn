# Relazione tra derivata direzionale e gradiente (caso multivariato)

In questa nota esploriamo in dettaglio la relazione tra **derivata direzionale** e **gradiente** per funzioni multivariate $f:\mathbb{R}^n \to \mathbb{R}$. L'obiettivo è fornire una comprensione sia concettuale che formale di come il gradiente descriva le variazioni di $f$ lungo qualsiasi direzione nello spazio, e come questa informazione sia fondamentale nelle applicazioni di ottimizzazione, analisi numerica e geometria differenziale.

## 1. Derivata direzionale in $\mathbb{R}^n$

Sia $f:\mathbb{R}^n \to \mathbb{R}$ differenziabile e sia $\mathbf{v} = (v_1, v_2, \dots, v_n) \in \mathbb{R}^n$ un vettore.  
La **derivata direzionale** di $f$ in un punto $P_0 \in \mathbb{R}^n$ lungo $\mathbf{v}$ è:

$$
\frac{df}{d\mathbf{v}}(P_0) = \lim_{t \to 0} \frac{f(P_0 + t \mathbf{v}) - f(P_0)}{t},
$$

dove $t$ è un parametro scalare che misura quanto ci spostiamo lungo la direzione del vettore $\mathbf v$.

Intuitivamente, misura **come cambia $f$ muovendosi da $P_0$ nella direzione di $\mathbf{v}$**.

## 2. Espressione tramite il gradiente

Per funzioni differenziabili, il **differenziale totale** di $f$ è:

$$
df = \sum_{i=1}^n \frac{\partial f}{\partial x_i} dx_i
$$

Considerciamo che $dt$ rappresenta un incremento infinitesimale di $t$.

Se ci spostiamo lungo $\mathbf{v}$, abbiamo $dx_i = v_i dt$, quindi:

$$
df = \sum_{i=1}^n \frac{\partial f}{\partial x_i} v_i dt
\quad \implies \quad
\frac{df}{dt} = \sum_{i=1}^n \frac{\partial f}{\partial x_i} v_i
$$

Valutando in $P_0$:

$$
\frac{df}{d\mathbf{v}}(P_0) = \sum_{i=1}^n \frac{\partial f}{\partial x_i}(P_0) v_i
$$

Definendo il **gradiente** in $P_0$ come

$$
\nabla f(P_0) = \begin{pmatrix} \frac{\partial f}{\partial x_1}(P_0) \\[1mm] \vdots \\[1mm] \frac{\partial f}{\partial x_n}(P_0) \end{pmatrix},
$$

otteniamo la formula compatta:

$$
\frac{df}{d\mathbf{v}}(P_0) = \langle \nabla f(P_0), \mathbf{v} \rangle
$$

dove $\langle \cdot, \cdot \rangle$ è il **prodotto scalare standard** in $\mathbb{R}^n$.

## 3. Interpretazione geometrica

- $\nabla f(P_0)$ indica la **direzione di massima crescita di $f$**.  
- La derivata direzionale misura **quanto cresce $f$ nella direzione $\mathbf{v}$**.  
- Massima derivata: $\mathbf{v} \parallel \nabla f(P_0)$
- Minima derivata: $\mathbf{v} \parallel -\nabla f(P_0)$
- Derivata nulla: $\mathbf{v} \perp \nabla f(P_0)$ → vettore tangente a una **curva (o superficie) di livello**.


## 4. Curve/superfici di livello

Definiamo la **superficie di livello** in $P_0$ come:

$$
\mathcal{L}_{f,c} = \{ \mathbf{x} \in \mathbb{R}^n \mid f(\mathbf{x}) = f(P_0) = c \}
$$

Se $\mathbf{v}$ è **tangente a $\mathcal{L}_{f,c}$**:

$$
\frac{df}{d\mathbf{v}}(P_0) = \langle \nabla f(P_0), \mathbf{v} \rangle = 0
$$

✅ **Conclusione generale:** ogni vettore tangente alla superficie di livello è ortogonale al gradiente.

## 5. Scelta di un vettore tangente concreto

Sia $\nabla f(P_0) = (g_1, \dots, g_n)$.  
Per trovare $\mathbf{v} \neq 0$ tangente:

$$
\langle \nabla f(P_0), \mathbf{v} \rangle = \sum_{i=1}^n g_i v_i = 0
$$

- Equazione lineare con $n$ incognite → **infinite soluzioni**.  
- Possiamo fissare $n-1$ componenti arbitrariamente e risolvere l’ultima.  
- Esempio in $n=3$:  
  $\nabla f(P_0) = (1,2,3)$ e scegliamo $v_1 = 1, v_2 = 0$, allora:

$$
1\cdot 1 + 2\cdot 0 + 3 v_3 = 0 \implies v_3 = -\frac{1}{3}
$$

Otteniamo $\mathbf{v} = (1,0,-1/3)$ tangente alla superficie di livello in $P_0$.

### 6. Sintesi concettuale

- **Derivata direzionale** lungo $\mathbf{v}$ in $P_0$:

$$
\frac{df}{d\mathbf{v}}(P_0) = \langle \nabla f(P_0), \mathbf{v} \rangle
$$

- **Gradiente**: direzione di massima crescita, modulo della derivata massima:

$$
\max_{\|\mathbf{v}\|=1} \frac{df}{d\mathbf{v}}(P_0) 
= \max_{\|\mathbf{v}\|=1} \langle \nabla f(P_0), \mathbf{v} \rangle 
= \|\nabla f(P_0)\| \cdot \max_{\|\mathbf{v}\|=1} \cos \theta 
= \|\nabla f(P_0)\| \cdot 1 
= \|\nabla f(P_0)\|
$$

- **Minima derivata**: direzione opposta al gradiente:

$$
\min_{\|\mathbf{v}\|=1} \frac{df}{d\mathbf{v}} = -\|\nabla f(P_0)\|, \quad 
\mathbf{v} = -\frac{\nabla f(P_0)}{\|\nabla f(P_0)\|}
$$

- **Vettori tangenti a curve/superfici di livello**: ortogonali al gradiente:

$$
\nabla f(P_0) \cdot \mathbf{v} = 0
$$

- **Generalizzazione a $n$ dimensioni**: scelta libera di $n-1$ componenti → infinite direzioni tangenti possibili.

> Questi concetti sono fondamentali nell'**ottimizzazione multivariata** e nello studio delle **superfici di livello**.
