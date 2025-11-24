# Metriche di Valutazione per Classificazione in Machine Learning

## 1. Introduzione

La valutazione di modelli di classificazione richiede metriche specifiche che quantifichino la qualità delle predizioni. Questo documento presenta una trattazione completa e rigorosa delle principali metriche utilizzate nel machine learning, con particolare attenzione alle differenze rispetto alle metriche biometriche.

## 2. Matrice di Confusione

La **matrice di confusione** è la base per calcolare tutte le metriche di classificazione. Per un problema binario:

|                    | **Predetto Positivo (P)** | **Predetto Negativo (N)** |
|--------------------|---------------------------|---------------------------|
| **Reale Positivo** | TP (True Positive)        | FN (False Negative)       |
| **Reale Negativo** | FP (False Positive)       | TN (True Negative)        |

### 2.1 Definizioni Rigorose

- **TP (True Positive)**: Istanze positive correttamente classificate come positive
  - In ML: Predizioni positive corrette
  - In Biometria: **Genuine Acceptance (GA)** o **Genuine Match (GM)**
  
- **TN (True Negative)**: Istanze negative correttamente classificate come negative
  - In ML: Predizioni negative corrette
  - In Biometria: **Genuine Rejection (GR)** o **Genuine Non-Match (GNM)**
  
- **FP (False Positive)**: Istanze negative erroneamente classificate come positive (Errore di Tipo II)
  - In ML: Falsi allarmi
  - In Biometria: **False Acceptance (FA)** o **False Match (FM)**
  
- **FN (False Negative)**: Istanze positive erroneamente classificate come negative (Errore di Tipo I)
  - In ML: Mancate rilevazioni
  - In Biometria: **False Rejection (FR)** o **False Non-Match (FNM)**

### 2.2 Visualizzazione della Matrice di Confusione

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matrice assoluta
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negativo', 'Positivo'])
disp.plot(cmap='Blues', ax=axes[0])
axes[0].set_title('Matrice di Confusione (Valori Assoluti)')

# Matrice normalizzata
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['Negativo', 'Positivo'])
disp_norm.plot(cmap='Greens', ax=axes[1], values_format='.2%')
axes[1].set_title('Matrice di Confusione (Normalizzata)')

plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
```

### 2.3 Interpretazione Probabilistica

Dato un sistema di classificazione con soglia $\tau$, possiamo definire due ipotesi:

- **H₀**: Le due istanze appartengono a classi diverse (impostor/negativo)
- **H₁**: Le due istanze appartengono alla stessa classe (genuine/positivo)

E due possibili decisioni:

- **D₀**: Classificare come negativo/diverso
- **D₁**: Classificare come positivo/stesso

Allora:
- **FPR** = P(D₁ | H₀ = true) = Probabilità di classificare come positivo quando è negativo
- **FNR** = P(D₀ | H₁ = true) = Probabilità di classificare come negativo quando è positivo

```python
# Visualizzazione distribuzioni genuine vs impostor
from scipy import stats

fig, ax = plt.subplots(figsize=(12, 6))

# Simula distribuzioni genuine e impostor
genuine_scores = np.random.beta(8, 2, 1000)
impostor_scores = np.random.beta(2, 8, 1000)

ax.hist(impostor_scores, bins=50, alpha=0.6, label='Impostor (H₀)', color='red', density=True)
ax.hist(genuine_scores, bins=50, alpha=0.6, label='Genuine (H₁)', color='green', density=True)

threshold = 0.5
ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Soglia τ = {threshold}')

# Area FPR e FNR
x_fp = np.linspace(threshold, 1, 100)
x_fn = np.linspace(0, threshold, 100)

ax.fill_between(x_fp, 0, stats.beta.pdf(x_fp, 2, 8), alpha=0.3, color='red', 
                label='FPR (Tipo II)')
ax.fill_between(x_fn, 0, stats.beta.pdf(x_fn, 8, 2), alpha=0.3, color='orange', 
                label='FNR (Tipo I)')

ax.set_xlabel('Score di Similarità')
ax.set_ylabel('Densità di Probabilità')
ax.set_title('Distribuzioni Genuine vs Impostor e Errori al variare della Soglia')
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

## 3. ATTENZIONE ALLA CONFUSIONE: ML vs Biometria

### 3.1 Confronto Terminologico

È **fondamentale** comprendere le differenze tra le metriche di Machine Learning e quelle biometriche, che spesso generano confusione:

| **Machine Learning** | **Biometria** | **Significato** |
|---------------------|---------------|-----------------|
| True Positive (TP) | Genuine Acceptance (GA) / Genuine Match (GM) | Positivo correttamente identificato |
| True Negative (TN) | Genuine Rejection (GR) / Genuine Non-Match (GNM) | Negativo correttamente identificato |
| False Positive (FP) | False Acceptance (FA) / False Match (FM) | Negativo erroneamente accettato (Tipo II) |
| False Negative (FN) | False Rejection (FR) / False Non-Match (FNM) | Positivo erroneamente rifiutato (Tipo I) |

### 3.2 Differenze Cruciali nelle Metriche

**PRECISION e RECALL** (ML) vs **FAR e FRR** (Biometria) **NON** esprimono le stesse statistiche di errore!

#### Metriche Machine Learning

**PRECISION** (Positive Predictive Value):
$$\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{Positivi Corretti}}{\text{Tutti i Predetti Positivi}}$$

**RECALL** (Sensitivity, True Positive Rate):
$$\text{Recall} = \frac{TP}{TP + FN} = \frac{\text{Positivi Corretti}}{\text{Tutti i Reali Positivi}}$$

Entrambe **partono dalle risposte corrette positive (TP)**!

#### Metriche Biometriche

**FAR** (False Acceptance Rate / False Match Rate):
$$\text{FAR} = \frac{FP}{FP + TN} = \frac{FA}{FA + GR}$$

Probabilità che un **impostor** venga accettato. Se FAR = 0.1%, significa che 1 su 1000 impostori riesce ad accedere.

**FRR** (False Rejection Rate / False Non-Match Rate):
$$\text{FRR} = \frac{FN}{FN + TP} = \frac{FR}{FR + GA}$$

Probabilità che un **utente genuino** venga rifiutato. Se FRR = 0.05%, significa che 1 su 2000 utenti autorizzati non viene riconosciuto.

Entrambe **partono dai due tipi di errore (FP e FN)**!

### 3.3 Equivalenze con Terminologia ML Standard

In termini di Machine Learning:

- **FRR** ≡ **Miss Rate** ≡ **False Negative Rate (FNR)**
  $$\text{FNR} = \frac{FN}{FN + TP} = 1 - \text{Recall}$$

- **FAR** ≡ **Fall-Out** ≡ **False Positive Rate (FPR)**
  $$\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

### 3.4 Visualizzazione Comparativa

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Dati di esempio dalla confusion matrix
labels = ['Precision\n(ML)', 'Recall\n(ML)', 'FAR\n(Biometria)', 'FRR\n(Biometria)']
formulas = [
    f'TP/(TP+FP)\n{tp}/{tp+fp} = {tp/(tp+fp):.3f}',
    f'TP/(TP+FN)\n{tp}/{tp+fn} = {tp/(tp+fn):.3f}',
    f'FP/(FP+TN)\n{fp}/{fp+tn} = {fp/(fp+tn):.3f}',
    f'FN/(FN+TP)\n{fn}/{fn+tp} = {fn/(fn+tp):.3f}'
]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

# Grafico 1: Formule a confronto
axes[0, 0].axis('off')
y_pos = [0.8, 0.6, 0.4, 0.2]
for i, (label, formula, color) in enumerate(zip(labels, formulas, colors)):
    axes[0, 0].text(0.1, y_pos[i], f'{label}:', fontsize=12, fontweight='bold', color=color)
    axes[0, 0].text(0.4, y_pos[i], formula, fontsize=11, family='monospace')
axes[0, 0].set_title('Confronto Formule: ML vs Biometria', fontsize=14, fontweight='bold')

# Grafico 2: Cosa misurano
axes[0, 1].axis('off')
descriptions = [
    'Quanti predetti\npositivi sono corretti',
    'Quanti reali positivi\nsono stati trovati',
    'Quanti impostori\nvengono accettati',
    'Quanti genuini\nvengono rifiutati'
]
for i, (label, desc, color) in enumerate(zip(labels, descriptions, colors)):
    axes[0, 1].text(0.1, y_pos[i], label, fontsize=11, fontweight='bold', color=color)
    axes[0, 1].text(0.45, y_pos[i], desc, fontsize=10)
axes[0, 1].set_title('Cosa Misurano', fontsize=14, fontweight='bold')

# Grafico 3: Matrice con evidenziazione Precision/Recall
cm_display = np.array([[tn, fp], [fn, tp]])
im1 = axes[1, 0].imshow(np.zeros((2, 2)), cmap='Greys', alpha=0.1)
axes[1, 0].text(0, 0, f'TN\n{tn}', ha='center', va='center', fontsize=14)
axes[1, 0].text(1, 0, f'FP\n{fp}', ha='center', va='center', fontsize=14, 
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
axes[1, 0].text(0, 1, f'FN\n{fn}', ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
axes[1, 0].text(1, 1, f'TP\n{tp}', ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(['Pred: Neg', 'Pred: Pos'])
axes[1, 0].set_yticklabels(['Real: Neg', 'Real: Pos'])
axes[1, 0].set_title('Precision (verde+rosso) & Recall (verde+giallo)', fontweight='bold')

# Grafico 4: Matrice con evidenziazione FAR/FRR
axes[1, 1].imshow(np.zeros((2, 2)), cmap='Greys', alpha=0.1)
axes[1, 1].text(0, 0, f'TN\n{tn}', ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
axes[1, 1].text(1, 0, f'FP\n{fp}', ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
axes[1, 1].text(0, 1, f'FN\n{fn}', ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
axes[1, 1].text(1, 1, f'TP\n{tp}', ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_xticklabels(['Pred: Neg', 'Pred: Pos'])
axes[1, 1].set_yticklabels(['Real: Neg', 'Real: Pos'])
axes[1, 1].set_title('FAR (rosso+blu) & FRR (giallo+verde)', fontweight='bold')

plt.tight_layout()
plt.show()
```

### 3.5 Tabella Riassuntiva delle Differenze

```python
import pandas as pd

comparison_data = {
    'Metrica': ['Precision', 'Recall', 'FAR', 'FRR'],
    'Formula': ['TP/(TP+FP)', 'TP/(TP+FN)', 'FP/(FP+TN)', 'FN/(FN+TP)'],
    'Equivalente Biometria': ['GA/(FA+GA)', 'GA/(GA+FR)', 'FA/(FA+GR)', 'FR/(FR+GA)'],
    'Parte da': ['TP corretti', 'TP corretti', 'FP errori', 'FN errori'],
    'Focus': ['Pred. Positivi', 'Reali Positivi', 'Reali Negativi', 'Reali Positivi'],
    'Obiettivo': ['↑ Massimizzare', '↑ Massimizzare', '↓ Minimizzare', '↓ Minimizzare']
}

df_comparison = pd.DataFrame(comparison_data)
print("\n" + "="*80)
print("DIFFERENZE FONDAMENTALI: ML vs BIOMETRIA")
print("="*80)
print(df_comparison.to_string(index=False))
print("="*80)
print("\nNOTA CRITICA:")
print("• Precision e Recall misurano la QUALITÀ delle predizioni positive")
print("• FAR e FRR misurano i TASSI DI ERRORE del sistema")
print("• Non sono intercambiabili nelle valutazioni!")
print("="*80)
```

## 4. Metriche Fondamentali di Machine Learning

### 4.1 Accuracy (Accuratezza)

L'**accuracy** misura la proporzione di predizioni corrette sul totale:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Vantaggi**: Intuitiva e semplice  
**Svantaggi**: Inadeguata per dataset sbilanciati (può essere alta anche con un modello che predice sempre la classe maggioritaria)

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Visualizzazione impatto dello sbilanciamento
class_ratios = [0.5, 0.7, 0.9, 0.95, 0.99]
dummy_accuracies = class_ratios

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(class_ratios, dummy_accuracies, 'o-', linewidth=2, markersize=8, 
        label='Accuracy modello "dummy" (predice sempre classe maggioritaria)')
ax.axhline(accuracy, color='red', linestyle='--', linewidth=2, 
          label=f'Accuracy modello reale: {accuracy:.3f}')
ax.set_xlabel('Proporzione Classe Maggioritaria', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy può essere ingannevole con dataset sbilanciati!', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

### 4.2 Precision (Precisione)

La **precision** misura la proporzione di predizioni positive che sono effettivamente corrette:

$$\text{Precision} = \frac{TP}{TP + FP}$$

Risponde alla domanda: *"Tra tutti i casi predetti come positivi, quanti sono realmente positivi?"*

**Quando è critica**: Alto costo dei falsi positivi (es. spam detection, diagnosi mediche che richiedono trattamenti invasivi)

**Equivalente biometrico**: Positive Predictive Value = GA/(FA+GA)

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print(f'Precision: {precision:.4f}')

# Visualizzazione interpretazione
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Predetti\nPositivi', 'di cui Veri\nPositivi (TP)', 'di cui Falsi\nPositivi (FP)']
values = [tp + fp, tp, fp]
colors = ['lightblue', 'green', 'red']

bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Numero di Campioni', fontsize=12)
ax.set_title(f'Precision = TP / (TP + FP) = {tp}/{tp+fp} = {precision:.3f}', fontsize=14)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
```

### 4.3 Recall (Sensibilità, True Positive Rate, TPR)

Il **recall** misura la proporzione di istanze positive che sono state correttamente identificate:

$$\text{Recall} = \text{TPR} = \text{Sensitivity} = \frac{TP}{TP + FN}$$

Risponde alla domanda: *"Tra tutti i casi realmente positivi, quanti sono stati identificati?"*

**Quando è critico**: Alto costo dei falsi negativi (es. rilevamento tumori, frodi finanziarie, sistemi di sicurezza)

**Equivalente biometrico**: GAR (Genuine Acceptance Rate) = GA/(GA+FR) = 1 - FRR

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
print(f'Recall: {recall:.4f}')

# Visualizzazione interpretazione
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Reali\nPositivi', 'di cui Identificati\n(TP)', 'di cui Mancati\n(FN)']
values = [tp + fn, tp, fn]
colors = ['lightblue', 'green', 'orange']

bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Numero di Campioni', fontsize=12)
ax.set_title(f'Recall = TP / (TP + FN) = {tp}/{tp+fn} = {recall:.3f}', fontsize=14)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
```

### 4.4 Specificity (Specificità, True Negative Rate)

La **specificity** misura la proporzione di istanze negative correttamente identificate:

$$\text{Specificity} = \text{TNR} = \frac{TN}{TN + FP}$$

**Relazione con FPR**:
$$\text{FPR} = 1 - \text{Specificity} = \frac{FP}{FP + TN}$$

```python
specificity = tn / (tn + fp)
fpr = fp / (fp + tn)
print(f'Specificity: {specificity:.4f}')
print(f'FPR (1-Specificity): {fpr:.4f}')
```

### 4.5 Trade-off Precision vs Recall

Precision e Recall sono tipicamente in trade-off: migliorare una peggiora l'altra.

```python
from sklearn.metrics import precision_recall_curve

y_proba = clf.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Curva Precision-Recall
axes[0].plot(recalls, precisions, linewidth=2, color='purple')
axes[0].scatter([recall], [precision], color='red', s=200, zorder=5, 
               label=f'Soglia attuale\nP={precision:.3f}, R={recall:.3f}')
axes[0].set_xlabel('Recall', fontsize=12)
axes[0].set_ylabel('Precision', fontsize=12)
axes[0].set_title('Trade-off Precision-Recall', fontsize=14)
axes[0].grid(alpha=0.3)
axes[0].legend()

# Precision e Recall vs Soglia
axes[1].plot(thresholds, precisions[:-1], label='Precision', linewidth=2)
axes[1].plot(thresholds, recalls[:-1], label='Recall', linewidth=2)
axes[1].set_xlabel('Soglia di Classificazione', fontsize=12)
axes[1].set_ylabel('Valore Metrica', fontsize=12)
axes[1].set_title('Precision e Recall al variare della Soglia', fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4.6 F1-Score

L'**F1-score** è la media armonica di precision e recall, bilanciando i due aspetti:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

La media armonica penalizza valori estremi: se Precision o Recall è bassa, anche F1 sarà bassa.

**Perché media armonica?** 
$$\text{Media Aritmetica} = \frac{P + R}{2} \quad \text{vs} \quad \text{Media Armonica} = \frac{2}{\frac{1}{P} + \frac{1}{R}}$$

La media armonica è più severa con valori sbilanciati.

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
print(f'F1-Score: {f1:.4f}')

# Confronto medie
test_cases = [(0.9, 0.9), (0.9, 0.5), (0.9, 0.1), (0.5, 0.5)]
fig, ax = plt.subplots(figsize=(10, 6))

for i, (p, r) in enumerate(test_cases):
    harmonic = 2 * p * r / (p + r) if (p + r) > 0 else 0
    arithmetic = (p + r) / 2
    
    x_pos = [i*3, i*3+1]
    ax.bar(x_pos, [arithmetic, harmonic], width=0.8, 
           color=['lightblue', 'orange'], alpha=0.7,
           label=['Media Aritmetica', 'Media Armonica (F1)'] if i == 0 else '')
    ax.text(i*3 + 0.5, max(arithmetic, harmonic) + 0.05, 
           f'P={p}, R={r}', ha='center', fontsize=10, fontweight='bold')

ax.set_xticks([i*3 + 0.5 for i in range(len(test_cases))])
ax.set_xticklabels([f'Caso {i+1}' for i in range(len(test_cases))])
ax.set_ylabel('Valore', fontsize=12)
ax.set_title('F1 (Media Armonica) penalizza valori sbilanciati', fontsize=14)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

### 4.7 F-Beta Score

Generalizzazione dell'F1-score che permette di pesare diversamente precision e recall:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

- **β < 1**: Maggior peso alla precision (favorire specificità)
- **β > 1**: Maggior peso al recall (favorire sensibilità)
- **β = 1**: F1-score standard (bilanciamento)
- **β = 2**: F2-score (recall conta il doppio)
- **β = 0.5**: F0.5-score (precision conta il doppio)

```python
from sklearn.metrics import fbeta_score

betas = [0.5, 1, 2, 3]
f_scores = [fbeta_score(y_test, y_pred, beta=b) for b in betas]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar([f'F{b}' for b in betas], f_scores, color=['red', 'purple', 'blue', 'green'], alpha=0.7)
ax.axhline(precision, color='red', linestyle='--', label=f'Precision: {precision:.3f}')
ax.axhline(recall, color='blue', linestyle='--', label=f'Recall: {recall:.3f}')
ax.set_ylabel('Score', fontsize=12)
ax.set_title('F-Beta Score: Bilanciamento tra Precision e Recall', fontsize=14)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar, score in zip(bars, f_scores):
    ax.text(bar.get_x() + bar.get_width()/2, score + 0.01,
           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print(f'F0.5-Score (favorisce Precision): {f_scores[0]:.4f}')
print(f'F1-Score (bilanciato): {f_scores[1]:.4f}')
print(f'F2-Score (favorisce Recall): {f_scores[2]:.4f}')
```

## 5. Tassi di Errore (Error Rates)

### 5.1 False Positive Rate (FPR) - Fall-out

Il **FPR** misura la proporzione di istanze negative erroneamente classificate come positive:

$$\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

**In Biometria**: **FAR** (False Acceptance Rate) - Tasso di accettazione di impostori

Se FAR = 0.1% → 1 su 1000 impostori viene accettato

### 5.2 False Negative Rate (FNR) - Miss Rate

Il **FNR** misura la proporzione di istanze positive erroneamente classificate come negative:

$\text{FNR} = \frac{FN}{FN + TP} = 1 - \text{Recall}$

**In Biometria**: **FRR** (False Rejection Rate) - Tasso di rifiuto di utenti genuini

Se FRR = 0.05% → 1 su 2000 utenti autorizzati viene rifiutato

### 5.3 False Discovery Rate (FDR)

Il **FDR** misura la proporzione di predizioni positive che sono errate:

$\text{FDR} = \frac{FP}{FP + TP} = 1 - \text{Precision}$

### 5.4 Visualizzazione Completa dei Tassi di Errore

```python
# Calcolo di tutti i tassi
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)
fdr = fp / (fp + tp)
tpr = tp / (tp + fn)  # = Recall
tnr = tn / (tn + fp)  # = Specificity

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Grafico 1: FPR vs TNR
rates_neg = [tnr, fpr]
labels_neg = ['TNR\n(Specificity)', 'FPR\n(Fall-out)']
colors_neg = ['green', 'red']
axes[0, 0].bar(labels_neg, rates_neg, color=colors_neg, alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 0].set_ylabel('Tasso', fontsize=12)
axes[0, 0].set_title('Classi Negative: TNR + FPR = 1', fontsize=14)
axes[0, 0].set_ylim([0, 1.1])
for i, (label, rate) in enumerate(zip(labels_neg, rates_neg)):
    axes[0, 0].text(i, rate + 0.02, f'{rate:.3f}', ha='center', fontsize=11, fontweight='bold')
axes[0, 0].axhline(1.0, color='black', linestyle='--', alpha=0.5)
axes[0, 0].grid(axis='y', alpha=0.3)

# Grafico 2: FNR vs TPR
rates_pos = [tpr, fnr]
labels_pos = ['TPR\n(Recall)', 'FNR\n(Miss Rate)']
colors_pos = ['green', 'orange']
axes[0, 1].bar(labels_pos, rates_pos, color=colors_pos, alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 1].set_ylabel('Tasso', fontsize=12)
axes[0, 1].set_title('Classi Positive: TPR + FNR = 1', fontsize=14)
axes[0, 1].set_ylim([0, 1.1])
for i, (label, rate) in enumerate(zip(labels_pos, rates_pos)):
    axes[0, 1].text(i, rate + 0.02, f'{rate:.3f}', ha='center', fontsize=11, fontweight='bold')
axes[0, 1].axhline(1.0, color='black', linestyle='--', alpha=0.5)
axes[0, 1].grid(axis='y', alpha=0.3)

# Grafico 3: Tutte le metriche complementari
all_rates = [tpr, fnr, tnr, fpr, precision, fdr]
all_labels = ['TPR\n(Recall)', 'FNR', 'TNR\n(Spec)', 'FPR', 'Precision', 'FDR']
all_colors = ['green', 'orange', 'green', 'red', 'blue', 'purple']
bars = axes[1, 0].bar(range(len(all_rates)), all_rates, color=all_colors, alpha=0.7, 
                      edgecolor='black', linewidth=2)
axes[1, 0].set_xticks(range(len(all_rates)))
axes[1, 0].set_xticklabels(all_labels, fontsize=10)
axes[1, 0].set_ylabel('Valore', fontsize=12)
axes[1, 0].set_title('Panoramica Completa delle Metriche', fontsize=14)
axes[1, 0].set_ylim([0, 1.1])
for bar, rate in zip(bars, all_rates):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, rate + 0.02,
                    f'{rate:.3f}', ha='center', fontsize=10, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Grafico 4: Relazioni complementari
axes[1, 1].axis('off')
relations = [
    'TPR + FNR = 1',
    'TNR + FPR = 1',
    'Precision + FDR = 1',
    '',
    'TPR = Recall = Sensitivity',
    'FPR = 1 - Specificity = Fall-out',
    'FNR = 1 - Recall = Miss Rate',
    '',
    'ML: FPR, FNR (tassi errore)',
    'Biometria: FAR ≡ FPR, FRR ≡ FNR'
]
y_start = 0.95
for i, rel in enumerate(relations):
    if rel:
        color = 'red' if 'Biometria' in rel or 'ML:' in rel else 'black'
        weight = 'bold' if 'Biometria' in rel or 'ML:' in rel else 'normal'
        axes[1, 1].text(0.1, y_start - i*0.08, rel, fontsize=11, 
                       color=color, fontweight=weight, family='monospace')
axes[1, 1].set_title('Relazioni e Equivalenze', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

### 5.5 False Acceptance Rate (FAR) e False Rejection Rate (FRR) in Biometria

In sistemi biometrici di verifica dell'identità:

**FAR (False Acceptance Rate)**:
$\text{FAR} = \frac{\text{Numero di impostori accettati}}{\text{Numero totale di tentativi da impostori}} = \frac{FP}{FP + TN}$

**FRR (False Rejection Rate)**:
$\text{FRR} = \frac{\text{Numero di genuini rifiutati}}{\text{Numero totale di tentativi genuini}} = \frac{FN}{FN + TP}$

**GAR (Genuine Acceptance Rate)**:
$\text{GAR} = 1 - \text{FRR} = \frac{TP}{TP + FN}$

```python
# Simulazione sistema biometrico
thresholds = np.linspace(0, 1, 100)
fars = []
frrs = []

for tau in thresholds:
    y_pred_tau = (y_proba >= tau).astype(int)
    cm_tau = confusion_matrix(y_test, y_pred_tau)
    if cm_tau.shape == (2, 2):
        tn_t, fp_t, fn_t, tp_t = cm_tau.ravel()
        far = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
        frr = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0
    else:
        far, frr = 0, 0
    fars.append(far)
    frrs.append(frr)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# FAR e FRR vs Threshold
axes[0].plot(thresholds, fars, label='FAR (False Acceptance Rate)', linewidth=2, color='red')
axes[0].plot(thresholds, frrs, label='FRR (False Rejection Rate)', linewidth=2, color='orange')

# Trova EER
diff = np.abs(np.array(fars) - np.array(frrs))
eer_idx = np.argmin(diff)
eer_threshold = thresholds[eer_idx]
eer_value = (fars[eer_idx] + frrs[eer_idx]) / 2

axes[0].plot(eer_threshold, eer_value, 'go', markersize=12, 
            label=f'EER = {eer_value:.3f} @ τ={eer_threshold:.3f}')
axes[0].axvline(eer_threshold, color='green', linestyle='--', alpha=0.5)
axes[0].axhline(eer_value, color='green', linestyle='--', alpha=0.5)

axes[0].set_xlabel('Soglia di Accettazione (τ)', fontsize=12)
axes[0].set_ylabel('Tasso di Errore', fontsize=12)
axes[0].set_title('FAR e FRR al variare della Soglia (Sistema Biometrico)', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])

# Interpretazione
axes[1].axis('off')
info_text = f"""
INTERPRETAZIONE SOGLIA IN BIOMETRIA:

Soglia Bassa (τ → 0):
  • Sistema PERMISSIVO
  • FRR basso (pochi genuini rifiutati)
  • FAR alto (molti impostori accettati)
  • Uso: Applicazioni con bassa sicurezza

Soglia Alta (τ → 1):
  • Sistema RESTRITTIVO
  • FAR basso (pochi impostori accettati)
  • FRR alto (molti genuini rifiutati)
  • Uso: Applicazioni ad alta sicurezza

Equal Error Rate (EER):
  • Soglia ottimale: τ = {eer_threshold:.3f}
  • EER = {eer_value:.3f}
  • FAR = FRR (bilanciamento)
  • Metrica comune per confronto sistemi

Valori Attuali:
  • FAR = {fars[eer_idx]:.4f} ({fars[eer_idx]*100:.2f}%)
  • FRR = {frrs[eer_idx]:.4f} ({frrs[eer_idx]*100:.2f}%)
  • GAR = {1-frrs[eer_idx]:.4f} ({(1-frrs[eer_idx])*100:.2f}%)
"""
axes[1].text(0.1, 0.9, info_text, fontsize=10, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print(f"Equal Error Rate (EER): {eer_value:.4f}")
print(f"EER Threshold: {eer_threshold:.4f}")
```

## 6. Curve ROC e AUC

### 6.1 Curva ROC (Receiver Operating Characteristic)

La **curva ROC** visualizza il trade-off tra TPR (Recall/Sensitivity) e FPR (Fall-out) al variare della soglia di classificazione:

- **Asse Y**: True Positive Rate (TPR) = Recall = Sensitivity = 1 - FRR
- **Asse X**: False Positive Rate (FPR) = Fall-out = 1 - Specificity = FAR

$\text{TPR}(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)}$
$\text{FPR}(\tau) = \frac{FP(\tau)}{FP(\tau) + TN(\tau)}$

**Interpretazione**:
- Punto (0, 0): Tutto classificato come negativo (soglia infinita)
- Punto (1, 1): Tutto classificato come positivo (soglia zero)
- Punto (0, 1): Classificatore perfetto
- Diagonale: Classificatore casuale

```python
from sklearn.metrics import roc_curve, auc

fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, y_proba)
roc_auc = auc(fpr_roc, tpr_roc)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Curva ROC standard
axes[0].plot(fpr_roc, tpr_roc, color='darkorange', lw=3, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier (AUC = 0.5)')
axes[0].fill_between(fpr_roc, tpr_roc, alpha=0.3, color='orange')
axes[0].scatter([0], [1], s=200, c='green', marker='*', zorder=5, 
               label='Classificatore Perfetto')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate (FPR = 1 - Specificity)', fontsize=12)
axes[0].set_ylabel('True Positive Rate (TPR = Recall = Sensitivity)', fontsize=12)
axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

# Punti notevoli sulla curva
idx_eer = np.argmin(np.abs(fpr_roc - (1 - tpr_roc)))
axes[0].plot(fpr_roc[idx_eer], tpr_roc[idx_eer], 'ro', markersize=10, 
            label=f'EER point')

# ROC con annotazioni
axes[1].plot(fpr_roc, tpr_roc, color='darkorange', lw=3)
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].fill_between(fpr_roc, tpr_roc, alpha=0.2, color='orange', label='AUC')

# Annotazioni interpretative
axes[1].annotate('Soglia Alta\n(Conservativo)\nFAR↓ FRR↑', 
                xy=(0.1, 0.5), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
axes[1].annotate('Soglia Bassa\n(Permissivo)\nFAR↑ FRR↓', 
                xy=(0.7, 0.9), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
axes[1].annotate('EER\nBilanciato', 
                xy=(fpr_roc[idx_eer], tpr_roc[idx_eer]), 
                xytext=(0.5, 0.4), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('FPR (False Positive Rate)', fontsize=12)
axes[1].set_ylabel('TPR (True Positive Rate)', fontsize=12)
axes[1].set_title('ROC: Interpretazione delle Zone', fontsize=14, fontweight='bold')
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 6.2 AUC (Area Under the ROC Curve)

L'**AUC** quantifica l'area sotto la curva ROC:

$\text{AUC} = \int_0^1 \text{TPR}(t) \, d(\text{FPR}(t)) = P(score_{positive} > score_{negative})$

L'AUC rappresenta la **probabilità che il classificatore assegni uno score più alto a un esempio positivo casuale rispetto a uno negativo casuale**.

**Interpretazione Rigorosa**:
- **AUC = 1.0**: Classificatore perfetto (separa completamente le classi)
- **AUC = 0.5**: Classificatore casuale (nessun potere discriminante)
- **AUC < 0.5**: Peggio del caso (predizioni invertite)
- **0.5 < AUC < 0.7**: Scarso
- **0.7 ≤ AUC < 0.8**: Accettabile  
- **0.8 ≤ AUC < 0.9**: Eccellente
- **AUC ≥ 0.9**: Outstanding

**Proprietà matematiche**:
- Invariante alla scala (dipende solo dall'ordinamento)
- Robusta a classi sbilanciate (confronta distribuzioni)
- Equivale al test di Wilcoxon-Mann-Whitney

```python
from sklearn.metrics import roc_auc_score

auc_score = roc_auc_score(y_test, y_proba)

# Visualizzazione interpretazione AUC
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Simulazione diversi classificatori
np.random.seed(42)
classifiers = {
    'Perfetto': (np.random.beta(9, 1, 500), np.random.beta(1, 9, 500)),
    'Eccellente': (np.random.beta(7, 2, 500), np.random.beta(2, 7, 500)),
    'Buono': (np.random.beta(5, 3, 500), np.random.beta(3, 5, 500)),
    'Casuale': (np.random.uniform(0, 1, 500), np.random.uniform(0, 1, 500))
}

for idx, (name, (pos_scores, neg_scores)) in enumerate(classifiers.items()):
    ax = axes[idx // 2, idx % 2]
    
    # Distribuzioni
    ax.hist(neg_scores, bins=30, alpha=0.6, label='Negativi', color='red', density=True)
    ax.hist(pos_scores, bins=30, alpha=0.6, label='Positivi', color='green', density=True)
    
    # Calcola AUC
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    auc_val = roc_auc_score(y_true, y_scores)
    
    ax.set_xlabel('Score', fontsize=11)
    ax.set_ylabel('Densità', fontsize=11)
    ax.set_title(f'{name}: AUC = {auc_val:.3f}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('AUC e Separabilità delle Distribuzioni', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print(f'AUC Score del modello: {auc_score:.4f}')
```

### 6.3 Proprietà e Vantaggi della ROC/AUC

```python
# Dimostrazione robustezza a classi sbilanciate
from sklearn.metrics import precision_recall_curve, average_precision_score

# Crea dataset sbilanciato
imbalance_ratios = [0.5, 0.7, 0.9, 0.95]
aucs = []
aps = []

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, ratio in enumerate(imbalance_ratios):
    # Genera dati sbilanciati
    n_samples = 1000
    n_positive = int(n_samples * (1 - ratio))
    n_negative = n_samples - n_positive
    
    X_imb, y_imb = make_classification(n_samples=n_samples, n_features=20,
                                       n_informative=15, weights=[ratio, 1-ratio],
                                       random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_imb, y_imb, test_size=0.3, random_state=42)
    
    clf_imb = RandomForestClassifier(random_state=42)
    clf_imb.fit(X_tr, y_tr)
    y_prob_imb = clf_imb.predict_proba(X_te)[:, 1]
    
    # ROC
    fpr_imb, tpr_imb, _ = roc_curve(y_te, y_prob_imb)
    auc_imb = auc(fpr_imb, tpr_imb)
    aucs.append(auc_imb)
    
    # Precision-Recall
    prec_imb, rec_imb, _ = precision_recall_curve(y_te, y_prob_imb)
    ap_imb = average_precision_score(y_te, y_prob_imb)
    aps.append(ap_imb)
    
    ax = axes[idx // 2, idx % 2]
    ax.plot(fpr_imb, tpr_imb, lw=2, label=f'ROC (AUC={auc_imb:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('FPR', fontsize=11)
    ax.set_ylabel('TPR', fontsize=11)
    ax.set_title(f'Classe Pos: {(1-ratio)*100:.0f}% (Ratio 1:{ratio/(1-ratio):.1f})', 
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('ROC-AUC è robusta a classi sbilanciate', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\nRobustezza AUC vs Sbilanciamento:")
for ratio, auc_val, ap_val in zip(imbalance_ratios, aucs, aps):
    print(f"  Positivi: {(1-ratio)*100:5.1f}% | AUC: {auc_val:.4f} | AP: {ap_val:.4f}")
```

## 7. Curva DET (Detection Error Trade-off)

La **curva DET** è una variante della ROC usata in biometria che plotta FRR vs FAR in scala logaritmica:

- **Asse Y**: FRR (False Rejection Rate) = FNR = 1 - TPR
- **Asse X**: FAR (False Acceptance Rate) = FPR

A differenza della ROC (che mostra il positivo), la DET enfatizza gli errori.

```python
from scipy.stats import norm

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Curva DET lineare
axes[0].plot(fpr_roc, 1 - tpr_roc, 'b-', lw=3, label='Curva DET')
axes[0].set_xlabel('FAR (False Acceptance Rate)', fontsize=12)
axes[0].set_ylabel('FRR (False Rejection Rate)', fontsize=12)
axes[0].set_title('DET Curve (scala lineare)', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].legend()

# EER point
axes[0].plot(fpr_roc[idx_eer], 1-tpr_roc[idx_eer], 'ro', markersize=12, 
            label=f'EER = {(fpr_roc[idx_eer]):.3f}')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

# Curva DET logaritmica
axes[1].plot(fpr_roc, 1 - tpr_roc, 'b-', lw=3, label='Curva DET')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel('FAR (False Acceptance Rate) [log]', fontsize=12)
axes[1].set_ylabel('FRR (False Rejection Rate) [log]', fontsize=12)
axes[1].set_title('DET Curve (scala logaritmica)', fontsize=14, fontweight='bold')
axes[1].grid(True, which="both", alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()
```

## 8. Curva Precision-Recall

La **curva Precision-Recall** è particolarmente utile per dataset sbilanciati dove la classe positiva è rara:

```python
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Curva PR
axes[0].plot(recall_curve, precision_curve, color='blue', lw=3,
            label=f'PR curve (AP = {avg_precision:.3f})')
axes[0].fill_between(recall_curve, precision_curve, alpha=0.3, color='blue')

baseline = np.sum(y_test) / len(y_test)
axes[0].axhline(baseline, color='red', linestyle='--', lw=2,
               label=f'Baseline (prevalenza = {baseline:.3f})')

axes[0].set_xlabel('Recall (TPR)', fontsize=12)
axes[0].set_ylabel('Precision (PPV)', fontsize=12)
axes[0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
axes[0].legend(loc="lower left")
axes[0].grid(alpha=0.3)
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])

# Confronto ROC vs PR per dataset sbilanciato
axes[1].plot(fpr_roc, tpr_roc, 'r-', lw=2, label=f'ROC (AUC={roc_auc:.3f})', alpha=0.7)
axes[1].set_xlabel('FPR / Recall*', fontsize=12)
axes[1].set_ylabel('TPR / Precision*', fontsize=12)

# Sovrapponi PR (scalata)
recall_scaled = recall_curve
precision_scaled = precision_curve  
axes[1].plot(recall_scaled, precision_scaled, 'b-', lw=2, 
            label=f'PR (AP={avg_precision:.3f})', alpha=0.7)

axes[1].set_title('ROC vs PR: Visualizzazione Comparativa', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.05])

plt.tight_layout()
plt.show()
```

### 8.1 Average Precision (AP)

L'**Average Precision** riassume la curva PR come la media pesata delle precision a ogni soglia:

$\text{AP} = \sum_n (R_n - R_{n-1}) \cdot P_n$

dove $P_n$ e $R_n$ sono precision e recall alla soglia $n$.

**Differenza con AUC**:
- AUC-ROC: Buona per dataset bilanciati, mostra trade-off TPR/FPR
- AUC-PR (AP): Preferibile per dataset sbilanciati, focus su classe positiva

```python
print(f"Average Precision: {avg_precision:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"\nPer dataset sbilanciato (classe positiva {baseline*100:.1f}%):")
print(f"  → AP è più informativa di AUC")
```

## 9. Equal Error Rate (EER) e Punti Operativi

### 9.1 Equal Error Rate (EER)

L'**EER** è il punto dove FAR = FRR (o FPR = FNR), rappresentando un bilanciamento tra i due tipi di errore:

$\text{EER} = \text{FAR}(\tau^*) = \text{FRR}(\tau^*) \quad \text{dove} \quad \tau^* : \text{FAR}(\tau) = \text{FRR}(\tau)$

```python
# Calcolo preciso EER
eer_value_precise = (fars[eer_idx] + frrs[eer_idx]) / 2
eer_threshold_precise = thresholds[eer_idx]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# FAR e FRR
axes[0, 0].plot(thresholds, fars, 'r-', lw=2, label='FAR')
axes[0, 0].plot(thresholds, frrs, 'orange', lw=2, label='FRR')
axes[0, 0].plot(eer_threshold_precise, eer_value_precise, 'go', markersize=12, 
               label=f'EER={eer_value_precise:.4f}')
axes[0, 0].axvline(eer_threshold_precise, color='green', linestyle='--', alpha=0.5)
axes[0, 0].axhline(eer_value_precise, color='green', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Soglia τ', fontsize=12)
axes[0, 0].set_ylabel('Tasso di Errore', fontsize=12)
axes[0, 0].set_title('Equal Error Rate (EER)', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# ROC con punto EER
axes[0, 1].plot(fpr_roc, tpr_roc, 'b-', lw=2, label='ROC')
axes[0, 1].plot([0, 1], [1, 0], 'k--', lw=1, alpha=0.5, label='FAR = 1-TPR line')
axes[0, 1].plot(fpr_roc[idx_eer], tpr_roc[idx_eer], 'ro', markersize=12, 
               label=f'EER point')
axes[0, 1].set_xlabel('FPR (FAR)', fontsize=12)
axes[0, 1].set_ylabel('TPR (1-FRR)', fontsize=12)
axes[0, 1].set_title('EER sulla Curva ROC', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# ZeroFRR e ZeroFAR
zero_frr_idx = np.where(np.array(frrs) == 0)[0]
zero_far_idx = np.where(np.array(fars) == 0)[0]

if len(zero_frr_idx) > 0:
    zerofrr_far = fars[zero_frr_idx[0]]
    zerofrr_tau = thresholds[zero_frr_idx[0]]
else:
    zerofrr_far, zerofrr_tau = np.nan, np.nan

if len(zero_far_idx) > 0:
    zerofar_frr = frrs[zero_far_idx[-1]]
    zerofar_tau = thresholds[zero_far_idx[-1]]
else:
    zerofar_frr, zerofar_tau = np.nan, np.nan

axes[1, 0].plot(thresholds, fars, 'r-', lw=2, label='FAR')
axes[1, 0].plot(thresholds, frrs, 'orange', lw=2, label='FRR')

if not np.isnan(zerofrr_tau):
    axes[1, 0].plot(zerofrr_tau, zerofrr_far, 'bs', markersize=10, 
                   label=f'ZeroFRR: FAR={zerofrr_far:.4f}')
if not np.isnan(zerofar_tau):
    axes[1, 0].plot(zerofar_tau, zerofar_frr, 'm^', markersize=10, 
                   label=f'ZeroFAR: FRR={zerofar_frr:.4f}')

axes[1, 0].set_xlabel('Soglia τ', fontsize=12)
axes[1, 0].set_ylabel('Tasso di Errore', fontsize=12)
axes[1, 0].set_title('Punti Operativi: ZeroFRR e ZeroFAR', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Tabella riassuntiva
axes[1, 1].axis('off')
summary_text = f"""
PUNTI OPERATIVI NOTEVOLI

Equal Error Rate (EER):
  • Soglia: τ = {eer_threshold_precise:.4f}
  • EER = {eer_value_precise:.4f}
  • FAR = FRR (bilanciato)
  • Uso: Confronto sistemi biometrici

ZeroFRR (Zero False Rejection):
  • Soglia: τ = {zerofrr_tau:.4f if not np.isnan(zerofrr_tau) else 'N/A'}
  • FAR = {zerofrr_far:.4f if not np.isnan(zerofrr_far) else 'N/A'}
  • FRR = 0 (nessun genuino rifiutato)
  • Uso: Max convenienza utente

ZeroFAR (Zero False Acceptance):
  • Soglia: τ = {zerofar_tau:.4f if not np.isnan(zerofar_tau) else 'N/A'}
  • FRR = {zerofar_frr:.4f if not np.isnan(zerofar_frr) else 'N/A'}
  • FAR = 0 (nessun impostor accettato)
  • Uso: Max sicurezza

SCELTA DELLA SOGLIA:
✓ EER → Bilanciamento generale
✓ ZeroFRR → Priorità UX (es. smartphone)
✓ ZeroFAR → Priorità sicurezza (es. banca)
✓ Custom → Bilanciamento specifico
"""
axes[1, 1].text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
               family='monospace', 
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.show()

print(f"\nPunti Operativi del Sistema:")
print(f"  EER: {eer_value_precise:.4f} @ τ={eer_threshold_precise:.4f}")
if not np.isnan(zerofrr_tau):
    print(f"  ZeroFRR: FAR={zerofrr_far:.4f} @ τ={zerofrr_tau:.4f}")
if not np.isnan(zerofar_tau):
    print(f"  ZeroFAR: FRR={zerofar_frr:.4f} @ τ={zerofar_tau:.4f}")
```

### 9.2 Confronto tra Sistemi usando EER

L'EER è una metrica standard per confrontare sistemi biometrici: **EER più basso = sistema migliore**.

```python
# Simulazione confronto sistemi
systems = {
    'Sistema A (Eccellente)': np.random.beta(8, 2, 1000),
    'Sistema B (Buono)': np.random.beta(6, 3, 1000),
    'Sistema C (Medio)': np.random.beta(5, 4, 1000),
    'Sistema D (Scarso)': np.random.beta(4, 5, 1000)
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

eers_systems = []
for idx, (name, pos_scores) in enumerate(systems.items()):
    ax = axes[idx // 2, idx % 2]
    
    # Genera impostori
    neg_scores = np.random.beta(2, 8, 1000)
    
    # Crea labels
    y_true_sys = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores_sys = np.concatenate([pos_scores, neg_scores])
    
    # Calcola FAR e FRR
    thresholds_sys = np.linspace(0, 1, 200)
    fars_sys, frrs_sys = [], []
    
    for tau in thresholds_sys:
        fa = np.sum((y_scores_sys[y_true_sys == 0] >= tau))
        fr = np.sum((y_scores_sys[y_true_sys == 1] < tau))
        tn = np.sum((y_scores_sys[y_true_sys == 0] < tau))
        tp = np.sum((y_scores_sys[y_true_sys == 1] >= tau))
        
        far_sys = fa / (fa + tn) if (fa + tn) > 0 else 0
        frr_sys = fr / (fr + tp) if (fr + tp) > 0 else 0
        fars_sys.append(far_sys)
        frrs_sys.append(frr_sys)
    
    # Trova EER
    diff_sys = np.abs(np.array(fars_sys) - np.array(frrs_sys))
    eer_idx_sys = np.argmin(diff_sys)
    eer_sys = (fars_sys[eer_idx_sys] + frrs_sys[eer_idx_sys]) / 2
    eers_systems.append((name, eer_sys))
    
    ax.plot(thresholds_sys, fars_sys, 'r-', lw=2, label='FAR')
    ax.plot(thresholds_sys, frrs_sys, 'orange', lw=2, label='FRR')
    ax.plot(thresholds_sys[eer_idx_sys], eer_sys, 'go', markersize=10, 
           label=f'EER={eer_sys:.4f}')
    ax.axhline(eer_sys, color='green', linestyle='--', alpha=0.3)
    ax.set_xlabel('Soglia', fontsize=11)
    ax.set_ylabel('Tasso Errore', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 0.5])

plt.suptitle('Confronto Sistemi Biometrici tramite EER', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# Ranking
eers_systems.sort(key=lambda x: x[1])
print("\nRanking Sistemi (EER crescente = migliore):")
for rank, (name, eer) in enumerate(eers_systems, 1):
    print(f"  {rank}. {name}: EER = {eer:.4f}")
```

## 10. Metriche Avanzate

### 10.1 Matthews Correlation Coefficient (MCC)

Il **MCC** è considerato una delle migliori metriche per classificazione binaria, specialmente con classi sbilanciate:

$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$

**Proprietà**:
- Range: [-1, 1]
- MCC = 1: Predizione perfetta
- MCC = 0: Predizione casuale
- MCC = -1: Totale disaccordo
- Simmetrico rispetto a classi positive/negative
- Robusto a sbilanciamenti

```python
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_test, y_pred)
print(f'Matthews Correlation Coefficient: {mcc:.4f}')

# Comparazione MCC con altre metriche su dataset sbilanciati
imbalance_ratios = [0.5, 0.7, 0.9, 0.95, 0.99]
metrics_comparison = {'Accuracy': [], 'F1': [], 'MCC': []}

fig, ax = plt.subplots(figsize=(12, 6))

for ratio in imbalance_ratios:
    X_imb, y_imb = make_classification(n_samples=1000, n_features=20,
                                       weights=[ratio, 1-ratio], random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_imb, y_imb, test_size=0.3, random_state=42)
    
    clf_imb = RandomForestClassifier(random_state=42)
    clf_imb.fit(X_tr, y_tr)
    y_pred_imb = clf_imb.predict(X_te)
    
    metrics_comparison['Accuracy'].append(accuracy_score(y_te, y_pred_imb))
    metrics_comparison['F1'].append(f1_score(y_te, y_pred_imb))
    metrics_comparison['MCC'].append(matthews_corrcoef(y_te, y_pred_imb))

for metric, values in metrics_comparison.items():
    ax.plot(imbalance_ratios, values, marker='o', markersize=8, lw=2, label=metric)

ax.set_xlabel('Proporzione Classe Maggioritaria', fontsize=12)
ax.set_ylabel('Valore Metrica', fontsize=12)
ax.set_title('MCC è più stabile con dataset sbilanciati rispetto ad Accuracy', 
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.show()
```

### 10.2 Cohen's Kappa

Il **Cohen's Kappa** misura l'accordo tra predizioni e valori reali, correggendo per l'accordo casuale:

$\kappa = \frac{p_o - p_e}{1 - p_e}$

dove:
- $p_o = \frac{TP + TN}{TP + TN + FP + FN}$ : Accuratezza osservata
- $p_e = \frac{(TP+FP)(TP+FN) + (TN+FP)(TN+FN)}{(TP+TN+FP+FN)^2}$ : Accordo casuale atteso

**Interpretazione**:
- κ < 0: Accordo peggiore del caso
- 0 ≤ κ < 0.20: Accordo lieve
- 0.20 ≤ κ < 0.40: Accordo discreto
- 0.40 ≤ κ < 0.60: Accordo moderato
- 0.60 ≤ κ < 0.80: Accordo sostanziale
- 0.80 ≤ κ ≤ 1.00: Accordo quasi perfetto

```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa: {kappa:.4f}")

# Visualizzazione interpretazione
kappa_ranges = [
    (0, 0.20, 'Lieve', 'lightcoral'),
    (0.20, 0.40, 'Discreto', 'lightyellow'),
    (0.40, 0.60, 'Moderato', 'lightblue'),
    (0.60, 0.80, 'Sostanziale', 'lightgreen'),
    (0.80, 1.00, 'Quasi Perfetto', 'darkgreen')
]

fig, ax = plt.subplots(figsize=(12, 3))

for start, end, label, color in kappa_ranges:
    ax.barh(0, end - start, left=start, height=0.5, color=color, 
           edgecolor='black', linewidth=2, alpha=0.7)
    ax.text((start + end) / 2, 0, f'{label}\n[{start:.1f}-{end:.1f}]', 
           ha='center', va='center', fontsize=10, fontweight='bold')

ax.plot([kappa, kappa], [-0.3, 0.3], 'r-', linewidth=4, label=f'Kappa={kappa:.3f}')
ax.scatter([kappa], [0], s=200, c='red', marker='v', zorder=5, edgecolor='black', linewidth=2)

ax.set_xlim([0, 1])
ax.set_ylim([-0.4, 0.4])
ax.set_xlabel("Cohen's Kappa", fontsize=12)
ax.set_yticks([])
ax.set_title("Interpretazione Cohen's Kappa", fontsize=14, fontweight='bold')
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()
```

### 10.3 Log Loss (Cross-Entropy Loss)

La **Log Loss** valuta la qualità delle probabilità predette, penalizzando fortemente predizioni con alta confidenza ma errate:

$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$

dove $p_i$ è la probabilità predetta per la classe positiva.

**Proprietà**:
- Range: [0, ∞), valori più bassi sono migliori
- Log Loss = 0: Predizioni probabilistiche perfette
- Penalizza errori con alta confidenza più di altri

```python
from sklearn.metrics import log_loss

logloss = log_loss(y_test, y_proba)
print(f'Log Loss: {logloss:.4f}')

# Dimostrazione penalizzazione
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grafico 1: Log Loss per diverse predizioni
true_label = 1
predicted_probs = np.linspace(0.01, 0.99, 100)
log_losses = [-np.log(p) if true_label == 1 else -np.log(1-p) for p in predicted_probs]

axes[0].plot(predicted_probs, log_losses, 'b-', lw=3)
axes[0].axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Incertezza (p=0.5)')
axes[0].set_xlabel('Probabilità Predetta (per classe positiva)', fontsize=12)
axes[0].set_ylabel('Log Loss', fontsize=12)
axes[0].set_title('Log Loss penalizza predizioni sbagliate con alta confidenza\n(Vera classe: Positiva)', 
                 fontsize=13, fontweight='bold')
axes[0].set_ylim([0, 5])
axes[0].grid(alpha=0.3)
axes[0].legend()

# Annotazioni
axes[0].annotate('Predizione corretta\ncon alta confidenza\n(Loss → 0)', 
                xy=(0.95, 0.1), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
axes[0].annotate('Predizione sbagliata\ncon alta confidenza\n(Loss → ∞)', 
                xy=(0.05, 4), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# Grafico 2: Esempi concreti
scenarios = [
    ('Corretto\nconfidente', 1, 0.95, 'green'),
    ('Corretto\nincerto', 1, 0.55, 'lightgreen'),
    ('Sbagliato\nincerto', 1, 0.45, 'orange'),
    ('Sbagliato\nconfidente', 1, 0.05, 'red')
]

losses_scenarios = []
for _, true, pred, _ in scenarios:
    loss = -np.log(pred) if true == 1 else -np.log(1-pred)
    losses_scenarios.append(loss)

bars = axes[1].bar(range(len(scenarios)), losses_scenarios, 
                   color=[c for _, _, _, c in scenarios], alpha=0.7, 
                   edgecolor='black', linewidth=2)
axes[1].set_xticks(range(len(scenarios)))
axes[1].set_xticklabels([s[0] for s in scenarios], fontsize=10)
axes[1].set_ylabel('Log Loss', fontsize=12)
axes[1].set_title('Log Loss per Scenari Diversi', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

for bar, loss, (_, _, pred, _) in zip(bars, losses_scenarios, scenarios):
    axes[1].text(bar.get_x() + bar.get_width()/2, loss + 0.05,
                f'{loss:.3f}\np={pred}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
```

### 10.4 Brier Score

Il **Brier Score** misura l'accuratezza delle predizioni probabilistiche come MSE:

$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$

**Proprietà**:
- Range: [0, 1], valori più bassi sono migliori
- Brier = 0: Probabilità perfette
- Equivalente al Mean Squared Error

```python
from sklearn.metrics import brier_score_loss

brier = brier_score_loss(y_test, y_proba)
print(f'Brier Score: {brier:.4f}')

# Confronto Log Loss vs Brier Score
fig, ax = plt.subplots(figsize=(10, 6))

true_label = 1
probs = np.linspace(0.01, 0.99, 100)
log_losses_comp = [-np.log(p) for p in probs]
brier_scores = [(p - true_label)**2 for p in probs]

ax.plot(probs, log_losses_comp, 'b-', lw=3, label='Log Loss')
ax2 = ax.twinx()
ax2.plot(probs, brier_scores, 'r-', lw=3, label='Brier Score')

ax.set_xlabel('Probabilità Predetta', fontsize=12)
ax.set_ylabel('Log Loss', fontsize=12, color='blue')
ax2.set_ylabel('Brier Score', fontsize=12, color='red')
ax.set_title('Log Loss vs Brier Score\n(Log Loss penalizza più severamente)', 
            fontsize=14, fontweight='bold')
ax.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='red')
ax.grid(alpha=0.3)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

plt.tight_layout()
plt.show()
```

## 11. Calibrazione delle Probabilità

La **calibration curve** mostra se le probabilità predette riflettono la vera probabilità:

```python
from sklearn.calibration import calibration_curve, CalibrationDisplay

fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_proba, n_bins=10)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Curva di calibrazione
axes[0].plot(mean_predicted_value, fraction_of_positives, "s-", 
            linewidth=2, markersize=8, label="Modello")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfettamente calibrato")
axes[0].set_xlabel("Probabilità Predetta", fontsize=12)
axes[0].set_ylabel("Frazione di Positivi", fontsize=12)
axes[0].set_title("Curva di Calibrazione", fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Distribuzione delle probabilità
axes[1].hist(y_proba[y_test == 0], bins=20, alpha=0.6, label='Negativi', 
            color='red', density=True)
axes[1].hist(y_proba[y_test == 1], bins=20, alpha=0.6, label='Positivi', 
            color='green', density=True)
axes[1].set_xlabel('Probabilità Predetta', fontsize=12)
axes[1].set_ylabel('Densità', fontsize=12)
axes[1].set_title('Distribuzione delle Probabilità Predette', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 12. Metriche per Classificazione Multi-classe

### 12.1 Strategie di Aggregazione

Per estendere le metriche binarie al caso multi-classe:

#### Macro-averaging
$\text{Metric}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{Metric}_k$
Tutte le classi hanno lo stesso peso.

#### Micro-averaging
$\text{Precision}_{\text{micro}} = \frac{\sum_{k=1}^{K} TP_k}{\sum_{k=1}^{K} (TP_k + FP_k)}$
Aggrega i contributi; favorisce classi frequenti.

#### Weighted-averaging
$\text{Metric}_{\text{weighted}} = \sum_{k=1}^{K} w_k \cdot \text{Metric}_k$
dove $w_k$ è la frequenza della classe $k$.

```python
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Dataset multi-classe
X_mc, y_mc = make_classification(n_samples=1000, n_features=20, n_classes=4,
                                  n_informative=15, n_redundant=5, n_clusters_per_class=1,
                                  random_state=42)
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(
    X_mc, y_mc, test_size=0.3, random_state=42)

clf_mc = RandomForestClassifier(random_state=42)
clf_mc.fit(X_train_mc, y_train_mc)
y_pred_mc = clf_mc.predict(X_test_mc)

# Calcolo metriche
averages = ['macro', 'micro', 'weighted']
metrics = {}

for avg in averages:
    metrics[avg] = {
        'Precision': precision_score(y_test_mc, y_pred_mc, average=avg, zero_division=0),
        'Recall': recall_score(y_test_mc, y_pred_mc, average=avg, zero_division=0),
        'F1-Score': f1_score(y_test_mc, y_pred_mc, average=avg, zero_division=0)
    }

# Visualizzazione
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Grafico comparativo
metric_names = list(metrics['macro'].keys())
x = np.arange(len(metric_names))
width = 0.25

for i, avg in enumerate(averages):
    values = [metrics[avg][m] for m in metric_names]
    axes[0].bar(x + i*width, values, width, label=avg.capitalize(), alpha=0.8)

axes[0].set_xlabel('Metrica', fontsize=12)
axes[0].set_ylabel('Valore', fontsize=12)
axes[0].set_title('Confronto Strategie di Aggregazione (Multi-classe)', 
                 fontsize=14, fontweight='bold')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metric_names)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 1.1])

# Matrice di confusione
cm_mc = confusion_matrix(y_test_mc, y_pred_mc)
im = axes[1].imshow(cm_mc, cmap='Blues')
axes[1].set_xlabel('Predetto', fontsize=12)
axes[1].set_ylabel('Reale', fontsize=12)
axes[1].set_title('Matrice di Confusione Multi-classe', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(4))
axes[1].set_yticks(range(4))

for i in range(4):
    for j in range(4):
        text = axes[1].text(j, i, cm_mc[i, j], ha="center", va="center",
                          color="white" if cm_mc[i, j] > cm_mc.max()/2 else "black",
                          fontsize=12, fontweight='bold')

plt.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.show()

# Report dettagliato
print("\n" + "="*60)
print("CLASSIFICATION REPORT (Multi-classe)")
print("="*60)
print(classification_report(y_test_mc, y_pred_mc, digits=4))
```

## 13. Ottimizzazione della Soglia

Analisi completa per trovare la soglia ottimale:

```python
thresholds_range = np.linspace(0, 1, 200)
metrics_vs_threshold = {
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'F2-Score': [],
    'Accuracy': []
}

for threshold in thresholds_range:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    
    metrics_vs_threshold['Precision'].append(
        precision_score(y_test, y_pred_thresh, zero_division=0))
    metrics_vs_threshold['Recall'].append(
        recall_score(y_test, y_pred_thresh, zero_division=0))
    metrics_vs_threshold['F1-Score'].append(
        f1_score(y_test, y_pred_thresh, zero_division=0))
    metrics_vs_threshold['F2-Score'].append(
        fbeta_score(y_test, y_pred_thresh, beta=2, zero_division=0))
    metrics_vs_threshold['Accuracy'].append(
        accuracy_score(y_test, y_pred_thresh))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Grafico 1: Tutte le metriche
for metric, values in metrics_vs_threshold.items():
    axes[0, 0].plot(thresholds_range, values, lw=2, label=metric)

optimal_f1_idx = np.argmax(metrics_vs_threshold['F1-Score'])
optimal_f1_threshold = thresholds_range[optimal_f1_idx]
axes[0, 0].axvline(optimal_f1_threshold, color='red', linestyle='--', lw=2, 
                  label=f'Ottimo F1: τ={optimal_f1_threshold:.3f}')
axes[0, 0].axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Default (0.5)')

axes[0, 0].set_xlabel('Soglia di Classificazione', fontsize=12)
axes[0, 0].set_ylabel('Valore Metrica', fontsize=12)
axes[0, 0].set_title('Metriche vs Soglia', fontsize=14, fontweight='bold')
axes[0, 0].legend(loc='best')
axes[0, 0].grid(alpha=0.3)

# Grafico 2: Focus Precision-Recall
axes[0, 1].plot(thresholds_range, metrics_vs_threshold['Precision'], 
               'b-', lw=2, label='Precision')
axes[0, 1].plot(thresholds_range, metrics_vs_threshold['Recall'], 
               'r-', lw=2, label='Recall')
axes[0, 1].plot(thresholds_range, metrics_vs_threshold['F1-Score'], 
               'g-', lw=2, label='F1-Score')

axes[0, 1].axvline(optimal_f1_threshold, color='green', linestyle='--', lw=2)
axes[0, 1].scatter([optimal_f1_threshold], 
                  [metrics_vs_threshold['F1-Score'][optimal_f1_idx]],
                  s=200, c='green', marker='*', zorder=5, edgecolor='black', linewidth=2)

axes[0, 1].set_xlabel('Soglia', fontsize=12)
axes[0, 1].set_ylabel('Valore', fontsize=12)
axes[0, 1].set_title('Trade-off Precision-Recall', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Grafico 3: Heatmap metriche per diversi obiettivi
objectives = {
    'Balanced (F1)': optimal_f1_idx,
    'High Recall (F2)': np.argmax(metrics_vs_threshold['F2-Score']),
    'High Precision': np.argmax(metrics_vs_threshold['Precision']),
    'Max Accuracy': np.argmax(metrics_vs_threshold['Accuracy'])
}

obj_data = []
for obj_name, idx in objectives.items():
    tau = thresholds_range[idx]
    obj_data.append([
        tau,
        metrics_vs_threshold['Precision'][idx],
        metrics_vs_threshold['Recall'][idx],
        metrics_vs_threshold['F1-Score'][idx],
        metrics_vs_threshold['Accuracy'][idx]
    ])

obj_df = pd.DataFrame(obj_data, 
                     columns=['Soglia', 'Precision', 'Recall', 'F1', 'Accuracy'],
                     index=list(objectives.keys()))

im = axes[1, 0].imshow(obj_df.iloc[:, 1:].values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
axes[1, 0].set_xticks(range(4))
axes[1, 0].set_xticklabels(['Precision', 'Recall', 'F1', 'Accuracy'], rotation=45)
axes[1, 0].set_yticks(range(len(objectives)))
axes[1, 0].set_yticklabels(list(objectives.keys()))
axes[1, 0].set_title('Metriche per Diversi Obiettivi', fontsize=14, fontweight='bold')

for i in range(len(objectives)):
    for j in range(4):
        text = axes[1, 0].text(j, i, f'{obj_df.iloc[i, j+1]:.3f}',
                              ha="center", va="center", color="black", fontsize=10)
    axes[1, 0].text(-1.2, i, f'τ={obj_df.iloc[i, 0]:.3f}',
                   ha="right", va="center", fontsize=9, style='italic')

plt.colorbar(im, ax=axes[1, 0])

# Grafico 4: Raccomandazioni
axes[1, 1].axis('off')
recommendations = f"""
RACCOMANDAZIONI SOGLIA OTTIMALE

1. Balanced (F1 massimo):
   • Soglia: τ = {thresholds_range[objectives['Balanced (F1)']].3f}
   • F1 = {metrics_vs_threshold['F1-Score'][objectives['Balanced (F1)']].4f}
   • Uso: Bilanciamento generale

2. High Recall (Sensibilità):
   • Soglia: τ = {thresholds_range[objectives['High Recall (F2)']].3f}
   • F2 = {metrics_vs_threshold['F2-Score'][objectives['High Recall (F2)']].4f}
   • Uso: Minimizzare falsi negativi
   • Es: Diagnosi mediche, frodi

3. High Precision (Specificità):
   • Soglia: τ = {thresholds_range[objectives['High Precision']].3f}
   • Prec = {metrics_vs_threshold['Precision'][objectives['High Precision']].4f}
   • Uso: Minimizzare falsi positivi
   • Es: Spam detection, raccomandazioni

4. Max Accuracy:
   • Soglia: τ = {thresholds_range[objectives['Max Accuracy']].3f}
   • Acc = {metrics_vs_threshold['Accuracy'][objectives['Max Accuracy']].4f}
   • Uso: Solo se classi bilanciate

SCELTA PRATICA:
✓ Start con F1 massimo
✓ Aggiusta in base al costo errori
✓ Valida su set di validazione
✓ Considera vincoli di business
"""

axes[1, 1].text(0.05, 0.95, recommendations, fontsize=9, verticalalignment='top',
               family='monospace', 
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("ANALISI SOGLIE OTTIMALI")
print("="*70)
print(obj_df.to_string())
print("="*70)
```

## 14. Lift e Gain Charts

### 14.1 Cumulative Gain Chart

Il **Cumulative Gain Chart** mostra la percentuale di target catturati in funzione della popolazione contattata:

```python
# Ordina per probabilità decrescente
sorted_indices = np.argsort(y_proba)[::-1]
y_sorted = y_test.iloc[sorted_indices].values if hasattr(y_test, 'iloc') else y_test[sorted_indices]

cumulative_gains = np.cumsum(y_sorted) / np.sum(y_sorted)
percentage_population = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Cumulative Gain
axes[0].plot(percentage_population * 100, cumulative_gains * 100, 
            label='Modello', lw=3, color='blue')
axes[0].plot([0, 100], [0, 100], 'k--', label='Random', lw=2)
axes[0].fill_between(percentage_population * 100, cumulative_gains * 100, 
                     percentage_population * 100, alpha=0.3, color='blue')

# Punto al 20% e 50%
idx_20 = int(0.2 * len(y_sorted))
idx_50 = int(0.5 * len(y_sorted))
gain_20 = cumulative_gains[idx_20] * 100
gain_50 = cumulative_gains[idx_50] * 100

axes[0].plot(20, gain_20, 'ro', markersize=12)
axes[0].plot(50, gain_50, 'go', markersize=12)
axes[0].annotate(f'{gain_20:.1f}% target\ncon 20% popolazione', 
                xy=(20, gain_20), xytext=(30, gain_20-15),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7))
axes[0].annotate(f'{gain_50:.1f}% target\ncon 50% popolazione', 
                xy=(50, gain_50), xytext=(60, gain_50-15),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

axes[0].set_xlabel('Percentuale della Popolazione (%)', fontsize=12)
axes[0].set_ylabel('Guadagno Cumulativo (%)', fontsize=12)
axes[0].set_title('Cumulative Gain Chart', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xlim([0, 100])
axes[0].set_ylim([0, 105])

# Gain per decile
n_deciles = 10
decile_gains = []
for i in range(n_deciles):
    start = int(i * len(y_sorted) / n_deciles)
    end = int((i + 1) * len(y_sorted) / n_deciles)
    if i == 0:
        gain = cumulative_gains[end-1]
    else:
        gain = cumulative_gains[end-1] - cumulative_gains[start-1]
    decile_gains.append(gain * 100)

axes[1].bar(range(1, n_deciles + 1), decile_gains, color='steelblue', 
           alpha=0.7, edgecolor='black', linewidth=2)
axes[1].axhline(10, color='red', linestyle='--', linewidth=2, label='Random (10% per decile)')
axes[1].set_xlabel('Decile', fontsize=12)
axes[1].set_ylabel('Gain (%)', fontsize=12)
axes[1].set_title('Gain per Decile', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

for i, gain in enumerate(decile_gains):
    axes[1].text(i+1, gain + 0.5, f'{gain:.1f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nCon il 20% della popolazione catturiamo {gain_20:.1f}% dei target")
print(f"Con il 50% della popolazione catturiamo {gain_50:.1f}% dei target")
```

### 14.2 Lift Chart

Il **Lift** misura quante volte il modello è migliore di una selezione casuale:

$\text{Lift} = \frac{\text{Precision at depth}}{\text{Overall prevalence}} = \frac{\text{Tasso di successo nel segmento}}{\text{Tasso di successo globale}}$

```python
n_deciles = 10
decile_size = len(y_sorted) // n_deciles
lifts = []
precisions_decile = []
baseline = np.mean(y_sorted)

for i in range(n_deciles):
    start_idx = i * decile_size
    end_idx = (i + 1) * decile_size if i < n_deciles - 1 else len(y_sorted)
    decile_precision = np.sum(y_sorted[start_idx:end_idx]) / (end_idx - start_idx)
    precisions_decile.append(decile_precision * 100)
    lift = decile_precision / baseline if baseline > 0 else 0
    lifts.append(lift)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Lift Chart
bars = axes[0].bar(range(1, n_deciles + 1), lifts, color='teal', 
                   alpha=0.7, edgecolor='black', linewidth=2)
axes[0].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (Lift=1)')
axes[0].set_xlabel('Decile', fontsize=12)
axes[0].set_ylabel('Lift', fontsize=12)
axes[0].set_title('Lift Chart per Decile', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

for bar, lift in zip(bars, lifts):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.05,
                f'{lift:.2f}x', ha='center', fontsize=10, fontweight='bold')

# Interpretazione Lift
axes[1].axis('off')
lift_interpretation = f"""
INTERPRETAZIONE LIFT

Baseline (prevalenza): {baseline*100:.2f}%

Top Decile (primi 10%):
  • Precision: {precisions_decile[0]:.1f}%
  • Lift: {lifts[0]:.2f}x
  • Interpretazione: {lifts[0]:.1f} volte meglio
    del caso casuale

Lift = 1.0: Come selezione casuale
Lift > 1.0: Meglio del caso
Lift < 1.0: Peggio del caso

APPLICAZIONI PRATICHE:

Marketing Campaigns:
  Lift = 3.0 al top decile significa:
  • Contattando 10% clienti selezionati
  • Otteniamo 3x più conversioni
  • Rispetto a selezione casuale

Fraud Detection:
  Lift = 5.0 significa:
  • Top 10% transazioni sospette
  • Contiene 5x più frodi
  • Rispetto alla media

STRATEGIA:
✓ Focus sui decili con Lift > 2
✓ Ignora decili con Lift < 1
✓ Ottimizza risorse su high-lift
"""

axes[1].text(0.05, 0.95, lift_interpretation, fontsize=9, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

plt.tight_layout()
plt.show()

print("\nLift per Decile:")
for i, (lift, prec) in enumerate(zip(lifts, precisions_decile), 1):
    print(f"  Decile {i}: Lift={lift:.3f}, Precision={prec:.1f}%")
```

## 15. Cross-Validation e Stabilità delle Metriche

Valutazione robusta attraverso k-fold cross-validation:

```python
from sklearn.model_selection import cross_val_score, cross_validate

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(clf, X, y, cv=5, scoring=scoring, 
                            return_train_score=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Box plot delle metriche
metrics_cv_data = [cv_results[f'test_{metric}'] for metric in scoring]
bp = axes[0].boxplot(metrics_cv_data, labels=[s.upper() for s in scoring],
                     patch_artist=True, showmeans=True)

for patch, color in zip(bp['boxes'], ['red', 'blue', 'green', 'purple', 'orange']):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Stabilità Metriche (5-Fold CV)', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 1.05])

# Media e deviazione standard
axes[1].axis('off')
cv_summary = "RISULTATI CROSS-VALIDATION (5-Fold)\n" + "="*50 + "\n\n"

for metric in scoring:
    scores = cv_results[f'test_{metric}']
    cv_summary += f"{metric.upper():12s}: "
    cv_summary += f"{scores.mean():.4f} (±{scores.std() * 2:.4f})\n"
    cv_summary += f"{'':12s}  Min: {scores.min():.4f}, Max: {scores.max():.4f}\n\n"

cv_summary += "\nINTERPRETAZIONE:\n"
cv_summary += "• Std bassa → Modello stabile\n"
cv_summary += "• Std alta → Sensibile ai dati\n"
cv_summary += "• Intervallo ±2σ copre ~95% casi\n"

axes[1].text(0.1, 0.9, cv_summary, fontsize=10, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.show()

print("\nCross-Validation Results (5-fold):")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"  {metric.upper():12s}: {scores.mean():.4f} (±{scores.std() * 2:.4f})")
```

## 16. Riepilogo Completo e Linee Guida

### 16.1 Tabella Riassuntiva Metriche

```python
# Calcolo completo di tutte le metriche
all_metrics = {
    'Metrica': [
        'Accuracy', 'Precision', 'Recall/TPR', 'Specificity/TNR',
        'F1-Score', 'F2-Score', 'FPR/FAR', 'FNR/FRR',
        'MCC', "Cohen's Kappa", 'AUC-ROC', 'AUC-PR',
        'Log Loss', 'Brier Score'
    ],
    'Valore': [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        tn / (tn + fp),
        f1_score(y_test, y_pred),
        fbeta_score(y_test, y_pred, beta=2),
        fp / (fp + tn),
        fn / (fn + tp),
        matthews_corrcoef(y_test, y_pred),
        cohen_kappa_score(y_test, y_pred),
        roc_auc_score(y_test, y_proba),
        average_precision_score(y_test, y_proba),
        log_loss(y_test, y_proba),
        brier_score_loss(y_test, y_proba)
    ],
    'Range': [
        '[0, 1]', '[0, 1]', '[0, 1]', '[0, 1]',
        '[0, 1]', '[0, 1]', '[0, 1]', '[0, 1]',
        '[-1, 1]', '[-1, 1]', '[0, 1]', '[0, 1]',
        '[0, ∞)', '[0, 1]'
    ],
    'Meglio': [
        '↑', '↑', '↑', '↑',
        '↑', '↑', '↓', '↓',
        '↑', '↑', '↑', '↑',
        '↓', '↓'
    ],
    'Categoria': [
        'Base', 'Base', 'Base', 'Base',
        'Combinata', 'Combinata', 'Errore', 'Errore',
        'Avanzata', 'Avanzata', 'Curva', 'Curva',
        'Probabilistica', 'Probabilistica'
    ]
}

df_metrics = pd.DataFrame(all_metrics)
df_metrics['Valore'] = df_metrics['Valore'].round(4)

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

colors = []
for cat in df_metrics['Categoria']:
    if cat == 'Base':
        colors.append('#e3f2fd')
    elif cat == 'Combinata':
        colors.append('#fff3e0')
    elif cat == 'Errore':
        colors.append('#ffebee')
    elif cat == 'Avanzata':
        colors.append('#f3e5f5')
    elif cat == 'Curva':
        colors.append('#e8f5e9')
    else:
        colors.append('#fce4ec')

table = ax.table(cellText=df_metrics.values, colLabels=df_metrics.columns,
                cellLoc='center', loc='center',
                colColours=['lightgray']*5)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(len(df_metrics)):
    for j in range(5):
        cell = table[(i+1, j)]
        cell.set_facecolor(colors[i])
        if j == 1:  # Colonna Valore
            cell.set_text_props(weight='bold')

ax.set_title('RIEPILOGO COMPLETO METRICHE DI CLASSIFICAZIONE', 
            fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("RIEPILOGO METRICHE DI CLASSIFICAZIONE")
print("="*70)
print(df_metrics.to_string(index=False))
print("="*70)
```

### 16.2 Linee Guida per la Scelta delle Metriche

```python
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

guidelines = """
╔══════════════════════════════════════════════════════════════════╗
║         LINEE GUIDA PER LA SCELTA DELLE METRICHE                ║
╚══════════════════════════════════════════════════════════════════╝

📊 SCENARIO: Dataset Bilanciato
   Metriche:  Accuracy, F1-Score, AUC-ROC
   Rationale: Accuracy è affidabile, F1 bilancia bene

📊 SCENARIO: Dataset Sbilanciato
   Metriche:  Precision, Recall, F1, AUC-PR, MCC
   Rationale: Accuracy è ingannevole, focus su classe rara
   ⚠️  NON usare: Accuracy da sola

📊 SCENARIO: Costo FP alto (es. Spam, Pubblicità)
   Metriche:  Precision, FPR, FDR
   Rationale: Minimizzare falsi allarmi
   Obiettivo: Alta Precision, Basso FPR

📊 SCENARIO: Costo FN alto (es. Medicina, Frodi)
   Metriche:  Recall, FNR, F2-Score
   Rationale: Non perdere casi positivi
   Obiettivo: Alto Recall, Basso FNR

📊 SCENARIO: Probabilità importanti
   Metriche:  Log Loss, Brier Score, Calibration
   Rationale: Valutare qualità delle probabilità
   Uso:       Sistemi che usano le probabilità

📊 SCENARIO: Multi-classe
   Metriche:  Macro/Micro/Weighted F1, MCC, Kappa
   Rationale: Considerare tutte le classi
   Strategia: Macro per bilanciamento, Micro per volume

📊 SCENARIO: Ranking importante (es. IR, Recommender)
   Metriche:  AUC-ROC, AUC-PR, Lift, Gain
   Rationale: Ordinamento più importante che soglia
   Uso:       Top-K raccomandazioni

📊 SCENARIO: Confronto Sistemi Biometrici
   Metriche:  EER, ZeroFAR, ZeroFRR, ROC, DET
   Rationale: Standard del settore biometrico
   Focus:     FAR vs FRR trade-off

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 PRINCIPI GENERALI:

1. ✓ NON esiste una metrica universale
   Ogni metrica cattura aspetti diversi delle performance

2. ✓ Usa SEMPRE multiple metriche complementari
   Una singola metrica può essere ingannevole

3. ✓ Considera il COSTO degli errori
   FP e FN hanno spesso costi asimmetrici nel mondo reale

4. ✓ Valuta su set di VALIDAZIONE separato
   Evita overfitting sulle metriche

5. ✓ Usa CROSS-VALIDATION per robustezza
   Singolo split può dare risultati ottimistici/pessimistici

6. ✓ Visualizza con CURVE (ROC, PR, DET)
   Forniscono insight oltre ai singoli numeri

7. ✓ Comunica l'INCERTEZZA
   Riporta sempre intervalli di confidenza

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  ATTENZIONE CONFUSIONE ML vs BIOMETRIA:

   ML (Machine Learning)        vs    Biometria
   ────────────────────────────────────────────────────
   Precision (PPV)                    GA/(GA+FA)
   Recall (Sensitivity, TPR)          GAR = 1-FRR
   FPR (Fall-out)                     FAR
   FNR (Miss Rate)                    FRR
   
   DIFFERENZA CHIAVE:
   • Precision/Recall partono da TP (predizioni corrette)
   • FAR/FRR partono da FP/FN (errori del sistema)
   
   ⚠️  NON sono intercambiabili!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 CHECKLIST VALUTAZIONE MODELLO:

☐ Matrice di confusione visualizzata
☐ Metriche base calcolate (P, R, F1)
☐ Metriche avanzate (MCC, Kappa, AUC)
☐ Curve visualizzate (ROC, PR)
☐ Calibrazione verificata
☐ Cross-validation eseguita
☐ Soglia ottimizzata per use case
☐ Trade-offs documentati
☐ Metriche contestualizzate al problema
☐ Limitazioni discusse
"""

ax.text(0.5, 0.5, guidelines, fontsize=9, verticalalignment='center',
       horizontalalignment='center', family='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1))

plt.tight_layout()
plt.show()
```

## 17. Conclusioni

La valutazione rigorosa di modelli di classificazione richiede:

1. **Comprensione profonda** delle metriche e delle loro proprietà matematiche
2. **Consapevolezza delle differenze** tra terminologia ML e biometrica
3. **Scelta contestuale** delle metriche appropriate al problema
4. **Analisi multi-dimensionale** con curve e visualizzazioni
5. **Validazione robusta** tramite cross-validation
6. **Comunicazione chiara** di limitazioni e trade-offs

**Ricorda**:
- ✓ Precision e Recall ≠ FAR e FRR (diverse statistiche!)
- ✓ Nessuna metrica è perfetta per tutti gli scenari
- ✓ Dataset sbilanciati richiedono metriche speciali
- ✓ Le curve (ROC, PR, DET) forniscono più informazioni dei singoli numeri
- ✓ La soglia di decisione è un parametro critico da ottimizzare
- ✓ Il contesto applicativo determina quali errori sono più costosi

---

## Appendice: Formule Complete e Relazioni

### Formule Matematiche Riassuntive

**Metriche Base**:
$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$

$\text{Precision} = \frac{TP}{TP + FP}$

$\text{Recall (TPR)} = \frac{TP}{TP + FN}$

$\text{Specificity (TNR)} = \frac{TN}{TN + FP}$

**Tassi di Errore**:
$\text{FPR (FAR)} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$

$\text{FNR (FRR)} = \frac{FN}{FN + TP} = 1 - \text{Recall}$

$\text{FDR} = \frac{FP}{FP + TP} = 1 - \text{Precision}$

**F-Scores**:
$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$

$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$

**Metriche Avanzate**:
$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$

$\kappa = \frac{p_o - p_e}{1 - p_e}$

**Metriche Probabilistiche**:
$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$

$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$

**Relazioni Fondamentali**:
$\text{TPR} + \text{FNR} = 1$
$\text{TNR} + \text{FPR} = 1$
$\text{Precision} + \text{FDR} = 1$
$\text{GAR} = 1 - \text{FRR} = \text{Recall}$

### Grafico Finale: Relazioni tra Metriche

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Diagramma di Venn delle metriche
ax = axes[0, 0]
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Cerchi per TP, FP, FN, TN
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch

# Confusion matrix visual
box_size = 3
ax.add_patch(Rectangle((1, 6), box_size, box_size, facecolor='lightgreen', 
                       edgecolor='black', linewidth=3, alpha=0.7))
ax.text(2.5, 7.5, f'TP\n{tp}', ha='center', va='center', fontsize=16, fontweight='bold')

ax.add_patch(Rectangle((5, 6), box_size, box_size, facecolor='lightcoral', 
                       edgecolor='black', linewidth=3, alpha=0.7))
ax.text(6.5, 7.5, f'FP\n{fp}', ha='center', va='center', fontsize=16, fontweight='bold')

ax.add_patch(Rectangle((1, 2), box_size, box_size, facecolor='lightyellow', 
                       edgecolor='black', linewidth=3, alpha=0.7))
ax.text(2.5, 3.5, f'FN\n{fn}', ha='center', va='center', fontsize=16, fontweight='bold')

ax.add_patch(Rectangle((5, 2), box_size, box_size, facecolor='lightblue', 
                       edgecolor='black', linewidth=3, alpha=0.7))
ax.text(6.5, 3.5, f'TN\n{tn}', ha='center', va='center', fontsize=16, fontweight='bold')

# Labels
ax.text(2.5, 9.5, 'Pred: POS', ha='center', fontsize=12, fontweight='bold')
ax.text(6.5, 9.5, 'Pred: NEG', ha='center', fontsize=12, fontweight='bold')
ax.text(0.2, 7.5, 'Real:\nPOS', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.2, 3.5, 'Real:\nNEG', ha='center', va='center', fontsize=12, fontweight='bold')

# Formule collegate
ax.text(2.5, 1, f'Precision = TP/(TP+FP) = {precision:.3f}', 
       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
ax.text(2.5, 0.3, f'Recall = TP/(TP+FN) = {recall:.3f}', 
       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
ax.text(6.5, 1, f'FPR = FP/(FP+TN) = {fpr:.3f}', 
       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
ax.text(6.5, 0.3, f'FNR = FN/(FN+TP) = {fnr:.3f}', 
       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

ax.set_title('Matrice di Confusione e Relazioni', fontsize=14, fontweight='bold')

# 2. Trade-offs principali
ax = axes[0, 1]
x_vals = np.linspace(0, 1, 100)

# Simula trade-off ideale
precision_tradeoff = 1 / (1 + 2*x_vals)
recall_tradeoff = x_vals

ax.plot(recall_tradeoff, precision_tradeoff, 'b-', lw=3, label='Precision vs Recall')
ax.scatter([recall], [precision], s=300, c='red', marker='*', 
          zorder=5, edgecolor='black', linewidth=2, label='Punto Attuale')

# F1 iso-lines
f1_values = [0.3, 0.5, 0.7, 0.9]
for f1_val in f1_values:
    recall_line = np.linspace(0.01, 1, 100)
    precision_line = (f1_val * recall_line) / (2 * recall_line - f1_val)
    precision_line = np.clip(precision_line, 0, 1)
    ax.plot(recall_line, precision_line, '--', alpha=0.4, 
           label=f'F1={f1_val}' if f1_val in [0.5, 0.9] else '')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Trade-off Precision-Recall con Iso-F1', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# 3. Comparazione ML vs Biometria
ax = axes[1, 0]
ax.axis('off')

comparison_text = """
╔═══════════════════════════════════════════════════════════════╗
║       DIFFERENZE CRITICHE: ML vs BIOMETRIA                   ║
╚═══════════════════════════════════════════════════════════════╝

TERMINOLOGIA:
┌────────────────┬─────────────────┬──────────────────────┐
│ Machine Learn. │    Biometria    │   Significato        │
├────────────────┼─────────────────┼──────────────────────┤
│ True Positive  │ Genuine Accept  │ Correttamente +      │
│ False Positive │ False Accept    │ Errore Tipo II       │
│ False Negative │ False Reject    │ Errore Tipo I        │
│ True Negative  │ Genuine Reject  │ Correttamente -      │
└────────────────┴─────────────────┴──────────────────────┘

METRICHE - ATTENZIONE ALLE FORMULE:
┌─────────────────────────────────────────────────────────────┐
│ PRECISION (ML) ≠ metriche biometriche                       │
│   Formula: TP / (TP + FP)                                   │
│   Significato: Accuratezza predizioni positive              │
│   Equivalente Bio: GA/(GA+FA) - Positive Predictive Value  │
│                                                             │
│ RECALL (ML) ≠ GAR ma RECALL = 1 - FRR                      │
│   Formula: TP / (TP + FN)                                   │
│   Significato: Copertura dei positivi                       │
│   Equivalente Bio: GAR (Genuine Acceptance Rate)           │
│                                                             │
│ FPR (ML) = FAR (Biometria) ✓                              │
│   Formula: FP / (FP + TN)                                   │
│   Significato: Tasso accettazione impostori                 │
│                                                             │
│ FNR (ML) = FRR (Biometria) ✓                              │
│   Formula: FN / (FN + TP)                                   │
│   Significato: Tasso rifiuto genuini                        │
└─────────────────────────────────────────────────────────────┘

FOCUS DIVERSO:
┌─────────────────────────────────────────────────────────────┐
│ ML (Precision/Recall):                                      │
│   → Partono da PREDIZIONI CORRETTE (TP)                    │
│   → Valutano QUALITÀ delle predizioni                       │
│   → Ottimizzazione: massimizzare TP                         │
│                                                             │
│ Biometria (FAR/FRR):                                       │
│   → Partono da ERRORI del sistema (FP, FN)                 │
│   → Valutano TASSI DI ERRORE                               │
│   → Ottimizzazione: minimizzare errori                      │
└─────────────────────────────────────────────────────────────┘

⚠️  NON INTERCAMBIABILI NELLE VALUTAZIONI!
"""

ax.text(0.5, 0.5, comparison_text, fontsize=8, family='monospace',
       verticalalignment='center', horizontalalignment='center',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# 4. Decision framework
ax = axes[1, 1]
ax.axis('off')

decision_text = """
╔═══════════════════════════════════════════════════════════════╗
║           FRAMEWORK DECISIONALE METRICHE                     ║
╚═══════════════════════════════════════════════════════════════╝

DOMANDE CHIAVE:

1️⃣  Le classi sono bilanciate?
   └─ SÌ  → Accuracy, F1, ROC-AUC
   └─ NO  → Precision, Recall, PR-AUC, MCC

2️⃣  Quale errore è più costoso?
   └─ FP  → Maximizza Precision, minimizza FPR
   └─ FN  → Maximizza Recall, minimizza FNR
   └─ Simile → F1-Score, EER

3️⃣  Le probabilità sono importanti?
   └─ SÌ  → Log Loss, Brier, Calibration
   └─ NO  → Metriche basate su soglia

4️⃣  È un ranking problem?
   └─ SÌ  → AUC-ROC, AUC-PR, Lift, Gain
   └─ NO  → Metriche basate su soglia fissa

5️⃣  Quante classi?
   └─ 2   → Metriche binarie
   └─ >2  → Macro/Micro/Weighted averaging

6️⃣  Dominio applicativo?
   └─ Biometria    → EER, FAR, FRR, DET
   └─ ML generale  → Precision, Recall, F1
   └─ IR/Ranking   → MAP, NDCG, MRR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMBINAZIONI RACCOMANDATE:

🏥 MEDICINA (diagnosi tumori)
   Primary: Recall (↑), FNR (↓)
   Secondary: F2, Specificity
   
💳 FRODI BANCARIE
   Primary: Recall (↑), Precision
   Secondary: F1, AUC-PR
   
📧 SPAM DETECTION  
   Primary: Precision (↑), FPR (↓)
   Secondary: F0.5, Specificity
   
🔐 BIOMETRIA (accesso)
   Primary: EER, FAR, FRR
   Secondary: ZeroFAR, ROC, DET
   
🎯 RECOMMENDER SYSTEMS
   Primary: Precision@K, Recall@K
   Secondary: MAP, NDCG, Lift
"""

ax.text(0.5, 0.5, decision_text, fontsize=8, family='monospace',
       verticalalignment='center', horizontalalignment='center',
       bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

plt.suptitle('SINTESI COMPLETA: Relazioni, Differenze e Framework Decisionale', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()
```

### Esempio Pratico Completo: Caso d'Uso End-to-End

```python
print("\n" + "="*80)
print("ESEMPIO PRATICO: VALUTAZIONE COMPLETA SISTEMA DI CLASSIFICAZIONE")
print("="*80)
print("\nScenario: Sistema di Rilevamento Frodi Bancarie")
print("-"*80)

# Metriche attuali
print("\n📊 METRICHE CALCOLATE:")
print(f"  Accuracy:           {accuracy:.4f}")
print(f"  Precision:          {precision:.4f}")
print(f"  Recall:             {recall:.4f}")
print(f"  F1-Score:           {f1:.4f}")
print(f"  F2-Score:           {fbeta_score(y_test, y_pred, beta=2):.4f}")
print(f"  MCC:                {mcc:.4f}")
print(f"  ROC-AUC:            {roc_auc:.4f}")
print(f"  PR-AUC:             {avg_precision:.4f}")
print(f"  FPR (FAR):          {fpr:.4f}")
print(f"  FNR (FRR):          {fnr:.4f}")

print("\n💡 INTERPRETAZIONE:")
prevalence = np.mean(y_test)
print(f"  Prevalenza frodi:   {prevalence*100:.2f}%")
print(f"  Dataset:            {'Sbilanciato' if prevalence < 0.3 else 'Bilanciato'}")

print("\n🎯 ANALISI PERFORMANCE:")
if recall > 0.85:
    print(f"  ✓ Recall alto ({recall:.2f}): Buona copertura delle frodi")
else:
    print(f"  ⚠ Recall basso ({recall:.2f}): Molte frodi non rilevate!")

if precision > 0.75:
    print(f"  ✓ Precision alta ({precision:.2f}): Pochi falsi allarmi")
else:
    print(f"  ⚠ Precision bassa ({precision:.2f}): Troppi falsi positivi")

if f1 > 0.80:
    print(f"  ✓ F1 alto ({f1:.2f}): Buon bilanciamento generale")
else:
    print(f"  ⚠ F1 medio ({f1:.2f}): Considerare ottimizzazione")

print("\n💰 IMPATTO BUSINESS:")
# Simula costi
cost_fp = 10  # Costo investigazione falso positivo
cost_fn = 1000  # Costo frode non rilevata
n_transactions = len(y_test)

total_cost_fp = fp * cost_fp
total_cost_fn = fn * cost_fn
total_cost = total_cost_fp + total_cost_fn

print(f"  Transazioni totali:     {n_transactions}")
print(f"  Costo FP (€{cost_fp}/caso):       €{total_cost_fp:,.2f}")
print(f"  Costo FN (€{cost_fn}/caso):      €{total_cost_fn:,.2f}")
print(f"  Costo Totale:           €{total_cost:,.2f}")

# Baseline (predice sempre negativo)
cost_baseline = np.sum(y_test) * cost_fn
saving = cost_baseline - total_cost
saving_pct = (saving / cost_baseline) * 100

print(f"\n  Costo Baseline (no modello): €{cost_baseline:,.2f}")
print(f"  Risparmio con modello:       €{saving:,.2f} ({saving_pct:.1f}%)")

print("\n🔧 RACCOMANDAZIONI:")
if recall < 0.85:
    print("  1. Abbassare soglia per aumentare Recall")
    print(f"     Soglia attuale: 0.5 → Suggerita: {optimal_f1_threshold:.3f}")
    
if fpr > 0.05:
    print("  2. FPR alto: considerare feature engineering")
    
if mcc < 0.7:
    print("  3. MCC basso: provare algoritmi più sofisticati")

print("\n📈 PROSSIMI PASSI:")
print("  1. Analizzare errori (FP e FN) per pattern")
print("  2. Calibrare soglia in base a costo FP vs FN")
print("  3. Validare su dati temporali futuri")
print("  4. Monitorare performance in produzione")
print("  5. Re-train periodicamente con nuovi dati")

print("\n" + "="*80)
print("REPORT COMPLETO GENERATO")
print("="*80)
```

---

## Fine del Documento

Questo documento ha fornito una trattazione **completa, rigorosa e dettagliata** di tutte le metriche di valutazione per la classificazione in machine learning, con particolare attenzione a:

✅ **Formule matematiche** precise e rigorose  
✅ **Differenze critiche** tra ML e biometria  
✅ **Visualizzazioni esplicative** con Python/Matplotlib/Seaborn  
✅ **Esempi pratici** e casi d'uso reali  
✅ **Linee guida** per la scelta appropriata  
✅ **Trade-offs** e ottimizzazioni  

**Metriche coperte**: Accuracy, Precision, Recall, F1/F-beta, Specificity, FPR/FAR, FNR/FRR, FDR, MCC, Cohen's Kappa, ROC-AUC, PR-AUC, EER, Log Loss, Brier Score, Calibration, Lift, Gain, e molto altro.

La valutazione rigorosa è **fondamentale** per lo sviluppo di sistemi di machine learning affidabili e adatti al contesto applicativo specifico.
