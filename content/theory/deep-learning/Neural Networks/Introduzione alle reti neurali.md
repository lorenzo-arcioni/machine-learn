# Script Video: Implementazione di un Perceptron con PyTorch

## **Introduzione**
*"Allora, oggi voglio mostrarvi come implementare un perceptron da zero con PyTorch, e... diciamo che l'obiettivo è veramente capire cosa succede sotto il cofano quando parliamo di reti neurali. Perché vedete, spesso si parla di deep learning, di neural network, però... beh, alla base di tutto c'è questo concetto fondamentale che è il perceptron. Quindi partiamo dalle basi, implementiamo tutto passo passo, e... sì, alla fine dovreste avere una comprensione molto più solida di come funzionano questi sistemi."*

## **Cos'è Veramente un Perceptron**

*"Allora, il perceptron... diciamo che è il modello più elementare di neurone artificiale, però non è che sia così banale come sembra.Fondamentalmente quello che fa è prendere degli input numerici, li moltiplica per dei pesi - che sono i parametri che il modello deve imparare - li somma, aggiunge un bias, e poi... ecco, passa tutto attraverso una funzione di attivazione. Nel nostro caso useremo la sigmoid."*

**Formula matematica:**
```
y = σ(Wx + b)
```

*"Okay, scomponiamo questa formula pezzo per pezzo, perché... diciamo che è importante capire cosa rappresenta ogni simbolo e soprattutto le dimensioni. Allora, y è il nostro output finale - quello che il perceptron ci restituisce, e nel nostro caso ha dimensione (200, 1) perché abbiamo 200 esempi e 1 output per esempio. Poi abbiamo σ, che è la sigma greca, e rappresenta la funzione di attivazione - nel nostro caso la sigmoid. Wx è... beh, il prodotto matriciale tra i pesi W e gli input x. E qui è dove le dimensioni diventano importanti: x ha shape (200, 2) perché abbiamo 200 punti bidimensionali, W ha shape (1, 2) perché dobbiamo andare da 2 input a 1 output, e... diciamo che il prodotto risulta in (200, 1). W sono i parametri che il modello deve imparare durante il training - praticamente i pesi delle connessioni sinaptiche, se vogliamo fare un'analogia biologica. Infine b è il bias, che ha dimensione (1,) - praticamente un singolo scalare che viene aggiunto a ogni esempio. E... ecco, è un termine costante che permette di traslare la funzione. Senza il bias, la nostra retta passerebbe sempre per l'origine, e... beh, questo limiterebbe molto la flessibilità del modello."*

*"Ora, geometricamente parlando... e questo è importante capirlo, eh... il perceptron cerca di trovare un iperpiano che separa le due classi. Se pensate a un problema bidimensionale, è come se stesse cercando una retta che divide i punti in due gruppi. Ecco, questo è quello che fa. Ovviamente funziona solo per problemi linearmente separabili, e... beh, questa era una delle limitazioni storiche che poi ha portato allo sviluppo delle reti multi-layer."*

**Diagramma dell'architettura:**
```
     x₁ ────────┐
                │  w₁
                ├────► Σ ────► σ ────► y
                │  w₂    +b
     x₂ ────────┘

Input Layer    Linear Transform    Activation    Output
   (2,)            (1,2)×(2,)        σ()        (1,)
```

*"Ecco, questo diagramma vi fa vedere esattamente la struttura del perceptron. Abbiamo due input - x₁ e x₂ - che... diciamo rappresentano le coordinate dei nostri punti. Ogni input ha il suo peso - w₁ e w₂ - e tutti convergono verso un singolo neurone che fa la somma pesata, aggiunge il bias b, e poi... applica la sigmoid. È proprio la rappresentazione visiva di quella formula che abbiamo visto prima."*

## **Setup dell'Ambiente**

```python
# Importiamo PyTorch e alcuni moduli:
# - torch: contiene le operazioni di base sui tensori
# - torch.nn: contiene i moduli per definire reti neurali
# - torch.optim: contiene gli algoritmi di ottimizzazione
# - matplotlib: ci serve per visualizzare i dati
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
```

*"Okay, partiamo con gli import. Allora, torch ovviamente è il core di PyTorch, poi abbiamo nn che... beh, contiene tutti i building blocks per costruire le reti neurali, torch.optim per gli ottimizzatori - SGD, Adam e così via - e matplotlib per le visualizzazioni. Ah, una cosa importante: in PyTorch tutto ruota attorno ai tensori, che sono... diciamo come gli array di NumPy ma ottimizzati per GPU e per il calcolo automatico dei gradienti."*

## **Creazione del Dataset**

```python
"In PyTorch tutto gira intorno ai tensori, che sono come array di NumPy ma ottimizzati per il calcolo su CPU e GPU."
# Creiamo un dataset fittizio

# Fissiamo un seme casuale (seed) per avere sempre gli stessi numeri casuali.
# In questo modo, se rieseguiamo il codice, otteniamo sempre lo stesso risultato.
torch.manual_seed(42)

# Numero di punti per ciascuna classe
n_points = 100

# torch.randn genera numeri casuali da una distribuzione normale (media = 0, varianza = 1).
# Creiamo 100 punti 2D intorno a (2,2) per la classe 0.
class0 = torch.randn(n_points, 2) + torch.tensor([2, 2])

# Creiamo altri 100 punti 2D intorno a (-2,-2) per la classe 1.
class1 = torch.randn(n_points, 2) + torch.tensor([-2, -2])

# Uniamo i due gruppi di punti in un unico tensore X di input.
X = torch.cat([class0, class1], dim=0)

# Creiamo i target (etichette): 0 per la prima classe, 1 per la seconda.
# unsqueeze(1) serve per trasformare il vettore in una colonna (necessario per PyTorch).
y = torch.cat([torch.zeros(n_points), torch.ones(n_points)]).unsqueeze(1)

# Visualizziamo i punti per capire come sono distribuiti.
plt.scatter(class0[:,0], class0[:,1], c="blue", label="Classe 0")
plt.scatter(class1[:,0], class1[:,1], c="red", label="Classe 1")
plt.legend()
plt.show()
```

*"Bene, ora creiamo i dati per l'esempio. Prima cosa: fisso il seed, perché... sennò ogni volta che eseguo il codice ottengo risultati diversi, e questo non va bene per la riproducibilità. Poi genero due cluster di punti: uno centrato su (2,2) per la classe 0, l'altro su (-2,-2) per la classe 1. Uso torch.randn che... appunto, genera numeri da una distribuzione normale, poi aggiungo un offset per spostare i cluster. È importante notare che torch.cat mi permette di concatenare i tensori, e... ecco, unsqueeze(1) serve perché PyTorch si aspetta che i target abbiano una certa forma. La visualizzazione ci fa vedere subito che i dati sono linearmente separabili."*

## **Implementazione del Modello**

```python
# Definiamo il nostro perceptron come sottoclasse di nn.Module.
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()  
        # Definiamo uno strato lineare con 2 input (x e y) e 1 output.
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        # L'output lineare viene trasformato con una funzione sigmoid,
        # che restituisce valori tra 0 e 1 (cioè probabilità).
        return torch.sigmoid(self.fc(x))

# Creiamo un'istanza del modello
model = Perceptron()
print(model)
```

*"Ecco, qui definiamo il nostro perceptron. Allora, in PyTorch ogni modello è una classe che eredita da nn.Module, e... questo è fondamentale perché ci dà accesso a tutto il sistema di automatic differentiation. Il layer lineare nn.Linear(2,1) implementa quella trasformazione Wx + b che abbiamo visto prima. I pesi vengono inizializzati automaticamente... PyTorch usa l'inizializzazione di Kaiming di default, che è... diciamo una buona scelta per la maggior parte dei casi. Poi nella forward, applico prima la trasformazione lineare e poi la sigmoid, che mi mappa tutto tra 0 e 1, quindi posso interpretarlo come una probabilità."*

## **Loss Function e Ottimizzatore**

```python
# Definiamo la funzione di costo (loss): Binary Cross Entropy,
# che misura l'errore tra probabilità predetta e valore vero (0 o 1).
criterion = nn.BCELoss()

# Definiamo l'ottimizzatore: Stochastic Gradient Descent (SGD).
# optimizer si occupa di aggiornare i pesi del modello durante il training.
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

*"Per la loss function uso la Binary Cross Entropy, che... è praticamente lo standard per classificazione binaria. Perché? Beh, matematicamente deriva dal principio di maximum likelihood, e... ha questa bella proprietà che quando le predizioni sono sbagliate la loss cresce molto rapidamente, quindi spinge il modello a correggere gli errori. Per l'optimizer, SGD con learning rate 0.1... è un valore che ho scelto empiricamente, diciamo che per questo tipo di problema funziona bene. Il learning rate controlla quanto aggressive sono gli aggiornamenti dei pesi, e... ecco, trovare il valore giusto è spesso una questione di trial and error."*

## **Training Loop**

```python
# Numero di epoche di addestramento
epochs = 50
losses = []

# Ciclo di training
for epoch in range(epochs):
    # Forward pass: calcoliamo le predizioni del modello
    y_pred = model(X)

    # Calcoliamo la loss confrontando predizioni e target
    loss = criterion(y_pred, y)
    losses.append(loss.item())  # .item() converte il tensore in numero Python

    # Backward pass:
    optimizer.zero_grad()  # azzera i gradienti accumulati
    loss.backward()        # calcola i gradienti tramite backpropagation
    optimizer.step()       # aggiorna i pesi del modello

    # Ogni 10 epoche stampiamo la loss
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Grafico dell'andamento della loss
plt.plot(losses)
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.show()
```

*"Bene, qui abbiamo il cuore del training. Allora... ogni epoca ha tre fasi principali. Prima il forward pass: passo i dati al modello e ottengo le predizioni. Poi calcolo la loss confrontando quello che ha predetto il modello con i target veri. E qui... attenzione, una cosa importante: zero_grad() prima del backward pass. Perché PyTorch accumula i gradienti, e se non li azzero a ogni iterazione... beh, ottengo risultati completamente sbagliati. loss.backward() è dove avviene la magia della backpropagation - PyTorch calcola automaticamente tutti i gradienti attraverso la chain rule. E infine optimizer.step() applica gli aggiornamenti ai parametri. Vedrete che la loss dovrebbe diminuire progressivamente... se tutto va bene, naturalmente."*

## **Valutazione e Analisi**

```python
# Facciamo le predizioni finali senza calcolare i gradienti
# torch.no_grad() disattiva il tracciamento dei gradienti
with torch.no_grad():
    preds = model(X).round()  # round arrotonda le probabilità a 0 o 1

# Calcoliamo l'accuracy: numero di predizioni corrette / totale
accuracy = (preds.eq(y).sum().item()) / len(y)
print(f"Accuracy: {accuracy*100:.2f}%")
```

*"Per la valutazione... ecco, uso torch.no_grad() che disabilita il tracking dei gradienti. Questo è importante perché durante l'inference non mi servono i gradienti, e... beh, risparmio memoria e velocità. Poi arrotondo le probabilità a 0 o 1 con una soglia a 0.5 - che è la scelta standard - e calcolo l'accuracy. Ovviamente l'accuracy è una metrica semplificata, in scenari reali dovreste guardare anche precision, recall, F1-score... però per questo esempio va bene così."*

## **Visualizzazione della Decision Boundary**

```python
import numpy as np

# Definiamo una griglia di punti 2D che copra l'intero spazio dei dati
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Trasformiamo la griglia in tensore PyTorch
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# Facciamo predizioni sulla griglia
with torch.no_grad():
    Z = model(grid).reshape(xx.shape)

# Disegniamo la decision boundary
plt.contourf(xx, yy, Z, levels=[0,0.5,1], alpha=0.3, cmap="RdBu")
plt.scatter(class0[:,0], class0[:,1], c="blue", label="Classe 0")
plt.scatter(class1[:,0], class1[:,1], c="red", label="Classe 1")
plt.show()
```

*"Questa parte è... secondo me una delle più interessanti, perché ci fa vedere geometricamente cosa ha imparato il modello. Quello che faccio è creare una griglia fitta di punti che copre tutto lo spazio dei dati, poi... eh, faccio predizioni su ogni punto di questa griglia. Il risultato è questa visualizzazione dove si vede chiaramente la decision boundary - cioè la linea che separa le due classi. E vedrete che è una linea retta, perché... appunto, il perceptron implementa un classificatore lineare. Se avessimo dati non linearmente separabili, beh... dovremmo usare architetture più complesse."*

## **Conclusioni e Prospettive**

*"Ecco, questo è il nostro perceptron implementato da zero. Abbiamo visto... diciamo tutti i pezzi fondamentali: la teoria matematica, l'implementazione pratica, il training loop con backpropagation, e anche come visualizzare quello che ha imparato il modello. Ora, ovviamente questo è solo l'inizio, perché... beh, i modelli moderni sono molto più complessi. Però questi concetti - il forward pass, la loss function, l'ottimizzazione basata su gradiente - sono sempre gli stessi, anche nelle architetture più avanzate. Quindi... sì, padroneggiare il perceptron è fondamentale per capire tutto il resto del deep learning."*
