import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import MainLayout from "@/components/layout/MainLayout";
import { ArrowRight, BookOpen, Code, BarChart } from "lucide-react";
import { useEffect, useRef } from "react";

const AINetworkAnimation = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Ottimizzazione: cache delle dimensioni del canvas
    let canvasWidth = 0;
    let canvasHeight = 0;
    let centerX = 0;
    let centerY = 0;

    // Set canvas size ottimizzato
    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvasWidth = rect.width;
      canvasHeight = rect.height;
      centerX = canvasWidth / 2;
      centerY = canvasHeight / 2;
      
      const dpr = Math.min(window.devicePixelRatio || 1, 2); // Limita DPR per performance
      canvas.width = canvasWidth * dpr;
      canvas.height = canvasHeight * dpr;
      ctx.scale(dpr, dpr);
    };
    
    resizeCanvas();

    // Ottimizzazione: pre-calcolo delle posizioni dei nodi (statiche)
    const nodes = [
      // Input layer
      { x: centerX - 170, y: centerY - 110, layer: 0, active: false, baseActive: false },
      { x: centerX - 170, y: centerY - 37, layer: 0, active: false, baseActive: false },
      { x: centerX - 170, y: centerY + 37, layer: 0, active: false, baseActive: false },
      { x: centerX - 170, y: centerY + 110, layer: 0, active: false, baseActive: false },
      
      // Hidden layer 1
      { x: centerX - 57, y: centerY - 140, layer: 1, active: false, baseActive: false },
      { x: centerX - 57, y: centerY - 70, layer: 1, active: false, baseActive: false },
      { x: centerX - 57, y: centerY, layer: 1, active: false, baseActive: false },
      { x: centerX - 57, y: centerY + 70, layer: 1, active: false, baseActive: false },
      { x: centerX - 57, y: centerY + 140, layer: 1, active: false, baseActive: false },
      
      // Hidden layer 2
      { x: centerX + 57, y: centerY - 110, layer: 2, active: false, baseActive: false },
      { x: centerX + 57, y: centerY - 37, layer: 2, active: false, baseActive: false },
      { x: centerX + 57, y: centerY + 37, layer: 2, active: false, baseActive: false },
      { x: centerX + 57, y: centerY + 110, layer: 2, active: false, baseActive: false },
      
      // Output layer
      { x: centerX + 170, y: centerY - 37, layer: 3, active: false, baseActive: false },
      { x: centerX + 170, y: centerY + 37, layer: 3, active: false, baseActive: false },
    ];

    // Pre-calcolo delle connessioni con cache degli indici
    const connections: Array<{ 
      from: number; 
      to: number; 
      weight: number; 
      active: number;
      fromNode: typeof nodes[0];
      toNode: typeof nodes[0];
    }> = [];
    
    // Ottimizzazione: pre-calcolo e cache delle connessioni
    for (let i = 0; i < nodes.length; i++) {
      for (let j = 0; j < nodes.length; j++) {
        if (nodes[j].layer === nodes[i].layer + 1) {
          connections.push({
            from: i,
            to: j,
            weight: Math.random() * 0.8 + 0.2,
            active: 0,
            fromNode: nodes[i], // Cache riferimento
            toNode: nodes[j]    // Cache riferimento
          });
        }
      }
    }

    // Ottimizzazione: cache del gradiente (creato una sola volta)
    let backgroundGradient: CanvasGradient;
    const createBackgroundGradient = () => {
      backgroundGradient = ctx.createLinearGradient(0, 0, canvasWidth, canvasHeight);
      backgroundGradient.addColorStop(0, 'rgba(240, 240, 240, 0.05)');
      backgroundGradient.addColorStop(1, 'rgba(200, 200, 200, 0.05)');
    };
    createBackgroundGradient();

    // Ottimizzazione: variabili per evitare calcoli ripetuti
    let animationTime = 0;
    let lastPulseStart = 0;
    const PULSE_INTERVAL = 3; // Più lento
    const ANIMATION_SPEED = 0.01; // Ultra smooth
    const LAYER_DELAY = 0.4; // Più lento per propagazione graduale

    // Ottimizzazione: pre-calcolo dei layer nodes
    const nodesByLayer = [
      nodes.filter(n => n.layer === 0),
      nodes.filter(n => n.layer === 1),
      nodes.filter(n => n.layer === 2),
      nodes.filter(n => n.layer === 3)
    ];

    // Ottimizzazione: pre-calcolo delle connessioni per nodo
    const nodeConnections = nodes.map((_, nodeIndex) => 
      connections.filter(c => c.to === nodeIndex)
    );

    // Inizio immediato dell'animazione invece di aspettare 3 secondi
    const initializeFirstPulse = () => {
      nodesByLayer[0].forEach(node => {
        node.baseActive = Math.random() > 0.3;
        node.active = node.baseActive;
      });
    };
    initializeFirstPulse();

    const animate = () => {
      animationTime += ANIMATION_SPEED;
      
      // Clear canvas ottimizzato
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);
      
      // Background con gradiente pre-calcolato
      ctx.fillStyle = backgroundGradient;
      ctx.fillRect(0, 0, canvasWidth, canvasHeight);

      // Trigger pulse ottimizzato
      const timeSinceLastPulse = animationTime - lastPulseStart;
      if (timeSinceLastPulse >= PULSE_INTERVAL) {
        lastPulseStart = animationTime;
        
        // Reset rapido
        nodes.forEach(node => {
          node.active = false;
          node.baseActive = false;
        });
        connections.forEach(conn => conn.active = 0);
        
        // Attiva input nodes
        nodesByLayer[0].forEach(node => {
          node.baseActive = Math.random() > 0.3;
          node.active = node.baseActive;
        });
      }

      // Propagazione ottimizzata
      const currentPulseTime = timeSinceLastPulse;
      if (currentPulseTime < 3) { // Durata più lunga per effetto più graduale
        // Ottimizzazione: elabora solo layer necessari
        for (let layerIndex = 1; layerIndex < 4; layerIndex++) {
          const layerDelay = layerIndex * LAYER_DELAY;
          if (currentPulseTime > layerDelay) {
            const currentLayerNodes = nodesByLayer[layerIndex];
            
            currentLayerNodes.forEach((node, nodeIndexInLayer) => {
              const globalNodeIndex = nodesByLayer.slice(0, layerIndex).reduce((sum, layer) => sum + layer.length, 0) + nodeIndexInLayer;
              const inputConnections = nodeConnections[globalNodeIndex];
              const hasActiveInput = inputConnections.some(c => c.fromNode.active);
              
              if (hasActiveInput && Math.random() > 0.2) {
                node.active = true;
              }
            });
          }
        }

        // Attivazione connessioni ultra smooth
        connections.forEach(conn => {
          if (conn.fromNode.active && currentPulseTime > (conn.toNode.layer * LAYER_DELAY)) {
            conn.active = Math.min(1, conn.active + 0.015); // Ultra graduale
          } else {
            conn.active = Math.max(0, conn.active - 0.008); // Fade out lentissimo
          }
        });
      }

      // Disegno delle connessioni ottimizzato
      connections.forEach(conn => {
        const fromNode = conn.fromNode;
        const toNode = conn.toNode;
        
        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        ctx.lineTo(toNode.x, toNode.y);
        
        if (conn.active > 0.1) {
          // Connessioni attive
          const baseOpacity = 0.5;
          const activeOpacity = 0.95;
          const opacity = baseOpacity + (conn.active * (activeOpacity - baseOpacity));
          const thickness = 1.5 + (conn.active * 1.8);
          
          const grayValue = Math.floor(130 - (conn.active * 70));
          ctx.strokeStyle = `rgba(${grayValue}, ${grayValue}, ${grayValue}, ${opacity})`;
          ctx.lineWidth = thickness;
          ctx.stroke();
          
          // Particelle ottimizzate (solo per connessioni molto attive)
          if (conn.active > 0.5) {
            const progress = (animationTime * 0.8) % 1; // Particelle ultra smooth
            const particleX = fromNode.x + (toNode.x - fromNode.x) * progress;
            const particleY = fromNode.y + (toNode.y - fromNode.y) * progress;
            
            ctx.beginPath();
            ctx.arc(particleX, particleY, 1.5, 0, Math.PI * 2);
            const particleGray = Math.floor(90 - (conn.active * 30));
            ctx.fillStyle = `rgba(${particleGray}, ${particleGray}, ${particleGray}, ${conn.active * 0.8})`;
            ctx.fill();
          }
        } else {
          // Connessioni inattive sempre visibili
          ctx.strokeStyle = 'rgba(160, 160, 160, 0.25)'; // Grigio chiaro sottile
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      });

      // Disegno dei nodi ottimizzato
      nodes.forEach(node => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, 8, 0, Math.PI * 2);
        
        if (node.active) {
          const pulseIntensity = 0.7 + Math.sin(animationTime * 3) * 0.3; // Ultra smooth pulse
          const grayValue = Math.floor(90 - (pulseIntensity * 40));
          ctx.fillStyle = `rgba(${grayValue}, ${grayValue}, ${grayValue}, ${pulseIntensity})`;
          ctx.fill();
          
          // Glow ottimizzato (solo per nodi attivi)
          ctx.beginPath();
          ctx.arc(node.x, node.y, 14, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(70, 70, 70, ${pulseIntensity * 0.3})`;
          ctx.fill();
        } else {
          ctx.fillStyle = 'rgba(140, 140, 140, 0.7)'; // Più visibile
          ctx.fill();
          ctx.strokeStyle = 'rgba(120, 120, 120, 0.8)';
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    // Avvio immediato dell'animazione
    animate();

    // Resize handler ottimizzato con throttling
    let resizeTimeout: NodeJS.Timeout;
    const handleResize = () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        resizeCanvas();
        createBackgroundGradient();
      }, 150);
    };
    
    window.addEventListener('resize', handleResize);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      window.removeEventListener('resize', handleResize);
      clearTimeout(resizeTimeout);
    };
  }, []);

  return (
    <div className="relative w-full h-80">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};

const Home = () => {
  return (
    <MainLayout>
      {/* Hero Section */}
      <section className="py-20 px-4 md:px-6 bg-gradient-to-br from-primary/5 via-background to-secondary/5">
        <div className="container mx-auto max-w-5xl">
          <div className="flex flex-col md:flex-row items-center gap-8 md:gap-16">
            <div className="flex-1 space-y-6">
              <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
                Padroneggia il Machine Learning con
                <span className="text-primary"> Teoria e Pratica</span>
              </h1>
              <p className="text-lg text-muted-foreground max-w-md">
                Una piattaforma moderna per imparare i concetti del machine learning e applicarli con esercizi pratici di programmazione.
              </p>
              <div className="flex flex-wrap gap-4">
                <Button asChild size="lg" className="gap-2">
                  <Link to="/theory">
                    Inizia a Imparare <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
                <Button asChild variant="outline" size="lg">
                  <Link to="/practice">Esplora gli Esercizi Pratici</Link>
                </Button>
              </div>
            </div>
            <div className="flex-1 relative">
              <div className="relative z-10 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg shadow-xl p-6 card-gradient">
                <AINetworkAnimation />
              </div>
              <div className="absolute -top-4 -bottom-4 -left-4 -right-4 bg-gradient-to-br from-primary/20 to-secondary/20 rounded-lg -z-0"></div>
            </div>
          </div>
        </div>
      </section>

      {/* Learning Pipeline Section */}
      <section className="py-24 px-4 md:px-6 bg-muted/30">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">Il Tuo Percorso di Apprendimento</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Un approccio strutturato che ti accompagna dalla teoria alla pratica professionale
            </p>
          </div>

          <div className="relative">
            {/* Pipeline Steps */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
              {/* Animated connecting line */}
              <div className="hidden md:block absolute top-1/2 left-1/4 right-1/4 h-0.5 bg-gradient-to-r from-primary/20 via-primary/60 to-primary/20 transform -translate-y-1/2">
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-primary to-transparent opacity-0 animate-pulse"></div>
              </div>

              {/* Step 1 */}
              <div className="relative bg-card rounded-xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 border border-primary/10">
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center text-white font-bold text-sm">1</div>
                </div>
                <div className="text-center pt-4">
                  <div className="h-16 w-16 bg-primary/10 rounded-lg flex items-center justify-center mb-6 mx-auto">
                    <BookOpen className="h-8 w-8 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold mb-3">Teoria Solida</h3>
                  <p className="text-muted-foreground">
                    Comprendi i fondamenti matematici e i concetti chiave del machine learning
                  </p>
                </div>
              </div>

              {/* Step 2 */}
              <div className="relative bg-card rounded-xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 border border-primary/10">
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center text-white font-bold text-sm">2</div>
                </div>
                <div className="text-center pt-4">
                  <div className="h-16 w-16 bg-primary/10 rounded-lg flex items-center justify-center mb-6 mx-auto">
                    <Code className="h-8 w-8 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold mb-3">Pratica Guidata</h3>
                  <p className="text-muted-foreground">
                    Implementa gli algoritmi step-by-step con esercizi pratici in Python
                  </p>
                </div>
              </div>

              {/* Step 3 */}
              <div className="relative bg-card rounded-xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 border border-primary/10">
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center text-white font-bold text-sm">3</div>
                </div>
                <div className="text-center pt-4">
                  <div className="h-16 w-16 bg-primary/10 rounded-lg flex items-center justify-center mb-6 mx-auto">
                    <BarChart className="h-8 w-8 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold mb-3">Progetti Reali</h3>
                  <p className="text-muted-foreground">
                    Applica le competenze acquisite su dataset e problemi del mondo reale
                  </p>
                </div>
              </div>
            </div>

            {/* Animated arrows for mobile */}
            <div className="flex md:hidden justify-center my-6">
              <div className="flex flex-col items-center space-y-2">
                <ArrowRight className="h-6 w-6 text-primary rotate-90" />
                <ArrowRight className="h-6 w-6 text-primary rotate-90" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Differentiators Section */}
      <section className="py-20 px-4 md:px-6 bg-muted/30">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">Perché Machine Learn è Diverso</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Non il solito corso online: un approccio unico che pone al centro la comprensione profonda e l'applicazione pratica
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            {/* Left side - Traditional approach */}
            <div className="space-y-6">
              <div className="bg-card/50 rounded-lg p-6 border-l-4 border-red-300">
                <h4 className="font-semibold text-red-700 dark:text-red-400 mb-2">❌ Approccio Tradizionale</h4>
                <ul className="space-y-2 text-muted-foreground">
                  <li>• Teoria superficiale e frammentata</li>
                  <li>• Uso di librerie "black box" senza capire</li>
                  <li>• Esempi giocattolo disconnessi dalla realtà</li>
                  <li>• Focus su certificazioni invece che competenze</li>
                </ul>
              </div>
            </div>

            {/* Right side - Our approach */}
            <div className="space-y-6">
              <div className="bg-card/50 rounded-lg p-6 border-l-4 border-green-400">
                <h4 className="font-semibold text-green-700 dark:text-green-400 mb-2">✅ Il Nostro Approccio</h4>
                <ul className="space-y-2 text-muted-foreground">
                  <li>• <strong>Didattica cristallina:</strong> Spiegazioni step-by-step chiare e complete</li>
                  <li>• <strong>Implementazione da zero:</strong> Capisci ogni riga di codice che scrivi</li>
                  <li>• <strong>Matematica accessibile:</strong> Concetti complessi resi comprensibili</li>
                  <li>• <strong>Progetti concreti:</strong> Risolvi problemi reali dell'industria</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Bottom highlight */}
          <div className="mt-16 text-center">
            <div className="bg-gradient-to-r from-primary/10 to-secondary/10 rounded-xl p-8 border border-primary/20">
              <h3 className="text-2xl font-bold mb-4">La Differenza che Fa la Differenza</h3>
              <p className="text-lg text-muted-foreground max-w-3xl mx-auto leading-relaxed">
                Mentre altri corsi ti insegnano <em>"come usare"</em> le librerie, noi ti insegniamo <em>"come funzionano"</em>. 
                Questa comprensione profonda ti rende un vero esperto, capace di innovare e risolvere problemi complessi autonomamente.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Technologies Section */}
      <section className="py-24 px-4 md:px-6">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-6">Tecnologie e Strumenti Professionali</h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              Impara utilizzando le stesse tecnologie e librerie usate dai professionisti del settore. 
              Non librerie didattiche, ma gli stessi strumenti che troverai nelle aziende più innovative.
            </p>
          </div>

          {/* Main Technologies Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-16">
            <div className="bg-card rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-blue-200 dark:border-blue-800">
              <div className="flex items-center mb-4">
                <div className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-950 dark:to-blue-900 px-3 py-1 rounded-full border border-blue-200 dark:border-blue-800">
                  <span className="text-blue-700 dark:text-blue-300 font-bold">Python</span>
                </div>
              </div>
              <h4 className="font-semibold mb-2">Linguaggio Base</h4>
              <p className="text-muted-foreground text-sm">
                Il linguaggio più popolare per data science e ML. Impari pattern avanzati e best practices per scrivere codice professionale.
              </p>
            </div>

            <div className="bg-card rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-orange-200 dark:border-orange-800">
              <div className="flex items-center mb-4">
                <div className="bg-gradient-to-r from-orange-50 to-orange-100 dark:from-orange-950 dark:to-orange-900 px-3 py-1 rounded-full border border-orange-200 dark:border-orange-800">
                  <span className="text-orange-700 dark:text-orange-300 font-bold">PyTorch</span>
                </div>
              </div>
              <h4 className="font-semibold mb-2">Deep Learning</h4>
              <p className="text-muted-foreground text-sm">
                Framework di Meta per neural networks. Costruisci reti neurali da zero e comprendi ogni componente: tensori, autograd, optimizers.
              </p>
            </div>

            <div className="bg-card rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-green-200 dark:border-green-800">
              <div className="flex items-center mb-4">
                <div className="bg-gradient-to-r from-green-50 to-green-100 dark:from-green-950 dark:to-green-900 px-3 py-1 rounded-full border border-green-200 dark:border-green-800">
                  <span className="text-green-700 dark:text-green-300 font-bold">NumPy</span>
                </div>
              </div>
              <h4 className="font-semibold mb-2">Calcolo Numerico</h4>
              <p className="text-muted-foreground text-sm">
                La base di tutto: array multidimensionali, algebra lineare, operazioni vettoriali. Capisci come funzionano i calcoli sotto al cofano.
              </p>
            </div>

            <div className="bg-card rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-purple-200 dark:border-purple-800">
              <div className="flex items-center mb-4">
                <div className="bg-gradient-to-r from-purple-50 to-purple-100 dark:from-purple-950 dark:to-purple-900 px-3 py-1 rounded-full border border-purple-200 dark:border-purple-800">
                  <span className="text-purple-700 dark:text-purple-300 font-bold">Pandas</span>
                </div>
              </div>
              <h4 className="font-semibold mb-2">Manipolazione Dati</h4>
              <p className="text-muted-foreground text-sm">
                Lo standard per data analysis. Impari tecniche avanzate di cleaning, transformation e feature engineering su dataset reali.
              </p>
            </div>

            <div className="bg-card rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-red-200 dark:border-red-800">
              <div className="flex items-center mb-4">
                <div className="bg-gradient-to-r from-red-50 to-red-100 dark:from-red-950 dark:to-red-900 px-3 py-1 rounded-full border border-red-200 dark:border-red-800">
                  <span className="text-red-700 dark:text-red-300 font-bold">Matplotlib</span>
                </div>
              </div>
              <h4 className="font-semibold mb-2">Visualizzazione</h4>
              <p className="text-muted-foreground text-sm">
                Crea grafici professionali e dashboard interattive. Impari a comunicare i risultati in modo efficace e convincente.
              </p>
            </div>

            <div className="bg-card rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-indigo-200 dark:border-indigo-800">
              <div className="flex items-center mb-4">
                <div className="bg-gradient-to-r from-indigo-50 to-indigo-100 dark:from-indigo-950 dark:to-indigo-900 px-3 py-1 rounded-full border border-indigo-200 dark:border-indigo-800">
                  <span className="text-indigo-700 dark:text-indigo-300 font-bold">Seaborn</span>
                </div>
              </div>
              <h4 className="font-semibold mb-2">Analisi Statistica</h4>
              <p className="text-muted-foreground text-sm">
                Visualizzazioni statistiche avanzate. Perfect per exploratory data analysis e per identificare pattern nei dati.
              </p>
            </div>
          </div>

          {/* Stats Section */}
          <div className="bg-gradient-to-r from-primary/5 to-secondary/5 rounded-2xl p-8 mb-8">
            <h3 className="text-2xl font-bold text-center mb-8">Quello che Costruirai</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="text-4xl font-bold text-primary mb-3">50+</div>
                <div className="text-xl font-semibold mb-3">Algoritmi da Zero</div>
                <p className="text-muted-foreground">
                  Implementi oltre 50 algoritmi di ML e data science partendo dai principi matematici. 
                  Nessuna "magia nera": capisci ogni riga di codice.
                </p>
              </div>
              
              <div className="text-center">
                <div className="text-4xl font-bold text-primary mb-3">100+</div>
                <div className="text-xl font-semibold mb-3">Esercizi Progressivi</div>
                <p className="text-muted-foreground">
                  Centinaia di esercizi che ti accompagnano dai concetti base a implementazioni complesse. 
                  Ogni esercizio costruisce sul precedente.
                </p>
              </div>
              
              <div className="text-center">
                <div className="text-4xl font-bold text-primary mb-3">20+</div>
                <div className="text-xl font-semibold mb-3">Dataset Reali</div>
                <p className="text-muted-foreground">
                  Lavori su dataset e problemi dell'industria: finanza, healthcare, computer vision, NLP. 
                  Esperienza pratica su casi reali.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 md:px-6">
        <div className="container mx-auto max-w-5xl text-center">
          <h2 className="text-3xl font-bold mb-4">Pronto per Iniziare il Tuo Viaggio nel ML?</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
            Inizia subito ad accedere a tutti gli esercizi pratici e alla teoria disponibili.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Button asChild size="lg">
              <Link to="/theory">Esplora la Teoria</Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link to="/practice">Esplora Prima la Pratica</Link>
            </Button>
          </div>
        </div>
      </section>
    </MainLayout>
  );
};

export default Home;