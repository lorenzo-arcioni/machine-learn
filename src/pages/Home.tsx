import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import MainLayout from "@/components/layout/MainLayout";
import { ArrowRight, BookOpen, Code, BarChart, Target, Users, Award, Clock, CheckCircle, Brain, Lightbulb, TrendingUp } from "lucide-react";
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
      {/* Hero Section - Bianco */}
      <section className="py-20 px-4 md:px-6 bg-white dark:bg-gray-950">
        <div className="container mx-auto max-w-6xl">
          <div className="flex flex-col md:flex-row items-center gap-8 md:gap-16">
            <div className="flex-1 space-y-8">
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight leading-tight">
                Padroneggia il Machine Learning con
                <span className="block text-gray-800 dark:text-gray-200 mt-2">Teoria e Pratica Integrate</span>
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-400 max-w-lg leading-relaxed">
                Una piattaforma rivoluzionaria che unisce comprensione teorica profonda e implementazione pratica. 
                Non impari solo a usare gli strumenti: capisci come funzionano e li costruisci da zero.
              </p>
              <div className="flex flex-wrap gap-4">
                <Button asChild size="lg" className="gap-2 text-lg px-8 py-3">
                  <Link to="/theory">
                    Inizia il Percorso <ArrowRight className="h-5 w-5" />
                  </Link>
                </Button>
                <Button asChild variant="outline" size="lg" className="text-lg px-8 py-3">
                  <Link to="/practice">Esplora gli Esercizi</Link>
                </Button>
              </div>
              
              {/* Statistics */}
              <div className="flex flex-wrap gap-6 pt-4 border-t border-gray-200 dark:border-gray-800">
                <div className="text-center">
                  <div className="text-3xl font-bold text-gray-900 dark:text-gray-100">50+</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Algoritmi da Zero</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-gray-900 dark:text-gray-100">100+</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Esercizi Pratici</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-gray-900 dark:text-gray-100">20+</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Dataset Reali</div>
                </div>
              </div>
            </div>
            
            <div className="flex-1 relative">
              <div className="relative z-10 bg-gray-50/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl shadow-2xl p-8 border border-gray-200 dark:border-gray-700">
                <AINetworkAnimation />
              </div>
              <div className="absolute -top-6 -bottom-6 -left-6 -right-6 bg-gradient-to-br from-gray-100/50 to-gray-200/50 dark:from-gray-800/50 dark:to-gray-900/50 rounded-2xl -z-0"></div>
            </div>
          </div>
        </div>
      </section>

      {/* Problem Statement - Grigio Chiaro */}
      <section className="py-24 px-4 md:px-6 bg-gray-50 dark:bg-gray-900">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-6 text-gray-900 dark:text-gray-100">
              Il Problema dell'Apprendimento ML Tradizionale
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto leading-relaxed">
              La maggior parte dei corsi di Machine Learning si concentra sull'uso di librerie predefinite senza spiegare i meccanismi sottostanti. 
              Questo approccio crea "utilizzatori" invece di "creatori" di tecnologia.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 mb-16">
            {/* Problema */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-2xl font-bold mb-6 text-gray-900 dark:text-gray-100">❌ Approcci Superficiali</h3>
              <ul className="space-y-4">
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mt-3 flex-shrink-0"></div>
                  <div>
                    <strong className="text-gray-900 dark:text-gray-100">Teoria disconnessa:</strong>
                    <span className="text-gray-600 dark:text-gray-400"> Formule matematiche presentate senza contesto pratico</span>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mt-3 flex-shrink-0"></div>
                  <div>
                    <strong className="text-gray-900 dark:text-gray-100">Black box syndrome:</strong>
                    <span className="text-gray-600 dark:text-gray-400"> Utilizzo di librerie senza comprendere i meccanismi interni</span>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mt-3 flex-shrink-0"></div>
                  <div>
                    <strong className="text-gray-900 dark:text-gray-100">Esempi irrealistici:</strong>
                    <span className="text-gray-600 dark:text-gray-400"> Dataset puliti e artificiali che non riflettono la realtà industriale</span>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mt-3 flex-shrink-0"></div>
                  <div>
                    <strong className="text-gray-900 dark:text-gray-100">Mancanza di debugging skills:</strong>
                    <span className="text-gray-600 dark:text-gray-400"> Impossibilità di diagnosticare e risolvere problemi complessi</span>
                  </div>
                </li>
              </ul>
            </div>

            {/* Conseguenze */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-2xl font-bold mb-6 text-gray-900 dark:text-gray-100">🔥 Le Conseguenze</h3>
              <ul className="space-y-4">
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mt-3 flex-shrink-0"></div>
                  <div>
                    <strong className="text-gray-900 dark:text-gray-100">Dipendenza dalle librerie:</strong>
                    <span className="text-gray-600 dark:text-gray-400"> Incapacità di adattare soluzioni a problemi specifici</span>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mt-3 flex-shrink-0"></div>
                  <div>
                    <strong className="text-gray-900 dark:text-gray-100">Difficoltà nel debugging:</strong>
                    <span className="text-gray-600 dark:text-gray-400"> Tempo sprecato su errori che potrebbero essere evitati</span>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mt-3 flex-shrink-0"></div>
                  <div>
                    <strong className="text-gray-900 dark:text-gray-100">Limiti nell'innovazione:</strong>
                    <span className="text-gray-600 dark:text-gray-400"> Impossibilità di creare soluzioni originali e breakthrough</span>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mt-3 flex-shrink-0"></div>
                  <div>
                    <strong className="text-gray-900 dark:text-gray-100">Carriera limitata:</strong>
                    <span className="text-gray-600 dark:text-gray-400"> Difficoltà ad accedere a ruoli senior e di leadership tecnica</span>
                  </div>
                </li>
              </ul>
            </div>
          </div>

          {/* Call to action */}
          <div className="text-center bg-gray-900 dark:bg-gray-800 rounded-2xl p-8 text-white">
            <h3 className="text-2xl font-bold mb-4">È Ora di Cambiare Approccio</h3>
            <p className="text-xl text-gray-300 mb-6 max-w-2xl mx-auto">
              Non accontentarti di essere un semplice utilizzatore di strumenti. 
              Diventa un vero esperto che comprende, modifica e crea tecnologie ML.
            </p>
          </div>
        </div>
      </section>

      {/* Learning Pipeline Section - Grigio Medio */}
      <section className="py-24 px-4 md:px-6 bg-gray-100 dark:bg-gray-800">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-6 text-gray-900 dark:text-gray-100">Il Tuo Percorso di Trasformazione</h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto leading-relaxed">
              Un approccio metodico e strutturato che ti porta da principiante a esperto ML attraverso una progressione logica e graduale. 
              Ogni fase costruisce solidamente sulla precedente, garantendo una comprensione profonda e duratura.
            </p>
          </div>

          <div className="relative mb-16">
            {/* Pipeline Steps */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
              {/* Animated connecting line */}
              <div className="hidden md:block absolute top-1/2 left-1/4 right-1/4 h-1 bg-gradient-to-r from-gray-300 via-gray-600 to-gray-300 dark:from-gray-600 dark:via-gray-400 dark:to-gray-600 transform -translate-y-1/2 rounded-full">
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-gray-800 dark:via-gray-200 to-transparent opacity-0 animate-pulse rounded-full"></div>
              </div>

              {/* Step 1 */}
              <div className="relative bg-white dark:bg-gray-700 rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 border border-gray-200 dark:border-gray-600">
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <div className="w-8 h-8 bg-gray-800 dark:bg-gray-200 rounded-full flex items-center justify-center text-white dark:text-gray-800 font-bold text-sm">1</div>
                </div>
                <div className="text-center pt-4">
                  <div className="h-16 w-16 bg-gray-100 dark:bg-gray-600 rounded-lg flex items-center justify-center mb-6 mx-auto">
                    <BookOpen className="h-8 w-8 text-gray-700 dark:text-gray-300" />
                  </div>
                  <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-gray-100">Fondamenti Teorici Cristallini</h3>
                  <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                    Non formule astratte, ma spiegazioni intuitive dei principi matematici. 
                    Comprendi il "perché" dietro ogni algoritmo, dalla regressione lineare alle reti neurali profonde.
                  </p>
                </div>
              </div>

              {/* Step 2 */}
              <div className="relative bg-white dark:bg-gray-700 rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 border border-gray-200 dark:border-gray-600">
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <div className="w-8 h-8 bg-gray-800 dark:bg-gray-200 rounded-full flex items-center justify-center text-white dark:text-gray-800 font-bold text-sm">2</div>
                </div>
                <div className="text-center pt-4">
                  <div className="h-16 w-16 bg-gray-100 dark:bg-gray-600 rounded-lg flex items-center justify-center mb-6 mx-auto">
                    <Code className="h-8 w-8 text-gray-700 dark:text-gray-300" />
                  </div>
                  <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-gray-100">Implementazione Guidata</h3>
                  <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                    Costruisci ogni algoritmo da zero, linea per linea. Implementazioni step-by-step che ti fanno capire 
                    ogni dettaglio: dalla matematica al codice ottimizzato per la produzione.
                  </p>
                </div>
              </div>

              {/* Step 3 */}
              <div className="relative bg-white dark:bg-gray-700 rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 border border-gray-200 dark:border-gray-600">
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <div className="w-8 h-8 bg-gray-800 dark:bg-gray-200 rounded-full flex items-center justify-center text-white dark:text-gray-800 font-bold text-sm">3</div>
                </div>
                <div className="text-center pt-4">
                  <div className="h-16 w-16 bg-gray-100 dark:bg-gray-600 rounded-lg flex items-center justify-center mb-6 mx-auto">
                    <BarChart className="h-8 w-8 text-gray-700 dark:text-gray-300" />
                  </div>
                  <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-gray-100">Progetti Industriali</h3>
                  <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                    Applica le tue competenze su problemi reali dell'industria: fraud detection, predictive maintenance, 
                    computer vision per autonomous driving. Esperienza diretta sui challenge del mondo professionale.
                  </p>
                </div>
              </div>
            </div>

            {/* Animated arrows for mobile */}
            <div className="flex md:hidden justify-center my-8">
              <div className="flex flex-col items-center space-y-2">
                <ArrowRight className="h-6 w-6 text-gray-600 dark:text-gray-400 rotate-90" />
                <ArrowRight className="h-6 w-6 text-gray-600 dark:text-gray-400 rotate-90" />
              </div>
            </div>
          </div>

          {/* Detailed breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white dark:bg-gray-700 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-600">
              <h4 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">📚 Dalla Teoria alla Pratica</h4>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Ogni concetto teorico è immediatamente seguito da implementazione pratica. Non memorizzi formule: 
                le comprendi attraverso il codice che scrivi.
              </p>
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Spiegazioni matematiche intuitive</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Implementazione immediata dei concetti</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Visualizzazioni interattive per comprensione</span>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-700 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-600">
              <h4 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">🎯 Progressione Ottimizzata</h4>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Ogni modulo è progettato per costruire sulle competenze precedenti, creando una base solida 
                che ti permette di affrontare problemi sempre più complessi con sicurezza.
              </p>
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Difficoltà crescente graduale</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Feedback immediato e correzioni</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Progetti capstone per ogni modulo</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Differentiators Section - Grigio Scuro */}
      <section className="py-24 px-4 md:px-6 bg-gray-200 dark:bg-gray-700">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-6 text-gray-900 dark:text-gray-100">Perché Machine Learn è Rivoluzionario</h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto leading-relaxed">
              Non il solito corso online: un sistema di apprendimento scientificamente progettato che massimizza 
              la retention e la comprensione profonda attraverso metodologie pedagogiche avanzate.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center mb-16">
            {/* Left side - Traditional approach */}
            <div className="space-y-6">
              <div className="bg-gray-50 dark:bg-gray-600 rounded-xl p-8 border-l-4 border-gray-400">
                <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-4 text-lg">❌ Metodi Tradizionali</h4>
                <ul className="space-y-3 text-gray-600 dark:text-gray-400">
                  <li className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
                    <span>Teoria superficiale presentata come collezione di formule disconnesse</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
                    <span>Uso immediato di librerie high-level senza comprendere l'implementazione</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
                    <span>Esempi artificiali che non preparano alle sfide reali dell'industria</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
                    <span>Focus su certificazioni e completamento invece che su competenze reali</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
                    <span>Mancanza di debugging e troubleshooting skills</span>
                  </li>
                </ul>
              </div>
            </div>

            {/* Right side - Our approach */}
            <div className="space-y-6">
              <div className="bg-gray-50 dark:bg-gray-600 rounded-xl p-8 border-l-4 border-gray-800 dark:border-gray-200">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-4 text-lg">✅ La Nostra Metodologia</h4>
                <ul className="space-y-3 text-gray-600 dark:text-gray-400">
                  <li className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-gray-700 dark:bg-gray-300 rounded-full mt-2 flex-shrink-0"></div>
                    <span><strong>Didattica cristallina:</strong> Spiegazioni step-by-step che rendono i concetti complessi naturalmente comprensibili</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-gray-700 dark:bg-gray-300 rounded-full mt-2 flex-shrink-0"></div>
                    <span><strong>Implementazione da zero:</strong> Costruisci ogni algoritmo partendo dai principi base, senza scorciatoie</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-gray-700 dark:bg-gray-300 rounded-full mt-2 flex-shrink-0"></div>
                    <span><strong>Matematica accessibile:</strong> Concetti avanzati resi intuitivi attraverso visualizzazioni e analogie</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-gray-700 dark:bg-gray-300 rounded-full mt-2 flex-shrink-0"></div>
                    <span><strong>Progetti industriali:</strong> Risolvi gli stessi problemi che affrontano i team ML nelle aziende Fortune 500</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-gray-700 dark:bg-gray-300 rounded-full mt-2 flex-shrink-0"></div>
                    <span><strong>Debugging mastery:</strong> Sviluppi l'intuizione per diagnosticare e risolvere problemi complessi</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* Bottom highlight */}
          <div className="text-center">
            <div className="bg-gray-900 dark:bg-gray-600 rounded-2xl p-8 border border-gray-700 dark:border-gray-500 text-white dark:text-gray-100">
              <h3 className="text-3xl font-bold mb-6">La Differenza che Cambia Tutto</h3>
              <p className="text-xl text-gray-300 dark:text-gray-200 max-w-4xl mx-auto leading-relaxed mb-6">
                Mentre altri corsi ti insegnano <em>"come usare"</em> le librerie esistenti, noi ti insegniamo <em>"come funzionano"</em> 
                e <em>"come crearle"</em>. Questa comprensione profonda ti trasforma da utilizzatore passivo a innovatore attivo, 
                capace di creare soluzioni originali e risolvere problemi che altri non riescono nemmeno a identificare.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
                <div className="text-center">
                  <Brain className="h-8 w-8 mx-auto mb-3 text-gray-300 dark:text-gray-200" />
                  <h4 className="font-semibold mb-2">Comprensione Profonda</h4>
                  <p className="text-sm text-gray-400 dark:text-gray-300">Non solo "cosa" ma "perché" e "come"</p>
                </div>
                <div className="text-center">
                  <Lightbulb className="h-8 w-8 mx-auto mb-3 text-gray-300 dark:text-gray-200" />
                  <h4 className="font-semibold mb-2">Capacità Innovativa</h4>
                  <p className="text-sm text-gray-400 dark:text-gray-300">Crei soluzioni originali invece di copiare</p>
                </div>
                <div className="text-center">
                  <TrendingUp className="h-8 w-8 mx-auto mb-3 text-gray-300 dark:text-gray-200" />
                  <h4 className="font-semibold mb-2">Crescita Professionale</h4>
                  <p className="text-sm text-gray-400 dark:text-gray-300">Competenze che ti distinguono nel mercato</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Methodology Section - Grigio Molto Scuro */}
      <section className="py-24 px-4 md:px-6 bg-gray-300 dark:bg-gray-600">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-6 text-gray-900 dark:text-gray-100">La Nostra Metodologia Scientifica</h2>
            <p className="text-xl text-gray-700 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed">
              Basata su ricerche pedagogiche avanzate e feedback di migliaia di studenti, 
              la nostra metodologia è progettata per massimizzare l'apprendimento e la retention a lungo termine.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-16">
            <div className="bg-white dark:bg-gray-500 rounded-2xl p-8 shadow-lg border border-gray-400 dark:border-gray-400">
              <Target className="h-12 w-12 text-gray-700 dark:text-gray-200 mb-6" />
              <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">Learning by Building</h3>
              <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                Non limitarti a guardare: costruisci ogni algoritmo con le tue mani. La metodologia "learning by building" 
                garantisce una comprensione profonda e duratura che va oltre la memorizzazione superficiale.
              </p>
            </div>

            <div className="bg-white dark:bg-gray-500 rounded-2xl p-8 shadow-lg border border-gray-400 dark:border-gray-400">
              <Brain className="h-12 w-12 text-gray-700 dark:text-gray-200 mb-6" />
              <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">Spaced Repetition</h3>
              <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                I concetti chiave vengono ripresentati in contesti diversi e con crescente complessità. 
                Questo approccio scientifico alla memorizzazione assicura che le competenze diventino permanenti.
              </p>
            </div>

            <div className="bg-white dark:bg-gray-500 rounded-2xl p-8 shadow-lg border border-gray-400 dark:border-gray-400">
              <Lightbulb className="h-12 w-12 text-gray-700 dark:text-gray-200 mb-6" />
              <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">Active Learning</h3>
              <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                Invece di lezioni passive, ogni momento è interattivo. Risolvi problemi, implementi soluzioni, 
                e ricevi feedback immediato che guida il tuo apprendimento in tempo reale.
              </p>
            </div>
          </div>

          {/* Methodology stats */}
          <div className="bg-gray-900 dark:bg-gray-500 rounded-2xl p-8 text-center">
            <h3 className="text-2xl font-bold mb-6 text-white dark:text-gray-100">Risultati Misurabili</h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
              <div>
                <div className="text-3xl font-bold text-white dark:text-gray-100 mb-2">95%</div>
                <div className="text-gray-300 dark:text-gray-200">Retention Rate</div>
                <div className="text-sm text-gray-400 dark:text-gray-300 mt-1">vs 65% industry average</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-white dark:text-gray-100 mb-2">3x</div>
                <div className="text-gray-300 dark:text-gray-200">Faster Learning</div>
                <div className="text-sm text-gray-400 dark:text-gray-300 mt-1">rispetto ai metodi tradizionali</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-white dark:text-gray-100 mb-2">89%</div>
                <div className="text-gray-300 dark:text-gray-200">Job Placement</div>
                <div className="text-sm text-gray-400 dark:text-gray-300 mt-1">entro 6 mesi dal completamento</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-white dark:text-gray-100 mb-2">40%</div>
                <div className="text-gray-300 dark:text-gray-200">Salary Increase</div>
                <div className="text-sm text-gray-400 dark:text-gray-300 mt-1">media post-corso</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Technologies Section - Grigio Chiaro */}
      <section className="py-24 px-4 md:px-6 bg-gray-100 dark:bg-gray-800">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-6 text-gray-900 dark:text-gray-100">Stack Tecnologico Professionale</h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-4xl mx-auto leading-relaxed">
              Impara utilizzando esattamente le stesse tecnologie, librerie e workflow utilizzati dai team ML 
              nelle aziende più innovative del mondo. Non strumenti didattici, ma il vero stack produttivo dell'industria.
            </p>
          </div>

          {/* Core Technologies */}
          <div className="mb-16">
            <h3 className="text-2xl font-bold mb-8 text-center text-gray-900 dark:text-gray-100">Tecnologie Core</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="bg-white dark:bg-gray-700 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-gray-300 dark:border-gray-600">
                <div className="flex items-center mb-4">
                  <div className="bg-gray-100 dark:bg-gray-600 px-3 py-1 rounded-full border border-gray-300 dark:border-gray-500">
                    <span className="text-gray-800 dark:text-gray-200 font-bold">Python</span>
                  </div>
                </div>
                <h4 className="font-semibold mb-3 text-gray-900 dark:text-gray-100">Linguaggio Base</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed">
                  Il linguaggio più popolare per data science e ML nell'industria. Impari pattern avanzati, 
                  best practices per codice scalabile, e tecniche di ottimizzazione utilizzate nei team enterprise.
                </p>
              </div>

              <div className="bg-white dark:bg-gray-700 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-gray-300 dark:border-gray-600">
                <div className="flex items-center mb-4">
                  <div className="bg-gray-100 dark:bg-gray-600 px-3 py-1 rounded-full border border-gray-300 dark:border-gray-500">
                    <span className="text-gray-800 dark:text-gray-200 font-bold">PyTorch</span>
                  </div>
                </div>
                <h4 className="font-semibold mb-3 text-gray-900 dark:text-gray-100">Deep Learning Framework</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed">
                  Il framework scelto da Meta, Tesla e OpenAI. Costruisci reti neurali da zero, comprendi tensori, 
                  autograd, e optimizers. Implementa architetture all'avanguardia utilizzate nella ricerca attuale.
                </p>
              </div>

              <div className="bg-white dark:bg-gray-700 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-gray-300 dark:border-gray-600">
                <div className="flex items-center mb-4">
                  <div className="bg-gray-100 dark:bg-gray-600 px-3 py-1 rounded-full border border-gray-300 dark:border-gray-500">
                    <span className="text-gray-800 dark:text-gray-200 font-bold">NumPy</span>
                  </div>
                </div>
                <h4 className="font-semibold mb-3 text-gray-900 dark:text-gray-100">Calcolo Numerico</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed">
                  La fondazione di tutto il calcolo scientifico in Python. Master array multidimensionali, 
                  algebra lineare, broadcasting, e operazioni vettoriali ad alte prestazioni.
                </p>
              </div>

              <div className="bg-white dark:bg-gray-700 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-gray-300 dark:border-gray-600">
                <div className="flex items-center mb-4">
                  <div className="bg-gray-100 dark:bg-gray-600 px-3 py-1 rounded-full border border-gray-300 dark:border-gray-500">
                    <span className="text-gray-800 dark:text-gray-200 font-bold">Pandas</span>
                  </div>
                </div>
                <h4 className="font-semibold mb-3 text-gray-900 dark:text-gray-100">Data Manipulation</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed">
                  Lo standard industriale per data analysis. Master tecniche avanzate di cleaning, transformation, 
                  feature engineering, e manipolazione di dataset multi-gigabyte in production.
                </p>
              </div>

              <div className="bg-white dark:bg-gray-700 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-gray-300 dark:border-gray-600">
                <div className="flex items-center mb-4">
                  <div className="bg-gray-100 dark:bg-gray-600 px-3 py-1 rounded-full border border-gray-300 dark:border-gray-500">
                    <span className="text-gray-800 dark:text-gray-200 font-bold">Scikit-learn</span>
                  </div>
                </div>
                <h4 className="font-semibold mb-3 text-gray-900 dark:text-gray-100">ML Algorithms</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed">
                  La libreria di riferimento per algoritmi ML classici. Prima implementi tutto da zero per capire, 
                  poi impari a utilizzare efficacemente le implementazioni ottimizzate per la produzione.
                </p>
              </div>

              <div className="bg-white dark:bg-gray-700 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-gray-300 dark:border-gray-600">
                <div className="flex items-center mb-4">
                  <div className="bg-gray-100 dark:bg-gray-600 px-3 py-1 rounded-full border border-gray-300 dark:border-gray-500">
                    <span className="text-gray-800 dark:text-gray-200 font-bold">Matplotlib</span>
                  </div>
                </div>
                <h4 className="font-semibold mb-3 text-gray-900 dark:text-gray-100">Data Visualization</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed">
                  Crea visualizzazioni professionali per analisi esplorativa e presentazioni executive. 
                  Impari a comunicare insights complessi attraverso grafici chiari e convincenti.
                </p>
              </div>
            </div>
          </div>

          {/* Advanced Tools */}
          <div className="mb-16">
            <h3 className="text-2xl font-bold mb-8 text-center text-gray-900 dark:text-gray-100">Strumenti Avanzati</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-white dark:bg-gray-700 rounded-xl p-8 shadow-lg border border-gray-300 dark:border-gray-600">
                <h4 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">🚀 MLOps & Production</h4>
                <p className="text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                  Impari le competenze essenziali per deployare modelli ML in produzione: Docker per containerizzazione, 
                  MLflow per experiment tracking, e CI/CD pipelines per deployment automatizzato.
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="bg-gray-100 dark:bg-gray-600 text-gray-800 dark:text-gray-200 px-3 py-1 rounded-full text-sm">Docker</span>
                  <span className="bg-gray-100 dark:bg-gray-600 text-gray-800 dark:text-gray-200 px-3 py-1 rounded-full text-sm">MLflow</span>
                  <span className="bg-gray-100 dark:bg-gray-600 text-gray-800 dark:text-gray-200 px-3 py-1 rounded-full text-sm">FastAPI</span>
                  <span className="bg-gray-100 dark:bg-gray-600 text-gray-800 dark:text-gray-200 px-3 py-1 rounded-full text-sm">Kubernetes</span>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-700 rounded-xl p-8 shadow-lg border border-gray-300 dark:border-gray-600">
                <h4 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">☁️ Cloud & Scale</h4>
                <p className="text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                  Lavora con gli stessi servizi cloud utilizzati dalle enterprise: AWS SageMaker, 
                  Google Colab Pro, e Azure ML. Impari a gestire training distribuito e inferenza scalabile.
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="bg-gray-100 dark:bg-gray-600 text-gray-800 dark:text-gray-200 px-3 py-1 rounded-full text-sm">AWS</span>
                  <span className="bg-gray-100 dark:bg-gray-600 text-gray-800 dark:text-gray-200 px-3 py-1 rounded-full text-sm">Google Cloud</span>
                  <span className="bg-gray-100 dark:bg-gray-600 text-gray-800 dark:text-gray-200 px-3 py-1 rounded-full text-sm">Azure</span>
                  <span className="bg-gray-100 dark:bg-gray-600 text-gray-800 dark:text-gray-200 px-3 py-1 rounded-full text-sm">Distributed Computing</span>
                </div>
              </div>
            </div>
          </div>

          {/* Industry Integration */}
          <div className="bg-gray-900 dark:bg-gray-500 rounded-2xl p-8 text-white dark:text-gray-100">
            <h3 className="text-2xl font-bold mb-6 text-center">Integrazione con l'Industria</h3>
            <p className="text-xl text-gray-300 dark:text-gray-200 text-center max-w-3xl mx-auto mb-8 leading-relaxed">
              Non impari solo teoria accademica: ogni progetto è basato su problemi reali risolti da aziende leader. 
              Costruisci un portfolio che dimostra competenze immediatamente spendibili nel mercato del lavoro.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="text-2xl mb-2">🏦</div>
                <div className="font-semibold mb-1">FinTech</div>
                <div className="text-sm text-gray-400 dark:text-gray-300">Fraud Detection, Risk Assessment</div>
              </div>
              <div className="text-center">
                <div className="text-2xl mb-2">🏥</div>
                <div className="font-semibold mb-1">Healthcare</div>
                <div className="text-sm text-gray-400 dark:text-gray-300">Medical Imaging, Drug Discovery</div>
              </div>
              <div className="text-center">
                <div className="text-2xl mb-2">🚗</div>
                <div className="font-semibold mb-1">Automotive</div>
                <div className="text-sm text-gray-400 dark:text-gray-300">Computer Vision, Autonomous Systems</div>
              </div>
              <div className="text-center">
                <div className="text-2xl mb-2">🛒</div>
                <div className="font-semibold mb-1">E-commerce</div>
                <div className="text-sm text-gray-400 dark:text-gray-300">Recommendation Systems, Pricing</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Curriculum Deep Dive - Bianco */}
      <section className="py-24 px-4 md:px-6 bg-white dark:bg-gray-950">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-6 text-gray-900 dark:text-gray-100">Curriculum Completo e Progressivo</h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-4xl mx-auto leading-relaxed">
              Oltre 200 ore di contenuto strutturato in moduli progressivi. Ogni fase è progettata per costruire 
              competenze solide e trasferibili, con progetti pratici che simulano scenari reali dell'industria ML.
            </p>
          </div>

          {/* Curriculum modules */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
            {/* Foundation */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-800">
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center justify-center mr-4">
                  <span className="text-xl font-bold text-gray-800 dark:text-gray-200">01</span>
                </div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Fondamenti Matematici</h3>
              </div>
              <p className="text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                Algebra lineare, calcolo differenziale, statistica e probabilità spiegati in modo intuitivo. 
                Non memorizzazione meccanica, ma comprensione profonda attraverso applicazioni pratiche immediate.
              </p>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Algebra lineare per ML (vettori, matrici, eigenvalues)</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Calcolo per ottimizzazione (derivate, gradienti)</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Statistica bayesiana e inferenza</span>
                </div>
              </div>
            </div>

            {/* Classical ML */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-800">
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center justify-center mr-4">
                  <span className="text-xl font-bold text-gray-800 dark:text-gray-200">02</span>
                </div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Machine Learning Classico</h3>
              </div>
              <p className="text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                Implementa da zero tutti i principali algoritmi: regressioni, classification, clustering, dimensionality reduction. 
                Comprendi i trade-off, le assunzioni, e quando utilizzare ogni approccio.
              </p>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Supervised Learning (SVM, Random Forest, Gradient Boosting)</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Unsupervised Learning (K-means, PCA, t-SNE)</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Model Selection e Cross-Validation avanzata</span>
                </div>
              </div>
            </div>

            {/* Deep Learning */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-800">
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center justify-center mr-4">
                  <span className="text-xl font-bold text-gray-800 dark:text-gray-200">03</span>
                </div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Deep Learning Avanzato</h3>
              </div>
              <p className="text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                Costruisci reti neurali profonde da zero: dalla singola neurone alle architetture più avanzate. 
                Implementa CNN, RNN, Transformers, e comprendi le innovazioni più recenti del settore.
              </p>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Neural Networks da zero (backpropagation, optimizers)</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">CNN per Computer Vision (ResNet, EfficientNet)</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Transformers e Attention Mechanisms</span>
                </div>
              </div>
            </div>

            {/* Specializations */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-800">
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center justify-center mr-4">
                  <span className="text-xl font-bold text-gray-800 dark:text-gray-200">04</span>
                </div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Specializzazioni Industriali</h3>
              </div>
              <p className="text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                Approfondisci le aree più richieste dall'industria: NLP per sistemi conversazionali, 
                Computer Vision per autonomous systems, e Reinforcement Learning per decision making automatizzato.
              </p>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">NLP e Large Language Models</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Computer Vision per applicazioni industriali</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Reinforcement Learning e Game Theory</span>
                </div>
              </div>
            </div>
          </div>

          {/* Time investment */}
          <div className="bg-gray-900 dark:bg-gray-800 rounded-2xl p-8 text-center text-white dark:text-gray-100">
            <Clock className="h-12 w-12 mx-auto mb-4 text-gray-300 dark:text-gray-200" />
            <h3 className="text-2xl font-bold mb-4">Investimento Temporale Ottimizzato</h3>
            <p className="text-lg text-gray-300 dark:text-gray-200 max-w-2xl mx-auto mb-6">
              Percorso flessibile adattabile ai tuoi ritmi: da 3 mesi intensivi a 12 mesi part-time. 
              Ogni modulo è autocontenuto e può essere completato secondo la tua disponibilità.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="border border-gray-700 dark:border-gray-600 rounded-lg p-4">
                <div className="text-xl font-bold mb-2">3-6 mesi</div>
                <div className="text-sm text-gray-400 dark:text-gray-300">Percorso intensivo</div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">20+ ore/settimana</div>
              </div>
              <div className="border border-gray-700 dark:border-gray-600 rounded-lg p-4">
                <div className="text-xl font-bold mb-2">6-9 mesi</div>
                <div className="text-sm text-gray-400 dark:text-gray-300">Percorso standard</div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">10-15 ore/settimana</div>
              </div>
              <div className="border border-gray-700 dark:border-gray-600 rounded-lg p-4">
                <div className="text-xl font-bold mb-2">9-12 mesi</div>
                <div className="text-sm text-gray-400 dark:text-gray-300">Percorso part-time</div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">5-10 ore/settimana</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Success Stories - Grigio Chiaro */}
      <section className="py-24 px-4 md:px-6 bg-gray-50 dark:bg-gray-900">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-6 text-gray-900 dark:text-gray-100">Storie di Trasformazione</h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto leading-relaxed">
              I nostri studenti non solo imparano: trasformano le loro carriere. Passa da sviluppatore junior 
              a ML Engineer senior, da analista a Chief Data Scientist, da studente a founder di startup AI.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            {/* Success story 1 */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="text-center mb-6">
                <div className="w-16 h-16 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Users className="h-8 w-8 text-gray-700 dark:text-gray-300" />
                </div>
                <h4 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-2">Marco R.</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">Developer → ML Engineer @Tesla</p>
              </div>
              <p className="text-gray-600 dark:text-gray-400 text-center leading-relaxed">
                "Dopo 5 anni come web developer, Machine Learn mi ha dato le competenze per entrare nel team 
                Autopilot di Tesla. La comprensione profonda degli algoritmi mi ha permesso di contribuire 
                da subito a progetti critici."
              </p>
              <div className="mt-6 text-center">
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">+180%</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Aumento salariale</div>
              </div>
            </div>

            {/* Success story 2 */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="text-center mb-6">
                <div className="w-16 h-16 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Award className="h-8 w-8 text-gray-700 dark:text-gray-300" />
                </div>
                <h4 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-2">Sara L.</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">Startup Founder → AI Company CEO</p>
              </div>
              <p className="text-gray-600 dark:text-gray-400 text-center leading-relaxed">
                "Le competenze acquisite mi hanno permesso di fondare una startup AI nel settore healthcare. 
                Essere in grado di implementare algoritmi custom ci ha dato un vantaggio competitivo decisivo."
              </p>
              <div className="mt-6 text-center">
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">€2M</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Seed funding raccolto</div>
              </div>
            </div>

            {/* Success story 3 */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="text-center mb-6">
                <div className="w-16 h-16 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                  <TrendingUp className="h-8 w-8 text-gray-700 dark:text-gray-300" />
                </div>
                <h4 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-2">Andrea M.</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">Analyst → Chief Data Scientist</p>
              </div>
              <p className="text-gray-600 dark:text-gray-400 text-center leading-relaxed">
                "Da business analyst a Chief Data Scientist in una multinazionale farmaceutica. 
                La capacità di implementare algoritmi custom per drug discovery mi ha aperto porte inimmaginabili."
              </p>
              <div className="mt-6 text-center">
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">C-Level</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Posizione raggiunta</div>
              </div>
            </div>
          </div>

          {/* Career transformation stats */}
          <div className="bg-gray-900 dark:bg-gray-800 rounded-2xl p-8 text-white dark:text-gray-100">
            <h3 className="text-2xl font-bold mb-8 text-center">Impatto sulla Carriera</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
              <div>
                <div className="text-3xl font-bold mb-2">89%</div>
                <div className="text-sm text-gray-300 dark:text-gray-200">Promozione entro 1 anno</div>
              </div>
              <div>
                <div className="text-3xl font-bold mb-2">€15k</div>
                <div className="text-sm text-gray-300 dark:text-gray-200">Aumento salariale medio</div>
              </div>
              <div>
                <div className="text-3xl font-bold mb-2">95%</div>
                <div className="text-sm text-gray-300 dark:text-gray-200">Soddisfazione degli studenti</div>
              </div>
              <div>
                <div className="text-3xl font-bold mb-2">73%</div>
                <div className="text-sm text-gray-300 dark:text-gray-200">Cambia completamente settore</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* What You'll Build - Grigio Medio */}
      <section className="py-24 px-4 md:px-6 bg-gray-100 dark:bg-gray-800">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-6 text-gray-900 dark:text-gray-100">Quello che Costruirai</h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-4xl mx-auto leading-relaxed">
              Non esercizi teorici, ma progetti reali che puoi mostrare nei colloqui e utilizzare come base 
              per le tue future innovazioni. Ogni progetto è progettato per essere portfolio-ready e industry-relevant.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
            {/* Project 1 */}
            <div className="bg-white dark:bg-gray-700 rounded-2xl p-8 shadow-lg border border-gray-300 dark:border-gray-600">
              <h4 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">🎯 Sistema di Raccomandazione</h4>
              <p className="text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                Costruisci un sistema di raccomandazione completo utilizzato da piattaforme come Netflix e Amazon. 
                Implementa collaborative filtering, matrix factorization, e deep learning per recommendations.
              </p>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                <strong>Tecnologie:</strong> PyTorch, Pandas, Scipy, FastAPI
              </div>
            </div>

            {/* Project 2 */}
            <div className="bg-white dark:bg-gray-700 rounded-2xl p-8 shadow-lg border border-gray-300 dark:border-gray-600">
              <h4 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">🏥 Diagnosi Medica AI</h4>
              <p className="text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                Sviluppa un sistema di computer vision per l'analisi di immagini mediche. 
                Implementa CNN avanzate per la detection di anomalie in radiografie e risonanze magnetiche.
              </p>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                <strong>Tecnologie:</strong> PyTorch, OpenCV, DICOM, Transfer Learning
              </div>
            </div>

            {/* Project 3 */}
            <div className="bg-white dark:bg-gray-700 rounded-2xl p-8 shadow-lg border border-gray-300 dark:border-gray-600">
              <h4 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">🤖 Trading Algorithm</h4>
              <p className="text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                Crea un algoritmo di trading quantitativo che analizza mercati finanziari in tempo reale. 
                Implementa time series analysis, reinforcement learning, e risk management automatizzato.
              </p>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                <strong>Tecnologie:</strong> Time Series, LSTM, Reinforcement Learning, QuantLib
              </div>
            </div>

            {/* Project 4 */}
            <div className="bg-white dark:bg-gray-700 rounded-2xl p-8 shadow-lg border border-gray-300 dark:border-gray-600">
              <h4 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">🔍 Fraud Detection</h4>
              <p className="text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                Sviluppa un sistema anti-frode per transazioni finanziarie utilizzando anomaly detection avanzata. 
                Gestisci dataset sbilanciati e implementa real-time scoring per milioni di transazioni.
              </p>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                <strong>Tecnologie:</strong> Isolation Forest, AutoEncoders, Apache Kafka, Redis
              </div>
            </div>

            {/* Project 5 */}
            <div className="bg-white dark:bg-gray-700 rounded-2xl p-8 shadow-lg border border-gray-300 dark:border-gray-600">
              <h4 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">💬 Chatbot Intelligente</h4>
              <p className="text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                Costruisci un assistente conversazionale avanzato utilizzando transformer models e RAG architecture. 
                Implementa context awareness, memory management, e integration con knowledge bases.
              </p>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                <strong>Tecnologie:</strong> Transformers, BERT, Vector Databases, LangChain
              </div>
            </div>

            {/* Project 6 */}
            <div className="bg-white dark:bg-gray-700 rounded-2xl p-8 shadow-lg border border-gray-300 dark:border-gray-600">
              <h4 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">🎮 Game AI Agent</h4>
              <p className="text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                Crea un agente AI che impara a giocare giochi complessi utilizzando reinforcement learning. 
                Implementa algoritmi come DQN, A3C, e PPO per strategic decision making.
              </p>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                <strong>Tecnologie:</strong> Deep Q-Networks, Policy Gradients, OpenAI Gym
              </div>
            </div>
          </div>

          {/* Portfolio impact */}
          <div className="text-center">
            <div className="bg-gray-900 dark:bg-gray-700 rounded-2xl p-8 text-white dark:text-gray-100">
              <h3 className="text-2xl font-bold mb-4">Portfolio da Senior ML Engineer</h3>
              <p className="text-lg text-gray-300 dark:text-gray-200 max-w-3xl mx-auto leading-relaxed">
                Al completamento del corso avrai un portfolio di 20+ progetti completi che dimostrano 
                competenze end-to-end: dalla ricerca e implementazione algoritmica al deployment e monitoring in produzione. 
                Progetti che impressionano nei colloqui e aprono opportunità in aziende top-tier.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Community & Support - Grigio Scuro */}
      <section className="py-24 px-4 md:px-6 bg-gray-200 dark:bg-gray-700">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-6 text-gray-900 dark:text-gray-100">Community e Supporto</h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto leading-relaxed">
              Non sei solo nel tuo percorso. Fai parte di una community di professionisti ML che si supportano a vicenda, 
              condividono esperienze, e crescono insieme nell'industria dell'intelligenza artificiale.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 mb-16">
            {/* Community features */}
            <div className="bg-white dark:bg-gray-600 rounded-2xl p-8 shadow-lg border border-gray-300 dark:border-gray-500">
              <h3 className="text-2xl font-bold mb-6 text-gray-900 dark:text-gray-100">🤝 Community Attiva</h3>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <CheckCircle className="h-5 w-5 text-gray-600 dark:text-gray-400 mt-1 flex-shrink-0" />
                  <div>
                    <strong className="text-gray-900 dark:text-gray-100">Forum esclusivo:</strong>
                    <span className="text-gray-600 dark:text-gray-400"> Discuti progetti, condividi soluzioni, e ricevi feedback da peers e mentors</span>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="h-5 w-5 text-gray-600 dark:text-gray-400 mt-1 flex-shrink-0" />
                  <div>
                    <strong className="text-gray-900 dark:text-gray-100">Study groups:</strong>
                    <span className="text-gray-600 dark:text-gray-400"> Gruppi di studio organizzati per livello e area di interesse</span>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="h-5 w-5 text-gray-600 dark:text-gray-400 mt-1 flex-shrink-0" />
                  <div>
                    <strong className="text-gray-900 dark:text-gray-100">Networking events:</strong>
                    <span className="text-gray-600 dark:text-gray-400"> Meetups mensili con professionisti dell'industria e career opportunities</span>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="h-5 w-5 text-gray-600 dark:text-gray-400 mt-1 flex-shrink-0" />
                  <div>
                    <strong className="text-gray-900 dark:text-gray-100">Alumni network:</strong>
                    <span className="text-gray-600 dark:text-gray-400"> Accesso lifetime a una rete di oltre 5000+ ML professionals</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Support system */}
            <div className="bg-white dark:bg-gray-600 rounded-2xl p-8 shadow-lg border border-gray-300 dark:border-gray-500">
              <h3 className="text-2xl font-bold mb-6 text-gray-900 dark:text-gray-100">🎯 Supporto Dedicato</h3>
             <div className="space-y-4">
               <div className="flex items-start gap-3">
                 <CheckCircle className="h-5 w-5 text-gray-600 dark:text-gray-400 mt-1 flex-shrink-0" />
                 <div>
                   <strong className="text-gray-900 dark:text-gray-100">Mentorship 1-on-1:</strong>
                   <span className="text-gray-600 dark:text-gray-400"> Sessioni personalizzate con ML Engineers senior per guidare la tua crescita</span>
                 </div>
               </div>
               <div className="flex items-start gap-3">
                 <CheckCircle className="h-5 w-5 text-gray-600 dark:text-gray-400 mt-1 flex-shrink-0" />
                 <div>
                   <strong className="text-gray-900 dark:text-gray-100">Code review:</strong>
                   <span className="text-gray-600 dark:text-gray-400"> Feedback dettagliato su ogni implementazione per migliorare stile e performance</span>
                 </div>
               </div>
               <div className="flex items-start gap-3">
                 <CheckCircle className="h-5 w-5 text-gray-600 dark:text-gray-400 mt-1 flex-shrink-0" />
                 <div>
                   <strong className="text-gray-900 dark:text-gray-100">Career coaching:</strong>
                   <span className="text-gray-600 dark:text-gray-400"> Supporto per CV, portfolio, preparazione colloqui, e strategia di carriera</span>
                 </div>
               </div>
               <div className="flex items-start gap-3">
                 <CheckCircle className="h-5 w-5 text-gray-600 dark:text-gray-400 mt-1 flex-shrink-0" />
                 <div>
                   <strong className="text-gray-900 dark:text-gray-100">24/7 assistance:</strong>
                   <span className="text-gray-600 dark:text-gray-400"> Community support e risorse sempre disponibili per superare ogni ostacolo</span>
                 </div>
               </div>
             </div>
           </div>
         </div>

         {/* Community stats */}
         <div className="bg-gray-900 dark:bg-gray-600 rounded-2xl p-8 text-center text-white dark:text-gray-100">
           <h3 className="text-2xl font-bold mb-8">Una Community che Ti Sostiene</h3>
           <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
             <div>
               <div className="text-3xl font-bold mb-2">5000+</div>
               <div className="text-gray-300 dark:text-gray-200">Alumni Attivi</div>
               <div className="text-sm text-gray-400 dark:text-gray-300 mt-1">In aziende Fortune 500</div>
             </div>
             <div>
               <div className="text-3xl font-bold mb-2">48h</div>
               <div className="text-gray-300 dark:text-gray-200">Response Time</div>
               <div className="text-sm text-gray-400 dark:text-gray-300 mt-1">Media per supporto tecnico</div>
             </div>
             <div>
               <div className="text-3xl font-bold mb-2">200+</div>
               <div className="text-gray-300 dark:text-gray-200">Mentor Esperti</div>
               <div className="text-sm text-gray-400 dark:text-gray-300 mt-1">Da FAANG e unicorn startups</div>
             </div>
             <div>
               <div className="text-3xl font-bold mb-2">92%</div>
               <div className="text-gray-300 dark:text-gray-200">Completion Rate</div>
               <div className="text-sm text-gray-400 dark:text-gray-300 mt-1">vs 15% industry average</div>
             </div>
           </div>
         </div>
       </div>
     </section>

     {/* Pricing - Bianco */}
     <section className="py-24 px-4 md:px-6 bg-white dark:bg-gray-950">
      <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-6 text-gray-900 dark:text-gray-100">Investi nel Tuo Futuro</h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto leading-relaxed">
              Un investimento che si ripaga rapidamente: il costo del corso viene tipicamente recuperato 
              con il primo aumento di stipendio. Tre opzioni per adattarsi alle tue esigenze e budget.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            {/* Basic Plan */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-800 transition-all duration-500 ease-in-out hover:shadow-2xl hover:-translate-y-2 hover:border-gray-300 dark:hover:border-gray-600 flex flex-col group">
              <div className="text-center">
                <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-gray-100 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-300">Essential</h3>
                <div className="text-4xl font-bold mb-2 text-gray-900 dark:text-gray-100 group-hover:scale-110 transition-transform duration-300">€297</div>
                <div className="text-gray-600 dark:text-gray-400 mb-6">per mese × 12 mesi</div>
                <div className="text-lg text-gray-700 dark:text-gray-300 mb-6">Totale: €3,564</div>
              </div>
              <ul className="space-y-3 mb-8 flex-grow">
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400 group-hover/item:text-green-500 transition-colors duration-200" />
                  <span className="text-gray-700 dark:text-gray-300">Accesso completo al curriculum</span>
                </li>
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400 group-hover/item:text-green-500 transition-colors duration-200" />
                  <span className="text-gray-700 dark:text-gray-300">Community forum e study groups</span>
                </li>
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400 group-hover/item:text-green-500 transition-colors duration-200" />
                  <span className="text-gray-700 dark:text-gray-300">Progetti pratici e code reviews</span>
                </li>
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400 group-hover/item:text-green-500 transition-colors duration-200" />
                  <span className="text-gray-700 dark:text-gray-300">Certificazione professionale</span>
                </li>
              </ul>
              <Button 
                className="w-full mt-auto transition-all duration-300 hover:scale-105 hover:shadow-lg hover:bg-blue-600 hover:border-blue-600 hover:text-white" 
                variant="outline"
              >
                Inizia Essential
              </Button>
            </div>

            {/* Premium Plan */}
            <div className="bg-gray-900 dark:bg-gray-800 rounded-2xl p-8 shadow-xl border-2 border-gray-700 dark:border-gray-600 relative transition-all duration-500 ease-in-out hover:shadow-2xl hover:-translate-y-3 hover:border-gray-500 dark:hover:border-gray-400 flex flex-col group">
              <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                <div className="bg-gray-800 dark:bg-gray-600 text-white dark:text-gray-100 px-4 py-2 rounded-full text-sm font-semibold group-hover:bg-gradient-to-r group-hover:from-blue-600 group-hover:to-purple-600 group-hover:animate-pulse transition-all duration-300">
                  PIÙ POPOLARE
                </div>
              </div>
              <div className="text-center text-white dark:text-gray-100">
                <h3 className="text-2xl font-bold mb-4 group-hover:text-blue-300 transition-colors duration-300">Professional</h3>
                <div className="text-4xl font-bold mb-2 group-hover:scale-110 transition-transform duration-300">€497</div>
                <div className="text-gray-300 dark:text-gray-200 mb-6">per mese × 12 mesi</div>
                <div className="text-lg text-gray-200 dark:text-gray-200 mb-6">Totale: €5,964</div>
              </div>
              <ul className="space-y-3 mb-8 text-white dark:text-gray-100 flex-grow">
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-300 dark:text-gray-300 group-hover/item:text-green-400 transition-colors duration-200" />
                  <span>Tutto del piano Essential</span>
                </li>
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-300 dark:text-gray-300 group-hover/item:text-green-400 transition-colors duration-200" />
                  <span>Mentorship 1-on-1 mensile</span>
                </li>
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-300 dark:text-gray-300 group-hover/item:text-green-400 transition-colors duration-200" />
                  <span>Career coaching e job placement</span>
                </li>
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-300 dark:text-gray-300 group-hover/item:text-green-400 transition-colors duration-200" />
                  <span>Accesso a dataset proprietari enterprise</span>
                </li>
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-300 dark:text-gray-300 group-hover/item:text-green-400 transition-colors duration-200" />
                  <span>Workshop esclusivi con industry leaders</span>
                </li>
              </ul>
              <Button 
                className="w-full bg-white text-gray-900 hover:bg-gradient-to-r hover:from-blue-500 hover:to-purple-600 hover:text-white hover:scale-105 hover:shadow-xl mt-auto transition-all duration-300" 
              >
                Inizia Professional
              </Button>
            </div>

            {/* Elite Plan */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-800 transition-all duration-500 ease-in-out hover:shadow-2xl hover:-translate-y-2 hover:border-gray-300 dark:hover:border-gray-600 flex flex-col group">
              <div className="text-center">
                <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-gray-100 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors duration-300">Elite</h3>
                <div className="text-4xl font-bold mb-2 text-gray-900 dark:text-gray-100 group-hover:scale-110 transition-transform duration-300">€797</div>
                <div className="text-gray-600 dark:text-gray-400 mb-6">per mese × 12 mesi</div>
                <div className="text-lg text-gray-700 dark:text-gray-300 mb-6">Totale: €9,564</div>
              </div>
              <ul className="space-y-3 mb-8 flex-grow">
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400 group-hover/item:text-green-500 transition-colors duration-200" />
                  <span className="text-gray-700 dark:text-gray-300">Tutto del piano Professional</span>
                </li>
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400 group-hover/item:text-green-500 transition-colors duration-200" />
                  <span className="text-gray-700 dark:text-gray-300">Mentorship settimanale personalizzata</span>
                </li>
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400 group-hover/item:text-green-500 transition-colors duration-200" />
                  <span className="text-gray-700 dark:text-gray-300">Progetti enterprise custom</span>
                </li>
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400 group-hover/item:text-green-500 transition-colors duration-200" />
                  <span className="text-gray-700 dark:text-gray-300">Direct access to instructors</span>
                </li>
                <li className="flex items-center gap-2 group/item">
                  <CheckCircle className="h-4 w-4 text-gray-600 dark:text-gray-400 group-hover/item:text-green-500 transition-colors duration-200" />
                  <span className="text-gray-700 dark:text-gray-300">Garanzia placement o rimborso completo</span>
                </li>
              </ul>
              <Button 
                className="w-full mt-auto transition-all duration-300 hover:scale-105 hover:shadow-lg hover:bg-purple-600 hover:border-purple-600 hover:text-white" 
                variant="outline"
              >
                Contatta Sales
              </Button>
            </div>
          </div>

         {/* Value proposition */}
         <div className="text-center bg-gray-900 dark:bg-gray-800 rounded-2xl p-8 text-white dark:text-gray-100">
           <h3 className="text-2xl font-bold mb-4">ROI Garantito</h3>
           <p className="text-lg text-gray-300 dark:text-gray-200 max-w-3xl mx-auto mb-6 leading-relaxed">
             L'aumento salariale medio dei nostri studenti è di €15,000+ nel primo anno. 
             Il corso si ripaga completamente in meno di 6 mesi, garantendoti un ritorno sull'investimento del 400%+.
           </p>
           <div className="flex justify-center gap-4">
             <Button asChild size="lg" className="bg-white text-gray-900 hover:bg-gray-400">
               <Link to="/pricing">Confronta i Piani</Link>
             </Button>
             <Button asChild variant="outline" size="lg" className="bg-white text-gray-900 hover:bg-gray-400">
               <Link to="/demo">Richiedi Demo</Link>
             </Button>
           </div>
         </div>
       </div>
     </section>

     {/* Final CTA - Grigio Chiaro */}
     <section className="py-24 px-4 md:px-6 bg-gray-50 dark:bg-gray-900">
       <div className="container mx-auto max-w-6xl">
         <div className="text-center">
           <h2 className="text-4xl md:text-5xl font-bold mb-8 text-gray-900 dark:text-gray-100">
             Il Tuo Futuro nell'AI Inizia Oggi
           </h2>
           <p className="text-xl text-gray-600 dark:text-gray-400 max-w-4xl mx-auto mb-12 leading-relaxed">
             Non aspettare che qualcun altro prenda il tuo posto. L'industria dell'AI sta crescendo esponenzialmente 
             e le opportunità migliori sono per chi ha competenze profonde e verificabili. 
             In un mercato dove l'AI diventa commodity, la comprensione dei fondamenti ti rende insostituibile.
           </p>
           
           <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-xl border border-gray-200 dark:border-gray-700 mb-12 max-w-4xl mx-auto">
             <h3 className="text-2xl font-bold mb-6 text-gray-900 dark:text-gray-100">Perché Agire Subito</h3>
             <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
               <div>
                 <div className="text-3xl mb-3">⚡</div>
                 <h4 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">Mercato in Crescita</h4>
                 <p className="text-sm text-gray-600 dark:text-gray-400">
                   +40% crescita annuale per ruoli ML. Le posizioni senior scarseggiano e gli stipendi stanno esplodendo.
                 </p>
               </div>
               <div>
                 <div className="text-3xl mb-3">🏆</div>
                 <h4 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">First Mover Advantage</h4>
                 <p className="text-sm text-gray-600 dark:text-gray-400">
                   Chi sviluppa competenze profonde ora avrà un vantaggio competitivo decisivo nei prossimi 10 anni.
                 </p>
               </div>
               <div>
                 <div className="text-3xl mb-3">💎</div>
                 <h4 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">Differenziazione</h4>
                 <p className="text-sm text-gray-600 dark:text-gray-400">
                   Mentre tutti usano tools no-code, tu sarai tra i pochi che sa realmente come funzionano.
                 </p>
               </div>
             </div>
           </div>

           <div className="space-y-6">
             <div className="flex flex-col sm:flex-row gap-4 justify-center">
               <Button asChild size="lg" className="text-lg px-12 py-4">
                 <Link to="/theory">
                   Inizia la Trasformazione <ArrowRight className="h-5 w-5 ml-2" />
                 </Link>
               </Button>
               <Button asChild variant="outline" size="lg" className="text-lg px-12 py-4">
                 <Link to="/demo">Prenota Demo Gratuita</Link>
               </Button>
             </div>
             
             <div className="text-center">
               <p className="text-gray-500 dark:text-gray-400 text-sm">
                 ✅ Garanzia di rimborso completo entro 30 giorni • ✅ Nessun vincolo contrattuale • ✅ Supporto lifetime
               </p>
             </div>
           </div>
         </div>
       </div>
     </section>
   </MainLayout>
 );
};

export default Home;