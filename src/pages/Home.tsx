import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import MainLayout from "@/components/layout/MainLayout";
import { ArrowRight, BookOpen, Code, BarChart } from "lucide-react";
import { useEffect, useRef } from "react";

const AINetworkAnimation = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Network nodes (centered and expanded)
    const centerX = canvas.offsetWidth / 2;
    const centerY = canvas.offsetHeight / 2;
    
    const nodes = [
      // Input layer
      { x: centerX - 170, y: centerY - 110, layer: 0, active: false },
      { x: centerX - 170, y: centerY - 37, layer: 0, active: false },
      { x: centerX - 170, y: centerY + 37, layer: 0, active: false },
      { x: centerX - 170, y: centerY + 110, layer: 0, active: false },
      
      // Hidden layer 1
      { x: centerX - 57, y: centerY - 140, layer: 1, active: false },
      { x: centerX - 57, y: centerY - 70, layer: 1, active: false },
      { x: centerX - 57, y: centerY, layer: 1, active: false },
      { x: centerX - 57, y: centerY + 70, layer: 1, active: false },
      { x: centerX - 57, y: centerY + 140, layer: 1, active: false },
      
      // Hidden layer 2
      { x: centerX + 57, y: centerY - 110, layer: 2, active: false },
      { x: centerX + 57, y: centerY - 37, layer: 2, active: false },
      { x: centerX + 57, y: centerY + 37, layer: 2, active: false },
      { x: centerX + 57, y: centerY + 110, layer: 2, active: false },
      
      // Output layer
      { x: centerX + 170, y: centerY - 37, layer: 3, active: false },
      { x: centerX + 170, y: centerY + 37, layer: 3, active: false },
    ];

    // Connections
    const connections: Array<{ from: number; to: number; weight: number; active: number }> = [];
    
    // Create connections between layers
    for (let i = 0; i < nodes.length; i++) {
      for (let j = 0; j < nodes.length; j++) {
        if (nodes[j].layer === nodes[i].layer + 1) {
          connections.push({
            from: i,
            to: j,
            weight: Math.random() * 0.8 + 0.2,
            active: 0
          });
        }
      }
    }

    let animationTime = 0;
    let pulseStart = 0;

    const animate = () => {
      animationTime += 0.02;
      
      // Clear canvas
      ctx.clearRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);
      
      // Create grayscale gradient background
      const gradient = ctx.createLinearGradient(0, 0, canvas.offsetWidth, canvas.offsetHeight);
      gradient.addColorStop(0, 'rgba(240, 240, 240, 0.05)');
      gradient.addColorStop(1, 'rgba(200, 200, 200, 0.05)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);

      // Trigger pulse every 3 seconds
      if (Math.floor(animationTime) > pulseStart + 3) {
        pulseStart = Math.floor(animationTime);
        // Reset all activations
        nodes.forEach(node => node.active = false);
        connections.forEach(conn => conn.active = 0);
        
        // Activate input nodes randomly
        nodes.filter(n => n.layer === 0).forEach(node => {
          node.active = Math.random() > 0.3;
        });
      }

      // Propagate activation through network
      const currentPulseTime = animationTime - pulseStart;
      if (currentPulseTime < 2) {
        for (let layer = 1; layer < 4; layer++) {
          const layerDelay = layer * 0.3;
          if (currentPulseTime > layerDelay) {
            nodes.filter(n => n.layer === layer).forEach(node => {
              // Check if any connected input nodes are active
              const inputConnections = connections.filter(c => c.to === nodes.indexOf(node));
              const hasActiveInput = inputConnections.some(c => nodes[c.from].active);
              if (hasActiveInput && Math.random() > 0.2) {
                node.active = true;
              }
            });
          }
        }

        // Activate connections with smoother transitions
        connections.forEach(conn => {
          if (nodes[conn.from].active && currentPulseTime > (nodes[conn.to].layer * 0.4 - 0.15)) {
            conn.active = Math.min(1, conn.active + 0.08);
          } else {
            conn.active = Math.max(0, conn.active - 0.03);
          }
        });
      }

      // Draw connections with smooth grayscale animation (darker when active)
      connections.forEach(conn => {
        const fromNode = nodes[conn.from];
        const toNode = nodes[conn.to];
        
        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        ctx.lineTo(toNode.x, toNode.y);
        
        // Improved visibility for inactive connections
        const baseOpacity = 0.4; // Increased from 0.15
        const activeOpacity = 0.9;
        const opacity = baseOpacity + (conn.active * (activeOpacity - baseOpacity));
        const thickness = 1.2 + (conn.active * 1.5); // Increased base thickness from 0.8
        
        // Darker grayscale values with better contrast
        const grayValue = Math.floor(130 - (conn.active * 60)); // 130-70 range (better visibility)
        ctx.strokeStyle = `rgba(${grayValue}, ${grayValue}, ${grayValue}, ${opacity})`;
        ctx.lineWidth = thickness;
        ctx.stroke();
        
        // Add flowing particles on active connections
        if (conn.active > 0.3) {
          const progress = (animationTime * 1.5) % 1;
          const particleX = fromNode.x + (toNode.x - fromNode.x) * progress;
          const particleY = fromNode.y + (toNode.y - fromNode.y) * progress;
          
          ctx.beginPath();
          ctx.arc(particleX, particleY, 1.5, 0, Math.PI * 2);
          const particleGray = Math.floor(100 - (conn.active * 30)); // Darker particles
          ctx.fillStyle = `rgba(${particleGray}, ${particleGray}, ${particleGray}, ${conn.active * 0.9})`;
          ctx.fill();
        }
      });

      // Draw nodes
      nodes.forEach((node, index) => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, 8, 0, Math.PI * 2);
        
        if (node.active) {
          // Active node with darker grayscale
          const pulseIntensity = 0.7 + Math.sin(animationTime * 6) * 0.3;
          const grayValue = Math.floor(100 - (pulseIntensity * 30)); // Darker when active (70-100 range)
          ctx.fillStyle = `rgba(${grayValue}, ${grayValue}, ${grayValue}, ${pulseIntensity})`;
          ctx.fill();
          
          // Darker glow effect
          ctx.beginPath();
          ctx.arc(node.x, node.y, 12, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(80, 80, 80, 0.4)`;
          ctx.fill();
        } else {
          // Inactive node - better visibility
          ctx.fillStyle = 'rgba(120, 120, 120, 0.6)'; // Darker and more opaque
          ctx.fill();
          ctx.strokeStyle = 'rgba(100, 100, 100, 0.8)'; // Darker stroke
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      });

      requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
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

      {/* Features Section */}
      <section className="py-20 px-4 md:px-6">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">Impara il Machine Learning nel Modo Giusto</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              La nostra piattaforma combina conoscenze teoriche con esercizi pratici di programmazione per aiutarti a padroneggiare i concetti del machine learning.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-card rounded-lg p-6 shadow-sm card-hover text-center">
              <div className="h-12 w-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <BookOpen className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Teoria Completa</h3>
              <p className="text-muted-foreground">
                Spiegazioni approfondite dei concetti di machine learning, algoritmi e matematica che li sottende.
              </p>
            </div>

            <div className="bg-card rounded-lg p-6 shadow-sm card-hover text-center">
              <div className="h-12 w-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <Code className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Implementazione Pratica</h3>
              <p className="text-muted-foreground">
                Scopri come funzionano gli algoritmi di machine learning e data science implementandoli in Python con PyTorch.
              </p>
            </div>

            <div className="bg-card rounded-lg p-6 shadow-sm card-hover text-center">
              <div className="h-12 w-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <BarChart className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Progetti Pratici</h3>
              <p className="text-muted-foreground">
                Costruisci progetti completi di machine learning per applicare le conoscenze acquisite a scenari reali.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Learning Pipeline Section */}
      <section className="py-20 px-4 md:px-6 bg-muted/30">
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