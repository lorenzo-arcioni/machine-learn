import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  ArrowLeft, 
  Clock, 
  BookOpen, 
  ExternalLink,
  Loader2,
  Github,
  FileText,
  RefreshCw
} from 'lucide-react';

declare global {
  interface Window {
    fs?: {
      readFile: (filename: string, options?: { encoding?: string }) => Promise<string | Uint8Array>;
    };
  }
}

interface Exercise {
  id: string;
  title: string;
  description: string;
  category: string;
  tags: string[];
  estimatedTime: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  level: string;
  prerequisites: string[];
  problemStatement: {
    overview: string;
    objectives: string[];
    context: string;
  };
  githubSolutionUrl: string;
  resources: {
    title: string;
    type: string;
    url: string;
    description: string;
  }[];
}

const NBViewer = ({ notebookUrl }: { notebookUrl: string }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [containerHeight, setContainerHeight] = useState(1000);
  const [notebookStats, setNotebookStats] = useState<{cells: number, estimatedHeight: number} | null>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Converte URL GitHub in raw URL
  const getRawNotebookUrl = (url: string) => {
    if (url.includes('github.com') && url.includes('/blob/')) {
      return url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/');
    }
    return url;
  };

  // Crea URL NBViewer
  const getNBViewerUrl = (url: string) => {
    if (url.includes('github.com')) {
      const githubPath = url.replace('https://github.com/', '');
      return `https://nbviewer.org/github/${githubPath}`;
    }
    return `https://nbviewer.org/url/${encodeURIComponent(url)}`;
  };

  const rawUrl = getRawNotebookUrl(notebookUrl);
  const nbviewerUrl = getNBViewerUrl(notebookUrl);

  // Analizza il notebook per stimare l'altezza
  const analyzeNotebook = async () => {
    try {
      const response = await fetch(rawUrl);
      if (!response.ok) throw new Error('Cannot fetch notebook');
      
      const notebook = await response.json();
      if (!notebook.cells) throw new Error('Invalid notebook format');

      let estimatedHeight = 150; // Header NBViewer
      const cellCount = notebook.cells.length;

      notebook.cells.forEach((cell: any) => {
        // Padding base per ogni cella
        estimatedHeight += 40;
        
        if (cell.cell_type === 'markdown') {
          const content = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');
          // ~20px per riga di testo, minimo 30px
          estimatedHeight += Math.max(30, content.split('\n').length * 20);
        } 
        else if (cell.cell_type === 'code') {
          const sourceLines = Array.isArray(cell.source) ? cell.source.length : (cell.source || '').split('\n').length;
          // ~18px per riga di codice
          estimatedHeight += sourceLines * 18;
          
          // Analizza gli output
          if (cell.outputs && cell.outputs.length > 0) {
            cell.outputs.forEach((output: any) => {
              if (output.output_type === 'display_data' || output.output_type === 'execute_result') {
                // Grafici o immagini - altezza fissa generosa
                estimatedHeight += 400;
              } else if (output.output_type === 'stream') {
                const textContent = Array.isArray(output.text) ? output.text.join('') : (output.text || '');
                estimatedHeight += Math.min(textContent.split('\n').length * 16, 300);
              } else if (output.output_type === 'error') {
                estimatedHeight += 120; // Traceback error
              }
            });
          }
        }
      });

      // Aggiungi margine di sicurezza
      const finalHeight = Math.max(600, Math.min(estimatedHeight + 200, 4000));
      
      setNotebookStats({ cells: cellCount, estimatedHeight: finalHeight });
      setContainerHeight(finalHeight);
      
    } catch (error) {
      console.log('Could not analyze notebook, using fallback height:', error);
      setContainerHeight(1200);
    }
  };

  // Carica analisi del notebook
  useEffect(() => {
    analyzeNotebook();
  }, [rawUrl]);

  // Gestione caricamento iframe
  const handleIframeLoad = () => {
    setIsLoading(false);
    
    // Prova tecniche avanzate per rilevare altezza effettiva
    setTimeout(() => {
      tryAdvancedHeightDetection();
    }, 2000);
  };

  // Tecniche avanzate per il rilevamento altezza
  const tryAdvancedHeightDetection = () => {
    const iframe = iframeRef.current;
    if (!iframe) return;

    try {
      // Tenta accesso cross-origin (funziona raramente ma vale la pena provare)
      const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
      if (iframeDoc) {
        const body = iframeDoc.body;
        const html = iframeDoc.documentElement;
        
        const height = Math.max(
          body?.scrollHeight || 0,
          body?.offsetHeight || 0,
          html?.clientHeight || 0,
          html?.scrollHeight || 0,
          html?.offsetHeight || 0
        );
        
        if (height > 200 && height !== containerHeight) {
          setContainerHeight(height + 100);
          console.log(`Detected actual height: ${height}px`);
        }
      }
    } catch (e) {
      // Cross-origin blocked - normale per NBViewer
      console.log('Cross-origin height detection blocked (expected)');
    }

    // Listener per messaggi dall'iframe
    const messageHandler = (event: MessageEvent) => {
      if (event.data && typeof event.data.height === 'number') {
        const newHeight = event.data.height + 50;
        if (newHeight > containerHeight && newHeight < 5000) {
          setContainerHeight(newHeight);
        }
      }
    };
    
    window.addEventListener('message', messageHandler);
    return () => window.removeEventListener('message', messageHandler);
  };

  const handleRefresh = () => {
    setIsLoading(true);
    if (iframeRef.current) {
      iframeRef.current.src = nbviewerUrl;
    }
    analyzeNotebook();
  };

  const handleOpenInNewTab = () => {
    window.open(nbviewerUrl, '_blank');
  };

  const handleOpenOriginal = () => {
    window.open(notebookUrl, '_blank');
  };

  const handleOpenInColab = () => {
    if (notebookUrl.includes('github.com')) {
      const githubPath = notebookUrl.replace('https://github.com/', '');
      const colabUrl = `https://colab.research.google.com/github/${githubPath}`;
      window.open(colabUrl, '_blank');
    } else {
      window.open('https://colab.research.google.com/', '_blank');
    }
  };

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-red-500 mb-4">{error}</p>
        <div className="space-x-2">
          <Button variant="outline" onClick={handleOpenInNewTab}>
            <ExternalLink className="h-4 w-4 mr-2" />
            Open in NBViewer
          </Button>
          <Button variant="outline" onClick={handleOpenOriginal}>
            <Github className="h-4 w-4 mr-2" />
            View Original
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Action Buttons */}
      <div className="flex flex-wrap gap-2 justify-between items-center">
        <div className="text-sm text-gray-600">
          {notebookStats && (
            <span className="bg-gray-100 px-2 py-1 rounded text-xs">
              📊 {notebookStats.cells} cells • 📏 {containerHeight}px
            </span>
          )}
        </div>
        
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={handleRefresh}>
            <RefreshCw className="h-4 w-4 mr-1" />
            Refresh
          </Button>
          
          <Button 
            size="sm" 
            variant="outline"
            onClick={handleOpenInColab}
            className="bg-green-50 hover:bg-green-100 border-green-200 text-green-700"
          >
            <ExternalLink className="h-4 w-4 mr-2" />
            Colab
          </Button>
          
          <Button 
            size="sm" 
            variant="outline"
            onClick={handleOpenInNewTab}
            className="bg-blue-50 hover:bg-blue-100 border-blue-200 text-blue-700"
          >
            <ExternalLink className="h-4 w-4 mr-2" />
            New Tab
          </Button>
          
          <Button 
            size="sm" 
            variant="outline" 
            onClick={handleOpenOriginal}
          >
            <Github className="h-4 w-4 mr-2" />
            GitHub
          </Button>
        </div>
      </div>

      {/* NBViewer Container */}
      <div 
        ref={containerRef}
        className="relative border border-gray-200 rounded-lg bg-white overflow-hidden shadow-sm"
        style={{ height: `${containerHeight}px` }}
      >
        {isLoading && (
          <div className="absolute inset-0 bg-white/90 backdrop-blur-sm flex items-center justify-center z-10">
            <div className="text-center">
              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-3 text-blue-600" />
              <p className="text-gray-700 font-medium">Loading notebook...</p>
              <div className="mt-2 flex justify-center space-x-1">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
          </div>
        )}
        
        <iframe
          ref={iframeRef}
          src={nbviewerUrl}
          className="w-full h-full border-0"
          onLoad={handleIframeLoad}
          onError={() => setError('Failed to load notebook')}
          title="Jupyter Notebook"
          sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
        />
      </div>

      {/* Footer Info */}
      <div className="text-xs text-gray-500 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4" />
          <span>NBViewer • Auto-sized container</span>
        </div>
        <div className="italic">
          No scrollbars needed ✨
        </div>
      </div>
    </div>
  );
};

export default function ExerciseDetail() {
  const exerciseId = "linear-regression-basics";
  
  const [exerciseData, setExerciseData] = useState<Exercise | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadExerciseData = async (id: string): Promise<Exercise | null> => {
    const filename = `${id}.json`;
    
    try {
      if (typeof window !== 'undefined' && window.fs?.readFile) {
        try {
          const fileContent = await window.fs.readFile(filename, { encoding: 'utf8' }) as string;
          const data = JSON.parse(fileContent);
          return data as Exercise;
        } catch (fsError) {
          console.log(`File ${filename} not found in uploaded files, trying fetch...`);
        }
      }

      const response = await fetch(`/data/exercises/${filename}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} - Exercise not found`);
      }
      const data = await response.json();
      return data as Exercise;
    } catch (error) {
      console.error(`Failed to load exercise ${filename}:`, error);
      throw error;
    }
  };

  useEffect(() => {
    const loadExercise = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const data = await loadExerciseData(exerciseId);
        if (data) {
          setExerciseData(data);
        } else {
          setError('Exercise data not found');
        }
      } catch (err: any) {
        setError(err.message || 'Failed to load exercise');
        console.error('Error loading exercise:', err);
      } finally {
        setLoading(false);
      }
    };

    if (exerciseId) {
      loadExercise();
    }
  }, [exerciseId]);

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case "Beginner": return "bg-green-500";
      case "Intermediate": return "bg-yellow-500";
      case "Advanced": return "bg-red-500";
      default: return "bg-gray-500";
    }
  };

  const handleBack = () => {
    window.history.back();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p>Loading exercise...</p>
        </div>
      </div>
    );
  }

  if (error || !exerciseData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Exercise Not Found</h1>
          <p className="text-gray-600 mb-4">{error}</p>
          <div className="space-x-4">
            <Button onClick={handleBack}>Go Back</Button>
            <Button variant="outline">Browse Exercises</Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto py-6 px-4 max-w-7xl">
        {/* Header */}
        <div className="flex items-center gap-4 mb-6">
          <Button variant="outline" size="sm" onClick={handleBack}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <Badge className={getDifficultyColor(exerciseData.difficulty)}>
                {exerciseData.difficulty}
              </Badge>
              <Badge variant="outline">{exerciseData.category}</Badge>
              <div className="flex items-center gap-1 text-sm text-gray-500">
                <Clock className="h-4 w-4" />
                {exerciseData.estimatedTime}
              </div>
            </div>
            <h1 className="text-3xl font-bold">{exerciseData.title}</h1>
            <p className="text-gray-600 mt-1">{exerciseData.description}</p>
          </div>
        </div>

        {/* Prerequisites */}
        {exerciseData.prerequisites?.length > 0 && (
          <Alert className="mb-6 border-blue-200 bg-blue-50">
            <BookOpen className="h-4 w-4" />
            <AlertDescription>
              <strong>Prerequisites:</strong> {exerciseData.prerequisites.join(", ")}
            </AlertDescription>
          </Alert>
        )}

        {/* Problem Statement */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Problem Statement</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="prose max-w-none">
              <h3 className="text-lg font-semibold mb-2">Overview</h3>
              <p className="mb-4">{exerciseData.problemStatement.overview}</p>

              <h3 className="text-lg font-semibold mb-2">Learning Objectives</h3>
              <ul className="list-disc list-inside space-y-1 mb-4">
                {exerciseData.problemStatement.objectives.map((obj: string, idx: number) => (
                  <li key={idx}>{obj}</li>
                ))}
              </ul>

              <h3 className="text-lg font-semibold mb-2">Context</h3>
              <p className="mb-4">{exerciseData.problemStatement.context}</p>

              <div className="flex gap-2 flex-wrap">
                {exerciseData.tags.map((tag: string) => (
                  <Badge key={tag} variant="secondary">{tag}</Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Jupyter Notebook */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Exercise Notebook
            </CardTitle>
          </CardHeader>
          <CardContent>
            {exerciseData.githubSolutionUrl ? (
              <NBViewer notebookUrl={exerciseData.githubSolutionUrl} />
            ) : (
              <div className="text-center py-8">
                <p className="text-gray-500 italic">No notebook available for this exercise</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Resources */}
        {exerciseData.resources && exerciseData.resources.length > 0 && (
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Additional Resources</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {exerciseData.resources.map((resource, index) => (
                  <div key={index} className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge variant="outline" className="text-xs">
                        {resource.type}
                      </Badge>
                    </div>
                    <h4 className="font-medium mb-1">{resource.title}</h4>
                    <p className="text-sm text-gray-600 mb-3">{resource.description}</p>
                    <Button variant="outline" size="sm" asChild>
                      <a href={resource.url} target="_blank" rel="noopener noreferrer">
                        <ExternalLink className="h-3 w-3 mr-1" />
                        View Resource
                      </a>
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}