import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  ArrowLeft, 
  Clock, 
  BookOpen, 
  Copy,
  Download,
  ExternalLink,
  Loader2,
  Github,
  Check,
  Play
} from 'lucide-react';
import Gist from 'react-gist';

// Componente per mostrare il Gist
const GistViewer = ({ gistUrl, filename }) => {
  const [gistId, setGistId] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [gistData, setGistData] = useState(null);
  const [copySuccess, setCopySuccess] = useState(false);

  useEffect(() => {
    if (gistUrl) {
      try {
        // Estrae l'ID del Gist dall'URL
        const id = gistUrl.split('/').pop()?.replace('.js', '');
        if (id) {
          setGistId(id);
          setError(null);
          fetchGistData(id);
        } else {
          setError('Invalid Gist URL');
          setIsLoading(false);
        }
      } catch (err) {
        setError('Failed to parse Gist URL');
        setIsLoading(false);
        console.error('Error parsing Gist URL:', err);
      }
    }
  }, [gistUrl]);

  const fetchGistData = async (id) => {
    try {
      setIsLoading(true);
      const response = await fetch(`https://api.github.com/gists/${id}`);
      if (!response.ok) {
        throw new Error('Failed to fetch Gist data');
      }
      const data = await response.json();
      setGistData(data);
      
      // Simula un piccolo delay per mostrare il loading
      setTimeout(() => {
        setIsLoading(false);
      }, 800);
    } catch (err) {
      setError('Failed to load Gist data');
      setIsLoading(false);
      console.error('Error fetching Gist data:', err);
    }
  };

  const getMainPythonFile = () => {
    if (!gistData?.files) return null;
    
    if (filename && gistData.files[filename]) {
      return gistData.files[filename];
    }
    
    // Trova il primo file .py o .ipynb
    const pythonFile = Object.values(gistData.files).find((file: any) => 
      file.filename?.endsWith('.py') || file.filename?.endsWith('.ipynb')
    );
    
    return pythonFile || Object.values(gistData.files)[0];
  };

  const handleCopyCode = async () => {
    const file = getMainPythonFile();
    if (file?.content) {
      try {
        await navigator.clipboard.writeText(file.content);
        setCopySuccess(true);
        setTimeout(() => setCopySuccess(false), 2000);
      } catch (err) {
        console.error('Failed to copy code:', err);
      }
    }
  };

  const handleDownloadCode = () => {
    const file = getMainPythonFile();
    if (file?.content) {
      const blob = new Blob([file.content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = (file as any).filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  const handleOpenInColab = () => {
    const file = getMainPythonFile();
    if (file?.content && gistUrl) {
      // URL per aprire direttamente in Colab
      const colabUrl = `https://colab.research.google.com/gist/${gistUrl.split('/').pop()}`;
      window.open(colabUrl, '_blank');
    }
  };

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-red-500 mb-4">{error}</p>
        <Button variant="outline" asChild>
          <a href={gistUrl} target="_blank" rel="noopener noreferrer">
            <ExternalLink className="h-4 w-4 mr-2" />
            View on GitHub
          </a>
        </Button>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="relative">
        {/* Loading Overlay */}
        <div className="absolute inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center z-10 rounded-lg">
          <div className="text-center">
            <div className="relative">
              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-3 text-blue-600" />
              <div className="absolute inset-0 rounded-full border-2 border-blue-100 animate-pulse"></div>
            </div>
            <p className="text-gray-600 font-medium">Loading Python code...</p>
            <div className="mt-2 flex justify-center space-x-1">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
            </div>
          </div>
        </div>
        
        {/* Placeholder content */}
        <div className="min-h-[400px] bg-gray-50 rounded-lg border-2 border-dashed border-gray-200 flex items-center justify-center">
          <div className="text-center text-gray-400">
            <Github className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p>Preparing code...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!gistId) {
    return (
      <div className="text-center py-8">
        <Loader2 className="h-6 w-6 animate-spin mx-auto mb-2" />
        <p className="text-gray-500">Loading Gist...</p>
      </div>
    );
  }

  const mainFile = getMainPythonFile();

  return (
    <div className="space-y-4">
      {/* Action Buttons */}
      <div className="flex flex-wrap gap-2 justify-end">
        <Button 
          size="sm" 
          variant="outline"
          onClick={handleOpenInColab}
          className="bg-orange-50 hover:bg-orange-100 border-orange-200 text-orange-700"
        >
          <Play className="h-4 w-4 mr-2" />
          Open in Colab
        </Button>
        
        <Button 
          size="sm" 
          variant="outline" 
          onClick={handleCopyCode}
          className="relative"
        >
          {copySuccess ? (
            <>
              <Check className="h-4 w-4 mr-2 text-green-600" />
              Copied!
            </>
          ) : (
            <>
              <Copy className="h-4 w-4 mr-2" />
              Copy Code
            </>
          )}
        </Button>
        
        <Button 
          size="sm" 
          variant="outline" 
          onClick={handleDownloadCode}
          disabled={!mainFile}
        >
          <Download className="h-4 w-4 mr-2" />
          Download
        </Button>
      </div>

      {/* Gist Content */}
      <div className="gist-container">
        {filename ? (
          <Gist id={gistId} file={filename} />
        ) : (
          <Gist id={gistId} />
        )}
      </div>

      {/* File Info */}
      {mainFile && (
        <div className="text-sm text-gray-500 mt-2">
          <span className="font-medium">{(mainFile as any).filename}</span>
          {(mainFile as any).size && (
            <span className="ml-2">({((mainFile as any).size / 1024).toFixed(1)} KB)</span>
          )}
        </div>
      )}
    </div>
  );
};

export default function ExerciseDetail() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [exerciseData, setExerciseData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("problem");

  // Aggiungi gli stili CSS per i Gist
  useEffect(() => {
    const style = document.createElement('style');
    style.innerHTML = `
      .gist-container .gist {
        font-size: 14px !important;
      }
      .gist-container .gist .gist-file {
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        overflow: hidden !important;
      }
      .gist-container .gist .gist-meta {
        background: #f9fafb !important;
        border-top: 1px solid #e5e7eb !important;
        padding: 12px 16px !important;
      }
      .gist-container .gist .highlight {
        background: #1f2937 !important;
      }
      .gist-container .gist .blob-code-inner {
        font-family: 'JetBrains Mono', 'Fira Code', 'Monaco', 'Consolas', monospace !important;
      }
      
      /* Loading animation styles */
      @keyframes shimmer {
        0% {
          background-position: -468px 0;
        }
        100% {
          background-position: 468px 0;
        }
      }
      
      .loading-shimmer {
        animation: shimmer 1.5s infinite linear;
        background: linear-gradient(to right, #f6f7f8 0%, #edeef1 20%, #f6f7f8 40%, #f6f7f8 100%);
        background-size: 800px 104px;
      }
    `;
    document.head.appendChild(style);

    return () => {
      document.head.removeChild(style);
    };
  }, []);

  // Carica i dati dell'esercizio
  useEffect(() => {
    const loadExercise = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/data/exercises/${id}.json`);
        if (!response.ok) {
          throw new Error('Exercise not found');
        }
        const data = await response.json();
        setExerciseData(data);
      } catch (err) {
        setError(err.message || 'Failed to load exercise');
        console.error('Error loading exercise:', err);
      } finally {
        setLoading(false);
      }
    };

    loadExercise();
  }, [id]);

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case "Beginner": return "bg-green-500";
      case "Intermediate": return "bg-yellow-500";
      case "Advanced": return "bg-red-500";
      default: return "bg-gray-500";
    }
  };

  const handleBack = () => {
    navigate('/practice');
  };

  const handleTabChange = (value) => {
    setActiveTab(value);
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

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Exercise Not Found</h1>
          <p className="text-gray-600 mb-4">{error}</p>
          <div className="space-x-4">
            <Button onClick={handleBack}>Go Back</Button>
            <Button variant="outline" onClick={() => navigate('/practice')}>
              Browse Exercises
            </Button>
          </div>
        </div>
      </div>
    );
  }

  if (!exerciseData) return null;

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto py-6 px-4">
        {/* Header */}
        <div className="flex items-center gap-4 mb-6">
          <Button variant="outline" size="sm" onClick={handleBack}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Practice
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

        <div className={`grid grid-cols-1 gap-6 ${activeTab === "problem" ? "lg:grid-cols-5" : "lg:grid-cols-1"}`}>
          {/* Left Column - Problem and Solution */}
          <div className={`space-y-6 ${activeTab === "problem" ? "lg:col-span-4" : "lg:col-span-1"}`}>
            <Tabs value={activeTab} onValueChange={handleTabChange}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="problem">Problem</TabsTrigger>
                <TabsTrigger value="solution">Solution</TabsTrigger>
              </TabsList>

              {/* Problem */}
              <TabsContent value="problem" className="mt-6">
                <Card className="min-h-[600px]">
                  <CardHeader>
                    <CardTitle>Problem Statement</CardTitle>
                  </CardHeader>
                  <CardContent className="prose max-w-none pb-8">
                    <h3 className="text-lg font-semibold mb-2">Overview</h3>
                    <p>{exerciseData.problemStatement?.overview}</p>

                    <h3 className="text-lg font-semibold mt-6 mb-2">Learning Objectives</h3>
                    <ul className="list-disc list-inside space-y-1">
                      {exerciseData.problemStatement?.objectives?.map((obj, idx) => (
                        <li key={idx}>{obj}</li>
                      ))}
                    </ul>

                    <h3 className="text-lg font-semibold mt-6 mb-2">Context</h3>
                    <p className="mb-6">{exerciseData.problemStatement?.context}</p>

                    <div className="flex gap-2 flex-wrap mt-6">
                      {exerciseData.tags?.map((tag) => (
                        <Badge key={tag} variant="secondary">{tag}</Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Solution */}
              <TabsContent value="solution" className="mt-6">
                <Card className="min-h-[600px]">
                  <CardHeader className="flex flex-row items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                      <Github className="h-5 w-5" />
                      Python Solution
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pb-8">
                    {exerciseData.githubSolutionUrl ? (
                      <GistViewer 
                        gistUrl={exerciseData.githubSolutionUrl}
                        filename={null} // Se vuoi mostrare un file specifico, passa il nome qui
                      />
                    ) : (
                      <div className="text-center py-8">
                        <p className="text-gray-500 italic">No solution available</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>

          {/* Right Column - Show only when Problem tab is active */}
          {activeTab === "problem" && (
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Quick Actions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {exerciseData.githubSolutionUrl && (
                    <Button className="w-full" asChild>
                      <a 
                        href={exerciseData.githubSolutionUrl.replace('.js', '')} 
                        target="_blank" 
                        rel="noopener noreferrer"
                      >
                        <Github className="h-4 w-4 mr-2" />
                        Open Gist
                      </a>
                    </Button>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Exercise Info</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium">Difficulty</h4>
                    <Badge className={getDifficultyColor(exerciseData.difficulty)}>
                      {exerciseData.difficulty}
                    </Badge>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium">Category</h4>
                    <Badge variant="outline">{exerciseData.category}</Badge>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium">Estimated Time</h4>
                    <div className="flex items-center gap-1 text-sm">
                      <Clock className="h-4 w-4" />
                      {exerciseData.estimatedTime}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}