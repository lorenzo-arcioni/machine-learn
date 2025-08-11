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
  RefreshCw,
  Download
} from 'lucide-react';

declare global {
  interface Window {
    fs?: {
      readFile: (filename: string, options?: { encoding?: string }) => Promise<string | Uint8Array>;
    };
    marked?: any;
    hljs?: any;
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

const NotebookViewer = ({ notebookUrl }: { notebookUrl: string }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notebook, setNotebook] = useState<any>(null);
  const [libsReady, setLibsReady] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Load all required libraries
  const loadLibraries = async () => {
    try {
      // Load libraries in parallel but ensure MathJax config is set first
      if (!window.MathJax) {
        window.MathJax = {
          tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            displayMath: [['$$', '$$'], ['\\[', '\\]']],
            processEscapes: true,
            processEnvironments: true
          },
          options: {
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
            ignoreHtmlClass: 'tex2jax_ignore',
            processHtmlClass: 'tex2jax_process'
          },
          startup: { 
            typeset: false,
            ready: () => {
              window.MathJax.startup.defaultReady();
              console.log('MathJax ready');
            }
          }
        };
      }

      const loadPromises = [];

      // Load Marked.js
      if (!window.marked) {
        loadPromises.push(new Promise((resolve, reject) => {
          const script = document.createElement('script');
          script.src = 'https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js';
          script.onload = resolve;
          script.onerror = reject;
          document.head.appendChild(script);
        }));
      }

      // Load Highlight.js
      if (!window.hljs) {
        loadPromises.push(new Promise((resolve, reject) => {
          const script = document.createElement('script');
          script.src = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js';
          script.onload = resolve;
          script.onerror = reject;
          document.head.appendChild(script);

          const css = document.createElement('link');
          css.rel = 'stylesheet';
          css.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
          document.head.appendChild(css);
        }));
      }

      // Load MathJax
      if (!document.querySelector('script[src*="mathjax"]')) {
        loadPromises.push(new Promise((resolve, reject) => {
          const script = document.createElement('script');
          script.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js';
          script.onload = () => {
            // Wait a bit for MathJax to fully initialize
            setTimeout(resolve, 500);
          };
          script.onerror = reject;
          document.head.appendChild(script);
        }));
      }

      await Promise.all(loadPromises);
      setLibsReady(true);
    } catch (err) {
      console.error('Error loading libraries:', err);
      setError('Failed to load required libraries');
    }
  };

  // Convert GitHub URL to raw URL
  const getRawNotebookUrl = (url: string) => {
    if (url.includes('github.com') && url.includes('/blob/')) {
      return url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/');
    }
    return url;
  };

  // Load notebook data
  const loadNotebook = async () => {
    if (!libsReady) return;

    try {
      setIsLoading(true);
      setError(null);

      const rawUrl = getRawNotebookUrl(notebookUrl);
      const response = await fetch(rawUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch notebook: ${response.status} ${response.statusText}`);
      }

      const notebookData = await response.json();
      if (!notebookData.cells) {
        throw new Error('Invalid notebook format: missing cells');
      }

      setNotebook(notebookData);
    } catch (err: any) {
      setError(err.message || 'Failed to load notebook');
      console.error('Error loading notebook:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Resolve image paths for GitHub notebooks
  const resolveImagePath = (src: string): string => {
    if (src.startsWith('http://') || src.startsWith('https://') || src.startsWith('data:')) {
      return src;
    }
    if (notebookUrl.includes('github.com')) {
      const baseUrl = notebookUrl.replace('/blob/', '/raw/').replace(/\/[^\/]*\.ipynb$/, '/');
      const cleanSrc = src.replace(/^\.\//, '');
      return baseUrl + cleanSrc;
    }
    return src;
  };

  // Process images in markdown content
  const processImages = (content: string): string => {
    return content
      .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, src) => {
        return `![${alt}](${resolveImagePath(src)})`;
      })
      .replace(/<img([^>]*)\s+src=["']([^"']+)["']([^>]*)>/gi, (match, before, src, after) => {
        return `<img${before} src="${resolveImagePath(src)}"${after}>`;
      });
  };

  // Render notebook cells
  const renderNotebook = () => {
    if (!notebook || !containerRef.current || !window.marked || !window.hljs) return;

    const container = containerRef.current;
    container.innerHTML = '';

    const notebookDiv = document.createElement('div');
    notebookDiv.className = 'jupyter-notebook';

    notebook.cells.forEach((cell: any, index: number) => {
      const cellDiv = document.createElement('div');
      cellDiv.className = `nb-cell nb-${cell.cell_type}-cell`;

      if (cell.cell_type === 'markdown') {
        const sourceContent = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');
        if (sourceContent.trim()) {
          const processedContent = processImages(sourceContent);
          const rendered = window.marked.parse(processedContent);
          
          const markdownDiv = document.createElement('div');
          markdownDiv.className = 'nb-markdown-container tex2jax_process';
          markdownDiv.innerHTML = rendered;
          cellDiv.appendChild(markdownDiv);
        }
      } else if (cell.cell_type === 'code') {
        // Code input
        const inputDiv = document.createElement('div');
        inputDiv.className = 'nb-code-input';
        
        const pre = document.createElement('pre');
        const code = document.createElement('code');
        code.className = 'language-python';
        
        const sourceContent = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');
        code.textContent = sourceContent;
        
        pre.appendChild(code);
        inputDiv.appendChild(pre);
        cellDiv.appendChild(inputDiv);

        // Code outputs
        if (cell.outputs && cell.outputs.length > 0) {
          cell.outputs.forEach((output: any) => {
            const outputDiv = renderOutput(output);
            if (outputDiv) cellDiv.appendChild(outputDiv);
          });
        }
      }

      notebookDiv.appendChild(cellDiv);
    });

    container.appendChild(notebookDiv);
    addNotebookStyles();

    // Highlight code and process MathJax
    setTimeout(() => {
      if (window.hljs) {
        window.hljs.highlightAll();
      }
      
      // Process MathJax with proper error handling and retries
      processMathJax();
    }, 100);
  };

  // Process MathJax with retry logic
  const processMathJax = async () => {
    if (!window.MathJax || !containerRef.current) return;

    let retries = 3;
    
    const tryProcess = async (): Promise<void> => {
      try {
        if (window.MathJax.typesetPromise) {
          await window.MathJax.typesetPromise([containerRef.current]);
          console.log('MathJax processed successfully');
        } else if (window.MathJax.Hub) {
          // Fallback for MathJax v2
          window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub, containerRef.current]);
        }
      } catch (err) {
        console.warn('MathJax processing error:', err);
        if (retries > 0) {
          retries--;
          setTimeout(tryProcess, 1000);
        }
      }
    };

    // Wait for MathJax to be fully ready
    if (window.MathJax.startup && window.MathJax.startup.promise) {
      await window.MathJax.startup.promise;
    }
    
    setTimeout(tryProcess, 200);
  };

  // Render cell outputs
  const renderOutput = (output: any) => {
    const outputDiv = document.createElement('div');
    outputDiv.className = 'nb-output';

    if (output.output_type === 'stream') {
      const streamDiv = document.createElement('pre');
      streamDiv.className = 'nb-stream-output';
      const text = Array.isArray(output.text) ? output.text.join('') : (output.text || '');
      streamDiv.textContent = text;
      outputDiv.appendChild(streamDiv);
    } else if (output.output_type === 'execute_result' || output.output_type === 'display_data') {
      if (output.data) {
        if (output.data['image/png']) {
          const img = document.createElement('img');
          img.src = `data:image/png;base64,${output.data['image/png']}`;
          img.className = 'nb-output-image';
          outputDiv.appendChild(img);
        } else if (output.data['image/jpeg']) {
          const img = document.createElement('img');
          img.src = `data:image/jpeg;base64,${output.data['image/jpeg']}`;
          img.className = 'nb-output-image';
          outputDiv.appendChild(img);
        } else if (output.data['text/html']) {
          const htmlDiv = document.createElement('div');
          htmlDiv.className = 'nb-html-output';
          const htmlContent = Array.isArray(output.data['text/html']) 
            ? output.data['text/html'].join('') 
            : output.data['text/html'];
          htmlDiv.innerHTML = htmlContent;
          outputDiv.appendChild(htmlDiv);
        } else if (output.data['text/plain']) {
          const textDiv = document.createElement('pre');
          textDiv.className = 'nb-text-output';
          const textContent = Array.isArray(output.data['text/plain'])
            ? output.data['text/plain'].join('')
            : output.data['text/plain'];
          textDiv.textContent = textContent;
          outputDiv.appendChild(textDiv);
        }
      }
    } else if (output.output_type === 'error') {
      const errorDiv = document.createElement('pre');
      errorDiv.className = 'nb-error-output';
      const traceback = output.traceback ? output.traceback.join('\n') : 
        `${output.ename}: ${output.evalue}`;
      errorDiv.textContent = traceback;
      outputDiv.appendChild(errorDiv);
    }

    return outputDiv.children.length > 0 ? outputDiv : null;
  };

  // Add notebook styles
  const addNotebookStyles = () => {
    const styleId = 'jupyter-notebook-styles';
    if (document.getElementById(styleId)) return;

    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = `
      .jupyter-notebook {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        line-height: 1.6;
        color: #24292f;
      }
      
      .nb-cell {
        margin-bottom: 1.5rem;
      }
      
      .nb-code-input pre {
        background: #f6f8fa;
        border: 1px solid #d1d9e0;
        border-radius: 6px;
        padding: 16px;
        margin: 0;
        overflow-x: auto;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.45;
      }
      
      .nb-code-input pre code {
        background: transparent;
        padding: 0;
        border: none;
        font-size: inherit;
        color: inherit;
      }
      
      .nb-output {
        margin-top: 8px;
      }
      
      .nb-stream-output, .nb-text-output {
        background: #f8f9fa;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 12px;
        margin: 4px 0;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
        font-size: 14px;
        overflow-x: auto;
        white-space: pre-wrap;
      }
      
      .nb-error-output {
        background: #fff5f5;
        border: 1px solid #fd6c6c;
        color: #d1242f;
        border-radius: 6px;
        padding: 12px;
        margin: 4px 0;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
        font-size: 14px;
        overflow-x: auto;
        white-space: pre-wrap;
      }
      
      .nb-output-image {
        max-width: 100%;
        height: auto;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 8px 0;
      }
      
      .nb-html-output table {
        border-collapse: collapse;
        margin: 1em 0;
        background: white;
      }
      
      .nb-html-output table th,
      .nb-html-output table td {
        border: 1px solid #d8dee4;
        padding: 8px 12px;
        text-align: left;
      }
      
      .nb-html-output table th {
        background: #f6f8fa;
        font-weight: 600;
      }
      
      .nb-markdown-container {
        margin: 1rem 0;
      }
      
      .nb-markdown-container h1,
      .nb-markdown-container h2,
      .nb-markdown-container h3,
      .nb-markdown-container h4,
      .nb-markdown-container h5,
      .nb-markdown-container h6 {
        margin: 1.5em 0 0.5em 0;
        font-weight: 600;
        line-height: 1.25;
      }
      
      .nb-markdown-container h1 {
        font-size: 2em;
        border-bottom: 1px solid #d8dee4;
        padding-bottom: 0.3em;
      }
      
      .nb-markdown-container h2 {
        font-size: 1.5em;
        border-bottom: 1px solid #d8dee4;
        padding-bottom: 0.3em;
      }
      
      .nb-markdown-container h3 {
        font-size: 1.25em;
      }
      
      .nb-markdown-container p {
        margin: 1em 0;
      }
      
      .nb-markdown-container ul,
      .nb-markdown-container ol {
        margin: 1em 0;
        padding-left: 2em;
      }
      
      .nb-markdown-container code {
        background: rgba(175, 184, 193, 0.2);
        border-radius: 3px;
        font-size: 85%;
        margin: 0;
        padding: 0.2em 0.4em;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
      }
      
      .nb-markdown-container pre {
        background: #f6f8fa;
        border: 1px solid #d1d9e0;
        border-radius: 6px;
        font-size: 85%;
        line-height: 1.45;
        overflow: auto;
        padding: 16px;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
      }
      
      .nb-markdown-container pre code {
        background: transparent;
        font-size: inherit;
        padding: 0;
      }
      
      .nb-markdown-container blockquote {
        border-left: 0.25em solid #d1d9e0;
        color: #656d76;
        margin: 0;
        padding: 0 1em;
      }
      
      .nb-markdown-container img {
        max-width: 100%;
        height: auto;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 16px 0;
      }
      
      /* MathJax styling */
      .MathJax {
        font-size: 1em !important;
      }
      
      .MathJax_Display {
        margin: 1em 0 !important;
        text-align: center;
      }
    `;
    document.head.appendChild(style);
  };

  // Effects
  useEffect(() => {
    loadLibraries();
  }, []);

  useEffect(() => {
    if (libsReady) {
      loadNotebook();
    }
  }, [libsReady, notebookUrl]);

  useEffect(() => {
    if (notebook && libsReady && !isLoading) {
      renderNotebook();
    }
  }, [notebook, libsReady, isLoading]);

  // Event handlers
  const handleRefresh = () => loadNotebook();
  
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

  const handleDownload = () => {
    if (notebook) {
      const blob = new Blob([JSON.stringify(notebook, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `notebook-${Date.now()}.ipynb`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  // Render component
  if (error) {
    return (
      <div className="text-center py-8">
        <div className="text-red-500 mb-4">
          <FileText className="h-12 w-12 mx-auto mb-2 opacity-50" />
          <p className="font-medium">Failed to load notebook</p>
          <p className="text-sm">{error}</p>
        </div>
        <div className="space-x-2">
          <Button variant="outline" onClick={handleRefresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
          <Button variant="outline" onClick={handleOpenOriginal}>
            <Github className="h-4 w-4 mr-2" />
            View on GitHub
          </Button>
        </div>
      </div>
    );
  }

  if (!libsReady) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-3 text-blue-600" />
          <p className="text-gray-700 font-medium">Loading notebook renderer...</p>
          <p className="text-sm text-gray-500 mt-2">Loading MathJax, Marked & Highlight.js...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Action Buttons */}
      <div className="flex flex-wrap gap-2 justify-between items-center">
        <div className="text-sm text-gray-600">
          {notebook && (
            <span className="bg-gray-100 px-2 py-1 rounded text-xs">
              📊 {notebook.cells?.length || 0} cells
            </span>
          )}
        </div>
        
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={handleRefresh} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 mr-1 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>

          {notebook && (
            <Button size="sm" variant="outline" onClick={handleDownload}>
              <Download className="h-4 w-4 mr-1" />
              Download
            </Button>
          )}
          
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
            onClick={handleOpenOriginal}
          >
            <Github className="h-4 w-4 mr-2" />
            GitHub
          </Button>
        </div>
      </div>

      {/* Notebook Container */}
      <div className="border border-gray-200 rounded-lg bg-white overflow-hidden shadow-sm">
        {isLoading ? (
          <div className="flex items-center justify-center py-16">
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
        ) : (
          <div 
            ref={containerRef}
            className="p-6 min-h-[200px]"
          />
        )}
      </div>

      {/* Footer Info */}
      {notebook && !isLoading && (
        <div className="text-xs text-gray-500 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            <span>Custom renderer • MathJax • External images • Syntax highlighting</span>
          </div>
          <div className="italic">
            Jupyter Notebook ✨
          </div>
        </div>
      )}
    </div>
  );
};

// Floating Back Button Component
const FloatingBackButton = ({ onClick }: { onClick: () => void }) => {
  return (
    <div className="fixed top-24 z-50" style={{ left: 'calc(50% + 448px + 2rem)' }}>
      <Button
        onClick={onClick}
        variant="outline"
        size="sm"
        className="bg-white hover:bg-gray-50 shadow-lg hover:shadow-xl transition-all duration-200 border-gray-300"
      >
        <ArrowLeft className="h-4 w-4 mr-2" />
        Torna agli Esercizi
      </Button>
    </div>
  );
};

export default function ExerciseDetail() {
  const getExerciseIdFromUrl = (): string | null => {
    const urlParams = new URLSearchParams(window.location.search);
    const queryId = urlParams.get('id');
    if (queryId) return queryId;
    
    const pathParts = window.location.pathname.split('/');
    const exerciseIndex = pathParts.indexOf('exercise');
    if (exerciseIndex !== -1 && pathParts[exerciseIndex + 1]) {
      return pathParts[exerciseIndex + 1];
    }
    
    const lastPart = pathParts[pathParts.length - 1];
    if (lastPart && lastPart !== 'exercise' && lastPart !== '') {
      return lastPart;
    }
    
    return null;
  };

  const exerciseId = getExerciseIdFromUrl();
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
          console.log(`Loaded exercise ${id} from uploaded files`);
          return data as Exercise;
        } catch (fsError) {
          console.log(`File ${filename} not found in uploaded files, trying fetch from /data/exercises/...`);
        }
      }

      const response = await fetch(`/data/exercises/${filename}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} - Exercise "${id}" not found at /data/exercises/${filename}`);
      }
      const data = await response.json();
      console.log(`Loaded exercise ${id} from /data/exercises/`);
      return data as Exercise;
    } catch (error) {
      console.error(`Failed to load exercise ${filename}:`, error);
      throw error;
    }
  };

  useEffect(() => {
    const loadExercise = async () => {
      if (!exerciseId) {
        setError('No exercise ID found in URL. Expected format: /exercise/exercise-id or /exercise?id=exercise-id');
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        setError(null);
        console.log(`Loading exercise: ${exerciseId}`);
        
        const data = await loadExerciseData(exerciseId);
        if (data) {
          setExerciseData(data);
        } else {
          setError(`Exercise "${exerciseId}" data not found`);
        }
      } catch (err: any) {
        setError(err.message || `Failed to load exercise "${exerciseId}"`);
        console.error('Error loading exercise:', err);
      } finally {
        setLoading(false);
      }
    };

    loadExercise();
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
    window.location.href = '/practice';
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading exercise...</p>
          {exerciseId && (
            <p className="text-sm text-gray-500 mt-2">Loading: {exerciseId}</p>
          )}
        </div>
      </div>
    );
  }

  if (error || !exerciseData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center max-w-md">
          <div className="mb-6">
            <FileText className="h-16 w-16 text-gray-300 mx-auto mb-4" />
            <h1 className="text-2xl font-bold mb-2">Exercise Not Found</h1>
            <p className="text-gray-600 mb-2">{error}</p>
            {exerciseId && (
              <div className="bg-gray-50 border rounded-lg p-3 text-left">
                <p className="text-sm text-gray-600 mb-1">Looking for:</p>
                <code className="text-xs bg-gray-100 px-2 py-1 rounded block">
                  /data/exercises/{exerciseId}.json
                </code>
                <p className="text-sm text-gray-500 mt-2">Current URL: <code className="text-xs">{window.location.pathname}</code></p>
                <p className="text-sm text-gray-500">Extracted ID: <code className="text-xs">{exerciseId}</code></p>
              </div>
            )}
          </div>
          <div className="space-x-4">
            <Button onClick={handleBack}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Torna Indietro
            </Button>
            <Button variant="outline" onClick={() => window.location.href = '/'}>
              Vai agli Esercizi
            </Button>
          </div>
        </div>
        {/* Floating Back Button for Error State */}
        <FloatingBackButton onClick={handleBack} />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 relative">
      <div className="container mx-auto py-6 px-4 max-w-7xl">
        {/* Header - senza pulsante back */}
        <div className="mb-6">
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
              <NotebookViewer notebookUrl={exerciseData.githubSolutionUrl} />
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

        {/* Spazio aggiuntivo in fondo per evitare che il contenuto sia nascosto dal floating button */}
        <div className="h-4"></div>
      </div>

      {/* Floating Back Button */}
      <FloatingBackButton onClick={handleBack} />
    </div>
  );
}