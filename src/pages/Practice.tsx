import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Search, Clock, Code, BookOpen, Grid3X3, List, AlertCircle } from 'lucide-react';

// Dichiarazione di tipo per window.fs
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
  difficulty: 'Principiante' | 'Intermedio' | 'Avanzato';
  prerequisites: string[];
}

interface ExercisesData {
  Principiante: Exercise[];
  Intermedio: Exercise[];
  Avanzato: Exercise[];
}

type DifficultyLevel = keyof ExercisesData;
type ViewMode = 'grid' | 'list';

export default function Practice() {
  const navigate = useNavigate();
  const [currentExercisesData, setCurrentExercisesData] = useState<ExercisesData>({
    Principiante: [],
    Intermedio: [],
    Avanzato: []
  });
  const [loading, setLoading] = useState(true);
  const [loadingErrors, setLoadingErrors] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('Tutte le Categorie');
  const [selectedDifficulty, setSelectedDifficulty] = useState('Tutte');
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [activeLevel, setActiveLevel] = useState<DifficultyLevel>('Principiante');

  const categories = ['Tutte le Categorie', 'Supervised Learning', 'Deep Learning', 'Data Processing', 'Optimization', 'NLP', 'Computer Vision', 'Reinforcement Learning'];
  const difficulties = ['Tutte', 'Principiante', 'Intermedio', 'Avanzato'];

  // Nome del file JSON da caricare
  const jsonFilename = 'exercise-index.json';

  // Funzione per caricare il file JSON con tutti gli esercizi
  const loadExercisesJson = async (): Promise<ExercisesData | null> => {
    try {
      // Tenta di caricare il file usando window.fs se disponibile (per file caricati dall'utente)
      if (typeof window !== 'undefined' && window.fs?.readFile) {
        try {
          const fileContent = await window.fs.readFile(jsonFilename, { encoding: 'utf8' }) as string;
          const data = JSON.parse(fileContent);
          return validateExercisesData(data);
        } catch (fsError) {
          console.log(`File ${jsonFilename} not found in uploaded files, trying fetch...`);
        }
      }

      // Fallback: tenta di caricare via fetch (per file nel progetto)
      const response = await fetch(`/data/${jsonFilename}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      return validateExercisesData(data);
    } catch (error) {
      console.warn(`Failed to load ${jsonFilename}:`, error);
      return null;
    }
  };

  // Funzione per validare e normalizzare i dati caricati
  const validateExercisesData = (data: any): ExercisesData => {
    const result: ExercisesData = {
      Principiante: [],
      Intermedio: [],
      Avanzato: []
    };

    if (data && typeof data === 'object') {
      if (Array.isArray(data.Principiante)) {
        result.Principiante = data.Principiante;
      }
      if (Array.isArray(data.Intermedio)) {
        result.Intermedio = data.Intermedio;
      }
      if (Array.isArray(data.Avanzato)) {
        result.Avanzato = data.Avanzato;
      }
    }

    return result;
  };

  // Carica tutti gli esercizi dal file JSON
  useEffect(() => {
    const loadAllExercises = async () => {
      setLoading(true);
      setLoadingErrors([]);
      
      const exercisesData = await loadExercisesJson();
      
      if (exercisesData) {
        setCurrentExercisesData(exercisesData);
        console.log(`Loaded exercises:`, {
          Principiante: exercisesData.Principiante.length,
          Intermedio: exercisesData.Intermedio.length,
          Avanzato: exercisesData.Avanzato.length
        });
        
        // Imposta il tab attivo sul primo livello che ha esercizi
        const availableLevels: DifficultyLevel[] = ['Principiante', 'Intermedio', 'Avanzato'];
        const firstAvailableLevel = availableLevels.find(level => exercisesData[level].length > 0);
        if (firstAvailableLevel) {
          setActiveLevel(firstAvailableLevel);
        }
      } else {
        setLoadingErrors([`Could not load ${jsonFilename}`]);
      }

      setLoading(false);
    };

    loadAllExercises();
  }, []);

  const getDifficultyColor = (difficulty: Exercise['difficulty']) => {
    switch (difficulty) {
      case 'Principiante': return 'bg-green-500';
      case 'Intermedio': return 'bg-yellow-500';
      case 'Avanzato': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const filterExercises = (exercises: Exercise[]) => {
    return exercises.filter(exercise => {
      const searchLower = searchQuery.toLowerCase();
      const matchesSearch =
        exercise.title.toLowerCase().includes(searchLower) ||
        exercise.description.toLowerCase().includes(searchLower) ||
        exercise.tags.some(tag => tag.toLowerCase().includes(searchLower));
      const matchesCategory = selectedCategory === 'Tutte le Categorie' || exercise.category === selectedCategory;
      const matchesDifficulty = selectedDifficulty === 'Tutte' || exercise.difficulty === selectedDifficulty;
      return matchesSearch && matchesCategory && matchesDifficulty;
    });
  };

  const handleStartExercise = (id: string) => {
    navigate(`/exercise/${id}`);
  };

  // Funzione per ricaricare i dati
  const handleReload = () => {
    window.location.reload();
  };

  const ExerciseCard: React.FC<{ exercise: Exercise; viewMode: ViewMode }> = ({ exercise, viewMode }) => (
    <Card className={`transition-all duration-300 hover:shadow-lg cursor-pointer ${viewMode === 'list' ? 'flex' : ''}`}>
      <CardHeader className={viewMode === 'list' ? 'flex-shrink-0 w-80' : ''}>
        <div className="flex gap-2 flex-wrap mb-2">
          <Badge className={getDifficultyColor(exercise.difficulty)}>{exercise.difficulty}</Badge>
          <Badge variant="outline">{exercise.category}</Badge>
        </div>
        <CardTitle className="text-lg">{exercise.title}</CardTitle>
        <div className="flex items-center gap-4 text-sm text-gray-500 mt-2">
          <Clock className="h-4 w-4" /> {exercise.estimatedTime}
        </div>
      </CardHeader>
      <CardContent className={viewMode === 'list' ? 'flex-1' : ''}>
        <p className={`text-gray-600 mb-4 ${viewMode === 'list' ? 'pt-4' : ''}`}>{exercise.description}</p>
        <div className="space-y-3">
          <div>
            <h4 className="text-sm font-medium mb-1">Prerequisiti:</h4>
            <div className="flex gap-1 flex-wrap">
              {exercise.prerequisites.map(prereq => (
                <Badge key={prereq} variant="secondary" className="text-xs">{prereq}</Badge>
              ))}
            </div>
          </div>
          <div>
            <h4 className="text-sm font-medium mb-1">Strumenti:</h4>
            <div className="flex gap-1 flex-wrap">
              {exercise.tags.map(tag => (
                <Badge key={tag} variant="outline" className="text-xs">{tag}</Badge>
              ))}
            </div>
          </div>
        </div>
        <div className="flex gap-2 mt-4">
          <Button className="flex-1" onClick={() => handleStartExercise(exercise.id)}>
            <Code className="h-4 w-4 mr-2" /> Vai all'Esercizio
          </Button>
          <Button variant="outline" size="icon" title="View Documentation">
            <BookOpen className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading exercises...</p>
        </div>
      </div>
    );
  }

  const totalExercises = currentExercisesData.Principiante.length + currentExercisesData.Intermedio.length + currentExercisesData.Avanzato.length;

  // Se non ci sono esercizi caricati
  if (totalExercises === 0) {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="container mx-auto py-8 px-4">
          <div className="mb-8">
            <h1 className="text-4xl font-bold mb-2">Machine Learning Practice</h1>
            <p className="text-gray-600 text-lg">
              Master machine learning through hands-on coding exercises.
            </p>
          </div>

          <div className="bg-white rounded-lg border p-8 text-center">
            <AlertCircle className="h-16 w-16 text-gray-300 mx-auto mb-4" />
            <h2 className="text-2xl font-semibold mb-2">No Exercise File Found</h2>
            <p className="text-gray-600 mb-4">
              Could not load the exercises file. Make sure the following file is available:
            </p>
            <ul className="text-sm text-gray-500 mb-6 space-y-1">
              <li>• /data/exercise-index.json</li>
            </ul>
            {loadingErrors.length > 0 && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                <h3 className="text-sm font-medium text-red-800 mb-2">Loading Errors:</h3>
                <ul className="text-sm text-red-600 space-y-1">
                  {loadingErrors.map((error, index) => (
                    <li key={index}>• {error}</li>
                  ))}
                </ul>
              </div>
            )}
            <Button onClick={handleReload}>
              Try Again
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto py-8 px-4">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4">Laboratorio Pratico di Machine Learning</h1>
          <p className="text-gray-600 text-lg w-full leading-relaxed">
            Metti alla prova le tue competenze con esercizi di programmazione guidati e progetti hands-on. 
            Il nostro laboratorio pratico ti offre un ambiente di apprendimento interattivo dove puoi 
            applicare immediatamente i concetti teorici del machine learning.
          </p>
        </div>

        {/* Mostra eventuali errori di caricamento */}
        {loadingErrors.length > 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle className="h-4 w-4 text-yellow-600" />
              <h3 className="text-sm font-medium text-yellow-800">Could not load exercises file:</h3>
            </div>
            <ul className="text-sm text-yellow-700 space-y-1">
              {loadingErrors.map((error, index) => (
                <li key={index}>• {error}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Filters */}
        <div className="bg-white rounded-lg border p-6 mb-8 shadow-sm">
          <div className="flex flex-col lg:flex-row gap-4 items-center">
            <div className="relative flex-1 min-w-0">
              <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Cerca esercizi, tags, categorie..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            <Select value={selectedCategory} onValueChange={setSelectedCategory}>
              <SelectTrigger className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {categories.map(category => (
                  <SelectItem key={category} value={category}>
                    {category}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={selectedDifficulty} onValueChange={setSelectedDifficulty}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {difficulties.map(diff => (
                  <SelectItem key={diff} value={diff}>
                    {diff}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="flex border rounded-md">
              <Button 
                variant={viewMode === 'grid' ? 'default' : 'ghost'} 
                size="sm" 
                onClick={() => setViewMode('grid')}
                className="rounded-r-none"
              >
                <Grid3X3 className="h-4 w-4" />
              </Button>
              <Button 
                variant={viewMode === 'list' ? 'default' : 'ghost'} 
                size="sm" 
                onClick={() => setViewMode('list')}
                className="rounded-l-none"
              >
                <List className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <Tabs value={activeLevel} onValueChange={(val) => setActiveLevel(val as DifficultyLevel)} className="w-full">
          <TabsList className="grid w-full max-w-md grid-cols-3 mb-6">
            <TabsTrigger value="Principiante" disabled={currentExercisesData.Principiante.length === 0}>
              Principiante ({currentExercisesData.Principiante.length})
            </TabsTrigger>
            <TabsTrigger value="Intermedio" disabled={currentExercisesData.Intermedio.length === 0}>
              Intermedio ({currentExercisesData.Intermedio.length})
            </TabsTrigger>
            <TabsTrigger value="Avanzato" disabled={currentExercisesData.Avanzato.length === 0}>
              Avanzato ({currentExercisesData.Avanzato.length})
            </TabsTrigger>
          </TabsList>

          {(['Principiante', 'Intermedio', 'Avanzato'] as DifficultyLevel[]).map(level => {
            const filteredExercises = filterExercises(currentExercisesData[level]);
            
            return (
              <TabsContent key={level} value={level}>
                {filteredExercises.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="mb-4">
                      <Search className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                      <p className="text-gray-500 text-lg mb-2">No exercises found</p>
                      <p className="text-gray-400">
                        {currentExercisesData[level].length === 0
                          ? `No ${level} exercises available`
                          : searchQuery || selectedCategory !== 'Tutte le Categorie' || selectedDifficulty !== 'Tutte'
                          ? 'Try adjusting your filters'
                          : 'No exercises available in this category'}
                      </p>
                    </div>
                    {(searchQuery || selectedCategory !== 'Tutte le Categorie' || selectedDifficulty !== 'Tutte') && (
                      <Button
                        variant="outline"
                        onClick={() => {
                          setSearchQuery('');
                          setSelectedCategory('Tutte le Categorie');
                          setSelectedDifficulty('Tutte');
                        }}
                      >
                        Clear All Filters
                      </Button>
                    )}
                  </div>
                ) : (
                  <div className={viewMode === 'grid' 
                    ? 'grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6' 
                    : 'space-y-4'
                  }>
                    {filteredExercises.map(exercise => (
                      <ExerciseCard key={exercise.id} exercise={exercise} viewMode={viewMode} />
                    ))}
                  </div>
                )}
              </TabsContent>
            );
          })}
        </Tabs>
      </div>
    </div>
  );
}