import MainLayout from "@/components/layout/MainLayout";
import { Link } from "react-router-dom";
import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  BookOpen,
  BarChart4,
  ScatterChart,
  BrainCircuit,
  MessageSquareText,
  Sigma,
} from "lucide-react";

// Mappa degli icon per associarli ai nomi nel JSON
const iconMap = {
  "Sigma": Sigma,
  "BookOpen": BookOpen,
  "BarChart4": BarChart4,
  "ScatterChart": ScatterChart,
  "BrainCircuit": BrainCircuit,
  "MessageSquareText": MessageSquareText,
};

const Theory = () => {
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadTopics = async () => {
      try {
        const response = await fetch('/data/theory.json');
        if (!response.ok) {
          throw new Error('Impossibile caricare gli argomenti teorici');
        }
        const data = await response.json();
        setTopics(data.topics || data); // Supporta sia { topics: [...] } che [...]
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    loadTopics();
  }, []);

  if (loading) {
    return (
      <MainLayout>
        <div className="container py-12">
          <div className="flex items-center justify-center min-h-64">
            <div className="text-lg">Caricamento argomenti...</div>
          </div>
        </div>
      </MainLayout>
    );
  }

  if (error) {
    return (
      <MainLayout>
        <div className="container py-12">
          <div className="flex items-center justify-center min-h-64">
            <div className="text-lg text-red-500">Errore: {error}</div>
          </div>
        </div>
      </MainLayout>
    );
  }

  return (
    <MainLayout>
      <div className="container py-12">
        <div className="mb-10">
          <h1 className="text-4xl font-bold mb-4">Teoria</h1>
          <p className="text-lg text-muted-foreground w-full leading-relaxed">
            Immergiti nel mondo del machine learning attraverso guide complete ed esempi pratici. 
            La nostra sezione teorica è progettata per offrirti una comprensione approfondita dei 
            concetti fondamentali dell'intelligenza artificiale, dalla matematica di base agli 
            algoritmi più avanzati. Ogni argomento è strutturato con un approccio graduale che 
            ti accompagnerà dalle basi teoriche alle applicazioni pratiche, permettendoti di 
            costruire solide fondamenta per la tua carriera nel campo dell'AI.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {topics.map((topic) => {
            const IconComponent = iconMap[topic.icon] || BookOpen;
            
            return (
              <Card key={topic.id} className="card-hover relative">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="h-10 w-10 bg-primary/10 rounded-lg flex items-center justify-center mb-2">
                      <IconComponent className="h-5 w-5 text-primary" />
                    </div>
                    <Badge>{topic.badge}</Badge>
                  </div>
                  <CardTitle>{topic.title}</CardTitle>
                  <CardDescription>{topic.description}</CardDescription>
                </CardHeader>
                <CardContent className="pb-16">
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    {topic.subtopics.map((subtopic, index) => (
                      <li key={index} className="flex items-center">
                        <span className="h-1.5 w-1.5 rounded-full bg-primary mr-2"></span>
                        {subtopic}
                      </li>
                    ))}
                  </ul>
                </CardContent>
                <CardFooter className="absolute bottom-4 right-4 p-0">
                  <Link
                    to={`/theory/${topic.id}`}
                    className="px-4 py-2 bg-black text-white text-sm font-medium rounded-md hover:bg-gray-800 transition-colors"
                  >
                    Esplora questo argomento →
                  </Link>
                </CardFooter>
              </Card>
            );
          })}
        </div>
      </div>
    </MainLayout>
  );
};

export default Theory;