import React, { useState, useEffect } from 'react';
import MainLayout from '@/components/layout/MainLayout';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ShoppingBag, Package, Laptop, BookOpen, Cpu, Calculator, Code, BookText, Loader2, ShoppingCart } from "lucide-react";
import { toast } from '@/components/ui/use-toast';
import { useNavigate } from 'react-router-dom';

// Funzione per determinare l'icona in base alla categoria
const getCategoryIcon = (category: string) => {
  switch (category) {
    case "Machine Learning":
      return Cpu;
    case "Matematica":
      return Calculator;
    case "Programmazione":
      return Code;
    case "Algoritmi":
      return BookText;
    case "Natural Language Processing":
      return BookOpen;
    case "Hardware":
      return Laptop;
    case "Software":
      return Package;
    default:
      return ShoppingBag;
  }
};

const Shop = () => {
  const [productsByCategory, setProductsByCategory] = useState<Record<string, any[]>>({});
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate(); 

  useEffect(() => {
    const loadProducts = async () => {
      try {
        setIsLoading(true);
        
        // Carica il file JSON locale dalla cartella public
        const response = await fetch('/data/shop.json');
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const productsArray = await response.json();
        
        // Raggruppa i prodotti per categoria e aggiungi le icone
        const productsByCategory = productsArray.reduce((acc: Record<string, any[]>, product: any) => {
          const category = product.category;
          
          if (!acc[category]) {
            acc[category] = [];
          }
          
          acc[category].push({
            ...product,
            icon: getCategoryIcon(category),
            // Aggiungiamo uno stato predefinito se non è presente
            status: product.status || 'available'
          });
          
          return acc;
        }, {});
        
        setProductsByCategory(productsByCategory);
      } catch (err) {
        console.error('Error loading products:', err);
        setError('Impossibile caricare i prodotti. Riprova più tardi.');
        toast({
          title: 'Errore',
          description: 'Impossibile caricare i prodotti. Riprova più tardi.',
          variant: 'destructive'
        });
      } finally {
        setIsLoading(false);
      }
    };

    loadProducts();
  }, []);

  const handleProductAction = (product: any) => {
    console.log('Product action:', product._id); // Debug log
    
    // Se il prodotto ha un link esterno, aprilo in una nuova scheda
    if (product.link || product.url) {
      const link = product.link || product.url;
      window.open(link, '_blank', 'noopener,noreferrer');
      return;
    }
    
    // Altrimenti naviga alla pagina del prodotto
    const productId = product._id || product.id;
    if (productId) {
      navigate(`/shop/${productId}`);
    } else {
      console.error('Product ID is missing'); // Debug log
      toast({
        title: 'Errore',
        description: 'ID del prodotto non disponibile',
        variant: 'destructive'
      });
    }
  };

  const handleAddToCart = (product: any, e: React.MouseEvent) => {
    e.stopPropagation(); // Previeni la navigazione quando si clicca "Aggiungi al carrello"
    
    // Qui potresti implementare la logica per aggiungere al carrello
    toast({
      title: 'Prodotto aggiunto!',
      description: `${product.title} è stato aggiunto al carrello`,
      variant: 'default'
    });
  };

  const getButtonText = (product: any) => {
    if (product.status === 'coming_soon') return 'Prossimamente';
    if (product.status === 'out_of_stock') return 'Esaurito';
    if (product.link || product.url) return 'Vai al prodotto';
    return 'Vedi dettagli';
  };

  const isProductUnavailable = (product: any) => {
    return product.status === 'coming_soon' || product.status === 'out_of_stock';
  };

  return (
    <MainLayout>
      <div className="container mx-auto py-8">
        <div className="flex items-center gap-3 mb-8">
          <ShoppingBag className="h-8 w-8" />
          <h1 className="text-4xl font-bold">Shop</h1>
        </div>

        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <Loader2 className="h-8 w-8 animate-spin" />
            <span className="ml-2">Caricamento prodotti...</span>
          </div>
        ) : error ? (
          <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-md">
            {error}
          </div>
        ) : Object.keys(productsByCategory).length === 0 ? (
          <div className="text-center py-12">
            <ShoppingBag className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <p className="text-lg text-gray-600">Nessun prodotto disponibile al momento.</p>
          </div>
        ) : (
          Object.entries(productsByCategory).map(([category, products]) => (
            <div key={category} className="mb-12">
              <div className="flex items-center gap-2 mb-6">
                {React.createElement(getCategoryIcon(category), { className: "h-6 w-6" })}
                <h2 className="text-2xl font-semibold">{category}</h2>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {products.map((product) => (
                  <Card key={String(product._id)} className="overflow-hidden flex flex-col hover:shadow-lg transition-shadow">
                    <div className="h-48 overflow-hidden relative">
                      <img
                        src={product.image_url}
                        alt={product.title}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          // Fallback se l'immagine non carica
                          (e.target as HTMLImageElement).src = '/placeholder-product.jpg';
                        }}
                      />
                      {product.status === 'coming_soon' && (
                        <div className="absolute top-2 right-2 bg-yellow-500 text-white px-2 py-1 rounded-md text-xs font-medium">
                          Presto
                        </div>
                      )}
                      {product.status === 'out_of_stock' && (
                        <div className="absolute top-2 right-2 bg-red-500 text-white px-2 py-1 rounded-md text-xs font-medium">
                          Esaurito
                        </div>
                      )}
                      {product.price === 'Free' && product.status === 'available' && (
                        <div className="absolute top-2 right-2 bg-green-500 text-white px-2 py-1 rounded-md text-xs font-medium">
                          Gratis
                        </div>
                      )}
                    </div>
                    <CardHeader>
                      <CardTitle className="text-lg">{product.title}</CardTitle>
                      <CardDescription className="text-sm line-clamp-3">{product.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="flex-grow">
                      <div className="grid grid-cols-1 gap-2 text-sm">
                        {product.instructor && (
                          <div>
                            <span className="font-medium">Creatore:</span> {product.instructor}
                          </div>
                        )}
                        {product.duration && (
                          <div>
                            <span className="font-medium">Durata:</span> {product.duration}
                          </div>
                        )}
                        {product.level && (
                          <div>
                            <span className="font-medium">Livello:</span> {product.level}
                          </div>
                        )}
                        <div className="flex items-center justify-between mt-4">
                          <div className="text-lg font-bold text-primary">
                            {product.price}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                    <CardFooter className="flex gap-2">
                      <Button 
                        className="flex-1" 
                        variant={isProductUnavailable(product) ? 'secondary' : 'default'}
                        onClick={() => {
                          if (!isProductUnavailable(product)) {
                            handleProductAction(product);
                          }
                        }}
                        disabled={isProductUnavailable(product)}
                      >
                        {getButtonText(product)}
                      </Button>
                      {product.status === 'available' && product.price !== 'Free' && (
                        <Button 
                          variant="outline"
                          size="icon"
                          onClick={(e) => handleAddToCart(product, e)}
                          title="Aggiungi al carrello"
                        >
                          <ShoppingCart className="h-4 w-4" />
                        </Button>
                      )}
                    </CardFooter>
                  </Card>
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </MainLayout>
  );
};

export default Shop;