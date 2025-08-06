import { useEffect, useState } from "react";
import { useParams, useNavigate, Link, useLocation } from "react-router-dom";
import MainLayout from "@/components/layout/MainLayout";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { ChevronLeft, Loader2, BookOpen, Brain, Network, Zap, Globe, ChevronRight, ChevronDown, Search, BookMarked, ArrowLeft, Book } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

// Tipizzazione globale per MathJax
declare global {
  interface Window {
    MathJax: any;
  }
}

interface ContentItem {
  name: string;
  path: string;
}

interface Category {
  subcategories: Record<string, Category>;
  files: ContentItem[];
}

interface TheoryContentResponse {
  title: string;
  content: string;
}

const topicConfig = {
  "math-for-ml": {
    title: "Mathematics for Machine Learning",
    icon: Brain,
    color: "from-gray-900 to-gray-800",
    bgPattern: "bg-gradient-to-br from-gray-50 to-gray-100",
    description: "Essential math concepts behind ML algorithms",
    badge: "Beginner"
  },
  introduction: {
    title: "Introduction to Machine Learning",
    icon: BookOpen,
    color: "from-gray-900 to-gray-800",
    bgPattern: "bg-gradient-to-br from-gray-50 to-gray-100",
    description: "Fundamentals and core concepts",
    badge: "Beginner"
  },
  "supervised-learning": {
    title: "Supervised Learning",
    icon: Brain,
    color: "from-gray-900 to-gray-800",
    bgPattern: "bg-gradient-to-br from-gray-50 to-gray-100",
    description: "Learning with labeled data",
    badge: "Intermediate"
  },
  "unsupervised-learning": {
    title: "Unsupervised Learning",
    icon: Network,
    color: "from-gray-900 to-gray-800",
    bgPattern: "bg-gradient-to-br from-gray-50 to-gray-100",
    description: "Discovering hidden patterns",
    badge: "Intermediate"
  },
  "deep-learning": {
    title: "Deep Learning",
    icon: Zap,
    color: "from-gray-900 to-gray-800",
    bgPattern: "bg-gradient-to-br from-gray-50 to-gray-100",
    description: "Neural networks and beyond",
    badge: "Advanced"
  },
  nlp: {
    title: "Natural Language Processing",
    icon: Globe,
    color: "from-gray-900 to-gray-800",
    bgPattern: "bg-gradient-to-br from-gray-50 to-gray-100",
    description: "Techniques for processing and understanding human language",
    badge: "Advanced"
  }
};

const TheoryTopic = () => {
  const { topicId } = useParams<{ topicId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const pathname = location.pathname;

  const [structure, setStructure] = useState<Record<string, Category>>({});
  const [isLoadingStructure, setIsLoadingStructure] = useState(true);
  const [content, setContent] = useState<TheoryContentResponse | null>(null);
  const [isLoadingContent, setIsLoadingContent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarExpanded, setSidebarExpanded] = useState(true);
  const [collapsedSections, setCollapsedSections] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState("");

  // Disable general page scrolling
  useEffect(() => {
    const originalBodyOverflow = document.body.style.overflow;
    const originalHtmlOverflow = document.documentElement.style.overflow;

    document.body.style.overflow = 'hidden';
    document.documentElement.style.overflow = 'hidden';

    return () => {
      document.body.style.overflow = originalBodyOverflow;
      document.documentElement.style.overflow = originalHtmlOverflow;
    };
  }, []);

  // Load MathJax config and script once on mount
  useEffect(() => {
    const configScript = document.createElement("script");
    configScript.type = "text/x-mathjax-config";
    configScript.text = `
      MathJax.Hub.Config({
        tex2jax: {
          inlineMath: [['$','$'], ['\\\\(','\\\\)']],
          displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
          processEscapes: true,
          processEnvironments: false
        },
        TeX: {
          Macros: {
            bm: ["\\\\boldsymbol{#1}", 1],
            argmin: "\\\\mathop{\\\\mathrm{argmin}}\\\\limits",
            argmax: "\\\\mathop{\\\\mathrm{argmax}}\\\\limits"
          }
        },
        "HTML-CSS": { 
          availableFonts: ["TeX"],
          webFont: "TeX"
        }
      });
    `;
    document.head.appendChild(configScript);

    const script = document.createElement("script");
    script.type = "text/javascript";
    script.async = true;
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML";
    document.head.appendChild(script);

    return () => {
      document.head.removeChild(configScript);
      document.head.removeChild(script);
    };
  }, []);

  // Force scroll to top when component mounts or location changes
  useEffect(() => {
    window.scrollTo({
      top: 0,
      left: 0,
      behavior: 'instant'
    });
    
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
    
    const timeoutId = setTimeout(() => {
      window.scrollTo({
        top: 0,
        left: 0,
        behavior: 'instant'
      });
      document.documentElement.scrollTop = 0;
      document.body.scrollTop = 0;
    }, 0);
    
    const animationTimeout = setTimeout(() => {
      window.scrollTo({
        top: 0,
        left: 0,
        behavior: 'instant'
      });
    }, 100);
    
    return () => {
      clearTimeout(timeoutId);
      clearTimeout(animationTimeout);
    };
  }, [location.pathname, location.key]);

  // Trigger MathJax render when content changes
  useEffect(() => {
    if (window.MathJax && window.MathJax.Hub) {
      window.MathJax.Hub.Queue(["Typeset", window.MathJax.Hub]);
    }
  }, [content]);

  const getContentPath = () => {
    if (!topicId) return null;
    const baseRoute = `/theory/${topicId}/`;
    if (pathname.startsWith(baseRoute)) {
      return pathname.substring(baseRoute.length);
    }
    return null;
  };

  // Utility function to normalize URLs for comparison
  const normalizeUrl = (url: string) => {
    return decodeURIComponent(url).replace(/\s+/g, ' ').trim();
  };

  // Function to check if a file is active
  const isFileActive = (file: ContentItem) => {
    const filePath = file.path.replace(/\.md$/, "");
    const linkPath = `/theory/${filePath}`;
    
    const normalizedCurrentPath = normalizeUrl(location.pathname);
    const normalizedLinkPath = normalizeUrl(linkPath);
    const normalizedFilePath = normalizeUrl(filePath);
    
    return normalizedCurrentPath === normalizedLinkPath || 
           normalizedCurrentPath.endsWith(normalizedFilePath);
  };

  // Load structure from static JSON
  useEffect(() => {
    const fetchStructure = async () => {
      try {
        const response = await fetch("/data/structure.json");
        if (!response.ok) {
          throw new Error(`Failed to load structure: ${response.status}`);
        }
        const data = await response.json();
        setStructure(data);
      } catch (err) {
        console.error("Failed to fetch theory structure:", err);
        setError("Failed to load the content structure. Please check if the static files have been generated.");
      } finally {
        setIsLoadingStructure(false);
      }
    };
    fetchStructure();
  }, []);

  // Load content from static JSON
  useEffect(() => {
    const contentPath = getContentPath();
    if (topicId && contentPath) {
      const fetchContent = async () => {
        setIsLoadingContent(true);
        try {
          // Costruisci il path al file JSON statico
          const jsonPath = `/data/${topicId}/${contentPath}.json`;
          const response = await fetch(jsonPath);
          
          if (!response.ok) {
            throw new Error(`Content not found: ${response.status}`);
          }
          
          const data = await response.json();
          setContent(data);
          setError(null);
        } catch (err) {
          console.error("Failed to fetch theory content:", err);
          setError("Failed to load the requested content. It might not exist or the static files haven't been generated yet.");
          setContent(null);
        } finally {
          setIsLoadingContent(false);
        }
      };
      fetchContent();
    } else {
      setContent(null);
    }
  }, [topicId, pathname]);

  const toggleSection = (sectionKey: string) => {
    const newCollapsed = new Set(collapsedSections);
    if (newCollapsed.has(sectionKey)) {
      newCollapsed.delete(sectionKey);
    } else {
      newCollapsed.add(sectionKey);
    }
    setCollapsedSections(newCollapsed);
  };

  const filterItems = (items: ContentItem[], term: string): ContentItem[] => {
    if (!term) return items;
    return items.filter(item => 
      item.name.toLowerCase().includes(term.toLowerCase())
    );
  };

  const hasVisibleContent = (category: Category, term: string): boolean => {
    const hasFiles = filterItems(category.files, term).length > 0;
    const hasSubcategories = Object.values(category.subcategories).some(sub => 
      hasVisibleContent(sub, term)
    );
    return hasFiles || hasSubcategories;
  };

  const renderNav = (category: Category, depth = 0, basePath = ""): JSX.Element => {
    const indent = depth * 16;
    const filteredFiles = filterItems(category.files, searchTerm);
    
    return (
      <div className="space-y-1">
        {Object.entries(category.subcategories).map(([key, subCategory]) => {
          const sectionPath = `${basePath}/${key}`;
          const isCollapsed = collapsedSections.has(sectionPath);
          const hasVisible = hasVisibleContent(subCategory, searchTerm);
          
          if (!hasVisible && searchTerm) return null;
          
          return (
            <div key={key} className="group">
              <div 
                style={{ paddingLeft: indent }} 
                className="flex items-center gap-2 font-semibold text-sm text-gray-700 dark:text-gray-300 py-2 px-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 shadow-sm hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors cursor-pointer"
                onClick={() => toggleSection(sectionPath)}
              >
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-6 h-6 p-0 hover:bg-gray-200 dark:hover:bg-gray-700"
                >
                  <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform duration-200 ${
                    isCollapsed ? '-rotate-90' : ''
                  }`} />
                </Button>
                <Globe className="w-4 h-4 text-gray-500" />
                <span className="flex-1">{key}</span>
                <span className="text-xs text-gray-400 bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded-full">
                  {subCategory.files.length + Object.keys(subCategory.subcategories).length}
                </span>
              </div>
              {!isCollapsed && (
                <div className="mt-2 space-y-1 animate-in slide-in-from-top-2 duration-200">
                  {renderNav(subCategory, depth + 1, sectionPath)}
                </div>
              )}
            </div>
          );
        })}
        {filteredFiles.map((file) => {
          const filePath = file.path.replace(/\.md$/, "");
          const linkPath = `/theory/${filePath}`;
          const isActive = isFileActive(file);
          
          return (
            <Link
              key={file.path}
              to={linkPath}
              style={{ paddingLeft: indent + 24 }}
              className={`group flex items-center gap-3 py-3 px-4 rounded-xl transition-all duration-200 hover:scale-[1.01] hover:shadow-sm ${
                isActive 
                  ? `bg-gradient-to-r from-gray-900 to-gray-800 dark:from-gray-100 dark:to-gray-200 text-white dark:text-gray-900 shadow-lg border-2 border-gray-700 dark:border-gray-400` 
                  : "hover:bg-gray-50 dark:hover:bg-gray-800/50 border border-transparent hover:border-gray-200 dark:hover:border-gray-700"
              }`}
            >
              <div className={`w-3 h-3 rounded-full transition-all duration-200 ${
                isActive ? 'bg-white dark:bg-gray-900 shadow-sm' : 'bg-gray-400 group-hover:bg-gray-600'
              }`} />
              <span className={`text-sm font-medium transition-colors flex-1 ${
                isActive ? 'text-white dark:text-gray-900' : 'text-gray-700 dark:text-gray-300 group-hover:text-gray-900 dark:group-hover:text-gray-100'
              }`}>
                {file.name}
              </span>
              {isActive && <BookMarked className="w-4 h-4 text-white dark:text-gray-900 ml-auto" />}
            </Link>
          );
        })}
      </div>
    );
  };

  const currentCategory = topicId && structure[topicId] ? structure[topicId] : null;
  const currentTopic = topicConfig[topicId as keyof typeof topicConfig];
  const IconComponent = currentTopic?.icon || BookOpen;

  return (
    <MainLayout hideFooter={true}>
      {/* Main Content - Full height panels with minimal padding */}
      <div className="w-full px-4 pt-4 pb-2 h-[calc(100vh-70px)]">
        <div className="flex gap-4 h-full">
          {/* Enhanced Sidebar - Con informazioni della macroarea */}
          <div className={`transition-all duration-300 ${sidebarExpanded ? 'w-[420px]' : 'w-0 overflow-hidden'} flex-shrink-0`}>
            <Card className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm border border-gray-200 dark:border-gray-700 shadow-xl h-full">
              <CardContent className="p-6 h-full flex flex-col">
                {/* Header con informazioni macroarea e pulsante back */}
                <div className="mb-6 pb-6 border-b border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between mb-4">
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      onClick={() => navigate("/theory")} 
                      className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 -ml-2"
                    >
                      <ArrowLeft className="mr-2 h-4 w-4" /> Back to Theory
                    </Button>
                    <Button 
                      variant="ghost" 
                      size="sm"
                      onClick={() => setSidebarExpanded(!sidebarExpanded)}
                      className="hover:bg-gray-100 dark:hover:bg-gray-800"
                    >
                      <ChevronRight className={`w-4 h-4 transition-transform ${sidebarExpanded ? 'rotate-180' : ''}`} />
                    </Button>
                  </div>
                  
                  {/* Informazioni della macroarea */}
                  {currentTopic && (
                    <div className="space-y-3">
                      <div className="flex items-center gap-3">
                        <div className={`p-2 rounded-lg bg-gradient-to-r ${currentTopic.color}`}>
                          <IconComponent className="w-5 h-5 text-white" />
                        </div>
                        <div className="flex-1">
                          <h3 className="font-bold text-lg text-gray-900 dark:text-gray-100 leading-tight">
                            {currentTopic.title}
                          </h3>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge variant="secondary" className="text-xs">
                              {currentTopic.badge}
                            </Badge>
                          </div>
                        </div>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                        {currentTopic.description}
                      </p>
                    </div>
                  )}
                </div>
                
                {/* Search Bar */}
                <div className="relative mb-6">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <Input
                    placeholder="Search topics..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700 focus:border-gray-400 dark:focus:border-gray-500"
                  />
                  {searchTerm && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setSearchTerm("")}
                      className="absolute right-2 top-1/2 transform -translate-y-1/2 w-6 h-6 p-0 hover:bg-gray-200 dark:hover:bg-gray-700"
                    >
                      ×
                    </Button>
                  )}
                </div>

                {/* Collapse/Expand All */}
                <div className="flex gap-2 mb-4">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCollapsedSections(new Set())}
                    className="flex-1 text-xs bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800"
                  >
                    Expand All
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (currentCategory) {
                        const allSections = new Set<string>();
                        const collectSections = (cat: Category, path = "") => {
                          Object.keys(cat.subcategories).forEach(key => {
                            const sectionPath = `${path}/${key}`;
                            allSections.add(sectionPath);
                            collectSections(cat.subcategories[key], sectionPath);
                          });
                        };
                        collectSections(currentCategory);
                        setCollapsedSections(allSections);
                      }
                    }}
                    className="flex-1 text-xs bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800"
                  >
                    Collapse All
                  </Button>
                </div>
                
                {isLoadingStructure ? (
                  <div className="flex items-center justify-center p-8">
                    <div className="flex flex-col items-center gap-3">
                      <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
                      <span className="text-sm text-gray-500">Loading topics...</span>
                    </div>
                  </div>
                ) : error && !currentCategory ? (
                  <div className="text-center p-8">
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
                      <span className="text-red-500 font-bold">!</span>
                    </div>
                    <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                  </div>
                ) : currentCategory ? (
                  <ScrollArea className="flex-1 pr-4">
                    <div className="space-y-2">
                      {renderNav(currentCategory)}
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="text-center p-8">
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                      <BookOpen className="w-8 h-8 text-gray-400" />
                    </div>
                    <p className="text-sm text-gray-500">No content available</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Toggle Button for Collapsed Sidebar */}
          {!sidebarExpanded && (
            <div className="fixed left-6 top-1/2 transform -translate-y-1/2 z-50">
              <Button 
                variant="outline"
                size="sm"
                onClick={() => setSidebarExpanded(true)}
                className="bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm hover:bg-white dark:hover:bg-gray-900 shadow-lg border-gray-300 dark:border-gray-600"
              >
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
          )}

          {/* Enhanced Content Area - Full height */}
          <div className="flex-1 min-w-0 h-full">
            <Card className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm border border-gray-200 dark:border-gray-700 shadow-xl h-full">
              <div className="h-full overflow-y-auto">
                <CardContent className="p-8">
                  {isLoadingContent ? (
                    <div className="flex items-center justify-center py-32">
                      <div className="flex flex-col items-center gap-4">
                        <div className="relative">
                          <div className="w-16 h-16 border-4 border-gray-200 dark:border-gray-700 rounded-full animate-spin border-t-gray-900 dark:border-t-gray-100"></div>
                          <BookMarked className="w-6 h-6 text-gray-500 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
                        </div>
                        <div className="text-center">
                          <h3 className="font-semibold text-gray-700 dark:text-gray-300">Loading Content</h3>
                          <p className="text-sm text-gray-500">Preparing your learning materials...</p>
                        </div>
                      </div>
                    </div>
                  ) : error ? (
                    <div className="p-8 rounded-2xl bg-gradient-to-r from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 border-l-4 border-red-500">
                      <div className="flex items-center gap-3 mb-2">
                        <div className="w-8 h-8 rounded-full bg-red-200 dark:bg-red-800 flex items-center justify-center">
                          <span className="text-red-700 dark:text-red-300 font-bold">!</span>
                        </div>
                        <h3 className="font-semibold text-red-800 dark:text-red-200">Content Error</h3>
                      </div>
                      <p className="text-red-700 dark:text-red-300 leading-relaxed">{error}</p>
                      <div className="mt-4 text-sm text-red-600 dark:text-red-400">
                        <p>Make sure you have:</p>
                        <ul className="list-disc list-inside mt-2 space-y-1">
                          <li>Run the build script: <code className="bg-red-200 dark:bg-red-800 px-1 rounded">python build-static-content.py</code></li>
                          <li>Generated the static files in the <code className="bg-red-200 dark:bg-red-800 px-1 rounded">public/data/</code> directory</li>
                          <li>Verified the content structure matches the expected format</li>
                        </ul>
                      </div>
                    </div>
                  ) : content ? (
                    <div className="space-y-6">
                      {/* Content Header */}
                      <div className="border-b border-gray-200 dark:border-gray-700 pb-6">
                        <div className="flex items-center gap-3 mb-3">
                          <div className="w-3 h-3 rounded-full bg-gradient-to-r from-emerald-500 to-emerald-600"></div>
                          <span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Static Content</span>
                        </div>
                        <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 dark:from-gray-100 dark:to-gray-300 bg-clip-text text-transparent mb-2">
                          {content.title}
                        </h1>
                      </div>
                      
                      {/* Enhanced Content Display */}
                      <article className="prose prose-lg max-w-none dark:prose-invert 
                        prose-headings:bg-gradient-to-r prose-headings:from-gray-900 prose-headings:to-gray-700 
                        dark:prose-headings:from-gray-100 dark:prose-headings:to-gray-300 
                        prose-headings:bg-clip-text prose-headings:text-transparent
                        prose-a:text-blue-600 dark:prose-a:text-blue-400 prose-a:no-underline hover:prose-a:underline
                        prose-code:bg-gray-100 dark:prose-code:bg-gray-800 prose-code:px-2 prose-code:py-1 prose-code:rounded prose-code:text-gray-800 dark:prose-code:text-gray-200
                        prose-pre:shadow-xl prose-pre:border prose-pre:border-gray-700 dark:prose-pre:border-gray-600
                        prose-blockquote:border-l-gray-600 dark:prose-blockquote:border-l-gray-400 prose-blockquote:bg-gray-50 dark:prose-blockquote:bg-gray-900/20 
                        prose-blockquote:py-4 prose-blockquote:px-6 prose-blockquote:rounded-r-lg
                        prose-table:border-collapse prose-th:border prose-th:border-gray-300 dark:prose-th:border-gray-600 prose-th:bg-gray-100 dark:prose-th:bg-gray-800
                        prose-td:border prose-td:border-gray-300 dark:prose-td:border-gray-600">
                        <div dangerouslySetInnerHTML={{ __html: content.content }} />
                      </article>
                    </div>
                  ) : (
                    <div className="text-center py-32">
                      <div className="max-w-md mx-auto">
                        <div className="w-24 h-24 mx-auto mb-6 rounded-full bg-gradient-to-r from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-700 flex items-center justify-center">
                          <BookOpen className="w-12 h-12 text-gray-400" />
                        </div>
                        <h3 className="text-xl font-semibold text-gray-700 dark:text-gray-300 mb-3">
                          Ready to Learn?
                        </h3>
                        <p className="text-gray-500 leading-relaxed">
                          Select a topic from the sidebar to begin your journey into machine learning concepts and theory.
                        </p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </MainLayout>
  );
};

export default TheoryTopic;