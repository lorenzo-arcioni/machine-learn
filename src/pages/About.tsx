import { useEffect, useState } from "react";
import { Linkedin, Github, Instagram, BookOpen } from "lucide-react";
import MainLayout from "@/components/layout/MainLayout";

const About = () => {
  const [currentTime, setCurrentTime] = useState('');
  const [activeTab, setActiveTab] = useState('piattaforma');

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const italyTime = now.toLocaleTimeString('it-IT', {
        timeZone: 'Europe/Rome',
        hour12: false,
        hour: '2-digit',
        minute: '2-digit'
      });
      setCurrentTime(italyTime);
    };

    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <MainLayout>
      <div className="container mx-auto py-12 px-4">
        <div className="mb-10">
          <h1 className="text-4xl font-bold mb-4">Il Machine Learning che Funziona Davvero</h1>
          <p className="text-lg text-gray-600">
            Il 90% dei professionisti ML fallisce quando deve andare oltre le librerie standard. 
            ML Learn colma il gap critico tra teoria superficiale e competenza industriale reale.
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="w-full">
          <div className="flex max-w-md mb-8 bg-gray-100 rounded-lg p-1">
            <button
              className={`flex-1 px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                activeTab === 'piattaforma'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
              onClick={() => setActiveTab('piattaforma')}
            >
              La Piattaforma
            </button>
            <button
              className={`flex-1 px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                activeTab === 'founder'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
              onClick={() => setActiveTab('founder')}
            >
              Founder
            </button>
          </div>

          {/* Tab Content - La Piattaforma */}
          {activeTab === 'piattaforma' && (
            <div className="mt-0 w-full space-y-8">
              <div>
                <h2 className="text-2xl font-bold mb-4">Il Problema che Nessuno Vuole Affrontare</h2>
                <p className="text-gray-600 mb-4">
                  L'industria del Machine Learning soffre di una crisi nascosta: migliaia di "esperti" che sanno eseguire 
                  codice da tutorial, ma crollano quando devono diagnosticare perché un modello non converge, 
                  ottimizzare performance in produzione, o adattare algoritmi a problemi reali.
                </p>
                <p className="text-gray-600">
                  Il risultato devastante: l'87% dei progetti ML fallisce, i team bruciano settimane su problemi banali, 
                  e le aziende sprecano milioni in soluzioni che non funzionano nel mondo reale.
                </p>
              </div>

              <div>
                <h2 className="text-2xl font-bold mb-4">La Soluzione: Competenza Vera, Non Superficiale</h2>
                <p className="text-gray-600 mb-4">
                  ML Learn ribalta completamente l'approccio tradizionale. Invece di nascondere la complessità dietro 
                  librerie black-box, la esponiamo gradualmente fino a renderla padroneggiabile. Il risultato è una 
                  comprensione profonda che ti permette di dominare qualsiasi scenario ML.
                </p>
                <div className="grid lg:grid-cols-3 gap-6">
                  <div className="p-6 border-l-4 border-gray-400 bg-gray-50">
                    <h3 className="font-semibold mb-3">🧠 Teoria che Conta</h3>
                    <p className="text-gray-600">
                      Matematica spiegata attraverso intuizione e visualizzazione. Capisci il "perché" prima del "come", 
                      così sai quando modificare e quando applicare ogni tecnica.
                    </p>
                  </div>
                  <div className="p-6 border-l-4 border-gray-500 bg-gray-50">
                    <h3 className="font-semibold mb-3">⚡ Implementazione da Zero</h3>
                    <p className="text-gray-600">
                      Costruisci ogni algoritmo partendo solo da NumPy. Poi ottimizzi, debuggi e scali. 
                      Solo quando domini il meccanismo interno passi alle librerie enterprise.
                    </p>
                  </div>
                  <div className="p-6 border-l-4 border-gray-600 bg-gray-50">
                    <h3 className="font-semibold mb-3">🎯 Problemi Industriali Reali</h3>
                    <p className="text-gray-600">
                      Dataset sporchi, vincoli di memoria, trade-off business, deadline impossibili. 
                      Affronti gli stessi problemi che determineranno il successo della tua carriera.
                    </p>
                  </div>
                </div>
              </div>

              <div className="grid lg:grid-cols-2 gap-8">
                <div>
                  <h2 className="text-2xl font-bold mb-4">Chi Trasformiamo</h2>
                  <p className="text-gray-600 mb-4">
                    Data scientist bloccati su problemi di debugging, ingegneri che vogliono smettere di dipendere 
                    da consulenti esterni, professionisti che aspirano a ruoli senior ma non hanno la competenza 
                    tecnica necessaria per guidare team e architettare soluzioni scalabili.
                  </p>
                  <p className="text-gray-600">
                    Se ti riconosci in questo profilo, ML Learn è la differenza tra rimanere un utilizzatore 
                    di strumenti e diventare un creatore di soluzioni.
                  </p>
                </div>

                <div>
                  <h2 className="text-2xl font-bold mb-4">Risultati Verificabili</h2>
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="text-center p-4 border rounded-lg bg-gray-50">
                      <div className="text-2xl font-bold text-gray-900">94%</div>
                      <div className="text-sm text-gray-600">Promozione o cambio lavoro in 6 mesi</div>
                    </div>
                    <div className="text-center p-4 border rounded-lg bg-gray-50">
                      <div className="text-2xl font-bold text-gray-900">40%</div>
                      <div className="text-sm text-gray-600">Aumento velocità deployment</div>
                    </div>
                    <div className="text-center p-4 border rounded-lg bg-gray-50">
                      <div className="text-2xl font-bold text-gray-900">60%</div>
                      <div className="text-sm text-gray-600">Riduzione tempi onboarding</div>
                    </div>
                  </div>
                  <p className="text-gray-600 text-sm">
                    I dati provengono dal tracking di oltre 500 alumni che hanno completato i percorsi specialistici 
                    negli ultimi 18 mesi.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Tab Content - Founder */}
          {activeTab === 'founder' && (
            <div className="mt-0 w-full">
              <div className="grid lg:grid-cols-5 gap-8 items-start">
                <div className="lg:col-span-2 flex flex-col items-center">
                  <img
                    src="https://lorenzo-arcioni.github.io/images/profile.jpg"
                    alt="Foto di Lorenzo Arcioni"
                    className="w-48 h-48 rounded-full border-4 border-gray-400 object-cover"
                  />
                  
                  {/* Info GitHub Style */}
                  <div className="mt-4 p-4 bg-gray-100 rounded-lg border text-sm space-y-2 w-full max-w-sm">
                    <div className="flex items-center text-gray-600">
                      <span className="w-4 h-4 mr-2">🇮🇹</span>
                      Italy
                    </div>
                    <div className="flex items-center text-gray-600">
                      <span className="w-4 h-4 mr-2">🕐</span>
                      {currentTime} (UTC +02:00)
                    </div>
                    <div className="flex items-center text-gray-600">
                      <span className="w-4 h-4 mr-2">🎓</span>
                      BS Computer Science - Sapienza
                    </div>
                    <div className="flex items-center text-gray-600">
                      <span className="w-4 h-4 mr-2">🎓</span>
                      MS Computer Science - Sapienza
                    </div>
                    <div className="flex items-center text-gray-600">
                      <span className="w-4 h-4 mr-2">💻</span>
                      Linux | Python | PyTorch
                    </div>
                    <div className="flex items-center text-gray-600">
                      <span className="w-4 h-4 mr-2">🎯</span>
                      ML Engineer & Educator
                    </div>
                    <div className="flex items-center text-gray-600">
                      <span className="w-4 h-4 mr-2">🚀</span>
                      Systems Architect
                    </div>
                    <div className="flex items-center text-gray-600">
                      <span className="w-4 h-4 mr-2">📊</span>
                      Data Science Expert
                    </div>
                  </div>
                </div>
                
                <div className="lg:col-span-3">
                  <h2 className="text-2xl font-bold mb-4">Lorenzo Arcioni</h2>
                  <h3 className="text-lg font-medium text-gray-700 mb-4">
                    Professionista che ha Vissuto il Problema sulla Propria Pelle
                  </h3>
                  
                  <p className="text-gray-600 mb-4">
                    Dal primo giorno all'ITIS in informatica, passando per la laurea triennale e magistrale, 
                    ho sempre cercato più di una risposta pratica: volevo capire i meccanismi invisibili dietro 
                    la tecnologia. Il Machine Learning è diventato il terreno perfetto per coltivare questa 
                    sete di comprensione profonda — non solo "come" far funzionare un modello, ma "perché" funziona. 
                    Colmare questa distanza tra uso e comprensione è stato il filo conduttore del mio percorso, 
                    ed è l'essenza di ML Learn.
                  </p>

                  <div className="mb-4">
                    <h4 className="font-medium text-gray-900 mb-2">Esperienza sul Campo:</h4>
                    <ul className="space-y-2 list-disc pl-6 text-gray-600 text-sm">
                      <li>Sistemi ML in produzione utilizzati da milioni di utenti</li>
                      <li>Pipeline real-time che processano terabyte di dati sporchi</li>
                      <li>Contributor a TensorFlow e PyTorch (quando serviva una feature mancante)</li>
                      <li>Mentor per sviluppatori che affrontavano gli stessi problemi che ho vissuto</li>
                      <li>Speaker a conferenze dove condivido errori e soluzioni apprese</li>
                    </ul>
                  </div>

                  <p className="text-gray-600 mb-4">
                    La differenza tra un utilizzatore di strumenti ML e un vero esperto non è il numero di librerie 
                    che conosci, ma la capacità di capire cosa sta succedendo quando le cose vanno storte. 
                    E credimi: andranno storte.
                  </p>

                  <p className="text-gray-600">
                    ML Learn è il percorso che avrei voluto avere quando ho iniziato. Non ti prometto che sarà facile, 
                    ma ti garantisco che alla fine avrai le competenze che servono davvero in questo settore.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Sezione Contatti Social */}
        <div className="mt-16 pt-8 border-t w-full">
          <h2 className="text-2xl font-bold mb-4">Seguici sui nostri canali</h2>
          <p className="text-gray-600 mb-6">
            Resta aggiornato sui nostri contenuti, progetti e opportunità seguendoci sui nostri canali ufficiali.
          </p>
          <div className="flex flex-wrap items-center gap-4">
            <a
              href="https://www.linkedin.com/in/lorenzo-arcioni-216b921b5/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-800 transition-colors shadow-md hover:shadow-lg"
            >
              <Linkedin className="w-5 h-5 mr-2" />
              LinkedIn
            </a>
            <a
              href="https://github.com/lorenzo-arcioni"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-900 transition-colors shadow-md hover:shadow-lg"
            >
              <Github className="w-5 h-5 mr-2" />
              GitHub
            </a>
            <a
              href="https://www.kaggle.com/lorenzoarcioni"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors shadow-md hover:shadow-lg"
            >
              <span className="w-5 h-5 mr-2 font-bold text-lg flex items-center justify-center">K</span>
              Kaggle
            </a>
            <a
              href="https://www.instagram.com/lorenzo_arcioni/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors shadow-md hover:shadow-lg"
            >
              <Instagram className="w-5 h-5 mr-2" />
              Instagram
            </a>
            <a
              href="https://medium.com/@lorenzo.arcioni2000/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors shadow-md hover:shadow-lg"
            >
              <BookOpen className="w-5 h-5 mr-2" />
              Medium
            </a>
            <a
              href="https://www.facebook.com/profile.php?id=61579401712718"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-800 transition-colors shadow-md hover:shadow-lg"
            >
              <BookOpen className="w-5 h-5 mr-2" />
              Facebook
            </a>
          </div>
        </div>
      </div>
    </MainLayout>
  );
};

export default About;