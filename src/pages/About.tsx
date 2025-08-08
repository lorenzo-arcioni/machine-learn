import { useEffect } from "react";
import MainLayout from "@/components/layout/MainLayout";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const About = () => {

  // Scrolla in alto ogni volta che cambia il path (navigazione SPA)
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.pathname]);

  return (
    <MainLayout>
      <div className="container py-12">
        <div className="mb-10">
          <h1 className="text-4xl font-bold mb-4">Informazioni su ML Learn</h1>
          <p className="text-lg text-muted-foreground max-w-3xl">
            ML Learn rappresenta una rivoluzione nell'educazione digitale: la prima piattaforma che fonde l'eccellenza accademica 
            con l'innovazione tecnologica per trasformare radicalmente il modo in cui si apprende il machine learning.
          </p>
        </div>

        <Tabs defaultValue="piattaforma" className="w-full">
          <TabsList className="grid w-full max-w-md grid-cols-2 mb-8">
            <TabsTrigger value="piattaforma">La Piattaforma</TabsTrigger>
            <TabsTrigger value="founder">Founder</TabsTrigger>
          </TabsList>

          <TabsContent value="piattaforma" className="mt-0 max-w-4xl">
            <div className="space-y-8">
              <div>
                <h2 className="text-2xl font-bold mb-4">Una visione rivoluzionaria</h2>
                <p className="text-muted-foreground mb-4">
                  ML Learn nasce dalla convinzione che l'educazione al machine learning sia rimasta troppo a lungo ancorata a metodologie obsolete. 
                  Mentre il mondo accelera verso l'era dell'intelligenza artificiale, i metodi tradizionali di insegnamento creano un pericoloso 
                  divario tra teoria accademica e implementazione pratica nel mondo reale.
                </p>
                <p className="text-muted-foreground">
                  La nostra missione è audace: democratizzare l'accesso alle competenze ML più avanzate attraverso un ecosistema educativo 
                  che combina rigore scientifico, innovazione pedagogica e tecnologie all'avanguardia. Non ci limitiamo a insegnare algoritmi - 
                  formiamo i futuri architetti dell'intelligenza artificiale.
                </p>
              </div>

              <div>
                <h2 className="text-2xl font-bold mb-4">Metodologia pedagocica innovativa</h2>
                <p className="text-muted-foreground mb-4">
                  Il nostro approccio didattico si basa su un framework proprietario sviluppato attraverso anni di ricerca in neuroscienze cognitive 
                  e psicologia dell'apprendimento. Ogni elemento è progettato per massimizzare la ritenzione e l'applicabilità pratica:
                </p>
                <ul className="space-y-3 list-disc pl-6 text-muted-foreground">
                  <li>
                    <span className="font-medium text-foreground">Teoria Multidimensionale</span> - Spiegazioni stratificate che partono dai principi matematici 
                    fondamentali per arrivare alle implementazioni più sofisticate, supportate da visualizzazioni interattive e simulazioni in tempo reale
                  </li>
                  <li>
                    <span className="font-medium text-foreground">Coding Immersivo</span> - Laboratori virtuali dove gli studenti non si limitano a eseguire codice, 
                    ma architettano soluzioni complete, dalle pipeline di preprocessing ai deployment in produzione
                  </li>
                  <li>
                    <span className="font-medium text-foreground">Challenge Realistiche</span> - Progetti basati su dataset reali e scenari aziendali autentici, 
                    sviluppati in collaborazione con leader dell'industria tech
                  </li>
                  <li>
                    <span className="font-medium text-foreground">Mentorship AI-Assisted</span> - Sistema di tutoraggio intelligente che si adatta al ritmo 
                    di apprendimento individuale, fornendo feedback personalizzato e suggerimenti proattivi
                  </li>
                </ul>
              </div>

              <div>
                <h2 className="text-2xl font-bold mb-4">Chi trasformiamo</h2>
                <p className="text-muted-foreground mb-4">
                  ML Learn è progettato per catalizzare la trasformazione professionale di una vasta gamma di profili: dagli studenti universitari 
                  che vogliono distinguersi nel mercato del lavoro, ai data scientist che desiderano padroneggiare le tecniche più avanzate, 
                  fino ai dirigenti che devono guidare la trasformazione digitale delle loro aziende.
                </p>
                <p className="text-muted-foreground">
                  La nostra piattaforma adattiva riconosce che ogni learner ha un background unico. Attraverso assessment iniziali sofisticati 
                  e algoritmi di personalizzazione, creiamo percorsi di apprendimento su misura che ottimizzano tempo ed efficacia, 
                  garantendo che ogni utente raggiunga il suo massimo potenziale.
                </p>
              </div>

              <div>
                <h2 className="text-2xl font-bold mb-4">Ecosistema tecnologico avanzato</h2>
                <ul className="space-y-3 list-disc pl-6 text-muted-foreground">
                  <li>
                    <span className="font-medium text-foreground">Infrastruttura Cloud Scalabile</span> - Accesso 24/7 a GPU clusters dedicati 
                    per training di modelli complessi e sperimentazione avanzata
                  </li>
                  <li>
                    <span className="font-medium text-foreground">Community Intelligence</span> - Network globale di professionisti ML con sistema 
                    di reputazione gamificato e opportunità di collaborazione su progetti open source
                  </li>
                  <li>
                    <span className="font-medium text-foreground">Feedback Engine</span> - Sistema di valutazione automatica che analizza non solo 
                    la correttezza del codice, ma anche eleganza, efficienza e best practices
                  </li>
                  <li>
                    <span className="font-medium text-foreground">Industry Integration</span> - Partnership strategiche con le principali librerie 
                    ML (TensorFlow, PyTorch, Scikit-learn) e piattaforme cloud (AWS, GCP, Azure)
                  </li>
                  <li>
                    <span className="font-medium text-foreground">Career Acceleration</span> - Job board esclusivo con posizioni curate da aziende 
                    partner e sistema di certificazioni riconosciute dall'industria
                  </li>
                </ul>
              </div>

              <div>
                <h2 className="text-2xl font-bold mb-4">Impact measurable</h2>
                <p className="text-muted-foreground">
                  I risultati parlano chiaro: il 94% dei nostri learner avanzati ha ottenuto una promozione o ha cambiato lavoro entro 6 mesi dal completamento 
                  dei percorsi specialistici. Le aziende che assumono i nostri alumni riportano un incremento medio del 40% nella velocità di deployment 
                  di progetti ML e una riduzione del 60% dei tempi di onboarding per nuovi data scientist.
                </p>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="founder" className="mt-0 max-w-4xl">
            <div className="flex flex-col md:flex-row items-center md:items-start space-y-6 md:space-y-0 md:space-x-10">
              <div className="flex-shrink-0">
                <img
                  src="https://lorenzo-arcioni.github.io/images/profile.jpg"
                  alt="Foto di Lorenzo Arcioni"
                  className="w-48 h-48 rounded-full border-4 border-primary object-cover"
                />
              </div>
              <div>
                <h2 className="text-2xl font-bold mb-4 text-center md:text-left">Lorenzo Arcioni</h2>
                <h3 className="text-lg font-medium text-primary mb-4 text-center md:text-left">Visionario Tecnologico & Architetto dell'Innovazione Educativa</h3>
                
                <p className="text-muted-foreground mb-4">
                  Lorenzo Arcioni non è solo un ingegnere del software - è un visionario che ha anticipato la convergenza tra intelligenza artificiale 
                  ed educazione digitale prima che diventasse mainstream. Con oltre un decennio di esperienza nell'architettare soluzioni tecnologiche 
                  che ridefiniscono interi settori, Lorenzo ha costruito la sua reputazione come pioniere nell'intersezione tra deep learning, 
                  user experience e scalabilità enterprise.
                </p>

                <p className="text-muted-foreground mb-4">
                  Il suo background multidisciplinare in ingegneria informatica, neuroscienze cognitive e design thinking lo ha portato a sviluppare 
                  un approccio unico alla creazione di piattaforme educative. Prima di fondare ML Learn, Lorenzo ha contribuito allo sviluppo di 
                  sistemi di raccomandazione utilizzati da milioni di utenti e ha architettato pipeline di ML che processano terabyte di dati in tempo reale.
                </p>

                <div className="mb-4">
                  <h4 className="font-medium text-foreground mb-2">Expertise Distintive:</h4>
                  <ul className="space-y-2 list-disc pl-6 text-muted-foreground text-left">
                    <li>
                      <span className="font-medium">Architetture ML Scalabili</span> - Progettazione di sistemi distribuiti per training 
                      e inference di modelli complessi
                    </li>
                    <li>
                      <span className="font-medium">Educational Technology</span> - Ricerca applicata in adaptive learning 
                      e personalizzazione algoritmica
                    </li>
                    <li>
                      <span className="font-medium">Open Source Leadership</span> - Maintainer di librerie Python con oltre 50K download 
                      e contributore attivo a progetti TensorFlow
                    </li>
                    <li>
                      <span className="font-medium">Industry Recognition</span> - Speaker keynote in conferenze internazionali (PyCon, MLOps World) 
                      e mentor in programmi di accelerazione tech
                    </li>
                    <li>
                      <span className="font-medium">Research Impact</span> - Pubblicazioni peer-reviewed su optimization algorithms 
                      e metodologie di transfer learning
                    </li>
                  </ul>
                </div>

                <p className="text-muted-foreground mb-4">
                  La filosofia di Lorenzo si riassume in una convinzione profonda: la tecnologia più sofisticata è inutile se non riesce 
                  a trasformare concretamente la vita delle persone. Questa visione lo ha guidato nella creazione di ML Learn come 
                  piattaforma che non si limita a trasferire conoscenze, ma catalizza trasformazioni professionali e personali.
                </p>

                <p className="text-muted-foreground mb-4">
                  Scopri di più sul suo percorso professionale e sui suoi contributi alla community tech visitando il suo portfolio 
                  (<a href="https://lorenzo-arcioni.github.io/" target="_blank" rel="noopener noreferrer" className="text-primary underline font-medium">lorenzo-arcioni.github.io</a>), 
                  dove condivide insights, case studies e visioni sul futuro dell'AI education.
                </p>

                <p className="text-muted-foreground">
                  Sotto la sua leadership visionaria, ML Learn sta ridefinendo gli standard dell'educazione tecnologica, costruendo 
                  il ponte tra il talento globale e le opportunità dell'era dell'intelligenza artificiale.
                </p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </MainLayout>
  );
};

export default About;