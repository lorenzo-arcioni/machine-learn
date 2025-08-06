// build-seo-pages.js - Script per generare pagine HTML statiche per Vercel
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Template HTML ottimizzato per Vercel
const createHTMLPage = (data) => {
  const { title, content, description, url, keywords, topicTitle } = data;
  
  return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title} | ${topicTitle} | ML Theory</title>
    <meta name="description" content="${description}">
    <meta name="keywords" content="${keywords.join(', ')}">
    <meta name="robots" content="index, follow">
    
    <!-- Open Graph -->
    <meta property="og:title" content="${title}">
    <meta property="og:description" content="${description}">
    <meta property="og:url" content="${url}">
    <meta property="og:type" content="article">
    <meta property="og:site_name" content="Machine Learning Theory">
    
    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="${title}">
    <meta name="twitter:description" content="${description}">
    
    <!-- Canonical URL -->
    <link rel="canonical" href="${url}">
    
    <!-- Preload critical resources -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://cdnjs.cloudflare.com">
    
    <!-- JSON-LD Structured Data -->
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "Article",
      "headline": "${title}",
      "description": "${description}",
      "url": "${url}",
      "datePublished": "${new Date().toISOString()}",
      "author": {
        "@type": "Organization",
        "name": "ML Theory Platform"
      },
      "publisher": {
        "@type": "Organization",
        "name": "ML Theory Platform"
      }
    }
    </script>
    
    <!-- Critical CSS -->
    <style>
      body { 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
        line-height: 1.6; 
        margin: 0; 
        padding: 20px;
        background: #fafafa;
      }
      .container { 
        max-width: 800px; 
        margin: 0 auto; 
        background: white;
        padding: 40px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      }
      h1 { 
        color: #1a1a1a; 
        margin-bottom: 20px; 
        font-size: 2.5rem;
        line-height: 1.2;
      }
      .meta { 
        color: #666; 
        margin-bottom: 30px; 
        padding-bottom: 20px;
        border-bottom: 1px solid #eee;
      }
      .content h2, .content h3 { 
        color: #2c3e50; 
        margin-top: 40px; 
        margin-bottom: 16px;
      }
      .content p { 
        margin-bottom: 16px; 
        color: #333;
      }
      .content code { 
        background: #f8f9fa; 
        padding: 2px 6px; 
        border-radius: 4px; 
        font-size: 0.9em;
        color: #e83e8c;
      }
      .content pre { 
        background: #f8f9fa; 
        padding: 20px; 
        border-radius: 8px; 
        overflow-x: auto;
        border: 1px solid #e9ecef;
      }
      .react-redirect {
        position: fixed;
        top: 20px;
        right: 20px;
        background: #007acc;
        color: white;
        padding: 10px 20px;
        border-radius: 6px;
        text-decoration: none;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0,122,204,0.3);
        transition: transform 0.2s;
      }
      .react-redirect:hover {
        transform: translateY(-1px);
      }
      @media (max-width: 768px) { 
        body { padding: 10px; }
        .container { padding: 20px; }
        h1 { font-size: 2rem; }
        .react-redirect { position: static; display: block; text-align: center; margin-bottom: 20px; }
      }
    </style>
</head>
<body>
    <!-- Link per versione interattiva -->
    <a href="${url}" class="react-redirect">🚀 View Interactive Version</a>
    
    <div class="container">
        <article>
            <header>
                <h1>${title}</h1>
                <div class="meta">
                    <strong>Topic:</strong> ${topicTitle} | 
                    <strong>Updated:</strong> ${new Date().toLocaleDateString()}
                </div>
            </header>
            
            <div class="content">
                ${content}
            </div>
            
            <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee;">
                <p><strong>Keywords:</strong> ${keywords.join(', ')}</p>
                <p><small>This is the SEO-optimized version. <a href="${url}">Click here for the interactive experience</a>.</small></p>
            </footer>
        </article>
    </div>
    
    <!-- Vercel Analytics (opzionale) -->
    <script>
      // Track SEO page views
      if (window.gtag) {
        gtag('config', 'GA_TRACKING_ID', {
          page_title: '${title}',
          page_location: '${url}'
        });
      }
    </script>
</body>
</html>`;
};

// Topic configuration
const topicConfig = {
  "math-for-ml": {
    title: "Mathematics for Machine Learning",
    description: "Essential math concepts behind ML algorithms",
    keywords: ["mathematics", "machine learning", "linear algebra", "calculus", "statistics"]
  },
  introduction: {
    title: "Introduction to Machine Learning", 
    description: "Fundamentals and core concepts",
    keywords: ["machine learning", "introduction", "basics", "fundamentals", "AI"]
  },
  "supervised-learning": {
    title: "Supervised Learning",
    description: "Learning with labeled data", 
    keywords: ["supervised learning", "labeled data", "classification", "regression"]
  },
  "unsupervised-learning": {
    title: "Unsupervised Learning",
    description: "Discovering hidden patterns",
    keywords: ["unsupervised learning", "clustering", "dimensionality reduction"]
  },
  "deep-learning": {
    title: "Deep Learning",
    description: "Neural networks and beyond",
    keywords: ["deep learning", "neural networks", "CNN", "RNN", "transformers"]  
  },
  nlp: {
    title: "Natural Language Processing",
    description: "Processing and understanding human language",
    keywords: ["NLP", "natural language processing", "text analysis", "language models"]
  }
};

function extractTextFromHTML(html) {
  return html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim();
}

function generateDescription(content) {
  const text = extractTextFromHTML(content);
  let description = text.substring(0, 155);
  const lastSpace = description.lastIndexOf(' ');
  if (lastSpace > 100) {
    description = description.substring(0, lastSpace);
  }
  return description.trim() + '...';
}

function generateKeywords(content, baseKeywords) {
  const text = extractTextFromHTML(content).toLowerCase();
  const mlTerms = ['algorithm', 'model', 'data', 'neural', 'network', 'training', 'learning'];
  const found = mlTerms.filter(term => text.includes(term));
  return [...new Set([...baseKeywords, ...found])].slice(0, 8);
}

async function buildSEOPages() {
  const baseUrl = process.env.VERCEL_URL 
    ? `https://${process.env.VERCEL_URL}` 
    : 'http://localhost:3000';
  
  console.log('🔧 Building SEO pages for Vercel...');
  console.log('🌐 Base URL:', baseUrl);
  
  // Leggi la struttura
  let structure;
  try {
    const structureData = readFileSync('./public/data/structure.json', 'utf8');
    structure = JSON.parse(structureData);
  } catch (error) {
    console.error('❌ Error reading structure.json:', error);
    return;
  }
  
  let pageCount = 0;
  const sitemapUrls = [];
  
  // Crea directory per pagine SEO
  const seoDir = './public/seo';
  if (!existsSync(seoDir)) {
    mkdirSync(seoDir, { recursive: true });
  }
  
  function processCategory(category, basePath = '', topicId = '') {
    category.files.forEach(file => {
      try {
        const filePath = file.path.replace(/\.md$/, '');
        const fullPath = filePath;
        
        // Leggi contenuto JSON
        const contentPath = `./public/data/${filePath}.json`;
        const contentData = JSON.parse(readFileSync(contentPath, 'utf8'));
        
        // Genera dati SEO
        const topic = topicConfig[topicId];
        const description = generateDescription(contentData.content);
        const keywords = generateKeywords(contentData.content, topic?.keywords || []);
        const url = `${baseUrl}/theory/${fullPath}`;
        
        // Crea HTML
        const htmlData = {
          title: contentData.title,
          content: contentData.content,
          description,
          url,
          keywords,
          topicTitle: topic?.title || 'Machine Learning'
        };
        
        const html = createHTMLPage(htmlData);
        
        // Crea directory
        const pageDir = join(seoDir, 'theory', dirname(fullPath));
        if (!existsSync(pageDir)) {
          mkdirSync(pageDir, { recursive: true });
        }
        
        // Scrivi file HTML
        const outputPath = join(seoDir, 'theory', `${fullPath}.html`);
        writeFileSync(outputPath, html);
        
        sitemapUrls.push(url);
        pageCount++;
        
        console.log(`✅ Generated: /seo/theory/${fullPath}.html`);
        
      } catch (error) {
        console.error(`❌ Error processing ${file.path}:`, error.message);
      }
    });
    
    // Processa sottocategorie
    Object.entries(category.subcategories).forEach(([key, subCategory]) => {
      processCategory(subCategory, `${basePath}/${key}`, topicId);
    });
  }
  
  // Processa tutti i topic
  Object.entries(structure).forEach(([topicId, category]) => {
    console.log(`📂 Processing: ${topicId}`);
    processCategory(category, topicId, topicId);
  });
  
  // Genera sitemap
  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${sitemapUrls.map(url => `  <url>
    <loc>${url}</loc>
    <lastmod>${new Date().toISOString().split('T')[0]}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>`).join('\n')}
</urlset>`;
  
  writeFileSync('./public/sitemap.xml', sitemap);
  
  // Genera robots.txt
  const robots = `User-agent: *
Allow: /

Sitemap: ${baseUrl}/sitemap.xml

# Block internal paths
Disallow: /data/
Disallow: /_next/
Disallow: /api/
`;
  
  writeFileSync('./public/robots.txt', robots);
  
  console.log(`🎉 Generated ${pageCount} SEO pages!`);
  console.log('📄 Sitemap: /sitemap.xml');
  console.log('🤖 Robots: /robots.txt');
}

// Esegui se chiamato direttamente
if (import.meta.url === `file://${process.argv[1]}`) {
  buildSEOPages().catch(console.error);
}

export { buildSEOPages };