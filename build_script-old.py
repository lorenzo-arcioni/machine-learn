#!/usr/bin/env python3
"""
Script per generare contenuto statico da Markdown
Converte file Markdown in JSON per il frontend React, gestisce immagini e link interni
"""

import os
import json
import re
import shutil
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from pygments.formatters import HtmlFormatter

# Configurazione percorsi
CONTENT_DIR = "content/theory"
OUTPUT_DIR = "public/data"
IMAGES_OUTPUT_DIR = "public/images"

# CSS per syntax highlighting e styling
PYGMENTS_CSS = HtmlFormatter(style='default').get_style_defs('.codehilite')

CODE_STYLING_CSS = '''
/* Styling per blocchi di codice */
.codehilite {
    background: transparent !important;
    border-radius: 8px;
    overflow: hidden;
}
.codehilite pre {
    background: transparent !important;
    margin: 0 !important;
    padding: 20px !important;
    font-family: 'Monaco', 'Menlo', 'Consolas', monospace !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
    white-space: pre !important;
    overflow-x: auto !important;
    color: inherit !important;
}
.codehilite code {
    background: transparent !important;
    padding: 0 !important;
    font-family: inherit !important;
}
'''

COPY_BUTTON_CSS = '''
.code-wrapper { 
    position: relative; 
}
.copy-button {
    position: absolute; 
    top: 12px; 
    right: 12px; 
    padding: 6px 12px; 
    font-size: 12px;
    cursor: pointer; 
    border: none; 
    border-radius: 4px; 
    background: rgba(255,255,255,0.9);
    color: #374151; 
    transition: all 0.2s ease;
    font-weight: 500;
}
.copy-button:hover { 
    background: rgba(255,255,255,1);
    transform: translateY(-1px);
}
'''

COLLAPSIBLE_CSS = '''
details.code-container {
    border: 1px solid #e5e7eb; 
    border-radius: 12px; 
    background: #f9fafb;
    margin: 16px 0;
    transition: all 0.3s ease;
}
details.code-container summary {
    padding: 12px 16px;
    font-size: 14px; 
    color: #6b7280; 
    cursor: pointer; 
    outline: none; 
    user-select: none;
    font-weight: 500;
}
details.code-container[open] summary::after { 
    content: " (Hide Code)"; 
    color: #9ca3af; 
}
details.code-container:not([open]) summary::after { 
    content: " (Show Code)"; 
    color: #d1d5db; 
}
details.code-container .code-wrapper {
    padding: 0;
    margin: 0;
}
'''

def setup_directories():
    """Crea le directory di output e copia le immagini"""
    directories = [
        OUTPUT_DIR, 
        IMAGES_OUTPUT_DIR, 
        os.path.join(IMAGES_OUTPUT_DIR, "posts"), 
        os.path.join(IMAGES_OUTPUT_DIR, "tikz")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Copia immagini
    images_copied = 0
    image_sources = [
        ("static/images/posts", os.path.join(IMAGES_OUTPUT_DIR, "posts")),
        ("static/images/tikz", os.path.join(IMAGES_OUTPUT_DIR, "tikz"))
    ]
    
    for source_dir, dest_dir in image_sources:
        if os.path.exists(source_dir):
            extensions = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp')
            for filename in os.listdir(source_dir):
                if filename.lower().endswith(extensions):
                    shutil.copy2(os.path.join(source_dir, filename), os.path.join(dest_dir, filename))
                    images_copied += 1
    
    print(f"✓ Setup complete - copied {images_copied} images")

def build_file_map():
    """Crea mappa nome_file -> percorso per risolvere i link Obsidian"""
    files_map = {}
    if not os.path.exists(CONTENT_DIR):
        return files_map
    
    for root, _, files in os.walk(CONTENT_DIR):
        for filename in files:
            if filename.endswith('.md'):
                relative_path = os.path.relpath(os.path.join(root, filename), CONTENT_DIR).replace('\\', '/')
                key = os.path.splitext(filename)[0].lower()
                if key not in files_map:
                    files_map[key] = relative_path
    
    print(f"✓ Built file map with {len(files_map)} entries")
    return files_map

def protect_math_content(content):
    """Protegge il contenuto matematico durante la conversione"""
    math_blocks = []
    
    # Proteggi blocchi display math $$...$$
    def replace_display_math(match):
        math_blocks.append(match.group(1))
        return f'@@MATH_BLOCK_{len(math_blocks)-1}@@'
    
    # Proteggi math inline $...$
    def replace_inline_math(match):
        math_blocks.append(match.group(0))
        return f'@@MATH_INLINE_{len(math_blocks)-1}@@'
    
    content = re.sub(r'\$\$(.*?)\$\$', replace_display_math, content, flags=re.DOTALL)
    content = re.sub(r'(?<!\\)\$([^\$]*?)\$', replace_inline_math, content)
    
    return content, math_blocks

def restore_math_content(html_content, math_blocks):
    """Ripristina il contenuto matematico"""
    for i, math in enumerate(math_blocks):
        html_content = html_content.replace(f'@@MATH_BLOCK_{i}@@', f'$${math}$$')
        html_content = html_content.replace(f'@@MATH_INLINE_{i}@@', math)
    return html_content

def process_obsidian_links(html_content, files_map):
    """Converte link Obsidian [[filename]] in link HTML"""
    def replace_link(match):
        file_name = match.group(1).strip()
        display_text = match.group(2).strip() if match.group(2) else file_name
        
        # Pulisci il nome del file
        clean_name = file_name.replace('&rsquo;', '\'').lower()
        
        if clean_name in files_map:
            target_path = files_map[clean_name].replace('.md', '')
            link_url = f"/theory/{target_path}"
            return f'<a href="{link_url}" class="text-blue-600 hover:underline">{display_text}</a>'
        else:
            return f'<span class="text-gray-600">{display_text}</span>'
    
    return re.sub(r'\[\[(.*?)(?:\|(.*?))?\]\]', replace_link, html_content, flags=re.DOTALL)

def fix_image_paths(html_content):
    """Corregge i percorsi delle immagini"""
    def fix_src(match):
        full_tag, src = match.groups()
        
        if src.startswith('/images/'):
            return full_tag
        
        if src.startswith('/static/images/'):
            new_src = src.replace('/static/images/', '/images/')
            return full_tag.replace(f'src="{src}"', f'src="{new_src}"')
        
        # Determina la cartella in base al tipo di file
        filename = src.rsplit('/', 1)[-1]
        if '/tikz/' in src or filename.endswith('.svg'):
            new_src = f'/images/tikz/{filename}'
        else:
            new_src = f'/images/posts/{filename}'
        
        return full_tag.replace(f'src="{src}"', f'src="{new_src}"')
    
    return re.sub(r'(<img[^>]*src="([^"]+)"[^>]*>)', fix_src, html_content)

def convert_markdown_to_html(file_path, files_map):
    """Converte un file Markdown in HTML"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Estrai il titolo dalla prima riga
        lines = content.split('\n')
        title = "Untitled"
        if lines and lines[0].startswith('# '):
            title = lines[0][2:].strip()
            content = '\n'.join(lines[1:])  # Rimuovi la riga del titolo
        
        # Proteggi la matematica
        protected_content, math_blocks = protect_math_content(content)
        
        # Configura Markdown
        md = markdown.Markdown(
            extensions=[
                'fenced_code',
                CodeHiliteExtension(css_class='codehilite', guess_lang=False),
                'tables', 'extra', 'smarty', 'toc', 'admonition'
            ]
        )
        
        # Converti in HTML
        html_body = md.convert(protected_content)
        
        # Aggiungi wrapper per blocchi di codice complessi
        def wrap_code_blocks(match):
            code_block = match.group(0)
            # Solo per codice con syntax highlighting (ha molti span)
            if code_block.count('<span') > 3:
                copy_button = '''<button class="copy-button" onclick="
                    const code = this.parentElement.querySelector('pre');
                    if (code) {
                        navigator.clipboard.writeText(code.innerText);
                        this.textContent = 'Copied!';
                        setTimeout(() => this.textContent = 'Copy', 2000);
                    }
                ">Copy</button>'''
                
                return f'''<details class="code-container">
<summary>Code</summary>
<div class="code-wrapper">
{copy_button}
{code_block}
</div>
</details>'''
            return code_block
        
        html_with_wrappers = re.sub(
            r'<div class="codehilite".*?</div>', 
            wrap_code_blocks, 
            html_body, 
            flags=re.DOTALL
        )
        
        # Combina tutto il CSS e l'HTML
        full_css = f"{PYGMENTS_CSS}\n{CODE_STYLING_CSS}\n{COPY_BUTTON_CSS}\n{COLLAPSIBLE_CSS}"
        full_html = f"<style>{full_css}</style>\n{html_with_wrappers}"
        
        # Ripristina la matematica
        full_html = restore_math_content(full_html, math_blocks)
        
        # Post-processing
        full_html = re.sub(r'<p>\s*(\$\$.*?\$\$)\s*</p>', r'\1', full_html, flags=re.DOTALL)
        full_html = full_html.replace('\\_', '_').replace('\\$', '$')
        full_html = process_obsidian_links(full_html, files_map)
        full_html = fix_image_paths(full_html)
        
        return {
            "title": title,
            "content": full_html
        }
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

def build_directory_structure():
    """Costruisce la struttura gerarchica delle directory"""
    if not os.path.exists(CONTENT_DIR):
        return {}
    
    categories = {}
    
    for root, _, files in os.walk(CONTENT_DIR):
        relative_path = os.path.relpath(root, CONTENT_DIR)
        if relative_path == '.':
            continue
            
        md_files = []
        for filename in files:
            if filename.endswith('.md'):
                file_path = os.path.join(relative_path, filename).replace('\\', '/')
                md_files.append({
                    "name": filename.replace('.md', ''),
                    "path": file_path
                })
        
        if md_files:
            categories[relative_path] = md_files
    
    # Costruisci la gerarchia
    def build_hierarchy(categories):
        hierarchy = {'subcategories': {}, 'files': []}
        
        for path, files in categories.items():
            parts = path.split(os.sep)
            current_node = hierarchy
            
            for part in parts:
                if part not in current_node['subcategories']:
                    current_node['subcategories'][part] = {'subcategories': {}, 'files': []}
                current_node = current_node['subcategories'][part]
            
            current_node['files'] = files
        
        return hierarchy['subcategories']
    
    return build_hierarchy(categories)

def generate_static_content():
    """Funzione principale"""
    print("🚀 Generating static content...")
    
    # Setup
    setup_directories()
    files_map = build_file_map()
    structure = build_directory_structure()
    
    # Salva la struttura
    with open(os.path.join(OUTPUT_DIR, 'structure.json'), 'w', encoding='utf-8') as f:
        json.dump(structure, f, indent=2, ensure_ascii=False)
    
    # Processa i file
    total_files = 0
    processed_files = 0
    
    for root, _, files in os.walk(CONTENT_DIR):
        for filename in files:
            if not filename.endswith('.md'):
                continue
                
            total_files += 1
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, CONTENT_DIR)
            
            # Percorso di output
            json_filename = relative_path.replace('.md', '.json').replace('\\', '/')
            output_path = os.path.join(OUTPUT_DIR, json_filename)
            
            # Crea directory se necessaria
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Converti e salva
            content_data = convert_markdown_to_html(file_path, files_map)
            if content_data:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(content_data, f, indent=2, ensure_ascii=False)
                processed_files += 1
    
    print(f"🎉 Completed! Processed {processed_files}/{total_files} files")
    print(f"📁 Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_static_content()