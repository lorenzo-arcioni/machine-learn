#!/bin/bash

# Colori per output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Funzioni per output colorato
print_step() {
    echo -e "${BLUE}🔄 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_separator() {
    echo -e "${BLUE}════════════════════════════════════════${NC}"
}

# Inizio build
print_separator
echo -e "${GREEN}🚀 Starting Machine Learning App Build Process${NC}"
print_separator

# Step 1: Update content
print_step "Updating content files..."
if ./update_content.sh; then
    print_success "Content updated successfully"
else
    print_error "Failed to update content"
    exit 1
fi

# Step 2: Generate static content
print_step "Generating static content from Markdown..."
if python build_script.py; then
    print_success "Static content generated"
else
    print_error "Failed to generate static content"
    exit 1
fi

# Step 3: Build React app
print_step "Building React application..."
if npm run build; then
    print_success "React app built successfully"
else
    print_error "Failed to build React app"
    exit 1
fi

# Step 4: Copy server file
print_step "Setting up development server..."
if cp server.py ./dist/; then
    print_success "Server file copied to dist/"
else
    print_error "Failed to copy server file"
    exit 1
fi

# Step 5: Final summary
print_separator
print_success "🎉 Build completed successfully!"
print_info "📁 Files ready in: ./dist/"
print_info "🌐 Starting development server..."
print_separator

# Step 6: Start server
print_step "Launching server at http://localhost:8000"
cd ./dist

python server.py