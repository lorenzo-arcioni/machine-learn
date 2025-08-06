# serve-spa.py (crea questo file)
#!/usr/bin/env python3
import http.server
import socketserver
import os
import sys
from urllib.parse import urlparse

class SPAHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        """Override per ridurre i log verbosi"""
        # Mostra solo errori importanti, non ogni richiesta
        if "404" in str(args) or "500" in str(args):
            super().log_message(format, *args)
    
    def do_GET(self):
        try:
            url_parts = urlparse(self.path)
            request_file_path = url_parts.path.strip('/')
            
            if not request_file_path:
                request_file_path = 'index.html'
            
            # Se il file esiste, servilo normalmente
            if os.path.exists(request_file_path) and os.path.isfile(request_file_path):
                return super().do_GET()
            
            # Se è una route che inizia con /theory, servi index.html
            if request_file_path.startswith('theory/'):
                self.path = '/index.html'
                return super().do_GET()
                
            # Per tutto il resto, comportamento normale
            return super().do_GET()
            
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            # Client ha chiuso la connessione, ignora silenziosamente
            pass
        except Exception as e:
            # Altri errori, log ma continua
            print(f"Request error (non-fatal): {e}", file=sys.stderr)

class SilentTCPServer(socketserver.TCPServer):
    """TCP Server che gestisce gli errori di connessione in modo silenzioso"""
    
    def handle_error(self, request, client_address):
        """Override per gestire errori comuni senza spam di log"""
        exc_type, exc_value = sys.exc_info()[:2]
        
        # Ignora errori comuni di connessione
        if exc_type in (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
            
        # Per altri errori, log minimale
        print(f"Server error from {client_address}: {exc_type.__name__}", file=sys.stderr)

if __name__ == "__main__":
    PORT = 8000
    
    # Permetti riuso dell'indirizzo per evitare "Address already in use"
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with SilentTCPServer(("", PORT), SPAHandler) as httpd:
            print(f"SPA Server running at http://localhost:{PORT}/")
            print("This server handles React Router routes correctly.")
            print("Press Ctrl+C to stop.")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        print("👋 Goodbye!")