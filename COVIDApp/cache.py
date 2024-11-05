# cache.py
from flask_caching import Cache

# Initialize Flask-Caching with configuration
cache = Cache(config={
    'CACHE_TYPE': 'simple',  # Simple in-memory cache
    'CACHE_DEFAULT_TIMEOUT': 300  # Cache timeout in seconds
})
