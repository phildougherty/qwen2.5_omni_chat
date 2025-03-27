from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .router import router

def create_app():
    app = FastAPI(title="Qwen2.5-Omni Speech Chat")
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For production, restrict to your frontend domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include router
    app.include_router(router)
    
    # Mount static files
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except:
        print("No static directory found. Skipping...")
        
    return app
