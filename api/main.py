from contextlib import asynccontextmanager
from fastapi import FastAPI

import api.models.severity_model as sm
from api.routers.classification_router import router as classification_router

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all machine learning models at startup and clean up at shutdown.
    """
    print("Starting up...")
    sm.severity_model.load()
    print("Models loaded successfully!")
    yield
    print("Shutting down...")
    # Optional: Add any cleanup code here, e.g., closing connections
    # sm.severity_model.unload() # If your model has an unload method

app = FastAPI(lifespan=lifespan)

# Include API routers
app.include_router(classification_router)
