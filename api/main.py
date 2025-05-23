from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.routers.classification_router import router as classification_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all machine learning models at startup and clean up at shutdown.
    """
    print("Starting up...")
    # Models are loaded automatically at import time
    print("Models loaded successfully!")
    yield
    print("Shutting down...")
    # Add any cleanup code here if needed


app = FastAPI(lifespan=lifespan)

# Include API routers
app.include_router(classification_router)
