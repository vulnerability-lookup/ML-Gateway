from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.models.severity_model import preload_models
from api.routers.classification_router import router as classification_router

# Load models at import time. Combined with gunicorn's ``--preload`` flag this
# happens once in the master process, so forked workers share the weights via
# copy-on-write rather than each loading their own copy.
preload_models()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Models ready, accepting requests.")
    yield
    print("Shutting down…")


app = FastAPI(lifespan=lifespan)

# Include API routers
app.include_router(classification_router)
