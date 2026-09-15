from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.models.attack_model import preload_models as preload_attack_models
from api.models.biencoder_model import loaded_models as loaded_biencoder_models
from api.models.biencoder_model import preload_models as preload_biencoder_models
from api.models.severity_model import preload_models
from api.routers.classification_router import router as classification_router
from api.routers.retrieval_router import router as retrieval_router

# Load models at import time. Combined with gunicorn's ``--preload`` flag this
# happens once in the master process, so forked workers share the weights via
# copy-on-write rather than each loading their own copy. Loading only: no
# inference may run here, because a torch forward pass before the fork leaves
# the workers with a broken OpenMP thread pool whose first inference hangs.
preload_models()
preload_attack_models()
preload_biencoder_models()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs in each worker process, after the fork: the first forward passes
    # (the bi-encoder's technique vectors) happen here, before requests are
    # accepted.
    for encoder in loaded_biencoder_models():
        encoder.warm_up()
    print("Models ready, accepting requests.", flush=True)
    yield
    print("Shutting down…", flush=True)


app = FastAPI(lifespan=lifespan)

# Include API routers
app.include_router(classification_router)
app.include_router(retrieval_router)
