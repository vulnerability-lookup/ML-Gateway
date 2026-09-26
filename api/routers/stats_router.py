import os

from cachetools import _CacheInfo
from fastapi import APIRouter, Depends

from api.availability import require_enabled
from api.schemas import CacheStats, InferenceStats, StatsResponse
from api.services.classification_service import _cached_predict, _cached_predict_attack
from api.services.retrieval_service import _cached_embed
from api.throttle import INFERENCE_GATE

"""
Operational counters, so an operator can tell repeats from new work and
admitted from refused calls before deciding on a shared cache or more
capacity. Everything here is per worker process: each call is answered by
whichever worker accepted the connection, identified by ``pid``, so poll a
few times and add the figures up, or compare one pid over time.
"""

router = APIRouter(dependencies=[Depends(require_enabled)])


def _cache_stats(info: _CacheInfo) -> CacheStats:
    # The stubs type the sizes as floats (a cache may weigh entries); ours
    # count entries, so they are whole numbers.
    return CacheStats(
        hits=info.hits,
        misses=info.misses,
        size=int(info.currsize),
        maxsize=int(info.maxsize) if info.maxsize is not None else 0,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats_endpoint() -> StatsResponse:
    """Cache hit rates and inference gate counters of the answering worker.

    Runs no model and never queues. ``caches`` counts lookups per result
    cache since the worker started (an hour's TTL and 10,000 entries each);
    ``inference`` reports the gate's limits, its current occupancy and how
    many calls it has served or refused with 503.
    """
    return StatsResponse(
        pid=os.getpid(),
        caches={
            "severity": _cache_stats(_cached_predict.cache_info()),
            "attack_techniques": _cache_stats(_cached_predict_attack.cache_info()),
            "embeddings": _cache_stats(_cached_embed.cache_info()),
        },
        inference=InferenceStats(
            concurrency=INFERENCE_GATE.concurrency,
            queue=INFERENCE_GATE.queue,
            max_wait_seconds=INFERENCE_GATE.max_wait,
            service_time_ms=None if INFERENCE_GATE.service_time is None else INFERENCE_GATE.service_time * 1000,
            expected_wait_seconds=INFERENCE_GATE.expected_wait(),
            running=INFERENCE_GATE.running,
            waiting=INFERENCE_GATE.waiting,
            served=INFERENCE_GATE.served,
            refused=INFERENCE_GATE.refused,
        ),
    )
