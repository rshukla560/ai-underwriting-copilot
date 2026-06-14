from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.routes.analyze import router as analyze_router

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# register analysis routes with /api/v1 prefix
app.include_router(analyze_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "app": settings.app_name
    }