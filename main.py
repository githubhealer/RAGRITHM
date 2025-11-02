import fastapi
from fastapi.middleware.cors import CORSMiddleware
from routers import blob

app = fastapi.FastAPI()

# CORS middleware for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://ragrithm-frontend-*-uc.a.run.app",  # Allow frontend Cloud Run
        "*",  # adjust/remove wildcard for production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Increase file upload limits
app.router.route_class = fastapi.routing.APIRoute

app.include_router(blob.router)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
