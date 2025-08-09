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
        "*",  # adjust/remove wildcard for production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(blob.router)

@app.post("/hackrx/run")
async def hackrx_run_direct(request: blob.HackRxRunRequest):
    """Direct HackRx endpoint without /blob prefix"""
    return await blob.hackrx_run(request)

# Alias to match external webhook format requirement
@app.post("/api/v1/hackrx/run")
async def hackrx_run_v1(request: blob.HackRxRunRequest):
    return await blob.hackrx_run(request)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
