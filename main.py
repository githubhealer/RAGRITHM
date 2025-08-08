import fastapi
from routers import blob

app = fastapi.FastAPI()

app.include_router(blob.router)

@app.post("/hackrx/run")
async def hackrx_run_direct(request: blob.HackRxRunRequest):
    """Direct HackRx endpoint without /blob prefix"""
    return await blob.hackrx_run(request)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
