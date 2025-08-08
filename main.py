import fastapi
from routers import blob , query

app = fastapi.FastAPI()

# Include routers
app.include_router(blob.router)
app.include_router(query.router)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
