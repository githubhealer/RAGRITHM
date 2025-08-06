import fastapi
from routers import documents, embeddings, query

app = fastapi.FastAPI()

# Include routers
app.include_router(documents.router)
app.include_router(embeddings.router)
app.include_router(query.router)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
