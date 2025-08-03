import fastapi


app = fastapi.FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}