# RAGRITHM

## Setup

**Create a Virtual Env**

```bash
$ python -m venv .venv
```

**Switch to virtualenv**

- Linux / OSX

```bash
$ source .venv/bin/activate
```

- Windows

```cmd
.venv\Scripts\activate.bat
```

**Install Packages**

```bash
$ pip install -r requirements.txt
```

**Run FastAPI**

```bash
$ fastapi dev main.py
```

## Frontend (React + Vite)

A minimal UI is provided in `frontend/` to interact with the API.

1. Install Node.js 18+.
2. Install deps and run the dev server:

```bash
# from the project root
cd frontend
npm install
npm run dev
```

The app runs on http://localhost:5173 and expects the API at http://localhost:8000.
You can override the API base by creating `frontend/.env` with:

```ini
VITE_API_BASE=http://localhost:8000
```
