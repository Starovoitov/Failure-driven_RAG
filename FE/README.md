# FE (React frontend)

Simple React UI for communicating with the FastAPI backend command endpoints.

## Run

1. Install dependencies:

```bash
npm install
```

2. Start frontend:

```bash
npm run dev
```

3. Open:

- `http://127.0.0.1:5173`

Backend expected default URL:
- `http://127.0.0.1:8000`

All commands run in async mode and stream live `stdout/stderr`
by polling backend task status.
