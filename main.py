from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.api.enroll import router as enroll_router

app = FastAPI(title="Face Identification Server")

app.include_router(enroll_router)

app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static"
)
