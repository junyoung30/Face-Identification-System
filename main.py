from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from app.api.enroll import router as enroll_router
from app.api.identify import router as identify_router
from app.api.gallery import router as gallery_router
from app.config import CROPPED_DIR

app = FastAPI(title="Face Identification Server")

app.include_router(enroll_router)
app.include_router(identify_router)
app.include_router(gallery_router)

app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static"
)

app.mount(
    "/cropped",
    StaticFiles(directory=str(CROPPED_DIR)),
    name="cropped"
)

@app.get("/")
def home():
    return FileResponse("app/static/index.html")
