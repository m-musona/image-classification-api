from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import shutil
from pathlib import Path

from classify import init, classify_image


# Create a FastAPI instance
app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Directory to save uploaded images
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Path to the HTML file
HTML_FILE = Path("templates/index.html")
results = {}

@app.get("/")
async def home(request: Request):
    """Endpoint to return a simple HTML page."""
    global results
    if not HTML_FILE.exists():
        raise HTTPException(status_code=404, detail="HTML file not found.")
    return templates.TemplateResponse("index.html", {"request": request, "results": results})


@app.post("/upload/")
async def upload_image(request: Request, file: UploadFile = File(...)):
    """Endpoint to upload an image file."""
    global results
    # Check file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    # Generate file path
    file_path = UPLOAD_DIR / f"image{Path(file.filename).suffix}"

    try:
        # Save the file to the upload directory
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        results = classify_image(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Redirect to the home page after upload
    return RedirectResponse(url="/", status_code=303) # status code 303 for redirecting as GET method

# Start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    init()
    uvicorn.run(app)
