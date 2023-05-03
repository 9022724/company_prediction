import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse

from leadgen_model import do_work

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def redirect():
    response = RedirectResponse(url='/leadtoopp')
    return response


@app.post("/leadtoopp", response_class=HTMLResponse)
def read_root(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...), model_type: str = Form(...)):
    prediction_results = do_work(file1.file, file2.file, model_type)
    return templates.TemplateResponse("index.html", {"request": request, "message": "Request processed successfully", "data": prediction_results.to_html()})


@app.get("/leadtoopp", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == '__main__':
    uvicorn.run("debug_server:app", host="0.0.0.0", port=80, reload=True)
