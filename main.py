import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse

from leadgen_model import do_work


valid_api_keys = ["8b60e192049fa9448fb57094"]

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def redirect():
    response = RedirectResponse(url='/leadtoopp')
    return response


@app.post("/leadtoopp", response_class=HTMLResponse)
def read_root(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...), model_type: str = Form(...), api_key: str=Form(...)):
    accept = request.headers.get("Accept")
    if api_key not in valid_api_keys:
        if accept and "text/html" in accept:
            return templates.TemplateResponse("index.html", {"request": request, "error_message": "Invalid API Key", "error": True})
        else:
            return JSONResponse(content={"status_code": 401, "message": "Invalid API key"})
    prediction_results = do_work(file1.file, file2.file, model_type)
    
    if accept and "text/html" in accept:
        return templates.TemplateResponse("index.html", {"request": request, "message": "Request processed successfully", "data": prediction_results.to_html()})
    else:
        new_prediction_results = prediction_results.to_dict(orient="records")
        return JSONResponse(content={"message": "Request processed successfully", "data": new_prediction_results})


@app.get("/leadtoopp", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == '__main__':
    uvicorn.run("debug_server:app", host="0.0.0.0", port=80, reload=True)
