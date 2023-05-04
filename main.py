import json
from oppornot import oppornot
from regenerate_model import regenerate_model
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
import secrets
import os
from leadgen_model import do_work



master_api_key = "448ab670efcfccc8f66f1236538219d6e705bb929116365a17e3abaa55731e68"

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def redirect():
    response = RedirectResponse(url='/leadtoopp')
    return response


@app.get("/generateapikey", response_class=HTMLResponse)
def generate_api_key(request: Request,  api_key: str=Form(...)):
    if api_key != master_api_key:
        return JSONResponse(content={"status_code": 401, "message": "Invalid API key"})
    random_bytes = secrets.token_bytes(32)
    api_key = random_bytes.hex()
    if os.path.isfile("api_keys.txt"):
        with open("api_keys.txt", "r") as f:
            json_string = f.read()
        api_key_list = json.loads(json_string)
        api_key_list.append(api_key)
        json_string = json.dumps(api_key_list)
        with open("api_keys.txt", "w") as f:
            f.write(json_string)
    else:
        json_string = json.dumps([api_key])
        with open("api_keys.txt", "w") as f:
            f.write(json_string)
    return JSONResponse(content={"status_code": 200, "api_key": api_key})



@app.post("/regeneratemodel", response_class=HTMLResponse)
def regeneratemodel(request: Request, file: UploadFile = File(...), api_key: str=Form(...)):
    with open("api_keys.txt", "r") as f:
        json_string = f.read()
    api_key_list = json.loads(json_string)
    if api_key not in api_key_list:
        return JSONResponse(content={"status_code": 401, "message": "Invalid API key"})
    regenerate_model(file.file)


@app.post("/oppornot", response_class=HTMLResponse)
def regeneratemodel(request: Request, file: UploadFile = File(...), api_key: str=Form(...)):
    with open("api_keys.txt", "r") as f:
        json_string = f.read()
    api_key_list = json.loads(json_string)
    if api_key not in api_key_list:
        return JSONResponse(content={"status_code": 401, "message": "Invalid API key"})
    prediction_results = oppornot(file.file)
    new_prediction_results = prediction_results.to_dict(orient="records")
    return JSONResponse(content={"message": "Request processed successfully", "data": new_prediction_results})
    

    


@app.post("/leadtoopp", response_class=HTMLResponse)
def read_root(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...), model_type: str = Form(...), api_key: str=Form(...)):
    accept = request.headers.get("Accept")
    if api_key != master_api_key:
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
