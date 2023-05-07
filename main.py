import io
import json
import shutil
import zipfile
import tempfile
from pathlib import Path

from oppornot import oppornot
from regenerate_model import regenerate_model
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse, FileResponse
from starlette.datastructures import URL
import secrets
import os
from leadgen_model import do_work

MASTER_API_KEY = "448ab670efcfccc8f66f1236538219d6e705bb929116365a17e3abaa55731e68"

app = FastAPI()
templates = Jinja2Templates(directory="templates")
templates.env.globals['URL'] = URL


@app.get("/download-multiple")
async def download_files(filenames):
    if type(filenames) != list:
        filenames = eval(filenames)
    # List of file paths to download
    # Create a temporary directory to store the files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy files to the temporary directory
        for path in filenames:
            file_name = Path(path).name
            shutil.copy2(path, temp_path / file_name)

        # Create a zip file
        shutil.make_archive(temp_path, "zip", temp_path)

        # Return the zip file as a response
        return FileResponse(f"{temp_path}.zip", filename="files.zip", media_type="application/zip")


@app.get("/")
async def redirect():
    response = RedirectResponse(url='/leadtoopp')
    return response


@app.post("/leadtoopp/generateapikey")
def generate_api_key(request: Request, master_api_key: str = Form(...)):
    accept = request.headers.get("Accept")
    if master_api_key != MASTER_API_KEY:
        if accept and "text/html" in accept:
            return templates.TemplateResponse("generate_api_key.html", {"request": request, "message": "Invalid Master API Key. "})
        else:
            return JSONResponse(content={"status_code": 401, "message": "Invalid Master API key"})
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
    if accept and "text/html" in accept:
        return templates.TemplateResponse("generate_api_key.html", {"request": request, "api_key": api_key})
    else:
        return JSONResponse(content={"status_code": 200, "api_key": api_key})


@app.post("/leadtoopp/regeneratemodel", response_class=HTMLResponse)
def regeneratemodel(request: Request, file: UploadFile = File(...), api_key: str=Form(...)):
    accept = request.headers.get("Accept")
    with open("api_keys.txt", "r") as f:
        json_string = f.read()
    api_key_list = json.loads(json_string)
    if api_key not in api_key_list:
        if accept and "text/html" in accept:
            return templates.TemplateResponse("regenerate_model.html", {"request": request, "message": "Invalid API Key", "error": True})
        else:
            return JSONResponse(content={"status_code": 401, "message": "Invalid API key"})
    try:
        filenames = regenerate_model(file.file)
        if accept and "text/html" in accept:
            return templates.TemplateResponse("regenerate_model.html",
                                              {"request": request, "message": "Models has been regenerated \
                                                                                You can start using <b>oppornot</b> API.",
                                                "error": False, "status_code": 200, "files": filenames })
        else:
            return JSONResponse(content={"status_code": 200, "message": "Models has been regenerated. \
                                                                        You can start using <b>oppornot</b> API."})
    except:
        if accept and "text/html" in accept:
            return templates.TemplateResponse("regenerate_model.html",
                                              {"request": request, "message": "File Parsing Failed.",
                                                "error": True, "status_code": 400 })
        else:
            return JSONResponse(content={"status_code": 400, "message": "File Parsing Failed."})


@app.post("/leadtoopp/oppornot", response_class=HTMLResponse)
def opp_or_not(request: Request, file: UploadFile = File(...), api_key: str=Form(...)):
    accept = request.headers.get("Accept")
    with open("api_keys.txt", "r") as f:
        json_string = f.read()
    api_key_list = json.loads(json_string)
    if api_key not in api_key_list:
        if accept and "text/html" in accept:
            return templates.TemplateResponse("oppor_not.html",
                                              {"request": request, "message": "Invalid API Key", "error": True})
        else:
            return JSONResponse(content={"status_code": 401, "message": "Invalid API key"})
    try:
        prediction_results = oppornot(file.file)
        new_prediction_results = prediction_results.to_dict(orient="records")
        if accept and "text/html" in accept:
            return templates.TemplateResponse("oppor_not.html",
                                              {"request": request, "message": "success",
                                               "data": prediction_results.to_html(),
                                               "error": False, "status_code": 200 })
        else:
            return JSONResponse(content={"message": "Request processed successfully", "data": new_prediction_results})
    except:
        if accept and "text/html" in accept:
            return templates.TemplateResponse("oppor_not.html",
                                              {"request": request, "message": "File Parsing Failed.",
                                                "error": True, "status_code": 400 })
        else:
            return JSONResponse(content={"status_code": 400, "message": "File Parsing Failed."})


@app.post("/leadtoopp", response_class=HTMLResponse)
def read_root(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...), model_type: str = Form(...), api_key: str=Form(...)):
    accept = request.headers.get("Accept")
    if api_key != MASTER_API_KEY:
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
def get_leadtoop(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/leadtoopp/generateapikey", response_class=HTMLResponse)
def get_generateapikey(request: Request):
    return templates.TemplateResponse("generate_api_key.html", {"request": request})


@app.get("/leadtoopp/regeneratemodel", response_class=HTMLResponse)
def get_regeneratemodel(request: Request):
    return templates.TemplateResponse("regenerate_model.html", {"request": request})


@app.get("/leadtoopp/oppornot", response_class=HTMLResponse)
def get_oppornot(request: Request):
    return templates.TemplateResponse("oppor_not.html", {"request": request})


if __name__ == '__main__':
    uvicorn.run("debug_server:app", host="0.0.0.0", port=80, reload=True)
