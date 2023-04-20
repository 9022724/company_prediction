import os
from multiprocessing import cpu_count

# Root Directory Path

ROOT_DIR = os.path.abspath(os.curdir)

# Socket Path

bind = f'unix:{ROOT_DIR}/gunicorn.sock'

# Worker Options

workers = cpu_count() + 1

worker_class = 'uvicorn.workers.UvicornWorker'

# Logging Options

loglevel = 'debug'

access_log = f'{ROOT_DIR}/access_log'

error_log = f'{ROOT_DIR}/error_log'
