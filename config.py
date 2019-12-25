
LASER_GRPC_API_PORT = 8100
LASER_GRPC_API_WORKERS = 10
LASER_REST_API_PORT = 8000
LASER_GUNICORN_WORKERS = 2


# NOTE: The default values above are replaced with environment vars if available
import os
import sys
import inspect

cfg_module = sys.modules[__name__]
cfg_keys = [m[0] for m in inspect.getmembers(cfg_module) if m[0].startswith("LASER_")]
env_vars = dict(os.environ)

for key in cfg_keys:
    if key in env_vars and env_vars[key]:
        setattr(cfg_module, key, env_vars[key])
