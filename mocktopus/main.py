import os
import socket
import platform
import sys
import random
import subprocess
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel, Field, AliasChoices

app = FastAPI(
    title="Mocktopus API",
    description="A simple mock API with various endpoints for testing",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Hello from HOSTNAME endpoint"""
    hostname = socket.gethostname()
    return {"message": f"Hello from {hostname}"}

@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/echo")
async def echo(
    request: Request,
    any_: Optional[str] = Query(None, alias="any", description="Arbitrary parameter named 'any'"),
    should: Optional[str] = Query(None, description="Arbitrary parameter named 'should'"),
):
    """Echo query parameters. Declares `any` and `should` for documentation purposes."""
    # Return all query params, not only the declared ones
    return dict(request.query_params)

class ProxyRequest(BaseModel):
    """Model for proxy request body."""
    method: str = Field(default="GET", description="HTTP method to use (GET, POST, PUT, DELETE, PATCH)")
    url: str = Field(default="https://httpbin.org/get", description="Target URL to proxy to")
    body: Optional[Any] = Field(default={}, description="Optional JSON body for methods that support it")
    repeat: int = Field(
        default=1,
        ge=1,
        le=50,
        description="Number of times to repeat the request",
        validation_alias=AliasChoices("repate", "repeat"),
    )


@app.post("/proxy")
async def proxy(data: ProxyRequest):
    """Proxy requests to other URLs"""
    try:
        method = data.method.upper()
        url = data.url
        body = data.body
        repeat = data.repeat
        
        if method not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            raise HTTPException(status_code=400, detail="Invalid HTTP method")
        
        results = []
        async with httpx.AsyncClient() as client:
            for i in range(repeat):
                try:
                    if method == "GET":
                        response = await client.get(url)
                    elif method == "POST":
                        response = await client.post(url, json=body)
                    elif method == "PUT":
                        response = await client.put(url, json=body)
                    elif method == "DELETE":
                        response = await client.delete(url)
                    elif method == "PATCH":
                        response = await client.patch(url, json=body)
                    
                    results.append({
                        "attempt": i + 1,
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "content": response.text[:1000]  # Limit content length
                    })
                except Exception as e:
                    results.append({
                        "attempt": i + 1,
                        "error": str(e)
                    })
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug")
async def debug(request: Request):
    """Debug endpoint with extensive system information"""
    try:
        # Get system information
        system_info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "platform": platform.platform(),
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "python_compiler": platform.python_compiler(),
                "python_build": platform.python_build(),
            },
            "environment": {
                "python_path": sys.path,
                "executable": sys.executable,
                "version_info": list(sys.version_info),
                "api_version": sys.api_version,
                "hexversion": sys.hexversion,
            },
            "network": {
                "hostname": socket.gethostname(),
                "fqdn": socket.getfqdn(),
            },
            "process": {
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "cwd": os.getcwd(),
                "environ_keys": list(os.environ.keys()),
            },
            "request": {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "client": {
                    "host": request.client.host if request.client else None,
                    "port": request.client.port if request.client else None,
                },
                "query_params": dict(request.query_params),
            }
        }
        
        # Try to get additional system info
        try:
            # CPU info
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r") as f:
                    cpu_info = f.read()
                system_info["cpu_info"] = cpu_info[:2000]  # Limit length
        except:
            pass
        
        try:
            # Memory info
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo", "r") as f:
                    mem_info = f.read()
                system_info["memory_info"] = mem_info[:1000]  # Limit length
        except:
            pass
        
        try:
            # Disk info
            disk_info = subprocess.run(["df", "-h"], capture_output=True, text=True)
            if disk_info.returncode == 0:
                system_info["disk_info"] = disk_info.stdout
        except:
            pass
        
        try:
            # Network interfaces
            if os.path.exists("/proc/net/dev"):
                with open("/proc/net/dev", "r") as f:
                    net_dev = f.read()
                system_info["network_interfaces"] = net_dev[:1000]  # Limit length
        except:
            pass
        
        try:
            # Load average
            load_avg = os.getloadavg()
            system_info["load_average"] = load_avg
        except:
            pass
        
        try:
            # Uptime
            if os.path.exists("/proc/uptime"):
                with open("/proc/uptime", "r") as f:
                    uptime = f.read().strip()
                system_info["uptime"] = uptime
        except:
            pass
        
        return system_info
    
    except Exception as e:
        return {"error": str(e), "traceback": str(e.__traceback__)}

@app.get("/failure")
async def failure():
    """Return a 500 error"""
    raise HTTPException(status_code=500, detail="Intentional failure for testing")

@app.get("/success")
async def success():
    """Return a 200 success"""
    return {"status": "success", "message": "Operation completed successfully"}

@app.get("/random")
async def random_status():
    """Return random status codes (200 or 500)"""
    status_codes = [200, 500]
    status_code = random.choice(status_codes)
    
    if status_code == 200:
        return {"status": "success", "code": 200}
    else:
        raise HTTPException(status_code=500, detail="Random 500 error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
