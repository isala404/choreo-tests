from fastapi import FastAPI, HTTPException
import random

app = FastAPI(
    title="Internal API",
    description="Project-scoped internal API endpoints for testing",
    version="0.1.0",
)


@app.get("/success")
async def success():
    return {
        "source": "internal-api",
        "status": "success",
        "message": "This is a success response from internal-api",
    }


@app.get("/failure")
async def failure():
    raise HTTPException(
        status_code=500,
        detail="This is a failure response from internal-api",
    )


@app.get("/random")
async def random_status():
    if random.choice([True, False]):
        return {
            "source": "internal-api",
            "status": "success",
            "message": "Random success from internal-api",
        }
    raise HTTPException(status_code=500, detail="Random failure from internal-api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
