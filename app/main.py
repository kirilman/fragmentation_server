import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", port=8787, host="0.0.0.0")
