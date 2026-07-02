from fastapi import FastAPI

app = FastAPI(
    title="Financial Complaints Classifier",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Financial Complaints Classifier API is running"
    }