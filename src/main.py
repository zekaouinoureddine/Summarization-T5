import sys
sys.path.append('./src/utils/')

from .utils.predict import predict
from .utils.utils import get_url_paragraphs

import time
from fastapi import FastAPI, Request


app = FastAPI(title="Serve BM-ASS T5 Model Using FastAPI")


@app.get("/")
def home():
    return "The API is working ..."

@app.get("/predict/")
async def prediction(article: str):

    try:
        article_text  = get_url_paragraphs(article) if article.startswith("http") else article

        start = time.time()
        important_paraghraph = predict(article_text)
        duration = time.time() - start

        return {
            "success": True,
            "result": {
                "article": str(article),
                "important_paraghraph": str(important_paraghraph),
                "time": str(duration)
            }
        }

    except Exception as error:
        return {
            "success": False,
            "errors": [str(error)]
        }