import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel
from retriever import search_book, format_fragments_for_prompt

app = FastAPI(title="Book RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    question: str


class Fragment(BaseModel):
    text: str
    book: str
    chapter: str
    start_page: str | int
    end_page: str | int


class QueryResponse(BaseModel):
    context: list[Fragment]
    formatted: str


@app.post("/query")
def query(req: QueryRequest):
    fragments = search_book(req.question)
    return QueryResponse(
        context=[
            Fragment(
                text=f["text"],
                book=f["book"],
                chapter=f.get("chapter", ""),
                start_page=f.get("start_page", ""),
                end_page=f.get("end_page", ""),
            )
            for f in fragments
        ],
        formatted=format_fragments_for_prompt(fragments),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
