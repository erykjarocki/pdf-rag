import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from src.adapters import supported_extensions
from src.ingest import index_document, ingest_folder
from src.qdrant_store import delete_collection, get_qdrant_client, list_collections
from src.retriever import format_fragments_for_prompt, search_book

app = FastAPI(title="PDF-RAG API", version="2.0.0")


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str
    book: str | None = None


class Fragment(BaseModel):
    """A single text fragment returned from search."""

    text: str
    book: str
    chapter: str
    start_page: str | int
    end_page: str | int


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""

    context: list[Fragment]
    formatted: str


class IngestPathRequest(BaseModel):
    """Request body for ingesting a file by path."""

    file_path: str
    reindex: bool = False


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""

    status: str
    book: str
    chunks: int
    format: str


class IngestFolderRequest(BaseModel):
    """Request body for ingesting a directory of documents."""

    directory: str
    reindex: bool = False


class IngestFolderResult(BaseModel):
    """Result for a single document in a folder ingestion."""

    name: str
    status: str
    chunks: int | None = None
    error: str | None = None


class IngestFolderResponse(BaseModel):
    """Response body for the /ingest-folder endpoint."""

    results: list[IngestFolderResult]
    total_indexed: int
    total_skipped: int
    total_errors: int


@app.post("/query")
def query(req: QueryRequest):
    """Search the knowledge base and return matching fragments."""
    fragments = search_book(req.question, book=req.book)
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


@app.get("/books")
def list_books():
    """List all indexed book collections."""
    return {"books": list_collections()}


@app.get("/collections")
def list_collections_endpoint():
    """List all indexed collections with metadata."""
    collections = list_collections()
    client = get_qdrant_client()
    result = []
    for c in sorted(collections):
        count_result = client.count(collection_name=c, exact=True)
        total = count_result.count if hasattr(count_result, "count") else 0
        result.append({"name": c, "chunks": total})
    return {"collections": result}


@app.delete("/books/{name}")
def remove_book(name: str):
    """Delete a collection from the knowledge base."""
    client = get_qdrant_client()
    collections = list_collections(client)
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    delete_collection(name, client)
    return {"status": "deleted", "book": name}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile | None = None, req: IngestPathRequest | None = None):
    """Ingest a document into the knowledge base.

    Accepts either a file upload or a file path. Provide one, not both.
    """
    if file is not None:
        # Save uploaded file to temp location
        suffix = os.path.splitext(file.filename or "upload.txt")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = index_document(tmp_path, reindex=False)
            return IngestResponse(
                status="indexed",
                book=result["book"],
                chunks=len(result["chunks"]),
                format=suffix.lstrip("."),
            )
        finally:
            os.unlink(tmp_path)

    if req is not None:
        if not os.path.exists(req.file_path):
            raise HTTPException(status_code=400, detail=f"File not found: {req.file_path}")
        try:
            result = index_document(req.file_path, reindex=req.reindex)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return IngestResponse(
            status="indexed",
            book=result["book"],
            chunks=len(result["chunks"]),
            format=os.path.splitext(req.file_path)[1].lstrip("."),
        )

    raise HTTPException(
        status_code=400,
        detail="Provide either a file upload or a file_path in the request body.",
    )


@app.post("/ingest-folder", response_model=IngestFolderResponse)
def ingest_folder_endpoint(req: IngestFolderRequest):
    """Ingest all supported documents from a directory.

    Scans the directory for files with supported extensions and indexes each.
    Returns per-file results with status (indexed/skipped/error).
    """
    if not os.path.isdir(req.directory):
        raise HTTPException(status_code=400, detail=f"Not a directory: {req.directory}")
    try:
        results = ingest_folder(req.directory, reindex=req.reindex)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return IngestFolderResponse(
        results=[IngestFolderResult(**r) for r in results],
        total_indexed=sum(1 for r in results if r["status"] == "indexed"),
        total_skipped=sum(1 for r in results if r["status"] == "skipped"),
        total_errors=sum(1 for r in results if r["status"] == "error"),
    )


@app.get("/formats")
def supported_formats():
    """List all supported document formats."""
    return {"formats": supported_extensions()}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
