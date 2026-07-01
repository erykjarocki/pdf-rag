#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastmcp import FastMCP
from retriever import search_book, format_fragments_for_prompt

mcp = FastMCP("book-rag")


@mcp.tool()
def search_book_tool(question: str) -> str:
    """Search the book knowledge base for relevant fragments.

    Use this whenever you need information from the book(s).
    Provide a detailed question to get the most relevant excerpts.
    Returns fragments with text and source (book, chapter, page).
    """
    fragments = search_book(question)
    if not fragments:
        return "No relevant fragments found in the knowledge base."
    return format_fragments_for_prompt(fragments)


@mcp.tool()
def search_book_raw(question: str) -> str:
    """Search the book knowledge base and return structured JSON.

    Use when you need the raw data including relevance scores.
    Returns JSON with text, book, chapter, page, and score.
    """
    import json
    fragments = search_book(question)
    return json.dumps(fragments, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
