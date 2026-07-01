import os
import traceback
from typing import Any, Dict, List, Optional

from cogniteam.config.settings import settings


_rag_index: Optional[Any] = None


def _get_chroma_client():
    try:
        import chromadb
        persist_dir = os.path.join(settings.project_root, ".cogniteam", "chroma")
        os.makedirs(persist_dir, exist_ok=True)
        return chromadb.PersistentClient(path=persist_dir)
    except ImportError:
        print("  chromadb no instalado. Instala con: pip install chromadb")
        return None


def index_document(
    doc_id: str,
    text: str,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    print(f"\n-- [rag.index_document] Indexando '{doc_id}' (len={len(text)})")
    client = _get_chroma_client()
    if not client:
        return {"result": "Error: chromadb no disponible."}

    try:
        collection = client.get_or_create_collection(
            name="cogniteam_rag",
            metadata={"hnsw:space": "cosine"},
        )

        chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"doc_id": doc_id, "chunk": str(i), **(metadata or {})}
            for i in range(len(chunks))
        ]

        collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )
        return {"result": f"OK: {len(chunks)} chunks indexados para '{doc_id}'."}
    except Exception as e:
        print(f"ERROR index_document: {e}")
        traceback.print_exc()
        return {"result": f"Error indexando: {e}"}


def query_documents(query: str, n_results: int = 5) -> Dict[str, Any]:
    print(f"\n-- [rag.query_documents] Query: '{query[:100]}...'")
    client = _get_chroma_client()
    if not client:
        return {"result": "chromadb no disponible.", "results": []}

    try:
        collection = client.get_collection(name="cogniteam_rag")
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, 20),
        )

        output = []
        if results["documents"]:
            for i, (doc, meta, dist) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0] if results["distances"] else [0] * len(results["documents"][0]),
                )
            ):
                output.append({
                    "rank": i + 1,
                    "doc_id": meta.get("doc_id", "unknown") if meta else "unknown",
                    "score": round(1.0 - dist, 4),
                    "text": doc[:500],
                })

        return {"result": f"OK: {len(output)} resultados.", "results": output}
    except Exception as e:
        if "does not exist" in str(e):
            return {"result": "Colección vacía. Indexa documentos primero.", "results": []}
        print(f"ERROR query_documents: {e}")
        traceback.print_exc()
        return {"result": f"Error consultando: {e}", "results": []}


def delete_document(doc_id: str) -> Dict[str, str]:
    print(f"\n-- [rag.delete_document] '{doc_id}'")
    client = _get_chroma_client()
    if not client:
        return {"result": "chromadb no disponible."}
    try:
        collection = client.get_collection(name="cogniteam_rag")
        collection.delete(where={"doc_id": doc_id})
        return {"result": f"OK: '{doc_id}' eliminado."}
    except Exception as e:
        print(f"ERROR delete_document: {e}")
        return {"result": f"Error eliminando: {e}"}


def list_documents() -> Dict[str, Any]:
    print("\n-- [rag.list_documents]")
    client = _get_chroma_client()
    if not client:
        return {"result": "chromadb no disponible.", "docs": []}
    try:
        collection = client.get_collection(name="cogniteam_rag")
        all_meta = collection.get(include=["metadatas"])
        seen = set()
        docs = []
        for meta in all_meta["metadatas"]:
            if meta and meta.get("doc_id") not in seen:
                seen.add(meta["doc_id"])
                docs.append(meta["doc_id"])
        return {"result": f"OK: {len(docs)} documentos.", "docs": docs}
    except Exception as e:
        if "does not exist" in str(e):
            return {"result": "Colección vacía.", "docs": []}
        print(f"ERROR list_documents: {e}")
        return {"result": f"Error: {e}", "docs": []}
