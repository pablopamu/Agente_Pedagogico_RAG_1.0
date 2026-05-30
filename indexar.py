import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import argparse

ENTRADA       = None
CHROMA_PATH   = Path("/home/p/lab/agpedagogico/chroma/chroma_db")
COLECCION     = "curriculo_mineduc"
MODELO_NOMBRE = "paraphrase-multilingual-MiniLM-L12-v2"

def indexar():
    global ENTRADA
    with open(ENTRADA, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Cargando modelo de embeddings...")
    modelo = SentenceTransformer(MODELO_NOMBRE)

    cliente   = chromadb.PersistentClient(path=str(CHROMA_PATH))
    coleccion = cliente.get_or_create_collection(
        name=COLECCION,
        metadata={"hnsw:space": "cosine"}
    )

    textos    = [c["texto"] for c in chunks]
    metadatas = []
    ids       = []
    transversal_contador = {}

    for i, c in enumerate(chunks):
        nivel_codigo = c.get("nivel_codigo", "") or ""
        if isinstance(nivel_codigo, list):
            nivel_codigo = ",".join(nivel_codigo)

        metadatas.append({
            "asignatura":    c.get("asignatura", ""),
            "nivel_legible": c.get("nivel_legible", "") or "",
            "nivel_codigo":  nivel_codigo,
            "tipo":          c.get("tipo", "") or "",
            "oa":            c.get("oa", "") or "",
            "fuente":        c.get("fuente", ""),
        })

        asignatura_slug = c.get("asignatura", "sin-asignatura")
        oa_slug         = (c.get("oa") or "sin-oa").replace(" ", "")
        nivel_slug      = nivel_codigo if nivel_codigo else "sin-nivel"

        if c.get("asignatura") == "transversal":
            key = f"transversal_{nivel_slug}"
            transversal_contador[key] = transversal_contador.get(key, 0) + 1
            ids.append(f"transversal_{nivel_slug}_{transversal_contador[key]}")
        else:
            ids.append(f"{asignatura_slug}_{nivel_slug}_{oa_slug}_{i}")

    print(f"Generando embeddings para {len(textos)} chunks...")
    embeddings = modelo.encode(textos, show_progress_bar=True).tolist()

    coleccion.upsert(
        ids=ids,
        documents=textos,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Indexados {len(chunks)} chunks en colección '{COLECCION}'")
    print(f"Base de datos en: {CHROMA_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="Ruta al archivo de chunks JSON")
    args = parser.parse_args()
    ENTRADA = Path(args.chunks)
    indexar()
