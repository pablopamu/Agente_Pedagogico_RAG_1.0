#!/usr/bin/env python3

"""
servidor.py — Laboratorio de agente pedagógico local
------------------------------------------------------
Correr con: python servidor.py
Luego abrir: http://localhost:5000
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import requests
import json
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ── RAG ──────────────────────────────────────────────────
CHROMA_PATH   = Path("/home/p/lab/agpedagogico/chroma/chroma_db")
COLECCION_RAG = "curriculo_mineduc"
MODELO_EMB    = "paraphrase-multilingual-MiniLM-L12-v2"

_modelo_emb      = None
_coleccion       = None
_cliente_chroma  = None

def init_rag():
    global _modelo_emb, _coleccion, _cliente_chroma
    try:
        _modelo_emb     = SentenceTransformer(MODELO_EMB)
        _cliente_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _coleccion      = _cliente_chroma.get_collection(COLECCION_RAG)
        print(f"[RAG] Colección '{COLECCION_RAG}' cargada — {_coleccion.count()} chunks")
    except Exception as e:
        print(f"[RAG] Error al inicializar: {e}")

app = Flask(__name__)

# =========================================================
# CONFIG INICIAL
# =========================================================

config = {
    "api_url":        "http://localhost:11434/v1/chat/completions",
    "model":          "qwen2.5:7b",
    "temperature":    0.6,
    "top_p":          0.92,
    "top_k":          40,
    "max_tokens":     800,
    "max_history":    8,
    "prompt_version": "v1",
    "umbral_rag":     0.35,
    "n_results":      8,
    "timeout":        180,
}

# =========================================================
# CARPETAS
# =========================================================

CARPETAS = [
    "logs", "guardados",
    "exitos", "errores", "dudosos",
    "jocosos", "interesantes", "para_revisar", "experimental"
]

for carpeta in CARPETAS:
    os.makedirs(carpeta, exist_ok=True)

# =========================================================
# ESTADO DE SESIÓN
# =========================================================

estado = {
    "system_prompt":    "",
    "historial":        [],
    "ultima_pregunta":  "",
    "ultima_respuesta": "",
    "turno":            0,
    "prompt_origen":    "ninguno",
    "rag_activo":       False,
}

# =========================================================
# DETECCIÓN DE NIVEL Y ASIGNATURA
# =========================================================

NIVELES_QUERY = {
    "primero básico":  "1B", "1° básico": "1B", "1ro básico": "1B", "1b": "1B",
    "segundo básico":  "2B", "2° básico": "2B", "2do básico": "2B", "2b": "2B",
    "tercero básico":  "3B", "3° básico": "3B", "3ro básico": "3B", "3b": "3B",
    "cuarto básico":   "4B", "4° básico": "4B", "4to básico": "4B", "4b": "4B",
    "quinto básico":   "5B", "5° básico": "5B", "5to básico": "5B", "5b": "5B",
    "sexto básico":    "6B", "6° básico": "6B", "6to básico": "6B", "6b": "6B",
    "séptimo básico":  "7B", "7° básico": "7B", "7mo básico": "7B", "7b": "7B",
    "octavo básico":   "8B", "8° básico": "8B", "8vo básico": "8B", "8b": "8B",
    "primero medio":   "1M", "1° medio":  "1M", "1ro medio":  "1M", "1m": "1M",
    "segundo medio":   "2M", "2° medio":  "2M", "2do medio":  "2M", "2m": "2M",
    "tercero medio":   "3M", "3° medio":  "3M", "3ro medio":  "3M", "3m": "3M",
    "cuarto medio":    "4M", "4° medio":  "4M", "4to medio":  "4M", "4m": "4M",
}

ASIGNATURAS_QUERY = {
    "música":                    "musica",
    "musica":                    "musica",
    "educación física":          "educacion_fisica",
    "educacion fisica":          "educacion_fisica",
    "ed. física":                "educacion_fisica",
    "ed fisica":                 "educacion_fisica",
    "educación fisica":          "educacion_fisica",
    "matemáticas":               "matematica",
    "matematicas":               "matematica",
    "matemática":                "matematica",
    "matematica":                "matematica",
    "lenguaje":                  "lengua_y_literatura",
    "lenguaje y comunicación":   "lengua_y_literatura",
    "lenguaje y comunicacion":   "lengua_y_literatura",
    "castellano":                "lengua_y_literatura",
    "lengua y literatura":       "lengua_y_literatura",
    "historia":                  "ciencias_sociales",
    "historia y geografía":      "ciencias_sociales",
    "historia y geografia":      "ciencias_sociales",
    "ciencias sociales":         "ciencias_sociales",
    "sociales":                  "ciencias_sociales",
}

def detectar_nivel_query(pregunta):
    texto = pregunta.lower()
    for patron, codigo in NIVELES_QUERY.items():
        if patron in texto:
            return codigo
    return None

def detectar_asignatura_query(pregunta):
    texto = pregunta.lower()
    for patron, codigo in ASIGNATURAS_QUERY.items():
        if patron in texto:
            return codigo
    return None

# =========================================================
# RAG
# =========================================================

def consultar_rag(pregunta):
    if _modelo_emb is None or _coleccion is None:
        return "", [], {}

    embedding         = _modelo_emb.encode(pregunta).tolist()
    nivel_detectado   = detectar_nivel_query(pregunta)
    asig_detectada    = detectar_asignatura_query(pregunta)
    transversal_query = any(p in pregunta.lower() for p in ["transversal", "transversales"])
    umbral            = config["umbral_rag"]

    meta_rag = {
        "nivel_detectado":      nivel_detectado,
        "asignatura_detectada": asig_detectada,
        "umbral_rag":           umbral,
        "n_results":            config["n_results"],
        "coleccion":            COLECCION_RAG,
    }

    if nivel_detectado:
        todos = _coleccion.get(include=["documents", "metadatas", "embeddings"])
        docs, metas, embeddings = todos["documents"], todos["metadatas"], todos["embeddings"]

        filtrados = []
        for doc, meta, emb in zip(docs, metas, embeddings):
            nivel_chunk = meta.get("nivel_codigo", "")
            if nivel_detectado not in nivel_chunk.split(","):
                continue
            if transversal_query and meta.get("tipo") != "Transversal":
                continue
            if not transversal_query and meta.get("tipo") == "Transversal":
                continue
            if asig_detectada and meta.get("asignatura") != asig_detectada:
                continue
            filtrados.append((doc, meta, emb))

        if not filtrados:
            return "", [], meta_rag

        q      = np.array(embedding)
        scored = []
        for doc, meta, emb in filtrados:
            e   = np.array(emb)
            sim = float(np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e)))
            scored.append((sim, doc, meta))
        scored.sort(reverse=True)

    else:
        where = None
        if asig_detectada and transversal_query:
            where = {"$and": [
                {"asignatura": {"$eq": asig_detectada}},
                {"tipo":       {"$eq": "Transversal"}}
            ]}
        elif asig_detectada:
            where = {"asignatura": {"$eq": asig_detectada}}

        resultados = _coleccion.query(
            query_embeddings=[embedding],
            n_results=config["n_results"],
            include=["documents", "metadatas", "distances"],
            **({"where": where} if where else {})
        )
        docs       = resultados["documents"][0]
        metas      = resultados["metadatas"][0]
        distancias = resultados["distances"][0]
        scored = [(1 - dist, doc, meta) for doc, meta, dist in zip(docs, metas, distancias)]

    contexto_lines = ["Contexto extraído del currículo oficial MINEDUC:\n"]
    fuentes        = []

    print(f"[RAG DEBUG] chunks en scored: {len(scored)}, nivel: {nivel_detectado}, asig: {asig_detectada}")
    for sim, doc, meta in scored:
        print(f"[RAG DEBUG] sim={sim:.3f} | {meta.get('oa','—')} | {meta.get('nivel_codigo','')}")
        if sim < umbral:
            continue
        contexto_lines.append(
            f"[{meta['nivel_legible']} — {meta.get('oa') or 'Transversal'}]\n{doc}\n"
        )
        fuentes.append({
            "nivel":      meta["nivel_legible"],
            "oa":         meta.get("oa") or "—",
            "tipo":       meta["tipo"],
            "fuente":     meta["fuente"],
            "similitud":  round(sim, 2),
            "asignatura": meta.get("asignatura", ""),
        })

    contexto = "\n".join(contexto_lines) if len(contexto_lines) > 1 else ""
    meta_rag["chunks_recuperados"] = fuentes
    return contexto, fuentes, meta_rag

# =========================================================
# UTILIDADES
# =========================================================

def limpiar_historial():
    if not estado["historial"]:
        return
    if estado["historial"][0]["role"] == "system":
        system       = [estado["historial"][0]]
        conversacion = estado["historial"][1:]
    else:
        system       = []
        conversacion = estado["historial"]
    max_msgs            = config["max_history"] * 2
    conversacion        = conversacion[-max_msgs:]
    estado["historial"] = system + conversacion


def log_total(role, content):
    registro = {
        "timestamp":      datetime.now().isoformat(),
        "prompt_version": config["prompt_version"],
        "role":           role,
        "content":        content
    }
    with open("logs/sesiones.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def cargar_modelos():
    ruta = "modelos.txt"
    if not os.path.isfile(ruta):
        return [config["model"]]
    with open(ruta, "r", encoding="utf-8") as f:
        modelos = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return modelos or [config["model"]]


def guardar_registro(categoria, etiqueta="", ficha=None, experimental=False):
    fecha         = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    etiqueta_slug = etiqueta.strip().replace(" ", "-").lower() if etiqueta else ""
    modelo_slug   = config["model"].replace("/", "-")
    origen_slug   = estado["prompt_origen"]

    partes = [categoria, modelo_slug, origen_slug]
    if etiqueta_slug:
        partes.append(etiqueta_slug)
    partes.append(fecha)
    nombre_base = "_".join(partes)

    conversacion = [m for m in estado["historial"] if m["role"] != "system"]

    data = {
        "timestamp":       fecha,
        "categoria":       categoria,
        "etiqueta":        etiqueta,
        "model":           config["model"],
        "prompt_origen":   estado["prompt_origen"],
        "temperature":     config["temperature"],
        "top_p":           config["top_p"],
        "top_k":           config["top_k"],
        "max_tokens":      config["max_tokens"],
        "max_history":     config["max_history"],
        "prompt_version":  config["prompt_version"],
        "rag_activo":      estado["rag_activo"],
        "coleccion":       COLECCION_RAG,
        "umbral_rag":      config["umbral_rag"],
        "n_results":       config["n_results"],
        **({"system_prompt_contenido": estado["system_prompt"]} if estado["prompt_origen"] == "custom"
           else {"system_prompt_archivo": estado["prompt_origen"]}),
        "ultima_pregunta":  estado["ultima_pregunta"],
        "ultima_respuesta": estado["ultima_respuesta"],
        "conversacion":     conversacion,
    }

    if ficha:
        data["ficha"] = ficha

    carpeta_map = {
        "exito":        "exitos",
        "error":        "errores",
        "dudoso":       "dudosos",
        "jocoso":       "jocosos",
        "interesante":  "interesantes",
        "para_revisar": "para_revisar"
    }
    carpeta = carpeta_map.get(categoria, "guardados")

    ruta_individual = f"{carpeta}/{nombre_base}.json"
    with open(ruta_individual, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with open("guardados/todos.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    if experimental and ficha:
        ruta_exp = f"experimental/{nombre_base}.json"
        with open(ruta_exp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return ruta_individual


def _construir_payload(stream=False):
    payload = {
        "model":       config["model"],
        "messages":    estado["historial"],
        "temperature": config["temperature"],
        "top_p":       config["top_p"],
        "max_tokens":  config["max_tokens"],
        "stream":      stream,
    }
    if config["top_k"] > 0:
        payload["top_k"] = config["top_k"]
    return payload


def preguntar_llm():
    """Llamada síncrona — devuelve texto completo."""
    if not estado["historial"]:
        raise ValueError("Historial vacío.")
    response = requests.post(
        config["api_url"],
        json=_construir_payload(stream=False),
        timeout=config["timeout"]
    )
    data = response.json()
    return data["choices"][0]["message"]["content"]


def preguntar_llm_stream():
    """Llamada streaming — genera fragmentos de texto a medida que llegan."""
    if not estado["historial"]:
        raise ValueError("Historial vacío.")
    response = requests.post(
        config["api_url"],
        json=_construir_payload(stream=True),
        timeout=config["timeout"],
        stream=True
    )
    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith("data: "):
            line = line[6:]
        if line.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(line)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                yield delta
        except (json.JSONDecodeError, KeyError):
            continue


def turnos_usados():
    conversacion = [m for m in estado["historial"] if m["role"] != "system"]
    return len(conversacion) // 2

# =========================================================
# RUTAS — interfaz
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")

# =========================================================
# RUTAS — API
# =========================================================

@app.route("/api/chat", methods=["POST"])
def chat():
    data        = request.get_json()
    mensaje     = data.get("mensaje", "").strip()
    usar_stream = data.get("stream", False)

    if not mensaje:
        return jsonify({"error": "Mensaje vacío"}), 400

    log_total("user", mensaje)

    fuentes     = []
    meta_rag    = {}
    mensaje_llm = mensaje

    if estado["rag_activo"]:
        contexto, fuentes, meta_rag = consultar_rag(mensaje)
        if contexto:
            mensaje_llm = f"{contexto}\n\nPregunta del docente: {mensaje}"
            print(f"\n[CHAT DEBUG] largo mensaje_llm: {len(mensaje_llm)} caracteres")

    estado["historial"].append({"role": "user", "content": mensaje_llm})
    limpiar_historial()

    # ── Modo streaming ────────────────────────────────────
    if usar_stream:
        def generar():
            respuesta_completa = []
            meta = {
                "tipo":          "meta",
                "fuentes":       fuentes,
                "rag_activo":    estado["rag_activo"],
                "meta_rag":      meta_rag,
                "turnos_usados": turnos_usados(),
                "max_history":   config["max_history"],
            }
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

            try:
                for fragmento in preguntar_llm_stream():
                    respuesta_completa.append(fragmento)
                    evento = {"tipo": "token", "texto": fragmento}
                    yield f"data: {json.dumps(evento, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'tipo': 'error', 'mensaje': str(e)})}\n\n"
                return

            respuesta = "".join(respuesta_completa)
            log_total("assistant", respuesta)
            estado["historial"].append({"role": "assistant", "content": respuesta})
            limpiar_historial()
            estado["ultima_pregunta"]  = mensaje
            estado["ultima_respuesta"] = respuesta
            estado["turno"] += 1

            fin = {"tipo": "fin", "turnos_usados": turnos_usados()}
            yield f"data: {json.dumps(fin, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generar()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control":     "no-cache",
                "X-Accel-Buffering": "no",
            }
        )

    # ── Modo síncrono (evaluador) ─────────────────────────
    try:
        respuesta = preguntar_llm()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    log_total("assistant", respuesta)
    estado["historial"].append({"role": "assistant", "content": respuesta})
    limpiar_historial()
    estado["ultima_pregunta"]  = mensaje
    estado["ultima_respuesta"] = respuesta
    estado["turno"] += 1

    return jsonify({
        "respuesta":     respuesta,
        "turnos_usados": turnos_usados(),
        "max_history":   config["max_history"],
        "fuentes":       fuentes,
        "rag_activo":    estado["rag_activo"],
        "meta_rag":      meta_rag,
    })


@app.route("/api/guardar", methods=["POST"])
def guardar():
    data         = request.get_json()
    categoria    = data.get("categoria", "exito")
    etiqueta     = data.get("etiqueta", "")
    ficha        = data.get("ficha", None)
    experimental = data.get("experimental", False)

    if not estado["ultima_pregunta"]:
        return jsonify({"error": "No hay conversación para guardar"}), 400

    ruta = guardar_registro(categoria, etiqueta, ficha, experimental)
    return jsonify({"ok": True, "archivo": ruta})


@app.route("/api/reset", methods=["POST"])
def reset():
    estado["historial"]        = ([{"role": "system", "content": estado["system_prompt"]}]
                                   if estado["system_prompt"] else [])
    estado["ultima_pregunta"]  = ""
    estado["ultima_respuesta"] = ""
    estado["turno"]            = 0
    return jsonify({"ok": True})


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "config":        config,
        "system_prompt": estado["system_prompt"],
        "prompt_origen": estado["prompt_origen"],
        "turnos_usados": turnos_usados(),
        "modelos":       cargar_modelos(),
    })


@app.route("/api/config", methods=["POST"])
def set_config():
    data   = request.get_json()
    campos = ["temperature", "top_p", "top_k", "max_tokens",
              "max_history", "model", "umbral_rag", "n_results", "timeout"]
    for campo in campos:
        if campo in data:
            config[campo] = data[campo]
    return jsonify({"ok": True, "config": config})


@app.route("/api/system_prompt", methods=["POST"])
def set_system_prompt():
    data   = request.get_json()
    nuevo  = data.get("prompt", "").strip()
    origen = data.get("origen", "custom")

    estado["system_prompt"] = nuevo
    estado["prompt_origen"] = origen

    if nuevo:
        if estado["historial"] and estado["historial"][0]["role"] == "system":
            estado["historial"][0] = {"role": "system", "content": nuevo}
        else:
            estado["historial"].insert(0, {"role": "system", "content": nuevo})
    else:
        if estado["historial"] and estado["historial"][0]["role"] == "system":
            estado["historial"].pop(0)

    return jsonify({"ok": True})


@app.route("/api/prompts", methods=["GET"])
def listar_prompts():
    archivos = []
    if os.path.isdir("prompts"):
        for f in sorted(os.listdir("prompts")):
            if f.endswith(".txt"):
                archivos.append(f)
    return jsonify({"prompts": archivos})


@app.route("/api/prompts/<nombre>", methods=["GET"])
def cargar_prompt(nombre):
    ruta = os.path.join("prompts", nombre)
    if not os.path.isfile(ruta):
        return jsonify({"error": "Archivo no encontrado"}), 404
    with open(ruta, "r", encoding="utf-8") as f:
        contenido = f.read()
    return jsonify({"contenido": contenido})


@app.route("/api/exportar", methods=["GET"])
def exportar():
    conversacion = [m for m in estado["historial"] if m["role"] != "system"]
    sesion = {
        "exportado":     datetime.now().isoformat(),
        "model":         config["model"],
        "prompt_origen": estado["prompt_origen"],
        "system_prompt": estado["system_prompt"],
        "rag_activo":    estado["rag_activo"],
        "umbral_rag":    config["umbral_rag"],
        "conversacion":  conversacion,
    }
    return jsonify(sesion)


@app.route("/api/rag/toggle", methods=["POST"])
def rag_toggle():
    estado["rag_activo"] = not estado["rag_activo"]
    return jsonify({"ok": True, "rag_activo": estado["rag_activo"]})


@app.route("/api/rag/status", methods=["GET"])
def rag_status():
    chunks = _coleccion.count() if _coleccion else 0
    return jsonify({
        "rag_activo": estado["rag_activo"],
        "chunks":     chunks,
        "coleccion":  COLECCION_RAG,
    })


@app.route("/api/rag/colecciones", methods=["GET"])
def listar_colecciones():
    if _cliente_chroma is None:
        return jsonify({"colecciones": []})
    cols = _cliente_chroma.list_collections()
    return jsonify({"colecciones": [c.name for c in cols], "activa": COLECCION_RAG})


@app.route("/api/rag/coleccion", methods=["POST"])
def cambiar_coleccion():
    global _coleccion, COLECCION_RAG
    data   = request.get_json()
    nombre = data.get("coleccion", "").strip()
    if not nombre:
        return jsonify({"error": "Nombre vacío"}), 400
    try:
        _coleccion    = _cliente_chroma.get_collection(nombre)
        COLECCION_RAG = nombre
        return jsonify({"ok": True, "coleccion": nombre, "chunks": _coleccion.count()})
    except Exception as e:
        return jsonify({"error": str(e)}), 404

# =========================================================
# ARRANQUE
# =========================================================

if __name__ == "__main__":
    print("\n========================================")
    print(" Laboratorio — agente pedagógico local")
    print("========================================")
    print(" Abrir: http://localhost:5000")
    print(" Ctrl+C para detener")
    print("========================================\n")
    init_rag()
    app.run(debug=False, port=5000)
