import json
import re
import argparse
import unicodedata
from pathlib import Path

SALIDA_DIR = Path(__file__).parent / "resultados"

NIVELES_SIMPLE = [
    ("TERCERO MEDIO Y CUARTO MEDIO", "Tercero y Cuarto Medio", ["3M","4M"]),
    ("PRIMERO BÁSICO",  "Primero Básico",  "1B"),
    ("SEGUNDO BÁSICO",  "Segundo Básico",  "2B"),
    ("TERCERO BÁSICO",  "Tercero Básico",  "3B"),
    ("CUARTO BÁSICO",   "Cuarto Básico",   "4B"),
    ("QUINTO BÁSICO",   "Quinto Básico",   "5B"),
    ("SEXTO BÁSICO",    "Sexto Básico",    "6B"),
    ("SÉPTIMO BÁSICO",  "Séptimo Básico",  "7B"),
    ("OCTAVO BÁSICO",   "Octavo Básico",   "8B"),
    ("PRIMERO MEDIO",   "Primero Medio",   "1M"),
    ("SEGUNDO MEDIO",   "Segundo Medio",   "2M"),
]

RANGOS_SIMPLE = [
    ("PRIMERO BÁSICO A SEXTO BÁSICO",  "1° Básico a 6° Básico",  ["1B","2B","3B","4B","5B","6B"]),
    ("SÉPTIMO BÁSICO A SEGUNDO MEDIO", "7° Básico a 2° Medio",   ["7B","8B","1M","2M"]),
]

SLUGS_ASIGNATURA = {
    "música":                                  "musica",
    "educación física y salud":                "educacion_fisica",
    "lenguaje y comunicación":                 "lenguaje",
    "matemática":                              "matematica",
    "historia, geografía y ciencias sociales": "ciencias_sociales",
    "ciencias naturales":                      "ciencias_naturales",
    "orientación":                             "orientacion",
    "tecnología":                              "tecnologia",
    "artes visuales":                          "artes_visuales",
    "inglés":                                  "ingles",
}

def nfc(texto):
    return unicodedata.normalize("NFC", texto)

def flat(texto):
    return re.sub(r"\s+", " ", nfc(texto))

def detectar_asignatura(texto_completo):
    patron = re.search(
        r"([A-ZÁÉÍÓÚÜÑa-záéíóúüñ ,]+)\s*\|\s*\d+|\d+\s*\|\s*([A-ZÁÉÍÓÚÜÑa-záéíóúüñ ,]+)",
        texto_completo
    )
    if not patron:
        return "desconocida", "desconocida"
    nombre = (patron.group(1) or patron.group(2)).strip()
    slug = SLUGS_ASIGNATURA.get(nombre.lower(), re.sub(r"\s+", "_", nombre.lower()))
    return nombre, slug

def detectar_nivel_pagina(texto_original, tiene_oa):
    tf = flat(texto_original)
    if tiene_oa:
        rangos_en_pagina = [nfc(r) for r, _, _ in RANGOS_SIMPLE if nfc(r) in tf]
        for clave, legible, codigo in NIVELES_SIMPLE:
            clave_n = nfc(clave)
            if clave_n in tf:
                es_parte_de_rango = any(clave_n in rango for rango in rangos_en_pagina)
                if not es_parte_de_rango:
                    return legible, codigo, "Basal"
    else:
        for clave, legible, codigo in RANGOS_SIMPLE:
            if nfc(clave) in tf:
                return legible, codigo, "Transversal"
    return None, None, None

def limpiar_boilerplate(texto):
    texto = re.sub(r"[\u00ad\-]\n([a-záéíóúüñ])", r"\1", texto)
    texto = re.sub(r"[A-ZÁÉÍÓÚÜÑa-záéíóúüñ ,]+\s*\|\s*\d+", "", texto)
    texto = re.sub(r"\d+\s*\|\s*[A-ZÁÉÍÓÚÜÑa-záéíóúüñ ,]+", "", texto)
    texto = re.sub(r"A continuación.*?Bases Curriculares\.", "", texto, flags=re.DOTALL)
    texto = re.sub(r"APRENDIZAJES BASALES", "", texto)
    texto = re.sub(r"APRENDIZAJES TRANSVERSALES\s*\d*", "", texto)
    texto = re.sub(r"^\s*\d{1,2}\s*$", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"Los Aprendizajes Transversales aluden.*?\.", "", texto, flags=re.DOTALL)
    texto = re.sub(r"(PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|OCTAVO)\s*\n\s*BÁSICO", "", texto)
    texto = re.sub(r"(PRIMERO|SEGUNDO|TERCERO|CUARTO)\s*\n\s*MEDIO", "", texto)
    texto = re.sub(r"TERCERO MEDIO Y CUARTO MEDIO", "", texto)
    texto = re.sub(r"EDUCACIÓN FÍSICA Y SALUD\s*[12]?", "", texto)
    texto = re.sub(r"PRIMERO BÁSICO A SEXTO BÁSICO", "", texto)
    texto = re.sub(r"SÉPTIMO BÁSICO A SEGUNDO MEDIO", "", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()

def chunker_generico(pdf_path):
    import fitz

    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    paginas = []
    for i, pagina in enumerate(doc):
        texto = pagina.get_text("text")
        if len(texto.strip()) > 50:
            paginas.append({"pagina": i + 1, "texto": texto})

    texto_completo = "\n".join(p["texto"] for p in paginas)
    nombre_asignatura, slug_asignatura = detectar_asignatura(texto_completo)
    print(f"Asignatura detectada: {nombre_asignatura} ({slug_asignatura})")

    chunks = []
    nivel_legible = None
    nivel_codigo  = None
    tipo_actual   = "Introduccion"

    def guardar_chunk(oa, contenido):
        if not contenido or len(contenido.strip()) < 30:
            return
        contenido = re.sub(r"\s+", " ", contenido).strip()
        codigo_str = ",".join(nivel_codigo) if isinstance(nivel_codigo, list) else (nivel_codigo or "")
        # transversales son del currículum general, no de una asignatura
        asignatura_chunk = "transversal" if tipo_actual == "Transversal" else slug_asignatura
        texto_chunk = f"{nivel_legible} — {oa}: {contenido}" if oa and nivel_legible else contenido
        chunks.append({
            "asignatura":    asignatura_chunk,
            "nivel_legible": nivel_legible or "",
            "nivel_codigo":  codigo_str,
            "tipo":          tipo_actual,
            "oa":            oa,
            "fuente":        pdf_path.name,
            "texto":         texto_chunk,
        })

    for p in paginas:
        texto_original = p["texto"]
        texto = limpiar_boilerplate(texto_original)
        tiene_oa = bool(re.search(r"^OA\s*\d+", texto_original, re.MULTILINE))

        nivel_det, codigo_det, tipo_det = detectar_nivel_pagina(texto_original, tiene_oa)

        if nivel_det:
            nivel_legible = nivel_det
            nivel_codigo  = codigo_det
            tipo_actual   = tipo_det

        # subsección específica Ed. Física (EFS 1 / EFS 2)
        match_sub = re.search(r"EDUCACIÓN FÍSICA Y SALUD\s*([12])", texto_original)
        if match_sub:
            sufijo = f" — EFS {match_sub.group(1)}"
            if nivel_legible and sufijo not in nivel_legible:
                nivel_legible = nivel_legible + sufijo

        # dividir por OA dentro de la página
        bloques = re.split(r"(?:^|\n|\s)(OA\s*\d+)[\n\s]", texto, flags=re.MULTILINE)
        i = 0
        while i < len(bloques):
            bloque = bloques[i].strip()
            if re.match(r"^OA\s*\d+$", bloque):
                oa = re.sub(r"OA\s*(\d+)", r"OA \1", bloque)
                contenido = bloques[i+1].strip() if i+1 < len(bloques) else ""
                guardar_chunk(oa, contenido)
                i += 2
            else:
                if not tiene_oa and len(bloque) > 30:
                    guardar_chunk(None, bloque)
                i += 1

    SALIDA_DIR.mkdir(exist_ok=True)
    salida = SALIDA_DIR / f"chunks_{slug_asignatura}.json"
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Generados {len(chunks)} chunks → {salida}")
    return salida

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunker genérico para priorizaciones curriculares Mineduc")
    parser.add_argument("--pdf", required=True, help="Ruta al PDF de priorización")
    args = parser.parse_args()
    chunker_generico(args.pdf)
