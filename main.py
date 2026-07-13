# main.py reescrito

import base64
import hashlib
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Configuracao por ambiente
# ---------------------------------------------------------------------------

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL_DEFAULT", "gpt-4o")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
PORT = int(os.getenv("PORT", "8000"))


# ---------------------------------------------------------------------------
# Constantes e regex
# ---------------------------------------------------------------------------

ALLOWED_MIME = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
}

MIME_TO_DATA_URL_PREFIX = {
    "image/png": "data:image/png;base64,",
    "image/jpeg": "data:image/jpeg;base64,",
    "image/jpg": "data:image/jpeg;base64,",
    "image/webp": "data:image/webp;base64,",
    "image/gif": "data:image/gif;base64,",
}

REFUSAL_PHRASES = [
    "I can't help with that",
    "I cannot help with that",
    "I'm not able to help",
    "I am not able to help",
    "I can't assist with that",
    "I cannot assist with that",
]

HEADER = [
    "NOME",
    "NOME_MAE",
    "NOME_PAI",
    "NASCIMENTO",
    "DATA_OBITO",
    "HORA_OBITO",
    "CIDADE_OBITO",
    "UF_OBITO",
    "CEP",
    "CAUSA_MORTE",
    "CAUSA_MORTE_2",
    "CAUSA_MORTE_3",
    "CAUSA_MORTE_4",
    "CAUSA_MORTE_5",
    "CID_MORTE",
    "CID_MORTE_2",
    "CID_MORTE_3",
    "CID_MORTE_4",
    "CID_MORTE_5",
    "CODIGO_CAUSA_MORTE",
    "CODIGO_CAUSA_MORTE_2",
    "CODIGO_CAUSA_MORTE_3",
    "CODIGO_CAUSA_MORTE_4",
    "CODIGO_CAUSA_MORTE_5",
    "CAUSA_BASICA",
    "CID_BASICA",
    "CODIGO_CAUSA_BASICA",
]

UF_VALIDAS = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA",
    "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN",
    "RS", "RO", "RR", "SC", "SP", "SE", "TO",
}

MESES_EXTENSO = {
    "janeiro": "01",
    "fevereiro": "02",
    "marco": "03",
    "abril": "04",
    "maio": "05",
    "junho": "06",
    "julho": "07",
    "agosto": "08",
    "setembro": "09",
    "outubro": "10",
    "novembro": "11",
    "dezembro": "12",
}

STOP_CAUSAS = {
    "causa da morte",
    "causas da morte",
    "causa do obito",
    "causas do obito",
    "parte i",
    "parte ii",
    "causa basica",
    "cid",
    "cid basica",
    "codigo da causa",
    "codigo da causa basica",
}

DATE_RE = re.compile(
    r"(\d{1,2})[\s/\-\.]+(?:de\s+)?([A-Za-zç]+|[0-9]{1,2})[\s/\-\.]*(?:de\s+)?(\d{2,4})"
)
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")
CID_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,3})?)\b")
CEP_RE = re.compile(r"\b(\d{5})\-(\d{3})\b")
DURACAO_RE = re.compile(r"\b\d+\s*(?:minuto|minutos|hora|horas|dia|dias|semana|semanas|mes|meses|ano|anos)\b", re.IGNORECASE)
NUMERO_RE = re.compile(r"^[\d\s\.,/\-:]+$")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="OCR Declaracao de Obito", version="1.0.0")


# ---------------------------------------------------------------------------
# Helpers de erro
# ---------------------------------------------------------------------------

def _internal_error(request_id: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "requestId": request_id,
            "error": "INTERNAL_ERROR",
            "message": message,
        },
    )


def _bad_request(request_id: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "requestId": request_id,
            "error": "BAD_REQUEST",
            "message": message,
        },
    )


def _unauthorized(request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "requestId": request_id,
            "error": "UNAUTHORIZED",
            "message": "Token de autenticacao invalido ou ausente.",
        },
    )


# ---------------------------------------------------------------------------
# Helpers de texto
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_line(line: str) -> str:
    if not line:
        return ""
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    return line


def _remove_accents(text: str) -> str:
    replacements = {
        "á": "a", "à": "a", "â": "a", "ã": "a", "ä": "a",
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "í": "i", "ì": "i", "î": "i", "ï": "i",
        "ó": "o", "ò": "o", "ô": "o", "õ": "o", "ö": "o",
        "ú": "u", "ù": "u", "û": "u", "ü": "u",
        "ç": "c", "ñ": "n",
        "Á": "A", "À": "A", "Â": "A", "Ã": "A", "Ä": "A",
        "É": "E", "È": "E", "Ê": "E", "Ë": "E",
        "Í": "I", "Ì": "I", "Î": "I", "Ï": "I",
        "Ó": "O", "Ò": "O", "Ô": "O", "Õ": "O", "Ö": "O",
        "Ú": "U", "Ù": "U", "Û": "U", "Ü": "U",
        "Ç": "C", "Ñ": "N",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def _normalize_label(text: str) -> str:
    text = _remove_accents(text.lower())
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_like_label(line: str) -> bool:
    """Conservador: so considera label se terminar com ':' ou for rótulo curto conhecido."""
    if not line:
        return False
    normalized = _normalize_label(line)
    if not normalized:
        return False
    if line.strip().endswith(":"):
        return True
    # Rotulos curtos sem valor aparente
    if len(normalized) <= 40 and normalized in STOP_CAUSAS:
        return True
    return False


def _is_numeric_line(line: str) -> bool:
    if not line:
        return False
    return bool(NUMERO_RE.match(line.strip()))


def _is_duration_line(line: str) -> bool:
    if not line:
        return False
    return bool(DURACAO_RE.search(line))


def _is_cid_only(line: str) -> bool:
    if not line:
        return False
    stripped = line.strip()
    if not stripped:
        return False
    return bool(CID_RE.fullmatch(stripped)) or bool(re.fullmatch(r"[A-Z]\d{2}(?:\.\d{1,3})?", stripped))


def _is_aux_line(line: str) -> bool:
    """Linhas auxiliares que nao devem ser usadas como valor de campo."""
    if not line:
        return True
    normalized = _normalize_label(line)
    if normalized in STOP_CAUSAS:
        return True
    if _looks_like_label(line):
        return True
    if _is_numeric_line(line):
        return True
    if _is_duration_line(line):
        return True
    if _is_cid_only(line):
        return True
    return False


# ---------------------------------------------------------------------------
# Helpers de extracao
# ---------------------------------------------------------------------------

def _split_lines(raw_text: str) -> List[str]:
    return [_normalize_line(l) for l in _normalize_text(raw_text).split("\n")]


def _find_label_index(lines: List[str], labels: List[str], start: int = 0) -> int:
    """Match por igualdade exata normalizada ou startswith do rotulo normalizado."""
    norm_labels = [_normalize_label(l) for l in labels if l]
    for idx in range(start, len(lines)):
        line = lines[idx]
        if not line:
            continue
        norm = _normalize_label(line)
        if not norm:
            continue
        for target in norm_labels:
            if not target:
                continue
            if norm == target or norm.startswith(target):
                return idx
    return -1


def _extract_text_after_label(lines: List[str], labels: List[str], start: int = 0) -> Tuple[str, int]:
    """Extrai texto apos rotulo. Aceita valor na mesma linha apos ':'.
    Senao procura nas proximas linhas, ignorando labels/numericas/duracao/CID puro/auxiliares.
    Retorna (valor, indice_usado).
    """
    idx = _find_label_index(lines, labels, start)
    if idx < 0:
        return "", -1

    line = lines[idx]
    if ":" in line:
        after = line.split(":", 1)[1].strip()
        if after and not _is_aux_line(after):
            return after, idx

    for j in range(idx + 1, min(idx + 6, len(lines))):
        candidate = lines[j]
        if not candidate:
            continue
        if _is_aux_line(candidate):
            continue
        return candidate, j

    return "", idx


def _normalize_date(day: str, month: str, year: str, forced_year: Optional[str] = None) -> str:
    day = day.strip().zfill(2)
    month = month.strip()
    if not month.isdigit():
        month_lower = _remove_accents(month.lower()).strip()
        month = MESES_EXTENSO.get(month_lower, "")
    month = month.zfill(2) if month else ""

    if forced_year:
        year = str(forced_year)
    else:
        year = year.strip()
        if len(year) == 2:
            year = "19" + year if int(year) > 30 else "20" + year
        elif len(year) == 4:
            year = year
        else:
            year = ""

    if day and month and year:
        return f"{day}/{month}/{year}"
    return ""


def _extract_date_after_label(
    lines: List[str],
    labels: List[str],
    start: int = 0,
    forced_year: Optional[str] = None,
) -> str:
    idx = _find_label_index(lines, labels, start)
    if idx < 0:
        return ""

    # Mesma linha
    line = lines[idx]
    if ":" in line:
        after = line.split(":", 1)[1].strip()
        m = DATE_RE.search(after)
        if m:
            return _normalize_date(m.group(1), m.group(2), m.group(3), forced_year)

    # Proximas linhas
    for j in range(idx + 1, min(idx + 6, len(lines))):
        candidate = lines[j]
        if not candidate:
            continue
        if _is_aux_line(candidate):
            continue
        m = DATE_RE.search(candidate)
        if m:
            return _normalize_date(m.group(1), m.group(2), m.group(3), forced_year)

    return ""


def _extract_time_after_label(lines: List[str], labels: List[str], start: int = 0) -> str:
    idx = _find_label_index(lines, labels, start)
    if idx < 0:
        return ""

    line = lines[idx]
    if ":" in line:
        after = line.split(":", 1)[1].strip()
        m = TIME_RE.search(after)
        if m:
            hh = m.group(1).zfill(2)
            mm = m.group(2).zfill(2)
            ss = m.group(3)
            return f"{hh}:{mm}:{ss}" if ss else f"{hh}:{mm}"

    for j in range(idx + 1, min(idx + 6, len(lines))):
        candidate = lines[j]
        if not candidate:
            continue
        if _is_aux_line(candidate):
            continue
        m = TIME_RE.search(candidate)
        if m:
            hh = m.group(1).zfill(2)
            mm = m.group(2).zfill(2)
            ss = m.group(3)
            return f"{hh}:{mm}:{ss}" if ss else f"{hh}:{mm}"

    return ""


def _extract_uf_near(lines: List[str], labels: List[str], start: int = 0) -> str:
    idx = _find_label_index(lines, labels, start)
    if idx < 0:
        return ""

    # Mesma linha
    line = lines[idx]
    if ":" in line:
        after = line.split(":", 1)[1].strip()
        tokens = re.findall(r"\b[A-Z]{2}\b", after)
        for tok in tokens:
            if tok in UF_VALIDAS:
                return tok

    # Proximas linhas
    for j in range(idx + 1, min(idx + 6, len(lines))):
        candidate = lines[j]
        if not candidate:
            continue
        tokens = re.findall(r"\b[A-Z]{2}\b", candidate)
        for tok in tokens:
            if tok in UF_VALIDAS:
                return tok

    return ""


def _extract_cep_near(lines: List[str], labels: List[str], start: int = 0) -> str:
    idx = _find_label_index(lines, labels, start)
    if idx < 0:
        return ""

    line = lines[idx]
    if ":" in line:
        after = line.split(":", 1)[1].strip()
        m = CEP_RE.search(after)
        if m:
            return f"{m.group(1)}-{m.group(2)}"

    for j in range(idx + 1, min(idx + 6, len(lines))):
        candidate = lines[j]
        if not candidate:
            continue
        m = CEP_RE.search(candidate)
        if m:
            return f"{m.group(1)}-{m.group(2)}"

    return ""


def _extract_causas(lines: List[str]) -> List[Dict[str, str]]:
    """Extrai causas da secao causal. Retorna lista de dicts com chaves 'text' e 'cid'.
    Bloqueia explicitamente legendas da secao causal.
    """
    causas: List[Dict[str, str]] = []

    # Localizar inicio da secao de causas
    start_idx = _find_label_index(lines, ["Causa da morte", "Causas da morte", "Causa do obito", "Causas do obito"])
    if start_idx < 0:
        start_idx = _find_label_index(lines, ["Parte I", "Parte i"])
    if start_idx < 0:
        start_idx = 0

    end_idx = len(lines)
    stop_idx = _find_label_index(lines, ["Causa basica", "CID basica", "Codigo da causa basica"], start_idx + 1)
    if stop_idx > start_idx:
        end_idx = stop_idx

    for idx in range(start_idx + 1, end_idx):
        line = lines[idx]
        if not line:
            continue
        norm = _normalize_label(line)
        if norm in STOP_CAUSAS:
            continue
        if _looks_like_label(line):
            continue
        if _is_numeric_line(line):
            continue
        if _is_duration_line(line):
            continue
        if _is_cid_only(line):
            continue

        # CID pode estar no final da linha
        cid_match = CID_RE.search(line)
        cid = cid_match.group(1) if cid_match else ""
        text = line
        if cid:
            text = line.replace(cid, "").strip(" -:;,")
        if not text:
            continue

        causas.append({"text": text, "cid": cid})

    return causas


def _extract_causa_basica(lines: List[str], causas: List[Dict[str, str]], raw_text: str) -> Tuple[str, str]:
    """Retorna (causa_basica_texto, cid_basica).
    CAUSA_BASICA nao pode virar CID puro nem legenda da secao.
    CID_BASICA alinhada com a mesma ultima causa valida; fallback em raw_text so se nenhuma causa valida tiver CID.
    """
    # Tentar rotulo explicito de causa basica
    idx = _find_label_index(lines, ["Causa basica"])
    if idx >= 0:
        line = lines[idx]
        if ":" in line:
            after = line.split(":", 1)[1].strip()
            if after and not _is_cid_only(after) and not _is_aux_line(after):
                cid_match = CID_RE.search(after)
                cid = cid_match.group(1) if cid_match else ""
                text = after.replace(cid, "").strip(" -:;,") if cid else after
                if text:
                    return text, cid

        for j in range(idx + 1, min(idx + 6, len(lines))):
            candidate = lines[j]
            if not candidate:
                continue
            if _is_cid_only(candidate):
                continue
            if _is_aux_line(candidate):
                continue
            cid_match = CID_RE.search(candidate)
            cid = cid_match.group(1) if cid_match else ""
            text = candidate.replace(cid, "").strip(" -:;,") if cid else candidate
            if text:
                return text, cid

    # Fallback: ultima causa valida
    valid_with_cid = [c for c in causas if c.get("text")]
    if valid_with_cid:
        last = valid_with_cid[-1]
        return last.get("text", ""), last.get("cid", "")

    # Fallback em raw_text so se nenhuma causa valida tiver CID
    if raw_text:
        m = CID_RE.search(raw_text)
        if m:
            return "", m.group(1)

    return "", ""


# ---------------------------------------------------------------------------
# parse_obito
# ---------------------------------------------------------------------------

def parse_obito(raw_text: str) -> Dict[str, str]:
    lines = _split_lines(raw_text)
    result: Dict[str, str] = {field: "" for field in HEADER}

    # NOME
    value, _ = _extract_text_after_label(lines, ["Nome", "Nome do falecido", "Nome do falecido(a)"])
    result["NOME"] = value

    # NOME_MAE
    value, _ = _extract_text_after_label(lines, ["Nome da mae", "Mae", "Nome da mae do falecido"])
    result["NOME_MAE"] = value

    # NOME_PAI
    value, _ = _extract_text_after_label(lines, ["Nome do pai", "Pai", "Nome do pai do falecido"])
    result["NOME_PAI"] = value

    # NASCIMENTO (sem forcar ano)
    result["NASCIMENTO"] = _extract_date_after_label(lines, ["Data de nascimento", "Nascimento", "Data de nascimento do falecido"])

    # DATA_OBITO (forcar ano 2026)
    result["DATA_OBITO"] = _extract_date_after_label(
        lines, ["Data do obito", "Data do obito", "Data do falecimento"], forced_year="2026"
    )

    # HORA_OBITO
    result["HORA_OBITO"] = _extract_time_after_label(lines, ["Hora do obito", "Hora do falecimento", "Hora"])

    # CIDADE_OBITO
    value, _ = _extract_text_after_label(lines, ["Municipio de ocorrencia", "Municipio de ocorrencia", "Local de ocorrencia", "Cidade do obito"])
    result["CIDADE_OBITO"] = value

    # UF_OBITO
    result["UF_OBITO"] = _extract_uf_near(lines, ["UF", "Estado", "UF do obito", "Unidade federativa"])

    # CEP
    result["CEP"] = _extract_cep_near(lines, ["CEP", "Codigo postal"])

    # Causas
    causas = _extract_causas(lines)

    causa_fields = ["CAUSA_MORTE", "CAUSA_MORTE_2", "CAUSA_MORTE_3", "CAUSA_MORTE_4", "CAUSA_MORTE_5"]
    cid_fields = ["CID_MORTE", "CID_MORTE_2", "CID_MORTE_3", "CID_MORTE_4", "CID_MORTE_5"]
    codigo_fields = [
        "CODIGO_CAUSA_MORTE",
        "CODIGO_CAUSA_MORTE_2",
        "CODIGO_CAUSA_MORTE_3",
        "CODIGO_CAUSA_MORTE_4",
        "CODIGO_CAUSA_MORTE_5",
    ]

    for i, causa in enumerate(causas[:5]):
        result[causa_fields[i]] = causa.get("text", "")
        result[cid_fields[i]] = causa.get("cid", "")
        result[codigo_fields[i]] = causa.get("cid", "")

    # CAUSA_BASICA / CID_BASICA / CODIGO_CAUSA_BASICA
    causa_basica, cid_basica = _extract_causa_basica(lines, causas, raw_text)
    result["CAUSA_BASICA"] = causa_basica
    result["CID_BASICA"] = cid_basica
    result["CODIGO_CAUSA_BASICA"] = cid_basica

    return result


# ---------------------------------------------------------------------------
# Validacao
# ---------------------------------------------------------------------------

def _compute_score(structured: Dict[str, str]) -> float:
    if not structured:
        return 0.0
    critical = ["NOME", "NOME_MAE", "NASCIMENTO", "DATA_OBITO", "CIDADE_OBITO", "UF_OBITO", "CAUSA_BASICA"]
    filled = sum(1 for f in critical if structured.get(f))
    return round(filled / len(critical), 2)


def validate_structured(structured: Dict[str, str]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not structured.get("NOME"):
        errors.append("NOME ausente.")
    if not structured.get("NOME_MAE"):
        warnings.append("NOME_MAE ausente.")
    if not structured.get("NASCIMENTO"):
        warnings.append("NASCIMENTO ausente.")
    if not structured.get("DATA_OBITO"):
        errors.append("DATA_OBITO ausente.")
    if not structured.get("CIDADE_OBITO"):
        warnings.append("CIDADE_OBITO ausente.")
    if not structured.get("UF_OBITO"):
        warnings.append("UF_OBITO ausente.")
    elif structured.get("UF_OBITO") not in UF_VALIDAS:
        errors.append("UF_OBITO invalida.")
    if not structured.get("CAUSA_BASICA"):
        errors.append("CAUSA_BASICA ausente.")

    score = _compute_score(structured)
    if errors:
        status = "invalid"
    elif warnings:
        status = "warning"
    else:
        status = "valid"

    nome_ok = bool(structured.get("NOME"))
    names_ok = bool(structured.get("NOME") and structured.get("NOME_MAE"))

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "computed": {
            "score": score,
            "status": status,
            "names_ok": names_ok,
            "nome_ok": nome_ok,
        },
        "score": score,
        "status": status,
        "names_ok": names_ok,
        "nome_ok": nome_ok,
    }


# ---------------------------------------------------------------------------
# Pos-processamento
# ---------------------------------------------------------------------------

def _post_process(structured: Dict[str, str]) -> Dict[str, str]:
    # Garantir UF valida
    uf = structured.get("UF_OBITO", "")
    if uf and uf.upper() not in UF_VALIDAS:
        structured["UF_OBITO"] = ""
    elif uf:
        structured["UF_OBITO"] = uf.upper()

    # Garantir que CAUSA_BASICA nao seja CID puro
    if structured.get("CAUSA_BASICA") and _is_cid_only(structured["CAUSA_BASICA"]):
        structured["CAUSA_BASICA"] = ""

    # Garantir que CAUSA_BASICA nao seja legenda da secao
    if structured.get("CAUSA_BASICA") and _normalize_label(structured["CAUSA_BASICA"]) in STOP_CAUSAS:
        structured["CAUSA_BASICA"] = ""

    # Alinhar CID_BASICA com CODIGO_CAUSA_BASICA
    structured["CODIGO_CAUSA_BASICA"] = structured.get("CID_BASICA", "")

    return structured


# ---------------------------------------------------------------------------
# Provider OCR (OpenAI-compatible)
# ---------------------------------------------------------------------------

def _build_ocr_payload(image_data_url: str, mime: str) -> Dict[str, Any]:
    prompt = (
        "Voce e um assistente especializado em extrair dados de declaracoes de obito. "
        "Extraia fielmente todos os campos visiveis no documento, preservando a ordem das linhas. "
        "Retorne apenas o texto extraido, sem comentarios adicionais."
    )
    return {
        "model": OPENAI_MODEL_DEFAULT,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "max_tokens": 2000,
    }


def _call_ocr_provider(image_data_url: str, mime: str) -> str:
    if not OPENAI_API_URL or not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_URL ou OPENAI_API_KEY nao configurados.")

    payload = _build_ocr_payload(image_data_url, mime)
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    content = ""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = ""

    if not content:
        raise RuntimeError("Resposta do provider OCR vazia.")

    for phrase in REFUSAL_PHRASES:
        if phrase.lower() in content.lower():
            raise RuntimeError("Provider recusou o processamento da imagem.")

    return content


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
 return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/ocr")
async def ocr(request: Request, x_auth_token: Optional[str] = Header(None)):
    request_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

    # Autenticacao
    if AUTH_TOKEN and (not x_auth_token or x_auth_token != AUTH_TOKEN):
        return _unauthorized(request_id)

    try:
        body = await request.json()
    except Exception:
        return _bad_request(request_id, "JSON invalido.")

    mime = body.get("mimeType", "")
    file_b64 = body.get("file", "")

    if not mime or not file_b64:
        return _bad_request(request_id, "Campos 'mimeType' e 'file' sao obrigatorios.")

    if mime not in ALLOWED_MIME:
        if mime == "application/pdf":
            return _bad_request(request_id, "PDF nao suportado no /ocr v1. Envie imagem (PNG/JPEG/WebP/GIF).")
        return _bad_request(request_id, f"MIME type nao suportado: {mime}")

    # Validar tamanho
    try:
        decoded = base64.b64decode(file_b64, validate=False)
    except Exception:
        return _bad_request(request_id, "Base64 invalido.")

    if len(decoded) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return _bad_request(request_id, f"Arquivo excede o tamanho maximo de {MAX_FILE_SIZE_MB}MB.")

    prefix = MIME_TO_DATA_URL_PREFIX.get(mime, "")
    if not prefix:
        return _bad_request(request_id, f"MIME type nao suportado: {mime}")

    image_data_url = prefix + file_b64

    # Chamada ao provider OCR
    try:
        raw_text = _call_ocr_provider(image_data_url, mime)
    except Exception as exc:
        return _internal_error(request_id, f"Falha no provider OCR: {exc}")

    # Parse + validacao
    try:
        structured = parse_obito(raw_text)
        structured = _post_process(structured)
        validation = validate_structured(structured)
    except Exception as exc:
        return _internal_error(request_id, f"Erro interno no parse/validacao: {exc}")

    return JSONResponse(
        status_code=200,
        content={
            "requestId": request_id,
            "warnings": validation.get("warnings", []),
            "rawText": raw_text,
            "structured": structured,
            "validation": validation,
            "headerOrder": HEADER,
            "provider": "openai-compatible",
            "confidence": 1.0,
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
