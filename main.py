# FILE: main.py
import base64
import hashlib
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

AUTH_TOKEN = os.environ.get("ENDPOINT_AUTH_TOKEN", "")
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_MODEL_DEFAULT", "gpt-4o")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "10"))
PORT = int(os.environ.get("PORT", "8080"))

ALLOWED_MIME = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",
}

MIME_TO_DATA_URL_PREFIX = {
    "image/jpeg": "data:image/jpeg;base64,",
    "image/png": "data:image/png;base64,",
    "image/webp": "data:image/webp;base64,",
}

REFUSAL_PHRASES = [
    "i'm sorry",
    "i can’t assist with that",
    "i can't assist with that",
    "cannot assist",
    "unable to help",
    "cannot help with that request",
    "i can't help with that",
    "desculpe",
    "não posso ajudar",
    "nao posso ajudar",
]

HEADER = [
    "NOME", "NOME_SOCIAL", "NASCIMENTO", "SEXO", "RACA_COR", "ESTADO_CIVIL",
    "NACIONALIDADE", "NOME_MAE", "NOME_PAI", "PROFISSAO", "LOGRADOURO",
    "NUMERO", "COMPLEMENTO", "BAIRRO", "CIDADE", "UF", "CEP",
    "CIDADE_NASCIMENTO", "UF_NASCIMENTO", "CPF", "RG", "ORGAO_EMISSOR_RG",
    "DATA_OBITO", "HORA_OBITO", "LOCAL_OBITO", "CIDADE_OBITO", "UF_OBITO",
    "CAUSA_MORTE", "CAUSA_MORTE_2", "CAUSA_MORTE_3", "CAUSA_MORTE_4",
    "CAUSA_MORTE_5", "CAUSA_BASICA", "CODIGO_CAUSA_BASICA",
    "CODIGO_CAUSA_MORTE", "CODIGO_CAUSA_MORTE_2", "CODIGO_CAUSA_MORTE_3",
    "CODIGO_CAUSA_MORTE_4", "CODIGO_CAUSA_MORTE_5", "CID_BASICA", "CID_MORTE",
    "CID_MORTE_2", "CID_MORTE_3", "CID_MORTE_4", "CID_MORTE_5", "TIPO_OBITO",
    "ASSISTIDO", "DATA_ATESTADO", "NOMES_OK", "NOME_OK", "GARBAGE_CODES",
    "QTD_GARBAGE", "PROTOCOLO_TEV", "ERROS", "QUALIDADE_SCORE", "HASH_ARQUIVO",
    "HASH_CONTEUDO", "STATUS", "NOME_MES", "DATA_PROCESSAMENTO",
]

UF_VALIDAS = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA",
    "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN",
    "RS", "RO", "RR", "SC", "SP", "SE", "TO",
}

MESES_EXTENSO = {
    "01": "JANEIRO", "02": "FEVEREIRO", "03": "MARCO", "04": "ABRIL",
    "05": "MAIO", "06": "JUNHO", "07": "JULHO", "08": "AGOSTO",
    "09": "SETEMBRO", "10": "OUTUBRO", "11": "NOVEMBRO", "12": "DEZEMBRO",
}

CID_RE = re.compile(r"\b[A-TV-Z]\d{2}(?:\.\d{1,3})?\b", re.IGNORECASE)
CEP_RE = re.compile(r"\d{5}-?\d{3}")
DATE_RE = re.compile(r"\b(\d{2})[/\s.-](\d{2})[/\s.-](\d{4})\b")
TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
DURATION_RE = re.compile(
    r"^(?:[<>]?\d+(?:[dhms]|min|dias?|horas?|minutos?)|\d+\s*(?:d|h|min|dias?|horas?|minutos?))$",
    re.IGNORECASE,
)

AUX_WORDS = {
    "ignorado", "ignorada", "dias", "dia", "horas", "hora",
    "minutos", "minuto", "min", "meses", "mes", "anos", "ano",
    "parte", "parte i", "parte ii", "i", "ii",
}

MONTH_WORDS = {
    "janeiro", "fevereiro", "marco", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
    "jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago",
    "set", "out", "nov", "dez",
}

STOP_CAUSAS = {
    "parte ii",
    "parte 2",
    "outras condicoes significativas",
    "outras condições significativas",
    "nome do medico",
    "nome do médico",
    "crm",
    "obito atestado por medico",
    "óbito atestado por médico",
    "provaveis circunstancias",
    "prováveis circunstâncias",
}

LABEL_MARKERS = [
    "nome", "data", "hora", "municipio", "uf", "cep", "cpf", "rg",
    "causa", "parte", "medico", "crm", "sexo", "raca", "cor",
    "estado civil", "nacionalidade", "profissao", "logradouro",
    "numero", "complemento", "bairro", "cidade", "obito", "idade",
]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="OCR Declaração de Óbito", version="1.0.0")


# ---------------------------------------------------------------------------
# Helpers de erro
# ---------------------------------------------------------------------------

def _error_response(status_code: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message, "requestId": request_id},
    )


def _ensure_request_id(body: Optional[Dict[str, Any]]) -> str:
    if body and isinstance(body.get("requestId"), str) and body["requestId"].strip():
        return body["requestId"].strip()
    return f"req_{int(time.time() * 1000)}"


def _check_auth(authorization: Optional[str]) -> None:
    if not AUTH_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail={
            "code": "UNAUTHORIZED",
            "message": "Token de autenticação ausente.",
            "requestId": "",
        })
    token = authorization[len("Bearer "):].strip()
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail={
            "code": "UNAUTHORIZED",
            "message": "Token de autenticação inválido.",
            "requestId": "",
        })


# ---------------------------------------------------------------------------
# Helpers de texto
# ---------------------------------------------------------------------------

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _normalize_line(line: str) -> str:
    line = (line or "").strip()
    if not line:
        return ""
    line = re.sub(r"^\d{1,3}(?:\.|\)|-)?\s+(?=[A-Za-zÀ-ú])", "", line)
    return _normalize_text(line)


def _looks_like_label(line: str) -> bool:
    if not line:
        return True
    low = line.lower().strip()
    if low.endswith(":"):
        return True
    if len(low) < 3:
        return True
    if re.fullmatch(r"[\d\s/.\-:]+", line):
        return True
    return any(low.startswith(m) for m in LABEL_MARKERS)


def _is_numeric_line(line: str) -> bool:
    if not line:
        return True
    return bool(re.fullmatch(r"[\d\s/.\-:]+", line))


def _is_duration_token(line: str) -> bool:
    if not line:
        return True
    return bool(DURATION_RE.fullmatch(line.strip()))


def _is_cid_only(line: str) -> bool:
    if not line:
        return False
    return bool(re.fullmatch(r"[A-TV-Z]\d{2}(?:\.\d{1,3})?", line.strip(), re.IGNORECASE))


def _is_aux_only(line: str) -> bool:
    if not line:
        return True
    low = line.lower().strip()
    if low in AUX_WORDS:
        return True
    tokens = [t for t in re.split(r"[\s,;\-/]+", low) if t]
    if not tokens:
        return True
    if all(t in AUX_WORDS or t in MONTH_WORDS for t in tokens):
        return True
    return False


def _strip_trailing_cid(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+" + CID_RE.pattern + r"\s*$", "", text, flags=re.IGNORECASE)
    return _normalize_text(text)


def _clean_causa_text(value: str) -> str:
    if not value:
        return ""
    value = _normalize_text(value)
    value = re.sub(r"^(?:d\s+|\(d\)\s+|-\s+|:\s+)", "", value, flags=re.IGNORECASE)
    value = _normalize_text(value)
    if not value:
        return ""
    if _is_cid_only(value):
        return ""
    if _is_duration_token(value):
        return ""
    if _is_aux_only(value):
        return ""
    return value


def _valid_date(value: str) -> bool:
    if not value:
        return False
    m = DATE_RE.search(value)
    if not m:
        return False
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return 1 <= d <= 31 and 1 <= mo <= 12 and 1900 <= y <= 2100


def _valid_time(value: str) -> bool:
    if not value:
        return False
    m = TIME_RE.search(value)
    if not m:
        return False
    h, mi = int(m.group(1)), int(m.group(2))
    return 0 <= h <= 23 and 0 <= mi <= 59


def _extract_cid(text: str) -> str:
    if not text:
        return ""
    m = CID_RE.search(text)
    return m.group(0).upper() if m else ""

# BLOCO DIRETO PARA SUBSTITUIR

def _find_label_index(lines: List[str], label: str) -> int:
    target = label.strip().lower().rstrip(":").strip()
    for i, line in enumerate(lines):
        norm = line.strip().lower().rstrip(":").strip()
        if norm == target or norm.startswith(target + " ") or norm.startswith(target + ":"):
            return i
    return -1


def _extract_text_after_label(lines: List[str], labels: List[str], window: int = 6) -> str:
    idx = -1
    for label in labels:
        idx = _find_label_index(lines, label)
        if idx >= 0:
            break
    if idx < 0:
        return ""

    line = lines[idx]
    if ":" in line:
        value = line.split(":", 1)[1].strip()
        if value and not _looks_like_label(value) and not _is_numeric_line(value):
            cid = _extract_cid(value)
            text = _strip_trailing_cid(value, cid) if cid else value
            text = _clean_causa_text(text) if text else text
            return text

    start = idx + 1
    end = min(len(lines), start + window)
    first_value = ""
    for j in range(start, end):
        candidate = lines[j].strip()
        if not candidate:
            continue
        if _looks_like_label(candidate):
            continue
        if _is_numeric_line(candidate):
            continue
        if _is_duration_token(candidate):
            continue
        if _is_cid_only(candidate):
            continue
        if _is_aux_only(candidate):
            continue
        if not any(ch.isalpha() for ch in candidate):
            continue

        cid = _extract_cid(candidate)
        text = _strip_trailing_cid(candidate, cid) if cid else candidate
        text = _clean_causa_text(text) if text else text
        first_value = text

        for k in range(j + 1, end):
            nxt = lines[k].strip()
            if not nxt:
                break
            if _looks_like_label(nxt):
                break
            if _is_numeric_line(nxt):
                break
            if _is_duration_token(nxt):
                break
            if _is_cid_only(nxt):
                break
            if _is_aux_only(nxt):
                break
            if not any(ch.isalpha() for ch in nxt):
                break
            first_value = (first_value + " " + nxt).strip()
        break

    return first_value


def _extract_date_after_label(lines: List[str], labels: List[str], window: int = 6, forced_year: str = "") -> str:
    idx = -1
    for label in labels:
        idx = _find_label_index(lines, label)
        if idx >= 0:
            break
    if idx < 0:
        return ""

    start = idx
    end = min(len(lines), idx + window + 1)
    for j in range(start, end):
        candidate = lines[j]
        match = DATE_RE.search(candidate)
        if match:
            date_str = match.group(0)
            if forced_year:
                parts = date_str.split("/")
                if len(parts) == 3:
                    parts[2] = forced_year
                    date_str = "/".join(parts)
            return date_str
    return ""


def _extract_time_after_label(lines: List[str], labels: List[str], window: int = 4) -> str:
    idx = -1
    for label in labels:
        idx = _find_label_index(lines, label)
        if idx >= 0:
            break
    if idx < 0:
        return ""

    start = idx
    end = min(len(lines), idx + window + 1)
    for j in range(start, end):
        candidate = lines[j]
        match = TIME_RE.search(candidate)
        if match:
            return match.group(0)
    return ""


def _extract_uf_near(lines: List[str], idx: int) -> str:
    if idx < 0:
        return ""
    end = min(len(lines), idx + 7)
    for j in range(idx, end):
        candidate = lines[j].strip().upper()
        tokens = candidate.replace(",", " ").replace("-", " ").split()
        for token in tokens:
            if token in UF_VALIDAS:
                return token
    return ""


def _extract_causas(lines: List[str]) -> List[Dict[str, str]]:
    causas: List[Dict[str, str]] = []
    start_idx = -1
    for i, line in enumerate(lines):
        norm = line.strip().lower()
        if norm.startswith("causas da morte"):
            start_idx = i
            break
    if start_idx < 0:
        return causas

    for j in range(start_idx + 1, len(lines)):
        raw = lines[j].strip()
        norm = raw.lower()

        if not raw:
            continue

        if any(norm.startswith(stop.lower()) or norm == stop.lower() for stop in STOP_CAUSAS):
            break

        if _is_duration_token(raw):
            continue
        if _is_cid_only(raw):
            continue
        if _is_aux_only(raw):
            continue
        if _looks_like_label(raw):
            continue
        if not any(ch.isalpha() for ch in raw):
            continue

        cid = _extract_cid(raw)
        text = _strip_trailing_cid(raw, cid) if cid else raw
        text = _clean_causa_text(text)
        if not text:
            continue

        causas.append({"text": text, "cid": cid or ""})

    return causas


def parse_obito(raw_text: str) -> Dict[str, Any]:
    result = {k: "" for k in HEADER}
    raw_lines = raw_text.splitlines()
    lines = [_normalize_line(ln) for ln in raw_lines]

    result["NOME"] = _extract_text_after_label(lines, ["Nome", "Nome do falecido", "Nome do paciente"], window=6)
    result["NOME_MAE"] = _extract_text_after_label(lines, ["Nome da mãe", "Nome da mae", "Mãe", "Mae"], window=6)
    result["NOME_PAI"] = _extract_text_after_label(lines, ["Nome do pai", "Pai"], window=6)

    result["NASCIMENTO"] = _extract_date_after_label(lines, ["Data de Nascimento", "Data de nascimento"], window=6)
    result["DATA_OBITO"] = _extract_date_after_label(lines, ["Data do óbito", "Data do obito", "Data de óbito", "Data de obito"], window=6, forced_year="2026")
    result["HORA_OBITO"] = _extract_time_after_label(lines, ["Hora", "Hora do óbito", "Hora do obito"], window=4)

    cidade_idx = -1
    for label in ["Município de ocorrência", "Municipio de ocorrencia", "Município do óbito", "Municipio do obito", "Local de ocorrência", "Local de ocorrencia"]:
        cidade_idx = _find_label_index(lines, label)
        if cidade_idx >= 0:
            break
    result["CIDADE_OBITO"] = _extract_text_after_label(lines, ["Município de ocorrência", "Municipio de ocorrencia", "Município do óbito", "Municipio do obito", "Local de ocorrência", "Local de ocorrencia"], window=6)
    result["UF_OBITO"] = _extract_uf_near(lines, cidade_idx)
    if not result["UF_OBITO"]:
        result["UF_OBITO"] = _extract_text_after_label(lines, ["UF", "Estado"], window=4).upper()

    for line in lines:
        match = CEP_RE.search(line)
        if match:
            result["CEP"] = match.group(0)
            break

    causas = _extract_causas(lines)
    for i, causa in enumerate(causas[:5], start=1):
        result[f"CAUSA_MORTE{i}"] = causa["text"]
        result[f"CID_MORTE{i}"] = causa["cid"]
        result[f"CODIGO_CAUSA_MORTE{i}"] = causa["cid"]

    causa_basica = ""
    cid_basica = ""
    for causa in reversed(causas):
        if causa["text"]:
            causa_basica = causa["text"]
            cid_basica = causa["cid"]
            break

    if not cid_basica:
        cid_basica = _extract_cid(raw_text)

    result["CAUSA_BASICA"] = causa_basica
    result["CID_BASICA"] = cid_basica
    result["CODIGO_CAUSA_BASICA"] = cid_basica

    return result


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

def _compute_score(structured: Dict[str, Any], errors: List[str], warnings: List[str]) -> int:
    score = 100
    score -= len(errors) * 15
    score -= len(warnings) * 5
    if not structured.get("NOME"):
        score -= 10
    if not structured.get("NOME_MAE") and not structured.get("NOME_PAI"):
        score -= 5
    if not structured.get("CID_BASICA"):
        score -= 5
    if not structured.get("CIDADE_OBITO"):
        score -= 3
    if not structured.get("UF_OBITO"):
        score -= 3
    return max(0, min(100, score))


def validate_structured(structured: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if structured.get("NASCIMENTO") and not _valid_date(structured.get("NASCIMENTO", "")):
        errors.append("NASCIMENTO inválido")
    if structured.get("DATA_OBITO") and not _valid_date(structured.get("DATA_OBITO", "")):
        errors.append("DATA_OBITO inválida")
    if structured.get("HORA_OBITO") and not _valid_time(structured.get("HORA_OBITO", "")):
        errors.append("HORA_OBITO inválida")
    if structured.get("UF_OBITO") and structured.get("UF_OBITO", "").upper() not in UF_VALIDAS:
        errors.append("UF_OBITO inválida")
    if structured.get("CEP") and not CEP_RE.fullmatch(structured.get("CEP", "")):
        errors.append("CEP inválido")

    if not structured.get("NOME"):
        errors.append("NOME ausente")
    if not structured.get("DATA_OBITO"):
        errors.append("DATA_OBITO ausente")
    if not structured.get("CAUSA_BASICA"):
        errors.append("CAUSA_BASICA ausente")

    if structured.get("NOME") and structured.get("NOME_PAI") and structured["NOME"] == structured["NOME_PAI"]:
        errors.append("NOME igual a NOME_PAI")
    if structured.get("NOME") and structured.get("NOME_MAE") and structured["NOME"] == structured["NOME_MAE"]:
        errors.append("NOME igual a NOME_MAE")

    if structured.get("CAUSA_BASICA") and re.fullmatch(
        r"[A-TV-Z]\d{2}(?:\.\d{1,3})?", structured["CAUSA_BASICA"], re.IGNORECASE
    ):
        errors.append("CAUSA_BASICA parece CID puro")

    if not structured.get("CID_BASICA"):
        warnings.append("CID_BASICA não localizado")

    nome_ok = bool(structured.get("NOME")) and not _looks_like_label(structured.get("NOME", ""))
    names_ok = all(
        bool(structured.get(c)) and not _looks_like_label(structured.get(c, ""))
        for c in ("NOME", "NOME_MAE", "NOME_PAI")
    )

    score = _compute_score(structured, errors, warnings)

    if errors:
        status = "REVISAR"
    elif score < 90 and warnings:
        status = "REVISAR"
    elif not structured.get("CAUSA_BASICA"):
        status = "REVISAR"
    elif not structured.get("CID_BASICA"):
        status = "REVISAR"
    else:
        status = "OK"

    ok = len(errors) == 0 and status == "OK"

    return {
        "ok": ok,
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
# Pós-processamento
# ---------------------------------------------------------------------------

def _month_name_from_date(date_str: str) -> str:
    if not date_str:
        return ""
    m = DATE_RE.search(date_str)
    if not m:
        return ""
    return MESES_EXTENSO.get(m.group(2), "")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# OCR provider
# ---------------------------------------------------------------------------

def _detect_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(phrase in low for phrase in REFUSAL_PHRASES)


def _call_ocr_provider(file_b64: str, mime_type: str, model: str) -> str:
    prefix = MIME_TO_DATA_URL_PREFIX.get(mime_type)
    if not prefix:
        raise ValueError(f"MIME não suportado para OCR: {mime_type}")

    data_url = f"{prefix}{file_b64}"
    prompt = (
        "Você é um especialista em extrair texto de Declarações de Óbito brasileiras. "
        "Extraia fielmente todo o texto visível do documento, preservando rótulos, "
        "valores, datas, horas, nomes, causas de morte e códigos CID. "
        "Retorne apenas o texto extraído, sem comentários adicionais."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        raise ValueError(f"Provider retornou HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("Resposta do provider sem conteúdo esperado.")

    if not content or not content.strip():
        raise ValueError("Conteúdo vazio retornado pelo provider.")

    if _detect_refusal(content):
        raise ValueError("Recusa explícita detectada pelo provider.")

    return content.strip()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ocr-obito", "version": "1.0.0"}


@app.post("/ocr")
async def ocr(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    try:
        body = await request.json()
    except Exception:
        body = {}

    request_id = _ensure_request_id(body if isinstance(body, dict) else {})

    _check_auth(authorization)

    if not isinstance(body, dict):
        return _error_response(422, "INVALID_BODY", "Corpo da requisição inválido.", request_id)

    file_b64 = body.get("file")
    mime_type = body.get("mimeType")
    model = body.get("model") or OPENAI_MODEL_DEFAULT

    if not file_b64 or not isinstance(file_b64, str):
        return _error_response(422, "MISSING_FILE", "Campo 'file' ausente ou inválido.", request_id)
    if not mime_type or mime_type not in ALLOWED_MIME:
        return _error_response(422, "UNSUPPORTED_MIME", "mimeType não suportado.", request_id)
    if mime_type == "application/pdf":
        return _error_response(422, "PDF_NOT_SUPPORTED_IN_V1", "PDF não suportado nesta versão.", request_id)

    try:
        file_bytes = base64.b64decode(file_b64, validate=True)
    except Exception:
        return _error_response(422, "INVALID_BASE64", "Base64 inválido.", request_id)

    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return _error_response(413, "FILE_TOO_LARGE", f"Arquivo excede {MAX_FILE_SIZE_MB}MB.", request_id)

    try:
        raw_text = _call_ocr_provider(file_b64, mime_type, model)
    except Exception as exc:
        return _error_response(502, "OCR_PROVIDER_ERROR", str(exc), request_id)

    try:
        structured = parse_obito(raw_text)
        validation = validate_structured(structured)

        structured["HASH_ARQUIVO"] = _sha256_bytes(file_bytes)
        structured["HASH_CONTEUDO"] = _sha256_text(raw_text)
        structured["DATA_PROCESSAMENTO"] = _now_iso()
        structured["NOME_MES"] = _month_name_from_date(structured.get("DATA_OBITO", ""))
        structured["ERROS"] = " | ".join(validation["errors"])
        structured["QUALIDADE_SCORE"] = validation["score"]
        structured["STATUS"] = validation["status"]
        structured["NOMES_OK"] = "SIM" if validation["names_ok"] else "NAO"
        structured["NOME_OK"] = "SIM" if validation["nome_ok"] else "NAO"

        for key in HEADER:
            if key not in structured:
                structured[key] = ""

        warnings = list(validation["warnings"])
        if not structured.get("CID_BASICA") and "CID_BASICA não localizado" not in warnings:
            warnings.append("CID_BASICA não localizado")

        return {
            "text": raw_text,
            "confidence": 1.0,
            "provider": "openai-compatible",
            "requestId": request_id,
            "warnings": warnings,
            "rawText": raw_text,
            "structured": structured,
            "validation": validation,
            "headerOrder": HEADER,
        }
    except Exception as exc:
        return _error_response(500, "INTERNAL_ERROR", f"Erro interno: {exc}", request_id)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
