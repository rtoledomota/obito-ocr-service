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


# BLOCO UNICO CORRIGIDO

def _find_label_index(lines: List[str], label: str) -> int:
    # Comparacao exata ou por prefixo do rotulo normalizado.
    # Nao usa busca ampla ("in") para evitar falsos positivos.
    target = _normalize_text(label).strip()
    if not target:
        return -1
    for i, line in enumerate(lines):
        norm = _normalize_text(line).strip()
        if not norm:
            continue
        if norm == target or norm.startswith(target):
            return i
    return -1


def _extract_text_after_label(lines: List[str], labels: List[str], window: int = 6) -> str:
    # Localiza o primeiro rotulo valido e extrai o texto seguinte.
    base_idx = -1
    for label in labels:
        idx = _find_label_index(lines, label)
        if idx != -1:
            base_idx = idx
            break
    if base_idx == -1:
        return ""

    def _aceitavel(cand: str) -> bool:
        cand = cand.strip()
        if not cand:
            return False
        if _looks_like_label(cand):
            return False
        if _is_numeric_line(cand):
            return False
        if _is_duration_token(cand):
            return False
        if _is_cid_only(cand):
            return False
        if _is_aux_only(cand):
            return False
        if not any(c.isalpha() for c in cand):
            return False
        return True

    # Aceita valor na mesma linha apos ':'.
    base_line = lines[base_idx]
    if ":" in base_line:
        after = base_line.split(":", 1)[1].strip()
        if _aceitavel(after):
            text = after
            if base_idx + 1 < len(lines):
                nxt = lines[base_idx + 1].strip()
                if _aceitavel(nxt) and not _looks_like_label(nxt):
                    text = (text + " " + nxt).strip()
            return text

    # Caso contrario, olha nas proximas linhas dentro da janela.
    for j in range(base_idx + 1, min(base_idx + 1 + window, len(lines))):
        cand = lines[j].strip()
        if not _aceitavel(cand):
            continue
        text = cand
        if j + 1 < len(lines):
            nxt = lines[j + 1].strip()
            if _aceitavel(nxt) and not _looks_like_label(nxt):
                text = (text + " " + nxt).strip()
        return text
    return ""


def _extract_date_after_label(lines: List[str], labels: List[str], window: int = 6, forced_year: str = "") -> str:
    # Unica definicao de extracao de data. Retorna sempre DD/MM/AAAA.
    base_idx = -1
    for label in labels:
        idx = _find_label_index(lines, label)
        if idx != -1:
            base_idx = idx
            break
    if base_idx == -1:
        return ""

    for j in range(base_idx, min(base_idx + 1 + window, len(lines))):
        line = lines[j]
        m = DATE_RE.search(line)
        if not m:
            continue
        digits = re.findall(r"\d+", m.group(0))
        if len(digits) < 2:
            continue
        day = digits[0].zfill(2)
        month = digits[1].zfill(2)
        if len(digits) >= 3:
            year = digits[2]
        else:
            year = ""
        if forced_year:
            year = str(forced_year).strip()
        if not year:
            continue
        year = year.zfill(4)
        return f"{day}/{month}/{year}"
    return ""


def _extract_time_after_label(lines: List[str], labels: List[str], window: int = 4) -> str:
    # Retorna HH:MM ou HH:MM:SS conforme o match.
    base_idx = -1
    for label in labels:
        idx = _find_label_index(lines, label)
        if idx != -1:
            base_idx = idx
            break
    if base_idx == -1:
        return ""

    for j in range(base_idx, min(base_idx + 1 + window, len(lines))):
        line = lines[j]
        m = TIME_RE.search(line)
        if not m:
            continue
        matched = m.group(0).strip()
        digits = re.findall(r"\d+", matched)
        if len(digits) < 2:
            continue
        hh = digits[0].zfill(2)
        mm = digits[1].zfill(2)
        if len(digits) >= 3:
            ss = digits[2].zfill(2)
            return f"{hh}:{mm}:{ss}"
        return f"{hh}:{mm}"
    return ""


def _extract_uf_near(lines: List[str], idx: int) -> str:
    # Busca apenas tokens de 2 letras nas proximas 6 linhas.
    # Retorna somente se for sigla UF valida. Nunca retorna frase inteira.
    if idx < 0:
        return ""
    for j in range(idx, min(idx + 7, len(lines))):
        cand = lines[j].strip().upper()
        if not cand:
            continue
        tokens = re.split(r"[\s,;:/\\-]+", cand)
        for tok in tokens:
            tok = tok.strip()
            if len(tok) == 2 and tok in UF_VALIDAS:
                return tok
    return ""


def _extract_causas(lines: List[str]) -> List[Dict[str, str]]:
    # Inicia na linha apos 'causas da morte'.
    start = _find_label_index(lines, "causas da morte")
    if start == -1:
        start = _find_label_index(lines, "causa da morte")
    if start == -1:
        return []

    # Legendas explicitas que devem ser bloqueadas.
    legendas_bloqueadas = [
        "intervalo entre o início e a morte",
        "intervalo entre o inicio e a morte",
        "devido ou como consequência de",
        "devido ou como consequencia de",
        "parte i",
        "parte ii",
        "cid",
        "meses dias horas minutos ignorado",
    ]

    def _e_legenda(low: str) -> bool:
        for leg in legendas_bloqueadas:
            if low == leg or low.startswith(leg) or leg in low:
                return True
        return False

    def _parar(low: str) -> bool:
        for stop in STOP_CAUSAS:
            stop_low = stop.lower().strip()
            if not stop_low:
                continue
            if low == stop_low or low.startswith(stop_low) or stop_low in low:
                return True
        return False

    causas: List[Dict[str, str]] = []
    for j in range(start + 1, len(lines)):
        line = lines[j].strip()
        low = line.lower().strip()

        if _parar(low):
            break
        if _e_legenda(low):
            continue
        if not line:
            continue
        if _is_duration_token(line):
            continue
        if _is_cid_only(line):
            continue
        if _is_aux_only(line):
            continue
        if _looks_like_label(line):
            continue
        if not any(c.isalpha() for c in line):
            continue

        # Extrai o CID da linha original antes de limpar.
        cid = _extract_cid(line)
        text_without_cid = _strip_trailing_cid(line)
        cleaned = _clean_causa_text(text_without_cid)
        if cleaned:
            causas.append({"text": cleaned, "cid": cid})
    return causas


def parse_obito(raw_text: str) -> Dict[str, Any]:
    result = {k: "" for k in HEADER}
    raw_lines = raw_text.splitlines()
    lines = [_normalize_line(ln) for ln in raw_lines]

    # Nomes.
    result["NOME"] = _extract_text_after_label(lines, ["Nome do Falecido"])
    result["NOME_MAE"] = _extract_text_after_label(lines, ["Nome da Mãe", "Nome da Mae"])
    result["NOME_PAI"] = _extract_text_after_label(lines, ["Nome do Pai"])

    # Datas.
    result["NASCIMENTO"] = _extract_date_after_label(lines, ["Data de Nascimento"], window=6)
    result["DATA_OBITO"] = _extract_date_after_label(lines, ["Data do óbito", "Data do obito"], window=6, forced_year="2026")

    # Hora.
    result["HORA_OBITO"] = _extract_time_after_label(lines, ["Hora"], window=4)

    # Cidade do obito.
    result["CIDADE_OBITO"] = _extract_text_after_label(lines, ["Município de ocorrência", "Municipio de ocorrencia"])

    # UF do obito: primeiro a partir do indice do municipio.
    mun_idx = -1
    for label in ["Município de ocorrência", "Municipio de ocorrencia"]:
        mun_idx = _find_label_index(lines, label)
        if mun_idx != -1:
            break
    uf = _extract_uf_near(lines, mun_idx) if mun_idx != -1 else ""
    if not uf:
        # Fallback seguro: token UF apenas na linha seguinte ao rotulo 'UF'.
        uf_idx = _find_label_index(lines, "UF")
        if uf_idx != -1 and uf_idx + 1 < len(lines):
            uf = _extract_uf_near(lines, uf_idx + 1)
    result["UF_OBITO"] = uf

    # CEP: primeiro match em lines.
    for line in lines:
        cep_match = CEP_RE.search(line)
        if cep_match:
            result["CEP"] = cep_match.group(0).strip()
            break

    # Causas.
    causas = _extract_causas(lines)

    causa_fields = ["CAUSA_MORTE", "CAUSA_MORTE_2", "CAUSA_MORTE_3", "CAUSA_MORTE_4", "CAUSA_MORTE_5"]
    cid_fields = ["CID_MORTE", "CID_MORTE_2", "CID_MORTE_3", "CID_MORTE_4", "CID_MORTE_5"]
    cod_fields = ["CODIGO_CAUSA_MORTE", "CODIGO_CAUSA_MORTE_2", "CODIGO_CAUSA_MORTE_3", "CODIGO_CAUSA_MORTE_4", "CODIGO_CAUSA_MORTE_5"]

    for i, causa in enumerate(causas[:5]):
        result[causa_fields[i]] = causa["text"]
        result[cid_fields[i]] = causa["cid"]
        result[cod_fields[i]] = causa["cid"]

    # CAUSA_BASICA = ultima causa valida; CID_BASICA alinhado a ela.
    if causas:
        ultima = causas[-1]
        result["CAUSA_BASICA"] = ultima["text"]
        result["CID_BASICA"] = ultima["cid"]
    else:
        result["CAUSA_BASICA"] = ""
        result["CID_BASICA"] = ""

    # Fallback ao raw_text somente se nenhuma causa valida tiver cid.
    if not any(c["cid"] for c in causas):
        fallback_cid = _extract_cid(raw_text)
        if fallback_cid and not result["CID_BASICA"]:
            result["CID_BASICA"] = fallback_cid

    result["CODIGO_CAUSA_BASICA"] = result["CID_BASICA"]

    return result


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
    _VALID_UFS = {"AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"}
    _CID_RE = re.compile(r'\b[A-TV-Z]\d{2}(?:\.\d{1,3})?\b', re.I)
    _ADMIN_WORDS = ("codigo", "registro", "ufs", "cartorio", "cartório", "declarante", "medico", "médico", "atestante")

    uf = (structured.get("UF_OBITO") or "").strip().upper()
    if uf in _VALID_UFS:
        structured["UF_OBITO"] = uf
    else:
        structured["UF_OBITO"] = ""

    causa = (structured.get("CAUSA_BASICA") or "").strip()
    causa_norm = _normalize_label(causa)
    cid_from_causa = ""
    if causa:
        m = _CID_RE.search(causa)
        if m:
            cid_from_causa = m.group(0).upper()

    is_cid_pure = bool(_CID_RE.fullmatch(causa))
    is_label = causa.endswith(":") or causa_norm in ("causa basica", "causa básica", "causa da morte", "causas da morte")
    is_admin = any(w in causa_norm for w in _ADMIN_WORDS)

    if not causa or is_cid_pure or is_label or is_admin:
        structured["CAUSA_BASICA"] = ""
        structured["CODIGO_CAUSA_BASICA"] = ""
    else:
        structured["CAUSA_BASICA"] = causa
        structured["CODIGO_CAUSA_BASICA"] = cid_from_causa

    nome = (structured.get("NOME") or "").strip()
    data_obito = (structured.get("DATA_OBITO") or "").strip()
    causa_basica = (structured.get("CAUSA_BASICA") or "").strip()
    uf_obito = (structured.get("UF_OBITO") or "").strip()

    erros: List[str] = []
    if not nome:
        erros.append("NOME ausente")
    if not data_obito:
        erros.append("DATA_OBITO ausente")
    if not causa_basica:
        erros.append("CAUSA_BASICA ausente")
    if not uf_obito:
        erros.append("UF_OBITO invalida")

    if nome and data_obito and causa_basica:
        structured["STATUS"] = "OK"
        structured["QUALIDADE_SCORE"] = "100"
        structured["ERROS"] = ""
    else:
        structured["STATUS"] = "REVISAR"
        structured["QUALIDADE_SCORE"] = "85"
        structured["ERROS"] = "; ".join(erros)

    return structured


# ---------------------------------------------------------------------------
# Provider OCR (OpenAI-compatible)
# ---------------------------------------------------------------------------

def _build_ocr_payload(image_data_url: str, mime: str) -> Dict[str, Any]:
    prompt = (
        "Transcreva fielmente todo o texto visivel na imagem. "
        "Preserve a ordem e as quebras de linha originais. "
        "Nao resuma, nao interprete e nao explique. "
        "Retorne apenas o texto transcrito."
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

    # Consolidar content mesmo se vier como lista/blocos
    content: Any = ""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = ""

    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if text:
                    parts.append(str(text))
            elif isinstance(block, str):
                parts.append(block)
        content = "\n".join(parts)
    elif not isinstance(content, str):
        content = str(content) if content is not None else ""

    content = content.strip()

    if not content:
        raise RuntimeError("Resposta vazia do provider OCR.")

    # Recusa apenas quando a resposta inteira comecar claramente com frases comuns
    refusal_starts = (
        "i can't help",
        "i cannot help",
        "i can't assist",
        "i cannot assist",
        "i'm not able to help",
        "i am not able to help",
        "i'm unable to help",
        "i am unable to help",
        "desculpe, mas nao posso",
        "desculpe, mas nao posso ajudar",
        "nao posso ajudar",
        "nao posso processar",
        "nao posso transcrever",
    )
    content_lower = content.lower().lstrip()
    if any(content_lower.startswith(prefix) for prefix in refusal_starts):
        snippet = content[:300]
        raise RuntimeError(f"Provider recusou o processamento da imagem. Resposta: {snippet}")

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
