# main.py consolidado

import base64
import hashlib
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Configuração por ambiente
# ---------------------------------------------------------------------------
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_MODEL_DEFAULT", "gpt-4o-mini")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "10"))
PORT = int(os.environ.get("PORT", "8080"))

# ---------------------------------------------------------------------------
# Constantes e regex
# ---------------------------------------------------------------------------
ALLOWED_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}

MIME_TO_DATA_URL_PREFIX = {
    "image/png": "data:image/png;base64,",
    "image/jpeg": "data:image/jpeg;base64,",
    "image/jpg": "data:image/jpeg;base64,",
    "image/webp": "data:image/webp;base64,",
    "image/gif": "data:image/gif;base64,",
}

REFUSAL_PHRASES = [
    "não posso ajudar",
    "nao posso ajudar",
    "não posso processar",
    "nao posso processar",
    "não posso fornecer",
    "nao posso fornecer",
    "não posso realizar",
    "nao posso realizar",
    "i'm sorry",
    "i am sorry",
    "sorry, i can",
    "i cannot",
    "i can't",
    "unable to",
    "as an ai",
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
    "CID_MORTE",
    "CODIGO_CAUSA_MORTE",
    "CAUSA_MORTE2",
    "CID_MORTE2",
    "CODIGO_CAUSA_MORTE2",
    "CAUSA_MORTE3",
    "CID_MORTE3",
    "CODIGO_CAUSA_MORTE3",
    "CAUSA_MORTE4",
    "CID_MORTE4",
    "CODIGO_CAUSA_MORTE4",
    "CAUSA_MORTE5",
    "CID_MORTE5",
    "CODIGO_CAUSA_MORTE5",
    "CAUSA_BASICA",
    "CID_BASICA",
    "CODIGO_CAUSA_BASICA",
    "STATUS",
    "QUALIDADE_SCORE",
    "ERROS",
]

UF_VALIDAS = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA",
    "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN",
    "RS", "RO", "RR", "SC", "SP", "SE", "TO",
}

MESES_EXTENSO = {
    "janeiro": "01", "fevereiro": "02", "marco": "03", "março": "03",
    "abril": "04", "maio": "05", "junho": "06", "julho": "07",
    "agosto": "08", "setembro": "09", "outubro": "10", "novembro": "11",
    "dezembro": "12", "jan": "01", "fev": "02", "mar": "03", "abr": "04",
    "mai": "05", "jun": "06", "jul": "07", "ago": "08", "set": "09",
    "out": "10", "nov": "11", "dez": "12",
}

STOP_CAUSAS = [
    "parte ii",
    "código", "codigo", "registro", "ufs", "cartório", "cartorio",
    "declarante", "médico", "medico", "atestante",
]

BLOCKED_LEGENDS = [
    "outras condições significativas que contribuíram para a morte",
    "outras condições significativas que contribuiram para a morte",
    "porém, na cadeia acima",
    "porem, na cadeia acima",
]

# Aceita separadores: espaço, '/', '-', '.', '|'; mês por extenso ou numérico.
DATE_RE = re.compile(
    r'\b(\d{1,2})\s*[ /\-.|]\s*'
    r'([A-Za-zÀ-ÿ]+|\d{1,2})\s*[ /\-.|]\s*'
    r'(\d{2,4})\b',
    re.IGNORECASE,
)

TIME_RE = re.compile(r'\b(\d{1,2})[:hH](\d{2})(?:[:mM](\d{2}))?\b')
CID_RE = re.compile(
    r'\b[A-TV-Z]\d{2}(?:\.\d)?(?:-[A-TV-Z]\d{2}(?:\.\d)?)?\b',
    re.IGNORECASE,
)
CEP_RE = re.compile(r'\b\d{5}-?\d{3}\b')

# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="OCR Declaração de Óbito", version="1.0.0")

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
            "message": "Token de autenticação inválido ou ausente.",
        },
    )

# ---------------------------------------------------------------------------
# Helpers de texto
# ---------------------------------------------------------------------------
def _remove_accents(text: str) -> str:
    if not text:
        return ""
    trans = str.maketrans(
        "áàâãäéèêëíìîïóòôõöúùûüçÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ",
        "aaaaaeeeeiiiiooooouuuucAAAAAEEEEIIIIOOOOOUUUUC",
    )
    return text.translate(trans)


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _normalize_line(line: str) -> str:
    if not line:
        return ""
    return re.sub(r'\s+', ' ', line).strip()


def _normalize_label(label: str) -> str:
    if not label:
        return ""
    label = _remove_accents(label)
    label = label.lower()
    label = re.sub(r'[^a-z0-9 ]', ' ', label)
    label = re.sub(r'\s+', ' ', label).strip()
    return label


def _looks_like_label(text: str) -> bool:
    """Heurística conservadora para identificar rótulos de formulário."""
    if not text:
        return False
    low = text.lower().strip()
    if len(low) > 80:
        return False
    if CID_RE.fullmatch(text.strip()):
        return False
    if re.fullmatch(r'[\d\s/\-.|:]+', text):
        return False
    if re.fullmatch(r'\d{1,2}\s*[/:hH]\s*\d{2}(?:\s*[:mM]\s*\d{2})?', text):
        return False
    if low.endswith(":") or low.endswith("."):
        return True
    if re.match(r'^\s*(?:\[\d+\]|\(\d+\)|\d+[\)\.]|\d+\s+)', text):
        return True
    return False

def _is_numeric_line(text: str) -> bool:
    if not text:
        return False
    return bool(re.fullmatch(r'[\d\s/\-.|]+', text))


def _is_duration(text: str) -> bool:
    if not text:
        return False
    return bool(re.fullmatch(r'\d+\s*(?:ano|anos|mes|meses|dia|dias|hora|horas|min|minuto|minutos)\b.*', text, re.IGNORECASE))


def _is_cid_only(text: str) -> bool:
    if not text:
        return False
    return bool(CID_RE.fullmatch(text.strip()))

# ---------------------------------------------------------------------------
# Helpers de extração
# ---------------------------------------------------------------------------
def _find_label_index(lines: List[str], labels: List[str], start: int = 0) -> int:
    """Localiza índice da linha que contém um rótulo, removendo prefixos numéricos típicos."""
    for i in range(start, len(lines)):
        raw = lines[i] or ""
        cleaned = re.sub(r'^\s*(?:\[\d+\]|\(\d+\)|\d+[\)\.]|\d+\s+)', '', raw)
        norm_line = _normalize_label(cleaned)
        if not norm_line:
            continue
        for label in labels:
            norm_label = _normalize_label(label)
            if not norm_label:
                continue
            if norm_line == norm_label or norm_line.startswith(norm_label):
                return i
    return -1


def _extract_text_after_label(
    lines: List[str],
    labels: List[str],
    start: int = 0,
    max_lines: int = 6,
) -> Tuple[int, str]:
    """Extrai valor após rótulo. Aceita 'label: valor' na mesma linha ou valor nas próximas linhas."""
    idx = _find_label_index(lines, labels, start)
    if idx < 0:
        return -1, ""

    line = _normalize_line(lines[idx])
    # Tenta valor após ':' na mesma linha
    if ":" in line:
        after = line.split(":", 1)[1].strip()
        if after and not _looks_like_label(after) and not _is_numeric_line(after) and not _is_cid_only(after):
            return idx, after

    # Procura nas próximas linhas
    for j in range(idx + 1, min(idx + 1 + max_lines, len(lines))):
        candidate = _normalize_line(lines[j])
        if not candidate:
            continue
        if _looks_like_label(candidate):
            continue
        if _is_numeric_line(candidate):
            continue
        if _is_duration(candidate):
            continue
        if _is_cid_only(candidate):
            continue
        return idx, candidate
    return idx, ""


def _normalize_date(day: str, month: str, year: str, forced_year: Optional[str] = None) -> str:
    """Normaliza data para DD/MM/AAAA. forced_year sobrescreve o ano quando informado."""
    try:
        d = int(day)
        if d < 1 or d > 31:
            return ""
        m_raw = month.strip().lower()
        if m_raw.isdigit():
            m = int(m_raw)
        else:
            m_clean = _remove_accents(m_raw).strip()
            m = int(MESES_EXTENSO.get(m_clean, 0))
        if m < 1 or m > 12:
            return ""

        if forced_year:
            y = int(forced_year)
        else:
            y = int(year)
            if y < 100:
                y += 2000 if y <= 50 else 1900
        if y < 1900 or y > 2100:
            return ""

        return f"{d:02d}/{m:02d}/{y:04d}"
    except Exception:
        return ""


def _extract_date_after_label(
    lines: List[str],
    labels: List[str],
    start: int = 0,
    forced_year: Optional[str] = None,
) -> str:
    """Procura data na mesma linha do rótulo, na linha seguinte e nas próximas linhas."""
    idx = _find_label_index(lines, labels, start)
    if idx < 0:
        return ""

    search_lines = [lines[i] for i in range(idx, min(idx + 4, len(lines)))]
    for line in search_lines:
        if not line:
            continue
        match = DATE_RE.search(line)
        if match:
            return _normalize_date(match.group(1), match.group(2), match.group(3), forced_year)
    return ""


def _extract_time_after_label(
    lines: List[str],
    labels: List[str],
    start: int = 0,
) -> str:
    """Procura hora na mesma linha do rótulo, na linha seguinte e nas próximas linhas."""
    idx = _find_label_index(lines, labels, start)
    if idx < 0:
        return ""

    search_lines = [lines[i] for i in range(idx, min(idx + 4, len(lines)))]
    for line in search_lines:
        if not line:
            continue
        match = TIME_RE.search(line)
        if match:
            hh = int(match.group(1))
            mm = int(match.group(2))
            ss = match.group(3)
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                if ss:
                    return f"{hh:02d}:{mm:02d}:{int(ss):02d}"
                return f"{hh:02d}:{mm:02d}"
    return ""


def _extract_uf_near(lines: List[str], labels: List[str], start: int = 0) -> str:
    """Retorna somente token de 2 letras válido em UF_VALIDAS próximo ao rótulo."""
    idx = _find_label_index(lines, labels, start)
    if idx < 0:
        return ""

    search_lines = [lines[i] for i in range(idx, min(idx + 5, len(lines)))]
    for line in search_lines:
        if not line:
            continue
        tokens = re.findall(r'\b[A-Za-z]{2}\b', line)
        for tok in tokens:
            up = tok.upper()
            if up in UF_VALIDAS:
                return up
    return ""


def _extract_cep_near(lines: List[str], labels: List[str], start: int = 0) -> str:
    idx = _find_label_index(lines, labels, start)
    if idx < 0:
        return ""
    search_lines = [lines[i] for i in range(idx, min(idx + 5, len(lines)))]
    for line in search_lines:
        if not line:
            continue
        match = CEP_RE.search(line)
        if match:
            cep = match.group(0)
            if "-" not in cep:
                cep = f"{cep[:5]}-{cep[5:]}"
            return cep
    return ""


def _extract_causas(lines: List[str]) -> List[Dict[str, str]]:
    """Extrai causas da morte. Inicia apenas em marcadores reais e para em Parte II/áreas administrativas."""
    start_markers = ["causas da morte", "causa da morte", "parte i"]
    stop_markers = ["parte ii"]
    admin_markers = STOP_CAUSAS

    start_idx = -1
    for i, line in enumerate(lines):
        low = (line or "").lower()
        if any(m in low for m in start_markers):
            start_idx = i
            break
    if start_idx < 0:
        return []

    results: List[Dict[str, str]] = []
    for line in lines[start_idx + 1:]:
        text = (line or "").strip()
        if not text:
            continue
        low = text.lower()

        if any(m in low for m in stop_markers):
            break
        if any(m in low for m in admin_markers):
            break
        if any(b in low for b in BLOCKED_LEGENDS):
            continue

        cid_match = CID_RE.search(text)
        cid = cid_match.group(0) if cid_match else ""

        clean = CID_RE.sub('', text)
        clean = re.sub(r'\s+', ' ', clean).strip(" .:;-")

        if not clean:
            continue
        if clean.isdigit():
            continue
        if re.fullmatch(r'[\d\s\.\-]+', clean):
            continue
        if re.fullmatch(r'[\dhms]+', clean, re.IGNORECASE) and len(clean) <= 8:
            continue
        if len(clean) <= 3:
            continue
        if _looks_like_label(clean):
            continue

        results.append({"text": clean, "cid": cid})

    return results
def _debug_slice(raw_text: str, start_markers: list[str], end_markers: list[str], limit: int = 1200) -> str:
    text = raw_text or ""
    lower = text.lower()

    start = -1
    for marker in start_markers:
        idx = lower.find(marker.lower())
        if idx != -1:
            start = idx
            break

    if start == -1:
        return ""

    end = len(text)
    for marker in end_markers:
        idx = lower.find(marker.lower(), start + 1)
        if idx != -1:
            end = min(end, idx)

    return text[start:end][:limit].strip()
# ---------------------------------------------------------------------------
# parse_obito
# ---------------------------------------------------------------------------
def parse_obito(raw_text: str) -> dict:
    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]

    causas_trecho = _debug_slice(
        raw_text,
        start_markers=[
            "causas da morte",
            "parte i",
            "causa da morte",
            "causas"
        ],
        end_markers=[
            "parte ii",
            "ii ",
            "atestante",
            "médico",
            "medico",
            "cartório",
            "cartorio"
        ]
    )
    print("DEBUG_CAUSAS_TRECHO:", causas_trecho)

    municipio_trecho = _debug_slice(
        raw_text,
        start_markers=[
            "município de ocorrência",
            "municipio de ocorrencia",
            "local de ocorrência",
            "local de ocorrencia"
        ],
        end_markers=[
            "sepultamento",
            "cemitério",
            "cemiterio",
            "declarante",
            "atestante",
            "causas da morte"
        ]
    )
    print("DEBUG_MUNICIPIO_TRECHO:", municipio_trecho)

    nome = _extract_text_after_label(
        lines,
        [
            "nome do falecido",
            "nome do falecida",
            "nome do",
            "falecido"
        ]
    )

    nome_mae = _extract_text_after_label(
        lines,
        [
            "nome da mae",
            "nome da mãe",
            "mae",
            "mãe"
        ]
    )

    nome_pai = _extract_text_after_label(
        lines,
        [
            "nome do pai",
            "pai"
        ]
    )

    nascimento = _extract_date_after_label(
        lines,
        [
            "data de nascimento",
            "nascimento",
            "nasc."
        ],
        forced_year=None
    )

    data_obito = _extract_date_after_label(
        lines,
        [
            "data do obito",
            "data do óbito",
            "obito",
            "óbito"
        ],
        forced_year="2026"
    )

    hora_obito = _extract_time_after_label(
        lines,
        [
            "hora",
            "hora do obito",
            "hora do óbito"
        ]
    )

    cidade_obito, uf_obito = _extract_city_state(lines, raw_text)

    causa_basica, cid_basica = _extract_causas(lines, raw_text)

    structured = {
        "NOME": nome or "",
        "NOME_MAE": nome_mae or "",
        "NOME_PAI": nome_pai or "",
        "NASCIMENTO": nascimento or "",
        "DATA_OBITO": data_obito or "",
        "HORA_OBITO": hora_obito or "",
        "CIDADE_OBITO": cidade_obito or "",
        "UF_OBITO": uf_obito or "",
        "CAUSA_BASICA": causa_basica or "",
        "CID_BASICA": cid_basica or "",
    }

    structured = _post_process(structured)
    return structured
    

# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------
def _compute_score(structured: Dict[str, str]) -> int:
    """Score simples e consistente baseado em campos principais."""
    pesos = {
        "NOME": 25,
        "DATA_OBITO": 20,
        "CAUSA_BASICA": 20,
        "UF_OBITO": 10,
        "NASCIMENTO": 10,
        "NOME_MAE": 5,
        "CIDADE_OBITO": 5,
        "CID_BASICA": 5,
    }
    score = 0
    for campo, peso in pesos.items():
        valor = (structured.get(campo) or "").strip()
        if valor:
            score += peso
    return score


def validate_structured(structured: Dict[str, str]) -> Dict[str, Any]:
    """Valida campos estruturados e retorna objeto de validação completo."""
    errors: List[str] = []
    warnings: List[str] = []

    nome = (structured.get("NOME") or "").strip()
    data_obito = (structured.get("DATA_OBITO") or "").strip()
    causa_basica = (structured.get("CAUSA_BASICA") or "").strip()
    uf = (structured.get("UF_OBITO") or "").strip()

    if not nome:
        errors.append("NOME ausente")
    if not data_obito:
        errors.append("DATA_OBITO ausente")
    if not causa_basica:
        errors.append("CAUSA_BASICA ausente")
    if uf and uf not in UF_VALIDAS:
        errors.append("UF_OBITO inválida")
    if not uf:
        warnings.append("UF_OBITO ausente")

    if not structured.get("NASCIMENTO"):
        warnings.append("NASCIMENTO ausente")
    if not structured.get("NOME_MAE"):
        warnings.append("NOME_MAE ausente")
    if not structured.get("CID_BASICA"):
        warnings.append("CID_BASICA ausente")

    score = _compute_score(structured)
    ok = len(errors) == 0
    status = "OK" if ok else "REVISAR"

    nome_ok = bool(nome)
    names_ok = bool(nome and (structured.get("NOME_MAE") or structured.get("NOME_PAI")))

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
ADMIN_WORDS = ["codigo", "código", "registro", "ufs", "cartorio", "cartório", "declarante", "medico", "médico", "atestante"]


def _is_administrative(text: str) -> bool:
    if not text:
        return False
    low = _remove_accents(text).lower()
    return any(w in low for w in ADMIN_WORDS)


def _post_process(structured: Dict[str, str]) -> Dict[str, str]:
    """Normaliza campos finais e reintroduz STATUS, QUALIDADE_SCORE e ERROS."""
    # Normaliza UF válida
    uf = (structured.get("UF_OBITO") or "").upper().strip()
    structured["UF_OBITO"] = uf if uf in UF_VALIDAS else ""

    # Limpa CAUSA_BASICA se for CID puro, label ou texto administrativo
    causa_basica = (structured.get("CAUSA_BASICA") or "").strip()
    if _is_cid_only(causa_basica) or _looks_like_label(causa_basica) or _is_administrative(causa_basica):
        structured["CAUSA_BASICA"] = ""

    # Alinha CODIGO_CAUSA_BASICA = CID_BASICA
    cid_basica = (structured.get("CID_BASICA") or "").strip()
    structured["CODIGO_CAUSA_BASICA"] = cid_basica

    # Regras de STATUS / QUALIDADE_SCORE / ERROS
    nome = (structured.get("NOME") or "").strip()
    data_obito = (structured.get("DATA_OBITO") or "").strip()
    causa_basica_final = (structured.get("CAUSA_BASICA") or "").strip()
    uf_final = (structured.get("UF_OBITO") or "").strip()

    faltantes: List[str] = []
    if not nome:
        faltantes.append("NOME")
    if not data_obito:
        faltantes.append("DATA_OBITO")
    if not causa_basica_final:
        faltantes.append("CAUSA_BASICA")
    if not uf_final:
        faltantes.append("UF_OBITO inválida")

    if nome and data_obito and causa_basica_final:
        structured["STATUS"] = "OK"
        structured["QUALIDADE_SCORE"] = "100"
        structured["ERROS"] = ""
    else:
        structured["STATUS"] = "REVISAR"
        structured["QUALIDADE_SCORE"] = "85"
        # ERROS: concatena faltantes relevantes
        erros_lista: List[str] = []
        if not nome:
            erros_lista.append("NOME")
        if not data_obito:
            erros_lista.append("DATA_OBITO")
        if not causa_basica_final:
            erros_lista.append("CAUSA_BASICA")
        if not uf_final:
            erros_lista.append("UF_OBITO inválida")
        structured["ERROS"] = "; ".join(erros_lista)

    return structured

# ---------------------------------------------------------------------------
# Provider OCR (OpenAI-compatible)
# ---------------------------------------------------------------------------
OCR_PROMPT = (
    "Você é um motor de OCR puro. Transcreva fielmente todo o texto visível na imagem, "
    "preservando a ordem, os números de campos, rótulos e valores. "
    "Não interprete, não resuma e não omita nada. "
    "Retorne apenas o texto transcrito, em português, sem comentários."
)


def _build_ocr_payload(image_data_url: str, mime: str) -> Dict[str, Any]:
    """Constrói payload para endpoint OpenAI-compatible com prompt neutro de OCR puro."""
    return {
        "model": OPENAI_MODEL_DEFAULT,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": OCR_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
    }


def _consolidate_content(content: Any) -> str:
    """Consolida content (string ou lista/blocos) em uma única string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                elif isinstance(item.get("type"), str) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
    return str(content)


def _is_refusal(text: str) -> bool:
    if not text:
        return False
    low = text.strip().lower()
    for phrase in REFUSAL_PHRASES:
        if low.startswith(phrase):
            return True
    return False


def _call_ocr_provider(image_data_url: str, mime: str) -> str:
    """Chama provider OpenAI-compatible e retorna o texto OCR consolidado."""
    if not OPENAI_API_URL:
        raise RuntimeError("OPENAI_API_URL não configurado.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY não configurado.")

    payload = _build_ocr_payload(image_data_url, mime)
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(OPENAI_API_URL, json=payload, headers=headers, timeout=60)
    except requests.RequestException as exc:
        raise RuntimeError(f"Erro de comunicação com provider OCR: {exc}")

    if resp.status_code >= 400:
        snippet = (resp.text or "")[:300]
        raise RuntimeError(f"Provider OCR retornou HTTP {resp.status_code}: {snippet}")

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError("Provider OCR retornou resposta não-JSON.")

    try:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices, list):
            summary = str(data)[:300]
            raise RuntimeError(f"Estrutura inesperada do provider OCR: {summary}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        text = _consolidate_content(content)
    except RuntimeError:
        raise
    except Exception as exc:
        summary = str(data)[:300]
        raise RuntimeError(f"Estrutura inesperada do provider OCR: {summary}")

    if not text or not text.strip():
        raise RuntimeError("Resposta vazia do provider OCR.")

    if _is_refusal(text):
        snippet = text.strip()[:200]
        raise RuntimeError(f"Provider OCR recusou o processamento: {snippet}")

    return text.strip()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.post("/ocr")
async def ocr(request: Request, x_auth_token: Optional[str] = Header(None)) -> JSONResponse:
    request_id = hashlib.sha1(f"{time.time()}{id(request)}".encode()).hexdigest()[:16]

    # Autenticação
    if AUTH_TOKEN:
        if not x_auth_token or x_auth_token != AUTH_TOKEN:
            return _unauthorized(request_id)

    # Leitura do JSON
    try:
        body = await request.json()
    except Exception:
        return _bad_request(request_id, "JSON inválido.")

    if not isinstance(body, dict):
        return _bad_request(request_id, "Payload deve ser um objeto JSON.")

    file_b64 = body.get("file") or ""
    mime = (body.get("mimeType") or "").lower().strip()

    if not file_b64:
        return _bad_request(request_id, "Campo 'file' (base64) é obrigatório.")
    if not mime:
        return _bad_request(request_id, "Campo 'mimeType' é obrigatório.")

    # PDF não suportado no /ocr v1
    if mime in {"application/pdf", "pdf"} or mime.endswith("pdf"):
        return _bad_request(request_id, "PDF não é suportado no /ocr v1. Envie imagem (PNG/JPEG/WebP/GIF).")

    if mime not in ALLOWED_MIME:
        return _bad_request(request_id, f"mimeType não permitido: {mime}")

    # Validação base64
    try:
        file_bytes = base64.b64decode(file_b64, validate=True)
    except Exception:
        return _bad_request(request_id, "base64 inválido no campo 'file'.")

    # Validação de tamanho
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return _bad_request(request_id, f"Arquivo excede o tamanho máximo de {MAX_FILE_SIZE_MB}MB.")

    # Monta data URL
    prefix = MIME_TO_DATA_URL_PREFIX.get(mime, f"data:{mime};base64,")
    data_url = f"{prefix}{file_b64.strip()}"

    # Chamada ao provider OCR
    try:
        raw_text = _call_ocr_provider(data_url, mime)
    except RuntimeError as exc:
        return _internal_error(request_id, str(exc))
    except Exception as exc:
        return _internal_error(request_id, f"Erro inesperado no provider OCR: {exc}")

    # Parse + pós-processamento + validação
    try:
        structured = parse_obito(raw_text)
        structured = _post_process(structured)
        validation = validate_structured(structured)
    except Exception as exc:
        return _internal_error(request_id, f"Erro ao processar OCR: {exc}")

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
# Inicialização
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
