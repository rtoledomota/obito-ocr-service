import os
import re
import json
import logging
import base64
import io
from datetime import datetime, timezone
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────
logger = logging.getLogger("ocr-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_MODEL_DEFAULT", "gpt-4o")

# ── FastAPI App ─────────────────────────────────────────────────────────
app = FastAPI(title="OCR Death Certificate Parser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ─────────────────────────────────────────────────────
class OCRRequest(BaseModel):
    image: str
    filename: Optional[str] = "image.jpg"

class OCRResponse(BaseModel):
    requestId: str
    status: str
    provider: str
    confidence: float
    rawText: str
    text: Optional[str] = ""
    structured: dict
    validation: dict
    warnings: list[str]
    processingTimeMs: int

# ── Regex Patterns ─────────────────────────────────────────────────────
DATE_RE = re.compile(
    r'(\d{1,2})\s*[/|\-.\s]\s*(\d{1,2})\s*[/|\-.\s]\s*(\d{2,4})'
)
TIME_RE = re.compile(r'(\d{1,2}):(\d{2})')
UF_VALIDAS = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO",
    "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
    "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"
}

# ── Helpers ────────────────────────────────────────────────────────────

def _looks_like_label(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if not t:
        return False
    labels = [
        "data do", "nome do", "nome da", "causa", "parte",
        "hora", "cartão", "cartao", "naturalidade",
        "município", "municipio", "sepultamento",
        "atestante", "médico", "medico", "declarante",
        "tipo de", "fetal", "não fetal", "nao fetal",
        "óbito", "obito", "nascimento", "pai", "mãe", "mae",
        "cartorio", "registro", "uf"
    ]
    return any(kw in t for kw in labels)

def _normalize_date(day: str, month: str, year: str, forced_year: str = None) -> str:
    try:
        d = int(day)
        m = int(month)
    except ValueError:
        return ""
    if d < 1 or d > 31 or m < 1 or m > 12:
        return ""
    y = forced_year if forced_year else year
    if len(y) == 2:
        y = "20" + y if int(y) < 50 else "19" + y
    return f"{d:02d}/{m:02d}/{y}"

def _normalize_uf(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip().upper()
    if raw in UF_VALIDAS:
        return raw
    for uf in UF_VALIDAS:
        if uf in raw:
            return uf
    return ""

def _find_label_index(lines: list[str], labels: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if not line:
            continue
        clean = line.strip()
        clean = re.sub(r'^\[\d+\]\s*', '', clean)
        clean = re.sub(r'^\(\d+\)\s*', '', clean)
        clean = re.sub(r'^\d+\)\s*', '', clean)
        clean = re.sub(r'^\d+\.\s*', '', clean)
        clean = re.sub(r'^\d+\s+', '', clean)
        clean_lower = clean.lower()
        for label in labels:
            if clean_lower.startswith(label.lower()):
                return i
    return None

def _extract_text_after_label(lines: list[str], labels: list[str], max_lines: int = 3) -> str:
    idx = _find_label_index(lines, labels)
    if idx is None:
        return ""
    for offset in range(1, max_lines + 1):
        if idx + offset >= len(lines):
            break
        candidate = lines[idx + offset].strip()
        if not candidate:
            continue
        if _looks_like_label(candidate):
            continue
        if len(candidate) < 2:
            continue
        return candidate
    return ""

def _extract_date_after_label(lines: list[str], labels: list[str], forced_year: str = None) -> str:
    idx = _find_label_index(lines, labels)
    if idx is None:
        return ""
    search_lines = []
    for offset in range(0, 6):
        if idx + offset < len(lines):
            search_lines.append(lines[idx + offset])
    text_block = " ".join(search_lines)
    match = DATE_RE.search(text_block)
    if match:
        return _normalize_date(match.group(1), match.group(2), match.group(3), forced_year)
    return ""

def _extract_time_after_label(lines: list[str], labels: list[str]) -> str:
    idx = _find_label_index(lines, labels)
    if idx is None:
        return ""
    search_lines = []
    for offset in range(0, 4):
        if idx + offset < len(lines):
            search_lines.append(lines[idx + offset])
    text_block = " ".join(search_lines)
    match = TIME_RE.search(text_block)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    return ""

def _extract_city_state(lines: list[str], raw_text: str) -> tuple[str, str]:
    text = raw_text or ""
    lower = text.lower()
    cidade = ""
    uf = ""

    # Tenta "município de ocorrência"
    for marker in ["município de ocorrência", "municipio de ocorrencia",
                    "local de ocorrência", "local de ocorrencia"]:
        idx = lower.find(marker)
        if idx == -1:
            continue
        block = text[idx:idx+300].splitlines()
        for line in block[1:4]:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            if any(kw in line.lower() for kw in
                   ["uf", "município", "municipio", "país", "se estrangeiro"]):
                continue
            parts = line.split()
            if len(parts) >= 2 and len(parts[-1]) == 2 and parts[-1].isalpha():
                cidade = " ".join(parts[:-1])
                uf = parts[-1].upper()
            elif not cidade:
                cidade = line
            break
        if cidade:
            break

    # Fallback: naturalidade
    if not cidade:
        idx = lower.find("naturalidade")
        if idx != -1:
            block = text[idx:idx+300].splitlines()
            for line in block[1:4]:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                if any(kw in line.lower() for kw in
                       ["uf", "município", "municipio", "país", "se estrangeiro"]):
                    continue
                parts = line.split()
                if len(parts) >= 2 and len(parts[-1]) == 2 and parts[-1].isalpha():
                    cidade = " ".join(parts[:-1])
                    uf = parts[-1].upper()
                elif not cidade:
                    cidade = line
                break

    # Fallback: UF por regex
    if not uf:
        for match in re.finditer(r'\b([A-Z]{2})\b', text):
            if match.group(1) in UF_VALIDAS:
                uf = match.group(1)
                break

    return cidade.strip(), uf

def _extract_causas(lines: list[str], raw_text: str) -> tuple[str, str]:
    text = raw_text or ""
    lower = text.lower()

    start_idx = -1
    for marker in ["causas da morte", "parte i", "causa da morte"]:
        idx = lower.find(marker)
        if idx != -1:
            start_idx = idx
            break

    if start_idx == -1:
        return "", ""

    end_idx = len(text)
    for marker in ["parte ii", "atestante", "médico", "medico",
                    "cartório", "cartorio", "declarante"]:
        idx = lower.find(marker, start_idx + 1)
        if idx != -1 and idx < end_idx:
            end_idx = idx

    causas_block = text[start_idx:end_idx].strip()
    block_lines = [l.strip() for l in causas_block.splitlines() if l.strip()]

    causa_basica = ""
    cid_basica = ""
    last_cid = ""

    for line in block_lines:
        ll = line.lower()
        if any(kw in ll for kw in ["parte", "condições significativas", "contribuiram",
                                     "não entraram", "cadeia acima", "código",
                                     "registro", "ufs"]):
            continue
        if len(line) < 4:
            continue
        if _looks_like_label(line) and len(line) < 80:
            continue

        cid_match = re.search(r'\b([A-Z]\d{2,3})\b', line)
        if cid_match:
            last_cid = cid_match.group(1)

        if not causa_basica and len(line) > 5:
            causa_basica = line
        elif len(line) > 5 and len(causa_basica) < 150:
            causa_basica += " | " + line

    if len(causa_basica) > 300:
        causa_basica = causa_basica[:300]

    cid_basica = last_cid if last_cid else ""

    return causa_basica, cid_basica

def _post_process(structured: dict) -> dict:
    result = dict(structured)
    campos_obrigatorios = ["NOME", "DATA_OBITO"]
    campos_importantes = ["NOME_MAE", "NASCIMENTO", "UF_OBITO",
                           "CAUSA_BASICA", "CID_BASICA", "CIDADE_OBITO"]

    erros = []
    warnings_list = []

    for campo in campos_obrigatorios:
        if not result.get(campo):
            erros.append(f"{campo} ausente")

    for campo in campos_importantes:
        if not result.get(campo):
            warnings_list.append(f"{campo} ausente")

    uf = result.get("UF_OBITO", "") or ""
    if uf and uf not in UF_VALIDAS:
        erros.append("UF_OBITO invalida")
        result["UF_OBITO"] = ""

    data = result.get("DATA_OBITO", "") or ""
    if data and not re.match(r'\d{2}/\d{2}/\d{4}', data):
        erros.append("DATA_OBITO formato invalido")
        result["DATA_OBITO"] = ""

    causa = result.get("CAUSA_BASICA", "") or ""
    if causa and any(kw in causa.lower() for kw in
                     ["condições significativas", "contribuiram",
                      "cadeia acima", "parte ii", "código", "registro"]):
        erros.append("CAUSA_BASICA")
        result["CAUSA_BASICA"] = ""

    if not erros and not warnings_list:
        result["STATUS"] = "OK"
    elif not erros:
        result["STATUS"] = "OK_COM_WARNINGS"
    else:
        result["STATUS"] = "REVISAR"

    penalty = len(erros) * 15 + len(warnings_list) * 5
    result["QUALIDADE_SCORE"] = max(0, min(100, 100 - penalty))
    result["ERROS"] = "; ".join(erros) if erros else ""

    return result

# ── Parsing Principal ──────────────────────────────────────────────────

def parse_obito(raw_text: str) -> dict:
    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]

    nome = _extract_text_after_label(
        lines, ["nome do falecido", "nome do falecida", "nome do", "falecido"]
    )
    nome_mae = _extract_text_after_label(
        lines, ["nome da mae", "nome da mãe", "mae", "mãe"]
    )
    nome_pai = _extract_text_after_label(
        lines, ["nome do pai", "pai"]
    )
    nascimento = _extract_date_after_label(
        lines, ["data de nascimento", "nascimento", "nasc."],
        forced_year=None
    )
    data_obito = _extract_date_after_label(
        lines, ["data do obito", "data do óbito", "obito", "óbito"],
        forced_year="2026"
    )
    hora_obito = _extract_time_after_label(
        lines, ["hora", "hora do obito", "hora do óbito"]
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

# ── OCR Provider ───────────────────────────────────────────────────────

def _build_ocr_payload(image_base64: str, filename: str = "image.jpg") -> dict:
    return {
        "model": OPENAI_MODEL_DEFAULT,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an OCR assistant. Transcribe all visible text in the image "
                    "faithfully, preserving line breaks and reading order exactly as they appear. "
                    "Do not summarize, interpret, paraphrase, or explain. Output only the raw text."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcreva fielmente todo o texto visível na imagem, "
                            "preservando a ordem, as quebras de linha e a formatação "
                            "exatamente como aparecem. Não resuma, não explique, não interprete."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0
    }

def _call_ocr_provider(image_base64: str, filename: str = "image.jpg") -> str:
    url = f"{OPENAI_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = _build_ocr_payload(image_base64, filename)

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    if not content or not content.strip():
        raise ValueError("Provider OCR retornou conteúdo vazio.")

    refusal_lower = content.strip().lower()
    refusal_patterns = [
        "não posso ajudar", "não posso processar", "cannot process",
        "não consigo extrair", "não é possível extrair",
        "desculpe", "sorry", "i cannot", "i can't",
    ]
    for pattern in refusal_patterns:
        if refusal_lower.startswith(pattern):
            raise ValueError(
                f"Provider recusou o processamento: '{content[:200]}'"
            )

    return content.strip()

# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/ocr", response_model=OCRResponse)
def ocr(request: OCRRequest):
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    start_time = datetime.now()

    try:
        raw_text = _call_ocr_provider(request.image, request.filename)
        structured = parse_obito(raw_text)

        validation_errors = structured.pop("VALIDATION_ERRORS", [])
        validation_warnings = structured.pop("VALIDATION_WARNINGS", [])
        score = structured.get("QUALIDADE_SCORE", 0)

        processing_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        return OCRResponse(
            requestId=request_id,
            status="processed",
            provider="openai-compatible",
            confidence=1.0,
            rawText=raw_text,
            text="",
            structured=structured,
            validation={
                "ok": len(validation_errors) == 0,
                "errors": validation_errors,
                "warnings": validation_warnings,
                "score": score,
                "status": structured.get("STATUS", "REVISAR")
            },
            warnings=[f"{campo} ausente" for campo in validation_warnings],
            processingTimeMs=processing_ms
        )

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Timeout do provider OCR")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Erro no provider OCR: {str(e)[:200]}")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Falha no provider OCR: {str(e)[:200]}")
    except Exception as e:
        logger.exception("Erro interno no OCR")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)[:200]}")

# ── Startup ────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    logger.info("=" * 50)
    logger.info("OCR API iniciando...")
    logger.info("Modelo configurado: %s", OPENAI_MODEL_DEFAULT)
    logger.info("API Base: %s", OPENAI_API_BASE)
    logger.info("Chave API configurada: %s", bool(OPENAI_API_KEY))
    logger.info("=" * 50)
