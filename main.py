import os
import re
import logging
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("ocr-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_MODEL_DEFAULT", "gpt-4o")

app = FastAPI(title="OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATE_RE = re.compile(r'(\d{1,2})\s*[/|\-.\s]\s*(\d{1,2})\s*[/|\-.\s]\s*(\d{2,4})')
TIME_RE = re.compile(r'(\d{1,2}):(\d{2})')
UF_VALIDAS = {"AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG","PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"}

def _looks_like_label(text):
    if not text: return False
    t = text.strip().lower()
    if not t: return False
    for kw in ["data do","nome do","nome da","causa","parte","hora","cartao","naturalidade","municipio","sepultamento","atestante","medico","declarante","tipo de","fetal","nao fetal","obito","nascimento","pai","mae","cartorio","registro","uf"]:
        if kw in t: return True
    return False

def _normalize_date(day, month, year, forced_year=None):
    try:
        d, m = int(day), int(month)
    except ValueError: return ""
    if d < 1 or d > 31 or m < 1 or m > 12: return ""
    y = forced_year or year
    if len(y) == 2: y = "20" + y if int(y) < 50 else "19" + y
    return "%02d/%02d/%s" % (d, m, y)

def _find_label_index(lines, labels):
    for i, line in enumerate(lines):
        if not line: continue
        clean = line.strip()
        clean = re.sub(r'^\[\d+\]\s*', '', clean)
        clean = re.sub(r'^\(\d+\)\s*', '', clean)
        clean = re.sub(r'^\d+\)\s*', '', clean)
        clean = re.sub(r'^\d+\.\s*', '', clean)
        clean = re.sub(r'^\d+\s+', '', clean)
        for label in labels:
            if clean.lower().startswith(label.lower()): return i
    return None

def _extract_text_after_label(lines, labels, max_lines=3):
    idx = _find_label_index(lines, labels)
    if idx is None: return ""
    for offset in range(1, max_lines + 1):
        if idx + offset >= len(lines): break
        c = lines[idx + offset].strip()
        if not c or _looks_like_label(c) or len(c) < 2: continue
        return c
    return ""

def _extract_date_after_label(lines, labels, forced_year=None):
    idx = _find_label_index(lines, labels)
    if idx is None: return ""
    sl = []
    for offset in range(0, 6):
        if idx + offset < len(lines): sl.append(lines[idx + offset])
    m = DATE_RE.search(" ".join(sl))
    if m: return _normalize_date(m.group(1), m.group(2), m.group(3), forced_year)
    return ""

def _extract_time_after_label(lines, labels):
    idx = _find_label_index(lines, labels)
    if idx is None: return ""
    sl = []
    for offset in range(0, 4):
        if idx + offset < len(lines): sl.append(lines[idx + offset])
    m = TIME_RE.search(" ".join(sl))
    if m: return "%s:%s" % (m.group(1), m.group(2))
    return ""

def _extract_city_state(lines, raw_text):
    text = raw_text or ""; lower = text.lower()
    cidade = ""; uf = ""
    for marker in ["municipio de ocorrencia", "local de ocorrencia"]:
        idx = lower.find(marker)
        if idx == -1: continue
        for line in text[idx:idx+300].splitlines()[1:4]:
            line = line.strip()
            if not line or len(line) < 3: continue
            if any(kw in line.lower() for kw in ["uf","municipio","pais","se estrangeiro"]): continue
            parts = line.split()
            if len(parts) >= 2 and len(parts[-1]) == 2 and parts[-1].isalpha():
                cidade = " ".join(parts[:-1]); uf = parts[-1].upper()
            elif not cidade: cidade = line
            break
        if cidade: break
    if not cidade:
        idx = lower.find("naturalidade")
        if idx != -1:
            for line in text[idx:idx+300].splitlines()[1:4]:
                line = line.strip()
                if not line or len(line) < 3: continue
                if any(kw in line.lower() for kw in ["uf","municipio","pais","se estrangeiro"]): continue
                parts = line.split()
                if len(parts) >= 2 and len(parts[-1]) == 2 and parts[-1].isalpha():
                    cidade = " ".join(parts[:-1]); uf = parts[-1].upper()
                elif not cidade: cidade = line
                break
    if not uf:
        for m in re.finditer(r'\b([A-Z]{2})\b', text):
            if m.group(1) in UF_VALIDAS: uf = m.group(1); break
    return cidade.strip(), uf

def _extract_causas(lines, raw_text):
    text = raw_text or ""; lower = text.lower()
    start = -1
    for mk in ["causas da morte", "parte i", "causa da morte"]:
        idx = lower.find(mk)
        if idx != -1: start = idx; break
    if start == -1: return "", ""
    end = len(text)
    for mk in ["parte ii", "atestante", "medico", "cartorio", "declarante"]:
        idx = lower.find(mk, start + 1)
        if idx != -1 and idx < end: end = idx
    blk = text[start:end].strip()
    blines = [l.strip() for l in blk.splitlines() if l.strip()]
    causa = ""; last_cid = ""
    for line in blines:
        ll = line.lower()
        if any(kw in ll for kw in ["parte","condicoes significativas","contribuiram","nao entraram","cadeia acima","codigo","registro","ufs"]): continue
        if len(line) < 4: continue
        if _looks_like_label(line) and len(line) < 80: continue
        cm = re.search(r'\b([A-Z]\d{2,3})\b', line)
        if cm: last_cid = cm.group(1)
        if not causa and len(line) > 5: causa = line
        elif len(line) > 5 and len(causa) < 150: causa += " | " + line
    if len(causa) > 300: causa = causa[:300]
    return causa, last_cid or ""

def _post_process(structured):
    r = dict(structured)
    erros = []; warns = []
    for c in ["NOME","DATA_OBITO"]:
        if not r.get(c): erros.append("%s ausente" % c)
    for c in ["NOME_MAE","NASCIMENTO","UF_OBITO","CAUSA_BASICA","CID_BASICA","CIDADE_OBITO"]:
        if not r.get(c): warns.append("%s ausente" % c)
    uf = r.get("UF_OBITO","") or ""
    if uf and uf not in UF_VALIDAS: erros.append("UF_OBITO invalida"); r["UF_OBITO"] = ""
    data = r.get("DATA_OBITO","") or ""
    if data and not re.match(r'\d{2}/\d{2}/\d{4}', data): erros.append("DATA_OBITO formato invalido"); r["DATA_OBITO"] = ""
    causa = r.get("CAUSA_BASICA","") or ""
    if causa and any(kw in causa.lower() for kw in ["condicoes significativas","contribuiram","cadeia acima","parte ii","codigo","registro"]):
        erros.append("CAUSA_BASICA"); r["CAUSA_BASICA"] = ""
    if not erros and not warns: r["STATUS"] = "OK"
    elif not erros: r["STATUS"] = "OK_COM_WARNINGS"
    else: r["STATUS"] = "REVISAR"
    penalty = len(erros) * 15 + len(warns) * 5
    r["QUALIDADE_SCORE"] = max(0, min(100, 100 - penalty))
    r["ERROS"] = "; ".join(erros) if erros else ""
    return r, erros, warns

def parse_obito(raw_text):
    lines = [l.strip() for l in (raw_text or "").splitlines() if l.strip()]
    nome = _extract_text_after_label(lines, ["nome do falecido","nome do falecida","nome do","falecido"])
    nome_mae = _extract_text_after_label(lines, ["nome da mae","mae"])
    nome_pai = _extract_text_after_label(lines, ["nome do pai","pai"])
    nascimento = _extract_date_after_label(lines, ["data de nascimento","nascimento","nasc."], forced_year=None)
    data_obito = _extract_date_after_label(lines, ["data do obito","obito"], forced_year="2026")
    hora_obito = _extract_time_after_label(lines, ["hora","hora do obito"])
    cidade_obito, uf_obito = _extract_city_state(lines, raw_text)
    causa_basica, cid_basica = _extract_causas(lines, raw_text)
    structured = {
        "NOME": nome or "", "NOME_MAE": nome_mae or "", "NOME_PAI": nome_pai or "",
        "NASCIMENTO": nascimento or "", "DATA_OBITO": data_obito or "",
        "HORA_OBITO": hora_obito or "", "CIDADE_OBITO": cidade_obito or "",
        "UF_OBITO": uf_obito or "", "CAUSA_BASICA": causa_basica or "",
        "CID_BASICA": cid_basica or "",
    }
    structured, erros, warns = _post_process(structured)
    return structured, erros, warns

def _build_ocr_payload(image_base64, filename="image.jpg"):
    return {
        "model": OPENAI_MODEL_DEFAULT,
        "messages": [
            {"role": "system", "content": "You are an OCR assistant. Transcribe all visible text in the image faithfully, preserving line breaks and reading order exactly as they appear. Do not summarize, interpret, paraphrase, or explain. Output only the raw text."},
            {"role": "user", "content": [
                {"type": "text", "text": "Transcreva fielmente todo o texto visivel na imagem, preservando a ordem, as quebras de linha e a formatacao exatamente como aparecem. Nao resuma, nao explique, nao interprete."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_base64, "detail": "high"}}
            ]}
        ],
        "max_tokens": 4096,
        "temperature": 0.0
    }

def _call_ocr_provider(image_base64, filename="image.jpg"):
    resp = requests.post(
        OPENAI_API_BASE + "/chat/completions",
        headers={"Authorization": "Bearer " + OPENAI_API_KEY, "Content-Type": "application/json"},
        json=_build_ocr_payload(image_base64, filename),
        timeout=120
    )
    resp.raise_for_status()
    content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content or not content.strip():
        raise ValueError("Provider OCR retornou conteudo vazio.")
    return content.strip()

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.post("/ocr")
async def ocr(request: Request):
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    start = datetime.now()

    try:
        body = await request.json()
    except Exception as e:
        logger.error("OCR %s: falha ao parsear body JSON: %s", request_id, str(e)[:200])
        return JSONResponse(status_code=422, content={
            "requestId": request_id,
            "error": "INVALID_JSON",
            "message": "Body nao e JSON valido: " + str(e)[:200]
        })

    logger.info("OCR %s: keys=%s, has_image=%s, image_len=%d, filename=%s",
                request_id, list(body.keys()), "image" in body,
                len(body.get("image","")) if isinstance(body.get("image"),str) else 0,
                body.get("filename",""))

    if "image" not in body or not isinstance(body.get("image"), str) or not body["image"]:
        return JSONResponse(status_code=422, content={
            "requestId": request_id, "error": "INVALID_IMAGE",
            "message": "Campo 'image' (string base64) obrigatorio"
        })

    try:
        raw_text = _call_ocr_provider(body["image"], body.get("filename", "image.jpg"))
        structured, erros, warns = parse_obito(raw_text)
        score = structured.get("QUALIDADE_SCORE", 0)
        status = structured.get("STATUS", "REVISAR")
        processing_ms = int((datetime.now() - start).total_seconds() * 1000)

        return JSONResponse(content={
            "requestId": request_id,
            "status": "processed",
            "provider": "openai-compatible",
            "confidence": 1.0,
            "rawText": raw_text,
            "text": "",
            "structured": structured,
            "validation": {
                "ok": len(erros) == 0,
                "errors": erros,
                "warnings": warns,
                "score": score,
                "status": status
            },
            "warnings": ["%s ausente" % w for w in warns],
            "processingTimeMs": processing_ms
        })
    except Exception as e:
        logger.exception("Erro no OCR %s", request_id)
        return JSONResponse(status_code=500, content={
            "requestId": request_id, "error": "INTERNAL_ERROR",
            "message": "Erro interno: " + str(e)[:500]
        })

@app.on_event("startup")
def startup():
    logger.info("=" * 50)
    logger.info("OCR API iniciando...")
    logger.info("Modelo=%s, key=%s", OPENAI_MODEL_DEFAULT, bool(OPENAI_API_KEY))
@app.post("/echo-request")
async def echo_request(request: Request):
    """Retorna exatamente o que o servidor recebeu — para diagnóstico."""
    try:
        body = await request.json()
        return JSONResponse(content={
            "status": "recebido",
            "keys": list(body.keys()),
            "has_image": "image" in body,
            "image_type": type(body.get("image", "")).__name__,
            "image_len": len(body.get("image", "")) if isinstance(body.get("image"), str) else 0,
            "image_prefix": (body.get("image", "")[:80] + "...") if isinstance(body.get("image"), str) else "",
            "filename": body.get("filename", ""),
            "content_type": request.headers.get("content-type", ""),
            "content_length": request.headers.get("content-length", ""),
        })
    except Exception as e:
        return JSONResponse(status_code=422, content={
            "status": "erro",
            "erro": str(e)[:200]
        })    logger.info("=" * 50)
function testarEcho() {
  // Reuse a mesma imagem do OCR
  var fileId = "1BsSuzIRH2BJ8e6SFpZrjTkcZP50rItOt";
  var file = DriveApp.getFileById(fileId);
  var blob = file.getBlob();
  var base64 = Utilities.base64Encode(blob.getBytes());

  var payload = {
    "image": base64,
    "filename": "img61.jpg"
  };

  var options = {
    "method": "post",
    "contentType": "application/json",
    "payload": JSON.stringify(payload),
    "muteHttpExceptions": true
  };

  var response = UrlFetchApp.fetch("https://seu-app.onrender.com/echo-request", options);
  Logger.log("Status: " + response.getResponseCode());
  Logger.log("Body: " + response.getContentText());
}
