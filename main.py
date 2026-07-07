# FILE: main.py
import base64
import hashlib
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Configuração de ambiente
# ---------------------------------------------------------------------------

ENDPOINT_AUTH_TOKEN = os.environ.get("ENDPOINT_AUTH_TOKEN", "")
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_MODEL_DEFAULT", "gpt-4o")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "10"))
PORT = int(os.environ.get("PORT", "8000"))

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

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
    "QTD_GARBAGE", "PROTOCOLO_TEV", "ERROS", "QUALIDADE_SCORE",
    "HASH_ARQUIVO", "HASH_CONTEUDO", "STATUS", "NOME_MES", "DATA_PROCESSAMENTO",
]

UF_VALIDAS = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
}

MESES = {
    "01": "JANEIRO", "02": "FEVEREIRO", "03": "MARCO", "04": "ABRIL",
    "05": "MAIO", "06": "JUNHO", "07": "JULHO", "08": "AGOSTO",
    "09": "SETEMBRO", "10": "OUTUBRO", "11": "NOVEMBRO", "12": "DEZEMBRO",
}

REFUSAL_PHRASES = [
    "i'm sorry", "i can't assist with that", "i can't assist with that",
    "cannot assist", "unable to help", "cannot help with that request",
    "i can't help with that", "desculpe", "não posso ajudar", "nao posso ajudar",
]


app = FastAPI(title="obito-ocr-service", version="1.0.0")


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "obito-ocr-service", "time": _iso_now()}


@app.post("/ocr")
async def ocr(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    # Autenticação Bearer
    if not ENDPOINT_AUTH_TOKEN:
        return _error_response("AUTH_NOT_CONFIGURED", "Token de autenticação não configurado", None, 500)
    if not authorization or not authorization.startswith("Bearer "):
        return _error_response("UNAUTHORIZED", "Token ausente ou inválido", None, 401)
    token = authorization[len("Bearer "):].strip()
    if token != ENDPOINT_AUTH_TOKEN:
        return _error_response("UNAUTHORIZED", "Token inválido", None, 401)

    # Parse do corpo
    try:
        body = await request.json()
    except Exception:
        return _error_response("INVALID_JSON", "Corpo não é JSON válido", None, 400)
    if not isinstance(body, dict):
        return _error_response("INVALID_JSON", "JSON deve ser um objeto", None, 400)

    request_id = body.get("requestId") or _generate_request_id()

    # Campos obrigatórios
    file_b64 = body.get("file")
    mime_type = body.get("mimeType")
    if not file_b64 or not isinstance(file_b64, str):
        return _error_response("MISSING_FILE", "Campo 'file' ausente ou inválido", request_id, 400)
    if not mime_type or mime_type not in ALLOWED_MIME:
        return _error_response("INVALID_MIME", "mimeType inválido", request_id, 422)
    if mime_type == "application/pdf":
        return _error_response("PDF_NOT_SUPPORTED_IN_V1", "PDF não suportado nesta versão", request_id, 422)

    # Decodifica base64
    try:
        file_bytes = base64.b64decode(file_b64, validate=True)
    except Exception:
        return _error_response("INVALID_BASE64", "Base64 inválido", request_id, 400)

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return _error_response("FILE_TOO_LARGE", f"Arquivo excede {MAX_FILE_SIZE_MB}MB", request_id, 413)

    model = body.get("model") or OPENAI_MODEL_DEFAULT

    # Chamada ao provedor OCR
    try:
        ocr_text, confidence = call_ocr_provider(file_bytes, mime_type, model)
    except OCRProviderError as exc:
        return _error_response("OCR_PROVIDER_ERROR", str(exc), request_id, 502)
    except Exception as exc:
        return _error_response("OCR_PROVIDER_ERROR", f"Falha no provedor OCR: {exc}", request_id, 502)

    # Parsing estruturado com proteção interna
    try:
        structured = parse_obito(ocr_text)
        validation = validate_structured(structured)
    except Exception as exc:
        return _error_response("INTERNAL_ERROR", f"Erro interno no parser/servidor: {exc}", request_id, 500)

    # Pós-processamento: hashes, datas, status
    structured["HASH_ARQUIVO"] = hashlib.sha256(file_bytes).hexdigest()
    structured["HASH_CONTEUDO"] = hashlib.sha256(ocr_text.encode("utf-8")).hexdigest()
    structured["DATA_PROCESSAMENTO"] = _iso_now()
    structured["NOME_MES"] = _month_name_from_date(structured.get("DATA_OBITO", ""))
    structured["ERROS"] = " | ".join(validation["errors"])
    structured["QUALIDADE_SCORE"] = validation["score"]
    structured["STATUS"] = validation["status"]
    structured["NOMES_OK"] = "SIM" if validation["names_ok"] else "NAO"
    structured["NOME_OK"] = "SIM" if validation.get("nome_ok", False) else "NAO"

    # Garantir todas as chaves do HEADER presentes
    for key in HEADER:
        structured.setdefault(key, "")

    warnings = validation["warnings"]

    return JSONResponse(
        status_code=200,
        content={
            "text": ocr_text,
            "confidence": confidence,
            "provider": "openai-compatible",
            "requestId": request_id,
            "warnings": warnings,
            "rawText": ocr_text,
            "structured": structured,
            "validation": validation,
            "headerOrder": HEADER,
        },
    )


# Erros auxiliares

def _error_response(code: str, message: str, request_id: Optional[str], status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "code": code,
            "message": message,
            "requestId": request_id or _generate_request_id(),
        },
    )


def _generate_request_id() -> str:
    return f"req_{int(time.time() * 1000)}"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# Exceção específica do provedor OCR

class OCRProviderError(Exception):
    pass


# Adaptador OCR (OpenAI-compatible)

def call_ocr_provider(file_bytes: bytes, mime_type: str, model: str) -> Tuple[str, float]:
    if not OPENAI_API_URL or not OPENAI_API_KEY:
        raise OCRProviderError("Provedor OCR não configurado")

    data_url = f"data:{mime_type};base64,{base64.b64encode(file_bytes).decode('ascii')}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é um motor de OCR especializado em certidões de óbito brasileiras. "
                    "Transcreva fielmente todo o texto visível, preservando a ordem das linhas, "
                    "rótulos e valores. Não resuma, não interprete, não omita campos."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extraia o texto completo da imagem, linha por linha, sem comentários.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        "max_tokens": 4000,
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=120)
    except Exception as exc:
        raise OCRProviderError(f"Erro de comunicação: {exc}")

    if resp.status_code != 200:
        raise OCRProviderError(f"Provedor retornou HTTP {resp.status_code}")

    try:
        data = resp.json()
    except Exception:
        raise OCRProviderError("Resposta do provedor não é JSON válido")

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise OCRProviderError("Estrutura de resposta do provedor inválida")

    if not isinstance(content, str) or not content.strip():
        raise OCRProviderError("Provedor retornou conteúdo vazio")

    # Detecção de recusa
    lower = content.lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in lower:
            raise OCRProviderError("Provedor recusou o processamento")

    confidence = 0.0
    try:
        confidence = float(data.get("usage", {}).get("completion_tokens", 0)) / 1000.0
        confidence = min(max(confidence, 0.0), 1.0)
    except Exception:
        confidence = 0.0

    return content.strip(), confidence


# Parser de certidão de óbito

def _normalize_line(line: str) -> str:
    """Normaliza uma linha removendo prefixos numéricos de rótulos, sem afetar valores."""
    s = line.strip()
    if not s:
        return ""
    # Remove prefixos como '2 ', '2. ', '2) ', '2- ' apenas quando seguidos de texto
    m = re.match(r"^(\d{1,2})[.)\-:]?\s+([A-Za-zÀ-ú].*)$", s)
    if m:
        return m.group(2).strip()
    return s


def _is_time_value(s: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}:\d{2}", s.strip()))


def _is_date_value(s: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}[ /\-]\d{1,2}[ /\-]\d{2,4}", s.strip()))


def _is_duration_token(s: str) -> bool:
    """Identifica tokens de duração/intervalo que não são causas válidas."""
    return bool(re.fullmatch(r"[<>]?\s*\d+\s*[dhms]", s.strip(), re.IGNORECASE))


def _normalize_date(value: str) -> str:
    """Normaliza DD MM AAAA, DD-MM-AAAA, DD/MM/AAAA para DD/MM/AAAA."""
    v = value.strip()
    m = re.match(r"^(\d{1,2})[ /\-](\d{1,2})[ /\-](\d{2,4})$", v)
    if not m:
        return v
    d, mo, y = m.group(1), m.group(2), m.group(3)
    if len(y) == 2:
        y = "19" + y if int(y) > 30 else "20" + y
    return f"{int(d):02d}/{int(mo):02d}/{y}"


def _normalize_time(value: str) -> str:
    v = value.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", v)
    if not m:
        return v
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def _find_next_useful(lines: List[str], idx: int) -> Tuple[int, str]:
    """Retorna índice e conteúdo da próxima linha útil a partir de idx+1."""
    for j in range(idx + 1, len(lines)):
        v = _normalize_line(lines[j])
        if v:
            return j, v
    return -1, ""


def _extract_cid_from_text(text: str) -> str:
    """Extrai o último CID encontrado no texto (formato letra + dígitos)."""
    matches = re.findall(r"\b([A-TV-Z]\d{2}(?:\.\d{1,2})?)\b", text, re.IGNORECASE)
    if not matches:
        return ""
    return matches[-1].upper()


def _clean_causa_text(value: str) -> str:
    if not value:
        return ""
    text = value.strip()
    # Limpa espaços extras
    text = re.sub(r"\s+", " ", text).strip()
    # Remove prefixos residuais no início: 'd ', '(d) ', '- ', ': ' e variações
    while True:
        prev = text
        text = re.sub(r"^\(?[a-zA-Z]\)?[\s\-\:]+", "", text)
        text = re.sub(r"^[\-\:\.\s]+", "", text)
        text = text.strip()
        if text == prev:
            break
    if not text:
        return ""
    low = text.lower()
    aux_words = {"meses", "dias", "horas", "minutos", "ignorado"}
    tokens = low.split()
    # Bloqueia palavras auxiliares isoladas ou combinações auxiliares
    if tokens and all(t in aux_words for t in tokens):
        return ""
    # Bloqueia combinações auxiliares conhecidas
    if low in {"meses dias", "dias horas", "horas minutos", "meses dias horas minutos ignorado"}:
        return ""
    # Bloqueia CID puro
    if re.fullmatch(r"[A-TV-Z]\d{2}(?:\.\d{1,2})?", text, re.IGNORECASE):
        return ""
    # Bloqueia duração/intervalo: >7d, <24h, 3d, 12h, 30m
    if re.fullmatch(r"[<>]?\s*\d+\s*[dhms]", text, re.IGNORECASE):
        return ""
    return text.strip()



def parse_obito(raw_text: str) -> Dict[str, Any]:
      # NOME: apenas rótulo 'Nome do Falecido' (variantes)
    for i, ln in enumerate(lines):
        if re.fullmatch(r"Nome\s+do\s+Falecido", ln, re.IGNORECASE):
            nome_parts: List[str] = []
            for j in range(i + 1, min(i + 7, len(lines))):
                cand = _normalize_line(lines[j]).strip()
                if not cand:
                    continue
                # Ignora linhas que são só número, como '6'
                if re.fullmatch(r"\d{1,2}", cand):
                    continue
                # Ignora rótulos
                if _looks_like_label(cand):
                    continue
                # Precisa conter letras
                if not re.search(r"[A-Za-zÀ-ú]", cand):
                    continue
                nome_parts.append(cand)
                # Tenta concatenar próxima linha se parecer continuação de nome
                if j + 1 < len(lines):
                    nxt = _normalize_line(lines[j + 1]).strip()
                    if (nxt
                            and not re.fullmatch(r"\d{1,2}", nxt)
                            and not _looks_like_label(nxt)
                            and re.search(r"[A-Za-zÀ-ú]", nxt)
                            and not re.search(r"(Nome|M[ãa]e|Pai|Data|Hora|Munic[íi]pio|UF|CPF|RG|Parte|CID|CRM|M[ée]dico)", nxt, re.IGNORECASE)):
                        nome_parts.append(nxt)
                break
            if nome_parts:
                result["NOME"] = " ".join(nome_parts).strip()
            break

    # NOME_MAE
    for i, ln in enumerate(lines):
        if re.fullmatch(r"Nome\s+da\s+M[ãa]e", ln, re.IGNORECASE):
            _, val = _find_next_useful(lines, i)
            if val and not _looks_like_label(val):
                result["NOME_MAE"] = val.strip()
                break

    # NOME_PAI
    for i, ln in enumerate(lines):
        if re.fullmatch(r"Nome\s+do\s+Pai", ln, re.IGNORECASE):
            _, val = _find_next_useful(lines, i)
            if val and not _looks_like_label(val):
                result["NOME_PAI"] = val.strip()
                break

    # NASCIMENTO
    for i, ln in enumerate(lines):
        if re.search(r"Data\s+de\s+Nascimento", ln, re.IGNORECASE):
            _, val = _find_next_useful(lines, i)
            if val and _is_date_value(val):
                result["NASCIMENTO"] = _normalize_date(val)
                break

    # DATA_OBITO
    for i, ln in enumerate(lines):
        if re.search(r"Data\s+do\s+[óo]bito", ln, re.IGNORECASE):
            _, val = _find_next_useful(lines, i)
            if val and _is_date_value(val):
                result["DATA_OBITO"] = _normalize_date(val)
                break

    # HORA_OBITO: linha exatamente 'Hora'
    for i, ln in enumerate(lines):
        if re.fullmatch(r"Hora", ln, re.IGNORECASE):
            _, val = _find_next_useful(lines, i)
            if val and _is_time_value(val):
                result["HORA_OBITO"] = _normalize_time(val)
                break

    # CIDADE_OBITO: 'Município de ocorrência' (variantes)
    for i, ln in enumerate(lines):
        if re.search(r"Munic[íi]pio\s+de\s+ocorr[êe]ncia", ln, re.IGNORECASE):
            _, val = _find_next_useful(lines, i)
            if val and not _looks_like_label(val):
                result["CIDADE_OBITO"] = val.strip()
                break

    # UF_OBITO: após município ou rótulo UF
    for i, ln in enumerate(lines):
        if re.search(r"Munic[íi]pio\s+de\s+ocorr[êe]ncia", ln, re.IGNORECASE):
            for j in range(i + 1, min(i + 6, len(lines))):
                cand = lines[j].strip()
                if cand.upper() in UF_VALIDAS:
                    result["UF_OBITO"] = cand.upper()
                    break
            if result["UF_OBITO"]:
                break
    if not result["UF_OBITO"]:
        for i, ln in enumerate(lines):
            if re.fullmatch(r"UF", ln, re.IGNORECASE):
                _, val = _find_next_useful(lines, i)
                if val and val.upper() in UF_VALIDAS:
                    result["UF_OBITO"] = val.upper()
                    break

    # Causas da morte (Parte I)
    causas = _extract_causas(lines)
    for k, causa in enumerate(causas[:5]):
        key = "CAUSA_MORTE" if k == 0 else f"CAUSA_MORTE_{k + 1}"
        result[key] = causa

    # CAUSA_BASICA: limpa e pega a última não vazia
    cleaned_causas = [_clean_causa_text(c) for c in causas]
    valid_causas = [c for c in cleaned_causas if c]
    causa_basica = valid_causas[-1] if valid_causas else ""
    result["CAUSA_BASICA"] = causa_basica

    # CID_BASICA: procura primeiro em CAUSA_BASICA, depois no raw_text
    cid_basica = _extract_cid_from_text(causa_basica) if causa_basica else ""
    if not cid_basica:
        cid_basica = _extract_cid_from_text(raw_text)
    result["CID_BASICA"] = cid_basica
    result["CODIGO_CAUSA_BASICA"] = cid_basica

    return result


def _looks_like_label(s: str) -> bool:
    """Heurística para identificar rótulos de formulário em vez de conteúdo."""
    if not s:
        return False
    s_stripped = s.strip()
    # Rótulos terminados em ':'
    if s_stripped.endswith(":"):
        return True
    # Palavras específicas
    low = s_stripped.lower()
    if low in {"do médico", "do medico", "horas minutos", "meses dias horas minutos ignorado"}:
        return True
    # Padrões conhecidos
    patterns = [
        r"^(parte\s+i|parte\s+ii|cid|crm|uf)$",
        r"^nome$", r"^nome\s+da\s+m[ãa]e$", r"^nome\s+do\s+pai$",
        r"^cpf$", r"^rg$", r"^cid$", r"^causa\s+morte$",
        r"^causa\s+b[áa]sica$", r"^data$", r"^hora$",
    ]
    if any(re.fullmatch(p, s_stripped, re.IGNORECASE) for p in patterns):
        return True
    return False


_extract_causas(lines: List[str]) -> List[str]:
    """Extrai causas da Parte I, parando em marcadores de seção posteriores.
    Ignora linhas auxiliares, tokens de duração, CID puro e palavras auxiliares isoladas.
    Remove CIDs colados ao final de descrições clínicas válidas."""
    start = -1
    for i, ln in enumerate(lines):
        if re.search(r"CAUSAS\s+DA\s+MORTE", ln, re.IGNORECASE):
            start = i + 1
            break
    if start < 0:
        return []

    stop_markers = [
        r"Parte\s+II", r"Outras\s+condi[çc][õo]es\s+significativas",
        r"Nome\s+do\s+M[ée]dico", r"CRM", r"[óo]bito\s+atestado\s+por\s+M[ée]dico",
        r"PROV[ÁA]VEIS\s+CIRCUNST[ÂA]NCIAS",
    ]
    ignore_markers = [
        r"Parte\s+I", r"Devido\s+ou\s+como\s+consequ[êe]ncia\s+de",
        r"Intervalo\s+entre\s+o\s+in[íi]cio\s+e\s+a\s+morte", r"^CID$",
        r"Meses\s+Dias\s+Horas\s+Minutos\s+Ignorado",
    ]

    aux_words = {"meses", "dias", "horas", "minutos", "ignorado"}

    def _is_pure_cid(s: str) -> bool:
        return bool(re.fullmatch(r"[A-TV-Z]\d{2}(?:\.\d{1,2})?", s.strip(), re.IGNORECASE))

    def _is_aux_only(s: str) -> bool:
        tokens = s.lower().split()
        return bool(tokens) and all(t in aux_words for t in tokens)

    causas: List[str] = []
    for ln in lines[start:]:
        s = ln.strip()
        if not s:
            continue
        if any(re.search(p, s, re.IGNORECASE) for p in stop_markers):
            break
        if any(re.search(p, s, re.IGNORECASE) for p in ignore_markers):
            continue
        if _is_duration_token(s):
            continue
        if _looks_like_label(s):
            continue
        if _is_pure_cid(s):
            continue
        if _is_aux_only(s):
            continue
        # Remove CID colado ao final da causa (ex.: "Infecção do trato urinário N39.0")
        clean = re.sub(r"\s+\b[A-TV-Z]\d{2}(?:\.\d{1,2})?\b$", "", s, flags=re.IGNORECASE).strip()
        if not clean or _is_pure_cid(clean) or _is_aux_only(clean):
            continue
        # Aplica _clean_causa_text antes de adicionar
        clean = _clean_causa_text(clean)
        if not clean:
            continue
        causas.append(clean)
    return causas

# Validação

def validate_structured(structured: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    nascimento = structured.get('NASCIMENTO')
    if nascimento and not _valid_date(nascimento):
        errors.append('NASCIMENTO inválido')

    data_obito = structured.get('DATA_OBITO')
    if data_obito and not _valid_date(data_obito):
        errors.append('DATA_OBITO inválida')

    hora_obito = structured.get('HORA_OBITO')
    if hora_obito and not _valid_time(hora_obito):
        errors.append('HORA_OBITO inválida')

    uf_obito = structured.get('UF_OBITO')
    if uf_obito and uf_obito not in UF_VALIDAS:
        errors.append('UF_OBITO inválida')

    cep = structured.get('CEP')
    if cep and not re.fullmatch(r'\d{5}-?\d{3}', str(cep)):
        errors.append('CEP inválido')

    nome = structured.get('NOME')
    if not nome:
        errors.append('NOME ausente')

    if not structured.get('DATA_OBITO'):
        errors.append('DATA_OBITO ausente')

    causa_basica = structured.get('CAUSA_BASICA')
    if not causa_basica:
        errors.append('CAUSA_BASICA ausente')
    else:
        if re.fullmatch(r'[A-TV-Z]\d{2}(?:\.\d{1,2})?', str(causa_basica), re.IGNORECASE):
            errors.append('CAUSA_BASICA parece ser um CID puro')

    nome_pai = structured.get('NOME_PAI')
    nome_mae = structured.get('NOME_MAE')

    if nome and nome_pai and nome == nome_pai:
        errors.append('NOME igual a NOME_PAI')

    if nome and nome_mae and nome == nome_mae:
        errors.append('NOME igual a NOME_MAE')

    if not structured.get('CID_BASICA'):
        warnings.append('CID_BASICA não localizado')

    nome_ok = bool(nome) and not _looks_like_label(nome)
    names_ok = all(
        bool(structured.get(campo)) and not _looks_like_label(structured.get(campo))
        for campo in ('NOME', 'NOME_MAE', 'NOME_PAI')
    )

    score = _compute_score(structured, errors, warnings)

    if errors:
        status = 'REVISAR'
    elif score < 90 and warnings:
        status = 'REVISAR'
    elif not structured.get('CAUSA_BASICA'):
        status = 'REVISAR'
    elif not structured.get('CID_BASICA'):
        status = 'REVISAR'
    else:
        status = 'OK'

    return {
        'ok': len(errors) == 0 and status == 'OK',
        'errors': errors,
        'warnings': warnings,
        'computed': {
            'score': score,
            'status': status,
            'names_ok': names_ok,
            'nome_ok': nome_ok,
        },
        'score': score,
        'status': status,
        'names_ok': names_ok,
        'nome_ok': nome_ok,
    }


def _valid_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{2}/\d{2}/\d{4}", value))


def _valid_time(value: str) -> bool:
    return bool(re.fullmatch(r"\d{2}:\d{2}", value))


def _compute_score(structured: Dict[str, Any], errors: List[str], warnings: List[str]) -> int:
    pesos = {
        "NOME": 15, "NOME_MAE": 8, "NOME_PAI": 8, "NASCIMENTO": 8,
        "DATA_OBITO": 12, "HORA_OBITO": 6, "CIDADE_OBITO": 8, "UF_OBITO": 6,
        "CAUSA_BASICA": 15, "CID_BASICA": 8, "SEXO": 3, "ESTADO_CIVIL": 3,
    }
    total = 0
    for campo, peso in pesos.items():
        val = structured.get(campo, "")
        if val and not _looks_like_label(val):
            total += peso
    total -= len(errors) * 5
    total -= len(warnings) * 2
    return max(0, min(100, total))


def _month_name_from_date(date_str: str) -> str:
    if not date_str:
        return ""
    m = re.match(r"^\d{2}/(\d{2})/\d{4}$", date_str)
    if not m:
        return ""
    return MESES.get(m.group(1), "")


# Ponto de entrada

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
