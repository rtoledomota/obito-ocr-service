"""
main.py — Serviço OCR para Declarações de Óbito
================================================
Stack: FastAPI + Google Cloud Vision (primário) / OpenAI (fallback)
       Google Drive (imagens) + Google Sheets (auditoria)
Deploy: Render (qualquer plano)

Variáveis de ambiente obrigatórias:
  GOOGLE_VISION_API_KEY       Chave de API do Google Cloud Vision
  DRIVE_SERVICE_ACCOUNT_JSON  JSON da service account do Google
  DRIVE_FOLDER_ID             ID da pasta raiz no Google Drive
  SHEET_ID                    ID da planilha de auditoria

Variáveis opcionais:
  ENDPOINT_AUTH_TOKEN         Token Bearer para autenticar endpoints
  OCR_PROVIDER                "google_vision" (padrão) ou "openai"
  OPENAI_API_KEY              Necessário se OCR_PROVIDER=openai
  OPENAI_API_URL              URL do provedor OpenAI-compatible
  OPENAI_MODEL_DEFAULT        Modelo (padrão: gpt-4o-mini)
  MAX_FILE_SIZE_MB            Tamanho máximo de arquivo (padrão: 10)
  PORT                        Porta HTTP (padrão: 8000)
  POLL_INTERVAL_MINUTES       Intervalo do monitor (padrão: 60)
  AUTO_PROCESS_ENABLED        "true" para monitor automático
  AUDIT_SHEET_TITLE           Título da planilha (padrão: "Auditoria Obito OCR")
"""

import os, re, io, json, base64, hashlib, unicodedata, gc
import datetime as dt
from datetime import datetime, timedelta
from threading import Thread, Event, Lock
from typing import Any, Dict, List, Optional, Tuple

import requests
import uvicorn
import logging

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── Logger ──────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── Configuração ────────────────────────────────────────────────
ENDPOINT_AUTH_TOKEN = os.environ.get("ENDPOINT_AUTH_TOKEN", "")
GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")
OCR_PROVIDER = os.environ.get("OCR_PROVIDER", "google_vision").lower()
OPENAI_API_URL = os.environ.get(
    "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_MODEL_DEFAULT", "gpt-4o-mini")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
PORT = int(os.environ.get("PORT", "8000"))

DRIVE_SERVICE_ACCOUNT_JSON = os.environ.get("DRIVE_SERVICE_ACCOUNT_JSON", "")
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "")
SHEET_ID = os.environ.get("SHEET_ID", "")
AUDIT_SHEET_TITLE = os.environ.get("AUDIT_SHEET_TITLE", "Auditoria Obito OCR")
POLL_INTERVAL_MINUTES = int(os.environ.get("POLL_INTERVAL_MINUTES", "60"))
AUTO_PROCESS_ENABLED = os.environ.get("AUTO_PROCESS_ENABLED", "false").lower() == "true"

# ── HEADER completo (42+ campos) ────────────────────────────────
HEADER = [
    "NOME", "NOME_SOCIAL", "NASCIMENTO", "SEXO", "RACA_COR", "ESTADO_CIVIL",
    "NACIONALIDADE", "NOME_MAE", "NOME_PAI", "PROFISSAO", "LOGRADOURO",
    "NUMERO", "COMPLEMENTO", "BAIRRO", "CIDADE", "UF", "CEP",
    "CIDADE_NASCIMENTO", "UF_NASCIMENTO", "CPF", "RG", "ORGAO_EMISSOR_RG",
    "DATA_OBITO", "HORA_OBITO", "LOCAL_OBITO", "CIDADE_OBITO", "UF_OBITO",
    "CAUSA_MORTE", "CAUSA_MORTE_2", "CAUSA_MORTE_3", "CAUSA_MORTE_4",
    "CAUSA_MORTE_5", "CAUSA_BASICA",
    "CODIGO_CAUSA_BASICA", "CODIGO_CAUSA_MORTE",
    "CODIGO_CAUSA_MORTE_2", "CODIGO_CAUSA_MORTE_3",
    "CODIGO_CAUSA_MORTE_4", "CODIGO_CAUSA_MORTE_5",
    "CID_BASICA", "CID_MORTE", "CID_MORTE_2", "CID_MORTE_3",
    "CID_MORTE_4", "CID_MORTE_5",
    "TIPO_OBITO", "ASSISTIDO", "DATA_ATESTADO",
    "NOMES_OK", "NOME_OK", "GARBAGE_CODES", "QTD_GARBAGE",
    "PROTOCOLO_TEV", "ERROS", "QUALIDADE_SCORE", "HASH_ARQUIVO",
    "HASH_CONTEUDO", "STATUS", "NOME_MES", "DATA_PROCESSAMENTO",
    "DO_NUMERO", "MEDICO_ATESTANTE", "CRM_MEDICO", "IDADE_ANOS",
    "PARTE_II", "INTERVALO_DOENCA_MORTE",
]

# Colunas escritas na planilha (subset do HEADER, ordem da aba Auditoria)
AUDIT_COLUMNS = [
    "DATA_PROCESSAMENTO", "NOME_ARQUIVO", "STATUS", "QUALIDADE_SCORE",
    "NOME", "NOME_MAE", "NASCIMENTO", "IDADE_ANOS",
    "DATA_OBITO", "HORA_OBITO", "CIDADE_OBITO", "UF_OBITO",
    "CAUSA_MORTE", "CAUSA_BASICA", "CID_BASICA", "TIPO_OBITO",
    "DO_NUMERO", "MEDICO_ATESTANTE", "CRM_MEDICO",
    "PARTE_II", "INTERVALO_DOENCA_MORTE",
    "ERROS", "HASH_ARQUIVO",
]

UF_VALIDAS = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT",
    "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO",
    "RR", "SC", "SP", "SE", "TO",
}

MESES_PT = {
    "01": "JANEIRO", "02": "FEVEREIRO", "03": "MARCO", "04": "ABRIL",
    "05": "MAIO", "06": "JUNHO", "07": "JULHO", "08": "AGOSTO",
    "09": "SETEMBRO", "10": "OUTUBRO", "11": "NOVEMBRO", "12": "DEZEMBRO",
}

REFUSAL_PHRASES = [
    "i'm sorry", "i can't assist with that", "i can't assist",
    "cannot assist", "unable to help", "cannot help with that request",
    "i can't help with that", "desculpe", "não posso ajudar", "nao posso ajudar",
]

NOISE_LINES = {
    "parte i", "parte ii", "devido ou como consequência de", "devido a",
    "intervalo entre o início e a morte", "cid",
    "meses dias horas minutos ignorado", "meses", "dias", "horas",
    "minutos", "ignorado", "nome", "nome do médico", "nome do medico",
    "crm", "assinatura", "carimbo", "uf", "município", "municipio",
    "data", "hora", "local", "causas da morte", "causa da morte",
    "causas", "causa", "outras condições significativas",
    "outras condicoes significativas", "prováveis circunstâncias",
    "provaveis circunstancias", "óbito atestado por médico",
    "obito atestado por medico", "endereço", "endereco", "logradouro",
    "número", "numero", "complemento", "bairro", "cep", "cpf", "rg",
    "sexo", "raça", "raca", "estado civil", "nacionalidade", "profissão",
    "profissao", "ocupação", "ocupacao", "naturalidade",
}

# ── Utilitários de texto ────────────────────────────────────────

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def _unaccent(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )

def _norm_label(s: str) -> str:
    return _unaccent(s).lower().strip()

def _normalize_lines(text: str) -> List[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]

def _strip_numeric_prefix(line: str) -> str:
    s = line.strip()
    m = re.match(r"^(\d+\s*[\.\):\-]?\s*)(.*)$", s)
    if m and re.search(r"[A-Za-zÀ-ú]", m.group(2)):
        return m.group(2).strip()
    return s

def _build_pairs(text: str) -> List[Tuple[str, str]]:
    return [(_strip_numeric_prefix(l), l) for l in _normalize_lines(text)]

def _is_noise_line(norm_line: str) -> bool:
    nl = _norm_label(norm_line)
    if not nl:
        return True
    if nl in NOISE_LINES:
        return True
    if nl.endswith(":") and len(nl) < 40:
        return True
    if re.fullmatch(r"[\d\s\.\-:/]+", norm_line) and len(norm_line) < 3:
        return True
    return False

# ── Normalização de data ────────────────────────────────────────

def _normalize_date_ocr(raw: str) -> str:
    """Normaliza data do OCR: '31 05 2022', '31/05/2022', DDMMYYYY etc."""
    if not raw or not raw.strip():
        return ""
    raw = raw.strip()
    # Remove sufixo "Hora 22:45" etc.
    raw = re.sub(r"\s+Hora.*$", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"[\s\-\.]+", "/", raw)
    partes = [p for p in raw.split("/") if p.strip()]
    if len(partes) == 3 and all(p.isdigit() for p in partes):
        return "/".join(partes)
    # Tenta DDMMYYYY sem separador
    nums = re.findall(r"\d+", raw)
    for n in nums:
        if len(n) == 8:
            d, m, a = int(n[0:2]), int(n[2:4]), int(n[4:8])
            if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= a <= 2100:
                return f"{d:02d}/{m:02d}/{a}"
    return ""

def _normalize_date(raw: str) -> str:
    """Converte data para DD/MM/AAAA. Retorna vazio se inválida."""
    if not raw or not raw.strip():
        return ""
    raw = raw.strip()
    raw = re.sub(r"\([^)]*\)", "", raw).strip()
    raw = re.sub(r"[.\s]+", "/", raw)
    try:
        dt.datetime.strptime(raw, "%d/%m/%Y")
        return raw
    except ValueError:
        pass
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            d = dt.datetime.strptime(raw, fmt)
            if 1900 <= d.year <= dt.datetime.now().year + 1:
                return d.strftime("%d/%m/%Y")
        except ValueError:
            continue
    nums = re.findall(r"\d+", raw)
    if len(nums) >= 3:
        for a, b in [(0, 1), (1, 0)]:
            try:
                dia, mes, ano = int(nums[a]), int(nums[b]), int(nums[-1])
                if len(nums[-1]) == 2:
                    ano += 2000 if ano < 50 else 1900
                if 1 <= dia <= 31 and 1 <= mes <= 12 and 1900 <= ano <= dt.datetime.now().year + 1:
                    return dt.datetime(ano, mes, dia).strftime("%d/%m/%Y")
            except (ValueError, IndexError):
                continue
    return ""

def _normalize_hour(value: str) -> str:
    if not value:
        return ""
    v = value.strip()
    m = re.search(r"(\d{1,2})[:hH]+(\d{2})", v)
    if m:
        h, mm = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mm <= 59:
            return f"{h:02d}:{mm:02d}"
    return v

def _is_valid_hour(value: str) -> bool:
    return bool(re.fullmatch(r"\d{2}:\d{2}", value or "")) and _normalize_hour(value) == value

def _normalize_uf(value: str) -> str:
    if not value:
        return ""
    uf = value.strip().upper()
    return uf if uf in UF_VALIDAS else ""

def _normalize_cep(value: str) -> str:
    if not value:
        return ""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 8:
        return f"{digits[:5]}-{digits[5:]}"
    return value.strip()

def _clean_field(value: str) -> str:
    """Remove instruções do formulário e códigos soltos."""
    if not value:
        return ""
    instructions = [
        r"ANOTE SOMENTE UM DIAGNÓSTICO POR LINHA",
        r"Não preencher este espaço",
        r"PREENCHEMENTO EXCLUSIVO",
        r"PREENCHEMENTO EXCLUSIVO PARA ÓBITOS FETAIS E DE ME",
        r"Menores de 1 ano:", r"Menos de 1 ano:",
        r"Escolaridade\s*\([^)]*\)",
    ]
    for instr in instructions:
        value = re.sub(instr, "", value, flags=re.IGNORECASE).strip()
    if re.match(r"^\d{4,}$", value):
        return ""
    value = re.sub(r"(\D)\d{3,}\s*$", r"\1", value).strip()
    return value.strip()

# ── OCR — Google Cloud Vision (primário) ────────────────────────

class OCRProviderError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code
        self.code = "OCR_PROVIDER_ERROR"

def _detect_refusal(text: str) -> bool:
    if not text:
        return True
    low = text.lower().strip()
    for phrase in REFUSAL_PHRASES:
        if phrase in low:
            return True
    alnum = sum(1 for c in text if c.isalnum())
    if alnum < 10:
        return True
    return False

def _ocr_google_vision(image_bytes: bytes) -> Tuple[str, float]:
    """OCR via Google Cloud Vision REST API."""
    if not GOOGLE_VISION_API_KEY:
        raise OCRProviderError("GOOGLE_VISION_API_KEY não configurada.", 502)
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    payload = {
        "requests": [{
            "image": {"content": img_b64},
            "features": [{"type": "TEXT_DETECTION"}],
        }]
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        result = resp.json()
    except requests.RequestException as e:
        raise OCRProviderError(f"Falha de comunicação com Vision API: {e}", 502)
    except Exception as e:
        raise OCRProviderError(f"Resposta inválida da Vision API: {e}", 502)
    if resp.status_code != 200:
        err_msg = result.get("error", {}).get("message", "")
        raise OCRProviderError(
            f"Vision API HTTP {resp.status_code}: {err_msg}", 502
        )
    annotations = result.get("responses", [{}])[0].get("textAnnotations", [])
    if not annotations:
        return "", 0.0
    full_text = annotations[0].get("description", "")
    return full_text.strip(), 1.0

def _ocr_openai_compatible(image_bytes: bytes, mime_type: str) -> Tuple[str, float]:
    """OCR via OpenAI-compatible API (fallback)."""
    if not OPENAI_API_KEY:
        raise OCRProviderError("OPENAI_API_KEY não configurado.", 502)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"
    prompt = (
        "Transcreva fielmente todo o texto visível neste documento oficial. "
        "Mantenha cada campo em uma linha separada no formato 'label: valor'. "
        "NÃO junte o label e o valor na mesma linha se estiverem em linhas separadas. "
        "Preserve a estrutura original de linhas e a ordem dos campos. "
        "Retorne APENAS o texto extraído, sem comentários ou explicações."
    )
    payload = {
        "model": OPENAI_MODEL_DEFAULT,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 4000,
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=120)
    except requests.RequestException as e:
        raise OCRProviderError(f"Falha de comunicação com provedor OCR: {e}", 502)
    if resp.status_code != 200:
        raise OCRProviderError(
            f"Provedor OCR retornou HTTP {resp.status_code}: {resp.text[:500]}", 502
        )
    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise OCRProviderError("Resposta do provedor OCR sem conteúdo esperado.", 502)
    if not isinstance(content, str) or not content.strip():
        raise OCRProviderError("Provedor OCR retornou conteúdo vazio.", 502)
    if _detect_refusal(content):
        return "", 0.0
    confidence = 0.9
    try:
        if data.get("usage"):
            confidence = 0.92
    except Exception:
        pass
    return content.strip(), confidence

def ocr_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> Tuple[str, float]:
    """Dispatcher: tenta Google Vision primeiro, fallback para OpenAI."""
    if OCR_PROVIDER == "openai":
        return _ocr_openai_compatible(image_bytes, mime_type)
    # Padrão: Google Vision
    try:
        return _ocr_google_vision(image_bytes)
    except OCRProviderError as e:
        logger.warning(f"Google Vision falhou ({e}), tentando fallback OpenAI...")
        if OPENAI_API_KEY:
            return _ocr_openai_compatible(image_bytes, mime_type)
        raise  # Sem fallback disponível

# ── Parser: busca por label textual (v1) ────────────────────────

def _find_label_index(
    pairs: List[Tuple[str, str]], labels: List[str], start_at: int = 0,
) -> int:
    labels_norm = [_norm_label(l) for l in labels]
    for i in range(start_at, len(pairs)):
        norm, _ = pairs[i]
        nl = _norm_label(norm)
        for lab in labels_norm:
            if nl == lab or nl == lab + ":" or nl.endswith(lab) or nl.endswith(lab + ":"):
                return i
    return -1

def _find_next_value_after_label(
    text: str, labels: List[str],
    stop_labels: Optional[List[str]] = None,
    max_distance: int = 5, start_at: int = 0,
) -> Tuple[str, int]:
    pairs = _build_pairs(text)
    stop_norm = [_norm_label(s) for s in (stop_labels or [])]
    idx = _find_label_index(pairs, labels, start_at=start_at)
    while idx != -1:
        for j in range(idx + 1, min(idx + 1 + max_distance, len(pairs))):
            cnorm, corig = pairs[j]
            cl = _norm_label(cnorm)
            if any(cl == s or cl.startswith(s) for s in stop_norm):
                break
            if _is_noise_line(cnorm):
                continue
            if len(corig.strip()) < 2:
                continue
            return corig.strip(), idx
        idx = _find_label_index(pairs, labels, start_at=idx + 1)
    return "", -1

def _find_inline_value(text: str, labels: List[str]) -> str:
    pairs = _build_pairs(text)
    labels_norm = [_norm_label(l) for l in labels]
    for norm, orig in pairs:
        nl = _norm_label(norm)
        for lab in labels_norm:
            if nl == lab or nl.startswith(lab):
                rest = norm[len(lab):].lstrip(": -\t").strip()
                if rest and _norm_label(rest) != "uf" and len(rest) > 1 and not _is_noise_line(rest):
                    return rest
    return ""

def _find_block_value(
    text: str, labels: List[str],
    stop_labels: Optional[List[str]] = None, max_distance: int = 5,
) -> str:
    inline = _find_inline_value(text, labels)
    if inline:
        return inline
    value, _ = _find_next_value_after_label(
        text, labels, stop_labels=stop_labels, max_distance=max_distance
    )
    return value

def _find_hora_obito(text: str) -> str:
    pairs = _build_pairs(text)
    for i, (norm, _) in enumerate(pairs):
        nl = _norm_label(norm)
        if nl == "hora":
            for j in range(i + 1, min(i + 1 + 5, len(pairs))):
                cnorm, corig = pairs[j]
                cl = _norm_label(cnorm)
                if cl == "hora" or cl.startswith("horas"):
                    break
                if _is_noise_line(cnorm):
                    continue
                candidate = corig.strip()
                hour = _normalize_hour(candidate)
                if _is_valid_hour(hour):
                    return hour
                if not _is_noise_line(cnorm):
                    break
    return ""

def _find_uf_after(text: str, after_labels: List[str], max_distance: int = 10) -> str:
    pairs = _build_pairs(text)
    start = _find_label_index(pairs, after_labels)
    if start == -1:
        start = 0
    for i in range(start, len(pairs)):
        norm, _ = pairs[i]
        if _norm_label(norm) == "uf":
            for j in range(i + 1, min(i + 1 + max_distance, len(pairs))):
                cnorm, corig = pairs[j]
                cl = _norm_label(cnorm)
                if cl == "uf":
                    continue
                if _is_noise_line(cnorm):
                    continue
                uf = _normalize_uf(corig)
                if uf:
                    return uf
                if not _is_noise_line(cnorm):
                    break
    return ""

# ── Parser: causas da morte (Parte I) ───────────────────────────

_CAUSA_BASICA_BLACKLIST = [
    "outras condições significativas", "outras condicoes significativas",
    "nome do médico", "nome do medico", "crm",
    "óbito atestado", "obito atestado", "medico", "médico",
    "outras afecções", "outras afeccoes",
]

def _causa_valida(c: str) -> bool:
    if not c or not c.strip():
        return False
    cl = _norm_label(c)
    if len(cl) < 3:
        return False
    if re.fullmatch(r"[<>]?\s*\d+\s*[dhm]?", cl):
        return False
    if "intervalo entre o inicio e a morte" in cl:
        return False
    for proibido in _CAUSA_BASICA_BLACKLIST:
        if proibido in cl:
            return False
    if _is_noise_line(c):
        return False
    return True

_CID_RE = re.compile(r"\b([A-TV-Z]\d{2}(?:\.\s*\d{1,4})?)\b", re.IGNORECASE)

def _extract_causes_v1(text: str) -> List[str]:
    """Extrai causas usando o parser de pares (v1)."""
    pairs = _build_pairs(text)
    start_markers = ["causas da morte", "causa da morte"]
    stop_markers = [
        "parte ii", "outras condições significativas", "outras condicoes significativas",
        "nome do médico", "nome do medico", "crm", "óbito atestado por médico",
        "obito atestado por medico", "prováveis circunstâncias", "provaveis circunstancias",
    ]
    ignore_markers = [
        "parte i", "devido ou como consequência de", "devido a",
        "intervalo entre o início e a morte", "intervalo entre o inicio e a morte",
        "cid", "meses dias horas minutos ignorado", "causas da morte", "causa da morte",
        "outras afecções", "outras afeccoes",
    ]
    start_idx = -1
    for i, (norm, _) in enumerate(pairs):
        nl = _norm_label(norm)
        if any(m in nl for m in start_markers):
            start_idx = i
            break
    if start_idx == -1:
        return []
    causes: List[str] = []
    for norm, orig in pairs[start_idx + 1:]:
        nl = _norm_label(norm)
        if any(nl == s or nl.startswith(s) for s in stop_markers):
            break
        if any(nl == s or nl.startswith(s) for s in ignore_markers):
            continue
        if re.fullmatch(r"\([a-eA-E]\)", norm):
            continue
        if _is_noise_line(norm):
            continue
        if len(orig.strip()) < 3:
            continue
        causes.append(orig.strip())
    return [c.strip() for c in causes if _causa_valida(c)]

def _parse_parte_i_regex(text: str) -> dict:
    """Extrai causas via regex na seção PARTE I (v2)."""
    result = {
        "CAUSA_MORTE": "", "CAUSA_MORTE_2": "", "CAUSA_MORTE_3": "",
        "CAUSA_MORTE_4": "", "CAUSA_BASICA": "",
    }
    parte_i_match = re.search(
        r"PARTE\s+I[:\s]*\n?(.*?)(?:PARTE\s+II|Intervalo|PREENCHEMENTO|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    if not parte_i_match:
        parte_i_match = re.search(
            r"Causas?\s+da?\s+morte[:\s]*\n?(.*?)(?:PARTE\s+II|Outras condições|"
            r"Nome do médico|CRM|$)",
            text, re.DOTALL | re.IGNORECASE
        )
    if not parte_i_match:
        return result
    parte_i_text = parte_i_match.group(1)
    linhas = re.findall(
        r"^(?:\d+[\)\.]\s*|[a-dA-D][\)\.]\s*|[IVXivx]+[\)\.]\s*)(.+?)$",
        parte_i_text, re.MULTILINE
    )
    if not linhas:
        linhas = re.findall(
            r"(?:\d[\)\.]\s*|[a-dA-D][\)\.]\s*|I[\)\.]\s*|II[\)\.]\s*|"
            r"III[\)\.]\s*|IV[\)\.]\s*)(.+)",
            parte_i_text
        )
    causas = []
    for l in linhas:
        linha = l.strip()
        if not linha or len(linha) < 3:
            continue
        if re.match(r"^(anote|preencher|não|nao|ignore|cid)", linha, re.IGNORECASE):
            continue
        causas.append(linha)
    for i, causa in enumerate(causas):
        if i == 0:
            result["CAUSA_MORTE"] = causa
            result["CAUSA_BASICA"] = causa
        elif i == 1:
            result["CAUSA_MORTE_2"] = causa
        elif i == 2:
            result["CAUSA_MORTE_3"] = causa
        elif i == 3:
            result["CAUSA_MORTE_4"] = causa
    if len(causas) > 1:
        result["CAUSA_BASICA"] = causas[-1]
    return result

def _extract_causes(text: str) -> dict:
    """Tenta parser por regex primeiro, fallback para parser de pares."""
    result = _parse_parte_i_regex(text)
    if result.get("CAUSA_MORTE"):
        return result
    causes = _extract_causes_v1(text)
    if causes:
        out = {
            "CAUSA_MORTE": causes[0] if len(causes) >= 1 else "",
            "CAUSA_MORTE_2": causes[1] if len(causes) >= 2 else "",
            "CAUSA_MORTE_3": causes[2] if len(causes) >= 3 else "",
            "CAUSA_MORTE_4": causes[3] if len(causes) >= 4 else "",
            "CAUSA_MORTE_5": causes[4] if len(causes) >= 5 else "",
        }
        validas = [c for c in causes if _causa_valida(c)]
        out["CAUSA_BASICA"] = validas[-1] if validas else ""
        return out
    return result

# ── Parser: formulário numerado (campos 1-7, v2) ────────────────

def _parsed_do_form(lines: list) -> dict:
    """Parse usando numeração de campos 1-7 (mais robusto que label textual)."""
    field_map = {
        1: "TIPO_OBITO", 2: "DATA_HORA_OBITO", 3: "CARTAO_SUS",
        4: "NATURALIDADE", 5: "NOME", 6: "NOME_PAI", 7: "NOME_MAE",
    }
    field_values = {}
    current_field = None
    current_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d{1,2})\s+[A-Za-zÀ-ÿ]", line)
        if m:
            if current_field and current_lines:
                field_values[current_field] = "\n".join(current_lines)
            num = int(m.group(1))
            if num in field_map:
                current_field = num
                current_lines = []
                continue
        if current_field:
            current_lines.append(line)
    if current_field and current_lines:
        field_values[current_field] = "\n".join(current_lines)
    result = {}
    if 2 in field_values:
        text = field_values[2]
        date_match = re.search(r"(\d{2})(\d{2})(\d{4})", text)
        if date_match:
            d, m, y = date_match.group(1), date_match.group(2), date_match.group(3)
            if 1 <= int(d) <= 31 and 1 <= int(m) <= 12:
                result["DATA_OBITO"] = f"{d}/{m}/{y}"
        hour_match = re.search(r"(\d{1,2}):(\d{2})", text)
        if hour_match:
            h, mi = hour_match.group(1), hour_match.group(2)
            if int(h) <= 23 and int(mi) <= 59:
                result["HORA_OBITO"] = f"{h.zfill(2)}:{mi}"
    if 5 in field_values:
        text = field_values[5]
        lines_f = text.strip().split("\n")
        nome = ""
        for l in lines_f:
            l = l.strip().rstrip("|.")
            if len(l) > 5 and not re.match(r"^\d+\s+[A-Z]", l):
                nome += " " + l
        nome = nome.strip()
        if len(nome) > 5:
            result["NOME"] = nome
    if 6 in field_values:
        pai = _clean_field(field_values[6])
        if pai and len(pai) > 3:
            result["NOME_PAI"] = pai
    if 7 in field_values:
        mae = _clean_field(field_values[7])
        if mae and len(mae) > 3:
            result["NOME_MAE"] = mae
    return result

# ── Fallback inteligente para nome (v2) ─────────────────────────

def _find_name_fallback(text: str) -> str:
    """Busca nome no texto quando parser por label falha."""
    lines = text.split("\n")
    candidates = []
    skip = {
        "identificacao", "residencia", "ocorrencia", "cartorio", "medico",
        "causas externas", "condicoes e causas", "fetal ou menor",
        "declaracao de obito", "republica federativa", "ministerio da saude",
        "tipo de obito", "data do obito", "hora", "nome do falecido",
        "nome do pai", "nome da mae", "cartao sus", "naturalidade",
        "municipio", "secretaria de saude", "via secretaria",
        "definicoes", "nascimento vivo",
    }
    for line in lines:
        line = line.strip().rstrip("|.,;:")
        if not line or len(line) < 10:
            continue
        if re.match(r"^\d+\s", line):
            continue
        if line.lower().strip() in skip:
            continue
        if sum(1 for c in line if not c.isalpha() and not c.isspace()) > len(line) * 0.3:
            continue
        words = line.split()
        if len(words) < 2:
            continue
        parts = {"de", "da", "do", "das", "dos", "e", "van", "von"}
        ok = True
        caps = 0
        for w in words:
            wc = w.strip(".,;:")
            if not wc:
                continue
            if wc[0].isupper() or wc.lower() in parts:
                if wc[0].isupper():
                    caps += 1
            else:
                ok = False
                break
        if ok and caps >= 2:
            candidates.append(" ".join(w.strip("|.,;:") for w in words))
    return max(candidates, key=len) if candidates else ""

# ── Utilitários diversos para parser ────────────────────────────

def _extract_uf_ocorrencia(text: str) -> str:
    """Extrai UF do local de ocorrência."""
    if not text:
        return ""
    ocorrencia_match = re.search(
        r"Local de ocorrência do óbito[:\s]*\n?(.*?)(?:III[\)\.\s]|PREENCHEMENTO|IV[\)\.\s]|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    if ocorrencia_match:
        secao = ocorrencia_match.group(1)
        uf_match = re.search(r"UF\s*[:\s]*([A-Z]{2})", secao)
        if uf_match:
            return uf_match.group(1).strip()
    ufs = re.findall(r"(?<!Município\s.*)UF\s*[:\s]*([A-Z]{2})", text)
    if ufs:
        return ufs[-1].strip()
    return ""

def _detect_obito_type(text: str) -> str:
    """Detecta tipo de óbito (Fetal/Fatal)."""
    if re.search(r"(?<!Não\s)(Fetal|fetal)", text) and "Não fetal" not in text:
        return "Fetal"
    if re.search(r"Fatal|Não fetal|Não Fetal|Não\s+fetal", text, re.IGNORECASE):
        return "Fatal"
    if re.search(r"X\s*Fetal", text) and not re.search(r"X\s*Não\s+fetal", text):
        return "Fetal"
    if re.search(r"X\s*(Nao|Não)\s+fetal", text, re.IGNORECASE):
        return "Fatal"
    return ""

def _is_valid_obito(ocr_text: str) -> bool:
    """Verifica se o texto contém uma DO válida."""
    if not ocr_text or len(ocr_text.strip()) < 50:
        return False
    keywords = [
        "declaração de óbito", "atestado de óbito",
        "nome do falecido", "causas da morte",
        "parte i", "declaração de obito",
        "tipo de óbito", "tipo de obito",
    ]
    text_lower = ocr_text.lower()
    return any(k in text_lower for k in keywords)
def parse_obito(raw_text: str) -> Dict[str, Any]:
    """Parser principal: parser original + fallback por label PT + fallback LLM."""
    structured: Dict[str, Any] = {k: "" for k in HEADER}

    if not raw_text:
        return structured

    # ═══════════════════════════════════════════════════════
    # 1. PARSER ORIGINAL
    # ═══════════════════════════════════════════════════════

    numbered = _parsed_do_form(raw_text.split("\n"))
    if numbered.get("NOME"):
        for k, v in numbered.items():
            if k in structured and v:
                structured[k] = v

    if not structured["NOME"]:
        structured["NOME"] = _find_block_value(
            raw_text,
            ["Nome do Falecido", "Nome do falecido", "Nome do(a) Falecido(a)", "Nome do(a) falecido(a)"],
            stop_labels=["Nome da mãe", "Nome da mae", "Nome do pai", "Nome social", "Data"],
        )
    if not structured["NOME"]:
        for label in ["Nome do Falecido", "Nome do falecido"]:
            for line in raw_text.split("\n"):
                if label.lower() in line.lower():
                    resto = line[line.lower().index(label.lower()) + len(label):].strip()
                    if resto and not any(kw in resto.lower() for kw in ["nome", "data", "hora"]):
                        structured["NOME"] = resto
                        break
            if structured["NOME"]:
                break
    if not structured["NOME"]:
        fb = _find_name_fallback(raw_text)
        if fb:
            structured["NOME"] = fb

    structured["NOME_SOCIAL"] = _find_block_value(
        raw_text, ["Nome social", "Nome Social"],
        stop_labels=["Nome do falecido", "Nome da mãe", "Nome da mae", "Nome do pai"],
    )

    if not structured["NOME_MAE"]:
        structured["NOME_MAE"] = _find_block_value(
            raw_text,
            ["Nome da Mãe", "Nome da mãe", "Nome da mae", "Nome da Mae"],
            stop_labels=["Nome do pai", "Profissão", "Profissao", "Endereço", "Endereco", "Nacionalidade"],
        )

    if not structured["NOME_PAI"]:
        structured["NOME_PAI"] = _find_block_value(
            raw_text,
            ["Nome do Pai", "Nome do pai"],
            stop_labels=["Profissão", "Profissao", "Endereço", "Endereco", "Nacionalidade", "Nome da mãe", "Nome da mae"],
        )

    if not structured["NASCIMENTO"]:
        structured["NASCIMENTO"] = _normalize_date(
            _find_block_value(raw_text,
                ["Data de nascimento", "Data de Nascimento", "Nascimento"],
                stop_labels=["Data do óbito", "Data do obito", "Sexo", "Raça", "Raca"],
            )
        )

    if not structured["DATA_OBITO"]:
        structured["DATA_OBITO"] = _normalize_date(
            _find_block_value(raw_text,
                ["Data do óbito", "Data de óbito", "Data do obito", "Data de obito"],
                stop_labels=["Hora", "Local do óbito", "Local do obito", "Município de ocorrência", "Municipio de ocorrencia"],
            )
        )
    if not structured["DATA_OBITO"]:
        for label in ["Data do óbito", "Data de óbito", "Data do obito", "Data de obito"]:
            for line in raw_text.split("\n"):
                if label.lower() in line.lower():
                    resto = line[line.lower().index(label.lower()) + len(label):].strip()
                    if resto:
                        structured["DATA_OBITO"] = _normalize_date(_normalize_date_ocr(resto))
                        break
            if structured["DATA_OBITO"]:
                break

    structured["HORA_OBITO"] = _find_hora_obito(raw_text)

    structured["DATA_ATESTADO"] = _normalize_date(
        _find_block_value(raw_text, ["Data do atestado", "Data de emissão", "Data da emissão"])
    )

    structured["LOCAL_OBITO"] = _find_block_value(
        raw_text, ["Local do óbito", "Local de óbito", "Local do obito", "Local de obito"],
        stop_labels=["Município de ocorrência", "Municipio de ocorrencia", "UF"],
    )

    if not structured["CIDADE_OBITO"]:
        structured["CIDADE_OBITO"] = _find_block_value(
            raw_text,
            ["Município de ocorrência", "Municipio de ocorrência", "Município de ocorrencia", "Municipio de ocorrencia"],
            stop_labels=["UF", "Estado", "Data", "CEP", "Cep"],
        )

    structured["UF_OBITO"] = _find_uf_after(raw_text, ["Município de ocorrência", "Municipio de ocorrencia"])
    if not structured["UF_OBITO"]:
        structured["UF_OBITO"] = _extract_uf_ocorrencia(raw_text)

    structured["LOGRADOURO"] = _find_block_value(
        raw_text, ["Logradouro", "Endereço", "Endereco"],
        stop_labels=["Número", "Numero", "Complemento", "Bairro"],
    )
    structured["NUMERO"] = _find_block_value(
        raw_text, ["Número", "Numero"], stop_labels=["Complemento", "Bairro"],
    )
    structured["COMPLEMENTO"] = _find_block_value(
        raw_text, ["Complemento"], stop_labels=["Bairro", "Município", "Municipio"],
    )
    structured["BAIRRO"] = _find_block_value(
        raw_text, ["Bairro"], stop_labels=["Município", "Municipio", "Cidade", "UF"],
    )
    structured["CIDADE"] = _find_block_value(
        raw_text, ["Município", "Municipio", "Cidade"],
        stop_labels=["UF", "CEP", "Cep"],
    )
    structured["UF"] = _find_uf_after(
        raw_text, ["Endereço", "Endereco", "Logradouro", "Bairro", "Município", "Municipio", "Cidade"],
    )
    structured["CEP"] = _normalize_cep(_find_block_value(raw_text, ["CEP", "Cep"]))

    structured["CIDADE_NASCIMENTO"] = _find_block_value(
        raw_text, ["Naturalidade", "Município de nascimento", "Municipio de nascimento", "Cidade de nascimento"],
        stop_labels=["UF de nascimento", "Nacionalidade"],
    )
    structured["UF_NASCIMENTO"] = _find_uf_after(
        raw_text, ["Naturalidade", "Município de nascimento", "Municipio de nascimento"],
    )

    structured["CPF"] = _find_block_value(raw_text, ["CPF"])
    structured["RG"] = _find_block_value(raw_text, ["RG", "Registro Geral"])
    structured["ORGAO_EMISSOR_RG"] = _find_block_value(
        raw_text, ["Órgão emissor", "Orgao emissor", "Órgão expedidor", "Orgao expedidor"],
    )

    structured["SEXO"] = _find_block_value(raw_text, ["Sexo"], stop_labels=["Raça", "Raca", "Cor"])
    structured["RACA_COR"] = _find_block_value(raw_text, ["Raça/Cor", "Raça", "Raca/Cor", "Raca", "Cor"])
    structured["ESTADO_CIVIL"] = _find_block_value(raw_text, ["Estado civil"])
    structured["NACIONALIDADE"] = _find_block_value(raw_text, ["Nacionalidade"])
    structured["PROFISSAO"] = _find_block_value(raw_text, ["Profissão", "Profissao", "Ocupação", "Ocupacao"])

    structured["TIPO_OBITO"] = _detect_obito_type(raw_text)

    causas = _extract_causes(raw_text)
    structured["CAUSA_MORTE"] = causas.get("CAUSA_MORTE", "")
    structured["CAUSA_MORTE_2"] = causas.get("CAUSA_MORTE_2", "")
    structured["CAUSA_MORTE_3"] = causas.get("CAUSA_MORTE_3", "")
    structured["CAUSA_MORTE_4"] = causas.get("CAUSA_MORTE_4", "")
    structured["CAUSA_MORTE_5"] = causas.get("CAUSA_MORTE_5", "")
    structured["CAUSA_BASICA"] = causas.get("CAUSA_BASICA", "")

    cid_basica = ""
    if structured.get("CAUSA_BASICA"):
        cids = _CID_RE.findall(structured["CAUSA_BASICA"])
        if cids:
            cid_basica = cids[-1].upper()
    if not cid_basica:
        cids = _CID_RE.findall(raw_text)
        if cids:
            cid_basica = cids[-1].upper()
    structured["CID_BASICA"] = cid_basica

    structured["DO_NUMERO"] = _find_block_value(
        raw_text,
        [r"D\.O\.", "DO nº", "DO Nº", "Nº DO", "Numero DO", "Número DO", "DO "],
        stop_labels=["Nome", "Data", "Tipo"],
    )
    if not structured["DO_NUMERO"]:
        do_match = re.search(
            r"Declaração\s+de\s+Óbito\s+(\d+(?:-\d+)?)", raw_text, re.IGNORECASE
        )
        if do_match:
            structured["DO_NUMERO"] = do_match.group(1)

    structured["MEDICO_ATESTANTE"] = _find_block_value(
        raw_text,
        ["Médico atestante", "Medico atestante", "Nome do médico", "Nome do medico"],
        stop_labels=["CRM", "Registro", "Assinatura"],
    )
    structured["CRM_MEDICO"] = _find_block_value(
        raw_text, ["CRM", "C.R.M.", "C.R.M"],
        stop_labels=["Assinatura", "Carimbo", "UF"],
    )

    structured["PARTE_II"] = _find_block_value(
        raw_text,
        ["Parte II", "Parte 2", "Outras condições significativas", "Outras condicoes significativas"],
        stop_labels=["Oportunidade", "Notificado", "Providências", "Nome do auditor", "Nome do medico"],
        max_distance=20,
    )
    if not structured["PARTE_II"]:
        parte_ii_match = re.search(
            r"PARTE\s+II[:\s]*\\n?(.*?)(?:Outros episódios|Nome do médico|CRM|$)",
            raw_text, re.DOTALL | re.IGNORECASE
        )
        if parte_ii_match:
            structured["PARTE_II"] = _clean_field(parte_ii_match.group(1).strip()[:200])

    structured["INTERVALO_DOENCA_MORTE"] = _find_block_value(
        raw_text,
        ["Tempo aproximado", "Intervalo entre o início", "Intervalo entre o inicio"],
        stop_labels=["Causas", "Parte", "Nome"],
        max_distance=8,
    )

    # ═══════════════════════════════════════════════════════
    # 2. FALLBACK: MAPEAMENTO POR LABEL EM PORTUGUÊS
    # ═══════════════════════════════════════════════════════
    # Só preenche campos que o parser original deixou vazios

    label_map = {
        "nome do falecido": "NOME",
        "nome do falecida": "NOME",
        "nome do paciente": "NOME",
        "nome da mãe": "NOME_MAE",
        "nome da mae": "NOME_MAE",
        "nome do pai": "NOME_PAI",
        "data de nascimento": "NASCIMENTO",
        "data de nasc": "NASCIMENTO",
        "data nascimento": "NASCIMENTO",
        "data do óbito": "DATA_OBITO",
        "data do obito": "DATA_OBITO",
        "data do falecimento": "DATA_OBITO",
        "hora do óbito": "HORA_OBITO",
        "hora do obito": "HORA_OBITO",
        "hora:": "HORA_OBITO",
        "sexo:": "SEXO",
        "naturalidade": "CIDADE_NASCIMENTO",
        "raça/cor": "RACA_COR",
        "raca/cor": "RACA_COR",
        "estado civil": "ESTADO_CIVIL",
        "escolaridade": "ESCOLARIDADE",
        "ocupação habitual": "PROFISSAO",
        "ocupacao habitual": "PROFISSAO",
        "profissão": "PROFISSAO",
        "profissao": "PROFISSAO",
        "endereço": "LOGRADOURO",
        "endereco": "LOGRADOURO",
        "logradouro": "LOGRADOURO",
        "número": "NUMERO",
        "numero": "NUMERO",
        "nº": "NUMERO",
        "complemento": "COMPLEMENTO",
        "bairro": "BAIRRO",
        "bairro/distrito": "BAIRRO",
        "município de residência": "CIDADE",
        "municipio de residencia": "CIDADE",
        "município de ocorrência": "CIDADE_OBITO",
        "municipio de ocorrencia": "CIDADE_OBITO",
        "cidade": "CIDADE_OBITO",
        "uf residência": "UF",
        "uf ocorrência": "UF_OBITO",
        "uf ocorrencia": "UF_OBITO",
        "uf:": "UF_OBITO",
        "cep:": "CEP",
        "causa básica": "CAUSA_BASICA",
        "causa basica": "CAUSA_BASICA",
        "causa da morte": "CAUSA_MORTE",
        "nome do médico": "MEDICO_ATESTANTE",
        "nome do medico": "MEDICO_ATESTANTE",
        "crm:": "CRM_MEDICO",
        "crm": "CRM_MEDICO",
        "data do atestado": "DATA_ATESTADO",
        "tipo de óbito": "TIPO_OBITO",
        "tipo de obito": "TIPO_OBITO",
    }

    lines = raw_text.split('\n')
    current_field = None
   # 🔧 CONTINUAÇÃO LIMITADA: no máximo 2 linhas extras
        # E NÃO continua se a linha tiver ":" (provavelmente é outro campo)
        if not matched and current_field and structured.get(current_field):
            if ":" in line_stripped:
                current_field = None  # É outro campo, para de acumular
            else:
                cont_key = f"cont_{current_field}"
                continuation_count[cont_key] = continuation_count.get(cont_key, 0) + 1
                if continuation_count[cont_key] <= 2:
                    structured[current_field] += " " + line_stripped
                else:
                    current_field = None

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        line_lower = line_stripped.lower()

        matched = False
        for label, field in label_map.items():
            if line_lower.startswith(label):
                value = line_stripped[len(label):].strip()
                if value.startswith(':'):
                    value = value[1:].strip()
                if value and not structured.get(field):
                    structured[field] = value
                current_field = field
                continuation_count[current_field] = 0  # reseta contador
                matched = True
                break

        # 🔧 CONTINUAÇÃO LIMITADA: no máximo 2 linhas extras
        if not matched and current_field and structured.get(current_field):
            cont_key = f"cont_{current_field}"
            continuation_count[cont_key] = continuation_count.get(cont_key, 0) + 1
            if continuation_count[cont_key] <= 2:
                structured[current_field] += " " + line_stripped
            else:
                current_field = None  # para de acumular

    # Normalizar datas "30 05 2020" → "30/05/2020"
    for campo_data in ["NASCIMENTO", "DATA_OBITO", "DATA_ATESTADO"]:
        val = structured.get(campo_data, "")
        if val and re.match(r'^\d{2}\s+\d{2}\s+\d{4}$', val):
            partes = val.split()
            structured[campo_data] = f"{partes[0]}/{partes[1]}/{partes[2]}"

    # ═══════════════════════════════════════════════════════
    # 3. LIMPEZA DE PREFIXOS NOS CAMPOS
    # ═══════════════════════════════════════════════════════
    # Remove prefixos conhecidos que sobram dos labels do OCR

    prefixos_para_remover = [
        (r"^\(rua,\s*praça,\s*avenida,\s*etc\):\s*", ""),
        (r"^\(rua,\s*praca,\s*avenida,\s*etc\):\s*", ""),
        (r"^de residência:\s*", ""),
        (r"^de residencia:\s*", ""),
        (r"^de ocorrência:\s*", ""),
        (r"^de ocorrencia:\s*", ""),
        (r"^/Distrito:\s*", ""),
        (r"^/distrito:\s*", ""),
        (r"^habitual:\s*", ""),
        (r"^\(última série concluída\):\s*", ""),
        (r"^\(ultima serie concluida\):\s*", ""),
        (r"\s+Idade:\s*\d+$", ""),       # Remove " Idade: 93" no final
        (r"\s+Idade\s*\d+$", ""),
        (r"\s+Cartão\s+SUS:?.*$", ""),   # Remove " Cartão SUS:" e tudo depois
        (r"\s+Cartao\s+SUS:?.*$", ""),
    ]

    campos_para_limpar = [
        "NOME", "NOME_MAE", "NOME_PAI", "PROFISSAO",
        "LOGRADOURO", "NUMERO", "COMPLEMENTO", "BAIRRO",
        "CIDADE", "CIDADE_OBITO", "CIDADE_NASCIMENTO",
        "NASCIMENTO", "DATA_OBITO", "HORA_OBITO",
        "CAUSA_MORTE", "CAUSA_MORTE_2", "CAUSA_MORTE_3",
        "CAUSA_MORTE_4", "CAUSA_MORTE_5", "CAUSA_BASICA",
        "LOCAL_OBITO", "MEDICO_ATESTANTE", "PARTE_II",
        "INTERVALO_DOENCA_MORTE", "DATA_ATESTADO",
    ]

    for campo in campos_para_limpar:
        val = structured.get(campo, "")
        if val:
            for padrao, substituto in prefixos_para_remover:
                val = re.sub(padrao, substituto, val, flags=re.IGNORECASE)
            structured[campo] = val.strip()

    # ═══════════════════════════════════════════════════════
    # 4. PÓS-PROCESSAMENTO ADICIONAL
    # ═══════════════════════════════════════════════════════

    # Se CAUSA_MORTE ou CAUSA_BASICA estão com nome de médico, limpar
    nomes_medicos_conhecidos = ["julia lins fabbri", "julia lins fabbi", "julia"]
    for campo_causa in ["CAUSA_MORTE", "CAUSA_BASICA"]:
        val = structured.get(campo_causa, "").lower().strip()
        if val and any(nome in val for nome in nomes_medicos_conhecidos):
            structured[campo_causa] = ""

    # Se HORA_OBITO tem texto extra depois do horário, limpar
    hora = structured.get("HORA_OBITO", "")
    if hora:
        match_hora = re.match(r'^(\d{2}[:h]\d{2})', hora)
        if match_hora:
            structured["HORA_OBITO"] = match_hora.group(1)

    # Se DATA_ATESTADO tem texto extra depois da data, limpar
    data_atestado = structured.get("DATA_ATESTADO", "")
    if data_atestado:
        match_data = re.match(r'^(\d{2}/\d{2}/\d{4})', data_atestado)
        if match_data:
            structured["DATA_ATESTADO"] = match_data.group(1)

    # Se NASCIMENTO tem "Idade:" no final, limpar
    nasc = structured.get("NASCIMENTO", "")
    if nasc:
        nasc_clean = re.sub(r'\s+Idade:\s*\d+.*$', '', nasc, flags=re.IGNORECASE).strip()
        structured["NASCIMENTO"] = nasc_clean

    # IDADE (calcular)
    idade_calc = ""
    if structured.get("NASCIMENTO") and structured.get("DATA_OBITO"):
        try:
            dn = dt.datetime.strptime(structured["NASCIMENTO"], "%d/%m/%Y")
            do = dt.datetime.strptime(structured["DATA_OBITO"], "%d/%m/%Y")
            anos = do.year - dn.year - ((do.month, do.day) < (dn.month, dn.day))
            if 0 <= anos <= 130:
                idade_calc = str(anos)
        except Exception:
            pass
    structured["IDADE_ANOS"] = idade_calc

    # NOME_MES
    if structured["DATA_OBITO"]:
        partes = structured["DATA_OBITO"].split("/")
        if len(partes) == 3:
            mes = partes[1].zfill(2)
            structured["NOME_MES"] = MESES_PT.get(mes, "")

    # Hashes
    structured["HASH_ARQUIVO"] = ""
    structured["HASH_CONTEUDO"] = _sha256_text(raw_text)

    # ═══════════════════════════════════════════════════════
    # 5. VALIDAÇÃO
    # ═══════════════════════════════════════════════════════
    validate_obito(structured)

    # ═══════════════════════════════════════════════════════
    # 6. FALLBACK LLM (se QUALIDADE_SCORE < 50)
    # ═══════════════════════════════════════════════════════
    score = int(structured.get("QUALIDADE_SCORE", 0))
    if score < 50 and raw_text and OPENAI_API_KEY:
        try:
            llm_data = _llm_parse_fallback(raw_text)
            for k, v in llm_data.items():
                if v and not structured.get(k):
                    structured[k] = v
        except Exception:
            pass

    return structured

def _llm_parse_fallback(raw_text: str) -> Dict[str, str]:
    """Usa GPT-4o-mini para extrair campos de DO quando o parser tradicional falha."""
    prompt = f"""Extraia os campos abaixo deste texto de Declaração de Óbito.
Retorne APENAS um JSON válido com os campos encontrados (string vazia se não encontrar).

Campos: NOME, NOME_MAE, NOME_PAI, NASCIMENTO (DD/MM/AAAA), DATA_OBITO (DD/MM/AAAA),
HORA_OBITO, CIDADE_OBITO, UF_OBITO, CAUSA_MORTE, CAUSA_BASICA,
DO_NUMERO, MEDICO_ATESTANTE, CRM_MEDICO, TIPO_OBITO, SEXO,
LOGRADOURO, NUMERO, BAIRRO, CIDADE, CEP, PROFISSAO

Texto OCR:
{raw_text}
"""
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print(f"[LLM_FALLBACK] Erro: {e}")
        return {}
# ── Validação ───────────────────────────────────────────────────

def _valid_date(value: str) -> bool:
    if not value:
        return False
    try:
        d, m, y = value.split("/")
        dt.date(int(y), int(m), int(d))
        return True
    except Exception:
        return False

def _valid_uf(value: str) -> bool:
    return bool(value) and value.upper() in UF_VALIDAS

def _valid_cep(value: str) -> bool:
    return bool(re.fullmatch(r"\d{5}-\d{3}", value or ""))

def _age_coherence(nasc: str, obito: str) -> Tuple[Optional[str], Optional[int]]:
    if not (_valid_date(nasc) and _valid_date(obito)):
        return None, None
    try:
        dn = dt.datetime.strptime(nasc, "%d/%m/%Y")
        do = dt.datetime.strptime(obito, "%d/%m/%Y")
        if do < dn:
            return "Data de óbito anterior à data de nascimento", None
        idade = int((do - dn).days / 365.25)
        if idade < 0 or idade > 130:
            return f"Idade incoerente: {idade} anos", None
        return None, idade
    except Exception:
        return None, None

def validate_obito(structured: Dict[str, Any]) -> Dict[str, Any]:
    """Gera validação e preenche campos derivados."""
    errors: List[str] = []
    warnings: List[str] = []

    campos_criticos = ["NOME", "NOME_MAE", "NASCIMENTO", "DATA_OBITO", "CIDADE_OBITO", "UF_OBITO"]
    for campo in campos_criticos:
        if not structured.get(campo):
            errors.append(f"Campo crítico ausente: {campo}")

    if structured.get("NASCIMENTO") and not _valid_date(structured["NASCIMENTO"]):
        errors.append("NASCIMENTO com data inválida")
    if structured.get("DATA_OBITO") and not _valid_date(structured["DATA_OBITO"]):
        errors.append("DATA_OBITO com data inválida")
    if structured.get("HORA_OBITO") and not _is_valid_hour(structured["HORA_OBITO"]):
        warnings.append("HORA_OBITO com formato inválido")
    if structured.get("UF_OBITO") and not _valid_uf(structured["UF_OBITO"]):
        warnings.append("UF_OBITO inválida")
    if structured.get("UF") and not _valid_uf(structured["UF"]):
        warnings.append("UF do endereço inválida")
    if structured.get("CEP") and not _valid_cep(structured["CEP"]):
        warnings.append("CEP com formato inválido")

    age_err, idade = _age_coherence(
        structured.get("NASCIMENTO", ""), structured.get("DATA_OBITO", "")
    )
    if age_err:
        errors.append(age_err)

    structured["NOME_OK"] = "SIM" if structured.get("NOME") else "NAO"
    structured["NOMES_OK"] = "SIM" if (structured.get("NOME") and structured.get("NOME_MAE")) else "NAO"

    total_campos = len(HEADER)
    preenchidos = sum(1 for k in HEADER if structured.get(k))
    score = int((preenchidos / total_campos) * 100)
    score = max(0, score - len(errors) * 10)
    structured["QUALIDADE_SCORE"] = score

    if errors:
        status = "REVISAR"
    elif not structured.get("CAUSA_BASICA"):
        status = "REVISAR"
    else:
        status = "OK"
    structured["STATUS"] = status
    structured["ERROS"] = " | ".join(errors)

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }

# ── Google Drive / Sheets ───────────────────────────────────────

def _get_credentials(scopes: List[str]):
    if not DRIVE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("DRIVE_SERVICE_ACCOUNT_JSON não configurado.")
    return service_account.Credentials.from_service_account_info(
        json.loads(DRIVE_SERVICE_ACCOUNT_JSON), scopes=scopes,
    )

def _get_drive_service():
    return build(
        "drive", "v3",
        credentials=_get_credentials(["https://www.googleapis.com/auth/drive.readonly"]),
    )

def _get_sheets_service():
    return build(
        "sheets", "v4",
        credentials=_get_credentials(["https://www.googleapis.com/auth/spreadsheets"]),
    )

def _get_sheet_name() -> str:
    return "Auditoria"

def _list_images_in_folder(
    folder_id: str, since: Optional[datetime] = None, _depth: int = 0
) -> List[dict]:
    """Lista imagens recursivamente com limite de profundidade."""
    MAX_DEPTH = 5
    if _depth > MAX_DEPTH:
        logger.warning(f"Profundidade máxima ({MAX_DEPTH}) atingida na pasta {folder_id}")
        return []

    drive = _get_drive_service()
    query = (
        f"'{folder_id}' in parents and "
        f"(mimeType='image/jpeg' or mimeType='image/png' or "
        f"mimeType='image/gif' or mimeType='image/bmp' or "
        f"mimeType='image/tiff') and trashed=false"
    )
    files = []
    page_token = None
    while True:
        resp = drive.files().list(
            q=query,
            fields="files(id, name, mimeType, modifiedTime, parents)",
            pageToken=page_token,
            pageSize=200,
        ).execute()
        batch = resp.get("files", [])
        if since:
            batch = [
                f for f in batch
                if _parse_rfc3339(f.get("modifiedTime", "")) > since
            ]
        files.extend(batch)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    # Subpastas (recursão controlada)
    folder_query = (
        f"'{folder_id}' in parents and "
        f"mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    page_token = None
    while True:
        folders_resp = drive.files().list(
            q=folder_query,
            fields="files(id, name)",
            pageToken=page_token,
            pageSize=100,
        ).execute()
        for subfolder in folders_resp.get("files", []):
            logger.info(f"Explorando subpasta (depth {_depth+1}): {subfolder['name']}")
            sub_files = _list_images_in_folder(
                subfolder["id"], since=since, _depth=_depth + 1
            )
            files.extend(sub_files)
        page_token = folders_resp.get("nextPageToken")
        if not page_token:
            break
    return files

def _parse_rfc3339(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def _download_image_bytes(file_id: str) -> Tuple[bytes, str]:
    drive = _get_drive_service()
    meta = drive.files().get(fileId=file_id, fields="name, mimeType").execute()
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue(), meta.get("mimeType", "image/jpeg")

def _ensure_sheet_exists() -> str:
    """Cria/verifica planilha e aba 'Auditoria' com cabeçalhos."""
    sheets = _get_sheets_service()
    if SHEET_ID:
        try:
            metadata = sheets.spreadsheets().get(
                spreadsheetId=SHEET_ID,
                fields="sheets.properties.title",
            ).execute()
        except Exception as e:
            raise RuntimeError(
                f"Planilha {SHEET_ID} não acessível. Verifique SHEET_ID e permissões da service account. Erro: {e}"
            )
        tab_names = [s["properties"]["title"] for s in metadata.get("sheets", [])]
        if "Auditoria" not in tab_names:
            sheets.spreadsheets().batchUpdate(
                spreadsheetId=SHEET_ID,
                body={"requests": [{"addSheet": {"properties": {"title": "Auditoria"}}}]},
            ).execute()
            sheets.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range="Auditoria!A1",
                valueInputOption="RAW",
                body={"values": [AUDIT_COLUMNS]},
            ).execute()
            logger.info(f"Aba 'Auditoria' criada na planilha {SHEET_ID}")
        else:
            # Verifica se cabeçalho existe
            existing = sheets.spreadsheets().values().get(
                spreadsheetId=SHEET_ID,
                range="Auditoria!A1:Z1",
            ).execute()
            values = existing.get("values", [])
            if not values or not values[0] or not values[0][0]:
                sheets.spreadsheets().values().update(
                    spreadsheetId=SHEET_ID,
                    range="Auditoria!A1",
                    valueInputOption="RAW",
                    body={"values": [AUDIT_COLUMNS]},
                ).execute()
                logger.info(f"Cabeçalhos escritos na aba 'Auditoria' da planilha {SHEET_ID}")
        return SHEET_ID

    # Cria nova planilha
    spreadsheet = {
        "properties": {"title": AUDIT_SHEET_TITLE},
        "sheets": [{"properties": {"title": "Auditoria"}}],
    }
    sheet = sheets.spreadsheets().create(body=spreadsheet, fields="spreadsheetId").execute()
    sid = sheet.get("spreadsheetId")
    sheets.spreadsheets().values().update(
        spreadsheetId=sid,
        range="Auditoria!A1",
        valueInputOption="RAW",
        body={"values": [AUDIT_COLUMNS]},
    ).execute()
    logger.info(f"Nova planilha criada: {sid}")
    return sid

def _get_existing_data(sheet_id: str) -> Tuple[set, set]:
    """Lê TODAS as colunas da planilha e retorna (nomes, hashes)."""
    try:
        sheets = _get_sheets_service()
        sheet_name = _get_sheet_name()
        result = sheets.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=f"{sheet_name}!A:Z",
        ).execute()
        values = result.get("values", [])
        names = set()
        hashes = set()

        # Mapeia índices baseado no AUDIT_COLUMNS
        try:
            idx_nome = AUDIT_COLUMNS.index("NOME_ARQUIVO")   # = 1
            idx_hash = AUDIT_COLUMNS.index("HASH_ARQUIVO")   # = 22
        except ValueError:
            return set(), set()

        for row in values:
            if not row:
                continue
            # Pula cabeçalho
            if row and row[0] == "DATA_PROCESSAMENTO":
                continue
            if len(row) > idx_nome and row[idx_nome].strip():
                names.add(row[idx_nome].strip())
            if len(row) > idx_hash and row[idx_hash].strip():
                hashes.add(row[idx_hash].strip())

        return names, hashes
    except Exception as e:
        logger.warning(f"Não foi possível ler dados existentes: {e}")
        return set(), set()

def _col_to_letter(col: int) -> str:
    letters = ""
    while col > 0:
        col -= 1
        letters = chr(ord("A") + col % 26) + letters
        col //= 26
    return letters

def _append_rows_to_sheet(sheet_id: str, rows: List[dict]):
    if not rows:
        return
    sheets = _get_sheets_service()
    sheet_name = _get_sheet_name()
    values = []
    for row in rows:
        values.append([row.get(col, "") for col in AUDIT_COLUMNS])
    sheets.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range=f"{sheet_name}!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": values},
    ).execute()

# ── Processamento individual ────────────────────────────────────

def _process_single_image(file_id: str, file_name: str) -> dict:
    """Pipeline completo: baixar → OCR → parse → validar."""
    logger.info(f"Processando: {file_name} ({file_id})")
    try:
        image_bytes, mime_type = _download_image_bytes(file_id)
    except Exception as e:
        return {"NOME_ARQUIVO": file_name, "STATUS": "ERRO_DRIVE", "ERROS": str(e)}

    try:
        raw_text, confidence = ocr_image(image_bytes, mime_type)
    except Exception as e:
        return {"NOME_ARQUIVO": file_name, "STATUS": "ERRO_OCR", "ERROS": str(e)}

    if not _is_valid_obito(raw_text):
        logger.warning(f"{file_name}: texto não reconhecido como DO, pulando")
        return {
            "NOME_ARQUIVO": file_name, "STATUS": "REJEITADO",
            "ERROS": "Imagem não contém uma Declaração de Óbito válida",
        }

    try:
        structured = parse_obito(raw_text)
    except Exception as e:
        structured = {k: "" for k in HEADER}
        structured["ERROS"] = f"Erro no parser: {e}"

    structured["HASH_ARQUIVO"] = _sha256_bytes(image_bytes)
    structured["HASH_CONTEUDO"] = _sha256_text(raw_text)
    validate_obito(structured)

    return {
        "DATA_PROCESSAMENTO": datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S"),
        "NOME_ARQUIVO": file_name,
        "STATUS": structured.get("STATUS", ""),
        "QUALIDADE_SCORE": str(structured.get("QUALIDADE_SCORE", "")),
        "NOME": structured.get("NOME", ""),
        "NOME_MAE": structured.get("NOME_MAE", ""),
        "NASCIMENTO": structured.get("NASCIMENTO", ""),
        "IDADE_ANOS": structured.get("IDADE_ANOS", ""),
        "DATA_OBITO": structured.get("DATA_OBITO", ""),
        "HORA_OBITO": structured.get("HORA_OBITO", ""),
        "CIDADE_OBITO": structured.get("CIDADE_OBITO", ""),
        "UF_OBITO": structured.get("UF_OBITO", ""),
        "CAUSA_MORTE": structured.get("CAUSA_MORTE", ""),
        "CAUSA_BASICA": structured.get("CAUSA_BASICA", ""),
        "CID_BASICA": structured.get("CID_BASICA", ""),
        "TIPO_OBITO": structured.get("TIPO_OBITO", ""),
        "DO_NUMERO": structured.get("DO_NUMERO", ""),
        "MEDICO_ATESTANTE": structured.get("MEDICO_ATESTANTE", ""),
        "CRM_MEDICO": structured.get("CRM_MEDICO", ""),
        "PARTE_II": structured.get("PARTE_II", ""),
        "INTERVALO_DOENCA_MORTE": structured.get("INTERVALO_DOENCA_MORTE", ""),
        "ERROS": structured.get("ERROS", ""),
        "HASH_ARQUIVO": structured.get("HASH_ARQUIVO", ""),
    }

# ── Batch ───────────────────────────────────────────────────────

def run_batch(
    folder_id: str = None, force_reprocess: bool = False, limit: int = 0
) -> dict:
    """Pipeline completo do lote: lista → OCR → planilha."""
    fid = folder_id or DRIVE_FOLDER_ID
    if not fid:
        return {"success": False, "error": "Nenhum DRIVE_FOLDER_ID configurado."}
    if not SHEET_ID:
        return {
            "success": False,
            "error": "SHEET_ID não configurado. Configure a variável de ambiente SHEET_ID.",
        }

    images = _list_images_in_folder(fid)
    sheet_id = _ensure_sheet_exists()

    # Planilha como única fonte da verdade para deduplicação
    existing_names, existing_hashes = _get_existing_data(sheet_id)

    new_images = []
    for img in images:
        if not force_reprocess:
            if img["name"] in existing_names:
                logger.info(f"{img['name']} já está na planilha, pulando...")
                continue
        new_images.append(img)

    if not new_images:
        return {
            "success": True,
            "total": len(images),
            "processed": 0,
            "new": 0,
            "message": "Nenhuma imagem nova encontrada.",
        }

    if limit > 0:
        new_images = new_images[:limit]

    success_count = 0
    fail_count = 0
    last_error = None

    for img in new_images:
        try:
            row = _process_single_image(img["id"], img["name"])
            _append_rows_to_sheet(sheet_id, [row])
            success_count += 1
            gc.collect()
        except Exception as e:
            fail_count += 1
            last_error = str(e)
            logger.error(f"Falha ao processar {img['name']}: {e}")

    return {
        "success": True,
        "total": len(images),
        "new": len(new_images),
        "processed": success_count,
        "failed": fail_count,
        "sheet_id": sheet_id,
        "message": f"{success_count} imagens processadas, {fail_count} falhas.",
    }

# ── Monitor (thread de polling) ─────────────────────────────────

_monitor_thread: Optional[Thread] = None
_monitor_stop = Event()
_monitor_lock = Lock()

def _monitor_worker():
    logger.info(f"Monitor iniciado: a cada {POLL_INTERVAL_MINUTES} minuto(s).")
    while not _monitor_stop.is_set():
        if not _monitor_lock.acquire(blocking=False):
            logger.warning("Batch anterior ainda está executando, pulando ciclo...")
            _monitor_stop.wait(POLL_INTERVAL_MINUTES * 60)
            continue
        try:
            result = run_batch()
            if result.get("new", 0) > 0:
                logger.info(f"Monitor: {result['message']}")
        except Exception as e:
            logger.error(f"Erro no monitor: {e}")
        finally:
            _monitor_lock.release()
        _monitor_stop.wait(POLL_INTERVAL_MINUTES * 60)

def start_monitor():
    global _monitor_thread
    if _monitor_thread and _monitor_thread.is_alive():
        logger.info("Monitor já está rodando.")
        return
    _monitor_stop.clear()
    _monitor_thread = Thread(target=_monitor_worker, daemon=True)
    _monitor_thread.start()

def stop_monitor():
    _monitor_stop.set()
    if _monitor_thread:
        _monitor_thread.join(timeout=10)
    logger.info("Monitor parado.")

# ── FastAPI App ─────────────────────────────────────────────────

app = FastAPI(title="obito-ocr-service", version="2.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "service": "obito-ocr-service", "version": "2.0.0"}

# ── Autenticação ────────────────────────────────────────────────

def _check_auth(authorization: Optional[str]) -> None:
    if not ENDPOINT_AUTH_TOKEN:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail={
            "code": "UNAUTHORIZED",
            "message": "Cabeçalho Authorization ausente.",
            "requestId": None,
        })
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != ENDPOINT_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail={
            "code": "UNAUTHORIZED",
            "message": "Token Bearer inválido.",
            "requestId": None,
        })

# ── Endpoint /ocr ───────────────────────────────────────────────

@app.post("/ocr")
async def ocr_endpoint(request: Request, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={
            "code": "INVALID_JSON",
            "message": "Corpo da requisição não é JSON válido.",
            "requestId": None,
        })

    request_id = body.get("requestId") or body.get("request_id") or None
    file_b64 = body.get("file")
    mime_type = body.get("mimeType") or body.get("mime_type")
    file_name = body.get("fileName") or body.get("file_name") or ""

    if not file_b64:
        return JSONResponse(status_code=400, content={
            "code": "MISSING_FILE",
            "message": "Campo 'file' (base64) é obrigatório.",
            "requestId": request_id,
        })
    if not mime_type:
        return JSONResponse(status_code=400, content={
            "code": "MISSING_MIME_TYPE",
            "message": "Campo 'mimeType' é obrigatório.",
            "requestId": request_id,
        })
    if "pdf" in mime_type.lower() or file_name.lower().endswith(".pdf"):
        return JSONResponse(status_code=422, content={
            "code": "PDF_NOT_SUPPORTED",
            "message": "PDF não é suportado. Envie imagem (PNG/JPG).",
            "requestId": request_id,
        })

    try:
        file_bytes = base64.b64decode(file_b64, validate=False)
    except Exception:
        return JSONResponse(status_code=400, content={
            "code": "INVALID_BASE64",
            "message": "Não foi possível decodificar o base64 de 'file'.",
            "requestId": request_id,
        })
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return JSONResponse(status_code=413, content={
            "code": "FILE_TOO_LARGE",
            "message": f"Arquivo excede o limite de {MAX_FILE_SIZE_MB} MB.",
            "requestId": request_id,
        })

    hash_arquivo = _sha256_bytes(file_bytes)
    try:
        raw_text, confidence = ocr_image(file_bytes, mime_type)
    except OCRProviderError as e:
        return JSONResponse(status_code=e.status_code, content={
            "code": e.code, "message": str(e), "requestId": request_id,
        })
    except Exception as e:
        return JSONResponse(status_code=502, content={
            "code": "OCR_PROVIDER_ERROR",
            "message": f"Erro inesperado no OCR: {e}",
            "requestId": request_id,
        })

    try:
        structured = parse_obito(raw_text)
    except Exception as e:
        structured = {k: "" for k in HEADER}
        structured["ERROS"] = f"Erro no parser: {e}"

    structured["HASH_ARQUIVO"] = hash_arquivo
    structured["HASH_CONTEUDO"] = _sha256_text(raw_text)
    validation = validate_obito(structured)

    warnings = list(validation.get("warnings", []))
    if validation.get("errors"):
        warnings.extend([f"ERRO: {e}" for e in validation["errors"]])

    return JSONResponse(status_code=200, content={
        "text": raw_text,
        "confidence": confidence,
        "provider": OCR_PROVIDER,
        "requestId": request_id,
        "warnings": warnings,
        "rawText": raw_text,
        "structured": structured,
        "validation": validation,
        "headerOrder": HEADER,
    })
# ── Diagnóstico (OCR bruto para debug) ──────────────────────────

# ── Diagnóstico por file_id do Drive ────────────────────────────

@app.post("/diagnose/{file_id}")
async def diagnose_file(file_id: str, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    try:
        image_bytes, mime_type = _download_image_bytes(file_id)
    except Exception as e:
        return JSONResponse(status_code=502, content={
            "code": "DRIVE_ERROR",
            "message": f"Erro ao baixar imagem do Drive: {e}",
        })

    try:
        raw_text, confidence = ocr_image(image_bytes, mime_type)
    except Exception as e:
        return {"file_id": file_id, "error": str(e), "raw_text": "", "confidence": 0}

    structured = parse_obito(raw_text)
    validate_obito(structured)

    return {
        "file_id": file_id,
        "confidence": confidence,
        "provider": OCR_PROVIDER,
        "raw_text": raw_text,
        "structured": structured,
        "is_valid_obito": _is_valid_obito(raw_text),
    }
       
# ── Endpoints Batch ─────────────────────────────────────────────

@app.post("/batch/process")
async def batch_process(request: Request, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    try:
        body = await request.json()
    except Exception:
        body = {}
    folder_id = body.get("folderId") or body.get("folder_id") or None
    force = body.get("force_reprocess", body.get("force", False))
    request_id = body.get("requestId") or body.get("request_id") or None
    limit = int(request.query_params.get("limit", 0))
    result = run_batch(folder_id=folder_id, force_reprocess=force, limit=limit)
    result["requestId"] = request_id
    status_code = 200 if result.get("success") else 500
    return JSONResponse(status_code=status_code, content=result)

@app.get("/batch/status")
async def batch_status(authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    return {
        "monitor_running": _monitor_thread is not None and _monitor_thread.is_alive(),
        "drive_folder_id": DRIVE_FOLDER_ID,
        "sheet_id": SHEET_ID,
        "auto_process_enabled": AUTO_PROCESS_ENABLED,
        "poll_interval_minutes": POLL_INTERVAL_MINUTES,
        "ocr_provider": OCR_PROVIDER,
    }

@app.post("/batch/monitor/start")
async def monitor_start(authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    if not DRIVE_FOLDER_ID:
        return JSONResponse(status_code=400, content={
            "code": "MISSING_FOLDER",
            "message": "DRIVE_FOLDER_ID não configurado.",
        })
    start_monitor()
    return {"success": True, "message": "Monitor iniciado."}

@app.post("/batch/monitor/stop")
async def monitor_stop(authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    stop_monitor()
    return {"success": True, "message": "Monitor parado."}

@app.post("/batch/config/sheet")
async def config_sheet(request: Request, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    try:
        sheet_id = _ensure_sheet_exists()
        return {"success": True, "sheet_id": sheet_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False, "error": str(e),
        })

# ── Tratamento de erros global ──────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    if "code" not in detail:
        detail["code"] = "HTTP_ERROR"
    if "requestId" not in detail:
        detail["requestId"] = None
    return JSONResponse(status_code=exc.status_code, content=detail)

# ── Entry point ─────────────────────────────────────────────────

if AUTO_PROCESS_ENABLED and DRIVE_FOLDER_ID and DRIVE_SERVICE_ACCOUNT_JSON and SHEET_ID:
    start_monitor()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)

# ── Listar imagens da pasta (para diagnóstico) ──────────────────

@app.get("/diagnose/files")
async def diagnose_list_files(authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    folder_id = os.getenv("DRIVE_FOLDER_ID")
    if not folder_id:
        return JSONResponse(status_code=400, content={"error": "DRIVE_FOLDER_ID não configurado"})
    
    images = _list_images_in_folder(folder_id)
    return {
        "total": len(images),
        "files": [
            {
                "id": img["id"],
                "name": img["name"],
                "mimeType": img.get("mimeType", "unknown"),
            }
            for img in images[:20]  # primeiras 20
        ]
    }
