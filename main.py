"""
Serviço obito-ocr-service
Backend FastAPI para OCR de declaração de óbito usando provedor OpenAI-compatible.
Variáveis de ambiente:
  ENDPOINT_AUTH_TOKEN      Token Bearer para autenticar POST /ocr
  OPENAI_API_URL           URL base do provedor OpenAI-compatible (chat completions)
  OPENAI_API_KEY           Chave de API do provedor
  OPENAI_MODEL_DEFAULT     Modelo padrão de visão
  MAX_FILE_SIZE_MB         Tamanho máximo de arquivo em MB
  PORT                     Porta HTTP
"""
import os
import re
import io
import time
import base64
from threading import Thread, Event
from typing import List, Optional
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json
import base64
import hashlib
import unicodedata
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple
import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
import logging
logger = logging.getLogger(__name__)

def _format_date(raw: str) -> str:
    """Formata data extraída, validando dia mês e ano com fallbacks."""
    if not raw:
        return raw
    cleaned = re.sub(r'(?<=\d)\s+(?=\d)', '', raw)
    digits = re.sub(r'\D', '', cleaned)
    if len(digits) < 6:
        return raw
    if len(digits) == 8:
        d, m, a = int(digits[:2]), int(digits[2:4]), int(digits[4:])
    elif len(digits) == 6:
        d, m, a = int(digits[:2]), int(digits[2:4]), int(digits[4:]) + 2000
    else:
        return raw
    if d > 31 and m <= 12:
        d, m = m, d
    if d > 31:
        d = min(d % 10, 31) if d % 10 <= 31 else 15
    if m > 12:
        m = m % 10 if m % 10 != 0 else 12
    if a < 1900 or a > 2100:
        if 0 <= a <= 30:
            a += 2000
        elif 31 <= a <= 99:
            a += 1900
        else:
            a = max(1900, min(a, 2100))
    return f"{d:02d}/{m:02d}/{a:04d}"

def _get_existing_hashes(sheet_id: str) -> set:
    """Lê todos os hashes já registrados na planilha para evitar reprocessamento."""
    try:
        sheets = _get_sheets_service()
        sheet_name = _get_sheet_name(sheet_id)
        result = sheets.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=f"{sheet_name}!Q:Q",
        ).execute()
        values = result.get("values", [])
        if not values:
            return set()
        hashes = set()
        for row in values:
            if row and row[0].strip() and row[0].strip() != "HASH_ARQUIVO":
                hashes.add(row[0].strip())
        return hashes
    except Exception as e:
        logger.warning(f"Não foi possível ler hashes existentes: {e}")
        return set()
# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
ENDPOINT_AUTH_TOKEN = os.environ.get("ENDPOINT_AUTH_TOKEN", "")
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_MODEL_DEFAULT", "gpt-4o")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "10"))
PORT = int(os.environ.get("PORT", "8000"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Ordem oficial do cabeçalho estruturado
HEADER = [
    'NOME', 'NOME_SOCIAL', 'NASCIMENTO', 'SEXO', 'RACA_COR', 'ESTADO_CIVIL',
    'NACIONALIDADE', 'NOME_MAE', 'NOME_PAI', 'PROFISSAO', 'LOGRADOURO',
    'NUMERO', 'COMPLEMENTO', 'BAIRRO', 'CIDADE', 'UF', 'CEP',
    'CIDADE_NASCIMENTO', 'UF_NASCIMENTO', 'CPF', 'RG', 'ORGAO_EMISSOR_RG',
    'DATA_OBITO', 'HORA_OBITO', 'LOCAL_OBITO', 'CIDADE_OBITO', 'UF_OBITO',
    'CAUSA_MORTE', 'CAUSA_MORTE_2', 'CAUSA_MORTE_3', 'CAUSA_MORTE_4',
    'CAUSA_MORTE_5', 'CAUSA_BASICA', 'CODIGO_CAUSA_BASICA',
    'CODIGO_CAUSA_MORTE', 'CODIGO_CAUSA_MORTE_2', 'CODIGO_CAUSA_MORTE_3',
    'CODIGO_CAUSA_MORTE_4', 'CODIGO_CAUSA_MORTE_5', 'CID_BASICA', 'CID_MORTE',
    'CID_MORTE_2', 'CID_MORTE_3', 'CID_MORTE_4', 'CID_MORTE_5', 'TIPO_OBITO',
    'ASSISTIDO', 'DATA_ATESTADO', 'NOMES_OK', 'NOME_OK', 'GARBAGE_CODES',
    'QTD_GARBAGE', 'PROTOCOLO_TEV', 'ERROS', 'QUALIDADE_SCORE',
    'HASH_ARQUIVO', 'HASH_CONTEUDO', 'STATUS', 'NOME_MES', 'DATA_PROCESSAMENTO'
]

UF_VALIDAS = {
    'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS',
    'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC',
    'SP', 'SE', 'TO'
}

MESES_PT = {
    '01': 'JANEIRO', '02': 'FEVEREIRO', '03': 'MARCO', '04': 'ABRIL',
    '05': 'MAIO', '06': 'JUNHO', '07': 'JULHO', '08': 'AGOSTO',
    '09': 'SETEMBRO', '10': 'OUTUBRO', '11': 'NOVEMBRO', '12': 'DEZEMBRO'
}

REFUSAL_PHRASES = [
    "i'm sorry",
    "i can't assist with that",
    "i can't assist",
    "cannot assist",
    "unable to help",
    "cannot help with that request",
    "i can't help with that",
    "desculpe",
    "não posso ajudar",
    "nao posso ajudar",
]

NOISE_LINES = {
    'parte i', 'parte ii', 'devido ou como consequência de', 'devido a',
    'intervalo entre o início e a morte', 'cid',
    'meses dias horas minutos ignorado', 'meses', 'dias', 'horas',
    'minutos', 'ignorado', 'nome', 'nome do médico', 'nome do medico',
    'crm', 'assinatura', 'carimbo', 'uf', 'município', 'municipio',
    'data', 'hora', 'local', 'causas da morte', 'causa da morte',
    'causas', 'causa', 'outras condições significativas',
    'outras condicoes significativas', 'prováveis circunstâncias',
    'provaveis circunstancias', 'óbito atestado por médico',
    'obito atestado por medico', 'endereço', 'endereco', 'logradouro',
    'número', 'numero', 'complemento', 'bairro', 'cep', 'cpf', 'rg',
    'sexo', 'raça', 'raca', 'estado civil', 'nacionalidade', 'profissão',
    'profissao', 'ocupação', 'ocupacao', 'naturalidade',
}

LABEL_KEYWORDS = [
    'nome', 'nome do', 'nome da', 'data', 'hora', 'local', 'município',
    'municipio', 'uf', 'cep', 'cpf', 'rg', 'sexo', 'raça', 'raca',
    'estado civil', 'nacionalidade', 'profissão', 'profissao', 'ocupação',
    'ocupacao', 'naturalidade', 'logradouro', 'endereço', 'endereco',
    'número', 'numero', 'complemento', 'bairro', 'cidade', 'parte',
    'devido', 'intervalo', 'cid', 'crm', 'médico', 'medico', 'assinatura',
    'carimbo', 'meses', 'dias', 'horas', 'minutos', 'causas', 'causa',
    'óbito atestado', 'obito atestado', 'outras condições', 'outras condicoes',
    'prováveis', 'provaveis',
]
# ── Configuração Batch / Drive / Sheets ─────────────────────────────
DRIVE_SERVICE_ACCOUNT_JSON = os.environ.get("DRIVE_SERVICE_ACCOUNT_JSON", "")
SHEET_ID = os.environ.get("SHEET_ID", "")               # ID da planilha existente (ou vazio pra criar nova)
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "")  # Pasta raiz a monitorar
POLL_INTERVAL_MINUTES = int(os.environ.get("POLL_INTERVAL_MINUTES", "60"))
AUDIT_SHEET_TITLE = os.environ.get("AUDIT_SHEET_TITLE", "Auditoria Obito OCR")
AUTO_PROCESS_ENABLED = os.environ.get("AUTO_PROCESS_ENABLED", "false").lower() == "true"
PROCESSED_IMAGES_LOG = "processed_images.log"  # arquivo local para evitar reprocessamento

# Ordem das colunas na planilha de auditoria
AUDIT_COLUMNS = [
    "DATA_PROCESSAMENTO",
    "NOME_ARQUIVO",
    "STATUS",
    "QUALIDADE_SCORE",
    "NOME",
    "NOME_MAE",
    "NASCIMENTO",
    "DATA_OBITO",
    "HORA_OBITO",
    "CIDADE_OBITO",
    "UF_OBITO",
    "CAUSA_MORTE",
    "CAUSA_BASICA",
    "CID_BASICA",
    "TIPO_OBITO",
    "ERROS",
    "HASH_ARQUIVO",
]

# ---------------------------------------------------------------------------
# Exceção de provedor OCR
# ---------------------------------------------------------------------------
class OCRProviderError(Exception):
    """Erro de provedor OCR (recusa, falha ou resposta inválida)."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code
        self.code = "OCR_PROVIDER_ERROR"

# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="obito-ocr-service", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "service": "obito-ocr-service", "version": "1.0.0"}

# ---------------------------------------------------------------------------
# Autenticação
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def _unaccent(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def _norm_label(s: str) -> str:
    return _unaccent(s).lower().strip()

def _normalize_lines(text: str) -> List[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]

def _strip_numeric_prefix(line: str) -> str:
    s = line.strip()
    m = re.match(r'^(\d+\s*[\.\):\-]?\s*)(.*)$', s)
    if m and re.search(r'[A-Za-zÀ-ú]', m.group(2)):
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
    if re.fullmatch(r'[\d\s\.\-:/]+', norm_line) and len(norm_line) < 3:
        return True
    for kw in LABEL_KEYWORDS:
        if nl == kw or nl.startswith(kw):
            return True
    return False

def _looks_like_label(norm_line: str) -> bool:
    return _is_noise_line(norm_line)

def _normalize_date(value: str) -> str:
    if not value:
        return ""
    v = value.strip()
    v = re.sub(r"[^0-9/\-\s]", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    m = re.match(r"^(\d{1,2})[\s/\-]+(\d{1,2})[\s/\-]+(\d{2,4})$", v)
    if not m:
        return value.strip()
    d, mo, y = m.group(1), m.group(2), m.group(3)
    if len(y) == 2:
        y = "19" + y if int(y) > 30 else "20" + y
    try:
        dt.date(int(y), int(mo), int(d))
    except ValueError:
        return value.strip()
    return f"{int(d):02d}/{int(mo):02d}/{y}"

def _normalize_hour(value: str) -> str:
    if not value:
        return ""
    v = value.strip()
    m = re.search(r"(\d{1,2})[:hH]+(\d{2})", v)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= h <= 23 and 0 <= mm <= 59:
            return f"{h:02d}:{mm:02d}"
    return v

def _is_valid_hour(value: str) -> bool:
    return bool(re.fullmatch(r"\d{2}:\d{2}", value or "")) and _normalize_hour(value) == value

def _normalize_uf(value: str) -> str:
    if not value:
        return ""
    v = value.strip().upper()
    if v == "UF":
        return ""
    m = re.search(r"\b([A-Z]{2})\b", v)
    if m and m.group(1) in UF_VALIDAS:
        return m.group(1)
    if v in UF_VALIDAS:
        return v
    return ""

def _normalize_cep(value: str) -> str:
    if not value:
        return ""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 8:
        return f"{digits[:5]}-{digits[5:]}"
    return value.strip()

# ---------------------------------------------------------------------------
# Parser: busca estrita por próxima linha
# ---------------------------------------------------------------------------
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
    value, _ = _find_next_value_after_label(text, labels, stop_labels=stop_labels, max_distance=max_distance)
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
                if _looks_like_label(cnorm):
                    break
    return ""

def _find_uf_after(
    text: str, after_labels: List[str], max_distance: int = 10,
) -> str:
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
                if _looks_like_label(cnorm):
                    break
    return ""
# ═══════════════════════════════════════════════════════════════════
# Módulo Batch: Google Drive / Sheets / Processamento em Lote
# ═══════════════════════════════════════════════════════════════════

def _get_drive_service():
    """Constrói cliente autenticado do Google Drive via service account."""
    if not DRIVE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("DRIVE_SERVICE_ACCOUNT_JSON não configurado.")
    creds = service_account.Credentials.from_service_account_info(
        json.loads(DRIVE_SERVICE_ACCOUNT_JSON),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)

def _get_sheets_service():
    """Constrói cliente autenticado do Google Sheets via service account."""
    if not DRIVE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("DRIVE_SERVICE_ACCOUNT_JSON não configurado.")
    creds = service_account.Credentials.from_service_account_info(
        json.loads(DRIVE_SERVICE_ACCOUNT_JSON),
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return build("sheets", "v4", credentials=creds)

def _list_images_in_folder(folder_id: str, since: Optional[datetime] = None) -> List[dict]:
    """
    Lista arquivos de imagem (.jpg, .jpeg, .png, .gif, .bmp, .tiff)
    dentro de folder_id de forma RECURSIVA (inclui subpastas).
    Se `since` for fornecido, retorna apenas arquivos modificados após aquela data.
    """
    drive = _get_drive_service()
    
    # 1. Busca IMAGENS na pasta atual
    query = (
        f"'{folder_id}' in parents and "
        f"(mimeType='image/jpeg' or mimeType='image/png' or "
        f"mimeType='image/gif' or mimeType='image/bmp' or "
        f"mimeType='image/tiff') and trashed=false"
    )
    page_token = None
    files = []
    while True:
        resp = drive.files().list(
            q=query,
            fields="files(id, name, mimeType, modifiedTime, parents)",
            pageToken=page_token,
            pageSize=200,
        ).execute()
        batch = resp.get('files', [])
        if since:
            batch = [
                f for f in batch
                if _parse_rfc3339(f.get("modifiedTime", "")) > since
            ]
        files.extend(batch)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    
    # 2. Busca SUBPASTAS (recursão)
    folder_query = (
        f"'{folder_id}' in parents and "
        f"mimeType='application/vnd.google-apps.folder' and "
        f"trashed=false"
    )
    page_token = None
    while True:
        folders_resp = drive.files().list(
            q=folder_query,
            fields="files(id, name)",
            pageToken=page_token,
            pageSize=100,
        ).execute()
        for subfolder in folders_resp.get('files', []):
            print(f"Explorando subpasta: {subfolder['name']}")
            sub_files = _list_images_in_folder(subfolder['id'], since=since)
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
    """Baixa o conteúdo binário de um arquivo do Drive e seu MIME type."""
    drive = _get_drive_service()
    meta = drive.files().get(fileId=file_id, fields="name, mimeType").execute()
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue(), meta.get("mimeType", "image/jpeg")

def _ocr_image_from_bytes(image_bytes: bytes, mime_type: str) -> Tuple[str, float]:
    """Chama o mesmo pipeline de OCR interno (sem HTTP)."""
    return ocr_openai_compatible(image_bytes, mime_type, OPENAI_MODEL_DEFAULT)

def _process_single_image(file_id: str, file_name: str) -> dict:
    """Pipeline completo para uma imagem: baixar → OCR → parse → validar."""
    logger.info(f"Processando: {file_name} ({file_id})")
    try:
        image_bytes, mime_type = _download_image_bytes(file_id)
    except Exception as e:
        return {"NOME_ARQUIVO": file_name, "STATUS": "ERRO_DRIVE", "ERROS": str(e)}
    
    try:
        raw_text, confidence = _ocr_image_from_bytes(image_bytes, mime_type)
    except Exception as e:
        return {"NOME_ARQUIVO": file_name, "STATUS": "ERRO_OCR", "ERROS": str(e)}
    
    try:
        structured = parse_obito(raw_text)
    except Exception as e:
        structured = {k: "" for k in HEADER}
        structured["ERROS"] = f"Erro no parser: {e}"
    
    structured["HASH_ARQUIVO"] = _sha256_bytes(image_bytes)
    structured["HASH_CONTEUDO"] = _sha256_text(raw_text)
    
    validate_obito(structured)
    
    row = {
        "DATA_PROCESSAMENTO": datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S"),
        "NOME_ARQUIVO": file_name,
        "STATUS": structured.get("STATUS", ""),
        "QUALIDADE_SCORE": str(structured.get("QUALIDADE_SCORE", "")),
        "NOME": structured.get("NOME", ""),
        "NOME_MAE": structured.get("NOME_MAE", ""),
        "NASCIMENTO": _format_date(structured.get("NASCIMENTO", "")),
        "DATA_OBITO": _format_date(structured.get("DATA_OBITO", "")),
        "HORA_OBITO": structured.get("HORA_OBITO", ""),
        "CIDADE_OBITO": structured.get("CIDADE_OBITO", ""),
        "UF_OBITO": structured.get("UF_OBITO", ""),
        "CAUSA_MORTE": structured.get("CAUSA_MORTE", ""),
        "CAUSA_BASICA": structured.get("CAUSA_BASICA", ""),
        "CID_BASICA": structured.get("CID_BASICA", ""),
        "TIPO_OBITO": structured.get("TIPO_OBITO", ""),
        "ERROS": structured.get("ERROS", ""),
        "HASH_ARQUIVO": structured.get("HASH_ARQUIVO", ""),
    }
    return row

def _ensure_sheet_exists() -> str:
    """Cria a planilha se não existir, ou garante que a aba 'Auditoria' exista com cabeçalhos."""
    sheets = _get_sheets_service()

    if SHEET_ID:
        # Verifica se a aba 'Auditoria' já existe
        metadata = sheets.spreadsheets().get(
            spreadsheetId=SHEET_ID,
            fields="sheets.properties.title",
        ).execute()
        tab_names = [s["properties"]["title"] for s in metadata.get("sheets", [])]

        if "Auditoria" not in tab_names:
            # Cria a aba
            sheets.spreadsheets().batchUpdate(
                spreadsheetId=SHEET_ID,
                body={"requests": [{"addSheet": {"properties": {"title": "Auditoria"}}}]},
            ).execute()
            headers = [[col for col in AUDIT_COLUMNS]]
            sheets.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range="Auditoria!A1",
                valueInputOption="RAW",
                body={"values": headers},
            ).execute()
            logger.info(f"Aba 'Auditoria' criada na planilha {SHEET_ID}")
        return SHEET_ID

    # Fluxo original para planilha nova
    spreadsheet = {
        "properties": {"title": AUDIT_SHEET_TITLE},
        "sheets": [{"properties": {"title": "Auditoria"}}],
    }
    sheet = sheets.spreadsheets().create(body=spreadsheet, fields="spreadsheetId").execute()
    sid = sheet.get("spreadsheetId")
    headers = [[col for col in AUDIT_COLUMNS]]
    sheets.spreadsheets().values().update(
        spreadsheetId=sid,
        range="Auditoria!A1",
        valueInputOption="RAW",
        body={"values": headers},
    ).execute()
    logger.info(f"Nova planilha criada: {sid}")
    return sid

def _get_sheet_name(sheet_id: str) -> str:
    """Sempre retorna 'Auditoria' como aba de destino."""
    return "Auditoria"

def _append_rows_to_sheet(sheet_id: str, rows: List[dict]):
    """Appenda linhas de resultado na planilha."""
    if not rows:
        return
    sheets = _get_sheets_service()
    sheet_name = _get_sheet_name(sheet_id)
    range_name = f"{sheet_name}!A1"
    values = []
    for row in rows:
        values.append([row.get(col, "") for col in AUDIT_COLUMNS])
    sheets.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range=range_name,
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": values},
    ).execute()

def _load_processed_ids() -> set:
    """Carrega IDs de imagens já processadas (para evitar repetição)."""
    if not os.path.exists(PROCESSED_IMAGES_LOG):
        return set()
    with open(PROCESSED_IMAGES_LOG, "r") as f:
        return set(line.strip() for line in f if line.strip())

def _save_processed_ids(ids: set):
    """Salva IDs processados no arquivo de log."""
    with open(PROCESSED_IMAGES_LOG, "a") as f:
        for fid in ids:
            f.write(fid + "\n")

def run_batch(folder_id: str = None, force_reprocess: bool = False, limit: int = 0) -> dict:
    """
    Pipeline completo do lote:
    1. Lista imagens não processadas na pasta
    2. OCR cada uma
    3. Escreve na planilha
    4. Marca como processadas
    Se limit > 0, processa no máximo esse número de imagens.
    """
    fid = folder_id or DRIVE_FOLDER_ID
    if not fid:
        return {"success": False, "error": "Nenhum DRIVE_FOLDER_ID configurado."}

    processed_ids = set() if force_reprocess else _load_processed_ids()
    images = _list_images_in_folder(fid)
    
    # Carrega hashes já registrados na planilha para evitar duplicatas (persistente)
    sheet_id = _ensure_sheet_exists()
    existing_hashes = _get_existing_hashes(sheet_id)
    
    # Filtra apenas não processadas
    new_images = []
    for img in images:
        if img["id"] in processed_ids:
            continue
        # Verifica pelo nome do arquivo na planilha (fallback após restart)
        filename = img["name"]
        if _is_filename_in_sheet(sheet_id, filename):
            logger.info(f"{filename} já está na planilha, pulando...")
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

    # Aplica limit se especificado
    if limit > 0:
        new_images = new_images[:limit]

    sheet_id = _ensure_sheet_exists()
    success_ids = set()
    fail_ids = set()
    last_error = None

    for img in new_images:
    # ✅ Pula se já está na planilha
    if _is_filename_in_sheet(sheet_id, img["name"]):
        print(f"{img['name']} já está na planilha, pulando...")
        continue

    try:
        row = _process_single_image(img["id"], img["name"])
        _append_rows_to_sheet(sheet_id, [row])
        success_ids.add(img["id"])

        import gc
        gc.collect()

    except Exception as e:
        fail_ids.add(img["id"])
        last_error = str(e)
        print(f"Falha ao processar {img['name']}: {e}")

    _save_processed_ids(success_ids)
    if os.path.exists(local_path):
       os.remove(local_path)

    return {
        "success": True,
        "total": len(images),
        "new": len(new_images),
        "processed": len(success_ids),
        "failed": len(fail_ids),
        "sheet_id": sheet_id,
        "message": f"{len(success_ids)} imagens processadas, {len(fail_ids)} falhas.",
    }
def _is_filename_in_sheet(sheet_id: str, filename: str) -> bool:
    """Verifica se o nome do arquivo já existe na coluna B da planilha."""
    try:
        sheets = _get_sheets_service()
        sheet_name = _get_sheet_name(sheet_id)
        result = sheets.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=f"{sheet_name}!B:B",
        ).execute()
        values = result.get("values", [])
        for row in values:
            if row and row[0].strip() == filename:
                return True
        return False
    except:
        return False
# ═══════════════════════════════════════════════════════════════════
# Background Monitor (thread de polling)
# ═══════════════════════════════════════════════════════════════════

_monitor_thread: Optional[Thread] = None
_monitor_stop = Event()

def _monitor_worker():
    """Thread que verifica periodicamente a pasta do Drive."""
    logger.info(f"Monitor iniciado: a cada {POLL_INTERVAL_MINUTES} minuto(s).")
    while not _monitor_stop.is_set():
        try:
            result = run_batch()
            if result.get("new", 0) > 0:
                logger.info(f"Monitor: {result['message']}")
        except Exception as e:
            logger.error(f"Erro no monitor: {e}")
        _monitor_stop.wait(POLL_INTERVAL_MINUTES * 60)

def start_monitor():
    """Inicia a thread de monitoramento em background."""
    global _monitor_thread
    if _monitor_thread and _monitor_thread.is_alive():
        logger.info("Monitor já está rodando.")
        return
    _monitor_stop.clear()
    _monitor_thread = Thread(target=_monitor_worker, daemon=True)
    _monitor_thread.start()

def stop_monitor():
    """Para a thread de monitoramento."""
    _monitor_stop.set()
    if _monitor_thread:
        _monitor_thread.join(timeout=10)
    logger.info("Monitor parado.")
# ── Constantes para extração de causas ──────────────────────────────────
_CAUSA_BASICA_BLACKLIST = [
    'outras condições significativas', 'outras condicoes significativas',
    'nome do médico', 'nome do medico', 'crm',
    'óbito atestado', 'obito atestado', 'medico', 'médico',
     'outras condições significativas', 'outras condicoes significativas',
    'nome do médico', 'nome do medico', 'crm',
    'óbito atestado', 'obito atestado', 'medico', 'médico',
    'outras afecções', 'outras afeccoes',
]

_INTERVALO_RE = re.compile(
    r'[<>]?\s*\d+\s*(dia|dias|hora|horas|min|minutos|d|h|m)'
)

def _causa_valida(c: str) -> bool:
    """Verifica se uma string é uma causa de morte válida (não é ruído)."""
    if not c or not c.strip():
        return False
    cl = _norm_label(c)
    if len(cl) < 3:
        return False
    # Rejeita coluna de intervalo/duração (ex.: >7d, <24h, 3d, 12h, 30m)
    if re.fullmatch(r'[<>]?\s*\d+\s*[dhm]?', cl):
        return False
    if _INTERVALO_RE.fullmatch(cl):
        return False
    if 'intervalo entre o inicio e a morte' in cl:
        return False
    for proibido in _CAUSA_BASICA_BLACKLIST:
        if proibido in cl:
            return False
    if _is_noise_line(c):
        return False
    return True

_CID_RE = re.compile(
    r'\b([A-TV-Z]\d{2}(?:\.\s*\d{1,4})?)\b',
    re.IGNORECASE
)

def _extract_causes(text: str) -> List[str]:
    """
    Extrai causas da morte em ordem.
    - Inicia em 'CAUSAS DA MORTE'
    - Para imediatamente em: 'Parte II', 'Outras condições significativas',
      'Nome do Médico', 'CRM', 'Óbito atestado por Médico', 'PROVÁVEIS CIRCUNSTÂNCIAS'
    - Ignora linhas auxiliares: 'Parte I', 'Devido ou como consequência de',
      'Intervalo entre o início e a morte', 'CID', 'Meses Dias Horas Minutos Ignorado'
    """
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
        "parte i", "devido ou como consequência de", "devido a", "intervalo entre o início e a morte", 
        "intervalo entre o inicio e a morte","cid", "meses dias horas minutos ignorado", "causas da morte",
        "causa da morte", "outras afecções","outras afeccoes",
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
    # ── Filtro adicional: remove causas inválidas pelo patch ──────────
    causes = [c.strip() for c in causes if _causa_valida(c)]
    return causes

# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------
def parse_obito(text: str) -> Dict[str, Any]:
    """Constrói o dicionário estruturado a partir do texto OCR."""
    structured: Dict[str, Any] = {k: "" for k in HEADER}

    structured["NOME"] = _find_block_value(
        text,
        ["Nome do Falecido", "Nome do falecido", "Nome do(a) Falecido(a)", "Nome do(a) falecido(a)"],
        stop_labels=["Nome da mãe", "Nome da mae", "Nome do pai", "Nome social", "Data"],
    )
    structured["NOME_SOCIAL"] = _find_block_value(
        text, ["Nome social", "Nome Social"],
        stop_labels=["Nome do falecido", "Nome da mãe", "Nome da mae", "Nome do pai"],
    )
    structured["NOME_MAE"] = _find_block_value(
        text,
        ["Nome da Mãe", "Nome da mãe", "Nome da mae", "Nome da Mae"],
        stop_labels=["Nome do pai", "Profissão", "Profissao", "Endereço", "Endereco", "Nacionalidade"],
    )
    structured["NOME_PAI"] = _find_block_value(
        text,
        ["Nome do Pai", "Nome do pai"],
        stop_labels=["Profissão", "Profissao", "Endereço", "Endereco", "Nacionalidade", "Nome da mãe", "Nome da mae"],
    )
    structured["NASCIMENTO"] = _normalize_date(
        _find_block_value(text,
            ["Data de nascimento", "Data de Nascimento", "Nascimento"],
            stop_labels=["Data do óbito", "Data do obito", "Sexo", "Raça", "Raca"],
        )
    )
    structured["DATA_OBITO"] = _normalize_date(
        _find_block_value(text,
            ["Data do óbito", "Data de óbito", "Data do obito", "Data de obito"],
            stop_labels=["Hora", "Local do óbito", "Local do obito", "Município de ocorrência", "Municipio de ocorrencia"],
        )
    )
    structured["HORA_OBITO"] = _find_hora_obito(text)
    structured["DATA_ATESTADO"] = _normalize_date(
        _find_block_value(text, ["Data do atestado", "Data de emissão", "Data da emissão"])
    )
    structured["LOCAL_OBITO"] = _find_block_value(
        text, ["Local do óbito", "Local de óbito", "Local do obito", "Local de obito"],
        stop_labels=["Município de ocorrência", "Municipio de ocorrencia", "UF"],
    )
    structured["CIDADE_OBITO"] = _find_block_value(
        text,
        ["Município de ocorrência", "Municipio de ocorrência", "Município de ocorrencia", "Municipio de ocorrencia"],
        stop_labels=["UF", "Estado", "Data", "CEP", "Cep"],
    )
    structured["UF_OBITO"] = _find_uf_after(text, ["Município de ocorrência", "Municipio de ocorrencia"])
    structured["LOGRADOURO"] = _find_block_value(text, ["Logradouro", "Endereço", "Endereco"], stop_labels=["Número", "Numero", "Complemento", "Bairro"])
    structured["NUMERO"] = _find_block_value(text, ["Número", "Numero"], stop_labels=["Complemento", "Bairro"])
    structured["COMPLEMENTO"] = _find_block_value(text, ["Complemento"], stop_labels=["Bairro", "Município", "Municipio"])
    structured["BAIRRO"] = _find_block_value(text, ["Bairro"], stop_labels=["Município", "Municipio", "Cidade", "UF"])
    structured["CIDADE"] = _find_block_value(text, ["Município", "Municipio", "Cidade"], stop_labels=["UF", "CEP", "Cep"])
    structured["UF"] = _find_uf_after(text, ["Endereço", "Endereco", "Logradouro", "Bairro", "Município", "Municipio", "Cidade"])
    structured["CEP"] = _normalize_cep(_find_block_value(text, ["CEP", "Cep"]))
    structured["CIDADE_NASCIMENTO"] = _find_block_value(
        text, ["Naturalidade", "Município de nascimento", "Municipio de nascimento", "Cidade de nascimento"],
        stop_labels=["UF de nascimento", "Nacionalidade"],
    )
    structured["UF_NASCIMENTO"] = _find_uf_after(text, ["Naturalidade", "Município de nascimento", "Municipio de nascimento"])
    structured["CPF"] = _find_block_value(text, ["CPF"])
    structured["RG"] = _find_block_value(text, ["RG", "Registro Geral"])
    structured["ORGAO_EMISSOR_RG"] = _find_block_value(text, ["Órgão emissor", "Orgao emissor", "Órgão expedidor", "Orgao expedidor"])
    structured["SEXO"] = _find_block_value(text, ["Sexo"], stop_labels=["Raça", "Raca", "Cor"])
    structured["RACA_COR"] = _find_block_value(text, ["Raça/Cor", "Raça", "Raca/Cor", "Raca", "Cor"])
    structured["ESTADO_CIVIL"] = _find_block_value(text, ["Estado civil"])
    structured["NACIONALIDADE"] = _find_block_value(text, ["Nacionalidade"])
    structured["PROFISSAO"] = _find_block_value(text, ["Profissão", "Profissao", "Ocupação", "Ocupacao"])

    # --- Causas (PATCH A APLICADO) ---
    causes = _extract_causes(text)

    if causes:
        structured['CAUSA_MORTE'] = causes[0] if len(causes) >= 1 else ''
        structured['CAUSA_MORTE_2'] = causes[1] if len(causes) >= 2 else ''
        structured['CAUSA_MORTE_3'] = causes[2] if len(causes) >= 3 else ''
        structured['CAUSA_MORTE_4'] = causes[3] if len(causes) >= 4 else ''
        structured['CAUSA_MORTE_5'] = causes[4] if len(causes) >= 5 else ''
        # CAUSA_BASICA = última causa realmente válida
        validas = [c for c in causes if _causa_valida(c)]
        structured['CAUSA_BASICA'] = validas[-1] if validas else ''
    else:
        for k in ('CAUSA_MORTE', 'CAUSA_MORTE_2', 'CAUSA_MORTE_3',
                  'CAUSA_MORTE_4', 'CAUSA_MORTE_5', 'CAUSA_BASICA'):
            structured[k] = ''

    # --- CID_BASICA: extração por regex, sem inventar ---
    cid_basica = ''
    if structured.get('CAUSA_BASICA'):
        cids = _CID_RE.findall(structured['CAUSA_BASICA'])
        if cids:
            cid_basica = cids[-1].upper()
    if not cid_basica:
        cids = _CID_RE.findall(text)
        if cids:
            cid_basica = cids[-1].upper()
    structured['CID_BASICA'] = cid_basica

    # --- Tipo de óbito / assistido ---
    structured["TIPO_OBITO"] = _find_block_value(text, ["Tipo de óbito", "Tipo de obito"])
    structured["ASSISTIDO"] = _find_block_value(text, ["Assistido", "Foi assistido"])
    structured["PROTOCOLO_TEV"] = _find_block_value(text, ["Protocolo TEV", "Protocolo"])

    # --- Hashes e processamento ---
    structured["HASH_CONTEUDO"] = _sha256_text(text)
    structured["DATA_PROCESSAMENTO"] = dt.datetime.utcnow().isoformat() + "Z"

    if structured["DATA_OBITO"]:
        partes = structured["DATA_OBITO"].split("/")
        if len(partes) == 3:
            mes = partes[1].zfill(2)
            structured["NOME_MES"] = MESES_PT.get(mes, "")

    return structured

# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------
def _valid_date(value: str) -> bool:
    if not value:
        return False
    try:
        d, m, y = value.split("/")
        dt.date(int(y), int(m), int(d))
        return True
    except Exception:
        return False

def _valid_hour(value: str) -> bool:
    if not value:
        return False
    try:
        h, mm = value.split(":")
        return 0 <= int(h) <= 23 and 0 <= int(mm) <= 59
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
    """Gera o bloco de validação e preenche campos derivados de qualidade."""
    errors: List[str] = []
    warnings: List[str] = []
    computed: Dict[str, Any] = {}

    campos_criticos = ["NOME", "NOME_MAE", "NASCIMENTO", "DATA_OBITO", "CIDADE_OBITO", "UF_OBITO"]
    for campo in campos_criticos:
        if not structured.get(campo):
            errors.append(f"Campo crítico ausente: {campo}")

    if structured.get("NASCIMENTO") and not _valid_date(structured["NASCIMENTO"]):
        errors.append("NASCIMENTO com data inválida")
    if structured.get("DATA_OBITO") and not _valid_date(structured["DATA_OBITO"]):
        errors.append("DATA_OBITO com data inválida")
    if structured.get("HORA_OBITO") and not _valid_hour(structured["HORA_OBITO"]):
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
        computed["idade_anos"] = None
    else:
        computed["idade_anos"] = idade

    structured["NOME_OK"] = "SIM" if structured.get("NOME") else "NAO"
    structured["NOMES_OK"] = "SIM" if (structured.get("NOME") and structured.get("NOME_MAE")) else "NAO"

    total_campos = len(HEADER)
    preenchidos = sum(1 for k in HEADER if structured.get(k))
    score = int((preenchidos / total_campos) * 100)
    score = max(0, score - len(errors) * 10)
    structured["QUALIDADE_SCORE"] = score

    # ── Regra operacional de STATUS ──
    if errors:
        status = 'REVISAR'
    elif not structured.get('CAUSA_BASICA'):
        status = 'REVISAR'
    else:
        status = 'OK'
    structured['STATUS'] = status

    structured["ERROS"] = " | ".join(errors)

    validation = {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "computed": computed,
    }
    return validation

# ---------------------------------------------------------------------------
# Adaptador OCR (OpenAI-compatible)
# ---------------------------------------------------------------------------
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

def ocr_openai_compatible(
    image_bytes: bytes, mime_type: str, model: str,
) -> Tuple[str, float]:
    if not OPENAI_API_KEY:
        raise OCRProviderError("OPENAI_API_KEY não configurado.", 502)
    if not OPENAI_API_URL:
        raise OCRProviderError("OPENAI_API_URL não configurado.", 502)

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    prompt = (
        "Você é um motor de OCR especializado em declarações de óbito brasileiras. "
        "Extraia TODO o texto visível no documento, preservando a estrutura por linhas, "
        "rótulos e ordem dos campos. Não resuma, não omita campos e não invente dados. "
        "Retorne apenas o texto extraído, sem comentários."
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
    except Exception:
        raise OCRProviderError("Resposta do provedor OCR não é JSON válido.", 502)

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise OCRProviderError("Resposta do provedor OCR sem conteúdo esperado.", 502)

    if not isinstance(content, str) or not content.strip():
        raise OCRProviderError("Provedor OCR retornou conteúdo vazio.", 502)

    if _detect_refusal(content):
        raise OCRProviderError(
            "Provedor OCR recusou processar a imagem ou retornou texto inválido.", 502
        )

    confidence = 0.9
    try:
        usage = data.get("usage", {})
        if usage:
            confidence = 0.92
    except Exception:
        pass

    return content.strip(), confidence

# ---------------------------------------------------------------------------
# Endpoint POST /ocr
# ---------------------------------------------------------------------------
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
    model = body.get("model") or OPENAI_MODEL_DEFAULT
    file_name = body.get("fileName") or body.get("file_name") or ""
    file_id = body.get("fileId") or body.get("file_id") or ""

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
    if "pdf" in mime_type.lower() or (file_name and file_name.lower().endswith(".pdf")):
        return JSONResponse(status_code=422, content={
            "code": "PDF_NOT_SUPPORTED_IN_V1",
            "message": "PDF não é suportado na versão 1. Envie imagem (PNG/JPG).",
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
        raw_text, confidence = ocr_openai_compatible(file_bytes, mime_type, model)
    except OCRProviderError as e:
        return JSONResponse(status_code=e.status_code, content={
            "code": e.code, "message": str(e), "requestId": request_id,
        })
    except Exception as e:
        return JSONResponse(status_code=502, content={
            "code": "OCR_PROVIDER_ERROR",
            "message": f"Erro inesperado no provedor OCR: {e}",
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
    warnings = validation.get("warnings", [])
    if validation.get("errors"):
        warnings.extend([f"ERRO: {e}" for e in validation["errors"]])

    response = {
        "text": raw_text,
        "confidence": confidence,
        "provider": "openai-compatible",
        "requestId": request_id,
        "warnings": warnings,
        "rawText": raw_text,
        "structured": structured,
        "validation": validation,
        "headerOrder": HEADER,
    }
    return JSONResponse(status_code=200, content=response)

# ---------------------------------------------------------------------------
# Tratamento de erros global
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    if "code" not in detail:
        detail["code"] = "HTTP_ERROR"
    if "requestId" not in detail:
        detail["requestId"] = None
    return JSONResponse(status_code=exc.status_code, content=detail)
# ═══════════════════════════════════════════════════════════════════
# Endpoints Batch
# ═══════════════════════════════════════════════════════════════════

@app.post("/batch/process")
async def batch_process(request: Request, authorization: Optional[str] = Header(None)):
    """Processa todas as imagens novas de uma pasta do Drive."""
    _check_auth(authorization)
    try:
        body = await request.json()
    except Exception:
        body = {}

    folder_id = body.get("folderId") or body.get("folder_id") or None
    force = body.get("force_reprocess", body.get("force", False))
    request_id = body.get("requestId") or body.get("request_id") or None

    if AUTO_PROCESS_ENABLED and not folder_id:
        folder_id = DRIVE_FOLDER_ID

    # Suporta limit via query parameter: /batch/process?limit=1
    limit = int(request.query_params.get("limit", 0))
    result = run_batch(folder_id=folder_id, force_reprocess=force, limit=limit)

    result["requestId"] = request_id
    status_code = 200 if result.get("success") else 500
    return JSONResponse(status_code=status_code, content=result)

@app.get("/batch/status")
async def batch_status(authorization: Optional[str] = Header(None)):
    """Status do monitor e da pasta configurada."""
    _check_auth(authorization)
    return {
        "monitor_running": _monitor_thread is not None and _monitor_thread.is_alive(),
        "drive_folder_id": DRIVE_FOLDER_ID,
        "sheet_id": SHEET_ID,
        "auto_process_enabled": AUTO_PROCESS_ENABLED,
        "poll_interval_minutes": POLL_INTERVAL_MINUTES,
        "processed_count": len(_load_processed_ids()),
    }

@app.post("/batch/monitor/start")
async def monitor_start(authorization: Optional[str] = Header(None)):
    """Inicia o monitoramento automático da pasta."""
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
    """Para o monitoramento automático."""
    _check_auth(authorization)
    stop_monitor()
    return {"success": True, "message": "Monitor parado."}

@app.post("/batch/config/sheet")
async def config_sheet(request: Request, authorization: Optional[str] = Header(None)):
    """Cria ou retorna o ID da planilha de auditoria."""
    _check_auth(authorization)
    try:
        sheet_id = _ensure_sheet_exists()
        return {"success": True, "sheet_id": sheet_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False, "error": str(e),
        })
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════
# Auto-start do monitor (se habilitado via env var)
# ═══════════════════════════════════════════════════════════════════

if AUTO_PROCESS_ENABLED and DRIVE_FOLDER_ID and DRIVE_SERVICE_ACCOUNT_JSON:
    start_monitor()
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
