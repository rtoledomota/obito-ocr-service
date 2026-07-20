import os, io, json, re, uuid, hashlib, logging, time, traceback
from datetime import datetime, timedelta, timezone
from collections import OrderedDict
from typing import Optional
from urllib.parse import urljoin
import threading

import requests
from PIL import Image
import pytesseract
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError
from fastapi import FastAPI
from pydantic import BaseModel

# ── Config logger ────────────────────────────────────────────────
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

# ── Service account: criar arquivo a partir da env var ──────────
SERVICE_ACCOUNT_JSON_ENV = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
if SERVICE_ACCOUNT_JSON_ENV:
    sa_path = "/tmp/service-account.json"
    with open(sa_path, "w") as f:
        f.write(SERVICE_ACCOUNT_JSON_ENV)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
    logger.info("Service account criada a partir da var de ambiente.")

# ── Constantes ───────────────────────────────────────────────────
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
SHEET_ID = os.getenv("SHEET_ID", "1ETms0jR61Idqxbfr0nBdTXJGOHeGWFBomQGIZHPUJTM")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "1iwc59VnBEhjuYtW-OoOYg-UKioQ81ZfN")
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./service-account.json")

# Cabeçalhos da planilha (primeira linha)
HEADER = [
    "NOME", "NOME_SOCIAL", "NASCIMENTO", "SEXO", "RACA_COR",
    "ESTADO_CIVIL", "NACIONALIDADE", "NOME_MAE", "NOME_PAI",
    "PROFISSAO", "LOGRADOURO", "NUMERO", "COMPLEMENTO", "BAIRRO",
    "CIDADE", "UF", "CEP", "CIDADE_NASCIMENTO", "UF_NASCIMENTO",
    "CPF", "RG", "ORGAO_EMISSOR_RG", "DATA_OBITO", "HORA_OBITO",
    "LOCAL_OBITO", "CIDADE_OBITO", "UF_OBITO",
    "CAUSA_MORTE", "CAUSA_MORTE_2", "CAUSA_MORTE_3", "CAUSA_MORTE_4", "CAUSA_MORTE_5",
    "CAUSA_BASICA",
    "CODIGO_CAUSA_BASICA", "CODIGO_CAUSA_MORTE",
    "CODIGO_CAUSA_MORTE_2", "CODIGO_CAUSA_MORTE_3",
    "CODIGO_CAUSA_MORTE_4", "CODIGO_CAUSA_MORTE_5",
    "CID_BASICA", "CID_MORTE", "CID_MORTE_2", "CID_MORTE_3", "CID_MORTE_4", "CID_MORTE_5",
    "TIPO_OBITO", "ASSISTIDO", "DATA_ATESTADO",
    "NOMES_OK", "NOME_OK", "GARBAGE_CODES", "QTD_GARBAGE",
    "PROTOCOLO_TEV", "ERROS", "QUALIDADE_SCORE", "HASH_ARQUIVO",
    "HASH_CONTEUDO", "STATUS",
    "NOME_MES", "DATA_PROCESSAMENTO",
    "DO_NUMERO", "MEDICO_ATESTANTE", "CRM_MEDICO",
    "IDADE_ANOS",
    "PARTE_II", "INTERVALO_DOENCA_MORTE",
]

_lock = threading.Lock()

# ── Funções de normalização de data ──────────────────────────────

def _normalize_date_ocr(raw: str) -> str:
    if not raw or not raw.strip():
        return ""
    raw = raw.strip().replace(" ", "/").replace("-", "/").replace(".", "/")
    partes = [p for p in raw.split("/") if p.strip()]
    if len(partes) != 3:
        return ""
    p1, p2, p3 = partes[0].strip(), partes[1].strip(), partes[2].strip()
    if not p1.isdigit() or not p2.isdigit() or not p3.isdigit():
        return ""
    d, m, y = int(p1), int(p2), int(p3)
    if len(p3) == 2:
        y += 2000 if y < 50 else 1900
    if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2100:
        return f"{d:02d}/{m:02d}/{y}"
    return ""

def _normalize_date(raw: str) -> str:
    if not raw or not raw.strip():
        return ""
    raw = raw.strip()
    raw = re.sub(r'\([^)]*\)', '', raw).strip()
    raw = re.sub(r'[.\s]+', '/', raw)
    try:
        dt_obj = datetime.strptime(raw, "%d/%m/%Y")
        return raw
    except ValueError:
        pass
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            dt_obj = datetime.strptime(raw, fmt)
            if 1900 <= dt_obj.year <= datetime.now().year + 1:
                return dt_obj.strftime("%d/%m/%Y")
        except ValueError:
            continue
    nums = re.findall(r"\d+", raw)
    if len(nums) >= 3:
        for a, b in [(0, 1), (1, 0)]:
            try:
                dia, mes, ano = int(nums[a]), int(nums[b]), int(nums[-1])
                if len(nums[-1]) == 2:
                    ano += 2000 if ano < 50 else 1900
                if 1 <= dia <= 31 and 1 <= mes <= 12 and 1900 <= ano <= datetime.now().year + 1:
                    return datetime(ano, mes, dia).strftime("%d/%m/%Y")
            except (ValueError, IndexError):
                continue
    return ""

# ── Funções de validação de DO e limpeza de campos ───────────────

def _is_valid_obito(ocr_text: str) -> bool:
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

def _extract_uf_ocorrencia(text: str) -> str:
    if not text:
        return ""
    ocorrencia_match = re.search(
        r'Local de ocorrência do óbito[:\s]*\n?(.*?)(?:III[\)\.\s]|PREENCHEMENTO|IV[\)\.\s]|$)',
        text, re.DOTALL | re.IGNORECASE
    )
    if ocorrencia_match:
        secao = ocorrencia_match.group(1)
        uf_match = re.search(r'UF\s*[:\s]*([A-Z]{2})', secao)
        if uf_match:
            return uf_match.group(1).strip()
    ufs = re.findall(r'(?<!Município\s.*)UF\s*[:\s]*([A-Z]{2})', text)
    if ufs:
        return ufs[-1].strip()
    return ""

def _parse_parte_i(text: str) -> dict:
    result = {
        "CAUSA_MORTE": "", "CAUSA_MORTE_2": "", "CAUSA_MORTE_3": "",
        "CAUSA_MORTE_4": "", "CAUSA_BASICA": "",
    }
    parte_i_match = re.search(
        r'PARTE\s+I[:\s]*\n?(.*?)(?:PARTE\s+II|Intervalo|PREENCHEMENTO|$)',
        text, re.DOTALL | re.IGNORECASE
    )
    if not parte_i_match:
        parte_i_match = re.search(
            r'Causas?\s+da?\s+morte[:\s]*\n?(.*?)(?:PARTE\s+II|Outras condições|'
            r'Nome do médico|CRM|$)',
            text, re.DOTALL | re.IGNORECASE
        )
    if not parte_i_match:
        return result
    parte_i_text = parte_i_match.group(1)
    linhas = re.findall(
        r'^(?:\d+[\)\.]\s*|[a-dA-D][\)\.]\s*|[IVXivx]+[\)\.]\s*)(.+?)$',
        parte_i_text, re.MULTILINE
    )
    if not linhas:
        linhas = re.findall(
            r'(?:\d[\)\.]\s*|[a-dA-D][\)\.]\s*|I[\)\.]\s*|II[\)\.]\s*|III[\)\.]\s*|IV[\)\.]\s*)(.+)',
            parte_i_text
        )
    causas = []
    for l in linhas:
        linha = l.strip()
        if not linha or len(linha) < 3:
            continue
        if re.match(r'^(anote|preencher|não|nao|ignore|cid)', linha, re.IGNORECASE):
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

def _clean_field(value: str) -> str:
    if not value:
        return ""
    instructions = [
        r'ANOTE SOMENTE UM DIAGNÓSTICO POR LINHA',
        r'Não preencher este espaço',
        r'PREENCHEMENTO EXCLUSIVO',
        r'PREENCHEMENTO EXCLUSIVO PARA ÓBITOS FETAIS E DE ME',
        r'Menores de 1 ano:', r'Menos de 1 ano:',
        r'Escolaridade\s*\([^)]*\)',
    ]
    for instr in instructions:
        value = re.sub(instr, '', value, flags=re.IGNORECASE).strip()
    if re.match(r'^\d{4,}$', value):
        return ""
    value = re.sub(r'(\D)\d{3,}\s*$', r'\1', value).strip()
    return value.strip()

# ── Google API ───────────────────────────────────────────────────

def _get_credentials():
    return service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

def _get_drive_service():
    creds = _get_credentials()
    return build("drive", "v3", credentials=creds)

def _get_sheets_service():
    creds = _get_credentials()
    return build("sheets", "v4", credentials=creds)

def _get_existing_data():
    mapping = {}
    try:
        sheets = _get_sheets_service()
        result = sheets.spreadsheets().values().get(
            spreadsheetId=SHEET_ID, range="A:A"
        ).execute()
        values = result.get("values", [])
        for i, row in enumerate(values):
            if row and row[0] == "HASH_ARQUIVO":
                continue
            if row and row[0]:
                mapping[row[0]] = i + 1
    except Exception as e:
        logger.warning(f"Não foi possível ler dados existentes: {e}")
    return mapping

def _append_rows_to_sheet(rows):
    try:
        sheets = _get_sheets_service()
        body = {"values": rows}
        result = sheets.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range="A:A",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body=body,
        ).execute()
        return result
    except Exception as e:
        logger.error(f"Erro ao inserir na planilha: {e}")
        return None

def _download_image_bytes(file_id):
    drive = _get_drive_service()
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    metadata = drive.files().get(fileId=file_id, fields="mimeType,name").execute()
    mime_type = metadata.get("mimeType", "image/jpeg")
    return fh.getvalue(), mime_type

# ── Busca recursiva em subpastas ─────────────────────────────────

def _list_all_files_recursive(folder_id: str, drive) -> list:
    """Lista recursivamente todos os arquivos (imagens/PDFs) dentro de uma pasta e subpastas."""
    files = []
    page_token = None
    query = (f"'{folder_id}' in parents and "
             f"(mimeType contains 'image/' or mimeType='application/pdf') "
             f"and trashed=false")
    while True:
        response = drive.files().list(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, createdTime)",
            pageToken=page_token,
            orderBy="createdTime asc",
        ).execute()
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    page_token = None
    query_folders = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    while True:
        response = drive.files().list(
            q=query_folders,
            spaces="drive",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token,
        ).execute()
        subfolders = response.get("files", [])
        for sub in subfolders:
            logger.info(f"  → Explorando subpasta: {sub.get('name', 'unknown')} ({sub['id']})")
            files.extend(_list_all_files_recursive(sub["id"], drive))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return files

def _list_new_images():
    """Lista imagens no Drive (incluindo subpastas) que ainda não foram processadas."""
    try:
        drive = _get_drive_service()
        logger.info(f"Listando arquivos recursivamente a partir da pasta {DRIVE_FOLDER_ID}...")
        all_files = _list_all_files_recursive(DRIVE_FOLDER_ID, drive)
        logger.info(f"Total de arquivos encontrados (incluindo subpastas): {len(all_files)}")
        existing = _get_existing_data()
        logger.info(f"Registros existentes na planilha: {len(existing)}")
        new_files = [f for f in all_files if f["id"] not in existing]
        logger.info(f"Arquivos novos: {len(new_files)}")
        return new_files
    except Exception as e:
        logger.error(f"Erro ao listar arquivos: {e}")
        return []

# ── Funções de hash ──────────────────────────────────────────────

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ── OCR ──────────────────────────────────────────────────────────

def _ocr_image_from_bytes(image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple:
    import base64
    logger.info(f"[OCR DEBUG] Model: gpt-4o-mini, API Key set: {bool(os.getenv('OPENAI_API_KEY'))}")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    prompt_ocr = (
        "Este é um documento público oficial (Declaração de Óbito) "
        "para fins de auditoria interna hospitalar, sem divulgação externa. "
        "Transcreva exatamente todo o texto visível nesta imagem, "
        "preservando a estrutura, quebras de linha e formatação. "
        "Não resuma, não interprete, apenas transcreva."
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_ocr},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 4096,
    }
    api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 120)
                logger.warning(f"[OCR] Rate limit (429), tentativa {attempt+1}/{max_retries}, aguardando {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "choices" not in data or not data["choices"]:
                logger.error(f"[OCR ERROR] Resposta inesperada: {data}")
                return "", 0.0
            texto = data["choices"][0].get("message", {}).get("content", "")
            partes_ocr = texto.split("`" * 3)
            texto_limpo = partes_ocr[2] if len(partes_ocr) >= 3 else texto
            if not texto_limpo:
                return "", 0.0
            return texto_limpo, 1.0
        except requests.exceptions.Timeout:
            logger.error("[OCR ERROR] Timeout na requisição.")
            return "", 0.0
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1 and resp is not None and resp.status_code == 429:
                wait = min(2 ** attempt * 5, 120)
                logger.warning(f"[OCR] Rate limit (429), tentativa {attempt+1}/{max_retries}, aguardando {wait}s...")
                time.sleep(wait)
                continue
            logger.error(f"[OCR ERROR] Erro na requisição: {e}")
            return "", 0.0
    return "", 0.0

# ── Parser da Declaração de Óbito ────────────────────────────────

MONTHS_PT = {
    "janeiro": "01", "fevereiro": "02", "março": "03", "abril": "04",
    "maio": "05", "junho": "06", "julho": "07", "agosto": "08",
    "setembro": "09", "outubro": "10", "novembro": "11", "dezembro": "12",
}

def _find_block_value(text: str, labels: list, stop_labels: list = None) -> str:
    if stop_labels is None:
        stop_labels = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        for label in labels:
            idx = line.lower().find(label.lower())
            if idx != -1:
                resto = line[idx + len(label):].strip().rstrip(":")
                if resto:
                    if ":" in resto:
                        resto = resto.split(":")[-1].strip()
                    if resto and not any(sl.lower() in resto.lower() for sl in stop_labels):
                        return _clean_field(resto)
                for j in range(i + 1, min(i + 5, len(lines))):
                    candidate = lines[j].strip()
                    if candidate and not any(sl.lower() in candidate.lower() for sl in stop_labels):
                        if not re.match(r'^[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+\s*\(', candidate):
                            return _clean_field(candidate)
    return ""

def _detect_obito_type(text: str) -> str:
    if re.search(r'(?<!Não\s)(Fetal|fetal)', text) and 'Não fetal' not in text:
        return "Fetal"
    if re.search(r'Fatal|Não fetal|Não Fetal|Não\s+fetal', text, re.IGNORECASE):
        return "Fatal"
    if re.search(r'X\s*Fetal', text) and not re.search(r'X\s*Não\s+fetal', text):
        return "Fetal"
    if re.search(r'X\s*(Nao|Não)\s+fetal', text, re.IGNORECASE):
        return "Fatal"
    return ""

def parse_obito(text: str) -> dict:
    structured = {k: "" for k in HEADER}
    structured["NOME"] = _find_block_value(text, [
        "Nome do Falecido", "Nome do falecido", "Falecido", "Nome",
    ], stop_labels=["Nome do Pai", "Nome da Mãe", "Nome do pai", "Nome da mãe"])
    structured["NOME_MAE"] = _find_block_value(text, [
        "Nome da Mãe", "Nome da mãe", "Nome da Mae", "Nome da mae",
    ], stop_labels=["Nome do Pai", "Nome do pai", "Endereço", "Logradouro"])
    structured["NOME_PAI"] = _find_block_value(text, [
        "Nome do Pai", "Nome do pai",
    ], stop_labels=["Nome da Mãe", "Nome da mãe", "Endereço", "Logradouro"])
    _raw_nasc = _find_block_value(text, [
        "Data de nascimento", "Data de Nascimento", "Nascimento", "Nasc.",
    ], stop_labels=["Data do óbito", "Data do obito", "Idade"])
    structured["NASCIMENTO"] = _normalize_date(_normalize_date_ocr(_raw_nasc))
    _raw_data_obito = _find_block_value(text, [
        "Data do óbito", "Data de óbito", "Data do obito", "Data de obito",
    ], stop_labels=["Hora", "Local do óbito", "Local do obito",
                    "Município de ocorrência", "Municipio de ocorrencia"])
    if not _raw_data_obito:
        for label in ["Data do óbito", "Data de óbito", "Data do obito", "Data de obito"]:
            for line in text.split('\n'):
                if label.lower() in line.lower():
                    resto = line[line.lower().index(label.lower()) + len(label):].strip()
                    if resto:
                        _raw_data_obito = resto
                        break
            if _raw_data_obito:
                break
    structured["DATA_OBITO"] = _normalize_date(_normalize_date_ocr(_raw_data_obito))
    _raw_hora = _find_block_value(text, [
        "Hora do óbito", "Hora do obito", "Hora",
    ], stop_labels=["Data do óbito", "Data do obito", "Local do óbito", "Local do obito"])
    if _raw_hora:
        h_match = re.search(r'(\d{1,2})[:\s]*(\d{2})', _raw_hora)
        if h_match:
            structured["HORA_OBITO"] = f"{h_match.group(1).zfill(2)}:{h_match.group(2)}"
    sexo_map = {"MASCULINO": "M", "FEMININO": "F", "M": "M", "F": "F"}
    _raw_sexo = _find_block_value(text, ["Sexo", "SEXO"])
    if _raw_sexo and _raw_sexo.upper() in sexo_map:
        structured["SEXO"] = sexo_map[_raw_sexo.upper()]
    structured["RACA_COR"] = _find_block_value(text, ["Raça/Cor", "Raça", "Cor", "Raca/Cor", "Raca"])
    structured["ESTADO_CIVIL"] = _find_block_value(text, ["Situação conjugal", "Estado civil", "Estado Civil"])
    structured["ESCOLARIDADE"] = _find_block_value(text, ["Escolaridade", "Escolaridade (última série concluída)"])
    structured["PROFISSAO"] = _find_block_value(text, ["Ocupação habitual", "Profissão", "Ocupacao habitual", "Ocupação"])
    structured["LOGRADOURO"] = _find_block_value(text, ["Logradouro", "Endereço", "Endereco"])
    structured["NUMERO"] = _find_block_value(text, ["Número", "Numero"])
    structured["COMPLEMENTO"] = _find_block_value(text, ["Complemento"])
    structured["BAIRRO"] = _find_block_value(text, ["Bairro", "Bairro/Distrito"])
    structured["CIDADE"] = _find_block_value(text, ["Município de residência", "Municipio de residencia", "Município de Residência"])
    structured["UF"] = _find_block_value(text, ["UF"], stop_labels=["Local de ocorrência"])
    structured["CEP"] = _find_block_value(text, ["CEP"])
    structured["LOCAL_OBITO"] = _find_block_value(text, [
        "Local de ocorrência do óbito", "Local de ocorrência",
        "Local do óbito", "Local do obito", "Local de ocorrencia do obito",
    ])
    structured["CIDADE_OBITO"] = _find_block_value(text, [
        "Município de ocorrência", "Municipio de ocorrencia", "Município de Ocorrência",
    ])
    structured["UF_OBITO"] = _extract_uf_ocorrencia(text)
    structured["TIPO_OBITO"] = _detect_obito_type(text)
    causas = _parse_parte_i(text)
    structured["CAUSA_MORTE"] = causas.get("CAUSA_MORTE", "")
    structured["CAUSA_MORTE_2"] = causas.get("CAUSA_MORTE_2", "")
    structured["CAUSA_MORTE_3"] = causas.get("CAUSA_MORTE_3", "")
    structured["CAUSA_MORTE_4"] = causas.get("CAUSA_MORTE_4", "")
    structured["CAUSA_BASICA"] = causas.get("CAUSA_BASICA", "")
    structured["MEDICO_ATESTANTE"] = _find_block_value(text, [
        "Médico", "Medico", "Nome do Médico", "Nome do medico",
    ], stop_labels=["CRM"])
    structured["CRM_MEDICO"] = _find_block_value(text, ["CRM"])
    do_match = re.search(r'Declaração\s+de\s+Óbito\s+(\d+(?:-\d+)?)', text, re.IGNORECASE)
    if do_match:
        structured["DO_NUMERO"] = do_match.group(1)
    parte_ii_match = re.search(
        r'PARTE\s+II[:\s]*\n?(.*?)(?:Outros episódios|Nome do médico|CRM|$)',
        text, re.DOTALL | re.IGNORECASE
    )
    if parte_ii_match:
        structured["PARTE_II"] = _clean_field(parte_ii_match.group(1).strip()[:200])
        structured["INTERVALO_DOENCA_MORTE"] = structured["PARTE_II"]
    idade_raw = _find_block_value(text, ["Idade", "IDADE"])
    if idade_raw:
        nums = re.findall(r'\d+', idade_raw)
        if nums:
            structured["IDADE_ANOS"] = nums[0]
    text_fields = [
        "NOME", "NOME_MAE", "NOME_PAI", "PROFISSAO",
        "LOGRADOURO", "BAIRRO", "CIDADE", "CIDADE_OBITO",
        "CAUSA_MORTE", "CAUSA_MORTE_2", "CAUSA_MORTE_3", "CAUSA_MORTE_4",
        "CAUSA_BASICA", "LOCAL_OBITO", "MEDICO_ATESTANTE",
        "PARTE_II", "INTERVALO_DOENCA_MORTE"
    ]
    for campo in text_fields:
        if campo in structured and structured[campo]:
            structured[campo] = _clean_field(structured[campo])
    return structured

# ── Validação ────────────────────────────────────────────────────

CRITICAL_FIELDS = ["NOME", "NOME_MAE", "NASCIMENTO", "DATA_OBITO",
                   "CIDADE_OBITO", "UF_OBITO", "CAUSA_MORTE"]

def validate_obito(structured: dict) -> None:
    missing_critical = []
    for field in CRITICAL_FIELDS:
        if not structured.get(field):
            missing_critical.append(field)
    raw_text_for_garbage = structured.get("GARBAGE_CODES", "")
    qtd_garbage = 0
    if raw_text_for_garbage:
        qtd_garbage = len(raw_text_for_garbage)
    total_fields = len(CRITICAL_FIELDS)
    filled_fields = sum(1 for f in CRITICAL_FIELDS if structured.get(f))
    score = round((filled_fields / total_fields) * 100, 1) if total_fields > 0 else 0
    structured["QUALIDADE_SCORE"] = str(score)
    structured["QTD_GARBAGE"] = str(qtd_garbage)
    if missing_critical:
        structured["STATUS"] = "REVISAR"
        structured["ERROS"] = " | ".join(f"Campo crítico ausente: {f}" for f in missing_critical)
    else:
        structured["STATUS"] = "OK"
        structured["ERROS"] = ""

# ── Processamento Individual ─────────────────────────────────────

def _process_single_image(file_id: str, file_name: str) -> dict:
    logger.info(f"Processando: {file_name} ({file_id})")
    try:
        image_bytes, mime_type = _download_image_bytes(file_id)
    except Exception as e:
        return {"NOME_ARQUIVO": file_name, "STATUS": "ERRO_DRIVE", "ERROS": str(e)}
        try:
        raw_text, confidence = _ocr_image_from_bytes(image_bytes, mime_type)
        # ── ADICIONE ESTAS 3 LINHAS AQUI ──
        if raw_text:
            logger.info(f"[OCR RESPONSE] {file_name}: primeiros 500 chars: {raw_text[:500]}")
        else:
            logger.warning(f"[OCR RESPONSE] {file_name}: texto vazio retornado")
        # ── FIM DAS 3 LINHAS ──
    except Exception as e:
        return {"NOME_ARQUIVO": file_name, "STATUS": "ERRO_OCR", "ERROS": str(e)}
    if not _is_valid_obito(raw_text):
        logger.warning(f"{file_name}: texto não reconhecido como DO, pulando")
        return {"NOME_ARQUIVO": file_name, "STATUS": "REJEITADO", "ERROS": "Imagem não contém uma Declaração de Óbito válida"}
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
        "NASCIMENTO": _normalize_date(structured.get("NASCIMENTO", "")),
        "DATA_OBITO": _normalize_date(structured.get("DATA_OBITO", "")),
        "HORA_OBITO": structured.get("HORA_OBITO", ""),
        "CIDADE_OBITO": structured.get("CIDADE_OBITO", ""),
        "UF_OBITO": structured.get("UF_OBITO", ""),
        "CAUSA_MORTE": structured.get("CAUSA_MORTE", ""),
        "CAUSA_BASICA": structured.get("CAUSA_BASICA", ""),
        "CID_BASICA": structured.get("CID_BASICA", ""),
        "TIPO_OBITO": structured.get("TIPO_OBITO", ""),
        "ERROS": structured.get("ERROS", ""),
        "HASH_ARQUIVO": structured.get("HASH_ARQUIVO", ""),
        "DO_NUMERO": structured.get("DO_NUMERO", ""),
        "MEDICO_ATESTANTE": structured.get("MEDICO_ATESTANTE", ""),
        "CRM_MEDICO": structured.get("CRM_MEDICO", ""),
        "IDADE_ANOS": structured.get("IDADE_ANOS", ""),
        "PARTE_II": structured.get("PARTE_II", ""),
        "INTERVALO_DOENCA_MORTE": structured.get("INTERVALO_DOENCA_MORTE", ""),
    }
    return row

# ── Batch ────────────────────────────────────────────────────────

def run_batch(limit: int = 10) -> dict:
    logger.info(f"Iniciando batch com limit={limit}")
    try:
        all_images = _list_new_images()
        if not all_images:
            return {
                "success": True, "total": 0, "new": 0, "processed": 0,
                "failed": 0, "sheet_id": SHEET_ID,
                "message": "Nenhuma imagem nova para processar.",
                "requestId": str(uuid.uuid4()),
            }
        total = len(all_images)
        to_process = all_images[:limit]
        processed_count = 0
        failed_ids = []
        rows_to_insert = []
        for img in to_process:
            time.sleep(1)  # ← delay de 1s entre imagens para evitar rate limit
            file_id = img["id"]
            file_name = img.get("name", "unknown")
            row = _process_single_image(file_id, file_name)
            if row.get("STATUS") == "OK":
                processed_count += 1
                rows_to_insert.append([row.get(h, "") for h in HEADER])
            elif row.get("STATUS") == "REJEITADO":
                logger.info(f"{file_name}: {row.get('ERROS', 'rejeitada')}")
            else:
                failed_ids.append(file_name)
                rows_to_insert.append([row.get(h, "") for h in HEADER])
        if rows_to_insert:
            result = _append_rows_to_sheet(rows_to_insert)
            if result:
                logger.info(f"Inseridas {len(rows_to_insert)} linhas na planilha.")
            else:
                logger.error("Falha ao inserir linhas na planilha.")
        msg = f"{processed_count} imagens processadas, {len(failed_ids)} falhas."
        if failed_ids:
            msg += f" IDs com falha: {', '.join(failed_ids[:5])}"
        return {
            "success": True, "total": total, "new": len(to_process),
            "processed": processed_count, "failed": len(failed_ids),
            "sheet_id": SHEET_ID, "message": msg, "requestId": str(uuid.uuid4()),
        }
    except Exception as e:
        logger.error(f"Erro no batch: {e}", exc_info=True)
        return {"success": False, "error": str(e), "message": "Erro interno no batch"}

# ── FastAPI App ──────────────────────────────────────────────────

app = FastAPI(title="Óbito OCR Service", version="2.0")

class BatchRequest(BaseModel):
    limit: int = 10

@app.get("/")
def root():
    return {"status": "running", "service": "Óbito OCR Service"}

@app.post("/batch/process")
def batch_process(request: BatchRequest):
    return run_batch(limit=request.limit)

@app.post("/batch/reprocess")
def batch_reprocess(limit: int = 10):
    """Reprocessa imagens ignorando verificação de duplicatas (busca recursiva em subpastas)."""
    logger.info(f"Iniciando reprocessamento com limit={limit}")
    try:
        drive = _get_drive_service()
        logger.info(f"Reprocess: listando recursivamente da pasta {DRIVE_FOLDER_ID}...")
        all_files = _list_all_files_recursive(DRIVE_FOLDER_ID, drive)
        total = len(all_files)
        to_process = all_files[:limit]
        processed_count = 0
        failed_ids = []
        rows_to_insert = []
        for img in to_process:
            time.sleep(1)  # ← delay de 1s entre imagens para evitar rate limit
            file_id = img["id"]
            file_name = img.get("name", "unknown")
            row = _process_single_image(file_id, file_name)
            if row.get("STATUS") == "OK":
                processed_count += 1
                rows_to_insert.append([row.get(h, "") for h in HEADER])
            elif row.get("STATUS") == "REJEITADO":
                logger.info(f"{file_name}: {row.get('ERROS', 'rejeitada')}")
            else:
                failed_ids.append(file_name)
                rows_to_insert.append([row.get(h, "") for h in HEADER])
        if rows_to_insert:
            result = _append_rows_to_sheet(rows_to_insert)
            if result:
                logger.info(f"Inseridas {len(rows_to_insert)} linhas na planilha.")
            else:
                logger.error("Falha ao inserir linhas na planilha.")
        msg = f"{processed_count} imagens processadas, {len(failed_ids)} falhas."
        if failed_ids:
            msg += f" IDs com falha: {', '.join(failed_ids[:5])}"
        return {
            "success": True, "total": total, "new": len(to_process),
            "processed": processed_count, "failed": len(failed_ids),
            "sheet_id": SHEET_ID, "message": msg, "requestId": str(uuid.uuid4()),
        }
    except Exception as e:
        logger.error(f"Erro no reprocessamento: {e}", exc_info=True)
        return {"success": False, "error": str(e), "message": "Erro interno no reprocessamento"}
