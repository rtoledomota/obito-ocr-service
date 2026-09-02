import os, io, re, uuid, hashlib, logging, time, base64
from datetime import datetime

import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

# ── Service account: suporta JSON inline via env var ─────────────
_SERVICE_ACCOUNT_JSON_ENV = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
if _SERVICE_ACCOUNT_JSON_ENV and _SERVICE_ACCOUNT_JSON_ENV.strip().startswith("{"):
    _sa_path = "/tmp/service-account.json"
    with open(_sa_path, "w") as f:
        f.write(_SERVICE_ACCOUNT_JSON_ENV)
    os.environ["SERVICE_ACCOUNT_FILE"] = _sa_path
    logger.info("Service account criada a partir de GOOGLE_SERVICE_ACCOUNT_JSON.")

def _find_service_account():
    candidates = [
        os.getenv("SERVICE_ACCOUNT_FILE", ""),
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        "./service-account.json",
        "./src/service-account.json",
        "/opt/render/project/src/service-account.json",
        "/opt/render/project/service-account.json",
        "/etc/secrets/service-account.json",
    ]
    for cand in candidates:
        if not cand or cand.startswith("AIza"):
            continue  # ignora API key usada como valor da env var
        if os.path.exists(cand):
            return cand
    return "./service-account.json"

SERVICE_ACCOUNT_FILE = _find_service_account()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
SHEET_ID = os.getenv("SHEET_ID", "1ETms0jR61Idqxbfr0nBdTXJGOHeGWFBomQGIZHPUJTM")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "1iwc59VnBEhjuYtW-OoOYg-UKioQ81ZfN")

# Chaves Vision: tenta em ordem ate uma funcionar
VISION_KEYS = [k for k in [
    os.getenv("GOOGLE_VISION_API_KEY", ""),
    os.getenv("VISION_KEY", ""),
    "AIzaSyCKXbsF8UNL0WSoE4FAh4yVXJ1f9Y2tiSU",   # chave da aba Config
    "AIzaSyB-45eklgCjDOI9XXTqNRFkZtQmCOywgYM",   # chave antiga hardcoded
] if k]

# HEADER REAL da aba Auditoria (A-W)
HEADER = [
    "DATA_PROCESSAMENTO", "NOME_ARQUIVO", "STATUS", "QUALIDADE_SCORE",
    "NOME", "NOME_MAE", "NASCIMENTO", "IDADE_ANOS", "DATA_OBITO", "HORA_OBITO",
    "CIDADE_OBITO", "UF_OBITO", "CAUSA_MORTE", "CAUSA_BASICA", "CID_BASICA",
    "TIPO_OBITO", "DO_NUMERO", "MEDICO_ATESTANTE", "CRM_MEDICO",
    "PARTE_II", "INTERVALO_DOENCA_MORTE", "ERROS", "HASH_ARQUIVO",
]

FORM_JUNK = [
    "nome do falecido", "data de nascimento", "data do óbito", "data do obito",
    "endereço do local do acidente", "descrição sumária do evento",
    "complexo hospitalar de clínicas", "identificação", "cartório",
    "município de residência", "município de ocorrência", "local de ocorrência",
    "nome do médico", "situação conjugal", "raça/cor", "ocupação habitual",
    "logradouro (rua", "bairro/distrito", "nome do pai", "nome da mãe",
    "data do atestado", "assistência médica", "parte ii", "causas da morte",
    "condições e causas", "causas externas", "fetal ou menor",
    "preenchimento exclusivo", "republica federativa", "ministerio da saude",
    "declaracao de obito", "declaração de óbito", "seqüência de causas",
    "sequencia de causas", "preencha o estado", "anote a cadeia",
    "anote somente", "devido ou como consequência", "tempo aproximado entre",
    "que contribuíram", "outras condições significativas",
    "nascidos vivos", "número de filhos", "tipo de gravidez", "tipo de parto",
    "peso ao nascer", "cartão sus", "naturalidade", "escolaridade",
    "meio de contato", "morte em relação", "óbito de mulher",
]

CITY_OCR_FIX = {
    "são cantando do sul": "São Caetano do Sul",
    "são cenário do sul": "São Caetano do Sul",
    "são caetano sul": "São Caetano do Sul",
    "caetano do sul": "São Caetano do Sul",
    "são carlos do sul": "São Caetano do Sul",
    "são gabriel do sul": "São Caetano do Sul",
    "são bernardo do sul": "São Caetano do Sul",
    "são lourenço do sul": "São Caetano do Sul",
    "santo antônio do sul": "São Caetano do Sul",
    "santa paula": "São Caetano do Sul",
    "são paulo - sp": "São Paulo",
}

VALID_UFS = {"AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG",
             "PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"}

CRITICAL_FIELDS = ["NOME", "NOME_MAE", "NASCIMENTO", "DATA_OBITO",
                   "CIDADE_OBITO", "UF_OBITO", "CAUSA_MORTE"]

# ── Google API ───────────────────────────────────────────────────

def _get_credentials():
    return service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

def _get_drive_service():
    return build("drive", "v3", credentials=_get_credentials())

def _get_sheets_service():
    return build("sheets", "v4", credentials=_get_credentials())

def _get_existing_data():
    """Retorna hashes (coluna W) e nomes de arquivo (coluna B) ja processados."""
    hashes, names = {}, set()
    try:
        sheets = _get_sheets_service()
        result = sheets.spreadsheets().values().get(
            spreadsheetId=SHEET_ID, range="Auditoria!B1:W1579"
        ).execute()
        for row in result.get("values", []):
            if not row:
                continue
            name = str(row[0]).strip() if len(row) > 0 else ""
            h = str(row[21]).strip() if len(row) > 21 else ""
            if name and name != "NOME_ARQUIVO":
                names.add(name)
            if h and h != "HASH_ARQUIVO":
                hashes[h] = True
    except Exception as e:
        logger.warning(f"Nao foi possivel ler dados existentes: {e}")
    return {"hashes": hashes, "names": names}

def _ensure_sheet_header():
    try:
        sheets = _get_sheets_service()
        result = sheets.spreadsheets().values().get(
            spreadsheetId=SHEET_ID, range="Auditoria!A1:W1"
        ).execute()
        vals = result.get("values", [])
        if vals and vals[0] and any(str(v).strip() for v in vals[0] if v):
            logger.info("Cabecalho ja existe na planilha.")
            return True
        sheets.spreadsheets().values().update(
            spreadsheetId=SHEET_ID, range="Auditoria!A1",
            valueInputOption="USER_ENTERED",
            body={"values": [HEADER]}
        ).execute()
        logger.info("Cabecalho escrito na planilha!")
        return True
    except Exception as e:
        logger.error(f"Erro ao garantir cabecalho: {e}")
        return False

def _append_rows_to_sheet(rows):
    try:
        sheets = _get_sheets_service()
        return sheets.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range="Auditoria!A:A",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": rows},
        ).execute()
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
    return fh.getvalue(), metadata.get("mimeType", "image/jpeg")

# ── Busca recursiva em subpastas ─────────────────────────────────

def _list_all_files_recursive(folder_id, drive):
    files = []
    page_token = None
    query = (f"'{folder_id}' in parents and "
             f"(mimeType contains 'image/' or mimeType='application/pdf') "
             f"and trashed=false")
    while True:
        response = drive.files().list(
            q=query, spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, createdTime)",
            pageToken=page_token, orderBy="createdTime asc",
        ).execute()
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    page_token = None
    query_folders = (f"'{folder_id}' in parents and "
                     f"mimeType='application/vnd.google-apps.folder' and trashed=false")
    while True:
        response = drive.files().list(
            q=query_folders, spaces="drive",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token,
        ).execute()
        for sub in response.get("files", []):
            logger.info(f"  -> Explorando subpasta: {sub.get('name','unknown')}")
            files.extend(_list_all_files_recursive(sub["id"], drive))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return files

# ── Hash ─────────────────────────────────────────────────────────

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ── OCR via Google Cloud Vision REST API (multi-chave) ───────────

def _ocr_image_from_bytes(image_bytes, mime_type="image/jpeg"):
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    gemini_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    if not gemini_key:
        logger.error("[OCR] GEMINI_API_KEY nao configurada")
        return "", 0.0
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={gemini_key}"
    prompt = (
        "Copie TODO o texto desta Declaracao de Obito EXATAMENTE como aparece, "
        "sem resumir, sem bullets, sem markdown, sem marcar campos como [Blank]. "
        "Transcreva cada rotulo e cada valor preenchido a mao, na ordem em que aparecem. "
        "NAO omita nenhuma linha. Inclua nome do falecido, data de nascimento, data do obito, municipio, UF e todas as causas da morte. Responda apenas com o texto transcrito, sem comentarios."
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": img_b64}},
            ]
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
    }
    try:
        resp = requests.post(url, json=payload, timeout=90)
        if resp.status_code != 200:
            err = resp.json().get("error", {}).get("message", "")
            logger.error(f"[OCR GEMINI] HTTP {resp.status_code}: {err}")
            return "", 0.0
        data = resp.json()
        # DUPLA LEITURA: roda o Gemini 2x e mescla os textos
        try:
            resp2 = requests.post(url, json=payload, timeout=90)
            if resp2.status_code == 200:
                data2 = resp2.json()
                parts2 = data2.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text2 = "".join(p.get("text", "") for p in parts2).strip()
                if len(text2) > len(text):
                    text = text2
        except Exception:
            pass
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        logger.info(f"[OCR GEMINI] OK - texto: {len(text)} chars")
        return text, 1.0
    except Exception as e:
        logger.error(f"[OCR GEMINI] erro: {e}")
        return "", 0.0
# Validacao / limpeza

def _is_valid_obito(ocr_text: str) -> bool:
    if not ocr_text or len(ocr_text.strip()) < 50:
        return False
    t = ocr_text.lower()
    if ("definiÃ§Ãµes" in t or "definicoes" in t) and ("cid-10" in t or "nascimento vivo" in t):
        return False
    strong = ["declaração de óbito", "declaracao de obito", "atestado de óbito",
              "nome do falecido", "causas da morte", "tipo de óbito",
              "tipo de obito", "parte i", "parte ii"]
    if any(k in t for k in strong):
        return True
    if len(ocr_text.strip()) > 400 and ("óbito" in t or "obito" in t):
        return True
    return False

def _clean_field(value: str) -> str:
    if not value:
        return ""
    v = str(value).strip()
    v = re.sub(r'\([^)]*\)', '', v).strip()
    if re.fullmatch(r'\d{4,}', v):
        return ""
    v = re.sub(r'(\D)\d{3,}\s*$', r'\1', v).strip()
    return v.strip()

def _collapse_repeats(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r'\b([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇ]{8,})\s+\1\b', r'\1', text)
    return text

def _sanitize_person_name(value: str) -> str:
    if not value:
        return ""
    v = str(value).strip().rstrip("|.,;:")
    v = re.sub(r'\s*Munic[ií]pio\s*/\s*UF\s*\(se\s+estrangeiro\s+informar\s+Pa[ií]s\)[:.\s]*$', '', v, flags=re.IGNORECASE)
    v = re.sub(r'^Munic[ií]pio\s*/\s*UF\s*\(se\s+estrangeiro\s+informar\s+Pa[ií]s\)[:.\s]*', '', v, flags=re.IGNORECASE)
    v = re.sub(r'^(nome do falecido|nome do\(a\)|falecido|nome|data de nascimento'
               r'|nome do pai|nome da mae|nome da mãe)\s*[:\-]?\s*', '', v, flags=re.IGNORECASE)
    if re.match(r'^\d{1,2}\s+[A-Za-zÀ-ÿ]', v):
        v = re.sub(r'^\d{1,2}\s+', '', v)
    v = re.sub(r'\s+(Identificação|Cartório|Médico|Nome do Pai|Nome da Mãe).*$',
               '', v, flags=re.IGNORECASE)
    v = _collapse_repeats(v).strip()
    low = v.lower()
    if any(j in low for j in FORM_JUNK):
        return ""
    if re.fullmatch(r'\d+', v) or len(v) < 5:
        return ""
    return v

def _clean_causa(value: str) -> str:
    if not value:
        return ""
    v = str(value).strip()
    v = re.sub(r'^\(?(?:a\s+)?\w+?\s+ou\s+estado\s+\w+?\s+que\s+causou\s+diretamente\s+a(?:\s+\w+)?\)?[: ]*', '', v, flags=re.IGNORECASE)
    v = re.sub(r'^[\u2022\u00b7\-\*]\s*', '', v).strip()
    v = re.sub(r'^99\s*ignorado(\s*99\s*ignorado)?', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^(?:causa\s+imediata|imediata)\s*[: ]*', '', v, flags=re.IGNORECASE).strip()
    v = re.sub(r'^a\s+(?=[A-Z\u00c0-\u00da])', '', v).strip()
    v = re.sub(r'^(?:parte\s+[iv]+)\s*[: ]*', '', v, flags=re.IGNORECASE).strip()

    v = re.sub(r'^\(?a doença ou estado mórbido que causou diretamente a morte\)?[: ]*',
               '', v, flags=re.IGNORECASE)
    v = re.sub(r'^(seqüência|sequencia) de causas[^:]*[: ]*', '', v, flags=re.IGNORECASE)
    v = re.sub(r'^(devido ou como consequência de)[: ]*', '', v, flags=re.IGNORECASE)
    v = re.sub(r'^(causa básica|causa basica)[: ]*', '', v, flags=re.IGNORECASE)
    v = v.strip()
    if re.fullmatch(r'\d+\s*(anos?|meses?|dias?|horas?|min\w*)?', v, re.IGNORECASE):
        return ""
    low = v.lower()
    if any(j in low for j in FORM_JUNK):
        return ""
    return v

def _normalize_date_ocr(raw: str) -> str:
    if not raw or not raw.strip():
        return ""
    raw = raw.strip()
    raw = re.sub(r"\s+", "", raw)
    raw = raw.replace(",", "/").replace("-", "/").replace(".", "/")
    partes = [p for p in raw.split("/") if p.strip()]
    if len(partes) != 3:
        nums = re.findall(r"\d+", raw)
        for n in nums:
            if len(n) == 8:
                d, m, a = int(n[0:2]), int(n[2:4]), int(n[4:8])
                if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= a <= 2100:
                    return f"{d:02d}/{m:02d}/{a}"
        return ""
    p1, p2, p3 = partes[0].strip(), partes[1].strip(), partes[2].strip()
    if not (p1.isdigit() and p2.isdigit() and p3.isdigit()):
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
    raw = re.sub(r'\([^)]*\)', '', raw.strip())
    raw = re.sub(r'[.\s]+', '/', raw)
    try:
        datetime.strptime(raw, "%d/%m/%Y")
        return raw
    except ValueError:
        pass
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(raw, fmt)
            if 1900 <= dt.year <= datetime.now().year + 1:
                return dt.strftime("%d/%m/%Y")
        except ValueError:
            continue
    return ""

def _normalize_hora(raw: str) -> str:
    if not raw:
        return ""
    m = re.search(r'(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)', str(raw))
    if not m:
        m = re.search(r'(?<!\d)([01]?\d|2[0-3])[hH]([0-5]\d)(?!\d)', str(raw))
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return ""

def _normalize_cidade(value: str) -> str:
    if not value:
        return ""
    v = str(value).strip().rstrip("|.,;:")
    v = re.sub(r'\s*Munic[ií]pio\s*/\s*UF\s*\(se\s+estrangeiro\s+informar\s+Pa[ií]s\)[:.\s]*$', '', v, flags=re.IGNORECASE)
    v = re.sub(r'^Munic[ií]pio\s*/\s*UF\s*\(se\s+estrangeiro\s+informar\s+Pa[ií]s\)[:.\s]*', '', v, flags=re.IGNORECASE)
    low = v.lower()
    if low in CITY_OCR_FIX:
        return CITY_OCR_FIX[low]
    if v.upper() in {"UF", "CEP", "MÉDICO", "MEDICO", "CARTÓRIO", "LOCAL DE OCORRÊNCIA"}:
        return ""
    if re.fullmatch(r'\d+', v) or len(v) < 3:
        return ""
    if any(j in low for j in FORM_JUNK):
        return ""
    return v

def _normalize_uf(value: str) -> str:
    if not value:
        return ""
    uf = str(value).strip().upper()
    uf = {"IC": "SP"}.get(uf, uf)
    return uf if uf in VALID_UFS else ""

# ── Parser da DO ─────────────────────────────────────────────────

def _find_block_value(text: str, labels: list, stop_labels: list = None) -> str:
    if stop_labels is None:
        stop_labels = []
    skip_headers = {
        "identificacao", "residencia", "ocorrencia", "cartorio", "medico",
        "causas externas", "condicoes e causas do obito",
        "fetal ou menor que 1 ano", "declaracao de obito",
        "republica federativa do brasil", "ministerio da saude",
        "hora", "1a via secretaria de saude", "via secretaria de saude",
        "tipo de obito", "data do obito", "nome do falecido",
        "nome do pai", "nome da mae", "cartao sus", "naturalidade",
    }
    lines = text.split("\n")
    for i, line in enumerate(lines):
        line_clean = line.strip()
        for label in labels:
            idx = line_clean.lower().find(label.lower())
            if idx == -1:
                continue
            resto = line_clean[idx + len(label):].strip().rstrip(":")
            if resto and not any(sl.lower() in resto.lower() for sl in stop_labels):
                if ":" in resto:
                    resto = resto.split(":")[-1].strip()
                val = _clean_field(resto)
                if val and len(val) > 1:
                    return val
            for j in range(i + 1, min(i + 25, len(lines))):
                candidate = lines[j].strip()
                if not candidate or len(candidate) < 2:
                    continue
                if re.match(r"^\d+\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ]", candidate):
                    continue
                cand_lower = candidate.lower().strip("|.,;:")
                if cand_lower in skip_headers:
                    continue
                if any(junk in cand_lower for junk in FORM_JUNK):
                    continue
                if len(candidate.split()) <= 1 and len(candidate) < 5:
                    continue
                if any(sl.lower() in candidate.lower() for sl in stop_labels):
                    break
                val = _clean_field(candidate)
                if val:
                    return val
            break
    return ""

def _find_name_fallback(text: str) -> str:
    lines = text.split("\n")
    candidates = []
    for line in lines:
        line = line.strip().rstrip("|.,;:")
        if not line or len(line) < 10:
            continue
        if re.match(r"^\d+\s", line):
            continue
        low = line.lower()
        if any(j in low for j in FORM_JUNK):
            continue
        if sum(1 for c in line if not c.isalpha() and not c.isspace()) > len(line) * 0.3:
            continue
        words = line.split()
        if len(words) < 2:
            continue
        parts = {"de", "da", "do", "das", "dos", "e", "van", "von"}
        caps = 0
        ok = True
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
    if not candidates:
        return ""
    best = max(candidates, key=len)
    return _sanitize_person_name(best)

def _parsed_do_form(lines: list) -> dict:
    field_map = {1: "TIPO_OBITO", 2: "DATA_HORA_OBITO", 3: "CARTAO_SUS",
                 4: "NATURALIDADE", 5: "NOME", 6: "NOME_PAI", 7: "NOME_MAE"}
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
        txt = field_values[2]
        dm = re.search(r"(\d{2})(\d{2})(\d{4})", txt)
        if dm and 1 <= int(dm.group(2)) <= 12 and 1 <= int(dm.group(1)) <= 31:
            result["DATA_OBITO"] = f"{int(dm.group(1)):02d}/{int(dm.group(2)):02d}/{dm.group(3)}"
        hm = re.search(r"(\d{1,2}):(\d{2})", txt)
        if hm:
            hora = _normalize_hora(f"{hm.group(1)}:{hm.group(2)}")
            if hora:
                result["HORA_OBITO"] = hora
    if 5 in field_values:
        nome = _sanitize_person_name(field_values[5].replace("\n", " "))
        if nome:
            result["NOME"] = nome
    if 7 in field_values:
        mae = _sanitize_person_name(field_values[7].replace("\n", " "))
        if mae:
            result["NOME_MAE"] = mae
    return result

def _detect_obito_type(text: str) -> str:
    if re.search(r'X\s*(Nao|Não)\s*fetal', text, re.IGNORECASE):
        return "Não Fetal"
    if re.search(r'X\s*Fetal', text) and not re.search(r'X\s*(Nao|Não)\s*fetal', text, re.IGNORECASE):
        return "Fetal"
    if re.search(r'\bFatal\b', text, re.IGNORECASE):
        return "Não Fetal"
    if re.search(r'\b(Nao|Não)\s*fetal\b', text, re.IGNORECASE):
        return "Não Fetal"
    if re.search(r'\bFetal\b', text, re.IGNORECASE):
        return "Fetal"
    return ""

def _extract_uf_ocorrencia(text: str) -> str:
    if not text:
        return ""
    ocorrencia_match = re.search(
        r'Local de ocorrência do óbito[:\s]*\n?(.*?)(?:III[\)\.\s]|PREENCHEMENTO|IV[\)\.\s]|$)',
        text, re.DOTALL | re.IGNORECASE
    )
    if ocorrencia_match:
        uf_match = re.search(r'UF\s*[:\s]*([A-Z]{2})', ocorrencia_match.group(1))
        if uf_match:
            return _normalize_uf(uf_match.group(1))
    ufs = re.findall(r'(?<!Município\s.*)UF\s*[:\s]*([A-Z]{2})', text)
    if ufs:
        return _normalize_uf(ufs[-1])
    return ""

def _parse_parte_i(text: str) -> dict:
    result = {"CAUSA_MORTE": "", "CAUSA_MORTE_2": "", "CAUSA_MORTE_3": "",
              "CAUSA_MORTE_4": "", "CAUSA_BASICA": ""}
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
    linhas = re.findall(
        r'^(?:\d+[\)\.]\s*|[a-dA-D][\)\.]\s*|[IVXivx]+[\)\.]\s*)(.+?)$',
        parte_i_match.group(1), re.MULTILINE
    ) or re.findall(
        r'(?:\d[\)\.]\s*|[a-dA-D][\)\.]\s*|I[\)\.]\s*|II[\)\.]\s*|III[\)\.]\s*|IV[\)\.]\s*)(.+)',
        parte_i_match.group(1)
    )
    causas = []
    for l in linhas:
        c = _clean_causa(l)
        if c and len(c) >= 3:
            causas.append(c)
    if not causas:
        return result
    result["CAUSA_MORTE"] = causas[0]
    result["CAUSA_BASICA"] = causas[-1]
    if len(causas) > 1:
        result["CAUSA_MORTE_2"] = causas[1]
    if len(causas) > 2:
        result["CAUSA_MORTE_3"] = causas[2]
    if len(causas) > 3:
        result["CAUSA_MORTE_4"] = causas[3]
    return result

def parse_obito(text: str) -> dict:
    structured = {k: "" for k in HEADER}

    nome = _sanitize_person_name(_find_block_value(text, [
        "Nome do Falecido", "Nome do falecido", "Falecido", "Nome",
    ], stop_labels=["Nome do Pai", "Nome da Mãe", "Nome do pai", "Nome da mãe"]))
    if not nome:
        nome = _sanitize_person_name(_parsed_do_form(text.split("\n")).get("NOME", ""))
    if not nome:
        nome = _find_name_fallback(text)
    structured["NOME"] = nome
# --- Ponto 1a: limpar nome poluido com labels do formulario ---
if nome and any(j in nome.lower() for j in FORM_JUNK):
    _labels = re.compile(
        r'descri[çc][ãa]o sum[áa]ria|endere[çc]o|logradouro|n[uú]mero|bairro|'
        r'munic[ií]pio|c[oó]digo|registro|complemento|cart[oó]rio|'
        r'ocupa[çc][ãa]o habitual|situa[çc][ãa]o conjugal|ra[çc]a/cor|'
        r'naturalidade|escolaridade|meio de contato|identifica[çc][ãa]o', re.IGNORECASE)
    _partes = [p.strip().rstrip("|.,;:") for p in _labels.split(nome) if p.strip()]
    _nome_limpo = ""
    for _p in reversed(_partes):
        if len(_p.split()) >= 2 and not re.fullmatch(r'\d+', _p) \
           and not any(j in _p.lower() for j in FORM_JUNK):
            _nome_limpo = _p
            break
    structured["NOME"] = _sanitize_person_name(_nome_limpo) if _nome_limpo else ""

    mae = _sanitize_person_name(_find_block_value(text, [
        "Nome da Mãe", "Nome da mãe", "Nome da Mae", "Nome da mae",
    ], stop_labels=["Nome do Pai", "Nome do pai", "Endereço", "Logradouro"]))
    if not mae:
        mae = _sanitize_person_name(_parsed_do_form(text.split("\n")).get("NOME_MAE", ""))
    structured["NOME_MAE"] = mae

    _raw_nasc = _find_block_value(text, [
        "Data de nascimento", "Data de Nascimento", "Nascimento", "Nasc.",
    ], stop_labels=["Data do óbito", "Data do obito", "Idade"])
    structured["NASCIMENTO"] = _normalize_date(_normalize_date_ocr(_raw_nasc))
    if not structured["NASCIMENTO"]:
        _lines_t = text.split("\n")
        for i, line in enumerate(_lines_t):
            if "data de nascimento" in line.lower():
                for j in range(i, min(i + 4, len(_lines_t))):
                    cand = _lines_t[j].replace(" ", "")
                    for d in re.findall(r"\d{2}/\d{2}/\d{4}", cand):
                        parsed = _normalize_date(d)
                        if parsed:
                            structured["NASCIMENTO"] = parsed
                            break
                    if structured["NASCIMENTO"]:
                        break
            if structured["NASCIMENTO"]:
                break

        _raw_data_obito = ""
    for label in ["Data do óbito", "Data de óbito", "Data do obito", "Data de obito"]:
        for line in text.split('\n'):
            if label.lower() in line.lower():
                resto = line[line.lower().index(label.lower()) + len(label):].strip()
                m = re.search(r'(\d{1,2})[/\s](\d{1,2})[/\s](\d{2,4})', resto)
                if m:
                    _raw_data_obito = f"{m.group(1)} {m.group(2)} {m.group(3)}"
                hm = re.search(r'(\d{1,2}):(\d{2})', resto)
                if hm:
                    structured["HORA_OBITO"] = _normalize_hora(f"{hm.group(1)}:{hm.group(2)}")
                break
        if _raw_data_obito:
            break
    structured["DATA_OBITO"] = _normalize_date(_normalize_date_ocr(_raw_data_obito))

    _raw_hora = _find_block_value(text, [
        "Hora do óbito", "Hora do obito", "Hora",
    ], stop_labels=["Data do óbito", "Data do obito", "Local do óbito", "Local do obito"])
    if _raw_hora:
        hora = _normalize_hora(_raw_hora)
        if hora:
            structured["HORA_OBITO"] = hora

    structured["CIDADE_OBITO"] = _normalize_cidade(_find_block_value(text, [
        "Município de ocorrência", "Municipio de ocorrencia", "Município de Ocorrência",
    ]))
    structured["UF_OBITO"] = _normalize_uf(_extract_uf_ocorrencia(text))
    if not structured["UF_OBITO"]:
        structured["UF_OBITO"] = _normalize_uf(_find_block_value(text, ["UF"]))

    structured["TIPO_OBITO"] = _detect_obito_type(text)
    if structured["TIPO_OBITO"] not in ("Fetal", "Não Fetal", ""):
        structured["TIPO_OBITO"] = ""

    causas = _parse_parte_i(text)
    structured["CAUSA_MORTE"] = causas.get("CAUSA_MORTE", "")
    structured["CAUSA_MORTE_2"] = causas.get("CAUSA_MORTE_2", "")
    structured["CAUSA_MORTE_3"] = causas.get("CAUSA_MORTE_3", "")
    structured["CAUSA_MORTE_4"] = causas.get("CAUSA_MORTE_4", "")
    structured["CAUSA_BASICA"] = causas.get("CAUSA_BASICA", "")
# --- Ponto 1b: deduplicar causas iguais em campos consecutivos ---
_causas_seq = []
for _k in ["CAUSA_MORTE", "CAUSA_MORTE_2", "CAUSA_MORTE_3", "CAUSA_MORTE_4"]:
    _v = _clean_causa(structured.get(_k, ""))
    _v = re.sub(r'^[a-z]\s+(?=[A-ZÀ-Ú])', '', _v, flags=re.IGNORECASE).strip()
    if _v:
        _causas_seq.append(_v)
_causas_dedup = []
for _c in _causas_seq:
    if not _causas_dedup or _c.lower() != _causas_dedup[-1].lower():
        _causas_dedup.append(_c)
for _idx, _k in enumerate(["CAUSA_MORTE", "CAUSA_MORTE_2", "CAUSA_MORTE_3", "CAUSA_MORTE_4"]):
    structured[_k] = _causas_dedup[_idx] if _idx < len(_causas_dedup) else ""
structured["CAUSA_BASICA"] = _clean_causa(structured.get("CAUSA_BASICA", ""))

    structured["MEDICO_ATESTANTE"] = _clean_field(_find_block_value(text, [
        "Médico", "Medico", "Nome do Médico", "Nome do medico",
    ], stop_labels=["CRM"]))
    structured["CRM_MEDICO"] = _clean_field(_find_block_value(text, ["CRM"]))

    do_match = re.search(r'Declaração\s+de\s+Óbito\s+(\d+(?:-\d+)?)', text, re.IGNORECASE)
    if do_match:
        structured["DO_NUMERO"] = do_match.group(1)

    parte_ii_match = re.search(
        r'PARTE\s+II[:\s]*\n?(.*?)(?:Outros episódios|Nome do médico|CRM|$)',
        text, re.DOTALL | re.IGNORECASE
    )
    if parte_ii_match:
        p2 = re.sub(r'^que contribuíram para a morte[^:]*[: ]*', '',
                    parte_ii_match.group(1).strip(), flags=re.IGNORECASE)
        p2 = re.sub(r'^outras condições significativas[^:]*[: ]*', '', p2, flags=re.IGNORECASE)
        p2 = _clean_field(p2[:200])
        if p2 and not any(j in p2.lower() for j in FORM_JUNK):
            structured["PARTE_II"] = p2
            structured["INTERVALO_DOENCA_MORTE"] = p2

    idade_raw = _find_block_value(text, ["Idade", "IDADE"])
    if idade_raw:
        nums = re.findall(r'\d+', idade_raw)
        if nums:
            structured["IDADE_ANOS"] = nums[0]

    return structured

# ── Validacao ────────────────────────────────────────────────────

def validate_obito(structured: dict) -> None:
    missing = [f for f in CRITICAL_FIELDS if not structured.get(f)]
    score = round((len(CRITICAL_FIELDS) - len(missing)) / len(CRITICAL_FIELDS) * 100, 1)
    structured["QUALIDADE_SCORE"] = str(score)
# --- Ponto 2: regras de consistencia cruzada ---
_erros_extra = []
_nome = structured.get("NOME", "")
if _nome and any(j in _nome.lower() for j in FORM_JUNK):
    _erros_extra.append("Nome contem texto do formulario")
_nasc = structured.get("NASCIMENTO", "")
_dob = structured.get("DATA_OBITO", "")
_idade = structured.get("IDADE_ANOS", "")
if _nasc and _dob and _idade and _idade.isdigit():
    try:
        from datetime import datetime as _dt
        _calc = (_dt.strptime(_dob, "%d/%m/%Y") - _dt.strptime(_nasc, "%d/%m/%Y")).days // 365
        if abs(_calc - int(_idade)) > 5:
            _erros_extra.append(f"Idade inconsistente com datas (calculada ~{_calc})")
    except Exception:
        pass
if structured.get("TIPO_OBITO") == "Fetal" and not structured.get("CAUSA_MORTE"):
    _erros_extra.append("Obito fetal sem causa de morte")
if _erros_extra:
    structured["STATUS"] = "REVISAR"
    structured["ERROS"] = ((structured.get("ERROS", "") + " | ") if structured.get("ERROS") else "") + " | ".join(_erros_extra)
    if missing:
        structured["STATUS"] = "REVISAR"
        structured["ERROS"] = " | ".join(f"Campo critico ausente: {f}" for f in missing)
    else:
        structured["STATUS"] = "OK"
        structured["ERROS"] = ""

# ── Processamento individual (com dedup) ─────────────────────────

def _process_single_image(file_id, file_name, existing):
    logger.info(f"Processando: {file_name} ({file_id})")
    try:
        image_bytes, mime_type = _download_image_bytes(file_id)
    except Exception as e:
        return {"NOME_ARQUIVO": file_name, "STATUS": "ERRO_DRIVE", "ERROS": str(e)}
    h = _sha256_bytes(image_bytes)
    if h in existing["hashes"]:
        logger.info(f"{file_name}: hash ja existente, pulando")
        return {"NOME_ARQUIVO": file_name, "STATUS": "DUPLICADO", "ERROS": ""}
    try:
        raw_text, confidence = _ocr_image_from_bytes(image_bytes, mime_type)
        if raw_text:
            logger.info(f"[OCR RESPONSE] {file_name}: {raw_text[:300]}")
    except Exception as e:
        return {"NOME_ARQUIVO": file_name, "STATUS": "ERRO_OCR", "ERROS": str(e)}
    if not _is_valid_obito(raw_text):
        logger.warning(f"{file_name}: nao reconhecido como DO, pulando")
        return {"NOME_ARQUIVO": file_name, "STATUS": "REJEITADO",
                "ERROS": "Imagem nao contem uma Declaracao de Obito valida"}
    try:
        structured = parse_obito(raw_text)
    except Exception as e:
        structured = {k: "" for k in HEADER}
        structured["ERROS"] = f"Erro no parser: {e}"
    structured["HASH_ARQUIVO"] = h
    structured["HASH_CONTEUDO"] = _sha256_text(raw_text)
    validate_obito(structured)
    structured["DATA_PROCESSAMENTO"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    structured["NOME_ARQUIVO"] = file_name
    return structured

# ── Batch ────────────────────────────────────────────────────────

def _run_batch(limit: int, reprocess: bool = False, min_score: float = None, files: str = None) -> dict:
    logger.info(f"Iniciando {'reprocessamento' if reprocess else 'batch'} com limit={limit}")
    try:
        drive = _get_drive_service()
        logger.info(f"Listando arquivos recursivamente de {DRIVE_FOLDER_ID}...")
        all_files = _list_all_files_recursive(DRIVE_FOLDER_ID, drive)
        total = len(all_files)
        logger.info(f"Total de arquivos no Drive: {total}")
        # Filtro por min_score (reprocessar so as de score baixo) ou files (nomes especificos)
        if reprocess and (min_score is not None or files):
            low_names = set()
            if min_score is not None:
                try:
                    sheets = _get_sheets_service()
                    res = sheets.spreadsheets().values().get(
                        spreadsheetId=SHEET_ID, range="Auditoria!B:D"
                    ).execute()
                    for r in res.get("values", []):
                        if len(r) < 3:
                            continue
                        fname = str(r[0]).strip()
                        try:
                            sc = float(r[2])
                        except Exception:
                            continue
                        if fname and fname != "NOME_ARQUIVO" and fname != "STATUS" and sc < min_score:
                            low_names.add(fname)
                except Exception as e:
                    logger.warning(f"Falha ao ler scores da planilha: {e}")
            if files:
                for f in files.split(","):
                    f = f.strip()
                    if f:
                        low_names.add(f)
            if low_names:
                to_process = [img for img in all_files if img.get("name", "") in low_names]
                logger.info(f"Filtro aplicado: {len(to_process)} arquivos (min_score={min_score}, files={files})")
            else:
                to_process = all_files[:limit]
        else:
            to_process = all_files[:limit]
        existing = {"hashes": {}, "names": set()} if reprocess else _get_existing_data()
        rows_to_insert = []
        processed, duplicates, rejected, failed = 0, 0, 0, 0
        _ensure_sheet_header()
        for img in to_process:
            time.sleep(1)
            row = _process_single_image(img["id"], img.get("name", "unknown"), existing)
            status = row.get("STATUS", "")
            if status == "DUPLICADO":
                duplicates += 1
                continue
            if status == "REJEITADO":
                rejected += 1
                continue
            if status in ("ERRO_DRIVE", "ERRO_OCR"):
                failed += 1
                continue
            processed += 1
            if row.get("HASH_ARQUIVO"):
                existing["hashes"][row["HASH_ARQUIVO"]] = True
            if row.get("NOME_ARQUIVO"):
                existing["names"].add(row["NOME_ARQUIVO"])
            rows_to_insert.append([row.get(h, "") for h in HEADER])
        if rows_to_insert:
            result = _append_rows_to_sheet(rows_to_insert)
            if result:
                logger.info(f"Inseridas {len(rows_to_insert)} linhas na planilha.")
            else:
                logger.error("Falha ao inserir linhas na planilha.")
        msg = (f"{processed} processadas, {duplicates} duplicadas puladas, "
               f"{rejected} rejeitadas, {failed} falhas (total no Drive: {total})")
        return {"success": True, "total": total, "new": len(to_process),
                "processed": processed, "duplicates": duplicates,
                "rejected": rejected, "failed": failed,
                "sheet_id": SHEET_ID, "message": msg, "requestId": str(uuid.uuid4())}
    except Exception as e:
        logger.error(f"Erro no batch: {e}", exc_info=True)
        return {"success": False, "error": str(e), "message": "Erro interno"}

# ── Dedupe da aba Auditoria ──────────────────────────────────────

def _dedupe_auditoria(sheet_id: str = SHEET_ID) -> dict:
    """Remove duplicatas mantendo o melhor registro por HASH_ARQUIVO.
    Grava em aba nova 'Auditoria_LIMPA' (nao altera a original)."""
    sheets = _get_sheets_service()
    result = sheets.spreadsheets().values().get(
        spreadsheetId=sheet_id, range="Auditoria!A1:W1579"
    ).execute()
    rows = result.get("values", [])
    if not rows:
        return {"success": False, "message": "Aba Auditoria vazia"}
    header = rows[0]
    data = rows[1:]

    def _num(v):
        try:
            return float(v)
        except Exception:
            return -1.0

    def _key(r):
        return (_num(r[3]), 1 if (r[2] or "").strip() == "OK" else 0, r[0] or "")

    best = {}
    rejected = {}
    for r in data:
        r = r + [""] * (len(header) - len(r))
        h = (r[22] or "").strip()
        fname = (r[1] or "").strip()
        if not h:
            if fname:
                rejected[fname] = r
            continue
        if h not in best or _key(r) > _key(best[h]):
            best[h] = r

    final = list(best.values()) + list(rejected.values())

    info = sheets.spreadsheets().get(spreadsheetId=sheet_id, fields="sheets.properties.title").execute()
    titles = [s["properties"]["title"] for s in info.get("sheets", [])]
    if "Auditoria_LIMPA" not in titles:
        sheets.spreadsheets().batchUpdate(
            spreadsheetId=sheet_id,
            body={"requests": [{"addSheet": {"properties": {"title": "Auditoria_LIMPA"}}}]}
        ).execute()

    sheets.spreadsheets().values().update(
        spreadsheetId=sheet_id, range="Auditoria_LIMPA!A1",
        valueInputOption="USER_ENTERED",
        body={"values": [header] + final}
    ).execute()

    return {
        "success": True,
        "linhas_originais": len(data),
        "linhas_unicas": len(final),
        "removidas": len(data) - len(final),
        "aba": "Auditoria_LIMPA",
        "observacao": "Revise a aba Auditoria_LIMPA. Se estiver ok, posso substituir a Auditoria pela versao limpa.",
    }

# ── FastAPI App ──────────────────────────────────────────────────

app = FastAPI(title="Obito OCR Service", version="3.1")

class BatchRequest(BaseModel):
    limit: int = 10

@app.get("/")
def root():
    return {"status": "running", "service": "Obito OCR Service", "version": "3.1"}

@app.post("/batch/process")
def batch_process(request: BatchRequest):
    return _run_batch(limit=request.limit, reprocess=False)

@app.post("/batch/reprocess")
def batch_reprocess(limit: int = 10, min_score: float = None, files: str = None):
    return _run_batch(limit=limit, reprocess=True, min_score=min_score, files=files)

@app.post("/admin/dedupe")
def admin_dedupe():
    """Remove duplicatas da aba Auditoria e grava em Auditoria_LIMPA."""
    try:
        return _dedupe_auditoria()
    except Exception as e:
        logger.error(f"Erro no dedupe: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
# deploy touch 28/08/2026 09:59:39
