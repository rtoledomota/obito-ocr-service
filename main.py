# FILE: main.py

"""
Serviço obito-ocr-service v2

API FastAPI para OCR de declarações de óbito brasileiras.
Compatível com o contrato atual do Apps Script.

Execução local:
    uvicorn main:app --host 0.0.0.0 --port $PORT

Variáveis de ambiente:
    ENDPOINT_AUTH_TOKEN      Token Bearer exigido no header Authorization
    OPENAI_API_URL           URL base do provedor OpenAI-compatible (chat/completions)
    OPENAI_API_KEY           Chave de API do provedor
    OPENAI_MODEL_DEFAULT     Modelo padrão de visão
    MAX_FILE_SIZE_MB         Tamanho máximo do arquivo em MB (default 10)
"""

import base64
import binascii
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Configuração via variáveis de ambiente
# ---------------------------------------------------------------------------

ENDPOINT_AUTH_TOKEN: str = os.getenv("ENDPOINT_AUTH_TOKEN", "")
OPENAI_API_URL: str = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT: str = os.getenv("OPENAI_MODEL_DEFAULT", "gpt-4o-mini")
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))

MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_MIME_TYPES: Tuple[str, ...] = (
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",
)

# Ordem exata exigida pelo Apps Script para o objeto structured.
HEADER: List[str] = [
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

# CIDs considerados "garbage" (causas básicas vagas/imprecisas) para declaração de óbito.
GARBAGE_CID_SET = {
    "R99", "R99X", "I99", "I99X", "I46", "I46X", "I469", "I46.9",
    "J96", "J96X", "J969", "J96.9", "R57", "R57X", "R579", "R57.9",
    "P95", "P95X", "I50", "I50X", "I509", "I50.9", "A41", "A41X",
    "A419", "A41.9", "E14", "E14X", "E149", "E14.9",
}

UF_BRASILEIRAS = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA",
    "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN",
    "RS", "RO", "RR", "SC", "SP", "SE", "TO",
}

MESES_PT = {
    1: "janeiro", 2: "fevereiro", 3: "marco", 4: "abril",
    5: "maio", 6: "junho", 7: "julho", 8: "agosto",
    9: "setembro", 10: "outubro", 11: "novembro", 12: "dezembro",
}


# ---------------------------------------------------------------------------
# Aplicação FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="obito-ocr-service", version="2.0.0")


# ---------------------------------------------------------------------------
# Utilitários de erro
# ---------------------------------------------------------------------------

def make_error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str = "",
) -> JSONResponse:
    """Padroniza respostas de erro em JSON com code/message/requestId."""
    return JSONResponse(
        status_code=status_code,
        content={
            "code": code,
            "message": message,
            "requestId": request_id,
        },
    )


async def authenticate(authorization: Optional[str]) -> None:
    """Valida o token Bearer no header Authorization."""
    if not ENDPOINT_AUTH_TOKEN:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail={
            "code": "UNAUTHORIZED",
            "message": "Header Authorization ausente.",
            "requestId": "",
        })
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != ENDPOINT_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail={
            "code": "UNAUTHORIZED",
            "message": "Token Bearer invalido.",
            "requestId": "",
        })


# ---------------------------------------------------------------------------
# Utilitários de base64 / arquivo
# ---------------------------------------------------------------------------

def decode_base64_file(b64: str) -> bytes:
    """Decodifica base64 removendo possíveis prefixos data URL e espaços."""
    cleaned = b64.strip()
    if cleaned.startswith("data:"):
        # remove prefixo data:<mime>;base64,
        cleaned = cleaned.split(",", 1)[-1]
    try:
        return base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Base64 invalido: {exc}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


# ---------------------------------------------------------------------------
# Adaptador de OCR (OpenAI-compatible)
# ---------------------------------------------------------------------------
# Este adaptador é o ÚNICO ponto de ajuste fino de provedor de OCR.
# Para trocar de provedor, basta modificar esta função mantendo a assinatura.
# ---------------------------------------------------------------------------

def ocr_image_with_openai(
    file_bytes: bytes,
    mime_type: str,
    model: str,
    request_id: str,
) -> Tuple[str, float, str]:
    """
    Envia a imagem para um provedor OpenAI-compatible (chat/completions)
    usando data URL e retorna (texto, confianca, provedor).

    Em caso de falha, levanta RuntimeError com mensagem controlada.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY nao configurada.")

    b64 = base64.b64encode(file_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{b64}"

    prompt = (
        "Voce é um especialista em extrair texto de declaracoes de obito brasileiras. "
        "Extraia TODO o texto visivel no documento, preservando a ordem e as quebras de linha. "
        "Nao invente informacoes. Se um campo nao estiver visivel, deixe em branco. "
        "Retorne apenas o texto extraido, sem comentarios."
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
        "max_tokens": 2000,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            OPENAI_API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Falha de comunicacao com provedor OCR: {exc}")

    if response.status_code != 200:
        raise RuntimeError(
            f"Provedor OCR retornou HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f"Resposta do provedor OCR malformada: {exc}")

    # Confiança heurística: provedores OpenAI não retornam score; usamos 0.90 como padrão.
    confidence = 0.90
    provider = "openai-compatible"
    return text, confidence, provider


# ---------------------------------------------------------------------------
# Parser heurístico de declaração de óbito
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normaliza texto para facilitar regex: troca tabulações por espaços."""
    return text.replace("\t", " ").replace("\r", "\n")


def _find_label_value(text: str, labels: List[str], max_chars: int = 120) -> str:
    """
    Procura por um rótulo (ex.: 'Nome:') e retorna o valor seguinte até nova linha
    ou limite de caracteres. Case-insensitive e tolerante a espaços.
    """
    norm = _normalize_text(text)
    for label in labels:
        # rótulo seguido de : ou não, capturando valor até quebra de linha
        pattern = re.compile(
            rf"{re.escape(label)}\s*[:\-]?\s*(.{{1,{max_chars}}}?)(?:\n|$)",
            re.IGNORECASE,
        )
        match = pattern.search(norm)
        if match:
            value = match.group(1).strip()
            # remove rótulos colados no valor
            value = re.sub(r"\s+", " ", value).strip(" :;-")
            if value:
                return value
    return ""


def _extract_date(text: str, labels: List[str]) -> str:
    """Extrai data no formato DD/MM/AAAA próxima de um rótulo."""
    norm = _normalize_text(text)
    for label in labels:
        pattern = re.compile(
            rf"{re.escape(label)}\s*[:\-]?\s*(\d{{2}}/\d{{2}}/\d{{4}})",
            re.IGNORECASE,
        )
        match = pattern.search(norm)
        if match:
            return match.group(1)
    # fallback: primeira data DD/MM/AAAA do texto
    fallback = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", norm)
    return fallback.group(1) if fallback else ""


def _extract_time(text: str, labels: List[str]) -> str:
    """Extrai hora no formato HH:MM próxima de um rótulo."""
    norm = _normalize_text(text)
    for label in labels:
        pattern = re.compile(
            rf"{re.escape(label)}\s*[:\-]?\s*(\d{{2}}:\d{{2}})",
            re.IGNORECASE,
        )
        match = pattern.search(norm)
        if match:
            return match.group(1)
    fallback = re.search(r"\b(\d{2}:\d{2})\b", norm)
    return fallback.group(1) if fallback else ""


def _extract_cep(text: str) -> str:
    match = re.search(r"\b(\d{5}-?\d{3})\b", text)
    if not match:
        return ""
    return match.group(1).replace("-", "")


def _extract_uf(text: str, labels: List[str]) -> str:
    norm = _normalize_text(text)
    for label in labels:
        pattern = re.compile(
            rf"{re.escape(label)}\s*[:\-]?\s*([A-Z]{{2}})\b",
            re.IGNORECASE,
        )
        match = pattern.search(norm)
        if match:
            uf = match.group(1).upper()
            if uf in UF_BRASILEIRAS:
                return uf
    return ""


def _extract_cids(text: str) -> List[str]:
    """Captura códigos CID-10 (ex.: I21.9, R99, J96.9) no texto."""
    matches = re.findall(r"\b([A-TV-Z]\d{2}(?:\.\d{1,2})?)\b", text)
    seen: List[str] = []
    for m in matches:
        if m not in seen:
            seen.append(m)
    return seen


def _extract_causes(text: str) -> List[str]:
    """
    Captura causas de morte a partir de rótulos comuns em declarações de óbito.
    Retorna até 5 causas na ordem em que aparecem.
    """
    causes: List[str] = []
    norm = _normalize_text(text)
    # rótulos típicos: "Causa", "Causa da morte", "Causas", "Causa basica", etc.
    patterns = [
        r"causa(?:s)?\s+(?:da\s+)?morte\s*[:\-]?\s*(.{5,200}?)(?:\n|$)",
        r"causa\s+basica\s*[:\-]?\s*(.{5,200}?)(?:\n|$)",
        r"causa\s*[:\-]?\s*(.{5,200}?)(?:\n|$)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, norm, re.IGNORECASE):
            value = m.group(1).strip(" :;-")
            value = re.sub(r"\s+", " ", value)
            if value and value not in causes:
                causes.append(value)
            if len(causes) >= 5:
                break
        if len(causes) >= 5:
            break
    return causes[:5]


def _detect_garbage(cids: List[str]) -> Tuple[List[str], int]:
    """Retorna lista de CIDs suspeitos e a quantidade."""
    garbage: List[str] = []
    for cid in cids:
        normalized = cid.upper().replace(".", "")
        # testa versão com e sem ponto
        variants = {cid.upper(), normalized, normalized[:3]}
        if variants & GARBAGE_CID_SET:
            garbage.append(cid.upper())
    # remove duplicados mantendo ordem
    seen: List[str] = []
    for g in garbage:
        if g not in seen:
            seen.append(g)
    return seen, len(seen)


def _nome_ok(nome: str) -> str:
    """Heurística simples: nome válido se tem ao menos 2 palavras e >= 5 caracteres."""
    if not nome:
        return "NAO"
    palavras = nome.strip().split()
    if len(palavras) >= 2 and len(nome.replace(" ", "")) >= 5:
        return "SIM"
    return "NAO"


def _nomes_ok(nome: str, mae: str, pai: str) -> str:
    """Heurística: SIM apenas se nome e ao menos um dos pais estiverem ok."""
    if _nome_ok(nome) == "SIM" and (_nome_ok(mae) == "SIM" or _nome_ok(pai) == "SIM"):
        return "SIM"
    return "NAO"


def _mes_nome(data_obito: str) -> str:
    """Deriva NOME_MES em português a partir de DATA_OBITO (DD/MM/AAAA)."""
    try:
        dia, mes, ano = data_obito.split("/")
        mes_int = int(mes)
        if 1 <= mes_int <= 12:
            return MESES_PT[mes_int]
    except (ValueError, KeyError):
        pass
    return ""


def _parse_date(data: str) -> Optional[datetime]:
    try:
        return datetime.strptime(data, "%d/%m/%Y")
    except ValueError:
        return None


def _idade_from_text(text: str) -> Optional[int]:
    """Tenta capturar idade explícita (ex.: 'Idade: 68 anos')."""
    m = re.search(r"idade\s*[:\-]?\s*(\d{1,3})\s*anos?", text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def parse_obito(raw_text: str, file_bytes: bytes) -> Dict[str, Any]:
    """
    Parser heurístico de declaração de óbito brasileira.
    Retorna o objeto structured com as chaves exatas de HEADER.
    """
    text = _normalize_text(raw_text)

    # --- Campos básicos ---
    nome = _find_label_value(text, ["Nome", "Nome do falecido", "Nome do(a) falecido(a)"])
    nome_social = _find_label_value(text, ["Nome social", "Nome Social"])
    nome_mae = _find_label_value(text, ["Nome da mae", "Nome da mãe", "Mae", "Mãe", "Filiação materna"])
    nome_pai = _find_label_value(text, ["Nome do pai", "Pai", "Filiação paterna"])
    profissao = _find_label_value(text, ["Profissao", "Profissão", "Ocupacao", "Ocupação"])

    nascimento = _extract_date(text, ["Nascimento", "Data de nascimento", "Nascido em", "Nasceu em"])
    data_obito = _extract_date(text, ["Data do obito", "Data do óbito", "Data de obito", "Data de óbito", "Faleceu em"])
    hora_obito = _extract_time(text, ["Hora do obito", "Hora do óbito", "Hora de obito", "Hora de óbito", "Hora"])
    data_atestado = _extract_date(text, ["Data do atestado", "Data de emissao", "Data de emissão", "Emitido em"])

    sexo = _find_label_value(text, ["Sexo", "Genero", "Gênero"])
    if sexo:
        sexo_low = sexo.lower()
        if sexo_low.startswith("m"):
            sexo = "M"
        elif sexo_low.startswith("f"):
            sexo = "F"

    raca_cor = _find_label_value(text, ["Raca/Cor", "Raça/Cor", "Raca", "Raça", "Cor"])
    estado_civil = _find_label_value(text, ["Estado civil", "Estado Civil"])
    nacionalidade = _find_label_value(text, ["Nacionalidade"])

    tipo_obito_raw = _find_label_value(text, ["Tipo de obito", "Tipo de óbito", "Tipo obito"])
    tipo_obito = tipo_obito_raw.upper() if tipo_obito_raw else ""
    if "FETA" in tipo_obito or "FETAL" in tipo_obito:
        tipo_obito = "FETAL"
    elif "MATER" in tipo_obito or "MATERN" in tipo_obito:
        tipo_obito = "MATERNO"
    elif tipo_obito and tipo_obito not in {"FETAL", "MATERNO"}:
        tipo_obito = "NAO_FETAL"

    assistido_raw = _find_label_value(text, ["Assistido", "Obito assistido", "Óbito assistido"])
    assistido = ""
    if assistido_raw:
        al = assistido_raw.lower()
        if al.startswith("s"):
            assistido = "SIM"
        elif al.startswith("n"):
            assistido = "NAO"

    # --- Endereço ---
    logradouro = _find_label_value(text, ["Logradouro", "Endereco", "Endereço", "Rua"])
    numero = _find_label_value(text, ["Numero", "Número"])
    complemento = _find_label_value(text, ["Complemento"])
    bairro = _find_label_value(text, ["Bairro"])
    cidade = _find_label_value(text, ["Cidade", "Municipio", "Município"])
    uf = _extract_uf(text, ["UF", "Estado"])
    cep = _extract_cep(text)

    cidade_nascimento = _find_label_value(text, ["Cidade de nascimento", "Naturalidade", "Municipio de nascimento"])
    uf_nascimento = _extract_uf(text, ["UF de nascimento", "UF nascimento", "Estado de nascimento"])

    local_obito = _find_label_value(text, ["Local do obito", "Local do óbito", "Local de obito", "Local de óbito", "Local"])
    cidade_obito = _find_label_value(text, ["Cidade do obito", "Cidade do óbito", "Cidade de obito"])
    uf_obito = _extract_uf(text, ["UF do obito", "UF do óbito", "UF obito"])

    cpf = _find_label_value(text, ["CPF"])
    cpf = re.sub(r"\D", "", cpf) if cpf else ""
    rg = _find_label_value(text, ["RG", "Identidade", "Registro geral"])
    orgao_emissor_rg = _find_label_value(text, ["Orgao emissor", "Órgão emissor", "OE", "Emissor"])

    # --- Causas e CIDs ---
    causes = _extract_causes(text)
    cids = _extract_cids(text)

    causa_basica = causes[-1] if causes else ""
    cid_basica = cids[-1] if cids else ""

    garbage_codes, qtd_garbage = _detect_garbage(cids)

    # --- Hashes e datas ---
    hash_arquivo = sha256_bytes(file_bytes)
    hash_conteudo = sha256_text(raw_text)
    data_processamento = datetime.now(timezone.utc).isoformat()
    nome_mes = _mes_nome(data_obito)

    # --- Heurísticas de nomes ---
    nome_ok = _nome_ok(nome)
    nomes_ok = _nomes_ok(nome, nome_mae, nome_pai)

    # --- Validação e erros ---
    validation = validate_obito(
        nascimento=nascimento,
        data_obito=data_obito,
        hora_obito=hora_obito,
        uf=uf,
        uf_obito=uf_obito,
        uf_nascimento=uf_nascimento,
        cep=cep,
        text=text,
        nome=nome,
        nome_mae=nome_mae,
        causa_basica=causa_basica,
    )

    erros = " | ".join(validation["errors"])
    status = "REVISAR" if validation["errors"] else "OK"

    # --- Qualidade score ---
    qualidade_score = compute_quality_score(
        nome=nome,
        nome_mae=nome_mae,
        nascimento=nascimento,
        data_obito=data_obito,
        hora_obito=hora_obito,
        causa_basica=causa_basica,
        cidade_obito=cidade_obito,
        uf_obito=uf_obito,
        erros=validation["errors"],
    )

    # --- Monta structured na ordem exata ---
    structured: Dict[str, Any] = {key: "" for key in HEADER}

    structured["NOME"] = nome
    structured["NOME_SOCIAL"] = nome_social
    structured["NASCIMENTO"] = nascimento
    structured["SEXO"] = sexo
    structured["RACA_COR"] = raca_cor
    structured["ESTADO_CIVIL"] = estado_civil
    structured["NACIONALIDADE"] = nacionalidade
    structured["NOME_MAE"] = nome_mae
    structured["NOME_PAI"] = nome_pai
    structured["PROFISSAO"] = profissao
    structured["LOGRADOURO"] = logradouro
    structured["NUMERO"] = numero
    structured["COMPLEMENTO"] = complemento
    structured["BAIRRO"] = bairro
    structured["CIDADE"] = cidade
    structured["UF"] = uf
    structured["CEP"] = cep
    structured["CIDADE_NASCIMENTO"] = cidade_nascimento
    structured["UF_NASCIMENTO"] = uf_nascimento
    structured["CPF"] = cpf
    structured["RG"] = rg
    structured["ORGAO_EMISSOR_RG"] = orgao_emissor_rg
    structured["DATA_OBITO"] = data_obito
    structured["HORA_OBITO"] = hora_obito
    structured["LOCAL_OBITO"] = local_obito
    structured["CIDADE_OBITO"] = cidade_obito
    structured["UF_OBITO"] = uf_obito

    # causas (até 5)
    for i, causa in enumerate(causes[:5]):
        if i == 0:
            structured["CAUSA_MORTE"] = causa
        else:
            structured[f"CAUSA_MORTE_{i + 1}"] = causa

    structured["CAUSA_BASICA"] = causa_basica

    # códigos de causa (espelham CIDs quando disponíveis)
    for i, cid in enumerate(cids[:5]):
        if i == 0:
            structured["CODIGO_CAUSA_MORTE"] = cid
            structured["CID_MORTE"] = cid
        else:
            structured[f"CODIGO_CAUSA_MORTE_{i + 1}"] = cid
            structured[f"CID_MORTE_{i + 1}"] = cid

    structured["CODIGO_CAUSA_BASICA"] = cid_basica
    structured["CID_BASICA"] = cid_basica

    structured["TIPO_OBITO"] = tipo_obito
    structured["ASSISTIDO"] = assistido
    structured["DATA_ATESTADO"] = data_atestado
    structured["NOMES_OK"] = nomes_ok
    structured["NOME_OK"] = nome_ok
    structured["GARBAGE_CODES"] = ",".join(garbage_codes)
    structured["QTD_GARBAGE"] = qtd_garbage
    structured["PROTOCOLO_TEV"] = ""
    structured["ERROS"] = erros
    structured["QUALIDADE_SCORE"] = qualidade_score
    structured["HASH_ARQUIVO"] = hash_arquivo
    structured["HASH_CONTEUDO"] = hash_conteudo
    structured["STATUS"] = status
    structured["NOME_MES"] = nome_mes
    structured["DATA_PROCESSAMENTO"] = data_processamento

    return structured


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

def validate_obito(
    nascimento: str,
    data_obito: str,
    hora_obito: str,
    uf: str,
    uf_obito: str,
    uf_nascimento: str,
    cep: str,
    text: str,
    nome: str,
    nome_mae: str,
    causa_basica: str,
) -> Dict[str, Any]:
    """Valida coerência de campos e retorna objeto validation."""
    errors: List[str] = []
    warnings: List[str] = []
    computed: Dict[str, Any] = {}

    # Validação de formato de datas
    if nascimento and not _parse_date(nascimento):
        errors.append("NASCIMENTO com formato invalido (esperado DD/MM/AAAA)")
    if data_obito and not _parse_date(data_obito):
        errors.append("DATA_OBITO com formato invalido (esperado DD/MM/AAAA)")

    # Validação de hora
    if hora_obito and not re.match(r"^\d{2}:\d{2}$", hora_obito):
        errors.append("HORA_OBITO com formato invalido (esperado HH:MM)")

    # Validação de UF
    if uf and uf not in UF_BRASILEIRAS:
        errors.append(f"UF invalida: {uf}")
    if uf_obito and uf_obito not in UF_BRASILEIRAS:
        errors.append(f"UF_OBITO invalida: {uf_obito}")
    if uf_nascimento and uf_nascimento not in UF_BRASILEIRAS:
        errors.append(f"UF_NASCIMENTO invalida: {uf_nascimento}")

    # Validação de CEP
    if cep and not re.match(r"^\d{8}$", cep):
        errors.append("CEP deve conter 8 digitos")

    # Coerência entre nascimento e óbito
    dt_nasc = _parse_date(nascimento) if nascimento else None
    dt_obito = _parse_date(data_obito) if data_obito else None
    if dt_nasc and dt_obito:
        if dt_obito < dt_nasc:
            errors.append("DATA_OBITO anterior a NASCIMENTO")
        else:
            idade_calculada = (dt_obito - dt_nasc).days // 365
            computed["idade_calculada"] = idade_calculada
            idade_texto = _idade_from_text(text)
            if idade_texto is not None:
                computed["idade_texto"] = idade_texto
                if abs(idade_calculada - idade_texto) > 1:
                    errors.append(
                        f"Idade incoerente: calculada {idade_calculada} x texto {idade_texto}"
                    )

    # Campos críticos
    if not nome:
        errors.append("NOME ausente")
    if not nome_mae:
        warnings.append("NOME_MAE ausente")
    if not data_obito:
        errors.append("DATA_OBITO ausente")
    if not causa_basica:
        errors.append("CAUSA_BASICA ausente")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "computed": computed,
    }


def compute_quality_score(
    nome: str,
    nome_mae: str,
    nascimento: str,
    data_obito: str,
    hora_obito: str,
    causa_basica: str,
    cidade_obito: str,
    uf_obito: str,
    erros: List[str],
) -> int:
    """Heurística de qualidade de 0 a 100 baseada em campos críticos e erros."""
    score = 0
    pesos = {
        "nome": 20,
        "nome_mae": 10,
        "nascimento": 10,
        "data_obito": 20,
        "hora_obito": 5,
        "causa_basica": 20,
        "cidade_obito": 5,
        "uf_obito": 5,
        "sem_erros": 5,
    }
    if nome:
        score += pesos["nome"]
    if nome_mae:
        score += pesos["nome_mae"]
    if nascimento:
        score += pesos["nascimento"]
    if data_obito:
        score += pesos["data_obito"]
    if hora_obito:
        score += pesos["hora_obito"]
    if causa_basica:
        score += pesos["causa_basica"]
    if cidade_obito:
        score += pesos["cidade_obito"]
    if uf_obito:
        score += pesos["uf_obito"]
    if not erros:
        score += pesos["sem_erros"]
    return min(score, 100)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    """Healthcheck simples para orquestradores (Render, etc.)."""
    return {
        "status": "ok",
        "service": "obito-ocr-service",
        "version": "2.0.0",
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/ocr")
async def ocr(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    """
    Endpoint de OCR compatível com o Apps Script.

    Body JSON:
        file (base64)        - obrigatório
        mimeType             - obrigatório
        model                - opcional
        fileName             - opcional
        fileId               - opcional
        requestId            - opcional

    Resposta JSON:
        text, confidence, provider, requestId, warnings,
        rawText, structured, validation, headerOrder
    """
    await authenticate(authorization)

    # --- Parse do body ---
    try:
        body = await request.json()
    except Exception:
        return make_error_response(400, "INVALID_JSON", "Body deve ser JSON valido.")

    if not isinstance(body, dict):
        return make_error_response(400, "INVALID_JSON", "Body deve ser um objeto JSON.")

    request_id: str = str(body.get("requestId") or "")
    file_b64 = body.get("file")
    mime_type = body.get("mimeType")
    model = body.get("model") or OPENAI_MODEL_DEFAULT

    if not file_b64 or not isinstance(file_b64, str):
        return make_error_response(
            400, "MISSING_FILE", "Campo 'file' (base64) é obrigatorio.", request_id
        )

    if not mime_type or not isinstance(mime_type, str):
        return make_error_response(
            400, "MISSING_MIME_TYPE", "Campo 'mimeType' é obrigatorio.", request_id
        )

    if mime_type not in ALLOWED_MIME_TYPES:
        return make_error_response(
            400,
            "UNSUPPORTED_MIME_TYPE",
            f"MIME type nao suportado: {mime_type}. Permitidos: {', '.join(ALLOWED_MIME_TYPES)}.",
            request_id,
        )

    # PDF ainda não suportado nesta versão (mantém contrato anterior).
    if mime_type == "application/pdf":
        return make_error_response(
            422,
            "PDF_NOT_SUPPORTED_IN_V1",
            "PDF ainda nao suportado nesta versao. Envie imagem (JPEG/PNG/WEBP).",
            request_id,
        )

    # --- Decodifica base64 ---
    try:
        file_bytes = decode_base64_file(file_b64)
    except ValueError as exc:
        return make_error_response(400, "INVALID_BASE64", str(exc), request_id)

    # --- Valida tamanho ---
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return make_error_response(
            413,
            "FILE_TOO_LARGE",
            f"Arquivo excede o tamanho maximo de {MAX_FILE_SIZE_MB} MB.",
            request_id,
        )

    # --- OCR via provedor ---
    warnings: List[str] = []
    try:
        raw_text, confidence, provider = ocr_image_with_openai(
            file_bytes=file_bytes,
            mime_type=mime_type,
            model=model,
            request_id=request_id,
        )
    except RuntimeError as exc:
        return make_error_response(502, "OCR_PROVIDER_ERROR", str(exc), request_id)

    if not raw_text:
        warnings.append("OCR retornou texto vazio.")

    # --- Parser estruturado ---
    try:
        structured = parse_obito(raw_text, file_bytes)
    except Exception as exc:  # pragma: no cover - defensivo
        warnings.append(f"Falha no parser estruturado: {exc}")
        structured = {key: "" for key in HEADER}

    # --- Validação (já calculada dentro do parser, mas exposta separadamente) ---
    validation = validate_obito(
        nascimento=structured.get("NASCIMENTO", ""),
        data_obito=structured.get("DATA_OBITO", ""),
        hora_obito=structured.get("HORA_OBITO", ""),
        uf=structured.get("UF", ""),
        uf_obito=structured.get("UF_OBITO", ""),
        uf_nascimento=structured.get("UF_NASCIMENTO", ""),
        cep=structured.get("CEP", ""),
        text=raw_text,
        nome=structured.get("NOME", ""),
        nome_mae=structured.get("NOME_MAE", ""),
        causa_basica=structured.get("CAUSA_BASICA", ""),
    )

    # --- Resposta final ---
    response_body = {
        "text": raw_text,
        "confidence": confidence,
        "provider": provider,
        "requestId": request_id,
        "warnings": warnings,
        "rawText": raw_text,
        "structured": structured,
        "validation": validation,
        "headerOrder": HEADER,
    }

    return JSONResponse(status_code=200, content=response_body)


# ---------------------------------------------------------------------------
# Execução local
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
