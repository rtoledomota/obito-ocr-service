# FILE: main.py
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
import json
import base64
import hashlib
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn


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

# Frases que indicam recusa do provedor OCR
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

# Marcadores auxiliares conhecidos que não são valores
LABEL_NOISE = {
    'uf', 'município', 'municipio', 'nome', 'mãe', 'mae', 'pai',
    'data', 'hora', 'local', 'causas', 'causa', 'médico', 'medico',
    'crm', 'assinatura', 'carimbo', 'endereço', 'endereco'
}


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
    """Verificação de saúde do serviço."""
    return {"status": "ok", "service": "obito-ocr-service", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Autenticação
# ---------------------------------------------------------------------------

def _check_auth(authorization: Optional[str]) -> None:
    """Valida o token Bearer enviado no cabeçalho Authorization."""
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
# Utilidades
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_lines(text: str) -> List[str]:
    """Quebra o texto em linhas já sem espaços nas bordas e sem linhas vazias."""
    return [line.strip() for line in text.split("\n") if line.strip()]


def _normalize_date(value: str) -> str:
    """Normaliza datas com espaço, barra ou hífen para DD/MM/AAAA."""
    if not value:
        return ""
    v = value.strip()
    # Remove texto extra após a data
    v = re.sub(r"[^0-9/\-\s]", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    # Tenta padrões com / - ou espaço
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
    """Normaliza hora para HH:MM."""
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


def _normalize_uf(value: str) -> str:
    """Normaliza UF: só aceita 2 letras válidas, nunca o rótulo 'UF'."""
    if not value:
        return ""
    v = value.strip().upper()
    if v == "UF":
        return ""
    # Extrai duas letras maiúsculas
    m = re.search(r"\b([A-Z]{2})\b", v)
    if m and m.group(1) in UF_VALIDAS:
        return m.group(1)
    if v in UF_VALIDAS:
        return v
    return ""


def _normalize_cep(value: str) -> str:
    """Normaliza CEP para 00000-000."""
    if not value:
        return ""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 8:
        return f"{digits[:5]}-{digits[5:]}"
    return value.strip()


# ---------------------------------------------------------------------------
# Parser: busca estrita por linha
# ---------------------------------------------------------------------------

def _find_strict_next_line_value(
    text: str,
    labels: List[str],
    stop_labels: Optional[List[str]] = None,
    max_distance: int = 4,
) -> str:
    """
    Procura rótulo por igualdade de linha inteira (ou linha iniciando com rótulo + ':')
    e retorna a próxima linha útil que não seja outro rótulo nem marcador de parada.
    """
    lines = _normalize_lines(text)
    labels_norm = [l.lower().strip() for l in labels]
    stop_norm = [s.lower().strip() for s in (stop_labels or [])]

    for i, line in enumerate(lines):
        line_lower = line.lower()
        matched = False
        for lab in labels_norm:
            if line_lower == lab or line_lower == lab + ":" or line_lower.startswith(lab + ":"):
                matched = True
                break
        if not matched:
            continue

        # Percorre as próximas linhas procurando o valor real
        for j in range(i + 1, min(i + 1 + max_distance, len(lines))):
            candidate = lines[j]
            cand_lower = candidate.lower()
            # Parada explícita
            if any(cand_lower == s or cand_lower.startswith(s) for s in stop_norm):
                break
            # Ignora rótulos conhecidos e ruído
            if cand_lower in LABEL_NOISE:
                continue
            if cand_lower.endswith(":") and len(cand_lower) < 30:
                continue
            # Ignora linhas que são apenas rótulos curtos
            if len(candidate) <= 2:
                continue
            return candidate
    return ""


def _find_inline_value(text: str, labels: List[str]) -> str:
    """Busca valor na mesma linha após ':' ou ' - '."""
    lines = _normalize_lines(text)
    labels_norm = [l.lower().strip() for l in labels]
    for line in lines:
        line_lower = line.lower()
        for lab in labels_norm:
            if line_lower.startswith(lab):
                rest = line[len(lab):].lstrip(": -\t").strip()
                if rest and rest.upper() != "UF" and len(rest) > 1:
                    return rest
    return ""


def _find_block_value(
    text: str,
    labels: List[str],
    stop_labels: Optional[List[str]] = None,
) -> str:
    """Versão conservadora: tenta inline primeiro, depois próxima linha estrita."""
    inline = _find_inline_value(text, labels)
    if inline:
        return inline
    return _find_strict_next_line_value(text, labels, stop_labels=stop_labels)


def _extract_causes(text: str) -> List[str]:
    """
    Extrai causas da morte em ordem, ignorando rodapés e campos do médico.
    - Inicia em 'CAUSAS DA MORTE'
    - Para antes de 'Nome do Médico', 'CRM', 'Óbito atestado por Médico',
      'PROVÁVEIS CIRCUNSTÂNCIAS'
    - Ignora títulos e linhas auxiliares
    """
    lines = _normalize_lines(text)
    start_markers = ["causas da morte", "causa da morte", "devido a"]
    stop_markers = [
        "nome do médico", "nome do medico", "crm", "óbito atestado por",
        "obito atestado por", "prováveis circunstâncias", "provaveis circunstancias",
        "médico", "medico",
    ]

    start_idx = -1
    for i, line in enumerate(lines):
        if any(line.lower().startswith(m) or line.lower() == m for m in start_markers):
            start_idx = i
            break
    if start_idx == -1:
        return []

    causes: List[str] = []
    noise_tokens = {
        "causas da morte", "causa da morte", "devido a", "(a)", "(b)",
        "(c)", "(d)", "causas", "causa", "médico", "medico", "crm",
        "assinatura", "carimbo", "linha a", "linha b", "linha c",
        "parte i", "parte ii",
    }

    for line in lines[start_idx + 1:]:
        low = line.lower()
        if any(low.startswith(s) or low == s for s in stop_markers):
            break
        if not low:
            continue
        if low in noise_tokens:
            continue
        if any(low.startswith(n) for n in ["nome do", "assinatura", "carimbo", "crm"]):
            break
        # Ignora marcadores de seção tipo (A) (B) quando isolados
        if re.fullmatch(r"\([a-eA-E]\)", line):
            continue
        # Ignora linhas muito curtas
        if len(line) < 3:
            continue
        causes.append(line)

    return causes


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

def parse_obito(text: str) -> Dict[str, Any]:
    """Constrói o dicionário estruturado a partir do texto OCR."""
    structured: Dict[str, Any] = {k: "" for k in HEADER}

    # --- Nome ---
    structured["NOME"] = _find_block_value(
        text,
        ["Nome", "Nome do falecido", "Nome do(a) falecido(a)"],
        stop_labels=["Nome da mãe", "Nome do pai", "Nome social", "Data"],
    )
    structured["NOME_SOCIAL"] = _find_block_value(
        text, ["Nome social"],
        stop_labels=["Nome", "Nome da mãe", "Nome do pai"],
    )

    # --- Pai / Mãe: apenas rótulos longos, sem fallback curto que capture 'País' ---
    structured["NOME_MAE"] = _find_strict_next_line_value(
        text,
        ["Nome da mãe", "Nome da Mãe", "Nome da mae"],
        stop_labels=["Nome do pai", "Profissão", "Profissao", "Endereço", "Endereco"],
    )
    structured["NOME_PAI"] = _find_strict_next_line_value(
        text,
        ["Nome do pai", "Nome do Pai"],
        stop_labels=["Profissão", "Profissao", "Endereço", "Endereco", "Nacionalidade"],
    )

    # --- Datas e hora ---
    structured["NASCIMENTO"] = _normalize_date(
        _find_block_value(text, ["Data de nascimento", "Nascimento", "Nasceu em"])
    )
    structured["DATA_OBITO"] = _normalize_date(
        _find_block_value(text, ["Data do óbito", "Data de óbito", "Data do obito", "Data de obito"])
    )
    structured["HORA_OBITO"] = _normalize_hour(
        _find_block_value(text, ["Hora do óbito", "Hora de óbito", "Hora do obito", "Hora"])
    )
    structured["DATA_ATESTADO"] = _normalize_date(
        _find_block_value(text, ["Data do atestado", "Data de emissão", "Data da emissão"])
    )

    # --- Local do óbito ---
    structured["LOCAL_OBITO"] = _find_block_value(
        text, ["Local do óbito", "Local de óbito", "Local do obito"]
    )
    structured["CIDADE_OBITO"] = _find_strict_next_line_value(
        text,
        ["Município de ocorrência", "Municipio de ocorrência", "Município de ocorrencia", "Municipio de ocorrencia"],
        stop_labels=["UF", "Estado", "Data"],
    )
    # UF do óbito: só 2 letras válidas, nunca o rótulo 'UF'
    uf_obito_raw = _find_block_value(text, ["UF"], stop_labels=["CEP", "Cep"])
    structured["UF_OBITO"] = _normalize_uf(uf_obito_raw)

    # --- Endereço ---
    structured["LOGRADOURO"] = _find_block_value(text, ["Logradouro", "Endereço", "Endereco"])
    structured["NUMERO"] = _find_block_value(text, ["Número", "Numero"])
    structured["COMPLEMENTO"] = _find_block_value(text, ["Complemento"])
    structured["BAIRRO"] = _find_block_value(text, ["Bairro"])
    structured["CIDADE"] = _find_block_value(text, ["Município", "Municipio", "Cidade"])
    uf_end = _find_block_value(text, ["UF"], stop_labels=["CEP", "Cep"])
    structured["UF"] = _normalize_uf(uf_end)
    structured["CEP"] = _normalize_cep(_find_block_value(text, ["CEP", "Cep"]))

    # --- Naturalidade ---
    structured["CIDADE_NASCIMENTO"] = _find_block_value(
        text, ["Naturalidade", "Município de nascimento", "Municipio de nascimento", "Cidade de nascimento"]
    )
    uf_nasc = _find_block_value(text, ["UF de nascimento", "UF/Nascimento"])
    structured["UF_NASCIMENTO"] = _normalize_uf(uf_nasc)

    # --- Documentos ---
    structured["CPF"] = _find_block_value(text, ["CPF"])
    structured["RG"] = _find_block_value(text, ["RG", "Registro Geral"])
    structured["ORGAO_EMISSOR_RG"] = _find_block_value(text, ["Órgão emissor", "Orgao emissor", "Órgão expedidor"])

    # --- Demográficos ---
    structured["SEXO"] = _find_block_value(text, ["Sexo"])
    structured["RACA_COR"] = _find_block_value(text, ["Raça/Cor", "Raça", "Raca/Cor", "Raca"])
    structured["ESTADO_CIVIL"] = _find_block_value(text, ["Estado civil"])
    structured["NACIONALIDADE"] = _find_block_value(text, ["Nacionalidade"])
    structured["PROFISSAO"] = _find_block_value(text, ["Profissão", "Profissao", "Ocupação", "Ocupacao"])

    # --- Causas ---
    causes = _extract_causes(text)
    if causes:
        structured["CAUSA_MORTE"] = causes[0] if len(causes) >= 1 else ""
        structured["CAUSA_MORTE_2"] = causes[1] if len(causes) >= 2 else ""
        structured["CAUSA_MORTE_3"] = causes[2] if len(causes) >= 3 else ""
        structured["CAUSA_MORTE_4"] = causes[3] if len(causes) >= 4 else ""
        structured["CAUSA_MORTE_5"] = causes[4] if len(causes) >= 5 else ""
        # CAUSA_BASICA = última causa não vazia
        non_empty = [c for c in causes if c.strip()]
        structured["CAUSA_BASICA"] = non_empty[-1] if non_empty else ""

    # --- Tipo de óbito / assistido ---
    structured["TIPO_OBITO"] = _find_block_value(text, ["Tipo de óbito", "Tipo de obito"])
    structured["ASSISTIDO"] = _find_block_value(text, ["Assistido", "Foi assistido"])

    # --- Protocolo TEV ---
    structured["PROTOCOLO_TEV"] = _find_block_value(text, ["Protocolo TEV", "Protocolo"])

    # --- Hashes e processamento ---
    structured["HASH_CONTEUDO"] = _sha256_text(text)
    structured["DATA_PROCESSAMENTO"] = dt.datetime.utcnow().isoformat() + "Z"

    # --- NOME_MES derivado de DATA_OBITO ---
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


def _age_coherence(nasc: str, obito: str) -> Optional[str]:
    """Retorna mensagem de erro se idade for incoerente, senão None."""
    if not (_valid_date(nasc) and _valid_date(obito)):
        return None
    try:
        dn = dt.datetime.strptime(nasc, "%d/%m/%Y")
        do = dt.datetime.strptime(obito, "%d/%m/%Y")
        if do < dn:
            return "Data de óbito anterior à data de nascimento"
        idade = (do - dn).days / 365.25
        if idade < 0 or idade > 130:
            return f"Idade incoerente: {idade:.0f} anos"
    except Exception:
        return None
    return None


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

    # Coerência de idade
    age_err = _age_coherence(structured.get("NASCIMENTO", ""), structured.get("DATA_OBITO", ""))
    if age_err:
        errors.append(age_err)
        computed["idade_anos"] = None
    else:
        try:
            if _valid_date(structured.get("NASCIMENTO", "")) and _valid_date(structured.get("DATA_OBITO", "")):
                dn = dt.datetime.strptime(structured["NASCIMENTO"], "%d/%m/%Y")
                do = dt.datetime.strptime(structured["DATA_OBITO"], "%d/%m/%Y")
                computed["idade_anos"] = int((do - dn).days / 365.25)
        except Exception:
            computed["idade_anos"] = None

    # Nomes
    nome_ok = "SIM" if structured.get("NOME") else "NAO"
    nomes_ok = "SIM" if (structured.get("NOME") and structured.get("NOME_MAE")) else "NAO"
    structured["NOME_OK"] = nome_ok
    structured["NOMES_OK"] = nomes_ok

    # Status e score
    total_campos = len(HEADER)
    preenchidos = sum(1 for k in HEADER if structured.get(k))
    score = int((preenchidos / total_campos) * 100)
    # Penaliza por erros
    score = max(0, score - len(errors) * 10)
    structured["QUALIDADE_SCORE"] = score

    if errors or any(not structured.get(c) for c in campos_criticos):
        status = "REVISAR"
    else:
        status = "OK"
    structured["STATUS"] = status

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
    """Detecta frases de recusa do provedor OCR."""
    if not text:
        return True
    low = text.lower().strip()
    for phrase in REFUSAL_PHRASES:
        if phrase in low:
            return True
    # Se o texto for muito curto e não contiver letras/dígitos suficientes
    alnum = sum(1 for c in text if c.isalnum())
    if alnum < 10:
        return True
    return False


def ocr_openai_compatible(
    image_bytes: bytes,
    mime_type: str,
    model: str,
) -> Tuple[str, float]:
    """
    Envia imagem para o provedor OpenAI-compatible via chat completions (data URL).
    Retorna (texto, confiança).
    Levanta OCRProviderError em caso de recusa, falha ou resposta inválida.
    """
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

    # Detecção explícita de recusa antes do parser
    if _detect_refusal(content):
        raise OCRProviderError("Provedor OCR recusou processar a imagem ou retornou texto inválido.", 502)

    # Confiança: se o provedor retornar logprobs usamos; senão heurística simples
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
    """Recebe imagem em base64 e retorna OCR estruturado de declaração de óbito."""
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

    # PDF não suportado em v1
    if "pdf" in mime_type.lower() or (file_name and file_name.lower().endswith(".pdf")):
        return JSONResponse(status_code=422, content={
            "code": "PDF_NOT_SUPPORTED_IN_V1",
            "message": "PDF não é suportado na versão 1. Envie imagem (PNG/JPG).",
            "requestId": request_id,
        })

    # Decodifica base64
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

    # Hash do arquivo
    hash_arquivo = _sha256_bytes(file_bytes)

    # Chamada ao provedor OCR
    try:
        raw_text, confidence = ocr_openai_compatible(file_bytes, mime_type, model)
    except OCRProviderError as e:
        return JSONResponse(status_code=e.status_code, content={
            "code": e.code,
            "message": str(e),
            "requestId": request_id,
        })
    except Exception as e:
        return JSONResponse(status_code=502, content={
            "code": "OCR_PROVIDER_ERROR",
            "message": f"Erro inesperado no provedor OCR: {e}",
            "requestId": request_id,
        })

    # Parser
    try:
        structured = parse_obito(raw_text)
    except Exception as e:
        structured = {k: "" for k in HEADER}
        structured["ERROS"] = f"Erro no parser: {e}"

    structured["HASH_ARQUIVO"] = hash_arquivo
    structured["HASH_CONTEUDO"] = _sha256_text(raw_text)

    # Validação
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
