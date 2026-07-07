# FILE: main.py

import os
import re
import json
import base64
import hashlib
import datetime
import unicodedata

import requests
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse


# =============================================================================
# CONFIGURAÇÃO E CONSTANTES
# =============================================================================

# Variáveis de ambiente esperadas no Render
AUTH_TOKEN = os.environ.get("ENDPOINT_AUTH_TOKEN", "")
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_MODEL_DEFAULT", "gpt-4o")
MAX_FILE_SIZE_MB = float(os.environ.get("MAX_FILE_SIZE_MB", "15"))
PORT = int(os.environ.get("PORT", "10000"))

# MIME types permitidos para OCR em V1
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "application/pdf"}

# Cabeçalho legado esperado pelo Apps Script — structured deve conter exatamente
# estas chaves e nesta ordem.
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

# Unidades federativas brasileiras válidas
UFS_VALIDAS = {
    'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS',
    'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC',
    'SP', 'SE', 'TO'
}

# Nomes dos meses em português para derivar NOME_MES
MESES_PT = {
    '01': 'janeiro', '02': 'fevereiro', '03': 'marco', '04': 'abril',
    '05': 'maio', '06': 'junho', '07': 'julho', '08': 'agosto',
    '09': 'setembro', '10': 'outubro', '11': 'novembro', '12': 'dezembro'
}

# CIDs considerados "garbage" (suspeitos/imprecisos) em declaração de óbito
GARBAGE_CID_SET = {
    'R99', 'R98', 'R97', 'R96', 'R95', 'R94', 'I46', 'I46.9', 'I46.0',
    'J96', 'J96.0', 'J96.9', 'P95', 'O80', 'A41', 'A41.9', 'U07.1', 'U07.2'
}

# -----------------------------------------------------------------------------
# PADRÕES DE RECUSA DO PROVEDOR OCR
# -----------------------------------------------------------------------------
# Frases que indicam que o provedor recusou processar a imagem ou retornou uma
# mensagem de bloqueio em vez do OCR real do documento. A detecção é feita em
# texto normalizado (sem acentos, minúsculo, sem pontuação) para cobrir
# variações de apostrofo (I'm / I'm) e codificação.
# A verificação é case-insensitive e tolerante a acentos/aspas diferentes.
REFUSAL_PATTERNS = [
    "i'm sorry",
    "i'm sorry",
    "i cant assist with that",
    "i cant assist with that",
    "i cannot assist",
    "cannot assist",
    "unable to help",
    "cannot help with that request",
    "i cant help with that",
    "i cant help with that",
    "desculpe",
    "nao posso ajudar",
    "nao posso ajudar com isso",
    "i am unable to",
    "i am not able to",
    "as an ai",
    "as a language model",
    "i cannot fulfill",
    "i cant fulfill",
    "i cannot complete",
    "i cant complete",
    "i cannot process",
    "i cant process",
    "i cannot provide",
    "i cant provide",
    "content policy",
    "safety policy",
    "i cannot analyze",
    "i cant analyze",
    "i cannot extract",
    "i cant extract",
    "i cannot read",
    "i cant read",
    "i cannot interpret",
    "i cant interpret",
    "this appears to be",
    "i can however",
    "i can help with",
    "please provide",
    "could you provide",
    "if youd like",
    "if you would like",
    "let me know",
    "feel free to",
]


# =============================================================================
# EXCEÇÕES PERSONALIZADAS
# =============================================================================

class OcrServiceError(Exception):
    """Exceção controlada do serviço de OCR."""
    def __init__(self, code, message, request_id, status=500):
        self.code = code
        self.message = message
        self.request_id = request_id
        self.status = status
        super().__init__(message)


# =============================================================================
# FUNÇÕES UTILITÁRIAS DE TEXTO
# =============================================================================

def _normalize_text(text):
    """Normaliza o texto bruto do OCR: remove caracteres invisíveis comuns,
    espaços extras e padroniza quebras de linha."""
    if not text:
        return ""
    # Substitui espaços não quebra-linha por espaço comum
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2007", " ")
    text = text.replace("\u202f", " ")
    # Remove caracteres de controle exceto quebra de linha e tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Padroniza quebras de linha
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Reduz espaços horizontais excessivos dentro de cada linha
    linhas = [re.sub(r"[ \t]+", " ", linha).strip() for linha in text.split("\n")]
    return "\n".join(linhas).strip()


def _lines(text):
    """Retorna apenas linhas úteis (não vazias) do texto normalizado."""
    texto = _normalize_text(text)
    return [linha.strip() for linha in texto.split("\n") if linha.strip()]


def _norm(s):
    """Normaliza string para comparação: minúscula, sem acentos, sem espaços/pontuação."""
    if not s:
        return ""
    s = s.lower()
    # Remove acentos
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Remove tudo que não for alfanumérico
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _norm_for_refusal(s):
    """Normaliza string para detecção de recusa: minúscula, sem acentos,
    mantendo espaços e apóstrofos removidos. Retorna texto sem pontuação
    mas com espaços preservados para casar frases completas."""
    if not s:
        return ""
    s = s.lower()
    # Remove acentos
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Remove apóstrofos e aspas (cobre I'm / I'm / I`m)
    s = s.replace("'", "").replace("'", "").replace("`", "").replace("’", "")
    s = s.replace("´", "")
    # Remove pontuação exceto espaços
    s = re.sub(r"[^a-z0-9\s]", "", s)
    # Reduz espaços múltiplos
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_provider_refusal(text):
    """Detecta explicitamente se o texto retornado pelo provedor OCR é uma
    recusa, desculpa automática, mensagem de bloqueio ou qualquer resposta
    que claramente não seja o OCR real do documento.

    Retorna True se o texto for vazio OU contiver algum padrão de recusa.
    A verificação é case-insensitive e tolerante a acentos e variações de
    apóstrofo (I'm / I'm).

    Esta função DEVE ser chamada logo após receber o texto do provedor,
    ANTES de chamar parse_obito, para impedir falso sucesso com status 200.
    """
    if not text:
        return True

    texto_norm = _norm_for_refusal(text)
    if not texto_norm:
        return True

    # Se o texto for muito curto (menos de 20 caracteres úteis), é suspeito.
    # Um OCR real de declaração de óbito sempre produz texto extenso.
    if len(texto_norm) < 20:
        return True

    for padrao in REFUSAL_PATTERNS:
        padrao_norm = _norm_for_refusal(padrao)
        if padrao_norm and padrao_norm in texto_norm:
            return True

    return False


def _normalizar_data_bruta(valor):
    """Normaliza datas em formatos com espaço, barra ou hífen para DD/MM/AAAA."""
    if not valor:
        return ""
    valor = valor.strip()
    # Remove textos residuais comuns vindos do OCR junto da data
    valor = re.sub(r"(?i)\bdata\b", "", valor).strip(": ")
    # Aceita separadores espaço, / ou -
    m = re.search(r"(\d{1,2})[\s/\-]+(\d{1,2})[\s/\-]+(\d{2,4})", valor)
    if not m:
        return ""
    dia, mes, ano = m.group(1), m.group(2), m.group(3)
    # Completa ano com 4 dígitos quando necessário
    if len(ano) == 2:
        ano = "19" + ano if int(ano) > 30 else "20" + ano
    # Zero à esquerda em dia/mês
    dia = dia.zfill(2)
    mes = mes.zfill(2)
    return f"{dia}/{mes}/{ano}"


def _find_value_after_index(lines, start_index):
    """Retorna o conteúdo da próxima linha útil a partir de start_index + 1."""
    if not lines:
        return ""
    for j in range(start_index + 1, len(lines)):
        val = lines[j].strip(": ")
        if val:
            return val
    return ""


def _find_next_line_value(text, labels):
    """Quando o rótulo está sozinho em uma linha, retorna o valor da próxima
    linha útil. Evita capturar o próprio rótulo como valor."""
    linhas = _lines(text)
    labels_norm = [_norm(l) for l in labels]
    for i, linha in enumerate(linhas):
        linha_norm = _norm(linha)
        for lbl in labels_norm:
            if linha_norm == lbl or linha_norm.endswith(lbl):
                if i + 1 < len(linhas):
                    return linhas[i + 1].strip(": ")
    return ""


def _find_block_value(text, labels, max_distance=3):
    """Busca valor em bloco: rótulo pode estar na mesma linha ou nas próximas
    linhas até max_distance. Ignora capturar o próprio rótulo como valor."""
    linhas = _lines(text)
    labels_norm = [_norm(l) for l in labels]
    for i, linha in enumerate(linhas):
        linha_norm = _norm(linha)
        for lbl in labels_norm:
            if lbl in linha_norm:
                # Tenta valor na mesma linha após o rótulo
                idx = linha_norm.find(lbl)
                resto = linha[idx + len(lbl):].strip(": \t")
                if resto and _norm(resto) != lbl:
                    return resto
                # Caso contrário, pega próxima linha útil
                val = _find_value_after_index(linhas, i)
                if val and _norm(val) != lbl:
                    return val
    return ""


def _extract_date(text, labels):
    """Extrai data suportando valores na mesma linha ou na linha seguinte,
    com separadores espaço, barra ou hífen."""
    linhas = _lines(text)
    labels_norm = [_norm(l) for l in labels]
    for i, linha in enumerate(linhas):
        linha_norm = _norm(linha)
        for lbl in labels_norm:
            if lbl in linha_norm:
                # Tenta extrair data da própria linha
                data = _normalizar_data_bruta(linha)
                if data:
                    return data
                # Tenta extrair da próxima linha útil
                val = _find_value_after_index(linhas, i)
                if val:
                    data = _normalizar_data_bruta(val)
                    if data:
                        return data
    return ""


def _extract_causes(text):
    """Captura apenas causas reais em ordem, ignorando títulos e marcadores da
    seção de causas do óbito. CAUSA_BASICA deve ser a última causa não vazia."""
    linhas = _lines(text)
    titulos_norm = {
        _norm(t) for t in [
            "CAUSAS DA MORTE", "Parte I", "Parte II", "CID",
            "Devido ou como consequencia de:",
            "Intervalo entre o inicio e a morte",
            "Causa da morte", "Causas da morte",
            "CAUSA DA MORTE", "CAUSAS DA MORTE",
        ]
    }

    # Localiza início da seção de causas
    inicio = None
    for i, linha in enumerate(linhas):
        linha_norm = _norm(linha)
        if "causasdamorte" in linha_norm or "causadamorte" in linha_norm:
            inicio = i + 1
            break

    # Se não encontrou a seção, tenta varrer todo o texto
    if inicio is None:
        inicio = 0

    causas = []
    for linha in linhas[inicio:]:
        linha_norm = _norm(linha)
        # Ignora títulos e marcadores
        if linha_norm in titulos_norm:
            continue
        # Ignora linhas que começam com "parte i" ou "parte ii"
        if re.match(r"(?i)^parte\s*(i|ii|iii)\b", linha.strip()):
            continue
        # Ignora linhas que parecem CID (ex.: A41, A41.9, U07.1)
        if re.match(r"(?i)^[a-z]\d{2}(\.\d+)?$", linha.strip()):
            continue
        # Ignora linhas com "devido" ou "consequencia" ou "intervalo"
        if re.search(r"(?i)devido|consequencia|intervalo", linha):
            continue
        # Ignora linhas muito curtas ou apenas símbolos
        if len(linha_norm) < 3:
            continue
        # Ignora possíveis rótulos residuais
        if re.search(r"(?i)^\s*(cid|parte|causa)\b", linha):
            continue
        # Ignora linhas que são apenas fragmentos do rótulo "causa da morte"
        if linha_norm in ("damorte", "samorte", "causa", "causas"):
            continue
        causas.append(linha.strip())

    # Remove duplicatas consecutivas mantendo ordem
    causas_limpas = []
    for c in causas:
        if not causas_limpas or causas_limpas[-1] != c:
            causas_limpas.append(c)

    return causas_limpas


def _extract_cids(text):
    """Extrai códigos CID-10 do texto (ex.: A41, A41.9, U07.1) em ordem de
    aparição."""
    if not text:
        return []
    matches = re.findall(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b", text)
    # Remove duplicatas consecutivas
    cids = []
    for c in matches:
        if not cids or cids[-1] != c:
            cids.append(c)
    return cids


def _detect_garbage_cids(cids):
    """Retorna lista de CIDs suspeitos (garbage) entre os informados."""
    if not cids:
        return []
    return [c for c in cids if c in GARBAGE_CID_SET]


def _nome_mes_from_data(data):
    """Deriva o nome do mês em português a partir de uma data DD/MM/AAAA."""
    if not data:
        return ""
    partes = data.split("/")
    if len(partes) != 3:
        return ""
    mes = partes[1].zfill(2)
    return MESES_PT.get(mes, "")


def _sha256_bytes(b):
    """Calcula SHA256 de bytes e retorna hexadecimal."""
    if not b:
        return ""
    return hashlib.sha256(b).hexdigest()


def _sha256_text(text):
    """Calcula SHA256 do texto e retorna hexadecimal."""
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_valid_date_ddmmaaaa(data):
    """Valida se a string está no formato DD/MM/AAAA e é uma data plausível."""
    if not data:
        return False
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", data)
    if not m:
        return False
    dia, mes, ano = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if mes < 1 or mes > 12:
        return False
    if dia < 1 or dia > 31:
        return False
    if ano < 1800 or ano > 2100:
        return False
    return True


def _is_valid_hora(hora):
    """Valida se a string está no formato HH:MM."""
    if not hora:
        return False
    m = re.match(r"^(\d{2}):(\d{2})$", hora)
    if not m:
        return False
    h, mi = int(m.group(1)), int(m.group(2))
    return 0 <= h <= 23 and 0 <= mi <= 59


def _is_valid_cep(cep):
    """Valida CEP com 8 dígitos."""
    if not cep:
        return True  # CEP ausente não é erro
    digitos = re.sub(r"\D", "", cep)
    return len(digitos) == 8


def _parse_data_to_date(data):
    """Converte DD/MM/AAAA para datetime.date, ou None se inválida."""
    if not _is_valid_date_ddmmaaaa(data):
        return None
    try:
        return datetime.datetime.strptime(data, "%d/%m/%Y").date()
    except ValueError:
        return None


# =============================================================================
# PARSER PRINCIPAL DE DECLARAÇÃO DE ÓBITO
# =============================================================================

def parse_obito(raw_text, file_bytes):
    """Extrai campos estruturados do texto bruto do OCR de uma declaração de
    óbito brasileira. Retorna um dict com exatamente as chaves de HEADER,
    na mesma ordem."""

    # Inicializa structured com todas as chaves de HEADER na ordem correta
    structured = {chave: "" for chave in HEADER}

    texto = _normalize_text(raw_text or "")
    linhas = _lines(texto)

    # --- Nome do falecido ---
    # Corrige caso em que o rótulo "Nome do Falecido" está em uma linha
    # e o nome real está na linha seguinte.
    nome = _find_block_value(texto, ["Nome do Falecido", "Nome do falecido", "Nome"])
    if nome and _norm(nome) in ("nomedofalecido", "dofalecido", "do", "falecido"):
        nome = _find_next_line_value(texto, ["Nome do Falecido", "Nome do falecido"])
    structured["NOME"] = nome or ""

    # --- Nome social ---
    structured["NOME_SOCIAL"] = _find_block_value(
        texto, ["Nome social", "Nome Social"]
    ) or ""

    # --- Nome da mãe ---
    structured["NOME_MAE"] = _find_block_value(
        texto, ["Nome da mae", "Nome da mãe", "Mae", "Mãe"]
    ) or ""

    # --- Nome do pai ---
    structured["NOME_PAI"] = _find_block_value(
        texto, ["Nome do pai", "Pai"]
    ) or ""

    # --- Sexo ---
    sexo = _find_block_value(texto, ["Sexo"], max_distance=2)
    if sexo:
        sexo_lower = sexo.strip().lower()[:1]
        if sexo_lower in ("m", "f"):
            structured["SEXO"] = "M" if sexo_lower == "m" else "F"
        else:
            structured["SEXO"] = sexo.strip()[:20]

    # --- Raça/cor ---
    structured["RACA_COR"] = _find_block_value(
        texto, ["Raca/cor", "Raca cor", "Raça/cor", "Raça cor", "Cor"]
    ) or ""

    # --- Estado civil ---
    structured["ESTADO_CIVIL"] = _find_block_value(
        texto, ["Estado civil"]
    ) or ""

    # --- Nacionalidade ---
    structured["NACIONALIDADE"] = _find_block_value(
        texto, ["Nacionalidade"]
    ) or ""

    # --- Profissão ---
    structured["PROFISSAO"] = _find_block_value(
        texto, ["Profissao", "Profissão", "Ocupacao", "Ocupação"]
    ) or ""

    # --- Logradouro ---
    structured["LOGRADOURO"] = _find_block_value(
        texto, ["Logradouro", "Endereco", "Endereço"]
    ) or ""

    # --- Número ---
    structured["NUMERO"] = _find_block_value(
        texto, ["Numero", "Número"], max_distance=2
    ) or ""

    # --- Complemento ---
    structured["COMPLEMENTO"] = _find_block_value(
        texto, ["Complemento"]
    ) or ""

    # --- Bairro ---
    structured["BAIRRO"] = _find_block_value(
        texto, ["Bairro"]
    ) or ""

    # --- Cidade (residência) ---
    structured["CIDADE"] = _find_block_value(
        texto, ["Municipio de residencia", "Município de residência", "Cidade de residencia"]
    ) or ""

    # --- UF (residência) ---
    uf_res = _find_block_value(texto, ["UF"], max_distance=2)
    if uf_res:
        m_uf = re.search(r"\b([A-Z]{2})\b", uf_res)
        structured["UF"] = m_uf.group(1) if m_uf else uf_res.upper()[:2]

    # --- CEP ---
    cep = _find_block_value(texto, ["CEP", "Cep"], max_distance=2)
    if cep:
        m_cep = re.search(r"(\d{5}\-?\d{3})", cep)
        if m_cep:
            structured["CEP"] = re.sub(r"\D", "", m_cep.group(1))

    # --- Cidade de nascimento ---
    structured["CIDADE_NASCIMENTO"] = _find_block_value(
        texto, ["Municipio de nascimento", "Município de nascimento",
                "Cidade de nascimento", "Naturalidade"]
    ) or ""

    # --- UF de nascimento ---
    uf_nasc = _find_block_value(
        texto, ["UF de nascimento", "UF nascimento"], max_distance=2
    )
    if uf_nasc:
        m_uf = re.search(r"\b([A-Z]{2})\b", uf_nasc)
        structured["UF_NASCIMENTO"] = m_uf.group(1) if m_uf else uf_nasc.upper()[:2]

    # --- CPF ---
    cpf = _find_block_value(texto, ["CPF", "Cpf"], max_distance=2)
    if cpf:
        m_cpf = re.search(r"(\d{3}\.?\d{3}\.?\d{3}\-?\d{2})", cpf)
        if m_cpf:
            structured["CPF"] = re.sub(r"\D", "", m_cpf.group(1))

    # --- RG ---
    structured["RG"] = _find_block_value(
        texto, ["RG", "Identidade", "Registro geral"], max_distance=2
    ) or ""

    # --- Órgão emissor RG ---
    structured["ORGAO_EMISSOR_RG"] = _find_block_value(
        texto, ["Orgao emissor", "Órgão emissor", "Orgao expedidor"]
    ) or ""

    # --- Data de nascimento ---
    # Corrige datas com espaço como "03 03 1952"
    structured["NASCIMENTO"] = _extract_date(
        texto, ["Data de nascimento", "Nascimento", "Data Nascimento",
                "Data de Nascimento"]
    ) or ""

    # --- Data do óbito ---
    # Corrige datas com espaço como "09 05 2020"
    structured["DATA_OBITO"] = _extract_date(
        texto, ["Data do obito", "Data do óbito", "Data obito", "Data óbito",
                "Data do falecimento", "Data de obito"]
    ) or ""

    # --- Hora do óbito ---
    hora_obito = ""
    # Primeiro tenta buscar por rótulo "Hora"
    hora_val = _find_block_value(texto, ["Hora", "Hora do obito", "Hora do óbito"], max_distance=2)
    if hora_val:
        m_hora = re.search(r"(\d{1,2}[:hH]\d{2})", hora_val)
        if m_hora:
            hora_obito = m_hora.group(1).replace("h", ":").replace("H", ":")
    # Fallback: procura qualquer hora no texto
    if not hora_obito:
        m_hora = re.search(r"(\d{1,2}[:hH]\d{2})", texto)
        if m_hora:
            hora_obito = m_hora.group(1).replace("h", ":").replace("H", ":")
    if hora_obito:
        partes = hora_obito.split(":")
        if len(partes) == 2:
            hora_obito = f"{partes[0].zfill(2)}:{partes[1].zfill(2)}"
    structured["HORA_OBITO"] = hora_obito

    # --- Local do óbito ---
    structured["LOCAL_OBITO"] = _find_block_value(
        texto, ["Local do obito", "Local do óbito", "Local de obito", "Local de óbito"]
    ) or ""

    # --- Cidade do óbito (município de ocorrência) ---
    # Corrige extração quando está em linha separada do rótulo
    cidade_obito = _find_block_value(
        texto, ["Municipio de ocorrencia", "Município de ocorrência",
                "Municipio de obito", "Município de óbito",
                "Cidade do obito", "Cidade do óbito"]
    )
    if not cidade_obito:
        cidade_obito = _find_next_line_value(
            texto, ["Municipio de ocorrencia", "Município de ocorrência"]
        )
    structured["CIDADE_OBITO"] = cidade_obito or ""

    # --- UF do óbito ---
    # Busca UF próxima ao município de ocorrência
    uf_obito = ""
    # Estratégia 1: procura rótulo "UF" após município de ocorrência
    linhas_norm = [_norm(l) for l in linhas]
    idx_ocorrencia = None
    for i, ln in enumerate(linhas_norm):
        if "municipiodeocorrencia" in ln or "municipiodeobito" in ln:
            idx_ocorrencia = i
            break
    if idx_ocorrencia is not None:
        # Procura UF nas próximas 4 linhas
        for j in range(idx_ocorrencia, min(idx_ocorrencia + 5, len(linhas))):
            if _norm(linhas[j]) == "uf":
                if j + 1 < len(linhas):
                    uf_obito = linhas[j + 1].strip(": ")
                break
            m_uf = re.search(r"\b([A-Z]{2})\b", linhas[j])
            if m_uf and m_uf.group(1) in UFS_VALIDAS:
                uf_obito = m_uf.group(1)
                break
    # Estratégia 2: fallback genérico
    if not uf_obito:
        uf_obito = _find_block_value(texto, ["UF"], max_distance=2)
    if uf_obito:
        m_uf = re.search(r"\b([A-Z]{2})\b", uf_obito)
        uf_obito = m_uf.group(1) if m_uf else uf_obito.upper()[:2]
    structured["UF_OBITO"] = uf_obito or ""

    # --- Causas da morte ---
    causas = _extract_causes(texto)
    if causas:
        structured["CAUSA_MORTE"] = causas[0] if len(causas) > 0 else ""
        structured["CAUSA_MORTE_2"] = causas[1] if len(causas) > 1 else ""
        structured["CAUSA_MORTE_3"] = causas[2] if len(causas) > 2 else ""
        structured["CAUSA_MORTE_4"] = causas[3] if len(causas) > 3 else ""
        structured["CAUSA_MORTE_5"] = causas[4] if len(causas) > 4 else ""
        # CAUSA_BASICA deve ser a última causa não vazia da cadeia
        structured["CAUSA_BASICA"] = causas[-1]
    else:
        structured["CAUSA_BASICA"] = ""

    # --- CIDs ---
    cids = _extract_cids(texto)
    if cids:
        # CID_BASICA deve refletir o CID da causa básica (última da cadeia)
        # Heurística: último CID encontrado corresponde à causa básica
        structured["CID_BASICA"] = cids[-1] if cids else ""
        structured["CID_MORTE"] = cids[0] if len(cids) > 0 else ""
        structured["CID_MORTE_2"] = cids[1] if len(cids) > 1 else ""
        structured["CID_MORTE_3"] = cids[2] if len(cids) > 2 else ""
        structured["CID_MORTE_4"] = cids[3] if len(cids) > 3 else ""
        structured["CID_MORTE_5"] = cids[4] if len(cids) > 4 else ""
        # Códigos numéricos espelham os CIDs
        structured["CODIGO_CAUSA_BASICA"] = structured["CID_BASICA"]
        structured["CODIGO_CAUSA_MORTE"] = structured["CID_MORTE"]
        structured["CODIGO_CAUSA_MORTE_2"] = structured["CID_MORTE_2"]
        structured["CODIGO_CAUSA_MORTE_3"] = structured["CID_MORTE_3"]
        structured["CODIGO_CAUSA_MORTE_4"] = structured["CID_MORTE_4"]
        structured["CODIGO_CAUSA_MORTE_5"] = structured["CID_MORTE_5"]

    # --- Tipo de óbito ---
    structured["TIPO_OBITO"] = _find_block_value(
        texto, ["Tipo de obito", "Tipo de óbito"]
    ) or ""

    # --- Assistido ---
    structured["ASSISTIDO"] = _find_block_value(
        texto, ["Assistido", "Foi assistido"]
    ) or ""

    # --- Data do atestado ---
    structured["DATA_ATESTADO"] = _extract_date(
        texto, ["Data do atestado", "Data do ato medico", "Data do ato médico",
                "Data de emissao", "Data de emissão"]
    ) or ""

    # --- NOMES_OK e NOME_OK (heurística simples) ---
    nomes_presentes = []
    for campo in ["NOME", "NOME_MAE", "NOME_PAI"]:
        val = structured.get(campo, "")
        if val and _norm(val) not in ("nomedofalecido", "nomedamae", "nomedopai",
                                       "dofalecido", "damae", "dopai", ""):
            nomes_presentes.append(campo)
    structured["NOMES_OK"] = "SIM" if len(nomes_presentes) >= 2 else "NAO"
    structured["NOME_OK"] = "SIM" if structured["NOME"] and _norm(structured["NOME"]) not in (
        "nomedofalecido", "dofalecido", "falecido", "") else "NAO"

    # --- GARBAGE_CODES e QTD_GARBAGE ---
    garbage = _detect_garbage_cids(cids)
    structured["GARBAGE_CODES"] = ", ".join(garbage)
    structured["QTD_GARBAGE"] = len(garbage)

    # --- PROTOCOLO_TEV (pode ficar vazio) ---
    structured["PROTOCOLO_TEV"] = ""

    # --- Hashes ---
    structured["HASH_ARQUIVO"] = _sha256_bytes(file_bytes)
    structured["HASH_CONTEUDO"] = _sha256_text(raw_text or "")

    # --- NOME_MES derivado de DATA_OBITO ---
    structured["NOME_MES"] = _nome_mes_from_data(structured["DATA_OBITO"])

    # --- DATA_PROCESSAMENTO (timestamp ISO UTC) ---
    structured["DATA_PROCESSAMENTO"] = datetime.datetime.utcnow().isoformat() + "Z"

    # --- Validação e qualidade ---
    validation = validate_obito(structured, raw_text or "")
    structured["QUALIDADE_SCORE"] = compute_quality_score(structured, validation)
    structured["STATUS"] = validation.get("status", "REVISAR")
    structured["ERROS"] = " | ".join(validation.get("errors", []))

    return structured


# =============================================================================
# VALIDAÇÃO
# =============================================================================

def validate_obito(structured, raw_text=""):
    """Valida os campos estruturados e retorna um objeto validation com:
    ok, errors, warnings, computed."""
    errors = []
    warnings = []
    computed = {}

    # Valida datas
    for campo in ["NASCIMENTO", "DATA_OBITO", "DATA_ATESTADO"]:
        val = structured.get(campo, "")
        if val and not _is_valid_date_ddmmaaaa(val):
            errors.append(f"{campo} com formato inválido: {val}")

    # Valida hora
    hora = structured.get("HORA_OBITO", "")
    if hora and not _is_valid_hora(hora):
        errors.append(f"HORA_OBITO com formato inválido: {hora}")

    # Valida UF do óbito
    uf_obito = structured.get("UF_OBITO", "")
    if uf_obito and uf_obito not in UFS_VALIDAS:
        errors.append(f"UF_OBITO inválida: {uf_obito}")

    # Valida UF de residência
    uf_res = structured.get("UF", "")
    if uf_res and uf_res not in UFS_VALIDAS:
        warnings.append(f"UF de residência inválida: {uf_res}")

    # Valida CEP quando presente
    cep = structured.get("CEP", "")
    if cep and not _is_valid_cep(cep):
        errors.append(f"CEP inválido: {cep}")

    # Coerência entre NASCIMENTO e DATA_OBITO
    data_nasc = _parse_data_to_date(structured.get("NASCIMENTO", ""))
    data_obito = _parse_data_to_date(structured.get("DATA_OBITO", ""))
    if data_nasc and data_obito:
        if data_obito < data_nasc:
            errors.append("DATA_OBITO anterior à NASCIMENTO")
        else:
            idade = (data_obito - data_nasc).days // 365
            computed["idade_calculada"] = idade
            # Se houver idade explícita no texto, verifica coerência
            m_idade = re.search(r"(?i)idade\D{0,5}(\d{1,3})", raw_text)
            if m_idade:
                idade_texto = int(m_idade.group(1))
                computed["idade_texto"] = idade_texto
                if abs(idade - idade_texto) > 1:
                    warnings.append(
                        f"Idade do texto ({idade_texto}) diverge da calculada ({idade})"
                    )

    # Campos críticos ausentes
    campos_criticos = ["NOME", "NASCIMENTO", "DATA_OBITO", "NOME_MAE",
                       "CIDADE_OBITO", "UF_OBITO", "CAUSA_BASICA"]
    faltantes = [c for c in campos_criticos if not structured.get(c, "")]
    for c in faltantes:
        warnings.append(f"Campo crítico ausente: {c}")

    # Determina status
    status = "OK"
    if errors or len(faltantes) >= 3:
        status = "REVISAR"

    ok = len(errors) == 0 and status == "OK"

    return {
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "computed": computed,
        "status": status
    }


def compute_quality_score(structured, validation):
    """Calcula score de qualidade de 0 a 100 baseado em campos críticos
    presentes e ausência de erros."""
    score = 0
    total = 0

    # Campos críticos (peso 10 cada)
    campos_criticos = ["NOME", "NASCIMENTO", "DATA_OBITO", "NOME_MAE",
                       "CIDADE_OBITO", "UF_OBITO", "CAUSA_BASICA"]
    for campo in campos_criticos:
        total += 10
        if structured.get(campo, ""):
            score += 10

    # Campos complementares (peso 5 cada)
    campos_comp = ["NOME_PAI", "HORA_OBITO", "LOCAL_OBITO", "SEXO",
                   "CID_BASICA", "DATA_ATESTADO", "CEP", "CPF"]
    for campo in campos_comp:
        total += 5
        if structured.get(campo, ""):
            score += 5

    # Penaliza erros
    num_errors = len(validation.get("errors", []))
    score -= num_errors * 10

    # Penaliza warnings levemente
    num_warnings = len(validation.get("warnings", []))
    score -= num_warnings * 2

    # Garante intervalo 0-100
    score = max(0, min(100, score))
    return score


# =============================================================================
# ADAPTADOR OCR (OpenAI-compatible)
# =============================================================================

def call_ocr_provider(file_bytes, mime_type, model, request_id):
    """Adaptador para provedor OCR compatível com OpenAI Chat Completions.

    Este é o ponto de ajuste do provedor: se no futuro for necessário trocar
    para outro serviço (Google Vision, Azure, etc.), basta modificar esta
    função mantendo o mesmo contrato de retorno (dict com text e confidence).

    Usa OPENAI_API_URL como endpoint base e envia a imagem como data URL.
    """
    if not OPENAI_API_URL:
        raise OcrServiceError(
            code="OCR_PROVIDER_NOT_CONFIGURED",
            message="OPENAI_API_URL não configurada.",
            request_id=request_id,
            status=500
        )
    if not OPENAI_API_KEY:
        raise OcrServiceError(
            code="OCR_PROVIDER_NOT_CONFIGURED",
            message="OPENAI_API_KEY não configurada.",
            request_id=request_id,
            status=500
        )

    # Converte bytes para base64 e monta data URL
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    # Prompt para extração de texto da declaração de óbito
    prompt = (
        "Você é um especialista em OCR de declarações de óbito brasileiras. "
        "Extraia TODO o texto visível no documento, preservando a ordem e as "
        "quebras de linha originais. Mantenha rótulos e valores exatamente como "
        "aparecem. Não invente dados. Retorne apenas o texto extraído."
    )

    payload = {
        "model": model or OPENAI_MODEL_DEFAULT,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.0
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(
            OPENAI_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
    except requests.exceptions.Timeout:
        raise OcrServiceError(
            code="OCR_PROVIDER_TIMEOUT",
            message="Provedor OCR demorou demais para responder.",
            request_id=request_id,
            status=504
        )
    except requests.exceptions.RequestException as e:
        raise OcrServiceError(
            code="OCR_PROVIDER_ERROR",
            message=f"Erro de comunicação com provedor OCR: {str(e)}",
            request_id=request_id,
            status=502
        )

    if resp.status_code != 200:
        raise OcrServiceError(
            code="OCR_PROVIDER_ERROR",
            message=f"Provedor OCR retornou HTTP {resp.status_code}: {resp.text[:500]}",
            request_id=request_id,
            status=502
        )

    try:
        data = resp.json()
    except Exception:
        raise OcrServiceError(
            code="OCR_PROVIDER_ERROR",
            message="Resposta do provedor OCR não é JSON válido.",
            request_id=request_id,
            status=502
        )

    # Extrai texto da resposta no formato OpenAI Chat Completions
    text = ""
    try:
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            text = msg.get("content", "") or ""
    except Exception:
        text = ""

    if not text:
        raise OcrServiceError(
            code="OCR_PROVIDER_ERROR",
            message="Provedor OCR retornou texto vazio ou não retornou OCR válido.",
            request_id=request_id,
            status=502
        )

    # -----------------------------------------------------------------------
    # DETECÇÃO EXPLÍCITA DE RECUSA DO PROVEDOR
    # -----------------------------------------------------------------------
    # Verifica se o provedor recusou processar a imagem ou retornou uma
    # mensagem de bloqueio (ex.: "I'm sorry, I can't assist with that.").
    # Esta veração ocorre ANTES de qualquer parseamento, impedindo falso
    # sucesso com status 200 e structured vazio.
    if is_provider_refusal(text):
        raise OcrServiceError(
            code="OCR_PROVIDER_ERROR",
            message=(
                "O provedor OCR recusou processar a imagem ou não retornou OCR "
                "válido do documento. Resposta recebida não corresponde ao "
                "conteúdo de uma declaração de óbito."
            ),
            request_id=request_id,
            status=502
        )

    # Confidence heurística: provedores OpenAI não retornam confidence direta,
    # então usamos 0.85 como padrão quando há texto válido.
    confidence = 0.85

    return {"text": text, "confidence": confidence}


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="obito-ocr-service", version="3.1.0")


@app.get("/health")
async def health():
    """Endpoint de saúde para monitoramento do Render."""
    return {"status": "ok", "service": "obito-ocr-service", "version": "3.1.0"}


@app.post("/ocr")
async def ocr(request: Request, authorization: str = Header(default="")):
    """Endpoint principal de OCR de declaração de óbito.

    Recebe JSON com: file (base64), mimeType, model (opcional),
    fileName (opcional), fileId (opcional), requestId (opcional).

    Retorna: text, confidence, provider, requestId, warnings, rawText,
    structured, validation, headerOrder.
    """
    request_id = ""

    # --- Autenticação Bearer ---
    if AUTH_TOKEN:
        if not authorization:
            return JSONResponse(
                status_code=401,
                content={
                    "code": "UNAUTHORIZED",
                    "message": "Token de autenticação ausente.",
                    "requestId": request_id
                }
            )
        token = authorization.replace("Bearer ", "").strip()
        if token != AUTH_TOKEN:
            return JSONResponse(
                status_code=401,
                content={
                    "code": "UNAUTHORIZED",
                    "message": "Token de autenticação inválido.",
                    "requestId": request_id
                }
            )

    # --- Parse do JSON ---
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "code": "INVALID_JSON",
                "message": "Corpo da requisição não é JSON válido.",
                "requestId": request_id
            }
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={
                "code": "INVALID_JSON",
                "message": "JSON deve ser um objeto.",
                "requestId": request_id
            }
        )

    # --- Campos obrigatórios ---
    file_b64 = body.get("file", "")
    mime_type = body.get("mimeType", "")
    model = body.get("model")
    file_name = body.get("fileName", "")
    file_id = body.get("fileId", "")
    request_id = body.get("requestId", "") or ""

    if not file_b64:
        return JSONResponse(
            status_code=400,
            content={
                "code": "MISSING_FILE",
                "message": "Campo 'file' (base64) é obrigatório.",
                "requestId": request_id
            }
        )

    if not mime_type:
        return JSONResponse(
            status_code=400,
            content={
                "code": "MISSING_MIME_TYPE",
                "message": "Campo 'mimeType' é obrigatório.",
                "requestId": request_id
            }
        )

    # --- Validação de MIME type ---
    if mime_type not in ALLOWED_MIME:
        return JSONResponse(
            status_code=415,
            content={
                "code": "UNSUPPORTED_MIME_TYPE",
                "message": f"mimeType '{mime_type}' não suportado. Permitidos: {', '.join(sorted(ALLOWED_MIME))}.",
                "requestId": request_id
            }
        )

    # --- PDF não suportado em V1 ---
    if mime_type == "application/pdf":
        return JSONResponse(
            status_code=422,
            content={
                "code": "PDF_NOT_SUPPORTED_IN_V1",
                "message": "PDF não é suportado nesta versão. Envie imagem (JPEG, PNG ou WebP).",
                "requestId": request_id
            }
        )

    # --- Decodifica base64 ---
    try:
        file_bytes = base64.b64decode(file_b64, validate=True)
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "code": "INVALID_BASE64",
                "message": "Campo 'file' não é base64 válido.",
                "requestId": request_id
            }
        )

    # --- Validação de tamanho ---
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return JSONResponse(
            status_code=413,
            content={
                "code": "FILE_TOO_LARGE",
                "message": f"Arquivo de {size_mb:.2f}MB excede o limite de {MAX_FILE_SIZE_MB}MB.",
                "requestId": request_id
            }
        )

    # --- Chama provedor OCR ---
    # O adaptador call_ocr_provider já inclui a detecção explícita de recusa
    # do provedor (is_provider_refusal) ANTES de retornar o texto. Se o
    # provedor recusar ou retornar texto vazio/bloqueio, uma OcrServiceError
    # com code=OCR_PROVIDER_ERROR e status=502 é levantada aqui.
    try:
        ocr_result = call_ocr_provider(file_bytes, mime_type, model, request_id)
    except OcrServiceError as e:
        return JSONResponse(
            status_code=e.status,
            content={
                "code": e.code,
                "message": e.message,
                "requestId": e.request_id
            }
        )

    raw_text = ocr_result["text"]
    confidence = ocr_result["confidence"]

    # --- Parser estruturado ---
    # Neste ponto o texto já passou pela detecção de recusa e é considerado
    # OCR válido do documento. O parser só é chamado quando há texto real.
    try:
        structured = parse_obito(raw_text, file_bytes)
    except Exception as e:
        # Se o parser falhar, ainda retorna o texto bruto com aviso
        structured = {chave: "" for chave in HEADER}
        structured["HASH_ARQUIVO"] = _sha256_bytes(file_bytes)
        structured["HASH_CONTEUDO"] = _sha256_text(raw_text)
        structured["DATA_PROCESSAMENTO"] = datetime.datetime.utcnow().isoformat() + "Z"
        structured["STATUS"] = "REVISAR"
        structured["ERROS"] = f"Erro no parser: {str(e)}"
        structured["QUALIDADE_SCORE"] = 0

    # --- Validação final ---
    validation = validate_obito(structured, raw_text)

    # --- Warnings consolidados ---
    warnings = list(validation.get("warnings", []))
    if structured.get("QTD_GARBAGE", 0) > 0:
        warnings.append(f"CIDs suspeitos detectados: {structured.get('GARBAGE_CODES', '')}")

    # --- Resposta compatível com Apps Script ---
    response = {
        "text": raw_text,
        "confidence": confidence,
        "provider": "openai-compatible",
        "requestId": request_id,
        "warnings": warnings,
        "rawText": raw_text,
        "structured": structured,
        "validation": {
            "ok": validation.get("ok", False),
            "errors": validation.get("errors", []),
            "warnings": validation.get("warnings", []),
            "computed": validation.get("computed", {})
        },
        "headerOrder": list(HEADER)
    }

    return JSONResponse(status_code=200, content=response)


# =============================================================================
# EXECUÇÃO LOCAL / RENDER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
