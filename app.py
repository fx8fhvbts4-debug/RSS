import streamlit as st
import feedparser
from newspaper import Article, ArticleException
import google.generativeai as genai
import time
from datetime import datetime, timedelta
import nltk
from bs4 import BeautifulSoup

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configuração da Página
st.set_page_config(
    page_title="Personal News Aggregator AI",
    page_icon="🗞️",
    layout="wide"
)

# Estilização Customizada
st.markdown("""
<style>
    /* Reset e Fonte */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Fundo Geral */
    .stApp {
        background-color: #F8F9FA;
    }

    /* Cards de Notícias */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        /* Tenta focar nos blocos internos, mas o melhor é usar st.container com border=True */
    }
    
    /* Botões */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Cabeçalhos */
    h1, h2, h3 {
        color: #1A1A1A;
    }
    
    a {
        text-decoration: none;
        color: #2563EB !important; /* Azul moderno */
    }
    a:hover {
        text-decoration: underline;
    }
    
    /* Ajuste da Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Lógica de Backend (Funções)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)  # Cache feeds por 1 hora
def fetch_feeds(urls):
    """
    Coleta notícias de uma lista de URLs RSS e as ordena cronologicamente.
    """
    all_news = []
    fixed_now = datetime.utcnow() # Snapshot de tempo para ordenação estável
    
    for url in urls:
        if not url.strip():
            continue
            
        try:
            feed = feedparser.parse(url)
            feed_title = feed.feed.get('title', url)
            
            for entry in feed.entries:
                # Tenta normalizar a data correta
                # feedparser.parsed devolve struct_time.
                raw_published = entry.get('published') or entry.get('updated') or str(datetime.now())
                published_parsed = entry.get('published_parsed') or entry.get('updated_parsed')
                
                has_time = True
                if published_parsed:
                    dt = datetime(*published_parsed[:6])
                else:
                    # Parse falhou (provavelmente data em PT-BR que o feedparser nao entende)
                    # Não vamos usar NOW, vamos usar uma data dummy ou tentar parser na display
                    # Por enquanto, mantemos NOW para ordenação, mas marcamos has_time=False para UI saber
                    dt = fixed_now
                    has_time = False

                # Tenta encontrar imagem (Estratégia Avançada e Robusta)
                image_url = None
                
                # 1. Media Content (MediaRSS padrão)
                if 'media_content' in entry:
                    # Pega o primeiro que parecer imagem
                    for media in entry.media_content:
                        if media.get('medium') == 'image' or media.get('type', '').startswith('image/'):
                            image_url = media.get('url')
                            break
                            
                # 2. Media Thumbnail (comum em feeds de vídeo/notícias)
                if not image_url and 'media_thumbnail' in entry and len(entry.media_thumbnail) > 0:
                    image_url = entry.media_thumbnail[0].get('url')
                    
                # 3. Enclosures (Anexos padrão RSS)
                if not image_url and 'enclosures' in entry:
                    for enclosure in entry.enclosures:
                        if enclosure.get('type', '').startswith('image/'):
                            image_url = enclosure.get('href')
                            break
                            
                # 4. Links com rel='enclosure' ou type='image/'
                if not image_url and 'links' in entry:
                    for link in entry.links:
                        if link.get('type', '').startswith('image/'):
                            image_url = link.get('href')
                            break

                # 5. Parsing do HTML (Resumo ou Conteúdo) via BeautifulSoup
                # Isso pega imagens incorporadas no texto da notícia
                if not image_url:
                    content_to_parse = ""
                    if 'summary' in entry:
                        content_to_parse += entry.summary
                    if 'content' in entry:
                        for c in entry.content:
                            content_to_parse += c.value
                    
                    if content_to_parse:
                        try:
                            soup = BeautifulSoup(content_to_parse, 'html.parser')
                            img_tag = soup.find('img')
                            if img_tag and img_tag.get('src'):
                                image_url = img_tag['src']
                        except Exception:
                            pass
                
                all_news.append({
                    'title': entry.get('title', 'Sem Título'),
                    'link': entry.get('link', ''),
                    'source': feed_title,
                    'published': dt,
                    'raw_date': raw_published, # Guardamos a string original
                    'has_time': has_time,
                    'summary_rss': entry.get('summary', ''),
                    'image': image_url
                })
        except Exception as e:
            st.error(f"Erro ao processar feed {url}: {e}")
            
    # Ordena: Mais recentes primeiro
    all_news.sort(key=lambda x: x['published'], reverse=True)
    return all_news

@st.cache_data(show_spinner=False)
def extract_article_content(url):
    """
    Extrai o texto principal de uma notícia usando newspaper3k.
    """
    if not url:
        return None
        
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except ArticleException:
        # Fallback simples se newspaper falhar, ou apenas retorna None
        return None
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def summarize_with_gemini(text, api_key):
    """
    Gera um resumo da notícia usando Google Gemini.
    Retorna 3 bullet points e um emoji.
    """
    if not text or len(text) < 100:
        return "⚠️ Conteúdo muito curto ou indisponível para resumo."

    genai.configure(api_key=api_key)
    
    # Lista de modelos baseada na chave do usuário (Early Access / Internal Versions)
    models_to_try = [
        'gemini-2.5-flash',
        'gemini-2.0-flash',
        'gemini-flash-latest',
        'gemini-2.5-pro',
        'gemini-pro-latest'
    ]
    
    errors = []
    
    prompt = f"""
    Você é um assistente especialista em resumos de notícias.
    Analise o texto abaixo e gere um resumo curto, neutro e informativo, em um único parágrafo fluido.
    Mantenha um tom jornalístico e objetivo, similar ao da notícia original.
    
    Texto da notícia:
    {text[:8000]}  # Limita tamanho para evitar tokens excessivos
    """

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Registra o erro e tenta o próximo
            errors.append(f"{model_name}: {str(e)}")
            continue
            
    # Se chegou aqui, nenhum funcionou. Vamos listar o que ESTÁ disponível.
    # Se chegou aqui, nenhum funcionou
    error_msg = "\n".join(errors)
    return f"❌ Falha ao gerar resumo. Erros:\n{error_msg}"

# -----------------------------------------------------------------------------
# Interface do Usuário (SideBar)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Interface do Usuário (Google News Style)
# -----------------------------------------------------------------------------

# CSS Apple HIG Style
st.markdown("""
<style>
    /* 1. Tipografia Global Apple System */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* 2. Cores de Fundo - iOS Light Gray - Adjusted for contrast */
    .stApp {
        background-color: #EBEBF0 !important;
    }

    /* Remove sidebar space if needed and adjust container */
    [data-testid="stSidebar"] { display: none; }
    
    .block-container {
        padding-top: 3rem;
        padding-bottom: 5rem;
        max-width: 900px;
    }

    h1 {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1C1C1E;
        font-weight: 700;
        font-size: 32px;
        letter-spacing: -0.5px;
        margin-bottom: 24px;
    }

    /* 3. Cartões de Notícias (Apple Widget Style) */
    /* Target o container com borda do Streamlit */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 18px !important;
        border: none !important; /* Remove borda do Streamlit */
        box-shadow: 0 6px 16px rgba(0,0,0,0.08) !important;
        padding: 24px !important; /* Espaço generoso */
        margin-bottom: 20px;
    }

    /* 4. Botões (iOS Action Style) - Principalmente "Resumo com IA" */
    .stButton button {
        background-color: #007AFF !important; /* Apple Blue */
        color: white !important;
        border: none !important;
        border-radius: 20px !important; /* Pill shape */
        padding: 8px 20px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: opacity 0.2s ease-in-out;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .stButton button:hover {
        background-color: #007AFF !important;
        color: white !important;
        border: none !important;
        opacity: 0.8 !important; /* Efeito de opacidade iOS */
    }
    
    .stButton button:active {
        opacity: 0.6;
    }

    /* 5. Tipografia Títulos e Metadados */
    .news-title a {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #1C1C1E !important; /* Preto Sólido */
        font-weight: 600 !important; /* Semi-Bold */
        font-size: 20px !important;
        text-decoration: none;
        letter-spacing: -0.01em;
        line-height: 1.3;
    }
    .news-title a:hover {
        opacity: 0.7; /* Feedback visual sutil */
        color: #1C1C1E !important;
        text-decoration: none;
    }
    
    .news-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    /* Fonte e Data Metadados */
    .news-source, .news-meta, .news-time {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #8E8E93 !important; /* Cinza Secundário Apple */
        font-size: 13px !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .news-summary {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 15px;
        color: #3A3A3C; /* Cinza texto corpo */
        line-height: 1.5;
        margin-top: 12px;
        margin-bottom: 16px;
        opacity: 0.9;
    }

    /* 6. Estilização do Contêiner de Resumo (AI Summary Box) */
    div[data-testid="stAlert"] {
        background-color: #F2F5F8 !important; /* Azul-acinzentado suave */
        border: none !important;
        border-radius: 12px !important;
        padding: 16px !important; /* Padding reduzido e compacto */
        color: #333333 !important;
    }
    
    div[data-testid="stAlert"] p {
        font-size: 15px !important;
        line-height: 1.4 !important;
        color: #333333 !important;
    }

</style>
""", unsafe_allow_html=True)

# --- Top Bar (Apple News Header) ---
# Date Formatting Logic (Manual for safety vs locale)
days_pt = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
months_pt = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
now = datetime.now()
date_str = f"{days_pt[now.weekday()]}, {now.day} DE {months_pt[now.month-1]}".upper()

col_header, col_refresh = st.columns([3, 1])

with col_header:
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <div style="font-family: -apple-system, sans-serif; font-size: 13px; font-weight: 600; color: #8E8E93; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px;">
            {date_str}
        </div>
        <div style="font-family: -apple-system, sans-serif; font-size: 34px; font-weight: 800; color: #1C1C1E; letter-spacing: -0.5px; line-height: 1.1;">
            Briefing Diário
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_refresh:
    # Espaçamento para alinhar verticalmente com o título
    st.write("") 
    st.write("")
    
    # Botão Ghost style
    if st.button("Atualizar", key="refresh_btn", help="Buscar novas notícias"):
        st.cache_data.clear()
        st.rerun()

# CSS Específico para o botão Ghost no Header e Correção Mobile
st.markdown("""
<style>
    /* Remove padding padrão do container block principal para o header ficar alinhado */
    div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlock"] {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        height: 100%;
    }

    button[key="refresh_btn"] {
        background-color: transparent !important;
        border: none !important;
        color: #007AFF !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: none !important;
        padding: 0px !important;
    }
    button[key="refresh_btn"]:hover {
        color: #0056b3 !important;
        background-color: transparent !important;
        opacity: 0.8;
    }
    button[key="refresh_btn"]:active {
        color: #007AFF !important;
        background-color: transparent !important;
        opacity: 0.5;
    }

    /* --- MOBILE TWEAKS --- */
    @media (max-width: 576px) {
        /* Força o primeiro bloco horizontal (Header) a ficar lado a lado */
        div[data-testid="stHorizontalBlock"]:nth-of-type(1) {
            flex-direction: row !important;
            align-items: flex-start !important; /* Alinha topo ou center conforme pref */
            gap: 0px !important;
        }
        
        /* Ajusta largura das colunas no mobile */
        /* Coluna do Título */
        div[data-testid="stHorizontalBlock"]:nth-of-type(1) > div[data-testid="column"]:nth-of-type(1) {
            width: 70% !important;
            flex: none !important;
            min-width: 0 !important;
        }
        
        /* Coluna do Botão */
        div[data-testid="stHorizontalBlock"]:nth-of-type(1) > div[data-testid="column"]:nth-of-type(2) {
            width: 30% !important;
            flex: none !important;
            display: flex;
            justify-content: flex-end;
            align-items: center; /* Centraliza verticalmente o botão */
            padding-top: 10px; /* Ajuste fino para alinhar visualmente com o título grande */
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Feed Logic (Carregado apenas dos Secrets) ---
if "RSS_FEEDS" in st.secrets:
    rss_urls = [url.strip() for url in st.secrets["RSS_FEEDS"].split('\n') if url.strip()]
else:
    st.warning("⚠️ Nenhuma fonte de notícias configurada em .streamlit/secrets.toml")
    rss_urls = []
    
# Check de API Key
if "GOOGLE_API_KEY" in st.secrets:
    api_key_input = st.secrets["GOOGLE_API_KEY"]
else:
    # Fallback se não tiver no secrets (o que não deve acontecer se estiver tudo config)
    api_key_input = None
    st.error("⚠️ Google API Key não encontrada em secrets.toml")

@st.cache_data(show_spinner=False)
def get_article_image(url):
    """
    Tenta capturar a imagem principal do artigo usando newspaper3k
    caso o feed RSS não tenha fornecido.
    """
    if not url:
        return None
    try:
        article = Article(url)
        article.download()
        article.parse()
        if article.top_image:
            return article.top_image
    except:
        return None
    return None

    return None

if not rss_urls:
    pass # Mensagem de warning ja exibida acima
else:
    with st.spinner("Atualizando feed..."):
        news_items = fetch_feeds(rss_urls)
        
        # Filtrar apenas notícias das últimas 2 horas (7200 segundos)
        # Feedparser retorna UTC naive, então comparamos com utcnow naive
        now_utc = datetime.utcnow()
        news_items = [
            item for item in news_items 
            if 0 <= (now_utc - item['published']).total_seconds() <= 7200
        ]
    
    if not news_items:
        st.warning("Nenhuma notícia encontrada nas últimas 2 horas.")
    else:
        # Layout em Grid/Lista
        for i, item in enumerate(news_items[:20]): # Limite fixo de 20 para performance
            
            # Container Customizado (Card com Fundo Levemente Escuro)
            with st.container(border=True):
                # Calcular tempo com fuso horário ajustado (Brasil -3h simples)
                # O RSS vem em UTC. Vamos converter para exibir local.
                dt_utc = item['published']
                has_time = item.get('has_time', True)
                
                if has_time:
                    dt_brazil = dt_utc - timedelta(hours=3)
                    # Formata HH:MM sempre
                    display_str = dt_brazil.strftime('%d/%m %H:%M')
                else:
                    # Se não conseugimos parsear (has_time=False), mostramos a string original do RSS
                    # Isso evita mostrar a hora atual errada.
                    # Ex: "Wed, 24 Dec 2024 10:00:00 GMT" ou "24/12/2024"
                    # Tentamos limpar um pouco se for muito longa
                    raw = item.get('raw_date', '')
                    display_str = raw[:25] + "..." if len(raw) > 30 else raw

                # --- Cabeçalho do Card (Fonte) ---
                st.markdown(f"""
                <div class='news-header-row'>
                    <div class='news-source'>
                        📰 {item['source']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # --- Título ---
                st.markdown(f"<span class='news-title'>[{item['title']}]({item['link']})</span>", unsafe_allow_html=True)
                
                # --- Data/Hora (Abaixo do Título) ---
                st.markdown(f"<div class='news-meta' style='margin-bottom:8px;'>📅 {display_str}</div>", unsafe_allow_html=True)
                
                # --- Resumo RSS (Texto do Card) ---
                # Limpa HTML básico se tiver, mas aqui vamos simples
                summary_rss = item.get('summary_rss', '')
                if summary_rss:
                    # Remove tags HTML básicas para o preview ficar bonito (opcional, mas bom pra UX)
                    import re
                    clean_summary = re.sub('<[^<]+?>', '', summary_rss)[:200] + "..."
                    st.markdown(f"<div class='news-summary'>{clean_summary}</div>", unsafe_allow_html=True)
                
                # --- Imagem (Grande, Embaixo) ---
                # Fallback: Se não tiver imagem do RSS, tenta buscar ao exibir
                display_image = item.get('image')
                if not display_image:
                    display_image = get_article_image(item['link'])
                    
                if display_image:
                     st.image(display_image, use_container_width=True)
                
                st.markdown("") # Espaçamento
                
                # --- Ações ---
                if api_key_input:
                    if st.button(f"✨ Resumir com IA", key=f"btn_{i}", help="Gera um resumo rápido usando Gemini"):
                        with st.spinner("Lendo e resumindo..."):
                            text = extract_article_content(item['link'])
                            if text:
                                resumo = summarize_with_gemini(text, api_key_input)
                                st.info(f"**Resumo da Notícia:**\n\n{resumo}")
                            else:
                                st.error("Erro ao ler artigo. (Site pode bloquear scrapers)")
