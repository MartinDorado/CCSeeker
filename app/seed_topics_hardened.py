
# seed_topics_hardened.py — domain-agnostic Channel-as-Seed helpers (locale-neutral)
import unicodedata, re, math
from collections import Counter
import streamlit as st

try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---- Normalization & tokenization ----
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn")

def _norm_text(s: str) -> str:
    return _strip_accents(s).lower().strip()

_WORD_RE = re.compile(r"[a-záéíóúñü]+", re.IGNORECASE)

# Language-specific stopwords
STOPWORDS_EN = {
    "the","and","for","with","from","this","that","these","those","about","into","over","under","very","more",
    "without","only","here","why","also","any","some","all","every","new","best","how","your","you","our","we"
}
STOPWORDS_ES = {
    "que","con","para","por","las","los","una","unos","unas","del","sus","este","esta","estas","estos",
    "sobre","entre","como","cuando","donde","desde","hasta","muy","más","mas","sin","solo","sólo","aqui","aquí",
    "porque","tambien","también","todo","toda","todos","todas","algo","algun","algún","alguna","algunas","algunos"
}
STOPWORDS_COMMON = {"oficial","official","channel","canal","clips","clip","podcast","tv","shorts","live","directo","en","de","y"}

# Unified stopwords set used by analyzers
STOPWORDS = set().union(STOPWORDS_EN, STOPWORDS_ES, STOPWORDS_COMMON)

# --- Lightweight language detection + translation helpers ---
def _detect_language_from_texts(texts: list[str]) -> str:
    """Very simple detector: compare EN vs ES stopword hits over tokens.
    Returns 'es' or 'en'. Defaults to 'en' on ties/low-signal.
    """
    en_hits, es_hits = 0, 0
    for t in texts or []:
        for w in _tokens(t):
            if w in STOPWORDS_EN:
                en_hits += 1
            if w in STOPWORDS_ES:
                es_hits += 1
    if es_hits > en_hits:
        return "es"
    return "en"

def translate_terms_with_gemini(terms: list[str], target_language: str, gemini_api_key: str | None):
    """Translate topic terms into the target language using Gemini. Fallback: return original.
    Expected target_language codes like 'en','es','pt','fr','de'.
    """
    if not terms:
        return terms
    if not gemini_api_key or genai is None:
        return terms
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        prompt = f"""
Traduce estas frases de TÓPICOS al idioma objetivo ({target_language}).
Mantén el significado y la concisión. Devuelve una lista simple (una por línea),
sin agregar ni quitar elementos.

Tópicos de entrada:
{chr(10).join(f"- {t}" for t in terms)}
"""
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        out = []
        seen = set()
        for line in text.splitlines():
            t = line.strip().lstrip("-•·").strip().strip('"').strip("'")
            low = _norm_text(t)
            if not low or low in seen:
                continue
            seen.add(low)
            out.append(t)
        return out or terms
    except Exception:
        return terms


def _tokens(text: str):
    for m in _WORD_RE.finditer(_norm_text(text or "")):
        w = m.group(0)
        if len(w) >= 3 and not any(ch.isdigit() for ch in w):
            yield w

# ---- Generic context filters (no locale hardcoding) ----
try:
    import pycountry
    COUNTRIES = {c.name.lower() for c in pycountry.countries}
    COUNTRIES |= {"us","usa","uk","uae","eu"}  # common short forms
except Exception:
    COUNTRIES = {"united states","us","usa","united kingdom","uk","spain","mexico","peru","chile","argentina","brazil","france"}

MONTHS_ES = {"enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","setiembre","octubre","noviembre","diciembre"}
MONTHS_EN = {"january","february","march","april","may","june","july","august","september","october","november","december"}

ORG_WORDS   = {"universidad","universidades","facultad","escuela","instituto","fundacion","fundación","centro","asociacion","asociación","sociedad","colegio","departamento"}
EVENT_WORDS = {"jornada","jornadas","charla","encuentro","ciclo","seminario","congreso","ponencia","presentacion","presentación","evento","workshop","taller","webinar","tour","gira","festival","expo"}
MEDIA_PROMO = {"trailer","avance","estreno","cine","pelicula","película","viral","fenomeno","fenómeno","exito","éxito","ventas","promocion","promoción","suscribete","suscríbete","like","share"}

def token_set(s: str) -> set[str]:
    return set((s or "").split())

def looks_contextual(term: str, user_penalties: set[str] | None = None) -> bool:
    """Return True if a term looks like org/geo/event/promo/time noise."""
    tokens = token_set(term.lower())
    if not tokens:
        return True

    # user-defined penalties
    if user_penalties and (tokens & user_penalties):
        return True

    # numbers / years
    if any(any(ch.isdigit() for ch in t) for t in tokens):
        return True
    if any(t.isdigit() and (len(t) == 4 and t.startswith(("19","20"))) for t in tokens):
        return True

    # months
    if tokens & MONTHS_ES or tokens & MONTHS_EN:
        return True

    # countries (dynamic when pycountry is present)
    if any(t in COUNTRIES for t in tokens):
        return True

    # org / event / media-promo
    if tokens & ORG_WORDS:
        return True
    if tokens & EVENT_WORDS:
        return True
    if tokens & MEDIA_PROMO:
        return True

    return False

# ---- Optional Gemini “topic cleanup” (domain-agnostic) ----
def refine_topics_with_gemini(candidates, language="es", max_terms=6, ban_terms=None, gemini_api_key=None):
    """
    Use Gemini to rewrite/clean up candidates into 4–6 *topic* phrases (domain-agnostic).
    Removes people/brands/places/events. Falls back to candidates if Gemini is unavailable.
    """
    ban_terms = set((ban_terms or []))
    if not gemini_api_key or genai is None:
        return candidates  # Gemini not configured

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        prompt = f"""
Eres un asistente que recibe una lista de términos ruidosos extraídos de títulos/etiquetas de un canal de YouTube.
Devuelve entre 4 y {max_terms} temas concisos (en {language}) que representen las ÁREAS DE CONTENIDO del canal.
Reglas:
- NO usar nombres propios de personas, marcas, lugares, instituciones o eventos.
- Evitar palabras de logística (jornada, universidad, congreso, presentación, etc.).
- Preferir sustantivos o frases cortas de tema (p. ej., "salud mental", "fotografía nocturna", "trading intradía").
- Responder como lista simple, una por línea, sin guiones ni numeración.

Términos de entrada:
{chr(10).join(f"- {t}" for t in candidates)}

Términos prohibidos:
{", ".join(sorted(ban_terms)) if ban_terms else "—"}
"""
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        terms = [t.strip().strip('"').strip("'") for t in text.splitlines() if t.strip()]
        out, seen = [], set()
        for t in terms:
            low = _norm_text(t)
            if not low or any(w in ban_terms for w in low.split()):
                continue
            if low in seen:
                continue
            seen.add(low)
            out.append(t)
            if len(out) >= max_terms:
                break
        return out or candidates
    except Exception as e:
        st.warning(f"Gemini refinement skipped: {e}")
        return candidates

def topics_from_titles_with_gemini(titles, language="es", max_terms=6, ban_terms=None, gemini_api_key=None):
    ban_terms = set(ban_terms or [])
    if not gemini_api_key or genai is None:
        return []
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        prompt = f"""
A partir de estos títulos de videos de un canal, devuelve entre 4 y {max_terms} TEMAS de contenido (no personas, marcas, lugares ni eventos). 
Evita palabras de logística (jornada, conferencia, universidad, tour), países/ciudades, años/fechas y términos promocionales.
Responde como lista simple, una por línea.

Títulos:
{chr(10).join(f"- {t}" for t in titles if t.strip())}

Términos prohibidos:
{", ".join(sorted(ban_terms)) if ban_terms else "—"}
"""
        out = (model.generate_content(prompt).text or "").strip().splitlines()
        clean, seen = [], set()
        for t in out:
            t = t.strip().strip("-•").strip().strip('"').strip("'")
            low = _norm_text(t)
            if not low or low in seen:
                continue
            if any(w in ban_terms for w in low.split()):
                continue
            seen.add(low)
            clean.append(t)
            if len(clean) >= max_terms:
                break
        return clean
    except Exception:
        return []


# ---- Main function (domain-agnostic, hardened) ----
def analyze_seed_channel(
    youtube_service,
    seed_channel_id,
    max_seed_videos=30,
    top_k=10,
    use_gemini=True,
    gemini_api_key=None,
    language="auto",
    include_descriptions=False,  # default OFF: descriptions are noisy
    user_penalties: set[str] | None = None,
):
    """
    Domain-agnostic seed analyzer (hardened):
    - strips channel-name tokens
    - prefers TAGS > TITLE BIGRAMS > TITLE UNIGRAMS
    - ignores descriptions by default
    - filters geo/org/event/promotional terms
    - requires terms to recur across many videos
    - optional Gemini cleanup to finalize topic phrases
    """
    st.info(f"Analyzing seed channel: {seed_channel_id}…")

    # 1) Channel info
    ch = youtube_service.channels().list(part="snippet,contentDetails", id=seed_channel_id).execute()
    items = ch.get("items", [])
    if not items:
        st.error("Could not retrieve the seed channel info.")
        return None
    ch_snippet = items[0]["snippet"]
    ch_title = ch_snippet.get("title", "")
    uploads = items[0]["contentDetails"]["relatedPlaylists"].get("uploads")
    if not uploads:
        st.error("Seed channel has no public uploads playlist.")
        return None

    # 2) Name/brand stoplist from channel title
    name_bits = set(_tokens(ch_title)) | {"oficial","official","canal","channel","tv","podcast","clips"}

    # 3) Recent videos from uploads playlist
    vids = youtube_service.playlistItems().list(
        part="snippet", playlistId=uploads, maxResults=min(50, max_seed_videos)
    ).execute().get("items", [])
    if not vids:
        st.error("No recent videos to analyze for the seed channel.")
        return None

    # 3b) Fetch real video snippets to access tags/descriptions
    video_ids = []
    for it in vids:
        sn = it.get("snippet", {})
        rid = (sn.get("resourceId", {}) or {}).get("videoId")
        if rid:
            video_ids.append(rid)
    video_meta: dict[str, dict] = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        vresp = youtube_service.videos().list(part="snippet", id=",".join(chunk)).execute()
        for v in vresp.get("items", []):
            vid = v.get("id")
            vsn = v.get("snippet", {})
            video_meta[vid] = {
                "tags": vsn.get("tags", []) or [],
                "description": vsn.get("description", "") or "",
                "title": vsn.get("title", "") or "",
            }

    # 3c) Detect original language from titles + fetched tags
    sample_titles = [it["snippet"].get("title", "") for it in vids]
    sample_tags = []
    for vid in video_ids:
        sample_tags.extend(video_meta.get(vid, {}).get("tags", []) or [])
    orig_lang = _detect_language_from_texts(sample_titles + sample_tags)
    st.info(f"Detected seed language: {orig_lang.upper()}")

    # 4) Collect per-video candidate sets (titles+tags; descriptions optional)
    title_unigrams, title_bigrams, tag_terms, desc_unigrams = [], [], [], []
    for it in vids:
        sn = it["snippet"]
        title = sn.get("title", "") or ""
        # Pull tags/description from videos().list, not playlistItems
        vid = (sn.get("resourceId", {}) or {}).get("videoId")
        meta = video_meta.get(vid, {}) if vid else {}
        tags  = meta.get("tags", []) or []
        desc  = meta.get("description", "") or ""

        # tokens
        t_unis = [w for w in _tokens(title) if w not in STOPWORDS and w not in name_bits]
        t_bigs = []
        if len(t_unis) >= 2:
            for a, b in zip(t_unis, t_unis[1:]):
                if a in STOPWORDS or b in STOPWORDS:
                    continue
                t_bigs.append(f"{a} {b}")

        tag_unis = [w for w in _tokens(" ".join(tags)) if w not in STOPWORDS and w not in name_bits]

        d_unis = []
        if include_descriptions:
            d_unis = [w for w in _tokens(desc) if w not in STOPWORDS and w not in name_bits]

        title_unigrams.append(set(t_unis))
        title_bigrams.append(set(t_bigs))
        tag_terms.append(set(tag_unis))
        desc_unigrams.append(set(d_unis))

    n = len(title_unigrams)
    if n == 0:
        st.error("No analyzable tokens found in the seed channel.")
        return None

    # 5) Document frequency per source
    df_title_uni, df_title_bi, df_tags, df_desc_uni = Counter(), Counter(), Counter(), Counter()
    for s in title_unigrams: df_title_uni.update(s)
    for s in title_bigrams: df_title_bi.update(s)
    for s in tag_terms:     df_tags.update(s)
    for s in desc_unigrams: df_desc_uni.update(s)

    min_df = max(2, math.ceil(0.20 * n))  # present in ≥20% of videos

    # 6) Score candidates (tags > title bigrams > title unigrams > desc) with generic context filter
    scored = []

    for t, df in df_tags.items():
        if df >= min_df and not looks_contextual(t, user_penalties):
            scored.append((t, df * 2.0))      # tags strongest

    for t, df in df_title_bi.items():
        if df >= min_df and not looks_contextual(t, user_penalties):
            scored.append((t, df * 1.6))      # title bigrams next

    for t, df in df_title_uni.items():
        if df >= min_df and not looks_contextual(t, user_penalties):
            scored.append((t, df * 1.0))      # title unigrams last

    if include_descriptions:
        for t, df in df_desc_uni.items():
            if df >= min_df and not looks_contextual(t, user_penalties):
                scored.append((t, df * 0.5))  # descriptions weakest

    # Prefer phrases over single tokens: keep bigrams first, fill with singles
    if scored:
        best = {}
        for term, sc in scored:
            best[term] = max(sc, best.get(term, 0.0))
        phrases = {t: s for t, s in best.items() if " " in t}
        singles = {t: s for t, s in best.items() if " " not in t}
        cand = [t for t, _ in sorted(phrases.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        if len(cand) < top_k:
            need = top_k - len(cand)
            cand += [t for t, _ in sorted(singles.items(), key=lambda x: x[1], reverse=True)[:need]]
    else:
        cand = []

    # final name/brand guard
    cand = [t for t in cand if all(w not in name_bits for w in t.split())]

    # --- NEW: robust fallback if we still have nothing ---
    if not cand:
        # a) Try turning on descriptions just for fallback scoring (still filtered)
        include_desc_backup = True
        if include_desc_backup:
            # recompute a light-weight score using descriptions (weak weight)
            scored_fallback = []
            for t, df in df_desc_uni.items():
                if df >= max(2, math.ceil(0.20 * n)) and not looks_contextual(t, user_penalties):
                    scored_fallback.append((t, df * 0.5))
            if scored_fallback:
                # prefer phrases, then singles
                best = {}
                for term, sc in scored_fallback:
                    best[term] = max(sc, best.get(term, 0.0))
                phrases = [t for t, _ in sorted(((t,s) for t,s in best.items() if " " in t), key=lambda x: x[1], reverse=True)]
                singles = [t for t, _ in sorted(((t,s) for t,s in best.items() if " " not in t), key=lambda x: x[1], reverse=True)]
                cand = (phrases + singles)[:top_k]
                cand = [t for t in cand if all(w not in name_bits for w in t.split())]

            # b) If still empty, ask Gemini to infer topics from titles directly
            if not cand and use_gemini and gemini_api_key:
                titles_only = [sn.get("title","") for it in vids for sn in [it["snippet"]]]
                inferred = topics_from_titles_with_gemini(
                    titles_only, language=language, max_terms=min(6, top_k),
                    ban_terms=name_bits, gemini_api_key=gemini_api_key
                )
                cand = inferred[:top_k]

    # 7) Optional Gemini cleanup to snap to topic phrases (in ORIGINAL language)
    if use_gemini and gemini_api_key and cand:
        cand = refine_topics_with_gemini(
            cand, language=orig_lang, max_terms=min(6, top_k),
            ban_terms=name_bits | COUNTRIES | ORG_WORDS | EVENT_WORDS | MEDIA_PROMO | MONTHS_ES | MONTHS_EN,
            gemini_api_key=gemini_api_key
        )

    # 7b) Optional translation of the final topic phrases into requested output language
    # language == 'auto' or 'original' => keep original
    desired = (language or "auto").lower()
    if cand and desired not in ("auto", "original") and desired in {"en","es","pt","fr","de"} and desired != orig_lang:
        cand = translate_terms_with_gemini(cand, desired, gemini_api_key)

    # 8) Build query (quote multi-word)
    final, seen = [], set()
    for t in cand:
        low = _norm_text(t)
        if not low or low in seen:
            continue
        seen.add(low)
        final.append(t if " " not in t else f'"{t}"')
        if len(final) >= 6:
            break

    if not final:
        st.error("Could not derive meaningful keywords from the seed channel.")
        return None

    new_query = " OR ".join(final)
    st.success(f"Generated topic query from seed channel: {new_query}")
    return new_query
