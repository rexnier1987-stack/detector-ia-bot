"""
=============================================================================
  DETECTOR DE IA — BOT DE TELEGRAM
  Detecta si un texto fue escrito por Claude, GPT-4, Gemini u otro LLM.
=============================================================================
"""

import os
import re
import math
import asyncio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from collections import Counter
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ── Token del bot (se lee desde variable de entorno para seguridad) ─────────
TOKEN = os.environ.get("TELEGRAM_TOKEN", "")

# ═══════════════════════════════════════════════════════════════════════════
#  MOTOR ESTADÍSTICO
# ═══════════════════════════════════════════════════════════════════════════

def tokenizar(texto: str) -> list:
    return re.findall(r'\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]{2,}\b', texto.lower())

def calcular_perplexidad(tokens: list) -> float:
    if len(tokens) < 3:
        return 50.0
    bigramas = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    conteo_bi  = Counter(bigramas)
    conteo_uni = Counter(tokens)
    vocab = len(set(tokens))
    log_prob = sum(
        math.log2((conteo_bi[bg] + 1) / (conteo_uni[bg[0]] + vocab))
        for bg in bigramas
    )
    return round(2 ** (-log_prob / len(bigramas)), 2)

def calcular_burstiness(tokens: list) -> float:
    if len(tokens) < 15:
        return 0.0
    scores = []
    for word, _ in Counter(tokens).most_common(30):
        pos = [i for i, t in enumerate(tokens) if t == word]
        if len(pos) < 2:
            continue
        iv = np.diff(pos).astype(float)
        mu = iv.mean()
        if mu == 0:
            continue
        cv = iv.std() / mu
        scores.append((cv - 1) / (cv + 1))
    return round(float(np.mean(scores)), 3) if scores else 0.0

def calcular_ttr(tokens: list) -> float:
    return round(len(set(tokens)) / len(tokens), 3) if tokens else 0.0

def calcular_varianza_oraciones(texto: str) -> float:
    ors = [o.strip() for o in re.split(r'[.!?]+', texto) if len(o.strip()) > 5]
    if len(ors) < 2:
        return 0.0
    return round(float(np.var([len(o.split()) for o in ors])), 2)

def calcular_conectores(texto: str) -> float:
    conectores = [
        'además','sin embargo','por lo tanto','en conclusión','asimismo',
        'cabe destacar','es importante','en este sentido','finalmente',
        'en primer lugar','por otro lado','en resumen','cabe mencionar',
        'es fundamental','es crucial','resulta importante','en definitiva',
        'dicho esto','vale la pena','en términos generales','en última instancia',
        'furthermore','however','therefore','in conclusion','notably',
        'moreover','thus','hence','it is important','in summary',
        'in addition','consequently','nevertheless','it is essential',
        'it is worth noting','delve into','in the realm of','tapestry',
    ]
    tl = texto.lower()
    hits = sum(1 for c in conectores if c in tl)
    return round(hits / max(len(tokenizar(texto)) / 100, 1), 3)

def calcular_vocabulario_ia(texto: str) -> float:
    palabras_robot = {
        'fundamental','crucial','esencial','primordial','relevante',
        'significativo','innegable','indispensable','invaluable',
        'exhaustivo','holístico','robusto','integral','óptimo',
        'delve','tapestry','nuanced','multifaceted','leverage',
        'streamline','paradigm','holistic','synergy','pivotal',
        'comprehensive','foster','realm','groundbreaking','transformative',
        'noteworthy','commendable','intricate','revolutionary',
    }
    tokens = set(tokenizar(texto))
    hits = len(tokens & palabras_robot)
    return round(hits / max(len(tokens) / 10, 1), 3)

def calcular_patron_estructura(texto: str) -> float:
    parrafos = [p.strip() for p in texto.split('\n') if len(p.strip()) > 20]
    if len(parrafos) < 2:
        return 0.0
    longs = [len(p.split()) for p in parrafos]
    mu = np.mean(longs)
    if mu == 0:
        return 0.0
    cv = np.std(longs) / mu
    return round(float(max(0.0, 1.0 - cv)), 3)

def calcular_longitud_media(texto: str) -> float:
    ors = [o.strip() for o in re.split(r'[.!?]+', texto) if len(o.strip()) > 5]
    if not ors:
        return 0.0
    return round(float(np.mean([len(o.split()) for o in ors])), 1)

def calcular_hedging(texto: str) -> float:
    hedges = [
        'podría decirse','en cierta medida','hasta cierto punto',
        'de alguna manera','en cierto sentido','cabe señalar',
        'es posible que','parece ser que','aparentemente',
        'it could be argued','to some extent','it seems that',
        'it appears that','one might say','arguably','seemingly',
    ]
    tl = texto.lower()
    hits = sum(1 for h in hedges if h in tl)
    return round(hits / max(len(tokenizar(texto)) / 100, 1), 3)

def calcular_score(metricas: dict) -> tuple:
    perp  = metricas['perplexidad']
    burst = metricas['burstiness']
    ttr   = metricas['ttr']
    var   = metricas['varianza']
    con   = metricas['conectores']
    pat   = metricas['patron']
    voc   = metricas['vocabulario_ia']
    lmo   = metricas['longitud_media']
    hdg   = metricas['hedging']

    perp_n  = max(0.0, min(1.0, 1 - (perp - 4) / 28))
    burst_n = max(0.0, min(1.0, (-burst + 0.2) / 0.7))
    ttr_n   = max(0.0, min(1.0, 1 - (ttr - 0.22) / 0.40))
    var_n   = max(0.0, min(1.0, 1 - var / 30))
    con_n   = min(1.0, con / 1.5)
    pat_n   = min(1.0, pat * 1.3)
    voc_n   = min(1.0, voc * 4)
    lmo_n   = max(0.0, min(1.0, (lmo - 8) / 22))
    hdg_n   = min(1.0, hdg / 1.2)

    score = (
        perp_n  * 0.17 +
        burst_n * 0.14 +
        ttr_n   * 0.12 +
        var_n   * 0.10 +
        con_n   * 0.15 +
        pat_n   * 0.09 +
        voc_n   * 0.13 +
        lmo_n   * 0.06 +
        hdg_n   * 0.04
    ) * 100 * 1.25

    score_ia = round(min(99.5, max(0.5, score)), 1)
    return score_ia, round(100 - score_ia, 1)

def analizar_oraciones(texto: str, score_global: float) -> list:
    palabras_robot = {
        'fundamental','crucial','esencial','primordial','invaluable',
        'exhaustivo','holístico','robusto','integral','óptimo',
        'delve','tapestry','nuanced','multifaceted','leverage',
        'streamline','paradigm','synergy','pivotal','comprehensive',
        'foster','realm','groundbreaking','transformative',
    }
    conectores_set = {
        'además','sin embargo','por lo tanto','en conclusión','asimismo',
        'finalmente','en primer lugar','por otro lado','en resumen',
        'furthermore','however','therefore','in conclusion','moreover',
        'thus','hence','in addition','consequently','nevertheless',
    }
    resultados = []
    for oracion in re.split(r'(?<=[.!?])\s+', texto.strip()):
        oracion = oracion.strip()
        if not oracion:
            continue
        tokens = tokenizar(oracion)
        if not tokens:
            resultados.append({'texto': oracion, 'etiqueta': 'neutral'})
            continue
        tset = set(tokens)
        score = (
            len(tset & palabras_robot) * 18 +
            len(tset & conectores_set) * 15 +
            (1 - len(tset) / len(tokens)) * 20 +
            (12 if len(tokens) > 22 else 0) +
            score_global * 0.25
        )
        score = min(100, max(0, score))
        umbral_ia = 45 if score_global > 60 else 55
        umbral_h  = 28 if score_global < 40 else 35
        if score >= umbral_ia:
            etiqueta = 'ia'
        elif score <= umbral_h:
            etiqueta = 'humano'
        else:
            etiqueta = 'neutral'
        resultados.append({'texto': oracion, 'etiqueta': etiqueta})
    return resultados


# ═══════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE IMAGEN (gráfico de tarta)
# ═══════════════════════════════════════════════════════════════════════════

def generar_imagen_resultado(score_ia: float, score_h: float) -> BytesIO:
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#0a0e14')
    ax.set_facecolor('#0a0e14')

    wedge_props = dict(linewidth=2, edgecolor='#0a0e14')
    _, _, autotexts = ax.pie(
        [score_ia, score_h], explode=(0.04, 0.04),
        colors=['#ff4d6d', '#00e676'], autopct='%1.1f%%',
        startangle=90, wedgeprops=wedge_props, pctdistance=0.65,
    )
    for at in autotexts:
        at.set_color('#0a0e14')
        at.set_fontsize(13)
        at.set_fontweight('bold')

    ax.add_patch(plt.Circle((0, 0), 0.40, color='#0a0e14'))
    ax.text(0, 0.08, f"{score_ia:.0f}%", ha='center', va='center',
            fontsize=22, fontweight='bold', color='#ff4d6d')
    ax.text(0, -0.18, "IA", ha='center', va='center',
            fontsize=11, color='#aaaaaa')
    ax.legend(
        handles=[
            mpatches.Patch(color='#ff4d6d', label=f'IA — {score_ia:.1f}%'),
            mpatches.Patch(color='#00e676', label=f'Humano — {score_h:.1f}%'),
        ],
        loc='lower center', bbox_to_anchor=(0.5, -0.08),
        ncol=2, frameon=False,
        labelcolor=['#ff4d6d', '#00e676'],
        prop={'size': 10}
    )
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', facecolor='#0a0e14', dpi=130)
    buf.seek(0)
    plt.close(fig)
    return buf


# ═══════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DEL TEXTO RESALTADO (en formato Telegram)
# ═══════════════════════════════════════════════════════════════════════════

def generar_texto_resaltado(analisis: list) -> str:
    """
    Telegram no soporta HTML de colores, así que usamos emojis:
    🔴 = probable IA
    🟢 = probable humano
    ⚪ = indeterminado
    """
    partes = []
    for item in analisis:
        if item['etiqueta'] == 'ia':
            partes.append(f"🔴 {item['texto']}")
        elif item['etiqueta'] == 'humano':
            partes.append(f"🟢 {item['texto']}")
        else:
            partes.append(f"⚪ {item['texto']}")
    return '\n'.join(partes)


# ═══════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DEL REPORTE
# ═══════════════════════════════════════════════════════════════════════════

def generar_reporte(score_ia: float, metricas: dict, analisis: list) -> str:
    n    = len(analisis)
    n_ia = sum(1 for o in analisis if o['etiqueta'] == 'ia')
    n_h  = sum(1 for o in analisis if o['etiqueta'] == 'humano')

    if score_ia >= 75:
        veredicto = "⚠️ MUY PROBABLE que sea IA"
    elif score_ia >= 55:
        veredicto = "⚡ INDICIOS SIGNIFICATIVOS de IA"
    elif score_ia >= 35:
        veredicto = "🔍 POSIBLE asistencia de IA"
    else:
        veredicto = "✅ Probable AUTORÍA HUMANA"

    m = metricas
    return f"""
📋 *REPORTE FORENSE*

*Veredicto:* {veredicto}
*Score:* IA={score_ia:.1f}% | Humano={100-score_ia:.1f}%

📐 *Métricas estadísticas:*
• Perplejidad: `{m['perplexidad']}` {'⚠' if m['perplexidad']<20 else '✓'}
• Burstiness: `{m['burstiness']}` {'⚠' if m['burstiness']<0 else '✓'}
• Riqueza léxica (TTR): `{m['ttr']}` {'⚠' if m['ttr']<0.45 else '✓'}
• Varianza oraciones: `{m['varianza']}` {'⚠' if m['varianza']<15 else '✓'}
• Conectores formales: `{m['conectores']}` {'⚠' if m['conectores']>1.2 else '✓'}
• Vocabulario IA típico: `{m['vocabulario_ia']}` {'⚠' if m['vocabulario_ia']>0.3 else '✓'}
• Uniformidad estructura: `{m['patron']}` {'⚠' if m['patron']>0.6 else '✓'}
• Long. media oraciones: `{m['longitud_media']}` {'⚠' if m['longitud_media']>20 else '✓'}

📊 *Oraciones analizadas:* {n}
🔴 Marcadas como IA: {n_ia} ({n_ia/max(n,1)*100:.0f}%)
🟢 Marcadas como Humano: {n_h} ({n_h/max(n,1)*100:.0f}%)

_⚠️ Nota: Herramienta de apoyo, no infalible._
""".strip()


# ═══════════════════════════════════════════════════════════════════════════
#  HANDLERS DEL BOT
# ═══════════════════════════════════════════════════════════════════════════

async def comando_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Hola! Soy el Detector de IA.*\n\n"
        "Envíame cualquier texto y te diré si fue escrito por una IA "
        "(Claude, ChatGPT, Gemini...) o por un humano.\n\n"
        "📌 *Cómo usarme:*\n"
        "Simplemente pega el texto aquí y te respondo con:\n"
        "• 📊 Gráfico con el porcentaje IA vs Humano\n"
        "• 🖍️ Texto resaltado por oración\n"
        "• 📋 Reporte técnico detallado\n\n"
        "⚡ Para mejores resultados, usa textos de más de 150 palabras.",
        parse_mode='Markdown'
    )

async def comando_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ *Ayuda*\n\n"
        "• Envía cualquier texto y lo analizo automáticamente.\n"
        "• Funciona mejor con textos de +150 palabras.\n"
        "• 🔴 = oración probable IA\n"
        "• 🟢 = oración probable humano\n"
        "• ⚪ = indeterminado\n\n"
        "Comandos disponibles:\n"
        "/start — Mensaje de bienvenida\n"
        "/ayuda — Esta ayuda",
        parse_mode='Markdown'
    )

async def analizar_texto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    texto = update.message.text.strip()

    # Validación mínima
    if len(texto) < 50:
        await update.message.reply_text(
            "⚠️ El texto es demasiado corto.\n"
            "Necesito al menos 50 caracteres para analizarlo."
        )
        return

    # Aviso de que está procesando
    msg = await update.message.reply_text("🔬 Analizando texto...")

    tokens = tokenizar(texto)
    metricas = {
        'perplexidad':   calcular_perplexidad(tokens),
        'burstiness':    calcular_burstiness(tokens),
        'ttr':           calcular_ttr(tokens),
        'varianza':      calcular_varianza_oraciones(texto),
        'conectores':    calcular_conectores(texto),
        'patron':        calcular_patron_estructura(texto),
        'vocabulario_ia':calcular_vocabulario_ia(texto),
        'longitud_media':calcular_longitud_media(texto),
        'hedging':       calcular_hedging(texto),
    }

    score_ia, score_h = calcular_score(metricas)
    analisis          = analizar_oraciones(texto, score_ia)
    texto_resaltado   = generar_texto_resaltado(analisis)
    reporte           = generar_reporte(score_ia, metricas, analisis)
    imagen            = generar_imagen_resultado(score_ia, score_h)

    # Borrar mensaje "analizando..."
    await msg.delete()

    # 1 — Enviar gráfico
    await update.message.reply_photo(
        photo=imagen,
        caption=f"{'⚠️ PROBABLE IA' if score_ia > 55 else '✅ PROBABLE HUMANO'}  —  IA: {score_ia:.1f}% | Humano: {score_h:.1f}%"
    )

    # 2 — Enviar texto resaltado (limitado a 4000 chars por límite de Telegram)
    if len(texto_resaltado) > 4000:
        texto_resaltado = texto_resaltado[:3900] + "\n\n_(texto recortado por longitud)_"

    await update.message.reply_text(
        f"🖍️ *Texto resaltado por oración:*\n\n{texto_resaltado}",
        parse_mode='Markdown'
    )

    # 3 — Enviar reporte
    await update.message.reply_text(reporte, parse_mode='Markdown')


# ═══════════════════════════════════════════════════════════════════════════
#  ARRANQUE DEL BOT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if not TOKEN:
        raise ValueError("❌ No se encontró el token. Revisa la variable TELEGRAM_TOKEN.")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", comando_start))
    app.add_handler(CommandHandler("ayuda", comando_ayuda))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analizar_texto))

    print("✅ Bot iniciado. Esperando mensajes...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
