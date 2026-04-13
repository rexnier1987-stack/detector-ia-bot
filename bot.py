import os, re, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from collections import Counter
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN = "8613319422:AAFUSej7AEApb1_bLd-AtYoe2it6O-F2-WQ"

def tokenizar(texto):
    return re.findall(r'\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]{2,}\b', texto.lower())

def calcular_perplexidad(tokens):
    if len(tokens) < 3:
        return 50.0
    bigramas = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    cb = Counter(bigramas)
    cu = Counter(tokens)
    v  = len(set(tokens))
    lp = sum(math.log2((cb[b]+1)/(cu[b[0]]+v)) for b in bigramas)
    return round(2**(-lp/len(bigramas)), 2)

def calcular_burstiness(tokens):
    if len(tokens) < 15:
        return 0.0
    scores = []
    for w, _ in Counter(tokens).most_common(30):
        pos = [i for i, t in enumerate(tokens) if t == w]
        if len(pos) < 2:
            continue
        iv = np.diff(pos).astype(float)
        mu = iv.mean()
        if mu == 0:
            continue
        cv = iv.std() / mu
        scores.append((cv-1)/(cv+1))
    return round(float(np.mean(scores)), 3) if scores else 0.0

def calcular_ttr(tokens):
    return round(len(set(tokens))/len(tokens), 3) if tokens else 0.0

def calcular_varianza(texto):
    ors = [o.strip() for o in re.split(r'[.!?]+', texto) if len(o.strip()) > 5]
    if len(ors) < 2:
        return 0.0
    return round(float(np.var([len(o.split()) for o in ors])), 2)

def calcular_conectores(texto):
    lista = [
        'ademas','sin embargo','por lo tanto','en conclusion','asimismo',
        'cabe destacar','es importante','finalmente','en primer lugar',
        'por otro lado','en resumen','es fundamental','es crucial',
        'en definitiva','en terminos generales','en ultima instancia',
        'furthermore','however','therefore','in conclusion','moreover',
        'thus','hence','in addition','consequently','nevertheless',
        'it is worth noting','delve into','tapestry',
    ]
    tl = texto.lower()
    hits = sum(1 for c in lista if c in tl)
    return round(hits / max(len(tokenizar(texto))/100, 1), 3)

def calcular_vocabulario_ia(texto):
    robot = {
        'fundamental','crucial','esencial','primordial','invaluable',
        'exhaustivo','holistico','robusto','integral','optimo',
        'delve','tapestry','nuanced','multifaceted','leverage',
        'streamline','paradigm','synergy','pivotal','comprehensive',
        'foster','realm','groundbreaking','transformative','noteworthy',
    }
    tokens = set(tokenizar(texto))
    return round(len(tokens & robot) / max(len(tokens)/10, 1), 3)

def calcular_patron(texto):
    ps = [p.strip() for p in texto.split('\n') if len(p.strip()) > 20]
    if len(ps) < 2:
        return 0.0
    ls = [len(p.split()) for p in ps]
    mu = np.mean(ls)
    if mu == 0:
        return 0.0
    return round(float(max(0.0, 1.0 - np.std(ls)/mu)), 3)

def calcular_longitud_media(texto):
    ors = [o.strip() for o in re.split(r'[.!?]+', texto) if len(o.strip()) > 5]
    if not ors:
        return 0.0
    return round(float(np.mean([len(o.split()) for o in ors])), 1)

def calcular_score(m):
    perp_n  = max(0.0, min(1.0, 1-(m['perplexidad']-4)/28))
    burst_n = max(0.0, min(1.0, (-m['burstiness']+0.2)/0.7))
    ttr_n   = max(0.0, min(1.0, 1-(m['ttr']-0.22)/0.40))
    var_n   = max(0.0, min(1.0, 1-m['varianza']/30))
    con_n   = min(1.0, m['conectores']/1.5)
    pat_n   = min(1.0, m['patron']*1.3)
    voc_n   = min(1.0, m['vocabulario_ia']*4)
    lmo_n   = max(0.0, min(1.0, (m['longitud_media']-8)/22))
    score = (perp_n*0.17 + burst_n*0.14 + ttr_n*0.12 + var_n*0.10 +
             con_n*0.15 + pat_n*0.09 + voc_n*0.13 + lmo_n*0.10) * 100 * 1.25
    score_ia = round(min(99.5, max(0.5, score)), 1)
    return score_ia, round(100-score_ia, 1)

def analizar_oraciones(texto, score_global):
    robot = {
        'fundamental','crucial','esencial','invaluable','exhaustivo',
        'holistico','robusto','integral','delve','tapestry','nuanced',
        'multifaceted','leverage','synergy','pivotal','comprehensive',
        'foster','realm','groundbreaking','transformative',
    }
    conectores = {
        'ademas','sin embargo','por lo tanto','en conclusion','asimismo',
        'finalmente','por otro lado','en resumen','furthermore','however',
        'therefore','moreover','thus','hence','consequently','nevertheless',
    }
    resultados = []
    for o in re.split(r'(?<=[.!?])\s+', texto.strip()):
        o = o.strip()
        if not o:
            continue
        tokens = tokenizar(o)
        if not tokens:
            resultados.append({'texto': o, 'etiqueta': 'neutral'})
            continue
        tset = set(tokens)
        score = (len(tset & robot)*18 + len(tset & conectores)*15 +
                 (1-len(tset)/len(tokens))*20 +
                 (12 if len(tokens) > 22 else 0) + score_global*0.25)
        score = min(100, max(0, score))
        if score >= (45 if score_global > 60 else 55):
            etiqueta = 'ia'
        elif score <= (28 if score_global < 40 else 35):
            etiqueta = 'humano'
        else:
            etiqueta = 'neutral'
        resultados.append({'texto': o, 'etiqueta': etiqueta})
    return resultados

def generar_imagen(score_ia, score_h):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#0a0e14')
    ax.set_facecolor('#0a0e14')
    _, _, ats = ax.pie(
        [score_ia, score_h], explode=(0.04, 0.04),
        colors=['#ff4d6d','#00e676'], autopct='%1.1f%%',
        startangle=90, wedgeprops=dict(linewidth=2, edgecolor='#0a0e14'),
        pctdistance=0.65,
    )
    for at in ats:
        at.set_color('#0a0e14')
        at.set_fontsize(13)
        at.set_fontweight('bold')
    ax.add_patch(plt.Circle((0,0), 0.40, color='#0a0e14'))
    ax.text(0, 0.08, f"{score_ia:.0f}%", ha='center', va='center',
            fontsize=22, fontweight='bold', color='#ff4d6d')
    ax.text(0, -0.18, "IA", ha='center', va='center', fontsize=11, color='#aaaaaa')
    ax.legend(
        handles=[mpatches.Patch(color='#ff4d6d', label=f'IA - {score_ia:.1f}%'),
                 mpatches.Patch(color='#00e676', label=f'Humano - {score_h:.1f}%')],
        loc='lower center', bbox_to_anchor=(0.5,-0.08), ncol=2, frameon=False,
        labelcolor=['#ff4d6d','#00e676'], prop={'size':10}
    )
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', facecolor='#0a0e14', dpi=130)
    buf.seek(0)
    plt.close(fig)
    return buf

def texto_resaltado(analisis):
    partes = []
    for item in analisis:
        if item['etiqueta'] == 'ia':
            partes.append(f"🔴 {item['texto']}")
        elif item['etiqueta'] == 'humano':
            partes.append(f"🟢 {item['texto']}")
        else:
            partes.append(f"⚪ {item['texto']}")
    return '\n'.join(partes)

def generar_reporte(score_ia, m, analisis):
    n    = len(analisis)
    n_ia = sum(1 for o in analisis if o['etiqueta'] == 'ia')
    n_h  = sum(1 for o in analisis if o['etiqueta'] == 'humano')
    if score_ia >= 75:   v = "MUY PROBABLE que sea IA"
    elif score_ia >= 55: v = "INDICIOS SIGNIFICATIVOS de IA"
    elif score_ia >= 35: v = "POSIBLE asistencia de IA"
    else:                v = "Probable AUTORIA HUMANA"
    return (
        f"REPORTE FORENSE\n\n"
        f"Veredicto: {v}\n"
        f"Score: IA={score_ia:.1f}% | Humano={100-score_ia:.1f}%\n\n"
        f"Metricas estadisticas:\n"
        f"Perplejidad: {m['perplexidad']} {'BAJA-IA' if m['perplexidad']<20 else 'OK'}\n"
        f"Burstiness: {m['burstiness']} {'UNIFORME-IA' if m['burstiness']<0 else 'OK'}\n"
        f"Riqueza lexica: {m['ttr']} {'BAJA-IA' if m['ttr']<0.45 else 'OK'}\n"
        f"Varianza oraciones: {m['varianza']} {'BAJA-IA' if m['varianza']<15 else 'OK'}\n"
        f"Conectores formales: {m['conectores']} {'ALTO-IA' if m['conectores']>1.2 else 'OK'}\n"
        f"Vocabulario IA: {m['vocabulario_ia']} {'DETECTADO' if m['vocabulario_ia']>0.3 else 'OK'}\n"
        f"Uniformidad estructura: {m['patron']} {'ALTA-IA' if m['patron']>0.6 else 'OK'}\n"
        f"Long. media oraciones: {m['longitud_media']} {'LARGA-IA' if m['longitud_media']>20 else 'OK'}\n\n"
        f"Oraciones analizadas: {n}\n"
        f"Marcadas como IA: {n_ia} ({n_ia/max(n,1)*100:.0f}%)\n"
        f"Marcadas como Humano: {n_h} ({n_h/max(n,1)*100:.0f}%)\n\n"
        f"Nota: herramienta de apoyo, no infalible."
    )

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola! Soy el Detector de IA.\n\n"
        "Envíame cualquier texto y te diré si fue escrito por una IA "
        "(Claude, ChatGPT, Gemini...) o por un humano.\n\n"
        "Te responderé con:\n"
        "- Gráfico IA vs Humano\n"
        "- Texto resaltado por oración\n"
        "- Reporte técnico\n\n"
        "Mejor con textos de más de 150 palabras."
    )

async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Ayuda:\n\n"
        "Envía cualquier texto y lo analizo.\n\n"
        "🔴 = oración probable IA\n"
        "🟢 = oración probable humano\n"
        "⚪ = indeterminado"
    )

async def analizar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    texto = update.message.text.strip()
    if len(texto) < 50:
        await update.message.reply_text("Texto muy corto. Necesito al menos 50 caracteres.")
        return

    msg = await update.message.reply_text("Analizando texto...")

    tokens = tokenizar(texto)
    m = {
        'perplexidad':    calcular_perplexidad(tokens),
        'burstiness':     calcular_burstiness(tokens),
        'ttr':            calcular_ttr(tokens),
        'varianza':       calcular_varianza(texto),
        'conectores':     calcular_conectores(texto),
        'patron':         calcular_patron(texto),
        'vocabulario_ia': calcular_vocabulario_ia(texto),
        'longitud_media': calcular_longitud_media(texto),
    }
    score_ia, score_h = calcular_score(m)
    analisis  = analizar_oraciones(texto, score_ia)
    resaltado = texto_resaltado(analisis)
    reporte   = generar_reporte(score_ia, m, analisis)
    imagen    = generar_imagen(score_ia, score_h)

    await msg.delete()

    await update.message.reply_photo(
        photo=imagen,
        caption=f"{'PROBABLE IA' if score_ia > 55 else 'PROBABLE HUMANO'} — IA: {score_ia:.1f}% | Humano: {score_h:.1f}%"
    )

    if len(resaltado) > 4000:
        resaltado = resaltado[:3900] + "\n\n(texto recortado por longitud)"
    await update.message.reply_text(f"Texto resaltado:\n\n{resaltado}")
    await update.message.reply_text(reporte)

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("ayuda", cmd_ayuda))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analizar))
    print("Bot iniciado.")
    app.run_polling()

if __name__ == "__main__":
    main()
