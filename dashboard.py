import os
import time
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageFilter

API = "http://127.0.0.1:8008/api/passages"
COULEURS = {"Voiture": "#00D4FF", "Moto": "#FF9F1C", "Bus": "#7CFF6B", "Camion": "#FF3D81"}
ROUGE, BLEU = "#FF3D81", "#00D4FF"
px.defaults.template = "plotly_dark"

st.set_page_config(page_title="Vidéo-Verbalisation Bénin", page_icon="🚓", layout="wide")
st.title("🚓 Vidéo-Verbalisation – Tableau de bord")
st.caption("PROTOTYPE de démonstration · vitesses estimées par IA, non homologuées")

# ---------------------------------------------------------------- Barre latérale
sb = st.sidebar
auto = sb.toggle("🔄 Actualisation auto (5 s)", True)


def attendre():
    if auto:
        time.sleep(5)
        st.rerun()


try:
    df = pd.DataFrame(requests.get(API, timeout=2).json())
except Exception:
    st.error("Serveur injoignable : lance d'abord `uvicorn server:app --port 8008`")
    attendre()
    st.stop()

if df.empty:
    st.info("En attente de données... lance `python3 app_engine.py`")
    attendre()
    st.stop()

df["date_heure"] = pd.to_datetime(df.date_heure)
df["heure"] = df.date_heure.dt.hour

sb.header("🎛️ Filtres")
limite_data = int(df.limite_kmh.fillna(50).iloc[0])      # limite utilisée par le moteur
seuil = sb.slider("Limite de vitesse (km/h)", 30, 150, limite_data)
tous_types = sorted(df.type_vehicule.unique())
types_sel = sb.multiselect("Type de véhicule", tous_types, default=tous_types)
vmin, vmax = int(df.vitesse_kmh.min()), int(df.vitesse_kmh.max()) + 1
plage = sb.slider("Plage de vitesse (km/h)", vmin, vmax, (vmin, vmax))
h1, h2 = sb.slider("Tranche horaire", 0, 23, (0, 23))
seulement = sb.checkbox("Infractions seulement")
flou = sb.checkbox("Flouter les preuves (mode public)", False)
if sb.button("🗑️ Réinitialiser les données"):
    requests.delete(API)
    st.rerun()

df["infraction"] = df.vitesse_kmh > seuil
f = df[df.type_vehicule.isin(types_sel) & df.vitesse_kmh.between(*plage) & df.heure.between(h1, h2)]
if seulement:
    f = f[f.infraction]
if f.empty:
    st.warning("Aucun résultat avec ces filtres.")
    attendre()
    st.stop()
inf = f[f.infraction]

# ---------------------------------------------------------------- KPI
k = st.columns(6)
k[0].metric("🚗 Passages", len(f), border=True)
k[1].metric("🚨 Infractions", len(inf), border=True)
k[2].metric("📊 Taux d'infraction", f"{len(inf) / len(f) * 100:.0f} %", border=True)
k[3].metric("⚡ Vitesse moyenne", f"{f.vitesse_kmh.mean():.0f} km/h", border=True)
k[4].metric("🔥 Plus grand excès", f"{inf.vitesse_kmh.max():.0f} km/h" if len(inf) else "–", border=True)
k[5].metric("🟢 Plus petit excès", f"{inf.vitesse_kmh.min():.0f} km/h" if len(inf) else "–", border=True)

t1, t2, t3, t4 = st.tabs(["📈 Vue d'ensemble", "🔬 Analyse", "📸 Preuves", "📋 Données"])

# ---------------------------------------------------------------- Vue d'ensemble
with t1:
    c1, c2 = st.columns([2, 1])

    fig = px.histogram(f, x="vitesse_kmh", color="type_vehicule", nbins=30,
                       color_discrete_map=COULEURS, title="Répartition des vitesses")
    fig.add_vline(x=seuil, line_dash="dash", line_color="red", annotation_text="Seuil")
    c1.plotly_chart(fig)

    c2.plotly_chart(px.pie(f, names="type_vehicule", hole=0.55, color="type_vehicule",
                           color_discrete_map=COULEURS, title="Types de véhicules"))

    g = f.sort_values("date_heure").copy()
    g["moyenne_mobile"] = g.vitesse_kmh.rolling(5, min_periods=1).mean()
    fig = px.scatter(g, x="date_heure", y="vitesse_kmh", color="type_vehicule",
                     color_discrete_map=COULEURS, title="Vitesse dans le temps (courbe = moyenne mobile)")
    fig.add_trace(go.Scatter(x=g.date_heure, y=g.moyenne_mobile, name="Moyenne mobile",
                             line=dict(color="white", width=3)))
    fig.add_hline(y=seuil, line_dash="dash", line_color="red")
    st.plotly_chart(fig)

    h = f.groupby(["heure", "infraction"]).size().reset_index(name="n")
    h["statut"] = h.infraction.map({True: "Infraction", False: "Conforme"})
    st.plotly_chart(px.bar(h, x="heure", y="n", color="statut", title="Passages par heure",
                           color_discrete_map={"Infraction": ROUGE, "Conforme": BLEU}))

# ---------------------------------------------------------------- Analyse
with t2:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.box(f, x="type_vehicule", y="vitesse_kmh", color="type_vehicule",
                           color_discrete_map=COULEURS, title="Vitesse par type de véhicule"))
    taux = f.groupby("type_vehicule").infraction.mean().mul(100).reset_index(name="taux")
    c2.plotly_chart(px.bar(taux, x="type_vehicule", y="taux", color="type_vehicule",
                           color_discrete_map=COULEURS, title="Taux d'infraction par type (%)"))

    c3, c4 = st.columns(2)
    c3.subheader("🔥 Top 5 des plus rapides")
    c3.dataframe(f.nlargest(5, "vitesse_kmh")[["date_heure", "type_vehicule", "vitesse_kmh"]])
    c4.subheader("🐢 Top 5 des plus lents")
    c4.dataframe(f.nsmallest(5, "vitesse_kmh")[["date_heure", "type_vehicule", "vitesse_kmh"]])

# ---------------------------------------------------------------- Preuves
with t3:
    preuves = f[(f.image_name != "") & f.infraction].sort_values("vitesse_kmh", ascending=False).head(12)
    if preuves.empty:
        st.info("Aucune preuve photo pour ces filtres.")
    cols = st.columns(3)
    for i, (_, r) in enumerate(preuves.iterrows()):
        chemin = os.path.join("captures", r.image_name)
        if os.path.exists(chemin):
            img = Image.open(chemin).convert("RGB")
            if flou:
                img = img.filter(ImageFilter.GaussianBlur(12))
            legende = f"#{r.track_id} · {r.type_vehicule} · {r.vitesse_kmh:.0f} km/h"
            if not flou:
                legende += f" · {r.plaque}"
            cols[i % 3].image(img, caption=legende)

# ---------------------------------------------------------------- Données
with t4:
    tableau = f.drop(columns=["image_name", "heure"] + (["plaque"] if flou else []))
    st.dataframe(tableau, column_config={
        "vitesse_kmh": st.column_config.ProgressColumn("Vitesse", min_value=0, max_value=150, format="%.0f km/h"),
        "infraction": st.column_config.CheckboxColumn("Infraction"),
    })
    st.download_button("⬇️ Télécharger en CSV", tableau.to_csv(index=False), "passages.csv")

attendre()
