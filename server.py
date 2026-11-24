from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3

app = FastAPI(title="API Vidéo-Verbalisation (prototype)")
DB = "passages.db"


def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS passages(
        id INTEGER PRIMARY KEY AUTOINCREMENT, date_heure TEXT, track_id INTEGER,
        type_vehicule TEXT, vitesse_kmh REAL, vitesse_moy REAL, infraction INTEGER,
        plaque TEXT, image_name TEXT)""")
    if "limite_kmh" not in [r[1] for r in con.execute("PRAGMA table_info(passages)")]:
        con.execute("ALTER TABLE passages ADD COLUMN limite_kmh REAL DEFAULT 50")
    return con


class Passage(BaseModel):
    date_heure: str
    track_id: int
    type_vehicule: str
    vitesse_kmh: float
    vitesse_moy: float
    infraction: bool = False
    limite_kmh: float = 50
    plaque: str = ""
    image_name: str = ""


@app.post("/api/passages")
def ajouter(p: Passage):
    with db() as con:
        con.execute(
            "INSERT INTO passages(date_heure,track_id,type_vehicule,vitesse_kmh,vitesse_moy,"
            "infraction,limite_kmh,plaque,image_name) VALUES(?,?,?,?,?,?,?,?,?)",
            (p.date_heure, p.track_id, p.type_vehicule, p.vitesse_kmh, p.vitesse_moy,
             int(p.infraction), p.limite_kmh, p.plaque, p.image_name))
    return {"ok": True}


@app.get("/api/passages")
def lister():
    return [dict(r) for r in db().execute("SELECT * FROM passages ORDER BY id DESC")]


@app.delete("/api/passages")
def vider():
    with db() as con:
        con.execute("DELETE FROM passages")
    return {"ok": True}
