# songGemini_final_v9.5_rock_solid.py
# ФИНАЛЬНАЯ СТАБИЛЬНАЯ МНОГОПОТОЧНАЯ ВЕРСИЯ.
# 1. Исправлена последняя ошибка NameError с messagebox при завершении работы.
# 2. Сохранена вся рабочая логика многопоточного рендера и нового GUI.

import os, re, random, tempfile, subprocess, json, hashlib, threading, itertools, logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
from difflib import SequenceMatcher
from multiprocessing import Pool, cpu_count
import shutil

LOG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "run.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True,
)

logger = logging.getLogger(__name__)

import numpy as np, librosa
from faster_whisper import WhisperModel
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip,
    CompositeVideoClip, concatenate_videoclips, vfx, ColorClip, TextClip
)

from PIL import Image, ImageDraw, ImageFont

import customtkinter as ctk
from tkinter import filedialog, colorchooser, messagebox

# ... (Вся часть с параметрами и функциями до `run_tasks` остается прежней) ...
# ===================== ПАРАМЕТРЫ ПО УМОЛЧАНИЮ =====================
FINAL_RES = (1920, 1080)
FPS = 24
SETTINGS_FILE = "settings.json"
CUT_SEGMENTS_ON_BEATS = True
INTRO_OUTRO_UNIQUE = True
BEAT_STEP_RANGE = "2-4"
MIN_SHOT = 0.25
PAD_SHOT = 0.02
GRID_FALLBACK = 0.5
ASR_MODEL = "small"
ASR_CACHE_DIR = os.path.join("output", "align_cache"); os.makedirs(ASR_CACHE_DIR, exist_ok=True)
DEFAULT_FONT_WIN = r"C:\Windows\Fonts\arialbd.ttf"
DEFAULT_FONT_LINUX = r"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH = DEFAULT_FONT_WIN if os.name == "nt" else DEFAULT_FONT_LINUX
SUB_FONTSIZE = 60
COLOR_BASE_TUPLE = (255, 255, 255)
COLOR_ACTIVE_TUPLE = (255, 235, 59)
TEXT_MARGIN_BOTTOM = 60
TEXT_MAX_W_RATIO = 0.9
LINE_SPACING = 1.05
ZOOM_STRENGTH = 0.06
LYRICS = [
    "[Verse 1]", "Trulala Trallero tralero trallala", "Urcalero urcalero urcalalala",
    "Bombardi bombardiro coccodrillo coccodrillo", "Bobrito bandito bobrito gangsterito", "",
    "[Chorus]", "Burballoli loli loli burballoli luli loli", "Bombombini bombombini bombi gussini", "",
    "[Verse 2]", "(Brrr) Patapim pim patapim pim patapim pim", "Tung Tung Tung Tung Tung Tung Tung Tung Tung Tung Tung sahur", "",
    "[Chorus]", "Capuccino assasino assasino capuccino", "Frulli frulli frulla pinguino bla bla bla bla", "",
    "[Verse 3]", "La vaca saturno saturnito saturnito saturnita", "Ballerina capuccina capuccina ah"
]
HERO_ALIASES = {
    "Trallero":  ["trallero","tralero","trulala","trallala"], "Urcalero":  ["urcalero","urcalalala"],
    "Bombardiro":["bombardi","bombardiro","coccodrillo"], "Bobrito":   ["bobrito"],
    "Burballoli":["burballoli","burba","luli"], "Bombombini":["bombombini","gussini","bombi"],
    "Patapim":   ["patapim","pim"], "Tung":      ["tung","sahur"],
    "Capuccino": ["capuccino","cappuccino"], "Frulli":    ["frulli","frulla"],
    "Saturnito": ["saturnito","saturno","saturnita"], "Ballerina": ["ballerina","capuccina","ah"],
}
CUSTOM_HERO_MAP = {alias.lower(): hero for hero, aliases in HERO_ALIASES.items() for alias in aliases}
ANCHOR_TOKENS = set(CUSTOM_HERO_MAP.keys())

def norm_word(s:str)->str: return re.sub(r"[^\w]+","", s.lower())
def line_tokens(line:str)->List[str]:
    line=re.sub(r"\[.*?\]|\(.*?\)"," ",line); return [norm_word(t) for t in line.split() if norm_word(t)]
def find_hero_folder(line:str)->Optional[str]:
    for t in line_tokens(line):
        if t in CUSTOM_HERO_MAP: return CUSTOM_HERO_MAP[t]
    return None

@dataclass
class Word: text:str; start:float; end:float
@dataclass
class LineInterval: idx:int; start:float; end:float; hits:int

def make_vocals_only(input_audio:str)->str:
    tmp = os.path.join(tempfile.gettempdir(), "vocals_for_asr.wav")
    cmd = ["ffmpeg","-y","-i", input_audio, "-af","highpass=f=120, lowpass=f=8500, afftdn=nf=-25, dynaudnorm=f=150:g=10", "-ac","1","-ar","16000", tmp]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err_msg = result.stderr.decode("utf-8", errors="ignore").strip()
        logging.warning(
            "ffmpeg failed to create vocals-only audio (code %s): %s",
            result.returncode,
            err_msg or "no error message",
        )
        return input_audio
    return tmp if os.path.exists(tmp) else input_audio

def _hash(path:str)->str:
    st=os.stat(path); return hashlib.md5(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()
def _cache_file(audio_path:str, model_name:str)->str:
    h = hashlib.md5(f"{_hash(audio_path)}|{model_name}".encode()).hexdigest()
    return os.path.join(ASR_CACHE_DIR, h + ".json")
def transcribe_words(audio_path:str, model_name: str)->List[Word]:
    cf=_cache_file(audio_path, model_name)
    if os.path.exists(cf):
        with open(cf,"r",encoding="utf-8") as f: data=json.load(f)
        logger.info("[ASR] из кэша: %s", os.path.basename(cf))
        return [Word(d["text"], float(d["start"]), float(d["end"])) for d in data]
    clean = make_vocals_only(audio_path)
    logger.info("[ASR] распознаю (Whisper %s, GPU)...", model_name)
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    initial = " ".join([re.sub(r"\s+"," ", re.sub(r'\[.*?\]|\(.*?\)','', l)).strip() for l in LYRICS if l.strip()])
    segs, _ = model.transcribe(clean, word_timestamps=True, vad_filter=False, temperature=0.0, best_of=1, beam_size=5, condition_on_previous_text=True, compression_ratio_threshold=2.6, initial_prompt=initial)
    out=[]
    for s in segs:
        for w in (s.words or []):
            t=norm_word(w.word)
            if t: out.append(Word(t, float(w.start), float(w.end)))
    with open(cf,"w",encoding="utf-8") as f: json.dump([w.__dict__ for w in out], f, ensure_ascii=False)
    logger.info("[ASR] слов: %s", len(out))
    return out

def word_sim(a:str,b:str)->float:
    if not a or not b: return 0.0
    sm = SequenceMatcher(None,a,b).ratio()
    if len(a)>=3 and len(b)>=3 and a[:3]==b[:3]: sm += 0.15
    elif len(a)>=2 and len(b)>=2 and a[:2]==b[:2]: sm += 0.08
    return min(1.0,sm)
def align_sequences(ref:List[str], hyp:List[str])->List[Optional[int]]:
    n,m=len(ref),len(hyp); INF=1e9
    dp=[[INF]*(m+1) for _ in range(n+1)]; prv=[[None]*(m+1) for _ in range(n+1)]
    dp[0][0]=0.0; INS=0.7; DEL=0.7
    for i in range(n+1):
        for j in range(m+1):
            if i<n and j<m:
                cost=1.0-word_sim(ref[i],hyp[j])
                if ref[i] in ANCHOR_TOKENS: cost*=0.25
                if dp[i+1][j+1]>dp[i][j]+cost:
                    dp[i+1][j+1]=dp[i][j]+cost; prv[i+1][j+1]=("M",i,j)
            if i<n and dp[i+1][j]>dp[i][j]+DEL:
                dp[i+1][j]=dp[i][j]+DEL; prv[i+1][j]=("D",i,j)
            if j<m and dp[i][j+1]>dp[i][j]+INS:
                dp[i][j+1]=dp[i][j]+INS; prv[i][j+1]=("I",i,j)
    i,j=n,m; mapping=[None]*n
    while i>0 or j>0:
        op=prv[i][j]
        if op is None: break
        t,pi,pj=op
        if t=="M": mapping[pi]=pj; i,j=pi,pj
        elif t=="D": i,j=pi,pj
        else: i,j=pi,pj
    return mapping

def hero_anchor_times(hyp_words:List[Word])->dict:
    times={}
    for w in hyp_words:
        best=None; bests=0.0
        for a in ANCHOR_TOKENS:
            s=word_sim(a,w.text)
            if s>bests: bests=s; best=a
        if best and bests>=0.6 and best not in times:
            times[best]=w.start
    return times

def intervals_by_lines(audio_path:str, lyrics_lines:List[str], model_name:str)->Tuple[List['LineInterval'], List[Word]]:
    hyp_words = transcribe_words(audio_path, model_name)
    hyp_tokens = [w.text for w in hyp_words]
    ref_tokens, tok2line, useful = [], [], []
    for li, line in enumerate(lyrics_lines):
        toks = line_tokens(line)
        if not toks or line.strip().startswith("["): continue
        useful.append(li)
        for t in toks: ref_tokens.append(t); tok2line.append(li)
    intervals=[]
    if len(hyp_tokens) >= 15:
        mapping = align_sequences(ref_tokens, hyp_tokens)
        per_st, per_en, per_hits = {}, {}, {}
        for ref_i, hyp_j in enumerate(mapping):
            li=tok2line[ref_i]
            if hyp_j is None: continue
            w=hyp_words[hyp_j]
            per_st[li]=min(per_st.get(li,w.start), w.start)
            per_en[li]=max(per_en.get(li,w.end),   w.end)
            per_hits[li]=per_hits.get(li,0)+1
        for li in useful:
            if li in per_st: st,en,h=per_st[li],per_en[li],per_hits.get(li,0)
            else: st,en,h=0.0,0.12,0
            intervals.append(LineInterval(li, st, max(en, st+0.12), h))
        logger.info("[ALIGN] DP-выравнивание по словам.")
    else:
        anchors = hero_anchor_times(hyp_words); last_t=0.0
        for k,li in enumerate(useful):
            line = lyrics_lines[li]; key = next((a for a in line_tokens(line) if a in ANCHOR_TOKENS), None); st = anchors.get(key, last_t); next_st = None
            if k+1 < len(useful):
                nline = lyrics_lines[useful[k+1]]; nkey  = next((a for a in line_tokens(nline) if a in ANCHOR_TOKENS), None); next_st = anchors.get(nkey, None)
            en = next_st if next_st and next_st>st else st+1.8
            intervals.append(LineInterval(li, st, max(en, st+0.12), 0)); last_t = en
        logger.info("[ALIGN] Якорный режим.")
    intervals.sort(key=lambda x:x.start)
    for i in range(1,len(intervals)):
        if intervals[i].start < intervals[i-1].end: intervals[i].start = intervals[i-1].end
        intervals[i].end = max(intervals[i].end, intervals[i].start+0.12)
    intervals.sort(key=lambda x:x.idx); return intervals, hyp_words

def detect_beats(audio_path:str)->List[float]:
    y,sr=librosa.load(audio_path, sr=None, mono=True); tempo,beats=librosa.beat.beat_track(y=y,sr=sr,units="time"); return list(beats)
def thin_beats(times:List[float], step:int, min_spacing:float)->List[float]:
    if step <= 1: base = times[:]
    else: base = [t for i,t in enumerate(times) if i % step == 0]
    out=[];
    for t in base:
        if not out or (t - out[-1]) >= min_spacing: out.append(t)
    return out

def _load_font(size:int)->ImageFont.FreeTypeFont:
    try: return ImageFont.truetype(FONT_PATH, size=size, encoding="unic")
    except: return ImageFont.truetype(DEFAULT_FONT_LINUX, size=size, encoding="unic")
def _text_w(font,s):
    try: return font.getlength(s)
    except: return font.getbbox(s)[2]-font.getbbox(s)[0]
def _line_h(font): asc,desc=font.getmetrics(); return int((asc+desc)*LINE_SPACING)
def _wrap(tokens,font,maxw):
    lines,cur,w=[],[],0.0; sp=_text_w(font," ")
    for t in tokens:
        tw=_text_w(font,t); add=tw if not cur else sp+tw
        if cur and w+add>maxw: lines.append(cur); cur,w=[t],tw
        else: cur.append(t); w+=add
    if cur: lines.append(cur); return lines
def _layout(lines,font,W,H):
    lh=_line_h(font); y0=H-TEXT_MARGIN_BOTTOM - lh*len(lines); sp=_text_w(font," "); coords=[]
    for i,line in enumerate(lines):
        widths=[_text_w(font,w) for w in line]; lw = sum(widths)+(len(line)-1)*sp if len(line)>1 else sum(widths)
        x0=int((W-lw)/2); y=int(y0+i*lh); x=x0
        for j,w in enumerate(line): coords.append((w,int(x),y)); x+=int(widths[j]+(sp if j<len(line)-1 else 0))
    return coords
def _render(base_text, active_idx, font_size, color_base, color_active):
    W,H=FINAL_RES; img=Image.new("RGBA",(W,H),(0,0,0,0)); dr=ImageDraw.Draw(img); f=_load_font(font_size)
    toks=[t for t in re.split(r"\s+", base_text.strip()) if t]
    if not toks: return np.array(img)
    lines=_wrap(toks,f,int(W*TEXT_MAX_W_RATIO)); coords=_layout(lines,f,W,H)
    for i,(w,x,y) in enumerate(coords): dr.text((x,y), w, font=f, fill=(*color_base, 255))
    if active_idx is not None and 0<=active_idx<len(coords): w,x,y=coords[active_idx]; dr.text((x,y), w, font=f, fill=(*color_active, 255))
    return np.array(img, dtype=np.uint8)
def _rgba_clip(arr, dur):
    rgb=arr[...,:3]; a=(arr[...,3].astype("float32")/255.0); c=ImageClip(rgb).set_duration(dur); m=ImageClip(a, ismask=True).set_duration(dur); return c.set_mask(m)
def _match_idx(base_text, pieces):
    base=[t.lower() for t in re.split(r"\s+", base_text.strip()) if t]; out=[]; k=0
    for p in pieces:
        idx=-1
        for j in range(k, len(base)):
            if base[j].startswith(p[:max(1,min(3,len(p)))]): idx=j; k=j+1; break
        if idx==-1: idx=min(len(base)//2, len(base)-1)
        out.append(idx)
    return out
def make_karaoke_overlays(line_text, words, st, en, sub_fontsize, color_base, color_active):
    dur=max(0.06,en-st); base_text=re.sub(r"\s+"," ", re.sub(r"\[.*?\]|\(.*?\)","", line_text)).strip()
    if not base_text: return []
    render = lambda active_idx: _render(base_text, active_idx, sub_fontsize, color_base, color_active)
    if not words: fr=render(None); return [_rgba_clip(fr,dur)]
    pcs=[];
    for w in words:
        ws=max(0,w.start-st); we=min(dur,w.end-st)
        if we>ws: pcs.append((w.text,ws,we))
    idxs=_match_idx(base_text, [p[0] for p in pcs]); out=[]
    if pcs and pcs[0][1]>0.05: fr=render(None); out.append(_rgba_clip(fr, pcs[0][1]))
    for (txt,ws,we),di in zip(pcs,idxs): fr=render(di); out.append(_rgba_clip(fr, we-ws).set_start(ws))
    if pcs:
        last=pcs[-1][2]
        if dur-last>0.05: fr=render(None); out.append(_rgba_clip(fr, dur-last).set_start(last))
    return out

def cover_resize(clip: VideoFileClip, size: tuple) -> VideoFileClip:
    W,H = size; scale = max(W/clip.w, H/clip.h); clip = clip.fx(vfx.resize, scale); return clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=W, height=H)
def list_video_files(folder):
    if not os.path.isdir(folder): return []
    return [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith((".mp4",".mov",".mkv",".webm"))]
def _find_style(hero_dir, style):
    if not os.path.isdir(hero_dir): return None
    for d in os.listdir(hero_dir):
        p=os.path.join(hero_dir,d)
        if os.path.isdir(p) and d.lower()==style.lower(): return p
    return None
def collect_pool(heroes_root, style=None, exclude:Optional[Set[str]]=None):
    files=[]
    for hero in HERO_ALIASES.keys():
        hero_dir=os.path.join(heroes_root, hero)
        sdir=_find_style(hero_dir, style) if style else None
        files+=list_video_files(sdir) if sdir else list_video_files(hero_dir)
    files=[f for f in files if os.path.isfile(f)]
    if exclude: files=[f for f in files if f not in exclude]
    return files
def pick_clip_by_duration(heroes_root, hero_folder, style, target_dur):
    hero_dir=os.path.join(heroes_root, hero_folder)
    style_dir=_find_style(hero_dir, style) if style else None
    pool=list_video_files(style_dir) if style_dir else []
    if not pool: pool=list_video_files(hero_dir)
    if not pool: return None
    scored=[]
    for fp in pool:
        try: v=VideoFileClip(fp); d=abs(v.duration-target_dur); v.close(); scored.append((d,fp))
        except: pass
    if not scored: return random.choice(pool)
    scored.sort(key=lambda x:x[0]); return scored[0][1]
def subclip_sized(fp, need_dur):
    v=VideoFileClip(fp).without_audio()
    if v.duration >= need_dur + 0.05:
        start=random.uniform(0, max(0, v.duration-need_dur-0.01)); out=cover_resize(v.subclip(start, start+need_dur), FINAL_RES); return out
    loops=int(need_dur//max(0.2, v.duration))+1; parts=[cover_resize(v.copy(), FINAL_RES) for _ in range(loops)]; out=concatenate_videoclips(parts).set_duration(need_dur); return out.without_audio()

def get_beat_step(beat_step_range_str):
    try:
        parts = [int(p.strip()) for p in beat_step_range_str.split('-')]
        if len(parts) == 1: return parts[0]
        if len(parts) == 2: return random.randint(min(parts), max(parts))
    except: pass
    return 2

def make_cut_points(st:float, en:float, beats:List[float], beat_step_range: str, min_shot: float)->List[float]:
    current_beat_step = get_beat_step(beat_step_range)
    local=[t for t in beats if st<=t<=en]; local=thin_beats(local, current_beat_step, min_shot*1.2)
    if not local or local[0] > st + min_shot: local=[st]+local
    if local[-1] < en - min_shot: local=local+[en]
    return local
def shots_for_hero_on_beats(heroes_root, hero, style, local_beats, st, en, beat_step_range, min_shot):
    dur=max(0.06, en-st); points = make_cut_points(st, en, local_beats, beat_step_range, min_shot)
    hero_dir=os.path.join(heroes_root, hero); style_dir=_find_style(hero_dir, style) if style else None
    pool=list_video_files(style_dir) if style_dir else list_video_files(hero_dir)
    if not pool: anyp=collect_pool(heroes_root, style); return [subclip_sized(random.choice(anyp), dur).set_start(0)], set()
    random.shuffle(pool); iter_pool=itertools.cycle(pool); used=set(); clips=[]
    for a,b in zip(points[:-1], points[1:]):
        seg_d = max(min_shot, b-a - PAD_SHOT)
        if seg_d <= 0.05: continue
        fp = next(iter_pool); used.add(fp); shot = subclip_sized(fp, seg_d).set_start(max(0,a-st)).fadein(0.03).fadeout(0.03)
        shot = shot.fx(vfx.resize, lambda t, _d=shot.duration:1.0+ZOOM_STRENGTH*(t/_d)).without_audio(); clips.append(shot)
    if not clips: fp = next(iter_pool); used.add(fp); clips=[subclip_sized(fp, dur)]
    return clips, used

def make_segment(line_text, words, st, en, beats, params):
    hero=find_hero_folder(line_text)
    if not hero: return None, set()
    hero_shots, used_files = shots_for_hero_on_beats(params['heroes_root'], hero, params['style'], beats, st, en, params['beat_step_range'], params['min_shot'])
    layers=[]
    layers += [ImageClip(np.full((FINAL_RES[1], FINAL_RES[0],3),255,dtype=np.uint8)).set_start(max(0,b-st-0.03)).set_duration(0.06).set_opacity(0.12) for b in [t for t in beats if st<=t<=en]]
    layers += make_karaoke_overlays(line_text, words, st, en, params['sub_fontsize'], params['color_base'], params['color_active'])
    base = CompositeVideoClip(hero_shots + layers, size=FINAL_RES).set_duration(en-st); return base, used_files

def beats_or_grid(beats, total_dur, grid_fallback):
    bt=[0.0]+[b for b in beats if 0.0<b<total_dur]+[total_dur]
    if len(bt)<4: bt=[i*grid_fallback for i in range(int(total_dur/grid_fallback)+1)];
    if bt[-1]<total_dur: bt.append(total_dur)
    return bt

def compute_gaps(intervals: List[LineInterval], total_dur: float) -> List[Tuple[float, float]]:
    if not intervals: return [(0, total_dur)]
    segs = sorted([(it.start, it.end) for it in intervals if find_hero_folder(LYRICS[it.idx])], key=lambda x: x[0])
    if not segs: return [(0, total_dur)]
    gaps, last_end_time = [], 0.0
    if segs[0][0] > 0.05: gaps.append((0.0, segs[0][0]))
    for start, end in segs:
        if start > last_end_time + 0.05: gaps.append((last_end_time, start))
        last_end_time = max(last_end_time, end)
    if total_dur > last_end_time + 0.05: gaps.append((last_end_time, total_dur))
    return gaps

def montage_for_range(start_time, end_time, params, exclude:Optional[Set[str]]=None):
    dur = end_time - start_time
    if dur <= 0.05: return None
    logger.info(
        "[BUILD] Создание нарезки для промежутка [%0.2f - %0.2f]...",
        start_time,
        end_time,
    )
    local_beats = [b for b in params['beats'] if start_time <= b <= end_time]
    current_beat_step = get_beat_step(params['beat_step_range'])
    cut_points = [start_time] + thin_beats(local_beats, current_beat_step * 2, params['min_shot'] * 1.5) + [end_time]
    cut_points = sorted(list(set(cut_points)))
    pool = collect_pool(params['heroes_root'], params['style'], exclude=exclude)
    if not pool:
        logger.warning("[WARN] Нет клипов для создания нарезки. Промежуток будет черным.")
        return ColorClip(size=FINAL_RES, color=(0,0,0)).set_duration(dur)
    random.shuffle(pool); cyc = itertools.cycle(pool); shots = []
    for a, b in zip(cut_points[:-1], cut_points[1:]):
        shot_dur = b - a
        if shot_dur < params['min_shot']: continue
        fp = next(cyc)
        try:
            shot = subclip_sized(fp, shot_dur).set_start(a - start_time)
            shots.append(shot)
        except Exception as e:
            logger.warning("[WARN] Ошибка при создании шотa для нарезки из '%s': %s", fp, e)
    if not shots: return ColorClip(size=FINAL_RES, color=(0,0,0)).set_duration(dur)
    return CompositeVideoClip(shots, size=FINAL_RES).set_duration(dur).without_audio()

def render_segment_task(task_args):
    """Функция-воркер, которая рендерит один сегмент."""
    task, params, temp_dir = task_args
    task_id, task_type = task['id'], task['type']
    logger.info("  [Поток %s] Начинаю задачу %s: %s...", os.getpid(), task_id, task_type)

    global LYRICS, HERO_ALIASES, CUSTOM_HERO_MAP, ANCHOR_TOKENS
    LYRICS = params['lyrics_data']['lyrics']
    HERO_ALIASES = params['lyrics_data']['aliases']
    CUSTOM_HERO_MAP = {alias.lower(): hero for hero, aliases in HERO_ALIASES.items() for alias in aliases}
    ANCHOR_TOKENS = set(CUSTOM_HERO_MAP.keys())
    
    out_path = os.path.join(temp_dir, f"segment_{task['start']:08.3f}.mp4")
    clip = None

    try:
        if task_type == 'hero':
            it = next(i for i in params['intervals'] if i.idx == task['idx'])
            line = task['line']
            wslice = [w for w in params['hyp_words'] if it.start-0.05 <= w.start <= it.end+0.05]
            clip, _ = make_segment(line, wslice, it.start, it.end, params['beats'], params)
        
        elif task_type == 'gap':
            clip = montage_for_range(task['start'], task['end'], params, exclude=task.get('exclude'))

        if clip:
            clip.write_videofile(out_path, fps=FPS, codec="libx264", audio=False, preset="ultrafast", logger=None)
            clip.close()
            logger.info("  [Поток %s] Задача %s (%s) завершена.", os.getpid(), task_id, task_type)
            return out_path
    except Exception:
        logger.exception("Ошибка при выполнении задачи %s (%s)", task_id, task_type)
        try:
            err_clip = TextClip(f"ERROR Task {task['id']}", fontsize=50, color='red', size=FINAL_RES).set_duration(task['end'] - task['start'])
            err_clip.write_videofile(out_path, fps=FPS, codec="libx264", audio=False, preset="ultrafast", logger=None)
            err_clip.close()
            return out_path
        except: return None
            
    return None

def _probe_nb_frames(video_path: Path) -> Optional[int]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1", str(video_path)
    ]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        value = result.stdout.strip()
        return int(value) if value else None
    except Exception as exc:
        logger.warning("Не удалось получить количество кадров для %s: %s", video_path, exc)
        return None

def build_video(params, app_instance):
    audio_path = params['audio_path']; out_path = params['out_path']
    app_instance.after(0, app_instance.update_status, "Анализ аудио (ASR)...")
    intervals, hyp_words = intervals_by_lines(audio_path, LYRICS, params['asr_model'])
    app_instance.after(0, app_instance.update_status, "Анализ ритма...")
    beats = detect_beats(audio_path)
    app_instance.after(0, app_instance.update_status, "Подготовка задач для рендера...")
    with AudioFileClip(str(audio_path)) as audio:
        audio_duration = audio.duration
    last_segment_end_time = max((it.end for it in intervals if find_hero_folder(LYRICS[it.idx])), default=0)
    total_dur = max(last_segment_end_time, audio_duration)
    
    tasks = []
    hero_intervals = [it for it in intervals if find_hero_folder(LYRICS[it.idx])]
    for it in hero_intervals:
         tasks.append({'id': len(tasks), 'type': 'hero', 'start': it.start, 'end': it.end, 'idx': it.idx, 'line': LYRICS[it.idx]})

    gaps = compute_gaps(intervals, total_dur)
    if params.get('fill_gaps_with_montage', True):
        for start, end in gaps:
            exclude_files = set()
            if params.get('intro_outro_unique', True) and hero_intervals:
                # В будущем можно добавить логику исключения файлов
                pass 
            tasks.append({'id': len(tasks), 'type': 'gap', 'start': start, 'end': end, 'exclude': exclude_files})
    
    tasks.sort(key=lambda x: x['start'])
    
    temp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp_montage")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    logger.info("Временные файлы будут создаваться в: %s", temp_dir)

    params['intervals'] = intervals; params['hyp_words'] = hyp_words; params['beats'] = beats
    params['lyrics_data'] = {'lyrics': LYRICS, 'aliases': HERO_ALIASES}
    task_args = [(task, params, temp_dir) for task in tasks]
    
    app_instance.after(0, app_instance.update_status, f"Рендер {len(tasks)} сегментов в {params['num_threads']} потоков...")
    
    temp_files_map = {}
    with Pool(processes=params['num_threads']) as pool:
        for i, result_path in enumerate(pool.imap_unordered(render_segment_task, task_args)):
            if result_path:
                start_time = float(os.path.basename(result_path).split('_')[1].replace('.mp4',''))
                temp_files_map[start_time] = result_path
            progress = int((i + 1) / len(task_args) * 100)
            app_instance.after(0, app_instance.update_progress, progress)

    app_instance.after(0, app_instance.update_status, "Финальная склейка...")
    
    sorted_files = [Path(temp_files_map[key]) for key in sorted(temp_files_map.keys())]
    if not sorted_files:
        raise RuntimeError("Ни один сегмент не был успешно отрендерен.")

    filtered_files = []
    filtered_count = 0
    for path in sorted_files:
        nb_frames = _probe_nb_frames(path)
        if nb_frames is not None and nb_frames < 3:
            filtered_count += 1
            logger.info("Пропускаю сегмент %s из-за малого количества кадров (%s)", path.name, nb_frames)
            continue
        filtered_files.append(path)

    logger.info("Сегментов для склейки: %s (отфильтровано: %s)", len(sorted_files), filtered_count)

    if not filtered_files:
        raise RuntimeError("Все сегменты отклонены из-за малого количества кадров.")

    list_file_path = Path(temp_dir) / "filelist.txt"
    with list_file_path.open("w", encoding='utf-8') as f:
        for path in filtered_files:
            f.write(f"file '{path.as_posix()}'\n")

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-safe', '0', '-fflags', '+genpts',
        '-f', 'concat', '-i', str(list_file_path), '-i', str(audio_path),
        '-r', '30', '-c:v', 'libx264', '-preset', 'slow', '-crf', '21',
        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
        '-shortest', '-movflags', '+faststart', str(out_path)
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning("Не удалось удалить временную папку %s: %s", temp_dir, e)

class App(ctk.CTk):
    def __init__(self):
        super().__init__(); self.title("Auto Montage Pro (v9.3)"); self.geometry("850x680"); self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(4, weight=1)
        self.audio_path_var = ctk.StringVar(); self.heroes_root = ctk.StringVar(); self.output_dir = ctk.StringVar()
        self.style = ctk.StringVar(); self.asr_model = ctk.StringVar(value="small")
        self.video_count = ctk.StringVar(value="1"); self.batch_all_styles = ctk.BooleanVar(value=False)
        self.beat_step_range = ctk.StringVar(value=BEAT_STEP_RANGE); self.min_shot = ctk.StringVar(value=str(MIN_SHOT))
        self.sub_fontsize = ctk.StringVar(value=str(SUB_FONTSIZE))
        self.color_base_hex = '#%02x%02x%02x' % COLOR_BASE_TUPLE; self.color_active_hex = '#%02x%02x%02x' % COLOR_ACTIVE_TUPLE
        self.num_threads = ctk.StringVar(value=str(max(1, cpu_count() - 2)))
        self.create_widgets(); self.load_settings()

    def create_widgets(self):
        path_frame = ctk.CTkFrame(self); path_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew"); path_frame.grid_columnconfigure(1, weight=1)
        row = 0
        def add_path_row(label, var):
            nonlocal row; ctk.CTkLabel(path_frame, text=label).grid(row=row, column=0, padx=10, pady=5, sticky="e"); entry = ctk.CTkEntry(path_frame, textvariable=var); entry.grid(row=row, column=1, padx=10, pady=5, sticky="ew"); btn = ctk.CTkButton(path_frame, text="...", width=40, command=lambda v=var: self.pick_path(v)); btn.grid(row=row, column=2, padx=10, pady=5); row += 1
        
        self.path_mode = ctk.StringVar(value="file")
        ctk.CTkLabel(path_frame, text="Источник аудио:").grid(row=row, column=0, padx=10, pady=5, sticky="e")
        path_radio_frame = ctk.CTkFrame(path_frame, fg_color="transparent")
        path_radio_frame.grid(row=row, column=1, padx=0, pady=0, sticky="w")
        ctk.CTkRadioButton(path_radio_frame, text="Файл", variable=self.path_mode, value="file").pack(side="left", padx=5)
        ctk.CTkRadioButton(path_radio_frame, text="Папка", variable=self.path_mode, value="folder").pack(side="left", padx=5)
        row += 1
        
        add_path_row("Аудио файл/папка:", self.audio_path_var)
        add_path_row("Папка героев:", self.heroes_root)
        add_path_row("Папка для вывода:", self.output_dir)
        
        settings_frame = ctk.CTkFrame(self); settings_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew"); settings_frame.grid_columnconfigure(1, weight=1); settings_frame.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(settings_frame, text="Стиль:").grid(row=0, column=0, padx=10, pady=5, sticky="e"); self.style_menu = ctk.CTkOptionMenu(settings_frame, variable=self.style, values=["-"]); self.style_menu.grid(row=0, column=1, padx=10, pady=5, sticky="w"); self.batch_checkbox = ctk.CTkCheckBox(settings_frame, text="Пакетный режим по всем стилям", variable=self.batch_all_styles, command=self.toggle_style_menu); self.batch_checkbox.grid(row=0, column=2, columnspan=2, padx=20, pady=5, sticky="w")
        ctk.CTkLabel(settings_frame, text="Beat Step (диапазон):").grid(row=1, column=0, padx=10, pady=5, sticky="e"); ctk.CTkEntry(settings_frame, textvariable=self.beat_step_range).grid(row=1, column=1, padx=10, pady=5, sticky="w"); ctk.CTkLabel(settings_frame, text="Min Shot (сек):").grid(row=1, column=2, padx=10, pady=5, sticky="e"); ctk.CTkEntry(settings_frame, textvariable=self.min_shot).grid(row=1, column=3, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(settings_frame, text="Модель ASR:").grid(row=2, column=0, padx=10, pady=5, sticky="e"); ctk.CTkOptionMenu(settings_frame, variable=self.asr_model, values=["tiny", "base", "small", "medium", "large-v3"]).grid(row=2, column=1, padx=10, pady=5, sticky="w"); ctk.CTkLabel(settings_frame, text="Сколько видео делать:").grid(row=2, column=2, padx=10, pady=5, sticky="e"); ctk.CTkEntry(settings_frame, textvariable=self.video_count).grid(row=2, column=3, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(settings_frame, text="Количество потоков:").grid(row=3, column=0, padx=10, pady=5, sticky="e"); ctk.CTkEntry(settings_frame, textvariable=self.num_threads).grid(row=3, column=1, padx=10, pady=5, sticky="w")
        
        sub_frame = ctk.CTkFrame(self); sub_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew"); sub_frame.grid_columnconfigure(4, weight=1)
        ctk.CTkLabel(sub_frame, text="Настройки субтитров:").grid(row=0, column=0, padx=10, pady=5); ctk.CTkLabel(sub_frame, text="Размер:").grid(row=0, column=1, padx=(10,0), pady=5); ctk.CTkEntry(sub_frame, textvariable=self.sub_fontsize, width=50).grid(row=0, column=2, padx=(0,10), pady=5)
        self.base_color_btn = ctk.CTkButton(sub_frame, text="Цвет текста", command=self.pick_base_color, fg_color=self.color_base_hex); self.base_color_btn.grid(row=0, column=3, padx=5, pady=5)
        self.active_color_btn = ctk.CTkButton(sub_frame, text="Цвет подсветки", command=self.pick_active_color, fg_color=self.color_active_hex); self.active_color_btn.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ctk.CTkButton(sub_frame, text="Предпросмотр", command=self.preview_subs).grid(row=0, column=5, padx=10, pady=5)
        
        self.status_label = ctk.CTkLabel(self, text="Готов к работе", font=ctk.CTkFont(size=14)); self.status_label.grid(row=3, column=0, padx=10, pady=(10,0), sticky="sw")
        self.progress_bar = ctk.CTkProgressBar(self); self.progress_bar.set(0); self.progress_bar.grid(row=4, column=0, padx=10, pady=(0,10), sticky="ew")

        button_frame = ctk.CTkFrame(self, fg_color="transparent"); button_frame.grid(row=5, column=0, padx=10, pady=10, sticky="ew"); button_frame.grid_columnconfigure(0, weight=1); button_frame.grid_columnconfigure(1, weight=1); button_frame.grid_columnconfigure(2, weight=1)
        self.save_btn = ctk.CTkButton(button_frame, text="Сохранить настройки", command=self.save_settings); self.save_btn.grid(row=0, column=0, padx=5, pady=5)
        self.load_btn = ctk.CTkButton(button_frame, text="Загрузить настройки", command=self.load_settings); self.load_btn.grid(row=0, column=1, padx=5, pady=5)
        self.run_button = ctk.CTkButton(button_frame, text="СМОНТИРОВАТЬ", command=self.run_full, height=40, font=ctk.CTkFont(size=16, weight="bold")); self.run_button.grid(row=0, column=2, padx=5, pady=5, sticky="e")

    def pick_path(self, var):
        is_audio = var == self.audio_path_var
        mode = self.path_mode.get() if is_audio else 'folder'
        is_dir = mode == 'folder'
        if var != self.audio_path_var: is_dir = True
        path = filedialog.askdirectory() if is_dir else filedialog.askopenfilename()
        if path: var.set(path)
        if var == self.heroes_root: self.update_styles_menu()
    
    def update_styles_menu(self):
        path = self.heroes_root.get()
        if not os.path.isdir(path): self.style_menu.configure(values=["-"]); self.style.set("-"); return
        styles = set()
        for hero_folder in os.listdir(path):
            hero_path = os.path.join(path, hero_folder)
            if os.path.isdir(hero_path):
                for sub_folder in os.listdir(hero_path):
                    if os.path.isdir(os.path.join(hero_path, sub_folder)): styles.add(sub_folder)
        style_list = sorted(list(styles)) if styles else ["-"]
        self.style_menu.configure(values=style_list)
        if self.style.get() not in style_list: self.style.set(style_list[0])
    def toggle_style_menu(self):
        if self.batch_all_styles.get(): self.style_menu.configure(state="disabled"); self.video_count.set("1")
        else: self.style_menu.configure(state="normal")
    def pick_base_color(self):
        color_code = colorchooser.askcolor(title="Выберите основной цвет текста");
        if color_code and color_code[1]: self.color_base_hex = color_code[1]; self.base_color_btn.configure(fg_color=self.color_base_hex)
    def pick_active_color(self):
        color_code = colorchooser.askcolor(title="Выберите цвет подсветки")
        if color_code and color_code[1]: self.color_active_hex = color_code[1]; self.active_color_btn.configure(fg_color=self.color_active_hex)
    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#'); return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    def preview_subs(self):
        try: fontsize = int(self.sub_fontsize.get()); color_base = self.hex_to_rgb(self.color_base_hex); color_active = self.hex_to_rgb(self.color_active_hex)
        except: messagebox.showerror("Ошибка", "Неверный формат размера шрифта."); return
        preview_text = "Пример текста для субтитров"
        preview_img_array = _render(preview_text, 1, fontsize, color_base, color_active)
        img = Image.fromarray(preview_img_array); top = ctk.CTkToplevel(self); top.title("Предпросмотр субтитров")
        bg_img = Image.new('RGB', (img.width, img.height), 'black'); bg_img.paste(img, (0, 0), img)
        ctk_img = ctk.CTkImage(light_image=bg_img, dark_image=bg_img, size=(img.width // 2, img.height // 2))
        label = ctk.CTkLabel(top, image=ctk_img, text=""); label.pack(padx=10, pady=10)
    def save_settings(self):
        settings = { "audio_path_var": self.audio_path_var.get(), "path_mode": self.path_mode.get(), "heroes_root": self.heroes_root.get(), "output_dir": self.output_dir.get(), "style": self.style.get(), "asr_model": self.asr_model.get(), "video_count": self.video_count.get(), "beat_step_range": self.beat_step_range.get(), "min_shot": self.min_shot.get(), "sub_fontsize": self.sub_fontsize.get(), "color_base_hex": self.color_base_hex, "color_active_hex": self.color_active_hex, "num_threads": self.num_threads.get() }
        try:
            with open(SETTINGS_FILE, "w", encoding='utf-8') as f: json.dump(settings, f, indent=4, ensure_ascii=False)
            self.update_status("Настройки сохранены.")
        except Exception as e: self.update_status(f"Ошибка сохранения: {e}")
    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding='utf-8') as f: settings = json.load(f)
                self.audio_path_var.set(settings.get("audio_path_var", "")); self.path_mode.set(settings.get("path_mode", "file")); self.heroes_root.set(settings.get("heroes_root", "")); self.output_dir.set(settings.get("output_dir", ""))
                self.style.set(settings.get("style", "-")); self.asr_model.set(settings.get("asr_model", "small")); self.video_count.set(settings.get("video_count", "1"))
                self.beat_step_range.set(settings.get("beat_step_range", BEAT_STEP_RANGE)); self.min_shot.set(settings.get("min_shot", str(MIN_SHOT)))
                self.sub_fontsize.set(settings.get("sub_fontsize", str(SUB_FONTSIZE))); self.color_base_hex = settings.get("color_base_hex", '#FFFFFF')
                self.color_active_hex = settings.get("color_active_hex", '#FFEB3B'); self.base_color_btn.configure(fg_color=self.color_base_hex)
                self.active_color_btn.configure(fg_color=self.color_active_hex); self.update_status("Настройки загружены.")
                self.num_threads.set(settings.get("num_threads", str(max(1, cpu_count() - 2))))
            except Exception as e: self.update_status(f"Ошибка загрузки: {e}")
        self.update_styles_menu()
    def _toggle_ui(self, enabled:bool):
        state = "normal" if enabled else "disabled"
        for child in self.winfo_children():
            queue = [child]
            while queue:
                widget = queue.pop(0)
                try:
                    if 'state' in widget.configure(): widget.configure(state=state)
                except: continue
                queue.extend(widget.winfo_children())
    def update_progress(self, value):
        self.progress_bar.set(value / 100); self.update_idletasks()
    def update_status(self, text):
        self.status_label.configure(text=text); self.update_idletasks()
    def run_full(self):
        self.save_settings()
        try:
            params = { "heroes_root": self.heroes_root.get(), "output_dir": self.output_dir.get(), "asr_model": self.asr_model.get(), "beat_step_range": self.beat_step_range.get(), "min_shot": float(self.min_shot.get()), "sub_fontsize": int(self.sub_fontsize.get()), "color_base": self.hex_to_rgb(self.color_base_hex), "color_active": self.hex_to_rgb(self.color_active_hex), "num_threads": int(self.num_threads.get()), "fill_gaps_with_montage": True, "intro_outro_unique": True}
            video_count = int(self.video_count.get())
            if not all([self.audio_path_var.get(), params["heroes_root"], params["output_dir"]]):
                messagebox.showerror("Ошибка", "Пожалуйста, заполните все пути."); return
        except Exception as e: messagebox.showerror("Ошибка в параметрах", f"Проверьте правильность введенных данных: {e}"); return
        audio_files = []
        source_path = self.audio_path_var.get()
        if self.path_mode.get() == "folder":
            if os.path.isdir(source_path):
                for f in os.listdir(source_path):
                    if f.lower().endswith((".mp3", ".wav", ".m4a")): audio_files.append(os.path.join(source_path, f))
            if not audio_files: messagebox.showerror("Ошибка", "В указанной папке нет аудиофайлов."); return
        else:
            if os.path.isfile(source_path): audio_files.append(source_path)
            else: messagebox.showerror("Ошибка", "Указанный аудиофайл не найден."); return
        tasks = []
        styles_to_process = self.style_menu.cget("values") if self.batch_all_styles.get() else [self.style.get()]
        if styles_to_process == ["-"]: messagebox.showerror("Ошибка", "Не найдены/не выбраны стили."); return
        for audio_file in audio_files:
            for style in styles_to_process:
                for i in range(video_count): tasks.append({'audio_path': audio_file, 'style': style})
        threading.Thread(target=self.run_tasks, args=(tasks, params), daemon=True).start()

    def run_tasks(self, tasks, params):
        self.after(0, self._toggle_ui, False)
        total_tasks = len(tasks)
        logger.info("\n%s\nНАЧИНАЮ РАБОТУ: %s видео в очереди.\n%s\n", "=" * 50, total_tasks, "=" * 50)

        for i, task in enumerate(tasks):
            logger.info(
                "\n--- Видео %s/%s (Файл: %s, Стиль: %s) ---\n",
                i + 1,
                total_tasks,
                os.path.basename(task['audio_path']),
                task['style'],
            )
            self.after(0, self.update_status, f"Видео {i+1}/{total_tasks}: {os.path.basename(task['audio_path'])} ({task['style']})")
            self.after(0, self.update_progress, 0)
            try:
                task_params = params.copy()
                task_params['audio_path'] = task['audio_path']
                task_params['style'] = task['style']
                audio_basename = os.path.splitext(os.path.basename(task['audio_path']))[0]
                output_folder = os.path.join(params['output_dir'], task['style'])
                os.makedirs(output_folder, exist_ok=True)
                file_index = 1
                while True:
                    file_suffix = f"_{file_index}"
                    output_filename = f"{audio_basename}_{task['style']}{file_suffix}.mp4"
                    out_path = os.path.join(output_folder, output_filename)
                    if not os.path.exists(out_path): break
                    file_index += 1
                task_params['out_path'] = out_path
                build_video(task_params, self)
            except Exception as e:
                logger.exception(
                    "Произошла ошибка при создании видео для стиля '%s'",
                    task['style'],
                )
                self.after(0, lambda e=e, s=task['style']: messagebox.showerror("Критическая ошибка", f"Произошла ошибка при создании видео для стиля '{s}':\n\n{e}"))
                break

        logger.info("\n%s\nВСЕ ЗАДАЧИ ВЫПОЛНЕНЫ.\n%s\n", "=" * 50, "=" * 50)
        self.after(0, self.update_status, "Готово!")
        self.after(0, self._toggle_ui, True)
        self.after(0, lambda: messagebox.showinfo("Завершено", "Все задачи по созданию видео выполнены."))

def main():
    if __name__ == "__main__":
        from multiprocessing import freeze_support
        freeze_support()
        
        app = App()
        app.mainloop()

main()