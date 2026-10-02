import json, os, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import requests

SPORT='americanfootball_nfl'
BOOK='draftkings'
PRICE_CEILING=-200
BASE='https://api.the-odds-api.com/v4'
MARKETS={
'player_pass_attempts':'pass_attempts',
'player_pass_completions':'pass_completions',
'player_pass_yds':'passing_yards',
'player_receptions':'receptions',
'player_reception_yds':'receiving_yards',
'player_rush_attempts':'rush_attempts',
'player_rush_yds':'rushing_yards',
'player_rush_reception_yds':'rec_plus_rush_yards'
}
TEAM_MAP={
'Arizona Cardinals':'ARI','Atlanta Falcons':'ATL','Baltimore Ravens':'BAL','Buffalo Bills':'BUF',
'Carolina Panthers':'CAR','Chicago Bears':'CHI','Cincinnati Bengals':'CIN','Cleveland Browns':'CLE',
'Dallas Cowboys':'DAL','Denver Broncos':'DEN','Detroit Lions':'DET','Green Bay Packers':'GB',
'Houston Texans':'HOU','Indianapolis Colts':'IND','Jacksonville Jaguars':'JAX','Kansas City Chiefs':'KC',
'Las Vegas Raiders':'LV','Los Angeles Chargers':'LAC','Los Angeles Rams':'LA','Miami Dolphins':'MIA',
'Minnesota Vikings':'MIN','New England Patriots':'NE','New Orleans Saints':'NO','New York Giants':'NYG',
'New York Jets':'NYJ','Philadelphia Eagles':'PHI','Pittsburgh Steelers':'PIT','San Francisco 49ers':'SF',
'Seattle Seahawks':'SEA','Tampa Bay Buccaneers':'TB','Tennessee Titans':'TEN','Washington Commanders':'WAS'
}

def norm(s):
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode().lower()
    s=re.sub(r'\b(jr|sr|ii|iii|iv)\b','',s)
    s=re.sub(r'[^a-z0-9]+',' ',s)
    return ' '.join(s.split())

def strict_json_value(v):
    if isinstance(v, dict):
        return {k: strict_json_value(x) for k,x in v.items()}
    if isinstance(v, list):
        return [strict_json_value(x) for x in v]
    if isinstance(v, (float, np.floating)) and not np.isfinite(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v

def main():
    key=os.environ.get('THE_ODDS_API_KEY','').strip()
    if not key:
        raise RuntimeError('THE_ODDS_API_KEY is required')
    out=Path('phoenix_generative_props_v6_1')
    props_path=sorted(out.glob('PHOENIX_NFL_GENERATIVE_PROPS_V6_1_*.csv'))[-1]
    ladder_path=sorted(out.glob('PHOENIX_NFL_GENERATIVE_FAIR_LADDER_V6_1_*.csv'))[-1]
    props=pd.read_csv(props_path)
    ladder=pd.read_csv(ladder_path)
    season=2026
    m=re.search(r'_(\d{4})_W(\d+)\.csv$',props_path.name)
    week=int(m.group(2)) if m else 0
    props['name_norm']=props.player_name.map(norm)
    ladder['name_norm']=ladder.player_name.map(norm)

    ev=requests.get(f'{BASE}/sports/{SPORT}/events',params={'apiKey':key},timeout=60)
    ev.raise_for_status()
    events=ev.json()
    event_by_pair={}
    for e in events:
        a=TEAM_MAP.get(e.get('away_team'))
        h=TEAM_MAP.get(e.get('home_team'))
        if a and h:
            event_by_pair[(a,h)]=e

    candidates=[]
    for gid,g in props.groupby('game_id'):
        parts=str(gid).split('_')
        away,home=(parts[-2],parts[-1]) if len(parts)>=2 else (None,None)
        e=event_by_pair.get((away,home))
        if not e:
            continue
        rr=requests.get(
            f"{BASE}/sports/{SPORT}/events/{e['id']}/odds",
            params={
                'apiKey':key,'regions':'us','bookmakers':BOOK,
                'markets':','.join(MARKETS),
                'oddsFormat':'american','dateFormat':'iso'
            },
            timeout=60
        )
        if rr.status_code!=200:
            print('Odds warning',gid,rr.status_code,rr.text[:160])
            continue
        ej=rr.json()
        for bm in ej.get('bookmakers',[]):
            if bm.get('key')!=BOOK:
                continue
            for mk in bm.get('markets',[]):
                stat=MARKETS.get(mk.get('key'))
                if not stat:
                    continue
                for o in mk.get('outcomes',[]):
                    side=str(o.get('name','')).lower()
                    desc=o.get('description','')
                    point=o.get('point')
                    price=o.get('price')
                    if side not in {'over','under'} or point is None or price is None:
                        continue
                    price=float(price)
                    line=float(point)
                    if price<PRICE_CEILING:
                        continue
                    nn=norm(desc)
                    meta=g[(g.name_norm==nn)&(g.stat==stat)]
                    if meta.empty:
                        continue
                    mr=meta.iloc[0]
                    if str(mr.availability_flag)!='CLEAR' or float(mr.availability_probability)<0.95:
                        continue
                    if str(mr.distribution_flag)!='OK':
                        continue
                    if pd.notna(mr.starter_probability) and float(mr.starter_probability)<0.90:
                        continue
                    z=ladder[
                        (ladder.game_id.astype(str)==str(gid))
                        &(ladder.name_norm==nn)
                        &(ladder.stat==stat)
                        &np.isclose(ladder.line.astype(float),line)
                    ]
                    if z.empty:
                        continue
                    lr=z.iloc[0]
                    p=float(lr.p_over if side=='over' else lr.p_under)
                    fair=float(lr.fair_over_odds if side=='over' else lr.fair_under_odds)
                    if p<=0.5:
                        continue
                    candidates.append({
                        'game_id':str(gid),
                        'team':str(mr.team),
                        'opponent':str(mr.opponent),
                        'player_name':str(mr.player_name),
                        'position':str(mr.position),
                        'stat':stat,
                        'side':side,
                        'line':line,
                        'bookmaker':BOOK,
                        'book_odds':price,
                        'phoenix_probability':p,
                        'phoenix_fair_odds':fair,
                        'availability_probability':float(mr.availability_probability),
                        'starter_probability':None if pd.isna(mr.starter_probability) else float(mr.starter_probability),
                        'history_reliability':float(mr.history_reliability),
                        'uncertainty_multiplier':float(mr.uncertainty_multiplier),
                        'distribution_flag':str(mr.distribution_flag),
                        'book_price_rule':'HARD_CEILING_ONLY_-200',
                        'fair_price_role':'INFORMATIONAL_ONLY_NO_COMPARISON',
                        'ranking_inputs':['phoenix_probability','availability','distribution_qa','history_reliability']
                    })
    df=pd.DataFrame(candidates)
    if df.empty:
        raise RuntimeError('No execution-eligible DraftKings props matched Phoenix ladders')
    df=df.sort_values(
        ['phoenix_probability','history_reliability','uncertainty_multiplier'],
        ascending=[False,False,True]
    ).drop_duplicates(['game_id','player_name','stat','side','line'])
    full_dest=out/f'PHOENIX_NFL_PROPS_FULL_MATCHED_BOARD_{season}_W{week}.csv'
    df.to_csv(full_dest,index=False)
    print(full_dest)
    top=df.head(10).copy()
    packet={
        'engine':'PHOENIX_NFL_PROPS_V6_2_0',
        'season':season,'week':week,'bookmaker':BOOK,
        'purpose':'ADVERSARIAL_ANALYSIS_TRANSFER',
        'model_first':True,'ev_used':False,
        'sportsbook_role':'POST_MODEL_PROPOSITION_MATCHING_AND_-200_CEILING_ONLY',
        'selection_rule':'Top 10 matched DraftKings propositions by Phoenix probability after CLEAR availability, OK distribution QA, starter>=90% when applicable, and -200 price ceiling. No EV/fair-price comparison.',
        'candidates':top.to_dict(orient='records')
    }
    dest=out/f'PHOENIX_NFL_PROPS_TOP10_ADVERSARIAL_{season}_W{week}.json'
    dest.write_text(json.dumps(strict_json_value(packet),indent=2,allow_nan=False),encoding='utf-8')
    print(dest)
    print(top[['player_name','stat','side','line','book_odds','phoenix_probability']].to_string(index=False))

if __name__=='__main__':
    main()
