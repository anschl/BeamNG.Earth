"""Sparse OSM route shields when no regional/weekly shield inventory is installed.

Full-way stations and local deterministic thinning make placement independent of
streaming tiles. This never guesses a route number or a mile reference.

The ways (read_routes' rows: an I / US ref on a motorway ... tertiary) come from the weekly OSM tag shards
(osmtagshards.py) when their build carries them (routes_ok: the route_* columns, tools/build_osm_tag_shards.py), else
from the Layercake file itself: its 9 MB footer and a matching row group's id / bbox / highway / ref columns were 46 MB
at Lone Pine CA, 24 MB at Hartford CT and 35 MB at San Francisco on a cold cache, per new US place (2026-11-12). The
shards' tiles under a cell and its halo are a dozen or so, the roads' own among them, each read once per build.
"""
from collections import OrderedDict
from concurrent.futures import CancelledError
import math
import threading
import numpy as np
import mapcells
import osmtags
import routeindex
import signstates

CELL_DEG = .1
CANDIDATE_M = 700.
SEPARATION_M = 2400.       # local minima: typically about three miles apart
HALO_M = SEPARATION_M + 200.
# The ways read_routes takes (tools/build_osm_tag_shards.py keeps a copy: its tests hold them equal).
ROUTE_WHERE = ("highway IN ('motorway','trunk','primary','secondary','tertiary') "
               "AND regexp_matches(ref, '(?i)(^|;)\\s*(I|US)[ -]?[0-9]{1,3}\\s*(;|$)')")
WANTED = ('oneway', 'bridge', 'tunnel', 'access', 'junction', 'direction')
# In the tag shards: these tags of exactly those ways ("route_highway" set marks one; oneway is osmtags' osm_oneway, the
# raw tag), their line in the tile's "geom", and the bbox of a way osmtags leaves out (no tag the roads use) in
# route_bx0... with its own bbox NaN (osmtags.cell_rows never matches it: an older server's tags stay as they were).
ROUTE_TEXT = tuple(f'route_{c}' for c in ('highway', 'ref') + WANTED if c != 'oneway')
ROUTE_BOX = ('route_bx0', 'route_by0', 'route_bx1', 'route_by1')
EDGE_DEG = 1e-4     # tiles within this of an area's edge are read too (a bbox's float32 against the area's doubles)


def read_routes(tags, remote, bounds, kind='demand'):
    """Read route refs separately so ordinary road-tag decoding stays unchanged."""
    rd, files = tags._reader, {osmtags.PATH: remote}
    con, proto, errors = rd._connect(files, kind)
    try:
        uri = f"'{proto}://{osmtags.PATH}'"
        groups = tags._groups(con, uri, remote, kind)
        candidates = tags._candidates(groups, bounds)
        if not candidates or not all('ref' in groups[g]['cols'] for g in candidates):
            return []
        cols = ('id', 'bbox', 'highway', 'ref')
        rd._fetch(files, {osmtags.PATH: [r for g in candidates for c in cols
                                       for r in groups[g]['cols'].get(c, ())]}, kind)
        x0, y0, x1, y1 = map(float, bounds)
        predicate = f"bbox.xmin <= {x1} AND bbox.xmax >= {x0} AND bbox.ymin <= {y1} AND bbox.ymax >= {y0} AND {ROUTE_WHERE}"
        rows = con.execute(f'SELECT file_row_number FROM read_parquet({uri}, file_row_number=true) WHERE {predicate}').fetchall()
        if not rows:
            return []
        hit = [g for g in candidates if any(groups[g]['first'] <= r[0] < groups[g]['first']+groups[g]['n'] for r in rows)]
        wanted = WANTED
        available = set.intersection(*(set(groups[g]['cols']) for g in hit))
        rd._fetch(files, {osmtags.PATH: [r for g in hit for c in (*wanted, 'geometry')
                                       for r in groups[g]['cols'].get(c, ())]}, kind)
        try:
            con.execute('SET enable_geoparquet_conversion = false')
        except Exception:
            pass
        columns = ', '.join(f'"{c}"' if c in available else f'NULL AS "{c}"' for c in wanted)
        numbers = ','.join(str(int(r[0])) for r in rows)
        result = []
        for wid, highway, ref, *rest in con.execute(
                f'SELECT id, highway, ref, {columns}, geometry FROM read_parquet({uri}, file_row_number=true) '
                f'WHERE file_row_number IN ({numbers})').fetchall():
            line = osmtags._linestring(bytes(rest[-1])) if rest[-1] is not None else None
            if line is not None and len(line) >= 2:
                attrs = dict(highway=highway, ref=ref, **{c: v for c, v in zip(wanted, rest) if v is not None})
                result.append((int(wid), line, attrs))
        return result
    except Exception:
        for err in errors:
            if isinstance(err, osmtags.rangeread.Changed):
                raise err
        raise
    finally:
        con.close()


def routes_ok(schema):
    """Whether a tag shard build's schema (the manifest's) carries read_routes' ways: the route_* columns."""
    try:
        num = {str(c): int(w) for c, w in schema['numeric']}
        text = {str(c) for c in schema['text']}
    except (KeyError, TypeError, ValueError):
        return False
    return all(num.get(c) == 4 for c in ROUTE_BOX) and set(ROUTE_TEXT) <= text


def _text(t, c, i):
    o = t[c + '_off']
    return t[c][o[i]:o[i + 1]].tobytes().decode('utf-8') if t[c + '_ok'][i] else None


def tile_routes(t, bounds):
    """{file row: read_routes' (way id, line, tags)} of a decoded tag shard tile (osmtagshards.TagShards.tile; None: no
    rows) for the ways whose bbox touches bounds, in float32 as osmtags.cell_rows. DuckDB's test in read_routes is
    float32 too, but casts its DECIMAL literal its own way (one step off in ~1/300 bounds): a way whose box meets the
    area's edge within that step lies HALO_M from the cell, beyond SEPARATION_M of all its shields, and thins none."""
    if t is None or not len(t['id']):
        return {}
    i = np.flatnonzero(t['route_highway_ok'])
    x0, y0, x1, y1 = (np.float32(v) for v in bounds)
    b = [np.where(np.isnan(t[c][i]), t[f'route_{c}'][i], t[c][i]).astype(np.float32) for c in ('bx0', 'by0', 'bx1', 'by1')]
    out = {}
    for k in i[(b[0] <= x1) & (b[2] >= x0) & (b[1] <= y1) & (b[3] >= y0)].tolist():
        f = int(t['frn'][k])
        line = t['geom'].get(f)
        if line is None or len(line) < 2:
            continue
        attrs = dict(highway=_text(t, 'route_highway', k), ref=_text(t, 'route_ref', k))
        for c in WANTED:
            v = _text(t, 'osm_oneway' if c == 'oneway' else f'route_{c}', k)
            if v is not None:
                attrs[c] = v
        out[f] = (int(t['id'][k]), np.asarray(line, np.float64), attrs)
    return out


def shard_routes(shards, m, bounds, kind='demand'):
    """read_routes' answer from the tag shards (osmtagshards.TagShards, its manifest m: routes_ok): every tile the
    bounds touch (EDGE_DEG more round them), a way in several of them once."""
    x0, y0, x1, y1 = bounds
    ways = {}
    for key in mapcells.cells_for_bbox(y0 - EDGE_DEG, x0 - EDGE_DEG, y1 + EDGE_DEG, x1 + EDGE_DEG, m['zoom']):
        ways.update(tile_routes(shards.tile(key, kind), bounds))
    return list(ways.values())


def candidates(ways):
    result = []
    for wid, line, tags in sorted(ways, key=lambda w: w[0]):
        routes = routeindex.route_refs(tags, ())
        if not routes or tags.get('highway') not in routeindex.ROAD_CLASSES or not routeindex.ground(tags):
            continue
        path = np.asarray(line, float)
        scale = np.array([111320.*math.cos(math.radians(float(path[0, 1]))), 111320.])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0)*scale, axis=1))]
        length = float(arc[-1])
        if length < 10.:
            continue
        at = lambda s: np.array([np.interp(s, arc, path[:, j]) for j in (0, 1)])
        one = routeindex.exitsigns._direction(tags)
        if not one and tags.get('highway') == 'motorway' and tags.get('oneway') not in ('no', 'false', '0'):
            one = 1
        stations = np.linspace(0., length, max(1, int(math.ceil(length/CANDIDATE_M)))+1)
        for k, station in enumerate((stations[:-1]+stations[1:])/2):
            point = at(station)
            a, b = max(0., station-60.), min(length, station+60.)
            local = np.vstack([at(a), path[(arc > a) & (arc < b)], at(b)])
            for direction in ((one,) if one else (1, -1)):
                approach = local if direction > 0 else local[::-1]
                axis = (approach[-1]-approach[0])*scale
                axis /= max(np.linalg.norm(axis), 1e-9)
                panels = [dict(kind='interstate' if r[0]=='I' else 'us', number=r[1],
                               direction=routeindex.posted_direction(tags, (), direction, r, axis)) for r in routes[:3]]
                extra = dict(type='shield', routes=panels, main_id=wid, point=point.tolist(),
                             approach=approach.tolist(), axis=axis.tolist())
                obj = routeindex.record(('fallback-shield-v1', wid, k, direction), point, extra)
                obj.pop('ways'); obj['roadside'] = extra
                result.append(obj)
    return result


def keys(obj):
    return {(r['kind'], r['number'], r.get('direction', '')) for r in obj.get('roadside', {}).get('routes', ())}


def distance(a, b):
    scale = math.cos(math.radians((a['lat']+b['lat'])/2))
    return 111320.*math.hypot((a['lon']-b['lon'])*scale, a['lat']-b['lat'])


def thin(objects, bounds):
    """Keep local priority minima; all competitors within HALO_M must be supplied."""
    x0, y0, x1, y1 = bounds
    out = []
    keyed = [(o, keys(o)) for o in objects]
    for obj, routes in keyed:
        if not (x0 <= obj['lon'] < x1 and y0 <= obj['lat'] < y1):
            continue
        if any(other['id'] < obj['id'] and routes & other_routes and distance(obj, other) < SEPARATION_M
               for other, other_routes in keyed):
            continue
        out.append(obj)
    if out:
        states = signstates.states(np.array([o['lat'] for o in out]), np.array([o['lon'] for o in out]))
        out = [o for o, state in zip(out, states) if state]
    return out


class Source:
    def __init__(self, raw_dir, fetcher, shards=None):
        """shards: a callable giving the process's osmtagshards.TagShards or None (mapdata.MapData._osm_shards: the
        roads' own, its tiles and down state shared); none: the Layercake file only."""
        self.tags = osmtags.OsmTags(raw_dir, fetcher)
        self.shards = shards
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def _keep(self, key, make):
        if key not in self.cache:
            self.cache[key] = make()
        self.cache.move_to_end(key)
        result = self.cache[key]
        while len(self.cache) > 32:
            self.cache.popitem(last=False)
        return result

    def _from_shards(self, x, y, box, area):
        """The cell's shields from the tag shards, or None when they cannot answer (off, a build without the route
        columns: the one published before them, a read that fails: the shards marked down as for the roads' tags)."""
        import osmtagshards
        shards = self.shards() if self.shards is not None else None
        got = shards.manifest() if shards is not None else None
        if got is None or not routes_ok(got[1].get('schema')):
            return None
        try:
            return self._keep(('shards', got[0], x, y), lambda: thin(candidates(shard_routes(shards, got[1], area)), box))
        except osmtagshards.NotInUse:
            return None
        except CancelledError:
            raise
        except Exception as e:
            shards.down(e)
            return None

    def _cell(self, x, y):
        box = (x*CELL_DEG, y*CELL_DEG, (x+1)*CELL_DEG, (y+1)*CELL_DEG)
        dy = HALO_M/110000.
        dx = dy/max(.25, math.cos(math.radians(max(abs(box[1]), abs(box[3]))+dy)))
        area = (box[0]-dx, box[1]-dy, box[2]+dx, box[3]+dy)
        got = self._from_shards(x, y, box, area)
        if got is not None:
            return got
        for attempt in (0, 1):
            version, remote = self.tags.version()
            try:
                return self._keep((version, x, y), lambda: thin(candidates(read_routes(self.tags, remote, area)), box))
            except osmtags.rangeread.Changed:
                self.tags._forget()
                if attempt:
                    raise

    def query(self, lon0, lat0, lon1, lat1, existing=()):
        # Explicit US refs only. Avoid an optional download outside US latitudes.
        if lon1 < -180 or lon0 > -60 or lat1 < 18 or lat0 > 72:
            return []
        result = []
        with self.lock:
            for x in range(math.floor(lon0/CELL_DEG), math.floor(lon1/CELL_DEG)+1):
                for y in range(math.floor(lat0/CELL_DEG), math.floor(lat1/CELL_DEG)+1):
                    result.extend(o for o in self._cell(x, y) if lon0 <= o['lon'] <= lon1 and lat0 <= o['lat'] <= lat1)
        shields = [(o, keys(o)) for o in existing if o.get('roadside', {}).get('type') == 'shield']
        return [o for o in result if not any(keys(o) & k and distance(o, old) < SEPARATION_M for old, k in shields)]
