"""Conservative page-level geometry for regularly spaced vertical rubbings.

Recognition labels are optional evidence: visible glyph positions must not be
merged merely because CTC missed a token. Non-rubbing/irregular pages keep the
ordinary row path. All coordinates are in the caller's OCR proxy.
"""
import copy
import cv2
import numpy as np


def _evidence(gray, pitch):
    size = max(3, int(round(pitch*1.5)) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (size,size)))
    strength = cv2.subtract(gray, background)
    threshold, _ = cv2.threshold(strength, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    strong = strength >= max(20, threshold)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strong.astype(np.uint8), 8)
    accepted = np.flatnonzero(stats[:,cv2.CC_STAT_AREA] >= max(3,int(pitch*pitch*.001)))
    accepted = accepted[accepted != 0]
    # A pale page edge can appear inside an expanded dark column. Unlike a
    # glyph stroke it continues across multiple character positions.
    accepted=np.array([k for k in accepted if not (
        stats[k,cv2.CC_STAT_HEIGHT] > 1.6*pitch
        and stats[k,cv2.CC_STAT_WIDTH] < .25*pitch
        and (stats[k,cv2.CC_STAT_LEFT] < 3 or
             stats[k,cv2.CC_STAT_LEFT]+stats[k,cv2.CC_STAT_WIDTH] > gray.shape[1]-3)
    )],dtype=int)
    strong = np.isin(labels, accepted)
    radius = max(1, round(pitch*.025))
    near = cv2.dilate(strong.astype(np.uint8), np.ones((2*radius+1,2*radius+1),np.uint8))
    weak = (strength >= max(10,threshold*.6)) & (near != 0)
    return weak, strong


def _period(profile, low, high):
    # Fourier phase estimates the recurring ink centre, not a CTC token edge.
    y = np.arange(len(profile),dtype=float)
    values = profile.astype(float)
    values = np.maximum(0, values-np.percentile(values,20))
    best = None
    for pitch in np.arange(low, high+.1, .25):
        wave = np.exp(2j*np.pi*y/pitch)
        z = np.sum(values*wave)
        score = abs(z)/max(float(values.sum()),1)
        if best is None or score > best[0]:
            best = (score,float(pitch),float(np.angle(z)%(2*np.pi)*pitch/(2*np.pi)))
    return best


def refine_regular_rubbing_page(image_bgr, rows):
    """Return handled row indices; unsupported pages are left untouched."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    candidates = []
    anchor_gaps = []
    for i,row in enumerate(rows):
        result = row.get('result',{})
        if result.get('char_box_strategy') != 'stroke-ownership':
            continue
        box = np.asarray(row['bbox'],float)
        left,top = box.min(axis=0); right,bottom = box.max(axis=0)
        if bottom-top < 3*(right-left) or right-left < 5:
            continue
        x0,x1 = max(0,int(left)),min(gray.shape[1],int(np.ceil(right)))
        y0,y1 = max(0,int(top)),min(gray.shape[0],int(np.ceil(bottom)))
        if x1<=x0 or y1<=y0 or np.median(gray[y0:y1,x0:x1]) >= 120:
            continue
        chars = result.get('char_list',[])
        ys = sorted(float(np.asarray(c['line'])[:,1].mean()) for c in chars if 'line' in c)
        anchor_gaps.extend(g for g in np.diff(ys) if .35*(right-left)<g<1.5*(right-left))
        candidates.append((i,(left+right)/2,top,bottom,right-left))
    if len(candidates)<8 or len(anchor_gaps)<40:
        return set()
    pitch_hint = float(np.median(anchor_gaps))
    groups = []
    for candidate in sorted(candidates,key=lambda c:c[1]):
        if groups and abs(candidate[1]-np.mean([c[1] for c in groups[-1]])) < pitch_hint*.45:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    if len(groups)<8:
        return set()
    # Require several sustained columns; isolated captions do not define a grid.
    if sum(max(c[3] for c in g)-min(c[2] for c in g)>8*pitch_hint for g in groups)<6:
        return set()
    # Merge short detector fragments into an established body column before
    # fitting its rhythm. Otherwise the fragment can duplicate a recovered slot.
    assigned = {c[0] for g in groups for c in g}
    for i,row in enumerate(rows):
        if i in assigned or row.get('result',{}).get('char_box_strategy') != 'stroke-ownership':
            continue
        box=np.asarray(row['bbox'],float)
        left,top=box.min(axis=0);right,bottom=box.max(axis=0)
        if bottom-top < .5*pitch_hint or right-left > 1.6*pitch_hint:
            continue
        group=min(groups,key=lambda g:abs((left+right)/2-np.median([c[1] for c in g])))
        if abs((left+right)/2-np.median([c[1] for c in group])) < .4*pitch_hint:
            group.append((i,(left+right)/2,top,bottom,right-left))
    columns = []
    for group in groups:
        cx = float(np.median([c[1] for c in group]))
        top = max(0,int(min(c[2] for c in group)))
        bottom = min(gray.shape[0],int(np.ceil(max(c[3] for c in group))))
        half = min(pitch_hint*.65, float(np.median([c[4] for c in group]))*.6)
        left,right = max(0,round(cx-half)),min(gray.shape[1],round(cx+half))
        weak,strong = _evidence(gray[top:bottom,left:right],pitch_hint)
        profile = strong.sum(axis=1).astype(float)
        fit = _period(profile,pitch_hint*.87,pitch_hint*1.13)
        columns.append(dict(group=group,cx=cx,left=left,top=top,weak=weak,strong=strong,profile=profile,fit=fit))
    reliable=[c for c in columns if c['fit'][0]>=.12]
    if len(reliable)<max(6,len(groups)*.7):
        return set()
    pitch = float(np.median([c['fit'][1] for c in reliable]))
    phases=[_period(c['profile'],pitch,pitch)[2]+c['top'] for c in reliable]
    shared_phase=float(np.angle(np.mean(np.exp(2j*np.pi*np.asarray(phases)/pitch)))%(2*np.pi)*pitch/(2*np.pi))
    proposals = []
    for col in columns:
        score,local_pitch,phase = _period(col['profile'],pitch*.98,pitch*1.02)
        if score < .12:
            local_pitch=pitch
            phase=(shared_phase-col['top'])%pitch
        height,width = col['weak'].shape
        centers = np.arange(phase-local_pitch, height+local_pitch, local_pitch)
        centers = centers[(centers > -.1*local_pitch)&(centers < height+.1*local_pitch)]
        # Boundaries follow nearby low-ink valleys. One damaged gap cannot
        # force the entire column back to unrestricted component grouping.
        smooth = np.convolve(col['profile'],np.ones(3)/3,mode='same')
        cuts = []
        for middle in np.r_[centers-local_pitch/2,centers[-1]+local_pitch/2]:
            low,high = max(0,round(middle-.32*local_pitch)),min(height-1,round(middle+.32*local_pitch))
            if low>=high:
                cuts.append(int(np.clip(round(middle),0,height)))
            else:
                positions = np.arange(low,high+1)
                costs = smooth[positions]/max(1,np.percentile(smooth,90)) + .12*((positions-middle)/(.32*local_pitch))**2
                cuts.append(int(positions[np.argmin(costs)]))
        for center,y0,y1 in zip(centers,cuts,cuts[1:]):
            if y1-y0 < .35*local_pitch:
                continue
            strong = col['strong'][y0:y1]
            count,labels,stats,_ = cv2.connectedComponentsWithStats(strong.astype(np.uint8),8)
            minimum = max(3,local_pitch*local_pitch*.002)
            accepted = np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=minimum)
            accepted = accepted[accepted!=0]
            seed = np.isin(labels,accepted)
            if seed.sum() < local_pitch*local_pitch*.012:
                continue
            radius = max(1,round(local_pitch*.025))
            near = cv2.dilate(seed.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8))
            pixels = col['weak'][y0:y1] & (near!=0)
            yy,xx = np.nonzero(pixels)
            if not len(xx):
                continue
            proposals.append(dict(col=col,center=center+col['top'],labels=labels,stats=stats,accepted=accepted,y0=y0,y1=y1,
                bounds=[col['left']+xx.min(),col['top']+y0+yy.min(),
                        col['left']+xx.max()+1,col['top']+y0+yy.max()+1]))
    if len(proposals)<40:
        return set()
    # Use the dominant body size to constrain background outliers, preserving
    # narrow/short glyphs such as 一. Never enlarge every glyph to a square.
    sizes=np.array([[p['bounds'][2]-p['bounds'][0],p['bounds'][3]-p['bounds'][1]] for p in proposals])
    normal=sizes[(sizes[:,0]>.45*pitch)&(sizes[:,1]>.45*pitch)]
    typical=np.median(normal,axis=0) if len(normal) else np.median(sizes,axis=0)
    typical_component=float(np.median([max(p['stats'][k,cv2.CC_STAT_AREA] for k in p['accepted']) for p in proposals]))
    typical_ink=float(np.median([sum(p['stats'][k,cv2.CC_STAT_AREA] for k in p['accepted']) for p in proposals]))
    handled=set()
    for col in columns:
        slots=[p for p in proposals if p['col'] is col]
        if not slots:
            continue
        slots.sort(key=lambda p:p['center'])
        source=[]
        for index,*_ in col['group']:
            for c in rows[index]['result'].get('char_list',[]):
                if 'line' in c:
                    source.append((float(np.asarray(c['line'])[:,1].mean()),copy.deepcopy(c)))
        # One-to-one assignment prevents a duplicate CTC position generating
        # overlapping copies. Unrecognized but visible slots remain unlabeled.
        assignment={}; used=set()
        pairs=sorted((abs(y-p['center']),i,j) for i,(y,c) in enumerate(source) for j,p in enumerate(slots))
        for distance,i,j in pairs:
            if distance > pitch*.6: break
            if i not in used and j not in assignment:
                used.add(i);assignment[j]=source[i][1]
        for index,*_ in col['group']:
            rows[index]['result']['char_list']=[]
            handled.add(index)
        for j,p in enumerate(slots):
            # A missing OCR token is not permission to turn every textured
            # grid cell into a character. Require coherent body-scale strokes.
            inferred=j not in assignment or not assignment[j].get('char','').strip()
            if inferred:
                largest=max(p['stats'][k,cv2.CC_STAT_AREA] for k in p['accepted'])
                ink=sum(p['stats'][k,cv2.CC_STAT_AREA] for k in p['accepted'])
                body_parts=[k for k in p['accepted']
                            if p['stats'][k,cv2.CC_STAT_WIDTH]>=typical[0]*.25
                            and p['stats'][k,cv2.CC_STAT_HEIGHT]>=typical[1]*.2]
                coherent=max((p['stats'][k,cv2.CC_STAT_AREA] for k in body_parts),default=0)
                if coherent < max(pitch*pitch*.012,typical_component*.3) or ink < typical_ink*.3:
                    continue
            left,top,right,bottom=map(float,p['bounds'])
            # Common size rejects detached peripheral specks, never clips a
            # continuous long stroke to a prescribed rectangle.
            stats=p['stats']; accepted=p['accepted']
            largest=max(stats[k,cv2.CC_STAT_AREA] for k in accepted)
            keep=[]
            for k in accepted:
                x,y,w,h,area=stats[k]
                cx=col['left']+x+w/2
                cy=col['top']+p['y0']+y+h/2
                if (area>=largest*.08 or
                    (abs(cx-col['cx'])<typical[0]*.5 and abs(cy-p['center'])<typical[1]*.5)):
                    keep.append(k)
            seed=np.isin(p['labels'],keep).astype(np.uint8)
            radius=max(1,round(pitch*.025))
            near=cv2.dilate(seed,np.ones((2*radius+1,2*radius+1),np.uint8))
            yy,xx=np.nonzero(col['weak'][p['y0']:p['y1']] & (near!=0))
            left,top=col['left']+xx.min(),col['top']+p['y0']+yy.min()
            right,bottom=col['left']+xx.max()+1,col['top']+p['y0']+yy.max()+1
            box=np.float32([[left-1,top-1],[right+1,top-1],[right+1,bottom+1],[left-1,bottom+1]])
            item=assignment.get(j,{'char':'','score':0.0,'line':[[left,p['center']],[right,p['center']]],'unrecognized':True})
            item['bbox']=box
            index=min(col['group'],key=lambda c:max(c[2]-p['center'],p['center']-c[3],0))[0]
            rows[index]['result']['char_list'].append(item)
    for index in handled:
        result=rows[index]['result']
        result['char_list'].sort(key=lambda c:float(np.asarray(c['bbox'])[:,1].mean()))
        result['line_pred']=''.join(c['char'] for c in result['char_list'])
        result['words']=[{'value':c['char'],'bbox':c['bbox'].copy()} for c in result['char_list'] if c['char']]
        result['geometry_source']='page-grid-ink'
        result['page_glyph_size']=typical.tolist()
    return handled
