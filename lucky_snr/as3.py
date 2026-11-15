"""Read AutoStakkert! session files (.as3) written next to each stack."""
import os
import re


def read_as3(path):
    """settings dict (keys without the leading underscore) + 'source' (the .ser that was stacked)."""
    st = {}
    src = None
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if s.startswith('Current file:') and src is None:
                src = s.split(':', 1)[1].strip()
            elif s.startswith('file ') and src is None:
                src = s[5:].strip()
            elif s.startswith('_'):
                parts = s.split(None, 1)
                st[parts[0][1:]] = parts[1].strip() if len(parts) > 1 else ''
    st['source'] = src
    return st


def basename_any(p):
    """file name from a Windows or POSIX path (AS! under Wine writes Z:\\...)."""
    return re.split(r'[\\/]', p)[-1] if p else ''


def describe(stack_path):
    """settings of an AS! stack, from its .as3 and its file name. Returns dict or None if not an AS! stack."""
    base = os.path.splitext(stack_path)[0]
    as3 = base + '.as3'
    name = os.path.basename(base)
    m = re.search(r'(?:^|_)AS_([PF])(\d+)', name) or re.search(r'AS_([PF])(\d+)', stack_path)
    if not os.path.exists(as3) and not m:
        return None
    st = read_as3(as3) if os.path.exists(as3) else {}
    d = {'stack': stack_path, 'source': basename_any(st.get('source')) or None}
    if m:
        d['select'] = ('%s%%' % m.group(2)) if m.group(1) == 'P' else ('%s frames' % m.group(2))
    else:
        d['select'] = st.get('stack_frames_percentage_1', '?') + '%'
    d['ap_size'] = st.get('frameview_ap_size')
    dz = re.search(r'(Drizzle\d+|Resample\d+)', name)
    d['drizzle'] = dz.group(1) if dz else (st.get('advanced_drizzle_resample') if st.get('advanced_drizzle_resample') not in (None, 'None') else '')
    q = re.search(r'_(lapl\d+|grad\d+)', name)
    d['quality'] = q.group(1) if q else ''
    n_ap = re.search(r'_ap(\d+)', name)
    d['n_ap'] = n_ap.group(1) if n_ap else ''
    d['sharpened'] = st.get('stack_sharpened', 'False') == 'True' or name.endswith('_conv')
    return d


def label(d):
    s = 'AS! AP%s %s' % (d['ap_size'] or '?', d['select'])
    if d['drizzle']:
        s += ' ' + d['drizzle']
    return s
