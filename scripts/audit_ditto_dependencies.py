#!/usr/bin/env python3
"""Static release audit; writes evidence only, never imports or deletes sources.

Counts resolved Python imports separately from filename mentions. Zero incoming
references do not prove a CLI, plugin, generated filename, or public API unused.
"""
import argparse
import ast
import collections
import csv
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = ('docs/ditto-open-source-audit/', 'test/ditto/archive/',
            'test/ditto/results/', 'test/ditto/speedup/head-mapping-ab/')
TEXT_EXT = {'.py', '.sh', '.cc', '.cpp', '.cu', '.cuh', '.h', '.hpp', '.toml',
            '.yaml', '.yml', '.json', '.md', '.txt', '.cmake', ''}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'docs/ditto-open-source-audit')
    args = parser.parse_args()
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    listed = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    files = sorted({p for p in listed if p and (ROOT/p).is_file()})
    texts = {}
    for p in files:
        if (p.startswith(EXCLUDED) or '/3rdparty/' in p or p.startswith('3rdparty/')
                or p in {'OPEN_SOURCE_AUDIT.md', 'scripts/audit_ditto_dependencies.py'}):
            continue
        if Path(p).suffix in TEXT_EXT and (ROOT/p).stat().st_size < 2_000_000:
            try:
                texts[p] = (ROOT/p).read_text(encoding="utf-8-sig")
            except (UnicodeError, OSError):
                pass
    scopes = ('python/sglang/ditto/', 'python/sglang/srt/models/ditto/', 'test/ditto/')
    targets = [p for p in texts if p.startswith(scopes) or
               p in {'sgl-kernel/python/sgl_kernel/kvlib.py', 'sgl-kernel/csrc/kvlib_bindings.cc'} or
               (p.startswith('sgl-kernel/csrc/kvlib/') and '/3rdparty/' not in p)]
    modules = {}
    for p in texts:
        if p.endswith('.py'):
            module = p[:-3].replace('/', '.')
            for prefix in ('python.', 'sgl-kernel.python.'):
                if module.startswith(prefix):
                    module = module[len(prefix):]
            if module.endswith('.__init__'):
                module = module[:-9]
            modules[module] = p
    inverse = {p:m for m,p in modules.items()}
    imports = collections.defaultdict(list)
    definitions = []
    parse_errors = []
    for p, text in texts.items():
        if not p.endswith('.py'):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            parse_errors.append({'path': p, 'line': exc.lineno})
            continue
        top = collections.defaultdict(list)
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                top[n.name].append(n)
        if p in targets:
            for name, nodes in top.items():
                if len(nodes) > 1:
                    definitions.append({'path':p,'symbol':name,
                                        'definitions':[[n.lineno,n.end_lineno] for n in nodes],
                                        'decorated':any(n.decorator_list for n in nodes)})
        package = inverse[p].split('.') if p.endswith('/__init__.py') else inverse[p].split('.')[:-1]
        for n in ast.walk(tree):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                prefix = '.'.join(package[:len(package)-n.level+1]) if n.level else ''
                base = '.'.join(filter(None, [prefix,n.module]))
                names = [base] + [base+'.'+a.name for a in n.names]
            for name in names:
                dest = modules.get(name)
                # Support plain sibling imports in runnable test scripts.
                if dest is None:
                    sibling = str(Path(p).parent/(name.replace('.', '/')+'.py'))
                    dest = sibling if sibling in texts else None
                if dest and dest != p:
                    edge = {'source':p,'line':n.lineno,'module':name}
                    if edge not in imports[dest]:
                        imports[dest].append(edge)
    by_basename = collections.defaultdict(list)
    for p in targets:
        if Path(p).name not in {'__init__.py', 'README.md'}:
            by_basename[Path(p).name].append(p)
    mentions = collections.defaultdict(list)
    pattern = re.compile('|'.join(re.escape(x) for x in sorted(by_basename,key=len,reverse=True)))
    for p,text in texts.items():
        for lineno,line in enumerate(text.splitlines(),1):
            if line.lstrip().startswith(('#','//')) and not line.lstrip().startswith('#include'):
                continue
            for name in set(pattern.findall(line)):
                for dest in by_basename[name]:
                    if dest != p:
                        mentions[dest].append({'source':p,'line':lineno,'ambiguous_basename':len(by_basename[name])>1})
    outgoing = collections.defaultdict(set)
    for dest, edges in imports.items():
        for edge in edges:
            outgoing[edge['source']].add(dest)
    rows=[]
    for p in targets:
        strong=sorted({e['source'] for e in imports[p]})
        refs=sorted({e['source'] for e in mentions[p]})
        docs=[x for x in refs if Path(x).suffix=='.md']
        code=[x for x in refs if x not in docs]
        rows.append({'path':p,'tracked':p in tracked,'bytes':(ROOT/p).stat().st_size,
                     'importing_files':len(strong),'imported_local_files':len(outgoing[p]),
                     'filename_code_mentions':len(code),
                     'filename_doc_mentions':len(docs),
                     'direct_entrypoint':p.endswith('.sh') or bool(re.search(r'__name__\s*==\s*[\'"]__main__|^#!.*(?:bash|sh)',texts[p],re.M)),
                     'machine_path_lines':sum(bool(re.search(r'/(?:jhe|data[0-9]*|root|models|datasets|home|auxiliary|preds)/',line)) for line in texts[p].splitlines()),
                     'importers':';'.join(strong),'imports_local':';'.join(sorted(outgoing[p])),
                     'code_mention_sources':';'.join(code)})
    duplicates=collections.defaultdict(list)
    for p in targets:
        if (ROOT/p).stat().st_size:
            duplicates[hashlib.sha256((ROOT/p).read_bytes()).hexdigest()].append(p)
    exact=[v for v in duplicates.values() if len(v)>1]
    shells=[p for p in targets if texts[p].startswith('#!') and 'bash' in texts[p].splitlines()[0]]
    similar=[]
    for i,a in enumerate(shells):
        for b in shells[i+1:]:
            x,y=texts[a].splitlines(),texts[b].splitlines()
            if min(len(x),len(y))<15 or min(len(x),len(y))/max(len(x),len(y))<.7:
                continue
            ratio=difflib.SequenceMatcher(None,x,y,autojunk=False).ratio()
            if ratio>=.8:
                similar.append({'a':a,'b':b,'line_similarity':round(ratio,4)})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    vendor_candidates = []
    for project, include in [('raft', 'cpp/include/'), ('rmm', 'include/'), ('spdlog', 'include/')]:
        prefix = f'sgl-kernel/csrc/kvlib/3rdparty/{project}/'
        for p in files:
            parts = Path(p).parts
            metadata = any(part.upper().startswith(('LICENSE', 'NOTICE', 'COPYING')) for part in parts)
            if (p in tracked and p.startswith(prefix) and not p[len(prefix):].startswith(include)
                    and not metadata and Path(p).name != 'README.md'):
                vendor_candidates.append(p)
    (args.output_dir/'vendor-trim-candidates.txt').write_text('\n'.join(sorted(vendor_candidates))+'\n')
    with (args.output_dir/'inventory.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); writer.writeheader();writer.writerows(rows)
    evidence={'scope':'Nonignored tracked and untracked working tree; no imports executed. Excludes archives, results, head-mapping artifacts and third-party trees from reference sources.',
              'limitations':'Imports are static and conditional imports count. Filename mentions are not proven calls. Dynamic paths, plugins, external users and generated inputs need manual review.',
              'source_files_scanned':len(texts),'target_files':len(targets),'parse_errors':parse_errors,
              'vendor_candidate_files':len(vendor_candidates),
              'vendor_candidate_bytes':sum((ROOT/p).stat().st_size for p in vendor_candidates),
              'imports':{p:imports[p] for p in targets if imports[p]},
              'filename_mentions':{p:mentions[p] for p in targets if mentions[p]},
              'shadowed_definitions':definitions,'exact_duplicate_files':exact,
              'similar_shell_pairs':sorted(similar,key=lambda x:-x['line_similarity'])}
    (args.output_dir/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({k:evidence[k] for k in ('source_files_scanned','target_files','parse_errors','shadowed_definitions','similar_shell_pairs')},indent=2))


if __name__=='__main__':
    main()
