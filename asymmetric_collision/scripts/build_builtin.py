#!/usr/bin/env python3
"""Build the builtin test problem from the candidate's own objects and headers."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

repo = Path(__file__).resolve().parents[2]
build = Path(sys.argv[1]).resolve()
out = build / 'validation'
out.mkdir(exist_ok=True)
units = json.loads((build / 'compile_commands.json').read_text())
unit = next(row for row in units if row['file'].endswith('/src/pgen/pgen.cpp'))
source = out / 'builtin.cpp'
source.write_text('#include "athena.hpp"\n#undef USER_PROBLEM_ENABLED\n'
    '#define USER_PROBLEM_ENABLED 0\n#include "' + str(repo / 'src/pgen/pgen.cpp') + '"\n')
command = shlex.split(unit['command'])
command[command.index('-o') + 1] = str(out / 'builtin.o')
command[command.index('-c') + 1] = str(source)
with (out / 'compile.log').open('w') as log:
    subprocess.run(command, cwd=unit['directory'], stdout=log, stderr=subprocess.STDOUT, check=True)
original = shlex.split((build / 'src/CMakeFiles/athena.dir/link.txt').read_text())
link = [str(out / 'builtin.o') if x.endswith('/pgen/pgen.cpp.o') else x
        for x in original if not x.endswith('/asymmetric_collision/asymmetric_collision.cpp.o')]
exe = out / 'athena-builtin'
link[link.index('-o') + 1] = str(exe)
with (out / 'link.log').open('w') as log:
    subprocess.run(link, cwd=build / 'src', stdout=log, stderr=subprocess.STDOUT, check=True)
(out / 'provenance.json').write_text(json.dumps(dict(compile=command, link=link,
    executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()), indent=2) + '\n')
print(exe, flush=True)
