import bpy
from pprint import pprint
import struct
import sys

args = []
for i, x in enumerate(sys.argv):
    if x == '--':
        args = sys.argv[i + 1:]
        break

assert len(args) == 1, 'usage: import_model.py <file.model>'
f = open(args[0], 'rb')

def read_struct(f, fmt):
    sz = struct.calcsize(fmt)
    buf = f.read(sz)
    assert len(buf) == sz
    return struct.unpack(fmt, buf)

def read_str(f):
    l = read_struct(f, '<H')
    return read_struct(f, '<%ds' % l)[0]

num_verts, num_tris = read_struct(f, '<II')

verts = [read_struct(f, '<3f') for _ in range(num_verts)]
tris = [read_struct(f, '<3I') for _ in range(num_tris)]

edges = [x for a,b,c in tris for x in [(a,b),(b,c),(c,a)]]

print(bpy.data)

objs = list(bpy.context.scene.objects.values())
for o in objs:
    bpy.context.scene.objects.unlink(o)

m = bpy.data.meshes.new('ImportedMesh')
m.from_pydata(verts, edges, tris)
ok = m.validate()
assert ok, 'mesh validation failed'
print(len(m.vertices))

o = bpy.data.objects.new('ImportedObject', m)
print(o)

bpy.context.scene.objects.link(o)

bpy.ops.wm.save_as_mainfile(filepath='out.blend')
