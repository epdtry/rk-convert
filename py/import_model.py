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
    return read_struct(f, '<%ds' % l)[0].decode('utf-8')

def read_bone(f):
    name = read_str(f)
    parent, = read_struct(f, '<I')
    if parent == 0xffffffff:
        parent = None
    matrix = read_struct(f, '<16f')
    return (name, parent, matrix)

def transform(m, v):
    # `m` is in column-major order
    x = m[ 0] * v[0] + m[ 4] * v[1] + m[ 8] * v[2] + m[12]
    y = m[ 1] * v[0] + m[ 5] * v[1] + m[ 9] * v[2] + m[13]
    z = m[ 2] * v[0] + m[ 6] * v[1] + m[10] * v[2] + m[14]
    return (x, y, z)

num_verts, num_tris, num_bones = read_struct(f, '<III')

verts = [read_struct(f, '<3f') for _ in range(num_verts)]
tris = [read_struct(f, '<3I') for _ in range(num_tris)]
bones = [read_bone(f) for _ in range(num_bones)]

edges = [x for a,b,c in tris for x in [(a,b),(b,c),(c,a)]]


# Remove initial objects

objs = list(bpy.context.scene.objects.values())
for o in objs:
    bpy.context.scene.objects.unlink(o)


# Create mesh object

m = bpy.data.meshes.new('ImportedMesh')
m.from_pydata(verts, edges, tris)
ok = m.validate()
assert ok, 'mesh validation failed'

o = bpy.data.objects.new('ImportedObject', m)

bpy.context.scene.objects.link(o)


# Create armature object

a = bpy.data.armatures.new('ImportedArmatureData')
o = bpy.data.objects.new('ImportedArmature', a)
bpy.context.scene.objects.link(o)
bpy.context.scene.objects.active = o
bpy.ops.object.mode_set(mode='EDIT', toggle=False)

edit_bones = a.edit_bones
bs = [a.edit_bones.new(name) for name, _, _ in bones]
for b, (_, parent, matrix) in zip(bs, bones):
    b.head = transform(matrix, (0, 0, 0))
    b.tail = transform(matrix, (1, 0, 0))
    if parent is not None:
        b.parent = bs[parent]

bpy.ops.object.mode_set(mode='OBJECT')


# Sove file

bpy.ops.wm.save_as_mainfile(filepath='out.blend')
