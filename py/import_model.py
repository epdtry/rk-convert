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

num_verts, num_tris, num_bones, num_weights = read_struct(f, '<IIII')

verts = [read_struct(f, '<3f') for _ in range(num_verts)]
tris = [read_struct(f, '<3I') for _ in range(num_tris)]
bones = [read_bone(f) for _ in range(num_bones)]
weights = [read_struct(f, '<IIf') for _ in range(num_weights)]

print('read %d verts, %d tris, %d bones, %d weights' %
        (len(verts), len(tris), len(bones), len(weights)))

edges = [x for a,b,c in tris for x in [(a,b),(b,c),(c,a)]]


# Remove initial objects

objs = list(bpy.context.scene.objects.values())
for o in objs:
    bpy.context.scene.objects.unlink(o)
del o


# Create mesh object

mesh = bpy.data.meshes.new('ImportedMesh')
mesh.from_pydata(verts, edges, tris)
ok = mesh.validate()
assert ok, 'mesh validation failed'

mesh_obj = bpy.data.objects.new('ImportedObject', mesh)

bpy.context.scene.objects.link(mesh_obj)



if len(bones) > 0:
    # Create armature object

    arm = bpy.data.armatures.new('ImportedArmatureData')
    arm_obj = bpy.data.objects.new('ImportedArmature', arm)
    bpy.context.scene.objects.link(arm_obj)
    bpy.context.scene.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)

    edit_bones = arm.edit_bones
    bs = [arm.edit_bones.new(name) for name, _, _ in bones]
    for b, (_, parent, matrix) in zip(bs, bones):
        b.head = transform(matrix, (0, 0, 0))
        b.tail = transform(matrix, (1, 0, 0))
        if parent is not None:
            b.parent = bs[parent]

    bpy.ops.object.mode_set(mode='OBJECT')


    # Create vertex groups (one per bone)
    vgs = []
    for (name, _, _) in bones:
        vgs.append(mesh_obj.vertex_groups.new(name))


    # Add vertex weights
    for v, b, w in weights:
        vgs[b].add((v,), w, type='REPLACE')



# Sove file

bpy.ops.wm.save_as_mainfile(filepath='out.blend')
