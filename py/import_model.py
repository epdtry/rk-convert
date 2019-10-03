import bpy
from collections import namedtuple
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


Model = namedtuple('Model', ('name', 'verts', 'tris', 'weights', 'edges'))

def read_model(f):
    num_verts, num_tris, num_weights = read_struct(f, '<III')

    name = read_str(f)

    verts = [read_struct(f, '<3f') for _ in range(num_verts)]
    tris = [read_struct(f, '<3I') for _ in range(num_tris)]
    weights = [read_struct(f, '<IIf') for _ in range(num_weights)]

    print('read %d verts, %d tris, %d weights' %
            (len(verts), len(tris), len(weights)))

    edges = [x for a,b,c in tris for x in [(a,b),(b,c),(c,a)]]

    return Model(name, verts, tris, weights, edges)



num_models, num_bones = read_struct(f, '<II')

models = [read_model(f) for _ in range(num_models)]
bones = [read_bone(f) for _ in range(num_bones)]
print('read %d models, %d bones' % (len(models), len(bones)))


# Remove initial objects

objs = list(bpy.context.scene.objects.values())
for o in objs:
    bpy.context.scene.objects.unlink(o)
del o


# Create armature object
if len(bones) > 0:
    arm = bpy.data.armatures.new('ArmatureData')
    arm_obj = bpy.data.objects.new('Armature', arm)
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


# Create mesh objects

for m in models:

    mesh = bpy.data.meshes.new('MeshData-%s' % m.name)
    mesh.from_pydata(m.verts, m.edges, m.tris)
    ok = mesh.validate()
    assert ok, 'mesh validation failed'

    mesh_obj = bpy.data.objects.new('Mesh-%s' % m.name, mesh)

    bpy.context.scene.objects.link(mesh_obj)

    if len(bones) > 0:
        # Create vertex groups (one per bone)
        vgs = []
        for (name, _, _) in bones:
            vgs.append(mesh_obj.vertex_groups.new(name))

        # Add vertex weights
        for v, b, w in m.weights:
            vgs[b].add((v,), w, type='REPLACE')


# Sove file

bpy.ops.wm.save_as_mainfile(filepath='out.blend')