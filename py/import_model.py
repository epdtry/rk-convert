import bpy
from collections import namedtuple
from pprint import pprint
import os
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


Triangle = namedtuple('Triangle', ('verts', 'uvs'))
Model = namedtuple('Model', ('name', 'verts', 'tris', 'weights', 'edges'))

def read_triangle(f):
    verts = read_struct(f, '<3I')
    uvs = [read_struct(f, '<2f') for _ in range(3)]
    return Triangle(verts, uvs)

def read_model(f):
    num_verts, num_tris, num_weights = read_struct(f, '<III')

    name = read_str(f)

    verts = [read_struct(f, '<3f') for _ in range(num_verts)]
    tris = [read_triangle(f) for _ in range(num_tris)]
    weights = [read_struct(f, '<IIf') for _ in range(num_weights)]

    print('read %d verts, %d tris, %d weights' %
            (len(verts), len(tris), len(weights)))

    edges = []
    for t in tris:
        a, b, c = t.verts
        edges.extend([(a,b),(b,c),(c,a)])

    return Model(name, verts, tris, weights, edges)



num_models, num_bones = read_struct(f, '<II')

models = [read_model(f) for _ in range(num_models)]
bones = [read_bone(f) for _ in range(num_bones)]
print('read %d models, %d bones' % (len(models), len(bones)))


# Remove initial objects

objs = list(bpy.context.scene.collection.children.values())
for o in objs:
    bpy.context.scene.collection.children.unlink(o)
del o


# Create armature object
if len(bones) > 0:
    arm = bpy.data.armatures.new('ArmatureData')
    arm_obj = bpy.data.objects.new('Armature', arm)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)

    edit_bones = arm.edit_bones
    bs = [arm.edit_bones.new(name) for name, _, _ in bones]
    for b, (_, parent, matrix) in zip(bs, bones):
        b.head = transform(matrix, (0, 0, 0))
        b.tail = transform(matrix, (1, 0, 0))
        if parent is not None:
            b.parent = bs[parent]

    bpy.ops.object.mode_set(mode='OBJECT')


# Create material object
mat = bpy.data.materials.new(name='Material')
mat.use_nodes = True
ntree = mat.node_tree

for n in list(ntree.nodes):
    ntree.nodes.remove(n)

n_output = ntree.nodes.new('ShaderNodeOutputMaterial')
n_emission = ntree.nodes.new('ShaderNodeEmission')
n_image = ntree.nodes.new('ShaderNodeTexImage')

ntree.links.new(n_emission.inputs['Color'], n_image.outputs['Color'])
ntree.links.new(n_output.inputs['Surface'], n_emission.outputs['Emission'])

n_image.image = bpy.data.images.load(os.path.abspath('pony.tga'))


# Create mesh objects

for m in models:

    mesh = bpy.data.meshes.new('MeshData-%s' % m.name)
    mesh.from_pydata(m.verts, m.edges, [t.verts for t in m.tris])
    ok = mesh.validate()
    assert ok, 'mesh validation failed'
    mesh.calc_loop_triangles()

    uvs = mesh.uv_layers.new()
    for (i, loop_tri) in enumerate(mesh.loop_triangles):
        for (j, loop_index) in enumerate(loop_tri.loops):
            uvs.data[loop_index].uv = m.tris[i].uvs[j]

    mesh.materials.append(mat)

    mesh_obj = bpy.data.objects.new('Mesh-%s' % m.name, mesh)

    bpy.context.scene.collection.objects.link(mesh_obj)

    if len(bones) > 0:
        # Create vertex groups (one per bone)
        vgs = []
        for (name, _, _) in bones:
            vgs.append(mesh_obj.vertex_groups.new(name=name))

        # Add vertex weights
        for v, b, w in m.weights:
            vgs[b].add((v,), w, type='REPLACE')

        # TODO: add armature modifier


# Sove file

bpy.ops.wm.save_as_mainfile(filepath='out.blend', relative_remap=False)
