bl_info = {
    "name": "Generate Armature",
    "description": "Generate armature for given mesh using user-specified vertex groups",
    "author": "Stanislaw Adaszewski",
    "version": (1, 0, 0),
    "blender": (2, 78, 0),
    "location": "Search > Generate Armature",
    "warning": "",
    "wiki_url": "http://algoholic.eu/blender-automatic-armature-generation/",
    "category": "Rigging",
}


import bpy
from mathutils import Quaternion, Matrix, Vector, Euler
from mathutils.bvhtree import BVHTree
from mathutils.interpolate import poly_3d_calc
import numpy as np
from collections import defaultdict
import bmesh

    
'''def get_group_neighbors(obj):
    n = len(obj.vertex_groups)
    neighbors = [False] * (n * n)
    for e in obj.data.edges:
        v_1 = obj.data.vertices[e.vertices[0]]
        v_2 = obj.data.vertices[e.vertices[1]]
        for g_1 in v_1.groups:
            for g_2 in v_2.groups:
                if g_1.group != g_2.group:
                    neighbors[g_1.group * n + g_2.group] = True
                    neighbors[g_2.group * n + g_1.group] = True
    return neighbors'''
    
    
def ncv(obj):
    ret = [None] * len(obj.data.vertices)
    
    '''connected = defaultdict(lambda: [])
    
    for e in obj.data.edges:
        v_1 = e.vertices[0]
        v_2 = e.vertices[1]
        connected[v_1].append(v_2)
        connected[v_2].append(v_1)
        
    for v_1 in range(len(obj.data.vertices)):
        n_1 = obj.data.vertices[v_1].normal
        sum = 0.0
        for v_2 in connected[v_1]:
            n_2 = obj.data.vertices[v_2].normal
            sum += 1.0 - np.dot(n_1, n_2)
        ret[v_1] = sum / len(connected[v_1])'''
        
    
        
    bpy.ops.object.mode_set(mode='EDIT')
    
    bm = bmesh.from_edit_mesh(obj.data)
    cnt = 0
    for v_1 in bm.verts:
        n_1 = v_1.normal
        sum = 0.0
        for e in v_1.link_edges:
            if e.verts[0] == v_1:
                v_2 = e.verts[1]
            else:
                v_2 = e.verts[0]
            n_2 = v_2.normal
            sum += 1.0 - np.dot(n_1, n_2)
        ret[cnt] = sum / len(v_1.link_edges)
        cnt += 1
    
    thr = np.mean(ret)
    
    cnt = 0
    for v in bm.verts:
        if ret[cnt] >= thr:
            v.select = True
        cnt += 1
        
    bpy.context.scene.objects.active = bpy.context.scene.objects.active
    
    # bpy.ops.object.mode_set(mode='OBJECT')
        
    return ret
    
    
def get_group_neighbors(obj):
    n = len(obj.vertex_groups)
    group_dist = np.eye(n)
    group_dist[group_dist == 0] = np.inf
    group_dist[group_dist == 1] = 0

    connected = defaultdict(lambda: [])
    
    for e in obj.data.edges:
        v_1 = e.vertices[0]
        v_2 = e.vertices[1]
        connected[v_1].append(v_2)
        connected[v_2].append(v_1)
    
    
    for g in obj.vertex_groups:
        fifo = []
        dist = np.ones(len(obj.data.vertices)) * np.inf
        
        for v in obj.data.vertices:
            groups = map(lambda ge: ge.group, v.groups)
            if g.index in groups:
                fifo.append(v.index)
                dist[v.index] = 0
                
        while len(fifo) > 0:
            v_1 = fifo.pop(0)
            for v_2 in connected[v_1]:
                d = np.linalg.norm(obj.data.vertices[v_2].co - obj.data.vertices[v_1].co)
                new_dist = dist[v_1] + d
                if new_dist < dist[v_2]:
                    dist[v_2] = new_dist
                    groups = list(map(lambda ge: ge.group, obj.data.vertices[v_2].groups))
                    if g.index in groups:
                        groups.remove(g.index)
                    if len(groups) > 0:
                        g_2 = groups[0]
                        shorter_dist = min(group_dist[g.index, g_2], new_dist)
                        group_dist[g.index, g_2] = \
                            group_dist[g_2, g.index] = shorter_dist
                    else:
                        fifo.append(v_2)
                        
    neighbors = [False] * (n * n)
    for i in range(n):
        for k in range(n):
            if not np.isinf(group_dist[i, k]):
                neighbors[i * n + k] = \
                    neighbors[k * n + i] = True
                        
    return neighbors
    

class GenerateArmature(bpy.types.Operator):
    """Place object on surface with its Z direction aligned with surface normal"""
    bl_idname = "rigging.generate_armature"
    bl_label = "Generate Armature"
    bl_options = {'REGISTER', 'UNDO'}
    
    '''align_with_normal = bpy.props.FloatProperty(
        name="Align with normal", min=0, max=1, default=1)
        
    use_smooth = bpy.props.BoolProperty(
        name="Use smooth normal", default=True)'''

    def execute(self, context):
        obj = bpy.context.active_object
        
        bpy.ops.object.add(
            type='ARMATURE', 
            enter_editmode=True,
            location=obj.location)
            
        amt = bpy.context.object.data
        amt.name = obj.name + 'Amt'
        
        neighbors = get_group_neighbors(obj)
        n = len(obj.vertex_groups)
        
        centers = [None] * n
        for g in obj.vertex_groups:
            pts = []
            for v in obj.data.vertices:
                groups = map(lambda ge: ge.group, v.groups)
                if g.index in groups:
                    pts.append(v.co)
            pts = np.array(pts)
            centers[g.index] = tuple(np.mean(pts, axis=0))
            
        tails = defaultdict(lambda: [])
        
        for i in range(n):
            for k in range(i + 1, n):
                if not neighbors[i * n + k]:
                    continue
                    
                p_1 = centers[i]
                p_2 = centers[k]
                
                bone = amt.edit_bones.new('%s_%s' %
                    (obj.vertex_groups[i].name,
                    obj.vertex_groups[k].name))
                if len(tails[p_1]) > 0:
                    parent = tails[p_1][0]
                    bone.parent = parent
                    bone.head = parent.tail
                    bone.tail = p_2
                    bone.use_connect = True
                    tails[p_2].append(bone)
                elif len(tails[p_2]) > 0:
                    parent = tails[p_2][0]
                    bone.parent = parent
                    bone.head = parent.tail
                    bone.tail = p_1
                    bone.use_connect = True
                    tails[p_1].append(bone)
                else:
                    bone.head = p_1
                    bone.tail = p_2
                    tails[p_2].append(bone)
            
        bpy.ops.object.mode_set(mode='OBJECT')
        
        return {'FINISHED'}


def register():
    bpy.utils.register_class(GenerateArmature)


def unregister():
    bpy.utils.unregister_class(GenerateArmature)
