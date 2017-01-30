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
import numpy as np

    
def _get_group_neighbors(obj):
    n_v = len(obj.data.vertices)
    n_g = len(obj.vertex_groups)
    
    connected = [[] for i in range(n_v)]
    
    for e in obj.data.edges:
        v_1 = e.vertices[0]
        v_2 = e.vertices[1]
        connected[v_1].append(v_2)
        connected[v_2].append(v_1)
    
    vtx_group = [-1] * n_v
    fifo = []
    for v in obj.data.vertices:
        groups = list(map(lambda ge: ge.group, v.groups))
        if len(groups) > 1:
            raise ValueError('Vertex %d has multiple groups: %s' % (v.index, str(', '.join(map(str, groups)))))
        if len(groups) == 1:
            g = groups[0]
            fifo.append(v.index)
            vtx_group[v.index] = g
            
    neighbors = [False] * (n_g * n_g)
    
    while len(fifo) > 0:
        v_1 = fifo.pop(0)
        g_1 = vtx_group[v_1]
        for v_2 in connected[v_1]:
            g_2 = vtx_group[v_2]
            if g_2 == g_1:
                continue
            elif g_2 == -1:
                # obj.vertex_groups[g_1].add([v_2], 1.0, 'ADD')
                vtx_group[v_2] = g_1
                fifo.append(v_2)
            else: # neighbors
                neighbors[g_1 * n_g + g_2] = True
                neighbors[g_2 * n_g + g_1] = True
    
    return neighbors
    

class GenerateArmature(bpy.types.Operator):
    """Generate armature from vertex groups"""
    bl_idname = "rigging.generate_armature"
    bl_label = "Generate Armature"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = bpy.context.active_object
        
        bpy.ops.object.add(
            type='ARMATURE', 
            enter_editmode=True,
            location=obj.location)
            
        amt = bpy.context.object.data
        amt.name = obj.name + '_Amt'
        
        neighbors = _get_group_neighbors(obj)
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
            
        tails = [[] for i in range(n)]
        
        for i in range(n):
            for k in range(i + 1, n):
                if not neighbors[i * n + k]:
                    continue
                    
                p_1 = centers[i]
                p_2 = centers[k]
                
                bone = amt.edit_bones.new('%s_%s' %
                    (obj.vertex_groups[i].name,
                    obj.vertex_groups[k].name))
                if len(tails[i]) > 0:
                    parent = tails[i][0]
                    bone.parent = parent
                    bone.head = parent.tail
                    bone.tail = p_2
                    bone.use_connect = True
                    tails[k].append(bone)
                elif len(tails[k]) > 0:
                    parent = tails[k][0]
                    bone.parent = parent
                    bone.head = parent.tail
                    bone.tail = p_1
                    bone.use_connect = True
                    tails[i].append(bone)
                else:
                    bone.head = p_1
                    bone.tail = p_2
                    tails[k].append(bone)
            
        bpy.ops.object.mode_set(mode='OBJECT')
        
        return {'FINISHED'}


def register():
    bpy.utils.register_class(GenerateArmature)


def unregister():
    bpy.utils.unregister_class(GenerateArmature)
