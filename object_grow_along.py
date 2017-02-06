bl_info = {
    "name": "Grow Along",
    "description": "Grow object along a surface",
    "author": "Stanislaw Adaszewski",
    "version": (1, 0, 0),
    "blender": (2, 78, 0),
    "location": "Search > Grow Along",
    "warning": "",
    "wiki_url": "http://algoholic.eu/blender-grow-along/",
    "category": "Object",
}


import bpy
from mathutils import Quaternion, Matrix, Vector, Euler
from mathutils.bvhtree import BVHTree
from mathutils.interpolate import poly_3d_calc
import heapq
import numpy as np


def _smooth_normal(obj, loc, index):
    obj_data = obj.to_mesh(bpy.context.scene, True, 'PREVIEW')
    vert_indices = [obj_data.loops[i].vertex_index for i in obj_data.polygons[index].loop_indices]
    vert_coords = [obj_data.vertices[i].co for i in vert_indices]
    vert_normals = [obj_data.vertices[i].normal for i in vert_indices]
    weights = poly_3d_calc(vert_coords, loc)
    # self.report({'INFO'}, str(vert_normals))
    sum = Vector((0.0, 0.0, 0.0))
    for i in range(len(weights)):
        sum += weights[i] * vert_normals[i]
    return (sum / len(weights))
    
    
def _dijkstra(verts, idx_from, idx_to, connected):
    Q = []
    heapq.heappush(Q, (0, idx_from))
    visited = np.zeros(len(verts), dtype=np.bool)
    dist = np.ones(len(verts)) * np.inf
    dist[idx_from] = 0
    shortest = -np.ones(len(verts), dtype=np.int)
    shortest[idx_from] = idx_from

    while len(Q) > 0:
        (d, idx) = heapq.heappop(Q)
        
        if idx == idx_to:
            break
        
        for neighbor in connected[idx]:
            L = np.linalg.norm(verts[neighbor].co - verts[idx].co)
            new_dist = d + L
            if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    shortest[neighbor] = idx
                    if not visited[neighbor]:
                        heapq.heappush(Q, (new_dist, neighbor))
        
        visited[idx] = 1
    
    if np.isinf(dist[idx_to]):
        return None
        
    path = []
    idx = idx_to
    while idx != idx_from:
        print('idx:', idx)
        path.append(idx)
        idx = shortest[idx]
    path.append(idx_from)
    path = list(reversed(path))

    return (dist[path], path)


class GrowAlong(bpy.types.Operator):
    """Place object on surface with its Z direction aligned with surface normal"""
    bl_idname = "object.grow_along"
    bl_label = "Grow Along"
    bl_options = {'REGISTER', 'UNDO'}
    
    
    def execute(self, context):
        obj = bpy.context.active_object
        surf = bpy.context.scene.objects['surface']
        growth = bpy.context.scene.objects['growth']
        
        verts_growth = growth.data.vertices
        verts_surf = surf.data.vertices
        verts_obj = obj.data.vertices
        
        bvh_surf = BVHTree.FromObject(surf, bpy.context.scene)
        bvh_obj = BVHTree.FromObject(obj, bpy.context.scene)
        
        connected = [[] for i in range(len(verts_surf))]
        for e in surf.data.edges:
            connected[e.vertices[0]].append(e.vertices[1])
            connected[e.vertices[1]].append(e.vertices[0])
        
        
        obj_z = [v.co[2] for v in verts_obj]
        obj_h = max(obj_z) - min(obj_z)
        print('obj_h:', obj_h)
        
        obj_radius = [np.linalg.norm([v.co[0], v.co[1], 0]) for v in verts_obj]
        obj_radius = np.max(obj_radius)
        print('obj_radius:', obj_radius)
        
        
        loc = growth.matrix_world * verts_growth[-1].co
        loc = surf.matrix_world.inverted() * loc
        (loc, normal, index, dist) = bvh_surf.find_nearest(loc)
        dist = [np.linalg.norm(verts_surf[v].co - loc)
            for v in surf.data.polygons[index].vertices]
        v_from = surf.data.polygons[index].vertices[np.argmin(dist)]
        f_from = index
        print('v_from:', v_from, 'f_from:', f_from)
        
        loc = bpy.context.scene.cursor_location
        loc = surf.matrix_world.inverted() * loc
        (loc, normal, index, dist) = bvh_surf.find_nearest(loc)
        dist = [np.linalg.norm(verts_surf[v].co - loc)
            for v in surf.data.polygons[index].vertices]
        v_to = surf.data.polygons[index].vertices[np.argmin(dist)]
        f_to = index
        print('v_to:', v_to, 'f_to:', f_to)
        
        
        (cumdist, path) = _dijkstra(verts_surf, v_from, v_to, connected)
        
        
        for i in range(1, len(path)):
            v = verts_surf[path[i]]
            z = cumdist[i]
            
            obj_z = np.mod(z, obj_h)
            print('v.co:', v.co, 'z:', z, 'obj_z:', obj_z)
            
            (loc, normal, index, dist) = bvh_surf.find_nearest(v.co)
            normal = _smooth_normal(surf, loc, index)
            
            radius = []
            for angle in np.linspace(0, 2*np.pi, 10)[:-1]:
                dx = np.cos(angle)
                dy = np.sin(angle)
                (loc, _, _, _) = bvh_obj.ray_cast(
                    Vector((dx * obj_radius * 2, dy * obj_radius * 2, obj_z)),
                    Vector((-dx, -dy, 0)))
                radius.append(np.linalg.norm([loc[0], loc[1], 0]))
            print('radius:', radius)
            radius = np.max(radius)
            
            
            verts_growth.add(1)
            verts_growth[-1].co = v.co + Vector(radius * normal)
        
        return {'FINISHED'}


def register():
    bpy.utils.register_class(GrowAlong)


def unregister():
    bpy.utils.unregister_class(GrowAlong)
