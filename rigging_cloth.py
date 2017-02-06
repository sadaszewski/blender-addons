bl_info = {
    "name": "Cloth to Armature",
    "description": "Parent cloth to armature using already rigged child",
    "author": "Stanislaw Adaszewski",
    "version": (1, 0, 0),
    "blender": (2, 78, 0),
    "location": "Search > Cloth to Armature",
    "warning": "",
    "wiki_url": "http://algoholic.eu/blender-cloth-armature/",
    "category": "Rigging",
}


import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
import heapq


##def _walk_dist(verts, idx_from, idx_to, connected):
##    dist = np.ones(len(verts)) * np.inf
##    dist[idx_from] = 0
##    u_visited = [i for i in range(len(verts))]
##
##    while len(u_visited) > 0:
##        u_dist = dist[u_visited]
##        idx = np.argmin(u_dist)
##        for neighbor in connected[idx]:
##            L = np.linalg.norm(verts[neighbor].co - verts[idx].co)
##            new_dist = dist[idx] + L
##            if new_dist < dist[neighbor]:
##                    dist[neighbor] = new_dist
##        u_visited.pop(idx)
##        if idx == idx_to:
##            break
##
##    return dist[idx_to]


def _walk_dist(verts, idx_from, idx_to, connected, limit):
    Q = []
    heapq.heappush(Q, (0, idx_from))
    visited = np.zeros(len(verts), dtype=np.bool)
    dist = np.ones(len(verts)) * np.inf
    dist[idx_from] = 0

    while len(Q) > 0:
        (d, idx) = heapq.heappop(Q)
        
        if idx == idx_to or d > limit:
            break
        
        for neighbor in connected[idx]:
            L = np.linalg.norm(verts[neighbor].co - verts[idx].co)
            new_dist = d + L
            if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    if not visited[neighbor]:
                        heapq.heappush(Q, (new_dist, neighbor))

        visited[idx] = 1

        

    return dist[idx_to]


def _neighbors(verts, idx, ttl, connected):
    Q = [(idx, ttl)]
    ret = set()

    while len(Q) > 0:
        p = Q.pop(0)
        ret.add(p[0])
        if p[1] > 0:
            for neigh in connected[p[0]]:
                Q.append((neigh, p[1] - 1))

    return list(ret)


class ClothArmature(bpy.types.Operator):
    """Parent cloth to armature using already rigged child"""
    bl_idname = "rigging.cloth_armature"
    bl_label = "Cloth to Armature"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        ch = bpy.context.active_object
        sel = bpy.context.selected_objects
        obj = sel[0] if sel[1] == ch else sel[1]
        amt = ch.parent
        
        bvh = BVHTree.FromObject(ch, bpy.context.scene)    

        obj.vertex_groups.clear()

        for g in ch.vertex_groups:
            obj.vertex_groups.new(g.name)

        closest_idx = np.zeros(len(obj.data.vertices), dtype=np.int)
        closest_dist = np.zeros(len(obj.data.vertices))

        for v in obj.data.vertices:
            loc = obj.matrix_world * v.co # to world
            loc = ch.matrix_world.inverted() * loc # to child
            (loc, normal, index, dist) = bvh.find_nearest(loc)
            # loc = ch.matrix_world * loc # to world
            # loc = obj.matrix_world.inverted() * loc # to obj
            # v.co = loc

            closest_idx[v.index] = index
            closest_dist[v.index] = dist

        ttl = 5

        n_obj_v = len(obj.data.vertices)
        obj_connected = [[] for i in range(n_obj_v)]
    
        for e in obj.data.edges:
            v_1 = e.vertices[0]
            v_2 = e.vertices[1]
            obj_connected[v_1].append(v_2)
            obj_connected[v_2].append(v_1)

        n_ch_v = len(ch.data.vertices)
        ch_connected = [[] for i in range(n_ch_v)]
    
        for e in ch.data.edges:
            v_1 = e.vertices[0]
            v_2 = e.vertices[1]
            ch_connected[v_1].append(v_2)
            ch_connected[v_2].append(v_1)

        # reassigned = True
        '''for i in range(10):
            print('i:', i)
            reassigned = 0
            for v in obj.data.vertices:
                if v.index % 100 == 0:
                    print('v.index:', v.index, 'reassigned:', reassigned)
                for neigh in obj_connected[v.index]:
                    # print('neigh:', neigh)
                    delta = np.linalg.norm(v.co - obj.data.vertices[neigh].co)
                    thresh = 50 * delta
                    
                    w_dist = _walk_dist(ch.data.vertices,
                        closest_idx[neigh],
                        closest_idx[v.index],
                        ch_connected, thresh)
                    # print('_walk_dist done')
                    # delta = closest_dist[v.index] - closest_dist[neigh]
                    
                    if w_dist > thresh:
                        closest_idx[v.index] = closest_idx[neigh]
                        closest_dist[v.index] = np.linalg.norm(
                            obj.data.vertices[v.index].co -
                            ch.data.vertices[closest_idx[v.index]].co)
                        reassigned += 1
                        # print('reassigned!')
                        break
            if reassigned == 0:
                break'''

        for v in obj.data.vertices:
            # N = _neighbors(obj.data.vertices, v.index, ttl, connected)

            # dists = closest_dist[N]
            # indices = closest_idx[N]
            # order = np.argsort(dists)
            # dists = dists[order]
            # indices = indices[order]
            # med_dist = dists[len(N) // 2]
            # med_idx = indices[len(N) // 2]
            
            index = closest_idx[v.index]
            f_ch = ch.data.polygons[index]
            weights = [0.0] * len(obj.vertex_groups)
            n_vert = len(f_ch.vertices)
            for v_ch_idx in f_ch.vertices:
                v_ch = ch.data.vertices[v_ch_idx]
                for ge in v_ch.groups:
                    weights[ge.group] += ge.weight

            for g in obj.vertex_groups:
                if weights[g.index] > 0:
                    g.add([v.index], weights[g.index] / n_vert, 'ADD')

        
        obj.parent = amt
        obj.matrix_parent_inverse = amt.matrix_world.inverted()

        mod = obj.modifiers.new('Armature', 'ARMATURE')
        mod.object = amt

        
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ClothArmature)


def unregister():
    bpy.utils.unregister_class(ClothArmature)
