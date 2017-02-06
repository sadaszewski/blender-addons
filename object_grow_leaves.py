bl_info = {
    "name": "Grow Leaves",
    "description": "Grow leaves on an object avoiding collisions with designated surface",
    "author": "Stanislaw Adaszewski",
    "version": (1, 0, 0),
    "blender": (2, 78, 0),
    "location": "Search > Grow Leaves",
    "warning": "",
    "wiki_url": "http://algoholic.eu/blender-grow-leaves/",
    "category": "Object",
}


import bpy
from mathutils import Quaternion, Matrix, Vector, Euler
from mathutils.bvhtree import BVHTree
from mathutils.interpolate import poly_3d_calc
import heapq
import numpy as np


def _smooth_normal(obj_data, loc, index):
    vert_indices = [obj_data.loops[i].vertex_index for i in obj_data.polygons[index].loop_indices]
    vert_coords = [obj_data.vertices[i].co for i in vert_indices]
    vert_normals = [obj_data.vertices[i].normal for i in vert_indices]
    weights = poly_3d_calc(vert_coords, loc)
    # print('weights:', weights)
    # self.report({'INFO'}, str(vert_normals))
    sum = Vector((0.0, 0.0, 0.0))
    for i in range(len(weights)):
        sum += weights[i] * vert_normals[i]
    return sum
    
    



class GrowLeaves(bpy.types.Operator):
    """Grow leaves on an object avoiding collisions with designated surface"""
    bl_idname = "object.grow_leaves"
    bl_label = "Grow Leaves"
    bl_options = {'REGISTER', 'UNDO'}
    
    
    number_of_leaves = bpy.props.IntProperty(
        name="Number of leaves", min=1, default=20)
        
    # use_smooth = bpy.props.BoolProperty(
        # name="Use smooth normal", default=True)
    
    
    def execute(self, context):
        leaf = bpy.context.scene.objects['leaf']
        branch = bpy.context.scene.objects['branch']
        surf = bpy.context.scene.objects['surface']
        
        faces_branch = branch.data.polygons
        n_faces_branch = len(faces_branch)
        verts_branch = branch.data.vertices
        
        
        bvh_surf = BVHTree.FromObject(surf, bpy.context.scene)
        bvh_branch = BVHTree.FromObject(branch, bpy.context.scene)
        
        (unused, branch_rot, unused) = branch.matrix_world.decompose()
        (unused, leaf_rot, leaf_scale) = leaf.matrix_world.decompose()
        
        branch_data = branch.to_mesh(bpy.context.scene, True, 'PREVIEW')
        
        for i in range(self.number_of_leaves):
            f = np.random.randint(n_faces_branch)
            face = faces_branch[f]
            w = np.random.random(len(face.vertices))
            w /= np.sum(w)
            loc = np.zeros(3)
            for k in range(len(w)):
                loc += w[k] * verts_branch[face.vertices[k]].co
            loc = Vector(loc) # branch
            loc = branch.matrix_world * loc # world
                
            normal = _smooth_normal(branch_data, loc, f) # branch
            normal = branch_rot * normal # world rotation
            vec = leaf_rot * Vector((0.0, 0.0, 1.0))
            q = vec.rotation_difference(normal) # from leaf to branch rot in world
            
            # bpy.context.scene.objects.active = leaf
            
            mat_scale = Matrix()
            for i in range(3): mat_scale[i][i] = leaf_scale[i]
            matrix_world = (Matrix.Translation(loc) *
                q.to_matrix().to_4x4() * leaf_rot.to_matrix().to_4x4() *
                mat_scale)
                
            verts = []
            faces = []
                
            for v in leaf.data.vertices:
                verts.append(matrix_world * v.co)
            for f in leaf.data.polygons:
                faces.append(f.vertices)
                
            bvh_leaf = BVHTree.FromPolygons(verts, faces)
            overlap_surf = bvh_leaf.overlap(bvh_surf)
            
            if len(overlap_surf) > 0:
                continue
                
            overlap_branch = bvh_leaf.overlap(bvh_branch)
            if len(overlap_branch) > 0:
                continue
                
            bpy.ops.object.duplicate()
            new_obj = bpy.context.active_object
            new_obj.matrix_world = matrix_world
            
            # new_obj.matrix_world = Matrix()
            # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        return {'FINISHED'}


def register():
    bpy.utils.register_class(GrowLeaves)


def unregister():
    bpy.utils.unregister_class(GrowLeaves)
