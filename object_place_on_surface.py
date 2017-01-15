bl_info = {
    "name": "Place On Surface",
    "description": "Place object on surface with its Z direction aligned with surface normal",
    "author": "Stanislaw Adaszewski",
    "version": (1, 0, 0),
    "blender": (2, 78, 0),
    "location": "Search > Place On Surface",
    "warning": "",
    "wiki_url": "http://algoholic.eu/place-object-on-surface/",
    "category": "Object",
}


import bpy
from mathutils import Quaternion, Matrix, Vector, Euler
from mathutils.bvhtree import BVHTree
from mathutils.interpolate import poly_3d_calc


def smooth_normal(obj, loc, index):
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


class PlaceObjectOnSurface(bpy.types.Operator):
    """Place object on surface with its Z direction aligned with surface normal"""
    bl_idname = "object.place_on_surface"
    bl_label = "Place On Surface"
    bl_options = {'REGISTER', 'UNDO'}
    
    align_with_normal = bpy.props.FloatProperty(
        name="Align with normal", min=0, max=1, default=1)
        
    use_smooth = bpy.props.BoolProperty(
        name="Use smooth normal", default=True)

    def execute(self, context):
        selected = bpy.context.selected_objects
        obj = selected[-1]
        surf = bpy.context.scene.objects['surface']
        
        loc = bpy.context.scene.cursor_location
        
        bvh = BVHTree.FromObject(surf, bpy.context.scene)
        
        loc = surf.matrix_world.inverted() * loc
        (loc, normal, index, dist) = bvh.find_nearest(loc)
        if self.use_smooth:
            normal = smooth_normal(surf, loc, index)
        loc = surf.matrix_world * loc
        
        bpy.ops.object.duplicate()
        new_obj = bpy.context.selected_objects[-1]
        
        (unused, surf_rot, unused) = surf.matrix_world.decompose()
        (unused, obj_rot, scale) = obj.matrix_world.decompose()
        
        normal = surf_rot * normal
        vec = obj_rot * Vector((0.0, 0.0, 1.0))
        
        q = vec.rotation_difference(normal)
        q = Quaternion().slerp(q, self.align_with_normal)
        mat_scale = Matrix()
        for i in range(3): mat_scale[i][i] = scale[i]
        new_obj.matrix_world = (Matrix.Translation(loc) *
            q.to_matrix().to_4x4() * obj_rot.to_matrix().to_4x4() *
            mat_scale)
        
        bpy.context.scene.objects.active = new_obj
        
        return {'FINISHED'}


def register():
    bpy.utils.register_class(PlaceObjectOnSurface)
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
        kmi = km.keymap_items.new('object.place_on_surface', 'P', 'PRESS', shift=False)


def unregister():
    bpy.utils.unregister_class(PlaceObjectOnSurface)
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc:
        km = kc.keymaps["3D View"]
        for kmi in km.keymap_items:
            if kmi.idname == 'object.place_on_surface':
                km.keymap_items.remove(kmi)
            break

