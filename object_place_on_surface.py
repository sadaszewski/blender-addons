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


# def qexp(q, t):
#
#
#def slerp(q0, q1, t):
#    q0_inv = q0.copy()
#    q0_inv.inverse()
#    (q0 * (q0_inv * q1)


class PlaceObjectOnSurface(bpy.types.Operator):
    """Place object on surface with its Z direction aligned with surface normal"""
    bl_idname = "object.place_on_surface"
    bl_label = "Place On Surface"
    bl_options = {'REGISTER', 'UNDO'}
    
    align_with_normal = bpy.props.FloatProperty(
        name="Align with Normal", min=0, max=1, default=1)
    
    #def invoke(self, context, event):
    #   wm = context.window_manager
    #   return wm.invoke_props_dialog(self)

    def execute(self, context):
        # self.report({'INFO'}, str(self.align_with_normal))
        selected = bpy.context.selected_objects
        obj = selected[-1]
        surf = bpy.context.scene.objects['surface'] # selected[-2]
        # self.report({'INFO'}, str(surf))
        
        loc = bpy.context.scene.cursor_location
        # self.report({'INFO'}, str(loc))
        # return {'FINISHED'}
        
        bvh = BVHTree.FromObject(surf, bpy.context.scene)
        
        loc = surf.matrix_world.inverted() * loc
        (loc, normal, index, dist) = bvh.find_nearest(loc)
        loc = surf.matrix_world * loc
        
        # surf.select = False
        bpy.ops.object.duplicate()
        new_obj = bpy.context.selected_objects[-1]
        # new_obj.select = False
        # surf.select = True
        # obj.select = True
            
        new_obj.location = loc
        
        # self.report({'INFO'}, str(normal))
        
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
        # bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        
        
        
        
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

