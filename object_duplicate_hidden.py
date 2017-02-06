bl_info = {
    "name": "Duplicate Hidden",
    "description": "Duplicate hierarchy including hidden objects",
    "author": "Stanislaw Adaszewski",
    "version": (1, 0, 0),
    "blender": (2, 78, 0),
    "location": "Search > Duplicate Hidden",
    "warning": "",
    "wiki_url": "http://algoholic.eu/blender-duplicate-hidden/",
    "category": "Object",
}


import bpy
import numpy as np
    

class DuplicateHidden(bpy.types.Operator):
    """Duplicate hierarchy including hidden objets"""
    bl_idname = "object.duplicate_hidden"
    bl_label = "Duplicate Hidden"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = bpy.context.active_object

        Q = [obj]
        saved_hide = []
        while len(Q) > 0:
            o = Q.pop(0)
            print('o.name:', o.name)
            saved_hide.append(o.hide)
            o.hide = False
            o.select = True
            for ch in o.children:
                Q.append(ch)

        print('saved_hide:', saved_hide)
        bpy.ops.object.duplicate()

        Q = [obj]
        cnt = 0
        while len(Q) > 0:
            o = Q.pop(0)
            o.hide = saved_hide[cnt]
            cnt += 1
            for ch in o.children:
                Q.append(ch)

        obj = bpy.context.active_object
        Q = [obj]
        cnt = 0
        while len(Q) > 0:
            o = Q.pop(0)
            o.hide = saved_hide[cnt]
            cnt += 1
            for ch in o.children:
                Q.append(ch)
                    
        return {'FINISHED'}


def register():
    bpy.utils.register_class(DuplicateHidden)


def unregister():
    bpy.utils.unregister_class(DuplicateHidden)
