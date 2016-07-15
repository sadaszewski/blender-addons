bl_info = {
    "name": "Union Loose Parts",
    "description": "Separate object based on loose parts and merge results using boolean union",
    "author": "Stanislaw Adaszewski",
    "version": (3, 1, 1),
    "blender": (2, 74, 0),
    "location": "Search > Union Loose Parts",
    "warning": "",
    "wiki_url": "http://algoholic.eu/union-loose-parts/",
    "category": "Object",
}


import bpy


class UnionLooseParts(bpy.types.Operator):
    """Separate object based on loose parts and merge results using boolean union"""
    bl_idname = "object.union_loose_parts"
    bl_label = "Union Loose Parts"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.mesh.separate(type='LOOSE')
        selected = bpy.context.selected_objects
        merged = selected[-1]
        for i in range(0, len(selected) - 1):
            modif = merged.modifiers.new('Modifier', 'BOOLEAN')
            modif.object = selected[i]
            modif.operation = 'UNION'
            bpy.context.scene.objects.active = merged
            bpy.ops.object.modifier_apply(apply_as='DATA', modifier=modif.name)
            bpy.context.scene.update()
            bpy.context.scene.objects.unlink(selected[i])
            print (selected[i])
        bpy.context.scene.objects.active = merged
        modif = merged.modifiers.new('Modifier', 'TRIANGULATE')
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier=modif.name)
        return {'FINISHED'}


def register():
    bpy.utils.register_class(UnionLooseParts)


def unregister():
    bpy.utils.unregister_class(UnionLooseParts)
