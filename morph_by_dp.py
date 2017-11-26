import bpy
from mathutils.kdtree import KDTree
from mathutils import Vector

def my_handler(scene):
    from_name='tire'
    to_name='tire.001'
    tgt_name='tire.002'
        
    from_obj=scene.objects[from_name]
    to_obj=scene.objects[to_name]
    tgt_obj=scene.objects[tgt_name]
    
    from_mesh=from_obj.data
    to_mesh=to_obj.data
    tgt_mesh=tgt_obj.data
    
    dp = from_mesh.vertex_colors['dp_paintmap'].data
    
    for i in range(len(from_mesh.vertices)):
        t = dp[i].color.r
        co = from_mesh.vertices[i].co * (1 - t) + \
            to_mesh.vertices[i].co * (t)
        no = from_mesh.vertices[i].normal * (1 - t) + \
            to_mesh.vertices[i].normal * (t)
        tgt_mesh.vertices[i].co = co
        tgt_mesh.vertices[i].normal = no
    scene.update()

def register():
    bpy.app.handlers.frame_change_post.append(my_handler)
    
def unregister():
    bpy.app.handlers.frame_change_post.remove(my_handler)

register()
