#
# Copyright (C) Stanislaw Adaszewski, 2017
#

import bpy
from mathutils.kdtree import KDTree
from mathutils import Vector

def my_handler(scene):
    from_name='tire'
    to_name='tire.001'
    tgt_name='tire.002'
    n=1024
    start_frame = 1
    end_frame = 250
    duration = 1
    frame_current = scene.frame_current
    
    from_obj=scene.objects[from_name]
    to_obj=scene.objects[to_name]
    tgt_obj=scene.objects[tgt_name]
    
    from_mesh=from_obj.data
    to_mesh=to_obj.data
    tgt_mesh=tgt_obj.data
    uvs=from_obj.data.uv_layers[0].data
    for i in range(len(from_mesh.vertices)):
        (u,v) = uvs[i].uv
        linear = u * 250 # round(v * n**2)/(n**2) + round(u*n)/(n**2)
        # linear = start_frame + (end_frame - start_frame) * linear
        if frame_current < linear:
            t = 0
        elif frame_current >= linear + duration:
            t = 1
        else:
            t = (frame_current - linear) / (duration)
        # linear = v + u / n
        # t = 1
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
