import bpy
from mathutils.kdtree import KDTree
from mathutils import Vector

def my_handler(scene):
    from_name='tire'
    to_name='tire.001'
    tgt_name='tire.002'
    brush_name = 'Cube'
        
    from_obj=scene.objects[from_name]
    to_obj=scene.objects[to_name]
    tgt_obj=scene.objects[tgt_name]
    brush_obj=scene.objects[brush_name]
    
    from_mesh=from_obj.data
    to_mesh=to_obj.data
    tgt_mesh=tgt_obj.data
    brush_mesh=brush_obj.data
    
    tgt_kd = KDTree(len(tgt_mesh.vertices))
    for i in range(len(tgt_mesh.vertices)):
        co = tgt_obj.matrix_world * tgt_mesh.vertices[i].co
        tgt_kd.insert(co, i)
    tgt_kd.balance()
    # return tgt_kd
    
    min_x = float('inf')
    max_x = float('-inf') # brush_mesh.vertices[0].co.x
    for i in range(len(brush_mesh.vertices)):
        co = brush_mesh.vertices[i].co * brush_obj.matrix_world
        max_x = max(co.x, max_x)
        min_x = min(co.x, min_x)
    radius = (max_x - min_x) / 2
    # raise ValueError('min_x: %f, max_x: %f, radius: %f' % (min_x, max_x, radius))
    
    pts = tgt_kd.find_range(brush_obj.location, radius)
    # raise ValueError('len(pts): %d' % len(pts))
    for (co, idx, dist) in pts:
        tgt_obj.vertex_groups[0].add([idx], 1.0, 'REPLACE')
            
    for i in range(len(from_mesh.vertices)):    
        t = tgt_obj.vertex_groups[0].weight(i)
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
