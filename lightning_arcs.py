bl_info = {
    "name": "Lightning Arcs",
    "description": "Use Laplacian Lightning Add-On to generate random arcs between particles",
    "author": "Stanislaw Adaszewski",
    "version": (1, 0, 0),
    "blender": (2, 74, 0),
    "location": "Search > Lightning Arcs",
    "warning": "",
    "wiki_url": "http://algoholic.eu/lightning-arcs/",
    "category": "Object",
}


import bpy
from mathutils import Vector, Quaternion
from math import sqrt
import random


def _arc_segment(v_1, v_2):
	ELorigin = bpy.context.scene.objects['ELorigin']
	ELground = bpy.context.scene.objects['ELground']

	v = v_2 - v_1
	d = v.length

	ELorigin.location = Vector((0, 0, 0))
	ELground.location = Vector((0, 0, -d))

	v_L = ELground.location - ELorigin.location

	q = Quaternion()
	c = Vector.cross(v_L, v)
	q.x = c.x
	q.y = c.y
	q.z = c.z
	q.w = sqrt((v_L.length ** 2) * (v.length ** 2)) + \
		Vector.dot(v_L, v)
	q.normalize()
	euler = q.to_euler()

	bpy.ops.object.runfslg_operator()

	laALL = bpy.context.scene.objects['laALL']
	laALL.name = 'lARC'
	laALL.rotation_euler = euler
	laALL.location = v_1

	bpy.context.active_object.select = False
	laALL.select = True
	bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
	laALL.select = False
	bpy.context.active_object.select = True	

	return laALL


def _arc_bezier(v_1, v_2):
	H = bpy.context.scene.objects['ARCheight'].location.z
	f_1 = Vector((v_1.x, v_1.y, 0))
	f_2 = Vector((v_2.x, v_2.y, 0))
	d = f_2 - f_1
	L = d.length
	d.normalize()
	c_1 = v_1 + (L / 3.) * d
	c_2 = v_2 - (L / 3.) * d
	c_1 += Vector((0, 0, H))
	c_2 += Vector((0, 0, H))
	n = 0
	prev_p = v_1
	for i in range(n):
		q = 1. * i / n
		p = (1 - q) ** 3 * v_1 + \
			3 * (1 - q) ** 2 * q * c_1 + \
			3 * (1 - q) * q ** 2 * c_2 + \
			q ** 3 * v_2
		_arc_segment(prev_p, p)
		prev_p = p
	_arc_segment(prev_p, v_2)
	

def _arc_bezier_2(v_1, v_2):
	laALL = _arc_segment(v_1, v_2)

	H = bpy.context.scene.objects['ARCheight'].location.z
	H = H / 2. + H / 2. * random.random()

	vertices = laALL.data.vertices
	n = len(vertices)
	for i in range(n):
		p_world = vertices[i].co

		q = (p_world - v_1).length / (v_2 - v_1).length
		h = 3 * (1 - q) ** 2 * q * H + \
			3 * (1 - q) * q ** 2 * H

		p_world.z += h
		
		# (result, location, normal, normal, index, object, matrix)
		hit = bpy.context.scene.ray_cast(origin=p_world, direction=Vector((0, 0, -1)))
		if hit[5] != bpy.context.active_object:
			hit = bpy.context.scene.ray_cast(origin=p_world, direction=Vector((0, 0, 1)))
		if hit[1].z > p_world.z: # bounce
			p_world.z = hit[1].z + (hit[1].z - p_world.z)
			

		vertices[i].co = p_world
		



class LightningArcs(bpy.types.Operator):
    """Use Laplacian Lightning Add-On to generate random arcs between particles"""
    bl_idname = "object.lightning_arcs"
    bl_label = "Lightning Arcs"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        psys = bpy.context.active_object.particle_systems[0]
        particles = psys.particles
        n = len(particles)
        if n % 2 == 1: n -= 1
        for i in range(0, n, 2):
            p_from = particles[i]
            p_to = particles[i + 1]
            _arc_bezier_2(p_from.location, p_to.location)        
        return {'FINISHED'}


def register():
    bpy.utils.register_class(LightningArcs)


def unregister():
    bpy.utils.unregister_class(LightningArcs)
