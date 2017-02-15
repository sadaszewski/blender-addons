bl_info = {
    "name": "Reaction Diffusion",
    "description": "Perform two-component reaction diffusion in 3D",
    "author": "Stanislaw Adaszewski",
    "version": (1, 0, 0),
    "blender": (2, 78, 0),
    "location": "Search > Reaction Diffusion",
    "warning": "",
    "wiki_url": "http://algoholic.eu/blender-reaction-diffusio/",
    "category": "Object",
}


import bpy
from mathutils import Quaternion, Matrix, Vector, Euler
from mathutils.bvhtree import BVHTree
import numpy as np
# from marching_cube import isosurface
from scipy.interpolate import RegularGridInterpolator


'''_params = {
    'Bacteria 1': (0.16, 0.08, 0.035, 0.065), # Bacteria 1
    'Bacteria 2': (0.14, 0.06, 0.035, 0.065), # Bacteria 2
    'Coral': (0.16, 0.08, 0.060, 0.062), # Coral
    'Fingerprint': (0.19, 0.05, 0.060, 0.062), # Fingerprint
    'Spirals': (0.10, 0.10, 0.018, 0.050), # Spirals
    'Spirals Dense': (0.12, 0.08, 0.020, 0.050), # Spirals Dense
    'Spirals Fast': (0.10, 0.16, 0.020, 0.050), # Spirals Fast
    'Unstable': (0.16, 0.08, 0.020, 0.055), # Unstable
    'Worms 1': (0.16, 0.08, 0.050, 0.065), # Worms 1
    'Worms 2': (0.16, 0.08, 0.054, 0.063), # Worms 2
    'Zebrafish': (0.16, 0.08, 0.035, 0.060) # Zebrafish
}'''

_params = {
    'Default': (0.037,  0.06), # Bacteria 1
    'Solitons': (0.03, 0.062), # Bacteria 2
    'Pulsating solitons': (0.025, 0.06), # Coral
    'Worms': (0.078, 0.061), # Fingerprint
    'Mazes': (0.029, 0.057), # Spirals
    'Holes': (0.039, 0.058), # Spirals Dense
    'Chaos': (0.026, 0.051), # Spirals Fast
    'Chaos and holes': ( 0.034, 0.056), # Unstable
    'Moving spots': (0.014, 0.054), # Worms 1
    'Spots and loops': (0.018, 0.051), # Worms 2
    'Waves': (0.014, 0.045), # Zebrafish
    'U-Skate World': (0.062, 0.06093) # Zebrafish
}


def ary_to_pix_rg(r, g):
    (w, h) = (r.shape[1], r.shape[0])
    return np.ravel(np.vstack([np.ravel(r), np.ravel(g),
        np.zeros(w * h), np.ones(w * h)]).T)

def im_to_array(im):
    ary = np.array(im.pixels)
    ary = np.reshape(ary, (im.size[1], im.size[0], 4))
    return ary
    
    
def array_to_im(ary, im):
    im.pixels = np.ravel(ary)


def new_f32_image(name, w, h):
    im = bpy.data.images.new(name, width=w, height=h, alpha=False, float_buffer=True)
    im.use_alpha = False
    return im
    
    
def clear_f32_image(im, val):
    print('clear_f32_image(), val:', val)

    (w, h) = (im.size[0], im.size[1])
    fill_val = np.ones((h, w)) * val

    im.pixels =np.ravel(np.vstack([np.ravel(fill_val), np.ravel(fill_val),
        np.ravel(fill_val), np.ones(w * h)]).T)
    
    
def clear_f32_image_rg(im, val_r, val_g):
    (w, h) = (im.size[0], im.size[1])
    
    fill_val_r = np.ones((h, w)) * val_r
    fill_val_g = np.ones((h, w)) * val_g

    im.pixels = np.ravel(np.vstack([np.ravel(fill_val_r), np.ravel(fill_val_g),
        np.zeros(w * h), np.ones(w * h)]).T)


def save_image(img, fname):
    im = bpy.data.images.new('RD_Temp_Image', width=img.shape[1], height=img.shape[0], alpha=False)
    im.use_alpha = False
    n = img.shape[0] * img.shape[1]
    pixels = np.vstack([np.ravel(img)] * 3 + [np.ones(n)]).T
    im.pixels = np.ravel(pixels)
    im.filepath_raw = fname
    im.file_format = 'PNG'
    im.save()
    bpy.data.images.remove(im)
    

def _save_image_rg(img_A, img_B, fname):
    img_A = _im_autorange(np.squeeze(img_A))
    img_B = _im_autorange(np.squeeze(img_B))

    #min_ = min(np.min(np.ravel(img_A)[:]), np.min(np.ravel(img_B)[:]))
    #max_ = max(np.max(np.ravel(img_A)[:]), np.max(np.ravel(img_B)[:]))

    #print('max_:', max_, 'min_:', min_)
    #img_A = (img_A ) / (max_ - min_)
    #img_B = (img_B ) / (max_ - min_)
    
    im = bpy.data.images.new('RD_Temp_Image',
        width=img_A.shape[1], height=img_B.shape[0],
        alpha=False)
    im.use_alpha = False
    n = img_A.shape[0] * img_B.shape[1]
    pixels = np.vstack([np.ravel(img_A), np.ravel(img_B),
        np.zeros(n), np.ones(n)]).T
    im.pixels = np.ravel(pixels)
    im.filepath_raw = fname
    im.file_format = 'PNG'
    im.save()
    bpy.data.images.remove(im)


def _diff_x(A):
    ret = np.zeros(A.shape)
    ret[1:-1, :, :] = (A[2:, :, :] - A[0:-2, :, :]) / 2.
    ret[0, :, :] = (A[1, :, :] - A[0, :, :])
    ret[-1, :, :] = (A[-1, :, :] - A[-2, :, :])
    return ret


def _diff_y(A):
    ret = np.zeros(A.shape)
    ret[:, 1:-1, :] = (A[:, 2:, :] - A[:, 0:-2, :]) / 2.
    ret[:, 0, :] = (A[:, 1, :] - A[:, 0, :])
    ret[:, -1, :] = (A[:, -1, :] - A[:, -2, :])
    return ret


def _diff_z(A):
    ret = np.zeros(A.shape)
    ret[:, :, 1:-1] = (A[:, :, 2:] - A[:, :, 0:-2]) / 2.
    ret[:, :, 0] = (A[:, :, 1] - A[:, :, 0])
    ret[:, :, -1] = (A[:, :, -1] - A[:, :, -2])
    return ret


def _d2_x(A):
    ret = np.zeros(A.shape)
    ret[1:-1, :, :] = A[2:, :, :] - 2 * A[1:-1, :, :] + A[:-2, :, :]
    ret[0, :, :] = A[2, :, :] - 2 * A[1, :, :] + A[0, :, :]
    ret[-1, :, :] = A[-1, :, :] - 2 * A[-2, :, :] + A[-3, :, :]
    return ret


def _d2_y(A):
    ret = np.zeros(A.shape)
    ret[:, 1:-1, :] = A[:, 2:, :] - 2 * A[:, 1:-1, :] + A[:, :-2, :]
    ret[:, 0, :] = A[:, 2, :] - 2 * A[:, 1, :] + A[:, 0, :]
    ret[:, -1, :] = A[:, -1, :] - 2 * A[:, -2, :] + A[:, -3, :]
    return ret


def _d2_z(A):
    ret = np.zeros(A.shape)
    ret[:, :, 1:-1] = A[:, :, 2:] - 2 * A[:, :, 1:-1] + A[:, :, :-2]
    ret[:, :, 0] = A[:, :, 2] - 2 * A[:, :, 1] + A[:, :, 0]
    ret[:, :, -1] = A[:, :, -1] - 2 * A[:, :, -2] + A[:, :, -3]
    return ret


def _d2_x_2d(A):
    ret = np.zeros(A.shape)
    ret[1:-1, :] = A[2:, :] - 2 * A[1:-1, :] + A[:-2, :]
    ret[0, :] = A[2, :] - 2 * A[1, :] + A[0, :]
    ret[-1, :] = A[-1, :] - 2 * A[-2, :] + A[-3, :]
    return ret


def _d2_y_2d(A):
    ret = np.zeros(A.shape)
    ret[:, 1:-1] = A[:, 2:] - 2 * A[:, 1:-1] + A[:, :-2]
    ret[:, 0] = A[:, 2] - 2 * A[:, 1] + A[:, 0]
    ret[:, -1] = A[:, -1] - 2 * A[:, -2] + A[:, -3]
    return ret


def _laplacian(A):
    return (_d2_x(A) + _d2_y(A) + _d2_z(A))


def _laplacian_2d(A):
    return (_d2_x_2d(A) + _d2_y_2d(A))


def _rd(A, B, D_A, D_B, f, k, dt=1.):
    A_prime = A + (D_A * _laplacian(A) - A * (B ** 2) + \
        f * (1 - A)) * dt
    B_prime = B + (D_B * _laplacian(B) + A * (B ** 2) - \
        (k + f) * B) * dt
    return (A_prime, B_prime)


def _rd_2d(A, B, D_A, D_B, f, k, dt=1.):
    A_prime = A + (D_A * _laplacian_2d(A) - A * (B ** 2) + \
        f * (1 - A)) * dt
    B_prime = B + (D_B * _laplacian_2d(B) + A * (B ** 2) - \
        (k + f) * B) * dt
    return (A_prime, B_prime)




def _im_autorange(im):
    a = np.max(im[:])
    b = np.min(im[:])
    return ((im - b) / (a - b))
    
    
def _preset_update(self, context):
    k = bpy.types.Scene.RD_Presets[1]['items'][bpy.context.scene['RD_Presets']][0]
    print('_preset_update(), k:', k)
    if k == 'Manual':
        return
    (bpy.context.scene['RD_Diffuse_A'],
        bpy.context.scene['RD_Diffuse_B']) = (0.2097, 0.105)
        
    (bpy.context.scene['RD_Feed_Rate'],
        bpy.context.scene['RD_Kill_Rate']) = _params[k]
    
    
def initSceneProperties():
    bpy.types.Scene.RD_Presets = bpy.props.EnumProperty(items=[('Manual', 'Manual', 'Manual')] +
        list(map(lambda k: (k, k, k),
        _params.keys())), name='Presets', description='Presets',
        update=_preset_update)
        
    bpy.types.Scene.RD_Diffuse_A = bpy.props.FloatProperty(
        name="Diffuse A", min=0, max=1, default=1.0)
        
    bpy.types.Scene.RD_Diffuse_B = bpy.props.FloatProperty(
        name="Diffuse B", min=0, max=1, default=.5)
        
    bpy.types.Scene.RD_Feed_Rate = bpy.props.FloatProperty(
        name="Feed Rate", min=0, max=1, default=.0367)

    bpy.types.Scene.RD_Kill_Rate = bpy.props.FloatProperty(
        name="Kill Rate", min=0, max=1, default=.0649)

    bpy.types.Scene.RD_Time_Step = bpy.props.FloatProperty(
        name="Time Step", min=0, max=1, default=.2)

    bpy.types.Scene.RD_Num_Steps = bpy.props.IntProperty(
        name="Number of Steps", min=1, default=5000)
        
    bpy.types.Scene.RD_Steps_Per_Redraw = bpy.props.IntProperty(
        name="Steps Per Redraw", min=1, default=100)

    bpy.types.Scene.RD_Iso_Level = bpy.props.FloatProperty(
        name="Isolevel", min=0, max=1, default=.5)
        
    bpy.types.Scene.RD_Diffuse_A_Map = bpy.props.StringProperty(
        name="Diffuse A Map", default='diffuse_a')
        
    bpy.types.Scene.RD_Diffuse_B_Map = bpy.props.StringProperty(
        name="Diffuse B Map", default='diffuse_b')
        
    bpy.types.Scene.RD_Feed_Map = bpy.props.StringProperty(
        name="Feed Map", default='feed_rate')
        
    bpy.types.Scene.RD_Kill_Map = bpy.props.StringProperty(
        name="Kill Map", default='kill_rate')

    bpy.types.Scene.RD_Input = bpy.props.StringProperty(
        name="Input", default='rd_input')    
    
    bpy.types.Scene.RD_Output = bpy.props.StringProperty(
        name="Output", default='rd_output')


def _fetchSceneProperties():
    for prop in ['RD_Diffuse_A', 'RD_Diffuse_B', 'RD_Feed_Rate',
        'RD_Kill_Rate', 'RD_Time_Step', 'RD_Num_Steps', 'RD_Steps_Per_Redraw',
        'RD_Iso_Level', 'RD_Diffuse_A_Map', 'RD_Diffuse_B_Map', 'RD_Feed_Map',
        'RD_Kill_Map', 'RD_Input', 'RD_Output']:
        try:
            val = bpy.context.scene[prop]
        except:
            bpy.context.scene[prop] = getattr(bpy.types.Scene, prop)[1]['default']
    
    
class RD_Run_Button(bpy.types.Operator):
    bl_idname = "rd.run"
    bl_label = "Run"
 
    def execute(self, context):
        return{'FINISHED'}
    
    
class RD_Panel(bpy.types.Panel):
    bl_label = 'Reaction Diffusion'
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'TOOLS'
    
    def draw(self, context):
        layout = self.layout
        scn = context.scene
        
        layout.prop(scn, 'RD_Presets')
        layout.prop(scn, 'RD_Diffuse_A')
        layout.prop(scn, 'RD_Diffuse_B')
        layout.prop(scn, 'RD_Feed_Rate')
        layout.prop(scn, 'RD_Kill_Rate')
        layout.prop(scn, 'RD_Time_Step')
        layout.prop(scn, 'RD_Num_Steps')
        layout.prop(scn, 'RD_Steps_Per_Redraw')
        layout.prop(scn, 'RD_Iso_Level')
        layout.prop(scn, 'RD_Diffuse_A_Map')
        layout.prop(scn, 'RD_Diffuse_B_Map')
        layout.prop(scn, 'RD_Feed_Map')
        layout.prop(scn, 'RD_Kill_Map')
        layout.prop(scn, 'RD_Input')
        layout.prop(scn, 'RD_Output')
        layout.operator('rd.rd_modal_timer_btn')
        
        
class RD_ModalTimerOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "rd.rd_modal_timer_btn"
    bl_label = "Run"

    _timer = None

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'} or \
            self._cnt >= bpy.context.scene['RD_Num_Steps']:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            print(self._cnt)
            
            steps_per_redraw = bpy.context.scene['RD_Steps_Per_Redraw']
            
            A = self._A
            B = self._B
            D_A = self._diffuse_a
            D_B = self._diffuse_b
            f = self._feed_rate
            k = self._kill_rate
            scn = bpy.context.scene
            dt = scn['RD_Time_Step']
            print('dt:', dt)
            for i in range(steps_per_redraw):
                (A, B) = _rd_2d(A, B, D_A, D_B, f, k, dt=dt)
                self._cnt += 1
                _save_image_rg(_im_autorange(A), _im_autorange(B), '/tmp/RD_%d.png' % self._cnt)
                
            self._A = A
            self._B = B
                
            self._output_im.pixels = ary_to_pix_rg(A, B)
            bpy.context.area.tag_redraw()
            
            # self._cnt += steps_per_redraw
            
            
            

        return {'PASS_THROUGH'}

    def execute(self, context):
        _fetchSceneProperties()
        self._cnt = 0
        
        scn = bpy.context.scene
        
        input_name = scn['RD_Input']
        if input_name in bpy.data.images:
            print(input_name, 'image exists')
            self._input_im = bpy.data.images[input_name]
            (w, h) = self._input_im.size
        else:
            w = h = 128
            self._input_im = new_f32_image(output_name, w, h)
            clear_f32_image_rg(self._input_im, 0.0, 1.0)
        
        output_name = scn['RD_Output']
        if output_name in bpy.data.images:
            print(output_name, 'image exists')
            # bpy.data.images[output_name].user_clear()
            bpy.data.images.remove(bpy.data.images[output_name])
        self._output_im = new_f32_image(output_name, w, h)
        
        diffuse_a_map = scn['RD_Diffuse_A_Map']
        if diffuse_a_map in bpy.data.images:
            print(diffuse_a_map, 'image exists')
            self._diffuse_a_im = bpy.data.images[diffuse_a_map]
        else:
            self._diffuse_a_im = new_f32_image(diffuse_a_map, w, h)
            clear_f32_image(self._diffuse_a_im, scn['RD_Diffuse_A'])
            
        diffuse_b_map = scn['RD_Diffuse_B_Map']
        if diffuse_b_map in bpy.data.images:
            print(diffuse_b_map, 'image exists')
            self._diffuse_b_im = bpy.data.images[diffuse_b_map]
        else:
            self._diffuse_b_im = new_f32_image(diffuse_b_map, w, h)
            clear_f32_image(self._diffuse_b_im, scn['RD_Diffuse_B'])
            
        feed_map = scn['RD_Feed_Map']
        if feed_map in bpy.data.images:
            print(feed_map, 'image exists')
            self._feed_im = bpy.data.images[feed_map]
        else:
            self._feed_im = new_f32_image(feed_map, w, h)
            clear_f32_image(self._feed_im, scn['RD_Feed_Rate'])
            
        kill_map = scn['RD_Kill_Map']
        if kill_map in bpy.data.images:
            print(kill_map, 'image exists')
            self._kill_im = bpy.data.images[kill_map]
        else:
            self._kill_im = new_f32_image(kill_map, w, h)
            clear_f32_image(self._kill_im, scn['RD_Kill_Rate'])
            
        self._diffuse_a = np.squeeze(im_to_array(self._diffuse_a_im)[:, :, 0])
        self._diffuse_b = np.squeeze(im_to_array(self._diffuse_b_im)[:, :, 0])
        self._feed_rate = np.squeeze(im_to_array(self._feed_im)[:, :, 0])
        self._kill_rate = np.squeeze(im_to_array(self._kill_im)[:, :, 0])
        self._A = np.squeeze(im_to_array(self._input_im)[:, :, 0])
        self._B = np.squeeze(im_to_array(self._input_im)[:, :, 1])
        
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, context.window)
        
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)


class ReactionDiffusion(bpy.types.Operator):
    """Perform two-component reaction diffusion in 3D"""
    bl_idname = "object.reaction_diffusion"
    bl_label = "ReactionDiffusion"
    bl_options = {'REGISTER', 'UNDO'}

    presets = bpy.props.EnumProperty(items=[('Manual', 'Manual', 'Manual')] +
        list(map(lambda k: (k, k, k),
        sorted(_params.keys()))), name='Presets', description='Presets')
    
    diffuse_a = bpy.props.FloatProperty(
        name="Diffuse A", min=0, max=1, default=1.0)

    diffuse_b = bpy.props.FloatProperty(
        name="Diffuse B", min=0, max=1, default=.5)

    feed_rate = bpy.props.FloatProperty(
        name="Feed Rate", min=0, max=1, default=.0367)

    kill_rate = bpy.props.FloatProperty(
        name="Kill Rate", min=0, max=1, default=.0649)

    time_step = bpy.props.FloatProperty(
        name="Time Step", min=0, max=1, default=.2)

    nsteps = bpy.props.IntProperty(
        name="Number of Steps", min=1, default=5000)

    isolevel = bpy.props.FloatProperty(
        name="Isolevel", min=0, max=1, default=.5)

    def execute(self, context):
        # self.report({'INFO'}, str(self.align_with_normal))
        selected = bpy.context.selected_objects

        res = 100
        A = np.ones((res,) * 2)
        B = np.zeros((res,) * 2)
        # A[25, 25, 25] = 1.0
        #for i in range(10):
            #xyz = np.random.randint(res, size=(2, 3))
            #xyz_start = np.min(xyz, axis=0)
            #xyz_end = np.max(xyz, axis=0)
            #B[xyz_start[0]:xyz_end[0]+1, xyz_start[1]:xyz_end[1]+1, xyz_start[2]:xyz_end[2]+1] = 1.0

        B[45:65, 45:55] = 1.0
        #B[49:52, 49:52] = 0.0
        # A = 1.0 - B
        # B[40:45, 35:45, 25] = 1.0

        if self.presets == 'Manual':
            diffuse_a = self.diffuse_a
            diffuse_b = self.diffuse_b
            feed_rate = self.feed_rate
            kill_rate = self.kill_rate
        else:
            (diffuse_a, diffuse_b, feed_rate, kill_rate) = \
                _params[self.presets]
                
        print('diffuse_a:', diffuse_a,
            'diffuse_b:', diffuse_b,
            'feed:', feed_rate,
            'kill:', kill_rate,
            'time_step:', self.time_step)

        big_B = np.zeros((res, res, res))

        for i in range(self.nsteps):
            print('Step:', i)
            (A, B) = _rd_2d(A, B, diffuse_a, diffuse_b,
                feed_rate, kill_rate, dt=self.time_step)

            big_B[:, :, i * (res - 1) // (self.nsteps - 1)] = B
           
            _save_image_rg(A, B, '/tmp/A_%03d.png' % i)


        (xx, yy, zz) = np.where(np.ones(big_B.shape))
        points = (list(range(res)),)*3
        values = big_B
        interp = RegularGridInterpolator(points, values, 'linear')

        def isofunc(pos):
            # print('isofunc(), pos:', pos)
            #if pos[0] < 50:
             #   return 0
            #else:
             #   return 1.
            val = interp([[pos[0], pos[1], pos[2]]])[0]
            if val is None:
                return 0.
            return val


        p0 = (0, 0, 0)
        p1 = (res, res, res-1)
            
        isosurface(p0, p1, (res, res, res), self.isolevel, big_B)
        
        return {'FINISHED'}



def register():
    initSceneProperties()
    bpy.utils.register_class(RD_ModalTimerOperator)
    bpy.utils.register_class(RD_Run_Button)
    bpy.utils.register_class(RD_Panel)


def unregister():
    bpy.utils.unregister_class(RD_Panel)
    bpy.utils.unregister_class(RD_Run_Button)
    bpy.utils.unregister_class(RD_ModalTimerOperator)
