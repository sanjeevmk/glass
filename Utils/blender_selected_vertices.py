import bpy, bmesh
obj = bpy.context.active_object
import csv

outFile = "tgt1_lh.csv"

if obj.mode == 'EDIT':
    bm = bmesh.from_edit_mesh(obj.data)
    vertices = bm.verts

else:
    vertices = obj.data.vertices

verts = [obj.matrix_world * vert.co for vert in vertices if vert.select] 

# coordinates as tuples
plain_verts = [list(vert.to_tuple()) for vert in verts]

csv.writer(open(outFile,'w'),delimiter=' ').writerows(plain_verts)