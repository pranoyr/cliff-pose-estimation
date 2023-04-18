import torch
from pytorch3d.io import load_objs_as_meshes, load_obj
from .vertex_map import get_textures
from pytorch3d.renderer import (
	TexturesVertex
)


class FLAME:
    def __init__(self, cfg, device) -> None:
        self.flame_verts, faces_idx, aux = load_obj(cfg.DATA.FLAME_PATH)
        self.flame_faces = faces_idx.verts_idx
        self.device = device
        self.flame_verts = self.flame_verts.to(self.device) * cfg.SCALE
        self.flame_faces = self.flame_faces.to(self.device)
       
    def add_to_flame_face(self, pred_vertices, hair_faces):
        # combime FLAME and HAIR vertices
        full_faces = torch.cat((self.flame_faces, hair_faces + self.flame_verts.shape[0]), dim=0)
        full_vertices = torch.cat((self.flame_verts, pred_vertices), dim=0)
        full_textures = torch.ones_like(full_vertices).cuda() # (1, V, 3)
        # get flame indices
        flame_indices = torch.arange(self.flame_verts.shape[0])
        flame_indices = flame_indices.to(self.device)
        # get hair indices
        hair_indices = torch.arange(self.flame_verts.shape[0], self.flame_verts.shape[0] + pred_vertices.shape[0])
        hair_indices = hair_indices.to(self.device)
        # black for hair
        full_textures[hair_indices.long()] = torch.tensor([0, 0, 0], dtype= torch.float32).to(self.device)
        # skin colour for face
        # tex = get_textures(flame_path, "/home/pranoy/code/hairnet/assets/pranoy.png").cuda()/255
        # full_textures[flame_indices.long()] = tex

        textures = TexturesVertex(verts_features=full_textures[None])
        return full_vertices , full_faces, textures