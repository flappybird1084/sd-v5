import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import CLIPProcessor, CLIPModel, CLIPTextModelWithProjection

class JointAttention(nn.Module):
  def __init__(self, d_model=256, image_tokens=256, prompt_tokens=1):
    super().__init__()
    self.dim = d_model
    
    self.image_q, self.image_v, self.image_k = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
    self.prompt_q, self.prompt_v, self.prompt_k = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
    
    self.mlp = nn.Sequential(
      nn.Linear(d_model, d_model*4),
      nn.ReLU(),
      nn.Linear(d_model*4, d_model)
    )
    
    half_dim = d_model //2
    freq = torch.exp(
      -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
    )
    self.register_buffer("timestep_freq", freq)     
    self.image_emb = nn.Embedding(image_tokens,self.dim)
    self.register_buffer("pos_ids", torch.arange(image_tokens))
    self.ln1 = nn.LayerNorm(self.dim)
    self.ln2 = nn.LayerNorm(self.dim)
    self.image_tokens = image_tokens
    # self.image_proj = nn.Linear(4*32*32//image_tokens,d_model)
    # self.prompt_proj = nn.Linear(512,d_model)
    # self.image_up = nn.Linear(d_model, 4*32*32//image_tokens)
    # self.prompt_up = nn.Linear(d_model, 512)
    self.prompt_tokens = prompt_tokens

  def timestep_embedding(self,t):
    freqs = self.timestep_freq.unsqueeze(0)
    # emb = t.unsqueeze(1) * freqs
    emb = t * freqs * 1000
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb 
  
  def forward(self, image, prompt, timestep):
    # image shape: b,256,512
    # prompt shape: b,1,512
    timestep_emb = self.timestep_embedding(timestep) 
    # timestep emb shape: b,1,512 (dim -1 follows d_model, no proj. required)
    # image = self.image_proj(image)
    # prompt = self.prompt_proj(prompt)
    # image: b,256,dim
    # propmt: b,1,dim

    # print(f"sanity: image: {image.shape}, prompt: {prompt.shape}, timestep_emb: {timestep_emb.shape}")
    image = image+timestep_emb 
    image = image+self.image_emb(self.pos_ids)
    prompt = prompt+timestep_emb 

    # print(f"sanity: image: {image.shape}, prompt: {prompt.shape}, timestep_emb: {timestep_emb.shape}")

    image_norm, prompt_norm = self.ln1(image), self.ln1(prompt)
    image_q, image_k, image_v = self.image_q(image_norm), self.image_k(image_norm), self.image_v(image_norm) 
    prompt_q, prompt_k, prompt_v = self.prompt_q(prompt_norm), self.prompt_k(prompt_norm), self.prompt_v(prompt_norm) 
    
    Q = torch.cat([image_q, prompt_q], dim=1) # B,257, dim
    K = torch.cat([image_k, prompt_k], dim=1) # ""
    V = torch.cat([image_v, prompt_v], dim=1) # ""
    
    out_resid = torch.cat([image, prompt], dim=1) # b,257,dim 
    attn_out = F.scaled_dot_product_attention(Q, K, V) 
    out = self.mlp(attn_out) + out_resid 
    out = self.ln2(out) 
    
    out_image, out_prompt = out[:,:self.image_tokens,:], out[:,self.image_tokens:,:] 
    # out_image = self.image_up(out_image)
    # out_prompt = self.prompt_up(out_prompt)
    return out_image, out_prompt
  
class DiffusionModel(nn.Module):
  def __init__(self, dim=256, num_layers=4, image_tokens=256, prompt_tokens=1):
    super(DiffusionModel, self).__init__()
    self.attn_layers = nn.ModuleList([JointAttention(dim, image_tokens, prompt_tokens) for _ in range(num_layers)])
    self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model_id="openai/clip-vit-base-patch32"
    self.clip_model = CLIPTextModelWithProjection.from_pretrained(model_id)

    self.image_proj = nn.Linear(4*32*32//image_tokens,dim)
    self.prompt_proj = nn.Linear(512,dim)
    self.image_up = nn.Linear(dim, 4*32*32//image_tokens)
    self.prompt_up = nn.Linear(dim, 512)

  def embed_text(self, text):
    # inputs = tokenizer([text], return_tensors="pt")
    inputs = self.processor(text=[text], return_tensors="pt", padding=True)
    # print(type(inputs))
    with torch.inference_mode():
      # outputs = clip_model(**inputs)    
      outputs = self.clip_model(**inputs)
    embeds=outputs.text_embeds.unsqueeze(1) # B,1,512
    # chunks = torch.chunk(embeds, chunks=2, dim=2)# tuple: B,1,256 each
    # embeds = torch.cat(chunks, dim=1) # B,2,256
    return embeds
    
  def forward(self, image_enc, prompt_enc, timestep):
    image_enc = self.image_proj(image_enc)
    prompt_enc = self.prompt_proj(prompt_enc)
    resid_image_enc, resid_prompt_enc = image_enc, prompt_enc
    for layer in self.attn_layers:
      image_enc, prompt_enc = layer(image_enc, prompt_enc, timestep)
      # print(f"sanity shapes: {image_enc.shape}, {prompt_enc.shape}, {resid_image_enc.shape}, {resid_prompt_enc.shape}")
      image_enc, prompt_enc = image_enc+resid_image_enc, prompt_enc+resid_prompt_enc
      resid_image_enc, resid_prompt_enc = image_enc, prompt_enc
    # return image_enc, prompt_enc
    image_enc = self.image_up(image_enc)
    return image_enc # velocity pred
    
if __name__ == "__main__":
  model = DiffusionModel()
  image = torch.randn(1,3,256,256)
  text = "a cat on a skateboard"
  image_emb = model.encode(image)
  prompt_emb = model.embed_text(text)
  # print(f"image_emb shape: {image_emb.shape}, prompt_emb shape: {prompt_emb.shape}")
  with torch.no_grad():
    out_image_emb= model(image_emb, prompt_emb, timestep=torch.tensor([10.0]))
  # print(f"out_image_emb shape: {out_image_emb.shape}, out_prompt_emb shape: {out_prompt_emb.shape}")
    out_image = model.decode(out_image_emb)
  print(out_image.shape)