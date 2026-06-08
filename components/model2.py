import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import CLIPProcessor, CLIPModel, CLIPTextModelWithProjection

class Encoder(nn.Module): # accepts B,3,256,256
  def __init__(self):
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1) # B,64,128,128
    self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # B,128,64,64
    self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # B,256,32,32
    self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) # B,512,16,16

    # self.norm = nn.LayerNorm([512, 16*16])
    self.norm = nn.LayerNorm([256,512])

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = (self.conv3(x))
    x = (self.conv4(x))
    x = x.view(x.size(0), 512, -1) # B,512,256
    x = x.permute(0,2,1) # B,512,16,16 -> B,256,512
    #why? 512 stacks -> each pixel in stack has its own patch position. thus, those pixels should be in the T dim of transformer
    x = self.norm(x)
    return x 

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # B,256,32,32
    self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # B,128,64,64
    self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # B,64,128,128
    self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1) # B,3,256,256

  def forward(self, x): # x start: B,256,512
    x = x.permute(0,2,1) # B,512,256
    x = x.view(x.size(0), 512, 16, 16) # B,512,256 -> B,512,16,16
    x = F.relu(self.deconv1(x))
    x = F.relu(self.deconv2(x))
    x = F.relu(self.deconv3(x))
    x = torch.sigmoid(self.deconv4(x)) # Use sigmoid to get output in range [0, 1]
    return x
  
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
    self.image_proj = nn.Linear(512,d_model)
    self.prompt_proj = nn.Linear(512,d_model)
    self.image_up = nn.Linear(d_model, 512)
    self.prompt_up = nn.Linear(d_model, 512)
    self.image_tokens = image_tokens
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
    image = self.image_proj(image)
    prompt = self.prompt_proj(prompt)
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
    out_image = self.image_up(out_image)
    out_prompt = self.prompt_up(out_prompt)
    return out_image, out_prompt
  
class DiffusionModel(nn.Module):
  def __init__(self, dim=256, num_layers=4):
    super(DiffusionModel, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.attn_layers = nn.ModuleList([JointAttention(dim) for _ in range(num_layers)])
    self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model_id="openai/clip-vit-base-patch32"
    self.clip_model = CLIPTextModelWithProjection.from_pretrained(model_id)

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
    
  def encode(self, image):
    return self.encoder(image)
  
  def decode(self, image_emb):
    return self.decoder(image_emb)
  
  def forward(self, image_enc, prompt_enc, timestep):
    resid_image_enc, resid_prompt_enc = image_enc, prompt_enc
    for layer in self.attn_layers:
      image_enc, prompt_enc = layer(image_enc, prompt_enc, timestep)
      # print(f"sanity shapes: {image_enc.shape}, {prompt_enc.shape}, {resid_image_enc.shape}, {resid_prompt_enc.shape}")
      image_enc, prompt_enc = image_enc+resid_image_enc, prompt_enc+resid_prompt_enc
      resid_image_enc, resid_prompt_enc = image_enc, prompt_enc
    # return image_enc, prompt_enc
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