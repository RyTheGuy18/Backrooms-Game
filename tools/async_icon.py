import numpy as np
from PIL import Image, ImageFilter
R='/home/claude/haunted-pines/'; S=1024; OFF=-14
im=Image.open(R+'menu/async.jpg').convert('RGB'); a=np.asarray(im).astype(float)
lum=a.mean(2); bgm=lum>90; bg=a[bgm].mean(0); bl=lum[bgm].mean()
al=np.clip((bl-lum)/(bl-15),0,1); al=np.clip((al-0.25)/0.6,0,1)
ys,xs=np.where(al>0.5); x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
al=al[max(0,y0-4):y1+5,max(0,x0-4):x1+5]
tw=int(S*0.8); th=int(al.shape[0]*tw/al.shape[1])
A=Image.fromarray((al*255).astype(np.uint8)).resize((tw,th),Image.LANCZOS)
c=Image.new('RGB',(S,S),tuple(int(v) for v in bg)); c.paste(Image.new('RGB',(tw,th),(6,6,12)),((S-tw)//2,(S-th)//2+OFF),A)
rng=np.random.default_rng(5)
base=np.asarray(c).astype(float)/255
def crop(p): return np.asarray(Image.open(p).convert('RGB').crop((120,80,420,280)).resize((S,S),Image.BILINEAR)).astype(float)/255
v=np.maximum(crop('vf.png'),crop('vf2.png'))
o=base.copy()
for y0_,hh,dx in [(300,18,26),(530,34,-34),(430,12,18)]:
    o[y0_:y0_+hh]=np.roll(o[y0_:y0_+hh],dx,axis=1)
o=1-(1-o)*(1-np.clip(v*1.3,0,1))
st=o[930:990].copy(); st=np.roll(st,40,axis=1)
st=st*0.6+np.clip(rng.normal(0.15,0.3,(60,S,3)),0,1)*0.6; o[930:990]=np.clip(st,0,1)
o[560:562]=np.clip(o[560:562]+0.45,0,1)
sp=rng.random((S,S))<0.0025; o[sp]=np.clip(o[sp]+0.7,0,1)
r=np.roll(o[...,0],10,axis=1); b=np.roll(o[...,2],-9,axis=1); o=np.stack([r,o[...,1],b],-1)
yy=np.arange(S)[:,None]; o*=(1-0.28*((yy//4)%2))[...,None]
xx=np.arange(S)[None,:]; vg=1-0.6*np.clip(((xx-512)**2+(yy-512)**2)/(600**2),0,1)**1.5; o*=vg[...,None]
o+=rng.normal(0,0.06,(S,S,1))
im=Image.fromarray((np.clip(o,0,1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.6))
im=im.resize((340,340),Image.BILINEAR).resize((S,S),Image.BILINEAR)
im.save('icon1024h.png')
for n,z in [('icon-512.png',512),('icon-192.png',192),('apple-touch-icon.png',180),('icon-maskable-512.png',512)]:
    im.resize((z,z),Image.LANCZOS).save(R+n,optimize=True)
ys,xs=np.where(np.asarray(A)>128); print('logo top/bottom', (S-th)//2+OFF, (S-th)//2+OFF+th)
