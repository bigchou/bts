from glob import glob
import matplotlib as mpl
import numpy as np
import cv2, os, pdb
import matplotlib.cm as cm
import PIL.Image as pil
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


# determine rgb size
size = (1216,352*2)
print('size:',size)



class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



# bts_size=512,
# checkpoint_path='./models/bts_eigen_v2_pytorch_mbv2/model-50000-best_d3_0.99849',
# data_path='../../dataset/kitti_dataset/',
#dataset='kitti',
#do_kb_crop=True,
#encoder='mobilenetv2_bts',
#filenames_file='../train_test_inputs/eigen_test_files_with_gt.txt',
#input_height=352,
#input_width=1216,
#max_depth=80.0, mode='test',
#model_name='bts_eigen_v2_pytorch_mbv2', save_lpg=False

params = {
    'bts_size':512,
    'checkpoint_path':'./models/bts_eigen_v2_pytorch_res18/model-120000-best_abs_rel_0.06316',
    'data_path':'../../dataset/kitti_dataset/',
    'dataset':'kitti',
    'encoder':'resnet18_bts',
    'filenames_file': '../train_test_inputs/eigen_test_files_with_gt.txt',
    'input_height': 352,
    'input_width': 1216,
    'max_depth': 80.0,
    'mode':'test',
    'do_kb_crop': True,
    'model_name':'bts_eigen_v2_pytorch_res18',
    'save_lpg':False
}
args = Namespace(**params)
from bts import BtsModel
model = BtsModel(params=args)
model = torch.nn.DataParallel(model)
checkpoint = torch.load(args.checkpoint_path)
model.load_state_dict(checkpoint['model'])
model = model.cuda()
model = model.eval()


num_samples = 10
#depth_pathlist = sorted(glob('%s/raw/*.png'%(root)))[:num_samples]
#image_pathlist = sorted(glob('%s/rgb/*.png'%(root)))[:num_samples]
image_pathlist = sorted(glob('data/*.png'))[:num_samples]

print(size)
out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 3, size)




transform = preprocessing_transforms('test')


for image_path in image_pathlist:
    # preproc
    image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
    if args.do_kb_crop is True:
        height = image.shape[0]
        width = image.shape[1]
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
    focal = 721.5377
    sample = {'image': image, 'focal': focal}
    sample = transform(sample)
    

    
    with torch.no_grad():
        image = torch.unsqueeze(sample['image'], dim=0).cuda() # torch.Size([1, 3, 352, 1216])
        print(image.shape)
        focal = torch.Tensor([focal]).cuda() # torch.Size([1]) = torch.Tensor([721.5377]).cuda()
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
    #pdb.set_trace()




    pred_depth = depth_est.cpu().numpy().squeeze()
    '''
    if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
        pred_depth_scaled = pred_depth * 256.0
    else:
        pred_depth_scaled = pred_depth * 1000.0
    pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
    cv2.imwrite('aaa.png', pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #depth = (pred_depth_scaled / 256.0).astype(np.uint8)
    depth = cv2.imread('aaa.png')
    '''
    depth = pred_depth


    print(image_path, depth.min(), depth.max(), depth.shape, depth.dtype)

    #pdb.set_trace()




    # image for visualization
    image = cv2.imread(image_path)
    if args.do_kb_crop:
        height = image.shape[0]
        width = image.shape[1]
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
    print('image.shape:', image.shape)



    # Saving colormapped depth image
    #vmax = np.percentile(depth, 95)
    #vmin, vmax = depth.min(), depth.max()
    vmin, vmax = 0.1, 80
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    #im.save('helloworld.png')

    # Display with matplotlib
    #plt.imshow(colormapped_im)
    #plt.axis('off')
    #cb = plt.colorbar(mapper, orientation='vertical')
    #cb.set_label('Depth Value')
    #plt.show()

    open_cv_image = np.array(im)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    #open_cv_image = cv2.resize(open_cv_image, (w, h))

    #pdb.set_trace()
    print('>>>',open_cv_image.shape, image.shape)
    open_cv_image = cv2.vconcat([open_cv_image, image])
    print("||||",open_cv_image.shape)
    #cv2.imwrite('bbbb.png', open_cv_image)
    #pdb.set_trace()

    out.write(open_cv_image)
out.release()
exit()

import pdb
pdb.set_trace()
print("===")
