from model import *
import torch
import cv2
# CUDA_LAUNCH_BLOCKING=1
# print(torch.cuda.is_available())
# exit()
def transform(image_np):
        ToPILImage = transforms.ToPILImage()
        image = ToPILImage(image_np)
        
        image = TF.to_tensor(image)

        return image

if __name__ == "__main__":
    device = torch.device('cpu')
    torch.manual_seed(42)
    model = AttU_Net()
    model = model.float()
    model = model.to(device)
    # model.load_state_dict(torch.load('/home/anlab/Desktop/Songuyen/Segment_solar_unet_attention/CP/epoch_11_iou0.6879278330790016cp.pth'))
    model.load_state_dict(torch.load('/home/anlab/Desktop/Songuyen/Segment_solar_unet_attention/CP/epoch_11_iou0.6879278330790016cp.pth', map_location=torch.device('cpu')))
    # model.eval()
    img_path = "/home/anlab/Desktop/Songuyen/Segment_solar_unet_attention/img_test/image_1.PNG"
    img = imread(img_path)[:,:,:3]
    img = resize(img, (256, 256), mode='constant', preserve_range=True)
    img = img.astype(np.uint8)
    img = transform(img)
    img = img.to(device)
    img = img.unsqueeze_(0)
    mask_pred = model(img.float())
    mask_pred = mask_pred.cpu()
    mask_pred = (mask_pred > 0.8)
    pred = TF.to_pil_image(mask_pred.float().squeeze(0))
    pred = np.array(pred)
    cv2.imwrite("out.png", pred)