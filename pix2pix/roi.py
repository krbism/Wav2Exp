import torch


def calculate_roi_loss(src_img, tar_img):
    cnt = 0
    roi = []
    # ganimation.feed_batch(src_img, exp_vec) #Adept Animator
    # ganimation.forward()
    # attn = ganimation.aus_mask.detach()
    attn = torch.randn([1,1,128,128])
    att_mean = attn.view(attn.size(0), -1).mean(1, keepdim=True)
    l1_loss = torch.nn.L1Loss().to(torch.device('cuda'))
    for j in range(attn.size(0)):
        zero = torch.zeros_like(src_img[j])
        roi_map_src = torch.where(attn[j]>att_mean[j].item(), zero, src_img[j])
        roi_map_tar = torch.where(attn[j]>att_mean[j].item(), zero, tar_img[j])
        roi_loss = torch.sum(torch.abs(roi_map_tar - roi_map_src))
        roi_loss = roi_loss/torch.count_nonzero(roi_map_tar) #L1 Loss for extracted elements w.r.t Attention maps.
        roi.append(roi_loss)
    roi_mean = sum(roi)/len(roi)
    return roi_mean, roi_mean_sparse

if __name__ == '__main__':
    gt = torch.randn([1,3,128,128])
    g  = torch.randn([1,3,128,128])

    roi_loss, roi_mean_sparse = calculate_roi_loss(g, gt)
    print("ROI Loss", roi_loss)
    print("Sparse ROI Loss", roi_mean_sparse)
