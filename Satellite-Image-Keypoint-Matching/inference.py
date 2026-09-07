import cv2
import torch
import numpy as np
import kornia.feature as KF
import kornia as K

class SatelliteImageMatcher:
    def __init__(self, device: str = None, confidence_threshold: float = 0.2):
        """
        Initialization of the LoFTR (Local Feature TRansformer) SOTA model
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        
        # Load the pre-trained LoFTR model for outdoor data.
        self.matcher = KF.LoFTR(pretrained="outdoor").to(self.device).eval()        

    def load_and_preprocess(self, img_path: str):
        """
        Loads an image and converts it to Tensor format for Kornia (grayscale)
        """
        # Reading grayscale images for LoFTR
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            raise FileNotFoundError(f"Cannot load image at {img_path}")
            
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        
        # Convert the tensor to PyTorch
        # Стало:
        tensor = K.image.image_to_tensor(img_gray, keepdim=False).float() / 255.0
        tensor = tensor.to(self.device)
        
        return img_raw, tensor

    def match_pair(self, img1_path: str, img2_path: str):
        """
        Finds matching points between two satellite images.
        """
        img1_bgr, t1 = self.load_and_preprocess(img1_path)
        img2_bgr, t2 = self.load_and_preprocess(img2_path)

        input_dict = {"image0": t1, "image1": t2}

        # Neural network inference
        with torch.no_grad():
            correspondences = self.matcher(input_dict)

        # Get points and confidence
        mkpts1 = correspondences['keypoints0'].cpu().numpy()
        mkpts2 = correspondences['keypoints1'].cpu().numpy()
        confidence = correspondences['confidence'].cpu().numpy()

        # Filtering by confidence threshold
        mask = confidence > self.confidence_threshold
        mkpts1 = mkpts1[mask]
        mkpts2 = mkpts2[mask]

        if len(mkpts1) < 4:
            print("Too few points for geometric validation (RANSAC)")
            return img1_bgr, img2_bgr, mkpts1, mkpts2, np.zeros(len(mkpts1), dtype=bool)

        # Homography calculation via RANSAC to determine Inliers / Outliers
        _, inliers_mask = cv2.findHomography(
            mkpts1, 
            mkpts2, 
            cv2.USAC_MAGSAC, 
            ransacReprojThreshold=3.0, 
            maxIters=5000
        )

        inliers_mask = inliers_mask.ravel().astype(bool) if inliers_mask is not None else np.zeros(len(mkpts1), dtype=bool)

        return img1_bgr, img2_bgr, mkpts1, mkpts2, inliers_mask


if __name__ == "__main__":    
    try:
        matcher = SatelliteImageMatcher()        
    except Exception as e:
        print(f" Error during testing: {e}")
 