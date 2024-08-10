import cv2
import numpy as np

def get_triangulation_indicesORIG(points):

    """Get indices triples for every triangle
    """
    # Bounding rectangle
    #input(points)
    bounding_rect = (*points.min(axis=0), *points.max(axis=0))
    # Triangulate all points
    subdiv = cv2.Subdiv2D(bounding_rect)
    for p in points:
        try:
            subdiv.insert([p])
        except Exception:
            print("E")
            pass
    # Iterate over all triangles
    for x1, y1, x2, y2, x3, y3 in subdiv.getTriangleList():
        # Get index of all points
        yield [(points==point).all(axis=1).nonzero()[0][0] for point in [(x1,y1), (x2,y2), (x3,y3)]]

def get_triangulation_indices(points):
    """Get indices triples for every triangle"""

    bounding_rect = (*points.min(axis=0), *points.max(axis=0))
    
    # Fixed bounding rectangle for a 256x256 image    
    #bounding_rect = (0, 0, 256, 256)
    ##bounding_rect = (0, 0, 512, 512)
    
    # Create Subdiv2D object
    subdiv = cv2.Subdiv2D(bounding_rect)
    
    # Insert points into Subdiv2D object
    for p in points:
        try:
            # Ensure point is a tuple of integers
            p = (int(p[0]), int(p[1]))
            #print(f"Inserting point: {p}")  # Debugging print
            subdiv.insert(p)
        except Exception as e:
            #print(f"Error inserting point {p}: {e}")
            pass
    
    # Get triangle list from Subdiv2D
    triangle_list = subdiv.getTriangleList()
    triangle_list = np.array(triangle_list, dtype=np.float32)
    
    indices_list = []
    for t in triangle_list:
        pts = t.reshape(3, 2)
        indices = []
        for pt in pts:
            idx = np.where((points == pt).all(axis=1))
            if idx[0].size > 0:
                indices.append(idx[0][0])
            else:
                #print(f"Point {pt} not found in original points")
                break
        if len(indices) == 3:
            indices_list.append(indices)
    
    return indices_list
            
def crop_to_triangle(img, triangle):
    """Crop image to triangle
    """
    # Get bounding rectangle
    bounding_rect = cv2.boundingRect(triangle)
    # Crop image to bounding box
    img_cropped = img[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],
                      bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    # Move triangle to coordinates in cropped image
    triangle_cropped = [(point[0]-bounding_rect[0], point[1]-bounding_rect[1]) for point in triangle]
    return triangle_cropped, img_cropped

def transform(src_img, src_points, dst_points): 
    """
    Transforms source image to target image, overwriting the target image.
    """
    
    # - soft face mask added
    
    image_height, image_width = src_img.shape[:2]
    face_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    segment1 = dst_points[0:32]  # from 0 to 33 inclusive
    segment2 = [dst_points[32], dst_points[128]]  # connect kp 33 to kp 128 directly
    segment3 = dst_points[128:97:-1]  # from 128 down to 98 (excluding 98)
    segment4 = [dst_points[98], dst_points[0]]  # connect kp 98 to kp 0 directly
    
    points = np.concatenate((segment1, [segment2[0]], [segment2[1]], segment3, [segment4[0]], [segment4[1]]), axis=0)
    points = points.astype(np.int32)
    points = points.reshape((-1, 1, 2))
    
    cv2.fillPoly(face_mask, [points], color=(255)) 

    kernel = np.ones((3,3), np.uint8)
    face_mask = cv2.erode(face_mask, kernel, iterations=3)
    face_mask = cv2.GaussianBlur(face_mask, (11,11), 0)
    
    #cv2.imshow("M",face_mask)
    
    face_mask = face_mask.astype(np.float64) / 255.0
    face_mask = np.expand_dims(face_mask, axis=-1)

    # -
    
    dst_img = src_img.copy()

    for indices in get_triangulation_indices(src_points):
        try:
            #print("W")
            # Get triangles from indices
            src_triangle = src_points[indices]
            dst_triangle = dst_points[indices]
            
            # Crop to triangle, to make calculations more efficient
            src_triangle_cropped, src_img_cropped = crop_to_triangle(src_img, src_triangle)
            dst_triangle_cropped, dst_img_cropped = crop_to_triangle(dst_img, dst_triangle)

            # Calculate transfrom to wrap from old image to new
            transform = cv2.getAffineTransform(np.float32(src_triangle_cropped), np.float32(dst_triangle_cropped))

            # Warp image
            dst_img_warped = cv2.warpAffine(src_img_cropped, transform, (dst_img_cropped.shape[1], dst_img_cropped.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

            # Create mask for the triangle we want to transform
            mask = np.zeros(dst_img_cropped.shape, dtype = np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst_triangle_cropped), (1.0, 1.0, 1.0), 16, 0);

            # Delete all existing pixels at given mask
            dst_img_cropped *= 1-mask
            # Add new pixels to masked area
            dst_img_cropped += dst_img_warped * mask


        except Exception as e:
            pass
            
    # - soft face mask added
    
    dst_img = face_mask * dst_img + (1 - face_mask) * (src_img)
    dst_img = np.clip(dst_img, 0, 255).astype(np.uint8)

    return dst_img
