from typing import List, Union, Optional, Sequence
import numpy as np
from sympy.geometry import Point, Segment, Ray, Polygon
from sympy import oo, pi, Integer


__all__ = [
    'ImageWithIrregularROI'
]


class Quadrangle:
    @property
    def list_points(self):
        return [self.points.T[i].tolist() for i in range(4)]

    @property
    def topleft(self):
        return Point(self.points.T[0])

    @property
    def topright(self):
        return Point(self.points.T[1])

    @property
    def bottomright(self):
        return Point(self.points.T[2])

    @property
    def bottomleft(self):
        return Point(self.points.T[3])

    p1 = topleft
    p2 = topright
    p3 = bottomright
    p4 = bottomleft

    @property
    def edge_top(self):
        return Segment(self.p1, self.p2)

    @property
    def edge_right(self):
        return Segment(self.p2, self.p3)

    @property
    def edge_buttom(self):
        return Segment(self.p3, self.p4)

    @property
    def edge_left(self):
        return Segment(self.p4, self.p1)

    edge1 = edge_top
    edge2 = edge_right
    edge3 = edge_buttom
    edge4 = edge_left

    @property
    def polygon(self):
        return Polygon(self.p1, self.p2, self.p3, self.p4)

    def set_new_point(self, point: Point, point_idx):
        pass

    def set_topleft(self, point):
        self.set_new_point(point, 0)

    def set_topright(self, point):
        self.set_new_point(point, 1)

    def set_bottomright(self, point):
        self.set_new_point(point, 2)

    def set_bottomleft(self, point):
        self.set_new_point(point, 3)

    set_p1 = set_topleft
    set_p2 = set_topright
    set_p3 = set_bottomright
    set_p4 = set_bottomleft

    def is_topleft_superfluous(self):
        if self.edge1.slope < 0:
            return self.edge4.slope < 0 or self.edge4.slope == oo
        elif self.edge1.slope == 0:
            return self.edge4.slope < 0
        return False

    def cut_topleft_superfluous(self):
        if self.is_topleft_superfluous():
            if self.edge1.slope < 0:
                ray = Ray(self.edge1.p1, Point(self.edge1.p1.x + 1, self.edge1.p1.y))
                new_p2 = self.edge2.intersection(ray)[0]
                self.set_p2(new_p2)
            if self.edge4.slope < 0:
                ray = Ray(self.edge4.p2, Point(self.edge4.p2.x, self.edge4.p2.y + 1))
                new_p4 = self.edge3.intersection(ray)[0]
                self.set_p4(new_p4)

    def is_orthogonal_rectangle(self):
        return np.all([i == pi / 2 for i in self.polygon.angles.values()]) and self.p1.y == self.p2.y

    def search_orthogonal_rectangle(self, max_grid = 100):
        if not self.is_topleft_superfluous():
            edge4_expr = self.edge4.arbitrary_point()
            var = list(edge4_expr.free_symbols)[0]
            edge4_x_pixels = abs(self.edge4.p2.x - self.edge4.p1.x)
            edge4_y_pixels = abs(self.edge4.p2.y - self.edge4.p1.y)
            edge4_pixels = min(max_grid, max(edge4_x_pixels, edge4_y_pixels))
            if self.edge4.slope < 0:
                ray1_dir = 1
                ray2_dir = 1
                ray1_tar = self.edge1
                ray2_tar = self.edge3
            elif self.edge4.slope > 0:
                ray1_dir = 1
                ray2_dir = -1
                ray1_tar = self.edge3
                ray2_tar = self.edge1
            max_area = -1
            max_poly = None
            for i in range(edge4_pixels - 1):
                point = edge4_expr.subs(var, 1 - (i + 1) / edge4_pixels)
                point = Point(Integer(point.x), Integer(point.y))
                ray1 = Ray(point, Point(point.x + ray1_dir, point.y))
                ray2 = Ray(point, Point(point.x, point.y + ray2_dir))
                intersection1 = ray1_tar.intersection(ray1)
                intersection2 = ray2_tar.intersection(ray2)
                if len(intersection1) == 0 or len(intersection2) == 0:
                    continue
                intersect_point1 = Point(Integer(intersection1[0].x), Integer(intersection1[0].y))
                intersect_point2 = Point(Integer(intersection2[0].x), Integer(intersection2[0].y))
                ray3 = Ray(intersect_point1, Point(intersect_point1.x, intersect_point1.y + ray2_dir))
                ray4 = Ray(intersect_point2, Point(intersect_point2.x + ray1_dir, intersect_point2.y))
                intersection3 = ray3.intersection(ray4)
                intersect_point3 = Point(Integer(intersection3[0].x), Integer(intersection3[0].y))
                if self.polygon.encloses(intersect_point3):
                    if ray2_dir == 1:
                        poly = Polygon(point, intersect_point1, intersect_point3, intersect_point2)
                    else:
                        poly = Polygon(intersect_point2, intersect_point3, intersect_point1, point)
                    if poly.area > max_area:
                        max_area = poly.area
                        max_poly = poly
            return max_area, max_poly
        print('Cannot apply current algorithm to Quadrangle with superfluous vertex.')
        return -1, None

class Image:
    @property
    def corner_topleft(self):
        return Point(0, 0)

    @property
    def corner_topright(self):
        return Point(self.width - 1, 0)

    @property
    def corner_bottomright(self):
        return Point(self.width - 1, self.height - 1)

    @property
    def corner_bottomleft(self):
        return Point(0, self.height - 1)

    c1 = corner_topleft
    c2 = corner_topright
    c3 = corner_bottomright
    c4 = corner_bottomleft

    @property
    def bounding_polygon(self):
        return Polygon(self.c1, self.c2, self.c3, self.c4)

    @property
    def size(self):
        return (self.width, self.height)

    def in_image(self, point: Point):
        return self.bounding_polygon.encloses(point)


class ImageView(Quadrangle, Image):
    permute_matrices = [
        np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]),
        np.array([[0, 0, 0, 1],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]]),
        np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0]]),
        np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [1, 0, 0, 0]]),
    ]
    trans_matrices = [
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 1], [-1, 0]]),
            np.array([[-1, 0], [0, -1]]),
            np.array([[0, -1], [1, 0]]),
        ]
    bias_index = [
        [-1, -1],
        [-1, 0],
        [0, 1],
        [1, -1]
    ]
    inverse_trans_matrices = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, -1], [1, 0]]),
        np.array([[-1, 0], [0, -1]]),
        np.array([[0, 1], [-1, 0]]),
    ]
    inverse_bias_index = [
        [-1, -1],
        [0, -1],
        [0, 1],
        [-1, 1]
    ]
    def __init__(self, image, perspective):
        self.image = image
        self.perspective = perspective
        self.trans_matrix = self.trans_matrices[perspective]
        # self.bias = np.array([[image.size[i] if i >= 0 else 0 for i in self.bias_index[perspective]]]).T
        self.inverse_trans_matrix = self.inverse_trans_matrices[perspective]
        # self.inverse_bias = np.array([[image.size[i] if i >= 0 else 0 for i in self.inverse_bias_index[perspective]]]).T
        self.permute_matrix = self.permute_matrices[perspective]

    @property
    def bias(self):
        return np.array([[self.image.size[i] - 1 if i >= 0 else 0 for i in self.bias_index[self.perspective]]]).T

    @property
    def inverse_bias(self):
        return np.array([[self.image.size[i] - 1 if i >= 0 else 0 for i in self.inverse_bias_index[self.perspective]]]).T

    @property
    def width(self):
        return self.image.width if self.perspective == 0 or self.perspective == 2 else self.image.height

    @property
    def height(self):
        return self.image.height if self.perspective == 0 or self.perspective == 2 else self.image.width

    @property
    def points(self):
        return np.matmul(np.matmul(self.trans_matrix, self.image.points) + self.bias, self.permute_matrix)

    def set_new_point(self, point: Point, point_idx):
        if not self.in_image(point):
            raise ValueError('Point beyond image boundary.')
        new_point_reverse = np.array([np.dot(self.inverse_trans_matrix[i], np.array([point.x, point.y])) + self.inverse_bias[i][0] for i in range(2)])
        self.image.points[:, np.argmax(self.permute_matrix.T[point_idx])] = new_point_reverse

    def set_new_ROI_poly(self, ROI_poly: Polygon):
        for i in range(4):
            self.set_new_point(ROI_poly.vertices[i], i)


class ImageWithIrregularROI(Quadrangle, Image):
    def __init__(self, width: int, height: int, ROI: List[List[int]]) -> object:
        self.width = width
        self.height = height
        self.points = np.array(ROI).T
        self.views = [ImageView(self, i) for i in range(4)]

    def view(self, perspective):
        return self.views[perspective]

    def rotate(self, angle: int):
        rotate_center = Point(Integer(self.width / 2), Integer(self.height / 2))

        points_boundary_poly = [self.bounding_polygon.vertices[i].rotate(
            angle=angle * pi / 180,
            pt=rotate_center
        ) for i in range(4)]
        points_boundary_poly = [Point(
            Integer(points_boundary_poly[i].x),
            Integer(points_boundary_poly[i].y)
        ) for i in range(4)]
        new_bounding_polygon = Polygon(*points_boundary_poly)
        self.width = int(new_bounding_polygon.bounds[2] - new_bounding_polygon.bounds[0] + 3)
        self.height = int(new_bounding_polygon.bounds[3] - new_bounding_polygon.bounds[1] + 3)

        points_ROI_poly = [self.polygon.vertices[i].rotate(
            angle=angle * pi / 180,
            pt=rotate_center
        ).translate(
            -new_bounding_polygon.bounds[0] + 1,
            -new_bounding_polygon.bounds[1] + 1
        ) for i in range(4)]
        points_ROI_poly = [Point(
            Integer(points_ROI_poly[i].x),
            Integer(points_ROI_poly[i].y)
        ) for i in range(4)]

        new_points = np.array([[points_ROI_poly[i].x, points_ROI_poly[i].y] for i in range(4)]).T
        sort = np.argsort(new_points[1])
        if new_points[0, sort[0]] > new_points[0, sort[1]]:
            _ = sort[0]
            sort[0] = sort[1]
            sort[1] = _
        if new_points[0, sort[2]] < new_points[0, sort[3]]:
            _ = sort[2]
            sort[2] = sort[3]
            sort[3] = _
        for i in range(4):
            self.points[:, i] = new_points[:, sort[i]]

    def hflip(self):
        self.points = np.array([
            [self.width - 1 - self.p2.x, self.p2.y],
            [self.width - 1 - self.p1.x, self.p1.y],
            [self.width - 1 - self.p4.x, self.p4.y],
            [self.width - 1 - self.p3.x, self.p3.y],
        ], dtype=np.int64).T

    def vflip(self):
        self.points = np.array([
            [self.p4.x, self.height - 1 - self.p4.y],
            [self.p3.x, self.height - 1 - self.p3.y],
            [self.p2.x, self.height - 1 - self.p2.y],
            [self.p1.x, self.height - 1 - self.p1.y],
        ], dtype=np.int64).T

    def is_superfluous_ROI(self):
        _is_superfluous = False
        for view in self.views:
            _is_superfluous = _is_superfluous or view.is_topleft_superfluous()
        return _is_superfluous

    def cut_superfluous_ROI(self):
        if not self.is_superfluous_ROI():
            print('No superfluous vertex in current ROI.')
            return
        while not self.is_orthogonal_rectangle():
            for view in self.views:
                view.cut_topleft_superfluous()

    def cut_non_superfluous_ROI(self):
        if self.is_superfluous_ROI():
            print("There's superfluous vertex in current ROI.")
            return
        max_area = -1
        max_poly = None
        max_poly_perspective = -1
        for view in self.views:
            max_area_view, max_poly_view = view.search_orthogonal_rectangle()
            if max_area_view > max_area:
                max_area = max_area_view
                max_poly = max_poly_view
                max_poly_perspective = self.views.index(view)
        self.view(max_poly_perspective).set_new_ROI_poly(max_poly)

    def cut_max_ROI(self):
        if self.is_orthogonal_rectangle():
            return
        if self.is_superfluous_ROI():
            self.cut_superfluous_ROI()
            return
        self.cut_non_superfluous_ROI()


if __name__ == "__main__":
    from torchvision import transforms as T
    from torchvision.transforms import functional as TF
    import os
    from pathlib import Path
    from PIL import Image as PILImage
    import cv2

    fld = Path('/mnt/WinD/pythondata/PyTorch/data/test_ROI_cut')
    trans_fld = fld / 'transformed'
    cut_fld = fld / 'cut'
    if not trans_fld.exists():
        os.makedirs(trans_fld)
    if not cut_fld.exists():
        os.makedirs(cut_fld)
    img_paths = fld.glob('*.jpg')
    # converter = T.PILToTensor()
    # inv_converter = T.ToPILImage()
    rp = T.RandomPerspective
    rr = T.RandomRotation
    for img_path in img_paths:
        img = PILImage.open(img_path)
        # img_tensor = converter(img)
        params = rp.get_params(img.width, img.height, 0.5)
        img = TF.perspective(img, *params)
        cutter = ImageWithIrregularROI(img.width, img.height, params[1])
        params2 = rr.get_params([-180, 180])
        print('Image: {img} size before rotation: ({w}, {h})'.format(img=img_path.stem, w=img.width, h=img.height))
        img = TF.rotate(img, params2, expand=True)
        print('Image: {img} size after rotation: ({w}, {h})'.format(img=img_path.stem, w=img.width, h=img.height))
        print('ImageObj: {img} size before rotation: ({w}, {h})'.format(img=img_path.stem, w=cutter.width, h=cutter.height))
        cutter.rotate(-params2)
        print('ImageObj: {img} size after rotation: ({w}, {h})'.format(img=img_path.stem, w=cutter.width, h=cutter.height))
        img = TF.hflip(img)
        img = TF.vflip(img)
        cutter.hflip()
        cutter.vflip()
        cutter_points = np.array(cutter.list_points)
        cutter.cut_max_ROI()
        img_cut = TF.crop(img, cutter.p1.y, cutter.p1.x, cutter.p4.y - cutter.p1.y, cutter.p2.x - cutter.p1.x)
        img = np.array(img, dtype=np.uint8)
        img = cv2.polylines(img, [cutter_points], True, (0, 0, 255), 3)
        img = cv2.polylines(img, [np.array(cutter.list_points)], True, (0, 255, 0), 3)
        # img_trans = inv_converter(img_tensor)
        # img.save(trans_fld / (img_path.stem + '_trans.jpg'))
        cv2.imwrite((trans_fld / (img_path.stem + '_trans.jpg')).as_posix(), img)
        img_cut.save(cut_fld / (img_path.stem + '_cut.jpg'))


    ii=0