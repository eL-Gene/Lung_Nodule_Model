import SimpleITK as sitk 
import numpy as np 
import glob


# converting from index, row, column -> x, y, z
def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    return coords_xyz

# converting from x, y, z -> index, row, column
def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return cri_a[2], cri_a[1], cri_a[0]


class Ct:
    def __init__(self, series_uid):
        self.series_uid = series_uid
        # self.mhd_path = glob.glob(f'D:/LUNASet/subset*/subset*/{self.series_uid}.mhd')[0]
        # self.mhd_path = glob.glob(f'../LUNASet/subset*/subset*/{self.series_uid}.mhd')[0]
        self.mhd_path = glob.glob(f'/media/e_quitee/Data Drive/LUNASet/subset*/subset*/{self.series_uid}.mhd')[0]
        self.ct_mhd = sitk.ReadImage(self.mhd_path)
        self.pre_ct_a = np.array(sitk.GetArrayFromImage(self.ct_mhd), dtype=np.float32)
        self.ct_a = self.pre_ct_a.clip(-1000, 1000, self.pre_ct_a)
        self.hu_a = self.ct_a

        # we are adding the following -- we are using methods from sitk modules
        # to construct the following attributes
        self.origin_xyz = self.ct_mhd.GetOrigin()
        self.vxSize_xyz = self.ct_mhd.GetSpacing()
        self.direction_a = np.array(self.ct_mhd.GetDirection()).reshape(3, 3)

    # # converting from index, row, column -> x, y, z
    # def irc2xyz(self, coord_irc, origin_xyz, vxSize_xyz, direction_a):
    #     cri_a = np.array(coord_irc)[::-1]
    #     origin_a = np.array(origin_xyz)
    #     vxSize_a = np.array(vxSize_xyz)
    #     coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    #     return coords_xyz
    
    # # converting from x, y, z -> index, row, column
    # def xyz2irc(self, coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    #     origin_a = np.array(origin_xyz)
    #     vxSize_a = np.array(vxSize_xyz)
    #     coord_a = np.array(coord_xyz)
    #     cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    #     cri_a = np.round(cri_a)
    #     return cri_a[2], cri_a[1], cri_a[0]

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            coord_xyz=center_xyz,
            origin_xyz=self.origin_xyz, 
            vxSize_xyz=self.vxSize_xyz,
            direction_a=self.direction_a
        )

        slice_list =[]
        # for axis, center_val in enumerate(center_irc):
        #     start_ndx = int(round(center_val - width_irc[axis]/2))
        #     end_ndx = int(start_ndx + width_irc[axis])
        #     slice_list.append(slice(start_ndx, end_ndx))

        # ct_chunk = self.hu_a[tuple(slice_list)]

        # return ct_chunk, center_irc

        # the implementation actually used by the book as opposed to the one written in 
        # the book
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc        

if __name__ == '__main__':
    pass