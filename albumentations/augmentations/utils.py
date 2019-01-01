import cv2


class ImageSource(object):
    """Image data source class to provide list like access.
    Set images or files argument. If files is set, all the images will be loaded
    when class instance is created.

    Args:
        images: Image data array like object, or None to create from file list.
        files: Image file name array like object, or None.
        transform: Albumentations' transforms to apply when accessed.
    """

    def __init__(self, images=None, files=None, transform=None):
        assert (images is not None) or (files is not None)
        self.images = images if images is not None else [cv2.imread(str(f)) for f in files]
        self.files = files
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        if self.transform is not None:
            result = self.transform(image=img)
            img = result['image']
        return img

    def __len__(self):
        return len(self.files)
