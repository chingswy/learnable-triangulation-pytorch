import numpy as np
import xml.etree.ElementTree as ET


class H36M_Metadata:
    def __init__(self, metadata_file):
        self.subjects = []
        self.sequence_mappings = {}
        self.action_names = {}
        self.camera_ids = []

        tree = ET.parse(metadata_file)
        root = tree.getroot()

        for i, tr in enumerate(root.find('mapping')):
            if i == 0:
                _, _, *self.subjects = [td.text for td in tr]
                self.sequence_mappings = {subject: {} for subject in self.subjects}
            elif i < 33:
                action_id, subaction_id, *prefixes = [td.text for td in tr]
                for subject, prefix in zip(self.subjects, prefixes):
                    self.sequence_mappings[subject][(action_id, subaction_id)] = prefix

        for i, elem in enumerate(root.find('actionnames')):
            action_id = str(i + 1)
            self.action_names[action_id] = elem.text

        self.camera_ids = [elem.text for elem in root.find('dbcameras/index2id')]

    def get_base_filename(self, subject, action, subaction, camera):
        return '{}.{}'.format(self.sequence_mappings[subject][(action, subaction)], camera)


def load_h36m_metadata():
    return H36M_Metadata('metadata.xml')

def rotation_matrix(args):

    (x, y, z) = args

    X = np.vstack([[1, 0, 0], [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    Y = np.vstack([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Z = np.vstack([[np.cos(z), -np.sin(z), 0], [np.sin(z),
                                                np.cos(z), 0], [0, 0, 1]])

    return (X.dot(Y)).dot(Z)

def read_cam_parameters(xml_path, sbj_id, cam_id):
    import xml.etree.ElementTree

    # use the notation from 0 -- more practical to access array
    sbj_id = sbj_id - 1
    cam_id = cam_id - 1

    n_sbjs = 11
    n_cams = 4

    root = xml.etree.ElementTree.parse(xml_path).getroot()

    for child in root:
        if child.tag == 'w0':
            all_cameras = child.text
            tokens = all_cameras.split(' ')
            tokens[0] = tokens[0].replace('[', '')
            tokens[-1] = tokens[-1].replace(']', '')

            start = (cam_id * n_sbjs) * 6 + sbj_id * 6
            extrs = tokens[start:start + 6]

            start = (n_cams * n_sbjs * 6) + cam_id * 9
            intrs = tokens[start:start + 9]

            rot = rotation_matrix(np.array(extrs[:3], dtype=float))

            rt = rot
            t = np.array(extrs[3:], dtype=float)

            f = np.array(intrs[:2], dtype=float)
            c = np.array(intrs[2:4], dtype=float)

            distortion = np.array(intrs[4:], dtype=float)

            k = np.hstack((distortion[:2], distortion[3:5], distortion[2:3]))
            return (rt, t, f, c, k)

def process_camera(xml_path, seq, cam):
    # for i, cam in enumerate(cams, 1):
    cams = ['54138969', '55011271', '58860488', '60457274']
    i = cams.index(cam) + 1
    (rt, t, f, c, k) = read_cam_parameters(xml_path, int(seq.replace('S', '')), i)
    K = np.eye(3)
    K[0, 0] = f[0]
    K[1, 1] = f[1]
    K[0, 2] = c[0]
    K[1, 2] = c[1]
    # camera center
    T = t.reshape(3, 1)
    T = -np.dot(rt, T)
    cameras = {
        'K': K,
        'R': rt,
        'T': T,
        'dist': k.reshape(1, 5)
    }
    return cameras


if __name__ == '__main__':
    metadata = load_h36m_metadata()
    print(metadata.subjects)
    print(metadata.sequence_mappings)
    print(metadata.action_names)
    print(metadata.camera_ids)
