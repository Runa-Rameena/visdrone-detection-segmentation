import xml.etree.ElementTree as ET
import os

class VisDroneParser:
    def __init__(self, xml_file):
        self.xml_file = xml_file
        self.annotations = self.parse_xml()

    def parse_xml(self):
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        annotations = []
        
        for image in root.findall('image'):
            file_name = image.get('name')
            for box in image.findall('box'):
                label = box.get('label')
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                annotations.append({
                    'file_name': file_name,
                    'label': label,
                    'bbox': [xtl, ytl, xbr, ybr]
                })
        return annotations

    def get_annotations(self):
        return self.annotations

# Example usage (Uncomment the following lines to use it):
# if __name__ == '__main__':
#     parser = VisDroneParser('path/to/annotation.xml')
#     print(parser.get_annotations())
