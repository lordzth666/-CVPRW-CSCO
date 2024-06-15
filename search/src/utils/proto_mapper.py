import re

from src.utils.prototype import ProtoType


def is_empty_line(line):
    return re.match(".*[0-9|a-z|A-Z].*", line) is None


class ProtoMapper:
    def __init__(self, prototxt):
        self.prototxt = prototxt
        pass

    def get_next_block(self, fp):
        while 1:
            line = fp.readline()
            if line == "":
                return None
            if not is_empty_line(line):
                break
        block_content = [line.strip("\n")]
        while 1:
            line = fp.readline()
            if is_empty_line(line):
                break
            block_content.append(line.strip("\n"))
        return block_content

    def parse_proto(self):
        block_defs = []
        with open(self.prototxt, "r") as fp:
            while 1:
                block_content = self.get_next_block(fp)
                if block_content is None:
                    break
                processed_block_content = ProtoType(block_content)()
                block_defs.append(processed_block_content)
        return block_defs
