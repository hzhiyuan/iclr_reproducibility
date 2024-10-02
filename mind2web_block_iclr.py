import re
import json
from PIL import Image
from tqdm import tqdm


def replace_and_add_block(s, is_test_data, block_num_w, block_num_h) -> str:
    pattern = re.compile(r'\(\[(\d{3}), (\d{3}), (\d{3}), (\d{3})\],')
    match = pattern.search(s)

    if match:
        numbers = [int(match.group(i)) for i in range(1, 5)]

        cx, cy, width, height = numbers
        block_w = 1000 / block_num_w
        block_h = 1000 / block_num_h
        block_id_x = int(cx // block_w)
        block_id_y = int(cy // block_h)
        assert 0 <= block_id_x < block_num_w
        assert 0 <= block_id_y < block_num_h

        block_id = block_num_w * block_id_y + block_id_x
        cx = (cx % block_w) / block_w
        cy = (cy % block_h) / block_h
        cx_rel = round(min(0.999, max(0, cx)) * 1000)
        cy_rel = round(min(0.999, max(0, cy)) * 1000)

        if not is_test_data:
            width = round(min(0.999, max(0, width / block_w)) * 1000)
            height = round(min(0.999, max(0, height / block_h)) * 1000)

        new_numbers = [block_id, cx_rel, cy_rel, width, height]
        replacement = f"([{str(new_numbers[0])}, {', '.join(str(num).zfill(3) for num in new_numbers[1:])}],"
        new_string = pattern.sub(replacement, s)

        return new_string
    else:
        raise ValueError('Match Fail!')


def main(src_path, dst_path, is_test_data, block_num_w, block_num_h):
    new_lines = []
    for line in tqdm(open(src_path).readlines()):
        meta = json.loads(line)
        meta['block_wh'] = [block_num_w, block_num_h]
        conversations = meta['conversations']
        assert len(conversations) == 2

        conversations[1]['value'] = replace_and_add_block(conversations[1]['value'], is_test_data, block_num_w, block_num_h)
        new_lines.append(json.dumps(meta))

    with open(dst_path, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')


block_num_w = 3
block_num_h = 2

src_path = 'xxx'
dst_path = 'xxx'

main(src_path, dst_path, is_test_data=True, block_num_w=block_num_w, block_num_h=block_num_h)
