from PIL import Image, ImageDraw
from constant import const
from utils.decorator import count_time
from utils import log
from utils.random_tools import Random
from core.layout.strategy import Strategy
from core.element.TextImg import TYPE_ORIENTATION_HORIZONTAL, \
    TYPE_ORIENTATION_VERTICAL, TextImg
import os
import hashlib
import json


class Block:
    def __init__(self, img: Image.Image, inner_x=0, inner_y=0, margin=0, rotate_angle=0):
        self.inner_box = ()
        self.outer_box = ()
        self.img = img
        self.img = self.img.rotate(angle=rotate_angle, expand=True)

        self.margin = margin

        self.inner_width = self.img.width
        self.inner_height = self.img.height
        self.inner_size = (self.inner_width, self.inner_height)

        self.outer_width = self.inner_width + 2 * margin
        self.outer_height = self.inner_height + 2 * margin
        self.outer_size = (self.outer_width, self.outer_height)

        self.locate_by_inner(inner_x=inner_x, inner_y=inner_y)

    def get_data(self):
        pass

    def get_orientation(self):
        if self.inner_width > self.inner_height:
            return TYPE_ORIENTATION_HORIZONTAL
        else:
            return TYPE_ORIENTATION_VERTICAL

    def locate_by_inner(self, inner_x, inner_y):
        self.inner_box = (inner_x, inner_y, inner_x + self.img.width, inner_y + self.img.height)
        self.outer_box = (inner_x - self.margin,
                          inner_y - self.margin,
                          inner_x + self.img.width + self.margin,
                          inner_y + self.img.height + self.margin)

    def locate_by_outter(self, outer_x, outer_y):
        inner_x = outer_x + self.margin
        inner_y = outer_y + self.margin
        self.locate_by_inner(inner_x, inner_y)

    def get_img(self) -> Image.Image:
        return self.img

    def get_alpha_mask(self):
        r, g, b, a = self.get_img().split()
        return a

    def crop_self(self, bg_img: Image.Image):
        return bg_img.crop(self.inner_box)


class TextBlock(Block):
    def __init__(self, text_img: TextImg, inner_x=0, inner_y=0, margin=0, rotate_angle=0):
        self.img = text_img.img
        super().__init__(self.img, inner_x, inner_y, margin=margin, rotate_angle=rotate_angle)
        # todo:???????????????????????????????????????
        # for char_obj in text_img.char_obj_list:
        #     char_obj.box = # ???????????????????????????????????????

        self.text_img = text_img

    def get_img(self) -> Image.Image:
        return super().get_img()

    def get_data(self):
        return str(self.text_img.text)

    def get_orientation(self):
        return self.text_img.orientation


class NextBlockGenerator:
    def auto_gen_next_img_block(self, width, height, strategy, bg_img, block_list, rotate_angle):
        pass


class BlockGroup:
    def __init__(self, bg_img: Image.Image, group_box,
                 rotate_angle_range,
                 next_block_generator=None,
                 strategy_list=[]):
        self.bg_img = bg_img
        self.group_box = group_box
        self.block_list = []
        self.width = group_box[2] - group_box[0]
        self.height = group_box[3] - group_box[1]
        self.bg_width = self.bg_img.width
        self.bg_height = self.bg_img.height
        self.rotate_angle_range = rotate_angle_range
        self.strategy_list = strategy_list

        self.next_block_generator = next_block_generator

    def auto_append_block(self):
        """
        ????????????block
        :return:
        """
        from core.layout.strategy import strategy_controller as sc
        strategy = sc.pick(self.strategy_list)
        # ????????????3??? ?????????????????????
        retry_times = 5
        while retry_times > 0:
            block = self._gen_block(strategy)
            r = False
            if block:
                r = strategy.logic(self, block)
                if r:
                    self.block_list.append(block)
                    if isinstance(block, TextBlock):
                        log.info("add text on box:[{group_box}] [{strategy}] [{orientation}] > {text}".format(
                            group_box=self.group_box,
                            strategy=strategy.name(),
                            text=block.text_img.text,
                            orientation="h" if block.text_img.orientation == 0 else "v"
                        ))
            if not r:
                retry_times -= 1
                log.info("retry auto append block")

    def _gen_block(self, strategy: Strategy):
        """
        ????????????block
        :return:
        """

        rotate_angle = Random.random_int(self.rotate_angle_range[0], self.rotate_angle_range[1])

        block = self.next_block_generator.auto_gen_next_img_block(width=self.width,
                                                                  height=self.height,
                                                                  strategy=strategy,
                                                                  bg_img=self.bg_img,
                                                                  block_list=self.block_list,
                                                                  rotate_angle=rotate_angle,
                                                                  )
        return block

    def preview(self, draw_rect=False):
        """
        ??????
        :param draw_rect:
        :return:
        """
        bg_img = self.render(draw_rect, on_origin=False)
        bg_img.show("")

    def render(self, draw_rect=False, on_origin=True):
        """
        ??????
        :param draw_rect:
        :param on_origin: ????????????????????????
        :return:
        """
        mode = "RGBA"
        if not self.bg_img:
            bg_img = Image.new(mode, size=(self.width, self.height))
        else:
            if on_origin:
                bg_img = self.bg_img
            else:
                bg_img = self.bg_img.copy()
        draw = ImageDraw.Draw(bg_img, mode)
        for block in self.block_list:
            img = block.get_img()
            mask = block.get_alpha_mask()
            bg_img.paste(img, block.inner_box, mask=mask)

            if draw_rect:
                draw.rectangle(xy=block.outer_box, width=1, outline=const.COLOR_RED)
                draw.rectangle(xy=block.inner_box, width=1, outline=const.COLOR_GREEN)

        if draw_rect:
            draw.rectangle(xy=self.group_box, width=0, outline=const.COLOR_TRANSPARENT,
                           fill=const.COLOR_HALF_TRANSPARENT)
        sub_img = bg_img.crop(self.group_box)
        return sub_img


class Layout:
    def __init__(self,
                 bg_img: Image.Image,
                 out_put_dir: str,
                 rotate_angle_range,
                 group_box_list: list = [],
                 next_block_generator=None,
                 strategy_list=[]):
        self.bg_img = bg_img
        self.out_put_dir = out_put_dir
        self.group_box_list = group_box_list
        self.next_block_generator = next_block_generator

        self.block_group_list = []
        for group_box in self.group_box_list:
            block_group = BlockGroup(bg_img, group_box, rotate_angle_range, self.next_block_generator, strategy_list)
            self.block_group_list.append(block_group)

    def get_all_block_list(self):
        """
        ???????????????block
        :return:
        """
        all_block_list = []
        for block_group in self.block_group_list:
            block_list = block_group.block_list
            all_block_list.extend(block_list)
        return all_block_list

    @count_time(tag="????????????????????????")
    def gen(self):
        """
        ??????????????????
        :return:
        """
        for index, block_group in enumerate(self.block_group_list):
            log.info("start append block ---- {index} ----".format(index=index))
            block_group.auto_append_block()
        self.render()

    def is_empty(self):
        """
        ??????????????????????????????
        :return:
        """
        for bg in self.block_group_list:
            if bg.block_list:
                return False
        return True

    @count_time(tag="???????????????")
    def collect_block_fragment(self):
        fragment_info_list = []
        for block in self.get_all_block_list():
            fragment_img = block.crop_self(self.bg_img)
            fragment_box = block.inner_box
            fragment_data = block.get_data()
            orientation = block.get_orientation()
            item = {
                "img": fragment_img,
                "box": fragment_box,
                "data": fragment_data,
                "orientation": 'horizontal' if orientation is TYPE_ORIENTATION_HORIZONTAL else 'vertical',
                "type": str(block.__class__.__name__)
            }
            fragment_info_list.append(item)
        return fragment_info_list

    @count_time("dump layout info")
    def dump(self):
        from service.base import get_data_dir, get_pic_dir, get_fragment_dir

        result = {}

        pic_dir = get_pic_dir(self.out_put_dir)
        fragment_dir = get_fragment_dir(self.out_put_dir)
        data_dir = get_data_dir(self.out_put_dir)

        os.makedirs(pic_dir, exist_ok=True)
        os.makedirs(fragment_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        name = hashlib.sha1(self.bg_img.tobytes()).hexdigest()

        pic_name = "pic_" + name + ".jpg"
        pic_path = os.path.join(pic_dir, pic_name)
        with open(pic_path, 'wb') as f:
            self.bg_img.save(f)
        result['pic_name'] = pic_name
        result['width'] = self.bg_img.width
        result['height'] = self.bg_img.height
        result['fragment'] = []

        for index, fragment in enumerate(self.collect_block_fragment()):
            fragment_img = fragment['img']
            # fragment_img_name = "fragment_" + name + str(index) + ".jpg"
            ### ??????
            fragment_img_name = "fragmentdot_" + name + str(index) + ".jpg"
            fragment_img_path = os.path.join(fragment_dir, fragment_img_name)
            with open(fragment_img_path, 'wb') as f:
                fragment_img.save(f)
            fragment.pop("img")
            fragment['fragment_name'] = fragment_img_name

            result['fragment'].append(fragment)

        json_file_name = name + ".json"
        json_file_path = os.path.join(data_dir, json_file_name)
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f)

        log.info("{name} dump success! ".format(name=name))

        return result

    def show(self, draw_rect=False):
        self.render(draw_rect=draw_rect)
        self.bg_img.show("")

    def render(self, draw_rect=False):
        """
        ??????
        :param draw_rect:
        :return:
        """
        for block_group in self.block_group_list:
            block_group.render(draw_rect=draw_rect, on_origin=True)


if __name__ == '__main__':
    # from service import text_img_provider
    # from service.provider.TextImgProvider import text_img_generator

    bg_img_path = "/Users/lijianan/Documents/workspace/github/TextGenerator/data/img/spider_man.jpeg"

    bg_img = Image.open(bg_img_path)
    layout = Layout(bg_img=bg_img, group_box_list=[
        (0, 0, 500, 200),
        (30, 300, 800, 800)
    ])
    layout.gen()
    layout.show(draw_rect=True)

    # # ?????????????????????????????????
    # fp = text_img_provider.next_font_path()
    #
    # # ??????????????????
    # text_img = text_img_generator.gen_text_img(text_img_provider, "hello world", color=const.COLOR_BLUE, font_path=fp)
    #
    # text_block = TextBlock(text_img=text_img, inner_x=200, inner_y=200, margin=40, rotate_angle=15)
    # bg_img_path = "/Users/lijianan/Documents/workspace/github/TextGenerator/data/img/spider_man.jpeg"
    # bg_img = Image.open(bg_img_path)
    # layout = Layout(bg_img=bg_img, block_list=[])
    # layout.preview(draw_rect=True)
