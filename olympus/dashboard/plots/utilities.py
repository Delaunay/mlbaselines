import altair as alt
import json

# The set of 26 colors for the Colour Alphabet Project suggested by Paul Green-Armytage in
# "A Colour Alphabet and the Limits of Colour Coding.
# This color set is designed for use with white background.
# from https://graphicdesign.stackexchange.com/questions/3682/where-can-i-find-a-large-palette-set-of-contrasting-colors-for-coloring-many-d
colour_alphabet = None


def _load_colour_alphabet():
    global colour_alphabet

    if colour_alphabet is None:
        colours = [
            (240, 163, 255), (0, 117, 220), (153, 63, 0), (76, 0, 92), (25, 25, 25),
            (0, 92, 49), (43, 206, 72), (255, 204, 153), (128, 128, 128), (148, 255, 181),

            (143, 124, 0), (157, 204, 0), (194, 0, 136), (0, 51, 128), (255, 164, 5),
            (255, 168, 187), (66, 102, 0), (255, 0, 16), (94, 241, 242), (0, 153, 143),

            (224, 255, 102), (116, 10, 255), (153, 0, 0), (255, 255, 128), (255, 255, 0),
            (255, 80, 5)]

        colour_alphabet = [f'rgb({c[0]}, {c[1]}, {c[2]})' for c in colours]


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    rgb = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'


# from https://stackoverflow.com/questions/33295120/how-to-generate-gif-256-colors-palette/33295456#33295456
colors_1024 = None


def _load_1024_colors():
    import os
    global colors_1024

    folder = os.path.dirname(__file__)
    if colors_1024 is None:
        with open(f'{folder}/1024_colors.json', 'r') as file:
            colors_1024 = [hex_to_rgb(c) for c in json.load(file)]


_load_1024_colors()
_load_colour_alphabet()


class AltairMatrix:
    """Matrix plot"""

    def __init__(self, data):
        self.data = data
        self.cols = []
        self.rows = []
        self.grid = []

    def fields(self, *args):
        self.cols = list(args)
        self.rows = list(args)
        self.grid = [[None for _ in self.rows] for _ in self.cols]
        return self

    def diag(self, make_chart):
        for i, r in enumerate(self.rows):
            self.grid[i][i] = make_chart(r)

        return self

    def lower(self, make_chart):
        for i, r in enumerate(self.rows):
            for j, c in enumerate(self.rows):
                if i > j:
                    self.grid[i][j] = make_chart(r, c)

        return self

    def upper(self, make_chart):
        for i, r in enumerate(self.rows):
            for j, c in enumerate(self.cols):
                if j > i:
                    self.grid[i][j] = make_chart(r, c)
        return self

    def render(self):
        chart = alt.vconcat(data=self.data)
        for i, _ in enumerate(self.rows):
            row = alt.hconcat()
            for j, _ in enumerate(self.cols):
                if self.grid[i][j] is not None:
                    row |= self.grid[i][j]
            chart &= row

        return chart
