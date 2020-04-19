import altair as alt


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
