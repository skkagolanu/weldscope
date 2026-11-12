"""Bound frontend redraw work while keeping the first chunk immediate."""

from time import perf_counter


class StreamDisplay:
    def __init__(self, paint, interval=0.05, clock=perf_counter):
        self.paint = paint
        self.interval = interval
        self.clock = clock
        self.parts = []
        self.last_paint = None
        self.render_count = 0

    def __call__(self, chunk):
        self.parts.append(chunk)
        now = self.clock()
        if self.last_paint is None or now - self.last_paint >= self.interval:
            self.paint("".join(self.parts) + "▌")
            self.render_count += 1
            self.last_paint = self.clock()

    def finish(self):
        text = "".join(self.parts)
        self.paint(text)
        self.render_count += 1
        return text
