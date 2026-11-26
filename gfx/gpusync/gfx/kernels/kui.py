"""GPU kernel compositing the interface (`UILayer`): alpha boxes, borders and 5×6 bitmap text in 1 dispatch."""

from __future__ import annotations

import gpusync as gs


@gs.kernel
def ui_composite(
    src: gs.u32["h", "w"],
    boxes: gs.i32["nb", "c12"],
    glyphs: gs.u32["rows", "cols"],
    n_boxes: gs.i32,
) -> gs.u32["h", "w"]:
    """Overlays every UI widget (`boxes` + `glyphs`) in a single full-screen GPU pass.

    Layout of each row of `boxes` (12 i32):
      0: x0, 1: y0, 2: x1, 3: y1
      4: bg_rgba (R | G<<8 | B<<16 | A<<24)
      5: border_rgba (A == 0: no border)
      6: glyph_row (< 0: no text)
      7: n_cols (text length in glyphs)
      8: scale (size of a glyph pixel)
      9: text_x, 10: text_y
      11: text_rgba (R | G<<8 | B<<16 | A<<24)
    """
    py, px = gs.index()
    c = src[py, px]
    if n_boxes <= 0:
        return c

    mask = gs.u32(255)
    r = int(c & mask)
    g = int((c >> gs.u32(8)) & mask)
    b = int((c >> gs.u32(16)) & mask)

    for bi in range(boxes.shape[0]):
        if bi < n_boxes:
            x0 = boxes[bi, 0]
            y0 = boxes[bi, 1]
            x1 = boxes[bi, 2]
            y1 = boxes[bi, 3]
            if px >= x0 and px < x1 and py >= y0 and py < y1:
                # Background with alpha transparency (0..255)
                bg = gs.u32(boxes[bi, 4])
                ba = int((bg >> gs.u32(24)) & mask)
                if ba > 0:
                    br = int(bg & mask)
                    bg_g = int((bg >> gs.u32(8)) & mask)
                    bb = int((bg >> gs.u32(16)) & mask)
                    if ba >= 255:
                        r = br
                        g = bg_g
                        b = bb
                    else:
                        inv_a = 255 - ba
                        r = (r * inv_a + br * ba) // 255
                        g = (g * inv_a + bg_g * ba) // 255
                        b = (b * inv_a + bb * ba) // 255

                # Bordure 1 px optionnelle
                bd = gs.u32(boxes[bi, 5])
                bda = int((bd >> gs.u32(24)) & mask)
                if bda > 0:
                    if px == x0 or px == x1 - 1 or py == y0 or py == y1 - 1:
                        bdr = int(bd & mask)
                        bdg = int((bd >> gs.u32(8)) & mask)
                        bdb = int((bd >> gs.u32(16)) & mask)
                        if bda >= 255:
                            r = bdr
                            g = bdg
                            b = bdb
                        else:
                            inv_ba = 255 - bda
                            r = (r * inv_ba + bdr * bda) // 255
                            g = (g * inv_ba + bdg * bda) // 255
                            b = (b * inv_ba + bdb * bda) // 255

                # Texte bitmap 5×6 optionnel
                grow = boxes[bi, 6]
                if grow >= 0:
                    ncols = boxes[bi, 7]
                    sc = boxes[bi, 8]
                    tx = px - boxes[bi, 9]
                    ty = py - boxes[bi, 10]
                    if tx >= 0 and ty >= 0 and sc > 0:
                        cell_w = 6 * sc
                        cell_h = 8 * sc
                        col = tx // cell_w
                        if col < ncols and ty < cell_h:
                            gx = (tx - col * cell_w) // sc
                            gy = ty // sc
                            if gx < 5 and gy < 6:
                                if ((glyphs[grow, col] >> gs.u32(gy * 5 + gx)) & gs.u32(1)) != gs.u32(0):
                                    tc = gs.u32(boxes[bi, 11])
                                    r = int(tc & mask)
                                    g = int((tc >> gs.u32(8)) & mask)
                                    b = int((tc >> gs.u32(16)) & mask)

    return gs.u32(r) | (gs.u32(g) << gs.u32(8)) | (gs.u32(b) << gs.u32(16)) | gs.u32(4278190080)
