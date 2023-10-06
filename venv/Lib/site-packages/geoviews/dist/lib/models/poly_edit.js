var _a;
import { keys } from "@bokehjs/core/util/object";
import { isArray } from "@bokehjs/core/util/types";
import { PolyEditTool, PolyEditToolView } from "@bokehjs/models/tools/edit/poly_edit_tool";
export class PolyVertexEditToolView extends PolyEditToolView {
    deactivate() {
        this._hide_vertices();
        if (!this._selected_renderer) {
            return;
        }
        else if (this._drawing) {
            this._remove_vertex();
            this._drawing = false;
        }
        this._emit_cds_changes(this._selected_renderer.data_source, false, true, false);
    }
    _pan(ev) {
        if (this._basepoint == null || this.model.vertex_renderer == null)
            return;
        const points = this._drag_points(ev, [this.model.vertex_renderer]);
        if (!ev.modifiers.shift) {
            this._move_linked(points);
        }
        if (this._selected_renderer)
            this._selected_renderer.data_source.change.emit();
    }
    _pan_end(ev) {
        if (this._basepoint == null || this.model.vertex_renderer == null)
            return;
        const points = this._drag_points(ev, [this.model.vertex_renderer]);
        if (!ev.modifiers.shift) {
            this._move_linked(points);
        }
        this._emit_cds_changes(this.model.vertex_renderer.data_source, false, true, true);
        if (this._selected_renderer) {
            this._emit_cds_changes(this._selected_renderer.data_source);
        }
        this._basepoint = null;
    }
    _drag_points(ev, renderers) {
        if (this._basepoint == null)
            return [];
        const [bx, by] = this._basepoint;
        const points = [];
        for (const renderer of renderers) {
            const basepoint = this._map_drag(bx, by, renderer);
            const point = this._map_drag(ev.sx, ev.sy, renderer);
            if (point == null || basepoint == null) {
                continue;
            }
            const [x, y] = point;
            const [px, py] = basepoint;
            const [dx, dy] = [x - px, y - py];
            // Type once dataspecs are typed
            const glyph = renderer.glyph;
            const cds = renderer.data_source;
            const [xkey, ykey] = [glyph.x.field, glyph.y.field];
            for (const index of cds.selected.indices) {
                const point = [];
                if (xkey) {
                    point.push(cds.data[xkey][index]);
                    cds.data[xkey][index] += dx;
                }
                if (ykey) {
                    point.push(cds.data[ykey][index]);
                    cds.data[ykey][index] += dy;
                }
                point.push(dx);
                point.push(dy);
                points.push(point);
            }
            cds.change.emit();
        }
        this._basepoint = [ev.sx, ev.sy];
        return points;
    }
    _set_vertices(xs, ys, styles) {
        if (this.model.vertex_renderer == null)
            return;
        const point_glyph = this.model.vertex_renderer.glyph;
        const point_cds = this.model.vertex_renderer.data_source;
        const [pxkey, pykey] = [point_glyph.x.field, point_glyph.y.field];
        if (pxkey) {
            if (isArray(xs))
                point_cds.data[pxkey] = xs;
            else
                point_glyph.x = { value: xs };
        }
        if (pykey) {
            if (isArray(ys))
                point_cds.data[pykey] = ys;
            else
                point_glyph.y = { value: ys };
        }
        if (styles != null) {
            for (const key of keys(styles)) {
                point_cds.data[key] = styles[key];
                point_glyph[key] = { field: key };
            }
        }
        else {
            for (const col of point_cds.columns()) {
                point_cds.data[col] = [];
            }
        }
        this._emit_cds_changes(point_cds, true, true, false);
    }
    _move_linked(points) {
        if (!this._selected_renderer)
            return;
        const renderer = this._selected_renderer;
        const glyph = renderer.glyph;
        const cds = renderer.data_source;
        const [xkey, ykey] = [glyph.xs.field, glyph.ys.field];
        const xpaths = cds.data[xkey];
        const ypaths = cds.data[ykey];
        for (const point of points) {
            const [x, y, dx, dy] = point;
            for (let index = 0; index < xpaths.length; index++) {
                const xs = xpaths[index];
                const ys = ypaths[index];
                for (let i = 0; i < xs.length; i++) {
                    if ((xs[i] == x) && (ys[i] == y)) {
                        xs[i] += dx;
                        ys[i] += dy;
                    }
                }
            }
        }
    }
    _tap(ev) {
        if (this.model.vertex_renderer == null)
            return;
        const renderer = this.model.vertex_renderer;
        const point = this._map_drag(ev.sx, ev.sy, renderer);
        if (point == null)
            return;
        else if (this._drawing && this._selected_renderer) {
            let [x, y] = point;
            const cds = renderer.data_source;
            // Type once dataspecs are typed
            const glyph = renderer.glyph;
            const [xkey, ykey] = [glyph.x.field, glyph.y.field];
            const indices = cds.selected.indices;
            [x, y] = this._snap_to_vertex(ev, x, y);
            const index = indices[0];
            cds.selected.indices = [index + 1];
            if (xkey) {
                const xs = cds.get_array(xkey);
                const nx = xs[index];
                xs[index] = x;
                xs.splice(index + 1, 0, nx);
            }
            if (ykey) {
                const ys = cds.get_array(ykey);
                const ny = ys[index];
                ys[index] = y;
                ys.splice(index + 1, 0, ny);
            }
            cds.change.emit();
            this._emit_cds_changes(this._selected_renderer.data_source, true, false, true);
            return;
        }
        this._select_event(ev, this._select_mode(ev), [renderer]);
    }
    _show_vertices(ev) {
        if (!this.model.active)
            return;
        const renderers = this._select_event(ev, "replace", this.model.renderers);
        if (!renderers.length) {
            this._hide_vertices();
            this._selected_renderer = null;
            this._drawing = false;
            return;
        }
        const renderer = renderers[0];
        const glyph = renderer.glyph;
        const cds = renderer.data_source;
        const index = cds.selected.indices[0];
        const [xkey, ykey] = [glyph.xs.field, glyph.ys.field];
        let xs;
        let ys;
        if (xkey) {
            xs = cds.data[xkey][index];
            if (!isArray(xs))
                cds.data[xkey][index] = xs = Array.from(xs);
        }
        else {
            xs = glyph.xs.value;
        }
        if (ykey) {
            ys = cds.data[ykey][index];
            if (!isArray(ys))
                cds.data[ykey][index] = ys = Array.from(ys);
        }
        else {
            ys = glyph.ys.value;
        }
        const styles = {};
        for (const key of keys(this.model.end_style))
            styles[key] = [this.model.end_style[key]];
        for (const key of keys(this.model.node_style)) {
            for (let index = 0; index < (xs.length - 2); index++) {
                styles[key].push(this.model.node_style[key]);
            }
        }
        for (const key of keys(this.model.end_style))
            styles[key].push(this.model.end_style[key]);
        this._selected_renderer = renderer;
        this._set_vertices(xs, ys, styles);
    }
}
PolyVertexEditToolView.__name__ = "PolyVertexEditToolView";
export class PolyVertexEditTool extends PolyEditTool {
    constructor(attrs) {
        super(attrs);
    }
}
_a = PolyVertexEditTool;
PolyVertexEditTool.__name__ = "PolyVertexEditTool";
PolyVertexEditTool.__module__ = "geoviews.models.custom_tools";
(() => {
    _a.prototype.default_view = PolyVertexEditToolView;
    _a.define(({ Any }) => ({
        end_style: [Any, {}],
        node_style: [Any, {}],
    }));
})();
//# sourceMappingURL=poly_edit.js.map