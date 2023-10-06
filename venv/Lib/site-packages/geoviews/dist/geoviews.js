/*!
 * Copyright (c) 2012 - 2023, Anaconda, Inc., and Bokeh Contributors
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * Neither the name of Anaconda nor the names of any contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */
(function(root, factory) {
  factory(root["Bokeh"], undefined);
})(this, function(Bokeh, version) {
  let define;
  return (function(modules, entry, aliases, externals) {
    const bokeh = typeof Bokeh !== "undefined" && (version != null ? Bokeh[version] : Bokeh);
    if (bokeh != null) {
      return bokeh.register_plugin(modules, entry, aliases);
    } else {
      throw new Error("Cannot find Bokeh " + version + ". You have to load it prior to loading plugins.");
    }
  })
({
"c764d38756": /* index.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    const tslib_1 = require("tslib");
    const GeoViews = tslib_1.__importStar(require("b4555bea44") /* ./models */);
    exports.GeoViews = GeoViews;
    const base_1 = require("@bokehjs/base");
    (0, base_1.register_models)(GeoViews);
},
"b4555bea44": /* models/index.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    var checkpoint_tool_1 = require("1d7c6c8516") /* ./checkpoint_tool */;
    __esExport("CheckpointTool", checkpoint_tool_1.CheckpointTool);
    var clear_tool_1 = require("1c15765006") /* ./clear_tool */;
    __esExport("ClearTool", clear_tool_1.ClearTool);
    var poly_draw_1 = require("cddc7627cc") /* ./poly_draw */;
    __esExport("PolyVertexDrawTool", poly_draw_1.PolyVertexDrawTool);
    var poly_edit_1 = require("8837add128") /* ./poly_edit */;
    __esExport("PolyVertexEditTool", poly_edit_1.PolyVertexEditTool);
    var restore_tool_1 = require("92294e6ad2") /* ./restore_tool */;
    __esExport("RestoreTool", restore_tool_1.RestoreTool);
},
"1d7c6c8516": /* models/checkpoint_tool.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    var _a;
    const array_1 = require("@bokehjs/core/util/array");
    const action_tool_1 = require("@bokehjs/models/tools/actions/action_tool");
    const column_data_source_1 = require("@bokehjs/models/sources/column_data_source");
    const icons_css_1 = require("@bokehjs/styles/icons.css");
    class CheckpointToolView extends action_tool_1.ActionToolView {
        doit() {
            const sources = this.model.sources;
            for (const source of sources) {
                if (!source.buffer) {
                    source.buffer = [];
                }
                let data_copy = {};
                for (const key in source.data) {
                    const column = source.data[key];
                    const new_column = [];
                    for (const arr of column) {
                        if (Array.isArray(arr) || (ArrayBuffer.isView(arr))) {
                            new_column.push((0, array_1.copy)(arr));
                        }
                        else {
                            new_column.push(arr);
                        }
                    }
                    data_copy[key] = new_column;
                }
                source.buffer.push(data_copy);
            }
        }
    }
    exports.CheckpointToolView = CheckpointToolView;
    CheckpointToolView.__name__ = "CheckpointToolView";
    class CheckpointTool extends action_tool_1.ActionTool {
        constructor(attrs) {
            super(attrs);
            this.tool_name = "Checkpoint";
            this.tool_icon = icons_css_1.tool_icon_save;
        }
    }
    exports.CheckpointTool = CheckpointTool;
    _a = CheckpointTool;
    CheckpointTool.__name__ = "CheckpointTool";
    CheckpointTool.__module__ = "geoviews.models.custom_tools";
    (() => {
        _a.prototype.default_view = CheckpointToolView;
        _a.define(({ Array, Ref }) => ({
            sources: [Array(Ref(column_data_source_1.ColumnDataSource)), []],
        }));
    })();
},
"1c15765006": /* models/clear_tool.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    var _a;
    const action_tool_1 = require("@bokehjs/models/tools/actions/action_tool");
    const column_data_source_1 = require("@bokehjs/models/sources/column_data_source");
    const icons_css_1 = require("@bokehjs/styles/icons.css");
    class ClearToolView extends action_tool_1.ActionToolView {
        doit() {
            for (var source of this.model.sources) {
                for (const column in source.data) {
                    source.data[column] = [];
                }
                source.change.emit();
                source.properties.data.change.emit();
            }
        }
    }
    exports.ClearToolView = ClearToolView;
    ClearToolView.__name__ = "ClearToolView";
    class ClearTool extends action_tool_1.ActionTool {
        constructor(attrs) {
            super(attrs);
            this.tool_name = "Clear data";
            this.tool_icon = icons_css_1.tool_icon_reset;
        }
    }
    exports.ClearTool = ClearTool;
    _a = ClearTool;
    ClearTool.__name__ = "ClearTool";
    ClearTool.__module__ = "geoviews.models.custom_tools";
    (() => {
        _a.prototype.default_view = ClearToolView;
        _a.define(({ Array, Ref }) => ({
            sources: [Array(Ref(column_data_source_1.ColumnDataSource)), []],
        }));
    })();
},
"cddc7627cc": /* models/poly_draw.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    var _a;
    const object_1 = require("@bokehjs/core/util/object");
    const types_1 = require("@bokehjs/core/util/types");
    const poly_draw_tool_1 = require("@bokehjs/models/tools/edit/poly_draw_tool");
    class PolyVertexDrawToolView extends poly_draw_tool_1.PolyDrawToolView {
        _split_path(x, y) {
            for (let r = 0; r < this.model.renderers.length; r++) {
                const renderer = this.model.renderers[r];
                const glyph = renderer.glyph;
                const cds = renderer.data_source;
                const [xkey, ykey] = [glyph.xs.field, glyph.ys.field];
                const xpaths = cds.data[xkey];
                const ypaths = cds.data[ykey];
                for (let index = 0; index < xpaths.length; index++) {
                    let xs = xpaths[index];
                    if (!(0, types_1.isArray)(xs)) {
                        xs = Array.from(xs);
                        cds.data[xkey][index] = xs;
                    }
                    let ys = ypaths[index];
                    if (!(0, types_1.isArray)(ys)) {
                        ys = Array.from(ys);
                        cds.data[ykey][index] = ys;
                    }
                    for (let i = 0; i < xs.length; i++) {
                        if ((xs[i] == x) && (ys[i] == y) && (i != 0) && (i != (xs.length - 1))) {
                            xpaths.splice(index + 1, 0, xs.slice(i));
                            ypaths.splice(index + 1, 0, ys.slice(i));
                            xs.splice(i + 1);
                            ys.splice(i + 1);
                            for (const column of cds.columns()) {
                                if ((column !== xkey) && (column != ykey))
                                    cds.data[column].splice(index + 1, 0, cds.data[column][index]);
                            }
                            return;
                        }
                    }
                }
            }
        }
        _snap_to_vertex(ev, x, y) {
            if (this.model.vertex_renderer) {
                // If an existing vertex is hit snap to it
                const vertex_selected = this._select_event(ev, "replace", [this.model.vertex_renderer]);
                const point_ds = this.model.vertex_renderer.data_source;
                // Type once dataspecs are typed
                const point_glyph = this.model.vertex_renderer.glyph;
                const [pxkey, pykey] = [point_glyph.x.field, point_glyph.y.field];
                if (vertex_selected.length) {
                    // If existing vertex is hit split path at that location
                    // converting to feature vertex
                    const index = point_ds.selected.indices[0];
                    if (pxkey)
                        x = point_ds.data[pxkey][index];
                    if (pykey)
                        y = point_ds.data[pykey][index];
                    if (ev.type != 'mousemove')
                        this._split_path(x, y);
                    point_ds.selection_manager.clear();
                }
            }
            return [x, y];
        }
        _set_vertices(xs, ys, styles) {
            if (this.model.vertex_renderer == null)
                return;
            const point_glyph = this.model.vertex_renderer.glyph;
            const point_cds = this.model.vertex_renderer.data_source;
            const [pxkey, pykey] = [point_glyph.x.field, point_glyph.y.field];
            if (pxkey) {
                if ((0, types_1.isArray)(xs))
                    point_cds.data[pxkey] = xs;
                else
                    point_glyph.x = { value: xs };
            }
            if (pykey) {
                if ((0, types_1.isArray)(ys))
                    point_cds.data[pykey] = ys;
                else
                    point_glyph.y = { value: ys };
            }
            if (styles != null) {
                for (const key of (0, object_1.keys)(styles)) {
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
        _show_vertices() {
            if (!this.model.active) {
                return;
            }
            const xs = [];
            const ys = [];
            const styles = {};
            for (const key of (0, object_1.keys)(this.model.end_style))
                styles[key] = [];
            for (let i = 0; i < this.model.renderers.length; i++) {
                const renderer = this.model.renderers[i];
                const cds = renderer.data_source;
                const glyph = renderer.glyph;
                const [xkey, ykey] = [glyph.xs.field, glyph.ys.field];
                for (const array of cds.get_array(xkey)) {
                    Array.prototype.push.apply(xs, array);
                    for (const key of (0, object_1.keys)(this.model.end_style))
                        styles[key].push(this.model.end_style[key]);
                    for (const key of (0, object_1.keys)(this.model.node_style)) {
                        for (let index = 0; index < (array.length - 2); index++) {
                            styles[key].push(this.model.node_style[key]);
                        }
                    }
                    for (const key of (0, object_1.keys)(this.model.end_style))
                        styles[key].push(this.model.end_style[key]);
                }
                for (const array of cds.get_array(ykey))
                    Array.prototype.push.apply(ys, array);
                if (this._drawing && (i == (this.model.renderers.length - 1))) {
                    // Skip currently drawn vertex
                    xs.splice(xs.length - 1, 1);
                    ys.splice(ys.length - 1, 1);
                    for (const key of (0, object_1.keys)(styles))
                        styles[key].splice(styles[key].length - 1, 1);
                }
            }
            this._set_vertices(xs, ys, styles);
        }
        _remove() {
            const renderer = this.model.renderers[0];
            const cds = renderer.data_source;
            const glyph = renderer.glyph;
            const [xkey, ykey] = [glyph.xs.field, glyph.ys.field];
            if (xkey) {
                const xidx = cds.data[xkey].length - 1;
                const xs = cds.get_array(xkey)[xidx];
                xs.splice(xs.length - 1, 1);
                if (xs.length == 1)
                    cds.data[xkey].splice(xidx, 1);
            }
            if (ykey) {
                const yidx = cds.data[ykey].length - 1;
                const ys = cds.get_array(ykey)[yidx];
                ys.splice(ys.length - 1, 1);
                if (ys.length == 1)
                    cds.data[ykey].splice(yidx, 1);
            }
            this._emit_cds_changes(cds);
            this._drawing = false;
            this._show_vertices();
        }
    }
    exports.PolyVertexDrawToolView = PolyVertexDrawToolView;
    PolyVertexDrawToolView.__name__ = "PolyVertexDrawToolView";
    class PolyVertexDrawTool extends poly_draw_tool_1.PolyDrawTool {
        constructor(attrs) {
            super(attrs);
        }
    }
    exports.PolyVertexDrawTool = PolyVertexDrawTool;
    _a = PolyVertexDrawTool;
    PolyVertexDrawTool.__name__ = "PolyVertexDrawTool";
    PolyVertexDrawTool.__module__ = "geoviews.models.custom_tools";
    (() => {
        _a.prototype.default_view = PolyVertexDrawToolView;
        _a.define(({ Any }) => ({
            end_style: [Any, {}],
            node_style: [Any, {}],
        }));
    })();
},
"8837add128": /* models/poly_edit.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    var _a;
    const object_1 = require("@bokehjs/core/util/object");
    const types_1 = require("@bokehjs/core/util/types");
    const poly_edit_tool_1 = require("@bokehjs/models/tools/edit/poly_edit_tool");
    class PolyVertexEditToolView extends poly_edit_tool_1.PolyEditToolView {
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
                if ((0, types_1.isArray)(xs))
                    point_cds.data[pxkey] = xs;
                else
                    point_glyph.x = { value: xs };
            }
            if (pykey) {
                if ((0, types_1.isArray)(ys))
                    point_cds.data[pykey] = ys;
                else
                    point_glyph.y = { value: ys };
            }
            if (styles != null) {
                for (const key of (0, object_1.keys)(styles)) {
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
                if (!(0, types_1.isArray)(xs))
                    cds.data[xkey][index] = xs = Array.from(xs);
            }
            else {
                xs = glyph.xs.value;
            }
            if (ykey) {
                ys = cds.data[ykey][index];
                if (!(0, types_1.isArray)(ys))
                    cds.data[ykey][index] = ys = Array.from(ys);
            }
            else {
                ys = glyph.ys.value;
            }
            const styles = {};
            for (const key of (0, object_1.keys)(this.model.end_style))
                styles[key] = [this.model.end_style[key]];
            for (const key of (0, object_1.keys)(this.model.node_style)) {
                for (let index = 0; index < (xs.length - 2); index++) {
                    styles[key].push(this.model.node_style[key]);
                }
            }
            for (const key of (0, object_1.keys)(this.model.end_style))
                styles[key].push(this.model.end_style[key]);
            this._selected_renderer = renderer;
            this._set_vertices(xs, ys, styles);
        }
    }
    exports.PolyVertexEditToolView = PolyVertexEditToolView;
    PolyVertexEditToolView.__name__ = "PolyVertexEditToolView";
    class PolyVertexEditTool extends poly_edit_tool_1.PolyEditTool {
        constructor(attrs) {
            super(attrs);
        }
    }
    exports.PolyVertexEditTool = PolyVertexEditTool;
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
},
"92294e6ad2": /* models/restore_tool.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    var _a;
    const action_tool_1 = require("@bokehjs/models/tools/actions/action_tool");
    const column_data_source_1 = require("@bokehjs/models/sources/column_data_source");
    const icons_css_1 = require("@bokehjs/styles/icons.css");
    class RestoreToolView extends action_tool_1.ActionToolView {
        doit() {
            const sources = this.model.sources;
            for (const source of sources) {
                if (!source.buffer || (source.buffer.length == 0)) {
                    continue;
                }
                source.data = source.buffer.pop();
                source.change.emit();
                source.properties.data.change.emit();
            }
        }
    }
    exports.RestoreToolView = RestoreToolView;
    RestoreToolView.__name__ = "RestoreToolView";
    class RestoreTool extends action_tool_1.ActionTool {
        constructor(attrs) {
            super(attrs);
            this.tool_name = "Restore";
            this.tool_icon = icons_css_1.tool_icon_undo;
        }
    }
    exports.RestoreTool = RestoreTool;
    _a = RestoreTool;
    RestoreTool.__name__ = "RestoreTool";
    RestoreTool.__module__ = "geoviews.models.custom_tools";
    (() => {
        _a.prototype.default_view = RestoreToolView;
        _a.define(({ Array, Ref }) => ({
            sources: [Array(Ref(column_data_source_1.ColumnDataSource)), []],
        }));
    })();
},
}, "c764d38756", {"index":"c764d38756","models/index":"b4555bea44","models/checkpoint_tool":"1d7c6c8516","models/clear_tool":"1c15765006","models/poly_draw":"cddc7627cc","models/poly_edit":"8837add128","models/restore_tool":"92294e6ad2"}, {});});
//# sourceMappingURL=geoviews.js.map
