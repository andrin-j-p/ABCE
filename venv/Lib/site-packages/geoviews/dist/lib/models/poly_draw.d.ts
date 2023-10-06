import * as p from "@bokehjs/core/properties";
import { UIEvent } from "@bokehjs/core/ui_events";
import { PolyDrawTool, PolyDrawToolView } from "@bokehjs/models/tools/edit/poly_draw_tool";
import { MultiLine } from "@bokehjs/models/glyphs/multi_line";
import { Patches } from "@bokehjs/models/glyphs/patches";
import { GlyphRenderer } from "@bokehjs/models/renderers/glyph_renderer";
export declare class PolyVertexDrawToolView extends PolyDrawToolView {
    model: PolyVertexDrawTool;
    _split_path(x: number, y: number): void;
    _snap_to_vertex(ev: UIEvent, x: number, y: number): [number, number];
    _set_vertices(xs: number[] | number, ys: number[] | number, styles?: any): void;
    _show_vertices(): void;
    _remove(): void;
}
export declare namespace PolyVertexDrawTool {
    type Attrs = p.AttrsOf<Props>;
    type Props = PolyDrawTool.Props & {
        node_style: p.Property<any>;
        end_style: p.Property<any>;
    };
}
export interface PolyVertexDrawTool extends PolyVertexDrawTool.Attrs {
}
export interface HasPolyGlyph {
    glyph: MultiLine | Patches;
}
export declare class PolyVertexDrawTool extends PolyDrawTool {
    properties: PolyVertexDrawTool.Props;
    renderers: (GlyphRenderer & HasPolyGlyph)[];
    constructor(attrs?: Partial<PolyVertexDrawTool.Attrs>);
    static __module__: string;
}
