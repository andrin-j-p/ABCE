var _a;
import { ActionTool, ActionToolView } from "@bokehjs/models/tools/actions/action_tool";
import { ColumnDataSource } from "@bokehjs/models/sources/column_data_source";
import { tool_icon_undo } from "@bokehjs/styles/icons.css";
export class RestoreToolView extends ActionToolView {
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
RestoreToolView.__name__ = "RestoreToolView";
export class RestoreTool extends ActionTool {
    constructor(attrs) {
        super(attrs);
        this.tool_name = "Restore";
        this.tool_icon = tool_icon_undo;
    }
}
_a = RestoreTool;
RestoreTool.__name__ = "RestoreTool";
RestoreTool.__module__ = "geoviews.models.custom_tools";
(() => {
    _a.prototype.default_view = RestoreToolView;
    _a.define(({ Array, Ref }) => ({
        sources: [Array(Ref(ColumnDataSource)), []],
    }));
})();
//# sourceMappingURL=restore_tool.js.map