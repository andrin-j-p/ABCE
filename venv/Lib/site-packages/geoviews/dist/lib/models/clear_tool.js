var _a;
import { ActionTool, ActionToolView } from "@bokehjs/models/tools/actions/action_tool";
import { ColumnDataSource } from "@bokehjs/models/sources/column_data_source";
import { tool_icon_reset } from "@bokehjs/styles/icons.css";
export class ClearToolView extends ActionToolView {
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
ClearToolView.__name__ = "ClearToolView";
export class ClearTool extends ActionTool {
    constructor(attrs) {
        super(attrs);
        this.tool_name = "Clear data";
        this.tool_icon = tool_icon_reset;
    }
}
_a = ClearTool;
ClearTool.__name__ = "ClearTool";
ClearTool.__module__ = "geoviews.models.custom_tools";
(() => {
    _a.prototype.default_view = ClearToolView;
    _a.define(({ Array, Ref }) => ({
        sources: [Array(Ref(ColumnDataSource)), []],
    }));
})();
//# sourceMappingURL=clear_tool.js.map