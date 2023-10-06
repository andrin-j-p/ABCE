var _a;
import { copy } from "@bokehjs/core/util/array";
import { ActionTool, ActionToolView } from "@bokehjs/models/tools/actions/action_tool";
import { ColumnDataSource } from "@bokehjs/models/sources/column_data_source";
import { tool_icon_save } from "@bokehjs/styles/icons.css";
export class CheckpointToolView extends ActionToolView {
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
                        new_column.push(copy(arr));
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
CheckpointToolView.__name__ = "CheckpointToolView";
export class CheckpointTool extends ActionTool {
    constructor(attrs) {
        super(attrs);
        this.tool_name = "Checkpoint";
        this.tool_icon = tool_icon_save;
    }
}
_a = CheckpointTool;
CheckpointTool.__name__ = "CheckpointTool";
CheckpointTool.__module__ = "geoviews.models.custom_tools";
(() => {
    _a.prototype.default_view = CheckpointToolView;
    _a.define(({ Array, Ref }) => ({
        sources: [Array(Ref(ColumnDataSource)), []],
    }));
})();
//# sourceMappingURL=checkpoint_tool.js.map