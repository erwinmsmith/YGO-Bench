## Current full-duel decision

You must answer with the `{{REQUIRED_RESPONDER}}` tool. The JSON below is the complete visible tactical state for this decision.

```json
{{STATE_JSON}}
```

Before acting, verify the selected index against the current `decision` object. If exact card text is necessary, call `inspect_card`; after its result, submit exactly one `{{REQUIRED_RESPONDER}}` call.
