# Function Filter

## Description

複数のフィルターを逐次実行するフィルターを作成します。

## Format

| Name  | Type                                  | Required | Description                                          |
| ----- | ------------------------------------- | -------- | ---------------------------------------------------- |
| class | string                                | yes      | プラグインの固有名称である `function` を指定します。 |
| rules | list of [Filter](../common/filter.md) | yes      | フィルターを指定します。                             |

# Note

フィルターが必要な文脈にて配列型を指定すると、システム内部で自動的にこのフィルターが生成されます。
指定した配列は `rules` に指定されたものとして扱われます。
