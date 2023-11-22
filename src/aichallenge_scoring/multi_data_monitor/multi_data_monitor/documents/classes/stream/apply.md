# Apply Stream

## Description

データにフィルターを適用するストリームを作成します。

## Format

| Name  | Type                          | Required | Description                                                  |
| ----- | ----------------------------- | -------- | ------------------------------------------------------------ |
| class | string                        | yes      | プラグインの固有名称である `apply` を指定します。            |
| rules | [Filter](../common/filter.md) | no       | 適用するフィルターを指定します。省略した場合は何もしません。 |
