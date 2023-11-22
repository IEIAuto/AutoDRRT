# SetIf Filter

## Description

条件が一致した時にデータを変更するフィルターを作成します。
条件判定とデータ変更の詳細については該当ページを参照してください。

## Format

| Name  | Type                                   | Required | Description                                       |
| ----- | -------------------------------------- | -------- | ------------------------------------------------- |
| class | string                                 | yes      | プラグインの固有名称である `SetIf` を指定します。 |
| type  | string                                 | yes      | 条件判定で使用するデータの型を指定します。        |
| -     | [Conditions](../others/conditions.md)  | no       | 条件判定を指定します。                            |
| -     | [ValueAttrs](../others/value-attrs.md) | no       | 条件成立時のデータ変更内容を指定します。          |
