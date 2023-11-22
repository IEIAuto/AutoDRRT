# Field Stream

## Description

このストリームをユーザーが直接使用することはできません。
システム内部で [subscription](./subscription.md) を扱うために使用されています。

## Format

| Name  | Type   | Required | Description                                              |
| ----- | ------ | -------- | -------------------------------------------------------- |
| class | string | yes      | プラグインの固有名称である `@field` が指定されています。 |
| name  | string | yes      | フィールドが指定されています。                           |
| type  | string | no       | フィールドの型が指定されています。                       |
