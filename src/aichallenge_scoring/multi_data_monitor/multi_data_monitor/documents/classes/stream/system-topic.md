# Topic Stream

## Description

このストリームをユーザーが直接使用することはできません。
システム内部で [subscription](./subscription.md) を扱うために使用されています。

## Format

| Name  | Type   | Required | Description                                              |
| ----- | ------ | -------- | -------------------------------------------------------- |
| class | string | yes      | プラグインの固有名称である `@topic` が指定されています。 |
| name  | string | yes      | トピックの名前が指定されています。                       |
| type  | string | no       | トピックの型が指定されています。                         |
| qos   | string | no       | トピックの QoS が指定されています。                      |
