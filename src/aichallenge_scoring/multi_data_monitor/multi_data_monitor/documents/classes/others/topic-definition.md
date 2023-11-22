# Topic Definition

## Description

コンフィグファイルの専用セクションにてトピックの定義を行います。購読するトピックを一括管理する場合や、単一トピックの複数フィールドにアクセスする場合に便利です。
このオブジェクトはシステム内部で [subscription](../stream/subscription.md) に変換されるため、各フィールドの詳細についてはそちらを参照してください。

## Topic Format

| Name  | Type          | Required | Description                        |
| ----- | ------------- | -------- | ---------------------------------- |
| name  | string        | yes      | トピックの名前を指定します。       |
| type  | string        | no       | トピックの型を指定します。         |
| qos   | string        | no       | トピックの型を指定します。         |
| field | list of field | yes      | フィールド定義の配列を指定します。 |

## Field Format

| Name  | Type   | Required | Description                                 |
| ----- | ------ | -------- | ------------------------------------------- |
| name  | string | yes      | フィールドの名前を指定します。              |
| type  | string | no       | フィールドの型を指定します。                |
| label | string | no       | ストリームの `label` フィールドと同じです。 |
