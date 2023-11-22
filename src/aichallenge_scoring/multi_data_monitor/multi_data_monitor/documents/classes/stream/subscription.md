# Subscription Stream

## Description

ROS のトピックを購読するストリームを作成します。

## Format

| Name       | Type   | Required | Description                                                                |
| ---------- | ------ | -------- | -------------------------------------------------------------------------- |
| class      | string | yes      | プラグインの固有名称である `subscription` を指定します。                   |
| topic      | string | yes      | トピックの名前を指定します。                                               |
| field      | string | yes      | フィールドの名前を指定します。空文字列にするとデータ全体だと解釈されます。 |
| qos        | string | no       | トピックの QoS を指定します。フォーマットについては後述します。            |
| topic-type | string | no       | トピックの型を指定します。                                                 |
| field-type | string | no       | フィールドの型を指定します。現在は使用されていません。                     |

## QoS

トピックの QoS は三文字以上の文字列で表現されます。最初の文字は reliability を、次の文字は durability を表します。
三文字目以降は depth を示す数字で、全体として `dd5` や `rt1` のような文字列になります。各文字の意味については以下のテーブルを参照してください。

| Reliability | Description    | Durability | Description     |
| ----------- | -------------- | ---------- | --------------- |
| d           | system default | d          | system default  |
| r           | reliable       | t          | transient_local |
| b           | best effort    | v          | volatile        |

## Note

トピックの型と QoS の両方が指定されている場合は直ちにトピックの購読が開始されます。
これらの情報が揃っていない場合、トピックの発行まで待機して同じ設定で購読を開始します。

システムの内部では `@topic` と `@field` という２つのストリームに変換して処理されます。
複数の箇所で同名トピックとフィールドが使用されている場合は変換を行う際にマージされます。
