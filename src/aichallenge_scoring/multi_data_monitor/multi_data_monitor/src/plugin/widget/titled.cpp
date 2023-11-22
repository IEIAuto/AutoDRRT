// Copyright 2022 Takagi, Isamu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <multi_data_monitor/widget.hpp>
#include <QLabel>
#include <QVBoxLayout>
#include <QVariant>
#include <QWidget>

namespace multi_data_monitor
{

class Titled : public BasicWidget
{
protected:
  void setup(YAML::Node yaml, const std::vector<QWidget *> & items) override;
  void apply(YAML::Node yaml) override;

private:
  QLabel * value_;
  QLabel * title_;
};

void Titled::setup(YAML::Node yaml, const std::vector<QWidget *> &)
{
  value_ = new QLabel();
  title_ = new QLabel(QString::fromStdString(yaml["title"].as<std::string>("")));
  value_->setAlignment(Qt::AlignCenter);
  title_->setAlignment(Qt::AlignCenter);

  auto * layout = new QVBoxLayout();
  layout->addWidget(value_);
  layout->addWidget(title_);
  layout->setSpacing(0);
  layout->setContentsMargins(0, 0, 0, 0);

  auto * widget = new QWidget();
  widget->setToolTip(QString::fromStdString(yaml["notes"].as<std::string>("")));
  widget->setLayout(layout);
  register_root_widget(widget);
  register_stylesheet_widget(value_, "value");
  register_stylesheet_widget(title_, "title");
}

void Titled::apply(YAML::Node yaml)
{
  const auto value = QString::fromStdString(yaml.as<std::string>());
  value_->setText(value);
}

}  // namespace multi_data_monitor

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(multi_data_monitor::Titled, multi_data_monitor::BasicWidget)
