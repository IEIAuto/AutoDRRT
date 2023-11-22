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

namespace multi_data_monitor
{

class Simple : public BasicWidget
{
protected:
  void setup(YAML::Node yaml, const std::vector<QWidget *> & items) override;
  void apply(YAML::Node yaml) override;

private:
  bool is_constant_;
  std::string title_;
  QLabel * label_;
};

void Simple::setup(YAML::Node yaml, const std::vector<QWidget *> &)
{
  std::string initial_value;
  if (yaml["const"])
  {
    is_constant_ = true;
    initial_value = yaml["const"].as<std::string>("");
  }
  else
  {
    is_constant_ = false;
    initial_value = yaml["value"].as<std::string>("");
  }
  title_ = yaml["title"].as<std::string>("");
  title_ = title_.empty() ? "" : title_ + ": ";

  label_ = new QLabel(QString::fromStdString(title_ + initial_value));
  label_->setAlignment(Qt::AlignCenter);
  label_->setToolTip(QString::fromStdString(yaml["notes"].as<std::string>("")));

  register_root_widget(label_);
  register_stylesheet_widget(label_);
}

void Simple::apply(YAML::Node yaml)
{
  if (!is_constant_)
  {
    const auto value = yaml.as<std::string>();
    label_->setText(QString::fromStdString(title_ + value));
  }
}

}  // namespace multi_data_monitor

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(multi_data_monitor::Simple, multi_data_monitor::BasicWidget)
