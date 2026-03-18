### 1. json的语法规则是什么?
- json的数据有**键值对** 构成
- json的顶层结构一般是**一个对象** 或 **一个数组** 
- 其中，键必须是 **字符串** 
- 值可以是： **对象、数组、字符串、数字、布尔值或空值null** 

除此之外，还有
- 必须使用双引号（字符串）
- 不能有注释

还有还有，还没完
- **对象**：无序的键值对集合，以 `{` 开始，以 `}` 结束。键值对之间用逗号分隔，但最后一个成员后不能有逗号
- **数组**：有序的值列表，以 `[` 开始，以 `]` 结束。值之间用逗号分隔，最后一个值后不能有逗号

### 2. 在Python中如何使用json呢？
导入json库 `import json` 

### 3. josn库的常用方法有什么？
#### （1）json.dumps() —— 将Python对象转换成**JSON字符串** 
```python
json_str = json.dumps(data, indent=4, ensure_ascii=False)
```

参数：
indent 缩进，美观
ensure_ascii 是否用ASCII码

#### （2）json.dump() —— 将Python对象转换成JSON字符串并**直接写入文件**
```python
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
```

#### （3）json.loads() —— 将**JSON字符串**转换成Python对象
```python
python_data = json.loads(json_str)
```

#### （4）json.load() —— 从**文件中读取JSON字符串**并转换成Python对象
```python
with open('data.json', 'r', encoding='utf-8') as f:
    python_data = json.load(f)
```

