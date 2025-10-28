# 📦 README_DATA.md — Amazon Reviews 2023 (Movies & TV)

Tài liệu mô tả **dữ liệu thực tế** theo kết quả quét schema của bạn.

Nguồn: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw  
Danh mục: **Movies_and_TV**  
Quy mô: ~17.3M reviews (reviews) + ~0.75M metadata (meta)

---

## 📁 Cấu trúc thư mục đề xuất
```
data/
└── raw/
    ├── Movies_and_TV.jsonl
    └── meta_Movies_and_TV.jsonl
```
> Lưu ý: Có thể dùng bản nén `.jsonl.gz` để tiết kiệm dung lượng.

---

## 🧾 Schema thực tế (đã quét)

### 1) Reviews — `Movies_and_TV.jsonl` (17,328,314 records)
| Trường | Kiểu | Ghi chú |
|---|---|---|
| `asin` | string | Mã sản phẩm (ASIN) của review |
| `parent_asin` | string | Nhóm/ASIN cha (dùng để join với meta) |
| `user_id` | string | ID người dùng |
| `text` | string | Nội dung đánh giá |
| `title` | string | Tiêu đề đánh giá |
| `rating` | float | Điểm sao (1.0–5.0) |
| `helpful_vote` | int | Số phiếu “hữu ích” |
| `verified_purchase` | bool | Đánh dấu đã mua |
| `images` | list | Danh sách ảnh đính kèm |
| `timestamp` | int | UNIX epoch (giây) |

### 2) Metadata — `meta_Movies_and_TV.jsonl` (748,224 records)
| Trường | Kiểu (có thể None) | Ghi chú |
|---|---|---|
| `parent_asin` | string | **Khoá chính để join** |
| `title` | str / None | Tên sản phẩm |
| `subtitle` | str / None | Phụ đề |
| `main_category` | str / None | Danh mục chính |
| `categories` | list / None | Mảng danh mục |
| `description` | list / None | Mô tả (thường là list đoạn) |
| `features` | list / None | Tính năng nổi bật |
| `images` | list | Ảnh của sản phẩm |
| `videos` | list | Video của sản phẩm |
| `store` | str / None | Store/brand |
| `price` | float / str / None | Giá (đôi khi ở dạng chuỗi, có ký hiệu) |
| `average_rating` | float / None | Điểm trung bình |
| `rating_number` | int / None | Số lượng đánh giá |
| `details` | dict / None | Thông tin chi tiết (key–value) |
| `author` | dict / None | Tác giả (hiếm) |
| `bought_together` | None | Không có dữ liệu trong tập này |

> **Vì sao có `NoneType`?** Nhiều bản ghi meta thiếu trường → khi parse JSON sẽ là `null` (`None` trong Python). Khi đưa vào Spark, `null` **hợp lệ** và có thể ép về kiểu đích.

---

## 🔗 Khóa join
- **Join key khuyến nghị:** `parent_asin` (có ở cả reviews & meta).  
- `asin` là sản phẩm cụ thể; `parent_asin` gom các biến thể → meta dùng `parent_asin`.

---

## 🧠 Label & đặc trưng gợi ý (cho bài toán đề tài 3)
- **Nhãn (label) baseline:** `is_helpful = 1` nếu `helpful_vote > 2` (có thể thay 0/5 theo EDA).
- **Đặc trưng text:** TF-IDF (1-2 gram), sentiment (VADER), độ dài review (`length`).
- **Đặc trưng meta:** `main_category`, `price`, `average_rating`, `rating_number`, v.v.

---

## 🧪 Đọc nhanh bằng PySpark
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

df_reviews = spark.read.json("data/raw/Movies_and_TV.jsonl")
df_meta    = spark.read.json("data/raw/meta_Movies_and_TV.jsonl")

df_reviews.printSchema()
df_meta.printSchema()
```

---

## ⚙️ Gợi ý chuẩn hoá (bronze → silver)
- Ép `timestamp (int)` → `timestamp` thật: `to_timestamp(col("timestamp").cast("long"))`
- Chuẩn hoá `price`: lọc số trong chuỗi rồi cast `double` (fallback khi không phải float).
- Mảng/dict có thể `None` → ép về `array<string>` / `map<string,string>` với `from_json`.

---

## 📌 Ghi chú
- Dữ liệu rất lớn → nên dùng Spark/HDFS hoặc đọc theo batch/chunks.
- Parquet (snappy) cho tầng **bronze/silver** để tăng tốc đọc/ghi.
- Không chỉnh sửa dữ liệu **raw**; chỉ đọc/append.

Cập nhật: 26-10-2025

----------------------------------------

## Cập nhật:28/10/2025

=== Movies_and_TV.jsonl ===
Records scanned: 17,328,314
asin  |  seen=17,328,314  |  types=str
helpful_vote  |  seen=17,328,314  |  types=int
images  |  seen=17,328,314  |  types=list
parent_asin  |  seen=17,328,314  |  types=str
rating  |  seen=17,328,314  |  types=float
text  |  seen=17,328,314  |  types=str
timestamp  |  seen=17,328,314  |  types=int
title  |  seen=17,328,314  |  types=str
user_id  |  seen=17,328,314  |  types=str
verified_purchase  |  seen=17,328,314  |  types=bool

--- First 3 records from Movies_and_TV.jsonl ---

Record 1:
{
  "rating": 5.0,
  "title": "Five Stars",
  "text": "Amazon, please buy the show! I'm hooked!",
  "images": [],
  "asin": "B013488XFS",
  "parent_asin": "B013488XFS",
  "user_id": "AGGZ357AO26RQZVRLGU4D4N52DZQ",
  "timestamp": 1440385637000,
  "helpful_vote": 0,
  "verified_purchase": true
}

Record 2:
{
  "rating": 5.0,
  "title": "Five Stars",
  "text": "My Kiddos LOVE this show!!",
  "images": [],
  "asin": "B00CB6VTDS",
  "parent_asin": "B00CB6VTDS",
  "user_id": "AGKASBHYZPGTEPO6LWZPVJWB2BVA",
  "timestamp": 1461100610000,
  "helpful_vote": 0,
  "verified_purchase": true
}

Record 3:
{
  "rating": 3.0,
  "title": "Some decent moments...but...",
  "text": "Annabella Sciorra did her character justice with her portrayal of a mentally ill, depressed and traumatized individual who projects much of her inner wounds onto others. The challenges she faces with her father were sensitively portrayed and resonate with understanding and love. The ending really isn't an ending, though and feels like it was abandoned with not enough of a closure but other than that, its a decent movie to sit through if you're the type of person who likes to people-watch or analyze the actions of others. Has an independent-movie feel which is also somewhat comforting.",
  "images": [],
  "asin": "B096Z8Z3R6",
  "parent_asin": "B096Z8Z3R6",
  "user_id": "AG2L7H23R5LLKDKLBEF2Q3L2MVDA",
  "timestamp": 1646271834582,
  "helpful_vote": 0,
  "verified_purchase": true
}

=== meta_Movies_and_TV.jsonl ===
Records scanned: 748,224
author  |  seen=508  |  types=NoneType,dict
average_rating  |  seen=748,224  |  types=NoneType,float
bought_together  |  seen=748,224  |  types=NoneType
categories  |  seen=748,224  |  types=NoneType,list
description  |  seen=748,224  |  types=NoneType,list
details  |  seen=748,224  |  types=NoneType,dict
features  |  seen=748,224  |  types=NoneType,list
images  |  seen=748,224  |  types=list
main_category  |  seen=748,224  |  types=NoneType,str
parent_asin  |  seen=748,224  |  types=str
price  |  seen=748,224  |  types=NoneType,float,str
rating_number  |  seen=748,224  |  types=NoneType,int
store  |  seen=748,224  |  types=NoneType,str
subtitle  |  seen=349,626  |  types=NoneType,str
title  |  seen=748,224  |  types=NoneType,str
videos  |  seen=748,224  |  types=list

--- First 3 records from meta_Movies_and_TV.jsonl ---

Record 1:
{
  "main_category": "Prime Video",
  "title": "Glee",
  "subtitle": "UnentitledUnentitled",
  "average_rating": 4.7,
  "rating_number": 2004,
  "features": [
    "IMDb 6.8",
    "2013",
    "22 episodes",
    "X-Ray",
    "TV-14"
  ],
  "description": [
    "Entering its fourth season, this year the members of New Directions compete amongst themselves to be the \"new Rachel\" and hold auditions to find new students. Meanwhile, the graduating class leaves the comforts of McKinley where Rachel struggles to please her demanding NYADA teacher (Kate Hudson) and Kurt second-guesses his decision to stay in Lima. Four newcomers also join the musical comedy."
  ],
  "price": 22.39,
  "images": [
    {
      "360w": "https://images-na.ssl-images-amazon.com/images/S/pv-target-images/8251ee0b9f888d262cd817a5f1aee0b29ffed56a4535af898b827292f881e169._RI_SX360_FMwebp_.jpg",
      "480w": "https://images-na.ssl-images-amazon.com/images/S/pv-target-images/8251ee0b9f888d262cd817a5f1aee0b29ffed56a4535af898b827292f881e169._RI_SX480_FMwebp_.jpg",
      "720w": "https://images-na.ssl-images-amazon.com/images/S/pv-target-images/8251ee0b9f888d262cd817a5f1aee0b29ffed56a4535af898b827292f881e169._RI_SX720_FMwebp_.jpg",
      "1080w": "https://images-na.ssl-images-amazon.com/images/S/pv-target-images/8251ee0b9f888d262cd817a5f1aee0b29ffed56a4535af898b827292f881e169._RI_SX1080_FMwebp_.jpg",
      "1440w": "https://images-na.ssl-images-amazon.com/images/S/pv-target-images/8251ee0b9f888d262cd817a5f1aee0b29ffed56a4535af898b827292f881e169._RI_SX1440_FMwebp_.jpg",
      "1920w": "https://images-na.ssl-images-amazon.com/images/S/pv-target-images/8251ee0b9f888d262cd817a5f1aee0b29ffed56a4535af898b827292f881e169._RI_SX1920_FMwebp_.jpg",
      "variant": "MAIN"
    }
  ],
  "videos": [],
  "store": null,
  "categories": [
    "Comedy",
    "Drama",
    "Arts, Entertainment, and Culture",
    "Music Videos and Concerts"
  ],
  "details": {
    "Content advisory": [
      "Violence",
      "substance use",
      "alcohol use",
      "smoking",
      "foul language",
      "sexual content"
    ],
    "Audio languages": [
      "English"
    ],
    "Subtitles": [
      "English [CC]"
    ],
    "Directors": [
      "Bradley Buecker",
      "Brad Falchuk",
      "Eric Stoltz",
      "Paris Barclay",
      "Ian Brennan",
      "Ryan Murphy",
      "Alfonso Gomez-Rejon",
      "Elodie Keene",
      "Adam Shankman",
      "Paul McCrane"
    ]
  },
  "parent_asin": "B00ABWKL3I",
  "bought_together": null
}

Record 2:
{
  "main_category": "Prime Video",
  "title": "One Perfect Wedding",
  "subtitle": null,
  "average_rating": 3.0,
  "rating_number": 6,
  "features": [
    "IMDb 6.1",
    "1 h 27 min",
    "2021",
    "ALL"
  ],
  "description": [
    "With her book tour in two weeks and his expanding business plans, Cara and Ben put their long engagement behind them and book the chalet for a small wedding with friends and family. Starring Taylor Cole and Jack Turner."
  ],
  "price": null,
  "images": [
    {
      "360w": "https://images-na.ssl-images-amazon.com/images/S/pv-target-images/79cd935f5fc3d27d45cb244c2a821ecaaef4624741111680d14009660282b63e._RI_SX360_FMwebp_.png",
      "480w": "https://images-na.ssl-images-amazon.com/images/S/pv-target-images/79cd935f5fc3d27d45cb244c2a821ecaaef4624741111680d14009660282b63e._RI_SX480_FMwebp_.png",
      "720w": "https://images-na.ssl-images-amazon.com/images/S/pv-target-images/79cd935f5fc3d27d45cb244c2a821ecaaef4624741111680d14009660282b63e._RI_SX720_FMwebp_.png",
      "1080w": "https://images-na.ssl-images-amazon.com/images/S/pv-target-images/79cd935f5fc3d27d45cb244c2a821ecaaef4624741111680d14009660282b63e._RI_SX1080_FMwebp_.png",
      "1440w": "https://images-na.ssl-images-amazon.com/images/S/pv-target-images/79cd935f5fc3d27d45cb244c2a821ecaaef4624741111680d14009660282b63e._RI_SX1440_FMwebp_.png",
      "1920w": "https://images-na.ssl-images-amazon.com/images/S/pv-target-images/79cd935f5fc3d27d45cb244c2a821ecaaef4624741111680d14009660282b63e._RI_SX1920_FMwebp_.png",
      "variant": "MAIN"
    }
  ],
  "videos": [],
  "store": null,
  "categories": [
    "Comedy",
    "Drama",
    "Romance"
  ],
  "details": {
    "Audio languages": [
      "English"
    ],
    "Subtitles": [
      "English [CC]"
    ],
    "Directors": [
      "Gary Yates"
    ],
    "Producers": [
      "Anthony Fankhauser",
      "Graem Luis",
      "Melinda Wells McCabe",
      "Emily Merlin",
      "Stan Spry",
      "Eric Scott Woods"
    ]
  },
  "parent_asin": "B09WDLJ4HP",
  "bought_together": null
}

Record 3:
{
  "main_category": "Movies & TV",
  "title": "How to Make Animatronic Characters - Organic Mechanics Part 2",
  "average_rating": 5.0,
  "rating_number": 7,
  "features": [],
  "description": [
    "Product Description",
    "In PART TWO of this incredible journey through the world of animatronics, the master of The Character Shop, Rick Lazzarini, not only builds upon the lessons he introduced in Part One, but also shares a host of additional mechanical techniques, as he continues with the construction of his lifelike puma.",
    "Here, Rick creates expressiveness in his animatronic creature by focusing on the movement of the jaw and the snarl of the lips. By the end, the range of emotion on display by the sum of these parts is truly amazing.",
    "WEBCOURSE HIGHLIGHTS",
    "Prepping & setting teeth into the vacuform skull",
    "Prepping & setting teeth into the vacuform skull",
    "Overview of cables, liners & housings",
    "Overview of cables, liners & housings",
    "Mounting & moving the jaw",
    "Mounting & moving the jaw",
    "Snarl of both the upper & lower lips",
    "Snarl of both the upper & lower lips",
    "Programming & troubleshooting ranges of motion",
    "Programming & troubleshooting ranges of motion",
    "About the Actor",
    "Rick makes creatures, animals, robots, and crazy things for a living. Steven Spielberg has called him a \"genius\". To his FACE! John Candy laughed at his jokes during the filming of SPACEBALLS, and Mel Brooks even cast him as both Pizza the Hut and a horseback riding chimp for the film's end sequence. Rick has thrown up on Charlie Sheen's boat. Guillermo Del Toro told him he makes the \"most realistic blood I've ever seen.\" Jim Cameron shared a banana with him once. In a limo.",     
    "Rick brings the skills and technologies of prosthetics, animatronics and puppetry together to make a world of incredible beings. Rick's amazing work has been featured in such films as PIRATES OF THE CARRIBEAN: ON STRANGER TIDES, SNAKES ON A PLANE, WILLARD, BIG TROUBLE, MIMIC, DUDE, WHERE'S MY CAR, CASPER, OUTBREAK, THE SANTA CLAUSE, HOCUS POCUS, GHOSTBUSTERS II, SPACEBALLS and many more. He's known for creating the running Facehugger mechanism, and inner Queen Alien animatronics for the Oscar-Winning ALIENS while working for Stan Winston Studio. Few can match the quality or quantity of memorable, iconic creatures and characters he's created for major feature films, and hundreds of television commercials that are recognized worldwide, such as the Foster Farms Chickens, an animatronic Alligator for Verizon, and others.",
    "Lazzarini has built and puppeteered creations from a miniature six inch replica of Julia Roberts as Tinkerbell for HOOK, to life- sized elephants for OPERATION: DUMBO DROP, all the way up to a beautiful 20 foot tall marionette for Mayflower Moving. The menagerie of animatronic animals in residence at Lazzarini's Character Shop include a gigantic raven, scores of snakes, a realistic puma, a trio of hyenas, several chickens, a sheep, turtles, apes, rats, a chipmunk, a water buffalo, and two dolphins ready to take a road trip. He's created Buzzards, Frogs, Dogs, Aliens, Anteaters, Tortoises, Lobsters, and so many more for Commercial clients such as Ford, Chevy, Coca Cola, McDonald's, Cheerios, Budweiser, Sony, Duracell...and the list just goes on and on.",
    "Rick had an early love of comic books and apes, behind-the-scenes revelations about PLANET OF THE APES, making Super 8 Movies with ambitious yet adorably amateur effects, and being a super-geek. Combined with tenacity, persistence, and hard work (and even a little bit of talent) this developed into a life-long passion for animatronics and effects. A graduate of Loyola Marymount University, Rick got his start at prop houses and working at Makeup Effects Labs, John Dykstra's Apogee, Richard Edlund's Boss Films, and Stan Winston Studio. He opened up his own studio, The Character Shop, in 1986, and has been an industry leader from the start.",
    "Lazzarini can sculpt, make molds, research and develop in the lab, come up with incredible animatronic designs and mechanisms, and then bring his characters to life with virtuoso puppeteering skills. He is one of the top Creature Makers in Hollywood and he's always innovating, always looking for a new or more clever way to make something. Rick s forte is putting together great crews, then challenging himself and his crew to fulfill their potential. Under his tutelage, Rick's team is able to combine many disciplines to create incredibly lifelike and complex moving creatures.",       
    "Rick is a man of honor who knows his stuff. He delivers the very best to his clients, on time and under budget. He has been a consultant to the medical community for which he was contracted to create lifelike surgical training models for Cedars Sinai. He is an excellent teacher, having taught courses at UCLA, AFI and USC. He heads his own Animatronics Institute, giving hands on lessons to eager students in animatronic creature design and creation. And he is honored to share his knowledge and expertise with you.",
    "About the Director",
    "ONLINE Special Effects Makeup and Creature Creation training by the world's finest artists and technicians.",
    "Stan Winston School of Character Arts is an educational media company dedicated to teaching the art and science of character creation. SWSCA is partnered with the majority of Hollywood's top effects artists and studios to develop, produce, and distribute ultra-high quality online Creature FX video tutorials teaching the innovative techniques and processes used to create cinema's most memorable characters.",
    "SWSCA's educational content is subscription-based and will cover the full spectrum of character effects, from traditional approaches to the state-of-the-art Hybrid Practical/Digital Approach that has become the cornerstone of today's box office hits.",
    "See more"
  ],
  "price": 64.99,
  "images": [
    {
      "thumb": "https://m.media-amazon.com/images/I/51xAtwVr3ML._SX38_SY50_CR,0,0,38,50_.jpg",     
      "large": "https://m.media-amazon.com/images/I/51xAtwVr3ML.jpg",
      "variant": "MAIN",
      "hi_res": "https://m.media-amazon.com/images/I/61rjEQ2h9PL._SL1000_.jpg"
    },
    {
      "thumb": "https://m.media-amazon.com/images/I/51DNo8piIsL._SX38_SY50_CR,0,0,38,50_.jpg",     
      "large": "https://m.media-amazon.com/images/I/51DNo8piIsL.jpg",
      "variant": "BACK",
      "hi_res": "https://m.media-amazon.com/images/I/71vHrEa+hkL._SL1000_.jpg"
    }
  ],
  "videos": [],
  "store": "Rick Lazzarini  (Actor),     Stan Winston School of Character Arts  (Director)    Format: DVD",
  "categories": [
    "Movies & TV",
    "Genre for Featured Categories",
    "Special Interests"
  ],
  "details": {
    "Package Dimensions": "7.52 x 5.31 x 0.71 inches; 3.84 Ounces",
    "Director": "Stan Winston School of Character Arts",
    "Media Format": "Multiple Formats, Dolby, NTSC, Widescreen",
    "Run time": "4 hours and 18 minutes",
    "Release date": "August 19, 2012",
    "Actors": "Rick Lazzarini",
    "Studio": "Stan Winston School of Character Arts",
    "Country of Origin": "USA",
    "Number of discs": "2"
  },
  "parent_asin": "B00AHN851G",
  "bought_together": null
}