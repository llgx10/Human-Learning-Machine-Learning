Thế kỷ 21 được cho là thế kỷ của “Biển và đại dương”. Các nước trên thế giới đã có những chiến lược phát triển biển mới và trong đó phát triển khoa học và công nghệ biển được coi là khâu đột phá, tạo thế mạnh trong cạnh tranh và đưa đất nước trở thành cường quốc. Việt Nam ta may mắn có được đường bờ biển kéo  dài 11.409km kéo dài từ Bắc chí Nam và cũng đang từng bước hướng ra biển lớn. Ở đây nhóm có được dữ liệu gdp bình quân đầu người ở 63 tỉnh thành nước ta (tính theo đơn vị triệu đồng) để phân tích xem các tỉnh giáp biển có nền kinh tế phát triển hơn các tỉnh không giáp biển hay không. Lấy số liệu của năm 2021 để xây dựng mô hình hồi quy.
<h3>Bước 1: Import các thư viện cần thiết</h3>
 
<h3>Bước 2: Import file csv vào chương trình</h3>
 
<h3>Bước 3: sử dụng hàm glm để tính mô hình hồi quy logistis</h3>
 
<h3>Dựa vào hình trên, ta có</h3>
- Hệ số β1 của biến gdp (x) 36x10^(-4)
- Hệ số β0 là -0.45
- Kết quả phương trình hồi qui logistic đơn biến có được là: 
P=  ⅇ^(-0.45 +37x10^(-4) x)/(1+ⅇ^(-0.45 +37x10^(-4) x) )
Giả sử như một tỉnh có GDP bình quân đầu người bằng 50,000,000 một năm thì tỉnh đó có 75% khả năng là giáp biển.
Hệ số β1 dương, chứng tỏ GDP bình quân đầu người của tỉnh nào càng cao thì khả năng giáp biển của tỉnh đó càng lớn.
