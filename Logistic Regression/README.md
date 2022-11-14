<h2>Bước 1: Import các thư viện cần thiết
 
<h2>Bước 2: Import file csv vào chương trình
 
<h2>Bước 3: sử dụng hàm glm để tính mô hình hồi quy logistis
 
<h2>Dựa vào hình trên, ta có
- Hệ số β1 của biến gdp (x) 36x10^(-4)
- Hệ số β0 là -0.45
- Kết quả phương trình hồi qui logistic đơn biến có được là: 
P=  ⅇ^(-0.45 +37x10^(-4) x)/(1+ⅇ^(-0.45 +37x10^(-4) x) )
Giả sử như một tỉnh có GDP bình quân đầu người bằng 50,000,000 một năm thì tỉnh đó có 75% khả năng là giáp biển.
Hệ số β1 dương, chứng tỏ GDP bình quân đầu người của tỉnh nào càng cao thì khả năng giáp biển của tỉnh đó càng lớn.
