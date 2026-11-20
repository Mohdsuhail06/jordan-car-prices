# LinkedIn post

Suggested images (attach up to 4, in this order):
1. `reports/figures/app_screenshot.jpg`
2. `reports/figures/fig05_ev_share.png`
3. `reports/figures/fig13_age_effect.png`
4. `reports/figures/fig11_model_comparison.png`

---

## English

I wanted to learn regression properly, so instead of the usual house-prices dataset I picked a question closer to home: how much is a used car actually worth in Jordan?

I used JUCars-2024, a dataset of 8,867 real listings from OpenSooq, Dubizzle, Motory and other Jordanian marketplaces, and built a model that predicts a car's listing price and explains why.

On cars the model never saw during training, the typical error is around 12%, and about 7 out of 10 cars land within 20% of their listed price.

The numbers are fine, but what I'll remember are the things I didn't expect.

About 1,700 cars had no mileage. I was about to fill them with the average when I noticed every single one was a "200,000+ km" listing. If I'd filled them blindly, the model would have believed those cars drove half of what they really did.

Then XGBoost and LightGBM barely beat a simple linear regression. I looked at the cars with the biggest errors and the model wasn't the problem, the prices were. A 10-year-old Land Cruiser listed for 1,010 JOD, a 24-year-old car for 100,000 JOD. Switching to a loss function that doesn't chase extreme errors (Huber) brought the typical error down from 16% to under 13%. I kept those listings in the test set on purpose, because removing them would have made the result look better than it really is.

The data also tells a very Jordanian story: 73% of listed cars from model year 2023 are electric. And when I opened the model with SHAP, it had learned on its own that young EVs lose a bigger share of their value in the first years than petrol and hybrid cars.

One more: Zarqa looked more expensive than Amman in the raw data. Once the model knows the car itself, the city barely matters. The free zone just has newer cars. A good reminder that correlation isn't causation.

Tools: Python, pandas, scikit-learn, LightGBM, XGBoost, Optuna, SHAP, Streamlit, and pytest.

Code, notebooks and a small app that explains each price:
https://github.com/Mohdsuhail06/jordan-car-prices

Thanks to Nisrean Thalji and Khalid Tahat for publishing the dataset (Data in Brief, 2026).

#MachineLearning #DataScience #Python #Regression #Jordan

---

## العربي

حبيت أتعلّم الـ Regression صح، فبدل داتا أسعار البيوت المعروفة، اخترت سؤال أقرب لنا: قديش بتسوى السيارة المستعملة فعلاً بالأردن؟

استخدمت بيانات JUCars-2024، وفيها 8,867 إعلان حقيقي من السوق المفتوح ودوبيزل وموتري ومواقع أردنية ثانية، وبنيت نموذج بيتوقّع سعر السيارة وبيشرح ليش.

على سيارات ما شافها النموذج وقت التدريب، الغلط النموذجي حوالي 12%، وتقريباً 7 من كل 10 سيارات بيطلع توقّعها ضمن 20% من سعرها المعروض.

الأرقام منيحة، بس اللي رح يضل معي هي الأشياء اللي ما توقعتها.

حوالي 1,700 سيارة ما إلها كيلومترات. كنت رح أعبّيها بالمعدّل، ولاحظت إنها كلها مكتوب عليها "أكثر من 200 ألف كم". لو عبّيتها بدون ما أفحص، كان النموذج رح يفكر إنها ماشية نص المسافة الحقيقية.

بعدها XGBoost و LightGBM بالكاد غلبوا الانحدار الخطي البسيط. فحصت السيارات اللي النموذج غلط فيها كثير، وطلعت المشكلة بالأسعار مش بالنموذج: لاندكروزر عمرها 10 سنين معروضة بـ 1,010 دنانير، وسيارة عمرها 24 سنة بـ 100 ألف. لما غيّرت دالة الخطأ لوحدة ما بتلحق الغلطات الشاذة (Huber)، نزل الغلط من 16% لأقل من 13%. وخلّيت هالإعلانات ببيانات الاختبار عن قصد، لأني لو شلتها كانت النتيجة رح تطلع أحلى من الحقيقة.

والبيانات بتحكي قصة أردنية: 73% من السيارات المعروضة موديل 2023 كهربائية. ولما فتحت النموذج بـ SHAP، لقيته تعلّم لحاله إن الكهربائي الجديد بيخسر نسبة أكبر من قيمته بأول سنين مقارنة بالبنزين والهايبرد.

وكمان: الزرقاء بانت أغلى من عمّان بالبيانات الخام، بس لما النموذج بيعرف السيارة نفسها، المدينة تقريباً ما بتفرق. المنطقة الحرة فيها سيارات أحدث وبس. تذكير حلو إن الارتباط مش سبب.

الأدوات: Python و pandas و scikit-learn و LightGBM و XGBoost و Optuna و SHAP و Streamlit و pytest.

الكود والتحليل وتطبيق صغير بيشرح كل سعر:
https://github.com/Mohdsuhail06/jordan-car-prices

شكراً لنسرين ثلجي وخالد طهطهات على نشر البيانات (Data in Brief، 2026).

#MachineLearning #DataScience #Python #Jordan
