import Mathlib

namespace NUMINAMATH_CALUDE_oranges_per_child_l3710_371034

/-- Given 4 children and 12 oranges in total, prove that each child has 3 oranges. -/
theorem oranges_per_child (num_children : ℕ) (total_oranges : ℕ) 
  (h1 : num_children = 4) (h2 : total_oranges = 12) : 
  total_oranges / num_children = 3 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_child_l3710_371034


namespace NUMINAMATH_CALUDE_salary_percentage_calculation_l3710_371077

/-- Given two employees X and Y with a total salary and Y's known salary,
    calculate the percentage of Y's salary that X is paid. -/
theorem salary_percentage_calculation
  (total_salary : ℝ) (y_salary : ℝ) (h1 : total_salary = 638)
  (h2 : y_salary = 290) :
  (total_salary - y_salary) / y_salary * 100 = 120 :=
by sorry

end NUMINAMATH_CALUDE_salary_percentage_calculation_l3710_371077


namespace NUMINAMATH_CALUDE_collins_initial_flowers_l3710_371064

/-- Proves that Collin's initial number of flowers is 25 given the problem conditions --/
theorem collins_initial_flowers :
  ∀ (collins_initial_flowers : ℕ) (ingrids_flowers : ℕ) (petals_per_flower : ℕ) (collins_total_petals : ℕ),
    ingrids_flowers = 33 →
    petals_per_flower = 4 →
    collins_total_petals = 144 →
    collins_total_petals = (collins_initial_flowers + ingrids_flowers / 3) * petals_per_flower →
    collins_initial_flowers = 25 :=
by sorry

end NUMINAMATH_CALUDE_collins_initial_flowers_l3710_371064


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3710_371019

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (ha : a = 5) 
  (hb : b = 7) 
  (hc : c = 10) : 
  a + b > c ∧ b + c > a ∧ c + a > b := by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3710_371019


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3710_371020

-- Define the quadratic function
def f (x : ℝ) := 3 * x^2 - 7 * x - 6

-- Define the solution set
def solution_set : Set ℝ := {x | -2/3 < x ∧ x < 3}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = solution_set :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3710_371020


namespace NUMINAMATH_CALUDE_ratio_when_a_is_20_percent_more_than_b_l3710_371066

theorem ratio_when_a_is_20_percent_more_than_b (A B : ℝ) (h : A = 1.2 * B) : A / B = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_when_a_is_20_percent_more_than_b_l3710_371066


namespace NUMINAMATH_CALUDE_abs_sum_inequality_sum_bound_from_square_sum_l3710_371071

-- Part I
theorem abs_sum_inequality (x a : ℝ) (ha : a > 0) :
  |x - 1/a| + |x + a| ≥ 2 := by sorry

-- Part II
theorem sum_bound_from_square_sum (x y z : ℝ) (h : x^2 + 4*y^2 + z^2 = 3) :
  |x + 2*y + z| ≤ 3 := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_sum_bound_from_square_sum_l3710_371071


namespace NUMINAMATH_CALUDE_test_has_hundred_questions_l3710_371029

/-- Represents a test with a specific scoring system -/
structure Test where
  total_questions : ℕ
  correct_responses : ℕ
  incorrect_responses : ℕ
  score : ℤ
  score_calculation : score = correct_responses - 2 * incorrect_responses
  total_questions_sum : total_questions = correct_responses + incorrect_responses

/-- Theorem stating that given the conditions, the test has 100 questions -/
theorem test_has_hundred_questions (t : Test) 
  (h1 : t.score = 79) 
  (h2 : t.correct_responses = 93) : 
  t.total_questions = 100 := by
  sorry


end NUMINAMATH_CALUDE_test_has_hundred_questions_l3710_371029


namespace NUMINAMATH_CALUDE_inner_triangle_area_l3710_371081

/-- Given a triangle with area T, the area of the smaller triangle formed by
    joining the points that divide each side into three equal segments is 4/9 * T -/
theorem inner_triangle_area (T : ℝ) (h : T > 0) :
  ∃ (inner_area : ℝ), inner_area = (4 / 9) * T := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_area_l3710_371081


namespace NUMINAMATH_CALUDE_library_books_l3710_371047

theorem library_books (original_books : ℕ) : 
  (original_books + 140 = (27 : ℚ) / 25 * original_books) → 
  original_books = 1750 := by
  sorry

end NUMINAMATH_CALUDE_library_books_l3710_371047


namespace NUMINAMATH_CALUDE_abs_neg_one_third_l3710_371016

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_third_l3710_371016


namespace NUMINAMATH_CALUDE_banana_sharing_l3710_371036

theorem banana_sharing (jefferson_bananas : ℕ) (walter_bananas : ℕ) : 
  jefferson_bananas = 56 →
  walter_bananas = jefferson_bananas - (jefferson_bananas / 4) →
  (jefferson_bananas + walter_bananas) / 2 = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_sharing_l3710_371036


namespace NUMINAMATH_CALUDE_geometric_series_third_term_l3710_371093

theorem geometric_series_third_term (a : ℝ) (S : ℝ) :
  S = a / (1 - (1/4 : ℝ)) →
  S = 40 →
  a * (1/4 : ℝ)^2 = 15/8 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_third_term_l3710_371093


namespace NUMINAMATH_CALUDE_min_value_a_plus_3b_l3710_371001

theorem min_value_a_plus_3b (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_eq : a + 3*b = 1/a + 3/b) : 
  ∀ x y : ℝ, x > 0 → y > 0 → x + 3*y = 1/x + 3/y → a + 3*b ≤ x + 3*y ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 3*y = 1/x + 3/y ∧ x + 3*y = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_3b_l3710_371001


namespace NUMINAMATH_CALUDE_train_speed_l3710_371054

/-- Given a train of length 350 meters that crosses a pole in 21 seconds, its speed is 60 km/hr. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 350) (h2 : crossing_time = 21) :
  (train_length / 1000) / (crossing_time / 3600) = 60 :=
sorry

end NUMINAMATH_CALUDE_train_speed_l3710_371054


namespace NUMINAMATH_CALUDE_abc_modulo_seven_l3710_371091

theorem abc_modulo_seven (a b c : ℕ) 
  (ha : a < 7) (hb : b < 7) (hc : c < 7)
  (h1 : (a + 2*b + 3*c) % 7 = 0)
  (h2 : (2*a + 3*b + c) % 7 = 4)
  (h3 : (3*a + b + 2*c) % 7 = 4) :
  (a * b * c) % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_abc_modulo_seven_l3710_371091


namespace NUMINAMATH_CALUDE_white_to_black_stone_ratio_l3710_371024

theorem white_to_black_stone_ratio :
  ∀ (total_stones white_stones black_stones : ℕ),
    total_stones = 100 →
    white_stones = 60 →
    black_stones = total_stones - white_stones →
    white_stones > black_stones →
    (white_stones : ℚ) / (black_stones : ℚ) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_white_to_black_stone_ratio_l3710_371024


namespace NUMINAMATH_CALUDE_sqrt_221_between_15_and_16_l3710_371040

theorem sqrt_221_between_15_and_16 : 15 < Real.sqrt 221 ∧ Real.sqrt 221 < 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_221_between_15_and_16_l3710_371040


namespace NUMINAMATH_CALUDE_max_servings_is_56_l3710_371000

/-- Represents the ingredients required for one serving of salad -/
structure ServingRequirement where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Represents the available ingredients in the warehouse -/
structure WarehouseStock where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Calculates the maximum number of servings that can be made -/
def maxServings (req : ServingRequirement) (stock : WarehouseStock) : ℕ :=
  min
    (stock.cucumbers / req.cucumbers)
    (min
      (stock.tomatoes / req.tomatoes)
      (min
        (stock.brynza / req.brynza)
        (stock.peppers / req.peppers)))

/-- Theorem stating the maximum number of servings that can be made -/
theorem max_servings_is_56 :
  let req := ServingRequirement.mk 2 2 75 1
  let stock := WarehouseStock.mk 117 116 4200 60
  maxServings req stock = 56 := by
  sorry

#eval maxServings (ServingRequirement.mk 2 2 75 1) (WarehouseStock.mk 117 116 4200 60)

end NUMINAMATH_CALUDE_max_servings_is_56_l3710_371000


namespace NUMINAMATH_CALUDE_triangle_PQR_area_l3710_371038

/-- The area of a triangle with vertices P(-4, 2), Q(6, 2), and R(2, -5) is 35 square units. -/
theorem triangle_PQR_area : 
  let P : ℝ × ℝ := (-4, 2)
  let Q : ℝ × ℝ := (6, 2)
  let R : ℝ × ℝ := (2, -5)
  let triangle_area := abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)
  triangle_area = 35 := by sorry

end NUMINAMATH_CALUDE_triangle_PQR_area_l3710_371038


namespace NUMINAMATH_CALUDE_min_students_is_fifteen_l3710_371027

/-- Represents the attendance for each day of the week -/
structure WeeklyAttendance where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the minimum number of students given weekly attendance -/
def minStudents (attendance : WeeklyAttendance) : Nat :=
  max (attendance.monday + attendance.wednesday + attendance.friday)
      (attendance.tuesday + attendance.thursday)

/-- Theorem: The minimum number of students who visited the library during the week is 15 -/
theorem min_students_is_fifteen (attendance : WeeklyAttendance)
  (h1 : attendance.monday = 5)
  (h2 : attendance.tuesday = 6)
  (h3 : attendance.wednesday = 4)
  (h4 : attendance.thursday = 8)
  (h5 : attendance.friday = 7) :
  minStudents attendance = 15 := by
  sorry

#eval minStudents ⟨5, 6, 4, 8, 7⟩

end NUMINAMATH_CALUDE_min_students_is_fifteen_l3710_371027


namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l3710_371069

theorem range_of_2a_minus_b (a b : ℝ) (ha : -2 < a ∧ a < 2) (hb : 2 < b ∧ b < 3) :
  ∀ x, (∃ a b, (-2 < a ∧ a < 2) ∧ (2 < b ∧ b < 3) ∧ x = 2*a - b) ↔ -7 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l3710_371069


namespace NUMINAMATH_CALUDE_sum_of_digits_10_95_minus_195_l3710_371007

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem: sum of digits of 10^95 - 195 is 841 -/
theorem sum_of_digits_10_95_minus_195 : sum_of_digits (10^95 - 195) = 841 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_10_95_minus_195_l3710_371007


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3710_371028

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  /-- The ratio of y to x for the asymptotes -/
  asymptote_slope : ℝ
  /-- The hyperbola has foci on the x-axis -/
  foci_on_x_axis : Bool

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem stating the eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_slope : h.asymptote_slope = 2/3) 
  (h_foci : h.foci_on_x_axis = true) : 
  eccentricity h = Real.sqrt 13 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3710_371028


namespace NUMINAMATH_CALUDE_stones_per_bracelet_l3710_371068

theorem stones_per_bracelet (total_stones : Float) (num_bracelets : Float) 
  (h1 : total_stones = 88.0) 
  (h2 : num_bracelets = 8.0) : 
  total_stones / num_bracelets = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_stones_per_bracelet_l3710_371068


namespace NUMINAMATH_CALUDE_divisors_of_72_l3710_371011

theorem divisors_of_72 : Finset.card ((Finset.range 73).filter (λ x => 72 % x = 0)) * 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_72_l3710_371011


namespace NUMINAMATH_CALUDE_inverse_sqrt_difference_equals_sum_l3710_371032

theorem inverse_sqrt_difference_equals_sum (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  1 / (Real.sqrt a - Real.sqrt b) = Real.sqrt a + Real.sqrt b :=
by sorry

end NUMINAMATH_CALUDE_inverse_sqrt_difference_equals_sum_l3710_371032


namespace NUMINAMATH_CALUDE_hotel_charges_l3710_371045

-- Define the charges for each hotel
variable (P R G S T : ℝ)

-- Define the relationships between the charges
axiom p_r : P = 0.75 * R
axiom p_g : P = 0.90 * G
axiom s_r : S = 1.15 * R
axiom t_g : T = 0.80 * G

-- Theorem to prove
theorem hotel_charges :
  S = 1.5333 * P ∧ 
  T = 0.8888 * P ∧ 
  (R - G) / G = 0.18 := by sorry

end NUMINAMATH_CALUDE_hotel_charges_l3710_371045


namespace NUMINAMATH_CALUDE_four_correct_statements_l3710_371070

theorem four_correct_statements (a b m : ℝ) : 
  -- Statement 1
  (∀ m, a * m^2 > b * m^2 → a > b) ∧
  -- Statement 2
  (a > b → a * |a| > b * |b|) ∧
  -- Statement 3
  (b > a ∧ a > 0 ∧ m > 0 → (a + m) / (b + m) > a / b) ∧
  -- Statement 4
  (a > b ∧ b > 0 ∧ |Real.log a| = |Real.log b| → 2 * a + b > 3) :=
by sorry

end NUMINAMATH_CALUDE_four_correct_statements_l3710_371070


namespace NUMINAMATH_CALUDE_area_of_wxuv_l3710_371092

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the division of a rectangle into four smaller rectangles -/
structure RectangleDivision where
  pqxw : Rectangle
  qrsx : Rectangle
  xstu : Rectangle
  wxuv : Rectangle

theorem area_of_wxuv (div : RectangleDivision)
  (h1 : div.pqxw.area = 9)
  (h2 : div.qrsx.area = 10)
  (h3 : div.xstu.area = 15) :
  div.wxuv.area = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_wxuv_l3710_371092


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3710_371012

theorem hyperbola_foci_distance (x y : ℝ) :
  x^2 / 25 - y^2 / 4 = 1 →
  ∃ (f₁ f₂ : ℝ × ℝ), (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 4 * 29 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3710_371012


namespace NUMINAMATH_CALUDE_flagstaff_height_is_correct_l3710_371082

/-- The height of the flagstaff in meters -/
def flagstaff_height : ℝ := 17.5

/-- The length of the flagstaff's shadow in meters -/
def flagstaff_shadow : ℝ := 40.25

/-- The height of the building in meters -/
def building_height : ℝ := 12.5

/-- The length of the building's shadow in meters -/
def building_shadow : ℝ := 28.75

/-- Theorem stating that the calculated flagstaff height is correct -/
theorem flagstaff_height_is_correct :
  flagstaff_height = (building_height * flagstaff_shadow) / building_shadow :=
by sorry

end NUMINAMATH_CALUDE_flagstaff_height_is_correct_l3710_371082


namespace NUMINAMATH_CALUDE_cereal_serving_size_l3710_371002

def cereal_box_problem (total_cups : ℕ) (total_servings : ℕ) : Prop :=
  total_cups ≠ 0 ∧ total_servings ≠ 0 → total_cups / total_servings = 2

theorem cereal_serving_size : cereal_box_problem 18 9 := by
  sorry

end NUMINAMATH_CALUDE_cereal_serving_size_l3710_371002


namespace NUMINAMATH_CALUDE_deck_size_l3710_371025

theorem deck_size (r b : ℕ) : 
  (r : ℚ) / (r + b) = 2/5 →
  (r : ℚ) / (r + b + 7) = 1/3 →
  r + b = 35 := by
sorry

end NUMINAMATH_CALUDE_deck_size_l3710_371025


namespace NUMINAMATH_CALUDE_floor_of_3_999_l3710_371004

theorem floor_of_3_999 : ⌊(3.999 : ℝ)⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_of_3_999_l3710_371004


namespace NUMINAMATH_CALUDE_complement_A_in_U_l3710_371057

def U : Set ℕ := {x : ℕ | x ≥ 2}
def A : Set ℕ := {x : ℕ | x^2 ≥ 5}

theorem complement_A_in_U : (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l3710_371057


namespace NUMINAMATH_CALUDE_sum_smallest_largest_fourdigit_l3710_371010

/-- A function that generates all four-digit numbers using the digits 0, 3, 4, and 8 -/
def fourDigitNumbers : List Nat := sorry

/-- The smallest four-digit number formed using 0, 3, 4, and 8 -/
def smallestNumber : Nat := sorry

/-- The largest four-digit number formed using 0, 3, 4, and 8 -/
def largestNumber : Nat := sorry

/-- Theorem stating that the sum of the smallest and largest four-digit numbers
    formed using 0, 3, 4, and 8 is 11478 -/
theorem sum_smallest_largest_fourdigit :
  smallestNumber + largestNumber = 11478 := by sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_fourdigit_l3710_371010


namespace NUMINAMATH_CALUDE_curve_C_properties_l3710_371037

-- Define the curve C
def C (x : ℝ) : ℝ := x^3 + 5*x^2 + 3*x

-- State the theorem
theorem curve_C_properties :
  -- The derivative of C is 3x² + 10x + 3
  (∀ x : ℝ, deriv C x = 3*x^2 + 10*x + 3) ∧
  -- The equation of the tangent line to C at x = 1 is 16x - y - 7 = 0
  (∀ y : ℝ, (C 1 = y) → (16 - y - 7 = 0 ↔ ∃ x : ℝ, y = 16*(x - 1) + C 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_curve_C_properties_l3710_371037


namespace NUMINAMATH_CALUDE_defective_probability_l3710_371090

/-- The probability of an item being produced by Machine 1 -/
def prob_machine1 : ℝ := 0.4

/-- The probability of an item being produced by Machine 2 -/
def prob_machine2 : ℝ := 0.6

/-- The probability of a defective item from Machine 1 -/
def defect_rate1 : ℝ := 0.03

/-- The probability of a defective item from Machine 2 -/
def defect_rate2 : ℝ := 0.02

/-- The probability of a randomly selected item being defective -/
def prob_defective : ℝ := prob_machine1 * defect_rate1 + prob_machine2 * defect_rate2

theorem defective_probability : prob_defective = 0.024 := by
  sorry

end NUMINAMATH_CALUDE_defective_probability_l3710_371090


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3710_371058

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  (∃ k : ℕ, k > 12 ∧ (∀ m : ℕ, m > 0 → k ∣ (m * (m + 1) * (m + 2) * (m + 3)))) →
  False :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3710_371058


namespace NUMINAMATH_CALUDE_shop_width_calculation_l3710_371013

/-- Given a shop with specified rent and dimensions, calculate its width -/
theorem shop_width_calculation (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_sqft : ℕ) : 
  monthly_rent = 2244 →
  length = 22 →
  annual_rent_per_sqft = 68 →
  (monthly_rent * 12) / annual_rent_per_sqft / length = 18 := by
  sorry

#check shop_width_calculation

end NUMINAMATH_CALUDE_shop_width_calculation_l3710_371013


namespace NUMINAMATH_CALUDE_yellow_balloons_count_l3710_371023

/-- The number of yellow balloons -/
def yellow_balloons : ℕ := 3414

/-- The number of black balloons -/
def black_balloons : ℕ := yellow_balloons + 1762

/-- The total number of balloons -/
def total_balloons : ℕ := yellow_balloons + black_balloons

theorem yellow_balloons_count : yellow_balloons = 3414 :=
  by
  have h1 : black_balloons = yellow_balloons + 1762 := rfl
  have h2 : total_balloons / 10 = 859 := by sorry
  sorry


end NUMINAMATH_CALUDE_yellow_balloons_count_l3710_371023


namespace NUMINAMATH_CALUDE_butter_price_is_correct_l3710_371075

/-- Represents the milk and butter sales problem --/
structure MilkButterSales where
  milk_price : ℚ
  milk_to_butter_ratio : ℚ
  num_cows : ℕ
  milk_per_cow : ℚ
  num_customers : ℕ
  milk_per_customer : ℚ
  total_earnings : ℚ

/-- Calculates the price per stick of butter --/
def butter_price (s : MilkButterSales) : ℚ :=
  let total_milk := s.num_cows * s.milk_per_cow
  let milk_sold := s.num_customers * s.milk_per_customer
  let milk_for_butter := total_milk - milk_sold
  let butter_sticks := milk_for_butter * s.milk_to_butter_ratio
  let milk_earnings := milk_sold * s.milk_price
  let butter_earnings := s.total_earnings - milk_earnings
  butter_earnings / butter_sticks

/-- Theorem stating that the butter price is $1.50 given the problem conditions --/
theorem butter_price_is_correct (s : MilkButterSales) 
  (h1 : s.milk_price = 3)
  (h2 : s.milk_to_butter_ratio = 2)
  (h3 : s.num_cows = 12)
  (h4 : s.milk_per_cow = 4)
  (h5 : s.num_customers = 6)
  (h6 : s.milk_per_customer = 6)
  (h7 : s.total_earnings = 144) :
  butter_price s = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_butter_price_is_correct_l3710_371075


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3710_371022

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n → n ≥ 102 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3710_371022


namespace NUMINAMATH_CALUDE_butter_calculation_l3710_371046

/-- Calculates the required amount of butter given a change in sugar amount -/
def required_butter (original_butter original_sugar new_sugar : ℚ) : ℚ :=
  (new_sugar / original_sugar) * original_butter

theorem butter_calculation (original_butter original_sugar new_sugar : ℚ) 
  (h1 : original_butter = 25)
  (h2 : original_sugar = 125)
  (h3 : new_sugar = 1000) :
  required_butter original_butter original_sugar new_sugar = 200 := by
  sorry

#eval required_butter 25 125 1000

end NUMINAMATH_CALUDE_butter_calculation_l3710_371046


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_and_evaluate_expression_l3710_371073

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  5 * a * b^2 - 3 * a * b^2 + (1/3) * a * b^2 = (7/3) * a * b^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (m n : ℝ) :
  (7 * m^2 * n - 5 * m) - (4 * m^2 * n - 5 * m) = 3 * m^2 * n := by sorry

-- Problem 3
theorem simplify_and_evaluate_expression (x y : ℝ) 
  (hx : x = -1/4) (hy : y = 2) :
  2 * x^2 * y - 2 * (x * y^2 + 2 * x^2 * y) + 2 * (x^2 * y - 3 * x * y^2) = 8 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_and_evaluate_expression_l3710_371073


namespace NUMINAMATH_CALUDE_f_min_value_l3710_371031

noncomputable def f (x : ℝ) := Real.exp x + 3 * x^2 - x + 2011

theorem f_min_value :
  ∃ (min : ℝ), min = 2012 ∧ ∀ (x : ℝ), f x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_f_min_value_l3710_371031


namespace NUMINAMATH_CALUDE_price_two_birdhouses_is_32_l3710_371086

/-- The price Denver charges for two birdhouses -/
def price_two_birdhouses : ℚ :=
  let pieces_per_birdhouse : ℕ := 7
  let price_per_piece : ℚ := 3/2  -- $1.50 as a rational number
  let profit_per_birdhouse : ℚ := 11/2  -- $5.50 as a rational number
  let cost_per_birdhouse : ℚ := pieces_per_birdhouse * price_per_piece
  let price_per_birdhouse : ℚ := cost_per_birdhouse + profit_per_birdhouse
  2 * price_per_birdhouse

/-- Theorem stating that the price for two birdhouses is $32.00 -/
theorem price_two_birdhouses_is_32 : price_two_birdhouses = 32 := by
  sorry

end NUMINAMATH_CALUDE_price_two_birdhouses_is_32_l3710_371086


namespace NUMINAMATH_CALUDE_mint_problem_solvable_l3710_371088

/-- Represents a set of coin denominations. -/
def CoinSet := Finset ℕ

/-- Checks if a given amount can be represented using at most 8 coins from the set. -/
def canRepresent (coins : CoinSet) (amount : ℕ) : Prop :=
  ∃ (representation : Finset ℕ), 
    representation.card ≤ 8 ∧ 
    (representation.sum (λ x => x * (coins.filter (λ c => c = x)).card)) = amount

/-- The main theorem stating that there exists a set of 12 coin denominations
    that can represent all amounts from 1 to 6543 using at most 8 coins. -/
theorem mint_problem_solvable : 
  ∃ (coins : CoinSet), 
    coins.card = 12 ∧ 
    ∀ (amount : ℕ), 1 ≤ amount ∧ amount ≤ 6543 → canRepresent coins amount :=
by
  sorry


end NUMINAMATH_CALUDE_mint_problem_solvable_l3710_371088


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l3710_371098

theorem no_positive_integer_solution :
  ¬∃ (p q r : ℕ+), 
    (p^2 : ℚ) / q = 4 / 5 ∧
    (q : ℚ) / r^2 = 2 / 3 ∧
    (p : ℚ) / r^3 = 6 / 7 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l3710_371098


namespace NUMINAMATH_CALUDE_decagon_triangles_l3710_371085

/-- The number of vertices in a decagon -/
def n : ℕ := 10

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem decagon_triangles : choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangles_l3710_371085


namespace NUMINAMATH_CALUDE_probability_is_three_fifths_l3710_371021

/-- The set of letters in the word "STATISTICS" -/
def statistics_letters : Finset Char := {'S', 'T', 'A', 'I', 'C'}

/-- The set of letters in the word "TEST" -/
def test_letters : Finset Char := {'T', 'E', 'S'}

/-- The number of occurrences of each letter in "STATISTICS" -/
def letter_count (c : Char) : ℕ :=
  if c = 'S' then 3
  else if c = 'T' then 3
  else if c = 'A' then 1
  else if c = 'I' then 2
  else if c = 'C' then 1
  else 0

/-- The total number of tiles -/
def total_tiles : ℕ := statistics_letters.sum letter_count

/-- The number of tiles with letters from "TEST" -/
def test_tiles : ℕ := (statistics_letters ∩ test_letters).sum letter_count

/-- The probability of selecting a tile with a letter from "TEST" -/
def probability : ℚ := test_tiles / total_tiles

theorem probability_is_three_fifths : probability = 3 / 5 := by
  sorry


end NUMINAMATH_CALUDE_probability_is_three_fifths_l3710_371021


namespace NUMINAMATH_CALUDE_f_not_equal_one_l3710_371052

theorem f_not_equal_one (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x = -f (-x))
  (h2 : ∀ x : ℝ, 1 < x ∧ x < 2 → f x > 0) :
  f (-1.5) ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_f_not_equal_one_l3710_371052


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_25_mod_26_l3710_371006

theorem largest_four_digit_congruent_to_25_mod_26 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n ≡ 25 [MOD 26] → n ≤ 9983 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_25_mod_26_l3710_371006


namespace NUMINAMATH_CALUDE_joy_reading_time_l3710_371015

/-- Given that Joy can read 8 pages in 20 minutes, prove that it takes her 5 hours to read 120 pages. -/
theorem joy_reading_time : 
  -- Define Joy's reading speed
  let pages_per_20_min : ℚ := 8
  let total_pages : ℚ := 120
  -- Calculate the time in hours
  let time_in_hours : ℚ := (total_pages / pages_per_20_min) * (20 / 60)
  -- Prove that the time is 5 hours
  ∀ (pages_per_20_min total_pages time_in_hours : ℚ), 
    pages_per_20_min = 8 → 
    total_pages = 120 → 
    time_in_hours = (total_pages / pages_per_20_min) * (20 / 60) → 
    time_in_hours = 5 := by
  sorry


end NUMINAMATH_CALUDE_joy_reading_time_l3710_371015


namespace NUMINAMATH_CALUDE_ali_spending_ratio_l3710_371008

theorem ali_spending_ratio :
  ∀ (initial_amount food_cost glasses_cost remaining : ℕ),
  initial_amount = 480 →
  glasses_cost = (initial_amount - food_cost) / 3 →
  remaining = initial_amount - food_cost - glasses_cost →
  remaining = 160 →
  food_cost * 2 = initial_amount :=
λ initial_amount food_cost glasses_cost remaining
  h_initial h_glasses h_remaining h_final =>
sorry

end NUMINAMATH_CALUDE_ali_spending_ratio_l3710_371008


namespace NUMINAMATH_CALUDE_power_function_property_l3710_371097

/-- Given a function f(x) = x^α where f(2) = 4, prove that f(-1) = 1 -/
theorem power_function_property (α : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x ^ α) 
  (h2 : f 2 = 4) : 
  f (-1) = 1 := by
sorry

end NUMINAMATH_CALUDE_power_function_property_l3710_371097


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3710_371067

/-- The eccentricity of a hyperbola with given equation and asymptote -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 4 / 3) : 
  let c := Real.sqrt (a^2 + b^2)
  c / a = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3710_371067


namespace NUMINAMATH_CALUDE_age_problem_l3710_371079

theorem age_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 28 →
  (a + c) / 2 = 29 →
  b = 26 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l3710_371079


namespace NUMINAMATH_CALUDE_unique_m_value_l3710_371072

theorem unique_m_value (m : ℝ) : 
  let A : Set ℝ := {m, 1}
  let B : Set ℝ := {m^2, -1}
  A = B → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_value_l3710_371072


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3710_371039

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ r s : ℝ, (72 - 18*x - x^2 = 0 ↔ (x = r ∨ x = s)) ∧ r + s = -18) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3710_371039


namespace NUMINAMATH_CALUDE_parallelogram_coordinate_sum_l3710_371049

/-- A parallelogram with vertices P, Q, R, S in 2D space -/
structure Parallelogram where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ

/-- The sum of coordinates of a point -/
def sum_coordinates (point : ℝ × ℝ) : ℝ := point.1 + point.2

/-- Theorem: In a parallelogram PQRS with P(-3,-2), Q(1,-5), R(9,1), and P, R opposite vertices,
    the sum of coordinates of S is 9 -/
theorem parallelogram_coordinate_sum (PQRS : Parallelogram) 
    (h1 : PQRS.P = (-3, -2))
    (h2 : PQRS.Q = (1, -5))
    (h3 : PQRS.R = (9, 1))
    (h4 : PQRS.P.1 + PQRS.R.1 = PQRS.Q.1 + PQRS.S.1) 
    (h5 : PQRS.P.2 + PQRS.R.2 = PQRS.Q.2 + PQRS.S.2) :
    sum_coordinates PQRS.S = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_coordinate_sum_l3710_371049


namespace NUMINAMATH_CALUDE_candy_has_nine_pencils_l3710_371003

/-- The number of pencils each person has -/
structure PencilCounts where
  calen : ℕ
  caleb : ℕ
  candy : ℕ
  darlene : ℕ

/-- The conditions of the pencil problem -/
def PencilProblem (p : PencilCounts) : Prop :=
  p.calen = p.caleb + 5 ∧
  p.caleb = 2 * p.candy - 3 ∧
  p.darlene = p.calen + p.caleb + p.candy + 4 ∧
  p.calen - 10 = 10

/-- The theorem stating that under the given conditions, Candy has 9 pencils -/
theorem candy_has_nine_pencils (p : PencilCounts) (h : PencilProblem p) : p.candy = 9 := by
  sorry

end NUMINAMATH_CALUDE_candy_has_nine_pencils_l3710_371003


namespace NUMINAMATH_CALUDE_reduce_to_single_digit_l3710_371096

/-- Represents the operation of splitting digits and summing -/
def digitSplitSum (n : ℕ) : ℕ → ℕ :=
  sorry

/-- Predicate for a number being single-digit -/
def isSingleDigit (n : ℕ) : Prop :=
  n < 10

/-- Theorem stating that any natural number can be reduced to a single digit in at most 15 steps -/
theorem reduce_to_single_digit (N : ℕ) :
  ∃ (seq : Fin 16 → ℕ), seq 0 = N ∧ isSingleDigit (seq 15) ∧
  ∀ i : Fin 15, seq (i + 1) = digitSplitSum (seq i) (seq i) :=
sorry

end NUMINAMATH_CALUDE_reduce_to_single_digit_l3710_371096


namespace NUMINAMATH_CALUDE_percentage_same_grade_l3710_371083

def total_students : ℕ := 50

def same_grade_A : ℕ := 3
def same_grade_B : ℕ := 6
def same_grade_C : ℕ := 8
def same_grade_D : ℕ := 2
def same_grade_F : ℕ := 1

def total_same_grade : ℕ := same_grade_A + same_grade_B + same_grade_C + same_grade_D + same_grade_F

theorem percentage_same_grade : 
  (total_same_grade : ℚ) / total_students * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_same_grade_l3710_371083


namespace NUMINAMATH_CALUDE_num_valid_assignments_is_72_l3710_371061

/-- Represents a valid assignment of doctors to positions -/
structure DoctorAssignment where
  assignments : Fin 5 → Fin 4
  all_positions_filled : ∀ p : Fin 4, ∃ d : Fin 5, assignments d = p
  first_two_different : assignments 0 ≠ assignments 1

/-- The number of valid doctor assignments -/
def num_valid_assignments : ℕ := sorry

/-- Theorem stating that the number of valid assignments is 72 -/
theorem num_valid_assignments_is_72 : num_valid_assignments = 72 := by sorry

end NUMINAMATH_CALUDE_num_valid_assignments_is_72_l3710_371061


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l3710_371060

theorem final_sum_after_transformation (S a b : ℝ) (h : a + b = S) :
  3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l3710_371060


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_is_zero_l3710_371056

theorem product_of_difference_and_sum_is_zero (a : ℝ) (x y : ℝ) 
  (h1 : x = a + 5)
  (h2 : a = 20)
  (h3 : y = 25) :
  (x - y) * (x + y) = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_is_zero_l3710_371056


namespace NUMINAMATH_CALUDE_ellipse_right_triangle_distance_to_x_axis_l3710_371053

/-- An ellipse with semi-major axis 4 and semi-minor axis 3 -/
structure Ellipse :=
  (x y : ℝ)
  (eq : x^2 / 16 + y^2 / 9 = 1)

/-- The foci of the ellipse -/
def foci (e : Ellipse) : ℝ × ℝ := sorry

/-- A point P on the ellipse forms a right triangle with the foci -/
def right_triangle_with_foci (e : Ellipse) (p : ℝ × ℝ) : Prop := sorry

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : ℝ × ℝ) : ℝ := sorry

theorem ellipse_right_triangle_distance_to_x_axis (e : Ellipse) (p : ℝ × ℝ) :
  p.1^2 / 16 + p.2^2 / 9 = 1 →
  right_triangle_with_foci e p →
  distance_to_x_axis p = 9/4 := by sorry

end NUMINAMATH_CALUDE_ellipse_right_triangle_distance_to_x_axis_l3710_371053


namespace NUMINAMATH_CALUDE_square_bound_values_l3710_371033

theorem square_bound_values (k : ℤ) : 
  (∃ (s : Finset ℤ), (∀ x ∈ s, 121 < x^2 ∧ x^2 < 225) ∧ s.card ≤ 3 ∧ 
   (∀ y : ℤ, 121 < y^2 ∧ y^2 < 225 → y ∈ s)) :=
by sorry

end NUMINAMATH_CALUDE_square_bound_values_l3710_371033


namespace NUMINAMATH_CALUDE_expected_groups_l3710_371014

/-- The expected number of alternating groups in a random sequence of zeros and ones -/
theorem expected_groups (k m : ℕ) : 
  let total := k + m
  let prob_diff := (2 * k * m) / (total * (total - 1))
  1 + (total - 1) * prob_diff = 1 + (2 * k * m) / total := by
  sorry

end NUMINAMATH_CALUDE_expected_groups_l3710_371014


namespace NUMINAMATH_CALUDE_multiples_properties_l3710_371062

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 5 * k) 
  (hb : ∃ m : ℤ, b = 4 * m) : 
  (∃ n : ℤ, b = 2 * n) ∧ (∃ p : ℤ, a - b = 5 * p) := by
  sorry

end NUMINAMATH_CALUDE_multiples_properties_l3710_371062


namespace NUMINAMATH_CALUDE_peter_money_carried_l3710_371080

/-- The amount of money Peter carried to the market -/
def money_carried : ℝ := sorry

/-- The price of potatoes per kilo -/
def potato_price : ℝ := 2

/-- The quantity of potatoes bought in kilos -/
def potato_quantity : ℝ := 6

/-- The price of tomatoes per kilo -/
def tomato_price : ℝ := 3

/-- The quantity of tomatoes bought in kilos -/
def tomato_quantity : ℝ := 9

/-- The price of cucumbers per kilo -/
def cucumber_price : ℝ := 4

/-- The quantity of cucumbers bought in kilos -/
def cucumber_quantity : ℝ := 5

/-- The price of bananas per kilo -/
def banana_price : ℝ := 5

/-- The quantity of bananas bought in kilos -/
def banana_quantity : ℝ := 3

/-- The amount of money Peter has remaining after buying all items -/
def money_remaining : ℝ := 426

theorem peter_money_carried :
  money_carried = 
    potato_price * potato_quantity +
    tomato_price * tomato_quantity +
    cucumber_price * cucumber_quantity +
    banana_price * banana_quantity +
    money_remaining :=
by sorry

end NUMINAMATH_CALUDE_peter_money_carried_l3710_371080


namespace NUMINAMATH_CALUDE_point_on_curve_iff_satisfies_equation_l3710_371065

-- Define a curve C in 2D space
def Curve (F : ℝ → ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | F p.1 p.2 = 0}

-- Define a point P
def Point (a b : ℝ) : ℝ × ℝ := (a, b)

-- Theorem statement
theorem point_on_curve_iff_satisfies_equation (F : ℝ → ℝ → ℝ) (a b : ℝ) :
  Point a b ∈ Curve F ↔ F a b = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_iff_satisfies_equation_l3710_371065


namespace NUMINAMATH_CALUDE_gumballs_last_42_days_l3710_371094

/-- The number of gumballs Kim gets for each pair of earrings -/
def gumballs_per_pair : ℕ := 9

/-- The number of pairs of earrings Kim brings on day 1 -/
def day1_pairs : ℕ := 3

/-- The number of pairs of earrings Kim brings on day 2 -/
def day2_pairs : ℕ := 2 * day1_pairs

/-- The number of pairs of earrings Kim brings on day 3 -/
def day3_pairs : ℕ := day2_pairs - 1

/-- The number of gumballs Kim eats per day -/
def gumballs_eaten_per_day : ℕ := 3

/-- The total number of gumballs Kim receives -/
def total_gumballs : ℕ := gumballs_per_pair * (day1_pairs + day2_pairs + day3_pairs)

/-- The number of days the gumballs will last -/
def days_gumballs_last : ℕ := total_gumballs / gumballs_eaten_per_day

theorem gumballs_last_42_days : days_gumballs_last = 42 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_last_42_days_l3710_371094


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3710_371059

noncomputable def f (x : ℝ) : ℝ := x^3
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance_between_curves :
  ∃ (min_val : ℝ), min_val = (1/3 : ℝ) + (1/3 : ℝ) * Real.log 3 ∧
  ∀ (x : ℝ), x > 0 → |f x - g x| ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l3710_371059


namespace NUMINAMATH_CALUDE_elevator_initial_floor_l3710_371050

def elevator_problem (initial_floor final_floor top_floor down_move up_move1 up_move2 : ℕ) : Prop :=
  final_floor = top_floor ∧
  top_floor = 13 ∧
  final_floor = initial_floor - down_move + up_move1 + up_move2 ∧
  down_move = 7 ∧
  up_move1 = 3 ∧
  up_move2 = 8

theorem elevator_initial_floor :
  ∀ initial_floor final_floor top_floor down_move up_move1 up_move2 : ℕ,
    elevator_problem initial_floor final_floor top_floor down_move up_move1 up_move2 →
    initial_floor = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_elevator_initial_floor_l3710_371050


namespace NUMINAMATH_CALUDE_jos_number_l3710_371017

theorem jos_number (n : ℕ) : 
  (∃ k l : ℕ, n = 9 * k - 2 ∧ n = 6 * l - 4) ∧ 
  n < 100 ∧ 
  (∀ m : ℕ, m < 100 → (∃ k' l' : ℕ, m = 9 * k' - 2 ∧ m = 6 * l' - 4) → m ≤ n) → 
  n = 86 := by
sorry

end NUMINAMATH_CALUDE_jos_number_l3710_371017


namespace NUMINAMATH_CALUDE_goose_egg_calculation_l3710_371074

theorem goose_egg_calculation (total_survived : ℕ) 
  (hatch_rate : ℚ) (first_month_survival : ℚ) 
  (first_year_death : ℚ) (migration_rate : ℚ) 
  (predator_survival : ℚ) :
  hatch_rate = 1/3 →
  first_month_survival = 4/5 →
  first_year_death = 3/5 →
  migration_rate = 1/4 →
  predator_survival = 2/3 →
  total_survived = 140 →
  ∃ (total_eggs : ℕ), 
    total_eggs = 1050 ∧
    (total_eggs : ℚ) * hatch_rate * first_month_survival * (1 - first_year_death) * 
    (1 - migration_rate) * predator_survival = total_survived := by
  sorry

#eval 1050

end NUMINAMATH_CALUDE_goose_egg_calculation_l3710_371074


namespace NUMINAMATH_CALUDE_percent_relation_l3710_371048

theorem percent_relation (a b : ℝ) (h : a = 1.2 * b) : 4 * b = (10 / 3) * a := by sorry

end NUMINAMATH_CALUDE_percent_relation_l3710_371048


namespace NUMINAMATH_CALUDE_problem_solution_l3710_371078

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k - |x - 4|

-- Define the theorem
theorem problem_solution (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sol : Set.Icc (-1 : ℝ) 1 = {x : ℝ | f 1 (x + 4) ≥ 0})
  (h_eq : 1/a + 1/(2*b) + 1/(3*c) = 1) :
  1 = 1 ∧ (1/9)*a + (2/9)*b + (3/9)*c ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l3710_371078


namespace NUMINAMATH_CALUDE_geometry_rhyme_probability_l3710_371018

def geometry_letters : Finset Char := {'G', 'E', 'O', 'M', 'E', 'T', 'R', 'Y'}
def rhyme_letters : Finset Char := {'R', 'H', 'Y', 'M', 'E'}

theorem geometry_rhyme_probability :
  (geometry_letters ∩ rhyme_letters).card / geometry_letters.card = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometry_rhyme_probability_l3710_371018


namespace NUMINAMATH_CALUDE_lcm_equality_and_inequality_l3710_371076

theorem lcm_equality_and_inequality (a b c : ℕ) : 
  (Nat.lcm a (a + 5) = Nat.lcm b (b + 5) → a = b) ∧
  (Nat.lcm a b ≠ Nat.lcm (a + c) (b + c)) := by
  sorry

end NUMINAMATH_CALUDE_lcm_equality_and_inequality_l3710_371076


namespace NUMINAMATH_CALUDE_refurbished_to_new_tshirt_ratio_l3710_371095

/-- The price of a new T-shirt in dollars -/
def new_tshirt_price : ℚ := 5

/-- The price of a pair of pants in dollars -/
def pants_price : ℚ := 4

/-- The price of a skirt in dollars -/
def skirt_price : ℚ := 6

/-- The total income from selling 2 new T-shirts, 1 pair of pants, 4 skirts, and 6 refurbished T-shirts -/
def total_income : ℚ := 53

/-- The number of new T-shirts sold -/
def new_tshirts_sold : ℕ := 2

/-- The number of pants sold -/
def pants_sold : ℕ := 1

/-- The number of skirts sold -/
def skirts_sold : ℕ := 4

/-- The number of refurbished T-shirts sold -/
def refurbished_tshirts_sold : ℕ := 6

/-- Theorem stating that the ratio of the price of a refurbished T-shirt to the price of a new T-shirt is 1/2 -/
theorem refurbished_to_new_tshirt_ratio :
  (total_income - (new_tshirt_price * new_tshirts_sold + pants_price * pants_sold + skirt_price * skirts_sold)) / refurbished_tshirts_sold / new_tshirt_price = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_refurbished_to_new_tshirt_ratio_l3710_371095


namespace NUMINAMATH_CALUDE_polynomial_roots_condition_l3710_371042

open Real

/-- The polynomial in question -/
def polynomial (q x : ℝ) : ℝ := x^4 + 2*q*x^3 + 3*x^2 + 2*q*x + 2

/-- Predicate for a number being a root of the polynomial -/
def is_root (q x : ℝ) : Prop := polynomial q x = 0

/-- Theorem stating the condition for the polynomial to have at least two distinct negative real roots with product 2 -/
theorem polynomial_roots_condition (q : ℝ) : 
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x * y = 2 ∧ is_root q x ∧ is_root q y) ↔ q < -7 * sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_polynomial_roots_condition_l3710_371042


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l3710_371041

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ m : ℤ, m ≥ 120 ∧ (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) →
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l3710_371041


namespace NUMINAMATH_CALUDE_binary_representation_1023_l3710_371026

/-- Represents a binary expansion of a natural number -/
def BinaryExpansion (n : ℕ) : List Bool :=
  sorry

/-- Counts the number of true values in a list of booleans -/
def countOnes (l : List Bool) : ℕ :=
  sorry

/-- Calculates the sum of indices where the value is true -/
def sumIndices (l : List Bool) : ℕ :=
  sorry

theorem binary_representation_1023 :
  let binary := BinaryExpansion 1023
  (sumIndices binary = 45) ∧ (countOnes binary = 10) :=
sorry

end NUMINAMATH_CALUDE_binary_representation_1023_l3710_371026


namespace NUMINAMATH_CALUDE_smallest_sum_of_four_primes_l3710_371043

def is_prime (n : ℕ) : Prop := sorry

def digits (n : ℕ) : List ℕ := sorry

theorem smallest_sum_of_four_primes : 
  ∃ (a b c d : ℕ),
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧
    (10 < d) ∧ (d < 100) ∧
    (a < 10) ∧ (b < 10) ∧ (c < 10) ∧
    (digits a ++ digits b ++ digits c ++ digits d).Nodup ∧
    (digits a ++ digits b ++ digits c ++ digits d).length = 9 ∧
    (∀ i, i ∈ digits a ++ digits b ++ digits c ++ digits d → 1 ≤ i ∧ i ≤ 9) ∧
    a + b + c + d = 53 ∧
    (∀ w x y z : ℕ, 
      is_prime w ∧ is_prime x ∧ is_prime y ∧ is_prime z ∧
      (10 < z) ∧ (z < 100) ∧
      (w < 10) ∧ (x < 10) ∧ (y < 10) ∧
      (digits w ++ digits x ++ digits y ++ digits z).Nodup ∧
      (digits w ++ digits x ++ digits y ++ digits z).length = 9 ∧
      (∀ i, i ∈ digits w ++ digits x ++ digits y ++ digits z → 1 ≤ i ∧ i ≤ 9) →
      w + x + y + z ≥ 53) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_four_primes_l3710_371043


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3710_371051

/-- Given two points (5, 3) and (-7, 9) as endpoints of a circle's diameter,
    prove that the sum of the coordinates of the circle's center is 5. -/
theorem circle_center_coordinate_sum : 
  let p1 : ℝ × ℝ := (5, 3)
  let p2 : ℝ × ℝ := (-7, 9)
  let center : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  center.1 + center.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3710_371051


namespace NUMINAMATH_CALUDE_root_equation_l3710_371055

noncomputable def f (x : ℝ) : ℝ := if x < 0 then -2*x else x^2 - 1

theorem root_equation (a : ℝ) :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧
  (∀ x : ℝ, f x + 2 * Real.sqrt (1 - x^2) + |f x - 2 * Real.sqrt (1 - x^2)| - 2*a*x - 4 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
  x₃ - x₂ = 2*(x₂ - x₁) →
  a = (Real.sqrt 17 - 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_root_equation_l3710_371055


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l3710_371089

theorem multiply_and_simplify (x : ℝ) : 
  (x^4 + 6*x^2 + 9) * (x^2 - 3) = x^4 + 6*x^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l3710_371089


namespace NUMINAMATH_CALUDE_dina_has_60_dolls_l3710_371030

/-- The number of dolls Ivy has -/
def ivy_dolls : ℕ := 30

/-- The number of dolls Dina has -/
def dina_dolls : ℕ := 2 * ivy_dolls

/-- The number of collectors edition dolls Ivy has -/
def ivy_collectors : ℕ := 20

theorem dina_has_60_dolls :
  (2 * ivy_dolls = dina_dolls) →
  (2 * ivy_collectors = 3 * ivy_dolls) →
  (ivy_collectors = 20) →
  dina_dolls = 60 := by
  sorry

end NUMINAMATH_CALUDE_dina_has_60_dolls_l3710_371030


namespace NUMINAMATH_CALUDE_largest_five_digit_sum_20_l3710_371005

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digit_sum (n : ℕ) : ℕ := 
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_five_digit_sum_20 : 
  ∀ n : ℕ, is_five_digit n → digit_sum n = 20 → n ≤ 99200 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_sum_20_l3710_371005


namespace NUMINAMATH_CALUDE_twenty_one_three_four_zero_is_base5_l3710_371099

def is_base5_digit (d : Nat) : Prop := d < 5

def is_base5_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 5 → is_base5_digit d

theorem twenty_one_three_four_zero_is_base5 :
  is_base5_number 21340 :=
sorry

end NUMINAMATH_CALUDE_twenty_one_three_four_zero_is_base5_l3710_371099


namespace NUMINAMATH_CALUDE_cube_corners_equivalence_l3710_371084

/-- A corner piece consists of three 1x1x1 cubes -/
def corner_piece : ℕ := 3

/-- The dimensions of the cube -/
def cube_dimension : ℕ := 3

/-- The number of corner pieces -/
def num_corners : ℕ := 9

/-- Theorem: The total number of 1x1x1 cubes in a 3x3x3 cube 
    is equal to the total number of 1x1x1 cubes in 9 corner pieces -/
theorem cube_corners_equivalence : 
  cube_dimension ^ 3 = num_corners * corner_piece := by
  sorry

end NUMINAMATH_CALUDE_cube_corners_equivalence_l3710_371084


namespace NUMINAMATH_CALUDE_tangent_line_at_y_axis_l3710_371044

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x + 1)

theorem tangent_line_at_y_axis (x : ℝ) :
  let y_intercept := f 0
  let slope := (deriv f) 0
  (fun x => slope * x + y_intercept) = (fun x => 2 * Real.exp 1 * x + Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_y_axis_l3710_371044


namespace NUMINAMATH_CALUDE_max_matches_theorem_l3710_371087

/-- The maximum number of matches in a table tennis tournament -/
def max_matches : ℕ := 120

/-- Represents the number of players in each team -/
structure TeamSizes where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the total number of matches given team sizes -/
def calculate_matches (teams : TeamSizes) : ℕ :=
  teams.x * teams.y + teams.y * teams.z + teams.x * teams.z

/-- Theorem stating the maximum number of matches -/
theorem max_matches_theorem :
  ∀ (teams : TeamSizes),
  teams.x + teams.y + teams.z = 19 →
  calculate_matches teams ≤ max_matches :=
by sorry

end NUMINAMATH_CALUDE_max_matches_theorem_l3710_371087


namespace NUMINAMATH_CALUDE_searchlight_revolutions_per_minute_l3710_371063

/-- 
Given a searchlight that completes one revolution in a time period where 
half of that period is 10 seconds of darkness, prove that the number of 
revolutions per minute is 3.
-/
theorem searchlight_revolutions_per_minute : 
  ∀ (r : ℝ), 
  (r > 0) →  -- r is positive (revolutions per minute)
  (60 / r / 2 = 10) →  -- half the period of one revolution is 10 seconds
  r = 3 := by sorry

end NUMINAMATH_CALUDE_searchlight_revolutions_per_minute_l3710_371063


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3710_371035

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) :
  (∀ n, a (n + 1) - a n = 2) →  -- arithmetic sequence with common difference 2
  (a 4)^2 = a 2 * a 5 →  -- a_2, a_4, a_5 form a geometric sequence
  a 2 = -8 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3710_371035


namespace NUMINAMATH_CALUDE_homologous_functions_count_l3710_371009

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the range
def range : Set ℝ := {1, 3}

-- Define a valid domain
def is_valid_domain (D : Set ℝ) : Prop :=
  (∀ x ∈ D, f x ∈ range) ∧ (∀ y ∈ range, ∃ x ∈ D, f x = y)

-- Theorem statement
theorem homologous_functions_count :
  ∃! (domains : Finset (Set ℝ)), 
    domains.card = 3 ∧ 
    (∀ D ∈ domains, is_valid_domain D) ∧
    (∀ D : Set ℝ, is_valid_domain D → D ∈ domains) :=
sorry

end NUMINAMATH_CALUDE_homologous_functions_count_l3710_371009
