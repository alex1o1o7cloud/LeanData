import Mathlib

namespace NUMINAMATH_CALUDE_gcd_38_23_l2679_267969

theorem gcd_38_23 : Nat.gcd 38 23 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_38_23_l2679_267969


namespace NUMINAMATH_CALUDE_right_angled_triangle_set_l2679_267990

theorem right_angled_triangle_set : 
  ∃! (a b c : ℝ), (a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 3) ∧ 
  a^2 + b^2 = c^2 ∧ 
  ((a = 2 ∧ b = 3 ∧ c = 4) ∨ 
   (a = 1 ∧ b = 1 ∧ c = 2) ∨ 
   (a = 5 ∧ b = 12 ∧ c = 15) ∨ 
   (a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_set_l2679_267990


namespace NUMINAMATH_CALUDE_odd_operations_l2679_267901

theorem odd_operations (a b : ℤ) (ha : Odd a) (hb : Odd b) :
  Odd (a * b) ∧ Odd (a ^ 2) ∧ ¬(∀ x y : ℤ, Odd x → Odd y → Odd (x + y)) ∧ ¬(∀ x y : ℤ, Odd x → Odd y → Odd (x - y)) :=
by sorry

end NUMINAMATH_CALUDE_odd_operations_l2679_267901


namespace NUMINAMATH_CALUDE_range_of_m_l2679_267958

def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m (m : ℝ) : 
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (m < -2 ∨ m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2679_267958


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2679_267953

theorem polynomial_simplification (x : ℝ) :
  (3 * x^5 - 2 * x^3 + 5 * x^2 - 8 * x + 6) + (7 * x^4 + x^3 - 3 * x^2 + x - 9) =
  3 * x^5 + 7 * x^4 - x^3 + 2 * x^2 - 7 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2679_267953


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2679_267991

theorem pie_eating_contest (first_student second_student : ℚ) 
  (h1 : first_student = 6/7)
  (h2 : second_student = 3/4) :
  first_student - second_student = 3/28 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2679_267991


namespace NUMINAMATH_CALUDE_james_weekly_earnings_l2679_267931

/-- Calculates the weekly earnings from car rental -/
def weekly_earnings (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hourly_rate * hours_per_day * days_per_week

/-- Proves that James' weekly earnings from car rental are $640 -/
theorem james_weekly_earnings :
  weekly_earnings 20 8 4 = 640 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_earnings_l2679_267931


namespace NUMINAMATH_CALUDE_min_point_of_translated_graph_l2679_267968

-- Define the function
def f (x : ℝ) : ℝ := |x - 1|^2 - 7

-- State the theorem
theorem min_point_of_translated_graph :
  ∃! p : ℝ × ℝ, p.1 = 1 ∧ p.2 = -7 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
sorry

end NUMINAMATH_CALUDE_min_point_of_translated_graph_l2679_267968


namespace NUMINAMATH_CALUDE_least_sum_m_n_l2679_267963

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (Nat.gcd (m + n) 231 = 1) ∧ 
  (∃ (k : ℕ), m ^ m.val = k * (n ^ n.val)) ∧ 
  (∀ (k : ℕ+), m ≠ k * n) ∧
  (m + n = 75) ∧
  (∀ (m' n' : ℕ+), 
    (Nat.gcd (m' + n') 231 = 1) → 
    (∃ (k : ℕ), m' ^ m'.val = k * (n' ^ n'.val)) → 
    (∀ (k : ℕ+), m' ≠ k * n') → 
    (m' + n' ≥ 75)) :=
sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l2679_267963


namespace NUMINAMATH_CALUDE_parabola_focus_centroid_l2679_267981

/-- Given three points A, B, C in a 2D plane, and a parabola y^2 = ax,
    if the focus of the parabola is exactly the centroid of triangle ABC,
    then a = 8. -/
theorem parabola_focus_centroid (A B C : ℝ × ℝ) (a : ℝ) : 
  A = (-1, 2) →
  B = (3, 4) →
  C = (4, -6) →
  let centroid := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  let focus := (a / 4, 0)
  centroid = focus →
  a = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_centroid_l2679_267981


namespace NUMINAMATH_CALUDE_function_inequality_l2679_267917

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property that f is differentiable
variable (hf : Differentiable ℝ f)

-- Define the condition f'(x) + f(x) < 0
variable (hf' : ∀ x, HasDerivAt f (f x) x → (deriv f x + f x < 0))

-- Define m as a real number
variable (m : ℝ)

-- State the theorem
theorem function_inequality :
  f (m - m^2) > Real.exp (m^2 - m + 1) * f 1 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l2679_267917


namespace NUMINAMATH_CALUDE_total_points_is_65_l2679_267945

/-- Represents the types of enemies in the game -/
inductive EnemyType
  | A
  | B
  | C

/-- The number of points earned for defeating each type of enemy -/
def pointsForEnemy (t : EnemyType) : ℕ :=
  match t with
  | EnemyType.A => 10
  | EnemyType.B => 15
  | EnemyType.C => 20

/-- The total number of enemies in the level -/
def totalEnemies : ℕ := 8

/-- The number of each type of enemy in the level -/
def enemyCount (t : EnemyType) : ℕ :=
  match t with
  | EnemyType.A => 3
  | EnemyType.B => 2
  | EnemyType.C => 3

/-- The number of enemies defeated for each type -/
def defeatedCount (t : EnemyType) : ℕ :=
  match t with
  | EnemyType.A => 3  -- All Type A enemies
  | EnemyType.B => 1  -- Half of Type B enemies
  | EnemyType.C => 1  -- One Type C enemy

/-- Calculates the total points earned -/
def totalPointsEarned : ℕ :=
  (defeatedCount EnemyType.A * pointsForEnemy EnemyType.A) +
  (defeatedCount EnemyType.B * pointsForEnemy EnemyType.B) +
  (defeatedCount EnemyType.C * pointsForEnemy EnemyType.C)

/-- Theorem stating that the total points earned is 65 -/
theorem total_points_is_65 : totalPointsEarned = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_65_l2679_267945


namespace NUMINAMATH_CALUDE_monday_sales_l2679_267965

/-- Represents the sales and pricing of a shoe store -/
structure ShoeStore where
  shoe_price : ℕ
  boot_price : ℕ
  monday_shoe_sales : ℕ
  monday_boot_sales : ℕ
  tuesday_shoe_sales : ℕ
  tuesday_boot_sales : ℕ
  tuesday_total_sales : ℕ

/-- The conditions of the problem -/
def store_conditions (s : ShoeStore) : Prop :=
  s.boot_price = s.shoe_price + 15 ∧
  s.monday_shoe_sales = 22 ∧
  s.monday_boot_sales = 16 ∧
  s.tuesday_shoe_sales = 8 ∧
  s.tuesday_boot_sales = 32 ∧
  s.tuesday_total_sales = 560 ∧
  s.tuesday_shoe_sales * s.shoe_price + s.tuesday_boot_sales * s.boot_price = s.tuesday_total_sales

/-- The theorem to be proved -/
theorem monday_sales (s : ShoeStore) (h : store_conditions s) : 
  s.monday_shoe_sales * s.shoe_price + s.monday_boot_sales * s.boot_price = 316 := by
  sorry


end NUMINAMATH_CALUDE_monday_sales_l2679_267965


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l2679_267936

theorem weekend_rain_probability
  (p_friday : ℝ)
  (p_saturday_given_friday : ℝ)
  (p_saturday_given_not_friday : ℝ)
  (p_sunday : ℝ)
  (h1 : p_friday = 0.3)
  (h2 : p_saturday_given_friday = 0.6)
  (h3 : p_saturday_given_not_friday = 0.25)
  (h4 : p_sunday = 0.4) :
  1 - (1 - p_friday) * (1 - p_saturday_given_not_friday * (1 - p_friday)) * (1 - p_sunday) = 0.685 := by
sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l2679_267936


namespace NUMINAMATH_CALUDE_purchase_price_is_31_l2679_267905

/-- Calculates the purchase price of a share given the dividend rate, face value, and return on investment. -/
def calculate_purchase_price (dividend_rate : ℚ) (face_value : ℚ) (roi : ℚ) : ℚ :=
  (dividend_rate * face_value) / roi

/-- Theorem stating that given the specific conditions, the purchase price is 31. -/
theorem purchase_price_is_31 :
  let dividend_rate : ℚ := 155 / 1000  -- 15.5%
  let face_value : ℚ := 50
  let roi : ℚ := 1 / 4  -- 25%
  calculate_purchase_price dividend_rate face_value roi = 31 := by
  sorry

#eval calculate_purchase_price (155 / 1000) 50 (1 / 4)

end NUMINAMATH_CALUDE_purchase_price_is_31_l2679_267905


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l2679_267916

theorem complex_sum_of_powers (i : ℂ) : i^2 = -1 → i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l2679_267916


namespace NUMINAMATH_CALUDE_max_value_expression_l2679_267964

theorem max_value_expression (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  ∃ (m : ℝ), m = 2 ∧ ∀ x y z w, 
    (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) → (0 ≤ w ∧ w ≤ 1) →
    x + y + z + w - x*y - y*z - z*w - w*x ≤ m :=
by
  sorry

#check max_value_expression

end NUMINAMATH_CALUDE_max_value_expression_l2679_267964


namespace NUMINAMATH_CALUDE_unique_valid_number_l2679_267996

def is_valid_number (n : ℕ) : Prop :=
  -- n is a four-digit number
  1000 ≤ n ∧ n < 10000 ∧
  -- n can be divided into two two-digit numbers
  let x := n / 100
  let y := n % 100
  10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧
  -- Adding a 0 to the end of the first two-digit number and adding it to the product of the two two-digit numbers equals the original four-digit number
  10 * x + x * y = n ∧
  -- The unit digit of the original number is 5
  n % 10 = 5

theorem unique_valid_number : ∃! n : ℕ, is_valid_number n ∧ n = 1995 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2679_267996


namespace NUMINAMATH_CALUDE_lilly_fish_count_l2679_267912

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 9

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 19

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := total_fish - rosy_fish

theorem lilly_fish_count : lilly_fish = 10 := by
  sorry

end NUMINAMATH_CALUDE_lilly_fish_count_l2679_267912


namespace NUMINAMATH_CALUDE_tangent_sum_problem_l2679_267955

theorem tangent_sum_problem (x y m : ℝ) :
  x^3 + Real.sin (2*x) = m →
  y^3 + Real.sin (2*y) = -m →
  x ∈ Set.Ioo (-π/4) (π/4) →
  y ∈ Set.Ioo (-π/4) (π/4) →
  Real.tan (x + y + π/3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_problem_l2679_267955


namespace NUMINAMATH_CALUDE_translated_minimum_point_l2679_267988

def f (x : ℝ) : ℝ := |x| - 4

def g (x : ℝ) : ℝ := f (x - 3) - 4

theorem translated_minimum_point :
  ∃ (x : ℝ), ∀ (y : ℝ), g y ≥ g x ∧ g x = -8 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_translated_minimum_point_l2679_267988


namespace NUMINAMATH_CALUDE_x_value_theorem_l2679_267985

theorem x_value_theorem (x n : ℕ) (h1 : x = 2^n - 32) 
  (h2 : (Nat.factors x).card = 3) 
  (h3 : 3 ∈ Nat.factors x) : 
  x = 480 ∨ x = 2016 := by
sorry

end NUMINAMATH_CALUDE_x_value_theorem_l2679_267985


namespace NUMINAMATH_CALUDE_smaller_circle_area_l2679_267960

/-- Two circles are externally tangent with common tangents. Given specific conditions, 
    prove that the area of the smaller circle is 5π/3. -/
theorem smaller_circle_area (r : ℝ) : 
  r > 0 → -- radius of smaller circle is positive
  (∃ (P A B : ℝ × ℝ), 
    -- PA and AB are tangent lines
    dist P A = dist A B ∧ 
    dist P A = 5 ∧
    -- Larger circle has radius 3r
    (∃ (C : ℝ × ℝ), dist C B = 3 * r)) →
  π * r^2 = 5 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_area_l2679_267960


namespace NUMINAMATH_CALUDE_critical_point_and_zeros_l2679_267922

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + (1 - 3 * Real.log x) / a

theorem critical_point_and_zeros (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → (deriv (f a)) x = 0 → x = 1 → a = 1) ∧
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ↔ 0 < a ∧ a < Real.exp (-1)) :=
sorry

end NUMINAMATH_CALUDE_critical_point_and_zeros_l2679_267922


namespace NUMINAMATH_CALUDE_parallel_vectors_m_l2679_267980

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem parallel_vectors_m (m : ℝ) : 
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (-2, m)
  are_parallel a (a.1 + 2 * b.1, a.2 + 2 * b.2) → m = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_l2679_267980


namespace NUMINAMATH_CALUDE_percentage_problem_l2679_267900

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : 3 * (x / 100 * x) = 18) : x = 10 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2679_267900


namespace NUMINAMATH_CALUDE_problem_statement_l2679_267925

theorem problem_statement (p q r : ℝ) 
  (h1 : p < q)
  (h2 : ∀ x, ((x - p) * (x - q)) / (x - r) ≥ 0 ↔ (x > 5 ∨ (7 ≤ x ∧ x ≤ 15))) :
  p + 2*q + 3*r = 52 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2679_267925


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l2679_267986

/-- 
For a quadratic equation kx^2 - 2x - 1 = 0, where k is a real number,
the equation has two real roots if and only if k ≥ -1 and k ≠ 0.
-/
theorem quadratic_two_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2*x - 1 = 0 ∧ k * y^2 - 2*y - 1 = 0) ↔ 
  (k ≥ -1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l2679_267986


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2679_267935

theorem arithmetic_calculation : -1^4 * 8 - 2^3 / (-4) * (-7 + 5) = -12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2679_267935


namespace NUMINAMATH_CALUDE_fathers_age_l2679_267952

/-- Proves that given the conditions about the father's and Ming Ming's ages, the father's age this year is 35 -/
theorem fathers_age (ming_age ming_age_3_years_ago father_age father_age_3_years_ago : ℕ) :
  father_age_3_years_ago = 8 * ming_age_3_years_ago →
  father_age = 5 * ming_age →
  father_age = ming_age + 3 →
  father_age_3_years_ago = father_age - 3 →
  father_age = 35 := by
sorry


end NUMINAMATH_CALUDE_fathers_age_l2679_267952


namespace NUMINAMATH_CALUDE_inequality_comparison_l2679_267987

theorem inequality_comparison (x y : ℝ) (h : x > y) : -3*x + 5 < -3*y + 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_comparison_l2679_267987


namespace NUMINAMATH_CALUDE_max_value_of_cosine_function_l2679_267977

theorem max_value_of_cosine_function :
  ∀ x : ℝ, 4 * (Real.cos x)^3 - 3 * (Real.cos x)^2 - 6 * (Real.cos x) + 5 ≤ 27/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_cosine_function_l2679_267977


namespace NUMINAMATH_CALUDE_special_function_sum_negative_l2679_267976

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  odd : ∀ x, f (x + 2) = -f (-x + 2)
  mono : ∀ x y, x > 2 → y > 2 → x < y → f x < f y

/-- The main theorem -/
theorem special_function_sum_negative (F : SpecialFunction) 
  (x₁ x₂ : ℝ) (h1 : x₁ + x₂ < 4) (h2 : (x₁ - 2) * (x₂ - 2) < 0) :
  F.f x₁ + F.f x₂ < 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_sum_negative_l2679_267976


namespace NUMINAMATH_CALUDE_negative_one_greater_than_negative_sqrt_two_l2679_267940

theorem negative_one_greater_than_negative_sqrt_two :
  -1 > -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_greater_than_negative_sqrt_two_l2679_267940


namespace NUMINAMATH_CALUDE_lengths_form_triangle_l2679_267974

/-- A function that checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the lengths 4, 6, and 9 can form a triangle -/
theorem lengths_form_triangle : can_form_triangle 4 6 9 := by
  sorry

end NUMINAMATH_CALUDE_lengths_form_triangle_l2679_267974


namespace NUMINAMATH_CALUDE_point_coordinates_l2679_267929

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates (p : Point) 
  (h1 : isInThirdQuadrant p)
  (h2 : distanceToXAxis p = 2)
  (h3 : distanceToYAxis p = 5) :
  p = Point.mk (-5) (-2) := by
  sorry


end NUMINAMATH_CALUDE_point_coordinates_l2679_267929


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2679_267930

theorem complex_equation_solution (a : ℂ) :
  a / (1 - Complex.I) = (1 + Complex.I) / Complex.I → a = -2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2679_267930


namespace NUMINAMATH_CALUDE_mean_median_difference_l2679_267908

theorem mean_median_difference (x : ℕ) : 
  let set := [x, x + 2, x + 4, x + 7, x + 27]
  let median := x + 4
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 27)) / 5
  mean = median + 4 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l2679_267908


namespace NUMINAMATH_CALUDE_I_max_min_zero_l2679_267934

noncomputable def f (x : ℝ) : ℝ := x^2 + 3

noncomputable def g (a x : ℝ) : ℝ := a * x + 3

noncomputable def I (a : ℝ) : ℝ := 3 * ∫ x in (-1)..(1), |f x - g a x|

theorem I_max_min_zero :
  (∀ a : ℝ, I a ≤ 0) ∧ (∃ a : ℝ, I a = 0) :=
sorry

end NUMINAMATH_CALUDE_I_max_min_zero_l2679_267934


namespace NUMINAMATH_CALUDE_window_purchase_savings_l2679_267921

/-- Calculates the cost of purchasing windows under the given offer -/
def cost_with_offer (num_windows : ℕ) : ℕ :=
  ((num_windows + 4) / 7 * 5 + (num_windows + 4) % 7) * 100

/-- Represents the window purchase problem -/
theorem window_purchase_savings (dave_windows doug_windows : ℕ) 
  (h1 : dave_windows = 10) (h2 : doug_windows = 11) : 
  (dave_windows + doug_windows) * 100 - cost_with_offer (dave_windows + doug_windows) = 
  (dave_windows * 100 - cost_with_offer dave_windows) + 
  (doug_windows * 100 - cost_with_offer doug_windows) :=
sorry

end NUMINAMATH_CALUDE_window_purchase_savings_l2679_267921


namespace NUMINAMATH_CALUDE_function_comparison_l2679_267984

/-- A function f is strictly decreasing on the non-negative real numbers -/
def StrictlyDecreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₂ < f x₁

/-- An even function -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem function_comparison 
  (f : ℝ → ℝ) 
  (h_even : EvenFunction f)
  (h_decreasing : StrictlyDecreasingOnNonnegative f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 :=
sorry

end NUMINAMATH_CALUDE_function_comparison_l2679_267984


namespace NUMINAMATH_CALUDE_exists_subset_with_unique_adjacency_l2679_267954

def adjacent (p q : ℤ × ℤ × ℤ) : Prop :=
  let (x, y, z) := p
  let (u, v, w) := q
  abs (x - u) + abs (y - v) + abs (z - w) = 1

theorem exists_subset_with_unique_adjacency :
  ∃ (S : Set (ℤ × ℤ × ℤ)), ∀ p : ℤ × ℤ × ℤ,
    (p ∈ S ∧ ∀ q, adjacent p q → q ∉ S) ∨
    (p ∉ S ∧ ∃! q, adjacent p q ∧ q ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_exists_subset_with_unique_adjacency_l2679_267954


namespace NUMINAMATH_CALUDE_symmetrical_line_intersection_l2679_267933

/-- Given points A and B, if the line symmetrical to AB about y=a intersects
    the circle (x+3)^2 + (y+2)^2 = 1, then 1/3 ≤ a ≤ 3/2 -/
theorem symmetrical_line_intersection (a : ℝ) : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (0, a)
  let symmetrical_line (x y : ℝ) := (3 - a) * x - 2 * y + 2 * a = 0
  let circle (x y : ℝ) := (x + 3)^2 + (y + 2)^2 = 1
  (∃ x y, symmetrical_line x y ∧ circle x y) → 1/3 ≤ a ∧ a ≤ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_symmetrical_line_intersection_l2679_267933


namespace NUMINAMATH_CALUDE_factorial_3_equals_6_l2679_267926

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_3_equals_6 : factorial 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_3_equals_6_l2679_267926


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2679_267973

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 1, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2679_267973


namespace NUMINAMATH_CALUDE_smallest_square_longest_ending_sequence_l2679_267939

/-- A function that returns the length of the longest sequence of the same non-zero digit at the end of a number -/
def longestEndingSequence (n : ℕ) : ℕ :=
  sorry

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  sorry

/-- The theorem stating that 1444 is the smallest square with the longest ending sequence of same non-zero digits -/
theorem smallest_square_longest_ending_sequence :
  ∀ n : ℕ, isPerfectSquare n → n ≠ 1444 → longestEndingSequence n ≤ longestEndingSequence 1444 :=
sorry

end NUMINAMATH_CALUDE_smallest_square_longest_ending_sequence_l2679_267939


namespace NUMINAMATH_CALUDE_adam_laundry_theorem_l2679_267920

/-- Given a total number of laundry loads and the number of loads already washed,
    calculate the number of loads still to be washed. -/
def loads_remaining (total : ℕ) (washed : ℕ) : ℕ :=
  total - washed

/-- Theorem: Given 25 total loads and 6 washed loads, 19 loads remain to be washed. -/
theorem adam_laundry_theorem :
  loads_remaining 25 6 = 19 := by
  sorry

end NUMINAMATH_CALUDE_adam_laundry_theorem_l2679_267920


namespace NUMINAMATH_CALUDE_cylinder_volume_theorem_l2679_267978

/-- The volume of a cylinder with a rectangular net of dimensions 2a and a -/
def cylinder_volume (a : ℝ) : Set ℝ :=
  {v | v = a^3 / Real.pi ∨ v = a^3 / (2 * Real.pi)}

/-- Theorem stating that the volume of the cylinder is either a³/π or a³/(2π) -/
theorem cylinder_volume_theorem (a : ℝ) (h : a > 0) :
  ∀ v ∈ cylinder_volume a, v = a^3 / Real.pi ∨ v = a^3 / (2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_theorem_l2679_267978


namespace NUMINAMATH_CALUDE_binomial_expansion_product_l2679_267994

theorem binomial_expansion_product (a a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x, (1 - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) : 
  (a + a₂ + a₄) * (a₁ + a₃ + a₅) = -256 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_product_l2679_267994


namespace NUMINAMATH_CALUDE_car_distance_ratio_l2679_267938

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ  -- Speed in km/hr
  time : ℝ   -- Time in hours

/-- Calculates the distance traveled by a car -/
def distance (c : Car) : ℝ := c.speed * c.time

/-- The problem statement -/
theorem car_distance_ratio :
  let car_a : Car := ⟨50, 8⟩
  let car_b : Car := ⟨25, 4⟩
  (distance car_a) / (distance car_b) = 4
  := by sorry

end NUMINAMATH_CALUDE_car_distance_ratio_l2679_267938


namespace NUMINAMATH_CALUDE_sets_equal_iff_m_eq_neg_two_sqrt_two_l2679_267918

def A (m : ℝ) : Set ℝ := {x | x^2 + m*x + 2 ≥ 0 ∧ x ≥ 0}

def B (m : ℝ) : Set ℝ := {y | ∃ x ∈ A m, y = Real.sqrt (x^2 + m*x + 2)}

theorem sets_equal_iff_m_eq_neg_two_sqrt_two (m : ℝ) :
  A m = B m ↔ m = -2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sets_equal_iff_m_eq_neg_two_sqrt_two_l2679_267918


namespace NUMINAMATH_CALUDE_trivia_team_grouping_l2679_267961

theorem trivia_team_grouping (total_students : ℕ) (students_not_picked : ℕ) (num_groups : ℕ)
  (h1 : total_students = 120)
  (h2 : students_not_picked = 22)
  (h3 : num_groups = 14)
  : (total_students - students_not_picked) / num_groups = 7 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_grouping_l2679_267961


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2679_267903

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2679_267903


namespace NUMINAMATH_CALUDE_tangent_slope_angle_range_l2679_267956

theorem tangent_slope_angle_range (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n = Real.sqrt 3 / 2) :
  let f : ℝ → ℝ := λ x ↦ (1/3) * x^3 + n^2 * x
  let k := (m^2 + n^2)
  let θ := Real.arctan k
  θ ∈ Set.Ici (π/3) ∩ Set.Iio (π/2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_range_l2679_267956


namespace NUMINAMATH_CALUDE_farm_output_growth_equation_l2679_267919

/-- Represents the relationship between initial value, final value, and growth rate over two years -/
theorem farm_output_growth_equation (initial_value final_value : ℝ) (growth_rate : ℝ) : 
  initial_value = 80 → final_value = 96.8 → 
  initial_value * (1 + growth_rate)^2 = final_value :=
by
  sorry

#check farm_output_growth_equation

end NUMINAMATH_CALUDE_farm_output_growth_equation_l2679_267919


namespace NUMINAMATH_CALUDE_inequality_proof_l2679_267946

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (a * b + 1) * (b + 1) ≥ (27 / 4) * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2679_267946


namespace NUMINAMATH_CALUDE_perfect_square_m_l2679_267997

theorem perfect_square_m (l m n : ℕ+) (p : ℕ) (h_prime : Prime p) 
  (h_perfect_square : ∃ k : ℕ, p^(2*l.val - 1) * m.val * (m.val * n.val + 1)^2 + m.val^2 = k^2) :
  ∃ r : ℕ, m.val = r^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_m_l2679_267997


namespace NUMINAMATH_CALUDE_inequality_proof_l2679_267937

theorem inequality_proof :
  (∀ x : ℝ, |x - 1| + |x - 2| ≥ 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    (1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) →
    a + 2 * b + 3 * c ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2679_267937


namespace NUMINAMATH_CALUDE_program_duration_l2679_267913

/-- Proves that the duration of each program is 30 minutes -/
theorem program_duration (num_programs : ℕ) (commercial_fraction : ℚ) (total_commercial_time : ℕ) :
  num_programs = 6 →
  commercial_fraction = 1/4 →
  total_commercial_time = 45 →
  ∃ (program_duration : ℕ),
    program_duration = 30 ∧
    (↑num_programs * commercial_fraction * ↑program_duration = ↑total_commercial_time) :=
by
  sorry

end NUMINAMATH_CALUDE_program_duration_l2679_267913


namespace NUMINAMATH_CALUDE_function_composition_l2679_267983

theorem function_composition (f : ℝ → ℝ) :
  (∀ x, f (x - 2) = 3 * x - 5) → (∀ x, f x = 3 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l2679_267983


namespace NUMINAMATH_CALUDE_cistern_fill_time_l2679_267950

theorem cistern_fill_time (empty_time second_tap : ℝ) (fill_time_both : ℝ) 
  (h1 : empty_time = 9)
  (h2 : fill_time_both = 7.2) : 
  ∃ (fill_time_first : ℝ), 
    fill_time_first = 4 ∧ 
    (1 / fill_time_first - 1 / empty_time = 1 / fill_time_both) :=
by sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l2679_267950


namespace NUMINAMATH_CALUDE_quarter_difference_in_nickels_l2679_267995

/-- The number of nickels in a quarter -/
def nickels_per_quarter : ℕ := 5

/-- The difference in nickels between two amounts of quarters -/
def nickel_difference (alice_quarters bob_quarters : ℕ) : ℤ :=
  (alice_quarters - bob_quarters) * nickels_per_quarter

theorem quarter_difference_in_nickels (q : ℕ) :
  nickel_difference (4 * q + 3) (2 * q + 8) = 10 * q - 25 := by sorry

end NUMINAMATH_CALUDE_quarter_difference_in_nickels_l2679_267995


namespace NUMINAMATH_CALUDE_sum_of_roots_l2679_267989

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x + 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2679_267989


namespace NUMINAMATH_CALUDE_average_bacon_calculation_l2679_267904

-- Define the price per pound of bacon
def price_per_pound : ℝ := 6

-- Define the revenue from a half-size pig
def revenue_from_half_pig : ℝ := 60

-- Define the average amount of bacon from a pig
def average_bacon_amount : ℝ := 20

-- Theorem statement
theorem average_bacon_calculation :
  price_per_pound * (average_bacon_amount / 2) = revenue_from_half_pig :=
by sorry

end NUMINAMATH_CALUDE_average_bacon_calculation_l2679_267904


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2679_267966

theorem system_of_equations_solution (x y z : ℝ) 
  (eq1 : 4 * x - 6 * y - 2 * z = 0)
  (eq2 : 2 * x + 6 * y - 28 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 - 6*x*y) / (y^2 + 4*z^2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2679_267966


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_100_l2679_267959

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_greater_than_100 :
  (¬ ∃ n : ℕ, 2^n > 100) ↔ (∀ n : ℕ, 2^n ≤ 100) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_100_l2679_267959


namespace NUMINAMATH_CALUDE_cubeRoot_of_negative_eight_eq_negative_two_l2679_267947

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem cubeRoot_of_negative_eight_eq_negative_two :
  cubeRoot (-8) = -2 := by sorry

end NUMINAMATH_CALUDE_cubeRoot_of_negative_eight_eq_negative_two_l2679_267947


namespace NUMINAMATH_CALUDE_math_books_together_probability_l2679_267951

def total_textbooks : ℕ := 15
def math_textbooks : ℕ := 4
def box_sizes : List ℕ := [4, 5, 6]

def is_valid_distribution (dist : List ℕ) : Prop :=
  dist.length = 3 ∧ 
  dist.sum = total_textbooks ∧
  ∀ b ∈ dist, b ≤ total_textbooks - math_textbooks + 1

def probability_math_books_together : ℚ :=
  4 / 273

theorem math_books_together_probability :
  probability_math_books_together = 
    (number_of_valid_distributions_with_math_books_together : ℚ) / 
    (total_number_of_valid_distributions : ℚ) :=
by sorry

#check math_books_together_probability

end NUMINAMATH_CALUDE_math_books_together_probability_l2679_267951


namespace NUMINAMATH_CALUDE_kelly_apples_l2679_267941

theorem kelly_apples (initial_apples target_apples : ℕ) 
  (h1 : initial_apples = 128) 
  (h2 : target_apples = 250) : 
  target_apples - initial_apples = 122 := by
  sorry

end NUMINAMATH_CALUDE_kelly_apples_l2679_267941


namespace NUMINAMATH_CALUDE_max_right_triangle_area_in_rectangle_l2679_267911

theorem max_right_triangle_area_in_rectangle :
  ∀ (a b : ℝ),
  a = 12 ∧ b = 5 →
  (∀ (x y : ℝ),
    x ≤ a ∧ y ≤ b →
    x * y / 2 ≤ 30) ∧
  ∃ (x y : ℝ),
    x ≤ a ∧ y ≤ b ∧
    x * y / 2 = 30 :=
by sorry

end NUMINAMATH_CALUDE_max_right_triangle_area_in_rectangle_l2679_267911


namespace NUMINAMATH_CALUDE_vector_equality_implies_equal_norm_vector_equality_transitivity_l2679_267942

variable {V : Type*} [NormedAddCommGroup V]

theorem vector_equality_implies_equal_norm (a b : V) :
  a = b → ‖a‖ = ‖b‖ := by sorry

theorem vector_equality_transitivity (a b c : V) :
  a = b → b = c → a = c := by sorry

end NUMINAMATH_CALUDE_vector_equality_implies_equal_norm_vector_equality_transitivity_l2679_267942


namespace NUMINAMATH_CALUDE_scooter_price_l2679_267928

/-- The total price of a scooter given the upfront payment and the percentage it represents -/
theorem scooter_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 ∧ 
  upfront_percentage = 20 ∧ 
  upfront_payment = (upfront_percentage / 100) * total_price →
  total_price = 1200 := by
sorry

end NUMINAMATH_CALUDE_scooter_price_l2679_267928


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l2679_267972

/-- A function satisfying the given inequality for all real numbers -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x * z) - f x * f (y * z) ≥ 1

/-- The theorem stating that there is a unique function satisfying the inequality -/
theorem unique_satisfying_function :
  ∃! f : ℝ → ℝ, SatisfiesInequality f ∧ ∀ x : ℝ, f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l2679_267972


namespace NUMINAMATH_CALUDE_litter_bag_weight_l2679_267910

theorem litter_bag_weight (gina_bags : ℕ) (neighborhood_multiplier : ℕ) (total_weight : ℕ) :
  gina_bags = 2 →
  neighborhood_multiplier = 82 →
  total_weight = 664 →
  ∃ (bag_weight : ℕ), 
    bag_weight = 4 ∧ 
    (gina_bags + neighborhood_multiplier * gina_bags) * bag_weight = total_weight :=
by sorry

end NUMINAMATH_CALUDE_litter_bag_weight_l2679_267910


namespace NUMINAMATH_CALUDE_lee_cookies_l2679_267927

/-- Given that Lee can make 24 cookies with 3 cups of flour,
    this function calculates how many cookies he can make with any amount of flour. -/
def cookies_from_flour (flour : ℚ) : ℚ :=
  (24 * flour) / 3

/-- Theorem stating that Lee can make 40 cookies with 5 cups of flour. -/
theorem lee_cookies : cookies_from_flour 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_l2679_267927


namespace NUMINAMATH_CALUDE_rectangular_field_dimension_exists_unique_l2679_267923

theorem rectangular_field_dimension_exists_unique (area : ℝ) :
  ∃! m : ℝ, m > 0 ∧ (3 * m + 8) * (m - 3) = area :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_dimension_exists_unique_l2679_267923


namespace NUMINAMATH_CALUDE_harry_pizza_order_cost_l2679_267943

/-- Calculates the total cost of Harry's pizza order -/
theorem harry_pizza_order_cost :
  let large_pizza_cost : ℚ := 14
  let topping_cost : ℚ := 2
  let num_pizzas : ℕ := 2
  let num_toppings : ℕ := 3
  let tip_percentage : ℚ := 25 / 100

  let pizza_with_toppings_cost := large_pizza_cost + num_toppings * topping_cost
  let total_pizza_cost := num_pizzas * pizza_with_toppings_cost
  let tip_amount := tip_percentage * total_pizza_cost
  let total_cost := total_pizza_cost + tip_amount

  total_cost = 50 := by sorry

end NUMINAMATH_CALUDE_harry_pizza_order_cost_l2679_267943


namespace NUMINAMATH_CALUDE_carbon_copies_invariant_l2679_267970

/-- Represents a stack of sheets with carbon paper -/
structure CarbonPaperStack :=
  (num_sheets : ℕ)
  (carbons_between : ℕ)

/-- Calculates the number of carbon copies produced by a stack -/
def carbon_copies (stack : CarbonPaperStack) : ℕ :=
  max 0 (stack.num_sheets - 1)

/-- Represents a folding operation on the stack -/
inductive FoldOperation
  | UpperLower
  | LeftRight
  | BackFront

/-- Applies a sequence of folding operations to a stack -/
def apply_folds (stack : CarbonPaperStack) (folds : List FoldOperation) : CarbonPaperStack :=
  stack

theorem carbon_copies_invariant (initial_stack : CarbonPaperStack) (folds : List FoldOperation) :
  initial_stack.num_sheets = 6 ∧ initial_stack.carbons_between = 2 →
  carbon_copies initial_stack = carbon_copies (apply_folds initial_stack folds) ∧
  carbon_copies initial_stack = 5 :=
sorry

end NUMINAMATH_CALUDE_carbon_copies_invariant_l2679_267970


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l2679_267907

theorem factorization_of_quadratic (x : ℝ) : x^2 - 5*x = x*(x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l2679_267907


namespace NUMINAMATH_CALUDE_number_of_female_students_l2679_267924

theorem number_of_female_students 
  (total_average : ℝ) 
  (num_male : ℕ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (h1 : total_average = 90)
  (h2 : num_male = 8)
  (h3 : male_average = 82)
  (h4 : female_average = 92) :
  ∃ (num_female : ℕ), 
    (num_male : ℝ) * male_average + (num_female : ℝ) * female_average = 
    ((num_male : ℝ) + (num_female : ℝ)) * total_average ∧ 
    num_female = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_of_female_students_l2679_267924


namespace NUMINAMATH_CALUDE_medication_price_reduction_l2679_267967

theorem medication_price_reduction (P : ℝ) (r : ℝ) : 
  P * (1 - r)^2 = 100 →
  P * (1 - r)^2 = P * 0.81 →
  0 < r →
  r < 1 →
  P * (1 - r)^3 = 90 := by
sorry

end NUMINAMATH_CALUDE_medication_price_reduction_l2679_267967


namespace NUMINAMATH_CALUDE_booklet_word_count_l2679_267998

theorem booklet_word_count (words_per_page : ℕ) : 
  words_per_page ≤ 150 →
  (120 * words_per_page) % 221 = 172 →
  words_per_page = 114 := by
sorry

end NUMINAMATH_CALUDE_booklet_word_count_l2679_267998


namespace NUMINAMATH_CALUDE_stamp_collection_value_l2679_267971

theorem stamp_collection_value 
  (total_stamps : ℕ) 
  (sample_stamps : ℕ) 
  (sample_value : ℝ) 
  (h1 : total_stamps = 20)
  (h2 : sample_stamps = 4)
  (h3 : sample_value = 16) :
  (total_stamps : ℝ) * (sample_value / sample_stamps) = 80 :=
by sorry

end NUMINAMATH_CALUDE_stamp_collection_value_l2679_267971


namespace NUMINAMATH_CALUDE_tangent_parabola_to_line_l2679_267999

theorem tangent_parabola_to_line (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 1 = x ∧ ∀ y : ℝ, y ≠ x → a * y^2 + 1 ≠ y) → a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parabola_to_line_l2679_267999


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l2679_267957

/-- The quadratic equation 3x^2 + 2x - 5 = 0 has exactly two distinct real solutions. -/
theorem parabola_x_intercepts :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  3 * x₁^2 + 2 * x₁ - 5 = 0 ∧
  3 * x₂^2 + 2 * x₂ - 5 = 0 ∧
  ∀ (x : ℝ), 3 * x^2 + 2 * x - 5 = 0 → (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l2679_267957


namespace NUMINAMATH_CALUDE_paint_usage_l2679_267906

def paint_problem (paint_large : ℕ) (paint_small : ℕ) (num_large : ℕ) (num_small : ℕ) : Prop :=
  paint_large * num_large + paint_small * num_small = 17

theorem paint_usage : paint_problem 3 2 3 4 := by
  sorry

end NUMINAMATH_CALUDE_paint_usage_l2679_267906


namespace NUMINAMATH_CALUDE_ellipse_foci_condition_l2679_267932

theorem ellipse_foci_condition (α : Real) (h1 : 0 < α) (h2 : α < π / 2) :
  (∀ x y : Real, x^2 / Real.sin α + y^2 / Real.cos α = 1 →
    ∃ c : Real, c > 0 ∧ 
      ∀ x₀ y₀ : Real, (x₀ + c)^2 + y₀^2 + (x₀ - c)^2 + y₀^2 = 
        2 * ((x^2 / Real.sin α + y^2 / Real.cos α) * (1 / Real.sin α + 1 / Real.cos α))) →
  π / 4 < α ∧ α < π / 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_condition_l2679_267932


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2679_267962

def polynomial (x : ℤ) : ℤ := x^4 + 4*x^3 - x^2 + 3*x - 18

def possible_roots : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial :
  {x : ℤ | polynomial x = 0} = possible_roots := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2679_267962


namespace NUMINAMATH_CALUDE_range_of_g_l2679_267915

noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_g :
  Set.range g = {π/12, -π/12} := by sorry

end NUMINAMATH_CALUDE_range_of_g_l2679_267915


namespace NUMINAMATH_CALUDE_f_range_l2679_267949

-- Define the function
def f (x : ℝ) : ℝ := 3 * x - x^3

-- State the theorem
theorem f_range :
  ∀ y ∈ Set.range f,
  ∃ x ∈ Set.Icc (0 : ℝ) 3,
  f x = y ∧ -18 ≤ y ∧ y ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_f_range_l2679_267949


namespace NUMINAMATH_CALUDE_retailer_profit_percent_l2679_267975

/-- Calculates the profit percent given purchase price, overhead expenses, and selling price -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percent for the given values is 45.83% -/
theorem retailer_profit_percent :
  profit_percent 225 15 350 = 45.83 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percent_l2679_267975


namespace NUMINAMATH_CALUDE_all_representable_l2679_267979

def powers_of_three : List ℕ := [1, 3, 9, 27, 81, 243, 729]

def can_represent (n : ℕ) (powers : List ℕ) : Prop :=
  ∃ (subset : List ℕ) (signs : List Bool),
    subset ⊆ powers ∧
    signs.length = subset.length ∧
    (List.zip subset signs).foldl
      (λ acc (p, sign) => if sign then acc + p else acc - p) 0 = n

theorem all_representable :
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 1093 → can_represent n powers_of_three :=
sorry

end NUMINAMATH_CALUDE_all_representable_l2679_267979


namespace NUMINAMATH_CALUDE_circles_tangent_internally_l2679_267909

def circle_O₁_center : ℝ × ℝ := (2, 0)
def circle_O₁_radius : ℝ := 1
def circle_O₂_center : ℝ × ℝ := (-1, 0)
def circle_O₂_radius : ℝ := 3

theorem circles_tangent_internally :
  let d := Real.sqrt ((circle_O₂_center.1 - circle_O₁_center.1)^2 + (circle_O₂_center.2 - circle_O₁_center.2)^2)
  d = circle_O₂_radius ∧ d > circle_O₁_radius :=
by sorry

end NUMINAMATH_CALUDE_circles_tangent_internally_l2679_267909


namespace NUMINAMATH_CALUDE_parametric_to_slope_intercept_l2679_267914

/-- A line parameterized by (x, y) = (3t + 6, 5t - 7) where t is a real number -/
def parametric_line (t : ℝ) : ℝ × ℝ := (3 * t + 6, 5 * t - 7)

/-- The slope-intercept form of a line -/
def slope_intercept_form (m b : ℝ) (x : ℝ) : ℝ := m * x + b

theorem parametric_to_slope_intercept :
  ∀ (x y : ℝ), (∃ t : ℝ, parametric_line t = (x, y)) →
  y = slope_intercept_form (5/3) (-17) x :=
by sorry

end NUMINAMATH_CALUDE_parametric_to_slope_intercept_l2679_267914


namespace NUMINAMATH_CALUDE_negative_one_exponent_division_l2679_267902

theorem negative_one_exponent_division : ((-1 : ℤ) ^ 2003) / ((-1 : ℤ) ^ 2004) = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_exponent_division_l2679_267902


namespace NUMINAMATH_CALUDE_base9_perfect_square_multiple_of_3_l2679_267944

/-- Represents a number in base 9 of the form ab4c -/
structure Base9Number where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a ≠ 0

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.b + 36 + n.c

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem base9_perfect_square_multiple_of_3 (n : Base9Number) 
  (h1 : isPerfectSquare (toDecimal n))
  (h2 : (toDecimal n) % 3 = 0) :
  n.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_base9_perfect_square_multiple_of_3_l2679_267944


namespace NUMINAMATH_CALUDE_talking_birds_l2679_267948

theorem talking_birds (total : ℕ) (non_talking : ℕ) (talking : ℕ) : 
  total = 77 → non_talking = 13 → talking = total - non_talking → talking = 64 := by
sorry

end NUMINAMATH_CALUDE_talking_birds_l2679_267948


namespace NUMINAMATH_CALUDE_fourth_term_is_seven_l2679_267992

/-- An arithmetic sequence with sum of first 7 terms equal to 49 -/
structure ArithmeticSequence where
  /-- The nth term of the sequence -/
  a : ℕ → ℝ
  /-- The sum of the first n terms -/
  S : ℕ → ℝ
  /-- Property: S_7 = 49 -/
  sum_7 : S 7 = 49
  /-- Property: S_n is the sum of first n terms -/
  sum_property : ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

/-- The 4th term of the arithmetic sequence is 7 -/
theorem fourth_term_is_seven (seq : ArithmeticSequence) : seq.a 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_seven_l2679_267992


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2679_267993

theorem smallest_prime_divisor_of_sum (n : ℕ) :
  2 = Nat.minFac (4^13 + 6^15) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2679_267993


namespace NUMINAMATH_CALUDE_probability_different_tens_digits_value_l2679_267982

def range_start : ℕ := 10
def range_end : ℕ := 59
def num_chosen : ℕ := 5

def probability_different_tens_digits : ℚ :=
  (10 ^ num_chosen : ℚ) / (Nat.choose (range_end - range_start + 1) num_chosen)

theorem probability_different_tens_digits_value :
  probability_different_tens_digits = 2500 / 52969 := by sorry

end NUMINAMATH_CALUDE_probability_different_tens_digits_value_l2679_267982
