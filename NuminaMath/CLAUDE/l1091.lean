import Mathlib

namespace NUMINAMATH_CALUDE_star_equality_implies_x_eq_five_l1091_109194

/-- Binary operation ★ on ordered pairs of integers -/
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

theorem star_equality_implies_x_eq_five :
  ∀ y : ℤ, star 4 5 1 1 = star x y 2 3 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_star_equality_implies_x_eq_five_l1091_109194


namespace NUMINAMATH_CALUDE_above_x_axis_on_line_l1091_109107

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)

-- Statement for the first part of the problem
theorem above_x_axis (m : ℝ) : 
  Complex.im (z m) > 0 ↔ m < -3 ∨ m > 5 := by sorry

-- Statement for the second part of the problem
theorem on_line (m : ℝ) :
  Complex.re (z m) + Complex.im (z m) + 5 = 0 ↔ 
  m = (-3 + Real.sqrt 41) / 4 ∨ m = (-3 - Real.sqrt 41) / 4 := by sorry

end NUMINAMATH_CALUDE_above_x_axis_on_line_l1091_109107


namespace NUMINAMATH_CALUDE_triangle_side_length_l1091_109146

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- Define the theorem
theorem triangle_side_length (t : Triangle) 
  (ha : t.a = 4) 
  (hb : t.b = 5) 
  (hS : t.S = 5 * Real.sqrt 3) :
  t.c = Real.sqrt 21 ∨ t.c = Real.sqrt 61 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l1091_109146


namespace NUMINAMATH_CALUDE_max_x_squared_y_l1091_109127

theorem max_x_squared_y (x y : ℕ+) (h : 7 * x.val + 4 * y.val = 140) :
  ∀ (a b : ℕ+), 7 * a.val + 4 * b.val = 140 → x.val^2 * y.val ≥ a.val^2 * b.val :=
by sorry

end NUMINAMATH_CALUDE_max_x_squared_y_l1091_109127


namespace NUMINAMATH_CALUDE_dividend_calculation_l1091_109183

theorem dividend_calculation (quotient divisor remainder : ℕ) 
  (h_quotient : quotient = 36)
  (h_divisor : divisor = 85)
  (h_remainder : remainder = 26) :
  (divisor * quotient) + remainder = 3086 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1091_109183


namespace NUMINAMATH_CALUDE_complex_abs_value_l1091_109135

theorem complex_abs_value : Complex.abs (-3 + (8/5) * Complex.I) = 17/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_value_l1091_109135


namespace NUMINAMATH_CALUDE_line_generates_surface_l1091_109169

-- Define the parabolas and the plane
def parabola1 (x y z : ℝ) : Prop := y^2 = 2*x ∧ z = 0
def parabola2 (x y z : ℝ) : Prop := 3*x = z^2 ∧ y = 0
def plane (y z : ℝ) : Prop := y = z

-- Define a line parallel to the plane y = z
def parallel_line (L : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ (p q : ℝ × ℝ × ℝ), p ∈ L → q ∈ L → plane p.2.1 p.2.2 = plane q.2.1 q.2.2

-- Define the intersection of the line with the parabolas
def intersects_parabolas (L : Set (ℝ × ℝ × ℝ)) : Prop :=
  (∃ p ∈ L, parabola1 p.1 p.2.1 p.2.2) ∧ (∃ q ∈ L, parabola2 q.1 q.2.1 q.2.2)

-- The main theorem
theorem line_generates_surface (L : Set (ℝ × ℝ × ℝ)) :
  parallel_line L → intersects_parabolas L →
  ∀ (x y z : ℝ), (x, y, z) ∈ L → x = (y - z) * (y/2 - z/3) :=
sorry

end NUMINAMATH_CALUDE_line_generates_surface_l1091_109169


namespace NUMINAMATH_CALUDE_buddy_fraction_l1091_109191

theorem buddy_fraction (t s : ℚ) 
  (h1 : t > 0) 
  (h2 : s > 0) 
  (h3 : (1/4) * t = (3/5) * s) : 
  ((1/4) * t + (3/5) * s) / (t + s) = 6/17 := by
sorry

end NUMINAMATH_CALUDE_buddy_fraction_l1091_109191


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1091_109130

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1091_109130


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_m_range_l1091_109159

theorem quadratic_distinct_roots_m_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   m * x^2 + (2*m + 1) * x + m = 0 ∧ 
   m * y^2 + (2*m + 1) * y + m = 0) ↔ 
  (m > -1/4 ∧ m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_m_range_l1091_109159


namespace NUMINAMATH_CALUDE_det_A_l1091_109186

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 0, -2; 5, 6, -4; 3, 3, 7]

theorem det_A : Matrix.det A = 168 := by sorry

end NUMINAMATH_CALUDE_det_A_l1091_109186


namespace NUMINAMATH_CALUDE_farm_sections_l1091_109197

theorem farm_sections (section_area : ℝ) (total_area : ℝ) (h1 : section_area = 60) (h2 : total_area = 300) :
  total_area / section_area = 5 := by
  sorry

end NUMINAMATH_CALUDE_farm_sections_l1091_109197


namespace NUMINAMATH_CALUDE_smallest_value_l1091_109116

theorem smallest_value (x : ℝ) (h : 0 < x ∧ x < 1) :
  x^3 < x ∧ x^3 < 3*x ∧ x^3 < x^(1/3) ∧ x^3 < 1/x^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_l1091_109116


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1091_109126

theorem trigonometric_equation_solution (k : ℤ) :
  (∃ x : ℝ, 2 + Real.cos x ^ 2 + Real.cos (4 * x) + Real.cos (2 * x) + 
   2 * Real.sin (3 * x) * Real.sin (7 * x) + Real.sin (7 * x) ^ 2 = 
   Real.cos (π * k / 2022) ^ 2) ↔ 
  (∃ m : ℤ, k = 2022 * m) ∧
  (∀ x : ℝ, 2 + Real.cos x ^ 2 + Real.cos (4 * x) + Real.cos (2 * x) + 
   2 * Real.sin (3 * x) * Real.sin (7 * x) + Real.sin (7 * x) ^ 2 = 
   Real.cos (π * k / 2022) ^ 2 →
   ∃ n : ℤ, x = π / 4 + π * n / 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1091_109126


namespace NUMINAMATH_CALUDE_total_dresses_l1091_109125

theorem total_dresses (ana_dresses : ℕ) (lisa_more_dresses : ℕ) : 
  ana_dresses = 15 → lisa_more_dresses = 18 → 
  ana_dresses + (ana_dresses + lisa_more_dresses) = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_dresses_l1091_109125


namespace NUMINAMATH_CALUDE_max_vacation_savings_l1091_109177

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Calculates the remaining money after buying a ticket -/
def remaining_money (savings : ℕ) (ticket_cost : ℕ) : ℕ :=
  base8_to_base10 savings - ticket_cost

theorem max_vacation_savings :
  remaining_money 5273 1500 = 1247 := by sorry

end NUMINAMATH_CALUDE_max_vacation_savings_l1091_109177


namespace NUMINAMATH_CALUDE_poly_descending_order_l1091_109143

/-- The original polynomial -/
def original_poly (x y : ℝ) : ℝ := 2 * x^2 * y - 3 * x^3 - x * y^3 + 1

/-- The polynomial arranged in descending order of x -/
def descending_poly (x y : ℝ) : ℝ := -3 * x^3 + 2 * x^2 * y - x * y^3 + 1

/-- Theorem stating that the original polynomial is equal to the descending order polynomial -/
theorem poly_descending_order : ∀ x y : ℝ, original_poly x y = descending_poly x y := by
  sorry

end NUMINAMATH_CALUDE_poly_descending_order_l1091_109143


namespace NUMINAMATH_CALUDE_three_dollar_neg_one_l1091_109117

def dollar_op (a b : ℤ) : ℤ := a * (b + 2) + a * b

theorem three_dollar_neg_one : dollar_op 3 (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_dollar_neg_one_l1091_109117


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1091_109132

theorem inequality_system_solution (x : ℝ) : 
  3 * x > x + 6 ∧ (1/2) * x < -x + 5 → 3 < x ∧ x < 10/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1091_109132


namespace NUMINAMATH_CALUDE_relationship_abc_l1091_109174

theorem relationship_abc (a b c : ℝ) : 
  a = Real.sin (145 * π / 180) →
  b = Real.cos (52 * π / 180) →
  c = Real.tan (47 * π / 180) →
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1091_109174


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1091_109133

theorem coin_flip_probability : ∃ p : ℝ, 
  p > 0 ∧ p < 1 ∧ 
  p^2 + (1-p)^2 = 4*p*(1-p) ∧
  ∀ q : ℝ, (q > 0 ∧ q < 1 ∧ q^2 + (1-q)^2 = 4*q*(1-q)) → q ≤ p ∧
  p = (3 + Real.sqrt 3) / 6 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1091_109133


namespace NUMINAMATH_CALUDE_greatest_x_value_l1091_109103

theorem greatest_x_value (x : ℝ) : 
  (x^2 - x - 30) / (x - 6) = 2 / (x + 4) → x ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l1091_109103


namespace NUMINAMATH_CALUDE_sum_inverse_g_eq_758_3125_l1091_109178

/-- The function g(n) that returns the integer closest to the cube root of n -/
def g (n : ℕ) : ℕ := sorry

/-- The sum of 1/g(k) from k=1 to 4096 -/
def sum_inverse_g : ℚ :=
  (Finset.range 4096).sum (fun k => 1 / (g (k + 1) : ℚ))

/-- Theorem stating that the sum equals 758.3125 -/
theorem sum_inverse_g_eq_758_3125 : sum_inverse_g = 758.3125 := by sorry

end NUMINAMATH_CALUDE_sum_inverse_g_eq_758_3125_l1091_109178


namespace NUMINAMATH_CALUDE_second_knife_set_price_l1091_109165

/-- Calculates the price of the second set of knives based on given sales data --/
def price_of_second_knife_set (
  houses_per_day : ℕ)
  (buy_percentage : ℚ)
  (first_set_price : ℕ)
  (weekly_sales : ℕ)
  (work_days : ℕ) : ℚ :=
  let buyers_per_day : ℚ := houses_per_day * buy_percentage
  let first_set_buyers_per_day : ℚ := buyers_per_day / 2
  let first_set_sales_per_day : ℚ := first_set_buyers_per_day * first_set_price
  let first_set_sales_per_week : ℚ := first_set_sales_per_day * work_days
  let second_set_sales_per_week : ℚ := weekly_sales - first_set_sales_per_week
  let second_set_buyers_per_week : ℚ := first_set_buyers_per_day * work_days
  second_set_sales_per_week / second_set_buyers_per_week

/-- Theorem stating that the price of the second set of knives is $150 --/
theorem second_knife_set_price :
  price_of_second_knife_set 50 (1/5) 50 5000 5 = 150 := by
  sorry

end NUMINAMATH_CALUDE_second_knife_set_price_l1091_109165


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l1091_109138

-- Define the number of red and yellow balls
def num_red_balls : ℕ := 3
def num_yellow_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_yellow_balls

-- Define the probability of selecting a yellow ball
def prob_yellow : ℚ := num_yellow_balls / total_balls

-- Theorem statement
theorem yellow_ball_probability : prob_yellow = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_probability_l1091_109138


namespace NUMINAMATH_CALUDE_problem_statement_l1091_109105

theorem problem_statement :
  (∀ a b c d : ℝ, a^6 + b^6 + c^6 + d^6 - 6*a*b*c*d ≥ -2) ∧
  (∀ k : ℕ, k % 2 = 1 → k ≥ 5 →
    ∃ M_k : ℝ, ∀ a b c d : ℝ, a^k + b^k + c^k + d^k - k*a*b*c*d ≥ M_k) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1091_109105


namespace NUMINAMATH_CALUDE_bob_cleaning_time_l1091_109111

theorem bob_cleaning_time (alice_time : ℕ) (bob_fraction : ℚ) : 
  alice_time = 40 → bob_fraction = 3 / 4 → bob_fraction * alice_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_bob_cleaning_time_l1091_109111


namespace NUMINAMATH_CALUDE_monthly_fixed_costs_correct_l1091_109193

/-- Represents the monthly fixed costs for producing electronic components -/
def monthly_fixed_costs : ℝ := 16399.50

/-- Represents the cost to produce one electronic component -/
def production_cost : ℝ := 80

/-- Represents the shipping cost for one electronic component -/
def shipping_cost : ℝ := 4

/-- Represents the number of components produced and sold monthly -/
def monthly_sales : ℕ := 150

/-- Represents the lowest selling price per component without loss -/
def break_even_price : ℝ := 193.33

/-- Theorem stating that the monthly fixed costs are correct given the other parameters -/
theorem monthly_fixed_costs_correct :
  monthly_fixed_costs = 
    monthly_sales * break_even_price - 
    monthly_sales * (production_cost + shipping_cost) :=
by sorry

end NUMINAMATH_CALUDE_monthly_fixed_costs_correct_l1091_109193


namespace NUMINAMATH_CALUDE_subtraction_addition_result_l1091_109192

/-- The result of subtracting 567.89 from 1234.56 and then adding 300.30 is equal to 966.97 -/
theorem subtraction_addition_result : 
  (1234.56 - 567.89 + 300.30 : ℚ) = 966.97 := by sorry

end NUMINAMATH_CALUDE_subtraction_addition_result_l1091_109192


namespace NUMINAMATH_CALUDE_congruence_solution_l1091_109163

theorem congruence_solution (n : ℤ) : 13 * n ≡ 9 [ZMOD 53] → n ≡ 17 [ZMOD 53] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1091_109163


namespace NUMINAMATH_CALUDE_existence_of_primes_with_gcd_one_l1091_109166

theorem existence_of_primes_with_gcd_one (n : ℕ) (h1 : n > 6) (h2 : Even n) :
  ∃ (p q : ℕ), Prime p ∧ Prime q ∧ Nat.gcd (n - p) (n - q) = 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_primes_with_gcd_one_l1091_109166


namespace NUMINAMATH_CALUDE_subtracted_value_l1091_109195

theorem subtracted_value (n v : ℝ) (h1 : n = -10) (h2 : 2 * n - v = -12) : v = -8 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l1091_109195


namespace NUMINAMATH_CALUDE_credit_card_more_beneficial_l1091_109114

/-- Represents the purchase amount in rubles -/
def purchase_amount : ℝ := 8000

/-- Represents the credit card cashback rate -/
def credit_cashback_rate : ℝ := 0.005

/-- Represents the debit card cashback rate -/
def debit_cashback_rate : ℝ := 0.0075

/-- Represents the monthly interest rate on the debit card balance -/
def debit_interest_rate : ℝ := 0.005

/-- Calculates the total income from using the credit card -/
def credit_card_income (amount : ℝ) : ℝ :=
  amount * credit_cashback_rate + amount * debit_interest_rate

/-- Calculates the total income from using the debit card -/
def debit_card_income (amount : ℝ) : ℝ :=
  amount * debit_cashback_rate

/-- Theorem stating that the credit card is more beneficial -/
theorem credit_card_more_beneficial :
  credit_card_income purchase_amount > debit_card_income purchase_amount :=
by sorry


end NUMINAMATH_CALUDE_credit_card_more_beneficial_l1091_109114


namespace NUMINAMATH_CALUDE_triangle_side_length_l1091_109164

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  b = Real.sqrt 3 →
  c = 3 →
  B = 30 * π / 180 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos B →
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1091_109164


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l1091_109108

/-- Given an original salary and a final salary after an increase followed by a decrease,
    calculate the initial percentage increase. -/
theorem salary_increase_percentage
  (original_salary : ℝ)
  (final_salary : ℝ)
  (decrease_percentage : ℝ)
  (h1 : original_salary = 6000)
  (h2 : final_salary = 6270)
  (h3 : decrease_percentage = 5)
  : ∃ x : ℝ,
    final_salary = original_salary * (1 + x / 100) * (1 - decrease_percentage / 100) ∧
    x = 10 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l1091_109108


namespace NUMINAMATH_CALUDE_f_properties_l1091_109120

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| - |x - a|

-- State the theorem
theorem f_properties (a : ℝ) (h : a ≤ 0) :
  -- Part 1: Solution set when a = 0
  (a = 0 → {x : ℝ | f 0 x < 1} = {x : ℝ | 0 < x ∧ x < 2}) ∧
  -- Part 2: Range of a when triangle area > 3/2
  (∃ (x y : ℝ), x < y ∧ 
    (1/2 * (y - x) * (max (f a x) (f a y))) > 3/2 → a < -1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1091_109120


namespace NUMINAMATH_CALUDE_largest_common_term_correct_l1091_109113

/-- First arithmetic progression with initial term 4 and common difference 5 -/
def ap1 (n : ℕ) : ℕ := 4 + 5 * n

/-- Second arithmetic progression with initial term 5 and common difference 10 -/
def ap2 (n : ℕ) : ℕ := 5 + 10 * n

/-- Predicate to check if a number is in both arithmetic progressions -/
def isCommonTerm (x : ℕ) : Prop :=
  ∃ n m : ℕ, ap1 n = x ∧ ap2 m = x

/-- The largest common term less than 300 -/
def largestCommonTerm : ℕ := 299

theorem largest_common_term_correct :
  isCommonTerm largestCommonTerm ∧
  largestCommonTerm < 300 ∧
  ∀ x : ℕ, isCommonTerm x → x < 300 → x ≤ largestCommonTerm :=
by sorry

#check largest_common_term_correct

end NUMINAMATH_CALUDE_largest_common_term_correct_l1091_109113


namespace NUMINAMATH_CALUDE_largest_group_size_l1091_109156

def round_fraction (n : ℕ) (d : ℕ) (x : ℕ) : ℕ :=
  (2 * n * x + d) / (2 * d)

theorem largest_group_size :
  ∀ x : ℕ, x ≤ 37 ↔
    round_fraction 1 2 x + round_fraction 1 3 x + round_fraction 1 5 x ≤ x + 1 ∧
    (∀ y : ℕ, y > x →
      round_fraction 1 2 y + round_fraction 1 3 y + round_fraction 1 5 y > y + 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_group_size_l1091_109156


namespace NUMINAMATH_CALUDE_triangle_theorem_l1091_109198

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  angle_sum : A + B + C = π
  tan_condition : 2 * (Real.tan A + Real.tan B) = Real.tan A / Real.cos B + Real.tan B / Real.cos A

/-- Main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) : 
  t.a + t.b = 2 * t.c ∧ Real.cos t.C ≥ 1/2 ∧ ∃ (t' : Triangle), Real.cos t'.C = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1091_109198


namespace NUMINAMATH_CALUDE_sphere_pyramid_height_l1091_109140

/-- The height of a square pyramid of spheres -/
def pyramid_height (n : ℕ) : ℝ :=
  2 * (n - 1)

/-- Theorem: The height of a square pyramid of spheres with radius 1,
    where the base layer has n^2 spheres and each subsequent layer has
    (n-1)^2 spheres until the top layer with 1 sphere, is 2(n-1). -/
theorem sphere_pyramid_height (n : ℕ) (h : n > 0) :
  let base_layer := n^2
  let top_layer := 1
  let sphere_radius := 1
  pyramid_height n = 2 * (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sphere_pyramid_height_l1091_109140


namespace NUMINAMATH_CALUDE_min_value_tangent_line_circle_l1091_109153

theorem min_value_tangent_line_circle (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, x + y + a = 0 → (x - b)^2 + (y - 1)^2 ≥ 2) → 
  (∃ x y : ℝ, x + y + a = 0 ∧ (x - b)^2 + (y - 1)^2 = 2) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∀ x y : ℝ, x + y + a' = 0 → (x - b')^2 + (y - 1)^2 ≥ 2) → 
    (∃ x y : ℝ, x + y + a' = 0 ∧ (x - b')^2 + (y - 1)^2 = 2) → 
    (3 - 2*b)^2 / (2*a) ≤ (3 - 2*b')^2 / (2*a')) → 
  (3 - 2*b)^2 / (2*a) = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_tangent_line_circle_l1091_109153


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l1091_109102

/-- Given two distinct real numbers k and b, prove that the x-coordinate of the 
    intersection point of the lines y = kx + b and y = bx + k is 1. -/
theorem intersection_x_coordinate (k b : ℝ) (h : k ≠ b) : 
  ∃ x : ℝ, x = 1 ∧ kx + b = bx + k := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l1091_109102


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1091_109161

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x * (x + 1) > 0}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1091_109161


namespace NUMINAMATH_CALUDE_dave_baseball_cards_l1091_109180

/-- Calculates the number of pages required to organize baseball cards in a binder -/
def pages_required (cards_per_page new_cards old_cards : ℕ) : ℕ :=
  (new_cards + old_cards + cards_per_page - 1) / cards_per_page

/-- Proves that Dave will use 2 pages to organize his baseball cards -/
theorem dave_baseball_cards : pages_required 8 3 13 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dave_baseball_cards_l1091_109180


namespace NUMINAMATH_CALUDE_expression_simplification_l1091_109129

theorem expression_simplification :
  Real.sqrt 3 + Real.sqrt (3 + 5) + Real.sqrt (3 + 5 + 7) + Real.sqrt (3 + 5 + 7 + 9) =
  Real.sqrt 3 + 2 * Real.sqrt 2 + Real.sqrt 15 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1091_109129


namespace NUMINAMATH_CALUDE_angle_between_specific_vectors_l1091_109196

/-- The angle between two 2D vectors -/
def angle_between (a b : ℝ × ℝ) : ℝ := sorry

/-- Converts degrees to radians -/
def deg_to_rad (deg : ℝ) : ℝ := sorry

theorem angle_between_specific_vectors :
  let a : ℝ × ℝ := (1, 0)
  let b : ℝ × ℝ := (-1/2, Real.sqrt 3/2)
  angle_between a b = deg_to_rad 120
  := by sorry

end NUMINAMATH_CALUDE_angle_between_specific_vectors_l1091_109196


namespace NUMINAMATH_CALUDE_inequality_theorem_l1091_109199

theorem inequality_theorem (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_eq : a * b / (c * d) = (a + b) / (c + d)) :
  (a + b) * (c + d) ≥ (a + c) * (b + d) := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1091_109199


namespace NUMINAMATH_CALUDE_fruit_bowl_problem_l1091_109112

/-- Proves that given a bowl with 14 apples and an unknown number of oranges,
    if removing 15 oranges results in apples being 70% of the remaining fruit,
    then the initial number of oranges was 21. -/
theorem fruit_bowl_problem (initial_oranges : ℕ) : 
  (14 : ℝ) / ((14 : ℝ) + (initial_oranges - 15)) = 0.7 → initial_oranges = 21 :=
by sorry

end NUMINAMATH_CALUDE_fruit_bowl_problem_l1091_109112


namespace NUMINAMATH_CALUDE_binary_product_theorem_l1091_109123

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- Converts a decimal number to its binary representation -/
def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : Nat) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- The main theorem stating that the product of the given binary numbers equals the expected result -/
theorem binary_product_theorem :
  let a := [false, false, true, true, false, true]  -- 101100₂
  let b := [true, true, true]                       -- 111₂
  let c := [false, true]                            -- 10₂
  let result := [false, false, true, false, true, true, false, false, true]  -- 100110100₂
  binary_to_decimal a * binary_to_decimal b * binary_to_decimal c = binary_to_decimal result := by
  sorry


end NUMINAMATH_CALUDE_binary_product_theorem_l1091_109123


namespace NUMINAMATH_CALUDE_incorrect_conversion_l1091_109134

def base4_to_decimal (n : ℕ) : ℕ :=
  (n / 10) * 4 + (n % 10)

def base2_to_decimal (n : ℕ) : ℕ :=
  (n / 10) * 2 + (n % 10)

theorem incorrect_conversion :
  base4_to_decimal 31 ≠ base2_to_decimal 62 :=
sorry

end NUMINAMATH_CALUDE_incorrect_conversion_l1091_109134


namespace NUMINAMATH_CALUDE_perfect_square_powers_of_two_l1091_109147

theorem perfect_square_powers_of_two :
  (∃! (n : ℕ), ∃ (k : ℕ), 2^n + 3 = k^2) ∧
  (∃! (n : ℕ), ∃ (k : ℕ), 2^n + 1 = k^2) :=
by
  constructor
  · -- Proof for 2^n + 3
    sorry
  · -- Proof for 2^n + 1
    sorry

#check perfect_square_powers_of_two

end NUMINAMATH_CALUDE_perfect_square_powers_of_two_l1091_109147


namespace NUMINAMATH_CALUDE_remainder_theorem_l1091_109136

theorem remainder_theorem (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1091_109136


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1091_109172

/-- Given a geometric sequence {aₙ} where a₁ = 1/3 and 2a₂ = a₄, prove that a₅ = 4/3 -/
theorem geometric_sequence_fifth_term (a : ℕ → ℚ) (h1 : a 1 = 1/3) (h2 : 2 * a 2 = a 4) :
  a 5 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1091_109172


namespace NUMINAMATH_CALUDE_sum_x_y_equals_one_l1091_109173

theorem sum_x_y_equals_one (x y : ℝ) 
  (eq1 : x + 2*y = 1) 
  (eq2 : 2*x + y = 2) : 
  x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_one_l1091_109173


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l1091_109104

theorem right_triangle_leg_square (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = a + 2 →        -- Given condition
  b^2 = 4*(a + 1) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l1091_109104


namespace NUMINAMATH_CALUDE_sum_of_odd_numbers_l1091_109152

theorem sum_of_odd_numbers (N : ℕ) : 
  1001 + 1003 + 1005 + 1007 + 1009 = 5050 - N → N = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_numbers_l1091_109152


namespace NUMINAMATH_CALUDE_k_negative_sufficient_not_necessary_l1091_109109

-- Define the condition for a hyperbola
def is_hyperbola (k : ℝ) : Prop := k * (1 - k) < 0

-- State the theorem
theorem k_negative_sufficient_not_necessary :
  (∀ k : ℝ, k < 0 → is_hyperbola k) ∧
  (∃ k : ℝ, is_hyperbola k ∧ ¬(k < 0)) :=
sorry

end NUMINAMATH_CALUDE_k_negative_sufficient_not_necessary_l1091_109109


namespace NUMINAMATH_CALUDE_max_remainder_l1091_109110

theorem max_remainder (m : ℕ) (n : ℕ) : 
  0 < m → m < 2015 → 2015 % m = n → n ≤ 1007 := by
  sorry

end NUMINAMATH_CALUDE_max_remainder_l1091_109110


namespace NUMINAMATH_CALUDE_children_getting_on_bus_l1091_109124

theorem children_getting_on_bus (initial : ℝ) (got_off : ℝ) (final : ℝ) 
  (h1 : initial = 42.5)
  (h2 : got_off = 21.3)
  (h3 : final = 35.8) :
  final - (initial - got_off) = 14.6 := by
  sorry

end NUMINAMATH_CALUDE_children_getting_on_bus_l1091_109124


namespace NUMINAMATH_CALUDE_percentage_of_m_l1091_109155

theorem percentage_of_m (j k l m : ℝ) : 
  (1.25 * j = 0.25 * k) →
  (1.5 * k = 0.5 * l) →
  (∃ p, 1.75 * l = p / 100 * m) →
  (0.2 * m = 7 * j) →
  (∃ p, 1.75 * l = p / 100 * m ∧ p = 75) := by
sorry

end NUMINAMATH_CALUDE_percentage_of_m_l1091_109155


namespace NUMINAMATH_CALUDE_shape_cutting_theorem_l1091_109137

/-- Represents a cell in the shape --/
inductive Cell
| Black
| Gray

/-- Represents the shape as a list of cells --/
def Shape := List Cell

/-- A function to count the number of ways to cut the shape --/
def count_cuts (shape : Shape) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem shape_cutting_theorem (shape : Shape) :
  shape.length = 17 →
  count_cuts shape = 10 :=
sorry

end NUMINAMATH_CALUDE_shape_cutting_theorem_l1091_109137


namespace NUMINAMATH_CALUDE_carnival_booth_rent_calculation_l1091_109189

def carnival_booth_rent (daily_popcorn_revenue : ℕ)
                        (cotton_candy_multiplier : ℕ)
                        (activity_days : ℕ)
                        (ingredient_cost : ℕ)
                        (total_earnings_after_expenses : ℕ) : Prop :=
  let daily_cotton_candy_revenue := daily_popcorn_revenue * cotton_candy_multiplier
  let total_revenue := (daily_popcorn_revenue + daily_cotton_candy_revenue) * activity_days
  let rent := total_revenue - ingredient_cost - total_earnings_after_expenses
  rent = 30

theorem carnival_booth_rent_calculation :
  carnival_booth_rent 50 3 5 75 895 := by
  sorry

end NUMINAMATH_CALUDE_carnival_booth_rent_calculation_l1091_109189


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1091_109144

/-- Given a hyperbola with asymptotes y = ±(3/4)x, its eccentricity is either 5/4 or 5/3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h : b / a = 3 / 4 ∨ a / b = 3 / 4) :
  let e := c / a
  c^2 = a^2 + b^2 →
  e = 5 / 4 ∨ e = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1091_109144


namespace NUMINAMATH_CALUDE_diamond_brace_to_ring_ratio_l1091_109160

def total_worth : ℕ := 14000
def ring_cost : ℕ := 4000
def car_cost : ℕ := 2000

def diamond_brace_cost : ℕ := total_worth - (ring_cost + car_cost)

theorem diamond_brace_to_ring_ratio :
  diamond_brace_cost / ring_cost = 2 := by sorry

end NUMINAMATH_CALUDE_diamond_brace_to_ring_ratio_l1091_109160


namespace NUMINAMATH_CALUDE_lcm_factor_theorem_l1091_109141

theorem lcm_factor_theorem (A B : ℕ) (hcf lcm X : ℕ) : 
  A > 0 → B > 0 → 
  A = 368 → 
  hcf = Nat.gcd A B → 
  hcf = 23 → 
  lcm = Nat.lcm A B → 
  lcm = hcf * X * 16 → 
  X = 1 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_theorem_l1091_109141


namespace NUMINAMATH_CALUDE_henry_kombucha_consumption_l1091_109101

/-- The number of bottles of kombucha Henry drinks per month -/
def bottles_per_month : ℕ := 15

/-- The cost of each bottle in dollars -/
def bottle_cost : ℚ := 3

/-- The cash refund for each bottle in dollars -/
def bottle_refund : ℚ := 1/10

/-- The number of bottles Henry can buy with his yearly refund -/
def bottles_from_refund : ℕ := 6

/-- The number of months in a year -/
def months_in_year : ℕ := 12

theorem henry_kombucha_consumption :
  bottles_per_month * bottle_refund * months_in_year = bottles_from_refund * bottle_cost :=
sorry

end NUMINAMATH_CALUDE_henry_kombucha_consumption_l1091_109101


namespace NUMINAMATH_CALUDE_student_marks_average_l1091_109131

theorem student_marks_average (P C M B : ℝ) 
  (h1 : P + C + M + B = P + B + 180)
  (h2 : P = 1.20 * B) :
  (C + M) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_average_l1091_109131


namespace NUMINAMATH_CALUDE_complement_of_union_l1091_109154

universe u

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def M : Set ℕ := {1,3,5,7}
def N : Set ℕ := {5,6,7}

theorem complement_of_union :
  (U \ (M ∪ N)) = {2,4,8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l1091_109154


namespace NUMINAMATH_CALUDE_smallest_n_containing_all_binary_l1091_109188

/-- Given a natural number n, returns true if the binary representation of 1/n
    contains the binary representations of all numbers from 1 to 1990 as
    contiguous substrings after the decimal point. -/
def containsAllBinaryRepresentations (n : ℕ) : Prop := sorry

/-- Theorem stating that 2053 is the smallest natural number satisfying
    the condition of containing all binary representations from 1 to 1990. -/
theorem smallest_n_containing_all_binary : ∀ n : ℕ,
  n < 2053 → ¬(containsAllBinaryRepresentations n) ∧ containsAllBinaryRepresentations 2053 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_containing_all_binary_l1091_109188


namespace NUMINAMATH_CALUDE_trig_identity_l1091_109167

theorem trig_identity : 
  Real.sin (160 * π / 180) * Real.sin (10 * π / 180) - 
  Real.cos (20 * π / 180) * Real.cos (10 * π / 180) = 
  - (Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_trig_identity_l1091_109167


namespace NUMINAMATH_CALUDE_females_without_daughters_l1091_109157

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  total_females : ℕ
  daughters_with_children : ℕ

/-- The actual family structure of Bertha -/
def berthas_family : BerthaFamily :=
  { daughters := 8
  , granddaughters := 40
  , total_females := 48
  , daughters_with_children := 5 }

/-- Theorem stating the number of females without daughters in Bertha's family -/
theorem females_without_daughters (b : BerthaFamily) (h1 : b = berthas_family) :
  b.daughters + b.granddaughters - b.daughters_with_children = 43 := by
  sorry

#check females_without_daughters

end NUMINAMATH_CALUDE_females_without_daughters_l1091_109157


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1091_109128

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - 3 * y + 5 = 0

-- Define the point that the parallel line passes through
def point : ℝ × ℝ := (-2, 1)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2 * x - 3 * y + 7 = 0

-- Theorem statement
theorem parallel_line_through_point :
  (∀ x y : ℝ, given_line x y ↔ 2 * x - 3 * y + 5 = 0) →
  parallel_line point.1 point.2 ∧
  (∀ x y : ℝ, parallel_line x y ↔ 2 * x - 3 * y + 7 = 0) ∧
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, given_line x y ↔ parallel_line x y) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1091_109128


namespace NUMINAMATH_CALUDE_cube_difference_as_sum_of_squares_l1091_109171

theorem cube_difference_as_sum_of_squares (n : ℤ) :
  (n + 2)^3 - n^3 = n^2 + (n + 2)^2 + (2*n + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_as_sum_of_squares_l1091_109171


namespace NUMINAMATH_CALUDE_triangle_properties_l1091_109145

-- Define what it means for a triangle to be equilateral
def is_equilateral (triangle : Type) : Prop := sorry

-- Define what it means for a triangle to be isosceles
def is_isosceles (triangle : Type) : Prop := sorry

-- The main theorem
theorem triangle_properties :
  (∀ t, is_equilateral t → is_isosceles t) ∧
  (∃ t, is_isosceles t ∧ ¬is_equilateral t) ∧
  (∃ t, ¬is_equilateral t ∧ is_isosceles t) := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1091_109145


namespace NUMINAMATH_CALUDE_trigonometric_equality_l1091_109119

theorem trigonometric_equality : 
  1 / Real.cos (40 * π / 180) - 2 * Real.sqrt 3 / Real.sin (40 * π / 180) = -4 * Real.tan (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l1091_109119


namespace NUMINAMATH_CALUDE_geometric_sequence_special_sum_l1091_109162

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_special_sum
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a3 : a 3 = Real.sqrt 2 - 1)
  (h_a5 : a 5 = Real.sqrt 2 + 1) :
  a 3 ^ 2 + 2 * a 2 * a 6 + a 3 * a 7 = 8 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_sum_l1091_109162


namespace NUMINAMATH_CALUDE_gcd_36_54_l1091_109148

theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_36_54_l1091_109148


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l1091_109168

theorem relationship_between_exponents 
  (m p t q : ℝ) 
  (n r s u : ℕ) 
  (h1 : (m^n)^2 = p^r)
  (h2 : p^r = t)
  (h3 : p^s = (m^u)^3)
  (h4 : (m^u)^3 = q)
  : 3 * u * r = 2 * n * s := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l1091_109168


namespace NUMINAMATH_CALUDE_girls_in_school_l1091_109185

/-- The number of girls in a school given stratified sampling information -/
theorem girls_in_school (total : ℕ) (sample : ℕ) (girl_sample : ℕ) 
  (h_total : total = 1600)
  (h_sample : sample = 200)
  (h_girl_sample : girl_sample = 95)
  (h_ratio : (girl_sample : ℚ) / sample = (↑girls : ℚ) / total) :
  girls = 760 :=
sorry

end NUMINAMATH_CALUDE_girls_in_school_l1091_109185


namespace NUMINAMATH_CALUDE_complex_multiplication_result_l1091_109158

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_multiplication_result : i^2 * (1 + i) = -1 - i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_result_l1091_109158


namespace NUMINAMATH_CALUDE_symmetry_probability_l1091_109184

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- The size of the square grid -/
def gridSize : Nat := 13

/-- The center point P -/
def centerPoint : GridPoint := ⟨7, 7⟩

/-- The total number of points in the grid -/
def totalPoints : Nat := gridSize * gridSize

/-- The number of points excluding the center point -/
def pointsExcludingCenter : Nat := totalPoints - 1

/-- Checks if a point is on a line of symmetry through the center -/
def isOnSymmetryLine (q : GridPoint) : Prop :=
  q.x = centerPoint.x ∨ 
  q.y = centerPoint.y ∨ 
  q.x - centerPoint.x = q.y - centerPoint.y ∨
  q.x - centerPoint.x = centerPoint.y - q.y

/-- The number of points on lines of symmetry (excluding the center) -/
def symmetricPoints : Nat := 48

/-- The theorem stating the probability of Q being on a line of symmetry -/
theorem symmetry_probability : 
  (symmetricPoints : ℚ) / pointsExcludingCenter = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_symmetry_probability_l1091_109184


namespace NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l1091_109175

/-- An ellipse E with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The area of the quadrilateral formed by the vertices of an ellipse -/
def quadrilateral_area (E : Ellipse) : ℝ := 2 * E.a * E.b

theorem ellipse_equation_from_conditions (E : Ellipse) 
  (h_vertex : ellipse_equation E 0 (-2))
  (h_area : quadrilateral_area E = 4 * Real.sqrt 5) :
  ∀ x y, ellipse_equation E x y ↔ x^2 / 5 + y^2 / 4 = 1 := by
  sorry

#check ellipse_equation_from_conditions

end NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l1091_109175


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_is_eight_l1091_109151

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of n consecutive integers starting from a -/
def sum_consecutive (a n : ℕ) : ℕ := n * a + sum_first_n (n - 1)

/-- The proposition that n is the largest number of positive consecutive integers summing to 36 -/
def is_largest_consecutive_sum (n : ℕ) : Prop :=
  (∃ a : ℕ, a > 0 ∧ sum_consecutive a n = 36) ∧
  (∀ m : ℕ, m > n → ∀ a : ℕ, a > 0 → sum_consecutive a m ≠ 36)

theorem largest_consecutive_sum_is_eight :
  is_largest_consecutive_sum 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_is_eight_l1091_109151


namespace NUMINAMATH_CALUDE_constant_quantity_l1091_109149

/-- A sequence of real numbers satisfying the recurrence relation a_{n+2} = a_{n+1} + a_n -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = a (n + 1) + a n

/-- The theorem stating that |a_n^2 - a_{n-1} a_{n+1}| is constant for n ≥ 2 -/
theorem constant_quantity (a : ℕ → ℝ) (h : RecurrenceSequence a) :
  ∃ c : ℝ, ∀ n : ℕ, n ≥ 2 → |a n ^ 2 - a (n - 1) * a (n + 1)| = c :=
sorry

end NUMINAMATH_CALUDE_constant_quantity_l1091_109149


namespace NUMINAMATH_CALUDE_problem_solution_l1091_109121

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 1) / Real.log (1/2)

def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 6

theorem problem_solution :
  ∀ a : ℝ,
  (∀ x : ℝ, g a x = g a (-x)) →
  (a = 0 ∧ ∀ x > 0, ∀ y > x, g a y > g a x) ∧
  (({x : ℝ | g a x < 0} = {x : ℝ | 2 < x ∧ x < 3}) →
    (∀ x > 1, g a x / (x - 1) ≥ 2 * Real.sqrt 2 - 3) ∧
    (∃ x > 1, g a x / (x - 1) = 2 * Real.sqrt 2 - 3)) ∧
  ((∀ x₁ ≥ 1, ∀ x₂ ∈ Set.Icc (-2) 4, f x₁ ≤ g a x₂) →
    -11/2 ≤ a ∧ a ≤ 2 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1091_109121


namespace NUMINAMATH_CALUDE_composite_function_evaluation_l1091_109181

/-- Given two functions f and g, prove that f(g(f(3))) = 79 -/
theorem composite_function_evaluation :
  let f : ℝ → ℝ := λ x ↦ 2 * x + 5
  let g : ℝ → ℝ := λ x ↦ 3 * x + 4
  f (g (f 3)) = 79 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_evaluation_l1091_109181


namespace NUMINAMATH_CALUDE_honey_purchase_cost_l1091_109118

def honey_problem (bulk_price min_spend tax_rate excess_pounds : ℕ) : Prop :=
  let min_pounds : ℕ := min_spend / bulk_price
  let total_pounds : ℕ := min_pounds + excess_pounds
  let pre_tax_cost : ℕ := total_pounds * bulk_price
  let tax_amount : ℕ := total_pounds * tax_rate
  let total_cost : ℕ := pre_tax_cost + tax_amount
  total_cost = 240

theorem honey_purchase_cost :
  honey_problem 5 40 1 32 := by sorry

end NUMINAMATH_CALUDE_honey_purchase_cost_l1091_109118


namespace NUMINAMATH_CALUDE_water_velocity_proof_l1091_109179

-- Define the relationship between force, height, and velocity
def force_relation (k : ℝ) (H : ℝ) (V : ℝ) : ℝ := k * H * V^3

-- Theorem statement
theorem water_velocity_proof :
  ∀ k : ℝ,
  -- Given conditions
  (force_relation k 1 5 = 100) →
  -- Prove that
  (force_relation k 8 10 = 6400) :=
by
  sorry

end NUMINAMATH_CALUDE_water_velocity_proof_l1091_109179


namespace NUMINAMATH_CALUDE_map_scale_theorem_l1091_109139

/-- Represents the scale of a map as a ratio of 1 to some natural number. -/
structure MapScale where
  ratio : ℕ
  property : ratio > 0

/-- Calculates the map scale given the real distance and the corresponding map distance. -/
def calculate_map_scale (real_distance : ℕ) (map_distance : ℕ) : MapScale :=
  { ratio := real_distance / map_distance
    property := sorry }

theorem map_scale_theorem (real_km : ℕ) (map_cm : ℕ) 
  (h1 : real_km = 30) (h2 : map_cm = 20) : 
  (calculate_map_scale (real_km * 100000) map_cm).ratio = 150000 := by
  sorry

#check map_scale_theorem

end NUMINAMATH_CALUDE_map_scale_theorem_l1091_109139


namespace NUMINAMATH_CALUDE_average_score_is_1_9_l1091_109187

/-- Represents the score distribution for a test -/
structure ScoreDistribution where
  threePoints : Rat
  twoPoints : Rat
  onePoint : Rat
  zeroPoints : Rat

/-- Calculates the average score given a score distribution and number of students -/
def averageScore (dist : ScoreDistribution) (numStudents : ℕ) : ℚ :=
  (3 * dist.threePoints + 2 * dist.twoPoints + dist.onePoint) * numStudents / 100

/-- Theorem: The average score for the given test is 1.9 -/
theorem average_score_is_1_9 :
  let dist : ScoreDistribution := {
    threePoints := 30,
    twoPoints := 40,
    onePoint := 20,
    zeroPoints := 10
  }
  averageScore dist 30 = 19/10 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_1_9_l1091_109187


namespace NUMINAMATH_CALUDE_cylinder_cut_face_area_l1091_109170

/-- Represents a cylinder with given radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents a cut through the cylinder -/
structure CylinderCut (c : Cylinder) where
  arcAngle : ℝ  -- Angle between the two points on the circular face

/-- The area of the rectangular face resulting from the cut -/
def cutFaceArea (c : Cylinder) (cut : CylinderCut c) : ℝ :=
  c.height * (2 * c.radius)

theorem cylinder_cut_face_area 
  (c : Cylinder) 
  (cut : CylinderCut c) 
  (h_radius : c.radius = 4) 
  (h_height : c.height = 10) 
  (h_angle : cut.arcAngle = π) : 
  cutFaceArea c cut = 80 := by
  sorry

#eval (80 : ℤ) + (0 : ℤ) + (1 : ℤ)  -- Should evaluate to 81

end NUMINAMATH_CALUDE_cylinder_cut_face_area_l1091_109170


namespace NUMINAMATH_CALUDE_root_sum_equals_three_l1091_109190

-- Define the equations
def equation1 (x : ℝ) : Prop := x + Real.log x = 3
def equation2 (x : ℝ) : Prop := x + (10 : ℝ)^x = 3

-- State the theorem
theorem root_sum_equals_three :
  ∀ x₁ x₂ : ℝ, equation1 x₁ → equation2 x₂ → x₁ + x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_equals_three_l1091_109190


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1091_109176

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 5 = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1091_109176


namespace NUMINAMATH_CALUDE_popsicle_stick_difference_l1091_109150

theorem popsicle_stick_difference :
  let num_boys : ℕ := 10
  let num_girls : ℕ := 12
  let sticks_per_boy : ℕ := 15
  let sticks_per_girl : ℕ := 12
  let total_boys_sticks := num_boys * sticks_per_boy
  let total_girls_sticks := num_girls * sticks_per_girl
  total_boys_sticks - total_girls_sticks = 6 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_stick_difference_l1091_109150


namespace NUMINAMATH_CALUDE_associated_equation_part1_associated_equation_part2_associated_equation_part3_l1091_109100

-- Part 1
theorem associated_equation_part1 (x : ℝ) : 
  (2 * x - 5 > 3 * x - 8 ∧ -4 * x + 3 < x - 4) → 
  (x - (3 * x + 1) = -5) :=
sorry

-- Part 2
theorem associated_equation_part2 (x : ℤ) : 
  (x - 1/4 < 1 ∧ 4 + 2 * x > -7 * x + 5) → 
  (x - 1 = 0) :=
sorry

-- Part 3
theorem associated_equation_part3 (m : ℝ) : 
  (∀ x : ℝ, (x < 2 * x - m ∧ x - 2 ≤ m) → 
  (2 * x - 1 = x + 2 ∨ 3 + x = 2 * (x + 1/2))) → 
  (1 ≤ m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_associated_equation_part1_associated_equation_part2_associated_equation_part3_l1091_109100


namespace NUMINAMATH_CALUDE_count_primes_between_50_and_70_l1091_109115

theorem count_primes_between_50_and_70 : 
  (Finset.filter Nat.Prime (Finset.range 19)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_primes_between_50_and_70_l1091_109115


namespace NUMINAMATH_CALUDE_problem_solution_l1091_109182

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

noncomputable def g (x : ℝ) : ℝ := x - 2 * Real.log x - 1

theorem problem_solution :
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 → f a x ≥ 1 + Real.log 2) ∧
  (∀ (x : ℝ), x > 0 → HasDerivAt g ((x - 2) / x) x) ∧
  (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ → (x₁ - x₂) / (Real.log x₁ - Real.log x₂) < 2 * x₂) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1091_109182


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_2100210021_base_3_l1091_109106

def base_3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem largest_prime_divisor_of_2100210021_base_3 :
  let n := base_3_to_decimal [1, 2, 0, 0, 1, 2, 0, 0, 1, 2]
  ∃ (p : Nat), is_prime p ∧ p ∣ n ∧ ∀ (q : Nat), is_prime q → q ∣ n → q ≤ p ∧ p = 46501 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_2100210021_base_3_l1091_109106


namespace NUMINAMATH_CALUDE_tree_spacing_l1091_109122

/-- Given a yard of length 400 meters with 26 trees planted at equal distances,
    including one at each end, the distance between two consecutive trees is 16 meters. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (tree_spacing : ℝ) :
  yard_length = 400 →
  num_trees = 26 →
  tree_spacing * (num_trees - 1) = yard_length →
  tree_spacing = 16 := by
sorry

end NUMINAMATH_CALUDE_tree_spacing_l1091_109122


namespace NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l1091_109142

/-- The number of tickets sold by the Richmond Tigers in the second half of the season -/
def second_half_tickets (total : ℕ) (first_half : ℕ) : ℕ :=
  total - first_half

/-- Theorem stating that the number of tickets sold in the second half of the season is 5703 -/
theorem richmond_tigers_ticket_sales :
  second_half_tickets 9570 3867 = 5703 := by
  sorry

end NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l1091_109142
