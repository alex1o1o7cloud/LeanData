import Mathlib

namespace NUMINAMATH_CALUDE_min_values_ab_and_fraction_l3960_396000

theorem min_values_ab_and_fraction (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 1/a + 3/b = 1) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 3/y = 1 ∧ x*y ≤ a*b) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 3/y = 1 ∧ x*y = 12) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 3/y = 1 → x*y ≥ 12) ∧
  (∃ (x y : ℝ), x > 1 ∧ y > 3 ∧ 1/x + 3/y = 1 ∧ 1/(x-1) + 3/(y-3) ≤ 1/(a-1) + 3/(b-3)) ∧
  (∃ (x y : ℝ), x > 1 ∧ y > 3 ∧ 1/x + 3/y = 1 ∧ 1/(x-1) + 3/(y-3) = 2) ∧
  (∀ (x y : ℝ), x > 1 → y > 3 → 1/x + 3/y = 1 → 1/(x-1) + 3/(y-3) ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_values_ab_and_fraction_l3960_396000


namespace NUMINAMATH_CALUDE_blue_paint_calculation_l3960_396093

/-- Given a ratio of blue paint to white paint and the amount of white paint used,
    calculate the amount of blue paint required. -/
def blue_paint_required (blue_ratio : ℚ) (white_ratio : ℚ) (white_amount : ℚ) : ℚ :=
  (blue_ratio / white_ratio) * white_amount

/-- Theorem stating that given a 5:6 ratio of blue to white paint and 18 quarts of white paint,
    15 quarts of blue paint are required. -/
theorem blue_paint_calculation :
  let blue_ratio : ℚ := 5
  let white_ratio : ℚ := 6
  let white_amount : ℚ := 18
  blue_paint_required blue_ratio white_ratio white_amount = 15 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_calculation_l3960_396093


namespace NUMINAMATH_CALUDE_special_point_properties_l3960_396071

/-- A point in the second quadrant with coordinate product -10 -/
def special_point : ℝ × ℝ := (-2, 5)

theorem special_point_properties :
  let (x, y) := special_point
  x < 0 ∧ y > 0 ∧ x * y = -10 := by
  sorry

end NUMINAMATH_CALUDE_special_point_properties_l3960_396071


namespace NUMINAMATH_CALUDE_melies_money_left_l3960_396036

/-- Calculates the amount of money left after buying meat. -/
def money_left (meat_amount : ℝ) (cost_per_kg : ℝ) (initial_money : ℝ) : ℝ :=
  initial_money - meat_amount * cost_per_kg

/-- Proves that Méliès has $16 left after buying meat. -/
theorem melies_money_left :
  let meat_amount : ℝ := 2
  let cost_per_kg : ℝ := 82
  let initial_money : ℝ := 180
  money_left meat_amount cost_per_kg initial_money = 16 := by
  sorry

end NUMINAMATH_CALUDE_melies_money_left_l3960_396036


namespace NUMINAMATH_CALUDE_dividend_divisor_quotient_problem_l3960_396008

theorem dividend_divisor_quotient_problem :
  ∀ (dividend divisor quotient : ℕ),
    dividend = 6 * divisor →
    divisor = 6 * quotient →
    dividend = divisor * quotient →
    dividend = 216 ∧ divisor = 36 ∧ quotient = 6 := by
  sorry

end NUMINAMATH_CALUDE_dividend_divisor_quotient_problem_l3960_396008


namespace NUMINAMATH_CALUDE_anna_bob_numbers_not_equal_l3960_396070

/-- Represents a number formed by concatenating consecutive positive integers -/
def ConsecutiveIntegerNumber (start : ℕ) (count : ℕ) : ℕ := sorry

/-- Anna's number is formed by 20 consecutive positive integers -/
def AnnaNumber (start : ℕ) : ℕ := ConsecutiveIntegerNumber start 20

/-- Bob's number is formed by 21 consecutive positive integers -/
def BobNumber (start : ℕ) : ℕ := ConsecutiveIntegerNumber start 21

/-- Theorem stating that Anna's and Bob's numbers cannot be equal -/
theorem anna_bob_numbers_not_equal :
  ∀ (a b : ℕ), AnnaNumber a ≠ BobNumber b :=
sorry

end NUMINAMATH_CALUDE_anna_bob_numbers_not_equal_l3960_396070


namespace NUMINAMATH_CALUDE_rational_sum_theorem_l3960_396051

theorem rational_sum_theorem (a b c : ℚ) 
  (h1 : a * b * c < 0) 
  (h2 : a + b + c = 0) : 
  (a - b - c) / abs a + (b - c - a) / abs b + (c - a - b) / abs c = 2 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_theorem_l3960_396051


namespace NUMINAMATH_CALUDE_power_of_power_l3960_396018

theorem power_of_power (x : ℝ) : (x^3)^2 = x^6 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l3960_396018


namespace NUMINAMATH_CALUDE_no_singleton_set_with_conditions_l3960_396099

theorem no_singleton_set_with_conditions :
  ¬ ∃ (A : Set ℝ), (∃ (a : ℝ), A = {a}) ∧
    (∀ a : ℝ, a ∈ A → (1 / (1 - a)) ∈ A) ∧
    (1 ∈ A) := by
  sorry

end NUMINAMATH_CALUDE_no_singleton_set_with_conditions_l3960_396099


namespace NUMINAMATH_CALUDE_probability_a_and_b_selected_l3960_396043

-- Define the total number of students
def total_students : ℕ := 5

-- Define the number of students to be selected
def selected_students : ℕ := 3

-- Define a function to calculate combinations
def combination (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Theorem statement
theorem probability_a_and_b_selected :
  (combination (total_students - 2) (selected_students - 2)) / 
  (combination total_students selected_students) = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_a_and_b_selected_l3960_396043


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3960_396089

theorem solve_linear_equation :
  ∃ x : ℚ, 4 * (2 * x - 1) = 1 - 3 * (x + 2) ∧ x = -1/11 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3960_396089


namespace NUMINAMATH_CALUDE_order_of_abc_l3960_396074

theorem order_of_abc : 
  let a : ℝ := 1 / (6 * Real.sqrt 15)
  let b : ℝ := (3/4) * Real.sin (1/60)
  let c : ℝ := Real.log (61/60)
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l3960_396074


namespace NUMINAMATH_CALUDE_canDisplay_totalCans_l3960_396045

/-- The number of cans in each layer forms an arithmetic sequence -/
def canSequence (n : ℕ) : ℕ := 35 - 3 * n

/-- The total number of layers in the display -/
def numLayers : ℕ := 12

/-- The total number of cans in the display -/
def totalCans : ℕ := (numLayers * (canSequence 0 + canSequence (numLayers - 1))) / 2

theorem canDisplay_totalCans : totalCans = 216 := by
  sorry

end NUMINAMATH_CALUDE_canDisplay_totalCans_l3960_396045


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l3960_396046

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l3960_396046


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3960_396060

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {1, 2, 4}

theorem complement_union_theorem : (U \ B) ∪ A = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3960_396060


namespace NUMINAMATH_CALUDE_shifted_line_equation_l3960_396096

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line vertically by a given amount -/
def shift_line (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift }

theorem shifted_line_equation (x y : ℝ) :
  let original_line := Line.mk (-2) 0
  let shifted_line := shift_line original_line 3
  y = shifted_line.slope * x + shifted_line.intercept ↔ y = -2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_shifted_line_equation_l3960_396096


namespace NUMINAMATH_CALUDE_sarah_trucks_l3960_396037

/-- The number of trucks Sarah had initially -/
def initial_trucks : ℕ := 51

/-- The number of trucks Sarah gave to Jeff -/
def trucks_given : ℕ := 13

/-- The number of trucks Sarah has now -/
def remaining_trucks : ℕ := initial_trucks - trucks_given

theorem sarah_trucks : remaining_trucks = 38 := by
  sorry

end NUMINAMATH_CALUDE_sarah_trucks_l3960_396037


namespace NUMINAMATH_CALUDE_intersection_M_N_l3960_396021

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3960_396021


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_fifty_l3960_396017

theorem largest_multiple_of_seven_less_than_fifty :
  ∃ n : ℕ, n = 49 ∧ 7 ∣ n ∧ n < 50 ∧ ∀ m : ℕ, 7 ∣ m → m < 50 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_fifty_l3960_396017


namespace NUMINAMATH_CALUDE_binomial_expansion_equal_terms_l3960_396065

theorem binomial_expansion_equal_terms (p q : ℝ) (hp : p > 0) (hq : q > 0) : 
  10 * p^9 * q = 45 * p^8 * q^2 → p + 2*q = 1 → p = 9/13 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_equal_terms_l3960_396065


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3960_396079

theorem fraction_sum_equality : (2 : ℚ) / 15 + 4 / 20 + 5 / 45 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3960_396079


namespace NUMINAMATH_CALUDE_gcd_lcm_product_75_90_l3960_396095

theorem gcd_lcm_product_75_90 : Nat.gcd 75 90 * Nat.lcm 75 90 = 6750 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_75_90_l3960_396095


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l3960_396083

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root if and only if b ∈ (-∞, -10] ∪ [10, ∞) -/
theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l3960_396083


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l3960_396010

/-- A curve defined by y = kx + ln x has a tangent at the point (1, k) that is parallel to the x-axis if and only if k = -1 -/
theorem tangent_parallel_to_x_axis (k : ℝ) : 
  (∃ f : ℝ → ℝ, f x = k * x + Real.log x) →
  (∃ t : ℝ → ℝ, t x = k * x + Real.log 1) →
  (∀ x : ℝ, (k + 1 / x) = 0) →
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l3960_396010


namespace NUMINAMATH_CALUDE_mineral_age_arrangements_eq_60_l3960_396073

/-- The number of arrangements for a six-digit number using 2, 2, 4, 4, 7, 9, starting with an odd digit -/
def mineral_age_arrangements : ℕ :=
  let digits : List ℕ := [2, 2, 4, 4, 7, 9]
  let odd_digits : List ℕ := digits.filter (λ d => d % 2 = 1)
  let remaining_digits : ℕ := digits.length - 1
  let repeated_digits : List ℕ := [2, 4]
  odd_digits.length * (remaining_digits.factorial / (repeated_digits.map (λ d => (digits.count d).factorial)).prod)

/-- Theorem stating that the number of possible arrangements is 60 -/
theorem mineral_age_arrangements_eq_60 : mineral_age_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_mineral_age_arrangements_eq_60_l3960_396073


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3960_396026

theorem min_value_of_expression (x y z : ℝ) (h : x + y + 3 * z = 6) :
  ∃ (m : ℝ), m = 0 ∧ ∀ (x' y' z' : ℝ), x' + y' + 3 * z' = 6 → x' * y' + 2 * x' * z' + 3 * y' * z' ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3960_396026


namespace NUMINAMATH_CALUDE_exponent_division_l3960_396088

theorem exponent_division (x : ℝ) : x^6 / x^2 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3960_396088


namespace NUMINAMATH_CALUDE_x_minus_y_values_l3960_396056

theorem x_minus_y_values (x y : ℝ) (h1 : x^2 = 4) (h2 : |y| = 3) (h3 : x + y < 0) :
  x - y = 1 ∨ x - y = 5 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l3960_396056


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3960_396030

/-- Given an ellipse and a hyperbola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b c m n c' : ℝ) (e e' : ℝ) : 
  (∀ x y : ℝ, 2 * x^2 + y^2 = 2) →  -- Ellipse equation
  (a^2 = 2 ∧ b^2 = 1) →             -- Semi-major and semi-minor axes of ellipse
  (c = (a^2 - b^2).sqrt) →          -- Focal length of ellipse
  (e = c / a) →                     -- Eccentricity of ellipse
  (m = a) →                         -- Semi-major axis of hyperbola
  (e' * e = 1) →                    -- Product of eccentricities
  (c' = m * e') →                   -- Focal length of hyperbola
  (n^2 = c'^2 - m^2) →              -- Semi-minor axis of hyperbola
  (∀ x y : ℝ, y^2 / n^2 - x^2 / m^2 = 1) →  -- Standard form of hyperbola
  (∀ x y : ℝ, y^2 - x^2 = 2) :=     -- Desired hyperbola equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3960_396030


namespace NUMINAMATH_CALUDE_function_decreasing_iff_a_in_range_l3960_396080

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x + 5 else 3 * a / x

theorem function_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) < 0) ↔ 0 < a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_function_decreasing_iff_a_in_range_l3960_396080


namespace NUMINAMATH_CALUDE_probability_of_pair_letter_l3960_396002

def word : String := "PROBABILITY"
def target : String := "PAIR"

theorem probability_of_pair_letter : 
  (word.toList.filter (fun c => target.contains c)).length / word.length = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_pair_letter_l3960_396002


namespace NUMINAMATH_CALUDE_polynomial_identity_l3960_396069

theorem polynomial_identity (x : ℝ) (h : x + 1/x = 3) :
  x^12 - 7*x^6 + x^2 = 45363*x - 17327 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3960_396069


namespace NUMINAMATH_CALUDE_chess_piece_position_l3960_396009

theorem chess_piece_position : ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 + x*y - 2*y^2 = 13 ∧ x = 5 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_chess_piece_position_l3960_396009


namespace NUMINAMATH_CALUDE_inequality_solution_l3960_396087

theorem inequality_solution (x : ℝ) : 
  (x^(1/4) + 3 / (x^(1/4) + 4) ≤ 1) ↔ 
  (x < (((-3 - Real.sqrt 5) / 2)^4) ∨ x > (((-3 + Real.sqrt 5) / 2)^4)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3960_396087


namespace NUMINAMATH_CALUDE_no_distinct_unit_fraction_sum_l3960_396090

theorem no_distinct_unit_fraction_sum (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  ¬∃ (a b : ℕ), a ≠ b ∧ a > 0 ∧ b > 0 ∧ (p - 1 : ℚ) / p = 1 / a + 1 / b :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_unit_fraction_sum_l3960_396090


namespace NUMINAMATH_CALUDE_equation_solutions_l3960_396016

theorem equation_solutions :
  (∃ x : ℝ, 5 * x - 2 = 2 * (x + 2) ∧ x = 2) ∧
  (∃ x : ℝ, 2 * x + (x - 3) / 2 = (2 - x) / 3 - 5 ∧ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3960_396016


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_18_and_640_l3960_396061

theorem smallest_n_divisible_by_18_and_640 : ∃! n : ℕ+, 
  (∀ m : ℕ+, m < n → (¬(18 ∣ m^2) ∨ ¬(640 ∣ m^3))) ∧ 
  (18 ∣ n^2) ∧ (640 ∣ n^3) :=
by
  use 120
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_18_and_640_l3960_396061


namespace NUMINAMATH_CALUDE_max_abs_f_value_l3960_396085

-- Define the band region type
def band_region (k l : ℝ) (y : ℝ) : Prop := k ≤ y ∧ y ≤ l

-- Define the quadratic function
variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem max_abs_f_value :
  ∀ a b c : ℝ,
  (band_region 0 4 (f a b c (-2) + 2)) ∧
  (band_region 0 4 (f a b c 0 + 2)) ∧
  (band_region 0 4 (f a b c 2 + 2)) →
  (∀ t : ℝ, band_region (-1) 3 (t + 1) → |f a b c t| ≤ 5/2) ∧
  (∃ t : ℝ, band_region (-1) 3 (t + 1) ∧ |f a b c t| = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_max_abs_f_value_l3960_396085


namespace NUMINAMATH_CALUDE_exactly_one_divisible_by_five_l3960_396092

theorem exactly_one_divisible_by_five (a : ℤ) (h : ¬ (5 ∣ a)) :
  (5 ∣ (a^2 - 1)) ≠ (5 ∣ (a^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_divisible_by_five_l3960_396092


namespace NUMINAMATH_CALUDE_tan_675_degrees_l3960_396034

theorem tan_675_degrees : Real.tan (675 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_675_degrees_l3960_396034


namespace NUMINAMATH_CALUDE_power_of_three_mod_ten_l3960_396077

theorem power_of_three_mod_ten : 3^19 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_power_of_three_mod_ten_l3960_396077


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_t_equals_one_l3960_396004

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is perpendicular to c, then t = 1 -/
theorem perpendicular_vectors_imply_t_equals_one (a b c : ℝ × ℝ) :
  a = (Real.sqrt 3, 1) →
  b = (0, 1) →
  c = (-Real.sqrt 3, t) →
  (a.1 + 2 * b.1, a.2 + 2 * b.2) • c = 0 →
  t = 1 := by
  sorry

#check perpendicular_vectors_imply_t_equals_one

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_t_equals_one_l3960_396004


namespace NUMINAMATH_CALUDE_simultaneous_inequalities_solution_l3960_396078

theorem simultaneous_inequalities_solution (x : ℝ) :
  (x^2 - 8*x + 12 < 0 ∧ 2*x - 4 > 0) ↔ (x > 2 ∧ x < 6) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_inequalities_solution_l3960_396078


namespace NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_angle_l3960_396094

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_of_angles : (angles 0) + (angles 1) + (angles 2) = 180

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Theorem statement
theorem triangle_at_most_one_obtuse_angle (t : Triangle) : 
  ¬(∃ (i j : Fin 3), i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j)) :=
sorry

end NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_angle_l3960_396094


namespace NUMINAMATH_CALUDE_tv_price_calculation_l3960_396019

/-- The actual selling price of a television set given its cost price,
    markup percentage, and discount percentage. -/
def actual_selling_price (cost_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  cost_price * (1 + markup_percent) * discount_percent

/-- Theorem stating that for a television with cost price 'a',
    25% markup, and 70% discount, the actual selling price is 70%(1+25%)a. -/
theorem tv_price_calculation (a : ℝ) :
  actual_selling_price a 0.25 0.7 = 0.7 * (1 + 0.25) * a := by
  sorry

#check tv_price_calculation

end NUMINAMATH_CALUDE_tv_price_calculation_l3960_396019


namespace NUMINAMATH_CALUDE_perimeter_ratio_triangles_l3960_396035

theorem perimeter_ratio_triangles :
  let small_triangle_sides : Fin 3 → ℝ := ![4, 8, 4 * Real.sqrt 3]
  let large_triangle_sides : Fin 3 → ℝ := ![8, 8, 8 * Real.sqrt 2]
  let small_perimeter := (Finset.univ.sum small_triangle_sides)
  let large_perimeter := (Finset.univ.sum large_triangle_sides)
  small_perimeter / large_perimeter = (4 + 8 + 4 * Real.sqrt 3) / (8 + 8 + 8 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_triangles_l3960_396035


namespace NUMINAMATH_CALUDE_matrix_product_AB_l3960_396044

def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, -2; 4, 0]
def B : Matrix (Fin 2) (Fin 1) ℝ := !![5; -1]

theorem matrix_product_AB :
  A * B = !![17; 20] := by sorry

end NUMINAMATH_CALUDE_matrix_product_AB_l3960_396044


namespace NUMINAMATH_CALUDE_largest_value_l3960_396022

def expr_a : ℤ := 2 * 0 * 2006
def expr_b : ℤ := 2 * 0 + 6
def expr_c : ℤ := 2 + 0 * 2006
def expr_d : ℤ := 2 * (0 + 6)
def expr_e : ℤ := 2006 * 0 + 0 * 6

theorem largest_value : 
  expr_d = max expr_a (max expr_b (max expr_c (max expr_d expr_e))) :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l3960_396022


namespace NUMINAMATH_CALUDE_chosen_number_l3960_396066

theorem chosen_number (x : ℝ) : x / 8 - 100 = 6 → x = 848 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_l3960_396066


namespace NUMINAMATH_CALUDE_cosine_sum_equality_l3960_396003

theorem cosine_sum_equality : 
  Real.cos (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (43 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_equality_l3960_396003


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3960_396048

theorem arithmetic_sequence_sum (d : ℝ) (h : d ≠ 0) :
  let a : ℕ → ℝ := λ n => (n - 1 : ℝ) * d
  ∃ m : ℕ, a m = (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) ∧ m = 37 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3960_396048


namespace NUMINAMATH_CALUDE_log7_2400_rounded_to_nearest_integer_l3960_396006

-- Define the logarithm base 7 function
noncomputable def log7 (x : ℝ) : ℝ := Real.log x / Real.log 7

-- Theorem statement
theorem log7_2400_rounded_to_nearest_integer :
  ⌊log7 2400 + 0.5⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_log7_2400_rounded_to_nearest_integer_l3960_396006


namespace NUMINAMATH_CALUDE_alice_monthly_increase_l3960_396098

/-- Represents Alice's savings pattern over three months -/
def aliceSavings (initialSavings : ℝ) (monthlyIncrease : ℝ) : ℝ :=
  initialSavings + (initialSavings + monthlyIncrease) + (initialSavings + 2 * monthlyIncrease)

/-- Theorem stating Alice's monthly savings increase -/
theorem alice_monthly_increase (initialSavings totalSavings : ℝ) 
  (h1 : initialSavings = 10)
  (h2 : totalSavings = 70)
  (h3 : ∃ x : ℝ, aliceSavings initialSavings x = totalSavings) :
  ∃ x : ℝ, x = 40 / 3 ∧ aliceSavings initialSavings x = totalSavings :=
sorry

end NUMINAMATH_CALUDE_alice_monthly_increase_l3960_396098


namespace NUMINAMATH_CALUDE_football_field_area_l3960_396027

theorem football_field_area (total_fertilizer : ℝ) (partial_fertilizer : ℝ) (partial_area : ℝ) :
  total_fertilizer = 1200 →
  partial_fertilizer = 400 →
  partial_area = 3600 →
  (total_fertilizer / (partial_fertilizer / partial_area)) = 10800 := by
  sorry

end NUMINAMATH_CALUDE_football_field_area_l3960_396027


namespace NUMINAMATH_CALUDE_expression_value_l3960_396012

theorem expression_value : ∀ x : ℝ, x ≠ 5 →
  (x^2 - 3*x - 10) / (x - 5) = x + 2 ∧
  ((1^2 - 3*1 - 10) / (1 - 5) = 3) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3960_396012


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3960_396020

theorem quadratic_inequality_solution_set : 
  {x : ℝ | x^2 - 3*x - 18 ≤ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 6} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3960_396020


namespace NUMINAMATH_CALUDE_course_size_is_400_l3960_396081

/-- Proves that the total number of students in a course is 400, given the distribution of grades --/
theorem course_size_is_400 (T : ℕ) 
  (grade_A : ℕ := T / 5)
  (grade_B : ℕ := T / 4)
  (grade_C : ℕ := T / 2)
  (grade_D : ℕ := 20)
  (total_sum : T = grade_A + grade_B + grade_C + grade_D) : T = 400 := by
  sorry

end NUMINAMATH_CALUDE_course_size_is_400_l3960_396081


namespace NUMINAMATH_CALUDE_number_of_factors_46464_l3960_396097

theorem number_of_factors_46464 : Nat.card (Nat.divisors 46464) = 36 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_46464_l3960_396097


namespace NUMINAMATH_CALUDE_deepak_age_l3960_396014

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 4 / 3 →
  arun_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l3960_396014


namespace NUMINAMATH_CALUDE_card_value_decrease_l3960_396040

theorem card_value_decrease (initial_value : ℝ) (h : initial_value > 0) : 
  let first_year_value := initial_value * (1 - 0.1)
  let second_year_value := first_year_value * (1 - 0.1)
  let total_decrease := (initial_value - second_year_value) / initial_value
  total_decrease = 0.19 := by
sorry

end NUMINAMATH_CALUDE_card_value_decrease_l3960_396040


namespace NUMINAMATH_CALUDE_probability_at_least_one_vowel_l3960_396015

/-- The probability of picking at least one vowel from two sets of letters -/
theorem probability_at_least_one_vowel (set1 set2 : Finset Char) 
  (vowels1 vowels2 : Finset Char) : 
  set1.card = 6 →
  set2.card = 6 →
  vowels1 ⊆ set1 →
  vowels2 ⊆ set2 →
  vowels1.card = 2 →
  vowels2.card = 1 →
  (set1.card * set2.card : ℚ)⁻¹ * 
    ((vowels1.card * set2.card) + (set1.card - vowels1.card) * vowels2.card) = 1/2 := by
  sorry

#check probability_at_least_one_vowel

end NUMINAMATH_CALUDE_probability_at_least_one_vowel_l3960_396015


namespace NUMINAMATH_CALUDE_factorization_proof_l3960_396001

theorem factorization_proof (x : ℝ) : (x + 3)^2 - (x + 3) = (x + 3) * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3960_396001


namespace NUMINAMATH_CALUDE_real_root_of_complex_quadratic_l3960_396024

theorem real_root_of_complex_quadratic (k : ℝ) (a : ℝ) :
  (∃ x : ℂ, x^2 + (k + 2*I)*x + (2 : ℂ) + k*I = 0) →
  (a^2 + (k + 2*I)*a + (2 : ℂ) + k*I = 0) →
  (a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_real_root_of_complex_quadratic_l3960_396024


namespace NUMINAMATH_CALUDE_arcade_spend_example_l3960_396039

/-- Calculates the total amount spent at an arcade given the time spent and cost per interval. -/
def arcade_spend (hours : ℕ) (cost_per_interval : ℚ) (interval_minutes : ℕ) : ℚ :=
  let total_minutes : ℕ := hours * 60
  let num_intervals : ℕ := total_minutes / interval_minutes
  ↑num_intervals * cost_per_interval

/-- Proves that spending 3 hours at an arcade using $0.50 every 6 minutes results in a total spend of $15. -/
theorem arcade_spend_example : arcade_spend 3 (1/2) 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arcade_spend_example_l3960_396039


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l3960_396023

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x + 1

-- State the theorem
theorem unique_function_satisfying_conditions :
  (∀ x y : ℝ, f (x^2) = (f x)^2 - 2*x*(f x)) ∧
  (∀ x : ℝ, f (-x) = f (x - 1)) ∧
  (∀ x y : ℝ, 1 < x → x < y → f x < f y) ∧
  (∀ x : ℝ, 0 < f x) ∧
  (∀ g : ℝ → ℝ, 
    ((∀ x y : ℝ, g (x^2) = (g x)^2 - 2*x*(g x)) ∧
     (∀ x : ℝ, g (-x) = g (x - 1)) ∧
     (∀ x y : ℝ, 1 < x → x < y → g x < g y) ∧
     (∀ x : ℝ, 0 < g x)) →
    (∀ x : ℝ, g x = f x)) :=
by sorry


end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l3960_396023


namespace NUMINAMATH_CALUDE_statement_D_is_false_l3960_396029

-- Define the set A_k
def A (k : ℕ) : Set ℤ := {x : ℤ | ∃ n : ℤ, x = 4 * n + k}

-- State the theorem
theorem statement_D_is_false : ¬ (∀ a b : ℤ, (a + b) ∈ A 3 → a ∈ A 1 ∧ b ∈ A 2) := by
  sorry

end NUMINAMATH_CALUDE_statement_D_is_false_l3960_396029


namespace NUMINAMATH_CALUDE_power_function_values_l3960_396059

-- Define the power function f
def f (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem power_function_values :
  (f 3 = 9) →
  (f 2 = 4) ∧ (∀ x, f (2*x + 1) = 4*x^2 + 4*x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_power_function_values_l3960_396059


namespace NUMINAMATH_CALUDE_figure_area_theorem_l3960_396072

theorem figure_area_theorem (x : ℝ) : 
  let square1_area := (2 * x)^2
  let square2_area := (5 * x)^2
  let triangle_area := (1/2) * (2 * x) * (5 * x)
  square1_area + square2_area + triangle_area = 850 → x = 5 := by
sorry

end NUMINAMATH_CALUDE_figure_area_theorem_l3960_396072


namespace NUMINAMATH_CALUDE_fliers_remaining_l3960_396082

theorem fliers_remaining (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ) : 
  total = 2000 →
  morning_fraction = 1 / 10 →
  afternoon_fraction = 1 / 4 →
  (total - total * morning_fraction) * (1 - afternoon_fraction) = 1350 := by
sorry

end NUMINAMATH_CALUDE_fliers_remaining_l3960_396082


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_value_l3960_396091

/-- Given vectors in R² -/
def a : Fin 2 → ℝ := ![3, 4]
def b (m : ℝ) : Fin 2 → ℝ := ![-1, 2*m]
def c (m : ℝ) : Fin 2 → ℝ := ![m, -4]

/-- The sum of vectors a and b -/
def a_plus_b (m : ℝ) : Fin 2 → ℝ := ![a 0 + b m 0, a 1 + b m 1]

/-- Dot product of two 2D vectors -/
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1)

/-- Theorem stating that if c is perpendicular to (a + b), then m = -8/3 -/
theorem perpendicular_vectors_imply_m_value :
  ∀ m : ℝ, dot_product (c m) (a_plus_b m) = 0 → m = -8/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_value_l3960_396091


namespace NUMINAMATH_CALUDE_intersection_and_union_subset_condition_l3960_396038

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Define set M
def M (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x < 2*a + 2}

-- Theorem for part 1
theorem intersection_and_union :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) := by sorry

-- Theorem for part 2
theorem subset_condition (a : ℝ) :
  M a ⊆ A ↔ a ≤ -3 ∨ a > 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_subset_condition_l3960_396038


namespace NUMINAMATH_CALUDE_belongs_to_32nd_group_l3960_396057

/-- The last number in the n-th group of odd numbers -/
def last_number_in_group (n : ℕ) : ℕ := 2 * n^2 - 1

/-- The first number in the n-th group of odd numbers -/
def first_number_in_group (n : ℕ) : ℕ := 2 * (n-1)^2 + 1

/-- Theorem stating that 1991 belongs to the 32nd group -/
theorem belongs_to_32nd_group : 
  first_number_in_group 32 ≤ 1991 ∧ 1991 ≤ last_number_in_group 32 :=
sorry

end NUMINAMATH_CALUDE_belongs_to_32nd_group_l3960_396057


namespace NUMINAMATH_CALUDE_man_speed_against_current_l3960_396064

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific speeds, the man's speed against the current is 20 km/hr. -/
theorem man_speed_against_current :
  speed_against_current 25 2.5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_against_current_l3960_396064


namespace NUMINAMATH_CALUDE_f_range_on_domain_l3960_396041

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem f_range_on_domain :
  ∃ (min max : ℝ), min = -1 ∧ max = 8 ∧
  (∀ x ∈ domain, min ≤ f x ∧ f x ≤ max) ∧
  (∃ x₁ ∈ domain, f x₁ = min) ∧
  (∃ x₂ ∈ domain, f x₂ = max) :=
sorry

end NUMINAMATH_CALUDE_f_range_on_domain_l3960_396041


namespace NUMINAMATH_CALUDE_prob_six_consecutive_heads_l3960_396086

/-- A fair coin is flipped 10 times. -/
def coin_flips : ℕ := 10

/-- The probability of getting heads on a single flip of a fair coin. -/
def prob_heads : ℚ := 1/2

/-- The set of all possible outcomes when flipping a coin 10 times. -/
def all_outcomes : Finset (Fin coin_flips → Bool) := sorry

/-- The set of outcomes with at least 6 consecutive heads. -/
def outcomes_with_six_consecutive_heads : Finset (Fin coin_flips → Bool) := sorry

/-- The probability of getting at least 6 consecutive heads in 10 flips of a fair coin. -/
theorem prob_six_consecutive_heads :
  (Finset.card outcomes_with_six_consecutive_heads : ℚ) / (Finset.card all_outcomes : ℚ) = 129/1024 :=
sorry

end NUMINAMATH_CALUDE_prob_six_consecutive_heads_l3960_396086


namespace NUMINAMATH_CALUDE_cycle_gain_percentage_l3960_396013

/-- Calculate the overall gain percentage for three cycles given their purchase and sale prices -/
theorem cycle_gain_percentage
  (purchase_a purchase_b purchase_c : ℕ)
  (sale_a sale_b sale_c : ℕ)
  (h_purchase_a : purchase_a = 1000)
  (h_purchase_b : purchase_b = 3000)
  (h_purchase_c : purchase_c = 6000)
  (h_sale_a : sale_a = 2000)
  (h_sale_b : sale_b = 4500)
  (h_sale_c : sale_c = 8000) :
  (((sale_a + sale_b + sale_c) - (purchase_a + purchase_b + purchase_c)) * 100) / (purchase_a + purchase_b + purchase_c) = 45 := by
  sorry


end NUMINAMATH_CALUDE_cycle_gain_percentage_l3960_396013


namespace NUMINAMATH_CALUDE_count_distinct_digits_eq_2352_l3960_396062

/-- The count of integers between 2000 and 9999 with four distinct digits, none of which is 5 -/
def count_distinct_digits : ℕ :=
  let first_digit := 7  -- 2, 3, 4, 6, 7, 8, 9
  let second_digit := 8 -- 0, 1, 2, 3, 4, 6, 7, 8, 9 (excluding the first digit)
  let third_digit := 7  -- remaining digits excluding 5 and the first two chosen
  let fourth_digit := 6 -- remaining digits excluding 5 and the first three chosen
  first_digit * second_digit * third_digit * fourth_digit

theorem count_distinct_digits_eq_2352 : count_distinct_digits = 2352 := by
  sorry

end NUMINAMATH_CALUDE_count_distinct_digits_eq_2352_l3960_396062


namespace NUMINAMATH_CALUDE_second_object_length_l3960_396033

def tape_length : ℕ := 5
def object1_length : ℕ := 100
def object2_length : ℕ := 780

theorem second_object_length :
  (tape_length ∣ object1_length) ∧ 
  (tape_length ∣ object2_length) ∧ 
  (∃ k : ℕ, k * tape_length = object2_length) →
  object2_length = 780 :=
by sorry

end NUMINAMATH_CALUDE_second_object_length_l3960_396033


namespace NUMINAMATH_CALUDE_rth_term_of_sequence_l3960_396047

-- Define the sum function for the arithmetic progression
def S (n : ℕ) : ℕ := 5 * n + 4 * n^2

-- Define the rth term of the sequence
def a_r (r : ℕ) : ℕ := S r - S (r - 1)

-- Theorem statement
theorem rth_term_of_sequence (r : ℕ) (h : r > 0) : a_r r = 8 * r + 1 := by
  sorry

end NUMINAMATH_CALUDE_rth_term_of_sequence_l3960_396047


namespace NUMINAMATH_CALUDE_boat_distance_proof_l3960_396063

/-- The speed of the boat in still water (mph) -/
def boat_speed : ℝ := 15.6

/-- The time taken for the trip against the current (hours) -/
def time_against : ℝ := 8

/-- The time taken for the return trip with the current (hours) -/
def time_with : ℝ := 5

/-- The speed of the current (mph) -/
def current_speed : ℝ := 3.6

/-- The distance traveled by the boat (miles) -/
def distance : ℝ := 96

theorem boat_distance_proof :
  distance = (boat_speed - current_speed) * time_against ∧
  distance = (boat_speed + current_speed) * time_with :=
by sorry

end NUMINAMATH_CALUDE_boat_distance_proof_l3960_396063


namespace NUMINAMATH_CALUDE_real_part_of_i_times_i_minus_one_l3960_396032

theorem real_part_of_i_times_i_minus_one :
  Complex.re (Complex.I * (Complex.I - 1)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_i_times_i_minus_one_l3960_396032


namespace NUMINAMATH_CALUDE_twelve_lines_theorem_l3960_396055

/-- A line in a plane. -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane. -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle in a plane. -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- The distance from a point to a line. -/
def distance_to_line (p : Point) (l : Line) : ℝ :=
  sorry

/-- Check if the distances from three points to a line are in one of the specified ratios. -/
def valid_ratio (A B C : Point) (l : Line) : Prop :=
  let dA := distance_to_line A l
  let dB := distance_to_line B l
  let dC := distance_to_line C l
  (dA = dB / 2 ∧ dA = dC / 2) ∨
  (dB = dA / 2 ∧ dB = dC / 2) ∨
  (dC = dA / 2 ∧ dC = dB / 2)

/-- The main theorem: there are exactly 12 lines satisfying the distance ratio condition for any triangle. -/
theorem twelve_lines_theorem (t : Triangle) : 
  ∃! (s : Finset Line), s.card = 12 ∧ ∀ l ∈ s, valid_ratio t.A t.B t.C l :=
sorry

end NUMINAMATH_CALUDE_twelve_lines_theorem_l3960_396055


namespace NUMINAMATH_CALUDE_removed_integer_problem_l3960_396053

theorem removed_integer_problem (n : ℕ) (x : ℕ) :
  x ≤ n →
  (n * (n + 1) / 2 - x) / (n - 1 : ℝ) = 163 / 4 →
  x = 61 :=
sorry

end NUMINAMATH_CALUDE_removed_integer_problem_l3960_396053


namespace NUMINAMATH_CALUDE_gillians_phone_bill_l3960_396054

theorem gillians_phone_bill (x : ℝ) : 
  (12 * (x * 1.1) = 660) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_gillians_phone_bill_l3960_396054


namespace NUMINAMATH_CALUDE_monic_quartic_with_specific_roots_l3960_396005

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 10*x^3 + 31*x^2 - 34*x - 7

-- Theorem statement
theorem monic_quartic_with_specific_roots :
  -- The polynomial is monic
  (∀ x, p x = x^4 + (-10)*x^3 + 31*x^2 + (-34)*x + (-7)) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d e : ℚ, ∀ x, p x = a*x^4 + b*x^3 + c*x^2 + d*x + e) ∧
  -- 3+√2 is a root
  p (3 + Real.sqrt 2) = 0 ∧
  -- 2-√5 is a root
  p (2 - Real.sqrt 5) = 0 :=
sorry

end NUMINAMATH_CALUDE_monic_quartic_with_specific_roots_l3960_396005


namespace NUMINAMATH_CALUDE_adjacent_repeat_percentage_is_16_l3960_396042

/-- The count of three-digit numbers -/
def three_digit_count : ℕ := 900

/-- The count of three-digit numbers with adjacent repeated digits -/
def adjacent_repeat_count : ℕ := 144

/-- The percentage of three-digit numbers with adjacent repeated digits -/
def adjacent_repeat_percentage : ℚ := adjacent_repeat_count / three_digit_count * 100

/-- Theorem stating that the percentage of three-digit numbers with adjacent repeated digits is 16.0% -/
theorem adjacent_repeat_percentage_is_16 :
  ⌊adjacent_repeat_percentage * 10⌋ / 10 = 16 :=
sorry

end NUMINAMATH_CALUDE_adjacent_repeat_percentage_is_16_l3960_396042


namespace NUMINAMATH_CALUDE_no_valid_tiling_l3960_396052

/-- Represents a tile on the grid -/
inductive Tile
  | OneByFour : Tile
  | TwoByTwo : Tile

/-- Represents a position on the 8x8 grid -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a placement of a tile on the grid -/
structure Placement :=
  (tile : Tile)
  (position : Position)

/-- Represents a tiling of the 8x8 grid -/
def Tiling := List Placement

/-- Checks if a tiling is valid (covers the entire grid without overlaps) -/
def isValidTiling (t : Tiling) : Prop := sorry

/-- Checks if a tiling uses exactly 15 1x4 tiles and 1 2x2 tile -/
def hasCorrectTileCount (t : Tiling) : Prop := sorry

/-- The main theorem stating that no valid tiling exists with the given constraints -/
theorem no_valid_tiling :
  ¬ ∃ (t : Tiling), isValidTiling t ∧ hasCorrectTileCount t := by
  sorry

end NUMINAMATH_CALUDE_no_valid_tiling_l3960_396052


namespace NUMINAMATH_CALUDE_mary_potatoes_l3960_396050

theorem mary_potatoes (initial_potatoes : ℕ) : 
  initial_potatoes - 3 = 5 → initial_potatoes = 8 := by
  sorry

end NUMINAMATH_CALUDE_mary_potatoes_l3960_396050


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l3960_396076

/-- The standard equation of an ellipse with given minor axis length and eccentricity -/
theorem ellipse_standard_equation (b : ℝ) (e : ℝ) : 
  b = 4 ∧ e = 3/5 → 
  ∃ (a : ℝ), (a > b) ∧ 
  ((∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ (x^2/25 + y^2/16 = 1 ∨ x^2/16 + y^2/25 = 1)) ∧
   e^2 = 1 - b^2/a^2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l3960_396076


namespace NUMINAMATH_CALUDE_sequence_term_exists_l3960_396007

theorem sequence_term_exists : ∃ n : ℕ, n * (n + 2) = 99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_exists_l3960_396007


namespace NUMINAMATH_CALUDE_remaining_payment_remaining_payment_specific_l3960_396084

/-- Given a product with a deposit, sales tax, and discount, calculate the remaining amount to be paid. -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) : ℝ :=
  let full_price := deposit / deposit_percentage
  let discounted_price := full_price * (1 - discount_rate)
  let final_price := discounted_price * (1 + sales_tax_rate)
  final_price - deposit

/-- Prove that the remaining payment for a product with given conditions is $733.20 -/
theorem remaining_payment_specific : 
  remaining_payment 80 0.1 0.07 0.05 = 733.20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_remaining_payment_specific_l3960_396084


namespace NUMINAMATH_CALUDE_fundamental_theorem_of_algebra_l3960_396049

-- Define a polynomial with complex coefficients
def ComplexPolynomial := ℂ → ℂ

-- State the fundamental theorem of algebra
theorem fundamental_theorem_of_algebra :
  ∀ (P : ComplexPolynomial), ∃ (z : ℂ), P z = 0 :=
sorry

end NUMINAMATH_CALUDE_fundamental_theorem_of_algebra_l3960_396049


namespace NUMINAMATH_CALUDE_range_of_expression_l3960_396028

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (z : ℝ), z = 5 * Real.arcsin x - 2 * Real.arccos y ∧ 
  -7/2 * Real.pi ≤ z ∧ z ≤ 3/2 * Real.pi ∧
  (∀ ε > 0, ∃ (x' y' : ℝ), x'^2 + y'^2 = 1 ∧
    (5 * Real.arcsin x' - 2 * Real.arccos y' < -7/2 * Real.pi + ε ∨
     5 * Real.arcsin x' - 2 * Real.arccos y' > 3/2 * Real.pi - ε)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l3960_396028


namespace NUMINAMATH_CALUDE_symmetry_sum_l3960_396011

/-- Two points are symmetric about the x-axis if they have the same x-coordinate and opposite y-coordinates -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetry_sum (x y : ℝ) :
  symmetric_about_x_axis (-2, y) (x, 3) → x + y = -5 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l3960_396011


namespace NUMINAMATH_CALUDE_first_traveler_constant_speed_second_traveler_constant_speed_l3960_396075

-- Define the speeds and distances
def speed1 : ℝ := 4
def speed2 : ℝ := 6
def total_distance : ℝ := 24

-- Define the constant speeds to be proven
def constant_speed1 : ℝ := 4.8
def constant_speed2 : ℝ := 5

-- Theorem for the first traveler
theorem first_traveler_constant_speed :
  let time1 := (total_distance / 2) / speed1
  let time2 := (total_distance / 2) / speed2
  let total_time := time1 + time2
  total_distance / total_time = constant_speed1 := by sorry

-- Theorem for the second traveler
theorem second_traveler_constant_speed :
  let total_time : ℝ := 2 -- Arbitrary total time
  let distance1 := speed1 * (total_time / 2)
  let distance2 := speed2 * (total_time / 2)
  let total_distance := distance1 + distance2
  total_distance / total_time = constant_speed2 := by sorry

end NUMINAMATH_CALUDE_first_traveler_constant_speed_second_traveler_constant_speed_l3960_396075


namespace NUMINAMATH_CALUDE_probability_both_truth_l3960_396025

theorem probability_both_truth (prob_A prob_B : ℝ) 
  (h_A : prob_A = 0.7) 
  (h_B : prob_B = 0.6) : 
  prob_A * prob_B = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_truth_l3960_396025


namespace NUMINAMATH_CALUDE_parabola_zeros_difference_l3960_396067

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The zeros of the parabola -/
def Parabola.zeros (p : Parabola) : Set ℝ :=
  {x : ℝ | p.y x = 0}

theorem parabola_zeros_difference (p : Parabola) :
  p.y 1 = -2 →  -- Vertex at (1, -2)
  p.y 3 = 10 →  -- Point (3, 10) on the parabola
  ∃ m n : ℝ,
    m ∈ p.zeros ∧
    n ∈ p.zeros ∧
    m > n ∧
    m - n = 2 * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_zeros_difference_l3960_396067


namespace NUMINAMATH_CALUDE_jake_peaches_count_l3960_396068

-- Define the variables
def steven_peaches : ℕ := 13
def steven_apples : ℕ := 52
def jake_apples : ℕ := steven_apples + 84

-- Define Jake's peaches in terms of Steven's
def jake_peaches : ℕ := steven_peaches - 10

-- Theorem to prove
theorem jake_peaches_count : jake_peaches = 3 := by
  sorry

end NUMINAMATH_CALUDE_jake_peaches_count_l3960_396068


namespace NUMINAMATH_CALUDE_correct_algorithm_statement_l3960_396058

-- Define the concept of an algorithm
def Algorithm : Type := Unit

-- Define the property of being correct for an algorithm
def is_correct (a : Algorithm) : Prop := sorry

-- Define the property of yielding a definite result
def yields_definite_result (a : Algorithm) : Prop := sorry

-- Define the property of ending within a finite number of steps
def ends_in_finite_steps (a : Algorithm) : Prop := sorry

-- Define the property of having clear and unambiguous steps
def has_clear_steps (a : Algorithm) : Prop := sorry

-- Define the property of being unique for solving a certain type of problem
def is_unique_for_problem (a : Algorithm) : Prop := sorry

-- Theorem stating that the only correct statement is (2)
theorem correct_algorithm_statement :
  ∀ (a : Algorithm),
    is_correct a →
    (yields_definite_result a ∧
    ¬(¬(has_clear_steps a)) ∧
    ¬(is_unique_for_problem a) ∧
    ¬(ends_in_finite_steps a)) :=
by sorry

end NUMINAMATH_CALUDE_correct_algorithm_statement_l3960_396058


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3960_396031

/-- The lateral surface area of a cone with base radius 3 and slant height 5 is 15π. -/
theorem cone_lateral_surface_area :
  let r : ℝ := 3  -- radius of the base
  let s : ℝ := 5  -- slant height
  let lateral_area := π * r * s  -- formula for lateral surface area of a cone
  lateral_area = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3960_396031
