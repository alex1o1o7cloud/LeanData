import Mathlib

namespace NUMINAMATH_CALUDE_smallest_cube_multiplier_l2530_253003

theorem smallest_cube_multiplier (n : ℕ) (h : n = 1512) :
  (∃ (y : ℕ), 49 * n = y^3) ∧
  (∀ (x : ℕ), x > 0 → x < 49 → ¬∃ (y : ℕ), x * n = y^3) :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_multiplier_l2530_253003


namespace NUMINAMATH_CALUDE_real_part_of_i_times_3_minus_i_l2530_253071

theorem real_part_of_i_times_3_minus_i : ∃ (z : ℂ), z = Complex.I * (3 - Complex.I) ∧ z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_i_times_3_minus_i_l2530_253071


namespace NUMINAMATH_CALUDE_gcd_problem_l2530_253002

theorem gcd_problem (h : Nat.Prime 101) :
  Nat.gcd (101^6 + 1) (3 * 101^6 + 101^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2530_253002


namespace NUMINAMATH_CALUDE_total_employees_calculation_l2530_253050

/-- Represents the number of employees in different categories and calculates the total full-time equivalents -/
def calculate_total_employees (part_time : ℕ) (full_time : ℕ) (remote : ℕ) (temporary : ℕ) : ℕ :=
  let hours_per_fte : ℕ := 40
  let total_hours : ℕ := part_time + full_time * hours_per_fte + remote * hours_per_fte + temporary * hours_per_fte
  (total_hours + hours_per_fte / 2) / hours_per_fte

/-- Theorem stating that given the specified number of employees in each category, 
    the total number of full-time equivalent employees is 76,971 -/
theorem total_employees_calculation :
  calculate_total_employees 2041 63093 5230 8597 = 76971 := by
  sorry

end NUMINAMATH_CALUDE_total_employees_calculation_l2530_253050


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_l2530_253030

theorem min_value_of_expression (a : ℝ) (h : a > 1) : a + 1 / (a - 1) ≥ 3 :=
sorry

theorem equality_condition (a : ℝ) (h : a > 1) : 
  a + 1 / (a - 1) = 3 ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_l2530_253030


namespace NUMINAMATH_CALUDE_profit_percentage_without_discount_l2530_253098

theorem profit_percentage_without_discount
  (cost_price : ℝ)
  (discount_rate : ℝ)
  (profit_rate_with_discount : ℝ)
  (h_positive_cost : cost_price > 0)
  (h_discount : discount_rate = 0.05)
  (h_profit_with_discount : profit_rate_with_discount = 0.1875) :
  let selling_price_with_discount := cost_price * (1 - discount_rate)
  let profit_amount := cost_price * profit_rate_with_discount
  let selling_price_without_discount := cost_price + profit_amount
  let profit_rate_without_discount := (selling_price_without_discount - cost_price) / cost_price
  profit_rate_without_discount = profit_rate_with_discount :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_without_discount_l2530_253098


namespace NUMINAMATH_CALUDE_two_numbers_sum_2014_l2530_253083

theorem two_numbers_sum_2014 : ∃ (x y : ℕ), x > y ∧ x + y = 2014 ∧ 3 * (x / 100) = y + 6 ∧ y = 51 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_2014_l2530_253083


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l2530_253086

theorem x_gt_one_sufficient_not_necessary_for_x_gt_zero :
  (∀ x : ℝ, x > 1 → x > 0) ∧
  (∃ x : ℝ, x > 0 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l2530_253086


namespace NUMINAMATH_CALUDE_b_days_proof_l2530_253008

/-- The number of days it takes A to complete the work alone -/
def a_days : ℝ := 6

/-- The total payment for the work -/
def total_payment : ℝ := 3600

/-- The number of days it takes A, B, and C to complete the work together -/
def abc_days : ℝ := 3

/-- The payment given to C -/
def c_payment : ℝ := 450

/-- The number of days it takes B to complete the work alone -/
def b_days : ℝ := 6

theorem b_days_proof :
  (1 / a_days + 1 / b_days) * abc_days = 1 ∧
  c_payment / total_payment = 1 - (1 / a_days + 1 / b_days) * abc_days :=
by sorry

end NUMINAMATH_CALUDE_b_days_proof_l2530_253008


namespace NUMINAMATH_CALUDE_reciprocal_of_one_fifth_l2530_253053

theorem reciprocal_of_one_fifth (x : ℚ) : 
  (x * (1 / x) = 1) → ((1 / (1 / 5)) = 5) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_one_fifth_l2530_253053


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2530_253029

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 12, height := 5 }

/-- Theorem: The maximum number of soap boxes that can theoretically fit in the carton is 150 -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 150 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2530_253029


namespace NUMINAMATH_CALUDE_larger_number_proof_l2530_253032

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 7 * S + 15) :
  L = 1590 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2530_253032


namespace NUMINAMATH_CALUDE_floor_abs_sum_l2530_253028

theorem floor_abs_sum : ⌊|(-5.7:ℝ)|⌋ + |⌊(-5.7:ℝ)⌋| = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l2530_253028


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_l2530_253007

/-- Calculates the final stock price after two years of growth -/
def final_stock_price (initial_price : ℝ) (first_year_growth : ℝ) (second_year_growth : ℝ) : ℝ :=
  initial_price * (1 + first_year_growth) * (1 + second_year_growth)

/-- Theorem stating that the stock price after two years of growth is $247.50 -/
theorem stock_price_after_two_years :
  final_stock_price 150 0.5 0.1 = 247.50 := by
  sorry

#eval final_stock_price 150 0.5 0.1

end NUMINAMATH_CALUDE_stock_price_after_two_years_l2530_253007


namespace NUMINAMATH_CALUDE_tax_revenue_change_l2530_253021

theorem tax_revenue_change (T C : ℝ) (T_new C_new R_new : ℝ) : 
  T_new = T * 0.9 →
  C_new = C * 1.1 →
  R_new = T_new * C_new →
  R_new = T * C * 0.99 := by
sorry

end NUMINAMATH_CALUDE_tax_revenue_change_l2530_253021


namespace NUMINAMATH_CALUDE_selectStudents_eq_30_l2530_253075

/-- The number of ways to select 3 students from 4 boys and 3 girls, ensuring both genders are represented -/
def selectStudents : ℕ :=
  Nat.choose 4 2 * Nat.choose 3 1 + Nat.choose 4 1 * Nat.choose 3 2

/-- Theorem stating that the number of selections is 30 -/
theorem selectStudents_eq_30 : selectStudents = 30 := by
  sorry

end NUMINAMATH_CALUDE_selectStudents_eq_30_l2530_253075


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2530_253018

theorem min_reciprocal_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 2) :
  (1 / a + 1 / b + 1 / c) ≥ 9 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2530_253018


namespace NUMINAMATH_CALUDE_equality_from_fraction_l2530_253039

theorem equality_from_fraction (a b : ℝ) (n : ℕ+) 
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_eq : ((a + b)^(n : ℕ) - (a - b)^(n : ℕ)) / ((a + b)^(n : ℕ) + (a - b)^(n : ℕ)) = a / b) :
  a = b := by sorry

end NUMINAMATH_CALUDE_equality_from_fraction_l2530_253039


namespace NUMINAMATH_CALUDE_min_value_problem_l2530_253087

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1/a + 1/b) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1/x + 1/y → 1/x + 2/y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l2530_253087


namespace NUMINAMATH_CALUDE_no_consecutive_sum_for_2004_l2530_253056

theorem no_consecutive_sum_for_2004 :
  ¬ ∃ (n : ℕ) (a : ℕ), n > 1 ∧ n * (2 * a + n - 1) = 4008 := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_sum_for_2004_l2530_253056


namespace NUMINAMATH_CALUDE_x_fourth_power_zero_l2530_253009

theorem x_fourth_power_zero (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (1 - x^2) + Real.sqrt (1 + x^2) = 2) : x^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_power_zero_l2530_253009


namespace NUMINAMATH_CALUDE_committee_selection_theorem_l2530_253005

/-- The number of candidates nominated for the committee -/
def total_candidates : ℕ := 20

/-- The number of candidates who have previously served on the committee -/
def past_members : ℕ := 9

/-- The number of positions available in the new committee -/
def committee_size : ℕ := 6

/-- The number of ways to select the committee with at least one past member -/
def selections_with_past_member : ℕ := 38298

theorem committee_selection_theorem :
  (Nat.choose total_candidates committee_size) - 
  (Nat.choose (total_candidates - past_members) committee_size) = 
  selections_with_past_member :=
sorry

end NUMINAMATH_CALUDE_committee_selection_theorem_l2530_253005


namespace NUMINAMATH_CALUDE_work_completion_time_l2530_253074

/-- The number of days y needs to finish the work -/
def y_days : ℝ := 15

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 10

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℝ := 6.000000000000001

/-- The number of days x needs to finish the entire work alone -/
def x_days : ℝ := 18

theorem work_completion_time :
  y_days = 15 ∧ y_worked = 10 ∧ x_remaining = 6.000000000000001 →
  x_days = 18 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2530_253074


namespace NUMINAMATH_CALUDE_luke_stickers_l2530_253081

theorem luke_stickers (initial bought gift given_away used remaining : ℕ) : 
  bought = 12 →
  gift = 20 →
  given_away = 5 →
  used = 8 →
  remaining = 39 →
  initial + bought + gift - given_away - used = remaining →
  initial = 20 := by
  sorry

end NUMINAMATH_CALUDE_luke_stickers_l2530_253081


namespace NUMINAMATH_CALUDE_two_shirts_per_package_l2530_253068

/-- Given a number of packages and a total number of t-shirts,
    calculate the number of t-shirts per package. -/
def tShirtsPerPackage (packages : ℕ) (totalShirts : ℕ) : ℚ :=
  totalShirts / packages

/-- Theorem stating that given 28 packages and 56 total t-shirts,
    the number of t-shirts per package is 2. -/
theorem two_shirts_per_package :
  tShirtsPerPackage 28 56 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_shirts_per_package_l2530_253068


namespace NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l2530_253058

/-- Given an arithmetic sequence where the first term is 3 and the 25th term is 51,
    prove that the 75th term is 151. -/
theorem arithmetic_sequence_75th_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 3 →                                -- first term is 3
    a 24 = 51 →                              -- 25th term is 51
    a 74 = 151 :=                            -- 75th term is 151
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l2530_253058


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l2530_253043

theorem sandy_shopping_money (remaining_money : ℝ) (spent_percentage : ℝ) (h1 : remaining_money = 224) (h2 : spent_percentage = 0.3) :
  let initial_money := remaining_money / (1 - spent_percentage)
  initial_money = 320 := by
sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l2530_253043


namespace NUMINAMATH_CALUDE_area_of_triangle_ABF_l2530_253010

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A square defined by four points -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Check if a point is inside a square -/
def isInside (p : Point) (s : Square) : Prop := sorry

/-- Find the intersection point of two line segments -/
def intersectionPoint (p1 p2 p3 p4 : Point) : Point := sorry

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

theorem area_of_triangle_ABF 
  (A B C D E F : Point)
  (square : Square)
  (triangle : Triangle)
  (h1 : square = Square.mk A B C D)
  (h2 : triangle = Triangle.mk A B E)
  (h3 : isEquilateral triangle)
  (h4 : isInside E square)
  (h5 : F = intersectionPoint B D A E)
  (h6 : (B.x - A.x)^2 + (B.y - A.y)^2 = 1 + Real.sqrt 3) :
  triangleArea (Triangle.mk A B F) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABF_l2530_253010


namespace NUMINAMATH_CALUDE_janelles_blue_marbles_gift_l2530_253064

/-- Calculates the number of blue marbles Janelle gave to her friend --/
def blue_marbles_given (initial_green : ℕ) (blue_bags : ℕ) (marbles_per_bag : ℕ) 
  (green_given : ℕ) (total_remaining : ℕ) : ℕ :=
  let total_blue := blue_bags * marbles_per_bag
  let total_before_gift := initial_green + total_blue
  let total_given := total_before_gift - total_remaining
  total_given - green_given

/-- Proves that Janelle gave 8 blue marbles to her friend --/
theorem janelles_blue_marbles_gift : 
  blue_marbles_given 26 6 10 6 72 = 8 := by
  sorry

end NUMINAMATH_CALUDE_janelles_blue_marbles_gift_l2530_253064


namespace NUMINAMATH_CALUDE_unique_divisible_number_l2530_253025

theorem unique_divisible_number : ∃! (x y z u v : ℕ),
  (x < 10 ∧ y < 10 ∧ z < 10 ∧ u < 10 ∧ v < 10) ∧
  (x * 10^9 + 6 * 10^8 + 1 * 10^7 + y * 10^6 + 0 * 10^5 + 6 * 10^4 + 4 * 10^3 + z * 10^2 + u * 10 + v) % 61875 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_number_l2530_253025


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2530_253024

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (5789 * 10 + N) % 6 = 0 → N ≤ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2530_253024


namespace NUMINAMATH_CALUDE_polynomial_coefficient_theorem_l2530_253062

theorem polynomial_coefficient_theorem (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, a₀ + a₁ * (2 * x - 1) + a₂ * (2 * x - 1)^2 + a₃ * (2 * x - 1)^3 + 
             a₄ * (2 * x - 1)^4 + a₅ * (2 * x - 1)^5 = x^5) →
  a₂ = 5/16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_theorem_l2530_253062


namespace NUMINAMATH_CALUDE_power_sum_product_equals_l2530_253036

theorem power_sum_product_equals : (6^3 + 4^2) * 7^5 = 3897624 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_product_equals_l2530_253036


namespace NUMINAMATH_CALUDE_new_average_age_with_teacher_l2530_253014

theorem new_average_age_with_teacher (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 30 →
  student_avg_age = 14 →
  teacher_age = 45 →
  ((num_students : ℝ) * student_avg_age + teacher_age) / ((num_students : ℝ) + 1) = 15 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_with_teacher_l2530_253014


namespace NUMINAMATH_CALUDE_derivative_symmetric_points_l2530_253093

/-- Given a function f(x) = ax^4 + bx^2 + c where f'(1) = 2, prove that f'(-1) = -2 -/
theorem derivative_symmetric_points 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^4 + b * x^2 + c)
  (h2 : deriv f 1 = 2) :
  deriv f (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_symmetric_points_l2530_253093


namespace NUMINAMATH_CALUDE_inequality_relations_l2530_253011

theorem inequality_relations (a b : ℝ) (h : a > b) : 
  (3 * a > 3 * b) ∧ (a + 2 > b + 2) ∧ (-5 * a < -5 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relations_l2530_253011


namespace NUMINAMATH_CALUDE_original_price_calculation_l2530_253015

/-- Proves that given an article sold for $25 with a gain percent of 150%, the original price of the article was $10. -/
theorem original_price_calculation (selling_price : ℝ) (gain_percent : ℝ) : 
  selling_price = 25 ∧ gain_percent = 150 → 
  ∃ (original_price : ℝ), 
    original_price = 10 ∧ 
    selling_price = original_price + (original_price * (gain_percent / 100)) :=
by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2530_253015


namespace NUMINAMATH_CALUDE_atomic_number_relationship_l2530_253073

/-- Given three elements with atomic numbers R, M, and Z, if their ions R^(X-), M^(n+), and Z^(m+) 
    have the same electronic structure, and n > m, then M > Z > R. -/
theorem atomic_number_relationship (R M Z n m x : ℤ) 
  (h1 : R + x = M - n) 
  (h2 : R + x = Z - m) 
  (h3 : n > m) : 
  M > Z ∧ Z > R := by sorry

end NUMINAMATH_CALUDE_atomic_number_relationship_l2530_253073


namespace NUMINAMATH_CALUDE_product_mod_25_l2530_253091

theorem product_mod_25 (n : ℕ) : 
  65 * 74 * 89 ≡ n [ZMOD 25] → 0 ≤ n ∧ n < 25 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_25_l2530_253091


namespace NUMINAMATH_CALUDE_inequality_proof_l2530_253040

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a * b * (b + 1) * (c + 1)) + 1 / (b * c * (c + 1) * (a + 1)) + 1 / (c * a * (a + 1) * (b + 1)) ≥ 3 / (1 + a * b * c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2530_253040


namespace NUMINAMATH_CALUDE_mean_proportional_proof_l2530_253044

theorem mean_proportional_proof :
  let a : ℝ := 7921
  let b : ℝ := 9481
  let m : ℝ := 8665
  m = (a * b).sqrt := by sorry

end NUMINAMATH_CALUDE_mean_proportional_proof_l2530_253044


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2530_253065

/-- Atomic weight of Barium in g/mol -/
def Ba_weight : ℝ := 137.33

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- Number of Barium atoms in the compound -/
def Ba_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 2

/-- Number of Hydrogen atoms in the compound -/
def H_count : ℕ := 2

/-- Calculates the molecular weight of the compound -/
def molecular_weight : ℝ := Ba_count * Ba_weight + O_count * O_weight + H_count * H_weight

/-- Theorem stating that the molecular weight of the compound is 171.35 g/mol -/
theorem compound_molecular_weight : molecular_weight = 171.35 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2530_253065


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l2530_253072

theorem polynomial_irreducibility (n : ℕ) (h : n > 1) :
  Irreducible (Polynomial.X ^ n + 5 * Polynomial.X ^ (n - 1) + 3 : Polynomial ℤ) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l2530_253072


namespace NUMINAMATH_CALUDE_toy_poodle_height_l2530_253076

/-- Heights of different poodle types -/
structure PoodleHeights where
  standard : ℝ
  miniature : ℝ
  toy : ℝ
  moyen : ℝ

/-- Conditions for poodle heights -/
def valid_poodle_heights (h : PoodleHeights) : Prop :=
  h.standard = h.miniature + 8.5 ∧
  h.miniature = h.toy + 6.25 ∧
  h.standard = h.moyen + 3.75 ∧
  h.moyen = h.toy + 4.75 ∧
  h.standard = 28

/-- Theorem: The toy poodle's height is 13.25 inches -/
theorem toy_poodle_height (h : PoodleHeights) 
  (hvalid : valid_poodle_heights h) : h.toy = 13.25 := by
  sorry

end NUMINAMATH_CALUDE_toy_poodle_height_l2530_253076


namespace NUMINAMATH_CALUDE_lines_intersect_at_same_point_l2530_253099

/-- Three lines intersect at the same point -/
theorem lines_intersect_at_same_point (m : ℝ) :
  ∃ (x y : ℝ), 
    (y = 3 * x + 5) ∧ 
    (y = -4 * x + m) ∧ 
    (y = 2 * x + (m + 30) / 7) := by
  sorry

end NUMINAMATH_CALUDE_lines_intersect_at_same_point_l2530_253099


namespace NUMINAMATH_CALUDE_max_profit_at_max_price_l2530_253084

/-- Represents the relationship between price and sales --/
def sales_function (x : ℝ) : ℝ := -3 * x + 240

/-- Represents the profit function --/
def profit_function (x : ℝ) : ℝ := -3 * x^2 + 360 * x - 9600

/-- The cost price of apples --/
def cost_price : ℝ := 40

/-- The maximum allowed selling price --/
def max_price : ℝ := 55

/-- Theorem stating that the maximum profit is achieved at the maximum allowed price --/
theorem max_profit_at_max_price : 
  ∀ x, x ≥ cost_price → x ≤ max_price → profit_function x ≤ profit_function max_price :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_max_price_l2530_253084


namespace NUMINAMATH_CALUDE_shortest_ribbon_length_l2530_253094

theorem shortest_ribbon_length (a b c d : ℕ) (ha : a = 2) (hb : b = 5) (hc : c = 7) (hd : d = 11) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 770 :=
by sorry

end NUMINAMATH_CALUDE_shortest_ribbon_length_l2530_253094


namespace NUMINAMATH_CALUDE_batsman_average_is_60_l2530_253057

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  total_innings : ℕ
  highest_score : ℕ
  lowest_score : ℕ
  average_excluding_extremes : ℚ

/-- Calculate the batting average given the batsman's statistics -/
def batting_average (stats : BatsmanStats) : ℚ :=
  let total_runs := (stats.total_innings - 2) * stats.average_excluding_extremes + stats.highest_score + stats.lowest_score
  total_runs / stats.total_innings

theorem batsman_average_is_60 (stats : BatsmanStats) :
  stats.total_innings = 46 ∧
  stats.highest_score - stats.lowest_score = 140 ∧
  stats.average_excluding_extremes = 58 ∧
  stats.highest_score = 174 →
  batting_average stats = 60 := by
  sorry

#eval batting_average {
  total_innings := 46,
  highest_score := 174,
  lowest_score := 34,
  average_excluding_extremes := 58
}

end NUMINAMATH_CALUDE_batsman_average_is_60_l2530_253057


namespace NUMINAMATH_CALUDE_count_acute_triangles_l2530_253054

/-- A triangle classification based on its angles -/
inductive TriangleType
  | Acute   : TriangleType
  | Right   : TriangleType
  | Obtuse  : TriangleType

/-- Represents a set of triangles -/
structure TriangleSet where
  total : Nat
  right : Nat
  obtuse : Nat

/-- Theorem: Given 7 triangles with 2 right angles and 3 obtuse angles, there are 2 acute triangles -/
theorem count_acute_triangles (ts : TriangleSet) :
  ts.total = 7 ∧ ts.right = 2 ∧ ts.obtuse = 3 →
  ts.total - ts.right - ts.obtuse = 2 := by
  sorry

#check count_acute_triangles

end NUMINAMATH_CALUDE_count_acute_triangles_l2530_253054


namespace NUMINAMATH_CALUDE_sum_of_simplified_fraction_75_135_l2530_253042

def simplify_fraction (n d : ℕ) : ℕ × ℕ :=
  let g := Nat.gcd n d
  (n / g, d / g)

theorem sum_of_simplified_fraction_75_135 :
  let (n, d) := simplify_fraction 75 135
  n + d = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_simplified_fraction_75_135_l2530_253042


namespace NUMINAMATH_CALUDE_intersection_theorem_l2530_253041

/-- The line y = x + m intersects the circle x^2 + y^2 - 2x + 4y - 4 = 0 at two distinct points A and B. -/
def intersects_at_two_points (m : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧
    (A.1^2 + A.2^2 - 2*A.1 + 4*A.2 - 4 = 0) ∧
    (B.1^2 + B.2^2 - 2*B.1 + 4*B.2 - 4 = 0) ∧
    (A.2 = A.1 + m) ∧ (B.2 = B.1 + m)

/-- The circle with diameter AB passes through the origin. -/
def circle_passes_origin (A B : ℝ × ℝ) : Prop :=
  (A.1 * B.1 + A.2 * B.2 = 0)

/-- Main theorem about the intersection of the line and circle. -/
theorem intersection_theorem :
  (∀ m : ℝ, intersects_at_two_points m ↔ -3-3*Real.sqrt 2 < m ∧ m < -3+3*Real.sqrt 2) ∧
  (∀ m : ℝ, intersects_at_two_points m →
    (∃ A B : ℝ × ℝ, A ≠ B ∧ circle_passes_origin A B) →
    (m = -4 ∨ m = 1)) :=
sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2530_253041


namespace NUMINAMATH_CALUDE_games_from_friend_l2530_253066

theorem games_from_friend (games_from_garage_sale : ℕ) 
  (non_working_games : ℕ) (good_games : ℕ) : ℕ :=
  by
  have h1 : games_from_garage_sale = 8 := by sorry
  have h2 : non_working_games = 23 := by sorry
  have h3 : good_games = 6 := by sorry
  
  let total_games := non_working_games + good_games
  
  have h4 : total_games = 29 := by sorry
  
  let games_from_friend := total_games - games_from_garage_sale
  
  have h5 : games_from_friend = 21 := by sorry
  
  exact games_from_friend

end NUMINAMATH_CALUDE_games_from_friend_l2530_253066


namespace NUMINAMATH_CALUDE_modulus_of_squared_complex_l2530_253085

theorem modulus_of_squared_complex (z : ℂ) (h : z^2 = 15 - 20*I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_squared_complex_l2530_253085


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2530_253048

theorem solution_set_inequality (x : ℝ) :
  x * (x - 1) > 0 ↔ x < 0 ∨ x > 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2530_253048


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l2530_253060

/-- Given a triangle with angles 40°, 3x, and x + 10°, prove that x = 32.5° --/
theorem triangle_angle_problem (x : ℝ) : 
  (40 : ℝ) + 3 * x + (x + 10) = 180 → x = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l2530_253060


namespace NUMINAMATH_CALUDE_unique_sums_count_l2530_253020

def bag_A : Finset ℕ := {1, 4, 9}
def bag_B : Finset ℕ := {16, 25, 36}

def possible_sums : Finset ℕ := (bag_A.product bag_B).image (fun p => p.1 + p.2)

theorem unique_sums_count : possible_sums.card = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_sums_count_l2530_253020


namespace NUMINAMATH_CALUDE_turtleneck_profit_l2530_253022

/-- Represents the pricing strategy and profit calculation for turtleneck sweaters -/
theorem turtleneck_profit (C : ℝ) (C_pos : C > 0) : 
  let initial_markup : ℝ := 0.20
  let new_year_markup : ℝ := 0.25
  let february_discount : ℝ := 0.09
  let SP1 : ℝ := C * (1 + initial_markup)
  let SP2 : ℝ := SP1 * (1 + new_year_markup)
  let SPF : ℝ := SP2 * (1 - february_discount)
  let profit : ℝ := SPF - C
  profit / C = 0.365 := by sorry

end NUMINAMATH_CALUDE_turtleneck_profit_l2530_253022


namespace NUMINAMATH_CALUDE_base_10_to_12_256_l2530_253047

/-- Converts a base-10 number to its base-12 representation -/
def toBase12 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base-12 to a natural number -/
def fromBase12 (digits : List ℕ) : ℕ :=
  sorry

theorem base_10_to_12_256 :
  toBase12 256 = [1, 9, 4] ∧ fromBase12 [1, 9, 4] = 256 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_12_256_l2530_253047


namespace NUMINAMATH_CALUDE_fraction_addition_l2530_253027

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2530_253027


namespace NUMINAMATH_CALUDE_min_max_z_values_l2530_253088

theorem min_max_z_values (x y z : ℝ) 
  (h1 : x^2 ≤ y + z) 
  (h2 : y^2 ≤ z + x) 
  (h3 : z^2 ≤ x + y) : 
  (-1/4 : ℝ) ≤ z ∧ z ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_max_z_values_l2530_253088


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2530_253006

theorem necessary_but_not_sufficient (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 1 → x^2 + y^2 ≥ 2) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 2 ∧ ¬(x ≥ 1 ∧ y ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2530_253006


namespace NUMINAMATH_CALUDE_trivia_game_points_per_round_l2530_253017

theorem trivia_game_points_per_round 
  (total_points : ℕ) 
  (num_rounds : ℕ) 
  (h1 : total_points = 78) 
  (h2 : num_rounds = 26) : 
  total_points / num_rounds = 3 := by
sorry

end NUMINAMATH_CALUDE_trivia_game_points_per_round_l2530_253017


namespace NUMINAMATH_CALUDE_empty_set_proof_l2530_253082

theorem empty_set_proof : {x : ℝ | x^2 - x + 1 = 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_empty_set_proof_l2530_253082


namespace NUMINAMATH_CALUDE_jellybean_box_capacity_l2530_253013

theorem jellybean_box_capacity (tim_capacity : ℕ) (scale_factor : ℕ) : 
  tim_capacity = 150 → scale_factor = 3 → 
  (scale_factor ^ 3 : ℕ) * tim_capacity = 4050 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_box_capacity_l2530_253013


namespace NUMINAMATH_CALUDE_stating_five_min_commercials_count_l2530_253037

/-- Represents the duration of the commercial break in minutes -/
def total_time : ℕ := 37

/-- Represents the number of 2-minute commercials -/
def two_min_commercials : ℕ := 11

/-- Represents the duration of a short commercial in minutes -/
def short_commercial_duration : ℕ := 2

/-- Represents the duration of a long commercial in minutes -/
def long_commercial_duration : ℕ := 5

/-- 
Theorem stating that given the total time and number of 2-minute commercials,
the number of 5-minute commercials is 3
-/
theorem five_min_commercials_count : 
  ∃ (x : ℕ), x * long_commercial_duration + two_min_commercials * short_commercial_duration = total_time ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_stating_five_min_commercials_count_l2530_253037


namespace NUMINAMATH_CALUDE_smallest_num_rectangles_l2530_253061

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- Whether two natural numbers are in the ratio 5:4 -/
def in_ratio_5_4 (a b : ℕ) : Prop := 5 * b = 4 * a

/-- The number of small rectangles needed to cover a larger rectangle -/
def num_small_rectangles (small large : Rectangle) : ℕ :=
  large.area / small.area

theorem smallest_num_rectangles :
  let small_rectangle : Rectangle := ⟨2, 3⟩
  ∃ (large_rectangle : Rectangle),
    in_ratio_5_4 large_rectangle.width large_rectangle.height ∧
    num_small_rectangles small_rectangle large_rectangle = 30 ∧
    ∀ (other_rectangle : Rectangle),
      in_ratio_5_4 other_rectangle.width other_rectangle.height →
      num_small_rectangles small_rectangle other_rectangle ≥ 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_num_rectangles_l2530_253061


namespace NUMINAMATH_CALUDE_scores_relative_to_average_l2530_253034

def scores : List ℤ := [95, 86, 90, 87, 92]
def average : ℚ := 90

theorem scores_relative_to_average :
  let relative_scores := scores.map (λ s => s - average)
  relative_scores = [5, -4, 0, -3, 2] := by
  sorry

end NUMINAMATH_CALUDE_scores_relative_to_average_l2530_253034


namespace NUMINAMATH_CALUDE_function_transformation_l2530_253038

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_transformation (h : f 4 = 2) : 
  ∃ x y, x = 4 ∧ y = -2 ∧ -f x = y :=
sorry

end NUMINAMATH_CALUDE_function_transformation_l2530_253038


namespace NUMINAMATH_CALUDE_boys_camp_problem_l2530_253069

theorem boys_camp_problem (total : ℕ) 
  (h1 : (total : ℚ) * (1/5) = (total : ℚ) * (20/100))
  (h2 : (total : ℚ) * (1/5) * (3/10) = (total : ℚ) * (1/5) * (30/100))
  (h3 : (total : ℚ) * (1/5) * (7/10) = 35) :
  total = 250 := by sorry

end NUMINAMATH_CALUDE_boys_camp_problem_l2530_253069


namespace NUMINAMATH_CALUDE_coin_drawing_probability_l2530_253052

/-- The number of shiny coins in the box -/
def shiny_coins : ℕ := 3

/-- The number of dull coins in the box -/
def dull_coins : ℕ := 4

/-- The total number of coins in the box -/
def total_coins : ℕ := shiny_coins + dull_coins

/-- The probability of needing more than 4 draws to select all shiny coins -/
def prob_more_than_four_draws : ℚ := 31 / 35

theorem coin_drawing_probability :
  let p := 1 - (Nat.choose shiny_coins shiny_coins * 
    (Nat.choose dull_coins 1 * Nat.choose shiny_coins shiny_coins + 
    Nat.choose (total_coins - 1) 3)) / Nat.choose total_coins 4
  p = prob_more_than_four_draws := by sorry

end NUMINAMATH_CALUDE_coin_drawing_probability_l2530_253052


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l2530_253004

/-- Given ratios between w, x, y, and z, prove the ratio of w to y -/
theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 2)
  (hy : y / w = 3 / 5)
  (hz : z / x = 1 / 3) :
  w / y = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l2530_253004


namespace NUMINAMATH_CALUDE_cube_volume_l2530_253059

/-- Given a box with dimensions 8 cm x 15 cm x 5 cm that can be built using a minimum of 60 cubes,
    the volume of each cube is 10 cm³. -/
theorem cube_volume (length width height min_cubes : ℕ) : 
  length = 8 → width = 15 → height = 5 → min_cubes = 60 →
  (length * width * height : ℚ) / min_cubes = 10 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l2530_253059


namespace NUMINAMATH_CALUDE_range_of_f_range_of_a_l2530_253079

-- Define the functions
def f (x : ℝ) : ℝ := 2 * abs (x - 1) - abs (x - 4)

def g (x a : ℝ) : ℝ := 2 * abs (x - 1) - abs (x - a)

-- State the theorems
theorem range_of_f : Set.range f = Set.Ici (-3) := by sorry

theorem range_of_a (h : ∀ x : ℝ, g x a ≥ -1) : a ∈ Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_a_l2530_253079


namespace NUMINAMATH_CALUDE_fallen_sheets_l2530_253049

def is_permutation (a b : Nat) : Prop :=
  (a.digits 10).sum = (b.digits 10).sum ∧
  (a.digits 10).prod = (b.digits 10).prod

theorem fallen_sheets (n : Nat) 
  (h1 : is_permutation 387 n)
  (h2 : n > 387)
  (h3 : Even n) :
  (n - 387 + 1) / 2 = 176 :=
sorry

end NUMINAMATH_CALUDE_fallen_sheets_l2530_253049


namespace NUMINAMATH_CALUDE_clique_six_and_best_degree_l2530_253095

/-- A graph with 1991 points where every point has degree at least 1593 -/
structure Graph1991 where
  vertices : Finset (Fin 1991)
  edges : Finset (Fin 1991 × Fin 1991)
  degree_condition : ∀ v : Fin 1991, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 1593

/-- A clique is a subset of vertices where every two distinct vertices are adjacent -/
def is_clique (G : Graph1991) (S : Finset (Fin 1991)) : Prop :=
  ∀ u v : Fin 1991, u ∈ S → v ∈ S → u ≠ v → (u, v) ∈ G.edges ∨ (v, u) ∈ G.edges

/-- The main theorem stating that there exists a clique of size 6 and 1593 is the best possible -/
theorem clique_six_and_best_degree (G : Graph1991) :
  (∃ S : Finset (Fin 1991), S.card = 6 ∧ is_clique G S) ∧
  ∀ d < 1593, ∃ H : Graph1991, ¬∃ S : Finset (Fin 1991), S.card = 6 ∧ is_clique H S :=
sorry

end NUMINAMATH_CALUDE_clique_six_and_best_degree_l2530_253095


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_equals_6_l2530_253089

def vector_a (m : ℝ) : ℝ × ℝ := (2, m)
def vector_b : ℝ × ℝ := (1, -1)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def vector_add (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2)

def vector_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

theorem perpendicular_vectors_m_equals_6 :
  ∀ m : ℝ, 
    let a := vector_a m
    let b := vector_b
    let sum := vector_add a (vector_scale 2 b)
    dot_product b sum = 0 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_equals_6_l2530_253089


namespace NUMINAMATH_CALUDE_count_to_200_words_l2530_253067

/-- Represents the number of words used to express a given number in English. --/
def wordsForNumber (n : ℕ) : ℕ :=
  if n ≤ 20 ∨ n = 30 ∨ n = 40 ∨ n = 50 ∨ n = 60 ∨ n = 70 ∨ n = 80 ∨ n = 90 ∨ n = 100 ∨ n = 200 then 1
  else if n ≤ 99 then 2
  else if n ≤ 199 then 3
  else 3

/-- The total number of words used to count from 1 to 200 in English. --/
def totalWordsUpTo200 : ℕ := (Finset.range 200).sum wordsForNumber + wordsForNumber 200

/-- Theorem stating that the total number of words used to count from 1 to 200 in English is 443. --/
theorem count_to_200_words : totalWordsUpTo200 = 443 := by
  sorry

end NUMINAMATH_CALUDE_count_to_200_words_l2530_253067


namespace NUMINAMATH_CALUDE_power_product_rule_l2530_253031

theorem power_product_rule (a : ℝ) : a^2 * a^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l2530_253031


namespace NUMINAMATH_CALUDE_sum_of_ages_in_five_years_l2530_253033

/-- Given that Linda's current age is 13 and she is 3 more than 2 times Jane's age,
    prove that the sum of their ages in five years will be 28. -/
theorem sum_of_ages_in_five_years (jane_age : ℕ) (linda_age : ℕ) : 
  linda_age = 13 → linda_age = 2 * jane_age + 3 → 
  linda_age + 5 + (jane_age + 5) = 28 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_in_five_years_l2530_253033


namespace NUMINAMATH_CALUDE_fifteenth_base_five_number_l2530_253092

/-- Represents a number in base 5 --/
def BaseFive : Type := Nat

/-- Converts a natural number to its base 5 representation --/
def toBaseFive (n : Nat) : BaseFive :=
  sorry

/-- The sequence of numbers in base 5 --/
def baseFiveSequence : Nat → BaseFive :=
  sorry

theorem fifteenth_base_five_number :
  baseFiveSequence 15 = toBaseFive 30 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_base_five_number_l2530_253092


namespace NUMINAMATH_CALUDE_infinite_symmetry_centers_l2530_253045

/-- A point in a 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A figure in a 2D space -/
structure Figure :=
  (points : Set Point)

/-- A symmetry transformation with respect to a center point -/
def symmetryTransform (center : Point) (p : Point) : Point :=
  { x := 2 * center.x - p.x,
    y := 2 * center.y - p.y }

/-- A center of symmetry for a figure -/
def isSymmetryCenter (f : Figure) (c : Point) : Prop :=
  ∀ p ∈ f.points, symmetryTransform c p ∈ f.points

/-- The set of all symmetry centers for a figure -/
def symmetryCenters (f : Figure) : Set Point :=
  { c | isSymmetryCenter f c }

/-- Main theorem: If a figure has more than one center of symmetry, 
    it must have infinitely many centers of symmetry -/
theorem infinite_symmetry_centers (f : Figure) :
  (∃ c₁ c₂ : Point, c₁ ≠ c₂ ∧ c₁ ∈ symmetryCenters f ∧ c₂ ∈ symmetryCenters f) →
  ¬ Finite (symmetryCenters f) :=
sorry

end NUMINAMATH_CALUDE_infinite_symmetry_centers_l2530_253045


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_half_l2530_253051

open Real

theorem trigonometric_expression_equals_half : 
  (cos (85 * π / 180) + sin (25 * π / 180) * cos (30 * π / 180)) / cos (25 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_half_l2530_253051


namespace NUMINAMATH_CALUDE_total_packs_eq_243_l2530_253012

/-- The total number of packs sold in six villages -/
def total_packs : ℕ := 23 + 28 + 35 + 43 + 50 + 64

/-- Theorem stating that the total number of packs sold equals 243 -/
theorem total_packs_eq_243 : total_packs = 243 := by
  sorry

end NUMINAMATH_CALUDE_total_packs_eq_243_l2530_253012


namespace NUMINAMATH_CALUDE_remainder_theorem_l2530_253096

theorem remainder_theorem (k : ℤ) : (1125 * 1127 * (12 * k + 1)) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2530_253096


namespace NUMINAMATH_CALUDE_fraction_problem_l2530_253080

theorem fraction_problem (N : ℝ) (f : ℝ) : 
  N = 180 →
  6 + (1/2) * (1/3) * f * N = (1/15) * N →
  f = 1/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l2530_253080


namespace NUMINAMATH_CALUDE_ratio_equality_product_l2530_253078

theorem ratio_equality_product (x : ℝ) :
  (2 * x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) →
  ∃ y : ℝ, (2 * y + 3) / (3 * y + 3) = (5 * y + 4) / (8 * y + 4) ∧ y ≠ x ∧ x * y = 0 :=
by sorry

end NUMINAMATH_CALUDE_ratio_equality_product_l2530_253078


namespace NUMINAMATH_CALUDE_pizza_combinations_l2530_253023

theorem pizza_combinations : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l2530_253023


namespace NUMINAMATH_CALUDE_chromium_percentage_in_cast_iron_l2530_253019

theorem chromium_percentage_in_cast_iron 
  (x y : ℝ) 
  (h1 : 5 * x + y = 6 * min x y) 
  (h2 : x + y = 0.16) : 
  (x = 0.11 ∧ y = 0.05) ∨ (x = 0.05 ∧ y = 0.11) :=
sorry

end NUMINAMATH_CALUDE_chromium_percentage_in_cast_iron_l2530_253019


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_384_l2530_253035

/-- A function that counts the number of 4-digit numbers beginning with 2 
    and having exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let non_two_digits := digits \ {2}
  let count_with_two_twos := 3 * (Finset.card non_two_digits - 1) * (Finset.card non_two_digits - 1)
  let count_with_non_two_pairs := 3 * (Finset.card non_two_digits) * (Finset.card non_two_digits - 1)
  count_with_two_twos + count_with_non_two_pairs

theorem count_special_numbers_eq_384 : count_special_numbers = 384 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_384_l2530_253035


namespace NUMINAMATH_CALUDE_initial_doctors_count_l2530_253046

theorem initial_doctors_count (initial_nurses : ℕ) (remaining_staff : ℕ) : initial_nurses = 18 → remaining_staff = 22 → ∃ initial_doctors : ℕ, initial_doctors = 11 ∧ initial_doctors + initial_nurses - 5 - 2 = remaining_staff :=
by
  sorry

end NUMINAMATH_CALUDE_initial_doctors_count_l2530_253046


namespace NUMINAMATH_CALUDE_max_value_when_a_zero_one_zero_iff_a_positive_l2530_253000

noncomputable section

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Theorem for part 1
theorem max_value_when_a_zero :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 0 y ≤ f 0 x ∧ f 0 x = -1 :=
sorry

-- Theorem for part 2
theorem one_zero_iff_a_positive :
  ∀ (a : ℝ), (∃! (x : ℝ), x > 0 ∧ f a x = 0) ↔ a > 0 :=
sorry

end

end NUMINAMATH_CALUDE_max_value_when_a_zero_one_zero_iff_a_positive_l2530_253000


namespace NUMINAMATH_CALUDE_yellow_ball_count_l2530_253063

theorem yellow_ball_count (red_count : ℕ) (total_count : ℕ) 
  (h1 : red_count = 10)
  (h2 : (red_count : ℚ) / total_count = 1 / 3) :
  total_count - red_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_count_l2530_253063


namespace NUMINAMATH_CALUDE_survey_analysis_l2530_253016

/-- Data from the survey --/
structure SurveyData where
  a : Nat  -- Females who understand
  b : Nat  -- Females who do not understand
  c : Nat  -- Males who understand
  d : Nat  -- Males who do not understand

/-- Chi-square calculation function --/
def chiSquare (data : SurveyData) : Rat :=
  let n := data.a + data.b + data.c + data.d
  n * (data.a * data.d - data.b * data.c)^2 / 
    ((data.a + data.b) * (data.c + data.d) * (data.a + data.c) * (data.b + data.d))

/-- Binomial probability calculation function --/
def binomialProb (n k : Nat) (p : Rat) : Rat :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- Main theorem --/
theorem survey_analysis (data : SurveyData) 
    (h_data : data.a = 140 ∧ data.b = 60 ∧ data.c = 180 ∧ data.d = 20) :
    chiSquare data = 25 ∧ 
    chiSquare data > (10828 : Rat) / 1000 ∧
    binomialProb 5 3 (4/5) = 128/625 := by
  sorry

#eval chiSquare ⟨140, 60, 180, 20⟩
#eval binomialProb 5 3 (4/5)

end NUMINAMATH_CALUDE_survey_analysis_l2530_253016


namespace NUMINAMATH_CALUDE_min_distance_to_parabola_l2530_253077

/-- Rectilinear distance between two points -/
def rectilinear_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

/-- A point on the parabola x² = y -/
def parabola_point (t : ℝ) : ℝ × ℝ := (t, t^2)

/-- The fixed point M(-1, 0) -/
def M : ℝ × ℝ := (-1, 0)

/-- Theorem: The minimum rectilinear distance from M(-1, 0) to the parabola x² = y is 3/4 -/
theorem min_distance_to_parabola :
  ∀ t : ℝ, rectilinear_distance (M.1) (M.2) (parabola_point t).1 (parabola_point t).2 ≥ 3/4 ∧
  ∃ t₀ : ℝ, rectilinear_distance (M.1) (M.2) (parabola_point t₀).1 (parabola_point t₀).2 = 3/4 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_parabola_l2530_253077


namespace NUMINAMATH_CALUDE_min_difference_ln2_l2530_253055

/-- Given functions f and g, prove that the minimum value of x₁ - x₂ is ln(2) when f(x₁) = g(x₂) -/
theorem min_difference_ln2 (f g : ℝ → ℝ) (x₁ x₂ : ℝ) 
  (hf : f = fun x ↦ Real.log (x / 2) + 1 / 2)
  (hg : g = fun x ↦ Real.exp (x - 2))
  (hx₁ : x₁ > 0)
  (hequal : f x₁ = g x₂) :
  ∃ (min : ℝ), min = Real.log 2 ∧ ∀ y₁ y₂, f y₁ = g y₂ → y₁ - y₂ ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_difference_ln2_l2530_253055


namespace NUMINAMATH_CALUDE_not_expressible_as_difference_of_squares_l2530_253026

theorem not_expressible_as_difference_of_squares (k x y : ℤ) : 
  ¬ (∃ n : ℤ, (n = 8*k + 3 ∨ n = 8*k + 5) ∧ n = x^2 - 2*y^2) :=
sorry

end NUMINAMATH_CALUDE_not_expressible_as_difference_of_squares_l2530_253026


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2530_253070

theorem simplify_fraction_product : (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2530_253070


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l2530_253001

theorem divisibility_by_eleven (n : ℤ) : 
  11 ∣ (n^2001 - n^4) ↔ n % 11 = 0 ∨ n % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l2530_253001


namespace NUMINAMATH_CALUDE_matrix_identity_sum_l2530_253090

/-- Given a matrix N with the specified structure, prove that if N^T N = I, then x^2 + y^2 + z^2 = 47/120 -/
theorem matrix_identity_sum (x y z : ℝ) : 
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![0, 3*y, 2*z; 2*x, y, -z; 2*x, -y, z]
  (N.transpose * N = 1) → x^2 + y^2 + z^2 = 47/120 := by
  sorry

end NUMINAMATH_CALUDE_matrix_identity_sum_l2530_253090


namespace NUMINAMATH_CALUDE_rectangle_height_twice_square_side_l2530_253097

/-- Given a square with side length s and a rectangle with base s and area twice that of the square,
    prove that the height of the rectangle is 2s. -/
theorem rectangle_height_twice_square_side (s : ℝ) (h : s > 0) : 
  let square_area := s^2
  let rectangle_base := s
  let rectangle_area := 2 * square_area
  rectangle_area / rectangle_base = 2 * s := by
  sorry

end NUMINAMATH_CALUDE_rectangle_height_twice_square_side_l2530_253097
