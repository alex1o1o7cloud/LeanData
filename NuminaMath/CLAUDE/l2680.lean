import Mathlib

namespace NUMINAMATH_CALUDE_first_recipe_cups_l2680_268072

/-- Represents the amount of soy sauce in various units --/
structure SoySauce where
  bottles : ℕ
  ounces : ℕ
  cups : ℕ

/-- Conversion factors and recipe requirements --/
def bottleSize : ℕ := 16 -- ounces per bottle
def ouncesPerCup : ℕ := 8
def recipe2Cups : ℕ := 1
def recipe3Cups : ℕ := 3
def totalBottles : ℕ := 3

/-- The main theorem to prove --/
theorem first_recipe_cups (sauce : SoySauce) : 
  sauce.bottles = totalBottles → 
  sauce.ounces = sauce.bottles * bottleSize → 
  sauce.cups = sauce.ounces / ouncesPerCup →
  sauce.cups = recipe2Cups + recipe3Cups + 2 :=
by sorry

end NUMINAMATH_CALUDE_first_recipe_cups_l2680_268072


namespace NUMINAMATH_CALUDE_piggy_bank_sequence_l2680_268027

theorem piggy_bank_sequence (sequence : Fin 6 → ℕ) 
  (h1 : sequence 0 = 72)
  (h2 : sequence 1 = 81)
  (h4 : sequence 3 = 99)
  (h5 : sequence 4 = 108)
  (h6 : sequence 5 = 117)
  (h_arithmetic : ∀ i : Fin 5, sequence (i + 1) - sequence i = sequence 1 - sequence 0) :
  sequence 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_sequence_l2680_268027


namespace NUMINAMATH_CALUDE_solution_pair_l2680_268037

theorem solution_pair : ∃! (x y : ℝ), 
  (x + y = (7 - x) + (7 - y)) ∧ 
  (x - y = (x - 2) + (y - 2)) ∧ 
  x = 3 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_solution_pair_l2680_268037


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2680_268091

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℤ, (x - 2) * (x + 1) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x + 4)) → 
  p = 5 ∧ q = -4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2680_268091


namespace NUMINAMATH_CALUDE_waiter_customers_l2680_268067

/-- Given a waiter with 9 tables, each having 7 women and 3 men, prove that the total number of customers is 90. -/
theorem waiter_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) 
  (h1 : num_tables = 9) 
  (h2 : women_per_table = 7) 
  (h3 : men_per_table = 3) : 
  num_tables * (women_per_table + men_per_table) = 90 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l2680_268067


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l2680_268094

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l2680_268094


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2680_268061

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b = a(cosC + (√3/3)sinC), a = √3, and c = 1, then C = π/6 -/
theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) : 
  b = a * (Real.cos C + (Real.sqrt 3 / 3) * Real.sin C) →
  a = Real.sqrt 3 →
  c = 1 →
  C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2680_268061


namespace NUMINAMATH_CALUDE_katy_summer_reading_l2680_268075

/-- The number of books Katy read during the summer -/
def summer_reading (june july august : ℕ) : ℕ := june + july + august

theorem katy_summer_reading :
  ∀ (june july august : ℕ),
  june = 8 →
  july = 2 * june →
  august = july - 3 →
  summer_reading june july august = 37 :=
by
  sorry

end NUMINAMATH_CALUDE_katy_summer_reading_l2680_268075


namespace NUMINAMATH_CALUDE_intersection_and_lines_l2680_268086

-- Define the lines
def l₁ (x y : ℝ) : Prop := x + y - 2 = 0
def l₂ (x y : ℝ) : Prop := x - 3*y + 10 = 0
def l₃ (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Define the intersection point
def M : ℝ × ℝ := (-1, 3)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 3*x - 4*y + 15 = 0

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 4*x + 3*y - 5 = 0

theorem intersection_and_lines :
  (∀ x y, l₁ x y ∧ l₂ x y ↔ (x, y) = M) ∧
  (parallel_line M.1 M.2 ∧ ∀ x y, parallel_line x y → l₃ x y → x = y) ∧
  (perpendicular_line M.1 M.2 ∧ ∀ x y, perpendicular_line x y → l₃ x y → 
    (x - M.1) * 3 + (y - M.2) * (-4) = 0) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_lines_l2680_268086


namespace NUMINAMATH_CALUDE_trigonometric_product_square_root_l2680_268062

theorem trigonometric_product_square_root : 
  let f (x : ℝ) := 512 * x^3 - 1152 * x^2 + 576 * x - 27
  (f (Real.sin (π / 9)^2) = 0) ∧ 
  (f (Real.sin (2 * π / 9)^2) = 0) ∧ 
  (f (Real.sin (4 * π / 9)^2) = 0) →
  Real.sqrt ((3 - Real.sin (π / 9)^2) * (3 - Real.sin (2 * π / 9)^2) * (3 - Real.sin (4 * π / 9)^2)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_square_root_l2680_268062


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l2680_268083

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 2^p) 
  (hy : y = 1 + 2^(-p)) : 
  y = x / (x - 1) := by
sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l2680_268083


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2680_268019

/-- A geometric sequence with a_3 = 4 and a_6 = 1/2 has a common ratio of 1/2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence property
  a 3 = 4 →                         -- given condition
  a 6 = 1 / 2 →                     -- given condition
  q = 1 / 2 :=                      -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2680_268019


namespace NUMINAMATH_CALUDE_savings_after_expense_l2680_268098

def weekly_savings (n : ℕ) : ℕ := 20 + 10 * n

def total_savings (weeks : ℕ) : ℕ :=
  (List.range weeks).map weekly_savings |>.sum

theorem savings_after_expense (weeks : ℕ) (expense : ℕ) : 
  weeks = 4 → expense = 75 → total_savings weeks - expense = 65 := by
  sorry

end NUMINAMATH_CALUDE_savings_after_expense_l2680_268098


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_sets_l2680_268059

theorem common_number_in_overlapping_sets (numbers : List ℝ) : 
  numbers.length = 9 ∧ 
  (numbers.take 5).sum / 5 = 7 ∧ 
  (numbers.drop 4).sum / 5 = 9 ∧ 
  numbers.sum / 9 = 25 / 3 →
  ∃ x ∈ numbers.take 5 ∩ numbers.drop 4, x = 5 := by
sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_sets_l2680_268059


namespace NUMINAMATH_CALUDE_digit2List_2000th_digit_l2680_268053

/-- A function that generates the list of positive integers with first digit 2 in increasing order -/
def digit2List : ℕ → ℕ 
| 0 => 2
| n + 1 => 
  let prev := digit2List n
  if prev < 10 then 20
  else if prev % 10 = 9 then prev + 11
  else prev + 1

/-- The number formed by the nth, (n+1)th, and (n+2)th digits in the digit2List -/
def threeDigitNumber (n : ℕ) : ℕ := sorry

theorem digit2List_2000th_digit : threeDigitNumber 1998 = 427 := by sorry

end NUMINAMATH_CALUDE_digit2List_2000th_digit_l2680_268053


namespace NUMINAMATH_CALUDE_simplify_powers_of_ten_l2680_268087

theorem simplify_powers_of_ten : 
  (10 ^ 0.4) * (10 ^ 0.5) * (10 ^ 0.2) * (10 ^ (-0.6)) * (10 ^ 0.5) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_powers_of_ten_l2680_268087


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2680_268081

theorem regular_polygon_sides (n : ℕ) : n ≥ 3 → (n * (n - 3) / 2 + 2 * n = 36) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2680_268081


namespace NUMINAMATH_CALUDE_problem_solution_l2680_268002

theorem problem_solution (a b c : ℕ+) (h1 : 3 * a = b^3) (h2 : 5 * a = c^2) (h3 : ∃ k : ℕ, a = k * 1^6) :
  (∃ m n : ℕ, a = 3 * m ∧ a = 5 * n) ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ a → p = 3 ∨ p = 5) ∧
  a = 1125 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2680_268002


namespace NUMINAMATH_CALUDE_a_less_than_b_less_than_one_l2680_268028

theorem a_less_than_b_less_than_one
  (x : ℝ) (a b : ℝ) 
  (hx : x > 0)
  (hab : a^x < b^x)
  (hb1 : b^x < 1)
  (ha_pos : a > 0)
  (hb_pos : b > 0) :
  a < b ∧ b < 1 := by
sorry

end NUMINAMATH_CALUDE_a_less_than_b_less_than_one_l2680_268028


namespace NUMINAMATH_CALUDE_number_of_phone_repairs_l2680_268009

/-- Represents the repair shop scenario -/
def repair_shop (phone_repairs : ℕ) : Prop :=
  let phone_cost : ℕ := 11
  let laptop_cost : ℕ := 15
  let computer_cost : ℕ := 18
  let laptop_repairs : ℕ := 2
  let computer_repairs : ℕ := 2
  let total_earnings : ℕ := 121
  phone_cost * phone_repairs + laptop_cost * laptop_repairs + computer_cost * computer_repairs = total_earnings

/-- Theorem stating that the number of phone repairs is 5 -/
theorem number_of_phone_repairs : ∃ (phone_repairs : ℕ), phone_repairs = 5 ∧ repair_shop phone_repairs :=
by sorry

end NUMINAMATH_CALUDE_number_of_phone_repairs_l2680_268009


namespace NUMINAMATH_CALUDE_expand_product_l2680_268077

theorem expand_product (x : ℝ) : 5 * (x - 3) * (x + 6) = 5 * x^2 + 15 * x - 90 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2680_268077


namespace NUMINAMATH_CALUDE_even_function_symmetric_and_f_not_odd_l2680_268085

-- Define even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define symmetry about y-axis
def SymmetricAboutYAxis (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the function f(x) = x³ + 1
def f (x : ℝ) : ℝ := x^3 + 1

theorem even_function_symmetric_and_f_not_odd :
  (∀ f : ℝ → ℝ, IsEven f → SymmetricAboutYAxis f) ∧
  ¬IsOdd f :=
by sorry

end NUMINAMATH_CALUDE_even_function_symmetric_and_f_not_odd_l2680_268085


namespace NUMINAMATH_CALUDE_fraction_ordering_l2680_268012

theorem fraction_ordering : (6 : ℚ) / 23 < 8 / 25 ∧ 8 / 25 < 10 / 29 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2680_268012


namespace NUMINAMATH_CALUDE_gas_pressure_change_l2680_268090

/-- Represents the pressure-volume relationship of a gas at constant temperature -/
structure GasState where
  volume : ℝ
  pressure : ℝ

/-- Verifies if two gas states follow the inverse proportionality law -/
def inverseProportion (s1 s2 : GasState) : Prop :=
  s1.pressure * s1.volume = s2.pressure * s2.volume

theorem gas_pressure_change 
  (initial final : GasState) 
  (h_initial : initial.volume = 3 ∧ initial.pressure = 6) 
  (h_final_volume : final.volume = 4.5) 
  (h_inverse : inverseProportion initial final) : 
  final.pressure = 4 := by
sorry

end NUMINAMATH_CALUDE_gas_pressure_change_l2680_268090


namespace NUMINAMATH_CALUDE_different_suit_card_selection_l2680_268095

theorem different_suit_card_selection :
  let total_cards : ℕ := 52
  let num_suits : ℕ := 4
  let cards_per_suit : ℕ := 13
  let selection_size : ℕ := 4

  (num_suits ^ selection_size) = 28561 :=
by
  sorry

end NUMINAMATH_CALUDE_different_suit_card_selection_l2680_268095


namespace NUMINAMATH_CALUDE_square_perimeter_l2680_268035

theorem square_perimeter (s : ℝ) (h1 : s > 0) (h2 : (5 * s) / 2 = 44) : 4 * s = 70.4 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2680_268035


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_x_plus_y_to_8_l2680_268092

theorem coefficient_x3y5_in_x_plus_y_to_8 :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k) * X^k * Y^(8 - k)) =
  56 * X^3 * Y^5 + (Finset.range 9).sum (fun k => if k ≠ 3 then (Nat.choose 8 k) * X^k * Y^(8 - k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_x_plus_y_to_8_l2680_268092


namespace NUMINAMATH_CALUDE_line_equation_proof_l2680_268007

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space using the general form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a function to check if two lines are perpendicular
def perpendicularLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Theorem statement
theorem line_equation_proof :
  let p : Point2D := { x := -1, y := 2 }
  let l1 : Line2D := { a := 2, b := -3, c := 4 }
  let l2 : Line2D := { a := 1, b := 4, c := -6 }
  (pointOnLine p l2) ∧ (perpendicularLines l1 l2) := by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2680_268007


namespace NUMINAMATH_CALUDE_payment_cases_count_l2680_268000

/-- The number of 500-won coins available -/
def num_500_coins : ℕ := 4

/-- The number of 100-won coins available -/
def num_100_coins : ℕ := 2

/-- The number of 10-won coins available -/
def num_10_coins : ℕ := 5

/-- The total number of non-zero payment cases -/
def total_cases : ℕ := (num_500_coins + 1) * (num_100_coins + 1) * (num_10_coins + 1) - 1

theorem payment_cases_count : total_cases = 89 := by
  sorry

end NUMINAMATH_CALUDE_payment_cases_count_l2680_268000


namespace NUMINAMATH_CALUDE_isabel_paper_calculation_l2680_268023

/-- The number of pieces of paper Isabel bought -/
def total_paper : ℕ := 900

/-- The number of pieces of paper Isabel used -/
def used_paper : ℕ := 156

/-- The number of pieces of paper Isabel has left -/
def remaining_paper : ℕ := total_paper - used_paper

theorem isabel_paper_calculation :
  remaining_paper = 744 :=
sorry

end NUMINAMATH_CALUDE_isabel_paper_calculation_l2680_268023


namespace NUMINAMATH_CALUDE_refrigerator_cost_proof_l2680_268078

/-- The cost of the refrigerator that satisfies the given conditions --/
def refrigerator_cost : ℝ := 15000

/-- The cost of the mobile phone --/
def mobile_phone_cost : ℝ := 8000

/-- The selling price of the refrigerator --/
def refrigerator_selling_price : ℝ := 0.96 * refrigerator_cost

/-- The selling price of the mobile phone --/
def mobile_phone_selling_price : ℝ := 1.09 * mobile_phone_cost

/-- The total profit --/
def total_profit : ℝ := 120

theorem refrigerator_cost_proof :
  refrigerator_selling_price + mobile_phone_selling_price = 
  refrigerator_cost + mobile_phone_cost + total_profit :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_cost_proof_l2680_268078


namespace NUMINAMATH_CALUDE_relay_team_selection_l2680_268006

/-- The number of sprinters available for selection -/
def total_sprinters : ℕ := 6

/-- The number of athletes needed for the relay race -/
def selected_athletes : ℕ := 4

/-- The number of mandatory athletes (A and B) -/
def mandatory_athletes : ℕ := 2

/-- The number of positions in the relay race -/
def race_positions : ℕ := 4

/-- The number of different arrangements for the relay race -/
def relay_arrangements : ℕ := 72

theorem relay_team_selection :
  (total_sprinters.choose (selected_athletes - mandatory_athletes)) *
  (mandatory_athletes.choose 1) *
  ((selected_athletes - 1).factorial) = relay_arrangements :=
sorry

end NUMINAMATH_CALUDE_relay_team_selection_l2680_268006


namespace NUMINAMATH_CALUDE_binary_digit_difference_l2680_268042

theorem binary_digit_difference (n m : ℕ) (hn : n = 950) (hm : m = 150) :
  (Nat.log 2 n + 1) - (Nat.log 2 m + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l2680_268042


namespace NUMINAMATH_CALUDE_function_max_min_difference_l2680_268033

theorem function_max_min_difference (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun x => a^x
  let max_val := max (f 1) (f 2)
  let min_val := min (f 1) (f 2)
  max_val = min_val + a / 3 → a = 4/3 ∨ a = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_function_max_min_difference_l2680_268033


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2680_268043

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) (m : ℕ) :
  d ≠ 0 →
  arithmetic_sequence a d →
  a 3 + a 6 + a 10 + a 13 = 32 →
  a 8 = 8 →
  a m = 8 →
  m = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2680_268043


namespace NUMINAMATH_CALUDE_cos_sin_sum_implies_cos_double_sum_zero_l2680_268018

theorem cos_sin_sum_implies_cos_double_sum_zero 
  (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z = 1)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 1) :
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_implies_cos_double_sum_zero_l2680_268018


namespace NUMINAMATH_CALUDE_blue_pens_count_l2680_268079

theorem blue_pens_count (red_price blue_price total_cost total_pens : ℕ) 
  (h1 : red_price = 5)
  (h2 : blue_price = 7)
  (h3 : total_cost = 102)
  (h4 : total_pens = 16) :
  ∃ (red_count blue_count : ℕ), 
    red_count + blue_count = total_pens ∧
    red_count * red_price + blue_count * blue_price = total_cost ∧
    blue_count = 11 := by
  sorry

end NUMINAMATH_CALUDE_blue_pens_count_l2680_268079


namespace NUMINAMATH_CALUDE_inequality_and_floor_function_l2680_268076

theorem inequality_and_floor_function (n : ℕ) : 
  (Real.sqrt (n + 1) + 2 * Real.sqrt n < Real.sqrt (9 * n + 3)) ∧ 
  ¬∃ n : ℕ, ⌊Real.sqrt (n + 1) + 2 * Real.sqrt n⌋ < ⌊Real.sqrt (9 * n + 3)⌋ :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_floor_function_l2680_268076


namespace NUMINAMATH_CALUDE_derivative_y_l2680_268024

noncomputable def y (x : ℝ) : ℝ := -1/2 * Real.log (Real.tanh (x/2)) - Real.cosh x / (2 * Real.sinh x ^ 2)

theorem derivative_y (x : ℝ) : 
  deriv y x = 1 / Real.sinh x ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_derivative_y_l2680_268024


namespace NUMINAMATH_CALUDE_point_3_4_in_first_quadrant_l2680_268047

/-- A point is in the first quadrant if both its x and y coordinates are positive. -/
def is_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The point (3,4) lies in the first quadrant. -/
theorem point_3_4_in_first_quadrant : is_first_quadrant 3 4 := by
  sorry

end NUMINAMATH_CALUDE_point_3_4_in_first_quadrant_l2680_268047


namespace NUMINAMATH_CALUDE_one_fourth_of_7_2_l2680_268016

theorem one_fourth_of_7_2 : (7.2 : ℚ) / 4 = 9 / 5 := by sorry

end NUMINAMATH_CALUDE_one_fourth_of_7_2_l2680_268016


namespace NUMINAMATH_CALUDE_height_difference_l2680_268025

theorem height_difference (tallest_height shortest_height : ℝ) 
  (h1 : tallest_height = 77.75)
  (h2 : shortest_height = 68.25) :
  tallest_height - shortest_height = 9.5 := by
sorry

end NUMINAMATH_CALUDE_height_difference_l2680_268025


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2680_268051

theorem quadratic_inequality_solution (m n : ℝ) : 
  (∀ x : ℝ, 2*x^2 + m*x + n > 0 ↔ x > 3 ∨ x < -2) → 
  m + n = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2680_268051


namespace NUMINAMATH_CALUDE_exists_n_sum_diff_gt_l2680_268040

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem exists_n_sum_diff_gt (m : ℕ) : 
  ∃ n : ℕ, n > 0 ∧ sum_of_digits n - sum_of_digits (n^2) > m := by sorry

end NUMINAMATH_CALUDE_exists_n_sum_diff_gt_l2680_268040


namespace NUMINAMATH_CALUDE_triangle_AXY_is_obtuse_l2680_268096

-- Define the triangular pyramid ABCD
structure TriangularPyramid where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the inscribed and exscribed spheres
structure InscribedSphere where
  center : Point
  radius : ℝ

structure ExscribedSphere where
  center : Point
  radius : ℝ

-- Define the points where the spheres touch face BCD
def X (pyramid : TriangularPyramid) (inscribedSphere : InscribedSphere) : Point := sorry
def Y (pyramid : TriangularPyramid) (exscribedSphere : ExscribedSphere) : Point := sorry

-- Define the angle AXY
def angle_AXY (pyramid : TriangularPyramid) (inscribedSphere : InscribedSphere) (exscribedSphere : ExscribedSphere) : ℝ := sorry

-- Theorem statement
theorem triangle_AXY_is_obtuse (pyramid : TriangularPyramid) (inscribedSphere : InscribedSphere) (exscribedSphere : ExscribedSphere) :
  angle_AXY pyramid inscribedSphere exscribedSphere > π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_AXY_is_obtuse_l2680_268096


namespace NUMINAMATH_CALUDE_parallel_vector_problem_l2680_268097

/-- Given two vectors a and b in ℝ², where a = (-2, 1), |b| = 5, and a is parallel to b,
    prove that b is either (-2√5, √5) or (2√5, -√5). -/
theorem parallel_vector_problem (a b : ℝ × ℝ) : 
  a = (-2, 1) → 
  ‖b‖ = 5 → 
  ∃ (k : ℝ), b = k • a → 
  b = (-2 * Real.sqrt 5, Real.sqrt 5) ∨ b = (2 * Real.sqrt 5, -Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_parallel_vector_problem_l2680_268097


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l2680_268041

/-- The quadratic function f(x) = 2x^2 - 3 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 3

/-- Theorem: The quadratic function f(x) = 2x^2 - 3 has exactly two distinct real roots -/
theorem quadratic_two_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l2680_268041


namespace NUMINAMATH_CALUDE_cherry_tomatoes_count_l2680_268058

theorem cherry_tomatoes_count (tomatoes_per_jar : ℕ) (jars_needed : ℕ) 
  (h1 : tomatoes_per_jar = 8) 
  (h2 : jars_needed = 7) : 
  tomatoes_per_jar * jars_needed = 56 := by
  sorry

end NUMINAMATH_CALUDE_cherry_tomatoes_count_l2680_268058


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2680_268069

theorem triangle_angle_measure (X Y Z : Real) (h1 : X = 72) 
  (h2 : Y = 4 * Z + 10) (h3 : X + Y + Z = 180) : Z = 19.6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2680_268069


namespace NUMINAMATH_CALUDE_central_number_is_14_l2680_268003

/-- Represents a 5x5 grid of natural numbers -/
def Grid := Fin 5 → Fin 5 → ℕ

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p q : Fin 5 × Fin 5) : Prop :=
  (p.1 = q.1 ∧ (p.2 = q.2 + 1 ∨ p.2 + 1 = q.2)) ∨
  (p.2 = q.2 ∧ (p.1 = q.1 + 1 ∨ p.1 + 1 = q.1)) ∨
  ((p.1 = q.1 + 1 ∨ p.1 + 1 = q.1) ∧ (p.2 = q.2 + 1 ∨ p.2 + 1 = q.2))

/-- The main theorem to prove -/
theorem central_number_is_14 (g : Grid) : 
  (∀ i j, g i j ∈ Finset.range 26) →
  (∀ i j k l, (i, j) ≠ (k, l) → g i j ≠ g k l) →
  (∀ i j, g i j ≠ 1 → g i j ≠ 2 → 
    ∃ p q, adjacent (i, j) p ∧ adjacent (i, j) q ∧ g i j = g p.1 p.2 + g q.1 q.2) →
  g 0 0 = 1 →
  g 1 1 = 16 →
  g 1 3 = 18 →
  g 2 0 = 17 →
  g 2 4 = 21 →
  g 3 1 = 23 →
  g 3 3 = 25 →
  g 1 2 = 14 := by
sorry

end NUMINAMATH_CALUDE_central_number_is_14_l2680_268003


namespace NUMINAMATH_CALUDE_sum_of_ages_l2680_268060

/-- Represents the ages of family members -/
structure FamilyAges where
  father : ℕ
  son : ℕ
  mother : ℕ
  daughter : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  (ages.father - 20 = 3 * (ages.son - 20)) ∧
  (ages.mother - 20 = 4 * (ages.daughter - 20)) ∧
  (ages.father = 2 * ages.son) ∧
  (ages.mother = 3 * ages.daughter)

/-- The theorem to be proved -/
theorem sum_of_ages (ages : FamilyAges) : 
  satisfiesConditions ages → ages.father + ages.son + ages.mother + ages.daughter = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2680_268060


namespace NUMINAMATH_CALUDE_intersection_point_is_on_line_and_plane_l2680_268070

/-- The line equation represented as a function of a parameter t -/
def line_equation (t : ℝ) : ℝ × ℝ × ℝ :=
  (-3, 2 - 3*t, -5 + 11*t)

/-- The plane equation -/
def plane_equation (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  5*x + 7*y + 9*z - 32 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (-3, -1, 6)

theorem intersection_point_is_on_line_and_plane :
  ∃ t : ℝ, line_equation t = intersection_point ∧ plane_equation intersection_point := by
  sorry

#check intersection_point_is_on_line_and_plane

end NUMINAMATH_CALUDE_intersection_point_is_on_line_and_plane_l2680_268070


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2680_268013

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8/15) (h2 : x - y = 1/35) : x^2 - y^2 = 8/525 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2680_268013


namespace NUMINAMATH_CALUDE_nursery_school_students_l2680_268029

theorem nursery_school_students (total : ℕ) 
  (h1 : (total : ℚ) / 10 = (total - (total - 50) : ℚ))
  (h2 : total - 50 ≥ 20) : total = 300 := by
  sorry

end NUMINAMATH_CALUDE_nursery_school_students_l2680_268029


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l2680_268063

theorem chocolate_box_problem (total_bars : ℕ) (bars_per_small_box : ℕ) (h1 : total_bars = 500) (h2 : bars_per_small_box = 25) :
  total_bars / bars_per_small_box = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l2680_268063


namespace NUMINAMATH_CALUDE_smallest_four_digit_number_with_remainders_l2680_268045

theorem smallest_four_digit_number_with_remainders : ∃ (n : ℕ),
  (n ≥ 1000 ∧ n < 10000) ∧
  (n % 5 = 1) ∧
  (n % 7 = 4) ∧
  (n % 11 = 9) ∧
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 5 = 1 ∧ m % 7 = 4 ∧ m % 11 = 9 → n ≤ m) ∧
  n = 1131 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_number_with_remainders_l2680_268045


namespace NUMINAMATH_CALUDE_cyclists_speed_product_l2680_268089

theorem cyclists_speed_product (u v : ℝ) : 
  (u > 0) →  -- Assume positive speeds
  (v > 0) →
  (v > u) →  -- Faster cyclist has higher speed
  (6 / u = 6 / v + 1 / 12) →  -- Faster cyclist travels 6 km in 5 minutes less
  (v / 3 = u / 3 + 4) →  -- In 20 minutes, faster cyclist travels 4 km more
  u * v = 864 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_speed_product_l2680_268089


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l2680_268020

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x

-- State the theorem
theorem f_derivative_at_zero : 
  (deriv f) 0 = -3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l2680_268020


namespace NUMINAMATH_CALUDE_complementary_event_a_l2680_268014

/-- A batch of products containing genuine and defective items. -/
structure Batch where
  genuine : ℕ
  defective : ℕ
  defective_gt_four : defective > 4

/-- A selection of 4 products from a batch. -/
structure Selection where
  batch : Batch
  items : Finset (Fin 4)
  items_card : items.card = 4

/-- Event A: At least one defective product in the selection. -/
def eventA (s : Selection) : Prop :=
  ∃ i ∈ s.items, i < s.batch.defective

/-- Complementary event of A: No defective products in the selection. -/
def complementEventA (s : Selection) : Prop :=
  ∀ i ∈ s.items, i ≥ s.batch.defective

/-- Theorem stating that the complementary event of A is "no defective products". -/
theorem complementary_event_a (s : Selection) :
  ¬(eventA s) ↔ complementEventA s :=
sorry

end NUMINAMATH_CALUDE_complementary_event_a_l2680_268014


namespace NUMINAMATH_CALUDE_circle_equation_valid_a_l2680_268088

def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x + 2*a*y + 2*a^2 + a - 1 = 0

def is_valid_a (a : ℝ) : Prop :=
  a < 1

def given_options : List ℝ := [-1, 1, 0, 3]

theorem circle_equation_valid_a :
  ∀ a ∈ given_options, circle_equation x y a → is_valid_a a ↔ (a = -1 ∨ a = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_valid_a_l2680_268088


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l2680_268004

theorem monic_quadratic_with_complex_root :
  ∃! (a b : ℝ), (∀ x : ℂ, x^2 + a*x + b = 0 ↔ x = 3 - 2*I ∨ x = 3 + 2*I) ∧
                (∀ x : ℂ, x^2 + a*x + b = (x - (3 - 2*I)) * (x - (3 + 2*I))) :=
by sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l2680_268004


namespace NUMINAMATH_CALUDE_cake_flour_requirement_l2680_268021

theorem cake_flour_requirement (total_flour : ℝ) (cake_flour : ℝ) (cupcake_flour : ℝ)
  (cupcake_requirement : ℝ) (cake_price : ℝ) (cupcake_price : ℝ) (total_earnings : ℝ)
  (h1 : total_flour = 6)
  (h2 : cake_flour = 4)
  (h3 : cupcake_flour = 2)
  (h4 : cupcake_requirement = 1/5)
  (h5 : cake_price = 2.5)
  (h6 : cupcake_price = 1)
  (h7 : total_earnings = 30) :
  cake_flour / (cake_flour / (total_earnings - cupcake_flour / cupcake_requirement * cupcake_price) * cake_price) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cake_flour_requirement_l2680_268021


namespace NUMINAMATH_CALUDE_car_speed_l2680_268038

theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 624) 
    (h2 : time = 3) 
    (h3 : speed = distance / time) : speed = 208 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l2680_268038


namespace NUMINAMATH_CALUDE_farm_animals_l2680_268001

theorem farm_animals (total_legs : ℕ) (sheep_count : ℕ) : 
  total_legs = 60 ∧ sheep_count = 10 → 
  ∃ (chicken_count : ℕ), 
    chicken_count * 2 + sheep_count * 4 = total_legs ∧
    chicken_count + sheep_count = 20 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_l2680_268001


namespace NUMINAMATH_CALUDE_coffee_package_size_l2680_268084

theorem coffee_package_size
  (total_coffee : ℕ)
  (num_larger_packages : ℕ)
  (num_small_packages : ℕ)
  (small_package_size : ℕ)
  (h1 : total_coffee = 55)
  (h2 : num_larger_packages = 3)
  (h3 : num_small_packages = num_larger_packages + 2)
  (h4 : small_package_size = 5)
  : ∃ (larger_package_size : ℕ),
    larger_package_size * num_larger_packages +
    small_package_size * num_small_packages = total_coffee ∧
    larger_package_size = 10 := by
  sorry

end NUMINAMATH_CALUDE_coffee_package_size_l2680_268084


namespace NUMINAMATH_CALUDE_expansion_equals_cube_l2680_268005

theorem expansion_equals_cube : 101^3 + 3*(101^2) + 3*101 + 1 = 102^3 := by sorry

end NUMINAMATH_CALUDE_expansion_equals_cube_l2680_268005


namespace NUMINAMATH_CALUDE_x_plus_x_squared_equals_twelve_l2680_268044

theorem x_plus_x_squared_equals_twelve (x : ℝ) (h : x = 3) : x + x^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_x_squared_equals_twelve_l2680_268044


namespace NUMINAMATH_CALUDE_troll_difference_l2680_268065

/-- The number of trolls hiding under the bridge -/
def T : ℕ := 18

/-- The number of trolls hiding by the path in the forest -/
def forest_trolls : ℕ := 6

/-- The number of trolls hiding in the plains -/
def plains_trolls : ℕ := T / 2

/-- The total number of trolls -/
def total_trolls : ℕ := 33

theorem troll_difference : 
  4 * forest_trolls - T = 6 ∧ 
  forest_trolls + T + plains_trolls = total_trolls :=
sorry

end NUMINAMATH_CALUDE_troll_difference_l2680_268065


namespace NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l2680_268031

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem stating that there are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_seven_balls_three_boxes : 
  distribute_balls 7 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l2680_268031


namespace NUMINAMATH_CALUDE_sum_of_5_and_8_l2680_268055

theorem sum_of_5_and_8 : 5 + 8 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_5_and_8_l2680_268055


namespace NUMINAMATH_CALUDE_imaginary_number_condition_l2680_268064

theorem imaginary_number_condition (m : ℝ) : 
  let z : ℂ := (m + Complex.I) / (1 + m * Complex.I)
  z.re = 0 ∧ z.im ≠ 0 → m = 0 := by
sorry

end NUMINAMATH_CALUDE_imaginary_number_condition_l2680_268064


namespace NUMINAMATH_CALUDE_xy_difference_l2680_268057

theorem xy_difference (x y : ℝ) (h : 10 * x^2 - 16 * x * y + 8 * y^2 + 6 * x - 4 * y + 1 = 0) : 
  x - y = -0.25 := by
sorry

end NUMINAMATH_CALUDE_xy_difference_l2680_268057


namespace NUMINAMATH_CALUDE_train_speed_l2680_268066

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 160) (h2 : time = 16) :
  length / time = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2680_268066


namespace NUMINAMATH_CALUDE_unit_digit_of_sum_powers_l2680_268056

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def unitDigit (n : ℕ) : ℕ := n % 10

theorem unit_digit_of_sum_powers (a b c : ℕ) :
  unitDigit (a^(sumFactorials a) + b^(sumFactorials b) + c^(sumFactorials c)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_unit_digit_of_sum_powers_l2680_268056


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2680_268049

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ x ≥ 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 4}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (Bᶜ) = {x | x < -1 ∨ x ≥ 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2680_268049


namespace NUMINAMATH_CALUDE_number_of_factors_of_M_l2680_268039

def M : ℕ := 58^6 + 6*58^5 + 15*58^4 + 20*58^3 + 15*58^2 + 6*58 + 1

theorem number_of_factors_of_M : 
  (Finset.filter (λ x => M % x = 0) (Finset.range (M + 1))).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_M_l2680_268039


namespace NUMINAMATH_CALUDE_registration_combinations_l2680_268036

/-- The number of students --/
def num_students : ℕ := 5

/-- The number of activity groups --/
def num_groups : ℕ := 2

/-- The total number of registration methods --/
def total_registrations : ℕ := num_groups ^ num_students

/-- Theorem stating the total number of registration methods --/
theorem registration_combinations :
  total_registrations = 32 := by sorry

end NUMINAMATH_CALUDE_registration_combinations_l2680_268036


namespace NUMINAMATH_CALUDE_gold_value_proof_l2680_268052

def gold_problem (legacy_bars : ℕ) (aleena_difference : ℕ) (bar_value : ℕ) : Prop :=
  let aleena_bars := legacy_bars - aleena_difference
  let total_bars := legacy_bars + aleena_bars
  let total_value := total_bars * bar_value
  total_value = 17600

theorem gold_value_proof :
  gold_problem 5 2 2200 := by
  sorry

end NUMINAMATH_CALUDE_gold_value_proof_l2680_268052


namespace NUMINAMATH_CALUDE_circle_intersection_condition_l2680_268046

def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 9

def line_l (x y k : ℝ) : Prop := y = k * x + 3

def point_on_chord (M : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

def circle_intersects (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 ≤ radius^2

theorem circle_intersection_condition (k : ℝ) :
  (∀ x y : ℝ, circle_C x y → line_l x y k → 
    ∀ A B : ℝ × ℝ, circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l A.1 A.2 k ∧ line_l B.1 B.2 k →
      ∀ M : ℝ × ℝ, point_on_chord M A B →
        ∀ x y : ℝ, circle_C x y → circle_intersects M 2 x y) →
  k ≥ -3/4 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_condition_l2680_268046


namespace NUMINAMATH_CALUDE_martha_crayons_l2680_268074

def crayons_problem (initial_crayons : ℕ) (total_after_buying : ℕ) : Prop :=
  let lost_crayons := initial_crayons / 2
  let remaining_crayons := initial_crayons - lost_crayons
  let new_set_size := total_after_buying - remaining_crayons
  new_set_size = 20

theorem martha_crayons : crayons_problem 18 29 := by
  sorry

end NUMINAMATH_CALUDE_martha_crayons_l2680_268074


namespace NUMINAMATH_CALUDE_number_problem_l2680_268082

theorem number_problem (x : ℤ) : x + 14 = 56 → 3 * x = 126 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2680_268082


namespace NUMINAMATH_CALUDE_exists_same_direction_interval_l2680_268068

/-- Represents a periodic motion on a line segment -/
structure PeriodicMotion where
  period : ℝ
  direction : ℝ → Bool  -- True for one direction, False for the opposite

/-- Theorem: Given three periodic motions with periods 12, 6, and 4 minutes,
    there always exists a 1-minute interval where all motions are in the same direction -/
theorem exists_same_direction_interval
  (m1 : PeriodicMotion)
  (m2 : PeriodicMotion)
  (m3 : PeriodicMotion)
  (h1 : m1.period = 12)
  (h2 : m2.period = 6)
  (h3 : m3.period = 4)
  : ∃ (t : ℝ), ∀ (s : ℝ), 0 ≤ s ∧ s ≤ 1 →
    (m1.direction (t + s) = m2.direction (t + s)) ∧
    (m2.direction (t + s) = m3.direction (t + s)) :=
sorry

end NUMINAMATH_CALUDE_exists_same_direction_interval_l2680_268068


namespace NUMINAMATH_CALUDE_square_root_divided_by_15_equals_4_l2680_268034

theorem square_root_divided_by_15_equals_4 (n : ℝ) : 
  (Real.sqrt n) / 15 = 4 → n = 3600 := by sorry

end NUMINAMATH_CALUDE_square_root_divided_by_15_equals_4_l2680_268034


namespace NUMINAMATH_CALUDE_at_most_one_obtuse_angle_l2680_268010

-- Define a triangle
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  sum_of_angles : angle1 + angle2 + angle3 = 180
  positive_angles : angle1 > 0 ∧ angle2 > 0 ∧ angle3 > 0

-- Define an obtuse angle
def is_obtuse (angle : Real) : Prop := angle > 90

-- Theorem statement
theorem at_most_one_obtuse_angle (t : Triangle) : 
  ¬(is_obtuse t.angle1 ∧ is_obtuse t.angle2) ∧ 
  ¬(is_obtuse t.angle1 ∧ is_obtuse t.angle3) ∧ 
  ¬(is_obtuse t.angle2 ∧ is_obtuse t.angle3) :=
sorry

end NUMINAMATH_CALUDE_at_most_one_obtuse_angle_l2680_268010


namespace NUMINAMATH_CALUDE_max_moves_le_four_max_moves_21x21_max_moves_20x21_l2680_268048

/-- Represents a rectangular grid with lights -/
structure Grid (m n : ℕ) where
  lights : Fin m → Fin n → Bool

/-- A move in the game -/
structure Move (m n : ℕ) where
  line : (Fin m × Fin n) → Bool
  affects_light : ∀ (i : Fin m) (j : Fin n), line (i, j) = false

/-- The game state -/
structure GameState (m n : ℕ) where
  grid : Grid m n
  moves : List (Move m n)

/-- The maximum number of moves for any rectangular grid is at most 4 -/
theorem max_moves_le_four (m n : ℕ) (g : GameState m n) :
  g.moves.length ≤ 4 :=
sorry

/-- For a 21x21 square grid, the maximum number of moves is 3 -/
theorem max_moves_21x21 (g : GameState 21 21) :
  g.moves.length ≤ 3 :=
sorry

/-- For a 20x21 rectangular grid, the maximum number of moves is 4 -/
theorem max_moves_20x21 (g : GameState 20 21) :
  g.moves.length ≤ 4 ∧ ∃ (g' : GameState 20 21), g'.moves.length = 4 :=
sorry

end NUMINAMATH_CALUDE_max_moves_le_four_max_moves_21x21_max_moves_20x21_l2680_268048


namespace NUMINAMATH_CALUDE_gathering_handshakes_l2680_268030

/-- The number of gremlins at the gathering -/
def num_gremlins : ℕ := 25

/-- The number of imps at the gathering -/
def num_imps : ℕ := 20

/-- The number of imps willing to shake hands with gremlins -/
def num_imps_shaking : ℕ := 10

/-- The number of gremlins each participating imp shakes hands with -/
def gremlin_per_imp : ℕ := 15

/-- Calculate the number of handshakes among gremlins -/
def gremlin_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculate the number of handshakes between gremlins and imps -/
def gremlin_imp_handshakes : ℕ := num_imps_shaking * gremlin_per_imp

/-- The total number of handshakes at the gathering -/
def total_handshakes : ℕ := gremlin_handshakes num_gremlins + gremlin_imp_handshakes

theorem gathering_handshakes : total_handshakes = 450 := by
  sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l2680_268030


namespace NUMINAMATH_CALUDE_fixed_distance_point_l2680_268008

open Real

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

noncomputable def s : ℝ := 9/8
noncomputable def v : ℝ := 1/8

theorem fixed_distance_point (a c p : n) 
  (h : ‖p - c‖ = 3 * ‖p - a‖) : 
  ∃ (k : ℝ), ‖p - (s • a + v • c)‖ = k := by
  sorry

end NUMINAMATH_CALUDE_fixed_distance_point_l2680_268008


namespace NUMINAMATH_CALUDE_digit_interchange_l2680_268017

theorem digit_interchange (x : ℕ) : x = 9 ↔ 32 - x = 23 := by sorry

end NUMINAMATH_CALUDE_digit_interchange_l2680_268017


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l2680_268099

theorem sqrt_expression_equality : 
  Real.sqrt 12 - Real.sqrt (1/3) - Real.sqrt 2 * Real.sqrt 6 = -(Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l2680_268099


namespace NUMINAMATH_CALUDE_leader_assistant_combinations_l2680_268093

/-- The number of ways to choose a team leader and an assistant of the same gender -/
def choose_leader_and_assistant (total : ℕ) (boys : ℕ) (girls : ℕ) : ℕ :=
  boys * (boys - 1) + girls * (girls - 1)

/-- Theorem: There are 98 ways to choose a team leader and an assistant of the same gender
    from a class of 15 students, consisting of 8 boys and 7 girls -/
theorem leader_assistant_combinations :
  choose_leader_and_assistant 15 8 7 = 98 := by
  sorry

end NUMINAMATH_CALUDE_leader_assistant_combinations_l2680_268093


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l2680_268080

theorem factorial_difference_quotient : (Nat.factorial 12 - Nat.factorial 11) / Nat.factorial 10 = 121 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l2680_268080


namespace NUMINAMATH_CALUDE_hexagon_circumscribable_l2680_268022

-- Define a hexagon
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

-- Define the property of parallel opposite sides
def has_parallel_opposite_sides (h : Hexagon) : Prop :=
  ∀ i : Fin 3, 
    let v1 := h.vertices i
    let v2 := h.vertices ((i + 1) % 6)
    let v3 := h.vertices ((i + 3) % 6)
    let v4 := h.vertices ((i + 4) % 6)
    (v2.1 - v1.1) * (v4.2 - v3.2) = (v2.2 - v1.2) * (v4.1 - v3.1)

-- Define the property of equal diagonals
def has_equal_diagonals (h : Hexagon) : Prop :=
  ∀ i : Fin 3,
    let v1 := h.vertices i
    let v2 := h.vertices ((i + 3) % 6)
    (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = 
    (h.vertices 0).1^2 + (h.vertices 0).2^2 + 
    (h.vertices 3).1^2 + (h.vertices 3).2^2 - 
    2 * ((h.vertices 0).1 * (h.vertices 3).1 + (h.vertices 0).2 * (h.vertices 3).2)

-- Theorem statement
theorem hexagon_circumscribable 
  (h : Hexagon) 
  (parallel : has_parallel_opposite_sides h) 
  (equal_diagonals : has_equal_diagonals h) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    ∀ i : Fin 6, 
      (h.vertices i).1^2 + (h.vertices i).2^2 - 
      2 * (center.1 * (h.vertices i).1 + center.2 * (h.vertices i).2) + 
      center.1^2 + center.2^2 = radius^2 :=
sorry

end NUMINAMATH_CALUDE_hexagon_circumscribable_l2680_268022


namespace NUMINAMATH_CALUDE_reciprocal_sum_pairs_l2680_268032

theorem reciprocal_sum_pairs : 
  (∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 6) ∧
    s.card = 9) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_pairs_l2680_268032


namespace NUMINAMATH_CALUDE_cost_of_thousand_pieces_l2680_268011

/-- The cost in dollars of purchasing a given number of pieces of gum -/
def gum_cost (pieces : ℕ) : ℚ :=
  if pieces ≤ 500 then
    (pieces : ℚ) / 100
  else
    (500 : ℚ) / 100 + ((pieces - 500 : ℚ) * 8) / 1000

/-- The cost of 1000 pieces of gum is $9.00 -/
theorem cost_of_thousand_pieces : gum_cost 1000 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_thousand_pieces_l2680_268011


namespace NUMINAMATH_CALUDE_range_of_a_plus_3b_l2680_268073

theorem range_of_a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b) (h2 : a + b ≤ 1) 
  (h3 : 1 ≤ a - 2*b) (h4 : a - 2*b ≤ 3) : 
  (-11/3 ≤ a + 3*b) ∧ (a + 3*b ≤ 1) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_plus_3b_l2680_268073


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2680_268071

theorem arithmetic_calculation : 8 / 4 - 3 - 9 + 3 * 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2680_268071


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l2680_268050

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * x + 1 / (x^2) ≥ 4 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 0) : 
  (3 * x + 1 / (x^2) = 4) ↔ (x = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l2680_268050


namespace NUMINAMATH_CALUDE_correct_statements_l2680_268026

theorem correct_statements (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (3*a + 2*b) / (2*a + 3*b) < a / b ∧
  Real.log ((a + b) / 2) > (Real.log a + Real.log b) / 2 :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_l2680_268026


namespace NUMINAMATH_CALUDE_complex_multiplication_division_l2680_268054

theorem complex_multiplication_division (z₁ z₂ : ℂ) : 
  z₁ = 1 - I → z₂ = 1 + I → (z₁ * z₂) / I = -2 * I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_division_l2680_268054


namespace NUMINAMATH_CALUDE_sin_even_translation_l2680_268015

theorem sin_even_translation (φ : Real) : 
  (∀ x, (1/2) * Real.sin (2*x + φ - π/4) = (1/2) * Real.sin (2*(-x) + φ - π/4)) → 
  ∃ k : Int, φ = π/4 + k * π :=
by sorry

end NUMINAMATH_CALUDE_sin_even_translation_l2680_268015
