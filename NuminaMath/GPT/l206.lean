import Mathlib

namespace NUMINAMATH_GPT_logan_snowfall_total_l206_20681

theorem logan_snowfall_total (wednesday thursday friday : ℝ) :
  wednesday = 0.33 → thursday = 0.33 → friday = 0.22 → wednesday + thursday + friday = 0.88 :=
by
  intros hw ht hf
  rw [hw, ht, hf]
  exact (by norm_num : (0.33 : ℝ) + 0.33 + 0.22 = 0.88)

end NUMINAMATH_GPT_logan_snowfall_total_l206_20681


namespace NUMINAMATH_GPT_final_price_l206_20667

variable (OriginalPrice : ℝ)

def salePrice (OriginalPrice : ℝ) : ℝ :=
  0.6 * OriginalPrice

def priceAfterCoupon (SalePrice : ℝ) : ℝ :=
  0.75 * SalePrice

theorem final_price (OriginalPrice : ℝ) :
  priceAfterCoupon (salePrice OriginalPrice) = 0.45 * OriginalPrice := by
  sorry

end NUMINAMATH_GPT_final_price_l206_20667


namespace NUMINAMATH_GPT_find_x_l206_20637

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 9^n - 1) (h2 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 ≥ 2 ∧ p2 ≥ 2 ∧ p3 ≥ 2 ∧ x = p1 * p2 * p3 ∧ (p1 = 11 ∨ p2 = 11 ∨ p3 = 11)) :
    x = 59048 := 
sorry

end NUMINAMATH_GPT_find_x_l206_20637


namespace NUMINAMATH_GPT_jackie_apples_l206_20646

theorem jackie_apples (a : ℕ) (j : ℕ) (h1 : a = 9) (h2 : a = j + 3) : j = 6 :=
by
  sorry

end NUMINAMATH_GPT_jackie_apples_l206_20646


namespace NUMINAMATH_GPT_gcd_consecutive_odd_product_l206_20630

theorem gcd_consecutive_odd_product (n : ℕ) (hn : n % 2 = 0 ∧ n > 0) : 
  Nat.gcd ((n+1)*(n+3)*(n+7)*(n+9)) 15 = 15 := 
sorry

end NUMINAMATH_GPT_gcd_consecutive_odd_product_l206_20630


namespace NUMINAMATH_GPT_total_cookies_collected_l206_20600

theorem total_cookies_collected 
  (abigail_boxes : ℕ) (grayson_boxes : ℕ) (olivia_boxes : ℕ) (cookies_per_box : ℕ)
  (h1 : abigail_boxes = 2) (h2 : grayson_boxes = 3) (h3 : olivia_boxes = 3) (h4 : cookies_per_box = 48) :
  (abigail_boxes * cookies_per_box) + ((grayson_boxes * (cookies_per_box / 4))) + (olivia_boxes * cookies_per_box) = 276 := 
by 
  sorry

end NUMINAMATH_GPT_total_cookies_collected_l206_20600


namespace NUMINAMATH_GPT_john_trip_time_30_min_l206_20659

-- Definitions of the given conditions
variables {D : ℝ} -- Distance John traveled
variables {T : ℝ} -- Time John took
variable (T_john : ℝ) -- Time it took John (in hours)
variable (T_beth : ℝ) -- Time it took Beth (in hours)
variable (D_john : ℝ) -- Distance John traveled (in miles)
variable (D_beth : ℝ) -- Distance Beth traveled (in miles)

-- Given conditions
def john_speed := 40 -- John's speed in mph
def beth_speed := 30 -- Beth's speed in mph
def additional_distance := 5 -- Additional distance Beth traveled in miles
def additional_time := 1 / 3 -- Additional time Beth took in hours

-- Proving the time it took John to complete the trip is 30 minutes (0.5 hours)
theorem john_trip_time_30_min : 
  ∀ (T_john T_beth : ℝ), 
    T_john = (D) / john_speed →
    T_beth = (D + additional_distance) / beth_speed →
    (T_beth = T_john + additional_time) →
    T_john = 1 / 2 :=
by
  intro T_john T_beth
  sorry

end NUMINAMATH_GPT_john_trip_time_30_min_l206_20659


namespace NUMINAMATH_GPT_find_ab_l206_20631

theorem find_ab (a b c : ℕ) (H_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (H_b : b = 1) (H_ccb : (10 * a + b)^2 = 100 * c + 10 * c + b) (H_gt : 100 * c + 10 * c + b > 300) : (10 * a + b) = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l206_20631


namespace NUMINAMATH_GPT_solve_for_x_l206_20611

theorem solve_for_x (x : ℝ) : (3 / 2) * x - 3 = 15 → x = 12 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l206_20611


namespace NUMINAMATH_GPT_prove_a_eq_b_l206_20674

theorem prove_a_eq_b 
  (p q a b : ℝ) 
  (h1 : p + q = 1) 
  (h2 : p * q ≠ 0) 
  (h3 : p / a + q / b = 1 / (p * a + q * b)) : 
  a = b := 
sorry

end NUMINAMATH_GPT_prove_a_eq_b_l206_20674


namespace NUMINAMATH_GPT_additional_savings_correct_l206_20682

def initial_order_amount : ℝ := 10000

def option1_discount1 : ℝ := 0.20
def option1_discount2 : ℝ := 0.20
def option1_discount3 : ℝ := 0.10
def option2_discount1 : ℝ := 0.40
def option2_discount2 : ℝ := 0.05
def option2_discount3 : ℝ := 0.05

def final_price_option1 : ℝ :=
  initial_order_amount * (1 - option1_discount1) *
  (1 - option1_discount2) *
  (1 - option1_discount3)

def final_price_option2 : ℝ :=
  initial_order_amount * (1 - option2_discount1) *
  (1 - option2_discount2) *
  (1 - option2_discount3)

def additional_savings : ℝ :=
  final_price_option1 - final_price_option2

theorem additional_savings_correct : additional_savings = 345 :=
by
  sorry

end NUMINAMATH_GPT_additional_savings_correct_l206_20682


namespace NUMINAMATH_GPT_ratio_a7_b7_l206_20692

variable (a b : ℕ → ℝ)
variable (S T : ℕ → ℝ)

-- Given conditions
axiom sum_S : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * a 2) -- Formula for sum of arithmetic series
axiom sum_T : ∀ n, T n = (n / 2) * (2 * b 1 + (n - 1) * b 2) -- Formula for sum of arithmetic series
axiom ratio_ST : ∀ n, S n / T n = (2 * n + 1) / (n + 3)

-- Prove the ratio of seventh terms
theorem ratio_a7_b7 : a 7 / b 7 = 27 / 16 :=
by
  sorry

end NUMINAMATH_GPT_ratio_a7_b7_l206_20692


namespace NUMINAMATH_GPT_least_possible_students_l206_20604

def TotalNumberOfStudents : ℕ := 35
def NumberOfStudentsWithBrownEyes : ℕ := 15
def NumberOfStudentsWithLunchBoxes : ℕ := 25
def NumberOfStudentsWearingGlasses : ℕ := 10

theorem least_possible_students (TotalNumberOfStudents NumberOfStudentsWithBrownEyes NumberOfStudentsWithLunchBoxes NumberOfStudentsWearingGlasses : ℕ) :
  ∃ n, n = 5 :=
sorry

end NUMINAMATH_GPT_least_possible_students_l206_20604


namespace NUMINAMATH_GPT_inequality_proving_l206_20697

theorem inequality_proving (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x^2 + y^2 + z^2 = 1) :
  (1 / x + 1 / y + 1 / z) - (x + y + z) ≥ 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proving_l206_20697


namespace NUMINAMATH_GPT_largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l206_20625

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_6 :
  ∃ x : ℤ, x < 100 ∧ (∃ n : ℤ, x = 6 * n + 4) ∧ x = 94 := sorry

end NUMINAMATH_GPT_largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l206_20625


namespace NUMINAMATH_GPT_value_of_a_l206_20634

theorem value_of_a (a : ℝ) (A : Set ℝ) (h : ∀ x, x ∈ A ↔ |x - a| < 1) : A = Set.Ioo 1 3 → a = 2 :=
by
  intro ha
  have : Set.Ioo 1 3 = {x | ∃ y, y ∈ Set.Ioi (1 : ℝ) ∧ y ∈ Set.Iio (3 : ℝ)} := by sorry
  sorry

end NUMINAMATH_GPT_value_of_a_l206_20634


namespace NUMINAMATH_GPT_unique_and_double_solutions_l206_20647

theorem unique_and_double_solutions (a : ℝ) :
  (∃ (x : ℝ), 5 + |x - 2| = a ∧ ∀ y, 5 + |y - 2| = a → y = x ∧ 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 7 - |2*x1 + 6| = a ∧ 7 - |2*x2 + 6| = a)) ∨
  (∃ (x : ℝ), 7 - |2*x + 6| = a ∧ ∀ y, 7 - |2*y + 6| = a → y = x ∧ 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 5 + |x1 - 2| = a ∧ 5 + |x2 - 2| = a)) ↔ a = 5 ∨ a = 7 :=
by
  sorry

end NUMINAMATH_GPT_unique_and_double_solutions_l206_20647


namespace NUMINAMATH_GPT_united_telephone_additional_charge_l206_20688

theorem united_telephone_additional_charge :
  ∃ x : ℝ, 
    (11 + 20 * x = 16) ↔ (x = 0.25) := by
  sorry

end NUMINAMATH_GPT_united_telephone_additional_charge_l206_20688


namespace NUMINAMATH_GPT_number_notebooks_in_smaller_package_l206_20673

theorem number_notebooks_in_smaller_package 
  (total_notebooks : ℕ)
  (large_packs : ℕ)
  (notebooks_per_large_pack : ℕ)
  (condition_1 : total_notebooks = 69)
  (condition_2 : large_packs = 7)
  (condition_3 : notebooks_per_large_pack = 7)
  (condition_4 : ∃ x : ℕ, x < 7 ∧ (total_notebooks - (large_packs * notebooks_per_large_pack)) % x = 0) :
  ∃ x : ℕ, x < 7 ∧ x = 5 := 
by 
  sorry

end NUMINAMATH_GPT_number_notebooks_in_smaller_package_l206_20673


namespace NUMINAMATH_GPT_inequality_abc_l206_20698

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l206_20698


namespace NUMINAMATH_GPT_sufficient_condition_of_necessary_condition_l206_20671

-- Define the necessary condition
def necessary_condition (A B : Prop) : Prop := A → B

-- The proof problem statement
theorem sufficient_condition_of_necessary_condition
  {A B : Prop} (h : necessary_condition A B) : necessary_condition A B :=
by
  exact h

end NUMINAMATH_GPT_sufficient_condition_of_necessary_condition_l206_20671


namespace NUMINAMATH_GPT_machine_copies_l206_20680

theorem machine_copies (x : ℕ) (h1 : ∀ t : ℕ, t = 30 → 30 * t = 900)
  (h2 : 900 + 30 * 30 = 2550) : x = 55 :=
by
  sorry

end NUMINAMATH_GPT_machine_copies_l206_20680


namespace NUMINAMATH_GPT_ratio_of_areas_l206_20638

theorem ratio_of_areas (r s_3 s_2 : ℝ) (h1 : s_3^2 = r^2) (h2 : s_2^2 = 2 * r^2) :
  (s_3^2 / s_2^2) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l206_20638


namespace NUMINAMATH_GPT_fraction_numerator_l206_20617

theorem fraction_numerator (x : ℚ) 
  (h1 : ∃ (n : ℚ), n = 4 * x - 9) 
  (h2 : x / (4 * x - 9) = 3 / 4) 
  : x = 27 / 8 := sorry

end NUMINAMATH_GPT_fraction_numerator_l206_20617


namespace NUMINAMATH_GPT_hyperbola_h_k_a_b_sum_l206_20603

noncomputable def h : ℝ := 1
noncomputable def k : ℝ := -3
noncomputable def a : ℝ := 3
noncomputable def c : ℝ := 3 * Real.sqrt 5
noncomputable def b : ℝ := 6

theorem hyperbola_h_k_a_b_sum :
  h + k + a + b = 7 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_h_k_a_b_sum_l206_20603


namespace NUMINAMATH_GPT_find_monthly_growth_rate_l206_20675

-- Define all conditions.
variables (March_sales May_sales : ℝ) (monthly_growth_rate : ℝ)

-- The conditions from the given problem
def initial_sales (March_sales : ℝ) : Prop := March_sales = 4 * 10^6
def final_sales (May_sales : ℝ) : Prop := May_sales = 9 * 10^6
def growth_occurred (March_sales May_sales : ℝ) (monthly_growth_rate : ℝ) : Prop :=
  May_sales = March_sales * (1 + monthly_growth_rate)^2

-- The Lean 4 theorem to be proven.
theorem find_monthly_growth_rate 
  (h1 : initial_sales March_sales) 
  (h2 : final_sales May_sales) 
  (h3 : growth_occurred March_sales May_sales monthly_growth_rate) : 
  400 * (1 + monthly_growth_rate)^2 = 900 := 
sorry

end NUMINAMATH_GPT_find_monthly_growth_rate_l206_20675


namespace NUMINAMATH_GPT_eval_fraction_expr_l206_20636

theorem eval_fraction_expr :
  (2 ^ 2010 * 3 ^ 2012) / (6 ^ 2011) = 3 / 2 := 
sorry

end NUMINAMATH_GPT_eval_fraction_expr_l206_20636


namespace NUMINAMATH_GPT_find_m_l206_20633

open Real

def vec := (ℝ × ℝ)

def dot_product (v1 v2 : vec) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def a : vec := (-1, 2)
def b (m : ℝ) : vec := (3, m)
def sum (m : ℝ) : vec := (a.1 + (b m).1, a.2 + (b m).2)

theorem find_m (m : ℝ) (h : dot_product a (sum m) = 0) : m = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_l206_20633


namespace NUMINAMATH_GPT_beautiful_ratio_l206_20629

theorem beautiful_ratio (A B C : Type) (l1 l2 b : ℕ) 
  (h : l1 + l2 + b = 20) (h1 : l1 = 8 ∨ l2 = 8 ∨ b = 8) :
  (b / l1 = 1/2) ∨ (b / l2 = 1/2) ∨ (l1 / l2 = 4/3) ∨ (l2 / l1 = 4/3) :=
by
  sorry

end NUMINAMATH_GPT_beautiful_ratio_l206_20629


namespace NUMINAMATH_GPT_ratio_of_fifteenth_terms_l206_20687

def S (n : ℕ) : ℝ := sorry
def T (n : ℕ) : ℝ := sorry
def a (n : ℕ) : ℝ := sorry
def b (n : ℕ) : ℝ := sorry

theorem ratio_of_fifteenth_terms 
  (h1: ∀ n, S n / T n = (5 * n + 3) / (3 * n + 35))
  (h2: ∀ n, a n = S n) -- Example condition
  (h3: ∀ n, b n = T n) -- Example condition
  : (a 15 / b 15) = 59 / 57 := 
  by 
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_ratio_of_fifteenth_terms_l206_20687


namespace NUMINAMATH_GPT_part1_part2_l206_20699

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |x - 2|

theorem part1 : {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -4/3 ∨ x ≥ 0} := sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, 0 < x → f x + a * x - 1 > 0) → a > -5/2 := sorry

end NUMINAMATH_GPT_part1_part2_l206_20699


namespace NUMINAMATH_GPT_find_m_l206_20652

noncomputable def f (x m : ℝ) : ℝ := (x^2 + m*x) * Real.exp x

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

theorem find_m (m : ℝ) :
  is_monotonically_decreasing (f (m := m)) (-3/2) 1 ∧
  (-3/2)^2 + (m + 2)*(-3/2) + m = 0 ∧
  1^2 + (m + 2)*1 + m = 0 →
  m = -3/2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l206_20652


namespace NUMINAMATH_GPT_smallest_value_of_y_square_l206_20648

-- Let's define the conditions
variable (EF GH y : ℝ)

-- The given conditions of the problem
def is_isosceles_trapezoid (EF GH y : ℝ) : Prop :=
  EF = 100 ∧ GH = 25 ∧ y > 0

def has_tangent_circle (EF GH y : ℝ) : Prop :=
  is_isosceles_trapezoid EF GH y ∧ 
  ∃ P : ℝ, P = EF / 2

-- Main proof statement
theorem smallest_value_of_y_square (EF GH y : ℝ)
  (h1 : is_isosceles_trapezoid EF GH y)
  (h2 : has_tangent_circle EF GH y) :
  y^2 = 1875 :=
  sorry

end NUMINAMATH_GPT_smallest_value_of_y_square_l206_20648


namespace NUMINAMATH_GPT_lower_bound_of_range_of_expression_l206_20663

theorem lower_bound_of_range_of_expression :
  ∃ L, (∀ n : ℤ, L < 4*n + 7 → 4*n + 7 < 100) ∧
  (∃! n_min n_max : ℤ, 4*n_min + 7 = L ∧ 4*n_max + 7 = 99 ∧ (n_max - n_min + 1 = 25)) :=
sorry

end NUMINAMATH_GPT_lower_bound_of_range_of_expression_l206_20663


namespace NUMINAMATH_GPT_determine_operation_l206_20672

theorem determine_operation (a b c d : Int) : ((a - b) + c - (3 * 1) = d) → ((a - b) + 2 = 6) → (a - b = 4) :=
by
  sorry

end NUMINAMATH_GPT_determine_operation_l206_20672


namespace NUMINAMATH_GPT_isoscelesTriangleDistanceFromAB_l206_20668

-- Given definitions
def isoscelesTriangleAreaInsideEquilateral (t m c x : ℝ) : Prop :=
  let halfEquilateralAltitude := m / 2
  let equilateralTriangleArea := (c^2 * (Real.sqrt 3)) / 4
  let equalsAltitudeCondition := x = m / 2
  let distanceFormula := x = (m + Real.sqrt (m^2 - 4 * t * Real.sqrt 3)) / 2 ∨ x = (m - Real.sqrt (m^2 - 4 * t * Real.sqrt 3)) / 2
  (2 * t = halfEquilateralAltitude * c / 2) ∧ 
  equalsAltitudeCondition ∧ distanceFormula

-- The theorem to prove given the above definition
theorem isoscelesTriangleDistanceFromAB (t m c x : ℝ) :
  isoscelesTriangleAreaInsideEquilateral t m c x →
  x = (m + Real.sqrt (m^2 - 4 * t * Real.sqrt 3)) / 2 ∨ x = (m - Real.sqrt (m^2 - 4 * t * Real.sqrt 3)) / 2 :=
sorry

end NUMINAMATH_GPT_isoscelesTriangleDistanceFromAB_l206_20668


namespace NUMINAMATH_GPT_stream_speed_l206_20653

theorem stream_speed (v : ℝ) (h1 : ∀ d : ℝ, d > 0 → ((1:ℝ) / (5 - v) = 2 * (1 / (5 + v)))) : 
  v = 5 / 3 :=
by
  -- Variables and assumptions
  have h1 : ∀ d : ℝ, d > 0 → ((1:ℝ) / (5 - v) = 2 * (1 / (5 + v))) := sorry
  -- To prove
  sorry

end NUMINAMATH_GPT_stream_speed_l206_20653


namespace NUMINAMATH_GPT_solution_x_percentage_of_alcohol_l206_20616

variable (P : ℝ) -- percentage of alcohol by volume in solution x, in decimal form

theorem solution_x_percentage_of_alcohol :
  (0.30 : ℝ) * 200 + P * 200 = 0.20 * 400 → P = 0.10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solution_x_percentage_of_alcohol_l206_20616


namespace NUMINAMATH_GPT_cube_root_of_5_irrational_l206_20628

theorem cube_root_of_5_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p : ℚ)^3 = 5 * (q : ℚ)^3 := by
  sorry

end NUMINAMATH_GPT_cube_root_of_5_irrational_l206_20628


namespace NUMINAMATH_GPT_solve_y_l206_20651

theorem solve_y (y : ℝ) (h : 5 * y^(1/4) - 3 * (y / y^(3/4)) = 9 + y^(1/4)) : y = 6561 := 
by 
  sorry

end NUMINAMATH_GPT_solve_y_l206_20651


namespace NUMINAMATH_GPT_dice_tower_even_n_l206_20662

/-- Given that n standard dice are stacked in a vertical tower,
and the total visible dots on each of the four vertical walls are all odd,
prove that n must be even.
-/
theorem dice_tower_even_n (n : ℕ)
  (h : ∀ (S T : ℕ), (S + T = 7 * n → (S % 2 = 1 ∧ T % 2 = 1))) : n % 2 = 0 :=
by sorry

end NUMINAMATH_GPT_dice_tower_even_n_l206_20662


namespace NUMINAMATH_GPT_number_of_diagonals_intersections_l206_20676

theorem number_of_diagonals_intersections (n : ℕ) (h : n ≥ 4) : 
  (∃ (I : ℕ), I = (n * (n - 1) * (n - 2) * (n - 3)) / 24) :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_diagonals_intersections_l206_20676


namespace NUMINAMATH_GPT_function_below_x_axis_l206_20654

theorem function_below_x_axis (k : ℝ) :
  (∀ x : ℝ, (k^2 - k - 2) * x^2 - (k - 2) * x - 1 < 0) ↔ (-2 / 5 < k ∧ k ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_function_below_x_axis_l206_20654


namespace NUMINAMATH_GPT_black_white_ratio_l206_20620

theorem black_white_ratio :
  let original_black := 18
  let original_white := 39
  let replaced_black := original_black + 13
  let inner_border_black := (9^2 - 7^2)
  let outer_border_white := (11^2 - 9^2)
  let total_black := replaced_black + inner_border_black
  let total_white := original_white + outer_border_white
  let ratio_black_white := total_black / total_white
  ratio_black_white = 63 / 79 :=
sorry

end NUMINAMATH_GPT_black_white_ratio_l206_20620


namespace NUMINAMATH_GPT_problem_statement_l206_20642

theorem problem_statement (a b : ℝ) (C : ℝ) (sin_C : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : sin_C = (Real.sqrt 15) / 4) :
  Real.cos C = 1 / 4 :=
sorry

end NUMINAMATH_GPT_problem_statement_l206_20642


namespace NUMINAMATH_GPT_original_population_960_l206_20618

variable (original_population : ℝ)

def new_population_increased := original_population + 800
def new_population_decreased := 0.85 * new_population_increased original_population

theorem original_population_960 
  (h1: new_population_decreased original_population = new_population_increased original_population + 24) :
  original_population = 960 := 
by
  -- here comes the proof, but we are omitting it as per the instructions
  sorry

end NUMINAMATH_GPT_original_population_960_l206_20618


namespace NUMINAMATH_GPT_ayen_total_jog_time_l206_20658

def jog_time_weekday : ℕ := 30
def jog_time_tuesday : ℕ := jog_time_weekday + 5
def jog_time_friday : ℕ := jog_time_weekday + 25

def total_weekday_jog_time : ℕ := jog_time_weekday * 3
def total_jog_time : ℕ := total_weekday_jog_time + jog_time_tuesday + jog_time_friday

theorem ayen_total_jog_time : total_jog_time / 60 = 3 := by
  sorry

end NUMINAMATH_GPT_ayen_total_jog_time_l206_20658


namespace NUMINAMATH_GPT_root_power_division_l206_20696

noncomputable def root4 (a : ℝ) : ℝ := a^(1/4)
noncomputable def root6 (a : ℝ) : ℝ := a^(1/6)

theorem root_power_division : 
  (root4 7) / (root6 7) = 7^(1/12) :=
by sorry

end NUMINAMATH_GPT_root_power_division_l206_20696


namespace NUMINAMATH_GPT_range_of_m_l206_20619

theorem range_of_m (x y : ℝ) (m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hineq : ∀ x > 0, ∀ y > 0, 2 * y / x + 8 * x / y ≥ m^2 + 2 * m) : 
  -4 ≤ m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l206_20619


namespace NUMINAMATH_GPT_isosceles_trapezoid_larger_base_l206_20693

theorem isosceles_trapezoid_larger_base (AD BC AC : ℝ) (h1 : AD = 10) (h2 : BC = 6) (h3 : AC = 14) :
  ∃ (AB : ℝ), AB = 16 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_larger_base_l206_20693


namespace NUMINAMATH_GPT_jordan_rectangle_width_l206_20656

theorem jordan_rectangle_width
  (w : ℝ)
  (len_carol : ℝ := 5)
  (wid_carol : ℝ := 24)
  (len_jordan : ℝ := 12)
  (area_carol_eq_area_jordan : (len_carol * wid_carol) = (len_jordan * w)) :
  w = 10 := by
  sorry

end NUMINAMATH_GPT_jordan_rectangle_width_l206_20656


namespace NUMINAMATH_GPT_parallelogram_area_l206_20649

theorem parallelogram_area (base height : ℝ) (h_base : base = 12) (h_height : height = 10) :
  base * height = 120 :=
by
  rw [h_base, h_height]
  norm_num

end NUMINAMATH_GPT_parallelogram_area_l206_20649


namespace NUMINAMATH_GPT_min_ineq_l206_20694

theorem min_ineq (x : ℝ) (hx : x > 0) : 3*x + 1/x^2 ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_ineq_l206_20694


namespace NUMINAMATH_GPT_perimeter_of_square_l206_20655

theorem perimeter_of_square (a : Real) (h_a : a ^ 2 = 144) : 4 * a = 48 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_l206_20655


namespace NUMINAMATH_GPT_jerry_total_games_l206_20606

-- Conditions
def initial_games : ℕ := 7
def birthday_games : ℕ := 2

-- Statement
theorem jerry_total_games : initial_games + birthday_games = 9 := by sorry

end NUMINAMATH_GPT_jerry_total_games_l206_20606


namespace NUMINAMATH_GPT_original_selling_price_l206_20641

/-- A boy sells a book for some amount and he gets a loss of 10%.
To gain 10%, the selling price should be Rs. 550.
Prove that the original selling price of the book was Rs. 450. -/
theorem original_selling_price (CP : ℝ) (h1 : 1.10 * CP = 550) :
    0.90 * CP = 450 := 
sorry

end NUMINAMATH_GPT_original_selling_price_l206_20641


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l206_20679

open Set Real

def U : Set ℝ := univ
def A : Set ℝ := { x | 1 ≤ x ∧ x < 5 }
def B : Set ℝ := { x | 2 < x ∧ x < 8 }
def C (a : ℝ) : Set ℝ := { x | -a < x ∧ x ≤ a + 3 }

theorem problem_1 :
  (A ∪ B) = { x | 1 ≤ x ∧ x < 8 } :=
sorry

theorem problem_2 :
  (U \ A) ∩ B = { x | 5 ≤ x ∧ x < 8 } :=
sorry

theorem problem_3 (a : ℝ) (h : C a ∩ A = C a) :
  a ≤ -1 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l206_20679


namespace NUMINAMATH_GPT_fraction_unclaimed_l206_20635

def exists_fraction_unclaimed (x : ℕ) : Prop :=
  let claimed_by_Eva := (1 / 2 : ℚ) * x
  let remaining_after_Eva := x - claimed_by_Eva
  let claimed_by_Liam := (3 / 8 : ℚ) * x
  let remaining_after_Liam := remaining_after_Eva - claimed_by_Liam
  let claimed_by_Noah := (1 / 8 : ℚ) * remaining_after_Eva
  let remaining_after_Noah := remaining_after_Liam - claimed_by_Noah
  remaining_after_Noah / x = (75 / 128 : ℚ)

theorem fraction_unclaimed {x : ℕ} : exists_fraction_unclaimed x :=
by
  sorry

end NUMINAMATH_GPT_fraction_unclaimed_l206_20635


namespace NUMINAMATH_GPT_julia_garden_area_l206_20669

theorem julia_garden_area
  (length perimeter walk_distance : ℝ)
  (h_length : length * 30 = walk_distance)
  (h_perimeter : perimeter * 12 = walk_distance)
  (h_perimeter_def : perimeter = 2 * (length + width))
  (h_walk_distance : walk_distance = 1500) :
  (length * width = 625) :=
by
  sorry

end NUMINAMATH_GPT_julia_garden_area_l206_20669


namespace NUMINAMATH_GPT_correct_statement_C_l206_20684

def V_m_rho_relation (V m ρ : ℝ) : Prop :=
  V = m / ρ

theorem correct_statement_C (V m ρ : ℝ) (h : ρ ≠ 0) : 
  ((∃ k : ℝ, k = ρ ∧ ∀ V' m' : ℝ, V' = m' / k → V' ≠ V) ∧ 
  (∃ v_var v_var', v_var = V ∧ v_var' = m ∧ V = m / ρ) →
  (∃ ρ_const : ℝ, ρ_const = ρ)) :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_C_l206_20684


namespace NUMINAMATH_GPT_katya_total_notebooks_l206_20666

-- Definitions based on the conditions provided
def cost_per_notebook : ℕ := 4
def total_rubles : ℕ := 150
def stickers_for_free_notebook : ℕ := 5
def initial_stickers : ℕ := total_rubles / cost_per_notebook

-- Hypothesis stating the total notebooks Katya can obtain
theorem katya_total_notebooks : initial_stickers + (initial_stickers / stickers_for_free_notebook) + 
    ((initial_stickers % stickers_for_free_notebook + (initial_stickers / stickers_for_free_notebook)) / stickers_for_free_notebook) +
    (((initial_stickers % stickers_for_free_notebook + (initial_stickers / stickers_for_free_notebook)) % stickers_for_free_notebook + 1) / stickers_for_free_notebook) = 46 :=
by
  sorry

end NUMINAMATH_GPT_katya_total_notebooks_l206_20666


namespace NUMINAMATH_GPT_LukaNeeds24CupsOfWater_l206_20683

theorem LukaNeeds24CupsOfWater
  (L S W : ℕ)
  (h1 : S = 2 * L)
  (h2 : W = 4 * S)
  (h3 : L = 3) :
  W = 24 := by
  sorry

end NUMINAMATH_GPT_LukaNeeds24CupsOfWater_l206_20683


namespace NUMINAMATH_GPT_param_line_segment_l206_20643

theorem param_line_segment:
  ∃ (a b c d : ℤ), b = 1 ∧ d = -3 ∧ a + b = -4 ∧ c + d = 9 ∧ a^2 + b^2 + c^2 + d^2 = 179 :=
by
  -- Here, you can use sorry to indicate that proof steps are not required as requested
  sorry

end NUMINAMATH_GPT_param_line_segment_l206_20643


namespace NUMINAMATH_GPT_triangle_equilateral_l206_20609

theorem triangle_equilateral
  (a b c : ℝ)
  (h : a^4 + b^4 + c^4 - a^2 * b^2 - b^2 * c^2 - a^2 * c^2 = 0) :
  a = b ∧ b = c ∧ a = c := 
by
  sorry

end NUMINAMATH_GPT_triangle_equilateral_l206_20609


namespace NUMINAMATH_GPT_number_of_students_taking_french_l206_20624

def total_students : ℕ := 79
def students_taking_german : ℕ := 22
def students_taking_both : ℕ := 9
def students_not_enrolled_in_either : ℕ := 25

theorem number_of_students_taking_french :
  ∃ F : ℕ, (total_students = F + students_taking_german - students_taking_both + students_not_enrolled_in_either) ∧ F = 41 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_taking_french_l206_20624


namespace NUMINAMATH_GPT_Gracie_height_is_correct_l206_20615

-- Given conditions
def Griffin_height : ℤ := 61
def Grayson_height : ℤ := Griffin_height + 2
def Gracie_height : ℤ := Grayson_height - 7

-- The proof problem: Prove that Gracie's height is 56 inches.
theorem Gracie_height_is_correct : Gracie_height = 56 := by
  sorry

end NUMINAMATH_GPT_Gracie_height_is_correct_l206_20615


namespace NUMINAMATH_GPT_each_worker_paid_40_l206_20608

variable (n_orchids : ℕ) (price_per_orchid : ℕ)
variable (n_money_plants : ℕ) (price_per_money_plant : ℕ)
variable (new_pots_cost : ℕ) (leftover_money : ℕ)
variable (n_workers : ℕ)

noncomputable def total_earnings : ℤ :=
  n_orchids * price_per_orchid + n_money_plants * price_per_money_plant

noncomputable def total_spent : ℤ :=
  new_pots_cost + leftover_money

noncomputable def amount_paid_to_workers : ℤ :=
  total_earnings n_orchids price_per_orchid n_money_plants price_per_money_plant - 
  total_spent new_pots_cost leftover_money

noncomputable def amount_paid_to_each_worker : ℤ :=
  amount_paid_to_workers n_orchids price_per_orchid n_money_plants price_per_money_plant 
    new_pots_cost leftover_money / n_workers

theorem each_worker_paid_40 :
  amount_paid_to_each_worker 20 50 15 25 150 1145 2 = 40 := by
  sorry

end NUMINAMATH_GPT_each_worker_paid_40_l206_20608


namespace NUMINAMATH_GPT_isosceles_triangle_sin_cos_rational_l206_20613

theorem isosceles_triangle_sin_cos_rational
  (a h : ℤ) -- Given BC and AD as integers
  (c : ℚ)  -- AB = AC = c
  (ha : 4 * c^2 = 4 * h^2 + a^2) : -- From c^2 = h^2 + (a^2 / 4)
  ∃ (sinA cosA : ℚ), 
    sinA = (a * h) / (h^2 + (a^2 / 4)) ∧
    cosA = (2 * h^2) / (h^2 + (a^2 / 4)) - 1 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_sin_cos_rational_l206_20613


namespace NUMINAMATH_GPT_quadratic_transformation_l206_20639

theorem quadratic_transformation (p q r : ℝ) :
  (∀ x : ℝ, p * x^2 + q * x + r = 3 * (x - 5)^2 + 15) →
  (∀ x : ℝ, 4 * p * x^2 + 4 * q * x + 4 * r = 12 * (x - 5)^2 + 60) :=
by
  intro h
  exact sorry

end NUMINAMATH_GPT_quadratic_transformation_l206_20639


namespace NUMINAMATH_GPT_anna_ate_cupcakes_l206_20644

-- Given conditions
def total_cupcakes : Nat := 60
def cupcakes_given_away (total : Nat) : Nat := (4 * total) / 5
def cupcakes_remaining (total : Nat) : Nat := total - cupcakes_given_away total
def anna_cupcakes_left : Nat := 9

-- Proving the number of cupcakes Anna ate
theorem anna_ate_cupcakes : cupcakes_remaining total_cupcakes - anna_cupcakes_left = 3 := by
  sorry

end NUMINAMATH_GPT_anna_ate_cupcakes_l206_20644


namespace NUMINAMATH_GPT_probability_both_red_is_one_fourth_l206_20685

noncomputable def probability_of_both_red (total_cards : ℕ) (red_cards : ℕ) (draws : ℕ) : ℚ :=
  (red_cards / total_cards) ^ draws

theorem probability_both_red_is_one_fourth :
  probability_of_both_red 52 26 2 = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_probability_both_red_is_one_fourth_l206_20685


namespace NUMINAMATH_GPT_find_distance_AC_l206_20691

noncomputable def distance_AC : ℝ :=
  let speed := 25  -- km per hour
  let angleA := 30  -- degrees
  let angleB := 135 -- degrees
  let distanceBC := 25 -- km
  (distanceBC * Real.sin (angleB * Real.pi / 180)) / (Real.sin (angleA * Real.pi / 180))

theorem find_distance_AC :
  distance_AC = 25 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_distance_AC_l206_20691


namespace NUMINAMATH_GPT_find_n_l206_20686

theorem find_n : ∃ n : ℕ, 5^3 - 7 = 6^2 + n ∧ n = 82 := by
  use 82
  sorry

end NUMINAMATH_GPT_find_n_l206_20686


namespace NUMINAMATH_GPT_palindrome_digital_clock_l206_20610

theorem palindrome_digital_clock (no_leading_zero : ∀ h : ℕ, h < 10 → ¬ ∃ h₂ : ℕ, h₂ = h * 1000)
                                 (max_hour : ∀ h : ℕ, h ≥ 24 → false) :
  ∃ n : ℕ, n = 61 := by
  sorry

end NUMINAMATH_GPT_palindrome_digital_clock_l206_20610


namespace NUMINAMATH_GPT_shelves_needed_l206_20660

variable (total_books : Nat) (books_taken : Nat) (books_per_shelf : Nat)

theorem shelves_needed (h1 : total_books = 14) 
                       (h2 : books_taken = 2) 
                       (h3 : books_per_shelf = 3) : 
    (total_books - books_taken) / books_per_shelf = 4 := by
  sorry

end NUMINAMATH_GPT_shelves_needed_l206_20660


namespace NUMINAMATH_GPT_tangent_line_intersecting_lines_l206_20602

variable (x y : ℝ)

-- Definition of the circle
def circle_C : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Definition of the point
def point_A : Prop := x = 1 ∧ y = 0

-- (I) Prove that if l is tangent to circle C and passes through A, l is 3x - 4y - 3 = 0
theorem tangent_line (l : ℝ → ℝ) (h : ∀ x, l x = k * (x - 1)) :
  (∀ {x y}, circle_C x y → 3 * x - 4 * y - 3 = 0) :=
by
  sorry

-- (II) Prove that the maximum area of triangle CPQ intersecting circle C is 2, and l's equations are y = 7x - 7 or y = x - 1
theorem intersecting_lines (k : ℝ) :
  (∃ x y, circle_C x y ∧ point_A x y) →
  (∃ k : ℝ, k = 7 ∨ k = 1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_intersecting_lines_l206_20602


namespace NUMINAMATH_GPT_line_length_after_erasure_l206_20690

-- Defining the initial and erased lengths
def initial_length_cm : ℕ := 100
def erased_length_cm : ℕ := 33

-- The statement we need to prove
theorem line_length_after_erasure : initial_length_cm - erased_length_cm = 67 := by
  sorry

end NUMINAMATH_GPT_line_length_after_erasure_l206_20690


namespace NUMINAMATH_GPT_marilyn_ends_up_with_55_caps_l206_20657

def marilyn_initial_caps := 165
def caps_shared_with_nancy := 78
def caps_received_from_charlie := 23

def remaining_caps (initial caps_shared caps_received: ℕ) :=
  initial - caps_shared + caps_received

def caps_given_away (total_caps: ℕ) :=
  total_caps / 2

def final_caps (initial caps_shared caps_received: ℕ) :=
  remaining_caps initial caps_shared caps_received - caps_given_away (remaining_caps initial caps_shared caps_received)

theorem marilyn_ends_up_with_55_caps :
  final_caps marilyn_initial_caps caps_shared_with_nancy caps_received_from_charlie = 55 :=
by
  sorry

end NUMINAMATH_GPT_marilyn_ends_up_with_55_caps_l206_20657


namespace NUMINAMATH_GPT_overall_average_tickets_sold_l206_20627

variable {M : ℕ} -- number of male members
variable {F : ℕ} -- number of female members
variable (male_to_female_ratio : M * 2 = F) -- 1:2 ratio
variable (average_female : ℕ) (average_male : ℕ) -- average tickets sold by female/male members
variable (total_tickets_female : F * average_female = 70 * F) -- Total tickets sold by female members
variable (total_tickets_male : M * average_male = 58 * M) -- Total tickets sold by male members

-- The overall average number of raffle tickets sold per member is 66.
theorem overall_average_tickets_sold 
  (h1 : 70 * F + 58 * M = 198 * M) -- total tickets sold
  (h2 : M + F = 3 * M) -- total number of members
  : (70 * F + 58 * M) / (M + F) = 66 := by
  sorry

end NUMINAMATH_GPT_overall_average_tickets_sold_l206_20627


namespace NUMINAMATH_GPT_fraction_zero_when_x_is_three_l206_20612

theorem fraction_zero_when_x_is_three (x : ℝ) (h : x ≠ -3) : (x^2 - 9) / (x + 3) = 0 ↔ x = 3 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_zero_when_x_is_three_l206_20612


namespace NUMINAMATH_GPT_identify_stolen_treasure_l206_20623

-- Define the magic square arrangement
def magic_square (bags : ℕ → ℕ) :=
  bags 0 + bags 1 + bags 2 = 15 ∧
  bags 3 + bags 4 + bags 5 = 15 ∧
  bags 6 + bags 7 + bags 8 = 15 ∧
  bags 0 + bags 3 + bags 6 = 15 ∧
  bags 1 + bags 4 + bags 7 = 15 ∧
  bags 2 + bags 5 + bags 8 = 15 ∧
  bags 0 + bags 4 + bags 8 = 15 ∧
  bags 2 + bags 4 + bags 6 = 15

-- Define the stolen treasure detection function
def stolen_treasure (bags : ℕ → ℕ) : Prop :=
  ∃ altered_bag_idx : ℕ, (bags altered_bag_idx ≠ altered_bag_idx + 1)

-- The main theorem
theorem identify_stolen_treasure (bags : ℕ → ℕ) (h_magic_square : magic_square bags) : ∃ altered_bag_idx : ℕ, stolen_treasure bags :=
sorry

end NUMINAMATH_GPT_identify_stolen_treasure_l206_20623


namespace NUMINAMATH_GPT_find_d_l206_20670

variable (x y d : ℤ)

-- Condition from the problem
axiom condition1 : (7 * x + 4 * y) / (x - 2 * y) = 13

-- The main proof goal
theorem find_d : x = 5 * y → x / (2 * y) = d / 2 → d = 5 :=
by
  intro h1 h2
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_d_l206_20670


namespace NUMINAMATH_GPT_find_fx_sum_roots_l206_20689

noncomputable def f : ℝ → ℝ
| x => if x = 2 then 1 else Real.log (abs (x - 2))

theorem find_fx_sum_roots
  (b c : ℝ)
  (x1 x2 x3 x4 x5 : ℝ)
  (h : ∀ x, (f x) ^ 2 + b * (f x) + c = 0)
  (h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x3 ≠ x4 ∧ x3 ≠ x5 ∧ x4 ≠ x5 ) :
  f (x1 + x2 + x3 + x4 + x5) = Real.log 8 :=
sorry

end NUMINAMATH_GPT_find_fx_sum_roots_l206_20689


namespace NUMINAMATH_GPT_ratio_of_b_to_a_l206_20640

open Real

theorem ratio_of_b_to_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a * sin (π / 5) + b * cos (π / 5)) / (a * cos (π / 5) - b * sin (π / 5)) = tan (8 * π / 15) 
  → b / a = sqrt 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ratio_of_b_to_a_l206_20640


namespace NUMINAMATH_GPT_maximumNumberOfGirls_l206_20626

theorem maximumNumberOfGirls {B : Finset ℕ} (hB : B.card = 5) :
  ∃ G : Finset ℕ, ∀ g ∈ G, ∃ b1 b2 : ℕ, b1 ≠ b2 ∧ b1 ∈ B ∧ b2 ∈ B ∧ dist g b1 = 5 ∧ dist g b2 = 5 ∧ G.card = 20 :=
sorry

end NUMINAMATH_GPT_maximumNumberOfGirls_l206_20626


namespace NUMINAMATH_GPT_middle_number_of_pairs_l206_20605

theorem middle_number_of_pairs (x y z : ℕ) (h1 : x + y = 15) (h2 : x + z = 18) (h3 : y + z = 21) : y = 9 := 
by
  sorry

end NUMINAMATH_GPT_middle_number_of_pairs_l206_20605


namespace NUMINAMATH_GPT_find_values_of_x_l206_20665

noncomputable def solution_x (x y : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ 
  x^2 + 1/y = 13 ∧ 
  y^2 + 1/x = 8 ∧ 
  (x = Real.sqrt 13 ∨ x = -Real.sqrt 13)

theorem find_values_of_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x^2 + 1/y = 13) (h2 : y^2 + 1/x = 8) : x = Real.sqrt 13 ∨ x = -Real.sqrt 13 :=
by { sorry }

end NUMINAMATH_GPT_find_values_of_x_l206_20665


namespace NUMINAMATH_GPT_find_y_coordinate_of_C_l206_20695

def point (x : ℝ) (y : ℝ) : Prop := y^2 = x + 4

def perp_slope (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) / (x2 - x1) * (y3 - y2) / (x3 - x2) = -1

def valid_y_coordinate_C (x0 : ℝ) : Prop :=
  x0 ≤ 0 ∨ 4 ≤ x0

theorem find_y_coordinate_of_C (x0 : ℝ) :
  (∀ (x y : ℝ), point x y) →
  (∃ (x2 y2 x3 y3 : ℝ), point x2 y2 ∧ point x3 y3 ∧ perp_slope 0 2 x2 y2 x3 y3) →
  valid_y_coordinate_C x0 :=
sorry

end NUMINAMATH_GPT_find_y_coordinate_of_C_l206_20695


namespace NUMINAMATH_GPT_portfolio_value_after_two_years_l206_20622

def initial_portfolio := 80

def first_year_growth_rate := 0.15
def add_after_6_months := 28
def withdraw_after_9_months := 10

def second_year_growth_first_6_months := 0.10
def second_year_decline_last_6_months := 0.04

def final_portfolio_value := 115.59

theorem portfolio_value_after_two_years 
  (initial_portfolio : ℝ)
  (first_year_growth_rate : ℝ)
  (add_after_6_months : ℕ)
  (withdraw_after_9_months : ℕ)
  (second_year_growth_first_6_months : ℝ)
  (second_year_decline_last_6_months : ℝ)
  (final_portfolio_value : ℝ) :
  (initial_portfolio = 80) →
  (first_year_growth_rate = 0.15) →
  (add_after_6_months = 28) →
  (withdraw_after_9_months = 10) →
  (second_year_growth_first_6_months = 0.10) →
  (second_year_decline_last_6_months = 0.04) →
  (final_portfolio_value = 115.59) :=
by
  sorry

end NUMINAMATH_GPT_portfolio_value_after_two_years_l206_20622


namespace NUMINAMATH_GPT_compare_negatives_l206_20678

theorem compare_negatives : -0.5 > -0.7 := 
by 
  exact sorry 

end NUMINAMATH_GPT_compare_negatives_l206_20678


namespace NUMINAMATH_GPT_build_bridge_l206_20645

/-- It took 6 days for 60 workers, all working together at the same rate, to build a bridge.
    Prove that if only 30 workers had been available, it would have taken 12 total days to build the bridge. -/
theorem build_bridge (days_60_workers : ℕ) (num_60_workers : ℕ) (same_rate : Prop) : 
  (days_60_workers = 6) → (num_60_workers = 60) → (same_rate = ∀ n m, n * days_60_workers = m * days_30_workers) → (days_30_workers = 12) :=
by
  sorry

end NUMINAMATH_GPT_build_bridge_l206_20645


namespace NUMINAMATH_GPT_mass_percentage_of_Cl_in_NH4Cl_l206_20614

-- Definition of the molar masses (conditions)
def molar_mass_N : ℝ := 14.01
def molar_mass_H : ℝ := 1.01
def molar_mass_Cl : ℝ := 35.45

-- Definition of the molar mass of NH4Cl
def molar_mass_NH4Cl : ℝ := molar_mass_N + 4 * molar_mass_H + molar_mass_Cl

-- The expected mass percentage of Cl in NH4Cl
def expected_mass_percentage_Cl : ℝ := 66.26

-- The proof statement
theorem mass_percentage_of_Cl_in_NH4Cl :
  (molar_mass_Cl / molar_mass_NH4Cl) * 100 = expected_mass_percentage_Cl :=
by 
  -- The body of the proof is omitted, as it is not necessary to provide the proof.
  sorry

end NUMINAMATH_GPT_mass_percentage_of_Cl_in_NH4Cl_l206_20614


namespace NUMINAMATH_GPT_jack_can_return_3900_dollars_l206_20601

/-- Jack's Initial Gift Card Values and Counts --/
def best_buy_card_value : ℕ := 500
def walmart_card_value : ℕ := 200
def initial_best_buy_cards : ℕ := 6
def initial_walmart_cards : ℕ := 9

/-- Jack's Sent Gift Card Counts --/
def sent_best_buy_cards : ℕ := 1
def sent_walmart_cards : ℕ := 2

/-- Calculate the remaining dollar value of Jack's gift cards. --/
def remaining_gift_cards_value : ℕ := 
  (initial_best_buy_cards * best_buy_card_value - sent_best_buy_cards * best_buy_card_value) +
  (initial_walmart_cards * walmart_card_value - sent_walmart_cards * walmart_card_value)

/-- Proving the remaining value of gift cards Jack can return is $3900. --/
theorem jack_can_return_3900_dollars : remaining_gift_cards_value = 3900 := by
  sorry

end NUMINAMATH_GPT_jack_can_return_3900_dollars_l206_20601


namespace NUMINAMATH_GPT_geometric_series_sum_l206_20621

theorem geometric_series_sum :
  let a := (2 : ℚ)
  let r := (1 / 3 : ℚ)
  let n := 6
  (a * (1 - r^n) / (1 - r) = 728 / 243) := 
by
  let a := (2 : ℚ)
  let r := (1 / 3 : ℚ)
  let n := 6
  show a * (1 - r^n) / (1 - r) = 728 / 243
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l206_20621


namespace NUMINAMATH_GPT_ceiling_floor_expression_l206_20607

theorem ceiling_floor_expression :
  (Int.ceil ((12:ℚ) / 5 * ((-19:ℚ) / 4 - 3)) - Int.floor (((12:ℚ) / 5) * Int.floor ((-19:ℚ) / 4)) = -6) :=
by 
  sorry

end NUMINAMATH_GPT_ceiling_floor_expression_l206_20607


namespace NUMINAMATH_GPT_annual_interest_rate_l206_20650

theorem annual_interest_rate
  (principal : ℝ) (monthly_payment : ℝ) (months : ℕ)
  (H1 : principal = 150) (H2 : monthly_payment = 13) (H3 : months = 12) :
  (monthly_payment * months - principal) / principal * 100 = 4 :=
by
  sorry

end NUMINAMATH_GPT_annual_interest_rate_l206_20650


namespace NUMINAMATH_GPT_championship_titles_l206_20632

theorem championship_titles {S T : ℕ} (h_S : S = 4) (h_T : T = 3) : S^T = 64 := by
  rw [h_S, h_T]
  norm_num

end NUMINAMATH_GPT_championship_titles_l206_20632


namespace NUMINAMATH_GPT_triangle_inequality_third_side_l206_20677

theorem triangle_inequality_third_side (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) :
  3 < x ∧ x < 13 ↔ (5 + 8 > x) ∧ (5 + x > 8) ∧ (8 + x > 5) :=
by 
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_triangle_inequality_third_side_l206_20677


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l206_20664

def M : Set ℝ := { x : ℝ | x^2 - x > 0 }
def N : Set ℝ := { x : ℝ | x ≥ 1 }

theorem intersection_of_M_and_N : M ∩ N = { x : ℝ | x > 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l206_20664


namespace NUMINAMATH_GPT_probability_heads_at_least_10_out_of_12_l206_20661

theorem probability_heads_at_least_10_out_of_12 (n m : Nat) (hn : n = 12) (hm : m = 10):
  let total_outcomes := 2^n
  let ways_10 := Nat.choose n m
  let ways_11 := Nat.choose n (m + 1)
  let ways_12 := Nat.choose n (m + 2)
  let successful_outcomes := ways_10 + ways_11 + ways_12
  total_outcomes = 4096 →
  successful_outcomes = 79 →
  (successful_outcomes : ℚ) / total_outcomes = 79 / 4096 :=
by
  sorry

end NUMINAMATH_GPT_probability_heads_at_least_10_out_of_12_l206_20661
