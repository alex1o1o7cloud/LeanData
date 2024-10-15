import Mathlib

namespace NUMINAMATH_GPT_solve_x_l335_33567

theorem solve_x 
  (x : ℝ) 
  (h : (2 / x) + (3 / x) / (6 / x) = 1.25) : 
  x = 8 / 3 := 
sorry

end NUMINAMATH_GPT_solve_x_l335_33567


namespace NUMINAMATH_GPT_no_positive_integers_exist_l335_33524

theorem no_positive_integers_exist 
  (a b c d : ℕ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (d_pos : 0 < d)
  (h₁ : a * b = c * d)
  (p : ℕ) 
  (hp : Nat.Prime p)
  (h₂ : a + b + c + d = p) : 
  False := 
by
  sorry

end NUMINAMATH_GPT_no_positive_integers_exist_l335_33524


namespace NUMINAMATH_GPT_angle_A_is_pi_div_3_length_b_l335_33534

open Real

theorem angle_A_is_pi_div_3
  (A B C : ℝ) (a b c : ℝ)
  (hABC : A + B + C = π)
  (m : ℝ × ℝ) (n : ℝ × ℝ)
  (hm : m = (sqrt 3, cos (π - A) - 1))
  (hn : n = (cos (π / 2 - A), 1))
  (horthogonal : m.1 * n.1 + m.2 * n.2 = 0) :
  A = π / 3 := 
sorry

theorem length_b 
  (A B : ℝ) (a b : ℝ)
  (hA : A = π / 3)
  (ha : a = 2)
  (hcosB : cos B = sqrt 3 / 3) :
  b = 4 * sqrt 2 / 3 :=
sorry

end NUMINAMATH_GPT_angle_A_is_pi_div_3_length_b_l335_33534


namespace NUMINAMATH_GPT_ratio_of_roots_ratio_l335_33579

noncomputable def sum_roots_first_eq (a b c : ℝ) := b / a
noncomputable def product_roots_first_eq (a b c : ℝ) := c / a
noncomputable def sum_roots_second_eq (a b c : ℝ) := a / c
noncomputable def product_roots_second_eq (a b c : ℝ) := b / c

theorem ratio_of_roots_ratio (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : c ≠ 0)
  (h3 : (b ^ 2 - 4 * a * c) > 0)
  (h4 : (a ^ 2 - 4 * c * b) > 0)
  (h5 : sum_roots_first_eq a b c ≥ 0)
  (h6 : product_roots_first_eq a b c = 9 * sum_roots_second_eq a b c) :
  sum_roots_first_eq a b c / product_roots_second_eq a b c = -3 :=
sorry

end NUMINAMATH_GPT_ratio_of_roots_ratio_l335_33579


namespace NUMINAMATH_GPT_g_value_l335_33576

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def g (ω φ x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ) - 1

theorem g_value (ω φ : ℝ) (h : ∀ x : ℝ, f ω φ (π / 4 - x) = f ω φ (π / 4 + x)) :
  g ω φ (π / 4) = -1 :=
sorry

end NUMINAMATH_GPT_g_value_l335_33576


namespace NUMINAMATH_GPT_circumradius_of_triangle_l335_33519

theorem circumradius_of_triangle (a b c : ℝ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 14) : 
  R = (35 * Real.sqrt 2) / 3 :=
by
  sorry

end NUMINAMATH_GPT_circumradius_of_triangle_l335_33519


namespace NUMINAMATH_GPT_full_time_employees_l335_33587

theorem full_time_employees (total_employees part_time_employees number_full_time_employees : ℕ)
  (h1 : total_employees = 65134)
  (h2 : part_time_employees = 2041)
  (h3 : number_full_time_employees = total_employees - part_time_employees)
  : number_full_time_employees = 63093 :=
by {
  sorry
}

end NUMINAMATH_GPT_full_time_employees_l335_33587


namespace NUMINAMATH_GPT_complement_of_angle_A_l335_33560

theorem complement_of_angle_A (A : ℝ) (h : A = 76) : 90 - A = 14 := by
  sorry

end NUMINAMATH_GPT_complement_of_angle_A_l335_33560


namespace NUMINAMATH_GPT_percent_other_sales_l335_33515

-- Define the given conditions
def s_brushes : ℝ := 0.45
def s_paints : ℝ := 0.28

-- Define the proof goal in Lean
theorem percent_other_sales :
  1 - (s_brushes + s_paints) = 0.27 := by
-- Adding the conditions to the proof environment
  sorry

end NUMINAMATH_GPT_percent_other_sales_l335_33515


namespace NUMINAMATH_GPT_check_independence_and_expected_value_l335_33531

noncomputable def contingency_table (students: ℕ) (pct_75 : ℕ) (pct_less10 : ℕ) (num_75_10 : ℕ) : (ℕ × ℕ) × (ℕ × ℕ) :=
  let num_75 := students * pct_75 / 100
  let num_less10 := students * pct_less10 / 100
  let num_75_less10 := num_75 - num_75_10
  let num_not75 := students - num_75
  let num_not75_less10 := num_less10 - num_75_less10
  let num_not75_10 := num_not75 - num_not75_less10
  ((num_not75_less10, num_75_less10), (num_not75_10, num_75_10))

noncomputable def chi_square_statistic (a b c d : ℕ) (n: ℕ) : ℚ :=
  (n * ((a * d - b * c)^2)) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem check_independence_and_expected_value :
  let students := 500
  let pct_75 := 30
  let pct_less10 := 50
  let num_75_10 := 100
  let ((num_not75_less10, num_75_less10), (num_not75_10, num_75_10)) := contingency_table students pct_75 pct_less10 num_75_10
  let chi2 := chi_square_statistic num_not75_less10 num_75_less10 num_not75_10 num_75_10 students
  let critical_value := 10.828
  let p0 := 1 / 84
  let p1 := 3 / 14
  let p2 := 15 / 28
  let p3 := 5 / 21
  let expected_x := 0 * p0 + 1 * p1 + 2 * p2 + 3 * p3
  (chi2 > critical_value) ∧ (expected_x = 2) :=
by 
  sorry

end NUMINAMATH_GPT_check_independence_and_expected_value_l335_33531


namespace NUMINAMATH_GPT_min_cuts_for_eleven_day_stay_max_days_with_n_cuts_l335_33583

-- Define the first problem
theorem min_cuts_for_eleven_day_stay : 
  (∀ (chain_len num_days : ℕ), chain_len = 11 ∧ num_days = 11 
  → (∃ (cuts : ℕ), cuts = 2)) := 
sorry

-- Define the second problem
theorem max_days_with_n_cuts : 
  (∀ (n chain_len days : ℕ), chain_len = (n + 1) * 2 ^ n - 1 
  → days = (n + 1) * 2 ^ n - 1) := 
sorry

end NUMINAMATH_GPT_min_cuts_for_eleven_day_stay_max_days_with_n_cuts_l335_33583


namespace NUMINAMATH_GPT_man_l335_33543

theorem man's_speed_upstream (v : ℝ) (downstream_speed : ℝ) (stream_speed : ℝ) :
  downstream_speed = v + stream_speed → stream_speed = 1 → downstream_speed = 10 → v - stream_speed = 8 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_man_l335_33543


namespace NUMINAMATH_GPT_equalize_nuts_l335_33591

open Nat

noncomputable def transfer (p1 p2 p3 : ℕ) : Prop :=
  ∃ (m1 m2 m3 : ℕ), 
    m1 ≤ p1 ∧ m1 ≤ p2 ∧ 
    m2 ≤ (p2 + m1) ∧ m2 ≤ p3 ∧ 
    m3 ≤ (p3 + m2) ∧ m3 ≤ (p1 - m1) ∧
    (p1 - m1 + m3 = 16) ∧ 
    (p2 + m1 - m2 = 16) ∧ 
    (p3 + m2 - m3 = 16)

theorem equalize_nuts : transfer 22 14 12 := 
  sorry

end NUMINAMATH_GPT_equalize_nuts_l335_33591


namespace NUMINAMATH_GPT_math_problem_l335_33574

theorem math_problem :
  ((-1)^2023 - (27^(1/3)) - (16^(1/2)) + (|1 - Real.sqrt 3|)) = -9 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l335_33574


namespace NUMINAMATH_GPT_base_5_conversion_correct_l335_33593

def base_5_to_base_10 : ℕ := 2 * 5^2 + 4 * 5^1 + 2 * 5^0

theorem base_5_conversion_correct : base_5_to_base_10 = 72 :=
by {
  -- Proof (not required in the problem statement)
  sorry
}

end NUMINAMATH_GPT_base_5_conversion_correct_l335_33593


namespace NUMINAMATH_GPT_angle_bisector_slope_l335_33500

-- Definitions of the conditions
def line1_slope := 2
def line2_slope := 4

-- The proof statement: Prove that the slope of the angle bisector is -12/7
theorem angle_bisector_slope : (line1_slope + line2_slope + Real.sqrt (line1_slope^2 + line2_slope^2 + 2 * line1_slope * line2_slope)) / 
                               (1 - line1_slope * line2_slope) = -12/7 :=
by
  sorry

end NUMINAMATH_GPT_angle_bisector_slope_l335_33500


namespace NUMINAMATH_GPT_simplify_expression_l335_33568

variable (x y : ℝ)

theorem simplify_expression : (-(3 * x * y - 2 * x ^ 2) - 2 * (3 * x ^ 2 - x * y)) = (-4 * x ^ 2 - x * y) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l335_33568


namespace NUMINAMATH_GPT_therapy_hours_l335_33554

variable (F A n : ℕ)
variable (h1 : F = A + 20)
variable (h2 : F + 2 * A = 188)
variable (h3 : F + A * (n - 1) = 300)

theorem therapy_hours : n = 5 := by
  sorry

end NUMINAMATH_GPT_therapy_hours_l335_33554


namespace NUMINAMATH_GPT_new_average_income_l335_33535

/-!
# Average Monthly Income Problem

## Problem Statement
Given:
1. The average monthly income of a family of 4 earning members was Rs. 735.
2. One of the earning members died, and the average income changed.
3. The income of the deceased member was Rs. 1170.

Prove that the new average monthly income of the family is Rs. 590.
-/

theorem new_average_income (avg_income : ℝ) (num_members : ℕ) (income_deceased : ℝ) (new_num_members : ℕ) 
  (h1 : avg_income = 735) 
  (h2 : num_members = 4) 
  (h3 : income_deceased = 1170) 
  (h4 : new_num_members = 3) : 
  (num_members * avg_income - income_deceased) / new_num_members = 590 := 
by 
  sorry

end NUMINAMATH_GPT_new_average_income_l335_33535


namespace NUMINAMATH_GPT_abc_plus_2p_zero_l335_33580

variable (a b c p : ℝ)

-- Define the conditions
def cond1 : Prop := a + 2 / b = p
def cond2 : Prop := b + 2 / c = p
def cond3 : Prop := c + 2 / a = p
def nonzero_and_distinct : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main statement we want to prove
theorem abc_plus_2p_zero (h1 : cond1 a b p) (h2 : cond2 b c p) (h3 : cond3 c a p) (h4 : nonzero_and_distinct a b c) : 
  a * b * c + 2 * p = 0 := 
by 
  sorry

end NUMINAMATH_GPT_abc_plus_2p_zero_l335_33580


namespace NUMINAMATH_GPT_empire_state_building_height_l335_33577

theorem empire_state_building_height (h_top_floor : ℕ) (h_antenna_spire : ℕ) (total_height : ℕ) :
  h_top_floor = 1250 ∧ h_antenna_spire = 204 ∧ total_height = h_top_floor + h_antenna_spire → total_height = 1454 :=
by
  sorry

end NUMINAMATH_GPT_empire_state_building_height_l335_33577


namespace NUMINAMATH_GPT_linear_function_solution_l335_33557

open Function

theorem linear_function_solution (f : ℝ → ℝ)
  (h_lin : ∃ k b, k ≠ 0 ∧ ∀ x, f x = k * x + b)
  (h_comp : ∀ x, f (f x) = 4 * x - 1) :
  (∀ x, f x = 2 * x - 1 / 3) ∨ (∀ x, f x = -2 * x + 1) :=
by
  sorry

end NUMINAMATH_GPT_linear_function_solution_l335_33557


namespace NUMINAMATH_GPT_meal_cost_before_tax_and_tip_l335_33563

theorem meal_cost_before_tax_and_tip (total_expenditure : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (base_meal_cost : ℝ):
  total_expenditure = 35.20 →
  tax_rate = 0.08 →
  tip_rate = 0.18 →
  base_meal_cost * (1 + tax_rate + tip_rate) = total_expenditure →
  base_meal_cost = 28 :=
by
  intros h_total h_tax h_tip h_eq
  sorry

end NUMINAMATH_GPT_meal_cost_before_tax_and_tip_l335_33563


namespace NUMINAMATH_GPT_find_salary_B_l335_33525

def salary_A : ℕ := 8000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000
def avg_salary : ℕ := 8000

theorem find_salary_B (S_B : ℕ) :
  (salary_A + S_B + salary_C + salary_D + salary_E) / 5 = avg_salary ↔ S_B = 5000 := by
  sorry

end NUMINAMATH_GPT_find_salary_B_l335_33525


namespace NUMINAMATH_GPT_find_radius_of_sphere_l335_33513

def radius_of_sphere_equal_to_cylinder_area (r : ℝ) (h : ℝ) (d : ℝ) : Prop :=
  (4 * Real.pi * r^2 = 2 * Real.pi * ((d / 2) * h))

theorem find_radius_of_sphere : ∃ r : ℝ, radius_of_sphere_equal_to_cylinder_area r 6 6 ∧ r = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_radius_of_sphere_l335_33513


namespace NUMINAMATH_GPT_money_left_over_l335_33501

theorem money_left_over 
  (num_books : ℕ) 
  (price_per_book : ℝ) 
  (num_records : ℕ) 
  (price_per_record : ℝ) 
  (total_books : num_books = 200) 
  (book_price : price_per_book = 1.5) 
  (total_records : num_records = 75) 
  (record_price : price_per_record = 3) :
  (num_books * price_per_book - num_records * price_per_record) = 75 :=
by 
  -- calculation
  sorry

end NUMINAMATH_GPT_money_left_over_l335_33501


namespace NUMINAMATH_GPT_two_person_subcommittees_l335_33573

def committee_size := 8
def sub_committee_size := 2
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem two_person_subcommittees : combination committee_size sub_committee_size = 28 := by
  sorry

end NUMINAMATH_GPT_two_person_subcommittees_l335_33573


namespace NUMINAMATH_GPT_max_profit_l335_33506

noncomputable def revenue (x : ℝ) : ℝ := 
  if (0 < x ∧ x ≤ 10) then 13.5 - (1 / 30) * x^2 
  else if (x > 10) then (168 / x) - (2000 / (3 * x^2)) 
  else 0

noncomputable def cost (x : ℝ) : ℝ := 
  20 + 5.4 * x

noncomputable def profit (x : ℝ) : ℝ := revenue x * x - cost x

theorem max_profit : 
  ∃ (x : ℝ), 0 < x ∧ x ≤ 10 ∧ (profit x = 8.1 * x - (1 / 30) * x^3 - 20) ∧ 
    (∀ (y : ℝ), 0 < y ∧ y ≤ 10 → profit y ≤ profit 9) ∧ 
    ∀ (z : ℝ), z > 10 → profit z ≤ profit 9 :=
by
  sorry

end NUMINAMATH_GPT_max_profit_l335_33506


namespace NUMINAMATH_GPT_find_unknown_number_l335_33512

theorem find_unknown_number :
  ∃ (x : ℝ), (786 * x) / 30 = 1938.8 → x = 74 :=
by 
  sorry

end NUMINAMATH_GPT_find_unknown_number_l335_33512


namespace NUMINAMATH_GPT_lena_glued_friends_pictures_l335_33527

-- Define the conditions
def clippings_per_friend : ℕ := 3
def glue_per_clipping : ℕ := 6
def total_glue : ℕ := 126

-- Define the proof problem statement
theorem lena_glued_friends_pictures : 
    ∃ (F : ℕ), F * (clippings_per_friend * glue_per_clipping) = total_glue ∧ F = 7 := 
by
  sorry

end NUMINAMATH_GPT_lena_glued_friends_pictures_l335_33527


namespace NUMINAMATH_GPT_sqrt_eq_sum_seven_l335_33565

open Real

theorem sqrt_eq_sum_seven (x : ℝ) (h : sqrt (64 - x^2) - sqrt (36 - x^2) = 4) :
    sqrt (64 - x^2) + sqrt (36 - x^2) = 7 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_eq_sum_seven_l335_33565


namespace NUMINAMATH_GPT_problem1_problem2_l335_33522

-- Definitions related to the given problem
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 9 / (Real.cos θ^2 + 9 * Real.sin θ^2)

def standard_curve (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Proving the standard equation of the curve
theorem problem1 (ρ θ : ℝ) (h : polar_curve ρ θ) : ∃ x y, standard_curve x y :=
  sorry

-- Proving the perpendicular condition and its consequence
theorem problem2 (ρ1 ρ2 α : ℝ)
  (hA : polar_curve ρ1 α)
  (hB : polar_curve ρ2 (α + π/2))
  (perpendicular : ∀ (A B : (ℝ × ℝ)), A ≠ B → A.1 * B.1 + A.2 * B.2 = 0) :
  (1 / ρ1^2) + (1 / ρ2^2) = 10 / 9 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l335_33522


namespace NUMINAMATH_GPT_relationship_between_abc_l335_33510

theorem relationship_between_abc (u v a b c : ℝ)
  (h1 : u - v = a) 
  (h2 : u^2 - v^2 = b)
  (h3 : u^3 - v^3 = c) : 
  3 * b ^ 2 + a ^ 4 = 4 * a * c :=
sorry

end NUMINAMATH_GPT_relationship_between_abc_l335_33510


namespace NUMINAMATH_GPT_middle_part_of_sum_is_120_l335_33564

theorem middle_part_of_sum_is_120 (x : ℚ) (h : 2 * x + x + (1 / 2) * x = 120) : 
  x = 240 / 7 := sorry

end NUMINAMATH_GPT_middle_part_of_sum_is_120_l335_33564


namespace NUMINAMATH_GPT_sqrt_mixed_number_l335_33585

theorem sqrt_mixed_number :
  (Real.sqrt (8 + 9/16)) = (Real.sqrt 137) / 4 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_mixed_number_l335_33585


namespace NUMINAMATH_GPT_larger_number_value_l335_33555

theorem larger_number_value (L S : ℕ) (h1 : L - S = 20775) (h2 : L = 23 * S + 143) : L = 21713 :=
sorry

end NUMINAMATH_GPT_larger_number_value_l335_33555


namespace NUMINAMATH_GPT_allan_has_4_more_balloons_than_jake_l335_33542

namespace BalloonProblem

def initial_balloons_allan : Nat := 6
def initial_balloons_jake : Nat := 2
def additional_balloons_jake : Nat := 3
def additional_balloons_allan : Nat := 4
def given_balloons_jake : Nat := 2
def given_balloons_allan : Nat := 3

def final_balloons_allan : Nat := (initial_balloons_allan + additional_balloons_allan) - given_balloons_allan
def final_balloons_jake : Nat := (initial_balloons_jake + additional_balloons_jake) - given_balloons_jake

theorem allan_has_4_more_balloons_than_jake :
  final_balloons_allan = final_balloons_jake + 4 :=
by
  -- proof is skipped with sorry
  sorry

end BalloonProblem

end NUMINAMATH_GPT_allan_has_4_more_balloons_than_jake_l335_33542


namespace NUMINAMATH_GPT_commencement_addresses_sum_l335_33518

noncomputable def addresses (S H L : ℕ) := 40

theorem commencement_addresses_sum
  (S H L : ℕ)
  (h1 : S = 12)
  (h2 : S = 2 * H)
  (h3 : L = S + 10) :
  S + H + L = addresses S H L :=
by
  sorry

end NUMINAMATH_GPT_commencement_addresses_sum_l335_33518


namespace NUMINAMATH_GPT_rectangle_area_l335_33549

variable (w l : ℕ)
variable (A : ℕ)
variable (H1 : l = 5 * w)
variable (H2 : 2 * l + 2 * w = 180)

theorem rectangle_area : A = 1125 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l335_33549


namespace NUMINAMATH_GPT_number_of_ways_to_choose_positions_l335_33595

-- Definition of the problem conditions
def number_of_people : ℕ := 8

-- Statement of the proof problem
theorem number_of_ways_to_choose_positions : 
  (number_of_people) * (number_of_people - 1) * (number_of_people - 2) = 336 := by
  -- skipping the proof itself
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_positions_l335_33595


namespace NUMINAMATH_GPT_polynomial_sum_l335_33586

noncomputable def p (x : ℝ) : ℝ := -2 * x^2 + 2 * x - 5
noncomputable def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
noncomputable def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = 12 * x - 12 := by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l335_33586


namespace NUMINAMATH_GPT_initial_candies_l335_33582

-- Define the conditions
def candies_given_older_sister : ℕ := 7
def candies_given_younger_sister : ℕ := 6
def candies_left : ℕ := 15

-- Conclude the initial number of candies
theorem initial_candies : (candies_given_older_sister + candies_given_younger_sister + candies_left) = 28 := by
  sorry

end NUMINAMATH_GPT_initial_candies_l335_33582


namespace NUMINAMATH_GPT_find_angle_BDC_l335_33526

theorem find_angle_BDC
  (CAB CAD DBA DBC : ℝ)
  (h1 : CAB = 40)
  (h2 : CAD = 30)
  (h3 : DBA = 75)
  (h4 : DBC = 25) :
  ∃ BDC : ℝ, BDC = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_BDC_l335_33526


namespace NUMINAMATH_GPT_max_S_is_9_l335_33572

-- Definitions based on the conditions
def a (n : ℕ) : ℤ := 28 - 3 * n
def S (n : ℕ) : ℤ := n * (25 + a n) / 2

-- The theorem to be proved
theorem max_S_is_9 : ∃ n : ℕ, n = 9 ∧ S n = 117 :=
by
  sorry

end NUMINAMATH_GPT_max_S_is_9_l335_33572


namespace NUMINAMATH_GPT_average_height_of_60_students_l335_33523

theorem average_height_of_60_students :
  (35 * 22 + 25 * 18) / 60 = 20.33 := 
sorry

end NUMINAMATH_GPT_average_height_of_60_students_l335_33523


namespace NUMINAMATH_GPT_negation_proposition_l335_33520

theorem negation_proposition (a b : ℝ) :
  (a * b ≠ 0) → (a ≠ 0 ∧ b ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l335_33520


namespace NUMINAMATH_GPT_students_in_both_clubs_l335_33566

variables (Total Students RoboticClub ScienceClub EitherClub BothClubs : ℕ)

theorem students_in_both_clubs
  (h1 : Total = 300)
  (h2 : RoboticClub = 80)
  (h3 : ScienceClub = 130)
  (h4 : EitherClub = 190)
  (h5 : EitherClub = RoboticClub + ScienceClub - BothClubs) :
  BothClubs = 20 :=
by
  sorry

end NUMINAMATH_GPT_students_in_both_clubs_l335_33566


namespace NUMINAMATH_GPT_find_d_l335_33503

theorem find_d (d : ℤ) :
  (∀ x : ℤ, 6 * x^3 + 19 * x^2 + d * x - 15 = 0) ->
  d = -32 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l335_33503


namespace NUMINAMATH_GPT_unattainable_value_of_y_l335_33584

theorem unattainable_value_of_y (x : ℚ) (h : x ≠ -5/4) :
  ¬ ∃ y : ℚ, y = (2 - 3 * x) / (4 * x + 5) ∧ y = -3/4 :=
by
  sorry

end NUMINAMATH_GPT_unattainable_value_of_y_l335_33584


namespace NUMINAMATH_GPT_no_month_5_mondays_and_5_thursdays_l335_33540

theorem no_month_5_mondays_and_5_thursdays (n : ℕ) (h : n = 28 ∨ n = 29 ∨ n = 30 ∨ n = 31) :
  ¬ (∃ (m : ℕ) (t : ℕ), m = 5 ∧ t = 5 ∧ 5 * (m + t) ≤ n) := by sorry

end NUMINAMATH_GPT_no_month_5_mondays_and_5_thursdays_l335_33540


namespace NUMINAMATH_GPT_number_of_candidates_is_9_l335_33569

-- Defining the problem
def num_ways_to_select_president_and_vp (n : ℕ) : ℕ :=
  n * (n - 1)

-- Main theorem statement
theorem number_of_candidates_is_9 (n : ℕ) (h : num_ways_to_select_president_and_vp n = 72) : n = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_candidates_is_9_l335_33569


namespace NUMINAMATH_GPT_evaluate_expression_l335_33541

theorem evaluate_expression (x : ℝ) : 
  (36 + 12 * x) ^ 2 - (12^2 * x^2 + 36^2) = 864 * x :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l335_33541


namespace NUMINAMATH_GPT_r_squared_sum_l335_33556

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end NUMINAMATH_GPT_r_squared_sum_l335_33556


namespace NUMINAMATH_GPT_employees_in_room_l335_33597

-- Define variables
variables (E : ℝ) (M : ℝ) (L : ℝ)

-- Given conditions
def condition1 : Prop := M = 0.99 * E
def condition2 : Prop := (M - L) / E = 0.98
def condition3 : Prop := L = 99.99999999999991

-- Prove statement
theorem employees_in_room (h1 : condition1 E M) (h2 : condition2 E M L) (h3 : condition3 L) : E = 10000 :=
by
  sorry

end NUMINAMATH_GPT_employees_in_room_l335_33597


namespace NUMINAMATH_GPT_intersection_count_l335_33561

theorem intersection_count :
  ∀ {x y : ℝ}, (2 * x - 2 * y + 4 = 0 ∨ 6 * x + 2 * y - 8 = 0) ∧ (y = -x^2 + 2 ∨ 4 * x - 10 * y + 14 = 0) → 
  (x ≠ 0 ∨ y ≠ 2) ∧ (x ≠ -1 ∨ y ≠ 1) ∧ (x ≠ 1 ∨ y ≠ -1) ∧ (x ≠ 2 ∨ y ≠ 2) → 
  ∃! (p : ℝ × ℝ), (p = (0, 2) ∨ p = (-1, 1) ∨ p = (1, -1) ∨ p = (2, 2)) := sorry

end NUMINAMATH_GPT_intersection_count_l335_33561


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l335_33553

-- Definitions extracted from the problem conditions
def isEllipse (k : ℝ) : Prop := (9 - k > 0) ∧ (k - 7 > 0) ∧ (9 - k ≠ k - 7)

-- The necessary but not sufficient condition for the ellipse equation
theorem necessary_but_not_sufficient : 
  (7 < k ∧ k < 9) → isEllipse k → (isEllipse k ↔ (7 < k ∧ k < 9)) := 
by 
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l335_33553


namespace NUMINAMATH_GPT_find_natural_number_l335_33504

theorem find_natural_number (n : ℕ) (k : ℤ) (h : 2^n + 3 = k^2) : n = 0 :=
sorry

end NUMINAMATH_GPT_find_natural_number_l335_33504


namespace NUMINAMATH_GPT_parallel_lines_m_l335_33538

theorem parallel_lines_m (m : ℝ) :
  (∀ x y : ℝ, x + 2 * m * y - 1 = 0 → (3 * m - 1) * x - m * y - 1 = 0)
  → m = 0 ∨ m = 1 / 6 := 
sorry

end NUMINAMATH_GPT_parallel_lines_m_l335_33538


namespace NUMINAMATH_GPT_tennis_tournament_l335_33590

theorem tennis_tournament (n : ℕ) (w m : ℕ) 
  (total_matches : ℕ)
  (women_wins men_wins : ℕ) :
  n + 2 * n = 3 * n →
  total_matches = (3 * n * (3 * n - 1)) / 2 →
  women_wins + men_wins = total_matches →
  women_wins / men_wins = 7 / 5 →
  n = 3 :=
by sorry

end NUMINAMATH_GPT_tennis_tournament_l335_33590


namespace NUMINAMATH_GPT_minimize_triangle_area_minimize_product_PA_PB_l335_33547

-- Define the initial conditions and geometry setup
def point (x y : ℝ) := (x, y)
def line_eq (a b : ℝ) := ∀ x y : ℝ, x / a + y / b = 1

-- Point P
def P := point 2 1

-- Condition: the line passes through point P and intersects the axes
def line_through_P (a b : ℝ) := line_eq a b ∧ (2 / a + 1 / b = 1) ∧ a > 2 ∧ b > 1

-- Prove that the line minimizing the area of triangle AOB is x + 2y - 4 = 0
theorem minimize_triangle_area (a b : ℝ) (h : line_through_P a b) :
  a = 4 ∧ b = 2 → line_eq 4 2 := 
sorry

-- Prove that the line minimizing the product |PA||PB| is x + y - 3 = 0
theorem minimize_product_PA_PB (a b : ℝ) (h : line_through_P a b) :
  a = 3 ∧ b = 3 → line_eq 3 3 := 
sorry

end NUMINAMATH_GPT_minimize_triangle_area_minimize_product_PA_PB_l335_33547


namespace NUMINAMATH_GPT_intersection_of_sets_l335_33509

theorem intersection_of_sets :
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | x < 0 ∨ x > 3}
  ∀ x, (x ∈ A ∧ x ∈ B) ↔ (-2 < x ∧ x < 0) :=
by
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | x < 0 ∨ x > 3}
  intro x
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l335_33509


namespace NUMINAMATH_GPT_chebyshev_substitution_even_chebyshev_substitution_odd_l335_33570

def T (n : ℕ) (x : ℝ) : ℝ := sorry -- Chebyshev polynomial of the first kind
def U (n : ℕ) (x : ℝ) : ℝ := sorry -- Chebyshev polynomial of the second kind

theorem chebyshev_substitution_even (k : ℕ) (α : ℝ) :
  T (2 * k) (Real.sin α) = (-1)^k * Real.cos ((2 * k) * α) ∧
  U ((2 * k) - 1) (Real.sin α) = (-1)^(k + 1) * (Real.sin ((2 * k) * α) / Real.cos α) :=
by
  sorry

theorem chebyshev_substitution_odd (k : ℕ) (α : ℝ) :
  T (2 * k + 1) (Real.sin α) = (-1)^k * Real.sin ((2 * k + 1) * α) ∧
  U (2 * k) (Real.sin α) = (-1)^k * (Real.cos ((2 * k + 1) * α) / Real.cos α) :=
by
  sorry

end NUMINAMATH_GPT_chebyshev_substitution_even_chebyshev_substitution_odd_l335_33570


namespace NUMINAMATH_GPT_inequality_div_two_l335_33530

theorem inequality_div_two (a b : ℝ) (h : a > b) : (a / 2) > (b / 2) :=
sorry

end NUMINAMATH_GPT_inequality_div_two_l335_33530


namespace NUMINAMATH_GPT_exists_p_for_q_l335_33507

noncomputable def sqrt_56 : ℝ := Real.sqrt 56
noncomputable def sqrt_58 : ℝ := Real.sqrt 58

theorem exists_p_for_q (q : ℕ) (hq : q > 0) (hq_ne_1 : q ≠ 1) (hq_ne_3 : q ≠ 3) :
  ∃ p : ℤ, sqrt_56 < (p : ℝ) / q ∧ (p : ℝ) / q < sqrt_58 :=
by sorry

end NUMINAMATH_GPT_exists_p_for_q_l335_33507


namespace NUMINAMATH_GPT_fraction_of_time_riding_at_15mph_l335_33562

variable (t_5 t_15 : ℝ)

-- Conditions
def no_stops : Prop := (t_5 ≠ 0 ∧ t_15 ≠ 0)
def average_speed (t_5 t_15 : ℝ) : Prop := (5 * t_5 + 15 * t_15) / (t_5 + t_15) = 10

-- Question to be proved
theorem fraction_of_time_riding_at_15mph (h1 : no_stops t_5 t_15) (h2 : average_speed t_5 t_15) :
  t_15 / (t_5 + t_15) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_fraction_of_time_riding_at_15mph_l335_33562


namespace NUMINAMATH_GPT_production_days_l335_33578

theorem production_days (n : ℕ) 
    (h1 : 70 * n + 90 = 75 * (n + 1)) : n = 3 := 
sorry

end NUMINAMATH_GPT_production_days_l335_33578


namespace NUMINAMATH_GPT_total_arrangements_l335_33575

def total_members : ℕ := 6
def days : ℕ := 3
def people_per_day : ℕ := 2

def A_cannot_on_14 (arrangement : ℕ → ℕ) : Prop :=
  ¬ arrangement 14 = 1

def B_cannot_on_16 (arrangement : ℕ → ℕ) : Prop :=
  ¬ arrangement 16 = 2

theorem total_arrangements (arrangement : ℕ → ℕ) :
  (∀ arrangement, A_cannot_on_14 arrangement ∧ B_cannot_on_16 arrangement) →
  (total_members.choose 2 * (total_members - 2).choose 2 - 
  2 * (total_members - 1).choose 1 * (total_members - 2).choose 2 +
  (total_members - 2).choose 1 * (total_members - 3).choose 1)
  = 42 := 
by
  sorry

end NUMINAMATH_GPT_total_arrangements_l335_33575


namespace NUMINAMATH_GPT_seven_by_seven_grid_partition_l335_33545

theorem seven_by_seven_grid_partition : 
  ∀ (x y : ℕ), 4 * x + 3 * y = 49 ∧ x + y ≥ 16 → x = 1 :=
by sorry

end NUMINAMATH_GPT_seven_by_seven_grid_partition_l335_33545


namespace NUMINAMATH_GPT_gross_pay_is_450_l335_33539

def net_pay : ℤ := 315
def taxes : ℤ := 135
def gross_pay : ℤ := net_pay + taxes

theorem gross_pay_is_450 : gross_pay = 450 := by
  sorry

end NUMINAMATH_GPT_gross_pay_is_450_l335_33539


namespace NUMINAMATH_GPT_sector_area_l335_33558

theorem sector_area (α : ℝ) (r : ℝ) (hα : α = (2 * Real.pi) / 3) (hr : r = Real.sqrt 3) :
  (1 / 2) * α * r ^ 2 = Real.pi := by
  sorry

end NUMINAMATH_GPT_sector_area_l335_33558


namespace NUMINAMATH_GPT_find_abc_l335_33528

theorem find_abc (a b c : ℕ) (k : ℕ) 
  (h1 : a = 2 * k) 
  (h2 : b = 3 * k) 
  (h3 : c = 4 * k) 
  (h4 : k ≠ 0)
  (h5 : 2 * a - b + c = 10) : 
  a = 4 ∧ b = 6 ∧ c = 8 :=
sorry

end NUMINAMATH_GPT_find_abc_l335_33528


namespace NUMINAMATH_GPT_total_difference_in_cards_l335_33529

theorem total_difference_in_cards (cards_chris : ℕ) (cards_charlie : ℕ) (cards_diana : ℕ) (cards_ethan : ℕ)
  (h_chris : cards_chris = 18)
  (h_charlie : cards_charlie = 32)
  (h_diana : cards_diana = 25)
  (h_ethan : cards_ethan = 40) :
  (cards_charlie - cards_chris) + (cards_diana - cards_chris) + (cards_ethan - cards_chris) = 43 := by
  sorry

end NUMINAMATH_GPT_total_difference_in_cards_l335_33529


namespace NUMINAMATH_GPT_complex_number_simplification_l335_33505

theorem complex_number_simplification (i : ℂ) (hi : i^2 = -1) : i - (1 / i) = 2 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_number_simplification_l335_33505


namespace NUMINAMATH_GPT_buddy_thursday_cards_l335_33548

-- Definitions from the given conditions
def monday_cards : ℕ := 30
def tuesday_cards : ℕ := monday_cards / 2
def wednesday_cards : ℕ := tuesday_cards + 12
def thursday_extra_cards : ℕ := tuesday_cards / 3
def thursday_cards : ℕ := wednesday_cards + thursday_extra_cards

-- Theorem to prove the total number of baseball cards on Thursday
theorem buddy_thursday_cards : thursday_cards = 32 :=
by
  -- Proof steps would go here, but we just provide the result for now
  sorry

end NUMINAMATH_GPT_buddy_thursday_cards_l335_33548


namespace NUMINAMATH_GPT_prime_divisor_congruent_one_mod_p_l335_33581

theorem prime_divisor_congruent_one_mod_p (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ q ∣ p^p - 1 ∧ q % p = 1 :=
sorry

end NUMINAMATH_GPT_prime_divisor_congruent_one_mod_p_l335_33581


namespace NUMINAMATH_GPT_joan_lost_balloons_l335_33511

theorem joan_lost_balloons :
  let initial_balloons := 9
  let current_balloons := 7
  let balloons_lost := initial_balloons - current_balloons
  balloons_lost = 2 :=
by
  sorry

end NUMINAMATH_GPT_joan_lost_balloons_l335_33511


namespace NUMINAMATH_GPT_combined_points_correct_l335_33594

-- Definitions for the points scored by each player
def points_Lemuel := 7 * 2 + 5 * 3 + 4
def points_Marcus := 4 * 2 + 6 * 3 + 7
def points_Kevin := 9 * 2 + 4 * 3 + 5
def points_Olivia := 6 * 2 + 3 * 3 + 6

-- Definition for the combined points scored by both teams
def combined_points := points_Lemuel + points_Marcus + points_Kevin + points_Olivia

-- Theorem statement to prove combined points equals 128
theorem combined_points_correct : combined_points = 128 :=
by
  -- Lean proof goes here
  sorry

end NUMINAMATH_GPT_combined_points_correct_l335_33594


namespace NUMINAMATH_GPT_clock_strike_time_l335_33544

theorem clock_strike_time (t : ℕ) (n m : ℕ) (I : ℕ) : 
  t = 12 ∧ n = 3 ∧ m = 6 ∧ 2 * I = t → (m - 1) * I = 30 := by 
  sorry

end NUMINAMATH_GPT_clock_strike_time_l335_33544


namespace NUMINAMATH_GPT_magician_earned_4_dollars_l335_33551

-- Define conditions
def price_per_deck := 2
def initial_decks := 5
def decks_left := 3

-- Define the number of decks sold
def decks_sold := initial_decks - decks_left

-- Define the total money earned
def money_earned := decks_sold * price_per_deck

-- Theorem to prove the money earned is 4 dollars
theorem magician_earned_4_dollars : money_earned = 4 := by
  sorry

end NUMINAMATH_GPT_magician_earned_4_dollars_l335_33551


namespace NUMINAMATH_GPT_integer_values_m_l335_33592

theorem integer_values_m (m x y : ℤ) (h1 : x - 2 * y = m) (h2 : 2 * x + 3 * y = 2 * m - 3)
    (h3 : 3 * x + y ≥ 0) (h4 : x + 5 * y < 0) : m = 1 ∨ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_integer_values_m_l335_33592


namespace NUMINAMATH_GPT_volume_of_water_in_prism_l335_33546

-- Define the given dimensions and conditions
def length_x := 20 -- cm
def length_y := 30 -- cm
def length_z := 40 -- cm
def angle := 30 -- degrees
def total_volume := 24 -- liters

-- The wet fraction of the upper surface
def wet_fraction := 1 / 4

-- Correct answer to be proven
def volume_water := 18.8 -- liters

theorem volume_of_water_in_prism :
  -- Given the conditions
  (length_x = 20) ∧ (length_y = 30) ∧ (length_z = 40) ∧ (angle = 30) ∧ (wet_fraction = 1 / 4) ∧ (total_volume = 24) →
  -- Prove that the volume of water is as calculated
  volume_water = 18.8 :=
sorry

end NUMINAMATH_GPT_volume_of_water_in_prism_l335_33546


namespace NUMINAMATH_GPT_functional_eqn_even_function_l335_33552

variable {R : Type*} [AddGroup R] (f : R → ℝ)

theorem functional_eqn_even_function
  (h_not_zero : ∃ x, f x ≠ 0)
  (h_func_eq : ∀ a b, f (a + b) + f (a - b) = 2 * f a + 2 * f b) :
  ∀ x, f (-x) = f x :=
by
  sorry

end NUMINAMATH_GPT_functional_eqn_even_function_l335_33552


namespace NUMINAMATH_GPT_find_amount_with_R_l335_33521

variable (P_amount Q_amount R_amount : ℝ)
variable (total_amount : ℝ) (r_has_twothirds : Prop)

noncomputable def amount_with_R (total_amount : ℝ) : ℝ :=
  let R_amount := 2 / 3 * (total_amount - R_amount)
  R_amount

theorem find_amount_with_R (P_amount Q_amount R_amount : ℝ) (total_amount : ℝ)
  (h_total : total_amount = 5000)
  (h_two_thirds : R_amount = 2 / 3 * (P_amount + Q_amount)) :
  R_amount = 2000 := by sorry

end NUMINAMATH_GPT_find_amount_with_R_l335_33521


namespace NUMINAMATH_GPT_fraction_equivalence_l335_33533

theorem fraction_equivalence : 
    (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 := by sorry

end NUMINAMATH_GPT_fraction_equivalence_l335_33533


namespace NUMINAMATH_GPT_product_eq_1280_l335_33588

axiom eq1 (a b c d : ℝ) : 2 * a + 4 * b + 6 * c + 8 * d = 48
axiom eq2 (a b c d : ℝ) : 4 * d + 2 * c = 2 * b
axiom eq3 (a b c d : ℝ) : 4 * b + 2 * c = 2 * a
axiom eq4 (a b c d : ℝ) : c - 2 = d
axiom eq5 (a b c d : ℝ) : d + b = 10

theorem product_eq_1280 (a b c d : ℝ) : 2 * a + 4 * b + 6 * c + 8 * d = 48 → 4 * d + 2 * c = 2 * b → 4 * b + 2 * c = 2 * a → c - 2 = d → d + b = 10 → a * b * c * d = 1280 :=
by 
  intro h1 h2 h3 h4 h5
  -- we put the proof here
  sorry

end NUMINAMATH_GPT_product_eq_1280_l335_33588


namespace NUMINAMATH_GPT_molecular_weight_of_Y_l335_33517

def molecular_weight_X : ℝ := 136
def molecular_weight_C6H8O7 : ℝ := 192
def moles_C6H8O7 : ℝ := 5

def total_mass_reactants := molecular_weight_X + moles_C6H8O7 * molecular_weight_C6H8O7

theorem molecular_weight_of_Y :
  total_mass_reactants = 1096 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_Y_l335_33517


namespace NUMINAMATH_GPT_solve_quadratic_equation_l335_33598

theorem solve_quadratic_equation (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l335_33598


namespace NUMINAMATH_GPT_find_a_b_l335_33536

theorem find_a_b (a b : ℤ) : (∀ (s : ℂ), s^2 + s - 1 = 0 → a * s^18 + b * s^17 + 1 = 0) → (a = 987 ∧ b = -1597) :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_l335_33536


namespace NUMINAMATH_GPT_train_length_approx_l335_33599

noncomputable def length_of_train (speed_km_hr : ℝ) (time_seconds : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time_seconds

theorem train_length_approx (speed_km_hr time_seconds : ℝ) (h_speed : speed_km_hr = 120) (h_time : time_seconds = 4) :
  length_of_train speed_km_hr time_seconds = 133.32 :=
by
  sorry

end NUMINAMATH_GPT_train_length_approx_l335_33599


namespace NUMINAMATH_GPT_trajectory_of_point_M_l335_33532

theorem trajectory_of_point_M (a x y : ℝ) (h: 0 < a) (A B M : ℝ × ℝ)
    (hA : A = (x, 0)) (hB : B = (0, y)) (hAB_length : Real.sqrt (x^2 + y^2) = 2 * a)
    (h_ratio : ∃ k, k ≠ 0 ∧ ∃ k', k' ≠ 0 ∧ A = k • M + k' • B ∧ (k + k' = 1) ∧ (k / k' = 1 / 2)) :
    (x / (4 / 3 * a))^2 + (y / (2 / 3 * a))^2 = 1 :=
sorry

end NUMINAMATH_GPT_trajectory_of_point_M_l335_33532


namespace NUMINAMATH_GPT_proof_combination_l335_33559

open Classical

theorem proof_combination :
  (∃ x : ℝ, x^3 < 1) ∧ (¬ ∃ x : ℚ, x^2 = 2) ∧ (¬ ∀ x : ℕ, x^3 > x^2) ∧ (∀ x : ℝ, x^2 + 1 > 0) :=
by
  have h1 : ∃ x : ℝ, x^3 < 1 := sorry
  have h2 : ¬ ∃ x : ℚ, x^2 = 2 := sorry
  have h3 : ¬ ∀ x : ℕ, x^3 > x^2 := sorry
  have h4 : ∀ x : ℝ, x^2 + 1 > 0 := sorry
  exact ⟨h1, h2, h3, h4⟩

end NUMINAMATH_GPT_proof_combination_l335_33559


namespace NUMINAMATH_GPT_find_other_factor_l335_33508

theorem find_other_factor (n : ℕ) (hn : n = 75) :
    ( ∃ k, k = 25 ∧ ∃ m, (k * 3^3 * m = 75 * 2^5 * 6^2 * 7^3) ) :=
by
  sorry

end NUMINAMATH_GPT_find_other_factor_l335_33508


namespace NUMINAMATH_GPT_john_unanswered_questions_l335_33502

theorem john_unanswered_questions :
  ∃ (c w u : ℕ), (30 + 4 * c - w = 84) ∧ (5 * c + 2 * u = 93) ∧ (c + w + u = 30) ∧ (u = 9) :=
by
  sorry

end NUMINAMATH_GPT_john_unanswered_questions_l335_33502


namespace NUMINAMATH_GPT_part1_solution_set_of_inequality_part2_range_of_m_l335_33589

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x - 3|

theorem part1_solution_set_of_inequality :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 5/2} :=
by
  sorry

theorem part2_range_of_m (m : ℝ) :
  (∀ x : ℝ, f x > 6 * m ^ 2 - 4 * m) ↔ -1/3 < m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_part1_solution_set_of_inequality_part2_range_of_m_l335_33589


namespace NUMINAMATH_GPT_complex_inverse_identity_l335_33596

theorem complex_inverse_identity : ∀ (i : ℂ), i^2 = -1 → (3 * i - 2 * i⁻¹)⁻¹ = -i / 5 :=
by
  -- Let's introduce the variables and the condition.
  intro i h

  -- Sorry is used to signify the proof is omitted.
  sorry

end NUMINAMATH_GPT_complex_inverse_identity_l335_33596


namespace NUMINAMATH_GPT_annie_passes_bonnie_first_l335_33537

def bonnie_speed (v : ℝ) := v
def annie_speed (v : ℝ) := 1.3 * v
def track_length := 500

theorem annie_passes_bonnie_first (v t : ℝ) (ht : 0.3 * v * t = track_length) : 
  (annie_speed v * t) / track_length = 4 + 1 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_annie_passes_bonnie_first_l335_33537


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l335_33514

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : ∀ x : ℝ, x > a → x > 2 ∧ ¬(x > 2 → x > a)) : a > 2 :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l335_33514


namespace NUMINAMATH_GPT_find_remainder_mod_105_l335_33550

-- Define the conditions as a set of hypotheses
variables {n a b c : ℕ}
variables (hn : n > 0)
variables (ha : a < 3) (hb : b < 5) (hc : c < 7)
variables (h3 : n % 3 = a) (h5 : n % 5 = b) (h7 : n % 7 = c)
variables (heq : 4 * a + 3 * b + 2 * c = 30)

-- State the theorem
theorem find_remainder_mod_105 : n % 105 = 29 :=
by
  -- Hypotheses block for documentation
  have ha_le : 0 ≤ a := sorry
  have hb_le : 0 ≤ b := sorry
  have hc_le : 0 ≤ c := sorry
  sorry

end NUMINAMATH_GPT_find_remainder_mod_105_l335_33550


namespace NUMINAMATH_GPT_sqrt_x_minus_1_domain_l335_33516

theorem sqrt_x_minus_1_domain (x : ℝ) : (∃ y, y^2 = x - 1) → (x ≥ 1) := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_x_minus_1_domain_l335_33516


namespace NUMINAMATH_GPT_weight_left_after_two_deliveries_l335_33571

-- Definitions and conditions
def initial_load : ℝ := 50000
def first_store_percentage : ℝ := 0.1
def second_store_percentage : ℝ := 0.2

-- The statement to be proven
theorem weight_left_after_two_deliveries :
  let weight_after_first_store := initial_load * (1 - first_store_percentage)
  let weight_after_second_store := weight_after_first_store * (1 - second_store_percentage)
  weight_after_second_store = 36000 :=
by sorry  -- Proof omitted

end NUMINAMATH_GPT_weight_left_after_two_deliveries_l335_33571
