import Mathlib

namespace NUMINAMATH_GPT_polynomial_root_fraction_l1525_152525

theorem polynomial_root_fraction (p q r s : ℝ) (h : p ≠ 0) 
    (h1 : p * 4^3 + q * 4^2 + r * 4 + s = 0)
    (h2 : p * (-3)^3 + q * (-3)^2 + r * (-3) + s = 0) : 
    (q + r) / p = -13 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_root_fraction_l1525_152525


namespace NUMINAMATH_GPT_exponent_subtraction_l1525_152505

theorem exponent_subtraction (a : ℝ) (m n : ℝ) (hm : a^m = 3) (hn : a^n = 5) : a^(m-n) = 3 / 5 := 
  sorry

end NUMINAMATH_GPT_exponent_subtraction_l1525_152505


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1525_152530

theorem solution_set_of_inequality {x : ℝ} : 
  (|2 * x - 1| - |x - 2| < 0) → (-1 < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1525_152530


namespace NUMINAMATH_GPT_geometric_sequence_k_value_l1525_152543

theorem geometric_sequence_k_value (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ)
  (hS : ∀ n, S n = k + 3^n)
  (h_geom : ∀ n, a (n+1) = S (n+1) - S n)
  (h_geo_seq : ∀ n, a (n+2) / a (n+1) = a (n+1) / a n) :
  k = -1 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_k_value_l1525_152543


namespace NUMINAMATH_GPT_problem_solution_l1525_152524

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x - a

theorem problem_solution (x₀ x₁ a : ℝ) (h₁ : 3 * x₀^2 - 2 * x₀ + a = 0) (h₂ : f x₁ a = f x₀ a) (h₃ : x₁ ≠ x₀) : x₁ + 2 * x₀ = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1525_152524


namespace NUMINAMATH_GPT_prove_total_payment_l1525_152519

-- Define the conditions under which the problem is set
def monthly_subscription_cost : ℝ := 14
def split_ratio : ℝ := 0.5
def months_in_year : ℕ := 12

-- Define the target amount to prove
def total_payment_after_one_year : ℝ := 84

-- Theorem statement
theorem prove_total_payment
  (h1: monthly_subscription_cost = 14)
  (h2: split_ratio = 0.5)
  (h3: months_in_year = 12) :
  monthly_subscription_cost * split_ratio * months_in_year = total_payment_after_one_year := 
  by
  sorry

end NUMINAMATH_GPT_prove_total_payment_l1525_152519


namespace NUMINAMATH_GPT_sequence_has_max_and_min_l1525_152575

noncomputable def a_n (n : ℕ) : ℝ := (4 / 9)^(n - 1) - (2 / 3)^(n - 1)

theorem sequence_has_max_and_min : 
  (∃ N, ∀ n, a_n n ≤ a_n N) ∧ 
  (∃ M, ∀ n, a_n n ≥ a_n M) :=
sorry

end NUMINAMATH_GPT_sequence_has_max_and_min_l1525_152575


namespace NUMINAMATH_GPT_jose_profit_share_l1525_152511

theorem jose_profit_share (investment_tom : ℕ) (months_tom : ℕ) 
                         (investment_jose : ℕ) (months_jose : ℕ) 
                         (total_profit : ℕ) :
                         investment_tom = 30000 →
                         months_tom = 12 →
                         investment_jose = 45000 →
                         months_jose = 10 →
                         total_profit = 63000 →
                         (investment_jose * months_jose / 
                         (investment_tom * months_tom + investment_jose * months_jose)) * total_profit = 35000 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  norm_num
  sorry

end NUMINAMATH_GPT_jose_profit_share_l1525_152511


namespace NUMINAMATH_GPT_total_towels_folded_in_one_hour_l1525_152531

-- Define the conditions for folding rates and breaks of each person
def Jane_folding_rate (minutes : ℕ) : ℕ :=
  if minutes % 8 < 5 then 5 * (minutes / 8 + 1) else 5 * (minutes / 8)

def Kyla_folding_rate (minutes : ℕ) : ℕ :=
  if minutes < 30 then 12 * (minutes / 10 + 1) else 36 + 6 * ((minutes - 30) / 10)

def Anthony_folding_rate (minutes : ℕ) : ℕ :=
  if minutes <= 40 then 14 * (minutes / 20)
  else if minutes <= 50 then 28
  else 28 + 14 * ((minutes - 50) / 20)

def David_folding_rate (minutes : ℕ) : ℕ :=
  let sets := minutes / 15
  let additional := sets / 3
  4 * (sets - additional) + 5 * additional

-- Definitions are months passing given in the questions
def hours_fold_towels (minutes : ℕ) : ℕ :=
  Jane_folding_rate minutes + Kyla_folding_rate minutes + Anthony_folding_rate minutes + David_folding_rate minutes

theorem total_towels_folded_in_one_hour : hours_fold_towels 60 = 134 := sorry

end NUMINAMATH_GPT_total_towels_folded_in_one_hour_l1525_152531


namespace NUMINAMATH_GPT_problem_divisibility_l1525_152590

theorem problem_divisibility (k : ℕ) (hk : k > 1) (p : ℕ) (hp : p = 6 * k + 1) (hprime : Prime p) 
  (m : ℕ) (hm : m = 2^p - 1) : 
  127 * m ∣ 2^(m - 1) - 1 := 
sorry

end NUMINAMATH_GPT_problem_divisibility_l1525_152590


namespace NUMINAMATH_GPT_total_height_of_buildings_l1525_152564

theorem total_height_of_buildings :
  let height_first_building := 600
  let height_second_building := 2 * height_first_building
  let height_third_building := 3 * (height_first_building + height_second_building)
  height_first_building + height_second_building + height_third_building = 7200 := by
    let height_first_building := 600
    let height_second_building := 2 * height_first_building
    let height_third_building := 3 * (height_first_building + height_second_building)
    show height_first_building + height_second_building + height_third_building = 7200
    sorry

end NUMINAMATH_GPT_total_height_of_buildings_l1525_152564


namespace NUMINAMATH_GPT_max_stories_on_odd_pages_l1525_152561

theorem max_stories_on_odd_pages 
    (stories : Fin 30 -> Fin 31) 
    (h_unique : Function.Injective stories) 
    (h_bounds : ∀ i, stories i < 31)
    : ∃ n, n = 23 ∧ ∃ f : Fin n -> Fin 30, ∀ j, f j % 2 = 1 := 
sorry

end NUMINAMATH_GPT_max_stories_on_odd_pages_l1525_152561


namespace NUMINAMATH_GPT_volume_uncovered_is_correct_l1525_152532

-- Define the volumes of the shoebox and the objects
def volume_shoebox : ℕ := 12 * 6 * 4
def volume_object1 : ℕ := 5 * 3 * 1
def volume_object2 : ℕ := 2 * 2 * 3
def volume_object3 : ℕ := 4 * 2 * 4

-- Define the total volume of the objects
def total_volume_objects : ℕ := volume_object1 + volume_object2 + volume_object3

-- Define the volume left uncovered
def volume_uncovered : ℕ := volume_shoebox - total_volume_objects

-- Prove that the volume left uncovered is 229 cubic inches
theorem volume_uncovered_is_correct : volume_uncovered = 229 := by
  -- This is where the proof would be written
  sorry

end NUMINAMATH_GPT_volume_uncovered_is_correct_l1525_152532


namespace NUMINAMATH_GPT_salt_fraction_l1525_152546

variables {a x : ℝ}

-- First condition: the shortfall in salt the first time
def shortfall_first (a x : ℝ) : ℝ := a - x

-- Second condition: the shortfall in salt the second time
def shortfall_second (a x : ℝ) : ℝ := a - 2 * x

-- Third condition: relationship given by the problem
axiom condition : shortfall_first a x = 2 * shortfall_second a x

-- Prove fraction of necessary salt added the first time is 1/3
theorem salt_fraction (a x : ℝ) (h : shortfall_first a x = 2 * shortfall_second a x) : x = a / 3 :=
by
  sorry

end NUMINAMATH_GPT_salt_fraction_l1525_152546


namespace NUMINAMATH_GPT_claire_apple_pies_l1525_152594

theorem claire_apple_pies (N : ℤ) 
  (h1 : N % 6 = 4) 
  (h2 : N % 8 = 5) 
  (h3 : N < 30) : 
  N = 22 :=
by
  sorry

end NUMINAMATH_GPT_claire_apple_pies_l1525_152594


namespace NUMINAMATH_GPT_geometric_series_squares_sum_l1525_152518

theorem geometric_series_squares_sum (a : ℝ) (r : ℝ) (h : -1 < r ∧ r < 1) :
  (∑' n : ℕ, (a * r^n)^2) = a^2 / (1 - r^2) :=
by sorry

end NUMINAMATH_GPT_geometric_series_squares_sum_l1525_152518


namespace NUMINAMATH_GPT_original_numbers_placement_l1525_152512

-- Define each letter stands for a given number
def A : ℕ := 1
def B : ℕ := 3
def C : ℕ := 2
def D : ℕ := 5
def E : ℕ := 6
def F : ℕ := 4

-- Conditions provided
def white_triangle_condition (x y z : ℕ) : Prop :=
x + y = z

-- Main problem reformulated as theorem
theorem original_numbers_placement :
  (A = 1) ∧ (B = 3) ∧ (C = 2) ∧ (D = 5) ∧ (E = 6) ∧ (F = 4) :=
sorry

end NUMINAMATH_GPT_original_numbers_placement_l1525_152512


namespace NUMINAMATH_GPT_mixed_number_calculation_l1525_152539

theorem mixed_number_calculation :
  (481 + 1/6) + (265 + 1/12) + (904 + 1/20) - (184 + 29/30) - (160 + 41/42) - (703 + 55/56) = 603 + 3/8 := 
sorry

end NUMINAMATH_GPT_mixed_number_calculation_l1525_152539


namespace NUMINAMATH_GPT_area_of_rectangle_l1525_152533

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l1525_152533


namespace NUMINAMATH_GPT_vodka_shot_size_l1525_152503

theorem vodka_shot_size (x : ℝ) (h1 : 8 / 2 = 4) (h2 : 4 * x = 2 * 3) : x = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_vodka_shot_size_l1525_152503


namespace NUMINAMATH_GPT_gcd_correct_l1525_152569

def gcd_765432_654321 : ℕ :=
  Nat.gcd 765432 654321

theorem gcd_correct : gcd_765432_654321 = 6 :=
by sorry

end NUMINAMATH_GPT_gcd_correct_l1525_152569


namespace NUMINAMATH_GPT_f_x_plus_1_even_f_x_plus_3_odd_l1525_152587

variable (R : Type) [CommRing R]

variable (f : R → R)

-- Conditions
axiom condition1 : ∀ x : R, f (1 + x) = f (1 - x)
axiom condition2 : ∀ x : R, f (x - 2) + f (-x) = 0

-- Prove that f(x + 1) is an even function
theorem f_x_plus_1_even (x : R) : f (x + 1) = f (-(x + 1)) :=
by sorry

-- Prove that f(x + 3) is an odd function
theorem f_x_plus_3_odd (x : R) : f (x + 3) = - f (-(x + 3)) :=
by sorry

end NUMINAMATH_GPT_f_x_plus_1_even_f_x_plus_3_odd_l1525_152587


namespace NUMINAMATH_GPT_simplify_fraction_l1525_152596

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = (65 : ℚ) / 12 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1525_152596


namespace NUMINAMATH_GPT_min_value_expression_l1525_152517

theorem min_value_expression (x : ℝ) (h : x ≠ -7) : 
  ∃ y, y = 1 ∧ ∀ z, z = (2 * x ^ 2 + 98) / ((x + 7) ^ 2) → y ≤ z := 
sorry

end NUMINAMATH_GPT_min_value_expression_l1525_152517


namespace NUMINAMATH_GPT_cos_formula_of_tan_l1525_152501

theorem cos_formula_of_tan (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi) :
  Real.cos (5 * Real.pi / 2 + 2 * α) = -4 / 5 := 
  sorry

end NUMINAMATH_GPT_cos_formula_of_tan_l1525_152501


namespace NUMINAMATH_GPT_sum_quotient_reciprocal_eq_one_point_thirty_five_l1525_152551

theorem sum_quotient_reciprocal_eq_one_point_thirty_five (x y : ℝ)
  (h1 : x + y = 45) (h2 : x * y = 500) : x / y + 1 / x + 1 / y = 1.35 := by
  -- Proof details would go here
  sorry

end NUMINAMATH_GPT_sum_quotient_reciprocal_eq_one_point_thirty_five_l1525_152551


namespace NUMINAMATH_GPT_password_probability_l1525_152536

theorem password_probability : 
  (5/10) * (51/52) * (9/10) = 459 / 1040 := by
  sorry

end NUMINAMATH_GPT_password_probability_l1525_152536


namespace NUMINAMATH_GPT_sum_of_solutions_comparison_l1525_152540

variable (a a' b b' c c' : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0)

theorem sum_of_solutions_comparison :
  ( (c - b) / a > (c' - b') / a' ) ↔ ( (c'-b') / a' < (c-b) / a ) :=
by sorry

end NUMINAMATH_GPT_sum_of_solutions_comparison_l1525_152540


namespace NUMINAMATH_GPT_employees_without_increase_l1525_152557

-- Define the constants and conditions
def total_employees : ℕ := 480
def salary_increase_percentage : ℕ := 10
def travel_allowance_increase_percentage : ℕ := 20

-- Define the calculations derived from conditions
def employees_with_salary_increase : ℕ := (salary_increase_percentage * total_employees) / 100
def employees_with_travel_allowance_increase : ℕ := (travel_allowance_increase_percentage * total_employees) / 100

-- Total employees who got increases assuming no overlap
def employees_with_increases : ℕ := employees_with_salary_increase + employees_with_travel_allowance_increase

-- The proof statement
theorem employees_without_increase :
  total_employees - employees_with_increases = 336 := by
  sorry

end NUMINAMATH_GPT_employees_without_increase_l1525_152557


namespace NUMINAMATH_GPT_photograph_perimeter_l1525_152593

-- Definitions of the conditions
def photograph_is_rectangular : Prop := True
def one_inch_border_area (w l m : ℕ) : Prop := (w + 2) * (l + 2) = m
def three_inch_border_area (w l m : ℕ) : Prop := (w + 6) * (l + 6) = m + 52

-- Lean statement of the problem
theorem photograph_perimeter (w l m : ℕ) 
  (h1 : photograph_is_rectangular)
  (h2 : one_inch_border_area w l m)
  (h3 : three_inch_border_area w l m) : 
  2 * (w + l) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_photograph_perimeter_l1525_152593


namespace NUMINAMATH_GPT_points_in_quadrants_l1525_152570

theorem points_in_quadrants :
  ∀ (x y : ℝ), (y > 3 * x) → (y > 5 - 2 * x) → ((0 ≤ x ∧ 0 ≤ y) ∨ (x ≤ 0 ∧ 0 ≤ y)) :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_points_in_quadrants_l1525_152570


namespace NUMINAMATH_GPT_work_efficiency_ratio_l1525_152556

theorem work_efficiency_ratio (a b k : ℝ) (ha : a = k * b) (hb : b = 1/15)
  (hab : a + b = 1/5) : k = 2 :=
by sorry

end NUMINAMATH_GPT_work_efficiency_ratio_l1525_152556


namespace NUMINAMATH_GPT_boy_walking_speed_l1525_152585

theorem boy_walking_speed 
  (travel_rate : ℝ) 
  (total_journey_time : ℝ) 
  (distance : ℝ) 
  (post_office_time : ℝ) 
  (walking_back_time : ℝ) 
  (walking_speed : ℝ): 
  travel_rate = 12.5 ∧ 
  total_journey_time = 5 + 48/60 ∧ 
  distance = 9.999999999999998 ∧ 
  post_office_time = distance / travel_rate ∧ 
  walking_back_time = total_journey_time - post_office_time ∧ 
  walking_speed = distance / walking_back_time 
  → walking_speed = 2 := 
by 
  intros h;
  sorry

end NUMINAMATH_GPT_boy_walking_speed_l1525_152585


namespace NUMINAMATH_GPT_point_slope_intersection_lines_l1525_152515

theorem point_slope_intersection_lines : 
  ∀ s : ℝ, ∃ x y : ℝ, 2*x - 3*y = 8*s + 6 ∧ x + 2*y = 3*s - 1 ∧ y = -((2*x)/25 + 182/175) := 
sorry

end NUMINAMATH_GPT_point_slope_intersection_lines_l1525_152515


namespace NUMINAMATH_GPT_repeating_decimal_sum_l1525_152521

theorem repeating_decimal_sum :
  let x := (0.3333333333333333 : ℚ) -- 0.\overline{3}
  let y := (0.0707070707070707 : ℚ) -- 0.\overline{07}
  let z := (0.008008008008008 : ℚ)  -- 0.\overline{008}
  x + y + z = 418 / 999 := by
sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l1525_152521


namespace NUMINAMATH_GPT_final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r_l1525_152548

variable (k r s N : ℝ)
variable (h_pos_k : 0 < k)
variable (h_pos_r : 0 < r)
variable (h_pos_s : 0 < s)
variable (h_pos_N : 0 < N)
variable (h_r_lt_80 : r < 80)

theorem final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r :
  N * (1 + k / 100) * (1 - r / 100) + 10 * s > N ↔ k > 100 * r / (100 - r) :=
sorry

end NUMINAMATH_GPT_final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r_l1525_152548


namespace NUMINAMATH_GPT_division_of_converted_values_l1525_152526

theorem division_of_converted_values 
  (h : 144 * 177 = 25488) : 
  254.88 / 0.177 = 1440 := by
  sorry

end NUMINAMATH_GPT_division_of_converted_values_l1525_152526


namespace NUMINAMATH_GPT_num_students_B_l1525_152588

-- Define the given conditions
variables (x : ℕ) -- The number of students who get a B

noncomputable def number_of_A := 2 * x
noncomputable def number_of_C := (12 / 10 : ℤ) * x -- Using (12 / 10) to approximate 1.2 in integers

-- Given total number of students is 42 for integer result
def total_students := 42

-- Lean statement to show number of students getting B is 10
theorem num_students_B : 4.2 * (x : ℝ) = 42 → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_num_students_B_l1525_152588


namespace NUMINAMATH_GPT_max_x_for_integer_fraction_l1525_152574

theorem max_x_for_integer_fraction (x : ℤ) (h : ∃ k : ℤ, x^2 + 2 * x + 11 = k * (x - 3)) : x ≤ 29 :=
by {
    -- This is where the proof would be,
    -- but we skip the proof per the instructions.
    sorry
}

end NUMINAMATH_GPT_max_x_for_integer_fraction_l1525_152574


namespace NUMINAMATH_GPT_fruit_problem_l1525_152566

variables (A O x : ℕ) -- Natural number variables for apples, oranges, and oranges put back

theorem fruit_problem :
  (A + O = 10) ∧
  (40 * A + 60 * O = 480) ∧
  (240 + 60 * (O - x) = 45 * (10 - x)) →
  A = 6 ∧ O = 4 ∧ x = 2 :=
  sorry

end NUMINAMATH_GPT_fruit_problem_l1525_152566


namespace NUMINAMATH_GPT_fraction_equivalence_l1525_152549

theorem fraction_equivalence : 
  (∀ (a b : ℕ), (a ≠ 0 ∧ b ≠ 0) → (15 * b = 25 * a ↔ a = 3 ∧ b = 5)) ∧
  (15 * 4 ≠ 25 * 3) ∧
  (15 * 3 ≠ 25 * 2) ∧
  (15 * 2 ≠ 25 * 1) ∧
  (15 * 7 ≠ 25 * 5) :=
by
  sorry

end NUMINAMATH_GPT_fraction_equivalence_l1525_152549


namespace NUMINAMATH_GPT_melissa_points_per_game_l1525_152583

variable (t g p : ℕ)

theorem melissa_points_per_game (ht : t = 36) (hg : g = 3) : p = t / g → p = 12 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_melissa_points_per_game_l1525_152583


namespace NUMINAMATH_GPT_loss_percentage_first_book_l1525_152595

theorem loss_percentage_first_book (C1 C2 : ℝ) 
    (total_cost : ℝ) 
    (gain_percentage : ℝ)
    (S1 S2 : ℝ)
    (cost_first_book : C1 = 175)
    (total_cost_condition : total_cost = 300)
    (gain_condition : gain_percentage = 0.19)
    (same_selling_price : S1 = S2)
    (second_book_cost : C2 = total_cost - C1)
    (selling_price_second_book : S2 = C2 * (1 + gain_percentage)) :
    (C1 - S1) / C1 * 100 = 15 :=
by
  sorry

end NUMINAMATH_GPT_loss_percentage_first_book_l1525_152595


namespace NUMINAMATH_GPT_problem_a_problem_b_problem_c_l1525_152538

open Real

noncomputable def conditions (x : ℝ) := x >= 1 / 2

/-- 
a) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = \sqrt{2} \)
valid if and only if x in [1/2, 1].
-/
theorem problem_a (x : ℝ) (h : conditions x) :
  (sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = sqrt 2) ↔ (1 / 2 ≤ x ∧ x ≤ 1) :=
  sorry

/-- 
b) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = 1 \)
has no solution.
-/
theorem problem_b (x : ℝ) (h : conditions x) :
  sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 1 → False :=
  sorry

/-- 
c) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = 2 \)
if and only if x = 3/2.
-/
theorem problem_c (x : ℝ) (h : conditions x) :
  (sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 2) ↔ (x = 3 / 2) :=
  sorry

end NUMINAMATH_GPT_problem_a_problem_b_problem_c_l1525_152538


namespace NUMINAMATH_GPT_geometric_sequence_third_term_l1525_152535

theorem geometric_sequence_third_term (a b c d : ℕ) (r : ℕ) 
  (h₁ : d * r = 81) 
  (h₂ : 81 * r = 243) 
  (h₃ : r = 3) : c = 27 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_l1525_152535


namespace NUMINAMATH_GPT_curve_is_circle_l1525_152508

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (2 * Real.sin θ - Real.cos θ)) :
  ∃ (a b r : ℝ), (r > 0) ∧ ((x + a)^2 + (y + b)^2 = r^2) :=
by
  sorry

end NUMINAMATH_GPT_curve_is_circle_l1525_152508


namespace NUMINAMATH_GPT_vector_operation_result_l1525_152579

variables {V : Type*} [AddCommGroup V] [Module ℝ V] (A B C O E : V)

theorem vector_operation_result :
  (A - B) - (C - B) + (O - E) - (O - C) = (A - E) :=
by
  sorry

end NUMINAMATH_GPT_vector_operation_result_l1525_152579


namespace NUMINAMATH_GPT_driver_total_distance_is_148_l1525_152507

-- Definitions of the distances traveled according to the given conditions
def distance_MWF : ℕ := 12 * 3
def total_distance_MWF : ℕ := distance_MWF * 3
def distance_T : ℕ := 9 * 5 / 2  -- using ℕ for 2.5 hours as 5/2
def distance_Th : ℕ := 7 * 5 / 2

-- Statement of the total distance calculation
def total_distance_week : ℕ :=
  total_distance_MWF + distance_T + distance_Th

-- Theorem stating the total distance traveled during the week
theorem driver_total_distance_is_148 : total_distance_week = 148 := by
  sorry

end NUMINAMATH_GPT_driver_total_distance_is_148_l1525_152507


namespace NUMINAMATH_GPT_direct_proportional_function_inverse_proportional_function_quadratic_function_power_function_l1525_152568

-- Direct Proportional Function
theorem direct_proportional_function (m : ℝ) :
  (m^2 + 2 * m ≠ 0) → (m^2 + m - 1 = 1) → m = 1 :=
by 
  sorry

-- Inverse Proportional Function
theorem inverse_proportional_function (m : ℝ) :
  (m^2 + 2 * m ≠ 0) → (m^2 + m - 1 = -1) → m = -1 :=
by 
  sorry

-- Quadratic Function
theorem quadratic_function (m : ℝ) :
  (m^2 + 2 * m ≠ 0) → (m^2 + m - 1 = 2) → (m = (-1 + Real.sqrt 13) / 2 ∨ m = (-1 - Real.sqrt 13) / 2) :=
by 
  sorry

-- Power Function
theorem power_function (m : ℝ) :
  (m^2 + 2 * m = 1) → (m = -1 + Real.sqrt 2 ∨ m = -1 - Real.sqrt 2) :=
by 
  sorry

end NUMINAMATH_GPT_direct_proportional_function_inverse_proportional_function_quadratic_function_power_function_l1525_152568


namespace NUMINAMATH_GPT_polynomial_root_transformation_l1525_152522

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem polynomial_root_transformation :
  let P (a b c d e : ℝ) (x : ℂ) := (x^6 : ℂ) + (a : ℂ) * x^5 + (b : ℂ) * x^4 + (c : ℂ) * x^3 + (d : ℂ) * x^2 + (e : ℂ) * x + 4096
  (∀ r : ℂ, P 0 0 0 0 0 r = 0 → P 0 0 0 0 0 (ω * r) = 0) →
  ∃ a b c d e : ℝ, ∃ p : ℕ, p = 3 := sorry

end NUMINAMATH_GPT_polynomial_root_transformation_l1525_152522


namespace NUMINAMATH_GPT_intersecting_lines_k_value_l1525_152534

theorem intersecting_lines_k_value :
  ∃ k : ℚ, (∀ x y : ℚ, y = 3 * x + 12 ∧ y = -5 * x - 7 → y = 2 * x + k) → k = 77 / 8 :=
sorry

end NUMINAMATH_GPT_intersecting_lines_k_value_l1525_152534


namespace NUMINAMATH_GPT_average_of_three_l1525_152542

theorem average_of_three {a b c d e : ℚ}
    (h1 : (a + b + c + d + e) / 5 = 12)
    (h2 : (d + e) / 2 = 24) :
    (a + b + c) / 3 = 4 := by
  sorry

end NUMINAMATH_GPT_average_of_three_l1525_152542


namespace NUMINAMATH_GPT_tetrahedron_min_green_edges_l1525_152504

theorem tetrahedron_min_green_edges : 
  ∃ (green_edges : Finset (Fin 6)), 
  (∀ face : Finset (Fin 6), face.card = 3 → ∃ edge ∈ face, edge ∈ green_edges) ∧ green_edges.card = 3 :=
by sorry

end NUMINAMATH_GPT_tetrahedron_min_green_edges_l1525_152504


namespace NUMINAMATH_GPT_jonah_first_intermission_lemonade_l1525_152552

theorem jonah_first_intermission_lemonade :
  ∀ (l1 l2 l3 l_total : ℝ)
  (h1 : l2 = 0.42)
  (h2 : l3 = 0.25)
  (h3 : l_total = 0.92)
  (h4 : l_total = l1 + l2 + l3),
  l1 = 0.25 :=
by sorry

end NUMINAMATH_GPT_jonah_first_intermission_lemonade_l1525_152552


namespace NUMINAMATH_GPT_sodium_bicarbonate_moles_l1525_152500

theorem sodium_bicarbonate_moles (HCl NaHCO3 CO2 : ℕ) (h1 : HCl = 1) (h2 : CO2 = 1) :
  NaHCO3 = 1 :=
by sorry

end NUMINAMATH_GPT_sodium_bicarbonate_moles_l1525_152500


namespace NUMINAMATH_GPT_three_nabla_four_l1525_152578

noncomputable def modified_operation (a b : ℝ) : ℝ :=
  (a + b^2) / (1 + a * b^2)

theorem three_nabla_four : modified_operation 3 4 = 19 / 49 := 
  by 
  sorry

end NUMINAMATH_GPT_three_nabla_four_l1525_152578


namespace NUMINAMATH_GPT_vanessa_deleted_files_l1525_152584

theorem vanessa_deleted_files (initial_music_files : ℕ) (initial_video_files : ℕ) (files_left : ℕ) (files_deleted : ℕ) :
  initial_music_files = 13 → initial_video_files = 30 → files_left = 33 → 
  files_deleted = (initial_music_files + initial_video_files) - files_left → files_deleted = 10 :=
by
  sorry

end NUMINAMATH_GPT_vanessa_deleted_files_l1525_152584


namespace NUMINAMATH_GPT_net_population_increase_per_day_l1525_152547

def birth_rate : Nat := 4
def death_rate : Nat := 2
def seconds_per_day : Nat := 24 * 60 * 60

theorem net_population_increase_per_day : 
  (birth_rate - death_rate) * (seconds_per_day / 2) = 86400 := by
  sorry

end NUMINAMATH_GPT_net_population_increase_per_day_l1525_152547


namespace NUMINAMATH_GPT_least_positive_integer_l1525_152571

theorem least_positive_integer (k : ℕ) (h : (528 + k) % 5 = 0) : k = 2 :=
sorry

end NUMINAMATH_GPT_least_positive_integer_l1525_152571


namespace NUMINAMATH_GPT_cone_base_circumference_l1525_152523

theorem cone_base_circumference
  (V : ℝ) (h : ℝ) (C : ℝ)
  (volume_eq : V = 18 * Real.pi)
  (height_eq : h = 3) :
  C = 6 * Real.sqrt 2 * Real.pi :=
sorry

end NUMINAMATH_GPT_cone_base_circumference_l1525_152523


namespace NUMINAMATH_GPT_burgers_per_day_l1525_152527

def calories_per_burger : ℝ := 20
def total_calories_after_two_days : ℝ := 120

theorem burgers_per_day :
  total_calories_after_two_days / (2 * calories_per_burger) = 3 := 
by
  sorry

end NUMINAMATH_GPT_burgers_per_day_l1525_152527


namespace NUMINAMATH_GPT_pipe_Q_drain_portion_l1525_152577

noncomputable def portion_liquid_drain_by_Q (T_Q T_P T_R : ℝ) (h1 : T_P = 3 / 4 * T_Q) (h2 : T_R = T_P) : ℝ :=
  let rate_P := 1 / T_P
  let rate_Q := 1 / T_Q
  let rate_R := 1 / T_R
  let combined_rate := rate_P + rate_Q + rate_R
  (rate_Q / combined_rate)

theorem pipe_Q_drain_portion (T_Q T_P T_R : ℝ) (h1 : T_P = 3 / 4 * T_Q) (h2 : T_R = T_P) :
  portion_liquid_drain_by_Q T_Q T_P T_R h1 h2 = 3 / 11 :=
by
  sorry

end NUMINAMATH_GPT_pipe_Q_drain_portion_l1525_152577


namespace NUMINAMATH_GPT_graph_paper_squares_below_line_l1525_152581

theorem graph_paper_squares_below_line
  (h : ∀ (x y : ℕ), 12 * x + 247 * y = 2976)
  (square_size : ℕ) 
  (xs : ℕ) (ys : ℕ)
  (line_eq : ∀ (x y : ℕ), y = 247 * x / 12)
  (n_squares : ℕ) :
  n_squares = 1358
  := by
    sorry

end NUMINAMATH_GPT_graph_paper_squares_below_line_l1525_152581


namespace NUMINAMATH_GPT_simplify_exponent_multiplication_l1525_152528

theorem simplify_exponent_multiplication :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end NUMINAMATH_GPT_simplify_exponent_multiplication_l1525_152528


namespace NUMINAMATH_GPT_correctPairsAreSkating_l1525_152506

def Friend := String
def Brother := String

structure SkatingPair where
  gentleman : Friend
  lady : Friend

-- Define the list of friends with their brothers
def friends : List Friend := ["Lyusya Egorova", "Olya Petrova", "Inna Krymova", "Anya Vorobyova"]
def brothers : List Brother := ["Andrey Egorov", "Serezha Petrov", "Dima Krymov", "Yura Vorobyov"]

-- Condition: The skating pairs such that gentlemen are taller than ladies and no one skates with their sibling
noncomputable def skatingPairs : List SkatingPair :=
  [ {gentleman := "Yura Vorobyov", lady := "Lyusya Egorova"},
    {gentleman := "Andrey Egorov", lady := "Olya Petrova"},
    {gentleman := "Serezha Petrov", lady := "Inna Krymova"},
    {gentleman := "Dima Krymov", lady := "Anya Vorobyova"} ]

-- Proving that the pairs are exactly as specified.
theorem correctPairsAreSkating :
  skatingPairs = 
    [ {gentleman := "Yura Vorobyov", lady := "Lyusya Egorova"},
      {gentleman := "Andrey Egorov", lady := "Olya Petrova"},
      {gentleman := "Serezha Petrov", lady := "Inna Krymova"},
      {gentleman := "Dima Krymov", lady := "Anya Vorobyova"} ] :=
by
  sorry

end NUMINAMATH_GPT_correctPairsAreSkating_l1525_152506


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_l1525_152576

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) ^ 2

theorem smallest_positive_period_of_f :
  ∀ T > 0, (∀ x, f (x + T) = f x) ↔ T = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_l1525_152576


namespace NUMINAMATH_GPT_area_and_perimeter_l1525_152563

-- Given a rectangle R with length l and width w
variables (l w : ℝ)
-- Define the area of R
def area_R : ℝ := l * w

-- Define a smaller rectangle that is cut out, with an area A_cut
variables (A_cut : ℝ)
-- Define the area of the resulting figure S
def area_S : ℝ := area_R l w - A_cut

-- Define the perimeter of R
def perimeter_R : ℝ := 2 * l + 2 * w

-- perimeter_R remains the same after cutting out the smaller rectangle
theorem area_and_perimeter (h_cut : 0 < A_cut) (h_cut_le : A_cut ≤ area_R l w) : 
  (area_S l w A_cut < area_R l w) ∧ (perimeter_R l w = perimeter_R l w) :=
by
  sorry

end NUMINAMATH_GPT_area_and_perimeter_l1525_152563


namespace NUMINAMATH_GPT_monotonically_increasing_interval_l1525_152586

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x ^ 2

theorem monotonically_increasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 1) → (f x > f 0) := 
by
  sorry

end NUMINAMATH_GPT_monotonically_increasing_interval_l1525_152586


namespace NUMINAMATH_GPT_symmetric_line_equation_l1525_152509

theorem symmetric_line_equation : 
  ∀ (P : ℝ × ℝ) (L : ℝ × ℝ × ℝ), 
  P = (1, 1) → 
  L = (2, 3, -6) → 
  (∃ (a b c : ℝ), a * 1 + b * 1 + c = 0 → a * x + b * y + c = 0 ↔ 2 * x + 3 * y - 4 = 0) 
:= 
sorry

end NUMINAMATH_GPT_symmetric_line_equation_l1525_152509


namespace NUMINAMATH_GPT_marys_birthday_l1525_152565

theorem marys_birthday (M : ℝ) (h1 : (3 / 4) * M - (3 / 20) * M = 60) : M = 100 := by
  -- Leave the proof as sorry for now
  sorry

end NUMINAMATH_GPT_marys_birthday_l1525_152565


namespace NUMINAMATH_GPT_unique_digit_sum_l1525_152545

theorem unique_digit_sum (A B C D : ℕ) (h1 : A + B + C + D = 20) (h2 : B + A + 1 = 11) (uniq : (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D)) : D = 8 :=
sorry

end NUMINAMATH_GPT_unique_digit_sum_l1525_152545


namespace NUMINAMATH_GPT_centroid_of_quadrant_arc_l1525_152582

def circle_equation (R : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = R^2
def density (ρ₀ x y : ℝ) : ℝ := ρ₀ * x * y

theorem centroid_of_quadrant_arc (R ρ₀ : ℝ) :
  (∃ x y, circle_equation R x y ∧ x ≥ 0 ∧ y ≥ 0) →
  ∃ x_c y_c, x_c = 2 * R / 3 ∧ y_c = 2 * R / 3 :=
sorry

end NUMINAMATH_GPT_centroid_of_quadrant_arc_l1525_152582


namespace NUMINAMATH_GPT_calculate_xy_yz_zx_l1525_152544

variable (x y z : ℝ)

theorem calculate_xy_yz_zx (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h1 : x^2 + x * y + y^2 = 75)
    (h2 : y^2 + y * z + z^2 = 49)
    (h3 : z^2 + z * x + x^2 = 124) : 
    x * y + y * z + z * x = 70 :=
sorry

end NUMINAMATH_GPT_calculate_xy_yz_zx_l1525_152544


namespace NUMINAMATH_GPT_angle_coincides_with_graph_y_eq_neg_abs_x_l1525_152597

noncomputable def angle_set (α : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = k * 360 + 225 ∨ α = k * 360 + 315}

theorem angle_coincides_with_graph_y_eq_neg_abs_x (α : ℝ) :
  α ∈ angle_set α ↔ 
  ∃ k : ℤ, (α = k * 360 + 225 ∨ α = k * 360 + 315) :=
by
  sorry

end NUMINAMATH_GPT_angle_coincides_with_graph_y_eq_neg_abs_x_l1525_152597


namespace NUMINAMATH_GPT_sum_is_ten_l1525_152520

variable (x y : ℝ) (S : ℝ)

-- Conditions
def condition1 : Prop := x + y = S
def condition2 : Prop := x = 25 / y
def condition3 : Prop := x^2 + y^2 = 50

-- Theorem
theorem sum_is_ten (h1 : condition1 x y S) (h2 : condition2 x y) (h3 : condition3 x y) : S = 10 :=
sorry

end NUMINAMATH_GPT_sum_is_ten_l1525_152520


namespace NUMINAMATH_GPT_single_shot_decrease_l1525_152553

theorem single_shot_decrease (S : ℝ) (r1 r2 r3 : ℝ) (h1 : r1 = 0.05) (h2 : r2 = 0.10) (h3 : r3 = 0.15) :
  (1 - (1 - r1) * (1 - r2) * (1 - r3)) * 100 = 27.325 := 
by
  sorry

end NUMINAMATH_GPT_single_shot_decrease_l1525_152553


namespace NUMINAMATH_GPT_narrow_black_stripes_l1525_152580

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end NUMINAMATH_GPT_narrow_black_stripes_l1525_152580


namespace NUMINAMATH_GPT_angle_ratio_l1525_152559

theorem angle_ratio (A B C : ℝ) (hA : A = 60) (hB : B = 80) (h_sum : A + B + C = 180) : B / C = 2 := by
  sorry

end NUMINAMATH_GPT_angle_ratio_l1525_152559


namespace NUMINAMATH_GPT_solve_problem_l1525_152516

noncomputable def problem_statement : Prop :=
  ∀ (a b c : ℕ),
    (a ≤ b) →
    (b ≤ c) →
    Nat.gcd (Nat.gcd a b) c = 1 →
    (a^2 * b) ∣ (a^3 + b^3 + c^3) →
    (b^2 * c) ∣ (a^3 + b^3 + c^3) →
    (c^2 * a) ∣ (a^3 + b^3 + c^3) →
    (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 2 ∧ c = 3)

-- Here we declare the main theorem but skip the proof.
theorem solve_problem : problem_statement :=
by sorry

end NUMINAMATH_GPT_solve_problem_l1525_152516


namespace NUMINAMATH_GPT_carlos_laundry_time_l1525_152529

def washing_time1 := 30
def washing_time2 := 45
def washing_time3 := 40
def washing_time4 := 50
def washing_time5 := 35
def drying_time1 := 85
def drying_time2 := 95

def total_laundry_time := washing_time1 + washing_time2 + washing_time3 + washing_time4 + washing_time5 + drying_time1 + drying_time2

theorem carlos_laundry_time : total_laundry_time = 380 :=
by
  sorry

end NUMINAMATH_GPT_carlos_laundry_time_l1525_152529


namespace NUMINAMATH_GPT_right_triangle_area_l1525_152592

theorem right_triangle_area (a b c : ℝ) (h1 : a + b + c = 90) (h2 : a^2 + b^2 + c^2 = 3362) (h3 : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 180 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1525_152592


namespace NUMINAMATH_GPT_intersection_complement_M_N_l1525_152598

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem intersection_complement_M_N :
  (U \ M) ∩ N = {-3, -4} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_complement_M_N_l1525_152598


namespace NUMINAMATH_GPT_carol_weight_l1525_152541

variable (a c : ℝ)

-- Conditions based on the problem statement
def combined_weight : Prop := a + c = 280
def weight_difference : Prop := c - a = c / 3

theorem carol_weight (h1 : combined_weight a c) (h2 : weight_difference a c) : c = 168 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_carol_weight_l1525_152541


namespace NUMINAMATH_GPT_line_tangent_to_parabola_proof_l1525_152573

noncomputable def line_tangent_to_parabola (d : ℝ) := (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1

theorem line_tangent_to_parabola_proof (d : ℝ) (h : ∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) : d = 1 :=
sorry

end NUMINAMATH_GPT_line_tangent_to_parabola_proof_l1525_152573


namespace NUMINAMATH_GPT_part1_part2_part3_l1525_152554

noncomputable def y1 (x : ℝ) : ℝ := 0.1 * x + 15
noncomputable def y2 (x : ℝ) : ℝ := 0.15 * x

-- Prove that the functions are as described
theorem part1 : ∀ x : ℝ, y1 x = 0.1 * x + 15 ∧ y2 x = 0.15 * x :=
by sorry

-- Prove that x = 300 results in equal charges for Packages A and B
theorem part2 : y1 300 = y2 300 :=
by sorry

-- Prove that Package A is more cost-effective when x > 300
theorem part3 : ∀ x : ℝ, x > 300 → y1 x < y2 x :=
by sorry

end NUMINAMATH_GPT_part1_part2_part3_l1525_152554


namespace NUMINAMATH_GPT_lilith_caps_collection_l1525_152572

theorem lilith_caps_collection :
  let first_year_caps := 3 * 12
  let subsequent_years_caps := 5 * 12 * 4
  let christmas_caps := 40 * 5
  let lost_caps := 15 * 5
  let total_caps := first_year_caps + subsequent_years_caps + christmas_caps - lost_caps
  total_caps = 401 := by
  sorry

end NUMINAMATH_GPT_lilith_caps_collection_l1525_152572


namespace NUMINAMATH_GPT_sum_of_their_ages_now_l1525_152537

variable (Nacho Divya : ℕ)

-- Conditions
def divya_current_age := 5
def nacho_in_5_years := 3 * (divya_current_age + 5)

-- Definition to determine current age of Nacho
def nacho_current_age := nacho_in_5_years - 5

-- Sum of current ages
def sum_of_ages := divya_current_age + nacho_current_age

-- Theorem to prove the sum of their ages now is 30
theorem sum_of_their_ages_now : sum_of_ages = 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_their_ages_now_l1525_152537


namespace NUMINAMATH_GPT_average_viewer_watches_two_videos_daily_l1525_152567

variable (V : ℕ)
variable (video_time : ℕ := 7)
variable (ad_time : ℕ := 3)
variable (total_time : ℕ := 17)

theorem average_viewer_watches_two_videos_daily :
  7 * V + 3 = 17 → V = 2 := 
by
  intro h
  have h1 : 7 * V = 14 := by linarith
  have h2 : V = 2 := by linarith
  exact h2

end NUMINAMATH_GPT_average_viewer_watches_two_videos_daily_l1525_152567


namespace NUMINAMATH_GPT_isosceles_with_base_c_l1525_152591

theorem isosceles_with_base_c (a b c: ℝ) (h: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (triangle_rel: 1/a - 1/b + 1/c = 1/(a - b + c)) : a = c ∨ b = c :=
sorry

end NUMINAMATH_GPT_isosceles_with_base_c_l1525_152591


namespace NUMINAMATH_GPT_taxi_trip_miles_l1525_152550

theorem taxi_trip_miles 
  (initial_fee : ℝ := 2.35)
  (additional_charge : ℝ := 0.35)
  (segment_length : ℝ := 2/5)
  (total_charge : ℝ := 5.50) :
  ∃ (miles : ℝ), total_charge = initial_fee + additional_charge * (miles / segment_length) ∧ miles = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_taxi_trip_miles_l1525_152550


namespace NUMINAMATH_GPT_sqrt_exp_cube_l1525_152599

theorem sqrt_exp_cube :
  ((Real.sqrt ((Real.sqrt 5)^4))^3 = 125) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_exp_cube_l1525_152599


namespace NUMINAMATH_GPT_trig_identity_simplification_l1525_152502

theorem trig_identity_simplification (θ : ℝ) (hθ : θ = 15 * Real.pi / 180) :
  (Real.sqrt 3 / 2 - Real.sqrt 3 * (Real.sin θ) ^ 2) = 3 / 4 := 
by sorry

end NUMINAMATH_GPT_trig_identity_simplification_l1525_152502


namespace NUMINAMATH_GPT_cakes_left_l1525_152560

def initial_cakes : ℕ := 62
def additional_cakes : ℕ := 149
def cakes_sold : ℕ := 144

theorem cakes_left : (initial_cakes + additional_cakes) - cakes_sold = 67 :=
by
  sorry

end NUMINAMATH_GPT_cakes_left_l1525_152560


namespace NUMINAMATH_GPT_not_p_suff_not_q_l1525_152514

theorem not_p_suff_not_q (x : ℝ) :
  ¬(|x| ≥ 1) → ¬(x^2 + x - 6 ≥ 0) :=
sorry

end NUMINAMATH_GPT_not_p_suff_not_q_l1525_152514


namespace NUMINAMATH_GPT_solve_x_eqns_solve_y_eqns_l1525_152562

theorem solve_x_eqns : ∀ x : ℝ, 2 * x^2 = 8 * x ↔ (x = 0 ∨ x = 4) :=
by
  intro x
  sorry

theorem solve_y_eqns : ∀ y : ℝ, y^2 - 10 * y - 1 = 0 ↔ (y = 5 + Real.sqrt 26 ∨ y = 5 - Real.sqrt 26) :=
by
  intro y
  sorry

end NUMINAMATH_GPT_solve_x_eqns_solve_y_eqns_l1525_152562


namespace NUMINAMATH_GPT_time_on_wednesday_is_40_minutes_l1525_152555

def hours_to_minutes (h : ℚ) : ℚ := h * 60

def time_monday : ℚ := hours_to_minutes (3 / 4)
def time_tuesday : ℚ := hours_to_minutes (1 / 2)
def time_wednesday (w : ℚ) : ℚ := w
def time_thursday : ℚ := hours_to_minutes (5 / 6)
def time_friday : ℚ := 75
def total_time : ℚ := hours_to_minutes 4

theorem time_on_wednesday_is_40_minutes (w : ℚ) 
    (h1 : time_monday = 45) 
    (h2 : time_tuesday = 30) 
    (h3 : time_thursday = 50) 
    (h4 : time_friday = 75)
    (h5 : total_time = 240) 
    (h6 : total_time = time_monday + time_tuesday + time_wednesday w + time_thursday + time_friday) 
    : w = 40 := 
by 
  sorry

end NUMINAMATH_GPT_time_on_wednesday_is_40_minutes_l1525_152555


namespace NUMINAMATH_GPT_compute_value_l1525_152558

theorem compute_value {a b : ℝ} 
  (h1 : ∀ x, (x + a) * (x + b) * (x + 12) = 0 → x ≠ -3 → x = -a ∨ x = -b ∨ x = -12)
  (h2 : ∀ x, (x + 2 * a) * (x + 3) * (x + 6) = 0 → x ≠ -b ∧ x ≠ -12 → x = -3) :
  100 * (3 / 2) + 6 = 156 :=
by
  sorry

end NUMINAMATH_GPT_compute_value_l1525_152558


namespace NUMINAMATH_GPT_inconsistent_b_positive_l1525_152513

theorem inconsistent_b_positive
  (a b c : ℝ)
  (h_solution_set : ∀ x : ℝ, -2 < x ∧ x < 3 / 2 → ax^2 + bx + c > 0) :
  ¬ b > 0 :=
sorry

end NUMINAMATH_GPT_inconsistent_b_positive_l1525_152513


namespace NUMINAMATH_GPT_perimeter_eq_120_plus_2_sqrt_1298_l1525_152510

noncomputable def total_perimeter_of_two_quadrilaterals (AB BC CD : ℝ) (AC : ℝ := Real.sqrt (AB ^ 2 + BC ^ 2)) (AD : ℝ := Real.sqrt (AC ^ 2 + CD ^ 2)) : ℝ :=
2 * (AB + BC + CD + AD)

theorem perimeter_eq_120_plus_2_sqrt_1298 (hAB : AB = 15) (hBC : BC = 28) (hCD : CD = 17) :
  total_perimeter_of_two_quadrilaterals 15 28 17 = 120 + 2 * Real.sqrt 1298 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_eq_120_plus_2_sqrt_1298_l1525_152510


namespace NUMINAMATH_GPT_find_a_l1525_152589

theorem find_a (x y z a : ℝ) (k : ℝ) (h1 : x = 2 * k) (h2 : y = 3 * k) (h3 : z = 5 * k)
    (h4 : x + y + z = 100) (h5 : y = a * x - 10) : a = 2 :=
  sorry

end NUMINAMATH_GPT_find_a_l1525_152589
