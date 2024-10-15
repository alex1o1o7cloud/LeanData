import Mathlib

namespace NUMINAMATH_GPT_bob_distance_when_meet_l430_43084

-- Definitions of the variables and conditions
def distance_XY : ℝ := 40
def yolanda_rate : ℝ := 2  -- Yolanda's walking rate in miles per hour
def bob_rate : ℝ := 4      -- Bob's walking rate in miles per hour
def yolanda_start_time : ℝ := 1 -- Yolanda starts 1 hour earlier 

-- Prove that Bob has walked 25.33 miles when he meets Yolanda
theorem bob_distance_when_meet : 
  ∃ t : ℝ, 2 * (t + yolanda_start_time) + 4 * t = distance_XY ∧ (4 * t = 25.33) := 
by
  sorry

end NUMINAMATH_GPT_bob_distance_when_meet_l430_43084


namespace NUMINAMATH_GPT_probability_at_least_one_alarm_on_time_l430_43078

noncomputable def P_alarm_A_on : ℝ := 0.80
noncomputable def P_alarm_B_on : ℝ := 0.90

theorem probability_at_least_one_alarm_on_time :
  (1 - (1 - P_alarm_A_on) * (1 - P_alarm_B_on)) = 0.98 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_alarm_on_time_l430_43078


namespace NUMINAMATH_GPT_tangent_line_to_circle_l430_43083

theorem tangent_line_to_circle (c : ℝ) (h : 0 < c) : 
  (∃ (x y : ℝ), x^2 + y^2 = 8 ∧ x + y = c) ↔ c = 4 :=
by sorry

end NUMINAMATH_GPT_tangent_line_to_circle_l430_43083


namespace NUMINAMATH_GPT_dog_roaming_area_comparison_l430_43020

theorem dog_roaming_area_comparison :
  let r := 10
  let a1 := (1/2) * Real.pi * r^2
  let a2 := (3/4) * Real.pi * r^2 - (1/4) * Real.pi * 6^2 
  a2 > a1 ∧ a2 - a1 = 16 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_dog_roaming_area_comparison_l430_43020


namespace NUMINAMATH_GPT_integer_solutions_l430_43045

theorem integer_solutions (n : ℕ) :
  n = 7 ↔ ∃ (x : ℤ), ∀ (x : ℤ), (3 * x^2 + 17 * x + 14 ≤ 20)  :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_l430_43045


namespace NUMINAMATH_GPT_geometric_sequence_properties_l430_43017

noncomputable def a_n (n : ℕ) : ℚ :=
if n = 0 then 1 / 4 else (1 / 4) * 2^(n-1)

def S_n (n : ℕ) : ℚ :=
(1/4) * (1 - 2^n) / (1 - 2)

theorem geometric_sequence_properties :
  (a_n 2 = 1 / 2) ∧ (∀ n : ℕ, 1 ≤ n → a_n n = 2^(n-3)) ∧ S_n 5 = 31 / 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_properties_l430_43017


namespace NUMINAMATH_GPT_greatest_c_for_expression_domain_all_real_l430_43062

theorem greatest_c_for_expression_domain_all_real :
  ∃ c : ℤ, c ≤ 7 ∧ c ^ 2 < 60 ∧ ∀ d : ℤ, d > 7 → ¬ (d ^ 2 < 60) := sorry

end NUMINAMATH_GPT_greatest_c_for_expression_domain_all_real_l430_43062


namespace NUMINAMATH_GPT_smallest_next_divisor_of_m_l430_43037

theorem smallest_next_divisor_of_m (m : ℕ) (h1 : m % 2 = 0) (h2 : 10000 ≤ m ∧ m < 100000) (h3 : 523 ∣ m) : 
  ∃ d : ℕ, 523 < d ∧ d ∣ m ∧ ∀ e : ℕ, 523 < e ∧ e ∣ m → d ≤ e :=
by
  sorry

end NUMINAMATH_GPT_smallest_next_divisor_of_m_l430_43037


namespace NUMINAMATH_GPT_factor_evaluate_l430_43068

theorem factor_evaluate (a b : ℤ) (h1 : a = 2) (h2 : b = -2) : 
  5 * a * (b - 2) + 2 * a * (2 - b) = -24 := by
  sorry

end NUMINAMATH_GPT_factor_evaluate_l430_43068


namespace NUMINAMATH_GPT_no_valid_placement_of_prisms_l430_43030

-- Definitions: Rectangular prism with edges parallel to OX, OY, and OZ axes.
structure RectPrism :=
  (x_interval : Set ℝ)
  (y_interval : Set ℝ)
  (z_interval : Set ℝ)

-- Function to determine if two rectangular prisms intersect.
def intersects (P Q : RectPrism) : Prop :=
  ¬ Disjoint P.x_interval Q.x_interval ∧
  ¬ Disjoint P.y_interval Q.y_interval ∧
  ¬ Disjoint P.z_interval Q.z_interval

-- Definition of the 12 rectangular prisms
def prisms := Fin 12 → RectPrism

-- Conditions for intersection:
def intersection_condition (prisms : prisms) : Prop :=
  ∀ i : Fin 12, ∀ j : Fin 12,
    (j = (i + 1) % 12) ∨ (j = (i - 1 + 12) % 12) ∨ intersects (prisms i) (prisms j)

theorem no_valid_placement_of_prisms :
  ¬ ∃ (prisms : prisms), intersection_condition prisms :=
sorry

end NUMINAMATH_GPT_no_valid_placement_of_prisms_l430_43030


namespace NUMINAMATH_GPT_polynomial_remainder_l430_43006

theorem polynomial_remainder (x : ℤ) : (x + 1) ∣ (x^15 + 1) ↔ x = -1 := sorry

end NUMINAMATH_GPT_polynomial_remainder_l430_43006


namespace NUMINAMATH_GPT_minimal_odd_sum_is_1683_l430_43010

/-!
# Proof Problem:
Prove that the minimal odd sum of two three-digit numbers and one four-digit number 
formed using the digits 0 through 9 exactly once is 1683.
-/
theorem minimal_odd_sum_is_1683 :
  ∃ (a b : ℕ) (c : ℕ), 
    100 ≤ a ∧ a < 1000 ∧ 
    100 ≤ b ∧ b < 1000 ∧ 
    1000 ≤ c ∧ c < 10000 ∧ 
    a + b + c % 2 = 1 ∧ 
    (∀ d e f : ℕ, 
      100 ≤ d ∧ d < 1000 ∧ 
      100 ≤ e ∧ e < 1000 ∧ 
      1000 ≤ f ∧ f < 10000 ∧ 
      d + e + f % 2 = 1 → a + b + c ≤ d + e + f) ∧ 
    a + b + c = 1683 := 
sorry

end NUMINAMATH_GPT_minimal_odd_sum_is_1683_l430_43010


namespace NUMINAMATH_GPT_time_to_traverse_nth_mile_l430_43039

theorem time_to_traverse_nth_mile (n : ℕ) (n_pos : n > 1) :
  let k := (1 / 2 : ℝ)
  let s_n := k / ((n-1) * (2 ^ (n-2)))
  let t_n := 1 / s_n
  t_n = 2 * (n-1) * 2^(n-2) := 
by sorry

end NUMINAMATH_GPT_time_to_traverse_nth_mile_l430_43039


namespace NUMINAMATH_GPT_molecular_weight_boric_acid_l430_43013

theorem molecular_weight_boric_acid :
  let H := 1.008  -- atomic weight of Hydrogen in g/mol
  let B := 10.81  -- atomic weight of Boron in g/mol
  let O := 16.00  -- atomic weight of Oxygen in g/mol
  let H3BO3 := 3 * H + B + 3 * O  -- molecular weight of H3BO3
  H3BO3 = 61.834 :=  -- correct molecular weight of H3BO3
by
  sorry

end NUMINAMATH_GPT_molecular_weight_boric_acid_l430_43013


namespace NUMINAMATH_GPT_find_a_l430_43096

theorem find_a :
  (∃ x1 x2, (x1 + x2 = -2 ∧ x1 * x2 = a) ∧ (∃ y1 y2, (y1 + y2 = - a ∧ y1 * y2 = 2) ∧ (x1^2 + x2^2 = y1^2 + y2^2))) → 
  (a = -4) := 
by
  sorry

end NUMINAMATH_GPT_find_a_l430_43096


namespace NUMINAMATH_GPT_problem_l430_43047

theorem problem (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_problem_l430_43047


namespace NUMINAMATH_GPT_base9_add_subtract_l430_43063

theorem base9_add_subtract :
  let n1 := 3 * 9^2 + 5 * 9 + 1
  let n2 := 4 * 9^2 + 6 * 9 + 5
  let n3 := 1 * 9^2 + 3 * 9 + 2
  let n4 := 1 * 9^2 + 4 * 9 + 7
  (n1 + n2 + n3 - n4 = 8 * 9^2 + 4 * 9 + 7) :=
by
  sorry

end NUMINAMATH_GPT_base9_add_subtract_l430_43063


namespace NUMINAMATH_GPT_math_problem_common_factors_and_multiples_l430_43050

-- Definitions
def a : ℕ := 180
def b : ℕ := 300

-- The Lean statement to be proved
theorem math_problem_common_factors_and_multiples :
    Nat.lcm a b = 900 ∧
    Nat.gcd a b = 60 ∧
    {d | d ∣ a ∧ d ∣ b} = {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60} :=
by
  sorry

end NUMINAMATH_GPT_math_problem_common_factors_and_multiples_l430_43050


namespace NUMINAMATH_GPT_total_earnings_to_afford_car_l430_43074

-- Define the earnings per month
def monthlyEarnings : ℕ := 4000

-- Define the savings per month
def monthlySavings : ℕ := 500

-- Define the total amount needed to buy the car
def totalNeeded : ℕ := 45000

-- Define the number of months needed to save enough money
def monthsToSave : ℕ := totalNeeded / monthlySavings

-- Theorem stating the total money earned before he saves enough to buy the car
theorem total_earnings_to_afford_car : monthsToSave * monthlyEarnings = 360000 := by
  sorry

end NUMINAMATH_GPT_total_earnings_to_afford_car_l430_43074


namespace NUMINAMATH_GPT_not_divisible_59_l430_43067

theorem not_divisible_59 (x y : ℕ) (hx : ¬ (59 ∣ x)) (hy : ¬ (59 ∣ y)) 
  (h : (3 * x + 28 * y) % 59 = 0) : (5 * x + 16 * y) % 59 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_not_divisible_59_l430_43067


namespace NUMINAMATH_GPT_compare_neg_fractions_l430_43046

theorem compare_neg_fractions : 
  (- (8:ℚ) / 21) > - (3 / 7) :=
by sorry

end NUMINAMATH_GPT_compare_neg_fractions_l430_43046


namespace NUMINAMATH_GPT_find_a_b_c_l430_43005

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_a_b_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hp1 : is_prime (a + b * c))
  (hp2 : is_prime (b + a * c))
  (hp3 : is_prime (c + a * b))
  (hdiv1 : (a + b * c) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)))
  (hdiv2 : (b + a * c) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)))
  (hdiv3 : (c + a * b) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1))) :
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end NUMINAMATH_GPT_find_a_b_c_l430_43005


namespace NUMINAMATH_GPT_m_value_l430_43085

open Polynomial

noncomputable def f (m : ℚ) : Polynomial ℚ := X^4 - 5*X^2 + 4*X - C m

theorem m_value (m : ℚ) : (2 * X + 1) ∣ f m ↔ m = -51/16 := by sorry

end NUMINAMATH_GPT_m_value_l430_43085


namespace NUMINAMATH_GPT_area_of_rectangle_abcd_l430_43079

-- Definition of the problem's conditions and question
def small_square_side_length : ℝ := 1
def large_square_side_length : ℝ := 1.5
def area_rectangle_abc : ℝ := 4.5

-- Lean 4 statement: Prove the area of rectangle ABCD is 4.5 square inches
theorem area_of_rectangle_abcd :
  (3 * small_square_side_length) * large_square_side_length = area_rectangle_abc :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_abcd_l430_43079


namespace NUMINAMATH_GPT_smallest_nonfactor_product_of_factors_of_48_l430_43023

theorem smallest_nonfactor_product_of_factors_of_48 :
  ∃ a b : ℕ, a ≠ b ∧ a ∣ 48 ∧ b ∣ 48 ∧ ¬ (a * b ∣ 48) ∧ (∀ c d : ℕ, c ≠ d ∧ c ∣ 48 ∧ d ∣ 48 ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ a * b = 18 :=
sorry

end NUMINAMATH_GPT_smallest_nonfactor_product_of_factors_of_48_l430_43023


namespace NUMINAMATH_GPT_proof_problem_l430_43029

noncomputable def log2 : ℝ := Real.log 3 / Real.log 2
noncomputable def log5 : ℝ := Real.log 3 / Real.log 5

variables {x y : ℝ}

theorem proof_problem
  (h1 : log2 > 1)
  (h2 : 0 < log5 ∧ log5 < 1)
  (h3 : (log2^x - log5^x) ≥ (log2^(-y) - log5^(-y))) :
  x + y ≥ 0 :=
sorry

end NUMINAMATH_GPT_proof_problem_l430_43029


namespace NUMINAMATH_GPT_ratio_x_y_z_l430_43049

variables (x y z : ℝ)

theorem ratio_x_y_z (h1 : 0.60 * x = 0.30 * y) 
                    (h2 : 0.80 * z = 0.40 * x) 
                    (h3 : z = 2 * y) : 
                    x / y = 4 ∧ y / y = 1 ∧ z / y = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_x_y_z_l430_43049


namespace NUMINAMATH_GPT_remainder_of_series_div_9_l430_43071

def sum (n : Nat) : Nat := n * (n + 1) / 2

theorem remainder_of_series_div_9 : (sum 20) % 9 = 3 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_remainder_of_series_div_9_l430_43071


namespace NUMINAMATH_GPT_find_g_three_l430_43043

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_three (h : ∀ x : ℝ, g (3^x) + (x + 1) * g (3^(-x)) = 3) : g 3 = -3 :=
sorry

end NUMINAMATH_GPT_find_g_three_l430_43043


namespace NUMINAMATH_GPT_correct_operation_l430_43031

theorem correct_operation :
  (∀ a : ℕ, a ^ 3 * a ^ 2 = a ^ 5) ∧
  (∀ a : ℕ, a + a ^ 2 ≠ a ^ 3) ∧
  (∀ a : ℕ, 6 * a ^ 2 / (2 * a ^ 2) = 3) ∧
  (∀ a : ℕ, (3 * a ^ 2) ^ 3 ≠ 9 * a ^ 6) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l430_43031


namespace NUMINAMATH_GPT_neg_sin_prop_iff_l430_43061

theorem neg_sin_prop_iff :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by sorry

end NUMINAMATH_GPT_neg_sin_prop_iff_l430_43061


namespace NUMINAMATH_GPT_acres_used_for_corn_l430_43003

-- Define the conditions in the problem:
def total_land : ℕ := 1034
def ratio_beans : ℕ := 5
def ratio_wheat : ℕ := 2
def ratio_corn : ℕ := 4
def total_ratio : ℕ := ratio_beans + ratio_wheat + ratio_corn

-- Proof problem statement: Prove the number of acres used for corn is 376 acres
theorem acres_used_for_corn : total_land * ratio_corn / total_ratio = 376 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_acres_used_for_corn_l430_43003


namespace NUMINAMATH_GPT_inequality_not_hold_l430_43021

theorem inequality_not_hold (a b : ℝ) (h : a < b ∧ b < 0) : (1 / (a - b) < 1 / a) :=
by
  sorry

end NUMINAMATH_GPT_inequality_not_hold_l430_43021


namespace NUMINAMATH_GPT_cooling_constant_l430_43051

theorem cooling_constant (θ0 θ1 θ t k : ℝ) (h1 : θ1 = 60) (h0 : θ0 = 15) (ht : t = 3) (hθ : θ = 42)
  (h_temp_formula : θ = θ0 + (θ1 - θ0) * Real.exp (-k * t)) :
  k = 0.17 :=
by sorry

end NUMINAMATH_GPT_cooling_constant_l430_43051


namespace NUMINAMATH_GPT_rationalize_denominator_and_product_l430_43015

theorem rationalize_denominator_and_product :
  let A := -11
  let B := -5
  let C := 5
  let expr := (3 + Real.sqrt 5) / (2 - Real.sqrt 5)
  (expr * (2 + Real.sqrt 5) / (2 + Real.sqrt 5) = A + B * Real.sqrt C) ∧ (A * B * C = 275) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_and_product_l430_43015


namespace NUMINAMATH_GPT_symmetric_with_origin_l430_43057

-- Define the original point P
def P : ℝ × ℝ := (2, -3)

-- Define the function for finding the symmetric point with respect to the origin
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- Prove that the symmetric point of P with respect to the origin is (-2, 3)
theorem symmetric_with_origin :
  symmetric_point P = (-2, 3) :=
by
  -- Placeholders for proof
  sorry

end NUMINAMATH_GPT_symmetric_with_origin_l430_43057


namespace NUMINAMATH_GPT_michael_twice_jacob_l430_43095

variable {J M Y : ℕ}

theorem michael_twice_jacob :
  (J + 4 = 13) → (M = J + 12) → (M + Y = 2 * (J + Y)) → (Y = 3) := by
  sorry

end NUMINAMATH_GPT_michael_twice_jacob_l430_43095


namespace NUMINAMATH_GPT_total_sum_step_l430_43098

-- Defining the conditions
def step_1_sum : ℕ := 2

-- Define the inductive process
def total_sum_labels (n : ℕ) : ℕ :=
  if n = 1 then step_1_sum
  else 2 * 3^(n - 1)

-- The theorem to prove
theorem total_sum_step (n : ℕ) : 
  total_sum_labels n = 2 * 3^(n - 1) :=
by
  sorry

end NUMINAMATH_GPT_total_sum_step_l430_43098


namespace NUMINAMATH_GPT_total_cost_of_new_movie_l430_43059

noncomputable def previous_movie_length_hours : ℕ := 2
noncomputable def new_movie_length_increase_percent : ℕ := 60
noncomputable def previous_movie_cost_per_minute : ℕ := 50
noncomputable def new_movie_cost_per_minute_factor : ℕ := 2 

theorem total_cost_of_new_movie : 
  let new_movie_length_hours := previous_movie_length_hours + (previous_movie_length_hours * new_movie_length_increase_percent / 100)
  let new_movie_length_minutes := new_movie_length_hours * 60
  let new_movie_cost_per_minute := previous_movie_cost_per_minute * new_movie_cost_per_minute_factor
  let total_cost := new_movie_length_minutes * new_movie_cost_per_minute
  total_cost = 19200 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_of_new_movie_l430_43059


namespace NUMINAMATH_GPT_octagon_has_20_diagonals_l430_43041

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_GPT_octagon_has_20_diagonals_l430_43041


namespace NUMINAMATH_GPT_goods_train_length_l430_43090

theorem goods_train_length 
  (v_kmph : ℝ) (L_p : ℝ) (t : ℝ) (v_mps : ℝ) (d : ℝ) (L_t : ℝ) 
  (h1 : v_kmph = 96) 
  (h2 : L_p = 480) 
  (h3 : t = 36) 
  (h4 : v_mps = v_kmph * (5/18)) 
  (h5 : d = v_mps * t) : 
  L_t = d - L_p :=
sorry

end NUMINAMATH_GPT_goods_train_length_l430_43090


namespace NUMINAMATH_GPT_max_dn_eq_401_l430_43080

open BigOperators

def a (n : ℕ) : ℕ := 100 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_dn_eq_401 : ∃ n, d n = 401 ∧ ∀ m, d m ≤ 401 := by
  -- Proof will be filled here
  sorry

end NUMINAMATH_GPT_max_dn_eq_401_l430_43080


namespace NUMINAMATH_GPT_least_whole_number_subtracted_l430_43025

theorem least_whole_number_subtracted {x : ℕ} (h : 6 > x ∧ 7 > x) :
  (6 - x) / (7 - x : ℝ) < 16 / 21 -> x = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_least_whole_number_subtracted_l430_43025


namespace NUMINAMATH_GPT_basketball_team_win_requirement_l430_43055

noncomputable def basketball_win_percentage_goal (games_played_so_far games_won_so_far games_remaining win_percentage_goal : ℕ) : ℕ :=
  let total_games := games_played_so_far + games_remaining
  let required_wins := (win_percentage_goal * total_games) / 100
  required_wins - games_won_so_far

theorem basketball_team_win_requirement :
  basketball_win_percentage_goal 60 45 50 75 = 38 := 
by
  sorry

end NUMINAMATH_GPT_basketball_team_win_requirement_l430_43055


namespace NUMINAMATH_GPT_triangle_perimeter_l430_43014

/-- Given a triangle with two sides of lengths 2 and 5, and the third side being a root of the equation
    x^2 - 8x + 12 = 0, the perimeter of the triangle is 13. --/
theorem triangle_perimeter
  (a b : ℕ) 
  (ha : a = 2) 
  (hb : b = 5)
  (c : ℕ)
  (h_c_root : c * c - 8 * c + 12 = 0)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 13 := 
sorry

end NUMINAMATH_GPT_triangle_perimeter_l430_43014


namespace NUMINAMATH_GPT_beta_speed_l430_43086

theorem beta_speed (d : ℕ) (S_s : ℕ) (t : ℕ) (S_b : ℕ) :
  d = 490 ∧ S_s = 37 ∧ t = 7 ∧ (S_s * t) + (S_b * t) = d → S_b = 33 := by
  sorry

end NUMINAMATH_GPT_beta_speed_l430_43086


namespace NUMINAMATH_GPT_union_of_A_and_B_l430_43018

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 2 * x < 0}
noncomputable def B : Set ℝ := {x : ℝ | 1 < x }

theorem union_of_A_and_B : A ∪ B = {x : ℝ | 0 < x} :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l430_43018


namespace NUMINAMATH_GPT_find_point_P_l430_43033

/-- 
Given two points A and B, find the coordinates of point P that lies on the line AB
and satisfies that the distance from A to P is half the vector from A to B.
-/
theorem find_point_P 
  (A B : ℝ × ℝ) 
  (hA : A = (3, -4)) 
  (hB : B = (-9, 2)) 
  (P : ℝ × ℝ) 
  (hP : P.1 - A.1 = (1/2) * (B.1 - A.1) ∧ P.2 - A.2 = (1/2) * (B.2 - A.2)) : 
  P = (-3, -1) := 
sorry

end NUMINAMATH_GPT_find_point_P_l430_43033


namespace NUMINAMATH_GPT_final_score_is_correct_l430_43066

-- Definitions based on given conditions
def speechContentScore : ℕ := 90
def speechDeliveryScore : ℕ := 85
def weightContent : ℕ := 6
def weightDelivery : ℕ := 4

-- The final score calculation theorem
theorem final_score_is_correct : 
  (speechContentScore * weightContent + speechDeliveryScore * weightDelivery) / (weightContent + weightDelivery) = 88 :=
  by
    sorry

end NUMINAMATH_GPT_final_score_is_correct_l430_43066


namespace NUMINAMATH_GPT_min_value_f_l430_43026

noncomputable def f (a x : ℝ) : ℝ := x ^ 2 - 2 * a * x - 1

theorem min_value_f (a : ℝ) : 
  (∀ x ∈ (Set.Icc (-1 : ℝ) 1), f a x ≥ 
    if a < -1 then 2 * a 
    else if -1 ≤ a ∧ a ≤ 1 then -1 - a ^ 2 
    else -2 * a) := 
by
  sorry

end NUMINAMATH_GPT_min_value_f_l430_43026


namespace NUMINAMATH_GPT_least_positive_value_of_cubic_eq_l430_43089

theorem least_positive_value_of_cubic_eq (x y z w : ℕ) 
  (hx : Nat.Prime x) (hy : Nat.Prime y) 
  (hz : Nat.Prime z) (hw : Nat.Prime w) 
  (sum_lt_50 : x + y + z + w < 50) : 
  24 * x^3 + 16 * y^3 - 7 * z^3 + 5 * w^3 = 1464 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_value_of_cubic_eq_l430_43089


namespace NUMINAMATH_GPT_ratio_doctors_to_lawyers_l430_43000

-- Definitions based on conditions
def average_age_doctors := 35
def average_age_lawyers := 50
def combined_average_age := 40

-- Define variables
variables (d l : ℕ) -- d is number of doctors, l is number of lawyers

-- Hypothesis based on the problem statement
axiom h : (average_age_doctors * d + average_age_lawyers * l) = combined_average_age * (d + l)

-- The theorem we need to prove is the ratio of doctors to lawyers is 2:1
theorem ratio_doctors_to_lawyers : d = 2 * l :=
by sorry

end NUMINAMATH_GPT_ratio_doctors_to_lawyers_l430_43000


namespace NUMINAMATH_GPT_max_sum_condition_l430_43072

theorem max_sum_condition (a b : ℕ) (h1 : 10 ≤ a ∧ a ≤ 99) (h2 : 10 ≤ b ∧ b ≤ 99) (h3 : Nat.gcd a b = 6) : a + b ≤ 186 :=
sorry

end NUMINAMATH_GPT_max_sum_condition_l430_43072


namespace NUMINAMATH_GPT_age_of_B_l430_43036

variables (A B C : ℕ)

theorem age_of_B (h1 : (A + B + C) / 3 = 25) (h2 : (A + C) / 2 = 29) : B = 17 := 
by
  -- Skipping the proof steps
  sorry

end NUMINAMATH_GPT_age_of_B_l430_43036


namespace NUMINAMATH_GPT_total_ingredients_used_l430_43052

theorem total_ingredients_used (water oliveOil salt : ℕ) 
  (h_ratio : water / oliveOil = 3 / 2) 
  (h_salt : water / salt = 3 / 1)
  (h_water_cups : water = 15) : 
  water + oliveOil + salt = 30 :=
sorry

end NUMINAMATH_GPT_total_ingredients_used_l430_43052


namespace NUMINAMATH_GPT_empty_subset_singleton_zero_l430_43024

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) :=
by
  sorry

end NUMINAMATH_GPT_empty_subset_singleton_zero_l430_43024


namespace NUMINAMATH_GPT_crayons_problem_l430_43034

theorem crayons_problem 
  (total_crayons : ℕ)
  (red_crayons : ℕ)
  (blue_crayons : ℕ)
  (green_crayons : ℕ)
  (pink_crayons : ℕ)
  (h1 : total_crayons = 24)
  (h2 : red_crayons = 8)
  (h3 : blue_crayons = 6)
  (h4 : green_crayons = 2 / 3 * blue_crayons)
  (h5 : pink_crayons = total_crayons - red_crayons - blue_crayons - green_crayons) :
  pink_crayons = 6 :=
by
  sorry

end NUMINAMATH_GPT_crayons_problem_l430_43034


namespace NUMINAMATH_GPT_probability_of_selecting_same_gender_l430_43091

def number_of_ways_to_choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_of_selecting_same_gender (total_students male_students female_students : ℕ) (h1 : total_students = 10) (h2 : male_students = 2) (h3 : female_students = 8) : 
  let total_combinations := number_of_ways_to_choose_two total_students
  let male_combinations := number_of_ways_to_choose_two male_students
  let female_combinations := number_of_ways_to_choose_two female_students
  let favorable_combinations := male_combinations + female_combinations
  total_combinations = 45 ∧
  male_combinations = 1 ∧
  female_combinations = 28 ∧
  favorable_combinations = 29 ∧
  (favorable_combinations : ℚ) / total_combinations = 29 / 45 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_same_gender_l430_43091


namespace NUMINAMATH_GPT_alpha_value_l430_43054

theorem alpha_value (b : ℝ) : (∀ x : ℝ, (|2 * x - 3| < 2) ↔ (x^2 + -3 * x + b < 0)) :=
by
  sorry

end NUMINAMATH_GPT_alpha_value_l430_43054


namespace NUMINAMATH_GPT_gab_score_ratio_l430_43094

theorem gab_score_ratio (S G C O : ℕ) (h1 : S = 20) (h2 : C = 2 * G) (h3 : O = 85) (h4 : S + G + C = O + 55) :
  G / S = 2 := 
by 
  sorry

end NUMINAMATH_GPT_gab_score_ratio_l430_43094


namespace NUMINAMATH_GPT_P_subset_Q_l430_43073

def P : Set ℕ := {1, 2, 4}
def Q : Set ℕ := {1, 2, 4, 8}

theorem P_subset_Q : P ⊂ Q := by
  sorry

end NUMINAMATH_GPT_P_subset_Q_l430_43073


namespace NUMINAMATH_GPT_smallest_integer_y_l430_43002

theorem smallest_integer_y (y : ℤ) (h : 7 - 3 * y < 20) : ∃ (y : ℤ), y = -4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_y_l430_43002


namespace NUMINAMATH_GPT_calc_expression_is_24_l430_43060

def calc_expression : ℕ := (30 / (8 + 2 - 5)) * 4

theorem calc_expression_is_24 : calc_expression = 24 :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_is_24_l430_43060


namespace NUMINAMATH_GPT_workers_and_days_l430_43011

theorem workers_and_days (x y : ℕ) (h1 : x * y = (x - 20) * (y + 5)) (h2 : x * y = (x + 15) * (y - 2)) :
  x = 60 ∧ y = 10 := 
by {
  sorry
}

end NUMINAMATH_GPT_workers_and_days_l430_43011


namespace NUMINAMATH_GPT_solve_for_x_l430_43070

theorem solve_for_x : ∃ x : ℝ, x^4 + 10 * x^3 + 9 * x^2 - 50 * x - 56 = 0 ↔ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l430_43070


namespace NUMINAMATH_GPT_number_of_factors_multiples_of_360_l430_43042

def n : ℕ := 2^10 * 3^14 * 5^8

theorem number_of_factors_multiples_of_360 (n : ℕ) (hn : n = 2^10 * 3^14 * 5^8) : 
  ∃ (k : ℕ), k = 832 ∧ 
  (∀ m : ℕ, m ∣ n → 360 ∣ m → k = 8 * 13 * 8) := 
sorry

end NUMINAMATH_GPT_number_of_factors_multiples_of_360_l430_43042


namespace NUMINAMATH_GPT_min_sum_a_b_l430_43065

theorem min_sum_a_b (a b : ℝ) (h1 : 4 * a + b = 1) (h2 : 0 < a) (h3 : 0 < b) :
  a + b ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_sum_a_b_l430_43065


namespace NUMINAMATH_GPT_no_point_on_line_y_eq_2x_l430_43077

theorem no_point_on_line_y_eq_2x
  (marked : Set (ℕ × ℕ))
  (initial_points : { p // p ∈ [(1, 1), (2, 3), (4, 5), (999, 111)] })
  (rule1 : ∀ a b, (a, b) ∈ marked → (b, a) ∈ marked ∧ (a - b, a + b) ∈ marked)
  (rule2 : ∀ a b c d, (a, b) ∈ marked ∧ (c, d) ∈ marked → (a * d + b * c, 4 * a * c - 4 * b * d) ∈ marked) :
  ∃ x, (x, 2 * x) ∈ marked → False := sorry

end NUMINAMATH_GPT_no_point_on_line_y_eq_2x_l430_43077


namespace NUMINAMATH_GPT_asymptotes_and_eccentricity_of_hyperbola_l430_43053

noncomputable def hyperbola_asymptotes_and_eccentricity : Prop :=
  let a := 1
  let b := Real.sqrt 2
  let c := Real.sqrt 3
  ∀ (x y : ℝ), x^2 - (y^2 / 2) = 1 →
    ((y = 2 * x ∨ y = -2 * x) ∧ Real.sqrt (1 + (b^2 / a^2)) = c)

theorem asymptotes_and_eccentricity_of_hyperbola :
  hyperbola_asymptotes_and_eccentricity :=
by
  sorry

end NUMINAMATH_GPT_asymptotes_and_eccentricity_of_hyperbola_l430_43053


namespace NUMINAMATH_GPT_max_value_sum_l430_43056

variable (n : ℕ) (x : Fin n → ℝ)

theorem max_value_sum 
  (h1 : ∀ i, 0 ≤ x i)
  (h2 : 2 ≤ n)
  (h3 : (Finset.univ : Finset (Fin n)).sum x = 1) :
  ∃ max_val, max_val = (1 / 4) :=
sorry

end NUMINAMATH_GPT_max_value_sum_l430_43056


namespace NUMINAMATH_GPT_find_missing_number_l430_43035

theorem find_missing_number (n : ℝ) :
  (0.0088 * 4.5) / (0.05 * 0.1 * n) = 990 → n = 0.008 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_missing_number_l430_43035


namespace NUMINAMATH_GPT_concentric_circles_circumference_difference_l430_43027

theorem concentric_circles_circumference_difference :
  ∀ (radius_diff inner_diameter : ℝ),
  radius_diff = 15 →
  inner_diameter = 50 →
  ((π * (inner_diameter + 2 * radius_diff)) - (π * inner_diameter)) = 30 * π :=
by
  sorry

end NUMINAMATH_GPT_concentric_circles_circumference_difference_l430_43027


namespace NUMINAMATH_GPT_samantha_score_l430_43097

variables (correct_answers geometry_correct_answers incorrect_answers unanswered_questions : ℕ)
          (points_per_correct : ℝ := 1) (additional_geometry_points : ℝ := 0.5)

def total_score (correct_answers geometry_correct_answers : ℕ) : ℝ :=
  correct_answers * points_per_correct + geometry_correct_answers * additional_geometry_points

theorem samantha_score 
  (Samantha_correct : correct_answers = 15)
  (Samantha_geometry : geometry_correct_answers = 4)
  (Samantha_incorrect : incorrect_answers = 5)
  (Samantha_unanswered : unanswered_questions = 5) :
  total_score correct_answers geometry_correct_answers = 17 := 
by
  sorry

end NUMINAMATH_GPT_samantha_score_l430_43097


namespace NUMINAMATH_GPT_find_a_l430_43040

theorem find_a (a b c : ℕ) (h1 : (18 ^ a) * (9 ^ (3 * a - 1)) * (c ^ (2 * a - 3)) = (2 ^ 7) * (3 ^ b)) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : a = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l430_43040


namespace NUMINAMATH_GPT_Kayla_points_on_first_level_l430_43019

theorem Kayla_points_on_first_level
(points_2 : ℕ) (points_3 : ℕ) (points_4 : ℕ) (points_5 : ℕ) (points_6 : ℕ)
(h2 : points_2 = 3) (h3 : points_3 = 5) (h4 : points_4 = 8) (h5 : points_5 = 12) (h6 : points_6 = 17) :
  ∃ (points_1 : ℕ), 
    (points_3 - points_2 = 2) ∧ 
    (points_4 - points_3 = 3) ∧ 
    (points_5 - points_4 = 4) ∧ 
    (points_6 - points_5 = 5) ∧ 
    (points_2 - points_1 = 1) ∧ 
    points_1 = 2 :=
by
  use 2
  repeat { split }
  sorry

end NUMINAMATH_GPT_Kayla_points_on_first_level_l430_43019


namespace NUMINAMATH_GPT_inequality_one_inequality_two_l430_43092

theorem inequality_one (x : ℝ) : 7 * x - 2 < 3 * (x + 2) → x < 2 :=
by
  sorry

theorem inequality_two (x : ℝ) : (x - 1) / 3 ≥ (x - 3) / 12 + 1 → x ≥ 13 / 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_one_inequality_two_l430_43092


namespace NUMINAMATH_GPT_range_of_a_l430_43004

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 3) : -2 ≤ a ∧ a ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l430_43004


namespace NUMINAMATH_GPT_prove_a_eq_b_l430_43012

theorem prove_a_eq_b (a b : ℝ) (h : 1 / (3 * a) + 2 / (3 * b) = 3 / (a + 2 * b)) : a = b :=
sorry

end NUMINAMATH_GPT_prove_a_eq_b_l430_43012


namespace NUMINAMATH_GPT_range_of_a_l430_43044

open Real

theorem range_of_a
  (a : ℝ)
  (curve : ∀ θ : ℝ, ∃ p : ℝ × ℝ, p = (a + 2 * cos θ, a + 2 * sin θ))
  (distance_two_points : ∀ θ : ℝ, dist (0,0) (a + 2 * cos θ, a + 2 * sin θ) = 2) :
  (-2 * sqrt 2 < a ∧ a < 0) ∨ (0 < a ∧ a < 2 * sqrt 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l430_43044


namespace NUMINAMATH_GPT_contractor_absent_days_l430_43001

variable (x y : ℝ) -- x for the number of days worked, y for the number of days absent

-- Conditions
def eng_days := x + y = 30
def total_money := 25 * x - 7.5 * y = 425

-- Theorem
theorem contractor_absent_days (x y : ℝ) (h1 : eng_days x y) (h2 : total_money x y) : y = 10 := 
sorry

end NUMINAMATH_GPT_contractor_absent_days_l430_43001


namespace NUMINAMATH_GPT_sequence_sum_a_b_l430_43081

theorem sequence_sum_a_b (a b : ℕ) (a_seq : ℕ → ℕ) 
  (h1 : a_seq 1 = a)
  (h2 : a_seq 2 = b)
  (h3 : ∀ n ≥ 1, a_seq (n+2) = (a_seq n + 2018) / (a_seq (n+1) + 1)) :
  a + b = 1011 ∨ a + b = 2019 :=
sorry

end NUMINAMATH_GPT_sequence_sum_a_b_l430_43081


namespace NUMINAMATH_GPT_cos_theta_when_f_maximizes_l430_43007

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x)

theorem cos_theta_when_f_maximizes (θ : ℝ) (h : ∀ x, f x ≤ f θ) : Real.cos θ = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_theta_when_f_maximizes_l430_43007


namespace NUMINAMATH_GPT_set_B_forms_triangle_l430_43028

theorem set_B_forms_triangle (a b c : ℝ) (h1 : a = 25) (h2 : b = 24) (h3 : c = 7):
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end NUMINAMATH_GPT_set_B_forms_triangle_l430_43028


namespace NUMINAMATH_GPT_distinct_real_roots_iff_m_lt_13_over_4_equal_real_roots_root_eq_3_over_2_l430_43058

variable (m : ℝ)

-- Part 1: Prove that if the quadratic equation has two distinct real roots, then m < 13/4.
theorem distinct_real_roots_iff_m_lt_13_over_4 (h : (3 * 3 - 4 * (m - 1)) > 0) : m < 13 / 4 := 
by
  sorry

-- Part 2: Prove that if the quadratic equation has two equal real roots, then the root is 3/2.
theorem equal_real_roots_root_eq_3_over_2 (h : (3 * 3 - 4 * (m - 1)) = 0) : m = 13 / 4 ∧ ∀ x, (x^2 + 3 * x + (13/4 - 1) = 0) → x = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_distinct_real_roots_iff_m_lt_13_over_4_equal_real_roots_root_eq_3_over_2_l430_43058


namespace NUMINAMATH_GPT_timber_logging_years_l430_43076

theorem timber_logging_years 
  (V0 : ℝ) (r : ℝ) (V : ℝ) (t : ℝ)
  (hV0 : V0 = 100000)
  (hr : r = 0.08)
  (hV : V = 400000)
  (hformula : V = V0 * (1 + r)^t)
  : t = (Real.log 4 / Real.log 1.08) :=
by
  sorry

end NUMINAMATH_GPT_timber_logging_years_l430_43076


namespace NUMINAMATH_GPT_find_S5_l430_43009

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

axiom arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : ∀ n : ℕ, a (n + 1) = a 1 + n * d
axiom sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : ∀ n : ℕ, S n = n / 2 * (a 1 + a n)

theorem find_S5 (h : a 1 + a 3 + a 5 = 3) : S 5 = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_S5_l430_43009


namespace NUMINAMATH_GPT_cuboid_volume_l430_43069

theorem cuboid_volume (a b c : ℕ) (h_incr_by_2_becomes_cube : c + 2 = a)
  (surface_area_incr : 2*a*(a + a + c + 2) - 2*a*(c + a + b) = 56) : a * b * c = 245 :=
sorry

end NUMINAMATH_GPT_cuboid_volume_l430_43069


namespace NUMINAMATH_GPT_sin_16_over_3_pi_l430_43032

theorem sin_16_over_3_pi : Real.sin (16 / 3 * Real.pi) = -Real.sqrt 3 / 2 := 
sorry

end NUMINAMATH_GPT_sin_16_over_3_pi_l430_43032


namespace NUMINAMATH_GPT_number_of_ways_to_place_rooks_l430_43038

theorem number_of_ways_to_place_rooks :
  let columns := 6
  let rows := 2006
  let rooks := 3
  ((Nat.choose columns rooks) * (rows * (rows - 1) * (rows - 2))) = 20 * 2006 * 2005 * 2004 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_ways_to_place_rooks_l430_43038


namespace NUMINAMATH_GPT_petya_vasya_sum_equality_l430_43016

theorem petya_vasya_sum_equality : ∃ (k m : ℕ), 2^(k+1) * 1023 = m * (m + 1) :=
by
  sorry

end NUMINAMATH_GPT_petya_vasya_sum_equality_l430_43016


namespace NUMINAMATH_GPT_cost_price_equal_l430_43008

theorem cost_price_equal (total_selling_price : ℝ) (profit_percent_first profit_percent_second : ℝ) (length_first_segment length_second_segment : ℝ) (C : ℝ) :
  total_selling_price = length_first_segment * (1 + profit_percent_first / 100) * C + length_second_segment * (1 + profit_percent_second / 100) * C →
  C = 15360 / (66 + 72) :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_price_equal_l430_43008


namespace NUMINAMATH_GPT_minimum_value_of_f_l430_43048

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x + 4

theorem minimum_value_of_f :
  ∃ x : ℝ, f x = -(4 / 3) :=
by
  use 2
  have hf : f 2 = -(4 / 3) := by
    sorry
  exact hf

end NUMINAMATH_GPT_minimum_value_of_f_l430_43048


namespace NUMINAMATH_GPT_sqrt_product_simplification_l430_43075

theorem sqrt_product_simplification (p : ℝ) : 
  (Real.sqrt (42 * p)) * (Real.sqrt (14 * p)) * (Real.sqrt (7 * p)) = 14 * p * (Real.sqrt (21 * p)) := 
  sorry

end NUMINAMATH_GPT_sqrt_product_simplification_l430_43075


namespace NUMINAMATH_GPT_batting_average_is_60_l430_43088

-- Definitions for conditions:
def highest_score : ℕ := 179
def difference_highest_lowest : ℕ := 150
def average_44_innings : ℕ := 58
def innings_excluding_highest_lowest : ℕ := 44
def total_innings : ℕ := 46

-- Lowest score
def lowest_score : ℕ := highest_score - difference_highest_lowest

-- Total runs in 44 innings
def total_runs_44 : ℕ := average_44_innings * innings_excluding_highest_lowest

-- Total runs in 46 innings
def total_runs_46 : ℕ := total_runs_44 + highest_score + lowest_score

-- Batting average in 46 innings
def batting_average_46 : ℕ := total_runs_46 / total_innings

-- The theorem to prove
theorem batting_average_is_60 :
  batting_average_46 = 60 :=
sorry

end NUMINAMATH_GPT_batting_average_is_60_l430_43088


namespace NUMINAMATH_GPT_remaining_black_cards_l430_43064

theorem remaining_black_cards 
  (total_cards : ℕ)
  (black_cards : ℕ)
  (red_cards : ℕ)
  (cards_taken_out : ℕ)
  (h1 : total_cards = 52)
  (h2 : black_cards = 26)
  (h3 : red_cards = 26)
  (h4 : cards_taken_out = 5) :
  black_cards - cards_taken_out = 21 := 
by {
  sorry
}

end NUMINAMATH_GPT_remaining_black_cards_l430_43064


namespace NUMINAMATH_GPT_original_number_is_1200_l430_43099

theorem original_number_is_1200 (x : ℝ) (h : 1.40 * x = 1680) : x = 1200 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_1200_l430_43099


namespace NUMINAMATH_GPT_area_on_map_correct_l430_43093

namespace FieldMap

-- Given conditions
def actual_length_m : ℕ := 200
def actual_width_m : ℕ := 100
def scale_factor : ℕ := 2000

-- Conversion from meters to centimeters
def length_cm := actual_length_m * 100
def width_cm := actual_width_m * 100

-- Dimensions on the map
def length_map_cm := length_cm / scale_factor
def width_map_cm := width_cm / scale_factor

-- Area on the map
def area_map_cm2 := length_map_cm * width_map_cm

-- Statement to prove
theorem area_on_map_correct : area_map_cm2 = 50 := by
  sorry

end FieldMap

end NUMINAMATH_GPT_area_on_map_correct_l430_43093


namespace NUMINAMATH_GPT_inequalities_always_hold_l430_43087

theorem inequalities_always_hold (x y a b : ℝ) (hxy : x > y) (hab : a > b) :
  (a + x > b + y) ∧ (x - b > y - a) :=
by
  sorry

end NUMINAMATH_GPT_inequalities_always_hold_l430_43087


namespace NUMINAMATH_GPT_dwayneA_students_l430_43022

-- Define the number of students who received an 'A' in Mrs. Carter's class
def mrsCarterA := 8
-- Define the total number of students in Mrs. Carter's class
def mrsCarterTotal := 20
-- Define the total number of students in Mr. Dwayne's class
def mrDwayneTotal := 30
-- Calculate the ratio of students who received an 'A' in Mrs. Carter's class
def carterRatio := mrsCarterA / mrsCarterTotal
-- Calculate the number of students who received an 'A' in Mr. Dwayne's class based on the same ratio
def mrDwayneA := (carterRatio * mrDwayneTotal)

-- Prove that the number of students who received an 'A' in Mr. Dwayne's class is 12
theorem dwayneA_students :
  mrDwayneA = 12 := 
by
  -- Since def calculation does not automatically prove equality, we will need to use sorry to skip the proof for now.
  sorry

end NUMINAMATH_GPT_dwayneA_students_l430_43022


namespace NUMINAMATH_GPT_min_ab_l430_43082

theorem min_ab (a b : ℝ) (h : (1 / a) + (1 / b) = Real.sqrt (a * b)) : a * b ≥ 2 := by
  sorry

end NUMINAMATH_GPT_min_ab_l430_43082
