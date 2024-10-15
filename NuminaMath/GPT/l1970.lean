import Mathlib

namespace NUMINAMATH_GPT_length_of_room_l1970_197013

theorem length_of_room (L : ℝ) 
  (h_width : 12 > 0) 
  (h_veranda_width : 2 > 0) 
  (h_area_veranda : (L + 4) * 16 - L * 12 = 140) : 
  L = 19 := 
by
  sorry

end NUMINAMATH_GPT_length_of_room_l1970_197013


namespace NUMINAMATH_GPT_fraction_value_l1970_197011

theorem fraction_value (x : ℝ) (h : x + 1/x = 3) : x^2 / (x^4 + x^2 + 1) = 1/8 :=
by sorry

end NUMINAMATH_GPT_fraction_value_l1970_197011


namespace NUMINAMATH_GPT_sphere_radius_geometric_mean_l1970_197046

-- Definitions from conditions
variable (r R ρ : ℝ)
variable (r_nonneg : 0 ≤ r)
variable (R_relation : R = 3 * r)
variable (ρ_relation : ρ = Real.sqrt 3 * r)

-- Problem statement
theorem sphere_radius_geometric_mean (tetrahedron : Prop):
  ρ * ρ = R * r :=
by
  sorry

end NUMINAMATH_GPT_sphere_radius_geometric_mean_l1970_197046


namespace NUMINAMATH_GPT_common_divisor_seven_l1970_197007

-- Definition of numbers A, B, and C based on given conditions
def A (m n : ℤ) : ℤ := n^2 + 2 * m * n + 3 * m^2 + 2
def B (m n : ℤ) : ℤ := 2 * n^2 + 3 * m * n + m^2 + 2
def C (m n : ℤ) : ℤ := 3 * n^2 + m * n + 2 * m^2 + 1

-- The proof statement ensuring A, B and C have a common divisor of 7
theorem common_divisor_seven (m n : ℤ) : ∃ d : ℤ, d > 1 ∧ d ∣ A m n ∧ d ∣ B m n ∧ d ∣ C m n → d = 7 :=
by
  sorry

end NUMINAMATH_GPT_common_divisor_seven_l1970_197007


namespace NUMINAMATH_GPT_simplification_l1970_197090

theorem simplification (a b c : ℤ) :
  (12 * a + 35 * b + 17 * c) + (13 * a - 15 * b + 8 * c) - (8 * a + 28 * b - 25 * c) = 17 * a - 8 * b + 50 * c :=
by
  sorry

end NUMINAMATH_GPT_simplification_l1970_197090


namespace NUMINAMATH_GPT_find_positive_number_l1970_197003

theorem find_positive_number (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end NUMINAMATH_GPT_find_positive_number_l1970_197003


namespace NUMINAMATH_GPT_probability_B_and_C_exactly_two_out_of_A_B_C_l1970_197093

variables (A B C : Prop)
noncomputable def P : Prop → ℚ := sorry

axiom hA : P A = 3 / 4
axiom hAC : P (¬ A ∧ ¬ C) = 1 / 12
axiom hBC : P (B ∧ C) = 1 / 4

theorem probability_B_and_C : P B = 3 / 8 ∧ P C = 2 / 3 :=
sorry

theorem exactly_two_out_of_A_B_C : 
  P (A ∧ B ∧ ¬ C) + P (A ∧ ¬ B ∧ C) + P (¬ A ∧ B ∧ C) = 15 / 32 :=
sorry

end NUMINAMATH_GPT_probability_B_and_C_exactly_two_out_of_A_B_C_l1970_197093


namespace NUMINAMATH_GPT_inequality_holds_equality_cases_l1970_197085

noncomputable def posReal : Type := { x : ℝ // 0 < x }

variables (a b c d : posReal)

theorem inequality_holds (a b c d : posReal) :
  (a.1 - b.1) * (a.1 - c.1) / (a.1 + b.1 + c.1) +
  (b.1 - c.1) * (b.1 - d.1) / (b.1 + c.1 + d.1) +
  (c.1 - d.1) * (c.1 - a.1) / (c.1 + d.1 + a.1) +
  (d.1 - a.1) * (d.1 - b.1) / (d.1 + a.1 + b.1) ≥ 0 :=
sorry

theorem equality_cases (a b c d : posReal) :
  (a.1 - b.1) * (a.1 - c.1) / (a.1 + b.1 + c.1) +
  (b.1 - c.1) * (b.1 - d.1) / (b.1 + c.1 + d.1) +
  (c.1 - d.1) * (c.1 - a.1) / (c.1 + d.1 + a.1) +
  (d.1 - a.1) * (d.1 - b.1) / (d.1 + a.1 + b.1) = 0 ↔
  (a.1 = c.1 ∧ b.1 = d.1) :=
sorry

end NUMINAMATH_GPT_inequality_holds_equality_cases_l1970_197085


namespace NUMINAMATH_GPT_num_5_digit_even_div_by_5_l1970_197064

theorem num_5_digit_even_div_by_5 : ∃! (n : ℕ), n = 500 ∧ ∀ (d : ℕ), 
  10000 ≤ d ∧ d ≤ 99999 → 
  (∀ i, i ∈ [0, 1, 2, 3, 4] → ((d / 10^i) % 10) % 2 = 0) ∧
  (d % 10 = 0) → 
  n = 500 := sorry

end NUMINAMATH_GPT_num_5_digit_even_div_by_5_l1970_197064


namespace NUMINAMATH_GPT_vasya_average_not_exceed_4_l1970_197036

variable (a b c d e : ℕ) 

-- Total number of grades
def total_grades : ℕ := a + b + c + d + e

-- Initial average condition
def initial_condition : Prop := 
  (a + 2 * b + 3 * c + 4 * d + 5 * e) < 3 * (total_grades a b c d e)

-- New average condition after grade changes
def changed_average (a b c d e : ℕ) : ℚ := 
  ((2 * b + 3 * (a + c) + 4 * d + 5 * e) : ℚ) / (total_grades a b c d e)

-- Proof problem to show the new average grade does not exceed 4
theorem vasya_average_not_exceed_4 (h : initial_condition a b c d e) : 
  (changed_average 0 b (c + a) d e) ≤ 4 := 
sorry

end NUMINAMATH_GPT_vasya_average_not_exceed_4_l1970_197036


namespace NUMINAMATH_GPT_cos_pi_div_four_minus_alpha_l1970_197094

theorem cos_pi_div_four_minus_alpha (α : ℝ) (h : Real.sin (π / 4 + α) = 2 / 3) : 
    Real.cos (π / 4 - α) = -Real.sqrt 5 / 3 :=
sorry

end NUMINAMATH_GPT_cos_pi_div_four_minus_alpha_l1970_197094


namespace NUMINAMATH_GPT_area_larger_sphere_l1970_197082

noncomputable def sphere_area_relation (A1: ℝ) (R1 R2: ℝ) := R2^2 / R1^2 * A1

-- Given Conditions
def radius_smaller_sphere : ℝ := 4.0  -- R1
def radius_larger_sphere : ℝ := 6.0    -- R2
def area_smaller_sphere : ℝ := 17.0    -- A1

-- Target Area Calculation based on Proportional Relationship
theorem area_larger_sphere :
  sphere_area_relation area_smaller_sphere radius_smaller_sphere radius_larger_sphere = 38.25 :=
by
  sorry

end NUMINAMATH_GPT_area_larger_sphere_l1970_197082


namespace NUMINAMATH_GPT_cannot_be_zero_l1970_197075

noncomputable def P (x : ℝ) (a b c d e : ℝ) := x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem cannot_be_zero (a b c d e : ℝ) (p q r s : ℝ) :
  e = 0 ∧ c = 0 ∧ (∀ x, P x a b c d e = x * (x - p) * (x - q) * (x - r) * (x - s)) ∧ 
  (p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) →
  d ≠ 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_cannot_be_zero_l1970_197075


namespace NUMINAMATH_GPT_total_servings_daily_l1970_197016

def cost_per_serving : ℕ := 14
def price_A : ℕ := 20
def price_B : ℕ := 18
def total_revenue : ℕ := 1120
def total_profit : ℕ := 280

theorem total_servings_daily (x y : ℕ) (h1 : price_A * x + price_B * y = total_revenue)
                             (h2 : (price_A - cost_per_serving) * x + (price_B - cost_per_serving) * y = total_profit) :
                             x + y = 60 := sorry

end NUMINAMATH_GPT_total_servings_daily_l1970_197016


namespace NUMINAMATH_GPT_find_years_in_future_l1970_197077

theorem find_years_in_future 
  (S F : ℕ)
  (h1 : F = 4 * S + 4)
  (h2 : F = 44) :
  ∃ x : ℕ, F + x = 2 * (S + x) + 20 ∧ x = 4 :=
by 
  sorry

end NUMINAMATH_GPT_find_years_in_future_l1970_197077


namespace NUMINAMATH_GPT_correct_subtraction_result_l1970_197041

-- Definitions based on the problem conditions
def initial_two_digit_number (X Y : ℕ) : ℕ := X * 10 + Y

-- Lean statement that expresses the proof problem
theorem correct_subtraction_result (X Y : ℕ) (H1 : initial_two_digit_number X Y = 99) (H2 : 57 = 57) :
  99 - 57 = 42 :=
by
  sorry

end NUMINAMATH_GPT_correct_subtraction_result_l1970_197041


namespace NUMINAMATH_GPT_green_pill_cost_l1970_197023

-- Given conditions
def days := 21
def total_cost := 903
def cost_difference := 2
def daily_cost := total_cost / days

-- Statement to prove
theorem green_pill_cost : (∃ (y : ℝ), y + (y - cost_difference) = daily_cost ∧ y = 22.5) :=
by
  sorry

end NUMINAMATH_GPT_green_pill_cost_l1970_197023


namespace NUMINAMATH_GPT_twelve_div_one_fourth_eq_48_l1970_197027

theorem twelve_div_one_fourth_eq_48 : 12 / (1 / 4) = 48 := by
  -- We know that dividing by a fraction is equivalent to multiplying by its reciprocal
  sorry

end NUMINAMATH_GPT_twelve_div_one_fourth_eq_48_l1970_197027


namespace NUMINAMATH_GPT_area_new_rectangle_greater_than_square_l1970_197026

theorem area_new_rectangle_greater_than_square (a b : ℝ) (h : a > b) : 
  (2 * (a + b) * (2 * b + a) / 3) > ((a + b) * (a + b)) := 
sorry

end NUMINAMATH_GPT_area_new_rectangle_greater_than_square_l1970_197026


namespace NUMINAMATH_GPT_rohan_monthly_salary_l1970_197058

theorem rohan_monthly_salary :
  ∃ S : ℝ, 
    (0.4 * S) + (0.2 * S) + (0.1 * S) + (0.1 * S) + 1000 = S :=
by
  sorry

end NUMINAMATH_GPT_rohan_monthly_salary_l1970_197058


namespace NUMINAMATH_GPT_both_sports_l1970_197029

-- Definitions based on the given conditions
def total_members := 80
def badminton_players := 48
def tennis_players := 46
def neither_players := 7

-- The theorem to be proved
theorem both_sports : (badminton_players + tennis_players - (total_members - neither_players)) = 21 := by
  sorry

end NUMINAMATH_GPT_both_sports_l1970_197029


namespace NUMINAMATH_GPT_lcm_first_ten_integers_l1970_197028

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end NUMINAMATH_GPT_lcm_first_ten_integers_l1970_197028


namespace NUMINAMATH_GPT_find_y_l1970_197089

theorem find_y (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_rem : x % y = 3) (h_div : (x:ℝ) / y = 96.15) : y = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1970_197089


namespace NUMINAMATH_GPT_sum_divisible_by_10_l1970_197078

theorem sum_divisible_by_10 :
    (111 ^ 111 + 112 ^ 112 + 113 ^ 113) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_divisible_by_10_l1970_197078


namespace NUMINAMATH_GPT_find_x_eq_l1970_197037

-- Given conditions
variables (c b θ : ℝ)

-- The proof problem
theorem find_x_eq :
  ∃ x : ℝ, x^2 + c^2 * (Real.sin θ)^2 = (b - x)^2 ∧
          x = (b^2 - c^2 * (Real.sin θ)^2) / (2 * b) :=
by
    sorry

end NUMINAMATH_GPT_find_x_eq_l1970_197037


namespace NUMINAMATH_GPT_exists_infinitely_many_n_odd_floor_l1970_197086

def even (n : ℤ) := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem exists_infinitely_many_n_odd_floor (α : ℝ) : 
  ∃ᶠ n in at_top, odd ⌊n^2 * α⌋ := sorry

end NUMINAMATH_GPT_exists_infinitely_many_n_odd_floor_l1970_197086


namespace NUMINAMATH_GPT_multiplication_equation_l1970_197049

-- Define the given conditions
def multiplier : ℕ := 6
def product : ℕ := 168
def multiplicand : ℕ := product - 140

-- Lean statement for the proof
theorem multiplication_equation : multiplier * multiplicand = product := by
  sorry

end NUMINAMATH_GPT_multiplication_equation_l1970_197049


namespace NUMINAMATH_GPT_pages_per_comic_l1970_197022

variable {comics_initial : ℕ} -- initially 5 untorn comics in the box
variable {comics_final : ℕ}   -- now there are 11 comics in the box
variable {pages_found : ℕ}    -- found 150 pages on the floor
variable {comics_assembled : ℕ} -- comics assembled from the found pages

theorem pages_per_comic (h1 : comics_initial = 5) (h2 : comics_final = 11) 
      (h3 : pages_found = 150) (h4 : comics_assembled = comics_final - comics_initial) :
      (pages_found / comics_assembled = 25) := 
sorry

end NUMINAMATH_GPT_pages_per_comic_l1970_197022


namespace NUMINAMATH_GPT_ball_hits_ground_l1970_197088

theorem ball_hits_ground :
  ∃ t : ℝ, -16 * t^2 + 20 * t + 100 = 0 ∧ t = (5 + Real.sqrt 425) / 8 :=
by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_l1970_197088


namespace NUMINAMATH_GPT_f_value_neg_five_half_one_l1970_197079

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom interval_definition : ∀ x, 0 < x ∧ x < 1 → f x = (4:ℝ) ^ x

-- The statement to prove
theorem f_value_neg_five_half_one : f (-5/2) + f 1 = -2 :=
by
  sorry

end NUMINAMATH_GPT_f_value_neg_five_half_one_l1970_197079


namespace NUMINAMATH_GPT_reflect_point_across_x_axis_l1970_197081

theorem reflect_point_across_x_axis {x y : ℝ} (h : (x, y) = (2, 3)) : (x, -y) = (2, -3) :=
by
  sorry

end NUMINAMATH_GPT_reflect_point_across_x_axis_l1970_197081


namespace NUMINAMATH_GPT_range_of_a_l1970_197025

open Real

theorem range_of_a (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : a - b + c = 3) (h₃ : a + b + c = 1) (h₄ : 0 < c ∧ c < 1) : 1 < a ∧ a < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1970_197025


namespace NUMINAMATH_GPT_min_value_of_y_l1970_197031

theorem min_value_of_y {y : ℤ} (h : ∃ x : ℤ, y^2 = (0 ^ 2 + 1 ^ 2 + 2 ^ 2 + 3 ^ 2 + 4 ^ 2 + 5 ^ 2 + (-1) ^ 2 + (-2) ^ 2 + (-3) ^ 2 + (-4) ^ 2 + (-5) ^ 2)) :
  y = -11 :=
by sorry

end NUMINAMATH_GPT_min_value_of_y_l1970_197031


namespace NUMINAMATH_GPT_evaluate_expression_l1970_197092

theorem evaluate_expression : (1023 * 1023) - (1022 * 1024) = 1 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1970_197092


namespace NUMINAMATH_GPT_even_function_k_value_l1970_197068

theorem even_function_k_value (k : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = k * x^2 + (k - 1) * x + 2)
  (even_f : ∀ x : ℝ, f x = f (-x)) : k = 1 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_even_function_k_value_l1970_197068


namespace NUMINAMATH_GPT_physics_marks_l1970_197030

theorem physics_marks (P C M : ℕ) 
  (h1 : P + C + M = 195)
  (h2 : P + M = 180)
  (h3 : P + C = 140) :
  P = 125 :=
by {
  sorry
}

end NUMINAMATH_GPT_physics_marks_l1970_197030


namespace NUMINAMATH_GPT_domain_f_1_minus_2x_is_0_to_half_l1970_197098

-- Define the domain of f(x) as a set.
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

-- Define the domain condition for f(1 - 2*x).
def domain_f_1_minus_2x (x : ℝ) : Prop := 0 ≤ 1 - 2 * x ∧ 1 - 2 * x ≤ 1

-- State the theorem: If x is in the domain of f(1 - 2*x), then x is in [0, 1/2].
theorem domain_f_1_minus_2x_is_0_to_half :
  ∀ x : ℝ, domain_f_1_minus_2x x ↔ (0 ≤ x ∧ x ≤ 1 / 2) := by
  sorry

end NUMINAMATH_GPT_domain_f_1_minus_2x_is_0_to_half_l1970_197098


namespace NUMINAMATH_GPT_floor_area_difference_l1970_197017

noncomputable def area_difference (r_outer : ℝ) (n : ℕ) (r_inner : ℝ) : ℝ :=
  let outer_area := Real.pi * r_outer^2
  let inner_area := n * Real.pi * r_inner^2
  outer_area - inner_area

theorem floor_area_difference :
  ∀ (r_outer : ℝ) (n : ℕ) (r_inner : ℝ), 
  n = 8 ∧ r_outer = 40 ∧ r_inner = 40 / (2*Real.sqrt 2 + 1) →
  ⌊area_difference r_outer n r_inner⌋ = 1150 :=
by
  intros
  sorry

end NUMINAMATH_GPT_floor_area_difference_l1970_197017


namespace NUMINAMATH_GPT_intersection_domain_range_l1970_197009

-- Define domain and function
def domain : Set ℝ := {-1, 0, 1}
def f (x : ℝ) : ℝ := |x|

-- Prove the theorem
theorem intersection_domain_range :
  let range : Set ℝ := {y | ∃ x ∈ domain, f x = y}
  let A : Set ℝ := domain
  let B : Set ℝ := range 
  A ∩ B = {0, 1} :=
by
  -- The proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_intersection_domain_range_l1970_197009


namespace NUMINAMATH_GPT_prime_iff_sum_four_distinct_products_l1970_197091

variable (n : ℕ) (a b c d : ℕ)

theorem prime_iff_sum_four_distinct_products (h : n ≥ 5) :
  (Prime n ↔ ∀ (a b c d : ℕ), n = a + b + c + d → a > 0 → b > 0 → c > 0 → d > 0 → ab ≠ cd) :=
sorry

end NUMINAMATH_GPT_prime_iff_sum_four_distinct_products_l1970_197091


namespace NUMINAMATH_GPT_suff_not_necc_condition_l1970_197047

theorem suff_not_necc_condition (x : ℝ) : (x=2) → ((x-2) * (x+5) = 0) ∧ ¬((x-2) * (x+5) = 0 → x=2) :=
by {
  sorry
}

end NUMINAMATH_GPT_suff_not_necc_condition_l1970_197047


namespace NUMINAMATH_GPT_distance_BC_l1970_197067

variable (AC AB : ℝ) (angleACB : ℝ)
  (hAC : AC = 2)
  (hAB : AB = 3)
  (hAngle : angleACB = 120)

theorem distance_BC (BC : ℝ) : BC = Real.sqrt 6 - 1 :=
by
  sorry

end NUMINAMATH_GPT_distance_BC_l1970_197067


namespace NUMINAMATH_GPT_lily_spent_on_shirt_l1970_197012

theorem lily_spent_on_shirt (S : ℝ) (initial_balance : ℝ) (final_balance : ℝ) : 
  initial_balance = 55 → 
  final_balance = 27 → 
  55 - S - 3 * S = 27 → 
  S = 7 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_lily_spent_on_shirt_l1970_197012


namespace NUMINAMATH_GPT_tan_alpha_minus_pi_six_l1970_197050

variable (α β : Real)

axiom tan_alpha_minus_beta : Real.tan (α - β) = 2 / 3
axiom tan_pi_six_minus_beta : Real.tan ((Real.pi / 6) - β) = 1 / 2

theorem tan_alpha_minus_pi_six : Real.tan (α - (Real.pi / 6)) = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_minus_pi_six_l1970_197050


namespace NUMINAMATH_GPT_sqrt_infinite_nested_problem_l1970_197095

theorem sqrt_infinite_nested_problem :
  ∃ m : ℝ, m = Real.sqrt (6 + m) ∧ m = 3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_infinite_nested_problem_l1970_197095


namespace NUMINAMATH_GPT_numberOfRottweilers_l1970_197097

-- Define the grooming times in minutes for each type of dog
def groomingTimeRottweiler := 20
def groomingTimeCollie := 10
def groomingTimeChihuahua := 45

-- Define the number of each type of dog groomed
def numberOfCollies := 9
def numberOfChihuahuas := 1

-- Define the total grooming time in minutes
def totalGroomingTime := 255

-- Compute the time spent on grooming Collies
def timeSpentOnCollies := numberOfCollies * groomingTimeCollie

-- Compute the time spent on grooming Chihuahuas
def timeSpentOnChihuahuas := numberOfChihuahuas * groomingTimeChihuahua

-- Compute the time spent on grooming Rottweilers
def timeSpentOnRottweilers := totalGroomingTime - timeSpentOnCollies - timeSpentOnChihuahuas

-- The main theorem statement
theorem numberOfRottweilers :
  timeSpentOnRottweilers / groomingTimeRottweiler = 6 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_numberOfRottweilers_l1970_197097


namespace NUMINAMATH_GPT_vector_addition_parallel_l1970_197074

def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -2)
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

theorem vector_addition_parallel:
  ∀ x : ℝ, parallel (2, 1) (x, -2) → a + b x = ((-2 : ℝ), -1) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_vector_addition_parallel_l1970_197074


namespace NUMINAMATH_GPT_new_boarders_l1970_197051

theorem new_boarders (init_boarders : ℕ) (init_day_students : ℕ) (ratio_b : ℕ) (ratio_d : ℕ) (ratio_new_b : ℕ) (ratio_new_d : ℕ) (x : ℕ) :
    init_boarders = 240 →
    ratio_b = 8 →
    ratio_d = 17 →
    ratio_new_b = 3 →
    ratio_new_d = 7 →
    init_day_students = (init_boarders * ratio_d) / ratio_b →
    (ratio_new_b * init_day_students) = ratio_new_d * (init_boarders + x) →
    x = 21 :=
by sorry

end NUMINAMATH_GPT_new_boarders_l1970_197051


namespace NUMINAMATH_GPT_females_in_orchestra_not_in_band_l1970_197084

theorem females_in_orchestra_not_in_band 
  (females_in_band : ℤ) 
  (males_in_band : ℤ) 
  (females_in_orchestra : ℤ) 
  (males_in_orchestra : ℤ) 
  (females_in_both : ℤ) 
  (total_members : ℤ) 
  (h1 : females_in_band = 120) 
  (h2 : males_in_band = 100) 
  (h3 : females_in_orchestra = 100) 
  (h4 : males_in_orchestra = 120) 
  (h5 : females_in_both = 80) 
  (h6 : total_members = 260) : 
  (females_in_orchestra - females_in_both = 20) := 
  sorry

end NUMINAMATH_GPT_females_in_orchestra_not_in_band_l1970_197084


namespace NUMINAMATH_GPT_expression_equals_5000_l1970_197063

theorem expression_equals_5000 :
  12 * 171 + 29 * 9 + 171 * 13 + 29 * 16 = 5000 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_5000_l1970_197063


namespace NUMINAMATH_GPT_quadratic_roots_distinct_l1970_197083

variable (a b c : ℤ)

theorem quadratic_roots_distinct (h_eq : 3 * a^2 - 3 * a - 4 = 0) : ∃ (x y : ℝ), x ≠ y ∧ (3 * x^2 - 3 * x - 4 = 0) ∧ (3 * y^2 - 3 * y - 4 = 0) := 
  sorry

end NUMINAMATH_GPT_quadratic_roots_distinct_l1970_197083


namespace NUMINAMATH_GPT_ball_hits_ground_in_2_72_seconds_l1970_197044

noncomputable def height_at_time (t : ℝ) : ℝ :=
  -16 * t^2 - 30 * t + 200

theorem ball_hits_ground_in_2_72_seconds :
  ∃ t : ℝ, t = 2.72 ∧ height_at_time t = 0 :=
by
  use 2.72
  sorry

end NUMINAMATH_GPT_ball_hits_ground_in_2_72_seconds_l1970_197044


namespace NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_l1970_197055

theorem ratio_of_areas_of_concentric_circles
  (Q : Type)
  (r₁ r₂ : ℝ)
  (C₁ C₂ : ℝ)
  (h₀ : r₁ > 0 ∧ r₂ > 0)
  (h₁ : C₁ = 2 * π * r₁)
  (h₂ : C₂ = 2 * π * r₂)
  (h₃ : (60 / 360) * C₁ = (30 / 360) * C₂) :
  (π * r₁^2) / (π * r₂^2) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_l1970_197055


namespace NUMINAMATH_GPT_gambler_target_win_percentage_l1970_197059

-- Define the initial conditions
def initial_games_played : ℕ := 20
def initial_win_rate : ℚ := 0.40

def additional_games_played : ℕ := 20
def additional_win_rate : ℚ := 0.80

-- Define the proof problem statement
theorem gambler_target_win_percentage 
  (initial_wins : ℚ := initial_win_rate * initial_games_played)
  (additional_wins : ℚ := additional_win_rate * additional_games_played)
  (total_games_played : ℕ := initial_games_played + additional_games_played)
  (total_wins : ℚ := initial_wins + additional_wins) :
  ((total_wins / total_games_played) * 100 : ℚ) = 60 := 
by
  -- Skipping the proof steps
  sorry

end NUMINAMATH_GPT_gambler_target_win_percentage_l1970_197059


namespace NUMINAMATH_GPT_isPossible_l1970_197034

structure Person where
  firstName : String
  patronymic : String
  surname : String

def conditions (people : List Person) : Prop :=
  people.length = 4 ∧
  ∀ p1 p2 p3 : Person, 
    p1 ∈ people → p2 ∈ people → p3 ∈ people →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    p1.firstName ≠ p2.firstName ∨ p2.firstName ≠ p3.firstName ∨ p1.firstName ≠ p3.firstName ∧
    p1.patronymic ≠ p2.patronymic ∨ p2.patronymic ≠ p3.patronymic ∨ p1.patronymic ≠ p3.patronymic ∧
    p1.surname ≠ p2.surname ∨ p2.surname ≠ p3.surname ∨ p1.surname ≠ p3.surname ∧
  ∀ p1 p2 : Person, 
    p1 ∈ people → p2 ∈ people →
    p1 ≠ p2 →
    p1.firstName = p2.firstName ∨ p1.patronymic = p2.patronymic ∨ p1.surname = p2.surname

theorem isPossible : ∃ people : List Person, conditions people := by
  sorry

end NUMINAMATH_GPT_isPossible_l1970_197034


namespace NUMINAMATH_GPT_cost_of_game_l1970_197005

theorem cost_of_game
  (number_of_ice_creams : ℕ) 
  (price_per_ice_cream : ℕ)
  (total_sold : number_of_ice_creams = 24)
  (price : price_per_ice_cream = 5) :
  (number_of_ice_creams * price_per_ice_cream) / 2 = 60 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_game_l1970_197005


namespace NUMINAMATH_GPT_doughnuts_per_person_l1970_197069

-- Define the number of dozens bought by Samuel
def samuel_dozens : ℕ := 2

-- Define the number of dozens bought by Cathy
def cathy_dozens : ℕ := 3

-- Define the number of doughnuts in one dozen
def dozen : ℕ := 12

-- Define the total number of people
def total_people : ℕ := 10

-- Prove that each person receives 6 doughnuts
theorem doughnuts_per_person :
  ((samuel_dozens * dozen) + (cathy_dozens * dozen)) / total_people = 6 :=
by
  sorry

end NUMINAMATH_GPT_doughnuts_per_person_l1970_197069


namespace NUMINAMATH_GPT_ryan_total_commuting_time_l1970_197048

def biking_time : ℕ := 30
def bus_time : ℕ := biking_time + 10
def bus_commutes : ℕ := 3
def total_bus_time : ℕ := bus_time * bus_commutes
def friend_time : ℕ := biking_time - (2 * biking_time / 3)
def total_commuting_time : ℕ := biking_time + total_bus_time + friend_time

theorem ryan_total_commuting_time :
  total_commuting_time = 160 :=
by
  sorry

end NUMINAMATH_GPT_ryan_total_commuting_time_l1970_197048


namespace NUMINAMATH_GPT_total_high_sulfur_samples_l1970_197032

-- Define the conditions as given in the problem
def total_samples : ℕ := 143
def heavy_oil_freq : ℚ := 2 / 11
def light_low_sulfur_freq : ℚ := 7 / 13
def no_low_sulfur_in_heavy_oil : Prop := ∀ (x : ℕ), (x / total_samples = heavy_oil_freq) → false

-- Define total high-sulfur samples
def num_heavy_oil := heavy_oil_freq * total_samples
def num_light_oil := total_samples - num_heavy_oil
def num_light_low_sulfur_oil := light_low_sulfur_freq * num_light_oil
def num_light_high_sulfur_oil := num_light_oil - num_light_low_sulfur_oil

-- Now state that we need to prove the total number of high-sulfur samples
theorem total_high_sulfur_samples : num_light_high_sulfur_oil + num_heavy_oil = 80 :=
by
  sorry

end NUMINAMATH_GPT_total_high_sulfur_samples_l1970_197032


namespace NUMINAMATH_GPT_statement_C_is_incorrect_l1970_197040

noncomputable def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem statement_C_is_incorrect : g (-2) ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_statement_C_is_incorrect_l1970_197040


namespace NUMINAMATH_GPT_probability_of_triangle_with_nonagon_side_l1970_197099

-- Definitions based on the given conditions
def num_vertices : ℕ := 9

def total_triangles : ℕ := Nat.choose num_vertices 3

def favorable_outcomes : ℕ :=
  let one_side_is_side_of_nonagon := num_vertices * 5
  let two_sides_are_sides_of_nonagon := num_vertices
  one_side_is_side_of_nonagon + two_sides_are_sides_of_nonagon

def probability : ℚ := favorable_outcomes / total_triangles

-- Lean 4 statement to prove the equivalence of the probability calculation
theorem probability_of_triangle_with_nonagon_side :
  probability = 9 / 14 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_triangle_with_nonagon_side_l1970_197099


namespace NUMINAMATH_GPT_area_of_pentagon_eq_fraction_l1970_197053

theorem area_of_pentagon_eq_fraction (w : ℝ) (h : ℝ) (fold_x : ℝ) (fold_y : ℝ)
    (hw3 : h = 3 * w)
    (hfold : fold_x = fold_y)
    (hx : fold_x ^ 2 + fold_y ^ 2 = 3 ^ 2)
    (hx_dist : fold_x = 4 / 3) :
  (3 * (1 / 2) + fold_x / 2) / (3 * w) = 13 / 18 := 
by 
  sorry

end NUMINAMATH_GPT_area_of_pentagon_eq_fraction_l1970_197053


namespace NUMINAMATH_GPT_james_planted_60_percent_l1970_197066

theorem james_planted_60_percent :
  let total_trees := 2
  let plants_per_tree := 20
  let seeds_per_plant := 1
  let total_seeds := total_trees * plants_per_tree * seeds_per_plant
  let planted_trees := 24
  (planted_trees / total_seeds) * 100 = 60 := 
by
  sorry

end NUMINAMATH_GPT_james_planted_60_percent_l1970_197066


namespace NUMINAMATH_GPT_exponential_equivalence_l1970_197000

theorem exponential_equivalence (a : ℝ) : a^4 * a^4 = (a^2)^4 := by
  sorry

end NUMINAMATH_GPT_exponential_equivalence_l1970_197000


namespace NUMINAMATH_GPT_interest_for_1_rs_l1970_197073

theorem interest_for_1_rs (I₅₀₀₀ : ℝ) (P : ℝ) (h : I₅₀₀₀ = 200) (hP : P = 5000) : I₅₀₀₀ / P = 0.04 :=
by
  rw [h, hP]
  norm_num

end NUMINAMATH_GPT_interest_for_1_rs_l1970_197073


namespace NUMINAMATH_GPT_total_repairs_cost_eq_l1970_197076

-- Assume the initial cost of the scooter is represented by a real number C.
variable (C : ℝ)

-- Given conditions
def spent_on_first_repair := 0.05 * C
def spent_on_second_repair := 0.10 * C
def spent_on_third_repair := 0.07 * C

-- Total repairs expenditure
def total_repairs := spent_on_first_repair C + spent_on_second_repair C + spent_on_third_repair C

-- Selling price and profit
def selling_price := 1.25 * C
def profit := 1500
def profit_calc := selling_price C - (C + total_repairs C)

-- Statement to be proved: The total repairs is equal to $11,000.
theorem total_repairs_cost_eq : total_repairs 50000 = 11000 := by
  sorry

end NUMINAMATH_GPT_total_repairs_cost_eq_l1970_197076


namespace NUMINAMATH_GPT_alex_needs_additional_coins_l1970_197019

theorem alex_needs_additional_coins : 
  let friends := 12
  let coins := 63
  let total_coins_needed := (friends * (friends + 1)) / 2 
  let additional_coins_needed := total_coins_needed - coins
  additional_coins_needed = 15 :=
by sorry

end NUMINAMATH_GPT_alex_needs_additional_coins_l1970_197019


namespace NUMINAMATH_GPT_average_percent_score_l1970_197001

theorem average_percent_score (num_students : ℕ)
    (students_95 students_85 students_75 students_65 students_55 students_45 : ℕ)
    (h : students_95 + students_85 + students_75 + students_65 + students_55 + students_45 = 120) :
  ((95 * students_95 + 85 * students_85 + 75 * students_75 + 65 * students_65 + 55 * students_55 + 45 * students_45) / 120 : ℚ) = 72.08 := 
by {
  sorry
}

end NUMINAMATH_GPT_average_percent_score_l1970_197001


namespace NUMINAMATH_GPT_evaluate_f_5_minus_f_neg_5_l1970_197057

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x + 3

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := 
  by
    sorry

end NUMINAMATH_GPT_evaluate_f_5_minus_f_neg_5_l1970_197057


namespace NUMINAMATH_GPT_book_page_count_l1970_197018

theorem book_page_count:
  (∃ (total_pages : ℕ), 
    (∃ (days_read : ℕ) (pages_per_day : ℕ), 
      days_read = 12 ∧ 
      pages_per_day = 8 ∧ 
      (days_read * pages_per_day) = 2 * (total_pages / 3)) 
  ↔ total_pages = 144) :=
by 
  sorry

end NUMINAMATH_GPT_book_page_count_l1970_197018


namespace NUMINAMATH_GPT_gray_region_area_l1970_197052

theorem gray_region_area (r : ℝ) (h1 : r > 0) (h2 : 3 * r - r = 3) : 
  (π * (3 * r) * (3 * r) - π * r * r) = 18 * π := by
  sorry

end NUMINAMATH_GPT_gray_region_area_l1970_197052


namespace NUMINAMATH_GPT_students_who_like_both_channels_l1970_197070

theorem students_who_like_both_channels (total_students : ℕ) 
    (sports_channel : ℕ) (arts_channel : ℕ) (neither_channel : ℕ)
    (h_total : total_students = 100) (h_sports : sports_channel = 68) 
    (h_arts : arts_channel = 55) (h_neither : neither_channel = 3) :
    ∃ x, (x = 26) :=
by
  have h_at_least_one := total_students - neither_channel
  have h_A_union_B := sports_channel + arts_channel - h_at_least_one
  use h_A_union_B
  sorry

end NUMINAMATH_GPT_students_who_like_both_channels_l1970_197070


namespace NUMINAMATH_GPT_total_points_correct_l1970_197062

-- Define the scores
def Marius (Darius : ℕ) : ℕ := Darius + 3
def Matt (Darius : ℕ) : ℕ := Darius + 5

-- Define the total points function
def total_points (Darius : ℕ) : ℕ :=
  Darius + Marius Darius + Matt Darius

-- Specific value for Darius's score
def Darius_score : ℕ := 10

-- The theorem that proves the total score is 38 given Darius's score
theorem total_points_correct :
  total_points Darius_score = 38 :=
by
  sorry

end NUMINAMATH_GPT_total_points_correct_l1970_197062


namespace NUMINAMATH_GPT_rhombus_area_l1970_197056

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : 
  (1 / 2) * d1 * d2 = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_rhombus_area_l1970_197056


namespace NUMINAMATH_GPT_arithmetic_calculation_l1970_197072

theorem arithmetic_calculation : 3 - (-5) + 7 = 15 := by
  sorry

end NUMINAMATH_GPT_arithmetic_calculation_l1970_197072


namespace NUMINAMATH_GPT_product_of_roots_in_range_l1970_197096

noncomputable def f (x : ℝ) : ℝ := abs (abs (x - 1) - 1)

theorem product_of_roots_in_range (m : ℝ) (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∃ x1 x2 x3 x4 : ℝ, 
        f x1 = m ∧ 
        f x2 = m ∧ 
        f x3 = m ∧ 
        f x4 = m ∧ 
        x1 ≠ x2 ∧ 
        x1 ≠ x3 ∧ 
        x1 ≠ x4 ∧ 
        x2 ≠ x3 ∧ 
        x2 ≠ x4 ∧ 
        x3 ≠ x4) :
  ∃ p : ℝ, p = (m * (2 - m) * (m + 2) * (-m)) ∧ -3 < p ∧ p < 0 :=
sorry

end NUMINAMATH_GPT_product_of_roots_in_range_l1970_197096


namespace NUMINAMATH_GPT_find_angle_at_A_l1970_197008

def triangle_angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def ab_lt_bc_lt_ac (AB BC AC : ℝ) : Prop :=
  AB < BC ∧ BC < AC

def angles_relation (α β γ : ℝ) : Prop :=
  (α = 2 * γ) ∧ (β = 3 * γ)

theorem find_angle_at_A
  (AB BC AC : ℝ)
  (α β γ : ℝ)
  (h1 : ab_lt_bc_lt_ac AB BC AC)
  (h2 : angles_relation α β γ)
  (h3 : triangle_angles_sum_to_180 α β γ) :
  α = 60 :=
sorry

end NUMINAMATH_GPT_find_angle_at_A_l1970_197008


namespace NUMINAMATH_GPT_exists_y_with_7_coprimes_less_than_20_l1970_197038

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1
def connection (a b : ℕ) : ℚ := Nat.lcm a b / (a * b)

theorem exists_y_with_7_coprimes_less_than_20 :
  ∃ y : ℕ, y < 20 ∧ (∃ x : ℕ, connection y x = 1) ∧ (Nat.totient y = 7) :=
by
  sorry

end NUMINAMATH_GPT_exists_y_with_7_coprimes_less_than_20_l1970_197038


namespace NUMINAMATH_GPT_find_divisor_l1970_197087

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) 
  (h1 : dividend = 62976) 
  (h2 : quotient = 123) 
  (h3 : dividend = divisor * quotient) 
  : divisor = 512 := 
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1970_197087


namespace NUMINAMATH_GPT_maximum_value_l1970_197060

noncomputable def p : ℝ := 1 + 1/2 + 1/2^2 + 1/2^3 + 1/2^4 + 1/2^5

theorem maximum_value (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h_constraint : (x - 1)^2 + (y - 1)^2 + (z - 1)^2 = 27) : 
  x^p + y^p + z^p ≤ 40.4 :=
sorry

end NUMINAMATH_GPT_maximum_value_l1970_197060


namespace NUMINAMATH_GPT_vector_projection_line_l1970_197071

theorem vector_projection_line (v : ℝ × ℝ) 
  (h : ∃ (x y : ℝ), v = (x, y) ∧ 
       (3 * x + 4 * y) / (3 ^ 2 + 4 ^ 2) = 1) :
  ∃ (x y : ℝ), v = (x, y) ∧ y = -3 / 4 * x + 25 / 4 :=
by
  sorry

end NUMINAMATH_GPT_vector_projection_line_l1970_197071


namespace NUMINAMATH_GPT_math_problem_solution_l1970_197043

noncomputable def math_problem (a b c d : ℝ) (h1 : a^2 + b^2 - c^2 - d^2 = 0) (h2 : a^2 - b^2 - c^2 + d^2 = (56 / 53) * (b * c + a * d)) : ℝ :=
  (a * b + c * d) / (b * c + a * d)

theorem math_problem_solution (a b c d : ℝ) (h1 : a^2 + b^2 - c^2 - d^2 = 0) (h2 : a^2 - b^2 - c^2 + d^2 = (56 / 53) * (b * c + a * d)) :
  math_problem a b c d h1 h2 = 45 / 53 := sorry

end NUMINAMATH_GPT_math_problem_solution_l1970_197043


namespace NUMINAMATH_GPT_julia_stairs_less_than_third_l1970_197042

theorem julia_stairs_less_than_third (J1 : ℕ) (T : ℕ) (T_total : ℕ) (J : ℕ) 
  (hJ1 : J1 = 1269) (hT : T = 1269 / 3) (hT_total : T_total = 1685) (hTotal : J1 + J = T_total) : 
  T - J = 7 := 
by
  sorry

end NUMINAMATH_GPT_julia_stairs_less_than_third_l1970_197042


namespace NUMINAMATH_GPT_y_coordinate_equidistant_l1970_197006

theorem y_coordinate_equidistant : ∃ y : ℝ, (∀ A B : ℝ × ℝ, A = (-3, 0) → B = (-2, 5) → dist (0, y) A = dist (0, y) B) ∧ y = 2 :=
by
  sorry

end NUMINAMATH_GPT_y_coordinate_equidistant_l1970_197006


namespace NUMINAMATH_GPT_area_of_pentagon_correct_l1970_197021

noncomputable def area_of_pentagon : ℝ :=
  let AB := 5
  let BC := 3
  let BD := 3
  let AC := Real.sqrt (AB^2 - BC^2)
  let AD := Real.sqrt (AB^2 - BD^2)
  let EC := 1
  let FD := 2
  let AE := AC - EC
  let AF := AD - FD
  let sin_alpha := BC / AB
  let cos_alpha := AC / AB
  let sin_2alpha := 2 * sin_alpha * cos_alpha
  let area_ABC := 0.5 * AB * BC
  let area_AEF := 0.5 * AE * AF * sin_2alpha
  2 * area_ABC - area_AEF

theorem area_of_pentagon_correct :
  area_of_pentagon = 9.12 := sorry

end NUMINAMATH_GPT_area_of_pentagon_correct_l1970_197021


namespace NUMINAMATH_GPT_simplify_sqrt_product_l1970_197014

theorem simplify_sqrt_product (x : ℝ) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (28 * x) * Real.sqrt (5 * x) =
  60 * x^2 * Real.sqrt 35 :=
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_product_l1970_197014


namespace NUMINAMATH_GPT_mark_more_hours_than_kate_l1970_197004

-- Definitions for the problem
variable (K : ℕ)  -- K is the number of hours charged by Kate
variable (P : ℕ)  -- P is the number of hours charged by Pat
variable (M : ℕ)  -- M is the number of hours charged by Mark

-- Conditions
def total_hours := K + P + M = 216
def pat_kate_relation := P = 2 * K
def pat_mark_relation := P = (1 / 3) * M

-- The statement to be proved
theorem mark_more_hours_than_kate (K P M : ℕ) (h1 : total_hours K P M)
  (h2 : pat_kate_relation K P) (h3 : pat_mark_relation P M) :
  (M - K = 120) :=
by
  sorry

end NUMINAMATH_GPT_mark_more_hours_than_kate_l1970_197004


namespace NUMINAMATH_GPT_sum_binomials_eq_l1970_197002

theorem sum_binomials_eq : 
  (Nat.choose 6 1) + (Nat.choose 6 2) + (Nat.choose 6 3) + (Nat.choose 6 4) + (Nat.choose 6 5) = 62 :=
by
  sorry

end NUMINAMATH_GPT_sum_binomials_eq_l1970_197002


namespace NUMINAMATH_GPT_student_marks_l1970_197065

variable (M P C : ℕ)

theorem student_marks (h1 : C = P + 20) (h2 : (M + C) / 2 = 20) : M + P = 20 :=
by
  sorry

end NUMINAMATH_GPT_student_marks_l1970_197065


namespace NUMINAMATH_GPT_quadratic_function_increases_l1970_197035

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := 2 * x ^ 2 - 4 * x + 5

-- Prove that for x > 1, the function value y increases as x increases
theorem quadratic_function_increases (x : ℝ) (h : x > 1) : 
  quadratic_function x > quadratic_function 1 :=
sorry

end NUMINAMATH_GPT_quadratic_function_increases_l1970_197035


namespace NUMINAMATH_GPT_largest_integer_le_zero_of_f_l1970_197080

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem largest_integer_le_zero_of_f :
  ∃ x₀ : ℝ, (f x₀ = 0) ∧ 2 ≤ x₀ ∧ x₀ < 3 ∧ (∀ k : ℤ, k ≤ x₀ → k = 2 ∨ k < 2) :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_le_zero_of_f_l1970_197080


namespace NUMINAMATH_GPT_train_length_l1970_197020

theorem train_length
    (V : ℝ) -- train speed in m/s
    (L : ℝ) -- length of the train in meters
    (H1 : L = V * 18) -- condition: train crosses signal pole in 18 sec
    (H2 : L + 333.33 = V * 38) -- condition: train crosses platform in 38 sec
    (V_pos : 0 < V) -- additional condition: speed must be positive
    : L = 300 :=
by
-- here goes the proof which is not required for our task
sorry

end NUMINAMATH_GPT_train_length_l1970_197020


namespace NUMINAMATH_GPT_swans_in_10_years_l1970_197033

def doubling_time := 2
def initial_swans := 15
def periods := 10 / doubling_time

theorem swans_in_10_years : 
  (initial_swans * 2 ^ periods) = 480 := 
by
  sorry

end NUMINAMATH_GPT_swans_in_10_years_l1970_197033


namespace NUMINAMATH_GPT_find_a_of_odd_function_l1970_197054

noncomputable def f (a : ℝ) (x : ℝ) := 1 + a / (2^x + 1)

theorem find_a_of_odd_function (a : ℝ) (h : ∀ x : ℝ, f a x = -f a (-x)) : a = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_of_odd_function_l1970_197054


namespace NUMINAMATH_GPT_sum_first_95_odds_equals_9025_l1970_197024

-- Define the nth odd positive integer
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

-- Define the sum of the first n odd positive integers
def sum_first_n_odds (n : ℕ) : ℕ := n^2

-- State the theorem to be proved
theorem sum_first_95_odds_equals_9025 : sum_first_n_odds 95 = 9025 :=
by
  -- We provide a placeholder for the proof
  sorry

end NUMINAMATH_GPT_sum_first_95_odds_equals_9025_l1970_197024


namespace NUMINAMATH_GPT_inclination_angle_range_l1970_197061

theorem inclination_angle_range (k : ℝ) (α : ℝ) (h1 : -1 ≤ k) (h2 : k < 1)
  (h3 : k = Real.tan α) (h4 : 0 ≤ α) (h5 : α < 180) :
  (0 ≤ α ∧ α < 45) ∨ (135 ≤ α ∧ α < 180) :=
sorry

end NUMINAMATH_GPT_inclination_angle_range_l1970_197061


namespace NUMINAMATH_GPT_squirrel_rise_per_circuit_l1970_197045

noncomputable def rise_per_circuit
    (height : ℕ)
    (circumference : ℕ)
    (distance : ℕ) :=
    height / (distance / circumference)

theorem squirrel_rise_per_circuit : rise_per_circuit 25 3 15 = 5 :=
by
  sorry

end NUMINAMATH_GPT_squirrel_rise_per_circuit_l1970_197045


namespace NUMINAMATH_GPT_triangle_is_isosceles_l1970_197039

theorem triangle_is_isosceles 
  (a b c : ℝ)
  (h : a^2 - b^2 + a * c - b * c = 0)
  (h_tri : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  : a = b := 
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l1970_197039


namespace NUMINAMATH_GPT_millet_percentage_in_mix_l1970_197015

def contribution_millet_brandA (percA mixA : ℝ) := percA * mixA
def contribution_millet_brandB (percB mixB : ℝ) := percB * mixB

theorem millet_percentage_in_mix
  (percA : ℝ) (percB : ℝ) (mixA : ℝ) (mixB : ℝ)
  (h1 : percA = 0.40) (h2 : percB = 0.65) (h3 : mixA = 0.60) (h4 : mixB = 0.40) :
  (contribution_millet_brandA percA mixA + contribution_millet_brandB percB mixB = 0.50) :=
by
  sorry

end NUMINAMATH_GPT_millet_percentage_in_mix_l1970_197015


namespace NUMINAMATH_GPT_percentage_proof_l1970_197010

/-- Lean 4 statement proving the percentage -/
theorem percentage_proof :
  ∃ P : ℝ, (800 - (P / 100) * 8000) = 796 ∧ P = 0.05 :=
by
  use 0.05
  sorry

end NUMINAMATH_GPT_percentage_proof_l1970_197010
