import Mathlib

namespace NUMINAMATH_GPT_negation_of_proposition_l490_49031

theorem negation_of_proposition (m : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ (∃ x : ℝ, x^2 + 2*x + m > 0) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l490_49031


namespace NUMINAMATH_GPT_complex_power_sum_eq_five_l490_49050

noncomputable def w : ℂ := sorry

theorem complex_power_sum_eq_five (h : w^3 + w^2 + 1 = 0) : 
  w^100 + w^101 + w^102 + w^103 + w^104 = 5 :=
sorry

end NUMINAMATH_GPT_complex_power_sum_eq_five_l490_49050


namespace NUMINAMATH_GPT_cubic_roots_reciprocal_sum_l490_49033

theorem cubic_roots_reciprocal_sum {α β γ : ℝ} 
  (h₁ : α + β + γ = 6)
  (h₂ : α * β + β * γ + γ * α = 11)
  (h₃ : α * β * γ = 6) :
  (1 / α^2) + (1 / β^2) + (1 / γ^2) = 49 / 36 := 
by 
  sorry

end NUMINAMATH_GPT_cubic_roots_reciprocal_sum_l490_49033


namespace NUMINAMATH_GPT_discriminant_divisible_l490_49037

theorem discriminant_divisible (a b: ℝ) (n: ℤ) (h: (∃ x1 x2: ℝ, 2018*x1^2 + a*x1 + b = 0 ∧ 2018*x2^2 + a*x2 + b = 0 ∧ x1 - x2 = n)): 
  ∃ k: ℤ, a^2 - 4 * 2018 * b = 2018^2 * k := 
by 
  sorry

end NUMINAMATH_GPT_discriminant_divisible_l490_49037


namespace NUMINAMATH_GPT_april_plant_arrangement_l490_49021

theorem april_plant_arrangement :
  let basil_plants := 5
  let tomato_plants := 4
  let total_units := (basil_plants - 2) + 1 + 1
  (Nat.factorial total_units) * (Nat.factorial tomato_plants) * (Nat.factorial 2) = 5760 :=
by
  sorry

end NUMINAMATH_GPT_april_plant_arrangement_l490_49021


namespace NUMINAMATH_GPT_simplify_fraction_expression_l490_49060

theorem simplify_fraction_expression (d : ℤ) :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 :=
by
  -- skip the proof by adding sorry
  sorry

end NUMINAMATH_GPT_simplify_fraction_expression_l490_49060


namespace NUMINAMATH_GPT_xy_value_l490_49028

theorem xy_value {x y : ℝ} (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 21 :=
by
  sorry

end NUMINAMATH_GPT_xy_value_l490_49028


namespace NUMINAMATH_GPT_quadratic_inequality_for_all_x_l490_49020

theorem quadratic_inequality_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a^2 + a) * x^2 - a * x + 1 > 0) ↔ (-4 / 3 < a ∧ a < -1) ∨ a = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_for_all_x_l490_49020


namespace NUMINAMATH_GPT_problem_1_solution_set_problem_2_min_value_l490_49040

-- Problem (1)
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_1_solution_set :
  {x : ℝ | f (x + 3/2) ≥ 0} = {x | -2 ≤ x ∧ x ≤ 2} :=
by
  sorry

-- Problem (2)
theorem problem_2_min_value (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r ≥ 9/4 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_solution_set_problem_2_min_value_l490_49040


namespace NUMINAMATH_GPT_solution_set_of_inequality_l490_49079

theorem solution_set_of_inequality :
  { x : ℝ | x ≠ 5 ∧ (x * (x + 1)) / ((x - 5) ^ 3) ≥ 25 } = 
  { x : ℝ | x ≤ 5 / 3 } ∪ { x : ℝ | x > 5 } := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l490_49079


namespace NUMINAMATH_GPT_triangle_min_diff_l490_49008

variable (XY YZ XZ : ℕ) -- Declaring the side lengths as natural numbers

theorem triangle_min_diff (h1 : XY < YZ ∧ YZ ≤ XZ) -- Condition for side length relations
  (h2 : XY + YZ + XZ = 2010) -- Condition for the perimeter
  (h3 : XY + YZ > XZ)
  (h4 : XY + XZ > YZ)
  (h5 : YZ + XZ > XY) :
  (YZ - XY) = 1 := -- Statement that the smallest possible value of YZ - XY is 1
sorry

end NUMINAMATH_GPT_triangle_min_diff_l490_49008


namespace NUMINAMATH_GPT_franks_age_l490_49023

variable (F G : ℕ)

def gabriel_younger_than_frank : Prop := G = F - 3
def total_age_is_seventeen : Prop := F + G = 17

theorem franks_age (h1 : gabriel_younger_than_frank F G) (h2 : total_age_is_seventeen F G) : F = 10 :=
by
  sorry

end NUMINAMATH_GPT_franks_age_l490_49023


namespace NUMINAMATH_GPT_roots_geometric_progression_two_complex_conjugates_l490_49092

theorem roots_geometric_progression_two_complex_conjugates (a : ℝ) :
  (∃ b k : ℝ, b ≠ 0 ∧ k ≠ 0 ∧ (k + 1/ k = 2) ∧ 
    (b * (1 + k + 1/k) = 9) ∧ (b^2 * (k + 1 + 1/k) = 27) ∧ (b^3 = -a)) →
  a = -27 :=
by sorry

end NUMINAMATH_GPT_roots_geometric_progression_two_complex_conjugates_l490_49092


namespace NUMINAMATH_GPT_percentage_error_l490_49087

theorem percentage_error (x : ℝ) : ((x * 3 - x / 5) / (x * 3) * 100) = 93.33 := 
  sorry

end NUMINAMATH_GPT_percentage_error_l490_49087


namespace NUMINAMATH_GPT_find_x_when_y_is_6_l490_49069

-- Condition for inverse variation
def inverse_var (k y : ℝ) (x : ℝ) : Prop := x = k / y^2

-- Given values
def given_value_x : ℝ := 1
def given_value_y : ℝ := 2
def new_value_y : ℝ := 6

-- The theorem to prove
theorem find_x_when_y_is_6 :
  ∃ k, inverse_var k given_value_y given_value_x → inverse_var k new_value_y (1/9) :=
by
  sorry

end NUMINAMATH_GPT_find_x_when_y_is_6_l490_49069


namespace NUMINAMATH_GPT_madeline_rent_l490_49041

noncomputable def groceries : ℝ := 400
noncomputable def medical_expenses : ℝ := 200
noncomputable def utilities : ℝ := 60
noncomputable def emergency_savings : ℝ := 200
noncomputable def hourly_wage : ℝ := 15
noncomputable def hours_worked : ℕ := 138
noncomputable def total_expenses_and_savings : ℝ := groceries + medical_expenses + utilities + emergency_savings
noncomputable def total_earnings : ℝ := hourly_wage * hours_worked

theorem madeline_rent : total_earnings - total_expenses_and_savings = 1210 := by
  sorry

end NUMINAMATH_GPT_madeline_rent_l490_49041


namespace NUMINAMATH_GPT_fourth_equation_general_expression_l490_49064

theorem fourth_equation :
  (10 : ℕ)^2 - 4 * (4 : ℕ)^2 = 36 := 
sorry

theorem general_expression (n : ℕ) (hn : n > 0) :
  (2 * n + 2)^2 - 4 * n^2 = 8 * n + 4 :=
sorry

end NUMINAMATH_GPT_fourth_equation_general_expression_l490_49064


namespace NUMINAMATH_GPT_initial_men_l490_49048

/-- Initial number of men M being catered for. 
Proof that the initial number of men M is equal to 760 given the conditions. -/
theorem initial_men (M : ℕ)
  (H1 : 22 * M = 20 * M)
  (H2 : 2 * (M + 3040) = M) : M = 760 := 
sorry

end NUMINAMATH_GPT_initial_men_l490_49048


namespace NUMINAMATH_GPT_condition_s_for_q_condition_r_for_q_condition_p_for_s_l490_49043

variables {p q r s : Prop}

-- Given conditions from a)
axiom h₁ : r → p
axiom h₂ : q → r
axiom h₃ : s → r
axiom h₄ : q → s

-- The corresponding proof problems based on c)
theorem condition_s_for_q : (s ↔ q) :=
by sorry

theorem condition_r_for_q : (r ↔ q) :=
by sorry

theorem condition_p_for_s : (s → p) :=
by sorry

end NUMINAMATH_GPT_condition_s_for_q_condition_r_for_q_condition_p_for_s_l490_49043


namespace NUMINAMATH_GPT_task_assignments_count_l490_49084

theorem task_assignments_count (S : Finset (Fin 5)) :
  ∃ (assignments : Fin 5 → Fin 3),  
    (∀ t, assignments t ≠ t) ∧ 
    (∀ v, ∃ t, assignments t = v) ∧ 
    (∀ t, (t = 4 → assignments t = 1)) ∧ 
    S.card = 60 :=
by sorry

end NUMINAMATH_GPT_task_assignments_count_l490_49084


namespace NUMINAMATH_GPT_total_number_of_letters_l490_49006

def jonathan_first_name_letters : Nat := 8
def jonathan_surname_letters : Nat := 10
def sister_first_name_letters : Nat := 5
def sister_surname_letters : Nat := 10

theorem total_number_of_letters : 
  jonathan_first_name_letters + jonathan_surname_letters + sister_first_name_letters + sister_surname_letters = 33 := 
by 
  sorry

end NUMINAMATH_GPT_total_number_of_letters_l490_49006


namespace NUMINAMATH_GPT_find_number_l490_49093

theorem find_number (x : ℝ) (h : 3034 - (1002 / x) = 2984) : x = 20.04 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l490_49093


namespace NUMINAMATH_GPT_binom_divisible_by_prime_l490_49059

theorem binom_divisible_by_prime {p k : ℕ} (hp : Nat.Prime p) (h1 : 1 ≤ k) (h2 : k ≤ p - 1) : p ∣ Nat.choose p k :=
sorry

end NUMINAMATH_GPT_binom_divisible_by_prime_l490_49059


namespace NUMINAMATH_GPT_candies_per_block_l490_49027

theorem candies_per_block (candies_per_house : ℕ) (houses_per_block : ℕ) (h1 : candies_per_house = 7) (h2 : houses_per_block = 5) :
  candies_per_house * houses_per_block = 35 :=
by 
  -- Placeholder for the formal proof
  sorry

end NUMINAMATH_GPT_candies_per_block_l490_49027


namespace NUMINAMATH_GPT_sum_of_selected_sections_l490_49080

-- Given volumes of a bamboo, we denote them as a1, a2, ..., a9 forming an arithmetic sequence.
-- Where the sum of the volumes of the top four sections is 3 liters, and the
-- sum of the volumes of the bottom three sections is 4 liters.

-- Definitions based on the conditions
def arith_seq (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

variables {a : ℕ → ℝ} {d : ℝ}
variable (sum_top_four : a 1 + a 2 + a 3 + a 4 = 3)
variable (sum_bottom_three : a 7 + a 8 + a 9 = 4)
variable (seq_condition : arith_seq a d)

theorem sum_of_selected_sections 
  (h1 : a 1 + a 2 + a 3 + a 4 = 3)
  (h2 : a 7 + a 8 + a 9 = 4)
  (h_seq : arith_seq a d) : 
  a 2 + a 3 + a 8 = 17 / 6 := 
sorry -- proof goes here

end NUMINAMATH_GPT_sum_of_selected_sections_l490_49080


namespace NUMINAMATH_GPT_max_clouds_through_planes_l490_49081

-- Define the problem parameters and conditions
def max_clouds (n : ℕ) : ℕ :=
  n + 1

-- Mathematically equivalent proof problem statement in Lean 4
theorem max_clouds_through_planes : max_clouds 10 = 11 :=
  by
    sorry  -- Proof skipped as required

end NUMINAMATH_GPT_max_clouds_through_planes_l490_49081


namespace NUMINAMATH_GPT_trig_simplification_l490_49076

theorem trig_simplification :
  (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180)) = 4 :=
sorry

end NUMINAMATH_GPT_trig_simplification_l490_49076


namespace NUMINAMATH_GPT_first_group_men_l490_49009

theorem first_group_men (M : ℕ) (h : M * 15 = 25 * 24) : M = 40 := sorry

end NUMINAMATH_GPT_first_group_men_l490_49009


namespace NUMINAMATH_GPT_kate_change_l490_49082

def first_candy_cost : ℝ := 0.54
def second_candy_cost : ℝ := 0.35
def third_candy_cost : ℝ := 0.68
def amount_given : ℝ := 5.00

theorem kate_change : amount_given - (first_candy_cost + second_candy_cost + third_candy_cost) = 3.43 := by
  sorry

end NUMINAMATH_GPT_kate_change_l490_49082


namespace NUMINAMATH_GPT_original_six_digit_number_is_285714_l490_49024

theorem original_six_digit_number_is_285714 
  (N : ℕ) 
  (h1 : ∃ x, N = 200000 + x ∧ 10 * x + 2 = 3 * (200000 + x)) :
  N = 285714 := 
sorry

end NUMINAMATH_GPT_original_six_digit_number_is_285714_l490_49024


namespace NUMINAMATH_GPT_calculate_triple_transform_l490_49089

def transformation (N : ℝ) : ℝ :=
  0.4 * N + 2

theorem calculate_triple_transform :
  transformation (transformation (transformation 20)) = 4.4 :=
by
  sorry

end NUMINAMATH_GPT_calculate_triple_transform_l490_49089


namespace NUMINAMATH_GPT_equalize_foma_ierema_l490_49001

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end NUMINAMATH_GPT_equalize_foma_ierema_l490_49001


namespace NUMINAMATH_GPT_divisible_by_8_l490_49085

theorem divisible_by_8 (k : ℤ) : 
  let m := 2 * k + 1 
  let n := 2 * k + 3 
  8 ∣ (7 * m^2 - 5 * n^2 - 2) :=
by 
  let m := 2 * k + 1 
  let n := 2 * k + 3 
  sorry

end NUMINAMATH_GPT_divisible_by_8_l490_49085


namespace NUMINAMATH_GPT_program_output_is_201_l490_49073

theorem program_output_is_201 :
  ∃ x S n, x = 3 + 2 * n ∧ S = n^2 + 4 * n ∧ S ≥ 10000 ∧ x = 201 :=
by
  sorry

end NUMINAMATH_GPT_program_output_is_201_l490_49073


namespace NUMINAMATH_GPT_total_bending_angle_l490_49075

theorem total_bending_angle (n : ℕ) (h : n > 4) (θ : ℝ) (hθ : θ = 360 / (2 * n)) : 
  ∃ α : ℝ, α = 180 :=
by
  sorry

end NUMINAMATH_GPT_total_bending_angle_l490_49075


namespace NUMINAMATH_GPT_rounding_increases_value_l490_49029

theorem rounding_increases_value (a b c d : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (rounded_a : ℕ := a + 1)
  (rounded_b : ℕ := b - 1)
  (rounded_c : ℕ := c + 1)
  (rounded_d : ℕ := d + 1) :
  (rounded_a * rounded_d) / rounded_b + rounded_c > (a * d) / b + c := 
sorry

end NUMINAMATH_GPT_rounding_increases_value_l490_49029


namespace NUMINAMATH_GPT_interest_first_year_correct_interest_second_year_correct_interest_third_year_correct_l490_49017

noncomputable def principal_first_year : ℝ := 9000
noncomputable def interest_rate_first_year : ℝ := 0.09
noncomputable def principal_second_year : ℝ := principal_first_year * (1 + interest_rate_first_year)
noncomputable def interest_rate_second_year : ℝ := 0.105
noncomputable def principal_third_year : ℝ := principal_second_year * (1 + interest_rate_second_year)
noncomputable def interest_rate_third_year : ℝ := 0.085

noncomputable def compute_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

theorem interest_first_year_correct :
  compute_interest principal_first_year interest_rate_first_year = 810 := by
  sorry

theorem interest_second_year_correct :
  compute_interest principal_second_year interest_rate_second_year = 1034.55 := by
  sorry

theorem interest_third_year_correct :
  compute_interest principal_third_year interest_rate_third_year = 922.18 := by
  sorry

end NUMINAMATH_GPT_interest_first_year_correct_interest_second_year_correct_interest_third_year_correct_l490_49017


namespace NUMINAMATH_GPT_find_range_of_m_l490_49018

def has_two_distinct_real_roots (m : ℝ) : Prop :=
  m^2 - 4 > 0

def inequality_holds_for_all_real_x (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * (m + 1) * x + m * (m + 1) > 0

def p (m : ℝ) : Prop := has_two_distinct_real_roots m
def q (m : ℝ) : Prop := inequality_holds_for_all_real_x m

theorem find_range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m > 2 ∨ (-2 ≤ m ∧ m < -1)) :=
sorry

end NUMINAMATH_GPT_find_range_of_m_l490_49018


namespace NUMINAMATH_GPT_perfect_square_trinomial_l490_49045

theorem perfect_square_trinomial (k : ℤ) : (∀ x : ℤ, x^2 + 2 * (k + 1) * x + 16 = (x + (k + 1))^2) → (k = 3 ∨ k = -5) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l490_49045


namespace NUMINAMATH_GPT_find_N_l490_49066

-- Define the problem parameters
def certain_value : ℝ := 0
def x : ℝ := 10

-- Define the main statement to be proved
theorem find_N (N : ℝ) : 3 * x = (N - x) + certain_value → N = 40 :=
  by sorry

end NUMINAMATH_GPT_find_N_l490_49066


namespace NUMINAMATH_GPT_necessary_not_sufficient_l490_49098

theorem necessary_not_sufficient (a b c : ℝ) : (a < b) → (ac^2 < b * c^2) ∧ ∀a b c : ℝ, (ac^2 < b * c^2) → (a < b) :=
sorry

end NUMINAMATH_GPT_necessary_not_sufficient_l490_49098


namespace NUMINAMATH_GPT_remaining_lives_l490_49022

theorem remaining_lives (initial_players quit1 quit2 player_lives : ℕ) (h1 : initial_players = 15) (h2 : quit1 = 5) (h3 : quit2 = 4) (h4 : player_lives = 7) :
  (initial_players - quit1 - quit2) * player_lives = 42 :=
by
  sorry

end NUMINAMATH_GPT_remaining_lives_l490_49022


namespace NUMINAMATH_GPT_PedoeInequalityHolds_l490_49047

noncomputable def PedoeInequality 
  (a b c a1 b1 c1 : ℝ) (Δ Δ1 : ℝ) :
  Prop :=
  a^2 * (b1^2 + c1^2 - a1^2) + 
  b^2 * (c1^2 + a1^2 - b1^2) + 
  c^2 * (a1^2 + b1^2 - c1^2) >= 16 * Δ * Δ1 

axiom areas_triangle 
  (a b c : ℝ) : ℝ 

axiom areas_triangle1 
  (a1 b1 c1 : ℝ) : ℝ 

theorem PedoeInequalityHolds 
  (a b c a1 b1 c1 : ℝ) 
  (Δ := areas_triangle a b c) 
  (Δ1 := areas_triangle1 a1 b1 c1) :
  PedoeInequality a b c a1 b1 c1 Δ Δ1 :=
sorry

end NUMINAMATH_GPT_PedoeInequalityHolds_l490_49047


namespace NUMINAMATH_GPT_range_of_a_l490_49068

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(x^2 - 2*x + 3 ≤ a^2 - 2*a - 1))
  ↔ (-1 < a ∧ a < 3) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l490_49068


namespace NUMINAMATH_GPT_order_of_expressions_l490_49096

theorem order_of_expressions (k : ℕ) (hk : k > 4) : (k + 2) < (2 * k) ∧ (2 * k) < (k^2) ∧ (k^2) < (2^k) := by
  sorry

end NUMINAMATH_GPT_order_of_expressions_l490_49096


namespace NUMINAMATH_GPT_right_triangle_c_squared_value_l490_49014

theorem right_triangle_c_squared_value (a b c : ℕ) (h : a = 9) (k : b = 12) (right_triangle : True) :
  c^2 = a^2 + b^2 ∨ c^2 = b^2 - a^2 :=
by sorry

end NUMINAMATH_GPT_right_triangle_c_squared_value_l490_49014


namespace NUMINAMATH_GPT_Shekar_marks_in_Science_l490_49039

theorem Shekar_marks_in_Science (S : ℕ) (h : (76 + S + 82 + 67 + 85) / 5 = 75) : S = 65 :=
sorry

end NUMINAMATH_GPT_Shekar_marks_in_Science_l490_49039


namespace NUMINAMATH_GPT_machines_in_first_scenario_l490_49053

theorem machines_in_first_scenario :
  ∃ M : ℕ, (∀ (units1 units2 : ℕ) (hours1 hours2 : ℕ),
    units1 = 20 ∧ hours1 = 10 ∧ units2 = 200 ∧ hours2 = 25 ∧
    (M * units1 / hours1 = 20 * units2 / hours2)) → M = 5 :=
by
  sorry

end NUMINAMATH_GPT_machines_in_first_scenario_l490_49053


namespace NUMINAMATH_GPT_value_of_f_15_l490_49042

def f (n : ℕ) : ℕ := n^2 + 2*n + 19

theorem value_of_f_15 : f 15 = 274 := 
by 
  -- Add proof here
  sorry

end NUMINAMATH_GPT_value_of_f_15_l490_49042


namespace NUMINAMATH_GPT_cost_of_ice_cream_l490_49097

/-- Alok ordered 16 chapatis, 5 plates of rice, 7 plates of mixed vegetable, and 6 ice-cream cups. 
    The cost of each chapati is Rs. 6, that of each plate of rice is Rs. 45, and that of mixed 
    vegetable is Rs. 70. Alok paid the cashier Rs. 931. Prove the cost of each ice-cream cup is Rs. 20. -/
theorem cost_of_ice_cream (n_chapatis n_rice n_vegetable n_ice_cream : ℕ) 
    (cost_chapati cost_rice cost_vegetable total_paid : ℕ)
    (h_chapatis : n_chapatis = 16) 
    (h_rice : n_rice = 5)
    (h_vegetable : n_vegetable = 7)
    (h_ice_cream : n_ice_cream = 6)
    (h_cost_chapati : cost_chapati = 6)
    (h_cost_rice : cost_rice = 45)
    (h_cost_vegetable : cost_vegetable = 70)
    (h_total_paid : total_paid = 931) :
    (total_paid - (n_chapatis * cost_chapati + n_rice * cost_rice + n_vegetable * cost_vegetable)) / n_ice_cream = 20 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_ice_cream_l490_49097


namespace NUMINAMATH_GPT_calc_expr_l490_49078

theorem calc_expr : 
  (1 / (1 + 1 / (2 + 1 / 5))) = 11 / 16 :=
by
  sorry

end NUMINAMATH_GPT_calc_expr_l490_49078


namespace NUMINAMATH_GPT_line_slope_product_l490_49057

theorem line_slope_product (x y : ℝ) (h1 : (x, 6) = (x, 6)) (h2 : (10, y) = (10, y)) (h3 : ∀ x, y = (1 / 2) * x) : x * y = 60 :=
sorry

end NUMINAMATH_GPT_line_slope_product_l490_49057


namespace NUMINAMATH_GPT_find_perimeter_square3_l490_49049

-- Define the conditions: perimeter of first and second square
def perimeter_square1 := 60
def perimeter_square2 := 48

-- Calculate side lengths based on the perimeter
def side_length_square1 := perimeter_square1 / 4
def side_length_square2 := perimeter_square2 / 4

-- Calculate areas of the two squares
def area_square1 := side_length_square1 * side_length_square1
def area_square2 := side_length_square2 * side_length_square2

-- Calculate the area of the third square
def area_square3 := area_square1 - area_square2

-- Calculate the side length of the third square
def side_length_square3 := Nat.sqrt area_square3

-- Define the perimeter of the third square
def perimeter_square3 := 4 * side_length_square3

/-- Theorem: The perimeter of the third square is 36 cm -/
theorem find_perimeter_square3 : perimeter_square3 = 36 := by
  sorry

end NUMINAMATH_GPT_find_perimeter_square3_l490_49049


namespace NUMINAMATH_GPT_correct_answers_is_36_l490_49067

noncomputable def num_correct_answers (c w : ℕ) : Prop :=
  (c + w = 50) ∧ (4 * c - w = 130)

theorem correct_answers_is_36 (c w : ℕ) (h : num_correct_answers c w) : c = 36 :=
by
  sorry

end NUMINAMATH_GPT_correct_answers_is_36_l490_49067


namespace NUMINAMATH_GPT_total_bird_families_l490_49012

-- Declare the number of bird families that flew to Africa
def a : Nat := 47

-- Declare the number of bird families that flew to Asia
def b : Nat := 94

-- Condition that Asia's number of bird families matches Africa + 47 more
axiom h : b = a + 47

-- Prove the total number of bird families is 141
theorem total_bird_families : a + b = 141 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_total_bird_families_l490_49012


namespace NUMINAMATH_GPT_gold_coins_count_l490_49074

theorem gold_coins_count (G : ℕ) 
  (h1 : 50 * G + 125 + 30 = 305) :
  G = 3 := 
by
  sorry

end NUMINAMATH_GPT_gold_coins_count_l490_49074


namespace NUMINAMATH_GPT_sarah_score_l490_49038

theorem sarah_score (s g : ℝ) (h1 : s = g + 50) (h2 : (s + g) / 2 = 110) : s = 135 := 
by 
  sorry

end NUMINAMATH_GPT_sarah_score_l490_49038


namespace NUMINAMATH_GPT_initial_ratio_of_liquids_l490_49071

theorem initial_ratio_of_liquids (p q : ℕ) (h1 : p + q = 40) (h2 : p / (q + 15) = 5 / 6) : p / q = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_initial_ratio_of_liquids_l490_49071


namespace NUMINAMATH_GPT_chord_property_l490_49062

noncomputable def chord_length (R r k : ℝ) : Prop :=
  k = 2 * Real.sqrt (R^2 - r^2)

theorem chord_property (P O : Point) (R k : ℝ) (hR : 0 < R) (hk : 0 < k) :
  ∃ r, r = Real.sqrt (R^2 - k^2 / 4) ∧ chord_length R r k :=
sorry

end NUMINAMATH_GPT_chord_property_l490_49062


namespace NUMINAMATH_GPT_sandwich_cost_90_cents_l490_49095

theorem sandwich_cost_90_cents :
  let cost_bread := 0.15
  let cost_ham := 0.25
  let cost_cheese := 0.35
  (2 * cost_bread + cost_ham + cost_cheese) * 100 = 90 := 
by
  sorry

end NUMINAMATH_GPT_sandwich_cost_90_cents_l490_49095


namespace NUMINAMATH_GPT_perpendicular_lines_a_equals_one_l490_49044

theorem perpendicular_lines_a_equals_one
  (a : ℝ)
  (l1 : ∀ x y : ℝ, x - 2 * y + 1 = 0)
  (l2 : ∀ x y : ℝ, 2 * x + a * y - 1 = 0)
  (perpendicular : ∀ x y : ℝ, (x - 2 * y + 1 = 0) ∧ (2 * x + a * y - 1 = 0) → 
    (-(1 / -2) * -(2 / a)) = -1) :
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_a_equals_one_l490_49044


namespace NUMINAMATH_GPT_overtime_pay_correct_l490_49016

theorem overtime_pay_correct
  (overlap_slow : ℝ := 69) -- Slow clock minute-hand overlap in minutes
  (overlap_normal : ℝ := 12 * 60 / 11) -- Normal clock minute-hand overlap in minutes
  (hours_worked : ℝ := 8) -- The normal working hours a worker believes working
  (hourly_wage : ℝ := 4) -- The normal hourly wage
  (overtime_rate : ℝ := 1.5) -- Overtime pay rate
  (expected_overtime_pay : ℝ := 2.60) -- The expected overtime pay
  
  : hours_worked * (overlap_slow / overlap_normal) * hourly_wage * (overtime_rate - 1) = expected_overtime_pay :=
by
  sorry

end NUMINAMATH_GPT_overtime_pay_correct_l490_49016


namespace NUMINAMATH_GPT_binomial_inequality_l490_49030

theorem binomial_inequality (n : ℕ) (x : ℝ) (h1 : 2 ≤ n) (h2 : |x| < 1) : 
  (1 - x)^n + (1 + x)^n < 2^n := 
by 
  sorry

end NUMINAMATH_GPT_binomial_inequality_l490_49030


namespace NUMINAMATH_GPT_total_amount_paid_l490_49090

variable (n : ℕ) (each_paid : ℕ)

/-- This is a statement that verifies the total amount paid given the number of friends and the amount each friend pays. -/
theorem total_amount_paid (h1 : n = 7) (h2 : each_paid = 70) : n * each_paid = 490 := by
  -- This proof will validate that the total amount paid is 490
  sorry

end NUMINAMATH_GPT_total_amount_paid_l490_49090


namespace NUMINAMATH_GPT_painting_time_l490_49005

theorem painting_time (karl_time leo_time : ℝ) (t : ℝ) (break_time : ℝ) : 
  karl_time = 6 → leo_time = 8 → break_time = 0.5 → 
  (1 / karl_time + 1 / leo_time) * (t - break_time) = 1 :=
by
  intros h_karl h_leo h_break
  rw [h_karl, h_leo, h_break]
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_painting_time_l490_49005


namespace NUMINAMATH_GPT_problem1_problem2_l490_49004

def p (x a : ℝ) : Prop := x^2 + 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - 6*x - 72 <= 0) ∧ (x^2 + x - 6 > 0)

theorem problem1 (x : ℝ) (a : ℝ) (h : a = -1): (p x a ∨ q x) → (-6 ≤ x ∧ x < -3) ∨ (1 < x ∧ x ≤ 12) :=
sorry

theorem problem2 (a : ℝ): (¬ ∃ x : ℝ, p x a) → (¬ ∃ x : ℝ, q x) → (-4 ≤ a ∧ a ≤ -2) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l490_49004


namespace NUMINAMATH_GPT_range_of_x2_y2_l490_49026

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - x^4

theorem range_of_x2_y2 (x y : ℝ) (h : x^2 + y^2 = 2 * x) : 
  0 ≤ x^2 * y^2 ∧ x^2 * y^2 ≤ 27 / 16 :=
sorry

end NUMINAMATH_GPT_range_of_x2_y2_l490_49026


namespace NUMINAMATH_GPT_jade_statue_ratio_l490_49056

/-!
Nancy carves statues out of jade. A giraffe statue takes 120 grams of jade and sells for $150.
An elephant statue sells for $350. Nancy has 1920 grams of jade, and the revenue from selling all
elephant statues is $400 more than selling all giraffe statues.
Prove that the ratio of the amount of jade used for an elephant statue to the amount used for a
giraffe statue is 2.
-/

theorem jade_statue_ratio
  (g_grams : ℕ := 120) -- grams of jade for a giraffe statue
  (g_price : ℕ := 150) -- price of a giraffe statue
  (e_price : ℕ := 350) -- price of an elephant statue
  (total_jade : ℕ := 1920) -- total grams of jade Nancy has
  (additional_revenue : ℕ := 400) -- additional revenue from elephant statues
  (r : ℕ) -- ratio of jade usage of elephant to giraffe statue
  (h : total_jade / g_grams * g_price + additional_revenue = (total_jade / (g_grams * r)) * e_price) :
  r = 2 :=
sorry

end NUMINAMATH_GPT_jade_statue_ratio_l490_49056


namespace NUMINAMATH_GPT_wholesale_price_l490_49051

theorem wholesale_price (RP SP W : ℝ) (h1 : RP = 120)
  (h2 : SP = 0.9 * RP)
  (h3 : SP = W + 0.2 * W) : W = 90 :=
by
  sorry

end NUMINAMATH_GPT_wholesale_price_l490_49051


namespace NUMINAMATH_GPT_two_digit_perfect_squares_divisible_by_3_l490_49091

theorem two_digit_perfect_squares_divisible_by_3 :
  ∃! n1 n2 : ℕ, (10 ≤ n1^2 ∧ n1^2 < 100 ∧ n1^2 % 3 = 0) ∧
               (10 ≤ n2^2 ∧ n2^2 < 100 ∧ n2^2 % 3 = 0) ∧
                (n1 ≠ n2) :=
by sorry

end NUMINAMATH_GPT_two_digit_perfect_squares_divisible_by_3_l490_49091


namespace NUMINAMATH_GPT_least_faces_combined_l490_49003

noncomputable def least_number_of_faces (c d : ℕ) : ℕ :=
c + d

theorem least_faces_combined (c d : ℕ) (h_cge8 : c ≥ 8) (h_dge8 : d ≥ 8)
  (h_sum9_prob : 8 / (c * d) = 1 / 2 * 16 / (c * d))
  (h_sum15_prob : ∃ m : ℕ, m / (c * d) = 1 / 15) :
  least_number_of_faces c d = 28 := sorry

end NUMINAMATH_GPT_least_faces_combined_l490_49003


namespace NUMINAMATH_GPT_find_number_l490_49088

theorem find_number (a b N : ℕ) (h1 : b = 7) (h2 : b - a = 2) (h3 : a * b = 2 * (a + b) + N) : N = 11 :=
  sorry

end NUMINAMATH_GPT_find_number_l490_49088


namespace NUMINAMATH_GPT_walt_age_l490_49099

variable (W M P : ℕ)

-- Conditions
def condition1 := M = 3 * W
def condition2 := M + 12 = 2 * (W + 12)
def condition3 := P = 4 * W
def condition4 := P + 15 = 3 * (W + 15)

theorem walt_age (W M P : ℕ) (h1 : condition1 W M) (h2 : condition2 W M) (h3 : condition3 W P) (h4 : condition4 W P) : 
  W = 30 :=
sorry

end NUMINAMATH_GPT_walt_age_l490_49099


namespace NUMINAMATH_GPT_adam_first_year_students_l490_49070

theorem adam_first_year_students (X : ℕ) 
  (remaining_years_students : ℕ := 9 * 50)
  (total_students : ℕ := 490) 
  (total_years_students : X + remaining_years_students = total_students) : X = 40 :=
by { sorry }

end NUMINAMATH_GPT_adam_first_year_students_l490_49070


namespace NUMINAMATH_GPT_books_bound_l490_49063

theorem books_bound (x : ℕ) (w c : ℕ) (h₀ : w = 92) (h₁ : c = 135) 
(h₂ : 92 - x = 2 * (135 - x)) :
x = 178 :=
by
  sorry

end NUMINAMATH_GPT_books_bound_l490_49063


namespace NUMINAMATH_GPT_police_officer_placement_l490_49055

-- The given problem's conditions
def intersections : Finset String := {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"}

def streets : List (Finset String) := [
    {"A", "B", "C", "D"},        -- Horizontal streets
    {"E", "F", "G"},
    {"H", "I", "J", "K"},
    {"A", "E", "H"},             -- Vertical streets
    {"B", "F", "I"},
    {"D", "G", "J"},
    {"H", "F", "C"},             -- Diagonal streets
    {"C", "G", "K"}
]

def chosen_intersections : Finset String := {"B", "G", "H"}

-- Proof problem
theorem police_officer_placement :
  ∀ street ∈ streets, ∃ p ∈ chosen_intersections, p ∈ street := by
  sorry

end NUMINAMATH_GPT_police_officer_placement_l490_49055


namespace NUMINAMATH_GPT_alice_profit_l490_49065

-- Define the variables and conditions
def total_bracelets : ℕ := 52
def material_cost : ℝ := 3.00
def bracelets_given_away : ℕ := 8
def sale_price : ℝ := 0.25

-- Calculate the number of bracelets sold
def bracelets_sold : ℕ := total_bracelets - bracelets_given_away

-- Calculate the revenue from selling the bracelets
def revenue : ℝ := bracelets_sold * sale_price

-- Define the profit as revenue minus material cost
def profit : ℝ := revenue - material_cost

-- The statement to prove
theorem alice_profit : profit = 8.00 := 
by
  sorry

end NUMINAMATH_GPT_alice_profit_l490_49065


namespace NUMINAMATH_GPT_problem_l490_49011

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

theorem problem (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) : 
  ((f b - f a) / (b - a) < 1 / (a * (a + 1))) :=
by
  sorry -- Proof steps go here

end NUMINAMATH_GPT_problem_l490_49011


namespace NUMINAMATH_GPT_least_number_to_subtract_l490_49077

theorem least_number_to_subtract (n : ℕ) (h : n = 427398) : ∃ k : ℕ, (n - k) % 11 = 0 ∧ k = 4 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l490_49077


namespace NUMINAMATH_GPT_pairwise_coprime_circle_l490_49058

theorem pairwise_coprime_circle :
  ∃ (circle : Fin 100 → ℕ),
    (∀ i, Nat.gcd (circle i) (Nat.gcd (circle ((i + 1) % 100)) (circle ((i - 1) % 100))) = 1) → 
    ∀ i j, i ≠ j → Nat.gcd (circle i) (circle j) = 1 :=
by
  sorry

end NUMINAMATH_GPT_pairwise_coprime_circle_l490_49058


namespace NUMINAMATH_GPT_sequence_general_term_l490_49083

theorem sequence_general_term (n : ℕ) : 
  (∀ (a : ℕ → ℚ), (a 1 = 1) ∧ (a 2 = 2 / 3) ∧ (a 3 = 3 / 7) ∧ (a 4 = 4 / 15) ∧ (a 5 = 5 / 31) → a n = n / (2^n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l490_49083


namespace NUMINAMATH_GPT_inequality_solution_range_l490_49036

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 2| + |x| ≤ a) ↔ a ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_range_l490_49036


namespace NUMINAMATH_GPT_only_rational_root_is_one_l490_49000

-- Define the polynomial
def polynomial_3x5_minus_2x4_plus_5x3_minus_x2_minus_7x_plus_2 (x : ℚ) : ℚ :=
  3 * x^5 - 2 * x^4 + 5 * x^3 - x^2 - 7 * x + 2

-- The main theorem stating that 1 is the only rational root
theorem only_rational_root_is_one : 
  ∀ x : ℚ, polynomial_3x5_minus_2x4_plus_5x3_minus_x2_minus_7x_plus_2 x = 0 ↔ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_only_rational_root_is_one_l490_49000


namespace NUMINAMATH_GPT_total_bottles_in_market_l490_49015

theorem total_bottles_in_market (j w : ℕ) (hj : j = 34) (hw : w = 3 / 2 * j + 3) : j + w = 88 :=
by
  sorry

end NUMINAMATH_GPT_total_bottles_in_market_l490_49015


namespace NUMINAMATH_GPT_sum_of_50th_terms_l490_49032

open Nat

-- Definition of arithmetic sequence
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Definition of geometric sequence
def geometric_sequence (g₁ r n : ℕ) : ℕ := g₁ * r^(n - 1)

-- Prove the sum of the 50th terms of the given sequences
theorem sum_of_50th_terms : 
  arithmetic_sequence 3 6 50 + geometric_sequence 2 3 50 = 297 + 2 * 3^49 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_50th_terms_l490_49032


namespace NUMINAMATH_GPT_carousel_ticket_cost_l490_49013

theorem carousel_ticket_cost :
  ∃ (x : ℕ), 
  (2 * 5) + (3 * x) = 19 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_carousel_ticket_cost_l490_49013


namespace NUMINAMATH_GPT_value_three_std_devs_less_than_mean_l490_49019

-- Define the given conditions as constants.
def mean : ℝ := 16.2
def std_dev : ℝ := 2.3

-- Translate the question into a proof statement.
theorem value_three_std_devs_less_than_mean : mean - 3 * std_dev = 9.3 :=
by sorry

end NUMINAMATH_GPT_value_three_std_devs_less_than_mean_l490_49019


namespace NUMINAMATH_GPT_move_up_4_units_l490_49046

-- Define the given points M and N
def M : ℝ × ℝ := (-1, -1)
def N : ℝ × ℝ := (-1, 3)

-- State the theorem to be proved
theorem move_up_4_units (M N : ℝ × ℝ) :
  (M = (-1, -1)) → (N = (-1, 3)) → (N = (M.1, M.2 + 4)) :=
by
  intros hM hN
  rw [hM, hN]
  sorry

end NUMINAMATH_GPT_move_up_4_units_l490_49046


namespace NUMINAMATH_GPT_max_value_of_f_l490_49007

def f (x : ℝ) : ℝ := 12 * x - 4 * x^2

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 9 :=
by
  have h₁ : ∀ x : ℝ, 12 * x - 4 * x^2 ≤ 9
  { sorry }
  exact h₁

end NUMINAMATH_GPT_max_value_of_f_l490_49007


namespace NUMINAMATH_GPT_square_side_length_s2_l490_49052

theorem square_side_length_s2 (s1 s2 s3 : ℕ)
  (h1 : s1 + s2 + s3 = 3322)
  (h2 : s1 - s2 + s3 = 2020) :
  s2 = 651 :=
by sorry

end NUMINAMATH_GPT_square_side_length_s2_l490_49052


namespace NUMINAMATH_GPT_total_equipment_cost_l490_49086

-- Definitions of costs in USD
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.20
def socks_cost : ℝ := 6.80
def number_of_players : ℝ := 16

-- Statement to prove
theorem total_equipment_cost :
  number_of_players * (jersey_cost + shorts_cost + socks_cost) = 752 :=
by
  sorry

end NUMINAMATH_GPT_total_equipment_cost_l490_49086


namespace NUMINAMATH_GPT_pauline_convertibles_l490_49025

theorem pauline_convertibles : 
  ∀ (total_cars regular_percentage truck_percentage sedan_percentage sports_percentage suv_percentage : ℕ),
  total_cars = 125 →
  regular_percentage = 38 →
  truck_percentage = 12 →
  sedan_percentage = 17 →
  sports_percentage = 22 →
  suv_percentage = 6 →
  (total_cars - (regular_percentage * total_cars / 100 + truck_percentage * total_cars / 100 + sedan_percentage * total_cars / 100 + sports_percentage * total_cars / 100 + suv_percentage * total_cars / 100)) = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_pauline_convertibles_l490_49025


namespace NUMINAMATH_GPT_side_length_square_eq_6_l490_49094

theorem side_length_square_eq_6
  (width length : ℝ)
  (h_width : width = 2)
  (h_length : length = 18) :
  (∃ s : ℝ, s^2 = width * length) ∧ (∀ s : ℝ, s^2 = width * length → s = 6) :=
by
  sorry

end NUMINAMATH_GPT_side_length_square_eq_6_l490_49094


namespace NUMINAMATH_GPT_cash_refund_per_bottle_l490_49035

-- Define the constants based on the conditions
def bottles_per_month : ℕ := 15
def cost_per_bottle : ℝ := 3.0
def bottles_can_buy_with_refund : ℕ := 6
def months_per_year : ℕ := 12

-- Define the total number of bottles consumed in a year
def total_bottles_per_year : ℕ := bottles_per_month * months_per_year

-- Define the total refund in dollars after 1 year
def total_refund_amount : ℝ := bottles_can_buy_with_refund * cost_per_bottle

-- Define the statement we need to prove
theorem cash_refund_per_bottle :
  total_refund_amount / total_bottles_per_year = 0.10 :=
by
  -- This is where the steps would be completed to prove the theorem
  sorry

end NUMINAMATH_GPT_cash_refund_per_bottle_l490_49035


namespace NUMINAMATH_GPT_like_terms_product_l490_49002

theorem like_terms_product :
  ∀ (m n : ℕ),
    (-x^3 * y^n) = (3 * x^m * y^2) → (m = 3 ∧ n = 2) → m * n = 6 :=
by
  intros m n h1 h2
  sorry

end NUMINAMATH_GPT_like_terms_product_l490_49002


namespace NUMINAMATH_GPT_repeat_45_fraction_repeat_245_fraction_l490_49072

-- Define the repeating decimal 0.454545... == n / d
def repeating_45_equiv : Prop := ∃ n d : ℕ, (d ≠ 0) ∧ (0.45454545 = (n : ℚ) / (d : ℚ))

-- First problem statement: 0.4545... == 5 / 11
theorem repeat_45_fraction : 0.45454545 = (5 : ℚ) / (11 : ℚ) :=
by
  sorry

-- Define the repeating decimal 0.2454545... == n / d
def repeating_245_equiv : Prop := ∃ n d : ℕ, (d ≠ 0) ∧ (0.2454545 = (n : ℚ) / (d : ℚ))

-- Second problem statement: 0.2454545... == 27 / 110
theorem repeat_245_fraction : 0.2454545 = (27 : ℚ) / (110 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_repeat_45_fraction_repeat_245_fraction_l490_49072


namespace NUMINAMATH_GPT_percentage_increase_in_second_year_l490_49010

def initial_deposit : ℝ := 5000
def first_year_balance : ℝ := 5500
def two_year_increase_percentage : ℝ := 21
def second_year_increase_percentage : ℝ := 10

theorem percentage_increase_in_second_year
  (initial_deposit first_year_balance : ℝ) 
  (two_year_increase_percentage : ℝ) 
  (h1 : first_year_balance = initial_deposit + 500) 
  (h2 : (initial_deposit * (1 + two_year_increase_percentage / 100)) = initial_deposit * 1.21) 
  : second_year_increase_percentage = 10 := 
sorry

end NUMINAMATH_GPT_percentage_increase_in_second_year_l490_49010


namespace NUMINAMATH_GPT_sum_first_seven_arithmetic_l490_49061

theorem sum_first_seven_arithmetic (a : ℕ) (d : ℕ) (h : a + 3 * d = 3) :
    let a1 := a
    let a2 := a + d
    let a3 := a + 2 * d
    let a4 := a + 3 * d
    let a5 := a + 4 * d
    let a6 := a + 5 * d
    let a7 := a + 6 * d
    a1 + a2 + a3 + a4 + a5 + a6 + a7 = 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_seven_arithmetic_l490_49061


namespace NUMINAMATH_GPT_min_value_f_l490_49034

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 24 * x + 128 / x^3

theorem min_value_f : ∃ x > 0, f x = 168 :=
by
  sorry

end NUMINAMATH_GPT_min_value_f_l490_49034


namespace NUMINAMATH_GPT_family_reunion_people_l490_49054

theorem family_reunion_people (pasta_per_person : ℚ) (total_pasta : ℚ) (recipe_people : ℚ) : 
  pasta_per_person = 2 / 7 ∧ total_pasta = 10 -> recipe_people = 35 :=
by
  sorry

end NUMINAMATH_GPT_family_reunion_people_l490_49054
