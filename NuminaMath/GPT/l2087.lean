import Mathlib

namespace NUMINAMATH_GPT_proof_problem_l2087_208720

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x - 1)^2

theorem proof_problem : f (g (-3)) = 67 := 
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l2087_208720


namespace NUMINAMATH_GPT_find_pairs_l2087_208701

theorem find_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0)
  (cond1 : (m^2 - n) ∣ (m + n^2))
  (cond2 : (n^2 - m) ∣ (n + m^2)) :
  (m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2) := 
sorry

end NUMINAMATH_GPT_find_pairs_l2087_208701


namespace NUMINAMATH_GPT_mike_earnings_first_job_l2087_208753

def total_earnings := 160
def hours_second_job := 12
def hourly_wage_second_job := 9
def earnings_second_job := hours_second_job * hourly_wage_second_job
def earnings_first_job := total_earnings - earnings_second_job

theorem mike_earnings_first_job : 
  earnings_first_job = 160 - (12 * 9) := by
  -- omitted proof
  sorry

end NUMINAMATH_GPT_mike_earnings_first_job_l2087_208753


namespace NUMINAMATH_GPT_smallest_number_exists_l2087_208736

theorem smallest_number_exists (x : ℤ) :
  (x + 3) % 18 = 0 ∧ 
  (x + 3) % 70 = 0 ∧ 
  (x + 3) % 100 = 0 ∧ 
  (x + 3) % 84 = 0 → 
  x = 6297 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_exists_l2087_208736


namespace NUMINAMATH_GPT_cost_price_l2087_208787

theorem cost_price (SP : ℝ) (profit_percent : ℝ) (C : ℝ) 
  (h1 : SP = 400) 
  (h2 : profit_percent = 25) 
  (h3 : SP = C + (profit_percent / 100) * C) : 
  C = 320 := 
by
  sorry

end NUMINAMATH_GPT_cost_price_l2087_208787


namespace NUMINAMATH_GPT_determine_k_l2087_208708

theorem determine_k (k : ℝ) (h1 : ∃ x y : ℝ, y = 4 * x + 3 ∧ y = -2 * x - 25 ∧ y = 3 * x + k) : k = -5 / 3 := by
  sorry

end NUMINAMATH_GPT_determine_k_l2087_208708


namespace NUMINAMATH_GPT_sum_of_sequence_l2087_208768

theorem sum_of_sequence (n : ℕ) (h : n ≥ 2) 
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, n ≥ 2 → S n * S (n-1) + a n = 0) :
  S n = 2 / (2 * n - 1) := by
  sorry

end NUMINAMATH_GPT_sum_of_sequence_l2087_208768


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_l2087_208733

theorem radius_of_inscribed_circle (a b c r : ℝ) (h : a^2 + b^2 = c^2) :
  r = a + b - c :=
sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_l2087_208733


namespace NUMINAMATH_GPT_acceleration_inverse_square_distance_l2087_208777

noncomputable def s (t : ℝ) : ℝ := t^(2/3)

noncomputable def v (t : ℝ) : ℝ := (deriv s t : ℝ)

noncomputable def a (t : ℝ) : ℝ := (deriv v t : ℝ)

theorem acceleration_inverse_square_distance
  (t : ℝ) (h : t ≠ 0) :
  ∃ k : ℝ, k = -2/9 ∧ a t = k / (s t)^2 :=
sorry

end NUMINAMATH_GPT_acceleration_inverse_square_distance_l2087_208777


namespace NUMINAMATH_GPT_sum_x_coords_Q3_is_132_l2087_208734

noncomputable def sum_x_coords_Q3 (x_coords: Fin 44 → ℝ) (sum_x1: ℝ) : ℝ :=
  sum_x1 -- given sum_x1 is the sum of x-coordinates of Q1 i.e., 132

theorem sum_x_coords_Q3_is_132 (x_coords: Fin 44 → ℝ) (sum_x1: ℝ) (h: sum_x1 = 132) :
  sum_x_coords_Q3 x_coords sum_x1 = 132 :=
by
  sorry

end NUMINAMATH_GPT_sum_x_coords_Q3_is_132_l2087_208734


namespace NUMINAMATH_GPT_disk_difference_l2087_208725

/-- Given the following conditions:
    1. Every disk is either blue, yellow, green, or red.
    2. The ratio of blue disks to yellow disks to green disks to red disks is 3 : 7 : 8 : 4.
    3. The total number of disks in the bag is 176.
    Prove that the number of green disks minus the number of blue disks is 40.
-/
theorem disk_difference (b y g r : ℕ) (h_ratio : b * 7 = y * 3 ∧ b * 8 = g * 3 ∧ b * 4 = r * 3) (h_total : b + y + g + r = 176) : g - b = 40 :=
by
  sorry

end NUMINAMATH_GPT_disk_difference_l2087_208725


namespace NUMINAMATH_GPT_find_x_l2087_208797

noncomputable def eq_num (x : ℝ) : Prop :=
  9 - 3 / (1 / 3) + x = 3

theorem find_x : ∃ x : ℝ, eq_num x ∧ x = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l2087_208797


namespace NUMINAMATH_GPT_find_s_l2087_208749

noncomputable def g (x : ℝ) (p q r s : ℝ) : ℝ := x^4 + p * x^3 + q * x^2 + r * x + s

theorem find_s (p q r s : ℝ)
  (h1 : ∀ (x : ℝ), g x p q r s = (x + 1) * (x + 10) * (x + 10) * (x + 10))
  (h2 : p + q + r + s = 2673) :
  s = 1000 := 
  sorry

end NUMINAMATH_GPT_find_s_l2087_208749


namespace NUMINAMATH_GPT_intersection_solution_l2087_208712

theorem intersection_solution (x : ℝ) (y : ℝ) (h₁ : y = 12 / (x^2 + 6)) (h₂ : x + y = 4) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_solution_l2087_208712


namespace NUMINAMATH_GPT_intersection_A_B_l2087_208750

open Set

def SetA : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def SetB : Set ℤ := {x | 0 ≤ x ∧ x ≤ 4}

theorem intersection_A_B :
  (SetA ∩ SetB) = ( {0, 2, 4} : Set ℤ ) :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2087_208750


namespace NUMINAMATH_GPT_Alyssa_spent_in_total_l2087_208758

def amount_paid_for_grapes : ℝ := 12.08
def refund_for_cherries : ℝ := 9.85
def total_spent : ℝ := amount_paid_for_grapes - refund_for_cherries

theorem Alyssa_spent_in_total : total_spent = 2.23 := by
  sorry

end NUMINAMATH_GPT_Alyssa_spent_in_total_l2087_208758


namespace NUMINAMATH_GPT_max_profit_l2087_208735

noncomputable def profit (x : ℕ) : ℝ := -0.15 * (x : ℝ)^2 + 3.06 * (x : ℝ) + 30

theorem max_profit :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 15 ∧ profit x = 45.6 ∧ ∀ y : ℕ, 0 ≤ y ∧ y ≤ 15 → profit y ≤ profit x :=
by
  sorry

end NUMINAMATH_GPT_max_profit_l2087_208735


namespace NUMINAMATH_GPT_P_projection_matrix_P_not_invertible_l2087_208745

noncomputable def v : ℝ × ℝ := (4, -1)

noncomputable def norm_v : ℝ := Real.sqrt (4^2 + (-1)^2)

noncomputable def u : ℝ × ℝ := (4 / norm_v, -1 / norm_v)

noncomputable def P : ℝ × ℝ × ℝ × ℝ :=
((4 * 4) / norm_v^2, (4 * -1) / norm_v^2, 
 (-1 * 4) / norm_v^2, (-1 * -1) / norm_v^2)

theorem P_projection_matrix :
  P = (16 / 17, -4 / 17, -4 / 17, 1 / 17) := by
  sorry

theorem P_not_invertible :
  ¬(∃ Q : ℝ × ℝ × ℝ × ℝ, P = Q) := by
  sorry

end NUMINAMATH_GPT_P_projection_matrix_P_not_invertible_l2087_208745


namespace NUMINAMATH_GPT_total_boxes_is_4575_l2087_208781

-- Define the number of boxes in each warehouse
def num_boxes_in_warehouse_A (x : ℕ) := x
def num_boxes_in_warehouse_B (x : ℕ) := 3 * x
def num_boxes_in_warehouse_C (x : ℕ) := (3 * x) / 2 + 100
def num_boxes_in_warehouse_D (x : ℕ) := 2 * ((3 * x) / 2 + 100) - 50
def num_boxes_in_warehouse_E (x : ℕ) := x + (2 * ((3 * x) / 2 + 100) - 50) - 200

-- Define the condition that warehouse B has 300 more boxes than warehouse E
def condition_B_E (x : ℕ) := 3 * x = num_boxes_in_warehouse_E x + 300

-- Define the total number of boxes calculation
def total_boxes (x : ℕ) := 
    num_boxes_in_warehouse_A x +
    num_boxes_in_warehouse_B x +
    num_boxes_in_warehouse_C x +
    num_boxes_in_warehouse_D x +
    num_boxes_in_warehouse_E x

-- The statement of the problem
theorem total_boxes_is_4575 (x : ℕ) (h : condition_B_E x) : total_boxes x = 4575 :=
by
    sorry

end NUMINAMATH_GPT_total_boxes_is_4575_l2087_208781


namespace NUMINAMATH_GPT_total_workers_in_workshop_l2087_208765

-- Definition of average salary calculation
def average_salary (total_salary : ℕ) (workers : ℕ) : ℕ := total_salary / workers

theorem total_workers_in_workshop :
  ∀ (W T R : ℕ),
  T = 5 →
  average_salary ((W - T) * 750) (W - T) = 700 →
  average_salary (T * 900) T = 900 →
  average_salary (W * 750) W = 750 →
  W = T + R →
  W = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_workers_in_workshop_l2087_208765


namespace NUMINAMATH_GPT_quotient_of_2213_div_13_in_base4_is_53_l2087_208704

-- Definitions of the numbers in base 4
def n₁ : ℕ := 2 * 4^3 + 2 * 4^2 + 1 * 4^1 + 3 * 4^0  -- 2213_4 in base 10
def n₂ : ℕ := 1 * 4^1 + 3 * 4^0  -- 13_4 in base 10

-- The correct quotient in base 4 (converted from quotient in base 10)
def expected_quotient : ℕ := 5 * 4^1 + 3 * 4^0  -- 53_4 in base 10

-- The proposition we want to prove
theorem quotient_of_2213_div_13_in_base4_is_53 : n₁ / n₂ = expected_quotient := by
  sorry

end NUMINAMATH_GPT_quotient_of_2213_div_13_in_base4_is_53_l2087_208704


namespace NUMINAMATH_GPT_relationship_abc_l2087_208786

noncomputable def a := (1 / 3 : ℝ) ^ (2 / 3)
noncomputable def b := (2 / 3 : ℝ) ^ (1 / 3)
noncomputable def c := Real.logb (1/2) (1/3)

theorem relationship_abc : c > b ∧ b > a :=
by
  sorry

end NUMINAMATH_GPT_relationship_abc_l2087_208786


namespace NUMINAMATH_GPT_find_a2_b2_l2087_208756

theorem find_a2_b2 (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 16 = 10 * a * b) : a^2 + b^2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_a2_b2_l2087_208756


namespace NUMINAMATH_GPT_not_product_24_pair_not_24_l2087_208716

theorem not_product_24 (a b : ℤ) : 
  (a, b) = (-4, -6) ∨ (a, b) = (-2, -12) ∨ (a, b) = (2, 12) ∨ (a, b) = (3/4, 32) → a * b = 24 :=
sorry

theorem pair_not_24 :
  ¬(1/3 * -72 = 24) :=
sorry

end NUMINAMATH_GPT_not_product_24_pair_not_24_l2087_208716


namespace NUMINAMATH_GPT_probability_of_Ace_then_King_l2087_208747

def numAces : ℕ := 4
def numKings : ℕ := 4
def totalCards : ℕ := 52

theorem probability_of_Ace_then_King : 
  (numAces / totalCards) * (numKings / (totalCards - 1)) = 4 / 663 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_Ace_then_King_l2087_208747


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2087_208785

theorem solution_set_of_inequality :
  { x : ℝ | 3 ≤ |2 * x - 5| ∧ |2 * x - 5| < 9 } = { x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) } :=
by 
  -- Conditions and steps omitted for the sake of the statement.
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2087_208785


namespace NUMINAMATH_GPT_range_of_a_l2087_208757

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x^2 / a^2 + y^2 / (a + 6) = 1 ∧ (a^2 > a + 6 ∧ a + 6 > 0)) → (a > 3 ∨ (-6 < a ∧ a < -2)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l2087_208757


namespace NUMINAMATH_GPT_sum_of_first_60_digits_l2087_208728

noncomputable def decimal_expansion_period : List ℕ := [0, 0, 0, 8, 1, 0, 3, 7, 2, 7, 7, 1, 4, 7, 4, 8, 7, 8, 4, 4, 4, 0, 8, 4, 2, 7, 8, 7, 6, 8]

def sum_of_list (l : List ℕ) : ℕ := l.foldl (· + ·) 0

theorem sum_of_first_60_digits : sum_of_list (decimal_expansion_period ++ decimal_expansion_period) = 282 := 
by
  simp [decimal_expansion_period, sum_of_list]
  sorry

end NUMINAMATH_GPT_sum_of_first_60_digits_l2087_208728


namespace NUMINAMATH_GPT_sam_has_12_nickels_l2087_208705

theorem sam_has_12_nickels (n d : ℕ) (h1 : n + d = 30) (h2 : 5 * n + 10 * d = 240) : n = 12 :=
sorry

end NUMINAMATH_GPT_sam_has_12_nickels_l2087_208705


namespace NUMINAMATH_GPT_max_sum_cos_l2087_208743

theorem max_sum_cos (a b c : ℝ) (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x) ≥ -1) : a + b + c ≤ 3 := by
  sorry

end NUMINAMATH_GPT_max_sum_cos_l2087_208743


namespace NUMINAMATH_GPT_solve_system_of_equations_l2087_208767

theorem solve_system_of_equations :
  ∀ (x y : ℝ),
  (3 * x - 2 * y = 7) →
  (2 * x + 3 * y = 8) →
  x = 37 / 13 :=
by
  intros x y h1 h2
  -- to prove x = 37 / 13 from the given system of equations
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2087_208767


namespace NUMINAMATH_GPT_calc_f_at_3_l2087_208738

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem calc_f_at_3 : f 3 = 328 := 
sorry

end NUMINAMATH_GPT_calc_f_at_3_l2087_208738


namespace NUMINAMATH_GPT_cupcakes_frosted_in_10_minutes_l2087_208759

theorem cupcakes_frosted_in_10_minutes :
  let cagney_rate := 1 / 25 -- Cagney's rate in cupcakes per second
  let lacey_rate := 1 / 35 -- Lacey's rate in cupcakes per second
  let total_time := 600 -- Total time in seconds for 10 minutes
  let lacey_break := 60 -- Break duration in seconds
  let lacey_work_time := total_time - lacey_break
  let cupcakes_by_cagney := total_time / 25 
  let cupcakes_by_lacey := lacey_work_time / 35
  cupcakes_by_cagney + cupcakes_by_lacey = 39 := 
by {
  sorry
}

end NUMINAMATH_GPT_cupcakes_frosted_in_10_minutes_l2087_208759


namespace NUMINAMATH_GPT_arrange_numbers_l2087_208722

theorem arrange_numbers :
  (2 : ℝ) ^ 1000 < (5 : ℝ) ^ 500 ∧ (5 : ℝ) ^ 500 < (3 : ℝ) ^ 750 :=
by
  sorry

end NUMINAMATH_GPT_arrange_numbers_l2087_208722


namespace NUMINAMATH_GPT_common_solution_l2087_208703

-- Define the conditions of the equations as hypotheses
variables (x y : ℝ)

-- First equation
def eq1 := x^2 + y^2 = 4

-- Second equation
def eq2 := x^2 = 4*y - 8

-- Proof statement: If there exists real numbers x and y such that both equations hold,
-- then y must be equal to 2.
theorem common_solution (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) : y = 2 :=
sorry

end NUMINAMATH_GPT_common_solution_l2087_208703


namespace NUMINAMATH_GPT_carbon_emission_l2087_208784

theorem carbon_emission (x y : ℕ) (h1 : x + y = 70) (h2 : x = 5 * y - 8) : y = 13 ∧ x = 57 := by
  sorry

end NUMINAMATH_GPT_carbon_emission_l2087_208784


namespace NUMINAMATH_GPT_triangle_perimeter_upper_bound_l2087_208795

theorem triangle_perimeter_upper_bound (a b : ℕ) (s : ℕ) (h₁ : a = 7) (h₂ : b = 23) 
  (h₃ : 16 < s) (h₄ : s < 30) : 
  ∃ n : ℕ, n = 60 ∧ n > a + b + s := 
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_upper_bound_l2087_208795


namespace NUMINAMATH_GPT_intersection_M_N_l2087_208751

def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2087_208751


namespace NUMINAMATH_GPT_num_packs_blue_tshirts_l2087_208731

def num_white_tshirts_per_pack : ℕ := 6
def num_packs_white_tshirts : ℕ := 5
def num_blue_tshirts_per_pack : ℕ := 9
def total_num_tshirts : ℕ := 57

theorem num_packs_blue_tshirts : (total_num_tshirts - num_white_tshirts_per_pack * num_packs_white_tshirts) / num_blue_tshirts_per_pack = 3 := by
  sorry

end NUMINAMATH_GPT_num_packs_blue_tshirts_l2087_208731


namespace NUMINAMATH_GPT_symmetric_circle_eq_l2087_208727

-- Define the original circle equation
def originalCircle (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 4

-- Define the equation of the circle symmetric to the original with respect to the y-axis
def symmetricCircle (x y : ℝ) : Prop :=
  (x + 1) ^ 2 + (y - 2) ^ 2 = 4

-- Theorem to prove that the symmetric circle equation is correct
theorem symmetric_circle_eq :
  ∀ x y : ℝ, originalCircle x y → symmetricCircle (-x) y := 
by
  sorry

end NUMINAMATH_GPT_symmetric_circle_eq_l2087_208727


namespace NUMINAMATH_GPT_boat_stream_ratio_l2087_208798

theorem boat_stream_ratio (B S : ℝ) (h : 1 / (B - S) = 2 * (1 / (B + S))) : B / S = 3 :=
by
  sorry

end NUMINAMATH_GPT_boat_stream_ratio_l2087_208798


namespace NUMINAMATH_GPT_sin_75_eq_sqrt6_add_sqrt2_div4_l2087_208775

theorem sin_75_eq_sqrt6_add_sqrt2_div4 :
  Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
sorry

end NUMINAMATH_GPT_sin_75_eq_sqrt6_add_sqrt2_div4_l2087_208775


namespace NUMINAMATH_GPT_least_positive_integer_solution_l2087_208760

theorem least_positive_integer_solution :
  ∃ b : ℕ, b ≡ 1 [MOD 4] ∧ b ≡ 2 [MOD 5] ∧ b ≡ 3 [MOD 6] ∧ b = 37 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_solution_l2087_208760


namespace NUMINAMATH_GPT_houses_with_only_one_pet_l2087_208780

theorem houses_with_only_one_pet (h_total : ∃ t : ℕ, t = 75)
                                 (h_dogs : ∃ d : ℕ, d = 40)
                                 (h_cats : ∃ c : ℕ, c = 30)
                                 (h_dogs_and_cats : ∃ dc : ℕ, dc = 10)
                                 (h_birds : ∃ b : ℕ, b = 8)
                                 (h_cats_and_birds : ∃ cb : ℕ, cb = 5)
                                 (h_no_dogs_and_birds : ∀ db : ℕ, ¬ (∃ db : ℕ, db = 1)) :
  ∃ n : ℕ, n = 48 :=
by
  have only_dogs := 40 - 10
  have only_cats := 30 - 10 - 5
  have only_birds := 8 - 5
  have result := only_dogs + only_cats + only_birds
  exact ⟨result, sorry⟩

end NUMINAMATH_GPT_houses_with_only_one_pet_l2087_208780


namespace NUMINAMATH_GPT_math_problem_l2087_208713

   theorem math_problem :
     6 * (-1 / 2) + Real.sqrt 3 * Real.sqrt 8 + (-15 : ℝ)^0 = 2 * Real.sqrt 6 - 2 :=
   by
     sorry
   
end NUMINAMATH_GPT_math_problem_l2087_208713


namespace NUMINAMATH_GPT_trapezoid_area_l2087_208789

open Real

theorem trapezoid_area 
  (r : ℝ) (BM CD AB : ℝ) (radius_nonneg : 0 ≤ r) 
  (BM_positive : 0 < BM) (CD_positive : 0 < CD) (AB_positive : 0 < AB)
  (circle_radius : r = 4) (BM_length : BM = 16) (CD_length : CD = 3) :
  let height := 2 * r
  let base_sum := AB + CD
  let area := height * base_sum / 2
  AB = BM + 8 → area = 108 :=
by
  intro hyp
  sorry

end NUMINAMATH_GPT_trapezoid_area_l2087_208789


namespace NUMINAMATH_GPT_triangle_solution_proof_l2087_208792

noncomputable def solve_triangle_proof (a b c : ℝ) (alpha beta gamma : ℝ) : Prop :=
  a = 631.28 ∧
  alpha = 63 + 35 / 60 + 30 / 3600 ∧
  b - c = 373 ∧
  beta = 88 + 12 / 60 + 15 / 3600 ∧
  gamma = 28 + 12 / 60 + 15 / 3600 ∧
  b = 704.55 ∧
  c = 331.55

theorem triangle_solution_proof : solve_triangle_proof 631.28 704.55 331.55 (63 + 35 / 60 + 30 / 3600) (88 + 12 / 60 + 15 / 3600) (28 + 12 / 60 + 15 / 3600) :=
  by { sorry }

end NUMINAMATH_GPT_triangle_solution_proof_l2087_208792


namespace NUMINAMATH_GPT_points_for_level_completion_l2087_208741

-- Condition definitions
def enemies_defeated : ℕ := 6
def points_per_enemy : ℕ := 9
def total_points : ℕ := 62

-- Derived definitions (based on the problem steps):
def points_from_enemies : ℕ := enemies_defeated * points_per_enemy
def points_for_completing_level : ℕ := total_points - points_from_enemies

-- Theorem statement
theorem points_for_level_completion : points_for_completing_level = 8 := by
  sorry

end NUMINAMATH_GPT_points_for_level_completion_l2087_208741


namespace NUMINAMATH_GPT_range_a_l2087_208769

variable (a : ℝ)

def p := (∀ x : ℝ, x^2 + x + a > 0)
def q := ∃ x y : ℝ, x^2 - 2 * a * x + 1 ≤ y

theorem range_a :
  ({a : ℝ | (p a ∧ ¬q a) ∨ (¬p a ∧ q a)} = {a : ℝ | a < -1} ∪ {a : ℝ | 1 / 4 < a ∧ a < 1}) := 
by
  sorry

end NUMINAMATH_GPT_range_a_l2087_208769


namespace NUMINAMATH_GPT_total_interest_correct_l2087_208719

-- Initial conditions
def initial_investment : ℝ := 1200
def annual_interest_rate : ℝ := 0.08
def additional_deposit : ℝ := 500
def first_period : ℕ := 2
def second_period : ℕ := 2

-- Calculate the accumulated value after the first period
def first_accumulated_value : ℝ := initial_investment * (1 + annual_interest_rate)^first_period

-- Calculate the new principal after additional deposit
def new_principal := first_accumulated_value + additional_deposit

-- Calculate the accumulated value after the second period
def final_value := new_principal * (1 + annual_interest_rate)^second_period

-- Calculate the total interest earned after 4 years
def total_interest_earned := final_value - initial_investment - additional_deposit

-- Final theorem statement to be proven
theorem total_interest_correct : total_interest_earned = 515.26 :=
by sorry

end NUMINAMATH_GPT_total_interest_correct_l2087_208719


namespace NUMINAMATH_GPT_sum_even_minus_odd_from_1_to_100_l2087_208782

noncomputable def sum_even_numbers : Nat :=
  (List.range' 2 99 2).sum

noncomputable def sum_odd_numbers : Nat :=
  (List.range' 1 100 2).sum

theorem sum_even_minus_odd_from_1_to_100 :
  sum_even_numbers - sum_odd_numbers = 50 :=
by
  sorry

end NUMINAMATH_GPT_sum_even_minus_odd_from_1_to_100_l2087_208782


namespace NUMINAMATH_GPT_problem1_sin_cos_problem2_linear_combination_l2087_208766

/-- Problem 1: Prove that sin(α) * cos(α) = -2/5 given that the terminal side of angle α passes through (-1, 2) --/
theorem problem1_sin_cos (α : ℝ) (x y : ℝ) (h1 : x = -1) (h2 : y = 2) (h3 : Real.sqrt (x^2 + y^2) ≠ 0) :
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

/-- Problem 2: Prove that 10sin(α) + 3cos(α) = 0 given that the terminal side of angle α lies on the line y = -3x --/
theorem problem2_linear_combination (α : ℝ) (x y : ℝ) (h1 : y = -3 * x) (h2 : (x = -1 ∧ y = 3) ∨ (x = 1 ∧ y = -3)) (h3 : Real.sqrt (x^2 + y^2) ≠ 0) :
  10 * Real.sin α + 3 / Real.cos α = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem1_sin_cos_problem2_linear_combination_l2087_208766


namespace NUMINAMATH_GPT_solve_r_l2087_208761

theorem solve_r (k r : ℝ) (h1 : 3 = k * 2^r) (h2 : 15 = k * 4^r) : 
  r = Real.log 5 / Real.log 2 := 
sorry

end NUMINAMATH_GPT_solve_r_l2087_208761


namespace NUMINAMATH_GPT_correct_computation_l2087_208796

theorem correct_computation (x : ℕ) (h : x - 20 = 52) : x / 4 = 18 :=
  sorry

end NUMINAMATH_GPT_correct_computation_l2087_208796


namespace NUMINAMATH_GPT_product_of_solutions_of_abs_eq_l2087_208702

theorem product_of_solutions_of_abs_eq (x : ℝ) (h : |x - 5| - 4 = 3) : x * (if x = 12 then -2 else if x = -2 then 12 else 1) = -24 :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_of_abs_eq_l2087_208702


namespace NUMINAMATH_GPT_distances_perimeter_inequality_l2087_208711

variable {Point Polygon : Type}

-- Definitions for the conditions
variables (O : Point) (M : Polygon)
variable (ρ : ℝ) -- perimeter of M
variable (d : ℝ) -- sum of distances to each vertex of M from O
variable (h : ℝ) -- sum of distances to each side of M from O

-- The theorem statement
theorem distances_perimeter_inequality :
  d^2 - h^2 ≥ ρ^2 / 4 :=
by
  sorry

end NUMINAMATH_GPT_distances_perimeter_inequality_l2087_208711


namespace NUMINAMATH_GPT_domain_of_sqrt_cosine_sub_half_l2087_208762

theorem domain_of_sqrt_cosine_sub_half :
  {x : ℝ | ∃ k : ℤ, (2 * k * π - π / 3) ≤ x ∧ x ≤ (2 * k * π + π / 3)} =
  {x : ℝ | ∃ k : ℤ, 2 * k * π - π / 3 ≤ x ∧ x ≤ 2 * k * π + π / 3} :=
by sorry

end NUMINAMATH_GPT_domain_of_sqrt_cosine_sub_half_l2087_208762


namespace NUMINAMATH_GPT_factorize_poly1_factorize_poly2_l2087_208755

variable (a b m n : ℝ)

theorem factorize_poly1 : 3 * a^2 - 6 * a * b + 3 * b^2 = 3 * (a - b)^2 :=
sorry

theorem factorize_poly2 : 4 * m^2 - 9 * n^2 = (2 * m - 3 * n) * (2 * m + 3 * n) :=
sorry

end NUMINAMATH_GPT_factorize_poly1_factorize_poly2_l2087_208755


namespace NUMINAMATH_GPT_combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400_l2087_208742

noncomputable def num_combinations_4_blocks_no_same_row_col :=
  (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4)

theorem combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400 :
  num_combinations_4_blocks_no_same_row_col = 5400 := 
by
  sorry

end NUMINAMATH_GPT_combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400_l2087_208742


namespace NUMINAMATH_GPT_intersection_complement_l2087_208721

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- The complement of B in U
def complement_U (U B : Set ℕ) : Set ℕ := U \ B

-- Statement to prove
theorem intersection_complement : A ∩ (complement_U U B) = {1} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_complement_l2087_208721


namespace NUMINAMATH_GPT_baseball_cap_problem_l2087_208794

theorem baseball_cap_problem 
  (n_first_week n_second_week n_third_week n_fourth_week total_caps : ℕ) 
  (h2 : n_second_week = 400) 
  (h3 : n_third_week = 300) 
  (h4 : n_fourth_week = (n_first_week + n_second_week + n_third_week) / 3) 
  (h_total : n_first_week + n_second_week + n_third_week + n_fourth_week = 1360) : 
  n_first_week = 320 := 
by 
  sorry

end NUMINAMATH_GPT_baseball_cap_problem_l2087_208794


namespace NUMINAMATH_GPT_solve_exp_equation_l2087_208744

theorem solve_exp_equation (e : ℝ) (x : ℝ) (h_e : e = Real.exp 1) :
  e^x + 2 * e^(-x) = 3 ↔ x = 0 ∨ x = Real.log 2 :=
sorry

end NUMINAMATH_GPT_solve_exp_equation_l2087_208744


namespace NUMINAMATH_GPT_days_for_Q_wages_l2087_208763

variables (P Q S : ℝ) (D : ℝ)

theorem days_for_Q_wages (h1 : S = 24 * P) (h2 : S = 15 * (P + Q)) : S = D * Q → D = 40 :=
by
  sorry

end NUMINAMATH_GPT_days_for_Q_wages_l2087_208763


namespace NUMINAMATH_GPT_solve_for_x_l2087_208788

theorem solve_for_x (x : ℚ) (h : 5 * (x - 6) = 3 * (3 - 3 * x) + 9) : x = 24 / 7 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l2087_208788


namespace NUMINAMATH_GPT_rectangle_area_l2087_208715

theorem rectangle_area (y : ℝ) (h : y > 0) 
    (h_area : ∃ (E F G H : ℝ × ℝ), 
        E = (0, 0) ∧ 
        F = (0, 5) ∧ 
        G = (y, 5) ∧ 
        H = (y, 0) ∧ 
        5 * y = 45) : 
    y = 9 := 
by
    sorry

end NUMINAMATH_GPT_rectangle_area_l2087_208715


namespace NUMINAMATH_GPT_no_such_function_exists_l2087_208740

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ m n : ℕ, (m + f n)^2 ≥ 3 * (f m)^2 + n^2 :=
by 
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l2087_208740


namespace NUMINAMATH_GPT_bc_lt_3ad_l2087_208717

theorem bc_lt_3ad {a b c d x1 x2 x3 : ℝ}
    (h1 : a ≠ 0)
    (h2 : x1 > 0 ∧ x2 > 0 ∧ x3 > 0)
    (h3 : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)
    (h4 : x1 + x2 + x3 = -b / a)
    (h5 : x1 * x2 + x2 * x3 + x1 * x3 = c / a)
    (h6 : x1 * x2 * x3 = -d / a) : 
    b * c < 3 * a * d := 
sorry

end NUMINAMATH_GPT_bc_lt_3ad_l2087_208717


namespace NUMINAMATH_GPT_number_of_seeds_per_row_l2087_208729

-- Define the conditions as variables
def rows : ℕ := 6
def total_potatoes : ℕ := 54
def seeds_per_row : ℕ := 9

-- State the theorem
theorem number_of_seeds_per_row :
  total_potatoes / rows = seeds_per_row :=
by
-- We ignore the proof here, it will be provided later
sorry

end NUMINAMATH_GPT_number_of_seeds_per_row_l2087_208729


namespace NUMINAMATH_GPT_ratio_of_sums_of_sides_and_sines_l2087_208783

theorem ratio_of_sums_of_sides_and_sines (A B C : ℝ) (a b c : ℝ) (hA : A = 60) (ha : a = 3) 
  (h : a / Real.sin A = b / Real.sin B ∧ a / Real.sin A = c / Real.sin C) : 
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_sums_of_sides_and_sines_l2087_208783


namespace NUMINAMATH_GPT_sum_of_cubes_mod_4_l2087_208752

theorem sum_of_cubes_mod_4 :
  let b := 2
  let n := 2010
  ( (n * (n + 1) / 2) ^ 2 ) % (b ^ 2) = 1 :=
by
  let b := 2
  let n := 2010
  sorry

end NUMINAMATH_GPT_sum_of_cubes_mod_4_l2087_208752


namespace NUMINAMATH_GPT_Beast_of_War_running_time_l2087_208726

theorem Beast_of_War_running_time 
  (M : ℕ) 
  (AE : ℕ) 
  (BoWAC : ℕ)
  (h1 : M = 120)
  (h2 : AE = M - 30)
  (h3 : BoWAC = AE + 10) : 
  BoWAC = 100 
  := 
sorry

end NUMINAMATH_GPT_Beast_of_War_running_time_l2087_208726


namespace NUMINAMATH_GPT_yen_checking_account_l2087_208714

theorem yen_checking_account (savings : ℕ) (total : ℕ) (checking : ℕ) (h1 : savings = 3485) (h2 : total = 9844) (h3 : checking = total - savings) :
  checking = 6359 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_yen_checking_account_l2087_208714


namespace NUMINAMATH_GPT_least_n_divisible_by_25_and_7_l2087_208779

theorem least_n_divisible_by_25_and_7 (n : ℕ) (h1 : n > 1) (h2 : n % 25 = 1) (h3 : n % 7 = 1) : n = 126 :=
by
  sorry

end NUMINAMATH_GPT_least_n_divisible_by_25_and_7_l2087_208779


namespace NUMINAMATH_GPT_fraction_power_minus_one_l2087_208754

theorem fraction_power_minus_one :
  (5 / 3) ^ 4 - 1 = 544 / 81 := 
by
  sorry

end NUMINAMATH_GPT_fraction_power_minus_one_l2087_208754


namespace NUMINAMATH_GPT_riya_speed_l2087_208771

theorem riya_speed 
  (R : ℝ)
  (priya_speed : ℝ) 
  (time : ℝ) 
  (distance : ℝ)
  (h_priya_speed : priya_speed = 22)
  (h_time : time = 1)
  (h_distance : distance = 43)
  : R + priya_speed * time = distance → R = 21 :=
by 
  sorry

end NUMINAMATH_GPT_riya_speed_l2087_208771


namespace NUMINAMATH_GPT_number_of_rectangular_arrays_l2087_208764

theorem number_of_rectangular_arrays (n : ℕ) (h : n = 48) : 
  ∃ k : ℕ, (k = 6 ∧ ∀ m p : ℕ, m * p = n → m ≥ 3 → p ≥ 3 → m = 3 ∨ m = 4 ∨ m = 6 ∨ m = 8 ∨ m = 12 ∨ m = 16 ∨ m = 24) :=
by
  sorry

end NUMINAMATH_GPT_number_of_rectangular_arrays_l2087_208764


namespace NUMINAMATH_GPT_positive_distinct_solutions_conditons_l2087_208773

-- Definitions corresponding to the conditions in the problem
variables {x y z a b : ℝ}

-- The statement articulates the condition
theorem positive_distinct_solutions_conditons (h1 : x + y + z = a) (h2 : x^2 + y^2 + z^2 = b^2) (h3 : xy = z^2) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) (h7 : x ≠ y) (h8 : y ≠ z) (h9 : x ≠ z) : 
  b^2 ≥ a^2 / 2 :=
sorry

end NUMINAMATH_GPT_positive_distinct_solutions_conditons_l2087_208773


namespace NUMINAMATH_GPT_rhombus_area_l2087_208799

-- Definitions
def side_length := 25 -- cm
def diagonal1 := 30 -- cm

-- Statement to prove
theorem rhombus_area (s : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_s : s = 25) 
  (h_d1 : d1 = 30)
  (h_side : s^2 = (d1/2)^2 + (d2/2)^2) :
  (d1 * d2) / 2 = 600 :=
by sorry

end NUMINAMATH_GPT_rhombus_area_l2087_208799


namespace NUMINAMATH_GPT_fraction_simplify_l2087_208739

theorem fraction_simplify : (7 + 21) / (14 + 42) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_fraction_simplify_l2087_208739


namespace NUMINAMATH_GPT_ratio_yellow_jelly_beans_l2087_208774

theorem ratio_yellow_jelly_beans :
  let bag_A_total := 24
  let bag_B_total := 30
  let bag_C_total := 32
  let bag_D_total := 34
  let bag_A_yellow_ratio := 0.40
  let bag_B_yellow_ratio := 0.30
  let bag_C_yellow_ratio := 0.25 
  let bag_D_yellow_ratio := 0.10
  let bag_A_yellow := bag_A_total * bag_A_yellow_ratio
  let bag_B_yellow := bag_B_total * bag_B_yellow_ratio
  let bag_C_yellow := bag_C_total * bag_C_yellow_ratio
  let bag_D_yellow := bag_D_total * bag_D_yellow_ratio
  let total_yellow := bag_A_yellow + bag_B_yellow + bag_C_yellow + bag_D_yellow
  let total_beans := bag_A_total + bag_B_total + bag_C_total + bag_D_total
  (total_yellow / total_beans) = 0.25 := by
  sorry

end NUMINAMATH_GPT_ratio_yellow_jelly_beans_l2087_208774


namespace NUMINAMATH_GPT_project_completion_time_l2087_208732

theorem project_completion_time (m n : ℝ) (hm : m > 0) (hn : n > 0):
  (1 / (1 / m + 1 / n)) = (m * n) / (m + n) :=
by
  sorry

end NUMINAMATH_GPT_project_completion_time_l2087_208732


namespace NUMINAMATH_GPT_People_Distribution_l2087_208706

theorem People_Distribution 
  (total_people : ℕ) 
  (total_buses : ℕ) 
  (equal_distribution : ℕ) 
  (h1 : total_people = 219) 
  (h2 : total_buses = 3) 
  (h3 : equal_distribution = total_people / total_buses) : 
  equal_distribution = 73 :=
by 
  intros 
  sorry

end NUMINAMATH_GPT_People_Distribution_l2087_208706


namespace NUMINAMATH_GPT_total_interest_paid_l2087_208700

-- Define the problem as a theorem in Lean 4
theorem total_interest_paid
  (initial_investment : ℝ)
  (interest_6_months : ℝ)
  (interest_10_months : ℝ)
  (interest_18_months : ℝ)
  (total_interest : ℝ) :
  initial_investment = 10000 ∧ 
  interest_6_months = 0.02 * initial_investment ∧
  interest_10_months = 0.03 * (initial_investment + interest_6_months) ∧
  interest_18_months = 0.04 * (initial_investment + interest_6_months + interest_10_months) ∧
  total_interest = interest_6_months + interest_10_months + interest_18_months →
  total_interest = 926.24 :=
by
  sorry

end NUMINAMATH_GPT_total_interest_paid_l2087_208700


namespace NUMINAMATH_GPT_contradiction_assumption_l2087_208772

variable (x y z : ℝ)

/-- The negation of "at least one is positive" for proof by contradiction is 
    "all three numbers are non-positive". -/
theorem contradiction_assumption (h : ¬ (x > 0 ∨ y > 0 ∨ z > 0)) : 
  (x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_contradiction_assumption_l2087_208772


namespace NUMINAMATH_GPT_rock_height_at_30_l2087_208723

theorem rock_height_at_30 (t : ℝ) (h : ℝ) 
  (h_eq : h = 80 - 9 * t - 5 * t^2) 
  (h_30 : h = 30) : 
  t = 2.3874 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_rock_height_at_30_l2087_208723


namespace NUMINAMATH_GPT_num_four_letter_initials_l2087_208770

theorem num_four_letter_initials : 
  (10 : ℕ)^4 = 10000 := 
by 
  sorry

end NUMINAMATH_GPT_num_four_letter_initials_l2087_208770


namespace NUMINAMATH_GPT_fraction_division_l2087_208737

variable {x : ℝ}
variable (hx : x ≠ 0)

theorem fraction_division (hx : x ≠ 0) : (3 / 8) / (5 * x / 12) = 9 / (10 * x) := 
by
  sorry

end NUMINAMATH_GPT_fraction_division_l2087_208737


namespace NUMINAMATH_GPT_dealer_sold_BMWs_l2087_208718

theorem dealer_sold_BMWs (total_cars : ℕ) (ford_pct toyota_pct nissan_pct bmw_pct : ℝ)
  (h_total_cars : total_cars = 300)
  (h_ford_pct : ford_pct = 0.1)
  (h_toyota_pct : toyota_pct = 0.2)
  (h_nissan_pct : nissan_pct = 0.3)
  (h_bmw_pct : bmw_pct = 1 - (ford_pct + toyota_pct + nissan_pct)) :
  total_cars * bmw_pct = 120 := by
  sorry

end NUMINAMATH_GPT_dealer_sold_BMWs_l2087_208718


namespace NUMINAMATH_GPT_add_base3_numbers_l2087_208707

theorem add_base3_numbers : 
  (2 + 1 * 3) + (0 + 2 * 3 + 1 * 3^2) + 
  (1 + 2 * 3 + 0 * 3^2 + 2 * 3^3) + (2 + 0 * 3 + 1 * 3^2 + 2 * 3^3)
  = 2 + 2 * 3 + 2 * 3^2 + 2 * 3^3 := 
by sorry

end NUMINAMATH_GPT_add_base3_numbers_l2087_208707


namespace NUMINAMATH_GPT_solution_l2087_208724

def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem solution (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by 
  -- Here we will skip the actual proof by using sorry
  sorry

end NUMINAMATH_GPT_solution_l2087_208724


namespace NUMINAMATH_GPT_find_width_of_room_eq_l2087_208776

noncomputable def total_cost : ℝ := 20625
noncomputable def rate_per_sqm : ℝ := 1000
noncomputable def length_of_room : ℝ := 5.5
noncomputable def area_paved : ℝ := total_cost / rate_per_sqm
noncomputable def width_of_room : ℝ := area_paved / length_of_room

theorem find_width_of_room_eq :
  width_of_room = 3.75 :=
sorry

end NUMINAMATH_GPT_find_width_of_room_eq_l2087_208776


namespace NUMINAMATH_GPT_sum_of_first_3n_terms_l2087_208778

-- Define the sums of the geometric sequence
variable (S_n S_2n S_3n : ℕ)

-- Given conditions
variable (h1 : S_n = 48)
variable (h2 : S_2n = 60)

-- The statement we need to prove
theorem sum_of_first_3n_terms (S_n S_2n S_3n : ℕ) (h1 : S_n = 48) (h2 : S_2n = 60) :
  S_3n = 63 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_3n_terms_l2087_208778


namespace NUMINAMATH_GPT_sequoia_taller_than_maple_l2087_208791

def height_maple_tree : ℚ := 13 + 3/4
def height_sequoia : ℚ := 20 + 1/2

theorem sequoia_taller_than_maple : (height_sequoia - height_maple_tree) = 6 + 3/4 :=
by
  sorry

end NUMINAMATH_GPT_sequoia_taller_than_maple_l2087_208791


namespace NUMINAMATH_GPT_problem_statement_l2087_208746

theorem problem_statement (a b c x y z : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < x) (h5 : 0 < y) (h6 : 0 < z)
  (h7 : a^2 + b^2 + c^2 = 16) (h8 : x^2 + y^2 + z^2 = 49) (h9 : a * x + b * y + c * z = 28) : 
  (a + b + c) / (x + y + z) = 4 / 7 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2087_208746


namespace NUMINAMATH_GPT_perpendicular_line_l2087_208709

theorem perpendicular_line 
  (a b c : ℝ) 
  (p : ℝ × ℝ) 
  (h₁ : p = (-1, 3)) 
  (h₂ : a * (-1) + b * 3 + c = 0) 
  (h₃ : a * p.fst + b * p.snd + c = 0) 
  (hp : a = 1 ∧ b = -2 ∧ c = 3) : 
  ∃ a₁ b₁ c₁ : ℝ, 
  a₁ * (-1) + b₁ * 3 + c₁ = 0 ∧ a₁ = 2 ∧ b₁ = 1 ∧ c₁ = -1 := 
by 
  sorry

end NUMINAMATH_GPT_perpendicular_line_l2087_208709


namespace NUMINAMATH_GPT_scientific_notation_of_29_47_thousand_l2087_208793

theorem scientific_notation_of_29_47_thousand :
  (29.47 * 1000 = 2.947 * 10^4) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_29_47_thousand_l2087_208793


namespace NUMINAMATH_GPT_find_q_l2087_208748

theorem find_q (p q : ℝ) (hp : 1 < p) (hq : 1 < q) (hcond1 : 1/p + 1/q = 1) (hcond2 : p * q = 9) :
    q = (9 + 3 * Real.sqrt 5) / 2 ∨ q = (9 - 3 * Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l2087_208748


namespace NUMINAMATH_GPT_subsets_containing_six_l2087_208730

theorem subsets_containing_six :
  ∃ s : Finset (Fin 6), s = {1, 2, 3, 4, 5, 6} ∧ (∃ n : ℕ, n = 32 ∧ n = 2 ^ 5) := by
  sorry

end NUMINAMATH_GPT_subsets_containing_six_l2087_208730


namespace NUMINAMATH_GPT_curve_is_circle_l2087_208710

theorem curve_is_circle : ∀ (θ : ℝ), ∃ r : ℝ, r = 3 * Real.cos θ → ∃ (x y : ℝ), x^2 + y^2 = (3/2)^2 :=
by
  intro θ
  use 3 * Real.cos θ
  sorry

end NUMINAMATH_GPT_curve_is_circle_l2087_208710


namespace NUMINAMATH_GPT_factorization_of_square_difference_l2087_208790

variable (t : ℝ)

theorem factorization_of_square_difference : t^2 - 144 = (t - 12) * (t + 12) := 
sorry

end NUMINAMATH_GPT_factorization_of_square_difference_l2087_208790
