import Mathlib

namespace NUMINAMATH_GPT_remainder_when_dividing_by_2x_minus_4_l1681_168129

def f (x : ℝ) := 4 * x^3 - 9 * x^2 + 12 * x - 14
def g (x : ℝ) := 2 * x - 4

theorem remainder_when_dividing_by_2x_minus_4 : f 2 = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_when_dividing_by_2x_minus_4_l1681_168129


namespace NUMINAMATH_GPT_prove_function_domain_l1681_168160

def function_domain := {x : ℝ | (x + 4 ≥ 0 ∧ x ≠ 0)}

theorem prove_function_domain :
  function_domain = {x : ℝ | x ∈ (Set.Icc (-4:ℝ) 0).diff ({0}:Set ℝ) ∪ (Set.Ioi 0)} :=
by
  sorry

end NUMINAMATH_GPT_prove_function_domain_l1681_168160


namespace NUMINAMATH_GPT_Jane_age_l1681_168111

theorem Jane_age (J A : ℕ) (h1 : J + A = 54) (h2 : J - A = 22) : A = 16 := 
by 
  sorry

end NUMINAMATH_GPT_Jane_age_l1681_168111


namespace NUMINAMATH_GPT_remainder_98765432101_div_240_l1681_168184

theorem remainder_98765432101_div_240 :
  (98765432101 % 240) = 61 :=
by
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_remainder_98765432101_div_240_l1681_168184


namespace NUMINAMATH_GPT_addition_problem_l1681_168199

theorem addition_problem (x y S : ℕ) 
    (h1 : x = S - 2000)
    (h2 : S = y + 6) :
    x = 6 ∧ y = 2000 ∧ S = 2006 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_addition_problem_l1681_168199


namespace NUMINAMATH_GPT_expression_in_terms_of_p_q_l1681_168192

-- Define the roots and the polynomials conditions
variable (α β γ δ : ℝ)
variable (p q : ℝ)

-- The conditions of the problem
axiom roots_poly1 : α * β = 1 ∧ α + β = -p
axiom roots_poly2 : γ * δ = 1 ∧ γ + δ = -q

theorem expression_in_terms_of_p_q :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
sorry

end NUMINAMATH_GPT_expression_in_terms_of_p_q_l1681_168192


namespace NUMINAMATH_GPT_simplest_form_eq_a_l1681_168107

theorem simplest_form_eq_a (a : ℝ) (h : a ≠ 1) : 1 - (1 / (1 + (a / (1 - a)))) = a :=
by sorry

end NUMINAMATH_GPT_simplest_form_eq_a_l1681_168107


namespace NUMINAMATH_GPT_part1_part2_l1681_168115

noncomputable section

def f (x : ℝ) (a : ℝ) : ℝ := abs (x + 2 * a)

theorem part1 (a : ℝ) :
  (∀ x : ℝ, -4 < x ∧ x < 4 ↔ f x a < 4 - 2 * a) →
  a = 0 := 
sorry

theorem part2 (m : ℝ) :
  (∀ x : ℝ, f x 1 - f (-2 * x) 1 ≤ x + m) →
  2 ≤ m :=
sorry

end NUMINAMATH_GPT_part1_part2_l1681_168115


namespace NUMINAMATH_GPT_leak_empty_tank_time_l1681_168191

-- Definitions based on given conditions
def rate_A := 1 / 2 -- Rate of Pipe A (1 tank per 2 hours)
def rate_A_plus_L := 2 / 5 -- Combined rate of Pipe A and leak

-- Theorem states the time leak takes to empty full tank is 10 hours
theorem leak_empty_tank_time : 1 / (rate_A - rate_A_plus_L) = 10 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_leak_empty_tank_time_l1681_168191


namespace NUMINAMATH_GPT_no_graph_for_equation_l1681_168168

theorem no_graph_for_equation (x y : ℝ) : 
  ¬ ∃ (x y : ℝ), x^2 + y^2 + 2*x + 4*y + 6 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_no_graph_for_equation_l1681_168168


namespace NUMINAMATH_GPT_number_of_boxes_l1681_168104

theorem number_of_boxes (eggs_per_box : ℕ) (total_eggs : ℕ) (h1 : eggs_per_box = 3) (h2 : total_eggs = 6) : total_eggs / eggs_per_box = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_boxes_l1681_168104


namespace NUMINAMATH_GPT_sara_grew_4_onions_l1681_168102

def onions_grown_by_sally : Nat := 5
def onions_grown_by_fred : Nat := 9
def total_onions_grown : Nat := 18

def onions_grown_by_sara : Nat :=
  total_onions_grown - (onions_grown_by_sally + onions_grown_by_fred)

theorem sara_grew_4_onions :
  onions_grown_by_sara = 4 :=
by
  sorry

end NUMINAMATH_GPT_sara_grew_4_onions_l1681_168102


namespace NUMINAMATH_GPT_solve_congruence_l1681_168128

theorem solve_congruence (n : ℕ) (h₀ : 0 ≤ n ∧ n < 47) (h₁ : 13 * n ≡ 5 [MOD 47]) :
  n = 4 :=
sorry

end NUMINAMATH_GPT_solve_congruence_l1681_168128


namespace NUMINAMATH_GPT_trig_identity_l1681_168122

open Real

theorem trig_identity (theta : ℝ) (h : tan theta = 2) : 
  (sin (π / 2 + theta) - cos (π - theta)) / (sin (π / 2 - theta) - sin (π - theta)) = -2 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1681_168122


namespace NUMINAMATH_GPT_find_x_l1681_168151

  -- Definition of the vectors
  def a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
  def b : ℝ × ℝ := (2, 1)

  -- Condition that vectors are parallel
  def are_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

  -- Theorem statement
  theorem find_x (x : ℝ) (h : are_parallel (a x) b) : x = 5 :=
  sorry
  
end NUMINAMATH_GPT_find_x_l1681_168151


namespace NUMINAMATH_GPT_total_charge_for_2_hours_l1681_168162

theorem total_charge_for_2_hours (F A : ℕ) 
  (h1 : F = A + 40) 
  (h2 : F + 4 * A = 375) : 
  F + A = 174 :=
by 
  sorry

end NUMINAMATH_GPT_total_charge_for_2_hours_l1681_168162


namespace NUMINAMATH_GPT_correct_system_of_equations_l1681_168103

theorem correct_system_of_equations (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔ 
  (y = 7 * x + 7) ∧ (y = 9 * (x - 1)) :=
by
  sorry

end NUMINAMATH_GPT_correct_system_of_equations_l1681_168103


namespace NUMINAMATH_GPT_smallest_number_is_33_l1681_168172

theorem smallest_number_is_33 
  (x : ℕ) 
  (h1 : ∀ y z, y = 2 * x → z = 4 * x → (x + y + z) / 3 = 77) : 
  x = 33 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_is_33_l1681_168172


namespace NUMINAMATH_GPT_instantaneous_rate_of_change_at_0_l1681_168121

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp (Real.sin x)

theorem instantaneous_rate_of_change_at_0 : (deriv f 0) = 2 :=
  by
  sorry

end NUMINAMATH_GPT_instantaneous_rate_of_change_at_0_l1681_168121


namespace NUMINAMATH_GPT_greatest_t_solution_l1681_168164

theorem greatest_t_solution :
  ∀ t : ℝ, t ≠ 8 ∧ t ≠ -5 →
  (t^2 - t - 40) / (t - 8) = 5 / (t + 5) →
  t ≤ -2 :=
by
  sorry

end NUMINAMATH_GPT_greatest_t_solution_l1681_168164


namespace NUMINAMATH_GPT_ellipse_equation_l1681_168149

theorem ellipse_equation (b : Real) (c : Real)
  (h₁ : 0 < b ∧ b < 5) 
  (h₂ : 25 - b^2 = c^2)
  (h₃ : 5 + c = 2 * b) :
  ∃ (b : Real), (b^2 = 16) ∧ (∀ x y : Real, (x^2 / 25 + y^2 / b^2 = 1 ↔ x^2 / 25 + y^2 / 16 = 1)) := 
sorry

end NUMINAMATH_GPT_ellipse_equation_l1681_168149


namespace NUMINAMATH_GPT_kona_distance_proof_l1681_168119

-- Defining the distances as constants
def distance_to_bakery : ℕ := 9
def distance_from_grandmother_to_home : ℕ := 27
def additional_trip_distance : ℕ := 6

-- Defining the variable for the distance from bakery to grandmother's house
def x : ℕ := 30

-- Main theorem to prove the distance
theorem kona_distance_proof :
  distance_to_bakery + x + distance_from_grandmother_to_home = 2 * x + additional_trip_distance :=
by
  sorry

end NUMINAMATH_GPT_kona_distance_proof_l1681_168119


namespace NUMINAMATH_GPT_incorrect_conclusion_intersection_l1681_168117

theorem incorrect_conclusion_intersection :
  ∀ (x : ℝ), (0 = -2 * x + 4) → (x = 2) :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_incorrect_conclusion_intersection_l1681_168117


namespace NUMINAMATH_GPT_negative_movement_south_l1681_168134

noncomputable def movement_interpretation (x : ℤ) : String :=
if x > 0 then 
  "moving " ++ toString x ++ "m north"
else 
  "moving " ++ toString (-x) ++ "m south"

theorem negative_movement_south : movement_interpretation (-50) = "moving 50m south" := 
by 
  sorry

end NUMINAMATH_GPT_negative_movement_south_l1681_168134


namespace NUMINAMATH_GPT_tan_neg4095_eq_one_l1681_168147

theorem tan_neg4095_eq_one : Real.tan (Real.pi / 180 * -4095) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_neg4095_eq_one_l1681_168147


namespace NUMINAMATH_GPT_min_value_of_expression_l1681_168196

theorem min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 
  x^2 + 4 * y^2 + 2 * x * y ≥ 3 / 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1681_168196


namespace NUMINAMATH_GPT_big_al_bananas_l1681_168116

-- Define conditions for the arithmetic sequence and total consumption
theorem big_al_bananas (a : ℕ) : 
  (a + (a + 6) + (a + 12) + (a + 18) + (a + 24) = 100) → 
  (a + 24 = 32) :=
by
  sorry

end NUMINAMATH_GPT_big_al_bananas_l1681_168116


namespace NUMINAMATH_GPT_certain_fraction_ratio_l1681_168133

theorem certain_fraction_ratio :
  ∃ x : ℚ,
    (2 / 5 : ℚ) / x = (0.46666666666666673 : ℚ) / (1 / 2) ∧ x = 3 / 7 :=
by sorry

end NUMINAMATH_GPT_certain_fraction_ratio_l1681_168133


namespace NUMINAMATH_GPT_remainder_of_3_pow_100_plus_5_mod_8_l1681_168161

theorem remainder_of_3_pow_100_plus_5_mod_8 :
  (3^100 + 5) % 8 = 6 := by
sorry

end NUMINAMATH_GPT_remainder_of_3_pow_100_plus_5_mod_8_l1681_168161


namespace NUMINAMATH_GPT_daily_savings_in_dollars_l1681_168165

-- Define the total savings and the number of days
def total_savings_in_dimes : ℕ := 3
def number_of_days : ℕ := 30

-- Define the conversion factor from dimes to dollars
def dime_to_dollar : ℝ := 0.10

-- Prove that the daily savings in dollars is $0.01
theorem daily_savings_in_dollars : total_savings_in_dimes / number_of_days * dime_to_dollar = 0.01 :=
by sorry

end NUMINAMATH_GPT_daily_savings_in_dollars_l1681_168165


namespace NUMINAMATH_GPT_third_jumper_height_l1681_168188

/-- 
  Ravi can jump 39 inches high.
  Ravi can jump 1.5 times higher than the average height of three other jumpers.
  The three jumpers can jump 23 inches, 27 inches, and some unknown height x.
  Prove that the unknown height x is 28 inches.
-/
theorem third_jumper_height (x : ℝ) (h₁ : 39 = 1.5 * (23 + 27 + x) / 3) : 
  x = 28 :=
sorry

end NUMINAMATH_GPT_third_jumper_height_l1681_168188


namespace NUMINAMATH_GPT_total_people_count_l1681_168136

-- Definitions based on given conditions
def Cannoneers : ℕ := 63
def Women : ℕ := 2 * Cannoneers
def Men : ℕ := 2 * Women
def TotalPeople : ℕ := Women + Men

-- Lean statement to prove
theorem total_people_count : TotalPeople = 378 := by
  -- placeholders for proof steps
  sorry

end NUMINAMATH_GPT_total_people_count_l1681_168136


namespace NUMINAMATH_GPT_geometric_series_sum_l1681_168144

theorem geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 4
  let n := 7
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 16383 / 49152 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1681_168144


namespace NUMINAMATH_GPT_integral_3x_plus_sin_x_l1681_168167

theorem integral_3x_plus_sin_x :
  ∫ x in (0 : ℝ)..(π / 2), (3 * x + Real.sin x) = (3 / 8) * π^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_integral_3x_plus_sin_x_l1681_168167


namespace NUMINAMATH_GPT_custom_mul_of_two_and_neg_three_l1681_168143

-- Define the custom operation "*"
def custom.mul (a b : Int) : Int := a * b

-- The theorem to prove that 2 * (-3) using custom.mul equals -6
theorem custom_mul_of_two_and_neg_three : custom.mul 2 (-3) = -6 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_custom_mul_of_two_and_neg_three_l1681_168143


namespace NUMINAMATH_GPT_range_of_k_l1681_168198

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  ((k+1)*x^2 + (k+3)*x + (2*k-8)) / ((2*k-1)*x^2 + (k+1)*x + (k-4))

theorem range_of_k 
  (k : ℝ) 
  (hk1 : k ≠ -1)
  (hk2 : (k+3)^2 - 4*(k+1)*(2*k-8) ≥ 0)
  (hk3 : (k+1)^2 - 4*(2*k-1)*(k-4) ≤ 0)
  (hk4 : (k+1)/(2*k-1) > 0) :
  k ∈ Set.Iio (-1) ∪ Set.Ioi (1 / 2) ∩ Set.Iic (41 / 7) := 
  sorry

end NUMINAMATH_GPT_range_of_k_l1681_168198


namespace NUMINAMATH_GPT_jerry_won_games_l1681_168194

theorem jerry_won_games 
  (T : ℕ) (K D J : ℕ) 
  (h1 : T = 32) 
  (h2 : K = D + 5) 
  (h3 : D = J + 3) : 
  J = 7 := 
sorry

end NUMINAMATH_GPT_jerry_won_games_l1681_168194


namespace NUMINAMATH_GPT_cookies_per_person_l1681_168163

theorem cookies_per_person (total_cookies : ℕ) (num_people : ℕ) (h1 : total_cookies = 35) (h2 : num_people = 5) :
  total_cookies / num_people = 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_cookies_per_person_l1681_168163


namespace NUMINAMATH_GPT_bars_cannot_form_triangle_l1681_168100

theorem bars_cannot_form_triangle 
  (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 10) : 
  ¬(a + b > c ∧ a + c > b ∧ b + c > a) :=
by 
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_bars_cannot_form_triangle_l1681_168100


namespace NUMINAMATH_GPT_weight_of_bowling_ball_l1681_168159

-- Define weights of bowling ball and canoe
variable (b c : ℚ)

-- Problem conditions
def cond1 : Prop := (9 * b = 5 * c)
def cond2 : Prop := (4 * c = 120)

-- The statement to prove
theorem weight_of_bowling_ball (h1 : cond1 b c) (h2 : cond2 c) : b = 50 / 3 := sorry

end NUMINAMATH_GPT_weight_of_bowling_ball_l1681_168159


namespace NUMINAMATH_GPT_local_min_4_l1681_168145

def seq (n : ℕ) : ℝ := n^3 - 48 * n + 5

theorem local_min_4 (m : ℕ) (h1 : seq (m-1) > seq m) (h2 : seq (m+1) > seq m) : m = 4 :=
sorry

end NUMINAMATH_GPT_local_min_4_l1681_168145


namespace NUMINAMATH_GPT_daily_expenditure_l1681_168185

theorem daily_expenditure (total_spent : ℕ) (days_in_june : ℕ) (equal_consumption : Prop) :
  total_spent = 372 ∧ days_in_june = 30 ∧ equal_consumption → (372 / 30) = 12.40 := by
  sorry

end NUMINAMATH_GPT_daily_expenditure_l1681_168185


namespace NUMINAMATH_GPT_non_working_games_count_l1681_168101

-- Definitions based on conditions
def total_games : Nat := 15
def total_earnings : Nat := 30
def price_per_game : Nat := 5

-- Definition to be proved
def working_games : Nat := total_earnings / price_per_game
def non_working_games : Nat := total_games - working_games

-- Statement to be proved
theorem non_working_games_count : non_working_games = 9 :=
by
  sorry

end NUMINAMATH_GPT_non_working_games_count_l1681_168101


namespace NUMINAMATH_GPT_no_x_satisfies_inequalities_l1681_168193

theorem no_x_satisfies_inequalities : ¬ ∃ x : ℝ, 4 * x - 2 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 9 * x - 5 :=
sorry

end NUMINAMATH_GPT_no_x_satisfies_inequalities_l1681_168193


namespace NUMINAMATH_GPT_melanie_total_plums_l1681_168153

-- Define the initial conditions
def melaniePlums : Float := 7.0
def samGavePlums : Float := 3.0

-- State the theorem to prove
theorem melanie_total_plums : melaniePlums + samGavePlums = 10.0 := 
by
  sorry

end NUMINAMATH_GPT_melanie_total_plums_l1681_168153


namespace NUMINAMATH_GPT_cos4_minus_sin4_15_eq_sqrt3_div2_l1681_168166

theorem cos4_minus_sin4_15_eq_sqrt3_div2 :
  (Real.cos 15)^4 - (Real.sin 15)^4 = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_cos4_minus_sin4_15_eq_sqrt3_div2_l1681_168166


namespace NUMINAMATH_GPT_birth_year_l1681_168123

theorem birth_year (x : ℤ) (h : 1850 < x^2 - 10 - x ∧ 1849 ≤ x^2 - 10 - x ∧ x^2 - 10 - x ≤ 1880) : 
x^2 - 10 - x ≠ 1849 ∧ x^2 - 10 - x ≠ 1855 ∧ x^2 - 10 - x ≠ 1862 ∧ x^2 - 10 - x ≠ 1871 ∧ x^2 - 10 - x ≠ 1880 := 
sorry

end NUMINAMATH_GPT_birth_year_l1681_168123


namespace NUMINAMATH_GPT_M_gt_N_l1681_168195

theorem M_gt_N (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) :
  let M := a * b
  let N := a + b - 1
  M > N := by
  sorry

end NUMINAMATH_GPT_M_gt_N_l1681_168195


namespace NUMINAMATH_GPT_simplify_expression_l1681_168187

theorem simplify_expression :
  8 * (18 / 5) * (-40 / 27) = - (128 / 3) := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1681_168187


namespace NUMINAMATH_GPT_log_ab_a2_plus_log_ab_b2_eq_2_l1681_168108

theorem log_ab_a2_plus_log_ab_b2_eq_2 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h_distinct : a ≠ b) (h_a_gt_2 : a > 2) (h_b_gt_2 : b > 2) :
  Real.log (a^2) / Real.log (a * b) + Real.log (b^2) / Real.log (a * b) = 2 :=
by
  sorry

end NUMINAMATH_GPT_log_ab_a2_plus_log_ab_b2_eq_2_l1681_168108


namespace NUMINAMATH_GPT_dan_total_marbles_l1681_168142

theorem dan_total_marbles (violet_marbles : ℕ) (red_marbles : ℕ) (h₁ : violet_marbles = 64) (h₂ : red_marbles = 14) : violet_marbles + red_marbles = 78 :=
sorry

end NUMINAMATH_GPT_dan_total_marbles_l1681_168142


namespace NUMINAMATH_GPT_lead_amount_in_mixture_l1681_168140

theorem lead_amount_in_mixture 
  (W : ℝ) 
  (h_copper : 0.60 * W = 12) 
  (h_mixture_composition : (0.15 * W = 0.15 * W) ∧ (0.25 * W = 0.25 * W) ∧ (0.60 * W = 0.60 * W)) :
  (0.25 * W = 5) :=
by
  sorry

end NUMINAMATH_GPT_lead_amount_in_mixture_l1681_168140


namespace NUMINAMATH_GPT_isosceles_triangle_base_angle_l1681_168181

theorem isosceles_triangle_base_angle (vertex_angle : ℝ) (base_angle : ℝ) 
  (h1 : vertex_angle = 60) 
  (h2 : 2 * base_angle + vertex_angle = 180) : 
  base_angle = 60 := 
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angle_l1681_168181


namespace NUMINAMATH_GPT_gcd_78_182_l1681_168126

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := 
by
  sorry

end NUMINAMATH_GPT_gcd_78_182_l1681_168126


namespace NUMINAMATH_GPT_triangle_PQR_area_l1681_168190

-- Define the points P, Q, and R
def P : (ℝ × ℝ) := (-2, 2)
def Q : (ℝ × ℝ) := (8, 2)
def R : (ℝ × ℝ) := (4, -4)

-- Define a function to calculate the area of triangle
def triangle_area (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Lean statement to prove the area of triangle PQR is 30 square units
theorem triangle_PQR_area : triangle_area P Q R = 30 := by
  sorry

end NUMINAMATH_GPT_triangle_PQR_area_l1681_168190


namespace NUMINAMATH_GPT_jacket_purchase_price_l1681_168110

theorem jacket_purchase_price (P S SP : ℝ)
  (h1 : S = P + 0.40 * S)
  (h2 : SP = 0.80 * S)
  (h3 : SP - P = 18) :
  P = 54 :=
by
  sorry

end NUMINAMATH_GPT_jacket_purchase_price_l1681_168110


namespace NUMINAMATH_GPT_largest_sphere_radius_l1681_168173

noncomputable def torus_inner_radius := 3
noncomputable def torus_outer_radius := 5
noncomputable def torus_center_circle := (4, 0, 1)
noncomputable def torus_radius := 1
noncomputable def torus_table_plane := 0

theorem largest_sphere_radius :
  ∀ (r : ℝ), 
  ∀ (O P : ℝ × ℝ × ℝ), 
  (P = (4, 0, 1)) → 
  (O = (0, 0, r)) → 
  4^2 + (r - 1)^2 = (r + 1)^2 → 
  r = 4 := 
by
  intros
  sorry

end NUMINAMATH_GPT_largest_sphere_radius_l1681_168173


namespace NUMINAMATH_GPT_probability_at_least_one_boy_and_one_girl_l1681_168150

noncomputable def mathematics_club_prob : ℚ :=
  let boys := 14
  let girls := 10
  let total_members := 24
  let total_committees := Nat.choose total_members 5
  let boys_committees := Nat.choose boys 5
  let girls_committees := Nat.choose girls 5
  let committees_with_at_least_one_boy_and_one_girl := total_committees - (boys_committees + girls_committees)
  let probability := (committees_with_at_least_one_boy_and_one_girl : ℚ) / (total_committees : ℚ)
  probability

theorem probability_at_least_one_boy_and_one_girl :
  mathematics_club_prob = (4025 : ℚ) / 4251 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_boy_and_one_girl_l1681_168150


namespace NUMINAMATH_GPT_correct_sentence_completion_l1681_168112

-- Define the possible options
inductive Options
| A : Options  -- "However he was reminded frequently"
| B : Options  -- "No matter he was reminded frequently"
| C : Options  -- "However frequently he was reminded"
| D : Options  -- "No matter he was frequently reminded"

-- Define the correctness condition
def correct_option : Options := Options.C

-- Define the proof problem
theorem correct_sentence_completion (opt : Options) : opt = correct_option :=
by sorry

end NUMINAMATH_GPT_correct_sentence_completion_l1681_168112


namespace NUMINAMATH_GPT_fraction_white_tulips_l1681_168131

theorem fraction_white_tulips : 
  ∀ (total_tulips yellow_fraction red_fraction pink_fraction white_fraction : ℝ),
  total_tulips = 60 →
  yellow_fraction = 1 / 2 →
  red_fraction = 1 / 3 →
  pink_fraction = 1 / 4 →
  white_fraction = 
    ((total_tulips * (1 - yellow_fraction)) * (1 - red_fraction) * (1 - pink_fraction)) / total_tulips →
  white_fraction = 1 / 4 :=
by
  intros total_tulips yellow_fraction red_fraction pink_fraction white_fraction 
    h_total h_yellow h_red h_pink h_white
  sorry

end NUMINAMATH_GPT_fraction_white_tulips_l1681_168131


namespace NUMINAMATH_GPT_sum_of_squares_l1681_168197

-- Define the proposition as a universal statement 
theorem sum_of_squares (a b : ℝ) : a^2 + b^2 + 2 * a * b = (a + b)^2 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1681_168197


namespace NUMINAMATH_GPT_prime_divides_factorial_plus_one_non_prime_not_divides_factorial_plus_one_factorial_mod_non_prime_is_zero_l1681_168178

-- Show that if \( p \) is a prime number, then \( p \) divides \( (p-1)! + 1 \).
theorem prime_divides_factorial_plus_one (p : ℕ) (hp : Nat.Prime p) : p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

-- Show that if \( n \) is not a prime number, then \( n \) does not divide \( (n-1)! + 1 \).
theorem non_prime_not_divides_factorial_plus_one (n : ℕ) (hn : ¬Nat.Prime n) : ¬(n ∣ (Nat.factorial (n - 1) + 1)) :=
sorry

-- Calculate the remainder of the division of \((n-1)!\) by \( n \).
theorem factorial_mod_non_prime_is_zero (n : ℕ) (hn : ¬Nat.Prime n) : (Nat.factorial (n - 1)) % n = 0 :=
sorry

end NUMINAMATH_GPT_prime_divides_factorial_plus_one_non_prime_not_divides_factorial_plus_one_factorial_mod_non_prime_is_zero_l1681_168178


namespace NUMINAMATH_GPT_pencil_eraser_cost_l1681_168124

variable (p e : ℕ)

theorem pencil_eraser_cost
  (h1 : 15 * p + 5 * e = 125)
  (h2 : p > e)
  (h3 : p > 0)
  (h4 : e > 0) :
  p + e = 11 :=
sorry

end NUMINAMATH_GPT_pencil_eraser_cost_l1681_168124


namespace NUMINAMATH_GPT_sum_of_coordinates_l1681_168138

-- Definitions of points and their coordinates
def pointC (x : ℝ) : ℝ × ℝ := (x, 8)
def pointD (x : ℝ) : ℝ × ℝ := (x, -8)

-- The goal is to prove that the sum of the four coordinate values of points C and D is 2x
theorem sum_of_coordinates (x : ℝ) :
  (pointC x).1 + (pointC x).2 + (pointD x).1 + (pointD x).2 = 2 * x :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_l1681_168138


namespace NUMINAMATH_GPT_sin_double_angle_second_quadrant_l1681_168186

theorem sin_double_angle_second_quadrant (α : ℝ) (h1 : Real.cos α = -3/5) (h2 : α ∈ Set.Ioo (π / 2) π) :
    Real.sin (2 * α) = -24 / 25 := by
  sorry

end NUMINAMATH_GPT_sin_double_angle_second_quadrant_l1681_168186


namespace NUMINAMATH_GPT_find_m_l1681_168170

variables (x m : ℝ)

def equation (x m : ℝ) : Prop := 3 * x - 2 * m = 4

theorem find_m (h1 : equation 6 m) : m = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1681_168170


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1681_168139

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 4 → a^2 > 16) ∧ (∃ a, (a < -4) ∧ (a^2 > 16)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1681_168139


namespace NUMINAMATH_GPT_probability_of_collinear_dots_in_5x5_grid_l1681_168118

def collinear_dots_probability (total_dots chosen_dots collinear_sets : ℕ) : ℚ :=
  (collinear_sets : ℚ) / (Nat.choose total_dots chosen_dots)

theorem probability_of_collinear_dots_in_5x5_grid :
  collinear_dots_probability 25 4 12 = 12 / 12650 := by
  sorry

end NUMINAMATH_GPT_probability_of_collinear_dots_in_5x5_grid_l1681_168118


namespace NUMINAMATH_GPT_sum_of_numbers_l1681_168189

theorem sum_of_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 8)
  (h4 : (a + b + c) / 3 = a + 12) (h5 : (a + b + c) / 3 = c - 20) :
  a + b + c = 48 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l1681_168189


namespace NUMINAMATH_GPT_min_a2_plus_b2_l1681_168174

-- Define circle and line intercept conditions
def circle_center : ℝ × ℝ := (-2, 1)
def circle_radius : ℝ := 2
def line_eq (a b x y : ℝ) : Prop := a * x + 2 * b * y - 4 = 0
def chord_length (chord_len : ℝ) : Prop := chord_len = 4

-- Define the final minimum value to prove
def min_value (a b : ℝ) : ℝ := a^2 + b^2

-- Proving the specific value considering the conditions
theorem min_a2_plus_b2 (a b : ℝ) (h1 : b = a + 2) (h2 : chord_length 4) : min_value a b = 2 := by
  sorry

end NUMINAMATH_GPT_min_a2_plus_b2_l1681_168174


namespace NUMINAMATH_GPT_fraction_is_one_fourth_l1681_168105

theorem fraction_is_one_fourth
  (f : ℚ)
  (m : ℕ)
  (h1 : (1 / 5) ^ m * f^2 = 1 / (10 ^ 4))
  (h2 : m = 4) : f = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_fraction_is_one_fourth_l1681_168105


namespace NUMINAMATH_GPT_martha_savings_l1681_168137

-- Definitions based on conditions
def weekly_latte_spending : ℝ := 4.00 * 5
def weekly_iced_coffee_spending : ℝ := 2.00 * 3
def total_weekly_coffee_spending : ℝ := weekly_latte_spending + weekly_iced_coffee_spending
def annual_coffee_spending : ℝ := total_weekly_coffee_spending * 52
def savings_percentage : ℝ := 0.25

-- The theorem to be proven
theorem martha_savings : annual_coffee_spending * savings_percentage = 338.00 := by
  sorry

end NUMINAMATH_GPT_martha_savings_l1681_168137


namespace NUMINAMATH_GPT_organizingCommitteeWays_l1681_168120

-- Define the problem context
def numberOfTeams : Nat := 5
def membersPerTeam : Nat := 8
def hostTeamSelection : Nat := 4
def otherTeamsSelection : Nat := 2

-- Define binomial coefficient
def binom (n k : Nat) : Nat := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the number of ways to select committee members
def totalCommitteeWays : Nat := numberOfTeams * 
                                 (binom membersPerTeam hostTeamSelection) * 
                                 ((binom membersPerTeam otherTeamsSelection) ^ (numberOfTeams - 1))

-- The theorem to prove
theorem organizingCommitteeWays : 
  totalCommitteeWays = 215134600 := 
    sorry

end NUMINAMATH_GPT_organizingCommitteeWays_l1681_168120


namespace NUMINAMATH_GPT_quarters_spent_l1681_168177

theorem quarters_spent (original : ℕ) (remaining : ℕ) (q : ℕ) 
  (h1 : original = 760) 
  (h2 : remaining = 342) 
  (h3 : q = original - remaining) : q = 418 := 
by
  sorry

end NUMINAMATH_GPT_quarters_spent_l1681_168177


namespace NUMINAMATH_GPT_find_value_simplify_expression_l1681_168109

-- Define the first part of the problem
theorem find_value (α : ℝ) (h : Real.tan α = 1/3) : 
  (1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2)) = 2 / 3 := 
  sorry

-- Define the second part of the problem
theorem simplify_expression (α : ℝ) (h : Real.tan α = 1/3) : 
  (Real.tan (π - α) * Real.cos (2 * π - α) * Real.sin (-α + 3 * π / 2)) / (Real.cos (-α - π) * Real.sin (-π - α)) = -1 := 
  sorry

end NUMINAMATH_GPT_find_value_simplify_expression_l1681_168109


namespace NUMINAMATH_GPT_alligators_hiding_correct_l1681_168114

def total_alligators := 75
def not_hiding_alligators := 56

def hiding_alligators (total not_hiding : Nat) : Nat :=
  total - not_hiding

theorem alligators_hiding_correct : hiding_alligators total_alligators not_hiding_alligators = 19 := 
by
  sorry

end NUMINAMATH_GPT_alligators_hiding_correct_l1681_168114


namespace NUMINAMATH_GPT_least_number_remainder_l1681_168183

theorem least_number_remainder (n : ℕ) (h : 20 ∣ (n - 5)) : n = 125 := sorry

end NUMINAMATH_GPT_least_number_remainder_l1681_168183


namespace NUMINAMATH_GPT_bananas_in_each_bunch_l1681_168157

theorem bananas_in_each_bunch (x: ℕ) : (6 * x + 5 * 7 = 83) → x = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_bananas_in_each_bunch_l1681_168157


namespace NUMINAMATH_GPT_rectangle_width_l1681_168176

theorem rectangle_width (P l: ℕ) (hP : P = 50) (hl : l = 13) : 
  ∃ w : ℕ, 2 * l + 2 * w = P ∧ w = 12 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_width_l1681_168176


namespace NUMINAMATH_GPT_james_nickels_count_l1681_168132

-- Definitions
def total_cents : ℕ := 685
def more_nickels_than_quarters := 11

-- Variables representing the number of nickels and quarters
variables (n q : ℕ)

-- Conditions
axiom h1 : 5 * n + 25 * q = total_cents
axiom h2 : n = q + more_nickels_than_quarters

-- Theorem stating the number of nickels
theorem james_nickels_count : n = 32 := 
by
  -- Proof will go here, marked as "sorry" to complete the statement
  sorry

end NUMINAMATH_GPT_james_nickels_count_l1681_168132


namespace NUMINAMATH_GPT_factorize_xcube_minus_x_l1681_168152

theorem factorize_xcube_minus_x (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by 
  sorry

end NUMINAMATH_GPT_factorize_xcube_minus_x_l1681_168152


namespace NUMINAMATH_GPT_days_to_complete_work_l1681_168175

theorem days_to_complete_work {D : ℝ} (h1 : D > 0)
  (h2 : (1 / D) + (2 / D) = 0.3) :
  D = 10 :=
sorry

end NUMINAMATH_GPT_days_to_complete_work_l1681_168175


namespace NUMINAMATH_GPT_x_intercept_of_line_l1681_168141

def point1 := (10, 3)
def point2 := (-12, -8)

theorem x_intercept_of_line :
  let m := (point2.snd - point1.snd) / (point2.fst - point1.fst)
  let line_eq (x : ℝ) := m * (x - point1.fst) + point1.snd
  ∃ x : ℝ, line_eq x = 0 ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_x_intercept_of_line_l1681_168141


namespace NUMINAMATH_GPT_find_initial_maple_trees_l1681_168158

def initial_maple_trees (final_maple_trees planted_maple_trees : ℕ) : ℕ :=
  final_maple_trees - planted_maple_trees

theorem find_initial_maple_trees : initial_maple_trees 11 9 = 2 := by
  sorry

end NUMINAMATH_GPT_find_initial_maple_trees_l1681_168158


namespace NUMINAMATH_GPT_determine_m_for_divisibility_by_11_l1681_168146

def is_divisible_by_11 (n : ℤ) : Prop :=
  n % 11 = 0

def sum_digits_odd_pos : ℤ :=
  8 + 6 + 2 + 8

def sum_digits_even_pos (m : ℤ) : ℤ :=
  5 + m + 4

theorem determine_m_for_divisibility_by_11 :
  ∃ m : ℤ, is_divisible_by_11 (sum_digits_odd_pos - sum_digits_even_pos m) ∧ m = 4 := 
by
  sorry

end NUMINAMATH_GPT_determine_m_for_divisibility_by_11_l1681_168146


namespace NUMINAMATH_GPT_purely_imaginary_l1681_168171

theorem purely_imaginary {m : ℝ} (h1 : m^2 - 3 * m = 0) (h2 : m^2 - 5 * m + 6 ≠ 0) : m = 0 :=
sorry

end NUMINAMATH_GPT_purely_imaginary_l1681_168171


namespace NUMINAMATH_GPT_smallest_integer_l1681_168180

theorem smallest_integer (n : ℕ) (h : n > 0) (h1 : lcm 36 n / gcd 36 n = 24) : n = 96 :=
sorry

end NUMINAMATH_GPT_smallest_integer_l1681_168180


namespace NUMINAMATH_GPT_max_third_side_length_l1681_168156

theorem max_third_side_length (x : ℕ) (h1 : 28 + x > 47) (h2 : 47 + x > 28) (h3 : 28 + 47 > x) :
  x = 74 :=
sorry

end NUMINAMATH_GPT_max_third_side_length_l1681_168156


namespace NUMINAMATH_GPT_b_finishes_remaining_work_in_5_days_l1681_168127

theorem b_finishes_remaining_work_in_5_days :
  let A_work_rate := 1 / 4
  let B_work_rate := 1 / 14
  let combined_work_rate := A_work_rate + B_work_rate
  let work_completed_together := 2 * combined_work_rate
  let work_remaining := 1 - work_completed_together
  let days_b_to_finish := work_remaining / B_work_rate
  days_b_to_finish = 5 :=
by
  let A_work_rate := 1 / 4
  let B_work_rate := 1 / 14
  let combined_work_rate := A_work_rate + B_work_rate
  let work_completed_together := 2 * combined_work_rate
  let work_remaining := 1 - work_completed_together
  let days_b_to_finish := work_remaining / B_work_rate
  show days_b_to_finish = 5
  sorry

end NUMINAMATH_GPT_b_finishes_remaining_work_in_5_days_l1681_168127


namespace NUMINAMATH_GPT_cos_60_eq_half_l1681_168169

theorem cos_60_eq_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_60_eq_half_l1681_168169


namespace NUMINAMATH_GPT_proof_F_4_f_5_l1681_168135

def f (a : ℤ) : ℤ := a - 2

def F (a b : ℤ) : ℤ := a * b + b^2

theorem proof_F_4_f_5 :
  F 4 (f 5) = 21 := by
  sorry

end NUMINAMATH_GPT_proof_F_4_f_5_l1681_168135


namespace NUMINAMATH_GPT_line_equation_l1681_168182

theorem line_equation {L : ℝ → ℝ → Prop} (h1 : L (-3) (-2)) 
  (h2 : ∃ a : ℝ, a ≠ 0 ∧ (L a 0 ∧ L 0 a)) :
  (∀ x y, L x y ↔ 2 * x - 3 * y = 0) ∨ (∀ x y, L x y ↔ x + y + 5 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_line_equation_l1681_168182


namespace NUMINAMATH_GPT_more_red_peaches_than_green_l1681_168106

-- Given conditions
def red_peaches : Nat := 17
def green_peaches : Nat := 16

-- Statement to prove
theorem more_red_peaches_than_green : red_peaches - green_peaches = 1 :=
by
  sorry

end NUMINAMATH_GPT_more_red_peaches_than_green_l1681_168106


namespace NUMINAMATH_GPT_distance_from_Q_to_EF_is_24_div_5_l1681_168113

-- Define the configuration of the square and points
def E := (0, 8)
def F := (8, 8)
def G := (8, 0)
def H := (0, 0)
def N := (4, 0) -- Midpoint of GH
def r1 := 4 -- Radius of the circle centered at N
def r2 := 8 -- Radius of the circle centered at E

-- Definition of the first circle centered at N with radius r1
def circle1 (x y : ℝ) := (x - 4)^2 + y^2 = r1^2

-- Definition of the second circle centered at E with radius r2
def circle2 (x y : ℝ) := x^2 + (y - 8)^2 = r2^2

-- Define the intersection point Q, other than H
def Q := (32 / 5, 16 / 5) -- Found as an intersection point between circle1 and circle2

-- Define the distance from point Q to the line EF
def dist_to_EF := 8 - (Q.2) -- (Q.2 is the y-coordinate of Q)

-- The main statement to prove
theorem distance_from_Q_to_EF_is_24_div_5 : dist_to_EF = 24 / 5 := by
  sorry

end NUMINAMATH_GPT_distance_from_Q_to_EF_is_24_div_5_l1681_168113


namespace NUMINAMATH_GPT_prob_divisible_by_5_l1681_168125

theorem prob_divisible_by_5 (M: ℕ) (h1: 100 ≤ M ∧ M < 1000) (h2: M % 10 = 5): 
  (∃ (k: ℕ), M = 5 * k) :=
by
  sorry

end NUMINAMATH_GPT_prob_divisible_by_5_l1681_168125


namespace NUMINAMATH_GPT_range_of_m_l1681_168154

theorem range_of_m (x y m : ℝ) 
  (h1 : x - 2 * y = 1) 
  (h2 : 2 * x + y = 4 * m) 
  (h3 : x + 3 * y < 6) : 
  m < 7 / 4 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1681_168154


namespace NUMINAMATH_GPT_will_remaining_balance_l1681_168148

theorem will_remaining_balance :
  ∀ (initial_money conversion_fee : ℝ) 
    (exchange_rate : ℝ)
    (sweater_cost tshirt_cost shoes_cost hat_cost socks_cost : ℝ)
    (shoes_refund_percentage : ℝ)
    (discount_percentage sales_tax_percentage : ℝ),
  initial_money = 74 →
  conversion_fee = 2 →
  exchange_rate = 1.5 →
  sweater_cost = 13.5 →
  tshirt_cost = 16.5 →
  shoes_cost = 45 →
  hat_cost = 7.5 →
  socks_cost = 6 →
  shoes_refund_percentage = 0.85 →
  discount_percentage = 0.10 →
  sales_tax_percentage = 0.05 →
  (initial_money - conversion_fee) * exchange_rate -
  ((sweater_cost + tshirt_cost + shoes_cost + hat_cost + socks_cost - shoes_cost * shoes_refund_percentage) *
   (1 - discount_percentage) * (1 + sales_tax_percentage)) /
  exchange_rate = 39.87 :=
by
  intros initial_money conversion_fee exchange_rate
        sweater_cost tshirt_cost shoes_cost hat_cost socks_cost
        shoes_refund_percentage discount_percentage sales_tax_percentage
        h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end NUMINAMATH_GPT_will_remaining_balance_l1681_168148


namespace NUMINAMATH_GPT_age_proof_l1681_168130

-- Let's define the conditions first
variable (s f : ℕ) -- s: age of the son, f: age of the father

-- Conditions derived from the problem statement
def son_age_condition : Prop := s = 8 - 1
def father_age_condition : Prop := f = 5 * s

-- The goal is to prove that the father's age is 35
theorem age_proof (s f : ℕ) (h₁ : son_age_condition s) (h₂ : father_age_condition s f) : f = 35 :=
by sorry

end NUMINAMATH_GPT_age_proof_l1681_168130


namespace NUMINAMATH_GPT_parallel_vectors_xy_sum_l1681_168179

theorem parallel_vectors_xy_sum (x y : ℚ) (k : ℚ) 
  (h1 : (2, 4, -5) = (2 * k, 4 * k, -5 * k)) 
  (h2 : (3, x, y) = (2 * k, 4 * k, -5 * k)) 
  (h3 : 3 = 2 * k) : 
  x + y = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_xy_sum_l1681_168179


namespace NUMINAMATH_GPT_range_of_x_l1681_168155

theorem range_of_x (x : ℝ) (h1 : 1 / x < 4) (h2 : 1 / x > -2) : x > -(1 / 2) :=
sorry

end NUMINAMATH_GPT_range_of_x_l1681_168155
