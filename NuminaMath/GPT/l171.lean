import Mathlib

namespace non_neg_solutions_l171_17112

theorem non_neg_solutions (x y z : ℕ) :
  (x^3 = 2 * y^2 - z) →
  (y^3 = 2 * z^2 - x) →
  (z^3 = 2 * x^2 - y) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by {
  sorry
}

end non_neg_solutions_l171_17112


namespace average_salary_correct_l171_17147

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 16000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def number_of_people : ℕ := 5

def average_salary : ℕ := total_salary / number_of_people

theorem average_salary_correct : average_salary = 8800 := by
  sorry

end average_salary_correct_l171_17147


namespace expected_total_rainfall_10_days_l171_17183

theorem expected_total_rainfall_10_days :
  let P_sun := 0.5
  let P_rain3 := 0.3
  let P_rain6 := 0.2
  let daily_rain := (P_sun * 0) + (P_rain3 * 3) + (P_rain6 * 6)
  daily_rain * 10 = 21 :=
by
  sorry

end expected_total_rainfall_10_days_l171_17183


namespace no_triangular_sides_of_specific_a_b_l171_17122

theorem no_triangular_sides_of_specific_a_b (a b c : ℕ) (h1 : a = 10^100 + 1002) (h2 : b = 1001) (h3 : ∃ n : ℕ, c = n^2) : ¬ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by sorry

end no_triangular_sides_of_specific_a_b_l171_17122


namespace polygon_sides_eight_l171_17149

theorem polygon_sides_eight {n : ℕ} (h : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_eight_l171_17149


namespace christopher_age_l171_17123

theorem christopher_age (G C : ℕ) (h1 : C = 2 * G) (h2 : C - 9 = 5 * (G - 9)) : C = 24 := 
by
  sorry

end christopher_age_l171_17123


namespace arithmetic_sequence_sum_20_l171_17116

open BigOperators

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) = a n + (a 1 - a 0)

theorem arithmetic_sequence_sum_20 {a : ℕ → ℤ} (h_arith : is_arithmetic_sequence a)
    (h1 : a 0 + a 1 + a 2 = -24)
    (h18 : a 17 + a 18 + a 19 = 78) :
    ∑ i in Finset.range 20, a i = 180 :=
sorry

end arithmetic_sequence_sum_20_l171_17116


namespace odd_function_behavior_on_interval_l171_17173

theorem odd_function_behavior_on_interval
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 4 → f x₁ < f x₂)
  (h_max : ∀ x, 1 ≤ x → x ≤ 4 → f x ≤ 5) :
  (∀ x, -4 ≤ x → x ≤ -1 → f (-4) ≤ f x ∧ f x ≤ f (-1)) ∧ f (-4) = -5 :=
sorry

end odd_function_behavior_on_interval_l171_17173


namespace ellipse_meets_sine_more_than_8_points_l171_17126

noncomputable def ellipse_intersects_sine_curve_more_than_8_times (a b : ℝ) (h k : ℝ) :=
  ∃ p : ℕ, p > 8 ∧ 
  ∃ (x y : ℝ), 
    (∃ (i : ℕ), y = Real.sin x ∧ 
    (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)

theorem ellipse_meets_sine_more_than_8_points : 
  ∀ (a b h k : ℝ), ellipse_intersects_sine_curve_more_than_8_times a b h k := 
by sorry

end ellipse_meets_sine_more_than_8_points_l171_17126


namespace average_weight_of_remaining_carrots_l171_17111

noncomputable def total_weight_30_carrots : ℕ := 5940
noncomputable def total_weight_3_carrots : ℕ := 540
noncomputable def carrots_count_30 : ℕ := 30
noncomputable def carrots_count_3_removed : ℕ := 3
noncomputable def carrots_count_remaining : ℕ := 27
noncomputable def average_weight_of_removed_carrots : ℕ := 180

theorem average_weight_of_remaining_carrots :
  (total_weight_30_carrots - total_weight_3_carrots) / carrots_count_remaining = 200 :=
  by
  sorry

end average_weight_of_remaining_carrots_l171_17111


namespace derivative_at_zero_l171_17176

noncomputable def f : ℝ → ℝ
| x => if x = 0 then 0 else Real.arcsin (x^2 * Real.cos (1 / (9 * x))) + (2 / 3) * x

theorem derivative_at_zero : HasDerivAt f (2 / 3) 0 := sorry

end derivative_at_zero_l171_17176


namespace find_a_l171_17137

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x + (a - 1) * x

theorem find_a {a : ℝ} : 
  (∀ x : ℝ, 0 < x → f x a ≤ x^2 * Real.exp x - Real.log x - 4 * x - 1) → 
  a ≤ -2 :=
sorry

end find_a_l171_17137


namespace container_marbles_volume_l171_17169

theorem container_marbles_volume {V₁ V₂ m₁ m₂ : ℕ} 
  (h₁ : V₁ = 24) (h₂ : m₁ = 75) (h₃ : V₂ = 72) :
  m₂ = 225 :=
by
  have proportion := (m₁ : ℚ) / V₁
  have proportion2 := (m₂ : ℚ) / V₂
  have h4 := proportion = proportion2
  sorry

end container_marbles_volume_l171_17169


namespace sin_cos_expr1_sin_cos_expr2_l171_17174

variable {x : ℝ}
variable (hx : Real.tan x = 2)

theorem sin_cos_expr1 : (2 / 3) * (Real.sin x)^2 + (1 / 4) * (Real.cos x)^2 = 7 / 12 := by
  sorry

theorem sin_cos_expr2 : 2 * (Real.sin x)^2 - (Real.sin x) * (Real.cos x) + (Real.cos x)^2 = 7 / 5 := by
  sorry

end sin_cos_expr1_sin_cos_expr2_l171_17174


namespace zero_knights_l171_17197

noncomputable def knights_count (n : ℕ) : ℕ := sorry

theorem zero_knights (n : ℕ) (half_lairs : n ≥ 205) :
  knights_count 410 = 0 :=
sorry

end zero_knights_l171_17197


namespace maximum_radius_l171_17142

open Set Real

-- Definitions of sets M, N, and D_r.
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd ≥ 1 / 4 * p.fst^2}

def N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd ≤ -1 / 4 * p.fst^2 + p.fst + 7}

def D_r (x₀ y₀ r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.fst - x₀)^2 + (p.snd - y₀)^2 ≤ r^2}

-- Theorem statement for the largest r
theorem maximum_radius {x₀ y₀ : ℝ} (H : D_r x₀ y₀ r ⊆ M ∩ N) :
  r = sqrt ((25 - 5 * sqrt 5) / 2) :=
sorry

end maximum_radius_l171_17142


namespace negation_prop_l171_17153

theorem negation_prop (p : Prop) : 
  (∀ (x : ℝ), x > 2 → x^2 - 1 > 0) → (¬(∀ (x : ℝ), x > 2 → x^2 - 1 > 0) ↔ (∃ (x : ℝ), x > 2 ∧ x^2 - 1 ≤ 0)) :=
by 
  sorry

end negation_prop_l171_17153


namespace drying_time_short_haired_dog_l171_17102

theorem drying_time_short_haired_dog (x : ℕ) (h1 : ∀ y, y = 2 * x) (h2 : 6 * x + 9 * (2 * x) = 240) : x = 10 :=
by
  sorry

end drying_time_short_haired_dog_l171_17102


namespace z_in_fourth_quadrant_l171_17195

noncomputable def z : ℂ := (3 * Complex.I - 2) / (Complex.I - 1) * Complex.I

theorem z_in_fourth_quadrant : z.re < 0 ∧ z.im > 0 := by
  sorry

end z_in_fourth_quadrant_l171_17195


namespace abs_diff_eq_l171_17172

-- Define the conditions
variables (x y : ℝ)
axiom h1 : x + y = 30
axiom h2 : x * y = 162

-- Define the problem to prove
theorem abs_diff_eq : |x - y| = 6 * Real.sqrt 7 :=
by sorry

end abs_diff_eq_l171_17172


namespace solutionY_materialB_correct_l171_17133

open Real

-- Definitions and conditions from step a
def solutionX_materialA : ℝ := 0.20
def solutionX_materialB : ℝ := 0.80
def solutionY_materialA : ℝ := 0.30
def mixture_materialA : ℝ := 0.22
def solutionX_in_mixture : ℝ := 0.80
def solutionY_in_mixture : ℝ := 0.20

-- The conjecture to prove
theorem solutionY_materialB_correct (B_Y : ℝ) 
  (h1 : solutionX_materialA = 0.20)
  (h2 : solutionX_materialB = 0.80) 
  (h3 : solutionY_materialA = 0.30) 
  (h4 : mixture_materialA = 0.22)
  (h5 : solutionX_in_mixture = 0.80)
  (h6 : solutionY_in_mixture = 0.20) :
  B_Y = 1 - solutionY_materialA := by 
  sorry

end solutionY_materialB_correct_l171_17133


namespace probability_of_not_shorter_than_one_meter_l171_17124

noncomputable def probability_of_event_A : ℝ := 
  let length_of_rope : ℝ := 3
  let event_A_probability : ℝ := 1 / 3
  event_A_probability

theorem probability_of_not_shorter_than_one_meter (l : ℝ) (h_l : l = 3) : 
    probability_of_event_A = 1 / 3 :=
sorry

end probability_of_not_shorter_than_one_meter_l171_17124


namespace percentage_calculation_l171_17141

theorem percentage_calculation :
  ( (2 / 3 * 2432 / 3 + 1 / 6 * 3225) / 450 * 100 ) = 239.54 := 
sorry

end percentage_calculation_l171_17141


namespace max_real_solution_under_100_l171_17154

theorem max_real_solution_under_100 (k a b c r : ℕ) (h0 : ∃ (m n p : ℕ), a = k^m ∧ b = k^n ∧ c = k^p)
  (h1 : r < 100) (h2 : b^2 = 4 * a * c) (h3 : r = b / (2 * a)) : r ≤ 64 :=
sorry

end max_real_solution_under_100_l171_17154


namespace lemonade_lemons_per_glass_l171_17167

def number_of_glasses : ℕ := 9
def total_lemons : ℕ := 18
def lemons_per_glass : ℕ := 2

theorem lemonade_lemons_per_glass :
  total_lemons / number_of_glasses = lemons_per_glass :=
by
  sorry

end lemonade_lemons_per_glass_l171_17167


namespace binom_150_1_l171_17159

-- Definition of the binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_150_1 : binom 150 1 = 150 := by
  -- The proof is skipped and marked as sorry
  sorry

end binom_150_1_l171_17159


namespace cyrus_shots_percentage_l171_17143

theorem cyrus_shots_percentage (total_shots : ℕ) (missed_shots : ℕ) (made_shots : ℕ)
  (h_total : total_shots = 20)
  (h_missed : missed_shots = 4)
  (h_made : made_shots = total_shots - missed_shots) :
  (made_shots / total_shots : ℚ) * 100 = 80 := by
  sorry

end cyrus_shots_percentage_l171_17143


namespace cricket_target_run_l171_17187

theorem cricket_target_run (run_rate1 run_rate2 : ℝ) (overs1 overs2 : ℕ) (T : ℝ) 
  (h1 : run_rate1 = 3.2) (h2 : overs1 = 10) (h3 : run_rate2 = 25) (h4 : overs2 = 10) :
  T = (run_rate1 * overs1) + (run_rate2 * overs2) → T = 282 :=
by
  sorry

end cricket_target_run_l171_17187


namespace tan_angle_addition_l171_17139

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l171_17139


namespace problem_statement_l171_17161

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- defining conditions
axiom a1_4_7 : a 1 + a 4 + a 7 = 39
axiom a2_5_8 : a 2 + a 5 + a 8 = 33
axiom is_arithmetic : arithmetic_seq a d

theorem problem_statement : a 5 + a 8 + a 11 = 15 :=
by sorry

end problem_statement_l171_17161


namespace cube_surface_area_l171_17186

theorem cube_surface_area (a : ℝ) : 
    let edge_length := 3 * a
    let face_area := edge_length^2
    let total_surface_area := 6 * face_area
    total_surface_area = 54 * a^2 := 
by sorry

end cube_surface_area_l171_17186


namespace compute_product_l171_17150

theorem compute_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 10 :=
sorry

end compute_product_l171_17150


namespace range_of_a_l171_17160

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- Define the conditions: f has a unique zero point x₀ and x₀ < 0
def unique_zero_point (a : ℝ) : Prop :=
  ∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0

-- The theorem we need to prove
theorem range_of_a (a : ℝ) : unique_zero_point a → a > 2 :=
sorry

end range_of_a_l171_17160


namespace tangent_line_at_x_neg1_l171_17115

-- Definition of the curve.
def curve (x : ℝ) : ℝ := 2*x - x^3

-- Definition of the point of tangency.
def point_of_tangency_x : ℝ := -1

-- Definition of the point of tangency.
def point_of_tangency_y : ℝ := curve point_of_tangency_x

-- Definition of the derivative of the curve.
def derivative (x : ℝ) : ℝ := -3*x^2 + 2

-- Slope of the tangent at the point of tangency.
def slope_at_tangency : ℝ := derivative point_of_tangency_x

-- Equation of the tangent line function.
def tangent_line (x y : ℝ) := x + y + 2 = 0

theorem tangent_line_at_x_neg1 :
  tangent_line point_of_tangency_x point_of_tangency_y :=
by
  -- Here we will perform the proof, which is omitted for the purposes of this task.
  sorry

end tangent_line_at_x_neg1_l171_17115


namespace value_of_n_l171_17135

theorem value_of_n (n : ℝ) : (∀ (x y : ℝ), x^2 + y^2 - 2 * n * x + 2 * n * y + 2 * n^2 - 8 = 0 → (x + 1)^2 + (y - 1)^2 = 2) → n = 1 :=
by
  sorry

end value_of_n_l171_17135


namespace remainder_of_67_pow_67_plus_67_mod_68_l171_17171

theorem remainder_of_67_pow_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  -- Add the conditions and final proof step
  sorry

end remainder_of_67_pow_67_plus_67_mod_68_l171_17171


namespace ratio_of_fish_cat_to_dog_l171_17110

theorem ratio_of_fish_cat_to_dog (fish_dog : ℕ) (cost_per_fish : ℕ) (total_spent : ℕ)
  (h1 : fish_dog = 40)
  (h2 : cost_per_fish = 4)
  (h3 : total_spent = 240) :
  (total_spent / cost_per_fish - fish_dog) / fish_dog = 1 / 2 := by
  sorry

end ratio_of_fish_cat_to_dog_l171_17110


namespace ratio_area_shaded_triangle_l171_17101

variables (PQ PX QR QY YR : ℝ)
variables {A : ℝ}

def midpoint_QR (QR QY YR : ℝ) : Prop := QR = QY + YR ∧ QY = YR

def fraction_PQ_PX (PQ PX : ℝ) : Prop := PX = (3 / 4) * PQ

noncomputable def area_square (PQ : ℝ) : ℝ := PQ * PQ

noncomputable def area_triangle (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem ratio_area_shaded_triangle
  (PQ PX QR QY YR : ℝ)
  (h_mid : midpoint_QR QR QY YR)
  (h_frac : fraction_PQ_PX PQ PX)
  (hQY_QR2 : QY = QR / 2)
  (hYR_QR2 : YR = QR / 2) :
  A = 5 / 16 :=
sorry

end ratio_area_shaded_triangle_l171_17101


namespace water_level_after_opening_valve_l171_17170

-- Define the initial conditions and final height to be proved
def initial_water_height_cm : ℝ := 40
def initial_oil_height_cm : ℝ := 40
def water_density : ℝ := 1000
def oil_density : ℝ := 700
def final_water_height_cm : ℝ := 34

-- The proof that the final height of water after equilibrium will be 34 cm
theorem water_level_after_opening_valve :
  ∀ (h_w h_o : ℝ),
  (water_density * h_w = oil_density * h_o) ∧ (h_w + h_o = initial_water_height_cm + initial_oil_height_cm) →
  h_w = final_water_height_cm :=
by
  -- Here goes the proof, skipped with sorry
  sorry

end water_level_after_opening_valve_l171_17170


namespace A_should_shoot_air_l171_17179

-- Define the problem conditions
def hits_A : ℝ := 0.3
def hits_B : ℝ := 1
def hits_C : ℝ := 0.5

-- Define turns
inductive Turn
| A | B | C

-- Define the strategic choice
inductive Strategy
| aim_C | aim_B | shoot_air

-- Define the outcome structure
structure DuelOutcome where
  winner : Option Turn
  probability : ℝ

-- Noncomputable definition given the context of probabilistic reasoning
noncomputable def maximize_survival : Strategy := 
sorry

-- Main theorem to prove the optimal strategy
theorem A_should_shoot_air : maximize_survival = Strategy.shoot_air := 
sorry

end A_should_shoot_air_l171_17179


namespace sample_var_interpretation_l171_17199

theorem sample_var_interpretation (squared_diffs : Fin 10 → ℝ) :
  (10 = 10) ∧ (∀ i, squared_diffs i = (i - 20)^2) →
  (∃ n: ℕ, n = 10 ∧ ∃ μ: ℝ, μ = 20) :=
by
  intro h
  sorry

end sample_var_interpretation_l171_17199


namespace fifty_percent_of_x_l171_17129

variable (x : ℝ)

theorem fifty_percent_of_x (h : 0.40 * x = 160) : 0.50 * x = 200 :=
by
  sorry

end fifty_percent_of_x_l171_17129


namespace total_suitcases_correct_l171_17166

-- Conditions as definitions
def num_siblings : Nat := 4
def suitcases_per_sibling : Nat := 2
def num_parents : Nat := 2
def suitcases_per_parent : Nat := 3

-- Total suitcases calculation
def total_suitcases :=
  (num_siblings * suitcases_per_sibling) + (num_parents * suitcases_per_parent)

-- Statement to prove
theorem total_suitcases_correct : total_suitcases = 14 :=
by
  sorry

end total_suitcases_correct_l171_17166


namespace find_k_l171_17117

theorem find_k (k : ℝ) : 
  (∀ α β : ℝ, (α * β = 15 ∧ α + β = -k ∧ (α + 3 + β + 3 = k)) → k = 3) :=
by 
  sorry

end find_k_l171_17117


namespace check_numbers_has_property_P_l171_17155

def has_property_P (n : ℤ) : Prop :=
  ∃ x y z : ℤ, n = x^3 + y^3 + z^3 - 3 * x * y * z

theorem check_numbers_has_property_P :
  has_property_P 1 ∧ has_property_P 5 ∧ has_property_P 2014 ∧ ¬has_property_P 2013 :=
by
  sorry

end check_numbers_has_property_P_l171_17155


namespace expression_c_is_negative_l171_17152

noncomputable def A : ℝ := -4.2
noncomputable def B : ℝ := 2.3
noncomputable def C : ℝ := -0.5
noncomputable def D : ℝ := 3.4
noncomputable def E : ℝ := -1.8

theorem expression_c_is_negative : D / B * C < 0 := 
by
  -- proof goes here
  sorry

end expression_c_is_negative_l171_17152


namespace total_listening_days_l171_17108

theorem total_listening_days (x y z t : ℕ) (h1 : x = 8) (h2 : y = 12) (h3 : z = 30) (h4 : t = 2) :
  (x + y + z) * t = 100 :=
by
  sorry

end total_listening_days_l171_17108


namespace intersection_A_B_l171_17105

def A := {x : ℝ | x > 3}
def B := {x : ℝ | (x - 1) * (x - 4) < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 3 < x ∧ x < 4} :=
by
  sorry

end intersection_A_B_l171_17105


namespace number_of_ways_to_construct_cube_l171_17157

theorem number_of_ways_to_construct_cube :
  let num_white_cubes := 5
  let num_blue_cubes := 3
  let cube_size := (2, 2, 2)
  let num_rotations := 24
  let num_constructions := 4
  ∃ (num_constructions : ℕ), num_constructions = 4 :=
sorry

end number_of_ways_to_construct_cube_l171_17157


namespace solve_for_x_l171_17194

theorem solve_for_x : ∃ x : ℝ, (1 / 6 + 6 / x = 15 / x + 1 / 15) ∧ x = 90 :=
by
  sorry

end solve_for_x_l171_17194


namespace correct_structure_l171_17104

-- Definitions for the conditions regarding flowchart structures
def loop_contains_conditional : Prop := ∀ (loop : Prop), ∃ (conditional : Prop), conditional ∧ loop
def unique_flowchart_for_boiling_water : Prop := ∀ (flowcharts : Prop), ∃! (boiling_process : Prop), flowcharts ∧ boiling_process
def conditional_does_not_contain_sequential : Prop := ∀ (conditional : Prop), ∃ (sequential : Prop), ¬ (conditional ∧ sequential)
def conditional_must_contain_loop : Prop := ∀ (conditional : Prop), ∃ (loop : Prop), conditional ∧ loop

-- The proof statement
theorem correct_structure (A B C D : Prop) (hA : A = loop_contains_conditional) 
  (hB : B = unique_flowchart_for_boiling_water) 
  (hC : C = conditional_does_not_contain_sequential) 
  (hD : D = conditional_must_contain_loop) : 
  A = loop_contains_conditional ∧ ¬ B ∧ ¬ C ∧ ¬ D :=
by {
  sorry
}

end correct_structure_l171_17104


namespace triangle_right_angle_AB_solution_l171_17177

theorem triangle_right_angle_AB_solution (AC BC AB : ℝ) (hAC : AC = 6) (hBC : BC = 8) :
  (AC^2 + BC^2 = AB^2 ∨ AB^2 + AC^2 = BC^2) ↔ (AB = 10 ∨ AB = 2 * Real.sqrt 7) :=
by
  sorry

end triangle_right_angle_AB_solution_l171_17177


namespace dino_second_gig_hourly_rate_l171_17132

theorem dino_second_gig_hourly_rate (h1 : 20 * 10 = 200)
  (h2 : 5 * 40 = 200) (h3 : 500 + 500 = 1000) : 
  let total_income := 1000 
  let income_first_gig := 200 
  let income_third_gig := 200 
  let income_second_gig := total_income - income_first_gig - income_third_gig 
  let hours_second_gig := 30 
  let hourly_rate := income_second_gig / hours_second_gig 
  hourly_rate = 20 := 
by 
  sorry

end dino_second_gig_hourly_rate_l171_17132


namespace arithmetic_sequence_formula_geometric_sequence_formula_sum_of_sequence_l171_17164

theorem arithmetic_sequence_formula (a : ℕ → ℕ) (d : ℕ) (h1 : d > 0) 
  (h2 : a 1 + a 4 + a 7 = 12) (h3 : a 1 * a 4 * a 7 = 28) :
  ∀ n, a n = n :=
sorry

theorem geometric_sequence_formula (b : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : b 1 = 16) (h2 : a 2 * b 2 = 4) :
  ∀ n, b n = 2^(n + 3) :=
sorry

theorem sum_of_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : ∀ n, a n = n) (h2 : ∀ n, b n = 2^(n + 3)) 
  (h3 : ∀ n, c n = a n * b n) :
  ∀ n, T n = 8 * (2^n * (n + 1) - 1) :=
sorry

end arithmetic_sequence_formula_geometric_sequence_formula_sum_of_sequence_l171_17164


namespace geometric_sum_of_first_four_terms_eq_120_l171_17127

theorem geometric_sum_of_first_four_terms_eq_120
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (ha2 : a 2 = 9)
  (ha5 : a 5 = 243) :
  a 1 * (1 - r^4) / (1 - r) = 120 := 
sorry

end geometric_sum_of_first_four_terms_eq_120_l171_17127


namespace simplify_frac_l171_17107

variable (m : ℝ)

theorem simplify_frac : m^2 ≠ 9 → (3 / (m^2 - 9) + m / (9 - m^2)) = - (1 / (m + 3)) :=
by
  intro h
  sorry

end simplify_frac_l171_17107


namespace fraction_of_income_from_tips_l171_17185

variable (S T I : ℝ)
variable (h : T = (5 / 4) * S)

theorem fraction_of_income_from_tips (h : T = (5 / 4) * S) (I : ℝ) (w : I = S + T) : (T / I) = 5 / 9 :=
by
  -- The proof goes here
  sorry

end fraction_of_income_from_tips_l171_17185


namespace f_fixed_point_l171_17190

-- Definitions and conditions based on the problem statement
def g (n : ℕ) : ℕ := sorry
def f (n : ℕ) : ℕ := sorry

-- Helper functions for the repeated application of f
noncomputable def f_iter (n x : ℕ) : ℕ := 
    Nat.iterate f (x^2023) n

axiom g_bijective : Function.Bijective g
axiom f_repeated : ∀ x : ℕ, f_iter x x = x
axiom f_div_g : ∀ (x y : ℕ), x ∣ y → f x ∣ g y

-- Main theorem statement
theorem f_fixed_point : ∀ x : ℕ, f x = x := by
  sorry

end f_fixed_point_l171_17190


namespace f_2_values_l171_17148

theorem f_2_values (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, |f x - f y| = |x - y|)
  (hf1 : f 1 = 3) :
  f 2 = 2 ∨ f 2 = 4 :=
sorry

end f_2_values_l171_17148


namespace intersecting_lines_a_b_sum_zero_l171_17189

theorem intersecting_lines_a_b_sum_zero
    (a b : ℝ)
    (h₁ : ∀ z : ℝ × ℝ, z = (3, -3) → z.1 = (1 / 3) * z.2 + a)
    (h₂ : ∀ z : ℝ × ℝ, z = (3, -3) → z.2 = (1 / 3) * z.1 + b)
    :
    a + b = 0 := by
  sorry

end intersecting_lines_a_b_sum_zero_l171_17189


namespace trapezoid_perimeter_l171_17100

noncomputable def perimeter_trapezoid 
  (AB CD AD BC : ℝ) 
  (h_AB_CD_parallel : AB = CD) 
  (h_AD_perpendicular : AD = 4 * Real.sqrt 2)
  (h_BC_perpendicular : BC = 4 * Real.sqrt 2)
  (h_AB_eq : AB = 10)
  (h_CD_eq : CD = 18)
  (h_height : Real.sqrt (AD ^ 2 - 1) = 4) 
  : ℝ :=
AB + BC + CD + AD

theorem trapezoid_perimeter
  (AB CD AD BC : ℝ)
  (h_AB_CD_parallel : AB = CD) 
  (h_AD_perpendicular : AD = 4 * Real.sqrt 2)
  (h_BC_perpendicular : BC = 4 * Real.sqrt 2)
  (h_AB_eq : AB = 10)
  (h_CD_eq : CD = 18)
  (h_height : Real.sqrt (AD ^ 2 - 1) = 4) 
  : perimeter_trapezoid AB CD AD BC h_AB_CD_parallel h_AD_perpendicular h_BC_perpendicular h_AB_eq h_CD_eq h_height = 28 + 8 * Real.sqrt 2 :=
by
  sorry

end trapezoid_perimeter_l171_17100


namespace temperature_difference_on_day_xianning_l171_17198

theorem temperature_difference_on_day_xianning 
  (highest_temp : ℝ) (lowest_temp : ℝ) 
  (h_highest : highest_temp = 2) (h_lowest : lowest_temp = -3) : 
  highest_temp - lowest_temp = 5 := 
by
  sorry

end temperature_difference_on_day_xianning_l171_17198


namespace positive_even_integers_less_than_1000_not_divisible_by_3_or_11_l171_17168

theorem positive_even_integers_less_than_1000_not_divisible_by_3_or_11 :
  ∃ n : ℕ, n = 108 ∧
    (∀ m : ℕ, 0 < m → 2 ∣ m → m < 1000 → (¬ (3 ∣ m) ∧ ¬ (11 ∣ m) ↔ m ≤ n)) :=
sorry

end positive_even_integers_less_than_1000_not_divisible_by_3_or_11_l171_17168


namespace largest_five_digit_number_with_product_l171_17119

theorem largest_five_digit_number_with_product :
  ∃ (x : ℕ), (x = 98752) ∧ (∀ (d : List ℕ), (x.digits 10 = d) → (d.prod = 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) ∧ (x < 100000) ∧ (x ≥ 10000) :=
by
  sorry

end largest_five_digit_number_with_product_l171_17119


namespace simplify_sqrt_expression_l171_17130

theorem simplify_sqrt_expression : 2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 75 = 5 * Real.sqrt 3 :=
by
  sorry

end simplify_sqrt_expression_l171_17130


namespace rectangle_segments_sum_l171_17175

theorem rectangle_segments_sum :
  let EF := 6
  let FG := 8
  let n := 210
  let diagonal_length := Real.sqrt (EF^2 + FG^2)
  let segment_length (k : ℕ) : ℝ := diagonal_length * (n - k) / n
  let sum_segments := 2 * (Finset.sum (Finset.range 210) segment_length) - diagonal_length
  sum_segments = 2080 := by
  sorry

end rectangle_segments_sum_l171_17175


namespace letter_puzzle_l171_17134

theorem letter_puzzle (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_diff : A ≠ B) :
  A^B = 10 * B + A ↔ (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_l171_17134


namespace problem_solution_l171_17193

variable (x y : ℝ)

theorem problem_solution
  (h1 : (x + y)^2 = 64)
  (h2 : x * y = 15) :
  (x - y)^2 = 4 := 
by
  sorry

end problem_solution_l171_17193


namespace probability_not_sit_next_to_each_other_l171_17151

noncomputable def total_ways_to_choose_two_chairs_excluding_broken : ℕ := 28

noncomputable def unfavorable_outcomes : ℕ := 6

theorem probability_not_sit_next_to_each_other :
  (1 - (unfavorable_outcomes / total_ways_to_choose_two_chairs_excluding_broken) = (11 / 14)) :=
by sorry

end probability_not_sit_next_to_each_other_l171_17151


namespace geometric_sequence_sum_l171_17158

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geo : ∀ n, a (n + 1) = q * a n)
  (h1 : a 1 + a 2 + a 3 = 7)
  (h2 : a 2 + a 3 + a 4 = 14) :
  a 4 + a 5 + a 6 = 56 :=
sorry

end geometric_sequence_sum_l171_17158


namespace intersection_M_N_l171_17121

def setM : Set ℝ := {x | x^2 - 1 ≤ 0}
def setN : Set ℝ := {x | x^2 - 3 * x > 0}

theorem intersection_M_N :
  {x | -1 ≤ x ∧ x < 0} = setM ∩ setN :=
by
  sorry

end intersection_M_N_l171_17121


namespace children_neither_blue_nor_red_is_20_l171_17191

-- Definitions
def num_children : ℕ := 45
def num_adults : ℕ := num_children / 3
def num_adults_blue : ℕ := num_adults / 3
def num_adults_red : ℕ := 4
def num_adults_other_colors : ℕ := num_adults - num_adults_blue - num_adults_red
def num_children_red : ℕ := 15
def num_remaining_children : ℕ := num_children - num_children_red
def num_children_other_colors : ℕ := num_remaining_children / 2
def num_children_blue : ℕ := 2 * num_adults_blue
def num_children_neither_blue_nor_red : ℕ := num_children - num_children_red - num_children_blue

-- Theorem statement
theorem children_neither_blue_nor_red_is_20 : num_children_neither_blue_nor_red = 20 :=
  by
  sorry

end children_neither_blue_nor_red_is_20_l171_17191


namespace number_of_drawings_on_first_page_l171_17184

-- Let D be the number of drawings on the first page.
variable (D : ℕ)

-- Conditions:
-- 1. D is the number of drawings on the first page.
-- 2. The number of drawings increases by 5 after every page.
-- 3. The total number of drawings in the first five pages is 75.

theorem number_of_drawings_on_first_page (h : D + (D + 5) + (D + 10) + (D + 15) + (D + 20) = 75) :
    D = 5 :=
by
  sorry

end number_of_drawings_on_first_page_l171_17184


namespace sarah_house_units_digit_l171_17128

-- Sarah's house number has two digits
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- The four statements about Sarah's house number
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0
def has_digit_7 (n : ℕ) : Prop := n / 10 = 7 ∨ n % 10 = 7

-- Exactly three out of the four statements are true
def exactly_three_true (n : ℕ) : Prop :=
  (is_multiple_of_5 n ∧ is_odd n ∧ is_divisible_by_3 n ∧ ¬has_digit_7 n) ∨
  (is_multiple_of_5 n ∧ is_odd n ∧ ¬is_divisible_by_3 n ∧ has_digit_7 n) ∨
  (is_multiple_of_5 n ∧ ¬is_odd n ∧ is_divisible_by_3 n ∧ has_digit_7 n) ∨
  (¬is_multiple_of_5 n ∧ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_7 n)

-- Main statement
theorem sarah_house_units_digit : ∃ n : ℕ, is_two_digit n ∧ exactly_three_true n ∧ n % 10 = 5 :=
by
  sorry

end sarah_house_units_digit_l171_17128


namespace greatest_visible_unit_cubes_from_corner_l171_17136

theorem greatest_visible_unit_cubes_from_corner
  (n : ℕ) (units : ℕ) 
  (cube_volume : ∀ x, x = 1000)
  (face_size : ∀ x, x = 10) :
  (units = 274) :=
by sorry

end greatest_visible_unit_cubes_from_corner_l171_17136


namespace abc_inequality_l171_17165

theorem abc_inequality (a b c : ℝ) (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1) (h4 : a^3 + b^3 + c^3 = 1) : 
  a + b + c + a^2 + b^2 + c^2 ≤ 4 := 
sorry

end abc_inequality_l171_17165


namespace complex_number_sum_equals_one_l171_17181

variable {a b c d : ℝ}
variable {ω : ℂ}

theorem complex_number_sum_equals_one
  (ha : a ≠ -1) 
  (hb : b ≠ -1) 
  (hc : c ≠ -1) 
  (hd : d ≠ -1) 
  (hω : ω^4 = 1) 
  (hω_ne : ω ≠ 1)
  (h_eq : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / ω)
  : (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 1 :=
by sorry

end complex_number_sum_equals_one_l171_17181


namespace ratio_a_to_c_l171_17196

variables {a b c d : ℚ}

theorem ratio_a_to_c
  (h1 : a / b = 5 / 2)
  (h2 : c / d = 4 / 1)
  (h3 : d / b = 3 / 10) :
  a / c = 25 / 12 :=
sorry

end ratio_a_to_c_l171_17196


namespace find_smallest_solution_l171_17178

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l171_17178


namespace multiple_of_Jills_age_l171_17180

theorem multiple_of_Jills_age (m : ℤ) : 
  ∀ (J R F : ℤ),
  J = 20 →
  F = 40 →
  R = m * J + 5 →
  (R + 15) - (J + 15) = (F + 15) - 30 →
  m = 2 :=
by
  intros J R F hJ hF hR hDiff
  sorry

end multiple_of_Jills_age_l171_17180


namespace rectangle_diagonals_equal_rhombus_not_l171_17138

/-- Define the properties for a rectangle -/
structure Rectangle :=
  (sides_parallel : Prop)
  (diagonals_equal : Prop)
  (diagonals_bisect : Prop)
  (angles_equal : Prop)

/-- Define the properties for a rhombus -/
structure Rhombus :=
  (sides_parallel : Prop)
  (diagonals_equal : Prop)
  (diagonals_bisect : Prop)
  (angles_equal : Prop)

/-- The property that distinguishes a rectangle from a rhombus is that the diagonals are equal. -/
theorem rectangle_diagonals_equal_rhombus_not
  (R : Rectangle)
  (H : Rhombus)
  (hR1 : R.sides_parallel)
  (hR2 : R.diagonals_equal)
  (hR3 : R.diagonals_bisect)
  (hR4 : R.angles_equal)
  (hH1 : H.sides_parallel)
  (hH2 : ¬H.diagonals_equal)
  (hH3 : H.diagonals_bisect)
  (hH4 : H.angles_equal) :
  (R.diagonals_equal) := by
  sorry

end rectangle_diagonals_equal_rhombus_not_l171_17138


namespace complete_work_in_12_days_l171_17145

def Ravi_rate_per_day : ℚ := 1 / 24
def Prakash_rate_per_day : ℚ := 1 / 40
def Suresh_rate_per_day : ℚ := 1 / 60
def combined_rate_per_day : ℚ := Ravi_rate_per_day + Prakash_rate_per_day + Suresh_rate_per_day

theorem complete_work_in_12_days : 
  (1 / combined_rate_per_day) = 12 := 
by
  sorry

end complete_work_in_12_days_l171_17145


namespace min_value_of_quadratic_l171_17103

theorem min_value_of_quadratic (m : ℝ) (x : ℝ) (hx1 : 3 ≤ x) (hx2 : x < 4) (h : x^2 - 4 * x ≥ m) : 
  m ≤ -3 :=
sorry

end min_value_of_quadratic_l171_17103


namespace correct_transformation_l171_17146

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0) : (a / b = 2 * a / 2 * b) :=
by
  sorry

end correct_transformation_l171_17146


namespace value_of_a6_in_arithmetic_sequence_l171_17192

/-- In the arithmetic sequence {a_n}, if a_2 and a_{10} are the two roots of the equation
    x^2 + 12x - 8 = 0, prove that the value of a_6 is -6. -/
theorem value_of_a6_in_arithmetic_sequence :
  ∃ a_2 a_10 : ℤ, (a_2 + a_10 = -12 ∧
  (2: ℤ) * ((a_2 + a_10) / (2 * 1)) = a_2 + a_10 ) → 
  ∃ a_6: ℤ, a_6 = -6 :=
by
  sorry

end value_of_a6_in_arithmetic_sequence_l171_17192


namespace julia_total_cost_l171_17114

theorem julia_total_cost
  (snickers_cost : ℝ := 1.5)
  (mm_cost : ℝ := 2 * snickers_cost)
  (pepsi_cost : ℝ := 2 * mm_cost)
  (bread_cost : ℝ := 3 * pepsi_cost)
  (snickers_qty : ℕ := 2)
  (mm_qty : ℕ := 3)
  (pepsi_qty : ℕ := 4)
  (bread_qty : ℕ := 5)
  (money_given : ℝ := 5 * 20) :
  ((snickers_qty * snickers_cost) + (mm_qty * mm_cost) + (pepsi_qty * pepsi_cost) + (bread_qty * bread_cost)) > money_given := 
by
  sorry

end julia_total_cost_l171_17114


namespace ellipse_standard_equation_l171_17140

theorem ellipse_standard_equation (a b : ℝ) (h1 : 2 * a = 2 * (2 * b)) (h2 : (2, 0) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} ∨ (2, 0) ∈ {p : ℝ × ℝ | (p.2^2 / a^2) + (p.1^2 / b^2) = 1}) :
  (∃ a b : ℝ, (a > b ∧ a > 0 ∧ b > 0 ∧ (2 * a = 2 * (2 * b)) ∧ (2, 0) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} ∧ (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} → (x^2 / 4 + y^2 / 1 = 1)) ∨ (x^2 / 16 + y^2 / 4 = 1))) :=
  sorry

end ellipse_standard_equation_l171_17140


namespace hula_hoop_radius_l171_17131

theorem hula_hoop_radius (d : ℝ) (hd : d = 14) : d / 2 = 7 :=
by
  rw [hd]
  norm_num

end hula_hoop_radius_l171_17131


namespace sum_remainder_product_remainder_l171_17182

open Nat

-- Define the modulus conditions
variables (x y z : ℕ)
def condition1 : Prop := x % 15 = 11
def condition2 : Prop := y % 15 = 13
def condition3 : Prop := z % 15 = 14

-- Proof statement for the sum remainder
theorem sum_remainder (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 z) : (x + y + z) % 15 = 8 :=
by
  sorry

-- Proof statement for the product remainder
theorem product_remainder (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 z) : (x * y * z) % 15 = 2 :=
by
  sorry

end sum_remainder_product_remainder_l171_17182


namespace montague_fraction_l171_17156

noncomputable def fraction_montague (M C : ℝ) : Prop :=
  M + C = 1 ∧
  (0.70 * C) / (0.20 * M + 0.70 * C) = 7 / 11

theorem montague_fraction : ∃ M C : ℝ, fraction_montague M C ∧ M = 2 / 3 :=
by sorry

end montague_fraction_l171_17156


namespace pen_cost_l171_17188

theorem pen_cost
  (p q : ℕ)
  (h1 : 6 * p + 5 * q = 380)
  (h2 : 3 * p + 8 * q = 298) :
  p = 47 :=
sorry

end pen_cost_l171_17188


namespace math_problem_l171_17162

variables {R : Type*} [Ring R] (x y z : R)

theorem math_problem (h : x * y + y * z + z * x = 0) : 
  3 * x * y * z + x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) = 0 :=
by 
  sorry

end math_problem_l171_17162


namespace borrowing_methods_l171_17163

theorem borrowing_methods (A_has_3_books : True) (B_borrows_at_least_one_book : True) :
  (∃ (methods : ℕ), methods = 7) :=
by
  existsi 7
  sorry

end borrowing_methods_l171_17163


namespace problem_1_problem_2_l171_17125

noncomputable def f (x k : ℝ) : ℝ := (2 * k * x) / (x * x + 6 * k)

theorem problem_1 (k m : ℝ) (hk : k > 0)
  (hsol : ∀ x, (f x k) > m ↔ x < -3 ∨ x > -2) :
  ∀ x, 5 * m * x ^ 2 + k * x + 3 > 0 ↔ -1 < x ∧ x < 3 / 2 :=
sorry

theorem problem_2 (k : ℝ) (hk : k > 0)
  (hsol : ∃ (x : ℝ), x > 3 ∧ (f x k) > 1) :
  k > 6 :=
sorry

end problem_1_problem_2_l171_17125


namespace median_line_eqn_l171_17118

theorem median_line_eqn (A B C : ℝ × ℝ)
  (hA : A = (3, 7)) (hB : B = (5, -1)) (hC : C = (-2, -5)) :
  ∃ m b : ℝ, (4, -3, -7) = (m, b, 0) :=
by sorry

end median_line_eqn_l171_17118


namespace sandwich_cost_l171_17120

theorem sandwich_cost (soda_cost sandwich_cost total_cost : ℝ) (h1 : soda_cost = 0.87) (h2 : total_cost = 10.46) (h3 : 4 * soda_cost + 2 * sandwich_cost = total_cost) :
  sandwich_cost = 3.49 :=
by
  sorry

end sandwich_cost_l171_17120


namespace tangent_circles_radii_l171_17106

noncomputable def radii_of_tangent_circles (R r : ℝ) (h : R > r) : Set ℝ :=
  { x | x = (R * r) / ((Real.sqrt R + Real.sqrt r)^2) ∨ x = (R * r) / ((Real.sqrt R - Real.sqrt r)^2) }

theorem tangent_circles_radii (R r : ℝ) (h : R > r) :
  ∃ x, x ∈ radii_of_tangent_circles R r h := sorry

end tangent_circles_radii_l171_17106


namespace job_positions_growth_rate_l171_17109

theorem job_positions_growth_rate (x : ℝ) :
  1501 * (1 + x) ^ 2 = 1815 := sorry

end job_positions_growth_rate_l171_17109


namespace miles_round_trip_time_l171_17113

theorem miles_round_trip_time : 
  ∀ (d : ℝ), d = 57 →
  ∀ (t : ℝ), t = 40 →
  ∀ (x : ℝ), x = 4 →
  10 = ((2 * d * x) / t) * 2 := 
by
  intros d hd t ht x hx
  rw [hd, ht, hx]
  sorry

end miles_round_trip_time_l171_17113


namespace carlos_books_in_june_l171_17144

def books_in_july : ℕ := 28
def books_in_august : ℕ := 30
def goal_books : ℕ := 100

theorem carlos_books_in_june :
  let books_in_july_august := books_in_july + books_in_august
  let books_needed_june := goal_books - books_in_july_august
  books_needed_june = 42 := 
by
  sorry

end carlos_books_in_june_l171_17144
