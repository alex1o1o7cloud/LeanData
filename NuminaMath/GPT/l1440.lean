import Mathlib

namespace NUMINAMATH_GPT_pencil_price_is_99c_l1440_144007

noncomputable def one_pencil_cost (total_spent : ℝ) (notebook_price : ℝ) (notebook_count : ℕ) 
                                  (ruler_pack_price : ℝ) (eraser_price : ℝ) (eraser_count : ℕ) 
                                  (pencil_count : ℕ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let notebooks_cost := notebook_count * notebook_price
  let discount_amount := discount * notebooks_cost
  let discounted_notebooks_cost := notebooks_cost - discount_amount
  let other_items_cost := ruler_pack_price + (eraser_count * eraser_price)
  let subtotal := discounted_notebooks_cost + other_items_cost
  let pencils_total_after_tax := total_spent - subtotal
  let pencils_total_before_tax := pencils_total_after_tax / (1 + tax)
  let pencil_price := pencils_total_before_tax / pencil_count
  pencil_price

theorem pencil_price_is_99c : one_pencil_cost 7.40 0.85 2 0.60 0.20 5 4 0.15 0.10 = 0.99 := 
sorry

end NUMINAMATH_GPT_pencil_price_is_99c_l1440_144007


namespace NUMINAMATH_GPT_f_2014_l1440_144056

noncomputable def f : ℕ → ℕ := sorry

axiom f_property : ∀ n, f (f n) + f n = 2 * n + 3
axiom f_zero : f 0 = 1

theorem f_2014 : f 2014 = 2015 := 
by sorry

end NUMINAMATH_GPT_f_2014_l1440_144056


namespace NUMINAMATH_GPT_decimal_to_binary_45_l1440_144063

theorem decimal_to_binary_45 :
  (45 : ℕ) = (0b101101 : ℕ) :=
sorry

end NUMINAMATH_GPT_decimal_to_binary_45_l1440_144063


namespace NUMINAMATH_GPT_instantaneous_velocity_at_t4_l1440_144073

-- Definition of the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- The proof problem statement: Proving that the derivative of s at t = 4 is 7
theorem instantaneous_velocity_at_t4 : deriv s 4 = 7 :=
by sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_t4_l1440_144073


namespace NUMINAMATH_GPT_different_types_of_players_l1440_144048

theorem different_types_of_players :
  ∀ (cricket hockey football softball : ℕ) (total_players : ℕ),
    cricket = 12 → hockey = 17 → football = 11 → softball = 10 → total_players = 50 →
    cricket + hockey + football + softball = total_players → 
    4 = 4 :=
by
  intros
  rfl

end NUMINAMATH_GPT_different_types_of_players_l1440_144048


namespace NUMINAMATH_GPT_remainder_98_pow_50_mod_100_l1440_144077

theorem remainder_98_pow_50_mod_100 :
  (98 : ℤ) ^ 50 % 100 = 24 := by
  sorry

end NUMINAMATH_GPT_remainder_98_pow_50_mod_100_l1440_144077


namespace NUMINAMATH_GPT_bridge_length_l1440_144047

def train_length : ℕ := 170 -- Train length in meters
def train_speed : ℕ := 45 -- Train speed in kilometers per hour
def crossing_time : ℕ := 30 -- Time to cross the bridge in seconds

noncomputable def speed_m_per_s : ℚ := (train_speed * 1000) / 3600

noncomputable def total_distance : ℚ := speed_m_per_s * crossing_time

theorem bridge_length : total_distance - train_length = 205 :=
by
  sorry

end NUMINAMATH_GPT_bridge_length_l1440_144047


namespace NUMINAMATH_GPT_polynomial_factor_l1440_144081

theorem polynomial_factor (a b : ℝ) : 
  (∃ c d : ℝ, (5 * c = a) ∧ (5 * d - 3 * c = b) ∧ (2 * c - 3 * d + 25 = 45) ∧ (2 * d - 15 = -18)) 
  → (a = 151.25 ∧ b = -98.25) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_factor_l1440_144081


namespace NUMINAMATH_GPT_tall_cupboard_glasses_l1440_144061

-- Define the number of glasses held by the tall cupboard (T)
variable (T : ℕ)

-- Condition: Wide cupboard holds twice as many glasses as the tall cupboard
def wide_cupboard_holds_twice_as_many (T : ℕ) : Prop :=
  ∃ W : ℕ, W = 2 * T

-- Condition: Narrow cupboard holds 15 glasses initially, 5 glasses per shelf, one shelf broken
def narrow_cupboard_holds_after_break : Prop :=
  ∃ N : ℕ, N = 10

-- Final statement to prove: Number of glasses in the tall cupboard is 5
theorem tall_cupboard_glasses (T : ℕ) (h1 : wide_cupboard_holds_twice_as_many T) (h2 : narrow_cupboard_holds_after_break) : T = 5 :=
sorry

end NUMINAMATH_GPT_tall_cupboard_glasses_l1440_144061


namespace NUMINAMATH_GPT_Elmer_eats_more_than_Penelope_l1440_144016

noncomputable def Penelope_food := 20
noncomputable def Greta_food := Penelope_food / 10
noncomputable def Milton_food := Greta_food / 100
noncomputable def Elmer_food := 4000 * Milton_food

theorem Elmer_eats_more_than_Penelope :
  Elmer_food - Penelope_food = 60 := 
by
  sorry

end NUMINAMATH_GPT_Elmer_eats_more_than_Penelope_l1440_144016


namespace NUMINAMATH_GPT_slightly_used_crayons_l1440_144009

theorem slightly_used_crayons (total_crayons : ℕ) (percent_new : ℚ) (percent_broken : ℚ) 
  (h1 : total_crayons = 250) (h2 : percent_new = 40/100) (h3 : percent_broken = 1/5) : 
  (total_crayons - percent_new * total_crayons - percent_broken * total_crayons) = 100 :=
by
  -- sorry here to indicate the proof is omitted
  sorry

end NUMINAMATH_GPT_slightly_used_crayons_l1440_144009


namespace NUMINAMATH_GPT_Randy_used_blocks_l1440_144062

theorem Randy_used_blocks (initial_blocks blocks_left used_blocks : ℕ) 
  (h1 : initial_blocks = 97) 
  (h2 : blocks_left = 72) 
  (h3 : used_blocks = initial_blocks - blocks_left) : 
  used_blocks = 25 :=
by
  sorry

end NUMINAMATH_GPT_Randy_used_blocks_l1440_144062


namespace NUMINAMATH_GPT_tangent_line_iff_l1440_144082

theorem tangent_line_iff (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 8 * y + 12 = 0 → ax + y + 2 * a = 0) ↔ a = -3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_iff_l1440_144082


namespace NUMINAMATH_GPT_units_digit_33_219_89_plus_89_19_l1440_144044

theorem units_digit_33_219_89_plus_89_19 :
  let units_digit x := x % 10
  units_digit (33 * 219 ^ 89 + 89 ^ 19) = 8 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_33_219_89_plus_89_19_l1440_144044


namespace NUMINAMATH_GPT_degree_of_monomial_x_l1440_144008

def is_monomial (e : Expr) : Prop := sorry -- Placeholder definition
def degree (e : Expr) : Nat := sorry -- Placeholder definition

theorem degree_of_monomial_x :
  degree x = 1 :=
by
  sorry

end NUMINAMATH_GPT_degree_of_monomial_x_l1440_144008


namespace NUMINAMATH_GPT_polynomial_remainder_l1440_144054

theorem polynomial_remainder (a b : ℤ) :
  (∀ x : ℤ, 3 * x ^ 6 - 2 * x ^ 4 + 5 * x ^ 2 - 9 = (x + 1) * (x + 2) * (q : ℤ) + a * x + b) →
  (a = -174 ∧ b = -177) :=
by sorry

end NUMINAMATH_GPT_polynomial_remainder_l1440_144054


namespace NUMINAMATH_GPT_trigonometric_identity_l1440_144038

theorem trigonometric_identity (α : Real) (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10) / Real.sin (α - Real.pi / 5) = 3) :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1440_144038


namespace NUMINAMATH_GPT_polygon_divided_l1440_144090

theorem polygon_divided (p q r : ℕ) : p - q + r = 1 :=
sorry

end NUMINAMATH_GPT_polygon_divided_l1440_144090


namespace NUMINAMATH_GPT_contractor_work_done_l1440_144039

def initial_people : ℕ := 10
def remaining_people : ℕ := 8
def total_days : ℕ := 100
def remaining_days : ℕ := 75
def fraction_done : ℚ := 1/4
def total_work : ℚ := 1

theorem contractor_work_done (x : ℕ) 
  (h1 : initial_people * x = fraction_done * total_work) 
  (h2 : remaining_people * remaining_days = (1 - fraction_done) * total_work) :
  x = 60 :=
by
  sorry

end NUMINAMATH_GPT_contractor_work_done_l1440_144039


namespace NUMINAMATH_GPT_min_cylinder_surface_area_l1440_144089

noncomputable def h := Real.sqrt (5^2 - 4^2)
noncomputable def V_cone := (1 / 3) * Real.pi * 4^2 * h
noncomputable def V_cylinder (r h': ℝ) := Real.pi * r^2 * h'
noncomputable def h' (r: ℝ) := 16 / r^2
noncomputable def S (r: ℝ) := 2 * Real.pi * r^2 + (32 * Real.pi) / r

theorem min_cylinder_surface_area : 
  ∃ r, r = 2 ∧ ∀ r', r' ≠ 2 → S r' > S 2 := sorry

end NUMINAMATH_GPT_min_cylinder_surface_area_l1440_144089


namespace NUMINAMATH_GPT_radius_of_circle_B_l1440_144014

-- Definitions of circles and their properties
noncomputable def circle_tangent_externally (r1 r2 : ℝ) := ∃ d : ℝ, d = r1 + r2
noncomputable def circle_tangent_internally (r1 r2 : ℝ) := ∃ d : ℝ, d = r2 - r1

-- Problem statement in Lean 4
theorem radius_of_circle_B
  (rA rB rC rD centerA centerB centerC centerD : ℝ)
  (h_rA : rA = 2)
  (h_congruent_B_C : rB = rC)
  (h_circle_A_tangent_to_B : circle_tangent_externally rA rB)
  (h_circle_A_tangent_to_C : circle_tangent_externally rA rC)
  (h_circle_B_C_tangent_e : circle_tangent_externally rB rC)
  (h_circle_B_D_tangent_i : circle_tangent_internally rB rD)
  (h_center_A_passes_D : centerA = centerD)
  (h_rD : rD = 4) : 
  rB = 1 := sorry

end NUMINAMATH_GPT_radius_of_circle_B_l1440_144014


namespace NUMINAMATH_GPT_number_of_segments_before_returning_to_start_l1440_144075

-- Definitions based on the conditions
def concentric_circles (r R : ℝ) (h_circle : r < R) : Prop := true

def tangent_chord (circle1 circle2 : Prop) (A B : Point) : Prop := 
  circle1 ∧ circle2

def angle_ABC_eq_60 (A B C : Point) (angle_ABC : ℝ) : Prop :=
  angle_ABC = 60

noncomputable def number_of_segments (n : ℕ) (m : ℕ) : Prop := 
  120 * n = 360 * m

theorem number_of_segments_before_returning_to_start (r R : ℝ)
  (h_circle : r < R)
  (circle1 circle2 : Prop := concentric_circles r R h_circle)
  (A B C : Point)
  (h_tangent : tangent_chord circle1 circle2 A B)
  (angle_ABC : ℝ := 0)
  (h_ABC_eq_60 : angle_ABC_eq_60 A B C angle_ABC) :
  ∃ n : ℕ, number_of_segments n 1 ∧ n = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_segments_before_returning_to_start_l1440_144075


namespace NUMINAMATH_GPT_detergent_required_l1440_144053

def ounces_of_detergent_per_pound : ℕ := 2
def pounds_of_clothes : ℕ := 9

theorem detergent_required :
  (ounces_of_detergent_per_pound * pounds_of_clothes) = 18 := by
  sorry

end NUMINAMATH_GPT_detergent_required_l1440_144053


namespace NUMINAMATH_GPT_first_term_geometric_progression_l1440_144004

theorem first_term_geometric_progression (S a : ℝ) (r : ℝ) 
  (h1 : S = 10) 
  (h2 : a = 10 * (1 - r)) 
  (h3 : a * (1 + r) = 7) : 
  a = 10 * (1 - Real.sqrt (3 / 10)) ∨ a = 10 * (1 + Real.sqrt (3 / 10)) := 
by 
  sorry

end NUMINAMATH_GPT_first_term_geometric_progression_l1440_144004


namespace NUMINAMATH_GPT_average_weight_20_boys_l1440_144079

theorem average_weight_20_boys 
  (A : Real)
  (numBoys₁ numBoys₂ : ℕ)
  (weight₂ : Real)
  (avg_weight_class : Real)
  (h_numBoys₁ : numBoys₁ = 20)
  (h_numBoys₂ : numBoys₂ = 8)
  (h_weight₂ : weight₂ = 45.15)
  (h_avg_weight_class : avg_weight_class = 48.792857142857144)
  (h_total_boys : numBoys₁ + numBoys₂ = 28)
  (h_eq_weight : numBoys₁ * A + numBoys₂ * weight₂ = 28 * avg_weight_class) :
  A = 50.25 :=
  sorry

end NUMINAMATH_GPT_average_weight_20_boys_l1440_144079


namespace NUMINAMATH_GPT_quadratic_roots_relation_l1440_144042

noncomputable def roots_relation (a b c : ℝ) : Prop :=
  ∃ α β : ℝ, (α * β = c / a) ∧ (α + β = -b / a) ∧ β = 3 * α

theorem quadratic_roots_relation (a b c : ℝ) (h : roots_relation a b c) : 3 * b^2 = 16 * a * c :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_relation_l1440_144042


namespace NUMINAMATH_GPT_range_of_x_l1440_144028

noncomputable def g (x : ℝ) : ℝ := 2^x + 2^(-x) + |x|

theorem range_of_x (x : ℝ) : g (2 * x - 1) < g 3 → -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_GPT_range_of_x_l1440_144028


namespace NUMINAMATH_GPT_min_value_of_expression_l1440_144026

theorem min_value_of_expression :
  ∀ (x y : ℝ), ∃ a b : ℝ, x = 5 ∧ y = -3 ∧ (x^2 + y^2 - 10*x + 6*y + 25) = -9 := 
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l1440_144026


namespace NUMINAMATH_GPT_volume_of_box_l1440_144094

-- Define the dimensions of the box
variables (L W H : ℝ)

-- Define the conditions as hypotheses
def side_face_area : Prop := H * W = 288
def top_face_area : Prop := L * W = 1.5 * 288
def front_face_area : Prop := L * H = 0.5 * (L * W)

-- Define the volume of the box
def box_volume : ℝ := L * W * H

-- The proof statement
theorem volume_of_box (h1 : side_face_area H W) (h2 : top_face_area L W) (h3 : front_face_area L H W) : box_volume L W H = 5184 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_box_l1440_144094


namespace NUMINAMATH_GPT_jellybean_count_l1440_144045

theorem jellybean_count (x : ℕ) (h : (0.7 : ℝ) ^ 3 * x = 34) : x = 99 :=
sorry

end NUMINAMATH_GPT_jellybean_count_l1440_144045


namespace NUMINAMATH_GPT_intersection_M_N_is_neq_neg1_0_1_l1440_144057

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N_is_neq_neg1_0_1 :
  M ∩ N = {-1, 0, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_is_neq_neg1_0_1_l1440_144057


namespace NUMINAMATH_GPT_Lisa_initial_pencils_l1440_144068

-- Variables
variable (G_L_initial : ℕ) (L_L_initial : ℕ) (G_L_total : ℕ)

-- Conditions
def G_L_initial_def := G_L_initial = 2
def G_L_total_def := G_L_total = 101
def Lisa_gives_pencils : Prop := G_L_total = G_L_initial + L_L_initial

-- Proof statement
theorem Lisa_initial_pencils (G_L_initial : ℕ) (G_L_total : ℕ)
  (h1 : G_L_initial = 2) (h2 : G_L_total = 101) (h3 : G_L_total = G_L_initial + L_L_initial) :
  L_L_initial = 99 := 
by 
  sorry

end NUMINAMATH_GPT_Lisa_initial_pencils_l1440_144068


namespace NUMINAMATH_GPT_find_a_value_l1440_144029

theorem find_a_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq1 : a^b = b^a) (h_eq2 : b = 3 * a) : a = Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_find_a_value_l1440_144029


namespace NUMINAMATH_GPT_inequality_arith_geo_mean_l1440_144067

theorem inequality_arith_geo_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b + b / Real.sqrt a) ≥ (Real.sqrt a + Real.sqrt b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_arith_geo_mean_l1440_144067


namespace NUMINAMATH_GPT_merchant_discount_l1440_144022

-- Definitions used in Lean 4 statement coming directly from conditions
def initial_cost_price : Real := 100
def marked_up_percentage : Real := 0.80
def profit_percentage : Real := 0.35

-- To prove the percentage discount offered
theorem merchant_discount (cp mp sp discount percentage_discount : Real) 
  (H1 : cp = initial_cost_price)
  (H2 : mp = cp + (marked_up_percentage * cp))
  (H3 : sp = cp + (profit_percentage * cp))
  (H4 : discount = mp - sp)
  (H5 : percentage_discount = (discount / mp) * 100) :
  percentage_discount = 25 := 
sorry

end NUMINAMATH_GPT_merchant_discount_l1440_144022


namespace NUMINAMATH_GPT_perfect_square_as_sum_of_powers_of_2_l1440_144040

theorem perfect_square_as_sum_of_powers_of_2 (n a b : ℕ) (h : n^2 = 2^a + 2^b) (hab : a ≥ b) :
  (∃ k : ℕ, n^2 = 4^(k + 1)) ∨ (∃ k : ℕ, n^2 = 9 * 4^k) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_as_sum_of_powers_of_2_l1440_144040


namespace NUMINAMATH_GPT_max_k_C_l1440_144099

theorem max_k_C (n : ℕ) (h1 : Odd n) (h2 : 0 < n) :
  ∃ k : ℕ, (k = ((n + 1) / 2) ^ 2) := 
sorry

end NUMINAMATH_GPT_max_k_C_l1440_144099


namespace NUMINAMATH_GPT_required_number_l1440_144043

-- Define the main variables and conditions
variables {i : ℂ} (z : ℂ)
axiom i_squared : i^2 = -1

-- State the theorem that needs to be proved
theorem required_number (h : z + (4 - 8 * i) = 1 + 10 * i) : z = -3 + 18 * i :=
by {
  -- the exact steps for the proof will follow here
  sorry
}

end NUMINAMATH_GPT_required_number_l1440_144043


namespace NUMINAMATH_GPT_find_pre_tax_remuneration_l1440_144035

def pre_tax_remuneration (x : ℝ) : Prop :=
  let taxable_amount := if x <= 4000 then x - 800 else x * 0.8
  let tax_due := taxable_amount * 0.2
  let final_tax := tax_due * 0.7
  final_tax = 280

theorem find_pre_tax_remuneration : ∃ x : ℝ, pre_tax_remuneration x ∧ x = 2800 := by
  sorry

end NUMINAMATH_GPT_find_pre_tax_remuneration_l1440_144035


namespace NUMINAMATH_GPT_evaluate_difference_of_squares_l1440_144030
-- Import necessary libraries

-- Define the specific values for a and b
def a : ℕ := 72
def b : ℕ := 48

-- State the theorem to be proved
theorem evaluate_difference_of_squares : a^2 - b^2 = (a + b) * (a - b) ∧ (a + b) * (a - b) = 2880 := 
by
  -- The proof would go here but should be omitted as per directions
  sorry

end NUMINAMATH_GPT_evaluate_difference_of_squares_l1440_144030


namespace NUMINAMATH_GPT_mike_training_hours_l1440_144058

-- Define the individual conditions
def first_weekday_hours : Nat := 2
def first_weekend_hours : Nat := 1
def first_week_days : Nat := 5
def first_weekend_days : Nat := 2

def second_weekday_hours : Nat := 3
def second_weekend_hours : Nat := 2
def second_week_days : Nat := 4  -- since the first day of second week is a rest day
def second_weekend_days : Nat := 2

def first_week_hours : Nat := (first_weekday_hours * first_week_days) + (first_weekend_hours * first_weekend_days)
def second_week_hours : Nat := (second_weekday_hours * second_week_days) + (second_weekend_hours * second_weekend_days)

def total_training_hours : Nat := first_week_hours + second_week_hours

-- The final proof statement
theorem mike_training_hours : total_training_hours = 28 := by
  exact sorry

end NUMINAMATH_GPT_mike_training_hours_l1440_144058


namespace NUMINAMATH_GPT_find_x_l1440_144095

def determinant (a b c d : ℚ) : ℚ := a * d - b * c

theorem find_x (x : ℚ) (h : determinant (2 * x) (-4) x 1 = 18) : x = 3 :=
  sorry

end NUMINAMATH_GPT_find_x_l1440_144095


namespace NUMINAMATH_GPT_expression_divisible_by_13_l1440_144021

theorem expression_divisible_by_13 (a b c : ℤ) (h : (a + b + c) % 13 = 0) : 
  (a ^ 2007 + b ^ 2007 + c ^ 2007 + 2 * 2007 * a * b * c) % 13 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_expression_divisible_by_13_l1440_144021


namespace NUMINAMATH_GPT_congruence_solution_l1440_144011

theorem congruence_solution (x : ℤ) (h : 5 * x + 11 ≡ 3 [ZMOD 19]) : 3 * x + 7 ≡ 6 [ZMOD 19] :=
sorry

end NUMINAMATH_GPT_congruence_solution_l1440_144011


namespace NUMINAMATH_GPT_power_greater_than_one_million_l1440_144093

theorem power_greater_than_one_million (α β γ δ : ℝ) (ε ζ η : ℕ)
  (h1 : α = 1.01) (h2 : β = 1.001) (h3 : γ = 1.000001) 
  (h4 : δ = 1000000) 
  (h_eps : ε = 99999900) (h_zet : ζ = 999999000) (h_eta : η = 999999000000) :
  α^ε > δ ∧ β^ζ > δ ∧ γ^η > δ :=
by
  sorry

end NUMINAMATH_GPT_power_greater_than_one_million_l1440_144093


namespace NUMINAMATH_GPT_power_function_at_100_l1440_144041

-- Given a power function f(x) = x^α that passes through the point (9, 3),
-- show that f(100) = 10.

theorem power_function_at_100 (α : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x ^ α)
  (h2 : f 9 = 3) : f 100 = 10 :=
sorry

end NUMINAMATH_GPT_power_function_at_100_l1440_144041


namespace NUMINAMATH_GPT_multiplier_is_3_l1440_144046

theorem multiplier_is_3 (x : ℝ) (num : ℝ) (difference : ℝ) (h1 : num = 15.0) (h2 : difference = 40) (h3 : x * num - 5 = difference) : x = 3 := 
by 
  sorry

end NUMINAMATH_GPT_multiplier_is_3_l1440_144046


namespace NUMINAMATH_GPT_find_a_b_l1440_144060

noncomputable def z : ℂ := 1 + Complex.I
noncomputable def lhs (a b : ℝ) := (z^2 + a*z + b) / (z^2 - z + 1)
noncomputable def rhs : ℂ := 1 - Complex.I

theorem find_a_b (a b : ℝ) (h : lhs a b = rhs) : a = -1 ∧ b = 2 :=
  sorry

end NUMINAMATH_GPT_find_a_b_l1440_144060


namespace NUMINAMATH_GPT_expected_area_convex_hull_correct_l1440_144098

def point_placement (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 10

def convex_hull_area (points : Finset (ℕ × ℤ)) : ℚ := 
  -- Definition of the area calculation goes here. This is a placeholder.
  0  -- Placeholder for the actual calculation

noncomputable def expected_convex_hull_area : ℚ := 
  -- Calculation of the expected area, which is complex and requires integration of the probability.
  sorry  -- Placeholder for the actual expected value

theorem expected_area_convex_hull_correct : 
  expected_convex_hull_area = 1793 / 128 :=
sorry

end NUMINAMATH_GPT_expected_area_convex_hull_correct_l1440_144098


namespace NUMINAMATH_GPT_find_d_l1440_144064

theorem find_d (d : ℤ) (h : ∀ x : ℤ, 8 * x^3 + 23 * x^2 + d * x + 45 = 0 → 2 * x + 5 = 0) : 
  d = 163 := 
sorry

end NUMINAMATH_GPT_find_d_l1440_144064


namespace NUMINAMATH_GPT_rolls_sold_to_uncle_l1440_144091

theorem rolls_sold_to_uncle (total_rolls : ℕ) (rolls_grandmother : ℕ) (rolls_neighbor : ℕ) (rolls_remaining : ℕ) (rolls_uncle : ℕ) :
  total_rolls = 12 →
  rolls_grandmother = 3 →
  rolls_neighbor = 3 →
  rolls_remaining = 2 →
  rolls_uncle = total_rolls - rolls_remaining - (rolls_grandmother + rolls_neighbor) →
  rolls_uncle = 4 :=
by
  intros h_total h_grandmother h_neighbor h_remaining h_compute
  rw [h_total, h_grandmother, h_neighbor, h_remaining] at h_compute
  exact h_compute

end NUMINAMATH_GPT_rolls_sold_to_uncle_l1440_144091


namespace NUMINAMATH_GPT_f_2015_l1440_144074

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f (x + 2) = f (2 - x) + 4 * f 2
axiom symmetric_about_neg1 : ∀ x : ℝ, f (x + 1) = f (-2 - (x + 1))
axiom f_at_1 : f 1 = 3

theorem f_2015 : f 2015 = -3 :=
by
  apply sorry

end NUMINAMATH_GPT_f_2015_l1440_144074


namespace NUMINAMATH_GPT_range_of_alpha_div_three_l1440_144059

open Real

theorem range_of_alpha_div_three {k : ℤ} {α : ℝ} 
  (h1 : sin α > 0)
  (h2 : cos α < 0)
  (h3 : sin (α / 3) > cos (α / 3)) :
  (2 * k * π + π / 4 < α / 3 ∧ α / 3 < 2 * k * π + π / 3) 
  ∨ (2 * k * π + 5 * π / 6 < α / 3 ∧ α / 3 < 2 * k * π + π) :=
sorry

end NUMINAMATH_GPT_range_of_alpha_div_three_l1440_144059


namespace NUMINAMATH_GPT_dina_dolls_l1440_144031

theorem dina_dolls (Ivy_collectors: ℕ) (h1: Ivy_collectors = 20) (h2: ∀ y : ℕ, 2 * y / 3 = Ivy_collectors) :
  ∃ x : ℕ, 2 * x = 60 :=
  sorry

end NUMINAMATH_GPT_dina_dolls_l1440_144031


namespace NUMINAMATH_GPT_cauliflower_sales_l1440_144020

theorem cauliflower_sales :
  let total_earnings := 500
  let b_sales := 57
  let c_sales := 2 * b_sales
  let s_sales := (c_sales / 2) + 16
  let t_sales := b_sales + s_sales
  let ca_sales := total_earnings - (b_sales + c_sales + s_sales + t_sales)
  ca_sales = 126 := by
  sorry

end NUMINAMATH_GPT_cauliflower_sales_l1440_144020


namespace NUMINAMATH_GPT_rides_with_remaining_tickets_l1440_144015

theorem rides_with_remaining_tickets (T_total : ℕ) (T_spent : ℕ) (C_ride : ℕ)
  (h1 : T_total = 40) (h2 : T_spent = 28) (h3 : C_ride = 4) :
  (T_total - T_spent) / C_ride = 3 := by
  sorry

end NUMINAMATH_GPT_rides_with_remaining_tickets_l1440_144015


namespace NUMINAMATH_GPT_binomial_prime_divisor_l1440_144017

theorem binomial_prime_divisor (p k : ℕ) (hp : Nat.Prime p) (hk1 : 1 ≤ k) (hk2 : k ≤ p - 1) : p ∣ Nat.choose p k :=
by
  sorry

end NUMINAMATH_GPT_binomial_prime_divisor_l1440_144017


namespace NUMINAMATH_GPT_line_curve_intersection_symmetric_l1440_144027

theorem line_curve_intersection_symmetric (a b : ℝ) 
    (h1 : ∃ p q : ℝ × ℝ, 
          (p.2 = a * p.1 + 1) ∧ 
          (q.2 = a * q.1 + 1) ∧ 
          (p ≠ q) ∧ 
          (p.1^2 + p.2^2 + b * p.1 - p.2 = 1) ∧ 
          (q.1^2 + q.2^2 + b * q.1 - q.2 = 1) ∧ 
          (p.1 + p.2 = -q.1 - q.2)) : 
  a + b = 2 :=
sorry

end NUMINAMATH_GPT_line_curve_intersection_symmetric_l1440_144027


namespace NUMINAMATH_GPT_rectangle_constant_k_l1440_144023

theorem rectangle_constant_k (d : ℝ) (x : ℝ) (h_ratio : 4 * x = (4 / 3) * (3 * x)) (h_diagonal : d^2 = (4 * x)^2 + (3 * x)^2) : 
  ∃ k : ℝ, k = 12 / 25 ∧ (4 * x) * (3 * x) = k * d^2 := 
sorry

end NUMINAMATH_GPT_rectangle_constant_k_l1440_144023


namespace NUMINAMATH_GPT_original_price_l1440_144084

theorem original_price 
  (SP : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h1 : SP = 15)
  (h2 : gain_percent = 0.50)
  (h3 : SP = P * (1 + gain_percent)) :
  P = 10 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l1440_144084


namespace NUMINAMATH_GPT_rate_per_kg_mangoes_is_55_l1440_144036

def total_amount : ℕ := 1125
def rate_per_kg_grapes : ℕ := 70
def weight_grapes : ℕ := 9
def weight_mangoes : ℕ := 9

def cost_grapes := rate_per_kg_grapes * weight_grapes
def cost_mangoes := total_amount - cost_grapes

theorem rate_per_kg_mangoes_is_55 (rate_per_kg_mangoes : ℕ) (h : rate_per_kg_mangoes = cost_mangoes / weight_mangoes) : rate_per_kg_mangoes = 55 :=
by
  -- proof construction
  sorry

end NUMINAMATH_GPT_rate_per_kg_mangoes_is_55_l1440_144036


namespace NUMINAMATH_GPT_odd_square_divisors_l1440_144097

theorem odd_square_divisors (n : ℕ) (h_odd : n % 2 = 1) : 
  ∃ (f g : ℕ), (f > g) ∧ (∀ d, d ∣ (n * n) → d % 4 = 1 ↔ (0 < f)) ∧ (∀ d, d ∣ (n * n) → d % 4 = 3 ↔ (0 < g)) :=
by
  sorry

end NUMINAMATH_GPT_odd_square_divisors_l1440_144097


namespace NUMINAMATH_GPT_compute_value_of_expression_l1440_144051

theorem compute_value_of_expression :
  ∃ p q : ℝ, (3 * p^2 - 3 * q^2) / (p - q) = 5 ∧ 3 * p^2 - 5 * p - 14 = 0 ∧ 3 * q^2 - 5 * q - 14 = 0 :=
sorry

end NUMINAMATH_GPT_compute_value_of_expression_l1440_144051


namespace NUMINAMATH_GPT_range_a_l1440_144083

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2 * x else x - 1

theorem range_a (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 ≠ a^2 - 2 * a ∧ f x2 ≠ a^2 - 2 * a ∧ f x3 ≠ a^2 - 2 * a) ↔ (0 < a ∧ a < 1 ∨ 1 < a ∧ a < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_a_l1440_144083


namespace NUMINAMATH_GPT_evaluate_expression_l1440_144032

def f (x : ℕ) : ℕ := 3 * x - 4
def g (x : ℕ) : ℕ := x - 1

theorem evaluate_expression : f (1 + g 3) = 5 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1440_144032


namespace NUMINAMATH_GPT_find_d_l1440_144019

theorem find_d (d : ℝ) : (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1 :=
by
  { sorry }

end NUMINAMATH_GPT_find_d_l1440_144019


namespace NUMINAMATH_GPT_inscribed_square_neq_five_l1440_144002

theorem inscribed_square_neq_five (a b : ℝ) 
  (h1 : a - b = 1)
  (h2 : a * b = 1)
  (h3 : a + b = Real.sqrt 5) : a^2 + b^2 ≠ 5 :=
by sorry

end NUMINAMATH_GPT_inscribed_square_neq_five_l1440_144002


namespace NUMINAMATH_GPT_find_x_l1440_144037

theorem find_x (x : ℝ) (h : x ≠ 3) : (x^2 - 9) / (x - 3) = 3 * x → x = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_find_x_l1440_144037


namespace NUMINAMATH_GPT_proof_problem_l1440_144018

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ (a1 d : α), ∀ n : ℕ, a n = a1 + n * d

def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (n * (a 0 + a (n - 1))) / 2

variables {a : ℕ → α}

theorem proof_problem (h_arith_seq : is_arithmetic_sequence a)
    (h_S6_gt_S7 : sum_first_n_terms a 6 > sum_first_n_terms a 7)
    (h_S7_gt_S5 : sum_first_n_terms a 7 > sum_first_n_terms a 5) :
    (∃ d : α, d < 0) ∧ (∃ S11 : α, sum_first_n_terms a 11 > 0) :=
  sorry

end NUMINAMATH_GPT_proof_problem_l1440_144018


namespace NUMINAMATH_GPT_tank_capacity_l1440_144065

theorem tank_capacity (C : ℝ) (h1 : 0.40 * C = 0.90 * C - 36) : C = 72 := 
sorry

end NUMINAMATH_GPT_tank_capacity_l1440_144065


namespace NUMINAMATH_GPT_oprah_winfrey_band_weights_l1440_144001

theorem oprah_winfrey_band_weights :
  let weight_trombone := 10
  let weight_tuba := 20
  let weight_drum := 15
  let num_trumpets := 6
  let num_clarinets := 9
  let num_trombones := 8
  let num_tubas := 3
  let num_drummers := 2
  let total_weight := 245

  15 * x = total_weight - (num_trombones * weight_trombone + num_tubas * weight_tuba + num_drummers * weight_drum) 
  → x = 5 := by
  sorry

end NUMINAMATH_GPT_oprah_winfrey_band_weights_l1440_144001


namespace NUMINAMATH_GPT_C_share_of_profit_l1440_144076

-- Given conditions
def investment_A : ℕ := 8000
def investment_B : ℕ := 4000
def investment_C : ℕ := 2000
def total_profit : ℕ := 252000

-- Objective to prove that C's share of the profit is given by 36000
theorem C_share_of_profit : (total_profit / (investment_A / investment_C + investment_B / investment_C + 1)) = 36000 :=
by
  sorry

end NUMINAMATH_GPT_C_share_of_profit_l1440_144076


namespace NUMINAMATH_GPT_prob_B_hits_once_prob_hits_with_ABC_l1440_144049

section
variable (P_A P_B P_C : ℝ)
variable (hA : P_A = 1 / 2)
variable (hB : P_B = 1 / 3)
variable (hC : P_C = 1 / 4)

-- Part (Ⅰ): Probability of hitting the target exactly once when B shoots twice
theorem prob_B_hits_once : 
  (P_B * (1 - P_B) + (1 - P_B) * P_B) = 4 / 9 := 
by
  rw [hB]
  sorry

-- Part (Ⅱ): Probability of hitting the target when A, B, and C each shoot once
theorem prob_hits_with_ABC :
  (1 - ((1 - P_A) * (1 - P_B) * (1 - P_C))) = 3 / 4 := 
by
  rw [hA, hB, hC]
  sorry

end

end NUMINAMATH_GPT_prob_B_hits_once_prob_hits_with_ABC_l1440_144049


namespace NUMINAMATH_GPT_fraction_addition_simplest_form_l1440_144010

theorem fraction_addition_simplest_form :
  (7 / 12) + (3 / 8) = 23 / 24 :=
by
  -- Adding a sorry to skip the proof
  sorry

end NUMINAMATH_GPT_fraction_addition_simplest_form_l1440_144010


namespace NUMINAMATH_GPT_shopper_saves_more_l1440_144034

-- Definitions and conditions
def cover_price : ℝ := 30
def percent_discount : ℝ := 0.25
def dollar_discount : ℝ := 5
def first_discounted_price : ℝ := cover_price * (1 - percent_discount)
def second_discounted_price : ℝ := first_discounted_price - dollar_discount
def first_dollar_discounted_price : ℝ := cover_price - dollar_discount
def second_percent_discounted_price : ℝ := first_dollar_discounted_price * (1 - percent_discount)

def additional_savings : ℝ := second_percent_discounted_price - second_discounted_price

-- Theorem stating the shopper saves 125 cents more with 25% first
theorem shopper_saves_more : additional_savings = 1.25 := by
  sorry

end NUMINAMATH_GPT_shopper_saves_more_l1440_144034


namespace NUMINAMATH_GPT_num_girls_on_playground_l1440_144024

-- Definitions based on conditions
def total_students : ℕ := 20
def classroom_students := total_students / 4
def playground_students := total_students - classroom_students
def boys_playground := playground_students / 3
def girls_playground := playground_students - boys_playground

-- Theorem statement
theorem num_girls_on_playground : girls_playground = 10 :=
by
  -- Begin preparing proofs
  sorry

end NUMINAMATH_GPT_num_girls_on_playground_l1440_144024


namespace NUMINAMATH_GPT_find_positive_k_l1440_144033

noncomputable def cubic_roots (a b k : ℝ) : Prop :=
  (3 * a * a * a + 9 * a * a - 135 * a + k = 0) ∧
  (a * a * b = -45 / 2)

theorem find_positive_k :
  ∃ (a b : ℝ), ∃ (k : ℝ) (pos : k > 0), (cubic_roots a b k) ∧ (k = 525) :=
by
  sorry

end NUMINAMATH_GPT_find_positive_k_l1440_144033


namespace NUMINAMATH_GPT_find_b_l1440_144086

open Real

variables {A B C a b c : ℝ}

theorem find_b 
  (hA : A = π / 4) 
  (h1 : 2 * b * sin B - c * sin C = 2 * a * sin A) 
  (h_area : 1 / 2 * b * c * sin A = 3) : 
  b = 3 := 
sorry

end NUMINAMATH_GPT_find_b_l1440_144086


namespace NUMINAMATH_GPT_train_length_l1440_144072

theorem train_length (L : ℝ) (v : ℝ)
  (h1 : L = v * 36)
  (h2 : L + 25 = v * 39) :
  L = 300 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1440_144072


namespace NUMINAMATH_GPT_energy_savings_l1440_144055

theorem energy_savings (x y : ℝ) 
  (h1 : x = y + 27) 
  (h2 : x + 2.1 * y = 405) :
  x = 149 ∧ y = 122 :=
by
  sorry

end NUMINAMATH_GPT_energy_savings_l1440_144055


namespace NUMINAMATH_GPT_find_b_l1440_144096

def operation (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem find_b (b : ℤ) (h : operation 11 b = 110) : b = 12 := 
by
  sorry

end NUMINAMATH_GPT_find_b_l1440_144096


namespace NUMINAMATH_GPT_negation_proposition_l1440_144088

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_proposition_l1440_144088


namespace NUMINAMATH_GPT_johns_age_l1440_144013

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 70) : j = 20 := by
  sorry

end NUMINAMATH_GPT_johns_age_l1440_144013


namespace NUMINAMATH_GPT_star_3_2_l1440_144078

-- Definition of the operation
def star (a b : ℤ) : ℤ := a * b^3 - b^2 + 2

-- The proof problem
theorem star_3_2 : star 3 2 = 22 :=
by
  sorry

end NUMINAMATH_GPT_star_3_2_l1440_144078


namespace NUMINAMATH_GPT_total_distance_traveled_l1440_144087

theorem total_distance_traveled
  (bike_time_min : ℕ) (bike_rate_mph : ℕ)
  (jog_time_min : ℕ) (jog_rate_mph : ℕ)
  (total_time_min : ℕ)
  (h_bike_time : bike_time_min = 30)
  (h_bike_rate : bike_rate_mph = 6)
  (h_jog_time : jog_time_min = 45)
  (h_jog_rate : jog_rate_mph = 8)
  (h_total_time : total_time_min = 75) :
  (bike_rate_mph * bike_time_min / 60) + (jog_rate_mph * jog_time_min / 60) = 9 :=
by sorry

end NUMINAMATH_GPT_total_distance_traveled_l1440_144087


namespace NUMINAMATH_GPT_trigonometric_identity_l1440_144069

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = (1 / (Real.cos (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180))) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1440_144069


namespace NUMINAMATH_GPT_tail_wind_distance_l1440_144050

-- Definitions based on conditions
def speed_still_air : ℝ := 262.5
def t1 : ℝ := 3
def t2 : ℝ := 4

def effective_speed_tail_wind (w : ℝ) : ℝ := speed_still_air + w
def effective_speed_against_wind (w : ℝ) : ℝ := speed_still_air - w

theorem tail_wind_distance (w : ℝ) (d : ℝ) :
  effective_speed_tail_wind w * t1 = effective_speed_against_wind w * t2 →
  d = t1 * effective_speed_tail_wind w →
  d = 900 :=
by
  sorry

end NUMINAMATH_GPT_tail_wind_distance_l1440_144050


namespace NUMINAMATH_GPT_pascals_triangle_row_20_fifth_element_l1440_144005

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.div (Nat.factorial n) ((Nat.factorial k) * (Nat.factorial (n - k)))

-- State the theorem about Row 20, fifth element in Pascal's triangle
theorem pascals_triangle_row_20_fifth_element :
  binomial 20 4 = 4845 := 
by
  sorry

end NUMINAMATH_GPT_pascals_triangle_row_20_fifth_element_l1440_144005


namespace NUMINAMATH_GPT_order_of_abc_l1440_144052

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := 2 ^ (4 / 3)
noncomputable def c : ℝ := Real.log 1 / 3 / Real.log 2

theorem order_of_abc : c < a ∧ a < b :=
by {
  -- The proof would go here
  sorry
}

end NUMINAMATH_GPT_order_of_abc_l1440_144052


namespace NUMINAMATH_GPT_Maxwell_age_l1440_144085

theorem Maxwell_age :
  ∀ (sister_age maxwell_age : ℕ),
    (sister_age = 2) → 
    (maxwell_age + 2 = 2 * (sister_age + 2)) →
    (maxwell_age = 6) :=
by
  intros sister_age maxwell_age h1 h2
  -- Definitions and hypotheses come directly from conditions
  sorry

end NUMINAMATH_GPT_Maxwell_age_l1440_144085


namespace NUMINAMATH_GPT_andre_wins_first_scenario_dalva_wins_first_scenario_andre_wins_second_scenario_dalva_wins_second_scenario_l1440_144066

/-- In the first scenario, given the conditions that there are 
3 white balls and 1 black ball, and each person draws a ball 
in alphabetical order without replacement, prove that the 
probability that André wins the book is 1/4. -/
theorem andre_wins_first_scenario : 
  let total_balls := 4
  let black_balls := 1
  let probability := (black_balls : ℚ) / total_balls
  probability = 1 / 4 := 
by 
  sorry

/-- In the first scenario, given the conditions that there are 
3 white balls and 1 black ball, and each person draws a ball 
in alphabetical order without replacement, prove that the 
probability that Dalva wins the book is 1/4. -/
theorem dalva_wins_first_scenario : 
  let total_balls := 4
  let black_balls := 1
  let andre_white := (3 / 4 : ℚ)
  let bianca_white := (2 / 3 : ℚ)
  let carlos_white := (1 / 2 : ℚ)
  let probability := andre_white * bianca_white * carlos_white * (black_balls / (total_balls - 3))
  probability = 1 / 4 := 
by 
  sorry

/-- In the second scenario, given the conditions that there are 
6 white balls and 2 black balls, and each person draws a ball 
in alphabetical order until the first black ball is drawn, 
prove that the probability that André wins the book is 5/14. -/
theorem andre_wins_second_scenario : 
  let total_balls := 8
  let black_balls := 2
  let andre_first_black := (black_balls : ℚ) / total_balls
  let andre_fifth_black := (((6 / 8 : ℚ) * (5 / 7 : ℚ) * (4 / 6 : ℚ) * (3 / 5 : ℚ)) * black_balls / (total_balls - 4))
  let probability := andre_first_black + andre_fifth_black
  probability = 5 / 14 := 
by 
  sorry

/-- In the second scenario, given the conditions that there are 
6 white balls and 2 black balls, and each person draws a ball 
in alphabetical order until the first black ball is drawn, 
prove that the probability that Dalva wins the book is 1/7. -/
theorem dalva_wins_second_scenario : 
  let total_balls := 8
  let black_balls := 2
  let andre_white := (6 / 8 : ℚ)
  let bianca_white := (5 / 7 : ℚ)
  let carlos_white := (4 / 6 : ℚ)
  let dalva_black := (black_balls / (total_balls - 3))
  let probability := andre_white * bianca_white * carlos_white * dalva_black
  probability = 1 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_andre_wins_first_scenario_dalva_wins_first_scenario_andre_wins_second_scenario_dalva_wins_second_scenario_l1440_144066


namespace NUMINAMATH_GPT_find_m_if_a_b_parallel_l1440_144080

theorem find_m_if_a_b_parallel :
  ∃ m : ℝ, (∃ a : ℝ × ℝ, a = (-2, 1)) ∧ (∃ b : ℝ × ℝ, b = (1, m)) ∧ (m * -2 = 1) ∧ (m = -1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_m_if_a_b_parallel_l1440_144080


namespace NUMINAMATH_GPT_num_rectangular_arrays_with_36_chairs_l1440_144006

theorem num_rectangular_arrays_with_36_chairs :
  ∃ n : ℕ, (∀ r c : ℕ, r * c = 36 ∧ r ≥ 2 ∧ c ≥ 2 ↔ n = 7) :=
sorry

end NUMINAMATH_GPT_num_rectangular_arrays_with_36_chairs_l1440_144006


namespace NUMINAMATH_GPT_problem_statement_l1440_144070

variable (f : ℝ → ℝ)

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

theorem problem_statement (h_odd : is_odd f) (h_decr : is_decreasing f) (a b : ℝ) (h_ab : a + b < 0) :
  f (a + b) > 0 ∧ f a + f b > 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1440_144070


namespace NUMINAMATH_GPT_polynomial_evaluation_l1440_144000

noncomputable def evaluate_polynomial (x : ℝ) : ℝ :=
  x^3 - 3 * x^2 - 9 * x + 5

theorem polynomial_evaluation (x : ℝ) (h_pos : x > 0) (h_eq : x^2 - 3 * x - 9 = 0) :
  evaluate_polynomial x = 5 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l1440_144000


namespace NUMINAMATH_GPT_krishan_money_l1440_144003

-- Define the constants
def Ram : ℕ := 490
def ratio1 : ℕ := 7
def ratio2 : ℕ := 17

-- Defining the relationship
def ratio_RG (Ram Gopal : ℕ) : Prop := Ram / Gopal = ratio1 / ratio2
def ratio_GK (Gopal Krishan : ℕ) : Prop := Gopal / Krishan = ratio1 / ratio2

-- Define the problem
theorem krishan_money (R G K : ℕ) (h1 : R = Ram) (h2 : ratio_RG R G) (h3 : ratio_GK G K) : K = 2890 :=
by
  sorry

end NUMINAMATH_GPT_krishan_money_l1440_144003


namespace NUMINAMATH_GPT_no_such_natural_number_exists_l1440_144025

theorem no_such_natural_number_exists :
  ¬ ∃ n : ℕ, ∃ m : ℕ, 3^n + 2 * 17^n = m^2 :=
by sorry

end NUMINAMATH_GPT_no_such_natural_number_exists_l1440_144025


namespace NUMINAMATH_GPT_sum_of_star_tips_l1440_144092

/-- Given ten points that are evenly spaced on a circle and connected to form a 10-pointed star,
prove that the sum of the angle measurements of the ten tips of the star is 720 degrees. -/
theorem sum_of_star_tips (n : ℕ) (h : n = 10) :
  (10 * 72 = 720) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_star_tips_l1440_144092


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_pos_solve_equation2_neg_l1440_144012

theorem solve_equation1 (x : ℝ) (h : 2 * x^3 = 16) : x = 2 :=
sorry

theorem solve_equation2_pos (x : ℝ) (h : (x - 1)^2 = 4) : x = 3 :=
sorry

theorem solve_equation2_neg (x : ℝ) (h : (x - 1)^2 = 4) : x = -1 :=
sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_pos_solve_equation2_neg_l1440_144012


namespace NUMINAMATH_GPT_right_triangle_legs_l1440_144071

theorem right_triangle_legs (R r : ℝ) : 
  ∃ a b : ℝ, a = Real.sqrt (2 * (R^2 + r^2)) ∧ b = Real.sqrt (2 * (R^2 - r^2)) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_legs_l1440_144071
