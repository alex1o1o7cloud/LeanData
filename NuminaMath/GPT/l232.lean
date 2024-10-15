import Mathlib

namespace NUMINAMATH_GPT_problem_statement_l232_23228

def contrapositive {P Q : Prop} (h : P → Q) : ¬Q → ¬P :=
by sorry

def sufficient_but_not_necessary (P Q : Prop) : (P → Q) ∧ ¬(Q → P) :=
by sorry

def proposition_C (p q : Prop) : ¬(p ∧ q) → (¬p ∨ ¬q) :=
by sorry

def negate_exists (P : ℝ → Prop) : (∃ x : ℝ, P x) → ¬(∀ x : ℝ, ¬P x) :=
by sorry

theorem problem_statement : 
¬ (∀ (P Q : Prop), ¬(P ∧ Q) → (¬P ∨ ¬Q)) :=
by sorry

end NUMINAMATH_GPT_problem_statement_l232_23228


namespace NUMINAMATH_GPT_doughnuts_left_l232_23258

theorem doughnuts_left (dozen : ℕ) (total_initial : ℕ) (eaten : ℕ) (initial : total_initial = 2 * dozen) (d : dozen = 12) : total_initial - eaten = 16 :=
by
  rcases d
  rcases initial
  sorry

end NUMINAMATH_GPT_doughnuts_left_l232_23258


namespace NUMINAMATH_GPT_average_age_of_5_l232_23209

theorem average_age_of_5 (h1 : 19 * 15 = 285) (h2 : 9 * 16 = 144) (h3 : 15 = 71) :
    (285 - 144 - 71) / 5 = 14 :=
sorry

end NUMINAMATH_GPT_average_age_of_5_l232_23209


namespace NUMINAMATH_GPT_value_of_x_sq_plus_inv_x_sq_l232_23289

theorem value_of_x_sq_plus_inv_x_sq (x : ℝ) (h : x + 1/x = 1.5) : x^2 + (1/x)^2 = 0.25 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_x_sq_plus_inv_x_sq_l232_23289


namespace NUMINAMATH_GPT_find_value_of_expression_l232_23226

variables (a b c : ℝ)

theorem find_value_of_expression
  (h1 : a ^ 4 * b ^ 3 * c ^ 5 = 18)
  (h2 : a ^ 3 * b ^ 5 * c ^ 4 = 8) :
  a ^ 5 * b * c ^ 6 = 81 / 2 :=
sorry

end NUMINAMATH_GPT_find_value_of_expression_l232_23226


namespace NUMINAMATH_GPT_ram_gohul_work_days_l232_23220

theorem ram_gohul_work_days (ram_days gohul_days : ℕ) (H_ram: ram_days = 10) (H_gohul: gohul_days = 15): 
  (ram_days * gohul_days) / (ram_days + gohul_days) = 6 := 
by
  sorry

end NUMINAMATH_GPT_ram_gohul_work_days_l232_23220


namespace NUMINAMATH_GPT_teams_in_each_group_l232_23275

theorem teams_in_each_group (n : ℕ) :
  (2 * (n * (n - 1) / 2) + 3 * n = 56) → n = 7 :=
by
  sorry

end NUMINAMATH_GPT_teams_in_each_group_l232_23275


namespace NUMINAMATH_GPT_lizette_quiz_average_l232_23243

theorem lizette_quiz_average
  (Q1 Q2 : ℝ)
  (Q3 : ℝ := 92)
  (h : (Q1 + Q2 + Q3) / 3 = 94) :
  (Q1 + Q2) / 2 = 95 := by
sorry

end NUMINAMATH_GPT_lizette_quiz_average_l232_23243


namespace NUMINAMATH_GPT_unique_prime_triplets_l232_23215

theorem unique_prime_triplets (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  (p ∣ 1 + q^r) ∧ (q ∣ 1 + r^p) ∧ (r ∣ 1 + p^q) ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) := 
by
  sorry

end NUMINAMATH_GPT_unique_prime_triplets_l232_23215


namespace NUMINAMATH_GPT_max_n_value_l232_23271

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem max_n_value (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n → (2 * (n + 0.5) = a n + a (n + 1))) 
  (h2 : S a 63 = 2020) (h3 : a 2 < 3) : 63 ∈ { n : ℕ | S a n = 2020 } :=
sorry

end NUMINAMATH_GPT_max_n_value_l232_23271


namespace NUMINAMATH_GPT_leila_total_cakes_l232_23294

def cakes_monday : ℕ := 6
def cakes_friday : ℕ := 9
def cakes_saturday : ℕ := 3 * cakes_monday
def total_cakes : ℕ := cakes_monday + cakes_friday + cakes_saturday

theorem leila_total_cakes : total_cakes = 33 := by
  sorry

end NUMINAMATH_GPT_leila_total_cakes_l232_23294


namespace NUMINAMATH_GPT_a_4_value_l232_23257

-- Defining the polynomial (2x - 3)^6
def polynomial_expansion (x : ℝ) := (2 * x - 3) ^ 6

-- Given conditions polynomial expansion in terms of (x - 1)
def polynomial_coefficients (x : ℝ) (a : Fin 7 → ℝ) : ℝ :=
  a 0 + a 1 * (x - 1) + a 2 * (x - 1) ^ 2 + a 3 * (x - 1) ^ 3 + a 4 * (x - 1) ^ 4 +
  a 5 * (x - 1) ^ 5 + a 6 * (x - 1) ^ 6

-- The proof problem asking to show a_4 = 240
theorem a_4_value : 
  ∀ a : Fin 7 → ℝ, (∀ x : ℝ, polynomial_expansion x = polynomial_coefficients x a) → a 4 = 240 := by 
  sorry

end NUMINAMATH_GPT_a_4_value_l232_23257


namespace NUMINAMATH_GPT_part1_part2_l232_23212

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x * (Real.sin x + Real.cos x)) - 1 / 2

theorem part1 (α : ℝ) (hα1 : 0 < α ∧ α < Real.pi / 2) (hα2 : Real.sin α = Real.sqrt 2 / 2) :
  f α = 1 / 2 :=
sorry

theorem part2 :
  ∀ (k : ℤ), ∀ (x : ℝ),
  -((3 : ℝ) * Real.pi / 8) + k * Real.pi ≤ x ∧ x ≤ (Real.pi / 8) + k * Real.pi →
  MonotoneOn f (Set.Icc (-((3 : ℝ) * Real.pi / 8) + k * Real.pi) ((Real.pi / 8) + k * Real.pi)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l232_23212


namespace NUMINAMATH_GPT_find_c_minus_a_l232_23272

variable (a b c d e : ℝ)

-- Conditions
axiom avg_ab : (a + b) / 2 = 40
axiom avg_bc : (b + c) / 2 = 60
axiom avg_de : (d + e) / 2 = 80
axiom geom_mean : (a * b * d) = (b * c * e)

theorem find_c_minus_a : c - a = 40 := by
  sorry

end NUMINAMATH_GPT_find_c_minus_a_l232_23272


namespace NUMINAMATH_GPT_tom_initial_amount_l232_23273

variables (t s j : ℝ)

theorem tom_initial_amount :
  t + s + j = 1200 →
  t - 200 + 3 * s + 2 * j = 1800 →
  t = 400 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_tom_initial_amount_l232_23273


namespace NUMINAMATH_GPT_power_function_value_at_quarter_l232_23229

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x^α

theorem power_function_value_at_quarter (α : ℝ) (h : f 4 α = 1 / 2) : f (1 / 4) α = 2 := 
  sorry

end NUMINAMATH_GPT_power_function_value_at_quarter_l232_23229


namespace NUMINAMATH_GPT_range_of_m_l232_23283

variable {x y m : ℝ}

theorem range_of_m (hx : 0 < x) (hy : 0 < y) (h : 4 / x + 1 / y = 1) :
  x + y ≥ m^2 + m + 3 ↔ -3 ≤ m ∧ m ≤ 2 := sorry

end NUMINAMATH_GPT_range_of_m_l232_23283


namespace NUMINAMATH_GPT_ceil_sqrt_250_eq_16_l232_23206

theorem ceil_sqrt_250_eq_16 : ⌈Real.sqrt 250⌉ = 16 :=
by
  have h1 : (15 : ℝ) < Real.sqrt 250 := sorry
  have h2 : Real.sqrt 250 < 16 := sorry
  exact sorry

end NUMINAMATH_GPT_ceil_sqrt_250_eq_16_l232_23206


namespace NUMINAMATH_GPT_brenda_ends_with_12_skittles_l232_23221

def initial_skittles : ℕ := 7
def bought_skittles : ℕ := 8
def given_away_skittles : ℕ := 3

theorem brenda_ends_with_12_skittles :
  initial_skittles + bought_skittles - given_away_skittles = 12 := by
  sorry

end NUMINAMATH_GPT_brenda_ends_with_12_skittles_l232_23221


namespace NUMINAMATH_GPT_algebraic_expression_value_l232_23222

theorem algebraic_expression_value
  (x y : ℚ)
  (h : |2 * x - 3 * y + 1| + (x + 3 * y + 5)^2 = 0) :
  (-2 * x * y)^2 * (-y^2) * 6 * x * y^2 = 192 :=
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l232_23222


namespace NUMINAMATH_GPT_second_number_value_l232_23253

theorem second_number_value (A B C : ℝ) 
    (h1 : A + B + C = 98) 
    (h2 : A = (2/3) * B) 
    (h3 : C = (8/5) * B) : 
    B = 30 :=
by 
  sorry

end NUMINAMATH_GPT_second_number_value_l232_23253


namespace NUMINAMATH_GPT_remaining_distance_proof_l232_23290

-- Define the conditions
def pascal_current_speed : ℝ := 8
def pascal_reduced_speed : ℝ := pascal_current_speed - 4
def pascal_increased_speed : ℝ := pascal_current_speed * 1.5

-- Define the remaining distance in terms of the current speed and time taken
noncomputable def remaining_distance (T : ℝ) : ℝ := pascal_current_speed * T

-- Define the times with the increased and reduced speeds
noncomputable def time_with_increased_speed (T : ℝ) : ℝ := T - 16
noncomputable def time_with_reduced_speed (T : ℝ) : ℝ := T + 16

-- Define the distances using increased and reduced speeds
noncomputable def distance_increased_speed (T : ℝ) : ℝ := pascal_increased_speed * (time_with_increased_speed T)
noncomputable def distance_reduced_speed (T : ℝ) : ℝ := pascal_reduced_speed * (time_with_reduced_speed T)

-- Main theorem stating that the remaining distance is 256 miles
theorem remaining_distance_proof (T : ℝ) (ht_eq: pascal_current_speed * T = 256) : 
  remaining_distance T = 256 := by
  sorry

end NUMINAMATH_GPT_remaining_distance_proof_l232_23290


namespace NUMINAMATH_GPT_distance_after_time_l232_23217

noncomputable def Adam_speed := 12 -- speed in mph
noncomputable def Simon_speed := 6 -- speed in mph
noncomputable def time_when_100_miles_apart := 100 / 15 -- hours

theorem distance_after_time (x : ℝ) : 
  (Adam_speed * x)^2 + (Simon_speed * x)^2 = 100^2 ->
  x = time_when_100_miles_apart := 
by
  sorry

end NUMINAMATH_GPT_distance_after_time_l232_23217


namespace NUMINAMATH_GPT_rons_siblings_product_l232_23297

theorem rons_siblings_product
  (H_sisters : ℕ)
  (H_brothers : ℕ)
  (Ha_sisters : ℕ)
  (Ha_brothers : ℕ)
  (R_sisters : ℕ)
  (R_brothers : ℕ)
  (Harry_cond : H_sisters = 4 ∧ H_brothers = 6)
  (Harriet_cond : Ha_sisters = 4 ∧ Ha_brothers = 6)
  (Ron_cond_sisters : R_sisters = Ha_sisters)
  (Ron_cond_brothers : R_brothers = Ha_brothers + 2)
  : R_sisters * R_brothers = 32 := by
  sorry

end NUMINAMATH_GPT_rons_siblings_product_l232_23297


namespace NUMINAMATH_GPT_no_positive_integers_satisfy_condition_l232_23223

theorem no_positive_integers_satisfy_condition :
  ∀ (n : ℕ), n > 0 → ¬∃ (a b m : ℕ), a > 0 ∧ b > 0 ∧ m > 0 ∧ 
  (a + b * Real.sqrt n) ^ 2023 = Real.sqrt m + Real.sqrt (m + 2022) := by
  sorry

end NUMINAMATH_GPT_no_positive_integers_satisfy_condition_l232_23223


namespace NUMINAMATH_GPT_remove_blue_to_get_80_percent_red_l232_23249

-- Definitions from the conditions
def total_balls : ℕ := 150
def red_balls : ℕ := 60
def initial_blue_balls : ℕ := total_balls - red_balls
def desired_percentage_red : ℤ := 80

-- Lean statement of the proof problem
theorem remove_blue_to_get_80_percent_red :
  ∃ (x : ℕ), (x ≤ initial_blue_balls) ∧ (red_balls * 100 = desired_percentage_red * (total_balls - x)) → x = 75 := sorry

end NUMINAMATH_GPT_remove_blue_to_get_80_percent_red_l232_23249


namespace NUMINAMATH_GPT_geometric_seq_l232_23201

def seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧ (∀ n : ℕ, S (n + 1) + a n = S n + 5 * 4 ^ n)

theorem geometric_seq (a S : ℕ → ℝ) (h : seq a S) :
  ∃ r : ℝ, ∃ a1 : ℝ, (∀ n : ℕ, (a (n + 1) - 4 ^ (n + 1)) = r * (a n - 4 ^ n)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_l232_23201


namespace NUMINAMATH_GPT_diagonal_length_count_l232_23248

theorem diagonal_length_count :
  ∃ (x : ℕ) (h : (3 < x ∧ x < 22)), x = 18 := by
    sorry

end NUMINAMATH_GPT_diagonal_length_count_l232_23248


namespace NUMINAMATH_GPT_no_real_solution_l232_23225

theorem no_real_solution (P : ℝ → ℝ) (h_cont : Continuous P) (h_no_fixed_point : ∀ x : ℝ, P x ≠ x) : ∀ x : ℝ, P (P x) ≠ x :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_l232_23225


namespace NUMINAMATH_GPT_rationalize_denominator_l232_23281

theorem rationalize_denominator : (35 : ℝ) / Real.sqrt 15 = (7 / 3 : ℝ) * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l232_23281


namespace NUMINAMATH_GPT_johny_total_travel_distance_l232_23270

def TravelDistanceSouth : ℕ := 40
def TravelDistanceEast : ℕ := TravelDistanceSouth + 20
def TravelDistanceNorth : ℕ := 2 * TravelDistanceEast
def TravelDistanceWest : ℕ := TravelDistanceNorth / 2

theorem johny_total_travel_distance
    (hSouth : TravelDistanceSouth = 40)
    (hEast  : TravelDistanceEast = 60)
    (hNorth : TravelDistanceNorth = 120)
    (hWest  : TravelDistanceWest = 60)
    (totalDistance : ℕ := TravelDistanceSouth + TravelDistanceEast + TravelDistanceNorth + TravelDistanceWest) :
    totalDistance = 280 := by
  sorry

end NUMINAMATH_GPT_johny_total_travel_distance_l232_23270


namespace NUMINAMATH_GPT_find_x_l232_23256

theorem find_x : ∃ (x : ℚ), (3 * x - 5) / 7 = 15 ∧ x = 110 / 3 := by
  sorry

end NUMINAMATH_GPT_find_x_l232_23256


namespace NUMINAMATH_GPT_hyperbola_asymptotes_m_value_l232_23279

theorem hyperbola_asymptotes_m_value : 
    (∀ x y : ℝ, (x^2 / 144 - y^2 / 81 = 1) → (y = (3/4) * x ∨ y = -(3/4) * x)) := 
by sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_m_value_l232_23279


namespace NUMINAMATH_GPT_proportional_division_middle_part_l232_23247

theorem proportional_division_middle_part : 
  ∃ x : ℕ, x = 8 ∧ 5 * x = 40 ∧ 3 * x + 5 * x + 7 * x = 120 := 
by
  sorry

end NUMINAMATH_GPT_proportional_division_middle_part_l232_23247


namespace NUMINAMATH_GPT_number_of_true_propositions_is_2_l232_23237

-- Definitions for the propositions
def original_proposition (x : ℝ) : Prop := x > -3 → x > -6
def converse_proposition (x : ℝ) : Prop := x > -6 → x > -3
def inverse_proposition (x : ℝ) : Prop := x ≤ -3 → x ≤ -6
def contrapositive_proposition (x : ℝ) : Prop := x ≤ -6 → x ≤ -3

-- The theorem we need to prove
theorem number_of_true_propositions_is_2 :
  (∀ x, original_proposition x) ∧ (∀ x, contrapositive_proposition x) ∧ 
  ¬ (∀ x, converse_proposition x) ∧ ¬ (∀ x, inverse_proposition x) → 2 = 2 := 
sorry

end NUMINAMATH_GPT_number_of_true_propositions_is_2_l232_23237


namespace NUMINAMATH_GPT_unit_conversion_factor_l232_23284

theorem unit_conversion_factor (u : ℝ) (h₁ : u = 5) (h₂ : (u * 0.9)^2 = 20.25) : u = 5 → (1 : ℝ) = 0.9  :=
sorry

end NUMINAMATH_GPT_unit_conversion_factor_l232_23284


namespace NUMINAMATH_GPT_find_marksman_hit_rate_l232_23285

-- Define the conditions
def independent_shots (p : ℝ) (n : ℕ) : Prop :=
  0 ≤ p ∧ p ≤ 1 ∧ (n ≥ 1)

def hit_probability (p : ℝ) (n : ℕ) : ℝ :=
  1 - (1 - p) ^ n

-- Stating the proof problem in Lean
theorem find_marksman_hit_rate (p : ℝ) (n : ℕ) 
  (h_independent : independent_shots p n) 
  (h_prob : hit_probability p n = 80 / 81) : 
  p = 2 / 3 :=
sorry

end NUMINAMATH_GPT_find_marksman_hit_rate_l232_23285


namespace NUMINAMATH_GPT_mixed_bead_cost_per_box_l232_23278

-- Definitions based on given conditions
def red_bead_cost : ℝ := 1.30
def yellow_bead_cost : ℝ := 2.00
def total_boxes : ℕ := 10
def red_boxes_used : ℕ := 4
def yellow_boxes_used : ℕ := 4

-- Theorem statement
theorem mixed_bead_cost_per_box :
  ((red_boxes_used * red_bead_cost) + (yellow_boxes_used * yellow_bead_cost)) / total_boxes = 1.32 :=
  by sorry

end NUMINAMATH_GPT_mixed_bead_cost_per_box_l232_23278


namespace NUMINAMATH_GPT_max_value_64_l232_23216

-- Define the types and values of gemstones
structure Gemstone where
  weight : ℕ
  value : ℕ

-- Introduction of the three types of gemstones
def gem1 : Gemstone := ⟨3, 9⟩
def gem2 : Gemstone := ⟨5, 16⟩
def gem3 : Gemstone := ⟨2, 5⟩

-- Maximum weight Janet can carry
def max_weight := 20

-- Problem statement: Proving the maximum value Janet can carry is $64
theorem max_value_64 (n1 n2 n3 : ℕ) (h1 : n1 ≥ 15) (h2 : n2 ≥ 15) (h3 : n3 ≥ 15) 
  (weight_limit : n1 * gem1.weight + n2 * gem2.weight + n3 * gem3.weight ≤ max_weight) : 
  n1 * gem1.value + n2 * gem2.value + n3 * gem3.value ≤ 64 :=
sorry

end NUMINAMATH_GPT_max_value_64_l232_23216


namespace NUMINAMATH_GPT_total_granola_bars_l232_23295

-- Problem conditions
def oatmeal_raisin_bars : ℕ := 6
def peanut_bars : ℕ := 8

-- Statement to prove
theorem total_granola_bars : oatmeal_raisin_bars + peanut_bars = 14 := 
by 
  sorry

end NUMINAMATH_GPT_total_granola_bars_l232_23295


namespace NUMINAMATH_GPT_option_one_correct_l232_23250

theorem option_one_correct (x : ℝ) : 
  (x ≠ 0 → x + |x| > 0) ∧ ¬((x + |x| > 0) → x ≠ 0) := 
by
  sorry

end NUMINAMATH_GPT_option_one_correct_l232_23250


namespace NUMINAMATH_GPT_hidden_dots_sum_l232_23276

-- Lean 4 equivalent proof problem definition
theorem hidden_dots_sum (d1 d2 d3 d4 : ℕ)
    (h1 : d1 ≠ d2 ∧ d1 + d2 = 7)
    (h2 : d3 ≠ d4 ∧ d3 + d4 = 7)
    (h3 : d1 = 2 ∨ d1 = 4 ∨ d2 = 2 ∨ d2 = 4)
    (h4 : d3 + 4 = 7) :
    d1 + 7 + 7 + d3 = 24 :=
sorry

end NUMINAMATH_GPT_hidden_dots_sum_l232_23276


namespace NUMINAMATH_GPT_table_height_l232_23274

theorem table_height
  (l d h : ℤ)
  (h_eq1 : l + h - d = 36)
  (h_eq2 : 2 * l + h = 46)
  (l_eq_d : l = d) :
  h = 36 :=
by
  sorry

end NUMINAMATH_GPT_table_height_l232_23274


namespace NUMINAMATH_GPT_kali_height_now_l232_23238

variable (K_initial J_initial : ℝ)
variable (K_growth J_growth : ℝ)
variable (J_current : ℝ)

theorem kali_height_now :
  J_initial = K_initial →
  J_growth = (2 / 3) * 0.3 * K_initial →
  K_growth = 0.3 * K_initial →
  J_current = 65 →
  J_current = J_initial + J_growth →
  K_current = K_initial + K_growth →
  K_current = 70.42 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_kali_height_now_l232_23238


namespace NUMINAMATH_GPT_coins_in_bag_l232_23260

theorem coins_in_bag (x : ℝ) (h : x + 0.5 * x + 0.25 * x = 140) : x = 80 :=
by sorry

end NUMINAMATH_GPT_coins_in_bag_l232_23260


namespace NUMINAMATH_GPT_time_to_overflow_equals_correct_answer_l232_23299

-- Definitions based on conditions
def pipeA_fill_time : ℚ := 32
def pipeB_fill_time : ℚ := pipeA_fill_time / 5

-- Derived rates from the conditions
def pipeA_rate : ℚ := 1 / pipeA_fill_time
def pipeB_rate : ℚ := 1 / pipeB_fill_time
def combined_rate : ℚ := pipeA_rate + pipeB_rate

-- The time to overflow when both pipes are filling the tank simultaneously
def time_to_overflow : ℚ := 1 / combined_rate

-- The statement we are going to prove
theorem time_to_overflow_equals_correct_answer : time_to_overflow = 16 / 3 :=
by sorry

end NUMINAMATH_GPT_time_to_overflow_equals_correct_answer_l232_23299


namespace NUMINAMATH_GPT_subtraction_of_negatives_l232_23202

theorem subtraction_of_negatives : (-1) - (-4) = 3 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_subtraction_of_negatives_l232_23202


namespace NUMINAMATH_GPT_solve_inequality_l232_23233

theorem solve_inequality (x : ℝ) : 2 - x < 1 → x > 1 := 
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l232_23233


namespace NUMINAMATH_GPT_total_rooms_in_hotel_l232_23236

def first_wing_floors : ℕ := 9
def first_wing_halls_per_floor : ℕ := 6
def first_wing_rooms_per_hall : ℕ := 32

def second_wing_floors : ℕ := 7
def second_wing_halls_per_floor : ℕ := 9
def second_wing_rooms_per_hall : ℕ := 40

def third_wing_floors : ℕ := 12
def third_wing_halls_per_floor : ℕ := 4
def third_wing_rooms_per_hall : ℕ := 50

def first_wing_total_rooms : ℕ := 
  first_wing_floors * first_wing_halls_per_floor * first_wing_rooms_per_hall

def second_wing_total_rooms : ℕ := 
  second_wing_floors * second_wing_halls_per_floor * second_wing_rooms_per_hall

def third_wing_total_rooms : ℕ := 
  third_wing_floors * third_wing_halls_per_floor * third_wing_rooms_per_hall

theorem total_rooms_in_hotel : 
  first_wing_total_rooms + second_wing_total_rooms + third_wing_total_rooms = 6648 := 
by 
  sorry

end NUMINAMATH_GPT_total_rooms_in_hotel_l232_23236


namespace NUMINAMATH_GPT_percent_of_employed_people_who_are_females_l232_23219

theorem percent_of_employed_people_who_are_females (p_employed p_employed_males : ℝ) 
  (h1 : p_employed = 64) (h2 : p_employed_males = 48) : 
  100 * (p_employed - p_employed_males) / p_employed = 25 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_employed_people_who_are_females_l232_23219


namespace NUMINAMATH_GPT_minimum_racing_stripes_l232_23235

variable 
  (totalCars : ℕ) (carsWithoutAirConditioning : ℕ) 
  (maxCarsWithAirConditioningWithoutStripes : ℕ)

-- Defining specific problem conditions
def conditions (totalCars carsWithoutAirConditioning maxCarsWithAirConditioningWithoutStripes : ℕ) : Prop :=
  totalCars = 100 ∧ 
  carsWithoutAirConditioning = 37 ∧ 
  maxCarsWithAirConditioningWithoutStripes = 59

-- The statement to be proved
theorem minimum_racing_stripes (h : conditions totalCars carsWithoutAirConditioning maxCarsWithAirConditioningWithoutStripes) :
   exists (R : ℕ ), R = 4 :=
by
  rcases h with ⟨h1, h2, h3⟩
  sorry

end NUMINAMATH_GPT_minimum_racing_stripes_l232_23235


namespace NUMINAMATH_GPT_total_tape_length_is_230_l232_23205

def tape_length (n : ℕ) (len_piece : ℕ) (overlap : ℕ) : ℕ :=
  len_piece + (n - 1) * (len_piece - overlap)

theorem total_tape_length_is_230 :
  tape_length 15 20 5 = 230 := 
    sorry

end NUMINAMATH_GPT_total_tape_length_is_230_l232_23205


namespace NUMINAMATH_GPT_fliers_left_l232_23291

theorem fliers_left (total : ℕ) (morning_fraction afternoon_fraction : ℚ) 
  (h1 : total = 1000)
  (h2 : morning_fraction = 1/5)
  (h3 : afternoon_fraction = 1/4) :
  let morning_sent := total * morning_fraction
  let remaining_after_morning := total - morning_sent
  let afternoon_sent := remaining_after_morning * afternoon_fraction
  let remaining_after_afternoon := remaining_after_morning - afternoon_sent
  remaining_after_afternoon = 600 :=
by
  sorry

end NUMINAMATH_GPT_fliers_left_l232_23291


namespace NUMINAMATH_GPT_solve_system_eq_solve_system_ineq_l232_23282

-- For the system of equations:
theorem solve_system_eq (x y : ℝ) (h1 : x + 2 * y = 7) (h2 : 3 * x + y = 6) : x = 1 ∧ y = 3 :=
sorry

-- For the system of inequalities:
theorem solve_system_ineq (x : ℝ) (h1 : 2 * (x - 1) + 1 > -3) (h2 : x - 1 ≤ (1 + x) / 3) : -1 < x ∧ x ≤ 2 :=
sorry

end NUMINAMATH_GPT_solve_system_eq_solve_system_ineq_l232_23282


namespace NUMINAMATH_GPT_real_solutions_infinite_l232_23259

theorem real_solutions_infinite : 
  ∃ (S : Set ℝ), (∀ x ∈ S, - (x^2 - 4) ≥ 0) ∧ S.Infinite :=
sorry

end NUMINAMATH_GPT_real_solutions_infinite_l232_23259


namespace NUMINAMATH_GPT_find_m_l232_23204

open Real

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) (m : ℝ) : ℝ :=
  2 * cos (ω * x + ϕ) + m

theorem find_m (ω ϕ : ℝ) (hω : 0 < ω)
  (symmetry : ∀ t : ℝ,  f (π / 4 - t) ω ϕ m = f t ω ϕ m)
  (f_π_8 : f (π / 8) ω ϕ m = -1) :
  m = -3 ∨ m = 1 := 
sorry

end NUMINAMATH_GPT_find_m_l232_23204


namespace NUMINAMATH_GPT_AplusBplusC_4_l232_23232

theorem AplusBplusC_4 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 1 ∧ Nat.gcd a c = 1 ∧ (a^2 + a * b + b^2 = c^2) ∧ (a + b + c = 4) :=
by
  sorry

end NUMINAMATH_GPT_AplusBplusC_4_l232_23232


namespace NUMINAMATH_GPT_LeRoy_should_pay_Bernardo_l232_23296

theorem LeRoy_should_pay_Bernardo 
    (initial_loan : ℕ := 100)
    (LeRoy_gas_expense : ℕ := 300)
    (LeRoy_food_expense : ℕ := 200)
    (Bernardo_accommodation_expense : ℕ := 500)
    (total_expense := LeRoy_gas_expense + LeRoy_food_expense + Bernardo_accommodation_expense)
    (shared_expense := total_expense / 2)
    (LeRoy_total_responsibility := shared_expense + initial_loan)
    (LeRoy_needs_to_pay := LeRoy_total_responsibility - (LeRoy_gas_expense + LeRoy_food_expense)) :
    LeRoy_needs_to_pay = 100 := 
by
    sorry

end NUMINAMATH_GPT_LeRoy_should_pay_Bernardo_l232_23296


namespace NUMINAMATH_GPT_max_ab_real_positive_l232_23269

theorem max_ab_real_positive (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 2) : 
  ab ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_ab_real_positive_l232_23269


namespace NUMINAMATH_GPT_trigonometric_identity_l232_23287

theorem trigonometric_identity :
  (1 / Real.cos 80) - (Real.sqrt 3 / Real.cos 10) = 4 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l232_23287


namespace NUMINAMATH_GPT_investment_calculation_l232_23207

noncomputable def initial_investment (final_amount : ℝ) (years : ℕ) (interest_rate : ℝ) : ℝ :=
  final_amount / ((1 + interest_rate / 100) ^ years)

theorem investment_calculation :
  initial_investment 504.32 3 12 = 359 :=
by
  sorry

end NUMINAMATH_GPT_investment_calculation_l232_23207


namespace NUMINAMATH_GPT_determine_h_l232_23245

noncomputable def h (x : ℝ) : ℝ := -4 * x^5 - 3 * x^3 - 4 * x^2 + 12 * x + 2

theorem determine_h (x : ℝ) :
  4 * x^5 + 5 * x^3 - 3 * x + h x = 2 * x^3 - 4 * x^2 + 9 * x + 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_h_l232_23245


namespace NUMINAMATH_GPT_evaluate_expression_l232_23246

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l232_23246


namespace NUMINAMATH_GPT_unique_solution_triple_l232_23234

def satisfies_system (x y z : ℝ) :=
  x^3 = 3 * x - 12 * y + 50 ∧
  y^3 = 12 * y + 3 * z - 2 ∧
  z^3 = 27 * z + 27 * x

theorem unique_solution_triple (x y z : ℝ) :
  satisfies_system x y z ↔ (x = 2 ∧ y = 4 ∧ z = 6) :=
by sorry

end NUMINAMATH_GPT_unique_solution_triple_l232_23234


namespace NUMINAMATH_GPT_bobby_pizzas_l232_23288

theorem bobby_pizzas (B : ℕ) (h_slices : (1 / 4 : ℝ) * B = 3) (h_slices_per_pizza : 6 > 0) :
  B / 6 = 2 := by
  sorry

end NUMINAMATH_GPT_bobby_pizzas_l232_23288


namespace NUMINAMATH_GPT_solve_inequality_l232_23241

theorem solve_inequality (x : ℝ) :
  -2 * x^2 - x + 6 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3 / 2 :=
sorry

end NUMINAMATH_GPT_solve_inequality_l232_23241


namespace NUMINAMATH_GPT_factorize_l232_23240

theorem factorize (a : ℝ) : 5*a^3 - 125*a = 5*a*(a + 5)*(a - 5) :=
sorry

end NUMINAMATH_GPT_factorize_l232_23240


namespace NUMINAMATH_GPT_correct_calculation_result_l232_23239

theorem correct_calculation_result (x : ℤ) (h : 4 * x + 16 = 32) : (x / 4) + 16 = 17 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_result_l232_23239


namespace NUMINAMATH_GPT_find_number_l232_23277

theorem find_number (number : ℚ) 
  (H1 : 8 * 60 = 480)
  (H2 : number / 6 = 16 / 480) :
  number = 1 / 5 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l232_23277


namespace NUMINAMATH_GPT_total_cans_needed_l232_23200

-- Definitions
def cans_per_box : ℕ := 4
def number_of_boxes : ℕ := 203

-- Statement of the problem
theorem total_cans_needed : cans_per_box * number_of_boxes = 812 := 
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_total_cans_needed_l232_23200


namespace NUMINAMATH_GPT_more_flour_than_sugar_l232_23208

variable (total_flour : ℕ) (total_sugar : ℕ)
variable (flour_added : ℕ)

def additional_flour_needed (total_flour flour_added : ℕ) : ℕ :=
  total_flour - flour_added

theorem more_flour_than_sugar :
  additional_flour_needed 10 7 - 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_more_flour_than_sugar_l232_23208


namespace NUMINAMATH_GPT_range_of_sum_l232_23242

theorem range_of_sum (a b : ℝ) (h1 : -2 < a) (h2 : a < -1) (h3 : -1 < b) (h4 : b < 0) : 
  -3 < a + b ∧ a + b < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_sum_l232_23242


namespace NUMINAMATH_GPT_sum_of_coefficients_l232_23252

theorem sum_of_coefficients
  (d : ℝ)
  (g h : ℝ)
  (h1 : (8 * d^2 - 4 * d + g) * (5 * d^2 + h * d - 10) = 40 * d^4 - 75 * d^3 - 90 * d^2 + 5 * d + 20) :
  g + h = 15.5 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l232_23252


namespace NUMINAMATH_GPT_parallel_vectors_x_val_l232_23251

open Real

theorem parallel_vectors_x_val (x : ℝ) :
  let a : ℝ × ℝ := (3, 4)
  let b : ℝ × ℝ := (x, 1/2)
  a.1 * b.2 = a.2 * b.1 →
  x = 3/8 := 
by
  intro h
  -- Use this line if you need to skip the proof
  sorry

end NUMINAMATH_GPT_parallel_vectors_x_val_l232_23251


namespace NUMINAMATH_GPT_range_of_set_l232_23214

theorem range_of_set (a b c : ℕ) (h1 : (a + b + c) / 3 = 6) (h2 : b = 6) (h3 : a = 2) : max a (max b c) - min a (min b c) = 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_set_l232_23214


namespace NUMINAMATH_GPT_circles_tangent_iff_l232_23280

noncomputable def C1 := { p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1 }
noncomputable def C2 (m: ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 - 8 * p.1 + 8 * p.2 + m = 0 }

theorem circles_tangent_iff (m: ℝ) : (∀ p ∈ C1, p ∈ C2 m → False) ↔ (m = -4 ∨ m = 16) := 
sorry

end NUMINAMATH_GPT_circles_tangent_iff_l232_23280


namespace NUMINAMATH_GPT_unique_n_in_range_satisfying_remainders_l232_23266

theorem unique_n_in_range_satisfying_remainders : 
  ∃! (n : ℤ) (k r : ℤ), 150 < n ∧ n < 250 ∧ 0 <= r ∧ r <= 6 ∧ n = 7 * k + r ∧ n = 9 * r + r :=
sorry

end NUMINAMATH_GPT_unique_n_in_range_satisfying_remainders_l232_23266


namespace NUMINAMATH_GPT_max_matrix_det_l232_23265

noncomputable def matrix_det (θ : ℝ) : ℝ :=
  by
    let M := ![
      ![1, 1, 1],
      ![1, 1 + Real.sin θ ^ 2, 1],
      ![1 + Real.cos θ ^ 2, 1, 1]
    ]
    exact Matrix.det M

theorem max_matrix_det : ∃ θ : ℝ, matrix_det θ = 3/4 :=
  sorry

end NUMINAMATH_GPT_max_matrix_det_l232_23265


namespace NUMINAMATH_GPT_kelsey_video_count_l232_23210

variable (E U K : ℕ)

noncomputable def total_videos : ℕ := 411
noncomputable def ekon_videos : ℕ := E
noncomputable def uma_videos : ℕ := E + 17
noncomputable def kelsey_videos : ℕ := E + 43

theorem kelsey_video_count (E U K : ℕ) 
  (h1 : total_videos = ekon_videos + uma_videos + kelsey_videos)
  (h2 : uma_videos = ekon_videos + 17)
  (h3 : kelsey_videos = ekon_videos + 43)
  : kelsey_videos = 160 := 
sorry

end NUMINAMATH_GPT_kelsey_video_count_l232_23210


namespace NUMINAMATH_GPT_solve_system_of_equations_l232_23203

theorem solve_system_of_equations :
  ∃ (x y : ℝ), (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ∧ (x = 5) ∧ (y = 5) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l232_23203


namespace NUMINAMATH_GPT_xy_pos_iff_div_pos_ab_leq_mean_sq_l232_23298

-- Definition for question 1
theorem xy_pos_iff_div_pos (x y : ℝ) : 
  (x * y > 0) ↔ (x / y > 0) :=
sorry

-- Definition for question 3
theorem ab_leq_mean_sq (a b : ℝ) : 
  a * b ≤ ((a + b) / 2) ^ 2 :=
sorry

end NUMINAMATH_GPT_xy_pos_iff_div_pos_ab_leq_mean_sq_l232_23298


namespace NUMINAMATH_GPT_manuscript_page_count_l232_23227

-- Define the main statement
theorem manuscript_page_count
  (P : ℕ)
  (cost_per_page : ℕ := 10)
  (rev1_pages : ℕ := 30)
  (rev2_pages : ℕ := 20)
  (total_cost : ℕ := 1350)
  (cost_rev1 : ℕ := 15)
  (cost_rev2 : ℕ := 20) 
  (remaining_pages_cost : ℕ := 10 * (P - (rev1_pages + rev2_pages))) :
  (remaining_pages_cost + rev1_pages * cost_rev1 + rev2_pages * cost_rev2 = total_cost)
  → P = 100 :=
by
  sorry

end NUMINAMATH_GPT_manuscript_page_count_l232_23227


namespace NUMINAMATH_GPT_power_function_inverse_l232_23230

theorem power_function_inverse (f : ℝ → ℝ) (h₁ : f 2 = (Real.sqrt 2) / 2) : f⁻¹ 2 = 1 / 4 :=
by
  -- Lean proof will be filled here
  sorry

end NUMINAMATH_GPT_power_function_inverse_l232_23230


namespace NUMINAMATH_GPT_cross_ratio_eq_one_implies_equal_points_l232_23255

-- Definitions corresponding to the points and hypothesis.
variable {A B C D : ℝ}
variable (h_line : collinear ℝ A B C D) (h_cross_ratio : cross_ratio A B C D = 1)

-- The theorem statement based on the given problem and solution.
theorem cross_ratio_eq_one_implies_equal_points :
  A = B ∨ C = D :=
sorry

end NUMINAMATH_GPT_cross_ratio_eq_one_implies_equal_points_l232_23255


namespace NUMINAMATH_GPT_smallest_possible_area_l232_23218

noncomputable def smallest_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 200 ∧ (l = 30 ∨ w = 30) then l * w else 0

theorem smallest_possible_area : ∃ l w : ℕ, 2 * l + 2 * w = 200 ∧ (l = 30 ∨ w = 30) ∧ smallest_area l w = 2100 := by
  sorry

end NUMINAMATH_GPT_smallest_possible_area_l232_23218


namespace NUMINAMATH_GPT_total_bricks_calculation_l232_23263

def bricks_in_row : Nat := 30
def rows_in_wall : Nat := 50
def number_of_walls : Nat := 2
def total_bricks_for_both_walls : Nat := 3000

theorem total_bricks_calculation (h1 : bricks_in_row = 30) 
                                      (h2 : rows_in_wall = 50) 
                                      (h3 : number_of_walls = 2) : 
                                      bricks_in_row * rows_in_wall * number_of_walls = total_bricks_for_both_walls :=
by
  sorry

end NUMINAMATH_GPT_total_bricks_calculation_l232_23263


namespace NUMINAMATH_GPT_find_vector_from_origin_to_line_l232_23254

theorem find_vector_from_origin_to_line :
  ∃ t : ℝ, (3 * t + 1, 2 * t + 3) = (16, 32 / 3) ∧
  ∃ k : ℝ, (16, 32 / 3) = (3 * k, 2 * k) :=
sorry

end NUMINAMATH_GPT_find_vector_from_origin_to_line_l232_23254


namespace NUMINAMATH_GPT_hyperbola_equation_l232_23244

-- Conditions
def center_origin (P : ℝ × ℝ) : Prop := P = (0, 0)
def focus_at (F : ℝ × ℝ) : Prop := F = (0, Real.sqrt 3)
def vertex_distance (d : ℝ) : Prop := d = Real.sqrt 3 - 1

-- Statement
theorem hyperbola_equation
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (d : ℝ)
  (h_center : center_origin center)
  (h_focus : focus_at focus)
  (h_vert_dist : vertex_distance d) :
  y^2 - (x^2 / 2) = 1 := 
sorry

end NUMINAMATH_GPT_hyperbola_equation_l232_23244


namespace NUMINAMATH_GPT_brokerage_percentage_l232_23262

theorem brokerage_percentage
  (f : ℝ) (d : ℝ) (c : ℝ) 
  (hf : f = 100)
  (hd : d = 0.08)
  (hc : c = 92.2)
  (h_disc_price : f - f * d = 92) :
  (c - (f - f * d)) / f * 100 = 0.2 := 
by
  sorry

end NUMINAMATH_GPT_brokerage_percentage_l232_23262


namespace NUMINAMATH_GPT_factorize_polynomial_value_of_x_cubed_l232_23292

-- Problem 1: Factorization
theorem factorize_polynomial (x : ℝ) : 42 * x^2 - 33 * x + 6 = 3 * (2 * x - 1) * (7 * x - 2) :=
sorry

-- Problem 2: Given condition and proof of x^3 + 1/x^3
theorem value_of_x_cubed (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^3 + 1 / x^3 = 18 :=
sorry

end NUMINAMATH_GPT_factorize_polynomial_value_of_x_cubed_l232_23292


namespace NUMINAMATH_GPT_trigonometric_identity_l232_23213

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) : 
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l232_23213


namespace NUMINAMATH_GPT_problem_1_solution_problem_2_solution_l232_23286

-- Definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 3) - abs (x - a)

-- Proof problem for question 1
theorem problem_1_solution (x : ℝ) : f x 2 ≤ -1/2 ↔ x ≥ 11/4 :=
by
  sorry

-- Proof problem for question 2
theorem problem_2_solution (a : ℝ) : (∀ x : ℝ, f x a ≥ a) ↔ a ∈ Set.Iic (3/2) :=
by
  sorry

end NUMINAMATH_GPT_problem_1_solution_problem_2_solution_l232_23286


namespace NUMINAMATH_GPT_emily_expenditure_l232_23261

-- Define the conditions
def price_per_flower : ℕ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2

-- Total flowers bought
def total_flowers (roses daisies : ℕ) : ℕ :=
  roses + daisies

-- Define the cost function
def cost (flowers price_per_flower : ℕ) : ℕ :=
  flowers * price_per_flower

-- Theorem to prove the total expenditure
theorem emily_expenditure : 
  cost (total_flowers roses_bought daisies_bought) price_per_flower = 12 :=
by
  sorry

end NUMINAMATH_GPT_emily_expenditure_l232_23261


namespace NUMINAMATH_GPT_platform_length_l232_23267

theorem platform_length (train_length : ℝ) (time_cross_pole : ℝ) (time_cross_platform : ℝ) (speed : ℝ) 
  (h1 : train_length = 300) 
  (h2 : time_cross_pole = 18) 
  (h3 : time_cross_platform = 54)
  (h4 : speed = train_length / time_cross_pole) :
  train_length + (speed * time_cross_platform) - train_length = 600 := 
by
  sorry

end NUMINAMATH_GPT_platform_length_l232_23267


namespace NUMINAMATH_GPT_cone_volume_l232_23231

theorem cone_volume (V_f : ℝ) (A1 A2 : ℝ) (V : ℝ)
  (h1 : V_f = 78)
  (h2 : A1 = 9 * A2) :
  V = 81 :=
sorry

end NUMINAMATH_GPT_cone_volume_l232_23231


namespace NUMINAMATH_GPT_tom_total_amount_after_saving_l232_23211

theorem tom_total_amount_after_saving :
  let hourly_rate := 6.50
  let work_hours := 31
  let saving_rate := 0.10
  let total_earnings := hourly_rate * work_hours
  let amount_set_aside := total_earnings * saving_rate
  let amount_for_purchases := total_earnings - amount_set_aside
  amount_for_purchases = 181.35 :=
by
  sorry

end NUMINAMATH_GPT_tom_total_amount_after_saving_l232_23211


namespace NUMINAMATH_GPT_number_of_triangles_with_perimeter_20_l232_23268

-- Declare the condition: number of triangles with integer side lengths and perimeter of 20
def integerTrianglesWithPerimeter (n : ℕ) : ℕ :=
  (Finset.range (n/2 + 1)).card

/-- Prove that the number of triangles with integer side lengths and a perimeter of 20 is 8. -/
theorem number_of_triangles_with_perimeter_20 : integerTrianglesWithPerimeter 20 = 8 := 
  sorry

end NUMINAMATH_GPT_number_of_triangles_with_perimeter_20_l232_23268


namespace NUMINAMATH_GPT_direct_proportion_function_l232_23293

theorem direct_proportion_function (m : ℝ) (h : ∀ x : ℝ, -2*x + m = k*x → m = 0) : m = 0 :=
sorry

end NUMINAMATH_GPT_direct_proportion_function_l232_23293


namespace NUMINAMATH_GPT_check_sufficient_condition_for_eq_l232_23264

theorem check_sufficient_condition_for_eq (a b c : ℤ) (h : a = c - 1 ∧ b = a - 1) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 = 1 := 
by
  sorry

end NUMINAMATH_GPT_check_sufficient_condition_for_eq_l232_23264


namespace NUMINAMATH_GPT_solution_largest_a_exists_polynomial_l232_23224

def largest_a_exists_polynomial : Prop :=
  ∃ (P : ℝ → ℝ) (a b c d e : ℝ),
    (∀ x, P x = a * x^4 + b * x^3 + c * x^2 + d * x + e) ∧
    (∀ x, -1 ≤ x ∧ x ≤ 1 → 0 ≤ P x ∧ P x ≤ 1) ∧
    a = 4

theorem solution_largest_a_exists_polynomial : largest_a_exists_polynomial :=
  sorry

end NUMINAMATH_GPT_solution_largest_a_exists_polynomial_l232_23224
