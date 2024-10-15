import Mathlib

namespace NUMINAMATH_GPT_min_value_l1371_137122

theorem min_value (a : ℝ) (h : a > 1) : a + 1 / (a - 1) ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_value_l1371_137122


namespace NUMINAMATH_GPT_lunch_cost_before_tax_and_tip_l1371_137115

theorem lunch_cost_before_tax_and_tip (C : ℝ) (h1 : 1.10 * C = 110) : C = 100 := by
  sorry

end NUMINAMATH_GPT_lunch_cost_before_tax_and_tip_l1371_137115


namespace NUMINAMATH_GPT_mark_increase_reading_time_l1371_137130

def initial_pages_per_day : ℕ := 100
def final_pages_per_week : ℕ := 1750
def days_in_week : ℕ := 7

def calculate_percentage_increase (initial_pages_per_day : ℕ) (final_pages_per_week : ℕ) (days_in_week : ℕ) : ℚ :=
  ((final_pages_per_week : ℚ) / ((initial_pages_per_day : ℚ) * (days_in_week : ℚ)) - 1) * 100

theorem mark_increase_reading_time :
  calculate_percentage_increase initial_pages_per_day final_pages_per_week days_in_week = 150 :=
by sorry

end NUMINAMATH_GPT_mark_increase_reading_time_l1371_137130


namespace NUMINAMATH_GPT_max_value_4x_plus_3y_l1371_137186

theorem max_value_4x_plus_3y :
  ∃ x y : ℝ, (x^2 + y^2 = 16 * x + 8 * y + 8) ∧ (∀ w, w = 4 * x + 3 * y → w ≤ 64) ∧ ∃ x y, 4 * x + 3 * y = 64 :=
sorry

end NUMINAMATH_GPT_max_value_4x_plus_3y_l1371_137186


namespace NUMINAMATH_GPT_largest_power_of_three_dividing_A_l1371_137117

theorem largest_power_of_three_dividing_A (A : ℕ)
  (h1 : ∃ (factors : List ℕ), (∀ b ∈ factors, b > 0) ∧ factors.sum = 2011 ∧ factors.prod = A)
  : ∃ k : ℕ, 3^k ∣ A ∧ ∀ m : ℕ, 3^m ∣ A → m ≤ 669 :=
by
  sorry

end NUMINAMATH_GPT_largest_power_of_three_dividing_A_l1371_137117


namespace NUMINAMATH_GPT_part1_l1371_137132

theorem part1 (m : ℝ) (a b : ℝ) (h : m > 0) : 
  ( (a + m * b) / (1 + m) )^2 ≤ (a^2 + m * b^2) / (1 + m) :=
sorry

end NUMINAMATH_GPT_part1_l1371_137132


namespace NUMINAMATH_GPT_find_m_n_sum_l1371_137108

theorem find_m_n_sum (n m : ℝ) (d : ℝ) 
(h1 : ∀ x y, 2*x + y + n = 0) 
(h2 : ∀ x y, 4*x + m*y - 4 = 0) 
(hd : d = (3/5) * Real.sqrt 5) 
: m + n = -3 ∨ m + n = 3 :=
sorry

end NUMINAMATH_GPT_find_m_n_sum_l1371_137108


namespace NUMINAMATH_GPT_sin_neg_1740_eq_sqrt3_div_2_l1371_137192

theorem sin_neg_1740_eq_sqrt3_div_2 : Real.sin (-1740 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_neg_1740_eq_sqrt3_div_2_l1371_137192


namespace NUMINAMATH_GPT_max_marks_l1371_137150

theorem max_marks (marks_obtained failed_by : ℝ) (passing_percentage : ℝ) (M : ℝ) : 
  marks_obtained = 180 ∧ failed_by = 40 ∧ passing_percentage = 0.45 ∧ (marks_obtained + failed_by = passing_percentage * M) → M = 489 :=
by 
  sorry

end NUMINAMATH_GPT_max_marks_l1371_137150


namespace NUMINAMATH_GPT_sum_of_ages_l1371_137174

theorem sum_of_ages (J L : ℕ) (h1 : J = L + 8) (h2 : J + 5 = 3 * (L - 6)) : (J + L) = 39 :=
by {
  -- Proof steps would go here, but are omitted for this task per instructions
  sorry
}

end NUMINAMATH_GPT_sum_of_ages_l1371_137174


namespace NUMINAMATH_GPT_line_through_points_decreasing_direct_proportion_function_m_l1371_137157

theorem line_through_points_decreasing (x₁ x₂ y₁ y₂ k b : ℝ) (h1 : x₁ < x₂) (h2 : y₁ = k * x₁ + b) (h3 : y₂ = k * x₂ + b) (h4 : k < 0) : y₁ > y₂ :=
sorry

theorem direct_proportion_function_m (x₁ x₂ y₁ y₂ m : ℝ) (h1 : x₁ < x₂) (h2 : y₁ = (1 - 2 * m) * x₁) (h3 : y₂ = (1 - 2 * m) * x₂) (h4 : y₁ > y₂) : m > 1/2 :=
sorry

end NUMINAMATH_GPT_line_through_points_decreasing_direct_proportion_function_m_l1371_137157


namespace NUMINAMATH_GPT_price_per_pound_of_peanuts_is_2_40_l1371_137194

-- Assume the conditions
def peanuts_price_per_pound (P : ℝ) : Prop :=
  let cashews_price := 6.00
  let mixture_weight := 60
  let mixture_price_per_pound := 3.00
  let cashews_weight := 10
  let total_mixture_price := mixture_weight * mixture_price_per_pound
  let total_cashews_price := cashews_weight * cashews_price
  let total_peanuts_price := total_mixture_price - total_cashews_price
  let peanuts_weight := mixture_weight - cashews_weight
  let P := total_peanuts_price / peanuts_weight
  P = 2.40

-- Prove the price per pound of peanuts
theorem price_per_pound_of_peanuts_is_2_40 (P : ℝ) : peanuts_price_per_pound P :=
by
  sorry

end NUMINAMATH_GPT_price_per_pound_of_peanuts_is_2_40_l1371_137194


namespace NUMINAMATH_GPT_sequence_properties_l1371_137142

theorem sequence_properties (S : ℕ → ℝ) (a : ℕ → ℝ) :
  S 2 = 4 →
  (∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1) →
  a 1 = 1 ∧ S 5 = 121 :=
by
  intros hS2 ha
  sorry

end NUMINAMATH_GPT_sequence_properties_l1371_137142


namespace NUMINAMATH_GPT_range_of_b_l1371_137126

theorem range_of_b {b : ℝ} (h_b_ne_zero : b ≠ 0) :
  (∃ x : ℝ, (0 ≤ x ∧ x ≤ 3) ∧ (2 * x + b = 3)) ↔ -3 ≤ b ∧ b ≤ 3 ∧ b ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l1371_137126


namespace NUMINAMATH_GPT_emails_received_l1371_137133

variable (x y : ℕ)

theorem emails_received (h1 : 3 + 6 = 9) (h2 : x + y + 9 = 10) : x + y = 1 := by
  sorry

end NUMINAMATH_GPT_emails_received_l1371_137133


namespace NUMINAMATH_GPT_factor_t_squared_minus_81_l1371_137113

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) :=
by
  sorry

end NUMINAMATH_GPT_factor_t_squared_minus_81_l1371_137113


namespace NUMINAMATH_GPT_binom_sum_l1371_137134

theorem binom_sum : Nat.choose 18 4 + Nat.choose 5 2 = 3070 := 
by
  sorry

end NUMINAMATH_GPT_binom_sum_l1371_137134


namespace NUMINAMATH_GPT_inequality_solution_set_l1371_137185

open Set -- Open the Set namespace to work with sets in Lean

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (2 - x) ≥ 1 ↔ (x ∈ Icc (3 / 4) 2 \ {2}) := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1371_137185


namespace NUMINAMATH_GPT_largest_base8_3digit_to_base10_l1371_137172

theorem largest_base8_3digit_to_base10 : (7 * 8^2 + 7 * 8^1 + 7 * 8^0) = 511 := by
  sorry

end NUMINAMATH_GPT_largest_base8_3digit_to_base10_l1371_137172


namespace NUMINAMATH_GPT_mechanic_worked_days_l1371_137128

-- Definitions of conditions as variables
def hourly_rate : ℝ := 60
def hours_per_day : ℝ := 8
def cost_of_parts : ℝ := 2500
def total_amount_paid : ℝ := 9220

-- Definition to calculate the total labor cost
def total_labor_cost : ℝ := total_amount_paid - cost_of_parts

-- Definition to calculate the daily labor cost
def daily_labor_cost : ℝ := hourly_rate * hours_per_day

-- Proof (statement only) that the number of days the mechanic worked on the car is 14
theorem mechanic_worked_days : total_labor_cost / daily_labor_cost = 14 := by
  sorry

end NUMINAMATH_GPT_mechanic_worked_days_l1371_137128


namespace NUMINAMATH_GPT_determinant_computation_l1371_137171

variable (x y z w : ℝ)
variable (det : ℝ)
variable (H : x * w - y * z = 7)

theorem determinant_computation : 
  (x + z) * w - (y + 2 * w) * z = 7 - w * z := by
  sorry

end NUMINAMATH_GPT_determinant_computation_l1371_137171


namespace NUMINAMATH_GPT_base9_perfect_square_l1371_137116

theorem base9_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : ∃ k : ℕ, (729 * a + 81 * b + 36 + d) = k * k) :
    d = 0 ∨ d = 1 ∨ d = 4 ∨ d = 7 :=
sorry

end NUMINAMATH_GPT_base9_perfect_square_l1371_137116


namespace NUMINAMATH_GPT_function_relation_l1371_137187

def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 4*x + c

theorem function_relation (c : ℝ) :
  f 1 c > f 0 c ∧ f 0 c > f (-2) c := by
  sorry

end NUMINAMATH_GPT_function_relation_l1371_137187


namespace NUMINAMATH_GPT_laura_annual_income_l1371_137191

theorem laura_annual_income (I T : ℝ) (q : ℝ)
  (h1 : I > 50000) 
  (h2 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * 20000 + 0.01 * (q + 5) * (I - 50000))
  (h3 : T = 0.01 * (q + 0.5) * I) : I = 56000 := 
by sorry

end NUMINAMATH_GPT_laura_annual_income_l1371_137191


namespace NUMINAMATH_GPT_y_plus_inv_l1371_137100

theorem y_plus_inv (y : ℝ) (h : y^3 + 1/y^3 = 110) : y + 1/y = 5 := 
by 
sorry

end NUMINAMATH_GPT_y_plus_inv_l1371_137100


namespace NUMINAMATH_GPT_find_larger_number_l1371_137153

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1375) (h2 : L = 6 * S + 15) : L = 1647 :=
by
  -- proof to be filled
  sorry

end NUMINAMATH_GPT_find_larger_number_l1371_137153


namespace NUMINAMATH_GPT_H2O_required_for_NaH_reaction_l1371_137136

theorem H2O_required_for_NaH_reaction
  (n_NaH : ℕ) (n_H2O : ℕ) (n_NaOH : ℕ) (n_H2 : ℕ)
  (h_eq : n_NaH = 2) (balanced_eq : n_NaH = n_H2O ∧ n_H2O = n_NaOH ∧ n_NaOH = n_H2) :
  n_H2O = 2 :=
by
  -- The proof is omitted as we only need to declare the statement.
  sorry

end NUMINAMATH_GPT_H2O_required_for_NaH_reaction_l1371_137136


namespace NUMINAMATH_GPT_fraction_of_paint_first_week_l1371_137199

-- Definitions based on conditions
def total_paint := 360
def fraction_first_week (f : ℚ) : ℚ := f * total_paint
def paint_remaining_first_week (f : ℚ) : ℚ := total_paint - fraction_first_week f
def fraction_second_week (f : ℚ) : ℚ := (1 / 5) * paint_remaining_first_week f
def total_paint_used (f : ℚ) : ℚ := fraction_first_week f + fraction_second_week f
def total_paint_used_value := 104

-- Proof problem statement
theorem fraction_of_paint_first_week (f : ℚ) (h : total_paint_used f = total_paint_used_value) : f = 1 / 9 := 
sorry

end NUMINAMATH_GPT_fraction_of_paint_first_week_l1371_137199


namespace NUMINAMATH_GPT_average_weight_correct_l1371_137137

-- Define the number of men and women
def number_of_men : ℕ := 8
def number_of_women : ℕ := 6

-- Define the average weights of men and women
def average_weight_men : ℕ := 190
def average_weight_women : ℕ := 120

-- Define the total weight of men and women
def total_weight_men : ℕ := number_of_men * average_weight_men
def total_weight_women : ℕ := number_of_women * average_weight_women

-- Define the total number of individuals
def total_individuals : ℕ := number_of_men + number_of_women

-- Define the combined total weight
def total_weight : ℕ := total_weight_men + total_weight_women

-- Define the average weight of all individuals
def average_weight_all : ℕ := total_weight / total_individuals

theorem average_weight_correct :
  average_weight_all = 160 :=
  by sorry

end NUMINAMATH_GPT_average_weight_correct_l1371_137137


namespace NUMINAMATH_GPT_carla_initial_marbles_l1371_137164

theorem carla_initial_marbles (total_marbles : ℕ) (bought_marbles : ℕ) (initial_marbles : ℕ) 
  (h1 : total_marbles = 187) (h2 : bought_marbles = 134) (h3 : total_marbles = initial_marbles + bought_marbles) : 
  initial_marbles = 53 := 
sorry

end NUMINAMATH_GPT_carla_initial_marbles_l1371_137164


namespace NUMINAMATH_GPT_joaozinho_card_mariazinha_card_pedrinho_error_l1371_137104

-- Define the card transformation function
def transform_card (number : ℕ) (color_adjustment : ℕ) : ℕ :=
  (number * 2 + 3) * 5 + color_adjustment

-- The proof problems
theorem joaozinho_card : transform_card 3 4 = 49 :=
by
  sorry

theorem mariazinha_card : ∃ number, ∃ color_adjustment, transform_card number color_adjustment = 76 :=
by
  sorry

theorem pedrinho_error : ∀ number color_adjustment, ¬ transform_card number color_adjustment = 61 :=
by
  sorry

end NUMINAMATH_GPT_joaozinho_card_mariazinha_card_pedrinho_error_l1371_137104


namespace NUMINAMATH_GPT_speed_of_stream_l1371_137141

def boatSpeedDownstream (V_b V_s : ℝ) : ℝ :=
  V_b + V_s

def boatSpeedUpstream (V_b V_s : ℝ) : ℝ :=
  V_b - V_s

theorem speed_of_stream (V_b V_s : ℝ) (h1 : V_b + V_s = 25) (h2 : V_b - V_s = 5) : V_s = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_speed_of_stream_l1371_137141


namespace NUMINAMATH_GPT_max_value_b_minus_inv_a_is_minus_one_min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one_l1371_137177

open Real

noncomputable def max_value_b_minus_inv_a (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : ℝ :=
b - (1 / a)

noncomputable def min_value_inv_3a_plus_1_plus_inv_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : ℝ :=
(1 / (3 * a + 1)) + (1 / (a + b))

theorem max_value_b_minus_inv_a_is_minus_one (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : 
  max_value_b_minus_inv_a a b ha hb h = -1 :=
sorry

theorem min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : 
  min_value_inv_3a_plus_1_plus_inv_a_plus_b a b ha hb h = 1 :=
sorry

end NUMINAMATH_GPT_max_value_b_minus_inv_a_is_minus_one_min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one_l1371_137177


namespace NUMINAMATH_GPT_squares_difference_sum_l1371_137196

theorem squares_difference_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 :=
by 
  sorry

end NUMINAMATH_GPT_squares_difference_sum_l1371_137196


namespace NUMINAMATH_GPT_cube_edge_length_close_to_six_l1371_137197

theorem cube_edge_length_close_to_six
  (a V S : ℝ)
  (h1 : V = a^3)
  (h2 : S = 6 * a^2)
  (h3 : V = S + 1) : abs (a - 6) < 1 :=
by
  sorry

end NUMINAMATH_GPT_cube_edge_length_close_to_six_l1371_137197


namespace NUMINAMATH_GPT_percent_of_x_is_y_l1371_137135

-- Given the condition
def condition (x y : ℝ) : Prop :=
  0.70 * (x - y) = 0.30 * (x + y)

-- Prove y / x = 0.40
theorem percent_of_x_is_y (x y : ℝ) (h : condition x y) : y / x = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_x_is_y_l1371_137135


namespace NUMINAMATH_GPT_birdhouse_price_l1371_137173

theorem birdhouse_price (S : ℤ) : 
  (2 * 22) + (2 * 16) + (3 * S) = 97 → 
  S = 7 :=
by
  sorry

end NUMINAMATH_GPT_birdhouse_price_l1371_137173


namespace NUMINAMATH_GPT_fractions_arithmetic_lemma_l1371_137156

theorem fractions_arithmetic_lemma : (8 / 15 : ℚ) - (7 / 9) + (3 / 4) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_fractions_arithmetic_lemma_l1371_137156


namespace NUMINAMATH_GPT_fraction_sum_eq_one_l1371_137107

variables {a b c x y z : ℝ}

-- Conditions
axiom h1 : 11 * x + b * y + c * z = 0
axiom h2 : a * x + 24 * y + c * z = 0
axiom h3 : a * x + b * y + 41 * z = 0
axiom h4 : a ≠ 11
axiom h5 : x ≠ 0

-- Theorem Statement
theorem fraction_sum_eq_one : 
  a/(a - 11) + b/(b - 24) + c/(c - 41) = 1 :=
by sorry

end NUMINAMATH_GPT_fraction_sum_eq_one_l1371_137107


namespace NUMINAMATH_GPT_find_a_l1371_137162

theorem find_a (x y a : ℝ) (h1 : x + 3 * y = 4 - a) 
  (h2 : x - y = -3 * a) (h3 : x + y = 0) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l1371_137162


namespace NUMINAMATH_GPT_counting_4digit_integers_l1371_137148

theorem counting_4digit_integers (x y : ℕ) (a b c d : ℕ) :
  (x = 1000 * a + 100 * b + 10 * c + d) →
  (y = 1000 * d + 100 * c + 10 * b + a) →
  (y - x = 3177) →
  (1 ≤ a) → (a ≤ 6) →
  (0 ≤ b) → (b ≤ 7) →
  (c = b + 2) →
  (d = a + 3) →
  ∃ n : ℕ, n = 48 := 
sorry

end NUMINAMATH_GPT_counting_4digit_integers_l1371_137148


namespace NUMINAMATH_GPT_triangle_cos_Z_l1371_137144

theorem triangle_cos_Z (X Y Z : ℝ) (hXZ : X + Y + Z = π) 
  (sinX : Real.sin X = 4 / 5) (cosY : Real.cos Y = 3 / 5) : 
  Real.cos Z = 7 / 25 := 
sorry

end NUMINAMATH_GPT_triangle_cos_Z_l1371_137144


namespace NUMINAMATH_GPT_length_of_shorter_piece_l1371_137119

theorem length_of_shorter_piece (x : ℕ) (h1 : x + (x + 12) = 68) : x = 28 :=
by
  sorry

end NUMINAMATH_GPT_length_of_shorter_piece_l1371_137119


namespace NUMINAMATH_GPT_correct_average_marks_l1371_137106

theorem correct_average_marks
  (n : ℕ) (avg_mks wrong_mk correct_mk correct_avg_mks : ℕ)
  (H1 : n = 10)
  (H2 : avg_mks = 100)
  (H3 : wrong_mk = 50)
  (H4 : correct_mk = 10)
  (H5 : correct_avg_mks = 96) :
  (n * avg_mks - wrong_mk + correct_mk) / n = correct_avg_mks :=
by
  sorry

end NUMINAMATH_GPT_correct_average_marks_l1371_137106


namespace NUMINAMATH_GPT_average_speed_of_trip_l1371_137175

theorem average_speed_of_trip :
  let distance_local := 60
  let speed_local := 20
  let distance_highway := 120
  let speed_highway := 60
  let total_distance := distance_local + distance_highway
  let time_local := distance_local / speed_local
  let time_highway := distance_highway / speed_highway
  let total_time := time_local + time_highway
  let average_speed := total_distance / total_time
  average_speed = 36 := 
by 
  sorry

end NUMINAMATH_GPT_average_speed_of_trip_l1371_137175


namespace NUMINAMATH_GPT_quarts_of_water_required_l1371_137129

-- Define the ratio of water to juice
def ratio_water_to_juice : Nat := 5 / 3

-- Define the total punch to prepare in gallons
def total_punch_in_gallons : Nat := 2

-- Define the conversion factor from gallons to quarts
def quarts_per_gallon : Nat := 4

-- Define the total number of parts
def total_parts : Nat := 5 + 3

-- Define the total punch in quarts
def total_punch_in_quarts : Nat := total_punch_in_gallons * quarts_per_gallon

-- Define the amount of water per part
def quarts_per_part : Nat := total_punch_in_quarts / total_parts

-- Prove the required amount of water in quarts
theorem quarts_of_water_required : quarts_per_part * 5 = 5 := 
by
  -- Proof is omitted, represented by sorry
  sorry

end NUMINAMATH_GPT_quarts_of_water_required_l1371_137129


namespace NUMINAMATH_GPT_downstream_speed_l1371_137103

-- Define the speed of the fish in still water
def V_s : ℝ := 45

-- Define the speed of the fish going upstream
def V_u : ℝ := 35

-- Define the speed of the stream
def V_r : ℝ := V_s - V_u

-- Define the speed of the fish going downstream
def V_d : ℝ := V_s + V_r

-- The theorem to be proved
theorem downstream_speed : V_d = 55 := by
  sorry

end NUMINAMATH_GPT_downstream_speed_l1371_137103


namespace NUMINAMATH_GPT_total_people_in_office_even_l1371_137109

theorem total_people_in_office_even (M W : ℕ) (h_even : M = W) (h_meeting_women : 6 = 20 / 100 * W) : 
  M + W = 60 :=
by
  sorry

end NUMINAMATH_GPT_total_people_in_office_even_l1371_137109


namespace NUMINAMATH_GPT_f_g_of_3_l1371_137188

def f (x : ℝ) := 4 * x - 3
def g (x : ℝ) := x^2 + 2 * x + 1

theorem f_g_of_3 : f (g 3) = 61 :=
by
  sorry

end NUMINAMATH_GPT_f_g_of_3_l1371_137188


namespace NUMINAMATH_GPT_triangle_perimeter_l1371_137114

theorem triangle_perimeter (x : ℕ) 
  (h1 : x % 2 = 1) 
  (h2 : 7 - 2 < x)
  (h3 : x < 2 + 7) :
  2 + 7 + x = 16 := 
sorry

end NUMINAMATH_GPT_triangle_perimeter_l1371_137114


namespace NUMINAMATH_GPT_greatest_x_value_l1371_137183

theorem greatest_x_value : 
  ∃ x : ℝ, (∀ y : ℝ, (y = (4 * x - 16) / (3 * x - 4)) → (y^2 + y = 12)) ∧ (x = 2) := by
  sorry

end NUMINAMATH_GPT_greatest_x_value_l1371_137183


namespace NUMINAMATH_GPT_adam_has_9_apples_l1371_137180

def jackie_apples : ℕ := 6
def difference : ℕ := 3

def adam_apples (j : ℕ) (d : ℕ) : ℕ := 
  j + d

theorem adam_has_9_apples : adam_apples jackie_apples difference = 9 := 
by 
  sorry

end NUMINAMATH_GPT_adam_has_9_apples_l1371_137180


namespace NUMINAMATH_GPT_complex_modulus_l1371_137154

noncomputable def z : ℂ := (1 + 3 * Complex.I) / (1 + Complex.I)

theorem complex_modulus 
  (h : (1 + Complex.I) * z = 1 + 3 * Complex.I) : 
  Complex.abs (z^2) = 5 := 
by
  sorry

end NUMINAMATH_GPT_complex_modulus_l1371_137154


namespace NUMINAMATH_GPT_find_value_of_expression_l1371_137170

noncomputable def root_finder (a b c : ℝ) : Prop :=
  a^3 - 30*a^2 + 65*a - 42 = 0 ∧
  b^3 - 30*b^2 + 65*b - 42 = 0 ∧
  c^3 - 30*c^2 + 65*c - 42 = 0

theorem find_value_of_expression {a b c : ℝ} (h : root_finder a b c) :
  a + b + c = 30 ∧ ab + bc + ca = 65 ∧ abc = 42 → 
  (a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)) = 770/43 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l1371_137170


namespace NUMINAMATH_GPT_probability_of_matching_colors_l1371_137118

theorem probability_of_matching_colors :
  let abe_jelly_beans := ["green", "red", "blue"]
  let bob_jelly_beans := ["green", "green", "yellow", "yellow", "red", "red", "red"]
  let abe_probs := (1 / 3, 1 / 3, 1 / 3)
  let bob_probs := (2 / 7, 3 / 7, 0)
  let matching_prob := (1 / 3 * 2 / 7) + (1 / 3 * 3 / 7)
  matching_prob = 5 / 21 := by sorry

end NUMINAMATH_GPT_probability_of_matching_colors_l1371_137118


namespace NUMINAMATH_GPT_michael_remaining_yards_l1371_137169

theorem michael_remaining_yards (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ) (m y : ℕ)
  (h1 : miles_per_marathon = 50)
  (h2 : yards_per_marathon = 800)
  (h3 : yards_per_mile = 1760)
  (h4 : num_marathons = 5)
  (h5 : y = (yards_per_marathon * num_marathons) % yards_per_mile)
  (h6 : m = miles_per_marathon * num_marathons + (yards_per_marathon * num_marathons) / yards_per_mile) :
  y = 480 :=
sorry

end NUMINAMATH_GPT_michael_remaining_yards_l1371_137169


namespace NUMINAMATH_GPT_fixed_point_of_invariant_line_l1371_137112

theorem fixed_point_of_invariant_line :
  ∀ (m : ℝ) (x y : ℝ), (3 * m + 4) * x + (5 - 2 * m) * y + 7 * m - 6 = 0 →
  (x = -1 ∧ y = 2) :=
by
  intro m x y h
  sorry

end NUMINAMATH_GPT_fixed_point_of_invariant_line_l1371_137112


namespace NUMINAMATH_GPT_max_gcd_b_eq_1_l1371_137146

-- Define bn as bn = 2^n - 1 for natural number n
def b (n : ℕ) : ℕ := 2^n - 1

-- Define en as the greatest common divisor of bn and bn+1
def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

-- The theorem to prove:
theorem max_gcd_b_eq_1 (n : ℕ) : e n = 1 :=
  sorry

end NUMINAMATH_GPT_max_gcd_b_eq_1_l1371_137146


namespace NUMINAMATH_GPT_omitted_angle_measure_l1371_137160

theorem omitted_angle_measure (initial_sum correct_sum : ℝ) (H_initial : initial_sum = 2083) (H_correct : correct_sum = 2160) :
  correct_sum - initial_sum = 77 :=
by sorry

end NUMINAMATH_GPT_omitted_angle_measure_l1371_137160


namespace NUMINAMATH_GPT_jones_elementary_school_students_l1371_137120

theorem jones_elementary_school_students
  (X : ℕ)
  (boys_percent_total : ℚ)
  (num_students_represented : ℕ)
  (percent_of_boys : ℚ)
  (h1 : boys_percent_total = 0.60)
  (h2 : num_students_represented = 90)
  (h3 : percent_of_boys * (boys_percent_total * X) = 90)
  : X = 150 :=
by
  sorry

end NUMINAMATH_GPT_jones_elementary_school_students_l1371_137120


namespace NUMINAMATH_GPT_maximize_GDP_growth_l1371_137168

def projectA_investment : ℕ := 20  -- million yuan
def projectB_investment : ℕ := 10  -- million yuan

def total_investment (a b : ℕ) : ℕ := a + b
def total_electricity (a b : ℕ) : ℕ := 20000 * a + 40000 * b
def total_jobs (a b : ℕ) : ℕ := 24 * a + 36 * b
def total_GDP_increase (a b : ℕ) : ℕ := 26 * a + 20 * b  -- scaled by 10 to avoid decimals

theorem maximize_GDP_growth : 
  total_investment projectA_investment projectB_investment ≤ 30 ∧
  total_electricity projectA_investment projectB_investment ≤ 1000000 ∧
  total_jobs projectA_investment projectB_investment ≥ 840 → 
  total_GDP_increase projectA_investment projectB_investment = 860 := 
by
  -- Proof would be provided here
  sorry

end NUMINAMATH_GPT_maximize_GDP_growth_l1371_137168


namespace NUMINAMATH_GPT_code_XYZ_to_base_10_l1371_137110

def base_6_to_base_10 (x y z : ℕ) : ℕ :=
  x * 6^2 + y * 6^1 + z * 6^0

theorem code_XYZ_to_base_10 :
  ∀ (X Y Z : ℕ), 
    X = 5 ∧ Y = 0 ∧ Z = 4 →
    base_6_to_base_10 X Y Z = 184 :=
by
  intros X Y Z h
  cases' h with hX hYZ
  cases' hYZ with hY hZ
  rw [hX, hY, hZ]
  exact rfl

end NUMINAMATH_GPT_code_XYZ_to_base_10_l1371_137110


namespace NUMINAMATH_GPT_inequality_transformation_l1371_137189

theorem inequality_transformation (a b : ℝ) (h : a > b) : -3 * a < -3 * b := by
  sorry

end NUMINAMATH_GPT_inequality_transformation_l1371_137189


namespace NUMINAMATH_GPT_unique_abc_solution_l1371_137149

theorem unique_abc_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
    (h4 : a^4 + b^2 * c^2 = 16 * a) (h5 : b^4 + c^2 * a^2 = 16 * b) (h6 : c^4 + a^2 * b^2 = 16 * c) : 
    (a, b, c) = (2, 2, 2) :=
  by
    sorry

end NUMINAMATH_GPT_unique_abc_solution_l1371_137149


namespace NUMINAMATH_GPT_hyperbola_equation_l1371_137145

variable (a b c : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (asymptote_cond : -b / a = -1 / 2)
variable (foci_cond : c = 5)
variable (hyperbola_rel : a^2 + b^2 = c^2)

theorem hyperbola_equation : 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ -b / a = -1 / 2 ∧ c = 5 ∧ a^2 + b^2 = c^2 
  ∧ ∀ x y : ℝ, (x^2 / 20 - y^2 / 5 = 1)) := 
sorry

end NUMINAMATH_GPT_hyperbola_equation_l1371_137145


namespace NUMINAMATH_GPT_smallest_n_for_sqrt_20n_int_l1371_137158

theorem smallest_n_for_sqrt_20n_int (n : ℕ) (h : ∃ k : ℕ, 20 * n = k^2) : n = 5 :=
by sorry

end NUMINAMATH_GPT_smallest_n_for_sqrt_20n_int_l1371_137158


namespace NUMINAMATH_GPT_exists_h_l1371_137167

noncomputable def F (x : ℝ) : ℝ := x^2 + 12 / x^2
noncomputable def G (x : ℝ) : ℝ := Real.sin (Real.pi * x^2)
noncomputable def H (x : ℝ) : ℝ := 1

theorem exists_h (h : ℝ → ℝ) (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 10) :
  |h x - x| < 1 / 3 :=
sorry

end NUMINAMATH_GPT_exists_h_l1371_137167


namespace NUMINAMATH_GPT_outfits_count_l1371_137139

def num_outfits (redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats : ℕ) : ℕ :=
  (redShirts * pairsPants * (greenHats + blueHats)) +
  (greenShirts * pairsPants * (redHats + blueHats)) +
  (blueShirts * pairsPants * (redHats + greenHats))

theorem outfits_count :
  ∀ (redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats : ℕ),
  redShirts = 4 → greenShirts = 4 → blueShirts = 4 →
  pairsPants = 7 →
  greenHats = 6 → redHats = 6 → blueHats = 6 →
  num_outfits redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats = 1008 :=
by
  intros redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats
  intros hredShirts hgreenShirts hblueShirts hpairsPants hgreenHats hredHats hblueHats
  rw [hredShirts, hgreenShirts, hblueShirts, hpairsPants, hgreenHats, hredHats, hblueHats]
  sorry

end NUMINAMATH_GPT_outfits_count_l1371_137139


namespace NUMINAMATH_GPT_fish_problem_l1371_137111

theorem fish_problem : 
  ∀ (B T S : ℕ), 
    B = 10 → 
    T = 3 * B → 
    S = 35 → 
    B + T + S + 2 * S = 145 → 
    S - T = 5 :=
by sorry

end NUMINAMATH_GPT_fish_problem_l1371_137111


namespace NUMINAMATH_GPT_probability_of_sequence_HTHT_l1371_137124

noncomputable def prob_sequence_HTHT : ℚ :=
  let p := 1 / 2
  (p * p * p * p)

theorem probability_of_sequence_HTHT :
  prob_sequence_HTHT = 1 / 16 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_sequence_HTHT_l1371_137124


namespace NUMINAMATH_GPT_mixed_oil_rate_l1371_137102

noncomputable def rate_of_mixed_oil
  (volume1 : ℕ) (price1 : ℕ) (volume2 : ℕ) (price2 : ℕ) : ℚ :=
(total_cost : ℚ) / (total_volume : ℚ)
where
  total_cost := volume1 * price1 + volume2 * price2
  total_volume := volume1 + volume2

theorem mixed_oil_rate :
  rate_of_mixed_oil 10 50 5 66 = 55.33 := 
by
  sorry

end NUMINAMATH_GPT_mixed_oil_rate_l1371_137102


namespace NUMINAMATH_GPT_additional_discount_percentage_l1371_137125

-- Define constants representing the conditions
def price_shoes : ℝ := 200
def discount_shoes : ℝ := 0.30
def price_shirt : ℝ := 80
def number_shirts : ℕ := 2
def final_spent : ℝ := 285

-- Define the theorem to prove the additional discount percentage
theorem additional_discount_percentage :
  let discounted_shoes := price_shoes * (1 - discount_shoes)
  let total_before_additional_discount := discounted_shoes + number_shirts * price_shirt
  let additional_discount := total_before_additional_discount - final_spent
  (additional_discount / total_before_additional_discount) * 100 = 5 :=
by
  -- Lean proof goes here, but we'll skip it for now with sorry
  sorry

end NUMINAMATH_GPT_additional_discount_percentage_l1371_137125


namespace NUMINAMATH_GPT_fraction_simplest_sum_l1371_137182

theorem fraction_simplest_sum (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (3975 : ℚ) / 10000 = (a : ℚ) / b) 
  (simp : ∀ (c : ℕ), c ∣ a ∧ c ∣ b → c = 1) : a + b = 559 :=
sorry

end NUMINAMATH_GPT_fraction_simplest_sum_l1371_137182


namespace NUMINAMATH_GPT_Jordan_Lee_debt_equal_l1371_137165

theorem Jordan_Lee_debt_equal (initial_debt_jordan : ℝ) (additional_debt_jordan : ℝ)
  (rate_jordan : ℝ) (initial_debt_lee : ℝ) (rate_lee : ℝ) :
  initial_debt_jordan + additional_debt_jordan + (initial_debt_jordan + additional_debt_jordan) * rate_jordan * 33.333333333333336 
  = initial_debt_lee + initial_debt_lee * rate_lee * 33.333333333333336 :=
by
  let t := 33.333333333333336
  have rate_jordan := 0.12
  have rate_lee := 0.08
  have initial_debt_jordan := 200
  have additional_debt_jordan := 20
  have initial_debt_lee := 300
  sorry

end NUMINAMATH_GPT_Jordan_Lee_debt_equal_l1371_137165


namespace NUMINAMATH_GPT_fixed_constant_t_l1371_137127

-- Representation of point on the Cartesian plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of the parabola y = 4x^2
def parabola (p : Point) : Prop := p.y = 4 * p.x^2

-- Definition of distance squared between two points
def distance_squared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Main theorem statement
theorem fixed_constant_t :
  ∃ (c : ℝ) (C : Point), c = 1/8 ∧ C = ⟨1, c⟩ ∧ 
  (∀ (A B : Point), parabola A ∧ parabola B ∧ 
  (∃ m k : ℝ, A.y = m * A.x + k ∧ B.y = m * B.x + k ∧ k = c - m) → 
  (1 / distance_squared A C + 1 / distance_squared B C = 16)) :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_fixed_constant_t_l1371_137127


namespace NUMINAMATH_GPT_largest_B_is_9_l1371_137161

def is_divisible_by_three (n : ℕ) : Prop :=
  n % 3 = 0

def is_divisible_by_four (n : ℕ) : Prop :=
  n % 4 = 0

def largest_B_divisible_by_3_and_4 (B : ℕ) : Prop :=
  is_divisible_by_three (21 + B) ∧ is_divisible_by_four 32

theorem largest_B_is_9 : largest_B_divisible_by_3_and_4 9 :=
by
  have h1 : is_divisible_by_three (21 + 9) := by sorry
  have h2 : is_divisible_by_four 32 := by sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_largest_B_is_9_l1371_137161


namespace NUMINAMATH_GPT_solve_inequality_l1371_137101

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 1 / x else (1 / 3) ^ x

theorem solve_inequality : { x : ℝ | |f x| ≥ 1 / 3 } = { x : ℝ | -3 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1371_137101


namespace NUMINAMATH_GPT_aluminum_percentage_range_l1371_137179

variable (x1 x2 x3 y : ℝ)

theorem aluminum_percentage_range:
  (0.15 * x1 + 0.3 * x2 = 0.2) →
  (x1 + x2 + x3 = 1) →
  y = 0.6 * x1 + 0.45 * x3 →
  (1/3 ≤ x2 ∧ x2 ≤ 2/3) →
  (0.15 ≤ y ∧ y ≤ 0.4) := by
  sorry

end NUMINAMATH_GPT_aluminum_percentage_range_l1371_137179


namespace NUMINAMATH_GPT_particle_motion_inverse_relationship_l1371_137131

theorem particle_motion_inverse_relationship 
  {k : ℝ} 
  (inverse_relationship : ∀ {n : ℕ}, ∃ t_n d_n, d_n = k / t_n)
  (second_mile : ∃ t_2 d_2, t_2 = 2 ∧ d_2 = 1) : 
  ∃ t_4 d_4, t_4 = 4 ∧ d_4 = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_particle_motion_inverse_relationship_l1371_137131


namespace NUMINAMATH_GPT_student_B_more_consistent_l1371_137195

noncomputable def standard_deviation_A := 5.09
noncomputable def standard_deviation_B := 3.72
def games_played := 7
noncomputable def average_score_A := 16
noncomputable def average_score_B := 16

theorem student_B_more_consistent :
  standard_deviation_B < standard_deviation_A :=
sorry

end NUMINAMATH_GPT_student_B_more_consistent_l1371_137195


namespace NUMINAMATH_GPT_four_integers_product_sum_l1371_137166

theorem four_integers_product_sum (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_prod : a * b * c * d = 2002) (h_sum : a + b + c + d < 40) :
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) :=
sorry

end NUMINAMATH_GPT_four_integers_product_sum_l1371_137166


namespace NUMINAMATH_GPT_positive_reals_inequality_l1371_137176

variable {a b c : ℝ}

theorem positive_reals_inequality (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (a * b)^(1/4) + (b * c)^(1/4) + (c * a)^(1/4) < 1/4 := 
sorry

end NUMINAMATH_GPT_positive_reals_inequality_l1371_137176


namespace NUMINAMATH_GPT_sin_cos_eq_values_l1371_137152

theorem sin_cos_eq_values (θ : ℝ) (hθ : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  (∃ t : ℝ, 
    0 < t ∧ 
    t ≤ 2 * Real.pi ∧ 
    (2 + 4 * Real.sin t - 3 * Real.cos (2 * t) = 0)) ↔ (∃ n : ℕ, n = 4) :=
by 
  sorry

end NUMINAMATH_GPT_sin_cos_eq_values_l1371_137152


namespace NUMINAMATH_GPT_apples_per_slice_is_two_l1371_137190

def number_of_apples_per_slice (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) : ℕ :=
  total_apples / total_pies / slices_per_pie

theorem apples_per_slice_is_two (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) :
  total_apples = 48 → total_pies = 4 → slices_per_pie = 6 → number_of_apples_per_slice total_apples total_pies slices_per_pie = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_apples_per_slice_is_two_l1371_137190


namespace NUMINAMATH_GPT_water_left_ratio_l1371_137147

theorem water_left_ratio (h1: 2 * (30 / 10) = 6)
                        (h2: 2 * (30 / 10) = 6)
                        (h3: 4 * (60 / 10) = 24)
                        (water_left: ℕ)
                        (total_water_collected: ℕ) 
                        (h4: water_left = 18)
                        (h5: total_water_collected = 36) : 
  water_left * 2 = total_water_collected :=
by
  sorry

end NUMINAMATH_GPT_water_left_ratio_l1371_137147


namespace NUMINAMATH_GPT_k_cubed_divisible_l1371_137178

theorem k_cubed_divisible (k : ℕ) (h : k = 84) : ∃ n : ℕ, k ^ 3 = 592704 * n :=
by
  sorry

end NUMINAMATH_GPT_k_cubed_divisible_l1371_137178


namespace NUMINAMATH_GPT_arithmetic_series_sum_l1371_137155

variable (a₁ aₙ d S : ℝ)
variable (n : ℕ)

-- Defining the conditions (a₁, aₙ, d, and the formula for arithmetic series sum)
def first_term : a₁ = 10 := sorry
def last_term : aₙ = 70 := sorry
def common_diff : d = 1 / 7 := sorry

-- Equation to find number of terms (n)
def find_n : 70 = 10 + (n - 1) * (1 / 7) := sorry

-- Formula for the sum of an arithmetic series
def series_sum : S = (n * (10 + 70)) / 2 := sorry

-- The proof problem statement
theorem arithmetic_series_sum : 
  a₁ = 10 → 
  aₙ = 70 → 
  d = 1 / 7 → 
  (70 = 10 + (n - 1) * (1 / 7)) → 
  S = (n * (10 + 70)) / 2 → 
  S = 16840 := by 
  intros h1 h2 h3 h4 h5 
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_l1371_137155


namespace NUMINAMATH_GPT_skating_speeds_ratio_l1371_137138

theorem skating_speeds_ratio (v_s v_f : ℝ) (h1 : v_f > v_s) (h2 : |v_f + v_s| / |v_f - v_s| = 5) :
  v_f / v_s = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_skating_speeds_ratio_l1371_137138


namespace NUMINAMATH_GPT_interest_rate_increase_60_percent_l1371_137121

noncomputable def percentage_increase (A P A' t : ℝ) : ℝ :=
  let r₁ := (A - P) / (P * t)
  let r₂ := (A' - P) / (P * t)
  ((r₂ - r₁) / r₁) * 100

theorem interest_rate_increase_60_percent :
  percentage_increase 920 800 992 3 = 60 := by
  sorry

end NUMINAMATH_GPT_interest_rate_increase_60_percent_l1371_137121


namespace NUMINAMATH_GPT_mrs_hilt_walks_240_feet_l1371_137181

-- Define the distances and trips as given conditions
def distance_to_fountain : ℕ := 30
def trips_to_fountain : ℕ := 4
def round_trip_distance : ℕ := 2 * distance_to_fountain
def total_distance_walked (round_trip_distance trips_to_fountain : ℕ) : ℕ :=
  round_trip_distance * trips_to_fountain

-- State the theorem
theorem mrs_hilt_walks_240_feet :
  total_distance_walked round_trip_distance trips_to_fountain = 240 :=
by
  sorry

end NUMINAMATH_GPT_mrs_hilt_walks_240_feet_l1371_137181


namespace NUMINAMATH_GPT_harmonious_division_condition_l1371_137163

theorem harmonious_division_condition (a b c d e k : ℕ) (h : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e) (hk : 3 * k = a + b + c + d + e) (hk_pos : k > 0) :
  (∀ i j l : ℕ, i ≠ j ∧ j ≠ l ∧ i ≠ l → a ≤ k) ↔ (a ≤ k) :=
sorry

end NUMINAMATH_GPT_harmonious_division_condition_l1371_137163


namespace NUMINAMATH_GPT_trapezoid_length_relation_l1371_137140

variables {A B C D M N : Type}
variables [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
variables (a b c d m n : A)
variables (h_parallel_ab_cd : A) (h_parallel_mn_ab : A) 

-- The required proof statement
theorem trapezoid_length_relation (H1 : a = h_parallel_ab_cd) 
(H2 : b = m * n + h_parallel_mn_ab - m * d)
(H3 : c = d * (h_parallel_mn_ab - a))
(H4 : n = d / (n - a))
(H5 : n = c - h_parallel_ab_cd) :
c * m * a + b * c * d = n * d * a :=
sorry

end NUMINAMATH_GPT_trapezoid_length_relation_l1371_137140


namespace NUMINAMATH_GPT_smallest_three_digit_divisible_by_4_and_5_l1371_137151

-- Define the problem conditions and goal as a Lean theorem statement
theorem smallest_three_digit_divisible_by_4_and_5 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 4 = 0 ∧ m % 5 = 0 → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_three_digit_divisible_by_4_and_5_l1371_137151


namespace NUMINAMATH_GPT_sum_of_solutions_l1371_137159

theorem sum_of_solutions (x : ℝ) (h : x^2 - 3 * x = 12) : x = 3 := by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l1371_137159


namespace NUMINAMATH_GPT_odd_f_even_g_fg_eq_g_increasing_min_g_sum_l1371_137123

noncomputable def f (x : ℝ) : ℝ := (0.5) * ((2:ℝ)^x - (2:ℝ)^(-x))
noncomputable def g (x : ℝ) : ℝ := (0.5) * ((2:ℝ)^x + (2:ℝ)^(-x))

theorem odd_f (x : ℝ) : f (-x) = -f (x) := sorry
theorem even_g (x : ℝ) : g (-x) = g (x) := sorry
theorem fg_eq (x : ℝ) : f (x) + g (x) = (2:ℝ)^x := sorry
theorem g_increasing (x : ℝ) : x ≥ 0 → ∀ y, 0 ≤ y ∧ y < x → g y < g x := sorry
theorem min_g_sum (x : ℝ) : ∃ t, t ≥ 2 ∧ (g x + g (2 * x) = 2) := sorry

end NUMINAMATH_GPT_odd_f_even_g_fg_eq_g_increasing_min_g_sum_l1371_137123


namespace NUMINAMATH_GPT_solve_for_x_l1371_137193

-- Define the problem with the given conditions
def sum_of_triangle_angles (x : ℝ) : Prop := x + 2 * x + 30 = 180

-- State the theorem
theorem solve_for_x : ∀ (x : ℝ), sum_of_triangle_angles x → x = 50 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1371_137193


namespace NUMINAMATH_GPT_parabolic_points_l1371_137184

noncomputable def A (x1 : ℝ) (y1 : ℝ) : Prop := y1 = x1^2 - 3
noncomputable def B (x2 : ℝ) (y2 : ℝ) : Prop := y2 = x2^2 - 3

theorem parabolic_points (x1 x2 y1 y2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2)
  (hA : A x1 y1) (hB : B x2 y2) : y1 < y2 :=
by
  sorry

end NUMINAMATH_GPT_parabolic_points_l1371_137184


namespace NUMINAMATH_GPT_train_speed_in_kmph_l1371_137143

theorem train_speed_in_kmph (length_in_m : ℝ) (time_in_s : ℝ) (length_in_m_eq : length_in_m = 800.064) (time_in_s_eq : time_in_s = 18) : 
  (length_in_m / 1000) / (time_in_s / 3600) = 160.0128 :=
by
  rw [length_in_m_eq, time_in_s_eq]
  /-
  To convert length in meters to kilometers, divide by 1000.
  To convert time in seconds to hours, divide by 3600.
  The speed is then computed by dividing the converted length by the converted time.
  -/
  sorry

end NUMINAMATH_GPT_train_speed_in_kmph_l1371_137143


namespace NUMINAMATH_GPT_number_of_pairs_l1371_137105

theorem number_of_pairs (H : ∀ x y : ℕ , 0 < x → 0 < y → x < y → 2 * x * y / (x + y) = 4 ^ 15) :
  ∃ n : ℕ, n = 29 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pairs_l1371_137105


namespace NUMINAMATH_GPT_additional_students_needed_l1371_137198

theorem additional_students_needed 
  (n : ℕ) 
  (r : ℕ) 
  (t : ℕ) 
  (h_n : n = 82) 
  (h_r : r = 2) 
  (h_t : t = 49) : 
  (t - n / r) * r = 16 := 
by 
  sorry

end NUMINAMATH_GPT_additional_students_needed_l1371_137198
