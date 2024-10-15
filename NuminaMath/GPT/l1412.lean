import Mathlib

namespace NUMINAMATH_GPT_playground_dimensions_l1412_141223

theorem playground_dimensions 
  (a b : ℕ) 
  (h1 : (a - 2) * (b - 2) = 4) : a * b = 2 * a + 2 * b :=
by
  sorry

end NUMINAMATH_GPT_playground_dimensions_l1412_141223


namespace NUMINAMATH_GPT_additional_width_is_25cm_l1412_141219

-- Definitions
def length_of_room_cm := 5000
def width_of_room_cm := 1100
def additional_width_cm := 25
def number_of_tiles := 9000
def side_length_of_tile_cm := 25

-- Statement to prove
theorem additional_width_is_25cm : additional_width_cm = 25 :=
by
  -- The proof is omitted, we assume the proof steps here
  sorry

end NUMINAMATH_GPT_additional_width_is_25cm_l1412_141219


namespace NUMINAMATH_GPT_probability_log_value_l1412_141206

noncomputable def f (x : ℝ) := Real.log x / Real.log 2 - 1

theorem probability_log_value (a : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ 10) :
  (4 / 9 : ℝ) = 
    ((8 - 4) / (10 - 1) : ℝ) := by
  sorry

end NUMINAMATH_GPT_probability_log_value_l1412_141206


namespace NUMINAMATH_GPT_percent_both_correct_proof_l1412_141245

-- Define the problem parameters
def totalTestTakers := 100
def percentFirstCorrect := 80
def percentSecondCorrect := 75
def percentNeitherCorrect := 5

-- Define the target proof statement
theorem percent_both_correct_proof :
  percentFirstCorrect + percentSecondCorrect - percentFirstCorrect + percentNeitherCorrect = 60 := 
by 
  sorry

end NUMINAMATH_GPT_percent_both_correct_proof_l1412_141245


namespace NUMINAMATH_GPT_f_is_periodic_l1412_141239

noncomputable def f (x : ℝ) : ℝ := x - ⌊x⌋

theorem f_is_periodic : ∀ x : ℝ, f (x + 1) = f x := by
  intro x
  sorry

end NUMINAMATH_GPT_f_is_periodic_l1412_141239


namespace NUMINAMATH_GPT_common_region_area_of_triangles_l1412_141224

noncomputable def area_of_common_region (a : ℝ) : ℝ :=
  (a^2 * (2 * Real.sqrt 3 - 3)) / Real.sqrt 3

theorem common_region_area_of_triangles (a : ℝ) (h : 0 < a) : 
  area_of_common_region a = (a^2 * (2 * Real.sqrt 3 - 3)) / Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_common_region_area_of_triangles_l1412_141224


namespace NUMINAMATH_GPT_fraction_of_area_l1412_141277

def larger_square_side : ℕ := 6
def shaded_square_side : ℕ := 2

def larger_square_area : ℕ := larger_square_side * larger_square_side
def shaded_square_area : ℕ := shaded_square_side * shaded_square_side

theorem fraction_of_area : (shaded_square_area : ℚ) / larger_square_area = 1 / 9 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_fraction_of_area_l1412_141277


namespace NUMINAMATH_GPT_bs_sequence_bounded_iff_f_null_l1412_141229

def is_bs_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = abs (a (n + 1) - a (n + 2))

def f_null (a : ℕ → ℝ) : Prop :=
  ∀ n k, a n * a k * (a n - a k) = 0

def bs_bounded (a : ℕ → ℝ) : Prop :=
  ∃ M, ∀ n, abs (a n) ≤ M

theorem bs_sequence_bounded_iff_f_null (a : ℕ → ℝ) :
  is_bs_sequence a →
  (bs_bounded a ↔ f_null a) := by
  sorry

end NUMINAMATH_GPT_bs_sequence_bounded_iff_f_null_l1412_141229


namespace NUMINAMATH_GPT_degree_of_monomial_neg2x2y_l1412_141244

def monomial_degree (coeff : ℤ) (exp_x exp_y : ℕ) : ℕ :=
  exp_x + exp_y

theorem degree_of_monomial_neg2x2y :
  monomial_degree (-2) 2 1 = 3 :=
by
  -- Definition matching conditions given
  sorry

end NUMINAMATH_GPT_degree_of_monomial_neg2x2y_l1412_141244


namespace NUMINAMATH_GPT_find_y_value_l1412_141248

theorem find_y_value (t : ℝ) (x : ℝ) (y : ℝ)
  (h1 : x = 3 - 2 * t)
  (h2 : y = 3 * t + 6)
  (h3 : x = -6)
  : y = 19.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_y_value_l1412_141248


namespace NUMINAMATH_GPT_stamp_problem_l1412_141295

/-- Define the context where we have stamps of 7, n, and (n + 2) cents, and 120 cents being the largest
    value that cannot be formed using these stamps -/
theorem stamp_problem (n : ℕ) (h : ∀ k, k > 120 → ∃ a b c, k = 7 * a + n * b + (n + 2) * c) (hn : ¬ ∃ a b c, 120 = 7 * a + n * b + (n + 2) * c) : n = 22 :=
sorry

end NUMINAMATH_GPT_stamp_problem_l1412_141295


namespace NUMINAMATH_GPT_average_weight_l1412_141253

theorem average_weight (w : ℕ) : 
  (64 < w ∧ w ≤ 67) → w = 66 :=
by sorry

end NUMINAMATH_GPT_average_weight_l1412_141253


namespace NUMINAMATH_GPT_num_diamonds_F10_l1412_141243

def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 4 else 4 * (3 * n - 2)

theorem num_diamonds_F10 : num_diamonds 10 = 112 := by
  sorry

end NUMINAMATH_GPT_num_diamonds_F10_l1412_141243


namespace NUMINAMATH_GPT_zero_ordered_triples_non_zero_satisfy_conditions_l1412_141207

theorem zero_ordered_triples_non_zero_satisfy_conditions :
  ∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 → a = b + c → b = c + a → c = a + b → a + b + c ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_ordered_triples_non_zero_satisfy_conditions_l1412_141207


namespace NUMINAMATH_GPT_percentage_by_which_x_is_less_than_y_l1412_141270

noncomputable def percentageLess (x y : ℝ) : ℝ :=
  ((y - x) / y) * 100

theorem percentage_by_which_x_is_less_than_y :
  ∀ (x y : ℝ),
  y = 125 + 0.10 * 125 →
  x = 123.75 →
  percentageLess x y = 10 :=
by
  intros x y h1 h2
  rw [h1, h2]
  unfold percentageLess
  sorry

end NUMINAMATH_GPT_percentage_by_which_x_is_less_than_y_l1412_141270


namespace NUMINAMATH_GPT_washingMachineCapacity_l1412_141266

-- Definitions based on the problem's conditions
def numberOfShirts : ℕ := 2
def numberOfSweaters : ℕ := 33
def numberOfLoads : ℕ := 5

-- Statement we need to prove
theorem washingMachineCapacity : 
  (numberOfShirts + numberOfSweaters) / numberOfLoads = 7 := sorry

end NUMINAMATH_GPT_washingMachineCapacity_l1412_141266


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l1412_141289

open Real

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := 
  sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l1412_141289


namespace NUMINAMATH_GPT_arccos_half_eq_pi_div_3_l1412_141278

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_arccos_half_eq_pi_div_3_l1412_141278


namespace NUMINAMATH_GPT_original_price_of_bag_l1412_141249

theorem original_price_of_bag (P : ℝ) 
  (h1 : ∀ x, 0 < x → x < 1 → x * 100 = 75)
  (h2 : 2 * (0.25 * P) = 3)
  : P = 6 :=
sorry

end NUMINAMATH_GPT_original_price_of_bag_l1412_141249


namespace NUMINAMATH_GPT_sum_of_geometric_sequence_l1412_141235

noncomputable def geometric_sequence_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem sum_of_geometric_sequence
  (a_1 q : ℝ) 
  (h1 : a_1^2 * q^6 = 2 * a_1 * q^2)
  (h2 : (a_1 * q^3 + 2 * a_1 * q^6) / 2 = 5 / 4)
  : geometric_sequence_sum a_1 q 4 = 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_geometric_sequence_l1412_141235


namespace NUMINAMATH_GPT_person_walk_rate_l1412_141240

theorem person_walk_rate (v : ℝ) (elevator_speed : ℝ) (length : ℝ) (time : ℝ) 
  (h1 : elevator_speed = 10) 
  (h2 : length = 112) 
  (h3 : time = 8) 
  (h4 : length = (v + elevator_speed) * time) 
  : v = 4 :=
by 
  sorry

end NUMINAMATH_GPT_person_walk_rate_l1412_141240


namespace NUMINAMATH_GPT_solution_exists_l1412_141290

open Real

theorem solution_exists (x : ℝ) (h1 : x > 9) (h2 : sqrt (x - 3 * sqrt (x - 9)) + 3 = sqrt (x + 3 * sqrt (x - 9)) - 3) : x ≥ 18 :=
sorry

end NUMINAMATH_GPT_solution_exists_l1412_141290


namespace NUMINAMATH_GPT_correct_statements_l1412_141298

theorem correct_statements : 
  (∃ n : ℕ, 24 = 4 * n) ∧ (∃ n : ℕ, 180 = 9 * n) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l1412_141298


namespace NUMINAMATH_GPT_ratio_n_over_p_l1412_141221

theorem ratio_n_over_p (m n p : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : p ≠ 0) 
  (h4 : ∃ r1 r2 : ℝ, r1 + r2 = -p ∧ r1 * r2 = m ∧ 3 * r1 + 3 * r2 = -m ∧ 9 * r1 * r2 = n) :
  n / p = -27 := 
by
  sorry

end NUMINAMATH_GPT_ratio_n_over_p_l1412_141221


namespace NUMINAMATH_GPT_general_equation_of_curve_l1412_141284

theorem general_equation_of_curve
  (t : ℝ) (ht : t > 0)
  (x : ℝ) (hx : x = (Real.sqrt t) - (1 / (Real.sqrt t)))
  (y : ℝ) (hy : y = 3 * (t + 1 / t) + 2) :
  x^2 = (y - 8) / 3 := by
  sorry

end NUMINAMATH_GPT_general_equation_of_curve_l1412_141284


namespace NUMINAMATH_GPT_shirts_per_minute_l1412_141279

theorem shirts_per_minute (total_shirts : ℕ) (total_minutes : ℕ) (h1 : total_shirts = 196) (h2 : total_minutes = 28) :
  total_shirts / total_minutes = 7 :=
by
  -- beginning of proof would go here
  sorry

end NUMINAMATH_GPT_shirts_per_minute_l1412_141279


namespace NUMINAMATH_GPT_consecutive_number_other_17_l1412_141250

theorem consecutive_number_other_17 (a b : ℕ) (h1 : b = 17) (h2 : a + b = 35) (h3 : a + b % 5 = 0) : a = 18 :=
sorry

end NUMINAMATH_GPT_consecutive_number_other_17_l1412_141250


namespace NUMINAMATH_GPT_polynomial_value_at_3_l1412_141233

def f (x : ℝ) : ℝ := 4*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

theorem polynomial_value_at_3 : f 3 = 1209.4 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_value_at_3_l1412_141233


namespace NUMINAMATH_GPT_time_to_build_wall_l1412_141288

theorem time_to_build_wall (t_A t_B t_C : ℝ) 
  (h1 : 1 / t_A + 1 / t_B = 1 / 25)
  (h2 : 1 / t_C = 1 / 35)
  (h3 : 1 / t_A = 1 / t_B + 1 / t_C) : t_B = 87.5 :=
by
  sorry

end NUMINAMATH_GPT_time_to_build_wall_l1412_141288


namespace NUMINAMATH_GPT_nonzero_fraction_power_zero_l1412_141205

theorem nonzero_fraction_power_zero (a b : ℤ) (h : a ≠ 0 ∧ b ≠ 0) : ((a : ℚ) / b)^0 = 1 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_nonzero_fraction_power_zero_l1412_141205


namespace NUMINAMATH_GPT_sampling_method_systematic_l1412_141225

theorem sampling_method_systematic 
  (inspect_interval : ℕ := 10)
  (products_interval : ℕ := 10)
  (position : ℕ) :
  inspect_interval = 10 ∧ products_interval = 10 → 
  (sampling_method = "Systematic Sampling") :=
by
  sorry

end NUMINAMATH_GPT_sampling_method_systematic_l1412_141225


namespace NUMINAMATH_GPT_remainder_3249_div_82_eq_51_l1412_141293

theorem remainder_3249_div_82_eq_51 : (3249 % 82) = 51 :=
by
  sorry

end NUMINAMATH_GPT_remainder_3249_div_82_eq_51_l1412_141293


namespace NUMINAMATH_GPT_find_x_l1412_141276

def balanced (a b c d : ℝ) : Prop :=
  a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem find_x (x : ℝ) : (∀ (a b c d : ℝ), balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔ x ≥ 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l1412_141276


namespace NUMINAMATH_GPT_russian_dolls_initial_purchase_l1412_141247

theorem russian_dolls_initial_purchase (cost_initial cost_discount : ℕ) (num_discount : ℕ) (savings : ℕ) :
  cost_initial = 4 → cost_discount = 3 → num_discount = 20 → savings = num_discount * cost_discount → 
  (savings / cost_initial) = 15 := 
by {
sorry
}

end NUMINAMATH_GPT_russian_dolls_initial_purchase_l1412_141247


namespace NUMINAMATH_GPT_area_of_gray_region_l1412_141220

theorem area_of_gray_region (r R : ℝ) (hr : r = 2) (hR : R = 3 * r) : 
  π * R ^ 2 - π * r ^ 2 = 32 * π :=
by
  have hr : r = 2 := hr
  have hR : R = 3 * r := hR
  sorry

end NUMINAMATH_GPT_area_of_gray_region_l1412_141220


namespace NUMINAMATH_GPT_maximum_value_of_f_minimum_value_of_f_l1412_141263

-- Define the function f
def f (x y : ℝ) : ℝ := 3 * |x + y| + |4 * y + 9| + |7 * y - 3 * x - 18|

-- Define the condition
def condition (x y : ℝ) : Prop := x^2 + y^2 ≤ 5

-- State the maximum value theorem
theorem maximum_value_of_f (x y : ℝ) (h : condition x y) :
  ∃ (x y : ℝ), f x y = 27 + 6 * Real.sqrt 5 := sorry

-- State the minimum value theorem
theorem minimum_value_of_f (x y : ℝ) (h : condition x y) :
  ∃ (x y : ℝ), f x y = 27 - 3 * Real.sqrt 10 := sorry

end NUMINAMATH_GPT_maximum_value_of_f_minimum_value_of_f_l1412_141263


namespace NUMINAMATH_GPT_average_speed_first_part_l1412_141258

noncomputable def speed_of_first_part (v : ℝ) : Prop :=
  let distance_first_part := 124
  let speed_second_part := 60
  let distance_second_part := 250 - distance_first_part
  let total_time := 5.2
  (distance_first_part / v) + (distance_second_part / speed_second_part) = total_time

theorem average_speed_first_part : speed_of_first_part 40 :=
  sorry

end NUMINAMATH_GPT_average_speed_first_part_l1412_141258


namespace NUMINAMATH_GPT_smallest_prime_factor_of_5_pow_5_minus_5_pow_3_l1412_141251

theorem smallest_prime_factor_of_5_pow_5_minus_5_pow_3 : Nat.Prime 2 ∧ (∀ p : ℕ, Nat.Prime p ∧ p ∣ (5^5 - 5^3) → p ≥ 2) := by
  sorry

end NUMINAMATH_GPT_smallest_prime_factor_of_5_pow_5_minus_5_pow_3_l1412_141251


namespace NUMINAMATH_GPT_sum_X_Y_l1412_141268

-- Define the variables and assumptions
variable (X Y : ℕ)

-- Hypotheses
axiom h1 : Y + 2 = X
axiom h2 : X + 5 = Y

-- Theorem statement
theorem sum_X_Y : X + Y = 12 := by
  sorry

end NUMINAMATH_GPT_sum_X_Y_l1412_141268


namespace NUMINAMATH_GPT_red_packet_grabbing_situations_l1412_141294

-- Definitions based on the conditions
def numberOfPeople := 5
def numberOfPackets := 4
def packets := [2, 2, 3, 5]  -- 2-yuan, 2-yuan, 3-yuan, 5-yuan

-- Main theorem statement
theorem red_packet_grabbing_situations : 
  ∃ situations : ℕ, situations = 60 :=
by
  sorry

end NUMINAMATH_GPT_red_packet_grabbing_situations_l1412_141294


namespace NUMINAMATH_GPT_second_player_wins_l1412_141260

-- Define the initial condition of the game
def initial_coins : Nat := 2016

-- Define the set of moves a player can make
def valid_moves : Finset Nat := {1, 2, 3}

-- Define the winning condition
def winning_player (coins : Nat) : String :=
  if coins % 4 = 0 then "second player"
  else "first player"

-- The theorem stating that second player has a winning strategy given the initial condition
theorem second_player_wins : winning_player initial_coins = "second player" :=
by
  sorry

end NUMINAMATH_GPT_second_player_wins_l1412_141260


namespace NUMINAMATH_GPT_captivating_quadruples_count_l1412_141254

theorem captivating_quadruples_count :
  (∃ n : ℕ, n = 682) ↔ 
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + d < b + c :=
sorry

end NUMINAMATH_GPT_captivating_quadruples_count_l1412_141254


namespace NUMINAMATH_GPT_range_of_a_l1412_141271

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 1| + |x + 1| > a) ↔ a < 2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1412_141271


namespace NUMINAMATH_GPT_glass_panels_in_neighborhood_l1412_141216

def total_glass_panels_in_neighborhood := 
  let double_windows_downstairs : ℕ := 6
  let glass_panels_per_double_window_downstairs : ℕ := 4
  let single_windows_upstairs : ℕ := 8
  let glass_panels_per_single_window_upstairs : ℕ := 3
  let bay_windows : ℕ := 2
  let glass_panels_per_bay_window : ℕ := 6
  let houses : ℕ := 10

  let glass_panels_in_one_house : ℕ := 
    (double_windows_downstairs * glass_panels_per_double_window_downstairs) +
    (single_windows_upstairs * glass_panels_per_single_window_upstairs) +
    (bay_windows * glass_panels_per_bay_window)

  houses * glass_panels_in_one_house

theorem glass_panels_in_neighborhood : total_glass_panels_in_neighborhood = 600 := by
  -- Calculation steps skipped
  sorry

end NUMINAMATH_GPT_glass_panels_in_neighborhood_l1412_141216


namespace NUMINAMATH_GPT_determine_m_range_l1412_141211

-- Define propositions P and Q
def P (t : ℝ) : Prop := ∃ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) = 1)
def Q (t m : ℝ) : Prop := 1 - m < t ∧ t < 1 + m ∧ m > 0

-- Define negation of propositions
def notP (t : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) ≠ 1)
def notQ (t m : ℝ) : Prop := ¬ (1 - m < t ∧ t < 1 + m)

-- Main problem: Determine the range of m where notP -> notQ is a sufficient but not necessary condition
theorem determine_m_range {m : ℝ} : (∃ t : ℝ, notP t → notQ t m) ↔ (0 < m ∧ m ≤ 3) := by
  sorry

end NUMINAMATH_GPT_determine_m_range_l1412_141211


namespace NUMINAMATH_GPT_checkout_speed_ratio_l1412_141234

theorem checkout_speed_ratio (n x y : ℝ) 
  (h1 : 40 * x = 20 * y + n)
  (h2 : 36 * x = 12 * y + n) : 
  x = 2 * y := 
sorry

end NUMINAMATH_GPT_checkout_speed_ratio_l1412_141234


namespace NUMINAMATH_GPT_quinn_frogs_caught_l1412_141287

-- Defining the conditions
def Alster_frogs : Nat := 2

def Quinn_frogs (Alster_caught: Nat) : Nat := Alster_caught

def Bret_frogs (Quinn_caught: Nat) : Nat := 3 * Quinn_caught

-- Given that Bret caught 12 frogs, prove the amount Quinn caught
theorem quinn_frogs_caught (Bret_caught: Nat) (h1: Bret_caught = 12) : Quinn_frogs Alster_frogs = 4 :=
by
  sorry

end NUMINAMATH_GPT_quinn_frogs_caught_l1412_141287


namespace NUMINAMATH_GPT_tan_identity_find_sum_l1412_141291

-- Given conditions
def is_geometric_sequence (a b c : ℝ) : Prop := b^2 = a * c

-- Specific problem statements
theorem tan_identity (a b c : ℝ) (A B C : ℝ)
  (h_geometric : is_geometric_sequence a b c)
  (h_cosB : Real.cos B = 3 / 4) :
  1 / Real.tan A + 1 / Real.tan C = 4 / Real.sqrt 7 :=
sorry

theorem find_sum (a b c : ℝ)
  (h_dot_product : a * c * 3 / 4 = 3 / 2) :
  a + c = 3 :=
sorry

end NUMINAMATH_GPT_tan_identity_find_sum_l1412_141291


namespace NUMINAMATH_GPT_angela_problems_l1412_141236

theorem angela_problems (total_problems martha_problems : ℕ) (jenna_problems mark_problems : ℕ) 
    (h1 : total_problems = 20) 
    (h2 : martha_problems = 2)
    (h3 : jenna_problems = 4 * martha_problems - 2)
    (h4 : mark_problems = jenna_problems / 2) :
    total_problems - (martha_problems + jenna_problems + mark_problems) = 9 := 
sorry

end NUMINAMATH_GPT_angela_problems_l1412_141236


namespace NUMINAMATH_GPT_sequence_inequality_for_k_l1412_141252

theorem sequence_inequality_for_k (k : ℝ) : 
  (∀ n : ℕ, 0 < n → (n + 1)^2 + k * (n + 1) + 2 > n^2 + k * n + 2) ↔ k > -3 :=
sorry

end NUMINAMATH_GPT_sequence_inequality_for_k_l1412_141252


namespace NUMINAMATH_GPT_binom_12_10_eq_66_l1412_141222

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_12_10_eq_66 : binom 12 10 = 66 := 
by
  -- We state the given condition for binomial coefficients.
  let binom_coeff := binom 12 10
  -- We will use the formula \binom{n}{k} = \frac{n!}{k!(n-k)!}.
  change binom_coeff = Nat.factorial 12 / (Nat.factorial 10 * Nat.factorial (12 - 10))
  -- Simplify using factorial values and basic arithmetic to reach the conclusion.
  sorry

end NUMINAMATH_GPT_binom_12_10_eq_66_l1412_141222


namespace NUMINAMATH_GPT_area_of_one_postcard_is_150_cm2_l1412_141273

/-- Define the conditions of the problem. -/
def perimeter_of_stitched_postcard : ℕ := 70
def vertical_length_of_postcard : ℕ := 15

/-- Definition stating that postcards are attached horizontally and do not overlap. 
    This logically implies that the horizontal length gets doubled and perimeter is 2V + 4H. -/
def attached_horizontally (V H : ℕ) (P : ℕ) : Prop :=
  2 * V + 4 * H = P

/-- Main theorem stating the question and the derived answer,
    proving that the area of one postcard is 150 square centimeters. -/
theorem area_of_one_postcard_is_150_cm2 :
  ∃ (H : ℕ), attached_horizontally vertical_length_of_postcard H perimeter_of_stitched_postcard ∧
  (vertical_length_of_postcard * H = 150) :=
by 
  sorry -- the proof is omitted

end NUMINAMATH_GPT_area_of_one_postcard_is_150_cm2_l1412_141273


namespace NUMINAMATH_GPT_smallest_positive_integer_remainder_conditions_l1412_141213

theorem smallest_positive_integer_remainder_conditions :
  ∃ b : ℕ, (b % 4 = 3 ∧ b % 6 = 5) ∧ ∀ n : ℕ, (n % 4 = 3 ∧ n % 6 = 5) → n ≥ b := 
by
  have b := 23
  use b
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_remainder_conditions_l1412_141213


namespace NUMINAMATH_GPT_carter_baseball_cards_l1412_141292

theorem carter_baseball_cards (m c : ℕ) (h1 : m = 210) (h2 : m = c + 58) : c = 152 := 
by
  sorry

end NUMINAMATH_GPT_carter_baseball_cards_l1412_141292


namespace NUMINAMATH_GPT_regular_polygon_sides_l1412_141217

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) : 
  ∃ n : ℕ, n = 12 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1412_141217


namespace NUMINAMATH_GPT_distinct_real_number_sum_and_square_sum_eq_l1412_141285

theorem distinct_real_number_sum_and_square_sum_eq
  (a b c d : ℝ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a + b + c + d = 3)
  (h_square_sum : a^2 + b^2 + c^2 + d^2 = 45) :
  (a^5 / (a - b) / (a - c) / (a - d)) + (b^5 / (b - a) / (b - c) / (b - d)) +
  (c^5 / (c - a) / (c - b) / (c - d)) + (d^5 / (d - a) / (d - b) / (d - c)) = -9 :=
by
  sorry

end NUMINAMATH_GPT_distinct_real_number_sum_and_square_sum_eq_l1412_141285


namespace NUMINAMATH_GPT_pos_solution_sum_l1412_141230

theorem pos_solution_sum (c d : ℕ) (h_c_pos : 0 < c) (h_d_pos : 0 < d) :
  (∃ x : ℝ, x ^ 2 + 16 * x = 100 ∧ x = Real.sqrt c - d) → c + d = 172 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_pos_solution_sum_l1412_141230


namespace NUMINAMATH_GPT_Q_has_exactly_one_negative_root_l1412_141262

def Q (x : ℝ) : ℝ := x^7 + 5 * x^5 + 5 * x^4 - 6 * x^3 - 2 * x^2 - 10 * x + 12

theorem Q_has_exactly_one_negative_root :
  ∃! r : ℝ, r < 0 ∧ Q r = 0 := sorry

end NUMINAMATH_GPT_Q_has_exactly_one_negative_root_l1412_141262


namespace NUMINAMATH_GPT_truncated_pyramid_properties_l1412_141257

noncomputable def truncatedPyramidSurfaceArea
  (a b c : ℝ) (theta m : ℝ) : ℝ :=
sorry -- Definition of the surface area function

noncomputable def truncatedPyramidVolume
  (a b c : ℝ) (theta m : ℝ) : ℝ :=
sorry -- Definition of the volume function

theorem truncated_pyramid_properties
  (a b c : ℝ) (theta m : ℝ)
  (h₀ : a = 148) 
  (h₁ : b = 156) 
  (h₂ : c = 208) 
  (h₃ : theta = 112.62) 
  (h₄ : m = 27) :
  (truncatedPyramidSurfaceArea a b c theta m = 74352) ∧
  (truncatedPyramidVolume a b c theta m = 395280) :=
by
  sorry -- The actual proof will go here

end NUMINAMATH_GPT_truncated_pyramid_properties_l1412_141257


namespace NUMINAMATH_GPT_even_function_m_value_l1412_141264

def f (x m : ℝ) : ℝ := (x - 2) * (x - m)

theorem even_function_m_value (m : ℝ) :
  (∀ x : ℝ, f x m = f (-x) m) → m = -2 := by
  sorry

end NUMINAMATH_GPT_even_function_m_value_l1412_141264


namespace NUMINAMATH_GPT_ratio_expression_l1412_141296

theorem ratio_expression (A B C : ℚ) (h : A / B = 3 / 1 ∧ B / C = 1 / 6) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 :=
by sorry

end NUMINAMATH_GPT_ratio_expression_l1412_141296


namespace NUMINAMATH_GPT_exists_special_integer_l1412_141227

-- Define the mathematical conditions and the proof
theorem exists_special_integer (P : Finset ℕ) (hP : ∀ p ∈ P, Nat.Prime p) : 
  ∃ x : ℕ, 
    (∀ p ∈ P, ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ x = a^p + b^p) ∧
    (∀ p ∉ P, ¬∃ a b : ℕ, 0 < a ∧ 0 < b ∧ x = a^p + b^p) :=
sorry

end NUMINAMATH_GPT_exists_special_integer_l1412_141227


namespace NUMINAMATH_GPT_positive_quadratic_expression_l1412_141246

theorem positive_quadratic_expression (m : ℝ) :
  (∀ x : ℝ, (4 - m) * x^2 - 3 * x + 4 + m > 0) ↔ (- (Real.sqrt 55) / 2 < m ∧ m < (Real.sqrt 55) / 2) := 
sorry

end NUMINAMATH_GPT_positive_quadratic_expression_l1412_141246


namespace NUMINAMATH_GPT_cyclist_return_trip_average_speed_l1412_141210

theorem cyclist_return_trip_average_speed :
  let first_leg_distance := 12
  let second_leg_distance := 24
  let first_leg_speed := 8
  let second_leg_speed := 12
  let round_trip_time := 7.5
  let distance_to_destination := first_leg_distance + second_leg_distance
  let time_to_destination := (first_leg_distance / first_leg_speed) + (second_leg_distance / second_leg_speed)
  let return_trip_time := round_trip_time - time_to_destination
  let return_trip_distance := distance_to_destination
  (return_trip_distance / return_trip_time) = 9 := 
by
  sorry

end NUMINAMATH_GPT_cyclist_return_trip_average_speed_l1412_141210


namespace NUMINAMATH_GPT_multiply_123_32_125_l1412_141204

theorem multiply_123_32_125 : 123 * 32 * 125 = 492000 := by
  sorry

end NUMINAMATH_GPT_multiply_123_32_125_l1412_141204


namespace NUMINAMATH_GPT_doug_marbles_l1412_141208

theorem doug_marbles (e_0 d_0 : ℕ) (h1 : e_0 = d_0 + 12) (h2 : e_0 - 20 = 17) : d_0 = 25 :=
by
  sorry

end NUMINAMATH_GPT_doug_marbles_l1412_141208


namespace NUMINAMATH_GPT_evaluate_expression_l1412_141232

theorem evaluate_expression :
  (1 / (-5^3)^4) * (-5)^15 * 5^2 = -3125 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1412_141232


namespace NUMINAMATH_GPT_parabola_distance_focus_P_l1412_141265

noncomputable def distance_PF : ℝ := sorry

theorem parabola_distance_focus_P : ∀ (P : ℝ × ℝ) (F : ℝ × ℝ),
  P.2^2 = 4 * P.1 ∧ F = (1, 0) ∧ P.1 = 4 → distance_PF = 5 :=
by
  intros P F h
  sorry

end NUMINAMATH_GPT_parabola_distance_focus_P_l1412_141265


namespace NUMINAMATH_GPT_triangle_value_a_l1412_141269

theorem triangle_value_a (a : ℕ) (h1: a + 2 > 6) (h2: a + 6 > 2) (h3: 2 + 6 > a) : a = 7 :=
sorry

end NUMINAMATH_GPT_triangle_value_a_l1412_141269


namespace NUMINAMATH_GPT_geometric_sequence_product_l1412_141200

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) (n : ℕ) (h : a 4 = 4) :
  a 2 * a 6 = 16 := by
  -- Definition of geomtric sequence
  -- a_n = a_0 * r^n
  -- Using the fact that the product of corresponding terms equidistant from two ends is constant
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l1412_141200


namespace NUMINAMATH_GPT_slope_of_decreasing_linear_function_l1412_141241

theorem slope_of_decreasing_linear_function (m b : ℝ) :
  (∀ x y : ℝ, x < y → mx + b > my + b) → m < 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_slope_of_decreasing_linear_function_l1412_141241


namespace NUMINAMATH_GPT_log_fraction_property_l1412_141231

noncomputable def log_base (a N : ℝ) : ℝ := Real.log N / Real.log a

theorem log_fraction_property :
  (log_base 3 4 / log_base 9 8) = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_log_fraction_property_l1412_141231


namespace NUMINAMATH_GPT_smallest_positive_integer_solution_l1412_141267

theorem smallest_positive_integer_solution :
  ∃ x : ℕ, 0 < x ∧ 5 * x ≡ 17 [MOD 34] ∧ (∀ y : ℕ, 0 < y ∧ 5 * y ≡ 17 [MOD 34] → x ≤ y) :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_solution_l1412_141267


namespace NUMINAMATH_GPT_hexagon_area_l1412_141209

theorem hexagon_area (s t_height : ℕ) (tri_area rect_area : ℕ) :
    s = 2 →
    t_height = 4 →
    tri_area = 1 / 2 * s * t_height →
    rect_area = (s + s + s) * (t_height + t_height) →
    rect_area - 4 * tri_area = 32 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_area_l1412_141209


namespace NUMINAMATH_GPT_circle_area_from_diameter_endpoints_l1412_141203

theorem circle_area_from_diameter_endpoints :
  let C := (-2, 3)
  let D := (4, -1)
  let diameter := Real.sqrt ((4 - (-2))^2 + ((-1) - 3)^2)
  let radius := diameter / 2
  let area := Real.pi * radius^2
  C = (-2, 3) ∧ D = (4, -1) → area = 13 * Real.pi := by
    sorry

end NUMINAMATH_GPT_circle_area_from_diameter_endpoints_l1412_141203


namespace NUMINAMATH_GPT_product_of_roots_l1412_141274

theorem product_of_roots (x : ℝ) (h : x + 16 / x = 12) : (8 : ℝ) * (4 : ℝ) = 32 :=
by
  -- Your proof would go here
  sorry

end NUMINAMATH_GPT_product_of_roots_l1412_141274


namespace NUMINAMATH_GPT_expression_not_defined_at_x_l1412_141228

theorem expression_not_defined_at_x :
  ∃ (x : ℝ), x = 10 ∧ (x^3 - 30 * x^2 + 300 * x - 1000) = 0 := 
sorry

end NUMINAMATH_GPT_expression_not_defined_at_x_l1412_141228


namespace NUMINAMATH_GPT_erasers_per_box_l1412_141218

theorem erasers_per_box (total_erasers : ℕ) (num_boxes : ℕ) (erasers_per_box : ℕ) : total_erasers = 40 → num_boxes = 4 → erasers_per_box = total_erasers / num_boxes → erasers_per_box = 10 :=
by
  intros h_total h_boxes h_div
  rw [h_total, h_boxes] at h_div
  norm_num at h_div
  exact h_div

end NUMINAMATH_GPT_erasers_per_box_l1412_141218


namespace NUMINAMATH_GPT_prism_volume_l1412_141255

-- Define the right triangular prism conditions

variables (AB BC AC : ℝ)
variable (S : ℝ)
variable (volume : ℝ)

-- Given conditions
axiom AB_eq_2 : AB = 2
axiom BC_eq_2 : BC = 2
axiom AC_eq_2sqrt3 : AC = 2 * Real.sqrt 3
axiom circumscribed_sphere_surface_area : S = 32 * Real.pi

-- Statement to prove
theorem prism_volume : volume = 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_prism_volume_l1412_141255


namespace NUMINAMATH_GPT_brian_holds_breath_for_60_seconds_l1412_141282

-- Definitions based on the problem conditions:
def initial_time : ℕ := 10
def after_first_week (t : ℕ) : ℕ := t * 2
def after_second_week (t : ℕ) : ℕ := t * 2
def after_final_week (t : ℕ) : ℕ := (t * 3) / 2

-- The Lean statement to prove:
theorem brian_holds_breath_for_60_seconds :
  after_final_week (after_second_week (after_first_week initial_time)) = 60 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_brian_holds_breath_for_60_seconds_l1412_141282


namespace NUMINAMATH_GPT_largest_possible_integer_smallest_possible_integer_l1412_141226

theorem largest_possible_integer : 3 * (15 + 20 / 4 + 1) = 63 := by
  sorry

theorem smallest_possible_integer : (3 * 15 + 20) / (4 + 1) = 13 := by
  sorry

end NUMINAMATH_GPT_largest_possible_integer_smallest_possible_integer_l1412_141226


namespace NUMINAMATH_GPT_distance_traveled_in_20_seconds_l1412_141202

-- Define the initial distance, common difference, and total time
def initial_distance : ℕ := 8
def common_difference : ℕ := 9
def total_time : ℕ := 20

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℕ := initial_distance + (n - 1) * common_difference

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_terms (n : ℕ) : ℕ := n * (initial_distance + nth_term n) / 2

-- The main theorem to be proven
theorem distance_traveled_in_20_seconds : sum_of_terms 20 = 1870 := 
by sorry

end NUMINAMATH_GPT_distance_traveled_in_20_seconds_l1412_141202


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1412_141261

theorem sufficient_but_not_necessary_condition
  (a b : ℝ) : (a - b) * a^2 < 0 → a < b :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1412_141261


namespace NUMINAMATH_GPT_false_statement_B_l1412_141259

theorem false_statement_B : ¬ ∀ α β : ℝ, (α < 90) ∧ (β < 90) → (α + β > 90) :=
by
  sorry

end NUMINAMATH_GPT_false_statement_B_l1412_141259


namespace NUMINAMATH_GPT_smallest_k_values_l1412_141283

def cos_squared_eq_one (k : ℕ) : Prop :=
  ∃ n : ℕ, k^2 + 49 = 180 * n

theorem smallest_k_values :
  ∃ (k1 k2 : ℕ), (cos_squared_eq_one k1) ∧ (cos_squared_eq_one k2) ∧
  (∀ k < k1, ¬ cos_squared_eq_one k) ∧ (∀ k < k2, ¬ cos_squared_eq_one k) ∧ 
  k1 = 31 ∧ k2 = 37 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_values_l1412_141283


namespace NUMINAMATH_GPT_alicia_read_more_books_than_ian_l1412_141214

def books_read : List Nat := [3, 5, 8, 6, 7, 4, 2, 1]

def alicia_books (books : List Nat) : Nat :=
  books.maximum?.getD 0

def ian_books (books : List Nat) : Nat :=
  books.minimum?.getD 0

theorem alicia_read_more_books_than_ian :
  alicia_books books_read - ian_books books_read = 7 :=
by
  -- By reviewing the given list of books read [3, 5, 8, 6, 7, 4, 2, 1]
  -- We find that alicia_books books_read = 8 and ian_books books_read = 1
  -- Thus, 8 - 1 = 7
  sorry

end NUMINAMATH_GPT_alicia_read_more_books_than_ian_l1412_141214


namespace NUMINAMATH_GPT_distance_qr_eq_b_l1412_141215

theorem distance_qr_eq_b
  (a b c : ℝ)
  (hP : b = c * Real.cosh (a / c))
  (hQ : ∃ Q : ℝ × ℝ, Q = (0, c) ∧ Q.2 = c * Real.cosh (Q.1 / c))
  : QR = b := by
  sorry

end NUMINAMATH_GPT_distance_qr_eq_b_l1412_141215


namespace NUMINAMATH_GPT_average_speed_is_correct_l1412_141286

-- Define the conditions
def initial_odometer : ℕ := 2552
def final_odometer : ℕ := 2882
def time_first_day : ℕ := 5
def time_second_day : ℕ := 7

-- Calculate total time and distance
def total_time : ℕ := time_first_day + time_second_day
def total_distance : ℕ := final_odometer - initial_odometer

-- Prove that the average speed is 27.5 miles per hour
theorem average_speed_is_correct : (total_distance : ℚ) / (total_time : ℚ) = 27.5 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_is_correct_l1412_141286


namespace NUMINAMATH_GPT_three_digit_difference_divisible_by_9_l1412_141280

theorem three_digit_difference_divisible_by_9 :
  ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    (100 * a + 10 * b + c - (a + b + c)) % 9 = 0 :=
by
  intros a b c h
  sorry

end NUMINAMATH_GPT_three_digit_difference_divisible_by_9_l1412_141280


namespace NUMINAMATH_GPT_triangle_area_l1412_141201

def vec2 := ℝ × ℝ

def area_of_triangle (a b : vec2) : ℝ :=
  0.5 * |a.1 * b.2 - a.2 * b.1|

def a : vec2 := (2, -3)
def b : vec2 := (4, -1)

theorem triangle_area : area_of_triangle a b = 5 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l1412_141201


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l1412_141281

noncomputable def f (x : ℝ) : ℝ := abs (4 * x - 1) - abs (x + 2)

-- Part 1: Prove the solution set of f(x) < 8 is -9 / 5 < x < 11 / 3
theorem part1_solution_set : {x : ℝ | f x < 8} = {x : ℝ | -9 / 5 < x ∧ x < 11 / 3} :=
sorry

-- Part 2: Prove the range of a such that the inequality has a solution
theorem part2_range_of_a (a : ℝ) : (∃ x : ℝ, f x + 5 * abs (x + 2) < a^2 - 8 * a) ↔ (a < -1 ∨ a > 9) :=
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l1412_141281


namespace NUMINAMATH_GPT_sum_of_tens_and_units_digit_l1412_141237

theorem sum_of_tens_and_units_digit (n : ℕ) (h : n = 11^2004 - 5) : 
  (n % 100 / 10) + (n % 10) = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_tens_and_units_digit_l1412_141237


namespace NUMINAMATH_GPT_problem_statement_l1412_141272

theorem problem_statement (m : ℂ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2005 = 2006 :=
  sorry

end NUMINAMATH_GPT_problem_statement_l1412_141272


namespace NUMINAMATH_GPT_tiffany_total_bags_l1412_141212

theorem tiffany_total_bags (monday_bags next_day_bags : ℕ) (h1 : monday_bags = 4) (h2 : next_day_bags = 8) :
  monday_bags + next_day_bags = 12 :=
by
  sorry

end NUMINAMATH_GPT_tiffany_total_bags_l1412_141212


namespace NUMINAMATH_GPT_probability_jerry_at_four_l1412_141297

theorem probability_jerry_at_four :
  let total_flips := 8
  let coordinate := 4
  let total_possible_outcomes := 2 ^ total_flips
  let favorable_outcomes := Nat.choose total_flips (total_flips / 2 + coordinate / 2)
  let P := favorable_outcomes / total_possible_outcomes
  let a := 7
  let b := 64
  ∃ (a b : ℕ), Nat.gcd a b = 1 ∧ P = a / b ∧ a + b = 71
:= sorry

end NUMINAMATH_GPT_probability_jerry_at_four_l1412_141297


namespace NUMINAMATH_GPT_Rachel_painting_time_l1412_141275

noncomputable def Matt_time : ℕ := 12
noncomputable def Patty_time (Matt_time : ℕ) : ℕ := Matt_time / 3
noncomputable def Rachel_time (Patty_time : ℕ) : ℕ := 5 + 2 * Patty_time

theorem Rachel_painting_time : Rachel_time (Patty_time Matt_time) = 13 := by
  sorry

end NUMINAMATH_GPT_Rachel_painting_time_l1412_141275


namespace NUMINAMATH_GPT_f_zero_eq_f_expression_alpha_value_l1412_141299

noncomputable def f (ω x : ℝ) : ℝ :=
  3 * Real.sin (ω * x + Real.pi / 6)

theorem f_zero_eq (ω : ℝ) (hω : ω > 0) (h_period : (2 * Real.pi / ω) = Real.pi / 2) :
  f ω 0 = 3 / 2 :=
by
  sorry

theorem f_expression (ω : ℝ) (hω : ω > 0) (h_period : (2 * Real.pi / ω) = Real.pi / 2) :
  ∀ x : ℝ, f ω x = f 4 x :=
by
  sorry

theorem alpha_value (f_4 : ℝ → ℝ) (α : ℝ) (hα : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h_f4 : ∀ x : ℝ, f_4 x = 3 * Real.sin (4 * x + Real.pi / 6)) (h_fα : f_4 (α / 2) = 3 / 2) :
  α = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_f_zero_eq_f_expression_alpha_value_l1412_141299


namespace NUMINAMATH_GPT_true_proposition_l1412_141238

variable (p q : Prop)
variable (hp : p = true)
variable (hq : q = false)

theorem true_proposition : (¬p ∨ ¬q) = true := by
  sorry

end NUMINAMATH_GPT_true_proposition_l1412_141238


namespace NUMINAMATH_GPT_calculate_total_prime_dates_l1412_141242

-- Define the prime months
def prime_months : List Nat := [2, 3, 5, 7, 11, 13]

-- Define the number of days in each month for a non-leap year
def days_in_month (month : Nat) : Nat :=
  if month = 2 then 28
  else if month = 3 then 31
  else if month = 5 then 31
  else if month = 7 then 31
  else if month = 11 then 30
  else if month = 13 then 31
  else 0

-- Define the prime days in a month
def prime_days : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

-- Calculate the number of prime dates in a given month
def prime_dates_in_month (month : Nat) : Nat :=
  (prime_days.filter (λ d => d <= days_in_month month)).length

-- Calculate the total number of prime dates for the year
def total_prime_dates : Nat :=
  (prime_months.map prime_dates_in_month).sum

theorem calculate_total_prime_dates : total_prime_dates = 62 := by
  sorry

end NUMINAMATH_GPT_calculate_total_prime_dates_l1412_141242


namespace NUMINAMATH_GPT_total_marbles_l1412_141256

theorem total_marbles (ratio_red_blue_green_yellow : ℕ → ℕ → ℕ → ℕ → Prop) (total : ℕ) :
  (∀ r b g y, ratio_red_blue_green_yellow r b g y ↔ r = 1 ∧ b = 5 ∧ g = 3 ∧ y = 2) →
  (∃ y, y = 20) →
  (total = y * 11 / 2) →
  total = 110 :=
by
  intros ratio_condition yellow_condition total_condition
  sorry

end NUMINAMATH_GPT_total_marbles_l1412_141256
