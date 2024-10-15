import Mathlib

namespace NUMINAMATH_GPT_linear_function_value_l382_38264

theorem linear_function_value (g : ℝ → ℝ) (h_linear : ∀ x y, g (x + y) = g x + g y)
  (h_scale : ∀ c x, g (c * x) = c * g x) (h : g 10 - g 0 = 20) : g 20 - g 0 = 40 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_value_l382_38264


namespace NUMINAMATH_GPT_smallest_divisor_l382_38278

theorem smallest_divisor (k n : ℕ) (x y : ℤ) :
  (∃ n : ℕ, k ∣ 2^n + 15) ∧ (∃ x y : ℤ, k = 3 * x^2 - 4 * x * y + 3 * y^2) → k = 23 := by
  sorry

end NUMINAMATH_GPT_smallest_divisor_l382_38278


namespace NUMINAMATH_GPT_sum_geometric_sequence_divisibility_l382_38262

theorem sum_geometric_sequence_divisibility (n : ℕ) (h_pos: n > 0) :
  (n % 2 = 1 ↔ (3^(n+1) - 2^(n+1)) % 5 = 0) :=
sorry

end NUMINAMATH_GPT_sum_geometric_sequence_divisibility_l382_38262


namespace NUMINAMATH_GPT_exists_x_such_that_f_x_eq_0_l382_38259

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then
  3 * x - 4
else
  -x^2 + 3 * x - 5

theorem exists_x_such_that_f_x_eq_0 :
  ∃ x : ℝ, f x = 0 ∧ x = 1.192 :=
sorry

end NUMINAMATH_GPT_exists_x_such_that_f_x_eq_0_l382_38259


namespace NUMINAMATH_GPT_max_value_3x_4y_l382_38271

noncomputable def y_geom_mean (x y : ℝ) : Prop :=
  y^2 = (1 - x) * (1 + x)

theorem max_value_3x_4y (x y : ℝ) (h : y_geom_mean x y) : 3 * x + 4 * y ≤ 5 :=
sorry

end NUMINAMATH_GPT_max_value_3x_4y_l382_38271


namespace NUMINAMATH_GPT_gasoline_fraction_used_l382_38234

theorem gasoline_fraction_used
  (speed : ℕ) (gas_usage : ℕ) (initial_gallons : ℕ) (travel_time : ℕ)
  (h_speed : speed = 50) (h_gas_usage : gas_usage = 30) 
  (h_initial_gallons : initial_gallons = 15) (h_travel_time : travel_time = 5) :
  (speed * travel_time) / gas_usage / initial_gallons = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_gasoline_fraction_used_l382_38234


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_a_l382_38266

noncomputable def inequality1 (a x : ℝ) : Prop :=
|a * x - 2| + |a * x - a| ≥ 2

theorem part1_solution_set : 
  (∀ x : ℝ, inequality1 1 x ↔ x ≥ 2.5 ∨ x ≤ 0.5) := 
sorry

theorem part2_range_a :
  (∀ x : ℝ, inequality1 a x) ↔ a ≥ 4 :=
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_a_l382_38266


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l382_38290

noncomputable def problem_statement (a : ℝ) : Prop :=
(a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2)

theorem sufficient_but_not_necessary (a : ℝ) : problem_statement a := 
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l382_38290


namespace NUMINAMATH_GPT_school_students_unique_l382_38246

theorem school_students_unique 
  (n : ℕ)
  (h1 : 70 < n) 
  (h2 : n < 130) 
  (h3 : n % 4 = 2) 
  (h4 : n % 5 = 2)
  (h5 : n % 6 = 2) : 
  (n = 92 ∨ n = 122) :=
  sorry

end NUMINAMATH_GPT_school_students_unique_l382_38246


namespace NUMINAMATH_GPT_arithmetic_sequence_length_l382_38230

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ ∀ (a_1 a_2 a_n : ℤ), a_1 = 2 ∧ a_2 = 6 ∧ a_n = 2006 →
  a_n = a_1 + (n - 1) * (a_2 - a_1) → n = 502 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_length_l382_38230


namespace NUMINAMATH_GPT_max_soap_boxes_in_carton_l382_38273

def carton_volume (length width height : ℕ) : ℕ :=
  length * width * height

def soap_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def max_soap_boxes (carton_volume soap_box_volume : ℕ) : ℕ :=
  carton_volume / soap_box_volume

theorem max_soap_boxes_in_carton :
  max_soap_boxes (carton_volume 25 42 60) (soap_box_volume 7 6 6) = 250 :=
by
  sorry

end NUMINAMATH_GPT_max_soap_boxes_in_carton_l382_38273


namespace NUMINAMATH_GPT_firetruck_reachable_area_l382_38235

theorem firetruck_reachable_area :
  let m := 700
  let n := 31
  let area := m / n -- The area in square miles
  let time := 1 / 10 -- The available time in hours
  let speed_highway := 50 -- Speed on the highway in miles/hour
  let speed_prairie := 14 -- Speed across the prairie in miles/hour
  -- The intersection point of highways is the origin (0, 0)
  -- The firetruck can move within the reachable area
  -- There exist regions formed by the intersection points of movement directions
  m + n = 731 :=
by
  sorry

end NUMINAMATH_GPT_firetruck_reachable_area_l382_38235


namespace NUMINAMATH_GPT_calc_expression_value_l382_38216

open Real

theorem calc_expression_value :
  sqrt ((16: ℝ) ^ 12 + (8: ℝ) ^ 15) / ((16: ℝ) ^ 5 + (8: ℝ) ^ 16) = (3 * sqrt 2) / 4 := sorry

end NUMINAMATH_GPT_calc_expression_value_l382_38216


namespace NUMINAMATH_GPT_probability_red_joker_is_1_over_54_l382_38219

-- Define the conditions as given in the problem
def total_cards : ℕ := 54
def red_joker_count : ℕ := 1

-- Define the function to calculate the probability
def probability_red_joker_top_card : ℚ := red_joker_count / total_cards

-- Problem: Prove that the probability of drawing the red joker as the top card is 1/54
theorem probability_red_joker_is_1_over_54 :
  probability_red_joker_top_card = 1 / 54 :=
by
  sorry

end NUMINAMATH_GPT_probability_red_joker_is_1_over_54_l382_38219


namespace NUMINAMATH_GPT_product_even_if_sum_odd_l382_38203

theorem product_even_if_sum_odd (a b : ℤ) (h : (a + b) % 2 = 1) : (a * b) % 2 = 0 :=
sorry

end NUMINAMATH_GPT_product_even_if_sum_odd_l382_38203


namespace NUMINAMATH_GPT_multiply_difference_of_cubes_l382_38276

def multiply_and_simplify (x : ℝ) : ℝ :=
  (x^4 + 25 * x^2 + 625) * (x^2 - 25)

theorem multiply_difference_of_cubes (x : ℝ) :
  multiply_and_simplify x = x^6 - 15625 :=
by
  sorry

end NUMINAMATH_GPT_multiply_difference_of_cubes_l382_38276


namespace NUMINAMATH_GPT_probability_correct_l382_38277

noncomputable def probability_two_queens_or_at_least_one_jack : ℚ :=
  let total_cards := 52
  let queens := 3
  let jacks := 1
  let prob_two_queens := (queens * (queens - 1)) / (total_cards * (total_cards - 1))
  let prob_one_jack := jacks / total_cards * (total_cards - jacks) / (total_cards - 1) + (total_cards - jacks) / total_cards * jacks / (total_cards - 1)
  prob_two_queens + prob_one_jack

theorem probability_correct : probability_two_queens_or_at_least_one_jack = 9 / 221 := by
  sorry

end NUMINAMATH_GPT_probability_correct_l382_38277


namespace NUMINAMATH_GPT_f_a_plus_b_eq_neg_one_l382_38204

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if x ≥ 0 then x * (x - b) else a * x * (x + 2)

theorem f_a_plus_b_eq_neg_one (a b : ℝ) 
  (h_odd : ∀ x : ℝ, f (-x) a b = -f x a b) 
  (ha : a = -1) 
  (hb : b = 2) : 
  f (a + b) a b = -1 :=
by
  sorry

end NUMINAMATH_GPT_f_a_plus_b_eq_neg_one_l382_38204


namespace NUMINAMATH_GPT_relay_race_team_members_l382_38220

theorem relay_race_team_members (n : ℕ) (d : ℕ) (h1 : n = 5) (h2 : d = 150) : d / n = 30 := 
by {
  -- Place the conditions here as hypotheses
  sorry
}

end NUMINAMATH_GPT_relay_race_team_members_l382_38220


namespace NUMINAMATH_GPT_ab_leq_1_l382_38223

theorem ab_leq_1 {a b : ℝ} (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 2) : ab ≤ 1 :=
sorry

end NUMINAMATH_GPT_ab_leq_1_l382_38223


namespace NUMINAMATH_GPT_find_a_l382_38245

def f (a : ℝ) (x : ℝ) := a * x^2 + 3 * x - 2

theorem find_a (a : ℝ) (h : deriv (f a) 2 = 7) : a = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_l382_38245


namespace NUMINAMATH_GPT_tank_full_capacity_l382_38291

theorem tank_full_capacity (w c : ℕ) (h1 : w = c / 6) (h2 : w + 4 = c / 3) : c = 12 :=
sorry

end NUMINAMATH_GPT_tank_full_capacity_l382_38291


namespace NUMINAMATH_GPT_pentagonal_tiles_count_l382_38252

theorem pentagonal_tiles_count (t s p : ℕ) 
  (h1 : t + s + p = 30) 
  (h2 : 3 * t + 4 * s + 5 * p = 120) : 
  p = 10 := by
  sorry

end NUMINAMATH_GPT_pentagonal_tiles_count_l382_38252


namespace NUMINAMATH_GPT_solve_for_x_l382_38217

theorem solve_for_x (x : ℚ) : (1 / 3) + (1 / x) = (3 / 4) → x = 12 / 5 :=
by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_solve_for_x_l382_38217


namespace NUMINAMATH_GPT_alice_bob_meet_same_point_in_5_turns_l382_38215

theorem alice_bob_meet_same_point_in_5_turns :
  ∃ k : ℕ, k = 5 ∧ 
  (∀ n, (1 + 7 * n) % 24 = 12 ↔ (n = k)) :=
by
  sorry

end NUMINAMATH_GPT_alice_bob_meet_same_point_in_5_turns_l382_38215


namespace NUMINAMATH_GPT_linear_function_third_quadrant_and_origin_l382_38263

theorem linear_function_third_quadrant_and_origin (k b : ℝ) (h1 : ∀ x < 0, k * x + b ≥ 0) (h2 : k * 0 + b ≠ 0) : k < 0 ∧ b > 0 :=
sorry

end NUMINAMATH_GPT_linear_function_third_quadrant_and_origin_l382_38263


namespace NUMINAMATH_GPT_find_angle_l382_38240

-- Definitions based on conditions
def is_complement (x : ℝ) : ℝ := 90 - x
def is_supplement (x : ℝ) : ℝ := 180 - x

-- Main statement
theorem find_angle (x : ℝ) (h : is_supplement x = 15 + 4 * is_complement x) : x = 65 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_l382_38240


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l382_38221

theorem arithmetic_sequence_fifth_term (x y : ℚ) 
  (h1 : a₁ = x + y) 
  (h2 : a₂ = x - y) 
  (h3 : a₃ = x * y) 
  (h4 : a₄ = x / y) 
  (h5 : a₂ - a₁ = -2 * y) 
  (h6 : a₃ - a₂ = -2 * y) 
  (h7 : a₄ - a₃ = -2 * y) 
  (hx : x = -9 / 8)
  (hy : y = -3 / 5) : 
  a₅ = 123 / 40 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l382_38221


namespace NUMINAMATH_GPT_pigs_count_l382_38261

-- Definitions from step a)
def pigs_leg_count : ℕ := 4 -- Each pig has 4 legs
def hens_leg_count : ℕ := 2 -- Each hen has 2 legs

variable {P H : ℕ} -- P is the number of pigs, H is the number of hens

-- Condition from step a) as a function
def total_legs (P H : ℕ) : ℕ := pigs_leg_count * P + hens_leg_count * H
def total_heads (P H : ℕ) : ℕ := P + H

-- Theorem to prove the number of pigs given the condition
theorem pigs_count {P H : ℕ} (h : total_legs P H = 2 * total_heads P H + 22) : P = 11 :=
  by 
    sorry

end NUMINAMATH_GPT_pigs_count_l382_38261


namespace NUMINAMATH_GPT_customer_ordered_bags_l382_38251

def bags_per_batch : Nat := 10
def initial_bags : Nat := 20
def days : Nat := 4
def batches_per_day : Nat := 1

theorem customer_ordered_bags : 
  initial_bags + days * batches_per_day * bags_per_batch = 60 :=
by
  sorry

end NUMINAMATH_GPT_customer_ordered_bags_l382_38251


namespace NUMINAMATH_GPT_all_odd_digits_n_squared_l382_38268

/-- Helper function to check if all digits in a number are odd -/
def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 1

/-- Main theorem stating that the only positive integers n such that all the digits of n^2 are odd are 1 and 3 -/
theorem all_odd_digits_n_squared (n : ℕ) :
  (n > 0) → (all_odd_digits (n^2)) → (n = 1 ∨ n = 3) :=
by
  sorry

end NUMINAMATH_GPT_all_odd_digits_n_squared_l382_38268


namespace NUMINAMATH_GPT_largest_two_digit_divisible_by_6_and_ends_in_4_l382_38250

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end NUMINAMATH_GPT_largest_two_digit_divisible_by_6_and_ends_in_4_l382_38250


namespace NUMINAMATH_GPT_range_of_a_l382_38218

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a*x^2 - a*x - 2 ≤ 0) → (-8 ≤ a ∧ a ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l382_38218


namespace NUMINAMATH_GPT_common_volume_of_tetrahedra_l382_38228

open Real

noncomputable def volume_of_common_part (a b c : ℝ) : ℝ :=
  min (a * sqrt 3 / 12) (min (b * sqrt 3 / 12) (c * sqrt 3 / 12))

theorem common_volume_of_tetrahedra (a b c : ℝ) :
  volume_of_common_part a b c =
  min (a * sqrt 3 / 12) (min (b * sqrt 3 / 12) (c * sqrt 3 / 12)) :=
by sorry

end NUMINAMATH_GPT_common_volume_of_tetrahedra_l382_38228


namespace NUMINAMATH_GPT_parabola_x_intercepts_l382_38239

theorem parabola_x_intercepts :
  ∃! (x : ℝ), ∃ (y : ℝ), y = 0 ∧ x = -2 * y^2 + y + 1 :=
sorry

end NUMINAMATH_GPT_parabola_x_intercepts_l382_38239


namespace NUMINAMATH_GPT_total_wicks_20_l382_38265

theorem total_wicks_20 (string_length_ft : ℕ) (length_wick_1 length_wick_2 : ℕ) (wicks_1 wicks_2 : ℕ) :
  string_length_ft = 15 →
  length_wick_1 = 6 →
  length_wick_2 = 12 →
  wicks_1 = wicks_2 →
  (string_length_ft * 12) = (length_wick_1 * wicks_1 + length_wick_2 * wicks_2) →
  (wicks_1 + wicks_2) = 20 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_total_wicks_20_l382_38265


namespace NUMINAMATH_GPT_hyperbola_problem_l382_38222

noncomputable def is_hyperbola (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) - ((y - 2)^2 / b^2) = 1

variables (s : ℝ)

theorem hyperbola_problem
  (h₁ : is_hyperbola 0 5 a b)
  (h₂ : is_hyperbola (-1) 6 a b)
  (h₃ : is_hyperbola s 3 a b)
  (hb : b^2 = 9)
  (ha : a^2 = 9 / 25) :
  s^2 = 2 / 5 :=
sorry

end NUMINAMATH_GPT_hyperbola_problem_l382_38222


namespace NUMINAMATH_GPT_pipe_flow_rate_is_correct_l382_38298

-- Definitions for the conditions
def tank_capacity : ℕ := 10000
def initial_water : ℕ := tank_capacity / 2
def fill_time : ℕ := 60
def drain1_rate : ℕ := 1000
def drain1_interval : ℕ := 4
def drain2_rate : ℕ := 1000
def drain2_interval : ℕ := 6

-- Calculation based on conditions
def total_water_needed : ℕ := tank_capacity - initial_water
def drain1_loss (time : ℕ) : ℕ := (time / drain1_interval) * drain1_rate
def drain2_loss (time : ℕ) : ℕ := (time / drain2_interval) * drain2_rate
def total_drain_loss (time : ℕ) : ℕ := drain1_loss time + drain2_loss time

-- Target flow rate for the proof
def total_fill (time : ℕ) : ℕ := total_water_needed + total_drain_loss time
def pipe_flow_rate : ℕ := total_fill fill_time / fill_time

-- Statement to prove
theorem pipe_flow_rate_is_correct : pipe_flow_rate = 500 := by  
  sorry

end NUMINAMATH_GPT_pipe_flow_rate_is_correct_l382_38298


namespace NUMINAMATH_GPT_min_expression_value_l382_38236

def distinct_elements (s : Set ℤ) : Prop := s = {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_expression_value :
  ∃ (p q r s t u v w : ℤ),
    distinct_elements {p, q, r, s, t, u, v, w} ∧
    (p + q + r + s) ≥ 5 ∧
    (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
     q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
     r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
     s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
     t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
     u ≠ v ∧ u ≠ w ∧
     v ≠ w) →
    (p + q + r + s)^2 + (t + u + v + w)^2 = 26 :=
sorry

end NUMINAMATH_GPT_min_expression_value_l382_38236


namespace NUMINAMATH_GPT_sum_of_sequence_l382_38253

noncomputable def sequence_sum (a : ℝ) (n : ℕ) : ℝ :=
if a = 1 then sorry else (5 * (1 - a ^ n) / (1 - a) ^ 2) - (4 + (5 * n - 4) * a ^ n) / (1 - a)

theorem sum_of_sequence (S : ℕ → ℝ) (a : ℝ) (h1 : S 1 = 1)
                       (h2 : ∀ n, S (n + 1) - S n = (5 * n + 1) * a ^ n) (h3 : |a| ≠ 1) :
  ∀ n, S n = sequence_sum a n :=
  sorry

end NUMINAMATH_GPT_sum_of_sequence_l382_38253


namespace NUMINAMATH_GPT_counterexample_exists_l382_38210

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop := ¬ is_prime n ∧ n > 1

theorem counterexample_exists :
  ∃ n, is_composite n ∧ is_composite (n - 3) ∧ n = 18 := by
  sorry

end NUMINAMATH_GPT_counterexample_exists_l382_38210


namespace NUMINAMATH_GPT_average_weight_is_15_l382_38201

-- Define the ages of the 10 children
def ages : List ℕ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

-- Define the regression function
def weight (age : ℕ) : ℕ := 2 * age + 7

-- Function to calculate average
def average (l : List ℕ) : ℕ := l.sum / l.length

-- Define the weights of the children based on the regression function
def weights : List ℕ := ages.map weight

-- State the theorem to find the average weight of the children
theorem average_weight_is_15 : average weights = 15 := by
  sorry

end NUMINAMATH_GPT_average_weight_is_15_l382_38201


namespace NUMINAMATH_GPT_at_most_two_zero_points_l382_38296

noncomputable def f (x a : ℝ) := x^3 - 12 * x + a

theorem at_most_two_zero_points (a : ℝ) (h : a ≥ 16) : ∃ l u : ℝ, (∀ x : ℝ, f x a = 0 → x < l ∨ l ≤ x ∧ x ≤ u ∨ u < x) := sorry

end NUMINAMATH_GPT_at_most_two_zero_points_l382_38296


namespace NUMINAMATH_GPT_gcd_of_q_and_r_l382_38233

theorem gcd_of_q_and_r (p q r : ℕ) (hpq : p > 0) (hqr : q > 0) (hpr : r > 0)
    (gcd_pq : Nat.gcd p q = 240) (gcd_pr : Nat.gcd p r = 540) : Nat.gcd q r = 60 := by
  sorry

end NUMINAMATH_GPT_gcd_of_q_and_r_l382_38233


namespace NUMINAMATH_GPT_width_of_metallic_sheet_l382_38241

theorem width_of_metallic_sheet 
  (length : ℕ)
  (new_volume : ℕ) 
  (side_length_of_square : ℕ)
  (height_of_box : ℕ)
  (new_length : ℕ)
  (new_width : ℕ)
  (w : ℕ) : 
  length = 48 → 
  new_volume = 5120 → 
  side_length_of_square = 8 → 
  height_of_box = 8 → 
  new_length = length - 2 * side_length_of_square → 
  new_width = w - 2 * side_length_of_square → 
  new_volume = new_length * new_width * height_of_box → 
  w = 36 := 
by 
  intros _ _ _ _ _ _ _ 
  sorry

end NUMINAMATH_GPT_width_of_metallic_sheet_l382_38241


namespace NUMINAMATH_GPT_reinforcement_calculation_l382_38269

theorem reinforcement_calculation
  (initial_men : ℕ := 2000)
  (initial_days : ℕ := 40)
  (days_until_reinforcement : ℕ := 20)
  (additional_days_post_reinforcement : ℕ := 10)
  (total_initial_provisions : ℕ := initial_men * initial_days)
  (remaining_provisions_post_20_days : ℕ := total_initial_provisions / 2)
  : ∃ (reinforcement_men : ℕ), reinforcement_men = 2000 :=
by
  have remaining_provisions := remaining_provisions_post_20_days
  have total_post_reinforcement := initial_men + ((remaining_provisions) / (additional_days_post_reinforcement))

  use (total_post_reinforcement - initial_men)
  sorry

end NUMINAMATH_GPT_reinforcement_calculation_l382_38269


namespace NUMINAMATH_GPT_amount_left_after_expenses_l382_38200

namespace GirlScouts

def totalEarnings : ℝ := 30
def poolEntryCosts : ℝ :=
  5 * 3.5 + 3 * 2.0 + 2 * 1.0
def transportationCosts : ℝ :=
  6 * 1.5 + 4 * 0.75
def snackCosts : ℝ :=
  3 * 3.0 + 4 * 2.5 + 3 * 2.0
def totalExpenses : ℝ :=
  poolEntryCosts + transportationCosts + snackCosts
def amountLeft : ℝ :=
  totalEarnings - totalExpenses

theorem amount_left_after_expenses :
  amountLeft = -32.5 :=
by
  sorry

end GirlScouts

end NUMINAMATH_GPT_amount_left_after_expenses_l382_38200


namespace NUMINAMATH_GPT_part1_part2_l382_38294

variable {a b c : ℚ}

theorem part1 (ha : a < 0) : (a / |a|) = -1 :=
sorry

theorem part2 (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  min (a * b / |a * b| + |b * c| / (b * c) + a * c / |a * c| + |a * b * c| / (a * b * c)) (-2) = -2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l382_38294


namespace NUMINAMATH_GPT_cats_and_dogs_biscuits_l382_38207

theorem cats_and_dogs_biscuits 
  (d c : ℕ) 
  (h1 : d + c = 10) 
  (h2 : 6 * d + 5 * c = 56) 
  : d = 6 ∧ c = 4 := 
by 
  sorry

end NUMINAMATH_GPT_cats_and_dogs_biscuits_l382_38207


namespace NUMINAMATH_GPT_emily_fishes_correct_l382_38297

/-- Given conditions:
1. Emily caught 4 trout weighing 2 pounds each.
2. Emily caught 3 catfish weighing 1.5 pounds each.
3. Bluegills weigh 2.5 pounds each.
4. Emily caught a total of 25 pounds of fish. -/
def emilyCatches : Prop :=
  ∃ (trout_count catfish_count bluegill_count : ℕ)
    (trout_weight catfish_weight bluegill_weight total_weight : ℝ),
    trout_count = 4 ∧ catfish_count = 3 ∧ 
    trout_weight = 2 ∧ catfish_weight = 1.5 ∧ 
    bluegill_weight = 2.5 ∧ 
    total_weight = 25 ∧
    (total_weight = (trout_count * trout_weight) + (catfish_count * catfish_weight) + (bluegill_count * bluegill_weight)) ∧
    bluegill_count = 5

theorem emily_fishes_correct : emilyCatches := by
  sorry

end NUMINAMATH_GPT_emily_fishes_correct_l382_38297


namespace NUMINAMATH_GPT_initial_volume_of_mixture_l382_38242

/-- A mixture contains 10% water. 
5 liters of water should be added to this so that the water becomes 20% in the new mixture.
Prove that the initial volume of the mixture is 40 liters. -/
theorem initial_volume_of_mixture 
  (V : ℚ) -- Define the initial volume of the mixture
  (h1 : 0.10 * V + 5 = 0.20 * (V + 5)) -- Condition on the mixture
  : V = 40 := -- The statement to prove
by
  sorry -- Proof not required

end NUMINAMATH_GPT_initial_volume_of_mixture_l382_38242


namespace NUMINAMATH_GPT_constant_chromosome_number_l382_38224

theorem constant_chromosome_number (rabbits : Type) 
  (sex_reproduction : rabbits → Prop)
  (maintain_chromosome_number : Prop)
  (meiosis : Prop)
  (fertilization : Prop) : 
  (meiosis ∧ fertilization) ↔ maintain_chromosome_number :=
sorry

end NUMINAMATH_GPT_constant_chromosome_number_l382_38224


namespace NUMINAMATH_GPT_cost_of_iPhone_l382_38255

theorem cost_of_iPhone (P : ℝ) 
  (phone_contract_cost : ℝ := 200)
  (case_percent_of_P : ℝ := 0.20)
  (headphones_percent_of_case : ℝ := 0.50)
  (total_yearly_cost : ℝ := 3700) :
  let year_phone_contract_cost := (phone_contract_cost * 12)
  let case_cost := (case_percent_of_P * P)
  let headphones_cost := (headphones_percent_of_case * case_cost)
  P + year_phone_contract_cost + case_cost + headphones_cost = total_yearly_cost → 
  P = 1000 :=
by
  sorry  -- proof not required

end NUMINAMATH_GPT_cost_of_iPhone_l382_38255


namespace NUMINAMATH_GPT_find_a_l382_38257

theorem find_a (a : ℝ) : (1 : ℝ)^2 + 1 + 2 * a = 0 → a = -1 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l382_38257


namespace NUMINAMATH_GPT_find_value_of_expression_l382_38295

variables (a b : ℝ)

-- Given the condition that 2a - 3b = 5, prove that 2a - 3b + 3 = 8.
theorem find_value_of_expression
  (h : 2 * a - 3 * b = 5) : 2 * a - 3 * b + 3 = 8 :=
by sorry

end NUMINAMATH_GPT_find_value_of_expression_l382_38295


namespace NUMINAMATH_GPT_solve_for_x_l382_38243

variable (a b c d x : ℝ)

theorem solve_for_x (h1 : a ≠ b) (h2 : b ≠ 0) 
  (h3 : d ≠ c) (h4 : c % x = 0) (h5 : d % x = 0) 
  (h6 : (2*a + x) / (3*b + x) = c / d) : 
  x = (3*b*c - 2*a*d) / (d - c) := 
sorry

end NUMINAMATH_GPT_solve_for_x_l382_38243


namespace NUMINAMATH_GPT_initial_population_l382_38286

theorem initial_population (P : ℝ) (h1 : 1.05 * (0.765 * P + 50) = 3213) : P = 3935 :=
by
  have h2 : 1.05 * (0.765 * P + 50) = 3213 := h1
  sorry

end NUMINAMATH_GPT_initial_population_l382_38286


namespace NUMINAMATH_GPT_same_grade_percentage_is_correct_l382_38229

def total_students : ℕ := 40

def grade_distribution : ℕ × ℕ × ℕ × ℕ :=
  (17, 40, 100)

def same_grade_percentage (total_students : ℕ) (same_grade_students : ℕ) : ℚ :=
  (same_grade_students / total_students) * 100

theorem same_grade_percentage_is_correct :
  let same_grade_students := 3 + 5 + 6 + 3
  same_grade_percentage total_students same_grade_students = 42.5 :=
by 
let same_grade_students := 3 + 5 + 6 + 3
show same_grade_percentage total_students same_grade_students = 42.5
sorry

end NUMINAMATH_GPT_same_grade_percentage_is_correct_l382_38229


namespace NUMINAMATH_GPT_volume_of_TABC_l382_38282

noncomputable def volume_pyramid_TABC : ℝ :=
  let TA : ℝ := 15
  let TB : ℝ := 15
  let TC : ℝ := 5 * Real.sqrt 3
  let area_ABT : ℝ := (1 / 2) * TA * TB
  (1 / 3) * area_ABT * TC

theorem volume_of_TABC :
  volume_pyramid_TABC = 187.5 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_volume_of_TABC_l382_38282


namespace NUMINAMATH_GPT_quarters_percentage_value_l382_38232

theorem quarters_percentage_value (dimes quarters : Nat) (value_dime value_quarter : Nat) (total_value quarter_value : Nat)
(h_dimes : dimes = 30)
(h_quarters : quarters = 40)
(h_value_dime : value_dime = 10)
(h_value_quarter : value_quarter = 25)
(h_total_value : total_value = dimes * value_dime + quarters * value_quarter)
(h_quarter_value : quarter_value = quarters * value_quarter) :
(quarter_value : ℚ) / (total_value : ℚ) * 100 = 76.92 := 
sorry

end NUMINAMATH_GPT_quarters_percentage_value_l382_38232


namespace NUMINAMATH_GPT_last_integer_in_sequence_is_21853_l382_38231

def is_divisible_by (n m : ℕ) : Prop := 
  ∃ k : ℕ, n = m * k

-- Conditions
def starts_with : ℕ := 590049
def divides_previous (a b : ℕ) : Prop := b = a / 3

-- The target hypothesis to prove
theorem last_integer_in_sequence_is_21853 :
  ∀ (a b c d : ℕ),
    a = starts_with →
    divides_previous a b →
    divides_previous b c →
    divides_previous c d →
    ¬ is_divisible_by d 3 →
    d = 21853 :=
by
  intros a b c d ha hb hc hd hnd
  sorry

end NUMINAMATH_GPT_last_integer_in_sequence_is_21853_l382_38231


namespace NUMINAMATH_GPT_ball_bounces_before_vertex_l382_38288

def bounces_to_vertex (v h : ℕ) (units_per_bounce_vert units_per_bounce_hor : ℕ) : ℕ :=
units_per_bounce_vert * v / units_per_bounce_hor * h

theorem ball_bounces_before_vertex (verts : ℕ) (h : ℕ) (units_per_bounce_vert units_per_bounce_hor : ℕ)
    (H_vert : verts = 10)
    (H_units_vert : units_per_bounce_vert = 2)
    (H_units_hor : units_per_bounce_hor = 7) :
    bounces_to_vertex verts h units_per_bounce_vert units_per_bounce_hor = 5 := 
by
  sorry

end NUMINAMATH_GPT_ball_bounces_before_vertex_l382_38288


namespace NUMINAMATH_GPT_sum_series_eq_four_ninths_l382_38205

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_series_eq_four_ninths_l382_38205


namespace NUMINAMATH_GPT_derek_joe_ratio_l382_38247

theorem derek_joe_ratio (D J T : ℝ) (h0 : J = 23) (h1 : T = 30) (h2 : T = (1/3 : ℝ) * D + 16) :
  D / J = 42 / 23 :=
by
  sorry

end NUMINAMATH_GPT_derek_joe_ratio_l382_38247


namespace NUMINAMATH_GPT_house_height_proof_l382_38237

noncomputable def height_of_house (house_shadow tree_height tree_shadow : ℕ) : ℕ :=
  house_shadow * tree_height / tree_shadow

theorem house_height_proof
  (house_shadow_length : ℕ)
  (tree_height : ℕ)
  (tree_shadow_length : ℕ)
  (expected_house_height : ℕ)
  (Hhouse_shadow_length : house_shadow_length = 56)
  (Htree_height : tree_height = 21)
  (Htree_shadow_length : tree_shadow_length = 24)
  (Hexpected_house_height : expected_house_height = 49) :
  height_of_house house_shadow_length tree_height tree_shadow_length = expected_house_height :=
by
  rw [Hhouse_shadow_length, Htree_height, Htree_shadow_length, Hexpected_house_height]
  -- Here we should compute the value and show it is equal to 49
  sorry

end NUMINAMATH_GPT_house_height_proof_l382_38237


namespace NUMINAMATH_GPT_range_of_a_l382_38249

variable (a x : ℝ)

theorem range_of_a (h : ax > 2) (h_transform: ax > 2 → x < 2/a) : a < 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l382_38249


namespace NUMINAMATH_GPT_pure_imaginary_a_zero_l382_38279

theorem pure_imaginary_a_zero (a : ℝ) (h : ∃ b : ℝ, (i : ℂ) * (1 + (a : ℂ) * i) = (b : ℂ) * i) : a = 0 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_a_zero_l382_38279


namespace NUMINAMATH_GPT_find_abc_values_l382_38209

-- Define the problem conditions as lean definitions
def represents_circle (a b c : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * a * x - b * y + c = 0

def circle_center_and_radius_condition (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 3^2

-- Lean 4 statement for the proof problem
theorem find_abc_values (a b c : ℝ) :
  (∀ x y : ℝ, represents_circle a b c x y ↔ circle_center_and_radius_condition x y) →
  a = -2 ∧ b = 6 ∧ c = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_abc_values_l382_38209


namespace NUMINAMATH_GPT_least_number_correct_l382_38283

def least_number_to_add_to_make_perfect_square (x : ℝ) : ℝ :=
  let y := 1 - x -- since 1 is the smallest whole number > sqrt(0.0320)
  y

theorem least_number_correct (x : ℝ) (h : x = 0.0320) : least_number_to_add_to_make_perfect_square x = 0.9680 :=
by {
  -- Proof is skipped
  -- The proof would involve verifying that adding this number to x results in a perfect square (1 in this case).
  sorry
}

end NUMINAMATH_GPT_least_number_correct_l382_38283


namespace NUMINAMATH_GPT_smallest_b_is_2_plus_sqrt_3_l382_38214

open Real

noncomputable def smallest_b (a b : ℝ) : ℝ :=
  if (2 < a ∧ a < b ∧ (¬(2 + a > b ∧ 2 + b > a ∧ a + b > 2)) ∧
    (¬(1 / b + 1 / a > 2 ∧ 1 / a + 2 > 1 / b ∧ 2 + 1 / b > 1 / a)))
  then b else 0

theorem smallest_b_is_2_plus_sqrt_3 (a b : ℝ) :
  2 < a ∧ a < b ∧ (¬(2 + a > b ∧ 2 + b > a ∧ a + b > 2)) ∧
    (¬(1 / b + 1 / a > 2 ∧ 1 / a + 2 > 1 / b ∧ 2 + 1 / b > 1 / a)) →
  b = 2 + sqrt 3 := sorry

end NUMINAMATH_GPT_smallest_b_is_2_plus_sqrt_3_l382_38214


namespace NUMINAMATH_GPT_negate_neg_two_l382_38292

theorem negate_neg_two : -(-2) = 2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_negate_neg_two_l382_38292


namespace NUMINAMATH_GPT_clara_total_points_l382_38212

-- Define the constants
def percentage_three_point_shots : ℝ := 0.25
def points_per_successful_three_point_shot : ℝ := 3
def percentage_two_point_shots : ℝ := 0.40
def points_per_successful_two_point_shot : ℝ := 2
def total_attempts : ℕ := 40

-- Define the function to calculate the total score
def total_score (x y : ℕ) : ℝ :=
  (percentage_three_point_shots * points_per_successful_three_point_shot) * x +
  (percentage_two_point_shots * points_per_successful_two_point_shot) * y

-- The proof statement
theorem clara_total_points (x y : ℕ) (h : x + y = total_attempts) : 
  total_score x y = 32 :=
by
  -- This is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_clara_total_points_l382_38212


namespace NUMINAMATH_GPT_simplify_first_expression_simplify_second_expression_l382_38281

theorem simplify_first_expression (x y : ℝ) : 3 * x - 2 * y + 1 + 3 * y - 2 * x - 5 = x + y - 4 :=
sorry

theorem simplify_second_expression (x : ℝ) : (2 * x ^ 4 - 5 * x ^ 2 - 4 * x + 3) - (3 * x ^ 3 - 5 * x ^ 2 - 4 * x) = 2 * x ^ 4 - 3 * x ^ 3 + 3 :=
sorry

end NUMINAMATH_GPT_simplify_first_expression_simplify_second_expression_l382_38281


namespace NUMINAMATH_GPT_find_floor_of_apt_l382_38274

-- Define the conditions:
-- Number of stories
def num_stories : Nat := 9
-- Number of entrances
def num_entrances : Nat := 10
-- Total apartments in entrance 10
def apt_num : Nat := 333
-- Number of apartments per floor in each entrance (which is to be found)
def apts_per_floor_per_entrance : Nat := 4 -- from solution b)

-- Assertion: The floor number that apartment number 333 is on in entrance 10
theorem find_floor_of_apt (num_stories num_entrances apt_num apts_per_floor_per_entrance : ℕ) :
  1 ≤ apt_num ∧ apt_num ≤ num_stories * num_entrances * apts_per_floor_per_entrance →
  (apt_num - 1) / apts_per_floor_per_entrance + 1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_floor_of_apt_l382_38274


namespace NUMINAMATH_GPT_grid_problem_l382_38285

theorem grid_problem 
  (n m : ℕ) 
  (h1 : ∀ (blue_cells : ℕ), blue_cells = m + n - 1 → (n * m ≠ 0) → (blue_cells = (n * m) / 2010)) :
  ∃ (k : ℕ), k = 96 :=
by
  sorry

end NUMINAMATH_GPT_grid_problem_l382_38285


namespace NUMINAMATH_GPT_arcsin_sqrt3_over_2_eq_pi_over_3_l382_38270

theorem arcsin_sqrt3_over_2_eq_pi_over_3 :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
by
  have h : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
    -- This is a known trigonometric identity.
    sorry
  -- Use the property of arcsin to get the result.
  sorry

end NUMINAMATH_GPT_arcsin_sqrt3_over_2_eq_pi_over_3_l382_38270


namespace NUMINAMATH_GPT_solve_for_x_l382_38299

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 17) : x = 4 := 
by sorry

end NUMINAMATH_GPT_solve_for_x_l382_38299


namespace NUMINAMATH_GPT_number_of_positive_expressions_l382_38287

-- Define the conditions
variable (a b c : ℝ)
variable (h_a : a < 0)
variable (h_b : b > 0)
variable (h_c : c < 0)

-- Define the expressions
def ab := a * b
def ac := a * c
def a_b_c := a + b + c
def a_minus_b_c := a - b + c
def two_a_plus_b := 2 * a + b
def two_a_minus_b := 2 * a - b

-- Problem statement
theorem number_of_positive_expressions :
  (ab < 0) → (ac > 0) → (a_b_c > 0) → (a_minus_b_c < 0) → (two_a_plus_b < 0) → (two_a_minus_b < 0)
  → (2 = 2) :=
by
  sorry

end NUMINAMATH_GPT_number_of_positive_expressions_l382_38287


namespace NUMINAMATH_GPT_vertices_sum_zero_l382_38226

theorem vertices_sum_zero
  (a b c d e f g h : ℝ)
  (h1 : a = (b + e + d) / 3)
  (h2 : b = (c + f + a) / 3)
  (h3 : c = (d + g + b) / 3)
  (h4 : d = (a + h + e) / 3)
  :
  (a + b + c + d) - (e + f + g + h) = 0 :=
by
  sorry

end NUMINAMATH_GPT_vertices_sum_zero_l382_38226


namespace NUMINAMATH_GPT_bug_visits_tiles_l382_38284

theorem bug_visits_tiles (width length : ℕ) (gcd_width_length : ℕ) (broken_tile : ℕ × ℕ)
  (h_width : width = 12) (h_length : length = 25) (h_gcd : gcd_width_length = Nat.gcd width length)
  (h_broken_tile : broken_tile = (12, 18)) :
  width + length - gcd_width_length = 36 := by
  sorry

end NUMINAMATH_GPT_bug_visits_tiles_l382_38284


namespace NUMINAMATH_GPT_find_sr_division_l382_38248

theorem find_sr_division (k : ℚ) (c r s : ℚ)
  (h_c : c = 10)
  (h_r : r = -3 / 10)
  (h_s : s = 191 / 10)
  (h_expr : 10 * k^2 - 6 * k + 20 = c * (k + r)^2 + s) :
  s / r = -191 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_sr_division_l382_38248


namespace NUMINAMATH_GPT_sum_possible_values_l382_38254

theorem sum_possible_values (N : ℤ) (h : N * (N - 8) = -7) : 
  ∀ (N1 N2 : ℤ), (N1 * (N1 - 8) = -7) ∧ (N2 * (N2 - 8) = -7) → (N1 + N2 = 8) :=
by
  sorry

end NUMINAMATH_GPT_sum_possible_values_l382_38254


namespace NUMINAMATH_GPT_problem_I_problem_II_l382_38208

def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 3)

theorem problem_I (x : ℝ) : (f x > 7 - x) ↔ (x < -6 ∨ x > 2) := 
by 
  sorry

theorem problem_II (m : ℝ) : (∃ x : ℝ, f x ≤ abs (3 * m - 2)) ↔ (m ≤ -1 ∨ m ≥ 7 / 3) := 
by 
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l382_38208


namespace NUMINAMATH_GPT_initial_volume_kola_solution_l382_38206

-- Initial composition of the kola solution
def initial_composition_sugar (V : ℝ) : ℝ := 0.20 * V

-- Final volume after additions
def final_volume (V : ℝ) : ℝ := V + 3.2 + 12 + 6.8

-- Final amount of sugar after additions
def final_amount_sugar (V : ℝ) : ℝ := initial_composition_sugar V + 3.2

-- Final percentage of sugar in the solution
def final_percentage_sugar (total_sol : ℝ) : ℝ := 0.1966850828729282 * total_sol

theorem initial_volume_kola_solution : 
  ∃ V : ℝ, final_amount_sugar V = final_percentage_sugar (final_volume V) :=
sorry

end NUMINAMATH_GPT_initial_volume_kola_solution_l382_38206


namespace NUMINAMATH_GPT_lcm_12_35_l382_38280

theorem lcm_12_35 : Nat.lcm 12 35 = 420 :=
by
  sorry

end NUMINAMATH_GPT_lcm_12_35_l382_38280


namespace NUMINAMATH_GPT_find_xyz_l382_38293

theorem find_xyz (x y z : ℝ) 
  (h1 : x * (y + z) = 180) 
  (h2 : y * (z + x) = 192) 
  (h3 : z * (x + y) = 204) 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z) : 
  x * y * z = 168 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_find_xyz_l382_38293


namespace NUMINAMATH_GPT_alyssa_hike_total_distance_l382_38275

theorem alyssa_hike_total_distance
  (e f g h i : ℝ)
  (h1 : e + f + g = 40)
  (h2 : f + g + h = 48)
  (h3 : g + h + i = 54)
  (h4 : e + h = 30) :
  e + f + g + h + i = 118 :=
by
  sorry

end NUMINAMATH_GPT_alyssa_hike_total_distance_l382_38275


namespace NUMINAMATH_GPT_sqrt_x_minus_2_range_l382_38267

theorem sqrt_x_minus_2_range (x : ℝ) : x - 2 ≥ 0 → x ≥ 2 :=
by sorry

end NUMINAMATH_GPT_sqrt_x_minus_2_range_l382_38267


namespace NUMINAMATH_GPT_number_of_large_boxes_l382_38256

theorem number_of_large_boxes (total_boxes : ℕ) (small_weight large_weight remaining_small remaining_large : ℕ) :
  total_boxes = 62 →
  small_weight = 5 →
  large_weight = 3 →
  remaining_small = 15 →
  remaining_large = 15 →
  ∀ (small_boxes large_boxes : ℕ),
    total_boxes = small_boxes + large_boxes →
    ((large_boxes * large_weight) + (remaining_small * small_weight) = (small_boxes * small_weight) + (remaining_large * large_weight)) →
    large_boxes = 27 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_number_of_large_boxes_l382_38256


namespace NUMINAMATH_GPT_certain_number_k_l382_38227

theorem certain_number_k (x : ℕ) (k : ℕ) (h1 : x = 14) (h2 : 2^x - 2^(x-2) = k * 2^12) : k = 3 := by
  sorry

end NUMINAMATH_GPT_certain_number_k_l382_38227


namespace NUMINAMATH_GPT_octagon_area_is_six_and_m_plus_n_is_seven_l382_38260

noncomputable def area_of_octagon (side_length : ℕ) (segment_length : ℚ) : ℚ :=
  let triangle_area := 1 / 2 * side_length * segment_length
  let octagon_area := 8 * triangle_area
  octagon_area

theorem octagon_area_is_six_and_m_plus_n_is_seven :
  area_of_octagon 2 (3/4) = 6 ∧ (6 + 1 = 7) :=
by
  sorry

end NUMINAMATH_GPT_octagon_area_is_six_and_m_plus_n_is_seven_l382_38260


namespace NUMINAMATH_GPT_intersection_point_exists_circle_equation_standard_form_l382_38258

noncomputable def line1 (x y : ℝ) : Prop := 2 * x + y = 0
noncomputable def line2 (x y : ℝ) : Prop := x + y = 2
noncomputable def line3 (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0

theorem intersection_point_exists :
  ∃ (C : ℝ × ℝ), (line1 C.1 C.2 ∧ line2 C.1 C.2) ∧ C = (-2, 4) :=
sorry

theorem circle_equation_standard_form :
  ∃ (center : ℝ × ℝ) (radius : ℝ), center = (-2, 4) ∧ radius = 3 ∧
  ∀ x y : ℝ, ((x + 2) ^ 2 + (y - 4) ^ 2 = 9) :=
sorry

end NUMINAMATH_GPT_intersection_point_exists_circle_equation_standard_form_l382_38258


namespace NUMINAMATH_GPT_bills_difference_l382_38244

variable (m j : ℝ)

theorem bills_difference :
  (0.10 * m = 2) → (0.20 * j = 2) → (m - j = 10) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_bills_difference_l382_38244


namespace NUMINAMATH_GPT_symmetry_axis_of_function_l382_38272

theorem symmetry_axis_of_function {x : ℝ} :
  (∃ k : ℤ, ∃ x : ℝ, (y = 2 * (Real.cos ((x / 2) + (Real.pi / 3))) ^ 2 - 1) ∧ (x + (2 * Real.pi) / 3 = k * Real.pi)) →
    x = (Real.pi / 3) ∧ 0 = y :=
sorry

end NUMINAMATH_GPT_symmetry_axis_of_function_l382_38272


namespace NUMINAMATH_GPT_number_of_roses_ian_kept_l382_38225

-- Definitions representing the conditions
def initial_roses : ℕ := 20
def roses_to_mother : ℕ := 6
def roses_to_grandmother : ℕ := 9
def roses_to_sister : ℕ := 4

-- The theorem statement we want to prove
theorem number_of_roses_ian_kept : (initial_roses - (roses_to_mother + roses_to_grandmother + roses_to_sister) = 1) :=
by
  sorry

end NUMINAMATH_GPT_number_of_roses_ian_kept_l382_38225


namespace NUMINAMATH_GPT_period_of_function_is_2pi_over_3_l382_38202

noncomputable def period_of_f (x : ℝ) : ℝ :=
  4 * (Real.sin x)^3 - Real.sin x + 2 * (Real.sin (x / 2) - Real.cos (x / 2))^2

theorem period_of_function_is_2pi_over_3 : ∀ x, period_of_f (x + (2 * Real.pi) / 3) = period_of_f x :=
by sorry

end NUMINAMATH_GPT_period_of_function_is_2pi_over_3_l382_38202


namespace NUMINAMATH_GPT_algebra_problem_l382_38238

theorem algebra_problem 
  (x : ℝ) 
  (h : x^2 - 2 * x = 3) : 
  2 * x^2 - 4 * x + 3 = 9 := 
by 
  sorry

end NUMINAMATH_GPT_algebra_problem_l382_38238


namespace NUMINAMATH_GPT_ways_to_select_books_l382_38289

theorem ways_to_select_books (nChinese nMath nEnglish : ℕ) (h1 : nChinese = 9) (h2 : nMath = 7) (h3 : nEnglish = 5) :
  (nChinese * nMath + nChinese * nEnglish + nMath * nEnglish) = 143 :=
by
  sorry

end NUMINAMATH_GPT_ways_to_select_books_l382_38289


namespace NUMINAMATH_GPT_no_solution_l382_38213

theorem no_solution (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) : 
  ¬ (x^2 + y^2 + 41 = 2^n) :=
by sorry

end NUMINAMATH_GPT_no_solution_l382_38213


namespace NUMINAMATH_GPT_number_of_true_propositions_l382_38211

-- Definitions based on conditions
def prop1 (x : ℝ) : Prop := x^2 - x + 1 > 0
def prop2 (x : ℝ) : Prop := x^2 + x - 6 < 0 → x ≤ 2
def prop3 (x : ℝ) : Prop := (x^2 - 5*x + 6 = 0) → x = 2

-- Main theorem
theorem number_of_true_propositions : 
  (∀ x : ℝ, prop1 x) ∧ (∀ x : ℝ, prop2 x) ∧ (∃ x : ℝ, ¬ prop3 x) → 
  2 = 2 :=
by sorry

end NUMINAMATH_GPT_number_of_true_propositions_l382_38211
