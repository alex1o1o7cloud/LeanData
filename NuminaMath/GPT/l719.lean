import Mathlib

namespace NUMINAMATH_GPT_remainder_of_division_l719_71951

variable (a : ℝ) (b : ℝ)

theorem remainder_of_division : a = 28 → b = 10.02 → ∃ r : ℝ, 0 ≤ r ∧ r < b ∧ ∃ q : ℤ, a = q * b + r ∧ r = 7.96 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_remainder_of_division_l719_71951


namespace NUMINAMATH_GPT_determine_m_l719_71911

theorem determine_m (m : ℝ) : (∀ x : ℝ, (0 < x ∧ x < 2) ↔ -1/2 * x^2 + 2 * x + m * x > 0) → m = -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_m_l719_71911


namespace NUMINAMATH_GPT_length_of_other_train_l719_71972

variable (L : ℝ)

theorem length_of_other_train
    (train1_length : ℝ := 260)
    (train1_speed_kmh : ℝ := 120)
    (train2_speed_kmh : ℝ := 80)
    (time_to_cross : ℝ := 9)
    (train1_speed : ℝ := train1_speed_kmh * 1000 / 3600)
    (train2_speed : ℝ := train2_speed_kmh * 1000 / 3600)
    (relative_speed : ℝ := train1_speed + train2_speed)
    (total_distance : ℝ := relative_speed * time_to_cross)
    (other_train_length : ℝ := total_distance - train1_length) :
    L = other_train_length := by
  sorry

end NUMINAMATH_GPT_length_of_other_train_l719_71972


namespace NUMINAMATH_GPT_hexagon_side_lengths_l719_71976

open Nat

/-- Define two sides AB and BC of a hexagon with their given lengths -/
structure Hexagon :=
  (AB BC AD BE CF DE: ℕ)
  (distinct_lengths : AB ≠ BC ∧ (AB = 7 ∧ BC = 8))
  (total_perimeter : AB + BC + AD + BE + CF + DE = 46)

-- Define a theorem to prove the number of sides measuring 8 units
theorem hexagon_side_lengths (h: Hexagon) :
  ∃ (n : ℕ), n = 4 ∧ n * 8 + (6 - n) * 7 = 46 :=
by
  -- Assume the proof here
  sorry

end NUMINAMATH_GPT_hexagon_side_lengths_l719_71976


namespace NUMINAMATH_GPT_fraction_equals_decimal_l719_71902

theorem fraction_equals_decimal : (1 / 4 : ℝ) = 0.25 := 
sorry

end NUMINAMATH_GPT_fraction_equals_decimal_l719_71902


namespace NUMINAMATH_GPT_matrix_addition_correct_l719_71961

def matrixA : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then 4 else -2
  else
    if j = 0 then -3 else 5

def matrixB : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then -6 else 0
  else
    if j = 0 then 7 else -8

def resultMatrix : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then -2 else -2
  else
    if j = 0 then 4 else -3

theorem matrix_addition_correct :
  matrixA + matrixB = resultMatrix :=
by
  sorry

end NUMINAMATH_GPT_matrix_addition_correct_l719_71961


namespace NUMINAMATH_GPT_problem_statement_l719_71924

namespace ProofProblem

variable (t : ℚ) (y : ℚ)

/-- Given equations and condition, we want to prove y = 21 / 2 -/
theorem problem_statement (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 6) (h3 : x = 0) : y = 21 / 2 :=
by sorry

end ProofProblem

end NUMINAMATH_GPT_problem_statement_l719_71924


namespace NUMINAMATH_GPT_max_value_of_expression_l719_71913

open Classical
open Real

theorem max_value_of_expression (a b : ℝ) (c : ℝ) (h1 : a^2 + b^2 = c^2 + ab) (h2 : c = 1) :
  ∃ x : ℝ, x = (1 / 2) * b + a ∧ x = (sqrt 21) / 3 := 
sorry

end NUMINAMATH_GPT_max_value_of_expression_l719_71913


namespace NUMINAMATH_GPT_volume_pyramid_problem_l719_71928

noncomputable def volume_of_pyramid : ℝ :=
  1 / 3 * 10 * 1.5

theorem volume_pyramid_problem :
  ∀ (AB BC CG : ℝ)
  (M : ℝ × ℝ × ℝ),
  AB = 4 →
  BC = 2 →
  CG = 3 →
  M = (2, 5, 1.5) →
  volume_of_pyramid = 5 := 
by
  intros AB BC CG M hAB hBC hCG hM
  sorry

end NUMINAMATH_GPT_volume_pyramid_problem_l719_71928


namespace NUMINAMATH_GPT_bus_trip_distance_l719_71986

theorem bus_trip_distance
  (D S : ℕ) (H1 : S = 55)
  (H2 : D / S - 1 = D / (S + 5))
  : D = 660 :=
sorry

end NUMINAMATH_GPT_bus_trip_distance_l719_71986


namespace NUMINAMATH_GPT_find_number_l719_71909

-- Define the conditions: 0.80 * x - 20 = 60
variables (x : ℝ)
axiom condition : 0.80 * x - 20 = 60

-- State the theorem that x = 100 given the condition
theorem find_number : x = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l719_71909


namespace NUMINAMATH_GPT_opposite_numbers_power_l719_71910

theorem opposite_numbers_power (a b : ℝ) (h : a + b = 0) : (a + b) ^ 2023 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_opposite_numbers_power_l719_71910


namespace NUMINAMATH_GPT_find_x_l719_71989

theorem find_x (x : ℝ) : (x + 3 * x + 1000 + 3000) / 4 = 2018 → x = 1018 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l719_71989


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l719_71965

noncomputable def function_y (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem monotonic_decreasing_interval : 
  ∀ x, -1 < x ∧ x < 3 →  (deriv function_y x < 0) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l719_71965


namespace NUMINAMATH_GPT_intersection_points_count_l719_71977

theorem intersection_points_count:
  let line1 := { p : ℝ × ℝ | ∃ x y : ℝ, 4 * y - 3 * x = 2 ∧ (p.1 = x ∧ p.2 = y) }
  let line2 := { p : ℝ × ℝ | ∃ x y : ℝ, x + 3 * y = 3 ∧ (p.1 = x ∧ p.2 = y) }
  let line3 := { p : ℝ × ℝ | ∃ x y : ℝ, 6 * x - 8 * y = 6 ∧ (p.1 = x ∧ p.2 = y) }
  ∃! p1 p2 : ℝ × ℝ, p1 ∈ line1 ∧ p1 ∈ line2 ∧ p2 ∈ line2 ∧ p2 ∈ line3 :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_count_l719_71977


namespace NUMINAMATH_GPT_wheel_revolutions_l719_71933

theorem wheel_revolutions (x y : ℕ) (h1 : y = x + 300)
  (h2 : 10 / (x : ℝ) = 10 / (y : ℝ) + 1 / 60) : 
  x = 300 ∧ y = 600 := 
by sorry

end NUMINAMATH_GPT_wheel_revolutions_l719_71933


namespace NUMINAMATH_GPT_evaluate_expression_at_2_l719_71903

theorem evaluate_expression_at_2 : ∀ (x : ℕ), x = 2 → (x^x)^(x^(x^x)) = 4294967296 := by
  intros x h
  rw [h]
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_2_l719_71903


namespace NUMINAMATH_GPT_find_fifth_integer_l719_71919

theorem find_fifth_integer (x y : ℤ) (h_pos : x > 0)
  (h_mean_median : (x + 2 + x + 7 + x + y) / 5 = x + 7) :
  y = 22 :=
sorry

end NUMINAMATH_GPT_find_fifth_integer_l719_71919


namespace NUMINAMATH_GPT_balloons_remaining_l719_71922

def bags_round : ℕ := 5
def balloons_per_bag_round : ℕ := 20
def bags_long : ℕ := 4
def balloons_per_bag_long : ℕ := 30
def balloons_burst : ℕ := 5

def total_round_balloons : ℕ := bags_round * balloons_per_bag_round
def total_long_balloons : ℕ := bags_long * balloons_per_bag_long
def total_balloons : ℕ := total_round_balloons + total_long_balloons
def balloons_left : ℕ := total_balloons - balloons_burst

theorem balloons_remaining : balloons_left = 215 := by 
  -- We leave this as sorry since the proof is not required
  sorry

end NUMINAMATH_GPT_balloons_remaining_l719_71922


namespace NUMINAMATH_GPT_simplify_trig_identity_l719_71956

theorem simplify_trig_identity (α β : ℝ) : 
  (Real.cos (α + β) * Real.cos β + Real.sin (α + β) * Real.sin β) = Real.cos α :=
by
  sorry

end NUMINAMATH_GPT_simplify_trig_identity_l719_71956


namespace NUMINAMATH_GPT_square_side_length_l719_71968

theorem square_side_length (A : ℝ) (side : ℝ) (h₁ : A = side^2) (h₂ : A = 12) : side = 2 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_square_side_length_l719_71968


namespace NUMINAMATH_GPT_option_C_correct_l719_71994

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Definitions for parallel and perpendicular relationships
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def line_parallel (l₁ l₂ : Line) : Prop := sorry

-- Theorem statement based on problem c) translation
theorem option_C_correct (H1 : line_parallel m n) (H2 : perpendicular m α) : perpendicular n α :=
sorry

end NUMINAMATH_GPT_option_C_correct_l719_71994


namespace NUMINAMATH_GPT_max_k_value_l719_71997

open Real

theorem max_k_value (x y k : ℝ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_pos_k : 0 < k)
  (h_eq : 6 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_max_k_value_l719_71997


namespace NUMINAMATH_GPT_total_sand_weight_is_34_l719_71990

-- Define the conditions
def eden_buckets : ℕ := 4
def mary_buckets : ℕ := eden_buckets + 3
def iris_buckets : ℕ := mary_buckets - 1
def weight_per_bucket : ℕ := 2

-- Define the total weight calculation
def total_buckets : ℕ := eden_buckets + mary_buckets + iris_buckets
def total_weight : ℕ := total_buckets * weight_per_bucket

-- The proof statement
theorem total_sand_weight_is_34 : total_weight = 34 := by
  sorry

end NUMINAMATH_GPT_total_sand_weight_is_34_l719_71990


namespace NUMINAMATH_GPT_son_time_to_complete_job_l719_71982

theorem son_time_to_complete_job (M S : ℝ) (hM : M = 1 / 5) (hMS : M + S = 1 / 4) : S = 1 / 20 → 1 / S = 20 :=
by
  sorry

end NUMINAMATH_GPT_son_time_to_complete_job_l719_71982


namespace NUMINAMATH_GPT_length_of_PQ_l719_71996

theorem length_of_PQ (R P Q : ℝ × ℝ) (hR : R = (10, 8))
(hP_line1 : ∃ p : ℝ, P = (p, 24 * p / 7))
(hQ_line2 : ∃ q : ℝ, Q = (q, 5 * q / 13))
(h_mid : ∃ (p q : ℝ), R = ((p + q) / 2, (24 * p / 14 + 5 * q / 26) / 2))
(answer_eq : ∃ (a b : ℕ), PQ_length = a / b ∧ a.gcd b = 1 ∧ a + b = 4925) : 
∃ a b : ℕ, a + b = 4925 := sorry

end NUMINAMATH_GPT_length_of_PQ_l719_71996


namespace NUMINAMATH_GPT_product_of_fractions_l719_71918

theorem product_of_fractions : (2 : ℚ) / 9 * (4 : ℚ) / 5 = 8 / 45 :=
by 
  sorry

end NUMINAMATH_GPT_product_of_fractions_l719_71918


namespace NUMINAMATH_GPT_red_black_probability_l719_71934

-- Define the number of cards and ranks
def num_cards : ℕ := 64
def num_ranks : ℕ := 16

-- Define the suits and their properties
def suits := 6
def red_suits := 3
def black_suits := 3
def cards_per_suit := num_ranks

-- Define the number of red and black cards
def red_cards := red_suits * cards_per_suit
def black_cards := black_suits * cards_per_suit

-- Prove the probability that the top card is red and the second card is black
theorem red_black_probability : 
  (red_cards * black_cards) / (num_cards * (num_cards - 1)) = 3 / 4 := by 
  sorry

end NUMINAMATH_GPT_red_black_probability_l719_71934


namespace NUMINAMATH_GPT_simplify_and_evaluate_l719_71944

variable (x y : ℤ)

theorem simplify_and_evaluate (h1 : x = 1) (h2 : y = 1) :
    2 * (x - 2 * y) ^ 2 - (2 * y + x) * (-2 * y + x) = 5 := by
    sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l719_71944


namespace NUMINAMATH_GPT_sum_ac_equals_seven_l719_71957

theorem sum_ac_equals_seven 
  (a b c d : ℝ)
  (h1 : ab + bc + cd + da = 42)
  (h2 : b + d = 6) :
  a + c = 7 := 
sorry

end NUMINAMATH_GPT_sum_ac_equals_seven_l719_71957


namespace NUMINAMATH_GPT_min_rice_weight_l719_71937

theorem min_rice_weight (o r : ℝ) (h1 : o ≥ 4 + 2 * r) (h2 : o ≤ 3 * r) : r ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_rice_weight_l719_71937


namespace NUMINAMATH_GPT_john_needs_20_nails_l719_71920

-- Define the given conditions
def large_planks (n : ℕ) := n = 12
def small_planks (n : ℕ) := n = 10
def nails_for_large_planks (n : ℕ) := n = 15
def nails_for_small_planks (n : ℕ) := n = 5

-- Define the total number of nails needed
def total_nails_needed (n : ℕ) :=
  ∃ (lp sp np_large np_small : ℕ),
  large_planks lp ∧ small_planks sp ∧ nails_for_large_planks np_large ∧ nails_for_small_planks np_small ∧ n = np_large + np_small

-- The theorem statement
theorem john_needs_20_nails : total_nails_needed 20 :=
by { sorry }

end NUMINAMATH_GPT_john_needs_20_nails_l719_71920


namespace NUMINAMATH_GPT_abs_diff_31st_terms_l719_71974

/-- Sequence C is an arithmetic sequence with a starting term 100 and a common difference 15. --/
def seqC (n : ℕ) : ℤ :=
  100 + 15 * (n - 1)

/-- Sequence D is an arithmetic sequence with a starting term 100 and a common difference -20. --/
def seqD (n : ℕ) : ℤ :=
  100 - 20 * (n - 1)

/-- Absolute value of the difference between the 31st terms of sequences C and D is 1050. --/
theorem abs_diff_31st_terms : |seqC 31 - seqD 31| = 1050 := by
  sorry

end NUMINAMATH_GPT_abs_diff_31st_terms_l719_71974


namespace NUMINAMATH_GPT_johns_score_is_101_l719_71950

variable (c w s : ℕ)
variable (h1 : s = 40 + 5 * c - w)
variable (h2 : s > 100)
variable (h3 : c ≤ 40)
variable (h4 : ∀ s' > 100, s' < s → ∃ c' w', s' = 40 + 5 * c' - w')

theorem johns_score_is_101 : s = 101 := by
  sorry

end NUMINAMATH_GPT_johns_score_is_101_l719_71950


namespace NUMINAMATH_GPT_binomials_product_l719_71901

noncomputable def poly1 (x y : ℝ) : ℝ := 2 * x^2 + 3 * y - 4
noncomputable def poly2 (y : ℝ) : ℝ := y + 6

theorem binomials_product (x y : ℝ) :
  (poly1 x y) * (poly2 y) = 2 * x^2 * y + 12 * x^2 + 3 * y^2 + 14 * y - 24 :=
by sorry

end NUMINAMATH_GPT_binomials_product_l719_71901


namespace NUMINAMATH_GPT_probability_of_die_showing_1_after_5_steps_l719_71930

def prob_showing_1 (steps : ℕ) : ℚ :=
  if steps = 5 then 37 / 192 else 0

theorem probability_of_die_showing_1_after_5_steps :
  prob_showing_1 5 = 37 / 192 :=
sorry

end NUMINAMATH_GPT_probability_of_die_showing_1_after_5_steps_l719_71930


namespace NUMINAMATH_GPT_valid_integers_count_l719_71907

def count_valid_integers : ℕ :=
  let digits : List ℕ := [0, 1, 2, 3, 4, 6, 7, 8, 9]
  let first_digit_count := 7  -- from 2 to 9 excluding 5
  let second_digit_count := 8
  let third_digit_count := 7
  let fourth_digit_count := 6
  first_digit_count * second_digit_count * third_digit_count * fourth_digit_count

theorem valid_integers_count : count_valid_integers = 2352 := by
  -- intermediate step might include nice counting macros
  sorry

end NUMINAMATH_GPT_valid_integers_count_l719_71907


namespace NUMINAMATH_GPT_overtime_hours_l719_71993

theorem overtime_hours (regular_rate: ℝ) (regular_hours: ℝ) (total_payment: ℝ) (overtime_rate_multiplier: ℝ) (overtime_hours: ℝ):
  regular_rate = 3 → regular_hours = 40 → total_payment = 198 → overtime_rate_multiplier = 2 → 
  overtime_hours = (total_payment - (regular_rate * regular_hours)) / (regular_rate * overtime_rate_multiplier) →
  overtime_hours = 13 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_overtime_hours_l719_71993


namespace NUMINAMATH_GPT_maximum_n_Sn_pos_l719_71969

def arithmetic_sequence := ℕ → ℝ

noncomputable def sum_first_n_terms (a : arithmetic_sequence) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

axiom a1_eq : ∀ (a : arithmetic_sequence), (a 1) = 2 * (a 2) + (a 4)

axiom S5_eq_5 : ∀ (a : arithmetic_sequence), sum_first_n_terms a 5 = 5

theorem maximum_n_Sn_pos : ∀ (a : arithmetic_sequence), (∃ (n : ℕ), n < 6 ∧ sum_first_n_terms a n > 0) → n = 5 :=
  sorry

end NUMINAMATH_GPT_maximum_n_Sn_pos_l719_71969


namespace NUMINAMATH_GPT_speed_including_stoppages_l719_71958

-- Definitions
def speed_excluding_stoppages : ℤ := 50 -- kmph
def stoppage_time_per_hour : ℕ := 24 -- minutes

-- Theorem to prove the speed of the train including stoppages
theorem speed_including_stoppages (h1 : speed_excluding_stoppages = 50)
                                  (h2 : stoppage_time_per_hour = 24) :
  ∃ s : ℤ, s = 30 := 
sorry

end NUMINAMATH_GPT_speed_including_stoppages_l719_71958


namespace NUMINAMATH_GPT_find_98_real_coins_l719_71914

-- We will define the conditions as variables and state the goal as a theorem.

-- Variables:
variable (Coin : Type) -- Type representing coins
variable [Fintype Coin] -- 100 coins in total, therefore a Finite type
variable (number_of_coins : ℕ) (h100 : number_of_coins = 100)
variable (real : Coin → Prop) -- Predicate indicating if the coin is real
variable (lighter_fake : Coin → Prop) -- Predicate indicating if the coin is the lighter fake
variable (balance_scale : Coin → Coin → Prop) -- Balance scale result

-- Conditions:
axiom real_coins_count : ∃ R : Finset Coin, R.card = 99 ∧ (∀ c ∈ R, real c)
axiom fake_coin_exists : ∃ F : Coin, lighter_fake F ∧ ¬ real F

theorem find_98_real_coins : ∃ S : Finset Coin, S.card = 98 ∧ (∀ c ∈ S, real c) := by
  sorry

end NUMINAMATH_GPT_find_98_real_coins_l719_71914


namespace NUMINAMATH_GPT_number_at_100th_row_1000th_column_l719_71927

axiom cell_numbering_rule (i j : ℕ) : ℕ

/-- 
  The cell located at the intersection of the 100th row and the 1000th column
  on an infinitely large chessboard, sequentially numbered with specific rules,
  will receive the number 900.
-/
theorem number_at_100th_row_1000th_column : cell_numbering_rule 100 1000 = 900 :=
sorry

end NUMINAMATH_GPT_number_at_100th_row_1000th_column_l719_71927


namespace NUMINAMATH_GPT_number_of_satisfying_ns_l719_71959

noncomputable def a_n (n : ℕ) : ℕ := (n-1)*(2*n-1)

def b_n (n : ℕ) : ℕ := 2^n * n

def condition (n : ℕ) : Prop := b_n n ≤ 2019 * a_n n

theorem number_of_satisfying_ns : 
  ∃ n : ℕ, n = 14 ∧ ∀ k : ℕ, (1 ≤ k ∧ k ≤ 14) → condition k := 
by
  sorry

end NUMINAMATH_GPT_number_of_satisfying_ns_l719_71959


namespace NUMINAMATH_GPT_james_training_hours_in_a_year_l719_71941

-- Definitions based on conditions
def trains_twice_a_day : ℕ := 2
def hours_per_training : ℕ := 4
def days_trains_per_week : ℕ := 7 - 2
def weeks_per_year : ℕ := 52

-- Resultant computation
def daily_training_hours : ℕ := trains_twice_a_day * hours_per_training
def weekly_training_hours : ℕ := daily_training_hours * days_trains_per_week
def yearly_training_hours : ℕ := weekly_training_hours * weeks_per_year

-- Statement to prove
theorem james_training_hours_in_a_year : yearly_training_hours = 2080 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_james_training_hours_in_a_year_l719_71941


namespace NUMINAMATH_GPT_digit_A_value_l719_71960

theorem digit_A_value :
  ∃ (A : ℕ), A < 10 ∧ (45 % A = 0) ∧ (172 * 10 + A * 10 + 6) % 8 = 0 ∧
    ∀ (B : ℕ), B < 10 ∧ (45 % B = 0) ∧ (172 * 10 + B * 10 + 6) % 8 = 0 → B = A := sorry

end NUMINAMATH_GPT_digit_A_value_l719_71960


namespace NUMINAMATH_GPT_Margarita_vs_Ricciana_l719_71948

-- Definitions based on the conditions.
def Ricciana_run : ℕ := 20
def Ricciana_jump : ℕ := 4
def Ricciana_total : ℕ := Ricciana_run + Ricciana_jump

def Margarita_run : ℕ := 18
def Margarita_jump : ℕ := 2 * Ricciana_jump - 1
def Margarita_total : ℕ := Margarita_run + Margarita_jump

-- The statement to be proved.
theorem Margarita_vs_Ricciana : (Margarita_total - Ricciana_total = 1) :=
by
  sorry

end NUMINAMATH_GPT_Margarita_vs_Ricciana_l719_71948


namespace NUMINAMATH_GPT_common_point_exists_l719_71964

theorem common_point_exists (a b c : ℝ) :
  ∃ x y : ℝ, y = a * x ^ 2 - b * x + c ∧ y = b * x ^ 2 - c * x + a ∧ y = c * x ^ 2 - a * x + b :=
  sorry

end NUMINAMATH_GPT_common_point_exists_l719_71964


namespace NUMINAMATH_GPT_complete_the_square_l719_71904

-- Definition of the initial condition
def eq1 : Prop := ∀ x : ℝ, x^2 + 4 * x + 1 = 0

-- The goal is to prove if the initial condition holds, then the desired result holds.
theorem complete_the_square (x : ℝ) (h : x^2 + 4 * x + 1 = 0) : (x + 2)^2 = 3 := by
  sorry

end NUMINAMATH_GPT_complete_the_square_l719_71904


namespace NUMINAMATH_GPT_determinant_difference_l719_71980

namespace MatrixDeterminantProblem

open Matrix

variables {R : Type*} [CommRing R]

theorem determinant_difference (a b c d : R) 
  (h : det ![![a, b], ![c, d]] = 15) :
  det ![![3 * a, 3 * b], ![3 * c, 3 * d]] - 
  det ![![3 * b, 3 * a], ![3 * d, 3 * c]] = 270 := 
by
  sorry

end MatrixDeterminantProblem

end NUMINAMATH_GPT_determinant_difference_l719_71980


namespace NUMINAMATH_GPT_more_stable_shooting_performance_l719_71988

theorem more_stable_shooting_performance :
  ∀ (SA2 SB2 : ℝ), SA2 = 1.9 → SB2 = 3 → (SA2 < SB2) → "A" = "Athlete with more stable shooting performance" :=
by
  intros SA2 SB2 h1 h2 h3
  sorry

end NUMINAMATH_GPT_more_stable_shooting_performance_l719_71988


namespace NUMINAMATH_GPT_storage_space_remaining_l719_71939

def total_space_remaining (first_floor second_floor: ℕ) (boxes: ℕ) : ℕ :=
  first_floor + second_floor - boxes

theorem storage_space_remaining :
  ∀ (first_floor second_floor boxes: ℕ),
  (first_floor = 2 * second_floor) →
  (boxes = 5000) →
  (boxes = second_floor / 4) →
  total_space_remaining first_floor second_floor boxes = 55000 :=
by
  intros first_floor second_floor boxes h1 h2 h3
  sorry

end NUMINAMATH_GPT_storage_space_remaining_l719_71939


namespace NUMINAMATH_GPT_inequality_solution_l719_71905

theorem inequality_solution (x : ℝ) (h₁ : 1 - x < 0) (h₂ : x - 3 ≤ 0) : 1 < x ∧ x ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l719_71905


namespace NUMINAMATH_GPT_solve_quadratic_l719_71917

theorem solve_quadratic (x : ℝ) : (x - 2) * (x + 3) = 0 → (x = 2 ∨ x = -3) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l719_71917


namespace NUMINAMATH_GPT_chess_tournament_points_l719_71984

theorem chess_tournament_points (boys girls : ℕ) (total_points : ℝ) 
  (total_matches : ℕ)
  (matches_among_boys points_among_boys : ℕ)
  (matches_among_girls points_among_girls : ℕ)
  (matches_between points_between : ℕ)
  (total_players : ℕ := boys + girls)
  (H1 : boys = 9) (H2 : girls = 3) (H3 : total_players = 12)
  (H4 : total_matches = total_players * (total_players - 1) / 2) 
  (H5 : total_points = total_matches) 
  (H6 : matches_among_boys = boys * (boys - 1) / 2) 
  (H7 : points_among_boys = matches_among_boys)
  (H8 : matches_among_girls = girls * (girls - 1) / 2) 
  (H9 : points_among_girls = matches_among_girls) 
  (H10 : matches_between = boys * girls) 
  (H11 : points_between = matches_between) :
  ¬ ∃ (P_B P_G : ℝ) (x : ℝ),
    P_B = points_among_boys + x ∧
    P_G = points_among_girls + (points_between - x) ∧
    P_B = P_G := by
  sorry

end NUMINAMATH_GPT_chess_tournament_points_l719_71984


namespace NUMINAMATH_GPT_original_price_l719_71979

theorem original_price (P : ℝ) (h₁ : 0.30 * P = 46) : P = 153.33 :=
  sorry

end NUMINAMATH_GPT_original_price_l719_71979


namespace NUMINAMATH_GPT_domain_of_f_l719_71947

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : 
  {x : ℝ | ¬ ((x - 3) + (x - 9) = 0)} = 
  {x : ℝ | x ≠ 6} := 
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l719_71947


namespace NUMINAMATH_GPT_hares_cuts_l719_71936

-- Definitions representing the given conditions
def intermediates_fallen := 10
def end_pieces_fixed := 2
def total_logs := intermediates_fallen + end_pieces_fixed

-- Theorem statement
theorem hares_cuts : total_logs - 1 = 11 := by 
  sorry

end NUMINAMATH_GPT_hares_cuts_l719_71936


namespace NUMINAMATH_GPT_find_degree_of_alpha_l719_71932

theorem find_degree_of_alpha
  (x : ℝ)
  (alpha : ℝ := x + 40)
  (beta : ℝ := 3 * x - 40)
  (h_parallel : alpha + beta = 180) :
  alpha = 85 :=
by
  sorry

end NUMINAMATH_GPT_find_degree_of_alpha_l719_71932


namespace NUMINAMATH_GPT_value_of_expression_l719_71943

theorem value_of_expression (x y : ℝ) (h1 : x = 1 / 2) (h2 : y = 2) : (1 / 3) * x ^ 8 * y ^ 9 = 2 / 3 :=
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_value_of_expression_l719_71943


namespace NUMINAMATH_GPT_johns_total_profit_l719_71935

theorem johns_total_profit
  (cost_price : ℝ) (selling_price : ℝ) (bags_sold : ℕ)
  (h_cost : cost_price = 4) (h_sell : selling_price = 8) (h_bags : bags_sold = 30) :
  (selling_price - cost_price) * bags_sold = 120 := by
    sorry

end NUMINAMATH_GPT_johns_total_profit_l719_71935


namespace NUMINAMATH_GPT_value_of_b_l719_71954

theorem value_of_b (b : ℝ) :
  (∀ x : ℝ, (-x^2 + b * x - 7 < 0) ↔ (x < 2 ∨ x > 6)) → b = 8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_l719_71954


namespace NUMINAMATH_GPT_option_B_is_not_polynomial_l719_71940

-- Define what constitutes a polynomial
def is_polynomial (expr : String) : Prop :=
  match expr with
  | "-26m" => True
  | "3m+5n" => True
  | "0" => True
  | _ => False

-- Given expressions
def expr_A := "-26m"
def expr_B := "m-n=1"
def expr_C := "3m+5n"
def expr_D := "0"

-- The Lean statement confirming option B is not a polynomial
theorem option_B_is_not_polynomial : ¬is_polynomial expr_B :=
by
  -- Since this statement requires a proof, we use 'sorry' as a placeholder.
  sorry

end NUMINAMATH_GPT_option_B_is_not_polynomial_l719_71940


namespace NUMINAMATH_GPT_constant_term_in_expansion_l719_71970

theorem constant_term_in_expansion :
  let f := (x - (2 / x^2))
  let expansion := f^9
  ∃ c: ℤ, expansion = c ∧ c = -672 :=
sorry

end NUMINAMATH_GPT_constant_term_in_expansion_l719_71970


namespace NUMINAMATH_GPT_truck_left_1_hour_later_l719_71921

theorem truck_left_1_hour_later (v_car v_truck : ℝ) (time_to_pass : ℝ) : 
  v_car = 55 ∧ v_truck = 65 ∧ time_to_pass = 6.5 → 
  1 = time_to_pass - (time_to_pass * (v_car / v_truck)) := 
by
  intros h
  sorry

end NUMINAMATH_GPT_truck_left_1_hour_later_l719_71921


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l719_71900

def is_hyperbola (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, a * x^2 + b * y^2 = c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a * b < 0

theorem necessary_but_not_sufficient (a b c : ℝ) (p : a * b < 0) (q : is_hyperbola a b c) :
  (∀ (a b c : ℝ), is_hyperbola a b c → a * b < 0) ∧ (¬ ∀ (a b c : ℝ), a * b < 0 → is_hyperbola a b c) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l719_71900


namespace NUMINAMATH_GPT_sum_of_nine_l719_71971

theorem sum_of_nine (S : ℕ → ℕ) (a : ℕ → ℕ) (h₀ : ∀ (n : ℕ), S n = n * (a 1 + a n) / 2)
(h₁ : S 3 = 30) (h₂ : S 6 = 100) : S 9 = 240 := 
sorry

end NUMINAMATH_GPT_sum_of_nine_l719_71971


namespace NUMINAMATH_GPT_sequence_length_l719_71962

theorem sequence_length (a d n : ℕ) (h1 : a = 3) (h2 : d = 5) (h3: 3 + (n-1) * d = 3008) : n = 602 := 
by
  sorry

end NUMINAMATH_GPT_sequence_length_l719_71962


namespace NUMINAMATH_GPT_expectation_S_tau_eq_varliminf_ratio_S_tau_l719_71985

noncomputable def xi : ℕ → ℝ := sorry
noncomputable def tau : ℝ := sorry

-- Statement (a)
theorem expectation_S_tau_eq (ES_tau : ℝ := sorry) (E_tau : ℝ := sorry) (E_xi1 : ℝ := sorry) :
  ES_tau = E_tau * E_xi1 := sorry

-- Statement (b)
theorem varliminf_ratio_S_tau (liminf_val : ℝ := sorry) (E_tau : ℝ := sorry) :
  (liminf_val = E_tau) := sorry

end NUMINAMATH_GPT_expectation_S_tau_eq_varliminf_ratio_S_tau_l719_71985


namespace NUMINAMATH_GPT_units_digit_of_product_l719_71923

theorem units_digit_of_product : 
  (4 * 6 * 9) % 10 = 6 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_of_product_l719_71923


namespace NUMINAMATH_GPT_negation_statement_l719_71946

variable {α : Type} (teacher generous : α → Prop)

theorem negation_statement :
  ¬ ∀ x, teacher x → generous x ↔ ∃ x, teacher x ∧ ¬ generous x := by
sorry

end NUMINAMATH_GPT_negation_statement_l719_71946


namespace NUMINAMATH_GPT_largest_integer_l719_71908

theorem largest_integer (n : ℕ) : n ^ 200 < 5 ^ 300 → n <= 11 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_l719_71908


namespace NUMINAMATH_GPT_exists_x_inequality_l719_71995

theorem exists_x_inequality (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * a * x + 9 < 0) ↔ a < -2 ∨ a > 2 :=
by
  sorry

end NUMINAMATH_GPT_exists_x_inequality_l719_71995


namespace NUMINAMATH_GPT_cone_height_correct_l719_71967

noncomputable def height_of_cone (R1 R2 R3 base_radius : ℝ) : ℝ :=
  if R1 = 20 ∧ R2 = 40 ∧ R3 = 40 ∧ base_radius = 21 then 28 else 0

theorem cone_height_correct :
  height_of_cone 20 40 40 21 = 28 :=
by sorry

end NUMINAMATH_GPT_cone_height_correct_l719_71967


namespace NUMINAMATH_GPT_ratio_january_february_l719_71953

variable (F : ℕ)

def total_savings := 19 + F + 8 

theorem ratio_january_february (h : total_savings F = 46) : 19 / F = 1 := by
  sorry

end NUMINAMATH_GPT_ratio_january_february_l719_71953


namespace NUMINAMATH_GPT_find_tangent_point_l719_71929

noncomputable def exp_neg (x : ℝ) : ℝ := Real.exp (-x)

theorem find_tangent_point :
  ∃ P : ℝ × ℝ, P = (-Real.log 2, 2) ∧ P.snd = exp_neg P.fst ∧ deriv exp_neg P.fst = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_tangent_point_l719_71929


namespace NUMINAMATH_GPT_probability_of_boys_and_girls_l719_71925

def total_outcomes := Nat.choose 7 4
def only_boys_outcomes := Nat.choose 4 4
def both_boys_and_girls_outcomes := total_outcomes - only_boys_outcomes
def probability := both_boys_and_girls_outcomes / total_outcomes

theorem probability_of_boys_and_girls :
  probability = 34 / 35 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_boys_and_girls_l719_71925


namespace NUMINAMATH_GPT_minimum_odd_correct_answers_l719_71945

theorem minimum_odd_correct_answers (students : Fin 50 → Fin 5) :
  (∀ S : Finset (Fin 50), S.card = 40 → 
    (∃ x ∈ S, students x = 3) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, students x₁ = 2 ∧ x₁ ≠ x₂ ∧ students x₂ = 2) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, ∃ x₃ ∈ S, students x₁ = 1 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ students x₂ = 1 ∧ students x₃ = 1) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, ∃ x₃ ∈ S, ∃ x₄ ∈ S, students x₁ = 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₄ ∧ students x₂ = 0 ∧ students x₃ = 0 ∧ students x₄ = 0)) →
  (∃ S : Finset (Fin 50), (∀ x ∈ S, (students x = 1 ∨ students x = 3)) ∧ S.card = 23) :=
by
  sorry

end NUMINAMATH_GPT_minimum_odd_correct_answers_l719_71945


namespace NUMINAMATH_GPT_find_a_l719_71991

-- Definitions given in the conditions
def f (x : ℝ) : ℝ := x^2 - 2
def g (x : ℝ) : ℝ := x^2 + 6

-- The main theorem to show
theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : f (g a) = 18) : a = Real.sqrt 14 := sorry

end NUMINAMATH_GPT_find_a_l719_71991


namespace NUMINAMATH_GPT_Pradeep_marks_l719_71926

variable (T : ℕ) (P : ℕ) (F : ℕ)

def passing_marks := P * T / 100

theorem Pradeep_marks (hT : T = 925) (hP : P = 20) (hF : F = 25) :
  (passing_marks P T) - F = 160 :=
by
  sorry

end NUMINAMATH_GPT_Pradeep_marks_l719_71926


namespace NUMINAMATH_GPT_find_y_l719_71983

open Classical

theorem find_y (a b c x y : ℚ)
  (h1 : a / b = 5 / 4)
  (h2 : b / c = 3 / x)
  (h3 : a / c = y / 4) :
  y = 15 / x :=
sorry

end NUMINAMATH_GPT_find_y_l719_71983


namespace NUMINAMATH_GPT_motorcycle_speed_for_10_minute_prior_arrival_l719_71975

noncomputable def distance_from_home_to_station (x : ℝ) : Prop :=
  x / 30 + 15 / 60 = x / 18 - 15 / 60

noncomputable def speed_to_arrive_10_minutes_before_departure (x : ℝ) (v : ℝ) : Prop :=
  v = x / (1 - 10 / 60)

theorem motorcycle_speed_for_10_minute_prior_arrival :
  (∀ x : ℝ, distance_from_home_to_station x) →
  (∃ x : ℝ, 
    ∃ v : ℝ, speed_to_arrive_10_minutes_before_departure x v ∧ v = 27) :=
by 
  intro h
  exists 22.5
  exists 27
  unfold distance_from_home_to_station at h
  unfold speed_to_arrive_10_minutes_before_departure
  sorry

end NUMINAMATH_GPT_motorcycle_speed_for_10_minute_prior_arrival_l719_71975


namespace NUMINAMATH_GPT_original_square_perimeter_l719_71906

theorem original_square_perimeter (p : ℕ) (x : ℕ) 
  (h1: p = 56) 
  (h2: 28 * x = p) : 4 * (2 * (x + 4 * x)) = 40 :=
by
  sorry

end NUMINAMATH_GPT_original_square_perimeter_l719_71906


namespace NUMINAMATH_GPT_trajectory_of_Q_existence_of_M_l719_71978

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := (x + 2) ^ 2 + y ^ 2 = 81 / 16
def C2 (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 1 / 16

-- Define the conditions about circle Q
def is_tangent_to_both (Q : ℝ → ℝ → Prop) : Prop :=
  ∃ r : ℝ, (∀ x y : ℝ, Q x y → (x + 2)^2 + y^2 = (r + 9/4)^2) ∧ (∀ x y : ℝ, Q x y → (x - 2)^2 + y^2 = (r + 1/4)^2)

-- Prove the trajectory of the center of Q
theorem trajectory_of_Q (Q : ℝ → ℝ → Prop) (h : is_tangent_to_both Q) :
  ∀ x y : ℝ, Q x y ↔ (x^2 - y^2 / 3 = 1 ∧ x ≥ 1) :=
sorry

-- Prove the existence and coordinates of M
theorem existence_of_M (M : ℝ) (Q : ℝ → ℝ → Prop) (h : is_tangent_to_both Q) :
  ∃ x y : ℝ, (x, y) = (-1, 0) ∧ (∀ x0 y0 : ℝ, Q x0 y0 → ((-y0 / (x0 - 2) = 2 * (y0 / (x0 - M)) / (1 - (y0 / (x0 - M))^2)) ↔ M = -1)) :=
sorry

end NUMINAMATH_GPT_trajectory_of_Q_existence_of_M_l719_71978


namespace NUMINAMATH_GPT_sum_of_xs_l719_71931

theorem sum_of_xs (x y z : ℂ) : (x + y * z = 8) ∧ (y + x * z = 12) ∧ (z + x * y = 11) → 
    ∃ S, ∀ (xi yi zi : ℂ), (xi + yi * zi = 8) ∧ (yi + xi * zi = 12) ∧ (zi + xi * yi = 11) →
        xi + yi + zi = S :=
by
  sorry

end NUMINAMATH_GPT_sum_of_xs_l719_71931


namespace NUMINAMATH_GPT_find_a_l719_71912

def g (x : ℝ) : ℝ := 5 * x - 7

theorem find_a (a : ℝ) : g a = 0 → a = 7 / 5 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l719_71912


namespace NUMINAMATH_GPT_positive_integer_expression_l719_71981

theorem positive_integer_expression (q : ℕ) (h : q > 0) : 
  ((∃ k : ℕ, k > 0 ∧ (5 * q + 18) = k * (3 * q - 8)) ↔ q = 3 ∨ q = 4 ∨ q = 5 ∨ q = 12) := 
sorry

end NUMINAMATH_GPT_positive_integer_expression_l719_71981


namespace NUMINAMATH_GPT_find_a_l719_71949

theorem find_a (a : ℝ) : 
  (∃ l : ℝ, l = 2 * Real.sqrt 3 ∧ 
  ∃ y, y ≤ 6 ∧ 
  (∀ x, x^2 + y^2 = a^2 ∧ 
  x^2 + y^2 + a * y - 6 = 0)) → 
  a = 2 ∨ a = -2 :=
by sorry

end NUMINAMATH_GPT_find_a_l719_71949


namespace NUMINAMATH_GPT_max_omega_value_l719_71973

noncomputable def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

def center_of_symmetry (ω φ : ℝ) := 
  ∃ n : ℤ, ω * (-Real.pi / 4) + φ = n * Real.pi

def extremum_point (ω φ : ℝ) :=
  ∃ n' : ℤ, ω * (Real.pi / 4) + φ = n' * Real.pi + Real.pi / 2

def monotonic_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < b → a < y ∧ y < b → x ≤ y → f x ≤ f y

theorem max_omega_value (ω : ℝ) (φ : ℝ) : 
  (ω > 0) →
  (|φ| ≤ Real.pi / 2) →
  center_of_symmetry ω φ →
  extremum_point ω φ →
  monotonic_in_interval (f ω φ) (5 * Real.pi / 18) (2 * Real.pi / 5) →
  ω = 5 :=
by
  sorry

end NUMINAMATH_GPT_max_omega_value_l719_71973


namespace NUMINAMATH_GPT_find_initial_quarters_l719_71916

-- Define the initial number of dimes, nickels, and quarters (unknown)
def initial_dimes : ℕ := 2
def initial_nickels : ℕ := 5
def initial_quarters (Q : ℕ) := Q

-- Define the additional coins given by Linda’s mother
def additional_dimes : ℕ := 2
def additional_quarters : ℕ := 10
def additional_nickels : ℕ := 2 * initial_nickels

-- Define the total number of each type of coin after Linda receives the additional coins
def total_dimes : ℕ := initial_dimes + additional_dimes
def total_quarters (Q : ℕ) : ℕ := additional_quarters + initial_quarters Q
def total_nickels : ℕ := initial_nickels + additional_nickels

-- Define the total number of coins
def total_coins (Q : ℕ) : ℕ := total_dimes + total_quarters Q + total_nickels

theorem find_initial_quarters : ∃ Q : ℕ, total_coins Q = 35 ∧ Q = 6 := by
  -- Provide the corresponding proof here
  sorry

end NUMINAMATH_GPT_find_initial_quarters_l719_71916


namespace NUMINAMATH_GPT_positive_difference_l719_71966

theorem positive_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 60 / 7 := 
sorry

end NUMINAMATH_GPT_positive_difference_l719_71966


namespace NUMINAMATH_GPT_actual_cost_of_article_l719_71999

theorem actual_cost_of_article (x : ℝ) (h : 0.60 * x = 1050) : x = 1750 := by
  sorry

end NUMINAMATH_GPT_actual_cost_of_article_l719_71999


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l719_71952

-- Definitions of conditions
def p (x : ℝ) : Prop := 1 / (x + 1) > 0
def q (x : ℝ) : Prop := (1/x > 0)

-- Main theorem statement
theorem sufficient_but_not_necessary :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, p x ∧ ¬ q x) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l719_71952


namespace NUMINAMATH_GPT_combined_length_of_straight_parts_l719_71963

noncomputable def length_of_straight_parts (R : ℝ) (p : ℝ) : ℝ := p * R

theorem combined_length_of_straight_parts :
  ∀ (R : ℝ) (p : ℝ), R = 80 ∧ p = 0.25 → length_of_straight_parts R p = 20 :=
by
  intros R p h
  cases' h with hR hp
  rw [hR, hp]
  simp [length_of_straight_parts]
  sorry

end NUMINAMATH_GPT_combined_length_of_straight_parts_l719_71963


namespace NUMINAMATH_GPT_root_equation_l719_71998

variables (m : ℝ)

theorem root_equation {m : ℝ} (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2023 = 2026 :=
by {
  sorry 
}

end NUMINAMATH_GPT_root_equation_l719_71998


namespace NUMINAMATH_GPT_wall_width_is_correct_l719_71992

-- Definitions based on the conditions
def brick_length : ℝ := 25  -- in cm
def brick_height : ℝ := 11.25  -- in cm
def brick_width : ℝ := 6  -- in cm
def num_bricks : ℝ := 5600
def wall_length : ℝ := 700  -- 7 m in cm
def wall_height : ℝ := 600  -- 6 m in cm
def total_volume : ℝ := num_bricks * (brick_length * brick_height * brick_width)

-- Prove that the inferred width of the wall is correct
theorem wall_width_is_correct : (total_volume / (wall_length * wall_height)) = 22.5 := by
  sorry

end NUMINAMATH_GPT_wall_width_is_correct_l719_71992


namespace NUMINAMATH_GPT_smallest_y2_l719_71955

theorem smallest_y2 :
  let y1 := 1 / (-2)
  let y2 := 1 / (-1)
  let y3 := 1 / (1)
  let y4 := 1 / (2)
  y2 < y1 ∧ y2 < y3 ∧ y2 < y4 :=
by
  let y1 := 1 / (-2)
  let y2 := 1 / (-1)
  let y3 := 1 / (1)
  let y4 := 1 / (2)
  show y2 < y1 ∧ y2 < y3 ∧ y2 < y4
  sorry

end NUMINAMATH_GPT_smallest_y2_l719_71955


namespace NUMINAMATH_GPT_problem_statement_l719_71938

noncomputable def g : ℝ → ℝ
| x => if x < 0 then -x
            else if x < 5 then x + 3
            else 2 * x ^ 2

theorem problem_statement : g (-6) + g 3 + g 8 = 140 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_problem_statement_l719_71938


namespace NUMINAMATH_GPT_diff_between_largest_and_smallest_fraction_l719_71987

theorem diff_between_largest_and_smallest_fraction : 
  let f1 := (3 : ℚ) / 4
  let f2 := (7 : ℚ) / 8
  let f3 := (13 : ℚ) / 16
  let f4 := (1 : ℚ) / 2
  let largest := max f1 (max f2 (max f3 f4))
  let smallest := min f1 (min f2 (min f3 f4))
  largest - smallest = (3 : ℚ) / 8 :=
by
  sorry

end NUMINAMATH_GPT_diff_between_largest_and_smallest_fraction_l719_71987


namespace NUMINAMATH_GPT_find_a2_plus_a8_l719_71942

variable {a_n : ℕ → ℤ}  -- Assume the sequence is indexed by natural numbers and maps to integers

-- Define the condition in the problem
def seq_property (a_n : ℕ → ℤ) := a_n 3 + a_n 4 + a_n 5 + a_n 6 + a_n 7 = 25

-- Statement to prove
theorem find_a2_plus_a8 (h : seq_property a_n) : a_n 2 + a_n 8 = 10 :=
sorry

end NUMINAMATH_GPT_find_a2_plus_a8_l719_71942


namespace NUMINAMATH_GPT_average_of_remaining_two_l719_71915

-- Given conditions
def average_of_six (S : ℝ) := S / 6 = 3.95
def average_of_first_two (S1 : ℝ) := S1 / 2 = 4.2
def average_of_next_two (S2 : ℝ) := S2 / 2 = 3.85

-- Prove that the average of the remaining 2 numbers equals 3.8
theorem average_of_remaining_two (S S1 S2 Sr : ℝ) (h1 : average_of_six S) (h2 : average_of_first_two S1) (h3: average_of_next_two S2) (h4 : Sr = S - S1 - S2) :
  Sr / 2 = 3.8 :=
by
  -- We can use the assumptions h1, h2, h3, and h4 to reach the conclusion
  sorry

end NUMINAMATH_GPT_average_of_remaining_two_l719_71915
