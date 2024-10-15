import Mathlib

namespace NUMINAMATH_GPT_discount_percent_l1544_154450

theorem discount_percent (CP MP SP : ℝ) (markup profit: ℝ) (h1 : CP = 100) (h2 : MP = CP + (markup * CP))
  (h3 : SP = CP + (profit * CP)) (h4 : markup = 0.75) (h5 : profit = 0.225) : 
  (MP - SP) / MP * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_discount_percent_l1544_154450


namespace NUMINAMATH_GPT_B_subset_A_implies_range_m_l1544_154475

variable {x m : ℝ}

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | -m < x ∧ x < m}

theorem B_subset_A_implies_range_m (m : ℝ) (h : B m ⊆ A) : m ≤ 1 := by
  sorry

end NUMINAMATH_GPT_B_subset_A_implies_range_m_l1544_154475


namespace NUMINAMATH_GPT_no_integer_solution_l1544_154400

theorem no_integer_solution (a b c d : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) :
  ¬ ∃ n : ℤ, n^4 - (a : ℤ)*n^3 - (b : ℤ)*n^2 - (c : ℤ)*n - (d : ℤ) = 0 :=
sorry

end NUMINAMATH_GPT_no_integer_solution_l1544_154400


namespace NUMINAMATH_GPT_two_real_solutions_only_if_c_zero_l1544_154461

theorem two_real_solutions_only_if_c_zero (x y c : ℝ) :
  (|x + y| = 99 ∧ |x - y| = c → (∃! (x y : ℝ), |x + y| = 99 ∧ |x - y| = c)) ↔ c = 0 :=
by
  sorry

end NUMINAMATH_GPT_two_real_solutions_only_if_c_zero_l1544_154461


namespace NUMINAMATH_GPT_smallest_angle_between_lines_l1544_154487

theorem smallest_angle_between_lines (r1 r2 r3 : ℝ) (S U : ℝ) (h1 : r1 = 4) (h2 : r2 = 3) 
  (h3 : r3 = 2) (total_area : ℝ := π * (r1^2 + r2^2 + r3^2)) 
  (h4 : S = (5 / 8) * U) (h5 : S + U = total_area) : 
  ∃ θ : ℝ, θ = (5 * π) / 13 :=
by
  sorry

end NUMINAMATH_GPT_smallest_angle_between_lines_l1544_154487


namespace NUMINAMATH_GPT_base_10_representation_l1544_154479

-- Conditions
variables (C D : ℕ)
variables (hC : 0 ≤ C ∧ C ≤ 7)
variables (hD : 0 ≤ D ∧ D ≤ 5)
variables (hEq : 8 * C + D = 6 * D + C)

-- Goal
theorem base_10_representation : 8 * C + D = 0 := by
  sorry

end NUMINAMATH_GPT_base_10_representation_l1544_154479


namespace NUMINAMATH_GPT_sin_theta_correct_l1544_154427

noncomputable def sin_theta (a : ℝ) (h1 : a ≠ 0) (h2 : Real.tan θ = -a) : Real :=
  -Real.sqrt 2 / 2

theorem sin_theta_correct (a : ℝ) (h1 : a ≠ 0) (h2 : Real.tan (Real.arctan (-a)) = -a) : sin_theta a h1 h2 = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_theta_correct_l1544_154427


namespace NUMINAMATH_GPT_batsman_average_46_innings_l1544_154438

theorem batsman_average_46_innings {hs ls t_44 : ℕ} (h_diff: hs - ls = 180) (h_avg_44: t_44 = 58 * 44) (h_hiscore: hs = 194) : 
  (t_44 + hs + ls) / 46 = 60 := 
sorry

end NUMINAMATH_GPT_batsman_average_46_innings_l1544_154438


namespace NUMINAMATH_GPT_range_of_a_for_false_proposition_l1544_154444

theorem range_of_a_for_false_proposition :
  ∀ a : ℝ, (¬ ∃ x : ℝ, a * x ^ 2 + a * x + 1 ≤ 0) ↔ (0 ≤ a ∧ a < 4) :=
by sorry

end NUMINAMATH_GPT_range_of_a_for_false_proposition_l1544_154444


namespace NUMINAMATH_GPT_range_of_f_l1544_154422

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem range_of_f :
  (∀ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → 0 ≤ f x ∧ f x ≤ 3) := sorry

end NUMINAMATH_GPT_range_of_f_l1544_154422


namespace NUMINAMATH_GPT_minimum_cost_for_18_oranges_l1544_154419

noncomputable def min_cost_oranges (x y : ℕ) : ℕ :=
  10 * x + 30 * y

theorem minimum_cost_for_18_oranges :
  (∃ x y : ℕ, 3 * x + 7 * y = 18 ∧ min_cost_oranges x y = 60) ∧ (60 / 18 = 10 / 3) :=
sorry

end NUMINAMATH_GPT_minimum_cost_for_18_oranges_l1544_154419


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1544_154476

theorem necessary_but_not_sufficient_condition :
  (∀ x, x > 2 → x^2 - 3*x + 2 > 0) ∧ (∃ x, x^2 - 3*x + 2 > 0 ∧ ¬ (x > 2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1544_154476


namespace NUMINAMATH_GPT_onions_left_l1544_154446

def sallyOnions : ℕ := 5
def fredOnions : ℕ := 9
def onionsGivenToSara : ℕ := 4

theorem onions_left : (sallyOnions + fredOnions) - onionsGivenToSara = 10 := by
  sorry

end NUMINAMATH_GPT_onions_left_l1544_154446


namespace NUMINAMATH_GPT_number_of_dodge_trucks_l1544_154411

theorem number_of_dodge_trucks (V T F D : ℕ) (h1 : V = 5)
  (h2 : T = 2 * V) 
  (h3 : F = 2 * T)
  (h4 : F = D / 3) :
  D = 60 := 
by
  sorry

end NUMINAMATH_GPT_number_of_dodge_trucks_l1544_154411


namespace NUMINAMATH_GPT_triangle_area_range_l1544_154474

theorem triangle_area_range (x₁ x₂ : ℝ) (h₀ : 0 < x₁) (h₁ : x₁ < 1) (h₂ : 1 < x₂) (h₃ : x₁ * x₂ = 1) :
  0 < (2 / (x₁ + 1 / x₁)) ∧ (2 / (x₁ + 1 / x₁)) < 1 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_range_l1544_154474


namespace NUMINAMATH_GPT_john_total_distance_traveled_l1544_154483

theorem john_total_distance_traveled :
  let d1 := 45 * 2.5
  let d2 := 60 * 3.5
  let d3 := 40 * 2
  let d4 := 55 * 3
  d1 + d2 + d3 + d4 = 567.5 := by
  sorry

end NUMINAMATH_GPT_john_total_distance_traveled_l1544_154483


namespace NUMINAMATH_GPT_directrix_of_parabola_l1544_154456

theorem directrix_of_parabola (a : ℝ) (P : ℝ × ℝ)
  (h1 : 3 * P.1 ^ 2 - P.2 ^ 2 = 3 * a ^ 2)
  (h2 : P.2 ^ 2 = 8 * a * P.1)
  (h3 : a > 0)
  (h4 : abs ((P.1 - 2 * a) ^ 2 + P.2 ^ 2) ^ (1 / 2) + abs ((P.1 + 2 * a) ^ 2 + P.2 ^ 2) ^ (1 / 2) = 12) :
  (a = 1) → P.1 = 6 - 3 * a → P.2 ^ 2 = 8 * a * (6 - 3 * a) → -2 * a = -2 := 
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1544_154456


namespace NUMINAMATH_GPT_Isaiah_types_more_l1544_154432

theorem Isaiah_types_more (Micah_rate Isaiah_rate : ℕ) (h_Micah : Micah_rate = 20) (h_Isaiah : Isaiah_rate = 40) :
  (Isaiah_rate * 60 - Micah_rate * 60) = 1200 :=
by
  -- Here we assume we need to prove this theorem
  sorry

end NUMINAMATH_GPT_Isaiah_types_more_l1544_154432


namespace NUMINAMATH_GPT_coin_flip_probability_l1544_154429

open Classical

noncomputable section

theorem coin_flip_probability :
  let total_outcomes := 2^10
  let exactly_five_heads_tails := Nat.choose 10 5 / total_outcomes
  let even_heads_probability := 1/2
  (even_heads_probability * (1 - exactly_five_heads_tails) / 2 = 193 / 512) :=
by
  sorry

end NUMINAMATH_GPT_coin_flip_probability_l1544_154429


namespace NUMINAMATH_GPT_jerry_removed_figures_l1544_154448

-- Definitions based on conditions
def initialFigures : ℕ := 3
def addedFigures : ℕ := 4
def currentFigures : ℕ := 6

-- Total figures after adding
def totalFigures := initialFigures + addedFigures

-- Proof statement defining how many figures were removed
theorem jerry_removed_figures : (totalFigures - currentFigures) = 1 := by
  sorry

end NUMINAMATH_GPT_jerry_removed_figures_l1544_154448


namespace NUMINAMATH_GPT_radio_cost_price_l1544_154492

theorem radio_cost_price (SP : ℝ) (Loss : ℝ) (CP : ℝ) (h1 : SP = 1110) (h2 : Loss = 0.26) (h3 : SP = CP * (1 - Loss)) : CP = 1500 :=
  by
  sorry

end NUMINAMATH_GPT_radio_cost_price_l1544_154492


namespace NUMINAMATH_GPT_believe_more_blue_l1544_154455

-- Define the conditions
def total_people : ℕ := 150
def more_green : ℕ := 90
def both_more_green_and_more_blue : ℕ := 40
def neither : ℕ := 20

-- Theorem statement: Prove that the number of people who believe teal is "more blue" is 80
theorem believe_more_blue : 
  total_people - neither - (more_green - both_more_green_and_more_blue) = 80 :=
by
  sorry

end NUMINAMATH_GPT_believe_more_blue_l1544_154455


namespace NUMINAMATH_GPT_probability_of_black_ball_l1544_154402

theorem probability_of_black_ball 
  (p_red : ℝ)
  (p_white : ℝ)
  (h_red : p_red = 0.43)
  (h_white : p_white = 0.27)
  : (1 - p_red - p_white) = 0.3 :=
by 
  sorry

end NUMINAMATH_GPT_probability_of_black_ball_l1544_154402


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l1544_154467

open Real

noncomputable def a_4 (a1 q : ℝ) : ℝ := a1 * q^3
noncomputable def sum_five_terms (a1 q : ℝ) : ℝ := a1 * (1 - q^5) / (1 - q)

theorem arithmetic_geometric_sequence :
  ∀ (a1 q : ℝ),
    (a1 + a1 * q^2 = 10) →
    (a1 * q^3 + a1 * q^5 = 5 / 4) →
    (a_4 a1 q = 1) ∧ (sum_five_terms a1 q = 31 / 2) :=
by
  intros a1 q h1 h2
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l1544_154467


namespace NUMINAMATH_GPT_right_triangle_condition_l1544_154493

theorem right_triangle_condition (a b c : ℝ) :
  (a^3 + b^3 + c^3 = a*b*(a + b) - b*c*(b + c) + a*c*(a + c)) ↔ (a^2 = b^2 + c^2) ∨ (b^2 = a^2 + c^2) ∨ (c^2 = a^2 + b^2) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_condition_l1544_154493


namespace NUMINAMATH_GPT_inverse_proportion_quadrants_l1544_154470

theorem inverse_proportion_quadrants (k b : ℝ) (h1 : b > 0) (h2 : k < 0) :
  ∀ x : ℝ, (x > 0 → (y = kb / x) → y < 0) ∧ (x < 0 → (y = kb / x) → y > 0) :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_quadrants_l1544_154470


namespace NUMINAMATH_GPT_multiplication_counts_l1544_154424

open Polynomial

noncomputable def horner_multiplications (n : ℕ) : ℕ := n

noncomputable def direct_summation_multiplications (n : ℕ) : ℕ := n * (n + 1) / 2

theorem multiplication_counts (P : Polynomial ℝ) (x₀ : ℝ) (n : ℕ)
  (h_degree : P.degree = n) :
  horner_multiplications n = n ∧ direct_summation_multiplications n = (n * (n + 1)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_counts_l1544_154424


namespace NUMINAMATH_GPT_chocolate_eggs_weeks_l1544_154409

theorem chocolate_eggs_weeks (e: ℕ) (d: ℕ) (w: ℕ) (total: ℕ) (weeks: ℕ) 
    (initialEggs : e = 40)
    (dailyEggs : d = 2)
    (schoolDays : w = 5)
    (totalWeeks : weeks = total):
    total = e / (d * w) := by
sorry

end NUMINAMATH_GPT_chocolate_eggs_weeks_l1544_154409


namespace NUMINAMATH_GPT_speed_of_first_train_l1544_154472

theorem speed_of_first_train
  (v : ℝ)
  (d : ℝ)
  (distance_between_stations : ℝ := 450)
  (speed_of_second_train : ℝ := 25)
  (additional_distance_first_train : ℝ := 50)
  (meet_time_condition : d / v = (d - additional_distance_first_train) / speed_of_second_train)
  (total_distance_condition : d + (d - additional_distance_first_train) = distance_between_stations) :
  v = 31.25 :=
by {
  sorry
}

end NUMINAMATH_GPT_speed_of_first_train_l1544_154472


namespace NUMINAMATH_GPT_n_is_prime_l1544_154404

variable {n : ℕ}

theorem n_is_prime (hn : n > 1) (hd : ∀ d : ℕ, d > 0 ∧ d ∣ n → d + 1 ∣ n + 1) :
  Prime n := 
sorry

end NUMINAMATH_GPT_n_is_prime_l1544_154404


namespace NUMINAMATH_GPT_terminative_decimal_of_45_div_72_l1544_154490

theorem terminative_decimal_of_45_div_72 :
  (45 / 72 : ℚ) = 0.625 :=
sorry

end NUMINAMATH_GPT_terminative_decimal_of_45_div_72_l1544_154490


namespace NUMINAMATH_GPT_sin_2x_value_l1544_154401

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x)

theorem sin_2x_value (x : ℝ) (h1 : f x = 5 / 3) (h2 : -Real.pi / 6 < x) (h3 : x < Real.pi / 6) :
  Real.sin (2 * x) = (Real.sqrt 3 - 2 * Real.sqrt 2) / 6 := 
sorry

end NUMINAMATH_GPT_sin_2x_value_l1544_154401


namespace NUMINAMATH_GPT_largest_number_divisible_by_48_is_9984_l1544_154431

def largest_divisible_by_48 (n : ℕ) := ∀ m ≥ n, m % 48 = 0 → m ≤ 9999

theorem largest_number_divisible_by_48_is_9984 :
  largest_divisible_by_48 9984 ∧ 9999 / 10^3 = 9 ∧ 48 ∣ 9984 ∧ 9984 < 10000 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_divisible_by_48_is_9984_l1544_154431


namespace NUMINAMATH_GPT_diamond_evaluation_l1544_154447

def diamond (X Y : ℚ) : ℚ := (2 * X + 3 * Y) / 5

theorem diamond_evaluation : diamond (diamond 3 15) 6 = 192 / 25 := 
by
  sorry

end NUMINAMATH_GPT_diamond_evaluation_l1544_154447


namespace NUMINAMATH_GPT_inequalities_not_hold_range_a_l1544_154416

theorem inequalities_not_hold_range_a (a : ℝ) :
  (¬ ∀ x : ℝ, x^2 - a * x + 1 ≤ 0) ∧ (¬ ∀ x : ℝ, a * x^2 + x - 1 > 0) ↔ (-2 < a ∧ a ≤ -1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_inequalities_not_hold_range_a_l1544_154416


namespace NUMINAMATH_GPT_vacation_cost_l1544_154463

theorem vacation_cost (C : ℝ) (h : C / 6 - C / 8 = 120) : C = 2880 :=
by
  sorry

end NUMINAMATH_GPT_vacation_cost_l1544_154463


namespace NUMINAMATH_GPT_sequence_constant_l1544_154421

theorem sequence_constant
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : ∀ n, Nat.Prime (Int.natAbs (a n)))
  (h2 : ∀ n, a (n + 2) = a (n + 1) + a n + d) :
  ∃ c : ℤ, ∀ n, a n = c :=
by
  sorry

end NUMINAMATH_GPT_sequence_constant_l1544_154421


namespace NUMINAMATH_GPT_bird_probability_l1544_154495

def uniform_probability (segment_count bird_count : ℕ) : ℚ :=
  if bird_count = segment_count then
    1 / (segment_count ^ bird_count)
  else
    0

theorem bird_probability :
  let wire_length := 10
  let birds := 10
  let distance := 1
  let segments := wire_length / distance
  segments = birds ->
  uniform_probability segments birds = 1 / (10 ^ 10) := by
  intros
  sorry

end NUMINAMATH_GPT_bird_probability_l1544_154495


namespace NUMINAMATH_GPT_negation_of_sum_of_squares_l1544_154494

variables (a b : ℝ)

theorem negation_of_sum_of_squares:
  ¬(a^2 + b^2 = 0) → (a ≠ 0 ∨ b ≠ 0) := 
by
  sorry

end NUMINAMATH_GPT_negation_of_sum_of_squares_l1544_154494


namespace NUMINAMATH_GPT_count_perfect_cubes_between_10_and_2000_l1544_154442

theorem count_perfect_cubes_between_10_and_2000 : 
  (∃ n_min n_max, n_min^3 ≥ 10 ∧ n_max^3 ≤ 2000 ∧ 
  (n_max - n_min + 1 = 10)) := 
sorry

end NUMINAMATH_GPT_count_perfect_cubes_between_10_and_2000_l1544_154442


namespace NUMINAMATH_GPT_partial_fraction_decomposition_l1544_154459

theorem partial_fraction_decomposition :
  ∀ (A B C : ℝ), 
  (∀ x : ℝ, x^4 - 3 * x^3 - 7 * x^2 + 15 * x - 10 ≠ 0 →
    (x^2 - 23) /
    (x^4 - 3 * x^3 - 7 * x^2 + 15 * x - 10) = 
    A / (x - 1) + B / (x + 2) + C / (x - 2)) →
  (A = 44 / 21 ∧ B = -5 / 2 ∧ C = -5 / 6 → A * B * C = 275 / 63)
  := by
  intros A B C h₁ h₂
  sorry

end NUMINAMATH_GPT_partial_fraction_decomposition_l1544_154459


namespace NUMINAMATH_GPT_year_when_mother_age_is_twice_jack_age_l1544_154477

noncomputable def jack_age_2010 := 12
noncomputable def mother_age_2010 := 3 * jack_age_2010

theorem year_when_mother_age_is_twice_jack_age :
  ∃ x : ℕ, mother_age_2010 + x = 2 * (jack_age_2010 + x) ∧ (2010 + x = 2022) :=
by
  sorry

end NUMINAMATH_GPT_year_when_mother_age_is_twice_jack_age_l1544_154477


namespace NUMINAMATH_GPT_q_zero_l1544_154403

noncomputable def q (x : ℝ) : ℝ := sorry -- Definition of the polynomial q(x) is required here.

theorem q_zero : 
  (∀ n : ℕ, n ≤ 7 → q (3^n) = 1 / 3^n) →
  q 0 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_q_zero_l1544_154403


namespace NUMINAMATH_GPT_mul_binom_expansion_l1544_154468

variable (a : ℝ)

theorem mul_binom_expansion : (a + 1) * (a - 1) = a^2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_mul_binom_expansion_l1544_154468


namespace NUMINAMATH_GPT_solution_set_inequality_l1544_154497

theorem solution_set_inequality (x : ℝ) : (x^2 + x - 2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 1) := 
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1544_154497


namespace NUMINAMATH_GPT_polynomial_divisibility_l1544_154498

theorem polynomial_divisibility :
  ∃ (p : Polynomial ℤ), (Polynomial.X ^ 2 - Polynomial.X + 2) * p = Polynomial.X ^ 15 + Polynomial.X ^ 2 + 100 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l1544_154498


namespace NUMINAMATH_GPT_net_population_change_l1544_154407

theorem net_population_change (P : ℝ) : 
  let P1 := P * (6/5)
  let P2 := P1 * (7/10)
  let P3 := P2 * (6/5)
  let P4 := P3 * (7/10)
  (P4 / P - 1) * 100 = -29 := 
by
  sorry

end NUMINAMATH_GPT_net_population_change_l1544_154407


namespace NUMINAMATH_GPT_three_digit_numbers_last_three_digits_of_square_l1544_154441

theorem three_digit_numbers_last_three_digits_of_square (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (n^2 % 1000) = n ↔ n = 376 ∨ n = 625 := 
sorry

end NUMINAMATH_GPT_three_digit_numbers_last_three_digits_of_square_l1544_154441


namespace NUMINAMATH_GPT_gcd_18_24_l1544_154457

theorem gcd_18_24 : Int.gcd 18 24 = 6 :=
by
  sorry

end NUMINAMATH_GPT_gcd_18_24_l1544_154457


namespace NUMINAMATH_GPT_johns_payment_ratio_is_one_half_l1544_154464

-- Define the initial conditions
def num_members := 4
def join_fee_per_person := 4000
def monthly_cost_per_person := 1000
def johns_payment_per_year := 32000

-- Calculate total cost for joining
def total_join_fee := num_members * join_fee_per_person

-- Calculate total monthly cost for a year
def total_monthly_cost := num_members * monthly_cost_per_person * 12

-- Calculate total cost for the first year
def total_cost_for_year := total_join_fee + total_monthly_cost

-- The ratio of John's payment to the total cost
def johns_ratio := johns_payment_per_year / total_cost_for_year

-- The statement to be proved
theorem johns_payment_ratio_is_one_half : johns_ratio = (1 / 2) := by sorry

end NUMINAMATH_GPT_johns_payment_ratio_is_one_half_l1544_154464


namespace NUMINAMATH_GPT_community_service_arrangements_l1544_154449

noncomputable def total_arrangements : ℕ :=
  let case1 := Nat.choose 6 3
  let case2 := 2 * Nat.choose 6 2
  let case3 := case2
  case1 + case2 + case3

theorem community_service_arrangements :
  total_arrangements = 80 :=
by
  sorry

end NUMINAMATH_GPT_community_service_arrangements_l1544_154449


namespace NUMINAMATH_GPT_grasshopper_opposite_corner_moves_l1544_154417

noncomputable def grasshopper_jump_count : ℕ :=
  Nat.factorial 27 / (Nat.factorial 9 * Nat.factorial 9 * Nat.factorial 9)

theorem grasshopper_opposite_corner_moves :
  grasshopper_jump_count = Nat.factorial 27 / (Nat.factorial 9 * Nat.factorial 9 * Nat.factorial 9) :=
by
  -- The detailed proof would go here.
  sorry

end NUMINAMATH_GPT_grasshopper_opposite_corner_moves_l1544_154417


namespace NUMINAMATH_GPT_jogging_friends_probability_l1544_154436

theorem jogging_friends_probability
  (n p q r : ℝ)
  (h₀ : 1 > 0) -- Positive integers condition
  (h₁ : n = p - q * Real.sqrt r)
  (h₂ : ∀ prime, ¬ (r ∣ prime ^ 2)) -- r is not divisible by the square of any prime
  (h₃ : (60 - n)^2 = 1800) -- Derived from 50% meeting probability
  (h₄ : p = 60) -- Identified values from solution
  (h₅ : q = 30)
  (h₆ : r = 2) : 
  p + q + r = 92 :=
by
  sorry

end NUMINAMATH_GPT_jogging_friends_probability_l1544_154436


namespace NUMINAMATH_GPT_circle_equation_of_diameter_l1544_154434

theorem circle_equation_of_diameter (A B : ℝ × ℝ) (hA : A = (-4, -5)) (hB : B = (6, -1)) :
  ∃ h k r : ℝ, (x - h)^2 + (y - k)^2 = r ∧ h = 1 ∧ k = -3 ∧ r = 29 := 
by
  sorry

end NUMINAMATH_GPT_circle_equation_of_diameter_l1544_154434


namespace NUMINAMATH_GPT_find_k_value_l1544_154406

variable {a : ℕ → ℕ} {S : ℕ → ℕ} 

axiom sum_of_first_n_terms (n : ℕ) (hn : n > 0) : S n = a n / n
axiom exists_Sk_inequality (k : ℕ) (hk : k > 0) : 1 < S k ∧ S k < 9

theorem find_k_value 
  (k : ℕ) (hk : k > 0) (hS : S k = a k / k) (hSk : 1 < S k ∧ S k < 9)
  (h_cond : ∀ n > 0, S n = n * S n ∧ S (n - 1) = S n * (n - 1)) : 
  k = 4 :=
sorry

end NUMINAMATH_GPT_find_k_value_l1544_154406


namespace NUMINAMATH_GPT_quadratic_solution_range_l1544_154405

theorem quadratic_solution_range (t : ℝ) :
  (∃ x : ℝ, x^2 - 2 * x - t = 0 ∧ -1 < x ∧ x < 4) ↔ (-1 ≤ t ∧ t < 8) := 
sorry

end NUMINAMATH_GPT_quadratic_solution_range_l1544_154405


namespace NUMINAMATH_GPT_cashier_overestimation_l1544_154480

def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def half_dollar_value := 50

def nickels_counted_as_dimes := 15
def quarters_counted_as_half_dollars := 10

noncomputable def overestimation_due_to_nickels_as_dimes : Nat := 
  (dime_value - nickel_value) * nickels_counted_as_dimes

noncomputable def overestimation_due_to_quarters_as_half_dollars : Nat := 
  (half_dollar_value - quarter_value) * quarters_counted_as_half_dollars

noncomputable def total_overestimation : Nat := 
  overestimation_due_to_nickels_as_dimes + overestimation_due_to_quarters_as_half_dollars

theorem cashier_overestimation : total_overestimation = 325 := by
  sorry

end NUMINAMATH_GPT_cashier_overestimation_l1544_154480


namespace NUMINAMATH_GPT_range_of_m_l1544_154445

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0)
  (h_equation : (2 / x) + (1 / y) = 1 / 3)
  (h_inequality : x + 2 * y > m^2 - 2 * m) : 
  -4 < m ∧ m < 6 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1544_154445


namespace NUMINAMATH_GPT_option_A_equal_l1544_154430

theorem option_A_equal : (-2: ℤ)^(3: ℕ) = ((-2: ℤ)^(3: ℕ)) :=
by
  sorry

end NUMINAMATH_GPT_option_A_equal_l1544_154430


namespace NUMINAMATH_GPT_cost_of_art_book_l1544_154482

theorem cost_of_art_book
  (total_cost m_c s_c : ℕ)
  (m_b s_b a_b : ℕ)
  (hm : m_c = 3)
  (hs : s_c = 3)
  (ht : total_cost = 30)
  (hm_books : m_b = 2)
  (hs_books : s_b = 6)
  (ha_books : a_b = 3)
  : ∃ (a_c : ℕ), a_c = 2 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_art_book_l1544_154482


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l1544_154413

def arithmetic_sum (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_ratio :
  let a1 := 3
  let d1 := 3
  let l1 := 99
  let a2 := 4
  let d2 := 4
  let l2 := 100
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let sum1 := arithmetic_sum a1 d1 n1
  let sum2 := arithmetic_sum a2 d2 n2
  sum1 / sum2 = 1683 / 1300 :=
by {
  let a1 := 3
  let d1 := 3
  let l1 := 99
  let a2 := 4
  let d2 := 4
  let l2 := 100
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let sum1 := arithmetic_sum a1 d1 n1
  let sum2 := arithmetic_sum a2 d2 n2
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l1544_154413


namespace NUMINAMATH_GPT_ratio_length_to_breadth_l1544_154496

theorem ratio_length_to_breadth (b l : ℕ) (A : ℕ) (h1 : b = 30) (h2 : A = 2700) (h3 : A = l * b) :
  l / b = 3 :=
by sorry

end NUMINAMATH_GPT_ratio_length_to_breadth_l1544_154496


namespace NUMINAMATH_GPT_find_x_l1544_154481

theorem find_x : 
  ∀ x : ℝ, (1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4)) → x = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l1544_154481


namespace NUMINAMATH_GPT_percentage_increase_l1544_154425

theorem percentage_increase
  (initial_earnings new_earnings : ℝ)
  (h_initial : initial_earnings = 55)
  (h_new : new_earnings = 60) :
  ((new_earnings - initial_earnings) / initial_earnings * 100) = 9.09 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1544_154425


namespace NUMINAMATH_GPT_find_coordinates_of_B_l1544_154451

-- Define points A and B, and vector a
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -1, y := 5 }
def a : Point := { x := 2, y := 3 }

-- Define the proof problem
theorem find_coordinates_of_B (B : Point) 
  (h1 : B.x + 1 = 3 * a.x)
  (h2 : B.y - 5 = 3 * a.y) : 
  B = { x := 5, y := 14 } := 
sorry

end NUMINAMATH_GPT_find_coordinates_of_B_l1544_154451


namespace NUMINAMATH_GPT_sufficient_condition_for_odd_l1544_154453

noncomputable def f (a x : ℝ) : ℝ :=
  Real.log (Real.sqrt (x^2 + a^2) - x)

theorem sufficient_condition_for_odd (a : ℝ) :
  (∀ x : ℝ, f 1 (-x) = -f 1 x) ∧
  (∀ x : ℝ, f (-1) (-x) = -f (-1) x) → 
  (a = 1 → ∀ x : ℝ, f a (-x) = -f a x) ∧ 
  (a ≠ 1 → ∃ x : ℝ, f a (-x) ≠ -f a x) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_odd_l1544_154453


namespace NUMINAMATH_GPT_prove_angle_BFD_l1544_154418

def given_conditions (A : ℝ) (AFG AGF : ℝ) : Prop :=
  A = 40 ∧ AFG = AGF

theorem prove_angle_BFD (A AFG AGF BFD : ℝ) (h1 : given_conditions A AFG AGF) : BFD = 110 :=
  by
  -- Utilize the conditions h1 stating that A = 40 and AFG = AGF
  sorry

end NUMINAMATH_GPT_prove_angle_BFD_l1544_154418


namespace NUMINAMATH_GPT_hyeoncheol_initial_money_l1544_154410

theorem hyeoncheol_initial_money
  (X : ℕ)
  (h1 : X / 2 / 2 = 1250) :
  X = 5000 :=
sorry

end NUMINAMATH_GPT_hyeoncheol_initial_money_l1544_154410


namespace NUMINAMATH_GPT_sin_C_value_l1544_154462

theorem sin_C_value (A B C : ℝ) (a b c : ℝ) 
  (h_a : a = 1) 
  (h_b : b = 1/2) 
  (h_cos_A : Real.cos A = (Real.sqrt 3) / 2) 
  (h_angles : A + B + C = Real.pi) 
  (h_sides : Real.sin A / a = Real.sin B / b) :
  Real.sin C = (Real.sqrt 15 + Real.sqrt 3) / 8 :=
by 
  sorry

end NUMINAMATH_GPT_sin_C_value_l1544_154462


namespace NUMINAMATH_GPT_constant_term_in_binomial_expansion_l1544_154460

theorem constant_term_in_binomial_expansion 
  (a b : ℕ) (n : ℕ)
  (sum_of_coefficients : (1 + 1)^n = 4)
  (A B : ℕ)
  (sum_A_B : A + B = 72) 
  (A_value : A = 4) :
  (b^2 = 9) :=
by sorry

end NUMINAMATH_GPT_constant_term_in_binomial_expansion_l1544_154460


namespace NUMINAMATH_GPT_ad_eb_intersect_on_altitude_l1544_154439

open EuclideanGeometry

variables {A B C D E F G K L C1 : Point}

-- Definitions for the problem
variables (triangleABC : Triangle A B C)
  (squareAEFC : Square A E F C)
  (squareBDGC : Square B D G C)
  (altitudeCC1 : Line C C1)
  (lineDA : Line A D)
  (lineEB : Line B E)

-- Definition of intersection
def intersects_on_altitude (pt : Point) : Prop :=
  pt ∈ lineDA ∧ pt ∈ lineEB ∧ pt ∈ altitudeCC1

-- The theorem to be proved
theorem ad_eb_intersect_on_altitude : 
  ∃ pt : Point, intersects_on_altitude lineDA lineEB altitudeCC1 pt := 
sorry

end NUMINAMATH_GPT_ad_eb_intersect_on_altitude_l1544_154439


namespace NUMINAMATH_GPT_shuffleboard_total_games_l1544_154485

theorem shuffleboard_total_games
    (jerry_wins : ℕ)
    (dave_wins : ℕ)
    (ken_wins : ℕ)
    (h1 : jerry_wins = 7)
    (h2 : dave_wins = jerry_wins + 3)
    (h3 : ken_wins = dave_wins + 5) :
    jerry_wins + dave_wins + ken_wins = 32 := 
by
  sorry

end NUMINAMATH_GPT_shuffleboard_total_games_l1544_154485


namespace NUMINAMATH_GPT_solve_system_eq_l1544_154478

theorem solve_system_eq (x y z b : ℝ) :
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧ 
  (x + y + z = 2 * b) ∧ 
  (x^2 + y^2 - z^2 = b^2) → 
  ( ∃ t : ℝ, (x = (1 + t) * b) ∧ (y = (1 - t) * b) ∧ (z = 0) ∧ t^2 = -1/2 ) :=
by
  -- proof will be filled in here
  sorry

end NUMINAMATH_GPT_solve_system_eq_l1544_154478


namespace NUMINAMATH_GPT_correct_choice_D_l1544_154491

theorem correct_choice_D (a : ℝ) :
  (2 * a ^ 2) ^ 3 = 8 * a ^ 6 ∧ 
  (a ^ 10 * a ^ 2 ≠ a ^ 20) ∧ 
  (a ^ 10 / a ^ 2 ≠ a ^ 5) ∧ 
  ((Real.pi - 3) ^ 0 ≠ 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_choice_D_l1544_154491


namespace NUMINAMATH_GPT_gas_cost_l1544_154433

theorem gas_cost 
  (x : ℝ)
  (h1 : 5 * (x / 5) = x)
  (h2 : 8 * (x / 8) = x)
  (h3 : (x / 5) - 15.50 = (x / 8)) : 
  x = 206.67 :=
by
  sorry

end NUMINAMATH_GPT_gas_cost_l1544_154433


namespace NUMINAMATH_GPT_slope_AA_l1544_154435

-- Define the points and conditions
variable (a b c d e f : ℝ)

-- Assumptions
#check (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
#check (a ≠ b ∧ c ≠ d ∧ e ≠ f)
#check (a+2 > 0 ∧ b > 0 ∧ c+2 > 0 ∧ d > 0 ∧ e+2 > 0 ∧ f > 0)

-- Main Statement
theorem slope_AA'_not_negative_one
    (H1: a > 0) (H2: b > 0) (H3: c > 0) (H4: d > 0)
    (H5: e > 0) (H6: f > 0) 
    (H7: a ≠ b) (H8: c ≠ d) (H9: e ≠ f)
    (H10: a + 2 > 0) (H11: c + 2 > 0) (H12: e + 2 > 0) : 
    (a ≠ b) → (c ≠ d) → (e ≠ f) → ¬( (a + 2 - b) / (b - a) = -1 ) :=
by
  sorry

end NUMINAMATH_GPT_slope_AA_l1544_154435


namespace NUMINAMATH_GPT_total_short_trees_after_planting_l1544_154471

def current_short_oak_trees := 3
def current_short_pine_trees := 4
def current_short_maple_trees := 5
def new_short_oak_trees := 9
def new_short_pine_trees := 6
def new_short_maple_trees := 4

theorem total_short_trees_after_planting :
  current_short_oak_trees + current_short_pine_trees + current_short_maple_trees +
  new_short_oak_trees + new_short_pine_trees + new_short_maple_trees = 31 := by
  sorry

end NUMINAMATH_GPT_total_short_trees_after_planting_l1544_154471


namespace NUMINAMATH_GPT_integer_product_l1544_154465

open Real

theorem integer_product (P Q R S : ℕ) (h1 : P + Q + R + S = 48)
    (h2 : P + 3 = Q - 3) (h3 : P + 3 = R * 3) (h4 : P + 3 = S / 3) :
    P * Q * R * S = 5832 :=
sorry

end NUMINAMATH_GPT_integer_product_l1544_154465


namespace NUMINAMATH_GPT_perfect_square_x4_x3_x2_x1_1_eq_x0_l1544_154423

theorem perfect_square_x4_x3_x2_x1_1_eq_x0 :
  ∀ x : ℤ, ∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2 ↔ x = 0 :=
by sorry

end NUMINAMATH_GPT_perfect_square_x4_x3_x2_x1_1_eq_x0_l1544_154423


namespace NUMINAMATH_GPT_correct_answer_is_option_d_l1544_154412

def is_quadratic (eq : String) : Prop :=
  eq = "a*x^2 + b*x + c = 0"

def OptionA : String := "1/x^2 + x - 1 = 0"
def OptionB : String := "3x + 1 = 5x + 4"
def OptionC : String := "x^2 + y = 0"
def OptionD : String := "x^2 - 2x + 1 = 0"

theorem correct_answer_is_option_d :
  is_quadratic OptionD :=
by
  sorry

end NUMINAMATH_GPT_correct_answer_is_option_d_l1544_154412


namespace NUMINAMATH_GPT_average_infection_rate_l1544_154408

theorem average_infection_rate (x : ℕ) : 
  1 + x + x * (1 + x) = 81 :=
sorry

end NUMINAMATH_GPT_average_infection_rate_l1544_154408


namespace NUMINAMATH_GPT_range_of_m_l1544_154452

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - 4*x ≥ m) → m ≤ -3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l1544_154452


namespace NUMINAMATH_GPT_mean_home_runs_correct_l1544_154499

def mean_home_runs (players: List ℕ) (home_runs: List ℕ) : ℚ :=
  let total_runs := (List.zipWith (· * ·) players home_runs).sum
  let total_players := players.sum
  total_runs / total_players

theorem mean_home_runs_correct :
  mean_home_runs [6, 4, 3, 1, 1, 1] [6, 7, 8, 10, 11, 12] = 121 / 16 :=
by
  -- The proof should go here
  sorry

end NUMINAMATH_GPT_mean_home_runs_correct_l1544_154499


namespace NUMINAMATH_GPT_number_of_friends_l1544_154489

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_friends_l1544_154489


namespace NUMINAMATH_GPT_Brandon_can_still_apply_l1544_154458

-- Definitions based on the given conditions
def total_businesses : ℕ := 72
def fired_businesses : ℕ := total_businesses / 2
def quit_businesses : ℕ := total_businesses / 3
def businesses_restricted : ℕ := fired_businesses + quit_businesses

-- The final proof statement
theorem Brandon_can_still_apply : total_businesses - businesses_restricted = 12 :=
by
  -- Note: Proof is omitted; replace sorry with detailed proof in practice.
  sorry

end NUMINAMATH_GPT_Brandon_can_still_apply_l1544_154458


namespace NUMINAMATH_GPT_tangent_line_intercept_l1544_154473

theorem tangent_line_intercept:
  ∃ (m b : ℚ), 
    m > 0 ∧ 
    b = 135 / 28 ∧ 
    (∀ x y : ℚ, (y - 3)^2 + (x - 1)^2 ≥ 3^2 → (y - 8)^2 + (x - 10)^2 ≥ 6^2 → y = m * x + b) := 
sorry

end NUMINAMATH_GPT_tangent_line_intercept_l1544_154473


namespace NUMINAMATH_GPT_color_plane_with_two_colors_l1544_154443

/-- Given a finite set of circles that divides the plane into regions, we can color the plane such that no two adjacent regions have the same color. -/
theorem color_plane_with_two_colors (circles : Finset (Set ℝ)) :
  (∀ (r1 r2 : Set ℝ), (r1 ∩ r2).Nonempty → ∃ (coloring : Set ℝ → Bool), (coloring r1 ≠ coloring r2)) :=
  sorry

end NUMINAMATH_GPT_color_plane_with_two_colors_l1544_154443


namespace NUMINAMATH_GPT_sin_double_angle_log_simplification_l1544_154484

-- Problem 1: Prove sin(2 * α) = 7 / 25 given sin(α - π / 4) = 3 / 5
theorem sin_double_angle (α : ℝ) (h : Real.sin (α - Real.pi / 4) = 3 / 5) : Real.sin (2 * α) = 7 / 25 :=
by
  sorry

-- Problem 2: Prove 2 * log₅ 10 + log₅ 0.25 = 2
theorem log_simplification : 2 * Real.log 10 / Real.log 5 + Real.log (0.25) / Real.log 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_log_simplification_l1544_154484


namespace NUMINAMATH_GPT_Malik_yards_per_game_l1544_154454

-- Definitions of the conditions
def number_of_games : ℕ := 4
def josiah_yards_per_game : ℕ := 22
def darnell_average_yards_per_game : ℕ := 11
def total_yards_all_athletes : ℕ := 204

-- The statement to prove
theorem Malik_yards_per_game (M : ℕ) 
  (H1 : number_of_games = 4) 
  (H2 : josiah_yards_per_game = 22) 
  (H3 : darnell_average_yards_per_game = 11) 
  (H4 : total_yards_all_athletes = 204) :
  4 * M + 4 * 22 + 4 * 11 = 204 → M = 18 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_Malik_yards_per_game_l1544_154454


namespace NUMINAMATH_GPT_cole_drive_time_l1544_154414

theorem cole_drive_time (D : ℝ) (T_work T_home : ℝ) 
  (h1 : T_work = D / 75) 
  (h2 : T_home = D / 105)
  (h3 : T_work + T_home = 4) : 
  T_work * 60 = 140 := 
by sorry

end NUMINAMATH_GPT_cole_drive_time_l1544_154414


namespace NUMINAMATH_GPT_product_expansion_l1544_154469

theorem product_expansion (x : ℝ) : 2 * (x + 3) * (x + 4) = 2 * x^2 + 14 * x + 24 := 
by
  sorry

end NUMINAMATH_GPT_product_expansion_l1544_154469


namespace NUMINAMATH_GPT_geometric_progression_fourth_term_l1544_154466

theorem geometric_progression_fourth_term :
  let a1 := 2^(1/2)
  let a2 := 2^(1/4)
  let a3 := 2^(1/8)
  a4 = 2^(1/16) :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_fourth_term_l1544_154466


namespace NUMINAMATH_GPT_part_I_part_II_l1544_154428

variable (a b c : ℝ)

theorem part_I (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : ∀ x : ℝ, |x + a| + |x - b| + c ≥ 4) : a + b + c = 4 :=
sorry

theorem part_II (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 4) : (1/4) * a^2 + (1/9) * b^2 + c^2 ≥ 8/7 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1544_154428


namespace NUMINAMATH_GPT_price_per_postcard_is_correct_l1544_154488

noncomputable def initial_postcards : ℕ := 18
noncomputable def sold_postcards : ℕ := initial_postcards / 2
noncomputable def price_per_postcard_sold : ℕ := 15
noncomputable def total_earned : ℕ := sold_postcards * price_per_postcard_sold
noncomputable def total_postcards_after : ℕ := 36
noncomputable def remaining_original_postcards : ℕ := initial_postcards - sold_postcards
noncomputable def new_postcards_bought : ℕ := total_postcards_after - remaining_original_postcards
noncomputable def price_per_new_postcard : ℕ := total_earned / new_postcards_bought

theorem price_per_postcard_is_correct:
  price_per_new_postcard = 5 :=
by
  sorry

end NUMINAMATH_GPT_price_per_postcard_is_correct_l1544_154488


namespace NUMINAMATH_GPT_woman_work_rate_l1544_154437

theorem woman_work_rate (W : ℝ) :
  (1 / 6) + W + (1 / 9) = (1 / 3) → W = (1 / 18) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_woman_work_rate_l1544_154437


namespace NUMINAMATH_GPT_average_annual_percentage_decrease_l1544_154420

theorem average_annual_percentage_decrease (P2018 P2020 : ℝ) (x : ℝ) 
  (h_initial : P2018 = 20000)
  (h_final : P2020 = 16200) :
  P2018 * (1 - x)^2 = P2020 :=
by
  sorry

end NUMINAMATH_GPT_average_annual_percentage_decrease_l1544_154420


namespace NUMINAMATH_GPT_correct_answer_l1544_154426

variables (A B : polynomial ℝ) (a : ℝ)

theorem correct_answer (hB : B = 3 * a^2 - 5 * a - 7) (hMistake : A - 2 * B = -2 * a^2 + 3 * a + 6) :
  A + 2 * B = 10 * a^2 - 17 * a - 22 :=
by
  sorry

end NUMINAMATH_GPT_correct_answer_l1544_154426


namespace NUMINAMATH_GPT_minimum_value_of_f_l1544_154486

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem minimum_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -1 / Real.exp 1 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1544_154486


namespace NUMINAMATH_GPT_simplify_expression_is_one_fourth_l1544_154440

noncomputable def fourth_root (x : ℝ) : ℝ := x ^ (1 / 4)
noncomputable def square_root (x : ℝ) : ℝ := x ^ (1 / 2)
noncomputable def simplified_expression : ℝ := (fourth_root 81 - square_root 12.25) ^ 2

theorem simplify_expression_is_one_fourth : simplified_expression = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_is_one_fourth_l1544_154440


namespace NUMINAMATH_GPT_correct_calculation_l1544_154415

variable (a : ℝ)

theorem correct_calculation (a : ℝ) : (2 * a)^2 / (4 * a) = a := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1544_154415
