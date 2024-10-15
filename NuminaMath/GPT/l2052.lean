import Mathlib

namespace NUMINAMATH_GPT_mira_additional_stickers_l2052_205204

-- Define the conditions
def mira_stickers : ℕ := 31
def row_size : ℕ := 7

-- Define the proof statement
theorem mira_additional_stickers (a : ℕ) (h : (31 + a) % 7 = 0) : 
  a = 4 := 
sorry

end NUMINAMATH_GPT_mira_additional_stickers_l2052_205204


namespace NUMINAMATH_GPT_find_ratio_PS_SR_l2052_205284

variable {P Q R S : Type}
variable [MetricSpace P]
variable [MetricSpace Q]
variable [MetricSpace R]
variable [MetricSpace S]

-- Given conditions
variable (PQ QR PR : ℝ)
variable (hPQ : PQ = 6)
variable (hQR : QR = 8)
variable (hPR : PR = 10)
variable (QS : ℝ)
variable (hQS : QS = 6)

-- Points on the segments
variable (PS : ℝ)
variable (SR : ℝ)

-- The theorem to be proven: the ratio PS : SR = 0 : 1
theorem find_ratio_PS_SR (hPQ : PQ = 6) (hQR : QR = 8) (hPR : PR = 10) (hQS : QS = 6) :
    PS = 0 ∧ SR = 10 → PS / SR = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_ratio_PS_SR_l2052_205284


namespace NUMINAMATH_GPT_exists_matrices_B_C_not_exists_matrices_commute_l2052_205227

-- Equivalent proof statement for part (a)
theorem exists_matrices_B_C (A : Matrix (Fin 2) (Fin 2) ℝ): 
  ∃ (B C : Matrix (Fin 2) (Fin 2) ℝ), A = B^2 + C^2 :=
by
  sorry

-- Equivalent proof statement for part (b)
theorem not_exists_matrices_commute (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (hA : A = ![![0, 1], ![1, 0]]) :
  ¬∃ (B C: Matrix (Fin 2) (Fin 2) ℝ), A = B^2 + C^2 ∧ B * C = C * B :=
by
  sorry

end NUMINAMATH_GPT_exists_matrices_B_C_not_exists_matrices_commute_l2052_205227


namespace NUMINAMATH_GPT_solve_for_k_l2052_205234

theorem solve_for_k (k : ℤ) : (∃ x : ℤ, x = 5 ∧ 4 * x + 2 * k = 8) → k = -6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l2052_205234


namespace NUMINAMATH_GPT_minimum_value_a5_a6_l2052_205210

-- Defining the arithmetic geometric sequence relational conditions.
def arithmetic_geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * q) ∧ (a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6) ∧ (∀ n, a n > 0)

-- The mathematical problem to prove:
theorem minimum_value_a5_a6 (a : ℕ → ℝ) (q : ℝ) (h : arithmetic_geometric_sequence_condition a q) :
  a 5 + a 6 = 48 :=
sorry

end NUMINAMATH_GPT_minimum_value_a5_a6_l2052_205210


namespace NUMINAMATH_GPT_cube_painting_l2052_205221

theorem cube_painting (n : ℕ) (h₁ : n > 4) 
  (h₂ : (2 * (n - 2)) = (n^2 - 2*n + 1)) : n = 5 :=
sorry

end NUMINAMATH_GPT_cube_painting_l2052_205221


namespace NUMINAMATH_GPT_total_cupcakes_correct_l2052_205202

def cupcakes_per_event : ℝ := 96.0
def num_events : ℝ := 8.0
def total_cupcakes : ℝ := cupcakes_per_event * num_events

theorem total_cupcakes_correct : total_cupcakes = 768.0 :=
by
  unfold total_cupcakes
  unfold cupcakes_per_event
  unfold num_events
  sorry

end NUMINAMATH_GPT_total_cupcakes_correct_l2052_205202


namespace NUMINAMATH_GPT_complement_intersection_l2052_205228

open Set

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3, 4}
def B : Set ℕ := {2, 4}

theorem complement_intersection :
  ((U \ A) ∩ B) = {2} :=
sorry

end NUMINAMATH_GPT_complement_intersection_l2052_205228


namespace NUMINAMATH_GPT_compute_expression_l2052_205299

theorem compute_expression : 9 * (1 / 13) * 26 = 18 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l2052_205299


namespace NUMINAMATH_GPT_lauren_time_8_miles_l2052_205238

-- Conditions
def time_alex_run_6_miles : ℕ := 36
def time_lauren_run_5_miles : ℕ := time_alex_run_6_miles / 3
def time_per_mile_lauren : ℚ := time_lauren_run_5_miles / 5

-- Proof statement
theorem lauren_time_8_miles : 8 * time_per_mile_lauren = 19.2 := by
  sorry

end NUMINAMATH_GPT_lauren_time_8_miles_l2052_205238


namespace NUMINAMATH_GPT_jacks_walking_rate_l2052_205286

theorem jacks_walking_rate :
  let distance := 8
  let time_in_minutes := 1 * 60 + 15
  let time := time_in_minutes / 60.0
  let rate := distance / time
  rate = 6.4 :=
by
  sorry

end NUMINAMATH_GPT_jacks_walking_rate_l2052_205286


namespace NUMINAMATH_GPT_arithmetic_sequence_inequality_l2052_205260

variables {a : ℕ → ℝ} {d a1 : ℝ}

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + (n - 1) * d

-- All terms are positive
def all_positive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

theorem arithmetic_sequence_inequality
  (h_arith_seq : is_arithmetic_sequence a a1 d)
  (h_non_zero_diff : d ≠ 0)
  (h_positive : all_positive a) :
  (a 1) * (a 8) < (a 4) * (a 5) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_inequality_l2052_205260


namespace NUMINAMATH_GPT_rationalize_denominator_sum_l2052_205270

theorem rationalize_denominator_sum :
  let A := -4
  let B := 7
  let C := 3
  let D := 13
  let E := 1
  A + B + C + D + E = 20 := by
    sorry

end NUMINAMATH_GPT_rationalize_denominator_sum_l2052_205270


namespace NUMINAMATH_GPT_smallest_integer_ends_3_divisible_5_l2052_205203

theorem smallest_integer_ends_3_divisible_5 : ∃ n : ℕ, (53 = n ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, (m % 10 = 3) ∧ (m % 5 = 0) → 53 ≤ m) :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_integer_ends_3_divisible_5_l2052_205203


namespace NUMINAMATH_GPT_sum_xy_l2052_205219

theorem sum_xy (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y + 10) : x + y = 14 ∨ x + y = -2 :=
sorry

end NUMINAMATH_GPT_sum_xy_l2052_205219


namespace NUMINAMATH_GPT_animal_costs_l2052_205226

theorem animal_costs :
  ∃ (C G S P : ℕ),
      C + G + S + P = 1325 ∧
      G + S + P = 425 ∧
      C + S + P = 1225 ∧
      G + P = 275 ∧
      C = 900 ∧
      G = 100 ∧
      S = 150 ∧
      P = 175 :=
by
  sorry

end NUMINAMATH_GPT_animal_costs_l2052_205226


namespace NUMINAMATH_GPT_clock_in_2023_hours_l2052_205201

theorem clock_in_2023_hours (current_time : ℕ) (h_current_time : current_time = 3) : 
  (current_time + 2023) % 12 = 10 := 
by {
  -- context: non-computational (time kept in modulo world and not real increments)
  sorry
}

end NUMINAMATH_GPT_clock_in_2023_hours_l2052_205201


namespace NUMINAMATH_GPT_add_like_terms_l2052_205276

variable (a : ℝ)

theorem add_like_terms : a^2 + 2 * a^2 = 3 * a^2 := 
by sorry

end NUMINAMATH_GPT_add_like_terms_l2052_205276


namespace NUMINAMATH_GPT_batsman_running_percentage_l2052_205295

theorem batsman_running_percentage (total_runs boundary_runs six_runs : ℕ) 
  (h1 : total_runs = 120) (h2 : boundary_runs = 3 * 4) (h3 : six_runs = 8 * 6) : 
  (total_runs - (boundary_runs + six_runs)) * 100 / total_runs = 50 := 
sorry

end NUMINAMATH_GPT_batsman_running_percentage_l2052_205295


namespace NUMINAMATH_GPT_students_in_ms_delmont_class_l2052_205265

-- Let us define the necessary conditions

def total_cupcakes : Nat := 40
def students_mrs_donnelly_class : Nat := 16
def adults_count : Nat := 4 -- Ms. Delmont, Mrs. Donnelly, the school nurse, and the school principal
def leftover_cupcakes : Nat := 2

-- Define the number of students in Ms. Delmont's class
def students_ms_delmont_class : Nat := 18

-- The statement to prove
theorem students_in_ms_delmont_class :
  total_cupcakes - adults_count - students_mrs_donnelly_class - leftover_cupcakes = students_ms_delmont_class :=
by
  sorry

end NUMINAMATH_GPT_students_in_ms_delmont_class_l2052_205265


namespace NUMINAMATH_GPT_probability_five_cards_one_from_each_suit_and_extra_l2052_205269

/--
Given five cards chosen with replacement from a standard 52-card deck, 
the probability of having exactly one card from each suit, plus one 
additional card from any suit, is 3/32.
-/
theorem probability_five_cards_one_from_each_suit_and_extra 
  (cards : ℕ) (total_suits : ℕ)
  (prob_first_diff_suit : ℚ) 
  (prob_second_diff_suit : ℚ) 
  (prob_third_diff_suit : ℚ) 
  (prob_fourth_diff_suit : ℚ) 
  (prob_any_suit : ℚ) 
  (total_prob : ℚ) :
  cards = 5 ∧ total_suits = 4 ∧ 
  prob_first_diff_suit = 3 / 4 ∧ 
  prob_second_diff_suit = 1 / 2 ∧ 
  prob_third_diff_suit = 1 / 4 ∧ 
  prob_fourth_diff_suit = 1 ∧ 
  prob_any_suit = 1 →
  total_prob = 3 / 32 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_five_cards_one_from_each_suit_and_extra_l2052_205269


namespace NUMINAMATH_GPT_jamie_cherry_pies_l2052_205256

theorem jamie_cherry_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h_total : total_pies = 36) (h_ratio : apple_ratio = 2 ∧ blueberry_ratio = 5 ∧ cherry_ratio = 4) : 
  (cherry_ratio * total_pies) / (apple_ratio + blueberry_ratio + cherry_ratio) = 144 / 11 := 
by {
  sorry
}

end NUMINAMATH_GPT_jamie_cherry_pies_l2052_205256


namespace NUMINAMATH_GPT_find_point_P_l2052_205215

noncomputable def f (x : ℝ) : ℝ := x^2 - x

theorem find_point_P :
  (∃ x y : ℝ, f x = y ∧ (2 * x - 1 = 1) ∧ (y = x^2 - x)) ∧ (x = 1) ∧ (y = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_point_P_l2052_205215


namespace NUMINAMATH_GPT_solve_eq_l2052_205240

theorem solve_eq : ∃ x : ℝ, 6 * x - 4 * x = 380 - 10 * (x + 2) ∧ x = 30 := 
by
  sorry

end NUMINAMATH_GPT_solve_eq_l2052_205240


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2052_205288

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 5) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2052_205288


namespace NUMINAMATH_GPT_invalid_speed_against_stream_l2052_205225

theorem invalid_speed_against_stream (rate_still_water speed_with_stream : ℝ) (h1 : rate_still_water = 6) (h2 : speed_with_stream = 20) :
  ∃ (v : ℝ), speed_with_stream = rate_still_water + v ∧ (rate_still_water - v < 0) → false :=
by
  sorry

end NUMINAMATH_GPT_invalid_speed_against_stream_l2052_205225


namespace NUMINAMATH_GPT_problem_value_l2052_205214

theorem problem_value :
  (1 / 3 * 9 * 1 / 27 * 81 * 1 / 243 * 729 * 1 / 2187 * 6561 * 1 / 19683 * 59049) = 243 := 
sorry

end NUMINAMATH_GPT_problem_value_l2052_205214


namespace NUMINAMATH_GPT_new_sequence_69th_term_l2052_205241

-- Definitions and conditions
def original_sequence (a : ℕ → ℕ) (n : ℕ) : ℕ := a n

def new_sequence (a : ℕ → ℕ) (k : ℕ) : ℕ :=
if k % 4 = 1 then a (k / 4 + 1) else 0  -- simplified modeling, the inserted numbers are denoted arbitrarily as 0

-- The statement to be proven
theorem new_sequence_69th_term (a : ℕ → ℕ) : new_sequence a 69 = a 18 :=
by
  sorry

end NUMINAMATH_GPT_new_sequence_69th_term_l2052_205241


namespace NUMINAMATH_GPT_max_value_l2052_205289

noncomputable def max_expression (x : ℝ) : ℝ :=
  3^x - 2 * 9^x

theorem max_value : ∃ x : ℝ, max_expression x = 1 / 8 :=
sorry

end NUMINAMATH_GPT_max_value_l2052_205289


namespace NUMINAMATH_GPT_pennies_to_quarters_ratio_l2052_205267

-- Define the given conditions as assumptions
variables (pennies dimes nickels quarters: ℕ)

-- Given conditions
axiom cond1 : dimes = pennies + 10
axiom cond2 : nickels = 2 * dimes
axiom cond3 : quarters = 4
axiom cond4 : nickels = 100

-- Theorem stating the final result should be a certain ratio
theorem pennies_to_quarters_ratio (hpn : pennies = 40) : pennies / quarters = 10 := 
by sorry

end NUMINAMATH_GPT_pennies_to_quarters_ratio_l2052_205267


namespace NUMINAMATH_GPT_count_100_digit_numbers_divisible_by_3_l2052_205277

def num_100_digit_numbers_divisible_by_3 : ℕ := (4^50 + 2) / 3

theorem count_100_digit_numbers_divisible_by_3 :
  ∃ n : ℕ, n = num_100_digit_numbers_divisible_by_3 :=
by
  use (4^50 + 2) / 3
  sorry

end NUMINAMATH_GPT_count_100_digit_numbers_divisible_by_3_l2052_205277


namespace NUMINAMATH_GPT_sarah_bottle_caps_total_l2052_205292

def initial_caps : ℕ := 450
def first_day_caps : ℕ := 175
def second_day_caps : ℕ := 95
def third_day_caps : ℕ := 220
def total_caps : ℕ := 940

theorem sarah_bottle_caps_total : 
    initial_caps + first_day_caps + second_day_caps + third_day_caps = total_caps :=
by
  sorry

end NUMINAMATH_GPT_sarah_bottle_caps_total_l2052_205292


namespace NUMINAMATH_GPT_fraction_inhabitable_earth_surface_l2052_205246

theorem fraction_inhabitable_earth_surface 
  (total_land_fraction: ℚ) 
  (inhabitable_land_fraction: ℚ) 
  (h1: total_land_fraction = 1/3) 
  (h2: inhabitable_land_fraction = 2/3) 
  : (total_land_fraction * inhabitable_land_fraction) = 2/9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_inhabitable_earth_surface_l2052_205246


namespace NUMINAMATH_GPT_sum_k1_k2_k3_l2052_205206

theorem sum_k1_k2_k3 :
  ∀ (k1 k2 k3 t1 t2 t3 : ℝ),
  t1 = 105 →
  t2 = 80 →
  t3 = 45 →
  t1 = (5 / 9) * (k1 - 32) →
  t2 = (5 / 9) * (k2 - 32) →
  t3 = (5 / 9) * (k3 - 32) →
  k1 + k2 + k3 = 510 :=
by
  intros k1 k2 k3 t1 t2 t3 ht1 ht2 ht3 ht1k1 ht2k2 ht3k3
  sorry

end NUMINAMATH_GPT_sum_k1_k2_k3_l2052_205206


namespace NUMINAMATH_GPT_zeros_of_f_l2052_205275

noncomputable def f (a : ℝ) (x : ℝ) :=
if x ≤ 1 then a + 2^x else (1/2) * x + a

theorem zeros_of_f (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ a ∈ Set.Ico (-2) (-1/2) :=
sorry

end NUMINAMATH_GPT_zeros_of_f_l2052_205275


namespace NUMINAMATH_GPT_sin_135_eq_sqrt2_div_2_l2052_205268

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end NUMINAMATH_GPT_sin_135_eq_sqrt2_div_2_l2052_205268


namespace NUMINAMATH_GPT_proof_problem_1_proof_problem_2_l2052_205248

noncomputable def problem_1 (a b : ℝ) : Prop :=
  ((2 * a^(3/2) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3))) / (-3 * a^(1/6) * b^(5/6)) = 4 * a^(11/6)

noncomputable def problem_2 : Prop :=
  ((2^(1/3) * 3^(1/2))^6 + (2^(1/2) * 2^(1/4))^(4/3) - 2^(1/4) * 2^(3/4 - 1) - (-2005)^0) = 100

theorem proof_problem_1 (a b : ℝ) : problem_1 a b := 
  sorry

theorem proof_problem_2 : problem_2 := 
  sorry

end NUMINAMATH_GPT_proof_problem_1_proof_problem_2_l2052_205248


namespace NUMINAMATH_GPT_suji_age_problem_l2052_205236

theorem suji_age_problem (x : ℕ) 
  (h1 : 5 * x + 6 = 13 * (4 * x + 6) / 11)
  (h2 : 11 * (4 * x + 6) = 9 * (3 * x + 6)) :
  4 * x = 16 :=
by
  sorry

end NUMINAMATH_GPT_suji_age_problem_l2052_205236


namespace NUMINAMATH_GPT_triplet_zero_solution_l2052_205216

theorem triplet_zero_solution (x y z : ℝ) 
  (h1 : x^3 + y = z^2) 
  (h2 : y^3 + z = x^2) 
  (h3 : z^3 + x = y^2) :
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end NUMINAMATH_GPT_triplet_zero_solution_l2052_205216


namespace NUMINAMATH_GPT_total_saplings_l2052_205290

theorem total_saplings (a_efficiency b_efficiency : ℝ) (A B T n : ℝ) 
  (h1 : a_efficiency = (3/4))
  (h2 : b_efficiency = 1)
  (h3 : B = n + 36)
  (h4 : T = 2 * n + 36)
  (h5 : n * (4/3) = n + 36)
  : T = 252 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_saplings_l2052_205290


namespace NUMINAMATH_GPT_no_integers_satisfy_l2052_205257

theorem no_integers_satisfy (a b c d : ℤ) : ¬ (a^4 + b^4 + c^4 + 2016 = 10 * d) :=
sorry

end NUMINAMATH_GPT_no_integers_satisfy_l2052_205257


namespace NUMINAMATH_GPT_find_p_of_five_l2052_205232

-- Define the cubic polynomial and the conditions
def cubic_poly (p : ℝ → ℝ) :=
  ∀ x, ∃ a b c d, p x = a * x^3 + b * x^2 + c * x + d

def satisfies_conditions (p : ℝ → ℝ) :=
  p 1 = 1 ^ 2 ∧
  p 2 = 2 ^ 2 ∧
  p 3 = 3 ^ 2 ∧
  p 4 = 4 ^ 2

-- Theorem statement to be proved
theorem find_p_of_five (p : ℝ → ℝ) (hcubic : cubic_poly p) (hconditions : satisfies_conditions p) : p 5 = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_p_of_five_l2052_205232


namespace NUMINAMATH_GPT_complement_union_eq_l2052_205281

namespace SetComplementUnion

-- Defining the universal set U, set M and set N.
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- Proving the desired equality
theorem complement_union_eq :
  (U \ M) ∪ N = {x | x > -1} :=
sorry

end SetComplementUnion

end NUMINAMATH_GPT_complement_union_eq_l2052_205281


namespace NUMINAMATH_GPT_projection_of_b_onto_a_l2052_205217

open Real

noncomputable def e1 : ℝ × ℝ := (1, 0)
noncomputable def e2 : ℝ × ℝ := (0, 1)

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (4, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def magnitude (u : ℝ × ℝ) : ℝ := sqrt (u.1 ^ 2 + u.2 ^ 2)
noncomputable def projection (u v : ℝ × ℝ) : ℝ := (dot_product u v) / (magnitude u)

theorem projection_of_b_onto_a : projection b a = 2 * sqrt 5 / 5 := by
  sorry

end NUMINAMATH_GPT_projection_of_b_onto_a_l2052_205217


namespace NUMINAMATH_GPT_money_left_after_shopping_l2052_205235

-- Conditions
def cost_mustard_oil : ℤ := 2 * 13
def cost_pasta : ℤ := 3 * 4
def cost_sauce : ℤ := 1 * 5
def total_cost : ℤ := cost_mustard_oil + cost_pasta + cost_sauce
def total_money : ℤ := 50

-- Theorem to prove
theorem money_left_after_shopping : total_money - total_cost = 7 := by
  sorry

end NUMINAMATH_GPT_money_left_after_shopping_l2052_205235


namespace NUMINAMATH_GPT_arithmetic_sequence_initial_term_l2052_205297

theorem arithmetic_sequence_initial_term (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum : ∀ n, S n = n * (a 1 + n * d / 2))
  (h_product : a 2 * a 3 = a 4 * a 5)
  (h_sum_9 : S 9 = 27)
  (h_d_nonzero : d ≠ 0) :
  a 1 = -5 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_initial_term_l2052_205297


namespace NUMINAMATH_GPT_percentage_profit_is_35_l2052_205263

-- Define the conditions
def initial_cost_price : ℝ := 100
def markup_percentage : ℝ := 0.5
def discount_percentage : ℝ := 0.1
def marked_price : ℝ := initial_cost_price * (1 + markup_percentage)
def selling_price : ℝ := marked_price * (1 - discount_percentage)

-- Define the statement/proof problem
theorem percentage_profit_is_35 :
  (selling_price - initial_cost_price) / initial_cost_price * 100 = 35 := by 
  sorry

end NUMINAMATH_GPT_percentage_profit_is_35_l2052_205263


namespace NUMINAMATH_GPT_fraction_of_work_left_l2052_205230

variable (A_work_days : ℕ) (B_work_days : ℕ) (work_days_together: ℕ)

theorem fraction_of_work_left (hA : A_work_days = 15) (hB : B_work_days = 20) (hT : work_days_together = 4):
  (1 - 4 * (1 / 15 + 1 / 20)) = (8 / 15) :=
sorry

end NUMINAMATH_GPT_fraction_of_work_left_l2052_205230


namespace NUMINAMATH_GPT_axis_of_symmetry_eq_l2052_205272

theorem axis_of_symmetry_eq : 
  ∃ k : ℤ, (λ x => 2 * Real.cos (2 * x)) = (λ x => 2 * Real.sin (2 * (x + π / 3) - π / 6)) ∧
            x = (1/2) * k * π ∧ x = -π / 2 := 
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_eq_l2052_205272


namespace NUMINAMATH_GPT_cats_left_l2052_205283

def initial_siamese_cats : ℕ := 12
def initial_house_cats : ℕ := 20
def cats_sold : ℕ := 20

theorem cats_left : (initial_siamese_cats + initial_house_cats - cats_sold) = 12 :=
by
sorry

end NUMINAMATH_GPT_cats_left_l2052_205283


namespace NUMINAMATH_GPT_sum_max_min_f_l2052_205252

noncomputable def f (x : ℝ) : ℝ :=
  1 + (Real.sin x / (2 + Real.cos x))

theorem sum_max_min_f {a b : ℝ} (ha : ∀ x, f x ≤ a) (hb : ∀ x, b ≤ f x) (h_max : ∃ x, f x = a) (h_min : ∃ x, f x = b) :
  a + b = 2 :=
sorry

end NUMINAMATH_GPT_sum_max_min_f_l2052_205252


namespace NUMINAMATH_GPT_minimum_expression_value_l2052_205250

theorem minimum_expression_value (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : 
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 := 
by
  sorry

end NUMINAMATH_GPT_minimum_expression_value_l2052_205250


namespace NUMINAMATH_GPT_min_time_needed_l2052_205264

-- Define the conditions and required time for shoeing horses
def num_blacksmiths := 48
def num_horses := 60
def hooves_per_horse := 4
def time_per_hoof := 5
def total_hooves := num_horses * hooves_per_horse
def total_time_one_blacksmith := total_hooves * time_per_hoof
def min_time (num_blacksmiths : Nat) (total_time_one_blacksmith : Nat) : Nat :=
  total_time_one_blacksmith / num_blacksmiths

-- Prove that the minimum time needed is 25 minutes
theorem min_time_needed : min_time num_blacksmiths total_time_one_blacksmith = 25 :=
by
  sorry

end NUMINAMATH_GPT_min_time_needed_l2052_205264


namespace NUMINAMATH_GPT_total_cows_is_108_l2052_205245

-- Definitions of the sons' shares and the number of cows the fourth son received
def first_son_share : ℚ := 2 / 3
def second_son_share : ℚ := 1 / 6
def third_son_share : ℚ := 1 / 9
def fourth_son_cows : ℕ := 6

-- The total number of cows in the herd
def total_cows (n : ℕ) : Prop :=
  first_son_share + second_son_share + third_son_share + (fourth_son_cows / n) = 1

-- Prove that given the number of cows the fourth son received, the total number of cows in the herd is 108
theorem total_cows_is_108 : total_cows 108 :=
by
  sorry

end NUMINAMATH_GPT_total_cows_is_108_l2052_205245


namespace NUMINAMATH_GPT_choir_members_l2052_205200

theorem choir_members (k m n : ℕ) (h1 : n = k^2 + 11) (h2 : n = m * (m + 5)) : n ≤ 325 :=
by
  sorry -- A proof would go here, showing that n = 325 meets the criteria

end NUMINAMATH_GPT_choir_members_l2052_205200


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l2052_205278

theorem arithmetic_geometric_sequence (x y z : ℤ) :
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
  ((x + y + z = 6) ∧ (y - x = z - y) ∧ (y^2 = x * z)) →
  (x = -4 ∧ y = 2 ∧ z = 8 ∨ x = 8 ∧ y = 2 ∧ z = -4) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l2052_205278


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l2052_205207

-- Define the sequences and conditions
variable {a b : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {T : ℕ → ℕ}
variable {d q : ℕ}
variable {b_initial : ℕ}

axiom geom_seq (n : ℕ) : b n = b_initial * q^n
axiom arith_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom sum_seq (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Problem conditions
axiom cond_geom_seq : b_initial = 2
axiom cond_geom_b2_b3 : b 2 + b 3 = 12
axiom cond_geom_ratio : q > 0
axiom cond_relation_b3_a4 : b 3 = a 4 - 2 * a 1
axiom cond_sum_S_11_b4 : S 11 = 11 * b 4

-- Theorem statement
theorem problem_part1 :
  (a n = 3 * n - 2) ∧ (b n = 2 ^ n) :=
  sorry

theorem problem_part2 :
  (T n = (3 * n - 2) / 3 * 4^(n + 1) + 8 / 3) :=
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l2052_205207


namespace NUMINAMATH_GPT_fraction_inequality_l2052_205251

theorem fraction_inequality (a : ℝ) (h : a ≠ 2) : (1 / (a^2 - 4 * a + 4) > 2 / (a^3 - 8)) :=
by sorry

end NUMINAMATH_GPT_fraction_inequality_l2052_205251


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2052_205209

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 4*x - 21 ≤ 0 ↔ -3 ≤ x ∧ x ≤ 7 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2052_205209


namespace NUMINAMATH_GPT_carpet_length_l2052_205261

theorem carpet_length (percent_covered : ℝ) (width : ℝ) (floor_area : ℝ) (carpet_length : ℝ) :
  percent_covered = 0.30 → width = 4 → floor_area = 120 → carpet_length = 9 :=
by
  sorry

end NUMINAMATH_GPT_carpet_length_l2052_205261


namespace NUMINAMATH_GPT_allan_balloons_l2052_205282

def jak_balloons : ℕ := 11
def diff_balloons : ℕ := 6

theorem allan_balloons (jake_allan_diff : jak_balloons = diff_balloons + 5) : jak_balloons - diff_balloons = 5 :=
by
  sorry

end NUMINAMATH_GPT_allan_balloons_l2052_205282


namespace NUMINAMATH_GPT_range_of_x_l2052_205247

theorem range_of_x {a : ℝ} : 
  (∀ a : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (x = 0 ∨ x = -2) :=
by sorry

end NUMINAMATH_GPT_range_of_x_l2052_205247


namespace NUMINAMATH_GPT_fred_balloons_l2052_205237

variable (initial_balloons : ℕ := 709)
variable (balloons_given : ℕ := 221)
variable (remaining_balloons : ℕ := 488)

theorem fred_balloons :
  initial_balloons - balloons_given = remaining_balloons :=
  by
    sorry

end NUMINAMATH_GPT_fred_balloons_l2052_205237


namespace NUMINAMATH_GPT_gain_percentage_l2052_205231

theorem gain_percentage (selling_price gain : ℕ) (h_sp : selling_price = 110) (h_gain : gain = 10) :
  (gain * 100) / (selling_price - gain) = 10 :=
by
  sorry

end NUMINAMATH_GPT_gain_percentage_l2052_205231


namespace NUMINAMATH_GPT_max_sqrt_distance_l2052_205239

theorem max_sqrt_distance (x y : ℝ) 
  (h : x^2 + y^2 - 4 * x - 4 * y + 6 = 0) : 
  ∃ z, z = 3 * Real.sqrt 2 ∧ ∀ w, w = Real.sqrt (x^2 + y^2) → w ≤ z :=
sorry

end NUMINAMATH_GPT_max_sqrt_distance_l2052_205239


namespace NUMINAMATH_GPT_tangency_condition_l2052_205222

-- Define the equation for the ellipse
def ellipse_eq (x y : ℝ) : Prop :=
  3 * x^2 + 9 * y^2 = 9

-- Define the equation for the hyperbola
def hyperbola_eq (x y m : ℝ) : Prop :=
  (x - 2)^2 - m * (y + 1)^2 = 1

-- Prove that for the ellipse and hyperbola to be tangent, m must equal 3
theorem tangency_condition (m : ℝ) :
  (∀ x y : ℝ, ellipse_eq x y ∧ hyperbola_eq x y m) → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_tangency_condition_l2052_205222


namespace NUMINAMATH_GPT_meeting_lamppost_l2052_205208

-- Define the initial conditions of the problem
def lampposts : ℕ := 400
def start_alla : ℕ := 1
def start_boris : ℕ := 400
def meet_alla : ℕ := 55
def meet_boris : ℕ := 321

-- Define a theorem that we need to prove: Alla and Boris will meet at the 163rd lamppost
theorem meeting_lamppost : ∃ (n : ℕ), n = 163 := 
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_meeting_lamppost_l2052_205208


namespace NUMINAMATH_GPT_boy_usual_time_reach_school_l2052_205280

theorem boy_usual_time_reach_school (R T : ℝ) (h : (7 / 6) * R * (T - 3) = R * T) : T = 21 := by
  sorry

end NUMINAMATH_GPT_boy_usual_time_reach_school_l2052_205280


namespace NUMINAMATH_GPT_no_integer_solutions_l2052_205255

theorem no_integer_solutions (w l : ℕ) (hw_pos : 0 < w) (hl_pos : 0 < l) : 
  (w * l = 24 ∧ (w = l ∨ 2 * l = w)) → false :=
by 
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l2052_205255


namespace NUMINAMATH_GPT_range_of_r_l2052_205259

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}

def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

theorem range_of_r (r : ℝ) (hr: 0 < r) : (M ∩ N r = N r) → r ≤ 2 - Real.sqrt 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_r_l2052_205259


namespace NUMINAMATH_GPT_number_of_integer_values_l2052_205220

def Q (x : ℤ) : ℤ := x^4 + 4 * x^3 + 9 * x^2 + 2 * x + 17

theorem number_of_integer_values :
  (∃ xs : List ℤ, xs.length = 4 ∧ ∀ x ∈ xs, Nat.Prime (Int.natAbs (Q x))) :=
by
  sorry

end NUMINAMATH_GPT_number_of_integer_values_l2052_205220


namespace NUMINAMATH_GPT_frank_initial_money_l2052_205242

theorem frank_initial_money (X : ℝ) (h1 : X * (4 / 5) * (3 / 4) * (6 / 7) * (2 / 3) = 600) : X = 2333.33 :=
sorry

end NUMINAMATH_GPT_frank_initial_money_l2052_205242


namespace NUMINAMATH_GPT_sampling_probabilities_equal_l2052_205253

noncomputable def populationSize (N : ℕ) := N
noncomputable def sampleSize (n : ℕ) := n

def P1 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)
def P2 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)
def P3 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)

theorem sampling_probabilities_equal (N n : ℕ) (hN : N > 0) (hn : n > 0) :
  P1 N n = P2 N n ∧ P2 N n = P3 N n :=
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_sampling_probabilities_equal_l2052_205253


namespace NUMINAMATH_GPT_not_diff_of_squares_2022_l2052_205212

theorem not_diff_of_squares_2022 :
  ¬ ∃ a b : ℤ, a^2 - b^2 = 2022 :=
by
  sorry

end NUMINAMATH_GPT_not_diff_of_squares_2022_l2052_205212


namespace NUMINAMATH_GPT_sarah_wide_reflections_l2052_205205

variables (tall_mirrors_sarah : ℕ) (tall_mirrors_ellie : ℕ) 
          (wide_mirrors_ellie : ℕ) (tall_count : ℕ) (wide_count : ℕ)
          (total_reflections : ℕ) (S : ℕ)

def reflections_in_tall_mirrors_sarah := 10 * tall_count
def reflections_in_tall_mirrors_ellie := 6 * tall_count
def reflections_in_wide_mirrors_ellie := 3 * wide_count
def total_reflections_no_wide_sarah := reflections_in_tall_mirrors_sarah + reflections_in_tall_mirrors_ellie + reflections_in_wide_mirrors_ellie

theorem sarah_wide_reflections :
  reflections_in_tall_mirrors_sarah = 30 →
  reflections_in_tall_mirrors_ellie = 18 →
  reflections_in_wide_mirrors_ellie = 15 →
  tall_count = 3 →
  wide_count = 5 →
  total_reflections = 88 →
  total_reflections = total_reflections_no_wide_sarah + 5 * S →
  S = 5 :=
sorry

end NUMINAMATH_GPT_sarah_wide_reflections_l2052_205205


namespace NUMINAMATH_GPT_first_pipe_time_l2052_205296

noncomputable def pool_filling_time (T : ℝ) : Prop :=
  (1 / T + 1 / 12 = 1 / 4.8) → (T = 8)

theorem first_pipe_time :
  ∃ T : ℝ, pool_filling_time T := by
  use 8
  sorry

end NUMINAMATH_GPT_first_pipe_time_l2052_205296


namespace NUMINAMATH_GPT_kayla_apples_correct_l2052_205229

-- Definition of Kylie and Kayla's apples
def total_apples : ℕ := 340
def kaylas_apples (k : ℕ) : ℕ := 4 * k + 10

-- The main statement to prove
theorem kayla_apples_correct :
  ∃ K : ℕ, K + kaylas_apples K = total_apples ∧ kaylas_apples K = 274 :=
sorry

end NUMINAMATH_GPT_kayla_apples_correct_l2052_205229


namespace NUMINAMATH_GPT_rectangle_area_l2052_205273

theorem rectangle_area (a b c: ℝ) (h₁ : a = 7.1) (h₂ : b = 8.9) (h₃ : c = 10.0) (L W: ℝ)
  (h₄ : L = 2 * W) (h₅ : 2 * (L + W) = a + b + c) : L * W = 37.54 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2052_205273


namespace NUMINAMATH_GPT_games_planned_to_attend_this_month_l2052_205274

theorem games_planned_to_attend_this_month (T A_l P_l M_l P_m : ℕ) 
  (h1 : T = 12) 
  (h2 : P_l = 17) 
  (h3 : M_l = 16) 
  (h4 : A_l = P_l - M_l) 
  (h5 : T = A_l + P_m) : P_m = 11 :=
by 
  sorry

end NUMINAMATH_GPT_games_planned_to_attend_this_month_l2052_205274


namespace NUMINAMATH_GPT_inequality_ge_one_l2052_205271

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end NUMINAMATH_GPT_inequality_ge_one_l2052_205271


namespace NUMINAMATH_GPT_exponential_function_passes_through_01_l2052_205294

theorem exponential_function_passes_through_01 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : (a^0 = 1) :=
by
  sorry

end NUMINAMATH_GPT_exponential_function_passes_through_01_l2052_205294


namespace NUMINAMATH_GPT_joshua_total_payment_is_correct_l2052_205249

noncomputable def total_cost : ℝ := 
  let t_shirt_price := 8
  let sweater_price := 18
  let jacket_price := 80
  let jeans_price := 35
  let shoes_price := 60
  let jacket_discount := 0.10
  let shoes_discount := 0.15
  let clothing_tax_rate := 0.05
  let shoes_tax_rate := 0.08

  let t_shirt_count := 6
  let sweater_count := 4
  let jacket_count := 5
  let jeans_count := 3
  let shoes_count := 2

  let t_shirts_subtotal := t_shirt_price * t_shirt_count
  let sweaters_subtotal := sweater_price * sweater_count
  let jackets_subtotal := jacket_price * jacket_count
  let jeans_subtotal := jeans_price * jeans_count
  let shoes_subtotal := shoes_price * shoes_count

  let jackets_discounted := jackets_subtotal * (1 - jacket_discount)
  let shoes_discounted := shoes_subtotal * (1 - shoes_discount)

  let total_before_tax := t_shirts_subtotal + sweaters_subtotal + jackets_discounted + jeans_subtotal + shoes_discounted

  let t_shirts_tax := t_shirts_subtotal * clothing_tax_rate
  let sweaters_tax := sweaters_subtotal * clothing_tax_rate
  let jackets_tax := jackets_discounted * clothing_tax_rate
  let jeans_tax := jeans_subtotal * clothing_tax_rate
  let shoes_tax := shoes_discounted * shoes_tax_rate

  total_before_tax + t_shirts_tax + sweaters_tax + jackets_tax + jeans_tax + shoes_tax

theorem joshua_total_payment_is_correct : total_cost = 724.41 := by
  sorry

end NUMINAMATH_GPT_joshua_total_payment_is_correct_l2052_205249


namespace NUMINAMATH_GPT_train_speed_in_kmh_l2052_205285

-- Definitions of conditions
def time_to_cross_platform := 30  -- in seconds
def time_to_cross_man := 17  -- in seconds
def length_of_platform := 260  -- in meters

-- Conversion factor from m/s to km/h
def meters_per_second_to_kilometers_per_hour (v : ℕ) : ℕ :=
  v * 36 / 10

-- The theorem statement
theorem train_speed_in_kmh :
  (∃ (L V : ℕ),
    L = V * time_to_cross_man ∧
    L + length_of_platform = V * time_to_cross_platform ∧
    meters_per_second_to_kilometers_per_hour V = 72) :=
sorry

end NUMINAMATH_GPT_train_speed_in_kmh_l2052_205285


namespace NUMINAMATH_GPT_ellipse_foci_coordinates_l2052_205224

theorem ellipse_foci_coordinates :
  (∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) → ∃ c : ℝ, (c = 4) ∧ (x = c ∨ x = -c) ∧ (y = 0)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_coordinates_l2052_205224


namespace NUMINAMATH_GPT_base_six_to_base_ten_equivalent_l2052_205279

theorem base_six_to_base_ten_equivalent :
  let n := 12345
  (5 * 6^0 + 4 * 6^1 + 3 * 6^2 + 2 * 6^3 + 1 * 6^4) = 1865 :=
by
  sorry

end NUMINAMATH_GPT_base_six_to_base_ten_equivalent_l2052_205279


namespace NUMINAMATH_GPT_min_solutions_f_eq_zero_l2052_205244

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 3) = f x)
variable (h_zero_at_2 : f 2 = 0)

theorem min_solutions_f_eq_zero : ∃ S : Finset ℝ, (∀ x ∈ S, f x = 0) ∧ 7 ≤ S.card ∧ (∀ x ∈ S, x > 0 ∧ x < 6) := 
sorry

end NUMINAMATH_GPT_min_solutions_f_eq_zero_l2052_205244


namespace NUMINAMATH_GPT_total_carrots_l2052_205291

theorem total_carrots (sandy_carrots: Nat) (sam_carrots: Nat) (h1: sandy_carrots = 6) (h2: sam_carrots = 3) : sandy_carrots + sam_carrots = 9 :=
by
  sorry

end NUMINAMATH_GPT_total_carrots_l2052_205291


namespace NUMINAMATH_GPT_positive_solution_in_interval_l2052_205266

def quadratic (x : ℝ) := x^2 + 3 * x - 5

theorem positive_solution_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ quadratic x = 0 :=
sorry

end NUMINAMATH_GPT_positive_solution_in_interval_l2052_205266


namespace NUMINAMATH_GPT_problem_statement_l2052_205287

theorem problem_statement :
  ∀ x a k n : ℤ, 
  (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + n → a - n + k = 3 :=
by  
  sorry

end NUMINAMATH_GPT_problem_statement_l2052_205287


namespace NUMINAMATH_GPT_price_of_first_candy_l2052_205243

theorem price_of_first_candy (P: ℝ) 
  (total_weight: ℝ) (price_per_lb_mixture: ℝ) 
  (weight_first: ℝ) (weight_second: ℝ) 
  (price_per_lb_second: ℝ) :
  total_weight = 30 →
  price_per_lb_mixture = 3 →
  weight_first = 20 →
  weight_second = 10 →
  price_per_lb_second = 3.1 →
  20 * P + 10 * price_per_lb_second = total_weight * price_per_lb_mixture →
  P = 2.95 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_price_of_first_candy_l2052_205243


namespace NUMINAMATH_GPT_no_a_satisfies_condition_l2052_205298

noncomputable def M : Set ℝ := {0, 1}
noncomputable def N (a : ℝ) : Set ℝ := {11 - a, Real.log a / Real.log 1, 2^a, a}

theorem no_a_satisfies_condition :
  ¬ ∃ a : ℝ, M ∩ N a = {1} :=
by
  sorry

end NUMINAMATH_GPT_no_a_satisfies_condition_l2052_205298


namespace NUMINAMATH_GPT_weighted_mean_calculation_l2052_205254

/-- Prove the weighted mean of the numbers 16, 28, and 45 with weights 2, 3, and 5 is 34.1 -/
theorem weighted_mean_calculation :
  let numbers := [16, 28, 45]
  let weights := [2, 3, 5]
  let total_weight := (2 + 3 + 5 : ℝ)
  let weighted_sum := ((16 * 2 + 28 * 3 + 45 * 5) : ℝ)
  (weighted_sum / total_weight) = 34.1 :=
by
  -- We only state the theorem without providing the proof
  sorry

end NUMINAMATH_GPT_weighted_mean_calculation_l2052_205254


namespace NUMINAMATH_GPT_total_guppies_correct_l2052_205218

-- Define the initial conditions as variables
def initial_guppies : ℕ := 7
def baby_guppies_1 : ℕ := 3 * 12
def baby_guppies_2 : ℕ := 9

-- Define the total number of guppies
def total_guppies : ℕ := initial_guppies + baby_guppies_1 + baby_guppies_2

-- Theorem: Proving the total number of guppies is 52
theorem total_guppies_correct : total_guppies = 52 :=
by
  sorry

end NUMINAMATH_GPT_total_guppies_correct_l2052_205218


namespace NUMINAMATH_GPT_divisible_by_12_l2052_205211

theorem divisible_by_12 (a b c d : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (hpos_d : 0 < d) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) := 
by
  sorry

end NUMINAMATH_GPT_divisible_by_12_l2052_205211


namespace NUMINAMATH_GPT_tom_savings_l2052_205258

theorem tom_savings :
  let insurance_cost_per_month := 20
  let total_months := 24
  let procedure_cost := 5000
  let insurance_coverage := 0.80
  let total_insurance_cost := total_months * insurance_cost_per_month
  let insurance_cover_amount := procedure_cost * insurance_coverage
  let out_of_pocket_cost := procedure_cost - insurance_cover_amount
  let savings := procedure_cost - total_insurance_cost - out_of_pocket_cost
  savings = 3520 :=
by
  sorry

end NUMINAMATH_GPT_tom_savings_l2052_205258


namespace NUMINAMATH_GPT_Mabel_gave_away_daisies_l2052_205213

-- Setting up the conditions
variables (d_total : ℕ) (p_per_daisy : ℕ) (p_remaining : ℕ)

-- stating the assumptions
def initial_petals (d_total p_per_daisy : ℕ) := d_total * p_per_daisy
def petals_given_away (d_total p_per_daisy p_remaining : ℕ) := initial_petals d_total p_per_daisy - p_remaining
def daisies_given_away (d_total p_per_daisy p_remaining : ℕ) := petals_given_away d_total p_per_daisy p_remaining / p_per_daisy

-- The main theorem
theorem Mabel_gave_away_daisies 
  (h1 : d_total = 5)
  (h2 : p_per_daisy = 8)
  (h3 : p_remaining = 24) :
  daisies_given_away d_total p_per_daisy p_remaining = 2 :=
sorry

end NUMINAMATH_GPT_Mabel_gave_away_daisies_l2052_205213


namespace NUMINAMATH_GPT_min_value_l2052_205233

theorem min_value (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_sum : a + b = 1) : 
  ∃ x : ℝ, (x = 25) ∧ x ≤ (4 / a + 9 / b) :=
by
  sorry

end NUMINAMATH_GPT_min_value_l2052_205233


namespace NUMINAMATH_GPT_largest_possible_median_l2052_205293

theorem largest_possible_median (l : List ℕ) (h1 : l.length = 10) 
  (h2 : ∀ x ∈ l, 0 < x) (exists6l : ∃ l1 : List ℕ, l1 = [3, 4, 5, 7, 8, 9]) :
  ∃ median_val : ℝ, median_val = 8.5 := 
sorry

end NUMINAMATH_GPT_largest_possible_median_l2052_205293


namespace NUMINAMATH_GPT_value_of_product_l2052_205262

theorem value_of_product : (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049 = 243) :=
by sorry

end NUMINAMATH_GPT_value_of_product_l2052_205262


namespace NUMINAMATH_GPT_base2_to_base4_conversion_l2052_205223

/-- Definition of base conversion from binary to quaternary. -/
def bin_to_quat (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  if n = 1 then 1 else
  if n = 10 then 2 else
  if n = 11 then 3 else
  0 -- (more cases can be added as necessary)

theorem base2_to_base4_conversion :
  bin_to_quat 1 * 4^4 + bin_to_quat 1 * 4^3 + bin_to_quat 10 * 4^2 + bin_to_quat 11 * 4^1 + bin_to_quat 10 * 4^0 = 11232 :=
by sorry

end NUMINAMATH_GPT_base2_to_base4_conversion_l2052_205223
