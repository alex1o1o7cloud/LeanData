import Mathlib

namespace NUMINAMATH_GPT_frac_sum_property_l1199_119932

theorem frac_sum_property (a b : ℕ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end NUMINAMATH_GPT_frac_sum_property_l1199_119932


namespace NUMINAMATH_GPT_find_larger_number_l1199_119994

theorem find_larger_number (x y : ℕ) (h1 : y - x = 1500) (h2 : y = 6 * x + 15) : y = 1797 := by
  sorry

end NUMINAMATH_GPT_find_larger_number_l1199_119994


namespace NUMINAMATH_GPT_y_directly_varies_as_square_l1199_119993

theorem y_directly_varies_as_square (k : ℚ) (y : ℚ) (x : ℚ) 
  (h1 : y = k * x ^ 2) (h2 : y = 18) (h3 : x = 3) : 
  ∃ y : ℚ, ∀ x : ℚ, x = 6 → y = 72 :=
by
  sorry

end NUMINAMATH_GPT_y_directly_varies_as_square_l1199_119993


namespace NUMINAMATH_GPT_taxi_distance_l1199_119945

variable (initial_fee charge_per_2_5_mile total_charge : ℝ)
variable (d : ℝ)

theorem taxi_distance 
  (h_initial_fee : initial_fee = 2.35)
  (h_charge_per_2_5_mile : charge_per_2_5_mile = 0.35)
  (h_total_charge : total_charge = 5.50)
  (h_eq : total_charge = initial_fee + (charge_per_2_5_mile / (2/5)) * d) :
  d = 3.6 :=
sorry

end NUMINAMATH_GPT_taxi_distance_l1199_119945


namespace NUMINAMATH_GPT_one_cow_one_bag_in_forty_days_l1199_119907

theorem one_cow_one_bag_in_forty_days
    (total_cows : ℕ)
    (total_bags : ℕ)
    (total_days : ℕ)
    (husk_consumption : total_cows * total_bags = total_cows * total_days) :
  total_days = 40 :=
by sorry

end NUMINAMATH_GPT_one_cow_one_bag_in_forty_days_l1199_119907


namespace NUMINAMATH_GPT_number_of_games_can_buy_l1199_119950

-- Definitions based on the conditions
def initial_money : ℕ := 42
def spent_money : ℕ := 10
def game_cost : ℕ := 8

-- The statement we need to prove: Mike can buy 4 games given the conditions
theorem number_of_games_can_buy : (initial_money - spent_money) / game_cost = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_games_can_buy_l1199_119950


namespace NUMINAMATH_GPT_fractional_exponent_equality_l1199_119990

theorem fractional_exponent_equality :
  (3 / 4 : ℚ) ^ 2017 * (- ((1:ℚ) + 1 / 3)) ^ 2018 = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fractional_exponent_equality_l1199_119990


namespace NUMINAMATH_GPT_range_of_a_no_solution_inequality_l1199_119973

theorem range_of_a_no_solution_inequality (a : ℝ) :
  (∀ x : ℝ, x + 2 > 3 → x < a) ↔ a ≤ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_no_solution_inequality_l1199_119973


namespace NUMINAMATH_GPT_find_missing_value_l1199_119927

theorem find_missing_value :
  300 * 2 + (12 + 4) * 1 / 8 = 602 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_value_l1199_119927


namespace NUMINAMATH_GPT_purely_imaginary_complex_iff_l1199_119923

theorem purely_imaginary_complex_iff (m : ℝ) :
  (m + 2 = 0) → (m = -2) :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_complex_iff_l1199_119923


namespace NUMINAMATH_GPT_no_solution_l1199_119933

theorem no_solution (a : ℝ) :
  (a < -12 ∨ a > 0) →
  ∀ x : ℝ, ¬(6 * (|x - 4 * a|) + (|x - a ^ 2|) + 5 * x - 4 * a = 0) :=
by
  intros ha hx
  sorry

end NUMINAMATH_GPT_no_solution_l1199_119933


namespace NUMINAMATH_GPT_range_of_f_l1199_119979

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem range_of_f :
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), (f x) ∈ Set.Icc (-1 : ℝ) 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1199_119979


namespace NUMINAMATH_GPT_equivar_proof_l1199_119978

variable {x : ℝ} {m : ℝ}

def p (m : ℝ) : Prop := m > 2

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem equivar_proof (m : ℝ) (h : ¬p m ∧ q m) : 1 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_GPT_equivar_proof_l1199_119978


namespace NUMINAMATH_GPT_good_horse_catchup_l1199_119988

theorem good_horse_catchup 
  (x : ℕ) 
  (good_horse_speed : ℕ) (slow_horse_speed : ℕ) (head_start_days : ℕ) 
  (H1 : good_horse_speed = 240)
  (H2 : slow_horse_speed = 150)
  (H3 : head_start_days = 12) :
  good_horse_speed * x - slow_horse_speed * x = slow_horse_speed * head_start_days :=
by
  sorry

end NUMINAMATH_GPT_good_horse_catchup_l1199_119988


namespace NUMINAMATH_GPT_overall_average_runs_l1199_119942

theorem overall_average_runs 
  (test_matches: ℕ) (test_avg: ℕ) 
  (odi_matches: ℕ) (odi_avg: ℕ) 
  (t20_matches: ℕ) (t20_avg: ℕ)
  (h_test_matches: test_matches = 25)
  (h_test_avg: test_avg = 48)
  (h_odi_matches: odi_matches = 20)
  (h_odi_avg: odi_avg = 38)
  (h_t20_matches: t20_matches = 15)
  (h_t20_avg: t20_avg = 28) :
  (25 * 48 + 20 * 38 + 15 * 28) / (25 + 20 + 15) = 39.67 :=
sorry

end NUMINAMATH_GPT_overall_average_runs_l1199_119942


namespace NUMINAMATH_GPT_min_distance_value_l1199_119953

theorem min_distance_value (x1 x2 y1 y2 : ℝ) 
  (h1 : (e ^ x1 + 2 * x1) / (3 * y1) = 1 / 3)
  (h2 : (x2 - 1) / y2 = 1 / 3) :
  ((x1 - x2)^2 + (y1 - y2)^2) = 8 / 5 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_value_l1199_119953


namespace NUMINAMATH_GPT_min_value_four_l1199_119941

noncomputable def min_value_T (a b c : ℝ) : ℝ :=
  1 / (2 * (a * b - 1)) + a * (b + 2 * c) / (a * b - 1)

theorem min_value_four (a b c : ℝ) (h1 : (1 / a) > 0)
  (h2 : b^2 - (4 * c) / a ≤ 0) (h3 : a * b > 1) : 
  min_value_T a b c = 4 := 
by 
  sorry

end NUMINAMATH_GPT_min_value_four_l1199_119941


namespace NUMINAMATH_GPT_mode_of_dataSet_is_3_l1199_119970

-- Define the data set
def dataSet : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

-- Define what it means to be the mode of a list
def is_mode (l : List ℕ) (n : ℕ) : Prop :=
  ∀ m, l.count n ≥ l.count m

-- Prove the mode of the data set
theorem mode_of_dataSet_is_3 : is_mode dataSet 3 :=
by
  sorry

end NUMINAMATH_GPT_mode_of_dataSet_is_3_l1199_119970


namespace NUMINAMATH_GPT_evaluate_expression_l1199_119966

theorem evaluate_expression (x b : ℝ) (h : x = b + 4) : 2 * x - b + 5 = b + 13 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1199_119966


namespace NUMINAMATH_GPT_elizabeth_stickers_count_l1199_119931

theorem elizabeth_stickers_count :
  let initial_bottles := 10
  let lost_at_school := 2
  let stolen_at_dance := 1
  let stickers_per_bottle := 3
  let remaining_bottles := initial_bottles - lost_at_school - stolen_at_dance
  remaining_bottles * stickers_per_bottle = 21 := by sorry

end NUMINAMATH_GPT_elizabeth_stickers_count_l1199_119931


namespace NUMINAMATH_GPT_divisible_check_l1199_119919

theorem divisible_check (n : ℕ) (h : n = 287) : 
  ¬ (n % 3 = 0) ∧  ¬ (n % 4 = 0) ∧  ¬ (n % 5 = 0) ∧ ¬ (n % 6 = 0) ∧ (n % 7 = 0) := 
by {
  sorry
}

end NUMINAMATH_GPT_divisible_check_l1199_119919


namespace NUMINAMATH_GPT_range_of_m_l1199_119943

-- Definition of p: x / (x - 2) < 0 implies 0 < x < 2
def p (x : ℝ) : Prop := x / (x - 2) < 0

-- Definition of q: 0 < x < m
def q (x m : ℝ) : Prop := 0 < x ∧ x < m

-- Main theorem: If p is a necessary but not sufficient condition for q to hold, then the range of m is (2, +∞)
theorem range_of_m {m : ℝ} (h : ∀ x, p x → q x m) (hs : ∃ x, ¬(q x m) ∧ p x) : 
  2 < m :=
sorry

end NUMINAMATH_GPT_range_of_m_l1199_119943


namespace NUMINAMATH_GPT_period2_students_is_8_l1199_119983

-- Definitions according to conditions
def period1_students : Nat := 11
def relationship (x : Nat) := 2 * x - 5

-- Lean 4 statement
theorem period2_students_is_8 (x: Nat) (h: relationship x = period1_students) : x = 8 := 
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_period2_students_is_8_l1199_119983


namespace NUMINAMATH_GPT_consecutive_odd_sum_count_l1199_119940

theorem consecutive_odd_sum_count (N : ℕ) :
  N = 20 ↔ (
    ∃ (ns : Finset ℕ), ∃ (js : Finset ℕ),
      (∀ n ∈ ns, n < 500) ∧
      (∀ j ∈ js, j ≥ 2) ∧
      ∀ n ∈ ns, ∃ j ∈ js, ∃ k, k = 3 ∧ N = j * (2 * k + j)
  ) :=
by
  sorry

end NUMINAMATH_GPT_consecutive_odd_sum_count_l1199_119940


namespace NUMINAMATH_GPT_min_value_condition_solve_inequality_l1199_119947

open Real

-- Define the function f(x) = |x - a| + |x + 2|
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 2)

-- Part I: Proving the values of a for f(x) having minimum value of 2
theorem min_value_condition (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 2) → (∃ x : ℝ, f x a = 2) → (a = 0 ∨ a = -4) :=
by
  sorry

-- Part II: Solving inequality f(x) ≤ 6 when a = 2
theorem solve_inequality : 
  ∀ x : ℝ, f x 2 ≤ 6 ↔ (x ≥ -3 ∧ x ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_min_value_condition_solve_inequality_l1199_119947


namespace NUMINAMATH_GPT_count_perfect_squares_l1199_119965

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def E1 : ℕ := 1^3 + 2^3
def E2 : ℕ := 1^3 + 2^3 + 3^3
def E3 : ℕ := 1^3 + 2^3 + 3^3 + 4^3
def E4 : ℕ := 1^3 + 2^3 + 3^3 + 4^3 + 5^3

theorem count_perfect_squares :
  (is_perfect_square E1 → true) ∧
  (is_perfect_square E2 → true) ∧
  (is_perfect_square E3 → true) ∧
  (is_perfect_square E4 → true) →
  (∀ n : ℕ, (n = 4) ↔
    ∃ E1 E2 E3 E4, is_perfect_square E1 ∧ is_perfect_square E2 ∧ is_perfect_square E3 ∧ is_perfect_square E4) :=
by
  sorry

end NUMINAMATH_GPT_count_perfect_squares_l1199_119965


namespace NUMINAMATH_GPT_total_balloons_l1199_119922

def tom_balloons : Nat := 9
def sara_balloons : Nat := 8

theorem total_balloons : tom_balloons + sara_balloons = 17 := 
by
  sorry

end NUMINAMATH_GPT_total_balloons_l1199_119922


namespace NUMINAMATH_GPT_odd_function_increasing_function_l1199_119905

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem odd_function (x : ℝ) : 
  (f (1 / 2) (-x)) = -(f (1 / 2) x) := 
by
  sorry

theorem increasing_function : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f (1 / 2) x₁ < f (1 / 2) x₂ := 
by
  sorry

end NUMINAMATH_GPT_odd_function_increasing_function_l1199_119905


namespace NUMINAMATH_GPT_stratified_sampling_male_athletes_l1199_119967

theorem stratified_sampling_male_athletes (total_males : ℕ) (total_females : ℕ) (sample_size : ℕ)
  (total_population : ℕ) (male_sample_fraction : ℚ) (n_sample_males : ℕ) :
  total_males = 56 →
  total_females = 42 →
  sample_size = 28 →
  total_population = total_males + total_females →
  male_sample_fraction = (sample_size : ℚ) / (total_population : ℚ) →
  n_sample_males = (total_males : ℚ) * male_sample_fraction →
  n_sample_males = 16 := by
  intros h_males h_females h_samples h_population h_fraction h_final
  sorry

end NUMINAMATH_GPT_stratified_sampling_male_athletes_l1199_119967


namespace NUMINAMATH_GPT_suitable_for_census_l1199_119908

-- Define types for each survey option.
inductive SurveyOption where
  | A : SurveyOption -- Understanding the vision of middle school students in our province
  | B : SurveyOption -- Investigating the viewership of "The Reader"
  | C : SurveyOption -- Inspecting the components of a newly developed fighter jet to ensure successful test flights
  | D : SurveyOption -- Testing the lifespan of a batch of light bulbs

-- Theorem statement asserting that Option C is the suitable one for a census.
theorem suitable_for_census : SurveyOption.C = SurveyOption.C :=
by
  exact rfl

end NUMINAMATH_GPT_suitable_for_census_l1199_119908


namespace NUMINAMATH_GPT_shortest_distance_l1199_119975

theorem shortest_distance 
  (C : ℝ × ℝ) (B : ℝ × ℝ) (stream : ℝ)
  (hC : C = (0, -3))
  (hB : B = (9, -8))
  (hStream : stream = 0) :
  ∃ d : ℝ, d = 3 + Real.sqrt 202 :=
by
  sorry

end NUMINAMATH_GPT_shortest_distance_l1199_119975


namespace NUMINAMATH_GPT_rearrangement_count_is_two_l1199_119995

def is_adjacent (c1 c2 : Char) : Bool :=
  (c1 = 'a' ∧ c2 = 'b') ∨
  (c1 = 'b' ∧ c2 = 'c') ∨
  (c1 = 'c' ∧ c2 = 'd') ∨
  (c1 = 'd' ∧ c2 = 'e') ∨
  (c1 = 'b' ∧ c2 = 'a') ∨
  (c1 = 'c' ∧ c2 = 'b') ∨
  (c1 = 'd' ∧ c2 = 'c') ∨
  (c1 = 'e' ∧ c2 = 'd')

def no_adjacent_letters (s : List Char) : Bool :=
  match s with
  | [] => true
  | [_] => true
  | c1 :: c2 :: cs => 
    ¬ is_adjacent c1 c2 ∧ no_adjacent_letters (c2 :: cs)

def valid_rearrangements_count : Nat :=
  let perms := List.permutations ['a', 'b', 'c', 'd', 'e']
  perms.filter no_adjacent_letters |>.length

theorem rearrangement_count_is_two :
  valid_rearrangements_count = 2 :=
by sorry

end NUMINAMATH_GPT_rearrangement_count_is_two_l1199_119995


namespace NUMINAMATH_GPT_JiaZi_second_column_l1199_119991

theorem JiaZi_second_column :
  let heavenlyStemsCycle := 10
  let earthlyBranchesCycle := 12
  let firstOccurrence := 1
  let lcmCycle := Nat.lcm heavenlyStemsCycle earthlyBranchesCycle
  let secondOccurrence := firstOccurrence + lcmCycle
  secondOccurrence = 61 :=
by
  sorry

end NUMINAMATH_GPT_JiaZi_second_column_l1199_119991


namespace NUMINAMATH_GPT_first_tray_holds_260_cups_l1199_119921

variable (x : ℕ)

def first_tray_holds_x_cups (tray1 : ℕ) := tray1 = x
def second_tray_holds_x_minus_20_cups (tray2 : ℕ) := tray2 = x - 20
def total_cups_in_both_trays (tray1 tray2: ℕ) := tray1 + tray2 = 500

theorem first_tray_holds_260_cups (tray1 tray2 : ℕ) :
  first_tray_holds_x_cups x tray1 →
  second_tray_holds_x_minus_20_cups x tray2 →
  total_cups_in_both_trays tray1 tray2 →
  x = 260 := by
  sorry

end NUMINAMATH_GPT_first_tray_holds_260_cups_l1199_119921


namespace NUMINAMATH_GPT_independence_test_purpose_l1199_119904

theorem independence_test_purpose:
  ∀ (test: String), test = "independence test" → 
  ∀ (purpose: String), purpose = "to provide the reliability of the relationship between two categorical variables" →
  (test = "independence test" ∧ purpose = "to provide the reliability of the relationship between two categorical variables") :=
by
  intros test h_test purpose h_purpose
  exact ⟨h_test, h_purpose⟩

end NUMINAMATH_GPT_independence_test_purpose_l1199_119904


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1199_119998

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1199_119998


namespace NUMINAMATH_GPT_statement_A_statement_B_statement_C_l1199_119969

variable {α : Type}

-- Conditions for statement A
def angle_greater (A B : ℝ) : Prop := A > B
def sin_greater (A B : ℝ) : Prop := Real.sin A > Real.sin B

-- Conditions for statement B
def acute_triangle (A B C : ℝ) : Prop := A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2
def sin_greater_than_cos (A B : ℝ) : Prop := Real.sin A > Real.cos B

-- Conditions for statement C
def obtuse_triangle (C : ℝ) : Prop := C > Real.pi / 2

-- Statement A in Lean
theorem statement_A (A B : ℝ) : angle_greater A B → sin_greater A B :=
sorry

-- Statement B in Lean
theorem statement_B {A B C : ℝ} : acute_triangle A B C → sin_greater_than_cos A B :=
sorry

-- Statement C in Lean
theorem statement_C {a b c : ℝ} (h : a^2 + b^2 < c^2) : obtuse_triangle C :=
sorry

-- Statement D in Lean (proof not needed as it's incorrect)
-- Theorem is omitted since statement D is incorrect

end NUMINAMATH_GPT_statement_A_statement_B_statement_C_l1199_119969


namespace NUMINAMATH_GPT_binomial_expansion_fraction_l1199_119949

theorem binomial_expansion_fraction 
    (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
    (h1 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 1)
    (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = 243) :
    (a_0 + a_2 + a_4) / (a_1 + a_3 + a_5) = -122 / 121 :=
by
  sorry

end NUMINAMATH_GPT_binomial_expansion_fraction_l1199_119949


namespace NUMINAMATH_GPT_find_k_l1199_119955

theorem find_k {k : ℚ} :
    (∃ x y : ℚ, y = 3 * x + 6 ∧ y = -4 * x - 20 ∧ y = 2 * x + k) →
    k = 16 / 7 := 
  sorry

end NUMINAMATH_GPT_find_k_l1199_119955


namespace NUMINAMATH_GPT_binom_12_9_plus_binom_12_3_l1199_119924

theorem binom_12_9_plus_binom_12_3 : (Nat.choose 12 9) + (Nat.choose 12 3) = 440 := by
  sorry

end NUMINAMATH_GPT_binom_12_9_plus_binom_12_3_l1199_119924


namespace NUMINAMATH_GPT_set_intersection_problem_l1199_119992

def set_product (A B : Set ℕ) : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}
def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 3}
def C : Set ℕ := {x | x^2 - 3 * x + 2 = 0}

theorem set_intersection_problem :
  (set_product A B) ∩ (set_product B C) = {2, 6} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_problem_l1199_119992


namespace NUMINAMATH_GPT_cara_bread_dinner_amount_240_l1199_119936

def conditions (B L D : ℕ) : Prop :=
  8 * L = D ∧ 6 * B = D ∧ B + L + D = 310

theorem cara_bread_dinner_amount_240 :
  ∃ (B L D : ℕ), conditions B L D ∧ D = 240 :=
by
  sorry

end NUMINAMATH_GPT_cara_bread_dinner_amount_240_l1199_119936


namespace NUMINAMATH_GPT_factorable_polynomial_l1199_119956

theorem factorable_polynomial (n : ℤ) :
  ∃ (a b c d e f : ℤ), 
    (a = 1) ∧ (d = 1) ∧ 
    (b + e = 2) ∧ 
    (f = b * e) ∧ 
    (c + f + b * e = 2) ∧ 
    (c * f + b * e = -n^2) ↔ 
    (n = 0 ∨ n = 2 ∨ n = -2) :=
by
  sorry

end NUMINAMATH_GPT_factorable_polynomial_l1199_119956


namespace NUMINAMATH_GPT_segment_ratios_l1199_119957

theorem segment_ratios 
  (AB_parts BC_parts : ℝ) 
  (hAB: AB_parts = 3) 
  (hBC: BC_parts = 4) 
  : AB_parts / (AB_parts + BC_parts) = 3 / 7 ∧ BC_parts / (AB_parts + BC_parts) = 4 / 7 := 
  sorry

end NUMINAMATH_GPT_segment_ratios_l1199_119957


namespace NUMINAMATH_GPT_complete_the_square_transforms_l1199_119997

theorem complete_the_square_transforms (x : ℝ) :
  (x^2 + 8 * x + 7 = 0) → ((x + 4) ^ 2 = 9) :=
by
  intro h
  have step1 : x^2 + 8 * x = -7 := by sorry
  have step2 : x^2 + 8 * x + 16 = -7 + 16 := by sorry
  have step3 : (x + 4) ^ 2 = 9 := by sorry
  exact step3

end NUMINAMATH_GPT_complete_the_square_transforms_l1199_119997


namespace NUMINAMATH_GPT_tan_sum_trig_identity_l1199_119963

variable {α : ℝ}

-- Part (I)
theorem tan_sum (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
by
  sorry

-- Part (II)
theorem trig_identity (h : Real.tan α = 2) : 
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_trig_identity_l1199_119963


namespace NUMINAMATH_GPT_smallest_b_gt_4_perfect_square_l1199_119937

theorem smallest_b_gt_4_perfect_square :
  ∃ b : ℕ, b > 4 ∧ ∃ k : ℕ, 4 * b + 5 = k^2 ∧ b = 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_gt_4_perfect_square_l1199_119937


namespace NUMINAMATH_GPT_geometric_sequence_product_geometric_sequence_sum_not_definitely_l1199_119946

theorem geometric_sequence_product (a b : ℕ → ℝ) (r1 r2 : ℝ) 
  (ha : ∀ n, a (n+1) = r1 * a n)
  (hb : ∀ n, b (n+1) = r2 * b n) :
  ∃ r3, ∀ n, (a n * b n) = r3 * (a (n-1) * b (n-1)) :=
sorry

theorem geometric_sequence_sum_not_definitely (a b : ℕ → ℝ) (r1 r2 : ℝ) 
  (ha : ∀ n, a (n+1) = r1 * a n)
  (hb : ∀ n, b (n+1) = r2 * b n) :
  ¬ ∀ r3, ∃ N, ∀ n ≥ N, (a n + b n) = r3 * (a (n-1) + b (n-1)) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_product_geometric_sequence_sum_not_definitely_l1199_119946


namespace NUMINAMATH_GPT_sequence_with_limit_is_bounded_bounded_sequence_does_not_imply_limit_l1199_119917

-- Part a) Prove that if a sequence has a limit, then it is bounded.
theorem sequence_with_limit_is_bounded (x : ℕ → ℝ) (x0 : ℝ) (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - x0| < ε) :
  ∃ C, ∀ n, |x n| ≤ C := by
  sorry

-- Part b) Is the converse statement true?
theorem bounded_sequence_does_not_imply_limit :
  ∃ (x : ℕ → ℝ), (∃ C, ∀ n, |x n| ≤ C) ∧ ¬(∃ x0, ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - x0| < ε) := by
  sorry

end NUMINAMATH_GPT_sequence_with_limit_is_bounded_bounded_sequence_does_not_imply_limit_l1199_119917


namespace NUMINAMATH_GPT_x_squared_minus_y_squared_l1199_119914

open Real

theorem x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 4/9)
  (h2 : x - y = 2/9) :
  x^2 - y^2 = 8/81 :=
by
  sorry

end NUMINAMATH_GPT_x_squared_minus_y_squared_l1199_119914


namespace NUMINAMATH_GPT_sum_gcd_lcm_l1199_119981

theorem sum_gcd_lcm (A B : ℕ) (hA : A = Nat.gcd 10 (Nat.gcd 15 25)) (hB : B = Nat.lcm 10 (Nat.lcm 15 25)) :
  A + B = 155 :=
by
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_l1199_119981


namespace NUMINAMATH_GPT_remaining_amoeba_is_blue_l1199_119986

-- Define the initial number of amoebas for red, blue, and yellow types.
def n1 := 47
def n2 := 40
def n3 := 53

-- Define the property that remains constant, i.e., the parity of differences
def parity_diff (a b : ℕ) : Bool := (a - b) % 2 == 1

-- Initial conditions based on the given problem
def initial_conditions : Prop :=
  parity_diff n1 n2 = true ∧  -- odd
  parity_diff n1 n3 = false ∧ -- even
  parity_diff n2 n3 = true    -- odd

-- Final statement: Prove that the remaining amoeba is blue
theorem remaining_amoeba_is_blue : Prop :=
  initial_conditions ∧ (∀ final : String, final = "Blue")

end NUMINAMATH_GPT_remaining_amoeba_is_blue_l1199_119986


namespace NUMINAMATH_GPT_min_distance_from_curve_to_line_l1199_119958

open Real

-- Definitions and conditions
def curve_eq (x y: ℝ) : Prop := (x^2 - y - 2 * log (sqrt x) = 0)
def line_eq (x y: ℝ) : Prop := (4 * x + 4 * y + 1 = 0)

-- The main statement
theorem min_distance_from_curve_to_line :
  ∃ (x y : ℝ), curve_eq x y ∧ y = x^2 - 2 * log (sqrt x) ∧ line_eq x y ∧ y = -x - 1/4 ∧ 
               |4 * (1/2) + 4 * ((1/4) + log 2) + 1| / sqrt 32 = sqrt 2 / 2 * (1 + log 2) :=
by
  -- We skip the proof as requested, using sorry:
  sorry

end NUMINAMATH_GPT_min_distance_from_curve_to_line_l1199_119958


namespace NUMINAMATH_GPT_area_of_circles_l1199_119939

theorem area_of_circles (BD AC : ℝ) (hBD : BD = 6) (hAC : AC = 12) : 
  ∃ S : ℝ, S = 225 / 4 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_circles_l1199_119939


namespace NUMINAMATH_GPT_right_triangle_acute_angle_l1199_119977

theorem right_triangle_acute_angle (x : ℝ) 
  (h1 : 5 * x = 90) : x = 18 :=
by sorry

end NUMINAMATH_GPT_right_triangle_acute_angle_l1199_119977


namespace NUMINAMATH_GPT_logarithm_identity_l1199_119996

noncomputable section

open Real

theorem logarithm_identity : 
  log 10 = (log (sqrt 5) / log 10 + (1 / 2) * log 20) :=
sorry

end NUMINAMATH_GPT_logarithm_identity_l1199_119996


namespace NUMINAMATH_GPT_find_15th_term_l1199_119900

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ a 2 = 4 ∧ ∀ n, a (n + 2) = a n

theorem find_15th_term :
  ∃ a : ℕ → ℕ, seq a ∧ a 15 = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_15th_term_l1199_119900


namespace NUMINAMATH_GPT_find_number_divided_by_3_equals_subtracted_5_l1199_119912

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end NUMINAMATH_GPT_find_number_divided_by_3_equals_subtracted_5_l1199_119912


namespace NUMINAMATH_GPT_intersection_of_lines_l1199_119929

theorem intersection_of_lines :
  ∃ x y : ℚ, (8 * x - 3 * y = 9) ∧ (6 * x + 2 * y = 20) ∧ (x = 39 / 17) ∧ (y = 53 / 17) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l1199_119929


namespace NUMINAMATH_GPT_find_b_perpendicular_lines_l1199_119952

theorem find_b_perpendicular_lines :
  ∀ (b : ℝ), (∀ x y : ℝ, 2 * x - 3 * y + 6 = 0 ∧ b * x - 3 * y + 6 = 0 →
      (2 / 3) * (b / 3) = -1) → b = -9 / 2 :=
sorry

end NUMINAMATH_GPT_find_b_perpendicular_lines_l1199_119952


namespace NUMINAMATH_GPT_messages_on_monday_l1199_119987

theorem messages_on_monday (M : ℕ) (h0 : 200 + 500 + 1000 = 1700) (h1 : M + 1700 = 2000) : M = 300 :=
by
  -- Maths proof step here
  sorry

end NUMINAMATH_GPT_messages_on_monday_l1199_119987


namespace NUMINAMATH_GPT_all_equal_l1199_119971

variable (a : ℕ → ℝ)

axiom h1 : a 1 - 3 * a 2 + 2 * a 3 ≥ 0
axiom h2 : a 2 - 3 * a 3 + 2 * a 4 ≥ 0
axiom h3 : a 3 - 3 * a 4 + 2 * a 5 ≥ 0
axiom h4 : ∀ n, 4 ≤ n ∧ n ≤ 98 → a n - 3 * a (n + 1) + 2 * a (n + 2) ≥ 0
axiom h99 : a 99 - 3 * a 100 + 2 * a 1 ≥ 0
axiom h100 : a 100 - 3 * a 1 + 2 * a 2 ≥ 0

theorem all_equal : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 100 → a i = a j := by
  sorry

end NUMINAMATH_GPT_all_equal_l1199_119971


namespace NUMINAMATH_GPT_lower_limit_total_people_l1199_119910

/-- 
  Given:
    1. Exactly 3/7 of the people in the room are under the age of 21.
    2. Exactly 5/10 of the people in the room are over the age of 65.
    3. There are 30 people in the room under the age of 21.
  Prove: The lower limit of the total number of people in the room is 70.
-/
theorem lower_limit_total_people (T : ℕ) (h1 : (3 / 7) * T = 30) : T = 70 := by
  sorry

end NUMINAMATH_GPT_lower_limit_total_people_l1199_119910


namespace NUMINAMATH_GPT_dot_product_is_five_l1199_119948

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-1, 3)

-- Define the condition that involves a and b
def condition : Prop := 2 • a - b = (3, 1)

-- Prove that the dot product of a and b equals 5 given the condition
theorem dot_product_is_five : condition → (a.1 * b.1 + a.2 * b.2) = 5 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_is_five_l1199_119948


namespace NUMINAMATH_GPT_mod_inverse_non_existence_mod_inverse_existence_l1199_119976

theorem mod_inverse_non_existence (a b c d : ℕ) (h1 : 1105 = a * b * c) (h2 : 15 = d * a) :
    ¬ ∃ x : ℕ, (15 * x) % 1105 = 1 := by sorry

theorem mod_inverse_existence (a b : ℕ) (h1 : 221 = a * b) :
    ∃ x : ℕ, (15 * x) % 221 = 59 := by sorry

end NUMINAMATH_GPT_mod_inverse_non_existence_mod_inverse_existence_l1199_119976


namespace NUMINAMATH_GPT_max_tiles_accommodated_l1199_119962

/-- 
The rectangular tiles, each of size 40 cm by 28 cm, must be laid horizontally on a rectangular floor
of size 280 cm by 240 cm, such that the tiles do not overlap, and they are placed in an alternating
checkerboard pattern with edges jutting against each other on all edges. A tile can be placed in any
orientation so long as its edges are parallel to the edges of the floor, and it follows the required
checkerboard pattern. No tile should overshoot any edge of the floor. Determine the maximum number 
of tiles that can be accommodated on the floor while adhering to the placement pattern.
-/
theorem max_tiles_accommodated (tile_len tile_wid floor_len floor_wid : ℕ)
  (h_tile_len : tile_len = 40)
  (h_tile_wid : tile_wid = 28)
  (h_floor_len : floor_len = 280)
  (h_floor_wid : floor_wid = 240) :
  tile_len * tile_wid * 12 ≤ floor_len * floor_wid :=
by 
  sorry

end NUMINAMATH_GPT_max_tiles_accommodated_l1199_119962


namespace NUMINAMATH_GPT_start_time_6am_l1199_119913

def travel_same_time (t : ℝ) (x : ℝ) (y : ℝ) (constant_speed : Prop) : Prop :=
  (x = t + 4) ∧ (y = t + 9) ∧ constant_speed 

theorem start_time_6am
  (x y t: ℝ)
  (constant_speed : Prop) 
  (meet_noon : travel_same_time t x y constant_speed)
  (eqn : 1/t + 1/(t + 4) + 1/(t + 9) = 1) :
  t = 6 :=
by
  sorry

end NUMINAMATH_GPT_start_time_6am_l1199_119913


namespace NUMINAMATH_GPT_product_of_positive_c_for_rational_solutions_l1199_119944

theorem product_of_positive_c_for_rational_solutions : 
  (∃ c₁ c₂ : ℕ, c₁ > 0 ∧ c₂ > 0 ∧ 
   (∀ x : ℝ, (3 * x ^ 2 + 7 * x + c₁ = 0) → ∃ r₁ r₂ : ℚ, x = r₁ ∨ x = r₂) ∧ 
   (∀ x : ℝ, (3 * x ^ 2 + 7 * x + c₂ = 0) → ∃ r₁ r₂ : ℚ, x = r₁ ∨ x = r₂) ∧ 
   c₁ * c₂ = 8) :=
sorry

end NUMINAMATH_GPT_product_of_positive_c_for_rational_solutions_l1199_119944


namespace NUMINAMATH_GPT_jackson_earned_on_monday_l1199_119925

-- Definitions
def goal := 1000
def tuesday_earnings := 40
def avg_rate := 10
def houses := 88
def days_remaining := 3
def total_collected_remaining_days := days_remaining * (houses / 4) * avg_rate

-- The proof problem statement
theorem jackson_earned_on_monday (m : ℕ) :
  m + tuesday_earnings + total_collected_remaining_days = goal → m = 300 :=
by
  -- We will eventually provide the proof here
  sorry

end NUMINAMATH_GPT_jackson_earned_on_monday_l1199_119925


namespace NUMINAMATH_GPT_expression_equals_five_l1199_119935

-- Define the innermost expression
def inner_expr : ℤ := -|( -2 + 3 )|

-- Define the next layer of the expression
def middle_expr : ℤ := |(inner_expr) - 2|

-- Define the outer expression
def outer_expr : ℤ := |middle_expr + 2|

-- The proof problem statement (in this case, without the proof)
theorem expression_equals_five : outer_expr = 5 :=
by
  -- Insert precise conditions directly from the problem statement
  have h_inner : inner_expr = -|1| := by sorry
  have h_middle : middle_expr = |-1 - 2| := by sorry
  have h_outer : outer_expr = |(-3) + 2| := by sorry
  -- The final goal that needs to be proven
  sorry

end NUMINAMATH_GPT_expression_equals_five_l1199_119935


namespace NUMINAMATH_GPT_ice_cream_cost_l1199_119903

-- Define the given conditions
def cost_brownie : ℝ := 2.50
def cost_syrup_per_unit : ℝ := 0.50
def cost_nuts : ℝ := 1.50
def cost_total : ℝ := 7.00
def scoops_ice_cream : ℕ := 2
def syrup_units : ℕ := 2

-- Define the hot brownie dessert cost equation
def hot_brownie_cost (cost_ice_cream_per_scoop : ℝ) : ℝ :=
  cost_brownie + (cost_syrup_per_unit * syrup_units) + cost_nuts + (scoops_ice_cream * cost_ice_cream_per_scoop)

-- Define the theorem we want to prove
theorem ice_cream_cost : hot_brownie_cost 1 = cost_total :=
by sorry

end NUMINAMATH_GPT_ice_cream_cost_l1199_119903


namespace NUMINAMATH_GPT_percentage_loss_is_25_l1199_119972

def cost_price := 1400
def selling_price := 1050
def loss := cost_price - selling_price
def percentage_loss := (loss / cost_price) * 100

theorem percentage_loss_is_25 : percentage_loss = 25 := by
  sorry

end NUMINAMATH_GPT_percentage_loss_is_25_l1199_119972


namespace NUMINAMATH_GPT_pieces_1994_impossible_pieces_1997_possible_l1199_119951

def P (n : ℕ) : ℕ := 1 + 4 * n

theorem pieces_1994_impossible : ∀ n : ℕ, P n ≠ 1994 := 
by sorry

theorem pieces_1997_possible : ∃ n : ℕ, P n = 1997 := 
by sorry

end NUMINAMATH_GPT_pieces_1994_impossible_pieces_1997_possible_l1199_119951


namespace NUMINAMATH_GPT_like_terms_monomials_l1199_119982

theorem like_terms_monomials (m n : ℕ) (h₁ : m = 2) (h₂ : n = 1) : m + n = 3 := 
by
  sorry

end NUMINAMATH_GPT_like_terms_monomials_l1199_119982


namespace NUMINAMATH_GPT_omega_bound_l1199_119934

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - Real.sin (ω * x)

theorem omega_bound (ω : ℝ) (h₁ : ω > 0)
  (h₂ : ∀ x : ℝ, -π / 2 < x ∧ x < π / 2 → (f ω x) ≤ (f ω (-π / 2))) :
  ω ≤ 1 / 2 :=
sorry

end NUMINAMATH_GPT_omega_bound_l1199_119934


namespace NUMINAMATH_GPT_symmetric_scanning_codes_count_l1199_119906

noncomputable def countSymmetricScanningCodes : ℕ :=
  let totalConfigs := 32
  let invalidConfigs := 2
  totalConfigs - invalidConfigs

theorem symmetric_scanning_codes_count :
  countSymmetricScanningCodes = 30 :=
by
  -- Here, we would detail the steps, but we omit the actual proof for now.
  sorry

end NUMINAMATH_GPT_symmetric_scanning_codes_count_l1199_119906


namespace NUMINAMATH_GPT_bridge_length_l1199_119916

def train_length : ℕ := 120
def train_speed : ℕ := 45
def crossing_time : ℕ := 30

theorem bridge_length :
  let speed_m_per_s := (train_speed * 1000) / 3600
  let total_distance := speed_m_per_s * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 255 := by
  sorry

end NUMINAMATH_GPT_bridge_length_l1199_119916


namespace NUMINAMATH_GPT_beehive_bee_count_l1199_119961

theorem beehive_bee_count {a : ℕ → ℕ} (h₀ : a 0 = 1)
  (h₁ : a 1 = 6)
  (hn : ∀ n, a (n + 1) = a n + 5 * a n) :
  a 6 = 46656 :=
  sorry

end NUMINAMATH_GPT_beehive_bee_count_l1199_119961


namespace NUMINAMATH_GPT_no_such_function_exists_l1199_119938

namespace ProofProblem

open Nat

-- Declaration of the proposed function
def f : ℕ+ → ℕ+ := sorry

-- Statement to be proved
theorem no_such_function_exists : 
  ¬ ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, f^[n] n = n + 1 :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_no_such_function_exists_l1199_119938


namespace NUMINAMATH_GPT_trams_required_l1199_119999

theorem trams_required (initial_trams : ℕ) (initial_interval : ℚ) (reduction_fraction : ℚ) :
  initial_trams = 12 ∧ initial_interval = 5 ∧ reduction_fraction = 1/5 →
  (initial_trams + initial_trams * reduction_fraction - initial_trams) = 3 :=
by
  sorry

end NUMINAMATH_GPT_trams_required_l1199_119999


namespace NUMINAMATH_GPT_tan_600_eq_sqrt3_l1199_119974

theorem tan_600_eq_sqrt3 : (Real.tan (600 * Real.pi / 180)) = Real.sqrt 3 := 
by 
  -- sorry to skip the actual proof steps
  sorry

end NUMINAMATH_GPT_tan_600_eq_sqrt3_l1199_119974


namespace NUMINAMATH_GPT_problem_result_l1199_119920

noncomputable def max_value (x y : ℝ) (hx : 2 * x^2 - x * y + y^2 = 15) : ℝ :=
  2 * x^2 + x * y + y^2

theorem problem (x y : ℝ) (hx : 2 * x^2 - x * y + y^2 = 15) :
  max_value x y hx = (75 + 60 * Real.sqrt 2) / 7 :=
sorry

theorem result : 75 + 60 + 2 + 7 = 144 :=
by norm_num

end NUMINAMATH_GPT_problem_result_l1199_119920


namespace NUMINAMATH_GPT_weigh_1_to_10_kg_l1199_119968

theorem weigh_1_to_10_kg (n : ℕ) : 1 ≤ n ∧ n ≤ 10 →
  ∃ (a b c : ℤ), 
    (abs a ≤ 1 ∧ abs b ≤ 1 ∧ abs c ≤ 1 ∧
    (n = a * 3 + b * 4 + c * 9)) :=
by sorry

end NUMINAMATH_GPT_weigh_1_to_10_kg_l1199_119968


namespace NUMINAMATH_GPT_tan_alpha_second_quadrant_l1199_119985

noncomputable def tan_alpha (α : ℝ) : ℝ := Real.tan α

theorem tan_alpha_second_quadrant (α : ℝ) 
  (h1 : α > π / 2 ∧ α < π)
  (h2 : Real.cos (π / 2 - α) = 4 / 5) :
  tan_alpha α = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_second_quadrant_l1199_119985


namespace NUMINAMATH_GPT_tan_beta_value_l1199_119928

theorem tan_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.tan α = 4 / 3)
  (h4 : Real.cos (α + β) = Real.sqrt 5 / 5) :
  Real.tan β = 2 / 11 := 
sorry

end NUMINAMATH_GPT_tan_beta_value_l1199_119928


namespace NUMINAMATH_GPT_g_1993_at_2_l1199_119954

def g (x : ℚ) : ℚ := (2 + x) / (1 - 4 * x^2)

def g_n : ℕ → ℚ → ℚ 
| 0     => id
| (n+1) => λ x => g (g_n n x)

theorem g_1993_at_2 : g_n 1993 2 = 65 / 53 := 
  sorry

end NUMINAMATH_GPT_g_1993_at_2_l1199_119954


namespace NUMINAMATH_GPT_lcm_of_3_8_9_12_l1199_119959

theorem lcm_of_3_8_9_12 : Nat.lcm (Nat.lcm 3 8) (Nat.lcm 9 12) = 72 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_3_8_9_12_l1199_119959


namespace NUMINAMATH_GPT_find_angleZ_l1199_119926

-- Definitions of angles
def angleX : ℝ := 100
def angleY : ℝ := 130

-- Define Z angle based on the conditions given
def angleZ : ℝ := 130

theorem find_angleZ (p q : Prop) (parallel_pq : p ∧ q)
  (h1 : angleX = 100)
  (h2 : angleY = 130) :
  angleZ = 130 :=
by
  sorry

end NUMINAMATH_GPT_find_angleZ_l1199_119926


namespace NUMINAMATH_GPT_represent_380000_in_scientific_notation_l1199_119960

theorem represent_380000_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 380000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.8 ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_represent_380000_in_scientific_notation_l1199_119960


namespace NUMINAMATH_GPT_max_mn_l1199_119984

theorem max_mn (m n : ℝ) (h : m + n = 1) : mn ≤ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_mn_l1199_119984


namespace NUMINAMATH_GPT_rational_inequality_solution_l1199_119980

open Set

theorem rational_inequality_solution (x : ℝ) :
  (x < -1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 5)) ↔ (x - 5) / ((x - 2) * (x^2 - 1)) < 0 := 
sorry

end NUMINAMATH_GPT_rational_inequality_solution_l1199_119980


namespace NUMINAMATH_GPT_a_n_value_l1199_119964

theorem a_n_value (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 3) (h2 : ∀ n, S (n + 1) = 2 * S n) (h3 : S 1 = a 1)
  (h4 : ∀ n, S n = 3 * 2^(n - 1)) : a 4 = 12 :=
sorry

end NUMINAMATH_GPT_a_n_value_l1199_119964


namespace NUMINAMATH_GPT_convert_decimal_to_fraction_l1199_119989

theorem convert_decimal_to_fraction : (3.75 : ℚ) = 15 / 4 := 
by
  sorry

end NUMINAMATH_GPT_convert_decimal_to_fraction_l1199_119989


namespace NUMINAMATH_GPT_acd_over_b_eq_neg_210_l1199_119901

theorem acd_over_b_eq_neg_210 
  (a b c d x : ℤ) 
  (h1 : x = (a + b*Real.sqrt c)/d) 
  (h2 : (7*x)/8 + 1 = 4/x) 
  : (a * c * d) / b = -210 := 
by 
  sorry

end NUMINAMATH_GPT_acd_over_b_eq_neg_210_l1199_119901


namespace NUMINAMATH_GPT_vertical_distance_to_Felix_l1199_119915

/--
  Dora is at point (8, -15).
  Eli is at point (2, 18).
  Felix is at point (5, 7).
  Calculate the vertical distance they need to walk to reach Felix.
-/
theorem vertical_distance_to_Felix :
  let Dora := (8, -15)
  let Eli := (2, 18)
  let Felix := (5, 7)
  let midpoint := ((Dora.1 + Eli.1) / 2, (Dora.2 + Eli.2) / 2)
  let vertical_distance := Felix.2 - midpoint.2
  vertical_distance = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_vertical_distance_to_Felix_l1199_119915


namespace NUMINAMATH_GPT_kate_money_ratio_l1199_119930

-- Define the cost of the pen and the amount Kate needs
def pen_cost : ℕ := 30
def additional_money_needed : ℕ := 20

-- Define the amount of money Kate has
def kate_savings : ℕ := pen_cost - additional_money_needed

-- Define the ratio of Kate's money to the cost of the pen
def ratio (a b : ℕ) : ℕ × ℕ := (a / Nat.gcd a b, b / Nat.gcd a b)

-- The target property: the ratio of Kate's savings to the cost of the pen
theorem kate_money_ratio : ratio kate_savings pen_cost = (1, 3) :=
by
  sorry

end NUMINAMATH_GPT_kate_money_ratio_l1199_119930


namespace NUMINAMATH_GPT_fraction_proof_l1199_119909

theorem fraction_proof (x y : ℕ) (h1 : y = 7) (h2 : x = 22) : 
  (y / (x - 1) = 1 / 3) ∧ ((y + 4) / x = 1 / 2) := by
  sorry

end NUMINAMATH_GPT_fraction_proof_l1199_119909


namespace NUMINAMATH_GPT_de_morgan_neg_or_l1199_119918

theorem de_morgan_neg_or (p q : Prop) (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
by sorry

end NUMINAMATH_GPT_de_morgan_neg_or_l1199_119918


namespace NUMINAMATH_GPT_force_is_correct_l1199_119911

noncomputable def force_computation : ℝ :=
  let m : ℝ := 5 -- kg
  let s : ℝ → ℝ := fun t => 2 * t + 3 * t^2 -- cm
  let a : ℝ := 6 / 100 -- acceleration in m/s^2
  m * a

theorem force_is_correct : force_computation = 0.3 := 
by
  -- Initial conditions
  sorry

end NUMINAMATH_GPT_force_is_correct_l1199_119911


namespace NUMINAMATH_GPT_balls_in_boxes_l1199_119902

/-
Prove that the number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes is 132.
-/
theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 132 ∧ ways = 
    (1) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 3) + 
    (Nat.choose 6 3 * Nat.choose 3 2) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 6) := 
by
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l1199_119902
