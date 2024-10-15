import Mathlib

namespace NUMINAMATH_GPT_coordinates_of_point_M_l1883_188347

theorem coordinates_of_point_M 
  (M : ℝ × ℝ) 
  (dist_x_axis : abs M.2 = 5) 
  (dist_y_axis : abs M.1 = 4) 
  (second_quadrant : M.1 < 0 ∧ M.2 > 0) : 
  M = (-4, 5) := 
sorry

end NUMINAMATH_GPT_coordinates_of_point_M_l1883_188347


namespace NUMINAMATH_GPT_binom_2n_2_eq_n_2n_minus_1_l1883_188305

theorem binom_2n_2_eq_n_2n_minus_1 (n : ℕ) (h : n > 0) : 
  (Nat.choose (2 * n) 2) = n * (2 * n - 1) := 
sorry

end NUMINAMATH_GPT_binom_2n_2_eq_n_2n_minus_1_l1883_188305


namespace NUMINAMATH_GPT_num_sets_satisfying_union_l1883_188339

theorem num_sets_satisfying_union : 
  ∃! (A : Set ℕ), ({1, 3} ∪ A = {1, 3, 5}) :=
by
  sorry

end NUMINAMATH_GPT_num_sets_satisfying_union_l1883_188339


namespace NUMINAMATH_GPT_smallest_n_with_square_ending_in_2016_l1883_188392

theorem smallest_n_with_square_ending_in_2016 : 
  ∃ n : ℕ, (n^2 % 10000 = 2016) ∧ (n = 996) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_with_square_ending_in_2016_l1883_188392


namespace NUMINAMATH_GPT_cos_of_angle_C_l1883_188377

theorem cos_of_angle_C (A B C : ℝ)
  (h1 : Real.sin (π - A) = 3 / 5)
  (h2 : Real.tan (π + B) = 12 / 5)
  (h_cos_A : Real.cos A = 4 / 5) :
  Real.cos C = 16 / 65 :=
sorry

end NUMINAMATH_GPT_cos_of_angle_C_l1883_188377


namespace NUMINAMATH_GPT_joan_total_spent_l1883_188375

theorem joan_total_spent (cost_basketball cost_racing total_spent : ℝ) 
  (h1 : cost_basketball = 5.20) 
  (h2 : cost_racing = 4.23) 
  (h3 : total_spent = cost_basketball + cost_racing) : 
  total_spent = 9.43 := 
by 
  sorry

end NUMINAMATH_GPT_joan_total_spent_l1883_188375


namespace NUMINAMATH_GPT_sequence_a_5_l1883_188352

theorem sequence_a_5 (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → S n = 2 * a n - 3) (h2 : ∀ n : ℕ, n > 0 → a n = S n - S (n - 1)) :
  a 5 = 48 := by
  -- The proof and implementations are omitted
  sorry

end NUMINAMATH_GPT_sequence_a_5_l1883_188352


namespace NUMINAMATH_GPT_divisors_end_with_1_l1883_188335

theorem divisors_end_with_1 (n : ℕ) (h : n > 0) :
  ∀ d : ℕ, d ∣ (10^(5^n) - 1) / 9 → d % 10 = 1 :=
sorry

end NUMINAMATH_GPT_divisors_end_with_1_l1883_188335


namespace NUMINAMATH_GPT_find_r_value_l1883_188388

theorem find_r_value (n : ℕ) (r s : ℕ) (h_s : s = 2^n - 1) (h_r : r = 3^s - s) (h_n : n = 3) : r = 2180 :=
by
  sorry

end NUMINAMATH_GPT_find_r_value_l1883_188388


namespace NUMINAMATH_GPT_op_evaluation_l1883_188366

-- Define the custom operation ⊕
def op (a b c : ℝ) : ℝ := b^2 - 3 * a * c

-- Statement of the theorem we want to prove
theorem op_evaluation : op 2 3 4 = -15 :=
by 
  -- This is a placeholder for the actual proof,
  -- which in a real scenario would involve computing the operation.
  sorry

end NUMINAMATH_GPT_op_evaluation_l1883_188366


namespace NUMINAMATH_GPT_probability_all_yellow_l1883_188315

-- Definitions and conditions
def total_apples : ℕ := 8
def red_apples : ℕ := 5
def yellow_apples : ℕ := 3
def chosen_apples : ℕ := 3

-- Theorem to prove
theorem probability_all_yellow :
  (yellow_apples.choose chosen_apples : ℚ) / (total_apples.choose chosen_apples) = 1 / 56 := sorry

end NUMINAMATH_GPT_probability_all_yellow_l1883_188315


namespace NUMINAMATH_GPT_average_weight_l1883_188326

theorem average_weight 
  (n₁ n₂ : ℕ) 
  (avg₁ avg₂ total_avg : ℚ) 
  (h₁ : n₁ = 24) 
  (h₂ : n₂ = 8)
  (h₃ : avg₁ = 50.25)
  (h₄ : avg₂ = 45.15)
  (h₅ : total_avg = 48.975) :
  ( (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = total_avg ) :=
sorry

end NUMINAMATH_GPT_average_weight_l1883_188326


namespace NUMINAMATH_GPT_problem_solution_l1883_188389

noncomputable def ellipse_properties (F1 F2 : ℝ × ℝ) (sum_dists : ℝ) : ℝ × ℝ × ℝ × ℝ := 
  let a := sum_dists / 2 
  let c := (Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2)) / 2
  let b := Real.sqrt (a^2 - c^2)
  let h := (F1.1 + F2.1) / 2
  let k := (F1.2 + F2.2) / 2
  (h, k, a, b)

theorem problem_solution :
  let F1 := (0, 1)
  let F2 := (6, 1)
  let sum_dists := 10
  let (h, k, a, b) := ellipse_properties F1 F2 sum_dists
  h + k + a + b = 13 :=
by
  -- assuming the proof here
  sorry

end NUMINAMATH_GPT_problem_solution_l1883_188389


namespace NUMINAMATH_GPT_find_k_l1883_188361

theorem find_k (k : ℕ) :
  (∑' n : ℕ, (5 + n * k) / 5 ^ n) = 12 → k = 90 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1883_188361


namespace NUMINAMATH_GPT_percentage_running_wickets_l1883_188391

-- Conditions provided as definitions and assumptions in Lean
def total_runs : ℕ := 120
def boundaries : ℕ := 3
def sixes : ℕ := 8
def boundary_runs (b : ℕ) := b * 4
def six_runs (s : ℕ) := s * 6

-- Calculate runs from boundaries and sixes
def runs_from_boundaries := boundary_runs boundaries
def runs_from_sixes := six_runs sixes
def runs_not_from_boundaries_and_sixes := total_runs - (runs_from_boundaries + runs_from_sixes)

-- Proof that the percentage of the total score by running between the wickets is 50%
theorem percentage_running_wickets :
  (runs_not_from_boundaries_and_sixes : ℝ) / (total_runs : ℝ) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_running_wickets_l1883_188391


namespace NUMINAMATH_GPT_sin_neg_270_eq_one_l1883_188374

theorem sin_neg_270_eq_one : Real.sin (-(270 : ℝ) * (Real.pi / 180)) = 1 := by
  sorry

end NUMINAMATH_GPT_sin_neg_270_eq_one_l1883_188374


namespace NUMINAMATH_GPT_faulty_keys_l1883_188328

noncomputable def faulty_digits (typed_sequence : List ℕ) : Set ℕ :=
  { d | d = 7 ∨ d = 9 }

theorem faulty_keys (typed_sequence : List ℕ) (h : typed_sequence.length = 10) :
  (∃ faulty_keys : Set ℕ, ∃ missing_digits : ℕ, missing_digits = 3 ∧ faulty_keys = {7, 9}) :=
sorry

end NUMINAMATH_GPT_faulty_keys_l1883_188328


namespace NUMINAMATH_GPT_range_of_a_l1883_188365

open Set Real

noncomputable def f (x a : ℝ) := x ^ 2 + 2 * x + a

theorem range_of_a (a : ℝ) :
  (∃ x, 1 ≤ x ∧ x ≤ 2 ∧ f x a ≥ 0) → a ≥ -8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1883_188365


namespace NUMINAMATH_GPT_water_added_16_l1883_188359

theorem water_added_16 (W : ℝ) 
  (h1 : ∃ W, 24 * 0.90 = 0.54 * (24 + W)) : 
  W = 16 := 
by {
  sorry
}

end NUMINAMATH_GPT_water_added_16_l1883_188359


namespace NUMINAMATH_GPT_unique_m_power_function_increasing_l1883_188306

theorem unique_m_power_function_increasing : 
  ∃! (m : ℝ), (∀ x : ℝ, x > 0 → (m^2 - m - 5) * x^(m-1) > 0) ∧ (m^2 - m - 5 = 1) ∧ (m - 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_unique_m_power_function_increasing_l1883_188306


namespace NUMINAMATH_GPT_telephone_call_duration_l1883_188301

theorem telephone_call_duration (x : ℝ) :
  (0.60 + 0.06 * (x - 4) = 0.08 * x) → x = 18 :=
by
  sorry

end NUMINAMATH_GPT_telephone_call_duration_l1883_188301


namespace NUMINAMATH_GPT_number_of_triangles_l1883_188338

theorem number_of_triangles (n : ℕ) (hn : 0 < n) :
  ∃ t, t = (n + 2) ^ 2 - 2 * (⌊ (n : ℝ) / 2 ⌋) / 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_triangles_l1883_188338


namespace NUMINAMATH_GPT_pencils_and_notebooks_cost_l1883_188369

theorem pencils_and_notebooks_cost
    (p n : ℝ)
    (h1 : 8 * p + 10 * n = 5.36)
    (h2 : 12 * (p - 0.05) + 5 * n = 4.05) :
    15 * (p - 0.05) + 12 * n = 7.01 := 
sorry

end NUMINAMATH_GPT_pencils_and_notebooks_cost_l1883_188369


namespace NUMINAMATH_GPT_divide_by_3_result_l1883_188364

-- Definitions
def n : ℕ := 4 * 12

theorem divide_by_3_result (h : n / 4 = 12) : n / 3 = 16 :=
by
  sorry

end NUMINAMATH_GPT_divide_by_3_result_l1883_188364


namespace NUMINAMATH_GPT_tan_alpha_frac_simplification_l1883_188327

theorem tan_alpha_frac_simplification (α : ℝ) (h : Real.tan α = -1 / 2) : 
  (2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 4 / 3 :=
by sorry

end NUMINAMATH_GPT_tan_alpha_frac_simplification_l1883_188327


namespace NUMINAMATH_GPT_election_votes_l1883_188346

variable (V : ℝ)

theorem election_votes (h1 : 0.70 * V - 0.30 * V = 192) : V = 480 :=
by
  sorry

end NUMINAMATH_GPT_election_votes_l1883_188346


namespace NUMINAMATH_GPT_M_lt_N_l1883_188341

variables (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

def N : ℝ := |a + b + c| + |2 * a - b|
def M : ℝ := |a - b + c| + |2 * a + b|

axiom h1 : f 1 < 0  -- a + b + c < 0
axiom h2 : f (-1) > 0  -- a - b + c > 0
axiom h3 : a > 0
axiom h4 : -b / (2 * a) > 1

theorem M_lt_N : M a b c < N a b c :=
by
  sorry

end NUMINAMATH_GPT_M_lt_N_l1883_188341


namespace NUMINAMATH_GPT_sum_third_three_l1883_188300

variables {a : ℕ → ℤ}

-- Define the properties of the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

-- Given conditions
axiom sum_first_three : a 1 + a 2 + a 3 = 9
axiom sum_second_three : a 4 + a 5 + a 6 = 27
axiom arithmetic_seq : is_arithmetic_sequence a

-- The proof goal
theorem sum_third_three : a 7 + a 8 + a 9 = 45 :=
by
  sorry  -- Proof is omitted here

end NUMINAMATH_GPT_sum_third_three_l1883_188300


namespace NUMINAMATH_GPT_bigger_part_l1883_188303

theorem bigger_part (x y : ℕ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) : y = 34 :=
sorry

end NUMINAMATH_GPT_bigger_part_l1883_188303


namespace NUMINAMATH_GPT_sum_of_ages_is_nineteen_l1883_188333

-- Definitions representing the conditions
def Bella_age : ℕ := 5
def Brother_is_older : ℕ := 9
def Brother_age : ℕ := Bella_age + Brother_is_older
def Sum_of_ages : ℕ := Bella_age + Brother_age

-- Mathematical statement (theorem) to be proved
theorem sum_of_ages_is_nineteen : Sum_of_ages = 19 := by
  sorry

end NUMINAMATH_GPT_sum_of_ages_is_nineteen_l1883_188333


namespace NUMINAMATH_GPT_max_reflections_l1883_188380

theorem max_reflections (A B D : Point) (n : ℕ) (angle_CDA : ℝ) (incident_angle : ℕ → ℝ)
  (h1 : angle_CDA = 12)
  (h2 : ∀ k : ℕ, k ≤ n → incident_angle k = k * angle_CDA)
  (h3 : incident_angle n = 90) :
  n = 7 := 
sorry

end NUMINAMATH_GPT_max_reflections_l1883_188380


namespace NUMINAMATH_GPT_valid_rod_count_l1883_188321

open Nat

theorem valid_rod_count :
  ∃ valid_rods : Finset ℕ,
    (∀ d ∈ valid_rods, 6 ≤ d ∧ d < 35 ∧ d ≠ 5 ∧ d ≠ 10 ∧ d ≠ 20) ∧ 
    valid_rods.card = 26 := sorry

end NUMINAMATH_GPT_valid_rod_count_l1883_188321


namespace NUMINAMATH_GPT_average_is_correct_l1883_188309

theorem average_is_correct (x : ℝ) : 
  (2 * x + 12 + 3 * x + 3 + 5 * x - 8) / 3 = 3 * x + 2 → x = -1 :=
by
  sorry

end NUMINAMATH_GPT_average_is_correct_l1883_188309


namespace NUMINAMATH_GPT_bags_with_chocolate_hearts_l1883_188397

-- Definitions for given conditions
def total_candies : ℕ := 63
def total_bags : ℕ := 9
def candies_per_bag : ℕ := total_candies / total_bags
def chocolate_kiss_bags : ℕ := 3
def not_chocolate_candies : ℕ := 28
def bags_not_chocolate : ℕ := not_chocolate_candies / candies_per_bag
def remaining_bags : ℕ := total_bags - chocolate_kiss_bags - bags_not_chocolate

-- Statement to be proved
theorem bags_with_chocolate_hearts :
  remaining_bags = 2 := by 
  sorry

end NUMINAMATH_GPT_bags_with_chocolate_hearts_l1883_188397


namespace NUMINAMATH_GPT_non_neg_sum_of_squares_l1883_188312

theorem non_neg_sum_of_squares (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (h : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_non_neg_sum_of_squares_l1883_188312


namespace NUMINAMATH_GPT_find_base_number_l1883_188351

theorem find_base_number (y : ℕ) (base : ℕ) (h : 9^y = base ^ 16) (hy : y = 8) : base = 3 :=
by
  -- We skip the proof steps and insert sorry here
  sorry

end NUMINAMATH_GPT_find_base_number_l1883_188351


namespace NUMINAMATH_GPT_inequality_holds_for_real_numbers_l1883_188370

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_real_numbers_l1883_188370


namespace NUMINAMATH_GPT_additive_inverse_of_half_l1883_188316

theorem additive_inverse_of_half :
  - (1 / 2) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_additive_inverse_of_half_l1883_188316


namespace NUMINAMATH_GPT_grace_is_14_l1883_188319

def GraceAge (G F C E D : ℕ) : Prop :=
  G = F - 6 ∧ F = C + 2 ∧ E = C + 3 ∧ D = E - 4 ∧ D = 17

theorem grace_is_14 (G F C E D : ℕ) (h : GraceAge G F C E D) : G = 14 :=
by sorry

end NUMINAMATH_GPT_grace_is_14_l1883_188319


namespace NUMINAMATH_GPT_champion_team_exists_unique_champion_wins_all_not_exactly_two_champions_l1883_188362

-- Define the structure and relationship between teams in the tournament
structure Tournament (Team : Type) :=
  (competes : Team → Team → Prop) -- teams play against each other
  (no_ties : ∀ A B : Team, (competes A B ∧ ¬competes B A) ∨ (competes B A ∧ ¬competes A B)) -- no ties
  (superior : Team → Team → Prop) -- superiority relationship
  (superior_def : ∀ A B : Team, superior A B ↔ (competes A B ∧ ¬competes B A) ∨ (∃ C : Team, superior A C ∧ superior C B))

-- The main theorem based on the given questions
theorem champion_team_exists {Team : Type} (tournament : Tournament Team) :
  ∃ champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B) :=
  sorry

theorem unique_champion_wins_all {Team : Type} (tournament : Tournament Team)
  (h : ∃! champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B)) :
  ∃! champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B ∧ tournament.competes champion B ∧ ¬tournament.competes B champion) :=
  sorry

theorem not_exactly_two_champions {Team : Type} (tournament : Tournament Team) :
  ¬∃ A B : Team, A ≠ B ∧ (∀ C : Team, C ≠ A → tournament.superior A C) ∧ (∀ C : Team, C ≠ B → tournament.superior B C) :=
  sorry

end NUMINAMATH_GPT_champion_team_exists_unique_champion_wins_all_not_exactly_two_champions_l1883_188362


namespace NUMINAMATH_GPT_bobby_pays_correct_amount_l1883_188308

noncomputable def bobby_total_cost : ℝ := 
  let mold_cost : ℝ := 250
  let material_original_cost : ℝ := 150
  let material_discount : ℝ := 0.20 * material_original_cost
  let material_cost : ℝ := material_original_cost - material_discount
  let hourly_rate_original : ℝ := 75
  let hourly_rate_increased : ℝ := hourly_rate_original + 10
  let work_hours : ℝ := 8
  let work_cost_original : ℝ := work_hours * hourly_rate_increased
  let work_cost_discount : ℝ := 0.80 * work_cost_original
  let cost_before_tax : ℝ := mold_cost + material_cost + work_cost_discount
  let tax : ℝ := 0.10 * cost_before_tax
  cost_before_tax + tax

theorem bobby_pays_correct_amount : bobby_total_cost = 1005.40 := sorry

end NUMINAMATH_GPT_bobby_pays_correct_amount_l1883_188308


namespace NUMINAMATH_GPT_find_other_x_intercept_l1883_188395

theorem find_other_x_intercept (a b c : ℝ) (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 4)^2 + 9)
  (h2 : a * 0^2 + b * 0 + c = 0) : ∃ x, x ≠ 0 ∧ a * x^2 + b * x + c = 0 ∧ x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_other_x_intercept_l1883_188395


namespace NUMINAMATH_GPT_algebraic_expression_value_l1883_188376

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 2) : a^2 - b^2 - 4*a = -4 := 
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1883_188376


namespace NUMINAMATH_GPT_gcd_6Pn_n_minus_2_l1883_188371

-- Auxiliary definition to calculate the nth pentagonal number
def pentagonal (n : ℕ) : ℕ := n ^ 2

-- Statement of the theorem
theorem gcd_6Pn_n_minus_2 (n : ℕ) (hn : 0 < n) : 
  ∃ d, d = Int.gcd (6 * pentagonal n) (n - 2) ∧ d ≤ 24 ∧ (∀ k, Int.gcd (6 * pentagonal k) (k - 2) ≤ 24) :=
sorry

end NUMINAMATH_GPT_gcd_6Pn_n_minus_2_l1883_188371


namespace NUMINAMATH_GPT_william_marbles_l1883_188334

theorem william_marbles :
  let initial_marbles := 10
  let shared_marbles := 3
  (initial_marbles - shared_marbles) = 7 := 
by
  sorry

end NUMINAMATH_GPT_william_marbles_l1883_188334


namespace NUMINAMATH_GPT_symbols_in_P_l1883_188340
-- Importing the necessary library

-- Define the context P and the operations
def context_P : Type := sorry

def mul_op (P : context_P) : String := "*"
def div_op (P : context_P) : String := "/"
def exp_op (P : context_P) : String := "∧"
def sqrt_op (P : context_P) : String := "SQR"
def abs_op (P : context_P) : String := "ABS"

-- Define what each symbol represents in the context of P
theorem symbols_in_P (P : context_P) :
  (mul_op P = "*") ∧
  (div_op P = "/") ∧
  (exp_op P = "∧") ∧
  (sqrt_op P = "SQR") ∧
  (abs_op P = "ABS") := 
sorry

end NUMINAMATH_GPT_symbols_in_P_l1883_188340


namespace NUMINAMATH_GPT_smallest_fraction_of_land_l1883_188332

noncomputable def smallest_share (n : ℕ) : ℚ :=
  if n = 150 then 1 / (2 * 3^49) else 0

theorem smallest_fraction_of_land :
  smallest_share 150 = 1 / (2 * 3^49) :=
sorry

end NUMINAMATH_GPT_smallest_fraction_of_land_l1883_188332


namespace NUMINAMATH_GPT_expand_and_simplify_l1883_188313

theorem expand_and_simplify :
  ∀ x : ℝ, (x^3 - 3*x + 3)*(x^2 + 3*x + 3) = x^5 + 3*x^4 - 6*x^2 + 9 := by sorry

end NUMINAMATH_GPT_expand_and_simplify_l1883_188313


namespace NUMINAMATH_GPT_equivalent_lemons_l1883_188393

theorem equivalent_lemons 
  (lemons_per_apple_approx : ∀ apples : ℝ, 3/4 * 14 = 9 → 1 = 9 / (3/4 * 14))
  (apples_to_lemons : ℝ) :
  5 / 7 * 7 = 30 / 7 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_lemons_l1883_188393


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1883_188385

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
    (h1 : a 1 = 3)
    (h2 : a 4 = 24)
    (hn : ∀ n, a n = a 1 * q ^ (n - 1)) :
    (a 3 + a 4 + a 5 = 84) :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1883_188385


namespace NUMINAMATH_GPT_jane_paid_five_l1883_188354

noncomputable def cost_of_apple : ℝ := 0.75
noncomputable def change_received : ℝ := 4.25
noncomputable def amount_paid : ℝ := cost_of_apple + change_received

theorem jane_paid_five : amount_paid = 5.00 :=
by
  sorry

end NUMINAMATH_GPT_jane_paid_five_l1883_188354


namespace NUMINAMATH_GPT_alex_jamie_casey_probability_l1883_188398

-- Probability definitions and conditions
def alex_win_prob := 1/3
def casey_win_prob := 1/6
def jamie_win_prob := 1/2

def total_rounds := 8
def alex_wins := 4
def jamie_wins := 3
def casey_wins := 1

-- The probability computation
theorem alex_jamie_casey_probability : 
  alex_win_prob ^ alex_wins * jamie_win_prob ^ jamie_wins * casey_win_prob ^ casey_wins * (Nat.choose total_rounds (alex_wins + jamie_wins + casey_wins)) = 35 / 486 := 
sorry

end NUMINAMATH_GPT_alex_jamie_casey_probability_l1883_188398


namespace NUMINAMATH_GPT_estimate_first_year_students_l1883_188350

noncomputable def number_of_first_year_students (N : ℕ) : Prop :=
  let p1 := (N - 90) / N
  let p2 := (N - 100) / N
  let p_both := 1 - p1 * p2
  p_both = 20 / N → N = 450

theorem estimate_first_year_students : ∃ N : ℕ, number_of_first_year_students N :=
by
  use 450
  -- sorry added to skip the proof part
  sorry

end NUMINAMATH_GPT_estimate_first_year_students_l1883_188350


namespace NUMINAMATH_GPT_one_hundred_fifty_sixth_digit_is_five_l1883_188325

def repeated_sequence := [0, 6, 0, 5, 1, 3]
def target_index := 156 - 1
def block_length := repeated_sequence.length

theorem one_hundred_fifty_sixth_digit_is_five :
  repeated_sequence[target_index % block_length] = 5 :=
by
  sorry

end NUMINAMATH_GPT_one_hundred_fifty_sixth_digit_is_five_l1883_188325


namespace NUMINAMATH_GPT_rectangular_solid_volume_l1883_188314

theorem rectangular_solid_volume 
  (a b c : ℝ) 
  (h1 : a * b = 18) 
  (h2 : b * c = 50) 
  (h3 : a * c = 45) : 
  a * b * c = 150 * Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_rectangular_solid_volume_l1883_188314


namespace NUMINAMATH_GPT_orchids_cut_l1883_188360

-- defining the initial conditions
def initial_orchids : ℕ := 3
def final_orchids : ℕ := 7

-- the question: prove the number of orchids cut
theorem orchids_cut : final_orchids - initial_orchids = 4 := by
  sorry

end NUMINAMATH_GPT_orchids_cut_l1883_188360


namespace NUMINAMATH_GPT_person2_speed_l1883_188381

variables (v_1 : ℕ) (v_2 : ℕ)

def meet_time := 4
def catch_up_time := 16

def meet_equation : Prop := v_1 + v_2 = 22
def catch_up_equation : Prop := v_2 - v_1 = 4

theorem person2_speed :
  meet_equation v_1 v_2 → catch_up_equation v_1 v_2 →
  v_1 = 6 → v_2 = 10 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_person2_speed_l1883_188381


namespace NUMINAMATH_GPT_reflection_matrix_solution_l1883_188384

variable (a b : ℚ)

def matrix_R : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, b], ![-(3/4 : ℚ), (4/5 : ℚ)]]

theorem reflection_matrix_solution (h : matrix_R a b ^ 2 = 1) :
    (a, b) = (-4/5, -3/5) := sorry

end NUMINAMATH_GPT_reflection_matrix_solution_l1883_188384


namespace NUMINAMATH_GPT_set_union_is_correct_l1883_188320

noncomputable def M (a : ℝ) : Set ℝ := {3, 2^a}
noncomputable def N (a b : ℝ) : Set ℝ := {a, b}

variable (a b : ℝ)
variable (h₁ : M a ∩ N a b = {2})
variable (h₂ : ∃ a b, N a b = {1, 2} ∧ M a = {3, 2} ∧ M a ∪ N a b = {1, 2, 3})

theorem set_union_is_correct :
  M 1 ∪ N 1 2 = {1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_set_union_is_correct_l1883_188320


namespace NUMINAMATH_GPT_no_intersection_abs_value_graphs_l1883_188331

theorem no_intersection_abs_value_graphs : 
  ∀ (x : ℝ), ¬ (|3 * x + 6| = -|4 * x - 1|) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_no_intersection_abs_value_graphs_l1883_188331


namespace NUMINAMATH_GPT_sin_cos_equiv_l1883_188373

theorem sin_cos_equiv (x : ℝ) (h : Real.cos x - 5 * Real.sin x = 2) :
  Real.sin x + 5 * Real.cos x = -1/2 ∨ Real.sin x + 5 * Real.cos x = 17/13 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_equiv_l1883_188373


namespace NUMINAMATH_GPT_weight_of_mixture_correct_l1883_188323

-- Defining the fractions of each component in the mixture
def sand_fraction : ℚ := 2 / 9
def water_fraction : ℚ := 5 / 18
def gravel_fraction : ℚ := 1 / 6
def cement_fraction : ℚ := 7 / 36
def limestone_fraction : ℚ := 1 - sand_fraction - water_fraction - gravel_fraction - cement_fraction

-- Given weight of limestone
def limestone_weight : ℚ := 12

-- Total weight of the mixture that we need to prove
def total_mixture_weight : ℚ := 86.4

-- Proof problem statement
theorem weight_of_mixture_correct : (limestone_fraction * total_mixture_weight = limestone_weight) :=
by
  have h_sand := sand_fraction
  have h_water := water_fraction
  have h_gravel := gravel_fraction
  have h_cement := cement_fraction
  have h_limestone := limestone_fraction
  have h_limestone_weight := limestone_weight
  have h_total_weight := total_mixture_weight
  sorry

end NUMINAMATH_GPT_weight_of_mixture_correct_l1883_188323


namespace NUMINAMATH_GPT_ratio_noah_to_joe_l1883_188387

def noah_age_after_10_years : ℕ := 22
def years_elapsed : ℕ := 10
def joe_age : ℕ := 6
def noah_age : ℕ := noah_age_after_10_years - years_elapsed

theorem ratio_noah_to_joe : noah_age / joe_age = 2 := by
  -- calculation omitted for brevity
  sorry

end NUMINAMATH_GPT_ratio_noah_to_joe_l1883_188387


namespace NUMINAMATH_GPT_sum_of_coeffs_l1883_188317

theorem sum_of_coeffs (A B C D : ℤ) (h₁ : A = 1) (h₂ : B = -1) (h₃ : C = -12) (h₄ : D = 3) :
  A + B + C + D = -9 := 
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end NUMINAMATH_GPT_sum_of_coeffs_l1883_188317


namespace NUMINAMATH_GPT_train_length_l1883_188322

theorem train_length (speed_km_hr : ℕ) (time_sec : ℕ) (h_speed : speed_km_hr = 72) (h_time : time_sec = 12) : 
  ∃ length_m : ℕ, length_m = 240 := 
by
  sorry

end NUMINAMATH_GPT_train_length_l1883_188322


namespace NUMINAMATH_GPT_total_pencils_given_out_l1883_188367

theorem total_pencils_given_out (n p : ℕ) (h1 : n = 10) (h2 : p = 5) : n * p = 50 :=
by
  sorry

end NUMINAMATH_GPT_total_pencils_given_out_l1883_188367


namespace NUMINAMATH_GPT_min_value_expression_l1883_188344

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 3) :
  ∃ x : ℝ, (x = (a^2 + b^2 + 22) / (a + b)) ∧ (x = 8) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1883_188344


namespace NUMINAMATH_GPT_total_dog_weight_l1883_188355

theorem total_dog_weight (weight_evans_dog weight_ivans_dog : ℕ)
  (h₁ : weight_evans_dog = 63)
  (h₂ : weight_evans_dog = 7 * weight_ivans_dog) :
  weight_evans_dog + weight_ivans_dog = 72 :=
sorry

end NUMINAMATH_GPT_total_dog_weight_l1883_188355


namespace NUMINAMATH_GPT_johns_final_push_time_l1883_188302

-- Definitions and initial conditions.
def john_initial_distance_behind_steve : ℝ := 12
def john_speed : ℝ := 4.2
def steve_speed : ℝ := 3.7
def john_final_distance_ahead_of_steve : ℝ := 2

-- The statement we want to prove:
theorem johns_final_push_time : ∃ t : ℝ, john_speed * t = steve_speed * t + john_initial_distance_behind_steve + john_final_distance_ahead_of_steve ∧ t = 28 := 
by 
  -- Adding blank proof body
  sorry

end NUMINAMATH_GPT_johns_final_push_time_l1883_188302


namespace NUMINAMATH_GPT_graph_of_equation_l1883_188304

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 :=
by sorry

end NUMINAMATH_GPT_graph_of_equation_l1883_188304


namespace NUMINAMATH_GPT_minimize_expression_l1883_188336

theorem minimize_expression (n : ℕ) (h : n > 0) : (n = 10) ↔ (∀ m : ℕ, m > 0 → (n / 2 + 50 / n: ℝ) ≤ (m / 2 + 50 / m: ℝ)) :=
sorry

end NUMINAMATH_GPT_minimize_expression_l1883_188336


namespace NUMINAMATH_GPT_solve_abs_inequality_l1883_188396

theorem solve_abs_inequality (x : ℝ) :
  |x + 2| + |x - 2| < x + 7 ↔ -7 / 3 < x ∧ x < 7 :=
sorry

end NUMINAMATH_GPT_solve_abs_inequality_l1883_188396


namespace NUMINAMATH_GPT_radius_of_third_circle_l1883_188390

theorem radius_of_third_circle (r₁ r₂ : ℝ) (r₁_val : r₁ = 23) (r₂_val : r₂ = 37) : 
  ∃ r : ℝ, r = 2 * Real.sqrt 210 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_third_circle_l1883_188390


namespace NUMINAMATH_GPT_ratio_H_over_G_l1883_188307

theorem ratio_H_over_G (G H : ℤ)
  (h : ∀ x : ℝ, x ≠ -5 → x ≠ 0 → x ≠ 4 →
    (G : ℝ)/(x + 5) + (H : ℝ)/(x^2 - 4*x) = (x^2 - 2*x + 10) / (x^3 + x^2 - 20*x)) :
  H / G = 2 :=
  sorry

end NUMINAMATH_GPT_ratio_H_over_G_l1883_188307


namespace NUMINAMATH_GPT_girls_at_picnic_l1883_188383

variables (g b : ℕ)

-- Conditions
axiom total_students : g + b = 1500
axiom students_at_picnic : (3/4) * g + (2/3) * b = 900

-- Goal: Prove number of girls who attended the picnic
theorem girls_at_picnic (hg : (3/4 : ℚ) * 1200 = 900) : (3/4 : ℚ) * 1200 = 900 :=
by sorry

end NUMINAMATH_GPT_girls_at_picnic_l1883_188383


namespace NUMINAMATH_GPT_triangle_existence_condition_l1883_188324

theorem triangle_existence_condition 
  (a b f_c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : f_c > 0) : 
  (2 * a * b / (a + b)) > f_c :=
sorry

end NUMINAMATH_GPT_triangle_existence_condition_l1883_188324


namespace NUMINAMATH_GPT_smallest_angle_range_l1883_188311

theorem smallest_angle_range {A B C : ℝ} (hA : 0 < A) (hABC : A + B + C = 180) (horder : A ≤ B ∧ B ≤ C) :
  0 < A ∧ A ≤ 60 := by
  sorry

end NUMINAMATH_GPT_smallest_angle_range_l1883_188311


namespace NUMINAMATH_GPT_find_points_l1883_188368

noncomputable def f (x y z : ℝ) : ℝ := (x^2 + y^2 + z^2) / (x + y + z)

theorem find_points :
  (∃ (x₀ y₀ z₀ : ℝ), 0 < x₀^2 + y₀^2 + z₀^2 ∧ x₀^2 + y₀^2 + z₀^2 < 1 / 1999 ∧
    1.999 < f x₀ y₀ z₀ ∧ f x₀ y₀ z₀ < 2) :=
  sorry

end NUMINAMATH_GPT_find_points_l1883_188368


namespace NUMINAMATH_GPT_holloway_soccer_team_l1883_188353

theorem holloway_soccer_team (P M : Finset ℕ) (hP_union_M : (P ∪ M).card = 20) 
(hP : P.card = 12) (h_int : (P ∩ M).card = 6) : M.card = 14 := 
by
  sorry

end NUMINAMATH_GPT_holloway_soccer_team_l1883_188353


namespace NUMINAMATH_GPT_amount_paid_to_Y_l1883_188399

-- Definition of the conditions.
def total_payment (X Y : ℕ) : Prop := X + Y = 330
def payment_relation (X Y : ℕ) : Prop := X = 12 * Y / 10

-- The theorem we want to prove.
theorem amount_paid_to_Y (X Y : ℕ) (h1 : total_payment X Y) (h2 : payment_relation X Y) : Y = 150 := 
by 
  sorry

end NUMINAMATH_GPT_amount_paid_to_Y_l1883_188399


namespace NUMINAMATH_GPT_N_intersect_M_complement_l1883_188337

-- Definitions based on given conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
def N : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def M_complement : Set ℝ := { x | x < -2 ∨ x > 3 }  -- complement of M in ℝ

-- Lean statement for the proof problem
theorem N_intersect_M_complement :
  N ∩ M_complement = { x | 3 < x ∧ x ≤ 4 } :=
sorry

end NUMINAMATH_GPT_N_intersect_M_complement_l1883_188337


namespace NUMINAMATH_GPT_exist_a_sequence_l1883_188329

theorem exist_a_sequence (n : ℕ) (h : n ≥ 2) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  ∃ (a : Fin (n+1) → ℝ), (a 0 + a n = 0) ∧ (∀ i, |a i| ≤ 1) ∧ (∀ i : Fin n, |a i.succ - a i| = x i) :=
by
  sorry

end NUMINAMATH_GPT_exist_a_sequence_l1883_188329


namespace NUMINAMATH_GPT_at_least_two_equal_l1883_188382

-- Define the problem
theorem at_least_two_equal (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x^2 / y) + (y^2 / z) + (z^2 / x) = (x^2 / z) + (y^2 / x) + (z^2 / y)) :
  x = y ∨ y = z ∨ z = x := 
by 
  sorry

end NUMINAMATH_GPT_at_least_two_equal_l1883_188382


namespace NUMINAMATH_GPT_bounded_fx_range_a_l1883_188386

-- Part (1)
theorem bounded_fx :
  ∃ M > 0, ∀ x ∈ Set.Icc (-(1/2):ℝ) (1/2), abs (x / (x + 1)) ≤ M :=
by
  sorry

-- Part (2)
theorem range_a (a : ℝ) :
  (∀ x ≥ 0, abs (1 + a * (1/2)^x + (1/4)^x) ≤ 3) → -5 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_bounded_fx_range_a_l1883_188386


namespace NUMINAMATH_GPT_distance_between_cyclists_l1883_188372

def cyclist_distance (v1 : ℝ) (v2 : ℝ) (t : ℝ) : ℝ :=
  (v1 + v2) * t

theorem distance_between_cyclists :
  cyclist_distance 10 25 1.4285714285714286 = 50 := by
  sorry

end NUMINAMATH_GPT_distance_between_cyclists_l1883_188372


namespace NUMINAMATH_GPT_unique_function_l1883_188357

noncomputable def f : ℝ → ℝ := sorry

theorem unique_function 
  (h_f : ∀ x > 0, ∀ y > 0, f x * f y = 2 * f (x + y * f x)) : ∀ x > 0, f x = 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_function_l1883_188357


namespace NUMINAMATH_GPT_base_extension_1_kilometer_l1883_188358

-- Definition of the original triangle with hypotenuse length and inclination angle
def original_triangle (hypotenuse : ℝ) (angle : ℝ) : Prop :=
  hypotenuse = 1 ∧ angle = 20

-- Definition of the extension required for the new inclination angle
def extension_required (new_angle : ℝ) (extension : ℝ) : Prop :=
  new_angle = 10 ∧ extension = 1

-- The proof problem statement
theorem base_extension_1_kilometer :
  ∀ (hypotenuse : ℝ) (original_angle : ℝ) (new_angle : ℝ),
    original_triangle hypotenuse original_angle →
    new_angle = 10 →
    ∃ extension : ℝ, extension_required new_angle extension :=
by
  -- Sorry is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_base_extension_1_kilometer_l1883_188358


namespace NUMINAMATH_GPT_perfect_square_A_plus_B_plus1_l1883_188342

-- Definitions based on conditions
def A (m : ℕ) : ℕ := (10^2*m - 1) / 9
def B (m : ℕ) : ℕ := 4 * (10^m - 1) / 9

-- Proof statement
theorem perfect_square_A_plus_B_plus1 (m : ℕ) : A m + B m + 1 = ((10^m + 2) / 3)^2 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_A_plus_B_plus1_l1883_188342


namespace NUMINAMATH_GPT_calculate_expression_l1883_188330

noncomputable def f (x : ℝ) : ℝ :=
  (x^3 + 5 * x^2 + 6 * x) / (x^3 - x^2 - 2 * x)

def num_holes (f : ℝ → ℝ) : ℕ := 1 -- hole at x = -2
def num_vertical_asymptotes (f : ℝ → ℝ) : ℕ := 2 -- vertical asymptotes at x = 0 and x = 1
def num_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := 0 -- no horizontal asymptote
def num_oblique_asymptotes (f : ℝ → ℝ) : ℕ := 1 -- oblique asymptote at y = x + 4

theorem calculate_expression : num_holes f + 2 * num_vertical_asymptotes f + 3 * num_horizontal_asymptotes f + 4 * num_oblique_asymptotes f = 9 :=
by
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_calculate_expression_l1883_188330


namespace NUMINAMATH_GPT_all_statements_false_l1883_188356

theorem all_statements_false (r1 r2 : ℝ) (h1 : r1 ≠ r2) (h2 : r1 + r2 = 5) (h3 : r1 * r2 = 6) :
  ¬(|r1 + r2| > 6) ∧ ¬(3 < |r1 * r2| ∧ |r1 * r2| < 8) ∧ ¬(r1 < 0 ∧ r2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_all_statements_false_l1883_188356


namespace NUMINAMATH_GPT_scientific_notation_l1883_188318

theorem scientific_notation : 899000 = 8.99 * 10^5 := 
by {
  -- We start by recognizing that we need to express 899,000 in scientific notation.
  -- Placing the decimal point after the first non-zero digit yields 8.99.
  -- Count the number of places moved (5 places to the left).
  -- Thus, 899,000 in scientific notation is 8.99 * 10^5.
  sorry
}

end NUMINAMATH_GPT_scientific_notation_l1883_188318


namespace NUMINAMATH_GPT_log_sum_eq_two_l1883_188379

theorem log_sum_eq_two (log6_3 log6_4 : ℝ) (H1 : Real.logb 6 3 = log6_3) (H2 : Real.logb 6 4 = log6_4) : 
  log6_3 + log6_4 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_log_sum_eq_two_l1883_188379


namespace NUMINAMATH_GPT_cross_section_prism_in_sphere_l1883_188343

noncomputable def cross_section_area 
  (a R : ℝ) 
  (h1 : a > 0) 
  (h2 : R > 0) 
  (h3 : a < 2 * R) : ℝ :=
  a * Real.sqrt (4 * R^2 - a^2)

theorem cross_section_prism_in_sphere 
  (a R : ℝ) 
  (h1 : a > 0) 
  (h2 : R > 0) 
  (h3 : a < 2 * R) :
  cross_section_area a R h1 h2 h3 = a * Real.sqrt (4 * R^2 - a^2) := 
  by
    sorry

end NUMINAMATH_GPT_cross_section_prism_in_sphere_l1883_188343


namespace NUMINAMATH_GPT_no_valid_pairs_l1883_188310

theorem no_valid_pairs : ∀ (m n : ℕ), m ≥ n → m^2 - n^2 = 150 → false :=
by sorry

end NUMINAMATH_GPT_no_valid_pairs_l1883_188310


namespace NUMINAMATH_GPT_sequence_an_sequence_Tn_l1883_188394

theorem sequence_an (a : ℕ → ℕ) (S : ℕ → ℕ) (h : ∀ n, 2 * S n = a n ^ 2 + a n):
  ∀ n, a n = n :=
sorry

theorem sequence_Tn (b : ℕ → ℕ) (T : ℕ → ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, 2 * S n = a n ^ 2 + a n) (h2 : ∀ n, a n = n) (h3 : ∀ n, b n = 2^n * a n):
  ∀ n, T n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end NUMINAMATH_GPT_sequence_an_sequence_Tn_l1883_188394


namespace NUMINAMATH_GPT_number_of_valid_pairs_is_34_l1883_188378

noncomputable def countValidPairs : Nat :=
  let primes : List Nat := [2, 3, 5, 7, 11, 13]
  let nonprimes : List Nat := [1, 4, 6, 8, 9, 10, 12, 14, 15]
  let countForN (n : Nat) : Nat :=
    match n with
    | 2 => Nat.choose 8 1
    | 3 => Nat.choose 7 2
    | 5 => Nat.choose 5 4
    | _ => 0
  primes.map countForN |>.sum

theorem number_of_valid_pairs_is_34 : countValidPairs = 34 :=
  sorry

end NUMINAMATH_GPT_number_of_valid_pairs_is_34_l1883_188378


namespace NUMINAMATH_GPT_sum_of_possible_values_of_d_l1883_188345

def base_digits (n : ℕ) (b : ℕ) : ℕ := 
  if n = 0 then 1 else Nat.log (n + 1) b

theorem sum_of_possible_values_of_d :
  let min_val_7 := 1 * 7^3
  let max_val_7 := 6 * 7^3 + 6 * 7^2 + 6 * 7^1 + 6 * 7^0
  let min_val_10 := 343
  let max_val_10 := 2400
  let d1 := base_digits min_val_10 3
  let d2 := base_digits max_val_10 3
  d1 + d2 = 13 := sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_d_l1883_188345


namespace NUMINAMATH_GPT_unit_digit_25_pow_2010_sub_3_pow_2012_l1883_188348

theorem unit_digit_25_pow_2010_sub_3_pow_2012 :
  (25^2010 - 3^2012) % 10 = 4 :=
by 
  sorry

end NUMINAMATH_GPT_unit_digit_25_pow_2010_sub_3_pow_2012_l1883_188348


namespace NUMINAMATH_GPT_sin_cos_unique_solution_l1883_188349

theorem sin_cos_unique_solution (α : ℝ) (hα1 : 0 < α) (hα2 : α < (π / 2)) :
  ∃! x : ℝ, (Real.sin α) ^ x + (Real.cos α) ^ x = 1 :=
sorry

end NUMINAMATH_GPT_sin_cos_unique_solution_l1883_188349


namespace NUMINAMATH_GPT_isosceles_triangle_base_l1883_188363

noncomputable def base_of_isosceles_triangle
  (height_to_base : ℝ)
  (height_to_side : ℝ)
  (is_isosceles : Bool) : ℝ :=
if is_isosceles then 7.5 else 0

theorem isosceles_triangle_base :
  base_of_isosceles_triangle 5 6 true = 7.5 :=
by
  -- The proof would go here, just placeholder for now
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_l1883_188363
