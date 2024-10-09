import Mathlib

namespace factorization_sum_l1102_110298

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, x ^ 2 + 9 * x + 18 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x ^ 2 + 19 * x + 90 = (x + b) * (x + c)) :
  a + b + c = 22 := by
sorry

end factorization_sum_l1102_110298


namespace cats_not_eating_either_l1102_110213

theorem cats_not_eating_either (total_cats : ℕ) (cats_like_apples : ℕ) (cats_like_chicken : ℕ) (cats_like_both : ℕ) 
  (h1 : total_cats = 80)
  (h2 : cats_like_apples = 15)
  (h3 : cats_like_chicken = 60)
  (h4 : cats_like_both = 10) : 
  total_cats - (cats_like_apples + cats_like_chicken - cats_like_both) = 15 :=
by sorry

end cats_not_eating_either_l1102_110213


namespace product_multiplication_rule_l1102_110287

theorem product_multiplication_rule (a : ℝ) : (a * a^3)^2 = a^8 := 
by  
  -- The proof will apply the rule of product multiplication here
  sorry

end product_multiplication_rule_l1102_110287


namespace max_min_values_of_f_l1102_110280

noncomputable def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem max_min_values_of_f :
  (∀ x : ℝ, f x ≤ 2) ∧ (∃ x : ℝ, f x = 2) ∧
  (∀ x : ℝ, -2 ≤ f x) ∧ (∃ x : ℝ, f x = -2) :=
by 
  sorry

end max_min_values_of_f_l1102_110280


namespace expected_intersections_100gon_l1102_110244

noncomputable def expected_intersections : ℝ :=
  let n := 100
  let total_pairs := (n * (n - 3) / 2)
  total_pairs * (1/3)

theorem expected_intersections_100gon :
  expected_intersections = 4850 / 3 :=
by
  sorry

end expected_intersections_100gon_l1102_110244


namespace average_age_students_l1102_110211

theorem average_age_students 
  (total_students : ℕ)
  (group1 : ℕ)
  (group1_avg_age : ℕ)
  (group2 : ℕ)
  (group2_avg_age : ℕ)
  (student15_age : ℕ)
  (avg_age : ℕ) 
  (h1 : total_students = 15)
  (h2 : group1_avg_age = 14)
  (h3 : group2 = 8)
  (h4 : group2_avg_age = 16)
  (h5 : student15_age = 13)
  (h6 : avg_age = (84 + 128 + 13) / 15)
  (h7 : avg_age = 15) :
  group1 = 6 :=
by sorry

end average_age_students_l1102_110211


namespace spent_on_basil_seeds_l1102_110237

-- Define the variables and conditions
variables (S cost_soil num_plants price_per_plant net_profit total_revenue total_expenses : ℝ)
variables (h1 : cost_soil = 8)
variables (h2 : num_plants = 20)
variables (h3 : price_per_plant = 5)
variables (h4 : net_profit = 90)

-- Definition of total revenue as the multiplication of number of plants and price per plant
def revenue_eq : Prop := total_revenue = num_plants * price_per_plant

-- Definition of total expenses as the sum of soil cost and cost of basil seeds
def expenses_eq : Prop := total_expenses = cost_soil + S

-- Definition of net profit
def profit_eq : Prop := net_profit = total_revenue - total_expenses

-- The theorem to prove
theorem spent_on_basil_seeds : S = 2 :=
by
  -- Since we define variables and conditions as inputs,
  -- the proof itself is omitted as per instructions
  sorry

end spent_on_basil_seeds_l1102_110237


namespace wendy_made_money_l1102_110286

-- Given conditions
def price_per_bar : ℕ := 3
def total_bars : ℕ := 9
def bars_sold : ℕ := total_bars - 3

-- Statement to prove: Wendy made $18
theorem wendy_made_money : bars_sold * price_per_bar = 18 := by
  sorry

end wendy_made_money_l1102_110286


namespace polynomial_simplification_simplify_expression_evaluate_expression_l1102_110292

-- Prove that the correct simplification of 6mn - 2m - 3(m + 2mn) results in -5m.
theorem polynomial_simplification (m n : ℤ) :
  6 * m * n - 2 * m - 3 * (m + 2 * m * n) = -5 * m :=
by {
  sorry
}

-- Prove that simplifying a^2b^3 - 1/2(4ab + 6a^2b^3 - 1) + 2(ab - a^2b^3) results in -4a^2b^3 + 1/2.
theorem simplify_expression (a b : ℝ) :
  a^2 * b^3 - 1/2 * (4 * a * b + 6 * a^2 * b^3 - 1) + 2 * (a * b - a^2 * b^3) = -4 * a^2 * b^3 + 1/2 :=
by {
  sorry
}

-- Prove that evaluating the expression -4a^2b^3 + 1/2 at a = 1/2 and b = 3 results in -26.5
theorem evaluate_expression :
  -4 * (1/2) ^ 2 * 3 ^ 3 + 1/2 = -26.5 :=
by {
  sorry
}

end polynomial_simplification_simplify_expression_evaluate_expression_l1102_110292


namespace a_n_formula_l1102_110258

open Nat

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else n * (n + 1) / 2

theorem a_n_formula (n : ℕ) (h : n > 0) 
  (S_n : ℕ → ℕ)
  (hS : ∀ n, S_n n = (n + 2) / 3 * a_n n) 
  : a_n n = n * (n + 1) / 2 := sorry

end a_n_formula_l1102_110258


namespace probability_diagonals_intersect_l1102_110227

theorem probability_diagonals_intersect (n : ℕ) : 
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  probability = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  sorry

end probability_diagonals_intersect_l1102_110227


namespace seonho_original_money_l1102_110266

variable (X : ℝ)
variable (spent_snacks : ℝ := (1/4) * X)
variable (remaining_after_snacks : ℝ := X - spent_snacks)
variable (spent_food : ℝ := (2/3) * remaining_after_snacks)
variable (final_remaining : ℝ := remaining_after_snacks - spent_food)

theorem seonho_original_money :
  final_remaining = 2500 -> X = 10000 := by
  -- Proof goes here
  sorry

end seonho_original_money_l1102_110266


namespace hyperbola_equation_l1102_110242

theorem hyperbola_equation {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0)
    (hfocal : 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 5)
    (hslope : b / a = 1 / 8) :
    (∀ x y : ℝ, (x^2 / 4 - y^2 = 1) ↔ (x^2 / a^2 - y^2 / b^2 = 1)) :=
by
  -- Goals and conditions to handle proof
  sorry

end hyperbola_equation_l1102_110242


namespace integer_average_problem_l1102_110231

theorem integer_average_problem (a b c d : ℤ) (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
(h_max : max a (max b (max c d)) = 90) (h_min : min a (min b (min c d)) = 29) : 
(a + b + c + d) / 4 = 45 := 
sorry

end integer_average_problem_l1102_110231


namespace consecutive_sum_36_unique_l1102_110201

def is_consecutive_sum (a b n : ℕ) :=
  (0 < n) ∧ ((n ≥ 2) ∧ (b = a + n - 1) ∧ (2 * a + n - 1) * n = 72)

theorem consecutive_sum_36_unique :
  ∃! n, ∃ a b, is_consecutive_sum a b n :=
by
  sorry

end consecutive_sum_36_unique_l1102_110201


namespace inequality_y_lt_x_div_4_l1102_110221

open Real

/-- Problem statement:
Given x ∈ (0, π / 6) and y ∈ (0, π / 6), and x * tan y = 2 * (1 - cos x),
prove that y < x / 4.
-/
theorem inequality_y_lt_x_div_4
  (x y : ℝ)
  (hx : 0 < x ∧ x < π / 6)
  (hy : 0 < y ∧ y < π / 6)
  (h : x * tan y = 2 * (1 - cos x)) :
  y < x / 4 := sorry

end inequality_y_lt_x_div_4_l1102_110221


namespace carl_additional_marbles_l1102_110220

def initial_marbles := 12
def lost_marbles := initial_marbles / 2
def additional_marbles_from_mom := 25
def marbles_in_jar_after_game := 41

theorem carl_additional_marbles :
  (marbles_in_jar_after_game - additional_marbles_from_mom) + lost_marbles - initial_marbles = 10 :=
by
  sorry

end carl_additional_marbles_l1102_110220


namespace person_b_days_work_alone_l1102_110235

theorem person_b_days_work_alone (B : ℕ) (h1 : (1 : ℚ) / 40 + 1 / B = 1 / 24) : B = 60 := 
by
  sorry

end person_b_days_work_alone_l1102_110235


namespace find_sums_of_integers_l1102_110217

theorem find_sums_of_integers (x y : ℤ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_prod_sum : x * y + x + y = 125) (h_rel_prime : Int.gcd x y = 1) (h_lt_x : x < 30) (h_lt_y : y < 30) : 
  (x + y = 25) ∨ (x + y = 23) ∨ (x + y = 21) := 
by 
  sorry

end find_sums_of_integers_l1102_110217


namespace number_of_ordered_triples_l1102_110233

noncomputable def count_triples : Nat := 50

theorem number_of_ordered_triples 
    (x y z : Nat)
    (hx : x > 0)
    (hy : y > 0)
    (hz : z > 0)
    (H1 : Nat.lcm x y = 500)
    (H2 : Nat.lcm y z = 1000)
    (H3 : Nat.lcm z x = 1000) :
    ∃ (n : Nat), n = count_triples := 
by
    use 50
    sorry

end number_of_ordered_triples_l1102_110233


namespace ratio_joe_sara_l1102_110238

variables (S J : ℕ) (k : ℕ)

-- Conditions
#check J + S = 120
#check J = k * S + 6
#check J = 82

-- The goal is to prove the ratio J / S = 41 / 19
theorem ratio_joe_sara (h1 : J + S = 120) (h2 : J = k * S + 6) (h3 : J = 82) : J / S = 41 / 19 :=
sorry

end ratio_joe_sara_l1102_110238


namespace prime_number_five_greater_than_perfect_square_l1102_110284

theorem prime_number_five_greater_than_perfect_square 
(p x : ℤ) (h1 : p - 5 = x^2) (h2 : p + 9 = (x + 1)^2) : 
  p = 41 :=
sorry

end prime_number_five_greater_than_perfect_square_l1102_110284


namespace find_sum_of_a_b_c_l1102_110230

theorem find_sum_of_a_b_c (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
(h4 : (a + b + c) ^ 3 - a ^ 3 - b ^ 3 - c ^ 3 = 210) : a + b + c = 11 :=
sorry

end find_sum_of_a_b_c_l1102_110230


namespace no_two_digit_numbers_satisfy_condition_l1102_110281

theorem no_two_digit_numbers_satisfy_condition :
  ¬ ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
  (10 * a + b) * (10 * c + d) = 1000 * a + 100 * b + 10 * c + d :=
by
  sorry

end no_two_digit_numbers_satisfy_condition_l1102_110281


namespace divide_circle_into_parts_l1102_110241

theorem divide_circle_into_parts : 
    ∃ (divide : ℕ → ℕ), 
        (divide 3 = 4 ∧ divide 3 = 5 ∧ divide 3 = 6 ∧ divide 3 = 7) :=
by
  -- This illustrates that we require a proof to show that for 3 straight cuts ('n = 3'), 
  -- we can achieve 4, 5, 6, and 7 segments in different settings (circle with strategic line placements).
  sorry

end divide_circle_into_parts_l1102_110241


namespace ratio_greater_than_one_ratio_greater_than_one_neg_l1102_110269

theorem ratio_greater_than_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b > 1) : a > b :=
by
  sorry

theorem ratio_greater_than_one_neg (a b : ℝ) (h1 : a < 0) (h2 : b < 0) (h3 : a / b > 1) : a < b :=
by
  sorry

end ratio_greater_than_one_ratio_greater_than_one_neg_l1102_110269


namespace math_proof_l1102_110202

noncomputable def side_length_of_smaller_square (d e f : ℕ) : ℝ :=
  (d - Real.sqrt e) / f

def are_positive_integers (d e f : ℕ) : Prop := d > 0 ∧ e > 0 ∧ f > 0
def is_not_divisible_by_square_of_any_prime (e : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p * p ∣ e)

def proof_problem : Prop :=
  ∃ (d e f : ℕ),
    are_positive_integers d e f ∧
    is_not_divisible_by_square_of_any_prime e ∧
    side_length_of_smaller_square d e f = (4 - Real.sqrt 10) / 3 ∧
    d + e + f = 17

theorem math_proof : proof_problem := sorry

end math_proof_l1102_110202


namespace train_crossing_time_l1102_110294

noncomputable def train_length : ℕ := 150
noncomputable def bridge_length : ℕ := 150
noncomputable def train_speed_kmph : ℕ := 36

noncomputable def kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

noncomputable def train_speed_mps : ℕ := kmph_to_mps train_speed_kmph

noncomputable def total_distance : ℕ := train_length + bridge_length

noncomputable def crossing_time_in_seconds (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem train_crossing_time :
  crossing_time_in_seconds total_distance train_speed_mps = 30 :=
by
  sorry

end train_crossing_time_l1102_110294


namespace find_a_value_l1102_110282

def quadratic_vertex_condition (a : ℚ) : Prop :=
  ∀ x y : ℚ,
  (x = 2) → (y = 5) →
  a * (x - 2)^2 + 5 = y

def quadratic_passing_point_condition (a : ℚ) : Prop :=
  ∀ x y : ℚ,
  (x = -1) → (y = -20) →
  a * (x - 2)^2 + 5 = y

theorem find_a_value : ∃ a : ℚ, quadratic_vertex_condition a ∧ quadratic_passing_point_condition a ∧ a = (-25)/9 := 
by 
  sorry

end find_a_value_l1102_110282


namespace tangent_line_parallel_range_a_l1102_110223

noncomputable def f (a x : ℝ) : ℝ :=
  Real.log x + 1/2 * x^2 + a * x

theorem tangent_line_parallel_range_a (a : ℝ) :
  (∃ x > 0, deriv (f a) x = 3) ↔ a ≤ 1 :=
by
  sorry

end tangent_line_parallel_range_a_l1102_110223


namespace arith_seq_S13_value_l1102_110218

variable {α : Type*} [LinearOrderedField α]

-- Definitions related to an arithmetic sequence
structure ArithSeq (α : Type*) :=
  (a : ℕ → α) -- the sequence itself
  (sum_first_n_terms : ℕ → α) -- sum of the first n terms

def is_arith_seq (seq : ArithSeq α) :=
  ∀ (n : ℕ), seq.a (n + 1) - seq.a n = seq.a 2 - seq.a 1

-- Our conditions
noncomputable def a5 (seq : ArithSeq α) := seq.a 5
noncomputable def a7 (seq : ArithSeq α) := seq.a 7
noncomputable def a9 (seq : ArithSeq α) := seq.a 9
noncomputable def S13 (seq : ArithSeq α) := seq.sum_first_n_terms 13

-- Problem statement
theorem arith_seq_S13_value (seq : ArithSeq α) 
  (h_arith_seq : is_arith_seq seq)
  (h_condition : 2 * (a5 seq) + 3 * (a7 seq) + 2 * (a9 seq) = 14) : 
  S13 seq = 26 := 
  sorry

end arith_seq_S13_value_l1102_110218


namespace min_value_am_hm_inequality_l1102_110265

theorem min_value_am_hm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
    (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
sorry

end min_value_am_hm_inequality_l1102_110265


namespace sum_mod_20_l1102_110267

/-- Define the elements that are summed. -/
def elements : List ℤ := [82, 83, 84, 85, 86, 87, 88, 89]

/-- The problem statement to prove. -/
theorem sum_mod_20 : (elements.sum % 20) = 15 := by
  sorry

end sum_mod_20_l1102_110267


namespace train_length_l1102_110208

theorem train_length :
  (∃ L : ℕ, (L / 15) = (L + 800) / 45) → L = 400 :=
by
  sorry

end train_length_l1102_110208


namespace n_mod_9_eq_6_l1102_110261

def n : ℕ := 2 + 333 + 44444 + 555555 + 6666666 + 77777777 + 888888888 + 9999999999

theorem n_mod_9_eq_6 : n % 9 = 6 :=
by
  sorry

end n_mod_9_eq_6_l1102_110261


namespace binom_12_6_eq_924_l1102_110259

theorem binom_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binom_12_6_eq_924_l1102_110259


namespace circle_center_l1102_110204

theorem circle_center (x y : ℝ) :
  x^2 + y^2 - 4 * x - 2 * y - 5 = 0 → (x - 2)^2 + (y - 1)^2 = 10 :=
by sorry

end circle_center_l1102_110204


namespace fraction_multiplication_l1102_110216

noncomputable def a : ℚ := 5 / 8
noncomputable def b : ℚ := 7 / 12
noncomputable def c : ℚ := 3 / 7
noncomputable def n : ℚ := 1350

theorem fraction_multiplication : a * b * c * n = 210.9375 := by
  sorry

end fraction_multiplication_l1102_110216


namespace album_cost_l1102_110225

-- Definitions for given conditions
def M (X : ℕ) : ℕ := X - 2
def K (X : ℕ) : ℕ := X - 34
def F (X : ℕ) : ℕ := X - 35

-- We need to prove that X = 35
theorem album_cost : ∃ X : ℕ, (M X) + (K X) + (F X) < X ∧ X = 35 :=
by
  sorry -- Proof not required.

end album_cost_l1102_110225


namespace complex_z_modulus_l1102_110249

open Complex

theorem complex_z_modulus (z : ℂ) (h1 : (z + 2 * I).re = z + 2 * I) (h2 : (z / (2 - I)).re = z / (2 - I)) :
  (z = 4 - 2 * I) ∧ abs (z / (1 + I)) = Real.sqrt 10 := by
  sorry

end complex_z_modulus_l1102_110249


namespace problem_1_problem_2_problem_3_l1102_110263

section Problem

-- Initial conditions
variable (a : ℕ → ℝ) (t m : ℝ)
def a_1 : ℝ := 3
def a_n (n : ℕ) (h : 2 ≤ n) : ℝ := 2 * a (n - 1) + (t + 1) * 2^n + 3 * m + t

-- Problem 1:
theorem problem_1 (h : t = 0) (h' : m = 0) :
  ∃ d, ∀ n, 2 ≤ n → (a n / 2^n) = (a (n - 1) / 2^(n-1)) + d := sorry

-- Problem 2:
theorem problem_2 (h : t = -1) (h' : m = 4/3) :
  ∃ r, ∀ n, 2 ≤ n → a n + 3 = r * (a (n - 1) + 3) := sorry

-- Problem 3:
theorem problem_3 (h : t = 0) (h' : m = 1) :
  (∀ n, 1 ≤ n → a n = (n + 2) * 2^n - 3) ∧
  (∃ S : ℕ → ℝ, ∀ n, S n = (n + 1) * 2^(n + 1) - 2 - 3 * n) := sorry

end Problem

end problem_1_problem_2_problem_3_l1102_110263


namespace find_length_d_l1102_110250

theorem find_length_d :
  ∀ (A B C P: Type) (AB AC BC : ℝ) (d : ℝ),
    AB = 425 ∧ BC = 450 ∧ AC = 510 ∧
    (∃ (JG FI HE : ℝ), JG = FI ∧ FI = HE ∧ JG = d ∧ 
      (d / BC + d / AC + d / AB = 2)) 
    → d = 306 :=
by {
  sorry
}

end find_length_d_l1102_110250


namespace find_k_l1102_110247

theorem find_k (x : ℝ) (a h k : ℝ) (h1 : 9 * x^2 - 12 * x = a * (x - h)^2 + k) : k = -4 := by
  sorry

end find_k_l1102_110247


namespace number_of_levels_l1102_110224

-- Definitions of the conditions
def blocks_per_step : ℕ := 3
def steps_per_level : ℕ := 8
def total_blocks_climbed : ℕ := 96

-- The theorem to prove
theorem number_of_levels : (total_blocks_climbed / blocks_per_step) / steps_per_level = 4 := by
  sorry

end number_of_levels_l1102_110224


namespace ratio_cars_to_dogs_is_two_l1102_110253

-- Definitions of the conditions
def initial_dogs : ℕ := 90
def initial_cars : ℕ := initial_dogs / 3
def additional_cars : ℕ := 210
def current_dogs : ℕ := 120
def current_cars : ℕ := initial_cars + additional_cars

-- The statement to be proven
theorem ratio_cars_to_dogs_is_two :
  (current_cars : ℚ) / (current_dogs : ℚ) = 2 := by
  sorry

end ratio_cars_to_dogs_is_two_l1102_110253


namespace matrix_calculation_l1102_110248

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![0, 2]]

def B15_minus_3B14 : Matrix (Fin 2) (Fin 2) ℝ :=
  B^15 - 3 * B^14

theorem matrix_calculation : B15_minus_3B14 = ![![0, 4], ![0, -1]] := by
  sorry

end matrix_calculation_l1102_110248


namespace circle_tangent_y_eq_2_center_on_y_axis_radius_1_l1102_110295

theorem circle_tangent_y_eq_2_center_on_y_axis_radius_1 :
  ∃ (y0 : ℝ), (∀ x y : ℝ, (x - 0)^2 + (y - y0)^2 = 1 ↔ y = y0 + 1 ∨ y = y0 - 1) := by
  sorry

end circle_tangent_y_eq_2_center_on_y_axis_radius_1_l1102_110295


namespace sufficient_cond_l1102_110251

theorem sufficient_cond (x : ℝ) (h : 1/x > 2) : x < 1/2 := 
by {
  sorry 
}

end sufficient_cond_l1102_110251


namespace large_square_min_side_and_R_max_area_l1102_110222

-- Define the conditions
variable (s : ℝ) -- the side length of the larger square
variable (rect_1_side1 rect_1_side2 : ℝ) -- sides of the first rectangle
variable (square_side : ℝ) -- side of the inscribed square
variable (R_area : ℝ) -- area of the rectangle R

-- The known dimensions
axiom h1 : rect_1_side1 = 2
axiom h2 : rect_1_side2 = 4
axiom h3 : square_side = 2
axiom h4 : ∀ x y : ℝ, x > 0 → y > 0 → R_area = x * y -- non-overlapping condition

-- Define the result to be proved
theorem large_square_min_side_and_R_max_area 
  (h_r_fit_1 : rect_1_side1 + square_side ≤ s)
  (h_r_fit_2 : rect_1_side2 + square_side ≤ s)
  (h_R_max_area : R_area = 4)
  : s = 4 ∧ R_area = 4 := 
by 
  sorry

end large_square_min_side_and_R_max_area_l1102_110222


namespace greatest_product_l1102_110252

theorem greatest_product (x : ℤ) (h : x + (2024 - x) = 2024) : 
  2024 * x - x^2 ≤ 1024144 :=
by
  sorry

end greatest_product_l1102_110252


namespace base_of_first_term_l1102_110274

-- Define the necessary conditions
def equation (x s : ℝ) : Prop :=
  x^16 * 25^s = 5 * 10^16

-- The proof goal
theorem base_of_first_term (x s : ℝ) (h : equation x s) : x = 2 / 5 :=
by
  sorry

end base_of_first_term_l1102_110274


namespace xiao_ming_total_score_l1102_110212

theorem xiao_ming_total_score :
  ∃ (a_1 a_2 a_3 a_4 a_5 : ℕ), 
  a_1 < a_2 ∧ 
  a_2 < a_3 ∧ 
  a_3 < a_4 ∧ 
  a_4 < a_5 ∧ 
  a_1 + a_2 = 10 ∧ 
  a_4 + a_5 = 18 ∧ 
  a_1 + a_2 + a_3 + a_4 + a_5 = 35 :=
by
  sorry

end xiao_ming_total_score_l1102_110212


namespace total_campers_correct_l1102_110246

-- Definitions for the conditions
def campers_morning : ℕ := 15
def campers_afternoon : ℕ := 17

-- Define total campers, question is to prove it is indeed 32
def total_campers : ℕ := campers_morning + campers_afternoon

theorem total_campers_correct : total_campers = 32 :=
by
  -- Proof omitted
  sorry

end total_campers_correct_l1102_110246


namespace problem_statement_l1102_110289

variable (a b c : ℝ)

theorem problem_statement 
  (h1 : ab / (a + b) = 1 / 3)
  (h2 : bc / (b + c) = 1 / 4)
  (h3 : ca / (c + a) = 1 / 5) :
  abc / (ab + bc + ca) = 1 / 6 := 
sorry

end problem_statement_l1102_110289


namespace func_passes_through_fixed_point_l1102_110215

theorem func_passes_through_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  a^(2 * (1 / 2) - 1) = 1 :=
by
  sorry

end func_passes_through_fixed_point_l1102_110215


namespace average_is_5x_minus_10_implies_x_is_50_l1102_110243

theorem average_is_5x_minus_10_implies_x_is_50 (x : ℝ) 
  (h : (1 / 3) * ((3 * x + 8) + (7 * x + 3) + (4 * x + 9)) = 5 * x - 10) : 
  x = 50 :=
by
  sorry

end average_is_5x_minus_10_implies_x_is_50_l1102_110243


namespace hyperbola_eccentricity_l1102_110268

theorem hyperbola_eccentricity : 
  let a := 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  e = Real.sqrt 5 / 2 := 
by
  let a := 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  sorry

end hyperbola_eccentricity_l1102_110268


namespace billy_sleep_total_l1102_110291

theorem billy_sleep_total :
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  day1 + day2 + day3 + day4 = 30 :=
by
  -- Definitions
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  -- Assertion
  have h : day1 + day2 + day3 + day4 = 30 := sorry
  exact h

end billy_sleep_total_l1102_110291


namespace worker_days_total_l1102_110273

theorem worker_days_total
  (W I : ℕ)
  (hw : 20 * W - 3 * I = 280)
  (hi : I = 40) :
  W + I = 60 :=
by
  sorry

end worker_days_total_l1102_110273


namespace leap_year_1996_l1102_110299

def divisible_by (n m : ℕ) : Prop := m % n = 0

def is_leap_year (y : ℕ) : Prop :=
  (divisible_by 4 y ∧ ¬divisible_by 100 y) ∨ divisible_by 400 y

theorem leap_year_1996 : is_leap_year 1996 :=
by
  sorry

end leap_year_1996_l1102_110299


namespace divides_expression_l1102_110293

theorem divides_expression (n : ℕ) (h1 : n ≥ 3) 
  (h2 : Prime (4 * n + 1)) : (4 * n + 1) ∣ (n^(2 * n) - 1) :=
by
  sorry

end divides_expression_l1102_110293


namespace white_patches_count_l1102_110297

-- Definitions based on the provided conditions
def total_patches : ℕ := 32
def white_borders_black (x : ℕ) : ℕ := 3 * x
def black_borders_white (x : ℕ) : ℕ := 5 * (total_patches - x)

-- The theorem we need to prove
theorem white_patches_count :
  ∃ x : ℕ, white_borders_black x = black_borders_white x ∧ x = 20 :=
by 
  sorry

end white_patches_count_l1102_110297


namespace f3_is_ideal_function_l1102_110236

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + f (-x) = 0

def is_strictly_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0

noncomputable def f3 (x : ℝ) : ℝ :=
  if x < 0 then x ^ 2 else -x ^ 2

theorem f3_is_ideal_function : is_odd_function f3 ∧ is_strictly_decreasing f3 := 
  sorry

end f3_is_ideal_function_l1102_110236


namespace cube_value_proportional_l1102_110214

theorem cube_value_proportional (side_length1 side_length2 : ℝ) (volume1 volume2 : ℝ) (value1 value2 : ℝ) :
  side_length1 = 4 → volume1 = side_length1 ^ 3 → value1 = 500 →
  side_length2 = 6 → volume2 = side_length2 ^ 3 → value2 = value1 * (volume2 / volume1) →
  value2 = 1688 :=
by
  sorry

end cube_value_proportional_l1102_110214


namespace find_number_l1102_110270

theorem find_number (a b x : ℝ) (H1 : 2 * a = x * b) (H2 : a * b ≠ 0) (H3 : (a / 3) / (b / 2) = 1) : x = 3 :=
by
  sorry

end find_number_l1102_110270


namespace oranges_purchase_cost_l1102_110257

/-- 
Oranges are sold at a rate of $3$ per three pounds.
If a customer buys 18 pounds and receives a discount of $5\%$ for buying more than 15 pounds,
prove that the total amount the customer pays is $17.10.
-/
theorem oranges_purchase_cost (rate : ℕ) (base_weight : ℕ) (discount_rate : ℚ)
  (total_weight : ℕ) (discount_threshold : ℕ) (final_cost : ℚ) :
  rate = 3 → base_weight = 3 → discount_rate = 0.05 → 
  total_weight = 18 → discount_threshold = 15 → final_cost = 17.10 := by
  sorry

end oranges_purchase_cost_l1102_110257


namespace even_sum_probability_l1102_110245

theorem even_sum_probability :
  let p_even_w1 := 3 / 4
  let p_even_w2 := 1 / 2
  let p_even_w3 := 1 / 4
  let p_odd_w1 := 1 - p_even_w1
  let p_odd_w2 := 1 - p_even_w2
  let p_odd_w3 := 1 - p_even_w3
  (p_even_w1 * p_even_w2 * p_even_w3) +
  (p_odd_w1 * p_odd_w2 * p_even_w3) +
  (p_odd_w1 * p_even_w2 * p_odd_w3) +
  (p_even_w1 * p_odd_w2 * p_odd_w3) = 1 / 2 := by
    sorry

end even_sum_probability_l1102_110245


namespace max_distance_origin_perpendicular_bisector_l1102_110276

theorem max_distance_origin_perpendicular_bisector :
  ∀ (k m : ℝ), k ≠ 0 → 
  (|m| = Real.sqrt (1 + k^2)) → 
  ∃ (d : ℝ), d = 4 / 3 :=
by
  sorry

end max_distance_origin_perpendicular_bisector_l1102_110276


namespace min_y_value_l1102_110277

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16 * x + 50 * y + 64) : y ≥ 0 :=
sorry

end min_y_value_l1102_110277


namespace maximum_marks_l1102_110264

theorem maximum_marks (M : ℝ) :
  (0.45 * M = 80) → (M = 180) :=
by
  sorry

end maximum_marks_l1102_110264


namespace prime_p_range_l1102_110296

open Classical

variable {p : ℤ} (hp_prime : Prime p)

def is_integer_root (a b c : ℤ) := 
  ∃ x y : ℤ, x * y = c ∧ x + y = -b

theorem prime_p_range (hp_roots : is_integer_root 1 p (-500 * p)) : 1 < p ∧ p ≤ 10 :=
by
  sorry

end prime_p_range_l1102_110296


namespace range_of_2a_minus_b_l1102_110279

theorem range_of_2a_minus_b (a b : ℝ) (h1 : 2 < a) (h2 : a < 3) (h3 : 1 < b) (h4 : b < 2) :
  2 < 2 * a - b ∧ 2 * a - b < 5 := 
sorry

end range_of_2a_minus_b_l1102_110279


namespace exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles_l1102_110255

theorem exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles
    (a b c d α β γ δ: ℝ) (h_conv: a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c)
    (h_angles: α < β + γ + δ ∧ β < α + γ + δ ∧ γ < α + β + δ ∧ δ < α + β + γ) :
    ∃ (a' b' c' d' α' β' γ' δ' : ℝ),
      (a' / b' = α / β) ∧ (b' / c' = β / γ) ∧ (c' / d' = γ / δ) ∧ (d' / a' = δ / α) ∧
      (a' < b' + c' + d') ∧ (b' < a' + c' + d') ∧ (c' < a' + b' + d') ∧ (d' < a' + b' + c') ∧
      (α' < β' + γ' + δ') ∧ (β' < α' + γ' + δ') ∧ (γ' < α' + β' + δ') ∧ (δ' < α' + β' + γ') :=
  sorry

end exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles_l1102_110255


namespace dogsled_race_time_difference_l1102_110229

theorem dogsled_race_time_difference :
  let D := 300  -- Distance in miles
  let V_W := 20  -- Team W's average speed in mph
  let V_A := 25  -- Team A's average speed in mph
  let T_W := D / V_W  -- Time taken by Team W
  let T_A := D / V_A  -- Time taken by Team A
  T_W - T_A = 3 :=
by
  let D := 300  -- Distance in miles
  let V_W := 20  -- Team W's average speed in mph
  let V_A := 25  -- Team A's average speed in mph
  let T_W := D / V_W  -- Time taken by Team W
  let T_A := D / V_A  -- Time taken by Team A
  sorry

end dogsled_race_time_difference_l1102_110229


namespace find_p_probability_of_match_ending_after_4_games_l1102_110254

variables (p : ℚ)

-- Conditions translated to Lean definitions
def probability_first_game_win : ℚ := 1 / 2

def probability_consecutive_games_win : ℚ := 5 / 16

-- Definitions based on conditions
def prob_second_game_win_if_won_first : ℚ := (1 + p) / 2

def prob_winning_consecutive_games (prob_first_game : ℚ) (prob_second_game_if_won_first : ℚ) : ℚ :=
prob_first_game * prob_second_game_if_won_first

-- Main Theorem Statements to be proved
theorem find_p 
    (h_eq : prob_winning_consecutive_games probability_first_game_win (prob_second_game_win_if_won_first p) = probability_consecutive_games_win) :
    p = 1 / 4 :=
sorry

-- Given p = 1/4, probabilities for each scenario the match ends after 4 games
def prob_scenario1 : ℚ := (1 / 2) * ((1 + 1/4) / 2) * ((1 - 1/4) / 2) * ((1 - 1/4) / 2)
def prob_scenario2 : ℚ := (1 / 2) * ((1 - 1/4) / 2) * ((1 - 1/4) / 2) * ((1 + 1/4) / 2)
def prob_scenario3 : ℚ := (1 / 2) * ((1 - 1/4) / 2) * ((1 + 1/4) / 2) * ((1 + 1/4) / 2)

def total_probability_ending_in_4_games : ℚ :=
2 * (prob_scenario1 + prob_scenario2 + prob_scenario3)

theorem probability_of_match_ending_after_4_games (hp : p = 1 / 4) :
    total_probability_ending_in_4_games = 165 / 512 :=
sorry

end find_p_probability_of_match_ending_after_4_games_l1102_110254


namespace bridge_length_problem_l1102_110240

noncomputable def length_of_bridge (num_carriages : ℕ) (length_carriage : ℕ) (length_engine : ℕ) (speed_kmph : ℕ) (crossing_time_min : ℕ) : ℝ :=
  let total_train_length := (num_carriages + 1) * length_carriage
  let speed_mps := (speed_kmph * 1000) / 3600
  let crossing_time_secs := crossing_time_min * 60
  let total_distance := speed_mps * crossing_time_secs
  let bridge_length := total_distance - total_train_length
  bridge_length

theorem bridge_length_problem :
  length_of_bridge 24 60 60 60 5 = 3501 :=
by
  sorry

end bridge_length_problem_l1102_110240


namespace number_of_other_numbers_l1102_110285

-- Definitions of the conditions
def avg_five_numbers (S : ℕ) : Prop := S / 5 = 20
def sum_three_numbers (S2 : ℕ) : Prop := 100 = S2 + 48
def avg_other_numbers (N S2 : ℕ) : Prop := S2 / N = 26

-- Theorem statement
theorem number_of_other_numbers (S S2 N : ℕ) 
  (h1 : avg_five_numbers S) 
  (h2 : sum_three_numbers S2) 
  (h3 : avg_other_numbers N S2) : 
  N = 2 := 
  sorry

end number_of_other_numbers_l1102_110285


namespace intersection_is_correct_l1102_110206

-- Conditions definitions
def setA : Set ℝ := {x | 2 < x ∧ x < 8}
def setB : Set ℝ := {x | x^2 - 5 * x - 6 ≤ 0}

-- Intersection definition
def intersection : Set ℝ := {x | 2 < x ∧ x ≤ 6}

-- Theorem statement
theorem intersection_is_correct : setA ∩ setB = intersection := 
by
  sorry

end intersection_is_correct_l1102_110206


namespace calculate_X_l1102_110210

theorem calculate_X
  (top_seg1 : ℕ) (top_seg2 : ℕ) (X : ℕ)
  (vert_seg : ℕ)
  (bottom_seg1 : ℕ) (bottom_seg2 : ℕ) (bottom_seg3 : ℕ)
  (h1 : top_seg1 = 3) (h2 : top_seg2 = 2)
  (h3 : vert_seg = 4)
  (h4 : bottom_seg1 = 4) (h5 : bottom_seg2 = 2) (h6 : bottom_seg3 = 5)
  (h_eq : 5 + X = 11) :
  X = 6 :=
by
  -- Proof is omitted as per instructions.
  sorry

end calculate_X_l1102_110210


namespace probability_fly_reaches_8_10_l1102_110256

theorem probability_fly_reaches_8_10 :
  let total_steps := 2^18
  let right_up_combinations := Nat.choose 18 8
  (right_up_combinations / total_steps : ℚ) = Nat.choose 18 8 / 2^18 := 
sorry

end probability_fly_reaches_8_10_l1102_110256


namespace abc_over_sum_leq_four_thirds_l1102_110209

theorem abc_over_sum_leq_four_thirds (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) 
  (h_a_leq_2 : a ≤ 2) (h_b_leq_2 : b ≤ 2) (h_c_leq_2 : c ≤ 2) :
  (abc / (a + b + c) ≤ 4/3) :=
by
  sorry

end abc_over_sum_leq_four_thirds_l1102_110209


namespace toy_factory_days_per_week_l1102_110283

theorem toy_factory_days_per_week (toys_per_week : ℕ) (toys_per_day : ℕ) (h₁ : toys_per_week = 4560) (h₂ : toys_per_day = 1140) : toys_per_week / toys_per_day = 4 := 
by {
  -- Proof to be provided
  sorry
}

end toy_factory_days_per_week_l1102_110283


namespace probability_letter_in_MATHEMATICS_l1102_110219

theorem probability_letter_in_MATHEMATICS :
  let alphabet := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  let mathematics := ['M', 'A', 'T', 'H', 'E', 'I', 'C', 'S']
  (mathematics.length : ℚ) / (alphabet.length : ℚ) = 4 / 13 :=
by
  sorry

end probability_letter_in_MATHEMATICS_l1102_110219


namespace correct_sum_of_integers_l1102_110275

theorem correct_sum_of_integers
  (x y : ℕ)
  (h1 : x - y = 5)
  (h2 : x * y = 84) :
  x + y = 19 :=
sorry

end correct_sum_of_integers_l1102_110275


namespace calculate_expression_l1102_110272

def seq (k : Nat) : Nat := 2^k + 3^k

def product_seq : Nat :=
  (2 + 3) * (2^3 + 3^3) * (2^6 + 3^6) * (2^12 + 3^12) * (2^24 + 3^24)

theorem calculate_expression :
  product_seq = (3^47 - 2^47) :=
sorry

end calculate_expression_l1102_110272


namespace find_4_oplus_2_l1102_110203

def operation (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem find_4_oplus_2 : operation 4 2 = 26 :=
by
  sorry

end find_4_oplus_2_l1102_110203


namespace total_items_and_cost_per_pet_l1102_110278

theorem total_items_and_cost_per_pet
  (treats_Jane : ℕ)
  (treats_Wanda : ℕ := treats_Jane / 2)
  (bread_Jane : ℕ := (3 * treats_Jane) / 4)
  (bread_Wanda : ℕ := 90)
  (bread_Carla : ℕ := 40)
  (treats_Carla : ℕ := 5 * bread_Carla / 2)
  (items_Peter : ℕ := 140)
  (treats_Peter : ℕ := items_Peter / 3)
  (bread_Peter : ℕ := 2 * treats_Peter)
  (x y z : ℕ) :
  (∀ B : ℕ, B = bread_Jane + bread_Wanda + bread_Carla + bread_Peter) ∧
  (∀ T : ℕ, T = treats_Jane + treats_Wanda + treats_Carla + treats_Peter) ∧
  (∀ Total : ℕ, Total = (bread_Jane + bread_Wanda + bread_Carla + bread_Peter) + (treats_Jane + treats_Wanda + treats_Carla + treats_Peter)) ∧
  (∀ ExpectedTotal : ℕ, ExpectedTotal = 427) ∧
  (∀ Cost : ℕ, Cost = (bread_Jane + bread_Wanda + bread_Carla + bread_Peter) * x + (treats_Jane + treats_Wanda + treats_Carla + treats_Peter) * y) ∧
  (∀ CostPerPet : ℕ, CostPerPet = Cost / z) ∧
  (B + T = 427) ∧
  ((Cost / z) = (235 * x + 192 * y) / z)
:=
  by
  sorry

end total_items_and_cost_per_pet_l1102_110278


namespace initial_number_of_apples_l1102_110226

-- Definitions based on the conditions
def number_of_trees : ℕ := 3
def apples_picked_per_tree : ℕ := 8
def apples_left_on_trees : ℕ := 9

-- The theorem to prove
theorem initial_number_of_apples (t: ℕ := number_of_trees) (a: ℕ := apples_picked_per_tree) (l: ℕ := apples_left_on_trees) : t * a + l = 33 :=
by
  sorry

end initial_number_of_apples_l1102_110226


namespace find_number_l1102_110239

theorem find_number (x : ℝ) (h : 2 = 0.04 * x) : x = 50 := 
sorry

end find_number_l1102_110239


namespace remainder_when_divided_l1102_110260

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l1102_110260


namespace exactly_one_valid_N_l1102_110228

def four_digit_number (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

def condition (N x a : ℕ) : Prop := 
  N = 1000 * a + x ∧ x = N / 7

theorem exactly_one_valid_N : 
  ∃! N : ℕ, ∃ x a : ℕ, four_digit_number N ∧ condition N x a :=
sorry

end exactly_one_valid_N_l1102_110228


namespace prob_at_least_two_diamonds_or_aces_in_three_draws_l1102_110290

noncomputable def prob_at_least_two_diamonds_or_aces: ℚ :=
  580 / 2197

def cards_drawn (draws: ℕ) : Prop :=
  draws = 3

def cards_either_diamonds_or_aces: ℕ :=
  16

theorem prob_at_least_two_diamonds_or_aces_in_three_draws:
  cards_drawn 3 →
  cards_either_diamonds_or_aces = 16 →
  prob_at_least_two_diamonds_or_aces = 580 / 2197 :=
  by
  intros
  sorry

end prob_at_least_two_diamonds_or_aces_in_three_draws_l1102_110290


namespace family_visit_cost_is_55_l1102_110262

def num_children := 4
def num_parents := 2
def num_grandmother := 1
def num_people := num_children + num_parents + num_grandmother

def entrance_ticket_cost := 5
def attraction_ticket_cost_kid := 2
def attraction_ticket_cost_adult := 4

def entrance_total_cost := num_people * entrance_ticket_cost
def attraction_total_cost_kids := num_children * attraction_ticket_cost_kid
def adults := num_parents + num_grandmother
def attraction_total_cost_adults := adults * attraction_ticket_cost_adult

def total_cost := entrance_total_cost + attraction_total_cost_kids + attraction_total_cost_adults

theorem family_visit_cost_is_55 : total_cost = 55 := by
  sorry

end family_visit_cost_is_55_l1102_110262


namespace fill_box_with_cubes_l1102_110200

-- Define the dimensions of the box
def boxLength : ℕ := 35
def boxWidth : ℕ := 20
def boxDepth : ℕ := 10

-- Define the greatest common divisor of the box dimensions
def gcdBoxDims : ℕ := Nat.gcd (Nat.gcd boxLength boxWidth) boxDepth

-- Define the smallest number of identical cubes that can fill the box
def smallestNumberOfCubes : ℕ := (boxLength / gcdBoxDims) * (boxWidth / gcdBoxDims) * (boxDepth / gcdBoxDims)

theorem fill_box_with_cubes :
  smallestNumberOfCubes = 56 :=
by
  -- Proof goes here
  sorry

end fill_box_with_cubes_l1102_110200


namespace infinite_series_value_l1102_110234

theorem infinite_series_value :
  ∑' n : ℕ, (n^3 + 2*n^2 + 5*n + 2) / (3^n * (n^3 + 3)) = 1 / 2 :=
sorry

end infinite_series_value_l1102_110234


namespace no_integer_solutions_l1102_110207

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), 21 * x - 35 * y = 59 :=
by
  sorry

end no_integer_solutions_l1102_110207


namespace max_integer_valued_fractions_l1102_110232

-- Problem Statement:
-- Given a set of natural numbers from 1 to 22,
-- the maximum number of fractions that can be formed such that each fraction is an integer
-- (where an integer fraction is defined as a/b being an integer if and only if b divides a) is 10.

open Nat

theorem max_integer_valued_fractions : 
  ∀ (S : Finset ℕ), (∀ x, x ∈ S → 1 ≤ x ∧ x ≤ 22) →
  ∃ P : Finset (ℕ × ℕ), (∀ (a b : ℕ), (a, b) ∈ P → b ∣ a) ∧ P.card = 11 → 
  10 ≤ (P.filter (λ p => p.1 % p.2 = 0)).card :=
by
  -- proof goes here
  sorry

end max_integer_valued_fractions_l1102_110232


namespace time_per_room_l1102_110288

theorem time_per_room (R P T: ℕ) (h: ℕ) (h₁ : R = 11) (h₂ : P = 2) (h₃ : T = 63) (h₄ : h = T / (R - P)) : h = 7 :=
by
  sorry

end time_per_room_l1102_110288


namespace exists_saddle_point_probability_l1102_110205

noncomputable def saddle_point_probability := (3 : ℝ) / 10

theorem exists_saddle_point_probability {A : ℕ → ℕ → ℝ}
  (h : ∀ i j, 0 ≤ A i j ∧ A i j ≤ 1 ∧ (∀ k l, (i ≠ k ∨ j ≠ l) → A i j ≠ A k l)) :
  (∃ (p : ℝ), p = saddle_point_probability) :=
by 
  sorry

end exists_saddle_point_probability_l1102_110205


namespace determine_m_value_l1102_110271

theorem determine_m_value (m : ℤ) (A : Set ℤ) : 
  A = {1, m + 2, m^2 + 4} → 5 ∈ A → m = 3 ∨ m = 1 := 
by
  sorry

end determine_m_value_l1102_110271
