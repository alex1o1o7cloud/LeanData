import Mathlib

namespace reduction_when_fifth_runner_twice_as_fast_l1014_101407

theorem reduction_when_fifth_runner_twice_as_fast (T T1 T2 T3 T4 T5 : ℝ)
  (h1 : T = T1 + T2 + T3 + T4 + T5)
  (h_T1 : (T1 / 2 + T2 + T3 + T4 + T5) = 0.95 * T)
  (h_T2 : (T1 + T2 / 2 + T3 + T4 + T5) = 0.90 * T)
  (h_T3 : (T1 + T2 + T3 / 2 + T4 + T5) = 0.88 * T)
  (h_T4 : (T1 + T2 + T3 + T4 / 2 + T5) = 0.85 * T)
  : (T1 + T2 + T3 + T4 + T5 / 2) = 0.92 * T := 
sorry

end reduction_when_fifth_runner_twice_as_fast_l1014_101407


namespace binomial_expansion_judgments_l1014_101402

theorem binomial_expansion_judgments :
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, n = 4 * r) ∧
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, n = 4 * r + 3) :=
by
  sorry

end binomial_expansion_judgments_l1014_101402


namespace correlation_comparison_l1014_101426

-- Definitions of the datasets
def data_XY : List (ℝ × ℝ) := [(10,1), (11.3,2), (11.8,3), (12.5,4), (13,5)]
def data_UV : List (ℝ × ℝ) := [(10,5), (11.3,4), (11.8,3), (12.5,2), (13,1)]

-- Definitions of the linear correlation coefficients
noncomputable def r1 : ℝ := sorry -- Calculation of correlation coefficient between X and Y
noncomputable def r2 : ℝ := sorry -- Calculation of correlation coefficient between U and V

-- The proof statement
theorem correlation_comparison :
  r2 < 0 ∧ 0 < r1 :=
sorry

end correlation_comparison_l1014_101426


namespace total_cost_tom_pays_for_trip_l1014_101478

/-- Tom needs to get 10 different vaccines and a doctor's visit to go to Barbados.
    Each vaccine costs $45.
    The doctor's visit costs $250.
    Insurance will cover 80% of these medical bills.
    The trip itself costs $1200.
    Prove that the total amount Tom has to pay for his trip to Barbados, including medical expenses, is $1340. -/
theorem total_cost_tom_pays_for_trip : 
  let cost_per_vaccine := 45
  let number_of_vaccines := 10
  let cost_doctor_visit := 250
  let insurance_coverage_rate := 0.8
  let trip_cost := 1200
  let total_medical_cost := (number_of_vaccines * cost_per_vaccine) + cost_doctor_visit
  let insurance_coverage := insurance_coverage_rate * total_medical_cost
  let net_medical_cost := total_medical_cost - insurance_coverage
  let total_cost := trip_cost + net_medical_cost
  total_cost = 1340 := 
by 
  sorry

end total_cost_tom_pays_for_trip_l1014_101478


namespace stratified_sampling_red_balls_l1014_101498

-- Define the conditions
def total_balls : ℕ := 1000
def red_balls : ℕ := 50
def sampled_balls : ℕ := 100

-- Prove that the number of red balls sampled using stratified sampling is 5
theorem stratified_sampling_red_balls :
  (red_balls : ℝ) / (total_balls : ℝ) * (sampled_balls : ℝ) = 5 := 
by
  sorry

end stratified_sampling_red_balls_l1014_101498


namespace simplify_expression_l1014_101411

-- Define the variables and conditions
variables {a b x y : ℝ}
variable (h1 : a * b * (x^2 - y^2) + x * y * (a^2 - b^2) ≠ 0)
variable (h2 : x ≠ -(a * y) / b)
variable (h3 : x ≠ (b * y) / a)

-- The Theorem to prove
theorem simplify_expression
  (a b x y : ℝ)
  (h1 : a * b * (x^2 - y^2) + x * y * (a^2 - b^2) ≠ 0)
  (h2 : x ≠ -(a * y) / b)
  (h3 : x ≠ (b * y) / a) :
  (a * b * (x^2 + y^2) + x * y * (a^2 + b^2)) *
  ((a * x + b * y)^2 - 4 * a * b * x * y) /
  (a * b * (x^2 - y^2) + x * y * (a^2 - b^2)) = 
  a^2 * x^2 - b^2 * y^2 :=
sorry

end simplify_expression_l1014_101411


namespace find_n_equal_roots_l1014_101491

theorem find_n_equal_roots (x n : ℝ) (hx : x ≠ 2) : n = -1 ↔
  let a := 1
  let b := -2
  let c := -(n^2 + 2 * n)
  b^2 - 4 * a * c = 0 :=
by
  sorry

end find_n_equal_roots_l1014_101491


namespace primes_satisfying_condition_l1014_101422

theorem primes_satisfying_condition :
    {p : ℕ | p.Prime ∧ ∀ q : ℕ, q.Prime ∧ q < p → ¬ ∃ n : ℕ, n^2 ∣ (p - (p / q) * q)} =
    {2, 3, 5, 7, 13} :=
by sorry

end primes_satisfying_condition_l1014_101422


namespace total_price_purchase_l1014_101409

variable (S T : ℝ)

theorem total_price_purchase (h1 : 2 * S + T = 2600) (h2 : 900 = 1200 * 0.75) : 2600 + 900 = 3500 := by
  sorry

end total_price_purchase_l1014_101409


namespace equation_completing_square_l1014_101406

theorem equation_completing_square :
  ∃ (a b c : ℤ), 64 * x^2 + 80 * x - 81 = 0 → 
  (a > 0) ∧ (2 * a * b = 80) ∧ (a^2 = 64) ∧ (a + b + c = 119) :=
sorry

end equation_completing_square_l1014_101406


namespace telescope_visual_range_increased_l1014_101479

/-- A certain telescope increases the visual range from 100 kilometers to 150 kilometers. 
    Proof that the visual range is increased by 50% using the telescope.
-/
theorem telescope_visual_range_increased :
  let original_range := 100
  let new_range := 150
  (new_range - original_range) / original_range * 100 = 50 := 
by
  sorry

end telescope_visual_range_increased_l1014_101479


namespace p_and_q_necessary_not_sufficient_l1014_101451

variable (a m x : ℝ) (P Q : Prop)

def p (a m : ℝ) : Prop := a < 0 ∧ m^2 - 4 * a * m + 3 * a^2 < 0

def q (m : ℝ) : Prop := ∀ x > 0, x + 4 / x ≥ 1 - m

theorem p_and_q_necessary_not_sufficient :
  (∀ (a m : ℝ), p a m → q m) ∧ (∀ a : ℝ, -1 ≤ a ∧ a < 0) :=
sorry

end p_and_q_necessary_not_sufficient_l1014_101451


namespace vector_CD_l1014_101474

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D : V)
variables (a b c : V)
variable (h1 : B - A = a)
variable (h2 : B - C = b)
variable (h3 : D - A = c)

theorem vector_CD :
  D - C = -a + b + c :=
by
  -- Proof omitted
  sorry

end vector_CD_l1014_101474


namespace trigonometric_relationship_l1014_101461

noncomputable def a : ℝ := Real.sin (46 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (46 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (46 * Real.pi / 180)

theorem trigonometric_relationship : c > a ∧ a > b :=
by
  -- This is the statement part; the proof will be handled here
  sorry

end trigonometric_relationship_l1014_101461


namespace min_ties_to_ensure_pairs_l1014_101405

variable (red blue green yellow : Nat)
variable (total_ties : Nat)
variable (pairs_needed : Nat)

-- Define the conditions
def conditions : Prop :=
  red = 120 ∧
  blue = 90 ∧
  green = 70 ∧
  yellow = 50 ∧
  total_ties = 27 ∧
  pairs_needed = 12

-- Define the statement to be proven
theorem min_ties_to_ensure_pairs : conditions red blue green yellow total_ties pairs_needed → total_ties = 27 :=
sorry

end min_ties_to_ensure_pairs_l1014_101405


namespace chord_length_l1014_101420

-- Define the key components.
structure Circle := 
(center : ℝ × ℝ)
(radius : ℝ)

-- Define the initial conditions.
def circle1 : Circle := { center := (0, 0), radius := 5 }
def circle2 : Circle := { center := (2, 0), radius := 3 }

-- Define the chord and tangency condition.
def touches_internally (C1 C2 : Circle) : Prop :=
  C1.radius > C2.radius ∧ dist C1.center C2.center = C1.radius - C2.radius

def chord_divided_ratio (AB_length : ℝ) (r1 r2 : ℝ) : Prop :=
  ∃ (x : ℝ), AB_length = 4 * x ∧ r1 = x ∧ r2 = 3 * x

-- The theorem to prove the length of the chord AB.
theorem chord_length (h1 : touches_internally circle1 circle2)
                     (h2 : chord_divided_ratio 8 2 (6)) : ∃ (AB_length : ℝ), AB_length = 8 :=
by
  sorry

end chord_length_l1014_101420


namespace larger_number_is_391_l1014_101472

-- Define the H.C.F and factors
def HCF := 23
def factor1 := 13
def factor2 := 17
def LCM := HCF * factor1 * factor2

-- Define the two numbers based on the factors
def number1 := HCF * factor1
def number2 := HCF * factor2

-- Theorem statement
theorem larger_number_is_391 : max number1 number2 = 391 := 
by
  sorry

end larger_number_is_391_l1014_101472


namespace hash_hash_hash_100_l1014_101470

def hash (N : ℝ) : ℝ := 0.4 * N + 3

theorem hash_hash_hash_100 : hash (hash (hash 100)) = 11.08 :=
by sorry

end hash_hash_hash_100_l1014_101470


namespace total_toys_l1014_101434

theorem total_toys (n : ℕ) (h1 : 3 * (n / 4) = 18) : n = 24 :=
by
  sorry

end total_toys_l1014_101434


namespace payment_required_l1014_101492

-- Definitions of the conditions
def price_suit : ℕ := 200
def price_tie : ℕ := 40
def num_suits : ℕ := 20
def discount_option_1 (x : ℕ) (hx : x > 20) : ℕ := price_suit * num_suits + (x - num_suits) * price_tie
def discount_option_2 (x : ℕ) (hx : x > 20) : ℕ := (price_suit * num_suits + x * price_tie) * 9 / 10

-- Theorem that needs to be proved
theorem payment_required (x : ℕ) (hx : x > 20) :
  discount_option_1 x hx = 40 * x + 3200 ∧ discount_option_2 x hx = 3600 + 36 * x :=
by sorry

end payment_required_l1014_101492


namespace arithmetic_mean_l1014_101480

variable {x b c : ℝ}

theorem arithmetic_mean (hx : x ≠ 0) (hb : b ≠ c) : 
  (1 / 2) * ((x + b) / x + (x - c) / x) = 1 + (b - c) / (2 * x) :=
by
  sorry

end arithmetic_mean_l1014_101480


namespace largest_sum_valid_set_l1014_101427

-- Define the conditions for the set S
def valid_set (S : Finset ℕ) : Prop :=
  (∀ x ∈ S, 0 < x ∧ x ≤ 15) ∧
  ∀ (A B : Finset ℕ), A ⊆ S → B ⊆ S → A ≠ B → A ∩ B = ∅ → A.sum id ≠ B.sum id

-- The theorem stating the largest sum of such a set
theorem largest_sum_valid_set : ∃ (S : Finset ℕ), valid_set S ∧ S.sum id = 61 :=
sorry

end largest_sum_valid_set_l1014_101427


namespace trapezoid_area_no_solutions_l1014_101404

noncomputable def no_solutions_to_trapezoid_problem : Prop :=
  ∀ (b1 b2 : ℕ), 
    (∃ (m n : ℕ), b1 = 10 * m ∧ b2 = 10 * n) →
    (b1 + b2 = 72) → false

theorem trapezoid_area_no_solutions : no_solutions_to_trapezoid_problem :=
by
  sorry

end trapezoid_area_no_solutions_l1014_101404


namespace quotient_of_m_and_n_l1014_101482

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem quotient_of_m_and_n (m n : ℝ) (h1 : 0 < m) (h2 : m < n) (h3 : f m = f n) (h4 : ∀ x ∈ Set.Icc (m^2) n, f x ≤ 2) :
  n / m = Real.exp 2 :=
by
  sorry

end quotient_of_m_and_n_l1014_101482


namespace multiply_powers_zero_exponent_distribute_term_divide_powers_l1014_101421

-- 1. Prove a^{2} \cdot a^{3} = a^{5}
theorem multiply_powers (a : ℝ) : a^2 * a^3 = a^5 := 
sorry

-- 2. Prove (3.142 - π)^{0} = 1
theorem zero_exponent : (3.142 - Real.pi)^0 = 1 := 
sorry

-- 3. Prove 2a(a^{2} - 1) = 2a^{3} - 2a
theorem distribute_term (a : ℝ) : 2 * a * (a^2 - 1) = 2 * a^3 - 2 * a := 
sorry

-- 4. Prove (-m^{3})^{2} \div m^{4} = m^{2}
theorem divide_powers (m : ℝ) : ((-m^3)^2) / (m^4) = m^2 := 
sorry

end multiply_powers_zero_exponent_distribute_term_divide_powers_l1014_101421


namespace exists_n_lt_p_minus_1_not_div_p2_l1014_101429

theorem exists_n_lt_p_minus_1_not_div_p2 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_3 : p > 3) :
  ∃ (n : ℕ), n < p - 1 ∧ ¬(p^2 ∣ (n^((p - 1)) - 1)) ∧ ¬(p^2 ∣ ((n + 1)^((p - 1)) - 1)) := 
sorry

end exists_n_lt_p_minus_1_not_div_p2_l1014_101429


namespace count_integers_l1014_101437

def satisfies_conditions (n : ℤ) (r : ℤ) : Prop :=
  200 < n ∧ n < 300 ∧ n % 7 = r ∧ n % 9 = r ∧ 0 ≤ r ∧ r < 5

theorem count_integers (n : ℤ) (r : ℤ) :
  (satisfies_conditions n r) → ∃! n, 200 < n ∧ n < 300 ∧ ∃ r, n % 7 = r ∧ n % 9 = r ∧ 0 ≤ r ∧ r < 5 :=
by
  sorry

end count_integers_l1014_101437


namespace total_candies_count_l1014_101466

variable (purple_candies orange_candies yellow_candies : ℕ)

theorem total_candies_count
  (ratio_condition : purple_candies / orange_candies = 2 / 4 ∧ purple_candies / yellow_candies = 2 / 5)
  (yellow_candies_count : yellow_candies = 40) :
  purple_candies + orange_candies + yellow_candies = 88 :=
by
  sorry

end total_candies_count_l1014_101466


namespace players_in_physics_class_l1014_101423

theorem players_in_physics_class (total players_math players_both : ℕ)
    (h1 : total = 15)
    (h2 : players_math = 9)
    (h3 : players_both = 4) :
    (players_math - players_both) + (total - (players_math - players_both + players_both)) + players_both = 10 :=
by {
  sorry
}

end players_in_physics_class_l1014_101423


namespace two_a_minus_b_equals_four_l1014_101493

theorem two_a_minus_b_equals_four (a b : ℕ) 
    (consec_integers : b = a + 1)
    (min_a : min (Real.sqrt 30) a = a)
    (min_b : min (Real.sqrt 30) b = Real.sqrt 30) : 
    2 * a - b = 4 := 
sorry

end two_a_minus_b_equals_four_l1014_101493


namespace pipe_A_fills_tank_in_16_hours_l1014_101468

theorem pipe_A_fills_tank_in_16_hours
  (A : ℝ)
  (h1 : ∀ t : ℝ, t = 12.000000000000002 → (1/A + 1/24) * t = 5/4) :
  A = 16 :=
by sorry

end pipe_A_fills_tank_in_16_hours_l1014_101468


namespace max_min_sum_l1014_101400

variable {α : Type*} [LinearOrderedField α]

def is_odd_function (g : α → α) : Prop :=
∀ x, g (-x) = - g x

def has_max_min (f : α → α) (M N : α) : Prop :=
  (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ (∀ x, N ≤ f x) ∧ (∃ x₁, f x₁ = N)

theorem max_min_sum (g f : α → α) (M N : α)
  (h_odd : is_odd_function g)
  (h_def : ∀ x, f x = g (x - 2) + 1)
  (h_max_min : has_max_min f M N) :
  M + N = 2 :=
sorry

end max_min_sum_l1014_101400


namespace original_bales_l1014_101464

/-
There were some bales of hay in the barn. Jason stacked 23 bales in the barn today.
There are now 96 bales of hay in the barn. Prove that the original number of bales of hay 
in the barn was 73.
-/

theorem original_bales (stacked : ℕ) (total : ℕ) (original : ℕ) 
  (h1 : stacked = 23) (h2 : total = 96) : original = 73 :=
by
  sorry

end original_bales_l1014_101464


namespace percentage_difference_l1014_101449

theorem percentage_difference (w x y z : ℝ) (h1 : w = 0.6 * x) (h2 : x = 0.6 * y) (h3 : z = 0.54 * y) : 
  ((z - w) / w) * 100 = 50 :=
by
  sorry

end percentage_difference_l1014_101449


namespace quadratic_no_real_roots_l1014_101477

theorem quadratic_no_real_roots (k : ℝ) : (∀ x : ℝ, x^2 + 2*x + k ≠ 0) ↔ k > 1 :=
by
  sorry

end quadratic_no_real_roots_l1014_101477


namespace minimum_sum_l1014_101403

theorem minimum_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) + ((a^2 * b) / (18 * b * c)) ≥ 4 / 9 :=
sorry

end minimum_sum_l1014_101403


namespace solve_real_numbers_l1014_101496

theorem solve_real_numbers (x y : ℝ) :
  (x = 3 * x^2 * y - y^3) ∧ (y = x^3 - 3 * x * y^2) ↔
  ((x = 0 ∧ y = 0) ∨ 
   (x = (Real.sqrt (2 + Real.sqrt 2)) / 2 ∧ y = (Real.sqrt (2 - Real.sqrt 2)) / 2) ∨
   (x = -(Real.sqrt (2 - Real.sqrt 2)) / 2 ∧ y = (Real.sqrt (2 + Real.sqrt 2)) / 2) ∨
   (x = -(Real.sqrt (2 + Real.sqrt 2)) / 2 ∧ y = -(Real.sqrt (2 - Real.sqrt 2)) / 2) ∨
   (x = (Real.sqrt (2 - Real.sqrt 2)) / 2 ∧ y = -(Real.sqrt (2 + Real.sqrt 2)) / 2)) :=
by
  sorry

end solve_real_numbers_l1014_101496


namespace perpendicular_value_of_k_parallel_value_of_k_l1014_101471

variables (a b : ℝ × ℝ) (k : ℝ)

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-3, 1)
def ka_plus_b (k : ℝ) : ℝ × ℝ := (2*k - 3, 3*k + 1)
def a_minus_3b : ℝ × ℝ := (11, 0)

theorem perpendicular_value_of_k 
  (h : a = vector_a ∧ b = vector_b ∧ (ka_plus_b k) = (2*k - 3, 3*k + 1) ∧ a_minus_3b = (11, 0)) :
  a - ka_plus_b k = a_minus_3b → k = (3 / 2) :=
sorry

theorem parallel_value_of_k 
  (h : a = vector_a ∧ b = vector_b ∧ (ka_plus_b k) = (2*k - 3, 3*k + 1) ∧ a_minus_3b = (11, 0)) :
  ∃ k, (ka_plus_b (-1/3)) = (-1/3 * 11, -1/3 * 0) ∧ k = -1 / 3 :=
sorry

end perpendicular_value_of_k_parallel_value_of_k_l1014_101471


namespace coffee_decaf_percentage_l1014_101414

variable (initial_stock : ℝ) (initial_decaf_percent : ℝ)
variable (new_stock : ℝ) (new_decaf_percent : ℝ)

noncomputable def decaf_coffee_percentage : ℝ :=
  let initial_decaf : ℝ := initial_stock * (initial_decaf_percent / 100)
  let new_decaf : ℝ := new_stock * (new_decaf_percent / 100)
  let total_decaf : ℝ := initial_decaf + new_decaf
  let total_stock : ℝ := initial_stock + new_stock
  (total_decaf / total_stock) * 100

theorem coffee_decaf_percentage :
  initial_stock = 400 →
  initial_decaf_percent = 20 →
  new_stock = 100 →
  new_decaf_percent = 50 →
  decaf_coffee_percentage initial_stock initial_decaf_percent new_stock new_decaf_percent = 26 :=
by
  intros
  sorry

end coffee_decaf_percentage_l1014_101414


namespace travel_time_l1014_101453

theorem travel_time (speed distance : ℕ) (h_speed : speed = 100) (h_distance : distance = 500) :
  distance / speed = 5 := by
  sorry

end travel_time_l1014_101453


namespace glove_probability_correct_l1014_101416

noncomputable def glove_probability : ℚ :=
  let red_pair := ("r1", "r2") -- pair of red gloves
  let black_pair := ("b1", "b2") -- pair of black gloves
  let white_pair := ("w1", "w2") -- pair of white gloves
  let all_pairs := [
    (red_pair.1, red_pair.2), 
    (black_pair.1, black_pair.2), 
    (white_pair.1, white_pair.2),
    (red_pair.1, black_pair.2), (red_pair.1, white_pair.2), 
    (red_pair.2, black_pair.1), (red_pair.2, white_pair.1),
    (black_pair.1, white_pair.2), (black_pair.2, white_pair.1)
  ]
  let valid_pairs := [(red_pair.1, black_pair.2), (red_pair.1, white_pair.2), 
                      (red_pair.2, black_pair.1), (red_pair.2, white_pair.1), 
                      (black_pair.1, white_pair.2), (black_pair.2, white_pair.1)]
  (valid_pairs.length : ℚ) / (all_pairs.length : ℚ)

theorem glove_probability_correct :
  glove_probability = 2 / 5 := 
by
  sorry

end glove_probability_correct_l1014_101416


namespace scientific_notation_113700_l1014_101483

theorem scientific_notation_113700 : (113700 : ℝ) = 1.137 * 10^5 :=
by
  sorry

end scientific_notation_113700_l1014_101483


namespace sufficient_but_not_necessary_l1014_101445

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 0 → x^2 + x > 0) ∧ (∃ y : ℝ, y < -1 ∧ y^2 + y > 0) :=
by
  sorry

end sufficient_but_not_necessary_l1014_101445


namespace inequality_solution_l1014_101401

theorem inequality_solution :
  {x : ℝ | (x - 3) * (x + 2) ≠ 0 ∧ (x^2 + 1) / ((x - 3) * (x + 2)) ≥ 0} = 
  {x : ℝ | x ≤ -2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end inequality_solution_l1014_101401


namespace trig_identity_l1014_101433

theorem trig_identity (x : ℝ) (h : Real.sin (x + π / 3) = 1 / 3) : 
  Real.sin ((5 * π) / 3 - x) - Real.cos (2 * x - π / 3) = 4 / 9 := 
by 
  sorry

end trig_identity_l1014_101433


namespace inradius_length_l1014_101436

noncomputable def inradius (BC AB AC IC : ℝ) (r : ℝ) : Prop :=
  ∀ (r : ℝ), ((BC = 40) ∧ (AB = AC) ∧ (IC = 24)) →
    r = 4 * Real.sqrt 11

theorem inradius_length (BC AB AC IC : ℝ) (r : ℝ) :
  (BC = 40) ∧ (AB = AC) ∧ (IC = 24) →
  r = 4 * Real.sqrt 11 := 
by
  sorry

end inradius_length_l1014_101436


namespace b11_eq_4_l1014_101432

variables {a : ℕ → ℤ} {b : ℕ → ℤ} {d r : ℤ} {a1 : ℤ}

-- Define non-zero arithmetic sequence {a_n} with common difference d
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define geometric sequence {b_n} with common ratio r
def is_geometric_sequence (b : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ n, b (n + 1) = b n * r

-- The given conditions
axiom a1_minus_a7_sq_plus_a13_eq_zero : a 1 - (a 7) ^ 2 + a 13 = 0
axiom b7_eq_a7 : b 7 = a 7

-- The problem statement to prove: b 11 = 4
theorem b11_eq_4
  (arith_seq : is_arithmetic_sequence a d)
  (geom_seq : is_geometric_sequence b r)
  (a1_non_zero : a1 ≠ 0) :
  b 11 = 4 :=
sorry

end b11_eq_4_l1014_101432


namespace ratio_of_length_to_height_l1014_101415

theorem ratio_of_length_to_height
  (w h l : ℝ)
  (h_eq : h = 6 * w)
  (vol_eq : 129024 = w * h * l)
  (w_eq : w = 8) :
  l / h = 7 := 
sorry

end ratio_of_length_to_height_l1014_101415


namespace total_amount_is_4200_l1014_101481

variables (p q r : ℕ)
variable (total_amount : ℕ)
variable (r_has_two_thirds : total_amount / 3 * 2 = 2800)
variable (r_value : r = 2800)

theorem total_amount_is_4200 (h1 : total_amount / 3 * 2 = 2800) (h2 : r = 2800) : total_amount = 4200 :=
by
  sorry

end total_amount_is_4200_l1014_101481


namespace fred_final_cards_l1014_101430

def initial_cards : ℕ := 40
def keith_bought : ℕ := 22
def linda_bought : ℕ := 15

theorem fred_final_cards : initial_cards - keith_bought - linda_bought = 3 :=
by sorry

end fred_final_cards_l1014_101430


namespace range_of_g_l1014_101413

noncomputable def g (a x : ℝ) : ℝ :=
  a * (Real.cos x)^4 - 2 * (Real.sin x) * (Real.cos x) + (Real.sin x)^4

theorem range_of_g (a : ℝ) (h : a > 0) :
  Set.range (g a) = Set.Icc (a - (3 - a) / 2) (a + (a + 1) / 2) :=
sorry

end range_of_g_l1014_101413


namespace original_deck_size_l1014_101497

noncomputable def initial_red_probability (r b : ℕ) : Prop := r / (r + b) = 1 / 4
noncomputable def added_black_probability (r b : ℕ) : Prop := r / (r + (b + 6)) = 1 / 6

theorem original_deck_size (r b : ℕ) 
  (h1 : initial_red_probability r b) 
  (h2 : added_black_probability r b) : 
  r + b = 12 := 
sorry

end original_deck_size_l1014_101497


namespace fraction_comparison_l1014_101460

theorem fraction_comparison : (9 / 16) > (5 / 9) :=
by {
  sorry -- the detailed proof is not required for this task
}

end fraction_comparison_l1014_101460


namespace emily_lives_lost_l1014_101462

variable (L : ℕ)
variable (initial_lives : ℕ) (extra_lives : ℕ) (final_lives : ℕ)

-- Conditions based on the problem statement
axiom initial_lives_def : initial_lives = 42
axiom extra_lives_def : extra_lives = 24
axiom final_lives_def : final_lives = 41

-- Mathematically equivalent proof statement
theorem emily_lives_lost : initial_lives - L + extra_lives = final_lives → L = 25 := by
  sorry

end emily_lives_lost_l1014_101462


namespace six_degree_below_zero_is_minus_six_degrees_l1014_101459

def temp_above_zero (temp: Int) : String := "+" ++ toString temp ++ "°C"

def temp_below_zero (temp: Int) : String := "-" ++ toString temp ++ "°C"

-- Statement of the theorem
theorem six_degree_below_zero_is_minus_six_degrees:
  temp_below_zero 6 = "-6°C" :=
by
  sorry

end six_degree_below_zero_is_minus_six_degrees_l1014_101459


namespace fruit_seller_stock_l1014_101425

-- Define the given conditions
def remaining_oranges : ℝ := 675
def remaining_percentage : ℝ := 0.25

-- Define the problem function
def original_stock (O : ℝ) : Prop :=
  remaining_percentage * O = remaining_oranges

-- Prove the original stock of oranges was 2700 kg
theorem fruit_seller_stock : original_stock 2700 :=
by
  sorry

end fruit_seller_stock_l1014_101425


namespace x_is_integer_if_conditions_hold_l1014_101443

theorem x_is_integer_if_conditions_hold (x : ℝ)
  (h1 : ∃ (k : ℤ), x^2 - x = k)
  (h2 : ∃ (n : ℕ), n ≥ 3 ∧ ∃ (m : ℤ), x^n - x = m) :
  ∃ (z : ℤ), x = z :=
sorry

end x_is_integer_if_conditions_hold_l1014_101443


namespace percentage_increase_of_gross_sales_l1014_101424

theorem percentage_increase_of_gross_sales 
  (P R : ℝ) 
  (orig_gross new_price new_qty new_gross : ℝ)
  (h1 : new_price = 0.8 * P)
  (h2 : new_qty = 1.8 * R)
  (h3 : orig_gross = P * R)
  (h4 : new_gross = new_price * new_qty) :
  ((new_gross - orig_gross) / orig_gross) * 100 = 44 :=
by sorry

end percentage_increase_of_gross_sales_l1014_101424


namespace parallel_lines_minimum_distance_l1014_101440

theorem parallel_lines_minimum_distance :
  ∀ (m n : ℝ) (k : ℝ), 
  k = 2 ∧ ∀ (L1 L2 : ℝ → ℝ), -- we define L1 and L2 as functions
  (L1 = λ y => 2 * y + 3) ∧ (L2 = λ y => k * y - 1) ∧ 
  ((L1 n = m) ∧ (L2 (n + k) = m + 2)) → 
  dist (m, n) (m + 2, n + 2) = 2 * Real.sqrt 2 := 
sorry

end parallel_lines_minimum_distance_l1014_101440


namespace mass_percentage_of_C_in_benzene_l1014_101475

theorem mass_percentage_of_C_in_benzene :
  let C_molar_mass := 12.01 -- g/mol
  let H_molar_mass := 1.008 -- g/mol
  let benzene_C_atoms := 6
  let benzene_H_atoms := 6
  let C_total_mass := benzene_C_atoms * C_molar_mass
  let H_total_mass := benzene_H_atoms * H_molar_mass
  let benzene_total_mass := C_total_mass + H_total_mass
  let mass_percentage_C := (C_total_mass / benzene_total_mass) * 100
  (mass_percentage_C = 92.26) :=
by
  sorry

end mass_percentage_of_C_in_benzene_l1014_101475


namespace original_fraction_is_two_thirds_l1014_101465

theorem original_fraction_is_two_thirds
  (x y : ℕ)
  (h1 : x / (y + 1) = 1 / 2)
  (h2 : (x + 1) / y = 1) :
  x / y = 2 / 3 := by
  sorry

end original_fraction_is_two_thirds_l1014_101465


namespace numbers_equal_l1014_101450

theorem numbers_equal (a b c d : ℕ)
  (h1 : (a + b)^2 % (c * d) = 0)
  (h2 : (a + c)^2 % (b * d) = 0)
  (h3 : (a + d)^2 % (b * c) = 0)
  (h4 : (b + c)^2 % (a * d) = 0)
  (h5 : (b + d)^2 % (a * c) = 0)
  (h6 : (c + d)^2 % (a * b) = 0) :
  a = b ∨ b = c ∨ c = d ∨ a = c ∨ a = d ∨ b = d ∨ (a = b ∧ b = c) ∨ (b = c ∧ c = d) ∨ (a = b ∧ b = d) ∨ (a = c ∧ c = d) :=
sorry

end numbers_equal_l1014_101450


namespace function_identity_l1014_101463

theorem function_identity (f : ℕ → ℕ) (h₁ : ∀ n, 0 < f n)
  (h₂ : ∀ n, f (n + 1) > f (f n)) :
∀ n, f n = n :=
sorry

end function_identity_l1014_101463


namespace additional_hours_to_travel_l1014_101488

theorem additional_hours_to_travel (distance1 time1 distance2 : ℝ) (rate : ℝ) 
  (h1 : distance1 = 270) 
  (h2 : time1 = 3)
  (h3 : distance2 = 180)
  (h4 : rate = distance1 / time1) :
  distance2 / rate = 2 := by
  sorry

end additional_hours_to_travel_l1014_101488


namespace rod_total_length_l1014_101442

theorem rod_total_length (n : ℕ) (piece_length : ℝ) (total_length : ℝ) 
  (h1 : n = 50) 
  (h2 : piece_length = 0.85) 
  (h3 : total_length = n * piece_length) : 
  total_length = 42.5 :=
by
  -- Proof steps will go here
  sorry

end rod_total_length_l1014_101442


namespace percentage_error_edge_percentage_error_edge_l1014_101499

open Real

-- Define the main context, E as the actual edge and E' as the calculated edge
variables (E E' : ℝ)

-- Condition: Error in calculating the area is 4.04%
axiom area_error : E' * E' = E * E * 1.0404

-- Statement: To prove that the percentage error in edge calculation is 2%
theorem percentage_error_edge : (sqrt 1.0404 - 1) * 100 = 2 :=
by sorry

-- Alternatively, include variable and condition definitions in the actual theorem statement
theorem percentage_error_edge' (E E' : ℝ) (h : E' * E' = E * E * 1.0404) : 
    (sqrt 1.0404 - 1) * 100 = 2 :=
by sorry

end percentage_error_edge_percentage_error_edge_l1014_101499


namespace quotient_of_37_div_8_l1014_101444

theorem quotient_of_37_div_8 : (37 / 8) = 4 :=
by
  sorry

end quotient_of_37_div_8_l1014_101444


namespace infinite_gcd_one_l1014_101428

theorem infinite_gcd_one : ∃ᶠ n in at_top, Int.gcd n ⌊Real.sqrt 2 * n⌋ = 1 := sorry

end infinite_gcd_one_l1014_101428


namespace bike_cost_l1014_101431

theorem bike_cost (days_in_two_weeks : ℕ) 
  (bracelets_per_day : ℕ)
  (price_per_bracelet : ℕ)
  (total_bracelets : ℕ)
  (total_money : ℕ) 
  (h1 : days_in_two_weeks = 2 * 7)
  (h2 : bracelets_per_day = 8)
  (h3 : price_per_bracelet = 1)
  (h4 : total_bracelets = days_in_two_weeks * bracelets_per_day)
  (h5 : total_money = total_bracelets * price_per_bracelet) :
  total_money = 112 :=
sorry

end bike_cost_l1014_101431


namespace function_value_at_minus_two_l1014_101435

theorem function_value_at_minus_two {f : ℝ → ℝ} (h : ∀ x : ℝ, x ≠ 0 → f (1/x) + (1/x) * f (-x) = 2 * x) : f (-2) = 7 / 2 :=
sorry

end function_value_at_minus_two_l1014_101435


namespace tetrahedron_pythagorean_theorem_l1014_101490

noncomputable section

variables {a b c : ℝ} {S_ABC S_VAB S_VBC S_VAC : ℝ}

-- Conditions
def is_right_triangle (a b c : ℝ) := c^2 = a^2 + b^2
def is_right_tetrahedron (S_ABC S_VAB S_VBC S_VAC : ℝ) := 
  S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2

-- Theorem Statement
theorem tetrahedron_pythagorean_theorem (a b c S_ABC S_VAB S_VBC S_VAC : ℝ) 
  (h1 : is_right_triangle a b c)
  (h2 : S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2) :
  S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2 := 
by sorry

end tetrahedron_pythagorean_theorem_l1014_101490


namespace exists_invertible_int_matrix_l1014_101438

theorem exists_invertible_int_matrix (m : ℕ) (k : Fin m → ℤ) : 
  ∃ A : Matrix (Fin m) (Fin m) ℤ,
    (∀ j, IsUnit (A + k j • (1 : Matrix (Fin m) (Fin m) ℤ))) :=
sorry

end exists_invertible_int_matrix_l1014_101438


namespace find_A_l1014_101439

theorem find_A (A B : ℕ) (h1: 3 + 6 * (100 + 10 * A + B) = 691) (h2 : 100 ≤ 6 * (100 + 10 * A + B) ∧ 6 * (100 + 10 * A + B) < 1000) : 
A = 8 :=
sorry

end find_A_l1014_101439


namespace Kim_morning_routine_time_l1014_101419

def total_employees : ℕ := 9
def senior_employees : ℕ := 3
def overtime_employees : ℕ := 4
def regular_employees : ℕ := total_employees - senior_employees
def non_overtime_employees : ℕ := total_employees - overtime_employees

def coffee_time : ℕ := 5
def status_update_time (regular senior : ℕ) : ℕ := (regular * 2) + (senior * 3)
def payroll_update_time (overtime non_overtime : ℕ) : ℕ := (overtime * 3) + (non_overtime * 1)
def email_time : ℕ := 10
def task_allocation_time : ℕ := 7

def total_morning_routine_time : ℕ :=
  coffee_time +
  status_update_time regular_employees senior_employees +
  payroll_update_time overtime_employees non_overtime_employees +
  email_time +
  task_allocation_time

theorem Kim_morning_routine_time : total_morning_routine_time = 60 := by
  sorry

end Kim_morning_routine_time_l1014_101419


namespace coordinates_of_Q_l1014_101456

theorem coordinates_of_Q (m : ℤ) (P Q : ℤ × ℤ) (hP : P = (m + 2, 2 * m + 4))
  (hQ_move : Q = (P.1, P.2 + 2)) (hQ_x_axis : Q.2 = 0) : Q = (-1, 0) :=
sorry

end coordinates_of_Q_l1014_101456


namespace geometric_progression_value_l1014_101458

variable (a : ℕ → ℕ)
variable (r : ℕ)
variable (h_geo : ∀ n, a (n + 1) = a n * r)

theorem geometric_progression_value (h2 : a 2 = 2) (h6 : a 6 = 162) : a 10 = 13122 :=
by
  sorry

end geometric_progression_value_l1014_101458


namespace tourist_groupings_l1014_101457

-- Assume a function to count valid groupings exists
noncomputable def num_groupings (guides tourists : ℕ) :=
  if tourists < guides * 2 then 0 
  else sorry -- placeholder for the actual combinatorial function

theorem tourist_groupings : num_groupings 4 8 = 105 := 
by
  -- The proof is omitted intentionally 
  sorry

end tourist_groupings_l1014_101457


namespace total_population_l1014_101476

-- Define the predicates for g, b, and s based on t
variables (g b t s : ℕ)

-- The conditions given in the problem
def condition1 : Prop := g = 4 * t
def condition2 : Prop := b = 6 * g
def condition3 : Prop := s = t / 2

-- The theorem stating the total population is equal to (59 * t) / 2
theorem total_population (g b t s : ℕ) (h1 : condition1 g t) (h2 : condition2 b g) (h3 : condition3 s t) :
  b + g + t + s = 59 * t / 2 :=
by sorry

end total_population_l1014_101476


namespace initial_average_correct_l1014_101446

theorem initial_average_correct (A : ℕ) 
  (num_students : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ)
  (wrong_avg : ℕ) (correct_avg : ℕ) 
  (h1 : num_students = 30)
  (h2 : wrong_mark = 70)
  (h3 : correct_mark = 10)
  (h4 : correct_avg = 98)
  (h5 : num_students * correct_avg = (num_students * A) - (wrong_mark - correct_mark)) :
  A = 100 := 
sorry

end initial_average_correct_l1014_101446


namespace smallest_number_is_28_l1014_101412

theorem smallest_number_is_28 (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : (a + b + c) / 3 = 30) (h4 : b = 29) (h5 : c = b + 4) : a = 28 :=
by
  sorry

end smallest_number_is_28_l1014_101412


namespace parallelogram_area_l1014_101473

theorem parallelogram_area (θ : ℝ) (a b : ℝ) (hθ : θ = 100) (ha : a = 20) (hb : b = 10):
  let angle_complement := 180 - θ
  let area := a * b * Real.sin angle_complement
  area = 200 * Real.cos 10 := 
by
  let angle_complement := 180 - θ
  let area := a * b * Real.sin angle_complement
  sorry

end parallelogram_area_l1014_101473


namespace evaluate_expression_l1014_101417

theorem evaluate_expression (x y z : ℚ) (hx : x = 1 / 2) (hy : y = 1 / 3) (hz : z = 2) : 
  (x^3 * y^4 * z)^2 = 1 / 104976 :=
by 
  sorry

end evaluate_expression_l1014_101417


namespace fish_catch_l1014_101455

theorem fish_catch (B : ℕ) (K : ℕ) (hB : B = 5) (hK : K = 2 * B) : B + K = 15 :=
by
  sorry

end fish_catch_l1014_101455


namespace total_fish_correct_l1014_101452

def Billy_fish : ℕ := 10
def Tony_fish : ℕ := 3 * Billy_fish
def Sarah_fish : ℕ := Tony_fish + 5
def Bobby_fish : ℕ := 2 * Sarah_fish
def Jenny_fish : ℕ := Bobby_fish - 4
def total_fish : ℕ := Billy_fish + Tony_fish + Sarah_fish + Bobby_fish + Jenny_fish

theorem total_fish_correct : total_fish = 211 := by
  sorry

end total_fish_correct_l1014_101452


namespace LCM_activities_l1014_101487

theorem LCM_activities :
  ∃ (d : ℕ), d = Nat.lcm 6 (Nat.lcm 4 (Nat.lcm 16 (Nat.lcm 12 8))) ∧ d = 48 :=
by
  sorry

end LCM_activities_l1014_101487


namespace return_speed_is_48_l1014_101486

variable (d r : ℕ)
variable (t_1 t_2 : ℚ)

-- Given conditions
def distance_each_way : Prop := d = 120
def time_to_travel_A_to_B : Prop := t_1 = d / 80
def time_to_travel_B_to_A : Prop := t_2 = d / r
def average_speed_round_trip : Prop := 60 * (t_1 + t_2) = 2 * d

-- Statement to prove
theorem return_speed_is_48 :
  distance_each_way d ∧
  time_to_travel_A_to_B d t_1 ∧
  time_to_travel_B_to_A d r t_2 ∧
  average_speed_round_trip d t_1 t_2 →
  r = 48 :=
by
  intros
  sorry

end return_speed_is_48_l1014_101486


namespace ordered_pairs_l1014_101467

theorem ordered_pairs (a b : ℝ) (hapos : 0 < a) (hbpos : 0 < b) (x : ℕ → ℝ)
  (h : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a * x (n + 1) - b * x n| < ε) :
  (a = 0 ∧ 0 < b) ∨ (0 < a ∧ |b / a| < 1) :=
sorry

end ordered_pairs_l1014_101467


namespace cos_8_identity_l1014_101494

theorem cos_8_identity (m : ℝ) (h : Real.sin 74 = m) : 
  Real.cos 8 = Real.sqrt ((1 + m) / 2) :=
sorry

end cos_8_identity_l1014_101494


namespace original_number_solution_l1014_101408

theorem original_number_solution (x : ℝ) (h1 : 0 < x) (h2 : 1000 * x = 3 * (1 / x)) : x = Real.sqrt 30 / 100 :=
by
  sorry

end original_number_solution_l1014_101408


namespace fraction_equality_solution_l1014_101447

theorem fraction_equality_solution (x : ℝ) : (5 + x) / (7 + x) = (2 + x) / (3 + x) → x = 1 :=
by
  intro h
  sorry

end fraction_equality_solution_l1014_101447


namespace total_kayaks_built_by_april_l1014_101441

def kayaks_built_february : ℕ := 5
def kayaks_built_next_month (n : ℕ) : ℕ := 3 * n
def kayaks_built_march : ℕ := kayaks_built_next_month kayaks_built_february
def kayaks_built_april : ℕ := kayaks_built_next_month kayaks_built_march

theorem total_kayaks_built_by_april : 
  kayaks_built_february + kayaks_built_march + kayaks_built_april = 65 :=
by
  -- proof goes here
  sorry

end total_kayaks_built_by_april_l1014_101441


namespace find_speeds_l1014_101489

theorem find_speeds 
  (x v u : ℝ)
  (hx : x = u / 4)
  (hv : 0 < v)
  (hu : 0 < u)
  (t_car : 30 / v + 1.25 = 30 / x)
  (meeting_cars : 0.05 * v + 0.05 * u = 5) :
  x = 15 ∧ v = 40 ∧ u = 60 :=
by 
  sorry

end find_speeds_l1014_101489


namespace sandwiches_sold_out_l1014_101485

-- Define the parameters as constant values
def original : ℕ := 9
def available : ℕ := 4

-- The theorem stating the problem and the expected result
theorem sandwiches_sold_out : (original - available) = 5 :=
by
  -- This is the placeholder for the proof
  sorry

end sandwiches_sold_out_l1014_101485


namespace inequality_proof_l1014_101495

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l1014_101495


namespace closest_point_l1014_101418

noncomputable def point_on_line_closest_to (x y : ℝ) : ℝ × ℝ :=
( -11 / 5, 7 / 5 )

theorem closest_point (x y : ℝ) (h_line : y = 2 * x + 3) (h_point : (x, y) = (3, -4)) :
  point_on_line_closest_to x y = ( -11 / 5, 7 / 5 ) :=
sorry

end closest_point_l1014_101418


namespace polynomial_coefficient_sum_equality_l1014_101454

theorem polynomial_coefficient_sum_equality :
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ),
    (∀ x : ℝ, (2 * x + 1)^4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
    (a₀ - a₁ + a₂ - a₃ + a₄ = 1) :=
by
  intros
  sorry

end polynomial_coefficient_sum_equality_l1014_101454


namespace sufficient_but_not_necessary_l1014_101448

def l1 (m : ℝ) : ℝ × ℝ → ℝ
| (x, y) => m * x + (m + 1) * y + 2

def l2 (m : ℝ) : ℝ × ℝ → ℝ
| (x, y) => (m + 1) * x + (m + 4) * y - 3

def perpendicular_slopes (m : ℝ) : Prop :=
  let slope_l1 := -m / (m + 1)
  let slope_l2 := -(m + 1) / (m + 4)
  slope_l1 * slope_l2 = -1

theorem sufficient_but_not_necessary (m : ℝ) : m = -2 → (∃ k, m = -k ∧ perpendicular_slopes k) :=
by
  sorry

end sufficient_but_not_necessary_l1014_101448


namespace traveler_distance_l1014_101484

theorem traveler_distance (a b c d : ℕ) (h1 : a = 24) (h2 : b = 15) (h3 : c = 10) (h4 : d = 9) :
  let net_ns := a - c
  let net_ew := b - d
  let distance := Real.sqrt ((net_ns ^ 2) + (net_ew ^ 2))
  distance = 2 * Real.sqrt 58 := 
by
  sorry

end traveler_distance_l1014_101484


namespace fraction_subtraction_l1014_101410

theorem fraction_subtraction :
  (8 / 23) - (5 / 46) = 11 / 46 := by
  sorry

end fraction_subtraction_l1014_101410


namespace trig_identity_condition_l1014_101469

open Real

theorem trig_identity_condition (a : Real) (h : ∃ x ≥ 0, (tan a = -1 ∧ cos a ≠ 0)) :
  (sin a / sqrt (1 - sin a ^ 2) + sqrt (1 - cos a ^ 2) / cos a) = 0 :=
by
  sorry

end trig_identity_condition_l1014_101469
