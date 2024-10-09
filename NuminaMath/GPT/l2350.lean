import Mathlib

namespace problem_solution_l2350_235089

/-- 
Assume we have points A, B, C, D, and E as defined in the problem with the following properties:
- Triangle ABC has a right angle at C
- AC = 4
- BC = 3
- Triangle ABD has a right angle at A
- AD = 15
- Points C and D are on opposite sides of line AB
- The line through D parallel to AC meets CB extended at E.

Prove that the ratio DE/DB simplifies to 57/80 where p = 57 and q = 80, making p + q = 137.
-/
theorem problem_solution :
  ∃ (p q : ℕ), gcd p q = 1 ∧ (∃ D E : ℝ, DE/DB = p/q ∧ p + q = 137) :=
by
  sorry

end problem_solution_l2350_235089


namespace FlyersDistributon_l2350_235099

variable (total_flyers ryan_flyers alyssa_flyers belinda_percentage : ℕ)
variable (scott_flyers : ℕ)

theorem FlyersDistributon (H : total_flyers = 200)
  (H1 : ryan_flyers = 42)
  (H2 : alyssa_flyers = 67)
  (H3 : belinda_percentage = 20)
  (H4 : scott_flyers = total_flyers - (ryan_flyers + alyssa_flyers + (belinda_percentage * total_flyers) / 100)) :
  scott_flyers = 51 :=
by
  simp [H, H1, H2, H3] at H4
  exact H4

end FlyersDistributon_l2350_235099


namespace extremum_implies_derivative_zero_derivative_zero_not_implies_extremum_l2350_235051

theorem extremum_implies_derivative_zero {f : ℝ → ℝ} {x₀ : ℝ}
    (h_deriv : DifferentiableAt ℝ f x₀) (h_extremum : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → f x ≤ f x₀ ∨ f x ≥ f x₀) :
  deriv f x₀ = 0 :=
sorry

theorem derivative_zero_not_implies_extremum {f : ℝ → ℝ} {x₀ : ℝ}
    (h_deriv : DifferentiableAt ℝ f x₀) (h_deriv_zero : deriv f x₀ = 0) :
  ¬ (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → f x ≤ f x₀ ∨ f x ≥ f x₀) :=
sorry

end extremum_implies_derivative_zero_derivative_zero_not_implies_extremum_l2350_235051


namespace parabola_focus_distance_area_l2350_235079

theorem parabola_focus_distance_area (p : ℝ) (hp : p > 0)
  (A : ℝ × ℝ) (hA : A.2^2 = 2 * p * A.1)
  (hDist : A.1 + p / 2 = 2 * A.1)
  (hArea : 1/2 * (p / 2) * |A.2| = 1) :
  p = 2 :=
sorry

end parabola_focus_distance_area_l2350_235079


namespace original_three_digit_number_a_original_three_digit_number_b_l2350_235000

section ProblemA

variables {x y z : ℕ}

/-- In a three-digit number, the first digit on the left was erased. Then, the resulting
  two-digit number was multiplied by 7, and the original three-digit number was obtained. -/
theorem original_three_digit_number_a (h : ∃ (N : ℕ), N = 100 * x + 10 * y + z ∧ 
  N = 7 * (10 * y + z)) : ∃ (N : ℕ), N = 350 :=
sorry

end ProblemA

section ProblemB

variables {x y z : ℕ}

/-- In a three-digit number, the middle digit was erased, and the resulting number 
  is 6 times smaller than the original. --/
theorem original_three_digit_number_b (h : ∃ (N : ℕ), N = 100 * x + 10 * y + z ∧ 
  6 * (10 * x + z) = N) : ∃ (N : ℕ), N = 108 :=
sorry

end ProblemB

end original_three_digit_number_a_original_three_digit_number_b_l2350_235000


namespace find_B_l2350_235020

variable {A B C D : ℕ}

-- Condition 1: The first dig site (A) was dated 352 years more recent than the second dig site (B)
axiom h1 : A = B + 352

-- Condition 2: The third dig site (C) was dated 3700 years older than the first dig site (A)
axiom h2 : C = A - 3700

-- Condition 3: The fourth dig site (D) was twice as old as the third dig site (C)
axiom h3 : D = 2 * C

-- Condition 4: The age difference between the second dig site (B) and the third dig site (C) was four times the difference between the fourth dig site (D) and the first dig site (A)
axiom h4 : B - C = 4 * (D - A)

-- Condition 5: The fourth dig site is dated 8400 BC.
axiom h5 : D = 8400

-- Prove the question
theorem find_B : B = 7548 :=
by
  sorry

end find_B_l2350_235020


namespace number_of_valid_three_digit_numbers_l2350_235004

def valid_three_digit_numbers : Nat :=
  -- Proving this will be the task: showing that there are precisely 24 such numbers
  24

theorem number_of_valid_three_digit_numbers : valid_three_digit_numbers = 24 :=
by
  -- Proof would go here.
  sorry

end number_of_valid_three_digit_numbers_l2350_235004


namespace range_of_m_l2350_235084

variable (x y m : ℝ)

def system_of_eq1 := 2 * x + y = -4 * m + 5
def system_of_eq2 := x + 2 * y = m + 4
def inequality1 := x - y > -6
def inequality2 := x + y < 8

theorem range_of_m:
  system_of_eq1 x y m → 
  system_of_eq2 x y m → 
  inequality1 x y → 
  inequality2 x y → 
  -5 < m ∧ m < 7/5 :=
by 
  intros h1 h2 h3 h4
  sorry

end range_of_m_l2350_235084


namespace smallest_n_l2350_235088

theorem smallest_n (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, n = 3 * k) (h3 : ∃ m : ℕ, 3 * n = 5 * m) : n = 15 :=
sorry

end smallest_n_l2350_235088


namespace Ariel_current_age_l2350_235082

-- Define the conditions
def Ariel_birth_year : Nat := 1992
def Ariel_start_fencing_year : Nat := 2006
def Ariel_fencing_years : Nat := 16

-- Define the problem as a theorem
theorem Ariel_current_age :
  (Ariel_start_fencing_year - Ariel_birth_year) + Ariel_fencing_years = 30 := by
sorry

end Ariel_current_age_l2350_235082


namespace union_M_N_intersection_complementM_N_l2350_235031

open Set  -- Open the Set namespace for convenient notation.

noncomputable def funcDomain : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
noncomputable def setN : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}
noncomputable def complementFuncDomain : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 2}

theorem union_M_N :
  (funcDomain ∪ setN) = {x : ℝ | -1 ≤ x ∧ x < 3} :=
by
  sorry

theorem intersection_complementM_N :
  (complementFuncDomain ∩ setN) = {x : ℝ | 2 ≤ x ∧ x < 3} :=
by
  sorry

end union_M_N_intersection_complementM_N_l2350_235031


namespace incorrect_statement_l2350_235096

noncomputable def a : ℝ × ℝ := (1, -2)
noncomputable def b : ℝ × ℝ := (2, 1)
noncomputable def c : ℝ × ℝ := (-4, -2)

-- Define the incorrect vector statement D
theorem incorrect_statement :
  ¬ ∀ (d : ℝ × ℝ), ∃ (k1 k2 : ℝ), d = (k1 * b.1 + k2 * c.1, k1 * b.2 + k2 * c.2) := sorry

end incorrect_statement_l2350_235096


namespace A_superset_B_l2350_235098

open Set

variable (N : Set ℕ)
def A : Set ℕ := {x | ∃ n ∈ N, x = 2 * n}
def B : Set ℕ := {x | ∃ n ∈ N, x = 4 * n}

theorem A_superset_B : A N ⊇ B N :=
by
  -- Proof to be written
  sorry

end A_superset_B_l2350_235098


namespace matching_red_pair_probability_l2350_235085

def total_socks := 8
def red_socks := 4
def blue_socks := 2
def green_socks := 2

noncomputable def total_pairs := Nat.choose total_socks 2
noncomputable def red_pairs := Nat.choose red_socks 2
noncomputable def blue_pairs := Nat.choose blue_socks 2
noncomputable def green_pairs := Nat.choose green_socks 2
noncomputable def total_matching_pairs := red_pairs + blue_pairs + green_pairs
noncomputable def probability_red := (red_pairs : ℚ) / total_matching_pairs

theorem matching_red_pair_probability : probability_red = 3 / 4 :=
  by sorry

end matching_red_pair_probability_l2350_235085


namespace jessica_total_monthly_payment_l2350_235026

-- Definitions for the conditions
def basicCableCost : ℕ := 15
def movieChannelsCost : ℕ := 12
def sportsChannelsCost : ℕ := movieChannelsCost - 3

-- The statement to be proven
theorem jessica_total_monthly_payment :
  basicCableCost + movieChannelsCost + sportsChannelsCost = 36 := 
by
  sorry

end jessica_total_monthly_payment_l2350_235026


namespace sum_of_reciprocals_l2350_235060

theorem sum_of_reciprocals (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 5 * x * y) : (1 / x) + (1 / y) = 5 :=
by
  sorry

end sum_of_reciprocals_l2350_235060


namespace proof_S5_l2350_235070

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q a1, ∀ n, a (n + 1) = a1 * q ^ (n + 1)

theorem proof_S5 (a : ℕ → ℝ) (S : ℕ → ℝ) (q a1 : ℝ) : 
  (geometric_sequence a) → 
  (a 2 * a 5 = 2 * a 3) → 
  ((a 4 + 2 * a 7) / 2 = 5 / 4) → 
  (S 5 = a1 * (1 - (1 / 2) ^ 5) / (1 - 1 / 2)) → 
  S 5 = 31 := 
by sorry

end proof_S5_l2350_235070


namespace b_investment_l2350_235053

theorem b_investment (a_investment : ℝ) (c_investment : ℝ) (total_profit : ℝ) (a_share_profit : ℝ) (b_investment : ℝ) : a_investment = 6300 → c_investment = 10500 → total_profit = 14200 → a_share_profit = 4260 → b_investment = 4220 :=
by
  intro h_a h_c h_total h_a_share
  have h1 : 6300 / (6300 + 4220 + 10500) = 4260 / 14200 := sorry
  have h2 : 6300 * 14200 = 4260 * (6300 + 4220 + 10500) := sorry
  have h3 : b_investment = 4220 := sorry
  exact h3

end b_investment_l2350_235053


namespace response_activity_solutions_l2350_235021

theorem response_activity_solutions (x y z : ℕ) :
  5 * x + 4 * y + 3 * z = 15 →
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 0 ∧ y = 3 ∧ z = 1) :=
by
  sorry

end response_activity_solutions_l2350_235021


namespace proof_problem_l2350_235049

variable {a b x y : ℝ}

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem proof_problem : dollar ((x + y) ^ 2) (y ^ 2 + x ^ 2) = 4 * x ^ 2 * y ^ 2 := by
  sorry

end proof_problem_l2350_235049


namespace equal_12_mn_P_2n_Q_m_l2350_235092

-- Define P and Q based on given conditions
def P (m : ℕ) : ℕ := 2 ^ m
def Q (n : ℕ) : ℕ := 3 ^ n

-- The theorem to prove
theorem equal_12_mn_P_2n_Q_m (m n : ℕ) : (12 ^ (m * n)) = (P m ^ (2 * n)) * (Q n ^ m) :=
by
  -- Proof goes here
  sorry

end equal_12_mn_P_2n_Q_m_l2350_235092


namespace expected_value_of_geometric_variance_of_geometric_l2350_235018

noncomputable def expected_value (p : ℝ) : ℝ :=
  1 / p

noncomputable def variance (p : ℝ) : ℝ :=
  (1 - p) / (p ^ 2)

theorem expected_value_of_geometric (p : ℝ) (hp : 0 < p ∧ p < 1) :
  ∑' n, (n + 1 : ℝ) * (1 - p) ^ n * p = expected_value p := by
  sorry

theorem variance_of_geometric (p : ℝ) (hp : 0 < p ∧ p < 1) :
  ∑' n, ((n + 1 : ℝ) ^ 2) * (1 - p) ^ n * p - (expected_value p) ^ 2 = variance p := by
  sorry

end expected_value_of_geometric_variance_of_geometric_l2350_235018


namespace units_digit_G_1000_l2350_235032

def G (n : ℕ) : ℕ := 3^(3^n) + 1

theorem units_digit_G_1000 : (G 1000) % 10 = 4 :=
  sorry

end units_digit_G_1000_l2350_235032


namespace smallest_integer_n_l2350_235087

theorem smallest_integer_n (n : ℤ) (h : n^2 - 9 * n + 20 > 0) : n ≥ 6 := 
sorry

end smallest_integer_n_l2350_235087


namespace blue_faces_cube_l2350_235023

theorem blue_faces_cube (n : ℕ) (h1 : n > 0) (h2 : (6 * n^2) = 1 / 3 * 6 * n^3) : n = 3 :=
by
  -- we only need the statement for now; the proof is omitted.
  sorry

end blue_faces_cube_l2350_235023


namespace root_analysis_l2350_235075

noncomputable def root1 (a : ℝ) : ℝ :=
2 * a + 2 * Real.sqrt (a^2 - 3 * a + 2)

noncomputable def root2 (a : ℝ) : ℝ :=
2 * a - 2 * Real.sqrt (a^2 - 3 * a + 2)

noncomputable def derivedRoot (a : ℝ) : ℝ :=
(3 * a - 2) / a

theorem root_analysis (a : ℝ) (ha : a > 0) :
( (2/3 ≤ a ∧ a < 1) ∨ (2 < a) → (root1 a ≥ 0 ∧ root2 a ≥ 0)) ∧
( 0 < a ∧ a < 2/3 → (derivedRoot a < 0 ∧ root1 a ≥ 0)) :=
sorry

end root_analysis_l2350_235075


namespace minimum_quotient_value_l2350_235007

-- Helper definition to represent the quotient 
def quotient (a b c d : ℕ) : ℚ := (1000 * a + 100 * b + 10 * c + d) / (a + b + c + d)

-- Conditions: digits are distinct and non-zero 
def distinct_and_nonzero (a b c d : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem minimum_quotient_value :
  ∀ (a b c d : ℕ), distinct_and_nonzero a b c d → quotient a b c d = 71.9 :=
by sorry

end minimum_quotient_value_l2350_235007


namespace a1_greater_than_500_l2350_235002

-- Set up conditions
variables (a : ℕ → ℕ) (h1 : ∀ n, 0 < a n ∧ a n < 20000)
variables (h2 : ∀ i j, i < j → gcd (a i) (a j) < a i)
variables (h3 : ∀ i j, i < j ∧ 1 ≤ i ∧ j ≤ 10000 → a i < a j)

/-- Statement to prove / lean concept as per mathematical problem  --/
theorem a1_greater_than_500 : 500 < a 1 :=
sorry

end a1_greater_than_500_l2350_235002


namespace probability_genuine_coins_given_weight_condition_l2350_235072

/--
Given the following conditions:
- Ten counterfeit coins of equal weight are mixed with 20 genuine coins.
- The weight of a counterfeit coin is different from the weight of a genuine coin.
- Two pairs of coins are selected randomly without replacement from the 30 coins. 

Prove that the probability that all 4 selected coins are genuine, given that the combined weight
of the first pair is equal to the combined weight of the second pair, is 5440/5481.
-/
theorem probability_genuine_coins_given_weight_condition :
  let num_coins := 30
  let num_genuine := 20
  let num_counterfeit := 10
  let pairs_selected := 2
  let pairs_remaining := num_coins - pairs_selected * 2
  let P := (num_genuine / num_coins) * ((num_genuine - 1) / (num_coins - 1)) * ((num_genuine - 2) / pairs_remaining) * ((num_genuine - 3) / (pairs_remaining - 1))
  let event_A_given_B := P / (7 / 16)
  event_A_given_B = 5440 / 5481 := 
sorry

end probability_genuine_coins_given_weight_condition_l2350_235072


namespace calc_expression_l2350_235061

def r (θ : ℚ) : ℚ := 1 / (1 + θ)
def s (θ : ℚ) : ℚ := θ + 1

theorem calc_expression : s (r (s (r (s (r 2))))) = 24 / 17 :=
by 
  sorry

end calc_expression_l2350_235061


namespace part1_part2_l2350_235024

-- Definitions for problem conditions and questions

/-- 
Let p and q be two distinct prime numbers greater than 5. 
Show that if p divides 5^q - 2^q then q divides p - 1.
-/
theorem part1 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : 5 < p) (hq_gt_5 : 5 < q) (h_distinct : p ≠ q) 
  (h_div : p ∣ 5^q - 2^q) : q ∣ p - 1 :=
by sorry

/-- 
Let p and q be two distinct prime numbers greater than 5.
Deduce that pq does not divide (5^p - 2^p)(5^q - 2^q).
-/
theorem part2 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : 5 < p) (hq_gt_5 : 5 < q) (h_distinct : p ≠ q) 
  (h_div_q_p1 : q ∣ p - 1)
  (h_div_p_q1 : p ∣ q - 1) : ¬(pq : ℕ) ∣ (5^p - 2^p) * (5^q - 2^q) :=
by sorry

end part1_part2_l2350_235024


namespace additional_profit_is_80000_l2350_235040

-- Define the construction cost of a regular house
def construction_cost_regular (C : ℝ) : ℝ := C

-- Define the construction cost of the special house
def construction_cost_special (C : ℝ) : ℝ := C + 200000

-- Define the selling price of a regular house
def selling_price_regular : ℝ := 350000

-- Define the selling price of the special house
def selling_price_special : ℝ := 1.8 * 350000

-- Define the profit from selling a regular house
def profit_regular (C : ℝ) : ℝ := selling_price_regular - (construction_cost_regular C)

-- Define the profit from selling the special house
def profit_special (C : ℝ) : ℝ := selling_price_special - (construction_cost_special C)

-- Define the additional profit made by building and selling the special house compared to a regular house
def additional_profit (C : ℝ) : ℝ := (profit_special C) - (profit_regular C)

-- Theorem to prove the additional profit is $80,000
theorem additional_profit_is_80000 (C : ℝ) : additional_profit C = 80000 :=
sorry

end additional_profit_is_80000_l2350_235040


namespace intersection_in_first_quadrant_l2350_235011

theorem intersection_in_first_quadrant (a : ℝ) :
  (∃ x y : ℝ, ax - y + 2 = 0 ∧ x + y - a = 0 ∧ x > 0 ∧ y > 0) ↔ a > 2 := 
by
  sorry

end intersection_in_first_quadrant_l2350_235011


namespace correct_negation_of_exactly_one_even_l2350_235050

-- Define a predicate to check if a natural number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define a predicate to check if a natural number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Problem statement in Lean
theorem correct_negation_of_exactly_one_even (a b c : ℕ) :
  ¬ ( (is_even a ∧ is_odd b ∧ is_odd c) ∨ 
      (is_odd a ∧ is_even b ∧ is_odd c) ∨ 
      (is_odd a ∧ is_odd b ∧ is_even c) ) ↔ 
  ( (is_odd a ∧ is_odd b ∧ is_odd c) ∨ 
    (is_even a ∧ is_even b ∧ is_even c) ) :=
by 
  sorry

end correct_negation_of_exactly_one_even_l2350_235050


namespace max_discarded_grapes_l2350_235069

theorem max_discarded_grapes (n : ℕ) : ∃ r, r < 8 ∧ n % 8 = r ∧ r = 7 :=
by
  sorry

end max_discarded_grapes_l2350_235069


namespace side_length_a_l2350_235005

theorem side_length_a (a b c : ℝ) (B : ℝ) (h1 : a = c - 2 * a * Real.cos B) (h2 : c = 5) (h3 : 3 * a = 2 * b) :
  a = 4 := by
  sorry

end side_length_a_l2350_235005


namespace value_of_y_l2350_235010

theorem value_of_y (x y : ℝ) (hx : x = 3) (h : x^(3 * y) = 9) : y = 2 / 3 := by
  sorry

end value_of_y_l2350_235010


namespace sequence_arithmetic_progression_l2350_235077

theorem sequence_arithmetic_progression (b : ℕ → ℕ) (b1_eq : b 1 = 1) (recurrence : ∀ n, b (n + 2) = b (n + 1) * b n + 1) : b 2 = 1 ↔ 
  ∃ d : ℕ, ∀ n, b (n + 1) - b n = d :=
sorry

end sequence_arithmetic_progression_l2350_235077


namespace probability_within_two_units_of_origin_correct_l2350_235038

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let square_area := 36
  let circle_area := 4 * Real.pi
  circle_area / square_area

theorem probability_within_two_units_of_origin_correct :
  probability_within_two_units_of_origin = Real.pi / 9 := by
  sorry

end probability_within_two_units_of_origin_correct_l2350_235038


namespace sequence_geometric_and_general_formula_find_minimum_n_l2350_235042

theorem sequence_geometric_and_general_formula 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → S n + n = 2 * a n) :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) + 1 = 2 * (a n + 1)) ∧ (∀ n : ℕ, n ≥ 1 → a n = 2^n - 1) :=
sorry

theorem find_minimum_n 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (b T : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → S n + n = 2 * a n)
  (h2 : ∀ n : ℕ, b n = (2 * n + 1) * a n + (2 * n + 1))
  (h3 : T 0 = 0)
  (h4 : ∀ n : ℕ, T (n + 1) = T n + b (n + 1)) :
  ∃ n : ℕ, n ≥ 1 ∧ (T n - 2) / (2 * n - 1) > 2010 :=
sorry

end sequence_geometric_and_general_formula_find_minimum_n_l2350_235042


namespace evaluate_at_10_l2350_235014

def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

theorem evaluate_at_10 : f 10 = 756 := by
  -- the proof is omitted
  sorry

end evaluate_at_10_l2350_235014


namespace planes_1_and_6_adjacent_prob_l2350_235097

noncomputable def probability_planes_adjacent (total_planes: ℕ) : ℚ :=
  if total_planes = 6 then 1/3 else 0

theorem planes_1_and_6_adjacent_prob :
  probability_planes_adjacent 6 = 1/3 := 
by
  sorry

end planes_1_and_6_adjacent_prob_l2350_235097


namespace original_price_per_kg_l2350_235078

theorem original_price_per_kg (P : ℝ) (S : ℝ) (reduced_price : ℝ := 0.8 * P) (total_cost : ℝ := 400) (extra_salt : ℝ := 10) :
  S * P = total_cost ∧ (S + extra_salt) * reduced_price = total_cost → P = 10 :=
by
  intros
  sorry

end original_price_per_kg_l2350_235078


namespace three_fifths_difference_products_l2350_235022

theorem three_fifths_difference_products :
  (3 / 5) * ((7 * 9) - (4 * 3)) = 153 / 5 :=
by
  sorry

end three_fifths_difference_products_l2350_235022


namespace average_difference_l2350_235080

-- Definitions for the conditions
def set1 : List ℕ := [20, 40, 60]
def set2 : List ℕ := [10, 60, 35]

-- Function to compute the average of a list of numbers
def average (lst : List ℕ) : ℚ :=
  lst.sum / lst.length

-- The main theorem to prove the difference between the averages is 5
theorem average_difference : average set1 - average set2 = 5 := by
  sorry

end average_difference_l2350_235080


namespace alpha_beta_sum_l2350_235059

theorem alpha_beta_sum (α β : ℝ) (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 102 * x + 2021) / (x^2 + 89 * x - 3960)) : α + β = 176 := by
  sorry

end alpha_beta_sum_l2350_235059


namespace parabola_equation_l2350_235030

theorem parabola_equation (h1: ∃ k, ∀ x y : ℝ, (x, y) = (4, -2) → y^2 = k * x) 
                          (h2: ∃ m, ∀ x y : ℝ, (x, y) = (4, -2) → x^2 = -2 * m * y) :
                          (y : ℝ)^2 = x ∨ (x : ℝ)^2 = -8 * y :=
by 
  sorry

end parabola_equation_l2350_235030


namespace min_overlap_l2350_235083

noncomputable def drinks_coffee := 0.60
noncomputable def drinks_tea := 0.50
noncomputable def drinks_neither := 0.10
noncomputable def drinks_either := 1 - drinks_neither
noncomputable def total_overlap := drinks_coffee + drinks_tea - drinks_either

theorem min_overlap (hcoffee : drinks_coffee = 0.60) (htea : drinks_tea = 0.50) (hneither : drinks_neither = 0.10) :
  total_overlap = 0.20 :=
by
  sorry

end min_overlap_l2350_235083


namespace hannah_trip_time_ratio_l2350_235055

theorem hannah_trip_time_ratio 
  (u : ℝ) -- Speed on the first trip in miles per hour.
  (u_pos : u > 0) -- Speed should be positive.
  (t1 t2 : ℝ) -- Time taken for the first and second trip respectively.
  (h_t1 : t1 = 30 / u) -- Time for the first trip.
  (h_t2 : t2 = 150 / (4 * u)) -- Time for the second trip.
  : t2 / t1 = 1.25 := by
  sorry

end hannah_trip_time_ratio_l2350_235055


namespace subtract_some_number_l2350_235017

theorem subtract_some_number
  (x : ℤ)
  (h : 913 - x = 514) :
  514 - x = 115 :=
by {
  sorry
}

end subtract_some_number_l2350_235017


namespace magic_8_ball_probability_l2350_235063

theorem magic_8_ball_probability :
  let p := 3 / 8
  let q := 5 / 8
  let n := 7
  let k := 3
  (Nat.choose 7 3) * (p^3) * (q^4) = 590625 / 2097152 :=
by
  let p := 3 / 8
  let q := 5 / 8
  let n := 7
  let k := 3
  sorry

end magic_8_ball_probability_l2350_235063


namespace angle_DNE_l2350_235019

theorem angle_DNE (DE EF FD : ℝ) (EFD END FND : ℝ) 
  (h1 : DE = 2 * EF) 
  (h2 : EF = FD) 
  (h3 : EFD = 34) 
  (h4 : END = 3) 
  (h5 : FND = 18) : 
  ∃ DNE : ℝ, DNE = 104 :=
by 
  sorry

end angle_DNE_l2350_235019


namespace katie_bead_necklaces_l2350_235058

theorem katie_bead_necklaces (B : ℕ) (gemstone_necklaces : ℕ := 3) (cost_each_necklace : ℕ := 3) (total_earnings : ℕ := 21) :
  gemstone_necklaces * cost_each_necklace + B * cost_each_necklace = total_earnings → B = 4 :=
by
  intro h
  sorry

end katie_bead_necklaces_l2350_235058


namespace percentage_shaded_is_14_29_l2350_235071

noncomputable def side_length : ℝ := 20
noncomputable def rect_length : ℝ := 35
noncomputable def rect_width : ℝ := side_length
noncomputable def rect_area : ℝ := rect_length * rect_width
noncomputable def overlap_length : ℝ := 2 * side_length - rect_length
noncomputable def overlap_area : ℝ := overlap_length * side_length
noncomputable def shaded_percentage : ℝ := (overlap_area / rect_area) * 100

theorem percentage_shaded_is_14_29 :
  shaded_percentage = 14.29 :=
sorry

end percentage_shaded_is_14_29_l2350_235071


namespace find_L_l2350_235074

-- Conditions definitions
def initial_marbles := 57
def marbles_won_second_game := 25
def final_marbles := 64

-- Definition of L
def L := initial_marbles - 18

theorem find_L (L : ℕ) (H1 : initial_marbles = 57) (H2 : marbles_won_second_game = 25) (H3 : final_marbles = 64) : 
(initial_marbles - L) + marbles_won_second_game = final_marbles -> 
L = 18 :=
by
  sorry

end find_L_l2350_235074


namespace count_multiples_of_5_l2350_235035

theorem count_multiples_of_5 (a b : ℕ) (h₁ : 50 ≤ a) (h₂ : a ≤ 300) (h₃ : 50 ≤ b) (h₄ : b ≤ 300) (h₅ : a % 5 = 0) (h₆ : b % 5 = 0) 
  (h₇ : ∀ n : ℕ, 50 ≤ n ∧ n ≤ 300 → n % 5 = 0 → a ≤ n ∧ n ≤ b) :
  b = a + 48 * 5 → (b - a) / 5 + 1 = 49 :=
by
  sorry

end count_multiples_of_5_l2350_235035


namespace num_of_cows_is_7_l2350_235073

variables (C H : ℕ)

-- Define the conditions
def cow_legs : ℕ := 4 * C
def chicken_legs : ℕ := 2 * H
def cow_heads : ℕ := C
def chicken_heads : ℕ := H

def total_legs : ℕ := cow_legs C + chicken_legs H
def total_heads : ℕ := cow_heads C + chicken_heads H
def legs_condition : Prop := total_legs C H = 2 * total_heads C H + 14

-- The theorem to be proved
theorem num_of_cows_is_7 (h : legs_condition C H) : C = 7 :=
by sorry

end num_of_cows_is_7_l2350_235073


namespace solutions_of_equation_l2350_235054

theorem solutions_of_equation :
  ∀ x : ℝ, x * (x - 3) = x - 3 ↔ x = 1 ∨ x = 3 :=
by sorry

end solutions_of_equation_l2350_235054


namespace rhombus_shorter_diagonal_l2350_235015

variable (d1 d2 : ℝ) (Area : ℝ)

def is_rhombus (Area : ℝ) (d1 d2 : ℝ) : Prop := Area = (d1 * d2) / 2

theorem rhombus_shorter_diagonal
  (h_d2 : d2 = 20)
  (h_Area : Area = 110)
  (h_rhombus : is_rhombus Area d1 d2) :
  d1 = 11 := by
  sorry

end rhombus_shorter_diagonal_l2350_235015


namespace number_of_paths_l2350_235028

theorem number_of_paths (r u : ℕ) (h_r : r = 5) (h_u : u = 4) : 
  (Nat.choose (r + u) u) = 126 :=
by
  -- The proof is omitted, as requested.
  sorry

end number_of_paths_l2350_235028


namespace john_heroes_on_large_sheets_front_l2350_235039

noncomputable def num_pictures_on_large_sheets_front : ℕ :=
  let total_pictures := 20
  let minutes_spent := 75 - 5
  let average_time_per_picture := 5
  let front_pictures := total_pictures / 2
  let x := front_pictures / 3
  2 * x

theorem john_heroes_on_large_sheets_front : num_pictures_on_large_sheets_front = 6 :=
by
  sorry

end john_heroes_on_large_sheets_front_l2350_235039


namespace evaluate_expression_l2350_235081

theorem evaluate_expression : 8 * ((1 : ℚ) / 3)^3 - 1 = -19 / 27 := by
  sorry

end evaluate_expression_l2350_235081


namespace number_of_thrown_out_carrots_l2350_235041

-- Definitions from the conditions
def initial_carrots : ℕ := 48
def picked_next_day : ℕ := 42
def total_carrots : ℕ := 45

-- Proposition stating the problem
theorem number_of_thrown_out_carrots (x : ℕ) : initial_carrots - x + picked_next_day = total_carrots → x = 45 :=
by
  sorry

end number_of_thrown_out_carrots_l2350_235041


namespace percent_of_x_is_z_l2350_235037

variable {x y z : ℝ}

theorem percent_of_x_is_z 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : y = 0.75 * x) : 
  z / x = 1.2 := 
sorry

end percent_of_x_is_z_l2350_235037


namespace number_of_questions_per_survey_is_10_l2350_235003

variable {Q : ℕ}  -- Q: Number of questions in each survey

def money_per_question : ℝ := 0.2
def surveys_on_monday : ℕ := 3
def surveys_on_tuesday : ℕ := 4
def total_money_earned : ℝ := 14

theorem number_of_questions_per_survey_is_10 :
    (surveys_on_monday + surveys_on_tuesday) * Q * money_per_question = total_money_earned → Q = 10 :=
by
  sorry

end number_of_questions_per_survey_is_10_l2350_235003


namespace tim_scored_sum_first_8_even_numbers_l2350_235043

-- Define the first 8 even numbers.
def first_8_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16]

-- Define the sum of those numbers.
def sum_first_8_even_numbers : ℕ := List.sum first_8_even_numbers

-- The theorem stating the problem.
theorem tim_scored_sum_first_8_even_numbers : sum_first_8_even_numbers = 72 := by
  sorry

end tim_scored_sum_first_8_even_numbers_l2350_235043


namespace polynomial_coefficient_sum_l2350_235044

theorem polynomial_coefficient_sum :
  let p := (x + 3) * (4 * x^3 - 2 * x^2 + 7 * x - 6)
  let q := 4 * x^4 + 10 * x^3 + x^2 + 15 * x - 18
  p = q →
  (4 + 10 + 1 + 15 - 18 = 12) :=
by
  intro p_eq_q
  sorry

end polynomial_coefficient_sum_l2350_235044


namespace dentist_cleaning_cost_l2350_235036

theorem dentist_cleaning_cost
  (F: ℕ)
  (C: ℕ)
  (B: ℕ)
  (tooth_extraction_cost: ℕ)
  (HC1: F = 120)
  (HC2: B = 5 * F)
  (HC3: tooth_extraction_cost = 290)
  (HC4: B = C + 2 * F + tooth_extraction_cost) :
  C = 70 :=
by
  sorry

end dentist_cleaning_cost_l2350_235036


namespace stickers_total_l2350_235067

theorem stickers_total (ryan_has : Ryan_stickers = 30)
  (steven_has : Steven_stickers = 3 * Ryan_stickers)
  (terry_has : Terry_stickers = Steven_stickers + 20) :
  Ryan_stickers + Steven_stickers + Terry_stickers = 230 :=
sorry

end stickers_total_l2350_235067


namespace max_area_quad_l2350_235045

noncomputable def MaxAreaABCD : ℝ :=
  let x : ℝ := 3
  let θ : ℝ := Real.pi / 2
  let φ : ℝ := Real.pi
  let area_ABC := (1/2) * x * 3 * Real.sin θ
  let area_BCD := (1/2) * 3 * 5 * Real.sin (φ - θ)
  area_ABC + area_BCD

theorem max_area_quad (x : ℝ) (h : x > 0)
  (BC_eq_3 : True)
  (CD_eq_5 : True)
  (centroids_form_isosceles : True) :
  MaxAreaABCD = 12 := by
  sorry

end max_area_quad_l2350_235045


namespace ratio_x_y_l2350_235057

theorem ratio_x_y (x y : ℝ) (h : (3 * x^2 - y) / (x + y) = 1 / 2) : 
  x / y = 3 / (6 * x - 1) := 
sorry

end ratio_x_y_l2350_235057


namespace diff_roots_eq_sqrt_2p2_add_2p_sub_2_l2350_235046

theorem diff_roots_eq_sqrt_2p2_add_2p_sub_2 (p : ℝ) :
  let a := 1
  let b := -2 * p
  let c := p^2 - p + 1
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let r1 := (-b + sqrt_discriminant) / (2 * a)
  let r2 := (-b - sqrt_discriminant) / (2 * a)
  r1 - r2 = Real.sqrt (2*p^2 + 2*p - 2) :=
by
  sorry

end diff_roots_eq_sqrt_2p2_add_2p_sub_2_l2350_235046


namespace dollars_saved_is_correct_l2350_235047

noncomputable def blender_in_store_price : ℝ := 120
noncomputable def juicer_in_store_price : ℝ := 80
noncomputable def blender_tv_price : ℝ := 4 * 28 + 12
noncomputable def total_in_store_price_with_discount : ℝ := (blender_in_store_price + juicer_in_store_price) * 0.90
noncomputable def dollars_saved : ℝ := total_in_store_price_with_discount - blender_tv_price

theorem dollars_saved_is_correct :
  dollars_saved = 56 := by
  sorry

end dollars_saved_is_correct_l2350_235047


namespace exist_divisible_n_and_n1_l2350_235090

theorem exist_divisible_n_and_n1 (d : ℕ) (hd : 0 < d) :
  ∃ (n n1 : ℕ), n % d = 0 ∧ n1 % d = 0 ∧ n ≠ n1 ∧
  (∃ (k a b c : ℕ), b ≠ 0 ∧ n = 10^k * (10 * a + b) + c ∧ n1 = 10^k * a + c) :=
by
  sorry

end exist_divisible_n_and_n1_l2350_235090


namespace chess_tournament_games_l2350_235068

theorem chess_tournament_games (n : ℕ) (h : n = 25) : 2 * n * (n - 1) = 1200 :=
by
  sorry

end chess_tournament_games_l2350_235068


namespace roots_expression_value_l2350_235025

theorem roots_expression_value {m n : ℝ} (h₁ : m^2 - 3 * m - 2 = 0) (h₂ : n^2 - 3 * n - 2 = 0) : 
  (7 * m^2 - 21 * m - 3) * (3 * n^2 - 9 * n + 5) = 121 := 
by 
  sorry

end roots_expression_value_l2350_235025


namespace div_fraction_l2350_235065

/-- The result of dividing 3/7 by 2 1/2 equals 6/35 -/
theorem div_fraction : (3/7) / (2 + 1/2) = 6/35 :=
by 
  sorry

end div_fraction_l2350_235065


namespace task1_on_time_task2_not_on_time_prob_l2350_235008

def task1_on_time_prob : ℚ := 3 / 8
def task2_on_time_prob : ℚ := 3 / 5

theorem task1_on_time_task2_not_on_time_prob :
  task1_on_time_prob * (1 - task2_on_time_prob) = 3 / 20 := by
  sorry

end task1_on_time_task2_not_on_time_prob_l2350_235008


namespace tank_capacity_l2350_235093

theorem tank_capacity
  (w c : ℝ)
  (h1 : w / c = 1 / 3)
  (h2 : (w + 5) / c = 2 / 5) :
  c = 75 :=
by
  sorry

end tank_capacity_l2350_235093


namespace money_left_after_spending_l2350_235048

def initial_money : ℕ := 24
def doris_spent : ℕ := 6
def martha_spent : ℕ := doris_spent / 2
def total_spent : ℕ := doris_spent + martha_spent
def money_left := initial_money - total_spent

theorem money_left_after_spending : money_left = 15 := by
  sorry

end money_left_after_spending_l2350_235048


namespace locus_eqn_l2350_235016

noncomputable def locus_of_centers (a b : ℝ) : Prop :=
  ∃ r : ℝ, 
    (a^2 + b^2 = (r + 1)^2) ∧ 
    ((a - 2)^2 + b^2 = (5 - r)^2)

theorem locus_eqn (a b : ℝ) : 
  locus_of_centers a b ↔ 3 * a^2 + b^2 + 44 * a + 121 = 0 :=
by
  -- Proof omitted
  sorry

end locus_eqn_l2350_235016


namespace remainder_of_x_divided_by_30_l2350_235013

theorem remainder_of_x_divided_by_30:
  ∀ x : ℤ,
    (4 + x ≡ 9 [ZMOD 8]) ∧ 
    (6 + x ≡ 8 [ZMOD 27]) ∧ 
    (8 + x ≡ 49 [ZMOD 125]) ->
    (x ≡ 17 [ZMOD 30]) :=
by
  intros x h
  sorry

end remainder_of_x_divided_by_30_l2350_235013


namespace contingency_table_confidence_l2350_235027

theorem contingency_table_confidence (k_squared : ℝ) (h1 : k_squared = 4.013) : 
  confidence_99 :=
  sorry

end contingency_table_confidence_l2350_235027


namespace M_intersect_N_l2350_235001

-- Definition of the sets M and N
def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 ≤ x}

-- Proposition to be proved
theorem M_intersect_N : M ∩ N = {0, 1} := 
by 
  sorry

end M_intersect_N_l2350_235001


namespace find_radius_of_large_circle_l2350_235052

noncomputable def radius_of_large_circle (r : ℝ) : Prop :=
  let r_A := 3
  let r_B := 2
  let d := 6
  (r - r_A)^2 + (r - r_B)^2 + 2 * (r - r_A) * (r - r_B) = d^2 ∧
  r = (5 + Real.sqrt 33) / 2

theorem find_radius_of_large_circle : ∃ (r : ℝ), radius_of_large_circle r :=
by {
  sorry
}

end find_radius_of_large_circle_l2350_235052


namespace symmetrical_point_with_respect_to_x_axis_l2350_235076

-- Define the point P with coordinates (-2, -1)
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the given point
def P : Point := { x := -2, y := -1 }

-- Define the symmetry with respect to the x-axis
def symmetry_x_axis (p : Point) : Point :=
{ x := p.x, y := -p.y }

-- Verify the symmetrical point
theorem symmetrical_point_with_respect_to_x_axis :
  symmetry_x_axis P = { x := -2, y := 1 } :=
by
  -- Skip the proof
  sorry

end symmetrical_point_with_respect_to_x_axis_l2350_235076


namespace problem1_problem2_l2350_235006

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 + 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - 6 * x - 72 ≤ 0) ∧ (x^2 + x - 6 > 0)

-- Problem 1: Proving the range of x
theorem problem1 (x : ℝ) (h₁ : a = -1) (h₂ : ∀ (x : ℝ), p x a → q x) : 
  x ∈ {x : ℝ | -6 ≤ x ∧ x < -3} ∨ x ∈ {x : ℝ | 1 < x ∧ x ≤ 12} := sorry

-- Problem 2: Proving the range of a
theorem problem2 (a : ℝ) (h₃ : (∀ x, q x → p x a) ∧ ¬ (∀ x, ¬q x → ¬p x a)) : 
  -4 ≤ a ∧ a ≤ -2 := sorry

end problem1_problem2_l2350_235006


namespace problem_statement_l2350_235056

theorem problem_statement
  (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 16 :=
by
  sorry

end problem_statement_l2350_235056


namespace price_of_mixture_l2350_235064

theorem price_of_mixture (P1 P2 P3 : ℝ) (h1 : P1 = 126) (h2 : P2 = 135) (h3 : P3 = 175.5) : 
  (P1 + P2 + 2 * P3) / 4 = 153 :=
by 
  -- Main goal is to show (126 + 135 + 2 * 175.5) / 4 = 153
  sorry

end price_of_mixture_l2350_235064


namespace second_batch_students_l2350_235086

theorem second_batch_students :
  ∃ x : ℕ,
    (40 * 45 + x * 55 + 60 * 65 : ℝ) / (40 + x + 60) = 56.333333333333336 ∧
    x = 50 :=
by
  use 50
  sorry

end second_batch_students_l2350_235086


namespace role_assignment_l2350_235062

theorem role_assignment (m w : ℕ) (m_roles w_roles e_roles : ℕ) 
  (hm : m = 5) (hw : w = 6) (hm_roles : m_roles = 2) (hw_roles : w_roles = 2) (he_roles : e_roles = 2) :
  ∃ (total_assignments : ℕ), total_assignments = 25200 :=
by
  sorry

end role_assignment_l2350_235062


namespace proof_problem_l2350_235012

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x
def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

theorem proof_problem (h_even : even_function f)
                      (h_period : ∀ x, f (x + 2) = -f x)
                      (h_incr : increasing_on f (-2) 0) :
                      periodic_function f 4 ∧ symmetric_about f 2 :=
by { sorry }

end proof_problem_l2350_235012


namespace more_white_birds_than_grey_l2350_235033

def num_grey_birds_in_cage : ℕ := 40
def num_remaining_birds : ℕ := 66

def num_grey_birds_freed : ℕ := num_grey_birds_in_cage / 2
def num_grey_birds_left_in_cage : ℕ := num_grey_birds_in_cage - num_grey_birds_freed
def num_white_birds : ℕ := num_remaining_birds - num_grey_birds_left_in_cage

theorem more_white_birds_than_grey : num_white_birds - num_grey_birds_in_cage = 6 := by
  sorry

end more_white_birds_than_grey_l2350_235033


namespace ratio_of_volumes_l2350_235034

theorem ratio_of_volumes (C D : ℚ) (h1: C = (3/4) * C) (h2: D = (5/8) * D) : C / D = 5 / 6 :=
sorry

end ratio_of_volumes_l2350_235034


namespace max_planes_determined_l2350_235009

-- Definitions for conditions
variables (Point Line Plane : Type)
variables (l : Line) (A B C : Point)
variables (contains : Point → Line → Prop)
variables (plane_contains_points : Plane → Point → Point → Point → Prop)
variables (plane_contains_line_and_point : Plane → Line → Point → Prop)
variables (non_collinear : Point → Point → Point → Prop)
variables (not_on_line : Point → Line → Prop)

-- Hypotheses based on the conditions
axiom three_non_collinear_points : non_collinear A B C
axiom point_not_on_line (P : Point) : not_on_line P l

-- Goal: Prove that the number of planes is 4
theorem max_planes_determined : 
  ∃ total_planes : ℕ, total_planes = 4 :=
sorry

end max_planes_determined_l2350_235009


namespace correct_operation_A_l2350_235029

-- Definitions for the problem
def division_rule (a : ℝ) (m n : ℕ) : Prop := a^m / a^n = a^(m - n)
def multiplication_rule (a : ℝ) (m n : ℕ) : Prop := a^m * a^n = a^(m + n)
def power_rule (a : ℝ) (m n : ℕ) : Prop := (a^m)^n = a^(m * n)
def addition_like_terms_rule (a : ℝ) (m : ℕ) : Prop := a^m + a^m = 2 * a^m

-- The theorem to prove
theorem correct_operation_A (a : ℝ) : division_rule a 4 2 :=
by {
  sorry
}

end correct_operation_A_l2350_235029


namespace find_common_ratio_l2350_235095

variable (a₁ : ℝ) (q : ℝ)

def S₁ (a₁ : ℝ) : ℝ := a₁
def S₃ (a₁ q : ℝ) : ℝ := a₁ + a₁ * q + a₁ * q ^ 2
def a₃ (a₁ q : ℝ) : ℝ := a₁ * q ^ 2

theorem find_common_ratio (h : 2 * S₃ a₁ q = S₁ a₁ + 2 * a₃ a₁ q) : q = -1 / 2 :=
by
  sorry

end find_common_ratio_l2350_235095


namespace complex_real_number_l2350_235066

-- Definition of the complex number z
def z (a : ℝ) : ℂ := (a^2 + 2011) + (a - 1) * Complex.I

-- The proof problem statement
theorem complex_real_number (a : ℝ) (h : z a = (a^2 + 2011 : ℂ)) : a = 1 :=
by
  sorry

end complex_real_number_l2350_235066


namespace three_digit_integer_condition_l2350_235091

theorem three_digit_integer_condition (n a b c : ℕ) (hn : 100 ≤ n ∧ n < 1000)
  (hdigits : n = 100 * a + 10 * b + c)
  (hdadigits : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (fact_condition : 2 * n / 3 = a.factorial * b.factorial * c.factorial) :
  n = 432 := sorry

end three_digit_integer_condition_l2350_235091


namespace fraction_of_work_completed_in_25_days_l2350_235094

def men_init : ℕ := 100
def days_total : ℕ := 50
def hours_per_day_init : ℕ := 8
def days_first : ℕ := 25
def men_add : ℕ := 60
def hours_per_day_later : ℕ := 10

theorem fraction_of_work_completed_in_25_days : 
  (men_init * days_first * hours_per_day_init) / (men_init * days_total * hours_per_day_init) = 1 / 2 :=
  by sorry

end fraction_of_work_completed_in_25_days_l2350_235094
