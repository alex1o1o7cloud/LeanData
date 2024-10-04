import Mathlib

namespace find_angle_A_max_triangle_area_l776_776252

noncomputable def angle_condition (b c a : ℝ) (A C : ℝ) : Prop :=
  2 * b * Real.cos A = c * Real.cos A + a * Real.cos C

theorem find_angle_A {a b c A C : ℝ} 
  (h1: angle_condition b c a A C) : A = Real.pi / 3 :=
sorry

theorem max_triangle_area {a b c A : ℝ} 
  (hA : A = Real.pi / 3)
  (ha : a = 4) : 
  let max_area := (4 : ℝ) * Real.sqrt 3 in
  Real.abs ((1 / 2) * b * c * Real.sin A) ≤ max_area :=
sorry

end find_angle_A_max_triangle_area_l776_776252


namespace pie_prices_l776_776878

theorem pie_prices 
  (d k : ℕ) (hd : 0 < d) (hk : 1 < k) 
  (p1 p2 : ℕ) 
  (prices : List ℕ) (h_prices : prices = [64, 64, 70, 72]) 
  (hP1 : (p1 + d) ∈ prices)
  (hZ1 : (k * p1) ∈ prices)
  (hP2 : (p2 + d) ∈ prices)
  (hZ2 : (k * p2) ∈ prices)
  (h_different_prices : p1 + d ≠ k * p1 ∧ p2 + d ≠ k * p2) : 
  ∃ p1 p2, p1 ≠ p2 :=
sorry

end pie_prices_l776_776878


namespace angle_between_lines_l776_776564

noncomputable def find_angle_between_lines (a b : ℝ × ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3 in
  let mag_a := real.sqrt (a.1^2 + a.2^2 + a.3^2) in
  let mag_b := real.sqrt (b.1^2 + b.2^2 + b.3^2) in
  real.arccos (dot_product / (mag_a * mag_b))

theorem angle_between_lines :
  find_angle_between_lines (-2, -3, 2 * real.sqrt 3) (-1, 2, -real.sqrt 3) = 3 * real.pi / 4 :=
by
  sorry

end angle_between_lines_l776_776564


namespace arithmetic_seq_formula_geometric_seq_formula_sum_c_n_formula_l776_776155

-- Definitions for sequences and conditions
def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 5 = 5 ∧ ∀ n, a (n + 1) - a n = 1

def geometric_seq_sum (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = 2 - 1 / (2 ^ (n - 1))

-- Target sequences and their formulas
def a_n (n : ℕ) : ℕ := n
def b_n (n : ℕ) : ℝ := (1 / 2) ^ (n - 1)

def c_n (n : ℕ) : ℝ := (a_n n : ℝ) * b_n n

-- Sum of the first n terms of c_n
def T_n (n : ℕ) : ℝ := (Finset.range n).sum (λ i, c_n (i + 1))

-- Proof goals
theorem arithmetic_seq_formula {a : ℕ → ℕ} (h : arithmetic_seq a) : ∀ n, a n = n :=
by
  sorry

theorem geometric_seq_formula {S : ℕ → ℝ} (h : geometric_seq_sum S) :
    ∀ n, (S n) - (S (n - 1)) = (1 / 2) ^ (n - 1) :=
by
  sorry

theorem sum_c_n_formula : ∀ n, T_n n = 4 - (n + 2) / 2 ^ (n - 1) :=
by
  sorry

end arithmetic_seq_formula_geometric_seq_formula_sum_c_n_formula_l776_776155


namespace train_crossing_time_l776_776424

theorem train_crossing_time :
  ∀ (train_length platform_length : ℝ) (train_speed_kmph : ℝ),
    train_length = 180 → platform_length = 208.92 → train_speed_kmph = 70 →
    let total_distance := train_length + platform_length in
    let train_speed_mps := train_speed_kmph * (1000 / 3600) in
    (total_distance / train_speed_mps) ≈ 20 :=
by
  intros train_length platform_length train_speed_kmph 
         h_train_length h_platform_length h_train_speed_kmph
  let total_distance := train_length + platform_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  sorry

end train_crossing_time_l776_776424


namespace a_18_value_l776_776077

-- Define the concept of an "Equally Summed Sequence"
def equallySummedSequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = c

-- Define the specific conditions for a_1 and the common sum
def specific_sequence (a : ℕ → ℝ) : Prop :=
  equallySummedSequence a 5 ∧ a 1 = 2

-- The theorem we want to prove
theorem a_18_value (a : ℕ → ℝ) (h : specific_sequence a) : a 18 = 3 :=
sorry

end a_18_value_l776_776077


namespace binomial_coefficient_10_3_l776_776469

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l776_776469


namespace problem_one_problem_two_l776_776445

-- Problem 1
theorem problem_one :
  √8 + (2 * √3)^2 + |√27 - 2 * √2| = 12 + 3 * √3 := 
sorry

-- Problem 2
theorem problem_two :
  (√5 - √2) * (√5 + √2) + (2 * √2 - 4) / √2 = 5 - 2 * √2 :=
sorry

end problem_one_problem_two_l776_776445


namespace functional_eq_product_l776_776652

noncomputable def functional_eq := ∀ (f : ℝ → ℝ), 
  (∀ x y : ℝ, f(f(x) + y) = f(x^2 - y) + 4 * f(x) * y) → 
  (Set.card { y | ∃ x : ℝ, f(x) = y } = 2 ∧ 
   ∑ y in { y | ∃ x : ℝ, f(x) = y }, y = 9)

theorem functional_eq_product : 
  functional_eq → 
  2 * 9 = 18 :=
by
  sorry

end functional_eq_product_l776_776652


namespace part_a_part_b_l776_776708

-- Given a twice-differentiable function f such that for all x in ℝ, f''(x) + f(x) = 0
variable (f : ℝ → ℝ)
variable h_diff : ∀ x, deriv (deriv f) x + f x = 0

-- Part a: Prove that if f(0) = 0 and f'(0) = 0, then f ≡ 0
theorem part_a (h_initial_f0 : f 0 = 0) (h_initial_f0' : deriv f 0 = 0) : 
  ∀ x, f x = 0 :=
by
  sorry

-- Part b: Prove that there exist a, b ∈ ℝ such that f(x) = a * sin x + b * cos x
theorem part_b : ∃ a b : ℝ, ∀ x, f x = a * sin x + b * cos x :=
by
  sorry

end part_a_part_b_l776_776708


namespace tan_sub_pi_over_4_l776_776125

-- Define the conditions and the problem statement
variable (α : ℝ) (h : Real.tan α = 2)

-- State the problem as a theorem
theorem tan_sub_pi_over_4 : Real.tan (α - Real.pi / 4) = 1 / 3 :=
by
  sorry

end tan_sub_pi_over_4_l776_776125


namespace pie_wholesale_price_l776_776817

theorem pie_wholesale_price (p1 p2: ℕ) (d: ℕ) (k: ℕ) (h1: d > 0) (h2: k > 1)
  (price_list: List ℕ)
  (p_plus_1: price_list = [p1 + d, k * p1, p2 + d, k * p2]) 
  (h3: price_list.perm [64, 64, 70, 72]) : 
  p1 = 60 ∧ p2 = 60 := 
sorry

end pie_wholesale_price_l776_776817


namespace number_of_subsets_le_l776_776662

noncomputable def sum_zero (n : ℕ) (x : Fin n → ℝ) : Prop :=
  (Finset.univ.sum (λ i, x i) = 0)

noncomputable def sum_of_squares_one (n : ℕ) (x : Fin n → ℝ) : Prop :=
  (Finset.univ.sum (λ i, (x i)^2) = 1)

noncomputable def S_A (n : ℕ) (x : Fin n → ℝ) (A : Finset (Fin n)) : ℝ :=
  (A.sum (λ i, x i))

theorem number_of_subsets_le (n : ℕ) (x : Fin n → ℝ) (λ : ℝ) (h_sum_zero : sum_zero n x) (h_sum_squares_one : sum_of_squares_one n x) (h_lambda_pos : 0 < λ) : 
  (∃ N : ℕ, ( ∏ A in Finset.powerset.univ, if S_A n x A ≥ λ then 1 else 0) ≤ N ∧ N ≤ 2^(n-3) / λ^2) :=
sorry

end number_of_subsets_le_l776_776662


namespace jose_share_of_profit_l776_776014

def investment_months (amount : ℕ) (months : ℕ) : ℕ := amount * months

def profit_share (investment_months : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment_months * total_profit) / total_investment_months

theorem jose_share_of_profit :
  let tom_investment := 30000
  let jose_investment := 45000
  let total_profit := 36000
  let tom_months := 12
  let jose_months := 10
  let tom_investment_months := investment_months tom_investment tom_months
  let jose_investment_months := investment_months jose_investment jose_months
  let total_investment_months := tom_investment_months + jose_investment_months
  profit_share jose_investment_months total_investment_months total_profit = 20000 :=
by
  sorry

end jose_share_of_profit_l776_776014


namespace binomial_coefficient_10_3_l776_776487

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l776_776487


namespace combined_resistance_l776_776627

variables (x y z : ℝ)

-- Conditions
def resistance_in_parallel (x y z : ℝ) : ℝ :=
  1 / ((1 / x) + (1 / y) + (1 / z))

-- Theorem Statement
theorem combined_resistance (hx : x = 3) (hy : y = 5) (hz : z = 7) :
  resistance_in_parallel x y z = 105 / 71 :=
by sorry

end combined_resistance_l776_776627


namespace find_S7_l776_776650

variable (a_n : ℕ → ℤ) -- The arithmetic sequence
variable (S : ℕ → ℤ) -- The sum of the first n terms of the sequence
variable (d : ℤ) -- Common difference
variable (a1 : ℤ) -- First term of arithmetic sequence

-- Definition for sum of first n terms of arithmetic sequence
def arithmetic_sum (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := n * a1 + (n * (n - 1)) / 2 * d

-- Given conditions
axiom h1 : S 5 = 5
axiom h2 : S 9 = 27
axiom h3 : ∀ n, S n = arithmetic_sum a1 d n

-- Proof statement
theorem find_S7 : S 7 = 14 :=
by
  -- We identify that this axiom should immediately be applied
  rw [h3],
  sorry

end find_S7_l776_776650


namespace determine_wholesale_prices_l776_776855

variables (p1 p2 d k : ℝ)
variables (h1 : 1 < k)
variables (prices : Finset ℝ)
variables (p_plus : ℝ → ℝ) (star : ℝ → ℝ)

noncomputable def BakeryPrices : Prop :=
  p_plus p1 = p1 + d ∧ p_plus p2 = p2 + d ∧ 
  star p1 = k * p1 ∧ star p2 = k * p2 ∧ 
  prices = {64, 64, 70, 72} ∧ 
  ∃ (q1 q2 q3 q4 : ℝ), 
    (q1 = p_plus p1 ∨ q1 = star p1 ∨ q1 = p_plus p2 ∨ q1 = star p2) ∧
    (q2 = p_plus p1 ∨ q2 = star p1 ∨ q2 = p_plus p2 ∨ q2 = star p2) ∧
    (q3 = p_plus p1 ∨ q3 = star p1 ∨ q3 = p_plus p2 ∨ q3 = star p2) ∧
    (q4 = p_plus p1 ∨ q4 = star p1 ∨ q4 = p_plus p2 ∨ q4 = star p2) ∧
    prices = {q1, q2, q3, q4}

theorem determine_wholesale_prices (p1 p2 : ℝ) (d k : ℝ)
    (h1 : 1 < k) (prices : Finset ℝ) :
  BakeryPrices p1 p2 d k h1 prices (λ p, p + d) (λ p, k * p) :=
sorry

end determine_wholesale_prices_l776_776855


namespace find_all_good_sets_l776_776510

def is_good_set (A : Finset ℕ) : Prop :=
  (∀ (a b c : ℕ), a ∈ A → b ∈ A → c ∈ A → a ≠ b → b ≠ c → a ≠ c → Nat.gcd a (Nat.gcd b c) = 1) ∧
  (∀ (b c : ℕ), b ∈ A → c ∈ A → b ≠ c → ∃ (a : ℕ), a ∈ A ∧ a ≠ b ∧ a ≠ c ∧ (b * c) % a = 0)

theorem find_all_good_sets : ∀ (A : Finset ℕ), is_good_set A ↔ 
  (A = {a, b, a * b} ∧ Nat.gcd a b = 1) ∨ 
  ∃ (p q r : ℕ), Nat.gcd p q = 1 ∧ Nat.gcd q r = 1 ∧ Nat.gcd r p = 1 ∧ A = {p * q, q * r, r * p} :=
by
  sorry

end find_all_good_sets_l776_776510


namespace find_other_intersection_point_l776_776731

-- Definitions
def parabola_eq (x : ℝ) : ℝ := x^2 - 2 * x - 3
def intersection_point1 : Prop := parabola_eq (-1) = 0
def intersection_point2 : Prop := parabola_eq 3 = 0

-- Proof problem
theorem find_other_intersection_point :
  intersection_point1 → intersection_point2 := by
  sorry

end find_other_intersection_point_l776_776731


namespace greatest_divisor_of_28_l776_776102

theorem greatest_divisor_of_28 : ∀ d : ℕ, d ∣ 28 → d ≤ 28 :=
by
  sorry

end greatest_divisor_of_28_l776_776102


namespace num_distinct_four_digit_integers_l776_776606

theorem num_distinct_four_digit_integers : 
  let digits := [3, 3, 5, 5]
  let unique_digits := list.nodup digits
  (list.permutations digits).length / (fact (list.count 3 digits) * fact (list.count 5 digits)) = 6 :=
by sorry

end num_distinct_four_digit_integers_l776_776606


namespace least_positive_three_digit_multiple_of_13_is_104_l776_776782

theorem least_positive_three_digit_multiple_of_13_is_104 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ n = 104 :=
by
  existsi 104
  split
  · show 100 ≤ 104
    exact le_refl 104
  split
  · show 104 < 1000
    exact dec_trivial
  split
  · show 104 % 13 = 0
    exact dec_trivial
  · show 104 = 104
    exact rfl

end least_positive_three_digit_multiple_of_13_is_104_l776_776782


namespace mary_earns_washing_cars_l776_776665

theorem mary_earns_washing_cars:
  ∀ (C : ℕ), 
  (∀ (total_earnings : ℕ), total_earnings = C + 40 → 
   ∀ (monthly_savings : ℕ), monthly_savings = total_earnings / 2 → 
   ∀ (total_savings : ℕ), total_savings = 150 → 
   (5 * monthly_savings = total_savings)) → 
  C = 20 :=
begin
  sorry,
end

end mary_earns_washing_cars_l776_776665


namespace midpoint_polar_coordinates_l776_776623

open Real

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * cos θ, r * sin θ)

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def cartesian_to_polar (x y : ℝ) : ℝ × ℝ :=
  if x = 0 then
    (sqrt (x^2 + y^2), if y > 0 then π / 2 else 3 * π / 2)
  else
    (sqrt (x^2 + y^2), atan2 y x)

theorem midpoint_polar_coordinates :
  let A := polar_to_cartesian 10 (π / 4)
  let B := polar_to_cartesian 10 (3 * π / 4)
  let M := midpoint A B
  cartesian_to_polar M.1 M.2 = (10, π / 2) :=
by
  let A := polar_to_cartesian 10 (π / 4)
  let B := polar_to_cartesian 10 (3 * π / 4)
  let M := midpoint A B
  have h1 : A = (10 * cos (π / 4), 10 * sin (π / 4)), from rfl
  have h2 : B = (10 * cos (3 * π / 4), 10 * sin (3 * π / 4)), from rfl
  have h3 : A = (5 * sqrt 2, 5 * sqrt 2), by sorry
  have h4 : B = (-5 * sqrt 2, 5 * sqrt 2), by sorry
  have h5 : M = (0, 5 * sqrt 2), by sorry
  show cartesian_to_polar M.1 M.2 = (10, π / 2), by sorry

end midpoint_polar_coordinates_l776_776623


namespace power_function_quadrant_IV_l776_776965

theorem power_function_quadrant_IV (a : ℝ) (h : a ∈ ({-1, 1/2, 2, 3} : Set ℝ)) :
  ∀ x : ℝ, x * x^a ≠ -x * (-x^a) := sorry

end power_function_quadrant_IV_l776_776965


namespace combination_10_3_l776_776446

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l776_776446


namespace find_q0_l776_776277

-- Definitions of p, q, and r as polynomials
variables (p q r : Polynomial ℝ)

/-- Given conditions as Lean definitions/axioms -/
def constant_term (f : Polynomial ℝ) : ℝ := 
  if f.coeff 0 = 0 then 0 else f.coeff 0

axiom h1 : r = p * q
axiom h2 : constant_term p = 6
axiom h3 : constant_term r = -12

theorem find_q0 : q.eval 0 = -2 :=
by
  sorry

end find_q0_l776_776277


namespace sum_fraction_equation_solution_l776_776127

theorem sum_fraction_equation_solution (x : Fin 2014 → ℝ)
  (h : ∀ n : ℕ, 1 ≤ n → n ≤ 2014 →
    ∑ k in Finset.range 2014, x k / (n + (k + 1)) = 1 / (2 * n + 1)) :
  ∑ k in Finset.range 2014, x k / (2 * (k + 1) + 1) = 1 / 4 * (1 - 1 / 4029 ^ 2) := by
  sorry

end sum_fraction_equation_solution_l776_776127


namespace makeup_exam_probability_l776_776617

theorem makeup_exam_probability (total_students : ℕ) (students_in_makeup_exam : ℕ)
  (h1 : total_students = 42) (h2 : students_in_makeup_exam = 3) :
  (students_in_makeup_exam : ℚ) / total_students = 1 / 14 := by
  sorry

end makeup_exam_probability_l776_776617


namespace inequality_proof_l776_776976

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) (h4 : a ≠ b) : 
  sqrt a + sqrt b < sqrt 2 ∧ sqrt 2 < (1 / 2 ^ a) + (1 / 2 ^ b) :=
  sorry

end inequality_proof_l776_776976


namespace smallest_ratio_l776_776807

theorem smallest_ratio (r s : ℤ) (h1 : 3 * r ≥ 2 * s - 3) (h2 : 4 * s ≥ r + 12) : 
  (∃ r s, (r : ℚ) / s = 1 / 2) :=
by 
  sorry

end smallest_ratio_l776_776807


namespace matt_peanut_revenue_l776_776674

theorem matt_peanut_revenue
    (plantation_length : ℕ)
    (plantation_width : ℕ)
    (peanut_production : ℕ)
    (peanut_to_peanut_butter_rate_peanuts : ℕ)
    (peanut_to_peanut_butter_rate_butter : ℕ)
    (peanut_butter_price_per_kg : ℕ)
    (expected_revenue : ℕ) :
    plantation_length = 500 →
    plantation_width = 500 →
    peanut_production = 50 →
    peanut_to_peanut_butter_rate_peanuts = 20 →
    peanut_to_peanut_butter_rate_butter = 5 →
    peanut_butter_price_per_kg = 10 →
    expected_revenue = 31250 :=
by
  sorry

end matt_peanut_revenue_l776_776674


namespace integer_divisibility_count_l776_776549

theorem integer_divisibility_count :
  (finset.filter (λ n : ℕ, (1 ≤ n ∧ n ≤ 100 ∧ (2 * n^2)! % (n!^(2 * n)) = 0)) (finset.range 101)).card = 92 :=
by
  sorry

end integer_divisibility_count_l776_776549


namespace log_comparison_theorem_CauchySchwarz_inequality_theorem_trigonometric_minimum_theorem_l776_776811

noncomputable def log_comparison (n : ℕ) (hn : 0 < n) : Prop := 
  Real.log n / Real.log (n + 1) < Real.log (n + 1) / Real.log (n + 2)

theorem log_comparison_theorem (n : ℕ) (hn : 0 < n) : log_comparison n hn := 
  sorry

def inequality_CauchySchwarz (a b x y : ℝ) : Prop :=
  (a*a + b*b) * (x*x + y*y) ≥ (a*x + b*y) * (a*x + b*y)

theorem CauchySchwarz_inequality_theorem (a b x y : ℝ) : inequality_CauchySchwarz a b x y :=
  sorry

noncomputable def trigonometric_minimum (x : ℝ) : ℝ := 
  (Real.sin x)^2 + (Real.cos x)^2

theorem trigonometric_minimum_theorem : ∀ x : ℝ, trigonometric_minimum x ≥ 9 :=
  sorry

end log_comparison_theorem_CauchySchwarz_inequality_theorem_trigonometric_minimum_theorem_l776_776811


namespace servant_compensation_l776_776892

theorem servant_compensation (Rs : ℝ) :
  ∃ X, X = 800 :=
by
  -- Rs 400 (cash) + Rs 200 (uniform) = Rs 600 compensation for 9 months
  let compensation_9_months := 400 + 200
  -- 9 months is 3/4 of one year
  let fraction_of_year := 3 / 4
  -- full year's agreed amount (X)
  let X := compensation_9_months / fraction_of_year
  use X
  norm_num
  rw [fraction_of_year]
  norm_num
  exact rfl
  sorry -- proof omitted

end servant_compensation_l776_776892


namespace george_and_hannah_received_A_grades_l776_776513

-- Define students as propositions
variables (Elena Fred George Hannah : Prop)

-- Define the conditions
def condition1 : Prop := Elena → Fred
def condition2 : Prop := Fred → George
def condition3 : Prop := George → Hannah
def condition4 : Prop := ∃ A1 A2 : Prop, A1 ∧ A2 ∧ (A1 ≠ A2) ∧ (A1 = George ∨ A1 = Hannah) ∧ (A2 = George ∨ A2 = Hannah)

-- The theorem to be proven: George and Hannah received A grades
theorem george_and_hannah_received_A_grades :
  condition1 Elena Fred →
  condition2 Fred George →
  condition3 George Hannah →
  condition4 George Hannah :=
by
  sorry

end george_and_hannah_received_A_grades_l776_776513


namespace greatest_possible_m_exists_greatest_m_l776_776038

def reverse_digits (m : ℕ) : ℕ :=
  -- Function to reverse digits of a four-digit number
  sorry 

theorem greatest_possible_m (m : ℕ) : 
  (1000 ≤ m ∧ m < 10000) ∧
  (let n := reverse_digits m in 1000 ≤ n ∧ n < 10000) ∧
  (m % 63 = 0) ∧
  (m % 11 = 0) →
  m ≤ 9696 :=
by
  sorry

theorem exists_greatest_m (m : ℕ) : 
  (1000 ≤ m ∧ m < 10000) ∧
  (let n := reverse_digits m in 1000 ≤ n ∧ n < 10000) ∧
  (m % 63 = 0) ∧
  (m % 11 = 0) →
  ∃ m, m = 9696 :=
by
  sorry

end greatest_possible_m_exists_greatest_m_l776_776038


namespace flagpole_break_height_l776_776405

noncomputable def find_break_height (AB BC: ℝ) : ℝ :=
  let AC := real.sqrt (AB^2 + BC^2)
  AC / 2

theorem flagpole_break_height : find_break_height 10 3 = real.sqrt 109 / 2 :=
by
  unfold find_break_height
  sorry

end flagpole_break_height_l776_776405


namespace part1_part2_l776_776163

open Real

noncomputable def tangent_line_at_point (f : ℝ → ℝ) (f' : ℝ → ℝ) (t : ℝ) :=
  ∃ a b : ℝ, ∀ x, f t + f' t * (x - t) = a * x + b

theorem part1 (t : ℝ) (a : ℝ) :
  (∀ x, f t + (1 / t + a) * (x - t) = 3 * x + 1)
  → a = 2 :=
by
  sorry

theorem part2 (k : ℝ) :
  (k ≤ 2 ∧ ∀ x > 1, f x > k * (1 - 3 / x) + 2 * x - 1)
  → -1 / 2 ≤ k ∧ k ≤ 2 :=
by
  sorry

end part1_part2_l776_776163


namespace min_x_plus_y_l776_776126

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
by
  sorry

end min_x_plus_y_l776_776126


namespace meaningful_sqrt_ge_2_l776_776224

theorem meaningful_sqrt_ge_2 {x : ℝ} (h : sqrt (x - 2) ≥ 0) : x ≥ 2 := sorry

end meaningful_sqrt_ge_2_l776_776224


namespace divisors_of_30_l776_776191

theorem divisors_of_30 : ∃ (n : ℕ), n = 16 ∧ (∀ d : ℤ, d ∣ 30 → (d ≤ 30 ∧ d ≥ -30)) :=
by
  sorry

end divisors_of_30_l776_776191


namespace smallest_q_is_505_l776_776323

noncomputable def smallest_q (x y z : ℕ) : ℕ :=
  (⌊x / 5⌋ + ⌊y / 5⌋ + ⌊z / 5⌋) +
  (⌊x / 25⌋ + ⌊y / 25⌋ + ⌊z / 25⌋) +
  (⌊x / 125⌋ + ⌊y / 125⌋ + ⌊z / 125⌋) +
  (⌊x / 625⌋ + ⌊y / 625⌋ + ⌊z / 625⌋)

theorem smallest_q_is_505 :
  ∀ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2026 →
  (∃ p : ℕ, x! * y! * z! = p * (10 ^ (smallest_q x y z))) →
  (∀ n : ℕ, ¬ (p % 10 = 0)) →
  smallest_q x y z = 505 :=
by
  sorry

end smallest_q_is_505_l776_776323


namespace bread_roll_combinations_l776_776815

-- Definitions based on conditions
def n : ℕ := 5
def k : ℕ := 3

-- Theorem statement
theorem bread_roll_combinations : (nat.choose (n + k - 1) (k - 1)) = 21 :=
  by sorry

end bread_roll_combinations_l776_776815


namespace integral_result_l776_776919

-- Define the integrand function
def integrand (x : ℝ) : ℝ := (3 * x^3 + 9 * x^2 + 10 * x + 2) / ((x-1) * (x+1)^3)

-- State the main theorem
theorem integral_result :
  ∫ (x : ℝ) in -∞..∞, integrand x = 3 * log (abs (x - 1)) - 1 / (2 * (x + 1)^2) + C :=
sorry

end integral_result_l776_776919


namespace exponent_identity_l776_776019

theorem exponent_identity (m : ℕ) : 5 ^ m = 5 * (25 ^ 4) * (625 ^ 3) ↔ m = 21 := by
  sorry

end exponent_identity_l776_776019


namespace lychees_remaining_l776_776676
-- Definitions of the given conditions
def initial_lychees : ℕ := 500
def sold_lychees : ℕ := initial_lychees / 2
def home_lychees : ℕ := initial_lychees - sold_lychees
def eaten_lychees : ℕ := (3 * home_lychees) / 5

-- Statement to prove
theorem lychees_remaining : home_lychees - eaten_lychees = 100 := by
  sorry

end lychees_remaining_l776_776676


namespace least_three_digit_multiple_of_13_l776_776769

-- Define what it means to be a multiple of 13
def is_multiple_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13 * k

-- Define the range of three-digit numbers
def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define our main theorem
theorem least_three_digit_multiple_of_13 : ∃ (n : ℕ), is_three_digit_number n ∧ is_multiple_of_13 n ∧
  ∀ (m : ℕ), is_three_digit_number m ∧ is_multiple_of_13 m → n ≤ m :=
begin
  -- We state the theorem without proof for simplicity
  sorry
end

end least_three_digit_multiple_of_13_l776_776769


namespace bakery_wholesale_price_exists_l776_776862

theorem bakery_wholesale_price_exists :
  ∃ (p1 p2 d k : ℕ), (d > 0) ∧ (k > 1) ∧
  ({p1 + d, p2 + d, k * p1, k * p2} = {64, 64, 70, 72}) :=
by sorry

end bakery_wholesale_price_exists_l776_776862


namespace unique_positive_root_l776_776548

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ := x^n - x^(n - 1) - (finset.range (n - 1)).sum (λ i, x^(i + 1)) - 1

theorem unique_positive_root (n : ℕ) (hn : n ≥ 2) :
  ∃! α : ℝ, α > 0 ∧ f_n n α = 0 :=
sorry

end unique_positive_root_l776_776548


namespace wholesale_prices_l776_776835

-- Definitions for the problem conditions
variable (p1 p2 d k : ℝ)
variable (h_d : d > 0)
variable (h_k : k > 1)
variable (prices : Finset ℝ)
variable (h_prices : prices = {64, 64, 70, 72})

-- The theorem statement to prove
theorem wholesale_prices :
  ∃ p1 p2, (p1 + d ∈ prices ∧ k * p1 ∈ prices) ∧ 
           (p2 + d ∈ prices ∧ k * p2 ∈ prices) ∧ 
           p1 ≠ p2
:= sorry

end wholesale_prices_l776_776835


namespace matt_peanut_revenue_l776_776673

theorem matt_peanut_revenue
    (plantation_length : ℕ)
    (plantation_width : ℕ)
    (peanut_production : ℕ)
    (peanut_to_peanut_butter_rate_peanuts : ℕ)
    (peanut_to_peanut_butter_rate_butter : ℕ)
    (peanut_butter_price_per_kg : ℕ)
    (expected_revenue : ℕ) :
    plantation_length = 500 →
    plantation_width = 500 →
    peanut_production = 50 →
    peanut_to_peanut_butter_rate_peanuts = 20 →
    peanut_to_peanut_butter_rate_butter = 5 →
    peanut_butter_price_per_kg = 10 →
    expected_revenue = 31250 :=
by
  sorry

end matt_peanut_revenue_l776_776673


namespace hexagon_transformations_l776_776899

-- Definitions for transformations
def R_a : equiv.perm (fin 6) := equiv.perm.rotate (fin 6) 1
def R_b : equiv.perm (fin 6) := R_a ^ 2
def H : equiv.perm (fin 6) :=
equiv.of_bijective
  (λ (i : fin 6), (if i.val < 3 then 5 - i.val else 5 - i.val))
  ⟨by
  { intros x y h,
    dsimp at *,
    split_ifs at h with h1 h2,
    { exact h },
    { exact h },
    { exact h },
    { exact h }, },
  by
  { intro y,
    dsimp,
    split_ifs with h1,
    { use 5 - y.val,
      simp [fin.ext_iff] },
    { use 5 - y.val,
      simp [fin.ext_iff], } }⟩
def V : equiv.perm (fin 6) :=
equiv.of_bijective
  (λ (i : fin 6), (if i.val % 2 == 1 then (i.val - 1) else (i.val + 1)))
  ⟨by
  { intros x y h,
    dsimp at *,
    split_ifs at h with h1 h2,
    { exact h },
    { exact h },
    { exact h },
    { exact h }, },
  by
  { intro y,
    dsimp,
    split_ifs with h1,
    { use (y.val + 1),
      simp [fin.ext_iff] },
    { use (y.val - 1),
      simp [fin.ext_iff], } }⟩

def transformations : list (equiv.perm (fin 6)) := [R_a, R_b, H, V]

def count_sequences : ℕ := (4 : ℕ)^24

theorem hexagon_transformations : (count_sequences = 4^24) :=
by sorry

end hexagon_transformations_l776_776899


namespace reflection_through_plane_R_l776_776648

noncomputable def reflectionMatrix (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let n : ℝ × ℝ × ℝ := (2, -1, 1)
  let dot_product (a b : ℝ × ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let scalar_mult (k : ℝ) (a : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (k * a.1, k * a.2, k * a.3)
  let subtract (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (a.1 - b.1, a.2 - b.2, a.3 - b.3)
  let n_dot_n : ℝ := dot_product n n
  let v_dot_n : ℝ := dot_product v n
  let proj_v_on_n : ℝ × ℝ × ℝ := scalar_mult (v_dot_n / n_dot_n) n
  let p : ℝ × ℝ × ℝ := subtract v proj_v_on_n
  let r : ℝ × ℝ × ℝ := scalar_mult 2 p
  subtract r v

theorem reflection_through_plane_R :
  reflectionMatrix = fun (v : ℝ × ℝ × ℝ) =>
    let matrix_R : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ :=
      λ v, ( -\frac{2}{6}, \frac{4}{6}, \frac{4}{6} ) v +
            ( \frac{8}{6}, -\frac{4}{6}, \frac{4}{6} ) v + 
            ( \frac{8}{6},  \frac{4}{6}, \frac{4}{6} ) v
    matrix_R v :=
  -- Proof goes here
  sorry

end reflection_through_plane_R_l776_776648


namespace find_q_l776_776175

open Real

noncomputable def q := (9 + 3 * Real.sqrt 5) / 2

theorem find_q (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end find_q_l776_776175


namespace sum_of_coincidental_numbers_eq_531_l776_776966

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def product_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (*) 1

def is_coincidental_number (n : ℕ) : Prop :=
  sum_of_digits n + product_of_digits n = n

def sum_of_all_coincidental_numbers (N : ℕ) : ℕ :=
  (Finset.range N).filter is_coincidental_number |>.sum id

theorem sum_of_coincidental_numbers_eq_531 : sum_of_all_coincidental_numbers 100 = 531 :=
by
  sorry

end sum_of_coincidental_numbers_eq_531_l776_776966


namespace compute_binomial_10_3_eq_120_l776_776482

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l776_776482


namespace pages_per_novel_l776_776217

theorem pages_per_novel (novels_per_month : ℕ) (months_per_year : ℕ) (total_pages_per_year : ℕ)
    (H1 : novels_per_month = 4)
    (H2 : months_per_year = 12)
    (H3 : total_pages_per_year = 9600) :
    total_pages_per_year / (novels_per_month * months_per_year) = 200 := 
by
  simp only [H1, H2, H3]
  -- remaining steps to complete the proof
  sorry

end pages_per_novel_l776_776217


namespace sum_of_cubes_mod_7_l776_776537

theorem sum_of_cubes_mod_7 :
  (∑ k in Finset.range 150, (k + 1) ^ 3) % 7 = 1 := 
sorry

end sum_of_cubes_mod_7_l776_776537


namespace wholesale_prices_l776_776834

-- Definitions for the problem conditions
variable (p1 p2 d k : ℝ)
variable (h_d : d > 0)
variable (h_k : k > 1)
variable (prices : Finset ℝ)
variable (h_prices : prices = {64, 64, 70, 72})

-- The theorem statement to prove
theorem wholesale_prices :
  ∃ p1 p2, (p1 + d ∈ prices ∧ k * p1 ∈ prices) ∧ 
           (p2 + d ∈ prices ∧ k * p2 ∈ prices) ∧ 
           p1 ≠ p2
:= sorry

end wholesale_prices_l776_776834


namespace combination_10_3_l776_776452

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l776_776452


namespace ab_value_l776_776722

theorem ab_value (a b : ℝ) (h1 : b^2 - a^2 = 4) (h2 : a^2 + b^2 = 25) : abs (a * b) = Real.sqrt (609 / 4) := 
sorry

end ab_value_l776_776722


namespace cross_product_and_scaling_l776_776098

def vector1 : ℝ × ℝ × ℝ := (3, -4, 7)
def vector2 : ℝ × ℝ × ℝ := (2, 0, 5)
def result_vector : ℝ × ℝ × ℝ := (-40, -2, 16)

theorem cross_product_and_scaling :
  (2 : ℝ) • ((vector1.2 * vector2.3 - vector1.3 * vector2.2,
              vector1.3 * vector2.1 - vector1.1 * vector2.3,
              vector1.1 * vector2.2 - vector1.2 * vector2.1) : ℝ × ℝ × ℝ) = result_vector 
  :=
by
  -- Proof goes here
  sorry

end cross_product_and_scaling_l776_776098


namespace largest_c_fractional_part_exist_n_eq_fractional_c_l776_776952

theorem largest_c_fractional_part (c : ℝ) (h : ∀ n : ℕ, n > 0 → (fract (n * real.sqrt 2) ≥ c / n)) :
  c ≤ real.sqrt 2 / 4 :=
begin
  sorry
end

theorem exist_n_eq_fractional_c (c : ℝ) (hc : c = real.sqrt 2 / 4) :
  ∃ n : ℕ, n > 0 ∧ fract (n * real.sqrt 2) = c / n :=
begin
  sorry
end

end largest_c_fractional_part_exist_n_eq_fractional_c_l776_776952


namespace johnny_hourly_wage_l776_776641

-- Define the conditions
def hours_worked : ℕ := 6
def total_earnings : ℝ := 28.5

-- Define the function to compute hourly wage
def hourly_wage (earnings : ℝ) (hours : ℕ) : ℝ := earnings / hours

-- State the theorem
theorem johnny_hourly_wage :
  hourly_wage total_earnings hours_worked = 4.75 :=
by
  -- Here the proof would be placed, we add sorry to skip the proof.
  sorry

end johnny_hourly_wage_l776_776641


namespace smallest_three_digit_multiple_of_13_l776_776788

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 13 = 0 → n ≤ m :=
⟨104, by sorry⟩

end smallest_three_digit_multiple_of_13_l776_776788


namespace sequence_property_l776_776929

open Nat

def seq_a : ℕ → ℕ 
| 0 := 1
| (n + 1) := (n + 1 - Nat.gcd (seq_a n) n) * (seq_a n)

theorem sequence_property (n : ℕ) :
  (seq_a (n + 1) / seq_a n = n ↔ Prime n ∨ n = 1) :=
by
  sorry

end sequence_property_l776_776929


namespace compute_binomial_10_3_eq_120_l776_776481

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l776_776481


namespace father_will_give_30_l776_776754

variable (x : ℝ)
variable (has_half : x / 2 = x / 2)
variable (received_30_and_three_quarters : x / 2 + 30 = 3 * x / 4)

theorem father_will_give_30:
  (x / 4 = 30) := 
begin
  sorry
end

end father_will_give_30_l776_776754


namespace bag_ratio_l776_776320

noncomputable def ratio_of_costs : ℚ := 1 / 2

theorem bag_ratio :
  ∃ (shirt_cost shoes_cost total_cost bag_cost : ℚ),
    shirt_cost = 7 ∧
    shoes_cost = shirt_cost + 3 ∧
    total_cost = 2 * shirt_cost + shoes_cost ∧
    bag_cost = 36 - total_cost ∧
    bag_cost / total_cost = ratio_of_costs :=
sorry

end bag_ratio_l776_776320


namespace sum_of_complex_numbers_l776_776743

variable (a b c d e f : ℂ)

theorem sum_of_complex_numbers (Hb : b = 5) (He : e = -2 * (a + c)) (Hsum : (a + b * complex.I) + (c + d * complex.I) + (e + f * complex.I) = 4 * complex.I) : d + f = -1 :=
sorry

end sum_of_complex_numbers_l776_776743


namespace compute_binomial_10_3_eq_120_l776_776480

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l776_776480


namespace smallest_three_digit_multiple_of_eleven_l776_776766

theorem smallest_three_digit_multiple_of_eleven : ∃ n, n = 110 ∧ 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n := by
  sorry

end smallest_three_digit_multiple_of_eleven_l776_776766


namespace log2_75_in_terms_of_a_b_l776_776972

noncomputable def log_base2 (x : ℝ) : ℝ := Real.log x / Real.log 2

variables (a b : ℝ)
variables (log2_9_eq_a : log_base2 9 = a)
variables (log2_5_eq_b : log_base2 5 = b)

theorem log2_75_in_terms_of_a_b : log_base2 75 = (1 / 2) * a + 2 * b :=
by sorry

end log2_75_in_terms_of_a_b_l776_776972


namespace car_return_speed_l776_776882

variable (d : ℕ) (r : ℕ)
variable (H0 : d = 180)
variable (H1 : ∀ t1 : ℕ, t1 = d / 90)
variable (H2 : ∀ t2 : ℕ, t2 = d / r)
variable (H3 : ∀ avg_rate : ℕ, avg_rate = 2 * d / (d / 90 + d / r))
variable (H4 : avg_rate = 60)

theorem car_return_speed : r = 45 :=
by sorry

end car_return_speed_l776_776882


namespace coefficient_of_fourth_term_l776_776948

theorem coefficient_of_fourth_term :
  let term_coefficient := (8.choose 3) * (-1/8)
  term_coefficient = -7 :=
by
  let term_coefficient := (8.choose 3) * (-1/8)
  sorry

end coefficient_of_fourth_term_l776_776948


namespace variance_eta_l776_776583

variables (ξ : ℝ → ℝ) (η : ℝ → ℝ)

-- Given conditions as definitions in Lean 4
def variance_xi := 4
def eta := λ ξ, 5 * ξ - 4

-- The statement to prove
theorem variance_eta (Dξ : ℝ) (h1 : Dξ = variance_xi) (η : ℝ → ℝ) (h2 : η = λ ξ, 5 * ξ - 4) : 
  ∃ Dη : ℝ, Dη = 25 * Dξ :=
by
  use 25 * Dξ
  have key : η = λ ξ, 5 * ξ - 4 := by assumption
  have vxi : Dξ = 4 := by assumption
  sorry

end variance_eta_l776_776583


namespace part1_cover_part2_range_l776_776286

def f1 (x : ℝ) : ℝ := -((1/2) ^ |x|)

def g1 (x : ℝ) : ℝ := 2 * sin (2 * x - (real.pi / 3))

def f2 (x : ℝ) : ℝ := real.log (2 ^ x + 2) / real.log (2 ^ x + 1)

def g2 (x : ℝ) (a : ℝ) : ℝ :=
if x > 1 then real.log x
else a * x ^ 2 + (2 * a - 3) * x + 1

theorem part1_cover (x : ℝ) :
  (0 ≤ x) → 
  (x ≤ 2 * real.pi) → 
  ∃ (n : ℕ), (n = 4) →
  ∃ (x1 x2 x3 x4 : ℝ), 
    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧ 
    (forall i, i ∈ [x1, x2, x3, x4].to_finset →
    g1 i = f1 x) :=
sorry

theorem part2_range (a : ℝ) :
  (∀ (x : ℝ),
    g2 x a = f2 x →
    ∃ (xa : ℝ),
    g2 xa a ≠ f2 x) →
    (a ≤ (2/3)) :=
sorry

end part1_cover_part2_range_l776_776286


namespace value_of_k_l776_776219

-- Define the lines l1 and l2
def l1 (k : ℝ) : ℝ × ℝ × ℝ := (1, 1 + k, -(2 - k))
def l2 (k : ℝ) : ℝ × ℝ × ℝ := (k, 2, 8)

-- Define the slope of a line in the form Ax + By + C = 0
def slope (A B : ℝ) : ℝ := -A / B

-- Define the condition that two lines are parallel
def parallel (k : ℝ) : Prop :=
  slope 1 (1 + k) = slope k 2

-- State the theorem to be proved
theorem value_of_k (k : ℝ) (h : parallel k) : k = 1 :=
  sorry

end value_of_k_l776_776219


namespace curve_C_cartesian_eq_line_l_general_eq_max_area_triangle_PAB_l776_776592

-- Definitions for the conditions
def curve_C_polar (ρ θ : ℝ) := ρ = 4 * Real.sin θ
def line_l_parametric (x y t : ℝ) := 
  x = (Real.sqrt 3 / 2) * t ∧ 
  y = 1 + (1 / 2) * t

-- Theorem statements
theorem curve_C_cartesian_eq : ∀ x y : ℝ,
  (∃ (ρ θ : ℝ), curve_C_polar ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  x^2 + (y - 2)^2 = 4 :=
by sorry

theorem line_l_general_eq : ∀ x y t : ℝ,
  line_l_parametric x y t →
  x - (Real.sqrt 3) * y + Real.sqrt 3 = 0 :=
by sorry

theorem max_area_triangle_PAB : ∀ (P A B : ℝ × ℝ),
  (∃ (θ : ℝ), P = ⟨2 * Real.cos θ, 2 + 2 * Real.sin θ⟩ ∧
   (∃ t : ℝ, line_l_parametric A.1 A.2 t) ∧
   (∃ t' : ℝ, line_l_parametric B.1 B.2 t') ∧
   A ≠ B) →
  (1/2) * Real.sqrt 13 * (2 + Real.sqrt 3 / 2) = (4 * Real.sqrt 13 + Real.sqrt 39) / 4 :=
by sorry

end curve_C_cartesian_eq_line_l_general_eq_max_area_triangle_PAB_l776_776592


namespace car_return_speed_l776_776884

theorem car_return_speed (d : ℕ) (r : ℕ) (h₁ : d = 180) (h₂ : (2 * d) / ((d / 90) + (d / r)) = 60) : r = 45 :=
by
  rw [h₁] at h₂
  have h3 : 2 * 180 / ((180 / 90) + (180 / r)) = 60 := h₂
  -- The rest of the proof involves solving for r, but here we only need the statement
  sorry

end car_return_speed_l776_776884


namespace mul_example_l776_776091

theorem mul_example : (3.6 * 0.5 = 1.8) := by
  sorry

end mul_example_l776_776091


namespace mean_median_mode_seq_equal_l776_776507

-- Define the sequence and necessary conditions
def sequence : List ℕ := [2, 3, 4, 6, 6, 7, 9]
def y : ℕ := 11

-- Define a function to calculate the mean of a list of natural numbers
def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

-- Define a function to calculate the median of a list of natural numbers (assuming sorted list)
def median (l : List ℕ) : ℚ :=
  if hl : (l.length % 2 = 1) then
    (l.nthLe (l.length / 2) (Nat.div_lt_self l.length 1 (Nat.succ_pos 1))).toℚ
  else
    ((l.nthLe (l.length / 2 - 1) (Nat.sub_pos_of_lt (Nat.div_lt_self l.length 1 (Nat.succ_pos 1)))).toℚ +
     (l.nthLe (l.length / 2) (Nat.div_lt_self l.length 1 (Nat.succ_pos 1))).toℚ) / 2

-- Define a function to calculate the mode of a list of natural numbers
def mode (l : List ℕ) : ℕ :=
  l.foldl (λ (acc : List (ℕ × ℕ)) n, 
    has_key_or_add acc n.succ) 
  |> List.sort (λ a b, b.snd < a.snd)
  |> List.head!.fst

-- The statement to be proved
theorem mean_median_mode_seq_equal : 
  mean (sequence ++ [y]) = 6 ∧ 
  median (sequence ++ [y]).sort = 6 ∧ 
  mode (sequence ++ [y]) = 6 :=
by
  sorry

end mean_median_mode_seq_equal_l776_776507


namespace sqrt_difference_l776_776737

theorem sqrt_difference :
  let a := 7 + 4 * Real.sqrt 3 in
  let b := 7 - 4 * Real.sqrt 3 in
  Real.sqrt a - Real.sqrt b = 2 * Real.sqrt 3 :=
by
  let a := 7 + 4 * Real.sqrt 3
  let b := 7 - 4 * Real.sqrt 3
  sorry

end sqrt_difference_l776_776737


namespace smallest_three_digit_multiple_of_13_l776_776790

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 13 = 0 → n ≤ m :=
⟨104, by sorry⟩

end smallest_three_digit_multiple_of_13_l776_776790


namespace x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2_l776_776314

theorem x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2 
  (x : ℤ) (p m n : ℕ) (hp : 0 < p) (hm : 0 < m) (hn : 0 < n) :
  (x^2 + x + 1) ∣ (x^(3 * p) + x^(3 * m + 1) + x^(3 * n + 2)) :=
by
  sorry

end x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2_l776_776314


namespace find_angle_between_vectors_l776_776601

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def angle_between_vectors (a b : V) : ℝ :=
  real.arccos ((inner_product_space.inner a b) / (∥a∥ * ∥b∥))

theorem find_angle_between_vectors (a b : V) 
  (h1 : inner (2 • a - b) (a + b) = 6) 
  (h2 : ∥a∥ = 2) 
  (h3 : ∥b∥ = 1) :
  angle_between_vectors a b = 2 * real.pi / 3 :=
begin
  sorry
end

end find_angle_between_vectors_l776_776601


namespace solve_for_n_l776_776700

theorem solve_for_n (n : ℕ) : 
  9^n * 9^n * 9^(2*n) = 81^4 → n = 2 :=
by
  sorry

end solve_for_n_l776_776700


namespace value_at_pi_over_12_decreasing_interval_of_f_l776_776161

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x + Real.cos x) ^ 2 + 2 * (Real.cos x) ^ 2

theorem value_at_pi_over_12 :
  f (Real.pi / 12) = (5 + Real.sqrt 3) / 2 :=
sorry

theorem decreasing_interval_of_f (k : ℤ) :
  ∀ x, k * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 8 → 
  f' x < 0 :=
sorry

end value_at_pi_over_12_decreasing_interval_of_f_l776_776161


namespace digit_inequality_l776_776322

theorem digit_inequality (d : ℕ) :
  (∃ k : ℕ, k = 9 ∧ ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, (d ≥ 3 ∧ d ≤ 9) → 3 + d.toReal / 1000 > 3.003) := 
begin
  sorry
end

end digit_inequality_l776_776322


namespace sqrt_two_irrational_l776_776798

theorem sqrt_two_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (ℝ.sqrt 2 = (p : ℝ) / (q : ℝ)) :=
sorry

end sqrt_two_irrational_l776_776798


namespace real_solution_2015x_equation_l776_776080

theorem real_solution_2015x_equation (k : ℝ) :
  (∃ x : ℝ, (4 * 2015^x - 2015^(-x)) / (2015^x - 3 * 2015^(-x)) = k) ↔ (k < 1/3 ∨ k > 4) := 
by sorry

end real_solution_2015x_equation_l776_776080


namespace correct_statement_D_l776_776908

theorem correct_statement_D
    (A : ¬(∀ (s : List ℝ), (∀ x ∈ s, x < s.avg) → (s.avg > List.maximum s ∧ s.avg < List.minimum s)))
    (B : ¬(∀ (s : List ℝ), (¬(s.modes.length > 1))))
    (C : ¬(∀ (s : List ℝ), (s.length % 2 = 0 → (s.median ∉ s))))
    (D : ∀ (s : List ℝ), (variance s > 0) → (fluctuation s > 0))
    : D := 
sorry

end correct_statement_D_l776_776908


namespace melted_ice_cream_depth_l776_776418

noncomputable def radius_sphere : ℝ := 3
noncomputable def radius_cylinder : ℝ := 10
noncomputable def height_cylinder : ℝ := 36 / 100

theorem melted_ice_cream_depth :
  (4 / 3) * Real.pi * radius_sphere^3 = Real.pi * radius_cylinder^2 * height_cylinder :=
by
  sorry

end melted_ice_cream_depth_l776_776418


namespace clea_ride_escalator_time_l776_776260

theorem clea_ride_escalator_time (x y k : ℝ) (h1 : 80 * x = y) (h2 : 30 * (x + k) = y) : (y / k) + 5 = 53 :=
by {
  sorry
}

end clea_ride_escalator_time_l776_776260


namespace Raven_age_l776_776228

-- Define the conditions
def Phoebe_age_current : Nat := 10
def Phoebe_age_in_5_years : Nat := Phoebe_age_current + 5

-- Define the hypothesis that in 5 years Raven will be 4 times as old as Phoebe
def Raven_in_5_years (R : Nat) : Prop := R + 5 = 4 * Phoebe_age_in_5_years

-- State the theorem to be proved
theorem Raven_age : ∃ R : Nat, Raven_in_5_years R ∧ R = 55 :=
by
  sorry

end Raven_age_l776_776228


namespace sin_double_angle_value_l776_776021

variable (α : ℝ)

axiom cos_add_pi_div_4 : cos (α + π / 4) = (3 * sqrt 2) / 5

theorem sin_double_angle_value : sin (2 * α) = -11 / 25 := by
  have h1 : cos (α + π / 4) = (3 * sqrt 2) / 5 := cos_add_pi_div_4
  sorry

end sin_double_angle_value_l776_776021


namespace max_value_f1_l776_776589

-- Definitions for the conditions
def f (x a b : ℝ) : ℝ := x^2 + a * b * x + a + 2 * b

-- Lean theorem statements
theorem max_value_f1 (a b : ℝ) (h : a + 2 * b = 4) :
  f 0 a b = 4 → f 1 a b ≤ 7 :=
sorry

end max_value_f1_l776_776589


namespace evaluate_expression_l776_776699

theorem evaluate_expression : (2 * (-1) + 3) * (2 * (-1) - 3) - ((-1) - 1) * ((-1) + 5) = 3 := by
  sorry

end evaluate_expression_l776_776699


namespace difference_between_lengths_l776_776005

noncomputable def difference_between_lengths_of_AP_and_CP (AP CP BP DP : ℝ) (angleB angleAPC : ℝ) : ℝ :=
  if h1 : AP + CP = 3 ∧
          DP^2 = AP^2 + CP^2 - AP * CP * Math.cos(angleB) ∧
          angleB = 60 ∧ angleAPC = 120 ∧ BP = 3 ∧ DP = 2
  then (AP - CP).abs
  else 0

-- Theorem to be proven: 
theorem difference_between_lengths (AP CP : ℝ) :
  (difference_between_lengths_of_AP_and_CP AP CP 3 2 60 120) = sqrt(21) / 3 :=
sorry

end difference_between_lengths_l776_776005


namespace sum_nonnegative_integers_sum_natural_numbers_l776_776010

-- Definition for binomial coefficient
def binom (n k : ℕ) : ℕ :=
  if k > n then 0 else nat.choose n k

-- Part (a)
theorem sum_nonnegative_integers (n m : ℕ) : 
  ∃ c : ℕ, c = binom (n + m - 1) n :=
by
  existsi (binom (n + m - 1) n)
  sorry

-- Part (b)
theorem sum_natural_numbers (n m : ℕ) : 
  ∃ c : ℕ, c = binom (n - 1) (n - m) :=
by
  existsi (binom (n - 1) (n - m))
  sorry

end sum_nonnegative_integers_sum_natural_numbers_l776_776010


namespace hexagon_equiangular_sides_eq_construct_hexagon_sides_l776_776329

-- Problem 1 (a):
theorem hexagon_equiangular_sides_eq (A B C D E F : Type _) [convex_hexagon A B C D E F] :
  (angle A B C = angle B C D) ∧ (angle B C D = angle C D E) ∧ (angle C D E = angle D E F) ∧ 
  (angle D E F = angle E F A) ∧ (angle E F A = angle F A B) →
  (AB - DE = EF - BC) ∧ (EF - BC = CD - FA) :=
by sorry

-- Problem 2 (b):
theorem construct_hexagon_sides (a1 a2 a3 a4 a5 a6 : ℝ) :
  a1 - a4 = a5 - a2 ∧ a5 - a2 = a3 - a6 →
  ∃ (hexagon : Type _) (A B C D E F : hexagon) [equiangular_hexagon hexagon A B C D E F], 
  (side_length A B = a1) ∧ (side_length B C = a2) ∧ (side_length C D = a3) ∧ 
  (side_length D E = a4) ∧ (side_length E F = a5) ∧ (side_length F A = a6) :=
by sorry

end hexagon_equiangular_sides_eq_construct_hexagon_sides_l776_776329


namespace problem_equivalence_l776_776903

theorem problem_equivalence : (7^2 - 3^2)^4 = 2560000 :=
by
  sorry

end problem_equivalence_l776_776903


namespace lychees_remaining_l776_776679

theorem lychees_remaining 
  (initial_lychees : ℕ) 
  (sold_fraction taken_fraction : ℕ → ℕ) 
  (initial_lychees = 500) 
  (sold_fraction = λ x, x / 2) 
  (taken_fraction = λ x, x * 2 / 5) 
  : initial_lychees - sold_fraction initial_lychees - taken_fraction (initial_lychees - sold_fraction initial_lychees) = 100 := 
by 
  sorry

end lychees_remaining_l776_776679


namespace distances_equal_l776_776685

theorem distances_equal (A B : ℝ) (n : ℕ) (points : fin n → ℝ) 
    (colors : fin (2 * n) → bool) :
    (∀ i, points i + points (n + i) = 2 * ((A + B) / 2)) →
    (∑ i in finset.range n, if colors i then abs (points i - A) else 0) =
    (∑ i in finset.range n, if ¬ colors i then abs (points (n + i) - B) else 0) :=
by 
-- sorry allows us to skip the proof as per the instructions.
sorry

end distances_equal_l776_776685


namespace combination_10_3_l776_776455

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l776_776455


namespace volume_frustum_as_fraction_of_original_l776_776047

theorem volume_frustum_as_fraction_of_original :
  let original_base_edge := 40
  let original_altitude := 20
  let smaller_altitude := original_altitude / 3
  let smaller_base_edge := original_base_edge / 3
  let volume_original := (1 / 3) * (original_base_edge * original_base_edge) * original_altitude
  let volume_smaller := (1 / 3) * (smaller_base_edge * smaller_base_edge) * smaller_altitude
  let volume_frustum := volume_original - volume_smaller
  (volume_frustum / volume_original) = (87 / 96) :=
by
  let original_base_edge := 40
  let original_altitude := 20
  let smaller_altitude := original_altitude / 3
  let smaller_base_edge := original_base_edge / 3
  let volume_original := (1 / 3) * (original_base_edge * original_base_edge) * original_altitude
  let volume_smaller := (1 / 3) * (smaller_base_edge * smaller_base_edge) * smaller_altitude
  let volume_frustum := volume_original - volume_smaller
  have h : volume_frustum / volume_original = 87 / 96 := sorry
  exact h

end volume_frustum_as_fraction_of_original_l776_776047


namespace find_wholesale_prices_l776_776843

-- Definitions based on conditions
def bakerPlusPrice (p : ℕ) (d : ℕ) : ℕ := p + d
def starPrice (p : ℕ) (k : ℚ) : ℕ := (p * k).toNat

-- The tuple of prices given
def prices : List ℕ := [64, 64, 70, 72]

-- Statement of the problem
theorem find_wholesale_prices (d : ℕ) (k : ℚ) (h_d : d > 0) (h_k : k > 1 ∧ k.denom = 1):
  ∃ p1 p2, 
  (p1 ∈ prices ∧ bakerPlusPrice p1 d ∈ prices ∧ starPrice p1 k ∈ prices) ∧
  (p2 ∈ prices ∧ bakerPlusPrice p2 d ∈ prices ∧ starPrice p2 k ∈ prices) :=
sorry

end find_wholesale_prices_l776_776843


namespace picnic_basket_cost_l776_776738

theorem picnic_basket_cost :
  let sandwich_cost := 5
  let fruit_salad_cost := 3
  let soda_cost := 2
  let snack_bag_cost := 4
  let num_people := 4
  let num_sodas_per_person := 2
  let num_snack_bags := 3
  (num_people * sandwich_cost) + (num_people * fruit_salad_cost) + (num_people * num_sodas_per_person * soda_cost) + (num_snack_bags * snack_bag_cost) = 60 :=
by
  sorry

end picnic_basket_cost_l776_776738


namespace not_affinity_function_f_C_l776_776211

-- Define the circle
def circle (x y : ℝ) := x^2 + y^2 = 9

-- Define the candidate functions
def f_A (x : ℝ) := 4 * x^3 + x
def f_B (x : ℝ) := Real.log ((5 - x) / (5 + x))
def f_C (x : ℝ) := (Real.exp x + Real.exp (-x)) / 2
def f_D (x : ℝ) := Real.tan (x / 5)

-- Define the properties of an affinity function for the circle
def is_affinity_function (f : ℝ → ℝ) := 
  f 0 = 0 ∧ 
  ∀ x, f (-x) + f x = 0

-- Prove that f_C is not an affinity function for the circle
theorem not_affinity_function_f_C : ¬ is_affinity_function f_C :=
by sorry

end not_affinity_function_f_C_l776_776211


namespace circles_externally_tangent_l776_776173

noncomputable def circle_center_radius (a b c : ℝ) : (ℝ × ℝ) × ℝ :=
let h := -(b * (4*a - b*b) / (4*a))
in ((-b / (2 * a), h), real.sqrt (h - c / a))

def distance_between_centers 
  (cx1 cy1 cx2 cy2 : ℝ) : ℝ :=
real.sqrt ((cx2 - cx1)^2 + (cy2 - cy1)^2)

def externally_tangent 
  (cx1 cy1 r1 cx2 cy2 r2 : ℝ) : Prop :=
distance_between_centers cx1 cy1 cx2 cy2 = r1 + r2

theorem circles_externally_tangent :
  externally_tangent
    (-1) (-2) (real.sqrt 2)
    2 1 (2 * real.sqrt 2) :=
by sorry

end circles_externally_tangent_l776_776173


namespace y_coordinate_of_P_is_six_over_seven_abc_sum_l776_776647

-- Define points A, B, C, D
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (-3, 2)
def C : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (4, 0)

-- Define the condition PA + PD = PB + PC = 10
def PA_plus_PD_eq_ten (P : ℝ × ℝ): Prop :=
  (real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2)) = 10

def PB_plus_PC_eq_ten (P : ℝ × ℝ): Prop :=
  (real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) + real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)) = 10

-- The problem: proving the y-coordinate of P under given conditions
theorem y_coordinate_of_P_is_six_over_seven :
  ∃ y : ℝ, PA_plus_PD_eq_ten (P.1, y) ∧ PB_plus_PC_eq_ten (P.1, y) ∧ y = 6 / 7 :=
sorry

-- Proving a + b + c + d = 13
theorem abc_sum :
  let a := 0
  let b := 6
  let c := 0
  let d := 7
  a + b + c + d = 13 :=
by 
  trivial

end y_coordinate_of_P_is_six_over_seven_abc_sum_l776_776647


namespace team_B_from_city_A_matches_l776_776234

/-- In a football invitational involving sixteen cities, each city sends two teams: team A and team B.
At most one match is held between any two teams, and teams from the same city do not compete against each other.
After the matches are concluded, it was found that, except for team A from city A, the number of matches each team has played is different.
Prove that team B from city A has played 15 matches. -/
theorem team_B_from_city_A_matches :
  ∀ (cities : Fin 16) (teams : cities → Fin 2 → Fin 32) (matches : Fin 32 → Fin 32 → Prop),
  (∀ a b, a ≠ b → matches a b → matches b a) →
  (∀ a, matches a a = false) →
  (∀ city, matches (teams city 0) (teams city 1) = false) →
  (∃ (a : Fin 32),
    (∀ b, b ≠ a → (∃ x : Fin 32, matches b x = false) → ∃ y, y ≠ x ∧ matches b y)) →
  ∃ matches_team_B_A : Fin 32, matches_team_B_A = 15 :=
by sorry

end team_B_from_city_A_matches_l776_776234


namespace incorrect_statement_C_l776_776003

def coeff_of_term (term : String) : Int := 
  if term = "-ab²c" then -1 else 0

def degree_of_term (term : String) : Int := 
  if term = "-ab²c" then 4 else 0

def is_polynomial (expr : String) : Bool := 
  if expr = "2xy-1" then true else false

def terms_of_expr (expr : String) : List String := 
  if expr = "6x²-3x+1" then ["6x²", "-3x", "1"] else []

def expression_type (expr : String) : String := 
  if expr = "2r+πr²" then "quadratic binomial" else ""

theorem incorrect_statement_C : ¬ (expression_type "2r+πr²" = "cubic binomial") :=
by 
  have h1: coeff_of_term "-ab²c" = -1 := rfl
  have h2: degree_of_term "-ab²c" = 4 := rfl
  have h3: is_polynomial "2xy-1" = true := rfl 
  have h4: terms_of_expr "6x²-3x+1" = ["6x²", "-3x", "1"] := rfl
  show ¬ (expression_type "2r+πr²" = "cubic binomial")
  from
    have h5: expression_type "2r+πr²" = "quadratic binomial"
    from rfl,
    h5 ▸ (by contradiction)

end incorrect_statement_C_l776_776003


namespace least_three_digit_multiple_of_13_l776_776770

-- Define what it means to be a multiple of 13
def is_multiple_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13 * k

-- Define the range of three-digit numbers
def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define our main theorem
theorem least_three_digit_multiple_of_13 : ∃ (n : ℕ), is_three_digit_number n ∧ is_multiple_of_13 n ∧
  ∀ (m : ℕ), is_three_digit_number m ∧ is_multiple_of_13 m → n ≤ m :=
begin
  -- We state the theorem without proof for simplicity
  sorry
end

end least_three_digit_multiple_of_13_l776_776770


namespace divisibility_56786730_polynomial_inequality_l776_776386

theorem divisibility_56786730 (m n : ℤ) : 56786730 ∣ m * n * (m^60 - n^60) :=
sorry

theorem polynomial_inequality (m n : ℤ) : m^5 + 3 * m^4 * n - 5 * m^3 * n^2 - 15 * m^2 * n^3 + 4 * m * n^4 + 12 * n^5 ≠ 33 :=
sorry

end divisibility_56786730_polynomial_inequality_l776_776386


namespace sum_c_l776_776134

section problem

variable {a : ℕ → ℕ} {b : ℕ → ℕ} {c : ℕ → ℕ}

/-- Definition of the arithmetic sequence {a_n} -/
def a_n (n : ℕ) : ℕ :=
  2 * n - 1

/-- Definition of the geometric sequence {b_n} -/
def b_n (n : ℕ) : ℕ :=
  3 ^ (n - 1)

/-- Definition of the sequence {c_n} satisfying the given conditions -/
def c_n (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2 * 3 ^ (n - 1)

/-- Prove the sum of c_1 to c_2015 equals 3^2015 -/
theorem sum_c (n : ℕ) (hn : n = 2015) : 
  ∑ i in finset.range n.succ, c_n (i+1) = 3 ^ 2015 :=
sorry

end problem

end sum_c_l776_776134


namespace unit_digit_product_l776_776390

-- Definition of unit digit function
def unit_digit (n : Nat) : Nat := n % 10

-- Conditions about unit digits of given powers
lemma unit_digit_3_pow_68 : unit_digit (3 ^ 68) = 1 := by sorry
lemma unit_digit_6_pow_59 : unit_digit (6 ^ 59) = 6 := by sorry
lemma unit_digit_7_pow_71 : unit_digit (7 ^ 71) = 3 := by sorry

-- Main statement
theorem unit_digit_product : unit_digit (3 ^ 68 * 6 ^ 59 * 7 ^ 71) = 8 := by
  have h3 := unit_digit_3_pow_68
  have h6 := unit_digit_6_pow_59
  have h7 := unit_digit_7_pow_71
  sorry

end unit_digit_product_l776_776390


namespace comb_10_3_eq_120_l776_776503

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l776_776503


namespace divisors_of_30_count_l776_776189

theorem divisors_of_30_count : 
  ∃ S : Set ℤ, (∀ d ∈ S, 30 % d = 0) ∧ S.card = 16 :=
by
  sorry

end divisors_of_30_count_l776_776189


namespace number_of_runners_l776_776643

theorem number_of_runners (signed_up_last_year: ℕ) (did_not_show_up_last_year: ℕ) (factor: ℕ) :
  signed_up_last_year = 200 → 
  did_not_show_up_last_year = 40 → 
  factor = 2 → 
  (signed_up_last_year - did_not_show_up_last_year) * factor = 320 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end number_of_runners_l776_776643


namespace sin_arccos_eq_l776_776068

theorem sin_arccos_eq : sin (arccos (12 / 13)) = 5 / 13 := by
  sorry

end sin_arccos_eq_l776_776068


namespace pq_sum_l776_776332

variable {R : Type} [Ring R]

def p (x : R) := x - 1
def q (x : R) := (x + 2) * (x - 11 / 5)

theorem pq_sum (x : R) (h_asymptote_y0 : ∀ x, p x / q x → (λ x => 0)) 
  (h_vertical_asymptote : q (-2) = 0)
  (h_q_quadratic : ∃ a b c, q x = a * x^2 + b * x + c)
  (h_p_3 : p 3 = 2)
  (h_q_3 : q 3 = 4) :
  p x + q x = x^2 + (6 / 5) * x - 27 / 5 := by
  sorry

end pq_sum_l776_776332


namespace combination_10_3_l776_776448

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l776_776448


namespace pie_prices_l776_776877

theorem pie_prices 
  (d k : ℕ) (hd : 0 < d) (hk : 1 < k) 
  (p1 p2 : ℕ) 
  (prices : List ℕ) (h_prices : prices = [64, 64, 70, 72]) 
  (hP1 : (p1 + d) ∈ prices)
  (hZ1 : (k * p1) ∈ prices)
  (hP2 : (p2 + d) ∈ prices)
  (hZ2 : (k * p2) ∈ prices)
  (h_different_prices : p1 + d ≠ k * p1 ∧ p2 + d ≠ k * p2) : 
  ∃ p1 p2, p1 ≠ p2 :=
sorry

end pie_prices_l776_776877


namespace max_rooks_on_chessboard_l776_776381

theorem max_rooks_on_chessboard (m n k : ℕ) (h_m : m = 20) (h_n : n = 20) (h_k : k = 18) :
  ∃ max_rooks : ℕ, max_rooks = 38 ∧ 
  ∀ (board : fin m × fin n → bool), 
    (∀ i j, i < m ∧ j < n → board (⟨i, h_m⟩, ⟨j, h_n⟩) = tt → (∃ p q, i ≠ p ∧ j ≠ q ∧ board (⟨p, h_m⟩, ⟨j, h_n⟩) = tt ∧ board (⟨i, h_m⟩, ⟨q, h_n⟩) = tt)) →
    ∃ rooks : fin m × fin n → bool, 
      (∀ i j, i < m ∧ j < n → rooks (⟨i, h_m⟩, ⟨j, h_n⟩) = tt → board (⟨i, h_m⟩, ⟨j, h_n⟩) = ff) ∧ 
      (∑ i j, if rooks (⟨i, h_m⟩, ⟨j, h_n⟩) then 1 else 0 = max_rooks) :=
sorry

end max_rooks_on_chessboard_l776_776381


namespace sum_of_angles_l776_776244

theorem sum_of_angles (ABC ABD : ℝ) (n_octagon n_triangle : ℕ) 
(h1 : n_octagon = 8) 
(h2 : n_triangle = 3) 
(h3 : ABC = 180 * (n_octagon - 2) / n_octagon)
(h4 : ABD = 180 * (n_triangle - 2) / n_triangle) : 
ABC + ABD = 195 :=
by {
  sorry
}

end sum_of_angles_l776_776244


namespace bakery_made_muffins_l776_776715

-- Definitions based on conditions
def muffins_per_box : ℕ := 5
def available_boxes : ℕ := 10
def additional_boxes_needed : ℕ := 9

-- Theorem statement
theorem bakery_made_muffins :
  (available_boxes * muffins_per_box) + (additional_boxes_needed * muffins_per_box) = 95 := 
by
  sorry

end bakery_made_muffins_l776_776715


namespace correct_operation_l776_776797

noncomputable def valid_operation (n : ℕ) (a b : ℕ) (c d : ℤ) (x : ℚ) : Prop :=
  match n with
  | 0 => (x ^ a / x ^ b = x ^ (a - b))
  | 1 => (x ^ a * x ^ b = x ^ (a + b))
  | 2 => (c * x ^ a + d * x ^ a = (c + d) * x ^ a)
  | 3 => ((c * x ^ a) ^ b = c ^ b * x ^ (a * b))
  | _ => False

theorem correct_operation (x : ℚ) : valid_operation 1 2 3 0 0 x :=
by sorry

end correct_operation_l776_776797


namespace fixed_numbers_in_diagram_has_six_solutions_l776_776246

-- Define the problem setup and constraints
def is_divisor (m n : ℕ) : Prop := ∃ k, n = k * m

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Formulating the main proof statement
theorem fixed_numbers_in_diagram_has_six_solutions : 
  ∃ (a b c k : ℕ),
    (14 * 4 * a = 14 * 6 * c) ∧
    (4 * a = 6 * c) ∧
    (2 * a = 3 * c) ∧
    (∃ k, c = 2 * k ∧ a = 3 * k) ∧
    (14 * 4 * 3 * k = 3 * k * b * 2 * k) ∧
    (∃ k, 56 * k = 6 * k^2 * b) ∧
    (b = 28 / k) ∧
    ((is_divisor k 28) ∧
     (k = 1 ∨ k = 2 ∨ k = 4 ∨ k = 7 ∨ k = 14 ∨ k = 28)) ∧
    (6 = 6) := sorry

end fixed_numbers_in_diagram_has_six_solutions_l776_776246


namespace solutions_to_equation_l776_776701

variable (x : ℝ)

def original_eq : Prop :=
  (3 * x - 9) / (x^2 - 6 * x + 8) = (x + 1) / (x - 2)

theorem solutions_to_equation : (original_eq 1 ∧ original_eq 5) :=
by
  sorry

end solutions_to_equation_l776_776701


namespace right_triangle_acute_angle_le_45_l776_776353

theorem right_triangle_acute_angle_le_45
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hright : a^2 + b^2 = c^2):
  ∃ θ φ : ℝ, θ + φ = 90 ∧ (θ ≤ 45 ∨ φ ≤ 45) :=
by
  sorry

end right_triangle_acute_angle_le_45_l776_776353


namespace ellipse_properties_l776_776104

theorem ellipse_properties :
  (∃ a e : ℝ, (∃ b c : ℝ, a^2 = 25 ∧ b^2 = 9 ∧ c^2 = a^2 - b^2 ∧ c = 4 ∧ e = c / a) ∧ a = 5 ∧ e = 4 / 5) :=
sorry

end ellipse_properties_l776_776104


namespace coloring_shapes_l776_776886

theorem coloring_shapes (A B C : Type) : 
  ∃ (colors : Finset ℕ), 
  colors.card = 5 ∧ 
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
  ∃ (f : Fin (5) → A) (g : Fin (5) → B) (h : Fin (5) → C), 
  f ≠ g ∧ g ≠ h ∧ f ≠ h ∧ 
  (f = g ∨ g = h ∨ f = h) → 
  (5 * 4 * 3 = 60) :=
begin
  sorry
end

end coloring_shapes_l776_776886


namespace vector_problem_l776_776974

variable (a b : ℝ × ℝ)
variable (m n : ℝ)

def vec_eq : Prop :=
  a = (2, 1) ∧ 
  b = (1, -2) ∧ 
  m * a + n * b = (9, -8)

theorem vector_problem :
  vec_eq a b m n → m - n = -3 :=
by
  sorry

end vector_problem_l776_776974


namespace triangle_area_l776_776226

section TriangleAreaProof

variables (a b c : ℝ) (A B C : ℝ)
  (cos_AB : ℝ)

theorem triangle_area (ha : a = 5) (hb : b = 4) (hcos : cos_AB = 31 / 32) :
  let S := 1 / 2 * a * b * (3 * sqrt 7 / 8) in
  S = 15 * sqrt 7 / 4 :=
by
  sorry

end TriangleAreaProof

end triangle_area_l776_776226


namespace perpendicular_line_to_plane_l776_776220

variables (u n : ℝ × ℝ × ℝ) 

def lin_perp_plane (u n : ℝ × ℝ × ℝ) : Prop :=
  u.1 * n.1 + u.2 * n.2 + u.3 * n.3 = 0

theorem perpendicular_line_to_plane :
  lin_perp_plane (1, 1, 2) (-3, 3, -6) :=
by
  -- Proof omitted
  sorry

end perpendicular_line_to_plane_l776_776220


namespace arithmetic_mean_of_primes_l776_776946

variable (list : List ℕ) 
variable (primes : List ℕ)
variable (h1 : list = [24, 25, 29, 31, 33])
variable (h2 : primes = [29, 31])

theorem arithmetic_mean_of_primes : (primes.sum / primes.length : ℝ) = 30 := by
  sorry

end arithmetic_mean_of_primes_l776_776946


namespace a_seq_formula_S_seq_formula_l776_776289

variable (n : ℕ)

def a_seq : ℕ → ℕ
| 1     => 1
| (n+1) => a_seq n + 3 * 2^(n-1)

def b_seq (n : ℕ) := n * (3 * 2^(n-1) - 2)

def S_seq : ℕ → ℕ
| 1     => b_seq 1
| (n+1) => S_seq n + b_seq (n+1)

/-- The general formula for the sequence {a_n} --/
theorem a_seq_formula (n : ℕ) : a_seq n = 3 * 2^(n-1) - 2 := 
sorry

/-- The sum of the first n terms of the sequence {b_n} --/
theorem S_seq_formula (n : ℕ) : S_seq n = 3 * (n-1) * 2^n - n*(n+1) + 3 :=
sorry

end a_seq_formula_S_seq_formula_l776_776289


namespace average_score_78_l776_776233

theorem average_score_78 (T : ℕ) (M F : ℕ) 
  (hM : M = 0.4 * T) 
  (hF : F = 0.6 * T) 
  (avgMaleScore : ℕ) 
  (avgFemaleScore : ℕ)
  (h_avgMale : avgMaleScore = 75)
  (h_avgFemale : avgFemaleScore = 80) :
  (75 * M + 80 * F) / T = 78 := 
by sorry

end average_score_78_l776_776233


namespace area_shaded_region_l776_776358

-- Define the problem conditions
def Point := (ℝ × ℝ)
def O : Point := (0, 0)
def A : Point := (6, 0)
def B : Point := (24, 0)
def C : Point := (24, 18)
def D : Point := (6, 18)
def E : Point := (6, 6)

-- Define computation for part lengths
def length (p1 p2 : Point) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Compute lengths EA, DE, and DC
def EA : ℝ := length E A
def DE : ℝ := length D E
def DC : ℝ := length D C

-- Define area computation for triangle CDE
def area_CDE : ℝ := (1/2) * DE * DC

-- Prove the area is 122 cm² rounded to the nearest whole number
theorem area_shaded_region : real.round area_CDE = 122 :=
by
  sorry

end area_shaded_region_l776_776358


namespace find_wholesale_prices_l776_776840

-- Definitions based on conditions
def bakerPlusPrice (p : ℕ) (d : ℕ) : ℕ := p + d
def starPrice (p : ℕ) (k : ℚ) : ℕ := (p * k).toNat

-- The tuple of prices given
def prices : List ℕ := [64, 64, 70, 72]

-- Statement of the problem
theorem find_wholesale_prices (d : ℕ) (k : ℚ) (h_d : d > 0) (h_k : k > 1 ∧ k.denom = 1):
  ∃ p1 p2, 
  (p1 ∈ prices ∧ bakerPlusPrice p1 d ∈ prices ∧ starPrice p1 k ∈ prices) ∧
  (p2 ∈ prices ∧ bakerPlusPrice p2 d ∈ prices ∧ starPrice p2 k ∈ prices) :=
sorry

end find_wholesale_prices_l776_776840


namespace combined_sale_price_correct_l776_776115

-- Define constants for purchase costs of items A, B, and C.
def purchase_cost_A : ℝ := 650
def purchase_cost_B : ℝ := 350
def purchase_cost_C : ℝ := 400

-- Define profit percentages for items A, B, and C.
def profit_percentage_A : ℝ := 0.40
def profit_percentage_B : ℝ := 0.25
def profit_percentage_C : ℝ := 0.30

-- Define the desired sale prices for items A, B, and C based on profit margins.
def sale_price_A : ℝ := purchase_cost_A * (1 + profit_percentage_A)
def sale_price_B : ℝ := purchase_cost_B * (1 + profit_percentage_B)
def sale_price_C : ℝ := purchase_cost_C * (1 + profit_percentage_C)

-- Calculate the combined sale price for all three items.
def combined_sale_price : ℝ := sale_price_A + sale_price_B + sale_price_C

-- The theorem stating that the combined sale price for all three items is $1867.50.
theorem combined_sale_price_correct :
  combined_sale_price = 1867.50 := 
sorry

end combined_sale_price_correct_l776_776115


namespace no_valid_rearrangement_l776_776716

theorem no_valid_rearrangement (board_size : ℕ)
  (cells_numbered : ∀ r c, 1 ≤ r ∧ r ≤ board_size → 1 ≤ c ∧ c ≤ board_size → ℕ)
  (initial_placement : fin board_size → fin board_size) :
    board_size = 8 →
    (∀ r : fin board_size, ∃ c : fin board_size, initial_placement r = c) →
    (∀ c : fin board_size, ∃ r : fin board_size, initial_placement r = c) →
    (∀ r : fin board_size, ∀ c : fin board_size, cells_numbered r c < cells_numbered (initial_placement r) c) → False :=
by
  sorry

end no_valid_rearrangement_l776_776716


namespace carpet_dimensions_l776_776426

-- Define the problem parameters
def width_a : ℕ := 50
def width_b : ℕ := 38

-- The dimensions x and y are integral numbers of feet
variables (x y : ℕ)

-- The same length L for both rooms that touches all four walls
noncomputable def length (x y : ℕ) : ℚ := (22 * (x^2 + y^2)) / (x * y)

-- The final theorem to be proven
theorem carpet_dimensions (x y : ℕ) (h : (x^2 + y^2) * 1056 = (x * y) * 48 * (length x y)) : (x = 50) ∧ (y = 25) :=
by
  sorry -- Proof is omitted

end carpet_dimensions_l776_776426


namespace function_monotonic_decreasing_range_l776_776221

theorem function_monotonic_decreasing_range (a : ℝ) :
  (∀ x ∈ Ioo (1/2 : ℝ) 3, (x^2 - a * x + 1) ≤ 0) ↔ (a ≥ 10/3) := 
sorry

end function_monotonic_decreasing_range_l776_776221


namespace sec_7pi_over_4_l776_776526

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_7pi_over_4 :
  sec (7 * Real.pi / 4) = Real.sqrt 2 := by
  have h1 : Real.cos (7 * Real.pi / 4) = Real.cos (45 * Real.pi / 180) := by sorry
  have h2 : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  rw [sec, h1, h2]
  field_simp
  norm_num

end sec_7pi_over_4_l776_776526


namespace pie_wholesale_price_l776_776818

theorem pie_wholesale_price (p1 p2: ℕ) (d: ℕ) (k: ℕ) (h1: d > 0) (h2: k > 1)
  (price_list: List ℕ)
  (p_plus_1: price_list = [p1 + d, k * p1, p2 + d, k * p2]) 
  (h3: price_list.perm [64, 64, 70, 72]) : 
  p1 = 60 ∧ p2 = 60 := 
sorry

end pie_wholesale_price_l776_776818


namespace find_length_of_BC_l776_776249

theorem find_length_of_BC 
  (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (right_angle_at_B : ∃ p : A × B × C, right_angle (p.1, p.2, p.3)) 
  (tan_A : ∀ p, tan (angle (p.1, p.2)) = 4 / 3)
  (AB_eq_3 : ∀ p, dist p.1 p.2 = 3) :
  ∃ (BC_length : ℝ), BC_length = 4 :=
sorry

end find_length_of_BC_l776_776249


namespace oreo_shop_purchases_l776_776910

theorem oreo_shop_purchases :
  let oreos := 7 in
  let milks := 4 in
  let total_flavors := oreos + milks in
  ∃ (total_ways : ℕ), 
    total_ways = 
      (nat.choose total_flavors 4)
      + (nat.choose total_flavors 3 * oreos)
      + (nat.choose total_flavors 2 * (nat.choose oreos 2 + oreos))
      + (nat.choose total_flavors 1 * (nat.choose oreos 3 + (oreos * (oreos - 1) * 3) / 2 + oreos))
      + (nat.choose oreos 4 + nat.choose oreos 2 + (oreos * (oreos - 1)) + oreos)
  ∧ total_ways = 4978 :=
by
  sorry

end oreo_shop_purchases_l776_776910


namespace shaded_area_correct_l776_776062

-- Conditions
def side_length_square := 40
def triangle1_base := 15
def triangle1_height := 15
def triangle2_base := 15
def triangle2_height := 15

-- Calculation
def square_area := side_length_square * side_length_square
def triangle1_area := 1 / 2 * triangle1_base * triangle1_height
def triangle2_area := 1 / 2 * triangle2_base * triangle2_height
def total_triangle_area := triangle1_area + triangle2_area
def shaded_region_area := square_area - total_triangle_area

-- Theorem to prove
theorem shaded_area_correct : shaded_region_area = 1375 := by
  sorry

end shaded_area_correct_l776_776062


namespace largest_valid_six_digit_number_l776_776953

-- Defining the conditions
def is_six_digit (n : Nat) : Prop :=
  100000 ≤ n ∧ n < 1000000

def all_digits_distinct (n : Nat) : Prop :=
  let digits := [n / 100000 % 10, n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  digits.nodup

def sum_or_difference_neighbors (d1 d2 d3 : Nat) : Prop :=
  d2 = d1 + d3 ∨ d2 = abs (d1 - d3)

def valid_six_digit_number (num : Nat) : Prop :=
  is_six_digit num ∧
  all_digits_distinct num ∧
  (let d := [num / 100000 % 10, num / 10000 % 10, num / 1000 % 10, num / 100 % 10, num / 10 % 10, num % 10] in
    sum_or_difference_neighbors d[0] d[1] d[2] ∧ 
    sum_or_difference_neighbors d[1] d[2] d[3] ∧ 
    sum_or_difference_neighbors d[2] d[3] d[4] ∧ 
    sum_or_difference_neighbors d[3] d[4] d[5])

-- Statement to prove
theorem largest_valid_six_digit_number : ∀ n, valid_six_digit_number n → n ≤ 972538 :=
by {
  sorry
}

end largest_valid_six_digit_number_l776_776953


namespace binom_10_3_l776_776465

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l776_776465


namespace area_of_triangle_with_sides_13_12_5_l776_776061

theorem area_of_triangle_with_sides_13_12_5 :
  let a := 13
  let b := 12
  let c := 5
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area = 30 :=
by
  let a := 13
  let b := 12
  let c := 5
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  sorry

end area_of_triangle_with_sides_13_12_5_l776_776061


namespace pipe_filling_time_l776_776308

theorem pipe_filling_time 
  (rate_A : ℚ := 1/8) 
  (rate_L : ℚ := 1/24) :
  (1 / (rate_A - rate_L) = 12) :=
by
  sorry

end pipe_filling_time_l776_776308


namespace probability_sum_multiple_of_3_l776_776351

def num_faces : ℕ := 6
def roll_outcomes : finset (ℕ × ℕ) := finset.product (finset.range num_faces) (finset.range num_faces)
def favorable_outcomes : finset (ℕ × ℕ) :=
  roll_outcomes.filter (λ p, (p.1 + p.2) % 3 = 0)

theorem probability_sum_multiple_of_3 :
  (favorable_outcomes.card : ℚ) / (roll_outcomes.card : ℚ) = 1 / 3 :=
by {
  sorry
}

end probability_sum_multiple_of_3_l776_776351


namespace bakery_wholesale_price_exists_l776_776858

theorem bakery_wholesale_price_exists :
  ∃ (p1 p2 d k : ℕ), (d > 0) ∧ (k > 1) ∧
  ({p1 + d, p2 + d, k * p1, k * p2} = {64, 64, 70, 72}) :=
by sorry

end bakery_wholesale_price_exists_l776_776858


namespace determine_wholesale_prices_l776_776831

-- Definitions and Conditions
-- Wholesale prices for days 1 and 2 are defined as p1 and p2
def wholesale_price_day1 : ℝ := p1
def wholesale_price_day2 : ℝ := p2

-- The price increment at "Baker Plus" store is a constant d
def price_increment_baker_plus : ℝ := d
-- The factor at "Star" store is a constant k > 1
def price_factor_star : ℝ := k
-- Given prices for two days: 64, 64, 70, 72 rubles
def given_prices : set ℝ := {64, 64, 70, 72}

-- Prices at the stores
def baker_plus_day1_price := wholesale_price_day1 + price_increment_baker_plus
def star_day1_price := wholesale_price_day1 * price_factor_star

def baker_plus_day2_price := wholesale_price_day2 + price_increment_baker_plus
def star_day2_price := wholesale_price_day2 * price_factor_star

-- Theorem to determine the wholesale prices for each day (p1 and p2)
theorem determine_wholesale_prices
    (baker_plus_day1_price ∈ given_prices)
    (star_day1_price ∈ given_prices)
    (baker_plus_day2_price ∈ given_prices)
    (star_day2_price ∈ given_prices)
    (h1 : baker_plus_day1_price ≠ star_day1_price)
    (h2 : baker_plus_day2_price ≠ star_day2_price) :
  (∃ p1 p2 d k > 1, 
      (p1 + d ∈ given_prices) ∧ (p1 * k ∈ given_prices) ∧
      (p2 + d ∈ given_prices) ∧ (p2 * k ∈ given_prices) ∧
      (p1 ≠ p2)) :=
  sorry

end determine_wholesale_prices_l776_776831


namespace unique_prime_solution_l776_776208

theorem unique_prime_solution :
  ∃! (p : ℕ), Prime p ∧ (∃ (k : ℤ), 2 * (p ^ 4) - 7 * (p ^ 2) + 1 = k ^ 2) := 
sorry

end unique_prime_solution_l776_776208


namespace company_l776_776694

-- Define conditions
def initial_outlay : ℝ := 10000

def material_cost_per_set_first_300 : ℝ := 20
def material_cost_per_set_beyond_300 : ℝ := 15

def exchange_rate : ℝ := 1.1

def import_tax_rate : ℝ := 0.10

def sales_price_per_set_first_400 : ℝ := 50
def sales_price_per_set_beyond_400 : ℝ := 45

def export_tax_threshold : ℕ := 500
def export_tax_rate : ℝ := 0.05

def production_and_sales : ℕ := 800

-- Helper functions for the problem
def material_cost_first_300_sets : ℝ :=
  300 * material_cost_per_set_first_300 * exchange_rate

def material_cost_next_500_sets : ℝ :=
  (production_and_sales - 300) * material_cost_per_set_beyond_300 * exchange_rate

def total_material_cost : ℝ :=
  material_cost_first_300_sets + material_cost_next_500_sets

def import_tax : ℝ := total_material_cost * import_tax_rate

def total_manufacturing_cost : ℝ :=
  initial_outlay + total_material_cost + import_tax

def sales_revenue_first_400_sets : ℝ :=
  400 * sales_price_per_set_first_400

def sales_revenue_next_400_sets : ℝ :=
  (production_and_sales - 400) * sales_price_per_set_beyond_400

def total_sales_revenue_before_export_tax : ℝ :=
  sales_revenue_first_400_sets + sales_revenue_next_400_sets

def sales_revenue_beyond_threshold : ℝ :=
  (production_and_sales - export_tax_threshold) * sales_price_per_set_beyond_400

def export_tax : ℝ := sales_revenue_beyond_threshold * export_tax_rate

def total_sales_revenue_after_export_tax : ℝ :=
  total_sales_revenue_before_export_tax - export_tax

def profit : ℝ :=
  total_sales_revenue_after_export_tax - total_manufacturing_cost

-- Lean 4 statement for the proof problem
theorem company's_profit_is_10990 :
  profit = 10990 := by
  sorry

end company_l776_776694


namespace lottery_winning_situations_l776_776235

theorem lottery_winning_situations :
  let num_tickets := 8
  let first_prize := 1
  let second_prize := 1
  let third_prize := 1
  let non_winning := 5
  let customers := 4
  let tickets_per_customer := 2
  let total_ways := 24 + 36
  total_ways = 60 :=
by
  let num_tickets := 8
  let first_prize := 1
  let second_prize := 1
  let third_prize := 1
  let non_winning := 5
  let customers := 4
  let tickets_per_customer := 2
  let total_ways := 24 + 36

  -- Skipping proof steps
  sorry

end lottery_winning_situations_l776_776235


namespace find_a_l776_776570

noncomputable def compute_a (a b : ℚ) : ℚ :=
a

theorem find_a 
  (h1 : Root (-2 - 3*Real.sqrt 3) (λ x, x^3 + a*x^2 + b*x + 49)) 
  (h2 : Root (-2 + 3*Real.sqrt 3) (λ x, x^3 + a*x^2 + b*x + 49)) 
  (h3 : a = - (∑ (r : ℚ) in {-2 - 3 * Real.sqrt 3, -2 + 3 * Real.sqrt 3, 49 / 23} , r)) : 
  compute_a a b = -3/23 :=
by {
  sorry
}

end find_a_l776_776570


namespace main_theorem_l776_776723

variable (a : ℝ)
variable (A B C D : ℝ × ℝ)
variable (O : ℝ × ℝ)
variable (line_eqn : ℝ → ℝ → ℝ)

-- Conditions
def line : Prop := ∀ x y, line_eqn x y = a * x + (1 / a) * y - 1
def circle : Prop := ∀ (x y : ℝ), (x - O.1) ^ 2 + (y - O.2) ^ 2 = 1
def intersection_a : Prop := ∃ x, line_eqn x 0 = 0 ∧ A = (x, 0)
def intersection_b : Prop := ∃ y, line_eqn 0 y = 0 ∧ B = (0, y)
def a_ge_one : Prop := a ≥ 1

-- Conclusion 1
def conclusion_1 : Prop := S_triangle A B (0, 0) = 1 / 2

-- Conclusion 2
def conclusion_2 : Prop := ∃ a, a ≥ 1 ∧ |A - B| < |C - D|

-- Conclusion 3
def conclusion_3 : Prop := ∃ a, a ≥ 1 ∧ S_triangle C D O < 1 / 2

-- Main Proposition
theorem main_theorem : (a_ge_one → intersection_a → intersection_b → conclusion_1 ∧ ¬ conclusion_2 ∧ conclusion_3) := sorry

end main_theorem_l776_776723


namespace combination_10_3_l776_776447

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l776_776447


namespace direct_proportion_inequality_l776_776150

theorem direct_proportion_inequality (k x1 x2 y1 y2 : ℝ) (h_k : k < 0) (h_y1 : y1 = k * x1) (h_y2 : y2 = k * x2) (h_x : x1 < x2) : y1 > y2 :=
by
  -- The proof will be written here, currently leaving it as sorry
  sorry

end direct_proportion_inequality_l776_776150


namespace geometric_series_has_value_a_l776_776130

theorem geometric_series_has_value_a (a : ℝ) (S : ℕ → ℝ)
  (h : ∀ n, S (n + 1) = a * (1 / 4) ^ n + 6) :
  a = -3 / 2 :=
sorry

end geometric_series_has_value_a_l776_776130


namespace max_intersection_points_l776_776116

theorem max_intersection_points (n : ℕ) (h : n = 4) (max_points_per_circle : ℕ) (h_max : max_points_per_circle = 2) : 
    (∃ max_points : ℕ, max_points = n * max_points_per_circle) → max_points = 8 :=
by
  intro h_exists
  cases h_exists with max_points h_max_points
  rw [h, h_max, mul_comm] at h_max_points
  exact h_max_points.symm

end max_intersection_points_l776_776116


namespace smallest_possible_n_l776_776935

theorem smallest_possible_n (n : ℕ) :
  ∃ n, 17 * n - 3 ≡ 0 [MOD 11] ∧ n = 6 :=
by
  sorry

end smallest_possible_n_l776_776935


namespace sides_and_expression_l776_776554

-- a = 4 and b = 6 are given conditions.
def a := 4
def b := 6

-- Define perimeter and c based on the given conditions.
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- c must be within the range between 2 and 8, since 2 < c < 8
def c_condition (c : ℕ) : Prop := 2 < c ∧ c < 8

-- c must be even for the perimeter to be even.
def is_even (n : ℕ) : Prop := n % 2 = 0

axiom c_values : (c : ℕ) → (perimeter a b c < 18 ∧ is_even (perimeter a b c)) → (c = 4 ∨ c = 6)

-- Simplifying |a + b - c| + |c - a - b| results in 2a + 2b - 2c
axiom simplify_expression : ∀ (c : ℕ), 2 * |a + b - c| = 2 * (a + b - c)

-- The main theorem to prove
theorem sides_and_expression (c : ℕ) :
  (perimeter a b c < 18 ∧ is_even (perimeter a b c)) →
  (c = 4 ∨ c = 6) ∧ (|a + b - c| + |c - a - b| = 2a + 2b - 2c) :=
by
  intro h
  have h_c : c = 4 ∨ c = 6 := c_values c h
  split
  { exact h_c }
  { cases h_c
    { simp [h_c] }
    { simp [h_c] } }

end sides_and_expression_l776_776554


namespace converse_of_corresponding_angles_postulate_l776_776379

theorem converse_of_corresponding_angles_postulate :
  (∀ (l1 l2 : Line) (t : Transversal), parallel l1 l2 → corresponding_angles_equal l1 l2 t) →
  (∀ (l1 l2 : Line) (t : Transversal), corresponding_angles_equal l1 l2 t → parallel l1 l2) :=
sorry

end converse_of_corresponding_angles_postulate_l776_776379


namespace polynomial_remainder_l776_776543

theorem polynomial_remainder (x : ℝ) :
  let dividend := x^4 + x^2 + 1
  let divisor := x^2 - 4 * x + 7
  let remainder := 12 * x - 69
  (∃ q : ℝ → ℝ, dividend = divisor * q + remainder) :=
sorry

end polynomial_remainder_l776_776543


namespace wholesale_price_is_60_l776_776869

variables (p d k : ℕ)
variables (P1 P2 Z1 Z2 : ℕ)

// Conditions from (a)
axiom H1 : P1 = p + d // Baker Plus price day 1
axiom H2 : P2 = p + d // Baker Plus price day 2
axiom H3 : Z1 = k * p // Star price day 1
axiom H4 : Z2 = k * p // Star price day 2
axiom H5 : k > 1 // Star store markup factor greater than 1
axiom H6 : P1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H7 : P2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H8 : Z1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H9 : Z2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))

theorem wholesale_price_is_60 : p = 60 := 
by {
    // Proof goes here, skipped with sorry
    sorry
}

end wholesale_price_is_60_l776_776869


namespace exists_c_in_interval_l776_776534

noncomputable def f : ℝ → ℝ := λ x, x^2 + 6 * x + 1

theorem exists_c_in_interval :
  ∃ c ∈ set.Ioo (-1 : ℝ) (3 : ℝ), deriv f c = 8 :=
begin
  have h1 : continuous_on f (set.Icc (-1) 3), from 
    (differentiable_on_polynomial (λ (x : ℝ), f x)).continuous_on,

  have h2 : ∀ x ∈ set.Icc (-1 : ℝ) 3, differentiable_at ℝ f x, from
    differentiable_on_polynomial (λ (x : ℝ), f x).differentiable_at,

  have h3 : f 3 - f (-1) = 32, by norm_num [f, (*), (^)],

  have h4 : (3 : ℝ) - (-1) = 4, by norm_num,

  let k := (f 3 - f (-1)) / (3 - (-1)),

  have h5 : k = 8, by norm_num [h3, h4],

  exact exists_deriv_eq_slope (set.mem_Icc.2 ⟨by norm_num, by norm_num⟩) h1 h2 h5,
end

end exists_c_in_interval_l776_776534


namespace general_form_a_n_l776_776595

-- Define the sequence {a_n} and its initial values
def a : ℕ → ℝ
| 0       := 0  -- placeholder since natural numbers in Lean start at 0
| 1       := 1
| 2       := 2
| (n + 3) := a (n + 2) + 2   -- this derives from a_{n+2} - 2a_{n+1} = 2

-- Sum of the sequence up to the nth term
def S : ℕ → ℝ
| 0       := 0
| n + 1   := S n + a (n + 1)

-- Statement to prove the general form of a_n
theorem general_form_a_n : ∀ n : ℕ, 
  a n = if n = 1 then 1 else if n = 2 then 2 else 2^n - 2 :=
sorry

end general_form_a_n_l776_776595


namespace geometric_seq_increasing_condition_l776_776663

theorem geometric_seq_increasing_condition (q : ℝ) (a : ℕ → ℝ): 
  (∀ n : ℕ, a (n + 1) = q * a n) → (¬ (∀ a : ℕ → ℝ, (∀ n : ℕ, a (n + 1) = q * a n) → ∀ n m : ℕ, n < m → a n < a m) ∧ ¬ (¬ (∀ a : ℕ → ℝ, (∀ n : ℕ, a (n + 1) = q * a n) → ∀ n m : ℕ, n < m → a n < a m))) :=
sorry

end geometric_seq_increasing_condition_l776_776663


namespace maximize_x_plus_y_l776_776960

noncomputable def max_x_plus_y (x y : ℝ) : ℝ :=
  x + y

theorem maximize_x_plus_y :
  ∀ (x y : ℝ), 
  (log (
    (x^2 + y^2) / 2) y ≥ 1 ∧ (x ≠ 0 ∨ y ≠ 0) ∧ (x^2 + y^2 ≠ 2)) →
  max_x_plus_y x y ≤ 1 + Real.sqrt 2 :=
begin
  sorry
end

end maximize_x_plus_y_l776_776960


namespace cat_greatest_distance_l776_776682

theorem cat_greatest_distance (P : (ℝ × ℝ)) (T : (ℝ × ℝ)) (r : ℝ) (A : (ℝ × ℝ))
  (hT : T = (5, 2)) (hr : r = 15) (hA : A = (1, 1)) :
  let distance := (λ (p₁ p₂ : ℝ × ℝ), Real.sqrt (((p₁.1 - p₂.1) ^ 2) + ((p₁.2 - p₂.2) ^ 2)))
  let greatest_distance := distance A T + r
  in greatest_distance = 19.123 :=
by
  sorry

end cat_greatest_distance_l776_776682


namespace least_three_digit_multiple_of_13_l776_776778

theorem least_three_digit_multiple_of_13 : ∃ n : ℕ, n ≥ 100 ∧ n % 13 = 0 ∧ ∀ m : ℕ, m ≥ 100 → m % 13 = 0 → n ≤ m :=
begin
  use 104,
  split,
  { exact nat.le_of_eq rfl },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero rfl) },
  { intros m hm hmod,
    have h8 : 8 * 13 = 104 := rfl,
    rw ←h8,
    exact nat.le_mul_of_pos_left (by norm_num) },
end

end least_three_digit_multiple_of_13_l776_776778


namespace problem_solution_l776_776212

theorem problem_solution (x y z : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 20) (h3 : 0.6 * y = z) : 
  z = 60 := by
  sorry

end problem_solution_l776_776212


namespace binomial_coefficient_10_3_l776_776490

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l776_776490


namespace fill_box_with_gamma_bricks_l776_776917

theorem fill_box_with_gamma_bricks
  (m n k : ℕ)
  (h_m : 1 < m)
  (h_n : 1 < n)
  (h_k : 1 < k)
  (unrestricted_supply : true)
  : ∃ (fill_function : ℕ × ℕ × ℕ → Prop),
      fill_function (m, n, k) → (∀ b, b = Γ → fill_function (3, 1, 1) ∧ fill_function (1, 3, 1) ∧ fill_function (1, 1, 3))
  :=
begin
  sorry
end

end fill_box_with_gamma_bricks_l776_776917


namespace no_twelve_right_triangles_with_ten_congruent_and_two_distinct_l776_776902

theorem no_twelve_right_triangles_with_ten_congruent_and_two_distinct 
  : ¬ ∃ (triangles : fin 12 → triangle) (square : polygon), 
      divides_into_triangles square triangles ∧ are_congruent (finset.filter (λ i, i < 10) triangles) ∧
      (∀ i j, i < 10 → j < 10 → i ≠ j → triangles i ≠ triangles j) ∧
      (∀ k, k = 10 ∨ k = 11 → triangles k ≠ triangles 0 ∧ triangles k ≠ triangles 1) ∧
      triangles 10 ≠ triangles 11 :=
begin
  -- Proof goes here.
  sorry
end

end no_twelve_right_triangles_with_ten_congruent_and_two_distinct_l776_776902


namespace max_value_of_a_l776_776584

theorem max_value_of_a (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : 0 < 2 * (a * 5 * real.sqrt (5 - 1) + b * 5 + c) 
       = 3 * (a * 10 * real.sqrt (10 - 1) + b * 10 + c)
       = 4 * (a * 17 * real.sqrt (17 - 1) + b * 17 + c) 
       ≤ 1) :
  a ≤ 3 / 200 :=
sorry

end max_value_of_a_l776_776584


namespace mul_example_l776_776092

theorem mul_example : (3.6 * 0.5 = 1.8) := by
  sorry

end mul_example_l776_776092


namespace determine_wholesale_prices_l776_776852

variables (p1 p2 d k : ℝ)
variables (h1 : 1 < k)
variables (prices : Finset ℝ)
variables (p_plus : ℝ → ℝ) (star : ℝ → ℝ)

noncomputable def BakeryPrices : Prop :=
  p_plus p1 = p1 + d ∧ p_plus p2 = p2 + d ∧ 
  star p1 = k * p1 ∧ star p2 = k * p2 ∧ 
  prices = {64, 64, 70, 72} ∧ 
  ∃ (q1 q2 q3 q4 : ℝ), 
    (q1 = p_plus p1 ∨ q1 = star p1 ∨ q1 = p_plus p2 ∨ q1 = star p2) ∧
    (q2 = p_plus p1 ∨ q2 = star p1 ∨ q2 = p_plus p2 ∨ q2 = star p2) ∧
    (q3 = p_plus p1 ∨ q3 = star p1 ∨ q3 = p_plus p2 ∨ q3 = star p2) ∧
    (q4 = p_plus p1 ∨ q4 = star p1 ∨ q4 = p_plus p2 ∨ q4 = star p2) ∧
    prices = {q1, q2, q3, q4}

theorem determine_wholesale_prices (p1 p2 : ℝ) (d k : ℝ)
    (h1 : 1 < k) (prices : Finset ℝ) :
  BakeryPrices p1 p2 d k h1 prices (λ p, p + d) (λ p, k * p) :=
sorry

end determine_wholesale_prices_l776_776852


namespace spring_bud_cup_eq_289_l776_776703

theorem spring_bud_cup_eq_289 (x : ℕ) (h : x + x = 578) : x = 289 :=
sorry

end spring_bud_cup_eq_289_l776_776703


namespace base_5_digits_1250_l776_776180

theorem base_5_digits_1250 (n : ℕ) (hn : n = 1250) : 
  ∃ d, d = 5 ∧ ∀ x, base_repr x 5 = d :=
by sorry

end base_5_digits_1250_l776_776180


namespace trig_identity_l776_776579

theorem trig_identity (α : ℝ) (h : sin α = -2 * cos α) : sin α ^ 2 + cos (2 * α + π / 2) = 8 / 5 :=
sorry

end trig_identity_l776_776579


namespace largest_affordable_apartment_size_l776_776912

-- Definition of conditions in the problem
def cost_per_sq_ft : ℝ := 0.85
def initial_budget : ℝ := 940
def initial_deposit : ℝ := 340

-- Remaining budget for rent after initial deposit
def remaining_budget : ℝ := initial_budget - initial_deposit

-- Definition of the size of the apartment in square feet that Jillian can afford
def max_affordable_size (c : ℝ) (r : ℝ) : ℝ := r / c

-- Lean statement to prove the largest apartment size Jillian can afford is 705 square feet
theorem largest_affordable_apartment_size : max_affordable_size cost_per_sq_ft remaining_budget = 705 :=
by
  unfold max_affordable_size
  unfold remaining_budget
  unfold cost_per_sq_ft
  sorry

end largest_affordable_apartment_size_l776_776912


namespace max_value_x_y_l776_776957

theorem max_value_x_y (x y : ℝ) 
  (h1 : log ((x^2 + y^2) / 2) y ≥ 1) 
  (h2 : (x, y) ≠ (0, 0)) 
  (h3 : x^2 + y^2 ≠ 2) : 
  x + y ≤ 1 + real.sqrt 2 := 
sorry

end max_value_x_y_l776_776957


namespace least_positive_three_digit_multiple_of_13_is_104_l776_776784

theorem least_positive_three_digit_multiple_of_13_is_104 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ n = 104 :=
by
  existsi 104
  split
  · show 100 ≤ 104
    exact le_refl 104
  split
  · show 104 < 1000
    exact dec_trivial
  split
  · show 104 % 13 = 0
    exact dec_trivial
  · show 104 = 104
    exact rfl

end least_positive_three_digit_multiple_of_13_is_104_l776_776784


namespace city_parking_fee_l776_776918

theorem city_parking_fee (weekly_salary : ℕ) (federal_tax_rate : ℚ) (state_tax_rate : ℚ) 
    (health_insurance : ℕ) (life_insurance : ℕ) (final_paycheck : ℕ) (city_parking_fee : ℕ) :
    weekly_salary = 450 ∧ federal_tax_rate = 1/3 ∧ state_tax_rate = 0.08 ∧
    health_insurance = 50 ∧ life_insurance = 20 ∧ final_paycheck = 184 →
    city_parking_fee = 22 :=
begin
  sorry
end

end city_parking_fee_l776_776918


namespace volume_of_prism_is_429_l776_776745

theorem volume_of_prism_is_429 (x y z : ℝ) (h1 : x * y = 56) (h2 : y * z = 57) (h3 : z * x = 58) : 
  x * y * z = 429 :=
by
  sorry

end volume_of_prism_is_429_l776_776745


namespace raven_current_age_l776_776231

variable (R P : ℕ) -- Raven's current age, Phoebe's current age
variable (h₁ : P = 10) -- Phoebe is currently 10 years old
variable (h₂ : R + 5 = 4 * (P + 5)) -- In 5 years, Raven will be 4 times as old as Phoebe

theorem raven_current_age : R = 55 := 
by
  -- h2: R + 5 = 4 * (P + 5)
  -- h1: P = 10
  sorry

end raven_current_age_l776_776231


namespace specified_time_eq_l776_776628

def distance : ℕ := 900
def ts (x : ℕ) : ℕ := x + 1
def tf (x : ℕ) : ℕ := x - 3

theorem specified_time_eq (x : ℕ) (h1 : x > 3) : 
  (distance / tf x) = 2 * (distance / ts x) :=
sorry

end specified_time_eq_l776_776628


namespace ratio_L1B_AN_length_L1N_volume_prism_KLMN_l776_776566

noncomputable def KLMN_prism_info :=
-- Define the condition that this is a regular prism and specify planes α and β
⦃
  base := KLMN : prism, 
  planes := planes α β,
  points_of_intersection := points A B,
  sphere_radius := 2
⦄

/-
First Statement: The ratio L₁B : AN = 2
-/
theorem ratio_L1B_AN (KLMN_prism_info : Type) :
  let ⦃K : point, L : point, M : point, N : point,
      K1 : point, L1 : point, M1 : point, N1 : point, 
      A : point, B : point⦄,
      plane_α through K,
      plane_β through N1,
      A, B ∈ intersection of α and β with L1N
  in L1B_ratio_AN := 2 :=
sorry

/-
Second Statement: The length of L₁N = 2(1 + sqrt(13))
-/
theorem length_L1N (KLMN_prism_info : Type) :
  let ⦃K : point, L : point, M : point, N : point,
      K1 : point, L1 : point, M1 : point, N1 : point,
      length_L1N := 2 * (1 + sqrt 13)⦄,
  in length (L1N) := length_L1N :=
sorry

/-
Third Statement: The volume of the prism V = 32√(6 + 2√13)
-/
theorem volume_prism_KLMN (KLMN_prism_info : Type) :
  let ⦃K : point, L : point, M : point, N : point,
      K1 : point, L1 : point, M1 : point, N1 : point⦄,
      base_length := 4,
      length_L1N := 2 * (1 + sqrt 13),
      volume := base_area * length_L1N
  in volume := 32 * sqrt (6 + 2 * sqrt 13) :=
sorry

end ratio_L1B_AN_length_L1N_volume_prism_KLMN_l776_776566


namespace Rosa_called_pages_last_week_l776_776642

theorem Rosa_called_pages_last_week : 
  ∀ (pages_called_this_week pages_called_in_all : ℝ), 
    pages_called_this_week = 8.6 → 
    pages_called_in_all = 18.8 → 
    (pages_called_in_all - pages_called_this_week) = 10.2 := 
by
  intros pages_called_this_week pages_called_in_all h1 h2
  have h : pages_called_in_all - pages_called_this_week = (18.8 - 8.6), from sorry
  rw [h1, h2] at h
  simp at h
  exact h

end Rosa_called_pages_last_week_l776_776642


namespace heather_oranges_left_l776_776604

theorem heather_oranges_left (initial : ℕ) (russell_takes : ℕ) (samantha_takes : ℕ) 
  (h_initial : initial = 60) (h_russell_takes : russell_takes = 35) (h_samantha_takes : samantha_takes = 12) :
  initial - russell_takes - samantha_takes = 13 := 
by 
  rw [h_initial, h_russell_takes, h_samantha_takes]
  sorry

end heather_oranges_left_l776_776604


namespace wholesale_price_is_60_l776_776870

variables (p d k : ℕ)
variables (P1 P2 Z1 Z2 : ℕ)

// Conditions from (a)
axiom H1 : P1 = p + d // Baker Plus price day 1
axiom H2 : P2 = p + d // Baker Plus price day 2
axiom H3 : Z1 = k * p // Star price day 1
axiom H4 : Z2 = k * p // Star price day 2
axiom H5 : k > 1 // Star store markup factor greater than 1
axiom H6 : P1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H7 : P2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H8 : Z1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H9 : Z2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))

theorem wholesale_price_is_60 : p = 60 := 
by {
    // Proof goes here, skipped with sorry
    sorry
}

end wholesale_price_is_60_l776_776870


namespace cos_of_sin_l776_776247

theorem cos_of_sin (B : ℝ) (hB : sin B = 3/5) (hA : ∠A = 90) : cos B = 4/5 :=
by
  sorry

end cos_of_sin_l776_776247


namespace determine_wholesale_prices_l776_776829

-- Definitions and Conditions
-- Wholesale prices for days 1 and 2 are defined as p1 and p2
def wholesale_price_day1 : ℝ := p1
def wholesale_price_day2 : ℝ := p2

-- The price increment at "Baker Plus" store is a constant d
def price_increment_baker_plus : ℝ := d
-- The factor at "Star" store is a constant k > 1
def price_factor_star : ℝ := k
-- Given prices for two days: 64, 64, 70, 72 rubles
def given_prices : set ℝ := {64, 64, 70, 72}

-- Prices at the stores
def baker_plus_day1_price := wholesale_price_day1 + price_increment_baker_plus
def star_day1_price := wholesale_price_day1 * price_factor_star

def baker_plus_day2_price := wholesale_price_day2 + price_increment_baker_plus
def star_day2_price := wholesale_price_day2 * price_factor_star

-- Theorem to determine the wholesale prices for each day (p1 and p2)
theorem determine_wholesale_prices
    (baker_plus_day1_price ∈ given_prices)
    (star_day1_price ∈ given_prices)
    (baker_plus_day2_price ∈ given_prices)
    (star_day2_price ∈ given_prices)
    (h1 : baker_plus_day1_price ≠ star_day1_price)
    (h2 : baker_plus_day2_price ≠ star_day2_price) :
  (∃ p1 p2 d k > 1, 
      (p1 + d ∈ given_prices) ∧ (p1 * k ∈ given_prices) ∧
      (p2 + d ∈ given_prices) ∧ (p2 * k ∈ given_prices) ∧
      (p1 ≠ p2)) :=
  sorry

end determine_wholesale_prices_l776_776829


namespace algebraic_expression_value_l776_776990

theorem algebraic_expression_value {a b c x : ℝ}
  (h1 : (2 - a)^2 + sqrt (a^2 + b + c) + abs (c + 8) = 0)
  (h2 : a * x^2 + b * x + c = 0) :
  3 * x^2 + 6 * x + 1 = 13 :=
sorry

end algebraic_expression_value_l776_776990


namespace solve_for_n_l776_776550

theorem solve_for_n (n : ℕ) (h : |Complex.mk 3 n| = 3 * Real.sqrt 10) : n = 9 := by
  sorry

end solve_for_n_l776_776550


namespace quadratic_expression_value_l776_776994

theorem quadratic_expression_value (a : ℝ)
  (h1 : ∃ x₁ x₂ : ℝ, x₁^2 + 2 * (a - 1) * x₁ + a^2 - 7 * a - 4 = 0 ∧ x₂^2 + 2 * (a - 1) * x₂ + a^2 - 7 * a - 4 = 0)
  (h2 : ∀ x₁ x₂ : ℝ, x₁ * x₂ - 3 * x₁ - 3 * x₂ - 2 = 0) :
  (1 + 4 / (a^2 - 4)) * (a + 2) / a = 2 := 
sorry

end quadratic_expression_value_l776_776994


namespace intervals_of_increase_and_decrease_l776_776103

noncomputable def f (x : ℝ) : ℝ := x^3 - 3/2 * x^2 - 6 * x + 4

theorem intervals_of_increase_and_decrease :
  (∀ x, x < -1 → has_deriv_at f (3 * (x^2 - x - 2)) x ∧ 3 * (x + 1) * (x - 2) > 0) ∧
  (∀ x, -1 < x ∧ x < 2 → has_deriv_at f (3 * (x^2 - x - 2)) x ∧ 3 * (x + 1) * (x - 2) < 0) ∧
  (∀ x, x > 2 → has_deriv_at f (3 * (x^2 - x - 2)) x ∧ 3 * (x + 1) * (x - 2) > 0) :=
sorry

end intervals_of_increase_and_decrease_l776_776103


namespace winning_percentage_l776_776348

noncomputable def total_votes (votes_winner votes_margin : ℕ) : ℕ :=
  votes_winner + (votes_winner - votes_margin)

noncomputable def percentage_votes (votes_winner total_votes : ℕ) : ℝ :=
  (votes_winner : ℝ) / (total_votes : ℝ) * 100

theorem winning_percentage
  (votes_winner : ℕ)
  (votes_margin : ℕ)
  (h_winner : votes_winner = 775)
  (h_margin : votes_margin = 300) :
  percentage_votes votes_winner (total_votes votes_winner votes_margin) = 62 :=
sorry

end winning_percentage_l776_776348


namespace point_not_on_segment_l776_776152

open Segment

namespace Geometry

def Point := ℝ × ℝ  -- Cartesian coordinate system for points

-- Define the length (distance) between two points
def length (A B : Point) : ℝ := ((A.1 - B.1)^2 + (A.2 - B.2)^2).sqrt

-- Define whether a point P is on the segment AB
def on_segment (A B P : Point) : Prop := 
  length A P + length P B = length A B

theorem point_not_on_segment (A B P : Point) (hAB: length A B = 10) (hPA_PB : length A P + length P B = 20) : ¬ on_segment A B P :=
by
  sorry

end Geometry

end point_not_on_segment_l776_776152


namespace detective_solves_case_l776_776079

theorem detective_solves_case :
  ∀ (persons : Fin 80 → Bool) (is_criminal : Fin 80 → Bool) (revealed : Fin 80 → Prop),
  (∃ cr : Fin 80, ∃ wit : Fin 80, cr ≠ wit ∧ is_criminal cr ∧ is_criminal wit = false ∧ ∀ n, persons n →
  -- On each day, the set of invited persons does not include the criminal but includes the witness
  (∃ day : Fin 12, ∃ invited : Fin 80 → Bool, 
   (∀ n, invited n → ¬is_criminal n) ∧ invited wit ∧ ∀ n, invited n → persons n)) →
  revealed wit :=
begin
  intros _ _ _ h,
  sorry
end

end detective_solves_case_l776_776079


namespace rational_ordering_l776_776069

theorem rational_ordering :
  (-3:ℚ)^2 < -1/3 ∧ (-1/3 < ((-3):ℚ)^2 ∧ ((-3:ℚ)^2 = |((-3:ℚ))^2|)) := 
by 
  sorry

end rational_ordering_l776_776069


namespace no_real_solution_for_equal_roots_l776_776073

noncomputable def quadratic_discriminant (a b c: ℝ) : ℝ := b^2 - 4 * a * c

theorem no_real_solution_for_equal_roots :
  ∀ (m : ℝ), ¬ ∃ x : ℝ, 
  (x * (x + 2) - (m + 3)) / ((x + 2) * (m + 2)) = x / m ∧ 
  quadratic_discriminant 1 4 (-m^2 - 3 * m) = 0 :=
begin
  sorry
end

end no_real_solution_for_equal_roots_l776_776073


namespace combination_10_3_l776_776454

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l776_776454


namespace odd_function_solution_set_l776_776991

theorem odd_function_solution_set (f : ℝ → ℝ)
    (h_odd : ∀ x, f (-x) = - f x)
    (h_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y):
  {x | f (x + 1) ≥ 0} = set.Ici (-1) := sorry

end odd_function_solution_set_l776_776991


namespace vector_sum_l776_776717

def vector_OA : ℂ := 1 + 4 * complex.i
def vector_OB : ℂ := -3 + 2 * complex.i

theorem vector_sum : vector_OA + vector_OB = -2 + 6 * complex.i := 
begin
  sorry
end

end vector_sum_l776_776717


namespace greatest_common_divisor_is_40_l776_776750

def distance_to_boston : ℕ := 840
def distance_to_atlanta : ℕ := 440

theorem greatest_common_divisor_is_40 :
  Nat.gcd distance_to_boston distance_to_atlanta = 40 :=
by
  -- The theorem statement as described is correct
  -- Proof is omitted as per instructions
  sorry

end greatest_common_divisor_is_40_l776_776750


namespace cyclist_wait_time_l776_776335

variable (hiker_speed : ℝ) (cyclist_speed : ℝ) (time_minutes : ℝ)

def wait_time (hiker_speed cyclist_speed time_minutes : ℝ) : ℝ :=
  let time_hours := time_minutes / 60
  let distance_cyclist := cyclist_speed * time_hours
  let distance_hiker := hiker_speed * time_hours
  let distance_ahead := distance_cyclist - distance_hiker
  (distance_ahead / hiker_speed) * 60

theorem cyclist_wait_time : 
  wait_time 6 30 12 = 48 := 
by
  sorry

end cyclist_wait_time_l776_776335


namespace volume_of_prism_correct_l776_776109

noncomputable def volume_of_prism (a α : ℝ) : ℝ :=
  (a^3 * real.sqrt (real.cos (2 * α))) / real.sin α

theorem volume_of_prism_correct (a α : ℝ) :
  volume_of_prism a α = (a^3 * real.sqrt (real.cos (2 * α))) / real.sin α :=
by sorry

end volume_of_prism_correct_l776_776109


namespace fraction_identity_l776_776214

theorem fraction_identity (x y : ℚ) (h : x / y = 5 / 3) : y / (x - y) = 3 / 2 :=
by { sorry }

end fraction_identity_l776_776214


namespace bisect_angle_DO_l776_776236

noncomputable def parallelogram := sorry

variables {A B C D E F O : Type} [parallelogram ABCD]
variables (on_AB : E ∈ AB) (on_BC : F ∈ BC) (CE_AF_eq : CE = AF) (CE_AF_inter : CE ∩ AF = O)

theorem bisect_angle_DO :
  bisects_angle DO AOC :=
sorry

end bisect_angle_DO_l776_776236


namespace number_of_even_permutations_l776_776533

variable {α : Type*} [DecidableEq α]

def is_even_permutation (σ : equiv.perm α) : Prop :=
  equiv.perm.sign σ = 1

def problem (a : Fin 6 → ℕ) : Prop :=
  (∀ i : Fin 6, a i ∈ Finset.univ.map ⟨Nat.succ, Nat.succ_injective⟩.to_fun) ∧
  ∏ i, (a i + i + 1) / 3 > 120

theorem number_of_even_permutations
  : ∃ (a : Finset (Fin 6 → ℕ)), a.card = 360 ∧ ∀ f ∈ a, problem f :=
sorry

end number_of_even_permutations_l776_776533


namespace foot_slide_distance_l776_776023

def ladder_foot_slide (l h_initial h_new x_initial d y: ℝ) : Prop :=
  l = 30 ∧ x_initial = 6 ∧ d = 6 ∧
  h_initial = Real.sqrt (l^2 - x_initial^2) ∧
  h_new = h_initial - d ∧
  (l^2 = h_new^2 + (x_initial + y) ^ 2) → y = 18

theorem foot_slide_distance :
  ladder_foot_slide 30 (Real.sqrt (30^2 - 6^2)) ((Real.sqrt (30^2 - 6^2)) - 6) 6 6 18 :=
by
  sorry

end foot_slide_distance_l776_776023


namespace number_of_times_difference_fits_is_20_l776_776692

-- Definitions for Ralph's pictures
def ralph_wild_animals := 75
def ralph_landscapes := 36
def ralph_family_events := 45
def ralph_cars := 20
def ralph_total_pictures := ralph_wild_animals + ralph_landscapes + ralph_family_events + ralph_cars

-- Definitions for Derrick's pictures
def derrick_wild_animals := 95
def derrick_landscapes := 42
def derrick_family_events := 55
def derrick_cars := 25
def derrick_airplanes := 10
def derrick_total_pictures := derrick_wild_animals + derrick_landscapes + derrick_family_events + derrick_cars + derrick_airplanes

-- Combined total number of pictures
def combined_total_pictures := ralph_total_pictures + derrick_total_pictures

-- Difference in wild animals pictures
def difference_wild_animals := derrick_wild_animals - ralph_wild_animals

-- Number of times the difference fits into the combined total (rounded down)
def times_difference_fits := combined_total_pictures / difference_wild_animals

-- Statement of the problem
theorem number_of_times_difference_fits_is_20 : times_difference_fits = 20 := by
  -- The proof will be written here
  sorry

end number_of_times_difference_fits_is_20_l776_776692


namespace compute_binomial_10_3_eq_120_l776_776478

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l776_776478


namespace calculate_elements_of_A_and_range_of_a_l776_776596

noncomputable def setA : set ℝ := {x | x^2 - 4 * x - 21 = 0}
noncomputable def setB (a : ℝ) : set ℝ := {x | 5 * x - a ≥ 3 * x + 2}

theorem calculate_elements_of_A_and_range_of_a (a : ℝ) :
  (setA = {-3, 7}) ∧ (setA ∪ setB a = setB a → a ≤ -8) :=
by
  sorry

end calculate_elements_of_A_and_range_of_a_l776_776596


namespace final_retail_price_l776_776410

theorem final_retail_price (wholesale_price markup_percentage discount_percentage desired_profit_percentage : ℝ)
  (h_wholesale : wholesale_price = 90)
  (h_markup : markup_percentage = 1)
  (h_discount : discount_percentage = 0.2)
  (h_desired_profit : desired_profit_percentage = 0.6) :
  let initial_retail_price := wholesale_price + (wholesale_price * markup_percentage)
  let discount_amount := initial_retail_price * discount_percentage
  let final_retail_price := initial_retail_price - discount_amount
  final_retail_price = 144 ∧ final_retail_price = wholesale_price + (wholesale_price * desired_profit_percentage) := by
 sorry

end final_retail_price_l776_776410


namespace selection_methods_count_l776_776887

def boys : Finset ℕ := {0, 1, 2, 3, 4}
def girls : Finset ℕ := {5, 6, 7, 8}

-- assuming boy A is represented by 0 and girl B is represented by 5

/-- Calculate the number of different selection methods for the given conditions -/
theorem selection_methods_count :
  let class := boys ∪ girls in
  let totalWays := class.card.choose 4 in
  let allBoysWays := boys.card.choose 4 in
  let allGirlsWays := girls.card.choose 4 in
  totalWays - allBoysWays - allGirlsWays = 86 := 
by
  let class := boys ∪ girls
  let totalWays := class.card.choose 4
  let allBoysWays := boys.card.choose 4
  let allGirlsWays := girls.card.choose 4
  have h1 : totalWays = 126 := sorry
  have h2 : allBoysWays = 1 := sorry
  have h3 : allGirlsWays = 1 := sorry
  show totalWays - allBoysWays - allGirlsWays = 86, by
    calc
      126 - 1 - 1 = 86 : by norm_num

end selection_methods_count_l776_776887


namespace conference_total_games_l776_776901

theorem conference_total_games :
  let teams := 12
  let divisions := 2
  let teams_per_division := 6
  let games_within_division := 3
  let games_with_other_division := 2
  let total_games := 162 
  teams = divisions * teams_per_division ∧
  (∀ d : ℕ, d < divisions →
    (∃ games_d : ℕ, games_d = teams_per_division * (teams_per_division - 1) * games_within_division / 2)) ∧
  (∃ games_inter : ℕ, games_inter = teams_per_division * teams_per_division * games_with_other_division / 2) →
  ( let total_games_within_division := divisions * teams_per_division * (teams_per_division - 1) * games_within_division / 2
    let total_games_inter_division := teams_per_division * teams_per_division * games_with_other_division / 2
    total_games_within_division + total_games_inter_division = total_games
  ) := by
  intros teams divisions teams_per_division games_within_division games_with_other_division total_games
  apply and.intro
  calc teams = divisions * teams_per_division

  sorry

end conference_total_games_l776_776901


namespace smallest_three_digit_multiple_of_eleven_l776_776765

theorem smallest_three_digit_multiple_of_eleven : ∃ n, n = 110 ∧ 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n := by
  sorry

end smallest_three_digit_multiple_of_eleven_l776_776765


namespace gcd_105_90_l776_776100

theorem gcd_105_90 : Nat.gcd 105 90 = 15 :=
by
  sorry

end gcd_105_90_l776_776100


namespace debtor_payment_l776_776437

-- Definitions based on conditions
def initial_balance : ℤ := 2000
def cheque_amount : ℤ := 600
def maintenance_cost : ℤ := 1200
def final_balance : ℤ := 1000

-- Theorem statement: How much did Ben's debtor pay him?
theorem debtor_payment (initial_balance cheque_amount maintenance_cost final_balance : ℤ) : ℤ :=
  initial_balance - cheque_amount - maintenance_cost + 800 = final_balance

#eval debtor_payment 2000 600 1200 1000 -- Expected output: 2000 - 600 - 1200 + 800 = 1000

end debtor_payment_l776_776437


namespace probability_is_correct_l776_776054

-- Definition of the problem: Rolling an 8-faced die 8 times.
def roll_die (n : ℕ) : Finset (Fin 8) → ℕ := sorry

-- Condition: Calculate the number of favorable outcomes (at least a seven)
def favorable_outcome : Fin 8 → Prop := λ n, n = 6 ∨ n = 7  -- 7 and 8 are 6 and 7 in 0-index.

def probability_at_least_seven_rolled_seven_times (total_rolls : ℕ := 8) (desired_outcome : ℕ := 7) : ℚ := begin
  -- favorable outcomes are (7 and 8) among die faces (1 to 8)
  have p_at_least_seven : ℚ := 2 / 8,
  have binom_8_7 : ℕ := Nat.choose 8 7,
  have prob_exact_7 : ℚ := binom_8_7 * (p_at_least_seven^7) * ((1 - p_at_least_seven)^1),
  have prob_exact_8 : ℚ := p_at_least_seven^8,
  have total_prob := prob_exact_7 + prob_exact_8,
  exact total_prob = 385 / 65536
end

theorem probability_is_correct : probability_at_least_seven_rolled_seven_times 8 7 = 385 / 65536 :=
by {
  exact sorry,
}

end probability_is_correct_l776_776054


namespace find_wholesale_prices_l776_776846

-- Definitions based on conditions
def bakerPlusPrice (p : ℕ) (d : ℕ) : ℕ := p + d
def starPrice (p : ℕ) (k : ℚ) : ℕ := (p * k).toNat

-- The tuple of prices given
def prices : List ℕ := [64, 64, 70, 72]

-- Statement of the problem
theorem find_wholesale_prices (d : ℕ) (k : ℚ) (h_d : d > 0) (h_k : k > 1 ∧ k.denom = 1):
  ∃ p1 p2, 
  (p1 ∈ prices ∧ bakerPlusPrice p1 d ∈ prices ∧ starPrice p1 k ∈ prices) ∧
  (p2 ∈ prices ∧ bakerPlusPrice p2 d ∈ prices ∧ starPrice p2 k ∈ prices) :=
sorry

end find_wholesale_prices_l776_776846


namespace part_a_l776_776036

theorem part_a (p n m : ℕ) (convex_polygon_divided_into_p_triangles : Prop) 
  (n_vertices_on_boundary : Prop) (m_vertices_inside_polygon : Prop) : 
  p = n + 2m - 2 :=
sorry

end part_a_l776_776036


namespace chessboard_knight_moves_l776_776301

theorem chessboard_knight_moves (marked_squares : Finset (Fin 8 × Fin 8)) :
  marked_squares.card = 17 →
  ∃ (sq1 sq2 : Fin 8 × Fin 8), sq1 ∈ marked_squares ∧ sq2 ∈ marked_squares ∧ 
  (sq1 ≠ sq2 ∧ knight_moves_required sq1 sq2 ≥ 3) := 
by
  -- Definitions relevant to knight's move and 2×2 figure partition are omitted in this statement for brevity.
  sorry

end chessboard_knight_moves_l776_776301


namespace min_quadrilateral_area_l776_776144

def circle_center : (ℝ × ℝ) := (1, 1)
def circle_radius : ℝ := 1
def line (x y : ℝ) : Prop := 3 * x + 4 * y + 8 = 0

def is_tangent (P A B : ℝ × ℝ) : Prop :=
  let PA := euclidean_distance P A in
  let PB := euclidean_distance P B in
  let C := circle_center in
  let r := circle_radius in
  PA = PB ∧ PA = real.sqrt ((euclidean_distance P C)^2 - r^2)

noncomputable def minimum_area_of_quadrilateral (P A B C : ℝ × ℝ) : ℝ :=
  2 * (real.sqrt 2)

theorem min_quadrilateral_area :
  ∀ (P : ℝ × ℝ), line P.1 P.2 →
  (∃ (A B : ℝ × ℝ), is_tangent P A B) →
  ∃ (C : ℝ × ℝ), C = circle_center →
  minimum_area_of_quadrilateral P A B circle_center = 2 * (real.sqrt 2) :=
by
  intro P hP
  rintro ⟨A, B, hTangent⟩
  rintro ⟨C, hC⟩
  rw hC
  exact sorry

end min_quadrilateral_area_l776_776144


namespace hyperbola_sufficient_asymptotes_l776_776719

open Real

def hyperbola_eq (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def asymptotes_eq (a b x y : ℝ) : Prop :=
  y = b / a * x ∨ y = - (b / a * x)

theorem hyperbola_sufficient_asymptotes (a b x y : ℝ) :
  (hyperbola_eq a b x y) → (asymptotes_eq a b x y) :=
by
  sorry

end hyperbola_sufficient_asymptotes_l776_776719


namespace sec_of_7pi_over_4_l776_776522

theorem sec_of_7pi_over_4 : (Real.sec (7 * Real.pi / 4)) = Real.sqrt 2 :=
by
  sorry

end sec_of_7pi_over_4_l776_776522


namespace closest_product_of_numbers_l776_776374

theorem closest_product_of_numbers :
  let product : ℝ := 0.00032 * 7500000
  let options := [210.0, 240.0, 2250.0, 2400.0, 24000.0]
  ∃ closest_option ∈ options, ∀ other_option ∈ options, 
    (abs (product - closest_option) ≤ abs (product - other_option)) :=
by
  let product := 0.00032 * 7500000
  let options := [210.0, 240.0, 2250.0, 2400.0, 24000.0]
  have : ∃ closest_option ∈ options, ∀ other_option ∈ options, 
    (abs (product - closest_option) ≤ abs (product - other_option)) := sorry
  assumption

end closest_product_of_numbers_l776_776374


namespace average_of_remaining_two_numbers_l776_776326

theorem average_of_remaining_two_numbers (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 3.95)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.6 :=
sorry

end average_of_remaining_two_numbers_l776_776326


namespace num_divisors_of_30_l776_776197

-- Define what it means to be a divisor of 30
def is_divisor (n : ℤ) : Prop :=
  n ≠ 0 ∧ 30 % n = 0

-- Define a function that counts the number of divisors of 30
def count_divisors (d : ℤ) : ℕ :=
  ((Multiset.filter is_divisor ((List.range' (-30) 61).map (Int.ofNat))).attach.to_finset.card)

-- State the theorem
theorem num_divisors_of_30 : count_divisors 30 = 16 := sorry

end num_divisors_of_30_l776_776197


namespace tetrahedron_division_l776_776074

noncomputable def regular_tetrahedron := sorry -- Placeholder for the definition of a regular tetrahedron

/-- A statement that the perpendicular bisector planes divide the tetrahedron into 24 parts, each with volume sqrt(2)/288. --/
theorem tetrahedron_division (T : regular_tetrahedron) :
  ∃ parts, 
    parts = 24 ∧ 
    ∀ part, volume part = (Real.sqrt 2) / 288 :=
sorry

end tetrahedron_division_l776_776074


namespace triangle_ratios_l776_776600

open EuclideanGeometry

theorem triangle_ratios (ABC : Triangle) (D : Point) (P Q : Point) (K L E F : Point)
  (hD : D ∈ lineSegment ABC.B ABC.C)
  (hP : P ∈ lineSegment ABC.A D)
  (hQ : Q ∈ lineSegment ABC.A D)
  (hK : K = intersection (line ABC.B P) (line ABC.C))
  (hL : L = intersection (line ABC.C P) (line ABC.B))
  (hE : E = intersection (line ABC.B Q) (line ABC.C))
  (hF : F = intersection (line ABC.C Q) (line ABC.B))
  (hKL_parallel_EF : parallel KL EF) :
  ratio (lineSegment ABC.B D) (lineSegment D ABC.C) = 1 := 
sorry

end triangle_ratios_l776_776600


namespace james_music_BPM_l776_776638

theorem james_music_BPM 
  (hours_per_day : ℕ)
  (beats_per_week : ℕ)
  (days_per_week : ℕ)
  (minutes_per_hour : ℕ)
  (minutes_per_day : ℕ)
  (total_minutes_per_week : ℕ)
  (BPM : ℕ)
  (h1 : hours_per_day = 2)
  (h2 : beats_per_week = 168000)
  (h3 : days_per_week = 7)
  (h4 : minutes_per_hour = 60)
  (h5 : minutes_per_day = hours_per_day * minutes_per_hour)
  (h6 : total_minutes_per_week = minutes_per_day * days_per_week)
  (h7 : BPM = beats_per_week / total_minutes_per_week)
  : BPM = 200 :=
sorry

end james_music_BPM_l776_776638


namespace determine_wholesale_prices_l776_776830

-- Definitions and Conditions
-- Wholesale prices for days 1 and 2 are defined as p1 and p2
def wholesale_price_day1 : ℝ := p1
def wholesale_price_day2 : ℝ := p2

-- The price increment at "Baker Plus" store is a constant d
def price_increment_baker_plus : ℝ := d
-- The factor at "Star" store is a constant k > 1
def price_factor_star : ℝ := k
-- Given prices for two days: 64, 64, 70, 72 rubles
def given_prices : set ℝ := {64, 64, 70, 72}

-- Prices at the stores
def baker_plus_day1_price := wholesale_price_day1 + price_increment_baker_plus
def star_day1_price := wholesale_price_day1 * price_factor_star

def baker_plus_day2_price := wholesale_price_day2 + price_increment_baker_plus
def star_day2_price := wholesale_price_day2 * price_factor_star

-- Theorem to determine the wholesale prices for each day (p1 and p2)
theorem determine_wholesale_prices
    (baker_plus_day1_price ∈ given_prices)
    (star_day1_price ∈ given_prices)
    (baker_plus_day2_price ∈ given_prices)
    (star_day2_price ∈ given_prices)
    (h1 : baker_plus_day1_price ≠ star_day1_price)
    (h2 : baker_plus_day2_price ≠ star_day2_price) :
  (∃ p1 p2 d k > 1, 
      (p1 + d ∈ given_prices) ∧ (p1 * k ∈ given_prices) ∧
      (p2 + d ∈ given_prices) ∧ (p2 * k ∈ given_prices) ∧
      (p1 ≠ p2)) :=
  sorry

end determine_wholesale_prices_l776_776830


namespace variance_of_data_set_l776_776984

open Real

-- Data set and conditions
def data_set : List ℝ := [5, 4, x, 3, 6]
def mean : ℝ := 5

-- Calculation of the variance
def variance (data : List ℝ) (mean : ℝ) : ℝ :=
  (1 / data.length) * (data.map (λ xi => (xi - mean)^2)).sum

theorem variance_of_data_set (x : ℝ) (h : (1 / 5) * (5 + 4 + x + 3 + 6) = 5) : variance [5, 4, x, 3, 6] 5 = 7 / 5 := by
  sorry

end variance_of_data_set_l776_776984


namespace avg_speed_is_17_l776_776349

-- Define the distances and speeds
def d1 := 50 -- miles
def d2 := 30 -- miles
def s1 := 20 -- miles per hour
def s2 := 15 -- miles per hour

-- Calculate the total distance
def total_distance := d1 + d2

-- Calculate the times for each segment
def t1 := d1 / s1
def t2 := d2 / s2

-- Calculate the total time
def total_time := t1 + t2

-- Define the average speed
def average_speed := total_distance / total_time

-- Prove that the average speed is approximately 17.78 miles per hour
theorem avg_speed_is_17.78 : average_speed ≈ 17.78 := by sorry

end avg_speed_is_17_l776_776349


namespace all_points_enclosed_in_circle_of_radius_1_l776_776810

theorem all_points_enclosed_in_circle_of_radius_1 
  (n : ℕ) 
  (points : Fin n → ℝ × ℝ)
  (h : ∀ (i j k : Fin n), (∃ (c : ℝ × ℝ), ∀ (p ∈ {i, j, k}.to_finset, dist (points p) c ≤ 1))) :
  ∃ (c : ℝ × ℝ), ∀ (p : Fin n), dist (points p) c ≤ 1 :=
sorry

end all_points_enclosed_in_circle_of_radius_1_l776_776810


namespace area_on_larger_sphere_l776_776415

-- Define the radii of the spheres
def r_small : ℝ := 1
def r_in : ℝ := 4
def r_out : ℝ := 6

-- Given the area on the smaller sphere
def A_small_sphere_area : ℝ := 37

-- Statement: Find the area on the larger sphere
theorem area_on_larger_sphere :
  (A_small_sphere_area * (r_out / r_in) ^ 2 = 83.25) := by
  sorry

end area_on_larger_sphere_l776_776415


namespace transform_sheet_to_figure_l776_776179

-- Definitions for the conditions
structure RectangularSheet (α : Type) :=
  (width : α)
  (height : α)

structure LineSegment (α : Type) :=
  (length : α)
  (position : α)  -- position can be represented as a coordinate, but simplified here for illustration

structure Diagram (α : Type) :=
  (a : LineSegment α)
  (b : LineSegment α)
  (c : LineSegment α)
  (fold_points : list α)  -- fold points could be a list of positions representing where folds need to occur

-- Definition of making cuts and folds
def makeCutsAndFolds (sheet : RectangularSheet ℝ) (diagram : Diagram ℝ) : RectangularSheet ℝ := 
  sorry  -- This would be the detailed cutting and folding logic

-- The theorem statement
theorem transform_sheet_to_figure (sheet : RectangularSheet ℝ) (diagram : Diagram ℝ) :
  (makeCutsAndFolds sheet diagram = desiredFigure) := 
  sorry

end transform_sheet_to_figure_l776_776179


namespace range_of_f_l776_776340

-- Define the piecewise function f(x)
def f (x : ℝ) : ℝ :=
if x ≥ 1 then log (1/3 : ℝ) x else 3 ^ x

-- Theorem statement
theorem range_of_f : range f = set.Iio 3 := by
  sorry

end range_of_f_l776_776340


namespace solve_inequality_l776_776660

def f (x : ℝ) : ℝ := sorry

axiom functional_eq (x1 x2 : ℝ) (hx1 : x1 ≠ 0) (hx2 : x2 ≠ 0) : f (x1 * x2) = f x1 + f x2

axiom increasing (x1 x2 : ℝ) (hx1_pos : 0 < x1) (hx2_pos : 0 < x2) (hx1_le_hx2 : x1 ≤ x2) :
  f x1 ≤ f x2

theorem solve_inequality : { x : ℝ // f x + f (x - 1/2) ≤ 0 } =
  set_of (λ x, (1 - real.sqrt 17) / 4 ≤ x ∧ x < 0 ∨
                  0 < x ∧ x < 1/2 ∨
                  1/2 < x ∧ x ≤ (1 + real.sqrt 17) / 4) :=
begin
  sorry
end

end solve_inequality_l776_776660


namespace solve_system_of_equations_l776_776702

-- Statement of our problem
theorem solve_system_of_equations
  (x y z : ℝ)
  (h1 : x + y + z = 15)
  (h2 : x^2 + y^2 + z^2 = 81)
  (h3 : xy + xz = 3yz) :
  (x = 6 ∧ y = 3 ∧ z = 6) ∨ (x = 6 ∧ y = 6 ∧ z = 3) :=
sorry

end solve_system_of_equations_l776_776702


namespace votes_difference_l776_776626

theorem votes_difference (T : ℕ) (V_a : ℕ) (V_f : ℕ) 
  (h1 : T = 330) (h2 : V_a = 40 * T / 100) (h3 : V_f = T - V_a) : V_f - V_a = 66 :=
by
  sorry

end votes_difference_l776_776626


namespace find_a_max_OA_OB_angle_l776_776635

namespace Geometry

-- Define the condition for curve C and line l
def curve_C (a : ℝ) (θ : ℝ) : ℝ := 4 * a * Real.cos θ
def line_l (θ : ℝ) : ℝ := 4 / Real.cos (θ - Real.pi / 3)

-- Given condition a > 0
variable {a : ℝ} (ha : a > 0)

-- Problem 1: Find 'a' such that C and l have exactly one intersection point
theorem find_a (ha : a > 0) : 
  (∃ θ, curve_C a θ = line_l θ) ↔ a = 4 / 3 :=
sorry

-- Auxiliary definitions for Problem 2
def polar_angle_OA := λ θ : ℝ, curve_C a θ
def polar_angle_OB := λ θ : ℝ, curve_C a (θ + Real.pi / 3)

-- Problem 2: Maximum value of |OA| + |OB| given a = 4 / 3 and ∠AOB = π / 3
theorem max_OA_OB_angle (a : ℝ) (ha : a > 0) 
  (θ : ℝ) (hθ: a = 4 / 3) :  
  let OA := polar_angle_OA θ,
      OB := polar_angle_OB θ in
  max (OA θ + OB θ) = 16 * Real.sqrt 3 / 3 :=
sorry

end Geometry

end find_a_max_OA_OB_angle_l776_776635


namespace cos_alpha_l776_776156

theorem cos_alpha
  (α : ℝ)
  (hx : cos α = 1 / 2)
  (hy : sin α = sqrt 3 / 2) :
  cos α = 1 / 2 := 
by
  sorry

end cos_alpha_l776_776156


namespace num_ways_to_place_two_marbles_in_boxes_l776_776345

theorem num_ways_to_place_two_marbles_in_boxes :
  ∃ (marbles : finset ℕ), 
  marbles.card = 3 ∧ 
  ∀ (boxes : finset (finset ℕ)), 
  boxes.card = 1 → 
  ∃ (ways : ℕ), ways = 3 :=
by
  sorry

end num_ways_to_place_two_marbles_in_boxes_l776_776345


namespace no_form3000001_is_perfect_square_l776_776691

theorem no_form3000001_is_perfect_square (n : ℕ) : 
  ∀ k : ℤ, (3 * 10^n + 1 ≠ k^2) :=
by
  sorry

end no_form3000001_is_perfect_square_l776_776691


namespace min_percentage_all_solved_l776_776622

variable {P_1 P_2 P_3 P_4 : Prop}
variable {participants : ℝ}

def percentage_solved (event : Prop) : ℝ := sorry

-- Conditions
axiom hP1 : percentage_solved P_1 = 0.85
axiom hP2 : percentage_solved P_2 = 0.80
axiom hP3 : percentage_solved P_3 = 0.75
axiom hP4 : percentage_solved P_4 = 0.70

-- Define the minimum percentage of participants who solved all problems
def min_percentage_solved_all : ℝ := 0.10

-- Statement to prove
theorem min_percentage_all_solved :
  minimum_percentage_all_solved participants :=
by 
  sorry

end min_percentage_all_solved_l776_776622


namespace rate_of_profit_is_hundred_l776_776384

-- Let's define cost price and selling price
def cost_price : ℝ := 50
def selling_price : ℝ := 100

-- Define profit as selling price minus cost price
def profit : ℝ := selling_price - cost_price

-- Define rate of profit as (profit / cost price) * 100%
def rate_of_profit : ℝ := (profit / cost_price) * 100

-- Now state the theorem that rate of profit is 100% given the conditions above
theorem rate_of_profit_is_hundred : rate_of_profit = 100 := by
  sorry

end rate_of_profit_is_hundred_l776_776384


namespace bakery_wholesale_price_exists_l776_776859

theorem bakery_wholesale_price_exists :
  ∃ (p1 p2 d k : ℕ), (d > 0) ∧ (k > 1) ∧
  ({p1 + d, p2 + d, k * p1, k * p2} = {64, 64, 70, 72}) :=
by sorry

end bakery_wholesale_price_exists_l776_776859


namespace polynomial_satisfying_condition_l776_776016

open Polynomial

theorem polynomial_satisfying_condition (P : Polynomial ℝ) :
  (∀ x : ℝ, (x - 16) * P.eval (2 * x) = 16 * (x - 1) * P.eval x) →
  ∃ λ : ℝ, P = C λ * (X - C 2) * (X - C 4) * (X - C 8) * (X - C 16) :=
by
  sorry

end polynomial_satisfying_condition_l776_776016


namespace three_digit_integers_with_2_not_5_l776_776207

theorem three_digit_integers_with_2_not_5 :
  let count := (648 - 448)
  in count = 200 :=
by
  -- The proof follows the solution steps described in the problem.
  -- Calculate total three-digit numbers without 2 and 5: 448
  let count_without_2_and_5 := 7 * 8 * 8
  -- Calculate total three-digit numbers without 5: 648
  let count_without_5 := 8 * 9 * 9
  -- The count of numbers with at least one 2 and not containing 5: 648 - 448
  have count := count_without_5 - count_without_2_and_5
  -- Assert the final count is 200
  show count = 200 from sorry


end three_digit_integers_with_2_not_5_l776_776207


namespace probability_mixed_sample_positive_optimal_testing_plan_l776_776582

-- Problem Part 1
theorem probability_mixed_sample_positive (p : ℝ) (n : ℕ)
    (hp : p = 0.1) (hn : n = 2) :
    let η := λ x, 1 - (1 - p) ^ x
in η n = 0.19 := by
  sorry

-- Problem Part 2
theorem optimal_testing_plan (p : ℝ) (n : ℕ) (hn : n = 4) (hp : p = 0.1) :
    let ev_plan1 := λ x, x
    let P2_X := [0.81 * 0.81, 2 * 0.81 * 0.19, 0.19 * 0.19]
    let ev_plan2 := 2 * P2_X.head! + 4 * P2_X[1]! + 6 * P2_X[2]!
    let P3_Y := [0.9^4, 1 - 0.9^4]
    let ev_plan3 := 1 * P3_Y.head! + 5 * P3_Y[1]!
in ev_plan3 < ev_plan2 ∧ ev_plan2 < 4 := by
  sorry

end probability_mixed_sample_positive_optimal_testing_plan_l776_776582


namespace sec_of_7pi_over_4_l776_776520

theorem sec_of_7pi_over_4 : (Real.sec (7 * Real.pi / 4)) = Real.sqrt 2 :=
by
  sorry

end sec_of_7pi_over_4_l776_776520


namespace total_weight_CaBr2_l776_776792

-- Definitions derived from conditions
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_Br : ℝ := 79.904
def mol_weight_CaBr2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_Br
def moles_CaBr2 : ℝ := 4

-- Theorem statement based on the problem and correct answer
theorem total_weight_CaBr2 : moles_CaBr2 * mol_weight_CaBr2 = 799.552 :=
by
  -- Prove the theorem step-by-step
  -- substitute the definition of mol_weight_CaBr2
  -- show lhs = rhs
  sorry

end total_weight_CaBr2_l776_776792


namespace ball_reaches_bottom_tube_l776_776015
open ProbabilityTheory

noncomputable def probability_reaching_bottom_tube : ℝ :=
  (∑ k in finset.range (5 + 1), (nat.choose 5 k) * (1 / 3) ^ k * (2 / 3) ^ (5 - k))

theorem ball_reaches_bottom_tube :
  probability_reaching_bottom_tube = 0.296 :=
by sorry

end ball_reaches_bottom_tube_l776_776015


namespace wickets_before_last_match_l776_776007

theorem wickets_before_last_match (R W : ℝ) (h1 : R = 12.4 * W) (h2 : R + 26 = 12 * (W + 7)) :
  W = 145 := 
by 
  sorry

end wickets_before_last_match_l776_776007


namespace value_of_z_is_7_l776_776105

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00
def molecular_weight_compound : ℝ := 192.0

def weight_C6 : ℝ := 6 * atomic_weight_C
def weight_H8 : ℝ := 8 * atomic_weight_H

theorem value_of_z_is_7 (z : ℝ) (h : molecular_weight_compound = weight_C6 + weight_H8 + z * atomic_weight_O) : z ≈ 7 :=
by sorry

end value_of_z_is_7_l776_776105


namespace power_function_odd_and_real_range_l776_776559

theorem power_function_odd_and_real_range {a : ℤ} (h : a ∈ {-1, 1, 2, 3}) :
  (∀ x : ℝ, x ≠ 0 → (λ x, x ^ a) x = - (λ x, x ^ a) (-x)) ∧ (∀ y : ℝ, ∃ x : ℝ, x ^ a = y) ↔ a = 1 ∨ a = 3 :=
by 
  sorry

end power_function_odd_and_real_range_l776_776559


namespace least_positive_three_digit_multiple_of_13_is_104_l776_776785

theorem least_positive_three_digit_multiple_of_13_is_104 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ n = 104 :=
by
  existsi 104
  split
  · show 100 ≤ 104
    exact le_refl 104
  split
  · show 104 < 1000
    exact dec_trivial
  split
  · show 104 % 13 = 0
    exact dec_trivial
  · show 104 = 104
    exact rfl

end least_positive_three_digit_multiple_of_13_is_104_l776_776785


namespace listens_doubling_l776_776264

theorem listens_doubling (m : ℕ) (init_listens : ℕ) (final_listens : ℕ) 
    (init_condition : init_listens = 60000)
    (final_condition : final_listens = 900000) :
    init_listens * 2 ^ m = final_listens → m = 3 :=
by
  intros H
  have h : 60000 * 2 ^ m = 900000, from H
  sorry

end listens_doubling_l776_776264


namespace solution_to_linear_1_l776_776001

theorem solution_to_linear_1 : (x y : ℤ) (h1 : x = 1) (h2 : y = -1) : x - 3 * y = 4 :=
by {
  intros,
  rw [h1, h2],
  norm_num,
}

end solution_to_linear_1_l776_776001


namespace average_divisible_by_7_l776_776947

theorem average_divisible_by_7 (s : Set ℕ) (H : s = {n | 10 ≤ n ∧ n ≤ 100 ∧ n % 7 = 0}) : 
  (∑ n in s, n) / s.card = 98 := 
sorry

end average_divisible_by_7_l776_776947


namespace pie_wholesale_price_l776_776822

theorem pie_wholesale_price (p1 p2: ℕ) (d: ℕ) (k: ℕ) (h1: d > 0) (h2: k > 1)
  (price_list: List ℕ)
  (p_plus_1: price_list = [p1 + d, k * p1, p2 + d, k * p2]) 
  (h3: price_list.perm [64, 64, 70, 72]) : 
  p1 = 60 ∧ p2 = 60 := 
sorry

end pie_wholesale_price_l776_776822


namespace sum_b_k_l776_776734

/-- Define the sequence a_n -/
def a : ℕ → ℝ
| 0     := 1
| (n+1) := -(a n + 3 * ↑(n+1)) / 2 -- This is based on the derived recurrence relation in the solution

/-- Define b_n based on a_n and a_(n+1) -/
def b (n : ℕ) : ℝ := a n * a (n+1)

/-- Prove the sum of b_k from k = 1 to 20 equals 6385 -/
theorem sum_b_k : (Finset.range 20).sum (λ k => b (k+1)) = 6385 :=
sorry

end sum_b_k_l776_776734


namespace max_angle_AFB_is_2pi_over_3_l776_776720

noncomputable def parabola := {x // ∃ y : ℝ, y^2 = 4 * x}

theorem max_angle_AFB_is_2pi_over_3
  (F : parabola)
  (A B : parabola)
  (x1 y1 x2 y2 : ℝ)
  (hA : (y1^2 = 4 * x1) ∧ A = ⟨x1, ⟨y1, rfl⟩⟩)
  (hB : (y2^2 = 4 * x2) ∧ B = ⟨x2, ⟨y2, rfl⟩⟩)
  (hAB : dist A B = (sqrt 3 / 2) * (x1 + x2 + 2)) :
  ∃ θ : ℝ, θ = (2 * π / 3) ∧ θ = ∠ A F B :=
begin
  sorry
end

end max_angle_AFB_is_2pi_over_3_l776_776720


namespace melted_ice_cream_depth_l776_776417

noncomputable def radius_sphere : ℝ := 3
noncomputable def radius_cylinder : ℝ := 10
noncomputable def height_cylinder : ℝ := 36 / 100

theorem melted_ice_cream_depth :
  (4 / 3) * Real.pi * radius_sphere^3 = Real.pi * radius_cylinder^2 * height_cylinder :=
by
  sorry

end melted_ice_cream_depth_l776_776417


namespace unique_function_satisfying_inequality_l776_776106

theorem unique_function_satisfying_inequality : 
  ∃! (f : ℝ → ℝ), (∀ x y z : ℝ, f(x * y) + f(x * z) + f(x) * f(y * z) ≥ 3) ∧ (∀ x : ℝ, f(x) = 1) :=
by
  sorry

end unique_function_satisfying_inequality_l776_776106


namespace correct_sequence_is_A_l776_776000

def Step := String
def Sequence := List Step

def correct_sequence : Sequence :=
  ["Buy a ticket", "Wait for the train", "Check the ticket", "Board the train"]

def option_A : Sequence :=
  ["Buy a ticket", "Wait for the train", "Check the ticket", "Board the train"]
def option_B : Sequence :=
  ["Wait for the train", "Buy a ticket", "Board the train", "Check the ticket"]
def option_C : Sequence :=
  ["Buy a ticket", "Wait for the train", "Board the train", "Check the ticket"]
def option_D : Sequence :=
  ["Repair the train", "Buy a ticket", "Check the ticket", "Board the train"]

theorem correct_sequence_is_A :
  correct_sequence = option_A :=
sorry

end correct_sequence_is_A_l776_776000


namespace clothing_store_earnings_l776_776033

-- Defining the conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def shirt_cost : ℕ := 10
def jeans_cost : ℕ := 2 * shirt_cost

-- Statement of the problem
theorem clothing_store_earnings :
  num_shirts * shirt_cost + num_jeans * jeans_cost = 400 :=
by
  sorry

end clothing_store_earnings_l776_776033


namespace binom_10_3_l776_776457

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l776_776457


namespace return_trip_time_l776_776042

-- conditions 
variables (d p w : ℝ) (h1 : d = 90 * (p - w)) (h2 : ∀ t : ℝ, t = d / p → d / (p + w) = t - 15)

--  statement
theorem return_trip_time :
  ∃ t : ℝ, t = 30 ∨ t = 45 :=
by
  -- placeholder proof 
  sorry

end return_trip_time_l776_776042


namespace initial_quantity_of_milk_in_A_l776_776429

theorem initial_quantity_of_milk_in_A (A : ℝ) 
  (h1: ∃ C B: ℝ, B = 0.375 * A ∧ C = 0.625 * A) 
  (h2: ∃ M: ℝ, M = 0.375 * A + 154 ∧ M = 0.625 * A - 154) 
  : A = 1232 :=
by
  -- you can use sorry to skip the proof
  sorry

end initial_quantity_of_milk_in_A_l776_776429


namespace closest_multiple_of_15_to_3500_l776_776795

theorem closest_multiple_of_15_to_3500 (N : ℤ) (h : abs(N - 3500) ≤ 7 ∧ N % 15 = 0) : N = 3510 :=
sorry

end closest_multiple_of_15_to_3500_l776_776795


namespace minimum_of_f_add_a_eq_sqrt3_l776_776747

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + (Real.pi / 3))

theorem minimum_of_f_add_a_eq_sqrt3
  (a : ℝ) :
  (∀ x ∈ set.Icc 0 (Real.pi / 2), f(x) + a ≥ sqrt 3) ∧
  (∃ x ∈ set.Icc 0 (Real.pi / 2), f(x) + a = sqrt 3)
  ↔ a = 2 * sqrt 3 := by
  sorry

end minimum_of_f_add_a_eq_sqrt3_l776_776747


namespace find_Y_l776_776615

-- Definition of the problem.
def arithmetic_sequence (a d n : ℕ) : ℕ := a + d * (n - 1)

-- Conditions provided in the problem.
-- Conditions of the first row
def first_row (a₁ a₄ : ℕ) : Prop :=
  a₁ = 4 ∧ a₄ = 16

-- Conditions of the last row
def last_row (a₁' a₄' : ℕ) : Prop :=
  a₁' = 10 ∧ a₄' = 40

-- Value of Y (the second element of the second row from the second column)
def center_top_element (Y : ℕ) : Prop :=
  Y = 12

-- The theorem to prove.
theorem find_Y (a₁ a₄ a₁' a₄' Y : ℕ) (h1 : first_row a₁ a₄) (h2 : last_row a₁' a₄') (h3 : center_top_element Y) : Y = 12 := 
by 
  sorry -- proof to be provided.

end find_Y_l776_776615


namespace probability_LM_accurate_l776_776341

noncomputable def defect_probability {K L M : Type} 
  (def_k : ℝ) (def_l : ℝ) (def_m : ℝ) 
  (two_defective : ℝ)
  (defective_lm : ℝ) : Prop :=
  def_k = 0.1 ∧ def_l = 0.2 ∧ def_m = 0.15 ∧ 
  two_defective = 0.056 ∧ 
  defective_lm = 0.4821

theorem probability_LM_accurate : 
  ∃ (K L M : Type), ∀ (def_k def_l def_m two_defective defective_lm : ℝ),
    defect_probability def_k def_l def_m two_defective defective_lm → 
    defective_lm ≈ 0.4821 :=
by 
  sorry

end probability_LM_accurate_l776_776341


namespace projectile_height_reaches_45_at_t_0_5_l776_776718

noncomputable def quadratic (a b c : ℝ) : ℝ → ℝ :=
  λ t => a * t^2 + b * t + c

theorem projectile_height_reaches_45_at_t_0_5 :
  ∃ t : ℝ, quadratic (-16) 98.5 (-45) t = 45 ∧ 0 ≤ t ∧ t = 0.5 :=
by
  sorry

end projectile_height_reaches_45_at_t_0_5_l776_776718


namespace odd_function_of_power_l776_776154

noncomputable def f (a b x : ℝ) : ℝ := (a - 1) * x ^ b

theorem odd_function_of_power (a b : ℝ) (h : f a b a = 1/2) : 
  ∀ x : ℝ, f a b (-x) = -f a b x := 
by
  sorry

end odd_function_of_power_l776_776154


namespace compute_binomial_10_3_eq_120_l776_776479

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l776_776479


namespace binom_10_3_l776_776460

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l776_776460


namespace comb_10_3_eq_120_l776_776502

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l776_776502


namespace not_proper_subset_of_intersection_l776_776599

def M : Set ℤ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem not_proper_subset_of_intersection :
  ¬(Set.subset (∅ : Set ℤ) ∪ {0} ∪ {1} (M ∩ N) ∧ Set.subset (M ∩ N) {0, 1}) :=
by
  sorry

end not_proper_subset_of_intersection_l776_776599


namespace num_divisors_of_30_l776_776200

-- Define what it means to be a divisor of 30
def is_divisor (n : ℤ) : Prop :=
  n ≠ 0 ∧ 30 % n = 0

-- Define a function that counts the number of divisors of 30
def count_divisors (d : ℤ) : ℕ :=
  ((Multiset.filter is_divisor ((List.range' (-30) 61).map (Int.ofNat))).attach.to_finset.card)

-- State the theorem
theorem num_divisors_of_30 : count_divisors 30 = 16 := sorry

end num_divisors_of_30_l776_776200


namespace wholesale_prices_l776_776838

-- Definitions for the problem conditions
variable (p1 p2 d k : ℝ)
variable (h_d : d > 0)
variable (h_k : k > 1)
variable (prices : Finset ℝ)
variable (h_prices : prices = {64, 64, 70, 72})

-- The theorem statement to prove
theorem wholesale_prices :
  ∃ p1 p2, (p1 + d ∈ prices ∧ k * p1 ∈ prices) ∧ 
           (p2 + d ∈ prices ∧ k * p2 ∈ prices) ∧ 
           p1 ≠ p2
:= sorry

end wholesale_prices_l776_776838


namespace ellipse_eccentricity_l776_776997

theorem ellipse_eccentricity (x y : ℝ) (h : x^2 / 25 + y^2 / 9 = 1) : 
  let a := 5
  let b := 3
  let c := 4
  let e := c / a
  e = 4 / 5 :=
by
  sorry

end ellipse_eccentricity_l776_776997


namespace sledding_ratio_l776_776380

theorem sledding_ratio
  (tall_hills : ℕ := 2)
  (sleds_per_tall_hill : ℕ := 4)
  (small_hills : ℕ := 3)
  (total_sleds : ℕ := 14)
  (x : ℕ := (total_sleds - 2 * sleds_per_tall_hill) / small_hills) :
  (3 * x) / (2 * sleds_per_tall_hill) = 3 / 4 :=
by
  -- Definitions based on conditions
  have total_sleds_tall : ℕ := tall_hills * sleds_per_tall_hill
  have total_sleds_small : ℕ := total_sleds - total_sleds_tall
  have x_val : ℕ := total_sleds_small / small_hills

  -- Proof of the ratio
  calc
    (3 * x_val) / (2 * sleds_per_tall_hill) = (3 * (total_sleds_small / small_hills)) / (2 * sleds_per_tall_hill) : by rfl
    ... = (3 * (total_sleds_small / small_hills)) / 8 : by simp only [sleds_per_tall_hill]
    ... = (3 * (6 / 3)) / 8 : by simp only [total_sleds, total_sleds_small]
    ... = 6 / 8 : by simpl (3 * 2)
    ... = 3 / 4 : by norm_num

-- Closing the proof with sorry to ensure it builds successfully without requiring the entire solution
sorry

end sledding_ratio_l776_776380


namespace motel_total_rent_l776_776388

theorem motel_total_rent (R40 R60 : ℕ) (total_rent : ℕ) 
  (h1 : total_rent = 40 * R40 + 60 * R60) 
  (h2 : 40 * (R40 + 10) + 60 * (R60 - 10) = total_rent - total_rent / 10) 
  (h3 : total_rent / 10 = 200) : 
  total_rent = 2000 := 
sorry

end motel_total_rent_l776_776388


namespace find_length_of_AB_to_nearest_tenth_l776_776237

-- Defining our setup including the conditions
def triangle_ABC_right (A B C : Type) [angleA : 30] [angleB : 90] [segmentBC : 7] : Prop := sorry

-- Stating the theorem
theorem find_length_of_AB_to_nearest_tenth (A B C : Type) [right_triangle : triangle_ABC_right A B C] : 
  (find_length_of_AB A B C) = 4.0 := 
sorry

end find_length_of_AB_to_nearest_tenth_l776_776237


namespace work_days_l776_776804

theorem work_days (hp : ℝ) (hq : ℝ) (fraction_left : ℝ) (d : ℝ) :
  hp = 1 / 20 → hq = 1 / 10 → fraction_left = 0.7 → (3 / 20) * d = (1 - fraction_left) → d = 2 :=
  by
  intros hp_def hq_def fraction_def work_eq
  sorry

end work_days_l776_776804


namespace least_positive_three_digit_multiple_of_11_l776_776762

theorem least_positive_three_digit_multiple_of_11 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 11 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 11 ∣ m → n ≤ m) :=
begin
  use 110,
  split,
  exact (100 ≤ 110),
  split,
  exact (110 ≤ 999),
  split,
  exact (11 ∣ 110),
  intros m Hm,
  cases Hm with Hm100 HmGCD,
  cases HmGCD with Hm999 HmDiv,
  sorry,
end

end least_positive_three_digit_multiple_of_11_l776_776762


namespace range_of_inverse_proportion_l776_776168

theorem range_of_inverse_proportion (x : ℝ) (h : 3 < x) :
    -1 < -3 / x ∧ -3 / x < 0 :=
by
  sorry

end range_of_inverse_proportion_l776_776168


namespace expec_sup_finite_l776_776138

open ProbabilityTheory

variables (X : ℕ → ℝ)
variable [MeasurableSpace X]
variable [ProbabilitySpace X]

theorem expec_sup_finite (h : P (supr X < ∞) = 1) : E (supr X) < ∞ :=
sorry

end expec_sup_finite_l776_776138


namespace actual_distance_l776_776893

-- Define the problem conditions
def scale_factor := 700
def map_distance := 5.5

-- State the theorem we want to prove
theorem actual_distance : map_distance * scale_factor = 3850 := 
by 
  sorry

end actual_distance_l776_776893


namespace find_A_max_area_triangle_l776_776257

-- Declare noncomputable theory as necessary
noncomputable theory

variables {A B C : ℝ} -- Angles A, B, C
variables {a b c : ℝ} -- Sides opposite to angles A, B, C

-- Given conditions and question for proving the value of angle A and finding the max area
def cond1 : Prop := 2 * b * Real.cos A = c * Real.cos A + a * Real.cos C
def cond2 : a = 4
def value_A := A = Real.pi / 3
def max_area := ∃ b c, 4 * Real.sqrt 3

-- Statement 1: Proving angle A
theorem find_A (h1 : cond1) : value_A := by
  sorry

-- Statement 2: Finding maximum area with a = 4
theorem max_area_triangle (h1 : cond1) (h2 : cond2) : max_area := by
  sorry

end find_A_max_area_triangle_l776_776257


namespace sign_of_c_l776_776145

/-
Define the context and conditions as Lean axioms.
-/

variables (a b c : ℝ)

-- Axiom: The sum of coefficients is less than zero
axiom h1 : a + b + c < 0

-- Axiom: The quadratic equation has no real roots, thus the discriminant is less than zero
axiom h2 : (b^2 - 4*a*c) < 0

/-
Formal statement of the proof problem:
-/

theorem sign_of_c : c < 0 :=
by
  -- We state that the proof of c < 0 follows from the given axioms
  sorry

end sign_of_c_l776_776145


namespace sec_7pi_over_4_l776_776528

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_7pi_over_4 :
  sec (7 * Real.pi / 4) = Real.sqrt 2 := by
  have h1 : Real.cos (7 * Real.pi / 4) = Real.cos (45 * Real.pi / 180) := by sorry
  have h2 : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  rw [sec, h1, h2]
  field_simp
  norm_num

end sec_7pi_over_4_l776_776528


namespace seq_formula_seq_sum_formula_l776_776153

-- Given conditions
def f (x : ℝ) : ℝ := a ^ x
axiom a_pos : a > 0
axiom a_ne_one : a ≠ 1
axiom point_on_graph : f 1 = 2

-- First part: general formula for the sequence {a_n}
def S (n : ℕ) : ℝ := f n - 1
def a_n (n : ℕ) : ℝ := if n = 1 then S 1 else S n - S (n - 1)

theorem seq_formula (n : ℕ) : a_n n = 2 ^ (n - 1) :=
sorry

-- Second part: sum of the terms of the sequence {a_n b_n}
def b_n (n : ℕ) : ℝ := Real.log a (a_n (n + 1))

def a_n_b_n (n : ℕ) : ℝ := n * 2 ^ (n - 1)
def T (n : ℕ) : ℝ := (n - 1) * 2 ^ n + 1

theorem seq_sum_formula (n : ℕ) : (finset.range n).sum (λ k, a_n_b_n k.succ) = T n :=
sorry

end seq_formula_seq_sum_formula_l776_776153


namespace work_completion_time_l776_776029

-- Conditions
def A_days := 40
def B_days := 60

-- Definitions for work rates
def A_work_rate := (1 : ℝ) / A_days
def B_work_rate := (1 : ℝ) / B_days

-- Combined work rate
def combined_work_rate := A_work_rate + B_work_rate

-- Proof that combined work rate results in 24 days to complete the work
theorem work_completion_time : (1 / combined_work_rate) = 24 := 
  sorry

end work_completion_time_l776_776029


namespace liars_numbers_l776_776634

theorem liars_numbers (numbers: finset ℕ)
  (h_unique: ∀ n ∈ numbers, 1 ≤ n ∧ n ≤ 10)
  (h_div2: ∑(λ n, if n % 2 == 0 then 1 else 0) (numbers : multiset ℕ) = 3)
  (h_div4: ∑(λ n, if n % 4 == 0 then 1 else 0) (numbers : multiset ℕ) = 6)
  (h_div5: ∑(λ n, if n % 5 == 0 then 1 else 0) (numbers : multiset ℕ) = 2):
  { 2, 5, 6, 10 } ⊆ numbers :=
by {
  sorry
}

end liars_numbers_l776_776634


namespace determine_wholesale_prices_l776_776853

variables (p1 p2 d k : ℝ)
variables (h1 : 1 < k)
variables (prices : Finset ℝ)
variables (p_plus : ℝ → ℝ) (star : ℝ → ℝ)

noncomputable def BakeryPrices : Prop :=
  p_plus p1 = p1 + d ∧ p_plus p2 = p2 + d ∧ 
  star p1 = k * p1 ∧ star p2 = k * p2 ∧ 
  prices = {64, 64, 70, 72} ∧ 
  ∃ (q1 q2 q3 q4 : ℝ), 
    (q1 = p_plus p1 ∨ q1 = star p1 ∨ q1 = p_plus p2 ∨ q1 = star p2) ∧
    (q2 = p_plus p1 ∨ q2 = star p1 ∨ q2 = p_plus p2 ∨ q2 = star p2) ∧
    (q3 = p_plus p1 ∨ q3 = star p1 ∨ q3 = p_plus p2 ∨ q3 = star p2) ∧
    (q4 = p_plus p1 ∨ q4 = star p1 ∨ q4 = p_plus p2 ∨ q4 = star p2) ∧
    prices = {q1, q2, q3, q4}

theorem determine_wholesale_prices (p1 p2 : ℝ) (d k : ℝ)
    (h1 : 1 < k) (prices : Finset ℝ) :
  BakeryPrices p1 p2 d k h1 prices (λ p, p + d) (λ p, k * p) :=
sorry

end determine_wholesale_prices_l776_776853


namespace pie_prices_l776_776879

theorem pie_prices 
  (d k : ℕ) (hd : 0 < d) (hk : 1 < k) 
  (p1 p2 : ℕ) 
  (prices : List ℕ) (h_prices : prices = [64, 64, 70, 72]) 
  (hP1 : (p1 + d) ∈ prices)
  (hZ1 : (k * p1) ∈ prices)
  (hP2 : (p2 + d) ∈ prices)
  (hZ2 : (k * p2) ∈ prices)
  (h_different_prices : p1 + d ≠ k * p1 ∧ p2 + d ≠ k * p2) : 
  ∃ p1 p2, p1 ≠ p2 :=
sorry

end pie_prices_l776_776879


namespace cube_root_approx_l776_776911
noncomputable def seq (A : ℝ) : ℕ → ℝ
| 0       := A
| (n + 1) := (seq A n + A / (seq A n) ^ 2) / 2

theorem cube_root_approx (A : ℝ) :
  ∃ (l : ℝ), l = real.cbrt A ∧ 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |seq A n - l| < ε :=
sorry

end cube_root_approx_l776_776911


namespace min_value_of_exponential_sum_l776_776575

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 2) : 2^a + 2^b ≥ 4 := 
by
  sorry

end min_value_of_exponential_sum_l776_776575


namespace distribution_problem_l776_776052

theorem distribution_problem (cards friends : ℕ) (h1 : cards = 7) (h2 : friends = 9) :
  (Nat.choose friends cards) * (Nat.factorial cards) = 181440 :=
by
  -- According to the combination formula and factorial definition
  -- We can insert specific values and calculations here, but as per the task requirements, 
  -- we are skipping the actual proof.
  sorry

end distribution_problem_l776_776052


namespace largest_coefficient_in_expansion_binomial_coefficient_sum_l776_776288

-- Define the function
def f (x : ℂ) (n : ℕ) : ℂ := (1 + x) ^ n

-- Define binomial coefficient function
noncomputable def binom (n k : ℕ) : ℕ := nat.choose n k

-- First theorem statement: Largest term in the expansion of f(x, 6)
theorem largest_coefficient_in_expansion :
  (f x 6).coeff 3 = 20 :=
sorry

-- Second theorem statement: Binomial coefficient sum for n=10
theorem binomial_coefficient_sum :
  (binom 10 1) - (binom 10 3) + (binom 10 5) - (binom 10 7) + (binom 10 9) = 32 :=
sorry

end largest_coefficient_in_expansion_binomial_coefficient_sum_l776_776288


namespace smallest_number_groups_l776_776372

theorem smallest_number_groups :
  ∃ x : ℕ, (∀ y : ℕ, (y % 12 = 0 ∧ y % 20 = 0 ∧ y % 6 = 0) → y ≥ x) ∧ 
           (x % 12 = 0 ∧ x % 20 = 0 ∧ x % 6 = 0) ∧ x = 60 :=
by
  sorry

end smallest_number_groups_l776_776372


namespace max_segments_l776_776302

-- Define given conditions and required proof

-- Let n be a positive integer
variable (n : ℕ) (h_pos : n > 0)

-- Maximum number of segments drawn within 4n points on a plane
-- such that no three points are collinear
theorem max_segments (h: no_three_collinear (4*n)) (d : distance_between_points_is_r (4*n) r) :
  max_possible_segments (4*n) = 1003 := 
sorry

end max_segments_l776_776302


namespace graph_symmetry_correct_l776_776430

theorem graph_symmetry_correct :
  (∀ x : ℝ, y = 3 ^ x) → 
  (∀ x : ℝ, y = -3 ^ -x) →
  (∀ x : ℝ, (3 ^ x, -3 ^ -x) = (0, 0)) :=
begin
  sorry
end

end graph_symmetry_correct_l776_776430


namespace mushrooms_gigi_cut_l776_776123

-- Definitions based on conditions in part a)
def pieces_per_mushroom := 4
def kenny_sprinkled := 38
def karla_sprinkled := 42
def pieces_remaining := 8

-- The total number of pieces is the sum of Kenny's, Karla's, and the remaining pieces.
def total_pieces := kenny_sprinkled + karla_sprinkled + pieces_remaining

-- The number of mushrooms GiGi cut up at the beginning is total_pieces divided by pieces_per_mushroom.
def mushrooms_cut := total_pieces / pieces_per_mushroom

-- The theorem to be proved.
theorem mushrooms_gigi_cut (h1 : pieces_per_mushroom = 4)
                           (h2 : kenny_sprinkled = 38)
                           (h3 : karla_sprinkled = 42)
                           (h4 : pieces_remaining = 8)
                           (h5 : total_pieces = kenny_sprinkled + karla_sprinkled + pieces_remaining)
                           (h6 : mushrooms_cut = total_pieces / pieces_per_mushroom) :
  mushrooms_cut = 22 :=
by
  sorry

end mushrooms_gigi_cut_l776_776123


namespace expected_occurrences_of_A_l776_776209

-- Definitions based on the conditions in the problem
def n : ℕ := 100
def p : ℝ := 0.5
def q : ℝ := 1 - p
def P_n_k : ℝ := 0.0484

-- Express the normal approximation conditions
def x (k : ℝ) : ℝ := (k - n * p) / (Real.sqrt (n * p * q))
def φ (x : ℝ) : ℝ := 0.242 -- This would typically be the standard normal CDF inverse, but we'll simplify for this problem

-- The proof statement
theorem expected_occurrences_of_A : 
  ∃ (k : ℝ), (φ (x k)) = P_n_k * (Real.sqrt (n * p * q) / n) ∧ k = 55 :=
begin
  sorry
end

end expected_occurrences_of_A_l776_776209


namespace cars_in_section_H_l776_776680

theorem cars_in_section_H
  (rows_G : ℕ) (cars_per_row_G : ℕ) (rows_H : ℕ)
  (cars_per_minute : ℕ) (minutes_spent : ℕ)  
  (total_cars_walked_past : ℕ) :
  rows_G = 15 →
  cars_per_row_G = 10 →
  rows_H = 20 →
  cars_per_minute = 11 →
  minutes_spent = 30 →
  total_cars_walked_past = (rows_G * cars_per_row_G) + ((cars_per_minute * minutes_spent) - (rows_G * cars_per_row_G)) →
  (total_cars_walked_past - (rows_G * cars_per_row_G)) / rows_H = 9 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end cars_in_section_H_l776_776680


namespace perimeter_of_triangle_l776_776398

theorem perimeter_of_triangle (x : ℝ) (hx : 7 < x ∧ x < 13) : 
  20 < 10 + 3 + x ∧ 10 + 3 + x < 26 :=
by {
  have h₁ : 10 + 3 = 13 := rfl,
  have h₂ : 10 - 3 = 7 := rfl,
  have : 7 < x := hx.left,
  have : x < 13 := hx.right,
  have h₃ : 20 < 13 + x, from add_lt_add_left hx.left 13,
  have h₄ : 13 + x < 26, from add_lt_add_right hx.right 13,
  exact ⟨h₃, h₄⟩,
 }

end perimeter_of_triangle_l776_776398


namespace compute_binomial_10_3_eq_120_l776_776484

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l776_776484


namespace maximize_x_l776_776569

theorem maximize_x (x y : ℝ) (h : x^2 + 3 * y^2 = 1) (hx_y_max := x + y) :
    x = sqrt(3) / 2 :=
begin
  sorry
end

end maximize_x_l776_776569


namespace sum_cubes_mod_7_l776_776542

theorem sum_cubes_mod_7 :
  (∑ i in Finset.range 151, i ^ 3) % 7 = 0 := by
  sorry

end sum_cubes_mod_7_l776_776542


namespace total_bike_clamps_given_away_l776_776395

-- Definitions for conditions
def bike_clamps_per_bike := 2
def bikes_sold_morning := 19
def bikes_sold_afternoon := 27

-- Theorem statement to be proven
theorem total_bike_clamps_given_away :
  bike_clamps_per_bike * bikes_sold_morning +
  bike_clamps_per_bike * bikes_sold_afternoon = 92 :=
by
  sorry -- Proof is to be filled in later

end total_bike_clamps_given_away_l776_776395


namespace sequence_sum_l776_776132

theorem sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
    (h1 : a 1 = 1)
    (h_rec : ∀ n, a (n + 2) = 1 / (a n + 1))
    (h6_2 : a 6 = a 2) :
    a 2016 + a 3 = (Real.sqrt 5) / 2 :=
by
  sorry

end sequence_sum_l776_776132


namespace problem1_problem2_problem3_problem4_l776_776394

theorem problem1 (α : ℝ) (h₁ : Real.sin α > 0) (h₂ : Real.tan α > 0) :
  α ∈ { x : ℝ | x >= 0 ∧ x < π/2 } := sorry

theorem problem2 (α : ℝ) (h₁ : Real.tan α * Real.sin α < 0) :
  α ∈ { x : ℝ | (x > π/2 ∧ x < π) ∨ (x > π ∧ x < 3 * π / 2) } := sorry

theorem problem3 (α : ℝ) (h₁ : Real.sin α * Real.cos α < 0) :
  α ∈ { x : ℝ | (x > π/2 ∧ x < π) ∨ (x > 3 * π / 2 ∧ x < 2 * π) } := sorry

theorem problem4 (α : ℝ) (h₁ : Real.cos α * Real.tan α > 0) :
  α ∈ { x : ℝ | x >= 0 ∧ x < π ∨ x > π ∧ x < 3 * π / 2 } := sorry

end problem1_problem2_problem3_problem4_l776_776394


namespace koschei_stopped_november_2023_l776_776681

theorem koschei_stopped_november_2023:
  (M : ℕ) (H1: M = 77 * 120) 
  (H2 : ∃ (m : ℕ), 12 * m = 9240 / 5) 
  (H3 : 9240 = 5 * (12 * M)): 
  let end_month := ((154 % 12) + 1) in
  let end_year := (154 / 12 + 2010) in
  end_month = 11 ∧ end_year = 2023 := 
  by
    sorry

end koschei_stopped_november_2023_l776_776681


namespace sqrt_expression_evaluation_l776_776444

def sqrt (x : ℝ) := real.sqrt x
def cbrt (x : ℝ) := x^(1/3 : ℝ)

theorem sqrt_expression_evaluation :
  sqrt 25 - sqrt 3 + abs (sqrt 3 - 2) + cbrt (-8) = 5 - 2 * sqrt 3 :=
by
  -- Define the constants involved in the expression
  let sqrt25 := sqrt 25
  let sqrt3 := sqrt 3
  let abs_term := abs (sqrt3 - 2)
  let cbrt_neg8 := cbrt (-8)
  
  -- Sorry placeholder for the proof steps
  sorry

end sqrt_expression_evaluation_l776_776444


namespace number_of_classes_l776_776263

theorem number_of_classes
  (p : ℕ) (s : ℕ) (t : ℕ) (c : ℕ)
  (hp : p = 2) (hs : s = 30) (ht : t = 360) :
  c = t / (p * s) :=
by
  simp [hp, hs, ht]
  sorry

end number_of_classes_l776_776263


namespace probability_all_genuine_l776_776518

-- Definitions used directly in the conditions
def p_coins {C: Type} (n m: ℕ) [fintype C] [decidable_eq C] (is_genuine: C → Prop) (genuine_weight: C → ℝ) (counterfeit_weight: C → ℝ → Prop) : ℝ :=
  (15/18 : ℝ) * (14/17) * (13/16) * (12/15) * (11/14) * (10/13)

theorem probability_all_genuine (C: Type) [fintype C] [decidable_eq C] 
  (genuine: set C) (counterfeit: set C) (genuine_weight: C → ℝ) (counterfeit_weight: C → ℝ → Prop) :
  ∀ (n: ℕ) (m: ℕ) (is_genuine: C → Prop),
  (card genuine = 15) →
  (card counterfeit = 3) →
  (∀ c ∈ genuine, ∀ cw ∈ counterfeit_weight c, genuine_weight c ≠ cw) →
  let pB := p_coins n m is_genuine genuine_weight counterfeit_weight in
  pB = 55/204 →
  (55/204) / (55/204) = 1 :=
by
  sorry

end probability_all_genuine_l776_776518


namespace eccentricity_range_l776_776993

-- Define the ellipse C1 and hyperbola C2 sharing the same foci
noncomputable def ellipse_C1 (x y m n : ℝ) : Prop := (x^2) / (m + 2) - (y^2) / (-n) = 1
noncomputable def hyperbola_C2 (x y m n : ℝ) : Prop := (x^2) / m + (y^2) / (-n) = 1

-- Define the eccentricity of the ellipse C1
noncomputable def eccentricity_C1 (m n : ℝ) : ℝ := real.sqrt (1 - (1 / (m + 2)))

-- Main theorem statement
theorem eccentricity_range (m : ℝ) (h_m : m > 0) :
  ∀ e1 : ℝ, (∃ x y n : ℝ, ellipse_C1 x y m n ∧ hyperbola_C2 x y m n ∧ n = -1 ∧ e1 = eccentricity_C1 m n) → 
  real.sqrt 2 / 2 < e1 ∧ e1 < 1 :=
sorry

end eccentricity_range_l776_776993


namespace acute_dihedral_angle_l776_776223

noncomputable def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem acute_dihedral_angle :
  let n := (1, 0, 1) in
  let ν := (-1, 1, 0) in
  ∃ θ : ℝ, θ = 60 ∧ real.cos θ = (dot_product n ν) / (norm n * norm ν) := by
  sorry

end acute_dihedral_angle_l776_776223


namespace xyz_inequality_l776_776272

theorem xyz_inequality
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (triangle_ineq1 : sqrt x + sqrt y > sqrt z)
  (triangle_ineq2 : sqrt y + sqrt z > sqrt x)
  (triangle_ineq3 : sqrt z + sqrt x > sqrt y)
  (ratio_condition : x / y + y / z + z / x = 5) :
  x * (y^2 - 2 * z^2) / z + y * (z^2 - 2 * x^2) / x + z * (x^2 - 2 * y^2) / y ≥ 0 := 
sorry

end xyz_inequality_l776_776272


namespace find_x_l776_776178

variables {x : ℝ}

def vector_a : ℝ × ℝ := (4, 2)
def vector_b : ℝ × ℝ := (x, 3)

theorem find_x (h : vector_a.1 * vector_b.2 = vector_a.2 * vector_b.1) : x = 6 :=
  by
    sorry

end find_x_l776_776178


namespace red_bordered_area_l776_776412

theorem red_bordered_area (r₁ r₂ : ℝ) (A₁ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 6) (h₃ : A₁ = 37) :
  ∃ A₂ : ℝ, A₂ = 83.25 :=
by {
  -- We use the fact that the area relationships are proportional to the square of radii.
  let k := (r₂ / r₁)^2,
  use k * A₁,
  have h_k : k = (6/4)^2, by simp [h₁, h₂],
  simp [h₃, h_k],
  -- Provided that A₂ indeed equals the calculated value.
  norm_num,
  sorry
}

end red_bordered_area_l776_776412


namespace binomial_coefficient_10_3_l776_776468

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l776_776468


namespace correct_function_period_and_increasing_l776_776431

def f (x : ℝ) : ℝ := sin (x / 2)
def g (x : ℝ) : ℝ := sin x
def h (x : ℝ) : ℝ := -tan x
def k (x : ℝ) : ℝ := -cos (2 * x)

theorem correct_function_period_and_increasing :
  (∀ x, f (x + 4 * π) = f x ∧ ∀ x ∈ Ioo (0 : ℝ) (π / 2), f' x > 0) → False ∧
  (∀ x, g (x + 2 * π) = g x ∧ ∀ x ∈ Ioo (0 : ℝ) (π / 2), g' x > 0) → False ∧
  (∀ x, h (x + π) = h x ∧ ∀ x ∈ Ioo (0 : ℝ) (π / 2), h' x > 0) → False ∧
  (∀ x, k (x + π) = k x ∧ ∀ x ∈ Ioo (0 : ℝ) (π / 2), k' x > 0) → True :=
by sorry

end correct_function_period_and_increasing_l776_776431


namespace equal_acute_angles_l776_776046

variable (r a b α β : ℝ)

-- Conditions
def rhombus_side (r α : ℝ) : ℝ := 2 * r / Real.sin α
def trapezoid_leg (r β : ℝ) : ℝ := 2 * r / Real.sin β

-- Area of rhombus and trapezoid
def area_rhombus (r α : ℝ) : ℝ := (2 * r / Real.sin α) * 2 * r
def area_trapezoid (r β : ℝ) : ℝ := (2 * r / Real.sin β) * 2 * r

-- The proof problem
theorem equal_acute_angles (inscribed_around_the_same_circle : a = rhombus_side r α)
                            (equal_areas : area_rhombus r α = area_trapezoid r β) :
                            α = β :=
by
  sorry

end equal_acute_angles_l776_776046


namespace find_eccentricity_of_ellipse_l776_776986

-- Definitions for the conditions
def ellipse (a b : ℝ) : Prop := 
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

def foci_conditions (a b c : ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  a > b ∧ b > 0 ∧ F1 = (-c, 0) ∧ F2 = (c, 0) ∧ c = real.sqrt (a^2 - b^2)

def line_through_focus (A B F2 : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, A.2 = m * (A.1 - F2.1) ∧ B.2 = m * (B.1 - F2.1)

def lengths_ratio (F1 A B : ℝ × ℝ) : Prop :=
  real.dist F1 B = (2 / 3) * real.dist F1 A

def angles_equality (F1 B A F2 : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, θ = real.angle (F1 - B) (A - B) ∧ θ = real.angle(F1 - F2) (B - F2)

-- Problem statement
theorem find_eccentricity_of_ellipse (a b c : ℝ) (F1 F2 A B : ℝ × ℝ)
  (ha : 0 < a) (hb : 0 < b) (hab : a > b)
  (hc : c = real.sqrt (a^2 - b^2))
  (hfoci : F1 = (-c, 0) ∧ F2 = (c, 0))
  (hellipse : ellipse a b)
  (hline : line_through_focus A B F2)
  (hlength : lengths_ratio F1 A B)
  (hangle : angles_equality F1 B A F2) :
  ∃ e : ℝ, e = (11 - real.sqrt 41) / 10 :=
sorry

end find_eccentricity_of_ellipse_l776_776986


namespace combination_10_3_l776_776450

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l776_776450


namespace range_of_a_l776_776585

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 2 < a ∧ a ≤ 3 :=
by {
  sorry
}

end range_of_a_l776_776585


namespace smallest_consecutive_integers_product_l776_776551

theorem smallest_consecutive_integers_product (n : ℕ) 
  (h : n * (n + 1) * (n + 2) * (n + 3) = 5040) : 
  n = 7 :=
sorry

end smallest_consecutive_integers_product_l776_776551


namespace divisors_of_30_l776_776201

theorem divisors_of_30 : 
  let positive_divisors := {1, 2, 3, 5, 6, 10, 15, 30}
  in 2 * positive_divisors.card = 16 :=
by
  let positive_divisors := {1, 2, 3, 5, 6, 10, 15, 30}
  have h1 : positive_divisors.card = 8 := sorry
  have h2 : 2 * 8 = 16 := sorry
  exact h2

end divisors_of_30_l776_776201


namespace smallest_three_digit_multiple_of_eleven_l776_776764

theorem smallest_three_digit_multiple_of_eleven : ∃ n, n = 110 ∧ 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n := by
  sorry

end smallest_three_digit_multiple_of_eleven_l776_776764


namespace parallel_lines_trans_parallel_planes_trans_l776_776141

variables {a b c : Type} [metric_space a] [metric_space b] [metric_space c]
variables {α β γ : Type} [metric_space α] [metric_space β] [metric_space γ]

-- Given linse and planes are distinct
axiom distinct_lines : a ≠ b ∧ a ≠ c ∧ b ≠ c
axiom distinct_planes : α ≠ β ∧ α ≠ γ ∧ β ≠ γ

-- Definitions for parallelism and perpendicularity
def parallel_lines (l1 l2 : Type) [metric_space l1] [metric_space l2] : Prop := sorry
def parallel_planes (p1 p2 : Type) [metric_space p1] [metric_space p2] : Prop := sorry

theorem parallel_lines_trans {a b c : Type} [metric_space a] [metric_space b] [metric_space c]
  (h : parallel_lines a b) (h' : parallel_lines a c) : parallel_lines b c := sorry

theorem parallel_planes_trans {α β γ : Type} [metric_space α] [metric_space β] [metric_space γ]
  (h : parallel_planes α β) (h' : parallel_planes α γ) : parallel_planes β γ := sorry

end parallel_lines_trans_parallel_planes_trans_l776_776141


namespace trig_fraction_identity_l776_776312

noncomputable def cos_63 := Real.cos (Real.pi * 63 / 180)
noncomputable def cos_3 := Real.cos (Real.pi * 3 / 180)
noncomputable def cos_87 := Real.cos (Real.pi * 87 / 180)
noncomputable def cos_27 := Real.cos (Real.pi * 27 / 180)
noncomputable def cos_132 := Real.cos (Real.pi * 132 / 180)
noncomputable def cos_72 := Real.cos (Real.pi * 72 / 180)
noncomputable def cos_42 := Real.cos (Real.pi * 42 / 180)
noncomputable def cos_18 := Real.cos (Real.pi * 18 / 180)
noncomputable def tan_24 := Real.tan (Real.pi * 24 / 180)

theorem trig_fraction_identity :
  (cos_63 * cos_3 - cos_87 * cos_27) / 
  (cos_132 * cos_72 - cos_42 * cos_18) = 
  -tan_24 := 
by
  sorry

end trig_fraction_identity_l776_776312


namespace ordered_triple_unique_l776_776284

variable (a b c : ℝ)

theorem ordered_triple_unique
  (h_pos_a : a > 4)
  (h_pos_b : b > 4)
  (h_pos_c : c > 4)
  (h_eq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  (a, b, c) = (12, 10, 8) := 
sorry

end ordered_triple_unique_l776_776284


namespace g_of_3_2_add_g_of_3_5_l776_776644

def g (x y : ℝ) : ℝ :=
if x + y ≤ 5 then (x * y - x + 3) / (3 * x)
else (x * y - y - 3) / (-3 * y)

theorem g_of_3_2_add_g_of_3_5 : g 3 2 + g 3 5 = 1 / 5 :=
by
  sorry

end g_of_3_2_add_g_of_3_5_l776_776644


namespace sum_of_cubes_mod_7_l776_776538

theorem sum_of_cubes_mod_7 :
  (∑ k in Finset.range 150, (k + 1) ^ 3) % 7 = 1 := 
sorry

end sum_of_cubes_mod_7_l776_776538


namespace find_other_intersection_point_l776_776730

-- Definitions
def parabola_eq (x : ℝ) : ℝ := x^2 - 2 * x - 3
def intersection_point1 : Prop := parabola_eq (-1) = 0
def intersection_point2 : Prop := parabola_eq 3 = 0

-- Proof problem
theorem find_other_intersection_point :
  intersection_point1 → intersection_point2 := by
  sorry

end find_other_intersection_point_l776_776730


namespace area_and_perimeter_correct_l776_776945

noncomputable def triangle_area_and_perimeter (a b c : ℝ) (ha : a = 10) (hb : b = 12) (hc : c = 12)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  (ℝ × ℝ) :=
let s := (a + b + c) / 2 in
let area := real.sqrt (s * (s - a) * (s - b) * (s - c)) in
let perimeter := a + b + c in
if ha && hb && hc then (5 * real.sqrt(119), 34) else (area, perimeter)

theorem area_and_perimeter_correct : triangle_area_and_perimeter 10 12 12 rfl rfl rfl (by norm_num) (by norm_num) (by norm_num) = (5 * real.sqrt 119, 34) :=
by sorry

end area_and_perimeter_correct_l776_776945


namespace fibboican_power_of_2_numerator_l776_776709

def fibboican_seq : ℕ → ℚ
| 1      := 1
| 2      := 1
| (n+1)  :=
  if (n+1) % 2 = 1 then
    fibboican_seq n + fibboican_seq (n - 1)
  else
    1 / (1 / fibboican_seq n + 1 / fibboican_seq (n - 1))

theorem fibboican_power_of_2_numerator (m : ℕ) (hm : m ≥ 1) :
  ∃ k : ℕ, (fibboican_seq m).numerator = 2^k :=
sorry

end fibboican_power_of_2_numerator_l776_776709


namespace correct_statement_among_A_B_C_D_is_C_l776_776378

theorem correct_statement_among_A_B_C_D_is_C :
  let A : Prop := degree_monomial (-2^3 * x * y^2) = 6
  let B : Prop := is_polynomial (3 / x - 5)
  let C : Prop := is_binomial (2 * x^3 * y + 4 * x) ∧ degree_polynomial (2 * x^3 * y + 4 * x) = 4
  let D : Prop := coeff a^2 * b * c^3 = 0
  ¬A ∧ ¬B ∧ C ∧ ¬D :=
by sorry

end correct_statement_among_A_B_C_D_is_C_l776_776378


namespace power_multiplication_l776_776443

theorem power_multiplication (a : ℝ) (b : ℝ) (m : ℕ) (n : ℕ) (h1 : a = 0.25) (h2 : b = 4) (h3 : m = 2023) (h4 : n = 2024) : 
  a^m * b^n = 4 := 
by 
  sorry

end power_multiplication_l776_776443


namespace car_return_speed_l776_776885

theorem car_return_speed (d : ℕ) (r : ℕ) (h₁ : d = 180) (h₂ : (2 * d) / ((d / 90) + (d / r)) = 60) : r = 45 :=
by
  rw [h₁] at h₂
  have h3 : 2 * 180 / ((180 / 90) + (180 / r)) = 60 := h₂
  -- The rest of the proof involves solving for r, but here we only need the statement
  sorry

end car_return_speed_l776_776885


namespace solve_log_eq_l776_776802

theorem solve_log_eq (x : ℝ) (hx : x > 0) 
  (h : 4^(Real.log x / Real.log 9 * 2) + Real.log 3 / (1/2 * Real.log 3) = 
       0.2 * (4^(2 + Real.log x / Real.log 9) - 4^(Real.log x / Real.log 9))) :
  x = 1 ∨ x = 3 :=
by sorry

end solve_log_eq_l776_776802


namespace wholesale_prices_l776_776833

-- Definitions for the problem conditions
variable (p1 p2 d k : ℝ)
variable (h_d : d > 0)
variable (h_k : k > 1)
variable (prices : Finset ℝ)
variable (h_prices : prices = {64, 64, 70, 72})

-- The theorem statement to prove
theorem wholesale_prices :
  ∃ p1 p2, (p1 + d ∈ prices ∧ k * p1 ∈ prices) ∧ 
           (p2 + d ∈ prices ∧ k * p2 ∈ prices) ∧ 
           p1 ≠ p2
:= sorry

end wholesale_prices_l776_776833


namespace max_value_of_a_l776_776573

variable {a : ℝ}

theorem max_value_of_a (h : a > 0) : 
  (∀ x : ℝ, x > 0 → (2 * x^2 - a * x + a > 0)) ↔ a ≤ 8 := 
sorry

end max_value_of_a_l776_776573


namespace parabola_intersection_l776_776728

theorem parabola_intersection :
  let y := fun x : ℝ => x^2 - 2*x - 3 in
  (∀ x : ℝ, y x = 0 ↔ x = 3 ∨ x = -1) ∧ y (-1) = 0 :=
by
  sorry

end parabola_intersection_l776_776728


namespace Mrs_Hilt_bought_two_cones_l776_776299

def ice_cream_cone_cost : ℕ := 99
def total_spent : ℕ := 198

theorem Mrs_Hilt_bought_two_cones : total_spent / ice_cream_cone_cost = 2 :=
by
  sorry

end Mrs_Hilt_bought_two_cones_l776_776299


namespace meet_again_time_l776_776801

-- Definitions
def circumference := 3000 -- meters
def yeonjeong_speed := 100 -- meters/minute
def donghun_speed := 150 -- meters/minute

-- Theorem: They meet again after 12 minutes
theorem meet_again_time : 
  let combined_speed := yeonjeong_speed + donghun_speed in
  let time_to_meet := circumference / combined_speed in
  time_to_meet = 12 := 
by {
  let combined_speed := yeonjeong_speed + donghun_speed,
  let time_to_meet := circumference / combined_speed,
  /- 
    Here would be the proof steps, which we are not required to write.
  -/
  sorry
}

end meet_again_time_l776_776801


namespace geometric_progression_sum_squares_l776_776071

theorem geometric_progression_sum_squares (a r : ℕ) (a_pos : 0 < a) (r_pos : 0 < r) 
  (h1 : a < 50) (h2 : a * (1 + r + r^2 + r^3) = 120)
  (h3 : ∀ n, n ∈ [a, a*r, a*r^2, a*r^3] → n < 50) :
  ∑ n in ([a, a*r, a*r^2, a*r^3].filter (λ x, ∃ m : ℕ, x = m^2)), id = 0 :=
by 
  sorry

end geometric_progression_sum_squares_l776_776071


namespace smallest_n_exists_sequence_l776_776963

theorem smallest_n_exists_sequence : ∃ (n : ℕ), (∀ (a : ℕ → ℤ),
    a 0 = 0 ∧
    a n = 2008 ∧
    (∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → |a i - a (i - 1)| = i^2) ∧
    (∀ (m : ℕ), m < n → ¬ (∀ (b : ℕ → ℤ), 
        b 0 = 0 ∧
        b m = 2008 ∧
        (∀ (j : ℕ), 1 ≤ j ∧ j ≤ m → |b j - b (j - 1)| = j^2)))
) :=
begin
  use 19,
  sorry
end

end smallest_n_exists_sequence_l776_776963


namespace roosters_per_3_hens_l776_776639

theorem roosters_per_3_hens :
  ∀ (hens roosters chicks total: ℕ), 
  hens = 12 → 
  (∀ h, chicks h = 5) → 
  total = 76 → 
  total = hens + roosters + (hens * 5) → 
  (∃ r, roosters = r ∧ hens / 3 = 4 / r) → 
  r = 1 := 
by
  intros hens roosters chicks total h_hens h_chicks h_total h_relation h_ratio
  sorry

end roosters_per_3_hens_l776_776639


namespace dasha_meeting_sasha_l776_776119

def stripes_on_zebra : ℕ := 360

variables {v : ℝ} -- speed of Masha
def dasha_speed (v : ℝ) : ℝ := 2 * v -- speed of Dasha (twice Masha's speed)

def masha_distance_before_meeting_sasha : ℕ := 180
def total_stripes_met : ℕ := stripes_on_zebra
def relative_speed_masha_sasha (v : ℝ) : ℝ := v + v -- combined speed of Masha and Sasha
def relative_speed_dasha_sasha (v : ℝ) : ℝ := 3 * v -- combined speed of Dasha and Sasha

theorem dasha_meeting_sasha (v : ℝ) (hv : 0 < v) :
  ∃ t' t'', 
  (t'' = 120 / v) ∧ (dasha_speed v * t' = 240) :=
by {
  sorry
}

end dasha_meeting_sasha_l776_776119


namespace angle_BFD_l776_776633

theorem angle_BFD (α : ℝ) (β : ℝ) (θ : ℝ)
  (hα : α = 40)
  (hβ : β = 20)
  (hθ : θ = 77.3) :
  True :=
by
  -- Define angles
  have h_ABF: ∠ABF = α,
    sorry,
  
  have h_ADF: ∠ADF = β,
    sorry,
  
  -- Calculation steps can be skipped as they will be used in the proof
  have h_BFD: ∠BFD = θ,
    sorry,
   
  trivial

end angle_BFD_l776_776633


namespace sequence_formula_k1_sequence_is_arithmetic_sum_condition_l776_776177

-- Problem 1
theorem sequence_formula_k1 (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 1) - (-1)^1 * a n = b n)
  (a1 : a 1 = 1)
  (b_sequence : ∀ n, b n = 2^n) :
  ∀ n, a n = 1/3 * (2^n + (-1)^(n-1)) :=
sorry

-- Problem 2.i
theorem sequence_is_arithmetic (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h4 : ∀ n, a (n + 4) - a n = b n)
  (b_const : ∀ n, b n = 8)
  (a1 : a 1 = 4) (a2 : a 2 = 6) (a3 : a 3 = 8) (a4 : a 4 = 10) :
  ∀ n, a n = 2n + 2 :=
sorry

-- Problem 2.ii
theorem sum_condition (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) (k n : ℕ)
  (h4 : ∀ n, a (n + 4) - a n = b n)
  (b_const : ∀ n, b n = 8)
  (a1 : a 1 = 4) (a2 : a 2 = 6) (a3 : a 3 = 8) (a4 : a 4 = 10)
  (a_arith : ∀ n, a n = 2n + 2)
  (sum_n : S = λ n, n^2 + 3n) :
  (S n + 1)^2 - 3/2 * a n + 33 = k^2 ↔ (n = 10 ∧ k = 131) :=
sorry

end sequence_formula_k1_sequence_is_arithmetic_sum_condition_l776_776177


namespace sara_picked_peaches_l776_776317

def peaches_original : ℕ := 24
def peaches_now : ℕ := 61
def peaches_picked (p_o p_n : ℕ) : ℕ := p_n - p_o

theorem sara_picked_peaches : peaches_picked peaches_original peaches_now = 37 :=
by
  sorry

end sara_picked_peaches_l776_776317


namespace find_g_neg5_l776_776661

-- Define the polynomial function g(x)
def g (x : ℝ) (p : ℝ) : ℝ := 2 * x^7 - 3 * x^5 + p * x^2 + 2 * x - 6

-- Declare the conditions as hypotheses
variable (p : ℝ)
variable h_g5 : g 5 p = 10

-- The proof goal to show
theorem find_g_neg5 : g (-5) p = -301383 := by
  sorry

end find_g_neg5_l776_776661


namespace wholesale_price_is_60_l776_776867

variables (p d k : ℕ)
variables (P1 P2 Z1 Z2 : ℕ)

// Conditions from (a)
axiom H1 : P1 = p + d // Baker Plus price day 1
axiom H2 : P2 = p + d // Baker Plus price day 2
axiom H3 : Z1 = k * p // Star price day 1
axiom H4 : Z2 = k * p // Star price day 2
axiom H5 : k > 1 // Star store markup factor greater than 1
axiom H6 : P1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H7 : P2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H8 : Z1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H9 : Z2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))

theorem wholesale_price_is_60 : p = 60 := 
by {
    // Proof goes here, skipped with sorry
    sorry
}

end wholesale_price_is_60_l776_776867


namespace three_a_minus_two_d_l776_776591

-- Define the function f(x)
def f (a b c d x : ℝ) : ℝ := (2 * a * x - b) / (3 * c * x + d)

-- Conditions: b ≠ 0, d ≠ 0, and abcd ≠ 0, and f(f(x)) = x
variable (a b c d : ℝ)
variable (h₀ : b ≠ 0)
variable (h₁ : d ≠ 0)
variable (h₂ : a * b * c * d ≠ 0)
variable (h₃ : ∀ x, f a b c d (f a b c d x) = x)

-- Theorem: 3a - 2d = -4.5c - 4b
theorem three_a_minus_two_d : 3 * a - 2 * d = -4.5 * c - 4 * b :=
by
  sorry

end three_a_minus_two_d_l776_776591


namespace least_positive_three_digit_multiple_of_11_l776_776760

theorem least_positive_three_digit_multiple_of_11 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n ∧ n = 110 :=
by {
  use 110,
  split,
  { exact Nat.le_refl _ },
  split,
  { norm_num },
  split,
  { use 10,
    norm_num },
  { refl },
  sorry
}

end least_positive_three_digit_multiple_of_11_l776_776760


namespace least_positive_three_digit_multiple_of_13_is_104_l776_776783

theorem least_positive_three_digit_multiple_of_13_is_104 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ n = 104 :=
by
  existsi 104
  split
  · show 100 ≤ 104
    exact le_refl 104
  split
  · show 104 < 1000
    exact dec_trivial
  split
  · show 104 % 13 = 0
    exact dec_trivial
  · show 104 = 104
    exact rfl

end least_positive_three_digit_multiple_of_13_is_104_l776_776783


namespace radishes_times_carrots_l776_776739

theorem radishes_times_carrots (cucumbers radishes carrots : ℕ) 
  (h1 : cucumbers = 15) 
  (h2 : radishes = 3 * cucumbers) 
  (h3 : carrots = 9) : 
  radishes / carrots = 5 :=
by
  sorry

end radishes_times_carrots_l776_776739


namespace ball_radius_l776_776880

theorem ball_radius (self_hole_diameter cm)  : 
(self_hole_depth ) : 
(self_ball_radius (self_hole_diameter  / 2) == self_hole_depth):=
 (ball_radius : 16.25) := ball_radius

proof ball_radius :

r,in

self_hole_diameter  :
function result
hole_radius := self_hole_diameter  / 2 = 15  :
centre_dist := (self_ball_radius - center_distance = self_hole_radius^2)

begin

x^2 + 15^2 = (x + self_hole_depth )^2 ,
 
result = r = 16.25.

end ball_radius_l776_776880


namespace binom_10_3_l776_776459

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l776_776459


namespace problem1_problem2_l776_776687

-- Definition for the first problem: determine the number of arrangements when no box is empty and ball 3 is in box B
def arrangements_with_ball3_in_B_and_no_empty_box : ℕ :=
  12

theorem problem1 : arrangements_with_ball3_in_B_and_no_empty_box = 12 :=
  by
    sorry

-- Definition for the second problem: determine the number of arrangements when ball 1 is not in box A and ball 2 is not in box B
def arrangements_with_ball1_not_in_A_and_ball2_not_in_B : ℕ :=
  36

theorem problem2 : arrangements_with_ball1_not_in_A_and_ball2_not_in_B = 36 :=
  by
    sorry

end problem1_problem2_l776_776687


namespace find_wholesale_prices_l776_776844

-- Definitions based on conditions
def bakerPlusPrice (p : ℕ) (d : ℕ) : ℕ := p + d
def starPrice (p : ℕ) (k : ℚ) : ℕ := (p * k).toNat

-- The tuple of prices given
def prices : List ℕ := [64, 64, 70, 72]

-- Statement of the problem
theorem find_wholesale_prices (d : ℕ) (k : ℚ) (h_d : d > 0) (h_k : k > 1 ∧ k.denom = 1):
  ∃ p1 p2, 
  (p1 ∈ prices ∧ bakerPlusPrice p1 d ∈ prices ∧ starPrice p1 k ∈ prices) ∧
  (p2 ∈ prices ∧ bakerPlusPrice p2 d ∈ prices ∧ starPrice p2 k ∈ prices) :=
sorry

end find_wholesale_prices_l776_776844


namespace length_of_goods_train_l776_776006

theorem length_of_goods_train
  (speed_kmph : ℕ)
  (length_of_platform : ℕ)
  (time_to_cross_platform : ℕ)
  (conversion_factor : ℚ)
  (speed_mspers: ℚ)
  (distance_covered: ℕ) :
  speed_kmph = 72 ->
  length_of_platform = 150 ->
  time_to_cross_platform = 26 ->
  conversion_factor = 5 / 18 ->
  speed_mspers = speed_kmph * conversion_factor ->
  distance_covered = speed_mspers * time_to_cross_platform →
  nat.sub distance_covered length_of_platform = 370 :=
by
  intros
  sorry

end length_of_goods_train_l776_776006


namespace constant_term_of_binomial_expansion_l776_776546

theorem constant_term_of_binomial_expansion : 
  let x := (x : ℚ) in
  (x + 1/x)^4 = 6 := sorry

end constant_term_of_binomial_expansion_l776_776546


namespace divisors_of_30_l776_776204

theorem divisors_of_30 : 
  let positive_divisors := {1, 2, 3, 5, 6, 10, 15, 30}
  in 2 * positive_divisors.card = 16 :=
by
  let positive_divisors := {1, 2, 3, 5, 6, 10, 15, 30}
  have h1 : positive_divisors.card = 8 := sorry
  have h2 : 2 * 8 = 16 := sorry
  exact h2

end divisors_of_30_l776_776204


namespace find_divisor_l776_776684

theorem find_divisor (d : ℕ) (H1 : 199 = d * 11 + 1) : d = 18 := 
sorry

end find_divisor_l776_776684


namespace school_outing_boats_l776_776732

theorem school_outing_boats (x : ℕ):
  (3 * x + 16 = 5 * (x - 1) + 1) -> (3 * x + 16 = 46) :=
by 
  intros h
  apply Nat.add_right_cancel with (n := 1)
  exact h

end school_outing_boats_l776_776732


namespace imaginary_part_of_i_mul_1_plus_i_l776_776216

noncomputable def i : ℂ := Complex.I

theorem imaginary_part_of_i_mul_1_plus_i :
  Complex.imag_part (i * (1 + i)) = 1 :=
by
  sorry

end imaginary_part_of_i_mul_1_plus_i_l776_776216


namespace wholesale_prices_l776_776832

-- Definitions for the problem conditions
variable (p1 p2 d k : ℝ)
variable (h_d : d > 0)
variable (h_k : k > 1)
variable (prices : Finset ℝ)
variable (h_prices : prices = {64, 64, 70, 72})

-- The theorem statement to prove
theorem wholesale_prices :
  ∃ p1 p2, (p1 + d ∈ prices ∧ k * p1 ∈ prices) ∧ 
           (p2 + d ∈ prices ∧ k * p2 ∈ prices) ∧ 
           p1 ≠ p2
:= sorry

end wholesale_prices_l776_776832


namespace gigi_mushrooms_l776_776120

-- Define the conditions
def pieces_per_mushroom := 4
def kenny_pieces := 38
def karla_pieces := 42
def remaining_pieces := 8

-- Main theorem
theorem gigi_mushrooms : (kenny_pieces + karla_pieces + remaining_pieces) / pieces_per_mushroom = 22 :=
by
  sorry

end gigi_mushrooms_l776_776120


namespace notable_points_of_triangle_l776_776925

-- Definitions of midpoints, perpendicular bisectors, medians, and notable points
structure Triangle :=
  (A B C : Point)

structure Point :=
  (x y : ℝ)

def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

def perpendicular_bisector (P Q : Point) : Line :=
  sorry

def median (T : Triangle) (M : Point) : Line :=
  sorry

def angle_bisector (T : Triangle) (M : Point) : Line :=
  sorry

def intersection (L1 L2 : Line) : Point :=
  sorry

-- Notable points using the midpoints
theorem notable_points_of_triangle (T : Triangle) (M_AB M_BC M_CA : Point)
 (H_AB : M_AB = midpoint T.A T.B)
 (H_BC : M_BC = midpoint T.B T.C)
 (H_CA : M_CA = midpoint T.C T.A) :
 ∃ (O I G H : Point),
   O = intersection (perpendicular_bisector T.AB) (perpendicular_bisector T.BC) ∧
   I = intersection (angle_bisector T M_AB) (angle_bisector T M_BC) ∧
   G = intersection (median T M_AB) (median T M_BC) ∧
   H = intersection (altitude T A) (altitude T B) :=
by sorry

end notable_points_of_triangle_l776_776925


namespace janet_clarinet_hours_l776_776261

theorem janet_clarinet_hours 
  (C : ℕ)  -- number of clarinet lessons hours per week
  (clarinet_cost_per_hour : ℕ := 40)
  (piano_cost_per_hour : ℕ := 28)
  (hours_of_piano_per_week : ℕ := 5)
  (annual_extra_piano_cost : ℕ := 1040) :
  52 * (piano_cost_per_hour * hours_of_piano_per_week - clarinet_cost_per_hour * C) = annual_extra_piano_cost → 
  C = 3 :=
by
  sorry

end janet_clarinet_hours_l776_776261


namespace compute_binomial_10_3_eq_120_l776_776477

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l776_776477


namespace count_n_with_consecutive_roots_and_m_div_by_4_l776_776206

theorem count_n_with_consecutive_roots_and_m_div_by_4 :
  let n_count := ∑ k in Finset.range 75, if (k % 4 = 0 ∨ (k + 3) % 4 = 0) then 1 else 0
  in n_count = 38 := by
  sorry

end count_n_with_consecutive_roots_and_m_div_by_4_l776_776206


namespace ellipse_focus_m_eq_3_l776_776135

theorem ellipse_focus_m_eq_3 (m : ℝ) (h : m > 0) : 
  (∃ a c : ℝ, a = 5 ∧ c = 4 ∧ c^2 = a^2 - m^2)
  → m = 3 :=
by
  sorry

end ellipse_focus_m_eq_3_l776_776135


namespace solve_real_triples_l776_776529

theorem solve_real_triples (a b c : ℝ) :
  (a * (b^2 + c) = c * (c + a * b) ∧
   b * (c^2 + a) = a * (a + b * c) ∧
   c * (a^2 + b) = b * (b + c * a)) ↔ 
  (∃ (x : ℝ), (a = x) ∧ (b = x) ∧ (c = x)) ∨ 
  (b = 0 ∧ c = 0) :=
sorry

end solve_real_triples_l776_776529


namespace exists_n_for_perfect_square_l776_776698

theorem exists_n_for_perfect_square (k : ℕ) (hk_pos : k > 0) :
  ∃ n : ℕ, n > 0 ∧ ∃ a : ℕ, a^2 = n * 2^k - 7 :=
by
  sorry

end exists_n_for_perfect_square_l776_776698


namespace combination_10_3_l776_776453

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l776_776453


namespace find_a_div_b_l776_776279

theorem find_a_div_b (a b : ℝ) (h_distinct : a ≠ b) 
  (h_eq : a / b + (a + 5 * b) / (b + 5 * a) = 2) : a / b = 0.6 :=
by
  sorry

end find_a_div_b_l776_776279


namespace father_overtakes_son_l776_776941

noncomputable def prove_overtake : Prop :=
  ∀ (x y t : ℝ),
    (7 * x = 4 * y) →
    (5 * t * x + 50 = 6 * t * y) →
    (5 * t * x + 50 < 100)

theorem father_overtakes_son : prove_overtake :=
by {
  intros x y t h1 h2,
  have h3 : 6 * t * y = 6 * t * (7/4) * x := by sorry,
  have h4 : 5 * t * x + 50 < 100 := by sorry,
  exact h4,
}

end father_overtakes_son_l776_776941


namespace compute_binomial_10_3_eq_120_l776_776476

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l776_776476


namespace identify_power_functions_l776_776796

-- Define each function
def f1 (x : ℝ) : ℝ := x^2 + 1
def f2 (x : ℝ) : ℝ := 2^x
def f3 (x : ℝ) : ℝ := 1 / x^2
def f4 (x : ℝ) : ℝ := (x - 1)^2
def f5 (x : ℝ) : ℝ := x^5
def f6 (x : ℝ) : ℝ := x^(x + 1)

-- Define what it means for a function to be a power function
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (α : ℝ), ∀ (x : ℝ), f x = x^α

-- Statement of the problem
theorem identify_power_functions :
  {f3, f5} = {f | is_power_function f} :=
sorry

end identify_power_functions_l776_776796


namespace acute_triangle_sine_cosine_inequality_l776_776657

theorem acute_triangle_sine_cosine_inequality (A B C : ℝ) 
  (hA : 0 < A ∧ A < π / 2) 
  (hB : 0 < B ∧ B < π / 2) 
  (hC : 0 < C ∧ C < π / 2)
  (h_sum : A + B + C = π) : 
  sin A + sin B > cos A + cos B + cos C :=
sorry

end acute_triangle_sine_cosine_inequality_l776_776657


namespace add_and_round_58_29_l776_776706

def add_and_round_to_nearest_ten (a b : ℕ) : ℕ :=
  let sum := a + b
  let rounded_sum := if sum % 10 < 5 then sum - (sum % 10) else sum + (10 - sum % 10)
  rounded_sum

theorem add_and_round_58_29 : add_and_round_to_nearest_ten 58 29 = 90 := by
  sorry

end add_and_round_58_29_l776_776706


namespace two_colonies_reach_limit_l776_776030

noncomputable def bacteria_growth (n : ℕ) : ℕ := 2^n

theorem two_colonies_reach_limit (days : ℕ) (h : bacteria_growth days = (2^20)) : 
  bacteria_growth days = bacteria_growth 20 := 
by sorry

end two_colonies_reach_limit_l776_776030


namespace positive_sum_sets_l776_776808

theorem positive_sum_sets (n : ℕ) (a : Fin 2n.succ → ℝ) 
  (h_pos_sum : (∑ i, a i) > 0) : 
  ∃ i, (∑ j in Finset.range n, a ((i + j) % (2 * n))) > 0 ∧ (∑ j in Finset.range n, a ((i + j + n) % (2 * n))) > 0 := by
  sorry

end positive_sum_sets_l776_776808


namespace line_intersects_y_axis_at_0_3_l776_776058

def point := (ℝ × ℝ)

def line_through (p1 p2 : point) (x : ℝ) : ℝ :=
  let m := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b := p1.2 - m * p1.1
  m * x + b

theorem line_intersects_y_axis_at_0_3 : 
  ∃ y : ℝ, line_through (2, 9) (4, 15) 0 = y ∧ y = 3 :=
by
  use 3
  sorry

end line_intersects_y_axis_at_0_3_l776_776058


namespace find_parabola_and_area_l776_776169

noncomputable def parabola_equation (p : ℝ) (hp : 0 < p) (x y : ℝ) : Prop :=
y^2 = 2 * p * x

def point_on_parabola (p x0 y0 : ℝ) (hp : 0 < p) : Prop :=
parabola_equation p hp x0 y0

def distance_condition (x0 y0 : ℝ) : Prop :=
(x0 - 2) ^ 2 + y0 ^ 2 = 3

def circle (x y : ℝ) : Prop :=
(x - 1) ^ 2 + y ^ 2 = 1

def tangents_intersect_y_axis (M : ℝ × ℝ) (a b : ℝ) : Prop :=
∃ k_ma : ℝ, let (x0, y0) := M in
  k_ma = (y0 - a) / x0 ∧
  let line_ma := λ x, k_ma * x + a in
  line_ma 0 = a ∧ ∃ k_mb : ℝ,
    k_mb = (y0 - b) / x0 ∧
    let line_mb := λ x, k_mb * x + b in
    line_mb 0 = b

theorem find_parabola_and_area (
  (p : ℝ) (hp : 0 < p) 
  (x0 y0 : ℝ) 
  (hx0_gt_2 : 2 < x0) (hM_on_C : point_on_parabola p x0 y0 hp)
  (h_dist : distance_condition x0 y0) 
  (ha : (0 : ℝ)) (hb : (0 : ℝ)) 
  (h_tangents : tangents_intersect_y_axis (x0, y0) a b)
) :
  parabola_equation p hp x0 y0 → 
  (y0^2 = 2 * x0) → 
  ∑ (area_min : ℝ), area_min = 8 := sorry

end find_parabola_and_area_l776_776169


namespace binomial_coefficient_10_3_l776_776472

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l776_776472


namespace exist_one_common_ball_l776_776983

theorem exist_one_common_ball (n : ℕ) (h_n : 5 ≤ n) (A : Fin (n+1) → Finset (Fin n))
  (hA_card : ∀ i, (A i).card = 3)
  (h_distinct : ∀ i j, i ≠ j → A i ≠ A j) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (A i ∩ A j).card = 1 :=
sorry

end exist_one_common_ball_l776_776983


namespace sec_seven_pi_div_four_l776_776524

noncomputable def sec (x : ℝ) := 1 / real.cos x

theorem sec_seven_pi_div_four : sec (7 * real.pi / 4) = real.sqrt 2 :=
by sorry

end sec_seven_pi_div_four_l776_776524


namespace determine_wholesale_prices_l776_776849

variables (p1 p2 d k : ℝ)
variables (h1 : 1 < k)
variables (prices : Finset ℝ)
variables (p_plus : ℝ → ℝ) (star : ℝ → ℝ)

noncomputable def BakeryPrices : Prop :=
  p_plus p1 = p1 + d ∧ p_plus p2 = p2 + d ∧ 
  star p1 = k * p1 ∧ star p2 = k * p2 ∧ 
  prices = {64, 64, 70, 72} ∧ 
  ∃ (q1 q2 q3 q4 : ℝ), 
    (q1 = p_plus p1 ∨ q1 = star p1 ∨ q1 = p_plus p2 ∨ q1 = star p2) ∧
    (q2 = p_plus p1 ∨ q2 = star p1 ∨ q2 = p_plus p2 ∨ q2 = star p2) ∧
    (q3 = p_plus p1 ∨ q3 = star p1 ∨ q3 = p_plus p2 ∨ q3 = star p2) ∧
    (q4 = p_plus p1 ∨ q4 = star p1 ∨ q4 = p_plus p2 ∨ q4 = star p2) ∧
    prices = {q1, q2, q3, q4}

theorem determine_wholesale_prices (p1 p2 : ℝ) (d k : ℝ)
    (h1 : 1 < k) (prices : Finset ℝ) :
  BakeryPrices p1 p2 d k h1 prices (λ p, p + d) (λ p, k * p) :=
sorry

end determine_wholesale_prices_l776_776849


namespace pie_wholesale_price_l776_776816

theorem pie_wholesale_price (p1 p2: ℕ) (d: ℕ) (k: ℕ) (h1: d > 0) (h2: k > 1)
  (price_list: List ℕ)
  (p_plus_1: price_list = [p1 + d, k * p1, p2 + d, k * p2]) 
  (h3: price_list.perm [64, 64, 70, 72]) : 
  p1 = 60 ∧ p2 = 60 := 
sorry

end pie_wholesale_price_l776_776816


namespace determine_wholesale_prices_l776_776854

variables (p1 p2 d k : ℝ)
variables (h1 : 1 < k)
variables (prices : Finset ℝ)
variables (p_plus : ℝ → ℝ) (star : ℝ → ℝ)

noncomputable def BakeryPrices : Prop :=
  p_plus p1 = p1 + d ∧ p_plus p2 = p2 + d ∧ 
  star p1 = k * p1 ∧ star p2 = k * p2 ∧ 
  prices = {64, 64, 70, 72} ∧ 
  ∃ (q1 q2 q3 q4 : ℝ), 
    (q1 = p_plus p1 ∨ q1 = star p1 ∨ q1 = p_plus p2 ∨ q1 = star p2) ∧
    (q2 = p_plus p1 ∨ q2 = star p1 ∨ q2 = p_plus p2 ∨ q2 = star p2) ∧
    (q3 = p_plus p1 ∨ q3 = star p1 ∨ q3 = p_plus p2 ∨ q3 = star p2) ∧
    (q4 = p_plus p1 ∨ q4 = star p1 ∨ q4 = p_plus p2 ∨ q4 = star p2) ∧
    prices = {q1, q2, q3, q4}

theorem determine_wholesale_prices (p1 p2 : ℝ) (d k : ℝ)
    (h1 : 1 < k) (prices : Finset ℝ) :
  BakeryPrices p1 p2 d k h1 prices (λ p, p + d) (λ p, k * p) :=
sorry

end determine_wholesale_prices_l776_776854


namespace find_k_l776_776292

open Real

def line_m (x : ℝ) : ℝ := 4 * x + 2
def line_n (k x : ℝ) : ℝ := k * x + 3

theorem find_k : ∃ k : ℝ, line_n k 1 = 6 :=
by
  use 3
  calc
    line_n 3 1 = 3 * 1 + 3 := by rfl
           ... = 6         := by norm_num

end find_k_l776_776292


namespace divisors_of_30_l776_776184

theorem divisors_of_30 : 
  ∃ n : ℕ, (∀ d : ℤ, d ∣ 30 → d > 0 → d ≤ 30) ∧ (∀ d : ℤ, d ∣ 30 → -d ∣ 30) ∧ (∀ d : ℤ, d ∣ 30 → -d = d → d = 0) ∧ 2 * (Finset.card (Finset.filter (λ d, 0 < d) (Finset.filter (λ d, d ∣ 30) (Finset.Icc 1 30).toFinset))) = 16 :=
by {
   sorry
}

end divisors_of_30_l776_776184


namespace max_val_of_m_l776_776281

noncomputable def maximum_min_difference (n : ℕ) (h : n ≥ 3) (a : Fin n → ℝ) 
  (hsum : ∑ i, a i ^ 2 = 1) : ℝ := 
  Real.sqrt (12 / (n * (n ^ 2 - 1)))

theorem max_val_of_m {n : ℕ} (h3 : n ≥ 3) (a : Fin n → ℝ) 
  (hsum : ∑ i, a i ^ 2 = 1) : 
  ∃ m, (∀ (i j : Fin n), i ≠ j → m ≤ abs (a i - a j)) ∧ 
    m = Real.sqrt (12 / (n * (n ^ 2 - 1))) := by
  use maximum_min_difference n h3 a hsum
  sorry

end max_val_of_m_l776_776281


namespace angle_of_inclination_l776_776096

theorem angle_of_inclination (α : ℝ) :
  (∃ m : ℝ, (∀ x y : ℝ, y = -sqrt 3 * x - 3) ∧ m = -√3 ∧ m = tan α) →
  α = 120 :=
by
  sorry

end angle_of_inclination_l776_776096


namespace number_of_valid_subsets_l776_776608

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def subset_is_prime (s : Finset ℕ) : Prop := ∀ n ∈ s, is_prime n

noncomputable def prime_subsets :=
  ((Finset.powerset (Finset.filter is_prime (Finset.range 10))).filter (λ s, 0 < s.card))

theorem number_of_valid_subsets : 
  prime_subsets.filter (λ s, (Finset.sum s id) > 10).card = 4 :=  
by 
  sorry

end number_of_valid_subsets_l776_776608


namespace sec_seven_pi_div_four_l776_776523

noncomputable def sec (x : ℝ) := 1 / real.cos x

theorem sec_seven_pi_div_four : sec (7 * real.pi / 4) = real.sqrt 2 :=
by sorry

end sec_seven_pi_div_four_l776_776523


namespace pascal_triangle_50_4_eq_22050_l776_776757

theorem pascal_triangle_50_4_eq_22050 :
  ∑ (n : ℕ) in finset.range 1, (50.choose 4) = 22050 :=
sorry

end pascal_triangle_50_4_eq_22050_l776_776757


namespace binary_to_octal_101101_l776_776075

theorem binary_to_octal_101101 : 
  (binary_to_octal "101101") = "55" := 
sorry

end binary_to_octal_101101_l776_776075


namespace area_of_regular_octagon_inscribed_in_circle_l776_776045

theorem area_of_regular_octagon_inscribed_in_circle (r : ℝ) (h1 : r = 2) :
  let area := 16 * Real.sqrt 2 in
  ∀ (A B C D E F G H : Point)
  (h2 : RegularOctagonInscribedInCircle A B C D E F G H r), 
  AreaOfOctagon A B C D E F G H = area :=
by
  sorry

end area_of_regular_octagon_inscribed_in_circle_l776_776045


namespace find_lines_that_intersect_l776_776532

noncomputable def conic_family_intersect_lines (m : ℝ) : Prop :=
  ∃ k b, (∀ x y, 4 * x^2 + 5 * y^2 - 8 * m * x - 20 * m * y + 24 * m^2 - 20 = 0) ∧
  ((y - (kx + b)) = 0) ∧
  (sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = (5 / 3) * sqrt 5) ∧
  ((k, b) = (2, 2) ∨ (k, b) = (2, -2))

theorem find_lines_that_intersect :
  ∀ m, conic_family_intersect_lines m :=
by
  sorry

end find_lines_that_intersect_l776_776532


namespace find_wholesale_prices_l776_776845

-- Definitions based on conditions
def bakerPlusPrice (p : ℕ) (d : ℕ) : ℕ := p + d
def starPrice (p : ℕ) (k : ℚ) : ℕ := (p * k).toNat

-- The tuple of prices given
def prices : List ℕ := [64, 64, 70, 72]

-- Statement of the problem
theorem find_wholesale_prices (d : ℕ) (k : ℚ) (h_d : d > 0) (h_k : k > 1 ∧ k.denom = 1):
  ∃ p1 p2, 
  (p1 ∈ prices ∧ bakerPlusPrice p1 d ∈ prices ∧ starPrice p1 k ∈ prices) ∧
  (p2 ∈ prices ∧ bakerPlusPrice p2 d ∈ prices ∧ starPrice p2 k ∈ prices) :=
sorry

end find_wholesale_prices_l776_776845


namespace Benny_and_Tim_have_47_books_together_l776_776060

/-
  Definitions and conditions:
  1. Benny_has_24_books : Benny has 24 books.
  2. Benny_gave_10_books_to_Sandy : Benny gave Sandy 10 books.
  3. Tim_has_33_books : Tim has 33 books.
  
  Goal:
  Prove that together Benny and Tim have 47 books.
-/

def Benny_has_24_books : ℕ := 24
def Benny_gave_10_books_to_Sandy : ℕ := 10
def Tim_has_33_books : ℕ := 33

def Benny_remaining_books : ℕ := Benny_has_24_books - Benny_gave_10_books_to_Sandy

def Benny_and_Tim_together : ℕ := Benny_remaining_books + Tim_has_33_books

theorem Benny_and_Tim_have_47_books_together :
  Benny_and_Tim_together = 47 := by
  sorry

end Benny_and_Tim_have_47_books_together_l776_776060


namespace sum_difference_g_f_l776_776967

def f (n : ℕ) : ℕ := 
  if h : 2 ≤ n then
    let k1 := ((n - 1) / 2) + 1 in
    let k2 := (n / 2) + 1 in
    -- The distinct solutions counting logic for sin(nx) = sin(x) in interval [0, π]
    k1 + k2 - 1 -- Considering both cases and adjusting for distinct counting
  else
    0 -- Define it as 0 for n < 2

def g (n : ℕ) : ℕ := 
  if h : 2 ≤ n then
    let k1 := ((n + 1) / 2) + 1 in
    let k2 := ((n - 1) / 2) + 1 in
    -- The distinct solutions counting logic for cos(nx) = cos(x) in interval [0, π]
    k1 + k2 - 1 -- Considering both cases and adjusting for distinct counting
  else
    0 -- Define it as 0 for n < 2
  
theorem sum_difference_g_f : ∑ n in finset.Icc 2 2017, (g n - f n) = 504 :=
by
  -- Proof goes here
  sorry

end sum_difference_g_f_l776_776967


namespace remaining_grass_area_l776_776399

theorem remaining_grass_area (d r p : ℝ) (A_seg : ℝ) (h_d : d = 20) (h_w : p = 4) (h_r : r = d / 2) (h_A : A_seg = r^2 * real.arccos(p / (2 * r)) - p * real.sqrt(r^2 - (p / 2)^2)) :
  100 * real.pi - A_seg = real.pi * r^2 - A_seg := 
by 
  sorry

end remaining_grass_area_l776_776399


namespace binomial_coefficient_10_3_l776_776495

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l776_776495


namespace sum_digits_odd_even_difference_l776_776053

theorem sum_digits_odd_even_difference :
  let even_sum := ∑ n in ((Finset.range 1001).filter (λ n : ℕ, n % 2 = 0)), n.digits.sum
  let odd_sum := ∑ n in ((Finset.range 1001).filter (λ n : ℕ, n % 2 = 1)), n.digits.sum
  odd_sum - even_sum = 499 :=
by {
  sorry
}

end sum_digits_odd_even_difference_l776_776053


namespace coefficient_of_x6_l776_776756

theorem coefficient_of_x6 : 
  (coeff (expand_binomial 8 3 4 6)) = 326592 :=
by
  sorry

end coefficient_of_x6_l776_776756


namespace calc_expr_l776_776066

theorem calc_expr : (3 * 5 * 7 + 4 * 6 * 8) * (2 * 12 * 5 - 20 * 3 * 2) = 0 := 
by
  calc (3 * 5 * 7 + 4 * 6 * 8) * (2 * 12 * 5 - 20 * 3 * 2)
    ... = (105 + 192) * (120 - 120) : by sorry
    ... = 297 * 0 : by sorry
    ... = 0 : by sorry

end calc_expr_l776_776066


namespace find_other_intersection_point_l776_776729

-- Definitions
def parabola_eq (x : ℝ) : ℝ := x^2 - 2 * x - 3
def intersection_point1 : Prop := parabola_eq (-1) = 0
def intersection_point2 : Prop := parabola_eq 3 = 0

-- Proof problem
theorem find_other_intersection_point :
  intersection_point1 → intersection_point2 := by
  sorry

end find_other_intersection_point_l776_776729


namespace math_problem_l776_776311

namespace Proof

variable (f : ℝ → ℝ)

def prop_p := ∃ x : ℝ, 2^x > x

def prop_q := (∀ x : ℝ, f(x-1) = f(-(x-1))) → 
  (∀ x : ℝ, f(x) = f(2 - x))

theorem math_problem : prop_p ∨ prop_q := by
  sorry

end Proof

end math_problem_l776_776311


namespace remainder_when_subtracted_l776_776805

theorem remainder_when_subtracted (s t : ℕ) (hs : s % 6 = 2) (ht : t % 6 = 3) (h : s > t) : (s - t) % 6 = 5 :=
by
  sorry -- Proof not required

end remainder_when_subtracted_l776_776805


namespace comb_10_3_eq_120_l776_776501

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l776_776501


namespace shaded_triangle_area_l776_776018

theorem shaded_triangle_area (b h : ℝ) (hb : b = 2) (hh : h = 3) : 
  (1 / 2 * b * h) = 3 := 
by
  rw [hb, hh]
  norm_num

end shaded_triangle_area_l776_776018


namespace comb_10_3_eq_120_l776_776504

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l776_776504


namespace range_of_f_l776_776536

noncomputable def f : ℝ → ℝ :=
  λ x, (cos x)^3 + 8 * (cos x)^2 - 5 * (cos x) + 4 * (1 - (cos x)^2) - 11 / (cos x - 2)

theorem range_of_f : ∀ x : ℝ, cos x ≠ 2 → f x ∈ set.Icc (-0.5) 11.5 :=
by
  sorry

end range_of_f_l776_776536


namespace sec_of_7pi_over_4_l776_776521

theorem sec_of_7pi_over_4 : (Real.sec (7 * Real.pi / 4)) = Real.sqrt 2 :=
by
  sorry

end sec_of_7pi_over_4_l776_776521


namespace question_cos_sum_l776_776980

theorem question_cos_sum (α β : ℝ) (h : (cos α * cos (β / 2) / cos (α - β / 2)) + (cos β * cos (α / 2) / cos (β - α / 2)) = 1) :
  cos α + cos β = 1 :=
sorry

end question_cos_sum_l776_776980


namespace determine_g_l776_776276

def real_function (g : ℝ → ℝ) :=
  ∀ c d : ℝ, g (c + d) + g (c - d) = g (c) * g (d) + g (d)

def non_zero_function (g : ℝ → ℝ) :=
  ∃ x : ℝ, g x ≠ 0

theorem determine_g (g : ℝ → ℝ) (h1 : real_function g) (h2 : non_zero_function g) : g 0 = 1 ∧ ∀ x : ℝ, g (-x) = g x := 
sorry

end determine_g_l776_776276


namespace committee_meeting_arrangements_l776_776401

-- Definitions based on conditions
def schools : Finset ℕ := {1, 2, 3, 4}
def members_per_school : ℕ := 4
def total_members : ℕ := schools.card * members_per_school
def choose_representatives (n k : ℕ) : ℕ := Nat.choose n k

-- Prove the number of possible ways to arrange the committee meeting
theorem committee_meeting_arrangements :
  let host_school_choices := schools.card
  let host_representatives_choice := choose_representatives members_per_school 3
  let other_representatives_choice := choose_representatives members_per_school 1
  ∑ (host_school in schools), 
  (host_representatives_choice * (other_representatives_choice ^ 3)) = 1024 :=
by
  sorry

end committee_meeting_arrangements_l776_776401


namespace sum_of_elements_exists_l776_776656

theorem sum_of_elements_exists
  (A : Set ℕ) [Infinite A]
  (n m : ℕ) (hn : 1 < n)
  (hm : 1 < m)
  (coprime_mn : Nat.gcd m n = 1)
  (prime_condition : ∀ p : ℕ, p.Prime → ¬ p ∣ n → ∃ᶠ a in A, ¬ p ∣ a) :
  ∃ (S : ℕ), (S ∈ Finset.unions (Finset.image (Finset.sum (· : Finset ℕ)) finite_subsets))
    ∧ (S % m = 1) ∧ (S % n = 0) :=
by
  sorry

where
  finite_subsets : Finset (Finset ℕ) := 
    {s | s.finite.to_finset ⊆ A.to_finset}

end sum_of_elements_exists_l776_776656


namespace problem_approximation_l776_776441

theorem problem_approximation :
  (2.54 * 7.89 * (4.21 + 5.79)).approximate = 200 := by
  sorry

end problem_approximation_l776_776441


namespace system_of_equations_solution_l776_776321

theorem system_of_equations_solution 
  (x y : ℝ) 
  (h1 : 3 * x + real.sqrt (3 * x - y) + y = 6)
  (h2 : 9 * x^2 + 3 * x - y - y^2 = 36) : 
  (x = 2 ∧ y = -3) ∨ (x = 6 ∧ y = -18) :=
by sorry

end system_of_equations_solution_l776_776321


namespace garden_dimensions_l776_776898

theorem garden_dimensions (l w : ℕ) (h1 : 2 * l + 2 * w = 60) (h2 : l * w = 221) : 
    (l = 17 ∧ w = 13) ∨ (l = 13 ∧ w = 17) :=
sorry

end garden_dimensions_l776_776898


namespace max_magnitude_a_l776_776659

noncomputable def complex_eq_max_magnitude (a b : ℂ) : Prop :=
a^3 - 3 * a * b^2 = 36 ∧ b^3 - 3 * b * a^2 = 28 * complex.I

theorem max_magnitude_a (a b : ℂ) (h : complex_eq_max_magnitude a b) : 
|a| = 3 ↔ (a = 3 ∨ a = - (3 / 2) + (3 * real.sqrt 3 / 2) * complex.I ∨ a = - (3 / 2) - (3 * real.sqrt 3 / 2) * complex.I) :=
sorry

end max_magnitude_a_l776_776659


namespace binom_10_3_l776_776463

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l776_776463


namespace find_A_max_area_triangle_l776_776256

-- Declare noncomputable theory as necessary
noncomputable theory

variables {A B C : ℝ} -- Angles A, B, C
variables {a b c : ℝ} -- Sides opposite to angles A, B, C

-- Given conditions and question for proving the value of angle A and finding the max area
def cond1 : Prop := 2 * b * Real.cos A = c * Real.cos A + a * Real.cos C
def cond2 : a = 4
def value_A := A = Real.pi / 3
def max_area := ∃ b c, 4 * Real.sqrt 3

-- Statement 1: Proving angle A
theorem find_A (h1 : cond1) : value_A := by
  sorry

-- Statement 2: Finding maximum area with a = 4
theorem max_area_triangle (h1 : cond1) (h2 : cond2) : max_area := by
  sorry

end find_A_max_area_triangle_l776_776256


namespace remainder_product_mod_7_l776_776439

theorem remainder_product_mod_7 :
    (1296 * 1444 * 1700 * 1875) % 7 = 0 :=
begin
  sorry
end

end remainder_product_mod_7_l776_776439


namespace range_of_a_l776_776165

def f (a x : ℝ) := x^3 - a * x^2 + 3 * x
def g (a x : ℝ) := x + a / (2 * x)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 3 * x^2 - 2 * a * x + 3 ≥ 0) ∧ 
  (∃ x ∈ Ioo 1 2, ∃ y ∈ Ioo 1 2, g a x = g a y) →
  2 < a ∧ a ≤ 3 :=
sorry

end range_of_a_l776_776165


namespace find_other_number_l776_776387

theorem find_other_number (A : ℕ) (hcf_cond : Nat.gcd A 48 = 12) (lcm_cond : Nat.lcm A 48 = 396) : A = 99 := by
    sorry

end find_other_number_l776_776387


namespace volume_of_carved_off_portion_l776_776037

noncomputable def volume_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

noncomputable def volume_carved_off (r h : ℝ) : ℝ :=
  (2 / 3) * volume_cylinder r h

theorem volume_of_carved_off_portion :
  volume_carved_off 5 12 = 200 * π :=
by
  -- This is where the proof would go
  sorry

end volume_of_carved_off_portion_l776_776037


namespace avg_length_one_third_wires_l776_776741

theorem avg_length_one_third_wires (x : ℝ) (L1 L2 L3 L4 L5 L6 : ℝ) 
  (h_total_wires : L1 + L2 + L3 + L4 + L5 + L6 = 6 * 80) 
  (h_avg_other_wires : (L3 + L4 + L5 + L6) / 4 = 85) 
  (h_avg_all_wires : (L1 + L2 + L3 + L4 + L5 + L6) / 6 = 80) :
  (L1 + L2) / 2 = 70 :=
by
  sorry

end avg_length_one_third_wires_l776_776741


namespace unique_min_point_l776_776686

noncomputable def f (n : ℕ) (points : Fin n → ℝ × ℝ) (M : ℝ × ℝ) : ℝ :=
  ∑ i, dist M (points i)

theorem unique_min_point (n : ℕ) (h : n ≥ 3) (points : Fin n → ℝ × ℝ)
  (hnoncollinear : ¬ ∃ l : ℝ × ℝ → Prop, ∀ i, l (points i))
  (M1 : ℝ × ℝ) (M2 : ℝ × ℝ)
  (hM1 : ∀ M, f n points M1 ≤ f n points M)
  (heq : f n points M1 = f n points M2) :
  M1 = M2 :=
by
  sorry

end unique_min_point_l776_776686


namespace f_increasing_infinitely_often_f_decreasing_infinitely_often_l776_776733

noncomputable def f (n : ℕ) : ℚ :=
  if h : 0 < n then
    let m := n in
    (∑ k in finset.range (n + 1), if k > 0 then (n / k : ℚ).floor else 0) / n
  else
    0

theorem f_increasing_infinitely_often : ∃ᶠ n in at_top, f (n + 1) > f n :=
sorry

theorem f_decreasing_infinitely_often : ∃ᶠ n in at_top, f (n + 1) < f n :=
sorry

end f_increasing_infinitely_often_f_decreasing_infinitely_often_l776_776733


namespace minimum_value_frac_abc_l776_776651

variable (a b c : ℝ)

theorem minimum_value_frac_abc
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + 2 * c = 2) :
  (a + b) / (a * b * c) ≥ 8 :=
sorry

end minimum_value_frac_abc_l776_776651


namespace phase_shift_sin_4x_sub_2pi_l776_776931

theorem phase_shift_sin_4x_sub_2pi :
  let b := 4
  let c := 2 * Real.pi
  phase_shift : Real := c / b
  phase_shift = (Real.pi / 2) := sorry

end phase_shift_sin_4x_sub_2pi_l776_776931


namespace vector_parallel_y_axis_l776_776170

theorem vector_parallel_y_axis (x : ℝ) :
  let a := (x, 1)
  let b := (-x, x^2)
  let sum := (a.1 + b.1, a.2 + b.2)
  sum = (0, 1 + x^2) → sum.1 = 0 ∧ sum.2 ≠ 0 :=
by
  intros
  let a := (x, 1)
  let b := (-x, x^2)
  let a_plus_b := (a.1 + b.1, a.2 + b.2)
  have h1 : a.1 + b.1 = 0, by sorry
  have h2 : a.2 + b.2 = 1 + x^2, by sorry
  exact ⟨h1, by sorry⟩

end vector_parallel_y_axis_l776_776170


namespace alice_knows_parity_l776_776427

def poly_game (f : ℤ → ℤ) : Prop :=
  (∀ n : ℤ, f n ∈ ℤ) ∧ degree f < 187

theorem alice_knows_parity :
  ∀ f, poly_game f → (∃ (N : ℕ), (∀ (k : ℕ), k ∈ finset.range(188) → (f k % 2 = 0 ∨ f k % 2 = 1)) → ∀ m, f m % 2 = f 0 % 2 ∧ N = 187) := sorry

end alice_knows_parity_l776_776427


namespace intersection_of_A_and_B_l776_776172

namespace ProofProblem

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 5} := by
  sorry

end ProofProblem

end intersection_of_A_and_B_l776_776172


namespace binom_10_3_l776_776458

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l776_776458


namespace knights_statements_l776_776306

theorem knights_statements (r ℓ : Nat) (hr : r ≥ 2) (hℓ : ℓ ≥ 2)
  (h : 2 * r * ℓ = 230) :
  (r + ℓ) * (r + ℓ - 1) - 230 = 526 :=
by
  sorry

end knights_statements_l776_776306


namespace length_of_copper_wire_is_greater_than_225m_l776_776354

-- Defining the given conditions
def copper_density : ℝ := 8900  -- in kg/m^3
noncomputable def copper_volume : ℝ := 0.5 * 10^(-3)  -- in m^3
def square_diagonal : ℝ := 2 * 10^(-3)  -- in meters

-- Useful abstractions
noncomputable def square_side : ℝ := square_diagonal / Real.sqrt 2
noncomputable def cross_sectional_area : ℝ := square_side^2

-- The statement to be proved
theorem length_of_copper_wire_is_greater_than_225m :
  copper_volume = cross_sectional_area * 250 :=
sorry

end length_of_copper_wire_is_greater_than_225m_l776_776354


namespace change_in_mean_and_median_l776_776330

def mean (data : List ℕ) : ℚ :=
  data.foldr (fun x acc => x + acc) 0 / data.length

def median (data : List ℕ) : ℚ :=
  let sorted_data := data.qsort (≤)
  let n := sorted_data.length
  if n % 2 = 0 then
    (sorted_data.get (n / 2 - 1) + sorted_data.get (n / 2)) / 2
  else
    sorted_data.get (n / 2)

def original_data := [20, 26, 16, 22, 16]
def corrected_data := [20, 26, 21, 22, 16]

theorem change_in_mean_and_median :
  mean corrected_data - mean original_data = 1 ∧
  median corrected_data - median original_data = 1 := 
sorry

end change_in_mean_and_median_l776_776330


namespace geometric_reasoning_l776_776593

-- Definitions of relationships between geometric objects
inductive GeometricObject
  | Line
  | Plane

open GeometricObject

def perpendicular (a b : GeometricObject) : Prop := 
  match a, b with
  | Plane, Plane => True  -- Planes can be perpendicular
  | Line, Plane => True   -- Lines can be perpendicular to planes
  | Plane, Line => True   -- Planes can be perpendicular to lines
  | Line, Line => True    -- Lines can be perpendicular to lines (though normally in a 3D space specific context)

def parallel (a b : GeometricObject) : Prop := 
  match a, b with
  | Plane, Plane => True  -- Planes can be parallel
  | Line, Plane => True   -- Lines can be parallel to planes under certain interpretation
  | Plane, Line => True
  | Line, Line => True    -- Lines can be parallel

axiom x : GeometricObject
axiom y : GeometricObject
axiom z : GeometricObject

-- Main theorem statement
theorem geometric_reasoning (hx : perpendicular x y) (hy : parallel y z) 
  : ¬ (perpendicular x z) → (x = Plane ∧ y = Plane ∧ z = Line) :=
  sorry

end geometric_reasoning_l776_776593


namespace ajith_rana_meet_l776_776011

/--
Ajith and Rana walk around a circular course 115 km in circumference, starting together from the same point.
Ajith walks at 4 km/h, and Rana walks at 5 km/h in the same direction.
Prove that they will meet after 115 hours.
-/
theorem ajith_rana_meet 
  (course_circumference : ℕ)
  (ajith_speed : ℕ)
  (rana_speed : ℕ)
  (relative_speed : ℕ)
  (time : ℕ)
  (start_point : Point)
  (ajith : Person)
  (rana : Person)
  (walk_in_same_direction : Prop)
  (start_time : ℕ)
  (meet_time : ℕ) :
  course_circumference = 115 →
  ajith_speed = 4 →
  rana_speed = 5 →
  relative_speed = rana_speed - ajith_speed →
  time = course_circumference / relative_speed →
  meet_time = start_time + time →
  meet_time = 115 :=
by
  sorry

end ajith_rana_meet_l776_776011


namespace find_m_p_pairs_l776_776094

theorem find_m_p_pairs (m p : ℕ) (h_prime : Nat.Prime p) (h_eq : ∃ (x : ℕ), 2^m * p^2 + 27 = x^3) :
  (m, p) = (1, 7) :=
sorry

end find_m_p_pairs_l776_776094


namespace raven_current_age_l776_776230

variable (R P : ℕ) -- Raven's current age, Phoebe's current age
variable (h₁ : P = 10) -- Phoebe is currently 10 years old
variable (h₂ : R + 5 = 4 * (P + 5)) -- In 5 years, Raven will be 4 times as old as Phoebe

theorem raven_current_age : R = 55 := 
by
  -- h2: R + 5 = 4 * (P + 5)
  -- h1: P = 10
  sorry

end raven_current_age_l776_776230


namespace permutation_combination_correctness_l776_776002

noncomputable def combination (n k : ℕ) : ℕ := n.choose k

theorem permutation_combination_correctness :
  (combination 5 3 ≠ 5 * 4 * 3) ∧
  ((combination 4 2) = 6) ∧
  ((∑ i in (finset.range 98).map (λ i, i + 3), combination i 2) ≠ combination 101 3) ∧
  (∀ n : ℕ, ((n * (n + 1) * (n + 2) * ... * (n + 100)) / 100! = 101 * (combination (n + 100) 101)))

end permutation_combination_correctness_l776_776002


namespace find_angle_A_find_max_area_l776_776255

noncomputable def triangle (a b c A B C : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  A + B + C = π ∧
  sin A > 0 ∧ sin B > 0 ∧ sin C > 0

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h : triangle a b c A B C) :
  (2 * b * cos A = c * cos A + a * cos C) → A = π / 3 :=
by
  sorry

theorem find_max_area (a b c : ℝ) (A B C : ℝ) (h : triangle a b c A B C) :
  (2 * b * cos A = c * cos A + a * cos C) → a = 4 → (1/2 * b * c * sin A ≤ 4 * sqrt 3) :=
by
  sorry 

end find_angle_A_find_max_area_l776_776255


namespace women_in_first_group_l776_776813

variable {w m : ℚ}

-- Conditions
variable (hw : w ≠ 0)
variable h1 : 3 * m + x * w = 6 * m + 2 * w
variable h2 : 3 * m + 2 * w = (4 / 7) * (3 * m + x * w)
variable h3 : m = 2 * w

theorem women_in_first_group : x = 8 :=
by
  sorry

end women_in_first_group_l776_776813


namespace cone_height_is_sqrt_35_l776_776581

noncomputable def sector_radius : ℝ := 6
noncomputable def central_angle : ℝ := Real.pi / 3
noncomputable def arc_length : ℝ := sector_radius * central_angle
noncomputable def base_circumference : ℝ := arc_length
noncomputable def base_radius : ℝ := base_circumference / (2 * Real.pi)

theorem cone_height_is_sqrt_35 : ∃ h : ℝ, h = Real.sqrt (sector_radius ^ 2 - base_radius ^ 2) ∧ h = Real.sqrt 35 := 
by
  use Real.sqrt 35
  split
  · have hs : base_radius = 1 := by
      calc
        base_radius = arc_length / (2 * Real.pi) : rfl
        ... = (6 * (Real.pi / 3)) / (2 * Real.pi) : rfl
        ... = 2*Real.pi / (2*Real.pi) : by simp [arc_length, sector_radius, central_angle]
        ... = 1 : by simp
    have ht : sector_radius ^ 2 - base_radius ^ 2 = 35 := by
      calc
        (6:ℝ) ^ 2 - 1 ^ 2 = 36 - 1 : by norm_num
        ... = 35 : by norm_num
    show Real.sqrt (sector_radius ^ 2 - base_radius ^ 2) = Real.sqrt 35, from congr_arg Real.sqrt ht
  · refl

end cone_height_is_sqrt_35_l776_776581


namespace least_positive_three_digit_multiple_of_11_l776_776763

theorem least_positive_three_digit_multiple_of_11 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 11 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 11 ∣ m → n ≤ m) :=
begin
  use 110,
  split,
  exact (100 ≤ 110),
  split,
  exact (110 ≤ 999),
  split,
  exact (11 ∣ 110),
  intros m Hm,
  cases Hm with Hm100 HmGCD,
  cases HmGCD with Hm999 HmDiv,
  sorry,
end

end least_positive_three_digit_multiple_of_11_l776_776763


namespace area_triangle_ABC_l776_776409

noncomputable def area_of_triangle_ABC : ℝ :=
  let base_AB : ℝ := 6 - 0
  let height_AB : ℝ := 2 - 0
  let base_BC : ℝ := 6 - 3
  let height_BC : ℝ := 8 - 0
  let base_CA : ℝ := 3 - 0
  let height_CA : ℝ := 8 - 2
  let area_ratio : ℝ := 1 / 2
  let area_I' : ℝ := area_ratio * base_AB * height_AB
  let area_II' : ℝ := area_ratio * 8 * 6
  let area_III' : ℝ := area_ratio * 8 * 3
  let total_small_triangles : ℝ := area_I' + area_II' + area_III'
  let total_area_rectangle : ℝ := 6 * 8
  total_area_rectangle - total_small_triangles

theorem area_triangle_ABC : area_of_triangle_ABC = 6 := 
by
  sorry

end area_triangle_ABC_l776_776409


namespace largest_three_prime_product_sum_digits_l776_776653

def is_prime (n : ℕ) : Prop := ∀ d, d ∣ n → d = 1 ∨ d = n

def single_digit_primes : List ℕ := [2, 3, 5, 7]

def is_single_digit_prime (n : ℕ) : Prop := n ∈ single_digit_primes

def digit_sum (n : ℕ) : ℕ :=
  (toString n).toList.map (λ c, c.toNat - '0'.toNat).sum

theorem largest_three_prime_product_sum_digits :
  ∃ (d e : ℕ),
    is_single_digit_prime d ∧
    is_single_digit_prime e ∧
    d ≠ e ∧
    is_prime (d + 10 * e) ∧
    (digit_sum (d * e * (d + 10 * e)) = 21) := sorry

end largest_three_prime_product_sum_digits_l776_776653


namespace k_chains_count_l776_776565

def is_good_set (I : Set ℤ) : Prop :=
  ∀ x y ∈ I, x - y ∈ I

def is_k_chain (k : ℕ) (I : Fin k → Set ℤ) : Prop :=
  (∀ i : Fin k, 168 ∈ I i) ∧
  (∀ i : Fin k, is_good_set (I i)) ∧
  (∀ i j : Fin k, i ≤ j → I i ⊇ I j)

theorem k_chains_count (k : ℕ) (I : Fin k → Set ℤ) :
  is_k_chain k I → ∃! n : ℕ, n = (k + 1) ^ 3 * (k + 2) * (k + 3) / 6 :=
sorry

end k_chains_count_l776_776565


namespace binom_10_3_l776_776464

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l776_776464


namespace letters_with_dot_not_line_l776_776232

-- Definitions from conditions
def D_inter_S : ℕ := 23
def S : ℕ := 42
def Total_letters : ℕ := 70

-- Problem statement
theorem letters_with_dot_not_line : (Total_letters - S - D_inter_S) = 5 :=
by sorry

end letters_with_dot_not_line_l776_776232


namespace collinear_A1_B1_C1_l776_776618

noncomputable def A_1 {P : Type*} [EuclideanSpace ℝ P] (A A' B C : P) : P :=
(intersect (line B C) (perp_bisector (segment A A'))) 

noncomputable def B_1 {P : Type*} [EuclideanSpace ℝ P] (B B' A C : P) : P :=
(intersect (line A C) (perp_bisector (segment B B'))) 

noncomputable def C_1 {P : Type*} [EuclideanSpace ℝ P] (C C' A B : P) : P :=
(intersect (line A B) (perp_bisector (segment C C'))) 

theorem collinear_A1_B1_C1 {P : Type*} [EuclideanSpace ℝ P] (A A' B B' C C' : P)
  (h1 : is_convex_hexagon (hexagon A C' B A' C B'))
  (h2 : opposite_sides_equal (hexagon A C' B A' C B')) :
  collinear_set {A_1 A A' B C B', B_1 B B' A C A', C_1 C C' A B A'} :=
by
  -- Steps go here
  sorry

end collinear_A1_B1_C1_l776_776618


namespace functional_equation_solution_l776_776093

theorem functional_equation_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f x * f y = f (x - y)) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) :=
sorry

end functional_equation_solution_l776_776093


namespace LiFangOutfitChoices_l776_776664

variable (shirts skirts dresses : Nat) 

theorem LiFangOutfitChoices (h_shirts : shirts = 4) (h_skirts : skirts = 3) (h_dresses : dresses = 2) :
  shirts * skirts + dresses = 14 :=
by 
  -- Given the conditions and the calculations, the expected result follows.
  sorry

end LiFangOutfitChoices_l776_776664


namespace ordering_inequality_l776_776979

noncomputable def x := log 2 3 - log 2 (sqrt 3)
noncomputable def y := log 0.5 3
noncomputable def z := 0.9^(-1.1)

theorem ordering_inequality : y < x ∧ x < z :=
by
  -- sorry is a placeholder to indicate where the proof would go
  sorry

end ordering_inequality_l776_776979


namespace smallest_value_n_l776_776086

theorem smallest_value_n :
  ∃ (n : ℕ), n * 25 = Nat.lcm (Nat.lcm 10 18) 20 ∧ (∀ m, m * 25 = Nat.lcm (Nat.lcm 10 18) 20 → n ≤ m) := 
sorry

end smallest_value_n_l776_776086


namespace series_convergence_l776_776688

noncomputable def sum_to_n (n : ℕ) (l : ℝ) : ℝ :=
∑ k in finset.range (n + 1), 1 / (k + 1 : ℝ) ^ l

theorem series_convergence {l : ℝ} (hl : l > 1) :
  ∃ C : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (sum_to_n n l - C) < ε ∧
  (1 / (l - 1) < C ∧ C < l / (l - 1)) := sorry

end series_convergence_l776_776688


namespace steve_speed_on_way_back_l776_776389

-- Let's define the variables and constants used in the problem.
def distance_to_work : ℝ := 30 -- in km
def total_time_on_road : ℝ := 6 -- in hours
def back_speed_ratio : ℝ := 2 -- Steve drives twice as fast on the way back

theorem steve_speed_on_way_back :
  ∃ v : ℝ, v > 0 ∧ (30 / v + 15 / v = 6) ∧ (2 * v = 15) := by
  sorry

end steve_speed_on_way_back_l776_776389


namespace equal_number_of_frogs_after_6_months_l776_776932

theorem equal_number_of_frogs_after_6_months :
  ∃ n : ℕ, 
    n = 6 ∧ 
    (∀ Dn Qn : ℕ, 
      (Dn = 5^(n + 1) ∧ Qn = 3^(n + 5)) → 
      Dn = Qn) :=
by
  sorry

end equal_number_of_frogs_after_6_months_l776_776932


namespace binomial_coefficient_10_3_l776_776471

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l776_776471


namespace log_base_2_of_16_sqrt_2_l776_776939

theorem log_base_2_of_16_sqrt_2 : Real.log 2 (16 * Real.sqrt 2) = 9 / 2 :=
by
  sorry

end log_base_2_of_16_sqrt_2_l776_776939


namespace least_three_digit_multiple_13_l776_776772

theorem least_three_digit_multiple_13 : 
  ∃ n : ℕ, (n ≥ 100) ∧ (∃ k : ℕ, n = 13 * k) ∧ ∀ m, m < n → (m < 100 ∨ ¬∃ k : ℕ, m = 13 * k) :=
by
  sorry

end least_three_digit_multiple_13_l776_776772


namespace math_problem_l776_776806

theorem math_problem :
  let a := 481 * 7
  let b := 426 * 5
  ((a + b) ^ 3 - 4 * a * b) = 166021128033 := 
by
  let a := 481 * 7
  let b := 426 * 5
  sorry

end math_problem_l776_776806


namespace wholesale_price_is_60_l776_776864

variables (p d k : ℕ)
variables (P1 P2 Z1 Z2 : ℕ)

// Conditions from (a)
axiom H1 : P1 = p + d // Baker Plus price day 1
axiom H2 : P2 = p + d // Baker Plus price day 2
axiom H3 : Z1 = k * p // Star price day 1
axiom H4 : Z2 = k * p // Star price day 2
axiom H5 : k > 1 // Star store markup factor greater than 1
axiom H6 : P1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H7 : P2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H8 : Z1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H9 : Z2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))

theorem wholesale_price_is_60 : p = 60 := 
by {
    // Proof goes here, skipped with sorry
    sorry
}

end wholesale_price_is_60_l776_776864


namespace trigonometric_relations_l776_776285

theorem trigonometric_relations (x : ℝ) (h0 : 0 < x) (h1 : x < (Real.pi / 2)) :
  sin x < x ∧ x < tan x := by
  sorry

end trigonometric_relations_l776_776285


namespace least_three_digit_multiple_of_13_l776_776780

theorem least_three_digit_multiple_of_13 : ∃ n : ℕ, n ≥ 100 ∧ n % 13 = 0 ∧ ∀ m : ℕ, m ≥ 100 → m % 13 = 0 → n ≤ m :=
begin
  use 104,
  split,
  { exact nat.le_of_eq rfl },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero rfl) },
  { intros m hm hmod,
    have h8 : 8 * 13 = 104 := rfl,
    rw ←h8,
    exact nat.le_mul_of_pos_left (by norm_num) },
end

end least_three_digit_multiple_of_13_l776_776780


namespace true_propositions_count_l776_776432

theorem true_propositions_count :
  let p1 := ∀ x y : ℝ, (x = 0 ∧ y = 0) → (x^2 + y^2 = 0)
      p2 := ¬ (∀ (Δ1 Δ2 : Triangle), Congruent Δ1 Δ2 → Similar Δ1 Δ2)
      p3 := ∀ m : ℝ, (¬ (∃ x : ℝ, x^2 + x - m = 0)) → m ≤ 0
      p4 := ∀ (a b c : ℝ), (c^2 ≠ a^2 + b^2) → (angleABC a b c ≠ 90)
  in (p1 ∧ p3 ∧ p4) ∧ ¬ p2 :=
  3 :=
by
  sorry

end true_propositions_count_l776_776432


namespace shift_symmetry_origin_l776_776160

theorem shift_symmetry_origin (f g : ℝ → ℝ)
  (h1 : ∀ x, f x = (sqrt 3 / 4) * sin x - (1 / 4) * cos x)
  (h2 : ∀ x, g x = f (x - m))
  (h3 : ∀ x, g (-x) = -g (x))
  (h4 : 0 < m ∧ m < π) :
  m = 5 * π / 6 :=
sorry

end shift_symmetry_origin_l776_776160


namespace length_of_copper_wire_is_greater_than_225m_l776_776355

-- Defining the given conditions
def copper_density : ℝ := 8900  -- in kg/m^3
noncomputable def copper_volume : ℝ := 0.5 * 10^(-3)  -- in m^3
def square_diagonal : ℝ := 2 * 10^(-3)  -- in meters

-- Useful abstractions
noncomputable def square_side : ℝ := square_diagonal / Real.sqrt 2
noncomputable def cross_sectional_area : ℝ := square_side^2

-- The statement to be proved
theorem length_of_copper_wire_is_greater_than_225m :
  copper_volume = cross_sectional_area * 250 :=
sorry

end length_of_copper_wire_is_greater_than_225m_l776_776355


namespace thomas_score_l776_776300

def average (scores : List ℕ) : ℚ := scores.sum / scores.length

variable (scores : List ℕ)

theorem thomas_score (h_length : scores.length = 19)
                     (h_avg_before : average scores = 78)
                     (h_avg_after : average ((98 :: scores)) = 79) :
  let thomas_score := 98
  thomas_score = 98 := sorry

end thomas_score_l776_776300


namespace geometric_series_first_term_l776_776056

theorem geometric_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h_r : r = 1/4) (h_S : S = 80)
  (h_sum : S = a / (1 - r)) : a = 60 :=
by
  -- proof steps
  sorry

end geometric_series_first_term_l776_776056


namespace bakery_wholesale_price_exists_l776_776856

theorem bakery_wholesale_price_exists :
  ∃ (p1 p2 d k : ℕ), (d > 0) ∧ (k > 1) ∧
  ({p1 + d, p2 + d, k * p1, k * p2} = {64, 64, 70, 72}) :=
by sorry

end bakery_wholesale_price_exists_l776_776856


namespace arrange_polynomial_l776_776913

-- Define the polynomial
def poly := 3 * x^2 - 1 - 6 * x^5 - 4 * x^3

-- Statement about the polynomial arranged in descending order
theorem arrange_polynomial : poly = -6 * x^5 - 4 * x^3 + 3 * x^2 - 1 :=
by sorry

end arrange_polynomial_l776_776913


namespace determine_wholesale_prices_l776_776824

-- Definitions and Conditions
-- Wholesale prices for days 1 and 2 are defined as p1 and p2
def wholesale_price_day1 : ℝ := p1
def wholesale_price_day2 : ℝ := p2

-- The price increment at "Baker Plus" store is a constant d
def price_increment_baker_plus : ℝ := d
-- The factor at "Star" store is a constant k > 1
def price_factor_star : ℝ := k
-- Given prices for two days: 64, 64, 70, 72 rubles
def given_prices : set ℝ := {64, 64, 70, 72}

-- Prices at the stores
def baker_plus_day1_price := wholesale_price_day1 + price_increment_baker_plus
def star_day1_price := wholesale_price_day1 * price_factor_star

def baker_plus_day2_price := wholesale_price_day2 + price_increment_baker_plus
def star_day2_price := wholesale_price_day2 * price_factor_star

-- Theorem to determine the wholesale prices for each day (p1 and p2)
theorem determine_wholesale_prices
    (baker_plus_day1_price ∈ given_prices)
    (star_day1_price ∈ given_prices)
    (baker_plus_day2_price ∈ given_prices)
    (star_day2_price ∈ given_prices)
    (h1 : baker_plus_day1_price ≠ star_day1_price)
    (h2 : baker_plus_day2_price ≠ star_day2_price) :
  (∃ p1 p2 d k > 1, 
      (p1 + d ∈ given_prices) ∧ (p1 * k ∈ given_prices) ∧
      (p2 + d ∈ given_prices) ∧ (p2 * k ∈ given_prices) ∧
      (p1 ≠ p2)) :=
  sorry

end determine_wholesale_prices_l776_776824


namespace gcd_computation_l776_776962

theorem gcd_computation (a b : ℕ) (h₁ : a = 7260) (h₂ : b = 540) : 
  Nat.gcd a b - 12 + 5 = 53 :=
by
  rw [h₁, h₂]
  sorry

end gcd_computation_l776_776962


namespace least_three_digit_multiple_13_l776_776776

theorem least_three_digit_multiple_13 : 
  ∃ n : ℕ, (n ≥ 100) ∧ (∃ k : ℕ, n = 13 * k) ∧ ∀ m, m < n → (m < 100 ∨ ¬∃ k : ℕ, m = 13 * k) :=
by
  sorry

end least_three_digit_multiple_13_l776_776776


namespace example_theorem_l776_776612

noncomputable def max_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : ∃ x y, x^2 + y^2 = 1 ∧ a * x + b * y = 1) 
  (h_area : ∃ A B : ℝ × ℝ, ((A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ 
                             (a * A.1 + b * A.2 = 1) ∧ (a * B.1 + b * B.2 = 1) ∧ 
                             (∃ O : ℝ × ℝ, O = (0,0) ∧ triangle_area O A B = 1 / 2))) : 
                             a + b ≤ 2 := sorry

theorem example_theorem : max_a_plus_b = 2 := sorry

end example_theorem_l776_776612


namespace wholesale_price_is_60_l776_776866

variables (p d k : ℕ)
variables (P1 P2 Z1 Z2 : ℕ)

// Conditions from (a)
axiom H1 : P1 = p + d // Baker Plus price day 1
axiom H2 : P2 = p + d // Baker Plus price day 2
axiom H3 : Z1 = k * p // Star price day 1
axiom H4 : Z2 = k * p // Star price day 2
axiom H5 : k > 1 // Star store markup factor greater than 1
axiom H6 : P1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H7 : P2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H8 : Z1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H9 : Z2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))

theorem wholesale_price_is_60 : p = 60 := 
by {
    // Proof goes here, skipped with sorry
    sorry
}

end wholesale_price_is_60_l776_776866


namespace probability_ace_then_king_l776_776403

-- Definitions of the conditions
def custom_deck := 65
def extra_spades := 14
def total_aces := 4
def total_kings := 4

-- Probability calculations
noncomputable def P_ace_first : ℚ := total_aces / custom_deck
noncomputable def P_king_second : ℚ := total_kings / (custom_deck - 1)

theorem probability_ace_then_king :
  (P_ace_first * P_king_second) = 1 / 260 :=
by
  sorry

end probability_ace_then_king_l776_776403


namespace sum_of_given_numbers_l776_776736

theorem sum_of_given_numbers : 30 + 80000 + 700 + 60 = 80790 :=
  by
    sorry

end sum_of_given_numbers_l776_776736


namespace proposition_A_implies_B_not_proposition_B_implies_A_l776_776310

variable {x y : ℕ}

def PropA : Prop := x + y ≠ 2010
def PropB : Prop := x ≠ 1010 ∨ y ≠ 1000

theorem proposition_A_implies_B : PropA → PropB := by
  intro h
  unfold PropA at h
  unfold PropB
  sorry

theorem not_proposition_B_implies_A : ¬ (PropB → PropA) := by
  intro h
  unfold PropA
  unfold PropB
  sorry

end proposition_A_implies_B_not_proposition_B_implies_A_l776_776310


namespace negation_statement_l776_776724

theorem negation_statement (x : ℝ) : ¬ (∀ x : ℝ, real.exp x > x) ↔ ∃ x : ℝ, real.exp x ≤ x :=
by 
  sorry

end negation_statement_l776_776724


namespace binom_10_3_l776_776461

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l776_776461


namespace find_f_2017_l776_776987

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (x + 2) = -f x
axiom f_neg1 : f (-1) = -3

theorem find_f_2017 : f 2017 = 3 := 
by
  sorry

end find_f_2017_l776_776987


namespace pie_prices_l776_776872

theorem pie_prices 
  (d k : ℕ) (hd : 0 < d) (hk : 1 < k) 
  (p1 p2 : ℕ) 
  (prices : List ℕ) (h_prices : prices = [64, 64, 70, 72]) 
  (hP1 : (p1 + d) ∈ prices)
  (hZ1 : (k * p1) ∈ prices)
  (hP2 : (p2 + d) ∈ prices)
  (hZ2 : (k * p2) ∈ prices)
  (h_different_prices : p1 + d ≠ k * p1 ∧ p2 + d ≠ k * p2) : 
  ∃ p1 p2, p1 ≠ p2 :=
sorry

end pie_prices_l776_776872


namespace coefficient_x3y5_of_expansion_l776_776367

theorem coefficient_x3y5_of_expansion : 
  (binom 8 5) * ((2 / 3) ^ 3) * ((-1 / 3) ^ 5) = -448 / 6561 := by
  -- We skip the proof as instructed
  sorry

end coefficient_x3y5_of_expansion_l776_776367


namespace strictly_increasing_solution_l776_776530

open Nat

-- Define the domain of positive integers
def positive_integers := {n : ℕ // n ≠ 0}

-- Define a strictly increasing function on positive integers
def strictly_increasing_function (f : positive_integers → positive_integers) : Prop :=
  ∀ n m, n < m → f n < f m

-- Define the main theorem stating the identified problem and conclusions
theorem strictly_increasing_solution (f : positive_integers → positive_integers) :
  (∀ n, f (f n) = f n * f n) ∧ strictly_increasing_function f →
  ∃ c : ℕ, c ≠ 0 ∧
    (∀ n, n ≠ 0 → (n = 1 → f n = 1 ∨ f n = c) ∧ (n > 1 → f n = c * n)) :=
sorry

end strictly_increasing_solution_l776_776530


namespace interval_of_monotonic_increase_ln_of_quadratic_l776_776336

theorem interval_of_monotonic_increase_ln_of_quadratic :
  ∀ x : ℝ, (x > 1 → monotonic_increasing (λ x, ln (x^2 - x))) :=
begin
  sorry
end

end interval_of_monotonic_increase_ln_of_quadratic_l776_776336


namespace wholesale_price_is_60_l776_776865

variables (p d k : ℕ)
variables (P1 P2 Z1 Z2 : ℕ)

// Conditions from (a)
axiom H1 : P1 = p + d // Baker Plus price day 1
axiom H2 : P2 = p + d // Baker Plus price day 2
axiom H3 : Z1 = k * p // Star price day 1
axiom H4 : Z2 = k * p // Star price day 2
axiom H5 : k > 1 // Star store markup factor greater than 1
axiom H6 : P1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H7 : P2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H8 : Z1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H9 : Z2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))

theorem wholesale_price_is_60 : p = 60 := 
by {
    // Proof goes here, skipped with sorry
    sorry
}

end wholesale_price_is_60_l776_776865


namespace binomial_coefficient_10_3_l776_776492

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l776_776492


namespace convert_point_to_cylindrical_l776_776076

def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.atan2 y x
  (r, θ, z)

theorem convert_point_to_cylindrical :
  rectangular_to_cylindrical (-4) (-4 * Real.sqrt 3) (-3) = (8, 4 * Real.pi / 3, -3) := by
  sorry

end convert_point_to_cylindrical_l776_776076


namespace comb_10_3_eq_120_l776_776497

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l776_776497


namespace actual_distance_between_locations_l776_776683

/-- 
Given:
- The scale of the map is 1:25000
- The distance between two locations A and B on the map is 5 cm
Prove:
- The actual distance between A and B is 1.25 km
--/

theorem actual_distance_between_locations
(scale : ℕ := 25000)
(measured_distance_cm : ℕ := 5)
(conversion_rate : ℕ := 100000)
: (measured_distance_cm * scale) / conversion_rate = 1.25 := by
  sorry

end actual_distance_between_locations_l776_776683


namespace smallest_positive_z_l776_776704

theorem smallest_positive_z (x z : ℝ) (m n : ℤ) (h1 : sin x = 0) (h2 : cos (x + z) = -1/2) :
  z = 2 * Real.pi / 3 := sorry

end smallest_positive_z_l776_776704


namespace cases_in_1990_eq_150450_l776_776227

-- Assuming cases decrease linearly from 1960 to 2000

def cases_1960 := 600000
def cases_2000 := 600

noncomputable def yearly_decrease : ℕ := (cases_1960 - cases_2000) / 40

def cases_in_year (year : ℕ) : ℕ := cases_1960 - yearly_decrease * (year - 1960)

theorem cases_in_1990_eq_150450 : cases_in_year 1990 = 150450 := by
  sorry

end cases_in_1990_eq_150450_l776_776227


namespace tetrahedron_median_planes_parallelepiped_median_planes_count_parallelepipeds_l776_776547

-- Assuming definitions for Tetrahedron, Parallelepiped, and MedianPlane exist

-- 1. Number of distinct median planes for a tetrahedron
theorem tetrahedron_median_planes (T : Tetrahedron) : 
  count_distinct_median_planes T = 7 := by
  sorry
  
-- 2. Number of distinct median planes for a parallelepiped
theorem parallelepiped_median_planes (P : Parallelepiped) : 
  count_distinct_median_planes P = 3 := by
  sorry

-- 3. Number of different parallelepipeds formed by four non-coplanar points
theorem count_parallelepipeds (A B C D : Point) (h_non_coplanar : ¬coplanar {A, B, C, D}) : 
  count_parallelepipeds {A, B, C, D} = 29 := by
  sorry

end tetrahedron_median_planes_parallelepiped_median_planes_count_parallelepipeds_l776_776547


namespace bakery_wholesale_price_exists_l776_776861

theorem bakery_wholesale_price_exists :
  ∃ (p1 p2 d k : ℕ), (d > 0) ∧ (k > 1) ∧
  ({p1 + d, p2 + d, k * p1, k * p2} = {64, 64, 70, 72}) :=
by sorry

end bakery_wholesale_price_exists_l776_776861


namespace num_divisors_of_30_l776_776199

-- Define what it means to be a divisor of 30
def is_divisor (n : ℤ) : Prop :=
  n ≠ 0 ∧ 30 % n = 0

-- Define a function that counts the number of divisors of 30
def count_divisors (d : ℤ) : ℕ :=
  ((Multiset.filter is_divisor ((List.range' (-30) 61).map (Int.ofNat))).attach.to_finset.card)

-- State the theorem
theorem num_divisors_of_30 : count_divisors 30 = 16 := sorry

end num_divisors_of_30_l776_776199


namespace comb_10_3_eq_120_l776_776499

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l776_776499


namespace math_problem_equiv_proof_l776_776131

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

variables {l : linear_map ℝ (ℝ × ℝ)} {C : set (ℝ × ℝ)} {E : set (ℝ × ℝ)}
variables (F : ℝ × ℝ) (N : ℝ × ℝ) (A B M P : ℝ × ℝ)

theorem math_problem_equiv_proof :
  (∀ P : ℝ × ℝ, distance P (l (1, -1)) = tangent_length P C →
                   P ∈ E ∧ E = {P | P.2^2 = 8 * P.1}) ∧
  (∀ A B M : ℝ × ℝ, A ∈ {P : ℝ × ℝ | P.2^2 = 8 * P.1} →
                     B ∈ {P : ℝ × ℝ | P.2^2 = 8 * P.1} →
                     midpoint A B = M →
                     F = (2, 0) →
                     N = (3, 0) →
                     |distance A B - 2 * distance M N| = 6) :=
by sorry


end math_problem_equiv_proof_l776_776131


namespace sarah_total_weeds_l776_776318

def tuesday_weeds : ℕ := 25

def wednesday_weeds : ℕ := 3 * tuesday_weeds

def thursday_weeds : ℕ := (1 / 5 : ℚ) * wednesday_weeds

def friday_weeds : ℕ := thursday_weeds.to_nat - 10

def total_weeds : ℕ := tuesday_weeds + wednesday_weeds + thursday_weeds.to_nat + friday_weeds

theorem sarah_total_weeds : total_weeds = 120 := by
  sorry

end sarah_total_weeds_l776_776318


namespace range_of_a_relationship_two_over_sum_l776_776590

open Real

-- Definitions of conditions from part a)
def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x
def has_two_distinct_zeros (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f(a, x1) = 0 ∧ f(a, x2) = 0

-- Part I: Proving the range of 'a'
theorem range_of_a :
  ∀ (a : ℝ), has_two_distinct_zeros a ↔ (0 < a ∧ a < 1 / exp 1) :=
sorry

-- Part II: Proving the relationship between 
theorem relationship_two_over_sum (a x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : f(a, x1) = 0) (h4 : f(a, x2) = 0) :
  2 / (x1 + x2) < a :=
sorry

end range_of_a_relationship_two_over_sum_l776_776590


namespace least_three_digit_multiple_13_l776_776775

theorem least_three_digit_multiple_13 : 
  ∃ n : ℕ, (n ≥ 100) ∧ (∃ k : ℕ, n = 13 * k) ∧ ∀ m, m < n → (m < 100 ∨ ¬∃ k : ℕ, m = 13 * k) :=
by
  sorry

end least_three_digit_multiple_13_l776_776775


namespace expression_value_l776_776084

theorem expression_value (x : ℝ) (h : x = 3) : x^4 - 4 * x^2 = 45 := by
  sorry

end expression_value_l776_776084


namespace solution_set_f_l776_776159

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 2^(x - 1) - 2 else 2^(1 - x) - 2

theorem solution_set_f (x : ℝ) : 
  (1 ≤ x ∧ x ≤ 3) ↔ (f (x - 1) ≤ 0) :=
sorry

end solution_set_f_l776_776159


namespace pie_prices_l776_776874

theorem pie_prices 
  (d k : ℕ) (hd : 0 < d) (hk : 1 < k) 
  (p1 p2 : ℕ) 
  (prices : List ℕ) (h_prices : prices = [64, 64, 70, 72]) 
  (hP1 : (p1 + d) ∈ prices)
  (hZ1 : (k * p1) ∈ prices)
  (hP2 : (p2 + d) ∈ prices)
  (hZ2 : (k * p2) ∈ prices)
  (h_different_prices : p1 + d ≠ k * p1 ∧ p2 + d ≠ k * p2) : 
  ∃ p1 p2, p1 ≠ p2 :=
sorry

end pie_prices_l776_776874


namespace integer_triplet_solution_l776_776280

def circ (a b : ℤ) : ℤ := a + b - a * b

theorem integer_triplet_solution (x y z : ℤ) :
  circ (circ x y) z + circ (circ y z) x + circ (circ z x) y = 0 ↔
  (x = 0 ∧ y = 0 ∧ z = 2) ∨ (x = 0 ∧ y = 2 ∧ z = 0) ∨ (x = 2 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end integer_triplet_solution_l776_776280


namespace baseball_games_in_season_l776_776347

def games_per_month : ℕ := 7
def months_in_season : ℕ := 2
def total_games_in_season : ℕ := games_per_month * months_in_season

theorem baseball_games_in_season : total_games_in_season = 14 := by
  sorry

end baseball_games_in_season_l776_776347


namespace trapezoid_area_l776_776072

theorem trapezoid_area (l w : ℝ) (h1 : l * w = 96) 
  (E_mid_AB : ∃ E, E = (l / 2))
  (F_mid_BC : ∃ F, F = (w / 2))
  (G_mid_CD : ∃ G, G = l / 2)
  (H_mid_DA : ∃ H, H = w / 2):
  let A_trapezoid := (1 / 2) * (l / 2 + l / 2) * w in
  A_trapezoid = 48 :=
by
  sorry

end trapezoid_area_l776_776072


namespace go_stones_total_l776_776377

theorem go_stones_total (n : ℕ) (h1 : n^2 + 3 + 44 = (n + 2)^2) : n = 10 → (n^2 + 3) = 103 :=
by {
  intro hn,
  rw hn,
  norm_num
}

end go_stones_total_l776_776377


namespace num_divisors_of_30_l776_776198

-- Define what it means to be a divisor of 30
def is_divisor (n : ℤ) : Prop :=
  n ≠ 0 ∧ 30 % n = 0

-- Define a function that counts the number of divisors of 30
def count_divisors (d : ℤ) : ℕ :=
  ((Multiset.filter is_divisor ((List.range' (-30) 61).map (Int.ofNat))).attach.to_finset.card)

-- State the theorem
theorem num_divisors_of_30 : count_divisors 30 = 16 := sorry

end num_divisors_of_30_l776_776198


namespace ethel_subtracts_l776_776746

theorem ethel_subtracts (h : 50^2 = 2500) : 2500 - 99 = 49^2 :=
by
  sorry

end ethel_subtracts_l776_776746


namespace divisors_of_30_l776_776183

theorem divisors_of_30 : 
  ∃ n : ℕ, (∀ d : ℤ, d ∣ 30 → d > 0 → d ≤ 30) ∧ (∀ d : ℤ, d ∣ 30 → -d ∣ 30) ∧ (∀ d : ℤ, d ∣ 30 → -d = d → d = 0) ∧ 2 * (Finset.card (Finset.filter (λ d, 0 < d) (Finset.filter (λ d, d ∣ 30) (Finset.Icc 1 30).toFinset))) = 16 :=
by {
   sorry
}

end divisors_of_30_l776_776183


namespace coefficient_x3y5_expansion_l776_776359

open Real

theorem coefficient_x3y5_expansion :
  let x := (2:ℚ) / 3
  let y := -(1:ℚ) / 3
  (binomial 8 3) * (x ^ 3) * (y ^ 5) = - (448:ℚ) / 6561 :=
by
  let x := (2:ℚ) / 3
  let y := -(1:ℚ) / 3
  sorry

end coefficient_x3y5_expansion_l776_776359


namespace smaller_square_area_proof_right_triangle_area_proof_l776_776922

-- The conditions
def larger_square_area : ℝ := 100
def side_of_larger_square : ℝ := real.sqrt larger_square_area
def side_of_smaller_square : ℝ := side_of_larger_square / real.sqrt 2
def area_of_smaller_square : ℝ := real.sqrt (side_of_larger_square * side_of_larger_square / 2)
def leg_of_triangle : ℝ := side_of_larger_square / 2
def area_of_triangle : ℝ := (1 / 2) * leg_of_triangle * leg_of_triangle

-- The theorem statements
theorem smaller_square_area_proof :
  area_of_smaller_square = 50 := 
by
  sorry 

theorem right_triangle_area_proof :
  area_of_triangle = 12.5 := 
by
  sorry 

end smaller_square_area_proof_right_triangle_area_proof_l776_776922


namespace Travis_annual_cereal_cost_l776_776749

def cost_of_box_A : ℚ := 2.50
def cost_of_box_B : ℚ := 3.50
def cost_of_box_C : ℚ := 4.00
def cost_of_box_D : ℚ := 5.25
def cost_of_box_E : ℚ := 6.00

def quantity_of_box_A : ℚ := 1
def quantity_of_box_B : ℚ := 0.5
def quantity_of_box_C : ℚ := 0.25
def quantity_of_box_D : ℚ := 0.75
def quantity_of_box_E : ℚ := 1.5

def cost_week1 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  cost_of_box_B * quantity_of_box_B +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  cost_of_box_E * quantity_of_box_E

def cost_week2 : ℚ :=
  let subtotal := 
    cost_of_box_A * quantity_of_box_A +
    cost_of_box_B * quantity_of_box_B +
    cost_of_box_C * quantity_of_box_C +
    cost_of_box_D * quantity_of_box_D +
    cost_of_box_E * quantity_of_box_E
  subtotal * 0.8

def cost_week3 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  0 +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  cost_of_box_E * quantity_of_box_E

def cost_week4 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  cost_of_box_B * quantity_of_box_B +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  let discounted_box_E := cost_of_box_E * quantity_of_box_E * 0.85
  cost_of_box_A * quantity_of_box_A +
  discounted_box_E
  
def monthly_cost : ℚ :=
  cost_week1 + cost_week2 + cost_week3 + cost_week4

def annual_cost : ℚ :=
  monthly_cost * 12

theorem Travis_annual_cereal_cost :
  annual_cost = 792.24 := by
  sorry

end Travis_annual_cereal_cost_l776_776749


namespace votes_switched_l776_776243

theorem votes_switched (x : ℕ) (total_votes : ℕ) (half_votes : ℕ) 
  (votes_first_round : ℕ) (votes_second_round_winner : ℕ) (votes_second_round_loser : ℕ)
  (cond1 : total_votes = 48000)
  (cond2 : half_votes = total_votes / 2)
  (cond3 : votes_first_round = half_votes)
  (cond4 : votes_second_round_winner = half_votes + x)
  (cond5 : votes_second_round_loser = half_votes - x)
  (cond6 : votes_second_round_winner = 5 * votes_second_round_loser) :
  x = 16000 := by
  -- Proof will go here
  sorry

end votes_switched_l776_776243


namespace final_ball_theorem_l776_776800

def ball_type : Type := {A B C : Type}

structure initial_ball_counts :=
  (count_A : ℕ)
  (count_B : ℕ)
  (count_C : ℕ)

constant collision_rules : {A B C : ball_type} →
  ((A, A) → C) ∧
  ((B, B) → C) ∧
  ((C, C) → C) ∧
  ((A, B) → C) ∧
  ((A, C) → B) ∧
  ((B, C) → A)

def final_ball_type 
  (counts : initial_ball_counts)
  (collision : ball_type) : ball_type :=
sorry

theorem final_ball_theorem (counts : initial_ball_counts)
  (h : final_ball_type counts collision_rules ≠ ball_type.C) :
  (final_ball_type counts collision_rules = ball_type.A) ∨
  (final_ball_type counts collision_rules = ball_type.B) :=
by sorry

example : final_ball_theorem
  { count_A := 12, count_B := 9, count_C := 10 } collision_rules :=
by trivial

end final_ball_theorem_l776_776800


namespace Raven_age_l776_776229

-- Define the conditions
def Phoebe_age_current : Nat := 10
def Phoebe_age_in_5_years : Nat := Phoebe_age_current + 5

-- Define the hypothesis that in 5 years Raven will be 4 times as old as Phoebe
def Raven_in_5_years (R : Nat) : Prop := R + 5 = 4 * Phoebe_age_in_5_years

-- State the theorem to be proved
theorem Raven_age : ∃ R : Nat, Raven_in_5_years R ∧ R = 55 :=
by
  sorry

end Raven_age_l776_776229


namespace weight_of_replaced_person_l776_776327

theorem weight_of_replaced_person
  (avg_weight_increase : ℝ)
  (num_persons : ℕ)
  (new_person_weight : ℝ)
  : (weight_of_replaced_person : ℝ) :=
  assume h1 : avg_weight_increase = 3.2,
  assume h2 : num_persons = 10,
  assume h3 : new_person_weight = 97,
  weight_of_replaced_person = new_person_weight - avg_weight_increase * num_persons
  sorry

end weight_of_replaced_person_l776_776327


namespace interval_contains_nonempty_interval_not_in_S_l776_776265

noncomputable def sequence_positive_real (r : ℕ → ℝ) : Prop :=
  (∀ n, r n > 0) ∧ (tendsto r at_top (𝓝 0))

def sum_set (r : ℕ → ℝ) : set ℝ :=
  {s | ∃ (i : fin 1994 → ℕ), (strict_mono i) ∧ (s = ∑ k, r (i k))}

theorem interval_contains_nonempty_interval_not_in_S (r : ℕ → ℝ) (a b : ℝ) (h_seq : sequence_positive_real r) (h_interval : a < b) :
  ∃ c d : ℝ, a < c ∧ c < d ∧ d < b ∧ (∀ s ∈ sum_set r, s < c ∨ s > d) :=
begin
  sorry
end

end interval_contains_nonempty_interval_not_in_S_l776_776265


namespace value_of_expression_l776_776794

theorem value_of_expression :
  625 = 5^4 → 
  (625^(Real.log 5000 / Real.log 5))^(1 / 4) = 5000 :=
by
  intros h
  sorry

end value_of_expression_l776_776794


namespace age_difference_l776_776013

variable (x y z : ℝ)

def overall_age_condition (x y z : ℝ) : Prop := (x + y = y + z + 10)

theorem age_difference (x y z : ℝ) (h : overall_age_condition x y z) : (x - z) / 10 = 1 :=
  by
    sorry

end age_difference_l776_776013


namespace sqrt2_minus1_mul_sqrt2_plus1_eq1_l776_776064

theorem sqrt2_minus1_mul_sqrt2_plus1_eq1 : (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) = 1 :=
  sorry

end sqrt2_minus1_mul_sqrt2_plus1_eq1_l776_776064


namespace find_wholesale_prices_l776_776841

-- Definitions based on conditions
def bakerPlusPrice (p : ℕ) (d : ℕ) : ℕ := p + d
def starPrice (p : ℕ) (k : ℚ) : ℕ := (p * k).toNat

-- The tuple of prices given
def prices : List ℕ := [64, 64, 70, 72]

-- Statement of the problem
theorem find_wholesale_prices (d : ℕ) (k : ℚ) (h_d : d > 0) (h_k : k > 1 ∧ k.denom = 1):
  ∃ p1 p2, 
  (p1 ∈ prices ∧ bakerPlusPrice p1 d ∈ prices ∧ starPrice p1 k ∈ prices) ∧
  (p2 ∈ prices ∧ bakerPlusPrice p2 d ∈ prices ∧ starPrice p2 k ∈ prices) :=
sorry

end find_wholesale_prices_l776_776841


namespace asymptote_slope_positive_value_l776_776107

theorem asymptote_slope_positive_value :
  (∃ x y : ℝ, 
    sqrt ((x-1)^2+(y+1)^2) - sqrt ((x-7)^2+(y+1)^2) = 4) →
  (∃ m : ℝ, m = sqrt 5 / 2 ∧ m > 0) :=
  by
    sorry

end asymptote_slope_positive_value_l776_776107


namespace shares_sold_value_correct_l776_776041

-- Given conditions
def total_value_business : ℝ := 10000
def man's_ownership_fraction : ℝ := 1 / 3
def fraction_shares_sold : ℝ := 3 / 5

-- Calculate the value of the man's shares
def value_of_mans_shares : ℝ := man's_ownership_fraction * total_value_business

-- Calculate the value of the shares sold
def value_of_shares_sold : ℝ := fraction_shares_sold * value_of_mans_shares

-- Problem statement in Lean 4 to prove
theorem shares_sold_value_correct : value_of_shares_sold = 2000 := 
by
  sorry

end shares_sold_value_correct_l776_776041


namespace simplify_expression_l776_776319

theorem simplify_expression (x : ℝ) (h : x^2 - x - 1 = 0) :
  ( ( (x - 1) / x - (x - 2) / (x + 1) ) / ( (2 * x^2 - x) / (x^2 + 2 * x + 1) ) ) = 1 := 
by
  sorry

end simplify_expression_l776_776319


namespace max_attendance_days_l776_776936

structure PersonAvailability where
  name : String
  available : List Bool -- List of booleans representing availability from Monday to Friday
  deriving DecidableEq

-- Define each person's availability based on the conditions described
def anna := PersonAvailability.mk "Anna" [false, true, true, false, false]
def bill := PersonAvailability.mk "Bill" [true, false, true, false, true]
def carl := PersonAvailability.mk "Carl" [false, false, true, true, false]
def dana := PersonAvailability.mk "Dana" [true, true, false, false, true]

-- Function to count how many people are available on a given day
def countAvailability (day: Fin 5) (people: List PersonAvailability) : Nat :=
  people.countp (λ person => person.available.get day)

-- List of people
def people := [anna, bill, carl, dana]

-- The main theorem
theorem max_attendance_days :
  let counts := (List.map (λ day => countAvailability day people) (List.finRange 5));
  List.max' counts 2 = 2 → counts.get 2 = 2 ∧ counts.get 3 = 2 :=
by
  intros counts max_val
  have h1 : countAvailability 2 people = 2 := sorry
  have h2 : countAvailability 3 people = 2 := sorry
  exact ⟨h1, h2⟩

end max_attendance_days_l776_776936


namespace exists_subgraph_with_properties_l776_776088

-- Definitions of graph, edges, vertices, and degrees
structure Graph :=
  (V : Type)
  (E : set (V × V))
  (nonempty : V.nonempty)
  (has_edges : E.nonempty)

def min_degree (G : Graph) : ℕ := sorry -- Define the minimum degree of a graph (placeholder)
def avg_degree (G : Graph) : ℝ := 2 * (G.E.to_finset.card) / (G.V.to_finset.card : ℝ) -- The average degree

theorem exists_subgraph_with_properties (G : Graph) (h : G.has_edges) :
  ∃ H : Graph, H ⊆ G ∧ min_degree(H) > avg_degree(H) ∧ avg_degree(H) ≥ avg_degree(G) := sorry

end exists_subgraph_with_properties_l776_776088


namespace find_specific_M_in_S_l776_776269

section MatrixProgression

variable {R : Type*} [CommRing R]

-- Definition of arithmetic progression in a 2x2 matrix.
def is_arithmetic_progression (a b c d : R) : Prop :=
  ∃ r : R, b = a + r ∧ c = a + 2 * r ∧ d = a + 3 * r

-- Definition of set S.
def S : Set (Matrix (Fin 2) (Fin 2) R) :=
  { M | ∃ a b c d : R, M = ![![a, b], ![c, d]] ∧ is_arithmetic_progression a b c d }

-- Main problem statement
theorem find_specific_M_in_S (M : Matrix (Fin 2) (Fin 2) ℝ) (k : ℕ) :
  k > 1 → M ∈ S → ∃ (α : ℝ), (M = α • ![![1, 1], ![1, 1]] ∨ (M = α • ![![ -3, -1], ![1, 3]] ∧ Odd k)) :=
by
  sorry

end MatrixProgression

end find_specific_M_in_S_l776_776269


namespace area_of_quadrilateral_l776_776636

theorem area_of_quadrilateral (A B C D E : Type) 
  (point : A → Prop) (on_line : A → A → A → Prop)
  (AC_diag : on_line A C) (BD_diag : on_line B D)
  (point_e : point E) (angle_abc_90 : ∠ABC = 90°) (angle_acd_90 : ∠ACD = 90°)
  (AC_len : dist A C = 20) (CD_len : dist C D = 30) (AE_len : dist A E = 5) :
  area_of_quadrilateral A B C D = 360 :=
begin
  sorry
end

end area_of_quadrilateral_l776_776636


namespace right_triangle_larger_acute_angle_l776_776625

theorem right_triangle_larger_acute_angle (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < 90) 
  (h3 : ∑ x in finset.univ, x = 180) 
  (h4 : α + (α - 10) + 90 = 180) : α = 50 :=
sorry

end right_triangle_larger_acute_angle_l776_776625


namespace current_speed_l776_776707

theorem current_speed (c : ℝ) :
  (∀ d1 t1 u v, d1 = 20 ∧ t1 = 2 ∧ u = 6 ∧ v = c → d1 = t1 * (u + v))
  ∧ (∀ d2 t2 u w, d2 = 4 ∧ t2 = 2 ∧ u = 6 ∧ w = c → d2 = t2 * (u - w)) 
  → c = 4 :=
by 
  intros
  sorry

end current_speed_l776_776707


namespace boxes_calculation_l776_776004

theorem boxes_calculation (total_bottles : ℕ) (bottles_per_bag : ℕ) (bags_per_box : ℕ) (boxes : ℕ) :
  total_bottles = 8640 → bottles_per_bag = 12 → bags_per_box = 6 → boxes = total_bottles / (bottles_per_bag * bags_per_box) → boxes = 120 :=
by
  intros h_total h_bottles_per_bag h_bags_per_box h_boxes
  rw [h_total, h_bottles_per_bag, h_bags_per_box] at h_boxes
  norm_num at h_boxes
  exact h_boxes

end boxes_calculation_l776_776004


namespace series_sum_l776_776442

theorem series_sum :
  let a_1 := 2
  let d := 3
  let s := [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56]
  -- We define the sequence in list form for clarity
  (s.sum = 29) :=
by
  let a_1 := 2
  let d := 3
  let s := [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56]
  sorry

end series_sum_l776_776442


namespace find_angle_A_find_max_area_l776_776254

noncomputable def triangle (a b c A B C : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  A + B + C = π ∧
  sin A > 0 ∧ sin B > 0 ∧ sin C > 0

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h : triangle a b c A B C) :
  (2 * b * cos A = c * cos A + a * cos C) → A = π / 3 :=
by
  sorry

theorem find_max_area (a b c : ℝ) (A B C : ℝ) (h : triangle a b c A B C) :
  (2 * b * cos A = c * cos A + a * cos C) → a = 4 → (1/2 * b * c * sin A ≤ 4 * sqrt 3) :=
by
  sorry 

end find_angle_A_find_max_area_l776_776254


namespace Matt_income_from_plantation_l776_776670

noncomputable def plantation_income :=
  let plantation_area := 500 * 500  -- square feet
  let grams_peanuts_per_sq_ft := 50 -- grams
  let grams_peanut_butter_per_20g_peanuts := 5  -- grams
  let price_per_kg_peanut_butter := 10 -- $

  -- Total revenue calculation
  plantation_area * grams_peanuts_per_sq_ft * grams_peanut_butter_per_20g_peanuts /
  20 / 1000 * price_per_kg_peanut_butter

theorem Matt_income_from_plantation :
  plantation_income = 31250 := sorry

end Matt_income_from_plantation_l776_776670


namespace sample_data_properties_l776_776900

-- Definitions of the properties
variables {n : ℕ} {x : Fin n → ℝ}
variable [Ordered AddCommGroup ℝ]

noncomputable def average (x : Fin n → ℝ) : ℝ :=
  (x.sum id) / n

noncomputable def variance (x : Fin n → ℝ) : ℝ :=
  (x.sum (λ xi, (xi - average x) ^ 2)) / (n - 1)

noncomputable def range (x : Fin n → ℝ) : ℝ :=
  x.toList.maximum - x.toList.minimum

noncomputable def median (x : Fin n → ℝ) : ℝ :=
  let sorted_x := x.toList.qsort (≤)
  if n % 2 = 0 then (sorted_x.get (n/2-1) + sorted_x.get (n/2)) / 2
  else sorted_x.get ((n-1) / 2)

-- Properties after removing the minimum and maximum values
variables {k : ℕ} {y : Fin k → ℝ}
variable (h : k = n - 2)
variable (y_def : (y : Fin k → ℝ) = (x.erase x.maximum).erase x.minimum)

def average' : ℝ := average y
def variance' : ℝ := variance y
def range' : ℝ := range y
def median' : ℝ := median y

-- The theorem to be proved
theorem sample_data_properties
  (x_unique : Function.Injective x) (n_ineq : n > 2) :
  median x = median' ∧ variance x > variance' ∧ range x > range' :=
  sorry

end sample_data_properties_l776_776900


namespace binomial_coefficient_10_3_l776_776494

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l776_776494


namespace MF_eq_2MG_l776_776267

variables {A B C D E M F G : Type}
variables (h_cyclic : cyclic_quad A B C D)
variables (h_AE : line_intersect_at E (angle_bisector_internal ∠BAD) (line BC))
variables (h_M_mid : M = midpoint A E)
variables (h_F : line_intersect_at F (angle_bisector_external ∠BCD) (line AD))
variables (h_G : line_intersect_at G (line_through M F) (line AB))
variables (h_AB_eq_2AD : dist A B = 2 * dist A D)

theorem MF_eq_2MG : dist M F = 2 * dist M G :=
by
  sorry

end MF_eq_2MG_l776_776267


namespace binomial_coefficient_10_3_l776_776467

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l776_776467


namespace cannot_be_score_60_l776_776621

variable (scores : List ℝ)
def n : ℕ := 50
def mean : ℝ := 82
def variance : ℝ := 8.2

theorem cannot_be_score_60 (h1 : scores.length = n)
  (h2 : (scores.sum / n) = mean)
  (h3 : (scores.map (λ x, (x - mean)^2).sum / n) = variance) :
  ∀ score ∈ scores, score ≠ 60 :=
sorry

end cannot_be_score_60_l776_776621


namespace number_of_true_statements_l776_776710

def D (x : ℝ) : ℝ := if x ∈ ℚ then 1 else 0

theorem number_of_true_statements : 
    let prop1 := ∀ x : ℝ, D (-x) = D x in
    let prop2 := ∃ x0 : ℝ, D (D x0) = 0 in
    let prop3 := ∀ (x : ℝ) (T : ℚ), T ≠ 0 → D (x + T) = D x in
    let prop4 := ∃ (x1 x2 x3 : ℝ), D x1 ≠ D x2 ∧ D x1 ≠ D x3 ∧ 
        ∃ (h1 : x1 ≠ x2) (h2 : x2 ≠ x3), 
        let A := (x1, D x1) in
        let B := (x2, D x2) in
        let C := (x3, D x3) in
        (A.2 - B.2) ^ 2 + (A.1 - B.1) ^ 2 = (A.2 - C.2) ^ 2 + (A.1 - C.1) ^ 2 in
    ((prop1) → true) ∧ (¬ prop2) ∧ (prop3) ∧ (¬ prop4) → (2 : ℕ) :=
by
  intros
  apply 2
  sorry

end number_of_true_statements_l776_776710


namespace determine_wholesale_prices_l776_776848

variables (p1 p2 d k : ℝ)
variables (h1 : 1 < k)
variables (prices : Finset ℝ)
variables (p_plus : ℝ → ℝ) (star : ℝ → ℝ)

noncomputable def BakeryPrices : Prop :=
  p_plus p1 = p1 + d ∧ p_plus p2 = p2 + d ∧ 
  star p1 = k * p1 ∧ star p2 = k * p2 ∧ 
  prices = {64, 64, 70, 72} ∧ 
  ∃ (q1 q2 q3 q4 : ℝ), 
    (q1 = p_plus p1 ∨ q1 = star p1 ∨ q1 = p_plus p2 ∨ q1 = star p2) ∧
    (q2 = p_plus p1 ∨ q2 = star p1 ∨ q2 = p_plus p2 ∨ q2 = star p2) ∧
    (q3 = p_plus p1 ∨ q3 = star p1 ∨ q3 = p_plus p2 ∨ q3 = star p2) ∧
    (q4 = p_plus p1 ∨ q4 = star p1 ∨ q4 = p_plus p2 ∨ q4 = star p2) ∧
    prices = {q1, q2, q3, q4}

theorem determine_wholesale_prices (p1 p2 : ℝ) (d k : ℝ)
    (h1 : 1 < k) (prices : Finset ℝ) :
  BakeryPrices p1 p2 d k h1 prices (λ p, p + d) (λ p, k * p) :=
sorry

end determine_wholesale_prices_l776_776848


namespace average_of_solutions_l776_776215

open Real

theorem average_of_solutions :
  (∃ x : ℝ, sqrt (3 * x^2 + 4 * x + 1) = sqrt 37) → 
  (∃ avg : ℝ, avg = -2/3) :=
by
  intro h,
  sorry

end average_of_solutions_l776_776215


namespace sqrt_12_minus_sqrt_27_l776_776087

theorem sqrt_12_minus_sqrt_27 :
  (Real.sqrt 12 - Real.sqrt 27 = -Real.sqrt 3) := by
  sorry

end sqrt_12_minus_sqrt_27_l776_776087


namespace rectangle_perimeter_l776_776924

def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem rectangle_perimeter
  (a1 a2 a3 a4 a5 a6 a7 a8 a9 l w : ℕ)
  (h1 : a1 + a2 + a3 = a9)
  (h2 : a1 + a2 = a3)
  (h3 : a1 + a3 = a4)
  (h4 : a3 + a4 = a5)
  (h5 : a4 + a5 = a6)
  (h6 : a2 + a3 + a5 = a7)
  (h7 : a2 + a7 = a8)
  (h8 : a1 + a4 + a6 = a9)
  (h9 : a6 + a9 = a7 + a8)
  (h_rel_prime : relatively_prime l w)
  (h_dimensions : l = 61)
  (h_dimensions_w : w = 69) :
  2 * l + 2 * w = 260 := by
  sorry

end rectangle_perimeter_l776_776924


namespace team_won_five_games_l776_776025
-- Import the entire Mathlib library

-- Number of games played (given as a constant)
def numberOfGamesPlayed : ℕ := 10

-- Number of losses definition based on the ratio condition
def numberOfLosses : ℕ := numberOfGamesPlayed / 2

-- The number of wins is defined as the total games played minus the number of losses
def numberOfWins : ℕ := numberOfGamesPlayed - numberOfLosses

-- Proof statement: The number of wins is 5
theorem team_won_five_games :
  numberOfWins = 5 := by
  sorry

end team_won_five_games_l776_776025


namespace max_value_proof_l776_776956

noncomputable def max_value : ℝ := 
  let a := (Real.sqrt 3) / 3
  let b := (Real.sqrt 3) / 3
  let c := (Real.sqrt 3) / 3
  Real.min (1 / a) (Real.min (1 / (b^2)) (Real.min (1 / (c^3)) (a + b^2 + c^3)))

theorem max_value_proof : max_value = Real.sqrt 3 := sorry

#print axioms max_value_proof

end max_value_proof_l776_776956


namespace compare_fractions_l776_776981

variables {a b : ℝ}

theorem compare_fractions (h : a + b > 0) : 
  (a / (b^2)) + (b / (a^2)) ≥ (1 / a) + (1 / b) :=
sorry

end compare_fractions_l776_776981


namespace probability_same_length_l776_776275

/-- Defining the set of all sides and diagonals of a regular hexagon. -/
def T : Finset ℚ := sorry

/-- There are exactly 6 sides in the set T. -/
def sides_count : ℕ := 6

/-- There are exactly 9 diagonals in the set T. -/
def diagonals_count : ℕ := 9

/-- The total number of segments in the set T. -/
def total_segments : ℕ := sides_count + diagonals_count

theorem probability_same_length :
  let prob_side := (6 : ℚ) / total_segments * (5 / (total_segments - 1))
  let prob_diagonal := (9 : ℚ) / total_segments * (4 / (total_segments - 1))
  prob_side + prob_diagonal = 17 / 35 := 
by
  admit

end probability_same_length_l776_776275


namespace divisors_of_30_l776_776193

theorem divisors_of_30 : ∃ (n : ℕ), n = 16 ∧ (∀ d : ℤ, d ∣ 30 → (d ≤ 30 ∧ d ≥ -30)) :=
by
  sorry

end divisors_of_30_l776_776193


namespace total_tiles_needed_l776_776897

-- Definitions of the given conditions
def blue_tiles : Nat := 48
def red_tiles : Nat := 32
def additional_tiles_needed : Nat := 20

-- Statement to prove the total number of tiles needed to complete the pool
theorem total_tiles_needed : blue_tiles + red_tiles + additional_tiles_needed = 100 := by
  sorry

end total_tiles_needed_l776_776897


namespace trapezoid_area_sum_l776_776904

noncomputable def area_sum (a b c d : ℕ) : ℚ :=
  -- Assuming function for summing areas of valid trapezoid configurations
  sorry

theorem trapezoid_area_sum :
  let a := 4
  let b := 6
  let c := 8
  let d := 10
  is_trapezoid a b c d → 
  area_sum a b c d = 935 / 3 :=
by
  intros
  sorry

def is_trapezoid (a b c d : ℕ) : Prop :=
  -- Assuming some properties and conditions for a valid trapezoid
  sorry

end trapezoid_area_sum_l776_776904


namespace acute_angle_coincidence_l776_776744

theorem acute_angle_coincidence (α : ℝ) (k : ℤ) :
  0 < α ∧ α < 180 ∧ 9 * α = k * 360 + α → α = 45 ∨ α = 90 ∨ α = 135 :=
by
  sorry

end acute_angle_coincidence_l776_776744


namespace power_of_four_l776_776544

theorem power_of_four (x : ℕ) 
  (h_expr : (4^x) * (7^5) * (11^2) = (2^(2*x)) * (7^5) * (11^2)) 
  (h_pf : prime_factors_count ((4^x) * (7^5) * (11^2)) = 29) 
  : x = 11 :=
sorry

end power_of_four_l776_776544


namespace table_sum_at_least_half_n_l776_776629

theorem table_sum_at_least_half_n (n : ℕ) (A : Fin n × Fin n → ℕ)
  (h_zero_condition : ∀ i j, A (i, j) = 0 →
    (∑ k, A (i, k) ≥ n) ∧ (∑ k, A (k, j) ≥ n)) :
  (∑ i, ∑ j, A (i, j)) ≥ n / 2 := sorry

end table_sum_at_least_half_n_l776_776629


namespace uncle_jerry_total_tomatoes_l776_776352

def tomatoes_reaped_yesterday : ℕ := 120
def tomatoes_reaped_more_today : ℕ := 50

theorem uncle_jerry_total_tomatoes : 
  tomatoes_reaped_yesterday + (tomatoes_reaped_yesterday + tomatoes_reaped_more_today) = 290 :=
by 
  sorry

end uncle_jerry_total_tomatoes_l776_776352


namespace max_value_x_y_l776_776958

theorem max_value_x_y (x y : ℝ) 
  (h1 : log ((x^2 + y^2) / 2) y ≥ 1) 
  (h2 : (x, y) ≠ (0, 0)) 
  (h3 : x^2 + y^2 ≠ 2) : 
  x + y ≤ 1 + real.sqrt 2 := 
sorry

end max_value_x_y_l776_776958


namespace calculate_GU_l776_776248

variables {XYZ : Type} [EuclideanSpace XYZ] (X Y Z G U V : XYZ)
variables (GV ZV GU : ℝ)

def is_centroid_of (G : XYZ) (X Y Z U V : XYZ) : Prop :=
  is_midpoint U Y Z ∧ is_midpoint V X Y ∧ collinear {X, G, U} ∧ collinear {Z, G, V} ∧ dist G V = (1 / 3) * dist Z V ∧ dist G U = (1 / 3) * dist X U

theorem calculate_GU (hG : is_centroid_of G X Y Z U V) (hGV : GV = 4) (hZV : ZV = 12) : GU = 4 :=
by
  sorry

end calculate_GU_l776_776248


namespace equal_dihedral_angles_between_non_parallel_faces_l776_776313

noncomputable def regular_dodecahedron : Type := sorry

structure Face (d : regular_dodecahedron) := 
  (is_regular_pentagon : Prop)

structure DihedralAngle (f1 f2 : Face regular_dodecahedron) := 
  (angle : ℝ)
  (is_dihedral : Prop)

axiom symmetry_property : ∀ (f1 f2 : Face regular_dodecahedron), f1 ≠ f2 ∧ (f1.is_regular_pentagon) ∧ (f2.is_regular_pentagon) → 
  ∃ (angle_val : ℝ), ∀ (f3 f4 : Face regular_dodecahedron), (DihedralAngle f1 f3).angle = angle_val ∧ (DihedralAngle f2 f4).angle = angle_val

theorem equal_dihedral_angles_between_non_parallel_faces :  
  ∀ (f1 f2 : Face regular_dodecahedron), f1 ≠ f2 ∧ ¬(∃ (p : Face regular_dodecahedron), (p = f1) ∧ (f2 ≠ p)) → 
    (DihedralAngle f1 f2).is_dihedral -> (DihedralAngle f1 f2).angle = (DihedralAngle f2 f1).angle :=
by
  sorry

end equal_dihedral_angles_between_non_parallel_faces_l776_776313


namespace intersection_sum_coordinates_l776_776333

theorem intersection_sum_coordinates :
  let f := λ x : ℝ, x^3 - 4 * x + 3
  let g := λ x : ℝ, (8 - x^2) / 4
  let is_intersection x := f x = g x
  let roots := { x : ℝ | is_intersection x }
  let A := ∑ x in roots, x
  let B := ∑ x in roots, 2 - x^2 / 4
  A = -1/4 ∧ B = 8 - (∑ x in roots, x^2) / 4 := by
  sorry

end intersection_sum_coordinates_l776_776333


namespace rectangle_shorter_side_l776_776755

theorem rectangle_shorter_side (b : ℝ) (h_b_pos : b > 0) (h_b_12 : b = 12) : 
  ∃ s : ℝ, s = 4 * Real.sqrt 3 ∧ ∃ (rectangle : set (ℝ × ℝ)), rectangle = {p | (p.1 = 0 ∨ p.1 = b) ∧ (p.2 = 0 ∨ p.2 = s)} := 
by
  sorry

end rectangle_shorter_side_l776_776755


namespace min_value_inverse_sum_l776_776560

variable {x y : ℝ}

theorem min_value_inverse_sum (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : (1/x + 1/y) ≥ 1 :=
  sorry

end min_value_inverse_sum_l776_776560


namespace area_of_remaining_figure_l776_776553

theorem area_of_remaining_figure :
  let side_length := 1
  let total_area := 6 * 6
  let dark_gray_area := 1 * 3
  let light_gray_area := 2 * 3
in total_area - dark_gray_area - light_gray_area = 27 := 
by
  sorry

end area_of_remaining_figure_l776_776553


namespace coefficient_of_x3y5_in_binomial_expansion_l776_776362

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  nat.choose n k

theorem coefficient_of_x3y5_in_binomial_expansion :
  let n := 8;
  let k := 3;
  let a := (2 : ℚ) / 3;
  let b := -(1 : ℚ) / 3;
  let x := 1;  -- Coefficients only involve rational numbers
  let y := 1;
  binomial_coefficient n k * (a ^ k) * (b ^ (n - k)) = -448 / 6561 :=
by
  sorry

end coefficient_of_x3y5_in_binomial_expansion_l776_776362


namespace red_pairs_count_l776_776914

theorem red_pairs_count (students_green : ℕ) (students_red : ℕ) (total_students : ℕ) (total_pairs : ℕ)
(pairs_green_green : ℕ) : 
students_green = 63 →
students_red = 69 →
total_students = 132 →
total_pairs = 66 →
pairs_green_green = 21 →
∃ (pairs_red_red : ℕ), pairs_red_red = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end red_pairs_count_l776_776914


namespace sample_variance_is_five_l776_776343
-- Import the necessary library

-- Define the sample list
def sample : List ℝ := [1, 3, 5, 7]

-- Define the mean of the sample
def mean (l : List ℝ) : ℝ := (l.foldl (· + ·) 0) / (l.length)

-- variance function
def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).foldl (· + ·) 0 / l.length

-- The theorem to prove
theorem sample_variance_is_five : variance sample = 5 := by
  sorry

end sample_variance_is_five_l776_776343


namespace simplify_expression_l776_776065

theorem simplify_expression : 
  (Real.sqrt 12) + (Real.sqrt 4) * ((Real.sqrt 5 - Real.pi) ^ 0) - (abs (-2 * Real.sqrt 3)) = 2 := 
by 
  sorry

end simplify_expression_l776_776065


namespace double_area_divisible_by_n_l776_776278

variable (n : ℕ) (is_odd : Odd n) (k : ℕ) 
          (A : Fin k → ℤ × ℤ) (is_convex : ConvexPolygon A)
          (lying_on_circle : LyingOnCircle A) (side_lengths_square_div_n : ∀ i j, ∃ m, A i ≠ A j ∧ (distance (A i) (A j))^2 = n * m)

def lying_on_circle (A : Fin k → ℤ × ℤ) : Prop :=
  ∃ O : (ℚ × ℚ), ∃ r : ℚ, ∀ i, dist (A i) O = r

def distance (p q : ℤ × ℤ) : ℚ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 : ℤ)

def ConvexPolygon (A : Fin k → ℤ × ℤ) : Prop :=
  ∀ i j, i ≠ j → ∃ (c : ℝ), 0 < c ∧ (intersection_point (A i) (A j) A (≠ a) c)
  -- Above is a placeholder. Exact definition depends on formalizing geometry.

theorem double_area_divisible_by_n : 
  (2 * polygon_area A : ℕ) ∣ n :=
sorry

def polygon_area (A : Fin k → ℤ × ℤ) : ℝ :=
  0.5 * | (∑ i in Fin.range (k - 1), (x (A i) * y (A (i + 1)) - y (A i) * x (A (i + 1)))) +
         (x (A (k-1)) * y (A 0) - y (A (k-1)) * x (A 0)) |
  -- Shoelace formula
  -- Assume x and y extract coordinates from A.

end double_area_divisible_by_n_l776_776278


namespace complex_div_i_l776_776630

open Complex

theorem complex_div_i (z : ℂ) (hz : z = -2 - i) : z / i = -1 + 2 * i :=
by
  sorry

end complex_div_i_l776_776630


namespace puppy_price_l776_776896

theorem puppy_price (P : ℕ) (kittens_price : ℕ) (total_earnings : ℕ) :
  (kittens_price = 2 * 6) → (total_earnings = 17) → (kittens_price + P = total_earnings) → P = 5 :=
by
  intros h1 h2 h3
  sorry

end puppy_price_l776_776896


namespace fixed_interval_range_t_l776_776334

-- Define the conditions as constraints
def f (x : ℝ) (t : ℝ) : ℝ := |2^x - t|
def g (x : ℝ) (t : ℝ) : ℝ := f (-x) t

-- The theorem statement addressing the range of t
theorem fixed_interval_range_t [∀ x : ℝ, f x t = |2^x - t|] [∀ x : ℝ, g x t = |2^(-x) - t|]
  (a b : ℝ) (h_monotonicity : ∀ x : ℝ, x ∈ set.Icc (a) (b) → (2^x - t) * (2^(-x) - t) ≤ 0)
  (fixed_interval : set.Icc 1 2 ⊆ set.Icc a b) : 
  set.Icc (1/2) 2 = { t | (1/2) ≤ t ∧ t ≤ 2 } :=
by sorry

end fixed_interval_range_t_l776_776334


namespace matchstick_ratio_is_one_half_l776_776296

def matchsticks_used (houses : ℕ) (matchsticks_per_house : ℕ) : ℕ :=
  houses * matchsticks_per_house

def ratio (a b : ℕ) : ℚ := a / b

def michael_original_matchsticks : ℕ := 600
def michael_houses : ℕ := 30
def matchsticks_per_house : ℕ := 10
def michael_used_matchsticks : ℕ := matchsticks_used michael_houses matchsticks_per_house

theorem matchstick_ratio_is_one_half :
  ratio michael_used_matchsticks michael_original_matchsticks = 1 / 2 :=
by
  sorry

end matchstick_ratio_is_one_half_l776_776296


namespace determine_y_l776_776632

theorem determine_y (EF GH : line) (EMF CMN GNM : Angle) 
  (hEF : EF.is_straight) (hGH : GH.is_straight)
  (hEMF : ∠EMF = 70) (hCMN : ∠CMN = 40) (hGNM : ∠GNM = 130) :
    y = 60 := sorry

end determine_y_l776_776632


namespace binomial_coefficient_10_3_l776_776470

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l776_776470


namespace coefficient_x3y5_expansion_l776_776361

open Real

theorem coefficient_x3y5_expansion :
  let x := (2:ℚ) / 3
  let y := -(1:ℚ) / 3
  (binomial 8 3) * (x ^ 3) * (y ^ 5) = - (448:ℚ) / 6561 :=
by
  let x := (2:ℚ) / 3
  let y := -(1:ℚ) / 3
  sorry

end coefficient_x3y5_expansion_l776_776361


namespace determine_wholesale_prices_l776_776828

-- Definitions and Conditions
-- Wholesale prices for days 1 and 2 are defined as p1 and p2
def wholesale_price_day1 : ℝ := p1
def wholesale_price_day2 : ℝ := p2

-- The price increment at "Baker Plus" store is a constant d
def price_increment_baker_plus : ℝ := d
-- The factor at "Star" store is a constant k > 1
def price_factor_star : ℝ := k
-- Given prices for two days: 64, 64, 70, 72 rubles
def given_prices : set ℝ := {64, 64, 70, 72}

-- Prices at the stores
def baker_plus_day1_price := wholesale_price_day1 + price_increment_baker_plus
def star_day1_price := wholesale_price_day1 * price_factor_star

def baker_plus_day2_price := wholesale_price_day2 + price_increment_baker_plus
def star_day2_price := wholesale_price_day2 * price_factor_star

-- Theorem to determine the wholesale prices for each day (p1 and p2)
theorem determine_wholesale_prices
    (baker_plus_day1_price ∈ given_prices)
    (star_day1_price ∈ given_prices)
    (baker_plus_day2_price ∈ given_prices)
    (star_day2_price ∈ given_prices)
    (h1 : baker_plus_day1_price ≠ star_day1_price)
    (h2 : baker_plus_day2_price ≠ star_day2_price) :
  (∃ p1 p2 d k > 1, 
      (p1 + d ∈ given_prices) ∧ (p1 * k ∈ given_prices) ∧
      (p2 + d ∈ given_prices) ∧ (p2 * k ∈ given_prices) ∧
      (p1 ≠ p2)) :=
  sorry

end determine_wholesale_prices_l776_776828


namespace Matt_income_from_plantation_l776_776669

noncomputable def plantation_income :=
  let plantation_area := 500 * 500  -- square feet
  let grams_peanuts_per_sq_ft := 50 -- grams
  let grams_peanut_butter_per_20g_peanuts := 5  -- grams
  let price_per_kg_peanut_butter := 10 -- $

  -- Total revenue calculation
  plantation_area * grams_peanuts_per_sq_ft * grams_peanut_butter_per_20g_peanuts /
  20 / 1000 * price_per_kg_peanut_butter

theorem Matt_income_from_plantation :
  plantation_income = 31250 := sorry

end Matt_income_from_plantation_l776_776669


namespace new_average_l776_776012

theorem new_average (avg : ℕ) (n : ℕ) (k : ℕ) (new_avg : ℕ) 
  (h1 : avg = 23) (h2 : n = 10) (h3 : k = 4) : 
  new_avg = (n * avg + n * k) / n → new_avg = 27 :=
by
  intro H
  sorry

end new_average_l776_776012


namespace triangle_angle_sum_l776_776240

theorem triangle_angle_sum (a b : ℝ) (ha : a = 40) (hb : b = 60) : ∃ x : ℝ, x = 180 - (a + b) :=
by
  use 80
  sorry

end triangle_angle_sum_l776_776240


namespace carol_rectangle_width_l776_776921

theorem carol_rectangle_width :
  ∃ W : ℝ, (12 * W = 6 * 30) → W = 15 :=
by
  use 15
  intros h
  rw [mul_comm 12, mul_comm 6] at h
  linarith only [h]

end carol_rectangle_width_l776_776921


namespace pie_prices_l776_776875

theorem pie_prices 
  (d k : ℕ) (hd : 0 < d) (hk : 1 < k) 
  (p1 p2 : ℕ) 
  (prices : List ℕ) (h_prices : prices = [64, 64, 70, 72]) 
  (hP1 : (p1 + d) ∈ prices)
  (hZ1 : (k * p1) ∈ prices)
  (hP2 : (p2 + d) ∈ prices)
  (hZ2 : (k * p2) ∈ prices)
  (h_different_prices : p1 + d ≠ k * p1 ∧ p2 + d ≠ k * p2) : 
  ∃ p1 p2, p1 ≠ p2 :=
sorry

end pie_prices_l776_776875


namespace solve_puzzle_l776_776605

theorem solve_puzzle
  (EH OY AY OH : ℕ)
  (h1 : EH = 4 * OY)
  (h2 : AY = 4 * OH) :
  EH + OY + AY + OH = 150 :=
sorry

end solve_puzzle_l776_776605


namespace sum_integers_where_mean_median_of_set_is_int_l776_776964

theorem sum_integers_where_mean_median_of_set_is_int :
  ∑ n in {n : ℕ | n < 20 ∧ n % 4 = 1}, n = 45 :=
by
  sorry

end sum_integers_where_mean_median_of_set_is_int_l776_776964


namespace proof_math_problem_l776_776895

-- Define the conditions
structure Conditions where
  person1_start_noon : ℕ -- Person 1 starts from Appleminster at 12:00 PM
  person2_start_2pm : ℕ -- Person 2 starts from Boniham at 2:00 PM
  meet_time : ℕ -- They meet at 4:55 PM
  finish_time_simultaneously : Bool -- They finish their journey simultaneously

-- Define the problem
def math_problem (c : Conditions) : Prop :=
  let arrival_time := 7 * 60 -- 7:00 PM in minutes
  c.person1_start_noon = 0 ∧ -- Noon as 0 minutes (12:00 PM)
  c.person2_start_2pm = 120 ∧ -- 2:00 PM as 120 minutes
  c.meet_time = 295 ∧ -- 4:55 PM as 295 minutes
  c.finish_time_simultaneously = true → arrival_time = 420 -- 7:00 PM in minutes

-- Prove the problem statement, skipping actual proof
theorem proof_math_problem (c : Conditions) : math_problem c :=
  by sorry

end proof_math_problem_l776_776895


namespace alice_frisbee_probability_l776_776428

/-- Alice and Bob play a game with a frisbee. On each turn, if Alice has the frisbee, 
there is a 2/3 chance that she will toss it to Bob and a 1/3 chance that she will keep it. 
If Bob has the frisbee, there is a 1/4 chance that he will toss it to Alice, 
and if he doesn't toss it, he keeps it. Alice starts with the frisbee.
This theorem states that the probability of Alice having the frisbee after three turns is 35/72. -/
theorem alice_frisbee_probability : 
  let PAtoB := 2/3,
      PAkeep := 1/3,
      PBkeep := 3/4 in
  PAtoB * PBkeep^2 + PAkeep^2 = 35/72 := 
by 
  let PAtoB := 2/3
  let PAkeep := 1/3
  let PBkeep := 3/4
  let PBtoA := 1 - PBkeep
  sorry

end alice_frisbee_probability_l776_776428


namespace cube_cross_section_S_squared_l776_776619

noncomputable def cube_side_length := sorry

/-- Vertices of the cube -/
def A := (0, 0, 0)
def B := (cube_side_length, 0, 0)
def G := (cube_side_length, cube_side_length, cube_side_length)
def H := (0, cube_side_length, cube_side_length)

/-- Midpoints of the segments -/
def K := ((cube_side_length / 2), 0, 0)
def L := (cube_side_length, (cube_side_length / 2), (cube_side_length / 2))

def area_face_of_cube := cube_side_length ^ 2

def S_squared : ℚ := ( 9 / 16 : ℚ )

theorem cube_cross_section_S_squared (s : ℚ) (cube : cube_side_length = s) :
  S_squared = 9 / 16 := sorry

end cube_cross_section_S_squared_l776_776619


namespace lychees_remaining_l776_776677
-- Definitions of the given conditions
def initial_lychees : ℕ := 500
def sold_lychees : ℕ := initial_lychees / 2
def home_lychees : ℕ := initial_lychees - sold_lychees
def eaten_lychees : ℕ := (3 * home_lychees) / 5

-- Statement to prove
theorem lychees_remaining : home_lychees - eaten_lychees = 100 := by
  sorry

end lychees_remaining_l776_776677


namespace max_sign_changes_l776_776646

theorem max_sign_changes (n : ℕ) (h : n > 0) (a : Fin n → ℝ) :
  ∃ seq : ℕ → Fin n → ℝ, 
    (∀ t : ℕ, seq t = (λ i : Fin n, if i = 0 then (seq (t - 1) 0 + seq (t - 1) 1) / 2 
                        else if i = n - 1 then (seq (t - 1) (n - 2) + seq (t - 1) (n - 1)) / 2
                        else (seq (t - 1) (i - 1) + seq (t - 1) i + seq (t - 1) (i + 1)) / 3)) ∧
    (∀ t : ℕ, seq 0 = a) ∧
    (∀ t : ℕ, seq t 0 * seq (t + 1) 0 < 0) ∧
    (∀ s : ℕ, ∃ t : ℕ, s = t → s < n - 1) ∧
    (∀ s : ℕ, ∀ t : ℕ, seq s 0 = seq t 0 → s = t). :=
sorry

end max_sign_changes_l776_776646


namespace arithmetic_series_sum_after_multiplication_l776_776920

theorem arithmetic_series_sum_after_multiplication :
  let s : List ℕ := [110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
  3 * s.sum = 3435 := by
  sorry

end arithmetic_series_sum_after_multiplication_l776_776920


namespace sum_of_cubes_mod_7_l776_776539

theorem sum_of_cubes_mod_7 :
  (∑ k in Finset.range 150, (k + 1) ^ 3) % 7 = 1 := 
sorry

end sum_of_cubes_mod_7_l776_776539


namespace solution_set_l776_776576

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then
  Real.exp x + Real.sin x
else 
  f -x

theorem solution_set (f_even : ∀ x : ℝ, f x = f (-x)) (h : ∀ x ≥ 0, f x = Real.exp x + Real.sin x) :
  {x : ℝ | f (2 * x - 1) < Real.exp Real.pi} = {x : ℝ | (1 - Real.pi) / 2 < x ∧ x < (1 + Real.pi) / 2} :=
sorry

end solution_set_l776_776576


namespace Matt_income_from_plantation_l776_776671

noncomputable def plantation_income :=
  let plantation_area := 500 * 500  -- square feet
  let grams_peanuts_per_sq_ft := 50 -- grams
  let grams_peanut_butter_per_20g_peanuts := 5  -- grams
  let price_per_kg_peanut_butter := 10 -- $

  -- Total revenue calculation
  plantation_area * grams_peanuts_per_sq_ft * grams_peanut_butter_per_20g_peanuts /
  20 / 1000 * price_per_kg_peanut_butter

theorem Matt_income_from_plantation :
  plantation_income = 31250 := sorry

end Matt_income_from_plantation_l776_776671


namespace max_min_diff_F_eq_24_l776_776578

theorem max_min_diff_F_eq_24 (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 4 * x + 1 / x + y + 9 / y = 26) :
  let F := 4 * x + y in
  (∀ F1 F2, (F1 = 4 * x + y ∧ F2 = 4 * x + y) → (F1 ≤ 25 ∧ F1 ≥ 1) → (F2 ≤ 25 ∧ F2 ≥ 1) → 
  (max F1 F2 - min F1 F2 = 24)) :=
sorry

end max_min_diff_F_eq_24_l776_776578


namespace binomial_coefficient_10_3_l776_776491

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l776_776491


namespace find_other_number_l776_776711

-- Definitions of LCM and HCF conditions.
variables (a b : ℕ)
def LCM := 2310
def HCF := 30

-- Known condition that one of the numbers is 210.
def known_number := 210

-- The corresponding Lean 4 statement:
theorem find_other_number (h_lcm : nat.lcm a b = LCM)
                         (h_hcf : nat.gcd a b = HCF)
                         (h_one_known : a = known_number) : b = 330 :=
sorry

end find_other_number_l776_776711


namespace divisors_of_30_count_l776_776187

theorem divisors_of_30_count : 
  ∃ S : Set ℤ, (∀ d ∈ S, 30 % d = 0) ∧ S.card = 16 :=
by
  sorry

end divisors_of_30_count_l776_776187


namespace solve_integer_divisibility_l776_776944

theorem solve_integer_divisibility :
  {n : ℕ | n < 589 ∧ 589 ∣ (n^2 + n + 1)} = {49, 216, 315, 482} :=
by
  sorry

end solve_integer_divisibility_l776_776944


namespace slope_of_AB_l776_776998

variable (x1 x2 y1 y2 : ℝ)

def ellipse (x y : ℝ) : Prop :=
  x^2 + (y^2)/2 = 1

def midpoint (x1 y1 x2 y2 k : ℝ) (xs ys : ℝ) : Prop :=
  ks = -1 ∧
  xs = (x1 + x2)/2 ∧
  ys = (y1 + y2)/2

theorem slope_of_AB (x1 x2 y1 y2 : ℝ) :
  (midpoint x1 y1 x2 y2 -1 1/2 1) →
  (ellipse x1 y1) →
  (ellipse x2 y2) →
  ((y1 - y2) / (x1 - x2)) = -1 :=
  sorry

end slope_of_AB_l776_776998


namespace inequality_for_an_l776_776218

open Nat

def a (n : ℕ) : ℝ := ∑ k in Finset.range n, 1 / (k + 1)

theorem inequality_for_an (n : ℕ) (h : n ≥ 2) :
  (a n) ^ 2 > 2 * ∑ k in Finset.range (n - 1), (a (k + 2)) / (k + 2) := 
sorry

end inequality_for_an_l776_776218


namespace bakery_wholesale_price_exists_l776_776857

theorem bakery_wholesale_price_exists :
  ∃ (p1 p2 d k : ℕ), (d > 0) ∧ (k > 1) ∧
  ({p1 + d, p2 + d, k * p1, k * p2} = {64, 64, 70, 72}) :=
by sorry

end bakery_wholesale_price_exists_l776_776857


namespace geometry_problem_l776_776562

section circle_geometry

variable {O : Type} [MetricSpace O] (r : ℝ := 10)
variable {A B : O} (hAOB : dist A B = 10)

noncomputable def central_angle (r : ℝ) (hAOB : dist A B = 10) : ℝ :=
  real.arccos ((dist A O ^ 2 + dist B O ^ 2 - dist A B ^ 2) / (2 * dist A O * dist B O))

noncomputable def arc_length (α : ℝ) (r : ℝ) : ℝ :=
  α * r

noncomputable def sector_area (α : ℝ) (r : ℝ) : ℝ :=
  (1 / 2) * α * r ^ 2

noncomputable def triangle_area (r : ℝ) (hAOB : dist A B = 10) : ℝ :=
  (1 / 2) * dist A B * (sqrt (r ^ 2 - (dist A B / 2) ^ 2))

noncomputable def sector_minus_triangle_area (r : ℝ) (hAOB : dist A B = 10) : ℝ :=
  (sector_area (central_angle r hAOB) r) - (triangle_area r hAOB)

theorem geometry_problem (r : ℝ := 10) (hAOB : dist A B = 10) :
  central_angle r hAOB = π / 3 ∧
  arc_length (π / 3) r = 10 * π / 3 ∧
  sector_area (π / 3) r = 50 * π / 3 ∧
  sector_minus_triangle_area r hAOB = 50 * (π / 3 - sqrt 3 / 2) :=
by
  sorry

end circle_geometry

end geometry_problem_l776_776562


namespace divisors_of_30_l776_776181

theorem divisors_of_30 : 
  ∃ n : ℕ, (∀ d : ℤ, d ∣ 30 → d > 0 → d ≤ 30) ∧ (∀ d : ℤ, d ∣ 30 → -d ∣ 30) ∧ (∀ d : ℤ, d ∣ 30 → -d = d → d = 0) ∧ 2 * (Finset.card (Finset.filter (λ d, 0 < d) (Finset.filter (λ d, d ∣ 30) (Finset.Icc 1 30).toFinset))) = 16 :=
by {
   sorry
}

end divisors_of_30_l776_776181


namespace sufficient_but_not_necessary_l776_776143

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1 → 2 * x^2 + x - 1 > 0) ∧ (¬(2 * x^2 + x - 1 > 0 → x < -1)) :=
by {
  -- Proving the condition that x < -1 implies 2x^2 + x - 1 > 0
  { intro h,
    calc
      2 * x^2 + x - 1 > 0  : sorry },
  -- Proving that it is not necessary by showing there are other intervals where 2x^2 + x - 1 > 0
  { intro h,
    sorry }
}

end sufficient_but_not_necessary_l776_776143


namespace arithmetic_seq_condition_l776_776176

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

noncomputable def sufficient_not_necessary (a b : ℕ → ℝ) : Prop :=
(∀ n : ℕ, b n = a n + a (n + 1)) →
(is_arithmetic_sequence a → is_arithmetic_sequence b) ∧
(¬ is_arithmetic_sequence b → ¬ is_arithmetic_sequence a)

theorem arithmetic_seq_condition (a b : ℕ → ℝ) :
sufficient_not_necessary a b :=
by
  intro h
  split
  sorry
  sorry

end arithmetic_seq_condition_l776_776176


namespace quadrilateral_is_rhombus_if_center_of_inscribed_circle_and_intersection_of_diagonals_l776_776689

-- Define the properties of inscribed circle, intersection of diagonals, and a rhombus
variables {A B C D O : Type} [metric_space A] [metric_space B] [metric_space C]
  [metric_space D] [metric_space O] [plane_geometry A B C D O]

-- Given conditions
def is_center_of_inscribed_circle (O : point) (ABCD : quadrilateral) : Prop := sorry
def is_intersection_of_diagonals (O : point) (ABCD : quadrilateral) : Prop := sorry
def is_rhombus (ABCD : quadrilateral) : Prop := sorry

-- The statement of the proof problem
theorem quadrilateral_is_rhombus_if_center_of_inscribed_circle_and_intersection_of_diagonals
  (ABCD : quadrilateral) (O : point) :
  is_center_of_inscribed_circle O ABCD →
  is_intersection_of_diagonals O ABCD →
  is_rhombus ABCD :=
sorry

end quadrilateral_is_rhombus_if_center_of_inscribed_circle_and_intersection_of_diagonals_l776_776689


namespace sum_cubes_mod_7_l776_776540

theorem sum_cubes_mod_7 :
  (∑ i in Finset.range 151, i ^ 3) % 7 = 0 := by
  sorry

end sum_cubes_mod_7_l776_776540


namespace problem_expression_bounds_l776_776283

theorem problem_expression_bounds (x y z w : ℝ) (hx : 0 ≤ x) (hx₁ : x ≤ 1)
    (hy : 0 ≤ y) (hy₁ : y ≤ 1)
    (hz : 0 ≤ z) (hz₁ : z ≤ 1)
    (hw : 0 ≤ w) (hw₁ : w ≤ 1) :
    2 * real.sqrt 2 ≤ real.sqrt (x ^ 2 + (1 - y) ^ 2) + real.sqrt (y ^ 2 + (1 - z) ^ 2) + real.sqrt (z ^ 2 + (1 - w) ^ 2) + real.sqrt (w ^ 2 + (1 - x) ^ 2) 
    ∧ real.sqrt (x ^ 2 + (1 - y) ^ 2) + real.sqrt (y ^ 2 + (1 - z) ^ 2) + real.sqrt (z ^ 2 + (1 - w) ^ 2) + real.sqrt (w ^ 2 + (1 - x) ^ 2) ≤ 4 :=
by
  sorry

end problem_expression_bounds_l776_776283


namespace circle_projections_distances_l776_776129

theorem circle_projections_distances (r u v : ℤ) (p q : ℤ) (m n : ℕ)
  (h_circle_eq : u^2 + v^2 = r^2)
  (h_odd_r : r % 2 = 1)
  (h_prime_p : p.prime)
  (h_prime_q : q.prime)
  (h_u : u = p^m)
  (h_v : v = q^n)
  (h_u_gt_v : u > v) :
  (abs (r - u) = 1) ∧ (abs (-r - u) = 9) ∧ (abs (-r - v) = 8) ∧ (abs (r - v) = 2) := sorry

end circle_projections_distances_l776_776129


namespace cube_edge_assignment_impossible_l776_776259

theorem cube_edge_assignment_impossible :
  ∀ (edge_values : Fin 12 → ℕ),
  (∀ i, edge_values i ∈ Finset.range 13) → -- Values from 1 to 12
  (∑ i, edge_values i = 78) → -- The sum of edge values is 78
  ¬(∃ k, ∀ v : Fin 8,
    (Finset.sum (Finset.univ : Finset (Fin 3)) (fun (_ : Fin 3) => edge_values (1 : Fin 12)) = k)) -- Vertices sum to a common value
 :=
sorry

end cube_edge_assignment_impossible_l776_776259


namespace complex_number_equality_l776_776971

theorem complex_number_equality (m n : ℝ) (i : ℂ) (h : i = complex.I) (h_eq : ((↑m + 2 * i) / i) = (↑n + i)) :
  n - m = 3 := 
by
  sorry

end complex_number_equality_l776_776971


namespace num_pi1_numbers_eq_2142_l776_776024

theorem num_pi1_numbers_eq_2142 :
  let pi1_numbers := {n : ℕ | 10000 ≤ n ∧ n ≤ 99999 ∧ 
                        let digits := (n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10) in
                        (digits.1 < digits.2) ∧ (digits.2 < digits.3) ∧ (digits.3 > digits.4) ∧ (digits.4 > digits.5)} in
  pi1_numbers.card = 2142 :=
by
  sorry

end num_pi1_numbers_eq_2142_l776_776024


namespace compute_binomial_10_3_eq_120_l776_776485

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l776_776485


namespace largest_binom_coeff_l776_776735

theorem largest_binom_coeff (n : ℕ) (hn : 0 < n) : 
  ∀ k, (n = k ∨ n = k + 1) ↔
    ∀ t, binomial (2 * n - 1) k ≥ binomial (2 * n - 1) t :=
sorry

end largest_binom_coeff_l776_776735


namespace x_power_12_l776_776142

theorem x_power_12 (x : ℝ) (h : x + 1 / x = 2) : x^12 = 1 :=
by sorry

end x_power_12_l776_776142


namespace equal_division_of_cookie_l776_776382

theorem equal_division_of_cookie (total_area : ℝ) (friends : ℕ) (area_per_person : ℝ) 
  (h1 : total_area = 81.12) 
  (h2 : friends = 6) 
  (h3 : area_per_person = total_area / friends) : 
  area_per_person = 13.52 :=
by 
  sorry

end equal_division_of_cookie_l776_776382


namespace find_t50_l776_776331

def sequence (t : ℕ → ℤ) : Prop :=
  (t 1 = 5) ∧ (∀ n, n ≥ 2 → t n - t (n - 1) = 2 * (n : ℤ) + 3)

theorem find_t50 (t : ℕ → ℤ) (h_seq : sequence t) : t 50 = 2700 :=
by
  sorry

end find_t50_l776_776331


namespace number_of_sunflowers_l776_776640

noncomputable def cost_per_red_rose : ℝ := 1.5
noncomputable def cost_per_sunflower : ℝ := 3
noncomputable def total_cost : ℝ := 45
noncomputable def cost_of_red_roses : ℝ := 24 * cost_per_red_rose
noncomputable def money_left_for_sunflowers : ℝ := total_cost - cost_of_red_roses

theorem number_of_sunflowers :
  (money_left_for_sunflowers / cost_per_sunflower) = 3 :=
by
  sorry

end number_of_sunflowers_l776_776640


namespace comb_10_3_eq_120_l776_776498

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l776_776498


namespace trig_identity_proof_l776_776083

theorem trig_identity_proof :
  sin (20 * Real.pi / 180) * cos (10 * Real.pi / 180) - cos (160 * Real.pi / 180) * sin (10 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trig_identity_proof_l776_776083


namespace algebraic_expression_value_l776_776373
open Real

def algebraic_expression (a : ℝ) := a + 2 * sqrt (a - 1) - sqrt (1 - a) + 3

theorem algebraic_expression_value (a : ℝ) (h1 : a - 1 ≥ 0) (h2 : 1 - a ≥ 0) : 
  algebraic_expression a = 4 :=
by
  sorry

end algebraic_expression_value_l776_776373


namespace parabola_intersection_l776_776726

theorem parabola_intersection :
  let y := fun x : ℝ => x^2 - 2*x - 3 in
  (∀ x : ℝ, y x = 0 ↔ x = 3 ∨ x = -1) ∧ y (-1) = 0 :=
by
  sorry

end parabola_intersection_l776_776726


namespace points_lie_on_circle_l776_776968

theorem points_lie_on_circle (t : ℝ) (ht : t ≠ 0) : 
  let x := (t^2 + 1) / t
  let y := (t^2 - 1) / t
  in ∃ a : ℝ, x^2 + y^2 = a := 
sorry

end points_lie_on_circle_l776_776968


namespace find_A_max_area_triangle_l776_776258

-- Declare noncomputable theory as necessary
noncomputable theory

variables {A B C : ℝ} -- Angles A, B, C
variables {a b c : ℝ} -- Sides opposite to angles A, B, C

-- Given conditions and question for proving the value of angle A and finding the max area
def cond1 : Prop := 2 * b * Real.cos A = c * Real.cos A + a * Real.cos C
def cond2 : a = 4
def value_A := A = Real.pi / 3
def max_area := ∃ b c, 4 * Real.sqrt 3

-- Statement 1: Proving angle A
theorem find_A (h1 : cond1) : value_A := by
  sorry

-- Statement 2: Finding maximum area with a = 4
theorem max_area_triangle (h1 : cond1) (h2 : cond2) : max_area := by
  sorry

end find_A_max_area_triangle_l776_776258


namespace minimize_set_elements_l776_776613

theorem minimize_set_elements (k : ℝ) :
  (∀ x ∈ ℤ, (kx - k^2 - 6) * (x - 4) > 0 → k < 0) ↔ (k < 0) :=
sorry

end minimize_set_elements_l776_776613


namespace pie_prices_l776_776873

theorem pie_prices 
  (d k : ℕ) (hd : 0 < d) (hk : 1 < k) 
  (p1 p2 : ℕ) 
  (prices : List ℕ) (h_prices : prices = [64, 64, 70, 72]) 
  (hP1 : (p1 + d) ∈ prices)
  (hZ1 : (k * p1) ∈ prices)
  (hP2 : (p2 + d) ∈ prices)
  (hZ2 : (k * p2) ∈ prices)
  (h_different_prices : p1 + d ≠ k * p1 ∧ p2 + d ≠ k * p2) : 
  ∃ p1 p2, p1 ≠ p2 :=
sorry

end pie_prices_l776_776873


namespace comb_10_3_eq_120_l776_776500

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l776_776500


namespace solve_for_b_l776_776894

theorem solve_for_b (b : ℝ) : 
  (∀ x y, (x = 1 ∧ y = 2) ∨ (x = -2 ∧ y = -1) → y = 2*x^2 + b*x + 3) → 
  b = 11 / 2 :=
begin
  intro h,
  cases h 1 2 (or.inl (and.intro rfl rfl)),
  cases h (-2) (-1) (or.inr (and.intro rfl rfl)),
  sorry, -- steps to verify the contradiction and prove b = 11/2
end

end solve_for_b_l776_776894


namespace least_number_to_add_l776_776369

-- Definition of LCM for given primes
def lcm_of_primes : ℕ := 5 * 7 * 11 * 13 * 17 * 19

theorem least_number_to_add (n : ℕ) : 
  (5432 + n) % 5 = 0 ∧ 
  (5432 + n) % 7 = 0 ∧ 
  (5432 + n) % 11 = 0 ∧ 
  (5432 + n) % 13 = 0 ∧ 
  (5432 + n) % 17 = 0 ∧ 
  (5432 + n) % 19 = 0 ↔ 
  n = 1611183 :=
by sorry

end least_number_to_add_l776_776369


namespace peanut_butter_revenue_l776_776666

theorem peanut_butter_revenue :
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  plantation_length * plantation_width * peanuts_per_sqft * butter_from_peanuts_ratio / 1000 * butter_price_per_kg = 31250 := 
by
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  sorry

end peanut_butter_revenue_l776_776666


namespace repeating_decimal_sum_fraction_l776_776516

theorem repeating_decimal_sum_fraction :
  (∑ x in [0.2, 0.04, 0.0008], x) = (878 / 9999 : ℚ) :=
by
  sorry

end repeating_decimal_sum_fraction_l776_776516


namespace john_paid_total_expenses_l776_776262

noncomputable def monthly_cost (amount : ℕ) (months : ℕ) : ℕ := amount * months
noncomputable def quarterly_cost (amount : ℕ) (quarters : ℕ) : ℕ := amount * quarters
noncomputable def annual_cost (amount : ℕ) : ℕ := amount

-- Conditions
def vet_appointments_cost := 400
def vet_appointments_count := 3
def medication_cost_per_month := 50
def pet_food_cost_per_month := 30
def play_toys_cost_per_month := 15
def grooming_cost_per_quarter := 60
def irregular_health_issue_cost := 200
def irregular_health_issue_count := 2
def insurance_cost := 100

-- Insurance coverage
def insurance_coverage_vet_appointments := 80 / 100 -- 80%
def insurance_coverage_medication := 50 / 100 -- 50%
def insurance_coverage_irregular_issues := 25 / 100 -- 25%

-- Calculations
def total_vet_appointments_cost (count : ℕ) (cost : ℕ) : ℕ :=
  let first_appointment := cost
  let subsequent_appointments := (count - 1) * cost
  let covered_amount := (subsequent_appointments : ℝ) * insurance_coverage_vet_appointments
  first_appointment + (subsequent_appointments - covered_amount.to_nat)

def total_medication_cost (med_cost : ℕ) : ℕ :=
  let annual_cost := monthly_cost med_cost 12
  let covered_amount := (annual_cost : ℝ) * insurance_coverage_medication
  annual_cost - covered_amount.to_nat

def total_irregular_health_issue_cost (count cost : ℕ) : ℕ :=
  let total_issue_cost := cost * count
  let covered_amount := (total_issue_cost : ℝ) * insurance_coverage_irregular_issues
  total_issue_cost - covered_amount.to_nat

def total_cost : ℕ :=
  total_vet_appointments_cost vet_appointments_count vet_appointments_cost +
  total_medication_cost medication_cost_per_month +
  monthly_cost pet_food_cost_per_month 12 +
  monthly_cost play_toys_cost_per_month 12 +
  quarterly_cost grooming_cost_per_quarter 4 +
  total_irregular_health_issue_cost irregular_health_issue_count irregular_health_issue_cost +
  annual_cost insurance_cost

theorem john_paid_total_expenses :
  total_cost = 2040 := by
  -- placeholder for proof
  sorry

end john_paid_total_expenses_l776_776262


namespace slope_of_line_through_points_l776_776988

theorem slope_of_line_through_points 
  (t : ℝ) 
  (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 12 * t + 6) 
  (h2 : 2 * x + 3 * y = 8 * t - 1) : 
  ∃ m b : ℝ, (∀ t : ℝ, y = m * x + b) ∧ m = 0 :=
by 
  sorry

end slope_of_line_through_points_l776_776988


namespace find_d_l776_776938

noncomputable def equilateral_triangle_side : ℝ := 800

-- Assume points A, B, and C form an equilateral triangle
def ABC_is_equilateral (A B C : Point) : Prop :=
  dist A B = equilateral_triangle_side ∧
  dist B C = equilateral_triangle_side ∧
  dist C A = equilateral_triangle_side

-- Assume Points P and Q such that the given conditions hold
def conditions (A B C P Q O : Point) : Prop :=
  dist P A = dist P B ∧ dist P B = dist P C ∧
  dist Q A = dist Q B ∧ dist Q B = dist Q C ∧
  dist O A = dist O B ∧ dist O B = dist O C ∧
  dist O C = dist O P ∧ dist O P = dist O Q ∧
  ∃ θ : ℝ, θ = 150 ∧ plane_dihedral_angle P A B = θ

-- The main theorem to prove: Given the above conditions, distance d is 800
theorem find_d (A B C P Q O : Point) :
  ABC_is_equilateral A B C → conditions A B C P Q O → dist O A = 800 :=
by
  -- Proof here, using the provided conditions
  sorry

end find_d_l776_776938


namespace divisors_of_30_l776_776182

theorem divisors_of_30 : 
  ∃ n : ℕ, (∀ d : ℤ, d ∣ 30 → d > 0 → d ≤ 30) ∧ (∀ d : ℤ, d ∣ 30 → -d ∣ 30) ∧ (∀ d : ℤ, d ∣ 30 → -d = d → d = 0) ∧ 2 * (Finset.card (Finset.filter (λ d, 0 < d) (Finset.filter (λ d, d ∣ 30) (Finset.Icc 1 30).toFinset))) = 16 :=
by {
   sorry
}

end divisors_of_30_l776_776182


namespace minimal_positive_sum_circle_integers_l776_776888

-- Definitions based on the conditions in the problem statement
def cyclic_neighbors (l : List Int) (i : ℕ) : Int :=
  l.getD (Nat.mod (i - 1) l.length) 0 + l.getD (Nat.mod (i + 1) l.length) 0

-- Problem statement in Lean: 
theorem minimal_positive_sum_circle_integers :
  ∃ (l : List Int), l.length ≥ 5 ∧ (∀ (i : ℕ), i < l.length → l.getD i 0 ∣ cyclic_neighbors l i) ∧ (0 < l.sum) ∧ l.sum = 2 :=
sorry

end minimal_positive_sum_circle_integers_l776_776888


namespace count_valid_two_digit_numbers_l776_776609

-- Definition of the two-digit natural numbers with unique digits from the set {0, 1, 7, 9}
def valid_two_digit_numbers (d1 d2 : ℕ) : Prop :=
  d1 ≠ d2 ∧ d1 ≠ 0 ∧ d1 ∈ {1, 7, 9} ∧ d2 ∈ {0, 1, 7, 9}

-- The total count of such valid two-digit numbers is 9
theorem count_valid_two_digit_numbers : ∃ cnt, cnt = 9 ∧ ∀ d1 d2, valid_two_digit_numbers d1 d2 → cnt = 9 := by
  sorry

end count_valid_two_digit_numbers_l776_776609


namespace knights_statements_l776_776305

theorem knights_statements (r ℓ : Nat) (hr : r ≥ 2) (hℓ : ℓ ≥ 2)
  (h : 2 * r * ℓ = 230) :
  (r + ℓ) * (r + ℓ - 1) - 230 = 526 :=
by
  sorry

end knights_statements_l776_776305


namespace bakery_wholesale_price_exists_l776_776863

theorem bakery_wholesale_price_exists :
  ∃ (p1 p2 d k : ℕ), (d > 0) ∧ (k > 1) ∧
  ({p1 + d, p2 + d, k * p1, k * p2} = {64, 64, 70, 72}) :=
by sorry

end bakery_wholesale_price_exists_l776_776863


namespace find_m_l776_776137

-- Definitions for the sets
def setA (x : ℝ) : Prop := -2 < x ∧ x < 8
def setB (m : ℝ) (x : ℝ) : Prop := 2 * m - 1 < x ∧ x < m + 3

-- Condition on the intersection
def intersection (m : ℝ) (a b : ℝ) (x : ℝ) : Prop := 2 * m - 1 < x ∧ x < m + 3 ∧ -2 < x ∧ x < 8

-- Theorem statement
theorem find_m (m a b : ℝ) (h₀ : b - a = 3) (h₁ : ∀ x, intersection m a b x ↔ (a < x ∧ x < b)) : m = -2 ∨ m = 1 :=
sorry

end find_m_l776_776137


namespace squared_sum_area_eq_10800_l776_776268

-- Definition of the initial triangle and its side lengths
def triangle_A1B1C1 : Type := {A1 B1 C1 : Type}

constant A1B1 : ℝ
constant B1C1 : ℝ
constant C1A1 : ℝ
axiom A1B1_val : A1B1 = 16
axiom B1C1_val : B1C1 = 14
axiom C1A1_val : C1A1 = 10

-- Sequence definition
def triangle_seq (i : ℕ) : Type := {Ai Bi Ci : Type}

constant circumcenter (i : ℕ) : Type

-- Triangles are defined recursively as per given conditions: 
constant triangle_step (i : ℕ) : triangle_seq (i+1)

-- Area of triangle
constant area : (i : ℕ) → ℝ
axiom initial_area : area 1 = 40 * real.sqrt 3

-- Definition of the infinite series sum and its square
def infinite_sum : ℝ := ∑' (i : ℕ), area i
def series_square := infinite_sum^2

-- Prove that the squared sum of the areas equals 10800
theorem squared_sum_area_eq_10800 : series_square = 10800 := sorry

end squared_sum_area_eq_10800_l776_776268


namespace sum_cubes_mod_7_l776_776541

theorem sum_cubes_mod_7 :
  (∑ i in Finset.range 151, i ^ 3) % 7 = 0 := by
  sorry

end sum_cubes_mod_7_l776_776541


namespace find_angle_A_max_triangle_area_l776_776250

noncomputable def angle_condition (b c a : ℝ) (A C : ℝ) : Prop :=
  2 * b * Real.cos A = c * Real.cos A + a * Real.cos C

theorem find_angle_A {a b c A C : ℝ} 
  (h1: angle_condition b c a A C) : A = Real.pi / 3 :=
sorry

theorem max_triangle_area {a b c A : ℝ} 
  (hA : A = Real.pi / 3)
  (ha : a = 4) : 
  let max_area := (4 : ℝ) * Real.sqrt 3 in
  Real.abs ((1 / 2) * b * c * Real.sin A) ≤ max_area :=
sorry

end find_angle_A_max_triangle_area_l776_776250


namespace binomial_coefficient_10_3_l776_776475

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l776_776475


namespace perpendicular_condition_l776_776949

theorem perpendicular_condition (a : ℝ) :
  (2 * a * x + (a - 1) * y + 2 = 0) ∧ ((a + 1) * x + 3 * a * y + 3 = 0) →
  (a = 1/5 ↔ ∃ x y: ℝ, ((- (2 * a / (a - 1))) * (-(a + 1) / (3 * a)) = -1)) :=
by
  sorry

end perpendicular_condition_l776_776949


namespace binomial_coefficient_10_3_l776_776473

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l776_776473


namespace sec_7pi_over_4_l776_776527

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_7pi_over_4 :
  sec (7 * Real.pi / 4) = Real.sqrt 2 := by
  have h1 : Real.cos (7 * Real.pi / 4) = Real.cos (45 * Real.pi / 180) := by sorry
  have h2 : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  rw [sec, h1, h2]
  field_simp
  norm_num

end sec_7pi_over_4_l776_776527


namespace inequality_proof_l776_776055

-- Assume f is an even function (f(-x) = f(x))
def is_even (f : ℝ → ℝ) := ∀ x, f(-x) = f(x)

-- Assume f(2 - x) = f(x)
def has_periodic_property (f : ℝ → ℝ) := ∀ x, f(2 - x) = f(x)

-- Assume f is decreasing on the interval [-3, -2]
def is_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f(x) > f(y)

-- Proof of the required inequality
theorem inequality_proof (f : ℝ → ℝ) (α β : ℝ) (h_alpha_acute : 0 < α ∧ α < π / 2) (h_beta_acute : 0 < β ∧ β < π / 2)
  (h_even : is_even f) (h_periodic : has_periodic_property f) (h_decreasing : is_decreasing_on_interval f (-3) (-2)) :
  f (Real.sin α) > f (Real.cos β) :=
sorry

end inequality_proof_l776_776055


namespace ordered_pairs_bound_l776_776271

variable (m n : ℕ) (a b : ℕ → ℝ)

theorem ordered_pairs_bound
  (h_m : m ≥ n)
  (h_n : n ≥ 2022)
  : (∃ (pairs : Finset (ℕ × ℕ)), 
      (∀ i j, (i, j) ∈ pairs → 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ |a i + b j - (i * j)| ≤ m) ∧
      pairs.card ≤ 3 * n * Real.sqrt (m * Real.log (n))) := 
  sorry

end ordered_pairs_bound_l776_776271


namespace divisors_of_30_l776_776205

theorem divisors_of_30 : 
  let positive_divisors := {1, 2, 3, 5, 6, 10, 15, 30}
  in 2 * positive_divisors.card = 16 :=
by
  let positive_divisors := {1, 2, 3, 5, 6, 10, 15, 30}
  have h1 : positive_divisors.card = 8 := sorry
  have h2 : 2 * 8 = 16 := sorry
  exact h2

end divisors_of_30_l776_776205


namespace find_first_number_l776_776339

theorem find_first_number :
  ∃ x : ℝ, let y := -4.5 in  x * y = 2 * x - 36 ∧ x ≈ 5.53846154 :=
by
  use 36 / 6.5
  split
  -- Verification of the equation x * y = 2 * x - 36 
  simp [*, ← mul_assoc, show 36 / 6.5 = _, by norm_num1]
  -- Verification of the approximation x ≈ 5.53846154
  rw [real.div_eq_mul_inv, show (36 : ℝ) * _ = 5.53846154, by norm_num1]
  norm_num1
  sorry

end find_first_number_l776_776339


namespace find_angle_A_find_max_area_l776_776253

noncomputable def triangle (a b c A B C : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  A + B + C = π ∧
  sin A > 0 ∧ sin B > 0 ∧ sin C > 0

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h : triangle a b c A B C) :
  (2 * b * cos A = c * cos A + a * cos C) → A = π / 3 :=
by
  sorry

theorem find_max_area (a b c : ℝ) (A B C : ℝ) (h : triangle a b c A B C) :
  (2 * b * cos A = c * cos A + a * cos C) → a = 4 → (1/2 * b * c * sin A ≤ 4 * sqrt 3) :=
by
  sorry 

end find_angle_A_find_max_area_l776_776253


namespace area_of_square_on_PS_l776_776752

-- Given parameters as conditions in the form of hypotheses
variables (PQ QR RS PS PR : ℝ)

-- Hypotheses based on problem conditions
def hypothesis1 : PQ^2 = 25 := sorry
def hypothesis2 : QR^2 = 49 := sorry
def hypothesis3 : RS^2 = 64 := sorry
def hypothesis4 : PR^2 = PQ^2 + QR^2 := sorry
def hypothesis5 : PS^2 = PR^2 - RS^2 := sorry

-- The main theorem we need to prove
theorem area_of_square_on_PS :
  PS^2 = 10 := 
by {
  sorry
}

end area_of_square_on_PS_l776_776752


namespace probability_of_being_drawn_first_simple_random_sampling_l776_776753

theorem probability_of_being_drawn_first_simple_random_sampling :
  (sample_size individuals : ℕ) (pop_size : ℕ) (a : ℕ) 
  (h_sample_size_pop_size : sample_size = 3) 
  (h_pop_size : pop_size = 10) 
  (h_valid_a : 1 ≤ a ∧ a ≤ pop_size) : 
  (Pr : ℚ) :=
by
  -- Given that the probability of any individual being drawn first in simple random sampling
  -- with a sample size of 3 from a population of 10 individuals is 1/10
  have Pr := 1 / 10
  exact Pr

#lint

end probability_of_being_drawn_first_simple_random_sampling_l776_776753


namespace circle_equation_l776_776148

noncomputable def find_circle_equation : (ℝ × ℝ) × (ℝ × ℝ) × ℝ :=
  let center : (ℝ × ℝ) := (1, 3)
  let radius : ℝ := 5
  ((1, 3), (-4, 3), radius)

theorem circle_equation :
  ∃ a b : ℝ,
    a > 0 ∧ b > 0 ∧
    2 * a - b + 1 = 0 ∧
    (a + 4)^2 + (b - 3)^2 = 25 ∧
    (x y : ℝ),
      ((x - a)^2 + (y - b)^2 = radius^2) :=
begin
  use [1, 3],
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  use (-4, 3),
  split,
  { simp },
  exact sorry,
end

end circle_equation_l776_776148


namespace triangle_area_is_500_over_3_l776_776291

structure Point := (x : ℝ) (y : ℝ)

structure Triangle :=
  (A B C : Point)
  (right_angle_at_C : (A.x - C.x) * (B.x - C.x) + (A.y - C.y) * (B.y - C.y) = 0)
  (hypotenuse_length : real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 50)
  (P_median_line : ∀ t : ℝ, (∃ a b : ℝ, b = a * t + 2 → Point (a * t + 2) (a * t)))
  (Q_median_line : ∀ t : ℝ, (∃ a b : ℝ, b = 2 * a * t + 3 → Point (a * t + 3) (2 * a * t)))

def area_triangle (T : Triangle) : ℝ :=
  real.abs ((T.A.x * (T.B.y - T.C.y) + T.B.x * (T.C.y - T.A.y) + T.C.x * (T.A.y - T.B.y)) / 2)

theorem triangle_area_is_500_over_3 : ∀ (T : Triangle),
  T.right_angle_at_C → 
  T.hypotenuse_length →
  T.P_median_line →
  T.Q_median_line →
  area_triangle T = 500 / 3 :=
by
  intros,
  sorry

end triangle_area_is_500_over_3_l776_776291


namespace coeff_x2_in_binomial_expansion_l776_776242

theorem coeff_x2_in_binomial_expansion :
  (∃ n : ℕ, 2^n = 64 ∧
     ∃ (c : ℚ), 
       (∀ x : ℚ, 
         (∑ k in Finset.range (n + 1), 
           (nat.choose n k) * ((√x / 2)^(n - k)) * ((-2 / √x)^k)) = 
           64) ∧
         (∃ r : ℕ, 3 - r = 2 ∧ 
           c = -(3 / 8))) :=
by
  sorry

end coeff_x2_in_binomial_expansion_l776_776242


namespace family_photo_order_is_correct_l776_776509

def isNextTo (a b : String) (ordering : List String) : Prop :=
  ∃ i, ordering.nth i = some a ∧ (ordering.nth (i + 1) = some b ∨ ordering.nth (i - 1) = some b)

def inMiddle (a : String) (ordering : List String) : Prop :=
  (ordering.length ≥ 3) ∧ (ordering.nth (ordering.length / 2) = some a)

theorem family_photo_order_is_correct (ordering : List String) :
  isNextTo "I" "Dad" ordering ∧ isNextTo "I" "Mom" ordering ∧
  isNextTo "Sister" "Mom" ordering ∧ isNextTo "Sister" "Brother" ordering →
  inMiddle "Mom" ordering :=
sorry

end family_photo_order_is_correct_l776_776509


namespace proof_problem_l776_776147

-- Given parametric equations of line l
def line_l_param_eqns (t : ℝ) : ℝ × ℝ := (t - 3, real.sqrt 3 * t)

-- Given polar equation of curve C
def curve_C_polar_eqn (ρ θ : ℝ) : Prop := ρ^2 = 4 * ρ * real.cos θ - 3

-- Ordinary equation of line l in rectangular coordinates
def line_l_eqn (x y : ℝ) : Prop := real.sqrt 3 * x - y + 3 * real.sqrt 3 = 0

-- Rectangular coordinate equation of curve C
def curve_C_rect_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 3 = 0

-- Range of distance from point P on curve C to line l
def distance_range_from_curve_C_to_line_l : set ℝ :=
  set.Icc (5 * real.sqrt 3 / 2 - 1) (5 * real.sqrt 3 / 2 + 1)

-- The theorem to prove
theorem proof_problem (t : ℝ) (ρ θ : ℝ) (x y : ℝ) :
  curve_C_polar_eqn ρ θ →
  curve_C_rect_eqn x y →
  line_l_eqn x y →
  line_l_param_eqns t = (x, y) →
  distance_range_from_curve_C_to_line_l = 
    set.Icc (5 * real.sqrt 3 / 2 - 1) (5 * real.sqrt 3 / 2 + 1) :=
sorry

end proof_problem_l776_776147


namespace scenario1_scenario2_scenario3_l776_776614

noncomputable def scenario1_possible_situations : Nat :=
  12

noncomputable def scenario2_possible_situations : Nat :=
  144

noncomputable def scenario3_possible_situations : Nat :=
  50

theorem scenario1 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) (not_consecutive : Prop) :
  shots = 10 ∧ hits = 7 ∧ consecutive_hits = 5 ∧ remaining_hits = 2 ∧ not_consecutive → 
  scenario1_possible_situations = 12 := by
  sorry

theorem scenario2 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) :
  shots = 10 ∧ hits = 7 ∧ consecutive_hits = 4 ∧ remaining_hits = 3 → 
  scenario2_possible_situations = 144 := by
  sorry

theorem scenario3 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) :
  shots = 10 ∧ hits = 6 ∧ consecutive_hits = 4 ∧ remaining_hits = 2 → 
  scenario3_possible_situations = 50 := by
  sorry

end scenario1_scenario2_scenario3_l776_776614


namespace ratio_of_x_to_y_l776_776793

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 7 * y) / (17 * x - 3 * y) = 4 / 7) : 
  x / y = 37 / 16 :=
by
  sorry

end ratio_of_x_to_y_l776_776793


namespace tan_315_degrees_angle_315_degrees_to_radians_l776_776506

def angle_in_degrees := 315
def angle_in_radians := (7 * Real.pi) / 4

theorem tan_315_degrees :
  Real.tan (angle_in_degrees * (Real.pi / 180)) = -1 :=
sorry

theorem angle_315_degrees_to_radians :
  angle_in_degrees * (Real.pi / 180) = angle_in_radians :=
sorry

end tan_315_degrees_angle_315_degrees_to_radians_l776_776506


namespace coal_extraction_in_four_months_l776_776017

theorem coal_extraction_in_four_months
  (x1 x2 x3 x4 : ℝ)
  (h1 : 4 * x1 + x2 + 2 * x3 + 5 * x4 = 10)
  (h2 : 2 * x1 + 3 * x2 + 2 * x3 + x4 = 7)
  (h3 : 5 * x1 + 2 * x2 + x3 + 4 * x4 = 14) :
  4 * (x1 + x2 + x3 + x4) = 12 :=
by
  sorry

end coal_extraction_in_four_months_l776_776017


namespace determine_wholesale_prices_l776_776825

-- Definitions and Conditions
-- Wholesale prices for days 1 and 2 are defined as p1 and p2
def wholesale_price_day1 : ℝ := p1
def wholesale_price_day2 : ℝ := p2

-- The price increment at "Baker Plus" store is a constant d
def price_increment_baker_plus : ℝ := d
-- The factor at "Star" store is a constant k > 1
def price_factor_star : ℝ := k
-- Given prices for two days: 64, 64, 70, 72 rubles
def given_prices : set ℝ := {64, 64, 70, 72}

-- Prices at the stores
def baker_plus_day1_price := wholesale_price_day1 + price_increment_baker_plus
def star_day1_price := wholesale_price_day1 * price_factor_star

def baker_plus_day2_price := wholesale_price_day2 + price_increment_baker_plus
def star_day2_price := wholesale_price_day2 * price_factor_star

-- Theorem to determine the wholesale prices for each day (p1 and p2)
theorem determine_wholesale_prices
    (baker_plus_day1_price ∈ given_prices)
    (star_day1_price ∈ given_prices)
    (baker_plus_day2_price ∈ given_prices)
    (star_day2_price ∈ given_prices)
    (h1 : baker_plus_day1_price ≠ star_day1_price)
    (h2 : baker_plus_day2_price ≠ star_day2_price) :
  (∃ p1 p2 d k > 1, 
      (p1 + d ∈ given_prices) ∧ (p1 * k ∈ given_prices) ∧
      (p2 + d ∈ given_prices) ∧ (p2 * k ∈ given_prices) ∧
      (p1 ≠ p2)) :=
  sorry

end determine_wholesale_prices_l776_776825


namespace determine_wholesale_prices_l776_776851

variables (p1 p2 d k : ℝ)
variables (h1 : 1 < k)
variables (prices : Finset ℝ)
variables (p_plus : ℝ → ℝ) (star : ℝ → ℝ)

noncomputable def BakeryPrices : Prop :=
  p_plus p1 = p1 + d ∧ p_plus p2 = p2 + d ∧ 
  star p1 = k * p1 ∧ star p2 = k * p2 ∧ 
  prices = {64, 64, 70, 72} ∧ 
  ∃ (q1 q2 q3 q4 : ℝ), 
    (q1 = p_plus p1 ∨ q1 = star p1 ∨ q1 = p_plus p2 ∨ q1 = star p2) ∧
    (q2 = p_plus p1 ∨ q2 = star p1 ∨ q2 = p_plus p2 ∨ q2 = star p2) ∧
    (q3 = p_plus p1 ∨ q3 = star p1 ∨ q3 = p_plus p2 ∨ q3 = star p2) ∧
    (q4 = p_plus p1 ∨ q4 = star p1 ∨ q4 = p_plus p2 ∨ q4 = star p2) ∧
    prices = {q1, q2, q3, q4}

theorem determine_wholesale_prices (p1 p2 : ℝ) (d k : ℝ)
    (h1 : 1 < k) (prices : Finset ℝ) :
  BakeryPrices p1 p2 d k h1 prices (λ p, p + d) (λ p, k * p) :=
sorry

end determine_wholesale_prices_l776_776851


namespace coefficient_x3y5_expansion_l776_776360

open Real

theorem coefficient_x3y5_expansion :
  let x := (2:ℚ) / 3
  let y := -(1:ℚ) / 3
  (binomial 8 3) * (x ^ 3) * (y ^ 5) = - (448:ℚ) / 6561 :=
by
  let x := (2:ℚ) / 3
  let y := -(1:ℚ) / 3
  sorry

end coefficient_x3y5_expansion_l776_776360


namespace value_of_x_l776_776571

variable (x y z a b c : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
variable (h1 : x * y / (x + y) = a)
variable (h2 : x * z / (x + z) = b)
variable (h3 : y * z / (y + z) = c)

theorem value_of_x : x = 2 * a * b * c / (a * c + b * c - a * b) :=
by sorry

end value_of_x_l776_776571


namespace wholesale_prices_l776_776837

-- Definitions for the problem conditions
variable (p1 p2 d k : ℝ)
variable (h_d : d > 0)
variable (h_k : k > 1)
variable (prices : Finset ℝ)
variable (h_prices : prices = {64, 64, 70, 72})

-- The theorem statement to prove
theorem wholesale_prices :
  ∃ p1 p2, (p1 + d ∈ prices ∧ k * p1 ∈ prices) ∧ 
           (p2 + d ∈ prices ∧ k * p2 ∈ prices) ∧ 
           p1 ≠ p2
:= sorry

end wholesale_prices_l776_776837


namespace Austin_fewer_pears_l776_776927

noncomputable def Dallas_apples : ℕ := 14
noncomputable def Dallas_pears : ℕ := 9
noncomputable def Austin_apples : ℕ := Dallas_apples + 6
noncomputable def Austin_total_fruit : ℕ := 24
noncomputable def Austin_pears : ℕ := Austin_total_fruit - Austin_apples

theorem Austin_fewer_pears : Dallas_pears - Austin_pears = 5 :=
by
  have D_apples : Dallas_apples = 14 := by rfl
  have D_pears : Dallas_pears = 9 := by rfl
  have A_apples : Austin_apples = 20 := by rw [Dallas_apples, D_apples]; rfl
  have A_total : Austin_total_fruit = 24 := by rfl
  have A_pears : Austin_pears = 4 := by rw [Austin_apples, A_apples, Austin_total_fruit, A_total]; rfl
  show Dallas_pears - Austin_pears = 5
  rw [D_pears, A_pears]
  rfl

end Austin_fewer_pears_l776_776927


namespace bakery_wholesale_price_exists_l776_776860

theorem bakery_wholesale_price_exists :
  ∃ (p1 p2 d k : ℕ), (d > 0) ∧ (k > 1) ∧
  ({p1 + d, p2 + d, k * p1, k * p2} = {64, 64, 70, 72}) :=
by sorry

end bakery_wholesale_price_exists_l776_776860


namespace greatest_possible_m_exists_greatest_m_l776_776039

def reverse_digits (m : ℕ) : ℕ :=
  -- Function to reverse digits of a four-digit number
  sorry 

theorem greatest_possible_m (m : ℕ) : 
  (1000 ≤ m ∧ m < 10000) ∧
  (let n := reverse_digits m in 1000 ≤ n ∧ n < 10000) ∧
  (m % 63 = 0) ∧
  (m % 11 = 0) →
  m ≤ 9696 :=
by
  sorry

theorem exists_greatest_m (m : ℕ) : 
  (1000 ≤ m ∧ m < 10000) ∧
  (let n := reverse_digits m in 1000 ≤ n ∧ n < 10000) ∧
  (m % 63 = 0) ∧
  (m % 11 = 0) →
  ∃ m, m = 9696 :=
by
  sorry

end greatest_possible_m_exists_greatest_m_l776_776039


namespace diameter_of_circle_l776_776751

theorem diameter_of_circle {a b c d e f D : ℕ} 
  (h1 : a = 15) (h2 : b = 20) (h3 : c = 25) (h4 : d = 33) (h5 : e = 56) (h6 : f = 65)
  (h_right_triangle1 : a^2 + b^2 = c^2)
  (h_right_triangle2 : d^2 + e^2 = f^2)
  (h_inscribed_triangles : true) -- This represents that both triangles are inscribed in the circle.
: D = 65 :=
sorry

end diameter_of_circle_l776_776751


namespace hamburger_cost_correct_l776_776295

def class_students (num_classes : ℕ) (students_per_class : ℕ) : ℕ :=
  num_classes * students_per_class

def total_students : ℕ :=
  class_students 5 30 + class_students 4 28 + class_students 4 27

def carrot_cost_per_student : ℝ := 0.50
def cookie_cost_per_student : ℝ := 0.20
def total_lunch_cost : ℝ := 1036.0

def total_carrot_cost : ℝ := total_students * carrot_cost_per_student
def total_cookie_cost : ℝ := total_students * cookie_cost_per_student
def total_carrot_cookie_cost : ℝ := total_carrot_cost + total_cookie_cost
def total_hamburger_cost : ℝ := total_lunch_cost - total_carrot_cookie_cost

def cost_of_one_hamburger : ℝ := total_hamburger_cost / total_students

theorem hamburger_cost_correct : cost_of_one_hamburger = 2.10 := by
  sorry

end hamburger_cost_correct_l776_776295


namespace line_through_P1_P2_l776_776174

variable {a1 b1 a2 b2 : ℝ}

-- Definitions translating the conditions
def line1_pass_through_A : Prop := 2 * a1 + 3 * b1 + 1 = 0
def line2_pass_through_A : Prop := 2 * a2 + 3 * b2 + 1 = 0

-- The statement we want to prove
theorem line_through_P1_P2 : 
  line1_pass_through_A ∧ line2_pass_through_A → 
  ∀ x y : ℝ, (x = a1 ∧ y = b1) ∨ (x = a2 ∧ y = b2) → 2 * x + 3 * y + 1 = 0 :=
by
  sorry

end line_through_P1_P2_l776_776174


namespace distribution_schemes_count_l776_776933

theorem distribution_schemes_count
  (V : Finset (Fin 5)) (A B : Fin 5)
  (V1 V2 V3 : Finset (Fin 5))
  (hv1 : ∀ v ∈ V, v ≠ A → v ≠ B)
  (hV1 : ∀ v ∈ V1, v ≠ A → v ≠ B)
  (hV2 : ∀ v ∈ V2, v ≠ A → v ≠ B)
  (hV3 : ∀ v ∈ V3, v ≠ A → v ≠ B)
  (h_disjoint : Disjoint V1 V2 ∧ Disjoint V1 V3 ∧ Disjoint V2 V3)
  (h_nonempty : V1.Nonempty ∧ V2.Nonempty ∧ V3.Nonempty)
  (h_union : V1 ∪ V2 ∪ V3 = V) :
  card {d : Finset (Fin 5) × Finset (Fin 5) × Finset (Fin 5) |
    let V := 1 \ v1, v2, v3
    in ∀ v ∈ V, v ≠ A → v ≠ B ∧ 
       Disjoint V1 V2 ∧ Disjoint V1 V3 ∧ Disjoint V2 V3 ∧ 
       V1.Nonempty ∧ V2.Nonempty ∧ V3.Nonempty ∧ 
       V1 ∪ V2 ∪ V3 = V} = 114 := sorry

end distribution_schemes_count_l776_776933


namespace common_point_X_common_point_Q_ratio_conditions_l776_776309

-- Definitions

-- Let's declare points A, B, C, P, A', B', C'
variables {A B C P A' B' C' : Type*}

-- Point A', B', C' are reflections of P across BC, CA, and AB respectively.
axiom reflection_A' : reflection P (line BC) = A'
axiom reflection_B' : reflection P (line CA) = B'
axiom reflection_C' : reflection P (line AB) = C'

-- Questions

-- 1. Prove there exists a point X common to the circumcircles of Σ1, Σ2, Σ3, Σ4
theorem common_point_X :
  ∃ X : Type*, on_circumcircle X (triangle A B' C') ∧ on_circumcircle X (triangle A' B C') ∧ on_circumcircle X (triangle A' B' C) ∧ on_circumcircle X (triangle A B C) :=
sorry

-- 2. Prove there exists a point Q common to the circumcircles of Δ1, Δ2, Δ3, Δ4
theorem common_point_Q :
  ∃ Q : Type*, on_circumcircle Q (triangle A' B C) ∧ on_circumcircle Q (triangle A B' C) ∧ on_circumcircle Q (triangle A B C') ∧ on_circumcircle Q (triangle A' B' C') :=
sorry

-- 3. Prove the ratio conditions for points I, J, K, and O
variables {I J K O Q : Type*}
axiom I_center : is_center I (circumcircle (triangle A' B C))
axiom J_center : is_center J (circumcircle (triangle A B' C))
axiom K_center : is_center K (circumcircle (triangle A B C'))
axiom O_center : is_center O (circumcircle (triangle A' B' C'))

theorem ratio_conditions :
  QI / OI = QJ / OJ ∧ QJ / OJ = QK / OK :=
sorry

end common_point_X_common_point_Q_ratio_conditions_l776_776309


namespace q_of_1_eq_neg6_l776_776654

theorem q_of_1_eq_neg6 {d e : ℤ} 
  (h1 : (λ x : ℤ, x^2 + d * x + e) ∣ (λ x : ℤ, x^4 + x^3 + 8 * x^2 + 7 * x + 18))
  (h2 : (λ x : ℤ, x^2 + d * x + e) ∣ (λ x : ℤ, 2 * x^4 + 3 * x^3 + 9 * x^2 + 8 * x + 20))
  : (λ x : ℤ, x^2 + d * x + e) 1 = -6 :=
sorry

end q_of_1_eq_neg6_l776_776654


namespace least_three_digit_multiple_13_l776_776774

theorem least_three_digit_multiple_13 : 
  ∃ n : ℕ, (n ≥ 100) ∧ (∃ k : ℕ, n = 13 * k) ∧ ∀ m, m < n → (m < 100 ∨ ¬∃ k : ℕ, m = 13 * k) :=
by
  sorry

end least_three_digit_multiple_13_l776_776774


namespace find_area_of_triangle_BMT_l776_776645

noncomputable def area_of_triangle_BMT (M A T B : Point) (square_MATH : square M A T H) (MA_eq_one : distance M A = 1) 
  (B_on_AT : on_line B A T) (angle_condition : angle M B T = 3.5 * angle B M T) : ℝ := 
    by
      -- Let's define the area of the triangle as shown in the problem
      let area_BMT : ℝ := (√3 - 1) / 2 
      -- Return the area
      exact area_BMT

theorem find_area_of_triangle_BMT :
  ∀ (M A T B : Point) (square_MATH : square M A T H) (MA_eq_one : distance M A = 1)
    (B_on_AT : on_line B A T) (angle_condition : angle M B T = 3.5 * angle B M T),
    area_of_triangle_BMT M A T B square_MATH MA_eq_one B_on_AT angle_condition = (√3 - 1) / 2 :=
  by sorry

end find_area_of_triangle_BMT_l776_776645


namespace least_three_digit_multiple_of_13_l776_776777

theorem least_three_digit_multiple_of_13 : ∃ n : ℕ, n ≥ 100 ∧ n % 13 = 0 ∧ ∀ m : ℕ, m ≥ 100 → m % 13 = 0 → n ≤ m :=
begin
  use 104,
  split,
  { exact nat.le_of_eq rfl },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero rfl) },
  { intros m hm hmod,
    have h8 : 8 * 13 = 104 := rfl,
    rw ←h8,
    exact nat.le_mul_of_pos_left (by norm_num) },
end

end least_three_digit_multiple_of_13_l776_776777


namespace integral_floor_x_sq_l776_776934

variable {f : ℝ → ℝ} {f' : ℝ → ℝ} (k : ℝ)
  (h : k > 1)
  [∀ (x : ℝ), DifferentiableAt ℝ f x]
  [∀ (x : ℝ), DifferentiableAt ℝ f' x]

theorem integral_floor_x_sq (f : ℝ → ℝ) (k : ℝ) (h : k > 1)
  (hf : ∀ x, DifferentiableAt ℝ f x) (hf' : ∀ x, DifferentiableAt ℝ f' x) :
  (∫ x in 1..k, ⌊x^2⌋ * f' x) = ⌊k^2⌋ * f k - ∑ n in finset.range (⌊k^2⌋.toNat + 1), f (sqrt n) :=
by
  sorry

end integral_floor_x_sq_l776_776934


namespace xn_bounds_l776_776705

theorem xn_bounds (x : ℕ → ℝ) (h : ∀ n, x n > 0) 
  (h_eq : ∀ n, x n ^ n = ∑ j in Finset.range n, x n ^ j) : 
  ∀ n, 2 - 1 / 2^(n-1) ≤ x n ∧ x n < 2 - 1 / 2^n :=
by
  sorry

end xn_bounds_l776_776705


namespace arc_BD_measure_l776_776020

-- Definitions
variables (A B C D : Point)
variable (angle_ADC : ℝ)
variable (is_extension : is_extension_of_diameter A B C)
variable (is_tangent : is_tangent CD)
variable (angle_ADC_eq : angle_ADC = 110)

-- The statement to prove
theorem arc_BD_measure : 
  angular_measure_arc BD = 40 :=
sorry

end arc_BD_measure_l776_776020


namespace sum_numerator_denominator_q_l776_776552

noncomputable def probability_q : ℚ := 32 / 70

theorem sum_numerator_denominator_q :
  (probability_q.num + probability_q.denom) = 51 :=
by
  -- Definitions and assumptions (corresponding to conditions)
  let a := {a_1, a_2, a_3, a_4 | a_1, a_2, a_3, a_4 ∈ (finset.range 1000).val}
  let b := {b_1, b_2, b_3, b_4 | b_1, b_2, b_3, b_4 ∈ (finset.range 1000).val \ a}
  -- Further definitions can be added as necessary, following the problem conditions

  -- Sorry is used to avoid proving manually
  sorry

end sum_numerator_denominator_q_l776_776552


namespace increasing_function_condition_l776_776721

variable {x : ℝ} {a : ℝ}

theorem increasing_function_condition (h : 0 < a) :
  (∀ x ≥ 1, deriv (λ x => x^3 - a * x) x ≥ 0) ↔ (0 < a ∧ a ≤ 3) :=
by
  sorry

end increasing_function_condition_l776_776721


namespace f_increasing_f_odd_is_a_one_l776_776588

variable {a : ℝ}

def f (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem f_increasing : ∀ a : ℝ, ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
by
  intros a x1 x2 hx
  sorry

theorem f_odd_is_a_one : f 0 = 0 → a = 1 :=
by
  intro h
  have h1 : f 0 = a - 2 / 2 := by sorry
  rw [h1] at h
  sorry

end f_increasing_f_odd_is_a_one_l776_776588


namespace compound_analysis_l776_776397

noncomputable def molecular_weight : ℝ := 18
noncomputable def atomic_weight_nitrogen : ℝ := 14.01
noncomputable def atomic_weight_hydrogen : ℝ := 1.01

theorem compound_analysis :
  ∃ (n : ℕ) (element : String), element = "hydrogen" ∧ n = 4 ∧
  (∃ remaining_weight : ℝ, remaining_weight = molecular_weight - atomic_weight_nitrogen ∧
   ∃ k, remaining_weight / atomic_weight_hydrogen = k ∧ k = n) :=
by
  sorry

end compound_analysis_l776_776397


namespace number_of_coins_l776_776803

-- Define the conditions
def equal_number_of_coins (x : ℝ) :=
  ∃ n : ℝ, n = x

-- Define the total value condition
def total_value (x : ℝ) :=
  x + 0.50 * x + 0.25 * x = 70

-- The theorem to be proved
theorem number_of_coins (x : ℝ) (h1 : equal_number_of_coins x) (h2 : total_value x) : x = 40 :=
by sorry

end number_of_coins_l776_776803


namespace integral_identity_proof_l776_776438

noncomputable def integral_identity : Prop :=
  ∫ x in (0 : Real)..(Real.pi / 2), (Real.cos (Real.cos x))^2 + (Real.sin (Real.sin x))^2 = Real.pi / 2

theorem integral_identity_proof : integral_identity :=
sorry

end integral_identity_proof_l776_776438


namespace mingi_initial_tomatoes_l776_776297

theorem mingi_initial_tomatoes (n m r : ℕ) (h1 : n = 15) (h2 : m = 20) (h3 : r = 6) : n * m + r = 306 := by
  sorry

end mingi_initial_tomatoes_l776_776297


namespace least_positive_three_digit_multiple_of_11_l776_776759

theorem least_positive_three_digit_multiple_of_11 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n ∧ n = 110 :=
by {
  use 110,
  split,
  { exact Nat.le_refl _ },
  split,
  { norm_num },
  split,
  { use 10,
    norm_num },
  { refl },
  sorry
}

end least_positive_three_digit_multiple_of_11_l776_776759


namespace magnitude_difference_l776_776515

variable (z1 z2 : ℂ)
variable h1 : z1 = 3 - 10 * complex.i
variable h2 : z2 = 2 + 5 * complex.i

theorem magnitude_difference : ∥(z1 - z2)∥ = real.sqrt 26 := by
  sorry

end magnitude_difference_l776_776515


namespace min_value_fraction_sum_l776_776978

open Real

theorem min_value_fraction_sum (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h_sum : x + y + z = 2) :
    ∃ m, m = (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ∧ m = 9/4 :=
by
  sorry

end min_value_fraction_sum_l776_776978


namespace circumscribed_sphere_surface_area_l776_776146

noncomputable def length_PC := 2 * Real.sqrt 3
def face_condition (P A B C : Type) := ∀ {x y z : P}, ∃ (h : x ≠ y), 
  right_triangle x y z

def right_triangle (a b c : Type) := ∃ (a b c : Type), a a = a(a(a))

theorem circumscribed_sphere_surface_area (P A B C : Type) [metric_space P] :
  face_condition P A B C → length_PC = 2 * Real.sqrt 3 → 
  surface_area_circumscribed_sphere P A B C = 12 * Real.pi :=
sorry

end circumscribed_sphere_surface_area_l776_776146


namespace divisors_of_30_l776_776192

theorem divisors_of_30 : ∃ (n : ℕ), n = 16 ∧ (∀ d : ℤ, d ∣ 30 → (d ≤ 30 ∧ d ≥ -30)) :=
by
  sorry

end divisors_of_30_l776_776192


namespace gigi_mushrooms_l776_776121

-- Define the conditions
def pieces_per_mushroom := 4
def kenny_pieces := 38
def karla_pieces := 42
def remaining_pieces := 8

-- Main theorem
theorem gigi_mushrooms : (kenny_pieces + karla_pieces + remaining_pieces) / pieces_per_mushroom = 22 :=
by
  sorry

end gigi_mushrooms_l776_776121


namespace ice_cream_melting_l776_776419

theorem ice_cream_melting :
  ∀ (r1 r2 : ℝ) (h : ℝ),
    r1 = 3 ∧ r2 = 10 →
    4 / 3 * π * r1^3 = π * r2^2 * h →
    h = 9 / 25 :=
by intros r1 r2 h hcond voldist
   sorry

end ice_cream_melting_l776_776419


namespace original_number_of_men_l776_776009

theorem original_number_of_men (x : ℕ) (h : 10 * x = 7 * (x + 10)) : x = 24 := 
by 
  -- Add your proof here 
  sorry

end original_number_of_men_l776_776009


namespace steer_weight_in_pounds_after_increase_l776_776048

noncomputable def initial_weight_kg : ℝ := 300
noncomputable def weight_increase_factor : ℝ := 1.15
noncomputable def kg_to_pounds_conversion : ℝ := 0.4536

theorem steer_weight_in_pounds_after_increase :
  (initial_weight_kg * weight_increase_factor) / kg_to_pounds_conversion ≈ 761 := by sorry

end steer_weight_in_pounds_after_increase_l776_776048


namespace lines_concurrent_l776_776051

noncomputable theory

variables {α : Type*} [EuclideanGeometry α] {A B C H M N : α}

def foot_of_perpendicular (A B C : α) : α := sorry -- Define the foot of the perpendicular from A to BC
def circumcenter (A B C : α) : α := sorry -- Define the circumcenter of triangle ABC

theorem lines_concurrent
  (h1 : acute_angled_triangle A B C)
  (h2 : H = foot_of_perpendicular A B C)
  (h3 : M = foot_of_perpendicular H A B)
  (h4 : N = foot_of_perpendicular H A C)
  (L_A : ∀ (x : α), perpendicular x A M N)
  (L_B : ∀ (y : α), perpendicular y B N C)
  (L_C : ∀ (z : α), perpendicular z C M B) :
  concurrent L_A L_B L_C :=
  sorry

end lines_concurrent_l776_776051


namespace sum_of_first_10_terms_l776_776140

variable (a : ℕ → ℝ)

-- Condition: the sequence {a_n} is an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, a (n + m) = a n + a m - a 0

-- Condition: a_3 + a_8 = 22
def condition2 (a : ℕ → ℝ) : Prop :=
a 3 + a 8 = 22

-- The problem is to prove that the sum of the first 10 terms of the sequence is 110
theorem sum_of_first_10_terms (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a)
  (h2 : condition2 a) : (∑ i in range 10, a i) = 110 := 
sorry

end sum_of_first_10_terms_l776_776140


namespace unique_integer_Tn_l776_776114

noncomputable def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem unique_integer_Tn (b : ℝ) (a : ℕ → ℝ) (n : ℕ) 
  (h_b : b = 4) 
  (h_sum : ∑ k in finset.range n, a k = 19) 
  (h_ak_pos : ∀ k, a k > 0) : 
  is_integer (∑ k in finset.range n, real.sqrt ((3 * k + 1)^2 + b * (a k)^2)) ↔ n = 17 :=
begin
  sorry
end

end unique_integer_Tn_l776_776114


namespace pie_wholesale_price_l776_776820

theorem pie_wholesale_price (p1 p2: ℕ) (d: ℕ) (k: ℕ) (h1: d > 0) (h2: k > 1)
  (price_list: List ℕ)
  (p_plus_1: price_list = [p1 + d, k * p1, p2 + d, k * p2]) 
  (h3: price_list.perm [64, 64, 70, 72]) : 
  p1 = 60 ∧ p2 = 60 := 
sorry

end pie_wholesale_price_l776_776820


namespace UVWXY_perimeter_l776_776241

theorem UVWXY_perimeter (U V W X Y Z : ℝ) 
  (hUV : UV = 5)
  (hVW : VW = 3)
  (hWY : WY = 5)
  (hYX : YX = 9)
  (hXU : XU = 7) :
  UV + VW + WY + YX + XU = 29 :=
by
  sorry

end UVWXY_perimeter_l776_776241


namespace knights_statements_count_l776_776303

-- Definitions for the conditions
variables (r l : ℕ) -- r is the number of knights, l is the number of liars
variables (hr : r ≥ 2) (hl : l ≥ 2)
variables (total_liar_statements : 2 * r * l = 230)

-- Theorem statement: Prove that the number of times "You are a knight!" was said is 526
theorem knights_statements_count : 
  let total_islanders := r + l in
  let total_statements := total_islanders * (total_islanders - 1) in
  let knight_statements := total_statements - 230 in
  knight_statements = 526 :=
by
  sorry

end knights_statements_count_l776_776303


namespace louie_junior_took_7_cookies_l776_776294

-- Let's define the problem in Lean
def cookies_taken_by_louie_junior : ℕ :=
  let cookies_eaten_by_lou_senior_day_1 := 3 in
  let cookies_taken_by_lou_senior_day_2 := 3 in
  let cookies_put_back_by_lou_senior := 2 in
  let cookies_remaining := 11 in
  let original_cookies := 22 in
  let cookies_eaten_by_lou_senior :=
    cookies_eaten_by_lou_senior_day_1 + (cookies_taken_by_lou_senior_day_2 - cookies_put_back_by_lou_senior) in
  let cookies_before_louie_junior := original_cookies - cookies_eaten_by_lou_senior in
  cookies_before_louie_junior - cookies_remaining

theorem louie_junior_took_7_cookies :
  cookies_taken_by_louie_junior = 7 :=
sorry

end louie_junior_took_7_cookies_l776_776294


namespace pie_prices_l776_776876

theorem pie_prices 
  (d k : ℕ) (hd : 0 < d) (hk : 1 < k) 
  (p1 p2 : ℕ) 
  (prices : List ℕ) (h_prices : prices = [64, 64, 70, 72]) 
  (hP1 : (p1 + d) ∈ prices)
  (hZ1 : (k * p1) ∈ prices)
  (hP2 : (p2 + d) ∈ prices)
  (hZ2 : (k * p2) ∈ prices)
  (h_different_prices : p1 + d ≠ k * p1 ∧ p2 + d ≠ k * p2) : 
  ∃ p1 p2, p1 ≠ p2 :=
sorry

end pie_prices_l776_776876


namespace range_of_a_l776_776287

def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 else 2 ^ x

theorem range_of_a (a : ℝ) : (f (f a) = 2 ^ (f a)) ↔ a ∈ set.Ici (2 / 3) :=
by
  sorry

end range_of_a_l776_776287


namespace find_wholesale_prices_l776_776842

-- Definitions based on conditions
def bakerPlusPrice (p : ℕ) (d : ℕ) : ℕ := p + d
def starPrice (p : ℕ) (k : ℚ) : ℕ := (p * k).toNat

-- The tuple of prices given
def prices : List ℕ := [64, 64, 70, 72]

-- Statement of the problem
theorem find_wholesale_prices (d : ℕ) (k : ℚ) (h_d : d > 0) (h_k : k > 1 ∧ k.denom = 1):
  ∃ p1 p2, 
  (p1 ∈ prices ∧ bakerPlusPrice p1 d ∈ prices ∧ starPrice p1 k ∈ prices) ∧
  (p2 ∈ prices ∧ bakerPlusPrice p2 d ∈ prices ∧ starPrice p2 k ∈ prices) :=
sorry

end find_wholesale_prices_l776_776842


namespace distance_is_13_angle_is_arctan_5_12_l776_776624

open Real

def point : ℝ × ℝ := (12, 5)

def distance_from_origin (p : ℝ × ℝ) : ℝ := 
  sqrt ((p.1)^2 + (p.2)^2)

def angle_with_x_axis (p : ℝ × ℝ) : ℝ :=
  arctan (p.2 / p.1)

theorem distance_is_13 : distance_from_origin point = 13 := 
  sorry

theorem angle_is_arctan_5_12 : angle_with_x_axis point = arctan (5 / 12) :=
  sorry

end distance_is_13_angle_is_arctan_5_12_l776_776624


namespace sum_of_specific_terms_l776_776594

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 2 then 2
  else sequence (n-1) + 3 * (n-1) - sequence (n-2)

theorem sum_of_specific_terms : 
  let a := sequence in
  a 2 + a 4 + a 6 + a 8 + a 10 + a 12 = 57 :=
by
  sorry

end sum_of_specific_terms_l776_776594


namespace mushrooms_gigi_cut_l776_776122

-- Definitions based on conditions in part a)
def pieces_per_mushroom := 4
def kenny_sprinkled := 38
def karla_sprinkled := 42
def pieces_remaining := 8

-- The total number of pieces is the sum of Kenny's, Karla's, and the remaining pieces.
def total_pieces := kenny_sprinkled + karla_sprinkled + pieces_remaining

-- The number of mushrooms GiGi cut up at the beginning is total_pieces divided by pieces_per_mushroom.
def mushrooms_cut := total_pieces / pieces_per_mushroom

-- The theorem to be proved.
theorem mushrooms_gigi_cut (h1 : pieces_per_mushroom = 4)
                           (h2 : kenny_sprinkled = 38)
                           (h3 : karla_sprinkled = 42)
                           (h4 : pieces_remaining = 8)
                           (h5 : total_pieces = kenny_sprinkled + karla_sprinkled + pieces_remaining)
                           (h6 : mushrooms_cut = total_pieces / pieces_per_mushroom) :
  mushrooms_cut = 22 :=
by
  sorry

end mushrooms_gigi_cut_l776_776122


namespace max_value_2x_minus_y_l776_776995

theorem max_value_2x_minus_y (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0) 
  (h2 : y + 1 ≥ 0) 
  (h3 : x + y + 1 ≤ 0): 
  (∃ x y, 2 * x - y = 1) :=
begin
  sorry
end

end max_value_2x_minus_y_l776_776995


namespace find_f_f_3_l776_776162

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2 / x

theorem find_f_f_3:
  f (f 3) = 13 / 9 := by 
  sorry

end find_f_f_3_l776_776162


namespace exists_segment_with_2n_blue_and_n_green_l776_776649

variable (n : ℕ)
variable (S : set ℕ)
variable (points : ℕ → ℕ)
variable (colors : ℕ → string)

-- Given conditions
axiom six_n_points : S = { x | x < 6 * n }
axiom blue_points : ∃ (blue : ℕ → Prop), (∀ x ∈ (1..4 * n), colors (points x) = "blue")
axiom green_points : ∃ (green : ℕ → Prop), (∀ x ∈ (1..2 * n), colors (points x) = "green")

-- Statement to prove
theorem exists_segment_with_2n_blue_and_n_green :
  ∃ (segment : finset ℕ), 
    (segment.card = 3 * n ∧ 
     (finset.filter (λ x, colors (points x) = "blue") segment).card = 2 * n ∧ 
     (finset.filter (λ x, colors (points x) = "green") segment).card = n) :=
sorry

end exists_segment_with_2n_blue_and_n_green_l776_776649


namespace number_of_ways_to_arrange_matches_l776_776937

open Nat

theorem number_of_ways_to_arrange_matches :
  (factorial 7) * (2 ^ 3) = 40320 := by
  sorry

end number_of_ways_to_arrange_matches_l776_776937


namespace least_positive_three_digit_multiple_of_11_l776_776758

theorem least_positive_three_digit_multiple_of_11 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n ∧ n = 110 :=
by {
  use 110,
  split,
  { exact Nat.le_refl _ },
  split,
  { norm_num },
  split,
  { use 10,
    norm_num },
  { refl },
  sorry
}

end least_positive_three_digit_multiple_of_11_l776_776758


namespace determine_wholesale_prices_l776_776826

-- Definitions and Conditions
-- Wholesale prices for days 1 and 2 are defined as p1 and p2
def wholesale_price_day1 : ℝ := p1
def wholesale_price_day2 : ℝ := p2

-- The price increment at "Baker Plus" store is a constant d
def price_increment_baker_plus : ℝ := d
-- The factor at "Star" store is a constant k > 1
def price_factor_star : ℝ := k
-- Given prices for two days: 64, 64, 70, 72 rubles
def given_prices : set ℝ := {64, 64, 70, 72}

-- Prices at the stores
def baker_plus_day1_price := wholesale_price_day1 + price_increment_baker_plus
def star_day1_price := wholesale_price_day1 * price_factor_star

def baker_plus_day2_price := wholesale_price_day2 + price_increment_baker_plus
def star_day2_price := wholesale_price_day2 * price_factor_star

-- Theorem to determine the wholesale prices for each day (p1 and p2)
theorem determine_wholesale_prices
    (baker_plus_day1_price ∈ given_prices)
    (star_day1_price ∈ given_prices)
    (baker_plus_day2_price ∈ given_prices)
    (star_day2_price ∈ given_prices)
    (h1 : baker_plus_day1_price ≠ star_day1_price)
    (h2 : baker_plus_day2_price ≠ star_day2_price) :
  (∃ p1 p2 d k > 1, 
      (p1 + d ∈ given_prices) ∧ (p1 * k ∈ given_prices) ∧
      (p2 + d ∈ given_prices) ∧ (p2 * k ∈ given_prices) ∧
      (p1 ≠ p2)) :=
  sorry

end determine_wholesale_prices_l776_776826


namespace train_second_speed_20_l776_776049

variable (x v: ℕ)

theorem train_second_speed_20 
  (h1 : (x / 40) + (2 * x / v) = (6 * x / 48)) : 
  v = 20 := by 
  sorry

end train_second_speed_20_l776_776049


namespace hyperbola_properties_l776_776954

section hyperbola_properties

variables {x y : ℝ}

def equation_hyperbola := 9 * y^2 - 16 * x^2 = 144

theorem hyperbola_properties :
  (equation_hyperbola → 
  (∃ a b c e : ℝ, 
    a = 4 ∧ b = 3 ∧ c = 5 ∧ e = 5 / 4 ∧ 
      (-foci : set (ℝ × ℝ)) = {(0, -5), (0, 5)} ∧
      (∃ m : ℝ, m = 4 / 3 ∧
        (∀ x : ℝ, ∃ y : ℝ, y = m * x ∨ y = -m * x)))) :=
sorry

end hyperbola_properties

end hyperbola_properties_l776_776954


namespace compute_binomial_10_3_eq_120_l776_776483

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l776_776483


namespace comb_10_3_eq_120_l776_776496

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l776_776496


namespace minimum_races_to_determine_top_3_l776_776915

/- Define the conditions -/
def maxHorsesPerRace : Nat := 6
def totalHorses : Nat := 30

/- Define the minimum number of races needed to determine the top 3 fastest horses -/
def minRacesNeededToDetermineTop3Horses (n : Nat) (k : Nat) : Nat := do
  if n = totalHorses ∧ k = maxHorsesPerRace then 7 else 0

/- The Proof Statement -/
theorem minimum_races_to_determine_top_3 (n k : Nat) (h : n = totalHorses ∧ k = maxHorsesPerRace) : 
  minRacesNeededToDetermineTop3Horses n k = 7 :=
begin
  sorry
end

end minimum_races_to_determine_top_3_l776_776915


namespace find_K_l776_776906

theorem find_K (s : ℝ) (h₁ : s = 3) (h₂ : 6 * s^2 = 4 * Real.pi * r^2) :
  ∃ K : ℝ, (4 / 3) * Real.pi * (sqrt (27 / (2 * Real.pi)))^3 = (K * sqrt 6 / sqrt Real.pi) ∧ K = 54 :=
by
  sorry

end find_K_l776_776906


namespace length_PR_l776_776324

variable (P Q R : Type) [Inhabited P] [Inhabited Q] [Inhabited R]
variable {xPR xQR xsinR : ℝ}
variable (hypotenuse_opposite_ratio : xsinR = (3/5))
variable (sideQR : xQR = 9)
variable (rightAngle : ∀ (P Q R : Type), P ≠ Q → Q ∈ line_through Q R)

theorem length_PR : (∃ xPR : ℝ, xPR = 15) :=
by
  sorry

end length_PR_l776_776324


namespace triangle_perimeter_is_six_l776_776574

noncomputable def perimeter_triangle (a b c : ℝ) (A B C : ℝ) (area : ℝ) : ℝ :=
  if a = 2 ∧ area = sqrt 3 ∧ a * cos C + sqrt 3 * a * sin C - b - c = 0 then a + b + c else 0

theorem triangle_perimeter_is_six :
  ∀ (a b c A B C : ℝ), 
  let area := sqrt 3 in
  (a = 2) → 
  (a * cos C + sqrt 3 * a * sin C - b - c = 0) → 
  (area_of_triangle a b c A = sqrt 3) → 
  a + b + c = 6 :=
begin
  intros,
  sorry
end

end triangle_perimeter_is_six_l776_776574


namespace angle_XOY_regular_octagon_l776_776368

theorem angle_XOY_regular_octagon (O X Y W Z : Point) (h : regular_octagon W X Y Z O) :
  angle X O Y = 22.5 :=
sorry

end angle_XOY_regular_octagon_l776_776368


namespace factorial_square_gt_power_l776_776315

theorem factorial_square_gt_power (n : ℕ) (h : n > 2) : (n!)^2 > n^n := by
  sorry

end factorial_square_gt_power_l776_776315


namespace probability_same_length_l776_776274

/-- Defining the set of all sides and diagonals of a regular hexagon. -/
def T : Finset ℚ := sorry

/-- There are exactly 6 sides in the set T. -/
def sides_count : ℕ := 6

/-- There are exactly 9 diagonals in the set T. -/
def diagonals_count : ℕ := 9

/-- The total number of segments in the set T. -/
def total_segments : ℕ := sides_count + diagonals_count

theorem probability_same_length :
  let prob_side := (6 : ℚ) / total_segments * (5 / (total_segments - 1))
  let prob_diagonal := (9 : ℚ) / total_segments * (4 / (total_segments - 1))
  prob_side + prob_diagonal = 17 / 35 := 
by
  admit

end probability_same_length_l776_776274


namespace correct_choice_l776_776158

open Real

-- Define the parabolas
def parabola_M (a b c : ℝ) : ℝ → ℝ := λ x, a * x^2 + b * x + c
def parabola_N (a b d : ℝ) : ℝ → ℝ := λ x, -a * x^2 + b * x + d

-- Define vertices P and Q
def vertex_P (a b c : ℝ) : ℝ × ℝ := let x := -b / (2 * a) in (x, a * x^2 + b * x + c)
def vertex_Q (a b d : ℝ) : ℝ × ℝ := let x := -b / (2 * a) in (x, -a * x^2 + b * x + d)

-- Proposition 1: If P lies on N, then Q lies on M
def prop1 (a b c d : ℝ) : Prop :=
  let P := vertex_P a b c
  let Q := vertex_Q a b d
  (parabola_N a b d P.1 = P.2) → (parabola_M a b c Q.1 = Q.2)

-- Proposition 2: If P does not lie on N, then Q does not lie on M
def prop2 (a b c d : ℝ) : Prop :=
  let P := vertex_P a b c
  let Q := vertex_Q a b d
  (parabola_N a b d P.1 ≠ P.2) → (parabola_M a b c Q.1 ≠ Q.2)

-- Theorem: Both Proposition 1 and Proposition 2 are true
theorem correct_choice (a b c d : ℝ) : prop1 a b c d ∧ prop2 a b c d :=
by
  sorry

end correct_choice_l776_776158


namespace binomial_coefficient_10_3_l776_776489

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l776_776489


namespace not_pass_first_quadrant_l776_776222

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  (1/5)^(x + 1) + m

theorem not_pass_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, (1/5)^(x+1) + m ≤ 0) ↔ m ≤ -(1/5) :=
  by
  sorry

end not_pass_first_quadrant_l776_776222


namespace copper_wire_length_l776_776357

theorem copper_wire_length
  (ρ : ℝ := 8900)                           -- Density of copper in kg/m³
  (V : ℝ := 0.5 * 10^(-3))                   -- Volume of copper in m³
  (diagonal_mm : ℝ := 2)                    -- Diagonal of the square cross-section in mm
  (length_threshold : ℝ := 225) :            -- Length threshold in meters
  let s_mm := diagonal_mm / Real.sqrt 2 in    -- Side length in mm
  let s := s_mm * 10^(-3) in                  -- Side length in meters
  let A := s^2 in                             -- Cross-sectional area in m²
  let L := V / A in                           -- Length of the wire in meters
  L > length_threshold := 
by
  -- Proof goes here
  sorry

end copper_wire_length_l776_776357


namespace son_present_age_l776_776385

variable (S F : ℕ)

-- Define the conditions
def fatherAgeCondition := F = S + 35
def twoYearsCondition := F + 2 = 2 * (S + 2)

-- The proof theorem
theorem son_present_age : 
  fatherAgeCondition S F → 
  twoYearsCondition S F → 
  S = 33 :=
by
  intros h1 h2
  sorry

end son_present_age_l776_776385


namespace no_m_slope_eq_2_minus_2m_l776_776164

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x - (1 / x) - 2 * m * Real.log x

theorem no_m_slope_eq_2_minus_2m (h1 : ∀ x : ℝ, 0 < x) :
  (∃ (x1 x2 m : ℝ), x1 ≠ x2 ∧ (∂ (f x1 m) / ∂ x) = 0 ∧ (∂ (f x2 m) / ∂ x) = 0 ∧
   (x1 + x2 = 2 * m) ∧ (x1 * x2 = 1) ∧ (∃ k : ℝ, k = (f x2 m - f x1 m) / (x2 - x1) ∧ k = 2 - 2 * m)) 
  → False :=
by
  sorry

end no_m_slope_eq_2_minus_2m_l776_776164


namespace minimum_games_for_80_percent_l776_776712

theorem minimum_games_for_80_percent :
  ∃ N : ℕ, ( ∀ N' : ℕ, (1 + N') / (5 + N') * 100 < 80 → N < N') ∧ (1 + N) / (5 + N) * 100 ≥ 80 :=
sorry

end minimum_games_for_80_percent_l776_776712


namespace a0_value_a10_plus_a11_value_l776_776557

theorem a0_value :
  (∃ (a : ℕ → ℤ), (x - 1) ^ 21 = ∑ i in Finset.range (21 + 1), a i * x ^ i) →
  (a 0 = -1) := by
  sorry

theorem a10_plus_a11_value :
  (∃ (a : ℕ → ℤ), (x - 1) ^ 21 = ∑ i in Finset.range (21 + 1), a i * x ^ i) →
  (a 10 + a 11 = 0) := by
  sorry

end a0_value_a10_plus_a11_value_l776_776557


namespace total_number_candies_l776_776293

theorem total_number_candies (n x : ℕ) (h1 : ∀ i ∈ List.range n, x = (List.sum (List.eraseNth (List.replicate n x) i)) - 7) (h2 : x > 1) : ∃ (total_candies : ℕ), total_candies = 21 :=
by
  sorry

end total_number_candies_l776_776293


namespace subset_A_implies_m_values_l776_776597

theorem subset_A_implies_m_values (m : ℝ) :
  {y ∈ {3, m^2} → y ∈ {1, 3, 2 * m + 3}} → (m = 1 ∨ m = 3) :=
by sorry

end subset_A_implies_m_values_l776_776597


namespace distinct_words_count_correct_l776_776616

-- Define the language's words as sequences of 10 digits composed of 0s and 1s
def is_word (w : List ℕ) : Prop :=
  w.length = 10 ∧ (∀ d ∈ w, d = 0 ∨ d = 1)

-- Define the transformation operation on words
def can_transform (w1 w2 : List ℕ) : Prop :=
  ∃ (i j : ℕ) (removed : List ℕ),
    0 ≤ i ∧ i ≤ j ∧ j < w1.length ∧
    removed = w1.drop i |>.take (j - i + 1) ∧
    (removed.sum % 2 = 0) ∧
    w2 = (w1.take i ++ removed.reverse ++ w1.drop (j + 1))

-- Define two words as synonyms under the transformation
def are_synonyms (w1 w2 : List ℕ) : Prop :=
  (is_word w1 ∧ is_word w2) ∧ can_transform w1 w2

-- Count distinct words given the synonym equivalence
def distinct_word_count : ℕ := sorry

-- Statement representing the mathematical proof problem
theorem distinct_words_count_correct :
  distinct_word_count = 56 :=
sorry

end distinct_words_count_correct_l776_776616


namespace divisors_of_30_l776_776203

theorem divisors_of_30 : 
  let positive_divisors := {1, 2, 3, 5, 6, 10, 15, 30}
  in 2 * positive_divisors.card = 16 :=
by
  let positive_divisors := {1, 2, 3, 5, 6, 10, 15, 30}
  have h1 : positive_divisors.card = 8 := sorry
  have h2 : 2 * 8 = 16 := sorry
  exact h2

end divisors_of_30_l776_776203


namespace poly_div_l776_776961

theorem poly_div (f : ℤ[X]) (q : ℤ[X]) (r : ℤ) :
  f = X^5 - 20 * X^3 + 15 * X^2 - 18 * X + 12 →
  (X - 2) * q + C r = f →
  q = X^4 + 2 * X^3 - 16 * X^2 - 17 * X - 52 ∧ r = -92 :=
by
  intros h1 h2
  sorry

end poly_div_l776_776961


namespace multiple_of_four_l776_776740

theorem multiple_of_four (n : ℕ) (h1 : ∃ k : ℕ, 12 + 4 * k = n) (h2 : 21 = (n - 12) / 4 + 1) : n = 96 := 
sorry

end multiple_of_four_l776_776740


namespace emissions_inspection_systematic_l776_776110

-- Define the conditions in Lean 4
def last_digit_condition (n : ℕ) : Bool := 
  (n % 10) = 6

-- Define the concept of systematic sampling based on the condition provided.
def is_systematic_sampling {α : Type} (selection: α → Bool) : Prop :=
  ∃ (k : ℕ), ∀ (x : α), selection x ↔ (x % k = 6)

-- State that choosing vehicles with a license plate ending in 6 is systematic sampling
theorem emissions_inspection_systematic :
  is_systematic_sampling last_digit_condition :=
sorry

end emissions_inspection_systematic_l776_776110


namespace proof_triple_positive_reals_l776_776531

theorem proof_triple_positive_reals 
  (x y z : ℝ) 
  (h1 : 2 * x * real.sqrt (x + 1) - y * (y + 1) = 1)
  (h2 : 2 * y * real.sqrt (y + 1) - z * (z + 1) = 1)
  (h3 : 2 * z * real.sqrt (z + 1) - x * (x + 1) = 1)
  (h4 : 0 < x) 
  (h5 : 0 < y) 
  (h6 : 0 < z) : 
  x = (1 + real.sqrt 5) / 2 ∧ y = (1 + real.sqrt 5) / 2 ∧ z = (1 + real.sqrt 5) / 2 :=
by
  sorry

end proof_triple_positive_reals_l776_776531


namespace sum_of_squares_of_a1_l776_776112

theorem sum_of_squares_of_a1 (F : ℕ → ℕ) (a : ℕ → ℝ) (hF1 : F 1 = 0) (hF2 : F 2 = 1) 
  (hF_rec : ∀ n, F (n+2) = F (n+1) + F n) (h_a_rec : ∀ n, a (n + 1) = 1 / (1 + a n)) 
  (h_initial : a 1 = a 2012) :
  (let α := (-1 + real.sqrt 5) / 2 in 
   let β := (-1 - real.sqrt 5) / 2 in 
   α^2 + β^2 = 3) :=
begin
  sorry
end

end sum_of_squares_of_a1_l776_776112


namespace perimeter_of_triangle_ABC_l776_776970

noncomputable def ellipse_perimeter : ℝ := sorry

theorem perimeter_of_triangle_ABC (a : ℝ) (h_a : a > 1) (F₁ F₂ A B C : ℝ × ℝ)
  (h_ellipse : ∀ x y : ℝ, (x, y) ∈ set_of (λ p : ℝ × ℝ, p.1^2 / a^2 + p.2^2 = 1))
  (h_foci : F₁ = (-sqrt (a^2 - 1), 0) ∧ F₂ = (sqrt (a^2 - 1), 0))
  (h_top : A = (0, 1))
  (l : ℝ × ℝ → set ℝ × ℝ)
  (h_line : ∃ B C, B ≠ C ∧ B ∈ l F₁ ∧ C ∈ l F₁ ∧ (B, C) ∈ set_of (λ p : ℝ × ℝ, (p.1^2 / a^2 + p.2^2 = 1)))
  (h_vert_bisect : let D := ((F₂.1 + A.1) / 2, (F₂.2 + A.2) / 2) in 
    ∃ l, ∀ x, x ∈ l F₁ → x ∈ l D)
  : ellipse_perimeter = 4 * (2 * sqrt 3 / 3) := sorry

end perimeter_of_triangle_ABC_l776_776970


namespace imag_part_quotient_l776_776577

def z1 : ℂ := 1 + complex.I
def z2 : ℂ := 1 - complex.I

theorem imag_part_quotient : complex.im (z1 / z2) = 1 :=
sorry

end imag_part_quotient_l776_776577


namespace pages_in_second_chapter_l776_776881

theorem pages_in_second_chapter (total_pages_in_book : ℕ) (pages_in_first_chapter : ℕ) :
  total_pages_in_book = 93 → pages_in_first_chapter = 60 → (total_pages_in_book - pages_in_first_chapter) = 33 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end pages_in_second_chapter_l776_776881


namespace divisors_of_30_l776_776202

theorem divisors_of_30 : 
  let positive_divisors := {1, 2, 3, 5, 6, 10, 15, 30}
  in 2 * positive_divisors.card = 16 :=
by
  let positive_divisors := {1, 2, 3, 5, 6, 10, 15, 30}
  have h1 : positive_divisors.card = 8 := sorry
  have h2 : 2 * 8 = 16 := sorry
  exact h2

end divisors_of_30_l776_776202


namespace correct_statement_is_4_l776_776907

-- Define the function to check the range
def func_range : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 0.7^(1/x)}

-- Define the lines and the condition for parallelism
def is_parallel (a : ℝ) : Prop :=
  let l1 := (λ x y : ℝ, 2*x + a*y - 1 = 0)
  let l2 := (λ x y : ℝ, (a-1)*x - a*y - 1 = 0)
  ∀ (x y : ℝ), l1 x y = l2 x y

-- Define the equation of the line
def has_equal_intercepts (A : ℝ × ℝ) (line_eq : ℝ → ℝ → Prop) : Prop :=
  line_eq A.1 A.2 ∧ ∃ k : ℝ, line_eq k 0 ∧ line_eq 0 k

-- Define the lateral surface area of the cylinder
def lateral_surface_area (r : ℝ) : ℝ := 2 * real.pi * r * 2 * r

-- Define the surface area of the sphere
def surface_area_sphere (r : ℝ) : ℝ := 4 * real.pi * r^2

theorem correct_statement_is_4 :
  (∀ y, y ∈ func_range → y > 0) ∧
  ¬ (∀ a, is_parallel a → a = -1) ∧
  ¬ has_equal_intercepts (1, 2) (λ x y, x + y = 3) ∧
  (∃ d : ℝ, ∀ r : ℝ, d = 2 * r → lateral_surface_area r = surface_area_sphere r) →
  true :=
by
  sorry

end correct_statement_is_4_l776_776907


namespace range_alpha_div_three_l776_776139

open Real

theorem range_alpha_div_three (α : ℝ) (k : ℤ) :
  sin α > 0 → cos α < 0 → sin (α / 3) > cos (α / 3) →
  ∃ k : ℤ,
    (2 * k * π + π / 4 < α / 3 ∧ α / 3 < 2 * k * π + π / 3) ∨
    (2 * k * π + 5 * π / 6 < α / 3 ∧ α / 3 < 2 * k * π + π) :=
by
  intros
  sorry

end range_alpha_div_three_l776_776139


namespace least_three_digit_multiple_13_l776_776773

theorem least_three_digit_multiple_13 : 
  ∃ n : ℕ, (n ≥ 100) ∧ (∃ k : ℕ, n = 13 * k) ∧ ∀ m, m < n → (m < 100 ∨ ¬∃ k : ℕ, m = 13 * k) :=
by
  sorry

end least_three_digit_multiple_13_l776_776773


namespace part1_i_part1_ii_part2_part2_expectation_l776_776117

noncomputable theory
open ProbabilityTheory

variables {μ σ : ℝ} {k : ℕ} {Y X : ℝ}

-- Given conditions
def given_distribution (X : ℝ → ℝ) (μ : ℝ) (σ : ℝ) : Prop := 
∀ x, X x = μ + σ * sorry -- X follows N(μ, σ^2)

def average_distribution (Y : ℝ → ℝ) (μ : ℝ) (σ : ℝ) (k : ℕ) : Prop :=
∀ y, Y y = μ + (σ / sqrt (k : ℝ)) * sorry -- Y follows N(μ, σ^2 / k)

def small_probability_event (p : ℝ) : Prop :=
p < 0.05

-- Part 1(i)
theorem part1_i (h : given_distribution X 1000 50) (k_val : k = 25) : 
  let Y := λ Y, average_distribution Y 1000 (50 * (1 / sqrt 25)) 25 in
  ∀ Y, P (Y ≤ 980) = 0.02275 := sorry

-- Part 1(ii)
theorem part1_ii (h : small_probability_event 0.02275) : 
  0.02275 < 0.05 := by 
  exact h

-- Part 2
def box1 : ℕ × ℕ := (6, 2) -- Box 1, 6 loaves, 2 black
def box2 : ℕ × ℕ := (8, 3) -- Box 2, 8 loaves, 3 black

def dist_table (p0 p1 p2 : ℝ) : Prop := 
  p0 = 53/140 ∧ p1 = 449/840 ∧ p2 = 73/840

theorem part2 : 
  let total_prob := 1 in
  ∃ p0 p1 p2 : ℝ, dist_table p0 p1 p2 ∧ (p0 + p1 + p2 = total_prob) := sorry

theorem part2_expectation :
  let E := (0 * (53/140) + 1 * (449/840) + 2 * (73/840)) in
  E = 17/24 := sorry

end part1_i_part1_ii_part2_part2_expectation_l776_776117


namespace car_return_speed_l776_776883

variable (d : ℕ) (r : ℕ)
variable (H0 : d = 180)
variable (H1 : ∀ t1 : ℕ, t1 = d / 90)
variable (H2 : ∀ t2 : ℕ, t2 = d / r)
variable (H3 : ∀ avg_rate : ℕ, avg_rate = 2 * d / (d / 90 + d / r))
variable (H4 : avg_rate = 60)

theorem car_return_speed : r = 45 :=
by sorry

end car_return_speed_l776_776883


namespace divisors_of_30_l776_776194

theorem divisors_of_30 : ∃ (n : ℕ), n = 16 ∧ (∀ d : ℤ, d ∣ 30 → (d ≤ 30 ∧ d ≥ -30)) :=
by
  sorry

end divisors_of_30_l776_776194


namespace divisors_of_30_count_l776_776186

theorem divisors_of_30_count : 
  ∃ S : Set ℤ, (∀ d ∈ S, 30 % d = 0) ∧ S.card = 16 :=
by
  sorry

end divisors_of_30_count_l776_776186


namespace max_non_managers_correct_l776_776035

noncomputable def max_non_managers_company : ℕ :=
  let marketing_projects := 6
  let hr_projects := 4
  let finance_projects := 5

  let marketing_managers := 9
  let hr_managers := 5
  let finance_managers := 6

  let marketing_nm :=  marketing_managers * 38 / 9 + (marketing_projects / 3).ceil + 2
  let hr_nm := hr_managers * 23 / 5 + (hr_projects / 3).ceil + 2
  let finance_nm := finance_managers * 31 / 6 + (finance_projects / 3).ceil + 2

  marketing_nm + hr_nm + finance_nm

theorem max_non_managers_correct : max_non_managers_company = 104 := by
  sorry

end max_non_managers_correct_l776_776035


namespace pie_wholesale_price_l776_776821

theorem pie_wholesale_price (p1 p2: ℕ) (d: ℕ) (k: ℕ) (h1: d > 0) (h2: k > 1)
  (price_list: List ℕ)
  (p_plus_1: price_list = [p1 + d, k * p1, p2 + d, k * p2]) 
  (h3: price_list.perm [64, 64, 70, 72]) : 
  p1 = 60 ∧ p2 = 60 := 
sorry

end pie_wholesale_price_l776_776821


namespace peanut_butter_revenue_l776_776667

theorem peanut_butter_revenue :
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  plantation_length * plantation_width * peanuts_per_sqft * butter_from_peanuts_ratio / 1000 * butter_price_per_kg = 31250 := 
by
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  sorry

end peanut_butter_revenue_l776_776667


namespace remove_blue_balls_to_achieve_desired_red_percentage_l776_776890

-- Definition of the problem conditions
def total_balls := 150
def red_percentage_initial := 0.40
def red_balls := (red_percentage_initial * total_balls).toNat
def blue_balls_initial := total_balls - red_balls
def red_percentage_desired := 0.80

-- Theorem statement for the proof
theorem remove_blue_balls_to_achieve_desired_red_percentage :
  ∃ x : ℕ, 
  x = 75 ∧
  ((red_balls).toFloat / ((total_balls - x).toFloat) = red_percentage_desired) :=
by
  -- numerical computations and proofs are skipped via sorry
  sorry

end remove_blue_balls_to_achieve_desired_red_percentage_l776_776890


namespace magnitude_of_b_l776_776973

variable (a b : ℝ × ℝ)
variable (mag : ℝ)
variable h₁ : a = (1, -2)
variable h₂ : a + b = (0, 2)
variable h₃ : mag = sqrt ((-1)^2 + 4^2)

theorem magnitude_of_b : 
  ∃ b, a + b = (0, 2) → ((fst b)^2 + (snd b)^2)^0.5 = mag := 
by
  use (-1, 4)
  intro h
  rw [h₁, h₂] at h
  simp at h
  split
  rw [h, h₃]
  simp
  sorry

end magnitude_of_b_l776_776973


namespace mul_3_6_0_5_l776_776090

theorem mul_3_6_0_5 : 3.6 * 0.5 = 1.8 :=
by
  sorry

end mul_3_6_0_5_l776_776090


namespace least_three_digit_multiple_of_13_l776_776781

theorem least_three_digit_multiple_of_13 : ∃ n : ℕ, n ≥ 100 ∧ n % 13 = 0 ∧ ∀ m : ℕ, m ≥ 100 → m % 13 = 0 → n ≤ m :=
begin
  use 104,
  split,
  { exact nat.le_of_eq rfl },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero rfl) },
  { intros m hm hmod,
    have h8 : 8 * 13 = 104 := rfl,
    rw ←h8,
    exact nat.le_mul_of_pos_left (by norm_num) },
end

end least_three_digit_multiple_of_13_l776_776781


namespace parabola_focus_directrix_distance_l776_776239

theorem parabola_focus_directrix_distance (p l : ℝ) (hp : p > 0) (hl_cond : l = 1)
    (distance_cond : (p / 2) + l = 3) :
    p = 4 → ((p / 2) - (- p / 2)) = 4 :=
by
  intros hp_eq
  rw hp_eq
  sorry

end parabola_focus_directrix_distance_l776_776239


namespace red_bordered_area_l776_776413

theorem red_bordered_area (r₁ r₂ : ℝ) (A₁ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 6) (h₃ : A₁ = 37) :
  ∃ A₂ : ℝ, A₂ = 83.25 :=
by {
  -- We use the fact that the area relationships are proportional to the square of radii.
  let k := (r₂ / r₁)^2,
  use k * A₁,
  have h_k : k = (6/4)^2, by simp [h₁, h₂],
  simp [h₃, h_k],
  -- Provided that A₂ indeed equals the calculated value.
  norm_num,
  sorry
}

end red_bordered_area_l776_776413


namespace tan_alpha_equals_two_l776_776157

-- Define the point P and the conditions involving x and y
def P : ℝ × ℝ := (1, 2)

-- Define α as an angle in the trigonometric sense
variable {α : ℝ}

-- Define the trigonometric tangent function
def tan (α : ℝ) : ℝ := real.sin α / real.cos α

-- State the main theorem to be proved
theorem tan_alpha_equals_two (h : ∃ α, tan α = 2 ∧ ∃ x y, P = (x, y) ∧ x = 1 ∧ y = 2) : tan α = 2 :=
by
  sorry

end tan_alpha_equals_two_l776_776157


namespace sec_seven_pi_div_four_l776_776525

noncomputable def sec (x : ℝ) := 1 / real.cos x

theorem sec_seven_pi_div_four : sec (7 * real.pi / 4) = real.sqrt 2 :=
by sorry

end sec_seven_pi_div_four_l776_776525


namespace desserts_brought_by_mom_l776_776675

-- Definitions for the number of each type of dessert
def num_coconut := 1
def num_meringues := 2
def num_caramel := 7

-- Conditions from the problem as definitions
def total_desserts := num_coconut + num_meringues + num_caramel = 10
def fewer_coconut_than_meringues := num_coconut < num_meringues
def most_caramel := num_caramel > num_meringues
def josef_jakub_condition := (num_coconut + num_meringues + num_caramel) - (4 * 2) = 1

-- We need to prove the answer based on these conditions
theorem desserts_brought_by_mom :
  total_desserts ∧ fewer_coconut_than_meringues ∧ most_caramel ∧ josef_jakub_condition → 
  num_coconut = 1 ∧ num_meringues = 2 ∧ num_caramel = 7 :=
by sorry

end desserts_brought_by_mom_l776_776675


namespace sufficient_but_not_necessary_condition_l776_776611

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x = 0 → (x^2 - 2 * x = 0)) ∧ (∃ y : ℝ, y ≠ 0 ∧ y ^ 2 - 2 * y = 0) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l776_776611


namespace mul_3_6_0_5_l776_776089

theorem mul_3_6_0_5 : 3.6 * 0.5 = 1.8 :=
by
  sorry

end mul_3_6_0_5_l776_776089


namespace derivative_at_zero_l776_776916

noncomputable def f : ℝ → ℝ :=
λ x, if x = 0 then 0 else (2^(Real.tan x) - 2^(Real.sin x)) / x^2

theorem derivative_at_zero :
  Real.deriv f 0 = Real.log (Real.sqrt 2) :=
sorry

end derivative_at_zero_l776_776916


namespace smallest_three_digit_multiple_of_13_l776_776789

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 13 = 0 → n ≤ m :=
⟨104, by sorry⟩

end smallest_three_digit_multiple_of_13_l776_776789


namespace coefficient_of_x3y5_in_binomial_expansion_l776_776364

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  nat.choose n k

theorem coefficient_of_x3y5_in_binomial_expansion :
  let n := 8;
  let k := 3;
  let a := (2 : ℚ) / 3;
  let b := -(1 : ℚ) / 3;
  let x := 1;  -- Coefficients only involve rational numbers
  let y := 1;
  binomial_coefficient n k * (a ^ k) * (b ^ (n - k)) = -448 / 6561 :=
by
  sorry

end coefficient_of_x3y5_in_binomial_expansion_l776_776364


namespace tangent_periodicity_l776_776151

theorem tangent_periodicity (ω m : ℝ) (hω : ω > 0) (h : ∃ x₁ x₂ : ℝ, x₂ = x₁ + 2 * π ∧ tan (ω * x₁) = m ∧ tan (ω * x₂) = m) : ω = 1 / 2 := 
sorry

end tangent_periodicity_l776_776151


namespace total_houses_in_development_l776_776620

-- Definitions for conditions
def G : ℕ := 50  -- Number of houses with a two-car garage
def P : ℕ := 40  -- Number of houses with a swimming pool
def B : ℕ := 35  -- Number of houses with both

-- The number of total houses in the development to be proven
def Total : ℕ := G + P - B

theorem total_houses_in_development : Total = 55 := by
  -- Provided conditions imply the following values
  have G_val : G = 50 := rfl
  have P_val : P = 40 := rfl
  have B_val : B = 35 := rfl

  -- Perform the calculation for Total
  calc
    Total = G + P - B := rfl
    ...  = 50 + 40 - 35 := by rw [G_val, P_val, B_val]
    ...  = 55 := rfl

end total_houses_in_development_l776_776620


namespace intersection_A_B_l776_776697

noncomputable def set_A : Set ℝ := { x | 2 ≤ x ∧ x < 4 }
noncomputable def set_B : Set ℝ := { x | 3 ≤ x }

theorem intersection_A_B :
  set_A ∩ set_B = { x | 3 ≤ x ∧ x < 4 } := 
sorry

end intersection_A_B_l776_776697


namespace f_1991_eq_1988_l776_776337

def f (n : ℕ) : ℕ := sorry

theorem f_1991_eq_1988 : f 1991 = 1988 :=
by sorry

end f_1991_eq_1988_l776_776337


namespace intersection_is_empty_l776_776290

def A : Set ℝ := { x | x^2 - 3*x + 2 ≤ 0 }
def B : Set (ℝ × ℝ) := { p | p.1 ∈ A ∧ p.2 ∈ A }

theorem intersection_is_empty : A ∩ B = ∅ :=
by
  sorry

end intersection_is_empty_l776_776290


namespace least_three_digit_multiple_of_13_l776_776767

-- Define what it means to be a multiple of 13
def is_multiple_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13 * k

-- Define the range of three-digit numbers
def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define our main theorem
theorem least_three_digit_multiple_of_13 : ∃ (n : ℕ), is_three_digit_number n ∧ is_multiple_of_13 n ∧
  ∀ (m : ℕ), is_three_digit_number m ∧ is_multiple_of_13 m → n ≤ m :=
begin
  -- We state the theorem without proof for simplicity
  sorry
end

end least_three_digit_multiple_of_13_l776_776767


namespace coloring_count_l776_776210

theorem coloring_count : 
  ∀ (n : ℕ), n = 2021 → 
  ∃ (ways : ℕ), ways = 3 * 2 ^ 2020 :=
by
  intros n hn
  existsi 3 * 2 ^ 2020
  sorry

end coloring_count_l776_776210


namespace maximize_x4y3_l776_776655

theorem maximize_x4y3 (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : x + y = 40) : 
    (x, y) = (160 / 7, 120 / 7) ↔ x ^ 4 * y ^ 3 ≤ (160 / 7) ^ 4 * (120 / 7) ^ 3 := 
sorry

end maximize_x4y3_l776_776655


namespace pyramid_volume_l776_776923

noncomputable def point := (ℝ × ℝ)
noncomputable def triangle := point × point × point

def midpoint (p1 p2 : point) : point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def centroid (t : triangle) : point :=
  ((t.1.1 + t.2.1 + t.3.1) / 3, (t.1.2 + t.2.2 + t.3.2) / 3)

def triangle_area (t : triangle) : ℝ :=
  0.5 * abs (t.1.1 * (t.2.2 - t.3.2) + t.2.1 * (t.3.2 - t.1.2) + t.3.1 * (t.1.2 - t.2.2))

theorem pyramid_volume :
  let A : point := (0, 0)
    let B : point := (42, 0)
    let C : point := (18, 30)
    let D : point := midpoint A B
    let E : point := midpoint B C
    let F : point := midpoint C A
    let DEF : triangle := (D, E, F)
    let G : point := centroid DEF
    let height : ℝ := 20
    let area : ℝ := triangle_area DEF
    let volume : ℝ := (1 / 3) * area * height
  in volume = 1050 :=
by
  sorry

end pyramid_volume_l776_776923


namespace find_wholesale_prices_l776_776847

-- Definitions based on conditions
def bakerPlusPrice (p : ℕ) (d : ℕ) : ℕ := p + d
def starPrice (p : ℕ) (k : ℚ) : ℕ := (p * k).toNat

-- The tuple of prices given
def prices : List ℕ := [64, 64, 70, 72]

-- Statement of the problem
theorem find_wholesale_prices (d : ℕ) (k : ℚ) (h_d : d > 0) (h_k : k > 1 ∧ k.denom = 1):
  ∃ p1 p2, 
  (p1 ∈ prices ∧ bakerPlusPrice p1 d ∈ prices ∧ starPrice p1 k ∈ prices) ∧
  (p2 ∈ prices ∧ bakerPlusPrice p2 d ∈ prices ∧ starPrice p2 k ∈ prices) :=
sorry

end find_wholesale_prices_l776_776847


namespace prob1_f_in_A_prob2_h_in_A_l776_776133

-- Define the set A
def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def is_in_interval_range (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a ≤ b ∧ ∀ x, a ≤ x ∧ x ≤ b → f a ≤ f x ∧ f x ≤ f b

def is_in_set_A (f : ℝ → ℝ) : Prop :=
  is_monotonic f ∧ (∃ a b, is_in_interval_range f a b ∧ f a = a / 2 ∧ f b = b / 2)

-- Problem 1: Determine if f(x) = x^3 is in set A and find intervals [a, b]
theorem prob1_f_in_A : is_in_set_A (λ x : ℝ, x^3) :=
  sorry

-- Problem 1: Find all intervals [a, b]
@[simp]
def intervals_f_in_A : set (ℝ × ℝ) :=
  { (a, b) | a ≤ b ∧ ((f a = a / 2 ∧ f b = b / 2) → f = (λ x, x^3)) }

-- Problem 2: Determine if exists t such that h(x) = sqrt(x-1) + t is in set A
theorem prob2_h_in_A : ∃ t : ℝ, is_in_set_A (λ x : ℝ, sqrt (x - 1) + t) :=
  sorry

-- Problem 2: Find the range of t
@[simp]
def range_t_in_A : set ℝ :=
  { t | 0 < t ∧ t ≤ 1 / 2 }

end prob1_f_in_A_prob2_h_in_A_l776_776133


namespace abc_system_proof_l776_776693

theorem abc_system_proof (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a^2 + a = b^2) (h5 : b^2 + b = c^2) (h6 : c^2 + c = a^2) :
  (a - b) * (b - c) * (c - a) = 1 :=
by
  sorry

end abc_system_proof_l776_776693


namespace value_of_A_l776_776519

-- Definition of variables and conditions
def numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} -- Set of numbers from 1 to 10

variable (A B C D E F G: ℕ) -- Declaring the variables

axiom h1 : A ∈ numbers
axiom h2 : B ∈ numbers
axiom h3 : C ∈ numbers
axiom h4 : D ∈ numbers
axiom h5 : E ∈ numbers
axiom h6 : F ∈ numbers
axiom h7 : G ∈ numbers

axiom condition1 : F = abs (A - B) -- F must be the absolute difference of A and B
axiom condition2 : G = D + E -- G must be the sum of D and E

-- Use the conditions to prove that A = 9
theorem value_of_A : A = 9 :=
by
  sorry

end value_of_A_l776_776519


namespace find_min_max_l776_776977

noncomputable def f (x : ℝ) : ℝ := 1 / (4^x) - 1 / (2^x) + 1

theorem find_min_max :
  (∀ x : ℝ, x ∈ set.Icc (-3 : ℝ) 2 → f x ≥ 3 / 4) ∧
  (∀ x : ℝ, x ∈ set.Icc (-3 : ℝ) 2 → f x ≤ 57) ∧
  (∃ x : ℝ, x ∈ set.Icc (-3 : ℝ) 2 ∧ f x = 3 / 4) ∧
  (∃ x : ℝ, x ∈ set.Icc (-3 : ℝ) 2 ∧ f x = 57) :=
  sorry

end find_min_max_l776_776977


namespace train_speed_in_m_per_s_l776_776008

-- Define the given train speed in kmph
def train_speed_kmph : ℕ := 72

-- Define the conversion factor from kmph to m/s
def km_per_hour_to_m_per_second (speed_in_kmph : ℕ) : ℕ := (speed_in_kmph * 1000) / 3600

-- State the theorem
theorem train_speed_in_m_per_s (h : train_speed_kmph = 72) : km_per_hour_to_m_per_second train_speed_kmph = 20 := by
  sorry

end train_speed_in_m_per_s_l776_776008


namespace goods_train_speed_l776_776040

def train_speed_km_per_hr (length_of_train length_of_platform time_to_cross : ℕ) : ℕ :=
  let total_distance := length_of_train + length_of_platform
  let speed_m_s := total_distance / time_to_cross
  speed_m_s * 36 / 10

-- Define the conditions given in the problem
def length_of_train : ℕ := 310
def length_of_platform : ℕ := 210
def time_to_cross : ℕ := 26

-- Define the target speed
def target_speed : ℕ := 72

-- The theorem proving the conclusion
theorem goods_train_speed :
  train_speed_km_per_hr length_of_train length_of_platform time_to_cross = target_speed := by
  sorry

end goods_train_speed_l776_776040


namespace time_to_run_100_meters_no_wind_l776_776050

-- Definitions based on the conditions
variables (v w : ℝ)
axiom speed_with_wind : v + w = 9
axiom speed_against_wind : v - w = 7

-- The theorem statement to prove
theorem time_to_run_100_meters_no_wind : (100 / v) = 12.5 :=
by 
  sorry

end time_to_run_100_meters_no_wind_l776_776050


namespace combination_10_3_l776_776451

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l776_776451


namespace division_rounded_nearest_hundredth_l776_776440

theorem division_rounded_nearest_hundredth :
  Float.round (285 * 387 / (981^2) * 100) / 100 = 0.11 :=
by
  sorry

end division_rounded_nearest_hundredth_l776_776440


namespace pie_wholesale_price_l776_776819

theorem pie_wholesale_price (p1 p2: ℕ) (d: ℕ) (k: ℕ) (h1: d > 0) (h2: k > 1)
  (price_list: List ℕ)
  (p_plus_1: price_list = [p1 + d, k * p1, p2 + d, k * p2]) 
  (h3: price_list.perm [64, 64, 70, 72]) : 
  p1 = 60 ∧ p2 = 60 := 
sorry

end pie_wholesale_price_l776_776819


namespace origin_moves_correct_distance_l776_776404

-- Define the problem variables and assumptions
variables {B B' : Real × Real}
variables {r1 r2 : Real}
variables {d : Real}

-- Given conditions from the problem
def center_B : B = (3, -1) := sorry
def radius_B : r1 = 4 := sorry
def center_B' : B' = (3, 7) := sorry
def radius_B' : r2 = 6 := sorry
def dilation_factor : Real := 1.5
def translation_y : Real := 8

-- Calculate movement
def distance_origin_moves : Real :=
  let k := dilation_factor
  let d1 := Math.sqrt(3^2 + (-1)^2)
  let d1_p := k * d1
  let C := translation_y - d1_p
  C

-- Theorem statement representing the math proof problem
theorem origin_moves_correct_distance : 
  distance_origin_moves = 6.5 := 
sorry

end origin_moves_correct_distance_l776_776404


namespace lychees_remaining_l776_776678

theorem lychees_remaining 
  (initial_lychees : ℕ) 
  (sold_fraction taken_fraction : ℕ → ℕ) 
  (initial_lychees = 500) 
  (sold_fraction = λ x, x / 2) 
  (taken_fraction = λ x, x * 2 / 5) 
  : initial_lychees - sold_fraction initial_lychees - taken_fraction (initial_lychees - sold_fraction initial_lychees) = 100 := 
by 
  sorry

end lychees_remaining_l776_776678


namespace percentage_increase_l776_776307

theorem percentage_increase (L : ℕ) (h : L + 60 = 240) : 
  ((60:ℝ) / (L:ℝ)) * 100 = 33.33 := 
by
  sorry

end percentage_increase_l776_776307


namespace coefficient_of_x3y5_in_binomial_expansion_l776_776363

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  nat.choose n k

theorem coefficient_of_x3y5_in_binomial_expansion :
  let n := 8;
  let k := 3;
  let a := (2 : ℚ) / 3;
  let b := -(1 : ℚ) / 3;
  let x := 1;  -- Coefficients only involve rational numbers
  let y := 1;
  binomial_coefficient n k * (a ^ k) * (b ^ (n - k)) = -448 / 6561 :=
by
  sorry

end coefficient_of_x3y5_in_binomial_expansion_l776_776363


namespace trigonometric_expression_value_l776_776996

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α ^ 2 - Real.cos α ^ 2) / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 3 / 5 := 
sorry

end trigonometric_expression_value_l776_776996


namespace solve_for_x_l776_776545

theorem solve_for_x (x : ℝ) :
  (16⁻³ : ℝ) = 2^(100/x) / (2^(50/x) * (16^(25/x) : ℝ)) ↔ 
  x = 25 / 6 :=
by
  sorry

end solve_for_x_l776_776545


namespace ratio_circle_to_triangle_area_l776_776411

theorem ratio_circle_to_triangle_area 
  (h d : ℝ) 
  (h_pos : 0 < h) 
  (d_pos : 0 < d) 
  (R : ℝ) 
  (R_def : R = h / 2) :
  (π * R^2) / (1/2 * h * d) = (π * h) / (2 * d) :=
by sorry

end ratio_circle_to_triangle_area_l776_776411


namespace sqrt_mixed_number_l776_776517

theorem sqrt_mixed_number :
  (Real.sqrt (8 + 9/16)) = (Real.sqrt 137) / 4 :=
by
  sorry

end sqrt_mixed_number_l776_776517


namespace sum_f_eq_21_l776_776111

noncomputable def f (n : ℕ) : ℝ :=
if ∃ k : ℤ, n = 2^k then real.log 7 n else 0

theorem sum_f_eq_21 :
  ∑ n in finset.range (128), f n = 21 :=
by
  sorry

end sum_f_eq_21_l776_776111


namespace range_of_m_l776_776149

theorem range_of_m (f : ℝ → ℝ) (h_dom : ∀ x, -2 ≤ x ∧ x ≤ 2 → ∃ y, f(y) = x)
  (h_increasing : ∀ a b, -2 ≤ a ∧ a ≤ 2 ∧ -2 ≤ b ∧ b ≤ 2 ∧ a < b → f(a) < f(b))
  (h_ineq : ∀ m, f(1 - m) < f(m)) :
  ∀ m, (m ∈ (Set.Ioo (1 / 2 : ℝ) 2) ↔ h_ineq m) :=
by
  sorry

end range_of_m_l776_776149


namespace oxygen_atoms_in_compound_l776_776402

-- Define given conditions as parameters in the problem.
def number_of_oxygen_atoms (molecular_weight : ℕ) (weight_Al : ℕ) (weight_H : ℕ) (weight_O : ℕ) (atoms_Al : ℕ) (atoms_H : ℕ) (weight : ℕ) : ℕ := 
  (weight - (atoms_Al * weight_Al + atoms_H * weight_H)) / weight_O

-- Define the actual problem using the defined conditions.
theorem oxygen_atoms_in_compound
  (molecular_weight : ℕ := 78) 
  (weight_Al : ℕ := 27) 
  (weight_H : ℕ := 1) 
  (weight_O : ℕ := 16) 
  (atoms_Al : ℕ := 1) 
  (atoms_H : ℕ := 3) : 
  number_of_oxygen_atoms molecular_weight weight_Al weight_H weight_O atoms_Al atoms_H molecular_weight = 3 := 
sorry

end oxygen_atoms_in_compound_l776_776402


namespace true_propositions_in_reverse_neg_neg_reverse_l776_776725

theorem true_propositions_in_reverse_neg_neg_reverse (a b : ℕ) : 
  (¬ (a ≠ 0 → a * b ≠ 0) ∧ ∃ (a : ℕ), (a = 0 ∧ a * b ≠ 0) ∨ (a ≠ 0 ∧ a * b = 0) ∧ ¬ (¬ ∃ (a : ℕ), a ≠ 0 ∧ a * b ≠ 0 ∧ ¬ ∃ (a : ℕ), a = 0 ∧ a * b = 0)) ∧ (0 = 1) :=
by {
  sorry
}

end true_propositions_in_reverse_neg_neg_reverse_l776_776725


namespace eval_expression_l776_776940

theorem eval_expression : 3 - (-1) + 4 - 5 + (-6) - (-7) + 8 - 9 = 3 := 
  sorry

end eval_expression_l776_776940


namespace range_of_a_l776_776391

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 2 * x else -(x^2 + 2 * x)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, x ≥ 0 → f x = x^2 + 2 * x) →
  f (2 - a^2) > f a ↔ -2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l776_776391


namespace projection_of_a_on_b_is_4_l776_776975

noncomputable def projection_of_a_on_b_eq_4 
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (ha : a = (4,3)) (hb : b = (1,0)) : 
  real :=
  let dot_prod := a.1 * b.1 + a.2 * b.2 in
  let norm_a := (a.1^2 + a.2^2)^(1/2) in
  let norm_b := (b.1^2 + b.2^2)^(1/2) in
  let cos_theta := dot_prod / (norm_a * norm_b) in
  norm_a * cos_theta

theorem projection_of_a_on_b_is_4 : 
  projection_of_a_on_b_eq_4 (4, 3) (1, 0) (rfl) (rfl) = 4 := by
    sorry

end projection_of_a_on_b_is_4_l776_776975


namespace prime_roots_eq_l776_776375

theorem prime_roots_eq (n : ℕ) (hn : 0 < n) :
  (∃ (x1 x2 : ℕ), Prime x1 ∧ Prime x2 ∧ 2*x1^2 - 8*n*x1 + 10*x1 - n^2 + 35*n - 76 = 0 ∧ 
                    2*x2^2 - 8*n*x2 + 10*x2 - n^2 + 35*n - 76 = 0 ∧ x1 ≠ x2 ∧ x1 < x2) →
  n = 3 ∧ ∃ x1 x2 : ℕ, x1 = 2 ∧ x2 = 5 ∧ Prime x1 ∧ Prime x2 ∧
    2*x1^2 - 8*n*x1 + 10*x1 - n^2 + 35*n - 76 = 0 ∧
    2*x2^2 - 8*n*x2 + 10*x2 - n^2 + 35*n - 76 = 0 := 
by
  sorry

end prime_roots_eq_l776_776375


namespace curve_transformation_l776_776992

/-- Defining the matrix M such that point (x, y) transforms to (3x, 3y). -/
def M : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 0], ![0, 3]]

/-- Defining the inverse of matrix M. -/
def M_inv : Matrix (Fin 2) (Fin 2) ℝ := ![![1/3, 0], ![0, 1/3]]

/-- Given point (x, y) is transformed to (3x, 3y) under matrix transformation M,
    and curve C is transformed to C': y^2 = 4x, find the equation of curve C. -/
theorem curve_transformation :
  (∀ x y : ℝ, (M.mul_vec ![x, y]) = ![3*x, 3*y]) ∧
  (∀ x y : ℝ, ∃ x' y', (M.mul_vec ![x, y]) = ![x', y'] ∧ (y'^2 = 4 * x') → (y^2 = 4/3 * x)) :=
by
  sorry

end curve_transformation_l776_776992


namespace maximize_integral_l776_776930
open Real

noncomputable def integral_to_maximize (a b : ℝ) : ℝ :=
  ∫ x in a..b, exp (cos x) * (380 - x - x^2)

theorem maximize_integral :
  ∀ (a b : ℝ), a ≤ b → integral_to_maximize a b ≤ integral_to_maximize (-20) 19 :=
by
  intros a b h
  sorry

end maximize_integral_l776_776930


namespace area_on_larger_sphere_l776_776414

-- Define the radii of the spheres
def r_small : ℝ := 1
def r_in : ℝ := 4
def r_out : ℝ := 6

-- Given the area on the smaller sphere
def A_small_sphere_area : ℝ := 37

-- Statement: Find the area on the larger sphere
theorem area_on_larger_sphere :
  (A_small_sphere_area * (r_out / r_in) ^ 2 = 83.25) := by
  sorry

end area_on_larger_sphere_l776_776414


namespace values_r_s_l776_776108

theorem values_r_s (r s : ℤ) : 8^2 = 4^(3*r + s) → (r = 1 ∧ s = 0) :=
by
  intro h
  have h1 : 2^6 = 4^(3*r + s), by exact h
  have h2 : 2^6 = 2^(2*(3*r + s)), by rw [←pow_mul, pow_two, pow_two] at h1; exact h1
  have h3 : 6 = 6*r + 2*s, from (eq_pow_eq_pow_of_eq_base (nat.prime_two)) h2
  sorry

end values_r_s_l776_776108


namespace find_t_l776_776603

-- Define vectors a and b
def a : ℝ × ℝ := (-3, 4)
def b (t : ℝ) : ℝ × ℝ := (t, 3)

-- Define the dot product of vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define the magnitude of vector a
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

-- Define the projection condition
def projection_condition (t : ℝ) : Prop := 
  (dot_product a (b t)) / (magnitude a) = -3

-- The main statement to prove
theorem find_t (t : ℝ) (ht : projection_condition t) : t = 9 :=
sorry

end find_t_l776_776603


namespace shortest_distance_between_circles_l776_776371

noncomputable def circle1 : set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 16}

noncomputable def circle2 : set (ℝ × ℝ) :=
  {p | (p.1 + 5)^2 + (p.2 + 6)^2 = 1}

theorem shortest_distance_between_circles :
  let center1 := (3 : ℝ, 4 : ℝ)
  let center2 := (-5 : ℝ, -6 : ℝ)
  let radius1 := 4
  let radius2 := 1
  real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2) - (radius1 + radius2) = real.sqrt 164 - 5 :=
by
  sorry

end shortest_distance_between_circles_l776_776371


namespace clothing_store_earnings_l776_776031

-- Definitions for the given conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def cost_per_shirt : ℕ := 10
def cost_per_jeans : ℕ := 2 * cost_per_shirt

-- Theorem statement
theorem clothing_store_earnings : 
  (num_shirts * cost_per_shirt + num_jeans * cost_per_jeans = 400) := 
sorry

end clothing_store_earnings_l776_776031


namespace maximize_x_plus_y_l776_776959

noncomputable def max_x_plus_y (x y : ℝ) : ℝ :=
  x + y

theorem maximize_x_plus_y :
  ∀ (x y : ℝ), 
  (log (
    (x^2 + y^2) / 2) y ≥ 1 ∧ (x ≠ 0 ∨ y ≠ 0) ∧ (x^2 + y^2 ≠ 2)) →
  max_x_plus_y x y ≤ 1 + Real.sqrt 2 :=
begin
  sorry
end

end maximize_x_plus_y_l776_776959


namespace coefficient_x3y5_of_expansion_l776_776366

theorem coefficient_x3y5_of_expansion : 
  (binom 8 5) * ((2 / 3) ^ 3) * ((-1 / 3) ^ 5) = -448 / 6561 := by
  -- We skip the proof as instructed
  sorry

end coefficient_x3y5_of_expansion_l776_776366


namespace wholesale_price_is_60_l776_776871

variables (p d k : ℕ)
variables (P1 P2 Z1 Z2 : ℕ)

// Conditions from (a)
axiom H1 : P1 = p + d // Baker Plus price day 1
axiom H2 : P2 = p + d // Baker Plus price day 2
axiom H3 : Z1 = k * p // Star price day 1
axiom H4 : Z2 = k * p // Star price day 2
axiom H5 : k > 1 // Star store markup factor greater than 1
axiom H6 : P1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H7 : P2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H8 : Z1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H9 : Z2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))

theorem wholesale_price_is_60 : p = 60 := 
by {
    // Proof goes here, skipped with sorry
    sorry
}

end wholesale_price_is_60_l776_776871


namespace double_sum_evaluation_l776_776514

theorem double_sum_evaluation :
  (∑ m in (Finset.range ∞).filter (λ m, m ≥ 2),
     ∑ n in (Finset.range ∞).filter (λ n, n ≥ 2),
       (1 : ℝ) / (m^2 * n * (m + n + 2))) = (1/4 : ℝ) :=
by sorry

end double_sum_evaluation_l776_776514


namespace maximum_of_f_l776_776955

noncomputable def f (x : ℝ) : ℝ := sqrt (x + 31) + sqrt (17 - x) + sqrt x

theorem maximum_of_f :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 17 ∧ f x = 12 ∧ (∀ y : ℝ, 0 ≤ y ∧ y ≤ 17 → f y ≤ 12) :=
sorry

end maximum_of_f_l776_776955


namespace distance_CD_l776_776969

theorem distance_CD : 
  ∀ (net_south : ℕ) (net_west : ℕ),
  net_south = 50 - 20 →
  net_west = 80 - 30 →
  (∃ (dist : ℝ), dist = 10 * Real.sqrt 34 ∧ dist = Real.sqrt ((net_south)^2 + (net_west)^2)) :=
by
  intros net_south net_west hsouth hwest
  existsi Real.sqrt ( (30:ℝ)^2 + (50:ℝ)^2 )
  split
  . exact (calc
    Real.sqrt ( (30: ℝ)^2 + (50: ℝ)^2 )
        = Real.sqrt 3400 : by sorry
    ... = 10 * Real.sqrt 34 : by sorry
  )
  . rfl

end distance_CD_l776_776969


namespace determine_wholesale_prices_l776_776827

-- Definitions and Conditions
-- Wholesale prices for days 1 and 2 are defined as p1 and p2
def wholesale_price_day1 : ℝ := p1
def wholesale_price_day2 : ℝ := p2

-- The price increment at "Baker Plus" store is a constant d
def price_increment_baker_plus : ℝ := d
-- The factor at "Star" store is a constant k > 1
def price_factor_star : ℝ := k
-- Given prices for two days: 64, 64, 70, 72 rubles
def given_prices : set ℝ := {64, 64, 70, 72}

-- Prices at the stores
def baker_plus_day1_price := wholesale_price_day1 + price_increment_baker_plus
def star_day1_price := wholesale_price_day1 * price_factor_star

def baker_plus_day2_price := wholesale_price_day2 + price_increment_baker_plus
def star_day2_price := wholesale_price_day2 * price_factor_star

-- Theorem to determine the wholesale prices for each day (p1 and p2)
theorem determine_wholesale_prices
    (baker_plus_day1_price ∈ given_prices)
    (star_day1_price ∈ given_prices)
    (baker_plus_day2_price ∈ given_prices)
    (star_day2_price ∈ given_prices)
    (h1 : baker_plus_day1_price ≠ star_day1_price)
    (h2 : baker_plus_day2_price ≠ star_day2_price) :
  (∃ p1 p2 d k > 1, 
      (p1 + d ∈ given_prices) ∧ (p1 * k ∈ given_prices) ∧
      (p2 + d ∈ given_prices) ∧ (p2 * k ∈ given_prices) ∧
      (p1 ≠ p2)) :=
  sorry

end determine_wholesale_prices_l776_776827


namespace comb_10_3_eq_120_l776_776505

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l776_776505


namespace find_y_l776_776213

theorem find_y (x y : ℝ) (h1 : 2 * (x - y) = 12) (h2 : x + y = 14) : y = 4 := 
by
  sorry

end find_y_l776_776213


namespace sum_of_squares_l776_776695

theorem sum_of_squares (x y : ℝ) : 2 * x^2 + 2 * y^2 = (x + y)^2 + (x - y)^2 := 
by
  sorry

end sum_of_squares_l776_776695


namespace wholesale_price_is_60_l776_776868

variables (p d k : ℕ)
variables (P1 P2 Z1 Z2 : ℕ)

// Conditions from (a)
axiom H1 : P1 = p + d // Baker Plus price day 1
axiom H2 : P2 = p + d // Baker Plus price day 2
axiom H3 : Z1 = k * p // Star price day 1
axiom H4 : Z2 = k * p // Star price day 2
axiom H5 : k > 1 // Star store markup factor greater than 1
axiom H6 : P1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H7 : P2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H8 : Z1 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))
axiom H9 : Z2 ∈ (set.insert 64 (set.insert 64 (set.insert 70 (set.singleton 72))))

theorem wholesale_price_is_60 : p = 60 := 
by {
    // Proof goes here, skipped with sorry
    sorry
}

end wholesale_price_is_60_l776_776868


namespace hyperbola_eccentricity_l776_776167

theorem hyperbola_eccentricity (a : ℝ) (h : a > 0) (h_asymptote : Real.tan (Real.pi / 6) = 1 / a) :
  let c := Real.sqrt (a^2 + 1)
  let e := c / a
  e = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_l776_776167


namespace part1_part2_l776_776561

noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y - 20 = 0

noncomputable def line_eq (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y = 7*m + 4

theorem part1 : ∀ (m : ℝ), ∃ x y : ℝ, circle_eq x y ∧ line_eq m x y :=
by
  sorry

theorem part2 : ∃ (M : ℝ) (l_eqn : ℝ → ℝ → Prop),
  (∀ (x y : ℝ), l_eqn x y → circle_eq x y) ∧
  (M = 4*Real.sqrt 5) ∧
  (l_eqn = λ x y, 2*x - y - 5 = 0) :=
by
  sorry

end part1_part2_l776_776561


namespace binom_10_3_l776_776462

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l776_776462


namespace factorization_of_polynomial_l776_776799

theorem factorization_of_polynomial :
  (x : ℤ) → x^10 + x^5 + 1 = (x^2 + x + 1) * (x^8 - x^7 + x^5 - x^4 + x^3 - x + 1) :=
by
  sorry

end factorization_of_polynomial_l776_776799


namespace cone_lateral_surface_area_l776_776563

-- Define the conditions
def slant_height : ℝ := 4
def angle_between_slant_and_axis : ℝ := π / 6  -- 30 degrees in radians

-- Define the lateral surface area calculation
def lateral_surface_area (r l : ℝ) : ℝ := π * r * l

-- Define the radius computation based on the angle and slant height
def radius (l θ : ℝ) : ℝ := l * Real.sin(θ)

-- State the theorem with our given conditions
theorem cone_lateral_surface_area :
  lateral_surface_area (radius slant_height angle_between_slant_and_axis) slant_height = 8 * π :=
by
  sorry

end cone_lateral_surface_area_l776_776563


namespace minimum_abc_value_l776_776809

def has_one_real_root (P : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, P = λ x, a * x^2 + b * x + c ∧ (b^2 - 4 * a * c = 0)

def has_three_real_roots (P : ℝ → ℝ) : Prop :=
  ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ P(r1) = 0 ∧ P(r2) = 0 ∧ P(r3) = 0

theorem minimum_abc_value :
  ∃ a b c : ℝ, 
    (has_one_real_root (λ x, a * x^2 + b * x + c)) ∧ 
    (has_three_real_roots (λ x, a * (a * (a * x^2 + b * x + c) ^ 2 + b * x + c) ^ 2 + b * x + c)) ∧ 
    (a * b * c = -2) := 
by
  sorry

end minimum_abc_value_l776_776809


namespace range_of_a_l776_776171

variable (a : ℝ)

def p (a : ℝ) : Prop := 3/2 < a ∧ a < 5/2
def q (a : ℝ) : Prop := 2 ≤ a ∧ a ≤ 4

theorem range_of_a (h₁ : ¬(p a ∧ q a)) (h₂ : p a ∨ q a) : (3/2 < a ∧ a < 2) ∨ (5/2 ≤ a ∧ a ≤ 4) :=
sorry

end range_of_a_l776_776171


namespace range_of_k_l776_776511

noncomputable def k (x : ℝ) : ℝ := (3 * x + 5) / (x - 4)

theorem range_of_k : set.range k = {y : ℝ | y < 3 ∨ y > 3} := by
  sorry

end range_of_k_l776_776511


namespace selection_count_l776_776118

theorem selection_count (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 4) :
  let total_ways := (4.choose 3) + (4.choose 2 * 4.choose 1)
  in total_ways = 28 :=
by
  have := h_boys
  have := h_girls
  sorry

end selection_count_l776_776118


namespace problem1_problem2_problem3_l776_776063

-- First problem
theorem problem1 : 24 - |(-2)| + (-16) - 8 = -2 := by
  sorry

-- Second problem
theorem problem2 : (-2) * (3 / 2) / (-3 / 4) * 4 = 4 := by
  sorry

-- Third problem
theorem problem3 : -1^2016 - (1 - 0.5) / 3 * (2 - (-3)^2) = 1 / 6 := by
  sorry

end problem1_problem2_problem3_l776_776063


namespace minimum_faces_for_six_neighbors_l776_776690

-- Define the concepts of convex polyhedron in Lean
structure ConvexPolyhedron (V E F : ℕ) :=
(euler_characteristic : V - E + F = 2)
(face_adjacency : ∀ (f : ℕ), f < F → f ≤ 5)

-- Define the proposition to prove
theorem minimum_faces_for_six_neighbors (N : ℕ) :
  (∀ (P : ConvexPolyhedron V E F), F ≥ 12 → ∃ f : ℕ, f < F ∧ face_adjacency f ≥ 6) :=
sorry

end minimum_faces_for_six_neighbors_l776_776690


namespace pie_wholesale_price_l776_776823

theorem pie_wholesale_price (p1 p2: ℕ) (d: ℕ) (k: ℕ) (h1: d > 0) (h2: k > 1)
  (price_list: List ℕ)
  (p_plus_1: price_list = [p1 + d, k * p1, p2 + d, k * p2]) 
  (h3: price_list.perm [64, 64, 70, 72]) : 
  p1 = 60 ∧ p2 = 60 := 
sorry

end pie_wholesale_price_l776_776823


namespace factor_polynomial_l776_776942

theorem factor_polynomial (x : ℝ) : 
    54 * x^4 - 135 * x^8 = -27 * x^4 * (5 * x^4 - 2) := 
by 
  sorry

end factor_polynomial_l776_776942


namespace initial_meals_l776_776067

theorem initial_meals (X : ℕ) (H : X + 50 = 78 + 85) : X = 113 := by
  calc
    X = 163 - 50 : by rw [H]
    ... = 113 : by norm_num

end initial_meals_l776_776067


namespace distance_from_sphere_center_to_triangle_plane_l776_776416

open Real

-- Definitions derived from conditions
def radius : ℝ := 5
def a : ℝ := 13
def b : ℝ := 13
def c : ℝ := 10

-- Restating the problem
theorem distance_from_sphere_center_to_triangle_plane
  (r : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (radius : ℝ)
  (tangent : ∀ side, side = r + radius) :
  r = 5 ∧ a = 13 ∧ b = 13 ∧ c = 10 →
  ∀ x, x = r * sqrt(5) / 3 :=
by
  sorry

end distance_from_sphere_center_to_triangle_plane_l776_776416


namespace knights_statements_count_l776_776304

-- Definitions for the conditions
variables (r l : ℕ) -- r is the number of knights, l is the number of liars
variables (hr : r ≥ 2) (hl : l ≥ 2)
variables (total_liar_statements : 2 * r * l = 230)

-- Theorem statement: Prove that the number of times "You are a knight!" was said is 526
theorem knights_statements_count : 
  let total_islanders := r + l in
  let total_statements := total_islanders * (total_islanders - 1) in
  let knight_statements := total_statements - 230 in
  knight_statements = 526 :=
by
  sorry

end knights_statements_count_l776_776304


namespace combination_10_3_l776_776449

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l776_776449


namespace geometric_sequence_formula_and_sum_l776_776572

theorem geometric_sequence_formula_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a 1 = 2) 
  (h2 : ∀ n, a (n+1) = 2 * a n) 
  (h_arith : a 1 = 2 ∧ 2 * (a 3 + 1) = a 1 + a 4)
  (h_b : ∀ n, b n = Nat.log2 (a n)) :
  (∀ n, a n = 2 ^ n) ∧ (S n = (n * (n + 1)) / 2) := 
by 
  sorry

end geometric_sequence_formula_and_sum_l776_776572


namespace find_function_ex_l776_776950

theorem find_function_ex (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  (∀ x : ℝ, f x = x - a) :=
by
  intros h x
  sorry

end find_function_ex_l776_776950


namespace midpoint_BC_on_circumcircle_l776_776425

-- Translation of the given conditions into Lean definitions

variables {A B C D E F P Q R M : Type} [AffineSpace ℝ ℝ]

-- ABC is an acute-angled triangle with altitudes AD, BE, CF
variables {triangle_ABC : Triangle A B C}
variables {altitudes_AD_BE_CF : Altitude_tangents D E F A B C}

-- The line through D parallel to EF meets AC at Q and AB at R
variables {D_parallel_EF_meets_AC_at_Q_and_AB_at_R : Parallel_tangent D E F Q R}

-- The line EF meets BC at P
variables {EF_meets_BC_at_P : Line_intersection D E F P}

-- Defining the midpoint M of BC
def M_midpoint_of_BC (B C : ℝ) := (B + C) / 2

-- Proof statement: The midpoint M of BC lies on the circumcircle of triangle PQR
theorem midpoint_BC_on_circumcircle (B C : ℝ) (M : ℝ := M_midpoint_of_BC B C) :  
  CyclicQuadrilateral P Q R M :=
begin
  sorry
end

end midpoint_BC_on_circumcircle_l776_776425


namespace divisors_of_30_l776_776195

theorem divisors_of_30 : ∃ (n : ℕ), n = 16 ∧ (∀ d : ℤ, d ∣ 30 → (d ≤ 30 ∧ d ≥ -30)) :=
by
  sorry

end divisors_of_30_l776_776195


namespace area_of_triangle_ACF_l776_776044

    -- Definitions and conditions
    variable (A B C D E F : Type) -- Vertices of the hexagon
    variable [f : RegularHexagon A B C D E F] -- Assume A, B, C, D, E, F form a regular hexagon
    variable (s : ℝ) (h_s : s = 4) -- Side length of the hexagon

    -- The theorem to prove
    theorem area_of_triangle_ACF : area (triangle A C F) = 4 * real.sqrt 3 :=
    by
      sorry -- Proof is omitted here, but needed in practice
    
end area_of_triangle_ACF_l776_776044


namespace least_three_digit_multiple_of_13_l776_776768

-- Define what it means to be a multiple of 13
def is_multiple_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13 * k

-- Define the range of three-digit numbers
def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define our main theorem
theorem least_three_digit_multiple_of_13 : ∃ (n : ℕ), is_three_digit_number n ∧ is_multiple_of_13 n ∧
  ∀ (m : ℕ), is_three_digit_number m ∧ is_multiple_of_13 m → n ≤ m :=
begin
  -- We state the theorem without proof for simplicity
  sorry
end

end least_three_digit_multiple_of_13_l776_776768


namespace perpendicular_to_PQ_passes_through_orthocenter_ACD_l776_776266

open EuclideanGeometry

variables {A B C D P Q H : Point}

-- Definitions:
def convex_quadrilateral (A B C D : Point) : Prop := 
  convex A B C D

def is_orthocenter (H : Point) (A B C : Point) : Prop :=
  orthocenter H A B C

def is_parallel (l1 l2 : Line) : Prop := 
  parallel l1 l2

-- Theorem statement:
theorem perpendicular_to_PQ_passes_through_orthocenter_ACD
  (h_convex : convex_quadrilateral A B C D)
  (h_ortho_ABC : is_orthocenter H (triangle A B C))
  (h_parallel_AD : ∃ l, l = Line H ∧ is_parallel l (Line A D))
  (h_parallel_CD : ∃ m, m = Line H ∧ is_parallel m (Line C D))
  (h_P_on_AB : on_line P (Line A B))
  (h_Q_on_BC : on_line Q (Line B C))
  (h_H_on_perpendicular : ∀ l, l = Line H ∧ perpendicular l (Line P Q)) :
  ∃ H_ACD : Point, is_orthocenter H_ACD (triangle A C D) ∧
    on_line H_ACD (Line H ∧ perpendicular (Line H) (Line P Q)) :=
sorry

end perpendicular_to_PQ_passes_through_orthocenter_ACD_l776_776266


namespace coefficient_of_x_in_expansion_l776_776097

theorem coefficient_of_x_in_expansion :
  -- Given the expression to expand
  let expr := (1 - x) ^ 4 * (1 - sqrt x) ^ 3 in

  -- Assume the term to find the coefficient
  ∃ c : ℚ,
    expr = ∑ i, c * x ^ i ∧ 
    (∀ j : ℕ, j ≠ 1 → coefficient (∑ i, c * x ^ i) x j = 0) ∧
    coefficient (∑ i, c * x ^ i) x 1 = -1 :=
sorry

end coefficient_of_x_in_expansion_l776_776097


namespace rectangle_division_unequal_areas_l776_776434

/-- 
  Given a rectangle ABCD with points M on AB and N on CD, and lines MC, BN, AN, and DM; 
  show that it divides the rectangle into seven regions with unequal areas.
-/
theorem rectangle_division_unequal_areas
  (A B C D M N : Point)
  (h_M_AB : M ∈ AB) (h_N_CD : N ∈ CD)
  (h_rectangle : Rectangle A B C D) :
  ∃ (regions : Fin 7 → Set Point), 
    (∀ i j, i ≠ j → ¬(regions i = regions j)) ∧
    (∑ i, area (regions i)) = area (Rectangle A B C D) :=
sorry

end rectangle_division_unequal_areas_l776_776434


namespace recurrence_relation_bounds_f2008_l776_776985

noncomputable def f : ℕ → ℝ
| 1     := 1
| 2     := 2
| (n+2) := f(n) * (f(n+1)^2 + 1) / (f(n)^2 + 1)

theorem recurrence_relation (n : ℕ) (hn : n ≥ 1) : 
  f(n + 1) = f(n) + 1 / f(n) :=
sorry

theorem bounds_f2008 : 
  63 < f 2008 ∧ f 2008 < 78 :=
sorry

end recurrence_relation_bounds_f2008_l776_776985


namespace sum_x_coords_Q3_l776_776814

theorem sum_x_coords_Q3 (x_coords_q1 : Fin 45 → ℝ) 
  (h_sum : (∑ i, x_coords_q1 i) = 135) : 
  ∑ i, ((0 : Fin 45 → ℝ) i) = 135 := 
by 
  -- Define the x-coordinates of vertices of Q2
  let x_coords_q2 : Fin 45 → ℝ := λ i, (x_coords_q1 i + x_coords_q1 ((i + 1) % 45)) / 2
  -- Assume the sum of the x-coordinates of Q2 is equal to that of Q1
  have h_q2_sum : (∑ i, x_coords_q2 i) = 135 := 
  by 
    calc (∑ i, x_coords_q2 i)
      = (∑ i, (x_coords_q1 i + x_coords_q1 ((i + 1) % 45)) / 2) : by sorry
    ... = (2 * (∑ i, x_coords_q1 i)) / 2 : by sorry
    ... = 135 : by rw h_sum
  -- Let the x-coordinates of Q3
  let x_coords_q3 : Fin 45 → ℝ := λ i, (x_coords_q2 i + x_coords_q2 ((i + 1) % 45)) / 2
  -- Sum of the x-coordinates of Q3 is the same as Q2
  calc (∑ i, x_coords_q3 i)
    = (∑ i, (x_coords_q2 i + x_coords_q2 ((i + 1) % 45)) / 2) : by sorry
    ... = (2 * (∑ i, x_coords_q2 i)) / 2 : by sorry
    ... = 135 : by rw h_q2_sum

end sum_x_coords_Q3_l776_776814


namespace average_of_11_numbers_l776_776714

theorem average_of_11_numbers (a b c d e f g h i j k : ℝ)
  (h_first_6_avg : (a + b + c + d + e + f) / 6 = 98)
  (h_last_6_avg : (f + g + h + i + j + k) / 6 = 65)
  (h_6th_number : f = 318) :
  ((a + b + c + d + e + f + g + h + i + j + k) / 11) = 60 :=
by
  sorry

end average_of_11_numbers_l776_776714


namespace sum_of_coefficients_l776_776610

theorem sum_of_coefficients (b_6 b_5 b_4 b_3 b_2 b_1 b_0 : ℤ) :
  (5 * x - 2) ^ 6 = b_6 * x ^ 6 + b_5 * x ^ 5 + b_4 * x ^ 4 + b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0 →
  b_6 + b_5 + b_4 + b_3 + b_2 + b_1 + b_0 = 729 :=
by
  sorry

end sum_of_coefficients_l776_776610


namespace ice_cream_melting_l776_776420

theorem ice_cream_melting :
  ∀ (r1 r2 : ℝ) (h : ℝ),
    r1 = 3 ∧ r2 = 10 →
    4 / 3 * π * r1^3 = π * r2^2 * h →
    h = 9 / 25 :=
by intros r1 r2 h hcond voldist
   sorry

end ice_cream_melting_l776_776420


namespace eccentricity_of_ellipse_l776_776099

theorem eccentricity_of_ellipse :
  let a : ℝ := 4
  let b : ℝ := 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let e : ℝ := c / a
  e = Real.sqrt 7 / 4 :=
by
  sorry

end eccentricity_of_ellipse_l776_776099


namespace divisors_of_30_count_l776_776188

theorem divisors_of_30_count : 
  ∃ S : Set ℤ, (∀ d ∈ S, 30 % d = 0) ∧ S.card = 16 :=
by
  sorry

end divisors_of_30_count_l776_776188


namespace minimum_number_of_cubes_l776_776889

-- Define the dimensions of the box
def length : ℝ := 8
def width : ℝ := 15
def height : ℝ := 5

-- Define the volume of one cube
def volume_of_one_cube : ℝ := 10

-- Define the volume of the box
def volume_of_box := length * width * height

-- The proof statement that the minimum number of cubes required to build the box is 60
theorem minimum_number_of_cubes : volume_of_box / volume_of_one_cube = 60 := by
  sorry

end minimum_number_of_cubes_l776_776889


namespace infinite_nonzero_values_l776_776078

noncomputable def sequence_a (n : ℕ) : ℕ :=
  (binom (2 * n) n) % 3

theorem infinite_nonzero_values : 
  ∃ᶠ n in Filter.atTop, sequence_a n ≠ 0 :=
sorry

end infinite_nonzero_values_l776_776078


namespace collinear_vectors_y_value_l776_776602

theorem collinear_vectors_y_value (y : ℝ) :
  let a := (-3,1) in
  let b := (6,y) in
  (a.1 * b.2 - a.2 * b.1 = 0) → y = -2 :=
by
  intro a b h
  sorry

end collinear_vectors_y_value_l776_776602


namespace peanut_butter_revenue_l776_776668

theorem peanut_butter_revenue :
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  plantation_length * plantation_width * peanuts_per_sqft * butter_from_peanuts_ratio / 1000 * butter_price_per_kg = 31250 := 
by
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  sorry

end peanut_butter_revenue_l776_776668


namespace remainder_when_divided_by_l776_776370

/- This definition states the polynomial P(x) -/
def P (x : ℝ) := 5 * x^6 - 3 * x^4 + 6 * x^3 - 8 * x + 10

/- This definition states the divisor polynomial Q(x) -/
def Q (x : ℝ) := 3 * x - 9

/- This theorem states that the remainder of P(x) divided by Q(x) is 3550 -/
theorem remainder_when_divided_by :
  ∀ x : ℝ, (Q(x) = 0) → P(x) = 3550 := by
  intro x hx
  sorry

end remainder_when_divided_by_l776_776370


namespace batsman_average_increase_l776_776027

/--
  Given that:
  - he batsman in his 12th innings makes a score of 65.
  - his average after the 12th innings is 32.
  - he had never been 'not out'.

  Prove that the increase in his average is 3 runs.
-/
theorem batsman_average_increase :
  ∀ (A : ℕ), 
  ((11 * A + 65 = 12 * 32) → 32 - A = 3) :=
by
  intro A
  assume h
  sorry

end batsman_average_increase_l776_776027


namespace minimum_value_of_a_plus_4b_l776_776989

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hgeo : Real.sqrt (a * b) = 2)

theorem minimum_value_of_a_plus_4b : a + 4 * b = 8 := by
  sorry

end minimum_value_of_a_plus_4b_l776_776989


namespace clothing_store_earnings_l776_776034

-- Defining the conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def shirt_cost : ℕ := 10
def jeans_cost : ℕ := 2 * shirt_cost

-- Statement of the problem
theorem clothing_store_earnings :
  num_shirts * shirt_cost + num_jeans * jeans_cost = 400 :=
by
  sorry

end clothing_store_earnings_l776_776034


namespace least_three_digit_multiple_of_13_l776_776779

theorem least_three_digit_multiple_of_13 : ∃ n : ℕ, n ≥ 100 ∧ n % 13 = 0 ∧ ∀ m : ℕ, m ≥ 100 → m % 13 = 0 → n ≤ m :=
begin
  use 104,
  split,
  { exact nat.le_of_eq rfl },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero rfl) },
  { intros m hm hmod,
    have h8 : 8 * 13 = 104 := rfl,
    rw ←h8,
    exact nat.le_mul_of_pos_left (by norm_num) },
end

end least_three_digit_multiple_of_13_l776_776779


namespace percentage_of_second_reduction_l776_776338

theorem percentage_of_second_reduction 
  (P : ℝ) 
  (hP : P > 0) 
  (x : ℝ) 
  (h1 : 0 < x < 1)
  (combined_effective_reduction : (1 - x) * 0.75 = 0.45) :
  x = 0.4 :=
by
  sorry

end percentage_of_second_reduction_l776_776338


namespace irrational_root_exists_l776_776658

noncomputable def P (a b c d : ℤ) : Polynomial ℝ :=
  Polynomial.C a * Polynomial.X ^ 3 +
  Polynomial.C b * Polynomial.X ^ 2 +
  Polynomial.C c * Polynomial.X +
  Polynomial.C d 

theorem irrational_root_exists (a b c d : ℤ) 
  (ha : a % 2 = 1) (hd : d % 2 = 1) 
  (hb : b % 2 = 0) (hc : c % 2 = 0) 
  (hreal : ∀ x, Polynomial.IsRoot (P a b c d) x → x.real_or_complex = real) : 
  ∃ x, Polynomial.IsRoot (P a b c d) x ∧ ¬ (∃ (q : ℚ), x = q) := 
sorry

end irrational_root_exists_l776_776658


namespace least_three_digit_multiple_of_13_l776_776771

-- Define what it means to be a multiple of 13
def is_multiple_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13 * k

-- Define the range of three-digit numbers
def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define our main theorem
theorem least_three_digit_multiple_of_13 : ∃ (n : ℕ), is_three_digit_number n ∧ is_multiple_of_13 n ∧
  ∀ (m : ℕ), is_three_digit_number m ∧ is_multiple_of_13 m → n ≤ m :=
begin
  -- We state the theorem without proof for simplicity
  sorry
end

end least_three_digit_multiple_of_13_l776_776771


namespace distribution_count_formula_l776_776512

theorem distribution_count_formula (n m : ℕ) :
  l(n, m) = ∑ k in Finset.range (n+1), (-1)^k * (Nat.choose n k) * (n - k - 1)^(n - k) * (n - k)^(m + k) :=
sorry

end distribution_count_formula_l776_776512


namespace ant_prob_reach_D_after_6_minutes_l776_776909

noncomputable def probability_ant_at_D_after_6_minutes (start_pos end_pos : ℤ × ℤ) (total_moves : ℕ) : ℚ :=
  if start_pos = (-1, -1) ∧ end_pos = (1, 1) ∧ total_moves = 6 then 1 / 8 else 0

theorem ant_prob_reach_D_after_6_minutes :
  probability_ant_at_D_after_6_minutes (-1, -1) (1, 1) 6 = 1 / 8 :=
by sorry

end ant_prob_reach_D_after_6_minutes_l776_776909


namespace sum_of_undefined_values_l776_776082

theorem sum_of_undefined_values :
    (∑ x in ({x | x^2 - 7 * x + 10 = 0}), x) = 7 :=
by sorry

end sum_of_undefined_values_l776_776082


namespace binomial_coeff_equal_l776_776812

theorem binomial_coeff_equal (n : ℕ) (h₁ : 6 ≤ n) (h₂ : (n.choose 5) * 3^5 = (n.choose 6) * 3^6) :
  n = 7 := sorry

end binomial_coeff_equal_l776_776812


namespace tetrahedron_min_f_l776_776325

/-- Given a tetrahedron ABCD with specified edge lengths, 
    the minimum possible value of f(X) defined as the sum of distances of X 
    to the vertices A, B, C, and D is 120, and thus m + n = 121 
    where the minimal value is expressed as m*sqrt(n). -/
theorem tetrahedron_min_f {A B C D X : Point}
  (hAD : distance A D = 30)
  (hBC : distance B C = 30)
  (hAC : distance A C = 50)
  (hBD : distance B D = 50)
  (hAB : distance A B = 60)
  (hCD : distance C D = 60)
  (f : Point → ℝ) (hf : ∀ (X : Point), f X = distance A X + distance B X + distance C X + distance D X) :
  ∃ (m n : ℕ), f X ≥ m * Real.sqrt n ∧ m = 120 ∧ n = 1 ∧ m + n = 121 :=
by sorry

end tetrahedron_min_f_l776_776325


namespace total_bottles_capped_in_10_minutes_l776_776346

-- Define the capacities per minute for the three machines
def machine_a_capacity : ℕ := 12
def machine_b_capacity : ℕ := machine_a_capacity - 2
def machine_c_capacity : ℕ := machine_b_capacity + 5

-- Define the total capping capacity for 10 minutes
def total_capacity_in_10_minutes (a b c : ℕ) : ℕ := a * 10 + b * 10 + c * 10

-- The theorem we aim to prove
theorem total_bottles_capped_in_10_minutes :
  total_capacity_in_10_minutes machine_a_capacity machine_b_capacity machine_c_capacity = 370 :=
by
  -- Directly use the capacities defined above
  sorry

end total_bottles_capped_in_10_minutes_l776_776346


namespace least_positive_three_digit_multiple_of_13_is_104_l776_776786

theorem least_positive_three_digit_multiple_of_13_is_104 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ n = 104 :=
by
  existsi 104
  split
  · show 100 ≤ 104
    exact le_refl 104
  split
  · show 104 < 1000
    exact dec_trivial
  split
  · show 104 % 13 = 0
    exact dec_trivial
  · show 104 = 104
    exact rfl

end least_positive_three_digit_multiple_of_13_is_104_l776_776786


namespace clothing_store_earnings_l776_776032

-- Definitions for the given conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def cost_per_shirt : ℕ := 10
def cost_per_jeans : ℕ := 2 * cost_per_shirt

-- Theorem statement
theorem clothing_store_earnings : 
  (num_shirts * cost_per_shirt + num_jeans * cost_per_jeans = 400) := 
sorry

end clothing_store_earnings_l776_776032


namespace positive_difference_between_mean_and_median_l776_776344

noncomputable def mean (xs : List ℚ) : ℚ :=
  xs.sum / xs.length

def median (xs : List ℚ) : ℚ :=
  let sorted_xs := xs.qsort (· < ·)
  if h : sorted_xs.length % 2 = 0 then
    (sorted_xs.get ⟨sorted_xs.length / 2 - 1, by simp [Nat.div_lt_self]; exact h⟩ +
     sorted_xs.get ⟨sorted_xs.length / 2, by simp [Nat.div_lt_self]; exact h⟩) / 2
  else
    sorted_xs.get ⟨sorted_xs.length / 2, by simp [Nat.div_lt_self]⟩

def positive_difference (a b : ℚ) : ℚ :=
  abs (a - b)

def vertical_drops : List ℚ := [180, 120, 150, 310, 210, 190]
def problem_mean : ℚ := mean vertical_drops
def problem_median : ℚ := median vertical_drops
def problem_difference : ℚ := positive_difference problem_mean problem_median

theorem positive_difference_between_mean_and_median :
  problem_difference = 8 + 1 / 3 :=
by
  sorry

end positive_difference_between_mean_and_median_l776_776344


namespace problem_statement_l776_776136

-- Definitions of propositions p and q
def p : Prop := ∀ x : ℝ, 0 < x → Real.log2 x < Real.log 4 x
def q : Prop := ∃ x : ℝ, Real.tan x = 1 - x

-- The statement to prove
theorem problem_statement : (¬p ∧ q) :=
by {
  -- Assuming the conditions of propositions p and q
  have p_false : ¬p,
  {
    sorry, -- Proof showing ¬p, as explained in the solution
  },
  have q_true : q,
  {
    sorry, -- Proof showing q is true, as explained in the solution
  },
  exact ⟨p_false, q_true⟩,
}

end problem_statement_l776_776136


namespace motorcyclist_speed_l776_776406

theorem motorcyclist_speed
  {v m t : ℝ} (hv : v > 0) (hm : m > 0) (ht : t > 0) :
  ∃ x : ℝ, x = (3 * m + real.sqrt (9 * m ^ 2 + 2500 * t ^ 2 * v ^ 2)) / (50 * t) :=
begin
  use (3 * m + real.sqrt (9 * m ^ 2 + 2500 * t ^ 2 * v ^ 2)) / (50 * t),
  exact rfl,
end

end motorcyclist_speed_l776_776406


namespace fraction_of_seniors_study_japanese_l776_776435

variable (J S : ℝ)
variable (fraction_seniors fraction_juniors : ℝ)
variable (total_fraction_study_japanese : ℝ)

theorem fraction_of_seniors_study_japanese 
  (h1 : S = 2 * J)
  (h2 : fraction_juniors = 3 / 4)
  (h3 : total_fraction_study_japanese = 1 / 3) :
  fraction_seniors = 1 / 8 :=
by
  -- Here goes the proof.
  sorry

end fraction_of_seniors_study_japanese_l776_776435


namespace number_of_unit_squares_in_50th_ring_l776_776070

def nth_ring_unit_squares (n : ℕ) : ℕ :=
  8 * n

-- Statement to prove
theorem number_of_unit_squares_in_50th_ring : nth_ring_unit_squares 50 = 400 :=
by
  -- Proof steps (skip with sorry)
  sorry

end number_of_unit_squares_in_50th_ring_l776_776070


namespace wholesale_prices_l776_776839

-- Definitions for the problem conditions
variable (p1 p2 d k : ℝ)
variable (h_d : d > 0)
variable (h_k : k > 1)
variable (prices : Finset ℝ)
variable (h_prices : prices = {64, 64, 70, 72})

-- The theorem statement to prove
theorem wholesale_prices :
  ∃ p1 p2, (p1 + d ∈ prices ∧ k * p1 ∈ prices) ∧ 
           (p2 + d ∈ prices ∧ k * p2 ∈ prices) ∧ 
           p1 ≠ p2
:= sorry

end wholesale_prices_l776_776839


namespace shadow_area_correct_l776_776891

noncomputable def shadow_area (R : ℝ) : ℝ := 3 * Real.pi * R^2

theorem shadow_area_correct (R r d R' : ℝ)
  (h1 : r = (Real.sqrt 3) * R / 2)
  (h2 : d = (3 * R) / 2)
  (h3 : R' = ((3 * R * r) / d)) :
  shadow_area R = Real.pi * R' ^ 2 :=
by
  sorry

end shadow_area_correct_l776_776891


namespace exist_positive_integers_for_perfect_squares_l776_776085

theorem exist_positive_integers_for_perfect_squares :
  ∃ (x y : ℕ), (0 < x ∧ 0 < y) ∧ (∃ a b c : ℕ, x + y = a^2 ∧ x^2 + y^2 = b^2 ∧ x^3 + y^3 = c^2) :=
by
  sorry

end exist_positive_integers_for_perfect_squares_l776_776085


namespace verify_value_of_2a10_minus_a12_l776_776238

-- Define the arithmetic sequence and the sum condition
variable {a : ℕ → ℝ}  -- arithmetic sequence
variable {a1 : ℝ}     -- the first term of the sequence
variable {d : ℝ}      -- the common difference of the sequence

-- Assume that the sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a n = a1 + n * d

-- Assume the sum condition
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- The goal is to prove that 2 * a 10 - a 12 = 24
theorem verify_value_of_2a10_minus_a12 (h_arith : arithmetic_sequence a a1 d) (h_sum : sum_condition a) :
  2 * a 10 - a 12 = 24 :=
  sorry

end verify_value_of_2a10_minus_a12_l776_776238


namespace pencils_average_l776_776433

section

variable (Anna Harry Lucy David : ℕ)

noncomputable def remaining_pencils (Anna Harry Lucy David : ℕ) : ℕ :=
  Anna + Harry + Lucy + David

def average_pencils_remaining (Anna Harry Lucy David : ℕ) : ℕ :=
  remaining_pencils 50 (2 * 50 - 19) (3 * 50 - 13) (4 * 50 - 21) / 4

theorem pencils_average :
  average_pencils_remaining 50 (2 * 50 - 19) (3 * 50 - 13) (4 * 50 - 21) = 111.75 :=
by
  sorry

end

end pencils_average_l776_776433


namespace wholesale_prices_l776_776836

-- Definitions for the problem conditions
variable (p1 p2 d k : ℝ)
variable (h_d : d > 0)
variable (h_k : k > 1)
variable (prices : Finset ℝ)
variable (h_prices : prices = {64, 64, 70, 72})

-- The theorem statement to prove
theorem wholesale_prices :
  ∃ p1 p2, (p1 + d ∈ prices ∧ k * p1 ∈ prices) ∧ 
           (p2 + d ∈ prices ∧ k * p2 ∈ prices) ∧ 
           p1 ≠ p2
:= sorry

end wholesale_prices_l776_776836


namespace gcd_105_90_l776_776101

theorem gcd_105_90 : Nat.gcd 105 90 = 15 :=
by
  sorry

end gcd_105_90_l776_776101


namespace coefficient_x3y5_of_expansion_l776_776365

theorem coefficient_x3y5_of_expansion : 
  (binom 8 5) * ((2 / 3) ^ 3) * ((-1 / 3) ^ 5) = -448 / 6561 := by
  -- We skip the proof as instructed
  sorry

end coefficient_x3y5_of_expansion_l776_776365


namespace parabola_intersection_l776_776727

theorem parabola_intersection :
  let y := fun x : ℝ => x^2 - 2*x - 3 in
  (∀ x : ℝ, y x = 0 ↔ x = 3 ∨ x = -1) ∧ y (-1) = 0 :=
by
  sorry

end parabola_intersection_l776_776727


namespace triangle_XOY_hypotenuse_l776_776270

theorem triangle_XOY_hypotenuse (a b : ℝ) (h1 : (a/2)^2 + b^2 = 22^2) (h2 : a^2 + (b/2)^2 = 19^2) :
  Real.sqrt (a^2 + b^2) = 26 :=
sorry

end triangle_XOY_hypotenuse_l776_776270


namespace divisors_of_30_count_l776_776190

theorem divisors_of_30_count : 
  ∃ S : Set ℤ, (∀ d ∈ S, 30 % d = 0) ∧ S.card = 16 :=
by
  sorry

end divisors_of_30_count_l776_776190


namespace away_to_home_loss_ratio_l776_776026

def total_games := 45
def away_games := 15
def home_wins := 6
def away_wins := 7
def home_wins_extra_innings := 3
def home_games := total_games - away_games
def home_losses := home_games - home_wins
def away_losses := away_games - away_wins

theorem away_to_home_loss_ratio :
  away_losses * 3 = home_losses :=
by
  unfold total_games away_games home_wins away_wins home_wins_extra_innings home_games home_losses away_losses
  have h1 : home_games = 30 := rfl
  have h2 : home_losses = 24 := rfl
  have h3 : away_losses = 8 := rfl
  calc
    away_losses * 3 = 8 * 3 : by rw h3
        ... = 24 : by norm_num
        ... = home_losses : by rw h2

end away_to_home_loss_ratio_l776_776026


namespace matt_peanut_revenue_l776_776672

theorem matt_peanut_revenue
    (plantation_length : ℕ)
    (plantation_width : ℕ)
    (peanut_production : ℕ)
    (peanut_to_peanut_butter_rate_peanuts : ℕ)
    (peanut_to_peanut_butter_rate_butter : ℕ)
    (peanut_butter_price_per_kg : ℕ)
    (expected_revenue : ℕ) :
    plantation_length = 500 →
    plantation_width = 500 →
    peanut_production = 50 →
    peanut_to_peanut_butter_rate_peanuts = 20 →
    peanut_to_peanut_butter_rate_butter = 5 →
    peanut_butter_price_per_kg = 10 →
    expected_revenue = 31250 :=
by
  sorry

end matt_peanut_revenue_l776_776672


namespace part_a_part_b_part_c_l776_776022

-- Definitions of properties for Lean statements based on problem conditions
def P (x : ℝ) : Prop := x ∈ ℚ ∧ (∀ ε > 0, f (x - ε) > f x ∧ f (x + ε) > f x)

def R (f : ℚ → ℝ) (I : set ℝ) : Prop := ∀ x ∈ I ∩ ℚ, ∃ y ∈ I ∩ ℚ, f y > f x

def R' (f : ℝ → ℝ) (I : set ℝ) : Prop := ∀ x ∈ I ∩ ℚ, f x > 0

-- Part (a) function to derive Q given P
theorem part_a (f : ℝ → ℝ) : (∀ x ∈ ℚ, P x) → (∀ x ∈ ℚ, f x = 1) :=
sorry

-- Part (b) function construction
theorem part_b {I : set ℝ} (f : ℚ → ℝ) : ((∀ x ∈ ℚ, P x) ∧ R f I) → true :=
sorry

-- Part (c) prove contradiction with R' implies not property P
theorem part_c (f : ℝ → ℝ) : (∀ I : set ℝ, R' f I) → ¬ ∀ x ∈ ℚ, P x :=
sorry

end part_a_part_b_part_c_l776_776022


namespace inequality_proof_l776_776568

theorem inequality_proof 
  (x y z : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (h_sum : x + y + z = 1) :
  (1 - 2*x) / real.sqrt (x * (1 - x)) + 
  (1 - 2*y) / real.sqrt (y * (1 - y)) + 
  (1 - 2*z) / real.sqrt (z * (1 - z)) ≥
  real.sqrt (x / (1 - x)) + real.sqrt (y / (1 - y)) + real.sqrt (z / (1 - z)) := 
by
  sorry

end inequality_proof_l776_776568


namespace smallest_three_digit_multiple_of_13_l776_776787

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 13 = 0 → n ≤ m :=
⟨104, by sorry⟩

end smallest_three_digit_multiple_of_13_l776_776787


namespace count_distinct_four_digit_numbers_with_product_18_l776_776081

theorem count_distinct_four_digit_numbers_with_product_18 : 
  (∃ digits : vector ℕ 4, (∀ d ∈ digits, d ≥ 1 ∧ d ≤ 9) ∧ digits.prod = 18) → 
  (count (λ digits : vector ℕ 4, (∀ d ∈ digits, d ≥ 1 ∧ d ≤ 9) ∧ digits.prod = 18) = 36) :=
sorry

end count_distinct_four_digit_numbers_with_product_18_l776_776081


namespace Carolina_Winning_Probability_Beto_Winning_Probability_Ana_Winning_Probability_l776_776057

section
  -- Define the types of participants and the colors
  inductive Participant
  | Ana | Beto | Carolina

  inductive Color
  | blue | green

  -- Define the strategies for each participant
  inductive Strategy
  | guessBlue | guessGreen | pass

  -- Probability calculations for each strategy
  def carolinaStrategyProbability : ℚ := 1 / 8
  def betoStrategyProbability : ℚ := 1 / 2
  def anaStrategyProbability : ℚ := 3 / 4

  -- Statements to prove the probabilities
  theorem Carolina_Winning_Probability :
    carolinaStrategyProbability = 1 / 8 :=
  sorry

  theorem Beto_Winning_Probability :
    betoStrategyProbability = 1 / 2 :=
  sorry

  theorem Ana_Winning_Probability :
    anaStrategyProbability = 3 / 4 :=
  sorry
end

end Carolina_Winning_Probability_Beto_Winning_Probability_Ana_Winning_Probability_l776_776057


namespace binomial_coefficient_10_3_l776_776488

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l776_776488


namespace find_a_l776_776392

noncomputable def curve (a : ℝ) := ∀ x y : ℝ, (x - a) ^ 2 + (y - a) ^ 2 = 2 * a ^ 2
noncomputable def line := ∀ m t : ℝ, 
  let x := m + (1 / 2) * t
  let y := (sqrt 3 / 2) * t 
  x = -1 + (1 / 2) * t ∧ y = (sqrt 3 / 2) * t

theorem find_a : 
  let x := -1 + (1 / 2) * t
  let y := (sqrt 3 / 2) * t 
  ((|(-1, 0), (x, y)| + |(-1, 0), (-1 + (1 / 2) * (5 - t), (sqrt 3 / 2) * (5 - t))| = 5) →
  ∀ a : ℝ, a > 0 → (a = 2 * sqrt 3 - 2) :=
sorry

end find_a_l776_776392


namespace scooter_gain_percent_l776_776696

theorem scooter_gain_percent (purchase_price repair_costs selling_price : ℝ) 
  (h_purchase : purchase_price = 900) 
  (h_repairs : repair_costs = 300) 
  (h_selling : selling_price = 1500) : 
  ((selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100) = 25 := 
by 
  rw [h_purchase, h_repairs, h_selling]
  norm_num
  sorry

end scooter_gain_percent_l776_776696


namespace instantaneous_velocity_at_2_l776_776407

def displacement (t : ℝ) : ℝ := (1/3) * t^3 - (3/2) * t^2 + 2 * t

theorem instantaneous_velocity_at_2 : 
  deriv displacement 2 = 0 :=
by
  sorry

end instantaneous_velocity_at_2_l776_776407


namespace rad_to_deg_conversion_l776_776508

-- Define the conversion factor from radians to degrees
def rad_to_deg := 180 / Real.pi

-- Given condition and goal
theorem rad_to_deg_conversion : (8 * Real.pi / 5) * rad_to_deg = 288 :=
by
  -- Proof is omitted
  sorry

end rad_to_deg_conversion_l776_776508


namespace part1_part2_part2_min_attained_l776_776555

noncomputable def f : ℝ → ℝ := λ α, Real.cos α
def g (x : ℝ) : ℝ := f (2 * x) - 2 * f x

theorem part1 (α : ℝ) (hα : 0 ≤ α ∧ α ≤ Real.pi) (h : f α = 1 / 3) : 
f (α - Real.pi / 3) = (1 / 6) + Real.sqrt 6 / 3 := by
  sorry

theorem part2 : ∀ x : ℝ, g x ≥ - (3 / 2) := by
  sorry

theorem part2_min_attained : ∃ x : ℝ, g x = - (3 / 2) := by
  sorry

end part1_part2_part2_min_attained_l776_776555


namespace max_value_function_l776_776558

theorem max_value_function (x y z : ℝ) (h : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : xy + yz + zx = 1) :
    sqrt (xy + 5) + sqrt (yz + 5) + sqrt (zx + 5) ≤ 4 * sqrt 3 :=
sorry

end max_value_function_l776_776558


namespace calculator_to_protractors_l776_776436

def calculator_to_rulers (c: ℕ) : ℕ := 100 * c
def rulers_to_compasses (r: ℕ) : ℕ := (r * 30) / 10
def compasses_to_protractors (p: ℕ) : ℕ := (p * 50) / 25

theorem calculator_to_protractors (c: ℕ) : compasses_to_protractors (rulers_to_compasses (calculator_to_rulers c)) = 600 * c :=
by
  sorry

end calculator_to_protractors_l776_776436


namespace road_width_l776_776400

noncomputable def width_of_road (r R : ℝ) : ℝ :=
  R - r

theorem road_width (r R : ℝ) (h : 2 * Real.pi * R - 2 * Real.pi * r = 66) : width_of_road r R = 66 / (2 * Real.pi) :=
by
  unfold width_of_road
  rw [mul_sub, sub_eq_add_neg, mul_neg, sub_eq_add_neg, mul_neg] at h
  ring at h
  rwa [eq_div_iff]
  apply mul_ne_zero
  exact two_ne_zero
  exact Real.pi_ne_zero

end road_width_l776_776400


namespace fish_in_pond_l776_776607

noncomputable def number_of_fish (marked_first: ℕ) (marked_second: ℕ) (catch_first: ℕ) (catch_second: ℕ) : ℕ :=
  (marked_first * catch_second) / marked_second

theorem fish_in_pond (h1 : marked_first = 30) (h2 : marked_second = 2) (h3 : catch_first = 30) (h4 : catch_second = 40) :
  number_of_fish marked_first marked_second catch_first catch_second = 600 :=
by
  rw [h1, h2, h3, h4]
  sorry

end fish_in_pond_l776_776607


namespace trapezoid_ad_length_l776_776748

theorem trapezoid_ad_length (A B C D O P : Point) (AB CD : Line) 
  (h_parallel : AB ∥ CD) 
  (h_eq1 : dist B C = 37) 
  (h_eq2 : dist C D = 37) 
  (h_perp : AD ⊥ BD) 
  (h_midpoint : midpoint O BD) 
  (h_height : height A O BD)
  (h_op : dist O P = 17) : 
  ∃ m n : ℕ, n.coprime (n.factor) ∧ AD = m * sqrt n ∧ m + n = 42 := 
by 
  sorry

end trapezoid_ad_length_l776_776748


namespace scatter_plot_convention_l776_776376

def explanatory_variable := "x-axis"
def predictor_variable := "y-axis"

theorem scatter_plot_convention :
  explanatory_variable = "x-axis" ∧ predictor_variable = "y-axis" :=
by sorry

end scatter_plot_convention_l776_776376


namespace binomial_coefficient_10_3_l776_776466

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l776_776466


namespace relationship_among_abc_l776_776556

variable {f : ℝ → ℝ}

-- Conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def odd_function_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x + 1) = -f (x + 1)

def decreasing_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → y ≤ 1 → f x ≥ f y

def nonnegative_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x

-- Result
theorem relationship_among_abc (h1 : even_function f) (h2 : odd_function_shifted f)
  (h3 : decreasing_on_unit_interval f) (h4 : nonnegative_on_unit_interval f) :
  let a := f 2010
  let b := f (5 / 4)
  let c := -f (1 / 2)
  in a < c ∧ c < b :=
sorry

end relationship_among_abc_l776_776556


namespace problem_statement_l776_776567

-- Define points as a structure
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Declare points A and B
def A : Point3D := { x := 3, y := 1, z := 3 }
def B : Point3D := { x := 1, y := 5, z := 0 }

-- Function to compute midpoint of two points
def midpoint (P Q : Point3D) : Point3D :=
  { x := (P.x + Q.x) / 2,
    y := (P.y + Q.y) / 2,
    z := (P.z + Q.z) / 2 }

-- Function to compute vector AB
def vectorAB (P Q : Point3D) : Point3D :=
  { x := Q.x - P.x,
    y := Q.y - P.y,
    z := Q.z - P.z }

-- Function to compute the length of a vector
def vectorLength (v : Point3D) : ℝ :=
  real.sqrt (v.x^2 + v.y^2 + v.z^2)

-- Function to compute distance between point P and A
def distancePA (P : Point3D) : ℝ :=
  real.sqrt ((P.x - A.x)^2 + (P.y - A.y)^2 + (P.z - A.z)^2)

-- Function to compute distance between point P and B
def distancePB (P : Point3D) : ℝ :=
  real.sqrt ((P.x - B.x)^2 + (P.y - B.y)^2 + (P.z - B.z)^2)

-- Problem statement
theorem problem_statement :
  midpoint A B = { x := 2, y := 3, z := 3/2 } ∧
  vectorLength (vectorAB A B) = real.sqrt 29 ∧
  ∀ (P : Point3D), distancePA P = distancePB P → 4 * P.x - 8 * P.y + 6 * P.z + 7 = 0 :=
by sorry

end problem_statement_l776_776567


namespace num_divisors_of_30_l776_776196

-- Define what it means to be a divisor of 30
def is_divisor (n : ℤ) : Prop :=
  n ≠ 0 ∧ 30 % n = 0

-- Define a function that counts the number of divisors of 30
def count_divisors (d : ℤ) : ℕ :=
  ((Multiset.filter is_divisor ((List.range' (-30) 61).map (Int.ofNat))).attach.to_finset.card)

-- State the theorem
theorem num_divisors_of_30 : count_divisors 30 = 16 := sorry

end num_divisors_of_30_l776_776196


namespace copper_wire_length_l776_776356

theorem copper_wire_length
  (ρ : ℝ := 8900)                           -- Density of copper in kg/m³
  (V : ℝ := 0.5 * 10^(-3))                   -- Volume of copper in m³
  (diagonal_mm : ℝ := 2)                    -- Diagonal of the square cross-section in mm
  (length_threshold : ℝ := 225) :            -- Length threshold in meters
  let s_mm := diagonal_mm / Real.sqrt 2 in    -- Side length in mm
  let s := s_mm * 10^(-3) in                  -- Side length in meters
  let A := s^2 in                             -- Cross-sectional area in m²
  let L := V / A in                           -- Length of the wire in meters
  L > length_threshold := 
by
  -- Proof goes here
  sorry

end copper_wire_length_l776_776356


namespace change_nonintegral_numbers_l776_776043

theorem change_nonintegral_numbers (A : list (list ℝ)) (h1 : ∀ row ∈ A, ⌊(list.sum row)⌋ = list.sum row) (h2 : ∀ col ∈ (list.transpose A), ⌊(list.sum col)⌋ = list.sum col) :
  ∃ B : list (list ℝ), (∀ i j, A[i][j] ∉ ℤ → B[i][j] = ⌊A[i][j]⌋ ∨ B[i][j] = ⌈A[i][j]⌉) ∧
  (∀ row ∈ B, ⌊(list.sum row)⌋ = list.sum row) ∧ (∀ col ∈ (list.transpose B), ⌊(list.sum col)⌋ = list.sum col) :=
  sorry

end change_nonintegral_numbers_l776_776043


namespace binomial_coefficient_10_3_l776_776474

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l776_776474


namespace largest_n_le_2_l776_776951

theorem largest_n_le_2 : 
  ∃ n : ℕ, (∀ (a b c d : ℝ), (n + 2) * real.sqrt (a^2 + b^2) + (n + 1) * real.sqrt (a^2 + c^2) + (n + 1) * real.sqrt (a^2 + d^2) 
  ≥ n * (a + b + c + d)) ∧ n = 2 :=
begin
  use 2,
  intros a b c d,
  sorry
end

end largest_n_le_2_l776_776951


namespace unique_or_not_l776_776631

def is_unique_pyramid
  (a : Real) -- slant height
  (h : Real) -- height of the pyramid
  (u : Real) -- side length of the base (unique side length)
  (v : Real) -- diagonal of the base
  (b : Real) -- base area
  (vol : Real) -- volume of the pyramid
  (e : Real) -- edge length along the base
  (s_area : Real) -- total surface area of the pyramid
  : Prop :=
  u = (2 * sqrt(v^2 - a^2)) ∧ h = (vol / b * 3) ∨ s_area = (b + 2 * e * sqrt(a^2 - (e / 2)^2) + e^2)

theorem unique_or_not 
  (a : Real) -- slant height
  (h : Real) -- height of the pyramid
  (u : Real) -- side length of the base
  (v : Real) -- diagonal of the base
  (b : Real) -- base area
  (vol : Real) -- volume of the pyramid
  (e : Real) -- edge length along the base
  (s_area : Real) -- total surface area of the pyramid
  (condA : ∃ u b vol s_area, ¬is_unique_pyramid a h u v b vol e s_area)
  : ¬condA := sorry

end unique_or_not_l776_776631


namespace problem_statement_l776_776124

theorem problem_statement (a : ℕ → ℝ) :
  (∀ x : ℝ, (1 - 2 * x) ^ 2017 = ∑ i in Finset.range 2018, a i * (x - 1) ^ i) → 
  a 1 - 2 * a 2 + 3 * a 3 - 4 * a 4 + ... - 2016 * a 2016 + 2017 * a 2017 = -4034 :=
by sorry

end problem_statement_l776_776124


namespace ellipse_problem_l776_776999

noncomputable def a_b_sum : ℕ := 305

theorem ellipse_problem (x y : ℝ) (k1 : ℝ) (h1 : k1 ≠ 0) (h2 : k1 ≠ ∞)
  (A B C D : ℝ × ℝ)
  (h3 : ∀ {x₁ y₁ x₂ y₂ : ℝ}, (x₁, y₁) ∈ C → A ∈ B → ∀ {k₁, k₂ : ℝ}, k₁ ≠ 0 → k₁ ≠ ∞ → 
  (y₁ * x₂ - y₂ * x₁ = 2 * (y₂ - y₁)) ∧
  ((x₁ - 1) / y₁ * y + 1) && (x₃, y₃) && (x ≠ (5 * x₁ - 9) / (x₁ - 5))}
  : a_b_sum = 305 := 
sorry

end ellipse_problem_l776_776999


namespace inequality_AM_GM_l776_776128

theorem inequality_AM_GM (a b t : ℝ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : 0 < t) : 
  (a^2 / (b^t - 1) + b^(2 * t) / (a^t - 1)) ≥ 8 :=
by
  sorry

end inequality_AM_GM_l776_776128


namespace total_miles_driven_l776_776316

-- Given constants and conditions
def city_mpg : ℝ := 30
def highway_mpg : ℝ := 37
def total_gallons : ℝ := 11
def highway_extra_miles : ℕ := 5

-- Variable for the number of city miles
variable (x : ℝ)

-- Conditions encapsulated in a theorem statement
theorem total_miles_driven:
  (x / city_mpg) + ((x + highway_extra_miles) / highway_mpg) = total_gallons →
  x + (x + highway_extra_miles) = 365 :=
by
  sorry

end total_miles_driven_l776_776316


namespace odd_periodic_function_l776_776928

variable {f : ℝ → ℝ}

-- Given conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic_function (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f x

-- Problem statement
theorem odd_periodic_function (h_odd : odd_function f)
  (h_period : periodic_function f) (h_half : f 0.5 = 1) : f 7.5 = -1 :=
sorry

end odd_periodic_function_l776_776928


namespace midpoint_trajectory_l776_776273

variables {x y x_0 y_0 : ℝ}

/-- 
  Given \( P = (x, y) \) is a point on the curve \( 2x^2 - y^2 = 1 \),
  and \( M = (x_0, y_0) \) is the midpoint of the line segment from the origin to \( P \),
  prove that \( M \) lies on the curve \( 8x_0^2 - y_0^2 = 1 \).
-/
theorem midpoint_trajectory :
  (2 * x^2 - y^2 = 1) → 
  (x_0 = x / 2) → 
  (y_0 = y / 2) → 
  (8 * x_0^2 - y_0^2 = 1) :=
begin
  intros h1 h2 h3,
  rw [h2, h3] at h1,
  sorry
end

end midpoint_trajectory_l776_776273


namespace dima_can_determine_equation_l776_776298

theorem dima_can_determine_equation (e1 e2 e3 e4 : ℝ)
  (h_distinct : e1 ≠ e2 ∧ e1 ≠ e3 ∧ e1 ≠ e4 ∧ e2 ≠ e3 ∧ e2 ≠ e4 ∧ e3 ≠ e4) :
  ∃ (a b c d : ℝ), a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (∃ (roots : Finset ℝ), roots = {c, d} ∧ (a ∈ {e1, e2, e3, e4}) ∧ (b ∈ {e1, e2, e3, e4})) :=
sorry

end dima_can_determine_equation_l776_776298


namespace teamA_is_champion_probability_l776_776245

noncomputable def teamA_probability_of_champ : ℝ :=
  let p := 1 / 2 in -- probability of winning a single game for both teams
  let p_A_wins_first_game := p in
  let p_A_loses_first_wins_second := (1 - p) * p in
  p_A_wins_first_game + p_A_loses_first_wins_second

theorem teamA_is_champion_probability : teamA_probability_of_champ = 3 / 4 :=
by
  sorry

end teamA_is_champion_probability_l776_776245


namespace lunch_special_cost_l776_776059

theorem lunch_special_cost (total_bill : ℕ) (num_people : ℕ) (cost_per_lunch_special : ℕ)
  (h1 : total_bill = 24) 
  (h2 : num_people = 3) 
  (h3 : cost_per_lunch_special = total_bill / num_people) : 
  cost_per_lunch_special = 8 := 
by
  sorry

end lunch_special_cost_l776_776059


namespace min_value_le_one_l776_776166

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x
noncomputable def g (a : ℝ) : ℝ := a - a * Real.log a

theorem min_value_le_one (a : ℝ) (ha : a > 0) :
  (∀ x : ℝ, f x a ≥ g a) ∧ g a ≤ 1 := sorry

end min_value_le_one_l776_776166


namespace product_of_constants_t_l776_776535

theorem product_of_constants_t (a b : Int) (t : Int) 
  (h1 : a * b = 12) 
  (h2 : t = a + b) :
  ∏ t in {t | ∃ a b : Int, a * b = 12 ∧ t = a + b}, t = 530816 :=
by
  sorry

end product_of_constants_t_l776_776535


namespace smallest_three_digit_multiple_of_13_l776_776791

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 13 = 0 → n ≤ m :=
⟨104, by sorry⟩

end smallest_three_digit_multiple_of_13_l776_776791


namespace fewest_number_of_printers_l776_776383

/-- Given the costs of two types of printers, assert the fewest total number of printers to purchase equal amounts for both types is 15. -/
theorem fewest_number_of_printers {cost1 cost2 : ℕ} (h1 : cost1 = 400) (h2 : cost2 = 350) :
  (let lcm := Nat.lcm cost1 cost2,
        numPr1 := lcm / cost1,
        numPr2 := lcm / cost2
   in numPr1 + numPr2 = 15) :=
by
  -- Proof omitted for this statement
  sorry

end fewest_number_of_printers_l776_776383


namespace biathlete_average_speed_l776_776028

open Real

def harmonic_mean (a b : ℝ) : ℝ :=
  2 / (1 / a + 1 / b)

theorem biathlete_average_speed
  (cycling_speed : ℝ)
  (running_speed : ℝ)
  (average_speed : ℝ)
  (h_cycle : cycling_speed = 18)
  (h_run : running_speed = 8)
  (h_avg : average_speed = harmonic_mean cycling_speed running_speed) :
  average_speed = 144 / 13 := by
  rw [h_cycle, h_run, h_avg]
  unfold harmonic_mean
  have : 2 / (1 / 18 + 1 / 8) = 144 / 13 := by
    norm_num
  exact this

end biathlete_average_speed_l776_776028


namespace find_speed_l776_776926

-- Define the necessary conditions
variables (t1 t2 d travel_time speed : ℝ)

-- Given constraints
axiom t2_eq_2t1 : t2 = 2 * t1
axiom time_difference : t2 - t1 = 3 / 4
axiom distance_eq : d = 20 * t1
axiom travel_time_eq : travel_time = 5 - 3.75  -- 5:00 PM - 3:45 PM in hours

theorem find_speed : speed = 12 :=
by
  -- Calculating t1
  have h1 : t1 = 3 / 4,
  { sorry }
  -- Calculating distance
  have h_dist : d = 15,
  { sorry }
  -- Calculating speed
  have h_speed : speed = d / travel_time,
  { sorry }
  -- Prove the final speed is 12 km/h
  sorry

end find_speed_l776_776926


namespace maximize_net_income_l776_776422

-- Define the conditions of the problem
def bicycles := 50
def management_cost := 115

def rental_income (x : ℕ) : ℕ :=
if x ≤ 6 then bicycles * x
else (bicycles - 3 * (x - 6)) * x

def net_income (x : ℕ) : ℤ :=
rental_income x - management_cost

-- Define the domain of the function
def domain (x : ℕ) : Prop := 3 ≤ x ∧ x ≤ 20

-- Define the piecewise function for y = f(x)
def f (x : ℕ) : ℤ :=
if 3 ≤ x ∧ x ≤ 6 then 50 * x - 115
else if 6 < x ∧ x ≤ 20 then -3 * x * x + 68 * x - 115
else 0  -- Out of domain

-- The theorem that we need to prove
theorem maximize_net_income :
  (∀ x, domain x → net_income x = f x) ∧
  (∃ x, domain x ∧ (∀ y, domain y → net_income y ≤ net_income x) ∧ x = 11) :=
by
  sorry

end maximize_net_income_l776_776422


namespace nested_function_evaluation_l776_776587

def f : ℝ → ℝ :=
  λ x, if x >= 0 then 3 * x - 2 else 2 ^ x

theorem nested_function_evaluation : f (f (-1)) = -1 / 2 :=
by
  sorry

end nested_function_evaluation_l776_776587


namespace binomial_coefficient_10_3_l776_776486

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l776_776486


namespace reunion_handshakes_l776_776905

/-- 
Given 15 boys at a reunion:
- 5 are left-handed and will only shake hands with other left-handed boys.
- Each boy shakes hands exactly once with each of the others unless they forget.
- Three boys each forget to shake hands with two others.

Prove that the total number of handshakes is 49. 
-/
theorem reunion_handshakes : 
  let total_boys := 15
  let left_handed := 5
  let forgetful_boys := 3
  let forgotten_handshakes_per_boy := 2

  let total_handshakes := total_boys * (total_boys - 1) / 2
  let left_left_handshakes := left_handed * (left_handed - 1) / 2
  let left_right_handshakes := left_handed * (total_boys - left_handed)
  let distinct_forgotten_handshakes := forgetful_boys * forgotten_handshakes_per_boy / 2

  total_handshakes 
    - left_right_handshakes 
    - distinct_forgotten_handshakes
    - left_left_handshakes
  = 49 := 
sorry

end reunion_handshakes_l776_776905


namespace angle_A_is_30_degrees_l776_776225

theorem angle_A_is_30_degrees
    (a b : ℝ)
    (B A : ℝ)
    (a_eq_4 : a = 4)
    (b_eq_4_sqrt2 : b = 4 * Real.sqrt 2)
    (B_eq_45 : B = Real.pi / 4) : 
    A = Real.pi / 6 := 
by 
    sorry

end angle_A_is_30_degrees_l776_776225


namespace determine_wholesale_prices_l776_776850

variables (p1 p2 d k : ℝ)
variables (h1 : 1 < k)
variables (prices : Finset ℝ)
variables (p_plus : ℝ → ℝ) (star : ℝ → ℝ)

noncomputable def BakeryPrices : Prop :=
  p_plus p1 = p1 + d ∧ p_plus p2 = p2 + d ∧ 
  star p1 = k * p1 ∧ star p2 = k * p2 ∧ 
  prices = {64, 64, 70, 72} ∧ 
  ∃ (q1 q2 q3 q4 : ℝ), 
    (q1 = p_plus p1 ∨ q1 = star p1 ∨ q1 = p_plus p2 ∨ q1 = star p2) ∧
    (q2 = p_plus p1 ∨ q2 = star p1 ∨ q2 = p_plus p2 ∨ q2 = star p2) ∧
    (q3 = p_plus p1 ∨ q3 = star p1 ∨ q3 = p_plus p2 ∨ q3 = star p2) ∧
    (q4 = p_plus p1 ∨ q4 = star p1 ∨ q4 = p_plus p2 ∨ q4 = star p2) ∧
    prices = {q1, q2, q3, q4}

theorem determine_wholesale_prices (p1 p2 : ℝ) (d k : ℝ)
    (h1 : 1 < k) (prices : Finset ℝ) :
  BakeryPrices p1 p2 d k h1 prices (λ p, p + d) (λ p, k * p) :=
sorry

end determine_wholesale_prices_l776_776850


namespace hyperbola_eccentricity_l776_776580

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (hx : b / a = 3 / 4 ∨ a / b = 3 / 4) : ℝ :=
  if b / a = 3 / 4 then
    real.sqrt (a^2 + (3/4 * a)^2) / a
  else
    if a / b = 3 / 4 then
      real.sqrt ((4/3 * b)^2 + b^2) / b
    else
      0 -- This should never happen but ensures the definition is total

theorem hyperbola_eccentricity (a b : ℝ) (hx : b / a = 3 / 4 ∨ a / b = 3 / 4) :
  eccentricity_of_hyperbola a b hx = 5 / 4 ∨ eccentricity_of_hyperbola a b hx = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l776_776580


namespace problem_statement_l776_776982

variable {ℝ}

section
  variable (f : ℝ → ℝ)
  variable (y : ℝ)

  -- Conditions
  def function_inverse : Prop := 
    ∃ (f_inv : ℝ → ℝ), ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

  def f_property : Prop := 
    ∀ x : ℝ, f x + f (- x) = 1

  -- Question and proof goal
  theorem problem_statement 
    (hf_inv : function_inverse f) 
    (hf_prop : f_property f) (x : ℝ) :
    (f^{-1}(2010 - x) + f^{-1}(x - 2009) = 0) :=
  sorry
end

end problem_statement_l776_776982


namespace solution1_solution2_l776_776637

noncomputable def problem1 (a c: ℝ) (C A: ℝ) (h₀ : c = 2) (h₁ : C = π / 3) (h₂ : a = (2 * sqrt 3) / 3) : Prop :=
  A = π / 6

noncomputable def problem2 (a b c: ℝ) (C: ℝ) (B A: ℝ) (h₀ : c = 2) (h₁ : C = π / 3) (h₂ : sin B = 2 * sin A) (h₃ : a = (2 * sqrt 3) / 3) : Prop :=
  let area := 1/2 * a * b * sin C in
  area = (2 * sqrt 3) / 3

theorem solution1 (a c A C: ℝ) (h₀ : c = 2) (h₁ : C = π / 3) (h₂ : a = (2 * sqrt 3) / 3) : problem1 a c C A h₀ h₁ h₂ :=
sorry

theorem solution2 (a b c A B C: ℝ) (h₀ : c = 2) (h₁ : C = π / 3) (h₂ : sin B = 2 * sin A) (h₃ : a = (2 * sqrt 3) / 3) (h4 : b = 2*a) : problem2 a b c C B A h₀ h₁ h₂ h₃ :=
sorry

end solution1_solution2_l776_776637


namespace percent_saved_by_purchasing_kit_l776_776396

def total_price_individual_filters : ℝ := (3 * 7.35) + (2 * 12.05) + (1 * 12.50)
def kit_price : ℝ := 75.50
def saved_amount : ℝ := kit_price - total_price_individual_filters
def percent_saved : ℝ := (saved_amount / total_price_individual_filters) * 100

theorem percent_saved_by_purchasing_kit : percent_saved = 28.72 := 
by
  sorry

end percent_saved_by_purchasing_kit_l776_776396


namespace divisors_of_30_l776_776185

theorem divisors_of_30 : 
  ∃ n : ℕ, (∀ d : ℤ, d ∣ 30 → d > 0 → d ≤ 30) ∧ (∀ d : ℤ, d ∣ 30 → -d ∣ 30) ∧ (∀ d : ℤ, d ∣ 30 → -d = d → d = 0) ∧ 2 * (Finset.card (Finset.filter (λ d, 0 < d) (Finset.filter (λ d, d ∣ 30) (Finset.Icc 1 30).toFinset))) = 16 :=
by {
   sorry
}

end divisors_of_30_l776_776185


namespace solution_set_f_inv_gt_2_l776_776586

theorem solution_set_f_inv_gt_2 (x : ℝ) (h : 0 < x) :
  (∃ y > 1, y < 3/2 ∧ f (f⁻¹ x) = x ∧ 1 + 1 / f⁻¹ y = y) ↔ (1 < x ∧ x < 3/2) :=
sorry

end solution_set_f_inv_gt_2_l776_776586


namespace least_positive_three_digit_multiple_of_11_l776_776761

theorem least_positive_three_digit_multiple_of_11 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 11 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 11 ∣ m → n ≤ m) :=
begin
  use 110,
  split,
  exact (100 ≤ 110),
  split,
  exact (110 ≤ 999),
  split,
  exact (11 ∣ 110),
  intros m Hm,
  cases Hm with Hm100 HmGCD,
  cases HmGCD with Hm999 HmDiv,
  sorry,
end

end least_positive_three_digit_multiple_of_11_l776_776761


namespace binom_10_3_l776_776456

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l776_776456


namespace binomial_coefficient_10_3_l776_776493

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l776_776493


namespace find_n_l776_776943

theorem find_n (n : ℤ) : 5^(4 * n) = (1/5)^(n - 30) → n = 6 := by
  intro h
  rw [←zpow_neg_one, ←zpow_mul, neg_eq_iff_add_eq_zero] at h
  sorry

end find_n_l776_776943


namespace problem_accelerations_l776_776350

/-- Define the given quantities and conditions. --/
def m1 : ℝ := 2
def m2 : ℝ := 3
def mu1 : ℝ := 0.1
def mu2 : ℝ := 0.2
def F : ℝ := 10
def g : ℝ := 10

/-- Define the friction forces. --/
def F_tr1 : ℝ := mu1 * m1 * g
def F_tr2 : ℝ := mu2 * m2 * g

/-- Define the tension in the string. --/
def T : ℝ := F / 2

/-- Define the accelerations of the blocks. --/
def a1 : ℝ := (T - F_tr1) / m1
def a2 : ℝ := 0

/-- Define the acceleration of the center of mass. --/
def a : ℝ := (m1 * a1 + m2 * a2) / (m1 + m2)

/-- The proof problem: Given the conditions, prove the accelerations. --/
theorem problem_accelerations :
  a1 = 1.5 ∧
  a2 = 0 ∧
  a = 0.6 :=
by
  sorry

end problem_accelerations_l776_776350


namespace probability_x_lt_3y_l776_776408

theorem probability_x_lt_3y (x y : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ 6) (h₃ : 0 ≤ y) (h₄ : y ≤ 2) :
    ((area {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 ∧ p.1 < 3 * p.2})
     / (area {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2})) = 1 / 2 := 
sorry

end probability_x_lt_3y_l776_776408


namespace total_tickets_sold_l776_776713

theorem total_tickets_sold 
  (A D : ℕ) 
  (cost_adv cost_door : ℝ) 
  (revenue : ℝ)
  (door_tickets_sold total_tickets : ℕ) 
  (h1 : cost_adv = 14.50) 
  (h2 : cost_door = 22.00)
  (h3 : revenue = 16640) 
  (h4 : door_tickets_sold = 672) : 
  (total_tickets = 800) :=
by
  sorry

end total_tickets_sold_l776_776713


namespace problem1_problem2_l776_776393

-- Problem (I)
theorem problem1 (a b : ℝ) (h : a ≥ b ∧ b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := sorry

-- Problem (II)
theorem problem2 (a b c x y z : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : a^2 + b^2 + c^2 = 10) 
  (h2 : x^2 + y^2 + z^2 = 40) 
  (h3 : a * x + b * y + c * z = 20) : 
  (a + b + c) / (x + y + z) = 1 / 2 := sorry

end problem1_problem2_l776_776393


namespace complex_number_quadrant_Z_l776_776328

open Complex

def is_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem complex_number_quadrant_Z : is_third_quadrant (-1 - 2 * Complex.i) :=
by
  sorry

end complex_number_quadrant_Z_l776_776328


namespace product_of_c_values_with_rational_solutions_l776_776742

theorem product_of_c_values_with_rational_solutions :
  let c_values := {c : ℕ | (∃ k : ℤ, 289 - 28 * c = k ^ 2) ∧ c > 0} in
  (∏ c in c_values, c) = 60 :=
by
  let c_values := {c : ℕ | (∃ k : ℤ, 289 - 28 * c = k ^ 2) ∧ c > 0}
  have h1 : c_values = {6, 10}, 
  -- Proof steps would go here
  sorry,
  have h2 : ∏ c in {6, 10}, c = 60, 
  -- Proof steps for the product go here
  sorry,
  exact h2

end product_of_c_values_with_rational_solutions_l776_776742


namespace time_to_cross_pole_l776_776423

-- Define the conditions
def train_length : ℝ := 110
def train_speed_kmh : ℝ := 144
def conversion_factor : ℝ := 1000 / 3600
def train_speed_ms : ℝ := train_speed_kmh * conversion_factor

-- State the theorem
theorem time_to_cross_pole (d : ℝ := train_length) (v : ℝ := train_speed_ms) : 
  d / v = 2.75 :=
by
  sorry

end time_to_cross_pole_l776_776423


namespace smallest_integer_with_2015_factors_sum_prime_factors_l776_776282

theorem smallest_integer_with_2015_factors_sum_prime_factors :
  let n := 2^30 * 3^12 * 5^4 in
  ∀ n, 
    (∀ d : ℕ, d ∣ n → d ∈ {d | d ∣ 2015}) → 
    (n = 2^30 * 3^12 * 5^4) → 
    (2 * 30 + 3 * 12 + 5 * 4 = 116) := 
by begin
  intros n h1 h2,
  have h3 : 2^30 * 3^12 * 5^4 = n, from sorry,
  have h4 : 2 * 30 + 3 * 12 + 5 * 4 = 116, from sorry,
  exact h4,
  sorry
end

end smallest_integer_with_2015_factors_sum_prime_factors_l776_776282


namespace find_a_l776_776598

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a*x - 1 = 0}

theorem find_a (a : ℝ) (h : B a ⊆ A) : a = 0 ∨ a = -1 ∨ a = (1 / 3) :=
sorry

end find_a_l776_776598


namespace log_50_between_integers_l776_776342

open Real

-- Declaration of the proof problem
theorem log_50_between_integers (a b : ℤ) (h1 : log 10 = 1) (h2 : log 100 = 2) (h3 : 10 < 50) (h4 : 50 < 100) :
  a + b = 3 :=
by
  sorry

end log_50_between_integers_l776_776342


namespace probability_top_card_king_or_queen_l776_776421

theorem probability_top_card_king_or_queen :
  let ranks := 13
  let suits := 4
  let deck_size := ranks * suits
  let kings := suits -- 4 Kings
  let queens := suits -- 4 Queens
  let favorable_outcomes := kings + queens
  (deck_size = 52) →
  (favorable_outcomes = 8) →
  (deck_size ≠ 0) →
  Probability 
    := favourable_outcomes / deck_size = 2 / 13 := 
  sorry

end probability_top_card_king_or_queen_l776_776421


namespace three_digits_divisibility_l776_776113

theorem three_digits_divisibility :
  {n : ℕ | n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ (23 * 10 + n) % n = 0}.to_finset.card = 3 :=
by
  sorry

end three_digits_divisibility_l776_776113


namespace find_angle_A_max_triangle_area_l776_776251

noncomputable def angle_condition (b c a : ℝ) (A C : ℝ) : Prop :=
  2 * b * Real.cos A = c * Real.cos A + a * Real.cos C

theorem find_angle_A {a b c A C : ℝ} 
  (h1: angle_condition b c a A C) : A = Real.pi / 3 :=
sorry

theorem max_triangle_area {a b c A : ℝ} 
  (hA : A = Real.pi / 3)
  (ha : a = 4) : 
  let max_area := (4 : ℝ) * Real.sqrt 3 in
  Real.abs ((1 / 2) * b * c * Real.sin A) ≤ max_area :=
sorry

end find_angle_A_max_triangle_area_l776_776251


namespace polynomial_form_l776_776095

theorem polynomial_form (p : ℝ → ℝ) (hp : ∀ a : ℝ, p a ∈ ℤ → a ∈ ℤ) : 
  ∃ C : ℤ, ∀ x : ℝ, p x = x + C := 
sorry

end polynomial_form_l776_776095
