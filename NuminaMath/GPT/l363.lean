import Mathlib

namespace conditional_probability_heads_then_tails_twice_l363_363780

theorem conditional_probability_heads_then_tails_twice :
  let p_heads := 1/2
  let p_tails := 1/2 
  P(B | A) = 1/2 :=
by
  have p_A : P(A) = p_heads := sorry
  have p_B_given_A : P(B | A) = (P(A) * p_tails) / P(A) := sorry
  rw p_A at p_B_given_A
  norm_num at p_B_given_A
  exact p_B_given_A

end conditional_probability_heads_then_tails_twice_l363_363780


namespace dice_prob_third_roll_red_l363_363880

theorem dice_prob_third_roll_red :
  let total_sides := 10
  let red_sides := 3
  let non_red_sides := total_sides - red_sides
  let p_non_red := (non_red_sides : ℚ) / total_sides
  let p_red := (red_sides : ℚ) / total_sides
  p_non_red * p_non_red * p_red = 147 / 1000 :=
by
  let total_sides := 10
  let red_sides := 3
  let non_red_sides := total_sides - red_sides
  let p_non_red := (non_red_sides : ℚ) / total_sides
  let p_red := (red_sides : ℚ) / total_sides
  calc
    p_non_red * p_non_red * p_red
      = (7/10 : ℚ) * (7/10) * (3/10) : by rfl
  ... = 147 / 1000 : by norm_num

end dice_prob_third_roll_red_l363_363880


namespace sum_of_first_K_natural_numbers_is_perfect_square_l363_363420

noncomputable def values_K (K : ℕ) : Prop := 
  ∃ N : ℕ, (K * (K + 1)) / 2 = N^2 ∧ (N + K < 120)

theorem sum_of_first_K_natural_numbers_is_perfect_square :
  ∀ K : ℕ, values_K K ↔ (K = 1 ∨ K = 8 ∨ K = 49) := by
  sorry

end sum_of_first_K_natural_numbers_is_perfect_square_l363_363420


namespace arithmetic_expression_evaluation_l363_363300

theorem arithmetic_expression_evaluation :
  4 * 6 + 8 * 3 - 28 / 2 = 34 := by
  sorry

end arithmetic_expression_evaluation_l363_363300


namespace solve_fractional_equation_l363_363209

theorem solve_fractional_equation (x : ℚ) (h: x ≠ 1) : 
  (x / (x - 1) = 3 / (2 * x - 2) - 2) ↔ (x = 7 / 6) := 
by
  sorry

end solve_fractional_equation_l363_363209


namespace problem1_problem2_l363_363086

def f (a b x : ℝ) : ℝ := a * x - b / x - 2 * (Real.log x)
def f_prime (a x : ℝ) : ℝ := a + a / (x ^ 2) - 2 / x

theorem problem1 (a b : ℝ) (h : f a b 1 = 0) :
  (∀ x > 0, f_prime a x ≥ 0) ∨ (∀ x > 0, f_prime a x ≤ 0) → 
  (a ∈ set.Iic 0 ∪ set.Ici 1) :=
sorry

theorem problem2 (f : ℝ → ℝ) (f_prime : ℝ → ℝ) (a_n : ℕ → ℝ)
  (h_prime_at_1 : f_prime 1 = 0) (h : ∀ n, a_n (n+1) = f' (1 / (a_n n + 1)) - n * a_n n + 1)
  (h_a1 : a_n 1 ≥ 3) :
  (∀ n, a_n n ≥ n + 2) :=
sorry

end problem1_problem2_l363_363086


namespace intersection_of_parabolas_l363_363292

noncomputable def intersection_points : set (ℝ × ℝ) :=
  {p | p.2 = 4 * p.1^2 + 5 * p.1 - 6 ∧ p.2 = p.1^2 + 14}

theorem intersection_of_parabolas :
  intersection_points = {(-4, 38), (5/3, 121/9)} :=
by
  sorry

end intersection_of_parabolas_l363_363292


namespace exists_integer_solution_l363_363321

theorem exists_integer_solution (x : ℤ) (h : x - 1 < 0) : ∃ y : ℤ, y < 1 :=
by
  sorry

end exists_integer_solution_l363_363321


namespace negation_statement_l363_363536

theorem negation_statement (x y : ℝ) :
  (x ≤ 0 ∨ y ≤ 0) → (x + y ≤ 0) :=
begin
  sorry
end

end negation_statement_l363_363536


namespace largest_divisor_three_consecutive_l363_363160

theorem largest_divisor_three_consecutive (u v w : ℤ) (h1 : u + 1 = v) (h2 : v + 1 = w) (h3 : ∃ n : ℤ, (u = 5 * n) ∨ (v = 5 * n) ∨ (w = 5 * n)) : 
  ∀ d ∈ {d | ∀ a b c : ℤ, a * b * c = u * v * w → d ∣ a * b * c}, 
  15 ∈ {d | ∀ a b c : ℤ, a * b * c = u * v * w → d ∣ a * b * c} :=
sorry

end largest_divisor_three_consecutive_l363_363160


namespace find_sum_of_squares_l363_363539

theorem find_sum_of_squares (x y z : ℝ)
  (h1 : x^2 + 3 * y = 8)
  (h2 : y^2 + 5 * z = -9)
  (h3 : z^2 + 7 * x = -16) : x^2 + y^2 + z^2 = 20.75 :=
sorry

end find_sum_of_squares_l363_363539


namespace arithmetic_sequence_and_sum_l363_363932

open BigOperators

noncomputable def a_seq (n : ℕ) : ℕ := 1 / 2^(n - 1)
noncomputable def b_seq (n : ℕ) : ℕ := 2 * n - 1
noncomputable def c_seq (n : ℕ) : ℕ := 2^(n - 1) + 2 * n - 1

theorem arithmetic_sequence_and_sum (n : ℕ) :
  (a_seq 1 = 1) ∧ (b_seq 1 = 1) ∧ (b_seq 2 + b_seq 4 = 10) ∧
  (∀ n, a_seq n ^ 2 - 2 * a_seq n * a_seq (n + 1) + a_seq n - 2 * a_seq (n + 1) = 0) ∧
  (S n = ∑ i in finset.range n, (1 / a_seq i + b_seq i)) →
  S n = 2^n + n^2 - 1 := 
sorry

end arithmetic_sequence_and_sum_l363_363932


namespace Mary_winning_strategy_l363_363197

-- Definitions to set up the game
def game (N : ℕ) (d : ℕ) : ℕ := if N % d == 0 then 0 else N - d

-- Proof statement
theorem Mary_winning_strategy : ∀ N : ℕ, N > 100 → ∃ ds : List ℕ, (∀ d ∈ ds, d > 1 ∧ d ∉ List.remove ds d) ∧ ∃ n, game N (List.head ds) = 0 := 
by
  sorry

end Mary_winning_strategy_l363_363197


namespace count_pairs_satisfy_eq_l363_363102

theorem count_pairs_satisfy_eq (N : ℤ) : 
  N = (∃ m n : ℤ, (m - 1) * (n - 1) = 2) 
    → finset.card (finset.filter (λ p, (p.1 - 1) * (p.2 - 1) = 2) (finset.univ : finset (ℤ × ℤ))) = 4 :=
sorry

end count_pairs_satisfy_eq_l363_363102


namespace quadratic_real_root_iff_b_range_l363_363587

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l363_363587


namespace quadratic_has_real_root_l363_363567

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l363_363567


namespace find_f_neg_5_l363_363925

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 5 + 1 else - (log (-x) / log 5 + 1)

theorem find_f_neg_5
  (h_odd : ∀ x : ℝ, f (-x) = -f (x))
  (h_domain : ∀ x : ℝ, x ∈ set.univ)
  (h_positive_def : ∀ x : ℝ, x > 0 → f x = log x / log 5 + 1)
  : f (-5) = -2 :=
by
  sorry

end find_f_neg_5_l363_363925


namespace insurance_calculation_l363_363351

-- Define the conditions
def baseRate : ℝ := 0.002 -- 0.2% as a decimal
def reducingCoefficient : ℝ := 0.8
def increasingCoefficient : ℝ := 1.3
def assessedValue : ℝ := 14500000
def cadasterValue : ℝ := 15000000
def loanAmount : ℝ := 20000000

-- Define the adjusted tariff
def adjustedTariff : ℝ := baseRate * reducingCoefficient * increasingCoefficient

-- Define the insurable amount
def insurableAmount : ℝ := max assessedValue cadasterValue

-- Define the proof problem to show equality
theorem insurance_calculation :
  adjustedTariff = 0.00208 ∧ insurableAmount * adjustedTariff = 31200 := by
  sorry

end insurance_calculation_l363_363351


namespace range_of_a_l363_363054

noncomputable def f (a x : ℝ) : ℝ := x - a * x * Real.log x

theorem range_of_a
  (a : ℝ) :
  (∃ x_0 ∈ Set.Icc Real.exp (Real.exp 2), f a x_0 ≤ (1 / 4) * Real.log x_0) → 
  a ≤ 1 - 1 / (4 * Real.exp) :=
by
  sorry

end range_of_a_l363_363054


namespace num_pairs_lcm_eq_num_divisors_l363_363670

open Nat

theorem num_pairs_lcm_eq_num_divisors (n : ℕ) :
  (∃ uv_pairs : Finset (ℕ × ℕ), 
    (∀ uv ∈ uv_pairs, let (u, v) := uv in lcm u v = n)
    ∧ set.card (uv_pairs.to_finset) = (nat.divisors (n^2)).card) :=
begin
  sorry
end

end num_pairs_lcm_eq_num_divisors_l363_363670


namespace area_enclosed_by_curves_l363_363219

theorem area_enclosed_by_curves :
  (∫ y in 1..Real.sqrt 3, (y^2 - y)) = 
  ∫ y in 1..Real.sqrt 3, (y^2 - y) := by
  sorry

end area_enclosed_by_curves_l363_363219


namespace sequence_formula_l363_363141

theorem sequence_formula (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (h : ∀ n, S_n n = 3 + 2 * a_n n) :
  ∀ n, a_n n = -3 * 2^(n - 1) :=
by
  sorry

end sequence_formula_l363_363141


namespace max_value_expression_l363_363179

theorem max_value_expression (x1 x2 x3 : ℝ) (h1 : x1 + x2 + x3 = 1) (h2 : 0 < x1) (h3 : 0 < x2) (h4 : 0 < x3) :
    x1 * x2^2 * x3 + x1 * x2 * x3^2 ≤ 27 / 1024 :=
sorry

end max_value_expression_l363_363179


namespace log_cos_x_eq_half_log_one_minus_b_2a_l363_363605

open Real

variable {b x a : ℝ}

theorem log_cos_x_eq_half_log_one_minus_b_2a (hb : 1 < b) (h_sin_pos : 0 < sin x) (h_cos_pos : 0 < cos x) (h_log_sin : log b (sin x) = a) : 
    log b (cos x) = (1/2) * log b (1 - b^(2 * a)) :=
  sorry

end log_cos_x_eq_half_log_one_minus_b_2a_l363_363605


namespace find_x_to_divisible_by_13_l363_363222

def base4_to_base10 (a b c d : ℕ) : ℕ :=
  a * 4^3 + b * 4^2 + c * 4 + d

theorem find_x_to_divisible_by_13 :
  ∃ x ∈ {0, 1, 2, 3}, (base4_to_base10 2 3 1 x) % 13 = 0 :=
by
  let n := base4_to_base10 2 3 1 1
  have h : n % 13 = 0 := by
    sorry
  refine ⟨1, _, h⟩
  sorry

end find_x_to_divisible_by_13_l363_363222


namespace proof_problem_l363_363479

def p : Prop := 3 ≥ 3
def q : Prop := 3 > 4

theorem proof_problem : (p ∨ q) ∧ (¬(p ∧ q)) ∧ (¬¬p) :=
by
  have h1 : p := by decide
  have h2 : ¬ q := by decide
  have h3 : p ∨ q := Or.inl h1
  have h4 : ¬ (p ∧ q) := by intro hx; cases hx; contradiction
  have h5 : ¬ ¬ p := h1
  exact ⟨h3, h4, h5⟩

end proof_problem_l363_363479


namespace quadratic_has_real_root_iff_b_in_interval_l363_363546

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l363_363546


namespace candies_left_l363_363182

-- Define the initial conditions
def initial_candies (linda : ℝ) : Prop :=
  linda = 34.0

def candies_given (chloe : ℝ) : Prop :=
  chloe = 28.0

-- Define what we want to prove
theorem candies_left (linda chloe : ℝ) (h_initial : initial_candies linda) (h_given : candies_given chloe) : linda - chloe = 6.0 :=
by
  rw [h_initial, h_given]
  norm_num
  sorry

end candies_left_l363_363182


namespace ellipse_eqn_triangle_PAB_not_equilateral_l363_363934

noncomputable section

def standard_equation_of_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  Π x y, (x, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1}

theorem ellipse_eqn
  (a b : ℝ) (ha : a > b) (hb : b > 0)
  (h1 : (3 / 2, - (sqrt 6) / 2) ∈ {p : ℝ × ℝ | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1})
  (h2 : sqrt (a ^ 2 - b ^ 2) / a = sqrt 3 / 3) 
  (h3 : a ^ 2 = b ^ 2 + c ^ 2) :
  ∀ x y, (2 * x ^ 2 / 9 + y ^ 2 / 3 = 1) ↔ (x, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1} :=
  sorry

theorem triangle_PAB_not_equilateral 
  (a b : ℝ) (ha : a > b) (hb : b > 0)
  (h1 : (3 / 2, - (sqrt 6) / 2) ∈ {p : ℝ × ℝ | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1})
  (h2 : sqrt (a ^ 2 - b ^ 2) / a = sqrt 3 / 3) 
  (h3 : a ^ 2 = b ^ 2 + c ^ 2)
  (P : ℝ × ℝ) (hP : P = (1,0)) :
  ∀ A B : ℝ × ℝ, A ≠ B 
  → (A, B) ∈ {p : ℝ × ℝ | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1} 
  → ¬ ∃ A B : ℝ × ℝ, is_equilateral_triangle P A B :=
  sorry

end ellipse_eqn_triangle_PAB_not_equilateral_l363_363934


namespace isabella_stops_l363_363017

def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem isabella_stops (P : ℕ → ℚ) (h : ∀ n, P n = 1 / (n * (n + 1))) : 
  ∃ n : ℕ, n = 55 ∧ P n < 1 / 3000 :=
by {
  sorry
}

end isabella_stops_l363_363017


namespace perimeter_of_triangle_l363_363967

namespace TrianglePerimeter

variables {a b c : ℝ}

-- Conditions translated into definitions
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def absolute_sum_condition (a b c : ℝ) : Prop :=
  |a + b - c| + |b + c - a| + |c + a - b| = 12

-- The theorem stating the perimeter under given conditions
theorem perimeter_of_triangle (h : is_valid_triangle a b c) (h_abs_sum : absolute_sum_condition a b c) : 
  a + b + c = 12 := 
sorry

end TrianglePerimeter

end perimeter_of_triangle_l363_363967


namespace complex_number_powers_l363_363352

theorem complex_number_powers (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 = -1 :=
sorry

end complex_number_powers_l363_363352


namespace correlation_regression_properties_l363_363227

theorem correlation_regression_properties
  (n : ℕ)
  (x y : Fin n → ℝ)
  (x_bar y_bar : ℝ)
  (r : ℝ)
  (b a : ℝ)
  (h_correlation : r = -0.8)
  (h_regression : ∀ i, y i = b * x i + a)
  (h_x_bar : x_bar = (Finset.univ.sum (λ i, x i)) / n)
  (h_y_bar : y_bar = (Finset.univ.sum (λ i, y i)) / n) :
  (y_bar = b * x_bar + a)
  ∧ (abs r > 0.75)
  ∧ (b < 0) :=
by
  sorry

end correlation_regression_properties_l363_363227


namespace quadratic_real_root_iff_b_range_l363_363586

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l363_363586


namespace smallest_value_of_A_plus_B_minus_C_l363_363477

theorem smallest_value_of_A_plus_B_minus_C :
  ∃ (A B C : ℕ),
    (A / 100) ≠ (A / 10 % 10) ∧ (A / 10 % 10) ≠ (A % 10) ∧ (A % 10) ≠ (A / 100) ∧
    (B / 100) ≠ (B / 10 % 10) ∧ (B / 10 % 10) ≠ (B % 10) ∧ (B % 10) ≠ (B / 100) ∧
    (C / 100) ≠ (C / 10 % 10) ∧ (C / 10 % 10) ≠ (C % 10) ∧ (C % 10) ≠ (C / 100) ∧
    List.card (List.nodup [A / 100, A / 10 % 10, A % 10, B / 100, B / 10 % 10, B % 10, C / 100, C / 10 % 10, C % 10]
              (A / 100 = 5 ∨ A / 100 = 6 ∨ A / 100 = 7 ∨ A / 100 = 8 ∨ A / 100 = 9) ∧
              (A / 10 % 10 = 5 ∨ A / 10 % 10 = 6 ∨ A / 10 % 10 = 7 ∨ A / 10 % 10 = 8 ∨ A / 10 % 10 = 9) ∧
              (A % 10 = 5 ∨ A % 10 = 6 ∨ A % 10 = 7 ∨ A % 10 = 8 ∨ A % 10 = 9) ∧
              (B / 100 = 5 ∨ B / 100 = 6 ∨ B / 100 = 7 ∨ B / 100 = 8 ∨ B / 100 = 9) ∧
              (B / 10 % 10 = 5 ∨ B / 10 % 10 = 6 ∨ B / 10 % 10 = 7 ∨ B / 10 % 10 = 8 ∨ B / 10 % 10 = 9) ∧
              (B % 10 = 5 ∨ B % 10 = 6 ∨ B % 10 = 7 ∨ B % 10 = 8 ∨ B % 10 = 9) ∧
              (C / 100 = 5 ∨ C / 100 = 6 ∨ C / 100 = 7 ∨ C / 100 = 8 ∨ C / 100 = 9) ∧
              (C / 10 % 10 = 5 ∨ C / 10 % 10 = 6 ∨ C / 10 % 10 = 7 ∨ C / 10 % 10 = 8 ∨ C / 10 % 10 = 9) ∧
              (C % 10 = 5 ∨ C % 10 = 6 ∨ C % 10 = 7 ∨ C % 10 = 8 ∨ C % 10 = 9)) = 9 ∧
    A + B - C = 149 := sorry

end smallest_value_of_A_plus_B_minus_C_l363_363477


namespace point_on_line_l363_363236

theorem point_on_line : ∀ (x y : ℝ), (x = 2 ∧ y = 7) → (y = 3 * x + 1) := 
by 
  intros x y h
  cases h with hx hy
  rw [hx, hy]
  sorry

end point_on_line_l363_363236


namespace quadratic_real_roots_l363_363560

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l363_363560


namespace part1_part2_l363_363181

def f (x a : ℝ) : ℝ := abs (x + a + 1) + abs (x - 4 / a)

theorem part1 (a x : ℝ) (h : a > 0) : f x a ≥ 5 :=
by sorry

theorem part2 (a : ℝ) (h : a > 0) (h1 : f 1 a < 6) : 1 < a ∧ a < 4 :=
by sorry

end part1_part2_l363_363181


namespace cos_identity_l363_363045

variable (θ : ℝ)
variable (h : cos (π / 6 + θ) = sqrt 3 / 3)

theorem cos_identity :
  cos (5 * π / 6 - θ) = - sqrt 3 / 3 :=
by
  sorry

end cos_identity_l363_363045


namespace apples_distribution_second_has_thirteen_l363_363274

-- Define the sequence in which Dasha places apples into the baskets.
def sequence := [1, 2, 3, 2, 1, 2, 3, 2]

-- Define a function that simulates the distribution of apples in the baskets.
def distribute_apples (steps : ℕ) : ℕ × ℕ × ℕ :=
  let cycle_len := 4
  let cycles := steps / cycle_len
  let remainder := steps % cycle_len
  let apples_second := cycles * 2
  let apples_first := cycles * 2 + if remainder > 0 then 1 else 0
  let apples_third := cycles * 2 + if remainder >= 2 then 1 else 0
  (apples_first, apples_second, apples_third)

-- Statement: Given the second basket has 13 apples, prove that the first basket has more apples than the third basket.
theorem apples_distribution_second_has_thirteen :
  ∃ (steps : ℕ), distribute_apples steps = (13, 13, 12) → 
  ∃ (steps : ℕ), (fst (distribute_apples steps) > snd (snd (distribute_apples steps))) :=
by
  sorry

end apples_distribution_second_has_thirteen_l363_363274


namespace problem1_range_of_f_problem2_min_value_problem3_nonexist_m_n_l363_363498

theorem problem1_range_of_f (a : ℝ) (x : ℝ) (h : a = 1 ∧ x ∈ set.Icc 0 1) : 
  set.range (λ x, 9^x - 2 * a * 3^x + 3) = set.Icc 2 6 := sorry

theorem problem2_min_value (a : ℝ) (x : ℝ) (h : x ∈ set.Icc (-1) 1) : 
  ∃ h_a : ℝ, h_a = 
    if a < 1/3 then (28/9 - 2*a/3) 
    else if 1/3 ≤ a ∧ a ≤ 3 then (3 - a^2) 
    else (12 - 6*a) := sorry

theorem problem3_nonexist_m_n : 
  ¬(∃ (m n : ℝ), n > m ∧ m > 3 ∧ 
     (∀ x, m ≤ x ∧ x ≤ n → (9^x - 2 * 3^x + 3) ∈ set.Icc (m^2) (n^2))) := sorry

end problem1_range_of_f_problem2_min_value_problem3_nonexist_m_n_l363_363498


namespace overall_gain_percent_l363_363700

-- Conditions:
def purchase1 := 900
def repair1 := 300
def sale1 := 1320

def purchase2 := 1100
def repair2 := 400
def sale2 := 1620

def purchase3 := 1200
def repair3 := 500
def sale3 := 1880

def total_cost := purchase1 + repair1 + purchase2 + repair2 + purchase3 + repair3
def total_selling_price := sale1 + sale2 + sale3
def gain := total_selling_price - total_cost
def gain_percent := (gain * 100) / total_cost

-- Question (Proving the Gain Percent):
theorem overall_gain_percent : gain_percent = 9.55 := by
  sorry

end overall_gain_percent_l363_363700


namespace find_k_l363_363460

def vector := prod ℝ ℝ

def m : vector := (1, 0)
def n : vector := (1, 1)

def perpendicular (v1 v2 : vector) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_k (k : ℝ) :
  perpendicular (m + (k • n)) m → k = -1 :=
sorry

end find_k_l363_363460


namespace prove_true_statement_l363_363319

-- Definitions based on conditions in the problem
def A_statement := ∀ x : ℝ, x = 2 → (x - 2) * (x - 1) = 0

-- Equivalent proof problem in Lean 4
theorem prove_true_statement : A_statement :=
by
  sorry

end prove_true_statement_l363_363319


namespace sam_container_marble_count_l363_363446

-- Definitions based on conditions
def ellie_dimensions := (2, 3, 4) : ℕ × ℕ × ℕ
def ellie_marble_capacity : ℕ := 200
def sam_scaling_factors := (3, 2, 1) : ℕ × ℕ × ℕ

-- Volume calculation utility
def volume (dims : ℕ × ℕ × ℕ) : ℕ :=
  dims.1 * dims.2 * dims.3

-- Calculate dimensions of Sam's container
def sam_dimensions (dims : ℕ × ℕ × ℕ) (scales : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  (dims.1 * scales.1, dims.2 * scales.2, dims.3 * scales.3)

-- Proof statement
theorem sam_container_marble_count : 
  volume ellie_dimensions = 24 →
  volume (sam_dimensions ellie_dimensions sam_scaling_factors) = 144 →
  (volume (sam_dimensions ellie_dimensions sam_scaling_factors)) / (volume ellie_dimensions) * ellie_marble_capacity = 1200 :=
by
  intros h1 h2
  simp only [ellie_dimensions, ellie_marble_capacity, sam_scaling_factors, volume, sam_dimensions] at h1 h2
  sorry

end sam_container_marble_count_l363_363446


namespace evaluate_expression_l363_363427

theorem evaluate_expression :
  (∃ (a b c : ℕ), a = 18 ∧ b = 3 ∧ c = 54 ∧ c = a * b ∧ (18^36 / 54^18) = (6^18)) :=
sorry

end evaluate_expression_l363_363427


namespace max_cities_condition_l363_363120

theorem max_cities_condition (num_cities : ℕ)
  (H1 : ∀ {a b : ℕ}, a < num_cities → b < num_cities → a ≠ b → ∃ (mode : ℕ), mode ∈ {0, 1, 2} ∧ (∀ {c : ℕ}, c < num_cities → c ≠ a → c ≠ b → mode ∉ {m | ∃ (a' b' : ℕ), a' < num_cities ∧ b' < num_cities ∧ a' ≠ b' ∧ m = if a = a' ∧ b = b' then mode else 3}))
  (H2 : ∀ {a : ℕ}, a < num_cities → ∃ (modes : finset ℕ), modes ⊆ {0, 1, 2} ∧ modes.card < 3)
  (H3 : ∀ {a b c : ℕ}, a < num_cities → b < num_cities → c < num_cities → a ≠ b → b ≠ c → a ≠ c → ∀ (mode : ℕ), mode ∉ {m | a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ m ∈ {x | ∃ (a' b' : ℕ), a' < num_cities ∧ b' < num_cities ∧ a' ≠ b' ∧ x = if a = a' ∧ b = b' then mode else 3}}) :
  num_cities ≤ 4 :=
sorry

end max_cities_condition_l363_363120


namespace cone_height_nearest_whole_number_l363_363356

noncomputable def cone_height (r V : ℝ) : ℝ := (3 * V) / (π * r^2)

theorem cone_height_nearest_whole_number :
  cone_height 3 93 ≈ 10 :=
by
  -- Simplify calculation and show it rounds to 10
  sorry

end cone_height_nearest_whole_number_l363_363356


namespace percentage_error_is_64_l363_363327

-- Definitions based on conditions
def original_number (x : ℝ) := x
def correct_calculation (x : ℝ) := x * (5 / 3)
def incorrect_calculation (x : ℝ) := x * (3 / 5)
def percentage_error (x : ℝ) := 
  (((correct_calculation x) - (incorrect_calculation x)) / (correct_calculation x)) * 100

-- Theorem to prove the percentage error is 64%
theorem percentage_error_is_64 (x : ℝ) : percentage_error x = 64 := by
  sorry

end percentage_error_is_64_l363_363327


namespace imply_count_is_four_l363_363858

theorem imply_count_is_four 
  (p q r : Prop)
  (s1 : p ∧ ¬ q ∧ ¬ r)
  (s2 : ¬ p ∧ ¬ q ∧ ¬ r)
  (s3 : p ∧ q ∧ ¬ r)
  (s4 : ¬ p ∧ q ∧ ¬ r) : 
  (if (p ∧ ¬ q ∧ ¬ r) then ((p → q) → ¬ r) else false) ∧
  (if (¬ p ∧ ¬ q ∧ ¬ r) then ((p → q) → ¬ r) else false) ∧
  (if (p ∧ q ∧ ¬ r) then ((p → q) → ¬ r) else false) ∧
  (if (¬ p ∧ q ∧ ¬ r) then ((p → q) → ¬ r) else false) ↔ (4 = 4) :=
by
  sorry

end imply_count_is_four_l363_363858


namespace sum_of_valid_two_digit_numbers_l363_363418

def is_valid_two_digit_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
               ((a + b ∣ 10 * a + b) ∧ (a^2 * b ∣ 10 * a + b))

theorem sum_of_valid_two_digit_numbers : 
  (∑ n in finset.filter is_valid_two_digit_number (finset.range 100)) = 0 :=
by
  sorry

end sum_of_valid_two_digit_numbers_l363_363418


namespace fly_passes_through_5_6_6_6_l363_363817

def probability_fly_through_5_6 : Prop :=
  let pathsTo5_6 := nat.choose 11 5
  let paths6_6_to_8_10 := nat.choose 6 2
  (pathsTo5_6 * paths6_6_to_8_10) / (2^18) = 26 / 1000

/-- The probability that the fly will be at the point (8, 10) passing through (5, 6) -> (6, 6) -/
theorem fly_passes_through_5_6_6_6 : probability_fly_through_5_6 :=
  sorry

end fly_passes_through_5_6_6_6_l363_363817


namespace real_solution_count_l363_363666

/-- Given \( \lfloor x \rfloor \) is the greatest integer less than or equal to \( x \),
prove that the number of real solutions to the equation \( 9x^2 - 36\lfloor x \rfloor + 20 = 0 \) is 2. --/
theorem real_solution_count (x : ℝ) (h : ⌊x⌋ = Int.floor x) :
  ∃ (S : Finset ℝ), S.card = 2 ∧ ∀ a ∈ S, 9 * a^2 - 36 * ⌊a⌋ + 20 = 0 :=
sorry

end real_solution_count_l363_363666


namespace program_output_is_44_l363_363387

theorem program_output_is_44 (choices : list ℕ) (output : ℕ) (H : choices = [42, 43, 44, 45]) : output = 44 :=
sorry -- proof omitted

end program_output_is_44_l363_363387


namespace values_of_A_l363_363455

theorem values_of_A (α : ℝ) (k : ℤ) :
  let A := (sin (k * Real.pi + α)) / (sin α) + (cos (k * Real.pi + α)) / (cos α)
  in A = 2 ∨ A = -2 :=
by
  sorry

end values_of_A_l363_363455


namespace compute_expression_l363_363856

theorem compute_expression : 12 * (1 / 15) * 30 = 24 := 
by 
  sorry

end compute_expression_l363_363856


namespace space_per_bookshelf_l363_363655

-- Defining the conditions
def S_room : ℕ := 400
def S_reserved : ℕ := 160
def n_shelves : ℕ := 3

-- Theorem statement
theorem space_per_bookshelf (S_room S_reserved n_shelves : ℕ)
  (h1 : S_room = 400) (h2 : S_reserved = 160) (h3 : n_shelves = 3) :
  (S_room - S_reserved) / n_shelves = 80 :=
by
  -- Placeholder for the proof
  sorry

end space_per_bookshelf_l363_363655


namespace rationalize_fraction_l363_363207

-- Define the conditions
def sqrt50_eq_5_sqrt2 : Prop := Real.sqrt 50 = 5 * Real.sqrt 2
def sqrt32_eq_4_sqrt2 : Prop := Real.sqrt 32 = 4 * Real.sqrt 2
def sqrt18_eq_9_sqrt2 : Prop := 3 * Real.sqrt 18 = 9 * Real.sqrt 2

-- State the theorem
theorem rationalize_fraction :
    sqrt50_eq_5_sqrt2 →
    sqrt32_eq_4_sqrt2 →
    sqrt18_eq_9_sqrt2 →
    (5 / (Real.sqrt 50 + Real.sqrt 32 + 3 * Real.sqrt 18) = 5 * Real.sqrt 2 / 36) :=
by
    intros h50 h32 h18
    simp [h50, h32, h18]
    sorry

end rationalize_fraction_l363_363207


namespace drew_correct_questions_l363_363621

-- Define properties
def Drew_wrong := 6
def Carla_correct := 14
def Carla_wrong := 2 * Drew_wrong
def Total_questions := 52

-- Define the proof problem
theorem drew_correct_questions : ∃ D : ℕ, D + Drew_wrong + (Carla_correct + Carla_wrong) = Total_questions ∧ D = 20 := 
by
  have h1: Drew_wrong = 6 := rfl
  have h2: Carla_correct = 14 := rfl
  have h3: Carla_wrong = 12 := by simp [Drew_wrong]
  have h4: Total_correct_questions := 52 := rfl
  use 20
  split
  have : 20 + 6 + (14 + 12) = 52 := by simp
  assumption
  rfl

end drew_correct_questions_l363_363621


namespace farmer_land_l363_363795

-- Define A to be the total land owned by the farmer
variables (A : ℝ)

-- Define the conditions of the problem
def condition_1 (A : ℝ) : ℝ := 0.90 * A
def condition_2 (cleared_land : ℝ) : ℝ := 0.20 * cleared_land
def condition_3 (cleared_land : ℝ) : ℝ := 0.70 * cleared_land
def condition_4 (cleared_land : ℝ) : ℝ := cleared_land - condition_2 cleared_land - condition_3 cleared_land

-- Define the assertion we need to prove
theorem farmer_land (h : condition_4 (condition_1 A) = 630) : A = 7000 :=
by
  sorry

end farmer_land_l363_363795


namespace estimate_points_in_interval_l363_363122

-- Define the conditions
def total_data_points : ℕ := 1000
def frequency_interval : ℝ := 0.16
def interval_estimation : ℝ := total_data_points * frequency_interval

-- Lean theorem statement
theorem estimate_points_in_interval : interval_estimation = 160 :=
by
  sorry

end estimate_points_in_interval_l363_363122


namespace gcd_288_123_l363_363767

theorem gcd_288_123 : gcd 288 123 = 3 :=
by
  sorry

end gcd_288_123_l363_363767


namespace find_digit_l363_363230

theorem find_digit (A : ℕ) (hA : A ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  9 ∣ (3 * 1000 + A * 110 + 1) ↔ A = 7 :=
by
  sorry

end find_digit_l363_363230


namespace sample_size_is_200_l363_363812
-- Define the total number of students and the number of students surveyed
def total_students : ℕ := 3600
def students_surveyed : ℕ := 200

-- Define the sample size
def sample_size := students_surveyed

-- Prove the sample size is 200
theorem sample_size_is_200 : sample_size = 200 :=
by
  -- Placeholder for the actual proof
  sorry

end sample_size_is_200_l363_363812


namespace probability_of_winning_pair_l363_363363

/-- Definition of the deck and conditions --/
def deck : Finset (Σ (c : Fin 3), Fin 4) :=
  { (0, 0), (0, 1), (0, 2), (0, 3), -- red cards
    (1, 0), (1, 1), (1, 2), (1, 3), -- green cards
    (2, 0), (2, 1), (2, 2), (2, 3)  -- blue cards
  }

/-- A pair of cards is winning if they have the same color or the same label --/
def winning_pair (a b : (Σ (c : Fin 3), Fin 4)) : Prop :=
  a.1 = b.1 ∨ a.2 = b.2

/-- The theorem to prove - The probability of drawing a winning pair is 5/11 --/
theorem probability_of_winning_pair :
  ((Finset.card ((deck.product deck).filter (λ pair, winning_pair pair.1 pair.2)) : ℚ) /
   (Finset.card (deck.product deck) : ℚ)) = 5 / 11 :=
by
  -- Outline logic goes here (proof skipped with sorry)
  sorry

end probability_of_winning_pair_l363_363363


namespace workshops_participation_l363_363754

variable (x y z a b c d : ℕ)
variable (A B C : Finset ℕ)

theorem workshops_participation:
  (A.card = 15) →
  (B.card = 14) →
  (C.card = 11) →
  (25 = x + y + z + a + b + c + d) →
  (12 = a + b + c + d) →
  (A.card = x + a + c + d) →
  (B.card = y + a + b + d) →
  (C.card = z + b + c + d) →
  d = 0 :=
by
  intro hA hB hC hTotal hAtLeastTwo hAkA hBkA hCkA
  -- The proof will go here
  -- Parsing these inputs shall lead to establishing d = 0
  sorry

end workshops_participation_l363_363754


namespace problem1_problem2_problem3_l363_363445

section
  variable (g : ℝ → ℝ) (f : ℝ → ℝ)

  -- sine-odd function g(x)
  def sine_odd_function (g : ℝ → ℝ) : Prop := ∀ x, sin (g (-x)) = -sin (g x)

  -- sine-odd monotonically increasing function f(x)
  def sine_odd_monotone_function (f : ℝ → ℝ) : Prop := 
    sine_odd_function f ∧ 
    (∀ a b, a < b → f a < f b) ∧ 
    (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧ 
    f 0 = 0

  -- Problem (1): necessary and sufficient condition for sin[g(x)] = 1 and sin[g(x)] = -1
  theorem problem1 (h : sine_odd_function g) (u₀ : ℝ) : 
    (sin (g u₀) = 1 ↔ sin (g (-u₀)) = -1) :=
  sorry

  -- Problem (2): find the value of a + b if f(a) = π/2 and f(b) = -π/2
  theorem problem2 (hf : sine_odd_monotone_function f) (a b : ℝ) (ha : f a = π / 2) (hb : f b = -π / 2) : 
    a + b = 0 :=
  sorry

  -- Problem (3): prove that f(x) is an odd function
  theorem problem3 (hf : sine_odd_monotone_function f) : 
    ∀ x, f (-x) = -f x :=
  sorry
end

end problem1_problem2_problem3_l363_363445


namespace sin_sum_identity_l363_363458

variable (α : ℝ)

-- Given condition
def given_condition : Prop :=
  cos (α + π / 6) - sin α = 4 * sqrt 3 / 5

-- To prove
theorem sin_sum_identity (h : given_condition α) : 
  sin (α + 11 * π / 6) = -4 / 5 := 
by 
  -- the proof would go here
  sorry

end sin_sum_identity_l363_363458


namespace max_digits_sum_digital_watch_l363_363193

theorem max_digits_sum_digital_watch : 
  (∀ (h m : ℕ), h < 24 → m < 60 → h.digits.sum + m.digits.sum ≤ 24) :=
by
  intros h m h_cond m_cond
  sorry

end max_digits_sum_digital_watch_l363_363193


namespace initial_sodium_chloride_percentage_l363_363834

theorem initial_sodium_chloride_percentage :
  ∀ (P : ℝ),
  (∃ (C : ℝ), C = 24) → -- Tank capacity
  (∃ (E_rate : ℝ), E_rate = 0.4) → -- Evaporation rate per hour
  (∃ (time : ℝ), time = 6) → -- Time in hours
  (1 / 4 * C = 6) → -- Volume of mixture
  (6 * P / 100 + (6 - 6 * P / 100 - E_rate * time) = 3.6) → -- Concentration condition
  P = 30 :=
by
  intros P hC hE_rate htime hvolume hconcentration
  rcases hC with ⟨C, hC⟩
  rcases hE_rate with ⟨E_rate, hE_rate⟩
  rcases htime with ⟨time, htime⟩
  rw [hC, hE_rate, htime] at *
  sorry

end initial_sodium_chloride_percentage_l363_363834


namespace interest_percentage_calculation_l363_363686

-- Definitions based on problem conditions
def purchase_price : ℝ := 110
def down_payment : ℝ := 10
def monthly_payment : ℝ := 10
def number_of_monthly_payments : ℕ := 12

-- Theorem statement:
theorem interest_percentage_calculation :
  let total_paid := down_payment + (monthly_payment * number_of_monthly_payments)
  let interest_paid := total_paid - purchase_price
  let interest_percent := (interest_paid / purchase_price) * 100
  interest_percent = 18.2 :=
by sorry

end interest_percentage_calculation_l363_363686


namespace ellipse_equation_line_pq_fixed_point_max_abs_diff_l363_363473

noncomputable def ellipse_eq : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (eccentricity = (k: ℝ) where (k = (sqrt 3 / 2)))
  ∧ (2 * b = 2) ∧ (a = 2) ∧ (b = 1)

theorem ellipse_equation (a b k1 k2 : ℝ) (h1: a > b) (h2: b > 0) (h3: sqrt 3 / 2 = sqrt (a^2 - b^2) / a) (h4: a * b = 2) :
  ∃ (a : ℝ), 
    (4 = a) ∧ (y : ℝ), y^2 = 1 ∧ (x : ℝ), x^2 / 4 + y^2 = 1 :=
sorry

theorem line_pq_fixed_point (k1 k2 : ℝ) (h1: 3 * k1 = 5 * k2) (h2 : x = ty - 1/2) :
  ∃ (M : ℝ × ℝ), M = (-1/2, 0) :=
sorry

theorem max_abs_diff (t : ℝ) :
  ∃ (S1 S2 : ℝ), 
    max_abs_diff = |(S1 - S2)| (max := (sqrt 15 / 4)) :=
sorry

end ellipse_equation_line_pq_fixed_point_max_abs_diff_l363_363473


namespace machines_work_together_l363_363759
noncomputable theory

open Real

theorem machines_work_together (y : ℝ) 
  (h₁ : ∃ (P Q R : Type*), P = y + 4 ∧ Q = y + 2 ∧ R = y ^ 2 )
  (h₂ : 1 / (y + 4) + 1 / (y + 2) + 1 / (y ^ 2) = 1 / y) : 
  y = ( -1 + sqrt 5 ) / 2 :=
sorry

end machines_work_together_l363_363759


namespace intersection_point_l363_363742

noncomputable def g (x : ℝ) (c : ℝ) := 4 * x + c

theorem intersection_point (c d : ℤ) (h₁ : g (-2) c = d) (h₂ : (∃ f, ∀ x, f (g x c) = x) ∧ ∀ x, (g x c = -2) → x = d) : d = -2 :=
by
  sorry

end intersection_point_l363_363742


namespace radical_axis_two_circles_radical_center_three_circles_l363_363329

-- Conditions for part (a)
def Circle1 (x y a1 b1 R1 : Real) : Prop :=
  (x - a1)^2 + (y - b1)^2 = R1^2

def Circle2 (x y a2 b2 R2 : Real) : Prop :=
  (x - a2)^2 + (y - b2)^2 = R2^2

-- Part (a): Radical Axis of Two Circles
theorem radical_axis_two_circles (a1 b1 a2 b2 R1 R2 x y : Real)
  (h1 : Circle1 x y a1 b1 R1) (h2 : Circle2 x y a2 b2 R2)
  (h_diff_centers : (a1, b1) ≠ (a2, b2)) :
  2 * (a2 - a1) * x + 2 * (b2 - b1) * y + R2^2 - R1^2 = 0 :=
sorry

-- Conditions for part (b)
def Circle (O : (Real × Real)) (R : Real) :=
  ∃ (x y : Real), (x - O.1)^2 + (y - O.2)^2 = R^2

-- Part (b): Radical Center of Three Circles
theorem radical_center_three_circles 
  (O1 O2 O3 : (Real × Real)) 
  (R1 R2 R3 : Real) 
  (h1 : Circle O1 R1) 
  (h2 : Circle O2 R2) 
  (h3 : Circle O3 R3) 
  (h_noncollinear : ¬ collinear [O1, O2, O3]) :
  ∃ S : (Real × Real), 
    (∃ (x y : Real), 2 * (O2.1 - O1.1) * x + 2 * (O2.2 - O1.2) * y + R2^2 - R1^2 = 0) ∧
    (∃ (x y : Real), 2 * (O3.1 - O2.1) * x + 2 * (O3.2 - O2.2) * y + R3^2 - R2^2 = 0) ∧
    (∃ (x y : Real), 2 * (O3.1 - O1.1) * x + 2 * (O3.2 - O1.2) * y + R3^2 - R1^2 = 0) :=
sorry

end radical_axis_two_circles_radical_center_three_circles_l363_363329


namespace troy_needs_additional_money_l363_363278

-- Defining the initial conditions
def price_of_new_computer : ℕ := 80
def initial_savings : ℕ := 50
def money_from_selling_old_computer : ℕ := 20

-- Defining the question and expected answer
def required_additional_money : ℕ :=
  price_of_new_computer - (initial_savings + money_from_selling_old_computer)

-- The proof statement
theorem troy_needs_additional_money : required_additional_money = 10 := by
  sorry

end troy_needs_additional_money_l363_363278


namespace largest_consecutive_odd_numbers_l363_363220

theorem largest_consecutive_odd_numbers (x : ℤ)
  (h : (x + (x + 2) + (x + 4) + (x + 6)) / 4 = 24) : 
  x + 6 = 27 :=
  sorry

end largest_consecutive_odd_numbers_l363_363220


namespace find_cost_per_batch_l363_363360

noncomputable def cost_per_tire : ℝ := 8
noncomputable def selling_price_per_tire : ℝ := 20
noncomputable def profit_per_tire : ℝ := 10.5
noncomputable def number_of_tires : ℕ := 15000

noncomputable def total_cost (C : ℝ) : ℝ := C + cost_per_tire * number_of_tires
noncomputable def total_revenue : ℝ := selling_price_per_tire * number_of_tires
noncomputable def total_profit : ℝ := profit_per_tire * number_of_tires

theorem find_cost_per_batch (C : ℝ) :
  total_profit = total_revenue - total_cost C → C = 22500 := by
  sorry

end find_cost_per_batch_l363_363360


namespace geometric_series_S_n_div_a_n_l363_363079

-- Define the conditions and the properties of the geometric sequence
variables (a_3 a_5 a_4 a_6 S_n a_n : ℝ) (n : ℕ)
variable (q : ℝ) -- common ratio of the geometric sequence

-- Conditions given in the problem
axiom h1 : a_3 + a_5 = 5 / 4
axiom h2 : a_4 + a_6 = 5 / 8

-- The value we want to prove
theorem geometric_series_S_n_div_a_n : 
  (a_3 + a_5) * q = 5 / 8 → 
  q = 1 / 2 → 
  S_n = a_n * (2^n - 1) :=
by
  intros h1 h2
  sorry

end geometric_series_S_n_div_a_n_l363_363079


namespace find_a_l363_363088

noncomputable def f (a x : ℝ) : ℝ := -x^2 + a * x - (a / 4) + (1 / 2)

theorem find_a (a : ℝ) (h : ∃ (x ∈ set.Icc 0 1), f a x = 2) : a = -6 ∨ a = 10 / 3 :=
begin
  sorry
end

end find_a_l363_363088


namespace solve_for_x_l363_363013

theorem solve_for_x: ∃ x : ℝ, 5^x * 25^(2*x) = 625^3 ∧ x = 12 / 5 :=
by {
  -- We prove the existence of such an x satisfying both the equation and the solution.
  sorry
}

end solve_for_x_l363_363013


namespace find_x_in_sequence_l363_363256

def seq (a : ℕ → ℕ) : Prop :=
  a 0 = 1 ∧ a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 3 ∧ a 5 = 8 ∧ a 6 = 13 ∧ a 7 = 21 ∧ 
  ∀ n ≥ 1, a (n + 2) = a (n + 1) + a n

theorem find_x_in_sequence : 
  ∃ x : ℕ, seq (λ n, if n = 4 then x else 
                if n = 0 then 1 else 
                if n = 1 then 1 else 
                if n = 2 then 2 else 
                if n = 3 then 3 else 
                if n = 5 then 8 else 
                if n = 6 then 13 else 
                if n = 7 then 21 else 0) ∧ x = 5 := 
by
  existsi 5
  -- Proof goes here, but for now we use sorry
  sorry

end find_x_in_sequence_l363_363256


namespace eval_expression_l363_363426

theorem eval_expression (x y z : ℝ) (hx : x = 1/3) (hy : y = 2/3) (hz : z = -9) :
  x^2 * y^3 * z = -8/27 :=
by
  subst hx
  subst hy
  subst hz
  sorry

end eval_expression_l363_363426


namespace find_t_l363_363422

theorem find_t (c o u n t s : ℕ)
    (hc : c ≠ 0) (ho : o ≠ 0) (hn : n ≠ 0) (ht : t ≠ 0) (hs : s ≠ 0)
    (h1 : c + o = u)
    (h2 : u + n = t + 1)
    (h3 : t + c = s)
    (h4 : o + n + s = 15) :
    t = 7 := 
sorry

end find_t_l363_363422


namespace solve_y_l363_363310

theorem solve_y (y : ℚ) (h : (3 * y) / 7 = 14) : y = 98 / 3 := 
by sorry

end solve_y_l363_363310


namespace perpendicular_vectors_l363_363098

theorem perpendicular_vectors (x y : ℝ) (a : ℝ × ℝ := (1, 2)) (b : ℝ × ℝ := (2 + x, 1 - y)) 
  (hperp : (a.1 * b.1 + a.2 * b.2 = 0)) : 2 * y - x = 4 :=
sorry

end perpendicular_vectors_l363_363098


namespace quadratic_real_root_iff_b_range_l363_363580

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l363_363580


namespace range_of_x_l363_363929

variable (f : ℝ → ℝ)

def even_function :=
  ∀ x : ℝ, f (-x) = f x

def monotonically_decreasing :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x

def f_value_at_2 := f 2 = 0

theorem range_of_x (h1 : even_function f) (h2 : monotonically_decreasing f) (h3 : f_value_at_2 f) :
  { x : ℝ | f (x - 1) > 0 } = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end range_of_x_l363_363929


namespace complement_A_complement_B_intersection_A_B_complement_union_A_B_l363_363095

def U := set ℝ
def A := {x : ℝ | x < -2 ∨ x > 5}
def B := {x : ℝ | 4 ≤ x ∧ x ≤ 6}

-- Complement of A in universal set U
theorem complement_A : set.compl A = {x : ℝ | -2 ≤ x ∧ x ≤ 5} :=
sorry

-- Complement of B in universal set U
theorem complement_B : set.compl B = {x : ℝ | x < 4 ∨ x > 6} :=
sorry

-- Intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 5 < x ∧ x ≤ 6} :=
sorry

-- Complement of (A union B) in universal set U
theorem complement_union_A_B : set.compl (A ∪ B) = {x : ℝ | -2 ≤ x ∧ x < 4} :=
sorry

end complement_A_complement_B_intersection_A_B_complement_union_A_B_l363_363095


namespace slope_range_of_line_l363_363467

-- Define points A, B, and P
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨-2, -3⟩
def B : Point := ⟨3, 0⟩
def P : Point := ⟨-1, 2⟩

-- Define slopes between points
def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

def k_AP : ℝ := slope A P
def k_BP : ℝ := slope B P

-- The main statement
theorem slope_range_of_line (k : ℝ) (l : Point → Prop) :
  (l P) ∧ (∃ M, ∀ x y, l (Point.mk x y) ↔ y = k * x + M) ∧
  (∃ t ∈ Icc 0 1, l (Point.mk (t * A.x + (1 - t) * B.x) (t * A.y + (1 - t) * B.y))) →
  k ≥ 5 ∨ k ≤ -1/2 := 
sorry

end slope_range_of_line_l363_363467


namespace limit_T_n_l363_363140

noncomputable def seq_a (n : ℕ) : ℝ :=
  if n = 1 then 1 else sorry

noncomputable def S_n (n : ℕ) : ℝ :=
  if n = 1 then 1 else 1 / (2 * n - 1)

noncomputable def b_n (n : ℕ) : ℝ :=
  S_n n / (2 * n + 1)

noncomputable def T_n (n : ℕ) : ℝ :=
  ∑ i in finset.range n, b_n (i + 1)

theorem limit_T_n :
  filter.tendsto T_n filter.at_top (nhds (1 / 2)) :=
sorry

end limit_T_n_l363_363140


namespace hyperbola_asymptote_focus_equation_l363_363129

theorem hyperbola_asymptote_focus_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ (x y : ℝ), y = x / (real.sqrt 3) ∨ y = -x / (real.sqrt 3) → let k := real.sqrt (1 + (real.sqrt 3)^2) in ∀ c : ℝ, abs c / k = 2 ↔ c = 4) ∧
  ∃ (h : 0 < real.sqrt 12 ∧ 0 < real.sqrt 4), 
  ((x : ℝ) (y : ℝ), 
    x ^ 2 / 12 - y ^ 2 / 4 = 1) :=
sorry

end hyperbola_asymptote_focus_equation_l363_363129


namespace segments_have_common_point_l363_363264

-- Define the predicate that checks if two segments intersect
def segments_intersect (seg1 seg2 : ℝ × ℝ) : Prop :=
  let (a1, b1) := seg1
  let (a2, b2) := seg2
  max a1 a2 ≤ min b1 b2

-- Define the main theorem
theorem segments_have_common_point (segments : Fin 2019 → ℝ × ℝ)
  (h_intersect : ∀ (i j : Fin 2019), i ≠ j → segments_intersect (segments i) (segments j)) :
  ∃ p : ℝ, ∀ i : Fin 2019, (segments i).1 ≤ p ∧ p ≤ (segments i).2 :=
by
  sorry

end segments_have_common_point_l363_363264


namespace identify_participants_with_questions_l363_363980

-- Definitions based on the problem conditions
variable (k : ℕ) -- total participants

-- chemists and alchemists properties
-- chemists always tell the truth
-- alchemists sometimes tell the truth, sometimes lie

-- The number of chemists is greater than the number of alchemists
variable (C A : ℕ) -- C: number of chemists, A: number of alchemists

-- Conditions from the problem
axiom chemists_greater : C > A
axiom total_participants : C + A = k

-- Statement to prove
theorem identify_participants_with_questions :
  ∃ (q : ℕ), q ≤ 2 * k - 3 ∧ (∀ (p : ℕ), p < k → (chemist p ∨ alchemist p)) :=
sorry

end identify_participants_with_questions_l363_363980


namespace quadratic_real_root_iff_b_range_l363_363583

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l363_363583


namespace ratio_PQ_AQ_l363_363532

section
variable (Q P A C : Point)
variable (AB CD : Line)
variable [Circle Q]
variable [Diameter Q AB] [Diameter Q CD]
variable [Perpendicular AB CD]
variable [LiesOn P AQ] (angle_QPC : ∠ Q P C = 45°)

theorem ratio_PQ_AQ : PQ / AQ = √2 / 2 := by
  sorry
end

end ratio_PQ_AQ_l363_363532


namespace greatest_distance_between_vertices_l363_363830

theorem greatest_distance_between_vertices (inner_perimeter outer_perimeter : ℝ) (h₁ : inner_perimeter = 20) (h₂ : outer_perimeter = 28) :
  ∃ d : ℝ, d = sqrt 65 :=
by
  sorry

end greatest_distance_between_vertices_l363_363830


namespace find_varphi_l363_363938

theorem find_varphi (φ : ℝ) (h1 : 0 < φ ∧ φ < 2 * Real.pi) 
    (h2 : ∀ x, x = 2 → Real.sin (Real.pi * x + φ) = 1) : 
    φ = Real.pi / 2 :=
-- The following is left as a proof placeholder
sorry

end find_varphi_l363_363938


namespace area_of_plywood_l363_363370

theorem area_of_plywood (width length : ℕ) (h_width : width = 6) (h_length : length = 4) : width * length = 24 :=
by
  rw [h_width, h_length]
  norm_num

end area_of_plywood_l363_363370


namespace construct_right_triangle_l363_363860

theorem construct_right_triangle (c : ℝ) :
  ∃ (a b : ℝ), a^2 + b^2 = c^2 ∧ (√(a * b) = c / 2) := by
  sorry

end construct_right_triangle_l363_363860


namespace quadratic_real_roots_l363_363557

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l363_363557


namespace theta_in_quad_2_or_3_l363_363900

theorem theta_in_quad_2_or_3 (θ : ℝ) (h : Real.cos θ = - 1 / 3) : 
  (∃ k : ℤ, θ = 2 * k * Real.pi + Real.pi - Real.arccos(-1 / 3)) ∨ 
  (∃ k : ℤ, θ = 2 * k * Real.pi + Real.pi + Real.arccos(-1 / 3)) := 
sorry

end theta_in_quad_2_or_3_l363_363900


namespace AF_div_AT_l363_363144

-- Variables for the points
variables {A B C D E F G : Type}

-- Definitions of distances
def AD := 2
def DB := 2
def AE := 3
def EC := 3

-- Given proportionality
-- BG : GC = 1 : 3
axiom BG_GC_ratio : ∀ (G : Type), ratio (segmentLength B G) (segmentLength G C) = 1 / 3

-- Prove the ratio AF/AT
theorem AF_div_AT (A B C D E F T : Type) 
  (hAD : distance A D = 2)
  (hDB : distance D B = 2)
  (hAE : distance A E = 3)
  (hEC : distance E C = 3)
  (hAT : angleBisector A T B C)
  (hIntersection : intersects T D E F) : 
  ratio (segmentLength A F) (segmentLength A T) = 1 / 3 := 
sorry

end AF_div_AT_l363_363144


namespace product_xyz_w_l363_363793

variables (x y z w : ℚ)

-- Conditions
def condition1 := 3 * x + 4 * y = 60
def condition2 := 6 * x - 4 * y = 12
def condition3 := 2 * x - 3 * z = 38
def condition4 := x + y + z = w

-- Proof statement
theorem product_xyz_w (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) :
  x * y * z * w = -5104 := by
  sorry

end product_xyz_w_l363_363793


namespace integral_partial_fraction_l363_363023

open Real

theorem integral_partial_fraction :
  ∫ x in 1..2, (1 / (x * (x + 1))) = ln (4 / 3) := by
  sorry

end integral_partial_fraction_l363_363023


namespace find_polynomials_l363_363434

def polynomial_satisfies_eq (P : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, P(x) + P(y) + P(z) + P(x + y + z) = P(x + y) + P(y + z) + P(z + x)

theorem find_polynomials (P : ℝ → ℝ) [polynomial_with_real_coeffs P] :
  polynomial_satisfies_eq P →
  (∃ (a b : ℝ), P = λ x, a * x + b) ∨ (∃ (a b : ℝ), P = λ x, a * x^2 + b * x) :=
sorry

end find_polynomials_l363_363434


namespace substitution_result_l363_363719

theorem substitution_result (x y : ℝ) (h1 : y = 2 * x + 1) (h2 : 5 * x - 2 * y = 7) : 5 * x - 4 * x - 2 = 7 :=
by
  sorry

end substitution_result_l363_363719


namespace base_salary_l363_363358

theorem base_salary {B : ℝ} {C : ℝ} :
  (B + 200 * C = 2000) → 
  (B + 200 * 15 = 4000) → 
  B = 1000 :=
by
  sorry

end base_salary_l363_363358


namespace total_tents_l363_363188

theorem total_tents (n_east n_center n_south : ℕ) :
  (n_east = 2 * 100) → (n_center = 4 * 100) → (n_south = 200) →
  100 + n_east + n_center + n_south = 900 :=
by
  intros h_east h_center h_south
  rw [h_east, h_center, h_south]
  norm_num

end total_tents_l363_363188


namespace mn_necessary_not_sufficient_l363_363481

variable (m n : ℝ)

def is_ellipse (m n : ℝ) : Prop := 
  (m > 0) ∧ (n > 0) ∧ (m ≠ n)

theorem mn_necessary_not_sufficient : (mn > 0) → (is_ellipse m n) ↔ false := 
by sorry

end mn_necessary_not_sufficient_l363_363481


namespace smallest_w_l363_363970

theorem smallest_w (w : ℕ) (w_pos : w > 0) (h1 : ∀ n : ℕ, 2^4 ∣ 1452 * w)
                              (h2 : ∀ n : ℕ, 3^3 ∣ 1452 * w)
                              (h3 : ∀ n : ℕ, 13^3 ∣ 1452 * w) :
  w = 676 := sorry

end smallest_w_l363_363970


namespace magnitude_of_z_l363_363933

-- Definitions of the problem's conditions
def complex_equation (z : ℂ) : Prop :=
  z + 2 * complex.conj z = 9 + 4 * complex.I

-- The main Lean statement to prove
theorem magnitude_of_z (z : ℂ) (h : complex_equation z) : complex.abs z = 5 :=
sorry

end magnitude_of_z_l363_363933


namespace find_value_of_y_l363_363308

theorem find_value_of_y (y : ℚ) (h : 3 * y / 7 = 14) : y = 98 / 3 := 
by
  /- Proof to be completed -/
  sorry

end find_value_of_y_l363_363308


namespace part_a_solution_part_b_solution_l363_363716

-- Part (a) proof problem
theorem part_a_solution (x : ℝ) (hx : abs(x) < 1) : 
  (2 * x + 1 + x^2 - x^3 + x^4 - x^5 + ...) = 13 / 6 :=
sorry

-- Part (b) proof problem
theorem part_b_solution (x : ℝ) (hx : abs(x) < 1) : 
  (1 / x + x + x^2 + x^3 + ...) = 7 / 2 :=
sorry

end part_a_solution_part_b_solution_l363_363716


namespace find_vector_at_t4_l363_363367

noncomputable def vector_at_t (a d : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (a.1 + t * d.1, a.2 + t * d.2, a.3 + t * d.3)

noncomputable def r1 := (2, 4, 9) : ℝ × ℝ × ℝ
noncomputable def r3 := (1, 1, 2) : ℝ × ℝ × ℝ
noncomputable def r4_expected := (0.5, -0.5, -1.5) : ℝ × ℝ × ℝ

theorem find_vector_at_t4 : 
  ∃ a d, vector_at_t a d 1 = r1 ∧ vector_at_t a d 3 = r3 ∧ vector_at_t a d 4 = r4_expected :=
sorry

end find_vector_at_t4_l363_363367


namespace system_of_DE_sol_l363_363212

open Real
noncomputable section

def x (t : ℝ) (C1 C2 : ℝ) : ℝ := C1 * exp (-t) + 2 * C2 * exp (2 * t) - cos t + 3 * sin t
def y (t : ℝ) (C1 C2 : ℝ) : ℝ := -C1 * exp (-t) + C2 * exp (2 * t) + 2 * cos t - sin t

theorem system_of_DE_sol (C1 C2 : ℝ) :
  (∀ t : ℝ, deriv (λ t, x t C1 C2) t = x t C1 C2 + 2 * y t C1 C2) ∧
  (∀ t : ℝ, deriv (λ t, y t C1 C2) t = x t C1 C2 - 5 * sin t) :=
  by 
    sorry

end system_of_DE_sol_l363_363212


namespace prime_product_probability_zero_l363_363295

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def spinner1 := {2, 4, 6, 8}
def spinner2 := {3, 5, 7, 9, 11}

theorem prime_product_probability_zero :
  (∃ x ∈ spinner1, ∃ y ∈ spinner2, is_prime (x * y)) = false :=
by
  sorry

end prime_product_probability_zero_l363_363295


namespace correct_proportion_l363_363782

-- Definitions of proportions
def is_proportion (a b c d : ℚ) : Prop := (a / b) = (c / d)

-- Prove that only 3:2 can form a proportion with 1/2:1/3
theorem correct_proportion :
  is_proportion (1/2) (1/3) (3) (2) ∧ 
  ¬ is_proportion (1/2) (1/3) (5) (4) ∧
  ¬ is_proportion (1/2) (1/3) (1/3) (1/4) ∧
  ¬ is_proportion (1/2) (1/3) (1/3) (1/2) :=
by
  -- The proof will go here
  sorry

end correct_proportion_l363_363782


namespace prove_range_of_a_l363_363478

noncomputable def problem_statement (x y a : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ x + y + 3 = x * y ∧ 
  ∀ t, (t = x + y) → t ∈ set.Ici (6 : ℝ) → (t^2 - a * t + 1 ≥ 0) → (a ≤ 37 / 6)

theorem prove_range_of_a :
  ∀ x y a : ℝ, problem_statement x y a → a ≤ 37 / 6 :=
begin
  sorry
end

end prove_range_of_a_l363_363478


namespace quadratic_has_real_root_iff_b_in_interval_l363_363545

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l363_363545


namespace find_integer_pairs_l363_363433

noncomputable def satisfies_equation (x y : ℤ) :=
  12 * x ^ 2 + 6 * x * y + 3 * y ^ 2 = 28 * (x + y)

theorem find_integer_pairs (m n : ℤ) :
  satisfies_equation (3 * m - 4 * n) (4 * n) :=
sorry

end find_integer_pairs_l363_363433


namespace coefficient_x3_y3_expansion_l363_363636

theorem coefficient_x3_y3_expansion (x y : ℝ) :
  (x + y^2 / x) * (x + y)^5 = 15 * x^3 * y^3 + (other terms) :=
sorry

end coefficient_x3_y3_expansion_l363_363636


namespace hyperbola_eccentricity_l363_363511

theorem hyperbola_eccentricity (m a b : ℝ) (h1 : m ≠ 0) (h2 : a > 0) (h3 : b > 0)
  (P : ℝ × ℝ := (m, 0))
  (A B : ℝ × ℝ)
  (hA1 : A = (ma / (3*b - a), mb / (3*b - a)))
  (hB1 : B = (-ma / (3*b + a), mb / (3*b + a)))
  (hP1 : |((m, 0): ℝ × ℝ) - A| = |((m, 0): ℝ × ℝ) - B|) :
  let e := sqrt (5 * (b^2)) / (2 * b)
  in e = sqrt(5) / 2 := by
  sorry

end hyperbola_eccentricity_l363_363511


namespace find_value_of_y_l363_363309

theorem find_value_of_y (y : ℚ) (h : 3 * y / 7 = 14) : y = 98 / 3 := 
by
  /- Proof to be completed -/
  sorry

end find_value_of_y_l363_363309


namespace locus_of_intersection_l363_363215

theorem locus_of_intersection 
  (a b : ℝ)
  (l m : ℝ → set (ℝ × ℝ))
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : ∀ k k' : ℝ, l = {p | p.2 - k * p.1 + k * a = 0} ∧ m = {p | p.2 + k' * p.1 - k' * b = 0})
  (h4 : ∃ (k k' : ℝ), ∀ (y x : ℝ), (y^2 = x) ∧ ((y - k * x + k * a = 0) ∨ (y + k' * x - k' * b = 0)))
  (h5 : ∃ (P : ℝ × ℝ), intersect l m P ∧ concyclic P (y^2 = x))
  : P.1 = (a + b) / 2 := by
  sorry

end locus_of_intersection_l363_363215


namespace three_mathematicians_napping_simultaneously_l363_363619

theorem three_mathematicians_napping_simultaneously :
  ∀ (n : ℕ) (m : ℕ),
  n = 5 → m = 2 →
  (∀ (i j : ℕ), i ≠ j → ∃ t : ℚ, (i, t) ∈ (finset.range n).product (finset.range m) ∧ (j, t) ∈ (finset.range n).product (finset.range m)) →
  ∃ t : ℚ, ∃ s : finset ℕ, s.card ≥ 3 ∧ ∀ i ∈ s, (i, t) ∈ (finset.range n).product (finset.range m) :=
begin
  sorry
end

end three_mathematicians_napping_simultaneously_l363_363619


namespace normal_distribution_probability_l363_363676

theorem normal_distribution_probability (σ : ℝ) (hσ : σ > 0) :
  ∀ (ζ : ℝ → Type) [normal_distribution ζ 4 σ] (P : set ℝ → ℝ),
  (P {x | 4 < x ∧ x < 8} = 0.3) →
  (P {x | x < 0} = 0.2) :=
by
  assume ζ hζ P hP,
  sorry

end normal_distribution_probability_l363_363676


namespace jerry_set_off_firecrackers_l363_363150

theorem jerry_set_off_firecrackers :
  ∀ (n : ℕ), n = 48 → 
    ∃ m : ℕ, m = 12 → 
      ∃ k : ℕ, k = n - m → 
        ∃ d : ℕ, d = k / 6 → 
          ∃ g : ℕ, g = k - d → 
            ∃ s : ℕ, s = g / 2 → 
              s = 15 :=
by 
  intros n hn m hm k hk d hd g hg s hs
  -- This line starts inserting the proof steps, which we leave to Lean to fill.
  sorry

end jerry_set_off_firecrackers_l363_363150


namespace complex_modulus_eq_l363_363891
open Complex

theorem complex_modulus_eq : ∃ n > 0, (|4 + (n : ℂ)*I| = 4 * Real.sqrt 13 ↔ n = 8 * Real.sqrt 3) :=
by
  sorry

end complex_modulus_eq_l363_363891


namespace investment_in_business_l363_363332

theorem investment_in_business (Q : ℕ) (P : ℕ) 
  (h1 : Q = 65000) 
  (h2 : 4 * Q = 5 * P) : 
  P = 52000 :=
by
  rw [h1] at h2
  linarith

end investment_in_business_l363_363332


namespace proof_problem_l363_363313

noncomputable def cos75_squared : ℝ :=
  cos 75 * cos 75

noncomputable def optionA : Prop :=
  cos75_squared = (2 - Real.sqrt 3) / 4

noncomputable def optionB : Prop :=
  (1 + tan 105) / (1 - tan 105) ≠ Real.sqrt 3 / 3

noncomputable def optionC : Prop :=
  tan 1 + tan 44 + tan 1 * tan 44 = 1

noncomputable def optionD : Prop :=
  sin 70 * ((Real.sqrt 3) / tan 40 - 1) ≠ 2

theorem proof_problem : optionA ∧ optionC ∧ optionB ∧ optionD := by
  sorry

end proof_problem_l363_363313


namespace sine_angle_BDC_l363_363611

noncomputable def triangle_problem : ℝ :=
let A := (0, 0) in  -- Assumption: A at origin
let B := (2, 0) in  -- Per condition, length of AB is 2
let C := (0, 2) in  -- Per condition, length of AC is 2
let E := ((B.1 + C.1) / 2, (B.2 + C.2) / 2) in  -- E midway between B and C
let D := (0, 1) in  -- Assuming D is mid-point of AC
let BD := Real.sqrt ((B.1 - D.1) ^ 2 + (B.2 - D.2) ^ 2) in  -- Pythagorean theorem
let DC := Real.sqrt ((D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2) in 
let angle_DEC := 30 * Real.pi / 180 in  -- Radians conversion for angle DEC
Real.sin (real.angle_of_2d_vectors (B - D) (C - D)) 

theorem sine_angle_BDC : triangle_problem = Real.sqrt(13) / 2 := 
sorry

end sine_angle_BDC_l363_363611


namespace range_of_m_l363_363942

def proposition_p (m : ℝ) : Prop := (m^2 - 4 ≥ 0)
def proposition_q (m : ℝ) : Prop := (4 - 4 * m < 0)
def p_or_q (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def not_p (m : ℝ) : Prop := ¬ proposition_p m

theorem range_of_m (m : ℝ) (h1 : p_or_q m) (h2 : not_p m) : 1 < m ∧ m < 2 :=
sorry

end range_of_m_l363_363942


namespace trapezium_area_l363_363029

theorem trapezium_area (a b h : ℝ) (h_a : a = 20) (h_b : b = 18) (h_h : h = 10) : 
  (1 / 2) * (a + b) * h = 190 := 
by
  -- We provide the conditions:
  rw [h_a, h_b, h_h]
  -- The proof steps will be skipped using 'sorry'
  sorry

end trapezium_area_l363_363029


namespace trigonometric_identity_l363_363486

noncomputable def tan_sum (alpha : ℝ) : Prop :=
  Real.tan (alpha + Real.pi / 4) = 2

noncomputable def trigonometric_expression (alpha : ℝ) : ℝ :=
  (Real.sin alpha + 2 * Real.cos alpha) / (Real.sin alpha - 2 * Real.cos alpha)

theorem trigonometric_identity (alpha : ℝ) (h : tan_sum alpha) : 
  trigonometric_expression alpha = -7 / 5 :=
sorry

end trigonometric_identity_l363_363486


namespace part_I_part_II_l363_363089

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (x a : ℝ) : ℝ := (f x a) + (g x)

theorem part_I (a : ℝ) :
  (∀ x > 0, f x a ≥ g x) → a ≤ 0.5 :=
by
  sorry

theorem part_II (a x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) 
  (hx1_lt_half : x1 < 0.5) :
  (h x1 a = 2 * x1^2 + Real.log x1) →
  (h x2 a = 2 * x2^2 + Real.log x2) →
  (x1 * x2 = 0.5) →
  h x1 a - h x2 a > (3 / 4) - Real.log 2 :=
by
  sorry

end part_I_part_II_l363_363089


namespace part_I_part_II_part_III_l363_363863

section part_I
variables {R : Type*} [linear_ordered_field R]
def op : R → R → R := λ x y, logb 10 (10^x + 10^y)

theorem part_I (a b c : R) : 
  (op a b - c) = op (a - c) (b - c) := 
sorry
end part_I

section part_II
variables {R : Type*} [linear_ordered_field R]
def op : R → R → R := λ x y, logb 10 (10^x + 10^y)

theorem part_II (a : R) : 
  ∃ x : set R, (∀ x ∈ x, x.to_real ∈ (x-1)^2 > (op (a^2 * x^2) (a^2 * x^2) - logb 10 2)) ∧ 
  cardinal.mk x = 3 ↔ a ∈ (-3/2, -4/3] ∪ [4/3, 3/2) := 
sorry
end part_II

section part_III
variables {R : Type*} [linear_ordered_field R]
def op : R → R → R := λ x y, logb 10 (10^x + 10^y)
def f (x : R) : R := logb 10 ((op (x + 4) (x + 4)) - sqrt (2 * x + 3) - logb 10 2)
def g (x : R) : R := op 1 x ⨁ op (-x)

theorem part_III (m : R) : 
  (∀ x1 : R, ∃ x2 ∈ set.Ici (-3/2: R), 
    g x1 = logb 10 (abs(3 * m - 2)) + f x2) ↔ 
    m ∈ (-4/3, 2/3) ∪ (2/3, 8/3) := 
sorry
end part_III

end part_I_part_II_part_III_l363_363863


namespace substitution_result_l363_363720

theorem substitution_result (x y : ℝ) (h1 : y = 2 * x + 1) (h2 : 5 * x - 2 * y = 7) : 5 * x - 4 * x - 2 = 7 :=
by
  sorry

end substitution_result_l363_363720


namespace machine_subtract_l363_363960

theorem machine_subtract (x : ℤ) (h1 : 26 + 15 - x = 35) : x = 6 :=
by
  sorry

end machine_subtract_l363_363960


namespace point_labeling_in_right_triangle_l363_363471

theorem point_labeling_in_right_triangle
    (A B C : Point) 
    (hC : angle C = 90)
    (n : ℕ)
    (P : fin n → Point) 
    (h_in_triangle : ∀ i, in_triangle P i A B C):
  ∃ (label : fin n → Point), 
    sum (λ i, dist2 (label i) (label (i + 1))) (range (n - 1)) ≤ dist2 A B := 
begin
  sorry
end

end point_labeling_in_right_triangle_l363_363471


namespace total_scheduling_methods_l363_363840

theorem total_scheduling_methods (staff : Finset ℕ) (A B : ℕ) (days : Finset ℕ) 
(h_staff : staff.card = 7) 
(h_days : days = {1, 2, 3, 4, 5, 6, 7}) 
(h_exclude_A_B_1st_2nd : ∀ d ∈ {1, 2}, A ∉ staff ∧ B ∉ staff) :
  ∃ (n : ℕ), n = 2400 := 
by
  existsi 2400
  sorry

end total_scheduling_methods_l363_363840


namespace min_value_in_interval_l363_363245

noncomputable def f (x : ℝ) : ℝ := (1/2)^x

theorem min_value_in_interval : ∃ x, x ∈ Icc (-2:ℝ) (-1:ℝ) ∧ is_min_on f (Icc (-2) (-1)) x ∧ f x = 2 :=
by
  sorry

end min_value_in_interval_l363_363245


namespace blocks_used_in_shed_l363_363698

theorem blocks_used_in_shed
  (shed_length shed_width shed_height : ℕ)
  (wall_thickness  block_volume : ℕ) :
  shed_length = 15 →
  shed_width = 12 →
  shed_height = 7 →
  wall_thickness = 3/2 →
  block_volume = 1 →
  (let
      shed_volume := shed_length * shed_width * shed_height,
      interior_length := shed_length - 2 * wall_thickness,
      interior_width := shed_width - 2 * wall_thickness,
      interior_height := shed_height - 2 * wall_thickness,
      interior_volume := interior_length * interior_width * interior_height,
      blocks_used := shed_volume - interior_volume
    in blocks_used = 828) :=
by
  intros
  sorry

end blocks_used_in_shed_l363_363698


namespace median_score_interval_l363_363859

theorem median_score_interval :
  -- Conditions describing the student scores and their intervals
  let num_students := 100
  let scores : List (ℕ × ℕ) := [
    (60, 64),
    (65, 69),
    (70, 74),
    (75, 79),
    (80, 84)
  ]
  let score_counts : List ℕ := [
    20, 20, 20, 20, 20
  ]
  (∑ count in score_counts, count) = num_students →
  -- There are exactly 100 students.
  -- Prove that the median score falls in the interval 70-74
  (∑ count in score_counts.take 3, count) ≤ 51 ∧ 51 ≤ (∑ count in score_counts.take 4, count) →
  (70 ≤ 71) ∧ (71 ≤ 74) :=
sorry

end median_score_interval_l363_363859


namespace discount_percentage_approx_l363_363184

noncomputable def cost_price : ℝ := 540
noncomputable def selling_price : ℝ := 456
noncomputable def marked_price : ℝ := cost_price + (15 / 100 * cost_price)
noncomputable def discount : ℝ := marked_price - selling_price
noncomputable def discount_percentage : ℝ := (discount / marked_price) * 100

theorem discount_percentage_approx : discount_percentage ≈ 26.57 := 
by
  sorry

end discount_percentage_approx_l363_363184


namespace vector_addition_l363_363514

variable (a : ℝ × ℝ)
variable (b : ℝ × ℝ)

theorem vector_addition (h1 : a = (-1, 2)) (h2 : b = (1, 0)) :
  3 • a + b = (-2, 6) :=
by
  -- proof goes here
  sorry

end vector_addition_l363_363514


namespace angle_ACB_45_l363_363761

noncomputable def Triangle (A B C : Type) := sorry

variable {A B C D E F : Type}

variables
  (AB AC: ℝ) (D : Type) (E : Type)
  (F : Type)
  (h1 : AB = 3 * AC)
  (h2 : ∃ (P1 : Type) (P2 : Type), (P1, P2, P1, P2) = (D, E, D, E))

def isosceles (T : Type) (x y : Type) := sorry

def isIntersection (A B C D : Type) := sorry

def sameAngle (A B : Type) := sorry

theorem angle_ACB_45:
  AB = 3 * AC →
  (∃ D E, sameAngle D E) →
  isIntersection A D E F →
  isosceles (Triangle C F E) (angle C F E) (angle F E C) →
  True := 
by
  intro h1 h2 h3 h4
  have : angle_ACB = 45 -- Define angle_precisely here
  exact sorry

end angle_ACB_45_l363_363761


namespace cubes_sum_eq_l363_363105

noncomputable def x (a : ℝ) : ℝ := sorry
noncomputable def y (b : ℝ) : ℝ := sorry
noncomputable def z (c : ℝ) : ℝ := sorry

theorem cubes_sum_eq (a b c : ℝ) (hx : x a ^ 2 = a) (hy : y b ^ 2 = b) (hz : z c ^ 2 = c) :
  (x a) ^ 3 + (y b) ^ 3 + (z c) ^ 3 = 
  (a^(3/2) + b^(3/2) + c^(3/2)) ∨ 
  (-(a^(3/2) + b^(3/2) + c^(3/2))) :=
sorry

end cubes_sum_eq_l363_363105


namespace area_of_triangle_ABC_rect_l363_363628

theorem area_of_triangle_ABC_rect (A B C D : Type) [EuclideanGeometry A B C D]
  (h: is_rectangle A B C D)
  (hAC: AC = 10)
  (hAB: AB = 8) :
  area_triangle ABC = 24 :=
sorry

end area_of_triangle_ABC_rect_l363_363628


namespace insurance_calculation_l363_363350

-- Define the conditions
def baseRate : ℝ := 0.002 -- 0.2% as a decimal
def reducingCoefficient : ℝ := 0.8
def increasingCoefficient : ℝ := 1.3
def assessedValue : ℝ := 14500000
def cadasterValue : ℝ := 15000000
def loanAmount : ℝ := 20000000

-- Define the adjusted tariff
def adjustedTariff : ℝ := baseRate * reducingCoefficient * increasingCoefficient

-- Define the insurable amount
def insurableAmount : ℝ := max assessedValue cadasterValue

-- Define the proof problem to show equality
theorem insurance_calculation :
  adjustedTariff = 0.00208 ∧ insurableAmount * adjustedTariff = 31200 := by
  sorry

end insurance_calculation_l363_363350


namespace smallest_perfect_square_divisible_by_2_and_5_l363_363777

theorem smallest_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℕ, n = k ^ 2) ∧ (n % 2 = 0) ∧ (n % 5 = 0) ∧ 
  (∀ m : ℕ, 0 < m ∧ (∃ k : ℕ, m = k ^ 2) ∧ (m % 2 = 0) ∧ (m % 5 = 0) → n ≤ m) :=
sorry

end smallest_perfect_square_divisible_by_2_and_5_l363_363777


namespace molly_next_two_flips_prob_l363_363680

theorem molly_next_two_flips_prob : 
  (let P := (1 / 2 : ℝ) in
   (P * P + P * P) = 1 / 2) :=
by
  let P := (1 / 2 : ℝ)
  calc
    P * P + P * P = (1 / 2) * (1 / 2) + (1 / 2) * (1 / 2) : by rfl
               ... = 1 / 4 + 1 / 4                   : by rfl
               ... = 1 / 2                           : by norm_num

end molly_next_two_flips_prob_l363_363680


namespace radius_of_circumscribed_circle_l363_363117

noncomputable def triangle_radius (α : ℝ) (AB : ℝ) (area : ℝ) : ℝ :=
  if h : α = 2 * Real.pi / 3 ∧ AB = 4 ∧ area = 2 * Real.sqrt 3 then
    let b := 2 in -- From area equation
    let a := 2 * Real.sqrt 7 in
    let R := (2 * Real.sqrt 21) / 3 in
    R
  else
    0 -- This else part ensures totality of the function but is not relevant for the problem

theorem radius_of_circumscribed_circle :
  triangle_radius (2 * Real.pi / 3) 4 (2 * Real.sqrt 3) = (2 * Real.sqrt 21) / 3 :=
by 
  unfold triangle_radius
  simp only [eq_self_iff_true, and_self, if_true]
  sorry -- Proof is omitted

end radius_of_circumscribed_circle_l363_363117


namespace insurance_premium_correct_l363_363345

noncomputable def compute_insurance_premium (loan_amount appraisal_value cadastral_value : ℕ)
  (basic_rate : ℚ) (reducing_coefficient increasing_coefficient : ℚ) : ℚ :=
let insured_amount := max appraisal_value cadastral_value in
let adjusted_tariff := basic_rate * reducing_coefficient * increasing_coefficient in
insured_amount * adjusted_tariff

theorem insurance_premium_correct :
  compute_insurance_premium 20000000 14500000 15000000 0.002 0.8 1.3 = 31200 :=
by
  sorry

end insurance_premium_correct_l363_363345


namespace paintable_fence_l363_363952

theorem paintable_fence :
  ∃ h t u : ℕ,  h > 1 ∧ t > 1 ∧ u > 1 ∧ 
  (∀ n, 4 + (n * h) ≠ 5 + (m * (2 * t))) ∧
  (∀ n, 4 + (n * h) ≠ 6 + (l * (3 * u))) ∧ 
  (∀ m l, 5 + (m * (2 * t)) ≠ 6 + (l * (3 * u))) ∧
  (100 * h + 20 * t + 2 * u = 390) :=
by 
  sorry

end paintable_fence_l363_363952


namespace magnitude_eq_l363_363894

theorem magnitude_eq : ∀ (n : ℝ), |4 + n * complex.I| = 4 * real.sqrt 13 → n = 8 * real.sqrt 3 := by
  intro n
  intro h
  sorry

end magnitude_eq_l363_363894


namespace smallest_n_for_P_lt_l363_363875

noncomputable def P (n : ℕ) : ℝ :=
  1 / (n * (n + 1))

theorem smallest_n_for_P_lt (n : ℕ) : 1 ≤ n ∧ n < 2010 → P(n) < 1 / 2010 ↔ n = 45 :=
by
  intros h1 h2
  sorry

end smallest_n_for_P_lt_l363_363875


namespace magnitude_eq_l363_363893

theorem magnitude_eq : ∀ (n : ℝ), |4 + n * complex.I| = 4 * real.sqrt 13 → n = 8 * real.sqrt 3 := by
  intro n
  intro h
  sorry

end magnitude_eq_l363_363893


namespace min_value_x_plus_y_l363_363052

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : y + 9 * x = x * y) :
  x + y ≥ 16 :=
by
  sorry

end min_value_x_plus_y_l363_363052


namespace find_variance_l363_363078

variables (μ σ : ℝ) (ξ : ℝ → ℝ) (X : ℝ) (p : ℝ) (n : ℕ)

-- Condition 1: ξ follows a normal distribution N(25.40, σ^2)
def normal_distribution (ξ : ℝ) : Prop := ξ ~ Normal μ (σ^2)

-- Condition 2: Probability P(ξ ≥ 25.45) = 0.1
def probability_exceeds (ξ : ℝ) (v : ℝ) : Prop := Probability ξ (≥ v) = 0.1

-- Condition 3: 3 components are randomly selected
def num_selected := n = 3

-- The probability of not being in (25.35, 25.45) interval
def out_of_range_prob := p = 0.2

-- The variance D(X) of the binomial distribution B(n, p)
def variance_DX := X ~ Binomial n p

def calculate_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Problem statement in Lean 4
theorem find_variance : 
  normal_distribution ξ → 
  probability_exceeds ξ 25.45 →
  num_selected →
  out_of_range_prob →
  calculate_variance 3 0.2 = 0.48 := 
by 
  sorry

end find_variance_l363_363078


namespace sum_powers_of_i_l363_363027

theorem sum_powers_of_i :
  (∑ k in Finset.range 2014, (k + 1) * (Complex.I ^ (k + 1))) = -1006 + 1005 * Complex.I :=
by
  sorry

end sum_powers_of_i_l363_363027


namespace magic_star_sum_is_26_l363_363432

noncomputable def magic_sum_of_6th_order_magic_star : ℕ :=
  let numbers := finset.range (12 + 1) -- Set of natural numbers from 1 to 12
  let total_sum := finset.sum numbers id -- Sum of numbers from 1 to 12
  let total_usage_sum := 2 * total_sum -- Since each number is used exactly twice
  total_usage_sum / 6 -- There are 6 lines in the 6th-order magic star

theorem magic_star_sum_is_26 :
  magic_sum_of_6th_order_magic_star = 26 :=
sorry  -- Proof omitted

end magic_star_sum_is_26_l363_363432


namespace problem_quadratic_has_real_root_l363_363600

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l363_363600


namespace parabola_eq_correct_line_eq_correct_l363_363077

-- Conditions
def hyperbola_vertex : (ℝ × ℝ) := (2, 0)
def parabola_focus : (ℝ × ℝ) := (2, 0)
def parabola_eq (p : ℝ) := sorry -- parabola equation definition
def line_eq (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Prove the equation of the parabola C1 is y^2 = 8x
theorem parabola_eq_correct : ∃ (p : ℝ), p = 4 ∧ parabola_eq p = sorry := 
sorry

-- Prove the equations of the line l are y = x + 1 and y = -5x + 1
theorem line_eq_correct (A B : ℝ × ℝ) (F : ℝ × ℝ) (h₁ : F = hyperbola_vertex)
  (h₂ : ∃ l : ℝ → ℝ, l = λ x, line_eq 1 x ∨ l = λ x, line_eq (-5) x) 
  (h₃ : ∃ (k : ℝ), k = 1 ∨ k = -5) :
  let k1 := (1 : ℝ), k2 := (-5 : ℝ) in
  (line_eq k1 = λ x, x + 1) ∧ (line_eq k2 = λ x, -5 * x + 1) := 
  sorry

end parabola_eq_correct_line_eq_correct_l363_363077


namespace square_area_equals_one_l363_363302

theorem square_area_equals_one (w : ℝ) (P_square P_rectangle : ℝ) (l : ℝ) (A_rectangle : ℝ) :
  l = 2 * w →
  A_rectangle = (8 : ℝ) / 9 →
  l * w = A_rectangle →
  P_rectangle = 2 * l + 2 * w →
  P_square = P_rectangle →
  ∃ s : ℝ, P_square = 4 * s ∧ s^2 = 1 := 
by
  intro h1 h2 h3 h4 h5
  have : s = 1, from sorry
  use 1
  simp [this]
  split
  · rw h5, rw h4, rw h1, rw h2, norm_num, norm_num
  · norm_num

end square_area_equals_one_l363_363302


namespace unique_extreme_point_iff_omega_in_range_l363_363608

noncomputable theory

-- Define the function f and its derivative
def f (ω x : ℝ) : ℝ := sin (ω * x) - sqrt 3 * cos (ω * x)
def f' (ω x : ℝ) : ℝ := derivative (λ x, f ω x)

-- Define the interval
def interval : set ℝ := set.Ioo (π / 6) (π / 2)

-- Define the range of omega
def omega_range : set ℝ := set.Icc (5 / 3) (11 / 3) ∪ set.Icc 5 (17 / 3)

-- Define the property that f has a unique extreme point in the interval
def unique_extreme_point_in_interval (ω : ℝ) : Prop :=
  ∃ c ∈ interval, ∀ x ∈ interval, f' ω x = 0

-- The final theorem statement
theorem unique_extreme_point_iff_omega_in_range :
  ∀ ω > 0, (unique_extreme_point_in_interval ω ↔ ω ∈ omega_range) :=
by
  sorry

end unique_extreme_point_iff_omega_in_range_l363_363608


namespace player_A_has_winning_strategy_l363_363293

/-- A game played on a rhombus board with special rules. -/
structure Game :=
  (n : ℕ)
  (board : Rhombus n) -- Assume Rhombus is a structure/function already defined
  (player_A_start : board.corner_60)
  (player_B_start : board.opposite_corner_60)
  (neighboring_cells : ∀ x : board.cells, set (board.cells))
  (move_token : board.cells → board.cells → Prop)

namespace Game
noncomputable def winning_strategy_A (game : Game) : Prop :=
  ∃ strategy : (ℕ → game.board.cells) → Bool, -- Step function explaining moves and outcomes
  (∀ steps : ℕ → game.board.cells, 
     strategy steps = true →
     (steps (4 * game.n - 4) = game.board.opposite_corner || steps (4 * game.n - 4) = game.player_B_start))

theorem player_A_has_winning_strategy (game : Game) : winning_strategy_A game :=
sorry
end Game

end player_A_has_winning_strategy_l363_363293


namespace number_of_true_propositions_l363_363090

noncomputable def f (x : ℝ) : ℝ := x^3
noncomputable def g (x : ℝ) : ℝ := 2^(1 - x)

def p : Prop := ∀ x, f x > f x ∧ g x > g x 
def q : Prop := ∃ x ∈ set.Ioo 0 2, f x - g x = 0

theorem number_of_true_propositions : 
    let P := ¬ p ∧ q in
    finset.card (finset.filter id {p ∧ q, p ∨ q, ¬ p ∧ q}) = 2 :=
by sorry

end number_of_true_propositions_l363_363090


namespace possible_values_a3_max_value_a64_l363_363057

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
a 1 = 0 ∧ ∀ n : ℕ, n > 0 → |a (n + 1) - a n| = n

theorem possible_values_a3 (a : ℕ → ℤ) (h : sequence a) :
  {a 3 | sequence a} = {-3, -1, 1, 3} := sorry

theorem max_value_a64 (a : ℕ → ℤ) (h : sequence a) :
  ∀ n : ℕ, n = 64 → a n ≤ 2016 := 
begin
  intros n hn,
  rw hn,
  suffices : ∀ n : ℕ, a n ≤ n * (n - 1) / 2,
  { exact this 64 },

  sorry
end


end possible_values_a3_max_value_a64_l363_363057


namespace equidistant_points_from_rays_l363_363948

open Set

-- Define the plane and points on the plane
variables (P : Type*) [MetricSpace P] [NormedAddTorsor ℝ₂ P]

-- Define the two rays on the plane
variables (r1 r2 : P → Prop)

-- Define the distance function from a point to the nearest point on a ray
def distance_to_ray {P : Type*} [MetricSpace P] (p : P) (ray : P → Prop) : ℝ := sorry

-- Formalization of the problem
theorem equidistant_points_from_rays (p : P) :
  let E := {x : P | distance_to_ray x r1 = distance_to_ray x r2} in
  ∃ C, 
  (C = ( -- Case 1: The endpoints of the rays do not coincide
          (∃ a1 a2 e1 e2 : P, ¬(e1 = e2) ∧
            (C = bisectors_of_angles a1 a2 ∪ perpendicular_bisector e1 e2 ∪ parabolas e1 e2)) ∨
          -- Case 2: The endpoints of the rays coincide
          (∃ e : P, 
            (C = angle_bisector e (r1 p) (r2 p) ∪ interior_region_within_angle e (r1 p) (r2 p))))
  sorry

end equidistant_points_from_rays_l363_363948


namespace ant_probability_after_10_minutes_l363_363395

-- Definitions based on the conditions given in the problem
def ant_start_at_A := true
def moves_each_minute (n : ℕ) := n == 10
def blue_dots (x y : ℤ) : Prop := 
  (x == 0 ∨ y == 0) ∧ (x + y) % 2 == 0
def A_at_center (x y : ℤ) : Prop := x == 0 ∧ y == 0
def B_north_of_A (x y : ℤ) : Prop := x == 0 ∧ y == 1

-- The probability we need to prove
def probability_ant_at_B_after_10_minutes := 1 / 9

-- We state our proof problem
theorem ant_probability_after_10_minutes :
  ant_start_at_A ∧ moves_each_minute 10 ∧ blue_dots 0 0 ∧ blue_dots 0 1 ∧ A_at_center 0 0 ∧ B_north_of_A 0 1
  → probability_ant_at_B_after_10_minutes = 1 / 9 := 
sorry

end ant_probability_after_10_minutes_l363_363395


namespace problem_statement_l363_363941

theorem problem_statement (m : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * x + m ≤ 0)) → m > 1 :=
by
  sorry

end problem_statement_l363_363941


namespace ratio_of_sums_l363_363946

noncomputable def sum_arith_prog (a d : ℕ) (n : ℕ) : ℕ :=
  n * a + n * (n - 1) * d / 2

noncomputable def sum_all_first_group : ℕ :=
  (∑ i in finset.range 1 16, sum_arith_prog i (2 * i) 10)

noncomputable def sum_all_second_group : ℕ :=
  (∑ i in finset.range 1 16, sum_arith_prog i (2 * i - 1) 10)

theorem ratio_of_sums : 
  (sum_all_first_group : ℚ) / sum_all_second_group = 160 / 151 :=
sorry

end ratio_of_sums_l363_363946


namespace linear_eq_find_m_l363_363493

theorem linear_eq_find_m (m : ℤ) (x : ℝ) 
  (h : (m - 5) * x^(|m| - 4) + 5 = 0) 
  (h_linear : |m| - 4 = 1) 
  (h_nonzero : m - 5 ≠ 0) : m = -5 :=
by
  sorry

end linear_eq_find_m_l363_363493


namespace quadratic_real_roots_l363_363558

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l363_363558


namespace triangle_perimeter_is_17_l363_363241

noncomputable def triangle_perimeter (x : ℝ) : ℝ :=
  if (x^2 - 8*x + 12 = 0) ∧ (7 - 4 < x) ∧ (x < 7 + 4) then 4 + 7 + x else 0

theorem triangle_perimeter_is_17 : ∃ x : ℝ, triangle_perimeter x = 17 :=
by {
  use 6,
  unfold triangle_perimeter,
  split_ifs,
  { simp,
    ring,
    sorry,
  },
  { sorry }
}

end triangle_perimeter_is_17_l363_363241


namespace value_of_a_l363_363609

theorem value_of_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y = 0 → 3 * x + y + a = 0) → a = 1 :=
by
  sorry

end value_of_a_l363_363609


namespace hexagon_angle_sum_l363_363135

theorem hexagon_angle_sum 
  (mA mB mC x y : ℝ)
  (hA : mA = 34)
  (hB : mB = 80)
  (hC : mC = 30)
  (hx' : x = 36 - y) : x + y = 36 :=
by
  sorry

end hexagon_angle_sum_l363_363135


namespace quadratic_roots_expression_l363_363909

theorem quadratic_roots_expression :
  ∀ (x₁ x₂ : ℝ), 
  (x₁ + x₂ = 3) →
  (x₁ * x₂ = -1) →
  (x₁^2 * x₂ + x₁ * x₂^2 = -3) :=
by
  intros x₁ x₂ h1 h2
  sorry

end quadratic_roots_expression_l363_363909


namespace rotated_graph_eq_l363_363238

-- Given the graph of y = log2 x
def graph_eq (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Prove that after rotating the graph by 180 degrees clockwise 
-- about the origin, the new equation is y = -log2(-x)
theorem rotated_graph_eq (x y : ℝ) (h : y = graph_eq x) : 
  (∀ x, y = graph_eq x → y = -graph_eq (-x)) :=
by
  intro x h
  rw [h]
  sorry

end rotated_graph_eq_l363_363238


namespace degree_of_polynomial_l363_363864

variable (a b c d e f : ℝ)

theorem degree_of_polynomial (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
                            (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) :
  degree ((X^5 + C a * X^8 + C b * X^2 + C c) * (X^4 + C d * X^3 + C e) * (X^2 + C f)) = 14 :=
by 
  sorry

end degree_of_polynomial_l363_363864


namespace problem_equiv_l363_363851

noncomputable def f (n : ℕ) : ℝ :=
  (n + 1)^3 / ((n - 1) * n) - (n - 1)^3 / (n * (n + 1))

theorem problem_equiv : (floor (f 1004)) = 8 :=
by {
  sorry
}

end problem_equiv_l363_363851


namespace prism_ratios_l363_363664

open EuclideanGeometry

-- Given a triangular prism ABC and its projected points A1, B1, C1
variable (A B C A1 B1 C1 M N K P Q : Point)

-- Definitions of the points involved and intersections
variables [Centroid M ABC]
variables [Intersection N (Diagonal AA1C1C)]
variables [Intersection K (Diagonal BB1C1C)]
variables [PlaneSection MNK (Line B1C1) P]
variables [PlaneSection MNK (Line CC1) Q]

-- Proof goal
theorem prism_ratios :
  (B1P / B1C1) = (2 / 3) ∧ (C1Q / CC1) = 1 :=
sorry

end prism_ratios_l363_363664


namespace find_new_x_value_l363_363459

section
variables (x y : ℕ)
hypothesis h1 : x / y = 6 / 3
hypothesis h2 : y = 15

theorem find_new_x_value : x + 10 = 40 :=
by
  sorry
end

end find_new_x_value_l363_363459


namespace area_PRQ_l363_363137

noncomputable def triangle_area (A B C : Point) : ℝ := sorry

noncomputable def is_equilateral (A B C : Point) : Prop := sorry

noncomputable def length (P Q : Point) : ℝ := sorry

variable {Point : Type*} -- Define Point as a Type

variable (A B C P Q R : Point)

-- Given conditions
axiom eq_triangle : is_equilateral A B C
axiom triangle_area_eq_one : triangle_area A B C = 1
axiom BP_one_third_BC : length B P = (1/3) * length B C
axiom AQ_eq_BQ : length A Q = length B Q
axiom PR_perp_AC : perp P R A C

-- Proof goal
theorem area_PRQ : triangle_area P R Q = 5 / 18 :=
sorry

end area_PRQ_l363_363137


namespace minimum_value_of_f_l363_363922

noncomputable def f (x : ℝ) : ℝ := sorry  -- define f such that f(x + 199) = 4x^2 + 4x + 3 for x ∈ ℝ

theorem minimum_value_of_f : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = 2 := by
  sorry  -- Prove that the minimum value of f(x) is 2

end minimum_value_of_f_l363_363922


namespace line_through_fixed_point_l363_363465

theorem line_through_fixed_point
  {segment : Segment}
  (chord : Chord segment)
  (circle1 circle2 : Circle)
  (H1 : circle1.touches_chord chord)
  (H2 : circle2.touches_chord chord)
  (H3 : circle1.touches_arc segment.arc)
  (H4 : circle2.touches_arc segment.arc)
  (M N : Point)
  (H5 : circle1.intersects circle2 M N) :
  passes_through_fixed_point (line_through M N) segment.arc.midpoint :=
sorry

end line_through_fixed_point_l363_363465


namespace softball_players_count_l363_363984

theorem softball_players_count :
  ∀ (cricket hockey football total_players softball : ℕ),
  cricket = 15 →
  hockey = 12 →
  football = 13 →
  total_players = 55 →
  total_players = cricket + hockey + football + softball →
  softball = 15 :=
by
  intros cricket hockey football total_players softball h_cricket h_hockey h_football h_total_players h_total
  sorry

end softball_players_count_l363_363984


namespace problem_f_increasing_l363_363083

theorem problem_f_increasing (a : ℝ) 
  (h1 : ∀ x, 2 ≤ x → 0 < x^2 - a * x + 3 * a) 
  (h2 : ∀ x, 2 ≤ x → 0 ≤ 2 * x - a) : 
  -4 < a ∧ a ≤ 4 := by
  sorry

end problem_f_increasing_l363_363083


namespace sum_of_intersection_points_l363_363747

-- Mathematical equivalence for Lean proof problem:
theorem sum_of_intersection_points :
  let x_roots : list ℝ := [x_1, x_2, x_3, x_4]
  let y_roots : list ℝ := [y_1, y_2, y_3, y_4]
  Σx_roots + Σy_roots = 0 :=
by
  sorry

end sum_of_intersection_points_l363_363747


namespace distribution_function_countable_discontinuities_l363_363706

-- Definitions based on problem conditions
def is_distribution_function (F : ℝ → ℝ) : Prop :=
  ∀ x ∈ ℝ, F (x - ϵ) < F x

-- The proof statement
theorem distribution_function_countable_discontinuities (F : ℝ → ℝ) 
    (hF: is_distribution_function F) : 
    ∃ S : set ℝ, S.countable ∧ ∀ x, F (x - ϵ) < F x → x ∈ S :=
  sorry

end distribution_function_countable_discontinuities_l363_363706


namespace insurance_premium_correct_l363_363346

noncomputable def compute_insurance_premium (loan_amount appraisal_value cadastral_value : ℕ)
  (basic_rate : ℚ) (reducing_coefficient increasing_coefficient : ℚ) : ℚ :=
let insured_amount := max appraisal_value cadastral_value in
let adjusted_tariff := basic_rate * reducing_coefficient * increasing_coefficient in
insured_amount * adjusted_tariff

theorem insurance_premium_correct :
  compute_insurance_premium 20000000 14500000 15000000 0.002 0.8 1.3 = 31200 :=
by
  sorry

end insurance_premium_correct_l363_363346


namespace evaluate_expression_l363_363022

theorem evaluate_expression :
  8^(-1/3 : ℝ) + (49^(-1/2 : ℝ))^(1/2 : ℝ) = (Real.sqrt 7 + 2) / (2 * Real.sqrt 7) := by
  sorry

end evaluate_expression_l363_363022


namespace increasing_iff_a_le_0_l363_363167

variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - a * x + 1

theorem increasing_iff_a_le_0 : (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 0 :=
by
  sorry

end increasing_iff_a_le_0_l363_363167


namespace inequality_x_y_l363_363920

theorem inequality_x_y (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2 * y^2 ≤ 3 := 
sorry

end inequality_x_y_l363_363920


namespace average_time_for_relay_race_l363_363799

noncomputable def average_leg_time (y_time z_time w_time x_time : ℕ) : ℚ :=
  (y_time + z_time + w_time + x_time) / 4

theorem average_time_for_relay_race :
  let y_time := 58
  let z_time := 26
  let w_time := 2 * z_time
  let x_time := 35
  average_leg_time y_time z_time w_time x_time = 42.75 := by
    sorry

end average_time_for_relay_race_l363_363799


namespace quadratic_has_real_root_l363_363569

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l363_363569


namespace lines_properties_l363_363512

theorem lines_properties (m : ℝ) :
  let l1 := {p : ℝ × ℝ | m * p.fst - p.snd + 2 = 0},
      l2 := {p : ℝ × ℝ | p.fst + m * p.snd + 2 = 0} in
  (∀ p1 p2, p1 ∈ l1 → p2 ∈ l2 → (p1.fst - p2.fst) * (p1.snd - p2.snd) = -1) ∧
  (0, 2) ∈ l1 ∧
  (-2, 0) ∈ l2 ∧
  (∃ M, M ∈ l1 ∧ M ∈ l2 ∧ |⟨0, 0⟩ - M| ≤ 2 * Real.sqrt 2) := sorry

end lines_properties_l363_363512


namespace positive_reals_power_equality_l363_363110

open Real

theorem positive_reals_power_equality (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : a < 1) : a = b := 
  by
  sorry

end positive_reals_power_equality_l363_363110


namespace troy_needs_additional_money_l363_363279

-- Defining the initial conditions
def price_of_new_computer : ℕ := 80
def initial_savings : ℕ := 50
def money_from_selling_old_computer : ℕ := 20

-- Defining the question and expected answer
def required_additional_money : ℕ :=
  price_of_new_computer - (initial_savings + money_from_selling_old_computer)

-- The proof statement
theorem troy_needs_additional_money : required_additional_money = 10 := by
  sorry

end troy_needs_additional_money_l363_363279


namespace find_b_values_l363_363553

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l363_363553


namespace volume_ratio_of_octahedrons_sum_l363_363620

-- Given: In a regular octahedron, the centers of the six faces form the vertices of a smaller octahedron.
-- Definitions
def regular_octahedron := sorry -- a regular octahedron definition

def face_center (v1 v2 v3 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((v1.1 + v2.1 + v3.1) / 3, (v1.2 + v2.2 + v3.2) / 3, (v1.3 + v2.3 + v3.3) / 3)

-- The volume ratio of the smaller octahedron to the larger one, expressed as m+n where m and n are coprime positive integers
theorem volume_ratio_of_octahedrons_sum (large small : regular_octahedron) 
  (h_centers: ∀ face, face ∈ large.faces → face_center face = small.vertices) :
  let ratio := 1 / 9 in
  let (m, n) := (1, 9) in
  m + n = 10 :=
by
  sorry

end volume_ratio_of_octahedrons_sum_l363_363620


namespace find_pencils_l363_363322

theorem find_pencils :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (6 ∣ n) ∧ (9 ∣ n) ∧ n % 7 = 1 ∧ n = 36 :=
by
  sorry

end find_pencils_l363_363322


namespace quadratic_real_roots_l363_363556

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l363_363556


namespace sufficient_but_not_necessary_condition_l363_363226

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, 0 < x → x < 4 → x^2 - 4 * x < 0) ∧ ¬ (∀ x : ℝ, x^2 - 4 * x < 0 → 0 < x ∧ x < 5) :=
sorry

end sufficient_but_not_necessary_condition_l363_363226


namespace valid_k_l363_363339

-- Define the deck of cards and the condition for a complete set
def deck := finset (finset (fin 10))

def is_complete_set (s : finset (finset (fin 10))) : Prop :=
  (⋃ x ∈ s, x) = finset.univ

-- Define the property that adding any card to the set forms a complete set
def adding_violates_condition (s : finset (finset (fin 10))) : Prop :=
  ∀ c ∈ (deck \ s), is_complete_set (s ∪ {c})

-- Prove that the valid number k is 512
theorem valid_k : ∀ s : finset (finset (fin 10)), (∃ k : ℕ, s.card = k ∧ ¬ is_complete_set s ∧ adding_violates_condition s) ↔ s.card = 512 := 
sorry

end valid_k_l363_363339


namespace shifts_needed_l363_363371

-- Given definitions
def total_workers : ℕ := 12
def workers_per_shift : ℕ := 2
def total_ways_to_assign : ℕ := 23760

-- Prove the number of shifts needed
theorem shifts_needed : total_workers / workers_per_shift = 6 := by
  sorry

end shifts_needed_l363_363371


namespace area_of_triangle_JKL_l363_363030

noncomputable section
open Real

theorem area_of_triangle_JKL :
  ∀ (J K L : Point) (a b : ℝ),
    angle J K L = π / 4 →
    dist J K = dist J L →
    dist K L = 20 →
    a = dist J K → b = dist J L →
    ∃ (area : ℝ), area = 1/2 * a * b ∧ area = 100 :=
by
  sorry

end area_of_triangle_JKL_l363_363030


namespace part1_part2_l363_363130

-- Define the coordinates of point P as functions of n
def pointP (n : ℝ) : ℝ × ℝ := (n + 3, 2 - 3 * n)

-- Condition 1: Point P is in the fourth quadrant
def inFourthQuadrant (n : ℝ) : Prop :=
  let point := pointP n
  point.1 > 0 ∧ point.2 < 0

-- Condition 2: Distance from P to the x-axis is 1 greater than the distance to the y-axis
def distancesCondition (n : ℝ) : Prop :=
  abs (2 - 3 * n) + 1 = abs (n + 3)

-- Definition of point Q
def pointQ (n : ℝ) : ℝ × ℝ := (n, -4)

-- Condition 3: PQ is parallel to the x-axis
def pqParallelX (n : ℝ) : Prop :=
  (pointP n).2 = (pointQ n).2

-- Theorems to prove the coordinates of point P and the length of PQ
theorem part1 (n : ℝ) (h1 : inFourthQuadrant n) (h2 : distancesCondition n) : pointP n = (6, -7) :=
sorry

theorem part2 (n : ℝ) (h1 : pqParallelX n) : abs ((pointP n).1 - (pointQ n).1) = 3 :=
sorry

end part1_part2_l363_363130


namespace sum_of_distinct_numbers_with_8_trailing_zeros_and_90_divisors_l363_363289

theorem sum_of_distinct_numbers_with_8_trailing_zeros_and_90_divisors :
  ∃ N1 N2 : ℕ, 
    (N1 ≠ N2) ∧
    (∃ k1 k2 : ℕ, N1 = 10^8 * k1 ∧ N2 = 10^8 * k2 ∧
     (∏ (d : ℕ) in (range (succ N1)), (d ∣ N1) ∧ N1 % d = 0).card = 90) ∧
    (∏ (d : ℕ) in (range (succ N2)), (d ∣ N2) ∧ N2 % d = 0).card = 90 →
    N1 + N2 = 700000000 :=
by
  sorry

end sum_of_distinct_numbers_with_8_trailing_zeros_and_90_divisors_l363_363289


namespace hyperbola_eccentricity_range_l363_363508

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity condition to prove
def eccentricity_condition (a b : ℝ) : Prop :=
  (a^2 + b^2) / a^2 ≥ 4

-- Define the main theorem
theorem hyperbola_eccentricity_range {a b : ℝ} (h : hyperbola a b) 
  (slope_condition : (b / a) ≥ real.sqrt 3) : eccentricity_condition a b :=
sorry

end hyperbola_eccentricity_range_l363_363508


namespace basketball_game_l363_363016

-- Define conditions and variables
variables {a r b d : ℕ}

-- Define the arithmetic sequence for the Lions
def lions_scores_first_quarter := b
def lions_scores_second_quarter := b + d
def lions_scores_third_quarter := b + 2 * d
def lions_scores_fourth_quarter := b + 3 * d

-- Define the geometric sequence for the Eagles
def eagles_scores_first_quarter := a
def eagles_scores_second_quarter := a * r
def eagles_scores_third_quarter := a * r^2
def eagles_scores_fourth_quarter := a * r^3

-- Define the sums for the first half
def eagles_first_half := eagles_scores_first_quarter + eagles_scores_second_quarter
def lions_first_half := lions_scores_first_quarter + lions_scores_second_quarter

-- Define the total scores for the game
def eagles_total := eagles_scores_first_quarter + eagles_scores_second_quarter + eagles_scores_third_quarter + eagles_scores_fourth_quarter
def lions_total := lions_scores_first_quarter + lions_scores_second_quarter + lions_scores_third_quarter + lions_scores_fourth_quarter

-- Define the total scores for the second half
def eagles_second_half := eagles_scores_third_quarter + eagles_scores_fourth_quarter
def lions_second_half := lions_scores_third_quarter + lions_scores_fourth_quarter

-- Problem statement in Lean 4
theorem basketball_game :
  eagles_first_half = lions_first_half →
  eagles_total = lions_total + 1 →
  eagles_total ≤ 100 →
  lions_total ≤ 100 →
  eagles_second_half + lions_second_half = 109 :=
sorry

end basketball_game_l363_363016


namespace arithmetic_mean_of_S_l363_363303

-- Define the set S of integers
def S : Set ℤ := {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6}

-- Define the arithmetic mean of a set of integers
noncomputable def mean (s : Set ℤ) : ℚ :=
  (s.to_finset.sum id : ℚ) / (s.to_finset.card : ℚ)

-- State the theorem
theorem arithmetic_mean_of_S : mean S = 3 / 2 :=
by
  sorry

end arithmetic_mean_of_S_l363_363303


namespace Jenny_ate_65_l363_363652

theorem Jenny_ate_65 (mike_squares : ℕ) (jenny_squares : ℕ)
  (h1 : mike_squares = 20)
  (h2 : jenny_squares = 3 * mike_squares + 5) :
  jenny_squares = 65 :=
by
  sorry

end Jenny_ate_65_l363_363652


namespace parallel_vectors_x_val_l363_363097

open Real

theorem parallel_vectors_x_val (x : ℝ) :
  let a : ℝ × ℝ := (3, 4)
  let b : ℝ × ℝ := (x, 1/2)
  a.1 * b.2 = a.2 * b.1 →
  x = 3/8 := 
by
  intro h
  -- Use this line if you need to skip the proof
  sorry

end parallel_vectors_x_val_l363_363097


namespace problem_I_problem_II_l363_363128

variable {k : Type*} [LinearOrderedField k]

-- Definitions
def symmetric (A O: k × k) (B: k × k) : Prop :=
  B = (2 * O.1 - A.1, 2 * O.2 - A.2)

def slope (P Q : k × k) : k :=
  (Q.2 - P.2) / (Q.1 - P.1)

def trajectory (P : k × k) : Prop :=
  P.1 ^ 2 + 3 * P.2 ^ 2 = 4 ∧ P.1 ≠ 1 ∧ P.1 ≠ -1

def equal_areas (A B P M N: k × k) : Prop :=
  let area (X Y Z: k × k) : k := (X.1 * (Y.2 - Z.2) + Y.1 * (Z.2 - X.2) + Z.1 * (X.2 - Y.2)).abs / 2
  area A B P = area P M N

-- Proof statements
theorem problem_I (P : k × k) :
  product_of_slopes_eq (A (-1, 1)) (B (1, -1)) P (-1/3) → trajectory P :=
sorry

theorem problem_II (P M N: k × k) :
  x₀ = 5/3 →
  equal_areas (A(-1, 1)) (B (1, -1)) (P (5/3, y₀)) M N →
  ∃ x₀ y₀, equal_areas (A(-1, 1)) (B (1, -1)) (P (x₀, y₀)) M N :=
sorry

end problem_I_problem_II_l363_363128


namespace students_above_120_l363_363615

noncomputable def class_size : ℕ := 50

noncomputable def mu : ℝ := 110
noncomputable def sigma : ℝ := 10

-- Hypothesize the normal distribution and given probability
axiom normal_distribution (ξ : ℝ → ℝ) : Prop := ξ ~ NormalDistribution(mu, sigma^2)
axiom prob_interval : ℝ := P(100 ≤ ξ ∧ ξ ≤ 110) = 0.36

-- The goal to prove
theorem students_above_120 : number_of_students_scored_above_120 = 7 :=
by
  sorry

end students_above_120_l363_363615


namespace probability_exactly_2_hits_probability_at_least_2_hits_probability_exactly_2_hits_third_hit_l363_363829

noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ := 
  (Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))) * p^k * (1 - p)^(n - k)

theorem probability_exactly_2_hits : 
  binomial_prob 5 2 0.8 ≈ 0.05 := sorry

theorem probability_at_least_2_hits :
  1 - binomial_prob 5 0 0.8 - binomial_prob 5 1 0.8 ≈ 0.99 := sorry

theorem probability_exactly_2_hits_third_hit :
  0.8 * binomial_prob 4 1 0.8 ≈ 0.02 := sorry

end probability_exactly_2_hits_probability_at_least_2_hits_probability_exactly_2_hits_third_hit_l363_363829


namespace determine_digit_phi_l363_363010

theorem determine_digit_phi (Φ : ℕ) (h1 : Φ > 0) (h2 : Φ < 10) (h3 : 504 / Φ = 40 + 3 * Φ) : Φ = 8 :=
by
  sorry

end determine_digit_phi_l363_363010


namespace problem_quadratic_has_real_root_l363_363596

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l363_363596


namespace parabola_directrix_l363_363737

theorem parabola_directrix (y x : ℝ) : y^2 = -8 * x → x = -1 :=
by
  sorry

end parabola_directrix_l363_363737


namespace solve_y_l363_363311

theorem solve_y (y : ℚ) (h : (3 * y) / 7 = 14) : y = 98 / 3 := 
by sorry

end solve_y_l363_363311


namespace trigonometric_identity_l363_363324

theorem trigonometric_identity (α : ℝ) : 
  (1 + 1 / (cos (2 * α)) + tan (2 * α)) * 
  (1 - 1 / (cos (2 * α)) + tan (2 * α)) = 
  2 * tan (2 * α) :=
by
  sorry

end trigonometric_identity_l363_363324


namespace geometry_problem_l363_363912

theorem geometry_problem
  (ABCD : Type) [square ABCD] 
  (A B C D E F : ABCD)
  (h1 : is_square ABCD)
  (h2 : is_equilateral_triangle ADE)
  (h3 : diagonal_intersect AC ED F)
  : CE = CF :=
sorry

end geometry_problem_l363_363912


namespace quadratic_has_real_root_iff_b_in_interval_l363_363544

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l363_363544


namespace mutually_exclusive_complements_union_is_certain_l363_363484

variable {Ω : Type} -- The sample space type
variable {M N : Set Ω} -- Events in the sample space
variable [Fact (M ∩ N = ∅)] -- The fact that M and N are mutually exclusive

theorem mutually_exclusive_complements_union_is_certain :
  is_certain_event (compl M ∪ compl N) :=
sorry

end mutually_exclusive_complements_union_is_certain_l363_363484


namespace problem_statement_l363_363778

-- Define the factorial function
def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the exponents of the prime factors of 10!
def exponents_of_prime_factors (n : ℕ) : ℕ × ℕ × ℕ :=
(if n = 10 then (8, 4, 2) else (0, 0, 0))

-- Calculate the largest perfect square that divides n!
noncomputable def largest_perfect_square (n : ℕ) : ℕ :=
let (a, b, c) := exponents_of_prime_factors n in
(2^a * 3^b * 5^c)

-- Calculate the square root of the perfect square
noncomputable def square_root (n : ℕ) : ℕ :=
let (a, b, c) := exponents_of_prime_factors n in
(2^(a/2) * 3^(b/2) * 5^(c/2))

-- Calculate the sum of the exponents of the prime factors of the square root
noncomputable def sum_exponents (n : ℕ) : ℕ :=
let (a, b, c) := exponents_of_prime_factors n in
(a / 2) + (b / 2) + (c / 2)

-- Lean 4 theorem statement
theorem problem_statement : sum_exponents 10 = 7 :=
by sorry -- Proof omitted

end problem_statement_l363_363778


namespace perimeter_eq_20_l363_363748

-- Define the lengths of the sides
def horizontal_sides := [2, 3]
def vertical_sides := [2, 3, 3, 2]

-- Define the perimeter calculation
def perimeter := horizontal_sides.sum + vertical_sides.sum

theorem perimeter_eq_20 : perimeter = 20 :=
by
  -- We assert that the calculations do hold
  sorry

end perimeter_eq_20_l363_363748


namespace range_of_k_l363_363180

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k}

-- State the theorem
theorem range_of_k (k : ℝ) : (M ∩ N k).Nonempty ↔ k ∈ Set.Ici (-1) :=
by
  sorry

end range_of_k_l363_363180


namespace problem_l363_363232

def f (x : ℝ) : ℝ := sorry  -- f is a function from ℝ to ℝ

theorem problem (h : ∀ x : ℝ, 3 * f x + f (2 - x) = 4 * x^2 + 1) : f 5 = 133 / 4 := 
by 
  sorry -- the proof is omitted

end problem_l363_363232


namespace min_packs_120_cans_l363_363208

theorem min_packs_120_cans (p8 p16 p32 : ℕ) (total_cans packs_needed : ℕ) :
  total_cans = 120 →
  p8 * 8 + p16 * 16 + p32 * 32 = total_cans →
  packs_needed = p8 + p16 + p32 →
  (∀ (q8 q16 q32 : ℕ), q8 * 8 + q16 * 16 + q32 * 32 = total_cans → q8 + q16 + q32 ≥ packs_needed) →
  packs_needed = 5 :=
by {
  sorry
}

end min_packs_120_cans_l363_363208


namespace crayons_total_l363_363977

def initial_crayons : Nat := 7
def added_crayons : Nat := 6
def total_crayons : Nat := initial_crayons + added_crayons

theorem crayons_total : total_crayons = 13 := 
by 
    unfold total_crayons 
    norm_num

end crayons_total_l363_363977


namespace possible_values_of_expression_l363_363072

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∃ v : ℝ, v = (a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|) ∧ 
            (v = 5 ∨ v = 1 ∨ v = -3 ∨ v = -5)) :=
by
  sorry

end possible_values_of_expression_l363_363072


namespace sqrt_sum_inequality_l363_363189

theorem sqrt_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 21) : sqrt a + sqrt b < 2 * sqrt 11 :=
by
  sorry

end sqrt_sum_inequality_l363_363189


namespace supremum_of_holomorphic_function_l363_363157

open Complex

noncomputable def unit_disk : Set ℂ := {z : ℂ | norm z < 1}

noncomputable def holomorphic_in_unit_disk (f : ℂ → ℂ) : Prop :=
  ∀ z ∈ unit_disk, ∃ U : Set ℂ, U ⊆ unit_disk ∧ is_open U ∧ z ∈ U ∧ differentiable_on ℂ f U

theorem supremum_of_holomorphic_function
  (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1)
  (f : ℂ → ℂ) (h₂ : ∀ z, z ∈ unit_disk → f z ≠ 0)
  (h₃ : holomorphic_in_unit_disk f)
  (h₄ : f a = 1)
  (h₅ : f (-a) = -1) :
  ∃ M : ℝ, ∀ z ∈ unit_disk, norm (f z) ≤ M ∧ M ≥ exp ((1 - a^2) / (4 * a) * π) :=
sorry

end supremum_of_holomorphic_function_l363_363157


namespace quadratic_real_roots_l363_363575

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l363_363575


namespace intersection_of_sets_l363_363483

open Set

theorem intersection_of_sets :
  let A := {x : ℤ | |x| < 3}
  let B := {x : ℤ | |x| > 1}
  A ∩ B = ({-2, 2} : Set ℤ) := by
  sorry

end intersection_of_sets_l363_363483


namespace nearest_integer_x_sub_y_l363_363661

theorem nearest_integer_x_sub_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : |x| - y = 4) 
  (h2 : |x| * y - x^3 = 1) : 
  abs (x - y - 4) < 1 :=
sorry

end nearest_integer_x_sub_y_l363_363661


namespace arithmetic_sequence_an_arithmetic_sequence_sn_geometric_sequence_tn_l363_363634

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1

noncomputable def S (n : ℕ) : ℕ := (1 to n).sum (λ k, a k)

noncomputable def b (n : ℕ) : ℕ := a n * 2^(n - 1)

noncomputable def T (n : ℕ) : ℕ := (1 to n).sum (λ k, b k)

theorem arithmetic_sequence_an (n : ℕ) :
  a 1 = 1 ∧ (∀ n, S (2 * n) / S n = 4) →
  (∀ n, a n = 2 * n - 1) :=
sorry

theorem arithmetic_sequence_sn (n : ℕ) :
  (∀ n, S (2 * n) / S n = 4) →
  (∀ n, S n = n ^ 2) :=
sorry

theorem geometric_sequence_tn (n : ℕ) :
  (∀ n, b n = a n * 2^(n - 1)) →
  (T n = (2 * n - 3) * 2^n + 3) :=
sorry

end arithmetic_sequence_an_arithmetic_sequence_sn_geometric_sequence_tn_l363_363634


namespace fixed_point_A_l363_363740

def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 2016) + 1

theorem fixed_point_A (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : f a 2016 = 2 :=
by
  -- Proof goes here
  sorry

end fixed_point_A_l363_363740


namespace total_tax_percentage_l363_363683

noncomputable def total_spent_excluding_taxes := 100 -- Assuming Jill spent $100 in total, excluding taxes
def clothing_percent := 0.50
def food_percent := 0.20
def other_items_percent := 0.30

def clothing_tax_percent := 0.05
def food_tax_percent := 0.0
def other_items_tax_percent := 0.10

-- Proof statement
theorem total_tax_percentage : 
  (clothing_percent * clothing_tax_percent +
   food_percent * food_tax_percent +
   other_items_percent * other_items_tax_percent) * total_spent_excluding_taxes / total_spent_excluding_taxes * 100 = 5.5 := 
by 
  sorry

end total_tax_percentage_l363_363683


namespace find_f_neg_5_l363_363923

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_domain : ∀ x : ℝ, true)
variable (h_positive : ∀ x : ℝ, x > 0 → f x = log 5 x + 1)

theorem find_f_neg_5 : f (-5) = -2 :=
by
  sorry

end find_f_neg_5_l363_363923


namespace ellipse_equation_and_fixed_point_l363_363928

theorem ellipse_equation_and_fixed_point :
let E := (x y : ℝ) (m n > 0) (m ≠ n),
    m * x^2 + n * y^2 = 1
    in
  (E (0 : ℝ) (-2 : ℝ)) -- point A
  ∧ (E (3 / 2 : ℝ) (-1 : ℝ)) -- point B
  ∧ (∀ P : ℝ × ℝ, P = (1, -2) → -- point P
    ∃ M N : ℝ × ℝ, 
      (E M.1 M.2 ∧ E N.1 N.2) -- points M and N on E
      ∧ (M ≠ N) 
      ∧ (∀ T H : ℝ × ℝ, 
            T.y = ((2/3) * T.x - 2) -- T (x * (2 / 3) - 2, y)
            → H = (2 * T - M) -- H is such that MT = TH
            → line_through H N passes (0, -2))
            ) :=
begin
  sorry, -- Proof
end

end ellipse_equation_and_fixed_point_l363_363928


namespace range_of_x_l363_363073

theorem range_of_x (x : ℝ) :
  (∃ A B C a b c: ℝ, a = sqrt 3 ∧ b = x ∧ C = 180 - A - B ∧ A = 60 ∧ 
   triangle ABC a b c ∧ B ∈ set.Ioo (60:ℝ) 120) → 
  sqrt 3 < x ∧ x < 2 := 
by
  sorry

end range_of_x_l363_363073


namespace last_two_digits_of_expression_l363_363694

theorem last_two_digits_of_expression (n : ℤ) (h : n % 2 = 1 ∧ n > 0) : 
  (2^(2*n) * (2^(2*n+1) - 1)) % 100 = 28 :=
by
  sorry

end last_two_digits_of_expression_l363_363694


namespace maximum_lambda_inequality_l363_363336

noncomputable def maximum_lambda := 8 * (81 / 80) ^ 45

theorem maximum_lambda_inequality :
  ∀ (a : ℕ → ℝ) (h : ∀ i, 1 ≤ i ∧ i ≤ 45 → 0 < a i ∧ a i ≤ 1),
    sqrt (45 / ∑ i in finset.range 45, a i) ≥ 1 + maximum_lambda * ∏ i in finset.range 45, (1 - a i) :=
by sorry

end maximum_lambda_inequality_l363_363336


namespace number_of_numbers_on_board_l363_363268

theorem number_of_numbers_on_board (numbers : List ℝ) (h_pos : ∀ n ∈ numbers, n > 0) (h_sum : numbers.sum = 1)
  (h_sum_5_largest : (numbers.sort (· ≤ ·)).reverse.take 5).sum = 0.29 * numbers.sum
  (h_sum_5_smallest : (numbers.sort (· ≤ ·)).take 5).sum = 0.26 * numbers.sum :
  numbers.length = 18 :=
sorry

end number_of_numbers_on_board_l363_363268


namespace horse_travel_distance_l363_363633

noncomputable def geom_sum (a₁ g : ℕ → ℚ) (n : ℕ) : ℚ :=
  a₁ * (1 - g ^ n) / (1 - g)

theorem horse_travel_distance 
  (a₁ : ℚ) 
  (g : ℚ := 1 / 2) 
  (S₇ : ℚ := 700) 
  (S₁₄ : ℚ := 22575 / 32) :
  geom_sum a₁ g 7 = 700 →
  geom_sum a₁ g 14 = 22575 / 32 :=
by
  sorry

end horse_travel_distance_l363_363633


namespace simplify_trig_expr_l363_363709

theorem simplify_trig_expr : 
  let θ := 60
  let tan_θ := Real.sqrt 3
  let cot_θ := (Real.sqrt 3)⁻¹
  (tan_θ^3 + cot_θ^3) / (tan_θ + cot_θ) = 7 / 3 :=
by
  sorry

end simplify_trig_expr_l363_363709


namespace number_of_subsets_of_M_l363_363943

-- Define the set M with three distinct elements.
def M : Set (ℕ) := {1, 2, 3}

-- Define the condition N ⊆ M.
def is_subset (N : Set (ℕ)) : Prop := N ⊆ M

-- Prove the number of subsets of M is 8.
theorem number_of_subsets_of_M : (Set.powerset M).card = 8 := by
  sorry

end number_of_subsets_of_M_l363_363943


namespace percentage_reduction_in_hours_worked_l363_363397

variables (W H : ℝ) -- Define the original hourly wage and original hours worked

-- Define the conditions
def original_weekly_income := W * H
def new_hourly_wage := 1.25 * W

-- Define the unchanged total weekly income condition and solve for new hours H'
def new_hours_worked := original_weekly_income / new_hourly_wage
def hours_reduction := 100 * (1 - (new_hours_worked / H))

theorem percentage_reduction_in_hours_worked (h₁: W > 0) (h₂: H > 0) : 
  hours_reduction W H = 20 :=
  sorry

end percentage_reduction_in_hours_worked_l363_363397


namespace num_O_atoms_l363_363362

def compound_molecular_weight : ℕ := 62
def atomic_weight_H : ℕ := 1
def atomic_weight_C : ℕ := 12
def atomic_weight_O : ℕ := 16
def num_H_atoms : ℕ := 2
def num_C_atoms : ℕ := 1

theorem num_O_atoms (H_weight : ℕ := num_H_atoms * atomic_weight_H)
                    (C_weight : ℕ := num_C_atoms * atomic_weight_C)
                    (total_weight : ℕ := compound_molecular_weight)
                    (O_weight := atomic_weight_O) : 
    (total_weight - (H_weight + C_weight)) / O_weight = 3 :=
by
  sorry

end num_O_atoms_l363_363362


namespace scientific_notation_36000_l363_363190

theorem scientific_notation_36000 : 36000 = 3.6 * (10^4) := 
by 
  -- Skipping the proof by adding sorry
  sorry

end scientific_notation_36000_l363_363190


namespace smallest_positive_perfect_square_divisible_by_2_and_5_l363_363775

theorem smallest_positive_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, n = k^2) ∧ (2 ∣ n) ∧ (5 ∣ n) ∧ n = 100 :=
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_and_5_l363_363775


namespace log_equation_solution_l363_363717

noncomputable def log_six (x : ℝ) : ℝ := log x / log 6

theorem log_equation_solution :
  (1 - log_six 3)^2 + log_six 2 * (log_six 2 + log_six 9) / log_six 4 = 1 :=
by
  sorry

end log_equation_solution_l363_363717


namespace fractional_part_water_final_mixture_l363_363806

theorem fractional_part_water_final_mixture (initial_volume : ℕ) (replaced_volume : ℕ) (iterations : ℕ) 
(h_initial : initial_volume = 20) 
(h_replaced : replaced_volume = 5) 
(h_iterations : iterations = 3) : 
  let fraction_remaining_after_replacements (frac : ℚ) (n : ℕ) := frac ^ n 
  in fraction_remaining_after_replacements (3/4) iterations = 27/64 := 
by 
  have iter_factor : ℚ := 3 / 4
  have result_after_replacements : ℚ := iter_factor ^ iterations 
  have : result_after_replacements = 27 / 64 := by sorry
  exact this

end fractional_part_water_final_mixture_l363_363806


namespace find_davids_marks_in_physics_l363_363861

theorem find_davids_marks_in_physics (marks_english : ℕ) (marks_math : ℕ) (marks_chemistry : ℕ) (marks_biology : ℕ)
  (average_marks : ℕ) (num_subjects : ℕ) (H1 : marks_english = 61) 
  (H2 : marks_math = 65) (H3 : marks_chemistry = 67) 
  (H4 : marks_biology = 85) (H5 : average_marks = 72) (H6 : num_subjects = 5) :
  ∃ (marks_physics : ℕ), marks_physics = 82 :=
by
  sorry

end find_davids_marks_in_physics_l363_363861


namespace quadratic_real_roots_l363_363576

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l363_363576


namespace trapezoid_perimeter_l363_363642

-- Definition of the isosceles trapezoid ABCD with AB and CD equal
structure IsoscelesTrapezoid (A B C D : Type) :=
  (AB CD : ℝ)
  (BC DA : ℝ)
  (AB_eq_CD : AB = CD)
  (perimeter_eq : AB + BC + CD + DA = 34)

-- Constant declaration to represent sides of trapezoid ABCD
constants (AB CD BC DA : ℝ)
constant (trapezoidABCD : IsoscelesTrapezoid AB CD BC DA)

-- Statement that checks the perimeter of trapezoid ABCD equals 34
theorem trapezoid_perimeter :
  trapezoidABCD.perimeter_eq := sorry

end trapezoid_perimeter_l363_363642


namespace isosceles_triangle_ratio_l363_363275

theorem isosceles_triangle_ratio
    {A B C M A1 B1 : Type}
    [Point A] [Point B] [Point C] [Point M] [Point A1] [Point B1]
    (isosceles : IsoscelesTriangle ABC A B C) (M_on_AB : M ∈ lineSegment AB)
    (A1_on_CA : ∃ l: Line, l.contains M ∧ l.intersectsAt CA A1)
    (B1_on_CB : ∃ l: Line, l.contains M ∧ l.intersectsAt CB B1) :
    ratio (A1 ⟶ A) (A1 ⟶ M) = ratio (B1 ⟶ B) (B1 ⟶ M) :=
sorry

end isosceles_triangle_ratio_l363_363275


namespace james_total_points_l363_363725

def points_per_correct_answer : ℕ := 2
def bonus_points_per_round : ℕ := 4
def total_rounds : ℕ := 5
def questions_per_round : ℕ := 5
def total_questions : ℕ := total_rounds * questions_per_round
def questions_missed_by_james : ℕ := 1
def questions_answered_by_james : ℕ := total_questions - questions_missed_by_james
def points_for_correct_answers : ℕ := questions_answered_by_james * points_per_correct_answer
def complete_rounds_by_james : ℕ := total_rounds - 1  -- Since James missed one question, he has 4 complete rounds
def bonus_points_by_james : ℕ := complete_rounds_by_james * bonus_points_per_round
def total_points : ℕ := points_for_correct_answers + bonus_points_by_james

theorem james_total_points : total_points = 64 := by
  sorry

end james_total_points_l363_363725


namespace intersection_M_N_l363_363096

-- Define the universal set U, and subsets M and N
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x < 1}

-- Prove that the intersection of M and N is as stated
theorem intersection_M_N :
  M ∩ N = {x | -2 ≤ x ∧ x < 1} :=
by
  -- This is where the proof would go
  sorry

end intersection_M_N_l363_363096


namespace ratio_of_games_played_to_losses_l363_363325

-- Definitions based on the conditions
def total_games_played : ℕ := 10
def games_won : ℕ := 5
def games_lost : ℕ := total_games_played - games_won

-- The proof problem
theorem ratio_of_games_played_to_losses : (total_games_played / Nat.gcd total_games_played games_lost) = 2 ∧ (games_lost / Nat.gcd total_games_played games_lost) = 1 :=
by
  sorry

end ratio_of_games_played_to_losses_l363_363325


namespace Megan_zoo_pictures_l363_363320
-- Import the entirety of the necessary library

-- Define the proof problem based on the given conditions and answer
theorem Megan_zoo_pictures :
  ∃ Z : ℕ, Z + 18 - 31 = 2 ∧ Z = 15 :=
by
  use 15
  split
  . -- Prove the equation part
    calc
      15 + 18 - 31 = 33 - 31 := by rfl
                    _ = 2   := by rfl
  . -- Prove that Z = 15
    rfl

end Megan_zoo_pictures_l363_363320


namespace connections_in_computer_lab_l363_363758

theorem connections_in_computer_lab (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 := by
  sorry

end connections_in_computer_lab_l363_363758


namespace trigonometric_identities_l363_363315

open Real

theorem trigonometric_identities :
  (cos 75 * cos 75 = (2 - sqrt 3) / 4) ∧
  ((1 + tan 105) / (1 - tan 105) ≠ sqrt 3 / 3) ∧
  (tan 1 + tan 44 + tan 1 * tan 44 = 1) ∧
  (sin 70 * (sqrt 3 / tan 40 - 1) ≠ 2) :=
by
  sorry

end trigonometric_identities_l363_363315


namespace max_score_AHSME_l363_363866

-- Define the variables
variables {s c w : ℕ}

-- Define the scoring function
def score (c w : ℕ) : ℕ :=
  30 + 4 * c - w

-- State the theorem
theorem max_score_AHSME : ∃ s c w, score c w = 130 ∧ 100 < score c w ∧ c ≤ 30 ∧ w ≥ 0 ∧ 
                        (∀ c' w', (score c' w' = 130 → (c', w') = (25, 30))) :=
by
  sorry

end max_score_AHSME_l363_363866


namespace probability_of_three_given_sum_seven_l363_363312

theorem probability_of_three_given_sum_seven : 
  (∃ (dice1 dice2 : ℕ), (1 ≤ dice1 ∧ dice1 ≤ 6 ∧ 1 ≤ dice2 ∧ dice2 ≤ 6) ∧ (dice1 + dice2 = 7) 
    ∧ (dice1 = 3 ∨ dice2 = 3)) →
  (∃ (dice1 dice2 : ℕ), (1 ≤ dice1 ∧ dice1 ≤ 6 ∧ 1 ≤ dice2 ∧ dice2 ≤ 6) ∧ (dice1 + dice2 = 7)) →
  ∃ (p : ℚ), p = 1/3 :=
by 
  sorry

end probability_of_three_given_sum_seven_l363_363312


namespace average_not_1380_l363_363304

-- Define the set of numbers
def numbers := [1200, 1400, 1510, 1520, 1530, 1200]

-- Define the claimed average
def claimed_avg := 1380

-- The sum of the numbers
def sumNumbers := numbers.sum

-- The number of items in the set
def countNumbers := numbers.length

-- The correct average calculation
def correct_avg : ℚ := sumNumbers / countNumbers

-- The proof problem: proving that the correct average is not equal to the claimed average
theorem average_not_1380 : correct_avg ≠ claimed_avg := by
  sorry

end average_not_1380_l363_363304


namespace sum_n_max_value_l363_363897

noncomputable def arithmetic_sequence (a_1 : Int) (d : Int) (n : Nat) : Int :=
  a_1 + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a_1 : Int) (d : Int) (n : Nat) : Int :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem sum_n_max_value :
  (∃ n : Nat, n = 9 ∧ sum_arithmetic_sequence 25 (-3) n = 117) :=
by
  let a1 := 25
  let d := -3
  use 9
  -- To complete the proof, we would calculate the sum of the first 9 terms
  -- of the arithmetic sequence with a1 = 25 and difference d = -3.
  sorry

end sum_n_max_value_l363_363897


namespace quad_split_equal_area_l363_363998

variables {A B C D M E : Type}
variables [Point A] [Point B] [Point C] [Point D] [Point M] [Point E]

-- Assume A, B, C, D are points forming a quadrilateral
-- Assume M is the midpoint of diagonal BD
-- Assume E is the intersection of the line through M parallel to AC and line AD

theorem quad_split_equal_area
  (ABCD : quadrilateral A B C D)
  (M_midpoint : midpoint M B D)
  (ME_parallel_AC : parallel (line_through M E) (line_through A C))
  (E_on_AD : lies_on E (line_through A D)) :
  divides_equal_area (segment C E) ABCD :=
sorry

end quad_split_equal_area_l363_363998


namespace number_of_shelves_l363_363386

-- Given conditions
def booksBeforeTrip : ℕ := 56
def booksBought : ℕ := 26
def avgBooksPerShelf : ℕ := 20
def booksLeftOver : ℕ := 2
def totalBooks : ℕ := booksBeforeTrip + booksBought

-- Statement to prove
theorem number_of_shelves :
  totalBooks - booksLeftOver = 80 →
  80 / avgBooksPerShelf = 4 := by
  intros h
  sorry

end number_of_shelves_l363_363386


namespace problem_ABCD_cos_l363_363626

/-- In convex quadrilateral ABCD, angle A = 2 * angle C, AB = 200, CD = 200, the perimeter of 
ABCD is 720, and AD ≠ BC. Find the floor of 1000 * cos A. -/
theorem problem_ABCD_cos (A C : ℝ) (AB CD AD BC : ℝ) (h1 : AB = 200)
  (h2 : CD = 200) (h3 : AD + BC = 320) (h4 : A = 2 * C)
  (h5 : AD ≠ BC) : ⌊1000 * Real.cos A⌋ = 233 := 
sorry

end problem_ABCD_cos_l363_363626


namespace unique_real_x_l363_363074

theorem unique_real_x (x : ℝ) (hx : x^2 ∈ ({0, 1, x} : set ℝ)) : 
  x = -1 :=
by {
  sorry
}

end unique_real_x_l363_363074


namespace value_of_x_l363_363659

variables (A B C D E F : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (AC AD DF BF : Real)

-- Conditions
def right_triangle (ABC : Type*) (angle_ABC : ℝ) (hypotenuse_AC : ℝ) :=
  angle_ABC = π/2 ∧ hypotenuse_AC = 20

def points_on_AC (AD DC : ℝ) (point_D point_E : Type*) :=
  AD = 12 ∧ DC = 8

def perpendicular_from_B_to_AC (BF : ℝ) :=
  BF = 2 * sqrt 21

-- Question
theorem value_of_x (x : ℝ) : 
  right_triangle ABC angle_ABC hypotenuse_AC →
  points_on_AC AD DC point_D point_E →
  perpendicular_from_B_to_AC BF →
  (DF = x) = (x = 12 - sqrt 232) :=
by
  sorry

end value_of_x_l363_363659


namespace find_f_pi_over_4_l363_363939

variable (f : ℝ → ℝ)
variable (h : ∀ x, f x = f (Real.pi / 4) * Real.cos x + Real.sin x)

theorem find_f_pi_over_4 : f (Real.pi / 4) = 1 := by
  sorry

end find_f_pi_over_4_l363_363939


namespace isomorphism_of_second_order_l363_363444

open Graph

structure SimpleGraph (V : Type) :=
(edge_set : set (V × V))
(symm : ∀ {u v}, edge_set (u, v) → edge_set (v, u))
(loopless : ∀ v, ¬ edge_set (v, v))

namespace SimpleGraph

def is_isomorphic {V V' : Type} (G G' : SimpleGraph V) : Prop :=
∃ f : V → V',
  (function.bijective f) ∧
  ∀ u v, G.edge_set (u, v) ↔ G'.edge_set (f u, f v)

def common_neighbor_graph (G : SimpleGraph V) : SimpleGraph V :=
{ edge_set := { ⟨u, v⟩ | u ≠ v ∧ ∃ w, G.edge_set (u, w) ∧ G.edge_set (v, w) },
  symm := begin
    rintros ⟨u, v⟩ h,
    rcases h with ⟨h₁, w, hu, hv⟩,
    exact ⟨h₁.symm, w, hv, hu⟩,
  end,
  loopless := begin
    intro v,
    intro h,
    rcases h with ⟨h₁, w, hwu, hwv⟩,
    exact h₁ rfl,
  end }

theorem isomorphism_of_second_order (G : SimpleGraph V) :
  (is_isomorphic G (common_neighbor_graph (common_neighbor_graph G))) →
  (is_isomorphic G (common_neighbor_graph G)) :=
begin
  sorry
end

end SimpleGraph

end isomorphism_of_second_order_l363_363444


namespace sum_of_midpoints_l363_363131

theorem sum_of_midpoints (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  simp [h]
  sorry

end sum_of_midpoints_l363_363131


namespace solve_for_c_l363_363106

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem solve_for_c (a b c d : ℝ) 
    (h : ∀ x : ℝ, quadratic_function a b c x ≥ d) : c = d + b^2 / (4 * a) :=
by
  sorry

end solve_for_c_l363_363106


namespace no_n_le_50_has_f50_eq_20_l363_363448

-- Define f_1 as twice the number of divisors of n
noncomputable def f1 (n : ℕ) : ℕ :=
  2 * (n.divisors.card)

-- Define f_j recursively
noncomputable def f : ℕ → ℕ → ℕ
| 1 n := f1 n
| (j + 1) n := f1 (f j n)

-- Define the problem statement in Lean
theorem no_n_le_50_has_f50_eq_20 : ∀ (n : ℕ), n ≤ 50 → f 50 n ≠ 20 :=
by
  intros n hn
  sorry

end no_n_le_50_has_f50_eq_20_l363_363448


namespace length_of_AC_l363_363627

-- Given conditions
variables {AB BC CD DA : ℝ}
variables {angle_ADC : ℝ}
variables (h1 : AB = 12) (h2 : BC = 12) (h3 : CD = 15) (h4 : DA = 15) (h5 : angle_ADC = 90)

-- Prove that AC = 15 * sqrt 2
theorem length_of_AC (h1 : AB = 12) (h2 : BC = 12) (h3 : CD = 15) (h4 : DA = 15) (h5 : angle_ADC = 90) :
  ∃ AC : ℝ, AC = 15 * sqrt 2 :=
by
  sorry

end length_of_AC_l363_363627


namespace initial_values_count_l363_363043

noncomputable def sequence (x₀ : ℝ) : ℕ → ℝ
| 0       := x₀
| (n + 1) := if 2 * (sequence x₀ n) < 1
             then 2 * (sequence x₀ n)
             else 2 * (sequence x₀ n) - 1

theorem initial_values_count : ∀ (x₀ : ℝ), 0 ≤ x₀ ∧ x₀ < 1 → 
  (set.finite {x₀ | sequence x₀ 0 = sequence x₀ 7}) ∧ 
  (set.card {x₀ | sequence x₀ 0 = sequence x₀ 7}) = 127 :=
by
  sorry

end initial_values_count_l363_363043


namespace line_passes_through_fixed_point_l363_363051

theorem line_passes_through_fixed_point (p q : ℝ) (h : p + 2 * q - 1 = 0) :
  p * (1/2) + 3 * (-1/6) + q = 0 :=
by
  -- placeholders for the actual proof steps
  sorry

end line_passes_through_fixed_point_l363_363051


namespace find_k_for_maximum_value_l363_363503

theorem find_k_for_maximum_value (k : ℝ) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 2 → k * x^2 + 2 * k * x + 1 ≤ 5) ∧
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 2 ∧ k * x^2 + 2 * k * x + 1 = 5) ↔
  k = 1 / 2 ∨ k = -4 :=
by
  sorry

end find_k_for_maximum_value_l363_363503


namespace A_squared_is_zero_l363_363658

variables {R : Type*} [field R]

def is_zero_matrix {n : Type*} [fintype n] [decidable_eq n] (A : matrix n n R) : Prop :=
∀ i j, A i j = 0

theorem A_squared_is_zero {A : matrix (fin 2) (fin 2) ℝ} (hA : A ^ 3 = 0) : A ^ 2 = 0 :=
begin
  sorry
end

end A_squared_is_zero_l363_363658


namespace quadratic_real_roots_l363_363562

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l363_363562


namespace probability_passing_through_intersections_C_and_D_l363_363832

-- Define the conditions
def total_moves (east_moves south_moves : ℕ) : ℕ := east_moves + south_moves

def binomial (n k : ℕ) : ℕ := nat.choose n k

def count_paths (east_moves south_moves : ℕ) : ℕ :=
  binomial (total_moves east_moves south_moves) east_moves

-- Define the probability calculation
def probability (num_paths total_paths : ℕ) : ℚ :=
  (num_paths : ℚ) / (total_paths : ℚ)

-- Define the statement
theorem probability_passing_through_intersections_C_and_D :
  let A_to_C_paths := count_paths 3 2,
      C_to_D_paths := count_paths 2 1,
      D_to_B_paths := count_paths 1 2,
      total_paths := count_paths 6 5 in
  probability (A_to_C_paths * C_to_D_paths * D_to_B_paths) total_paths = 15 / 77 := sorry

end probability_passing_through_intersections_C_and_D_l363_363832


namespace circle_radius_l363_363693

noncomputable def point := ℝ × ℝ

noncomputable def P : point := (9, 17)
noncomputable def Q : point := (18, 15)
noncomputable def y_intersect := 2
noncomputable def radius_of_circle (P Q : point) (y_intersect : ℝ) : ℝ := real.sqrt 21.25

theorem circle_radius (P Q : point) (y_intersect : ℝ) (hP : P = (9, 17)) (hQ : Q = (18, 15)) (hy : y_intersect = 2): radius_of_circle P Q y_intersect = real.sqrt 21.25 :=
by
  -- Definitions and conditions are used here.
  -- Proof will be elaborated later.
  sorry

end circle_radius_l363_363693


namespace nine_b_equals_eighteen_l363_363964

theorem nine_b_equals_eighteen (a b : ℝ) (h1 : 6 * a + 3 * b = 0) (h2 : a = b - 3) : 9 * b = 18 :=
  sorry

end nine_b_equals_eighteen_l363_363964


namespace problem_statement_l363_363962

theorem problem_statement (a b : ℝ) (h₁ : 30 ^ a = 2) (h₂ : 30 ^ b = 3) : 
  10 ^ ((1 - a - b) / (2 * (1 - b))) = real.sqrt 5 :=
sorry

end problem_statement_l363_363962


namespace parabola_axis_of_symmetry_l363_363751

def axis_of_symmetry (a b c : ℝ) : ℝ :=
  -b / (2 * a)

theorem parabola_axis_of_symmetry : axis_of_symmetry (-1/4) 1 (-4) = 2 :=
by 
  rw [axis_of_symmetry]
  simp
  norm_num
  sorry

end parabola_axis_of_symmetry_l363_363751


namespace joggers_meetings_l363_363692

theorem joggers_meetings (road_length : ℝ)
  (speed_A speed_B : ℝ)
  (start_time : ℝ)
  (meeting_time : ℝ) :
  road_length = 400 → 
  speed_A = 3 → 
  speed_B = 2.5 →
  start_time = 0 → 
  meeting_time = 1200 → 
  ∃ y : ℕ, y = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end joggers_meetings_l363_363692


namespace angle_BAC_l363_363143

theorem angle_BAC : 
  ∀ (A B C X Y Z : Type)
    (AX XY YZ ZB BC : ℝ)
    (angle_ABC : ℝ),
    AX = XY → XY = YZ → YZ = ZB → ZB = BC →
    angle_ABC = 150 →
    ∃ t : ℝ, t = 26.25 ∧ (∠ BAC = t) := 
by
  intros A B C X Y Z AX XY YZ ZB BC angle_ABC hAX hXY hYZ hZB hangle_ABC
  use 26.25
  sorry

end angle_BAC_l363_363143


namespace max_sets_of_three_l363_363682

theorem max_sets_of_three (n : ℕ) (h : n = 100) : 
  ∃ S : finset (finset ℕ), 
    (∀ s ∈ S, s.card = 3 ∧ (∃ a b c ∈ s, a * b = c ∨ b * c = a ∨ c * a = b)) ∧
    S.card = 8 := 
sorry

end max_sets_of_three_l363_363682


namespace solid_of_revolution_surface_area_l363_363823

theorem solid_of_revolution_surface_area
  (p d : ℝ) 
  (parallelogram : parallelogram) 
  (has_diagonal : parallelogram.has_diagonal d)
  (perimeter : parallelogram.perimeter = 2 * p) :
  parallelogram.surface_area_of_solid_of_revolution = 2 * π * d * p :=
sorry

end solid_of_revolution_surface_area_l363_363823


namespace stock_price_end_second_year_l363_363870

theorem stock_price_end_second_year
  (P₀ : ℝ) (r₁ r₂ : ℝ) 
  (h₀ : P₀ = 150)
  (h₁ : r₁ = 0.80)
  (h₂ : r₂ = 0.30) :
  let P₁ := P₀ + r₁ * P₀
  let P₂ := P₁ - r₂ * P₁
  P₂ = 189 :=
by
  sorry

end stock_price_end_second_year_l363_363870


namespace distribute_books_l363_363421

theorem distribute_books :
  let books := [1, 2, 3, 4, 5, 6]
  let people := [A, B, C]
  (∃ (f : fin 6 → people),
    (∀ p, 1 ≤ (multiset.card (multiset.filter (λ i, people.index_of p ≡ books[i]) books)) ) ∧
    (multiset.card (multiset.filter (λ i, f i = A) books) +
        multiset.card (multiset.filter (λ i, f i = B) books) +
        multiset.card (multiset.filter (λ i, f i = C) books) = 6)) → 
  let ways_411 := (finset.powerset_len 4 (finset.univ : finset (fin 6))).card *
                  (finset.powerset_len 1 (finset.univ \\ (finset.powerset_len 4 (finset.univ : finset (fin 6)))).card *
                  (finset.powerset_len 1 ((finset.univ \\ (finset.powerset_len 4 (finset.univ : finset (fin 6))) \\
                      (finset.powerset_len 1 (finset.univ \\ (finset.powerset_len 4 (finset.univ : finset (fin 6)))).card
    let ways_321 := (finset.powerset_len 3 (finset.univ : finset (fin 6))).card *
                  (finset.powerset_len 2 (finset.univ \\ (finset.powerset_len 3 (finset.univ : finset (fin 6))))).card *
                  (finset.powerset_len 1 (finset.univ \\ (finset.powerset_len 3 (finset.univ : finset (fin 6))) \\
                      (finset.powerset_len 2 (finset.univ \\ (finset.powerset_len 3 (finset.univ : finset (fin 6)))).card
    let ways_222 := (finset.powerset_len 2 (finset.univ : finset (fin 6))).card *
                  (finset.powerset_len 2 (finset.univ \\ (finset.powerset_len 2 (finset.univ : finset (fin 6))))).card *
                  (finset.powerset_len 2 (finset.univ \\ (finset.powerset_len 2 (finset.univ : finset (fin 6))) \\
                      (finset.powerset_len 2 (finset.univ \\ (finset.powerset_len 2 (finset.univ : finset (fin 6)))).card
  ways_411 + ways_321 + ways_222 = 540 := sorry

end distribute_books_l363_363421


namespace triangle_perimeter_is_17_l363_363242

noncomputable def triangle_perimeter (x : ℝ) : ℝ :=
  if (x^2 - 8*x + 12 = 0) ∧ (7 - 4 < x) ∧ (x < 7 + 4) then 4 + 7 + x else 0

theorem triangle_perimeter_is_17 : ∃ x : ℝ, triangle_perimeter x = 17 :=
by {
  use 6,
  unfold triangle_perimeter,
  split_ifs,
  { simp,
    ring,
    sorry,
  },
  { sorry }
}

end triangle_perimeter_is_17_l363_363242


namespace projection_of_a_on_b_l363_363519

structure Vector2 :=
(x : ℝ)
(y : ℝ)

def dot_product (a b : Vector2) : ℝ :=
a.x * b.x + a.y * b.y

def norm (v : Vector2) : ℝ :=
Real.sqrt (v.x ^ 2 + v.y ^ 2)

def projection (a b : Vector2) : ℝ :=
(dot_product a b) / (norm b)

theorem projection_of_a_on_b (a b : Vector2) (h_dot : dot_product a b = -3) (hb : b = ⟨3, 4⟩) : projection a b = -3 / 5 :=
by
  sorry

end projection_of_a_on_b_l363_363519


namespace complex_magnitudes_l363_363172

noncomputable def z_w_complex (z w : ℂ) (theta : ℝ) : Prop :=
  (|z| = 2) ∧ (|w| = 4) ∧ (theta = real.pi / 3) ∧ (|z + w| = 3)

theorem complex_magnitudes (z w : ℂ) (theta : ℝ) (h : z_w_complex z w theta) : 
  ∣1 / z + 1 / w∣ = (3 / 8) :=
by
  sorry

end complex_magnitudes_l363_363172


namespace avg_equivalence_l363_363727

-- Definition of binary average [a, b]
def avg2 (a b : ℤ) : ℤ := (a + b) / 2

-- Definition of ternary average {a, b, c}
def avg3 (a b c : ℤ) : ℤ := (a + b + c) / 3

-- Lean statement for proving the given problem
theorem avg_equivalence : avg3 (avg3 2 2 (-1)) (avg2 3 (-1)) 1 = 1 := by
  sorry

end avg_equivalence_l363_363727


namespace truth_tellers_in_groups_l363_363194

-- Setup the conditions for the two groups on the mysterious island.
def consistent_responses_in_first_group {n : ℕ} (responses : Finset ℕ) : Prop :=
  ∀ r ∈ responses, responses = {r}

def responses_in_second_group (responses : List ℕ) : Prop :=
  responses = [0, 1, 3, 3, 3, 4]

-- The proof that under these conditions, the number of truth-tellers are as stated.
theorem truth_tellers_in_groups :
  ∀ (group1_responses : Finset ℕ) (group2_responses : List ℕ),
    consistent_responses_in_first_group group1_responses →
    responses_in_second_group group2_responses →
    (group1_responses = {4} ∨ group1_responses = {0}) ∧
    (group2_responses.count 3 = 3 ∧ group2_responses.count 1 = 1) :=
by
  intros group1_responses group2_responses h1 h2
  sorry

end truth_tellers_in_groups_l363_363194


namespace triangle_circumcircles_intersect_at_point_l363_363147

open EuclideanGeometry
open Triangle

-- Given conditions
variables {V : Type*} [MetricSpace V] [NormedAddCommGroup V] [NormedSpace ℝ V]
variables {A B C A₀ C₁ B₁ M : V}
variables [Nonempty V]

-- Definitions based on conditions
def midpoint (X Y : V) : V := (X + Y) / 2

def line_bisect_segment (X Y M : V) : Prop := dist X M = dist Y M

-- Statements to prove the conditions
theorem triangle_circumcircles_intersect_at_point 
  (hABC: Triangle A B C)
  (hC₁ : C₁ = midpoint A B)
  (hB₁ : B₁ = midpoint A C)
  (hA₀ : altitude_point A B C = A₀)
  (hCircle1 : on_circumcircle (A B₁ C₁) M)
  (hCircle2 : on_circumcircle (B C₁ A₀) M)
  (hCircle3 : on_circumcircle (C B₁ A₀) M)
  : line_bisect_segment B₁ C₁ A₀ :=
sorry

end triangle_circumcircles_intersect_at_point_l363_363147


namespace find_area_standard_widescreen_l363_363271

def length_and_height (length_ratio height_ratio diagonal : ℝ) : ℝ × ℝ :=
  let coeff := (diagonal^2 / (length_ratio^2 + height_ratio^2)).sqrt
  (length_ratio * coeff, height_ratio * coeff)

def area (length height : ℝ) : ℝ :=
  length * height

theorem find_area_standard_widescreen 
  (diagonal : ℝ)
  (standard_length_ratio standard_height_ratio : ℝ)
  (widescreen_length_ratio widescreen_height_ratio : ℝ)
  (A_value : ℝ)
  (widescreen_area_ratio : ℝ)
  (ratio : A_value / widescreen_area_ratio = (area (16 * (diagonal^2 / (16^2 + 9^2)).sqrt) (9 * (diagonal^2 / (16^2 + 9^2)).sqrt)) / area (4 * (diagonal^2 / (4^2 + 3^2)).sqrt) (3 * (diagonal^2 / (4^2 + 3^2)).sqrt)) :
  A_value = 337 :=
by
  sorry

end find_area_standard_widescreen_l363_363271


namespace find_f_neg_5_l363_363926

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 5 + 1 else - (log (-x) / log 5 + 1)

theorem find_f_neg_5
  (h_odd : ∀ x : ℝ, f (-x) = -f (x))
  (h_domain : ∀ x : ℝ, x ∈ set.univ)
  (h_positive_def : ∀ x : ℝ, x > 0 → f x = log x / log 5 + 1)
  : f (-5) = -2 :=
by
  sorry

end find_f_neg_5_l363_363926


namespace number_of_numbers_on_board_l363_363267

theorem number_of_numbers_on_board (numbers : List ℝ) (h_pos : ∀ n ∈ numbers, n > 0) (h_sum : numbers.sum = 1)
  (h_sum_5_largest : (numbers.sort (· ≤ ·)).reverse.take 5).sum = 0.29 * numbers.sum
  (h_sum_5_smallest : (numbers.sort (· ≤ ·)).take 5).sum = 0.26 * numbers.sum :
  numbers.length = 18 :=
sorry

end number_of_numbers_on_board_l363_363267


namespace find_m_l363_363618

noncomputable theory
open_locale classical

-- Definitions representing the given conditions
def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ} {q : ℝ} {m : ℕ}

-- Conditions given in the problem
axiom a1_eq_1 : a 1 = 1
axiom geometric : geometric_seq a q
axiom q_ne_one : |q| ≠ 1
axiom am_eq_product : a m = a 1 * a 2 * a 3 * a 4 * a 5

-- Theorem we need to prove
theorem find_m :
  m = 11 :=
sorry

end find_m_l363_363618


namespace original_population_multiple_of_3_l363_363749

theorem original_population_multiple_of_3 (x y z : ℕ) (h1 : x^2 + 121 = y^2) (h2 : y^2 + 121 = z^2) :
  3 ∣ x^2 :=
sorry

end original_population_multiple_of_3_l363_363749


namespace quadratic_has_real_root_iff_b_in_interval_l363_363542

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l363_363542


namespace problem1_problem2_l363_363092

open Set

variable {α : Type*} [Preorder α]

/- Conditions -/
def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 3 < x ∧ x ≤ 7}

-- Problem 1: Prove A ∩ B = {x | 3 < x < 5}
theorem problem1 : A ∩ B = {x | 3 < x ∧ x < 5} :=
sorry

-- Problem 2: Prove (compl U A) ∪ B = [1, 2) ∪ (3, 7]
def compl_U_A : Set ℝ := U \ A
def compl_U_A_union_B : Set ℝ := compl_U_A ∪ B

theorem problem2 : compl_U_A_union_B = {x | (1 ≤ x ∧ x < 2) ∨ (3 < x ∧ x ≤ 7)} :=
sorry

end problem1_problem2_l363_363092


namespace number_of_outfits_l363_363785

-- Define the counts of each item
def redShirts : Nat := 6
def greenShirts : Nat := 4
def pants : Nat := 7
def greenHats : Nat := 10
def redHats : Nat := 9

-- Total number of outfits satisfying the conditions
theorem number_of_outfits :
  (redShirts * greenHats * pants) + (greenShirts * redHats * pants) = 672 :=
by
  sorry

end number_of_outfits_l363_363785


namespace cos_alpha_beta_l363_363441

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x * (sin x) ^ 2 - (1 / 2)

theorem cos_alpha_beta :
  ∀ (α β : ℝ), 
    (0 < α ∧ α < π / 2) →
    (0 < β ∧ β < π / 2) →
    f (α / 2) = sqrt 5 / 5 →
    f (β / 2) = 3 * sqrt 10 / 10 →
    cos (α - β) = sqrt 2 / 2 :=
by
  intros α β hα hβ h1 h2
  sorry

end cos_alpha_beta_l363_363441


namespace complement_A_is_closed_interval_l363_363518

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define the set A with the given condition
def A : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the complement of A with respect to U
def complement_A : Set ℝ := Set.compl A

theorem complement_A_is_closed_interval :
  complement_A = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry  -- Proof to be inserted

end complement_A_is_closed_interval_l363_363518


namespace todd_repay_amount_l363_363760

-- Definitions based on conditions
def initial_loan : ℝ := 100
def ingredients_cost : ℝ := 75
def snow_cones_sold : ℕ := 200
def price_per_snow_cone : ℝ := 0.75
def remaining_amount_after_repay : ℝ := 65

-- Proof statement
theorem todd_repay_amount :
  let total_sales := snow_cones_sold * price_per_snow_cone in
  let money_before_repay := total_sales - ingredients_cost in
  let total_money := money_before_repay + initial_loan in
  total_money - remaining_amount_after_repay = 110 :=
by {
  sorry
}

end todd_repay_amount_l363_363760


namespace general_term_sum_first_n_terms_l363_363640

section

-- Define the sequence function a_n recursively
def a : ℕ → ℕ
| 1 := 4
| (n+1) := (n+2)*(2*(n+1)^2 + 2*(n+1))/(n+1)  -- Using the given recursive relation

-- Prove the general term formula
theorem general_term (n : ℕ) (h : n > 0) : a n = 2*n^2 + 2*n :=
by
  -- Proof would go here
  sorry

-- Define the sum of the sequence 1/a_n
def sum_of_reciprocals (n : ℕ) : ℝ :=
∑ i in finset.range (n+1), (1 / (a (i+1) : ℝ))

-- Prove the sum of the first n terms of the sequence 1/a_n
theorem sum_first_n_terms (n : ℕ) : sum_of_reciprocals n = n / (2 * (n + 1)) :=
by
  -- Proof would go here
  sorry

end

end general_term_sum_first_n_terms_l363_363640


namespace problem_quadratic_has_real_root_l363_363603

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l363_363603


namespace tetrahedron_opposite_angles_l363_363990

noncomputable def angle_opposite_edges (a b c : ℝ) : (ℝ × ℝ × ℝ) :=
( real.arccos (|b^2 - a^2| / c^2), 
  real.arccos (|a^2 - c^2| / b^2), 
  real.arccos (|c^2 - b^2| / a^2))

theorem tetrahedron_opposite_angles (a b c : ℝ) (h1 : AB = CD = a) (h2 : BC = DA = b) (h3 : CA = BD = c) :
  angle_opposite_edges a b c = 
    ( real.arccos ( abs(b^2 - a^2) / c^2), 
      real.arccos ( abs(a^2 - c^2) / b^2), 
      real.arccos ( abs(c^2 - b^2) / a^2) ) :=
sorry

end tetrahedron_opposite_angles_l363_363990


namespace hostel_cost_l363_363330

def first_week_rate : ℝ := 18
def additional_week_rate : ℝ := 12
def first_week_days : ℕ := 7
def total_days : ℕ := 23

theorem hostel_cost :
  (first_week_days * first_week_rate + 
  (total_days - first_week_days) / first_week_days * first_week_days * additional_week_rate + 
  (total_days - first_week_days) % first_week_days * additional_week_rate) = 318 := 
by
  sorry

end hostel_cost_l363_363330


namespace cos_double_angle_of_tangent_is_2_l363_363495

theorem cos_double_angle_of_tangent_is_2
  (θ : ℝ)
  (h_tan : Real.tan θ = 2) :
  Real.cos (2 * θ) = -3 / 5 := 
by
  sorry

end cos_double_angle_of_tangent_is_2_l363_363495


namespace desired_profit_is_8000_l363_363361

-- Define the conditions
def cost_per_program : ℝ := 0.70
def num_programs : ℕ := 35000
def selling_price_per_program : ℝ := 0.50
def advertisement_revenue : ℝ := 15000.0

-- Translate the proof problem into a Lean 4 statement
theorem desired_profit_is_8000 :
  let total_production_cost := cost_per_program * num_programs
  let total_sales_revenue := selling_price_per_program * num_programs
  let total_revenue := total_sales_revenue + advertisement_revenue
  let desired_profit := total_revenue - total_production_cost
  in desired_profit = 8000 := by
  -- State the proof is omitted
  sorry

end desired_profit_is_8000_l363_363361


namespace twin_primes_iff_divisibility_l363_363291

open Nat

theorem twin_primes_iff_divisibility (n : ℕ) (h1 : Odd n) (h2 : Prime n) (h3 : Prime (n + 2)) :
  (Prime (n + 2)) ↔ (n * (n + 2) ∣ 4 * ((factorial (n - 1)) + 1) + n) :=
by
  sorry

end twin_primes_iff_divisibility_l363_363291


namespace ratio_of_a_to_b_l363_363788

open Nat

def ratio_of_amounts (A B C : ℕ) : ℕ × ℕ :=
  (A / gcd A B, B / gcd A B)

theorem ratio_of_a_to_b (A B C : ℕ) (h1 : A + B + C = 645) (h2 : B = C + 25) (h3 : B = 134) :
  ratio_of_amounts A B C = (3, 1) := by
  sorry

end ratio_of_a_to_b_l363_363788


namespace proposition_correctness_l363_363366

theorem proposition_correctness :
  ∀ {A B : Type} 
    (f : A → B) 
    (P1 : ¬ ∀ x1 x2: ℝ, f x1 = f x2 → x1 = x2)
    (P2 : ∀ (f : A → B), (∀ x1 x2 : A, f x1 = f x2 → x1 = x2) → (∀ x1 x2 : A, x1 ≠ x2 → f x1 ≠ f x2))
    (P3 : ∀ (f : A → B), (∀ x1 x2 : A, f x1 = f x2 → x1 = x2) → (∀ b : B, ∃! a : A, f a = b))
    (P4 : ¬ ∀ (f : ℝ → ℝ) (I : set ℝ) (h : is_monotone_on f I), ∀ x1 x2 : ℝ, f x1 = f x2 → x1 = x2), 
  P2 ∧ P3 :=
by
  sorry

end proposition_correctness_l363_363366


namespace sum_first_12_even_numbers_l363_363335

theorem sum_first_12_even_numbers : 
  let n := 12
  let a := 2
  let l := (a + (n - 1) * 2)
  let sum := n / 2 * (a + l)
  sum = 156 :=
by
  have n := 12
  have a := 2
  have l := (a + (n - 1) * 2)
  have sum := n / 2 * (a + l)
  show sum = 156
  sorry

end sum_first_12_even_numbers_l363_363335


namespace point_A_inside_circle_l363_363115

variables (O A : Type) (r d : ℝ)

-- Given conditions as assumptions
def circle_radius : ℝ := 5
def distance_to_center : ℝ := 4

-- Proof statement
theorem point_A_inside_circle (h1 : r = circle_radius) (h2 : d = distance_to_center) : d < r := 
by 
  rw [h1, h2]
  exact lt_add_one _

end point_A_inside_circle_l363_363115


namespace max_value_expression_l363_363480

noncomputable def polynomial_roots {α : Type*} [LinearOrderedField α] (a b c : α) (λ : α)
    (x1 x2 x3 : α) : Prop :=
  x1 + x2 + x3 = -a ∧
  x1 * x2 + x2 * x3 + x3 * x1 = b ∧
  x1 * x2 * x3 = -c

theorem max_value_expression {α : Type*} [LinearOrderedField α] {a b c λ x1 x2 x3 : α}
  (h_roots : polynomial_roots a b c λ x1 x2 x3)
  (h_pos : λ > 0)
  (h_diff : x2 - x1 = λ)
  (h_cond : x3 > (1 / 2) * (x1 + x2)) :
  (2 * a^3 + 27 * c - 9 * a * b) / (λ^3) = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end max_value_expression_l363_363480


namespace fuel_A_volume_l363_363398

-- Let V_A and V_B be defined as the volumes of fuel A and B respectively.
def V_A : ℝ := sorry
def V_B : ℝ := sorry

-- Given conditions:
axiom h1 : V_A + V_B = 214
axiom h2 : 0.12 * V_A + 0.16 * V_B = 30

-- Prove that the volume of fuel A added, V_A, is 106 gallons.
theorem fuel_A_volume : V_A = 106 := 
by
  sorry

end fuel_A_volume_l363_363398


namespace troy_needs_more_money_to_buy_computer_l363_363284

theorem troy_needs_more_money_to_buy_computer :
  ∀ (price_new_computer savings sale_old_computer : ℕ),
  price_new_computer = 80 →
  savings = 50 →
  sale_old_computer = 20 →
  (price_new_computer - (savings + sale_old_computer)) = 10 :=
by
  intros price_new_computer savings sale_old_computer Hprice Hsavings Hsale
  sorry

end troy_needs_more_money_to_buy_computer_l363_363284


namespace quadratic_root_shift_c_value_l363_363663

theorem quadratic_root_shift_c_value
  (r s : ℝ)
  (h1 : r + s = 2)
  (h2 : r * s = -5) :
  ∃ b : ℝ, x^2 + b * x - 2 = 0 :=
by
  sorry

end quadratic_root_shift_c_value_l363_363663


namespace only_prime_n_has_cyclic_permutation_complete_residue_system_l363_363008

open Nat

def is_cyclic_permutation_complete_residue_system (n : ℕ) : Prop :=
  ∃ (a : ℕ → ℕ) (ha : ∀ i, a i < n) (ha_perm : ∀ i, ∃ j, a j = i + 1), -- a is a permutation of (1, 2, ..., n)
    (finset.range n).map ⟨λ k, (list.prod (list.range k).map a) % n, sorry⟩ = finset.range n

theorem only_prime_n_has_cyclic_permutation_complete_residue_system (n : ℕ) :
  is_cyclic_permutation_complete_residue_system n → nat.prime n := sorry

end only_prime_n_has_cyclic_permutation_complete_residue_system_l363_363008


namespace perfect_square_trinomial_l363_363107

theorem perfect_square_trinomial (y : ℝ) (m : ℝ) : 
  (∃ b : ℝ, y^2 - m*y + 9 = (y + b)^2) → (m = 6 ∨ m = -6) :=
by
  intro h
  sorry

end perfect_square_trinomial_l363_363107


namespace point_not_on_graph_l363_363468

theorem point_not_on_graph : 
  ∀ (k : ℝ), (k ≠ 0) → (∀ x y : ℝ, y = k * x → (x, y) = (1, 2)) → ¬ (∀ x y : ℝ, y = k * x → (x, y) = (1, -2)) :=
by
  sorry

end point_not_on_graph_l363_363468


namespace complex_number_solution_l363_363487

theorem complex_number_solution (z : ℂ) (i : ℂ) (h1 : i * z = (1 - 2 * i) ^ 2) (h2 : i * i = -1) : z = -4 + 3 * i := by
  sorry

end complex_number_solution_l363_363487


namespace projection_vector_l363_363927

variable (a b : ℝ)
variable (a_vec b_vec : EuclideanSpace ℝ (Fin 2))
variable [innerProductSpace ℝ (EuclideanSpace ℝ (Fin 2))]

noncomputable
def angle_condition : Prop := real.angle a_vec b_vec = real.pi / 3

noncomputable
def magnitude_condition : Prop := ‖a_vec - 2 • b_vec‖ = ‖a_vec + b_vec‖

theorem projection_vector (ha : angle_condition a_vec b_vec) (hb : magnitude_condition a_vec b_vec) : 
  (innerProductSpace.proj b_vec a_vec) = (1/2) • b_vec := 
sorry

end projection_vector_l363_363927


namespace ratio_expression_value_l363_363965

theorem ratio_expression_value (p q s u : ℚ) (h1 : p / q = 5 / 2) (h2 : s / u = 11 / 7) : 
  (5 * p * s - 3 * q * u) / (7 * q * u - 2 * p * s) = -233 / 12 :=
by {
  -- Proof will be provided here.
  sorry
}

end ratio_expression_value_l363_363965


namespace perpendicular_sufficient_not_necessary_l363_363947

theorem perpendicular_sufficient_not_necessary :
  (∀ k : ℝ, (l1_perp_l2: Prop) → k = 2 → l1_perp_l2) ∧ 
  (∃ k : ℝ, k ≠ 2 ∧ l1_perp_l2) :=
by 
  let l1 := λ x : ℝ, -(1/4)*x - 1
  let l2 := λ k x : ℝ, (k^2)*x - 2
  def l1_perp_l2 (k : ℝ) : Prop := (-1/4) * (k^2) = -1

  have sufficient : ∀ k : ℝ, k = 2 → l1_perp_l2 k,
  from sorry,

  have not_necessary : ∃ k : ℝ, k ≠ 2 ∧ l1_perp_l2 k,
  from sorry,

  exact ⟨sufficient, not_necessary⟩

end perpendicular_sufficient_not_necessary_l363_363947


namespace minimum_moves_to_arrange_pieces_l363_363338

/-- Representing the board state and the logic for minimum moves to arrange pieces. -/
structure BoardState :=
  (row1 : list ℕ := [1, 2])
  (row2 : list ℕ := [1, 3])
  (row3 : list ℕ := [3, 4])
  (row4 : list ℕ := [2, 4])

/-- Adjacent cell condition means moving vertically, horizontally, or diagonally. -/
def adjacent_cells (x y : ℕ × ℕ) : Prop :=
  abs (x.fst - y.fst) ≤ 1 ∧ abs (x.snd - y.snd) ≤ 1

/-- The main theorem stating the minimum number of moves required to achieve the goal. -/
theorem minimum_moves_to_arrange_pieces (b : BoardState) : 
  ∃ m ≥ 0, (∀ r, r ∈ [b.row1, b.row2, b.row3, b.row4] → list.length r = 2) 
  ∧ (∀ c, c ∈ [[1], [2], [3], [4]] → list.length (filter (λ r, c ∈ r) [b.row1, b.row2, b.row3, b.row4]) = 2) 
  ∧ m = 2 :=
  by
    sorry

end minimum_moves_to_arrange_pieces_l363_363338


namespace quadratic_real_roots_l363_363561

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l363_363561


namespace complex_magnitude_l363_363673

noncomputable def root_two : ℝ := real.sqrt 2

theorem complex_magnitude :
  let i := complex.I in
  ∥(5 + 3 * i) / (4 - i)∥ = root_two :=
by
  clear _let_match_i
  sorry

end complex_magnitude_l363_363673


namespace shift_right_by_one_l363_363703

theorem shift_right_by_one (x : ℝ) : 2^(x + 1 - 1) = 2^x := by 
  sorry

end shift_right_by_one_l363_363703


namespace sum_of_positive_differences_l363_363161

theorem sum_of_positive_differences : 
  let T := {0, 1, 2, 3, 4, 5, 6, 7}.map (fun x => 3^x) in
  (∑ i in Finset.range 8, i * 3^i) - (∑ i in Finset.range 8, 3^i * (7 - i)) = 20324 :=
sorry

end sum_of_positive_differences_l363_363161


namespace sum_of_distinct_numbers_with_8_trailing_zeros_and_90_divisors_l363_363290

theorem sum_of_distinct_numbers_with_8_trailing_zeros_and_90_divisors :
  ∃ N1 N2 : ℕ, 
    (N1 ≠ N2) ∧
    (∃ k1 k2 : ℕ, N1 = 10^8 * k1 ∧ N2 = 10^8 * k2 ∧
     (∏ (d : ℕ) in (range (succ N1)), (d ∣ N1) ∧ N1 % d = 0).card = 90) ∧
    (∏ (d : ℕ) in (range (succ N2)), (d ∣ N2) ∧ N2 % d = 0).card = 90 →
    N1 + N2 = 700000000 :=
by
  sorry

end sum_of_distinct_numbers_with_8_trailing_zeros_and_90_divisors_l363_363290


namespace degree_of_h_l363_363668

open Polynomial

-- Define the polynomial f(x) given in the conditions
def f : Polynomial ℝ := -6 * X^5 + 2 * X^4 + 5 * X^2 - 4

-- Define the condition that f(x) + h(x) should have degree 2
def h (p : Polynomial ℝ) : Prop := (f + p).degree = 2

-- The proof problem
theorem degree_of_h :
  ∃ (p : Polynomial ℝ), h p ∧ p.degree = 5 :=
sorry

end degree_of_h_l363_363668


namespace median_successful_free_throws_best_free_throw_percentage_l363_363355

-- Conditions: list of successful free throws and list of attempted free throws
def successful_free_throws : List ℕ := [8, 17, 15, 22, 14, 12, 24, 10, 20, 16]
def attempted_free_throws  : List ℕ := [10, 20, 18, 25, 16, 15, 27, 12, 22, 19]

-- Convert the lists to floats for percentage calculation
def successful_free_throws_float : List ℚ := successful_free_throws.map (λ x, x)
def attempted_free_throws_float : List ℚ := attempted_free_throws.map (λ x, x)

-- The shooting percentages for the respective games
def shooting_percentages : List ℚ :=
  (List.zipWith (λ s a => (s / a) * 100) successful_free_throws_float attempted_free_throws_float)

-- Proof problem 1: Prove the median of successful free throws is 15.5
theorem median_successful_free_throws :
  (List.median successful_free_throws) = 15.5 :=
by
  sorry

-- Proof problem 2: Prove the best free-throw shooting percentage game is 90.91%
theorem best_free_throw_percentage :
    List.maximum shooting_percentages = some 90.91 :=
by
  sorry

end median_successful_free_throws_best_free_throw_percentage_l363_363355


namespace determinant_of_matrix_l363_363411

-- Define the 2x2 matrix given in the problem
def mat : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, x], ![-3, 9]]

-- Define the determinant expression we need to prove
def det_expected : ℝ := 45 + 3 * x

-- Define the proof problem statement
theorem determinant_of_matrix : Matrix.det mat = det_expected := by 
  sorry

end determinant_of_matrix_l363_363411


namespace volume_of_sphere_from_area_l363_363492

-- Define the radius based on the given area of the main view
def radius_from_area (A : ℝ) : ℝ := sqrt (A / π)

-- Define the volume of the sphere based on the radius
def volume_of_sphere (R : ℝ) : ℝ := (4 / 3) * π * R^3

-- Given that the area of the sphere's main view is 9π, prove that the volume is 36π
theorem volume_of_sphere_from_area : volume_of_sphere (radius_from_area (9 * π)) = 36 * π := by
  sorry

end volume_of_sphere_from_area_l363_363492


namespace total_cost_is_correct_l363_363835

def bus_ride_cost : ℝ := 1.75
def train_ride_cost : ℝ := bus_ride_cost + 6.35
def total_cost : ℝ := bus_ride_cost + train_ride_cost

theorem total_cost_is_correct : total_cost = 9.85 :=
by
  -- proof here
  sorry

end total_cost_is_correct_l363_363835


namespace trigonometric_solution_count_l363_363527

theorem trigonometric_solution_count : 
  let count := (finset.filter (λ x : ℝ, x * (π / 180) ∈ [0, 2*π) ∧ real.sin (x * (π / 180)) = -1/2 ∧ real.cos (x * (π / 180)) = (real.sqrt 3) / 2) (finset.range 360)).card
  in count = 1 :=
by
  -- Definitions for sine and cosine values
  let sin_neg_half := -1 / 2
  let cos_sqrt_3_over_2 := (real.sqrt 3) / 2

  -- Define the filter condition
  let condition x := x * (π / 180) ∈ [0, 2*π) ∧ real.sin (x * (π / 180)) = sin_neg_half ∧ real.cos (x * (π / 180)) = cos_sqrt_3_over_2

  -- Define the count of valid x values
  let count := (finset.filter condition (finset.range 360)).card

  -- Assert the count is 1
  show count = 1,
  sorry

end trigonometric_solution_count_l363_363527


namespace find_digit_l363_363607

-- Definitions from the conditions
def digit_positions := (9, 6, 5, n, 8, 4, 3, 2)

def odd_position_sum := 9 + 5 + 8 + 2 -- sum of digits in odd positions
def even_position_sum (n : ℕ) := 6 + n + 4 + 3 -- sum of digits in even positions

-- The main theorem
theorem find_digit (n : ℕ) (h : n < 10) : 
  (odd_position_sum - even_position_sum n) % 11 = 0 -> n = 1 :=
by
  -- Proof would go here
  sorry

end find_digit_l363_363607


namespace cake_stand_cost_calculation_l363_363026

-- Define the constants given in the problem
def flour_cost : ℕ := 5
def money_given : ℕ := 43
def change_received : ℕ := 10

-- Define the cost of the cake stand based on the problem's conditions
def cake_stand_cost : ℕ := (money_given - change_received) - flour_cost

-- The theorem we want to prove
theorem cake_stand_cost_calculation : cake_stand_cost = 28 :=
by
  sorry

end cake_stand_cost_calculation_l363_363026


namespace find_S9_l363_363061

variables {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Condition: an arithmetic sequence with the sum of first n terms S_n.
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition: a_3 + a_4 + a_5 + a_6 + a_7 = 20.
def given_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 20

-- The sum of the first n terms.
def sum_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n / 2 : ℝ) * (a 1 + a n)

-- Prove that S_9 = 36.
theorem find_S9 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arithmetic_sequence : arithmetic_sequence a) 
  (h_given_condition : given_condition a)
  (h_sum_terms : sum_terms S a) : 
  S 9 = 36 :=
sorry

end find_S9_l363_363061


namespace concurrency_of_am_dn_xy_l363_363337

variables (A B C D X Y Z P M N : Point)
variables [InLine A B C D] [Intersection (Circle.diameter A C) (Circle.diameter B D) X Y]
variables [IntersectionLine (Line XY) (Line BC) Z]
variables [OnLineWithException P (Line XY) Z]
variables [IntersectionLine (Line CP) (Circle.diameter A C) C M]
variables [IntersectionLine (Line BP) (Circle.diameter B D) B N]

theorem concurrency_of_am_dn_xy :
  Concurrent (Line AM) (Line DN) (Line XY) :=
sorry

end concurrency_of_am_dn_xy_l363_363337


namespace original_radius_l363_363646

theorem original_radius (r z : ℝ) (h : r > 0) :
  (let V := π * r^2 * 4 in
   let V₁ := π * (r + 4)^2 * 4 in
   let V₂ := π * r^2 * 9 in
   V₁ - V = z ∧ V₂ - V = z) →
   r = 8 :=
begin
  intros h₁,
  -- Translate conditions to Lean expressions
  have hV : π * r^2 * 4 = V := rfl,
  have hV₁ : π * (r + 4)^2 * 4 = V₁ := rfl,
  have hV₂ : π * r^2 * 9 = V₂ := rfl,
  
  -- Proceed with the given condition statement
  cases h₁ with hz₁ hz₂,
  rw [hV₁, hV] at hz₁,
  rw [hV₂, hV] at hz₂,
  
  -- Use the simplified condition where V₁ - V = V₂ - V
  have eq1 := hz₁,
  have eq2 := hz₂,
  
  -- Further steps to derive r = 8 would go here
  sorry
end

end original_radius_l363_363646


namespace ratio_of_other_triangle_to_square_l363_363372

noncomputable def ratio_of_triangle_areas (m : ℝ) : ℝ :=
  let side_of_square := 2
  let area_of_square := side_of_square ^ 2
  let area_of_smaller_triangle := m * area_of_square
  let r := area_of_smaller_triangle / (side_of_square / 2)
  let s := side_of_square * side_of_square / r
  let area_of_other_triangle := side_of_square * s / 2
  area_of_other_triangle / area_of_square

theorem ratio_of_other_triangle_to_square (m : ℝ) (h : m > 0) :
  ratio_of_triangle_areas m = 1 / (4 * m) :=
sorry

end ratio_of_other_triangle_to_square_l363_363372


namespace find_M_and_ellipse_equation_l363_363476

open Real

-- Definitions based on the given conditions.
def ellipse_center_origin (x y : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def line_intersects_ellipse (x y : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), (A.1 = x ∧ A.2 = y ∧ (A.2 = sqrt 3 * (A.1 + 1))) ∧
                   (B.1 = x ∧ B.2 = y ∧ (B.2 = sqrt 3 * (B.1 + 1))) ∧
                   (abs (B.1 - A.1) + abs (B.2 - A.2) = 2)

def midpoint_M_at_distance_1 (x y : ℝ) : Prop :=
  ∃ (M : ℝ × ℝ), (M.1 = x ∧ M.2 = y ∧ dist (0, 0) M = 1)

-- Stating the proof problem
theorem find_M_and_ellipse_equation :
  ∃ (M : ℝ × ℝ) a b,
    midpoint_M_at_distance_1 (-1/2) (sqrt 3 / 2) ∧
    line_intersects_ellipse M.1 M.2 ∧
    ellipse_center_origin M.1 M.2 ∧
    (abs (A.1 - B.1) + abs (A.2 - B.2) = 2) ∧
    (a = 1 ∧ b = sqrt 3) ∧ 
    (x^2 / a^2 + y^2 / b^2 = 1) := sorry

end find_M_and_ellipse_equation_l363_363476


namespace correlation_comparison_l363_363735

/-- The data for variables x and y are (1, 3), (2, 5.3), (3, 6.9), (4, 9.1), and (5, 10.8) -/
def xy_data : List (Int × Float) := [(1, 3), (2, 5.3), (3, 6.9), (4, 9.1), (5, 10.8)]

/-- The data for variables U and V are (1, 12.7), (2, 10.2), (3, 7), (4, 3.6), and (5, 1) -/
def UV_data : List (Int × Float) := [(1, 12.7), (2, 10.2), (3, 7), (4, 3.6), (5, 1)]

/-- r1 is the linear correlation coefficient between y and x -/
noncomputable def r1 : Float := sorry

/-- r2 is the linear correlation coefficient between V and U -/
noncomputable def r2 : Float := sorry

/-- The problem is to prove that r2 < 0 < r1 given the data conditions -/
theorem correlation_comparison : r2 < 0 ∧ 0 < r1 := 
by 
  sorry

end correlation_comparison_l363_363735


namespace num_pens_l363_363811

theorem num_pens (pencils : ℕ) (students : ℕ) (pens : ℕ)
  (h_pencils : pencils = 520)
  (h_students : students = 40)
  (h_div : pencils % students = 0)
  (h_pens_per_student : pens = (pencils / students) * students) :
  pens = 520 := by
  sorry

end num_pens_l363_363811


namespace least_k_value_l363_363971

theorem least_k_value (k : ℤ) (h : 0.000010101 * 10^k > 10000) : k ≥ 9 :=
sorry

end least_k_value_l363_363971


namespace samantha_tenth_finger_l363_363216

def g (x : ℕ) : ℕ :=
  match x with
  | 2 => 2
  | _ => 0  -- Assume a simple piecewise definition for the sake of the example.

theorem samantha_tenth_finger : g (2) = 2 :=
by  sorry

end samantha_tenth_finger_l363_363216


namespace negation_of_p_is_sufficient_but_not_necessary_for_negation_of_q_l363_363064

theorem negation_of_p_is_sufficient_but_not_necessary_for_negation_of_q
  (x : ℝ) (p : Prop) (q : Prop)
  (h1 : p ↔ |x + 1| > 2)
  (h2 : q ↔ 5x - 6 > x^2) :
  (¬p → ¬q) ∧ ¬(¬q → ¬p) :=
by {
  sorry
}

end negation_of_p_is_sufficient_but_not_necessary_for_negation_of_q_l363_363064


namespace value_of_3y_l363_363931

theorem value_of_3y (x y z : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 11) (h4 : z = 3) :
  3 * y = 12 :=
by
  sorry

end value_of_3y_l363_363931


namespace quadratic_real_roots_l363_363578

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l363_363578


namespace find_m_for_pure_imaginary_l363_363111

noncomputable def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem find_m_for_pure_imaginary (m : ℝ) : 
  is_pure_imaginary ((1 + m * Complex.i) * (2 - Complex.i)) →
  m = -2 :=
by
  sorry

end find_m_for_pure_imaginary_l363_363111


namespace range_of_a_l363_363482

noncomputable def A (x : ℝ) : Prop := x < -2 ∨ x ≥ 1
noncomputable def B (x : ℝ) (a : ℝ) : Prop := x ≥ a

theorem range_of_a (a : ℝ) : (∀ x, A x ∨ B x a) ↔ a ≤ -2 :=
by sorry

end range_of_a_l363_363482


namespace average_age_of_school_l363_363988

def total_students : ℕ := 632
def avg_age_boys : ℕ := 12
def avg_age_girls : ℕ := 11
def total_girls : ℕ := 158

def total_boys : ℕ := total_students - total_girls
def total_age_boys : ℕ := avg_age_boys * total_boys
def total_age_girls : ℕ := avg_age_girls * total_girls
def total_age_students : ℕ := total_age_boys + total_age_girls

def avg_age_school : ℕ := total_age_students / total_students

theorem average_age_of_school :
  avg_age_school ≈ 12 * 474 + 11 * 158 / 632 := 
sorry

end average_age_of_school_l363_363988


namespace area_of_BDE_l363_363136

/-- Let \(ABC\) be a right triangle with \(\angle C = 90^\circ\). Let \(D\) be the midpoint of \(\overline{AB}\) and let \(DE \perp AB\). Given that \(\overline{AB} = 24\) and \(\overline{AC} = 10\), the area of triangle \(BDE\) is \(\frac{360}{\sqrt{119}}\). -/
theorem area_of_BDE (A B C D E : Point) :
  ∠C = 90 ∧
  AD = DB ∧
  on_line D A B ∧
  on_line E D AB ∧
  dist A B = 24 ∧
  dist A C = 10 →
  area_of_triangle B D E = 360 / sqrt 119 :=
begin
  sorry
end

end area_of_BDE_l363_363136


namespace ellipse_standard_eq_and_line_exists_l363_363474

-- Define the ellipse
def ellipse (a b : ℝ) := ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Define focus condition
def focus1 (a b c : ℝ) := c = 1 ∧ c^2 = a^2 - b^2

-- Define origin distance to line B1F1
def distance_O_to_B1F1 (a b : ℝ) := ∀ x y : ℝ, abs(b * x - y + b) / sqrt(b^2 + 1) = sqrt(3) / 2

-- Define point T condition
def point_T_on_circle (x y : ℝ) := x^2 + y^2 = 2

-- Define the proof problem
theorem ellipse_standard_eq_and_line_exists (a b : ℝ) (ellipseC : ellipse a b) (focusCond : focus1 a b 1) (distCond : distance_O_to_B1F1 a b) :
  (a^2 - b^2 = 1) ∧ (b = sqrt(3)) ∧ (a = 2) ∧ ellipse a b :=
  sorry

end ellipse_standard_eq_and_line_exists_l363_363474


namespace quadratic_has_real_root_l363_363568

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l363_363568


namespace intersection_card_eq_two_l363_363069

open Set

def setA : Set ℤ := {x | -1 ≤ x ∧ x ≤ 2}
def setB : Set ℤ := {x | 2^x < 1}

theorem intersection_card_eq_two :
  ∃ (A B : Set ℤ), A = setA ∧ B = setB ∧ card (A ∩ B) = 2 :=
by
  sorry

end intersection_card_eq_two_l363_363069


namespace volume_of_tetrahedron_P_QRS_l363_363385

theorem volume_of_tetrahedron_P_QRS 
  (P Q R S : ℝ)
  (base_area : ℝ)
  (height_PP' : ℝ) 
  (unit_cube : bool) 
  (diagonal_cut : bool)
  (further_dissection : bool)
  (base_triangle : bool)
  (right_triangle_legs : bool) 
  (leg_length : ℝ) 
  (height : ℝ) 
  (volume : ℝ) : 
  unit_cube = true →
  diagonal_cut = true →
  further_dissection = true →
  base_triangle = true →
  right_triangle_legs = true →
  leg_length = 0.5 →
  height = 0.5 →
  base_area = (1 / 2) * (1 / 2) * (1 / 2) →
  height_PP' = 0.5 →
  volume = (1 / 3) * base_area * height_PP' →
  volume = 1 / 48 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end volume_of_tetrahedron_P_QRS_l363_363385


namespace probability_units_digit_three_l363_363871

def is_valid_sum (n m : ℕ) : Prop :=
  (n + m) % 10 = 3

def favorable_outcomes : list (ℕ × ℕ) :=
  [(1, 2), (2, 1), (1, 12), (2, 11), (3, 10), (4, 9), (5, 8),
   (6, 7), (7, 6), (8, 5), (9, 4), (10, 3), (11, 2), (12, 1),
   (11, 12), (12, 11)]

theorem probability_units_digit_three :
  let total_outcomes := 144 in
  let favorable_count := 16 in
  favorable_outcomes.length = favorable_count ∧
  (favorable_count / total_outcomes : ℚ) = 1 / 9 :=
by sorry

end probability_units_digit_three_l363_363871


namespace quadratic_roots_interval_l363_363594

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l363_363594


namespace boarding_students_l363_363981

def total_students (x : ℕ) : ℕ := 4 * x + 10

theorem boarding_students (x : ℕ) :
    4 * x + 10 ≥ 6 * (x - 1) + 1 ∧ 
    4 * x + 10 ≤ 6 * (x - 1) + 5 ∧ 
    (x = 6 ∨ x = 7)
    → (total_students x = 34 ∨ total_students x = 38) :=
by {
    intro h,
    cases h.2.2,
    { rw h_1, simp [total_students] },
    { rw h_1, simp [total_students] }
}

end boarding_students_l363_363981


namespace incorrect_conclusion_B_l363_363862

-- Define the function f and the conditions given
def f : ℝ → ℝ := sorry

-- Assume the conditions
axiom f_at_0 : f 0 = -1
axiom f_prime_gt_k (k : ℝ) (h_k : k > 1) : ∀ x, f' x > k

-- The proof that f(1/(k-1)) ≥ 1/(k-1) given the conditions
theorem incorrect_conclusion_B (k : ℝ) (h_k : k > 1) : f (1 / (k - 1)) ≥ 1 / (k - 1) :=
sorry

end incorrect_conclusion_B_l363_363862


namespace triangle_geometry_l363_363622

theorem triangle_geometry 
  (A B C O D E F M : Point)
  (hABCacute : AcuteTriangle A B C)
  (hABAC : AB > AC)
  (hO : Circumcenter O A B C)
  (hD : Midpoint D B C)
  (hCircle : Circle (LineSegment A D) (Intersects E F))
  (hE : Intersects E (LineSegment A B (Circle (LineSegment A D))))
  (hF : Intersects F (LineSegment A C (Circle (LineSegment A D))))
  (hParallel : Parallel (Line D M) (Line A O))
  (hM : Intersects M (LineSegment E F)) :
  (Distance E M = Distance M F) :=
sorry

end triangle_geometry_l363_363622


namespace cricketer_total_score_l363_363815

variable (T : ℝ)
variable (boundaries sixes : ℝ)
variable (percentage : ℝ)

-- Given conditions
def runs_from_boundaries := 12 * 4
def runs_from_sixes := 2 * 6
def total_runs_from_boundaries_and_sixes := runs_from_boundaries + runs_from_sixes
def percentage_runs_between_wickets := 60.526315789473685 / 100

-- Lean statement to prove
theorem cricketer_total_score (h1 : T = total_runs_from_boundaries_and_sixes + percentage_runs_between_wickets * T) :
  T ≈ 152 :=
by
  sorry -- Proof omitted

end cricketer_total_score_l363_363815


namespace find_a_plus_b_l363_363047

variables {a b : ℝ}

theorem find_a_plus_b (h1 : a - b = -3) (h2 : a * b = 2) : a + b = Real.sqrt 17 ∨ a + b = -Real.sqrt 17 := by
  -- Proof can be filled in here
  sorry

end find_a_plus_b_l363_363047


namespace prob_lamp_first_factory_standard_prob_lamp_standard_l363_363424

noncomputable def P_B1 : ℝ := 0.35
noncomputable def P_B2 : ℝ := 0.50
noncomputable def P_B3 : ℝ := 0.15

noncomputable def P_B1_A : ℝ := 0.70
noncomputable def P_B2_A : ℝ := 0.80
noncomputable def P_B3_A : ℝ := 0.90

-- Question A
theorem prob_lamp_first_factory_standard : P_B1 * P_B1_A = 0.245 :=
by 
  sorry

-- Question B
theorem prob_lamp_standard : (P_B1 * P_B1_A) + (P_B2 * P_B2_A) + (P_B3 * P_B3_A) = 0.78 :=
by 
  sorry

end prob_lamp_first_factory_standard_prob_lamp_standard_l363_363424


namespace Z_in_first_quadrant_l363_363994

-- Define a condition checking if a complex number is in the first quadrant
def is_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

-- Given condition
def Z : ℂ := (5 + 4 * complex.I) + (-1 + 2 * complex.I)

-- Theorem statement that needs proof
theorem Z_in_first_quadrant : is_first_quadrant Z :=
by
  sorry

end Z_in_first_quadrant_l363_363994


namespace horner_method_operations_l363_363499

def f (x : ℝ) : ℝ := x^6 + 1

theorem horner_method_operations : ∃ (exp mult add : ℕ), 
  exp = 0 ∧ mult = 6 ∧ add = 6 :=
by
  use 0
  use 6
  use 6
  split; sorry -- proof skipped

end horner_method_operations_l363_363499


namespace trapezoid_area_eq_fifty_over_three_l363_363736

variables (height diagonal1 diagonal2 : ℝ)

def is_right_trapezoid (h : ℝ) (d1 d2 : ℝ) : Prop :=
  h = 4 ∧ d1 = 5 ∧ d2 = 3 -- The height, one diagonal, and the base are as defined by the given conditions.

def trapezoid_area (d1 d2 : ℝ) : ℝ :=
  1/2 * d1 * d2

theorem trapezoid_area_eq_fifty_over_three (h : ℝ) (d1 d2 : ℝ) 
  (h_right_trap : is_right_trapezoid h d1 d2) : 
  trapezoid_area d1 d2 = (50 / 3) :=
begin
  -- Proof here
  sorry
end

end trapezoid_area_eq_fifty_over_three_l363_363736


namespace quadratic_has_real_root_l363_363565

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l363_363565


namespace question_1_question_2_l363_363674

section
variable (a b c : ℝ)
variable (f : ℝ → ℝ := fun x => (1/3) * x^3 - (a/2) * x^2 + b * x + c)

theorem question_1 (h1 : a > 0) (h2 : ∀ x, f x = (1/3) * x^3 - (a/2) * x^2 + b * x + c) (h3 : ∀ x, y = f x → ∃ l, y = l * (x - 0) + 1) :
  b = 0 ∧ c = 1 :=
sorry

theorem question_2 (h4 : ∀ x, f x = (1/3) * x^3 - (a/2) * x^2 + b * x + c) (h5 : ∀ x, y = f x → ∃ t, (2 - f t = f' t * (-t)) ∧ g(t) = (2/3) * t^3 - (1/2) * a * t^2 + 1 = 0) :
  0 < a ∧ a < 2 * real.cbrt 3 :=
sorry
end

end question_1_question_2_l363_363674


namespace total_players_correct_l363_363985

-- Define the number of players for each type of sport
def cricket_players : Nat := 12
def hockey_players : Nat := 17
def football_players : Nat := 11
def softball_players : Nat := 10

-- The theorem we aim to prove
theorem total_players_correct : 
  cricket_players + hockey_players + football_players + softball_players = 50 := by
  sorry

end total_players_correct_l363_363985


namespace quad_form_unique_solution_l363_363650

theorem quad_form_unique_solution (d e f : ℤ) (h1 : d * d = 16) (h2 : 2 * d * e = -40) (h3 : e * e + f = -56) : d * e = -20 :=
by sorry

end quad_form_unique_solution_l363_363650


namespace number_of_valid_4_digit_numbers_l363_363524

theorem number_of_valid_4_digit_numbers : 
  ∃ n : ℕ,
    (∀ (a b : ℕ), 
       10 ≤ a * 10 + b ∧ a * 10 + b < 100 ∧ (a + b) % 3 = 0) ∧ 
    n = (∑ a in finset.range 10, ∑ b in finset.range 10, if (a + b) % 3 = 0 then 1 else 0) / 3 ∧
    n = 30 := sorry

end number_of_valid_4_digit_numbers_l363_363524


namespace percentage_milk_in_B_l363_363390

theorem percentage_milk_in_B :
  ∀ (A B C : ℕ),
  A = 1200 → B + C = A → B + 150 = C - 150 →
  (B:ℝ) / (A:ℝ) * 100 = 37.5 :=
by
  intros A B C hA hBC hE
  sorry

end percentage_milk_in_B_l363_363390


namespace ellipse_solution_chord_length_solution_l363_363913

-- Define variables
variable (a b : ℝ)
variable (h1 : a > b)
variable (h2 : b > 0)
variable (h3 : a^2 - b^2 = 3)
variable (h4 : (sqrt 3, 1/2) = (sqrt 3, 1/2))
variable (h5 : (sqrt 3)^2 / a^2 + (1/2)^2 / b^2 = 1)

noncomputable def ellipse_equation : Prop :=
  ∃ (a b : ℝ), 
    (a > b) ∧ 
    (b > 0) ∧ 
    (a^2 - b^2 = 3) ∧ 
    (sqrt 3)^2 / a^2 + (1/2)^2 / b^2 = 1 ∧ 
    (a = 2) ∧ 
    (b = 1) ∧ 
    ( ∀(x y : ℝ), (x^2 / 4 + y^2 = 1))

theorem ellipse_solution : ellipse_equation :=
  sorry

-- Define variables for chord length problem
variable (x1 x2 y1 y2 : ℝ)
variable (FOCUS : (sqrt 3, 0))
variable (LINE : ∀ (x y : ℝ), y = (1/2) * (x - sqrt 3))
variable (EL : ( ∀(x y : ℝ), (x^2 / 4 + y^2 = 1)))
variable (h6 : x1 + x2 = sqrt 3)
variable (h7 : x1 * x2 = -1/2)

noncomputable def chord_length : Prop :=
  ∀ (x1 x2 : ℝ), 
    (x1 + x2 = sqrt 3) ∧ 
    (x1 * x2 = -1/2) ∧ 
    (sqrt 1 + (1/4)) * ((x1 + x2)^2 - 4 * x1 * x2) / 2 = 5 / 2

theorem chord_length_solution : chord_length :=
  sorry

end ellipse_solution_chord_length_solution_l363_363913


namespace man_l363_363368

-- Define the conditions as constants.
def speed_current_first_section : ℝ := 1.5 -- km/hr
def speed_current_second_section : ℝ := 2.5 -- km/hr
def speed_current_third_section : ℝ := 3.5 -- km/hr
def speed_with_current_first_section : ℝ := 25 -- km/hr

-- Define the question as a theorem.
theorem man's_speed_against_current :
    let speed_still_water := speed_with_current_first_section - speed_current_first_section in
    let speed_against_current_first := speed_still_water - speed_current_first_section in
    let speed_against_current_second := speed_still_water - speed_current_second_section in
    let speed_against_current_third := speed_still_water - speed_current_third_section in
    speed_against_current_first = 22 ∧
    speed_against_current_second = 21 ∧
    speed_against_current_third = 20 :=
by
  -- proof placeholder.
  sorry

end man_l363_363368


namespace factors_of_P_factorization_of_P_factorize_expression_l363_363805

noncomputable def P (a b c : ℝ) : ℝ :=
  a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b)

theorem factors_of_P (a b c : ℝ) :
  (a - b ∣ P a b c) ∧ (b - c ∣ P a b c) ∧ (c - a ∣ P a b c) :=
sorry

theorem factorization_of_P (a b c : ℝ) :
  P a b c = -(a - b) * (b - c) * (c - a) :=
sorry

theorem factorize_expression (x y z : ℝ) :
  (x + y + z)^3 - x^3 - y^3 - z^3 = 3 * (x + y) * (y + z) * (z + x) :=
sorry

end factors_of_P_factorization_of_P_factorize_expression_l363_363805


namespace matrix_system_solution_range_l363_363094

theorem matrix_system_solution_range (m : ℝ) :
  (∃ x y: ℝ, 
    (m * x + y = m + 1) ∧ 
    (x + m * y = 2 * m)) ↔ m ≠ -1 :=
by
  sorry

end matrix_system_solution_range_l363_363094


namespace max_value_of_function_l363_363538

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x + π / 4)

theorem max_value_of_function :
  ∃ x ∈ set.Icc (0 : ℝ) (π / 4), f x = sqrt 3 ∧ ∀ y ∈ set.Icc (0 : ℝ) (π / 4), f y ≤ sqrt 3 :=
by {
  sorry
}

end max_value_of_function_l363_363538


namespace tan_alpha_minus_three_pi_div_four_l363_363921

theorem tan_alpha_minus_three_pi_div_four (α β : ℝ) 
  (h1 : sin (α - β) * cos β + cos (α - β) * sin β = 3 / 5)
  (h2 : π / 2 < α ∧ α < π) : tan (α - 3 * π / 4) = 1 :=
by
  sorry

end tan_alpha_minus_three_pi_div_four_l363_363921


namespace jenny_ate_65_chocolates_l363_363654

-- Define the number of chocolate squares Mike ate
def MikeChoc := 20

-- Define the function that calculates the chocolates Jenny ate
def JennyChoc (mikeChoc : ℕ) := 3 * mikeChoc + 5

-- The theorem stating the solution
theorem jenny_ate_65_chocolates (h : MikeChoc = 20) : JennyChoc MikeChoc = 65 := by
  -- Automatic proof step
  sorry

end jenny_ate_65_chocolates_l363_363654


namespace star_of_15_star_eq_neg_15_l363_363447

def y_star (y : ℤ) : ℤ := 10 - y
def star_y (y : ℤ) : ℤ := y - 10

theorem star_of_15_star_eq_neg_15 : star_y (y_star 15) = -15 :=
by {
  -- applying given definitions;
  sorry
}

end star_of_15_star_eq_neg_15_l363_363447


namespace FO_perp_BC_l363_363610

variables (A B C D E F O : Type) [inhabited A] [inhabited B] [inhabited C]
[inhabited D] [inhabited E] [inhabited F] [inhabited O]
(m : Rat)  -- since D is the midpoint, assuming Rat as Rational numbers
(h1 : midpoint D B C)  -- D is the midpoint of BC
(h2 : circle_contains O A C)  -- Circle O passes through points A and C
(h3 : tangent DA A O)  -- Circle O is tangent to DA at A
(h4 : intersect_extension BA E O)  -- The extension of BA intersects circle O at E
(h5 : intersect_extension CE F DA)  -- The extension of CE intersects DA at F

-- Prove FO ⊥ BC
theorem FO_perp_BC : perpendicular FO BC :=
sorry

end FO_perp_BC_l363_363610


namespace numberOfBijections_l363_363119

noncomputable def S : Set (ℤ × ℤ × ℤ) := 
  {p | p.1 ∈ {0, 1} ∧ p.2 ∈ {0, 1} ∧ p.3 ∈ {0, 1}}

def isBijective (f : (ℤ × ℤ × ℤ) → (ℤ × ℤ × ℤ)) (S : Set (ℤ × ℤ × ℤ)) : Prop :=
  ∀ x ∈ S, ∃! y ∈ S, f x = y

def metric (A B : (ℤ × ℤ × ℤ)) : ℤ :=
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 + (A.3 - B.3) ^ 2

def satisfiesCondition (f : (ℤ × ℤ × ℤ) → (ℤ × ℤ × ℤ)) (S : Set (ℤ × ℤ × ℤ)) : Prop :=
  ∀ x ∈ S, metric x (f x) = 1

theorem numberOfBijections : 
  ∃! (f : (ℤ × ℤ × ℤ) → (ℤ × ℤ × ℤ)), isBijective f S ∧ satisfiesCondition f S ∧ (finset.card S = 81) :=
sorry

end numberOfBijections_l363_363119


namespace rectangle_height_l363_363972

-- Defining the conditions
def base : ℝ := 9
def area : ℝ := 33.3

-- Stating the proof problem
theorem rectangle_height : (area / base) = 3.7 :=
by
  sorry

end rectangle_height_l363_363972


namespace perfect_cube_constructed_l363_363205

theorem perfect_cube_constructed (n : ℕ) (h : n ≥ 1) :
  let A_n := (10 ^ (n + 1) - 1) ^ 3 in
  ∃ k : ℕ, A_n = k^3 :=
by
  sorry

end perfect_cube_constructed_l363_363205


namespace max_profundity_eq_Fib_l363_363635

open Nat

-- Definitions as per conditions
def profundity (word : List Char) : Nat :=
  (word.sublists.map List.length).length
  
def dog_dictionary (n : Nat) : Set (List Char) :=
  { word | word.length = n ∧ word.all (fun c => c = 'A' ∨ c = 'U') }

def max_profundity (n : Nat) : Nat :=
  Sup (profundity '' (dog_dictionary n))

-- Theorem statement
theorem max_profundity_eq_Fib (n : Nat) : max_profundity n = Fibonacci (n + 3) - 3 := 
  sorry

end max_profundity_eq_Fib_l363_363635


namespace triangle_perimeter_l363_363244

theorem triangle_perimeter:
  ∀ (x : ℝ), x^2 - 8 * x + 12 = 0 → (4 + 7 + x) = 17 ∧
  (x = 2 ∨ x = 6) ∧ (x ≠ 2) :=
begin
  sorry
end

end triangle_perimeter_l363_363244


namespace james_total_points_l363_363726

def points_per_correct_answer : ℕ := 2
def bonus_points_per_round : ℕ := 4
def total_rounds : ℕ := 5
def questions_per_round : ℕ := 5
def total_questions : ℕ := total_rounds * questions_per_round
def questions_missed_by_james : ℕ := 1
def questions_answered_by_james : ℕ := total_questions - questions_missed_by_james
def points_for_correct_answers : ℕ := questions_answered_by_james * points_per_correct_answer
def complete_rounds_by_james : ℕ := total_rounds - 1  -- Since James missed one question, he has 4 complete rounds
def bonus_points_by_james : ℕ := complete_rounds_by_james * bonus_points_per_round
def total_points : ℕ := points_for_correct_answers + bonus_points_by_james

theorem james_total_points : total_points = 64 := by
  sorry

end james_total_points_l363_363726


namespace max_sin_cos_expression_l363_363885

open Real

noncomputable def max_value_sin_cos_expression : ℝ :=
  real.sqrt 5 * 8 / 25

theorem max_sin_cos_expression :
  ∀ θ : ℝ, 0 < θ ∧ θ < 3 * π / 2 →
    sin (θ / 3) * (1 + cos (2 * θ / 3)) ≤ max_value_sin_cos_expression :=
begin
  sorry,
end

example : ∃ θ : ℝ, 0 < θ ∧ θ < 3 * π / 2 ∧ 
  sin (θ / 3) * (1 + cos (2 * θ / 3)) = max_value_sin_cos_expression :=
begin
  use [π/2, by norm_num, by norm_num],
  -- computation for the critical point can be inserted here
  sorry,
end

end max_sin_cos_expression_l363_363885


namespace troy_needs_more_money_to_buy_computer_l363_363285

theorem troy_needs_more_money_to_buy_computer :
  ∀ (price_new_computer savings sale_old_computer : ℕ),
  price_new_computer = 80 →
  savings = 50 →
  sale_old_computer = 20 →
  (price_new_computer - (savings + sale_old_computer)) = 10 :=
by
  intros price_new_computer savings sale_old_computer Hprice Hsavings Hsale
  sorry

end troy_needs_more_money_to_buy_computer_l363_363285


namespace no_four_digit_with_five_units_divisible_by_ten_l363_363100

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def units_place_is_five (n : ℕ) : Prop :=
  n % 10 = 5

def divisible_by_ten (n : ℕ) : Prop :=
  n % 10 = 0

theorem no_four_digit_with_five_units_divisible_by_ten : ∀ n : ℕ, 
  is_four_digit n → units_place_is_five n → ¬ divisible_by_ten n :=
by
  intro n h1 h2
  rw [units_place_is_five] at h2
  rw [divisible_by_ten, h2]
  sorry

end no_four_digit_with_five_units_divisible_by_ten_l363_363100


namespace arithmetic_sequence_length_l363_363956

theorem arithmetic_sequence_length (a d : ℕ) (l : ℕ) (h_a : a = 6) (h_d : d = 4) (h_l : l = 154) :
  ∃ n : ℕ, l = a + (n-1) * d ∧ n = 38 :=
by
  use 38
  split
  { rw [h_a, h_d]
    calc 154 = 6 + (38 - 1) * 4 : by norm_num
          ... = 6 + 37 * 4       : by rfl
          ... = 6 + 148          : by norm_num
          ... = 154              : by norm_num }
  { rfl }

end arithmetic_sequence_length_l363_363956


namespace approx_num_chars_in_ten_thousand_units_l363_363803

-- Define the number of characters in the book
def num_chars : ℕ := 731017

-- Define the conversion factor from characters to units of 'ten thousand'
def ten_thousand : ℕ := 10000

-- Define the number of characters in units of 'ten thousand'
def chars_in_ten_thousand_units : ℚ := num_chars / ten_thousand

-- Define the rounded number of units to the nearest whole number
def rounded_chars_in_ten_thousand_units : ℤ := round chars_in_ten_thousand_units

-- Theorem to state the approximate number of characters in units of 'ten thousand' is 73
theorem approx_num_chars_in_ten_thousand_units : rounded_chars_in_ten_thousand_units = 73 := 
by sorry

end approx_num_chars_in_ten_thousand_units_l363_363803


namespace problem_statement_l363_363896

theorem problem_statement (a b : ℝ) (h1 : 3^a = 15) (h2 : 5^b = 15) :
  (a - 1) ^ 2 + (b - 1) ^ 2 = 2 :=
sorry

end problem_statement_l363_363896


namespace problem_part_1_problem_part_2_l363_363055

noncomputable def hyperbola_equation_of_center_origin_eccentricity_and_point (e : ℝ) (p : ℝ × ℝ) : Prop := 
  ∃ a b : ℝ, e = real.sqrt (1 + (b ^ 2 / a ^ 2)) ∧ 
              (p.fst ^ 2 / a ^ 2 - p.snd ^ 2 / b ^ 2 = 1) ∧ 
              a ^ 2 = b ^ 2

noncomputable def hyperbola_passes_through_point (h : ℝ × ℝ → Prop) (p : ℝ × ℝ) : Prop := 
  h p

def triangle_area_of_hyperbola_foci (e : ℝ) (p m : ℝ × ℝ) : Prop := 
  ∃ a b c : ℝ, e = real.sqrt (1 + (b ^ 2 / a ^ 2)) ∧ 
                  (p.fst ^ 2 / a ^ 2 - p.snd ^ 2 / b ^ 2 = 1) ∧ 
                  a ^ 2 = b ^ 2 ∧ 
                  2 * c = 4 * real.sqrt (3) ∧ 
                  m.snd = real.sqrt 3 ∧ 
                  0.5 * 4 * real.sqrt (3) * real.sqrt (3) = 6

theorem problem_part_1 : 
  hyperbola_equation_of_center_origin_eccentricity_and_point (real.sqrt 2) (4, -real.sqrt 10) := 
sorry

theorem problem_part_2 :
  triangle_area_of_hyperbola_foci (real.sqrt 2) (4, -real.sqrt 10) (3, real.sqrt 3) := 
sorry

end problem_part_1_problem_part_2_l363_363055


namespace determine_set_A_l363_363857

-- Define the function f as described
def f (n : ℕ) (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else (x - 1) / 2 + 2^(n - 1)

-- Define the set A
def A (n : ℕ) : Set ℕ :=
  { x | (Nat.iterate (f n) n x) = x }

-- State the theorem
theorem determine_set_A (n : ℕ) (hn : n > 0) :
    A n = { x | 1 ≤ x ∧ x ≤ 2^n } :=
sorry

end determine_set_A_l363_363857


namespace taimour_paints_fence_alone_in_15_hours_l363_363794

theorem taimour_paints_fence_alone_in_15_hours :
  ∀ (T : ℝ), (∀ (J : ℝ), J = T / 2 → (1 / J + 1 / T = 1 / 5)) → T = 15 :=
by
  intros T h
  have h1 := h (T / 2) rfl
  sorry

end taimour_paints_fence_alone_in_15_hours_l363_363794


namespace distance_center_to_line_is_3_l363_363883

noncomputable def point := ℝ × ℝ

def circle_center (h : ∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y + 4 = 0) : point :=
(1, 2)

def line_equation (x y : ℝ) : Prop := 3 * x + 4 * y + 4 = 0

def distance_from_point_to_line (p : point) (h : ∀ x y : ℝ, 3 * x + 4 * y + 4 = 0) : ℝ :=
(abs (3 * p.1 + 4 * p.2 + 4)) / (sqrt (3 ^ 2 + 4 ^ 2))

theorem distance_center_to_line_is_3
  (h₁ : ∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y + 4 = 0)
  (h₂ : ∀ x y : ℝ, 3 * x + 4 * y + 4 = 0) :
  distance_from_point_to_line (circle_center h₁) h₂ = 3 :=
sorry

end distance_center_to_line_is_3_l363_363883


namespace integral_inequality_l363_363696

open Real

theorem integral_inequality :
  ln ((sqrt 2009 + sqrt 2010) / (sqrt 2008 + sqrt 2009)) <
    ∫ x in sqrt 2008 .. sqrt 2009, (sqrt (1 - exp (-x^2)) / x) ∧
    (∫ x in sqrt 2008 .. sqrt 2009, sqrt (1 - exp (-x^2)) / x) < (sqrt 2009 - sqrt 2008) :=
sorry

end integral_inequality_l363_363696


namespace t_squared_value_l363_363396

theorem t_squared_value :
  (∃ (t : ℝ), t > 0 ∧ ∃ (a b : ℝ), a = 4 ∧ ( ∀ (x y : ℝ), (x, y) ∈ [(4, 0), (3, 3), (0, t)] → x^2 / a^2 + y^2 / b^2 = 1) ∧ b^2 = 144 / 7) →
  ∃ (t : ℝ), t^2 = 144 / 7 :=
begin
  sorry
end

end t_squared_value_l363_363396


namespace jerry_set_off_firecrackers_l363_363149

theorem jerry_set_off_firecrackers :
  ∀ (n : ℕ), n = 48 → 
    ∃ m : ℕ, m = 12 → 
      ∃ k : ℕ, k = n - m → 
        ∃ d : ℕ, d = k / 6 → 
          ∃ g : ℕ, g = k - d → 
            ∃ s : ℕ, s = g / 2 → 
              s = 15 :=
by 
  intros n hn m hm k hk d hd g hg s hs
  -- This line starts inserting the proof steps, which we leave to Lean to fill.
  sorry

end jerry_set_off_firecrackers_l363_363149


namespace tan_alpha_tan_beta_l363_363528

theorem tan_alpha_tan_beta (α β : ℝ) (h1 : Real.cos (α + β) = 3 / 5) (h2 : Real.cos (α - β) = 4 / 5) :
  Real.tan α * Real.tan β = 1 / 7 := by
  sorry

end tan_alpha_tan_beta_l363_363528


namespace distance_moon_earth_distance_moon_plane_ecliptic_length_moon_shadow_cone_intersection_point_sun_moon_line_l363_363523

-- Define the problem conditions
def diameter_moon : ℝ := 3476 -- in km
def angular_diameter_moon : ℝ := 31.1 * (π / 180 / 60) -- converted to radians
def ecliptic_latitude_moon : ℝ := 0.3329 * (π / 180) -- converted to radians
def distance_earth_sun : ℝ := 151290000 -- in km
def angular_size_sun : ℝ := 31.65 * (π / 180 / 60) -- converted to radians
def sun_declination : ℝ := 19.923 -- in degrees

-- Define the problem statements
theorem distance_moon_earth (D_H η_H : ℝ) (hD_H : D_H = diameter_moon) (hη_H : η_H = angular_diameter_moon) :
  FH = 384200 := sorry

theorem distance_moon_plane_ecliptic (FH LA_H : ℝ) (hFH : FH = 384200) (hLA_H : LA_H = ecliptic_latitude_moon) :
  d_H = 2232 := sorry

theorem length_moon_shadow_cone (FN FH D_H D_N : ℝ)
  (hFN : FN = distance_earth_sun) (hFH : FH = 384200)
  (hD_H : D_H = diameter_moon) (hD_N : D_N = FN * η_N)
  (hη_N : η_N = angular_size_sun) : 
  CH = 377400 := sorry

theorem intersection_point_sun_moon_line (sun_declination : ℝ) : 
  intersection_point_sun_moon_line = "Ankara, Turkey" := sorry

end distance_moon_earth_distance_moon_plane_ecliptic_length_moon_shadow_cone_intersection_point_sun_moon_line_l363_363523


namespace singh_gain_l363_363402

def initial_amounts (B A S : ℕ) : Prop :=
  B = 70 ∧ A = 70 ∧ S = 70

def ratio_Ashtikar_Singh (A S : ℕ) : Prop :=
  2 * A = S

def ratio_Singh_Bhatia (S B : ℕ) : Prop :=
  4 * B = S

def total_conservation (A S B : ℕ) : Prop :=
  A + S + B = 210

theorem singh_gain : ∀ B A S fA fB fS : ℕ,
  initial_amounts B A S →
  ratio_Ashtikar_Singh fA fS →
  ratio_Singh_Bhatia fS fB →
  total_conservation fA fS fB →
  fS - S = 50 :=
by
  intros B A S fA fB fS
  intros i rA rS tC
  sorry

end singh_gain_l363_363402


namespace determine_integer_n_l363_363867

noncomputable def tan_pi_over_6 := Real.tan (Real.pi / 6)

theorem determine_integer_n :
  (∃ n : ℤ, 0 ≤ n ∧ n < 12 ∧ (↑(Complex.ofReal tan_pi_over_6 + Complex.i) / (Complex.ofReal tan_pi_over_6 - Complex.i))
  = Complex.exp (Complex.i * (2 * n * Real.pi / 12))) :=
sorry

end determine_integer_n_l363_363867


namespace sum_of_a_b_l363_363968

variable {a b : ℝ}

theorem sum_of_a_b (h1 : a^2 = 4) (h2 : b^2 = 9) (h3 : a * b < 0) : a + b = 1 ∨ a + b = -1 := 
by 
  sorry

end sum_of_a_b_l363_363968


namespace unique_solution_nat_triplet_l363_363435

theorem unique_solution_nat_triplet (x y l : ℕ) (h : x^3 + y^3 - 53 = 7^l) : (x, y, l) = (3, 3, 0) :=
sorry

end unique_solution_nat_triplet_l363_363435


namespace ways_to_return_0_non_negative_ways_to_return_0_l363_363625

open Nat

-- Define the number of ways to return to flower 0 after k jumps using binomial coefficients
def ways_to_return_to_0_after_k_jumps (k : ℕ) : ℕ :=
  if h : Even k then binomial k (k / 2) else 0

-- Define the number of ways to return to flower 0 after k jumps without landing on a negative index using Catalan numbers
def non_negative_ways_to_return_to_0_after_k_jumps (k : ℕ) : ℕ :=
  if h : Even k then Catalan (k / 2) else 0

-- Prove the number of ways to return to flower 0 after k jumps is binomial coefficient when k is even
theorem ways_to_return_0 (k : ℕ) (h : Even k) :
  ways_to_return_to_0_after_k_jumps k = binomial k (k / 2) :=
by {
  rw [ways_to_return_to_0_after_k_jumps],
  exact if_pos h
}

-- Prove the number of ways to return to flower 0 after k jumps without landing on a negative index
-- is the Catalan number when k is even
theorem non_negative_ways_to_return_0 (k : ℕ) (h : Even k) :
  non_negative_ways_to_return_to_0_after_k_jumps k = Catalan (k / 2) :=
by {
  rw [non_negative_ways_to_return_to_0_after_k_jumps],
  exact if_pos h
}

end ways_to_return_0_non_negative_ways_to_return_0_l363_363625


namespace floor_expression_eq_eight_l363_363852

theorem floor_expression_eq_eight :
  (Int.floor (1005^3 / (1003 * 1004) - 1003^3 / (1004 * 1005)) = 8) :=
sorry

end floor_expression_eq_eight_l363_363852


namespace part1_inequality_part2_range_of_a_l363_363461

-- Definitions and conditions
def f (x a : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- First proof problem for a = 1
theorem part1_inequality (x : ℝ) : f x 1 > 1 ↔ x > 1/2 :=
by sorry

-- Second proof problem for range of a when f(x) > x in (0, 1)
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → f x a > x) → 0 < a ∧ a ≤ 2 :=
by sorry

end part1_inequality_part2_range_of_a_l363_363461


namespace frequency_of_group_of_samples_l363_363376

def sample_capacity : ℝ := 32
def frequency_rate : ℝ := 0.125

theorem frequency_of_group_of_samples : frequency_rate * sample_capacity = 4 :=
by 
  sorry

end frequency_of_group_of_samples_l363_363376


namespace math_problem_l363_363046

variables {a b : ℝ}
open Real

theorem math_problem (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (a - 1) * (b - 1) = 1 ∧ 
  (∀ b : ℝ, (a = 2 * b → a + 4 * b = 9)) ∧ 
  (∀ b : ℝ, (b = 3 → (1 / a^2 + 2 / b^2) = 2 / 3)) :=
by
  sorry

end math_problem_l363_363046


namespace determine_gallons_l363_363266

def current_amount : ℝ := 7.75
def desired_total : ℝ := 14.75
def needed_to_add (x : ℝ) : Prop := desired_total = current_amount + x

theorem determine_gallons : needed_to_add 7 :=
by
  sorry

end determine_gallons_l363_363266


namespace real_roots_quadratic_l363_363975

theorem real_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, (k - 2) * x^2 - 2 * k * x + k - 6 = 0) ↔ (k ≥ 1.5 ∧ k ≠ 2) :=
by {
  sorry
}

end real_roots_quadratic_l363_363975


namespace increasing_sequence_lambda_condition_l363_363082

theorem increasing_sequence_lambda_condition (λ : ℝ) : 
  (∀ n : ℕ, 1 ≤ n → n^2 + λ * n < (n + 1)^2 + λ * (n + 1)) ↔ (λ > -3) :=
by 
  sorry

end increasing_sequence_lambda_condition_l363_363082


namespace smallest_number_among_four_l363_363391

theorem smallest_number_among_four (a b c d : ℤ) (h1 : a = 2023) (h2 : b = 2022) (h3 : c = -2023) (h4 : d = -2022) : 
  min (min a (min b c)) d = -2023 :=
by
  rw [h1, h2, h3, h4]
  sorry

end smallest_number_among_four_l363_363391


namespace quadratic_real_root_iff_b_range_l363_363584

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l363_363584


namespace perimeter_of_square_C_l363_363724

theorem perimeter_of_square_C (s_A s_B s_C : ℝ)
  (h1 : 4 * s_A = 16)
  (h2 : 4 * s_B = 32)
  (h3 : s_C = s_B - s_A) :
  4 * s_C = 16 :=
by
  sorry

end perimeter_of_square_C_l363_363724


namespace statement_2_statement_4_l363_363837

theorem statement_2 (x : ℝ) : ∃ x₀, (x + x₀ = -π/3 ∧ (f : ℝ → ℝ) = cos (2 * x - π/3)) :=
sorry

theorem statement_4 : ∃ min_val, ∀ x ∈ Ioo (-π/2) (π/2), ((f : ℝ → ℝ) = (λ x, (cos x + 3) / cos x) ∧ min_val ≤ f x ∧ ¬∃ max_val, f x < max_val) :=
sorry

end statement_2_statement_4_l363_363837


namespace count_matrices_l363_363450

theorem count_matrices (n : ℕ) : 
  let A (m n : ℕ) := { M : Matrix (Fin 2) (Fin (m * n)) ℕ | 
    (∀ i, ∀ j < m*n, M 0 j ≤ M 0 (j+1) ∧ M 1 j ≤ M 1 (j+1)) ∧
    (∀ j < m*n, M 0 j < M 1 j) ∧ 
    (∀ k ∈ finset.range m, finset.card ({ j | M 0 j = k+1 } ∪ { j | M 1 j = k+1 }) = 2*n) } in
  ∃ (M ∈ A 5 n), fincard A (5, n) = (5 * n ^ 2 + 5 * n + 2) / 2 :=
sorry

end count_matrices_l363_363450


namespace circles_intersect_in_two_points_l363_363958

def circle1 (x y : ℝ) : Prop := x^2 + (y - 3/2)^2 = (3/2)^2
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

theorem circles_intersect_in_two_points :
  ∃! (p : ℝ × ℝ), (circle1 p.1 p.2) ∧ (circle2 p.1 p.2) := 
sorry

end circles_intersect_in_two_points_l363_363958


namespace time_for_ball_to_hit_ground_l363_363401

noncomputable def time_to_hit_ground_eqn : Real :=
  -14.5 * t^2 - 30 * t + 200

theorem time_for_ball_to_hit_ground :
  ∃ t : Real, time_to_hit_ground_eqn = 0 ∧ t ≈ 2.82 :=
by
  -- Skipping the proof
  sorry

end time_for_ball_to_hit_ground_l363_363401


namespace equal_functions_D_l363_363317

noncomputable def f_A : ℝ → ℝ := λ x, real.sqrt (x^2)
noncomputable def g_A : ℝ → ℝ := λ x, x

noncomputable def f_B : ℝ → ℝ := λ x, x
noncomputable def g_B : ℝ → ℝ := λ x, if x ≠ 0 then x^2 / x else 0 -- Domain issue encoded

noncomputable def f_C : ℝ → ℝ := λ x, real.log (x^2)
noncomputable def g_C : ℝ → ℝ := λ x, 2 * real.log x

noncomputable def f_D (a : ℝ) (ha1 : 0 < a) (ha2 : a ≠ 1) : ℝ → ℝ := λ x, real.log a (a^x)
noncomputable def g_D : ℝ → ℝ := λ x, x^(1/3) * (x^2)^(1/3)

theorem equal_functions_D (a : ℝ) (ha1 : 0 < a) (ha2 : a ≠ 1) :
  ∀ x : ℝ, f_D a ha1 ha2 x = g_D x :=
by
  sorry

end equal_functions_D_l363_363317


namespace problem_quadratic_has_real_root_l363_363601

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l363_363601


namespace fifteenth_number_in_digit_sum_15_list_l363_363389

def digits_sum (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

theorem fifteenth_number_in_digit_sum_15_list :
  ∃ n : ℕ, (digits_sum n = 15 ∧ 
            ∃ k : ℕ, 
              list.sort (λ x y, x < y) 
              (list.filter (λ x, digits_sum x = 15) (list.range 1000)).nth k = some n 
              ∧
              k = 14) 
  ∧ n = 366 := sorry

end fifteenth_number_in_digit_sum_15_list_l363_363389


namespace find_dividend_l363_363884

theorem find_dividend (k : ℕ) (quotient : ℕ) (dividend : ℕ) (h1 : k = 8) (h2 : quotient = 8) (h3 : dividend = k * quotient) : dividend = 64 := 
by 
  sorry

end find_dividend_l363_363884


namespace find_incorrect_statement_l363_363457

variables (a b : line) (α β : plane)

-- Conditions
axiom cond1 : (a ∥ b) ∧ (a ⟂ α) → b ⟂ α
axiom cond2 : (a ∥ α) ∧ (α ∩ β = b) → a ∥ b
axiom cond3 : (a ⟂ α) ∧ (a ⟂ β) → α ∥ β
axiom cond4 : (a ⟂ α) ∧ (a ⊆ β) → α ⟂ β

-- Theorem to prove
theorem find_incorrect_statement : 
  ¬ (((a ∥ α) ∧ (α ∩ β = b) → a ∥ b) ∧ ((a ∥ b) ∧ (a ⟂ α) → b ⟂ α) ∧ ((a ⟂ α) ∧ (a ⟂ β) → α ∥ β) ∧ ((a ⟂ α) ∧ (a ⊆ β) → α ⟂ β)) :=
sorry

end find_incorrect_statement_l363_363457


namespace cos_theta_correct_l363_363163

open Real

def normal_vector_plane_1 : ℝ × ℝ × ℝ := (3, -2, 1)
def normal_vector_plane_2 : ℝ × ℝ × ℝ := (9, -6, -4)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2 + v.3^2)

def cos_theta : ℝ :=
  dot_product normal_vector_plane_1 normal_vector_plane_2 / 
  (magnitude normal_vector_plane_1 * magnitude normal_vector_plane_2)

theorem cos_theta_correct : cos_theta = 35 / sqrt 1862 := by
  sorry

end cos_theta_correct_l363_363163


namespace negation_q_sufficient_not_necessary_negation_p_l363_363170

theorem negation_q_sufficient_not_necessary_negation_p :
  (∃ x : ℝ, (∃ p : 16 - x^2 < 0, (x ∈ [-4, 4]))) →
  (∃ x : ℝ, (∃ q : x^2 + x - 6 > 0, (x ∈ [-3, 2]))) :=
sorry

end negation_q_sufficient_not_necessary_negation_p_l363_363170


namespace prob_both_societies_have_at_least_one_participant_prob_AB_same_CD_not_same_l363_363755

-- Define the students and societies
inductive Student
| A | B | C | D

inductive Society
| LoveHeart | LiteraryStyle

open Student Society

-- Probability that both societies have at least one participant
theorem prob_both_societies_have_at_least_one_participant :
  P (has_participants LoveHeart ∧ has_participants LiteraryStyle) = 1 / 8 := 
sorry 

-- Probability that A and B are in the same society, while C and D are not
theorem prob_AB_same_CD_not_same :
  P (same_society A B ∧ ¬ same_society C D) = 1 / 4 := 
sorry

-- Helper definition: A student being in a society
def in_society (s : Student) (soc : Society) : Prop := sorry

-- Helper definition: A society having at least one participant
def has_participants (soc : Society) : Prop := sorry

-- Helper definition: Two students being in the same society
def same_society (s1 s2 : Student) : Prop := sorry

end prob_both_societies_have_at_least_one_participant_prob_AB_same_CD_not_same_l363_363755


namespace prove_n_eq_4_l363_363452

theorem prove_n_eq_4 (a : ℕ) (a_1 : ℕ) (a_2 : ℕ) … (a_n : ℕ) (x : ℕ) (n : ℕ)
  (h1 : (1 + x)^n = a + a_1 * x + a_2 * x^2 + … + a_n * x^n)
  (h2 : a + a_1 + a_2 + … + a_n = 16) : n = 4 :=
by
  sorry

end prove_n_eq_4_l363_363452


namespace infinite_solutions_l363_363173

def P (x : ℕ) : ℕ :=
  -- Implementation of the product of digits of x
  sorry

def S (x : ℕ) : ℕ :=
  -- Implementation of the sum of digits of x
  sorry

theorem infinite_solutions : 
  ∃ᶠ x in nat, P(P(x)) + P(S(x)) + S(P(x)) + S(S(x)) = 1984 :=
sorry

end infinite_solutions_l363_363173


namespace repeating_decimal_bc_cabc_l363_363516

theorem repeating_decimal_bc_cabc (b c : ℕ) (bc : ℕ := 10 * b + c) (cabc : ℕ := 100 * c + 10 * b + c) :
  (0.bc_overline_bc : ℝ) = (bc : ℕ) / 99 ∧ 
  (0.cabc_overline_cabc : ℝ) = (cabc : ℕ) / 999 ∧
  (bc : ℝ) / 99 + (cabc : ℝ) / 999 = 83 / 222 →
  bc = 11 :=
sorry

end repeating_decimal_bc_cabc_l363_363516


namespace insurance_calculation_l363_363342

def loan_amount : ℝ := 20000000
def appraisal_value : ℝ := 14500000
def cadastral_value : ℝ := 15000000
def basic_tariff : ℝ := 0.2 / 100
def coefficient_no_transition : ℝ := 0.8
def coefficient_no_certificates : ℝ := 1.3

noncomputable def adjusted_tariff : ℝ := basic_tariff * coefficient_no_transition * coefficient_no_certificates
noncomputable def insured_amount : ℝ := max appraisal_value cadastral_value
noncomputable def insurance_premium : ℝ := insured_amount * adjusted_tariff

theorem insurance_calculation :
  adjusted_tariff = 0.00208 ∧ insurance_premium = 31200 := 
by
  sorry

end insurance_calculation_l363_363342


namespace line_intersects_circle_two_points_find_line_l363_363903

-- Definitions for circle C and line l
def circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5
def line (m x y : ℝ) : Prop := mx - y + 1 - m = 0

-- Part (1): Prove that for any m ∈ ℝ, line l intersects circle C at two points A, B
theorem line_intersects_circle_two_points (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), circle x₁ y₁ ∧ circle x₂ y₂ ∧ line m x₁ y₁ ∧ line m x₂ y₂ ∧ (x₁, y₁) ≠ (x₂, y₂) :=
by 
  sorry

-- Part (2): With fixed point P(1,1) dividing chord AB such that |AP| = 1/2|PB|, find the equation of l
theorem find_line (P : ℝ × ℝ) (hP : P = (1, 1)) :
  ∀ (A B : ℝ × ℝ), (fst A) ≠ (fst B) → 
  circle (fst A) (snd A) → 
  circle (fst B) (snd B) → 
  line (fst P) (fst A) (snd A) →
  line (fst P) (fst B) (snd B) →
  |fst A - fst P| = abs (1/2 * (fst B - fst P)) → 
  line (fst P) (fst P) 1 ∨
  line (fst P) (fst P + snd P - 2) 1 :=
by 
  sorry

end line_intersects_circle_two_points_find_line_l363_363903


namespace quadratic_has_real_root_l363_363570

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l363_363570


namespace ella_savings_l363_363020

theorem ella_savings
  (initial_cost_per_lamp : ℝ)
  (num_lamps : ℕ)
  (discount_rate : ℝ)
  (additional_discount : ℝ)
  (initial_total_cost : ℝ := num_lamps * initial_cost_per_lamp)
  (discounted_lamp_cost : ℝ := initial_cost_per_lamp - (initial_cost_per_lamp * discount_rate))
  (total_cost_with_discount : ℝ := num_lamps * discounted_lamp_cost)
  (total_cost_after_additional_discount : ℝ := total_cost_with_discount - additional_discount) :
  initial_cost_per_lamp = 15 →
  num_lamps = 3 →
  discount_rate = 0.25 →
  additional_discount = 5 →
  initial_total_cost - total_cost_after_additional_discount = 16.25 :=
by
  intros
  sorry

end ella_savings_l363_363020


namespace distinct_integer_solutions_l363_363886

-- Define the equation as a predicate on x and y
def equation (x y : ℤ) : Prop := x^2 + x + y = 5 + x^2 * y + x * y^2 - y * x

-- The problem to prove
theorem distinct_integer_solutions : 
  {p : ℤ × ℤ | equation p.1 p.2}.to_finset.card = 4 :=
sorry

end distinct_integer_solutions_l363_363886


namespace actual_length_correct_l363_363192

-- Definitions based on the conditions
def blueprint_scale : ℝ := 20
def measured_length_cm : ℝ := 16

-- Statement of the proof problem
theorem actual_length_correct :
  measured_length_cm * blueprint_scale = 320 := 
sorry

end actual_length_correct_l363_363192


namespace distance_from_center_to_endpoint_l363_363249

structure CircleDiameterEndpoints where
  P1 : ℝ × ℝ
  P2 : ℝ × ℝ

def circle_center (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((P1.1 - P2.1) ^ 2 + (P1.2 - P2.2) ^ 2)

example : CircleDiameterEndpoints :=
  { P1 := (12, -8), P2 := (-6, 4) }

theorem distance_from_center_to_endpoint (cd : CircleDiameterEndpoints) :
  distance (circle_center cd.P1 cd.P2) cd.P1 = real.sqrt 117 := by
  sorry

end distance_from_center_to_endpoint_l363_363249


namespace sum_circumferences_of_smaller_circles_l363_363224

theorem sum_circumferences_of_smaller_circles (C : ℝ) :
  (∀ n : ℕ, n > 0 → 
    let d := C / n in 
    let circumference := fun d => Real.pi * d in 
      (∑ i in Finset.range n, circumference d) = Real.pi * C) → 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (∑ i in Finset.range n, Real.pi * (C / n) - C) < ε := 
by 
  intros n hn d circumference sum_circumferences ε hε 
  use sorry

end sum_circumferences_of_smaller_circles_l363_363224


namespace quadratic_real_roots_l363_363573

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l363_363573


namespace problem_1_problem_2_l363_363091

open Set

-- First problem: when a = 2
theorem problem_1:
  ∀ (x : ℝ), 2 * x^2 - x - 1 > 0 ↔ (x < -(1 / 2) ∨ x > 1) :=
by
  sorry

-- Second problem: when a > -1
theorem problem_2 (a : ℝ) (h : a > -1) :
  ∀ (x : ℝ), 
    (if a = 0 then x - 1 > 0 else if a > 0 then  a * x ^ 2 + (1 - a) * x - 1 > 0 ↔ (x < -1 / a ∨ x > 1) 
    else a * x ^ 2 + (1 - a) * x - 1 > 0 ↔ (1 < x ∧ x < -1 / a)) :=
by
  sorry

end problem_1_problem_2_l363_363091


namespace prove_Φ_eq_8_l363_363011

-- Define the structure of the problem.
def condition (Φ : ℕ) : Prop := 504 / Φ = 40 + 3 * Φ

-- Define the main proof question.
theorem prove_Φ_eq_8 (Φ : ℕ) (h : condition Φ) : Φ = 8 := 
sorry

end prove_Φ_eq_8_l363_363011


namespace bisectors_intersect_on_midline_l363_363198

open EuclideanGeometry

variables {A B C D O M : Point}

-- Conditions: Trapezoid ABCD with AB || CD and AD, BC are the two lateral sides.
variables (trapezoid : Trapezoid A B C D)

-- Function that returns the midpoint of a line segment.
def midpoint (X Y : Point) : Point := sorry

-- Definitions for angle bisectors and midline
def bisects_angle (P Q R : Point) := sorry -- Definition for an angle bisector
def midline (X Y Z W : Point) := sorry    -- Definition of midline in trapezoid

-- The point O is the intersection of the bisectors of ∠BAD and ∠BCD
def intersection_of_bisectors (A D B C : Point) : Point := sorry

-- Proposition Statement: The point O lies on the midline of trapezoid ABCD.
theorem bisectors_intersect_on_midline 
  (h1 : Trapezoid A B C D)
  (h2 : bisects_angle A D O)
  (h3 : bisects_angle C B O)
  : midline (midpoint A D) (midpoint B C) O :=
  sorry

end bisectors_intersect_on_midline_l363_363198


namespace problem_quadratic_has_real_root_l363_363599

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l363_363599


namespace solve_trig_eq_l363_363715

open Real

theorem solve_trig_eq (x : ℝ) (m k : ℤ) : 
  (∀ k : ℤ, x ≠ (π / 2) * k) → 
  (sin (3 * x) ^ 2 / sin x ^ 2 = 8 * cos (4 * x) + cos (3 * x) ^ 2 / cos x ^ 2) ↔ 
  ∃ m : ℤ, x = (π / 3) * (2 * m + 1) :=
by
  sorry

end solve_trig_eq_l363_363715


namespace buzz_waiter_ratio_l363_363843

def total_slices : Nat := 78
def waiter_condition (W : Nat) : Prop := W - 20 = 28

theorem buzz_waiter_ratio (W : Nat) (h : waiter_condition W) : 
  let buzz_slices := total_slices - W
  let ratio_buzz_waiter := buzz_slices / W
  ratio_buzz_waiter = 5 / 8 :=
by
  sorry

end buzz_waiter_ratio_l363_363843


namespace sum_of_cosines_l363_363841

theorem sum_of_cosines
  (x y z : ℝ)
  (h1 : cos x + cos (y + π / 3) + cos (z - π / 3) = 0)
  (h2 : sin x + sin (y + π / 3) + sin (z - π / 3) = 0) :
  cos (2 * x) + cos (2 * y) + cos (2 * z) = 0 := by
  sorry

end sum_of_cosines_l363_363841


namespace prove_Φ_eq_8_l363_363012

-- Define the structure of the problem.
def condition (Φ : ℕ) : Prop := 504 / Φ = 40 + 3 * Φ

-- Define the main proof question.
theorem prove_Φ_eq_8 (Φ : ℕ) (h : condition Φ) : Φ = 8 := 
sorry

end prove_Φ_eq_8_l363_363012


namespace equation_solution_l363_363134

theorem equation_solution : ∃ x : ℝ, (3 / 20) + (3 / x) = (8 / x) + (1 / 15) ∧ x = 60 :=
by
  use 60
  -- skip the proof
  sorry

end equation_solution_l363_363134


namespace find_x_y_l363_363454

def A := (1, 2)
def B := (5, 4)
def C (x : Int) := (x, 3)
def D (y : Int) := (-3, y)

def vector_eq (p1 p2 : (Int × Int)) (p3 p4 : (Int × Int)) : Prop :=
  (p2.1 - p1.1 = p4.1 - p3.1) ∧ (p2.2 - p1.2 = p4.2 - p3.2)

theorem find_x_y (x y : Int) :
  let AB := vector_eq A B in
  let CD := vector_eq (C x) (D y) in
  AB = CD → x = -7 ∧ y = 5 :=
by
  intros h
  sorry

end find_x_y_l363_363454


namespace quadratic_has_real_root_iff_b_in_interval_l363_363543

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l363_363543


namespace negation_of_prop_l363_363515

theorem negation_of_prop :
  ¬ (∀ x : ℝ, x^2 - 1 > 0) ↔ ∃ x : ℝ, x^2 - 1 ≤ 0 :=
sorry

end negation_of_prop_l363_363515


namespace product_min_max_eq_24_l363_363171

theorem product_min_max_eq_24 (x y : ℝ) (h : 2 * x^2 + 3 * x * y + y^2 = 2) :
  let k := 4 * x^2 + 4 * x * y + 3 * y^2 in
  let m := Inf {k | ∃ x y : ℝ, 2 * x^2 + 3 * x * y + y^2 = 2 ∧ k = 4 * x^2 + 4 * x * y + 3 * y^2} in
  let M := Sup {k | ∃ x y : ℝ, 2 * x^2 + 3 * x * y + y^2 = 2 ∧ k = 4 * x^2 + 4 * x * y + 3 * y^2} in
  m * M = 24 :=
sorry

end product_min_max_eq_24_l363_363171


namespace no_rational_solutions_l363_363436

theorem no_rational_solutions : 
  ¬ ∃ (x y z : ℚ), 11 = x^5 + 2 * y^5 + 5 * z^5 := 
sorry

end no_rational_solutions_l363_363436


namespace find_base_l363_363255

-- Definitions based on the conditions of the problem
def is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
def is_perfect_cube (n : ℕ) := ∃ m : ℕ, m * m * m = n
def is_perfect_fourth (n : ℕ) := ∃ m : ℕ, m * m * m * m = n

-- Define the number A in terms of base a
def A (a : ℕ) : ℕ := 4 * a * a + 4 * a + 1

-- Problem statement: find a base a > 4 such that A is both a perfect cube and a perfect fourth power
theorem find_base (a : ℕ)
  (ha : a > 4)
  (h_square : is_perfect_square (A a)) :
  is_perfect_cube (A a) ∧ is_perfect_fourth (A a) :=
sorry

end find_base_l363_363255


namespace area_region_sum_l363_363250

theorem area_region_sum (r1 r2 : ℝ) (angle : ℝ) (a b c : ℕ) : 
  r1 = 6 → r2 = 3 → angle = 30 → (54 * Real.sqrt 3 + (9 : ℝ) * Real.pi - (9 : ℝ) * Real.pi = a * Real.sqrt b + c * Real.pi) → a + b + c = 10 :=
by
  intros
  -- We fill this with the actual proof steps later
  sorry

end area_region_sum_l363_363250


namespace length_EQ_is_correct_l363_363989

noncomputable def problem_statement : Prop :=
  let E := (0, 1) in
  let center_omega := (0.5, -0.5) in
  let N := (0.5, -1) in
  -- Equation of the circle: (x-0.5)^2 + (y+0.5)^2 = 1
  let EQ_length := 0.35 in
  (E.1 - 0.5)^2 + (E.2 + 0.5)^2 != 1 ∧
  (N.1 - 0.5)^2 + (N.2 + 0.5)^2 == 1 ∧
  E != N → 
  distance E center_omega != 0 →  -- Constraint derived from not being concentric
  EQ_length == 0.35

theorem length_EQ_is_correct : problem_statement :=
by sorry

end length_EQ_is_correct_l363_363989


namespace arithmetic_geometric_sequence_never_progression_l363_363062

theorem arithmetic_geometric_sequence_never_progression
  (a : ℕ → ℕ)
  (h_seq : ∀ n : ℕ, (a (2 * n) = (a (2 * n - 1) + a (2 * n + 1)) / 2) ∨ (a (2 * n + 1) = Int.to_nat (Int.sqrt (a (2 * n) * a (2 * n + 2)))))
  : ∀ N : ℕ, ¬ (∀ i : ℕ, i ≥ N → (a (i + 1) - a i = a (i + 2) - a (i + 1))) ∧ ¬ (∀ i : ℕ, i ≥ N → (a (i + 1) / a i = a (i + 2) / a (i + 1))) :=
by sorry

end arithmetic_geometric_sequence_never_progression_l363_363062


namespace arun_crosses_train_B_in_12_seconds_l363_363297

def length_train_A : ℝ := 150
def length_train_B : ℝ := 150
def speed_train_A_kmh : ℝ := 54
def speed_train_B_kmh : ℝ := 36
def kmh_to_ms (v : ℝ) : ℝ := v * (5 / 18)

theorem arun_crosses_train_B_in_12_seconds :
  let speed_train_A_ms := kmh_to_ms speed_train_A_kmh,
      speed_train_B_ms := kmh_to_ms speed_train_B_kmh,
      relative_speed := speed_train_A_ms + speed_train_B_ms,
      total_distance := length_train_A + length_train_B,
      crossing_time := total_distance / relative_speed
  in crossing_time = 12 :=
by
  sorry

end arun_crosses_train_B_in_12_seconds_l363_363297


namespace troy_needs_more_money_l363_363281

theorem troy_needs_more_money (initial_savings : ℕ) (sold_computer : ℕ) (new_computer_cost : ℕ) :
  initial_savings = 50 → sold_computer = 20 → new_computer_cost = 80 → 
  new_computer_cost - (initial_savings + sold_computer) = 10 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end troy_needs_more_money_l363_363281


namespace number_board_total_18_l363_363270

-- Define the conditions as Lean hypotheses
theorem number_board_total_18 (k : ℕ) (h1 : (k > 0)) (nums : list ℝ) 
  (h_sum_total : nums.sum = 1)
  (h_sum_five_smallest : ((nums.take 5).sum = 0.26))
  (h_sum_five_largest : ((nums.drop (k + 5)).sum = 0.29)) :
  (k + 10) = 18 :=
sorry

end number_board_total_18_l363_363270


namespace jenny_ate_65_chocolates_l363_363653

-- Define the number of chocolate squares Mike ate
def MikeChoc := 20

-- Define the function that calculates the chocolates Jenny ate
def JennyChoc (mikeChoc : ℕ) := 3 * mikeChoc + 5

-- The theorem stating the solution
theorem jenny_ate_65_chocolates (h : MikeChoc = 20) : JennyChoc MikeChoc = 65 := by
  -- Automatic proof step
  sorry

end jenny_ate_65_chocolates_l363_363653


namespace remainder_sum_binom_2024_div_2027_l363_363246

theorem remainder_sum_binom_2024_div_2027 :
  let R := ∑ k in finset.range (71), nat.choose 2024 k
  in R % 2027 = 1297 :=
by
  sorry

end remainder_sum_binom_2024_div_2027_l363_363246


namespace dan_final_produce_l363_363005

def initial_potatoes := 7
def initial_cantaloupes := 4
def initial_cucumbers := 5

def rabbits_ate_potatoes := initial_potatoes / 2
def squirrels_ate_cantaloupes := initial_cantaloupes / 4
def rabbits_ate_cucumbers := 2
def harvested_cucumbers := (initial_cucumbers - rabbits_ate_cucumbers) * 0.75
def gifted_cantaloupes := initial_cantaloupes * 0.5

theorem dan_final_produce :
  let remaining_potatoes := initial_potatoes - rabbits_ate_potatoes.floor in
  let remaining_cantaloupes := (initial_cantaloupes - squirrels_ate_cantaloupes) + gifted_cantaloupes in
  let remaining_cucumbers := (initial_cucumbers - rabbits_ate_cucumbers) - harvested_cucumbers.floor in
  remaining_potatoes = 4 ∧ remaining_cantaloupes = 5 ∧ remaining_cucumbers = 1 :=
by
  sorry

end dan_final_produce_l363_363005


namespace smallest_range_is_C_l363_363394

def range (s : List Int) : Int :=
  s.maximum - s.minimum

theorem smallest_range_is_C :
  let A := [0, 1, 2, 3, 4]
  let B := [-2, -1, -2, 3]
  let C := [110, 111, 112, 110, 109]
  let D := [-100, -200, -300, -400]
  range C < range A ∧ range C < range B ∧ range C < range D :=
by
  sorry

end smallest_range_is_C_l363_363394


namespace min_value_of_m_cauchy_schwarz_inequality_l363_363071

theorem min_value_of_m (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m = a + 1 / ((a - b) * b)) : 
  ∃ t, t = 3 ∧ ∀ a b : ℝ, a > b → b > 0 → m = a + 1 / ((a - b) * b) → m ≥ t :=
sorry

theorem cauchy_schwarz_inequality (x y z : ℝ) :
  (x^2 + 4 * y^2 + z^2 = 3) → |x + 2 * y + z| ≤ 3 :=
sorry

end min_value_of_m_cauchy_schwarz_inequality_l363_363071


namespace discount_on_pickles_l363_363156

theorem discount_on_pickles :
  ∀ (meat_weight : ℝ) (meat_price_per_pound : ℝ) (bun_price : ℝ) (lettuce_price : ℝ)
    (tomato_weight : ℝ) (tomato_price_per_pound : ℝ) (pickles_price : ℝ) (total_paid : ℝ) (change : ℝ),
  meat_weight = 2 ∧
  meat_price_per_pound = 3.50 ∧
  bun_price = 1.50 ∧
  lettuce_price = 1.00 ∧
  tomato_weight = 1.5 ∧
  tomato_price_per_pound = 2.00 ∧
  pickles_price = 2.50 ∧
  total_paid = 20.00 ∧
  change = 6 →
  pickles_price - (total_paid - change - (meat_weight * meat_price_per_pound + tomato_weight * tomato_price_per_pound + bun_price + lettuce_price)) = 1 := 
by
  -- Begin the proof here (not required for this task)
  sorry

end discount_on_pickles_l363_363156


namespace root_of_equation_l363_363440

theorem root_of_equation (x : ℝ) : 
  169 * (157 - 77 * x)^2 + 100 * (201 - 100 * x)^2 = 26 * (77 * x - 157) * (1000 * x - 2010) ↔ x = 31 := 
by 
  sorry

end root_of_equation_l363_363440


namespace ex_sq_sum_l363_363530

theorem ex_sq_sum (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = -1) : x^2 + y^2 = 11 :=
by
  sorry

end ex_sq_sum_l363_363530


namespace point_on_parabola_distance_line_parabola_intersections_l363_363513

-- Part (Ⅰ)
theorem point_on_parabola_distance (M : ℝ × ℝ) (focus : ℝ × ℝ := (1, 0)) :
  (∃ (y : ℝ), M = (-y^2 / 4, y) ∧ dist M focus = 5) ↔ 
  (M = (-4, 4) ∨ M = (-4, -4)) := 
sorry

-- Part (Ⅱ)
theorem line_parabola_intersections (k : ℝ) :
  (∃ (P : ℝ × ℝ), P = (1, 2) ∧
    if k = 0 then
      (∃! (M : ℝ × ℝ), M ∈ {(x, y) | y^2 = -4 * x} ∧ x = -1 ∧ y = 2) -- exactly one common point when k = 0
    else if k = 1 + sqrt 2 ∨ k = 1 - sqrt 2 then
      (∃! (M : ℝ × ℝ), M ∈ {(x, y) | y^2 = -4 * x} ∧ ∃ (y : ℝ), y = k * x - k + 2) -- one common point when k = 1 ± √2
    else if 1 - sqrt 2 < k ∧ k < 1 + sqrt 2 then
      (∃ (M1 M2 : ℝ × ℝ), M1 ≠ M2 ∧ M1 ∈ {(x, y) | y^2 = -4 * x} ∧ M2 ∈ {(x, y) | y^2 = -4 * x}
      ∧ ∃ (y1 y2 : ℝ), y1 = k * x - k + 2 ∧ y2 = k * x - k + 2) -- two common points when 1 - √2 < k < 1 + √2
    else
      (∀ (M : ℝ × ℝ), M ∉ {(x, y) | y^2 = -4 * x ∧ ∃ (y : ℝ), y = k * x - k + 2})) -- no common points when k > 1 + √2 or k < 1 - √2 :=
sorry

end point_on_parabola_distance_line_parabola_intersections_l363_363513


namespace quadratic_real_root_iff_b_range_l363_363582

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l363_363582


namespace only_set_d_forms_triangle_l363_363783

/-- Definition of forming a triangle given three lengths -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem only_set_d_forms_triangle :
  ¬ can_form_triangle 3 5 10 ∧ ¬ can_form_triangle 5 4 9 ∧ 
  ¬ can_form_triangle 5 5 10 ∧ can_form_triangle 4 6 9 :=
by {
  sorry
}

end only_set_d_forms_triangle_l363_363783


namespace arithmetic_sequence_length_l363_363955

theorem arithmetic_sequence_length (a d : ℕ) (l : ℕ) (h_a : a = 6) (h_d : d = 4) (h_l : l = 154) :
  ∃ n : ℕ, l = a + (n-1) * d ∧ n = 38 :=
by
  use 38
  split
  { rw [h_a, h_d]
    calc 154 = 6 + (38 - 1) * 4 : by norm_num
          ... = 6 + 37 * 4       : by rfl
          ... = 6 + 148          : by norm_num
          ... = 154              : by norm_num }
  { rfl }

end arithmetic_sequence_length_l363_363955


namespace not_possible_equal_monochromatic_dichromatic_sides_l363_363261

theorem not_possible_equal_monochromatic_dichromatic_sides :
  ∀ (color : Fin 222 → Prop), 
  let monochromatic := λ (i : Fin 222), color i = color (i + 1) in
  let dichromatic := λ (i : Fin 222), color i ≠ color (i + 1) in
  (Finset.card (Finset.filter monochromatic (Finset.univ : Finset (Fin 222)))
   ≠ Finset.card (Finset.filter dichromatic (Finset.univ : Finset (Fin 222)))) :=
by
  sorry

end not_possible_equal_monochromatic_dichromatic_sides_l363_363261


namespace limit_seq_eq_two_fifths_l363_363846

noncomputable def lim_seq : ℝ := 
  lim (λ n, (n^2 + real.sqrt n - 1) / (finset.sum (finset.range n) (λ k, 5*k - 3 + 2)))

theorem limit_seq_eq_two_fifths : lim_seq = 2 / 5 := by
  sorry

end limit_seq_eq_two_fifths_l363_363846


namespace certain_number_l363_363604

theorem certain_number (x y : ℝ) (h1 : 0.20 * x = 0.15 * y - 15) (h2 : x = 1050) : y = 1500 :=
by
  sorry

end certain_number_l363_363604


namespace range_of_quadratic_log_l363_363506

theorem range_of_quadratic_log (
    f : ℝ → ℝ → ℝ,
    h_f : ∀ (x : ℝ) (a : ℝ), f x a = (a^2 - 1)*x^2 - 2*(a - 1)*x + 3,
    h_log : ∀ (x : ℝ) (a : ℝ), ∃ (y : ℝ), y = Real.log (f x a)
) : ∀ (a : ℝ), -2 ≤ a ∧ a ≤ -1 ↔ ∀ (x : ℝ), f x a > 0 :=
by
  intros,
  sorry

end range_of_quadratic_log_l363_363506


namespace quadratic_has_real_root_iff_b_in_interval_l363_363547

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l363_363547


namespace simplify_expr_1_l363_363713

theorem simplify_expr_1 : 
  ( (16 / 81) ^ (-3 / 4) - 0.5 ^ (-1 / 2) + Real.pi ^ 0 - Real.sqrt ((Real.sqrt 2 - 3) ^ 2) ) = -46 / 27 :=
by
  sorry

end simplify_expr_1_l363_363713


namespace digit_A_of_3AA1_divisible_by_9_l363_363229

theorem digit_A_of_3AA1_divisible_by_9 (A : ℕ) (h : (3 + A + A + 1) % 9 = 0) : A = 7 :=
sorry

end digit_A_of_3AA1_divisible_by_9_l363_363229


namespace roots_in_quadrants_l363_363177

noncomputable def poly : Complex → Complex := λ z => z^6 + 6*z + 10

theorem roots_in_quadrants :
  ∃ roots : List Complex,
    roots.length = 6 ∧
    (roots.count (λ z => z.im > 0 ∧ z.re > 0) = 1) ∧
    (roots.count (λ z => z.im > 0 ∧ z.re < 0) = 2) ∧
    (roots.count (λ z => z.im < 0 ∧ z.re < 0) = 2) ∧
    (roots.count (λ z => z.im < 0 ∧ z.re > 0) = 1) ∧
    ∀ r ∈ roots, poly r = 0 :=
by
  sorry

end roots_in_quadrants_l363_363177


namespace part1_part2_l363_363504

noncomputable def f (x : ℝ) : ℝ := |x + 1| + 2 * |x - 1|

theorem part1 (x : ℝ) : f x < 4 ↔ -1 < x ∧ x < (5:ℝ)/3 := by
  sorry

theorem part2 (a : ℝ) : (∀ x, f x ≥ |a + 1|) ↔ -3 ≤ a ∧ a ≤ 1 := by
  sorry

end part1_part2_l363_363504


namespace quadratic_has_real_root_l363_363571

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l363_363571


namespace log_ordering_l363_363166

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 8 / Real.log 4
noncomputable def c : ℝ := Real.log 10 / Real.log 5

theorem log_ordering : a > b ∧ b > c :=
by {
  sorry
}

end log_ordering_l363_363166


namespace fixed_point_exists_with_ratio_l363_363063

noncomputable theory

variable (P Q R : Type) [metric_space R]

def circleO (p : P) : Prop :=
  p.x ^ 2 + p.y ^ 2 = 1

def point_M (p : P) : Prop :=
  p.x = 4 ∧ p.y = 2

def circle_M_eq (p : P) : Prop :=
  (p.x - 4) ^ 2 + (p.y - 2) ^ 2 = 9

def tangent_condition (P Q R : P) (λ : ℝ) : Prop :=
  let PQ := dist P Q
  let PR := dist P R
  (PQ / PR) = λ

theorem fixed_point_exists_with_ratio (P Q : P) : 
  ∃ (R : P) (λ : ℝ), 
  circleO Q →  (tangent_condition P Q R λ ∧ (R.x = 2 ∧ R.y = 1 ∧ λ = real.sqrt 2) ∨
               (R.x = 0.4 ∧ R.y = 0.2 ∧ λ = real.sqrt 10 / 3)) := 
sorry

end fixed_point_exists_with_ratio_l363_363063


namespace floor_expression_eq_eight_l363_363853

theorem floor_expression_eq_eight :
  (Int.floor (1005^3 / (1003 * 1004) - 1003^3 / (1004 * 1005)) = 8) :=
sorry

end floor_expression_eq_eight_l363_363853


namespace subset_implies_a_geq_4_l363_363456

open Set

def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + 3 ≤ 0}

theorem subset_implies_a_geq_4 (a : ℝ) :
  A ⊆ B a → a ≥ 4 := sorry

end subset_implies_a_geq_4_l363_363456


namespace drums_per_day_l363_363842

theorem drums_per_day (D : ℕ) (T : ℕ) (n_pick : ℕ) (h : n_pick = 94) (h1 : D = 90) (h2 : T = 6) :
  D / T = 15 :=
by
  -- Conditions in Lean will convert as hypothesis for the theorem
  rw [h1, h2]
  -- Simplify the division to show the desired result
  norm_num

-- A note to mention that the proof is replaced by sorry for now
-- the proof would norm_num to show the result
-- end the proof with q.e.d (Latin: quod erat demonstrandum) meaning "which was to be demonstrated"

end drums_per_day_l363_363842


namespace find_sale4_l363_363818

variable (sale1 sale2 sale3 sale5 sale6 sale7 : ℕ)

-- Conditions
def sale1 : ℕ := 5921
def sale2 : ℕ := 5468
def sale3 : ℕ := 5568
def sale5 : ℕ := 6433
def sale6 : ℕ := 5922
def avg_sale : ℕ := 5900
def number_of_months : ℕ := 6

-- Question (prove that sale4 = 6088 to maintain the average of 5900)
theorem find_sale4 (sale4 : ℕ) :
  sale1 + sale2 + sale3 + sale4 + sale5 + sale6 = avg_sale * number_of_months → sale4 = 6088 :=
by
  intros h
  sorry

end find_sale4_l363_363818


namespace sufficient_condition_l363_363916

variable (a b c d : ℝ)

-- Condition p: a and b are the roots of the equation.
def condition_p : Prop := a * a + b * b + c * (a + b) + d = 0

-- Condition q: a + b + c = 0
def condition_q : Prop := a + b + c = 0

theorem sufficient_condition : condition_p a b c d → condition_q a b c := by
  sorry

end sufficient_condition_l363_363916


namespace intervals_of_monotonic_increase_max_area_of_acute_triangle_l363_363168

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - cos (x + π/4)^2

theorem intervals_of_monotonic_increase :
  ∀ k : ℤ, ∃ a b : ℝ, [-π/4 + k * π, π/4 + k * π] = Icc a b :=
sorry

theorem max_area_of_acute_triangle (A : ℝ) (B C b c : ℝ) (hA : 0 < A ∧ A < π/2) 
  (ha : a = 1) (hf : f (A/2) = 0) (h : 0 < B ∧ B < π/2) 
  (hc : cos A = √3/2)
  (hb : b^2 + c^2 = 1 + √3 * b * c) :
  ∃ S : ℝ, S = (2 + √3) / 4 :=
sorry

end intervals_of_monotonic_increase_max_area_of_acute_triangle_l363_363168


namespace substitution_and_elimination_l363_363721

theorem substitution_and_elimination {x y : ℝ} :
  y = 2 * x + 1 → 5 * x - 2 * y = 7 → 5 * x - 4 * x - 2 = 7 :=
by
  intros h₁ h₂
  rw [h₁] at h₂
  exact h₂

end substitution_and_elimination_l363_363721


namespace simplify_and_evaluate_expression_l363_363714

theorem simplify_and_evaluate_expression 
  (x : ℝ) 
  (hx : x = -3.2) : 
  ((x^2 - 4x + 4) / (x^2 - 4) / (x - 2) / (x^2 + 2x) + 3 = -10.8666) :=
by 
  sorry

end simplify_and_evaluate_expression_l363_363714


namespace cumulative_distribution_function_correct_l363_363826

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ real.pi / 4 then 0
  else if x ≤ real.pi / 2 then 2 * real.sin (2 * x)
  else 0 

noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ real.pi / 4 then 0
  else if x ≤ real.pi / 2 then -real.cos (2 * x)
  else 1

theorem cumulative_distribution_function_correct : 
  ∀ x, F x = ∫ t in Iic x, f t := 
sorry

end cumulative_distribution_function_correct_l363_363826


namespace original_number_is_7_l363_363839

theorem original_number_is_7 (x : ℤ) (h : (((3 * (x + 3) + 3) - 3) / 3) = 10) : x = 7 :=
sorry

end original_number_is_7_l363_363839


namespace analytic_expression_monotonic_intervals_max_min_values_l363_363507

-- Definitions based on conditions
def f (x : ℝ) : ℝ := a * x^2 + b * x + c
def a_not_zero : Prop := a ≠ 0
def f_at_0 : Prop := f 0 = 2
def f_difference : Prop := ∀ x : ℝ, f (x + 1) - f x = 2 * x - 1

-- Main Theorems
theorem analytic_expression (a b c : ℝ) (h_a_not_zero : a_not_zero) (h_f_at_0 :  f_at_0) (h_f_difference : f_difference) : 
  f x = x^2 - 2*x + 2 := 
sorry

theorem monotonic_intervals (a b c : ℝ) (h_a_not_zero : a_not_zero) (h_f_at_0 : f_at_0) (h_f_difference : f_difference) : 
  (∀ x : ℝ, x > 1 → f (x + 1) ≥ f x) ∧ (∀ x : ℝ, x < 1 → f (x + 1) ≤ f x) := 
sorry

theorem max_min_values (a b c : ℝ) (h_a_not_zero : a_not_zero) (h_f_at_0 : f_at_0) (h_f_difference : f_difference) : 
  ∀ x ∈ [-1, 2], ∃ (f_max f_min : ℝ), (f x) ≤ f_max ∧ (f x) ≥ f_min ∧ f_max = 5 ∧ f_min = 1 :=
sorry

end analytic_expression_monotonic_intervals_max_min_values_l363_363507


namespace ratio_black_to_white_l363_363039

-- Define the radii of the concentric circles
def r1 : ℝ := 3
def r2 : ℝ := 5
def r3 : ℝ := 7
def r4 : ℝ := 9

-- Define the areas of the individual circles
def A1 : ℝ := π * r1^2
def A2 : ℝ := π * r2^2
def A3 : ℝ := π * r3^2
def A4 : ℝ := π * r4^2

-- Calculate the black and white areas
def white_area : ℝ := A1 + (A3 - A2)
def black_area : ℝ := (A2 - A1) + (A4 - A3)

-- The target ratio
def target_ratio : ℝ := 16 / 11

-- The statement to be proved
theorem ratio_black_to_white : black_area / white_area = target_ratio := by
  sorry

end ratio_black_to_white_l363_363039


namespace trigonometric_identities_l363_363316

open Real

theorem trigonometric_identities :
  (cos 75 * cos 75 = (2 - sqrt 3) / 4) ∧
  ((1 + tan 105) / (1 - tan 105) ≠ sqrt 3 / 3) ∧
  (tan 1 + tan 44 + tan 1 * tan 44 = 1) ∧
  (sin 70 * (sqrt 3 / tan 40 - 1) ≠ 2) :=
by
  sorry

end trigonometric_identities_l363_363316


namespace regular_222_gon_no_equal_mono_dichromatic__l363_363259

def regular_n_gon (n : ℕ) := true

def is_colored (vertices : ℕ → bool) := true

def side_is_monochromatic (vertices : ℕ → bool) (i : ℕ) : bool :=
vertices i = vertices ((i + 1) % 222)

def side_is_dichromatic (vertices : ℕ → bool) (i : ℕ) : bool :=
¬ side_is_monochromatic vertices i

theorem regular_222_gon_no_equal_mono_dichromatic_ (vertices : ℕ → bool)
  (h: regular_n_gon 222) (hv: is_colored vertices) :
  ¬ (∃ m d : ℕ, m + d = 222 ∧ m = d ∧ (∀ i, side_is_monochromatic vertices i ∨ side_is_dichromatic vertices i)) :=
sorry

end regular_222_gon_no_equal_mono_dichromatic__l363_363259


namespace same_sum_sufficient_days_l363_363789

variable {S Wb Wc : ℝ}
variable (h1 : S = 12 * Wb)
variable (h2 : S = 24 * Wc)

theorem same_sum_sufficient_days : ∃ D : ℝ, D = 8 ∧ S = D * (Wb + Wc) :=
by
  use 8
  sorry

end same_sum_sufficient_days_l363_363789


namespace car_late_speed_l363_363359

theorem car_late_speed :
  ∀ (d : ℝ) (t_on_time : ℝ) (t_late : ℝ) (v_on_time : ℝ) (v_late : ℝ),
  d = 225 →
  v_on_time = 60 →
  t_on_time = d / v_on_time →
  t_late = t_on_time + 0.75 →
  v_late = d / t_late →
  v_late = 50 :=
by
  intros d t_on_time t_late v_on_time v_late hd hv_on_time ht_on_time ht_late hv_late
  sorry

end car_late_speed_l363_363359


namespace apple_juice_cost_l363_363299

noncomputable def cost_of_apple_juice (cost_per_orange_juice : ℝ) (total_bottles : ℕ) (total_cost : ℝ) (orange_juice_bottles : ℕ) : ℝ :=
  (total_cost - cost_per_orange_juice * orange_juice_bottles) / (total_bottles - orange_juice_bottles)

theorem apple_juice_cost :
  let cost_per_orange_juice := 0.7
  let total_bottles := 70
  let total_cost := 46.2
  let orange_juice_bottles := 42
  cost_of_apple_juice cost_per_orange_juice total_bottles total_cost orange_juice_bottles = 0.6 := by
    sorry

end apple_juice_cost_l363_363299


namespace total_cost_is_9220_l363_363821

-- Define the conditions
def hourly_rate := 60
def hours_per_day := 8
def total_days := 14
def cost_of_parts := 2500

-- Define the total cost the car's owner had to pay based on conditions
def total_hours := hours_per_day * total_days
def labor_cost := total_hours * hourly_rate
def total_cost := labor_cost + cost_of_parts

-- Theorem stating that the total cost is $9220
theorem total_cost_is_9220 : total_cost = 9220 := by
  sorry

end total_cost_is_9220_l363_363821


namespace proof_problem_l363_363066

variables {a b : Type} {α β : Type} [Lin a] [Lin b] [Plane α] [Plane β]
variables {a_parallel_b : ∀ {a b : Line}, Prop}
variables {a_parallel_alpha : ∀ {a : Line} {α : Plane}, Prop}
variables {a_perp_b : ∀ {a b : Line}, Prop}
variables {alpha_perp_beta : ∀ {α β : Plane}, Prop}
variables {alpha_cap_beta : ∀ {α β : Plane}, Line}

-- Definitions for translating propositions
def proposition_1 (a b : Line) : Prop := (a_parallel_b a b) → ∀ (π : Plane), ∀ (x : Line), (x = b) → ¬(a \parallel π)
def proposition_2 (a : Line) (α : Plane) (b : Line) : Prop := (a_parallel_alpha a α) ∧ (a_parallel_alpha b α) → ¬(a_parallel_b a b)
def proposition_3 (a : Line) (α : Plane) (b : Line) (β : Plane) : Prop := (a_parallel_alpha a α) ∧ (alpha_perp_beta α β) ∧ (a_parallel_alpha b β) → ¬(a_perp_b a b)
def proposition_4 (a : Line) (α : Plane) (b : Line) (β : Plane) : Prop := (alpha_cap_beta α β = a) ∧ (a_parallel_alpha b α) → ¬(a_parallel_b b a)

theorem proof_problem : proposition_1 a b → proposition_2 a α b → proposition_3 a α b β → proposition_4 a α b β
:= by sorry

end proof_problem_l363_363066


namespace frank_fencemaker_fence_length_l363_363792

theorem frank_fencemaker_fence_length :
  ∃ (L W : ℕ), W = 40 ∧
               (L * W = 200) ∧
               (2 * L + W = 50) :=
by
  sorry

end frank_fencemaker_fence_length_l363_363792


namespace bridge_length_calculation_l363_363333

def length_of_bridge
  (length_of_train : ℕ)
  (speed_kmh : ℕ)
  (time_seconds : ℕ)
  (bridge_length : ℝ) : Prop :=
  let speed_mps := (speed_kmh * 1000.0) / 3600.0 in
  let distance_travelled := speed_mps * time_seconds in
  (distance_travelled - (length_of_train : ℝ) = bridge_length)

-- Now we state the theorem
theorem bridge_length_calculation :
  length_of_bridge 150 35 25 93.05 :=
by
  sorry

end bridge_length_calculation_l363_363333


namespace find_stream_speed_l363_363809

-- Define the conditions
def boat_speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def upstream_time : ℝ := 1.5
def speed_of_stream (v : ℝ) : Prop :=
  let downstream_speed := boat_speed_in_still_water + v
  let upstream_speed := boat_speed_in_still_water - v
  (downstream_speed * downstream_time) = (upstream_speed * upstream_time)

-- Define the theorem to prove
theorem find_stream_speed : ∃ v, speed_of_stream v ∧ v = 3 :=
by {
  sorry
}

end find_stream_speed_l363_363809


namespace min_value_l363_363178

noncomputable def min_value_fun (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) : ℝ :=
  1 / (Real.sin x)^2 + (12 * Real.sqrt 3) / Real.cos x

theorem min_value : ∀ x ∈ Ioo 0 (Real.pi / 2), min_value_fun x ⟨x.1, x.2⟩ ≥ 28 :=
begin
  sorry
end

end min_value_l363_363178


namespace trains_crossing_time_l363_363798

-- Definitions based on given conditions
def length_train1 : ℝ := 140   -- Length of the first train in meters
def length_train2 : ℝ := 190   -- Length of the second train in meters
def speed_train1_kmh : ℝ := 60 -- Speed of the first train in km/hr
def speed_train2_kmh : ℝ := 40 -- Speed of the second train in km/hr

-- Conversion factors
def kmh_to_mps (v : ℝ) : ℝ := v * (5 / 18) -- Convert km/hr to m/s

-- Speeds in meters per second
def speed_train1_mps : ℝ := kmh_to_mps speed_train1_kmh
def speed_train2_mps : ℝ := kmh_to_mps speed_train2_kmh

-- Relative speed in meters per second
def relative_speed_mps : ℝ := speed_train1_mps + speed_train2_mps

-- Total distance to be covered when trains cross each other
def total_distance : ℝ := length_train1 + length_train2

-- Correct answer
def expected_time : ℝ := 11.88

-- Statement to prove
theorem trains_crossing_time : total_distance / relative_speed_mps = expected_time := by
  sorry

end trains_crossing_time_l363_363798


namespace log_sum_l363_363531

theorem log_sum {a b : ℝ} (h₁ : 10^a = 5) (h₂ : 10^b = 2) : a + b = 1 :=
by
  sorry

end log_sum_l363_363531


namespace simplify_fraction_l363_363710

open Real

theorem simplify_fraction : 
  (60 * (π / 180) = real.pi / 3) →
  (∀ x, tan x = sin x / cos x) →
  (∀ x, cot x = 1 / tan x) →
  let t := tan (real.pi / 3)
  let c := cot (real.pi / 3)
  t = sqrt 3 →
  c = 1 / sqrt 3 →
  (t ^ 3 + c ^ 3) / (t + c) = 7 / 3 :=
by
  intro h60 htan hcot t_def c_def
  sorry

end simplify_fraction_l363_363710


namespace ratio_of_areas_l363_363976

noncomputable def side_length_of_second_square := s2 : Real

noncomputable def diagonal_of_second_square (s2 : Real) : Real :=
  s2 * Real.sqrt 2

noncomputable def side_length_of_first_square (s2 : Real) : Real :=
  2 * (diagonal_of_second_square s2)

noncomputable def area_of_first_square (s2 : Real) : Real :=
  (side_length_of_first_square s2) ^ 2

noncomputable def area_of_second_square (s2 : Real) : Real :=
  s2 ^ 2

theorem ratio_of_areas (s2 : Real) : 
  (area_of_first_square s2) / (area_of_second_square s2) = 8 :=
by
  sorry

end ratio_of_areas_l363_363976


namespace quadratic_roots_interval_l363_363589

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l363_363589


namespace least_difference_l363_363701

def geometric_sequence (a₀ r : ℕ) (limit : ℕ) : List ℕ :=
  List.takeWhile (≤ limit) (List.iterate (· * r) a₀)

def arithmetic_sequence (a₀ d : ℕ) (limit : ℕ) : List ℕ :=
  List.takeWhile (≤ limit) (List.iterate (· + d) a₀)

def min_positive_difference (seqA seqB : List ℕ) : ℕ :=
  seqA.bind (λ a => seqB.filter (λ b => b > a).map (λ b => b - a))
       |>.minimum
       |>.getOrElse 0

theorem least_difference :
  let A := geometric_sequence 3 2 400
  let B := arithmetic_sequence 10 15 400
  min_positive_difference A B = 4 := by
  sorry

end least_difference_l363_363701


namespace find_BD_correct_l363_363644

variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (a b c d : A) (ab bd : ℝ)

def triangle_AC_BC_equal (a b c : A) : A :=
  AC = 10 ∧ BC = 10

def point_on_AB (a b d : A) : Prop :=
  ∃ t ∈ segment (A,B), d = (1 - t) * a + t * b

def find_BD (a b c : A) (d_bd : ℝ) : Prop :=
  (BD = 7.85)

theorem find_BD_correct (a b c d : A) (h1 : AC = 10) (h2 : BC = 10) (h3 : AD = 12) (h4 : CD = 5) (h5 : point_on_AB a b d) :
  BD = 7.85 :=
sorry

end find_BD_correct_l363_363644


namespace general_formula_for_an_sum_of_bn_exists_lambda_l363_363059

variable {a b c : ℕ → ℕ}
variable {λ : ℝ}

/- (1) Given S_n = n^2 + 2n, prove a_n = 2n + 1 -/
def Sn (n : ℕ) := n^2 + 2 * n
def an (n : ℕ) := 2 * n + 1

theorem general_formula_for_an (n : ℕ) :
  ∑ i in finset.range n, an i = Sn n :=
sorry

/- (2) Given bn = 1 / Sn and Sn = n^2 + 2n, prove Tn = (3n^2 + 5n) / (4(n + 1)(n + 2)) -/
def bn (n : ℕ) := 1 / (Sn n : ℝ)
def Tn (n : ℕ) := (3 * n^2 + 5 * n) / (4 * (n + 1) * (n + 2))

theorem sum_of_bn (n : ℕ) :
  ∑ i in finset.range n, bn i = Tn n :=
sorry

/- (3) Given c_(n+1) = a_(c_n) + 2^n and c_1 = 3, prove there exists λ = 1 such that the sequence { (c_n + λ) / 2^n } forms an arithmetic sequence -/
def cn : ℕ → ℕ
| 0 := 3
| (n + 1) := an (cn n) + 2^(n : ℕ)

def form_arithmetic_seq (λ : ℝ) (n : ℕ) :=
  (cn (n + 1) + λ) / 2^(n + 1) - (cn n + λ) / 2^n = (cn 1 + λ) / 2

theorem exists_lambda (λ : ℝ) :
  (λ = 1) → ∀ n : ℕ, form_arithmetic_seq λ n :=
sorry

end general_formula_for_an_sum_of_bn_exists_lambda_l363_363059


namespace no_such_polynomial_l363_363204

theorem no_such_polynomial :
  ¬ ∃ P : ℚ[X], ∀ n : ℕ, P.eval n = (n^2 + 1)^(1/3 : ℝ) :=
by {
  sorry
}

end no_such_polynomial_l363_363204


namespace katie_new_games_l363_363155

theorem katie_new_games (K : ℕ) (h : K + 8 = 92) : K = 84 :=
by
  sorry

end katie_new_games_l363_363155


namespace find_nat_numbers_l363_363881

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

theorem find_nat_numbers (n : ℕ) :
  (n + sum_of_digits n = 2021) ↔ (n = 2014 ∨ n = 1996) :=
by
  sorry

end find_nat_numbers_l363_363881


namespace eva_laces_l363_363021

theorem eva_laces (pairs_of_shoes : ℕ) (laces_per_pair : ℕ) (h1 : pairs_of_shoes = 26) (h2 : laces_per_pair = 2) : 
  pairs_of_shoes * laces_per_pair = 52 := 
by 
  rw [h1, h2]
  exact Nat.mul_comm 26 2 ▸ rfl

end eva_laces_l363_363021


namespace problem_l363_363048

def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

def f' (x : ℝ) : ℝ := - (Real.sin x / x) - (Real.cos x / (x * x))

theorem problem (h1 : f π = -1 / π) (h2 : f' (π / 2) = -2 / π) : f π + f' (π / 2) = -3 / π := by
  sorry

end problem_l363_363048


namespace fraction_surface_area_red_l363_363019

theorem fraction_surface_area_red :
  ∀ (num_unit_cubes : ℕ) (side_length_large_cube : ℕ) (total_surface_area_painted : ℕ) (total_surface_area_unit_cubes : ℕ),
    num_unit_cubes = 8 →
    side_length_large_cube = 2 →
    total_surface_area_painted = 6 * (side_length_large_cube ^ 2) →
    total_surface_area_unit_cubes = num_unit_cubes * 6 →
    (total_surface_area_painted : ℝ) / total_surface_area_unit_cubes = 1 / 2 :=
by
  intros num_unit_cubes side_length_large_cube total_surface_area_painted total_surface_area_unit_cubes
  sorry

end fraction_surface_area_red_l363_363019


namespace quadrilateral_angles_l363_363065

-- Lean 4 statement
theorem quadrilateral_angles (A B C D : ℝ × ℝ) :
  dist A B = dist B C ∧ dist B C = dist C D ∧
  dist B D = dist D A ∧ dist D A = dist A C →
  ∃ θ φ : ℝ, θ = 72 ∧ φ = 108 ∧
  interiorAngle A B C = θ ∧
  interiorAngle B C D = θ ∧
  interiorAngle C D A = φ ∧
  interiorAngle D A B = φ :=
sorry

end quadrilateral_angles_l363_363065


namespace find_p_and_q_sqrt_sum_leq_l363_363116

/-- Given the inequality |2x - 3| < 4, determine the values of p and q such that the solution set
of x^2 + px + q < 0 is the same. Prove that this implies p = -3 and q = -7/4. -/
theorem find_p_and_q (p q : ℝ) :
  (∀ x : ℝ, |2 * x - 3| < 4 ↔ x^2 + p * x + q < 0) →
  p = -3 ∧ q = -7 / 4 :=
sorry

/-- Given  a, b, c in ℝ_+ and a + b + c = 2p - 4q, prove that sqrt(a) + sqrt(b) + sqrt(c) ≤ sqrt(3). -/
theorem sqrt_sum_leq (a b c p q : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c = 2 * p - 4 * q →
  sqrt a + sqrt b + sqrt c ≤ sqrt 3 :=
sorry

  
end find_p_and_q_sqrt_sum_leq_l363_363116


namespace value_of_m_l363_363485

theorem value_of_m (m : ℝ) (h : (m - 3) * x ^ |m - 2| + 6 = 0) (hx : is_linear (λ x, (m - 3) * x ^ |m - 2| + 6)) : m = 1 :=
by
  sorry

end value_of_m_l363_363485


namespace additional_cost_per_pint_proof_l363_363381

-- Definitions based on the problem conditions
def pints_sold := 54
def total_revenue_on_sale := 216
def revenue_difference := 108

-- Derived definitions
def revenue_if_not_on_sale := total_revenue_on_sale + revenue_difference
def cost_per_pint_on_sale := total_revenue_on_sale / pints_sold
def cost_per_pint_not_on_sale := revenue_if_not_on_sale / pints_sold
def additional_cost_per_pint := cost_per_pint_not_on_sale - cost_per_pint_on_sale

-- Proof statement
theorem additional_cost_per_pint_proof :
  additional_cost_per_pint = 2 :=
by
  -- Placeholder to indicate that the proof is not provided
  sorry

end additional_cost_per_pint_proof_l363_363381


namespace hemisphere_surface_area_including_cutout_l363_363254

theorem hemisphere_surface_area_including_cutout (r R : ℝ) (hsphere : 4 * π * R^2) :
  R = 10 → r = 3 → 291 * π = (π * R^2 + (1/2) * hsphere - π * r^2) :=
by
  intros hR hr
  sorry

end hemisphere_surface_area_including_cutout_l363_363254


namespace fibonacci_6_l363_363729

def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

theorem fibonacci_6 : fib 6 = 8 := by
  -- Here we would provide the proof steps, but we use sorry for now
  sorry

end fibonacci_6_l363_363729


namespace quadratic_has_real_root_l363_363566

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l363_363566


namespace smallest_positive_perfect_square_divisible_by_2_and_5_l363_363774

theorem smallest_positive_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, n = k^2) ∧ (2 ∣ n) ∧ (5 ∣ n) ∧ n = 100 :=
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_and_5_l363_363774


namespace equation_of_ellipse_range_of_ratio_l363_363631

-- Define the condition: a > b > 0 and a^2 - b^2 = 1
variables {a b : ℝ} (ha : a > 0) (hb : b > 0) (h_ab : a > b) (h1 : a^2 - b^2 = 1)

-- Define the ellipse and parabola equations
def ellipse (x y a b : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def parabola (x y : ℝ) := y^2 = 4 * x

-- Define point F2 (focus of the parabola) and its fixed value
def F2 : ℝ × ℝ := (1, 0)

-- Intersection point M and distance |MF2| = 5/3
variables {M_x M_y : ℝ}
def intersection_point (M_x M_y : ℝ) := ellipse M_x M_y a b ∧ parabola M_x M_y ∧ (M_x - 1)^2 + M_y^2 = (5 / 3)^2

-- Definition of point D and line l passing through D intersecting the ellipse at A and B
def D : ℝ × ℝ := (4, 0)
variables {A B : ℝ × ℝ}
def line_l (m : ℝ) := ∃ (x y : ℝ), (x = m * y + 4) ∧ (x^2 / 4 + y^2 / 3 = 1) ∧ A = (x, y) ∧ B = (x, y) ∧ (D.1 < A.1 < B.1)

-- Prove the equation of the ellipse
theorem equation_of_ellipse (hM : intersection_point M_x M_y) : ∀ x y, ellipse x y 2.sqrt 3.sqrt := by
  sorry

-- Prove the range of the ratio λ
theorem range_of_ratio (hA hB : line_l 4) (λ : ℝ) : 1 / 3 < λ ∧ λ < 1 := by
  sorry

end equation_of_ellipse_range_of_ratio_l363_363631


namespace find_k_range_l363_363037

/--
For the sequence {a_n}, define An as 
Aₙ = (a₁ + 2a₂ + ⋯ + 2^(n-1)•aₙ) / n 
as the "good number" of the sequence {a_n}. 

It is known that the "good number" Aₙ = 2^(n+1) for a certain sequence {a_n}. 
Let the sum of the first n terms of the sequence {a_n - kn} be Sₙ. 

Then prove that the condition Sₙ ≤ S₆ for all n ∈ ℕ* implies k ∈ [16/7, 7/3].
-/
theorem find_k_range (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℝ) :
  (∀ n : ℕ, n > 0 → (S n = ∑ i in finset.range n, (a (i + 1) - k * n)) → S n ≤ S 6) →
  (16 / 7 : ℝ) ≤ k ∧ k ≤ (7 / 3 : ℝ) :=
by 
  sorry

end find_k_range_l363_363037


namespace problem_statement_l363_363517

-- Definitions of sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2x > 0}

theorem problem_statement : 
  (A ∩ B = {x | x < 3 / 2}) ∧
  (A ∪ B ≠ {x | x < 3 / 2}) ∧
  (A ∪ B ≠ Set.univ) :=
by
  -- To be proven
  sorry

end problem_statement_l363_363517


namespace sum_of_slope_and_intercept_l363_363133

theorem sum_of_slope_and_intercept
  (A B C : ℝ × ℝ)
  (D : ℝ × ℝ)
  (hA : A = (0, 8))
  (hB : B = (0, 0))
  (hC : C = (10, 0))
  (hD : D = ((0 + 0) / 2, (8 + 0) / 2)) :
  let slope := (D.snd - C.snd) / (D.fst - C.fst),
      y_intercept := D.snd
  in slope + y_intercept = 18 / 5 :=
by
  sorry

end sum_of_slope_and_intercept_l363_363133


namespace find_a_solve_inequality_intervals_of_monotonicity_l363_363898

-- Problem 1: Prove a = 2 given conditions
theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : Real.log 3 / Real.log a > Real.log 2 / Real.log a) 
    (h₃ : Real.log (2 * a) / Real.log a - Real.log a / Real.log a = 1) : a = 2 := 
  by
  sorry

-- Problem 2: Prove the solution interval for inequality
theorem solve_inequality (x a : ℝ) (h₀ : 1 < x) (h₁ : x < 3 / 2) : 
    Real.log (x - 1) / Real.log (1 / 3) > Real.log (a - x) / Real.log (1 / 3) :=
  by
  have ha : a = 2 := sorry
  sorry

-- Problem 3: Prove intervals of monotonicity for g(x)
theorem intervals_of_monotonicity (x : ℝ) : 
  (∀ x : ℝ, 0 < x → x ≤ 2 → (|Real.log x / Real.log 2 - 1| : ℝ) = 1 - Real.log x / Real.log 2) ∧ 
  (∀ x : ℝ, x > 2 → (|Real.log x / Real.log 2 - 1| : ℝ) = Real.log x / Real.log 2 - 1) :=
  by
  sorry

end find_a_solve_inequality_intervals_of_monotonicity_l363_363898


namespace largest_whole_number_solution_for_inequality_l363_363305

theorem largest_whole_number_solution_for_inequality :
  ∀ (x : ℕ), ((1 : ℝ) / 4 + (x : ℝ) / 5 < 2) → x ≤ 23 :=
by sorry

end largest_whole_number_solution_for_inequality_l363_363305


namespace find_locus_of_centroids_l363_363472

noncomputable def locus_of_centroids_of_triangle (A B : Point) (α β : Line) (hα : A ∈ α) (hβ : A ∈ β)
  (C D : Point) (γ : Circle) (hγA : A ∈ γ) (hγB : B ∈ γ) (hC : C ∈ γ ∧ C ≠ A ∧ C ∈ α)
  (hD : D ∈ γ ∧ D ≠ A ∧ D ∈ β) : Set Point :=
{ G : Point | ∃ N, N ∈ LineSegment C D ∧ N = Midpoint C D ∧ 
  G = Centroid A C D ∧ is_parallel (LineThrough G N) (LineThrough A B / 3)}

theorem find_locus_of_centroids (A B C D : Point) (α β : Line) 
  (hα : A ∈ α) (hβ : A ∈ β) (hC : C ∈ α) (hD : D ∈ β)
  (γ : Circle) (hγA : A ∈ γ) (hγB : B ∈ γ) (hCγ : C ∈ γ ∧ C ≠ A)
  (hDγ : D ∈ γ ∧ D ≠ A) :
  locus_of_centroids_of_triangle A B α β hα hβ C D γ hγA hγB hCγ hDγ 
  = { G | ∃ K, K ∈ LineThrough A B / 3 ∧ G = PointOnLineParallelTo LK A / 2} :=
sorry

end find_locus_of_centroids_l363_363472


namespace point_on_line_l363_363237

theorem point_on_line : ∀ (x y : ℝ), (x = 2 ∧ y = 7) → (y = 3 * x + 1) := 
by 
  intros x y h
  cases h with hx hy
  rw [hx, hy]
  sorry

end point_on_line_l363_363237


namespace complex_conjugate_roots_l363_363660

theorem complex_conjugate_roots (c d : ℝ) :
  (∀ w : ℂ, w^2 + (15 + c * complex.I) * w + (40 + d * complex.I) = 0 →
   ∃ u v : ℝ, w = u + v * complex.I ∧ v ≠ 0 ∧ w = u - v * complex.I) →
  (c = 0 ∧ d = 0) := 
sorry

end complex_conjugate_roots_l363_363660


namespace monotonically_decreasing_m_maximum_value_f_l363_363087

def f (m x : ℝ) : ℝ := x^2 + 2 * m * x + 3 * m + 4

-- To be proven that m ∈ (-∞, -1] implies f(x) is monotonically decreasing on (-∞, 1]
theorem monotonically_decreasing_m (m : ℝ) : (∀ x y : ℝ, x ≤ y → x ∈ (-∞, 1] → y ∈ (-∞, 1] → (f m y ≤ f m x)) ↔ m ∈ (-∞, -1]) :=
by sorry

-- To be proven that the maximum value on [0, 2] is correctly defined by g(m)
def g (m : ℝ) : ℝ :=
if m ≤ -1 then 3 * m + 4 else 7 * m + 8

theorem maximum_value_f (m : ℝ) : g m = (if m ≤ -1 then 3 * m + 4 else 7 * m + 8) :=
by sorry

end monotonically_decreasing_m_maximum_value_f_l363_363087


namespace minimum_value_expression_l363_363464

theorem minimum_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_sum : x + y < 27) :
  ∃ m, m = 1 ∧ (∀ (x y : ℝ), 0 < x → 0 < y → x + y < 27 → m ≤ (sqrt x + sqrt y) / sqrt (x * y) + 1 / sqrt (27 - x - y)) :=
sorry

end minimum_value_expression_l363_363464


namespace second_divisor_is_340_l363_363307

theorem second_divisor_is_340 
  (n : ℕ)
  (h1 : n = 349)
  (h2 : n % 13 = 11)
  (h3 : n % D = 9) : D = 340 :=
by
  sorry

end second_divisor_is_340_l363_363307


namespace area_of_triangle_AQB_l363_363379

theorem area_of_triangle_AQB :
  let (A B C D : ℝ) := (0, 0, 8, 8)
  let Q (y : ℝ) := (y, y, y)
  Point Q such that (Q A = Q B ∧ Q B = Q C) 
  segment QC ⊥ segment FD
  then area of triangle AQB = 12 :=
begin
  sorry
end

end area_of_triangle_AQB_l363_363379


namespace boat_speed_in_still_water_l363_363808

-- Definitions and conditions
def Vs : ℕ := 5  -- Speed of the stream in km/hr
def distance : ℕ := 135  -- Distance traveled in km
def time : ℕ := 5  -- Time in hours

-- Statement to prove
theorem boat_speed_in_still_water : 
  ((distance = (Vb + Vs) * time) -> Vb = 22) :=
by
  sorry

end boat_speed_in_still_water_l363_363808


namespace no_real_square_root_neg_check_square_roots_l363_363318

def has_real_square_root (x : ℝ) : Prop :=
  ∃ y : ℝ, y^2 = x

theorem no_real_square_root_neg (x : ℝ) (hx : x < 0) : ¬ has_real_square_root x :=
by
  intro h
  cases h with y hy
  have : y^2 >= 0 := pow_two_nonneg y
  linarith

theorem check_square_roots (A B C D : ℝ) (hA : A = -2.5) (hB : B = 0) (hC : C = 2.1) (hD : D = 6) :
  (¬ has_real_square_root A) ∧ has_real_square_root B ∧ has_real_square_root C ∧ has_real_square_root D :=
by
  split
  { rw hA, apply no_real_square_root_neg, norm_num }
  split
  { rw hB, use 0, norm_num }
  split
  { rw hC, use real.sqrt 2.1, exact real.sq_sqrt (by norm_num) }
  { rw hD, use real.sqrt 6, exact real.sq_sqrt (by norm_num) }

end no_real_square_root_neg_check_square_roots_l363_363318


namespace females_who_chose_malt_l363_363331

-- Definitions
def total_cheerleaders : ℕ := 26
def total_males : ℕ := 10
def total_females : ℕ := 16
def males_who_chose_malt : ℕ := 6

-- Main statement
theorem females_who_chose_malt (C M F : ℕ) (hM : M = 2 * C) (h_total : C + M = total_cheerleaders) (h_males_malt : males_who_chose_malt = total_males) : F = 10 :=
sorry

end females_who_chose_malt_l363_363331


namespace proof_problem_l363_363314

noncomputable def cos75_squared : ℝ :=
  cos 75 * cos 75

noncomputable def optionA : Prop :=
  cos75_squared = (2 - Real.sqrt 3) / 4

noncomputable def optionB : Prop :=
  (1 + tan 105) / (1 - tan 105) ≠ Real.sqrt 3 / 3

noncomputable def optionC : Prop :=
  tan 1 + tan 44 + tan 1 * tan 44 = 1

noncomputable def optionD : Prop :=
  sin 70 * ((Real.sqrt 3) / tan 40 - 1) ≠ 2

theorem proof_problem : optionA ∧ optionC ∧ optionB ∧ optionD := by
  sorry

end proof_problem_l363_363314


namespace radius_of_circle_with_tangent_parabolas_l363_363018

theorem radius_of_circle_with_tangent_parabolas (r : ℝ) : 
  (∀ x : ℝ, (x^2 + r = x → ∃ x0 : ℝ, x^2 + r = x0)) → r = 1 / 4 :=
by
  sorry

end radius_of_circle_with_tangent_parabolas_l363_363018


namespace quadrilateral_area_l363_363257

theorem quadrilateral_area (a : ℕ) (h : 
    abs (a * real.sqrt (a + 1) + (a + 1) * real.sqrt (a + 2) + (a + 2) * real.sqrt (a + 3) + (a + 3) * real.sqrt a - 
    (real.sqrt a * (a + 1) + real.sqrt (a + 1) * (a + 2) + real.sqrt (a + 2) * (a + 3) + real.sqrt (a + 3) * a)) = 4) :
    a = 1 :=
by
  sorry

end quadrilateral_area_l363_363257


namespace smallest_perfect_square_divisible_by_2_and_5_l363_363776

theorem smallest_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℕ, n = k ^ 2) ∧ (n % 2 = 0) ∧ (n % 5 = 0) ∧ 
  (∀ m : ℕ, 0 < m ∧ (∃ k : ℕ, m = k ^ 2) ∧ (m % 2 = 0) ∧ (m % 5 = 0) → n ≤ m) :=
sorry

end smallest_perfect_square_divisible_by_2_and_5_l363_363776


namespace ratio_sharks_to_pelicans_l363_363890

-- Define the conditions given in the problem
def original_pelican_count {P : ℕ} (h : (2/3 : ℚ) * P = 20) : Prop :=
  P = 30

-- Define the final ratio we want to prove
def shark_to_pelican_ratio (sharks pelicans : ℕ) : ℚ :=
  sharks / pelicans

theorem ratio_sharks_to_pelicans
  (P : ℕ) (h : (2/3 : ℚ) * P = 20) (number_sharks : ℕ) (number_pelicans : ℕ)
  (H_sharks : number_sharks = 60) (H_pelicans : number_pelicans = P)
  (H_original_pelicans : original_pelican_count h) :
  shark_to_pelican_ratio number_sharks number_pelicans = 2 :=
by
  -- proof skipped
  sorry

end ratio_sharks_to_pelicans_l363_363890


namespace common_sum_faces_l363_363247

theorem common_sum_faces (vertices : Fin 8 → ℕ) (v9 : ℕ)
    (h_distinct : vertices ≠ v9) (h_range : ∀ i, vertices i ∈ Finset.range 9) 
    (h_sum : (Finset.univ.image vertices).sum + v9 = 53) 
    (h_faces : ∀ f : Fin 6, (Finset.univ.image vertices).sum = vertices_sum) : 
    common_sum_faces = 26.5 := 
by
    have h1 : ∑ i in Finset.range 10, if ∃ j : Fin 8, vertices j = i then vertices i else (if v9 = i then 1 else 0) = 53 :=
        sorry
    have h2 : 3 * 53 = 159 :=
        sorry
    have h3 : 159 / 6 = 26.5 :=
        sorry
    sorry

end common_sum_faces_l363_363247


namespace fraction_product_eq_l363_363847

theorem fraction_product_eq :
  (1 / 3) * (3 / 5) * (5 / 7) * (7 / 9) = 1 / 9 := by
  sorry

end fraction_product_eq_l363_363847


namespace inequality_x_y_l363_363918

theorem inequality_x_y (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2 * y^2 ≤ 3 := 
sorry

end inequality_x_y_l363_363918


namespace average_of_numbers_divisible_by_five_between_6_and_34_l363_363791

-- Define the set of numbers between 6 and 34 that are divisible by 5
def divisible_by_five_between_6_and_34 : Finset ℕ :=
  Finset.filter (λ x, x % 5 = 0) (Finset.Icc 6 34)

-- Define the sum of elements in the set
def sum_divisible_by_five : ℕ :=
  divisible_by_five_between_6_and_34.sum id

-- Define the count of elements in the set
def count_divisible_by_five : ℕ :=
  divisible_by_five_between_6_and_34.card

-- Define the average of the elements in the set
def average_divisible_by_five : ℕ :=
  sum_divisible_by_five / count_divisible_by_five

-- The theorem to prove that the average is 20
theorem average_of_numbers_divisible_by_five_between_6_and_34 : average_divisible_by_five = 20 := by
  sorry

end average_of_numbers_divisible_by_five_between_6_and_34_l363_363791


namespace sum_of_repeating_decimals_l363_363429

noncomputable def x : ℚ := 1 / 9
noncomputable def y : ℚ := 2 / 99
noncomputable def z : ℚ := 3 / 999

theorem sum_of_repeating_decimals :
  x + y + z = 134 / 999 := by
  sorry

end sum_of_repeating_decimals_l363_363429


namespace q_multiplier_l363_363112

variables (w d z : ℝ) (q : ℝ)
def q_def (w d z : ℝ) : ℝ := 5 * w / (4 * d * (z^2))

theorem q_multiplier
  (hw : 4 * w)
  (hd : 2 * d)
  (hz : 3 * z)
  : q_def (4 * w) (2 * d) (3 * z) = (5 / 18) * (q_def w d z) :=
by
  sorry

end q_multiplier_l363_363112


namespace select_16_real_coins_l363_363265

theorem select_16_real_coins (coins : Fin 40 → ℝ) (is_real : Fin 40 → Bool) :
  (∃ fakes : Finset (Fin 40), fakes.card = 3 ∧ ∀ c ∈ fakes, ¬is_real c) →
  (∀ a b, is_real a → is_real b → coins a = coins b) →
  (∃ reals : Finset (Fin 40), reals.card = 16 ∧ ∀ r ∈ reals, is_real r) :=
begin
  -- proof would go here
  sorry
end

end select_16_real_coins_l363_363265


namespace farthest_vertex_of_dilated_square_l363_363214

def center : Point := (-3, 4)
def side_len : ℝ := sqrt 16
def dilation_center : Point := (0, 0)
def dilation_factor : ℝ := 3

def vertices : Set Point := {(-5, 6), (-1, 6), (-1, 2), (-5, 2)}

def dilated_vertices : Set Point := vertices.map (λ (p : Point), (p.1 * dilation_factor, p.2 * dilation_factor))

def farthest_vertex_from_origin (points : Set Point) : Point :=
  points.maxBy (λ (p : Point), p.1 ^ 2 + p.2 ^ 2) -- This square of the distance to avoid the sqrt in comparison.

theorem farthest_vertex_of_dilated_square :
  farthest_vertex_from_origin dilated_vertices = (-15, 18) :=
sorry

end farthest_vertex_of_dilated_square_l363_363214


namespace constant_difference_of_equal_derivatives_l363_363535

theorem constant_difference_of_equal_derivatives
  {f g : ℝ → ℝ}
  (h : ∀ x, deriv f x = deriv g x) :
  ∃ C : ℝ, ∀ x, f x - g x = C := 
sorry

end constant_difference_of_equal_derivatives_l363_363535


namespace exist_equal_distant_colored_points_l363_363412

-- Definitions for vertices and their colorings.
variable {V : Type} [DecidableEq V] [Fintype V] (vertices : Fin 100 → V)

-- Definitions for red and blue color sets.
variable {red blue : Finset V}
variable (hr : red.card = 10) (hb : blue.card = 10)
variable (disjoint_colors : Disjoint red blue)

-- Definition for vertices A, B, C, D.
variable {A B C D : V}

-- Main theorem statement.
theorem exist_equal_distant_colored_points
  (hA : A ∈ blue) (hB : B ∈ blue) 
  (hC : C ∈ red) (hD : D ∈ red) : 
  ∃ A B C D, A ∈ blue ∧ B ∈ blue ∧ C ∈ red ∧ D ∈ red ∧ dist A B = dist C D := by
  sorry

end exist_equal_distant_colored_points_l363_363412


namespace factor_of_change_l363_363221

-- Given conditions
def avg_marks_before : ℕ := 45
def avg_marks_after : ℕ := 90
def num_students : ℕ := 30

-- Prove the factor F by which marks are changed
theorem factor_of_change : ∃ F : ℕ, avg_marks_before * F = avg_marks_after := 
by
  use 2
  have h1 : 30 * avg_marks_before = 30 * 45 := rfl
  have h2 : 30 * avg_marks_after = 30 * 90 := rfl
  sorry

end factor_of_change_l363_363221


namespace area_of_abc_l363_363124

theorem area_of_abc
    (A B C : Type) [nonempty A] [nonempty B] [nonempty C]
    (h₁ : ∀ (a b c : A), is_acute_triangle a b c)
    (h₂ : ∀ (a d b c : A), is_altitude a d b c)
    (h₃ : ∀ (a b c : A), (dist a b) + (dist a c) * 4 * sqrt (2 / 17) = sqrt 17)
    (h₄ : ∀ (a d b c : A), 4 * dist b c + 5 * dist a d = 17) :
    area_triangle A B C = 289 / 250 := sorry

end area_of_abc_l363_363124


namespace find_b_values_l363_363554

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l363_363554


namespace trigonometric_expression_value_l363_363662

variables (x y : ℝ)
hypothesis (h1 : sin x / sin y = 2)
hypothesis (h2 : cos x / cos y = 1 / 3)

theorem trigonometric_expression_value :
  (sin (x / 2) / sin (y / 2)) + (cos (x / 2) / cos (y / 2)) = (sin (x / 2) / sin (y / 2)) + (cos (x / 2) / cos (y / 2)) :=
  by sorry

end trigonometric_expression_value_l363_363662


namespace find_S9_l363_363003

noncomputable def S (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => a_1
  | n + 1 => S n + a_1 * r ^ (n + 1)

theorem find_S9 (S_3 S_6 : ℚ) (h1 : S 3 = 8) (h2 : S 6 = 10) : S 9 = 21 / 2 := by
  sorry

end find_S9_l363_363003


namespace relationship_among_abc_l363_363488

theorem relationship_among_abc (x : ℝ) (e : ℝ) (ln : ℝ → ℝ) (half_pow : ℝ → ℝ) (exp : ℝ → ℝ) 
  (x_in_e_e2 : x > e ∧ x < exp 2) 
  (def_a : ln x = ln x)
  (def_b : half_pow (ln x) = ((1/2)^(ln x)))
  (def_c : exp (ln x) = x):
  (exp (ln x)) > (ln x) ∧ (ln x) > ((1/2)^(ln x)) :=
by 
  sorry

end relationship_among_abc_l363_363488


namespace sum_of_solutions_l363_363442

theorem sum_of_solutions :
  let k := 36
  let eq := λ x, x^2 - 10 * x + k = 0
  (∃ x₁ x₂ : ℝ, eq x₁ ∧ eq x₂ ∧ x₁ + x₂ = 10) :=
by 
  let k := 36
  let eq := λ x, x^2 - 10 * x + k = 0
  have h1: eq 6 := by sorry
  have h2: eq 4 := by sorry
  use [6, 4]
  split
  . exact h1
  . split
  . exact h2
  . exact rfl

end sum_of_solutions_l363_363442


namespace number_board_total_18_l363_363269

-- Define the conditions as Lean hypotheses
theorem number_board_total_18 (k : ℕ) (h1 : (k > 0)) (nums : list ℝ) 
  (h_sum_total : nums.sum = 1)
  (h_sum_five_smallest : ((nums.take 5).sum = 0.26))
  (h_sum_five_largest : ((nums.drop (k + 5)).sum = 0.29)) :
  (k + 10) = 18 :=
sorry

end number_board_total_18_l363_363269


namespace Jamie_drink_limit_l363_363649

theorem Jamie_drink_limit :
  let milk := 250 -- milliliters
  let orange_juice := 5 * 29.57 -- 5 ounces converted to milliliters
  let grape_juice := 10 * 29.57 -- 10 ounces converted to milliliters
  let soda := 0.1 * 1000 -- 0.1 liters converted to milliliters
  let total_consumed := milk + orange_juice + grape_juice + soda
  let limit := 1.2 * 1000 -- 1.2 liters converted to milliliters
  let remaining_ml := limit - total_consumed
  let remaining_oz := remaining_ml / 29.57 -- Convert remaining milliliters to ounces
  remaining_oz ≈ 13.74 :=
by {
  sorry
}

end Jamie_drink_limit_l363_363649


namespace triangle_side_lengths_l363_363510

theorem triangle_side_lengths (a : ℝ) :
  (∃ (b c : ℝ), b = 1 - 2 * a ∧ c = 8 ∧ (3 + b > c ∧ 3 + c > b ∧ b + c > 3)) ↔ (-5 < a ∧ a < -2) :=
sorry

end triangle_side_lengths_l363_363510


namespace work_completion_time_l363_363697

variable (K : ℝ) -- Efficiency of Krish
variable (T : ℝ) [hK_pos : Fact (0 < K)] -- Total work defined using the efficiency of Krish
variable (time : ℝ) -- Time taken to complete the task by Ram, Krish, and Vikas together

-- Condition: Ram is half as efficient as Krish and takes 21 days to complete the task alone
def efficiency_ram := K / 2
def ram_time := 21

-- Condition: Vikas is two-thirds as efficient as Krish
def efficiency_vikas := (2/3) * K

-- Define the combined work rate of Ram, Krish, and Vikas
def combined_work_rate := efficiency_ram K + K + efficiency_vikas K

-- Express the total work in terms of K and verify the time taken is approximately 4.846 days
theorem work_completion_time : 
  T = (efficiency_ram K) * ram_time → 
  time ~ (T / combined_work_rate K) → time ≈ 63 / 13 :=
by 
  intros hT htime
  have h1 : T = (K / 2) * 21 := hT
  have h2 : combined_work_rate K = (K / 2) + K + (2 / 3) * K := rfl
  rw [h1, h2] at htime
  linarith

end work_completion_time_l363_363697


namespace solve_for_a_l363_363915

noncomputable def z1 (a : ℂ) : ℂ := (3 : ℂ) / (a + 5) + (10 - a^2)*complex.I
noncomputable def z2 (a : ℂ) : ℂ := (2 : ℂ) / (1 - a) + (2*a - 5)*complex.I

theorem solve_for_a (a : ℂ) : z1 a + z2 a ∈ ℝ ↔ a = 3 := 
sorry

end solve_for_a_l363_363915


namespace sum_of_incircles_radii_gt_r_l363_363825

-- Definitions based on the conditions
variable {n : ℕ} -- Number of triangles
variable {r : ℝ} -- Radius of the circle circumscribed around the polygon
variable {radii : Fin n → ℝ} -- Radii of the incircles of the triangles
variable {perimeters : Fin n → ℝ} -- Perimeters of the triangles
variable {areas : Fin n → ℝ} -- Areas of the triangles
variable {P : ℝ} -- Perimeter of the original polygon
variable {S : ℝ} -- Area of the original polygon

-- Conditions stated
axiom circumscribed_polygon (h : Polygon circumscribed around circle of radius r)
axiom division_into_triangles (h : Polygon divided into n triangles)

-- Equivalent proof problem statement in Lean 4
theorem sum_of_incircles_radii_gt_r
  (hradii : ∀ i, radii i = 2 * areas i / perimeters i)
  (hperim_gt : ∀ i, perimeters i < P)
  (harea_sum : (Finset.univ.sum areas) = S)
  (hS_div_P : 2 * S / P = r) :
  (Finset.univ.sum radii) > r := 
sorry

end sum_of_incircles_radii_gt_r_l363_363825


namespace increased_speed_l363_363831

theorem increased_speed
  (d : ℝ) (s1 s2 : ℝ) (t1 t2 : ℝ) 
  (h1 : d = 2) 
  (h2 : s1 = 2) 
  (h3 : t1 = 1)
  (h4 : t2 = 2 / 3)
  (h5 : s1 * t1 = d)
  (h6 : s2 * t2 = d) :
  s2 - s1 = 1 := 
sorry

end increased_speed_l363_363831


namespace midpoint_range_l363_363490

variable {x0 y0 : ℝ}

-- Conditions
def point_on_line1 (P : ℝ × ℝ) := P.1 + 2 * P.2 - 1 = 0
def point_on_line2 (Q : ℝ × ℝ) := Q.1 + 2 * Q.2 + 3 = 0
def is_midpoint (P Q M : ℝ × ℝ) := P.1 + Q.1 = 2 * M.1 ∧ P.2 + Q.2 = 2 * M.2
def midpoint_condition (M : ℝ × ℝ) := M.2 > M.1 + 2

-- Theorem
theorem midpoint_range
  (P Q M : ℝ × ℝ)
  (hP : point_on_line1 P)
  (hQ : point_on_line2 Q)
  (hM : is_midpoint P Q M)
  (h_cond : midpoint_condition M)
  (hx0 : x0 = M.1)
  (hy0 : y0 = M.2)
  : - (1 / 2) < y0 / x0 ∧ y0 / x0 < - (1 / 5) :=
sorry

end midpoint_range_l363_363490


namespace total_cloth_sold_l363_363838

variable (commissionA commissionB salesA salesB totalWorth : ℝ)

def agentA_commission := 0.025 * salesA
def agentB_commission := 0.03 * salesB
def total_worth_of_cloth_sold := salesA + salesB

theorem total_cloth_sold 
  (hA : agentA_commission = 21) 
  (hB : agentB_commission = 27)
  : total_worth_of_cloth_sold = 1740 :=
by
  sorry

end total_cloth_sold_l363_363838


namespace smallest_n_for_P_lt_l363_363874

noncomputable def P (n : ℕ) : ℝ :=
  1 / (n * (n + 1))

theorem smallest_n_for_P_lt (n : ℕ) : 1 ≤ n ∧ n < 2010 → P(n) < 1 / 2010 ↔ n = 45 :=
by
  intros h1 h2
  sorry

end smallest_n_for_P_lt_l363_363874


namespace find_parabola_l363_363908

-- Define the conditions in the Lean environment
variables (S : Type) [AffineSpace ℝ S] [AffineMap ℝ S ℝ]
variables (O F : S) -- vertex O and focus F
variables (A B C D : S) -- points A, B, C, D creating the quadrilateral
variables (area : ℝ) -- the given area of quadrilateral ABCD

-- Define the mathematical structure and properties
def parabola_vertex_focus (S : Type) [AffineSpace ℝ S] [AffineMap ℝ S ℝ] :=
  ∃ (p : ℝ) (x y : ℝ) (A B : S), 
    p > 0 ∧
    (Aᵀy - y = p * x ∧ Bᵀy - y = - p * x) ∧
    exists x_A x_B y_A y_B, 
      A = mkAffine x_A y_A ∧ B = mkAffine x_B y_B ∧ 
      (x_A, y_A ∈ S) ∧ (x_B, y_B ∈ S)

-- Define the quadrilateral with area constraint
def quadrilateral_area_minimum (S : Type) [AffineSpace ℝ S] [AffineMap ℝ S ℝ] 
  (O F A B C D : S) (area : ℝ) : Prop :=
  ∃ x_A y_A x_B y_B,
    A = mkAffine x_A y_A ∧ B = mkAffine x_B y_B ∧ 
    area * 2 = 8 ∧
    x_A + p = C_x ∧
    x_B + p = D_x ∧
    area_triangle O D F + area_triangle O C F = area

-- Prove the equation of the parabola
theorem find_parabola {S : Type} [AffineSpace ℝ S] [AffineMap ℝ S ℝ]
  (O F A B C D : S) (area : ℝ) 
  (h1 : parabola_vertex_focus S)
  (h2 : quadrilateral_area_minimum O F A B C D area) :
  ∃ p : ℝ, y^2 = 2 * p * x ∧ p > 0 ∧ area = 8 ∧ 
    (y = √4x ∨ y = -√4x) :=
sorry

end find_parabola_l363_363908


namespace problem_statement_l363_363768

open List

noncomputable def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (≤)
  sorted.get! (sorted.length / 2)

noncomputable def mode (l : List ℕ) : ℕ :=
  l.groupBy id (λ x y, x = y) 
   |>.map Prod.fst
   |>.maxBy (λ n, l.count n)

theorem problem_statement : median [7, 5, 6, 8, 9, 9, 10] = 8 ∧ mode [7, 5, 6, 8, 9, 9, 10] = 9 :=
by
  sorry

end problem_statement_l363_363768


namespace sequence_satisfies_general_formula_l363_363058

noncomputable def sequence (n : ℕ) : ℝ := 
if n = 0 then 1 else real.sqrt n

theorem sequence_satisfies (n : ℕ) (h : n > 0) :
  let a_n := sequence n in
  let a_n_plus_1 := sequence (n + 1) in
  (a_n_plus_1 ^ 2 = a_n ^ 2 + 1) :=
by
  sorry

theorem general_formula (a : ℕ → ℝ) (h1 : a 1 = 1)
(h2 : ∀ n, a (n + 1) ^ 2 = a n ^ 2 + 1) :
  ∀ n, a n = real.sqrt n :=
by
  sorry

end sequence_satisfies_general_formula_l363_363058


namespace min_value_fraction_l363_363606

noncomputable def circle_eq : ℝ → ℝ → ℝ :=
  λ x y, x^2 + y^2 + 2 * x - 4 * y + 1

noncomputable def line_eq (a b : ℝ) : ℝ → ℝ → ℝ :=
  λ x y, 2 * a * x - b * y + 2

theorem min_value_fraction (a b : ℝ)
  (ha : a > 0) (hb : b > 0)
  (H : ∀ (x y : ℝ), circle_eq x y = 0 → line_eq a b x y = 0 ↔ line_eq a b (x - 1) (y - 2) = 0) :
  (1 / a) + (4 / b) = 9 :=
by
  sorry

end min_value_fraction_l363_363606


namespace B_works_alone_in_12_days_l363_363357

theorem B_works_alone_in_12_days (A_days : ℕ) (AB_days : ℕ) (B_days : ℕ) 
    (hA : A_days = 6) (hAB : AB_days = 4) : B_days = 12 := by
  -- Define the work rates
  let A_work_rate := 1 / A_days
  let B_work_rate := 1 / B_days
  let AB_work_rate := 1 / AB_days
  -- Setup the condition for the combined work rate
  have h_combined : A_work_rate + B_work_rate = AB_work_rate, by sorry
  -- Use the given conditions
  rw [hA, hAB] at h_combined
  -- Solve for B_days
  sorry

end B_works_alone_in_12_days_l363_363357


namespace reciprocals_form_arithmetic_sequence_l363_363910

-- Define the conditions given in the problem
structure Quadrilateral (A B C D : Type) :=
(K L G F : Type) -- Points of intersection

axiom quadrilateral_intersections 
  {A B C D K L G F : Type}
  [quadrilateral : Quadrilateral A B C D]
  (extensions_intersect_opposite_sides : 
     ∀ (P Q : Type), P ≠ Q → ∃!_unique (ext : Type), ext = K ∧ ext = L)
  (diagonals_intersect_KL :
     ∃!_unique (P Q : Type), P = G ∧ Q = F ∧ (P ∈ K ∧ Q ∈ L)) : Prop

-- Statement declaring the proof goal
theorem reciprocals_form_arithmetic_sequence
  {A B C D K L G F : Type}
  [quadrilateral : Quadrilateral A B C D] :
  quadrilateral_intersections {A B C D} {K L G F} → 
  ∃ (KF KL KG : ℝ), 1 / KF, 1 / KL, 1 / KG ∈ ℝ ∧ 
                     (2 / KL = 1 / KF + 1 / KG) :=
by
  sorry

end reciprocals_form_arithmetic_sequence_l363_363910


namespace cube_root_neg_27_l363_363733

theorem cube_root_neg_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 :=
by {
    use -3,
    split,
    { norm_num },
    { refl }
}

end cube_root_neg_27_l363_363733


namespace surface_area_reduction_of_spliced_cuboid_l363_363723

theorem surface_area_reduction_of_spliced_cuboid 
  (initial_faces : ℕ := 12)
  (faces_lost : ℕ := 2)
  (percentage_reduction : ℝ := (2 / 12) * 100) :
  percentage_reduction = 16.7 :=
by
  sorry

end surface_area_reduction_of_spliced_cuboid_l363_363723


namespace total_toothpicks_2520_l363_363678

theorem total_toothpicks_2520 :
  ∀ (length width : ℕ), length = 30 → width = 40 →
  let vertical_toothpicks := (length + 1) * width in
  let horizontal_toothpicks := (width + 1) * length in
  let diagonal_toothpicks := Nat.sqrt (length ^ 2 + width ^ 2) in
  vertical_toothpicks + horizontal_toothpicks + diagonal_toothpicks = 2520 :=
by
  intros length width h_length h_width
  let vertical_toothpicks := (length + 1) * width
  let horizontal_toothpicks := (width + 1) * length
  let diagonal_toothpicks := Nat.sqrt (length ^ 2 + width ^ 2)
  have h1 : vertical_toothpicks = 1240, from by rw [h_length, h_width]; exact rfl
  have h2 : horizontal_toothpicks = 1230, from by rw [h_length, h_width]; exact rfl
  have h3 : diagonal_toothpicks = 50, from by rw [h_length, h_width]; exact rfl
  rw [h1, h2, h3]
  exact rfl

end total_toothpicks_2520_l363_363678


namespace initial_liquid_A_amount_l363_363326

noncomputable def initial_amount_of_A (x : ℚ) : ℚ :=
  3 * x

theorem initial_liquid_A_amount {x : ℚ} (h : (3 * x - 3) / (2 * x + 3) = 3 / 5) : initial_amount_of_A (8 / 3) = 8 := by
  sorry

end initial_liquid_A_amount_l363_363326


namespace union_set_eq_l363_363944

open Set

def P := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
def Q := {x : ℝ | x^2 ≤ 4}

theorem union_set_eq : P ∪ Q = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by
  sorry

end union_set_eq_l363_363944


namespace ice_cream_eaten_l363_363388

variables (f : ℝ)

theorem ice_cream_eaten (h : f + 0.25 = 3.5) : f = 3.25 :=
sorry

end ice_cream_eaten_l363_363388


namespace qrs_company_profit_increase_l363_363334

theorem qrs_company_profit_increase (P : ℝ) : 
  let profitApril := 1.5 * P,
      profitMay := 1.5 * P - 0.2 * (1.5 * P),
      profitJune := (1.5 * P - 0.3 * P) * 1.5
  in profitJune = 1.8 * P ∧ 
     ((profitJune - P) / P) * 100 = 80 :=
by
  let profitApril := 1.5 * P
  let profitMay := 1.5 * P - 0.2 * (1.5 * P)
  let profitJune := (1.5 * P - 0.3 * P) * 1.5
  have hJune_profit : profitJune = 1.8 * P
  have hPercent_increase : ((profitJune - P) / P) * 100 = 80
  exact ⟨hJune_profit, hPercent_increase⟩
  sorry

end qrs_company_profit_increase_l363_363334


namespace transport_sand_gravel_l363_363263

theorem transport_sand_gravel (x y : ℕ) (hx : 3 * 5 + 5 * 1 = 20) (hy : 5 * 4 = 20) :
  ∃ (x y : ℕ), 3 * x + 5 * y = 20 :=
by {
  use (5, 1),
  use (0, 4),
  assumption,
  sorry
}

end transport_sand_gravel_l363_363263


namespace original_selling_price_l363_363405

-- Definitions and conditions
def cost_price (CP : ℝ) := CP
def profit (CP : ℝ) := 1.25 * CP
def loss (CP : ℝ) := 0.75 * CP
def loss_price (CP : ℝ) := 600

-- Main theorem statement
theorem original_selling_price (CP : ℝ) (h1 : loss CP = loss_price CP) : profit CP = 1000 :=
by
  -- Note: adding the proof that CP = 800 and then profit CP = 1000 would be here.
  sorry

end original_selling_price_l363_363405


namespace exists_four_distinct_indices_l363_363174

theorem exists_four_distinct_indices
  (a : Fin 5 → ℝ)
  (h : ∀ i, 0 < a i) :
  ∃ i j k l : (Fin 5), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
  |a i / a j - a k / a l| < 1 / 2 :=
by
  sorry

end exists_four_distinct_indices_l363_363174


namespace sin_alpha_plus_2_beta_l363_363044

theorem sin_alpha_plus_2_beta (α β : ℝ) 
  (hα : 0 < α ∧ α < π)
  (hβ : -π/2 < β ∧ β < π/2)
  (h1 : Real.sin (α + π/3) = 1/3)
  (h2 : Real.cos (β - π/6) = sqrt 6 / 6) :
  Real.sin (α + 2 * β) = (2 * sqrt 10 - 2) / 9 := 
by {
  sorry
}

end sin_alpha_plus_2_beta_l363_363044


namespace smallest_elements_union_l363_363702

open Set

variable {α : Type*} {A B : Set α}

theorem smallest_elements_union (hA : A.card = 30) (hB : B.card = 25) : 
  (A ∪ B).card ≥ 30 :=
by
  sorry

end smallest_elements_union_l363_363702


namespace profit_calculation_l363_363679

def total_investment := 800 + 200 + 600 + 400

def mary_ratio := 800 / total_investment
def mike_ratio := 200 / total_investment
def anna_ratio := 600 / total_investment
def ben_ratio := 400 / total_investment

def equal_share (P: ℝ) := P / 12
def investment_share (P: ℝ) := P / 3
def effort_share (P: ℝ) : ℝ*ratios → ℝ := sorry  -- details omitted for simplicity

theorem profit_calculation (P : ℝ) (mary mike anna ben: ℝ) 
  (mary_ratio mike_ratio anna_ratio ben_ratio: ℝ) :
  let mary_total := equal_share P + (mary_ratio * investment_share P) + (2 * investment_share P / 10)
  let mike_total := equal_share P + (mike_ratio * investment_share P) + (1 * investment_share P / 10)
  let anna_total := equal_share P + (anna_ratio * investment_share P) + (3 * investment_share P / 10)
  let ben_total := equal_share P + (ben_ratio * investment_share P) + (4 * investment_share P / 10)
  mary_total - mike_total = 900 → 
  anna_total - ben_total = 600 → 
  P = 6000 := 
by
  sorry

end profit_calculation_l363_363679


namespace Jenny_ate_65_l363_363651

theorem Jenny_ate_65 (mike_squares : ℕ) (jenny_squares : ℕ)
  (h1 : mike_squares = 20)
  (h2 : jenny_squares = 3 * mike_squares + 5) :
  jenny_squares = 65 :=
by
  sorry

end Jenny_ate_65_l363_363651


namespace jerry_firecrackers_l363_363151

def firecrackers_set_off (initial : ℕ) (confiscated : ℕ) (defective_fraction : ℚ) (set_off_fraction : ℚ) : ℕ :=
  let remaining := initial - confiscated
  let defective := (remaining * defective_fraction).natAbs -- Ensures we get an integer number of defective firecrackers
  let good := remaining - defective
  (good * set_off_fraction).natAbs -- Ensures we get an integer number of set-off firecrackers

theorem jerry_firecrackers :
  firecrackers_set_off 48 12 (1/6) (1/2) = 15 :=
by simp [firecrackers_set_off]; norm_num; done; sorry

end jerry_firecrackers_l363_363151


namespace largest_shaded_area_l363_363415

theorem largest_shaded_area :
  let S := 9 -- area of each square
  let AX := S - 2.25 * Real.pi -- shaded area of Figure X
  let AY := S - Real.pi -- shaded area of Figure Y
  let AZ := 4.5 * Real.pi - S in -- shaded area of Figure Z
  AY > AX ∧ AY > AZ := 
  sorry

end largest_shaded_area_l363_363415


namespace find_b_values_l363_363548

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l363_363548


namespace pine_trees_cover_equator_l363_363203

noncomputable def eq_east_west_cover (trees : ℕ → ℝ) (h_tall : ∀ n, trees n ≤ 100) : Prop :=
  (∀ k, ∃ n, k ∈ segment (trees n) (trees n + 100)) → (∀ k, ∃ n, k ∈ segment (trees n - 100) (trees n))

theorem pine_trees_cover_equator (trees : ℕ → ℝ) (h_tall : ∀ n, trees n ≤ 100)
  (h_cover_east : ∀ k, ∃ n, k ∈ segment (trees n) (trees n + 100)) :
  ∀ k, ∃ n, k ∈ segment (trees n - 100) (trees n) :=
begin
  sorry
end

end pine_trees_cover_equator_l363_363203


namespace jerry_firecrackers_l363_363152

def firecrackers_set_off (initial : ℕ) (confiscated : ℕ) (defective_fraction : ℚ) (set_off_fraction : ℚ) : ℕ :=
  let remaining := initial - confiscated
  let defective := (remaining * defective_fraction).natAbs -- Ensures we get an integer number of defective firecrackers
  let good := remaining - defective
  (good * set_off_fraction).natAbs -- Ensures we get an integer number of set-off firecrackers

theorem jerry_firecrackers :
  firecrackers_set_off 48 12 (1/6) (1/2) = 15 :=
by simp [firecrackers_set_off]; norm_num; done; sorry

end jerry_firecrackers_l363_363152


namespace part_i_part_ii_part_iii_l363_363827

-- Definitions of the given values
variables (A B C P Q I : Type*)
variables (a b c p q : ℝ)
variables (PA PB PQ QA QC : ℝ)
variables (IP IB IC : vector_space ℝ)

-- Assume the conditions given in the problem
variable h1 : PA * p = PB 
variable h2 : QA * q = QC 
variable h3 : PQ * (1 + p) = PB 
variable h4 : PQ * (1 + q) = QC

-- Part (i)
theorem part_i (h1 : PA * p = PB) (h2 : QA * q = QC) :
  a * (1 + p) * IP = (a - p * b) * IB - p * c * IC :=
sorry

-- Part (ii)
theorem part_ii (h1 : PA * p = PB) (h2 : QA * q = QC) :
  a = b * p + c * q :=
sorry

-- Part (iii)
theorem part_iii (h1 : PA * p = PB) (h2 : QA * q = QC) (h3 : a^2 = 4 * b * c * p * q) :
  ∃ (R : Type*), concurrent AI BQ CP R :=
sorry

end part_i_part_ii_part_iii_l363_363827


namespace range_of_a_l363_363936

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → (x + a * sin x ≤ y + a * sin y)) → a ∈ set.Icc (-1 : ℝ) (1 : ℝ) :=
by 
  sorry

end range_of_a_l363_363936


namespace repeating_decimals_sum_as_fraction_l363_363025

theorem repeating_decimals_sum_as_fraction :
  (0.3333...).to_rat + (0.020202...).to_rat + (0.00030003...).to_rat = 3538 / 9999 := by
sorry

end repeating_decimals_sum_as_fraction_l363_363025


namespace Hyperbola_Proof_l363_363466

variables (t : ℝ)
-- t is positive
variable (ht : t > 0)

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = t

-- Define the right focus F
def F := (sqrt(2 * t), (0 : ℝ))

-- Define line through F intersecting right branch of the hyperbola at M and N
-- Midpoint of M and N is used in calculating perpendicular bisector intersecting x-axis at P
def FP_ratio_is_half_sqrt2 (t : ℝ) [ht : t > 0] := 
  -- The statement to prove
  |((sqrt (2 * t)) * (sqrt (2 * t)) + 1)/ ((2 * sqrt (t) * (sqrt (2 * t)) + 1) / (sqrt (2 * t)))| = |sqrt (2) / 2| 

theorem Hyperbola_Proof:
  ∀ (t : ℝ), t > 0 → FP_ratio_is_half_sqrt2 t :=
begin
  intros t ht,
  sorry
end

end Hyperbola_Proof_l363_363466


namespace square_product_hypotenuses_eq_l363_363764

noncomputable def square_product_of_hypotenuses (x y : ℝ) (h₁ : x * y / 2 = 2) (h₂ : x * y / 2 = 3)
  (similar : x / y = sqrt 3) : ℝ :=
  let z := sqrt (x^2 + y^2) in
  let w := sqrt (x^2 / 3 + y^2 / 3) in
  (z * w) ^ 2

theorem square_product_hypotenuses_eq : 
  square_product_of_hypotenuses x y h₁ h₂ (by field_simp; norm_num; exact sorry) = 9216 / 25 := 
  sorry

end square_product_hypotenuses_eq_l363_363764


namespace find_original_number_l363_363822

def original_number (x : ℝ) : Prop :=
  let step1 := 1.20 * x
  let step2 := step1 * 0.85
  let final_value := step2 * 1.30
  final_value = 1080

theorem find_original_number : ∃ x : ℝ, original_number x :=
by
  use 1080 / (1.20 * 0.85 * 1.30)
  sorry

end find_original_number_l363_363822


namespace cube_of_expression_l363_363406

theorem cube_of_expression :
  (√(2 + √(2 + √(2)))) ^ 3 = 8 := by
  sorry

end cube_of_expression_l363_363406


namespace infinite_power_tower_eq_4_l363_363889

theorem infinite_power_tower_eq_4 (x : ℝ) (h_converges : ∃ y : ℝ, x^(x^(x^(⋯))) = y) : x = real.sqrt 2 ↔ x^(x^(x^(⋯))) = 4 :=
by
  sorry

end infinite_power_tower_eq_4_l363_363889


namespace number_of_friends_is_three_l363_363689

-- Define the types for Friend and Foe
inductive Citizen
| Friend
| Foe

open Citizen

def tells_truth : Citizen → Prop
| Friend => true
| Foe => false

def lies : Citizen → Prop
| Friend => false
| Foe => true

-- Define the circle of citizens
variables (A B C D E F G : Citizen)

-- Define the statements that each citizen makes
def statement (c1 c2 c3 : Citizen) : Prop :=
  tells_truth c1 ∧ lies c2 ∧ lies c3

-- Define the condition:
-- Each of the seven citizens states, "I am sitting between two Foes"
def conditions : Prop :=
  statement A G B ∧ statement B A C ∧ statement C B D ∧
  statement D C E ∧ statement E D F ∧ statement F E G ∧
  statement G F A

-- Prove that the number of Friends is exactly 3
theorem number_of_friends_is_three : 
  ∀ (A B C D E F G : Citizen), 
  conditions A B C D E F G → 
  (tells_truth A).count + (tells_truth B).count + 
  (tells_truth C).count + (tells_truth D).count + 
  (tells_truth E).count + (tells_truth F).count + 
  (tells_truth G).count = 3 :=
by
  intros
  sorry

end number_of_friends_is_three_l363_363689


namespace supplement_complement_diff_l363_363225

theorem supplement_complement_diff (α : ℝ) : (180 - α) - (90 - α) = 90 := 
by
  sorry

end supplement_complement_diff_l363_363225


namespace find_digit_l363_363231

theorem find_digit (A : ℕ) (hA : A ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  9 ∣ (3 * 1000 + A * 110 + 1) ↔ A = 7 :=
by
  sorry

end find_digit_l363_363231


namespace digit_456_fraction_17_59_l363_363301

noncomputable def decimal_expansion_17_59 : String := "28813559322033898305084745762711"

theorem digit_456_fraction_17_59 :
  (decimal_expansion_17_59.get? 20).isSome ∧ (decimal_expansion_17_59.get! 20) = '8' :=
by
  sorry

end digit_456_fraction_17_59_l363_363301


namespace average_speed_of_car_l363_363810

-- Define the conditions
def first_half_speed : ℝ := 75
def second_half_speed : ℝ := 50
def total_distance (D : ℝ) : Prop := D > 0

-- Define the theorem stating the result
theorem average_speed_of_car (D : ℝ) (hD : total_distance D) : 
  (2 * D / (D / first_half_speed + D / second_half_speed)) = 60 :=
by
  -- Using the conditions directly
  have h1 : D / 2 / first_half_speed = D / (2 * first_half_speed), from sorry, -- Need to equate times properly
  have h2 : D / 2 / second_half_speed = D / (2 * second_half_speed), from sorry, 
  have total_time := D / (2 * first_half_speed) + D / (2 * second_half_speed), from sorry,
  -- Using the transformed definition to state the average_speed
  calc (2 * D / (2 * first_half_speed + 2 * second_half_speed))
       = 60 : sorry

-- Placeholder 'sorry' added to skip the proofs for have and calc steps.

end average_speed_of_car_l363_363810


namespace quadratic_has_real_root_iff_b_in_interval_l363_363540

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l363_363540


namespace product_of_prs_l363_363963

theorem product_of_prs 
  (p r s : Nat) 
  (h1 : 3^p + 3^5 = 270) 
  (h2 : 2^r + 58 = 122) 
  (h3 : 7^2 + 5^s = 2504) : 
  p * r * s = 54 := 
sorry

end product_of_prs_l363_363963


namespace no_polynomial_exists_l363_363015

-- Definitions for the polynomial and its conditions
noncomputable def polynomial (x y : ℝ) : ℝ := A * x^2 + B * x * y + C * y^2 + D * x + E * y + F

-- Statement of non-existence of such a polynomial
theorem no_polynomial_exists (A B C D E F : ℝ) : ¬(∃ (P : ℝ → ℝ → ℝ),
  (∀ (x y : ℝ), P x y = A * x^2 + B * x * y + C * y^2 + D * x + E * y + F) ∧
  (P 1 1 ≠ P 1 2) ∧ (P 1 1 ≠ P 1 3) ∧ (P 1 1 ≠ P 2 1) ∧
  (P 1 1 ≠ P 2 2) ∧ (P 1 1 ≠ P 2 3) ∧ (P 1 1 ≠ P 3 1) ∧
  (P 1 1 ≠ P 3 2) ∧ (P 1 1 ≠ P 3 3) ∧
  (P 1 2 ≠ P 1 3) ∧ (P 1 2 ≠ P 2 1) ∧ (P 1 2 ≠ P 2 2) ∧
  (P 1 2 ≠ P 2 3) ∧ (P 1 2 ≠ P 3 1) ∧ (P 1 2 ≠ P 3 2) ∧
  (P 1 2 ≠ P 3 3) ∧ (P 1 3 ≠ P 2 1) ∧ (P 1 3 ≠ P 2 2) ∧
  (P 1 3 ≠ P 2 3) ∧ (P 1 3 ≠ P 3 1) ∧ (P 1 3 ≠ P 3 2) ∧
  (P 1 3 ≠ P 3 3) ∧ (P 2 1 ≠ P 2 2) ∧ (P 2 1 ≠ P 2 3) ∧
  (P 2 1 ≠ P 3 1) ∧ (P 2 1 ≠ P 3 2) ∧ (P 2 1 ≠ P 3 3) ∧
  (P 2 2 ≠ P 2 3) ∧ (P 2 2 ≠ P 3 1) ∧ (P 2 2 ≠ P 3 2) ∧
  (P 2 2 ≠ P 3 3) ∧ (P 2 3 ≠ P 3 1) ∧ (P 2 3 ≠ P 3 2) ∧
  (P 2 3 ≠ P 3 3) ∧ (P 3 1 ≠ P 3 2) ∧ (P 3 1 ≠ P 3 3) ∧
  (P 3 2 ≠ P 3 3) ∧
  (P 1 1 = 1 ∨ P 1 1 = 2 ∨ P 1 1 = 3 ∨ P 1 1 = 4 ∨ P 1 1 = 5 ∨ P 1 1 = 6 ∨ P 1 1 = 7 ∨ P 1 1 = 8 ∨ P 1 1 = 10) ∧
  (P 1 2 = 1 ∨ P 1 2 = 2 ∨ P 1 2 = 3 ∨ P 1 2 = 4 ∨ P 1 2 = 5 ∨ P 1 2 = 6 ∨ P 1 2 = 7 ∨ P 1 2 = 8 ∨ P 1 2 = 10) ∧
  (P 1 3 = 1 ∨ P 1 3 = 2 ∨ P 1 3 = 3 ∨ P 1 3 = 4 ∨ P 1 3 = 5 ∨ P 1 3 = 6 ∨ P 1 3 = 7 ∨ P 1 3 = 8 ∨ P 1 3 = 10) ∧
  (P 2 1 = 1 ∨ P 2 1 = 2 ∨ P 2 1 = 3 ∨ P 2 1 = 4 ∨ P 2 1 = 5 ∨ P 2 1 = 6 ∨ P 2 1 = 7 ∨ P 2 1 = 8 ∨ P 2 1 = 10) ∧
  (P 2 2 = 1 ∨ P 2 2 = 2 ∨ P 2 2 = 3 ∨ P 2 2 = 4 ∨ P 2 2 = 5 ∨ P 2 2 = 6 ∨ P 2 2 = 7 ∨ P 2 2 = 8 ∨ P 2 2 = 10) ∧
  (P 2 3 = 1 ∨ P 2 3 = 2 ∨ P 2 3 = 3 ∨ P 2 3 = 4 ∨ P 2 3 = 5 ∨ P 2 3 = 6 ∨ P 2 3 = 7 ∨ P 2 3 = 8 ∨ P 2 3 = 10) ∧
  (P 3 1 = 1 ∨ P 3 1 = 2 ∨ P 3 1 = 3 ∨ P 3 1 = 4 ∨ P 3 1 = 5 ∨ P 3 1 = 6 ∨ P 3 1 = 7 ∨ P 3 1 = 8 ∨ P 3 1 = 10) ∧
  (P 3 2 = 1 ∨ P 3 2 = 2 ∨ P 3 2 = 3 ∨ P 3 2 = 4 ∨ P 3 2 = 5 ∨ P 3 2 = 6 ∨ P 3 2 = 7 ∨ P 3 2 = 8 ∨ P 3 2 = 10) ∧
  (P 3 3 = 1 ∨ P 3 3 = 2 ∨ P 3 3 = 3 ∨ P 3 3 = 4 ∨ P 3 3 = 5 ∨ P 3 3 = 6 ∨ P 3 3 = 7 ∨ P 3 3 = 8 ∨ P 3 3 = 10))) :=
by sorry

end no_polynomial_exists_l363_363015


namespace find_a_and_c_find_sin_A_minus_B_l363_363164

variables {A B C a c b : ℝ}

-- Conditions
def triangle_conditions : Prop :=
  a + c = 6 ∧
  b = 2 ∧
  cos B = 7 / 9

-- To prove that a = 3 and c = 3
theorem find_a_and_c (T : triangle_conditions) : a = 3 ∧ c = 3 :=
  sorry

-- To prove the value of sin(A - B)
theorem find_sin_A_minus_B (T : triangle_conditions) (H : a = 3 ∧ c = 3) : sin (A - B) = 10 * real.sqrt 2 / 27 :=
  sorry

end find_a_and_c_find_sin_A_minus_B_l363_363164


namespace range_of_a_for_common_points_l363_363081

theorem range_of_a_for_common_points (a : ℝ) : (∃ x : ℝ, x > 0 ∧ ax^2 = Real.exp x) ↔ a ≥ Real.exp 2 / 4 :=
sorry

end range_of_a_for_common_points_l363_363081


namespace number_of_students_playing_both_l363_363691

open Set

variable (U : Type) (F C : Set U)

theorem number_of_students_playing_both (T : ℕ) (nF : ℕ) (nC : ℕ) (nNeither : ℕ)
  (H1 : T = 470) 
  (H2 : nF = 325) 
  (H3 : nC = 175) 
  (H4 : nNeither = 50) :
  Nat.card (F ∩ C) = 80 := by 
  sorry

end number_of_students_playing_both_l363_363691


namespace range_of_b_l363_363496

-- Define the conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16
def line_eq (x y b : ℝ) : Prop := y = x + b
def distance_point_line_eq (x y b d : ℝ) : Prop := 
  d = abs (b) / (Real.sqrt 2)
def at_least_three_points_on_circle_at_distance_one (b : ℝ) : Prop := 
  ∃ p1 p2 p3 : ℝ × ℝ, circle_eq p1.1 p1.2 ∧ circle_eq p2.1 p2.2 ∧ circle_eq p3.1 p3.2 ∧ 
  distance_point_line_eq p1.1 p1.2 b 1 ∧ distance_point_line_eq p2.1 p2.2 b 1 ∧ distance_point_line_eq p3.1 p3.2 b 1

-- The theorem statement to prove
theorem range_of_b (b : ℝ) (h : at_least_three_points_on_circle_at_distance_one b) : 
  -3 * Real.sqrt 2 ≤ b ∧ b ≤ 3 * Real.sqrt 2 := 
sorry

end range_of_b_l363_363496


namespace sum_of_x_coordinates_mod_20_l363_363688

theorem sum_of_x_coordinates_mod_20 (y x : ℤ) (h1 : y ≡ 7 * x + 3 [ZMOD 20]) (h2 : y ≡ 13 * x + 17 [ZMOD 20]) 
: ∃ (x1 x2 : ℤ), (0 ≤ x1 ∧ x1 < 20) ∧ (0 ≤ x2 ∧ x2 < 20) ∧ x1 ≡ 1 [ZMOD 10] ∧ x2 ≡ 11 [ZMOD 10] ∧ x1 + x2 = 12 := sorry

end sum_of_x_coordinates_mod_20_l363_363688


namespace explosion_eccentricity_l363_363765

noncomputable def eccentricity_of_curve
  (AB : ℝ)
  (t : ℝ)
  (v : ℝ) : ℝ :=
  let c := AB / 2 in
  let MA_MB := t * v in
  let a := MA_MB / 2 in
  c / a

theorem explosion_eccentricity :
  eccentricity_of_curve 1400 3 340 = 70 / 51 :=
by
  -- Left as an exercise to the reader
  sorry

end explosion_eccentricity_l363_363765


namespace point_reflection_x_axis_l363_363632

-- Definition of the original point P
def P : ℝ × ℝ := (-2, 5)

-- Function to reflect a point across the x-axis
def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Our theorem
theorem point_reflection_x_axis :
  reflect_x_axis P = (-2, -5) := by
  sorry

end point_reflection_x_axis_l363_363632


namespace approx_log_base_5_10_l363_363801

noncomputable def log_base (b a : ℝ) : ℝ := (Real.log a) / (Real.log b)

theorem approx_log_base_5_10 :
  let lg2 := 0.301
  let lg3 := 0.477
  let lg10 := 1
  let lg5 := lg10 - lg2
  log_base 5 10 = 10 / 7 :=
  sorry

end approx_log_base_5_10_l363_363801


namespace chairperson_and_committee_ways_l363_363125

-- Definitions based on conditions
def total_people : ℕ := 10
def ways_to_choose_chairperson : ℕ := total_people
def ways_to_choose_committee (remaining_people : ℕ) (committee_size : ℕ) : ℕ :=
  Nat.choose remaining_people committee_size

-- The resulting theorem
theorem chairperson_and_committee_ways :
  ways_to_choose_chairperson * ways_to_choose_committee (total_people - 1) 3 = 840 :=
by
  sorry

end chairperson_and_committee_ways_l363_363125


namespace students_attending_events_l363_363195

theorem students_attending_events :
    (M P G MP MG PG MPG total : ℕ)
    (hotal : total = M + P + G - (MP + MG + PG) + MPG)
    (hM : M = 50)
    (hP : P = 80)
    (hG : G = 60)
    (hMP : MP = 35)
    (hMG : MG = 10)
    (hPG : PG = 20)
    (hMPG : MPG = 8) :
    total = 133 :=
by
  rw [hotal, hM, hP, hG, hMP, hMG, hPG, hMPG]
  exact  rfl


end students_attending_events_l363_363195


namespace smallest_multiple_of_2009_with_digit_sum_2009_l363_363887

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits.sum

theorem smallest_multiple_of_2009_with_digit_sum_2009 :
  ∃ N, N = 5 * 10^223 - 10^220 - 10^49 - 1 ∧ 
        N % 2009 = 0 ∧ 
        sum_of_digits N = 2009 ∧ 
        (∀ M, M % 2009 = 0 ∧ sum_of_digits M = 2009 → N ≤ M) := 
  sorry

end smallest_multiple_of_2009_with_digit_sum_2009_l363_363887


namespace trains_crossing_time_l363_363296

-- Definitions of lengths and speeds
def train1_length : ℝ := 250 -- meters
def train2_length : ℝ := 120 -- meters
def train1_speed : ℝ := 80 * (5/18) -- km/hr to m/s
def train2_speed : ℝ := 50 * (5/18) -- km/hr to m/s

-- Definition of relative speed since trains are moving in opposite directions
def relative_speed : ℝ := train1_speed + train2_speed

-- Definition of total distance to be covered when they cross
def total_distance : ℝ := train1_length + train2_length

-- Definition of the time taken to cross each other
def time_to_cross : ℝ := total_distance / relative_speed

-- Theorem to be proven
theorem trains_crossing_time : time_to_cross = 370 * (9/325) :=
by
  sorry

end trains_crossing_time_l363_363296


namespace factorization_correct_l363_363431

def polynomial := λ a : ℝ, a^2 - 5 * a - 6
def factorized := λ a : ℝ, (a - 6) * (a + 1)

theorem factorization_correct (a : ℝ) : polynomial a = factorized a :=
  sorry

end factorization_correct_l363_363431


namespace smallest_positive_period_of_f_max_sin_B_sin_C_l363_363501
noncomputable theory

def f (x : ℝ) : ℝ := sin (π / 4 + x) * sin (π / 4 - x) + sqrt 3 * sin x * cos x

theorem smallest_positive_period_of_f : 
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ ε > 0, ε < T → ∃ x, f (x + ε) ≠ f x) ∧ T = π := 
sorry

theorem max_sin_B_sin_C (A B C : ℝ) (hA : 0 < A ∧ A < π):
  (∀ (B C : ℝ), 0 < B ∧ B < π → 0 < C ∧ C < π → A + B + C = π) →
  f (A / 2) = 1 → 
  ∃ (B C : ℝ), sin B + sin C = sqrt 3 :=
sorry

end smallest_positive_period_of_f_max_sin_B_sin_C_l363_363501


namespace harper_collected_more_than_maggie_l363_363099

def harper_percentage_more_than_maggie (P : ℝ) : Prop :=
  let maggie_collected := 50
  let harper_collected := maggie_collected + (P / 100 * maggie_collected)
  let neil_collected := harper_collected * 1.4
  neil_collected = 91 → P = 30

theorem harper_collected_more_than_maggie : harper_percentage_more_than_maggie 30 :=
by
  let maggie_collected := 50
  let harper_collected := maggie_collected + (30 / 100 * maggie_collected)
  let neil_collected := harper_collected * 1.4
  have h : neil_collected = 91 := sorry
  exact sorry

end harper_collected_more_than_maggie_l363_363099


namespace interval_of_monotonicity_l363_363050

def f (x: ℝ) : ℝ := sin x ^ 2 - sqrt 3 * cos x * cos (x + π / 2)

theorem interval_of_monotonicity :
  ∀ x, (0 ≤ x ∧ x ≤ π / 2) → (0 ≤ x ∧ x ≤ π / 3) :=
sorry

end interval_of_monotonicity_l363_363050


namespace compare_powers_l363_363781

theorem compare_powers :
  100^100 > 50^50 * 150^50 := sorry

end compare_powers_l363_363781


namespace quadratic_roots_interval_l363_363592

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l363_363592


namespace Razorback_tshirt_shop_made_total_money_l363_363218

noncomputable def t_shirt_price : ℝ := 16
noncomputable def total_sold : ℕ := 45
noncomputable def buy_3_get_1_free (total: ℕ) : ℕ := total / 4
noncomputable def discount_rate : ℝ := 0.10
noncomputable def sales_tax_rate : ℝ := 0.06

def total_money_made (price : ℝ) (sold : ℕ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let free_t_shirts := buy_3_get_1_free sold
  let paid_t_shirts := sold - free_t_shirts
  let total_cost_after_promotion := paid_t_shirts * price
  let discount_amount := discount * total_cost_after_promotion
  let total_cost_after_discount := total_cost_after_promotion - discount_amount
  let sales_tax_amount := tax * total_cost_after_discount
  total_cost_after_discount + sales_tax_amount

theorem Razorback_tshirt_shop_made_total_money :
  total_money_made t_shirt_price total_sold discount_rate sales_tax_rate = 518.98 := 
sorry

end Razorback_tshirt_shop_made_total_money_l363_363218


namespace correct_sequence_hangs_l363_363369

def configuration : list char :=
  ['a₁', 'a₂', 'a₃', 'a₁'⁻¹, 'a₂'⁻¹, 'a₃'⁻¹]

-- Composite predicate to ensure the desired hanging conditions
def hangs (conf: list char) (nails_removed: set char) : Prop :=
  match nails_removed.toList with
  | []         => conf = configuration -- All nails present
  | [a]        => conf = ['a₂', 'a₃', 'a₂'⁻¹, 'a₃'⁻¹] -- One nail removed
  | [a, b]     => conf = [] -- Two nails removed
  | _          => false -- More than two nails removed

theorem correct_sequence_hangs :
  hangs configuration ∅ ∧
  (∀ a, hangs configuration ({a} : set char)) ∧
  (∀ a b, hangs configuration ({a, b} : set char) = false) :=
sorry

end correct_sequence_hangs_l363_363369


namespace sum_of_two_numbers_with_8_trailing_zeros_and_90_divisors_l363_363287

theorem sum_of_two_numbers_with_8_trailing_zeros_and_90_divisors :
  ∃ (N1 N2 : ℕ), (trailingZeroes N1 = 8) ∧ (trailingZeroes N2 = 8) ∧ 
  (numDivisors N1 = 90) ∧ (numDivisors N2 = 90) ∧ (N1 ≠ N2) ∧ (N1 + N2 = 700000000) :=
sorry

-- Definitions for the trailing zeros and number of divisors functions
def trailingZeroes (n : ℕ) : ℕ :=
  nat.find_greatest (λ k, 10^k ∣ n) n

def numDivisors (n : ℕ) : ℕ :=
  (n.divisors.count)

end sum_of_two_numbers_with_8_trailing_zeros_and_90_divisors_l363_363287


namespace closest_value_sqrt_difference_l363_363014

theorem closest_value_sqrt_difference :
  let values := [0.10, 0.12, 0.14, 0.16, 0.18] in
  abs (sqrt 101 - sqrt 99 - 0.10) = min (abs (sqrt 101 - sqrt 99 - 0.10))
                                    (min (abs (sqrt 101 - sqrt 99 - 0.12))
                                         (min (abs (sqrt 101 - sqrt 99 - 0.14))
                                              (min (abs (sqrt 101 - sqrt 99 - 0.16))
                                                   (abs (sqrt 101 - sqrt 99 - 0.18))))) :=
by
  sorry

end closest_value_sqrt_difference_l363_363014


namespace consecutive_even_integers_sum_l363_363251

theorem consecutive_even_integers_sum :
  ∀ (y : Int), (y = 2 * (y + 2)) → y + (y + 2) = -6 :=
by
  intro y
  intro h
  sorry

end consecutive_even_integers_sum_l363_363251


namespace complex_division_l363_363409

theorem complex_division :
  (1 - complex.i) / (1 + complex.i) = -complex.i :=
by
  sorry

end complex_division_l363_363409


namespace supermarket_loss_l363_363833

theorem supermarket_loss 
  (s : ℕ) (a : ℕ) (k : ℕ) 
  (h_s : s = 54) (h_a : a = 216) (h_k : k = 2) :
  let sale_price := a / s in
  let regular_price := sale_price + k in
  let amount_no_sale := regular_price * s in
  let amount_less := amount_no_sale - a in
  amount_less = 108 := 
by
  -- Proof omitted
  sorry

end supermarket_loss_l363_363833


namespace simplify_trig_expression_l363_363712

theorem simplify_trig_expression :
  (sin (40 * real.pi / 180) + sin (80 * real.pi / 180)) /
  (cos (40 * real.pi / 180) + cos (80 * real.pi / 180)) = tan (60 * real.pi / 180) :=
by
  sorry

end simplify_trig_expression_l363_363712


namespace square_area_l363_363223

def circle_equation (x y : ℝ) : Prop := x^2 - y^2 + 10 * x - 6 * y + 15 = 0

theorem square_area {x y : ℝ}
  (h : circle_equation x y)
  (inscribed_in_square : ∃ s : ℝ, s > 0 ∧ ∀ a b : ℝ, circle_equation a b → a ∈ Icc (-s/2) (s/2) ∧ b ∈ Icc (-s/2) (s/2)) :
  ∃ (s : ℝ), s^2 = 76 :=
begin
  sorry
end

end square_area_l363_363223


namespace tan_A_of_right_triangle_trisected_angle_l363_363677

theorem tan_A_of_right_triangle_trisected_angle 
  (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (right_triangle_ABC : A) 
  (right_angle_at_C : ∀ {x y z : A}, x = B → y = C → z = A → x = 90)
  (D_on_AB : ∀ {x y z : A}, x = B → y = D → z = A → D ∈ line(x,y))
  (trisects_angle_ACB : ∀ {x y : A}, x = C → y = D → angle(C, D, A) = angle(C, D, B) / 3)
  (BD_AD_ratio : ∀ {x y : A}, x = B → y = D → (BD / AD) = (3 / 7))
  : ∃ (tanA : Real), tanA = sqrt(47) / 10 :=
by
  sorry

end tan_A_of_right_triangle_trisected_angle_l363_363677


namespace insurance_premium_correct_l363_363347

noncomputable def compute_insurance_premium (loan_amount appraisal_value cadastral_value : ℕ)
  (basic_rate : ℚ) (reducing_coefficient increasing_coefficient : ℚ) : ℚ :=
let insured_amount := max appraisal_value cadastral_value in
let adjusted_tariff := basic_rate * reducing_coefficient * increasing_coefficient in
insured_amount * adjusted_tariff

theorem insurance_premium_correct :
  compute_insurance_premium 20000000 14500000 15000000 0.002 0.8 1.3 = 31200 :=
by
  sorry

end insurance_premium_correct_l363_363347


namespace six_digit_even_numbers_l363_363746

/--
The number of six-digit even numbers that can be formed using the digits {1, 2, 3, 4, 5, 6}
without repeating any digits and where neither 1 nor 3 are adjacent to 5 is 108.
-/
theorem six_digit_even_numbers:
  ∃ n : ℕ, n = 108 ∧
    (∃ l : list ℕ, l.length = 6 ∧
                   (∀ x ∈ l, x ∈ ({1, 2, 3, 4, 5, 6} : finset ℕ)) ∧
                   (∀ x : ℕ, (x ∈ ({1, 2, 3, 4, 5, 6} : finset ℕ)) → x ∈ l) ∧
                   (l.nodup) ∧
                   (l.ilast = some 2 ∨ l.ilast = some 4 ∨ l.ilast = some 6) ∧
                   (∀ i : ℕ, i < l.length - 1 → ¬ (l.nth i = some 5 ∧ (l.nth (i + 1) = some 1 ∨ l.nth (i + 1) = some 3)) ∧
                                 ¬ (l.nth i = some 1 ∧ l.nth (i + 1) = some 5) ∧
                                 ¬ (l.nth i = some 3 ∧ l.nth (i + 1) = some 5))) := 
sorry

end six_digit_even_numbers_l363_363746


namespace sages_success_l363_363800

-- Assume we have a finite type representing our 1000 colors
inductive Color
| mk : Fin 1000 → Color

open Color

-- Define the sages
def Sage : Type := Fin 11

-- Define the problem conditions into a Lean structure
structure Problem :=
  (sages : Fin 11)
  (colors : Fin 1000)
  (assignments : Sage → Color)
  (strategies : Sage → (Fin 1024 → Fin 2))

-- Define the success condition
def success (p : Problem) : Prop :=
  ∃ (strategies : Sage → (Fin 1024 → Fin 2)),
    ∀ (assignment : Sage → Color),
      ∃ (color_guesses : Sage → Color),
        (∀ s, color_guesses s = assignment s)

-- The sages will succeed in determining the colors of their hats.
theorem sages_success : ∀ (p : Problem), success p := by
  sorry

end sages_success_l363_363800


namespace problem_statement_l363_363502

-- Define the function f(x)
def f (a x : ℝ) : ℝ := a * Real.log x + x^2 - x

-- Define the condition that f(x) ≥ 0 for x ≥ 1
theorem problem_statement (a : ℝ) (x : ℝ) (h1 : x ≥ 1) : a ≥ -1 → f a x ≥ 0 :=
sorry

end problem_statement_l363_363502


namespace ratio_of_fish_cat_to_dog_l363_363656

theorem ratio_of_fish_cat_to_dog (fish_dog : ℕ) (cost_per_fish : ℕ) (total_spent : ℕ)
  (h1 : fish_dog = 40)
  (h2 : cost_per_fish = 4)
  (h3 : total_spent = 240) :
  (total_spent / cost_per_fish - fish_dog) / fish_dog = 1 / 2 := by
  sorry

end ratio_of_fish_cat_to_dog_l363_363656


namespace eval_36_pow_five_over_two_l363_363877

theorem eval_36_pow_five_over_two : (36 : ℝ)^(5/2) = 7776 := by
  sorry

end eval_36_pow_five_over_two_l363_363877


namespace troy_needs_more_money_l363_363282

theorem troy_needs_more_money (initial_savings : ℕ) (sold_computer : ℕ) (new_computer_cost : ℕ) :
  initial_savings = 50 → sold_computer = 20 → new_computer_cost = 80 → 
  new_computer_cost - (initial_savings + sold_computer) = 10 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end troy_needs_more_money_l363_363282


namespace five_rams_blocks_different_kinds_l363_363217

-- Define the conditions: 6 faces, each painted either red or blue, each color on at least one face
def face_count : ℕ := 6
def color_count : ℕ := 2
def min_use_per_color : ℕ := 1

theorem five_rams_blocks_different_kinds :
  ∃ n, n = 62 ∧ 
  (∀ f : Fin face_count → Fin color_count, 
    (f.range.Count = face_count ∧ ∀ c ∈ Finset.univ, ↑(f.count c) ≥ min_use_per_color)) :=
sorry

end five_rams_blocks_different_kinds_l363_363217


namespace number_of_terms_in_sequence_l363_363953

theorem number_of_terms_in_sequence : ∃ n : ℕ, 6 + (n-1) * 4 = 154 ∧ n = 38 :=
by
  sorry

end number_of_terms_in_sequence_l363_363953


namespace max_elements_in_A_l363_363053

open Nat

def satisfies_condition (A : Finset ℕ) :=
  ∀ (x y : ℕ), x ≠ y → x ∈ A → y ∈ A → |x - y| * 25 ≥ x * y

theorem max_elements_in_A (A : Finset ℕ) (hA : satisfies_condition A) :
  A.card ≤ 12 := sorry

end max_elements_in_A_l363_363053


namespace range_of_a_l363_363914

noncomputable def f : ℝ → ℝ := sorry

variables (a : ℝ)
  (h1 : ∀ x ∈ ℝ, f (-x) = -f x)
  (h2 : ∀ x ∈ ℝ, f (x+3) = -f x)
  (h3 : f 1 > 3)
  (h4 : f 11 = (2*a - 1) / (3 - a))

theorem range_of_a (h5 : ∀ x ∈ ℝ, f (x+6) = f x) : 3 < a ∧ a < 8 :=
  sorry

end range_of_a_l363_363914


namespace not_possible_equal_monochromatic_dichromatic_sides_l363_363260

theorem not_possible_equal_monochromatic_dichromatic_sides :
  ∀ (color : Fin 222 → Prop), 
  let monochromatic := λ (i : Fin 222), color i = color (i + 1) in
  let dichromatic := λ (i : Fin 222), color i ≠ color (i + 1) in
  (Finset.card (Finset.filter monochromatic (Finset.univ : Finset (Fin 222)))
   ≠ Finset.card (Finset.filter dichromatic (Finset.univ : Finset (Fin 222)))) :=
by
  sorry

end not_possible_equal_monochromatic_dichromatic_sides_l363_363260


namespace sum_of_squares_of_coefficients_l363_363779

theorem sum_of_squares_of_coefficients :
  let poly := 5 * (X^6 + 4 * X^4 + 2 * X^2 + 1)
  let coeffs := [5, 20, 10, 5]
  (coeffs.map (λ c => c * c)).sum = 550 := 
by
  sorry

end sum_of_squares_of_coefficients_l363_363779


namespace polar_coord_circle_eqn_l363_363139

theorem polar_coord_circle_eqn :
  ∃ ρ θ, (circle_center : ℝ × ℝ) (pole : ℝ × ℝ),
  circle_center = (10, 0) ∧ pole = (0, 0) ∧
  (ρ, θ) = (20 * real.cos θ, θ) :=
by
  sorry

end polar_coord_circle_eqn_l363_363139


namespace quadratic_roots_interval_l363_363593

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l363_363593


namespace cyclic_sum_inequality_l363_363159

variable {a b c : ℝ} (n : ℤ)
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_abc : a * b * c = 1)
variable (h_n : n ≥ 2)

theorem cyclic_sum_inequality :
  (a/Real.root (b + c) (n:ℝ)) + (b/Real.root (c + a) (n:ℝ)) + (c/Real.root (a + b) (n:ℝ)) ≥ 3/Real.root 2 n := 
sorry

end cyclic_sum_inequality_l363_363159


namespace limit_cos_plus_one_pow_sin_l363_363407

open Real

theorem limit_cos_plus_one_pow_sin (f : ℝ → ℝ) :
  (∀ x, f x = (cos x + 1) ^ (sin x)) → 
  filter.tendsto f (nhds π/2) (nhds 1) :=
begin
  intro h_f,
  sorry
end

end limit_cos_plus_one_pow_sin_l363_363407


namespace soccer_team_lineups_l363_363690

theorem soccer_team_lineups :
  let total_players := 18
  let goalkeeper_choices := total_players
  let defender_choices := Nat.choose (total_players - 1) 4
  let midfield_or_forward_choices := Nat.choose (total_players - 1 - 4) 4
  goalkeeper_choices * defender_choices * midfield_or_forward_choices = 30_544_200 :=
by
  sorry

end soccer_team_lineups_l363_363690


namespace sum_of_coefficients_l363_363000

def P (x : ℝ) : ℝ :=
  -3 * (x^8 - x^5 + 2*x^3 - 6) + 5 * (x^4 + 3*x^2) - 4 * (x^6 - 5)

theorem sum_of_coefficients : P 1 = 48 := by
  sorry

end sum_of_coefficients_l363_363000


namespace area_of_original_triangle_l363_363383

theorem area_of_original_triangle (a : ℝ) : 
  let hypotenuse := sqrt 2 * a in
  let leg1 := sqrt 2 * a in
  let leg2 := 2 * a in
  let area := (1/2) * leg1 * leg2 in
  area = sqrt 2 * a ^ 2 :=
by
  sorry

end area_of_original_triangle_l363_363383


namespace simplify_expression_l363_363206

variable (a b c d x y z : ℝ)

theorem simplify_expression :
  (cx * (b^2 * x^3 + 3 * a^2 * y^3 + c^2 * z^3) + dz * (a^2 * x^3 + 3 * c^2 * y^3 + b^2 * z^3)) / (cx + dz) =
  b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3 :=
sorry

end simplify_expression_l363_363206


namespace max_friends_is_m_l363_363617

-- Define the basic concepts: set of persons, friendship relation
variables {α : Type*} [fintype α] (m : ℕ)
variables (friends : α → α → Prop)

-- Axioms based on the provided conditions
axiom mutual_friendship : ∀ {a b : α}, friends a b → friends b a
axiom no_self_friendship : ∀ {a : α}, ¬ friends a a
axiom unique_common_friend : ∀ (s : finset α), s.card = m → ∃! (f : α), ∀ (a : α), a ∈ s → friends a f

-- Prove the maximum number of friends a person can have is exactly m
theorem max_friends_is_m : ∀ (a : α), (finset.univ.filter (friends a)).card ≤ m :=
by sorry

end max_friends_is_m_l363_363617


namespace determine_digit_phi_l363_363009

theorem determine_digit_phi (Φ : ℕ) (h1 : Φ > 0) (h2 : Φ < 10) (h3 : 504 / Φ = 40 + 3 * Φ) : Φ = 8 :=
by
  sorry

end determine_digit_phi_l363_363009


namespace part1_part2_l363_363904

-- Defining the function f(x) and the given conditions
def f (x a : ℝ) := x^2 - a * x + 2 * a - 2

-- Given conditions
variables (a : ℝ)
axiom f_condition : ∀ (x : ℝ), f (2 + x) a * f (2 - x) a = 4
axiom a_gt_0 : a > 0
axiom fx_bounds : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → 1 ≤ f x a ∧ f x a ≤ 3

-- To prove (part 1)
theorem part1 (h : f 2 a + f 3 a = 6) : a = 2 := sorry

-- To prove (part 2)
theorem part2 : (4 - (2 * Real.sqrt 6) / 3) ≤ a ∧ a ≤ 5 / 2 := sorry

end part1_part2_l363_363904


namespace number_of_subsets_l363_363093

theorem number_of_subsets (n : ℕ) (S : finset (fin n)) :
  S.card = n → (2 ^ n = (finset.powerset S).card) :=
by sorry

end number_of_subsets_l363_363093


namespace hugo_mountain_elevation_l363_363959

theorem hugo_mountain_elevation (H : ℕ) 
  (C1 : ∀ H, Boris : ℕ, Boris = H - 2500)
  (C2 : 3 * H = 4 * (H - 2500)) : 
  H = 10000 :=
by
  sorry

end hugo_mountain_elevation_l363_363959


namespace problem_quadratic_has_real_root_l363_363598

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l363_363598


namespace substitution_and_elimination_l363_363722

theorem substitution_and_elimination {x y : ℝ} :
  y = 2 * x + 1 → 5 * x - 2 * y = 7 → 5 * x - 4 * x - 2 = 7 :=
by
  intros h₁ h₂
  rw [h₁] at h₂
  exact h₂

end substitution_and_elimination_l363_363722


namespace nonnegative_for_interval_l363_363895

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 * (x - 2)^2) / ((1 - x) * (1 + x + x^2))

theorem nonnegative_for_interval (x : ℝ) : (f x >= 0) ↔ (0 <= x) :=
by
  sorry

end nonnegative_for_interval_l363_363895


namespace area_triangle_ABC_correct_l363_363645

open_locale real

noncomputable def area_of_triangle_ABC 
  (A B : ℝ) 
  (DE : ℝ) 
  (angle_A : A = 45) 
  (angle_B : B = 30) 
  (median_CM : Prop)
  (inscribed_circles_touchpoints : Prop)
  (length_DE : DE = 4 * (real.sqrt 2 - 1)) : ℝ := 
  16 * (real.sqrt 3 + 1)

theorem area_triangle_ABC_correct 
  (A B : ℝ) 
  (DE : ℝ) 
  (angle_A : A = 45) 
  (angle_B : B = 30) 
  (median_CM : Prop)
  (inscribed_circles_touchpoints : Prop)
  (length_DE : DE = 4 * (real.sqrt 2 - 1)) : 
  area_of_triangle_ABC A B DE angle_A angle_B median_CM inscribed_circles_touchpoints length_DE = 16 * (real.sqrt 3 + 1) := 
  sorry

end area_triangle_ABC_correct_l363_363645


namespace a_arithmetic_sum_terms_l363_363060

-- Define sequences a_n and b_n
variable (a b : ℕ → ℕ)
variable n : ℕ

-- Conditions provided by the problem
def a_positive : Prop := ∀ n, a n > 0
def b_positive : Prop := ∀ n, b n > 0
def a1 : Prop := a 1 = 1
def b1 : Prop := b 1 = 1
def sum_b3 : Prop := b 1 + b 2 + b 3 = 7
def b4_condition : Prop := b 4 = (a 1 + a 2) * b 2
def b2_b4_condition : Prop := b 2 * b 4 = 16
def recurrence_relation : Prop := ∀ n, (a (n + 1))^2 - 2 * a (n + 1) = (a n)^2 + 2 * a n

-- Part 1: Prove that the sequence a_n is an arithmetic sequence
theorem a_arithmetic (a_positive a1 recurrence_relation : Prop) :
  ∀ n, a (n + 1) = a n + 2 :=
sorry

-- Part 2: Find the sum of the first n terms of the sequence a_n * b_n
theorem sum_terms (a_positive b_positive a1 b1 sum_b3 b4_condition b2_b4_condition recurrence_relation : Prop) :
  ∑ i in finset.range n, a i * b i = (2 * n - 3) * 2 ^ n + 3 :=
sorry

end a_arithmetic_sum_terms_l363_363060


namespace distance_from_point_to_x_axis_l363_363469

-- Define the conditions as variables or hypotheses
variable (M : ℝ × ℝ)  -- M is a point with x and y coordinates

def parabola_equation : Prop := M.2 ^ 2 = 12 * M.1
def distance_to_focus (focus : ℝ × ℝ) : Prop := Real.dist M focus = 9

-- Define the focus of the given parabola
def focus : ℝ × ℝ := (3, 0)

-- Here's the theorem statement
theorem distance_from_point_to_x_axis 
  (M : ℝ × ℝ) 
  (h1 : parabola_equation M) 
  (h2 : distance_to_focus M focus) : 
  M.2 = 6 * Real.sqrt 2 := by sorry

end distance_from_point_to_x_axis_l363_363469


namespace sin_alpha_value_l363_363961

variables (α : ℝ)

-- Define the conditions given in the problem
def cos_alpha := (cos α = 3 / 5)
def alpha_range := (3 / 2 * π < α ∧ α < 2 * π)

-- State the problem as a theorem in Lean 4
theorem sin_alpha_value (h1 : cos_alpha α) (h2 : alpha_range α) : sin α = -4 / 5 :=
sorry

end sin_alpha_value_l363_363961


namespace christine_amount_l363_363848

theorem christine_amount (S C : ℕ) 
  (h1 : S + C = 50)
  (h2 : C = S + 30) :
  C = 40 :=
by
  -- Proof goes here.
  -- This part should be filled in to complete the proof.
  sorry

end christine_amount_l363_363848


namespace find_b_of_fraction_eq_decimal_l363_363443

theorem find_b_of_fraction_eq_decimal (b : ℕ) (h₁ : 0.82 = (5*b + 21)/(7*b + 13)) (h₂ : b > 0) : b = 14 :=
sorry

end find_b_of_fraction_eq_decimal_l363_363443


namespace troy_needs_more_money_to_buy_computer_l363_363286

theorem troy_needs_more_money_to_buy_computer :
  ∀ (price_new_computer savings sale_old_computer : ℕ),
  price_new_computer = 80 →
  savings = 50 →
  sale_old_computer = 20 →
  (price_new_computer - (savings + sale_old_computer)) = 10 :=
by
  intros price_new_computer savings sale_old_computer Hprice Hsavings Hsale
  sorry

end troy_needs_more_money_to_buy_computer_l363_363286


namespace cannot_determine_total_inhabitants_without_additional_info_l363_363123

variable (T : ℝ) (M F : ℝ)

axiom inhabitants_are_males_females : M + F = 1
axiom twenty_percent_of_males_are_literate : M * 0.20 * T = 0.20 * M * T
axiom twenty_five_percent_of_all_literates : 0.25 = 0.25 * T / T
axiom thirty_two_five_percent_of_females_are_literate : F = 1 - M ∧ F * 0.325 * T = 0.325 * (1 - M) * T

theorem cannot_determine_total_inhabitants_without_additional_info :
  ∃ (T : ℝ), True ↔ False := by
  sorry

end cannot_determine_total_inhabitants_without_additional_info_l363_363123


namespace find_b_values_l363_363555

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l363_363555


namespace expression_evaluates_to_3_l363_363879

theorem expression_evaluates_to_3 :
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = 3 :=
sorry

end expression_evaluates_to_3_l363_363879


namespace right_triangle_product_l363_363126

theorem right_triangle_product 
  (ABC : Triangle)
  (C B A : Point)
  (right_triangle : ABC.angle B = 90)
  (P : Point)
  (Q : Point)
  (perpendicular_PC : perpendicular_from_point_to_line C A B P)
  (perpendicular_QB : perpendicular_from_point_to_line B A C Q)
  (circumcircle : Circle)
  (XY_pnts : XY_points_intersect_circumcircle ABC circumcircle P Q)
  (XP : ℝ)
  (PQ : ℝ)
  (QY : ℝ)
  (condition_XP : XP = 12)
  (condition_PQ : PQ = 30)
  (condition_QY : QY = 18) : 
  AB * AC = 30 := 
sorry

end right_triangle_product_l363_363126


namespace perimeter_inner_triangle_l363_363277

variables (P Q R U V W : Type)
variables (PQ QR PR ℓ_P ℓ_Q ℓ_R : ℝ)
variables (PQ_len QR_len PR_len ℓ_P_int ℓ_Q_int ℓ_R_int : ℝ)

-- Given the side lengths of triangle PQR
def side_lengths : PQ_len = 150 ∧ QR_len = 270 ∧ PR_len = 210 := sorry

-- Given the segment lengths of the intersections of lines drawn parallel
def segment_intersections : ℓ_P_int = 75 ∧ ℓ_Q_int = 35 ∧ ℓ_R_int = 25 := sorry

-- Prove the perimeter of the inner triangle formed by ℓ_P, ℓ_Q, ℓ_R is 135
theorem perimeter_inner_triangle :
  side_lengths ∧ segment_intersections → ℓ_P_int + ℓ_Q_int + ℓ_R_int = 135 := sorry

end perimeter_inner_triangle_l363_363277


namespace area_of_white_portion_of_sign_l363_363262

/-- The word "SHARP" in block letters is painted in black with strokes 1 unit wide on a 6 by 20 rectangular white sign.
Prove that the area of the white portion of the sign is 63 square units. -/
def white_portion_area : Prop :=
  let total_area := 6 * 20 in
  let black_area_S := 2 * (3 * 1) + 1 * 1 in
  let black_area_H := 2 * (6 * 1) + 1 * 4 in
  let black_area_A := 6 * 1 + 2 * (1 * 2) in
  let black_area_R := 6 * 1 + 1 * 4 + 1 * 2 in
  let black_area_P := 6 * 1 + 4 * 1 + 2 * 1 in
  let total_black_area := black_area_S + black_area_H + black_area_A + black_area_R + black_area_P in
  let white_area := total_area - total_black_area in
  white_area = 63

theorem area_of_white_portion_of_sign : white_portion_area :=
  by sorry

end area_of_white_portion_of_sign_l363_363262


namespace systematic_sampling_works_l363_363041

def missiles : List ℕ := List.range' 1 60 

-- Define the systematic sampling function
def systematic_sampling (start interval n : ℕ) : List ℕ :=
  List.range' 0 n |>.map (λ i => start + i * interval)

-- Stating the proof problem.
theorem systematic_sampling_works :
  systematic_sampling 5 12 5 = [5, 17, 29, 41, 53] :=
sorry

end systematic_sampling_works_l363_363041


namespace inclination_angle_l363_363974

/-
Given the line equation y - 2x - 1 = 0,
Prove that the inclination angle, α, of the line is arctan(2).
-/
theorem inclination_angle (α : ℝ) (h : ∀ x y : ℝ, y - 2 * x - 1 = 0 → tan α = 2 ∧ 0 < α ∧ α < π) : α = Real.arctan 2 := 
sorry

end inclination_angle_l363_363974


namespace min_sum_of_m_n_l363_363901

theorem min_sum_of_m_n (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 3) (h3 : 8 ∣ (180 * m * n - 360 * m)) : m + n = 5 :=
sorry

end min_sum_of_m_n_l363_363901


namespace more_children_got_off_than_got_on_l363_363353

-- Define the initial number of children on the bus
def initial_children : ℕ := 36

-- Define the number of children who got off the bus
def children_got_off : ℕ := 68

-- Define the total number of children on the bus after changes
def final_children : ℕ := 12

-- Define the unknown number of children who got on the bus
def children_got_on : ℕ := sorry -- We will use the conditions to solve for this in the proof

-- The main proof statement
theorem more_children_got_off_than_got_on : (children_got_off - children_got_on = 24) :=
by
  -- Write the equation describing the total number of children after changes
  have h1 : initial_children - children_got_off + children_got_on = final_children := sorry
  -- Solve for the number of children who got on the bus (children_got_on)
  have h2 : children_got_on = final_children + (children_got_off - initial_children) := sorry
  -- Substitute to find the required difference
  have h3 : children_got_off - final_children - (children_got_off - initial_children) = 24 := sorry
  -- Conclude the proof
  exact sorry


end more_children_got_off_than_got_on_l363_363353


namespace tangent_line_at_one_two_tangent_line_through_one_one_l363_363505

noncomputable def f (x : ℝ) : ℝ := x^2 + 1

theorem tangent_line_at_one_two :
  tangent_line f 1 = λ x : ℝ, 2 * x :=
sorry

theorem tangent_line_through_one_one :
  ∃ x₀, tangent_line f x₀ = λ x : ℝ, 4 * x - 3 :=
sorry

end tangent_line_at_one_two_tangent_line_through_one_one_l363_363505


namespace number_of_terms_in_sequence_l363_363954

theorem number_of_terms_in_sequence : ∃ n : ℕ, 6 + (n-1) * 4 = 154 ∧ n = 38 :=
by
  sorry

end number_of_terms_in_sequence_l363_363954


namespace train_cross_bridge_in_56_seconds_l363_363382

noncomputable def train_pass_time (length_train length_bridge : ℝ) (speed_train_kmh : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_train_ms := speed_train_kmh * (1000 / 3600)
  total_distance / speed_train_ms

theorem train_cross_bridge_in_56_seconds :
  train_pass_time 560 140 45 = 56 :=
by
  -- The proof can be added here
  sorry

end train_cross_bridge_in_56_seconds_l363_363382


namespace mos_to_ory_bus_encounter_l363_363404

def encounter_buses (departure_time : Nat) (encounter_bus_time : Nat) (travel_time : Nat) : Nat := sorry

theorem mos_to_ory_bus_encounter :
  encounter_buses 0 30 5 = 10 :=
sorry

end mos_to_ory_bus_encounter_l363_363404


namespace perfect_square_trinomial_l363_363534

theorem perfect_square_trinomial (a k : ℝ) : (∃ b : ℝ, (a^2 + 2*k*a + 9 = (a + b)^2)) ↔ (k = 3 ∨ k = -3) := 
by
  sorry

end perfect_square_trinomial_l363_363534


namespace problem_quadratic_has_real_root_l363_363597

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l363_363597


namespace impossible_to_achieve_goal_l363_363684

theorem impossible_to_achieve_goal :
  ∀ p : ℕ → ℕ, (∀ i, i ≤ 2018 → p i = Nat.prime i) →
  ∀ P : ℕ → ℕ → Prop,
    (∀ n m, P n m → ∃ k, P k (n + m + 1)) →
    (∀ n k, P n k → ∀ a b, k = a + b → P (k + 1) a) →
  ¬(∃ p' : ℕ → ℕ, (∀ i, i ≤ 2018 → p' i = 2018)) :=
by sorry

end impossible_to_achieve_goal_l363_363684


namespace find_pairs_of_positive_integers_l363_363007

theorem find_pairs_of_positive_integers (x y : ℕ) (h : x > 0 ∧ y > 0) (h_eq : x + y + x * y = 2006) :
  (x, y) = (2, 668) ∨ (x, y) = (668, 2) ∨ (x, y) = (8, 222) ∨ (x, y) = (222, 8) :=
sorry

end find_pairs_of_positive_integers_l363_363007


namespace insurance_calculation_l363_363348

-- Define the conditions
def baseRate : ℝ := 0.002 -- 0.2% as a decimal
def reducingCoefficient : ℝ := 0.8
def increasingCoefficient : ℝ := 1.3
def assessedValue : ℝ := 14500000
def cadasterValue : ℝ := 15000000
def loanAmount : ℝ := 20000000

-- Define the adjusted tariff
def adjustedTariff : ℝ := baseRate * reducingCoefficient * increasingCoefficient

-- Define the insurable amount
def insurableAmount : ℝ := max assessedValue cadasterValue

-- Define the proof problem to show equality
theorem insurance_calculation :
  adjustedTariff = 0.00208 ∧ insurableAmount * adjustedTariff = 31200 := by
  sorry

end insurance_calculation_l363_363348


namespace knights_probability_sum_l363_363272

theorem knights_probability_sum :
  let n := 30 in
  let k := 4 in
  let total_ways := Nat.choose n k in
  let non_adjacent_ways := Nat.choose (n - k) k in 
  let Q := 1 - (non_adjacent_ways / total_ways : ℚ) in
  Q.num + Q.denom = 799 := by
  sorry

end knights_probability_sum_l363_363272


namespace sum_of_two_numbers_with_8_trailing_zeros_and_90_divisors_l363_363288

theorem sum_of_two_numbers_with_8_trailing_zeros_and_90_divisors :
  ∃ (N1 N2 : ℕ), (trailingZeroes N1 = 8) ∧ (trailingZeroes N2 = 8) ∧ 
  (numDivisors N1 = 90) ∧ (numDivisors N2 = 90) ∧ (N1 ≠ N2) ∧ (N1 + N2 = 700000000) :=
sorry

-- Definitions for the trailing zeros and number of divisors functions
def trailingZeroes (n : ℕ) : ℕ :=
  nat.find_greatest (λ k, 10^k ∣ n) n

def numDivisors (n : ℕ) : ℕ :=
  (n.divisors.count)

end sum_of_two_numbers_with_8_trailing_zeros_and_90_divisors_l363_363288


namespace single_elimination_games_l363_363639

theorem single_elimination_games (n : ℕ) (h : n = 32) : 
  ∃ k : ℕ, k = n - 1 := 
by {
  use 31,
  rw h,
  exact rfl,
  sorry -- Assuming the theorem statement here as per instructions
}

end single_elimination_games_l363_363639


namespace g_increasing_on_interval_l363_363085

noncomputable def f (x : ℝ) : ℝ := Real.sin ((1/5) * x + 13 * Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin ((1/5) * (x - 10 * Real.pi / 3) + 13 * Real.pi / 6)

theorem g_increasing_on_interval : ∀ x y : ℝ, (π ≤ x ∧ x < y ∧ y ≤ 2 * π) → g x < g y :=
by
  intro x y h
  -- Mathematical steps to prove this
  sorry

end g_increasing_on_interval_l363_363085


namespace solve_inequality_l363_363211

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (log x (7 * x - 6)) ^ 2 - 4 * (cos (Real.pi * x) - 1) ≤ 0

theorem solve_inequality (x : ℝ) :
  satisfies_inequality x ↔ 
  (6 / 7 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 6) ∨ (∃ n : ℕ, x = 2 * n ∧ n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 3) :=
sorry

end solve_inequality_l363_363211


namespace infinite_natural_solutions_l363_363437

theorem infinite_natural_solutions :
  ∃ (x y z : ℕ), (x^3 + y^4 = z^5) →
  ∀ (k : ℕ), (∃ (x y z : ℕ), (x = k^20 * x) ∧ (y = k^15 * y) ∧ (z = k^12 * z) ∧ (x^3 + y^4 = z^5)) :=
begin
  sorry
end

end infinite_natural_solutions_l363_363437


namespace find_b_values_l363_363551

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l363_363551


namespace angle_B_in_triangle_l363_363978

/-- In triangle ABC, if BC = √3, AC = √2, and ∠A = π/3,
then ∠B = π/4. -/
theorem angle_B_in_triangle
  (BC AC : ℝ) (A B : ℝ)
  (hBC : BC = Real.sqrt 3)
  (hAC : AC = Real.sqrt 2)
  (hA : A = Real.pi / 3) :
  B = Real.pi / 4 :=
sorry

end angle_B_in_triangle_l363_363978


namespace maximum_total_profit_l363_363816

def q (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 20 then 1260 / (x + 1)
  else if 20 ≤ x ∧ x ≤ 180 then 90 - 3 * real.sqrt (5 * x)
  else 0

def W (x : ℝ) : ℝ :=
  x * q x

theorem maximum_total_profit : ∃ x : ℝ, x = 80 ∧ W x = 240000 :=
by {
  sorry
}

end maximum_total_profit_l363_363816


namespace DM_eq_DN_l363_363647

open Real

variables {A B C P M N D : Point}
variable {triangle : Triangle A B C}
variable (h1 : ∠PAC = ∠PBC)
variable (h2 : IsPerpendicularProjection P AC M)
variable (h3 : IsPerpendicularProjection P BC N)
variable (h4 : IsMidpoint D A B)

theorem DM_eq_DN 
  (h1 : ∠PAC = ∠PBC)
  (h2 : IsPerpendicularProjection P AC M)
  (h3 : IsPerpendicularProjection P BC N)
  (h4 : IsMidpoint D A B) :
  dist D M = dist D N :=
sorry

end DM_eq_DN_l363_363647


namespace cos_theta_plus_5pi_over_6_l363_363070

-- Define the conditions
variables (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : sin(θ / 2 + π / 6) = 4 / 5)

-- Statement of the theorem
theorem cos_theta_plus_5pi_over_6 :
  cos(θ + 5 * π / 6) = -24 / 25 :=
sorry

end cos_theta_plus_5pi_over_6_l363_363070


namespace find_value_of_expression_l363_363966

variable (α : ℝ)

theorem find_value_of_expression 
  (h : Real.tan α = 3) : 
  (Real.sin (2 * α) / (Real.cos α)^2) = 6 := 
by 
  sorry

end find_value_of_expression_l363_363966


namespace top_leftmost_rectangle_is_B_l363_363876

-- Definitions for the side lengths of each rectangle
def A_w : ℕ := 6
def A_x : ℕ := 2
def A_y : ℕ := 7
def A_z : ℕ := 10

def B_w : ℕ := 2
def B_x : ℕ := 1
def B_y : ℕ := 4
def B_z : ℕ := 8

def C_w : ℕ := 5
def C_x : ℕ := 11
def C_y : ℕ := 6
def C_z : ℕ := 3

def D_w : ℕ := 9
def D_x : ℕ := 7
def D_y : ℕ := 5
def D_z : ℕ := 9

def E_w : ℕ := 11
def E_x : ℕ := 4
def E_y : ℕ := 9
def E_z : ℕ := 1

-- The problem statement to prove
theorem top_leftmost_rectangle_is_B : 
  (B_w = 2 ∧ B_y = 4) ∧ 
  (A_w = 6 ∨ D_w = 9 ∨ C_w = 5 ∨ E_w = 11) ∧
  (A_y = 7 ∨ D_y = 5 ∨ C_y = 6 ∨ E_y = 9) → 
  (B_w = 2 ∧ ∀ w : ℕ, w = 6 ∨ w = 5 ∨ w = 9 ∨ w = 11 → B_w < w) :=
by {
  -- skipping the proof
  sorry
}

end top_leftmost_rectangle_is_B_l363_363876


namespace compute_expression_l363_363855

theorem compute_expression : 12 * (1 / 15) * 30 = 24 := 
by 
  sorry

end compute_expression_l363_363855


namespace part_I_has_one_zero_l363_363084

-- Condition: given the function f(x)
def f_I (x : ℝ) : ℝ := Real.log x + (1/2) * x^2 - 2 * x
  
-- Proposition statement for Part (I)
theorem part_I_has_one_zero : 
  (∃! x : ℝ, x > 0 ∧ f_I x = 0) :=
sorry

end part_I_has_one_zero_l363_363084


namespace Mika_stickers_l363_363187

section stickers_problem

variables (initial_stickers bought_stickers birthday_stickers
           given_sister_stickers used_decorate_stickers : ℕ)

theorem Mika_stickers :
  initial_stickers = 20 →
  bought_stickers = 26 →
  birthday_stickers = 20 →
  given_sister_stickers = 6 →
  used_decorate_stickers = 58 →
  initial_stickers + bought_stickers + birthday_stickers - given_sister_stickers - used_decorate_stickers = 2 :=
begin
  intros h₁ h₂ h₃ h₄ h₅,
  rw [h₁, h₂, h₃, h₄, h₅],
  norm_num,
end

end stickers_problem

end Mika_stickers_l363_363187


namespace complex_modulus_eq_l363_363892
open Complex

theorem complex_modulus_eq : ∃ n > 0, (|4 + (n : ℂ)*I| = 4 * Real.sqrt 13 ↔ n = 8 * Real.sqrt 3) :=
by
  sorry

end complex_modulus_eq_l363_363892


namespace only_one_P_Q_l363_363520

def P (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def Q (a : ℝ) : Prop := ∀ x : ℝ, x^2 - x + a = 0

theorem only_one_P_Q (a : ℝ) :
  (P a ∧ ¬ Q a) ∨ (Q a ∧ ¬ P a) ↔
  (a < 0) ∨ (1/4 < a ∧ a < 4) :=
sorry

end only_one_P_Q_l363_363520


namespace insurance_premium_correct_l363_363344

noncomputable def compute_insurance_premium (loan_amount appraisal_value cadastral_value : ℕ)
  (basic_rate : ℚ) (reducing_coefficient increasing_coefficient : ℚ) : ℚ :=
let insured_amount := max appraisal_value cadastral_value in
let adjusted_tariff := basic_rate * reducing_coefficient * increasing_coefficient in
insured_amount * adjusted_tariff

theorem insurance_premium_correct :
  compute_insurance_premium 20000000 14500000 15000000 0.002 0.8 1.3 = 31200 :=
by
  sorry

end insurance_premium_correct_l363_363344


namespace cube_root_neg_27_l363_363734

theorem cube_root_neg_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 :=
by {
    use -3,
    split,
    { norm_num },
    { refl }
}

end cube_root_neg_27_l363_363734


namespace construct_point_O_exists_l363_363685

noncomputable def angle (A B C : Point) : ℝ := sorry

theorem construct_point_O_exists
  (A B C : Point)
  (D E F : ℝ)
  (h1 : D < 180)
  (h2 : E < 180)
  (h3 : F < 180)
  (h_sum : D + E + F = 360) :
  ∃ O : Point, angle A O B = D ∧ angle B O C = E ∧ angle C O A = F :=
begin
  sorry
end

end construct_point_O_exists_l363_363685


namespace smallest_subset_X_cardinality_l363_363380

-- Define the set of all two-digit numbers
def two_digit_numbers : set (ℕ × ℕ) := { (a, b) | a < 10 ∧ b < 10 }

-- Define a subset X of two-digit numbers
def subset_X (X : set (ℕ × ℕ)) := 
  ∀ (seq : ℕ → ℕ), (∀ n, seq n < 10) → ∃ n, (seq n, seq (n+1)) ∈ X

-- The main theorem
theorem smallest_subset_X_cardinality : 
  ∃ (X : set (ℕ × ℕ)), (subset_X X ∧ X.card = 55) := 
by
  -- Here we'd provide the proof
  sorry

end smallest_subset_X_cardinality_l363_363380


namespace simplify_fraction_l363_363711

open Real

theorem simplify_fraction : 
  (60 * (π / 180) = real.pi / 3) →
  (∀ x, tan x = sin x / cos x) →
  (∀ x, cot x = 1 / tan x) →
  let t := tan (real.pi / 3)
  let c := cot (real.pi / 3)
  t = sqrt 3 →
  c = 1 / sqrt 3 →
  (t ^ 3 + c ^ 3) / (t + c) = 7 / 3 :=
by
  intro h60 htan hcot t_def c_def
  sorry

end simplify_fraction_l363_363711


namespace probability_of_hitting_blue_zone_l363_363813

theorem probability_of_hitting_blue_zone :
  let P_red := (2 : ℚ) / 5
  let P_green := (1 : ℚ) / 4
  ∃ P_blue : ℚ, P_blue = 1 - (P_red + P_green) ∧ P_blue = 7 / 20 :=
by
  let P_red := (2 : ℚ) / 5
  let P_green := (1 : ℚ) / 4
  let P_blue := 1 - (P_red + P_green)
  use P_blue
  split
  · exact rfl
  · sorry

end probability_of_hitting_blue_zone_l363_363813


namespace percentage_of_women_who_do_not_speak_french_is_correct_l363_363616

noncomputable def total_employees : ℕ := 100

-- Conditions
def percent_men : ℝ := 0.45
def men_who_speak_french_percent : ℝ := 0.60
def french_speakers_percent : ℝ := 0.40

-- Definitions derived from conditions
def total_men := total_employees * percent_men
def total_french_speakers := total_employees * french_speakers_percent
def men_who_speak_french := total_men * men_who_speak_french_percent
def total_women := total_employees - total_men
def women_who_speak_french := total_french_speakers - men_who_speak_french
def women_who_do_not_speak_french := total_women - women_who_speak_french
def percent_women_who_do_not_speak_french := (women_who_do_not_speak_french / total_women) * 100

-- Goal
theorem percentage_of_women_who_do_not_speak_french_is_correct : 
  percent_women_who_do_not_speak_french = 76.36 := 
by {
  sorry
}

end percentage_of_women_who_do_not_speak_french_is_correct_l363_363616


namespace sequence_integers_l363_363669

theorem sequence_integers (a : ℕ) (m : ℕ) (a1 : ℕ) (h_m : m = 2) (h_a1 : a1 > 0) :
  (∀ n : ℕ, ∃ a_n : ℕ, 
    (λ a_n, ((a_n < 2^m) → (a_n^2 + 2^m)) ∧ (a_n ≥ 2^m → a_n / 2)) = a_n) :=
begin
  sorry
end

example (m : ℕ) (a1 : ℕ) (hm : m = 2) :
  ∃ n : ℕ, ∃ a : ℕ, (a = 2^n ∧ n > 0) :=
begin
  use [1, 2],
  dsimp,
  rw [hm],
  split,
  {
    apply nat.pow_zero,
  },
  {
    linarith,
  }
end

end sequence_integers_l363_363669


namespace quadratic_has_real_root_l363_363564

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l363_363564


namespace insurance_calculation_l363_363349

-- Define the conditions
def baseRate : ℝ := 0.002 -- 0.2% as a decimal
def reducingCoefficient : ℝ := 0.8
def increasingCoefficient : ℝ := 1.3
def assessedValue : ℝ := 14500000
def cadasterValue : ℝ := 15000000
def loanAmount : ℝ := 20000000

-- Define the adjusted tariff
def adjustedTariff : ℝ := baseRate * reducingCoefficient * increasingCoefficient

-- Define the insurable amount
def insurableAmount : ℝ := max assessedValue cadasterValue

-- Define the proof problem to show equality
theorem insurance_calculation :
  adjustedTariff = 0.00208 ∧ insurableAmount * adjustedTariff = 31200 := by
  sorry

end insurance_calculation_l363_363349


namespace bus_stops_12_minutes_per_hour_l363_363024

noncomputable def stopping_time (speed_excluding_stoppages : ℝ) (speed_including_stoppages : ℝ) : ℝ :=
  let distance_lost_per_hour := speed_excluding_stoppages - speed_including_stoppages
  let speed_per_minute := speed_excluding_stoppages / 60
  distance_lost_per_hour / speed_per_minute

theorem bus_stops_12_minutes_per_hour :
  stopping_time 50 40 = 12 :=
by
  sorry

end bus_stops_12_minutes_per_hour_l363_363024


namespace discriminant_of_quadratic_5x2_minus_2x_minus_7_l363_363769

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

theorem discriminant_of_quadratic_5x2_minus_2x_minus_7 :
  quadratic_discriminant 5 (-2) (-7) = 144 :=
by
  sorry

end discriminant_of_quadratic_5x2_minus_2x_minus_7_l363_363769


namespace ind_test_l363_363624

def null_hypothesis (X Y : Type) : Prop := ¬ (X = Y)

def prob_K2 (K2 : ℝ) : Prop := K2 ≥ 6.635

theorem ind_test
  (X Y : Type)
  (H0 : null_hypothesis X Y)
  (PK2 : ∀ K2, prob_K2 K2 → P(K2) ≈ 0.010) :
  P(X related_to Y) ≈ 0.99 :=
sorry

end ind_test_l363_363624


namespace minimal_sum_roots_qq_three_solutions_and_tildeq2_is_9_l363_363002

noncomputable def q (a b x : ℝ) : ℝ := x^2 - (a + b) * x + a * b

lemma discriminant_eq_zero (a b : ℝ) :
  (a + b)^2 - 4 * (a * b - b) = 0 → q(q 0 0) = 0 := sorry

theorem minimal_sum_roots_qq_three_solutions_and_tildeq2_is_9 :
  (∀ a b : ℝ, q(q x (x^2 - (a + b) * x + a * b)) = 0 → 
  (a = -1 ∧ b = -1)) →
  q (-1) (-1) 2 = 9 :=
begin
  intro h,
  sorry
end

end minimal_sum_roots_qq_three_solutions_and_tildeq2_is_9_l363_363002


namespace dot_product_of_vectors_l363_363969

variable (a b : EuclideanSpace ℝ (Fin 2))

def magnitude (v : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  Real.sqrt (v.inner v)

theorem dot_product_of_vectors 
  (h_a : magnitude a = 2) 
  (h_b : magnitude b = 1 / 2) 
  (angle_ab : real.angle.ofReal (a.angle b) = real.angle.ofReal (π / 3)) : 
  (a.inner b) = 1 / 2 := by 
  sorry

end dot_product_of_vectors_l363_363969


namespace sum_first_10_terms_eq_85_l363_363949

-- Definitions for the problem
def is_arithmetic_seq (u : ℕ → ℕ) (d : ℕ) := ∀ n : ℕ, u (n+1) = u n + d

noncomputable def a : ℕ → ℕ := λ n, a_1 + (n - 1)
noncomputable def b : ℕ → ℕ := λ n, b_1 + (n - 1)

-- Conditions
variable (a_1 b_1 : ℕ) (h_a1_b1_sum : a_1 + b_1 = 5) (h_a1_pos : a_1 > 0) (h_b1_pos : b_1 > 0)
variable (h_a_seq : is_arithmetic_seq a 1) (h_b_seq : is_arithmetic_seq b 1)

-- Main theorem
theorem sum_first_10_terms_eq_85 :
  (Σ i in finset.range 10, (a i * b i)) = 85 :=
sorry

end sum_first_10_terms_eq_85_l363_363949


namespace number_of_sets_without_perfect_squares_l363_363162

-- Define the set T_i
def T (i : ℕ) : Set ℕ := { n : ℕ | 200 * i ≤ n ∧ n < 200 * (i + 1) }

-- Define the property of the set containing no perfect squares
def no_perfect_square (s : Set ℕ) : Prop :=
  ∀ n ∈ s, ∀ k : ℕ, k * k ≠ n

-- State the main theorem
theorem number_of_sets_without_perfect_squares : 
  (Finset.range 500).filter (λ i, no_perfect_square (T i)).card = 378 :=
by
  sorry

end number_of_sets_without_perfect_squares_l363_363162


namespace problem_m_eq_3_l363_363930

-- conditions
def arithmetic_mean (a b : ℝ) := (a + b) / 2

-- given problem
theorem problem_m_eq_3 (m : ℝ) (h : m = arithmetic_mean 1 5) : m = 3 :=
by
  -- proof goes here
  sorry

end problem_m_eq_3_l363_363930


namespace total_cost_of_tires_and_battery_l363_363451

theorem total_cost_of_tires_and_battery :
  (4 * 42 + 56 = 224) := 
  by
    sorry

end total_cost_of_tires_and_battery_l363_363451


namespace number_that_multiplies_b_l363_363109

theorem number_that_multiplies_b (a b x : ℝ) (h0 : 4 * a = x * b) (h1 : a * b ≠ 0) (h2 : (a / 5) / (b / 4) = 1) : x = 5 :=
by
  sorry

end number_that_multiplies_b_l363_363109


namespace items_counted_l363_363364

def convert_counter (n : Nat) : Nat := sorry

theorem items_counted
  (counter_reading : Nat) 
  (condition_1 : ∀ d, d ∈ [5, 6, 7] → ¬(d ∈ [0, 1, 2, 3, 4, 8, 9]))
  (condition_2 : ∀ d1 d2, d1 = 4 → d2 = 8 → ¬(d2 = 5 ∨ d2 = 6 ∨ d2 = 7)) :
  convert_counter 388 = 151 :=
sorry

end items_counted_l363_363364


namespace triangle_perimeter_l363_363243

theorem triangle_perimeter:
  ∀ (x : ℝ), x^2 - 8 * x + 12 = 0 → (4 + 7 + x) = 17 ∧
  (x = 2 ∨ x = 6) ∧ (x ≠ 2) :=
begin
  sorry
end

end triangle_perimeter_l363_363243


namespace ex_sq_sum_l363_363529

theorem ex_sq_sum (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = -1) : x^2 + y^2 = 11 :=
by
  sorry

end ex_sq_sum_l363_363529


namespace quadratic_real_roots_l363_363572

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l363_363572


namespace product_of_segments_l363_363410

noncomputable def Circle := ℝ → ℝ → Prop

variables (O A B C D N Q : ℝ) (r : ℝ)
variable (chord_AN : ℝ → Prop)

-- Definition of the circle with center O and radius r
def is_diameter (x y : ℝ) := (x = 0 ∧ y = 2 * r) ∨ (x = 2 * r ∧ y = 0)

-- Conditions
axiom diameters_perpendicular (h : ℝ) :
  (is_diameter A B) ∧ (is_diameter C D) ∧ 
  (∃ r, Q = O * r ∧ A = O ∧ B = O ∧ C = O ∧ D = O ∧ AN = Q * r)

-- Prove the required product
theorem product_of_segments (AQ AN AO AB : ℝ) :
  AQ * AN = AO * AB :=
by {
  sorry
}

end product_of_segments_l363_363410


namespace remainder_mod56_l363_363687

theorem remainder_mod56 (N : ℕ) (h : N % 8 = 5) : ∃ r, r ∈ {5, 13, 21, 29, 37, 45, 53} ∧ N % 56 = r :=
by
  sorry

end remainder_mod56_l363_363687


namespace quadratic_real_roots_l363_363559

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l363_363559


namespace billy_raspberry_juice_billy_raspberry_juice_quarts_l363_363403

theorem billy_raspberry_juice (V : ℚ) (h : V / 12 + 1 = 3) : V = 24 :=
by sorry

theorem billy_raspberry_juice_quarts (V : ℚ) (h : V / 12 + 1 = 3) : V / 4 = 6 :=
by sorry

end billy_raspberry_juice_billy_raspberry_juice_quarts_l363_363403


namespace minimum_value_of_dot_product_l363_363248

noncomputable def parabola_equation (p : ℝ) : ℝ → ℝ → Prop :=
  λ x y, y^2 = 2 * p * x

def minimization_problem (p : ℝ) (P : ℝ × ℝ) (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop :=
  let (Px, Py) := P,
      (Mx, My) := M,
      (Fx, Fy) := F in
    P = (Px, Py) ∧
    parabola_equation 2 P.1 P.2 ∧
    M = (2, 0) ∧
    P.1^2 + P.1 + 2 = 2 -- equivalence to the minimum condition

theorem minimum_value_of_dot_product :
  ∀ (p : ℝ), p > 0 →
    ∃ P M F, minimization_problem p P M F → 
    (-2 ≤ 0 ∧ P.1^2 + P.1 + 2 = 2) := 
sorry

end minimum_value_of_dot_product_l363_363248


namespace volume_of_larger_cube_l363_363819

-- Definitions based on conditions from part (a)
def largerCubeVolume (V : ℝ) : Prop :=
  (∃ s : ℝ, s^3 = V) ∧
  (∑₁ (1 : ℝ)) = 216

def totalSurfaceAreaDifference (V : ℝ) (D : ℝ) : Prop :=
  (∑₁ (6 : ℝ)) - 6 * V^(2/3) = D

-- The theorem stating the volume of the larger cube is 216 cubic inches given the conditions
theorem volume_of_larger_cube :
  (∑₁ (1 : ℝ)) = 216 →
  totalSurfaceAreaDifference 216 1080 →
  (∃ V : ℝ, V = 216) := 
sorry

end volume_of_larger_cube_l363_363819


namespace find_integer_pairs_l363_363790

theorem find_integer_pairs (x y : ℤ) :
  x^4 + (y+2)^3 = (x+2)^4 ↔ (x, y) = (0, 0) ∨ (x, y) = (-1, -2) := sorry

end find_integer_pairs_l363_363790


namespace original_price_of_shirt_l363_363154

theorem original_price_of_shirt (P : ℝ) (h : 0.5625 * P = 18) : P = 32 := 
by 
sorry

end original_price_of_shirt_l363_363154


namespace sandy_nickels_remaining_l363_363201

def original_nickels : ℕ := 31
def nickels_borrowed : ℕ := 20

theorem sandy_nickels_remaining : (original_nickels - nickels_borrowed) = 11 :=
by
  sorry

end sandy_nickels_remaining_l363_363201


namespace computer_price_l363_363814

variable (produce_per_day : ℕ) (days_per_week : ℕ) (total_earnings_week : ℕ) (weekly_production : ℕ) (price_per_computer : ℝ)

-- Conditions
def condition_1 : produce_per_day = 1500 := by sorry
def condition_2 : days_per_week = 7 := by sorry
def condition_3 : total_earnings_week = 1575000 := by sorry
def condition_4 : weekly_production = produce_per_day * days_per_week := by sorry

-- Question: Prove the price per computer
theorem computer_price :
  price_per_computer = total_earnings_week / weekly_production :=
by
  rw [condition_1, condition_2, condition_3, condition_4]
  sorry

end computer_price_l363_363814


namespace moles_of_CH4_needed_l363_363438

theorem moles_of_CH4_needed
  (moles_C6H6_needed : ℕ)
  (reaction_balance : ∀ (C6H6 CH4 C6H5CH3 H2 : ℕ), 
    C6H6 + CH4 = C6H5CH3 + H2 → C6H6 = 1 ∧ CH4 = 1 ∧ C6H5CH3 = 1 ∧ H2 = 1)
  (H : moles_C6H6_needed = 3) :
  (3 : ℕ) = 3 :=
by 
  -- The actual proof would go here
  sorry

end moles_of_CH4_needed_l363_363438


namespace midpoints_of_triangle_l363_363802

variables {A B C P M K : Type} [AddGroup A] [AddGroup B] [AddGroup C]
  [AddGroup P] [AddGroup M] [AddGroup K]
  [AddAdditive Vect
    ( AM : P → M) ( BK : A → K) ( CP : C → P)] 

/--
Given a triangle ABC with vertices A, B, and C, and points P, M, and K lying on sides 
AB, BC, and AC respectively, if the lines AM, BK, and CP are concurrent and 
\( \overrightarrow{AM} + \overrightarrow{BK} + \overrightarrow{CP} = 0 \), 
then prove that points P, M, and K are the midpoints of the sides of the triangle.
-/
theorem midpoints_of_triangle
  (P_lies_on_AB : P ∈ AB)
  (M_lies_on_BC : M ∈ BC)
  (K_lies_on_AC : K ∈ AC)
  (concurrent : ∃ O, intersect (AM O) (BK O) (CP O))
  (vector_sum_zero : AdditiveGroup.Add 
    (AM P) (AdditiveGroup.Add (BK M) (CP K)) = 0) 
  : midpoint P AB ∧ midpoint M BC ∧ midpoint K AC :=
sorry

end midpoints_of_triangle_l363_363802


namespace moores_law_transistors_2010_l363_363681

theorem moores_law_transistors_2010 :
  ∀ (initial_transistors : ℕ) (start_year : ℕ) (end_year : ℕ),
    initial_transistors = 2000000 →
    start_year = 1995 →
    end_year = 2010 →
    let years_passed := end_year - start_year,
        doubling_period := 2,
        doublings := years_passed / doubling_period,
        final_transistors := initial_transistors * 2^doublings
    in final_transistors = 256000000 := 
by
  intros initial_transistors start_year end_year h₁ h₂ h₃
  let years_passed := end_year - start_year
  let doubling_period := 2
  let doublings := years_passed / doubling_period 
  let final_transistors := initial_transistors * 2^doublings 
  have h4 : years_passed = 15 := by rw [h₂, h₃]; norm_num
  have h5 : doublings = 7 := by rw [h4]; norm_num 
  have h6 : final_transistors = 2000000 * 128 := by { rw [h₁, h5], norm_num }
  have h7 : 2000000 * 128 = 256000000 := by norm_num
  rw [h6]
  rw [h7]
  exact rfl

end moores_law_transistors_2010_l363_363681


namespace odd_three_digit_numbers_with_sum_of_tens_units_12_eq_36_l363_363101

open Finset

noncomputable def count_odd_three_digit_numbers_with_sum_of_tens_units_12 : Nat :=
  let units := filter (λ u, u % 2 = 1) (range 10) -- odd units digits
  let tens_units := filter (λ tu, tu.1 + tu.2 = 12)
                          (product (range 10) units) -- valid tens and units pairs
  let hundreds := range 9 -- valid values for hundreds place (1 to 9)
  (card hundreds) * (card tens_units)

theorem odd_three_digit_numbers_with_sum_of_tens_units_12_eq_36 :
  count_odd_three_digit_numbers_with_sum_of_tens_units_12 = 36 := by
  sorry

end odd_three_digit_numbers_with_sum_of_tens_units_12_eq_36_l363_363101


namespace sum_of_cubes_of_integers_l363_363252

theorem sum_of_cubes_of_integers (n: ℕ) (h1: (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 = 8830) : 
  (n-1)^3 + n^3 + (n+1)^3 + (n+2)^3 = 52264 :=
by
  sorry

end sum_of_cubes_of_integers_l363_363252


namespace sodium_acetate_production_l363_363525

def acetic_acid := ℝ
def sodium_hydroxide := ℝ
def sodium_acetate := ℝ
def water := ℝ

noncomputable def moles_of_sodium_acetate_formed (acetic_acid : ℝ) (sodium_hydroxide : ℝ) : sodium_acetate :=
  if acetic_acid = 2 ∧ sodium_hydroxide = 2 then 2 else 0

theorem sodium_acetate_production :
  moles_of_sodium_acetate_formed 2 2 = 2 :=
by
  sorry

end sodium_acetate_production_l363_363525


namespace point_on_graph_l363_363234

theorem point_on_graph (x y : ℝ) (h : y = 3 * x + 1) : (x, y) = (2, 7) :=
sorry

end point_on_graph_l363_363234


namespace vector_dot_product_AC_BE_l363_363911

def vector (ℝ → (ℝ × ℝ)) -- a simple type definition for 2D vectors

constant A B C D E : vector
-- Defining basic properties given in conditions
constant side_length : ℝ
constant AB_is_x_axis AD_is_y_axis : Prop
constant midpoint_CD : Prop
constant square : Prop

axiom square_length : side_length = 2

-- Defining assumptions
axiom ABCD_is_square : square
axiom E_is_midpoint : midpoint_CD

-- Dot product definition
def dot_product (v1 v2: vector) : ℝ := sorry -- placeholder to make code build successfully

-- Defining vectors as per Cartesian coordinates
def AC: vector := (2,2)
def BE: vector := (-1,2)

-- Main theorem to prove the question == correct answer
theorem vector_dot_product_AC_BE :
  (dot_product AC BE) = 2 :=
sorry

end vector_dot_product_AC_BE_l363_363911


namespace trig_identity_proof_l363_363199

noncomputable def problem_statement (α : ℝ) : Prop :=
  (cos ((π/2) - (α/4)) - sin ((π/2) - (α/4)) * tan (α/8)) / 
  (sin ((7/2) * π - (α/4)) + sin ((α/4) - 3 * π) * tan (α/8)) = -tan (α/8)

theorem trig_identity_proof (α : ℝ) : problem_statement α :=
  sorry

end trig_identity_proof_l363_363199


namespace train_meeting_distance_l363_363298

theorem train_meeting_distance :
  ∀ (speed_X speed_Y distance total_speed time meet_distance : ℝ),
  speed_X = 180 / 5 →
  speed_Y = 180 / 4 →
  distance = 180 →
  total_speed = speed_X + speed_Y →
  time = distance / total_speed →
  meet_distance = time * speed_X →
  meet_distance = 79.92 :=
by
  intros speed_X speed_Y distance total_speed time meet_distance h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  exact h6


end train_meeting_distance_l363_363298


namespace max_value_fraction_on_circle_l363_363491

theorem max_value_fraction_on_circle : 
  ∀ (x y : ℝ), x^2 + y^2 = 1 → ∃ M : ℝ, ∀ (k : ℝ), (y / (x + 2) = k → k ≤ M) :=
begin
  sorry
end

end max_value_fraction_on_circle_l363_363491


namespace find_a_f_odd_f_increasing_l363_363497

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - a / x

theorem find_a : (f 1 a = 3) → (a = -1) :=
by
  sorry

noncomputable def f_1 (x : ℝ) : ℝ := 2 * x + 1 / x

theorem f_odd : ∀ x : ℝ, f_1 (-x) = -f_1 x :=
by
  sorry

theorem f_increasing : ∀ x1 x2 : ℝ, (x1 > 1) → (x2 > 1) → (x1 > x2) → (f_1 x1 > f_1 x2) :=
by
  sorry

end find_a_f_odd_f_increasing_l363_363497


namespace quadratic_real_roots_l363_363574

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l363_363574


namespace cos_double_angle_l363_363804
open Real

theorem cos_double_angle (α : ℝ) (h : tan (α - π / 4) = 2) : cos (2 * α) = -4 / 5 := 
sorry

end cos_double_angle_l363_363804


namespace log_10_50_between_consecutive_integers_l363_363752

theorem log_10_50_between_consecutive_integers :
  (∃ c d : ℤ, c < log 50 / log 10 ∧ log 50 / log 10 < d ∧ (d = c + 1) ∧ (c + d = 3)) :=
by
  sorry

end log_10_50_between_consecutive_integers_l363_363752


namespace no_vertical_symmetry_max_value_l363_363750

def f (x : ℝ) : ℝ := Real.exp (-x)

theorem no_vertical_symmetry :
  ¬ ∃ (a : ℝ), ∀ (x : ℝ), f (a - x) = f (a + x) :=
by
  sorry

theorem max_value : ∃ x : ℝ, (∀ y : ℝ, f y ≤ f x) ∧ f x = 1 :=
by
  let x_max := 0
  use x_max
  split
  · intros y
    calc
      f y = Real.exp (-y) : rfl
         ... ≤ Real.exp 0 : by sorry
         ... = 1 : by sorry
  · calc
      f x_max = f 0 : rfl
           ... = Real.exp 0 : by rfl
           ... = 1 : by sorry

end no_vertical_symmetry_max_value_l363_363750


namespace find_focal_radius_l363_363075

-- Definitions for the problem setup
def is_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  let (x, y) := A in y^2 = 2 * p * x

def focal_radius (A F : ℝ × ℝ) : ℝ :=
  let (xA, yA) := A in xA + (yA^2) / (2 * xA)

-- The proof problem statement
theorem find_focal_radius (A F : ℝ × ℝ) (p : ℝ)
  (hA : A = (1, 2))
  (h_parabola : is_on_parabola A p) :
  focal_radius A F = 2 :=
by
  sorry

end find_focal_radius_l363_363075


namespace find_k_l363_363950

theorem find_k (k : ℝ) : 
  let a := (3 : ℝ, 1 : ℝ)
  let b := (1 : ℝ, 3 : ℝ)
  let c := (k, 7 : ℝ)
  (a.1 - c.1) / b.1 = (a.2 - c.2) / b.2 ↔ k = 5 := 
by {
  -- Definitions for vectors
  let a := (3 : ℝ, 1 : ℝ)
  let b := (1 : ℝ, 3 : ℝ)
  let c := (k, 7 : ℝ)

  -- Given condition
  have h_parallel := (a.1 - c.1) / b.1 = (a.2 - c.2) / b.2
  
  -- Prove k = 5
  sorry
}

end find_k_l363_363950


namespace sin_X_value_l363_363730

theorem sin_X_value (a b X : ℝ) (h₁ : (1/2) * a * b * Real.sin X = 72) (h₂ : Real.sqrt (a * b) = 16) :
  Real.sin X = 9 / 16 := by
  sorry

end sin_X_value_l363_363730


namespace collinear_vectors_l363_363521

variables {R : Type*} [LinearOrderedField R]
variables (e1 e2 : R) (λ : R)
variables (a b : R → R) 

theorem collinear_vectors (h1 : e1 ≠ 0) 
(hλ : λ ∈ set.univ) 
(ha : a = e1 + λ * e2) 
(hb : b = 2 * e1) 
(h_collinear : ∃ k : R, a = k * b) : 
(e1 = 0 ∨ λ = 0) :=
sorry

end collinear_vectors_l363_363521


namespace range_of_eccentricity_l363_363475

variables (a b c : ℝ) (e : ℝ)
variables (h1 : a > b) (h2 : b > c) (h3 : c > 0)
-- Condition of the ellipse
variables (hx : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
-- Minimum length condition
variables (hPT : ∃ P T : (ℝ × ℝ), P ∈ { (x, y) | x^2 / a^2 + y^2 / b^2 = 1 } ∧ dist P T = (√3) / 2 * (a - c))

theorem range_of_eccentricity :
  ∃ e : ℝ, c = e * a ∧ (3 / 5 ≤ e ∧ e < sqrt 2 / 2) := 
sorry

end range_of_eccentricity_l363_363475


namespace select_n_plus_2_numbers_l363_363705

theorem select_n_plus_2_numbers (n : ℕ) (h : 0 < n) (s : finset ℕ) (hs : s.card = n+2)
  (h_subset : s ⊆ finset.range (3*n+1)) : 
  ∃ x y ∈ s, n < abs (x - y) ∧ abs (x - y) < 2*n :=
by 
  sorry

end select_n_plus_2_numbers_l363_363705


namespace relationship_xy_qz_l363_363104

theorem relationship_xy_qz
  (a c b d : ℝ)
  (x y q z : ℝ)
  (h1 : a^(2 * x) = c^(2 * q) ∧ c^(2 * q) = b^2)
  (h2 : c^(3 * y) = a^(3 * z) ∧ a^(3 * z) = d^2) :
  x * y = q * z :=
by
  sorry

end relationship_xy_qz_l363_363104


namespace equilateral_implies_isosceles_converse_not_true_inverse_not_true_neither_true_l363_363945

def is_equilateral (t : Triangle) : Prop := 
  t.a = t.b ∧ t.b = t.c

def is_isosceles (t : Triangle) : Prop := 
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

theorem equilateral_implies_isosceles (t : Triangle) : 
  is_equilateral t → is_isosceles t :=
begin
  intro h,
  cases h,
  left,
  exact h.left,
end

theorem converse_not_true (t : Triangle) : 
  ¬ (is_isosceles t → is_equilateral t) :=
begin
  intro h,
  let t := ⟨1, 1, 2⟩,
  unfold is_equilateral at *,
  unfold is_isosceles at *,
  have h1 : t.a = t.b := by simp,
  have h2 : ¬ (t.a = t.b ∧ t.b = t.c) := by simp,
  specialize h h1,
  contradiction,
end

theorem inverse_not_true (t : Triangle) :
  ¬ (¬ is_equilateral t → ¬ is_isosceles t) :=
begin
  intro h,
  let t := ⟨1, 1, 2⟩,
  unfold is_equilateral at *,
  unfold is_isosceles at *,
  have h1 : t.a = t.b := by simp,
  have h2 : t.a ≠ t.b ∨ t.b ≠ t.c := by simp,
  specialize h h2,
  contradiction,
end

theorem neither_true :
  ¬ (∀ t : Triangle, is_isosceles t → is_equilateral t) ∧
  ¬ (∀ t : Triangle, ¬ is_equilateral t → ¬ is_isosceles t) :=
begin
  split,
  apply converse_not_true,
  apply inverse_not_true,
end

end equilateral_implies_isosceles_converse_not_true_inverse_not_true_neither_true_l363_363945


namespace troy_needs_more_money_l363_363283

theorem troy_needs_more_money (initial_savings : ℕ) (sold_computer : ℕ) (new_computer_cost : ℕ) :
  initial_savings = 50 → sold_computer = 20 → new_computer_cost = 80 → 
  new_computer_cost - (initial_savings + sold_computer) = 10 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end troy_needs_more_money_l363_363283


namespace total_distance_of_journey_l363_363328

-- Define the parameters and conditions given in the problem
variables (D : ℝ)

-- First half journey speed and second half journey speed in km/hr
def speed_first_half : ℝ := 21
def speed_second_half : ℝ := 24

-- Time to complete the journey in hours
def total_time : ℝ := 15

-- Time taken for the first half of the journey
def time_first_half : ℝ := (D / 2) / speed_first_half

-- Time taken for the second half of the journey
def time_second_half : ℝ := (D / 2) / speed_second_half

-- Total time is the sum of times for each half and it is given as 15 hours
def travel_time_condition : Prop := total_time = time_first_half + time_second_half

-- Prove that the total distance of the journey is 336 km given the conditions
theorem total_distance_of_journey
  (h : travel_time_condition D total_time speed_first_half speed_second_half) :
  D = 336 :=
sorry

end total_distance_of_journey_l363_363328


namespace quadratic_real_root_iff_b_range_l363_363581

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l363_363581


namespace shaded_area_quadrilateral_l363_363739

theorem shaded_area_quadrilateral :
  let large_square_area := 11 * 11
  let small_square_area_1 := 1 * 1
  let small_square_area_2 := 2 * 2
  let small_square_area_3 := 3 * 3
  let small_square_area_4 := 4 * 4
  let other_non_shaded_areas := 12 + 15 + 14
  let total_non_shaded := small_square_area_1 + small_square_area_2 + small_square_area_3 + small_square_area_4 + other_non_shaded_areas
  let shaded_area := large_square_area - total_non_shaded
  shaded_area = 35 := by
  sorry

end shaded_area_quadrilateral_l363_363739


namespace possible_values_xy_l363_363138

-- Given the coordinates of point E
variables {x y : ℝ}

-- Definitions of congruence conditions
def congruent_triangles (A B C D E : ℝ × ℝ) : Prop :=
  let AB := real.dist A B in
  let AC := real.dist A C in
  let BC := real.dist B C in
  let AD := real.dist A D in
  let AE := real.dist A E in
  let DE := real.dist D E in
  AB = AD ∧ AC = AE ∧ BC = DE

-- Assume \(x\) and \(y\) are the coordinates of point E
def E := (x, y)

-- Main statement to prove
theorem possible_values_xy (A B C D : ℝ × ℝ) (H : congruent_triangles A B C D E) : x * y = 14 ∨ x * y = 18 ∨ x * y = 40 :=
  sorry

end possible_values_xy_l363_363138


namespace new_avg_weight_is_79_l363_363732

variable (A B C D E : ℝ)

def avg_weight_abc (A B C : ℝ) : ℝ := (A + B + C) / 3
def total_weight_abc (A B C : ℝ) : ℝ := 84 * 3

def avg_weight_abcd (A B C D : ℝ) : ℝ := (A + B + C + D) / 4
def total_weight_abcd (A B C D : ℝ) : ℝ := 80 * 4

def weight_d (A B C D : ℝ) : ℝ := total_weight_abcd A B C D - total_weight_abc A B C

def weight_e (D : ℝ) : ℝ := D + 6

def total_weight_bcde (B C D E : ℝ) (A : ℝ) : ℝ := total_weight_abcd A B C D - A + E

def new_avg_weight_bcde (B C D E : ℝ) (A : ℝ) : ℝ := total_weight_bcde B C D E A / 4

theorem new_avg_weight_is_79 (A B C D E : ℝ) (h1 : avg_weight_abc A B C = 84)
                                (h2 : avg_weight_abcd A B C D = 80)
                                (h3 : E = weight_e D)
                                (h4 : A = 78) :
                                new_avg_weight_bcde B C D E A = 79 :=
by
  sorry

end new_avg_weight_is_79_l363_363732


namespace frequency_correct_l363_363056

-- Define the frequency function and intervals
def intervals : List (Real × Real) := [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70)]
def frequencies : List Nat := [2, 3, 4, 5, 4, 2]

-- Given sample size
def sample_size : Nat := 20

-- Frequency calculation for interval (15, 50]
def frequency_interval_15_50 : Real :=
  let partial_freq := 1 -- (15, 20] is half of (10, 20] with frequency 2, so 2 * 1/2 = 1
  let freq_sum := partial_freq + frequencies[1] + frequencies[2] + frequencies[3]
  freq_sum / sample_size

theorem frequency_correct : frequency_interval_15_50 = 0.65 :=
  by
    -- The proof should be provided here
    sorry

end frequency_correct_l363_363056


namespace find_b_values_l363_363549

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l363_363549


namespace probability_of_four_ones_is_correct_l363_363423

-- Define the problem
def eight_sided_dice (n : ℕ) := (1 <= n ∧ n <= 8)

-- Helper function to compute binomial coefficients
def binom : ℕ → ℕ → ℕ
| 0, 0 => 1
| 0, _ => 0
| _, 0 => 1
| n, k => binom (n - 1) k + binom (n - 1) (k - 1)

noncomputable def probability_exactly_four_ones : ℕ :=
let num_ways := binom 8 4 in
let prob_one := (1/8 : ℝ) in
let prob_not_one := (7/8 : ℝ) in
(num_ways * prob_one^4 * prob_not_one^4 * (16777216 / 16777216)).toInt

theorem probability_of_four_ones_is_correct :
  probability_exactly_four_ones = Int.ofNat 0.010 :=
sorry

end probability_of_four_ones_is_correct_l363_363423


namespace smallest_m_to_reach_zero_in_7_steps_l363_363888

def next_term (n : ℕ) : ℕ := n - (Int.toNat (Real.sqrt n).floor)^2

def sequence (m : ℕ) : List ℕ :=
  if m = 0 then [0]
  else List.unfoldr (λ n => if n = 0 then none else some (n, next_term n)) m

def sequence_length (m : ℕ) : ℕ := (sequence m).length

theorem smallest_m_to_reach_zero_in_7_steps (m : ℕ) :
  m > 0 ∧ (sequence_length m = 7) → m = 7 := sorry

end smallest_m_to_reach_zero_in_7_steps_l363_363888


namespace right_triangle_side_length_l363_363987

theorem right_triangle_side_length (a c b : ℕ) (h₁ : a = 6) (h₂ : c = 10) (h₃ : c * c = a * a + b * b) : b = 8 :=
by {
  sorry
}

end right_triangle_side_length_l363_363987


namespace percent_absent_correct_l363_363001

-- Define the initially given conditions
def total_students := 180
def boys := 100
def girls := 80
def fraction_absent_boys := 1/5
def fraction_absent_girls := 1/4

-- Calculate the number of absent boys
def absent_boys := fraction_absent_boys * boys

-- Calculate the number of absent girls
def absent_girls := fraction_absent_girls * girls

-- Calculate the total number of absent students
def total_absent_students := absent_boys + absent_girls

-- Calculate the percentage of absent students
def percent_absent := (total_absent_students / total_students) * 100

-- State the theorem to prove
theorem percent_absent_correct : percent_absent = 22.22 := by
  sorry

end percent_absent_correct_l363_363001


namespace arithmetic_sequence_squares_l363_363707

theorem arithmetic_sequence_squares (a b c : ℝ) :
  (1 / (a + b) - 1 / (b + c) = 1 / (c + a) - 1 / (b + c)) →
  (2 * b^2 = a^2 + c^2) :=
by
  intro h
  sorry

end arithmetic_sequence_squares_l363_363707


namespace hypotenuse_length_l363_363629

theorem hypotenuse_length (AB AC : ℝ) (right_triangle : (AB = 6) ∧ (AC = 8)) :
  (∃ BC : ℝ, BC = 8) ∨ (∃ BC : ℝ, BC = Real.sqrt (AC^2 + AB^2) ∧ BC = 10) := 
by
  have h1 : AB = 6 := right_triangle.1
  have h2 : AC = 8 := right_triangle.2
  use 8
  left
  refl
  right
  use (Real.sqrt (8^2 + 6^2))
  sorry

end hypotenuse_length_l363_363629


namespace sin_of_angle_F_l363_363630

theorem sin_of_angle_F 
  (DE EF DF : ℝ) 
  (h : DE = 12) 
  (h0 : EF = 20) 
  (h1 : DF = Real.sqrt (DE^2 + EF^2)) : 
  Real.sin (Real.arctan (DF / EF)) = 12 / Real.sqrt (DE^2 + EF^2) := 
by 
  sorry

end sin_of_angle_F_l363_363630


namespace simplify_trig_expr_l363_363708

theorem simplify_trig_expr : 
  let θ := 60
  let tan_θ := Real.sqrt 3
  let cot_θ := (Real.sqrt 3)⁻¹
  (tan_θ^3 + cot_θ^3) / (tan_θ + cot_θ) = 7 / 3 :=
by
  sorry

end simplify_trig_expr_l363_363708


namespace troy_needs_additional_money_l363_363280

-- Defining the initial conditions
def price_of_new_computer : ℕ := 80
def initial_savings : ℕ := 50
def money_from_selling_old_computer : ℕ := 20

-- Defining the question and expected answer
def required_additional_money : ℕ :=
  price_of_new_computer - (initial_savings + money_from_selling_old_computer)

-- The proof statement
theorem troy_needs_additional_money : required_additional_money = 10 := by
  sorry

end troy_needs_additional_money_l363_363280


namespace h_80_eq_5_l363_363006

noncomputable def h : ℕ → ℝ
| x := if (∃ (n : ℤ), (logBase 3 x : ℝ) = n) then (logBase 3 x : ℝ)
        else 1 + h (x + 1)

theorem h_80_eq_5 : h 80 = 5 :=
by sorry

end h_80_eq_5_l363_363006


namespace degree_of_polynomial_l363_363865

variable (a b c d e f : ℝ)

theorem degree_of_polynomial (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
                            (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) :
  degree ((X^5 + C a * X^8 + C b * X^2 + C c) * (X^4 + C d * X^3 + C e) * (X^2 + C f)) = 14 :=
by 
  sorry

end degree_of_polynomial_l363_363865


namespace no_such_polyhedron_exists_l363_363648

def convex_polyhedron (P : Type _) [Polyhedron P] : Prop :=
  ∀ (plane : Plane), (¬ ∃ vertex : Vertex, plane ∋ vertex) → 
    ∃ (polygon : Polygon), poly_intersection P plane = polygon ∧ odd_sides polygon

theorem no_such_polyhedron_exists : 
  ¬ ∃ P : Type _, convex_polyhedron P :=
sorry

end no_such_polyhedron_exists_l363_363648


namespace framing_required_l363_363807

theorem framing_required :
  let original_length := 3
  let original_width := 5
  let enlarge_factor := 3
  let border := 3
  let enlarged_length := enlarge_factor * original_length
  let enlarged_width := enlarge_factor * original_width
  let final_length := enlarged_length + 2 * border
  let final_width := enlarged_width + 2 * border
  let perimeter_in_inches := 2 * (final_length + final_width)
  let inches_to_feet := 12
  let perimeter_in_feet := perimeter_in_inches / inches_to_feet
  (perimeter_in_feet = 6) :=
by
  let original_length := 3
  let original_width := 5
  let enlarge_factor := 3
  let border := 3
  let enlarged_length := enlarge_factor * original_length
  let enlarged_width := enlarge_factor * original_width
  let final_length := enlarged_length + 2 * border
  let final_width := enlarged_width + 2 * border
  let perimeter_in_inches := 2 * (final_length + final_width)
  let inches_to_feet := 12
  let perimeter_in_feet := perimeter_in_inches / inches_to_feet
  show perimeter_in_feet = 6
    from sorry

end framing_required_l363_363807


namespace circumscribed_sphere_distance_to_PAB_l363_363641

-- Definitions for vertices and point D
variables {A B C P D : Type*}

-- Given conditions as assumptions (formalizing geometry)
axiom angles_at_A : ∑ α ∈ {angle_BPC, angle_APC, angle_BPA}, α = 180
axiom angles_at_B : ∑ β ∈ {angle_APB, angle_BPC, angle_BPA}, β = 180
axiom PC_eq_AB : dist P C = dist A B
axiom distances_from_D : ∑ δ ∈ {dist D (plane PAB), dist D (plane PAC), dist D (plane PBC)}, δ = 7
axiom volume_ratio : volume (pyramid P A B C) = 8 * volume (pyramid B A B C)

-- Theorem to prove
theorem circumscribed_sphere_distance_to_PAB : 
  ∃ (O : Type*) (R : ℝ),  -- Center O of circumscribed sphere and its radius R
  let r := dist O (plane PAB) in r = 2 :=
sorry  -- The detailed proof would go here

end circumscribed_sphere_distance_to_PAB_l363_363641


namespace pairs_of_rackets_sold_l363_363378

theorem pairs_of_rackets_sold (total_sales : ℝ) (average_price_per_pair : ℝ) (h1 : total_sales = 686) (h2 : average_price_per_pair = 9.8) :
  total_sales / average_price_per_pair = 70 :=
by
  rw [h1, h2]
  norm_num
  sorry

end pairs_of_rackets_sold_l363_363378


namespace quadratic_roots_interval_l363_363595

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l363_363595


namespace inequality_x_y_l363_363919

theorem inequality_x_y (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2 * y^2 ≤ 3 := 
sorry

end inequality_x_y_l363_363919


namespace rectangle_area_l363_363993

theorem rectangle_area (A B C D : Type) (AB AC : ℝ) (ha : AB = 4) (hc : AC = 5)
  (hre : C ∈ bit0 ABCD) : (AB * sqrt (AC^2 - AB^2)) = 12 :=
by
  sorry

end rectangle_area_l363_363993


namespace volume_of_pyramid_l363_363273

noncomputable def generator := Real.sqrt 8
noncomputable def alpha := Real.pi / 6
noncomputable def beta := Real.pi / 4

theorem volume_of_pyramid :
  let l := generator
  let AO1 := l * Real.cos alpha
  let AO2 := l * Real.cos alpha
  let AO3 := l * Real.cos beta
  let volume := Real.sqrt (Real.sqrt 3 + 1)
  ∀ O1 O2 O3 A : ℝ × ℝ × ℝ,
  -- Condition 1: Three cones with apex A and generator sqrt(8)
  -- Conditions 2, 3: Two angles (pi/6) and one angle (pi/4)
  ∀ (h1 : dist A O1 = AO1) (h2 : dist A O2 = AO2) (h3 : dist A O3 = AO3),
  -- Conclusion
  volume_of_pyramid O1 O2 O3 A = volume :=
sorry

end volume_of_pyramid_l363_363273


namespace proof_primeAreaPairsCardinality_l363_363526

noncomputable def primeAreaPairsCardinality : Nat := 
  let primes_up_to_80 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
  let mn_values := primes_up_to_80.map (λ p => 2 * p)
  let num_divisors n := 
    (Finset.filter (λ d => n % d = 0) (Finset.range (n + 1))).card
  let num_pairs := mn_values.map num_divisors.sum
  num_pairs.sum

theorem proof_primeAreaPairsCardinality : primeAreaPairsCardinality = 87 :=
  sorry

end proof_primeAreaPairsCardinality_l363_363526


namespace reflected_point_is_correct_l363_363142

def original_point : ℝ × ℝ × ℝ := (-3, 1, 5)

def reflect_x_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(p.1, -p.2, -p.3)

theorem reflected_point_is_correct :
  reflect_x_axis original_point = (-3, -1, -5) :=
by
  sorry

end reflected_point_is_correct_l363_363142


namespace arithmetic_sequence_nth_term_l363_363035

theorem arithmetic_sequence_nth_term (S : ℕ → ℕ) (h : ∀ n, S n = 5 * n + 4 * n^2) (r : ℕ) : 
  S r - S (r - 1) = 8 * r + 1 := 
by
  sorry

end arithmetic_sequence_nth_term_l363_363035


namespace insurance_calculation_l363_363343

def loan_amount : ℝ := 20000000
def appraisal_value : ℝ := 14500000
def cadastral_value : ℝ := 15000000
def basic_tariff : ℝ := 0.2 / 100
def coefficient_no_transition : ℝ := 0.8
def coefficient_no_certificates : ℝ := 1.3

noncomputable def adjusted_tariff : ℝ := basic_tariff * coefficient_no_transition * coefficient_no_certificates
noncomputable def insured_amount : ℝ := max appraisal_value cadastral_value
noncomputable def insurance_premium : ℝ := insured_amount * adjusted_tariff

theorem insurance_calculation :
  adjusted_tariff = 0.00208 ∧ insurance_premium = 31200 := 
by
  sorry

end insurance_calculation_l363_363343


namespace system_of_equations_solution_exists_l363_363470

theorem system_of_equations_solution_exists :
  ∀ (p : ℕ) (a : Fin p → Fin (2 * p) → ℤ),
  (∀ i j, a i j = -1 ∨ a i j = 0 ∨ a i j = 1) →
  ∃ (x : Fin (2 * p) → ℤ),
    (∀ i, ∑ j in Finset.finRange (2 * p), a i j * x j = 0) ∧
    (∃ i, x i ≠ 0) ∧
    (∀ i, |x i| ≤ 2 * p) := sorry

end system_of_equations_solution_exists_l363_363470


namespace y_coordinate_of_first_point_l363_363638

theorem y_coordinate_of_first_point :
  ∃ x2 : ℝ, ∃ y1 : ℝ, y1 = 7.5 ∧ 
  (let m := (0 - y1) / (4 - (-6)) in
   let line_eq := m * (x2 + 6) in
   3 - y1 = line_eq) :=
begin
  existsi (arbitrary ℝ), -- x2 can be arbitrary because it cancels out in the equations.
  existsi 7.5,         -- y1 is the value we're solving for.
  split,
  refl,                -- Prove y1 = 7.5.
  dsimp,               -- Simplify the definition.
  sorry                -- Proof is omitted.
end

end y_coordinate_of_first_point_l363_363638


namespace cannot_cut_square_into_7_rectangles_l363_363185

theorem cannot_cut_square_into_7_rectangles (a : ℝ) :
  ¬ ∃ (x : ℝ), 7 * 2 * x ^ 2 = a ^ 2 ∧ 
    ∀ (i : ℕ), 0 ≤ i → i < 7 → (∃ (rect : ℝ × ℝ), rect.1 = x ∧ rect.2 = 2 * x ) :=
by
  sorry

end cannot_cut_square_into_7_rectangles_l363_363185


namespace donation_to_first_home_l363_363728

theorem donation_to_first_home :
  let total_donation := 700
  let donation_to_second := 225
  let donation_to_third := 230
  total_donation - donation_to_second - donation_to_third = 245 :=
by
  sorry

end donation_to_first_home_l363_363728


namespace intersection_result_l363_363068

noncomputable def A : Set ℝ := { x | x^2 - 5*x - 6 < 0 }
noncomputable def B : Set ℝ := { x | 2022^x > Real.sqrt 2022 }
noncomputable def intersection : Set ℝ := { x | A x ∧ B x }

theorem intersection_result : intersection = Set.Ioo (1/2 : ℝ) 6 := by
  sorry

end intersection_result_l363_363068


namespace symmetric_line_equation_l363_363940

theorem symmetric_line_equation (a b : ℝ) :
  (∀ x : ℝ, y = 2 * x + 3) ∧ l2_symmetric_to_l1_y_axis (y = 2 * x + 3) (y = -2 * x + 3) :=
begin
  sorry
end

end symmetric_line_equation_l363_363940


namespace base_for_346_is_odd_four_digit_l363_363038

theorem base_for_346_is_odd_four_digit :
  ∃ b : ℕ, (b^3 ≤ 346 ∧ 346 < b^4) ∧ nat.digits b 346 ≠ [] ∧ (nat.digits b 346).head! % 2 = 1 :=
by {
  use 7,
  split,
  { split,
    { -- base 7 meets the criterion b^3 <= 346
      norm_num,
      exact le_of_lt (nat.lt_of_subs_le (show 343 <= 346 by norm_num)),
    },
    { -- base 7 meets the criterion 346 < b^4
      norm_num,
      exact nat.lt_of_subs_le (show 346 < 2401 by norm_num),
    },
  },
  split,
  { -- ensuring that the digits list is not empty
    norm_num,
    intro h,
    exact list.head_cons (1 :: 0 :: 0 :: 3 :: list.nil) h,
  },
  { -- ensuring that the last digit is odd
    norm_num,
  },
  sorry
}

end base_for_346_is_odd_four_digit_l363_363038


namespace find_b_values_l363_363552

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l363_363552


namespace quadratic_real_root_iff_b_range_l363_363585

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l363_363585


namespace problem_1_problem_2_l363_363148

-- Define the conditions
def ellipse_eq (m : ℝ) : Prop := 9 - m > 2 * m ∧ 2 * m > 0
def hyperbola_eq (m : ℝ) : Prop := sqrt(6) / 2 < sqrt(5 + m) / 5 ∧ sqrt(5 + m) / 5 < sqrt(2)

-- Define the condition for making p true
def p_condition (m : ℝ) : Prop := ellipse_eq m

-- Define the condition for making q true
def q_condition (m : ℝ) : Prop := hyperbola_eq m

-- Given condition that foci of ellipse and vertices of hyperbola coincide
def coincide_condition (m : ℝ) : Prop := sqrt(9 - 3 * m) = sqrt(5)

-- Problem (1): Prove that m = 4/3 when the foci of the ellipse and the vertices of the hyperbola coincide.
theorem problem_1 : ∃ m : ℝ, coincide_condition m ∧ m = 4 / 3 :=
by sorry

-- Problem (2): Prove the range of m values given p ∧ q are true.
theorem problem_2 : ∀ m : ℝ, (p_condition m ∧ q_condition m) → (5 / 2 < m ∧ m < 3) :=
by sorry

end problem_1_problem_2_l363_363148


namespace find_integer_that_satisfies_conditions_l363_363417

theorem find_integer_that_satisfies_conditions : 
  ∃ n : ℤ, (n + 15 > 16) ∧ (-3 * n > -9) ∧ (n = 2) :=
by
  use 2
  split
  { linarith }
  split
  { linarith }
  { rfl }

end find_integer_that_satisfies_conditions_l363_363417


namespace complex_expression_l363_363462

theorem complex_expression (z : ℂ) (hz : z = 4 + 3 * complex.i) :
  (conj z / complex.abs z) = (4 / 5) - (3 / 5) * complex.i := by 
  -- Proof to be provided
  sorry

end complex_expression_l363_363462


namespace perpendicular_lines_find_b_l363_363762

theorem perpendicular_lines_find_b :
  (∃ b : ℝ, (⟨4, -5⟩ : ℝ × ℝ) • ⟨b, 8⟩ = 0) → b = 10 :=
begin
  sorry
end

end perpendicular_lines_find_b_l363_363762


namespace computation_l363_363175

def h (x : ℝ) : ℝ := x + 3
def j (x : ℝ) : ℝ := x / 4
def h_inv (x : ℝ) : ℝ := x - 3
def j_inv (x : ℝ) : ℝ := 4 * x

theorem computation :
  h (j_inv (h_inv (h_inv (j (h 20))))) = 2 := by
  sorry

end computation_l363_363175


namespace find_b_values_l363_363550

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l363_363550


namespace permutations_satisfying_condition_l363_363509

noncomputable def count_permutations (n : ℕ) (h : n ≥ 2) : ℕ :=
  2 * 3^(n-2)

theorem permutations_satisfying_condition {n : ℕ} (h : n ≥ 2) :
  ∃ (σ : Equiv.Perm (Fin n)), 
    (∀ k : Fin n, σ k + 2 ≥ k) ∧ 
    Finset.card {σ | ∀ k : Fin n, σ k + 2 ≥ k} = count_permutations n h :=
by
  sorry

end permutations_satisfying_condition_l363_363509


namespace tucker_circle_center_on_KO_l363_363695

variable {A B C : Point}

-- Definition of Lemoine point and circumcenter for triangle ABC
def isLemoinePoint (K : Point) (A B C : Point) : Prop := sorry
def isCircumcenter (O : Point) (A B C : Point) : Prop := sorry

-- Definition of Tucker circle center
def isTuckerCircleCenter (O1 : Point) (A B C : Point) : Prop := sorry

-- Proof statement: Center of the Tucker circle lies on the line KO
theorem tucker_circle_center_on_KO
  {A B C K O O1 : Point}
  (hK : isLemoinePoint K A B C)
  (hO : isCircumcenter O A B C)
  (hO1 : isTuckerCircleCenter O1 A B C) :
  liesOnLine O1 (lineThrough K O) :=
sorry

end tucker_circle_center_on_KO_l363_363695


namespace speed_of_boat_in_still_water_l363_363787

variables (V_s : ℝ) (V_b : ℝ) (D_downstream : ℝ) (D_upstream : ℝ)

-- Conditions from the problem
def downstream_time := 1
def upstream_time := 3 / 2
def speed_of_stream := 3
def distance_downstream := (V_b + V_s) * downstream_time
def distance_upstream := (V_b - V_s) * upstream_time

-- The statement we need to prove
theorem speed_of_boat_in_still_water 
  (h1 : V_s = speed_of_stream)
  (h2 : D_downstream = distance_downstream)
  (h3 : D_upstream = distance_upstream)
  (h4 : D_downstream = D_upstream) :
  V_b = 15 :=
by
  sorry

end speed_of_boat_in_still_water_l363_363787


namespace Katona_theorem_l363_363704

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem Katona_theorem {n k : ℕ} (h : k ≤ n / 2) 
(F : set (finset (fin n) × finset (fin n))) 
(hF : ∀ A B ∈ F, finset.inter A.1 B.1 ≠ ∅ ∧ finset.inter A.2 B.2 ≠ ∅) :
  F.card ≤ (binomial (n-1) (k-1)) * (binomial (n-1) (k-1)) :=
sorry

end Katona_theorem_l363_363704


namespace number_of_children_l363_363657

def male_adults : ℕ := 60
def female_adults : ℕ := 60
def total_people : ℕ := 200

def total_adults : ℕ := male_adults + female_adults

theorem number_of_children : total_people - total_adults = 80 :=
by sorry

end number_of_children_l363_363657


namespace max_area_of_equilateral_triangle_in_rectangle_l363_363375

noncomputable def maxEquilateralTriangleArea : ℝ :=
  205 * Real.sqrt 3 - 468

theorem max_area_of_equilateral_triangle_in_rectangle
  (a b : ℝ) (h₁ : a = 12) (h₂ : b = 13) :
  ∃ area, area = maxEquilateralTriangleArea :=
by
  use maxEquilateralTriangleArea
  exact maxEquilateralTriangleArea

end max_area_of_equilateral_triangle_in_rectangle_l363_363375


namespace last_digit_base5_89_l363_363416

theorem last_digit_base5_89 (n : ℕ) (h : n = 89) : (n % 5) = 4 :=
by 
  sorry

end last_digit_base5_89_l363_363416


namespace part_II_l363_363675

noncomputable def f (x : ℝ) : ℝ := |x - 2| + 2 * x - 3

def M : set ℝ := {x | x ≤ 0}

theorem part_II (x : ℝ) (hx : x ∈ M) : x * (f x)^2 - x^2 * f x ≤ 0 := sorry

end part_II_l363_363675


namespace inequality_x_y_l363_363917

theorem inequality_x_y (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2 * y^2 ≤ 3 := 
sorry

end inequality_x_y_l363_363917


namespace monomial_sum_l363_363036

variable {x y : ℝ}

theorem monomial_sum (a : ℝ) (h : -2 * x^2 * y^3 + 5 * x^(a-1) * y^3 = c * x^k * y^3) : a = 3 :=
  by
  sorry

end monomial_sum_l363_363036


namespace insurance_calculation_l363_363341

def loan_amount : ℝ := 20000000
def appraisal_value : ℝ := 14500000
def cadastral_value : ℝ := 15000000
def basic_tariff : ℝ := 0.2 / 100
def coefficient_no_transition : ℝ := 0.8
def coefficient_no_certificates : ℝ := 1.3

noncomputable def adjusted_tariff : ℝ := basic_tariff * coefficient_no_transition * coefficient_no_certificates
noncomputable def insured_amount : ℝ := max appraisal_value cadastral_value
noncomputable def insurance_premium : ℝ := insured_amount * adjusted_tariff

theorem insurance_calculation :
  adjusted_tariff = 0.00208 ∧ insurance_premium = 31200 := 
by
  sorry

end insurance_calculation_l363_363341


namespace jerry_runs_more_than_tom_l363_363613

theorem jerry_runs_more_than_tom :
  let side_length := 500 in
  let street_width := 30 in
  let tom_perimeter := 4 * side_length in
  let jerry_perimeter := 4 * (side_length + 2 * street_width) in
  jerry_perimeter - tom_perimeter = 240 :=
by
  sorry

end jerry_runs_more_than_tom_l363_363613


namespace complex_number_solution_l363_363973

theorem complex_number_solution (z : ℂ) (h : 2 * (z + conj z) = -2 * complex.I) : z = 0 - (1/2) * complex.I :=
by sorry

end complex_number_solution_l363_363973


namespace dot_product_result_l363_363031

def u : ℝ × ℝ × ℝ := (4, -3, 5)
def v : ℝ × ℝ × ℝ := (-3, 6, -2)
def a : ℝ := 2

theorem dot_product_result : let u' := (a * u.1, a * u.2, a * u.3) in
  u'.1 * v.1 + u'.2 * v.2 + u'.3 * v.3 = -80 :=
by
  let u' := (a * u.1, a * u.2, a * u.3)
  sorry

end dot_product_result_l363_363031


namespace segment_length_and_midpoint_l363_363439

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def midpoint (p1 p2 : point) : point :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  ((x1 + x2) / 2, (y1 + y2) / 2)

def A : point := (1, 2)
def B : point := (9, 14)

theorem segment_length_and_midpoint :
  distance A B = 4 * Real.sqrt 13 ∧ midpoint A B = (5, 8) :=
by
  sorry

end segment_length_and_midpoint_l363_363439


namespace total_cost_is_9220_l363_363820

-- Define the conditions
def hourly_rate := 60
def hours_per_day := 8
def total_days := 14
def cost_of_parts := 2500

-- Define the total cost the car's owner had to pay based on conditions
def total_hours := hours_per_day * total_days
def labor_cost := total_hours * hourly_rate
def total_cost := labor_cost + cost_of_parts

-- Theorem stating that the total cost is $9220
theorem total_cost_is_9220 : total_cost = 9220 := by
  sorry

end total_cost_is_9220_l363_363820


namespace perpendicular_vectors_solution_l363_363522

theorem perpendicular_vectors_solution (m : ℝ) (a : ℝ × ℝ := (m-1, 2)) (b : ℝ × ℝ := (m, -3)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : m = 3 ∨ m = -2 :=
by sorry

end perpendicular_vectors_solution_l363_363522


namespace remainder_of_product_mod_43_l363_363032

theorem remainder_of_product_mod_43 : 
  ((∏ n in Finset.range 42, (n + 1)^2 + 1) % 43) = 4 :=
sorry

end remainder_of_product_mod_43_l363_363032


namespace Aaron_sweaters_count_l363_363425

theorem Aaron_sweaters_count:
  ∃ S : ℕ, let scarves_wool := 10 * 3 in
           let enid_wool := 8 * 4 in
           let total_wool := 82 in
           let sweaters_wool := 4 * S in
           scarves_wool + enid_wool + sweaters_wool = total_wool ∧
           S = 5 :=
by
  use 5
  sorry

end Aaron_sweaters_count_l363_363425


namespace quadratic_real_roots_l363_363579

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l363_363579


namespace nickel_ate_4_chocolates_l363_363200

theorem nickel_ate_4_chocolates (R N : ℕ) (h1 : R = 13) (h2 : R = N + 9) : N = 4 :=
by
  sorry

end nickel_ate_4_chocolates_l363_363200


namespace problem_1_problem_2_l363_363899

theorem problem_1 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 + c^2 = 9) : abc ≤ 3 * Real.sqrt 3 := 
sorry

theorem problem_2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 + c^2 = 9) : 
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) > (a + b + c) / 3 := 
sorry

end problem_1_problem_2_l363_363899


namespace correct_statements_count_l363_363413

open Real

def f (x : ℝ) := 3 * sin (2 * x - π / 3)

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x, f (2 * a - x) = f x

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) :=
  ∀ x y ∈ I, x < y → f x < f y

def shifted_function (f : ℝ → ℝ) (a : ℝ) (x : ℝ) := f (x - a)

theorem correct_statements_count :
  let C := f
  let is_symmetric := is_symmetric_about f (11 * π / 12)
  let is_increasing := is_increasing_on f (Set.Icc (-π / 12) (5 * π / 12))
  let shifted := ∀ x, shifted_function (λ x, 3 * sin (2 * x)) (π / 3) x = C x
  num_correct := if is_symmetric ∧ is_increasing ∧ ¬shifted then 2 else 1
  num_correct = 2 := 
by
  sorry

end correct_statements_count_l363_363413


namespace num_good_triples_at_least_l363_363665

noncomputable def num_good_triples (S : Finset (ℕ × ℕ)) (n m : ℕ) : ℕ :=
  4 * m * (m - n^2 / 4) / (3 * n)

theorem num_good_triples_at_least
  (S : Finset (ℕ × ℕ))
  (n m : ℕ)
  (h_S : ∀ (x : ℕ × ℕ), x ∈ S → 1 ≤ x.1 ∧ x.1 < x.2 ∧ x.2 ≤ n)
  (h_m : S.card = m)
  : ∃ t ≤ num_good_triples S n m, True := 
sorry

end num_good_triples_at_least_l363_363665


namespace sum_max_min_eq_four_l363_363937

noncomputable def f : ℝ → ℝ := λ x, (x ^ 2 - 2 * x) * Real.sin (x - 1) + x + 1

theorem sum_max_min_eq_four : 
  let I := Set.Icc (-1 : ℝ) (3 : ℝ) in
  let M := ⨆ x ∈ I, f x in
  let m := ⨀ x ∈ I, f x in
  M + m = 4 :=
begin
  sorry
end

end sum_max_min_eq_four_l363_363937


namespace inequality_proof_l363_363158

section
variable {a b x y : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hab : a + b = 1) :
  (1 / (a / x + b / y) ≤ a * x + b * y) ∧ (1 / (a / x + b / y) = a * x + b * y ↔ a * y = b * x) :=
  sorry
end

end inequality_proof_l363_363158


namespace smallest_number_increased_by_3_divisible_l363_363773

theorem smallest_number_increased_by_3_divisible (n : ℤ) 
    (h1 : (n + 3) % 18 = 0)
    (h2 : (n + 3) % 70 = 0)
    (h3 : (n + 3) % 25 = 0)
    (h4 : (n + 3) % 21 = 0) : 
    n = 3147 :=
by
  sorry

end smallest_number_increased_by_3_divisible_l363_363773


namespace fleas_to_the_right_l363_363176

theorem fleas_to_the_right (n : ℕ) (h : n ≥ 2) (λ : ℝ) :
  (∀ (M : ℝ) (initial_positions : Fin n → ℝ), ∃ (N : ℕ) (positions_after_N := λ positions : Fin n → ℝ, ∀ i, positions i > M),
  (∃ m ≤ N, positions_after_N (move_sequence initial_positions m)) ) ↔ λ ≥ (1 / (n - 1)) :=
sorry

end fleas_to_the_right_l363_363176


namespace almonds_received_l363_363951

variable (Lydia Max : ℕ)

theorem almonds_received :
  Lydia = Max + 8 ∧ Max = Lydia / 3 → Lydia = 12 :=
begin
  sorry
end

end almonds_received_l363_363951


namespace calculate_m_l363_363844

theorem calculate_m : ∃ m : ℚ, (256 : ℝ)^(1/3) = (2 : ℝ)^m ∧ m = 8/3 :=
by
  use 8/3
  split
  · sorry
  · refl

end calculate_m_l363_363844


namespace solve_inequality_l363_363718

open Set

theorem solve_inequality (a x : ℝ) : 
  (x - 2) * (a * x - 2) > 0 → 
  (a = 0 ∧ x < 2) ∨ 
  (a < 0 ∧ (2/a) < x ∧ x < 2) ∨ 
  (0 < a ∧ a < 1 ∧ ((x < 2 ∨ x > 2/a))) ∨ 
  (a = 1 ∧ x ≠ 2) ∨ 
  (a > 1 ∧ ((x < 2/a ∨ x > 2)))
  := sorry

end solve_inequality_l363_363718


namespace problem_quadratic_has_real_root_l363_363602

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l363_363602


namespace total_bouncy_balls_l363_363183

def red_packs := 4
def yellow_packs := 8
def green_packs := 4
def balls_per_pack := 10

theorem total_bouncy_balls:
  (red_packs * balls_per_pack + yellow_packs * balls_per_pack + green_packs * balls_per_pack) = 160 :=
by 
  sorry

end total_bouncy_balls_l363_363183


namespace quadratic_roots_interval_l363_363588

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l363_363588


namespace equivalent_integer_l363_363533

theorem equivalent_integer (a b n : ℤ) (h1 : a ≡ 33 [ZMOD 60]) (h2 : b ≡ 85 [ZMOD 60]) (hn : 200 ≤ n ∧ n ≤ 251) : 
  a - b ≡ 248 [ZMOD 60] :=
sorry

end equivalent_integer_l363_363533


namespace limit_cos_plus_one_pow_sin_l363_363408

open Real

theorem limit_cos_plus_one_pow_sin (f : ℝ → ℝ) :
  (∀ x, f x = (cos x + 1) ^ (sin x)) → 
  filter.tendsto f (nhds π/2) (nhds 1) :=
begin
  intro h_f,
  sorry
end

end limit_cos_plus_one_pow_sin_l363_363408


namespace obtuse_triangle_consecutive_sides_l363_363240

theorem obtuse_triangle_consecutive_sides :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ b = a + 1 ∧ c = b + 1 ∧ (a^2 + b^2 < c^2) :=
by {
  use [2, 3, 4],
  simp,
  sorry
}

end obtuse_triangle_consecutive_sides_l363_363240


namespace first_player_win_l363_363763

noncomputable def first_player_always_wins : Prop :=
  ∀ board : Array (Array Bool) (100) (3),
  (∀ move : ℕ, 
    (∃ p1, p1_places_tile board move) ∨ (∃ p2, p2_places_tile board move)) →
  ∃ winning_strategy : (board → (ℕ → Prop)), ∀ b, winning_strategy b

theorem first_player_win :
  first_player_always_wins := 
sorry

end first_player_win_l363_363763


namespace figure_perimeter_l363_363772

-- Define the side length of the square and the triangles.
def square_side_length : ℕ := 3
def triangle_side_length : ℕ := 2

-- Calculate the perimeter of the figure
def perimeter (a b : ℕ) : ℕ := 2 * a + 2 * b

-- Statement to prove
theorem figure_perimeter : perimeter square_side_length triangle_side_length = 10 := 
by 
  -- "sorry" denotes that the proof is omitted.
  sorry

end figure_perimeter_l363_363772


namespace longer_subsegment_of_YZ_l363_363146

/-- In triangle XYZ with sides in the ratio 3:4:5, and side YZ being 12 cm.
    The angle bisector XW divides side YZ into segments YW and ZW.
    Prove that the length of ZW is 48/7 cm. --/
theorem longer_subsegment_of_YZ (YZ : ℝ) (hYZ : YZ = 12)
    (XY XZ : ℝ) (hRatio : XY / XZ = 3 / 4) : 
    ∃ ZW : ℝ, ZW = 48 / 7 :=
by
  -- We would provide proof here
  sorry

end longer_subsegment_of_YZ_l363_363146


namespace max_value_of_tangent_and_cosine_difference_l363_363537

theorem max_value_of_tangent_and_cosine_difference:
  (∀ x : ℝ, x ∈ Icc (-5 * Real.pi / 12) (-Real.pi / 3) →
    (∃ z : ℝ, z = max (tan (x + 2 * Real.pi / 3) - tan (x + Real.pi / 6) + (cos (x + Real.pi / 6))) → 
        z = (11 * Real.sqrt 3) / 6) :=
begin
  sorry
end

end max_value_of_tangent_and_cosine_difference_l363_363537


namespace rope_in_two_months_period_l363_363869

theorem rope_in_two_months_period :
  let week1 := 6
  let week2 := 3 * week1
  let week3 := week2 - 4
  let week4 := - (week2 / 2)
  let week5 := week1 + 2
  let week6 := - (2 / 2)
  let week7 := 3 * (2 / 2)
  let week8 := - 10
  let total_length := (week1 + week2 + week3 + week4 + week5 + week6 + week7 + week8)
  total_length * 12 = 348
:= sorry

end rope_in_two_months_period_l363_363869


namespace collinear_A_E_F_l363_363992

noncomputable def point : Type := (ℝ × ℝ)

structure triangle :=
(A B C : point)

structure circumcenter (t : triangle) :=
(O : point)

structure centroid (t : triangle) :=
(G : point)

structure midpoint (p1 p2 : point) :=
(M : point)

variables {t : triangle}
variables (O : circumcenter t)
variables (G : centroid t)
variables (A1 : midpoint (t.B t.C))
variables (B1 : midpoint (t.C t.A))
variables (C1 : midpoint (t.A t.B))

def line_perpendicular_to (P1 P2 : point) : Type := { L : point × point // L.1 = P1 ∧ ∃ θ : ℝ, P2.2 = P1.2 + θ * (P2.1 - P1.1) }

axiom l_B : line_perpendicular_to t.B G.G
axiom l_C : line_perpendicular_to t.C G.G

variables (E : point) (F : point)

axiom intersect_l_B_A1C1 : ∃ (P : point), P = E ∧ ∃ L ∈ l_B, P ∈ A1.M.C1.M
axiom intersect_l_C_A1B1 : ∃ (P : point), P = F ∧ ∃ L ∈ l_C, P ∈ A1.M.B1.M

theorem collinear_A_E_F (t : triangle) (O : circumcenter t) (G : centroid t) 
  (A1 : midpoint (t.B t.C)) (B1 : midpoint (t.C t.A)) (C1 : midpoint (t.A t.B))
  (l_B : line_perpendicular_to t.B G.G) (l_C : line_perpendicular_to t.C G.G)
  (E F : point) 
  (intersect_l_B_A1C1 : ∃ (P : point), P = E ∧ ∃ L ∈ l_B, P ∈ A1.M.C1.M)
  (intersect_l_C_A1B1 : ∃ (P : point), P = F ∧ ∃ L ∈ l_C, P ∈ A1.M.B1.M) : 
  collinear t.A E F := 
sorry

end collinear_A_E_F_l363_363992


namespace range_of_possible_slopes_l363_363907

noncomputable def line_eq_through_point (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y, k * x - y + (k - 1) = 0

noncomputable def circle_eq : ℝ → ℝ → Prop :=
  λ x y, (x - 1)^2 + (y + 3)^2 = 4

theorem range_of_possible_slopes : ∀ (k : ℝ),
  (∃ x y : ℝ, line_eq_through_point k x y ∧ circle_eq x y ) ↔ k < 0 :=
by
  sorry

end range_of_possible_slopes_l363_363907


namespace max_value_of_p_l363_363671

open Real

theorem max_value_of_p (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : x * y * z + x + z = y) : 
  ∃ (p : ℝ), p = \frac{2}{x^{2}+1} - \frac{2}{y^{2}+1} + \frac{3}{z^{2}+1} ∧ p ≤ \frac{10}{3} :=
sorry

end max_value_of_p_l363_363671


namespace alice_password_probability_is_correct_l363_363836

-- Define the condition for even numbers and nonzero numbers
def even_digit (x : ℕ) : Prop := x % 2 = 0 ∧ x ≤ 9
def non_zero_digit (x : ℕ) : Prop := x > 0 ∧ x ≤ 9

-- Define the conditions of the password
def valid_password : Type := string × ℕ × ℕ

-- Define the main probability function in terms of conditions
def probability_valid_password (p : valid_password) : ℚ :=
  if ∃ x y : ℕ, even_digit x ∧ non_zero_digit y 
     then (1/2) * (9/10)
     else 0

-- The statement of the theorem we need to prove
theorem alice_password_probability_is_correct (p : valid_password) :
  ∃ x y : ℕ, even_digit x ∧ non_zero_digit y → probability_valid_password p = (9/20) :=
by
  sorry

end alice_password_probability_is_correct_l363_363836


namespace majority_votes_l363_363991

-- Definitions based on the conditions
def total_valid_votes : ℕ := 470
def winning_percentage : ℝ := 0.70
def losing_percentage : ℝ := 0.30

-- The proof statement
theorem majority_votes (W : ℕ) (L : ℕ) :
  W = (winning_percentage * total_valid_votes).to_nat →
  L = (losing_percentage * total_valid_votes).to_nat →
  W - L = 188 :=
by
  intros hW hL
  sorry

end majority_votes_l363_363991


namespace trick_deck_cost_l363_363040

theorem trick_deck_cost :
  ∀ (x : ℝ), 3 * x + 2 * x = 35 → x = 7 :=
by
  sorry

end trick_deck_cost_l363_363040


namespace find_b_from_law_of_sines_l363_363489

noncomputable def sin60 : ℝ := Real.sin (Real.pi / 3)  -- sin(60°)
noncomputable def sin45 : ℝ := Real.sin (Real.pi / 4)  -- sin(45°)

theorem find_b_from_law_of_sines (a : ℝ) (A B : ℝ) (ha : a = 3) (hA : A = Real.pi / 3) (hB : B = Real.pi / 4) :
  ∃ b : ℝ, b = Real.sqrt 6 :=
by
  have hsinA : Real.sin A = sin60 := by simp [hA, sin60]
  have hsinB : Real.sin B = sin45 := by simp [hB, sin45]
  use (a * Real.sin B / Real.sin A)
  rw [ha, hsinA, hsinB]
  simp
  sorry

end find_b_from_law_of_sines_l363_363489


namespace colony_never_extinct_l363_363757

noncomputable def probability_never_extinct (cells : ℕ) (p_die : ℚ) (p_split : ℚ) : ℚ :=
  let p_survive := 1 - p_die
  in 1 - (p_survive / p_split) ^ cells

theorem colony_never_extinct :
  probability_never_extinct 100 (1/3) (2/3) = 1 - (1/2) ^ 100 :=
by
  sorry

end colony_never_extinct_l363_363757


namespace correct_proportion_l363_363612

theorem correct_proportion {a b c x y : ℝ} 
  (h1 : x + y = b)
  (h2 : x * c = y * a) :
  y / a = b / (a + c) :=
sorry

end correct_proportion_l363_363612


namespace problem_equiv_l363_363850

noncomputable def f (n : ℕ) : ℝ :=
  (n + 1)^3 / ((n - 1) * n) - (n - 1)^3 / (n * (n + 1))

theorem problem_equiv : (floor (f 1004)) = 8 :=
by {
  sorry
}

end problem_equiv_l363_363850


namespace probability_stops_at_n_smallest_n_for_probability_condition_l363_363872

theorem probability_stops_at_n (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 2010) :
  (1 : ℝ) / (n * (n + 1)) < 1 / 2010 :=
sorry

theorem smallest_n_for_probability_condition :
  ∃ n : ℕ, (1 ≤ n ∧ n ≤ 2010) ∧ (1 / (n * (n + 1)) < 1 / 2010) ∧ ∀ m : ℕ, (1 ≤ m ∧ m < n) → ¬(1 / (m * (m + 1)) < 1 / 2010) :=
by
  use 45
  split
  -- Prove 1 ≤ 45 and 45 ≤ 2010
  { split, exact nat.one_le_of_lt (by norm_num), exact nat.le_refl 45 }
  split
  -- Prove 1 / (45 * (45 + 1)) < 1 / 2010
  { exact probability_stops_at_n 45 (by norm_num) (by norm_num) }
  -- Prove ∀ m : ℕ, (1 ≤ m ∧ m < 45) → ¬(1 / (m * (m + 1)) < 1 / 2010)
  { intros m hm 
    cases hm with hm1 hm2
    have h : m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 
    ∨ m = 6 ∨ m = 7 ∨ m = 8 ∨ m = 9 ∨ m = 10
    ∨ m = 11 ∨ m = 12 ∨ m = 13 ∨ m = 14 ∨ m = 15
    ∨ m = 16 ∨ m = 17 ∨ m = 18 ∨ m = 19 ∨ m = 20
    ∨ m = 21 ∨ m = 22 ∨ m = 23 ∨ m = 24 ∨ m = 25
    ∨ m = 26 ∨ m = 27 ∨ m = 28 ∨ m = 29 ∨ m = 30
    ∨ m = 31 ∨ m = 32 ∨ m = 33 ∨ m = 34 ∨ m = 35
    ∨ m = 36 ∨ m = 37 ∨ m = 38 ∨ m = 39 ∨ m = 40
    ∨ m = 41 ∨ m = 42 ∨ m = 43 ∨ m = 44,
    { sorry }
  }

end probability_stops_at_n_smallest_n_for_probability_condition_l363_363872


namespace cost_of_parts_l363_363153

theorem cost_of_parts (C : ℝ) 
  (h1 : ∀ n ∈ List.range 60, (1.4 * C * n) = (1.4 * C * 60))
  (h2 : 5000 + 3000 = 8000)
  (h3 : 60 * C * 1.4 - (60 * C + 8000) = 11200) : 
  C = 800 := by
  sorry

end cost_of_parts_l363_363153


namespace intersection_A_B_l363_363067

open Set

def A := {0, 1, 3, 5, 7}
def B := {2, 4, 6, 8, 0}

theorem intersection_A_B : A ∩ B = {0} := by
  sorry

end intersection_A_B_l363_363067


namespace maria_coins_difference_l363_363186

theorem maria_coins_difference :
  ∀ (p n d : ℕ), p + n + d = 3030 ∧ p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 →
  let max_value := 30286
  let min_value := 3043
  in max_value - min_value = 27243 :=
begin
  intros p n d h,
  rcases h with ⟨h1, h2, h3, h4⟩,
  let max_value := 1 + 5 * 1 + 10 * (3030 - 1 - 1),
  let min_value := 3030 - 1 - 1 + 5 * 1 + 10 * 1,
  have : max_value - min_value = 27243, sorry,
  exact this,
end

end maria_coins_difference_l363_363186


namespace eq_of_curve_and_tangent_circle_l363_363076

-- Given conditions and definitions
def curve_eq (m n : ℝ) (x y : ℝ) : Prop := m * x^2 + n * y^2 = 1

def point_A : ℝ × ℝ := (sqrt 2 / 4, sqrt 2 / 2)
def point_B : ℝ × ℝ := (sqrt 6 / 6, sqrt 3 / 3)

def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Proving statements
theorem eq_of_curve_and_tangent_circle :
  ∃ (m n : ℝ), (m > 0) ∧ (n > 0) ∧ 
  (curve_eq m n (point_A.1) (point_A.2)) ∧ (curve_eq m n (point_B.1) (point_B.2)) ∧
  (∀ (x1 y1 x2 y2 : ℝ), curve_eq 4 1 x1 y1 → 
                        curve_eq 4 1 x2 y2 → 
                        perpendicular x1 y1 x2 y2 → 
                        (∃ (k : ℝ), k = sqrt 5 / 5 ∧
                        x1^2 + x2^2 + y1^2 + y2^2 = 
                        (x1 - x2)^2 + (y1 - y2)^2 + k^2)) :=
begin
  sorry
end

end eq_of_curve_and_tangent_circle_l363_363076


namespace participant_A_can_determine_number_l363_363766

def binary_representation (n : ℕ) : list ℕ :=
  let rec to_binary (k : ℕ) (acc : list ℕ) : list ℕ :=
    if k = 0 then acc else to_binary (k / 2) (k % 2 :: acc)
  to_binary n []

def encode_number (n : ℕ) : list bool :=
  (binary_representation n).map (λ b, b = 1)

def decode_number (lst : list bool) : ℕ :=
  lst.foldr (λ b acc, acc * 2 + if b then 1 else 0) 0

theorem participant_A_can_determine_number (n : ℕ) (h : 0 ≤ n ∧ n < 32) :
  decode_number (encode_number n) = n :=
by
  sorry

end participant_A_can_determine_number_l363_363766


namespace solution_proof_l363_363034

noncomputable def proof_problem : Prop :=
∀ (f : ℝ → ℝ), (∀ x, differentiable ℝ f) →
(∀ x, (x - 1) * (deriv f x) ≤ 0) →
f 0 + f 2 ≤ 2 * f 1

theorem solution_proof : proof_problem :=
by
  sorry

end solution_proof_l363_363034


namespace proof_problem_l363_363672

variable {a b c : ℝ}
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variable (h4 : (a+1) * (b+1) * (c+1) = 8)

theorem proof_problem :
  a + b + c ≥ 3 ∧ abc ≤ 1 :=
by
  sorry

end proof_problem_l363_363672


namespace least_value_divisibility_l363_363771

theorem least_value_divisibility : ∃ (x : ℕ), (23 * x) % 3 = 0  ∧ (∀ y : ℕ, ((23 * y) % 3 = 0 → x ≤ y)) := 
  sorry

end least_value_divisibility_l363_363771


namespace candy_lollipops_l363_363756

theorem candy_lollipops (κ c l : ℤ) 
  (h1 : κ = l + c - 8)
  (h2 : c = l + κ - 14) :
  l = 11 :=
by
  sorry

end candy_lollipops_l363_363756


namespace find_new_songs_l363_363323

-- Definitions for the conditions
def initial_songs : ℕ := 6
def deleted_songs : ℕ := 3
def final_songs : ℕ := 23

-- The number of new songs added
def new_songs_added : ℕ := 20

-- Statement of the proof problem
theorem find_new_songs (n d f x : ℕ) (h1 : n = initial_songs) (h2 : d = deleted_songs) (h3 : f = final_songs) : f = n - d + x → x = new_songs_added :=
by
  intros h4
  sorry

end find_new_songs_l363_363323


namespace find_specific_function_l363_363393

theorem find_specific_function :
  ∃ (f : ℝ → ℝ), 
  f = (λ x, -Real.log (Real.abs x)) ∧ 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 ≥ f x2) ∧ 
  (∀ g : ℝ → ℝ, (g = (λ x, x^2) ∨ g = (λ x, -x^3) ∨ g = (λ x, -Real.log (Real.abs x)) ∨
    g = (λ x, Real.exp x)) →
    (g = (λ x, -Real.log (Real.abs x))) :=
by
  sorry

end find_specific_function_l363_363393


namespace quadratic_real_roots_l363_363563

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l363_363563


namespace boris_fudge_consumption_l363_363276

def convert_to_ounces (pounds : ℝ) : ℝ := pounds * 16

theorem boris_fudge_consumption (tomas_fudge_katya_fudge_total_fudge : ℝ × ℝ × ℝ) :
  let (tomas_pounds, katya_pounds, total_ounces) := tomas_fudge_katya_fudge_total_fudge in
  convert_to_ounces tomas_pounds + convert_to_ounces katya_pounds + convert_to_ounces 2 = total_ounces :=
by 
  let tomas_ounces := 1.5 * 16 
  let katya_ounces := 0.5 * 16 
  let boris_ounces := total_ounces - (tomas_ounces + katya_ounces)
  have h : boris_ounces / 16 = 2, by sorry
  exact h

#eval boris_fudge_consumption (1.5, 0.5, 64)

end boris_fudge_consumption_l363_363276


namespace yield_difference_correct_l363_363400

noncomputable def tomato_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def corn_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def onion_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def carrot_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)

theorem yield_difference_correct :
  let tomato_initial := 2073
  let corn_initial := 4112
  let onion_initial := 985
  let carrot_initial := 6250
  let tomato_growth := 12
  let corn_growth := 15
  let onion_growth := 8
  let carrot_growth := 10
  let tomato_total := tomato_yield tomato_initial tomato_growth
  let corn_total := corn_yield corn_initial corn_growth
  let onion_total := onion_yield onion_initial onion_growth
  let carrot_total := carrot_yield carrot_initial carrot_growth
  let highest_yield := max (max tomato_total corn_total) (max onion_total carrot_total)
  let lowest_yield := min (min tomato_total corn_total) (min onion_total carrot_total)
  highest_yield - lowest_yield = 5811.2 := by
  sorry

end yield_difference_correct_l363_363400


namespace geometric_sequence_relation_l363_363080

noncomputable def sum_geometric_sequence (a r : ℂ) : ℕ → ℂ
| 0       := 0
| (n + 1) := a + r * sum_geometric_sequence a r n

theorem geometric_sequence_relation (a r : ℂ) (hn : r ≠ 1) (k : ℕ) :
  let S_n := sum_geometric_sequence a r k in
  let S_2n := sum_geometric_sequence a r (2 * k) in
  let S_3n := sum_geometric_sequence a r (3 * k) in
  S_n^2 + S_2n^2 = S_n * (S_2n + S_3n) :=
by sorry

end geometric_sequence_relation_l363_363080


namespace regular_222_gon_no_equal_mono_dichromatic__l363_363258

def regular_n_gon (n : ℕ) := true

def is_colored (vertices : ℕ → bool) := true

def side_is_monochromatic (vertices : ℕ → bool) (i : ℕ) : bool :=
vertices i = vertices ((i + 1) % 222)

def side_is_dichromatic (vertices : ℕ → bool) (i : ℕ) : bool :=
¬ side_is_monochromatic vertices i

theorem regular_222_gon_no_equal_mono_dichromatic_ (vertices : ℕ → bool)
  (h: regular_n_gon 222) (hv: is_colored vertices) :
  ¬ (∃ m d : ℕ, m + d = 222 ∧ m = d ∧ (∀ i, side_is_monochromatic vertices i ∨ side_is_dichromatic vertices i)) :=
sorry

end regular_222_gon_no_equal_mono_dichromatic__l363_363258


namespace initial_men_count_l363_363213

theorem initial_men_count (M : ℕ) (P : ℝ) 
  (h1 : P = M * 12) 
  (h2 : P = (M + 300) * 9.662337662337663) :
  M = 1240 :=
sorry

end initial_men_count_l363_363213


namespace number_of_paths_from_to_l363_363354

def path_in_square (x y : ℤ) : Prop :=
  (-3 ≤ x ∧ x ≤ 3) ∧ (-3 ≤ y ∧ y ≤ 3)

def valid_step (p1 p2 : ℤ × ℤ) : Prop :=
  (p1.2 - p2.2 = 1 ∨ p2.2 - p1.2 = 1 ∨ p1.1 - p2.1 = 1 ∨ p2.1 - p1.1 = 1)

theorem number_of_paths_from_to (x y : ℤ) :
  ∃ (n : ℕ), 
    n = 41762 ∧ 
    n_steps_from_to (-6, -6) (6, 6) (24) ∧
    (∀ p, p ∈ path_satisfies_condition → ¬ path_in_square p.fst p.snd) :=
sorry

end number_of_paths_from_to_l363_363354


namespace income_final_amount_l363_363824

noncomputable def final_amount (income : ℕ) : ℕ :=
  let children_distribution := (income * 45) / 100
  let wife_deposit := (income * 30) / 100
  let remaining_after_distribution := income - children_distribution - wife_deposit
  let donation := (remaining_after_distribution * 5) / 100
  remaining_after_distribution - donation

theorem income_final_amount : final_amount 200000 = 47500 := by
  -- Proof omitted
  sorry

end income_final_amount_l363_363824


namespace find_a4_a5_l363_363996

variable {α : Type*} [LinearOrderedField α]

-- Variables representing the terms of the geometric sequence
variables (a₁ a₂ a₃ a₄ a₅ q : α)

-- Conditions given in the problem
-- Geometric sequence condition
def is_geometric_sequence (a₁ a₂ a₃ a₄ a₅ q : α) : Prop :=
  a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q ∧ a₅ = a₄ * q

-- First condition
def condition1 : Prop := a₁ + a₂ = 3

-- Second condition
def condition2 : Prop := a₂ + a₃ = 6

-- Theorem stating that a₄ + a₅ = 24 given the conditions
theorem find_a4_a5
  (h1 : condition1 a₁ a₂)
  (h2 : condition2 a₂ a₃)
  (hg : is_geometric_sequence a₁ a₂ a₃ a₄ a₅ q) :
  a₄ + a₅ = 24 := 
sorry

end find_a4_a5_l363_363996


namespace simplify_fraction_l363_363428

variable (x : ℤ)
hypothesis (hx : x ≠ 0)

theorem simplify_fraction :
  (x^3 - 3 * x^2 * (x + 2) + 4 * x * (x + 2)^2 - (x + 2)^3 + 2) / (x * (x + 2)) = 2 / (x * (x + 2)) := by
  sorry

end simplify_fraction_l363_363428


namespace quadratic_roots_interval_l363_363590

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l363_363590


namespace final_price_of_suit_after_discount_l363_363796

theorem final_price_of_suit_after_discount :
  let original_price := 200
  let increase_rate := 0.25
  let discount_rate := 0.25
  let increased_price := original_price * (1 + increase_rate)
  let final_price := increased_price * (1 - discount_rate)
  final_price = 187.5 :=
by
  let original_price := 200
  let increase_rate := 0.25
  let discount_rate := 0.25
  let increased_price := original_price * (1 + increase_rate)
  let final_price := increased_price * (1 - discount_rate)
  sorry

end final_price_of_suit_after_discount_l363_363796


namespace fourth_student_id_l363_363982

def class_size := 52
def sample_size := 4
def systematic_sample_interval (ids : List ℕ) : ℕ := match ids with
  | [] => 0
  | x :: y :: _ => y - x
  | _ => 0

def student_ids := [3, 29, 42]

theorem fourth_student_id :
  systematic_sample_interval student_ids = 26 → student_ids = [3, 29, 42] →
  List.get? (3 :: List.range (class_size - 1)) (3 + 3 * systematic_sample_interval student_ids) = some 16 :=
begin
  -- Proof here
  sorry
end

end fourth_student_id_l363_363982


namespace triangle_ratio_condition_l363_363979

theorem triangle_ratio_condition (a b c : ℝ) (A B C : ℝ) (h1 : b * Real.cos C + c * Real.cos B = 2 * b)
  (h2 : a = b * Real.sin A / Real.sin B)
  (h3 : b = a * Real.sin B / Real.sin A)
  (h4 : c = a * Real.sin C / Real.sin A)
  (h5 : ∀ x, Real.sin (B + C) = Real.sin x): 
  b / a = 1 / 2 :=
by
  sorry

end triangle_ratio_condition_l363_363979


namespace find_common_ratio_l363_363905

noncomputable def geometric_seq_sum (a₁ q : ℂ) (n : ℕ) :=
if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem find_common_ratio (a₁ q : ℂ) :
(geometric_seq_sum a₁ q 8) / (geometric_seq_sum a₁ q 4) = 2 → q = 1 :=
by
  intro h
  sorry

end find_common_ratio_l363_363905


namespace smoking_lung_cancer_problem_l363_363999

-- Defining the confidence relationship
def smoking_related_to_lung_cancer (confidence: ℝ) := confidence > 0.99

-- Statement 4: Among 100 smokers, it is possible that not a single person has lung cancer.
def statement_4 (N: ℕ) (p: ℝ) := N = 100 ∧ 0 ≤ p ∧ p ≤ 1 ∧ p ^ 100 > 0

-- The main theorem statement in Lean 4
theorem smoking_lung_cancer_problem (confidence: ℝ) (N: ℕ) (p: ℝ) 
  (h1: smoking_related_to_lung_cancer confidence): 
  statement_4 N p :=
by
  sorry -- Proof goes here

end smoking_lung_cancer_problem_l363_363999


namespace inequality_holds_for_m_l363_363033

theorem inequality_holds_for_m (m : ℝ) :
  (-2 : ℝ) ≤ m ∧ m ≤ (3 : ℝ) ↔ ∀ x : ℝ, x < -1 →
    (m - m^2) * (4 : ℝ)^x + (2 : ℝ)^x + 1 > 0 :=
by sorry

end inequality_holds_for_m_l363_363033


namespace trapezoid_area_l363_363744

theorem trapezoid_area (a : ℝ) 
  (base1 : ℝ := 25) 
  (height : ℝ := 13)
  (area : ℝ := 286) 
  : a = 19 :=
by {
  -- We'll use the provided information to set up the equation:
  -- 286 = (1 / 2) * (25 + a) * 13
  
  have area_formula : area = (1 / 2) * (base1 + a) * height, sorry,
  -- From this, we need to prove that a = 19.
  sorry
}

end trapezoid_area_l363_363744


namespace find_vector_from_origin_to_line_l363_363414

theorem find_vector_from_origin_to_line :
  ∃ t : ℝ, (3 * t + 1, 2 * t + 3) = (16, 32 / 3) ∧
  ∃ k : ℝ, (16, 32 / 3) = (3 * k, 2 * k) :=
sorry

end find_vector_from_origin_to_line_l363_363414


namespace log_base_2_of_4_squared_eq_4_l363_363878

theorem log_base_2_of_4_squared_eq_4 : log 2 (4 ^ 2) = 4 := by
  sorry

end log_base_2_of_4_squared_eq_4_l363_363878


namespace shortest_side_of_right_triangle_l363_363384

theorem shortest_side_of_right_triangle
  (a b c : ℝ)
  (h : a = 5) (k : b = 13) (rightangled : a^2 + c^2 = b^2) : c = 12 := 
sorry

end shortest_side_of_right_triangle_l363_363384


namespace chord_length_calc_final_chord_length_l363_363845

noncomputable def circle_center : ℝ × ℝ := (3, 0)
noncomputable def radius : ℝ := 3
noncomputable def line : ℝ × ℝ × ℝ := (2, -1, -2)
noncomputable def distance_from_center_to_line : ℝ :=
  (abs (2 * 3 + -1 * 0 - 2)) / (sqrt ((2:ℝ)^2 + (-1)^2))
noncomputable def chord_length : ℝ :=
  2 * sqrt (radius^2 - distance_from_center_to_line^2)

theorem chord_length_calc : chord_length = 2 * sqrt (3^2 - (4 * sqrt 5 / 5)^2) :=
by
  sorry

theorem final_chord_length : chord_length = 2 * sqrt 209 / 5 :=
by
  sorry

end chord_length_calc_final_chord_length_l363_363845


namespace university_box_cost_l363_363306

theorem university_box_cost (length width height : ℕ) (cost_per_box total_volume : ℕ) (h1 : length = 20) (h2 : width = 20) (h3 : height = 15) (h4 : cost_per_box = 90) (h5 : total_volume = 3060000) :
  let volume_of_one_box := length * width * height in
  let boxes_needed := (total_volume + volume_of_one_box - 1) / volume_of_one_box in
  (boxes_needed * cost_per_box) / 100 = 459 :=
by
  sorry

end university_box_cost_l363_363306


namespace probability_neither_alive_l363_363797

-- Define the probabilities for the man and his wife being alive for 10 more years.
def man_alive_10_years : ℝ := 1 / 4
def wife_alive_10_years : ℝ := 1 / 3

-- Define the probability of them being not alive for 10 more years.
def man_not_alive_10_years : ℝ := 1 - man_alive_10_years
def wife_not_alive_10_years : ℝ := 1 - wife_alive_10_years

-- Define the independence of their lifespans.
def independent_events : ℝ := man_not_alive_10_years * wife_not_alive_10_years

theorem probability_neither_alive :
  independent_events = 1 / 2 :=
by {
  -- Conditional definitions
  have h_man : man_not_alive_10_years = 3 / 4 := by sorry,
  have h_wife : wife_not_alive_10_years = 2 / 3 := by sorry,
  -- Calculation using independence and given probabilities
  calc
    independent_events = man_not_alive_10_years * wife_not_alive_10_years : by rfl
    ... = (3 / 4) * (2 / 3) : by rw [h_man, h_wife]
    ... = 1 / 2 : by norm_num
}

end probability_neither_alive_l363_363797


namespace average_large_basket_weight_l363_363365

-- Definitions derived from the conditions
def small_basket_capacity := 25  -- Capacity of each small basket in kilograms
def num_small_baskets := 28      -- Number of small baskets used
def num_large_baskets := 10      -- Number of large baskets used
def leftover_weight := 50        -- Leftover weight in kilograms

-- Statement of the problem
theorem average_large_basket_weight :
  (small_basket_capacity * num_small_baskets - leftover_weight) / num_large_baskets = 65 :=
by
  sorry

end average_large_basket_weight_l363_363365


namespace max_teams_advance_l363_363121

def teams := {1, 2, 3, 4, 5, 6, 7, 8}

def total_games := (8 * 7) / 2

def total_points_distributed := 28 * 3

def max_teams_qualifying := 84 / 15

theorem max_teams_advance :
  ∃ n, n ≤ 5 ∧ n * 15 ≤ 84 ∧ n * 15 > 60 := 
sorry

end max_teams_advance_l363_363121


namespace remainder_when_690_div_170_l363_363239

theorem remainder_when_690_div_170 :
  ∃ r : ℕ, ∃ k l : ℕ, 
    gcd (690 - r) (875 - 25) = 170 ∧
    r = 690 % 170 ∧
    l = 875 / 170 ∧
    r = 10 :=
by 
  sorry

end remainder_when_690_div_170_l363_363239


namespace geometric_sum_of_ratios_l363_363169

theorem geometric_sum_of_ratios (k p r : ℝ) (a2 a3 b2 b3 : ℝ) 
  (ha2 : a2 = k * p) (ha3 : a3 = k * p^2) 
  (hb2 : b2 = k * r) (hb3 : b3 = k * r^2) 
  (h : a3 - b3 = 5 * (a2 - b2)) :
  p + r = 5 :=
by {
  sorry
}

end geometric_sum_of_ratios_l363_363169


namespace quadratic_real_roots_l363_363577

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l363_363577


namespace proof_problem_l363_363449

def S (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k => 1 / (k + 1 : ℝ))

def T (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k => S (k + 1))

def U (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k => (1 / (k + 2 : ℝ)) * T (k + 1))

theorem proof_problem : 
  ∃ a b c d : ℕ, 
  0 < a ∧ a < 1000000 ∧ 0 < b ∧ b < 1000000 ∧ 0 < c ∧ c < 1000000 ∧ 0 < d ∧ d < 1000000 ∧
  a = 1989 ∧ b = 1989 ∧ c = 1990 ∧ d = 3978 ∧ 
  T 1988 = a * S 1989 - b ∧ 
  U 1988 = c * S 1989 - d :=
begin
  -- We state the correct values directly.
  use [1989, 1989, 1990, 3978],
  -- Verify the constraints and the equalities
  repeat { split; try { norm_num }, },
  sorry,  -- Proof to be completed
end

end proof_problem_l363_363449


namespace find_BD_correct_l363_363643

variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (a b c d : A) (ab bd : ℝ)

def triangle_AC_BC_equal (a b c : A) : A :=
  AC = 10 ∧ BC = 10

def point_on_AB (a b d : A) : Prop :=
  ∃ t ∈ segment (A,B), d = (1 - t) * a + t * b

def find_BD (a b c : A) (d_bd : ℝ) : Prop :=
  (BD = 7.85)

theorem find_BD_correct (a b c d : A) (h1 : AC = 10) (h2 : BC = 10) (h3 : AD = 12) (h4 : CD = 5) (h5 : point_on_AB a b d) :
  BD = 7.85 :=
sorry

end find_BD_correct_l363_363643


namespace trigonometric_expression_value_l363_363854

theorem trigonometric_expression_value :
  (tan (real.pi / 6))^2 - (cos (real.pi / 6))^2) / ( (tan (real.pi / 6))^2 * (cos (real.pi / 6))^2) = 1 / 3 := 
by 
 sorry

end trigonometric_expression_value_l363_363854


namespace hyperbola_equation_slope_ratio_immutable_l363_363906

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
    (H1 : 2 * b = 4) (H2 : (b / a) = 2) :
  (a = 1 ∧ b = 2 ∧ (∀ x y : ℝ, (x^2 - (y^2 / (2 : ℝ)^2)) = 1)) :=
by
  sorry

theorem slope_ratio_immutable
    -- Define conditions for hyperbola and vertices A and B
    (C : { x y : ℝ // (x^2 - y^2/4 = 1) })
    (A B : ℝ × ℝ)
    (H3 : A = (-1, 0)) (H4 : B = (1, 0))
    -- Define point T
    (T : ℝ × ℝ) (H5 : T = (2, 0))
    -- Define slopes k1 and k2 for lines passing T and vertices
    (k1 k2 : ℝ) 
    (H6 : ∀ y1 y2 x1 x2 : ℝ, 
        (x1 = 1) → (x2 = -1) → 
        C.val.1 = 1 → 
        C.val.2 = 2 →
        y1 + y2 = -(16 * (k1) / (4 * k1 ^ 2 - 1)) →
        y1 * y2 = 12 / (4 * k1 ^ 2 - 1) → 
        k1 = y1 / (x1 + 1) → 
        k2 = y2 / (x2 - 1)) :
  (k1 / k2 = -1 / 3) :=
by
  sorry

end hyperbola_equation_slope_ratio_immutable_l363_363906


namespace quadratic_roots_interval_l363_363591

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l363_363591


namespace total_books_l363_363784

variable (a : ℕ)

theorem total_books (h₁ : 5 = 5) (h₂ : a = a) : 5 + a = 5 + a :=
by
  sorry

end total_books_l363_363784


namespace find_number_A_l363_363253

theorem find_number_A (A B : ℝ) (h₁ : A + B = 14.85) (h₂ : B = 10 * A) : A = 1.35 :=
sorry

end find_number_A_l363_363253


namespace greatest_distance_is_correct_l363_363191

-- Define the coordinates of the post.
def post_coordinate : ℝ × ℝ := (6, -2)

-- Define the length of the rope.
def rope_length : ℝ := 12

-- Define the origin.
def origin : ℝ × ℝ := (0, 0)

-- Define the formula to calculate the Euclidean distance between two points in ℝ².
noncomputable def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ := by
  sorry

-- Define the distance from the origin to the post.
noncomputable def distance_origin_to_post : ℝ := euclidean_distance origin post_coordinate

-- Define the greatest distance the dog can be from the origin.
noncomputable def greatest_distance_from_origin : ℝ := distance_origin_to_post + rope_length

-- Prove that the greatest distance the dog can be from the origin is 12 + 2 * sqrt 10.
theorem greatest_distance_is_correct : greatest_distance_from_origin = 12 + 2 * Real.sqrt 10 := by
  sorry

end greatest_distance_is_correct_l363_363191


namespace quadratic_has_real_root_iff_b_in_interval_l363_363541

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l363_363541


namespace bisection_method_algorithm_structures_l363_363028

theorem bisection_method_algorithm_structures :
  ∀ (f : ℝ → ℝ) (a b : ℝ), 
  (f = λ x => x^2 - 10) →
  (a < b ∧ f a * f b < 0) →
  ∃ (algorithm_structure : Type) (sequential_structure conditional_structure loop_structure : algorithm_structure),
    bisection_method_exists f a b sequential_structure ∧
    bisection_method_exists f a b conditional_structure ∧
    bisection_method_exists f a b loop_structure :=
sorry

end bisection_method_algorithm_structures_l363_363028


namespace ratio_of_sums_l363_363997

noncomputable def curve_intersection_ratio (a b : ℝ) : ℝ :=
  a / b

theorem ratio_of_sums :
  ∀ (a b : ℝ),
    (∀ (x y : ℝ), y = cos x ∧ x = 100 * cos (100 * y) → 
      (x > 0 ∧ y > 0) → 
      ∃ a b : ℝ, 
        a = 100 * b) →
  curve_intersection_ratio a b = 100 :=
by
  intros a b h
  have h_eq : a = 100 * b := sorry
  unfold curve_intersection_ratio
  rw h_eq
  exact eq_of_div_eq_one_right h_eq

end ratio_of_sums_l363_363997


namespace evaluate_expression_l363_363419

theorem evaluate_expression :
  let a := 2020
  let b := 2016
  (2^a + 2^b) / (2^a - 2^b) = 17 / 15 :=
by
  sorry

end evaluate_expression_l363_363419


namespace lattice_points_10D_l363_363042

theorem lattice_points_10D (a b c d e f g h i j : ℤ)
  (h1 : a^2 + b^2 + c^2 + d^2 + e^2 + f^2 + g^2 + h^2 + i^2 + j^2 = 9) :
  -- Here, we should prove that this equation has exactly 1720 integer solutions.
  sorry

end lattice_points_10D_l363_363042


namespace parabola_equation_l363_363233

-- Definitions of the given conditions
def passes_through_point (x y : ℝ) := (1, 5) = (x, y)
def focus_y_coord (focus_y : ℝ) := focus_y = 3
def axis_parallel_x := true -- Given as a condition
def vertex_on_y_axis (vx vy : ℝ) := vx = 0

-- Statement to be proven
theorem parabola_equation :
  (∃ k : ℝ, (passes_through_point (k * (5 - focus_y_coord 3)^2) 5) ∧ (focus_y_coord 3) ∧ axis_parallel_x ∧ (vertex_on_y_axis 0 3)) →
  (∀ x y : ℝ, y^2 - 4 * x - 6 * y + 9 = 0):=
begin
  intro h,
  sorry
end

end parabola_equation_l363_363233


namespace trajectory_of_P_l363_363453

-- Define the points and circle.
def A := (-1 / 2 : ℝ, 0 : ℝ)
def F_center := (1 / 2 : ℝ, 0 : ℝ)
def F_radius := 2 : ℝ

-- Define the statement to prove based on the given conditions.
theorem trajectory_of_P :
  ∃ (P : ℝ × ℝ), 
    (P.fst - F_center.fst) ^ 2 + P.snd ^ 2 = 1 →
    (P.fst ^ 2 + (4 / 3) * P.snd ^ 2 = 1) :=
by
  sorry

end trajectory_of_P_l363_363453


namespace philip_animals_on_farm_l363_363196

open Nat

theorem philip_animals_on_farm :
  let cows := 20
  let ducks := cows + (cows / 2)
  let horses := (cows + ducks) / 5
  let pigs := (cows + ducks + horses) / 5
  let chickens := 3 * (horses - cows)
  cows + ducks + horses + pigs + chickens = 102 :=
by
  let cows := 20
  let ducks := cows + (cows / 2)
  let horses := ((cows + ducks) * 2) / 10
  let pigs := ((cows + ducks + horses) * 2) / 10
  let chickens := 3 * (horses - cows)
  calc
    cows + ducks + horses + pigs + chickens
      = 20 + (20 + 10) + 10 + 12 + 30 : by
        simp [cows, ducks, horses, pigs, chickens]
    ... = 102 : by
        norm_num

end philip_animals_on_farm_l363_363196


namespace f_periodic_odd_l363_363494

noncomputable def f (x : ℝ) : ℝ := sorry
def a_seq : ℕ → ℝ
| 1 := -1
| (n + 1) := 2 * a_seq n + n + 2

def S_seq (n : ℕ) : ℝ := (List.sum (List.range n).map a_seq)

theorem f_periodic_odd :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, f (3 / 2 - x) = f x) ∧
  f (-2) = -3 ∧
  ∀ n : ℕ, (n > 0) →
           (S_seq n / n = 2 * a_seq n / n + 1) →
           (f (a_seq 5) + f (a_seq 6) = 3) :=
by { sorry }

end f_periodic_odd_l363_363494


namespace find_a_l363_363165

variable (a : ℕ) (N : ℕ)
variable (h1 : Nat.gcd (2 * a + 1) (2 * a + 2) = 1) 
variable (h2 : Nat.gcd (2 * a + 1) (2 * a + 3) = 1)
variable (h3 : Nat.gcd (2 * a + 2) (2 * a + 3) = 2)
variable (hN : N = Nat.lcm (2 * a + 1) (Nat.lcm (2 * a + 2) (2 * a + 3)))
variable (hDiv : (2 * a + 4) ∣ N)

theorem find_a (h_pos : a > 0) : a = 1 :=
by
  -- Lean proof code will go here
  sorry

end find_a_l363_363165


namespace domain_of_g_cauchy_schwarz_inequality_l363_363049

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Question 1: Prove the domain of g(x) = log(f(x) - 2) is {x | 0.5 < x < 2.5}
theorem domain_of_g : {x : ℝ | 0.5 < x ∧ x < 2.5} = {x : ℝ | 0.5 < x ∧ x < 2.5} :=
by
  sorry

-- Minimum value of f(x)
def m : ℝ := 1

-- Question 2: Prove a^2 + b^2 + c^2 ≥ 1/3 given a + b + c = m
theorem cauchy_schwarz_inequality (a b c : ℝ) (h : a + b + c = m) : a^2 + b^2 + c^2 ≥ 1 / 3 :=
by
  sorry

end domain_of_g_cauchy_schwarz_inequality_l363_363049


namespace symmetric_points_l363_363114

theorem symmetric_points (m n : ℝ) 
  (h1 : (-m, 3) = y_axis_symmetry (-5, n)) : m = -5 ∧ n = 3 := by
  sorry

end symmetric_points_l363_363114


namespace geometric_sequence_sum_l363_363986

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) ^ 2 = a n * a (n + 2))
  (h_pos : ∀ n, 0 < a n) (h_given : a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16) :
  a 3 + a 6 = 4 := 
sorry

end geometric_sequence_sum_l363_363986


namespace find_f_neg_5_l363_363924

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_domain : ∀ x : ℝ, true)
variable (h_positive : ∀ x : ℝ, x > 0 → f x = log 5 x + 1)

theorem find_f_neg_5 : f (-5) = -2 :=
by
  sorry

end find_f_neg_5_l363_363924


namespace sum_first_8_terms_arithmetic_sequence_l363_363132

theorem sum_first_8_terms_arithmetic_sequence (a : ℕ → ℝ) (h : a 4 + a 5 = 12) :
    (8 * (a 1 + a 8)) / 2 = 48 :=
by
  sorry

end sum_first_8_terms_arithmetic_sequence_l363_363132


namespace point_on_graph_l363_363235

theorem point_on_graph (x y : ℝ) (h : y = 3 * x + 1) : (x, y) = (2, 7) :=
sorry

end point_on_graph_l363_363235


namespace sum_exp_function_max_min_value_l363_363745

open Real

theorem sum_exp_function_max_min_value (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1)
  (h₃ : ∃ x ∈ Icc (0 : ℝ) 2, 2 * a * x - 1 = 7) :
  (max (a^0) + min (a^3)) = 9 :=
by
  sorry

end sum_exp_function_max_min_value_l363_363745


namespace angle_DAE_measurement_l363_363399

-- Definitions based on the problem conditions
def is_equilateral_triangle (A B C : Point) : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C A)

def is_regular_pentagon (B C D E F : Point) : Prop :=
  (dist B C = dist C D) ∧ (dist C D = dist D E) ∧ 
  (dist D E = dist E F) ∧ (dist E F = dist F B) ∧
  (∠ B C D = 108) ∧ (∠ C D E = 108) ∧ (∠ D E F = 108) ∧ (∠ E F B = 108) ∧ (∠ F B C = 108)

-- The target proof statement
theorem angle_DAE_measurement
  (A B C D E F : Point)
  (h1 : is_equilateral_triangle A B C)
  (h2 : is_regular_pentagon B C D E F)
  (h3 : dist B C ≠ 0) :
  ∠ D A E = 108 :=
sorry

end angle_DAE_measurement_l363_363399


namespace initial_discount_l363_363849

theorem initial_discount (P D : ℝ) 
  (h1 : P - 71.4 = 5.25)
  (h2 : P * (1 - D) * 1.25 = 71.4) : 
  D = 0.255 :=
by {
  sorry
}

end initial_discount_l363_363849


namespace solve_ODE_1_solve_ODE_2_solve_ODE_3_solve_ODE_4_solve_ODE_5_solve_ODE_6_l363_363210

-- 1. Solve y'' - 5y' - 6y = 0 
theorem solve_ODE_1 (C1 C2 : ℝ) (y : ℝ → ℝ) :
  (∀ x, y x = C1 * Real.exp (6 * x) + C2 * Real.exp (-x)) →
  ∀ x, ∂² (y x) ∂ x² - 5 * ∂ (y x) ∂ x - 6 * y x = 0 :=
by
  sorry

-- 2. Solve y''' - 6y'' + 13y' = 0 
theorem solve_ODE_2 (C1 C2 C3 : ℝ) (y : ℝ → ℝ) :
  (∀ x, y x = C1 + Real.exp (3 * x) * (C2 * Real.cos (2 * x) + C3 * Real.sin (2 * x))) →
  ∀ x, ∂³ (y x) ∂ x³ - 6 * ∂² (y x) ∂ x² + 13 * ∂ (y x) ∂ x = 0 :=
by
  sorry

-- 3. Solve d^2 S / dt^2 + 4 dS / dt + 4S = 0
theorem solve_ODE_3 (C1 C2 : ℝ) (S : ℝ → ℝ) :
  (∀ t, S t = Real.exp (-2 * t) * (C1 + C2 * t)) →
  ∀ t, ∂² (S t) ∂ t² + 4 * ∂ (S t) ∂ t + 4 * S t = 0 :=
by
  sorry

-- 4. Solve d^4 y / dx^4 - y = 0 
theorem solve_ODE_4 (C1 C2 C3 C4 : ℝ) (y : ℝ → ℝ) :
  (∀ x, y x = C1 * Real.exp x + C2 * Real.exp (-x) + C3 * Real.cos x + C4 * Real.sin x) →
  ∀ x, ∂⁴ (y x) ∂ x⁴ - y x = 0 :=
by
  sorry

-- 5. Solve y^4 + 13 y^2 + 36 y = 0 
theorem solve_ODE_5 (C1 C2 C3 C4 : ℝ) (y : ℝ → ℝ) :
  (∀ x, y x = C1 * Real.cos (2 * x) + C2 * Real.sin (2 * x) + C3 * Real.cos (3 * x) + C4 * Real.sin (3 * x)) →
  ∀ x, ∂⁴ (y x) ∂ x⁴ + 13 * ∂² (y x) ∂ x² + 36 * y x = 0 :=
by
  sorry

-- 6. Solve y^2 + 2 y^5 + y^3 = 0 
theorem solve_ODE_6 (C1 C2 C3 C4 C5 C6 C7: ℝ) (y : ℝ → ℝ) :
  (∀ x, y x = C1 + C2 * x + C3 * x^2 + (C4 + C5 * x) * Real.cos x + (C6 + C7 * x) * Real.sin x) →
  ∀ x, ∂ (y x) ∂ x^2 + 2 * ∂ (y x) ∂ x^5 + ∂ (y x) ∂ x^3 = 0 :=
by
  sorry

end solve_ODE_1_solve_ODE_2_solve_ODE_3_solve_ODE_4_solve_ODE_5_solve_ODE_6_l363_363210


namespace counterexample_l363_363004

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_composite (n : ℕ) : Prop := ∃ d : ℕ, 1 < d ∧ d < n ∧ n % d = 0
def is_prime (n : ℕ) : Prop := n > 1 ∧ ¬(∃ d : ℕ, 1 < d ∧ d < n ∧ n % d = 0)

theorem counterexample : 
  let n := 25 in 
  is_odd n ∧ is_composite n ∧ ¬ is_prime (n + 2) := 
by 
  let n := 25
  have h1 : is_odd n := by sorry
  have h2 : is_composite n := by sorry
  have h3 : ¬ is_prime (n + 2) := by sorry
  exact ⟨h1, h2, h3⟩

end counterexample_l363_363004


namespace length_of_smaller_box_l363_363614

theorem length_of_smaller_box 
  (L : ℕ)
  (lg_box_len : ℕ := 600) (lg_box_wid : ℕ := 500) (lg_box_hei : ℕ := 400)
  (sm_box_wid : ℕ := 50) (sm_box_hei : ℕ := 40)
  (max_boxes : ℕ := 1000)
  (h1 : 1000 * (L * 50 * 40) ≤ 600 * 500 * 400) : 
  L = 60 := 
begin 
  -- Proof steps go here 
  sorry 
end

end length_of_smaller_box_l363_363614


namespace find_intersection_points_l363_363294

theorem find_intersection_points (x1 x2 : ℝ) (h : x1 < x2) (hx1 : x1 = 0) (hx2 : x2 = 2) :
  let yC := (2/3) * Real.exp x1 + (1/3) * Real.exp x2
  let yD := (1/3) * Real.exp x1 + (2/3) * Real.exp x2
  x3 = Real.log ((2/3) + (1/3) * Real.exp 2) ∧
  x4 = Real.log ((1/3) + (2/3) * Real.exp 2) :=
by
  intros
  have : yC = (2/3) * Real.exp 0 + (1/3) * Real.exp 2 := by 
    rw [hx1, hx2]
  have : yD = (1/3) * Real.exp 0 + (2/3) * Real.exp 2 := by 
    rw [hx1, hx2]
  exact ⟨
    show x3 = Real.log ((2/3) + (1/3) * Real.exp 2),
    sorry,
    show x4 = Real.log ((1/3) + (2/3) * Real.exp 2),
    sorry⟩

end find_intersection_points_l363_363294


namespace right_angled_triangles_with_cathetus_2021_l363_363957

theorem right_angled_triangles_with_cathetus_2021 :
  ∃ n : Nat, n = 4 ∧ ∀ (a b c : ℕ), ((a = 2021 ∧ a * a + b * b = c * c) ↔ (a = 2021 ∧ 
    ∃ m n, (m > n ∧ m > 0 ∧ n > 0 ∧ 2021 = m^2 - n^2 ∧ b = 2 * m * n ∧ c = m^2 + n^2))) :=
sorry

end right_angled_triangles_with_cathetus_2021_l363_363957


namespace digit_A_of_3AA1_divisible_by_9_l363_363228

theorem digit_A_of_3AA1_divisible_by_9 (A : ℕ) (h : (3 + A + A + 1) % 9 = 0) : A = 7 :=
sorry

end digit_A_of_3AA1_divisible_by_9_l363_363228


namespace triangle_segment_length_l363_363145

theorem triangle_segment_length (a b c : ℕ) (r s : ℚ) (h_ratio : a : b : c = 3 : 4 : 5)
  (length_EG : ℚ) (h_length_EG : length_EG = 12)
  (angle_bisector_FH : r : s = 3 : 5) :
  max r s = 15 / 2 :=
by
  sorry

end triangle_segment_length_l363_363145


namespace range_of_f_less_than_2008_l363_363667

noncomputable def f : ℕ → ℕ := sorry -- function definition is omitted as it's derived from conditions

axiom condition1 : ∀ n : ℕ, (f (2 * n + 1))^2 - (f (2 * n))^2 = 6 * f n + 1
axiom condition2 : ∀ n : ℕ, f (2 * n) ≥ f n

theorem range_of_f_less_than_2008 : {y : ℕ | ∃ x : ℕ, f x = y ∧ y < 2008}.to_finset.card = 128 :=
by sorry -- Proof is omitted

end range_of_f_less_than_2008_l363_363667


namespace efficiency_and_days_l363_363699

noncomputable def sakshi_efficiency : ℝ := 1 / 25
noncomputable def tanya_efficiency : ℝ := 1.25 * sakshi_efficiency
noncomputable def ravi_efficiency : ℝ := 0.70 * sakshi_efficiency
noncomputable def combined_efficiency : ℝ := sakshi_efficiency + tanya_efficiency + ravi_efficiency
noncomputable def days_to_complete_work : ℝ := 1 / combined_efficiency

theorem efficiency_and_days:
  combined_efficiency = 29.5 / 250 ∧
  days_to_complete_work = 250 / 29.5 :=
by
  sorry

end efficiency_and_days_l363_363699


namespace neg_p_sufficient_for_neg_q_l363_363902

def p (x : ℝ) : Prop := |2 * x - 3| > 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem neg_p_sufficient_for_neg_q :
  (∀ x, ¬ p x → ¬ q x) ∧ ¬ (∀ x, ¬ q x → ¬ p x) :=
by
  -- Placeholder to indicate skipping the proof
  sorry

end neg_p_sufficient_for_neg_q_l363_363902


namespace b_share_is_approx_1885_71_l363_363786

noncomputable def investment_problem (x : ℝ) : ℝ := 
  let c_investment := x
  let b_investment := (2 / 3) * c_investment
  let a_investment := 3 * b_investment
  let total_investment := a_investment + b_investment + c_investment
  let b_share := (b_investment / total_investment) * 6600
  b_share

theorem b_share_is_approx_1885_71 (x : ℝ) : abs (investment_problem x - 1885.71) < 0.01 := sorry

end b_share_is_approx_1885_71_l363_363786


namespace angle_equality_l363_363753

-- Given a triangle ABC with an acute angle at A,
-- O is the center of the circumcircle, and AH is the altitude from A to BC,
-- prove that ∠BAH = ∠OAC
theorem angle_equality (A B C O H : Type) [triangle A B C] 
  (circumcenter : center O (circumcircle A B C)) 
  (altitude : height A H (line B C)) :
  ∠BAH = ∠OAC := 
sorry

end angle_equality_l363_363753


namespace max_angle_line_l363_363127

-- Define the planes and the point of intersection
variables (τ σ : Plane) (A : Point)
-- Assume the planes intersect at line AK
variables (AK : Line)
-- Assume A lies on line AK and is the intersection point of planes τ and σ
axiom IntersectingPlanes (h1 : A ∈ τ ∧ A ∈ σ ∧ A ∈ AK) (h2 : τ ≠ σ)

-- Proof statement
theorem max_angle_line
(hτ : contains τ AK)
(hσ : contains σ AK)
(hA : A ∈ τ ∧ A ∈ σ) :
∀ (L : Line), (L ∈ τ ∧ A ∈ L) → 
(∀ (x : Line), (x ∈ τ ∧ A ∈ x) → angle_between_line_and_plane L σ ≥ angle_between_line_and_plane x σ) → 
is_perpendicular L AK
:= sorry

end max_angle_line_l363_363127


namespace find_a20_l363_363623

variable {a : ℕ → ℝ}

-- The sequence {a_n} is an arithmetic sequence with common difference d
def arithmetic_sequence (d : ℝ) := ∀ n, a (n + 1) = a n + d

-- The sequence is increasing
def increasing_arithmetic_sequence := ∀ n, a n < a (n + 1)

-- The roots of the quadratic equation
def root_of_quadratic (x : ℝ) := x^2 - 10 * x + 24 = 0

-- Conditions based on the problem statement
def conditions : Prop :=
  increasing_arithmetic_sequence ∧
  root_of_quadratic (a 4) ∧
  root_of_quadratic (a 6)

-- The theorem to prove
theorem find_a20 (h : conditions) : a 20 = 20 :=
sorry

end find_a20_l363_363623


namespace greatest_three_digit_multiple_of_thirteen_l363_363770

theorem greatest_three_digit_multiple_of_thirteen : 
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (13 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ (13 ∣ m) → m ≤ n) ∧ n = 988 :=
  sorry

end greatest_three_digit_multiple_of_thirteen_l363_363770


namespace acute_triangle_angles_l363_363743

/-- Given an acute triangle where the largest angle is five times the smallest angle,
and knowing all angles are whole numbers, prove that the angles of the triangle are 
17 degrees, 78 degrees, and 85 degrees. -/
theorem acute_triangle_angles (
  (α β γ : ℕ) 
  (h1: α ≥ β)
  (h2: β ≥ γ)
  (h3: α = 5 * γ)
  (h_sum: α + β + γ = 180)
  (h_acute: α < 90)
): 
  α = 85 ∧ β = 78 ∧ γ = 17 := 
by {
  sorry
}

end acute_triangle_angles_l363_363743


namespace find_a_l363_363113

theorem find_a (a m : ℝ) (h1 : 0 < a ∧ a ≠ 1)
  (h2 : ∀ x ∈ Icc (-1 : ℝ) 2, f x = a^x → f x ≤ 4) 
  (h3 : ∀ x ∈ Icc (-1 : ℝ) 2, f x = a^x → f x ≥ m)
  (h4 : ∀ x ∈ Icc (0 : ℝ) (0 : ℝ) ∞, g x = (1 - 4 * m) * (sqrt x) → x ≤ infinity)
: a = (1 / 4) :=
by
  sorry

end find_a_l363_363113


namespace insurance_calculation_l363_363340

def loan_amount : ℝ := 20000000
def appraisal_value : ℝ := 14500000
def cadastral_value : ℝ := 15000000
def basic_tariff : ℝ := 0.2 / 100
def coefficient_no_transition : ℝ := 0.8
def coefficient_no_certificates : ℝ := 1.3

noncomputable def adjusted_tariff : ℝ := basic_tariff * coefficient_no_transition * coefficient_no_certificates
noncomputable def insured_amount : ℝ := max appraisal_value cadastral_value
noncomputable def insurance_premium : ℝ := insured_amount * adjusted_tariff

theorem insurance_calculation :
  adjusted_tariff = 0.00208 ∧ insurance_premium = 31200 := 
by
  sorry

end insurance_calculation_l363_363340


namespace abs_eq_inequality_l363_363108

theorem abs_eq_inequality (m : ℝ) (h : |m - 9| = 9 - m) : m ≤ 9 :=
sorry

end abs_eq_inequality_l363_363108


namespace greatest_possible_xy_sum_l363_363377

theorem greatest_possible_xy_sum (a b c d e x y : ℕ) 
  (h_pair_sums : multiset.sum ({150, 273, 230, x, y, 350, 176, 290, 312, 405} : multiset ℕ) = 5 * (a + b + c + d + e) / 2) :
  x + y ≤ 630 :=
sorry

end greatest_possible_xy_sum_l363_363377


namespace range_of_x_l363_363935

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 else real.log (x + 1)

theorem range_of_x {x : ℝ} : f (2 - x^2) > f x ↔ -2 < x ∧ x < 1 :=
by sorry

end range_of_x_l363_363935


namespace solve_system_l363_363882

def F (t : ℝ) : ℝ := 32 * t ^ 5 + 48 * t ^ 3 + 17 * t - 15

def system_of_equations (x y z : ℝ) : Prop :=
  (1 / x = (32 / y ^ 5) + (48 / y ^ 3) + (17 / y) - 15) ∧
  (1 / y = (32 / z ^ 5) + (48 / z ^ 3) + (17 / z) - 15) ∧
  (1 / z = (32 / x ^ 5) + (48 / x ^ 3) + (17 / x) - 15)

theorem solve_system : ∃ (x y z : ℝ), system_of_equations x y z ∧ x = 2 ∧ y = 2 ∧ z = 2 :=
by
  sorry -- Proof not included

end solve_system_l363_363882


namespace chord_length_squared_l363_363983

-- Define configuration and conditions
def radius1 : ℝ := 7
def radius2 : ℝ := 9
def distance_between_centers : ℝ := 14
def P : ℝ × ℝ := (0, 0)  -- Just a symbol for point of intersection

-- Define the chords with lengths of equal length x
variables (Q R : ℝ × ℝ) -- Points Q and R in the plane
-- Chords lengths are equal, let x be the length of QP and PR
variable (x : ℝ)

-- Assume the conditions given directly in the problem
def condition1 := dist (fst Q) 0 = radius1
def condition2 := dist (snd R) 0 = radius2
def condition3 := dist (fst P) (snd P) = distance_between_centers
def condition4 := dist Q P = x
def condition5 := dist P R = x

-- The final proof statement
theorem chord_length_squared :
  condition1 →
  condition2 →
  condition3 →
  condition4 →
  condition5 →
  x ^ 2 = 145 :=
sorry

end chord_length_squared_l363_363983


namespace selection_ways_l363_363202

theorem selection_ways (boys girls total_selection : ℕ) 
  (at_least_one_boy : boys > 0) (at_least_one_girl : girls > 0) 
  (total_boys_and_girls : boys + girls = 12) 
  (four_selections : total_selection = 4) :
  (nat.choose 12 4) - (nat.choose boys 4) - (nat.choose girls 4) = 
      nat.choose (boys + girls) total_selection - 
      nat.choose boys total_selection - 
      nat.choose girls total_selection :=
by
  sorry

end selection_ways_l363_363202


namespace calf_rope_length_increase_l363_363828

noncomputable def new_rope_length (r : ℝ) (additional_area : ℝ) : ℝ :=
  sqrt (r^2 + additional_area / π)

theorem calf_rope_length_increase :
  new_rope_length 16 858 = 23 :=
by
  sorry

end calf_rope_length_increase_l363_363828


namespace cannot_form_regular_hexagon_l363_363374

-- Definitions of the conditions
def regular_hexagonal_base : Type := sorry -- A type representing the regular hexagonal base
def square_lateral_faces : Type := sorry -- A type representing the square lateral faces
def at_least_three_non_collinear_vertices (p: Type) : Prop :=
-- A condition representing that the plane intersects at least three vertices not all on the same face/base
  sorry 

-- Definition of the cross-section property
def cross_section_of_planes_intersecting_prism (p: Type) : Type := sorry

-- Statement of the theorem
theorem cannot_form_regular_hexagon (p : Type)
  (H1 : regular_hexagonal_base p)
  (H2 : square_lateral_faces p)
  (H3 : at_least_three_non_collinear_vertices p) :
  ¬ cross_section_of_planes_intersecting_prism p = regular_hexagon := 
sorry

end cannot_form_regular_hexagon_l363_363374


namespace power_function_solution_l363_363741

theorem power_function_solution (m : ℝ) 
    (h1 : m^2 - 3 * m + 3 = 1) 
    (h2 : m - 1 ≠ 0) : m = 2 := 
by
  sorry

end power_function_solution_l363_363741


namespace range_of_a_l363_363500

def f (a : ℝ) : ℝ → ℝ
| x => if x ≥ 0 then x^2 - 2*a*x - a + 1 else Real.log (-x)

def g (a : ℝ) (x : ℝ) : ℝ := x^2 + 1 - 2*a

def y (a : ℝ) (x : ℝ) : ℝ := f a (g a x)

def has_four_zeros (a : ℝ) : Prop :=
  ∃ xs : Finset ℝ, xs.card = 4 ∧ ∀ x ∈ xs, y a x = 0

theorem range_of_a :
  ∀ a : ℝ, has_four_zeros a ↔ (a ∈ Set.Ioo (-1 + Real.sqrt 5 / 2) 1 ∪ Set.Ioi 1) :=
by
  sorry

end range_of_a_l363_363500


namespace initial_principal_amount_approximation_l363_363373

noncomputable def compound_interest (A : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  A / (1 + r / n)^(n * t)

theorem initial_principal_amount_approximation :
  let A := 4813
  let r := 0.0625
  let n := 365
  let t := 3
  compound_interest A r n t ≈ 4020.04 :=
by
  sorry

end initial_principal_amount_approximation_l363_363373


namespace max_segments_no_triangles_l363_363463

-- Define the main theorem using the conditions provided
theorem max_segments_no_triangles (n : ℕ) (h : ∀ (points : Finset (ℝ × ℝ)), points.card = n → 
  (∀ (p1 p2 p3 : (ℝ × ℝ)), p1 ≠ p2 → p2 ≠ p3 → p3 ≠ p1 → p1 ≠ p3 → 
  ¬ collinear {p1, p2, p3})) 
  (h_seg : ∀ (segments : Finset (ℝ × ℝ × ℝ × ℝ)), (∀ (s1 s2 s3 : (ℝ × ℝ × ℝ × ℝ)),
  s1 ∈ segments → s2 ∈ segments → s3 ∈ segments →
  ¬ forms_triangle s1 s2 s3)) :
  ∃ S : ℕ, S ≤ ⌊ (n * n) / 4 ⌋ := 
begin
  sorry
end

end max_segments_no_triangles_l363_363463


namespace find_q_l363_363103

theorem find_q {q : ℕ} (h : 27^8 = 9^q) : q = 12 := by
  sorry

end find_q_l363_363103


namespace single_digit_geometric_sequence_l363_363868

-- Define the single-digit natural numbers
def is_single_digit (n : ℕ) : Prop := n < 10

-- Representation of the numbers in the geometric sequence
def overline (a c : ℕ) : ℕ := 10 * a + c

theorem single_digit_geometric_sequence :
  ∃ (a b c : ℕ),
  is_single_digit a ∧ is_single_digit b ∧ is_single_digit c ∧
  ∃ (r : ℕ), r > 0 ∧ b = a * r ∧ overline a c = b * r :=
begin
  use ((1 : ℕ), (4 : ℕ), (6 : ℕ)),
  split,
  { -- a is a single-digit number
    unfold is_single_digit,
    exact dec_trivial,
  },
  split,
  { -- b is a single-digit number
    unfold is_single_digit,
    exact dec_trivial,
  },
  split,
  { -- c is a single-digit number
    unfold is_single_digit,
    exact dec_trivial,
  },
  use (4 : ℕ),
  split,
  { -- r is a positive number
   exact dec_trivial,
  },
  split,
  { -- b = a * r
    exact dec_trivial,
  },
  { -- overline a c = b * r
    unfold overline,
    exact dec_trivial,
  },
end

end single_digit_geometric_sequence_l363_363868


namespace ratio_of_radii_l363_363637

variables (a b : ℝ) (h : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2)

theorem ratio_of_radii (h : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) : 
  a / b = Real.sqrt 5 / 5 :=
sorry

end ratio_of_radii_l363_363637


namespace hyperbola_focus_asymptotes_l363_363738

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  (y^2 / 12) - (x^2 / 24) = 1

theorem hyperbola_focus_asymptotes (x y : ℝ) (h_focus : (x = 0 ∧ y = 6))
    (h_asymptotes : (∀ (x y : ℝ), (x^2 / 2) - y^2 = 1 → (y/x = ±(√2))) :
  hyperbola_eq x y :=
by
  sorry

end hyperbola_focus_asymptotes_l363_363738


namespace probability_stops_at_n_smallest_n_for_probability_condition_l363_363873

theorem probability_stops_at_n (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 2010) :
  (1 : ℝ) / (n * (n + 1)) < 1 / 2010 :=
sorry

theorem smallest_n_for_probability_condition :
  ∃ n : ℕ, (1 ≤ n ∧ n ≤ 2010) ∧ (1 / (n * (n + 1)) < 1 / 2010) ∧ ∀ m : ℕ, (1 ≤ m ∧ m < n) → ¬(1 / (m * (m + 1)) < 1 / 2010) :=
by
  use 45
  split
  -- Prove 1 ≤ 45 and 45 ≤ 2010
  { split, exact nat.one_le_of_lt (by norm_num), exact nat.le_refl 45 }
  split
  -- Prove 1 / (45 * (45 + 1)) < 1 / 2010
  { exact probability_stops_at_n 45 (by norm_num) (by norm_num) }
  -- Prove ∀ m : ℕ, (1 ≤ m ∧ m < 45) → ¬(1 / (m * (m + 1)) < 1 / 2010)
  { intros m hm 
    cases hm with hm1 hm2
    have h : m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 
    ∨ m = 6 ∨ m = 7 ∨ m = 8 ∨ m = 9 ∨ m = 10
    ∨ m = 11 ∨ m = 12 ∨ m = 13 ∨ m = 14 ∨ m = 15
    ∨ m = 16 ∨ m = 17 ∨ m = 18 ∨ m = 19 ∨ m = 20
    ∨ m = 21 ∨ m = 22 ∨ m = 23 ∨ m = 24 ∨ m = 25
    ∨ m = 26 ∨ m = 27 ∨ m = 28 ∨ m = 29 ∨ m = 30
    ∨ m = 31 ∨ m = 32 ∨ m = 33 ∨ m = 34 ∨ m = 35
    ∨ m = 36 ∨ m = 37 ∨ m = 38 ∨ m = 39 ∨ m = 40
    ∨ m = 41 ∨ m = 42 ∨ m = 43 ∨ m = 44,
    { sorry }
  }

end probability_stops_at_n_smallest_n_for_probability_condition_l363_363873


namespace cyclic_route_exists_l363_363118

def city := Type
def airline := Type
constant C1 C2 C3 : airline

constant flight : airline → city → city → Prop
constant routes : ∀ c₁ c₂ : city, (flight C1 c₁ c₂ ∨ flight C2 c₁ c₂ ∨ flight C3 c₁ c₂) ∧
  (¬ (flight C1 c₁ c₂ ∧ flight C2 c₁ c₂) ∧ ¬ (flight C2 c₁ c₂ ∧ flight C3 c₁ c₂) ∧ ¬ (flight C3 c₁ c₂ ∧ flight C1 c₁ c₂))

constant all_airlines_in_cities : ∀ c : city, ∃ c₁ c₂ c₃ : city, 
  ((flight C1 c c₁) ∧ (flight C2 c c₂) ∧ (flight C3 c c₃))

theorem cyclic_route_exists : ∃ start : city, ∃ route : list (city × airline), 
  (route.head = (start, _)) ∧ 
  (route.last = (start, _)) ∧ 
  (∀ (s : city × airline), s ∈ route → ∃ (next : city × airline), next ∈ route ∧ s = next) ∧
  (∀ j ≠ k, route.nth j ≠ route.nth k) ∧
  ((route.map prod.snd).eraseDups = [C1, C2, C3]) :=
sorry

end cyclic_route_exists_l363_363118


namespace employee_count_l363_363731

theorem employee_count (avg_salary : ℕ) (manager_salary : ℕ) (new_avg_increase : ℕ) (E : ℕ) :
  (avg_salary = 1500) ∧ (manager_salary = 4650) ∧ (new_avg_increase = 150) →
  1500 * E + 4650 = 1650 * (E + 1) → E = 20 :=
by
  sorry

end employee_count_l363_363731


namespace express_y_in_terms_of_y_l363_363430

variable (x : ℝ)

theorem express_y_in_terms_of_y (y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
sorry

end express_y_in_terms_of_y_l363_363430


namespace geom_seq_general_term_arith_seq_sum_l363_363995

theorem geom_seq_general_term (q : ℕ → ℕ) (a_1 a_2 a_3 : ℕ) (h1 : a_1 = 2)
  (h2 : (a_1 + a_3) / 2 = a_2 + 1) (h3 : a_2 = q 2) (h4 : a_3 = q 3)
  (g : ℕ → ℕ) (Sn : ℕ → ℕ) (gen_term : ∀ n, q n = 2^n) (sum_term : ∀ n, Sn n = 2^(n+1) - 2) :
  q n = g n :=
sorry

theorem arith_seq_sum (a_1 a_2 a_4 : ℕ) (b : ℕ → ℕ) (Tn : ℕ → ℕ) (h1 : a_1 = 2)
  (h2 : a_2 = 4) (h3 : a_4 = 16) (h4 : b 2 = a_1) (h5 : b 8 = a_2 + a_4)
  (gen_term : ∀ n, b n = 1 + 3 * (n - 1)) (sum_term : ∀ n, Tn n = (3 * n^2 - n) / 2) :
  Tn n = (3 * n^2 - 1) / 2 :=
sorry

end geom_seq_general_term_arith_seq_sum_l363_363995


namespace smallest_number_among_four_l363_363392

theorem smallest_number_among_four (a b c d : ℤ) (h1 : a = 2023) (h2 : b = 2022) (h3 : c = -2023) (h4 : d = -2022) : 
  min (min a (min b c)) d = -2023 :=
by
  rw [h1, h2, h3, h4]
  sorry

end smallest_number_among_four_l363_363392
