import Mathlib

namespace jake_bitcoins_l782_78233

theorem jake_bitcoins (initial : ℕ) (donation1 : ℕ) (fraction : ℕ) (multiplier : ℕ) (donation2 : ℕ) :
  initial = 80 →
  donation1 = 20 →
  fraction = 2 →
  multiplier = 3 →
  donation2 = 10 →
  (initial - donation1) / fraction * multiplier - donation2 = 80 :=
by
  sorry

end jake_bitcoins_l782_78233


namespace smallest_square_side_length_paintings_l782_78273

theorem smallest_square_side_length_paintings (n : ℕ) :
  ∃ n : ℕ, (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 2020 → 1 * i ≤ n * n) → n = 1430 :=
by
  sorry

end smallest_square_side_length_paintings_l782_78273


namespace factorize_polynomial_l782_78248

variable (a x y : ℝ)

theorem factorize_polynomial (a x y : ℝ) :
  3 * a * x ^ 2 - 3 * a * y ^ 2 = 3 * a * (x + y) * (x - y) := by
  sorry

end factorize_polynomial_l782_78248


namespace solve_quadratic_1_solve_quadratic_2_l782_78211

theorem solve_quadratic_1 : ∀ x : ℝ, x^2 - 5 * x + 4 = 0 ↔ x = 4 ∨ x = 1 :=
by sorry

theorem solve_quadratic_2 : ∀ x : ℝ, x^2 = 4 - 2 * x ↔ x = -1 + Real.sqrt 5 ∨ x = -1 - Real.sqrt 5 :=
by sorry

end solve_quadratic_1_solve_quadratic_2_l782_78211


namespace find_divisor_l782_78228

variable (n : ℤ) (d : ℤ)

theorem find_divisor 
    (h1 : ∃ k : ℤ, n = k * d + 4)
    (h2 : ∃ m : ℤ, n + 15 = m * 5 + 4) :
    d = 5 :=
sorry

end find_divisor_l782_78228


namespace y_divides_x_squared_l782_78266

-- Define the conditions and proof problem in Lean 4
theorem y_divides_x_squared (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
(h : ∃ (n : ℕ), n = (x^2 / y) + (y^2 / x)) : y ∣ x^2 :=
by {
  -- Proof steps are skipped
  sorry
}

end y_divides_x_squared_l782_78266


namespace percent_first_question_l782_78292

variable (A B : ℝ) (A_inter_B : ℝ) (A_union_B : ℝ)

-- Given conditions
def condition1 : B = 0.49 := sorry
def condition2 : A_inter_B = 0.32 := sorry
def condition3 : A_union_B = 0.80 := sorry
def union_formula : A_union_B = A + B - A_inter_B := 
by sorry

-- Prove that A = 0.63
theorem percent_first_question (h1 : B = 0.49) 
                               (h2 : A_inter_B = 0.32) 
                               (h3 : A_union_B = 0.80) 
                               (h4 : A_union_B = A + B - A_inter_B) : 
                               A = 0.63 :=
by sorry

end percent_first_question_l782_78292


namespace similar_triangles_x_value_l782_78229

theorem similar_triangles_x_value : ∃ (x : ℝ), (12 / x = 9 / 6) ∧ x = 8 := by
  use 8
  constructor
  · sorry
  · rfl

end similar_triangles_x_value_l782_78229


namespace mean_of_six_numbers_sum_three_quarters_l782_78239

theorem mean_of_six_numbers_sum_three_quarters :
  let sum := (3 / 4 : ℝ)
  let n := 6
  (sum / n) = (1 / 8 : ℝ) :=
by
  sorry

end mean_of_six_numbers_sum_three_quarters_l782_78239


namespace quadratic_inequality_l782_78251

variable (a b c A B C : ℝ)

theorem quadratic_inequality
  (h₁ : a ≠ 0)
  (h₂ : A ≠ 0)
  (h₃ : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| :=
sorry

end quadratic_inequality_l782_78251


namespace b_is_arithmetic_sequence_a_general_formula_l782_78237

open Nat

-- Define the sequence a_n
def a : ℕ → ℤ
| 0     => 1
| 1     => 2
| (n+2) => 2 * (a (n+1)) - (a n) + 2

-- Define the sequence b_n
def b (n : ℕ) : ℤ := a (n+1) - a n

-- Part 1: The sequence b_n is an arithmetic sequence
theorem b_is_arithmetic_sequence : ∀ n : ℕ, b (n+1) - b n = 2 := by
  sorry

-- Part 2: Find the general formula for a_n
theorem a_general_formula : ∀ n : ℕ, a (n+1) = n^2 + 1 := by
  sorry

end b_is_arithmetic_sequence_a_general_formula_l782_78237


namespace lower_limit_b_l782_78276

theorem lower_limit_b (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : b < 29) 
  (h4 : ∃ min_b max_b, min_b = 4 ∧ max_b ≤ 29 ∧ 3.75 = (16 : ℚ) / (min_b : ℚ) - (7 : ℚ) / (max_b : ℚ)) : 
  b ≥ 4 :=
sorry

end lower_limit_b_l782_78276


namespace find_A_for_diamondsuit_l782_78296

-- Define the operation
def diamondsuit (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

-- Define the specific instance of the operation equated to 57
theorem find_A_for_diamondsuit :
  ∃ A : ℝ, diamondsuit A 10 = 57 ↔ A = 20 := by
  sorry

end find_A_for_diamondsuit_l782_78296


namespace geometric_sequence_common_ratio_l782_78209

theorem geometric_sequence_common_ratio (a q : ℝ) (h : a = a * q / (1 - q)) : q = 1 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l782_78209


namespace retail_price_before_discounts_l782_78264

theorem retail_price_before_discounts 
  (wholesale_price profit_rate tax_rate discount1 discount2 total_effective_price : ℝ) 
  (h_wholesale_price : wholesale_price = 108)
  (h_profit_rate : profit_rate = 0.20)
  (h_tax_rate : tax_rate = 0.15)
  (h_discount1 : discount1 = 0.10)
  (h_discount2 : discount2 = 0.05)
  (h_total_effective_price : total_effective_price = 126.36) :
  ∃ (retail_price_before_discounts : ℝ), retail_price_before_discounts = 147.78 := 
by
  sorry

end retail_price_before_discounts_l782_78264


namespace distance_they_both_run_l782_78204

theorem distance_they_both_run
  (time_A time_B : ℕ)
  (distance_advantage: ℝ)
  (speed_A speed_B : ℝ)
  (D : ℝ) :
  time_A = 198 →
  time_B = 220 →
  distance_advantage = 300 →
  speed_A = D / time_A →
  speed_B = D / time_B →
  speed_A * time_B = D + distance_advantage →
  D = 2700 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end distance_they_both_run_l782_78204


namespace series_sum_l782_78216

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l782_78216


namespace least_whole_number_l782_78274

theorem least_whole_number (n : ℕ) 
  (h1 : n % 2 = 1)
  (h2 : n % 3 = 1)
  (h3 : n % 4 = 1)
  (h4 : n % 5 = 1)
  (h5 : n % 6 = 1)
  (h6 : 7 ∣ n) : 
  n = 301 := 
sorry

end least_whole_number_l782_78274


namespace unique_element_in_set_l782_78278

theorem unique_element_in_set (A : Set ℝ) (h₁ : ∃ x, A = {x})
(h₂ : ∀ x ∈ A, (x + 3) / (x - 1) ∈ A) : ∃ x, x ∈ A ∧ (x = 3 ∨ x = -1) := by
  sorry

end unique_element_in_set_l782_78278


namespace calculate_AH_l782_78218

def square (a : ℝ) := a ^ 2
def area_square (s : ℝ) := s ^ 2
def area_triangle (b h : ℝ) := 0.5 * b * h

theorem calculate_AH (s DG DH AH : ℝ) 
  (h_square : area_square s = 144) 
  (h_area_triangle : area_triangle DG DH = 63)
  (h_perpendicular : DG = DH)
  (h_hypotenuse : square AH = square s + square DH) :
  AH = 3 * Real.sqrt 30 :=
by
  -- Proof would be provided here
  sorry

end calculate_AH_l782_78218


namespace increasing_interval_l782_78285

noncomputable def function_y (x : ℝ) : ℝ :=
  2 * (Real.logb (1/2) x) ^ 2 - 2 * Real.logb (1/2) x + 1

theorem increasing_interval :
  ∀ x : ℝ, x > 0 → (∀ {y}, y ≥ x → function_y y ≥ function_y x) ↔ x ∈ Set.Ici (Real.sqrt 2 / 2) :=
by
  sorry

end increasing_interval_l782_78285


namespace distance_between_stations_l782_78235

theorem distance_between_stations
  (v₁ v₂ : ℝ)
  (D₁ D₂ : ℝ)
  (T : ℝ)
  (h₁ : v₁ = 20)
  (h₂ : v₂ = 25)
  (h₃ : D₂ = D₁ + 70)
  (h₄ : D₁ = v₁ * T)
  (h₅ : D₂ = v₂ * T) : 
  D₁ + D₂ = 630 := 
by
  sorry

end distance_between_stations_l782_78235


namespace solve_for_y_l782_78205

theorem solve_for_y (y : ℚ) (h : 1 / 3 + 1 / y = 7 / 9) : y = 9 / 4 :=
by
  sorry

end solve_for_y_l782_78205


namespace log_expression_value_l782_78283

noncomputable def log_expression : ℝ :=
  (Real.log (Real.sqrt 27) + Real.log 8 - 3 * Real.log (Real.sqrt 10)) / Real.log 1.2

theorem log_expression_value : log_expression = 3 / 2 :=
  sorry

end log_expression_value_l782_78283


namespace negation_example_l782_78297

open Classical
variable (x : ℝ)

theorem negation_example :
  (¬ (∀ x : ℝ, 2 * x - 1 > 0)) ↔ (∃ x : ℝ, 2 * x - 1 ≤ 0) :=
by
  sorry

end negation_example_l782_78297


namespace B_Bons_wins_probability_l782_78290

theorem B_Bons_wins_probability :
  let roll_six := (1 : ℚ) / 6
  let not_roll_six := (5 : ℚ) / 6
  let p := (5 : ℚ) / 11
  p = (5 / 36) + (25 / 36) * p :=
by
  sorry

end B_Bons_wins_probability_l782_78290


namespace sum_eq_sum_l782_78246

theorem sum_eq_sum {a b c d : ℝ} (h1 : a + b = c + d) (h2 : ac = bd) (h3 : a + b ≠ 0) : a + c = b + d := 
by
  sorry

end sum_eq_sum_l782_78246


namespace min_value_a_l782_78255

theorem min_value_a (a b c : ℤ) (α β : ℝ)
  (h_a_pos : a > 0) 
  (h_eq : ∀ x : ℝ, a * x^2 + b * x + c = 0 → (x = α ∨ x = β))
  (h_alpha_beta_order : 0 < α ∧ α < β ∧ β < 1) :
  a ≥ 5 :=
sorry

end min_value_a_l782_78255


namespace probability_of_losing_l782_78226

noncomputable def odds_of_winning : ℕ := 5
noncomputable def odds_of_losing : ℕ := 3
noncomputable def total_outcomes : ℕ := odds_of_winning + odds_of_losing

theorem probability_of_losing : 
  (odds_of_losing : ℚ) / (total_outcomes : ℚ) = 3 / 8 := 
by
  sorry

end probability_of_losing_l782_78226


namespace train_speed_is_25_kmph_l782_78270

noncomputable def train_speed_kmph (train_length_m : ℕ) (man_speed_kmph : ℕ) (cross_time_s : ℕ) : ℕ :=
  let man_speed_mps := (man_speed_kmph * 1000) / 3600
  let relative_speed_mps := train_length_m / cross_time_s
  let train_speed_mps := relative_speed_mps - man_speed_mps
  let train_speed_kmph := (train_speed_mps * 3600) / 1000
  train_speed_kmph

theorem train_speed_is_25_kmph : train_speed_kmph 270 2 36 = 25 := by
  sorry

end train_speed_is_25_kmph_l782_78270


namespace sum_mod_30_l782_78259

theorem sum_mod_30 (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 7) 
  (h3 : c % 30 = 18) : 
  (a + 2 * b + c) % 30 = 17 := 
by
  sorry

end sum_mod_30_l782_78259


namespace combined_boys_average_l782_78288

noncomputable def average_boys_score (C c D d : ℕ) : ℚ :=
  (68 * C + 74 * 3 * c / 4) / (C + 3 * c / 4)

theorem combined_boys_average:
  ∀ (C c D d : ℕ),
  (68 * C + 72 * c) / (C + c) = 70 →
  (74 * D + 88 * d) / (D + d) = 82 →
  (72 * c + 88 * d) / (c + d) = 83 →
  C = c →
  4 * D = 3 * d →
  average_boys_score C c D d = 48.57 :=
by
  intros C c D d h_clinton h_dixon h_combined_girls h_C_eq_c h_D_eq_d
  sorry

end combined_boys_average_l782_78288


namespace distance_to_other_asymptote_is_8_l782_78242

-- Define the hyperbola and the properties
def hyperbola (x y : ℝ) : Prop := (x^2) / 2 - (y^2) / 8 = 1

-- Define the asymptotes
def asymptote_1 (x y : ℝ) : Prop := y = 2 * x
def asymptote_2 (x y : ℝ) : Prop := y = -2 * x

-- Given conditions
variables (P : ℝ × ℝ)
variable (distance_to_one_asymptote : ℝ)
variable (distance_to_other_asymptote : ℝ)

axiom point_on_hyperbola : hyperbola P.1 P.2
axiom distance_to_one_asymptote_is_1_over_5 : distance_to_one_asymptote = 1 / 5

-- The proof statement
theorem distance_to_other_asymptote_is_8 :
  distance_to_other_asymptote = 8 := sorry

end distance_to_other_asymptote_is_8_l782_78242


namespace number_of_black_squares_in_56th_row_l782_78238

def total_squares (n : Nat) : Nat := 3 + 2 * (n - 1)

def black_squares (n : Nat) : Nat :=
  if total_squares n % 2 == 1 then
    (total_squares n - 1) / 2
  else
    total_squares n / 2

theorem number_of_black_squares_in_56th_row :
  black_squares 56 = 56 :=
by
  sorry

end number_of_black_squares_in_56th_row_l782_78238


namespace odd_function_m_zero_l782_78202

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 + m

theorem odd_function_m_zero (m : ℝ) : (∀ x : ℝ, f (-x) m = -f x m) → m = 0 :=
by
  sorry

end odd_function_m_zero_l782_78202


namespace solve_for_x_l782_78277

theorem solve_for_x (x : ℝ) (h : -3 * x - 12 = 8 * x + 5) : x = -17 / 11 :=
by
  sorry

end solve_for_x_l782_78277


namespace total_vessels_l782_78203

theorem total_vessels (C G S F : ℕ) (h1 : C = 4) (h2 : G = 2 * C) (h3 : S = G + 6) (h4 : S = 7 * F) : 
  C + G + S + F = 28 :=
by
  sorry

end total_vessels_l782_78203


namespace exists_digit_sum_divisible_by_27_not_number_l782_78261

-- Definitions
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def divisible_by (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

-- Theorem statement
theorem exists_digit_sum_divisible_by_27_not_number (n : ℕ) :
  divisible_by (sum_of_digits n) 27 ∧ ¬ divisible_by n 27 :=
  sorry

end exists_digit_sum_divisible_by_27_not_number_l782_78261


namespace other_continent_passengers_l782_78244

noncomputable def totalPassengers := 240
noncomputable def northAmericaFraction := (1 / 3 : ℝ)
noncomputable def europeFraction := (1 / 8 : ℝ)
noncomputable def africaFraction := (1 / 5 : ℝ)
noncomputable def asiaFraction := (1 / 6 : ℝ)

theorem other_continent_passengers :
  (totalPassengers : ℝ) - (totalPassengers * northAmericaFraction +
                           totalPassengers * europeFraction +
                           totalPassengers * africaFraction +
                           totalPassengers * asiaFraction) = 42 :=
by
  sorry

end other_continent_passengers_l782_78244


namespace sum_of_squares_l782_78231

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 :=
by
  sorry

end sum_of_squares_l782_78231


namespace sum_of_three_squares_l782_78220

-- Using the given conditions to define the problem.
variable (square triangle : ℝ)

-- Conditions
axiom h1 : square + triangle + 2 * square + triangle = 34
axiom h2 : triangle + square + triangle + 3 * square = 40

-- Statement to prove
theorem sum_of_three_squares : square + square + square = 66 / 7 :=
by
  sorry

end sum_of_three_squares_l782_78220


namespace mary_needs_to_add_l782_78219

-- Define the conditions
def total_flour_required : ℕ := 7
def flour_already_added : ℕ := 2

-- Define the statement that corresponds to the mathematical equivalent proof problem
theorem mary_needs_to_add :
  total_flour_required - flour_already_added = 5 :=
by
  sorry

end mary_needs_to_add_l782_78219


namespace length_more_than_breadth_l782_78294

theorem length_more_than_breadth (b x : ℕ) 
  (h1 : 60 = b + x) 
  (h2 : 4 * b + 2 * x = 200) : x = 20 :=
by {
  sorry
}

end length_more_than_breadth_l782_78294


namespace part_I_part_II_l782_78217

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x + x^2 - a * x

theorem part_I (x : ℝ) (a : ℝ) (h_inc : ∀ x > 0, (1/x + 2*x - a) ≥ 0) : a ≤ 2 * Real.sqrt 2 :=
sorry

noncomputable def g (x : ℝ) (a : ℝ) := f x a + 2 * Real.log ((a * x + 2) / (6 * Real.sqrt x))

theorem part_II (a : ℝ) (k : ℝ) (h_a : 2 < a ∧ a < 4) (h_ex : ∃ x : ℝ, (3/2) ≤ x ∧ x ≤ 2 ∧ g x a > k * (4 - a^2)) : k ≥ 1/3 :=
sorry

end part_I_part_II_l782_78217


namespace minimum_value_l782_78208

theorem minimum_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ x : ℝ, 
    (x = 2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b) ^ 2) ∧ 
    (∀ y, y = 2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b) ^ 2 → x ≤ y) ∧ 
    x = 7 :=
by 
  sorry

end minimum_value_l782_78208


namespace prove_inequalities_l782_78291

theorem prove_inequalities (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^3 * b > a * b^3 ∧ a - b / a > b - a / b :=
by
  sorry

end prove_inequalities_l782_78291


namespace student_total_marks_l782_78234

variable (M P C : ℕ)

theorem student_total_marks :
  C = P + 20 ∧ (M + C) / 2 = 25 → M + P = 30 :=
by
  sorry

end student_total_marks_l782_78234


namespace square_area_problem_l782_78282

theorem square_area_problem
    (x1 y1 x2 y2 : ℝ)
    (h1 : y1 = x1^2)
    (h2 : y2 = x2^2)
    (line_eq : ∃ a : ℝ, a = 2 ∧ ∃ b : ℝ, b = -22 ∧ ∀ x y : ℝ, y = 2 * x - 22 → (y = y1 ∨ y = y2)) :
    ∃ area : ℝ, area = 180 ∨ area = 980 :=
sorry

end square_area_problem_l782_78282


namespace part1_part2_l782_78289

-- Part 1: Define the sequence and sum function, then state the problem.
def a_1 : ℚ := 3 / 2
def d : ℚ := 1

def S_n (n : ℕ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem part1 (k : ℕ) (h : S_n (k^2) = (S_n k)^2) : k = 4 := sorry

-- Part 2: Define the general sequence and state the problem.
def arith_seq (a_1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a_1 + (n - 1) * d

def S_n_general (a_1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n * a_1) + (n * (n - 1) / 2) * d

theorem part2 (a_1 : ℚ) (d : ℚ) :
  (∀ k : ℕ, S_n_general a_1 d (k^2) = (S_n_general a_1 d k)^2) ↔
  (a_1 = 0 ∧ d = 0) ∨
  (a_1 = 1 ∧ d = 0) ∨
  (a_1 = 1 ∧ d = 2) := sorry

end part1_part2_l782_78289


namespace probability_red_or_white_ball_l782_78298

theorem probability_red_or_white_ball :
  let red_balls := 3
  let yellow_balls := 2
  let white_balls := 1
  let total_balls := red_balls + yellow_balls + white_balls
  let favorable_outcomes := red_balls + white_balls
  (favorable_outcomes / total_balls : ℚ) = 2 / 3 := by
  sorry

end probability_red_or_white_ball_l782_78298


namespace length_to_width_ratio_l782_78299

/-- Let the perimeter of the rectangular sandbox be 30 feet,
    the width be 5 feet, and the length be some multiple of the width.
    Prove that the ratio of the length to the width is 2:1. -/
theorem length_to_width_ratio (P w : ℕ) (h1 : P = 30) (h2 : w = 5) (h3 : ∃ k, l = k * w) : 
  ∃ l, (P = 2 * (l + w)) ∧ (l / w = 2) := 
sorry

end length_to_width_ratio_l782_78299


namespace a4_is_5_l782_78295

-- Define the condition x^5 = a_n + a_1(x-1) + a_2(x-1)^2 + a_3(x-1)^3 + a_4(x-1)^4 + a_5(x-1)^5
noncomputable def polynomial_identity (x a_n a_1 a_2 a_3 a_4 a_5 : ℝ) : Prop :=
  x^5 = a_n + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5

-- Define the theorem statement
theorem a4_is_5 (x a_n a_1 a_2 a_3 a_5 : ℝ) (h : polynomial_identity x a_n a_1 a_2 a_3 5 a_5) : a_4 = 5 :=
 by
 sorry

end a4_is_5_l782_78295


namespace find_a1_l782_78249

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

def is_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem find_a1 (h1 : is_arithmetic_seq a (-2)) 
               (h2 : sum_n_terms S a) 
               (h3 : S 10 = S 11) : 
  a 1 = 20 :=
sorry

end find_a1_l782_78249


namespace evaluate_expression_l782_78256

theorem evaluate_expression :
  8^(-1/3 : ℝ) + (49^(-1/2 : ℝ))^(1/2 : ℝ) = (Real.sqrt 7 + 2) / (2 * Real.sqrt 7) := by
  sorry

end evaluate_expression_l782_78256


namespace evaluate_expression_l782_78286

theorem evaluate_expression :
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = 137 / 52 :=
by
  sorry

end evaluate_expression_l782_78286


namespace jinhee_pages_per_day_l782_78207

noncomputable def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  (total_pages + days - 1) / days

theorem jinhee_pages_per_day : 
  ∀ (total_pages : ℕ) (days : ℕ), total_pages = 220 → days = 7 → pages_per_day total_pages days = 32 :=
by 
  intros total_pages days hp hd
  rw [hp, hd]
  -- the computation of the function
  show pages_per_day 220 7 = 32
  sorry

end jinhee_pages_per_day_l782_78207


namespace line_intersects_circle_l782_78223

theorem line_intersects_circle (m : ℝ) : 
  ∃ (x y : ℝ), y = m * x - 3 ∧ x^2 + (y - 1)^2 = 25 :=
sorry

end line_intersects_circle_l782_78223


namespace volume_of_snow_correct_l782_78275

noncomputable def volume_of_snow : ℝ :=
  let sidewalk_length := 30
  let sidewalk_width := 3
  let depth := 3 / 4
  let sidewalk_volume := sidewalk_length * sidewalk_width * depth
  
  let garden_path_leg1 := 3
  let garden_path_leg2 := 4
  let garden_path_area := (garden_path_leg1 * garden_path_leg2) / 2
  let garden_path_volume := garden_path_area * depth
  
  let total_volume := sidewalk_volume + garden_path_volume
  total_volume

theorem volume_of_snow_correct : volume_of_snow = 72 := by
  sorry

end volume_of_snow_correct_l782_78275


namespace ratio_of_Phil_to_Bob_l782_78247

-- There exists real numbers P, J, and B such that
theorem ratio_of_Phil_to_Bob (P J B : ℝ) (h1 : J = 2 * P) (h2 : B = 60) (h3 : J = B - 20) : P / B = 1 / 3 :=
by
  sorry

end ratio_of_Phil_to_Bob_l782_78247


namespace unique_sequence_exists_and_bounded_l782_78260

theorem unique_sequence_exists_and_bounded (a : ℝ) (n : ℕ) :
  ∃! (x : ℕ → ℝ), -- There exists a unique sequence x : ℕ → ℝ
    (x 1 = x (n - 1)) ∧ -- x_1 = x_{n-1}
    (∀ i, 1 ≤ i ∧ i ≤ n → (1 / 2) * (x (i - 1) + x i) = x i + x i ^ 3 - a ^ 3) ∧ -- Condition for all 1 ≤ i ≤ n
    (∀ i, 0 ≤ i ∧ i ≤ n + 1 → |x i| ≤ |a|) -- Bounding condition for all 0 ≤ i ≤ n + 1
:= sorry

end unique_sequence_exists_and_bounded_l782_78260


namespace remainder_division_l782_78214

theorem remainder_division (x : ℤ) (hx : x % 82 = 5) : (x + 7) % 41 = 12 := 
by 
  sorry

end remainder_division_l782_78214


namespace gcd_problem_l782_78236

theorem gcd_problem : ∃ b : ℕ, gcd (20 * b) (18 * 24) = 2 :=
by { sorry }

end gcd_problem_l782_78236


namespace dihedral_angle_ge_l782_78284

-- Define the problem conditions and goal in Lean
theorem dihedral_angle_ge (n : ℕ) (h : 3 ≤ n) (ϕ : ℝ) :
  ϕ ≥ π * (1 - 2 / n) := 
sorry

end dihedral_angle_ge_l782_78284


namespace simplify_and_evaluate_expression_l782_78254

variable (a b : ℚ)

theorem simplify_and_evaluate_expression
  (ha : a = 1 / 2)
  (hb : b = -1 / 3) :
  b^2 - a^2 + 2 * (a^2 + a * b) - (a^2 + b^2) = -1 / 3 :=
by
  -- The proof will be inserted here
  sorry

end simplify_and_evaluate_expression_l782_78254


namespace average_words_per_hour_l782_78240

theorem average_words_per_hour
  (total_words : ℕ := 60000)
  (total_hours : ℕ := 150)
  (first_period_hours : ℕ := 50)
  (first_period_words : ℕ := total_words / 2) :
  first_period_words / first_period_hours = 600 ∧ total_words / total_hours = 400 := 
by
  sorry

end average_words_per_hour_l782_78240


namespace heroes_can_reduce_heads_to_zero_l782_78221

-- Definition of the Hero strikes
def IlyaMurometsStrikes (H : ℕ) : ℕ := H / 2 - 1
def DobrynyaNikitichStrikes (H : ℕ) : ℕ := 2 * H / 3 - 2
def AlyoshaPopovichStrikes (H : ℕ) : ℕ := 3 * H / 4 - 3

-- The ultimate goal is proving this theorem
theorem heroes_can_reduce_heads_to_zero (H : ℕ) : 
  ∃ (n : ℕ), ∀ i ≤ n, 
  (if i % 3 = 0 then H = 0 
   else if i % 3 = 1 then IlyaMurometsStrikes H = 0 
   else if i % 3 = 2 then DobrynyaNikitichStrikes H = 0 
   else AlyoshaPopovichStrikes H = 0)
:= sorry

end heroes_can_reduce_heads_to_zero_l782_78221


namespace card_statements_has_four_true_l782_78222

noncomputable def statement1 (S : Fin 5 → Bool) : Prop := S 0 = true -> (S 1 = false ∧ S 2 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement2 (S : Fin 5 → Bool) : Prop := S 1 = true -> (S 0 = false ∧ S 2 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement3 (S : Fin 5 → Bool) : Prop := S 2 = true -> (S 0 = false ∧ S 1 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement4 (S : Fin 5 → Bool) : Prop := S 3 = true -> (S 0 = false ∧ S 1 = false ∧ S 2 = false ∧ S 4 = false)
noncomputable def statement5 (S : Fin 5 → Bool) : Prop := S 4 = true -> (S 0 = false ∧ S 1 = false ∧ S 2 = false ∧ S 3 = false)

theorem card_statements_has_four_true : ∃ (S : Fin 5 → Bool), 
  (statement1 S ∧ statement2 S ∧ statement3 S ∧ statement4 S ∧ statement5 S ∧ 
  ((S 0 = true ∨ S 1 = true ∨ S 2 = true ∨ S 3 = true ∨ S 4 = true) ∧ 
  4 = (if S 0 then 1 else 0) + (if S 1 then 1 else 0) + 
      (if S 2 then 1 else 0) + (if S 3 then 1 else 0) + 
      (if S 4 then 1 else 0))) :=
sorry

end card_statements_has_four_true_l782_78222


namespace largest_of_seven_consecutive_integers_l782_78241

-- Define the main conditions as hypotheses
theorem largest_of_seven_consecutive_integers (n : ℕ) (h_sum : 7 * n + 21 = 2401) : 
  n + 6 = 346 :=
by
  -- Conditions from the problem are utilized here
  sorry

end largest_of_seven_consecutive_integers_l782_78241


namespace square_minus_self_divisible_by_2_l782_78213

theorem square_minus_self_divisible_by_2 (a : ℕ) : 2 ∣ (a^2 - a) :=
by sorry

end square_minus_self_divisible_by_2_l782_78213


namespace sum_of_powers_eq_zero_l782_78267

theorem sum_of_powers_eq_zero
  (a b c : ℝ)
  (n : ℝ)
  (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = 0) :
  a^(2* ⌊n⌋ + 1) + b^(2* ⌊n⌋ + 1) + c^(2* ⌊n⌋ + 1) = 0 := by
  sorry

end sum_of_powers_eq_zero_l782_78267


namespace repeating_decmials_sum_is_fraction_l782_78225

noncomputable def x : ℚ := 2/9
noncomputable def y : ℚ := 2/99
noncomputable def z : ℚ := 2/9999

theorem repeating_decmials_sum_is_fraction :
  (x + y + z) = 2426 / 9999 := by
  sorry

end repeating_decmials_sum_is_fraction_l782_78225


namespace no_friendly_triplet_in_range_l782_78200

open Nat

def isFriendly (a b c : ℕ) : Prop :=
  (a ∣ (b * c) ∨ b ∣ (a * c) ∨ c ∣ (a * b))

theorem no_friendly_triplet_in_range (n : ℕ) (a b c : ℕ) :
  n^2 < a ∧ a < n^2 + n → n^2 < b ∧ b < n^2 + n → n^2 < c ∧ c < n^2 + n → a ≠ b → b ≠ c → a ≠ c →
  ¬ isFriendly a b c :=
by sorry

end no_friendly_triplet_in_range_l782_78200


namespace angle_B_measure_triangle_area_l782_78268

noncomputable def triangle (A B C : ℝ) : Type := sorry

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Given conditions:
axiom eq1 : b * Real.cos C = (2 * a - c) * Real.cos B

-- Part 1: Prove the measure of angle B
theorem angle_B_measure : B = Real.pi / 3 :=
by
  have b_cos_C := eq1
  sorry

-- Part 2: Given additional conditions and find the area
variable (b_value : ℝ := Real.sqrt 7)
variable (sum_ac : ℝ := 4)

theorem triangle_area : (1 / 2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4) :=
by
  have b_value_def := b_value
  have sum_ac_def := sum_ac
  sorry

end angle_B_measure_triangle_area_l782_78268


namespace original_number_l782_78250

theorem original_number (n : ℕ) (h : (2 * (n + 2) - 2) / 2 = 7) : n = 6 := by
  sorry

end original_number_l782_78250


namespace arithmetic_mean_reciprocals_primes_l782_78279

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l782_78279


namespace find_metal_sheet_width_l782_78280

-- The given conditions
def metalSheetLength : ℝ := 100
def cutSquareSide : ℝ := 10
def boxVolume : ℝ := 24000

-- Statement to prove
theorem find_metal_sheet_width (w : ℝ) (h : w - 2 * cutSquareSide > 0):
  boxVolume = (metalSheetLength - 2 * cutSquareSide) * (w - 2 * cutSquareSide) * cutSquareSide → 
  w = 50 := 
by {
  sorry
}

end find_metal_sheet_width_l782_78280


namespace martha_no_daughters_count_l782_78253

-- Definitions based on conditions
def total_people : ℕ := 40
def martha_daughters : ℕ := 8
def granddaughters_per_child (x : ℕ) : ℕ := if x = 1 then 8 else 0

-- Statement of the problem
theorem martha_no_daughters_count : 
  (total_people - martha_daughters) +
  (martha_daughters - (total_people - martha_daughters) / 8) = 36 := 
  by
    sorry

end martha_no_daughters_count_l782_78253


namespace fraction_water_by_volume_l782_78201

theorem fraction_water_by_volume
  (A W : ℝ) 
  (h1 : A / W = 0.5)
  (h2 : A / (A + W) = 1/7) : 
  W / (A + W) = 2/7 :=
by
  sorry

end fraction_water_by_volume_l782_78201


namespace students_who_like_both_l782_78263

def total_students : ℕ := 50
def apple_pie_lovers : ℕ := 22
def chocolate_cake_lovers : ℕ := 20
def neither_dessert_lovers : ℕ := 15

theorem students_who_like_both : 
  (apple_pie_lovers + chocolate_cake_lovers) - (total_students - neither_dessert_lovers) = 7 :=
by
  -- Calculation steps (skipped)
  sorry

end students_who_like_both_l782_78263


namespace sally_remaining_cards_l782_78262

variable (total_cards : ℕ) (torn_cards : ℕ) (bought_cards : ℕ)

def intact_cards (total_cards : ℕ) (torn_cards : ℕ) : ℕ := total_cards - torn_cards
def remaining_cards (intact_cards : ℕ) (bought_cards : ℕ) : ℕ := intact_cards - bought_cards

theorem sally_remaining_cards :
  intact_cards 39 9 - 24 = 6 :=
by
  -- sorry for proof
  sorry

end sally_remaining_cards_l782_78262


namespace hcf_of_two_numbers_l782_78243

theorem hcf_of_two_numbers (A B : ℕ) (h1 : A * B = 4107) (h2 : A = 111) : (Nat.gcd A B) = 37 :=
by
  -- Given conditions
  have h3 : B = 37 := by
    -- Deduce B from given conditions
    sorry
  -- Prove hcf (gcd) is 37
  sorry

end hcf_of_two_numbers_l782_78243


namespace dividend_calculation_l782_78210

theorem dividend_calculation (divisor quotient remainder dividend : ℕ)
  (h1 : divisor = 36)
  (h2 : quotient = 20)
  (h3 : remainder = 5)
  (h4 : dividend = (divisor * quotient) + remainder)
  : dividend = 725 := 
by
  -- We skip the proof here
  sorry

end dividend_calculation_l782_78210


namespace linear_system_solution_l782_78269

theorem linear_system_solution :
  ∃ (x y z : ℝ), (x ≠ 0) ∧ (y ≠ 0) ∧ (z ≠ 0) ∧
  (x + (85/3) * y + 4 * z = 0) ∧ 
  (4 * x + (85/3) * y + z = 0) ∧ 
  (3 * x + 5 * y - 2 * z = 0) ∧ 
  (x * z) / (y ^ 2) = 25 := 
sorry

end linear_system_solution_l782_78269


namespace actual_price_of_good_l782_78212

variables (P : Real)

theorem actual_price_of_good:
  (∀ (P : ℝ), 0.5450625 * P = 6500 → P = 6500 / 0.5450625) :=
  by sorry

end actual_price_of_good_l782_78212


namespace patanjali_distance_first_day_l782_78265

theorem patanjali_distance_first_day
  (h : ℕ)
  (H1 : 3 * h + 4 * (h - 1) + 4 * h = 62) :
  3 * h = 18 :=
by
  sorry

end patanjali_distance_first_day_l782_78265


namespace fractions_with_same_denominators_fractions_with_same_numerators_fractions_with_different_numerators_and_denominators_l782_78257

theorem fractions_with_same_denominators {a b c : ℤ} (h_c : c ≠ 0) :
  (a > b → a / (c:ℚ) > b / (c:ℚ)) ∧ (a < b → a / (c:ℚ) < b / (c:ℚ)) :=
by sorry

theorem fractions_with_same_numerators {a c d : ℤ} (h_c : c ≠ 0) (h_d : d ≠ 0) :
  (c < d → a / (c:ℚ) > a / (d:ℚ)) ∧ (c > d → a / (c:ℚ) < a / (d:ℚ)) :=
by sorry

theorem fractions_with_different_numerators_and_denominators {a b c d : ℤ} (h_c : c ≠ 0) (h_d : d ≠ 0) :
  a > b ∧ c < d → a / (c:ℚ) > b / (d:ℚ) :=
by sorry

end fractions_with_same_denominators_fractions_with_same_numerators_fractions_with_different_numerators_and_denominators_l782_78257


namespace percentage_increase_each_job_l782_78232

-- Definitions of original and new amounts for each job as given conditions
def original_first_job : ℝ := 65
def new_first_job : ℝ := 70

def original_second_job : ℝ := 240
def new_second_job : ℝ := 315

def original_third_job : ℝ := 800
def new_third_job : ℝ := 880

-- Proof problem statement
theorem percentage_increase_each_job :
  (new_first_job - original_first_job) / original_first_job * 100 = 7.69 ∧
  (new_second_job - original_second_job) / original_second_job * 100 = 31.25 ∧
  (new_third_job - original_third_job) / original_third_job * 100 = 10 := by
  sorry

end percentage_increase_each_job_l782_78232


namespace rational_numbers_countable_l782_78252

theorem rational_numbers_countable : ∃ (f : ℚ → ℕ), Function.Bijective f :=
by
  sorry

end rational_numbers_countable_l782_78252


namespace negation_of_p_l782_78245

open Classical

variable {x : ℝ}

def p : Prop := ∃ x : ℝ, x > 1

theorem negation_of_p : ¬p ↔ ∀ x : ℝ, x ≤ 1 :=
by
  sorry

end negation_of_p_l782_78245


namespace houses_after_boom_l782_78272

theorem houses_after_boom (h_pre_boom : ℕ) (h_built : ℕ) (h_count : ℕ)
  (H1 : h_pre_boom = 1426)
  (H2 : h_built = 574)
  (H3 : h_count = h_pre_boom + h_built) :
  h_count = 2000 :=
by {
  sorry
}

end houses_after_boom_l782_78272


namespace vote_ratio_l782_78230

theorem vote_ratio (X Y Z : ℕ) (hZ : Z = 25000) (hX : X = 22500) (hX_Y : X = Y + (1/2 : ℚ) * Y) 
    : Y / (Z - Y) = 2 / 5 := 
by 
  sorry

end vote_ratio_l782_78230


namespace complement_intersection_l782_78224

noncomputable def M : Set ℝ := {x | 2 / x < 1}
noncomputable def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 1)}

theorem complement_intersection : 
  ((Set.univ \ M) ∩ N) = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end complement_intersection_l782_78224


namespace star_example_l782_78271

def star (x y : ℝ) : ℝ := 2 * x * y - 3 * x + y

theorem star_example : (star 6 4) - (star 4 6) = -8 := by
  sorry

end star_example_l782_78271


namespace minimum_value_condition_l782_78258

theorem minimum_value_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) 
                                (h_line : ∀ x y : ℝ, m * x + n * y + 2 = 0 → (x + 3)^2 + (y + 1)^2 = 1) 
                                (h_chord : ∀ x1 y1 x2 y2 : ℝ, m * x1 + n * y1 + 2 = 0 ∧ (x1 + 3)^2 + (y1 + 1)^2 = 1 ∧
                                           m * x2 + n * y2 + 2 = 0 ∧ (x2 + 3)^2 + (y2 + 1)^2 = 1 ∧
                                           (x1 - x2)^2 + (y1 - y2)^2 = 4) 
                                (h_relation : 3 * m + n = 2) : 
    ∃ (C : ℝ), C = 6 ∧ (C = (1 / m + 3 / n)) := 
by
  sorry

end minimum_value_condition_l782_78258


namespace average_difference_l782_78215

theorem average_difference :
  let avg1 := (200 + 400) / 2
  let avg2 := (100 + 200) / 2
  avg1 - avg2 = 150 :=
by
  sorry

end average_difference_l782_78215


namespace solve_for_x_l782_78287

theorem solve_for_x (x y z w : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) 
(h3 : x * z + y * w = 50) (h4 : z - w = 5) : x = 20 := 
by 
  sorry

end solve_for_x_l782_78287


namespace pie_eaten_after_four_trips_l782_78206

theorem pie_eaten_after_four_trips : 
  let trip1 := (1 / 3 : ℝ)
  let trip2 := (1 / 3^2 : ℝ)
  let trip3 := (1 / 3^3 : ℝ)
  let trip4 := (1 / 3^4 : ℝ)
  trip1 + trip2 + trip3 + trip4 = (40 / 81 : ℝ) :=
by
  sorry

end pie_eaten_after_four_trips_l782_78206


namespace card_arrangement_probability_l782_78281

/-- 
This problem considers the probability of arranging four distinct cards,
each labeled with a unique character, in such a way that they form one of two specific
sequences. Specifically, the sequences are "我爱数学" (I love mathematics) and "数学爱我" (mathematics loves me).
-/
theorem card_arrangement_probability :
  let cards := ["我", "爱", "数", "学"]
  let total_permutations := 24
  let favorable_outcomes := 2
  let probability := favorable_outcomes / total_permutations
  probability = 1 / 12 :=
by
  sorry

end card_arrangement_probability_l782_78281


namespace correct_operation_l782_78293

theorem correct_operation (a : ℝ) : (-a^3)^4 = a^12 :=
by sorry

end correct_operation_l782_78293


namespace goldie_total_earnings_l782_78227

-- Define weekly earnings based on hours and rates
def earnings_first_week (hours_dog_walking hours_medication : ℕ) : ℕ :=
  (hours_dog_walking * 5) + (hours_medication * 8)

def earnings_second_week (hours_feeding hours_cleaning hours_playing : ℕ) : ℕ :=
  (hours_feeding * 6) + (hours_cleaning * 4) + (hours_playing * 3)

-- Given conditions for hours worked each task in two weeks
def hours_dog_walking : ℕ := 12
def hours_medication : ℕ := 8
def hours_feeding : ℕ := 10
def hours_cleaning : ℕ := 15
def hours_playing : ℕ := 5

-- Proof statement: Total earnings over two weeks equals $259
theorem goldie_total_earnings : 
  (earnings_first_week hours_dog_walking hours_medication) + 
  (earnings_second_week hours_feeding hours_cleaning hours_playing) = 259 :=
by
  sorry

end goldie_total_earnings_l782_78227
