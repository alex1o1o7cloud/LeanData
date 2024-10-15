import Mathlib

namespace NUMINAMATH_GPT_simplify_evaluate_expression_l2382_238271

theorem simplify_evaluate_expression (a b : ℤ) (h1 : a = -1) (h2 : b = 2) :
  3 * (a^2 * b + a * b^2) - 2 * (a^2 * b - 1) - 2 * (a * b^2) - 2 = -2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_evaluate_expression_l2382_238271


namespace NUMINAMATH_GPT_solve_eq_64_16_pow_x_minus_1_l2382_238224

theorem solve_eq_64_16_pow_x_minus_1 (x : ℝ) (h : 64 = 4 * (16 : ℝ) ^ (x - 1)) : x = 2 :=
sorry

end NUMINAMATH_GPT_solve_eq_64_16_pow_x_minus_1_l2382_238224


namespace NUMINAMATH_GPT_find_e_l2382_238277

-- Define the conditions and state the theorem.
def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) 
  (h1: ∃ a b c : ℝ, (a + b + c)/3 = -3 ∧ a * b * c = -3 ∧ 3 + d + e + f = -3)
  (h2: Q 0 d e f = 9) : e = -42 :=
by
  sorry

end NUMINAMATH_GPT_find_e_l2382_238277


namespace NUMINAMATH_GPT_joint_savings_account_total_l2382_238273

theorem joint_savings_account_total :
  let kimmie_earnings : ℕ := 450
  let zahra_earnings : ℕ := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings : ℕ := kimmie_earnings / 2
  let zahra_savings : ℕ := zahra_earnings / 2
  kimmie_savings + zahra_savings = 375 :=
by
  let kimmie_earnings := 450
  let zahra_earnings := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings := kimmie_earnings / 2
  let zahra_savings := zahra_earnings / 2
  have h : kimmie_savings + zahra_savings = 375 := sorry
  exact h

end NUMINAMATH_GPT_joint_savings_account_total_l2382_238273


namespace NUMINAMATH_GPT_maximum_value_of_function_l2382_238284

theorem maximum_value_of_function : ∃ x, x > (1 : ℝ) ∧ (∀ y, y > 1 → (x + 1 / (x - 1) ≥ y + 1 / (y - 1))) ∧ (x = 2 ∧ (x + 1 / (x - 1) = 3)) :=
sorry

end NUMINAMATH_GPT_maximum_value_of_function_l2382_238284


namespace NUMINAMATH_GPT_sum_infinite_series_eq_half_l2382_238233

theorem sum_infinite_series_eq_half :
  (∑' n : ℕ, (n^5 + 2*n^3 + 5*n^2 + 20*n + 20) / (2^(n + 1) * (n^5 + 5))) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_sum_infinite_series_eq_half_l2382_238233


namespace NUMINAMATH_GPT_solve_for_x_l2382_238291

theorem solve_for_x (x : ℚ) : (2/3 : ℚ) - 1/4 = 1/x → x = 12/5 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2382_238291


namespace NUMINAMATH_GPT_expression_nonnegative_l2382_238246

theorem expression_nonnegative (x : ℝ) :
  0 <= x ∧ x < 3 → (2*x - 6*x^2 + 9*x^3) / (9 - x^3) ≥ 0 := 
by
  sorry

end NUMINAMATH_GPT_expression_nonnegative_l2382_238246


namespace NUMINAMATH_GPT_grade_assignment_ways_l2382_238293

-- Define the number of students and the number of grade choices
def students : ℕ := 12
def grade_choices : ℕ := 4

-- Define the number of ways to assign grades
def num_ways_to_assign_grades : ℕ := grade_choices ^ students

-- Prove that the number of ways to assign grades is 16777216
theorem grade_assignment_ways :
  num_ways_to_assign_grades = 16777216 :=
by
  -- Calculation validation omitted (proof step)
  sorry

end NUMINAMATH_GPT_grade_assignment_ways_l2382_238293


namespace NUMINAMATH_GPT_contrapositive_of_happy_people_possess_it_l2382_238268

variable (P Q : Prop)

theorem contrapositive_of_happy_people_possess_it
  (h : P → Q) : ¬ Q → ¬ P := by
  intro hq
  intro p
  apply hq
  apply h
  exact p

#check contrapositive_of_happy_people_possess_it

end NUMINAMATH_GPT_contrapositive_of_happy_people_possess_it_l2382_238268


namespace NUMINAMATH_GPT_triangle_angle_type_l2382_238265

theorem triangle_angle_type (a b c R : ℝ) (hc_max : c ≥ a ∧ c ≥ b) :
  (a^2 + b^2 + c^2 - 8 * R^2 > 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ α < π / 2 ∧ β < π / 2 ∧ γ < π / 2) ∧
  (a^2 + b^2 + c^2 - 8 * R^2 = 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ (α = π / 2 ∨ β = π / 2 ∨ γ = π / 2)) ∧
  (a^2 + b^2 + c^2 - 8 * R^2 < 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ (α > π / 2 ∨ β > π / 2 ∨ γ > π / 2)) :=
sorry

end NUMINAMATH_GPT_triangle_angle_type_l2382_238265


namespace NUMINAMATH_GPT_Monet_paintings_consecutively_l2382_238238

noncomputable def probability_Monet_paintings_consecutively (total_art_pieces Monet_paintings : ℕ) : ℚ :=
  let numerator := 9 * Nat.factorial (total_art_pieces - Monet_paintings) * Nat.factorial Monet_paintings
  let denominator := Nat.factorial total_art_pieces
  numerator / denominator

theorem Monet_paintings_consecutively :
  probability_Monet_paintings_consecutively 12 4 = 18 / 95 := by
  sorry

end NUMINAMATH_GPT_Monet_paintings_consecutively_l2382_238238


namespace NUMINAMATH_GPT_sum_of_ages_is_50_l2382_238208

def youngest_child_age : ℕ := 4

def age_intervals : ℕ := 3

def ages_sum (n : ℕ) : ℕ :=
  youngest_child_age + (youngest_child_age + age_intervals) +
  (youngest_child_age + 2 * age_intervals) +
  (youngest_child_age + 3 * age_intervals) +
  (youngest_child_age + 4 * age_intervals)

theorem sum_of_ages_is_50 : ages_sum 5 = 50 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_is_50_l2382_238208


namespace NUMINAMATH_GPT_parabola_standard_equation_l2382_238218

variable (a : ℝ) (h : a < 0)

theorem parabola_standard_equation :
  (∃ p : ℝ, y^2 = -2 * p * x ∧ p = -2 * a) → y^2 = 4 * a * x :=
by
  sorry

end NUMINAMATH_GPT_parabola_standard_equation_l2382_238218


namespace NUMINAMATH_GPT_arithmetic_series_first_term_l2382_238230

theorem arithmetic_series_first_term 
  (a d : ℚ)
  (h1 : 15 * (2 * a +  29 * d) = 450)
  (h2 : 15 * (2 * a + 89 * d) = 1650) :
  a = -13 / 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_series_first_term_l2382_238230


namespace NUMINAMATH_GPT_edward_spring_earnings_l2382_238225

-- Define the relevant constants and the condition
def springEarnings := 2
def summerEarnings := 27
def expenses := 5
def totalEarnings := 24

-- The condition
def edwardCondition := summerEarnings - expenses = 22

-- The statement to prove
theorem edward_spring_earnings (h : edwardCondition) : springEarnings + 22 = totalEarnings :=
by
  -- Provide the proof here, but we'll use sorry to skip it
  sorry

end NUMINAMATH_GPT_edward_spring_earnings_l2382_238225


namespace NUMINAMATH_GPT_abs_inequality_solution_l2382_238276

theorem abs_inequality_solution :
  {x : ℝ | |x - 2| + |x + 3| < 7} = {x : ℝ | -4 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_abs_inequality_solution_l2382_238276


namespace NUMINAMATH_GPT_ratio_value_l2382_238278

theorem ratio_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
(h1 : (y + 1) / (x - z + 1) = (x + y + 2) / (z + 2)) 
(h2 : (x + y + 2) / (z + 2) = (x + 1) / (y + 1)) :
  (x + 1) / (y + 1) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_value_l2382_238278


namespace NUMINAMATH_GPT_multiplication_scaling_l2382_238280

theorem multiplication_scaling (h : 28 * 15 = 420) : 
  (28 / 10) * (15 / 10) = 2.8 * 1.5 ∧ 
  (28 / 100) * 1.5 = 0.28 * 1.5 ∧ 
  (28 / 1000) * (15 / 100) = 0.028 * 0.15 :=
by 
  sorry

end NUMINAMATH_GPT_multiplication_scaling_l2382_238280


namespace NUMINAMATH_GPT_weekly_tax_percentage_is_zero_l2382_238247

variables (daily_expense : ℕ) (daily_revenue_fries : ℕ) (daily_revenue_poutine : ℕ) (weekly_net_income : ℕ)

def weekly_expense := daily_expense * 7
def weekly_revenue := daily_revenue_fries * 7 + daily_revenue_poutine * 7
def weekly_total_income := weekly_net_income + weekly_expense
def weekly_tax := weekly_total_income - weekly_revenue

theorem weekly_tax_percentage_is_zero
  (h1 : daily_expense = 10)
  (h2 : daily_revenue_fries = 12)
  (h3 : daily_revenue_poutine = 8)
  (h4 : weekly_net_income = 56) :
  weekly_tax = 0 :=
by sorry

end NUMINAMATH_GPT_weekly_tax_percentage_is_zero_l2382_238247


namespace NUMINAMATH_GPT_solve_for_a_l2382_238294

theorem solve_for_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l2382_238294


namespace NUMINAMATH_GPT_longest_side_of_triangle_l2382_238266

theorem longest_side_of_triangle :
  ∃ y : ℚ, 6 + (y + 3) + (3 * y - 2) = 40 ∧ max (6 : ℚ) (max (y + 3) (3 * y - 2)) = 91 / 4 :=
by
  sorry

end NUMINAMATH_GPT_longest_side_of_triangle_l2382_238266


namespace NUMINAMATH_GPT_complex_magnitude_squared_l2382_238206

open Complex Real

theorem complex_magnitude_squared :
  ∃ (z : ℂ), z + abs z = 3 + 7 * i ∧ abs z ^ 2 = 841 / 9 :=
by
  sorry

end NUMINAMATH_GPT_complex_magnitude_squared_l2382_238206


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2382_238257

variable (a b : ℝ) (lna lnb : ℝ)

theorem necessary_but_not_sufficient_condition (h1 : lna < lnb) (h2 : lna = Real.log a) (h3 : lnb = Real.log b) :
  (a > 0 ∧ b > 0 ∧ a < b ∧ a ^ 3 < b ^ 3) ∧ ¬(a ^ 3 < b ^ 3 → 0 < a ∧ a < b ∧ 0 < b) :=
by {
  sorry
}

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2382_238257


namespace NUMINAMATH_GPT_winnie_lollipops_remainder_l2382_238243

theorem winnie_lollipops_remainder :
  ∃ (k : ℕ), k = 505 % 14 ∧ k = 1 :=
by
  sorry

end NUMINAMATH_GPT_winnie_lollipops_remainder_l2382_238243


namespace NUMINAMATH_GPT_x_intercepts_of_parabola_l2382_238269

theorem x_intercepts_of_parabola : 
  (∃ y : ℝ, x = -3 * y^2 + 2 * y + 2) → ∃ y : ℝ, y = 0 ∧ x = 2 ∧ ∀ y' ≠ 0, x ≠ -3 * y'^2 + 2 * y' + 2 :=
by
  sorry

end NUMINAMATH_GPT_x_intercepts_of_parabola_l2382_238269


namespace NUMINAMATH_GPT_new_average_doubled_l2382_238258

theorem new_average_doubled (n : ℕ) (avg : ℝ) (h1 : n = 12) (h2 : avg = 50) :
  2 * avg = 100 := by
sorry

end NUMINAMATH_GPT_new_average_doubled_l2382_238258


namespace NUMINAMATH_GPT_part1_part2_l2382_238232

-- Part 1: Showing x range for increasing actual processing fee
theorem part1 (x : ℝ) : (x ≤ 99.5) ↔ (∀ y, 0 < y → y ≤ x → (1/2) * Real.log (2 * y + 1) - y / 200 ≤ (1/2) * Real.log (2 * (y + 0.1) + 1) - (y + 0.1) / 200) :=
sorry

-- Part 2: Showing m range for no losses in processing production
theorem part2 (m x : ℝ) (hx : x ∈ Set.Icc 10 20) : 
  (m ≤ (Real.log 41 - 2) / 40) ↔ ((1/2) * Real.log (2 * x + 1) - m * x ≥ (1/20) * x) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2382_238232


namespace NUMINAMATH_GPT_James_total_area_l2382_238290

theorem James_total_area :
  let initial_length := 13
  let initial_width := 18
  let increased_length := initial_length + 2
  let increased_width := initial_width + 2
  let single_room_area := increased_length * increased_width
  let four_rooms_area := 4 * single_room_area
  let larger_room_area := 2 * single_room_area
  let total_area := four_rooms_area + larger_room_area
  total_area = 1800 :=
by
  let initial_length := 13
  let initial_width := 18
  let increased_length := initial_length + 2
  let increased_width := initial_width + 2
  let single_room_area := increased_length * increased_width
  let four_rooms_area := 4 * single_room_area
  let larger_room_area := 2 * single_room_area
  let total_area := four_rooms_area + larger_room_area
  have h : total_area = 1800 := by sorry
  exact h

end NUMINAMATH_GPT_James_total_area_l2382_238290


namespace NUMINAMATH_GPT_prove_expression_l2382_238221

-- Define the operation for real numbers
def op (a b c : ℝ) : ℝ := (a - b + c) ^ 2

-- Stating the theorem for the given expression
theorem prove_expression (x z : ℝ) :
  op ((x + z) ^ 2) ((z - x) ^ 2) ((x - z) ^ 2) = (x + z) ^ 4 := 
by  sorry

end NUMINAMATH_GPT_prove_expression_l2382_238221


namespace NUMINAMATH_GPT_evaluate_g_at_3_l2382_238261

def g (x : ℝ) : ℝ := 9 * x^3 - 5 * x^2 + 3 * x - 7

theorem evaluate_g_at_3 : g 3 = 200 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_3_l2382_238261


namespace NUMINAMATH_GPT_func_equiv_l2382_238201

noncomputable def f (x : ℝ) : ℝ := if x = 0 then 0 else x + 1 / x

theorem func_equiv {a b : ℝ} (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) :
  (∀ x, f (2 * x) = a * f x + b * x) ∧ (∀ x y, y ≠ 0 → f x * f y = f (x * y) + f (x / y)) :=
sorry

end NUMINAMATH_GPT_func_equiv_l2382_238201


namespace NUMINAMATH_GPT_haman_dropped_trays_l2382_238297

def initial_trays_to_collect : ℕ := 10
def additional_trays : ℕ := 7
def eggs_sold : ℕ := 540
def eggs_per_tray : ℕ := 30

theorem haman_dropped_trays :
  ∃ dropped_trays : ℕ,
  (initial_trays_to_collect + additional_trays - dropped_trays)*eggs_per_tray = eggs_sold → dropped_trays = 8 :=
sorry

end NUMINAMATH_GPT_haman_dropped_trays_l2382_238297


namespace NUMINAMATH_GPT_length_of_cube_side_l2382_238286

theorem length_of_cube_side (SA : ℝ) (h₀ : SA = 600) (h₁ : SA = 6 * a^2) : a = 10 := by
  sorry

end NUMINAMATH_GPT_length_of_cube_side_l2382_238286


namespace NUMINAMATH_GPT_find_n_l2382_238212

theorem find_n (n : ℕ) (h_pos : n > 0) (h_ineq : n < Real.sqrt 65 ∧ Real.sqrt 65 < n + 1) : n = 8 := by sorry

end NUMINAMATH_GPT_find_n_l2382_238212


namespace NUMINAMATH_GPT_framed_painting_ratio_l2382_238237

theorem framed_painting_ratio (x : ℝ) (h : (15 + 2 * x) * (30 + 4 * x) = 900) : (15 + 2 * x) / (30 + 4 * x) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_framed_painting_ratio_l2382_238237


namespace NUMINAMATH_GPT_domain_of_f_l2382_238298

theorem domain_of_f (c : ℝ) :
  (∀ x : ℝ, -7 * x^2 + 5 * x + c ≠ 0) ↔ c < -25 / 28 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l2382_238298


namespace NUMINAMATH_GPT_evaluate_at_two_l2382_238259

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 10) / (x^2 - x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem evaluate_at_two : f (g 2) + g (f 2) = 38 / 7 := by
  sorry

end NUMINAMATH_GPT_evaluate_at_two_l2382_238259


namespace NUMINAMATH_GPT_percentage_difference_l2382_238214

theorem percentage_difference (y : ℝ) (h : y ≠ 0) (x z : ℝ) (hx : x = 5 * y) (hz : z = 1.20 * y) :
  ((z - y) / x * 100) = 4 :=
by
  rw [hz, hx]
  simp
  sorry

end NUMINAMATH_GPT_percentage_difference_l2382_238214


namespace NUMINAMATH_GPT_player_current_average_l2382_238229

theorem player_current_average
  (A : ℕ) -- Assume A is a natural number (non-negative)
  (cond1 : 10 * A + 78 = 11 * (A + 4)) :
  A = 34 :=
by
  sorry

end NUMINAMATH_GPT_player_current_average_l2382_238229


namespace NUMINAMATH_GPT_prime_p_and_cube_l2382_238216

noncomputable def p : ℕ := 307

theorem prime_p_and_cube (a : ℕ) (h : a^3 = 16 * p + 1) : 
  Nat.Prime p := by
  sorry

end NUMINAMATH_GPT_prime_p_and_cube_l2382_238216


namespace NUMINAMATH_GPT_a_plus_2b_eq_21_l2382_238223

-- Definitions and conditions based on the problem statement
def a_log_250_2_plus_b_log_250_5_eq_3 (a b : ℤ) : Prop :=
  a * Real.log 2 / Real.log 250 + b * Real.log 5 / Real.log 250 = 3

-- The theorem that needs to be proved
theorem a_plus_2b_eq_21 (a b : ℤ) (h : a_log_250_2_plus_b_log_250_5_eq_3 a b) : a + 2 * b = 21 := 
  sorry

end NUMINAMATH_GPT_a_plus_2b_eq_21_l2382_238223


namespace NUMINAMATH_GPT_rahim_books_second_shop_l2382_238256

variable (x : ℕ)

-- Definitions of the problem's conditions
def total_cost : ℕ := 520 + 248
def total_books (x : ℕ) : ℕ := 42 + x
def average_price : ℕ := 12

-- The problem statement in Lean 4
theorem rahim_books_second_shop : x = 22 → total_cost / total_books x = average_price :=
  sorry

end NUMINAMATH_GPT_rahim_books_second_shop_l2382_238256


namespace NUMINAMATH_GPT_prove_R36_div_R6_minus_R3_l2382_238252

noncomputable def R (k : ℕ) : ℤ := (10^k - 1) / 9

theorem prove_R36_div_R6_minus_R3 :
  (R 36 / R 6) - R 3 = 100000100000100000100000100000099989 := sorry

end NUMINAMATH_GPT_prove_R36_div_R6_minus_R3_l2382_238252


namespace NUMINAMATH_GPT_count_primes_with_digit_three_l2382_238234

def is_digit_three (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := Prime n

def primes_with_digit_three_count (lim : ℕ) (count : ℕ) : Prop :=
  ∀ n < lim, is_digit_three n → is_prime n → count = 9

theorem count_primes_with_digit_three (lim : ℕ) (count : ℕ) :
  primes_with_digit_three_count 150 9 := 
by
  sorry

end NUMINAMATH_GPT_count_primes_with_digit_three_l2382_238234


namespace NUMINAMATH_GPT_white_balls_count_l2382_238202

-- Definitions for the conditions
variable (x y : ℕ) 

-- Lean statement representing the problem
theorem white_balls_count : 
  x < y ∧ y < 2 * x ∧ 2 * x + 3 * y = 60 → x = 9 := 
sorry

end NUMINAMATH_GPT_white_balls_count_l2382_238202


namespace NUMINAMATH_GPT_ivy_covering_the_tree_l2382_238263

def ivy_stripped_per_day := 6
def ivy_grows_per_night := 2
def days_to_strip := 10
def net_ivy_stripped_per_day := ivy_stripped_per_day - ivy_grows_per_night

theorem ivy_covering_the_tree : net_ivy_stripped_per_day * days_to_strip = 40 := by
  have h1 : net_ivy_stripped_per_day = 4 := by
    unfold net_ivy_stripped_per_day
    rfl
  rw [h1]
  show 4 * 10 = 40
  rfl

end NUMINAMATH_GPT_ivy_covering_the_tree_l2382_238263


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2382_238236

theorem sufficient_but_not_necessary (x : ℝ) (h : 2 < x ∧ x < 3) :
  x * (x - 5) < 0 ∧ ∃ y, y * (y - 5) < 0 ∧ (2 ≤ y ∧ y ≤ 3) → False :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2382_238236


namespace NUMINAMATH_GPT_expand_and_simplify_l2382_238251

theorem expand_and_simplify (x : ℝ) : 20 * (3 * x - 4) = 60 * x - 80 := 
by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l2382_238251


namespace NUMINAMATH_GPT_steve_fraction_of_skylar_l2382_238244

variables (S : ℤ) (Stacy Skylar Steve : ℤ)

-- Given conditions
axiom h1 : 32 = 3 * Steve + 2 -- Stacy's berries = 2 + 3 * Steve's berries
axiom h2 : Skylar = 20        -- Skylar has 20 berries
axiom h3 : Stacy = 32         -- Stacy has 32 berries

-- Final goal
theorem steve_fraction_of_skylar (h1: 32 = 3 * Steve + 2) (h2: 20 = Skylar) (h3: Stacy = 32) :
  Steve = Skylar / 2 := 
sorry

end NUMINAMATH_GPT_steve_fraction_of_skylar_l2382_238244


namespace NUMINAMATH_GPT_smallest_perfect_square_gt_100_has_odd_number_of_factors_l2382_238279

theorem smallest_perfect_square_gt_100_has_odd_number_of_factors : 
  ∃ n : ℕ, (n > 100) ∧ (∃ k : ℕ, n = k * k) ∧ (∀ m > 100, ∃ t : ℕ, m = t * t → n ≤ m) := 
sorry

end NUMINAMATH_GPT_smallest_perfect_square_gt_100_has_odd_number_of_factors_l2382_238279


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l2382_238210

theorem angle_in_third_quadrant
  (α : ℝ) (hα : 270 < α ∧ α < 360) : 90 < 180 - α ∧ 180 - α < 180 :=
by
  sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l2382_238210


namespace NUMINAMATH_GPT_total_swimming_hours_over_4_weeks_l2382_238281

def weekday_swimming_per_day : ℕ := 2  -- Percy swims 2 hours per weekday
def weekday_days_per_week : ℕ := 5     -- Percy swims for 5 days a week
def weekend_swimming_per_week : ℕ := 3 -- Percy swims 3 hours on the weekend
def weeks : ℕ := 4                     -- The number of weeks is 4

-- Define the total swimming hours over 4 weeks
theorem total_swimming_hours_over_4_weeks :
  weekday_swimming_per_day * weekday_days_per_week * weeks + weekend_swimming_per_week * weeks = 52 :=
by
  sorry

end NUMINAMATH_GPT_total_swimming_hours_over_4_weeks_l2382_238281


namespace NUMINAMATH_GPT_last_digit_largest_prime_l2382_238235

-- Definition and conditions
def largest_known_prime : ℕ := 2^216091 - 1

-- The statement of the problem we want to prove
theorem last_digit_largest_prime : (largest_known_prime % 10) = 7 := by
  sorry

end NUMINAMATH_GPT_last_digit_largest_prime_l2382_238235


namespace NUMINAMATH_GPT_Iris_pairs_of_pants_l2382_238239

theorem Iris_pairs_of_pants (jacket_cost short_cost pant_cost total_spent n_jackets n_shorts n_pants : ℕ) :
  (jacket_cost = 10) →
  (short_cost = 6) →
  (pant_cost = 12) →
  (total_spent = 90) →
  (n_jackets = 3) →
  (n_shorts = 2) →
  (n_jackets * jacket_cost + n_shorts * short_cost + n_pants * pant_cost = total_spent) →
  (n_pants = 4) := 
by
  intros h_jacket_cost h_short_cost h_pant_cost h_total_spent h_n_jackets h_n_shorts h_eq
  sorry

end NUMINAMATH_GPT_Iris_pairs_of_pants_l2382_238239


namespace NUMINAMATH_GPT_sum_of_first_10_terms_l2382_238205

noncomputable def sum_first_n_terms (a_1 d : ℕ) (n : ℕ) : ℕ :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

theorem sum_of_first_10_terms (a : ℕ → ℕ) (a_2_a_4_sum : a 2 + a 4 = 4) (a_3_a_5_sum : a 3 + a 5 = 10) :
  sum_first_n_terms (a 1) (a 2 - a 1) 10 = 95 :=
  sorry

end NUMINAMATH_GPT_sum_of_first_10_terms_l2382_238205


namespace NUMINAMATH_GPT_cp_of_apple_l2382_238282

theorem cp_of_apple (SP : ℝ) (hSP : SP = 17) (loss_fraction : ℝ) (h_loss_fraction : loss_fraction = 1 / 6) : 
  ∃ CP : ℝ, CP = 20.4 ∧ SP = CP - loss_fraction * CP :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_cp_of_apple_l2382_238282


namespace NUMINAMATH_GPT_initial_water_amount_l2382_238270

theorem initial_water_amount (x : ℝ) (h : x + 6.8 = 9.8) : x = 3 := 
by
  sorry

end NUMINAMATH_GPT_initial_water_amount_l2382_238270


namespace NUMINAMATH_GPT_average_sitting_time_l2382_238217

theorem average_sitting_time (number_of_students : ℕ) (number_of_seats : ℕ) (total_travel_time : ℕ) 
  (h1 : number_of_students = 6) 
  (h2 : number_of_seats = 4) 
  (h3 : total_travel_time = 192) :
  (number_of_seats * total_travel_time) / number_of_students = 128 :=
by
  sorry

end NUMINAMATH_GPT_average_sitting_time_l2382_238217


namespace NUMINAMATH_GPT_min_moves_is_22_l2382_238211

def casket_coins : List ℕ := [9, 17, 12, 5, 18, 10, 20]

def target_coins (total_caskets : ℕ) (total_coins : ℕ) : ℕ :=
  total_coins / total_caskets

def total_caskets : ℕ := 7

def total_coins (coins : List ℕ) : ℕ :=
  coins.foldr (· + ·) 0

noncomputable def min_moves_to_equalize (coins : List ℕ) (target : ℕ) : ℕ := sorry

theorem min_moves_is_22 :
  min_moves_to_equalize casket_coins (target_coins total_caskets (total_coins casket_coins)) = 22 :=
sorry

end NUMINAMATH_GPT_min_moves_is_22_l2382_238211


namespace NUMINAMATH_GPT_perpendicular_vectors_implies_k_eq_2_l2382_238267

variable (k : ℝ)
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-1, k)

theorem perpendicular_vectors_implies_k_eq_2 (h : (2 : ℝ) * (-1 : ℝ) + (1 : ℝ) * k = 0) : k = 2 := by
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_implies_k_eq_2_l2382_238267


namespace NUMINAMATH_GPT_section_b_students_can_be_any_nonnegative_integer_l2382_238272

def section_a_students := 36
def avg_weight_section_a := 30
def avg_weight_section_b := 30
def avg_weight_whole_class := 30

theorem section_b_students_can_be_any_nonnegative_integer (x : ℕ) :
  let total_weight_section_a := section_a_students * avg_weight_section_a
  let total_weight_section_b := x * avg_weight_section_b
  let total_weight_whole_class := (section_a_students + x) * avg_weight_whole_class
  (total_weight_section_a + total_weight_section_b = total_weight_whole_class) :=
by 
  sorry

end NUMINAMATH_GPT_section_b_students_can_be_any_nonnegative_integer_l2382_238272


namespace NUMINAMATH_GPT_find_n_l2382_238213

theorem find_n (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ) (h : (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 + (1 + x)^6 + (1 + x)^7
                      = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7)
  (h_sum : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 29 - 7) : 7 = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2382_238213


namespace NUMINAMATH_GPT_combined_gross_profit_correct_l2382_238254

def calculate_final_selling_price (initial_price : ℝ) (markup : ℝ) (discounts : List ℝ) : ℝ :=
  let marked_up_price := initial_price * (1 + markup)
  let final_price := List.foldl (λ price discount => price * (1 - discount)) marked_up_price discounts
  final_price

def calculate_gross_profit (initial_price : ℝ) (markup : ℝ) (discounts : List ℝ) : ℝ :=
  calculate_final_selling_price initial_price markup discounts - initial_price

noncomputable def combined_gross_profit : ℝ :=
  let earrings_gross_profit := calculate_gross_profit 240 0.25 [0.15]
  let bracelet_gross_profit := calculate_gross_profit 360 0.30 [0.10, 0.05]
  let necklace_gross_profit := calculate_gross_profit 480 0.40 [0.20, 0.05]
  let ring_gross_profit := calculate_gross_profit 600 0.35 [0.10, 0.05, 0.02]
  let pendant_gross_profit := calculate_gross_profit 720 0.50 [0.20, 0.03, 0.07]
  earrings_gross_profit + bracelet_gross_profit + necklace_gross_profit + ring_gross_profit + pendant_gross_profit

theorem combined_gross_profit_correct : combined_gross_profit = 224.97 :=
  by
  sorry

end NUMINAMATH_GPT_combined_gross_profit_correct_l2382_238254


namespace NUMINAMATH_GPT_incorrect_directions_of_opening_l2382_238292

-- Define the functions
def f (x : ℝ) : ℝ := 2 * (x - 3)^2
def g (x : ℝ) : ℝ := -2 * (x - 3)^2

-- The theorem (statement) to prove
theorem incorrect_directions_of_opening :
  ¬(∀ x, (f x > 0 ∧ g x > 0) ∨ (f x < 0 ∧ g x < 0)) :=
sorry

end NUMINAMATH_GPT_incorrect_directions_of_opening_l2382_238292


namespace NUMINAMATH_GPT_game_result_l2382_238285

theorem game_result (a : ℤ) : ((2 * a + 6) / 2 - a = 3) :=
by
  sorry

end NUMINAMATH_GPT_game_result_l2382_238285


namespace NUMINAMATH_GPT_roots_of_modified_quadratic_l2382_238227

theorem roots_of_modified_quadratic 
  (k : ℝ) (hk : 0 < k) :
  (∃ z₁ z₂ : ℂ, (12 * z₁^2 - 4 * I * z₁ - k = 0) ∧ (12 * z₂^2 - 4 * I * z₂ - k = 0) ∧ (z₁ ≠ z₂) ∧ (z₁.im = 0) ∧ (z₂.im ≠ 0)) ↔ (k = 1/4) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_modified_quadratic_l2382_238227


namespace NUMINAMATH_GPT_probability_one_white_ball_initial_find_n_if_one_red_ball_l2382_238220

-- Define the initial conditions: 5 red balls and 3 white balls
def initial_red_balls := 5
def initial_white_balls := 3
def total_initial_balls := initial_red_balls + initial_white_balls

-- Define the probability of drawing exactly one white ball initially
def prob_draw_one_white := initial_white_balls / total_initial_balls

-- Define the number of white balls added
variable (n : ℕ)

-- Define the total number of balls after adding n white balls
def total_balls_after_adding := total_initial_balls + n

-- Define the probability of drawing exactly one red ball after adding n white balls
def prob_draw_one_red := initial_red_balls / total_balls_after_adding

-- Prove that the probability of drawing one white ball initially is 3/8
theorem probability_one_white_ball_initial : prob_draw_one_white = 3 / 8 := by
  sorry

-- Prove that, if the probability of drawing one red ball after adding n white balls is 1/2, then n = 2
theorem find_n_if_one_red_ball : prob_draw_one_red = 1 / 2 -> n = 2 := by
  sorry

end NUMINAMATH_GPT_probability_one_white_ball_initial_find_n_if_one_red_ball_l2382_238220


namespace NUMINAMATH_GPT_remaining_bottles_l2382_238231

variable (s : ℕ) (b : ℕ) (ps : ℚ) (pb : ℚ)

theorem remaining_bottles (h1 : s = 6000) (h2 : b = 14000) (h3 : ps = 0.20) (h4 : pb = 0.23) : 
  s - Nat.floor (ps * s) + b - Nat.floor (pb * b) = 15580 :=
by
  sorry

end NUMINAMATH_GPT_remaining_bottles_l2382_238231


namespace NUMINAMATH_GPT_largest_stamps_per_page_l2382_238219

theorem largest_stamps_per_page (a b c : ℕ) (h1 : a = 924) (h2 : b = 1260) (h3 : c = 1386) : 
  Nat.gcd (Nat.gcd a b) c = 42 := by
  sorry

end NUMINAMATH_GPT_largest_stamps_per_page_l2382_238219


namespace NUMINAMATH_GPT_smallest_x_condition_l2382_238245

theorem smallest_x_condition (x : ℕ) : (∃ x > 0, (3 * x + 28)^2 % 53 = 0) -> x = 26 := 
by
  sorry

end NUMINAMATH_GPT_smallest_x_condition_l2382_238245


namespace NUMINAMATH_GPT_circles_disjoint_l2382_238200

theorem circles_disjoint :
  ∀ (x y u v : ℝ),
  (x^2 + y^2 = 1) →
  ((u-2)^2 + (v+2)^2 = 1) →
  (2^2 + (-2)^2) > (1 + 1)^2 :=
by sorry

end NUMINAMATH_GPT_circles_disjoint_l2382_238200


namespace NUMINAMATH_GPT_bird_families_flew_away_l2382_238255

def initial_families : ℕ := 41
def left_families : ℕ := 14

theorem bird_families_flew_away :
  initial_families - left_families = 27 :=
by
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_bird_families_flew_away_l2382_238255


namespace NUMINAMATH_GPT_num_friends_solved_problems_l2382_238215

theorem num_friends_solved_problems (x y n : ℕ) (h1 : 24 * x + 28 * y = 256) (h2 : n = x + y) : n = 10 :=
by
  -- Begin the placeholder proof
  sorry

end NUMINAMATH_GPT_num_friends_solved_problems_l2382_238215


namespace NUMINAMATH_GPT_problem_solution_1_problem_solution_2_l2382_238240

def Sn (n : ℕ) := n * (n + 2)

def a_n (n : ℕ) := 2 * n + 1

def b_n (n : ℕ) := 2 ^ (n - 1)

def c_n (n : ℕ) := if n % 2 = 1 then 2 / Sn n else b_n n

def T_n (n : ℕ) : ℤ := (Finset.range n).sum (λ i => c_n (i + 1))

theorem problem_solution_1 : 
  ∀ (n : ℕ), a_n n = 2 * n + 1 ∧ b_n n = 2 ^ (n - 1) := 
  by sorry

theorem problem_solution_2 (n : ℕ) : 
  T_n (2 * n) = (2 * n) / (2 * n + 1) + (2 / 3) * (4 ^ n - 1) := 
  by sorry

end NUMINAMATH_GPT_problem_solution_1_problem_solution_2_l2382_238240


namespace NUMINAMATH_GPT_system_equations_solution_l2382_238242

theorem system_equations_solution (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  (1 / x + 1 / y + 1 / z = 3) ∧ 
  (1 / (x * y) + 1 / (y * z) + 1 / (z * x) = 3) ∧ 
  (1 / (x * y * z) = 1) → 
  x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end NUMINAMATH_GPT_system_equations_solution_l2382_238242


namespace NUMINAMATH_GPT_solve_linear_system_l2382_238287

theorem solve_linear_system :
  ∃ (x1 x2 x3 : ℚ), 
  (2 * x1 + 5 * x2 - 4 * x3 = 8) ∧ 
  (3 * x1 + 15 * x2 - 9 * x3 = 5) ∧ 
  (5 * x1 + 5 * x2 - 7 * x3 = 27) ∧
  (x1 = 19 / 3 + x3) ∧ 
  (x2 = -14 / 15 + 2 / 5 * x3) := 
by 
  sorry

end NUMINAMATH_GPT_solve_linear_system_l2382_238287


namespace NUMINAMATH_GPT_work_required_to_lift_satellite_l2382_238289

noncomputable def satellite_lifting_work (m H R3 g : ℝ) : ℝ :=
  m * g * R3^2 * ((1 / R3) - (1 / (R3 + H)))

theorem work_required_to_lift_satellite :
  satellite_lifting_work (7.0 * 10^3) (200 * 10^3) (6380 * 10^3) 10 = 13574468085 :=
by sorry

end NUMINAMATH_GPT_work_required_to_lift_satellite_l2382_238289


namespace NUMINAMATH_GPT_infinite_3_stratum_numbers_l2382_238241

-- Condition for 3-stratum number
def is_3_stratum_number (n : ℕ) : Prop :=
  ∃ (A B C : Finset ℕ), A ∪ B ∪ C = (Finset.range (n + 1)).filter (λ x => n % x = 0) ∧
  A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧
  A.sum id = B.sum id ∧ B.sum id = C.sum id

-- Part (a): Find a 3-stratum number
example : is_3_stratum_number 120 := sorry

-- Part (b): Prove there are infinitely many 3-stratum numbers
theorem infinite_3_stratum_numbers : ∃ (f : ℕ → ℕ), ∀ n, is_3_stratum_number (f n) := sorry

end NUMINAMATH_GPT_infinite_3_stratum_numbers_l2382_238241


namespace NUMINAMATH_GPT_rook_reaches_right_total_rook_reaches_right_seven_moves_l2382_238264

-- Definition of the conditions for the problem
def rook_ways_total (n : Nat) :=
  2 ^ (n - 2)

def rook_ways_in_moves (n k : Nat) :=
  Nat.choose (n - 2) (k - 1)

-- Proof problem statements
theorem rook_reaches_right_total : rook_ways_total 30 = 2 ^ 28 := 
by sorry

theorem rook_reaches_right_seven_moves : rook_ways_in_moves 30 7 = Nat.choose 28 6 := 
by sorry

end NUMINAMATH_GPT_rook_reaches_right_total_rook_reaches_right_seven_moves_l2382_238264


namespace NUMINAMATH_GPT_temperature_difference_l2382_238283

theorem temperature_difference (t_low t_high : ℝ) (h_low : t_low = -2) (h_high : t_high = 5) :
  t_high - t_low = 7 :=
by
  rw [h_low, h_high]
  norm_num

end NUMINAMATH_GPT_temperature_difference_l2382_238283


namespace NUMINAMATH_GPT_find_expression_for_an_l2382_238226

-- Definitions for the problem conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def problem_conditions (a : ℕ → ℝ) (q : ℝ) :=
  geometric_sequence a q ∧
  a 1 + a 3 = 10 ∧
  a 2 + a 4 = 5

-- Statement of the problem
theorem find_expression_for_an (a : ℕ → ℝ) (q : ℝ) :
  problem_conditions a q → ∀ n : ℕ, a n = 2 ^ (4 - n) :=
sorry

end NUMINAMATH_GPT_find_expression_for_an_l2382_238226


namespace NUMINAMATH_GPT_beta_minus_alpha_l2382_238228

open Real

noncomputable def vector_a (α : ℝ) := (cos α, sin α)
noncomputable def vector_b (β : ℝ) := (cos β, sin β)

theorem beta_minus_alpha (α β : ℝ)
  (h₁ : 0 < α)
  (h₂ : α < β)
  (h₃ : β < π)
  (h₄ : |2 * vector_a α + vector_b β| = |vector_a α - 2 * vector_b β|) :
  β - α = π / 2 :=
sorry

end NUMINAMATH_GPT_beta_minus_alpha_l2382_238228


namespace NUMINAMATH_GPT_fraction_yellow_surface_area_l2382_238203

theorem fraction_yellow_surface_area
  (cube_edge : ℕ)
  (small_cubes : ℕ)
  (yellow_cubes : ℕ)
  (total_surface_area : ℕ)
  (yellow_surface_area : ℕ)
  (fraction_yellow : ℚ) :
  cube_edge = 4 ∧
  small_cubes = 64 ∧
  yellow_cubes = 15 ∧
  total_surface_area = 6 * cube_edge * cube_edge ∧
  yellow_surface_area = 16 ∧
  fraction_yellow = yellow_surface_area / total_surface_area →
  fraction_yellow = 1/6 :=
by
  sorry

end NUMINAMATH_GPT_fraction_yellow_surface_area_l2382_238203


namespace NUMINAMATH_GPT_range_g_l2382_238262

variable (x : ℝ)
noncomputable def g (x : ℝ) : ℝ := 1 / (x - 1)^2

theorem range_g (y : ℝ) : 
  (∃ x, g x = y) ↔ y > 0 :=
by
  sorry

end NUMINAMATH_GPT_range_g_l2382_238262


namespace NUMINAMATH_GPT_max_area_of_triangle_l2382_238222

theorem max_area_of_triangle (a b c : ℝ) (hC : C = 60) (h1 : 3 * a * b = 25 - c^2) :
  (∃ S : ℝ, S = (a * b * (Real.sqrt 3)) / 4 ∧ S = 25 * (Real.sqrt 3) / 16) :=
sorry

end NUMINAMATH_GPT_max_area_of_triangle_l2382_238222


namespace NUMINAMATH_GPT_exists_a_l2382_238296

noncomputable def a : ℕ → ℕ := sorry

theorem exists_a : a (a (a (a 1))) = 458329 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_exists_a_l2382_238296


namespace NUMINAMATH_GPT_initial_bananas_l2382_238209

theorem initial_bananas (bananas_left: ℕ) (eaten: ℕ) (basket: ℕ) 
                        (h_left: bananas_left = 100) 
                        (h_eaten: eaten = 70) 
                        (h_basket: basket = 2 * eaten): 
  bananas_left + eaten + basket = 310 :=
by
  sorry

end NUMINAMATH_GPT_initial_bananas_l2382_238209


namespace NUMINAMATH_GPT_riverton_soccer_physics_l2382_238248

theorem riverton_soccer_physics : 
  let total_players := 15
  let math_players := 9
  let both_subjects := 3
  let only_physics := total_players - math_players
  let physics_players := only_physics + both_subjects
  physics_players = 9 :=
by
  sorry

end NUMINAMATH_GPT_riverton_soccer_physics_l2382_238248


namespace NUMINAMATH_GPT_solve_problem_l2382_238204

def f (x : ℝ) : ℝ := x^2 - 4*x + 7
def g (x : ℝ) : ℝ := 2*x + 1

theorem solve_problem : f (g 3) - g (f 3) = 19 := by
  sorry

end NUMINAMATH_GPT_solve_problem_l2382_238204


namespace NUMINAMATH_GPT_expression_evaluation_l2382_238274

theorem expression_evaluation :
  (3 : ℝ) + 3 * Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (3 - Real.sqrt 3)) = 4 + 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_expression_evaluation_l2382_238274


namespace NUMINAMATH_GPT_river_current_speed_l2382_238275

/-- A man rows 18 miles upstream in three hours more time than it takes him to row 
the same distance downstream. If he halves his usual rowing rate, the time upstream 
becomes only two hours more than the time downstream. Prove that the speed of 
the river's current is 2 miles per hour. -/
theorem river_current_speed (r w : ℝ) 
    (h1 : 18 / (r - w) - 18 / (r + w) = 3)
    (h2 : 18 / (r / 2 - w) - 18 / (r / 2 + w) = 2) : 
    w = 2 := 
sorry

end NUMINAMATH_GPT_river_current_speed_l2382_238275


namespace NUMINAMATH_GPT_marble_prob_l2382_238288

theorem marble_prob (T : ℕ) (hT1 : T > 12) 
  (hP : ((T - 12) / T : ℚ) * ((T - 12) / T) = 36 / 49) : T = 84 :=
sorry

end NUMINAMATH_GPT_marble_prob_l2382_238288


namespace NUMINAMATH_GPT_inscribed_square_sum_c_d_eq_200689_l2382_238250

theorem inscribed_square_sum_c_d_eq_200689 :
  ∃ (c d : ℕ), Nat.gcd c d = 1 ∧ (∃ x : ℚ, x = (c : ℚ) / (d : ℚ) ∧ 
    let a := 48
    let b := 55
    let longest_side := 73
    let s := (a + b + longest_side) / 2
    let area := Real.sqrt (s * (s - a) * (s - b) * (s - longest_side))
    area = 1320 ∧ x = 192720 / 7969 ∧ c + d = 200689) :=
sorry

end NUMINAMATH_GPT_inscribed_square_sum_c_d_eq_200689_l2382_238250


namespace NUMINAMATH_GPT_function_solution_l2382_238299

theorem function_solution (f : ℝ → ℝ) (α : ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) →
  (∀ x : ℝ, f x = x ^ 3 + α * x) :=
by
  sorry

end NUMINAMATH_GPT_function_solution_l2382_238299


namespace NUMINAMATH_GPT_sum_of_digits_is_11_l2382_238253

def digits_satisfy_conditions (A B C : ℕ) : Prop :=
  (C = 0 ∨ C = 5) ∧
  (A = 2 * B) ∧
  (A * B * C = 40)

theorem sum_of_digits_is_11 (A B C : ℕ) (h : digits_satisfy_conditions A B C) : A + B + C = 11 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_is_11_l2382_238253


namespace NUMINAMATH_GPT_soccer_substitutions_mod_2000_l2382_238295

theorem soccer_substitutions_mod_2000 :
  let a_0 := 1
  let a_1 := 11 * 11
  let a_2 := 11 * 10 * a_1
  let a_3 := 11 * 9 * a_2
  let a_4 := 11 * 8 * a_3
  let n := a_0 + a_1 + a_2 + a_3 + a_4
  n % 2000 = 942 :=
by
  sorry

end NUMINAMATH_GPT_soccer_substitutions_mod_2000_l2382_238295


namespace NUMINAMATH_GPT_smallest_x_mod_conditions_l2382_238207

theorem smallest_x_mod_conditions :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 4 ∧ x % 6 = 5 ∧ x % 7 = 6 ∧ x = 209 := by
  sorry

end NUMINAMATH_GPT_smallest_x_mod_conditions_l2382_238207


namespace NUMINAMATH_GPT_blue_sequins_per_row_l2382_238260

theorem blue_sequins_per_row : 
  ∀ (B : ℕ),
  (6 * B) + (5 * 12) + (9 * 6) = 162 → B = 8 :=
by
  intro B
  sorry

end NUMINAMATH_GPT_blue_sequins_per_row_l2382_238260


namespace NUMINAMATH_GPT_sister_sandcastle_height_l2382_238249

theorem sister_sandcastle_height (miki_height : ℝ)
                                (height_diff : ℝ)
                                (h_miki : miki_height = 0.8333333333333334)
                                (h_diff : height_diff = 0.3333333333333333) :
  miki_height - height_diff = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_sister_sandcastle_height_l2382_238249
