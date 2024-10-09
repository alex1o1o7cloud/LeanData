import Mathlib

namespace percentage_of_difference_is_50_l880_88073

noncomputable def percentage_of_difference (x y : ℝ) (p : ℝ) :=
  (p / 100) * (x - y) = 0.20 * (x + y)

noncomputable def y_is_percentage_of_x (x y : ℝ) :=
  y = 0.42857142857142854 * x

theorem percentage_of_difference_is_50 (x y : ℝ) (p : ℝ)
  (h1 : percentage_of_difference x y p)
  (h2 : y_is_percentage_of_x x y) :
  p = 50 :=
by
  sorry

end percentage_of_difference_is_50_l880_88073


namespace nickys_running_pace_l880_88017

theorem nickys_running_pace (head_start : ℕ) (pace_cristina : ℕ) (time_nicky : ℕ) (distance_meet : ℕ) :
  head_start = 12 →
  pace_cristina = 5 →
  time_nicky = 30 →
  distance_meet = (pace_cristina * (time_nicky - head_start)) →
  (distance_meet / time_nicky = 3) :=
by
  intros h_start h_pace_c h_time_n d_meet
  sorry

end nickys_running_pace_l880_88017


namespace is_composite_1010_pattern_l880_88053

theorem is_composite_1010_pattern (k : ℕ) (h : k ≥ 2) : (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (1010^k + 101 = a * b)) :=
  sorry

end is_composite_1010_pattern_l880_88053


namespace total_revenue_full_price_l880_88085

theorem total_revenue_full_price (f d p : ℕ) (h1 : f + d = 200) (h2 : f * p + d * (3 * p) / 4 = 2800) : 
  f * p = 680 :=
by
  -- proof omitted
  sorry

end total_revenue_full_price_l880_88085


namespace ava_planted_9_trees_l880_88025

theorem ava_planted_9_trees
  (L : ℕ)
  (hAva : ∀ L, Ava = L + 3)
  (hTotal : L + (L + 3) = 15) : 
  Ava = 9 :=
by
  sorry

end ava_planted_9_trees_l880_88025


namespace combined_seq_20th_term_l880_88014

def arithmetic_seq (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d
def geometric_seq (g : ℕ) (r : ℕ) (n : ℕ) : ℕ := g * r^(n - 1)

theorem combined_seq_20th_term :
  let a := 3
  let d := 4
  let g := 2
  let r := 2
  let n := 20
  arithmetic_seq a d n + geometric_seq g r n = 1048655 :=
by 
  sorry

end combined_seq_20th_term_l880_88014


namespace solution_set_l880_88035

variable {f : ℝ → ℝ}
variable (h1 : ∀ x, x < 0 → x * deriv f x - 2 * f x > 0)
variable (h2 : ∀ x, x < 0 → f x ≠ 0)

theorem solution_set (h3 : ∀ x, -2024 < x ∧ x < -2023 → f (x + 2023) - (x + 2023)^2 * f (-1) < 0) :
    {x : ℝ | f (x + 2023) - (x + 2023)^2 * f (-1) < 0} = {x : ℝ | -2024 < x ∧ x < -2023} :=
by
  sorry

end solution_set_l880_88035


namespace triangles_with_vertex_A_l880_88034

theorem triangles_with_vertex_A : 
  ∃ (A : Point) (remaining_points : Finset Point), 
    (remaining_points.card = 8) → 
    (∃ (n : ℕ), n = (Nat.choose 8 2) ∧ n = 28) :=
by
  sorry

end triangles_with_vertex_A_l880_88034


namespace quadratic_completion_l880_88050

theorem quadratic_completion 
    (x : ℝ) 
    (h : 16*x^2 - 32*x - 512 = 0) : 
    ∃ r s : ℝ, (x + r)^2 = s ∧ s = 33 :=
by sorry

end quadratic_completion_l880_88050


namespace number_of_ways_to_divide_friends_l880_88027

theorem number_of_ways_to_divide_friends :
  (4 ^ 8 = 65536) := by
  -- result obtained through the multiplication principle
  sorry

end number_of_ways_to_divide_friends_l880_88027


namespace find_x_l880_88036

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end find_x_l880_88036


namespace value_of_4k_minus_1_l880_88009

theorem value_of_4k_minus_1 (k x y : ℝ)
  (h1 : x + y - 5 * k = 0)
  (h2 : x - y - 9 * k = 0)
  (h3 : 2 * x + 3 * y = 6) :
  4 * k - 1 = 2 :=
  sorry

end value_of_4k_minus_1_l880_88009


namespace xiao_hua_spent_7_yuan_l880_88077

theorem xiao_hua_spent_7_yuan :
  ∃ (a b c d: ℕ), a + b + c + d = 30 ∧
                   ((a = 5 ∧ b = 5 ∧ c = 10 ∧ d = 10) ∨
                    (a = 5 ∧ b = 10 ∧ c = 5 ∧ d = 10) ∨
                    (a = 10 ∧ b = 5 ∧ c = 5 ∧ d = 10) ∨
                    (a = 10 ∧ b = 10 ∧ c = 5 ∧ d = 5) ∨
                    (a = 5 ∧ b = 10 ∧ c = 10 ∧ d = 5) ∨
                    (a = 10 ∧ b = 5 ∧ c = 10 ∧ d = 5)) ∧
                   10 * c + 15 * a + 25 * b + 40 * d = 700 :=
by {
  sorry
}

end xiao_hua_spent_7_yuan_l880_88077


namespace car_speed_first_hour_l880_88021

theorem car_speed_first_hour
  (x : ℕ)
  (speed_second_hour : ℕ := 80)
  (average_speed : ℕ := 90)
  (total_time : ℕ := 2)
  (h : average_speed = (x + speed_second_hour) / total_time) :
  x = 100 :=
by
  sorry

end car_speed_first_hour_l880_88021


namespace x_one_minus_f_eq_one_l880_88019

noncomputable def x : ℝ := (1 + Real.sqrt 2) ^ 500
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem x_one_minus_f_eq_one : x * (1 - f) = 1 :=
by
  sorry

end x_one_minus_f_eq_one_l880_88019


namespace unique_rectangle_exists_l880_88062

theorem unique_rectangle_exists (a b : ℝ) (h_ab : a < b) :
  ∃! (x y : ℝ), x < b ∧ y < b ∧ x + y = (a + b) / 2 ∧ x * y = (a * b) / 4 :=
by
  sorry

end unique_rectangle_exists_l880_88062


namespace least_possible_value_l880_88063

theorem least_possible_value (y q p : ℝ) (h1: 5 < y) (h2: y < 7)
  (hq: q = 7) (hp: p = 5) : q - p = 2 :=
by
  sorry

end least_possible_value_l880_88063


namespace Sandra_brought_20_pairs_l880_88020

-- Definitions for given conditions
variable (S : ℕ) -- S for Sandra's pairs of socks
variable (C : ℕ) -- C for Lisa's cousin's pairs of socks

-- Conditions translated into Lean definitions
def initial_pairs : ℕ := 12
def mom_pairs : ℕ := 3 * initial_pairs + 8 -- Lisa's mom brought 8 more than three times the number of pairs Lisa started with
def cousin_pairs (S : ℕ) : ℕ := S / 5       -- Lisa's cousin brought one-fifth the number of pairs that Sandra bought
def total_pairs (S : ℕ) : ℕ := initial_pairs + S + cousin_pairs S + mom_pairs -- Total pairs of socks Lisa ended up with

-- The theorem to prove
theorem Sandra_brought_20_pairs (h : total_pairs S = 80) : S = 20 :=
by
  sorry

end Sandra_brought_20_pairs_l880_88020


namespace number_of_three_digit_multiples_of_9_with_odd_digits_l880_88066

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

def consists_only_of_odd_digits (n : ℕ) : Prop :=
  (∀ d ∈ (n.digits 10), d % 2 = 1)

theorem number_of_three_digit_multiples_of_9_with_odd_digits :
  ∃ t, t = 11 ∧
  (∀ n, is_three_digit_number n ∧ is_multiple_of_9 n ∧ consists_only_of_odd_digits n) → 1 ≤ t ∧ t ≤ 11 :=
sorry

end number_of_three_digit_multiples_of_9_with_odd_digits_l880_88066


namespace walt_age_l880_88044

-- Conditions
variables (T W : ℕ)
axiom h1 : T = 3 * W
axiom h2 : T + 12 = 2 * (W + 12)

-- Goal: Prove W = 12
theorem walt_age : W = 12 :=
sorry

end walt_age_l880_88044


namespace alice_daily_savings_l880_88042

theorem alice_daily_savings :
  ∀ (d total_days : ℕ) (dime_value : ℝ),
  d = 4 → total_days = 40 → dime_value = 0.10 →
  (d * dime_value) / total_days = 0.01 :=
by
  intros d total_days dime_value h_d h_total_days h_dime_value
  sorry

end alice_daily_savings_l880_88042


namespace simplify_expression_l880_88057

theorem simplify_expression (a b : ℕ) (h₁ : a = 2999) (h₂ : b = 3000) :
  b^3 - a * b^2 - a^2 * b + a^3 = 5999 :=
by 
  sorry

end simplify_expression_l880_88057


namespace coins_in_box_l880_88095

theorem coins_in_box (n : ℕ) 
    (h1 : n % 8 = 7) 
    (h2 : n % 7 = 5) : 
    n = 47 ∧ (47 % 9 = 2) :=
sorry

end coins_in_box_l880_88095


namespace sculpture_cost_in_chinese_yuan_l880_88074

theorem sculpture_cost_in_chinese_yuan
  (usd_to_nad : ℝ)
  (usd_to_cny : ℝ)
  (cost_nad : ℝ)
  (h1 : usd_to_nad = 8)
  (h2 : usd_to_cny = 5)
  (h3 : cost_nad = 160) :
  (cost_nad / usd_to_nad) * usd_to_cny = 100 :=
by
  sorry

end sculpture_cost_in_chinese_yuan_l880_88074


namespace prob_neither_defective_l880_88011

-- Definitions for the conditions
def totalPens : ℕ := 8
def defectivePens : ℕ := 2
def nonDefectivePens : ℕ := totalPens - defectivePens
def selectedPens : ℕ := 2

-- Theorem statement for the probability that neither of the two selected pens is defective
theorem prob_neither_defective : 
  (nonDefectivePens / totalPens) * ((nonDefectivePens - 1) / (totalPens - 1)) = 15 / 28 := 
  sorry

end prob_neither_defective_l880_88011


namespace determine_g_x2_l880_88047

noncomputable def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem determine_g_x2 (x : ℝ) (h : x^2 ≠ 4) : g (x^2) = (2 * x^2 + 3) / (x^2 - 2) :=
by sorry

end determine_g_x2_l880_88047


namespace range_of_abs_2z_minus_1_l880_88001

open Complex

theorem range_of_abs_2z_minus_1
  (z : ℂ)
  (h : abs (z + 2 - I) = 1) :
  abs (2 * z - 1) ∈ Set.Icc (Real.sqrt 29 - 2) (Real.sqrt 29 + 2) :=
sorry

end range_of_abs_2z_minus_1_l880_88001


namespace cans_per_person_on_second_day_l880_88038

theorem cans_per_person_on_second_day :
  ∀ (initial_stock : ℕ) (people_first_day : ℕ) (cans_taken_first_day : ℕ)
    (restock_first_day : ℕ) (people_second_day : ℕ)
    (restock_second_day : ℕ) (total_cans_given : ℕ) (cans_per_person_second_day : ℚ),
    cans_taken_first_day = 1 →
    initial_stock = 2000 →
    people_first_day = 500 →
    restock_first_day = 1500 →
    people_second_day = 1000 →
    restock_second_day = 3000 →
    total_cans_given = 2500 →
    cans_per_person_second_day = total_cans_given / people_second_day →
    cans_per_person_second_day = 2.5 := by
  sorry

end cans_per_person_on_second_day_l880_88038


namespace incorrect_calculation_l880_88052

theorem incorrect_calculation
  (ξ η : ℝ)
  (Eξ : ℝ)
  (Eη : ℝ)
  (E_min : ℝ)
  (hEξ : Eξ = 3)
  (hEη : Eη = 5)
  (hE_min : E_min = 3.67) :
  E_min > Eξ :=
by
  sorry

end incorrect_calculation_l880_88052


namespace cos_2alpha_value_l880_88056

noncomputable def cos_double_angle (α : ℝ) : ℝ := Real.cos (2 * α)

theorem cos_2alpha_value (α : ℝ): 
  (∃ a : ℝ, α = Real.arctan (-3) + 2 * a * Real.pi) → cos_double_angle α = -4 / 5 :=
by
  intro h
  sorry

end cos_2alpha_value_l880_88056


namespace contest_paths_correct_l880_88072

noncomputable def count_contest_paths : Nat := sorry

theorem contest_paths_correct : count_contest_paths = 127 := sorry

end contest_paths_correct_l880_88072


namespace expand_expression_l880_88082

theorem expand_expression (x y : ℝ) : (3 * x + 15) * (4 * y + 12) = 12 * x * y + 36 * x + 60 * y + 180 := 
  sorry

end expand_expression_l880_88082


namespace cost_of_flowers_cost_function_minimum_cost_l880_88003

-- Define the costs in terms of yuan
variables (n m : ℕ) -- n is the cost of one lily, m is the cost of one carnation.

-- Define the conditions
axiom cost_condition1 : 2 * n + m = 14
axiom cost_condition2 : 3 * m = 2 * n + 2

-- Prove the cost of one carnation and one lily
theorem cost_of_flowers : n = 5 ∧ m = 4 :=
by {
  sorry
}

-- Variables for the second part
variables (w x : ℕ) -- w is the total cost, x is the number of carnations.

-- Define the conditions
axiom total_condition : 11 = 2 + x + (11 - x)
axiom min_lilies_condition : 11 - x ≥ 2

-- State the relationship between w and x
theorem cost_function : w = 55 - x :=
by {
  sorry
}

-- Prove the minimum cost
theorem minimum_cost : ∃ x, (x ≤ 9 ∧  w = 46) :=
by {
  sorry
}

end cost_of_flowers_cost_function_minimum_cost_l880_88003


namespace prime_numbers_r_s_sum_l880_88024

theorem prime_numbers_r_s_sum (p q r s : ℕ) (hp : Fact (Nat.Prime p)) (hq : Fact (Nat.Prime q)) 
  (hr : Fact (Nat.Prime r)) (hs : Fact (Nat.Prime s)) (h1 : p < q) (h2 : q < r) (h3 : r < s) 
  (eqn : p * q * r * s + 1 = 4^(p + q)) : r + s = 274 :=
by
  sorry

end prime_numbers_r_s_sum_l880_88024


namespace degrees_to_radians_l880_88094

theorem degrees_to_radians (deg: ℝ) (h : deg = 120) : deg * (π / 180) = 2 * π / 3 :=
by
  simp [h]
  sorry

end degrees_to_radians_l880_88094


namespace initial_bananas_per_child_l880_88048

theorem initial_bananas_per_child : 
  ∀ (B n m x : ℕ), 
  n = 740 → 
  m = 370 → 
  (B = n * x) → 
  (B = (n - m) * (x + 2)) → 
  x = 2 := 
by
  intros B n m x h1 h2 h3 h4
  sorry

end initial_bananas_per_child_l880_88048


namespace largest_m_for_negative_integral_solutions_l880_88023

theorem largest_m_for_negative_integral_solutions :
  ∃ m : ℕ, (∀ p q : ℤ, 10 * p * p + (-m) * p + 560 = 0 ∧ p < 0 ∧ q < 0 ∧ p * q = 56 → m ≤ 570) ∧ m = 570 :=
sorry

end largest_m_for_negative_integral_solutions_l880_88023


namespace least_positive_integer_reducible_fraction_l880_88060

theorem least_positive_integer_reducible_fraction :
  ∃ n : ℕ, n > 0 ∧ gcd (n - 17) (7 * n + 4) > 1 ∧ (∀ m : ℕ, m > 0 ∧ gcd (m - 17) (7 * m + 4) > 1 → n ≤ m) :=
by sorry

end least_positive_integer_reducible_fraction_l880_88060


namespace abs_of_neg_square_add_l880_88071

theorem abs_of_neg_square_add (a b : ℤ) : |-a^2 + b| = 10 :=
by
  sorry

end abs_of_neg_square_add_l880_88071


namespace polynomial_determination_l880_88088

theorem polynomial_determination (P : Polynomial ℝ) :
  (∀ X : ℝ, P.eval (X^2) = (X^2 + 1) * P.eval X) →
  (∃ a : ℝ, ∀ X : ℝ, P.eval X = a * (X^2 - 1)) :=
by
  sorry

end polynomial_determination_l880_88088


namespace half_way_fraction_l880_88055

def half_way_between (a b : ℚ) : ℚ := (a + b) / 2

theorem half_way_fraction : 
  half_way_between (1/3) (3/4) = 13/24 :=
by 
  -- Proof follows from the calculation steps, but we leave it unproved.
  sorry

end half_way_fraction_l880_88055


namespace car_gas_consumption_l880_88078

theorem car_gas_consumption
  (miles_today : ℕ)
  (miles_tomorrow : ℕ)
  (total_gallons : ℕ)
  (h1 : miles_today = 400)
  (h2 : miles_tomorrow = miles_today + 200)
  (h3 : total_gallons = 4000)
  : (∃ g : ℕ, 400 * g + (400 + 200) * g = total_gallons ∧ g = 4) :=
by
  sorry

end car_gas_consumption_l880_88078


namespace find_number_l880_88067

-- Define the condition given in the problem
def condition (x : ℤ) := 13 * x - 272 = 105

-- Prove that given the condition, x equals 29
theorem find_number : ∃ x : ℤ, condition x ∧ x = 29 :=
by
  use 29
  unfold condition
  sorry

end find_number_l880_88067


namespace div_pow_eq_l880_88046

theorem div_pow_eq (n : ℕ) (h : n = 16 ^ 2023) : n / 4 = 4 ^ 4045 :=
by
  rw [h]
  sorry

end div_pow_eq_l880_88046


namespace min_theta_l880_88093

theorem min_theta (theta : ℝ) (k : ℤ) (h : theta + 2 * k * Real.pi = -11 / 4 * Real.pi) : 
  theta = -3 / 4 * Real.pi :=
  sorry

end min_theta_l880_88093


namespace solve_equation_l880_88016

theorem solve_equation (x a b : ℝ) (h : x^2 - 6*x + 11 = 27) (sol_a : a = 8) (sol_b : b = -2) :
  3 * a - 2 * b = 28 :=
by
  sorry

end solve_equation_l880_88016


namespace part1_part2_l880_88037

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1 + Real.log x) / x
noncomputable def g (x : ℝ) : ℝ := 3 * Real.exp (1 - x) + 1

theorem part1 (a : ℝ) (x : ℝ) (h : 0 < x) : (∀ x > 0, f a x ≤ Real.exp 1) → a ≤ 1 := 
sorry

theorem part2 (a : ℝ) : (∃! x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 = g x1 ∧ f a x2 = g x2 ∧ f a x3 = g x3) → a = 3 :=
sorry

end part1_part2_l880_88037


namespace number_of_pairs_exterior_angles_l880_88045

theorem number_of_pairs_exterior_angles (m n : ℕ) :
  (3 ≤ m ∧ 3 ≤ n ∧ 360 = m * n) ↔ 20 = 20 := 
by sorry

end number_of_pairs_exterior_angles_l880_88045


namespace choose_copresidents_l880_88004

theorem choose_copresidents (total_members : ℕ) (departments : ℕ) (members_per_department : ℕ) 
    (h1 : total_members = 24) (h2 : departments = 4) (h3 : members_per_department = 6) :
    ∃ ways : ℕ, ways = 54 :=
by
  sorry

end choose_copresidents_l880_88004


namespace max_marks_test_l880_88013

theorem max_marks_test (M : ℝ) : 
  (0.30 * M = 80 + 100) -> 
  M = 600 :=
by 
  sorry

end max_marks_test_l880_88013


namespace total_students_l880_88098

-- Definitions
def is_half_reading (S : ℕ) (half_reading : ℕ) := half_reading = S / 2
def is_third_playing (S : ℕ) (third_playing : ℕ) := third_playing = S / 3
def is_total_students (S half_reading third_playing homework : ℕ) := half_reading + third_playing + homework = S

-- Homework is given to be 4
def homework : ℕ := 4

-- Total number of students
theorem total_students (S : ℕ) (half_reading third_playing : ℕ)
    (h₁ : is_half_reading S half_reading) 
    (h₂ : is_third_playing S third_playing) 
    (h₃ : is_total_students S half_reading third_playing homework) :
    S = 24 := 
sorry

end total_students_l880_88098


namespace problem1_problem2_l880_88026

def M (x : ℝ) : Prop := (x + 5) / (x - 8) ≥ 0

def N (x : ℝ) (a : ℝ) : Prop := a - 1 ≤ x ∧ x ≤ a + 1

theorem problem1 : ∀ (x : ℝ), (M x ∨ (N x 9)) ↔ (x ≤ -5 ∨ x ≥ 8) :=
by
  sorry

theorem problem2 : ∀ (a : ℝ), (∀ (x : ℝ), N x a → M x) ↔ (a ≤ -6 ∨ 9 < a) :=
by
  sorry

end problem1_problem2_l880_88026


namespace max_smoothie_servings_l880_88076

-- Define the constants based on the problem conditions
def servings_per_recipe := 4
def bananas_per_recipe := 3
def yogurt_per_recipe := 1 -- cup
def honey_per_recipe := 2 -- tablespoons
def strawberries_per_recipe := 2 -- cups

-- Define the total amount of ingredients Lynn has
def total_bananas := 12
def total_yogurt := 6 -- cups
def total_honey := 16 -- tablespoons (since 1 cup = 16 tablespoons)
def total_strawberries := 8 -- cups

-- Define the calculation for the number of servings each ingredient can produce
def servings_from_bananas := (total_bananas / bananas_per_recipe) * servings_per_recipe
def servings_from_yogurt := (total_yogurt / yogurt_per_recipe) * servings_per_recipe
def servings_from_honey := (total_honey / honey_per_recipe) * servings_per_recipe
def servings_from_strawberries := (total_strawberries / strawberries_per_recipe) * servings_per_recipe

-- Define the minimum number of servings that can be made based on all ingredients
def max_servings := min servings_from_bananas (min servings_from_yogurt (min servings_from_honey servings_from_strawberries))

theorem max_smoothie_servings : max_servings = 16 :=
by
  sorry

end max_smoothie_servings_l880_88076


namespace right_triangle_area_hypotenuse_30_deg_l880_88061

theorem right_triangle_area_hypotenuse_30_deg
  (h : Real)
  (θ : Real)
  (A : Real)
  (H1 : θ = 30)
  (H2 : h = 12)
  : A = 18 * Real.sqrt 3 := by
  sorry

end right_triangle_area_hypotenuse_30_deg_l880_88061


namespace set_subset_of_inter_union_l880_88043

variable {α : Type} [Nonempty α]
variables {A B C : Set α}

-- The main theorem based on the problem statement
theorem set_subset_of_inter_union (h : A ∩ B = B ∪ C) : C ⊆ B :=
by
  sorry

end set_subset_of_inter_union_l880_88043


namespace hyperbola_focus_l880_88064

-- Definition of the hyperbola equation and foci
def is_hyperbola (x y : ℝ) (k : ℝ) : Prop :=
  x^2 - k * y^2 = 1

-- Definition of the hyperbola having a focus at (3, 0) and the value of k
def has_focus_at (k : ℝ) : Prop :=
  ∃ x y : ℝ, is_hyperbola x y k ∧ (x, y) = (3, 0)

theorem hyperbola_focus (k : ℝ) (h : has_focus_at k) : k = 1 / 8 :=
  sorry

end hyperbola_focus_l880_88064


namespace anna_clara_age_l880_88099

theorem anna_clara_age :
  ∃ x : ℕ, (54 - x) * 3 = 80 - x ∧ x = 41 :=
by
  sorry

end anna_clara_age_l880_88099


namespace grocer_second_month_sale_l880_88070

theorem grocer_second_month_sale (sale_1 sale_3 sale_4 sale_5 sale_6 avg_sale n : ℕ) 
(h1 : sale_1 = 6435) 
(h3 : sale_3 = 6855) 
(h4 : sale_4 = 7230) 
(h5 : sale_5 = 6562) 
(h6 : sale_6 = 7391) 
(havg : avg_sale = 6900) 
(hn : n = 6) : 
  sale_2 = 6927 :=
by
  sorry

end grocer_second_month_sale_l880_88070


namespace complex_fraction_simplification_l880_88068

theorem complex_fraction_simplification : 
  ∀ (i : ℂ), i^2 = -1 → (1 + i^2017) / (1 - i) = i :=
by
  intro i h_imag_unit
  sorry

end complex_fraction_simplification_l880_88068


namespace count_multiples_of_70_in_range_200_to_500_l880_88000

theorem count_multiples_of_70_in_range_200_to_500 : 
  ∃! count, count = 5 ∧ (∀ n, 200 ≤ n ∧ n ≤ 500 ∧ (n % 70 = 0) ↔ n = 210 ∨ n = 280 ∨ n = 350 ∨ n = 420 ∨ n = 490) :=
by
  sorry

end count_multiples_of_70_in_range_200_to_500_l880_88000


namespace problem_trapezoid_l880_88028

noncomputable def ratio_of_areas (AB CD : ℝ) (h : ℝ) (ratio : ℝ) :=
  let area_trapezoid := (AB + CD) * h / 2
  let area_triangle_AZW := (4 * h) / 15
  ratio = area_triangle_AZW / area_trapezoid

theorem problem_trapezoid :
  ratio_of_areas 2 5 h (8 / 105) :=
by
  sorry

end problem_trapezoid_l880_88028


namespace total_bags_sold_l880_88087

theorem total_bags_sold (first_week second_week third_week fourth_week total : ℕ) 
  (h1 : first_week = 15) 
  (h2 : second_week = 3 * first_week) 
  (h3 : third_week = 20) 
  (h4 : fourth_week = 20) 
  (h5 : total = first_week + second_week + third_week + fourth_week) : 
  total = 100 := 
sorry

end total_bags_sold_l880_88087


namespace total_space_after_compaction_correct_l880_88015

noncomputable def problem : Prop :=
  let num_small_cans := 50
  let num_large_cans := 50
  let small_can_size := 20
  let large_can_size := 40
  let small_can_compaction := 0.30
  let large_can_compaction := 0.40
  let small_cans_compacted := num_small_cans * small_can_size * small_can_compaction
  let large_cans_compacted := num_large_cans * large_can_size * large_can_compaction
  let total_space_after_compaction := small_cans_compacted + large_cans_compacted
  total_space_after_compaction = 1100

theorem total_space_after_compaction_correct :
  problem :=
  by
    unfold problem
    sorry

end total_space_after_compaction_correct_l880_88015


namespace combined_resistance_parallel_l880_88054

theorem combined_resistance_parallel (R1 R2 R3 R : ℝ)
  (h1 : R1 = 2) (h2 : R2 = 5) (h3 : R3 = 6)
  (h4 : 1/R = 1/R1 + 1/R2 + 1/R3) :
  R = 15/13 := 
by
  sorry

end combined_resistance_parallel_l880_88054


namespace emily_irises_after_addition_l880_88008

theorem emily_irises_after_addition
  (initial_roses : ℕ)
  (added_roses : ℕ)
  (ratio_irises_roses : ℕ)
  (ratio_roses_irises : ℕ)
  (h_ratio : ratio_irises_roses = 3 ∧ ratio_roses_irises = 7)
  (h_initial_roses : initial_roses = 35)
  (h_added_roses : added_roses = 30) :
  ∃ irises_after_addition : ℕ, irises_after_addition = 27 :=
  by
    sorry

end emily_irises_after_addition_l880_88008


namespace probability_sqrt_two_digit_less_than_seven_l880_88091

noncomputable def prob_sqrt_less_than_seven : ℚ := 
  let favorable := 39
  let total := 90
  favorable / total

theorem probability_sqrt_two_digit_less_than_seven : 
  prob_sqrt_less_than_seven = 13 / 30 := by
  sorry

end probability_sqrt_two_digit_less_than_seven_l880_88091


namespace gaeun_taller_than_nana_l880_88005

def nana_height_m : ℝ := 1.618
def gaeun_height_cm : ℝ := 162.3
def nana_height_cm : ℝ := nana_height_m * 100

theorem gaeun_taller_than_nana : gaeun_height_cm - nana_height_cm = 0.5 := by
  sorry

end gaeun_taller_than_nana_l880_88005


namespace total_fish_weight_is_25_l880_88022

-- Define the conditions and the problem
def num_trout : ℕ := 4
def weight_trout : ℝ := 2
def num_catfish : ℕ := 3
def weight_catfish : ℝ := 1.5
def num_bluegills : ℕ := 5
def weight_bluegill : ℝ := 2.5

-- Calculate the total weight of each type of fish
def total_weight_trout : ℝ := num_trout * weight_trout
def total_weight_catfish : ℝ := num_catfish * weight_catfish
def total_weight_bluegills : ℝ := num_bluegills * weight_bluegill

-- Calculate the total weight of all fish
def total_weight_fish : ℝ := total_weight_trout + total_weight_catfish + total_weight_bluegills

-- Statement to be proved
theorem total_fish_weight_is_25 : total_weight_fish = 25 := by
  sorry

end total_fish_weight_is_25_l880_88022


namespace annika_total_distance_l880_88002

/--
Annika hikes at a constant rate of 12 minutes per kilometer. She has hiked 2.75 kilometers
east from the start of a hiking trail when she realizes that she has to be back at the start
of the trail in 51 minutes. Prove that the total distance Annika hiked east is 3.5 kilometers.
-/
theorem annika_total_distance :
  (hike_rate : ℝ) = 12 → 
  (initial_distance_east : ℝ) = 2.75 → 
  (total_time : ℝ) = 51 → 
  (total_distance_east : ℝ) = 3.5 :=
by 
  intro hike_rate initial_distance_east total_time 
  sorry

end annika_total_distance_l880_88002


namespace pizza_toppings_problem_l880_88090

theorem pizza_toppings_problem
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (olive_slices : ℕ)
  (pepperoni_mushroom_slices : ℕ)
  (pepperoni_olive_slices : ℕ)
  (mushroom_olive_slices : ℕ)
  (pepperoni_mushroom_olive_slices : ℕ) :
  total_slices = 20 →
  pepperoni_slices = 12 →
  mushroom_slices = 14 →
  olive_slices = 12 →
  pepperoni_mushroom_slices = 8 →
  pepperoni_olive_slices = 8 →
  mushroom_olive_slices = 8 →
  total_slices = pepperoni_slices + mushroom_slices + olive_slices
    - pepperoni_mushroom_slices - pepperoni_olive_slices - mushroom_olive_slices
    + pepperoni_mushroom_olive_slices →
  pepperoni_mushroom_olive_slices = 6 :=
by
  intros
  sorry

end pizza_toppings_problem_l880_88090


namespace halfway_between_one_third_and_one_fifth_l880_88086

theorem halfway_between_one_third_and_one_fifth : (1/3 + 1/5) / 2 = 4/15 := 
by 
  sorry

end halfway_between_one_third_and_one_fifth_l880_88086


namespace log_problem_l880_88032

open Real

theorem log_problem : 2 * log 5 + log 4 = 2 := by
  sorry

end log_problem_l880_88032


namespace class_A_students_l880_88031

variable (A B : ℕ)

theorem class_A_students 
    (h1 : A = (5 * B) / 7)
    (h2 : A + 3 = (4 * (B - 3)) / 5) :
    A = 45 :=
sorry

end class_A_students_l880_88031


namespace range_of_b_l880_88084

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 3^x + b

theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, f x b ≥ 0) ↔ b ≤ -1 :=
by sorry

end range_of_b_l880_88084


namespace mean_value_z_l880_88010

theorem mean_value_z (z : ℚ) (h : (7 + 10 + 23) / 3 = (18 + z) / 2) : z = 26 / 3 :=
by
  sorry

end mean_value_z_l880_88010


namespace nine_b_value_l880_88097

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : a = b - 3) : 
  9 * b = 216 / 11 :=
by
  sorry

end nine_b_value_l880_88097


namespace smallest_possible_gcd_l880_88041

theorem smallest_possible_gcd (m n p : ℕ) (h1 : Nat.gcd m n = 180) (h2 : Nat.gcd m p = 240) :
  ∃ k, k = Nat.gcd n p ∧ k = 60 := by
  sorry

end smallest_possible_gcd_l880_88041


namespace range_of_7a_minus_5b_l880_88075

theorem range_of_7a_minus_5b (a b : ℝ) (h1 : 5 ≤ a - b ∧ a - b ≤ 27) (h2 : 6 ≤ a + b ∧ a + b ≤ 30) : 
  36 ≤ 7 * a - 5 * b ∧ 7 * a - 5 * b ≤ 192 :=
sorry

end range_of_7a_minus_5b_l880_88075


namespace last_three_digits_of_2_pow_10000_l880_88080

theorem last_three_digits_of_2_pow_10000 (h : 2^500 ≡ 1 [MOD 1250]) : (2^10000) % 1000 = 1 :=
by
  sorry

end last_three_digits_of_2_pow_10000_l880_88080


namespace cost_of_pen_l880_88040

theorem cost_of_pen 
  (total_amount_spent : ℕ)
  (total_items : ℕ)
  (number_of_pencils : ℕ)
  (cost_of_pencil : ℕ)
  (cost_of_pen : ℕ)
  (h1 : total_amount_spent = 2000)
  (h2 : total_items = 36)
  (h3 : number_of_pencils = 16)
  (h4 : cost_of_pencil = 25)
  (remaining_amount_spent : ℕ)
  (number_of_pens : ℕ)
  (h5 : remaining_amount_spent = total_amount_spent - (number_of_pencils * cost_of_pencil))
  (h6 : number_of_pens = total_items - number_of_pencils)
  (total_cost_of_pens : ℕ)
  (h7 : total_cost_of_pens = remaining_amount_spent)
  (h8 : total_cost_of_pens = number_of_pens * cost_of_pen)
  : cost_of_pen = 80 := by
  sorry

end cost_of_pen_l880_88040


namespace find_xyz_l880_88089

theorem find_xyz (x y z : ℝ)
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 27)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11)
  (h₃ : x + y + z = 3) :
  x * y * z = 16 / 3 := 
  sorry

end find_xyz_l880_88089


namespace marked_price_l880_88006

theorem marked_price (initial_price : ℝ) (discount_percent : ℝ) (profit_margin_percent : ℝ) (final_discount_percent : ℝ) (marked_price : ℝ) :
  initial_price = 40 → 
  discount_percent = 0.25 → 
  profit_margin_percent = 0.50 → 
  final_discount_percent = 0.10 → 
  marked_price = 50 := by
  sorry

end marked_price_l880_88006


namespace no_common_root_l880_88079

theorem no_common_root (a b c d : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) :
  ¬∃ x : ℝ, (x^2 + b * x + c = 0) ∧ (x^2 + a * x + d = 0) := 
sorry

end no_common_root_l880_88079


namespace storks_minus_birds_l880_88092

/-- Define the initial values --/
def s : ℕ := 6         -- Number of storks
def b1 : ℕ := 2        -- Initial number of birds
def b2 : ℕ := 3        -- Number of additional birds

/-- Calculate the total number of birds --/
def b : ℕ := b1 + b2   -- Total number of birds

/-- Prove the number of storks minus the number of birds --/
theorem storks_minus_birds : s - b = 1 :=
by sorry

end storks_minus_birds_l880_88092


namespace trigonometric_fraction_value_l880_88069

theorem trigonometric_fraction_value (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) /
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := by
  sorry

end trigonometric_fraction_value_l880_88069


namespace quadruples_positive_integers_l880_88083

theorem quadruples_positive_integers (x y z n : ℕ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ n > 0 ∧ (x^2 + y^2 + z^2 + 1 = 2^n) →
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
sorry

end quadruples_positive_integers_l880_88083


namespace find_whole_number_N_l880_88012

theorem find_whole_number_N (N : ℕ) (h1 : 6.75 < (N / 4 : ℝ)) (h2 : (N / 4 : ℝ) < 7.25) : N = 28 := 
by 
  sorry

end find_whole_number_N_l880_88012


namespace school_accomodation_proof_l880_88029

theorem school_accomodation_proof
  (total_classrooms : ℕ) 
  (fraction_classrooms_45 : ℕ) 
  (fraction_classrooms_38 : ℕ)
  (fraction_classrooms_32 : ℕ)
  (fraction_classrooms_25 : ℕ)
  (desks_45 : ℕ)
  (desks_38 : ℕ)
  (desks_32 : ℕ)
  (desks_25 : ℕ)
  (student_capacity_limit : ℕ) :
  total_classrooms = 50 ->
  fraction_classrooms_45 = (3 / 10) * total_classrooms -> 
  fraction_classrooms_38 = (1 / 4) * total_classrooms -> 
  fraction_classrooms_32 = (1 / 5) * total_classrooms -> 
  fraction_classrooms_25 = (total_classrooms - fraction_classrooms_45 - fraction_classrooms_38 - fraction_classrooms_32) ->
  desks_45 = 15 * 45 -> 
  desks_38 = 12 * 38 -> 
  desks_32 = 10 * 32 -> 
  desks_25 = fraction_classrooms_25 * 25 -> 
  student_capacity_limit = 1800 -> 
  fraction_classrooms_45 * 45 +
  fraction_classrooms_38 * 38 +
  fraction_classrooms_32 * 32 + 
  fraction_classrooms_25 * 25 = 1776 + sorry
  :=
sorry

end school_accomodation_proof_l880_88029


namespace find_speed_of_stream_l880_88096

variable (b s : ℝ)

-- Equation derived from downstream condition
def downstream_equation := b + s = 24

-- Equation derived from upstream condition
def upstream_equation := b - s = 10

theorem find_speed_of_stream
  (b s : ℝ)
  (h1 : downstream_equation b s)
  (h2 : upstream_equation b s) :
  s = 7 := by
  -- placeholder for the proof
  sorry

end find_speed_of_stream_l880_88096


namespace trivia_team_points_l880_88033

theorem trivia_team_points (total_members: ℕ) (total_points: ℕ) (points_per_member: ℕ) (members_showed_up: ℕ) (members_did_not_show_up: ℕ):
  total_members = 7 → 
  total_points = 20 → 
  points_per_member = 4 → 
  members_showed_up = total_points / points_per_member → 
  members_did_not_show_up = total_members - members_showed_up → 
  members_did_not_show_up = 2 := 
by 
  intros h1 h2 h3 h4 h5
  sorry

end trivia_team_points_l880_88033


namespace sum_of_coeffs_binomial_eq_32_l880_88039

noncomputable def sum_of_coeffs_binomial (x : ℝ) : ℝ :=
  (3 * x - 1 / Real.sqrt x)^5

theorem sum_of_coeffs_binomial_eq_32 :
  sum_of_coeffs_binomial 1 = 32 :=
by
  sorry

end sum_of_coeffs_binomial_eq_32_l880_88039


namespace range_of_m_l880_88030

theorem range_of_m (m x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (h3 : (1 - 2 * m) / x1 < (1 - 2 * m) / x2) : m < 1 / 2 :=
sorry

end range_of_m_l880_88030


namespace find_pairs_l880_88059

theorem find_pairs (p a : ℕ) (hp_prime : Nat.Prime p) (hp_ge_2 : p ≥ 2) (ha_ge_1 : a ≥ 1) (h_p_ne_a : p ≠ a) :
  (a + p) ∣ (a^2 + p^2) → (a = p ∧ p = p) ∨ (a = p^2 - p ∧ p = p) ∨ (a = 2 * p^2 - p ∧ p = p) :=
by
  sorry

end find_pairs_l880_88059


namespace vacation_trip_l880_88018

theorem vacation_trip (airbnb_cost : ℕ) (car_rental_cost : ℕ) (share_per_person : ℕ) (total_people : ℕ) :
  airbnb_cost = 3200 → car_rental_cost = 800 → share_per_person = 500 → airbnb_cost + car_rental_cost / share_per_person = 8 :=
by
  intros h1 h2 h3
  sorry

end vacation_trip_l880_88018


namespace three_pow_1234_mod_5_l880_88081

theorem three_pow_1234_mod_5 : (3^1234) % 5 = 4 := 
by 
  have h1 : 3^4 % 5 = 1 := by norm_num
  sorry

end three_pow_1234_mod_5_l880_88081


namespace second_supplier_more_cars_l880_88065

-- Define the constants and conditions given in the problem
def total_production := 5650000
def first_supplier := 1000000
def fourth_fifth_supplier := 325000

-- Define the unknown variable for the second supplier
noncomputable def second_supplier : ℕ := sorry

-- Define the equation based on the conditions
def equation := first_supplier + second_supplier + (first_supplier + second_supplier) + (4 * fourth_fifth_supplier / 2) = total_production

-- Prove that the second supplier receives 500,000 more cars than the first supplier
theorem second_supplier_more_cars : 
  ∃ X : ℕ, equation → (X = first_supplier + 500000) :=
sorry

end second_supplier_more_cars_l880_88065


namespace average_salary_increase_l880_88051

theorem average_salary_increase :
  let avg_salary := 1200
  let num_employees := 20
  let manager_salary := 3300
  let new_num_people := num_employees + 1
  let total_salary := num_employees * avg_salary
  let new_total_salary := total_salary + manager_salary
  let new_avg_salary := new_total_salary / new_num_people
  let increase := new_avg_salary - avg_salary
  increase = 100 :=
by
  sorry

end average_salary_increase_l880_88051


namespace transform_eq_l880_88007

theorem transform_eq (x y : ℝ) (h : 5 * x - 6 * y = 4) : 
  y = (5 / 6) * x - (2 / 3) :=
  sorry

end transform_eq_l880_88007


namespace solution_volume_l880_88058

theorem solution_volume (concentration volume_acid volume_solution : ℝ) 
  (h_concentration : concentration = 0.25) 
  (h_acid : volume_acid = 2.5) 
  (h_formula : concentration = volume_acid / volume_solution) : 
  volume_solution = 10 := 
by
  sorry

end solution_volume_l880_88058


namespace sufficiency_condition_a_gt_b_sq_gt_sq_l880_88049

theorem sufficiency_condition_a_gt_b_sq_gt_sq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a > b → a^2 > b^2) ∧ (∀ (h : a^2 > b^2), ∃ c > 0, ∃ d > 0, c^2 > d^2 ∧ ¬(c > d)) :=
by
  sorry

end sufficiency_condition_a_gt_b_sq_gt_sq_l880_88049
