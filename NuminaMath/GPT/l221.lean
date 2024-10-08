import Mathlib

namespace closest_ratio_to_one_l221_221819

theorem closest_ratio_to_one (a c : ℕ) (h1 : 2 * a + c = 130) (h2 : a ≥ 1) (h3 : c ≥ 1) : 
  a = 43 ∧ c = 44 :=
by {
    sorry 
}

end closest_ratio_to_one_l221_221819


namespace homework_checked_on_friday_l221_221835

theorem homework_checked_on_friday
  (prob_no_check : ℚ := 1/2)
  (prob_check_on_friday_given_check : ℚ := 1/5)
  (prob_a : ℚ := 3/5)
  : 1/3 = prob_check_on_friday_given_check / prob_a :=
by
  sorry

end homework_checked_on_friday_l221_221835


namespace instrument_price_problem_l221_221914

theorem instrument_price_problem (v t p : ℝ) (h1 : 1.5 * v = 0.5 * t + 50) (h2 : 1.5 * t = 0.5 * p + 50) : 
  ∃ m n : ℤ, m = 80 ∧ n = 80 ∧ (100 + m) * v / 100 = n + (100 - m) * p / 100 := 
by
  use 80, 80
  sorry

end instrument_price_problem_l221_221914


namespace hcf_of_two_numbers_of_given_conditions_l221_221870

theorem hcf_of_two_numbers_of_given_conditions :
  ∃ B H, (588 = H * 84) ∧ H = Nat.gcd 588 B ∧ H = 7 :=
by
  use 84, 7
  have h₁ : 588 = 7 * 84 := by sorry
  have h₂ : 7 = Nat.gcd 588 84 := by sorry
  exact ⟨h₁, h₂, rfl⟩

end hcf_of_two_numbers_of_given_conditions_l221_221870


namespace pears_for_apples_l221_221872

-- Define the costs of apples, oranges, and pears.
variables {cost_apples cost_oranges cost_pears : ℕ}

-- Condition 1: Ten apples cost the same as five oranges
axiom apples_equiv_oranges : 10 * cost_apples = 5 * cost_oranges

-- Condition 2: Three oranges cost the same as four pears
axiom oranges_equiv_pears : 3 * cost_oranges = 4 * cost_pears

-- Theorem: Tyler can buy 13 pears for the price of 20 apples
theorem pears_for_apples : 20 * cost_apples = 13 * cost_pears :=
sorry

end pears_for_apples_l221_221872


namespace sum_of_squares_not_perfect_square_l221_221220

theorem sum_of_squares_not_perfect_square (n : ℤ) : ¬ (∃ k : ℤ, k^2 = (n-2)^2 + (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2) :=
by
  sorry

end sum_of_squares_not_perfect_square_l221_221220


namespace larger_number_is_correct_l221_221817

theorem larger_number_is_correct : ∃ L : ℝ, ∃ S : ℝ, S = 48 ∧ (L - S = (1 : ℝ) / (3 : ℝ) * L) ∧ L = 72 :=
by
  sorry

end larger_number_is_correct_l221_221817


namespace probability_htth_l221_221487

def probability_of_sequence_HTTH := (1 / 2) * (1 / 2) * (1 / 2) * (1 / 2)

theorem probability_htth : probability_of_sequence_HTTH = 1 / 16 := by
  sorry

end probability_htth_l221_221487


namespace min_people_wearing_both_l221_221792

theorem min_people_wearing_both (n : ℕ) (h1 : n % 3 = 0)
  (h_gloves : ∃ g, g = n / 3 ∧ g = 1) (h_hats : ∃ h, h = (2 * n) / 3 ∧ h = 2) :
  ∃ x, x = 0 := by
  sorry

end min_people_wearing_both_l221_221792


namespace find_m_l221_221590

variable (m x1 x2 : ℝ)

def quadratic_eqn (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m * x + 2 * m - 1 = 0

def roots_condition (m x1 x2 : ℝ) : Prop :=
  x1^2 + x2^2 = 23 ∧
  x1 + x2 = m ∧
  x1 * x2 = 2 * m - 1

theorem find_m (m x1 x2 : ℝ) : 
  quadratic_eqn m → 
  roots_condition m x1 x2 → 
  m = -3 :=
by
  intro hQ hR
  sorry

end find_m_l221_221590


namespace probability_of_winning_l221_221120

def roll_is_seven (d1 d2 : ℕ) : Prop :=
  d1 + d2 = 7

theorem probability_of_winning (d1 d2 : ℕ) (h : roll_is_seven d1 d2) :
  (1/6 : ℚ) = 1/6 :=
by
  sorry

end probability_of_winning_l221_221120


namespace factorial_division_l221_221130

-- Define factorial using the standard library's factorial function
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- The problem statement
theorem factorial_division :
  (factorial 15) / (factorial 6 * factorial 9) = 834 :=
sorry

end factorial_division_l221_221130


namespace total_spent_on_pens_l221_221182

/-- Dorothy, Julia, and Robert go to the store to buy school supplies.
    Dorothy buys half as many pens as Julia.
    Julia buys three times as many pens as Robert.
    Robert buys 4 pens.
    The cost of one pen is $1.50.
    Prove that the total amount of money spent on pens by the three friends is $33. 
-/
theorem total_spent_on_pens :
  let cost_per_pen := 1.50
  let robert_pens := 4
  let julia_pens := 3 * robert_pens
  let dorothy_pens := julia_pens / 2
  let total_pens := robert_pens + julia_pens + dorothy_pens
  total_pens * cost_per_pen = 33 := 
by
  let cost_per_pen := 1.50
  let robert_pens := 4
  let julia_pens := 3 * robert_pens
  let dorothy_pens := julia_pens / 2
  let total_pens := robert_pens + julia_pens + dorothy_pens
  sorry

end total_spent_on_pens_l221_221182


namespace count_integers_in_interval_l221_221034

theorem count_integers_in_interval : 
  ∃ (k : ℤ), k = 46 ∧ 
  (∀ n : ℤ, -5 * (2.718 : ℝ) ≤ (n : ℝ) ∧ (n : ℝ) ≤ 12 * (2.718 : ℝ) → (-13 ≤ n ∧ n ≤ 32)) ∧ 
  (∀ n : ℤ, -13 ≤ n ∧ n ≤ 32 → -5 * (2.718 : ℝ) ≤ (n : ℝ) ∧ (n : ℝ) ≤ 12 * (2.718 : ℝ)) :=
sorry

end count_integers_in_interval_l221_221034


namespace remainder_mod_7_l221_221877

theorem remainder_mod_7 (n m p : ℕ) 
  (h₁ : n % 4 = 3)
  (h₂ : m % 7 = 5)
  (h₃ : p % 5 = 2) :
  (7 * n + 3 * m - p) % 7 = 6 :=
by
  sorry

end remainder_mod_7_l221_221877


namespace present_number_of_teachers_l221_221809

theorem present_number_of_teachers (S T : ℕ) (h1 : S = 50 * T) (h2 : S + 50 = 25 * (T + 5)) : T = 3 := 
by 
  sorry

end present_number_of_teachers_l221_221809


namespace sufficient_but_not_necessary_l221_221501

noncomputable def problem_statement (a : ℝ) : Prop :=
(a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2)

theorem sufficient_but_not_necessary (a : ℝ) : problem_statement a := 
sorry

end sufficient_but_not_necessary_l221_221501


namespace solve_for_M_plus_N_l221_221371

theorem solve_for_M_plus_N (M N : ℕ) (h1 : 4 * N = 588) (h2 : 4 * 63 = 7 * M) : M + N = 183 := by
  sorry

end solve_for_M_plus_N_l221_221371


namespace total_wicks_20_l221_221439

theorem total_wicks_20 (string_length_ft : ℕ) (length_wick_1 length_wick_2 : ℕ) (wicks_1 wicks_2 : ℕ) :
  string_length_ft = 15 →
  length_wick_1 = 6 →
  length_wick_2 = 12 →
  wicks_1 = wicks_2 →
  (string_length_ft * 12) = (length_wick_1 * wicks_1 + length_wick_2 * wicks_2) →
  (wicks_1 + wicks_2) = 20 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end total_wicks_20_l221_221439


namespace only_set_d_forms_triangle_l221_221968

/-- Definition of forming a triangle given three lengths -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem only_set_d_forms_triangle :
  ¬ can_form_triangle 3 5 10 ∧ ¬ can_form_triangle 5 4 9 ∧ 
  ¬ can_form_triangle 5 5 10 ∧ can_form_triangle 4 6 9 :=
by {
  sorry
}

end only_set_d_forms_triangle_l221_221968


namespace ratio_of_colored_sheets_l221_221552

theorem ratio_of_colored_sheets
    (total_sheets : ℕ)
    (num_binders : ℕ)
    (sheets_colored_by_justine : ℕ)
    (sheets_per_binder : ℕ)
    (h1 : total_sheets = 2450)
    (h2 : num_binders = 5)
    (h3 : sheets_colored_by_justine = 245)
    (h4 : sheets_per_binder = total_sheets / num_binders) :
    (sheets_colored_by_justine / Nat.gcd sheets_colored_by_justine sheets_per_binder) /
    (sheets_per_binder / Nat.gcd sheets_colored_by_justine sheets_per_binder) = 1 / 2 := by
  sorry

end ratio_of_colored_sheets_l221_221552


namespace portion_of_pizza_eaten_l221_221112

-- Define the conditions
def total_slices : ℕ := 16
def slices_left : ℕ := 4
def slices_eaten : ℕ := total_slices - slices_left

-- Define the portion of pizza eaten
def portion_eaten := (slices_eaten : ℚ) / (total_slices : ℚ)

-- Statement to prove
theorem portion_of_pizza_eaten : portion_eaten = 3 / 4 :=
by sorry

end portion_of_pizza_eaten_l221_221112


namespace earthquake_energy_multiple_l221_221032

theorem earthquake_energy_multiple (E : ℕ → ℝ) (n9 n7 : ℕ)
  (h1 : E n9 = 10 ^ n9) 
  (h2 : E n7 = 10 ^ n7) 
  (hn9 : n9 = 9) 
  (hn7 : n7 = 7) : 
  E n9 / E n7 = 100 := 
by 
  sorry

end earthquake_energy_multiple_l221_221032


namespace difference_of_coordinates_l221_221059

-- Define point and its properties in Lean.
structure Point where
  x : ℝ
  y : ℝ

-- Define the midpoint property.
def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

-- Given points A and M
def A : Point := {x := 8, y := 0}
def M : Point := {x := 4, y := 1}

-- Assume B is a point with coordinates x and y
variable (B : Point)

-- The theorem to prove.
theorem difference_of_coordinates :
  is_midpoint M A B → B.x - B.y = -2 :=
by
  sorry

end difference_of_coordinates_l221_221059


namespace base12_addition_l221_221698

theorem base12_addition : ∀ a b : ℕ, a = 956 ∧ b = 273 → (a + b) = 1009 := by
  sorry

end base12_addition_l221_221698


namespace find_constant_C_l221_221542

def polynomial_remainder (C : ℝ) (x : ℝ) : ℝ :=
  C * x^3 - 3 * x^2 + x - 1

theorem find_constant_C :
  (polynomial_remainder 2 (-1) = -7) → 2 = 2 :=
by
  sorry

end find_constant_C_l221_221542


namespace PQR_product_l221_221249

def PQR_condition (P Q R S : ℕ) : Prop :=
  P + Q + R + S = 100 ∧
  ∃ x : ℕ, P = x - 4 ∧ Q = x + 4 ∧ R = x / 4 ∧ S = 4 * x

theorem PQR_product (P Q R S : ℕ) (h : PQR_condition P Q R S) : P * Q * R * S = 61440 :=
by 
  sorry

end PQR_product_l221_221249


namespace mouse_jump_distance_l221_221891

theorem mouse_jump_distance
  (g : ℕ) 
  (f : ℕ) 
  (m : ℕ)
  (h1 : g = 25)
  (h2 : f = g + 32)
  (h3 : m = f - 26) : 
  m = 31 :=
by
  sorry

end mouse_jump_distance_l221_221891


namespace one_fourth_of_56_equals_75_l221_221206

theorem one_fourth_of_56_equals_75 : (5.6 / 4) = 7 / 5 := 
by
  -- Temporarily omitting the actual proof
  sorry

end one_fourth_of_56_equals_75_l221_221206


namespace original_cube_volume_l221_221113

theorem original_cube_volume 
  (a : ℕ) 
  (h : 3 * a * (a - a / 2) * a - a^3 = 2 * a^2) : 
  a = 4 → a^3 = 64 := 
by
  sorry

end original_cube_volume_l221_221113


namespace clock_angle_9_30_l221_221404

theorem clock_angle_9_30 : 
  let hour_hand_pos := 9.5 
  let minute_hand_pos := 6 
  let degrees_per_division := 30 
  let divisions_apart := hour_hand_pos - minute_hand_pos
  let angle := divisions_apart * degrees_per_division
  angle = 105 :=
by
  sorry

end clock_angle_9_30_l221_221404


namespace simplify_expression_l221_221168

theorem simplify_expression : 1 - (1 / (2 + Real.sqrt 5)) + (1 / (2 - Real.sqrt 5)) = 1 - 2 * Real.sqrt 5 :=
by
  sorry

end simplify_expression_l221_221168


namespace inverse_proportional_k_value_l221_221558

theorem inverse_proportional_k_value (k : ℝ) :
  (∃ x y : ℝ, y = k / x ∧ x = - (Real.sqrt 2) / 2 ∧ y = Real.sqrt 2) → 
  k = -1 :=
by
  sorry

end inverse_proportional_k_value_l221_221558


namespace sufficient_but_not_necessary_l221_221917

theorem sufficient_but_not_necessary (a b c : ℝ) :
  (b^2 = a * c → (c ≠ 0 ∧ a ≠ 0 ∧ b * b = a * c) ∨ (b = 0)) ∧ 
  ¬ ((c ≠ 0 ∧ a ≠ 0 ∧ b * b = a * c) → b^2 = a * c) :=
by
  sorry

end sufficient_but_not_necessary_l221_221917


namespace crayons_total_l221_221190

theorem crayons_total (rows : ℕ) (crayons_per_row : ℕ) (total_crayons : ℕ) :
  rows = 15 → crayons_per_row = 42 → total_crayons = rows * crayons_per_row → total_crayons = 630 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end crayons_total_l221_221190


namespace domain_of_function_l221_221379

theorem domain_of_function :
  ∀ x : ℝ, 3 * x - 2 > 0 ∧ 2 * x - 1 > 0 ↔ x > (2 / 3) := by
  intro x
  sorry

end domain_of_function_l221_221379


namespace find_ending_number_divisible_by_eleven_l221_221808

theorem find_ending_number_divisible_by_eleven (start n end_num : ℕ) (h1 : start = 29) (h2 : n = 5) (h3 : ∀ k : ℕ, ∃ m : ℕ, m = start + k * 11) : end_num = 77 :=
sorry

end find_ending_number_divisible_by_eleven_l221_221808


namespace geometric_sequence_a_sequence_b_l221_221256

theorem geometric_sequence_a (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : 2 * a 2 + a 3 = 60) :
  ∀ n, a n = 4 * 3^(n - 1) :=
sorry

theorem sequence_b (b a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : 2 * a 2 + a 3 = 60)
  (h3 : ∀ n, b (n + 1) = b n + a n) (h4 : b 1 = a 2) :
  ∀ n, b n = 2 * 3^n + 10 :=
sorry

end geometric_sequence_a_sequence_b_l221_221256


namespace distance_between_trees_l221_221062

theorem distance_between_trees 
  (rows columns : ℕ)
  (boundary_distance garden_length d : ℝ)
  (h_rows : rows = 10)
  (h_columns : columns = 12)
  (h_boundary_distance : boundary_distance = 5)
  (h_garden_length : garden_length = 32) :
  (9 * d + 2 * boundary_distance = garden_length) → 
  d = 22 / 9 := 
by 
  intros h_eq
  sorry

end distance_between_trees_l221_221062


namespace distinct_pos_numbers_implies_not_zero_at_least_one_of_abc_impossible_for_all_neq_l221_221676

noncomputable section

variables (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) (h1 : 0 < a) 
(h2 : 0 < b) (h3 : 0 < c)

theorem distinct_pos_numbers_implies_not_zero :
  (a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2 ≠ 0 :=
sorry

theorem at_least_one_of_abc :
  a > b ∨ a < b ∨ a = b :=
sorry

theorem impossible_for_all_neq :
  ¬(a ≠ c ∧ b ≠ c ∧ a ≠ b) :=
sorry

end distinct_pos_numbers_implies_not_zero_at_least_one_of_abc_impossible_for_all_neq_l221_221676


namespace min_value_x_plus_y_l221_221149

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 / y + 1 / x = 4) :
  x + y ≥ 9 / 4 :=
sorry

end min_value_x_plus_y_l221_221149


namespace negation_of_exists_l221_221255

theorem negation_of_exists (h : ∃ x : ℝ, x > 0 ∧ x^2 + 3*x + 1 < 0) : ∀ x : ℝ, x > 0 → x^2 + 3*x + 1 ≥ 0 :=
sorry

end negation_of_exists_l221_221255


namespace hiker_total_distance_l221_221214

-- Define conditions based on the problem description
def day1_distance : ℕ := 18
def day1_speed : ℕ := 3
def day2_speed : ℕ := day1_speed + 1
def day1_time : ℕ := day1_distance / day1_speed
def day2_time : ℕ := day1_time - 1
def day3_speed : ℕ := 5
def day3_time : ℕ := 3

-- Define the total distance walked based on the conditions
def total_distance : ℕ :=
  day1_distance + (day2_speed * day2_time) + (day3_speed * day3_time)

-- The theorem stating the hiker walked a total of 53 miles
theorem hiker_total_distance : total_distance = 53 := by
  sorry

end hiker_total_distance_l221_221214


namespace passenger_capacity_passenger_capacity_at_5_max_profit_l221_221730

section SubwayProject

-- Define the time interval t and the passenger capacity function p(t)
def p (t : ℕ) : ℕ :=
  if 2 ≤ t ∧ t < 10 then 300 + 40 * t - 2 * t^2
  else if 10 ≤ t ∧ t ≤ 20 then 500
  else 0

-- Define the net profit function Q(t)
def Q (t : ℕ) : ℚ :=
  if 2 ≤ t ∧ t < 10 then (8 * p t - 2656) / t - 60
  else if 10 ≤ t ∧ t ≤ 20 then (1344 : ℚ) / t - 60
  else 0

-- Statement 1: Prove the correct expression for p(t) and its value at t = 5
theorem passenger_capacity (t : ℕ) (ht1 : 2 ≤ t) (ht2 : t ≤ 20) :
  (p t = if 2 ≤ t ∧ t < 10 then 300 + 40 * t - 2 * t^2 else 500) :=
sorry

theorem passenger_capacity_at_5 : p 5 = 450 :=
sorry

-- Statement 2: Prove the time interval t and the maximum value of Q(t)
theorem max_profit : ∃ t : ℕ, 2 ≤ t ∧ t ≤ 10 ∧ Q t = 132 ∧ (∀ u : ℕ, 2 ≤ u ∧ u ≤ 10 → Q u ≤ Q t) :=
sorry

end SubwayProject

end passenger_capacity_passenger_capacity_at_5_max_profit_l221_221730


namespace p_necessary_but_not_sufficient_for_q_l221_221356

noncomputable def p (x : ℝ) : Prop := abs x ≤ 2
noncomputable def q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

theorem p_necessary_but_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) := 
by 
  sorry

end p_necessary_but_not_sufficient_for_q_l221_221356


namespace calculateTotalProfit_l221_221303

-- Defining the initial investments and changes
def initialInvestmentA : ℕ := 5000
def initialInvestmentB : ℕ := 8000
def initialInvestmentC : ℕ := 9000

def additionalInvestmentA : ℕ := 2000
def withdrawnInvestmentB : ℕ := 1000
def additionalInvestmentC : ℕ := 3000

-- Defining the durations
def months1 : ℕ := 4
def months2 : ℕ := 8
def months3 : ℕ := 6

-- C's share of the profit
def shareOfC : ℕ := 45000

-- Total profit to be proved
def totalProfit : ℕ := 103571

-- Lean 4 theorem statement
theorem calculateTotalProfit :
  let ratioA := (initialInvestmentA * months1) + ((initialInvestmentA + additionalInvestmentA) * months2)
  let ratioB := (initialInvestmentB * months1) + ((initialInvestmentB - withdrawnInvestmentB) * months2)
  let ratioC := (initialInvestmentC * months3) + ((initialInvestmentC + additionalInvestmentC) * months3)
  let totalRatio := ratioA + ratioB + ratioC
  (shareOfC / ratioC : ℚ) = (totalProfit / totalRatio : ℚ) :=
sorry

end calculateTotalProfit_l221_221303


namespace possible_lengths_of_c_l221_221210

-- Definitions of the given conditions
variables (a b c : ℝ) (S : ℝ)
variables (h₁ : a = 4)
variables (h₂ : b = 5)
variables (h₃ : S = 5 * Real.sqrt 3)

-- The main theorem stating the possible lengths of c
theorem possible_lengths_of_c : c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
  sorry

end possible_lengths_of_c_l221_221210


namespace julia_drove_214_miles_l221_221911

def daily_rate : ℝ := 29
def cost_per_mile : ℝ := 0.08
def total_cost : ℝ := 46.12

theorem julia_drove_214_miles :
  (total_cost - daily_rate) / cost_per_mile = 214 :=
by
  sorry

end julia_drove_214_miles_l221_221911


namespace find_a_perpendicular_lines_l221_221176

theorem find_a_perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, ax + (a + 2) * y + 1 = 0 ∧ x + a * y + 2 = 0) → a = -3 :=
sorry

end find_a_perpendicular_lines_l221_221176


namespace find_meeting_time_l221_221302

-- Define the context and the problem parameters
def lisa_speed : ℝ := 9  -- Lisa's speed in mph
def adam_speed : ℝ := 7  -- Adam's speed in mph
def initial_distance : ℝ := 6  -- Initial distance in miles

-- The time in minutes for Lisa to meet Adam
theorem find_meeting_time : (initial_distance / (lisa_speed + adam_speed)) * 60 = 22.5 := by
  -- The proof is omitted for this statement
  sorry

end find_meeting_time_l221_221302


namespace max_value_of_a_plus_b_l221_221701

def max_possible_sum (a b : ℝ) (h1 : 4 * a + 3 * b ≤ 10) (h2 : a + 2 * b ≤ 4) : ℝ :=
  a + b

theorem max_value_of_a_plus_b :
  ∃a b : ℝ, (4 * a + 3 * b ≤ 10) ∧ (a + 2 * b ≤ 4) ∧ (a + b = 14 / 5) :=
by {
  sorry
}

end max_value_of_a_plus_b_l221_221701


namespace combined_molecular_weight_l221_221344

theorem combined_molecular_weight :
  let CaO_molecular_weight := 56.08
  let CO2_molecular_weight := 44.01
  let HNO3_molecular_weight := 63.01
  let moles_CaO := 5
  let moles_CO2 := 3
  let moles_HNO3 := 2
  moles_CaO * CaO_molecular_weight + moles_CO2 * CO2_molecular_weight + moles_HNO3 * HNO3_molecular_weight = 538.45 :=
by sorry

end combined_molecular_weight_l221_221344


namespace girls_bought_balloons_l221_221511

theorem girls_bought_balloons (initial_balloons boys_bought girls_bought remaining_balloons : ℕ)
  (h1 : initial_balloons = 36)
  (h2 : boys_bought = 3)
  (h3 : remaining_balloons = 21)
  (h4 : initial_balloons - remaining_balloons = boys_bought + girls_bought) :
  girls_bought = 12 := by
  sorry

end girls_bought_balloons_l221_221511


namespace geometric_sequence_sum_l221_221575

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_common_ratio : ∀ n, a (n + 1) = 2 * a n)
    (h_sum : a 1 + a 2 + a 3 = 21) : a 3 + a 4 + a 5 = 84 :=
sorry

end geometric_sequence_sum_l221_221575


namespace find_asterisk_value_l221_221984

theorem find_asterisk_value :
  ∃ x : ℤ, (x / 21) * (42 / 84) = 1 ↔ x = 21 :=
by
  sorry

end find_asterisk_value_l221_221984


namespace evaluate_expression_l221_221290

variable (x y : ℚ)

theorem evaluate_expression 
  (hx : x = 2) 
  (hy : y = -1 / 5) : 
  (2 * x - 3)^2 - (x + 2 * y) * (x - 2 * y) - 3 * y^2 + 3 = 1 / 25 :=
by
  sorry

end evaluate_expression_l221_221290


namespace brokerage_percentage_l221_221766

theorem brokerage_percentage
  (cash_realized : ℝ)
  (cash_before_brokerage : ℝ)
  (h₁ : cash_realized = 109.25)
  (h₂ : cash_before_brokerage = 109) :
  ((cash_realized - cash_before_brokerage) / cash_before_brokerage) * 100 = 0.23 := 
by
  sorry

end brokerage_percentage_l221_221766


namespace total_items_18_l221_221175

-- Define the number of dogs, biscuits per dog, and boots per set
def num_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 5
def boots_per_set : ℕ := 4

-- Calculate the total number of items
def total_items (num_dogs biscuits_per_dog boots_per_set : ℕ) : ℕ :=
  (num_dogs * biscuits_per_dog) + (num_dogs * boots_per_set)

-- Prove that the total number of items is 18
theorem total_items_18 : total_items num_dogs biscuits_per_dog boots_per_set = 18 := by
  -- Proof is not provided
  sorry

end total_items_18_l221_221175


namespace find_a1_l221_221361

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a n = a 0 + n * d

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = n / 2 * (a 1 + a n)

theorem find_a1 (d : ℝ) (h1 : a 13 = 13) (h2 : S 13 = 13) : a 0 = -11 :=
by
  sorry

end find_a1_l221_221361


namespace blue_tshirt_count_per_pack_l221_221686

theorem blue_tshirt_count_per_pack :
  ∀ (total_tshirts white_packs blue_packs tshirts_per_white_pack tshirts_per_blue_pack : ℕ), 
    white_packs = 3 →
    blue_packs = 2 → 
    tshirts_per_white_pack = 6 → 
    total_tshirts = 26 →
    total_tshirts = white_packs * tshirts_per_white_pack + blue_packs * tshirts_per_blue_pack →
  tshirts_per_blue_pack = 4 :=
by
  intros total_tshirts white_packs blue_packs tshirts_per_white_pack tshirts_per_blue_pack
  intros h1 h2 h3 h4 h5
  sorry

end blue_tshirt_count_per_pack_l221_221686


namespace bakery_flour_total_l221_221212

theorem bakery_flour_total :
  (0.2 + 0.1 + 0.15 + 0.05 + 0.1 = 0.6) :=
by {
  sorry
}

end bakery_flour_total_l221_221212


namespace reconstruct_quadrilateral_l221_221337

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A A'' B'' C'' D'' : V)

def trisect_segment (P Q R : V) : Prop :=
  Q = (1 / 3 : ℝ) • P + (2 / 3 : ℝ) • R

theorem reconstruct_quadrilateral
  (hB : trisect_segment A B A'')
  (hC : trisect_segment B C B'')
  (hD : trisect_segment C D C'')
  (hA : trisect_segment D A D'') :
  A = (2 / 26) • A'' + (6 / 26) • B'' + (6 / 26) • C'' + (12 / 26) • D'' :=
sorry

end reconstruct_quadrilateral_l221_221337


namespace highest_vs_lowest_temp_difference_l221_221068

theorem highest_vs_lowest_temp_difference 
  (highest_temp lowest_temp : ℤ) 
  (h_highest : highest_temp = 26) 
  (h_lowest : lowest_temp = 14) : 
  highest_temp - lowest_temp = 12 := 
by 
  sorry

end highest_vs_lowest_temp_difference_l221_221068


namespace algebraic_expression_evaluation_l221_221334

theorem algebraic_expression_evaluation (a b : ℝ) (h : -2 * a + 3 * b + 8 = 18) : 9 * b - 6 * a + 2 = 32 := by
  sorry

end algebraic_expression_evaluation_l221_221334


namespace gasoline_fraction_used_l221_221444

theorem gasoline_fraction_used
  (speed : ℕ) (gas_usage : ℕ) (initial_gallons : ℕ) (travel_time : ℕ)
  (h_speed : speed = 50) (h_gas_usage : gas_usage = 30) 
  (h_initial_gallons : initial_gallons = 15) (h_travel_time : travel_time = 5) :
  (speed * travel_time) / gas_usage / initial_gallons = 5 / 9 :=
by
  sorry

end gasoline_fraction_used_l221_221444


namespace smallest_base_10_integer_l221_221759

-- Given conditions
def is_valid_base (a b : ℕ) : Prop := a > 2 ∧ b > 2

def base_10_equivalence (a b n : ℕ) : Prop := (2 * a + 1 = n) ∧ (b + 2 = n)

-- The smallest base-10 integer represented as 21_a and 12_b
theorem smallest_base_10_integer :
  ∃ (a b n : ℕ), is_valid_base a b ∧ base_10_equivalence a b n ∧ n = 7 :=
by
  sorry

end smallest_base_10_integer_l221_221759


namespace arctan_sum_l221_221200

theorem arctan_sum : 
  let x := (3 : ℝ) / 7
  let y := 7 / 3
  x * y = 1 → (Real.arctan x + Real.arctan y = Real.pi / 2) :=
by
  intros x y h
  -- Proof goes here
  sorry

end arctan_sum_l221_221200


namespace value_of_expression_l221_221209

variables {a b c d e f : ℝ}

theorem value_of_expression
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 1 / 4 :=
sorry

end value_of_expression_l221_221209


namespace total_money_shared_l221_221754

theorem total_money_shared (A B C : ℕ) (rA rB rC : ℕ) (bens_share : ℕ) 
  (h_ratio : rA = 2 ∧ rB = 3 ∧ rC = 8)
  (h_ben : B = bens_share)
  (h_bensShareGiven : bens_share = 60) : 
  (rA * (bens_share / rB)) + bens_share + (rC * (bens_share / rB)) = 260 :=
by
  -- sorry to skip the proof
  sorry

end total_money_shared_l221_221754


namespace common_ratio_of_geometric_sequence_l221_221838

theorem common_ratio_of_geometric_sequence (S : ℕ → ℝ) (a_1 a_2 : ℝ) (q : ℝ)
  (h1 : S 3 = a_1 * (1 + q + q^2))
  (h2 : 2 * S 3 = 2 * a_1 + a_2) : 
  q = -1/2 := 
sorry

end common_ratio_of_geometric_sequence_l221_221838


namespace solution_set_l221_221315

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Given conditions
axiom differentiable_f : Differentiable ℝ f
axiom f_deriv : ∀ x, deriv f x = f' x
axiom f_at_3 : f 3 = 1
axiom inequality : ∀ x, 3 * f x + x * f' x > 1

-- Goal to prove
theorem solution_set :
  {x : ℝ | (x - 2017) ^ 3 * f (x - 2017) - 27 > 0} = {x | 2020 < x} :=
  sorry

end solution_set_l221_221315


namespace largest_prime_divisor_of_sum_of_squares_l221_221395

def a : ℕ := 35
def b : ℕ := 84

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, Prime p ∧ p = 13 ∧ (a^2 + b^2) % p = 0 := by
  sorry

end largest_prime_divisor_of_sum_of_squares_l221_221395


namespace player_jump_height_to_dunk_l221_221738

/-- Definitions given in the conditions -/
def rim_height : ℕ := 120
def player_height : ℕ := 72
def player_reach_above_head : ℕ := 22

/-- The statement to be proven -/
theorem player_jump_height_to_dunk :
  rim_height - (player_height + player_reach_above_head) = 26 :=
by
  sorry

end player_jump_height_to_dunk_l221_221738


namespace clara_total_points_l221_221462

-- Define the constants
def percentage_three_point_shots : ℝ := 0.25
def points_per_successful_three_point_shot : ℝ := 3
def percentage_two_point_shots : ℝ := 0.40
def points_per_successful_two_point_shot : ℝ := 2
def total_attempts : ℕ := 40

-- Define the function to calculate the total score
def total_score (x y : ℕ) : ℝ :=
  (percentage_three_point_shots * points_per_successful_three_point_shot) * x +
  (percentage_two_point_shots * points_per_successful_two_point_shot) * y

-- The proof statement
theorem clara_total_points (x y : ℕ) (h : x + y = total_attempts) : 
  total_score x y = 32 :=
by
  -- This is a placeholder for the actual proof
  sorry

end clara_total_points_l221_221462


namespace find_x_minus_y_l221_221215

theorem find_x_minus_y (x y : ℝ) (h1 : |x| + x - y = 14) (h2 : x + |y| + y = 6) : x - y = 8 :=
sorry

end find_x_minus_y_l221_221215


namespace jake_final_bitcoins_l221_221908

def initial_bitcoins : ℕ := 120
def investment_bitcoins : ℕ := 40
def returned_investment : ℕ := investment_bitcoins * 2
def bitcoins_after_investment : ℕ := initial_bitcoins - investment_bitcoins + returned_investment
def first_charity_donation : ℕ := 25
def bitcoins_after_first_donation : ℕ := bitcoins_after_investment - first_charity_donation
def brother_share : ℕ := 67
def bitcoins_after_giving_to_brother : ℕ := bitcoins_after_first_donation - brother_share
def debt_payment : ℕ := 5
def bitcoins_after_taking_back : ℕ := bitcoins_after_giving_to_brother + debt_payment
def quadrupled_bitcoins : ℕ := bitcoins_after_taking_back * 4
def second_charity_donation : ℕ := 15
def final_bitcoins : ℕ := quadrupled_bitcoins - second_charity_donation

theorem jake_final_bitcoins : final_bitcoins = 277 := by
  unfold final_bitcoins
  unfold quadrupled_bitcoins
  unfold bitcoins_after_taking_back
  unfold debt_payment
  unfold bitcoins_after_giving_to_brother
  unfold brother_share
  unfold bitcoins_after_first_donation
  unfold first_charity_donation
  unfold bitcoins_after_investment
  unfold returned_investment
  unfold investment_bitcoins
  unfold initial_bitcoins
  sorry

end jake_final_bitcoins_l221_221908


namespace three_digit_numbers_with_repeats_l221_221667

theorem three_digit_numbers_with_repeats :
  (let total_numbers := 9 * 10 * 10
   let non_repeating_numbers := 9 * 9 * 8
   total_numbers - non_repeating_numbers = 252) :=
by
  sorry

end three_digit_numbers_with_repeats_l221_221667


namespace correct_calculation_result_l221_221878

-- Define the conditions in Lean
variable (num : ℤ) (mistake_mult : ℤ) (result : ℤ)
variable (h_mistake : mistake_mult = num * 10) (h_result : result = 50)

-- The statement we want to prove
theorem correct_calculation_result 
  (h_mistake : mistake_mult = num * 10) 
  (h_result : result = 50) 
  (h_num_correct : num = result / 10) :
  (20 / num = 4) := sorry

end correct_calculation_result_l221_221878


namespace project_completion_days_l221_221377

-- Define the work rates and the total number of days to complete the project
variables (a_rate b_rate : ℝ) (days_to_complete : ℝ)
variable (a_quit_before_completion : ℝ)

-- Define the conditions
def A_rate := 1 / 20
def B_rate := 1 / 20
def quit_before_completion := 10 

-- The total work done in the project as 1 project 
def total_work := 1

-- Define the equation representing the amount of work done by A and B
def total_days := 
  A_rate * (days_to_complete - a_quit_before_completion) + B_rate * days_to_complete

-- The theorem statement
theorem project_completion_days :
  A_rate = a_rate → 
  B_rate = b_rate → 
  quit_before_completion = a_quit_before_completion → 
  total_days = total_work → 
  days_to_complete = 15 :=
by 
  -- placeholders for the conditions
  intros h1 h2 h3 h4
  sorry

end project_completion_days_l221_221377


namespace sum_geometric_sequence_divisibility_l221_221447

theorem sum_geometric_sequence_divisibility (n : ℕ) (h_pos: n > 0) :
  (n % 2 = 1 ↔ (3^(n+1) - 2^(n+1)) % 5 = 0) :=
sorry

end sum_geometric_sequence_divisibility_l221_221447


namespace triangle_perimeter_l221_221740

theorem triangle_perimeter (side1 side2 side3 : ℕ) (h1 : side1 = 40) (h2 : side2 = 50) (h3 : side3 = 70) : 
  side1 + side2 + side3 = 160 :=
by 
  sorry

end triangle_perimeter_l221_221740


namespace op_correct_l221_221856

-- Definition of the operation * for non-zero integers
def op (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 / b)

theorem op_correct (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 12) (h2 : a * b = 32) :
  op a b = 3 / 8 :=
by
  -- Proof, sorry for now
  sorry

end op_correct_l221_221856


namespace complement_union_sets_l221_221339

open Set

theorem complement_union_sets :
  ∀ (U A B : Set ℕ), (U = {1, 2, 3, 4}) → (A = {2, 3}) → (B = {3, 4}) → (U \ (A ∪ B) = {1}) :=
by
  intros U A B hU hA hB
  rw [hU, hA, hB]
  simp 
  sorry

end complement_union_sets_l221_221339


namespace difference_of_squares_expression_l221_221786

theorem difference_of_squares_expression
  (x y : ℝ) :
  (x + 2 * y) * (x - 2 * y) = x^2 - (2 * y)^2 :=
by sorry

end difference_of_squares_expression_l221_221786


namespace students_going_to_tournament_l221_221977

-- Defining the conditions
def total_students : ℕ := 24
def fraction_in_chess_program : ℚ := 1 / 3
def fraction_going_to_tournament : ℚ := 1 / 2

-- The final goal to prove
theorem students_going_to_tournament : 
  (total_students • fraction_in_chess_program) • fraction_going_to_tournament = 4 := 
by
  sorry

end students_going_to_tournament_l221_221977


namespace lines_intersection_l221_221225

theorem lines_intersection :
  ∃ (x y : ℝ), 
  (3 * y = -2 * x + 6) ∧ 
  (-4 * y = 3 * x + 4) ∧ 
  (x = -36) ∧ 
  (y = 26) :=
sorry

end lines_intersection_l221_221225


namespace polynomial_divisibility_l221_221030

-- Define the polynomial f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^3 - 8 * x^2 + m * x - 16

-- Prove that f(x) is divisible by x-2 if and only if m=8
theorem polynomial_divisibility (m : ℝ) :
  (∀ (x : ℝ), (x - 2) ∣ f x m) ↔ m = 8 := 
by
  sorry

end polynomial_divisibility_l221_221030


namespace boys_and_girls_solution_l221_221924

theorem boys_and_girls_solution (x y : ℕ) 
  (h1 : 3 * x + y > 24) 
  (h2 : 7 * x + 3 * y < 60) : x = 8 ∧ y = 1 :=
by
  sorry

end boys_and_girls_solution_l221_221924


namespace school_students_unique_l221_221448

theorem school_students_unique 
  (n : ℕ)
  (h1 : 70 < n) 
  (h2 : n < 130) 
  (h3 : n % 4 = 2) 
  (h4 : n % 5 = 2)
  (h5 : n % 6 = 2) : 
  (n = 92 ∨ n = 122) :=
  sorry

end school_students_unique_l221_221448


namespace divisor_of_7_l221_221620

theorem divisor_of_7 (a n : ℤ) (h1 : a ≥ 1) (h2 : a ∣ (n + 2)) (h3 : a ∣ (n^2 + n + 5)) : a = 1 ∨ a = 7 :=
by
  sorry

end divisor_of_7_l221_221620


namespace negation_of_exists_l221_221946

theorem negation_of_exists :
  ¬ (∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ ∀ x : ℝ, x^2 - 2*x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l221_221946


namespace binary_sum_eq_669_l221_221597

def binary111111111 : ℕ := 511
def binary1111111 : ℕ := 127
def binary11111 : ℕ := 31

theorem binary_sum_eq_669 :
  binary111111111 + binary1111111 + binary11111 = 669 :=
by
  sorry

end binary_sum_eq_669_l221_221597


namespace octagon_area_is_six_and_m_plus_n_is_seven_l221_221457

noncomputable def area_of_octagon (side_length : ℕ) (segment_length : ℚ) : ℚ :=
  let triangle_area := 1 / 2 * side_length * segment_length
  let octagon_area := 8 * triangle_area
  octagon_area

theorem octagon_area_is_six_and_m_plus_n_is_seven :
  area_of_octagon 2 (3/4) = 6 ∧ (6 + 1 = 7) :=
by
  sorry

end octagon_area_is_six_and_m_plus_n_is_seven_l221_221457


namespace f_f_minus_two_l221_221897

def f (x : ℚ) : ℚ := x⁻¹ + (x⁻¹ / (1 + x⁻¹))

theorem f_f_minus_two : f (f (-2)) = -8 / 3 := by
  sorry

end f_f_minus_two_l221_221897


namespace average_length_remaining_strings_l221_221883

theorem average_length_remaining_strings 
  (T1 : ℕ := 6) (avg_length1 : ℕ := 80) 
  (T2 : ℕ := 2) (avg_length2 : ℕ := 70) :
  (6 * avg_length1 - 2 * avg_length2) / 4 = 85 := 
by
  sorry

end average_length_remaining_strings_l221_221883


namespace interest_difference_correct_l221_221493

noncomputable def principal : ℝ := 1000
noncomputable def rate : ℝ := 0.10
noncomputable def time : ℝ := 4

noncomputable def simple_interest (P r t : ℝ) : ℝ := P * r * t
noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r)^t - P

noncomputable def interest_difference (P r t : ℝ) : ℝ := 
  compound_interest P r t - simple_interest P r t

theorem interest_difference_correct :
  interest_difference principal rate time = 64.10 :=
by
  sorry

end interest_difference_correct_l221_221493


namespace problem1_problem2_problem3_l221_221247

-- Definition of operation T
def T (x y m n : ℚ) := (m * x + n * y) * (x + 2 * y)

-- Problem 1: Given T(1, -1) = 0 and T(0, 2) = 8, prove m = 1 and n = 1
theorem problem1 (m n : ℚ) (h1 : T 1 (-1) m n = 0) (h2 : T 0 2 m n = 8) : m = 1 ∧ n = 1 := by
  sorry

-- Problem 2: Given the system of inequalities in terms of p and knowing T(x, y) = (mx + ny)(x + 2y) with m = 1 and n = 1
--            has exactly 3 integer solutions, prove the range of values for a is 42 ≤ a < 54
theorem problem2 (a : ℚ) 
  (h1 : ∃ p : ℚ, T (2 * p) (2 - p) 1 1 > 4 ∧ T (4 * p) (3 - 2 * p) 1 1 ≤ a)
  (h2 : ∃! p : ℤ, -1 < p ∧ p ≤ (a - 18) / 12) : 42 ≤ a ∧ a < 54 := by
  sorry

-- Problem 3: Given T(x, y) = T(y, x) when x^2 ≠ y^2, prove m = 2n
theorem problem3 (m n : ℚ) 
  (h : ∀ x y : ℚ, x^2 ≠ y^2 → T x y m n = T y x m n) : m = 2 * n := by
  sorry

end problem1_problem2_problem3_l221_221247


namespace smallest_positive_integer_l221_221918

def smallest_x (x : ℕ) : Prop :=
  (540 * x) % 800 = 0

theorem smallest_positive_integer (x : ℕ) : smallest_x x → x = 80 :=
by {
  sorry
}

end smallest_positive_integer_l221_221918


namespace john_initial_candies_l221_221861

theorem john_initial_candies : ∃ x : ℕ, (∃ (x3 : ℕ), x3 = ((x - 2) / 2) ∧ x3 = 6) ∧ x = 14 := by
  sorry

end john_initial_candies_l221_221861


namespace derek_joe_ratio_l221_221466

theorem derek_joe_ratio (D J T : ℝ) (h0 : J = 23) (h1 : T = 30) (h2 : T = (1/3 : ℝ) * D + 16) :
  D / J = 42 / 23 :=
by
  sorry

end derek_joe_ratio_l221_221466


namespace ice_rink_rental_fee_l221_221887

/-!
  # Problem:
  An ice skating rink charges $5 for admission and a certain amount to rent skates. 
  Jill can purchase a new pair of skates for $65. She would need to go to the rink 26 times 
  to justify buying the skates rather than renting a pair. How much does the rink charge to rent skates?
-/

/-- Lean statement of the problem. --/
theorem ice_rink_rental_fee 
  (admission_fee : ℝ) (skates_cost : ℝ) (num_visits : ℕ)
  (total_buying_cost : ℝ) (total_renting_cost : ℝ)
  (rental_fee : ℝ) :
  admission_fee = 5 ∧
  skates_cost = 65 ∧
  num_visits = 26 ∧
  total_buying_cost = skates_cost + (admission_fee * num_visits) ∧
  total_renting_cost = (admission_fee + rental_fee) * num_visits ∧
  total_buying_cost = total_renting_cost →
  rental_fee = 2.50 :=
by
  intros h
  sorry

end ice_rink_rental_fee_l221_221887


namespace rectangular_prism_has_8_vertices_l221_221926

def rectangular_prism_vertices := 8

theorem rectangular_prism_has_8_vertices : rectangular_prism_vertices = 8 := by
  sorry

end rectangular_prism_has_8_vertices_l221_221926


namespace simplify_complex_expression_l221_221158

theorem simplify_complex_expression :
  (1 / (-8 ^ 2) ^ 4 * (-8) ^ 9) = -8 := by
  sorry

end simplify_complex_expression_l221_221158


namespace girls_to_boys_ratio_l221_221999

variable (g b : ℕ)
variable (h_total : g + b = 36)
variable (h_diff : g = b + 6)

theorem girls_to_boys_ratio (g b : ℕ) (h_total : g + b = 36) (h_diff : g = b + 6) :
  g / b = 7 / 5 := by
  sorry

end girls_to_boys_ratio_l221_221999


namespace second_chapter_pages_l221_221899

theorem second_chapter_pages (x : ℕ) (h1 : 48 = x + 37) : x = 11 := 
sorry

end second_chapter_pages_l221_221899


namespace percentage_discount_l221_221671

def cost_per_ball : ℝ := 0.1
def number_of_balls : ℕ := 10000
def amount_paid : ℝ := 700

theorem percentage_discount : (number_of_balls * cost_per_ball - amount_paid) / (number_of_balls * cost_per_ball) * 100 = 30 :=
by
  sorry

end percentage_discount_l221_221671


namespace find_xyz_l221_221474

theorem find_xyz (x y z : ℝ) 
  (h1 : x * (y + z) = 180) 
  (h2 : y * (z + x) = 192) 
  (h3 : z * (x + y) = 204) 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z) : 
  x * y * z = 168 * Real.sqrt 6 :=
sorry

end find_xyz_l221_221474


namespace percentage_scientists_born_in_june_l221_221007

theorem percentage_scientists_born_in_june :
  (18 / 200 * 100) = 9 :=
by sorry

end percentage_scientists_born_in_june_l221_221007


namespace problem_1_problem_2_problem_3_l221_221852

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (2^x + 1)
noncomputable def f_inv (x : ℝ) : ℝ := Real.logb 2 (2^x - 1)

theorem problem_1 : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f_inv x = m + f x) ↔ 
  m ∈ (Set.Icc (Real.logb 2 (1/3)) (Real.logb 2 (3/5))) :=
sorry

theorem problem_2 : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f_inv x > m + f x) ↔ 
  m ∈ (Set.Iio (Real.logb 2 (3/5))) :=
sorry

theorem problem_3 : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f_inv x > m + f x) ↔ 
  m ∈ (Set.Iio (Real.logb 2 (1/3))) :=
sorry

end problem_1_problem_2_problem_3_l221_221852


namespace MrSlinkums_total_count_l221_221791

variable (T : ℕ)

-- Defining the conditions as given in the problem
def placed_on_shelves (T : ℕ) : ℕ := (20 * T) / 100
def storage (T : ℕ) : ℕ := (80 * T) / 100

-- Stating the main theorem to prove
theorem MrSlinkums_total_count 
    (h : storage T = 120) : 
    T = 150 :=
sorry

end MrSlinkums_total_count_l221_221791


namespace total_spent_correct_l221_221160

def cost_ornamental_plants : Float := 467.00
def cost_garden_tool_set : Float := 85.00
def cost_potting_soil : Float := 38.00

def discount_plants : Float := 0.15
def discount_tools : Float := 0.10
def discount_soil : Float := 0.00

def sales_tax_rate : Float := 0.08
def surcharge : Float := 12.00

def discounted_price (original_price : Float) (discount_rate : Float) : Float :=
  original_price * (1.0 - discount_rate)

def subtotal (price_plants : Float) (price_tools : Float) (price_soil : Float) : Float :=
  price_plants + price_tools + price_soil

def sales_tax (amount : Float) (tax_rate : Float) : Float :=
  amount * tax_rate

def total (subtotal : Float) (sales_tax : Float) (surcharge : Float) : Float :=
  subtotal + sales_tax + surcharge

def final_total_spent : Float :=
  let price_plants := discounted_price cost_ornamental_plants discount_plants
  let price_tools := discounted_price cost_garden_tool_set discount_tools
  let price_soil := cost_potting_soil
  let subtotal_amount := subtotal price_plants price_tools price_soil
  let tax_amount := sales_tax subtotal_amount sales_tax_rate
  total subtotal_amount tax_amount surcharge

theorem total_spent_correct : final_total_spent = 564.37 :=
  by sorry

end total_spent_correct_l221_221160


namespace solve_for_p_l221_221164

theorem solve_for_p (q p : ℝ) (h : p^2 * q = p * q + p^2) : 
  p = 0 ∨ p = q / (q - 1) :=
by
  sorry

end solve_for_p_l221_221164


namespace find_P_coordinates_l221_221837

-- Define points A and B
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, -3)

-- Define the theorem
theorem find_P_coordinates :
  ∃ P : ℝ × ℝ, P = (8, -15) ∧ (P.1 - A.1, P.2 - A.2) = (3 * (B.1 - A.1), 3 * (B.2 - A.2)) :=
sorry

end find_P_coordinates_l221_221837


namespace quarters_percentage_value_l221_221418

theorem quarters_percentage_value (dimes quarters : Nat) (value_dime value_quarter : Nat) (total_value quarter_value : Nat)
(h_dimes : dimes = 30)
(h_quarters : quarters = 40)
(h_value_dime : value_dime = 10)
(h_value_quarter : value_quarter = 25)
(h_total_value : total_value = dimes * value_dime + quarters * value_quarter)
(h_quarter_value : quarter_value = quarters * value_quarter) :
(quarter_value : ℚ) / (total_value : ℚ) * 100 = 76.92 := 
sorry

end quarters_percentage_value_l221_221418


namespace open_box_volume_l221_221580

theorem open_box_volume (l w s : ℝ) (hl : l = 48) (hw : w = 36) (hs : s = 8) :
  (l - 2 * s) * (w - 2 * s) * s = 5120 :=
by
  sorry

end open_box_volume_l221_221580


namespace cost_price_one_metre_l221_221588

noncomputable def selling_price : ℤ := 18000
noncomputable def total_metres : ℕ := 600
noncomputable def loss_per_metre : ℤ := 5

noncomputable def total_loss : ℤ := loss_per_metre * (total_metres : ℤ) -- Note the cast to ℤ for multiplication
noncomputable def cost_price : ℤ := selling_price + total_loss
noncomputable def cost_price_per_metre : ℚ := cost_price / (total_metres : ℤ)

theorem cost_price_one_metre : cost_price_per_metre = 35 := by
  sorry

end cost_price_one_metre_l221_221588


namespace john_new_cards_l221_221909

def cards_per_page : ℕ := 3
def old_cards : ℕ := 16
def pages_used : ℕ := 8

theorem john_new_cards : (pages_used * cards_per_page) - old_cards = 8 := by
  sorry

end john_new_cards_l221_221909


namespace molecular_weight_CaO_is_56_08_l221_221864

-- Define the atomic weights of Calcium and Oxygen
def atomic_weight_Ca := 40.08 -- in g/mol
def atomic_weight_O := 16.00 -- in g/mol

-- Define the molecular weight of the compound
def molecular_weight_CaO := atomic_weight_Ca + atomic_weight_O

-- State the theorem
theorem molecular_weight_CaO_is_56_08 : molecular_weight_CaO = 56.08 :=
by
  -- The proof will be filled in here
  sorry

end molecular_weight_CaO_is_56_08_l221_221864


namespace simplify_expression_l221_221967

noncomputable def q (x a b c d : ℝ) :=
  (x + a)^4 / ((a - b) * (a - c) * (a - d))
  + (x + b)^4 / ((b - a) * (b - c) * (b - d))
  + (x + c)^4 / ((c - a) * (c - b) * (c - d))
  + (x + d)^4 / ((d - a) * (d - b) * (d - c))

theorem simplify_expression (a b c d x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) :
  q x a b c d = a + b + c + d + 4 * x :=
by
  sorry

end simplify_expression_l221_221967


namespace quadratic_factorization_l221_221771

theorem quadratic_factorization (C D : ℤ) (h : (15 * y^2 - 74 * y + 48) = (C * y - 16) * (D * y - 3)) :
  C * D + C = 20 :=
sorry

end quadratic_factorization_l221_221771


namespace seedlings_planted_l221_221274

theorem seedlings_planted (x : ℕ) (h1 : 2 * x + x = 1200) : x = 400 :=
by {
  sorry
}

end seedlings_planted_l221_221274


namespace total_amount_paid_l221_221223

def original_price_per_card : Int := 12
def discount_per_card : Int := 2
def number_of_cards : Int := 10

theorem total_amount_paid :
  original_price_per_card - discount_per_card * number_of_cards = 100 :=
by
  sorry

end total_amount_paid_l221_221223


namespace preimages_of_one_under_f_l221_221229

theorem preimages_of_one_under_f :
  {x : ℝ | (x^3 - x + 1 = 1)} = {-1, 0, 1} := by
  sorry

end preimages_of_one_under_f_l221_221229


namespace pigs_count_l221_221446

-- Definitions from step a)
def pigs_leg_count : ℕ := 4 -- Each pig has 4 legs
def hens_leg_count : ℕ := 2 -- Each hen has 2 legs

variable {P H : ℕ} -- P is the number of pigs, H is the number of hens

-- Condition from step a) as a function
def total_legs (P H : ℕ) : ℕ := pigs_leg_count * P + hens_leg_count * H
def total_heads (P H : ℕ) : ℕ := P + H

-- Theorem to prove the number of pigs given the condition
theorem pigs_count {P H : ℕ} (h : total_legs P H = 2 * total_heads P H + 22) : P = 11 :=
  by 
    sorry

end pigs_count_l221_221446


namespace remainder_approximately_14_l221_221907

def dividend : ℝ := 14698
def quotient : ℝ := 89
def divisor : ℝ := 164.98876404494382
def remainder : ℝ := dividend - (quotient * divisor)

theorem remainder_approximately_14 : abs (remainder - 14) < 1e-10 := 
by
-- using abs since the problem is numerical/approximate
sorry

end remainder_approximately_14_l221_221907


namespace pentagon_largest_angle_l221_221665

theorem pentagon_largest_angle
  (F G H I J : ℝ)
  (hF : F = 90)
  (hG : G = 70)
  (hH_eq_I : H = I)
  (hJ : J = 2 * H + 20)
  (sum_angles : F + G + H + I + J = 540) :
  max F (max G (max H (max I J))) = 200 :=
by
  sorry

end pentagon_largest_angle_l221_221665


namespace number_of_prize_orders_l221_221313

/-- At the end of a professional bowling tournament, the top 6 bowlers have a playoff.
    - #6 and #5 play a game. The loser receives the 6th prize and the winner plays #4.
    - The loser of the second game receives the 5th prize and the winner plays #3.
    - The loser of the third game receives the 4th prize and the winner plays #2.
    - The loser of the fourth game receives the 3rd prize and the winner plays #1.
    - The winner of the final game gets 1st prize and the loser gets 2nd prize.

    We want to determine the number of possible orders in which the bowlers can receive the prizes.
-/
theorem number_of_prize_orders : 2^5 = 32 := by
  sorry

end number_of_prize_orders_l221_221313


namespace non_adjacent_boys_arrangements_l221_221712

-- We define the number of boys and girls
def boys := 4
def girls := 6

-- The function to compute combinations C(n, k)
def combinations (n k : ℕ) : ℕ := Nat.choose n k

-- The function to compute permutations P(n, k)
def permutations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- The total arrangements where 2 selected boys are not adjacent
def total_non_adjacent_arrangements : ℕ :=
  (combinations boys 2) * (combinations girls 3) * (permutations 3 3) * (permutations (3 + 1) 2)

theorem non_adjacent_boys_arrangements :
  total_non_adjacent_arrangements = 8640 := by
  sorry

end non_adjacent_boys_arrangements_l221_221712


namespace max_n_factorable_l221_221399

theorem max_n_factorable :
  ∃ n : ℤ, (∀ A B : ℤ, 3 * A * B = 24 → 3 * B + A = n) ∧ (n = 73) :=
sorry

end max_n_factorable_l221_221399


namespace find_total_pupils_l221_221277

-- Define the conditions for the problem
def diff1 : ℕ := 85 - 45
def diff2 : ℕ := 79 - 49
def diff3 : ℕ := 64 - 34
def total_diff : ℕ := diff1 + diff2 + diff3
def avg_increase : ℕ := 3

-- Assert that the number of pupils n satisfies the given conditions
theorem find_total_pupils (n : ℕ) (h_diff : total_diff = 100) (h_avg_inc : avg_increase * n = total_diff) : n = 33 :=
by
  sorry

end find_total_pupils_l221_221277


namespace range_of_k_l221_221384

def f : ℝ → ℝ := sorry

axiom cond1 (a b : ℝ) : f (a + b) = f a + f b + 2 * a * b
axiom cond2 (k : ℝ) : ∀ x : ℝ, f (x + k) = f (k - x)
axiom cond3 : ∀ x y : ℝ, 1 ≤ x → x ≤ y → y ≤ 2 → f x ≤ f y

theorem range_of_k (k : ℝ) : k ≤ 1 :=
sorry

end range_of_k_l221_221384


namespace param_A_valid_param_B_valid_l221_221093

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := y = 2 * x - 4

-- Parameterization A
def param_A (t : ℝ) : ℝ × ℝ := (2 - t, -2 * t)

-- Parameterization B
def param_B (t : ℝ) : ℝ × ℝ := (5 * t, 10 * t - 4)

-- Theorem to prove that parameterization A satisfies the line equation
theorem param_A_valid (t : ℝ) : line_eq (param_A t).1 (param_A t).2 := by
  sorry

-- Theorem to prove that parameterization B satisfies the line equation
theorem param_B_valid (t : ℝ) : line_eq (param_B t).1 (param_B t).2 := by
  sorry

end param_A_valid_param_B_valid_l221_221093


namespace arithmetic_sequence_8th_term_l221_221595

theorem arithmetic_sequence_8th_term 
  (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 41) : 
  a + 7 * d = 59 := 
by 
  sorry

end arithmetic_sequence_8th_term_l221_221595


namespace fraction_by_foot_l221_221785

theorem fraction_by_foot (D distance_by_bus distance_by_car distance_by_foot : ℕ) (h1 : D = 24) 
  (h2 : distance_by_bus = D / 4) (h3 : distance_by_car = 6) 
  (h4 : distance_by_foot = D - (distance_by_bus + distance_by_car)) : 
  (distance_by_foot : ℚ) / D = 1 / 2 :=
by
  sorry

end fraction_by_foot_l221_221785


namespace find_star_1993_1935_l221_221594

axiom star (x y : ℕ) : ℕ
axiom star_idempotent (x : ℕ) : star x x = 0
axiom star_assoc (x y z : ℕ) : star x (star y z) = star x y + z

theorem find_star_1993_1935 : star 1993 1935 = 58 :=
by
  sorry

end find_star_1993_1935_l221_221594


namespace mod_division_l221_221816

theorem mod_division (N : ℕ) (h₁ : N = 5 * 2 + 0) : N % 4 = 2 :=
by sorry

end mod_division_l221_221816


namespace sum_of_x_and_y_l221_221869

theorem sum_of_x_and_y :
  ∀ (x y : ℚ), (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 :=
by
  intros x y h1 h2
  sorry

end sum_of_x_and_y_l221_221869


namespace customer_ordered_bags_l221_221469

def bags_per_batch : Nat := 10
def initial_bags : Nat := 20
def days : Nat := 4
def batches_per_day : Nat := 1

theorem customer_ordered_bags : 
  initial_bags + days * batches_per_day * bags_per_batch = 60 :=
by
  sorry

end customer_ordered_bags_l221_221469


namespace area_of_tangent_triangle_l221_221204

noncomputable def tangentTriangleArea : ℝ :=
  let y := λ x : ℝ => x^3 + x
  let dy := λ x : ℝ => 3 * x^2 + 1
  let slope := dy 1
  let y_intercept := 2 - slope * 1
  let x_intercept := - y_intercept / slope
  let base := x_intercept
  let height := - y_intercept
  0.5 * base * height

theorem area_of_tangent_triangle :
  tangentTriangleArea = 1 / 2 :=
by
  sorry

end area_of_tangent_triangle_l221_221204


namespace woman_speed_still_water_l221_221028

theorem woman_speed_still_water (v_w v_c : ℝ) 
    (h1 : 120 = (v_w + v_c) * 10)
    (h2 : 24 = (v_w - v_c) * 14) : 
    v_w = 48 / 7 :=
by {
  sorry
}

end woman_speed_still_water_l221_221028


namespace smallest_sum_of_squares_l221_221987

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 187) : x^2 + y^2 ≥ 205 := sorry

end smallest_sum_of_squares_l221_221987


namespace red_pencil_count_l221_221895

-- Definitions for provided conditions
def blue_pencils : ℕ := 20
def ratio : ℕ × ℕ := (5, 3)
def red_pencils (blue : ℕ) (rat : ℕ × ℕ) : ℕ := (blue / rat.fst) * rat.snd

-- Theorem statement
theorem red_pencil_count : red_pencils blue_pencils ratio = 12 := 
by
  sorry

end red_pencil_count_l221_221895


namespace min_a2_b2_l221_221031

theorem min_a2_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) :
  a^2 + b^2 ≥ 4 / 5 :=
sorry

end min_a2_b2_l221_221031


namespace probability_of_both_selected_l221_221026

theorem probability_of_both_selected :
  let pX := 1 / 5
  let pY := 2 / 7
  (pX * pY) = 2 / 35 :=
by
  let pX := 1 / 5
  let pY := 2 / 7
  show (pX * pY) = 2 / 35
  sorry

end probability_of_both_selected_l221_221026


namespace trace_bags_weight_l221_221317

theorem trace_bags_weight :
  ∀ (g1 g2 t1 t2 t3 t4 t5 : ℕ),
    g1 = 3 →
    g2 = 7 →
    (g1 + g2) = (t1 + t2 + t3 + t4 + t5) →
    (t1 = t2 ∧ t2 = t3 ∧ t3 = t4 ∧ t4 = t5) →
    t1 = 2 :=
by
  intros g1 g2 t1 t2 t3 t4 t5 hg1 hg2 hsum hsame
  sorry

end trace_bags_weight_l221_221317


namespace paint_area_is_correct_l221_221003

-- Define the dimensions of the wall, window, and door
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window_height : ℕ := 3
def window_length : ℕ := 5
def door_height : ℕ := 1
def door_length : ℕ := 7

-- Calculate area
def wall_area : ℕ := wall_height * wall_length
def window_area : ℕ := window_height * window_length
def door_area : ℕ := door_height * door_length

-- Calculate area to be painted
def area_to_be_painted : ℕ := wall_area - window_area - door_area

-- The theorem statement
theorem paint_area_is_correct : area_to_be_painted = 128 := 
by
  -- The proof would go here (omitted)
  sorry

end paint_area_is_correct_l221_221003


namespace profit_percentage_l221_221928

theorem profit_percentage (SP CP : ℝ) (h_SP : SP = 150) (h_CP : CP = 120) : 
  ((SP - CP) / CP) * 100 = 25 :=
by {
  sorry
}

end profit_percentage_l221_221928


namespace product_of_three_numbers_l221_221353

theorem product_of_three_numbers (a b c : ℚ) 
  (h₁ : a + b + c = 30)
  (h₂ : a = 6 * (b + c))
  (h₃ : b = 5 * c) : 
  a * b * c = 22500 / 343 := 
sorry

end product_of_three_numbers_l221_221353


namespace evaluate_expression_l221_221118

theorem evaluate_expression :
  (4^2 - 4) + (5^2 - 5) - (7^3 - 7) + (3^2 - 3) = -298 :=
by
  sorry

end evaluate_expression_l221_221118


namespace solution_values_sum_l221_221157

theorem solution_values_sum (x y : ℝ) (p q r s : ℕ) 
  (hx : x + y = 5) 
  (hxy : 2 * x * y = 5) 
  (hx_form : x = (p + q * Real.sqrt r) / s ∨ x = (p - q * Real.sqrt r) / s) 
  (hpqs_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0) : 
  p + q + r + s = 23 := 
sorry

end solution_values_sum_l221_221157


namespace total_rain_duration_l221_221695

theorem total_rain_duration:
  let first_day_duration := 10
  let second_day_duration := first_day_duration + 2
  let third_day_duration := 2 * second_day_duration
  first_day_duration + second_day_duration + third_day_duration = 46 :=
by
  sorry

end total_rain_duration_l221_221695


namespace sin_zero_necessary_not_sufficient_l221_221640

theorem sin_zero_necessary_not_sufficient:
  (∀ α : ℝ, (∃ k : ℤ, α = 2 * k * Real.pi) → (Real.sin α = 0)) ∧
  ¬ (∀ α : ℝ, (Real.sin α = 0) → (∃ k : ℤ, α = 2 * k * Real.pi)) :=
by
  sorry

end sin_zero_necessary_not_sufficient_l221_221640


namespace carson_gold_stars_l221_221397

theorem carson_gold_stars (gold_stars_yesterday gold_stars_today : ℕ) (h1 : gold_stars_yesterday = 6) (h2 : gold_stars_today = 9) : 
  gold_stars_yesterday + gold_stars_today = 15 := 
by
  sorry

end carson_gold_stars_l221_221397


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l221_221468

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l221_221468


namespace solve_for_x_l221_221651

theorem solve_for_x (x: ℝ) (h: (x-3)^4 = 16): x = 5 := 
by
  sorry

end solve_for_x_l221_221651


namespace algebraic_expression_is_product_l221_221349

def algebraicExpressionMeaning (x : ℝ) : Prop :=
  -7 * x = -7 * x

theorem algebraic_expression_is_product (x : ℝ) :
  algebraicExpressionMeaning x :=
by
  sorry

end algebraic_expression_is_product_l221_221349


namespace arithmetic_sequence_fifth_term_l221_221419

theorem arithmetic_sequence_fifth_term (x y : ℚ) 
  (h1 : a₁ = x + y) 
  (h2 : a₂ = x - y) 
  (h3 : a₃ = x * y) 
  (h4 : a₄ = x / y) 
  (h5 : a₂ - a₁ = -2 * y) 
  (h6 : a₃ - a₂ = -2 * y) 
  (h7 : a₄ - a₃ = -2 * y) 
  (hx : x = -9 / 8)
  (hy : y = -3 / 5) : 
  a₅ = 123 / 40 :=
by
  sorry

end arithmetic_sequence_fifth_term_l221_221419


namespace scientific_notation_of_million_l221_221236

theorem scientific_notation_of_million (x : ℝ) (h : x = 56.99) : 56.99 * 10^6 = 5.699 * 10^7 :=
by
  sorry

end scientific_notation_of_million_l221_221236


namespace minimal_rope_cost_l221_221043

theorem minimal_rope_cost :
  let pieces_needed := 10
  let length_per_piece := 6 -- inches
  let total_length_needed := pieces_needed * length_per_piece -- inches
  let one_foot_length := 12 -- inches
  let cost_six_foot_rope := 5 -- dollars
  let cost_one_foot_rope := 1.25 -- dollars
  let six_foot_length := 6 * one_foot_length -- inches
  let one_foot_total_cost := (total_length_needed / one_foot_length) * cost_one_foot_rope
  let six_foot_total_cost := cost_six_foot_rope
  total_length_needed <= six_foot_length ∧ six_foot_total_cost < one_foot_total_cost →
  six_foot_total_cost = 5 := sorry

end minimal_rope_cost_l221_221043


namespace probability_four_coins_l221_221490

-- Define four fair coin flips, having 2 possible outcomes for each coin
def four_coin_flips_outcomes : ℕ := 2 ^ 4

-- Define the favorable outcomes: all heads or all tails
def favorable_outcomes : ℕ := 2

-- The probability of getting all heads or all tails
def probability_all_heads_or_tails : ℚ := favorable_outcomes / four_coin_flips_outcomes

-- The theorem stating the answer to the problem
theorem probability_four_coins:
  probability_all_heads_or_tails = 1 / 8 := by
  sorry

end probability_four_coins_l221_221490


namespace find_b_l221_221483

theorem find_b 
    (x1 x2 b c : ℝ)
    (h_distinct : x1 ≠ x2)
    (h_root_x : ∀ x, (x^2 + 5 * b * x + c = 0) → x = x1 ∨ x = x2)
    (h_common_root : ∃ y, (y^2 + 2 * x1 * y + 2 * x2 = 0) ∧ (y^2 + 2 * x2 * y + 2 * x1 = 0)) :
  b = 1 / 10 := 
sorry

end find_b_l221_221483


namespace skittles_transfer_l221_221958

-- Define the initial number of Skittles Bridget and Henry have
def bridget_initial_skittles := 4
def henry_initial_skittles := 4

-- The main statement we want to prove
theorem skittles_transfer :
  bridget_initial_skittles + henry_initial_skittles = 8 :=
by
  sorry

end skittles_transfer_l221_221958


namespace fraction_combination_l221_221639

theorem fraction_combination (x y : ℝ) (h : y / x = 3 / 4) : (x + y) / x = 7 / 4 :=
by
  -- Proof steps will be inserted here (for now using sorry)
  sorry

end fraction_combination_l221_221639


namespace alice_bob_meet_same_point_in_5_turns_l221_221472

theorem alice_bob_meet_same_point_in_5_turns :
  ∃ k : ℕ, k = 5 ∧ 
  (∀ n, (1 + 7 * n) % 24 = 12 ↔ (n = k)) :=
by
  sorry

end alice_bob_meet_same_point_in_5_turns_l221_221472


namespace max_dot_and_area_of_triangle_l221_221851

noncomputable def triangle_data (A B C : ℝ) (m n : ℝ × ℝ) : Prop :=
  A + B + C = Real.pi ∧
  (m = (2, 2 * (Real.cos ((B + C) / 2))^2 - 1)) ∧
  (n = (Real.sin (A / 2), -1))

noncomputable def is_max_dot_product (A : ℝ) (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = (if A = Real.pi / 3 then 3 / 2 else 0)

noncomputable def max_area (A B C : ℝ) : ℝ :=
  let a : ℝ := 2
  let b : ℝ := 2
  let c : ℝ := 2
  if A = Real.pi / 3 then (Real.sqrt 3) else 0

theorem max_dot_and_area_of_triangle {A B C : ℝ} {m n : ℝ × ℝ}
  (h_triangle : triangle_data A B C m n) :
  is_max_dot_product (Real.pi / 3) m n ∧ max_area A B C = Real.sqrt 3 := by sorry

end max_dot_and_area_of_triangle_l221_221851


namespace meaningful_iff_x_ne_2_l221_221276

theorem meaningful_iff_x_ne_2 (x : ℝ) : (x ≠ 2) ↔ (∃ y : ℝ, y = (x - 3) / (x - 2)) := 
by
  sorry

end meaningful_iff_x_ne_2_l221_221276


namespace allie_betty_total_points_product_l221_221944

def score (n : Nat) : Nat :=
  if n % 3 == 0 then 9
  else if n % 2 == 0 then 3
  else if n % 2 == 1 then 1
  else 0

def allie_points : List Nat := [5, 2, 6, 1, 3]
def betty_points : List Nat := [6, 4, 1, 2, 5]

def total_points (rolls: List Nat) : Nat :=
  rolls.foldl (λ acc n => acc + score n) 0

theorem allie_betty_total_points_product : 
  total_points allie_points * total_points betty_points = 391 := by
  sorry

end allie_betty_total_points_product_l221_221944


namespace part_I_l221_221549

variable (a b c n p q : ℝ)

theorem part_I (hne0 : a ≠ 0) (bne0 : b ≠ 0) (cne0 : c ≠ 0)
    (h1 : a^2 + b^2 + c^2 = 2) (h2 : n^2 + p^2 + q^2 = 2) :
    (n^4 / a^2 + p^4 / b^2 + q^4 / c^2) ≥ 2 := 
sorry

end part_I_l221_221549


namespace Lisa_income_percentage_J_M_combined_l221_221560

variables (T M J L : ℝ)

-- Conditions as definitions
def Mary_income_eq_1p6_T (M T : ℝ) : Prop := M = 1.60 * T
def Tim_income_eq_0p5_J (T J : ℝ) : Prop := T = 0.50 * J
def Lisa_income_eq_1p3_M (L M : ℝ) : Prop := L = 1.30 * M
def Lisa_income_eq_0p75_J (L J : ℝ) : Prop := L = 0.75 * J

-- Theorem statement
theorem Lisa_income_percentage_J_M_combined (M T J L : ℝ)
  (h1 : Mary_income_eq_1p6_T M T)
  (h2 : Tim_income_eq_0p5_J T J)
  (h3 : Lisa_income_eq_1p3_M L M)
  (h4 : Lisa_income_eq_0p75_J L J) :
  (L / (M + J)) * 100 = 41.67 := 
sorry

end Lisa_income_percentage_J_M_combined_l221_221560


namespace inequality_solution_l221_221105

theorem inequality_solution (x : ℝ) :
    (x < 1 ∨ (3 < x ∧ x < 4) ∨ (4 < x ∧ x < 5) ∨ (5 < x ∧ x < 6) ∨ x > 6) ↔
    ((x - 1) * (x - 3) * (x - 4) / ((x - 2) * (x - 5) * (x - 6)) > 0) := by
  sorry

end inequality_solution_l221_221105


namespace cookies_last_days_l221_221201

variable (c1 c2 t : ℕ)

/-- Jackson's oldest son gets 4 cookies after school each day, and his youngest son gets 2 cookies. 
There are 54 cookies in the box, so the number of days the box will last is 9. -/
theorem cookies_last_days (h1 : c1 = 4) (h2 : c2 = 2) (h3 : t = 54) : 
  t / (c1 + c2) = 9 := by
  sorry

end cookies_last_days_l221_221201


namespace remaining_money_l221_221941

theorem remaining_money (m : ℝ) (c f t r : ℝ)
  (h_initial : m = 1500)
  (h_clothes : c = (1 / 3) * m)
  (h_food : f = (1 / 5) * (m - c))
  (h_travel : t = (1 / 4) * (m - c - f))
  (h_remaining : r = m - c - f - t) :
  r = 600 := 
by
  sorry

end remaining_money_l221_221941


namespace symmetry_axis_of_function_l221_221507

theorem symmetry_axis_of_function {x : ℝ} :
  (∃ k : ℤ, ∃ x : ℝ, (y = 2 * (Real.cos ((x / 2) + (Real.pi / 3))) ^ 2 - 1) ∧ (x + (2 * Real.pi) / 3 = k * Real.pi)) →
    x = (Real.pi / 3) ∧ 0 = y :=
sorry

end symmetry_axis_of_function_l221_221507


namespace algebraic_identity_l221_221222

theorem algebraic_identity (a b c d : ℝ) : a - b + c - d = a + c - (b + d) :=
by
  sorry

end algebraic_identity_l221_221222


namespace problem_l221_221548

noncomputable def a : ℝ := (Real.sqrt 5 + Real.sqrt 3) / (Real.sqrt 5 - Real.sqrt 3)
noncomputable def b : ℝ := (Real.sqrt 5 - Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3)

theorem problem :
  a^4 + b^4 + (a + b)^4 = 7938 := by
  sorry

end problem_l221_221548


namespace remi_water_bottle_capacity_l221_221678

-- Let's define the problem conditions
def daily_refills : ℕ := 3
def days : ℕ := 7
def total_spilled : ℕ := 5 + 8 -- Total spilled water in ounces
def total_intake : ℕ := 407 -- Total amount of water drunk in 7 days

-- The capacity of Remi's water bottle is the quantity we need to prove
def bottle_capacity (x : ℕ) : Prop :=
  daily_refills * days * x - total_spilled = total_intake

-- Statement of the proof problem
theorem remi_water_bottle_capacity : bottle_capacity 20 :=
by
  sorry

end remi_water_bottle_capacity_l221_221678


namespace area_of_triangle_given_conditions_l221_221094

noncomputable def area_triangle_ABC (a b B : ℝ) : ℝ :=
  0.5 * a * b * Real.sin B

theorem area_of_triangle_given_conditions :
  area_triangle_ABC 2 (Real.sqrt 3) (Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end area_of_triangle_given_conditions_l221_221094


namespace speed_of_woman_in_still_water_l221_221335

noncomputable def V_w : ℝ := 5
variable (V_s : ℝ)

-- Conditions:
def downstream_condition : Prop := (V_w + V_s) * 6 = 54
def upstream_condition : Prop := (V_w - V_s) * 6 = 6

theorem speed_of_woman_in_still_water 
    (h1 : downstream_condition V_s) 
    (h2 : upstream_condition V_s) : 
    V_w = 5 :=
by
    -- Proof omitted
    sorry

end speed_of_woman_in_still_water_l221_221335


namespace k_value_if_divisible_l221_221773

theorem k_value_if_divisible :
  ∀ k : ℤ, (x^2 + k * x - 3) % (x - 1) = 0 → k = 2 :=
by
  intro k
  sorry

end k_value_if_divisible_l221_221773


namespace ball_bounces_before_vertex_l221_221524

def bounces_to_vertex (v h : ℕ) (units_per_bounce_vert units_per_bounce_hor : ℕ) : ℕ :=
units_per_bounce_vert * v / units_per_bounce_hor * h

theorem ball_bounces_before_vertex (verts : ℕ) (h : ℕ) (units_per_bounce_vert units_per_bounce_hor : ℕ)
    (H_vert : verts = 10)
    (H_units_vert : units_per_bounce_vert = 2)
    (H_units_hor : units_per_bounce_hor = 7) :
    bounces_to_vertex verts h units_per_bounce_vert units_per_bounce_hor = 5 := 
by
  sorry

end ball_bounces_before_vertex_l221_221524


namespace count_multiples_of_4_between_300_and_700_l221_221584

noncomputable def num_multiples_of_4_in_range (a b : ℕ) : ℕ :=
  (b - (b % 4) - (a - (a % 4) + 4)) / 4 + 1

theorem count_multiples_of_4_between_300_and_700 : 
  num_multiples_of_4_in_range 301 699 = 99 := by
  sorry

end count_multiples_of_4_between_300_and_700_l221_221584


namespace two_pow_geq_n_cubed_for_n_geq_ten_l221_221235

theorem two_pow_geq_n_cubed_for_n_geq_ten (n : ℕ) (hn : n ≥ 10) : 2^n ≥ n^3 := 
sorry

end two_pow_geq_n_cubed_for_n_geq_ten_l221_221235


namespace no_solution_for_m_l221_221812

theorem no_solution_for_m (m : ℝ) : ¬ ∃ x : ℝ, x ≠ 1 ∧ (mx - 1) / (x - 1) = 3 ↔ m = 1 ∨ m = 3 := 
by sorry

end no_solution_for_m_l221_221812


namespace determine_a_l221_221726

theorem determine_a (a : ℕ) (p1 p2 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : 2 * p1 * p2 = a) (h4 : p1 + p2 = 15) : 
  a = 52 :=
by
  sorry

end determine_a_l221_221726


namespace initial_weight_of_solution_Y_is_8_l221_221628

theorem initial_weight_of_solution_Y_is_8
  (W : ℝ)
  (hw1 : 0.25 * W = 0.20 * W + 0.4)
  (hw2 : W ≠ 0) : W = 8 :=
by
  sorry

end initial_weight_of_solution_Y_is_8_l221_221628


namespace hyperbola_eccentricity_l221_221983

theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (e : ℝ) (h3 : e = (Real.sqrt 3) / 2) 
  (h4 : a ^ 2 = b ^ 2 + (Real.sqrt 3) ^ 2) : (Real.sqrt 5) / 2 = 
    (Real.sqrt (a ^ 2 + b ^ 2)) / a :=
by
  sorry

end hyperbola_eccentricity_l221_221983


namespace firetruck_reachable_area_l221_221445

theorem firetruck_reachable_area :
  let m := 700
  let n := 31
  let area := m / n -- The area in square miles
  let time := 1 / 10 -- The available time in hours
  let speed_highway := 50 -- Speed on the highway in miles/hour
  let speed_prairie := 14 -- Speed across the prairie in miles/hour
  -- The intersection point of highways is the origin (0, 0)
  -- The firetruck can move within the reachable area
  -- There exist regions formed by the intersection points of movement directions
  m + n = 731 :=
by
  sorry

end firetruck_reachable_area_l221_221445


namespace arithmetic_expression_equality_l221_221715

theorem arithmetic_expression_equality : 
  (1/4 : ℝ) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 * (1/4096) * 8192 = 64 := 
by
  sorry

end arithmetic_expression_equality_l221_221715


namespace determine_B_l221_221889

open Set

-- Define the universal set U and the sets A and B
variable (U A B : Set ℕ)

-- Definitions based on the problem conditions
def U_def : U = A ∪ B := 
  by sorry

def cond1 : (U = {1, 2, 3, 4, 5, 6, 7}) := 
  by sorry

def cond2 : (A ∩ (U \ B) = {2, 4, 6}) := 
  by sorry

-- The main statement
theorem determine_B (h1 : U = {1, 2, 3, 4, 5, 6, 7}) (h2 : A ∩ (U \ B) = {2, 4, 6}) : B = {1, 3, 5, 7} :=
  by sorry

end determine_B_l221_221889


namespace one_cow_one_bag_l221_221405

-- Definitions based on the conditions provided.
def cows : ℕ := 45
def bags : ℕ := 45
def days : ℕ := 45

-- Problem statement: Prove that one cow will eat one bag of husk in 45 days.
theorem one_cow_one_bag (h : cows * bags = bags * days) : days = 45 :=
by
  sorry

end one_cow_one_bag_l221_221405


namespace deformable_to_triangle_l221_221936

-- Definition of the planar polygon with n sides
structure Polygon (n : ℕ) := 
  (vertices : Fin n → ℝ × ℝ) -- This is a simplified representation of a planar polygon using vertex coordinates

noncomputable def canDeformToTriangle (poly : Polygon n) : Prop := sorry

theorem deformable_to_triangle (n : ℕ) (h : n > 4) (poly : Polygon n) : canDeformToTriangle poly := 
  sorry

end deformable_to_triangle_l221_221936


namespace trees_to_plant_l221_221352

def road_length : ℕ := 156
def interval : ℕ := 6
def trees_needed (road_length interval : ℕ) := road_length / interval + 1

theorem trees_to_plant : trees_needed road_length interval = 27 := by
  sorry

end trees_to_plant_l221_221352


namespace fixed_point_range_l221_221294

theorem fixed_point_range (a : ℝ) : (∃ x : ℝ, x = x^2 + x + a) → a ≤ 0 :=
sorry

end fixed_point_range_l221_221294


namespace relay_race_team_members_l221_221431

theorem relay_race_team_members (n : ℕ) (d : ℕ) (h1 : n = 5) (h2 : d = 150) : d / n = 30 := 
by {
  -- Place the conditions here as hypotheses
  sorry
}

end relay_race_team_members_l221_221431


namespace find_pair_l221_221273

theorem find_pair (a b : ℤ) :
  (∀ x : ℝ, (a * x^4 + b * x^3 + 20 * x^2 - 12 * x + 10) = (2 * x^2 + 3 * x - 4) * (c * x^2 + d * x + e)) → 
  (a = 2) ∧ (b = 27) :=
sorry

end find_pair_l221_221273


namespace number_of_5_letter_words_number_of_5_letter_words_with_all_different_letters_number_of_5_letter_words_with_no_consecutive_repeating_letters_l221_221939

-- Define the statement about the total number of 5-letter words.
theorem number_of_5_letter_words : 26^5 = 26^5 := by
  sorry

-- Define the statement about the total number of 5-letter words with all different letters.
theorem number_of_5_letter_words_with_all_different_letters : 
  26 * 25 * 24 * 23 * 22 = 26 * 25 * 24 * 23 * 22 := by
  sorry

-- Define the statement about the total number of 5-letter words with no consecutive letters being the same.
theorem number_of_5_letter_words_with_no_consecutive_repeating_letters : 
  26 * 25 * 25 * 25 * 25 = 26 * 25 * 25 * 25 * 25 := by
  sorry

end number_of_5_letter_words_number_of_5_letter_words_with_all_different_letters_number_of_5_letter_words_with_no_consecutive_repeating_letters_l221_221939


namespace great_dane_more_than_triple_pitbull_l221_221709

variables (C P G : ℕ)
variables (h1 : G = 307) (h2 : P = 3 * C) (h3 : C + P + G = 439)

theorem great_dane_more_than_triple_pitbull
  : G - 3 * P = 10 :=
by
  sorry

end great_dane_more_than_triple_pitbull_l221_221709


namespace product_of_repeating_decimal_l221_221538

theorem product_of_repeating_decimal (x : ℚ) (h : x = 456 / 999) : 7 * x = 355 / 111 :=
by
  sorry

end product_of_repeating_decimal_l221_221538


namespace find_inverse_l221_221842

noncomputable def f (x : ℝ) := (x^7 - 1) / 5

theorem find_inverse :
  (f⁻¹ (-1 / 80) = (15 / 16)^(1 / 7)) :=
sorry

end find_inverse_l221_221842


namespace max_value_2ac_minus_abc_l221_221244

theorem max_value_2ac_minus_abc (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 7) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c <= 4) : 
  2 * a * c - a * b * c ≤ 28 :=
sorry

end max_value_2ac_minus_abc_l221_221244


namespace men_absent_is_5_l221_221760

-- Define the given conditions
def original_number_of_men : ℕ := 30
def planned_days : ℕ := 10
def actual_days : ℕ := 12

-- Prove the number of men absent (x) is 5, under given conditions
theorem men_absent_is_5 : ∃ x : ℕ, 30 * planned_days = (original_number_of_men - x) * actual_days ∧ x = 5 :=
by
  sorry

end men_absent_is_5_l221_221760


namespace problem_1_problem_2_l221_221656

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (x + 1) / Real.log 2 else 2^(-x) - 1

theorem problem_1 : f (f (-2)) = 2 := by 
  sorry

theorem problem_2 (x_0 : ℝ) (h : f x_0 < 3) : -2 < x_0 ∧ x_0 < 7 := by
  sorry

end problem_1_problem_2_l221_221656


namespace compute_expression_l221_221297

theorem compute_expression :
  45 * 72 + 28 * 45 = 4500 :=
  sorry

end compute_expression_l221_221297


namespace max_ratio_of_sequence_l221_221723

theorem max_ratio_of_sequence (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n : ℕ, S n = (n + 2) / 3 * a n) :
  ∃ n : ℕ, ∀ m : ℕ, (n = 2 → m ≠ 1) → (a n / a (n - 1)) ≤ (a m / a (m - 1)) :=
by
  sorry

end max_ratio_of_sequence_l221_221723


namespace probability_of_specific_roll_l221_221706

noncomputable def probability_event : ℚ :=
  let favorable_outcomes_first_die := 3 -- 1, 2, 3
  let total_outcomes_die := 8
  let probability_first_die := favorable_outcomes_first_die / total_outcomes_die
  
  let favorable_outcomes_second_die := 4 -- 5, 6, 7, 8
  let probability_second_die := favorable_outcomes_second_die / total_outcomes_die
  
  probability_first_die * probability_second_die

theorem probability_of_specific_roll :
  probability_event = 3 / 16 := 
  by
    sorry

end probability_of_specific_roll_l221_221706


namespace sequence_general_formula_l221_221075

theorem sequence_general_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n + 2 * n - 1) :
  ∀ n : ℕ, a n = (2 / 3) * 3^n - n :=
by
  sorry

end sequence_general_formula_l221_221075


namespace pizza_non_crust_percentage_l221_221801

theorem pizza_non_crust_percentage (total_weight crust_weight : ℕ) (h₁ : total_weight = 200) (h₂ : crust_weight = 50) :
  (total_weight - crust_weight) * 100 / total_weight = 75 :=
by
  sorry

end pizza_non_crust_percentage_l221_221801


namespace find_slope_of_line_l_l221_221992

theorem find_slope_of_line_l :
  ∃ k : ℝ, (k = 3 * Real.sqrt 5 / 10 ∨ k = -3 * Real.sqrt 5 / 10) :=
by
  -- Given conditions
  let F1 : ℝ := 6 / 5 * Real.sqrt 5
  let PF : ℝ := 4 / 5 * Real.sqrt 5
  let slope_PQ : ℝ := 1
  let slope_RF1 : ℝ := sorry  -- we need to prove/extract this from the given
  let k := 3 / 2 * slope_RF1
  -- to prove this
  sorry

end find_slope_of_line_l_l221_221992


namespace hydrogen_moles_l221_221327

-- Define the balanced chemical reaction as a relation between moles
def balanced_reaction (NaH H₂O NaOH H₂ : ℕ) : Prop :=
  NaH = NaOH ∧ H₂ = NaOH ∧ NaH = H₂

-- Given conditions
def given_conditions (NaH H₂O : ℕ) : Prop :=
  NaH = 2 ∧ H₂O = 2

-- Problem statement to prove
theorem hydrogen_moles (NaH H₂O NaOH H₂ : ℕ)
  (h₁ : balanced_reaction NaH H₂O NaOH H₂)
  (h₂ : given_conditions NaH H₂O) :
  H₂ = 2 :=
by sorry

end hydrogen_moles_l221_221327


namespace processing_time_l221_221929

theorem processing_time 
  (pictures : ℕ) (minutes_per_picture : ℕ) (minutes_per_hour : ℕ)
  (h1 : pictures = 960) (h2 : minutes_per_picture = 2) (h3 : minutes_per_hour = 60) : 
  (pictures * minutes_per_picture) / minutes_per_hour = 32 :=
by 
  sorry

end processing_time_l221_221929


namespace part1_part2_axis_of_symmetry_part2_center_of_symmetry_l221_221550

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * (Real.cos x) ^ 2, Real.sin x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)

def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem part1 (x : ℝ) (h1 : 0 < x ∧ x < π) (h2 : perpendicular (m x) (n x)) :
  x = π / 2 ∨ x = 3 * π / 4 :=
sorry

theorem part2_axis_of_symmetry (k : ℤ) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = f (2 * c - x) ∧ 
    ((2 * x + π / 4) = k * π + π / 2 → x = k * π / 2 + π / 8) :=
sorry

theorem part2_center_of_symmetry (k : ℤ) :
  ∃ x c : ℝ, f x = 1 ∧ ((2 * x + π / 4) = k * π → x = k * π / 2 - π / 8) :=
sorry

end part1_part2_axis_of_symmetry_part2_center_of_symmetry_l221_221550


namespace fg_equals_gf_l221_221866

theorem fg_equals_gf (m n p q : ℝ) (h : m + q = n + p) : ∀ x : ℝ, (m * (p * x + q) + n = p * (m * x + n) + q) :=
by sorry

end fg_equals_gf_l221_221866


namespace percent_increase_l221_221196

-- Definitions based on conditions
def initial_price : ℝ := 10
def final_price : ℝ := 15

-- Goal: Prove that the percent increase in the price per share is 50%
theorem percent_increase : ((final_price - initial_price) / initial_price) * 100 = 50 := 
by
  sorry  -- Proof is not required, so we skip it with sorry.

end percent_increase_l221_221196


namespace sin_alpha_eq_sin_beta_l221_221128

theorem sin_alpha_eq_sin_beta (α β : Real) (k : Int) 
  (h_symmetry : α + β = 2 * k * Real.pi + Real.pi) : 
  Real.sin α = Real.sin β := 
by 
  sorry

end sin_alpha_eq_sin_beta_l221_221128


namespace inequality_solution_set_correct_l221_221073

noncomputable def inequality_solution_set (a b c x : ℝ) : Prop :=
  (a > c) → (b + c > 0) → ((x - b < 0 ∧ x < c) ∨ (x > a)) → ((x - c) * (x + b) / (x - a) > 0)

theorem inequality_solution_set_correct (a b c : ℝ) :
  a > c → b + c > 0 → ∀ x, ((a > c) → (b + c > 0) → (((x - b < 0 ∧ x < c) ∨ (x > a)) → ((x - c) * (x + b) / (x - a) > 0))) :=
by
  intros h1 h2 x
  sorry

end inequality_solution_set_correct_l221_221073


namespace walnut_trees_planted_l221_221848

-- The number of walnut trees before planting
def walnut_trees_before : ℕ := 22

-- The number of walnut trees after planting
def walnut_trees_after : ℕ := 55

-- The number of walnut trees planted today
def walnut_trees_planted_today : ℕ := 33

-- Theorem statement to prove that the number of walnut trees planted today is 33
theorem walnut_trees_planted:
  walnut_trees_after - walnut_trees_before = walnut_trees_planted_today :=
by sorry

end walnut_trees_planted_l221_221848


namespace gray_region_area_l221_221797

noncomputable def area_of_gray_region (length width : ℝ) (angle_deg : ℝ) : ℝ :=
  if (length = 55 ∧ width = 44 ∧ angle_deg = 45) then 10 else 0

theorem gray_region_area :
  area_of_gray_region 55 44 45 = 10 :=
by sorry

end gray_region_area_l221_221797


namespace ab_leq_1_l221_221425

theorem ab_leq_1 {a b : ℝ} (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 2) : ab ≤ 1 :=
sorry

end ab_leq_1_l221_221425


namespace optimal_garden_dimensions_l221_221657

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), 2 * l + 2 * w = 400 ∧ l ≥ 100 ∧ w ≥ 50 ∧ l ≥ w + 20 ∧ l * w = 9600 :=
by
  sorry

end optimal_garden_dimensions_l221_221657


namespace problem_I_number_of_zeros_problem_II_inequality_l221_221298

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x * Real.exp 1 - 1

theorem problem_I_number_of_zeros : 
  ∃! (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 := 
sorry

theorem problem_II_inequality (a : ℝ) (h_a : a ≤ 0) (x : ℝ) (h_x : x ≥ 1) : 
  f x ≥ a * Real.log x - 1 := 
sorry

end problem_I_number_of_zeros_problem_II_inequality_l221_221298


namespace ratio_of_brownies_l221_221293

def total_brownies : ℕ := 15
def eaten_on_monday : ℕ := 5
def eaten_on_tuesday : ℕ := total_brownies - eaten_on_monday

theorem ratio_of_brownies : eaten_on_tuesday / eaten_on_monday = 2 := 
by
  sorry

end ratio_of_brownies_l221_221293


namespace asymptotes_of_hyperbola_l221_221141

theorem asymptotes_of_hyperbola (x y : ℝ) :
  (x ^ 2 / 4 - y ^ 2 / 9 = -1) →
  (y = (3 / 2) * x ∨ y = -(3 / 2) * x) :=
sorry

end asymptotes_of_hyperbola_l221_221141


namespace triangle_inequality_difference_l221_221048

theorem triangle_inequality_difference :
  ∀ (x : ℤ), (x + 8 > 3) → (x + 3 > 8) → (8 + 3 > x) →
  ( 10 - 6 = 4 ) :=
by sorry

end triangle_inequality_difference_l221_221048


namespace no_solution_l221_221443

theorem no_solution (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) : 
  ¬ (x^2 + y^2 + 41 = 2^n) :=
by sorry

end no_solution_l221_221443


namespace unpainted_cubes_l221_221000

theorem unpainted_cubes (n : ℕ) (cubes_per_face : ℕ) (faces : ℕ) (total_cubes : ℕ) (painted_cubes : ℕ) :
  n = 6 → cubes_per_face = 4 → faces = 6 → total_cubes = 216 → painted_cubes = 24 → 
  total_cubes - painted_cubes = 192 := by
  intros
  sorry

end unpainted_cubes_l221_221000


namespace bird_watcher_total_l221_221643

theorem bird_watcher_total
  (M : ℕ) (T : ℕ) (W : ℕ)
  (h1 : M = 70)
  (h2 : T = M / 2)
  (h3 : W = T + 8) :
  M + T + W = 148 :=
by
  -- proof omitted
  sorry

end bird_watcher_total_l221_221643


namespace sqrt_224_between_14_and_15_l221_221046

theorem sqrt_224_between_14_and_15 : 14 < Real.sqrt 224 ∧ Real.sqrt 224 < 15 := by
  sorry

end sqrt_224_between_14_and_15_l221_221046


namespace house_height_proof_l221_221433

noncomputable def height_of_house (house_shadow tree_height tree_shadow : ℕ) : ℕ :=
  house_shadow * tree_height / tree_shadow

theorem house_height_proof
  (house_shadow_length : ℕ)
  (tree_height : ℕ)
  (tree_shadow_length : ℕ)
  (expected_house_height : ℕ)
  (Hhouse_shadow_length : house_shadow_length = 56)
  (Htree_height : tree_height = 21)
  (Htree_shadow_length : tree_shadow_length = 24)
  (Hexpected_house_height : expected_house_height = 49) :
  height_of_house house_shadow_length tree_height tree_shadow_length = expected_house_height :=
by
  rw [Hhouse_shadow_length, Htree_height, Htree_shadow_length, Hexpected_house_height]
  -- Here we should compute the value and show it is equal to 49
  sorry

end house_height_proof_l221_221433


namespace largest_y_coordinate_of_graph_l221_221919

theorem largest_y_coordinate_of_graph :
  ∀ (x y : ℝ), (x^2 / 49 + (y - 3)^2 / 25 = 0) → y = 3 :=
by
  sorry

end largest_y_coordinate_of_graph_l221_221919


namespace pugs_cleaning_time_l221_221608

theorem pugs_cleaning_time : 
  (∀ (p t: ℕ), 15 * 12 = p * t ↔ 15 * 12 = 4 * 45) :=
by
  sorry

end pugs_cleaning_time_l221_221608


namespace f_monotonic_decreasing_interval_l221_221284

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x^2 - 2*x)

theorem f_monotonic_decreasing_interval : 
  ∀ x1 x2 : ℝ, 1 ≤ x1 → x1 ≤ x2 → f x2 ≤ f x1 := 
sorry

end f_monotonic_decreasing_interval_l221_221284


namespace find_x_l221_221788

variable (x y : ℚ)

-- Condition
def condition : Prop :=
  (x / (x - 2)) = ((y^3 + 3 * y - 2) / (y^3 + 3 * y - 5))

-- Assertion to prove
theorem find_x (h : condition x y) : x = ((2 * y^3 + 6 * y - 4) / 3) :=
sorry

end find_x_l221_221788


namespace number_of_boys_l221_221642

theorem number_of_boys (x g : ℕ) 
  (h1 : x + g = 150) 
  (h2 : g = (x * 150) / 100) 
  : x = 60 := 
by 
  sorry

end number_of_boys_l221_221642


namespace fruit_salad_cherries_l221_221199

theorem fruit_salad_cherries (b r g c : ℕ) 
  (h1 : b + r + g + c = 390)
  (h2 : r = 3 * b)
  (h3 : g = 2 * c)
  (h4 : c = 5 * r) :
  c = 119 :=
by
  sorry

end fruit_salad_cherries_l221_221199


namespace find_roses_last_year_l221_221167

-- Definitions based on conditions
def roses_last_year : ℕ := sorry
def roses_this_year := roses_last_year / 2
def roses_needed := 2 * roses_last_year
def rose_cost := 3 -- cost per rose in dollars
def total_spent := 54 -- total spent in dollars

-- Formulate the problem
theorem find_roses_last_year (h : 2 * roses_last_year - roses_this_year = 18)
  (cost_eq : total_spent / rose_cost = 18) :
  roses_last_year = 12 :=
by
  sorry

end find_roses_last_year_l221_221167


namespace point_on_x_axis_l221_221145

theorem point_on_x_axis (a : ℝ) (h : a + 2 = 0) : (a - 1, a + 2) = (-3, 0) :=
by
  sorry

end point_on_x_axis_l221_221145


namespace additional_weekly_rate_l221_221153

theorem additional_weekly_rate (rate_first_week : ℝ) (total_days_cost : ℝ) (days_first_week : ℕ) (total_days : ℕ) (cost_total : ℝ) (cost_first_week : ℝ) (days_after_first_week : ℕ) : 
  (rate_first_week * days_first_week = cost_first_week) → 
  (total_days = days_first_week + days_after_first_week) → 
  (cost_total = cost_first_week + (days_after_first_week * (rate_first_week * 7 / days_first_week))) →
  (rate_first_week = 18) →
  (cost_total = 350) →
  total_days = 23 → 
  (days_first_week = 7) → 
  cost_first_week = 126 →
  (days_after_first_week = 16) →
  rate_first_week * 7 / days_first_week * days_after_first_week = 14 := 
by 
  sorry

end additional_weekly_rate_l221_221153


namespace unique_last_digit_divisible_by_7_l221_221546

theorem unique_last_digit_divisible_by_7 :
  ∃! d : ℕ, (∃ n : ℕ, n % 7 = 0 ∧ n % 10 = d) :=
sorry

end unique_last_digit_divisible_by_7_l221_221546


namespace fewer_mpg_in_city_l221_221065

def city_mpg := 14
def city_distance := 336
def highway_distance := 480

def tank_size := city_distance / city_mpg
def highway_mpg := highway_distance / tank_size
def fewer_mpg := highway_mpg - city_mpg

theorem fewer_mpg_in_city : fewer_mpg = 6 := by
  sorry

end fewer_mpg_in_city_l221_221065


namespace sum_of_variables_l221_221767

theorem sum_of_variables (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) (ha : a = 2 * Real.sqrt 6) (hb : b = 3 * Real.sqrt 6) (hc : c = 6 * Real.sqrt 6) : 
  a + b + c = 11 * Real.sqrt 6 :=
by
  sorry

end sum_of_variables_l221_221767


namespace number_of_boys_l221_221343

theorem number_of_boys (n : ℕ)
    (incorrect_avg_weight : ℝ)
    (misread_weight new_weight : ℝ)
    (correct_avg_weight : ℝ)
    (h1 : incorrect_avg_weight = 58.4)
    (h2 : misread_weight = 56)
    (h3 : new_weight = 66)
    (h4 : correct_avg_weight = 58.9)
    (h5 : n * correct_avg_weight = n * incorrect_avg_weight + (new_weight - misread_weight)) :
  n = 20 := by
  sorry

end number_of_boys_l221_221343


namespace combined_total_score_l221_221295

-- Define the conditions
def num_single_answer_questions : ℕ := 50
def num_multiple_answer_questions : ℕ := 20
def single_answer_score : ℕ := 2
def multiple_answer_score : ℕ := 4
def wrong_single_penalty : ℕ := 1
def wrong_multiple_penalty : ℕ := 2
def jose_wrong_single : ℕ := 10
def jose_wrong_multiple : ℕ := 5
def jose_lost_marks : ℕ := (jose_wrong_single * wrong_single_penalty) + (jose_wrong_multiple * wrong_multiple_penalty)
def jose_correct_single : ℕ := num_single_answer_questions - jose_wrong_single
def jose_correct_multiple : ℕ := num_multiple_answer_questions - jose_wrong_multiple
def jose_single_score : ℕ := jose_correct_single * single_answer_score
def jose_multiple_score : ℕ := jose_correct_multiple * multiple_answer_score
def jose_score : ℕ := (jose_single_score + jose_multiple_score) - jose_lost_marks
def alison_score : ℕ := jose_score - 50
def meghan_score : ℕ := jose_score - 30

-- Prove the combined total score
theorem combined_total_score :
  jose_score + alison_score + meghan_score = 280 :=
by
  sorry

end combined_total_score_l221_221295


namespace range_of_x_sqrt_4_2x_l221_221211

theorem range_of_x_sqrt_4_2x (x : ℝ) : (4 - 2 * x ≥ 0) ↔ (x ≤ 2) :=
by
  sorry

end range_of_x_sqrt_4_2x_l221_221211


namespace ring_binder_price_l221_221865

theorem ring_binder_price (x : ℝ) (h1 : 50 + 5 = 55) (h2 : ∀ x, 55 + 3 * (x - 2) = 109) :
  x = 20 :=
by
  sorry

end ring_binder_price_l221_221865


namespace fixed_cost_is_50000_l221_221556

-- Definition of conditions
def fixed_cost : ℕ := 50000
def books_sold : ℕ := 10000
def revenue_per_book : ℕ := 9 - 4

-- Theorem statement: Proving that the fixed cost of making books is $50,000
theorem fixed_cost_is_50000 (F : ℕ) (h : revenue_per_book * books_sold = F) : 
  F = fixed_cost :=
by sorry

end fixed_cost_is_50000_l221_221556


namespace yellow_beads_needed_l221_221985

variable (Total green yellow : ℕ)

theorem yellow_beads_needed (h_green : green = 4) (h_yellow : yellow = 0) (h_fraction : (4 / 5 : ℚ) = 4 / (green + yellow + 16)) :
    4 + 16 + green = Total := by
  sorry

end yellow_beads_needed_l221_221985


namespace convex_quadrilateral_max_two_obtuse_l221_221296

theorem convex_quadrilateral_max_two_obtuse (a b c d : ℝ)
  (h1 : a + b + c + d = 360)
  (h2 : a < 180) (h3 : b < 180) (h4 : c < 180) (h5 : d < 180)
  : (∃ A1 A2, a = A1 ∧ b = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ c < 90 ∧ d < 90) ∨
    (∃ A1 A2, a = A1 ∧ c = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ b < 90 ∧ d < 90) ∨
    (∃ A1 A2, a = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ b < 90 ∧ c < 90) ∨
    (∃ A1 A2, b = A1 ∧ c = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ d < 90) ∨
    (∃ A1 A2, b = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ c < 90) ∨
    (∃ A1 A2, c = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ b < 90) ∨
    (¬∃ x y z, (x > 90) ∧ (y > 90) ∧ (z > 90) ∧ x + y + z ≤ 360) := sorry

end convex_quadrilateral_max_two_obtuse_l221_221296


namespace ratio_of_sides_l221_221504

theorem ratio_of_sides 
  (a b c d : ℝ) 
  (h1 : (a * b) / (c * d) = 0.16) 
  (h2 : b / d = 2 / 5) : 
  a / c = 0.4 := 
by 
  sorry

end ratio_of_sides_l221_221504


namespace calculate_expression_l221_221956

theorem calculate_expression :
  ((1 / 3 : ℝ) ^ (-2 : ℝ)) + Real.tan (Real.pi / 4) - Real.sqrt ((-10 : ℝ) ^ 2) = 0 := by
  sorry

end calculate_expression_l221_221956


namespace median_isosceles_right_triangle_leg_length_l221_221650

theorem median_isosceles_right_triangle_leg_length (m : ℝ) (h : ℝ) (x : ℝ)
  (H1 : m = 15)
  (H2 : m = h / 2)
  (H3 : 2 * x * x = h * h) : x = 15 * Real.sqrt 2 :=
by
  sorry

end median_isosceles_right_triangle_leg_length_l221_221650


namespace eq_exponents_l221_221554

theorem eq_exponents (m n : ℤ) : ((5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n) → (m = 0 ∧ n = 0) :=
by
  sorry

end eq_exponents_l221_221554


namespace y_intercept_of_line_l221_221736

theorem y_intercept_of_line (m : ℝ) (x₀ : ℝ) (y₀ : ℝ) (h_slope : m = -3) (h_intercept : (x₀, y₀) = (7, 0)) : (0, 21) = (0, (y₀ - m * x₀)) :=
by
  sorry

end y_intercept_of_line_l221_221736


namespace band_row_lengths_l221_221292

theorem band_row_lengths (n : ℕ) (h1 : n = 108) (h2 : ∃ k, 10 ≤ k ∧ k ≤ 18 ∧ 108 % k = 0) : 
  (∃ count : ℕ, count = 2) :=
by 
  sorry

end band_row_lengths_l221_221292


namespace elvins_first_month_bill_l221_221357

variable (F C : ℕ)

def total_bill_first_month := F + C
def total_bill_second_month := F + 2 * C

theorem elvins_first_month_bill :
  total_bill_first_month F C = 46 ∧
  total_bill_second_month F C = 76 ∧
  total_bill_second_month F C - total_bill_first_month F C = 30 →
  total_bill_first_month F C = 46 :=
by
  intro h
  sorry

end elvins_first_month_bill_l221_221357


namespace part1_part2_l221_221516

variable {a b c : ℚ}

theorem part1 (ha : a < 0) : (a / |a|) = -1 :=
sorry

theorem part2 (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  min (a * b / |a * b| + |b * c| / (b * c) + a * c / |a * c| + |a * b * c| / (a * b * c)) (-2) = -2 :=
sorry

end part1_part2_l221_221516


namespace successful_pair_exists_another_with_same_arithmetic_mean_l221_221246

theorem successful_pair_exists_another_with_same_arithmetic_mean
  (a b : ℕ)
  (h_distinct : a ≠ b)
  (h_arith_mean_nat : ∃ m : ℕ, 2 * m = a + b)
  (h_geom_mean_nat : ∃ g : ℕ, g * g = a * b) :
  ∃ (c d : ℕ), c ≠ d ∧ ∃ m' : ℕ, 2 * m' = c + d ∧ ∃ g' : ℕ, g' * g' = c * d ∧ m' = (a + b) / 2 :=
sorry

end successful_pair_exists_another_with_same_arithmetic_mean_l221_221246


namespace sarah_speed_for_rest_of_trip_l221_221636

def initial_speed : ℝ := 15  -- miles per hour
def initial_time : ℝ := 1  -- hour
def total_distance : ℝ := 45  -- miles
def extra_time_if_same_speed : ℝ := 1  -- hour (late)
def arrival_early_time : ℝ := 0.5  -- hour (early)

theorem sarah_speed_for_rest_of_trip (remaining_distance remaining_time : ℝ) :
  remaining_distance = total_distance - initial_speed * initial_time →
  remaining_time = (remaining_distance / initial_speed - extra_time_if_same_speed) + arrival_early_time →
  remaining_distance / remaining_time = 20 :=
by
  intros h1 h2
  sorry

end sarah_speed_for_rest_of_trip_l221_221636


namespace f_decreasing_increasing_find_b_range_l221_221700

-- Define the function f(x) and prove its properties for x > 0 and x < 0
noncomputable def f (x a : ℝ) : ℝ := x + a / x

theorem f_decreasing_increasing (a : ℝ) (h : a > 0):
  (∀ x : ℝ, 0 < x → x ≤ Real.sqrt a → ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 ≤ Real.sqrt a) → f x1 a > f x2 a) ∧ 
  (∀ x : ℝ, 0 < Real.sqrt a → Real.sqrt a ≤ x → ∀ x1 x2 : ℝ, (Real.sqrt a ≤ x1 ∧ x1 < x2) → f x1 a < f x2 a) ∧ 
  (∀ x : ℝ, x < 0 → -Real.sqrt a ≤ x ∧ x < 0 → f x1 a > f x2 a) ∧ 
  (∀ x : ℝ, x < 0 → x < -Real.sqrt a → f x1 a < f x2 a)
:= sorry

-- Define the function h(x) and find the range of b
noncomputable def h (x : ℝ) : ℝ := x + 4 / x - 8
noncomputable def g (x b : ℝ) : ℝ := -x - 2 * b

theorem find_b_range:
  (∀ x1 : ℝ, 1 ≤ x1 ∧ x1 ≤ 3 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 3 ∧ g x2 b = h x1) ↔
  1/2 ≤ b ∧ b ≤ 1
:= sorry

end f_decreasing_increasing_find_b_range_l221_221700


namespace sum_of_sequence_l221_221460

noncomputable def sequence_sum (a : ℝ) (n : ℕ) : ℝ :=
if a = 1 then sorry else (5 * (1 - a ^ n) / (1 - a) ^ 2) - (4 + (5 * n - 4) * a ^ n) / (1 - a)

theorem sum_of_sequence (S : ℕ → ℝ) (a : ℝ) (h1 : S 1 = 1)
                       (h2 : ∀ n, S (n + 1) - S n = (5 * n + 1) * a ^ n) (h3 : |a| ≠ 1) :
  ∀ n, S n = sequence_sum a n :=
  sorry

end sum_of_sequence_l221_221460


namespace cost_ratio_two_pastries_pies_l221_221124

theorem cost_ratio_two_pastries_pies (s p : ℝ) (h1 : 2 * s = 3 * (2 * p)) :
  (s + p) / (2 * p) = 2 :=
by
  sorry

end cost_ratio_two_pastries_pies_l221_221124


namespace tangent_sum_problem_l221_221708

theorem tangent_sum_problem
  (α β : ℝ)
  (h_eq_root : ∃ (x y : ℝ), (x = Real.tan α) ∧ (y = Real.tan β) ∧ (6*x^2 - 5*x + 1 = 0) ∧ (6*y^2 - 5*y + 1 = 0))
  (h_range_α : 0 < α ∧ α < π/2)
  (h_range_β : π < β ∧ β < 3*π/2) :
  (Real.tan (α + β) = 1) ∧ (α + β = 5*π/4) := 
sorry

end tangent_sum_problem_l221_221708


namespace range_of_alpha_minus_beta_l221_221346

theorem range_of_alpha_minus_beta (α β : Real) (h₁ : -180 < α) (h₂ : α < β) (h₃ : β < 180) :
  -360 < α - β ∧ α - β < 0 :=
by
  sorry

end range_of_alpha_minus_beta_l221_221346


namespace solve_inequality_system_l221_221747

theorem solve_inequality_system (x : ℝ) (h1 : 2 * x + 1 < 5) (h2 : 2 - x ≤ 1) : 1 ≤ x ∧ x < 2 :=
by
  sorry

end solve_inequality_system_l221_221747


namespace acid_solution_l221_221860

theorem acid_solution (x y : ℝ) (h1 : 0.3 * x + 0.1 * y = 90)
  (h2 : x + y = 600) : x = 150 ∧ y = 450 :=
by
  sorry

end acid_solution_l221_221860


namespace alpha_plus_2beta_l221_221005

noncomputable def sin_square (θ : ℝ) := (Real.sin θ)^2
noncomputable def sin_double (θ : ℝ) := Real.sin (2 * θ)

theorem alpha_plus_2beta (α β : ℝ) (hα : 0 < α ∧ α < Real.pi / 2) 
(hβ : 0 < β ∧ β < Real.pi / 2) 
(h1 : 3 * sin_square α + 2 * sin_square β = 1)
(h2 : 3 * sin_double α - 2 * sin_double β = 0) : 
α + 2 * β = 5 * Real.pi / 6 :=
by
  sorry

end alpha_plus_2beta_l221_221005


namespace original_average_l221_221373

theorem original_average (A : ℝ)
  (h : 2 * A = 160) : A = 80 :=
by sorry

end original_average_l221_221373


namespace minimal_overlap_facebook_instagram_l221_221086

variable (P : ℝ → Prop)
variable [Nonempty (Set.Icc 0 1)]

theorem minimal_overlap_facebook_instagram :
  ∀ (f i : ℝ), f = 0.85 → i = 0.75 → ∃ b : ℝ, 0 ≤ b ∧ b ≤ 1 ∧ b = 0.6 :=
by
  intros
  sorry

end minimal_overlap_facebook_instagram_l221_221086


namespace range_of_function_l221_221737

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem range_of_function : Set.Icc (-2 : ℝ) 7 = Set.image f (Set.Icc (-3 : ℝ) 2) :=
by
  sorry

end range_of_function_l221_221737


namespace proof_problem_l221_221871

noncomputable def otimes (a b : ℝ) : ℝ := a^3 / b^2

theorem proof_problem : ((otimes (otimes 2 3) 4) - otimes 2 (otimes 3 4)) = -224/81 :=
by
  sorry

end proof_problem_l221_221871


namespace max_regions_7_dots_l221_221022

-- Definitions based on conditions provided.
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def R (n : ℕ) : ℕ := 1 + binom n 2 + binom n 4

-- The goal is to state the proposition that the maximum number of regions created by joining 7 dots on a circle is 57.
theorem max_regions_7_dots : R 7 = 57 :=
by
  -- The proof is to be filled in here
  sorry

end max_regions_7_dots_l221_221022


namespace number_of_pounds_of_vegetables_l221_221271

-- Defining the conditions
def beef_cost_per_pound : ℕ := 6  -- Beef costs $6 per pound
def vegetable_cost_per_pound : ℕ := 2  -- Vegetables cost $2 per pound
def beef_pounds : ℕ := 4  -- Troy buys 4 pounds of beef
def total_cost : ℕ := 36  -- The total cost of everything is $36

-- Prove the number of pounds of vegetables Troy buys is 6
theorem number_of_pounds_of_vegetables (V : ℕ) :
  beef_cost_per_pound * beef_pounds + vegetable_cost_per_pound * V = total_cost → V = 6 :=
by
  sorry  -- Proof to be filled in later

end number_of_pounds_of_vegetables_l221_221271


namespace p_minus_q_eq_16_sqrt_2_l221_221041

theorem p_minus_q_eq_16_sqrt_2 (p q : ℝ) (h_eq : ∀ x : ℝ, (x - 4) * (x + 4) = 28 * x - 84 → x = p ∨ x = q)
  (h_distinct : p ≠ q) (h_p_gt_q : p > q) : p - q = 16 * Real.sqrt 2 :=
sorry

end p_minus_q_eq_16_sqrt_2_l221_221041


namespace find_n_tangent_l221_221139

theorem find_n_tangent (n : ℤ) (h1 : -180 < n) (h2 : n < 180) (h3 : ∃ k : ℤ, 210 = n + 180 * k) : n = 30 :=
by
  -- Proof steps would go here
  sorry

end find_n_tangent_l221_221139


namespace nonagon_arithmetic_mean_property_l221_221765

def is_equilateral_triangle (A : Fin 9 → ℤ) (i j k : Fin 9) : Prop :=
  (j = (i + 3) % 9) ∧ (k = (i + 6) % 9)

def is_arithmetic_mean (A : Fin 9 → ℤ) (i j k : Fin 9) : Prop :=
  A j = (A i + A k) / 2

theorem nonagon_arithmetic_mean_property :
  ∀ (A : Fin 9 → ℤ),
    (∀ i, A i = 2016 + i) →
    (∀ i j k : Fin 9, is_equilateral_triangle A i j k → is_arithmetic_mean A i j k) :=
by
  intros
  sorry

end nonagon_arithmetic_mean_property_l221_221765


namespace solve_quadratic_eq_l221_221845

theorem solve_quadratic_eq (x : ℝ) : 4 * x^2 - (x^2 - 2 * x + 1) = 0 ↔ x = 1 / 3 ∨ x = -1 := by
  sorry

end solve_quadratic_eq_l221_221845


namespace larger_number_is_23_l221_221259

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l221_221259


namespace percentage_greater_than_l221_221192

-- Definitions of the variables involved
variables (X Y Z : ℝ)

-- Lean statement to prove the formula
theorem percentage_greater_than (X Y Z : ℝ) : 
  (100 * (X - Y)) / (Y + Z) = (100 * (X - Y)) / (Y + Z) :=
by
  -- skipping the actual proof
  sorry

end percentage_greater_than_l221_221192


namespace only_zero_sol_l221_221411

theorem only_zero_sol (x y z t : ℤ) : x^2 + y^2 + z^2 + t^2 = 2 * x * y * z * t → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 :=
by
  sorry

end only_zero_sol_l221_221411


namespace total_stoppage_time_l221_221547

theorem total_stoppage_time (stop1 stop2 stop3 : ℕ) (h1 : stop1 = 5)
  (h2 : stop2 = 8) (h3 : stop3 = 10) : stop1 + stop2 + stop3 = 23 :=
sorry

end total_stoppage_time_l221_221547


namespace intersection_point_exists_circle_equation_standard_form_l221_221442

noncomputable def line1 (x y : ℝ) : Prop := 2 * x + y = 0
noncomputable def line2 (x y : ℝ) : Prop := x + y = 2
noncomputable def line3 (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0

theorem intersection_point_exists :
  ∃ (C : ℝ × ℝ), (line1 C.1 C.2 ∧ line2 C.1 C.2) ∧ C = (-2, 4) :=
sorry

theorem circle_equation_standard_form :
  ∃ (center : ℝ × ℝ) (radius : ℝ), center = (-2, 4) ∧ radius = 3 ∧
  ∀ x y : ℝ, ((x + 2) ^ 2 + (y - 4) ^ 2 = 9) :=
sorry

end intersection_point_exists_circle_equation_standard_form_l221_221442


namespace factorize_expression_l221_221205

variable {R : Type} [Ring R]
variables (a b x y : R)

theorem factorize_expression :
  8 * a * x - b * y + 4 * a * y - 2 * b * x = (4 * a - b) * (2 * x + y) :=
sorry

end factorize_expression_l221_221205


namespace sales_quota_50_l221_221978

theorem sales_quota_50 :
  let cars_sold_first_three_days := 5 * 3
  let cars_sold_next_four_days := 3 * 4
  let additional_cars_needed := 23
  let total_quota := cars_sold_first_three_days + cars_sold_next_four_days + additional_cars_needed
  total_quota = 50 :=
by
  -- proof goes here
  sorry

end sales_quota_50_l221_221978


namespace polynomial_integer_root_l221_221834

theorem polynomial_integer_root (b : ℤ) :
  (∃ x : ℤ, x^3 + 5 * x^2 + b * x + 9 = 0) ↔ b = -127 ∨ b = -74 ∨ b = -27 ∨ b = -24 ∨ b = -15 ∨ b = -13 :=
by
  sorry

end polynomial_integer_root_l221_221834


namespace percentage_increase_in_overtime_rate_l221_221799

def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def total_compensation : ℝ := 976
def total_hours_worked : ℝ := 52

theorem percentage_increase_in_overtime_rate :
  ((total_compensation - (regular_rate * regular_hours)) / (total_hours_worked - regular_hours) - regular_rate) / regular_rate * 100 = 75 :=
by
  sorry

end percentage_increase_in_overtime_rate_l221_221799


namespace intersection_eq_l221_221688

-- Define Set A based on the given condition
def setA : Set ℝ := {x | 1 < (3:ℝ)^x ∧ (3:ℝ)^x ≤ 9}

-- Define Set B based on the given condition
def setB : Set ℝ := {x | (x + 2) / (x - 1) ≤ 0}

-- Define the intersection of Set A and Set B
def intersection : Set ℝ := {x | x > 0 ∧ x < 1}

-- Prove that the intersection of setA and setB equals (0, 1)
theorem intersection_eq : {x | x > 0 ∧ x < 1} = {x | x ∈ setA ∧ x ∈ setB} :=
by
  sorry

end intersection_eq_l221_221688


namespace probability_correct_l221_221015

-- Define the set of segment lengths
def segment_lengths : List ℕ := [1, 3, 5, 7, 9]

-- Define the triangle inequality condition
def forms_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Calculate the number of favorable outcomes, i.e., sets that can form a triangle
def favorable_sets : List (ℕ × ℕ × ℕ) :=
  [(3, 5, 7), (3, 7, 9), (5, 7, 9)]

-- Define the total number of ways to select three segments out of five
def total_combinations : ℕ :=
  10

-- Define the number of favorable sets
def number_of_favorable_sets : ℕ :=
  favorable_sets.length

-- Calculate the probability of selecting three segments that form a triangle
def probability_of_triangle : ℚ :=
  number_of_favorable_sets / total_combinations

-- The theorem to prove
theorem probability_correct : probability_of_triangle = 3 / 10 :=
  by {
    -- Placeholder for the proof
    sorry
  }

end probability_correct_l221_221015


namespace kendalls_total_distance_l221_221537

-- Definitions of the conditions
def distance_with_mother : ℝ := 0.17
def distance_with_father : ℝ := 0.5

-- The theorem to prove the total distance
theorem kendalls_total_distance : distance_with_mother + distance_with_father = 0.67 :=
by
  sorry

end kendalls_total_distance_l221_221537


namespace number_of_nurses_l221_221674

variables (D N : ℕ)

-- Condition: The total number of doctors and nurses is 250
def total_staff := D + N = 250

-- Condition: The ratio of doctors to nurses is 2 to 3
def ratio_doctors_to_nurses := D = (2 * N) / 3

-- Proof: The number of nurses is 150
theorem number_of_nurses (h1 : total_staff D N) (h2 : ratio_doctors_to_nurses D N) : N = 150 :=
sorry

end number_of_nurses_l221_221674


namespace janet_initial_stickers_l221_221268

variable (x : ℕ)

theorem janet_initial_stickers (h : x + 53 = 56) : x = 3 := by
  sorry

end janet_initial_stickers_l221_221268


namespace total_number_of_fish_l221_221571

theorem total_number_of_fish :
  let goldfish := 8
  let angelfish := goldfish + 4
  let guppies := 2 * angelfish
  let tetras := goldfish - 3
  let bettas := tetras + 5
  goldfish + angelfish + guppies + tetras + bettas = 59 := by
  -- Provide the proof here.
  sorry

end total_number_of_fish_l221_221571


namespace product_increase_l221_221601

variable (x : ℤ)

theorem product_increase (h : 53 * x = 1585) : 1585 - (35 * x) = 535 :=
by sorry

end product_increase_l221_221601


namespace max_value_3x_4y_l221_221429

noncomputable def y_geom_mean (x y : ℝ) : Prop :=
  y^2 = (1 - x) * (1 + x)

theorem max_value_3x_4y (x y : ℝ) (h : y_geom_mean x y) : 3 * x + 4 * y ≤ 5 :=
sorry

end max_value_3x_4y_l221_221429


namespace temperature_on_tuesday_l221_221012

variable (T W Th F : ℝ)

-- Conditions
axiom H1 : (T + W + Th) / 3 = 42
axiom H2 : (W + Th + F) / 3 = 44
axiom H3 : F = 43

-- Proof statement
theorem temperature_on_tuesday : T = 37 :=
by
  -- This would be the place to fill in the proof using H1, H2, and H3
  sorry

end temperature_on_tuesday_l221_221012


namespace find_greater_number_l221_221308

theorem find_greater_number (a b : ℕ) (h1 : a * b = 4107) (h2 : Nat.gcd a b = 37) (h3 : a > b) : a = 111 :=
sorry

end find_greater_number_l221_221308


namespace range_of_B_l221_221802

theorem range_of_B (A : ℝ × ℝ) (hA : A = (1, 2)) (h : 2 * A.1 - B * A.2 + 3 ≥ 0) : B ≤ 2.5 :=
by sorry

end range_of_B_l221_221802


namespace expand_binomials_l221_221162

variable {x y : ℝ}

theorem expand_binomials (x y : ℝ) : 
  (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := 
by
  sorry

end expand_binomials_l221_221162


namespace triangle_construction_l221_221400

-- Define the problem statement in Lean
theorem triangle_construction (a b c : ℝ) :
  correct_sequence = [3, 1, 4, 2] :=
sorry

end triangle_construction_l221_221400


namespace factorial_expression_l221_221197

open Nat

theorem factorial_expression : ((sqrt (5! * 4!)) ^ 2 + 3!) = 2886 := by
  sorry

end factorial_expression_l221_221197


namespace number_of_roses_ian_kept_l221_221421

-- Definitions representing the conditions
def initial_roses : ℕ := 20
def roses_to_mother : ℕ := 6
def roses_to_grandmother : ℕ := 9
def roses_to_sister : ℕ := 4

-- The theorem statement we want to prove
theorem number_of_roses_ian_kept : (initial_roses - (roses_to_mother + roses_to_grandmother + roses_to_sister) = 1) :=
by
  sorry

end number_of_roses_ian_kept_l221_221421


namespace chantel_final_bracelets_l221_221896

-- Definitions of the conditions in Lean
def initial_bracelets_7_days := 7 * 4
def after_school_giveaway := initial_bracelets_7_days - 8
def bracelets_10_days := 10 * 5
def total_after_10_days := after_school_giveaway + bracelets_10_days
def after_soccer_giveaway := total_after_10_days - 12
def crafting_club_bracelets := 4 * 6
def total_after_crafting_club := after_soccer_giveaway + crafting_club_bracelets
def weekend_trip_bracelets := 2 * 3
def total_after_weekend_trip := total_after_crafting_club + weekend_trip_bracelets
def final_total := total_after_weekend_trip - 10

-- Lean statement to prove the final total bracelets
theorem chantel_final_bracelets : final_total = 78 :=
by
  -- Note: The proof is not required, hence the sorry
  sorry

end chantel_final_bracelets_l221_221896


namespace divides_necklaces_l221_221925

/-- Define the number of ways to make an even number of necklaces each of length at least 3. -/
def D_0 (n : ℕ) : ℕ := sorry

/-- Define the number of ways to make an odd number of necklaces each of length at least 3. -/
def D_1 (n : ℕ) : ℕ := sorry

/-- Main theorem: Prove that (n - 1) divides (D_1(n) - D_0(n)) for n ≥ 2 -/
theorem divides_necklaces (n : ℕ) (h : n ≥ 2) : (n - 1) ∣ (D_1 n - D_0 n) := sorry

end divides_necklaces_l221_221925


namespace orthocenter_of_triangle_l221_221248

theorem orthocenter_of_triangle (A : ℝ × ℝ) (x y : ℝ) 
  (h₁ : x + y = 0) (h₂ : 2 * x - 3 * y + 1 = 0) : 
  A = (1, 2) → (x, y) = (-1 / 5, 1 / 5) :=
by
  sorry

end orthocenter_of_triangle_l221_221248


namespace remainder_when_divided_by_x_minus_4_l221_221044

noncomputable def f (x : ℝ) : ℝ := x^4 - 9 * x^3 + 21 * x^2 + x - 18

theorem remainder_when_divided_by_x_minus_4 : f 4 = 2 :=
by
  sorry

end remainder_when_divided_by_x_minus_4_l221_221044


namespace bug_visits_tiles_l221_221478

theorem bug_visits_tiles (width length : ℕ) (gcd_width_length : ℕ) (broken_tile : ℕ × ℕ)
  (h_width : width = 12) (h_length : length = 25) (h_gcd : gcd_width_length = Nat.gcd width length)
  (h_broken_tile : broken_tile = (12, 18)) :
  width + length - gcd_width_length = 36 := by
  sorry

end bug_visits_tiles_l221_221478


namespace part1_solution_set_part2_range_a_l221_221422

noncomputable def inequality1 (a x : ℝ) : Prop :=
|a * x - 2| + |a * x - a| ≥ 2

theorem part1_solution_set : 
  (∀ x : ℝ, inequality1 1 x ↔ x ≥ 2.5 ∨ x ≤ 0.5) := 
sorry

theorem part2_range_a :
  (∀ x : ℝ, inequality1 a x) ↔ a ≥ 4 :=
sorry

end part1_solution_set_part2_range_a_l221_221422


namespace find_number_l221_221612

theorem find_number (x : ℝ) (h : x / 0.025 = 40) : x = 1 := 
by sorry

end find_number_l221_221612


namespace sum_of_reciprocals_l221_221366

-- We state that for all non-zero real numbers x and y, if x + y = xy,
-- then the sum of their reciprocals equals 1.
theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) :
  1/x + 1/y = 1 :=
by
  sorry

end sum_of_reciprocals_l221_221366


namespace temperature_43_l221_221409

theorem temperature_43 (T W Th F : ℝ)
  (h1 : (T + W + Th) / 3 = 42)
  (h2 : (W + Th + F) / 3 = 44)
  (h3 : T = 37) : F = 43 :=
by
  sorry

end temperature_43_l221_221409


namespace find_unit_prices_minimize_total_cost_l221_221847

def unit_prices_ (x y : ℕ) :=
  x + 2 * y = 40 ∧ 2 * x + 3 * y = 70
  
theorem find_unit_prices (x y: ℕ) (h: unit_prices_ x y): x = 20 ∧ y = 10 := 
  sorry

def total_cost (m: ℕ) := 20 * m + 10 * (60 - m)

theorem minimize_total_cost (m : ℕ) (h1 : 60 ≥ m) (h2 : m ≥ 20) : 
  total_cost m = 800 → m = 20 :=
  sorry

end find_unit_prices_minimize_total_cost_l221_221847


namespace gcd_of_q_and_r_l221_221436

theorem gcd_of_q_and_r (p q r : ℕ) (hpq : p > 0) (hqr : q > 0) (hpr : r > 0)
    (gcd_pq : Nat.gcd p q = 240) (gcd_pr : Nat.gcd p r = 540) : Nat.gcd q r = 60 := by
  sorry

end gcd_of_q_and_r_l221_221436


namespace remainder_of_exponentiated_sum_modulo_seven_l221_221803

theorem remainder_of_exponentiated_sum_modulo_seven :
  (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end remainder_of_exponentiated_sum_modulo_seven_l221_221803


namespace unique_line_intercept_l221_221544

noncomputable def is_positive_integer (n : ℕ) : Prop := n > 0
noncomputable def is_prime (n : ℕ) : Prop := n = 2 ∨ (n > 2 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem unique_line_intercept (a b : ℕ) :
  ((is_positive_integer a) ∧ (is_prime b) ∧ (6 * b + 5 * a = a * b)) ↔ (a = 11 ∧ b = 11) :=
by
  sorry

end unique_line_intercept_l221_221544


namespace remainder_when_sum_divided_by_40_l221_221622

theorem remainder_when_sum_divided_by_40 (x y : ℤ) 
  (h1 : x % 80 = 75) 
  (h2 : y % 120 = 115) : 
  (x + y) % 40 = 30 := 
  sorry

end remainder_when_sum_divided_by_40_l221_221622


namespace no_two_or_more_consecutive_sum_30_l221_221857

theorem no_two_or_more_consecutive_sum_30 :
  ∀ (a n : ℕ), n ≥ 2 → (n * (2 * a + n - 1) = 60) → false :=
by
  intro a n hn h
  sorry

end no_two_or_more_consecutive_sum_30_l221_221857


namespace solve_for_y_l221_221963

theorem solve_for_y (y : ℝ) (h : 9 / (y^2) = y / 81) : y = 9 :=
by
  sorry

end solve_for_y_l221_221963


namespace acute_triangle_angles_l221_221001

theorem acute_triangle_angles (α β γ : ℕ) (h1 : α ≥ β) (h2 : β ≥ γ) (h3 : α = 5 * γ) (h4 : α + β + γ = 180) :
  (α = 85 ∧ β = 78 ∧ γ = 17) :=
sorry

end acute_triangle_angles_l221_221001


namespace at_most_two_zero_points_l221_221518

noncomputable def f (x a : ℝ) := x^3 - 12 * x + a

theorem at_most_two_zero_points (a : ℝ) (h : a ≥ 16) : ∃ l u : ℝ, (∀ x : ℝ, f x a = 0 → x < l ∨ l ≤ x ∧ x ≤ u ∨ u < x) := sorry

end at_most_two_zero_points_l221_221518


namespace find_h_l221_221040

noncomputable def y1 (x h j : ℝ) := 4 * (x - h) ^ 2 + j
noncomputable def y2 (x h k : ℝ) := 3 * (x - h) ^ 2 + k

theorem find_h (h j k : ℝ)
  (C1 : y1 0 h j = 2024)
  (C2 : y2 0 h k = 2025)
  (H1 : y1 x h j = 0 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ * x₂ = 506)
  (H2 : y2 x h k = 0 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ * x₂ = 675) :
  h = 22.5 :=
sorry

end find_h_l221_221040


namespace drone_height_l221_221250

theorem drone_height (r s h : ℝ) 
  (h_distance_RS : r^2 + s^2 = 160^2)
  (h_DR : h^2 + r^2 = 170^2) 
  (h_DS : h^2 + s^2 = 150^2) : 
  h = 30 * Real.sqrt 43 :=
by 
  sorry

end drone_height_l221_221250


namespace solution_count_l221_221646

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem solution_count (a : ℝ) : 
  (∃ x : ℝ, f x = a) ↔ 
  ((a > 2 ∨ a < -2 ∧ ∃! x₁, f x₁ = a) ∨ 
   ((a = 2 ∨ a = -2) ∧ ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = a ∧ f x₂ = a) ∨ 
   (-2 < a ∧ a < 2 ∧ ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = a ∧ f x₂ = a ∧ f x₃ = a)) := 
by sorry

end solution_count_l221_221646


namespace ted_candy_bars_l221_221240

theorem ted_candy_bars (b : ℕ) (n : ℕ) (h : b = 5) (h2 : n = 3) : b * n = 15 :=
by
  sorry

end ted_candy_bars_l221_221240


namespace range_of_m_l221_221230

def P (x : ℝ) : Prop := |(4 - x) / 3| ≤ 2
def q (x m : ℝ) : Prop := (x + m - 1) * (x - m - 1) ≤ 0

theorem range_of_m (m : ℝ) (h : m > 0) : (∀ x, ¬P x → ¬q x m) → m ≥ 9 :=
by
  intros
  sorry

end range_of_m_l221_221230


namespace coles_average_speed_l221_221722

theorem coles_average_speed (t_work : ℝ) (t_round : ℝ) (s_return : ℝ) (t_return : ℝ) (d : ℝ) (t_work_min : ℕ) :
  t_work_min = 72 ∧ t_round = 2 ∧ s_return = 90 ∧ 
  t_work = t_work_min / 60 ∧ t_return = t_round - t_work ∧ d = s_return * t_return →
  d / t_work = 60 := 
by
  intro h
  sorry

end coles_average_speed_l221_221722


namespace total_cars_all_own_l221_221127

theorem total_cars_all_own :
  ∀ (C L S K : ℕ), 
  (C = 5) →
  (L = C + 4) →
  (K = 2 * C) →
  (S = K - 2) →
  (C + L + K + S = 32) :=
by
  intros C L S K
  intro hC
  intro hL
  intro hK
  intro hS
  sorry

end total_cars_all_own_l221_221127


namespace fraction_operations_l221_221091

theorem fraction_operations :
  let a := 1 / 3
  let b := 1 / 4
  let c := 1 / 2
  (a + b = 7 / 12) ∧ ((7 / 12) / c = 7 / 6) := by
{
  sorry
}

end fraction_operations_l221_221091


namespace bounded_harmonic_is_constant_l221_221243

noncomputable def is_harmonic (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ (x y : ℤ), f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1) = 4 * f (x, y)

theorem bounded_harmonic_is_constant (f : ℤ × ℤ → ℝ) (M : ℝ) 
  (h_bound : ∀ (x y : ℤ), |f (x, y)| ≤ M)
  (h_harmonic : is_harmonic f) :
  ∃ c : ℝ, ∀ x y : ℤ, f (x, y) = c :=
sorry

end bounded_harmonic_is_constant_l221_221243


namespace algebraic_expression_zero_iff_x_eq_2_l221_221173

theorem algebraic_expression_zero_iff_x_eq_2 (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (1 / (x - 1) + 3 / (1 - x^2) = 0) ↔ (x = 2) :=
by
  sorry

end algebraic_expression_zero_iff_x_eq_2_l221_221173


namespace correct_result_without_mistake_l221_221267

variable {R : Type*} [CommRing R] (a b c : R)
variable (A : R)

theorem correct_result_without_mistake :
  A + 2 * (ab + 2 * bc - 4 * ac) = (3 * ab - 2 * ac + 5 * bc) → 
  A - 2 * (ab + 2 * bc - 4 * ac) = -ab + 14 * ac - 3 * bc :=
by
  sorry

end correct_result_without_mistake_l221_221267


namespace exists_difference_divisible_by_11_l221_221947

theorem exists_difference_divisible_by_11 (a : Fin 12 → ℤ) :
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
  sorry

end exists_difference_divisible_by_11_l221_221947


namespace SamBalloonsCount_l221_221341

-- Define the conditions
def FredBalloons : ℕ := 10
def DanBalloons : ℕ := 16
def TotalBalloons : ℕ := 72

-- Define the function to calculate Sam's balloons and the main theorem to prove
def SamBalloons := TotalBalloons - (FredBalloons + DanBalloons)

theorem SamBalloonsCount : SamBalloons = 46 := by
  -- The proof is omitted here
  sorry

end SamBalloonsCount_l221_221341


namespace eventB_is_not_random_l221_221741

def eventA := "The sun rises in the east and it rains in the west"
def eventB := "It's not cold when it snows but cold when it melts"
def eventC := "It rains continuously during the Qingming festival"
def eventD := "It's sunny every day when the plums turn yellow"

def is_random_event (event : String) : Prop :=
  event = eventA ∨ event = eventC ∨ event = eventD

theorem eventB_is_not_random : ¬ is_random_event eventB :=
by
  unfold is_random_event
  sorry

end eventB_is_not_random_l221_221741


namespace competition_total_races_l221_221998

theorem competition_total_races (sprinters : ℕ) (sprinters_with_bye : ℕ) (lanes_preliminary : ℕ) (lanes_subsequent : ℕ) 
  (eliminated_per_race : ℕ) (first_round_advance : ℕ) (second_round_advance : ℕ) (third_round_advance : ℕ) 
  : sprinters = 300 → sprinters_with_bye = 16 → lanes_preliminary = 8 → lanes_subsequent = 6 → 
    eliminated_per_race = 7 → first_round_advance = 36 → second_round_advance = 9 → third_round_advance = 2 
    → first_round_races = 36 → second_round_races = 9 → third_round_races = 2 → final_race = 1
    → first_round_races + second_round_races + third_round_races + final_race = 48 :=
by 
  intros sprinters_eq sprinters_with_bye_eq lanes_preliminary_eq lanes_subsequent_eq eliminated_per_race_eq 
         first_round_advance_eq second_round_advance_eq third_round_advance_eq 
         first_round_races_eq second_round_races_eq third_round_races_eq final_race_eq
  sorry

end competition_total_races_l221_221998


namespace mixed_water_temp_l221_221123

def cold_water_temp : ℝ := 20   -- Temperature of cold water
def hot_water_temp : ℝ := 40    -- Temperature of hot water

theorem mixed_water_temp :
  (cold_water_temp + hot_water_temp) / 2 = 30 := 
by sorry

end mixed_water_temp_l221_221123


namespace solve_for_x_l221_221541

theorem solve_for_x (x : ℝ) :
    (1 / 3 * ((x + 8) + (7 * x + 3) + (3 * x + 9)) = 5 * x - 10) → x = 12.5 :=
by
  intro h
  sorry

end solve_for_x_l221_221541


namespace neg_p_l221_221670

variable {x : ℝ}

def p := ∀ x > 0, Real.sin x ≤ 1

theorem neg_p : ¬ p ↔ ∃ x > 0, Real.sin x > 1 :=
by
  sorry

end neg_p_l221_221670


namespace simplify_and_evaluate_l221_221827

-- Problem statement with conditions translated into Lean
theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 5 + 1) :
  (a / (a^2 - 2*a + 1)) / (1 + 1 / (a - 1)) = Real.sqrt 5 / 5 := sorry

end simplify_and_evaluate_l221_221827


namespace mike_practice_hours_l221_221318

def weekday_practice_hours_per_day : ℕ := 3
def days_per_weekday_practice : ℕ := 5
def saturday_practice_hours : ℕ := 5
def weeks_until_game : ℕ := 3

def total_weekday_practice_hours : ℕ := weekday_practice_hours_per_day * days_per_weekday_practice
def total_weekly_practice_hours : ℕ := total_weekday_practice_hours + saturday_practice_hours
def total_practice_hours : ℕ := total_weekly_practice_hours * weeks_until_game

theorem mike_practice_hours :
  total_practice_hours = 60 := by
  sorry

end mike_practice_hours_l221_221318


namespace angle_sum_around_point_l221_221036

theorem angle_sum_around_point (y : ℝ) (h : 170 + y + y = 360) : y = 95 := 
sorry

end angle_sum_around_point_l221_221036


namespace no_nontrivial_sum_periodic_functions_l221_221052

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

def is_nontrivial_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := 
  periodic f p ∧ ∃ x y, x ≠ y ∧ f x ≠ f y

theorem no_nontrivial_sum_periodic_functions (g h : ℝ → ℝ) :
  is_nontrivial_periodic_function g 1 →
  is_nontrivial_periodic_function h π →
  ¬ ∃ T > 0, ∀ x, (g + h) (x + T) = (g + h) x :=
sorry

end no_nontrivial_sum_periodic_functions_l221_221052


namespace tshirt_cost_l221_221125

-- Definitions based on conditions
def pants_cost : ℝ := 80
def shoes_cost : ℝ := 150
def discount : ℝ := 0.1
def total_paid : ℝ := 558

-- Variables based on the problem
variable (T : ℝ) -- Cost of one T-shirt
def num_tshirts : ℝ := 4
def num_pants : ℝ := 3
def num_shoes : ℝ := 2

-- Theorem: The cost of one T-shirt is $20
theorem tshirt_cost : T = 20 :=
by
  have total_cost : ℝ := (num_tshirts * T) + (num_pants * pants_cost) + (num_shoes * shoes_cost)
  have discounted_total : ℝ := (1 - discount) * total_cost
  have payment_condition : discounted_total = total_paid := sorry
  sorry -- detailed proof

end tshirt_cost_l221_221125


namespace race_winner_and_liar_l221_221962

def Alyosha_statement (pos : ℕ → Prop) : Prop := ¬ pos 1 ∧ ¬ pos 4
def Borya_statement (pos : ℕ → Prop) : Prop := ¬ pos 4
def Vanya_statement (pos : ℕ → Prop) : Prop := pos 1
def Grisha_statement (pos : ℕ → Prop) : Prop := pos 4

def three_true_one_false (s1 s2 s3 s4 : Prop) : Prop := 
  (s1 ∧ s2 ∧ s3 ∧ ¬ s4) ∨
  (s1 ∧ s2 ∧ ¬ s3 ∧ s4) ∨
  (s1 ∧ ¬ s2 ∧ s3 ∧ s4) ∨
  (¬ s1 ∧ s2 ∧ s3 ∧ s4)

def race_result (pos : ℕ → Prop) : Prop :=
  Vanya_statement pos ∧
  three_true_one_false (Alyosha_statement pos) (Borya_statement pos) (Vanya_statement pos) (Grisha_statement pos) ∧
  Borya_statement pos = false

theorem race_winner_and_liar:
  ∃ (pos : ℕ → Prop), race_result pos :=
sorry

end race_winner_and_liar_l221_221962


namespace problem1_problem2_l221_221824

-- Problem 1: Prove that the minimum value of f(x) is at least m for all x ∈ ℝ when k = 0
theorem problem1 (f : ℝ → ℝ) (m : ℝ) (h : ∀ x : ℝ, f x = Real.exp x - x) : m ≤ 1 := 
sorry

-- Problem 2: Prove that there exists exactly one zero of f(x) in the interval (k, 2k) when k > 1
theorem problem2 (f : ℝ → ℝ) (k : ℝ) (hk : k > 1) (h : ∀ x : ℝ, f x = Real.exp (x - k) - x) :
  ∃! (x : ℝ), x ∈ Set.Ioo k (2 * k) ∧ f x = 0 := 
sorry

end problem1_problem2_l221_221824


namespace baker_cakes_total_l221_221702

-- Conditions
def initial_cakes : ℕ := 121
def cakes_sold : ℕ := 105
def cakes_bought : ℕ := 170

-- Proof Problem
theorem baker_cakes_total :
  initial_cakes - cakes_sold + cakes_bought = 186 :=
by
  sorry

end baker_cakes_total_l221_221702


namespace eggs_for_dinner_l221_221155

-- Definitions of the conditions
def eggs_for_breakfast := 2
def eggs_for_lunch := 3
def total_eggs := 6

-- The quantity of eggs for dinner needs to be proved
theorem eggs_for_dinner :
  ∃ x : ℕ, x + eggs_for_breakfast + eggs_for_lunch = total_eggs ∧ x = 1 :=
by
  sorry

end eggs_for_dinner_l221_221155


namespace extended_pattern_ratio_l221_221783

noncomputable def original_black_tiles : ℕ := 12
noncomputable def original_white_tiles : ℕ := 24
noncomputable def original_total_tiles : ℕ := 36
noncomputable def extended_total_tiles : ℕ := 64
noncomputable def border_black_tiles : ℕ := 24 /- The new border adds 24 black tiles -/
noncomputable def extended_black_tiles : ℕ := 36
noncomputable def extended_white_tiles := original_white_tiles

theorem extended_pattern_ratio :
  (extended_black_tiles : ℚ) / extended_white_tiles = 3 / 2 :=
by
  sorry

end extended_pattern_ratio_l221_221783


namespace exists_real_A_l221_221265

theorem exists_real_A (t : ℝ) (n : ℕ) (h_root: t^2 - 10 * t + 1 = 0) :
  ∃ A : ℝ, (A = t) ∧ ∀ n : ℕ, ∃ k : ℕ, A^n + 1/(A^n) - k^2 = 2 :=
by
  sorry

end exists_real_A_l221_221265


namespace min_abs_diff_is_11_l221_221324

noncomputable def min_abs_diff (k l : ℕ) : ℤ := abs (36^k - 5^l)

theorem min_abs_diff_is_11 :
  ∃ k l : ℕ, min_abs_diff k l = 11 :=
by
  sorry

end min_abs_diff_is_11_l221_221324


namespace average_speed_round_trip_l221_221037

noncomputable def distance_AB : ℝ := 120
noncomputable def speed_AB : ℝ := 30
noncomputable def speed_BA : ℝ := 40

theorem average_speed_round_trip :
  (2 * distance_AB * speed_AB * speed_BA) / (distance_AB * (speed_AB + speed_BA)) = 34 := 
  by 
    sorry

end average_speed_round_trip_l221_221037


namespace find_fraction_l221_221564

def number : ℕ := 16

theorem find_fraction (f : ℚ) : f * number + 5 = 13 → f = 1 / 2 :=
by
  sorry

end find_fraction_l221_221564


namespace selling_price_l221_221355

theorem selling_price (cost_price : ℕ) (profit_percent : ℕ) (selling_price : ℕ) : 
  cost_price = 2400 ∧ profit_percent = 6 → selling_price = 2544 := by
  sorry

end selling_price_l221_221355


namespace satisfies_differential_equation_l221_221637

noncomputable def y (x : ℝ) : ℝ := (Real.sin x) / x

theorem satisfies_differential_equation (x : ℝ) (hx : x ≠ 0) : 
  x * (deriv (fun x => (Real.sin x) / x) x) + (Real.sin x) / x = Real.cos x := 
by
  -- the proof goes here
  sorry

end satisfies_differential_equation_l221_221637


namespace number_of_books_to_break_even_is_4074_l221_221143

-- Definitions from problem conditions
def fixed_costs : ℝ := 35630
def variable_cost_per_book : ℝ := 11.50
def selling_price_per_book : ℝ := 20.25

-- The target number of books to sell for break-even
def target_books_to_break_even : ℕ := 4074

-- Lean statement to prove that number of books to break even is 4074
theorem number_of_books_to_break_even_is_4074 :
  let total_costs (x : ℝ) := fixed_costs + variable_cost_per_book * x
  let total_revenue (x : ℝ) := selling_price_per_book * x
  ∃ x : ℝ, total_costs x = total_revenue x → x = target_books_to_break_even := by
  sorry

end number_of_books_to_break_even_is_4074_l221_221143


namespace senior_junior_ratio_l221_221272

variable (S J : ℕ) (k : ℕ)

theorem senior_junior_ratio (h1 : S = k * J) 
                           (h2 : (1/8 : ℚ) * S + (3/4 : ℚ) * J = (1/3 : ℚ) * (S + J)) : 
                           k = 2 :=
by
  sorry

end senior_junior_ratio_l221_221272


namespace Natasha_speed_over_limit_l221_221055

theorem Natasha_speed_over_limit (d : ℕ) (t : ℕ) (speed_limit : ℕ) 
    (h1 : d = 60) 
    (h2 : t = 1) 
    (h3 : speed_limit = 50) : (d / t - speed_limit = 10) :=
by
  -- Because d = 60, t = 1, and speed_limit = 50, we need to prove (60 / 1 - 50) = 10
  sorry

end Natasha_speed_over_limit_l221_221055


namespace total_books_is_10033_l221_221950

variable (P C B M H : ℕ)
variable (x : ℕ) (h_P : P = 3 * x) (h_C : C = 2 * x)
variable (h_B : B = (3 / 2) * x)
variable (h_M : M = (3 / 5) * x)
variable (h_H : H = (4 / 5) * x)
variable (total_books : ℕ)
variable (h_total : total_books = P + C + B + M + H)
variable (h_bound : total_books > 10000)

theorem total_books_is_10033 : total_books = 10033 :=
  sorry

end total_books_is_10033_l221_221950


namespace intersection_is_2_to_inf_l221_221821

-- Define the set A
def setA (x : ℝ) : Prop :=
 x > 1

-- Define the set B
def setB (y : ℝ) : Prop :=
 ∃ x : ℝ, y = Real.sqrt (x^2 + 2*x + 5)

-- Define the intersection of A and B
def setIntersection : Set ℝ :=
{ y | setA y ∧ setB y }

-- Statement to prove the intersection
theorem intersection_is_2_to_inf : setIntersection = { y | y ≥ 2 } :=
sorry -- Proof is omitted

end intersection_is_2_to_inf_l221_221821


namespace difference_of_averages_l221_221058

theorem difference_of_averages :
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 70 + 16) / 3
  avg1 - avg2 = 8 :=
by
  sorry

end difference_of_averages_l221_221058


namespace dean_marathon_time_l221_221633

/-- 
Micah runs 2/3 times as fast as Dean, and it takes Jake 1/3 times more time to finish the marathon
than it takes Micah. The total time the three take to complete the marathon is 23 hours.
Prove that the time it takes Dean to finish the marathon is approximately 7.67 hours.
-/
theorem dean_marathon_time (D M J : ℝ)
  (h1 : M = D * (3 / 2))
  (h2 : J = M + (1 / 3) * M)
  (h3 : D + M + J = 23) : 
  D = 23 / 3 :=
by
  sorry

end dean_marathon_time_l221_221633


namespace min_fraction_value_l221_221183

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 2

theorem min_fraction_value : ∀ x ∈ (Set.Ici (7 / 4)), (f x)^2 + 2 / (f x) ≥ 81 / 28 :=
by
  sorry

end min_fraction_value_l221_221183


namespace simplify_sqrt_sum_l221_221477

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_sum_l221_221477


namespace spadesuit_evaluation_l221_221691

def spadesuit (a b : ℤ) : ℤ := Int.natAbs (a - b)

theorem spadesuit_evaluation :
  spadesuit 5 (spadesuit 3 9) = 1 := 
by 
  sorry

end spadesuit_evaluation_l221_221691


namespace slope_parallel_l221_221565

theorem slope_parallel (x y : ℝ) (m : ℝ) : (3:ℝ) * x - (6:ℝ) * y = (9:ℝ) → m = (1:ℝ) / (2:ℝ) :=
by
  sorry

end slope_parallel_l221_221565


namespace distance_is_twenty_cm_l221_221307

noncomputable def distance_between_pictures_and_board (picture_width: ℕ) (board_width_m: ℕ) (board_width_cm: ℕ) (number_of_pictures: ℕ) : ℕ :=
  let board_total_width := board_width_m * 100 + board_width_cm
  let total_pictures_width := number_of_pictures * picture_width
  let total_distance := board_total_width - total_pictures_width
  let total_gaps := number_of_pictures + 1
  total_distance / total_gaps

theorem distance_is_twenty_cm :
  distance_between_pictures_and_board 30 3 20 6 = 20 :=
by
  sorry

end distance_is_twenty_cm_l221_221307


namespace find_cool_triple_x_eq_5_find_cool_triple_x_eq_7_two_distinct_cool_triples_for_odd_x_find_cool_triple_x_even_l221_221704

-- Define the nature of a "cool" triple.
def is_cool_triple (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 1 ∧ z > 0 ∧ x^2 - 3 * y^2 = z^2 - 3

-- Part (a) i: For x = 5.
theorem find_cool_triple_x_eq_5 : ∃ (y z : ℕ), is_cool_triple 5 y z := sorry

-- Part (a) ii: For x = 7.
theorem find_cool_triple_x_eq_7 : ∃ (y z : ℕ), is_cool_triple 7 y z := sorry

-- Part (b): For every x ≥ 5 and odd, there are at least two distinct cool triples.
theorem two_distinct_cool_triples_for_odd_x (x : ℕ) (h1 : x ≥ 5) (h2 : x % 2 = 1) : 
  ∃ (y₁ z₁ y₂ z₂ : ℕ), is_cool_triple x y₁ z₁ ∧ is_cool_triple x y₂ z₂ ∧ (y₁, z₁) ≠ (y₂, z₂) := sorry

-- Part (c): Find a cool type triple with x even.
theorem find_cool_triple_x_even : ∃ (x y z : ℕ), x % 2 = 0 ∧ is_cool_triple x y z := sorry

end find_cool_triple_x_eq_5_find_cool_triple_x_eq_7_two_distinct_cool_triples_for_odd_x_find_cool_triple_x_even_l221_221704


namespace initial_volume_of_mixture_l221_221435

/-- A mixture contains 10% water. 
5 liters of water should be added to this so that the water becomes 20% in the new mixture.
Prove that the initial volume of the mixture is 40 liters. -/
theorem initial_volume_of_mixture 
  (V : ℚ) -- Define the initial volume of the mixture
  (h1 : 0.10 * V + 5 = 0.20 * (V + 5)) -- Condition on the mixture
  : V = 40 := -- The statement to prove
by
  sorry -- Proof not required

end initial_volume_of_mixture_l221_221435


namespace slope_of_parallel_line_l221_221104

theorem slope_of_parallel_line (a b c : ℝ) (h : a = 3 ∧ b = -6 ∧ c = 12) :
  ∃ m : ℝ, (∀ (x y : ℝ), 3 * x - 6 * y = 12 → y = m * x - 2) ∧ m = 1/2 := 
sorry

end slope_of_parallel_line_l221_221104


namespace length_of_wall_l221_221570

theorem length_of_wall (side_mirror length_wall width_wall : ℕ) 
  (mirror_area wall_area : ℕ) (H1 : side_mirror = 54) 
  (H2 : mirror_area = side_mirror * side_mirror) 
  (H3 : wall_area = 2 * mirror_area) 
  (H4 : width_wall = 68) 
  (H5 : wall_area = length_wall * width_wall) : 
  length_wall = 86 :=
by
  sorry

end length_of_wall_l221_221570


namespace number_of_students_surveyed_l221_221615

noncomputable def M : ℕ := 60
noncomputable def N : ℕ := 90
noncomputable def B : ℕ := M / 3

theorem number_of_students_surveyed : M + B + N = 170 := by
  rw [M, N, B]
  norm_num
  sorry

end number_of_students_surveyed_l221_221615


namespace shaded_area_T_shape_l221_221724

theorem shaded_area_T_shape (a b c d e: ℕ) (square_side_length rect_length rect_width: ℕ)
  (h_side_lengths: ∀ x, x = 2 ∨ x = 4) (h_square: square_side_length = 6) 
  (h_rect_dim: rect_length = 4 ∧ rect_width = 2)
  (h_areas: [a, b, c, d, e] = [4, 4, 4, 8, 4]) :
  a + b + d + e = 20 :=
by
  sorry

end shaded_area_T_shape_l221_221724


namespace negation_of_proposition_l221_221018

variable (a b : ℝ)

theorem negation_of_proposition :
  (¬ (a * b = 0 → a = 0 ∨ b = 0)) ↔ (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :=
by
  sorry

end negation_of_proposition_l221_221018


namespace triangle_side_calculation_l221_221943

theorem triangle_side_calculation
  (a : ℝ) (A B : ℝ)
  (ha : a = 3)
  (hA : A = 30)
  (hB : B = 15) :
  let C := 180 - A - B
  let c := a * (Real.sin C) / (Real.sin A)
  c = 3 * Real.sqrt 2 := by
  sorry

end triangle_side_calculation_l221_221943


namespace price_of_first_variety_of_oil_l221_221107

theorem price_of_first_variety_of_oil 
  (P : ℕ) 
  (x : ℕ) 
  (cost_second_variety : ℕ) 
  (volume_second_variety : ℕ)
  (cost_mixture_per_liter : ℕ) 
  : x = 160 ∧ cost_second_variety = 60 ∧ volume_second_variety = 240 ∧ cost_mixture_per_liter = 52 → P = 40 :=
by
  sorry

end price_of_first_variety_of_oil_l221_221107


namespace total_number_of_flags_is_12_l221_221719

def number_of_flags : Nat :=
  3 * 2 * 2

theorem total_number_of_flags_is_12 : number_of_flags = 12 := by
  sorry

end total_number_of_flags_is_12_l221_221719


namespace tank_full_capacity_l221_221479

theorem tank_full_capacity (w c : ℕ) (h1 : w = c / 6) (h2 : w + 4 = c / 3) : c = 12 :=
sorry

end tank_full_capacity_l221_221479


namespace find_a_and_b_l221_221203

theorem find_a_and_b (a b : ℚ) :
  ((∃ x y : ℚ, 3 * x - y = 7 ∧ a * x + y = b) ∧
   (∃ x y : ℚ, x + b * y = a ∧ 2 * x + y = 8)) →
  a = -7/5 ∧ b = -11/5 :=
by sorry

end find_a_and_b_l221_221203


namespace remainder_division_l221_221572

/-- A number when divided by a certain divisor left a remainder, 
when twice the number was divided by the same divisor, the remainder was 112. 
The divisor is 398.
Prove that the remainder when the original number is divided by the divisor is 56. -/
theorem remainder_division (N R : ℤ) (D : ℕ) (Q Q' : ℤ)
  (hD : D = 398)
  (h1 : N = D * Q + R)
  (h2 : 2 * N = D * Q' + 112) :
  R = 56 :=
sorry

end remainder_division_l221_221572


namespace rancher_problem_l221_221635

theorem rancher_problem (s c : ℕ) (h : 30 * s + 35 * c = 1500) : (s = 1 ∧ c = 42) ∨ (s = 36 ∧ c = 12) := 
by
  sorry

end rancher_problem_l221_221635


namespace rhombus_diagonal_length_l221_221363

theorem rhombus_diagonal_length
  (area : ℝ) (d2 : ℝ) (d1 : ℝ)
  (h_area : area = 432) 
  (h_d2 : d2 = 24) :
  d1 = 36 :=
by
  sorry

end rhombus_diagonal_length_l221_221363


namespace matrix_inverse_problem_l221_221108

theorem matrix_inverse_problem
  (x y z w : ℚ)
  (h1 : 2 * x + 3 * w = 1)
  (h2 : x * z = 15)
  (h3 : 4 * w = -8)
  (h4 : 4 * z = 5 * y) :
  x * y * z * w = -102.857 := by
    sorry

end matrix_inverse_problem_l221_221108


namespace find_value_of_expression_l221_221517

variables (a b : ℝ)

-- Given the condition that 2a - 3b = 5, prove that 2a - 3b + 3 = 8.
theorem find_value_of_expression
  (h : 2 * a - 3 * b = 5) : 2 * a - 3 * b + 3 = 8 :=
by sorry

end find_value_of_expression_l221_221517


namespace calc_fraction_l221_221011
-- Import necessary libraries

-- Define the necessary fractions and the given expression
def expr := (5 / 6) * (1 / (7 / 8 - 3 / 4))

-- State the theorem
theorem calc_fraction : expr = 20 / 3 := 
by
  sorry

end calc_fraction_l221_221011


namespace number_of_racks_l221_221023

theorem number_of_racks (cds_per_rack total_cds : ℕ) (h1 : cds_per_rack = 8) (h2 : total_cds = 32) :
  total_cds / cds_per_rack = 4 :=
by
  -- actual proof goes here
  sorry

end number_of_racks_l221_221023


namespace solve_for_x_l221_221415

variable (a b c d x : ℝ)

theorem solve_for_x (h1 : a ≠ b) (h2 : b ≠ 0) 
  (h3 : d ≠ c) (h4 : c % x = 0) (h5 : d % x = 0) 
  (h6 : (2*a + x) / (3*b + x) = c / d) : 
  x = (3*b*c - 2*a*d) / (d - c) := 
sorry

end solve_for_x_l221_221415


namespace stadium_breadth_l221_221882

theorem stadium_breadth (P L B : ℕ) (h1 : P = 800) (h2 : L = 100) :
  2 * (L + B) = P → B = 300 :=
by
  sorry

end stadium_breadth_l221_221882


namespace stormi_cars_washed_l221_221555

-- Definitions based on conditions
def cars_earning := 10
def lawns_number := 2
def lawn_earning := 13
def bicycle_cost := 80
def needed_amount := 24

-- Auxiliary calculations
def lawns_total_earning := lawns_number * lawn_earning
def already_earning := bicycle_cost - needed_amount
def cars_total_earning := already_earning - lawns_total_earning

-- Main problem statement
theorem stormi_cars_washed : (cars_total_earning / cars_earning) = 3 :=
  by sorry

end stormi_cars_washed_l221_221555


namespace student_ticket_cost_l221_221776

theorem student_ticket_cost :
  ∀ (S : ℤ),
  (525 - 388) * S + 388 * 6 = 2876 → S = 4 :=
by
  sorry

end student_ticket_cost_l221_221776


namespace max_boxes_in_warehouse_l221_221563

def warehouse_length : ℕ := 50
def warehouse_width : ℕ := 30
def warehouse_height : ℕ := 5
def box_edge_length : ℕ := 2

theorem max_boxes_in_warehouse : (warehouse_length / box_edge_length) * (warehouse_width / box_edge_length) * (warehouse_height / box_edge_length) = 750 := 
by
  sorry

end max_boxes_in_warehouse_l221_221563


namespace probability_of_rolling_number_less_than_5_is_correct_l221_221621

noncomputable def probability_of_rolling_number_less_than_5 : ℚ :=
  let total_outcomes := 8
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_of_rolling_number_less_than_5_is_correct :
  probability_of_rolling_number_less_than_5 = 1 / 2 := by
  sorry

end probability_of_rolling_number_less_than_5_is_correct_l221_221621


namespace length_of_platform_l221_221997

theorem length_of_platform {train_length platform_crossing_time signal_pole_crossing_time : ℚ}
  (h_train_length : train_length = 300)
  (h_platform_crossing_time : platform_crossing_time = 40)
  (h_signal_pole_crossing_time : signal_pole_crossing_time = 18) :
  ∃ L : ℚ, L = 1100 / 3 :=
by
  sorry

end length_of_platform_l221_221997


namespace find_n_l221_221412

theorem find_n
  (n : ℕ)
  (h1 : 2287 % n = r)
  (h2 : 2028 % n = r)
  (h3 : 1806 % n = r)
  (h_r_non_zero : r ≠ 0) : 
  n = 37 :=
by
  sorry

end find_n_l221_221412


namespace values_of_m_l221_221690

def A : Set ℝ := { -1, 2 }
def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

theorem values_of_m (m : ℝ) : (A ∪ B m = A) ↔ (m = -1/2 ∨ m = 0 ∨ m = 1) := by
  sorry

end values_of_m_l221_221690


namespace ratio_of_ages_l221_221148

theorem ratio_of_ages (S M : ℕ) (h1 : M = S + 24) (h2 : M + 2 = (S + 2) * 2) (h3 : S = 22) : (M + 2) / (S + 2) = 2 := 
by {
  sorry
}

end ratio_of_ages_l221_221148


namespace slope_angle_at_point_l221_221669

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4 * x + 8

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := 3 * x^2 - 4

-- State the problem: Prove the slope angle at point (1, 5) is 135 degrees
theorem slope_angle_at_point (θ : ℝ) (h : θ = 135) :
    f' 1 = -1 := 
by 
    sorry

end slope_angle_at_point_l221_221669


namespace min_expression_value_l221_221332

theorem min_expression_value (x y z : ℝ) : ∃ x y z : ℝ, (xy - z)^2 + (x + y + z)^2 = 0 :=
by
  sorry

end min_expression_value_l221_221332


namespace interest_rate_is_20_percent_l221_221675

theorem interest_rate_is_20_percent (P A : ℝ) (t : ℝ) (r : ℝ) 
  (h1 : P = 500) (h2 : A = 1000) (h3 : t = 5) :
  A = P * (1 + r * t) → r = 0.20 :=
by
  intro h
  sorry

end interest_rate_is_20_percent_l221_221675


namespace first_place_beats_joe_by_two_points_l221_221380

def points (wins draws : ℕ) : ℕ := 3 * wins + draws

theorem first_place_beats_joe_by_two_points
  (joe_wins joe_draws first_place_wins first_place_draws : ℕ)
  (h1 : joe_wins = 1)
  (h2 : joe_draws = 3)
  (h3 : first_place_wins = 2)
  (h4 : first_place_draws = 2) :
  points first_place_wins first_place_draws - points joe_wins joe_draws = 2 := by
  sorry

end first_place_beats_joe_by_two_points_l221_221380


namespace slices_with_both_toppings_l221_221394

-- Definitions and conditions directly from the problem statement
def total_slices : ℕ := 24
def pepperoni_slices : ℕ := 15
def mushroom_slices : ℕ := 14

-- Theorem proving the number of slices with both toppings
theorem slices_with_both_toppings :
  (∃ n : ℕ, n + (pepperoni_slices - n) + (mushroom_slices - n) = total_slices) → ∃ n : ℕ, n = 5 := 
by 
  sorry

end slices_with_both_toppings_l221_221394


namespace initial_number_of_mice_l221_221927

theorem initial_number_of_mice (x : ℕ) 
  (h1 : x % 2 = 0)
  (h2 : (x / 2) % 3 = 0)
  (h3 : (x / 2 - x / 6) % 4 = 0)
  (h4 : (x / 2 - x / 6 - (x / 2 - x / 6) / 4) % 5 = 0)
  (h5 : (x / 5) = (x / 6) + 2) : 
  x = 60 := 
by sorry

end initial_number_of_mice_l221_221927


namespace calculate_expression_l221_221013

theorem calculate_expression (h₁ : x = 7 / 8) (h₂ : y = 5 / 6) (hx : x ≠ 0) (hy : y ≠ 0) :
  (4 * x - 6 * y) / (60 * x * y) = -6 / 175 := 
sorry

end calculate_expression_l221_221013


namespace find_floor_of_apt_l221_221475

-- Define the conditions:
-- Number of stories
def num_stories : Nat := 9
-- Number of entrances
def num_entrances : Nat := 10
-- Total apartments in entrance 10
def apt_num : Nat := 333
-- Number of apartments per floor in each entrance (which is to be found)
def apts_per_floor_per_entrance : Nat := 4 -- from solution b)

-- Assertion: The floor number that apartment number 333 is on in entrance 10
theorem find_floor_of_apt (num_stories num_entrances apt_num apts_per_floor_per_entrance : ℕ) :
  1 ≤ apt_num ∧ apt_num ≤ num_stories * num_entrances * apts_per_floor_per_entrance →
  (apt_num - 1) / apts_per_floor_per_entrance + 1 = 3 :=
by
  sorry

end find_floor_of_apt_l221_221475


namespace find_m_find_min_value_l221_221258

-- Conditions
def A (m : ℤ) : Set ℝ := { x | abs (x + 1) + abs (x - m) < 5 }

-- First Problem: Prove m = 3 given 3 ∈ A
theorem find_m (m : ℤ) (h : 3 ∈ A m) : m = 3 := sorry

-- Second Problem: Prove a^2 + b^2 + c^2 ≥ 1 given a + 2b + 2c = 3
theorem find_min_value (a b c : ℝ) (h : a + 2 * b + 2 * c = 3) : (a^2 + b^2 + c^2) ≥ 1 := sorry

end find_m_find_min_value_l221_221258


namespace doubled_base_and_exponent_l221_221916

theorem doubled_base_and_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h : (2 * a) ^ (2 * b) = a ^ b * x ^ 3) : 
  x = (4 ^ b * a ^ b) ^ (1 / 3) :=
by
  sorry

end doubled_base_and_exponent_l221_221916


namespace discount_percent_l221_221142

theorem discount_percent (MP CP SP : ℝ)
  (h1 : CP = 0.64 * MP)
  (h2 : (SP - CP) / CP * 100 = 34.375) :
  ((MP - SP) / MP * 100) = 14 :=
by
  -- Proof would go here
  sorry

end discount_percent_l221_221142


namespace factorize_x_squared_minus_121_l221_221239

theorem factorize_x_squared_minus_121 (x : ℝ) : (x^2 - 121) = (x + 11) * (x - 11) :=
by
  sorry

end factorize_x_squared_minus_121_l221_221239


namespace equal_charge_at_250_l221_221777

/-- Define the monthly fee for Plan A --/
def planA_fee (x : ℕ) : ℝ :=
  0.4 * x + 50

/-- Define the monthly fee for Plan B --/
def planB_fee (x : ℕ) : ℝ :=
  0.6 * x

/-- Prove that the charges for Plan A and Plan B are equal when the call duration is 250 minutes --/
theorem equal_charge_at_250 : planA_fee 250 = planB_fee 250 :=
by
  sorry

end equal_charge_at_250_l221_221777


namespace juice_expense_l221_221194

theorem juice_expense (M P : ℕ) 
  (h1 : M + P = 17) 
  (h2 : 5 * M + 6 * P = 94) : 6 * P = 54 :=
by 
  sorry

end juice_expense_l221_221194


namespace number_of_three_digit_prime_integers_l221_221169

def prime_digits : Set Nat := {2, 3, 5, 7}

theorem number_of_three_digit_prime_integers : 
  (∃ count, count = 4 * 4 * 4 ∧ count = 64) :=
by
  sorry

end number_of_three_digit_prime_integers_l221_221169


namespace real_roots_exist_l221_221099

theorem real_roots_exist (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) :=
by
  sorry  -- Proof goes here

end real_roots_exist_l221_221099


namespace hyperbola_problem_l221_221430

noncomputable def is_hyperbola (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) - ((y - 2)^2 / b^2) = 1

variables (s : ℝ)

theorem hyperbola_problem
  (h₁ : is_hyperbola 0 5 a b)
  (h₂ : is_hyperbola (-1) 6 a b)
  (h₃ : is_hyperbola s 3 a b)
  (hb : b^2 = 9)
  (ha : a^2 = 9 / 25) :
  s^2 = 2 / 5 :=
sorry

end hyperbola_problem_l221_221430


namespace nested_expression_rational_count_l221_221116

theorem nested_expression_rational_count : 
  let count := Nat.card {n : ℕ // 1 ≤ n ∧ n ≤ 2021 ∧ ∃ m : ℕ, m % 2 = 1 ∧ m * m = 1 + 4 * n}
  count = 44 := 
by sorry

end nested_expression_rational_count_l221_221116


namespace tan_alpha_sub_60_l221_221370

theorem tan_alpha_sub_60 
  (alpha : ℝ) 
  (h : Real.tan alpha = 4 * Real.sin (420 * Real.pi / 180)) : 
  Real.tan (alpha - 60 * Real.pi / 180) = (Real.sqrt 3) / 7 :=
by sorry

end tan_alpha_sub_60_l221_221370


namespace min_value_f_range_of_a_l221_221823

-- Define the function f(x) with parameter a.
def f (x a : ℝ) := |x + a| + |x - a|

-- (Ⅰ) Statement: Prove that for a = 1, the minimum value of f(x) is 2.
theorem min_value_f (x : ℝ) : f x 1 ≥ 2 :=
  by sorry

-- (Ⅱ) Statement: Prove that if f(2) > 5, then the range of values for a is (-∞, -5/2) ∪ (5/2, +∞).
theorem range_of_a (a : ℝ) : f 2 a > 5 → a < -5 / 2 ∨ a > 5 / 2 :=
  by sorry

end min_value_f_range_of_a_l221_221823


namespace inequality_with_sum_one_l221_221165

theorem inequality_with_sum_one
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1)
  (x y : ℝ) :
  (a * x + b * y) * (b * x + a * y) ≥ x * y :=
by
  sorry

end inequality_with_sum_one_l221_221165


namespace subtraction_of_decimals_l221_221481

theorem subtraction_of_decimals : (3.75 - 0.48) = 3.27 :=
by
  sorry

end subtraction_of_decimals_l221_221481


namespace volume_of_TABC_l221_221539

noncomputable def volume_pyramid_TABC : ℝ :=
  let TA : ℝ := 15
  let TB : ℝ := 15
  let TC : ℝ := 5 * Real.sqrt 3
  let area_ABT : ℝ := (1 / 2) * TA * TB
  (1 / 3) * area_ABT * TC

theorem volume_of_TABC :
  volume_pyramid_TABC = 187.5 * Real.sqrt 3 :=
sorry

end volume_of_TABC_l221_221539


namespace find_a_l221_221417

theorem find_a (a : ℝ) : (1 : ℝ)^2 + 1 + 2 * a = 0 → a = -1 := 
by 
  sorry

end find_a_l221_221417


namespace tangent_product_eq_three_l221_221523

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l221_221523


namespace product_of_numbers_l221_221071

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : x * y = 200 :=
sorry

end product_of_numbers_l221_221071


namespace linear_eq_with_one_variable_is_B_l221_221340

-- Define the equations
def eqA (x y : ℝ) : Prop := 2 * x = 3 * y
def eqB (x : ℝ) : Prop := 7 * x + 5 = 6 * (x - 1)
def eqC (x : ℝ) : Prop := x^2 + (1 / 2) * (x - 1) = 1
def eqD (x : ℝ) : Prop := (1 / x) - 2 = x

-- State the problem
theorem linear_eq_with_one_variable_is_B :
  ∃ x : ℝ, ¬ (∃ y : ℝ, eqA x y) ∧ eqB x ∧ ¬ eqC x ∧ ¬ eqD x :=
by {
  -- mathematical content goes here
  sorry
}

end linear_eq_with_one_variable_is_B_l221_221340


namespace total_cost_john_paid_l221_221285

theorem total_cost_john_paid 
  (meters_of_cloth : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ)
  (h1 : meters_of_cloth = 9.25)
  (h2 : cost_per_meter = 48)
  (h3 : total_cost = meters_of_cloth * cost_per_meter) :
  total_cost = 444 :=
sorry

end total_cost_john_paid_l221_221285


namespace compute_remainder_l221_221369

/-- T is the sum of all three-digit positive integers 
  where the digits are distinct, the hundreds digit is at least 2,
  and the digit 1 is not used in any place. -/
def T : ℕ := 
  let hundreds_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 56 * 100
  let tens_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 49 * 10
  let units_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 49
  hundreds_sum + tens_sum + units_sum

/-- Theorem: Compute the remainder when T is divided by 1000. -/
theorem compute_remainder : T % 1000 = 116 := by
  sorry

end compute_remainder_l221_221369


namespace smallest_divisor_l221_221498

theorem smallest_divisor (k n : ℕ) (x y : ℤ) :
  (∃ n : ℕ, k ∣ 2^n + 15) ∧ (∃ x y : ℤ, k = 3 * x^2 - 4 * x * y + 3 * y^2) → k = 23 := by
  sorry

end smallest_divisor_l221_221498


namespace sofia_running_time_l221_221915

theorem sofia_running_time :
  ∃ t : ℤ, t = 8 * 60 + 20 ∧ 
  (∀ (laps : ℕ) (d1 d2 v1 v2 : ℤ),
    laps = 5 →
    d1 = 200 →
    v1 = 4 →
    d2 = 300 →
    v2 = 6 →
    t = laps * ((d1 / v1 + d2 / v2))) :=
by
  sorry

end sofia_running_time_l221_221915


namespace soccer_teams_participation_l221_221512

theorem soccer_teams_participation (total_games : ℕ) (teams_play : ℕ → ℕ) (x : ℕ) :
  (total_games = 20) → (teams_play x = x * (x - 1)) → x = 5 :=
by
  sorry

end soccer_teams_participation_l221_221512


namespace even_sum_probability_correct_l221_221494

-- Definition: Calculate probabilities based on the given wheels
def even_probability_wheel_one : ℚ := 1/3
def odd_probability_wheel_one : ℚ := 2/3
def even_probability_wheel_two : ℚ := 1/4
def odd_probability_wheel_two : ℚ := 3/4

-- Probability of both numbers being even
def both_even_probability : ℚ := even_probability_wheel_one * even_probability_wheel_two

-- Probability of both numbers being odd
def both_odd_probability : ℚ := odd_probability_wheel_one * odd_probability_wheel_two

-- Final probability of the sum being even
def even_sum_probability : ℚ := both_even_probability + both_odd_probability

theorem even_sum_probability_correct : even_sum_probability = 7/12 := 
sorry

end even_sum_probability_correct_l221_221494


namespace marie_stamps_giveaway_l221_221703

theorem marie_stamps_giveaway :
  let notebooks := 4
  let stamps_per_notebook := 20
  let binders := 2
  let stamps_per_binder := 50
  let fraction_to_keep := 1/4
  let total_stamps := notebooks * stamps_per_notebook + binders * stamps_per_binder
  let stamps_to_keep := fraction_to_keep * total_stamps
  let stamps_to_give_away := total_stamps - stamps_to_keep
  stamps_to_give_away = 135 :=
by
  sorry

end marie_stamps_giveaway_l221_221703


namespace smallest_positive_period_of_f_max_min_values_of_f_l221_221076

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) :=
sorry

theorem max_min_values_of_f :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), 0 ≤ f x ∧ f x ≤ 1 + Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 0) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1 + Real.sqrt 2) :=
sorry

end smallest_positive_period_of_f_max_min_values_of_f_l221_221076


namespace cost_of_iPhone_l221_221464

theorem cost_of_iPhone (P : ℝ) 
  (phone_contract_cost : ℝ := 200)
  (case_percent_of_P : ℝ := 0.20)
  (headphones_percent_of_case : ℝ := 0.50)
  (total_yearly_cost : ℝ := 3700) :
  let year_phone_contract_cost := (phone_contract_cost * 12)
  let case_cost := (case_percent_of_P * P)
  let headphones_cost := (headphones_percent_of_case * case_cost)
  P + year_phone_contract_cost + case_cost + headphones_cost = total_yearly_cost → 
  P = 1000 :=
by
  sorry  -- proof not required

end cost_of_iPhone_l221_221464


namespace count_numbers_with_digit_2_l221_221867

def contains_digit_2 (n : Nat) : Prop :=
  n / 100 = 2 ∨ (n / 10 % 10) = 2 ∨ (n % 10) = 2

theorem count_numbers_with_digit_2 (N : Nat) (H : 200 ≤ N ∧ N ≤ 499) : 
  Nat.card {n // 200 ≤ n ∧ n ≤ 499 ∧ contains_digit_2 n} = 138 :=
by
  sorry

end count_numbers_with_digit_2_l221_221867


namespace equation_solution_l221_221949

theorem equation_solution (x : ℝ) : 
  (x - 3)^4 = 16 → x = 5 :=
by
  sorry

end equation_solution_l221_221949


namespace garden_perimeter_l221_221981

-- formally defining the conditions of the problem
variables (x y : ℝ)
def diagonal_of_garden : Prop := x^2 + y^2 = 900
def area_of_garden : Prop := x * y = 216

-- final statement to prove the perimeter of the garden
theorem garden_perimeter (h1 : diagonal_of_garden x y) (h2 : area_of_garden x y) : 2 * (x + y) = 73 := sorry

end garden_perimeter_l221_221981


namespace divide_8_friends_among_4_teams_l221_221672

def num_ways_to_divide_friends (n : ℕ) (teams : ℕ) :=
  teams ^ n

theorem divide_8_friends_among_4_teams :
  num_ways_to_divide_friends 8 4 = 65536 :=
by sorry

end divide_8_friends_among_4_teams_l221_221672


namespace plane_speed_west_l221_221975

theorem plane_speed_west (v t : ℝ) : 
  (300 * t + 300 * t = 1200) ∧ (t = 7 - t) → 
  (v = 300 * t / (7 - t)) ∧ (t = 2) → 
  v = 120 :=
by
  intros h1 h2
  sorry

end plane_speed_west_l221_221975


namespace man_older_than_son_l221_221940

theorem man_older_than_son (S M : ℕ) (hS : S = 27) (hM : M + 2 = 2 * (S + 2)) : M - S = 29 := 
by {
  sorry
}

end man_older_than_son_l221_221940


namespace nests_count_l221_221684

theorem nests_count (birds nests : ℕ) (h1 : birds = 6) (h2 : birds - nests = 3) : nests = 3 := by
  sorry

end nests_count_l221_221684


namespace correct_factorization_l221_221364

variable (x y : ℝ)

theorem correct_factorization :
  x^2 - 2 * x * y + x = x * (x - 2 * y + 1) :=
by sorry

end correct_factorization_l221_221364


namespace sum_of_coefficients_l221_221787

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 a6 a7 : ℤ) (a_eq : (1 - 2 * (0:ℤ)) ^ 7 = a)
  (hx_eq : ∀ (x : ℤ), (1 - 2 * x) ^ 7 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) :
  a1 + a2 + a3 + a4 + a5 + a6 + a7 = -2 :=
by
  sorry

end sum_of_coefficients_l221_221787


namespace digit_place_value_ratio_l221_221607

theorem digit_place_value_ratio :
  let number := 86304.2957
  let digit_6_value := 1000
  let digit_5_value := 0.1
  digit_6_value / digit_5_value = 10000 :=
by
  let number := 86304.2957
  let digit_6_value := 1000
  let digit_5_value := 0.1
  sorry

end digit_place_value_ratio_l221_221607


namespace original_class_strength_l221_221389

theorem original_class_strength (T N : ℕ) (h1 : T = 40 * N) (h2 : T + 12 * 32 = 36 * (N + 12)) : N = 12 :=
by
  sorry

end original_class_strength_l221_221389


namespace abs_x_minus_y_zero_l221_221649

theorem abs_x_minus_y_zero (x y : ℝ) 
  (h_avg : (x + y + 30 + 29 + 31) / 5 = 30)
  (h_var : ((x - 30)^2 + (y - 30)^2 + (30 - 30)^2 + (29 - 30)^2 + (31 - 30)^2) / 5 = 2) : 
  |x - y| = 0 :=
  sorry

end abs_x_minus_y_zero_l221_221649


namespace root_of_quadratic_l221_221299

theorem root_of_quadratic {x a : ℝ} (h : x = 2 ∧ x^2 - x + a = 0) : a = -2 := 
by
  sorry

end root_of_quadratic_l221_221299


namespace boat_upstream_time_is_1_5_hours_l221_221207

noncomputable def time_to_cover_distance_upstream
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ) : ℝ :=
  distance_downstream / (speed_boat_still_water - speed_stream)

theorem boat_upstream_time_is_1_5_hours
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (downstream_distance : ℝ)
  (h1 : speed_stream = 3)
  (h2 : speed_boat_still_water = 15)
  (h3 : time_downstream = 1)
  (h4 : downstream_distance = speed_boat_still_water + speed_stream) :
  time_to_cover_distance_upstream speed_stream speed_boat_still_water time_downstream downstream_distance = 1.5 :=
by
  sorry

end boat_upstream_time_is_1_5_hours_l221_221207


namespace all_flowers_bloom_simultaneously_l221_221713

-- Define days of the week
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

-- Define bloom conditions for the flowers
def sunflowers_bloom (d : Day) : Prop :=
  d ≠ Tuesday ∧ d ≠ Thursday ∧ d ≠ Sunday

def lilies_bloom (d : Day) : Prop :=
  d ≠ Thursday ∧ d ≠ Saturday

def peonies_bloom (d : Day) : Prop :=
  d ≠ Sunday

-- Define the main theorem
theorem all_flowers_bloom_simultaneously : ∃ d : Day, 
  sunflowers_bloom d ∧ lilies_bloom d ∧ peonies_bloom d ∧
  (∀ d', d' ≠ d → ¬ (sunflowers_bloom d' ∧ lilies_bloom d' ∧ peonies_bloom d')) :=
by
  sorry

end all_flowers_bloom_simultaneously_l221_221713


namespace algebra_problem_l221_221441

theorem algebra_problem 
  (x : ℝ) 
  (h : x^2 - 2 * x = 3) : 
  2 * x^2 - 4 * x + 3 = 9 := 
by 
  sorry

end algebra_problem_l221_221441


namespace player_A_wins_if_n_equals_9_l221_221410

-- Define the conditions
def drawing_game (n : ℕ) : Prop :=
  ∃ strategy : ℕ → ℕ,
    strategy 0 = 1 ∧ -- Player A always starts by drawing 1 ball
    (∀ k, 1 ≤ strategy k ∧ strategy k ≤ 3) ∧ -- Players draw between 1 and 3 balls
    ∀ b, 1 ≤ b → b ≤ 3 → (n - 1 - strategy (b - 1)) ≤ 3 → (strategy (n - 1 - (b - 1)) = n - (b - 1) - 1)

-- State the problem to prove Player A has a winning strategy if n = 9
theorem player_A_wins_if_n_equals_9 : drawing_game 9 :=
sorry

end player_A_wins_if_n_equals_9_l221_221410


namespace jessica_cut_r_l221_221016

variable (r_i r_t r_c : ℕ)

theorem jessica_cut_r : r_i = 7 → r_g = 59 → r_t = 20 → r_c = r_t - r_i → r_c = 13 :=
by
  intros h_i h_g h_t h_c
  have h1 : r_i = 7 := h_i
  have h2 : r_t = 20 := h_t
  have h3 : r_c = r_t - r_i := h_c
  have h_correct : r_c = 13
  · sorry
  exact h_correct

end jessica_cut_r_l221_221016


namespace find_certain_number_l221_221720

theorem find_certain_number (x certain_number : ℕ) (h1 : certain_number + x = 13200) (h2 : x = 3327) : certain_number = 9873 :=
by
  sorry

end find_certain_number_l221_221720


namespace solution_l221_221582

-- Definitions for perpendicular and parallel relations
def perpendicular (a b : Type) : Prop := sorry -- Abstraction for perpendicularity
def parallel (a b : Type) : Prop := sorry -- Abstraction for parallelism

-- Here we define x, y, z as variables
variables {x y : Type} {z : Type}

-- Conditions for Case 2
def case2_lines_plane (x y : Type) (z : Type) := 
  (perpendicular x z) ∧ (perpendicular y z) → (parallel x y)

-- Conditions for Case 3
def case3_planes_line (x y : Type) (z : Type) := 
  (perpendicular x z) ∧ (perpendicular y z) → (parallel x y)

-- Theorem statement combining both cases
theorem solution : case2_lines_plane x y z ∧ case3_planes_line x y z := 
sorry

end solution_l221_221582


namespace cornbread_pieces_l221_221237

theorem cornbread_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ)
  (hl : pan_length = 20) (hw : pan_width = 18) (hp : piece_length = 2) (hq : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 90 :=
by
  sorry

end cornbread_pieces_l221_221237


namespace scientific_notation_219400_l221_221745

def scientific_notation (n : ℝ) (m : ℝ) : Prop := n = m * 10^5

theorem scientific_notation_219400 : scientific_notation 219400 2.194 := 
by
  sorry

end scientific_notation_219400_l221_221745


namespace intersection_M_N_l221_221647

open Set

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x^2 + x ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := sorry

end intersection_M_N_l221_221647


namespace f_2_plus_f_5_eq_2_l221_221126

noncomputable def f : ℝ → ℝ := sorry

open Real

-- Conditions: f(3^x) = x * log 9
axiom f_cond (x : ℝ) : f (3^x) = x * log 9

-- Question: f(2) + f(5) = 2
theorem f_2_plus_f_5_eq_2 : f 2 + f 5 = 2 := sorry

end f_2_plus_f_5_eq_2_l221_221126


namespace simplify_and_rationalize_l221_221402

noncomputable def simplify_expr : ℝ :=
  1 / (1 - (1 / (Real.sqrt 5 - 2)))

theorem simplify_and_rationalize :
  simplify_expr = (1 - Real.sqrt 5) / 4 := by
  sorry

end simplify_and_rationalize_l221_221402


namespace altitudes_bounded_by_perimeter_l221_221964

theorem altitudes_bounded_by_perimeter (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a + b + c = 2) :
  ¬ (∀ (ha hb hc : ℝ), ha = 2 / a * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     hb = 2 / b * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     hc = 2 / c * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     ha > 1 / Real.sqrt 3 ∧ 
                     hb > 1 / Real.sqrt 3 ∧ 
                     hc > 1 / Real.sqrt 3 ) :=
sorry

end altitudes_bounded_by_perimeter_l221_221964


namespace solution_set_a1_range_of_a_l221_221933

def f (x a : ℝ) : ℝ := abs (x - a) * abs (x + abs (x - 2)) * abs (x - a)

theorem solution_set_a1 (x : ℝ) : f x 1 < 0 ↔ x < 1 :=
by
  sorry

theorem range_of_a (a : ℝ) : (∀ x, x < 1 → f x a < 0) ↔ 1 ≤ a :=
by
  sorry

end solution_set_a1_range_of_a_l221_221933


namespace solveForN_l221_221638

-- Define the condition that sqrt(8 + n) = 9
def condition (n : ℝ) : Prop := Real.sqrt (8 + n) = 9

-- State the main theorem that given the condition, n must be 73
theorem solveForN (n : ℝ) (h : condition n) : n = 73 := by
  sorry

end solveForN_l221_221638


namespace greatest_divisor_of_630_lt_35_and_factor_of_90_l221_221822

theorem greatest_divisor_of_630_lt_35_and_factor_of_90 : ∃ d : ℕ, d < 35 ∧ d ∣ 630 ∧ d ∣ 90 ∧ ∀ e : ℕ, (e < 35 ∧ e ∣ 630 ∧ e ∣ 90) → e ≤ d := 
sorry

end greatest_divisor_of_630_lt_35_and_factor_of_90_l221_221822


namespace terminating_decimal_count_l221_221216

def count_terminating_decimals (n: ℕ): ℕ :=
  (n / 17)

theorem terminating_decimal_count : count_terminating_decimals 493 = 29 := by
  sorry

end terminating_decimal_count_l221_221216


namespace abs_negative_five_l221_221392

theorem abs_negative_five : abs (-5) = 5 :=
by
  sorry

end abs_negative_five_l221_221392


namespace film_finishes_earlier_on_first_channel_l221_221660

-- Definitions based on conditions
def DurationSegmentFirstChannel (n : ℕ) : ℝ := n * 22
def DurationSegmentSecondChannel (k : ℕ) : ℝ := k * 11

-- The time when first channel starts the n-th segment
def StartNthSegmentFirstChannel (n : ℕ) : ℝ := (n - 1) * 22

-- The number of segments second channel shows by the time first channel starts the n-th segment
def SegmentsShownSecondChannel (n : ℕ) : ℕ := ((n - 1) * 22) / 11

-- If first channel finishes earlier than second channel
theorem film_finishes_earlier_on_first_channel (n : ℕ) (hn : 1 < n) :
  DurationSegmentFirstChannel n < DurationSegmentSecondChannel (SegmentsShownSecondChannel n + 1) :=
sorry

end film_finishes_earlier_on_first_channel_l221_221660


namespace prob_enter_A_and_exit_F_l221_221054

-- Define the problem description
def entrances : ℕ := 2
def exits : ℕ := 3

-- Define the probabilities
def prob_enter_A : ℚ := 1 / entrances
def prob_exit_F : ℚ := 1 / exits

-- Statement that encapsulates the proof problem
theorem prob_enter_A_and_exit_F : prob_enter_A * prob_exit_F = 1 / 6 := 
by sorry

end prob_enter_A_and_exit_F_l221_221054


namespace laura_owes_correct_amount_l221_221365

def principal : ℝ := 35
def annual_rate : ℝ := 0.07
def time_years : ℝ := 1
def interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * R * T
def total_amount_owed (P : ℝ) (I : ℝ) : ℝ := P + I

theorem laura_owes_correct_amount :
  total_amount_owed principal (interest principal annual_rate time_years) = 37.45 :=
sorry

end laura_owes_correct_amount_l221_221365


namespace exists_digit_a_l221_221526

theorem exists_digit_a : 
  ∃ (a : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (1111 * a - 1 = (a - 1) ^ (a - 2)) :=
by {
  sorry
}

end exists_digit_a_l221_221526


namespace complete_square_l221_221793

theorem complete_square (y : ℝ) : y^2 + 12 * y + 40 = (y + 6)^2 + 4 :=
by {
  sorry
}

end complete_square_l221_221793


namespace height_of_spherical_cap_case1_height_of_spherical_cap_case2_l221_221959

variable (R : ℝ) (c : ℝ)
variable (h_c_gt_1 : c > 1)

-- Case 1: Not including the circular cap in the surface area
theorem height_of_spherical_cap_case1 : ∃ m : ℝ, m = (2 * R * (c - 1)) / c :=
by
  sorry

-- Case 2: Including the circular cap in the surface area
theorem height_of_spherical_cap_case2 : ∃ m : ℝ, m = (2 * R * (c - 2)) / (c - 1) :=
by
  sorry

end height_of_spherical_cap_case1_height_of_spherical_cap_case2_l221_221959


namespace HCF_of_numbers_l221_221750

theorem HCF_of_numbers (a b : ℕ) (h₁ : a * b = 84942) (h₂ : Nat.lcm a b = 2574) : Nat.gcd a b = 33 :=
by
  sorry

end HCF_of_numbers_l221_221750


namespace central_angle_of_regular_polygon_l221_221115

theorem central_angle_of_regular_polygon (n : ℕ) (h : 360 ∣ 360 - 36 * n) :
  n = 10 :=
by
  sorry

end central_angle_of_regular_polygon_l221_221115


namespace sum_of_cubes_of_roots_l221_221525

theorem sum_of_cubes_of_roots (P : Polynomial ℝ)
  (hP : P = Polynomial.C (-1) + Polynomial.X ^ 3 - Polynomial.C 3 * Polynomial.X) 
  (x1 x2 x3 : ℝ) 
  (hr : P.eval x1 = 0 ∧ P.eval x2 = 0 ∧ P.eval x3 = 0) :
  x1^3 + x2^3 + x3^3 = 3 := 
sorry

end sum_of_cubes_of_roots_l221_221525


namespace find_value_l221_221393

theorem find_value (a : ℝ) (h : a^2 - 2*a = -1) : 3*a^2 - 6*a + 2027 = 2024 :=
sorry

end find_value_l221_221393


namespace mutually_exclusive_event_l221_221133

def Event := String  -- define a simple type for events

/-- Define the events -/
def at_most_one_hit : Event := "at most one hit"
def two_hits : Event := "two hits"

/-- Define a function to check mutual exclusiveness -/
def mutually_exclusive (e1 e2 : Event) : Prop := 
  e1 ≠ e2

theorem mutually_exclusive_event :
  mutually_exclusive at_most_one_hit two_hits :=
by
  sorry

end mutually_exclusive_event_l221_221133


namespace train_cross_bridge_time_l221_221111

/-
  Define the given conditions:
  - Length of the train (lt): 200 m
  - Speed of the train (st_kmh): 72 km/hr
  - Length of the bridge (lb): 132 m
-/

namespace TrainProblem

def length_of_train : ℕ := 200
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 132

/-
  Convert speed from km/hr to m/s
-/
def speed_of_train_ms : ℕ := speed_of_train_kmh * 1000 / 3600

/-
  Calculate total distance to be traveled (train length + bridge length).
-/
def total_distance : ℕ := length_of_train + length_of_bridge

/-
  Use the formula Time = Distance / Speed
-/
def time_to_cross_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_cross_bridge_time : 
  (length_of_train = 200) →
  (speed_of_train_kmh = 72) →
  (length_of_bridge = 132) →
  time_to_cross_bridge = 16.6 :=
by
  intros lt st lb
  sorry

end TrainProblem

end train_cross_bridge_time_l221_221111


namespace abs_frac_lt_one_l221_221050

theorem abs_frac_lt_one (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  |(x - y) / (1 - x * y)| < 1 :=
sorry

end abs_frac_lt_one_l221_221050


namespace us_supermarkets_count_l221_221705

-- Definition of variables and conditions
def total_supermarkets : ℕ := 84
def difference_us_canada : ℕ := 10

-- Proof statement
theorem us_supermarkets_count (C : ℕ) (H : 2 * C + difference_us_canada = total_supermarkets) :
  C + difference_us_canada = 47 :=
sorry

end us_supermarkets_count_l221_221705


namespace find_x_l221_221281

theorem find_x (x : ℝ) : (1 + (1 / (1 + x)) = 2 * (1 / (1 + x))) → x = 0 :=
by
  intro h
  sorry

end find_x_l221_221281


namespace geo_seq_product_l221_221407

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m / a 1

theorem geo_seq_product
  {a : ℕ → ℝ}
  (h_pos : ∀ n, a n > 0)
  (h_seq : geometric_sequence a)
  (h_roots : ∃ x y, (x*x - 10 * x + 16 = 0) ∧ (y*y - 10 * y + 16 = 0) ∧ a 1 = x ∧ a 19 = y) :
  a 8 * a 10 * a 12 = 64 := 
sorry

end geo_seq_product_l221_221407


namespace calculation_l221_221198

theorem calculation :
  12 - 10 + 8 / 2 * 5 + 4 - 6 * 3 + 1 = 9 :=
by
  sorry

end calculation_l221_221198


namespace determine_a_for_quadratic_l221_221097

theorem determine_a_for_quadratic (a : ℝ) : 
  (∃ x : ℝ, 3 * x ^ (a - 1) - x = 5 ∧ a - 1 = 2) → a = 3 := 
sorry

end determine_a_for_quadratic_l221_221097


namespace ways_to_select_books_l221_221500

theorem ways_to_select_books (nChinese nMath nEnglish : ℕ) (h1 : nChinese = 9) (h2 : nMath = 7) (h3 : nEnglish = 5) :
  (nChinese * nMath + nChinese * nEnglish + nMath * nEnglish) = 143 :=
by
  sorry

end ways_to_select_books_l221_221500


namespace calculation_result_l221_221831

theorem calculation_result :
  (2 : ℝ)⁻¹ - (1 / 2 : ℝ)^0 + (2 : ℝ)^2023 * (-0.5 : ℝ)^2023 = -3 / 2 := sorry

end calculation_result_l221_221831


namespace find_x_value_l221_221623

theorem find_x_value (a b x : ℤ) (h : a * b = (a - 1) * (b - 1)) (h2 : x * 9 = 160) :
  x = 21 :=
sorry

end find_x_value_l221_221623


namespace cubicsum_l221_221784

theorem cubicsum (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : a^3 + b^3 = 1008 := 
by 
  sorry

end cubicsum_l221_221784


namespace no_solution_in_positive_rationals_l221_221854

theorem no_solution_in_positive_rationals (n : ℕ) (hn : n > 0) (x y : ℚ) (hx : x > 0) (hy : y > 0) :
  x + y + (1 / x) + (1 / y) ≠ 3 * n :=
sorry

end no_solution_in_positive_rationals_l221_221854


namespace total_weight_l221_221666

axiom D : ℕ -- Daughter's weight
axiom C : ℕ -- Grandchild's weight
axiom M : ℕ -- Mother's weight

-- Given conditions from the problem
axiom h1 : D + C = 60
axiom h2 : C = M / 5
axiom h3 : D = 50

-- The statement to be proven
theorem total_weight : M + D + C = 110 :=
by sorry

end total_weight_l221_221666


namespace max_value_of_f_l221_221886

noncomputable def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_f (a b : ℝ) (h_symmetric : ∀ x : ℝ, f (-2 - x) a b = f (-2 + x) a b) :
  ∃ x : ℝ, f x a b = 16 := by
  sorry

end max_value_of_f_l221_221886


namespace problem_range_of_k_l221_221083

theorem problem_range_of_k (k : ℝ) : 
  (∀ x : ℝ, x^2 - 11 * x + (30 + k) = 0 → x > 5) → (0 < k ∧ k ≤ 1 / 4) :=
by
  sorry

end problem_range_of_k_l221_221083


namespace value_of_a5_l221_221890

variable (a : ℕ → ℕ)

-- The initial condition
axiom initial_condition : a 1 = 2

-- The recurrence relation
axiom recurrence_relation : ∀ n : ℕ, n ≠ 0 → n * a (n+1) = 2 * (n + 1) * a n

theorem value_of_a5 : a 5 = 160 := 
sorry

end value_of_a5_l221_221890


namespace find_ratio_l221_221328

theorem find_ratio (a b : ℝ) (h1 : ∀ x, ax^2 + bx + 2 < 0 ↔ (x < -1/2 ∨ x > 1/3)) :
  (a - b) / a = 5 / 6 := 
sorry

end find_ratio_l221_221328


namespace farmer_total_acres_l221_221988

theorem farmer_total_acres (x : ℕ) (H1 : 4 * x = 376) : 
  5 * x + 2 * x + 4 * x = 1034 :=
by
  -- This placeholder is indicating unfinished proof
  sorry

end farmer_total_acres_l221_221988


namespace area_EYH_trapezoid_l221_221136

theorem area_EYH_trapezoid (EF GH : ℕ) (EF_len : EF = 15) (GH_len : GH = 35) 
(Area_trapezoid : (EF + GH) * 16 / 2 = 400) : 
∃ (EYH_area : ℕ), EYH_area = 84 := by
  sorry

end area_EYH_trapezoid_l221_221136


namespace find_length_AD_l221_221843

noncomputable def length_AD (AB AC BC : ℝ) (is_equal_AB_AC : AB = AC) (BD DC : ℝ) (D_midpoint : BD = DC) : ℝ :=
  let BE := BC / 2
  let AE := Real.sqrt (AB ^ 2 - BE ^ 2)
  AE

theorem find_length_AD (AB AC BC BD DC : ℝ) (is_equal_AB_AC : AB = AC) (D_midpoint : BD = DC) (H1 : AB = 26) (H2 : AC = 26) (H3 : BC = 24) (H4 : BD = 12) (H5 : DC = 12) :
  length_AD AB AC BC is_equal_AB_AC BD DC D_midpoint = 2 * Real.sqrt 133 :=
by
  -- the steps of the proof would go here
  sorry

end find_length_AD_l221_221843


namespace find_k_l221_221910

noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem find_k (k : ℝ) (h_pos : 0 < k) (h_exists : ∃ x₀ : ℝ, 1 ≤ x₀ ∧ g x₀ ≤ k * (-x₀^2 + 3 * x₀)) : 
  k > (1 / 2) * (Real.exp 1 + 1 / Real.exp 1) :=
sorry

end find_k_l221_221910


namespace students_not_playing_either_game_l221_221491

theorem students_not_playing_either_game
  (total_students : ℕ) -- There are 20 students in the class
  (play_basketball : ℕ) -- Half of them play basketball
  (play_volleyball : ℕ) -- Two-fifths of them play volleyball
  (play_both : ℕ) -- One-tenth of them play both basketball and volleyball
  (h_total : total_students = 20)
  (h_basketball : play_basketball = 10)
  (h_volleyball : play_volleyball = 8)
  (h_both : play_both = 2) :
  total_students - (play_basketball + play_volleyball - play_both) = 4 := by
  sorry

end students_not_playing_either_game_l221_221491


namespace range_of_a_l221_221506

noncomputable def f (x a : ℝ) : ℝ := x^3 - 3 * a^2 * x + 1
def intersects_at_single_point (f : ℝ → ℝ → ℝ) (a : ℝ) : Prop :=
∃! x, f x a = 3

theorem range_of_a (a : ℝ) :
  intersects_at_single_point f a ↔ -1 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l221_221506


namespace triangle_DEF_angle_l221_221047

noncomputable def one_angle_of_triangle_DEF (x : ℝ) : ℝ :=
  let arc_DE := 2 * x + 40
  let arc_EF := 3 * x + 50
  let arc_FD := 4 * x - 30
  if (arc_DE + arc_EF + arc_FD = 360)
  then (1 / 2) * arc_EF
  else 0

theorem triangle_DEF_angle (x : ℝ) (h : 2 * x + 40 + 3 * x + 50 + 4 * x - 30 = 360) :
  one_angle_of_triangle_DEF x = 75 :=
by sorry

end triangle_DEF_angle_l221_221047


namespace range_of_a_l221_221450

variable (a x : ℝ)

theorem range_of_a (h : ax > 2) (h_transform: ax > 2 → x < 2/a) : a < 0 :=
sorry

end range_of_a_l221_221450


namespace part_A_part_B_part_D_l221_221948

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < 1)
variable (hβ : 0 < β ∧ β < 1)

-- Part A: single transmission probability
theorem part_A (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 :=
by sorry

-- Part B: triple transmission probability
theorem part_B (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  β * (1 - β)^2 = β * (1 - β)^2 :=
by sorry

-- Part D: comparing single and triple transmission
theorem part_D (α β : ℝ) (hα : 0 < α ∧ α < 0.5) (hβ : 0 < β ∧ β < 1) :
  (1 - α) < (1 - α)^3 + 3 * α * (1 - α)^2 :=
by sorry

end part_A_part_B_part_D_l221_221948


namespace arithmetic_sequence_product_l221_221971

theorem arithmetic_sequence_product (a : ℕ → ℤ) (d : ℤ) (h_inc : ∀ n m, n < m → a n < a m) 
  (h_arith : ∀ n, a (n + 1) = a n + d) (h_prod : a 4 * a 5 = 12) : a 2 * a 7 = 6 :=
sorry

end arithmetic_sequence_product_l221_221971


namespace construct_right_triangle_l221_221932

theorem construct_right_triangle (c m n : ℝ) (hc : c > 0) (hm : m > 0) (hn : n > 0) : 
  ∃ a b : ℝ, a^2 + b^2 = c^2 ∧ a / b = m / n :=
by
  sorry

end construct_right_triangle_l221_221932


namespace gcd_154_and_90_l221_221922

theorem gcd_154_and_90 : Nat.gcd 154 90 = 2 := by
  sorry

end gcd_154_and_90_l221_221922


namespace num_right_triangles_with_incenter_origin_l221_221592

theorem num_right_triangles_with_incenter_origin (p : ℕ) (hp : Nat.Prime p) :
  let M : ℤ × ℤ := (p * 1994, 7 * p * 1994)
  let is_lattice_point (x : ℤ × ℤ) : Prop := True  -- All points considered are lattice points
  let is_right_angle_vertex (M : ℤ × ℤ) : Prop := True
  let is_incenter_origin (M : ℤ × ℤ) : Prop := True
  let num_triangles (p : ℕ) : ℕ :=
    if p = 2 then 18
    else if p = 997 then 20
    else 36
  num_triangles p = if p = 2 then 18 else if p = 997 then 20 else 36 := (

  by sorry

 )

end num_right_triangles_with_incenter_origin_l221_221592


namespace perfect_square_fraction_l221_221664

open Nat

theorem perfect_square_fraction (a b : ℕ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : (ab + 1) ∣ (a^2 + b^2)) : ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
by 
  sorry

end perfect_square_fraction_l221_221664


namespace point_in_fourth_quadrant_l221_221876

noncomputable def a : ℤ := 2

theorem point_in_fourth_quadrant (x y : ℤ) (h1 : x = a - 1) (h2 : y = a - 3) (h3 : x > 0) (h4 : y < 0) : a = 2 := by
  sorry

end point_in_fourth_quadrant_l221_221876


namespace vector_scalar_sub_l221_221004

def a : ℝ × ℝ := (3, -9)
def b : ℝ × ℝ := (2, -8)
def scalar1 : ℝ := 4
def scalar2 : ℝ := 3

theorem vector_scalar_sub:
  scalar1 • a - scalar2 • b = (6, -12) := by
  sorry

end vector_scalar_sub_l221_221004


namespace type_R_completion_time_l221_221060

theorem type_R_completion_time :
  (∃ R : ℝ, (2 / R + 3 / 7 = 1 / 1.2068965517241381) ∧ abs (R - 5) < 0.01) :=
  sorry

end type_R_completion_time_l221_221060


namespace gallon_of_water_weighs_eight_pounds_l221_221386

theorem gallon_of_water_weighs_eight_pounds
  (pounds_per_tablespoon : ℝ := 1.5)
  (cubic_feet_per_gallon : ℝ := 7.5)
  (cost_per_tablespoon : ℝ := 0.50)
  (total_cost : ℝ := 270)
  (bathtub_capacity_cubic_feet : ℝ := 6)
  : (6 * 7.5) * pounds_per_tablespoon = 270 / cost_per_tablespoon / 1.5 :=
by
  sorry

end gallon_of_water_weighs_eight_pounds_l221_221386


namespace simplify_fraction_l221_221744

theorem simplify_fraction :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := 
sorry

end simplify_fraction_l221_221744


namespace tangent_line_at_b_l221_221815

theorem tangent_line_at_b (b : ℝ) : (∃ x : ℝ, (4*x^3 = 4) ∧ (4*x + b = x^4 - 1)) ↔ (b = -4) := 
by 
  sorry

end tangent_line_at_b_l221_221815


namespace correct_completion_l221_221795

-- Definitions of conditions
def sentence_template := "By the time he arrives, all the work ___, with ___ our teacher will be content."
def option_A := ("will be accomplished", "that")
def option_B := ("will have been accomplished", "which")
def option_C := ("will have accomplished", "it")
def option_D := ("had been accomplished", "him")

-- The actual proof statement
theorem correct_completion : (option_B.fst = "will have been accomplished") ∧ (option_B.snd = "which") :=
by
  sorry

end correct_completion_l221_221795


namespace arithmetic_sequence_length_l221_221432

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ ∀ (a_1 a_2 a_n : ℤ), a_1 = 2 ∧ a_2 = 6 ∧ a_n = 2006 →
  a_n = a_1 + (n - 1) * (a_2 - a_1) → n = 502 := by
  sorry

end arithmetic_sequence_length_l221_221432


namespace original_price_of_shoes_l221_221177

theorem original_price_of_shoes (P : ℝ) (h1 : 2 * 0.60 * P + 0.80 * 100 = 140) : P = 50 :=
by
  sorry

end original_price_of_shoes_l221_221177


namespace arithmetic_geometric_sequence_general_term_l221_221966

theorem arithmetic_geometric_sequence_general_term :
  ∃ q a1 : ℕ, (∀ n : ℕ, a2 = 6 ∧ 6 * a1 + a3 = 30) →
  (∀ n : ℕ, (q = 2 ∧ a1 = 3 → a_n = 3 * 3^(n-1)) ∨ (q = 3 ∧ a1 = 2 → a_n = 2 * 2^(n-1))) :=
by
  sorry

end arithmetic_geometric_sequence_general_term_l221_221966


namespace pq_work_together_in_10_days_l221_221600

theorem pq_work_together_in_10_days 
  (p q r : ℝ)
  (hq : 1/q = 1/28)
  (hr : 1/r = 1/35)
  (hp : 1/p = 1/q + 1/r) :
  1/p + 1/q = 1/10 :=
by sorry

end pq_work_together_in_10_days_l221_221600


namespace quadratic_roots_l221_221188

theorem quadratic_roots (a b: ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0)
  (root_condition1 : a * (-1/2)^2 + b * (-1/2) + 2 = 0)
  (root_condition2 : a * (1/3)^2 + b * (1/3) + 2 = 0) 
  : a - b = -10 := 
by {
  sorry
}

end quadratic_roots_l221_221188


namespace choir_members_l221_221088

theorem choir_members (n k c : ℕ) (h1 : n = k^2 + 11) (h2 : n = c * (c + 5)) : n = 300 :=
sorry

end choir_members_l221_221088


namespace find_sum_l221_221989

variable {f : ℝ → ℝ}

-- Conditions of the problem
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def condition_2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
def condition_3 (f : ℝ → ℝ) : Prop := f 1 = 9

theorem find_sum (h_odd : odd_function f) (h_cond2 : condition_2 f) (h_cond3 : condition_3 f) :
  f 2010 + f 2011 + f 2012 = -9 :=
sorry

end find_sum_l221_221989


namespace trigonometric_identity_l221_221181

theorem trigonometric_identity :
  (Real.cos (17 * Real.pi / 180) * Real.sin (43 * Real.pi / 180) + 
   Real.sin (163 * Real.pi / 180) * Real.sin (47 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end trigonometric_identity_l221_221181


namespace largest_square_area_l221_221090

theorem largest_square_area (a b c : ℝ)
  (h1 : a^2 + b^2 = c^2)
  (h2 : a^2 + b^2 + c^2 = 450) :
  c^2 = 225 :=
by
  sorry

end largest_square_area_l221_221090


namespace second_movie_time_difference_l221_221368

def first_movie_length := 90 -- 1 hour and 30 minutes in minutes
def popcorn_time := 10 -- Time spent making popcorn in minutes
def fries_time := 2 * popcorn_time -- Time spent making fries in minutes
def total_time := 4 * 60 -- Total time for cooking and watching movies in minutes

theorem second_movie_time_difference :
  (total_time - (popcorn_time + fries_time + first_movie_length)) - first_movie_length = 30 :=
by
  sorry

end second_movie_time_difference_l221_221368


namespace appropriate_word_count_l221_221757

-- Define the conditions of the problem
def min_minutes := 40
def max_minutes := 55
def words_per_minute := 120

-- Define the bounds for the number of words
def min_words := min_minutes * words_per_minute
def max_words := max_minutes * words_per_minute

-- Define the appropriate number of words
def appropriate_words (words : ℕ) : Prop :=
  words >= min_words ∧ words <= max_words

-- The specific numbers to test
def words1 := 5000
def words2 := 6200

-- The main proof statement
theorem appropriate_word_count : 
  appropriate_words words1 ∧ appropriate_words words2 :=
by
  -- We do not need to provide the proof steps, just state the theorem
  sorry

end appropriate_word_count_l221_221757


namespace josanna_minimum_test_score_l221_221311

def test_scores := [90, 80, 70, 60, 85]

def target_average_increase := 3

def current_average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def sixth_test_score_needed (scores : List ℕ) (increase : ℚ) : ℚ :=
  let current_avg := current_average scores
  let target_avg := current_avg + increase
  target_avg * (scores.length + 1) - scores.sum

theorem josanna_minimum_test_score :
  sixth_test_score_needed test_scores target_average_increase = 95 := sorry

end josanna_minimum_test_score_l221_221311


namespace calculate_A_l221_221954

theorem calculate_A (D B E C A : ℝ) :
  D = 2 * 4 →
  B = 2 * D →
  E = 7 * 2 →
  C = 7 * E →
  A^2 = B * C →
  A = 28 * Real.sqrt 2 :=
by
  sorry

end calculate_A_l221_221954


namespace other_factor_of_LCM_l221_221245

-- Definitions and conditions
def A : ℕ := 624
def H : ℕ := 52 
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Hypotheses based on the problem statement
axiom h_hcf : HCF A 52 = 52

-- The desired statement to prove
theorem other_factor_of_LCM (B : ℕ) (y : ℕ) : HCF A B = H → (A * y = 624) → y = 1 := 
by 
  intro h1 h2
  -- Actual proof steps are omitted
  sorry

end other_factor_of_LCM_l221_221245


namespace sum_of_prime_factors_l221_221991

theorem sum_of_prime_factors (x : ℕ) (h1 : x = 2^10 - 1) 
  (h2 : 2^10 - 1 = (2^5 + 1) * (2^5 - 1)) 
  (h3 : 2^5 - 1 = 31) 
  (h4 : 2^5 + 1 = 33) 
  (h5 : 33 = 3 * 11) : 
  (31 + 3 + 11 = 45) := 
  sorry

end sum_of_prime_factors_l221_221991


namespace minimize_fractions_sum_l221_221718

theorem minimize_fractions_sum {A B C D E : ℕ}
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : A ≠ E)
  (h5 : B ≠ C) (h6 : B ≠ D) (h7 : B ≠ E)
  (h8 : C ≠ D) (h9 : C ≠ E) (h10 : D ≠ E)
  (h11 : A ≠ 9) (h12 : B ≠ 9) (h13 : C ≠ 9) (h14 : D ≠ 9) (h15 : E ≠ 9)
  (hA : 1 ≤ A) (hB : 1 ≤ B) (hC : 1 ≤ C) (hD : 1 ≤ D) (hE : 1 ≤ E)
  (hA' : A ≤ 9) (hB' : B ≤ 9) (hC' : C ≤ 9) (hD' : D ≤ 9) (hE' : E ≤ 9) :
  A / B + C / D + E / 9 = 125 / 168 :=
sorry

end minimize_fractions_sum_l221_221718


namespace linear_eq_k_l221_221114

theorem linear_eq_k (k : ℕ) : (∀ x : ℝ, x^(k-1) + 3 = 0 ↔ k = 2) :=
by
  sorry

end linear_eq_k_l221_221114


namespace k_value_for_inequality_l221_221241

theorem k_value_for_inequality :
    (∀ a b c d : ℝ, a ≥ -1 → b ≥ -1 → c ≥ -1 → d ≥ -1 → a^3 + b^3 + c^3 + d^3 + 1 ≥ (3/4) * (a + b + c + d)) ∧
    (∀ k : ℝ, (∀ a b c d : ℝ, a ≥ -1 → b ≥ -1 → c ≥ -1 → d ≥ -1 → a^3 + b^3 + c^3 + d^3 + 1 ≥ k * (a + b + c + d)) → k = 3/4) :=
sorry

end k_value_for_inequality_l221_221241


namespace goat_age_l221_221625

theorem goat_age : 26 + 42 = 68 := 
by 
  -- Since we only need the statement,
  -- we add sorry to skip the proof.
  sorry

end goat_age_l221_221625


namespace shaded_percentage_correct_l221_221314

def total_squares : ℕ := 6 * 6
def shaded_squares : ℕ := 18
def percentage_shaded (total shaded : ℕ) : ℕ := (shaded * 100) / total

theorem shaded_percentage_correct : percentage_shaded total_squares shaded_squares = 50 := by
  sorry

end shaded_percentage_correct_l221_221314


namespace xiaoying_school_trip_l221_221286

theorem xiaoying_school_trip :
  ∃ (x y : ℝ), 
    (1200 / 1000) = (3 / 60) * x + (5 / 60) * y ∧ 
    x + y = 16 :=
by
  sorry

end xiaoying_school_trip_l221_221286


namespace lcm_12_35_l221_221476

theorem lcm_12_35 : Nat.lcm 12 35 = 420 :=
by
  sorry

end lcm_12_35_l221_221476


namespace fraction_meaningful_l221_221061

theorem fraction_meaningful (x : ℝ) : (x - 2 ≠ 0) ↔ (x ≠ 2) :=
by 
  sorry

end fraction_meaningful_l221_221061


namespace part1_part2_l221_221820

section

variables {x m : ℝ}

def f (x m : ℝ) : ℝ := 3 * x^2 + (4 - m) * x - 6 * m
def g (x m : ℝ) : ℝ := 2 * x^2 - x - m

theorem part1 (m : ℝ) (h : m = 1) : 
  {x : ℝ | f x m > 0} = {x : ℝ | x < -2 ∨ x > 1} :=
sorry

theorem part2 (m : ℝ) (h : m > 0) : 
  {x : ℝ | f x m ≤ g x m} = {x : ℝ | -5 ≤ x ∧ x ≤ m} :=
sorry
     
end

end part1_part2_l221_221820


namespace find_certain_number_l221_221761

theorem find_certain_number (x : ℝ) (h : 34 = (4/5) * x + 14) : x = 25 :=
by
  sorry

end find_certain_number_l221_221761


namespace mario_oranges_l221_221081

theorem mario_oranges (M L N T x : ℕ) 
  (H_L : L = 24) 
  (H_N : N = 96) 
  (H_T : T = 128) 
  (H_total : x + L + N = T) : 
  x = 8 :=
by
  rw [H_L, H_N, H_T] at H_total
  linarith

end mario_oranges_l221_221081


namespace five_students_in_a_row_five_students_with_constraints_five_students_into_three_classes_l221_221644

-- Definition: Number of ways to arrange n items in a row
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Question (1)
theorem five_students_in_a_row : factorial 5 = 120 :=
by sorry

-- Question (2) - Rather than performing combinatorial steps directly, we'll assume a function to calculate the specific arrangement
def specific_arrangement (students: ℕ) : ℕ :=
  if students = 5 then 24 else 0

theorem five_students_with_constraints : specific_arrangement 5 = 24 :=
by sorry

-- Question (3) - Number of ways to divide n students into k classes with at least one student in each class
def number_of_ways_to_divide (students: ℕ) (classes: ℕ) : ℕ :=
  if students = 5 ∧ classes = 3 then 150 else 0

theorem five_students_into_three_classes : number_of_ways_to_divide 5 3 = 150 :=
by sorry

end five_students_in_a_row_five_students_with_constraints_five_students_into_three_classes_l221_221644


namespace average_weight_increase_l221_221530

theorem average_weight_increase
  (A : ℝ) -- Average weight of the two persons
  (w1 : ℝ) (h1 : w1 = 65) -- One person's weight is 65 kg 
  (w2 : ℝ) (h2 : w2 = 74) -- The new person's weight is 74 kg
  :
  ((A * 2 - w1 + w2) / 2 - A = 4.5) :=
by
  simp [h1, h2]
  sorry

end average_weight_increase_l221_221530


namespace num_special_fractions_eq_one_l221_221993

-- Definitions of relatively prime and positive
def are_rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1
def is_positive (n : ℕ) : Prop := n > 0

-- Statement to prove the number of such fractions
theorem num_special_fractions_eq_one : 
  (∀ (x y : ℕ), is_positive x → is_positive y → are_rel_prime x y → 
    (x + 1) * 10 * y = (y + 1) * 11 * x →
    ((x = 5 ∧ y = 11) ∨ False)) := sorry

end num_special_fractions_eq_one_l221_221993


namespace students_not_make_cut_l221_221077

theorem students_not_make_cut (girls boys called_back : ℕ) 
  (h_girls : girls = 42) (h_boys : boys = 80)
  (h_called_back : called_back = 25) : 
  (girls + boys - called_back = 97) := by
  sorry

end students_not_make_cut_l221_221077


namespace farmer_children_l221_221082

theorem farmer_children (n : ℕ) 
  (h1 : 15 * n - 8 - 7 = 60) : n = 5 := 
by
  sorry

end farmer_children_l221_221082


namespace tangent_line_equation_is_correct_l221_221591

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_equation_is_correct :
  let p : ℝ × ℝ := (0, 1)
  let f' := fun x => x * Real.exp x + Real.exp x
  let slope := f' 0
  let tangent_line := fun x y => slope * (x - p.1) - (y - p.2)
  tangent_line = (fun x y => x - y + 1) :=
by
  intros
  sorry

end tangent_line_equation_is_correct_l221_221591


namespace certain_number_k_l221_221424

theorem certain_number_k (x : ℕ) (k : ℕ) (h1 : x = 14) (h2 : 2^x - 2^(x-2) = k * 2^12) : k = 3 := by
  sorry

end certain_number_k_l221_221424


namespace min_cubes_required_l221_221980

/--
A lady builds a box with dimensions 10 cm length, 18 cm width, and 4 cm height using 12 cubic cm cubes. Prove that the minimum number of cubes required to build the box is 60.
-/
def min_cubes_for_box (length width height volume_cube : ℕ) : ℕ :=
  (length * width * height) / volume_cube

theorem min_cubes_required :
  min_cubes_for_box 10 18 4 12 = 60 :=
by
  -- The proof details are omitted.
  sorry

end min_cubes_required_l221_221980


namespace hot_drink_sales_l221_221253

theorem hot_drink_sales (x y : ℝ) (h : y = -2.35 * x + 147.7) (hx : x = 2) : y = 143 := 
by sorry

end hot_drink_sales_l221_221253


namespace max_area_of_region_S_l221_221652

-- Define the radii of the circles
def radii : List ℕ := [2, 4, 6, 8]

-- Define the function for the maximum area of region S given the conditions
def max_area_region_S : ℕ := 75

-- Prove the maximum area of region S is 75π
theorem max_area_of_region_S {radii : List ℕ} (h : radii = [2, 4, 6, 8]) 
: max_area_region_S = 75 := by 
  sorry

end max_area_of_region_S_l221_221652


namespace driver_actual_speed_l221_221673

theorem driver_actual_speed (v t : ℝ) 
  (h1 : t > 0) 
  (h2 : v > 0) 
  (cond : v * t = (v + 18) * (2 / 3 * t)) : 
  v = 36 :=
by 
  sorry

end driver_actual_speed_l221_221673


namespace ninth_term_arith_seq_l221_221934

-- Define the arithmetic sequence.
def arith_seq (a₁ d : ℚ) (n : ℕ) := a₁ + n * d

-- Define the third and fifteenth terms of the sequence.
def third_term := (5 : ℚ) / 11
def fifteenth_term := (7 : ℚ) / 8

-- Prove that the ninth term is 117/176 given the conditions.
theorem ninth_term_arith_seq :
    ∃ (a₁ d : ℚ), 
    arith_seq a₁ d 2 = third_term ∧ 
    arith_seq a₁ d 14 = fifteenth_term ∧
    arith_seq a₁ d 8 = 117 / 176 :=
by
  sorry

end ninth_term_arith_seq_l221_221934


namespace solve_equation_l221_221885

theorem solve_equation (x y z t : ℤ) (h : x^4 - 2*y^4 - 4*z^4 - 8*t^4 = 0) : x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 :=
by
  sorry

end solve_equation_l221_221885


namespace jenny_ate_65_chocolates_l221_221694

-- Define the number of chocolate squares Mike ate
def MikeChoc := 20

-- Define the function that calculates the chocolates Jenny ate
def JennyChoc (mikeChoc : ℕ) := 3 * mikeChoc + 5

-- The theorem stating the solution
theorem jenny_ate_65_chocolates (h : MikeChoc = 20) : JennyChoc MikeChoc = 65 := by
  -- Automatic proof step
  sorry

end jenny_ate_65_chocolates_l221_221694


namespace required_remaining_speed_l221_221849

-- Definitions for the given problem
variables (D T : ℝ) 

-- Given conditions from the problem
def speed_first_part (D T : ℝ) : Prop := 
  40 = (2 * D / 3) / (T / 3)

def remaining_distance_time (D T : ℝ) : Prop :=
  10 = (D / 3) / (2 * (2 * D / 3) / 40 / 3)

-- Theorem to be proved
theorem required_remaining_speed (D T : ℝ) 
  (h1 : speed_first_part D T)
  (h2 : remaining_distance_time D T) :
  10 = (D / 3) / (2 * (T / 3)) :=
  sorry  -- Proof is skipped

end required_remaining_speed_l221_221849


namespace tournament_total_players_l221_221027

theorem tournament_total_players (n : ℕ) (total_points : ℕ) (total_games : ℕ) (half_points : ℕ → ℕ) :
  (∀ k, half_points k * 2 = total_points) ∧ total_points = total_games ∧
  total_points = n * (n + 11) + 132 ∧
  total_games = (n + 12) * (n + 11) / 2 →
  n + 12 = 24 :=
by
  sorry

end tournament_total_players_l221_221027


namespace ratio_of_investments_l221_221960

-- Define the conditions
def ratio_of_profits (p q : ℝ) : Prop := 7/12 = (p * 5) / (q * 12)

-- Define the problem: given the conditions, prove the ratio of investments is 7/5
theorem ratio_of_investments (P Q : ℝ) (h : ratio_of_profits P Q) : P / Q = 7 / 5 :=
by
  sorry

end ratio_of_investments_l221_221960


namespace probability_correct_l221_221497

noncomputable def probability_two_queens_or_at_least_one_jack : ℚ :=
  let total_cards := 52
  let queens := 3
  let jacks := 1
  let prob_two_queens := (queens * (queens - 1)) / (total_cards * (total_cards - 1))
  let prob_one_jack := jacks / total_cards * (total_cards - jacks) / (total_cards - 1) + (total_cards - jacks) / total_cards * jacks / (total_cards - 1)
  prob_two_queens + prob_one_jack

theorem probability_correct : probability_two_queens_or_at_least_one_jack = 9 / 221 := by
  sorry

end probability_correct_l221_221497


namespace mean_proportional_c_l221_221855

theorem mean_proportional_c (a b c : ℝ) (h1 : a = 3) (h2 : b = 27) (h3 : c^2 = a * b) : c = 9 := by
  sorry

end mean_proportional_c_l221_221855


namespace probability_top_card_special_l221_221242

-- Definition of the problem conditions
def deck_size : ℕ := 52
def special_card_count : ℕ := 16

-- The statement we need to prove
theorem probability_top_card_special : 
  (special_card_count : ℚ) / deck_size = 4 / 13 := 
  by sorry

end probability_top_card_special_l221_221242


namespace negate_existential_l221_221905

theorem negate_existential :
  ¬ (∃ x0 : ℝ, x0^2 - 2 * x0 + 4 > 0) ↔ ∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0 :=
by
  sorry

end negate_existential_l221_221905


namespace inequality_solution_set_l221_221749

theorem inequality_solution_set :
  { x : ℝ | -3 < x ∧ x < 2 } = { x : ℝ | abs (x - 1) + abs (x + 2) < 5 } :=
by
  sorry

end inequality_solution_set_l221_221749


namespace fried_chicken_total_l221_221566

-- The Lean 4 statement encapsulates the problem conditions and the correct answer
theorem fried_chicken_total :
  let kobe_initial := 5
  let pau_initial := 2 * kobe_initial
  let another_set := 2
  pau_initial * another_set = 20 :=
by
  let kobe_initial := 5
  let pau_initial := 2 * kobe_initial
  let another_set := 2
  show pau_initial * another_set = 20
  sorry

end fried_chicken_total_l221_221566


namespace find_a9_l221_221134

noncomputable def polynomial_coefficients : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ),
  ∀ (x : ℤ),
    x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 =
    a₀ + a₁ * (1 + x) + a₂ * (1 + x)^2 + a₃ * (1 + x)^3 + a₄ * (1 + x)^4 + 
    a₅ * (1 + x)^5 + a₆ * (1 + x)^6 + a₇ * (1 + x)^7 + a₈ * (1 + x)^8 + 
    a₉ * (1 + x)^9 + a₁₀ * (1 + x)^10

theorem find_a9 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ) (h : polynomial_coefficients) : a₉ = -9 := by
  sorry

end find_a9_l221_221134


namespace find_angle_C_l221_221260

noncomputable def angle_C_value (A B : ℝ) : ℝ :=
  180 - A - B

theorem find_angle_C (A B : ℝ) 
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) :
  angle_C_value A B = 30 :=
sorry

end find_angle_C_l221_221260


namespace three_digit_cubes_divisible_by_8_l221_221396

theorem three_digit_cubes_divisible_by_8 : ∃ (count : ℕ), count = 2 ∧
  ∀ (n : ℤ), (100 ≤ 8 * n^3) ∧ (8 * n^3 ≤ 999) → 
  (8 * n^3 = 216 ∨ 8 * n^3 = 512) := by
  sorry

end three_digit_cubes_divisible_by_8_l221_221396


namespace tank_capacity_l221_221398

theorem tank_capacity (x : ℝ) (h₁ : 0.40 * x = 60) : x = 150 :=
by
  -- a suitable proof would go here
  -- since we are only interested in the statement, we place sorry in place of the proof
  sorry

end tank_capacity_l221_221398


namespace eval_custom_op_l221_221278

def custom_op (a b : ℤ) : ℤ := 2 * b + 5 * a - a^2 - b

theorem eval_custom_op : custom_op 3 4 = 10 :=
by
  sorry

end eval_custom_op_l221_221278


namespace two_point_line_l221_221087

theorem two_point_line (k b : ℝ) (h_k : k ≠ 0) :
  (∀ (x y : ℝ), (y = k * x + b → (x, y) = (0, 0) ∨ (x, y) = (1, 1))) →
  (∀ (x y : ℝ), (y = k * x + b → (x, y) ≠ (2, 0))) :=
by
  sorry

end two_point_line_l221_221087


namespace parabola_vertex_coordinates_l221_221663

theorem parabola_vertex_coordinates :
  ∃ (x y : ℝ), (∀ x : ℝ, y = 3 * x^2 + 2) ∧ x = 0 ∧ y = 2 :=
by
  sorry

end parabola_vertex_coordinates_l221_221663


namespace yeast_cells_at_10_30_l221_221069

def yeast_population (initial_population : ℕ) (intervals : ℕ) (growth_rate : ℝ) (decay_rate : ℝ) : ℝ :=
  initial_population * (growth_rate * (1 - decay_rate)) ^ intervals

theorem yeast_cells_at_10_30 :
  yeast_population 50 6 3 0.10 = 52493 := by
  sorry

end yeast_cells_at_10_30_l221_221069


namespace not_possible_d_count_l221_221748

open Real

theorem not_possible_d_count (t s d : ℝ) (h1 : 3 * t - 4 * s = 1989) (h2 : t - s = d) (h3 : 4 * s > 0) :
  ∃ k : ℕ, k = 663 ∧ ∀ n : ℕ, 1 ≤ n ∧ n ≤ k → d ≠ n :=
by
  sorry

end not_possible_d_count_l221_221748


namespace exists_five_consecutive_divisible_by_2014_l221_221900

theorem exists_five_consecutive_divisible_by_2014 :
  ∃ (a b c d e : ℕ), 53 = a ∧ 54 = b ∧ 55 = c ∧ 56 = d ∧ 57 = e ∧ 100 > a ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ 2014 ∣ (a * b * c * d * e) :=
by 
  sorry

end exists_five_consecutive_divisible_by_2014_l221_221900


namespace geometric_series_sum_l221_221545

noncomputable def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  geometric_sum (2/3) (2/3) 10 = 116050 / 59049 :=
by
  sorry

end geometric_series_sum_l221_221545


namespace number_being_divided_l221_221742

theorem number_being_divided (divisor quotient remainder number : ℕ) 
  (h_divisor : divisor = 3) 
  (h_quotient : quotient = 7) 
  (h_remainder : remainder = 1)
  (h_number : number = divisor * quotient + remainder) : 
  number = 22 :=
by
  rw [h_divisor, h_quotient, h_remainder] at h_number
  exact h_number

end number_being_divided_l221_221742


namespace negation_is_false_l221_221807

-- Define even numbers
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define the original proposition P
def P (a b : ℕ) : Prop := even a ∧ even b → even (a + b)

-- The negation of the proposition P
def notP (a b : ℕ) : Prop := ¬(even a ∧ even b → even (a + b))

-- The theorem to prove
theorem negation_is_false : ∀ a b : ℕ, ¬notP a b :=
by
  sorry

end negation_is_false_l221_221807


namespace additional_money_needed_l221_221358

def original_num_bales : ℕ := 10
def original_cost_per_bale : ℕ := 15
def new_cost_per_bale : ℕ := 18

theorem additional_money_needed :
  (2 * original_num_bales * new_cost_per_bale) - (original_num_bales * original_cost_per_bale) = 210 :=
by
  sorry

end additional_money_needed_l221_221358


namespace crossing_time_l221_221770

-- Define the conditions
def walking_speed_kmh : Float := 10
def bridge_length_m : Float := 1666.6666666666665

-- Convert the man's walking speed to meters per minute
def walking_speed_mpm : Float := walking_speed_kmh * (1000 / 60)

-- State the theorem we want to prove
theorem crossing_time 
  (ws_kmh : Float := walking_speed_kmh)
  (bl_m : Float := bridge_length_m)
  (ws_mpm : Float := walking_speed_mpm) :
  bl_m / ws_mpm = 10 :=
by
  sorry

end crossing_time_l221_221770


namespace point_position_after_time_l221_221264

noncomputable def final_position (initial : ℝ × ℝ) (velocity : ℝ × ℝ) (time : ℝ) : ℝ × ℝ :=
  (initial.1 + velocity.1 * time, initial.2 + velocity.2 * time)

theorem point_position_after_time :
  final_position (-10, 10) (4, -3) 5 = (10, -5) :=
by
  sorry

end point_position_after_time_l221_221264


namespace sin_add_arctan_arcsin_l221_221122

theorem sin_add_arctan_arcsin :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan 3
  (Real.sin a = 4 / 5) →
  (Real.tan b = 3) →
  Real.sin (a + b) = (13 * Real.sqrt 10) / 50 :=
by
  intros _ _
  sorry

end sin_add_arctan_arcsin_l221_221122


namespace constant_chromosome_number_l221_221426

theorem constant_chromosome_number (rabbits : Type) 
  (sex_reproduction : rabbits → Prop)
  (maintain_chromosome_number : Prop)
  (meiosis : Prop)
  (fertilization : Prop) : 
  (meiosis ∧ fertilization) ↔ maintain_chromosome_number :=
sorry

end constant_chromosome_number_l221_221426


namespace boxes_of_apples_with_cherries_l221_221233

-- Define everything in the conditions
variable (A P Sp Sa : ℕ)
variable (box_cherries box_apples : ℕ)

-- Given conditions
axiom price_relation : 2 * P = 3 * A
axiom size_relation  : Sa = 12 * Sp
axiom cherries_per_box : box_cherries = 12

-- The problem statement (to be proved)
theorem boxes_of_apples_with_cherries : box_apples * A = box_cherries * P → box_apples = 18 :=
by
  sorry

end boxes_of_apples_with_cherries_l221_221233


namespace all_propositions_correct_l221_221903

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

theorem all_propositions_correct (m n : ℝ) (a b : α) (h1 : m ≠ 0) (h2 : a ≠ 0) : 
  (∀ (m : ℝ) (a b : α), m • (a - b) = m • a - m • b) ∧
  (∀ (m n : ℝ) (a : α), (m - n) • a = m • a - n • a) ∧
  (∀ (m : ℝ) (a b : α), m • a = m • b → a = b) ∧
  (∀ (m n : ℝ) (a : α), m • a = n • a → m = n) :=
by {
  sorry
}

end all_propositions_correct_l221_221903


namespace sum_repeating_decimals_l221_221403

theorem sum_repeating_decimals : (0.14 + 0.27) = (41 / 99) := by
  sorry

end sum_repeating_decimals_l221_221403


namespace quadratic_nonneg_iff_l221_221774

variable {a b c : ℝ}

theorem quadratic_nonneg_iff :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4 * a * c ≤ 0) :=
by sorry

end quadratic_nonneg_iff_l221_221774


namespace find_mother_age_l221_221727

-- Definitions for the given conditions
def serena_age_now := 9
def years_in_future := 6
def serena_age_future := serena_age_now + years_in_future
def mother_age_future (M : ℕ) := 3 * serena_age_future

-- The main statement to prove
theorem find_mother_age (M : ℕ) (h1 : M = mother_age_future M - years_in_future) : M = 39 :=
by
  sorry

end find_mother_age_l221_221727


namespace max_mark_is_600_l221_221752

-- Define the conditions
def forty_percent (M : ℝ) : ℝ := 0.40 * M
def student_score : ℝ := 175
def additional_marks_needed : ℝ := 65

-- The goal is to prove that the maximum mark is 600
theorem max_mark_is_600 (M : ℝ) :
  forty_percent M = student_score + additional_marks_needed → M = 600 := 
by 
  sorry

end max_mark_is_600_l221_221752


namespace increasing_function_l221_221049

theorem increasing_function (k b : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k + 1) * x1 + b < (2 * k + 1) * x2 + b) ↔ k > -1/2 := 
by
  sorry

end increasing_function_l221_221049


namespace solution_set_for_inequality_l221_221179

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x else x^2 - 2*x - 5

theorem solution_set_for_inequality :
  {x : ℝ | f x >= -2} = {x | -2 <= x ∧ x < 1 ∨ x >= 3} := sorry

end solution_set_for_inequality_l221_221179


namespace pipe_flow_rate_is_correct_l221_221527

-- Definitions for the conditions
def tank_capacity : ℕ := 10000
def initial_water : ℕ := tank_capacity / 2
def fill_time : ℕ := 60
def drain1_rate : ℕ := 1000
def drain1_interval : ℕ := 4
def drain2_rate : ℕ := 1000
def drain2_interval : ℕ := 6

-- Calculation based on conditions
def total_water_needed : ℕ := tank_capacity - initial_water
def drain1_loss (time : ℕ) : ℕ := (time / drain1_interval) * drain1_rate
def drain2_loss (time : ℕ) : ℕ := (time / drain2_interval) * drain2_rate
def total_drain_loss (time : ℕ) : ℕ := drain1_loss time + drain2_loss time

-- Target flow rate for the proof
def total_fill (time : ℕ) : ℕ := total_water_needed + total_drain_loss time
def pipe_flow_rate : ℕ := total_fill fill_time / fill_time

-- Statement to prove
theorem pipe_flow_rate_is_correct : pipe_flow_rate = 500 := by  
  sorry

end pipe_flow_rate_is_correct_l221_221527


namespace rectangle_measurement_error_l221_221937

theorem rectangle_measurement_error
  (L W : ℝ)
  (x : ℝ)
  (h1 : ∀ x, L' = L * (1 + x / 100))
  (h2 : W' = W * 0.9)
  (h3 : A = L * W)
  (h4 : A' = A * 1.08) :
  x = 20 :=
by
  sorry

end rectangle_measurement_error_l221_221937


namespace hyperbola_through_point_has_asymptotes_l221_221310

-- Definitions based on condition (1)
def hyperbola_asymptotes (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Definition of the problem
def hyperbola_eqn (x y : ℝ) : Prop := (x^2 / 5) - (y^2 / 20) = 1

-- Main statement including all conditions and proving the correct answer
theorem hyperbola_through_point_has_asymptotes :
  ∀ x y : ℝ, hyperbola_eqn x y ↔ (hyperbola_asymptotes x y ∨ (x, y) = (-3, 4)) :=
by
  -- The proof part is skipped with sorry
  sorry

end hyperbola_through_point_has_asymptotes_l221_221310


namespace sum_of_angles_equal_360_l221_221067

variables (A B C D F G : ℝ)

-- Given conditions.
def is_quadrilateral_interior_sum (A B C D : ℝ) : Prop := A + B + C + D = 360
def split_internal_angles (F G : ℝ) (C D : ℝ) : Prop := F + G = C + D

-- Proof problem statement.
theorem sum_of_angles_equal_360
  (h1 : is_quadrilateral_interior_sum A B C D)
  (h2 : split_internal_angles F G C D) :
  A + B + C + D + F + G = 360 :=
sorry

end sum_of_angles_equal_360_l221_221067


namespace center_of_symmetry_l221_221942

def symmetry_center (f : ℝ → ℝ) (p : ℝ × ℝ) :=
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f x

/--
  Given the function f(x) := sin x - sqrt(3) * cos x,
  prove that (π/3, 0) is the center of symmetry for f.
-/
theorem center_of_symmetry : symmetry_center (fun x => Real.sin x - Real.sqrt 3 * Real.cos x) (Real.pi / 3, 0) :=
by
  sorry

end center_of_symmetry_l221_221942


namespace hyperbola_eccentricity_correct_l221_221495

noncomputable def hyperbola_eccentricity : ℝ := 2

variables {a b : ℝ}
variables (ha_pos : 0 < a) (hb_pos : 0 < b)
variables (h_hyperbola : ∃ x y, x^2/a^2 - y^2/b^2 = 1)
variables (h_circle_chord_len : ∃ d, d = 2 ∧ ∃ x y, ((x - 2)^2 + y^2 = 4) ∧ (x * b/a = -y))

theorem hyperbola_eccentricity_correct :
  ∀ (a b : ℝ), 0 < a → 0 < b → (∃ x y, x^2 / a^2 - y^2 / b^2 = 1) 
  ∧ (∃ d, d = 2 ∧ ∃ x y, (x - 2)^2 + y^2 = 4 ∧ (x * b / a = -y)) →
  (eccentricity = 2) :=
by
  intro a b ha_pos hb_pos h_conditions
  have e := hyperbola_eccentricity
  sorry


end hyperbola_eccentricity_correct_l221_221495


namespace initial_workers_count_l221_221039

theorem initial_workers_count (W : ℕ) 
  (h1 : W * 30 = W * 30) 
  (h2 : W * 15 = (W - 5) * 20)
  (h3 : W > 5) 
  : W = 20 :=
by {
  sorry
}

end initial_workers_count_l221_221039


namespace hiring_manager_acceptance_l221_221385

theorem hiring_manager_acceptance 
    (average_age : ℤ) (std_dev : ℤ) (num_ages : ℤ)
    (applicant_ages_are_int : ∀ (x : ℤ), x ≥ (average_age - std_dev) ∧ x ≤ (average_age + std_dev)) :
    (∃ k : ℤ, (average_age + k * std_dev) - (average_age - k * std_dev) + 1 = num_ages) → k = 1 :=
by 
  intros h
  sorry

end hiring_manager_acceptance_l221_221385


namespace red_crayons_count_l221_221266

variable (R : ℕ) -- Number of red crayons
variable (B : ℕ) -- Number of blue crayons
variable (Y : ℕ) -- Number of yellow crayons

-- Conditions
axiom h1 : B = R + 5
axiom h2 : Y = 2 * B - 6
axiom h3 : Y = 32

-- Statement to prove
theorem red_crayons_count : R = 14 :=
by
  sorry

end red_crayons_count_l221_221266


namespace find_a_l221_221438

def f (a : ℝ) (x : ℝ) := a * x^2 + 3 * x - 2

theorem find_a (a : ℝ) (h : deriv (f a) 2 = 7) : a = 1 :=
by {
  sorry
}

end find_a_l221_221438


namespace number_of_zeros_of_f_l221_221376

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then 4 * Real.exp x - 2
else abs (2 - Real.log x / Real.log 2)

theorem number_of_zeros_of_f :
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = Real.log (1 / 2) ∧ x₂ = 4 :=
by
  sorry

end number_of_zeros_of_f_l221_221376


namespace system_of_equations_a_solution_l221_221627

theorem system_of_equations_a_solution (x y a : ℝ) (h1 : 4 * x + y = a) (h2 : 3 * x + 4 * y^2 = 3 * a) (hx : x = 3) : a = 15 ∨ a = 9.75 :=
by
  sorry

end system_of_equations_a_solution_l221_221627


namespace number_line_problem_l221_221778

theorem number_line_problem (x : ℤ) (h : x + 7 - 4 = 0) : x = -3 :=
by
  -- The proof is omitted as only the statement is required.
  sorry

end number_line_problem_l221_221778


namespace sum_of_three_largest_l221_221879

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l221_221879


namespace value_of_a_l221_221846

theorem value_of_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y = 0 → 3 * x + y + a = 0) → a = 1 :=
by
  sorry

end value_of_a_l221_221846


namespace basketball_game_first_half_points_l221_221589

theorem basketball_game_first_half_points (a b r d : ℕ) (H1 : a = b)
  (H2 : a * (1 + r + r^2 + r^3) = 4 * a + 6 * d + 1) 
  (H3 : 15 * a ≤ 100) (H4 : b + (b + d) + b + 2 * d + b + 3 * d < 100) : 
  (a + a * r + b + b + d) = 34 :=
by sorry

end basketball_game_first_half_points_l221_221589


namespace total_oranges_l221_221602

theorem total_oranges :
  let capacity_box1 := 80
  let capacity_box2 := 50
  let fullness_box1 := (3/4 : ℚ)
  let fullness_box2 := (3/5 : ℚ)
  let oranges_box1 := fullness_box1 * capacity_box1
  let oranges_box2 := fullness_box2 * capacity_box2
  oranges_box1 + oranges_box2 = 90 := 
by
  sorry

end total_oranges_l221_221602


namespace last_integer_in_sequence_is_21853_l221_221470

def is_divisible_by (n m : ℕ) : Prop := 
  ∃ k : ℕ, n = m * k

-- Conditions
def starts_with : ℕ := 590049
def divides_previous (a b : ℕ) : Prop := b = a / 3

-- The target hypothesis to prove
theorem last_integer_in_sequence_is_21853 :
  ∀ (a b c d : ℕ),
    a = starts_with →
    divides_previous a b →
    divides_previous b c →
    divides_previous c d →
    ¬ is_divisible_by d 3 →
    d = 21853 :=
by
  intros a b c d ha hb hc hd hnd
  sorry

end last_integer_in_sequence_is_21853_l221_221470


namespace find_number_type_l221_221729

-- Definitions of the problem conditions
def consecutive (a b c d : ℤ) : Prop := (b = a + 2) ∧ (c = a + 4) ∧ (d = a + 6)
def sum_is_52 (a b c d : ℤ) : Prop := a + b + c + d = 52
def third_number_is_14 (c : ℤ) : Prop := c = 14

-- The proof problem statement
theorem find_number_type (a b c d : ℤ) 
                         (h1 : consecutive a b c d) 
                         (h2 : sum_is_52 a b c d) 
                         (h3 : third_number_is_14 c) :
  (∃ (k : ℤ), a = 2 * k ∧ b = 2 * k + 2 ∧ c = 2 * k + 4 ∧ d = 2 * k + 6) 
  := sorry

end find_number_type_l221_221729


namespace sum_modulo_9_l221_221100

theorem sum_modulo_9 : 
  (88000 + 88002 + 87999 + 88001 + 88003 + 87998) % 9 = 0 := 
by
  sorry

end sum_modulo_9_l221_221100


namespace intersection_points_l221_221598

theorem intersection_points (a : ℝ) (h : 2 < a) :
  (∃ n : ℕ, (n = 1 ∨ n = 2) ∧ (∃ x1 x2 : ℝ, y = (a-3)*x^2 - x - 1/4 ∧ x1 ≠ x2)) :=
sorry

end intersection_points_l221_221598


namespace product_eval_at_3_l221_221561

theorem product_eval_at_3 : (3 - 2) * (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) = 720 := by
  sorry

end product_eval_at_3_l221_221561


namespace arithmetic_sequence_y_value_l221_221151

theorem arithmetic_sequence_y_value (y : ℝ) (h₁ : 2 * y - 3 = -5 * y + 11) : y = 2 := by
  sorry

end arithmetic_sequence_y_value_l221_221151


namespace correct_calculation_l221_221505

theorem correct_calculation (x : ℕ) (h : x / 9 = 30) : x - 37 = 233 :=
by sorry

end correct_calculation_l221_221505


namespace perpendicular_vectors_l221_221330

theorem perpendicular_vectors (k : ℝ) (a b : ℝ × ℝ) 
  (ha : a = (0, 2)) 
  (hb : b = (Real.sqrt 3, 1)) 
  (h : (a.1 - k * b.1) * (k * a.1 + b.1) + (a.2 - k * b.2) * (k * a.2 + b.2) = 0) :
  k = -1 ∨ k = 1 :=
sorry

end perpendicular_vectors_l221_221330


namespace gcd_sequence_condition_l221_221677

theorem gcd_sequence_condition (p q : ℕ) (hp : 0 < p) (hq : 0 < q)
  (a : ℕ → ℕ)
  (ha1 : a 1 = 1) (ha2 : a 2 = 1) 
  (ha_rec : ∀ n, a (n + 2) = p * a (n + 1) + q * a n) 
  (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (gcd (a m) (a n) = a (gcd m n)) ↔ (p = 1) := 
sorry

end gcd_sequence_condition_l221_221677


namespace find_c_l221_221902

theorem find_c (a b c : ℚ) (h1 : ∀ y : ℚ, 1 = a * (3 - 1)^2 + b * (3 - 1) + c) (h2 : ∀ y : ℚ, 4 = a * (1)^2 + b * (1) + c)
  (h3 : ∀ y : ℚ, 1 = a * (y - 1)^2 + 4) : c = 13 / 4 :=
by
  sorry

end find_c_l221_221902


namespace cinema_meeting_day_l221_221593

-- Define the cycles for Kolya, Seryozha, and Vanya.
def kolya_cycle : ℕ := 4
def seryozha_cycle : ℕ := 5
def vanya_cycle : ℕ := 6

-- The problem statement requiring proof.
theorem cinema_meeting_day : ∃ n : ℕ, n > 0 ∧ n % kolya_cycle = 0 ∧ n % seryozha_cycle = 0 ∧ n % vanya_cycle = 0 ∧ n = 60 := 
  sorry

end cinema_meeting_day_l221_221593


namespace cos_C_max_ab_over_c_l221_221092

theorem cos_C_max_ab_over_c
  (a b c S : ℝ) (A B C : ℝ)
  (h1 : 6 * S = a^2 * Real.sin A + b^2 * Real.sin B)
  (h2 : a / Real.sin A = b / Real.sin B)
  (h3 : b / Real.sin B = c / Real.sin C)
  (h4 : S = 0.5 * a * b * Real.sin C)
  : Real.cos C = 7 / 9 := 
sorry

end cos_C_max_ab_over_c_l221_221092


namespace problem_1_problem_2_l221_221103

theorem problem_1 
  : (∃ (m n : ℝ), m = -1 ∧ n = 1 ∧ ∀ (x : ℝ), |x + 1| + |2 * x - 1| ≤ 3 ↔ m ≤ x ∧ x ≤ n) :=
sorry

theorem problem_2 
  : (∀ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 → 
    ∃ (min_val : ℝ), min_val = 9 / 2 ∧ 
    ∀ (x : ℝ), x = (1 / a + 1 / b + 1 / c) → min_val ≤ x) :=
sorry

end problem_1_problem_2_l221_221103


namespace num_small_triangles_l221_221794

-- Define the lengths of the legs of the large and small triangles
variables (a h b k : ℕ)

-- Define the areas of the large and small triangles
def area_large_triangle (a h : ℕ) : ℕ := (a * h) / 2
def area_small_triangle (b k : ℕ) : ℕ := (b * k) / 2

-- Define the main theorem
theorem num_small_triangles (ha : a = 6) (hh : h = 4) (hb : b = 2) (hk : k = 1) :
  (area_large_triangle a h) / (area_small_triangle b k) = 12 :=
by
  sorry

end num_small_triangles_l221_221794


namespace solution_set_of_inequality_l221_221578

theorem solution_set_of_inequality (x : ℝ) : 
  (x * |x - 1| > 0) ↔ ((0 < x ∧ x < 1) ∨ (x > 1)) := 
by
  sorry

end solution_set_of_inequality_l221_221578


namespace inv_mod_35_l221_221331

theorem inv_mod_35 : ∃ x : ℕ, 5 * x ≡ 1 [MOD 35] :=
by
  use 29
  sorry

end inv_mod_35_l221_221331


namespace part1_part2_part3_l221_221680

-- Definitions from the problem
def initial_cost_per_bottle := 16
def initial_selling_price := 20
def initial_sales_volume := 60
def sales_decrease_per_yuan_increase := 5

def daily_sales_volume (x : ℕ) : ℕ :=
  initial_sales_volume - sales_decrease_per_yuan_increase * x

def profit_per_bottle (x : ℕ) : ℕ :=
  (initial_selling_price - initial_cost_per_bottle) + x

def daily_profit (x : ℕ) : ℕ :=
  daily_sales_volume x * profit_per_bottle x

-- The proofs we need to establish
theorem part1 (x : ℕ) : 
  daily_sales_volume x = 60 - 5 * x ∧ profit_per_bottle x = 4 + x :=
sorry

theorem part2 (x : ℕ) : 
  daily_profit x = 300 → x = 6 ∨ x = 2 :=
sorry

theorem part3 : 
  ∃ x : ℕ, ∀ y : ℕ, (daily_profit x < daily_profit y) → 
              (daily_profit x = 320 ∧ x = 4) :=
sorry

end part1_part2_part3_l221_221680


namespace sum_infinite_series_l221_221096

theorem sum_infinite_series : 
  ∑' n : ℕ, (2 * (n + 1) + 3) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2) * ((n + 1) + 3)) = 9 / 4 := by
  sorry

end sum_infinite_series_l221_221096


namespace wenlock_olympian_games_first_held_year_difference_l221_221768

theorem wenlock_olympian_games_first_held_year_difference :
  2012 - 1850 = 162 :=
sorry

end wenlock_olympian_games_first_held_year_difference_l221_221768


namespace units_digit_problem_l221_221029

open BigOperators

-- Define relevant constants
def A : ℤ := 21
noncomputable def B : ℤ := 14 -- since B = sqrt(196) = 14

-- Define the terms
noncomputable def term1 : ℤ := (A + B) ^ 20
noncomputable def term2 : ℤ := (A - B) ^ 20

-- Statement of the theorem
theorem units_digit_problem :
  ((term1 - term2) % 10) = 4 := 
sorry

end units_digit_problem_l221_221029


namespace find_a_if_y_is_even_l221_221562

noncomputable def y (x a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem find_a_if_y_is_even (a : ℝ) (h : ∀ x : ℝ, y x a = y (-x) a) : a = 2 :=
by
  sorry

end find_a_if_y_is_even_l221_221562


namespace haley_initial_trees_l221_221728

theorem haley_initial_trees (dead_trees trees_left initial_trees : ℕ) 
    (h_dead: dead_trees = 2)
    (h_left: trees_left = 10)
    (h_initial: initial_trees = trees_left + dead_trees) : 
    initial_trees = 12 := 
by sorry

end haley_initial_trees_l221_221728


namespace value_of_u_when_m_is_3_l221_221009

theorem value_of_u_when_m_is_3 :
  ∀ (u t m : ℕ), (t = 3^m + m) → (u = 4^t - 3 * t) → m = 3 → u = 4^30 - 90 :=
by
  intros u t m ht hu hm
  sorry

end value_of_u_when_m_is_3_l221_221009


namespace roots_of_polynomial_l221_221696

def poly (x : ℝ) : ℝ := x^3 - 3 * x^2 - 4 * x + 12

theorem roots_of_polynomial : 
  (poly 2 = 0) ∧ (poly (-2) = 0) ∧ (poly 3 = 0) ∧ 
  (∀ x, poly x = 0 → x = 2 ∨ x = -2 ∨ x = 3) :=
by
  sorry

end roots_of_polynomial_l221_221696


namespace five_aliens_have_more_limbs_than_five_martians_l221_221813

-- Definitions based on problem conditions

def number_of_alien_arms : ℕ := 3
def number_of_alien_legs : ℕ := 8

-- Martians have twice as many arms as Aliens and half as many legs
def number_of_martian_arms : ℕ := 2 * number_of_alien_arms
def number_of_martian_legs : ℕ := number_of_alien_legs / 2

-- Total limbs for five aliens and five martians
def total_limbs_for_aliens (n : ℕ) : ℕ := n * (number_of_alien_arms + number_of_alien_legs)
def total_limbs_for_martians (n : ℕ) : ℕ := n * (number_of_martian_arms + number_of_martian_legs)

-- The theorem to prove
theorem five_aliens_have_more_limbs_than_five_martians :
  total_limbs_for_aliens 5 - total_limbs_for_martians 5 = 5 :=
sorry

end five_aliens_have_more_limbs_than_five_martians_l221_221813


namespace sum_of_fractions_l221_221329

theorem sum_of_fractions : (1 / 3 : ℚ) + (2 / 7) = 13 / 21 :=
by
  sorry

end sum_of_fractions_l221_221329


namespace quadratic_inequality_solution_l221_221781

theorem quadratic_inequality_solution :
  { x : ℝ | x^2 + 7*x + 6 < 0 } = { x : ℝ | -6 < x ∧ x < -1 } :=
by sorry

end quadratic_inequality_solution_l221_221781


namespace kite_area_correct_l221_221251

open Real

structure Point where
  x : ℝ
  y : ℝ

def Kite (p1 p2 p3 p4 : Point) : Prop :=
  let triangle_area (a b c : Point) : ℝ :=
    abs (0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)))
  triangle_area p1 p2 p4 + triangle_area p1 p3 p4 = 102

theorem kite_area_correct : ∃ (p1 p2 p3 p4 : Point), 
  p1 = Point.mk 0 10 ∧ 
  p2 = Point.mk 6 14 ∧ 
  p3 = Point.mk 12 10 ∧ 
  p4 = Point.mk 6 0 ∧ 
  Kite p1 p2 p3 p4 :=
by
  sorry

end kite_area_correct_l221_221251


namespace find_g_neg2_l221_221017

-- Definitions of the conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x 

variables (f : ℝ → ℝ) (g : ℝ → ℝ)
variables (h_even_f : even_function f)
variables (h_g_def : ∀ x, g x = f x + x^3)
variables (h_g_2 : g 2 = 10)

-- Statement to prove
theorem find_g_neg2 : g (-2) = -6 :=
sorry

end find_g_neg2_l221_221017


namespace is_opposite_if_differ_in_sign_l221_221681

-- Define opposite numbers based on the given condition in the problem:
def opposite_numbers (a b : ℝ) : Prop := a = -b

-- State the theorem based on the translation in c)
theorem is_opposite_if_differ_in_sign (a b : ℝ) (h : a = -b) : opposite_numbers a b := by
  sorry

end is_opposite_if_differ_in_sign_l221_221681


namespace least_number_of_attendees_l221_221275

-- Definitions based on problem conditions
inductive Person
| Anna
| Bill
| Carl
deriving DecidableEq

inductive Day
| Mon
| Tues
| Wed
| Thurs
| Fri
deriving DecidableEq

def attends : Person → Day → Prop
| Person.Anna, Day.Mon => true
| Person.Anna, Day.Tues => false
| Person.Anna, Day.Wed => true
| Person.Anna, Day.Thurs => false
| Person.Anna, Day.Fri => false
| Person.Bill, Day.Mon => false
| Person.Bill, Day.Tues => true
| Person.Bill, Day.Wed => false
| Person.Bill, Day.Thurs => true
| Person.Bill, Day.Fri => true
| Person.Carl, Day.Mon => true
| Person.Carl, Day.Tues => true
| Person.Carl, Day.Wed => false
| Person.Carl, Day.Thurs => true
| Person.Carl, Day.Fri => false

-- Proof statement
theorem least_number_of_attendees : 
  (∀ d : Day, (∀ p : Person, attends p d → p = Person.Anna ∨ p = Person.Bill ∨ p = Person.Carl) ∧
              (d = Day.Wed ∨ d = Day.Fri → (∃ n : ℕ, n = 2 ∧ (∀ p : Person, attends p d → n = 2))) ∧
              (d = Day.Mon ∨ d = Day.Tues ∨ d = Day.Thurs → (∃ n : ℕ, n = 1 ∧ (∀ p : Person, attends p d → n = 1))) ∧
              ¬ (d = Day.Wed ∨ d = Day.Fri)) :=
sorry

end least_number_of_attendees_l221_221275


namespace quadratic_coefficients_l221_221185

theorem quadratic_coefficients : 
  ∀ (b k : ℝ), (∀ x : ℝ, x^2 + b * x + 5 = (x - 2)^2 + k) → b = -4 ∧ k = 1 :=
by
  intro b k h
  have h1 := h 0
  have h2 := h 1
  sorry

end quadratic_coefficients_l221_221185


namespace parabola_x_intercepts_l221_221467

theorem parabola_x_intercepts :
  ∃! (x : ℝ), ∃ (y : ℝ), y = 0 ∧ x = -2 * y^2 + y + 1 :=
sorry

end parabola_x_intercepts_l221_221467


namespace cos_alpha_minus_pi_over_2_l221_221689

theorem cos_alpha_minus_pi_over_2 (α : ℝ) 
  (h1 : ∃ k : ℤ, α = k * (2 * Real.pi) ∨ α = k * (2 * Real.pi) + Real.pi / 2 ∨ α = k * (2 * Real.pi) + Real.pi ∨ α = k * (2 * Real.pi) + 3 * Real.pi / 2)
  (h2 : Real.cos α = 4 / 5)
  (h3 : Real.sin α = -3 / 5) : 
  Real.cos (α - Real.pi / 2) = -3 / 5 := 
by 
  sorry

end cos_alpha_minus_pi_over_2_l221_221689


namespace largest_percentage_increase_l221_221078

def student_count (year: ℕ) : ℝ :=
  match year with
  | 2010 => 80
  | 2011 => 88
  | 2012 => 95
  | 2013 => 100
  | 2014 => 105
  | 2015 => 112
  | _    => 0  -- Because we only care about 2010-2015

noncomputable def percentage_increase (year1 year2 : ℕ) : ℝ :=
  ((student_count year2 - student_count year1) / student_count year1) * 100

theorem largest_percentage_increase :
  (∀ x y, percentage_increase 2010 2011 ≥ percentage_increase x y) :=
by sorry

end largest_percentage_increase_l221_221078


namespace exact_days_two_friends_visit_l221_221217

-- Define the periodicities of Alice, Beatrix, and Claire
def periodicity_alice : ℕ := 1
def periodicity_beatrix : ℕ := 5
def periodicity_claire : ℕ := 7

-- Define the total days to be considered
def total_days : ℕ := 180

-- Define the number of days three friends visit together
def lcm_ab := Nat.lcm periodicity_alice periodicity_beatrix
def lcm_ac := Nat.lcm periodicity_alice periodicity_claire
def lcm_bc := Nat.lcm periodicity_beatrix periodicity_claire
def lcm_abc := Nat.lcm lcm_ab periodicity_claire

-- Define the counts of visitations
def count_ab := total_days / lcm_ab - total_days / lcm_abc
def count_ac := total_days / lcm_ac - total_days / lcm_abc
def count_bc := total_days / lcm_bc - total_days / lcm_abc

-- Finally calculate the number of days exactly two friends visit together
def days_two_friends_visit : ℕ := count_ab + count_ac + count_bc

-- The theorem to prove
theorem exact_days_two_friends_visit : days_two_friends_visit = 51 :=
by 
  -- This is where the actual proof would go
  sorry

end exact_days_two_friends_visit_l221_221217


namespace seating_arrangement_l221_221733

theorem seating_arrangement :
  let total_arrangements := Nat.factorial 8
  let adjacent_arrangements := Nat.factorial 7 * 2
  total_arrangements - adjacent_arrangements = 30240 :=
by
  sorry

end seating_arrangement_l221_221733


namespace exists_x_such_that_f_x_eq_0_l221_221456

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then
  3 * x - 4
else
  -x^2 + 3 * x - 5

theorem exists_x_such_that_f_x_eq_0 :
  ∃ x : ℝ, f x = 0 ∧ x = 1.192 :=
sorry

end exists_x_such_that_f_x_eq_0_l221_221456


namespace incorrect_statement_l221_221859

open Set

theorem incorrect_statement 
  (M : Set ℝ := {x : ℝ | 0 < x ∧ x < 1})
  (N : Set ℝ := {y : ℝ | 0 < y})
  (R : Set ℝ := univ) : M ∪ N ≠ R :=
by
  sorry

end incorrect_statement_l221_221859


namespace interval_necessary_not_sufficient_l221_221957

theorem interval_necessary_not_sufficient :
  (∀ x, x^2 - x - 2 = 0 → (-1 ≤ x ∧ x ≤ 2)) ∧ (∃ x, x^2 - x - 2 = 0 ∧ ¬(-1 ≤ x ∧ x ≤ 2)) → False :=
by
  sorry

end interval_necessary_not_sufficient_l221_221957


namespace sufficient_condition_l221_221603

theorem sufficient_condition (a b : ℝ) (h1 : a > 1) (h2 : b > 1) : ab > 1 :=
sorry

end sufficient_condition_l221_221603


namespace find_angle_l221_221452

-- Definitions based on conditions
def is_complement (x : ℝ) : ℝ := 90 - x
def is_supplement (x : ℝ) : ℝ := 180 - x

-- Main statement
theorem find_angle (x : ℝ) (h : is_supplement x = 15 + 4 * is_complement x) : x = 65 :=
by
  sorry

end find_angle_l221_221452


namespace probability_two_dice_sum_seven_l221_221732

theorem probability_two_dice_sum_seven (z : ℕ) (w : ℚ) (h : z = 2) : w = 1 / 6 :=
by sorry

end probability_two_dice_sum_seven_l221_221732


namespace sum_of_all_possible_values_l221_221739

theorem sum_of_all_possible_values (x y : ℝ) (h : x * y - x^2 - y^2 = 4) :
  (x - 2) * (y - 2) = 4 :=
sorry

end sum_of_all_possible_values_l221_221739


namespace number_of_sides_l221_221567

-- Define the given conditions as Lean definitions

def exterior_angle := 72
def sum_of_exterior_angles := 360

-- Now state the theorem based on these conditions

theorem number_of_sides (n : ℕ) (h1 : exterior_angle = 72) (h2 : sum_of_exterior_angles = 360) : n = 5 :=
by
  sorry

end number_of_sides_l221_221567


namespace total_distance_traveled_is_correct_l221_221609

-- Definitions of given conditions
def Vm : ℕ := 8
def Vr : ℕ := 2
def round_trip_time : ℝ := 1

-- Definitions needed for intermediate calculations (speed computations)
def upstream_speed (Vm Vr : ℕ) : ℕ := Vm - Vr
def downstream_speed (Vm Vr : ℕ) : ℕ := Vm + Vr

-- The equation representing the total time for the round trip
def time_equation (D : ℝ) (Vm Vr : ℕ) : Prop :=
  D / upstream_speed Vm Vr + D / downstream_speed Vm Vr = round_trip_time

-- Prove that the total distance traveled by the man is 7.5 km
theorem total_distance_traveled_is_correct : ∃ D : ℝ, D / upstream_speed Vm Vr + D / downstream_speed Vm Vr = round_trip_time ∧ 2 * D = 7.5 :=
by
  sorry

end total_distance_traveled_is_correct_l221_221609


namespace pocket_knife_worth_40_l221_221839

def value_of_pocket_knife (x : ℕ) (p : ℕ) (R : ℕ) : Prop :=
  p = 10 * x ∧
  R = 10 * x^2 ∧
  (∃ num_100_bills : ℕ, 2 * num_100_bills * 100 + 40 = R)

theorem pocket_knife_worth_40 (x : ℕ) (p : ℕ) (R : ℕ) :
  value_of_pocket_knife x p R → (∃ knife_value : ℕ, knife_value = 40) :=
by
  sorry

end pocket_knife_worth_40_l221_221839


namespace negate_neg_two_l221_221480

theorem negate_neg_two : -(-2) = 2 := by
  -- The proof goes here
  sorry

end negate_neg_two_l221_221480


namespace average_temperature_for_july_4th_l221_221976

def avg_temperature_july_4th : ℤ := 
  let temperatures := [90, 90, 90, 79, 71]
  let sum := List.sum temperatures
  sum / temperatures.length

theorem average_temperature_for_july_4th :
  avg_temperature_july_4th = 84 := 
by
  sorry

end average_temperature_for_july_4th_l221_221976


namespace sum_of_digits_is_twenty_l221_221232

theorem sum_of_digits_is_twenty (a b c d : ℕ) (h1 : c + b = 9) (h2 : a + d = 10) 
  (H1 : a ≠ b) (H2 : a ≠ c) (H3 : a ≠ d) 
  (H4 : b ≠ c) (H5 : b ≠ d) (H6 : c ≠ d) :
  a + b + c + d = 20 := 
sorry

end sum_of_digits_is_twenty_l221_221232


namespace solve_for_x_l221_221529

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 17) : x = 4 := 
by sorry

end solve_for_x_l221_221529


namespace sum_fractions_l221_221935

theorem sum_fractions :
  (1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) + 1 / (8 * 9)) = (2 / 9) :=
by
  sorry

end sum_fractions_l221_221935


namespace rubber_ball_radius_l221_221325

theorem rubber_ball_radius (r : ℝ) (radius_exposed_section : ℝ) (depth : ℝ) 
  (h1 : radius_exposed_section = 20) 
  (h2 : depth = 12) 
  (h3 : (r - depth)^2 + radius_exposed_section^2 = r^2) : 
  r = 22.67 :=
by
  sorry

end rubber_ball_radius_l221_221325


namespace main_theorem_l221_221614

-- Define even functions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define odd functions
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : is_even_function f)
variable (h2 : is_odd_function g)
variable (h3 : ∀ x, g x = f (x - 1))

-- Theorem to prove
theorem main_theorem : f 2017 + f 2019 = 0 := sorry

end main_theorem_l221_221614


namespace quadratic_roots_identity_l221_221573

theorem quadratic_roots_identity (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0) (hmn : m * n = -5) (hm_plus_n : m + n = -2) : m^2 + m * n + 2 * m = 0 :=
by {
    sorry
}

end quadratic_roots_identity_l221_221573


namespace median_computation_l221_221038

noncomputable def length_of_median (A B C A1 P Q R : ℝ) : Prop :=
  let AB := 10
  let AC := 6
  let BC := Real.sqrt (AB^2 - AC^2)
  let A1C := 24 / 7
  let A1B := 32 / 7
  let QR := Real.sqrt (A1B^2 - A1C^2)
  let median_length := QR / 2
  median_length = 4 * Real.sqrt 7 / 7

theorem median_computation (A B C A1 P Q R : ℝ) :
  length_of_median A B C A1 P Q R := by
  sorry

end median_computation_l221_221038


namespace perpendicular_vectors_l221_221789

theorem perpendicular_vectors (x : ℝ) : (2 * x + 3 = 0) → (x = -3 / 2) :=
by
  intro h
  sorry

end perpendicular_vectors_l221_221789


namespace ladder_distance_l221_221388

theorem ladder_distance (x : ℝ) (h1 : (13:ℝ) = Real.sqrt (x ^ 2 + 12 ^ 2)) : 
  x = 5 :=
by 
  sorry

end ladder_distance_l221_221388


namespace symmetric_line_equation_l221_221629

theorem symmetric_line_equation :
  (∃ line : ℝ → ℝ, ∀ x y, x + 2 * y - 3 = 0 → line 1 = 1 ∧ (∃ b, line 0 = b → x - 2 * y + 1 = 0)) :=
sorry

end symmetric_line_equation_l221_221629


namespace number_of_positive_expressions_l221_221492

-- Define the conditions
variable (a b c : ℝ)
variable (h_a : a < 0)
variable (h_b : b > 0)
variable (h_c : c < 0)

-- Define the expressions
def ab := a * b
def ac := a * c
def a_b_c := a + b + c
def a_minus_b_c := a - b + c
def two_a_plus_b := 2 * a + b
def two_a_minus_b := 2 * a - b

-- Problem statement
theorem number_of_positive_expressions :
  (ab < 0) → (ac > 0) → (a_b_c > 0) → (a_minus_b_c < 0) → (two_a_plus_b < 0) → (two_a_minus_b < 0)
  → (2 = 2) :=
by
  sorry

end number_of_positive_expressions_l221_221492


namespace gcd_of_items_l221_221693

theorem gcd_of_items :
  ∀ (plates spoons glasses bowls : ℕ),
  plates = 3219 →
  spoons = 5641 →
  glasses = 1509 →
  bowls = 2387 →
  Nat.gcd (Nat.gcd (Nat.gcd plates spoons) glasses) bowls = 1 :=
by
  intros plates spoons glasses bowls
  intros Hplates Hspoons Hglasses Hbowls
  rw [Hplates, Hspoons, Hglasses, Hbowls]
  sorry

end gcd_of_items_l221_221693


namespace correct_calculation_l221_221725

theorem correct_calculation (a : ℝ) :
  2 * a^4 * 3 * a^5 = 6 * a^9 :=
by
  sorry

end correct_calculation_l221_221725


namespace functional_equation_solution_l221_221348

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)) ↔ (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = 1 - x) :=
sorry

end functional_equation_solution_l221_221348


namespace sum_of_fractions_l221_221008

theorem sum_of_fractions :
  (2 / 8) + (4 / 8) + (6 / 8) + (8 / 8) + (10 / 8) + 
  (12 / 8) + (14 / 8) + (16 / 8) + (18 / 8) + (20 / 8) = 13.75 :=
by sorry

end sum_of_fractions_l221_221008


namespace chimney_problem_l221_221025

variable (x : ℕ) -- number of bricks in the chimney
variable (t : ℕ)
variables (brenda_hours brandon_hours : ℕ)

def brenda_rate := x / brenda_hours
def brandon_rate := x / brandon_hours
def combined_rate := (brenda_rate + brandon_rate - 15) * t

theorem chimney_problem (h1 : brenda_hours = 9)
    (h2 : brandon_hours = 12)
    (h3 : t = 6)
    (h4 : combined_rate = x) : x = 540 := sorry

end chimney_problem_l221_221025


namespace good_numbers_100_2010_ex_good_and_not_good_x_y_l221_221697

-- Definition of a good number
def is_good_number (n : ℤ) : Prop := ∃ a b : ℤ, n = a^2 + 161 * b^2

-- (1) Prove 100 and 2010 are good numbers
theorem good_numbers_100_2010 : is_good_number 100 ∧ is_good_number 2010 :=
by sorry

-- (2) Prove there exist positive integers x and y such that x^161 + y^161 is a good number, 
-- but x + y is not a good number
theorem ex_good_and_not_good_x_y : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ is_good_number (x^161 + y^161) ∧ ¬ is_good_number (x + y) :=
by sorry

end good_numbers_100_2010_ex_good_and_not_good_x_y_l221_221697


namespace perpendicular_lines_iff_a_eq_1_l221_221982

theorem perpendicular_lines_iff_a_eq_1 :
  ∀ a : ℝ, (∀ x y, (y = a * x + 1) → (y = (a - 2) * x - 1) → (a = 1)) ↔ (a = 1) :=
by sorry

end perpendicular_lines_iff_a_eq_1_l221_221982


namespace incorrect_statement_B_l221_221270

-- Define the plane vector operation "☉".
def vector_operation (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

-- Define the mathematical problem based on the given conditions.
theorem incorrect_statement_B (a b : ℝ × ℝ) : vector_operation a b ≠ vector_operation b a := by
  sorry

end incorrect_statement_B_l221_221270


namespace smallest_digit_d_l221_221144

theorem smallest_digit_d (d : ℕ) (hd : d < 10) :
  (∃ d, (20 - (8 + d)) % 11 = 0 ∧ d < 10) → d = 1 :=
by
  sorry

end smallest_digit_d_l221_221144


namespace evaluate_expression_l221_221619

noncomputable def x : ℚ := 4 / 7
noncomputable def y : ℚ := 6 / 8

theorem evaluate_expression : (7 * x + 8 * y) / (56 * x * y) = 5 / 12 := by
  sorry

end evaluate_expression_l221_221619


namespace slope_of_line_between_intersections_of_circles_l221_221202

theorem slope_of_line_between_intersections_of_circles :
  ∀ C D : ℝ × ℝ, 
    -- Conditions: equations of the circles
    (C.1^2 + C.2^2 - 6 * C.1 + 4 * C.2 - 8 = 0) ∧ (C.1^2 + C.2^2 - 8 * C.1 - 2 * C.2 + 10 = 0) →
    (D.1^2 + D.2^2 - 6 * D.1 + 4 * D.2 - 8 = 0) ∧ (D.1^2 + D.2^2 - 8 * D.1 - 2 * D.2 + 10 = 0) →
    -- Question: slope of line CD
    ((C.2 - D.2) / (C.1 - D.1) = -1 / 3) :=
by
  sorry

end slope_of_line_between_intersections_of_circles_l221_221202


namespace linear_function_value_l221_221465

theorem linear_function_value (g : ℝ → ℝ) (h_linear : ∀ x y, g (x + y) = g x + g y)
  (h_scale : ∀ c x, g (c * x) = c * g x) (h : g 10 - g 0 = 20) : g 20 - g 0 = 40 :=
by
  sorry

end linear_function_value_l221_221465


namespace radius_distance_relation_l221_221569

variables {A B C : Point} (Γ₁ Γ₂ ω₀ : Circle)
variables (ω : ℕ → Circle)
variables (r d : ℕ → ℝ)

def diam_circle (P Q : Point) : Circle := sorry  -- This is to define a circle with diameter PQ
def tangent (κ κ' κ'' : Circle) : Prop := sorry  -- This is to define that three circles are mutually tangent

-- Defining the properties as given in the conditions
axiom Γ₁_def : Γ₁ = diam_circle A B
axiom Γ₂_def : Γ₂ = diam_circle A C
axiom ω₀_def : ω₀ = diam_circle B C
axiom ω_def : ∀ n : ℕ, tangent (if n = 0 then ω₀ else ω (n - 1)) Γ₁ (ω n) ∧ tangent (if n = 0 then ω₀ else ω (n - 1)) Γ₂ (ω n) -- ωₙ is tangent to previous circle, Γ₁ and Γ₂

-- The main proof statement
theorem radius_distance_relation (n : ℕ) : r n = 2 * n * d n :=
sorry

end radius_distance_relation_l221_221569


namespace brian_expenses_l221_221568

def cost_apples_per_bag : ℕ := 14
def cost_kiwis : ℕ := 10
def cost_bananas : ℕ := cost_kiwis / 2
def subway_fare_one_way : ℕ := 350
def maximum_apples : ℕ := 24

theorem brian_expenses : 
  cost_kiwis + cost_bananas + (cost_apples_per_bag * (maximum_apples / 12)) + (subway_fare_one_way * 2) = 50 := by
sorry

end brian_expenses_l221_221568


namespace mary_lambs_count_l221_221758

def initial_lambs : Nat := 6
def baby_lambs : Nat := 2 * 2
def traded_lambs : Nat := 3
def extra_lambs : Nat := 7

theorem mary_lambs_count : initial_lambs + baby_lambs - traded_lambs + extra_lambs = 14 := by
  sorry

end mary_lambs_count_l221_221758


namespace sum_of_perimeters_geq_4400_l221_221952

theorem sum_of_perimeters_geq_4400 (side original_side : ℕ) 
  (h_side_le_10 : ∀ s, s ≤ side → s ≤ 10) 
  (h_original_square : original_side = 100) 
  (h_cut_condition : side ≤ 10) : 
  ∃ (small_squares : ℕ → ℕ × ℕ), (original_side / side = n) → 4 * n * side ≥ 4400 :=
by
  sorry

end sum_of_perimeters_geq_4400_l221_221952


namespace find_value_of_x_l221_221805

theorem find_value_of_x (a b c d e f x : ℕ) (h1 : a ≠ 1 ∧ a ≠ 6 ∧ b ≠ 1 ∧ b ≠ 6 ∧ c ≠ 1 ∧ c ≠ 6 ∧ d ≠ 1 ∧ d ≠ 6 ∧ e ≠ 1 ∧ e ≠ 6 ∧ f ≠ 1 ∧ f ≠ 6 ∧ x ≠ 1 ∧ x ≠ 6)
  (h2 : a + x + d = 18)
  (h3 : b + x + f = 18)
  (h4 : c + x + 6 = 18)
  (h5 : a + b + c + d + e + f + x + 6 + 1 = 45) :
  x = 7 :=
sorry

end find_value_of_x_l221_221805


namespace trigonometric_identity_l221_221391

theorem trigonometric_identity (x : ℝ) (h : (1 + Real.sin x) / Real.cos x = -1/2) : 
  Real.cos x / (Real.sin x - 1) = 1/2 := 
sorry

end trigonometric_identity_l221_221391


namespace union_eq_l221_221161

def A : Set ℤ := {-1, 0, 3}
def B : Set ℤ := {-1, 1, 2, 3}

theorem union_eq : A ∪ B = {-1, 0, 1, 2, 3} := 
by 
  sorry

end union_eq_l221_221161


namespace tens_digit_6_pow_45_l221_221137

theorem tens_digit_6_pow_45 : (6 ^ 45 % 100) / 10 = 0 := 
by 
  sorry

end tens_digit_6_pow_45_l221_221137


namespace cost_of_each_entree_l221_221804

def cost_of_appetizer : ℝ := 10
def number_of_entrees : ℝ := 4
def tip_percentage : ℝ := 0.20
def total_spent : ℝ := 108

theorem cost_of_each_entree :
  ∃ E : ℝ, total_spent = cost_of_appetizer + number_of_entrees * E + tip_percentage * (cost_of_appetizer + number_of_entrees * E) ∧ E = 20 :=
by
  sorry

end cost_of_each_entree_l221_221804


namespace sum_of_squares_eq_frac_squared_l221_221945

theorem sum_of_squares_eq_frac_squared (x y z a b c : ℝ) (hxya : x * y = a) (hxzb : x * z = b) (hyzc : y * z = c)
  (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hz0 : z ≠ 0) (ha0 : a ≠ 0) (hb0 : b ≠ 0) (hc0 : c ≠ 0) :
  x^2 + y^2 + z^2 = ((a * b)^2 + (a * c)^2 + (b * c)^2) / (a * b * c) :=
by
  sorry

end sum_of_squares_eq_frac_squared_l221_221945


namespace ticket_cost_l221_221261

theorem ticket_cost (total_amount_collected : ℕ) (average_tickets_per_day : ℕ) (days : ℕ) 
  (h1 : total_amount_collected = 960) 
  (h2 : average_tickets_per_day = 80) 
  (h3 : days = 3) : 
  total_amount_collected / (average_tickets_per_day * days) = 4 :=
  sorry

end ticket_cost_l221_221261


namespace greatest_m_div_36_and_7_l221_221488

def reverse_digits (m : ℕ) : ℕ :=
  let d1 := (m / 1000) % 10
  let d2 := (m / 100) % 10
  let d3 := (m / 10) % 10
  let d4 := m % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem greatest_m_div_36_and_7
  (m : ℕ) (n : ℕ := reverse_digits m)
  (h1 : 1000 ≤ m ∧ m < 10000)
  (h2 : 1000 ≤ n ∧ n < 10000)
  (h3 : 36 ∣ m ∧ 36 ∣ n)
  (h4 : 7 ∣ m) :
  m = 9828 := 
sorry

end greatest_m_div_36_and_7_l221_221488


namespace simplify_expression1_simplify_expression2_l221_221401

variable {a b x y : ℝ}

theorem simplify_expression1 : 3 * a - 5 * b - 2 * a + b = a - 4 * b :=
by sorry

theorem simplify_expression2 : 4 * x^2 + 5 * x * y - 2 * (2 * x^2 - x * y) = 7 * x * y :=
by sorry

end simplify_expression1_simplify_expression2_l221_221401


namespace find_amplitude_l221_221868

-- Conditions
variables (a b c d : ℝ)

theorem find_amplitude
  (h1 : ∀ x, a * Real.sin (b * x + c) + d ≤ 5)
  (h2 : ∀ x, a * Real.sin (b * x + c) + d ≥ -3) :
  a = 4 :=
by 
  sorry

end find_amplitude_l221_221868


namespace common_tangent_and_inequality_l221_221585

noncomputable def f (x : ℝ) := Real.log (1 + x)
noncomputable def g (x : ℝ) := x - (1 / 2) * x^2 + (1 / 3) * x^3

theorem common_tangent_and_inequality :
  -- Condition: common tangent at (0, 0)
  (∀ x, deriv f x = deriv g x) →
  -- Condition: values of a and b found to be 0 and 1 respectively
  (∀ x, f x ≤ g x) :=
by
  intro h
  sorry

end common_tangent_and_inequality_l221_221585


namespace right_triangle_inequality_equality_condition_l221_221305

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  3 * a + 4 * b ≤ 5 * c :=
by 
  sorry

theorem equality_condition (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  3 * a + 4 * b = 5 * c ↔ a / b = 3 / 4 :=
by
  sorry

end right_triangle_inequality_equality_condition_l221_221305


namespace probability_no_two_boys_same_cinema_l221_221486

-- Definitions
def total_cinemas := 10
def total_boys := 7

def total_arrangements : ℕ := total_cinemas ^ total_boys
def favorable_arrangements : ℕ := 10 * 9 * 8 * 7 * 6 * 5 * 4
def probability := (favorable_arrangements : ℚ) / total_arrangements

-- Mathematical proof problem
theorem probability_no_two_boys_same_cinema : 
  probability = 0.06048 := 
by {
  sorry -- Proof goes here
}

end probability_no_two_boys_same_cinema_l221_221486


namespace region_area_l221_221514

theorem region_area (x y : ℝ) : 
  (x^2 + y^2 + 14 * x + 18 * y = 0) → 
  (π * 130) = 130 * π :=
by 
  sorry

end region_area_l221_221514


namespace number_added_l221_221798

theorem number_added (x y : ℝ) (h1 : x = 33) (h2 : x / 4 + y = 15) : y = 6.75 :=
by sorry

end number_added_l221_221798


namespace at_least_one_no_less_than_two_l221_221953

variable (a b c : ℝ)
variable (ha : 0 < a)
variable (hb : 0 < b)
variable (hc : 0 < c)

theorem at_least_one_no_less_than_two :
  ∃ x ∈ ({a + 1/b, b + 1/c, c + 1/a} : Set ℝ), 2 ≤ x := by
  sorry

end at_least_one_no_less_than_two_l221_221953


namespace operation_value_l221_221387

-- Define the operations as per the conditions.
def star (m n : ℤ) : ℤ := n^2 - m
def hash (m k : ℤ) : ℚ := (k + 2 * m) / 3

-- State the theorem we want to prove.
theorem operation_value : hash (star 3 3) (star 2 5) = 35 / 3 :=
  by
  sorry

end operation_value_l221_221387


namespace translated_graph_pass_through_origin_l221_221974

theorem translated_graph_pass_through_origin 
    (φ : ℝ) (h : 0 < φ ∧ φ < π / 2) 
    (passes_through_origin : 0 = Real.sin (-2 * φ + π / 3)) : 
    φ = π / 6 := 
sorry

end translated_graph_pass_through_origin_l221_221974


namespace boat_downstream_distance_l221_221252

theorem boat_downstream_distance 
  (Vb Vr T D U : ℝ)
  (h1 : Vb + Vr = 21)
  (h2 : Vb - Vr = 12)
  (h3 : U = 48)
  (h4 : T = 4)
  (h5 : D = 20) :
  (Vb + Vr) * D = 420 :=
by
  sorry

end boat_downstream_distance_l221_221252


namespace arc_length_of_sector_l221_221923

theorem arc_length_of_sector (n r : ℝ) (h_angle : n = 60) (h_radius : r = 3) : 
  (n * Real.pi * r / 180) = Real.pi :=
by 
  sorry

end arc_length_of_sector_l221_221923


namespace prove_x_value_l221_221383

-- Definitions of the conditions
variable (x y z w : ℕ)
variable (h1 : x = y + 8)
variable (h2 : y = z + 15)
variable (h3 : z = w + 25)
variable (h4 : w = 90)

-- The goal is to prove x = 138 given the conditions
theorem prove_x_value : x = 138 := by
  sorry

end prove_x_value_l221_221383


namespace modulus_of_complex_l221_221930

open Complex

theorem modulus_of_complex (z : ℂ) (h : (1 + z) / (1 - z) = ⟨0, 1⟩) : abs z = 1 := 
sorry

end modulus_of_complex_l221_221930


namespace ab_value_l221_221707

theorem ab_value (a b : ℤ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 50) : a * b = 7 := by
  sorry

end ab_value_l221_221707


namespace x_cubed_inverse_cubed_l221_221753

theorem x_cubed_inverse_cubed (x : ℝ) (hx : x + 1/x = 3) : x^3 + 1/x^3 = 18 :=
by
  sorry

end x_cubed_inverse_cubed_l221_221753


namespace range_of_a_max_area_of_triangle_l221_221042

variable (p a : ℝ) (h : p > 0)

def parabola_eq (x y : ℝ) := y ^ 2 = 2 * p * x
def line_eq (x y : ℝ) := y = x - a
def intersects_parabola (A B : ℝ × ℝ) := parabola_eq p A.fst A.snd ∧ line_eq a A.fst A.snd ∧ parabola_eq p B.fst B.snd ∧ line_eq a B.fst B.snd
def ab_length_le_2p (A B : ℝ × ℝ) := (Real.sqrt ((A.fst - B.fst)^2 + (A.snd - B.snd)^2) ≤ 2 * p)

theorem range_of_a
  (A B : ℝ × ℝ)
  (h_intersects : intersects_parabola a p A B)
  (h_ab_length : ab_length_le_2p p A B) :
  - p / 2 < a ∧ a ≤ - p / 4 := sorry

theorem max_area_of_triangle
  (A B : ℝ × ℝ) (N : ℝ × ℝ)
  (h_intersects : intersects_parabola a p A B)
  (h_ab_length : ab_length_le_2p p A B)
  (h_N : N.snd = 0) :
  ∃ (S : ℝ), S = Real.sqrt 2 * p^2 := sorry

end range_of_a_max_area_of_triangle_l221_221042


namespace train_cross_time_l221_221906

theorem train_cross_time (length_train : ℝ) (length_bridge : ℝ) (speed_kmph : ℝ) : 
  length_train = 100 →
  length_bridge = 150 →
  speed_kmph = 63 →
  (length_train + length_bridge) / (speed_kmph * (1000 / 3600)) = 14.29 :=
by
  sorry

end train_cross_time_l221_221906


namespace even_product_when_eight_cards_drawn_l221_221913

theorem even_product_when_eight_cards_drawn :
  ∀ (s : Finset ℕ), (∀ n ∈ s, n ∈ Finset.range 15) →
  s.card ≥ 8 →
  (∃ m ∈ s, Even m) :=
by
  sorry

end even_product_when_eight_cards_drawn_l221_221913


namespace gloria_money_left_l221_221489

theorem gloria_money_left 
  (cost_of_cabin : ℕ) (cash : ℕ)
  (num_cypress_trees num_pine_trees num_maple_trees : ℕ)
  (price_per_cypress_tree price_per_pine_tree price_per_maple_tree : ℕ)
  (money_left : ℕ)
  (h_cost_of_cabin : cost_of_cabin = 129000)
  (h_cash : cash = 150)
  (h_num_cypress_trees : num_cypress_trees = 20)
  (h_num_pine_trees : num_pine_trees = 600)
  (h_num_maple_trees : num_maple_trees = 24)
  (h_price_per_cypress_tree : price_per_cypress_tree = 100)
  (h_price_per_pine_tree : price_per_pine_tree = 200)
  (h_price_per_maple_tree : price_per_maple_tree = 300)
  (h_money_left : money_left = (num_cypress_trees * price_per_cypress_tree + 
                                num_pine_trees * price_per_pine_tree + 
                                num_maple_trees * price_per_maple_tree + 
                                cash) - cost_of_cabin)
  : money_left = 350 :=
by
  sorry

end gloria_money_left_l221_221489


namespace steve_bought_3_boxes_of_cookies_l221_221938

variable (total_cost : ℝ)
variable (milk_cost : ℝ)
variable (cereal_cost : ℝ)
variable (banana_cost : ℝ)
variable (apple_cost : ℝ)
variable (chicken_cost : ℝ)
variable (peanut_butter_cost : ℝ)
variable (bread_cost : ℝ)
variable (cookie_box_cost : ℝ)
variable (cookie_box_count : ℝ)

noncomputable def proves_steve_cookie_boxes : Prop :=
  total_cost = 50 ∧
  milk_cost = 4 ∧
  cereal_cost = 3 ∧
  banana_cost = 0.2 ∧
  apple_cost = 0.75 ∧
  chicken_cost = 10 ∧
  peanut_butter_cost = 5 ∧
  bread_cost = (2 * cereal_cost) / 2 ∧
  cookie_box_cost = (milk_cost + peanut_butter_cost) / 3 ∧
  cookie_box_count = (total_cost - (milk_cost + 3 * cereal_cost + 6 * banana_cost + 8 * apple_cost + chicken_cost + peanut_butter_cost + bread_cost)) / cookie_box_cost

theorem steve_bought_3_boxes_of_cookies :
  proves_steve_cookie_boxes 50 4 3 0.2 0.75 10 5 3 ((4 + 5) / 3) 3 :=
by
  sorry

end steve_bought_3_boxes_of_cookies_l221_221938


namespace basic_spatial_data_source_l221_221014

def source_of_basic_spatial_data (s : String) : Prop :=
  s = "Detailed data provided by high-resolution satellite remote sensing technology" ∨
  s = "Data from various databases provided by high-speed networks" ∨
  s = "Various data collected and organized through the information highway" ∨
  s = "Various spatial exchange data provided by GIS"

theorem basic_spatial_data_source :
  source_of_basic_spatial_data "Data from various databases provided by high-speed networks" :=
sorry

end basic_spatial_data_source_l221_221014


namespace set_operations_l221_221532

def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { x | 0 < x ∧ x < 5 }
def U : Set ℝ := Set.univ  -- Universal set ℝ
def complement (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem set_operations :
  (A ∩ B = { x | 0 < x ∧ x < 2 }) ∧ 
  (complement A ∪ B = { x | 0 < x }) :=
by {
  sorry
}

end set_operations_l221_221532


namespace largest_square_side_length_l221_221746

theorem largest_square_side_length (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) : 
  ∃ x : ℝ, x = (a * b) / (a + b) := 
sorry

end largest_square_side_length_l221_221746


namespace sum_possible_values_l221_221463

theorem sum_possible_values (N : ℤ) (h : N * (N - 8) = -7) : 
  ∀ (N1 N2 : ℤ), (N1 * (N1 - 8) = -7) ∧ (N2 * (N2 - 8) = -7) → (N1 + N2 = 8) :=
by
  sorry

end sum_possible_values_l221_221463


namespace geometric_sequence_term_l221_221288

theorem geometric_sequence_term (a : ℕ → ℕ) (q : ℕ) (hq : q = 2) (ha2 : a 2 = 8) :
  a 6 = 128 :=
by
  sorry

end geometric_sequence_term_l221_221288


namespace bad_carrots_l221_221711

theorem bad_carrots (carol_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) (total_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : mom_carrots = 16)
  (h3 : good_carrots = 38)
  (h4 : total_carrots = carol_carrots + mom_carrots)
  (h5 : bad_carrots = total_carrots - good_carrots) :
  bad_carrots = 7 := by
  sorry

end bad_carrots_l221_221711


namespace sasha_remaining_questions_l221_221053

theorem sasha_remaining_questions
  (qph : ℕ) (total_questions : ℕ) (hours_worked : ℕ)
  (h_qph : qph = 15) (h_total_questions : total_questions = 60) (h_hours_worked : hours_worked = 2) :
  total_questions - (qph * hours_worked) = 30 :=
by
  sorry

end sasha_remaining_questions_l221_221053


namespace train_times_comparison_l221_221257

-- Defining the given conditions
variables (V1 T1 T2 D : ℝ)
variables (h1 : T1 = 2) (h2 : T2 = 7/3)
variables (train1_speed : V1 = D / T1)
variables (train2_speed : V2 = (3/5) * V1)

-- The proof statement to show that T2 is 1/3 hour longer than T1
theorem train_times_comparison 
  (h1 : (6/7) * V1 = D / (T1 + 1/3))
  (h2 : (3/5) * V1 = D / (T2 + 1)) :
  T2 - T1 = 1/3 :=
sorry

end train_times_comparison_l221_221257


namespace remaining_student_number_l221_221280

-- Definitions based on given conditions
def total_students := 48
def sample_size := 6
def sampled_students := [5, 21, 29, 37, 45]

-- Interval calculation and pattern definition based on systematic sampling
def sampling_interval := total_students / sample_size
def sampled_student_numbers (n : Nat) : Nat := 5 + sampling_interval * (n - 1)

-- Prove the student number within the sample
theorem remaining_student_number : ∃ n, n ∉ sampled_students ∧ sampled_student_numbers n = 13 :=
by
  sorry

end remaining_student_number_l221_221280


namespace total_apples_picked_l221_221208

def number_of_children : Nat := 33
def apples_per_child : Nat := 10
def number_of_adults : Nat := 40
def apples_per_adult : Nat := 3

theorem total_apples_picked :
  (number_of_children * apples_per_child) + (number_of_adults * apples_per_adult) = 450 := by
  -- You need to provide proof here
  sorry

end total_apples_picked_l221_221208


namespace initial_pens_l221_221551

-- Conditions as definitions
def initial_books := 108
def books_after_sale := 66
def books_sold := 42
def pens_after_sale := 59

-- Theorem statement proving the initial number of pens
theorem initial_pens:
  initial_books - books_after_sale = books_sold →
  ∃ (P : ℕ), P - pens_sold = pens_after_sale ∧ (P = 101) :=
by
  sorry

end initial_pens_l221_221551


namespace plane_eq_of_point_and_parallel_l221_221381

theorem plane_eq_of_point_and_parallel (A B C D : ℤ) 
  (h1 : A = 3) (h2 : B = -2) (h3 : C = 4) 
  (point : ℝ × ℝ × ℝ) (hpoint : point = (2, -3, 5))
  (h4 : 3 * (2 : ℝ) - 2 * (-3 : ℝ) + 4 * (5 : ℝ) + (D : ℝ) = 0)
  (hD : D = -32)
  (hGCD : Int.gcd (Int.natAbs 3) (Int.gcd (Int.natAbs (-2)) (Int.gcd (Int.natAbs 4) (Int.natAbs (-32)))) = 1) : 
  3 * (x : ℝ) - 2 * (y : ℝ) + 4 * (z : ℝ) - 32 = 0 :=
sorry

end plane_eq_of_point_and_parallel_l221_221381


namespace find_sr_division_l221_221449

theorem find_sr_division (k : ℚ) (c r s : ℚ)
  (h_c : c = 10)
  (h_r : r = -3 / 10)
  (h_s : s = 191 / 10)
  (h_expr : 10 * k^2 - 6 * k + 20 = c * (k + r)^2 + s) :
  s / r = -191 / 3 :=
by
  sorry

end find_sr_division_l221_221449


namespace Alan_ate_1_fewer_pretzel_than_John_l221_221224

/-- Given that there are 95 pretzels in a bowl, John ate 28 pretzels, 
Marcus ate 12 more pretzels than John, and Marcus ate 40 pretzels,
prove that Alan ate 1 fewer pretzel than John. -/
theorem Alan_ate_1_fewer_pretzel_than_John 
  (h95 : 95 = 95)
  (John_ate : 28 = 28)
  (Marcus_ate_more : ∀ (x : ℕ), 40 = x + 12 → x = 28)
  (Marcus_ate : 40 = 40) :
  ∃ (Alan : ℕ), Alan = 27 ∧ 28 - Alan = 1 :=
by
  sorry

end Alan_ate_1_fewer_pretzel_than_John_l221_221224


namespace min_value_of_a3b2c_l221_221132

theorem min_value_of_a3b2c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1 / a + 1 / b + 1 / c = 9) : 
  a^3 * b^2 * c ≥ 1 / 2916 :=
by 
  sorry

end min_value_of_a3b2c_l221_221132


namespace problem_l221_221961

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

noncomputable def f_deriv (x : ℝ) : ℝ := - (1 / x^2) * Real.cos x - (1 / x) * Real.sin x

theorem problem (h_pi_ne_zero : Real.pi ≠ 0) (h_pi_div_two_ne_zero : Real.pi / 2 ≠ 0) :
  f Real.pi + f_deriv (Real.pi / 2) = -3 / Real.pi  := by
  sorry

end problem_l221_221961


namespace alyssa_hike_total_distance_l221_221520

theorem alyssa_hike_total_distance
  (e f g h i : ℝ)
  (h1 : e + f + g = 40)
  (h2 : f + g + h = 48)
  (h3 : g + h + i = 54)
  (h4 : e + h = 30) :
  e + f + g + h + i = 118 :=
by
  sorry

end alyssa_hike_total_distance_l221_221520


namespace ratio_frogs_to_dogs_l221_221002

variable (D C F : ℕ)

-- Define the conditions as given in the problem statement
def cats_eq_dogs_implied : Prop := C = Nat.div (4 * D) 5
def frogs : Prop := F = 160
def total_animals : Prop := D + C + F = 304

-- Define the statement to be proved
theorem ratio_frogs_to_dogs (h1 : cats_eq_dogs_implied D C) (h2 : frogs F) (h3 : total_animals D C F) : F / D = 2 := by
  sorry

end ratio_frogs_to_dogs_l221_221002


namespace range_of_g_l221_221263

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos x)^2 - (Real.arcsin x)^2

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -((Real.pi^2) / 4) ≤ g x ∧ g x ≤ (3 * (Real.pi^2)) / 4 :=
by
  intros x hx
  sorry

end range_of_g_l221_221263


namespace min_value_of_fraction_l221_221862

theorem min_value_of_fraction (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) 
    (h3 : (m * (-3) + n * (-1) + 2 = 0)) 
    (h4 : (m * (-2) + n * 0 + 2 = 0)) : 
    (1 / m + 3 / n) = 6 :=
by
  sorry

end min_value_of_fraction_l221_221862


namespace trigonometric_identity_l221_221610

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 1 / 2) : 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 :=
by
  sorry

end trigonometric_identity_l221_221610


namespace largest_fraction_among_list_l221_221716

theorem largest_fraction_among_list :
  ∃ (f : ℚ), f = 105 / 209 ∧ 
  (f > 5 / 11) ∧ 
  (f > 9 / 20) ∧ 
  (f > 23 / 47) ∧ 
  (f > 205 / 409) := 
by
  sorry

end largest_fraction_among_list_l221_221716


namespace find_xy_yz_xz_l221_221390

noncomputable def xy_yz_xz (x y z : ℝ) : ℝ := x * y + y * z + x * z

theorem find_xy_yz_xz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 + x * y + y^2 = 48) (h2 : y^2 + y * z + z^2 = 16) (h3 : z^2 + x * z + x^2 = 64) :
  xy_yz_xz x y z = 32 :=
sorry

end find_xy_yz_xz_l221_221390


namespace sequence_a_b_10_l221_221574

theorem sequence_a_b_10 (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7) (h5 : a^5 + b^5 = 11) : 
  a^10 + b^10 = 123 := 
sorry

end sequence_a_b_10_l221_221574


namespace solve_for_x_l221_221898

theorem solve_for_x (x : ℂ) (h : 5 - 2 * I * x = 4 - 5 * I * x) : x = I / 3 :=
by
  sorry

end solve_for_x_l221_221898


namespace common_volume_of_tetrahedra_l221_221454

open Real

noncomputable def volume_of_common_part (a b c : ℝ) : ℝ :=
  min (a * sqrt 3 / 12) (min (b * sqrt 3 / 12) (c * sqrt 3 / 12))

theorem common_volume_of_tetrahedra (a b c : ℝ) :
  volume_of_common_part a b c =
  min (a * sqrt 3 / 12) (min (b * sqrt 3 / 12) (c * sqrt 3 / 12)) :=
by sorry

end common_volume_of_tetrahedra_l221_221454


namespace simple_interest_principal_l221_221156

theorem simple_interest_principal (R T SI : ℝ) (hR : R = 9 / 100) (hT : T = 1) (hSI : SI = 900) : 
  (SI / (R * T) = 10000) :=
by
  sorry

end simple_interest_principal_l221_221156


namespace simplify_expression_l221_221323

variable (x y : ℤ)

theorem simplify_expression : 
  (15 * x + 45 * y) + (7 * x + 18 * y) - (6 * x + 35 * y) = 16 * x + 28 * y :=
by
  sorry

end simplify_expression_l221_221323


namespace december_19th_day_l221_221021

theorem december_19th_day (december_has_31_days : true)
  (december_1st_is_monday : true)
  (day_of_week : ℕ → ℕ) :
  day_of_week 19 = 5 :=
sorry

end december_19th_day_l221_221021


namespace width_of_metallic_sheet_l221_221453

theorem width_of_metallic_sheet 
  (length : ℕ)
  (new_volume : ℕ) 
  (side_length_of_square : ℕ)
  (height_of_box : ℕ)
  (new_length : ℕ)
  (new_width : ℕ)
  (w : ℕ) : 
  length = 48 → 
  new_volume = 5120 → 
  side_length_of_square = 8 → 
  height_of_box = 8 → 
  new_length = length - 2 * side_length_of_square → 
  new_width = w - 2 * side_length_of_square → 
  new_volume = new_length * new_width * height_of_box → 
  w = 36 := 
by 
  intros _ _ _ _ _ _ _ 
  sorry

end width_of_metallic_sheet_l221_221453


namespace inequality_solution_intervals_l221_221080

theorem inequality_solution_intervals (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (x / (x - 1) + (x + 3) / (2 * x) ≥ 4) ↔ (0 < x ∧ x < 1) := 
sorry

end inequality_solution_intervals_l221_221080


namespace eggs_in_each_basket_l221_221553

theorem eggs_in_each_basket (n : ℕ) (h1 : 30 % n = 0) (h2 : 42 % n = 0) (h3 : n ≥ 5) :
  n = 6 :=
by sorry

end eggs_in_each_basket_l221_221553


namespace ratio_of_length_to_width_l221_221536

-- Definitions of conditions
def width := 5
def area := 75

-- Theorem statement proving the ratio is 3
theorem ratio_of_length_to_width {l : ℕ} (h1 : l * width = area) : l / width = 3 :=
by sorry

end ratio_of_length_to_width_l221_221536


namespace book_price_l221_221070

theorem book_price (n p : ℕ) (h : n * p = 104) (hn : 10 < n ∧ n < 60) : p = 2 ∨ p = 4 ∨ p = 8 :=
sorry

end book_price_l221_221070


namespace geometric_sequence_min_l221_221300

theorem geometric_sequence_min (a : ℕ → ℝ) (q : ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_condition : 2 * (a 4) + (a 3) - 2 * (a 2) - (a 1) = 8)
  (h_geometric : ∀ n, a (n+1) = a n * q) :
  ∃ min_val, min_val = 12 * Real.sqrt 3 ∧ min_val = 2 * (a 5) + (a 4) :=
sorry

end geometric_sequence_min_l221_221300


namespace length_of_uncovered_side_l221_221066

variables (L W : ℝ)

-- Conditions
def area_eq_680 := (L * W = 680)
def fence_eq_178 := (2 * W + L = 178)

-- Theorem statement to prove the length of the uncovered side
theorem length_of_uncovered_side (h1 : area_eq_680 L W) (h2 : fence_eq_178 L W) : L = 170 := 
sorry

end length_of_uncovered_side_l221_221066


namespace polynomial_non_negative_for_all_real_iff_l221_221979

theorem polynomial_non_negative_for_all_real_iff (a : ℝ) :
  (∀ x : ℝ, x^4 + (a - 1) * x^2 + 1 ≥ 0) ↔ a ≥ -1 :=
by sorry

end polynomial_non_negative_for_all_real_iff_l221_221979


namespace initial_population_l221_221528

theorem initial_population (P : ℝ) (h1 : 1.05 * (0.765 * P + 50) = 3213) : P = 3935 :=
by
  have h2 : 1.05 * (0.765 * P + 50) = 3213 := h1
  sorry

end initial_population_l221_221528


namespace gear_ratio_l221_221557

variable (a b c : ℕ) (ωG ωH ωI : ℚ)

theorem gear_ratio :
  (a * ωG = b * ωH) ∧ (b * ωH = c * ωI) ∧ (a * ωG = c * ωI) →
  ωG / ωH = bc / ac ∧ ωH / ωI = ac / ab ∧ ωG / ωI = bc / ab :=
by
  sorry

end gear_ratio_l221_221557


namespace cost_of_book_sold_at_loss_l221_221367

theorem cost_of_book_sold_at_loss
  (C1 C2 : ℝ)
  (total_cost : C1 + C2 = 360)
  (selling_price1 : 0.85 * C1 = 1.19 * C2) :
  C1 = 210 :=
sorry

end cost_of_book_sold_at_loss_l221_221367


namespace problem_statement_l221_221826

theorem problem_statement : 25 * 15 * 9 * 5.4 * 3.24 = 3 ^ 10 := 
by 
  sorry

end problem_statement_l221_221826


namespace base4_to_base10_conversion_l221_221654

theorem base4_to_base10_conversion :
  2 * 4^4 + 0 * 4^3 + 3 * 4^2 + 1 * 4^1 + 2 * 4^0 = 566 :=
by
  sorry

end base4_to_base10_conversion_l221_221654


namespace games_played_l221_221599

theorem games_played (x : ℕ) (h1 : x * 26 + 42 * (20 - x) = 600) : x = 15 :=
by {
  sorry
}

end games_played_l221_221599


namespace median_of_triangle_l221_221064

variable (a b c : ℝ)

noncomputable def AM : ℝ :=
  (Real.sqrt (2 * b * b + 2 * c * c - a * a)) / 2

theorem median_of_triangle :
  abs (((b + c) / 2) - (a / 2)) < AM a b c ∧ 
  AM a b c < (b + c) / 2 := 
by
  sorry

end median_of_triangle_l221_221064


namespace cost_of_10_pound_bag_is_correct_l221_221362

noncomputable def cost_of_5_pound_bag : ℝ := 13.80
noncomputable def cost_of_25_pound_bag : ℝ := 32.25
noncomputable def min_pounds_needed : ℝ := 65
noncomputable def max_pounds_allowed : ℝ := 80
noncomputable def least_possible_cost : ℝ := 98.73

def min_cost_10_pound_bag : ℝ := 1.98

theorem cost_of_10_pound_bag_is_correct :
  ∀ (x : ℝ), (x >= min_pounds_needed / cost_of_25_pound_bag ∧ x <= max_pounds_allowed / cost_of_5_pound_bag ∧ least_possible_cost = (3 * cost_of_25_pound_bag + x)) → x = min_cost_10_pound_bag :=
by
  sorry

end cost_of_10_pound_bag_is_correct_l221_221362


namespace housewife_spending_l221_221596

theorem housewife_spending (P R M : ℝ) (h1 : R = 65) (h2 : R = 0.75 * P) (h3 : M / R - M / P = 5) :
  M = 1300 :=
by
  -- Proof steps will be added here.
  sorry

end housewife_spending_l221_221596


namespace probability_red_joker_is_1_over_54_l221_221440

-- Define the conditions as given in the problem
def total_cards : ℕ := 54
def red_joker_count : ℕ := 1

-- Define the function to calculate the probability
def probability_red_joker_top_card : ℚ := red_joker_count / total_cards

-- Problem: Prove that the probability of drawing the red joker as the top card is 1/54
theorem probability_red_joker_is_1_over_54 :
  probability_red_joker_top_card = 1 / 54 :=
by
  sorry

end probability_red_joker_is_1_over_54_l221_221440


namespace cliff_total_rocks_l221_221699

theorem cliff_total_rocks (I S : ℕ) (h1 : S = 2 * I) (h2 : I / 3 = 30) :
  I + S = 270 :=
sorry

end cliff_total_rocks_l221_221699


namespace vertices_sum_zero_l221_221423

theorem vertices_sum_zero
  (a b c d e f g h : ℝ)
  (h1 : a = (b + e + d) / 3)
  (h2 : b = (c + f + a) / 3)
  (h3 : c = (d + g + b) / 3)
  (h4 : d = (a + h + e) / 3)
  :
  (a + b + c + d) - (e + f + g + h) = 0 :=
by
  sorry

end vertices_sum_zero_l221_221423


namespace video_down_votes_l221_221796

theorem video_down_votes 
  (up_votes : ℕ)
  (ratio_up_down : up_votes / 1394 = 45 / 17)
  (up_votes_known : up_votes = 3690) : 
  3690 / 1394 = 45 / 17 :=
by
  sorry

end video_down_votes_l221_221796


namespace sequence_term_2023_l221_221189

theorem sequence_term_2023 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h : ∀ n, 2 * S n = a n * (a n + 1)) : 
  a 2023 = 2023 :=
sorry

end sequence_term_2023_l221_221189


namespace actual_time_is_1240pm_l221_221611

def kitchen_and_cellphone_start (t : ℕ) : Prop := t = 8 * 60  -- 8:00 AM in minutes
def kitchen_clock_after_breakfast (t : ℕ) : Prop := t = 8 * 60 + 30  -- 8:30 AM in minutes
def cellphone_after_breakfast (t : ℕ) : Prop := t = 8 * 60 + 20  -- 8:20 AM in minutes
def kitchen_clock_at_3pm (t : ℕ) : Prop := t = 15 * 60  -- 3:00 PM in minutes

theorem actual_time_is_1240pm : 
  (kitchen_and_cellphone_start 480) ∧ 
  (kitchen_clock_after_breakfast 510) ∧ 
  (cellphone_after_breakfast 500) ∧
  (kitchen_clock_at_3pm 900) → 
  real_time_at_kitchen_clock_time_3pm = 12 * 60 + 40 :=
by
  sorry

end actual_time_is_1240pm_l221_221611


namespace scientific_notation_113700_l221_221301

theorem scientific_notation_113700 :
  ∃ (a : ℝ) (b : ℤ), 113700 = a * 10 ^ b ∧ a = 1.137 ∧ b = 5 :=
by
  sorry

end scientific_notation_113700_l221_221301


namespace problem_inequality_solution_set_inequality_proof_l221_221283

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem problem_inequality_solution_set :
  {x : ℝ | f x < 4} = {x : ℝ | -2 < x ∧ x < 2} :=
sorry

theorem inequality_proof (x y : ℝ) (hx : -2 < x ∧ x < 2) (hy : -2 < y ∧ y < 2) :
  |x + y| < |(x * y) / 2 + 2| :=
sorry

end problem_inequality_solution_set_inequality_proof_l221_221283


namespace fourteen_divisible_by_7_twenty_eight_divisible_by_7_thirty_five_divisible_by_7_forty_nine_divisible_by_7_l221_221279

def is_divisible_by_7 (n: ℕ): Prop := n % 7 = 0

theorem fourteen_divisible_by_7: is_divisible_by_7 14 :=
by
  sorry

theorem twenty_eight_divisible_by_7: is_divisible_by_7 28 :=
by
  sorry

theorem thirty_five_divisible_by_7: is_divisible_by_7 35 :=
by
  sorry

theorem forty_nine_divisible_by_7: is_divisible_by_7 49 :=
by
  sorry

end fourteen_divisible_by_7_twenty_eight_divisible_by_7_thirty_five_divisible_by_7_forty_nine_divisible_by_7_l221_221279


namespace no_two_digit_factorization_1729_l221_221668

noncomputable def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem no_two_digit_factorization_1729 :
  ¬ ∃ (a b : ℕ), a * b = 1729 ∧ is_two_digit a ∧ is_two_digit b :=
by
  sorry

end no_two_digit_factorization_1729_l221_221668


namespace valid_p_interval_l221_221306

theorem valid_p_interval :
  ∀ p, (∀ q, q > 0 → (4 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q) ↔ 0 ≤ p ∧ p < 4 :=
sorry

end valid_p_interval_l221_221306


namespace solve_for_x_l221_221471

theorem solve_for_x (x : ℚ) : (1 / 3) + (1 / x) = (3 / 4) → x = 12 / 5 :=
by
  intro h
  -- Proof goes here
  sorry

end solve_for_x_l221_221471


namespace polynomial_sum_eq_l221_221769

-- Definitions of the given polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

-- The theorem to prove
theorem polynomial_sum_eq (x : ℝ) : 
  p x + q x + r x + s x = -x^2 + 10 * x - 11 :=
by 
  -- Proof steps are omitted here
  sorry

end polynomial_sum_eq_l221_221769


namespace kiran_currency_notes_l221_221186

theorem kiran_currency_notes :
  ∀ (n50_amount n100_amount total50 total100 : ℝ),
    n50_amount = 3500 →
    total50 = 5000 →
    total100 = 5000 - 3500 →
    n100_amount = total100 →
    (n50_amount / 50 + total100 / 100) = 85 :=
by
  intros n50_amount n100_amount total50 total100 n50_amount_eq total50_eq total100_eq n100_amount_eq
  sorry

end kiran_currency_notes_l221_221186


namespace total_price_is_correct_l221_221354

-- Define the cost of an adult ticket
def cost_adult : ℕ := 22

-- Define the cost of a children ticket
def cost_child : ℕ := 7

-- Define the number of adults in the family
def num_adults : ℕ := 2

-- Define the number of children in the family
def num_children : ℕ := 2

-- Define the total price the family will pay
def total_price : ℕ := cost_adult * num_adults + cost_child * num_children

-- The proof to check the total price
theorem total_price_is_correct : total_price = 58 :=
by 
  -- Here we would solve the proof
  sorry

end total_price_is_correct_l221_221354


namespace proof_equiv_expression_l221_221131

variable (x y : ℝ)

def P : ℝ := x^2 + y^2
def Q : ℝ := x^2 - y^2

theorem proof_equiv_expression :
  ( (P x y)^2 + (Q x y)^2 ) / ( (P x y)^2 - (Q x y)^2 ) - 
  ( (P x y)^2 - (Q x y)^2 ) / ( (P x y)^2 + (Q x y)^2 ) = 
  (x^4 - y^4) / (x^2 * y^2) :=
by
  sorry

end proof_equiv_expression_l221_221131


namespace parking_lot_vehicle_spaces_l221_221751

theorem parking_lot_vehicle_spaces
  (total_spaces : ℕ)
  (spaces_per_caravan : ℕ)
  (num_caravans : ℕ)
  (remaining_spaces : ℕ) :
  total_spaces = 30 →
  spaces_per_caravan = 2 →
  num_caravans = 3 →
  remaining_spaces = total_spaces - (spaces_per_caravan * num_caravans) →
  remaining_spaces = 24 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end parking_lot_vehicle_spaces_l221_221751


namespace tangent_line_y_intercept_l221_221226

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

theorem tangent_line_y_intercept (a : ℝ) (h : 3 * (1:ℝ)^2 - a = 1) :
  (∃ (m b : ℝ), ∀ (x : ℝ), m = 1 ∧ y = x - 2 → y = m * x + b) := 
 by
  sorry

end tangent_line_y_intercept_l221_221226


namespace bills_difference_l221_221437

variable (m j : ℝ)

theorem bills_difference :
  (0.10 * m = 2) → (0.20 * j = 2) → (m - j = 10) :=
by
  intros h1 h2
  sorry

end bills_difference_l221_221437


namespace even_sum_of_digits_residue_l221_221714

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem even_sum_of_digits_residue (k : ℕ) (h : 2 ≤ k) (r : ℕ) (hr : r < k) :
  ∃ n : ℕ, sum_of_digits n % 2 = 0 ∧ n % k = r := 
sorry

end even_sum_of_digits_residue_l221_221714


namespace sequence_is_geometric_l221_221372

theorem sequence_is_geometric (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 3) 
  (h_rec : ∀ n, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  ∀ n, a n = 2 ^ (n - 1) + 1 := 
by
  sorry

end sequence_is_geometric_l221_221372


namespace proof_problem_l221_221782

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem proof_problem (x1 x2 : ℝ) (h₁ : x1 ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) 
                                (h₂ : x2 ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) 
                                (h₃ : f x1 + f x2 > 0) : 
  x1 + x2 > 0 :=
sorry

end proof_problem_l221_221782


namespace ellipse_equation_l221_221853

theorem ellipse_equation
  (a b : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)
  (eccentricity : ℝ)
  (eccentricity_eq : eccentricity = (Real.sqrt 3 / 3))
  (perimeter_triangle : ℝ)
  (perimeter_eq : perimeter_triangle = 4 * Real.sqrt 3) :
  a = Real.sqrt 3 ∧ b = Real.sqrt 2 ∧ (a > b) ∧ (eccentricity = 1 / Real.sqrt 3) →
  (∀ x y : ℝ, (x^2 / 3 + y^2 / 2 = 1)) :=
by
  sorry

end ellipse_equation_l221_221853


namespace no_real_solutions_l221_221172

noncomputable def equation (x : ℝ) := x + 48 / (x - 3) + 1

theorem no_real_solutions : ∀ x : ℝ, equation x ≠ 0 :=
by
  intro x
  sorry

end no_real_solutions_l221_221172


namespace initial_percentage_of_milk_l221_221227

theorem initial_percentage_of_milk (M : ℝ) (H1 : M / 100 * 60 = 0.58 * 86.9) : M = 83.99 :=
by
  sorry

end initial_percentage_of_milk_l221_221227


namespace chocolate_chip_more_than_raisin_l221_221888

def chocolate_chip_yesterday : ℕ := 19
def chocolate_chip_morning : ℕ := 237
def raisin_cookies : ℕ := 231

theorem chocolate_chip_more_than_raisin : 
  (chocolate_chip_yesterday + chocolate_chip_morning) - raisin_cookies = 25 :=
by 
  sorry

end chocolate_chip_more_than_raisin_l221_221888


namespace arithmetic_sequence_a5_value_l221_221818

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_nonzero : ∀ n : ℕ, a n ≠ 0)
  (h_cond : (a 5)^2 - a 3 - a 7 = 0) 
  : a 5 = 2 := 
sorry

end arithmetic_sequence_a5_value_l221_221818


namespace solve_x_of_det_8_l221_221319

variable (x : ℝ)

def matrix_det (a b c d : ℝ) : ℝ := a * d - b * c

theorem solve_x_of_det_8
  (h : matrix_det (x + 1) (1 - x) (1 - x) (x + 1) = 8) : x = 2 := by
  sorry

end solve_x_of_det_8_l221_221319


namespace age_difference_l221_221035

variable {A B C : ℕ}

-- Definition of conditions
def condition1 (A B C : ℕ) : Prop := A + B > B + C
def condition2 (A C : ℕ) : Prop := C = A - 16

-- The theorem stating the math problem
theorem age_difference (h1 : condition1 A B C) (h2 : condition2 A C) :
  (A + B) - (B + C) = 16 := by
  sorry

end age_difference_l221_221035


namespace prism_faces_l221_221322

-- Define a structure for a prism with a given number of edges
def is_prism (edges : ℕ) := 
  ∃ (n : ℕ), 3 * n = edges

-- Define the theorem to prove the number of faces in a prism given it has 21 edges
theorem prism_faces (h : is_prism 21) : ∃ (faces : ℕ), faces = 9 :=
by
  sorry

end prism_faces_l221_221322


namespace arithmetic_sequence_sum_l221_221010

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h : a 2 + a 10 = 16) : a 4 + a 6 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l221_221010


namespace number_of_sweaters_l221_221735

theorem number_of_sweaters 
(total_price_shirts : ℝ)
(total_shirts : ℕ)
(total_price_sweaters : ℝ)
(price_difference : ℝ) :
total_price_shirts = 400 ∧ total_shirts = 25 ∧ total_price_sweaters = 1500 ∧ price_difference = 4 →
(total_price_sweaters / ((total_price_shirts / total_shirts) + price_difference) = 75) :=
by
  intros
  sorry

end number_of_sweaters_l221_221735


namespace geometric_series_sixth_term_l221_221986

theorem geometric_series_sixth_term :
  ∃ r : ℝ, r > 0 ∧ (16 * r^7 = 11664) ∧ (16 * r^5 = 3888) :=
by 
  sorry

end geometric_series_sixth_term_l221_221986


namespace probability_triplet_1_2_3_in_10_rolls_l221_221117

noncomputable def probability_of_triplet (n : ℕ) : ℝ :=
  let A0 := (6^10 : ℝ)
  let A1 := (8 * 6^7 : ℝ)
  let A2 := (15 * 6^4 : ℝ)
  let A3 := (4 * 6 : ℝ)
  let total := A0
  let p := (A0 - (total - (A1 - A2 + A3))) / total
  p

theorem probability_triplet_1_2_3_in_10_rolls : 
  abs (probability_of_triplet 10 - 0.0367) < 0.0001 :=
by
  sorry

end probability_triplet_1_2_3_in_10_rolls_l221_221117


namespace simplify_expr1_simplify_expr2_l221_221218

theorem simplify_expr1 (a b : ℤ) : 2 * a - (4 * a + 5 * b) + 2 * (3 * a - 4 * b) = 4 * a - 13 * b :=
by sorry

theorem simplify_expr2 (x y : ℤ) : 5 * x^2 - 2 * (3 * y^2 - 5 * x^2) + (-4 * y^2 + 7 * x * y) = 15 * x^2 - 10 * y^2 + 7 * x * y :=
by sorry

end simplify_expr1_simplify_expr2_l221_221218


namespace smallest_b_is_2_plus_sqrt_3_l221_221451

open Real

noncomputable def smallest_b (a b : ℝ) : ℝ :=
  if (2 < a ∧ a < b ∧ (¬(2 + a > b ∧ 2 + b > a ∧ a + b > 2)) ∧
    (¬(1 / b + 1 / a > 2 ∧ 1 / a + 2 > 1 / b ∧ 2 + 1 / b > 1 / a)))
  then b else 0

theorem smallest_b_is_2_plus_sqrt_3 (a b : ℝ) :
  2 < a ∧ a < b ∧ (¬(2 + a > b ∧ 2 + b > a ∧ a + b > 2)) ∧
    (¬(1 / b + 1 / a > 2 ∧ 1 / a + 2 > 1 / b ∧ 2 + 1 / b > 1 / a)) →
  b = 2 + sqrt 3 := sorry

end smallest_b_is_2_plus_sqrt_3_l221_221451


namespace june_earnings_l221_221231

theorem june_earnings 
    (total_clovers : ℕ := 300)
    (pct_3_petals : ℕ := 70)
    (pct_2_petals : ℕ := 20)
    (pct_4_petals : ℕ := 8)
    (pct_5_petals : ℕ := 2)
    (earn_3_petals : ℕ := 1)
    (earn_2_petals : ℕ := 2)
    (earn_4_petals : ℕ := 5)
    (earn_5_petals : ℕ := 10)
    (earn_total : ℕ := 510) : 
  (pct_3_petals * total_clovers) / 100 * earn_3_petals + 
  (pct_2_petals * total_clovers) / 100 * earn_2_petals + 
  (pct_4_petals * total_clovers) / 100 * earn_4_petals + 
  (pct_5_petals * total_clovers) / 100 * earn_5_petals = earn_total := 
by
  -- Proof of this theorem involves calculating each part and summing them. Skipping detailed steps with sorry.
  sorry

end june_earnings_l221_221231


namespace total_bill_is_270_l221_221163

-- Conditions as Lean definitions
def totalBill (T : ℝ) : Prop :=
  let eachShare := T / 10
  9 * (eachShare + 3) = T

-- Theorem stating that the total bill T is 270
theorem total_bill_is_270 (T : ℝ) (h : totalBill T) : T = 270 :=
sorry

end total_bill_is_270_l221_221163


namespace red_balls_count_is_correct_l221_221374

-- Define conditions
def total_balls : ℕ := 100
def white_balls : ℕ := 50
def green_balls : ℕ := 30
def yellow_balls : ℕ := 10
def purple_balls : ℕ := 3
def non_red_purple_prob : ℝ := 0.9

-- Define the number of red balls
def number_of_red_balls (red_balls : ℕ) : Prop :=
  total_balls - (white_balls + green_balls + yellow_balls + purple_balls) = red_balls
  
-- The proof statement
theorem red_balls_count_is_correct : number_of_red_balls 7 := by
  sorry

end red_balls_count_is_correct_l221_221374


namespace standard_deviation_of_distribution_l221_221482

theorem standard_deviation_of_distribution (μ σ : ℝ) 
    (h₁ : μ = 15) (h₂ : μ - 2 * σ = 12) : σ = 1.5 := by
  sorry

end standard_deviation_of_distribution_l221_221482


namespace nth_term_arithmetic_seq_l221_221350

theorem nth_term_arithmetic_seq (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : ∀ n : ℕ, ∃ m : ℝ, a (n + 1) = a n + m)
  (h_d_neg : d < 0)
  (h_condition1 : a 2 * a 4 = 12)
  (h_condition2 : a 2 + a 4 = 8):
  ∀ n : ℕ, a n = -2 * n + 10 :=
by
  sorry

end nth_term_arithmetic_seq_l221_221350


namespace min_value_proof_l221_221184

noncomputable def min_value (t c : ℝ) :=
  (t^2 + c^2 - 2 * t * c + 2 * c^2) / 2

theorem min_value_proof (a b t c : ℝ) (h : a + b = t) :
  (a^2 + (b + c)^2) ≥ min_value t c :=
by
  sorry

end min_value_proof_l221_221184


namespace emily_fishes_correct_l221_221502

/-- Given conditions:
1. Emily caught 4 trout weighing 2 pounds each.
2. Emily caught 3 catfish weighing 1.5 pounds each.
3. Bluegills weigh 2.5 pounds each.
4. Emily caught a total of 25 pounds of fish. -/
def emilyCatches : Prop :=
  ∃ (trout_count catfish_count bluegill_count : ℕ)
    (trout_weight catfish_weight bluegill_weight total_weight : ℝ),
    trout_count = 4 ∧ catfish_count = 3 ∧ 
    trout_weight = 2 ∧ catfish_weight = 1.5 ∧ 
    bluegill_weight = 2.5 ∧ 
    total_weight = 25 ∧
    (total_weight = (trout_count * trout_weight) + (catfish_count * catfish_weight) + (bluegill_count * bluegill_weight)) ∧
    bluegill_count = 5

theorem emily_fishes_correct : emilyCatches := by
  sorry

end emily_fishes_correct_l221_221502


namespace smallest_six_factors_l221_221904

theorem smallest_six_factors (n : ℕ) (h : (n = 2 * 3^2)) : n = 18 :=
by {
    sorry -- proof goes here
}

end smallest_six_factors_l221_221904


namespace find_divisor_l221_221874

theorem find_divisor :
  ∃ d : ℕ, (d = 859560) ∧ ∃ n : ℕ, (n + 859622) % d = 0 ∧ n = 859560 :=
by
  sorry

end find_divisor_l221_221874


namespace distinct_cube_arrangements_count_l221_221969

def is_valid_face_sum (face : Finset ℕ) : Prop :=
  face.sum id = 34

def is_valid_opposite_sum (v1 v2 : ℕ) : Prop :=
  v1 + v2 = 16

def is_unique_up_to_rotation (cubes : List (Finset ℕ)) : Prop := sorry -- Define rotational uniqueness check

noncomputable def count_valid_arrangements : ℕ := sorry -- Define counting logic

theorem distinct_cube_arrangements_count : count_valid_arrangements = 3 :=
  sorry

end distinct_cube_arrangements_count_l221_221969


namespace greatest_second_term_l221_221806

-- Definitions and Conditions
def is_arithmetic_sequence (a d : ℕ) : Bool := (a > 0) && (d > 0)
def sum_four_terms (a d : ℕ) : Bool := (4 * a + 6 * d = 80)
def integer_d (a d : ℕ) : Bool := ((40 - 2 * a) % 3 = 0)

-- Theorem statement to prove
theorem greatest_second_term : ∃ a d : ℕ, is_arithmetic_sequence a d ∧ sum_four_terms a d ∧ integer_d a d ∧ (a + d = 19) :=
sorry

end greatest_second_term_l221_221806


namespace sum_of_squares_l221_221685

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 14) (h2 : a * b + b * c + a * c = 72) : 
  a^2 + b^2 + c^2 = 52 :=
by
  sorry

end sum_of_squares_l221_221685


namespace test_total_points_l221_221763

theorem test_total_points (computation_points_per_problem : ℕ) (word_points_per_problem : ℕ) (total_problems : ℕ) (computation_problems : ℕ) :
  computation_points_per_problem = 3 →
  word_points_per_problem = 5 →
  total_problems = 30 →
  computation_problems = 20 →
  (computation_problems * computation_points_per_problem + 
  (total_problems - computation_problems) * word_points_per_problem) = 110 :=
by
  intros h1 h2 h3 h4
  sorry

end test_total_points_l221_221763


namespace number_of_large_boxes_l221_221416

theorem number_of_large_boxes (total_boxes : ℕ) (small_weight large_weight remaining_small remaining_large : ℕ) :
  total_boxes = 62 →
  small_weight = 5 →
  large_weight = 3 →
  remaining_small = 15 →
  remaining_large = 15 →
  ∀ (small_boxes large_boxes : ℕ),
    total_boxes = small_boxes + large_boxes →
    ((large_boxes * large_weight) + (remaining_small * small_weight) = (small_boxes * small_weight) + (remaining_large * large_weight)) →
    large_boxes = 27 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end number_of_large_boxes_l221_221416


namespace f_neg_a_l221_221764

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 2

theorem f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 2 := by
  sorry

end f_neg_a_l221_221764


namespace length_of_arc_l221_221312

def radius : ℝ := 5
def area_of_sector : ℝ := 10
def expected_length_of_arc : ℝ := 4

theorem length_of_arc (r : ℝ) (A : ℝ) (l : ℝ) (h₁ : r = radius) (h₂ : A = area_of_sector) : l = expected_length_of_arc := by
  sorry

end length_of_arc_l221_221312


namespace train_length_is_499_96_l221_221326

-- Define the conditions
def speed_train_kmh : ℕ := 75   -- Speed of the train in km/h
def speed_man_kmh : ℕ := 3     -- Speed of the man in km/h
def time_cross_s : ℝ := 24.998 -- Time taken for the train to cross the man in seconds

-- Define the conversion factors
def km_to_m : ℕ := 1000        -- Conversion from kilometers to meters
def hr_to_s : ℕ := 3600        -- Conversion from hours to seconds

-- Define relative speed in m/s
def relative_speed_ms : ℕ := (speed_train_kmh - speed_man_kmh) * km_to_m / hr_to_s

-- Prove the length of the train in meters
def length_of_train : ℝ := relative_speed_ms * time_cross_s

theorem train_length_is_499_96 : length_of_train = 499.96 := sorry

end train_length_is_499_96_l221_221326


namespace four_digit_number_divisibility_l221_221336

theorem four_digit_number_divisibility 
  (E V I L : ℕ) 
  (hE : 0 ≤ E ∧ E < 10) 
  (hV : 0 ≤ V ∧ V < 10) 
  (hI : 0 ≤ I ∧ I < 10) 
  (hL : 0 ≤ L ∧ L < 10)
  (h1 : (1000 * E + 100 * V + 10 * I + L) % 73 = 0) 
  (h2 : (1000 * V + 100 * I + 10 * L + E) % 74 = 0)
  : 1000 * L + 100 * I + 10 * V + E = 5499 := 
  sorry

end four_digit_number_divisibility_l221_221336


namespace expected_value_full_circles_l221_221282

-- Definition of the conditions
def num_small_triangles (n : ℕ) : ℕ :=
  n^2

def potential_full_circle_vertices (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 2) * (n - 1) / 2

def prob_full_circle : ℚ :=
  1 / 729

-- The expected number of full circles formed
def expected_full_circles (n : ℕ) : ℚ :=
  potential_full_circle_vertices n * prob_full_circle

-- The mathematical equivalence to be proved
theorem expected_value_full_circles (n : ℕ) : expected_full_circles n = (n - 2) * (n - 1) / 1458 := 
  sorry

end expected_value_full_circles_l221_221282


namespace find_width_of_rectangle_l221_221154

variable (w : ℝ) (l : ℝ) (P : ℝ)

def width_correct (h1 : P = 150) (h2 : l = w + 15) : Prop :=
  w = 30

-- Theorem statement in Lean
theorem find_width_of_rectangle (h1 : P = 150) (h2 : l = w + 15) (h3 : P = 2 * l + 2 * w) : width_correct w l P h1 h2 :=
by
  sorry

end find_width_of_rectangle_l221_221154


namespace positive_integer_solution_lcm_eq_sum_l221_221338

def is_lcm (x y z m : Nat) : Prop :=
  ∃ (d : Nat), x = d * (Nat.gcd y z) ∧ y = d * (Nat.gcd x z) ∧ z = d * (Nat.gcd x y) ∧
  x * y * z / Nat.gcd x (Nat.gcd y z) = m

theorem positive_integer_solution_lcm_eq_sum :
  ∀ (a b c : Nat), 0 < a → 0 < b → 0 < c → is_lcm a b c (a + b + c) → (a, b, c) = (a, 2 * a, 3 * a) := by
    sorry

end positive_integer_solution_lcm_eq_sum_l221_221338


namespace initial_average_customers_l221_221814

theorem initial_average_customers (x A : ℕ) (h1 : x = 1) (h2 : (A + 120) / 2 = 90) : A = 60 := by
  sorry

end initial_average_customers_l221_221814


namespace largest_square_area_l221_221193

theorem largest_square_area (XY YZ XZ : ℝ)
  (h1 : XZ^2 = XY^2 + YZ^2)
  (h2 : XY^2 + YZ^2 + XZ^2 = 450) :
  XZ^2 = 225 :=
by
  sorry

end largest_square_area_l221_221193


namespace distance_foci_l221_221146

noncomputable def distance_between_foci := 
  let F1 := (4, 5)
  let F2 := (-6, 9)
  Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) 

theorem distance_foci : 
  ∃ (F1 F2 : ℝ × ℝ), 
    F1 = (4, 5) ∧ 
    F2 = (-6, 9) ∧ 
    distance_between_foci = 2 * Real.sqrt 29 := by {
  sorry
}

end distance_foci_l221_221146


namespace bruce_eggs_lost_l221_221836

theorem bruce_eggs_lost :
  ∀ (initial_eggs remaining_eggs eggs_lost : ℕ), 
  initial_eggs = 75 → remaining_eggs = 5 →
  eggs_lost = initial_eggs - remaining_eggs →
  eggs_lost = 70 :=
by
  intros initial_eggs remaining_eggs eggs_lost h_initial h_remaining h_loss
  sorry

end bruce_eggs_lost_l221_221836


namespace calc_expression_value_l221_221473

open Real

theorem calc_expression_value :
  sqrt ((16: ℝ) ^ 12 + (8: ℝ) ^ 15) / ((16: ℝ) ^ 5 + (8: ℝ) ^ 16) = (3 * sqrt 2) / 4 := sorry

end calc_expression_value_l221_221473


namespace apple_order_for_month_l221_221510

def Chandler_apples (week : ℕ) : ℕ :=
  23 + 2 * week

def Lucy_apples (week : ℕ) : ℕ :=
  19 - week

def Ross_apples : ℕ :=
  15

noncomputable def total_apples : ℕ :=
  (Chandler_apples 0 + Chandler_apples 1 + Chandler_apples 2 + Chandler_apples 3) +
  (Lucy_apples 0 + Lucy_apples 1 + Lucy_apples 2 + Lucy_apples 3) +
  (Ross_apples * 4)

theorem apple_order_for_month : total_apples = 234 := by
  sorry

end apple_order_for_month_l221_221510


namespace max_soap_boxes_in_carton_l221_221508

def carton_volume (length width height : ℕ) : ℕ :=
  length * width * height

def soap_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def max_soap_boxes (carton_volume soap_box_volume : ℕ) : ℕ :=
  carton_volume / soap_box_volume

theorem max_soap_boxes_in_carton :
  max_soap_boxes (carton_volume 25 42 60) (soap_box_volume 7 6 6) = 250 :=
by
  sorry

end max_soap_boxes_in_carton_l221_221508


namespace cos_double_angle_l221_221762

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by 
  sorry

end cos_double_angle_l221_221762


namespace number_of_customers_who_tipped_is_3_l221_221382

-- Definitions of conditions
def charge_per_lawn : ℤ := 33
def lawns_mowed : ℤ := 16
def total_earnings : ℤ := 558
def tip_per_customer : ℤ := 10

-- Calculate intermediate values
def earnings_from_mowing : ℤ := lawns_mowed * charge_per_lawn
def earnings_from_tips : ℤ := total_earnings - earnings_from_mowing
def number_of_tips : ℤ := earnings_from_tips / tip_per_customer

-- Theorem stating our proof
theorem number_of_customers_who_tipped_is_3 : number_of_tips = 3 := by
  sorry

end number_of_customers_who_tipped_is_3_l221_221382


namespace min_expression_value_l221_221458

def distinct_elements (s : Set ℤ) : Prop := s = {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_expression_value :
  ∃ (p q r s t u v w : ℤ),
    distinct_elements {p, q, r, s, t, u, v, w} ∧
    (p + q + r + s) ≥ 5 ∧
    (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
     q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
     r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
     s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
     t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
     u ≠ v ∧ u ≠ w ∧
     v ≠ w) →
    (p + q + r + s)^2 + (t + u + v + w)^2 = 26 :=
sorry

end min_expression_value_l221_221458


namespace find_g4_l221_221828

noncomputable def g : ℝ → ℝ := sorry

theorem find_g4 (h : ∀ x y : ℝ, x * g y = 2 * y * g x) (h₁ : g 10 = 5) : g 4 = 4 :=
sorry

end find_g4_l221_221828


namespace num_factors_180_l221_221234

-- Conditions: The prime factorization of 180
def fact180 : ℕ := 180
def fact180_prime_decomp : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Definition of counting the number of factors from prime factorization
def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (fun acc p => acc * (p.snd + 1)) 1

-- Theorem statement: The number of positive factors of 180 is 18 
theorem num_factors_180 : number_of_factors fact180_prime_decomp = 18 := 
by
  sorry

end num_factors_180_l221_221234


namespace min_sum_l221_221576

namespace MinimumSum

theorem min_sum (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (hc : 98 * m = n^3) : m + n = 42 :=
sorry

end MinimumSum

end min_sum_l221_221576


namespace initial_observations_count_l221_221079

theorem initial_observations_count (S x n : ℕ) (h1 : S = 12 * n) (h2 : S + x = 11 * (n + 1)) (h3 : x = 5) : n = 6 :=
sorry

end initial_observations_count_l221_221079


namespace increase_factor_l221_221912

-- Definition of parameters: number of letters, digits, and symbols.
def num_letters : ℕ := 26
def num_digits : ℕ := 10
def num_symbols : ℕ := 5

-- Definition of the number of old license plates and new license plates.
def num_old_plates : ℕ := num_letters ^ 2 * num_digits ^ 3
def num_new_plates : ℕ := num_letters ^ 3 * num_digits ^ 3 * num_symbols

-- The proof problem statement: Prove that the increase factor is 130.
theorem increase_factor : num_new_plates / num_old_plates = 130 := by
  sorry

end increase_factor_l221_221912


namespace extreme_point_at_one_l221_221360

def f (a x : ℝ) : ℝ := a*x^3 + x^2 - (a+2)*x + 1
def f' (a x : ℝ) : ℝ := 3*a*x^2 + 2*x - (a+2)

theorem extreme_point_at_one (a : ℝ) :
  (f' a 1 = 0) → (a = 0) :=
by
  intro h
  have : 3 * a * 1^2 + 2 * 1 - (a + 2) = 0 := h
  sorry

end extreme_point_at_one_l221_221360


namespace sum_arithmetic_sequence_l221_221135

theorem sum_arithmetic_sequence (S : ℕ → ℕ) :
  S 7 = 21 ∧ S 17 = 34 → S 27 = 27 :=
by
  sorry

end sum_arithmetic_sequence_l221_221135


namespace polynomial_value_at_one_l221_221892

theorem polynomial_value_at_one
  (a b c : ℝ)
  (h1 : -a - b - c + 1 = 6)
  : a + b + c + 1 = -4 :=
by {
  sorry
}

end polynomial_value_at_one_l221_221892


namespace mod_product_prob_l221_221659

def prob_mod_product (a b : ℕ) : ℚ :=
  let quotient := a * b % 4
  if quotient = 0 then 1/2
  else if quotient = 1 then 1/8
  else if quotient = 2 then 1/4
  else if quotient = 3 then 1/8
  else 0

theorem mod_product_prob (a b : ℕ) :
  (∃ n : ℚ, n = prob_mod_product a b) :=
by
  sorry

end mod_product_prob_l221_221659


namespace sabi_share_removed_l221_221533

theorem sabi_share_removed :
  ∀ (N S M x : ℝ), N - 5 = 2 * (S - x) / 8 ∧ S - x = 4 * (6 * (M - 4)) / 16 ∧ M = 102 ∧ N + S + M = 1100 
  → x = 829.67 := by
  sorry

end sabi_share_removed_l221_221533


namespace probability_of_problem_being_solved_l221_221990

-- Define the probabilities of solving the problem.
def prob_A_solves : ℚ := 1 / 5
def prob_B_solves : ℚ := 1 / 3

-- Define the proof statement
theorem probability_of_problem_being_solved :
  (1 - ((1 - prob_A_solves) * (1 - prob_B_solves))) = 7 / 15 :=
by
  sorry

end probability_of_problem_being_solved_l221_221990


namespace xy_gt_1_necessary_but_not_sufficient_l221_221840

-- To define the conditions and prove the necessary and sufficient conditions.

variable (x y : ℝ)

-- The main statement to prove once conditions are defined.
theorem xy_gt_1_necessary_but_not_sufficient : 
  (x > 1 ∧ y > 1 → x * y > 1) ∧ ¬ (x * y > 1 → x > 1 ∧ y > 1) := 
by 
  sorry

end xy_gt_1_necessary_but_not_sufficient_l221_221840


namespace nontrivial_solution_fraction_l221_221780

theorem nontrivial_solution_fraction (x y z : ℚ)
  (h₁ : x - 6 * y + 3 * z = 0)
  (h₂ : 3 * x - 6 * y - 2 * z = 0)
  (h₃ : x + 6 * y - 5 * z = 0)
  (hne : x ≠ 0) :
  (y * z) / (x^2) = 2 / 3 :=
by
  sorry

end nontrivial_solution_fraction_l221_221780


namespace age_problem_l221_221228

open Classical

variable (A B C : ℕ)

theorem age_problem (h1 : A + 10 = 2 * (B - 10))
                    (h2 : C = 3 * (A - 5))
                    (h3 : A = B + 9)
                    (h4 : C = A + 4) :
  B = 39 :=
sorry

end age_problem_l221_221228


namespace parabola_symmetric_y_axis_intersection_l221_221129

theorem parabola_symmetric_y_axis_intersection :
  ∀ (x y : ℝ),
  (x = y ∨ x*x + y*y - 6*y = 0) ∧ (x*x = 3 * y) :=
by 
  sorry

end parabola_symmetric_y_axis_intersection_l221_221129


namespace triathlete_average_speed_is_approx_3_5_l221_221618

noncomputable def triathlete_average_speed : ℝ :=
  let x : ℝ := 1; -- This represents the distance of biking/running segment
  let swimming_speed := 2; -- km/h
  let biking_speed := 25; -- km/h
  let running_speed := 12; -- km/h
  let swimming_distance := 2 * x; -- 2x km
  let biking_distance := x; -- x km
  let running_distance := x; -- x km
  let total_distance := swimming_distance + biking_distance + running_distance; -- 4x km
  let swimming_time := swimming_distance / swimming_speed; -- x hours
  let biking_time := biking_distance / biking_speed; -- x/25 hours
  let running_time := running_distance / running_speed; -- x/12 hours
  let total_time := swimming_time + biking_time + running_time; -- 1.12333x hours
  total_distance / total_time -- This should be the average speed

theorem triathlete_average_speed_is_approx_3_5 :
  abs (triathlete_average_speed - 3.5) < 0.1 := 
by
  sorry

end triathlete_average_speed_is_approx_3_5_l221_221618


namespace roots_of_equation_l221_221195

theorem roots_of_equation (
  x y: ℝ
) (h1: x + y = 10) (h2: |x - y| = 12):
  (x = 11 ∧ y = -1) ∨ (x = -1 ∧ y = 11) ↔ ∃ (a b: ℝ), a = 11 ∧ b = -1 ∨ a = -1 ∧ b = 11 ∧ a^2 - 10*a - 22 = 0 ∧ b^2 - 10*b - 22 = 0 := 
by sorry

end roots_of_equation_l221_221195


namespace contrapositive_proposition_l221_221779

theorem contrapositive_proposition {a b : ℝ} :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) → (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
sorry

end contrapositive_proposition_l221_221779


namespace average_of_second_class_l221_221893

variable (average1 : ℝ) (average2 : ℝ) (combined_average : ℝ) (n1 : ℕ) (n2 : ℕ)

theorem average_of_second_class
  (h1 : n1 = 25) 
  (h2 : average1 = 40) 
  (h3 : n2 = 30) 
  (h4 : combined_average = 50.90909090909091) 
  (h5 : n1 + n2 = 55) 
  (h6 : n2 * average2 = 55 * combined_average - n1 * average1) :
  average2 = 60 := by
  sorry

end average_of_second_class_l221_221893


namespace sum_first_four_terms_geo_seq_l221_221830

theorem sum_first_four_terms_geo_seq (q : ℝ) (a_1 : ℝ)
  (h1 : q ≠ 1) 
  (h2 : a_1 * (a_1 * q) * (a_1 * q^2) = -1/8)
  (h3 : 2 * (a_1 * q^3) = (a_1 * q) + (a_1 * q^2)) :
  (a_1 + (a_1 * q) + (a_1 * q^2) + (a_1 * q^3)) = 5 / 8 :=
  sorry

end sum_first_four_terms_geo_seq_l221_221830


namespace min_value_expression_l221_221616

theorem min_value_expression (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 1) : 
  ∃(x : ℝ), x ≤ (a - b) * (b - c) * (c - d) * (d - a) ∧ x = -1/8 :=
sorry

end min_value_expression_l221_221616


namespace spadesuit_value_l221_221772

def spadesuit (a b : ℤ) : ℤ :=
  |a^2 - b^2|

theorem spadesuit_value :
  spadesuit 3 (spadesuit 5 2) = 432 :=
by
  sorry

end spadesuit_value_l221_221772


namespace find_omega_l221_221756

noncomputable def f (ω x : ℝ) : ℝ := 3 * Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem find_omega (ω : ℝ) (h₁ : ∀ x₁ x₂, (-ω < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 * ω) → f ω x₁ < f ω x₂)
  (h₂ : ∀ x, f ω x = f ω (-2 * ω - x)) :
  ω = Real.sqrt (3 * Real.pi) / 3 :=
by
  sorry

end find_omega_l221_221756


namespace polynomial_lt_factorial_l221_221901

theorem polynomial_lt_factorial (A B C : ℝ) : ∃N : ℕ, ∀n : ℕ, n > N → An^2 + Bn + C < n! := 
by
  sorry

end polynomial_lt_factorial_l221_221901


namespace all_odd_digits_n_squared_l221_221420

/-- Helper function to check if all digits in a number are odd -/
def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 1

/-- Main theorem stating that the only positive integers n such that all the digits of n^2 are odd are 1 and 3 -/
theorem all_odd_digits_n_squared (n : ℕ) :
  (n > 0) → (all_odd_digits (n^2)) → (n = 1 ∨ n = 3) :=
by
  sorry

end all_odd_digits_n_squared_l221_221420


namespace ratio_of_x_intercepts_l221_221687

theorem ratio_of_x_intercepts (c : ℝ) (u v : ℝ) (h1 : c ≠ 0) 
  (h2 : u = -c / 8) (h3 : v = -c / 4) : u / v = 1 / 2 :=
by {
  sorry
}

end ratio_of_x_intercepts_l221_221687


namespace find_triangle_altitude_l221_221033

variable (A b h : ℝ)

theorem find_triangle_altitude (h_eq_40 :  A = 800 ∧ b = 40) : h = 40 :=
sorry

end find_triangle_altitude_l221_221033


namespace smallest_K_exists_l221_221995

theorem smallest_K_exists (S : Finset ℕ) (h_S : S = (Finset.range 51).erase 0) :
  ∃ K, ∀ (T : Finset ℕ), T ⊆ S ∧ T.card = K → 
  ∃ a b, a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ (a + b) ∣ (a * b) ∧ K = 39 :=
by
  use 39
  sorry

end smallest_K_exists_l221_221995


namespace group_division_ways_l221_221140

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem group_division_ways : 
  choose 30 10 * choose 20 10 * choose 10 10 = Nat.factorial 30 / (Nat.factorial 10 * Nat.factorial 10 * Nat.factorial 10) := 
by
  sorry

end group_division_ways_l221_221140


namespace factor_expression_l221_221262

theorem factor_expression (a b c : ℝ) :
  let num := (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3
  let denom := (a - b)^3 + (b - c)^3 + (c - a)^3
  (denom ≠ 0) →
  num / denom = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_expression_l221_221262


namespace non_negative_real_sum_expressions_l221_221630

theorem non_negative_real_sum_expressions (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) (h_sum : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by
  sorry

end non_negative_real_sum_expressions_l221_221630


namespace quadratic_real_roots_l221_221951

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ k ≥ -9 / 4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_real_roots_l221_221951


namespace probability_no_rain_five_days_l221_221810

noncomputable def probability_of_no_rain (rain_prob : ℚ) (days : ℕ) :=
  (1 - rain_prob) ^ days

theorem probability_no_rain_five_days :
  probability_of_no_rain (2/3) 5 = 1/243 :=
by sorry

end probability_no_rain_five_days_l221_221810


namespace cakes_served_yesterday_l221_221641

theorem cakes_served_yesterday (cakes_today_lunch : ℕ) (cakes_today_dinner : ℕ) (total_cakes : ℕ)
  (h1 : cakes_today_lunch = 5) (h2 : cakes_today_dinner = 6) (h3 : total_cakes = 14) :
  total_cakes - (cakes_today_lunch + cakes_today_dinner) = 3 :=
by
  -- Import necessary libraries
  sorry

end cakes_served_yesterday_l221_221641


namespace arcsin_sqrt3_over_2_eq_pi_over_3_l221_221428

theorem arcsin_sqrt3_over_2_eq_pi_over_3 :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
by
  have h : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
    -- This is a known trigonometric identity.
    sorry
  -- Use the property of arcsin to get the result.
  sorry

end arcsin_sqrt3_over_2_eq_pi_over_3_l221_221428


namespace find_N_l221_221790

noncomputable def sum_of_sequence : ℤ :=
  985 + 987 + 989 + 991 + 993 + 995 + 997 + 999

theorem find_N : ∃ (N : ℤ), 8000 - N = sum_of_sequence ∧ N = 64 := by
  use 64
  -- The actual proof steps will go here
  sorry

end find_N_l221_221790


namespace two_digit_number_l221_221375

theorem two_digit_number (x y : ℕ) (h1 : x + y = 11) (h2 : 10 * y + x = 10 * x + y + 63) : 10 * x + y = 29 := 
by 
  sorry

end two_digit_number_l221_221375


namespace reinforcement_calculation_l221_221427

theorem reinforcement_calculation
  (initial_men : ℕ := 2000)
  (initial_days : ℕ := 40)
  (days_until_reinforcement : ℕ := 20)
  (additional_days_post_reinforcement : ℕ := 10)
  (total_initial_provisions : ℕ := initial_men * initial_days)
  (remaining_provisions_post_20_days : ℕ := total_initial_provisions / 2)
  : ∃ (reinforcement_men : ℕ), reinforcement_men = 2000 :=
by
  have remaining_provisions := remaining_provisions_post_20_days
  have total_post_reinforcement := initial_men + ((remaining_provisions) / (additional_days_post_reinforcement))

  use (total_post_reinforcement - initial_men)
  sorry

end reinforcement_calculation_l221_221427


namespace point_M_in_second_quadrant_l221_221347

-- Given conditions
def m : ℤ := -2
def n : ℤ := 1

-- Definitions to identify the quadrants
def point_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

-- Problem statement to prove
theorem point_M_in_second_quadrant : 
  point_in_second_quadrant m n :=
by
  sorry

end point_M_in_second_quadrant_l221_221347


namespace probability_5800_in_three_spins_l221_221543

def spinner_labels : List String := ["Bankrupt", "$600", "$1200", "$4000", "$800", "$2000", "$150"]

def total_outcomes (spins : Nat) : Nat :=
  let segments := spinner_labels.length
  segments ^ spins

theorem probability_5800_in_three_spins :
  (6 / total_outcomes 3 : ℚ) = 6 / 343 :=
by
  sorry

end probability_5800_in_three_spins_l221_221543


namespace percent_carnations_l221_221147

theorem percent_carnations (F : ℕ) (H1 : 3 / 5 * F = pink) (H2 : 1 / 5 * F = white) 
(H3 : F - pink - white = red) (H4 : 1 / 2 * pink = pink_roses)
(H5 : pink - pink_roses = pink_carnations) (H6 : 1 / 2 * red = red_carnations)
(H7 : white = white_carnations) : 
100 * (pink_carnations + red_carnations + white_carnations) / F = 60 :=
sorry

end percent_carnations_l221_221147


namespace max_value_f_l221_221496

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem max_value_f (h : ∀ ε > (0 : ℝ), ∃ x : ℝ, x < 1 ∧ ε < f x) : ∀ x : ℝ, x < 1 → f x ≤ -1 :=
by
  intros x hx
  dsimp [f]
  -- Proof steps are omitted.
  sorry

example (h: ∀ ε > 0, ∃ x : ℝ, x < 1 ∧ ε < f x) : ∃ x : ℝ, x < 1 ∧ f x = -1 :=
by
  use 0
  -- Proof steps are omitted.
  sorry

end max_value_f_l221_221496


namespace labourer_income_l221_221304

noncomputable def monthly_income : ℤ := 75

theorem labourer_income:
  ∃ (I D : ℤ),
  (80 * 6 = 480) ∧
  (I * 6 - D + (I * 4) = 480 + 240 + D + 30) →
  I = monthly_income :=
by
  sorry

end labourer_income_l221_221304


namespace inequality_range_of_a_l221_221171

theorem inequality_range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |2 * x - a| > x - 1) ↔ a < 3 ∨ a > 5 :=
by
  sorry

end inequality_range_of_a_l221_221171


namespace even_expression_l221_221106

theorem even_expression (m n : ℤ) (hm : Odd m) (hn : Odd n) : Even (m + 5 * n) :=
by
  sorry

end even_expression_l221_221106


namespace sqrt_D_rational_sometimes_not_l221_221894

-- Definitions and conditions
def D (a : ℤ) : ℤ := a^2 + (a + 2)^2 + (a * (a + 2))^2

-- The statement to prove
theorem sqrt_D_rational_sometimes_not (a : ℤ) : ∃ x : ℚ, x = Real.sqrt (D a) ∧ ¬(∃ y : ℤ, x = y) ∨ ∃ y : ℤ, Real.sqrt (D a) = y :=
by 
  sorry

end sqrt_D_rational_sometimes_not_l221_221894


namespace martin_speed_l221_221655

theorem martin_speed (distance : ℝ) (time : ℝ) (h₁ : distance = 12) (h₂ : time = 6) : (distance / time = 2) :=
by 
  -- Note: The proof is not required as per instructions, so we use 'sorry'
  sorry

end martin_speed_l221_221655


namespace expression_result_l221_221534

-- We denote k as a natural number representing the number of digits in A, B, C, and D.
variable (k : ℕ)

-- Definitions of the numbers A, B, C, D, and E based on the problem statement.
def A : ℕ := 3 * ((10 ^ (k - 1) - 1) / 9)
def B : ℕ := 4 * ((10 ^ (k - 1) - 1) / 9)
def C : ℕ := 6 * ((10 ^ (k - 1) - 1) / 9)
def D : ℕ := 7 * ((10 ^ (k - 1) - 1) / 9)
def E : ℕ := 5 * ((10 ^ (2 * k) - 1) / 9)

-- The statement we want to prove.
theorem expression_result :
  E - A * D - B * C + 1 = (10 ^ (k + 1) - 1) / 9 :=
by
  sorry

end expression_result_l221_221534


namespace same_grade_percentage_is_correct_l221_221455

def total_students : ℕ := 40

def grade_distribution : ℕ × ℕ × ℕ × ℕ :=
  (17, 40, 100)

def same_grade_percentage (total_students : ℕ) (same_grade_students : ℕ) : ℚ :=
  (same_grade_students / total_students) * 100

theorem same_grade_percentage_is_correct :
  let same_grade_students := 3 + 5 + 6 + 3
  same_grade_percentage total_students same_grade_students = 42.5 :=
by 
let same_grade_students := 3 + 5 + 6 + 3
show same_grade_percentage total_students same_grade_students = 42.5
sorry

end same_grade_percentage_is_correct_l221_221455


namespace find_m_n_l221_221921

theorem find_m_n 
  (a b c d m n : ℕ) 
  (h₁ : a^2 + b^2 + c^2 + d^2 = 1989)
  (h₂ : a + b + c + d = m^2)
  (h₃ : a = max (max a b) (max c d) ∨ b = max (max a b) (max c d) ∨ c = max (max a b) (max c d) ∨ d = max (max a b) (max c d))
  (h₄ : exists k, k^2 = max (max a b) (max c d))
  : m = 9 ∧ n = 6 :=
by
  -- Proof omitted
  sorry

end find_m_n_l221_221921


namespace problem_l221_221581

def f (x : ℝ) (a b : ℝ) := x^5 + a * x^3 + b * x - 2

-- We are given f(-2) = m
variables (a b m : ℝ)
theorem problem (h : f (-2) a b = m) : f 2 a b + f (-2) a b = -4 :=
by sorry

end problem_l221_221581


namespace grid_problem_l221_221503

theorem grid_problem 
  (n m : ℕ) 
  (h1 : ∀ (blue_cells : ℕ), blue_cells = m + n - 1 → (n * m ≠ 0) → (blue_cells = (n * m) / 2010)) :
  ∃ (k : ℕ), k = 96 :=
by
  sorry

end grid_problem_l221_221503


namespace domain_of_f_l221_221996

noncomputable def f (x : ℝ) : ℝ :=
  (x - 4)^0 + Real.sqrt (2 / (x - 1))

theorem domain_of_f :
  ∀ x : ℝ, (1 < x ∧ x < 4) ∨ (4 < x) ↔
    ∃ y : ℝ, f y = f x :=
sorry

end domain_of_f_l221_221996


namespace smallest_b_for_factoring_l221_221110

theorem smallest_b_for_factoring (b : ℕ) (p q : ℕ) (h1 : p * q = 1800) (h2 : p + q = b) : b = 85 :=
by
  sorry

end smallest_b_for_factoring_l221_221110


namespace luke_piles_coins_l221_221213

theorem luke_piles_coins (x : ℕ) (h_total_piles : 10 = 5 + 5) (h_total_coins : 10 * x = 30) :
  x = 3 :=
by
  sorry

end luke_piles_coins_l221_221213


namespace apple_production_total_l221_221333

def apples_first_year := 40
def apples_second_year := 8 + 2 * apples_first_year
def apples_third_year := apples_second_year - (1 / 4) * apples_second_year
def total_apples := apples_first_year + apples_second_year + apples_third_year

-- Math proof problem statement
theorem apple_production_total : total_apples = 194 :=
  sorry

end apple_production_total_l221_221333


namespace sqrt_x_minus_2_range_l221_221461

theorem sqrt_x_minus_2_range (x : ℝ) : x - 2 ≥ 0 → x ≥ 2 :=
by sorry

end sqrt_x_minus_2_range_l221_221461


namespace houses_count_l221_221084

theorem houses_count (n : ℕ) 
  (h1 : ∃ k : ℕ, k + 7 = 12)
  (h2 : ∃ m : ℕ, m + 25 = 30) :
  n = 32 :=
sorry

end houses_count_l221_221084


namespace fill_time_l221_221170

noncomputable def time_to_fill (X Y Z : ℝ) : ℝ :=
  1 / X + 1 / Y + 1 / Z

theorem fill_time 
  (V X Y Z : ℝ) 
  (h1 : X + Y = V / 3) 
  (h2 : X + Z = V / 2) 
  (h3 : Y + Z = V / 4) :
  1 / time_to_fill X Y Z = 24 / 13 :=
by
  sorry

end fill_time_l221_221170


namespace unqualified_weight_l221_221721

theorem unqualified_weight (w : ℝ) (upper_limit lower_limit : ℝ) 
  (h1 : upper_limit = 10.1) 
  (h2 : lower_limit = 9.9) 
  (h3 : w = 9.09 ∨ w = 9.99 ∨ w = 10.01 ∨ w = 10.09) :
  ¬ (9.09 ≥ lower_limit ∧ 9.09 ≤ upper_limit) :=
by
  sorry

end unqualified_weight_l221_221721


namespace allison_total_video_hours_l221_221166

def total_video_hours_uploaded (total_days: ℕ) (half_days: ℕ) (first_half_rate: ℕ) (second_half_rate: ℕ): ℕ :=
  first_half_rate * half_days + second_half_rate * (total_days - half_days)

theorem allison_total_video_hours :
  total_video_hours_uploaded 30 15 10 20 = 450 :=
by
  sorry

end allison_total_video_hours_l221_221166


namespace price_per_jin_of_tomatoes_is_3yuan_3jiao_l221_221920

/-- Definitions of the conditions --/
def cucumbers_cost_jin : ℕ := 5
def cucumbers_cost_yuan : ℕ := 11
def cucumbers_cost_jiao : ℕ := 8
def tomatoes_cost_jin : ℕ := 4
def difference_cost_yuan : ℕ := 1
def difference_cost_jiao : ℕ := 4

/-- Converting cost in yuan and jiao to decimal yuan --/
def cost_in_yuan (yuan jiao : ℕ) : ℕ := yuan + jiao / 10

/-- Given conditions in decimal --/
def cucumbers_cost := cost_in_yuan cucumbers_cost_yuan cucumbers_cost_jiao
def difference_cost := cost_in_yuan difference_cost_yuan difference_cost_jiao
def tomatoes_cost := cucumbers_cost + difference_cost

/-- Proof statement: price per jin of tomatoes in yuan and jiao --/
theorem price_per_jin_of_tomatoes_is_3yuan_3jiao :
  tomatoes_cost / tomatoes_cost_jin = 3 + 3 / 10 :=
by
  sorry

end price_per_jin_of_tomatoes_is_3yuan_3jiao_l221_221920


namespace arithmetic_sequence_sum_l221_221119

theorem arithmetic_sequence_sum (a1 d : ℝ)
  (h1 : a1 + 11 * d = -8)
  (h2 : 9 / 2 * (a1 + (a1 + 8 * d)) = -9) :
  16 / 2 * (a1 + (a1 + 15 * d)) = -72 := by
  sorry

end arithmetic_sequence_sum_l221_221119


namespace rationalize_denominator_ABC_value_l221_221057

def A := 11 / 4
def B := 5 / 4
def C := 5

theorem rationalize_denominator : 
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

theorem ABC_value :
  A * B * C = 275 :=
sorry

end rationalize_denominator_ABC_value_l221_221057


namespace range_of_a_l221_221931

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x ≥ 4 ∧ y ≥ 4 ∧ x ≤ y → (x^2 + 2*(a-1)*x + 2) ≤ (y^2 + 2*(a-1)*y + 2)) ↔ a ∈ Set.Ici (-3) :=
by
  sorry

end range_of_a_l221_221931


namespace speed_of_second_train_l221_221406

-- Define the given values
def length_train1 := 290.0 -- in meters
def speed_train1 := 120.0 -- in km/h
def length_train2 := 210.04 -- in meters
def crossing_time := 9.0 -- in seconds

-- Define the conversion factors and useful calculations
def meters_per_second_to_kmph (v : Float) : Float := v * 3.6
def total_distance := length_train1 + length_train2
def relative_speed_ms := total_distance / crossing_time
def relative_speed_kmph := meters_per_second_to_kmph relative_speed_ms

-- Define the proof statement
theorem speed_of_second_train : relative_speed_kmph - speed_train1 = 80.0 :=
by
  sorry

end speed_of_second_train_l221_221406


namespace num_perfect_square_factors_1800_l221_221972

theorem num_perfect_square_factors_1800 :
  let factors_1800 := [(2, 3), (3, 2), (5, 2)]
  ∃ n : ℕ, (n = 8) ∧
           (∀ p_k ∈ factors_1800, ∃ (e : ℕ), (e = 0 ∨ e = 2) ∧ n = 2 * 2 * 2 → n = 8) :=
sorry

end num_perfect_square_factors_1800_l221_221972


namespace quadratic_roots_opposite_l221_221873

theorem quadratic_roots_opposite (a : ℝ) (h : ∀ x1 x2 : ℝ, 
  (x1 + x2 = 0 ∧ x1 * x2 = a - 1) ∧
  (x1 - (-(x1)) = 0 ∧ x2 - x1 = 0)) :
  a = 0 :=
sorry

end quadratic_roots_opposite_l221_221873


namespace sum_of_b_for_unique_solution_l221_221613

theorem sum_of_b_for_unique_solution :
  (∃ b1 b2, (3 * (0:ℝ)^2 + (b1 + 6) * 0 + 7 = 0 ∧ 3 * (0:ℝ)^2 + (b2 + 6) * 0 + 7 = 0) ∧ 
   ((b1 + 6)^2 - 4 * 3 * 7 = 0) ∧ ((b2 + 6)^2 - 4 * 3 * 7 = 0) ∧ 
   b1 + b2 = -12)  :=
by
  sorry

end sum_of_b_for_unique_solution_l221_221613


namespace alan_carla_weight_l221_221221

variable (a b c d : ℝ)

theorem alan_carla_weight (h1 : a + b = 280) (h2 : b + c = 230) (h3 : c + d = 250) (h4 : a + d = 300) :
  a + c = 250 := by
sorry

end alan_carla_weight_l221_221221


namespace price_restoration_percentage_l221_221658

noncomputable def original_price := 100
def reduced_price (P : ℝ) := 0.8 * P
def restored_price (P : ℝ) (x : ℝ) := P = x * reduced_price P

theorem price_restoration_percentage (P : ℝ) (x : ℝ) (h : restored_price P x) : x = 1.25 :=
by
  sorry

end price_restoration_percentage_l221_221658


namespace gus_buys_2_dozen_l221_221734

-- Definitions from conditions
def dozens_to_golf_balls (d : ℕ) : ℕ := d * 12
def total_golf_balls : ℕ := 132
def golf_balls_per_dozen : ℕ := 12
def dan_buys : ℕ := 5
def chris_buys_golf_balls : ℕ := 48

-- The number of dozens Gus buys
noncomputable def gus_buys (total_dozens dan_dozens chris_dozens : ℕ) : ℕ := total_dozens - dan_dozens - chris_dozens

theorem gus_buys_2_dozen : gus_buys (total_golf_balls / golf_balls_per_dozen) dan_buys (chris_buys_golf_balls / golf_balls_per_dozen) = 2 := by
  sorry

end gus_buys_2_dozen_l221_221734


namespace circular_garden_area_l221_221509

theorem circular_garden_area (AD DB DC R : ℝ) 
  (h1 : AD = 10) 
  (h2 : DB = 10) 
  (h3 : DC = 12) 
  (h4 : AD^2 + DC^2 = R^2) : 
  π * R^2 = 244 * π := 
  by 
    sorry

end circular_garden_area_l221_221509


namespace g_49_l221_221631

noncomputable def g : ℝ → ℝ := sorry

axiom g_func_eqn (x y : ℝ) : g (x^2 * y) = x * g y
axiom g_one_val : g 1 = 6

theorem g_49 : g 49 = 42 := by
  sorry

end g_49_l221_221631


namespace nomogram_relation_l221_221863

noncomputable def root_of_eq (x p q : ℝ) : Prop :=
  x^2 + p * x + q = 0

theorem nomogram_relation (x p q : ℝ) (hx : root_of_eq x p q) : 
  q = -x * p - x^2 :=
by 
  sorry

end nomogram_relation_l221_221863


namespace problem_lean_l221_221351

theorem problem_lean (k b : ℤ) : 
  ∃ n : ℤ, n = 25 ∧ n^2 = (k + 1)^4 - k^4 ∧ 3 * n + 100 = b^2 :=
sorry

end problem_lean_l221_221351


namespace sufficient_condition_for_inequality_l221_221269

theorem sufficient_condition_for_inequality (a b : ℝ) (h_nonzero : a * b ≠ 0) : (a < b ∧ b < 0) → (1 / a ^ 2 > 1 / b ^ 2) :=
by
  intro h
  sorry

end sufficient_condition_for_inequality_l221_221269


namespace polygon_sides_eq_seven_l221_221320

theorem polygon_sides_eq_seven (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) → n = 7 :=
by
  sorry

end polygon_sides_eq_seven_l221_221320


namespace ratio_cost_to_marked_price_l221_221289

theorem ratio_cost_to_marked_price (x : ℝ) 
  (h_discount: ∀ y, y = marked_price → selling_price = (3/4) * y)
  (h_cost: ∀ z, z = selling_price → cost_price = (2/3) * z) :
  cost_price / marked_price = 1 / 2 :=
by
  sorry

end ratio_cost_to_marked_price_l221_221289


namespace angle_C_is_65_deg_l221_221074

-- Defining a triangle and its angles.
structure Triangle :=
  (A B C : ℝ) -- representing the angles in degrees

-- Defining the conditions of the problem.
def given_triangle : Triangle :=
  { A := 75, B := 40, C := 180 - 75 - 40 }

-- Statement of the problem, proving that the measure of ∠C is 65°.
theorem angle_C_is_65_deg (t : Triangle) (hA : t.A = 75) (hB : t.B = 40) (hSum : t.A + t.B + t.C = 180) : t.C = 65 :=
  by sorry

end angle_C_is_65_deg_l221_221074


namespace intersection_of_A_and_B_is_B_implies_m_leq_4_over_3_l221_221617

noncomputable def f (x : ℝ) : ℝ := (1 / (Real.sqrt (x + 2))) + Real.log (3 - x)
def A : Set ℝ := { x | -2 < x ∧ x < 3 }
def B (m : ℝ) : Set ℝ := { x | 1 - m < x ∧ x < 3 * m - 1 }

theorem intersection_of_A_and_B_is_B_implies_m_leq_4_over_3 (m : ℝ) 
    (h : A ∩ B m = B m) : m ≤ 4 / 3 := by
  sorry

end intersection_of_A_and_B_is_B_implies_m_leq_4_over_3_l221_221617


namespace isabella_hair_growth_l221_221577

def initial_hair_length : ℝ := 18
def final_hair_length : ℝ := 24
def hair_growth : ℝ := final_hair_length - initial_hair_length

theorem isabella_hair_growth : hair_growth = 6 := by
  sorry

end isabella_hair_growth_l221_221577


namespace john_average_score_change_l221_221413

/-- Given John's scores on his biology exams, calculate the change in his average score after the fourth exam. -/
theorem john_average_score_change :
  let first_three_scores := [84, 88, 95]
  let fourth_score := 92
  let first_average := (84 + 88 + 95) / 3
  let new_average := (84 + 88 + 95 + 92) / 4
  new_average - first_average = 0.75 :=
by
  sorry

end john_average_score_change_l221_221413


namespace simplify_first_expression_simplify_second_expression_l221_221519

theorem simplify_first_expression (x y : ℝ) : 3 * x - 2 * y + 1 + 3 * y - 2 * x - 5 = x + y - 4 :=
sorry

theorem simplify_second_expression (x : ℝ) : (2 * x ^ 4 - 5 * x ^ 2 - 4 * x + 3) - (3 * x ^ 3 - 5 * x ^ 2 - 4 * x) = 2 * x ^ 4 - 3 * x ^ 3 + 3 :=
sorry

end simplify_first_expression_simplify_second_expression_l221_221519


namespace find_distance_CD_l221_221632

-- Define the ellipse and the required points
def ellipse (x y : ℝ) : Prop := 16 * (x-3)^2 + 4 * (y+2)^2 = 64

-- Define the center and the semi-axes lengths
noncomputable def center : (ℝ × ℝ) := (3, -2)
noncomputable def semi_major_axis_length : ℝ := 4
noncomputable def semi_minor_axis_length : ℝ := 2

-- Define the points C and D on the ellipse
def point_C (x y : ℝ) : Prop := ellipse x y ∧ (x = 3 + semi_major_axis_length ∨ x = 3 - semi_major_axis_length) ∧ y = -2
def point_D (x y : ℝ) : Prop := ellipse x y ∧ x = 3 ∧ (y = -2 + semi_minor_axis_length ∨ y = -2 - semi_minor_axis_length)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Main theorem to prove
theorem find_distance_CD : 
  ∃ C D : ℝ × ℝ, 
    (point_C C.1 C.2 ∧ point_D D.1 D.2) → 
    distance C D = 2 * Real.sqrt 5 := 
sorry

end find_distance_CD_l221_221632


namespace sum_of_square_roots_l221_221531

theorem sum_of_square_roots : 
  (Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + 
  Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9) + 
  Real.sqrt (1 + 3 + 5 + 7 + 9 + 11)) = 21 :=
by
  -- Proof here
  sorry

end sum_of_square_roots_l221_221531


namespace tax_deduction_is_correct_l221_221994

-- Define the hourly wage and tax rate
def hourly_wage_dollars : ℝ := 25
def tax_rate : ℝ := 0.021

-- Define the conversion from dollars to cents
def dollars_to_cents (dollars : ℝ) : ℝ := dollars * 100

-- Calculate the hourly wage in cents
def hourly_wage_cents : ℝ := dollars_to_cents hourly_wage_dollars

-- Calculate the tax deducted in cents per hour
def tax_deduction_cents (wage : ℝ) (rate : ℝ) : ℝ := rate * wage

-- State the theorem that needs to be proven
theorem tax_deduction_is_correct :
  tax_deduction_cents hourly_wage_cents tax_rate = 52.5 :=
by
  sorry

end tax_deduction_is_correct_l221_221994


namespace circle_constant_ratio_l221_221605

theorem circle_constant_ratio (b : ℝ) :
  (∀ (x y : ℝ), (x + 4)^2 + (y + b)^2 = 16 → 
    ∃ k : ℝ, 
      ∀ P : ℝ × ℝ, 
        P = (x, y) → 
        dist P (-2, 0) / dist P (4, 0) = k)
  → b = 0 :=
by
  intros h
  sorry

end circle_constant_ratio_l221_221605


namespace prove_cuboid_properties_l221_221121

noncomputable def cuboid_length := 5
noncomputable def cuboid_width := 4
noncomputable def cuboid_height := 3

theorem prove_cuboid_properties :
  (min (cuboid_length * cuboid_width) (min (cuboid_length * cuboid_height) (cuboid_width * cuboid_height)) = 12) ∧
  (max (cuboid_length * cuboid_width) (max (cuboid_length * cuboid_height) (cuboid_width * cuboid_height)) = 20) ∧
  ((cuboid_length + cuboid_width + cuboid_height) * 4 = 48) ∧
  (2 * (cuboid_length * cuboid_width + cuboid_length * cuboid_height + cuboid_width * cuboid_height) = 94) ∧
  (cuboid_length * cuboid_width * cuboid_height = 60) :=
by
  sorry

end prove_cuboid_properties_l221_221121


namespace P_equals_neg12_l221_221359

def P (a b : ℝ) : ℝ :=
  (2 * a + 3 * b)^2 - (2 * a + b) * (2 * a - b) - 2 * b * (3 * a + 5 * b)

lemma simplified_P (a b : ℝ) : P a b = 6 * a * b :=
  by sorry

theorem P_equals_neg12 (a b : ℝ) (h : b = -2 / a) : P a b = -12 :=
  by sorry

end P_equals_neg12_l221_221359


namespace probability_inside_octahedron_l221_221606

noncomputable def probability_of_octahedron : ℝ := 
  let cube_volume := 8
  let octahedron_volume := 4 / 3
  octahedron_volume / cube_volume

theorem probability_inside_octahedron :
  probability_of_octahedron = 1 / 6 :=
  by
    sorry

end probability_inside_octahedron_l221_221606


namespace solve_sys_eqns_l221_221521

def sys_eqns_solution (x y : ℝ) : Prop :=
  y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y

theorem solve_sys_eqns :
  ∃ (x y : ℝ),
  (sys_eqns_solution x y ∧
  ((x = 0 ∧ y = 0) ∨
  (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2) ∨
  (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2))) :=
by
  sorry

end solve_sys_eqns_l221_221521


namespace contradiction_method_at_most_one_positive_l221_221316

theorem contradiction_method_at_most_one_positive :
  (∃ a b c : ℝ, (a > 0 → (b ≤ 0 ∧ c ≤ 0)) ∧ (b > 0 → (a ≤ 0 ∧ c ≤ 0)) ∧ (c > 0 → (a ≤ 0 ∧ b ≤ 0))) → 
  (¬(∃ a b c : ℝ, (a > 0 ∧ b > 0) ∨ (b > 0 ∧ c > 0) ∨ (a > 0 ∧ c > 0))) :=
by sorry

end contradiction_method_at_most_one_positive_l221_221316


namespace mixed_doubles_pairing_l221_221970

def num_ways_to_pair (men women : ℕ) (select_men select_women : ℕ) : ℕ :=
  (Nat.choose men select_men) * (Nat.choose women select_women) * 2

theorem mixed_doubles_pairing : num_ways_to_pair 5 4 2 2 = 120 := by
  sorry

end mixed_doubles_pairing_l221_221970


namespace total_triangles_l221_221692

theorem total_triangles (small_triangles : ℕ)
    (triangles_4_small : ℕ)
    (triangles_9_small : ℕ)
    (triangles_16_small : ℕ)
    (number_small_triangles : small_triangles = 20)
    (number_triangles_4_small : triangles_4_small = 5)
    (number_triangles_9_small : triangles_9_small = 1)
    (number_triangles_16_small : triangles_16_small = 1) :
    small_triangles + triangles_4_small + triangles_9_small + triangles_16_small = 27 := 
by 
    -- proof omitted
    sorry

end total_triangles_l221_221692


namespace new_students_correct_l221_221755

variable 
  (students_start_year : Nat)
  (students_left : Nat)
  (students_end_year : Nat)

def new_students (students_start_year students_left students_end_year : Nat) : Nat :=
  students_end_year - (students_start_year - students_left)

theorem new_students_correct :
  ∀ (students_start_year students_left students_end_year : Nat),
  students_start_year = 10 →
  students_left = 4 →
  students_end_year = 48 →
  new_students students_start_year students_left students_end_year = 42 :=
by
  intros students_start_year students_left students_end_year h1 h2 h3
  rw [h1, h2, h3]
  unfold new_students
  norm_num

end new_students_correct_l221_221755


namespace cricket_player_innings_l221_221710

theorem cricket_player_innings (n : ℕ) (T : ℕ) 
  (h1 : T = n * 48) 
  (h2 : T + 178 = (n + 1) * 58) : 
  n = 12 :=
by
  sorry

end cricket_player_innings_l221_221710


namespace metal_relative_atomic_mass_is_24_l221_221006

noncomputable def relative_atomic_mass (metal_mass : ℝ) (hcl_mass_percent : ℝ) (hcl_total_mass : ℝ) (mol_mass_hcl : ℝ) : ℝ :=
  let moles_hcl := (hcl_total_mass * hcl_mass_percent / 100) / mol_mass_hcl
  let maximum_molar_mass := metal_mass / (moles_hcl / 2)
  let minimum_molar_mass := metal_mass / (moles_hcl / 2)
  if 20 < maximum_molar_mass ∧ maximum_molar_mass < 28 then
    24
  else
    0

theorem metal_relative_atomic_mass_is_24
  (metal_mass_1 : ℝ)
  (metal_mass_2 : ℝ)
  (hcl_mass_percent : ℝ)
  (hcl_total_mass : ℝ)
  (mol_mass_hcl : ℝ)
  (moles_used_1 : ℝ)
  (moles_used_2 : ℝ)
  (excess : Bool)
  (complete : Bool) :
  relative_atomic_mass 3.5 18.25 50 36.5 = 24 :=
by
  sorry

end metal_relative_atomic_mass_is_24_l221_221006


namespace beneficiary_received_32_176_l221_221152

noncomputable def A : ℝ := 19520 / 0.728
noncomputable def B : ℝ := 1.20 * A
noncomputable def C : ℝ := 1.44 * A
noncomputable def D : ℝ := 1.728 * A

theorem beneficiary_received_32_176 :
    round B = 32176 :=
by
    sorry

end beneficiary_received_32_176_l221_221152


namespace value_of_x_l221_221559

theorem value_of_x (x : ℝ) (h : ∃ k < 0, (x, 1) = k • (4, x)) : x = -2 :=
sorry

end value_of_x_l221_221559


namespace find_totally_damaged_cartons_l221_221833

def jarsPerCarton : ℕ := 20
def initialCartons : ℕ := 50
def reducedCartons : ℕ := 30
def damagedJarsPerCarton : ℕ := 3
def damagedCartons : ℕ := 5
def totalGoodJars : ℕ := 565

theorem find_totally_damaged_cartons :
  (initialCartons * jarsPerCarton - ((initialCartons - reducedCartons) * jarsPerCarton + damagedJarsPerCarton * damagedCartons - totalGoodJars)) / jarsPerCarton = 1 := by
  sorry

end find_totally_damaged_cartons_l221_221833


namespace brother_growth_is_one_l221_221743

-- Define measurements related to Stacy's height.
def Stacy_previous_height : ℕ := 50
def Stacy_current_height : ℕ := 57

-- Define the condition that Stacy's growth is 6 inches more than her brother's growth.
def Stacy_growth := Stacy_current_height - Stacy_previous_height
def Brother_growth := Stacy_growth - 6

-- Prove that Stacy's brother grew 1 inch.
theorem brother_growth_is_one : Brother_growth = 1 :=
by
  sorry

end brother_growth_is_one_l221_221743


namespace min_value_of_expression_l221_221102

noncomputable def minValue (a : ℝ) : ℝ :=
  1 / (3 - 2 * a) + 2 / (a - 1)

theorem min_value_of_expression : ∀ a : ℝ, 1 < a ∧ a < 3 / 2 → (1 / (3 - 2 * a) + 2 / (a - 1)) ≥ 16 / 9 :=
by
  intro a h
  sorry

end min_value_of_expression_l221_221102


namespace min_pencils_for_each_color_max_pencils_remaining_each_color_max_red_pencils_to_ensure_five_remaining_l221_221850

-- Condition Definitions
def blue := 5
def red := 9
def green := 6
def yellow := 4

-- Theorem Statements
theorem min_pencils_for_each_color :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ min_pencils : ℕ, min_pencils = 21 := by
  sorry

theorem max_pencils_remaining_each_color :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ max_pencils : ℕ, max_pencils = 3 := by
  sorry

theorem max_red_pencils_to_ensure_five_remaining :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ max_red_pencils : ℕ, max_red_pencils = 4 := by
  sorry

end min_pencils_for_each_color_max_pencils_remaining_each_color_max_red_pencils_to_ensure_five_remaining_l221_221850


namespace soccer_team_starters_l221_221624

open Nat

-- Definitions representing the conditions
def total_players : ℕ := 18
def twins_included : ℕ := 2
def remaining_players : ℕ := total_players - twins_included
def starters_to_choose : ℕ := 7 - twins_included

-- Theorem statement to assert the solution
theorem soccer_team_starters :
  Nat.choose remaining_players starters_to_choose = 4368 :=
by
  -- Placeholder for proof
  sorry

end soccer_team_starters_l221_221624


namespace min_value_of_objective_function_l221_221586

theorem min_value_of_objective_function : 
  ∃ (x y : ℝ), 
    (2 * x + y - 2 ≥ 0) ∧ 
    (x - 2 * y + 4 ≥ 0) ∧ 
    (x - 1 ≤ 0) ∧ 
    (∀ (u v: ℝ), 
      (2 * u + v - 2 ≥ 0) → 
      (u - 2 * v + 4 ≥ 0) → 
      (u - 1 ≤ 0) → 
      (3 * u + 2 * v ≥ 3)) :=
  sorry

end min_value_of_objective_function_l221_221586


namespace birds_meeting_distance_l221_221825

theorem birds_meeting_distance 
  (D : ℝ) (S1 : ℝ) (S2 : ℝ) (t : ℝ)
  (H1 : D = 45)
  (H2 : S1 = 6)
  (H3 : S2 = 2.5)
  (H4 : t = D / (S1 + S2)) :
  S1 * t = 31.76 :=
by
  sorry

end birds_meeting_distance_l221_221825


namespace polynomial_sum_correct_l221_221095

def f (x : ℝ) : ℝ := -4 * x^3 + 2 * x^2 - x - 5
def g (x : ℝ) : ℝ := -6 * x^3 - 7 * x^2 + 4 * x - 2
def h (x : ℝ) : ℝ := 2 * x^3 + 8 * x^2 + 6 * x + 3
def sum_polynomials (x : ℝ) : ℝ := -8 * x^3 + 3 * x^2 + 9 * x - 4

theorem polynomial_sum_correct (x : ℝ) : f x + g x + h x = sum_polynomials x :=
by sorry

end polynomial_sum_correct_l221_221095


namespace least_number_correct_l221_221535

def least_number_to_add_to_make_perfect_square (x : ℝ) : ℝ :=
  let y := 1 - x -- since 1 is the smallest whole number > sqrt(0.0320)
  y

theorem least_number_correct (x : ℝ) (h : x = 0.0320) : least_number_to_add_to_make_perfect_square x = 0.9680 :=
by {
  -- Proof is skipped
  -- The proof would involve verifying that adding this number to x results in a perfect square (1 in this case).
  sorry
}

end least_number_correct_l221_221535


namespace find_first_month_sale_l221_221254

/-- Given the sales for months two to six and the average sales over six months,
    prove the sale in the first month. -/
theorem find_first_month_sale
  (sales_2 : ℤ) (sales_3 : ℤ) (sales_4 : ℤ) (sales_5 : ℤ) (sales_6 : ℤ)
  (avg_sales : ℤ)
  (h2 : sales_2 = 5468) (h3 : sales_3 = 5568) (h4 : sales_4 = 6088)
  (h5 : sales_5 = 6433) (h6 : sales_6 = 5922) (h_avg : avg_sales = 5900) : 
  ∃ (sale_1 : ℤ), sale_1 = 5921 := 
by
  have total_sales : ℤ := avg_sales * 6
  have known_sales_sum : ℤ := sales_2 + sales_3 + sales_4 + sales_5
  use total_sales - known_sales_sum - sales_6
  sorry

end find_first_month_sale_l221_221254


namespace find_k_for_parallel_lines_l221_221287

theorem find_k_for_parallel_lines (k : ℝ) :
  (∀ x y : ℝ, (k - 2) * x + (4 - k) * y + 1 = 0) →
  (∀ x y : ℝ, 2 * (k - 2) * x - 2 * y + 3 = 0) →
  (k = 2 ∨ k = 5) :=
sorry

end find_k_for_parallel_lines_l221_221287


namespace cost_per_ice_cream_l221_221604

theorem cost_per_ice_cream (chapati_count : ℕ)
                           (rice_plate_count : ℕ)
                           (mixed_vegetable_plate_count : ℕ)
                           (ice_cream_cup_count : ℕ)
                           (cost_per_chapati : ℕ)
                           (cost_per_rice_plate : ℕ)
                           (cost_per_mixed_vegetable : ℕ)
                           (amount_paid : ℕ)
                           (total_cost_chapatis : ℕ)
                           (total_cost_rice : ℕ)
                           (total_cost_mixed_vegetable : ℕ)
                           (total_non_ice_cream_cost : ℕ)
                           (total_ice_cream_cost : ℕ)
                           (cost_per_ice_cream_cup : ℕ) :
    chapati_count = 16 →
    rice_plate_count = 5 →
    mixed_vegetable_plate_count = 7 →
    ice_cream_cup_count = 6 →
    cost_per_chapati = 6 →
    cost_per_rice_plate = 45 →
    cost_per_mixed_vegetable = 70 →
    amount_paid = 961 →
    total_cost_chapatis = chapati_count * cost_per_chapati →
    total_cost_rice = rice_plate_count * cost_per_rice_plate →
    total_cost_mixed_vegetable = mixed_vegetable_plate_count * cost_per_mixed_vegetable →
    total_non_ice_cream_cost = total_cost_chapatis + total_cost_rice + total_cost_mixed_vegetable →
    total_ice_cream_cost = amount_paid - total_non_ice_cream_cost →
    cost_per_ice_cream_cup = total_ice_cream_cost / ice_cream_cup_count →
    cost_per_ice_cream_cup = 25 :=
by
    intros; sorry

end cost_per_ice_cream_l221_221604


namespace jorge_goals_l221_221841

theorem jorge_goals (g_last g_total g_this : ℕ) (h_last : g_last = 156) (h_total : g_total = 343) :
  g_this = g_total - g_last → g_this = 187 :=
by
  intro h
  rw [h_last, h_total] at h
  apply h

end jorge_goals_l221_221841


namespace six_digit_palindrome_count_l221_221653

def num_six_digit_palindromes : Nat :=
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindrome_count : num_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindrome_count_l221_221653


namespace select_at_least_8_sticks_l221_221662

theorem select_at_least_8_sticks (S : Finset ℕ) (hS : S = (Finset.range 92 \ {0})) :
  ∃ (sticks : Finset ℕ) (h_sticks : sticks.card = 8),
    ∃ (a b c : ℕ) (h_a : a ∈ sticks) (h_b : b ∈ sticks) (h_c : c ∈ sticks),
    (a + b > c) ∧ (b + c > a) ∧ (c + a > b) :=
by
  -- Proof required here
  sorry

end select_at_least_8_sticks_l221_221662


namespace percent_increase_lines_l221_221683

theorem percent_increase_lines (final_lines increase : ℕ) (h1 : final_lines = 5600) (h2 : increase = 1600) :
  (increase * 100) / (final_lines - increase) = 40 := 
sorry

end percent_increase_lines_l221_221683


namespace max_area_inscribed_octagon_l221_221024

theorem max_area_inscribed_octagon
  (R : ℝ)
  (s : ℝ)
  (a b : ℝ)
  (h1 : s^2 = 5)
  (h2 : (a * b) = 4)
  (h3 : (s * Real.sqrt 2) = (2*R))
  (h4 : (Real.sqrt (a^2 + b^2)) = 2 * R) :
  ∃ A : ℝ, A = 3 * Real.sqrt 5 :=
by
  sorry

end max_area_inscribed_octagon_l221_221024


namespace math_problem_proof_l221_221109

def ratio_area_BFD_square_ABCE (x : ℝ) (AF FE DE CD : ℝ) (h1 : AF = FE / 3) (h2 : CD = 3 * DE) : Prop :=
  let AE := (AF + FE)
  let area_square := (AE)^2
  let area_triangle_BFD := area_square - (1/2 * AF * (AE - FE) + 1/2 * (AE - FE) * FE + 1/2 * DE * CD)
  (area_triangle_BFD / area_square) = (1/16)
  
theorem math_problem_proof (x AF FE DE CD : ℝ) (h1 : AF = FE / 3) (h2 : CD = 3 * DE) (area_ratio : area_triangle_BFD / area_square = 1/16) : ratio_area_BFD_square_ABCE x AF FE DE CD h1 h2 :=
sorry

end math_problem_proof_l221_221109


namespace sum_of_three_consecutive_integers_l221_221881

theorem sum_of_three_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 7) : a + b + c = 18 :=
sorry

end sum_of_three_consecutive_integers_l221_221881


namespace slope_of_line_l221_221089

theorem slope_of_line
  (m : ℝ)
  (b : ℝ)
  (h1 : b = 4)
  (h2 : ∀ x y : ℝ, y = m * x + b → (x = 199 ∧ y = 800) → True) :
  m = 4 :=
by
  sorry

end slope_of_line_l221_221089


namespace linda_age_difference_l221_221321

/-- 
Linda is some more than 2 times the age of Jane.
In five years, the sum of their ages will be 28.
Linda's age at present is 13.
Prove that Linda's age is 3 years more than 2 times Jane's age.
-/
theorem linda_age_difference {L J : ℕ} (h1 : L = 13)
  (h2 : (L + 5) + (J + 5) = 28) : L - 2 * J = 3 :=
by sorry

end linda_age_difference_l221_221321


namespace maximize_profit_l221_221645

noncomputable def profit (t : ℝ) : ℝ :=
  27 - (18 / t) - t

theorem maximize_profit : ∀ t > 0, profit t ≤ 27 - 6 * Real.sqrt 2 ∧ profit (3 * Real.sqrt 2) = 27 - 6 * Real.sqrt 2 := by {
  sorry
}

end maximize_profit_l221_221645


namespace expression_value_l221_221515

theorem expression_value 
  (a b c : ℕ) 
  (ha : a = 12) 
  (hb : b = 2) 
  (hc : c = 7) :
  (a - (b - c)) - ((a - b) - c) = 14 := 
by 
  sorry

end expression_value_l221_221515


namespace solution_is_three_l221_221309

def equation (x : ℝ) : Prop := 
  Real.sqrt (4 - 3 * Real.sqrt (10 - 3 * x)) = x - 2

theorem solution_is_three : equation 3 :=
by sorry

end solution_is_three_l221_221309


namespace simplify_fraction_l221_221063

theorem simplify_fraction :
  10 * (15 / 8) * (-40 / 45) = -(50 / 3) :=
sorry

end simplify_fraction_l221_221063


namespace smallest_portion_l221_221679

theorem smallest_portion
    (a_1 d : ℚ)
    (h1 : 5 * a_1 + 10 * d = 10)
    (h2 : (a_1 + 2 * d + a_1 + 3 * d + a_1 + 4 * d) / 7 = a_1 + a_1 + d) :
  a_1 = 1 / 6 := 
sorry

end smallest_portion_l221_221679


namespace range_of_a_l221_221414

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a*x^2 - a*x - 2 ≤ 0) → (-8 ≤ a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l221_221414


namespace decrease_in_profit_due_to_idle_loom_correct_l221_221682

def loom_count : ℕ := 80
def total_sales_value : ℕ := 500000
def monthly_manufacturing_expenses : ℕ := 150000
def establishment_charges : ℕ := 75000
def efficiency_level_idle_loom : ℕ := 100
def sales_per_loom : ℕ := total_sales_value / loom_count
def expenses_per_loom : ℕ := monthly_manufacturing_expenses / loom_count
def profit_contribution_idle_loom : ℕ := sales_per_loom - expenses_per_loom

def decrease_in_profit_due_to_idle_loom : ℕ := 4375

theorem decrease_in_profit_due_to_idle_loom_correct :
  profit_contribution_idle_loom = decrease_in_profit_due_to_idle_loom :=
by sorry

end decrease_in_profit_due_to_idle_loom_correct_l221_221682


namespace roster_method_A_l221_221731

def A : Set ℤ := {x | 0 < x ∧ x ≤ 2}

theorem roster_method_A :
  A = {1, 2} :=
by
  sorry

end roster_method_A_l221_221731


namespace geo_seq_4th_term_l221_221345

theorem geo_seq_4th_term (a r : ℝ) (h₀ : a = 512) (h₆ : a * r^5 = 32) :
  a * r^3 = 64 :=
by 
  sorry

end geo_seq_4th_term_l221_221345


namespace mixed_oil_rate_l221_221174

/-- Given quantities and prices of three types of oils, any combination
that satisfies the volume and price conditions will achieve a final mixture rate of Rs. 65 per litre. -/
theorem mixed_oil_rate (x y z : ℝ) : 
  12.5 * 55 + 7.75 * 70 + 3.25 * 82 = 1496.5 ∧ 12.5 + 7.75 + 3.25 = 23.5 →
  x + y + z = 23.5 ∧ 55 * x + 70 * y + 82 * z = 65 * 23.5 →
  true :=
by
  intros h1 h2
  sorry

end mixed_oil_rate_l221_221174


namespace area_of_rectangle_l221_221858

-- Define the given conditions
def length : Real := 5.9
def width : Real := 3
def expected_area : Real := 17.7

theorem area_of_rectangle : (length * width) = expected_area := 
by 
  sorry

end area_of_rectangle_l221_221858


namespace sum_cubic_polynomial_l221_221884

noncomputable def q : ℤ → ℤ := sorry  -- We use a placeholder definition for q

theorem sum_cubic_polynomial :
  q 3 = 2 ∧ q 8 = 22 ∧ q 12 = 10 ∧ q 17 = 32 →
  (q 2 + q 3 + q 4 + q 5 + q 6 + q 7 + q 8 + q 9 + q 10 + q 11 + q 12 + q 13 + q 14 + q 15 + q 16 + q 17 + q 18) = 272 :=
sorry

end sum_cubic_polynomial_l221_221884


namespace linear_function_third_quadrant_and_origin_l221_221459

theorem linear_function_third_quadrant_and_origin (k b : ℝ) (h1 : ∀ x < 0, k * x + b ≥ 0) (h2 : k * 0 + b ≠ 0) : k < 0 ∧ b > 0 :=
sorry

end linear_function_third_quadrant_and_origin_l221_221459


namespace min_garden_cost_l221_221634

theorem min_garden_cost : 
  let flower_cost (flower : String) : Real :=
    if flower = "Asters" then 1 else
    if flower = "Begonias" then 2 else
    if flower = "Cannas" then 2 else
    if flower = "Dahlias" then 3 else
    if flower = "Easter lilies" then 2.5 else
    0
  let region_area (region : String) : Nat :=
    if region = "Bottom left" then 10 else
    if region = "Top left" then 9 else
    if region = "Bottom right" then 20 else
    if region = "Top middle" then 2 else
    if region = "Top right" then 7 else
    0
  let min_cost : Real :=
    (flower_cost "Dahlias" * region_area "Top middle") + 
    (flower_cost "Easter lilies" * region_area "Top right") + 
    (flower_cost "Cannas" * region_area "Top left") + 
    (flower_cost "Begonias" * region_area "Bottom left") + 
    (flower_cost "Asters" * region_area "Bottom right")
  min_cost = 81.5 :=
by
  sorry

end min_garden_cost_l221_221634


namespace mayor_vice_mayor_happy_people_l221_221187

theorem mayor_vice_mayor_happy_people :
  (∃ (institutions_per_institution : ℕ) (num_institutions : ℕ),
    institutions_per_institution = 80 ∧
    num_institutions = 6 ∧
    num_institutions * institutions_per_institution = 480) :=
by
  sorry

end mayor_vice_mayor_happy_people_l221_221187


namespace tangent_lines_create_regions_l221_221045

theorem tangent_lines_create_regions (n : ℕ) (h : n = 26) : ∃ k, k = 68 :=
by
  have h1 : ∃ k, k = 68 := ⟨68, rfl⟩
  exact h1

end tangent_lines_create_regions_l221_221045


namespace voting_for_marty_l221_221955

/-- Conditions provided in the problem -/
def total_people : ℕ := 400
def percentage_biff : ℝ := 0.30
def percentage_clara : ℝ := 0.20
def percentage_doc : ℝ := 0.10
def percentage_ein : ℝ := 0.05
def percentage_undecided : ℝ := 0.15

/-- Statement to prove the number of people voting for Marty -/
theorem voting_for_marty : 
  (1 - percentage_biff - percentage_clara - percentage_doc - percentage_ein - percentage_undecided) * total_people = 80 :=
by
  sorry

end voting_for_marty_l221_221955


namespace trees_not_pine_trees_l221_221875

theorem trees_not_pine_trees
  (total_trees : ℕ)
  (percentage_pine : ℝ)
  (number_pine : ℕ)
  (number_not_pine : ℕ)
  (h_total : total_trees = 350)
  (h_percentage : percentage_pine = 0.70)
  (h_pine : number_pine = percentage_pine * total_trees)
  (h_not_pine : number_not_pine = total_trees - number_pine)
  : number_not_pine = 105 :=
sorry

end trees_not_pine_trees_l221_221875


namespace remainder_theorem_div_l221_221829

noncomputable
def p (A B C : ℝ) (x : ℝ) : ℝ := A * x^6 + B * x^4 + C * x^2 + 5

theorem remainder_theorem_div (A B C : ℝ) (h : p A B C 2 = 13) : p A B C (-2) = 13 :=
by
  -- Proof goes here
  sorry

end remainder_theorem_div_l221_221829


namespace speed_of_train_l221_221101

theorem speed_of_train (length : ℝ) (time : ℝ) (conversion_factor : ℝ) (speed_kmh : ℝ) 
  (h1 : length = 240) (h2 : time = 16) (h3 : conversion_factor = 3.6) :
  speed_kmh = (length / time) * conversion_factor := 
sorry

end speed_of_train_l221_221101


namespace total_hours_correct_l221_221408

/-- Definitions for the times each person has left to finish their homework. -/
noncomputable def Jacob_time : ℕ := 18
noncomputable def Greg_time : ℕ := Jacob_time - 6
noncomputable def Patrick_time : ℕ := 2 * Greg_time - 4

/-- Proving the total time left for Patrick, Greg, and Jacob to finish their homework. -/

theorem total_hours_correct : Jacob_time + Greg_time + Patrick_time = 50 := by
  sorry

end total_hours_correct_l221_221408


namespace rotation_result_l221_221775

def initial_vector : ℝ × ℝ × ℝ := (3, -1, 1)

def rotate_180_z (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match v with
  | (x, y, z) => (-x, -y, z)

theorem rotation_result :
  rotate_180_z initial_vector = (-3, 1, 1) :=
by
  sorry

end rotation_result_l221_221775


namespace minimum_n_value_l221_221072

-- Define a multiple condition
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

-- Given conditions
def conditions (n : ℕ) : Prop := 
  (n ≥ 8) ∧ is_multiple 4 n ∧ is_multiple 8 n

-- Lean theorem statement for the problem
theorem minimum_n_value (n : ℕ) (h : conditions n) : n = 8 :=
  sorry

end minimum_n_value_l221_221072


namespace average_minutes_per_day_l221_221800

-- Definitions based on the conditions
variables (f : ℕ)
def third_graders := 6 * f
def fourth_graders := 2 * f
def fifth_graders := f

def third_graders_time := 10 * third_graders f
def fourth_graders_time := 12 * fourth_graders f
def fifth_graders_time := 15 * fifth_graders f

def total_students := third_graders f + fourth_graders f + fifth_graders f
def total_time := third_graders_time f + fourth_graders_time f + fifth_graders_time f

-- Proof statement
theorem average_minutes_per_day : total_time f / total_students f = 11 := sorry

end average_minutes_per_day_l221_221800


namespace x_squared_plus_y_squared_l221_221811

theorem x_squared_plus_y_squared (x y : ℝ) (h₀ : x + y = 10) (h₁ : x * y = 15) : x^2 + y^2 = 70 :=
by
  sorry

end x_squared_plus_y_squared_l221_221811


namespace circle_intersection_l221_221513

theorem circle_intersection (m : ℝ) :
  (x^2 + y^2 - 2*m*x + m^2 - 4 = 0 ∧ x^2 + y^2 + 2*x - 4*m*y + 4*m^2 - 8 = 0) →
  (-12/5 < m ∧ m < -2/5) ∨ (0 < m ∧ m < 2) :=
by sorry

end circle_intersection_l221_221513


namespace max_value_y_l221_221717

theorem max_value_y (x : ℝ) (h : x < -1) : x + 1/(x + 1) ≤ -3 :=
by sorry

end max_value_y_l221_221717


namespace gcd_g50_g52_l221_221583

def g (x : ℤ) := x^2 - 2*x + 2022

theorem gcd_g50_g52 : Int.gcd (g 50) (g 52) = 2 := by
  sorry

end gcd_g50_g52_l221_221583


namespace larger_exceeds_smaller_by_5_l221_221965

-- Define the problem's parameters and conditions.
variables (x n m : ℕ)
variables (subtracted : ℕ := 5)

-- Define the two numbers based on the given ratio.
def larger_number := 6 * x
def smaller_number := 5 * x

-- Condition when a number is subtracted
def new_ratio_condition := (larger_number - subtracted) * 4 = (smaller_number - subtracted) * 5

-- The main goal
theorem larger_exceeds_smaller_by_5 (hx : new_ratio_condition) : larger_number - smaller_number = 5 :=
sorry

end larger_exceeds_smaller_by_5_l221_221965


namespace speed_boat_25_kmph_l221_221051

noncomputable def speed_of_boat_in_still_water (V_s : ℝ) (time : ℝ) (distance : ℝ) : ℝ :=
  let V_d := distance / time
  V_d - V_s

theorem speed_boat_25_kmph (h_vs : V_s = 5) (h_time : time = 4) (h_distance : distance = 120) :
  speed_of_boat_in_still_water V_s time distance = 25 :=
by
  rw [h_vs, h_time, h_distance]
  unfold speed_of_boat_in_still_water
  simp
  norm_num

end speed_boat_25_kmph_l221_221051


namespace combined_rate_l221_221661

theorem combined_rate
  (earl_rate : ℕ)
  (ellen_time : ℚ)
  (total_envelopes : ℕ)
  (total_time : ℕ)
  (combined_total_envelopes : ℕ)
  (combined_total_time : ℕ) :
  earl_rate = 36 →
  ellen_time = 1.5 →
  total_envelopes = 36 →
  total_time = 1 →
  combined_total_envelopes = 180 →
  combined_total_time = 3 →
  (earl_rate + (total_envelopes / ellen_time)) = 60 :=
by
  sorry

end combined_rate_l221_221661


namespace twenty_five_percent_less_than_80_is_twenty_five_percent_more_of_l221_221291

theorem twenty_five_percent_less_than_80_is_twenty_five_percent_more_of (n : ℝ) (h : 1.25 * n = 80 - 0.25 * 80) : n = 48 :=
by
  sorry

end twenty_five_percent_less_than_80_is_twenty_five_percent_more_of_l221_221291


namespace find_smallest_k_l221_221579

theorem find_smallest_k : ∃ (k : ℕ), 64^k > 4^20 ∧ ∀ (m : ℕ), (64^m > 4^20) → m ≥ k := sorry

end find_smallest_k_l221_221579


namespace largest_five_digit_palindromic_number_l221_221522

theorem largest_five_digit_palindromic_number (a b c d e : ℕ)
  (h1 : ∃ a b c, 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
                 ∃ d e, 0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
                 (10001 * a + 1010 * b + 100 * c = 45 * (1001 * d + 110 * e))) :
  10001 * 5 + 1010 * 9 + 100 * 8 = 59895 :=
by
  sorry

end largest_five_digit_palindromic_number_l221_221522


namespace B_pow_2017_eq_B_l221_221159

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![ ![0, 1, 0], ![0, 0, 1], ![1, 0, 0] ]

theorem B_pow_2017_eq_B : B^2017 = B := by
  sorry

end B_pow_2017_eq_B_l221_221159


namespace longest_side_of_similar_triangle_l221_221626

theorem longest_side_of_similar_triangle :
  ∀ (x : ℝ),
    let a := 8
    let b := 10
    let c := 12
    let s₁ := a * x
    let s₂ := b * x
    let s₃ := c * x
    a + b + c = 30 → 
    30 * x = 150 → 
    s₁ > 30 → 
    max s₁ (max s₂ s₃) = 60 :=
by
  intros x a b c s₁ s₂ s₃ h₁ h₂ h₃
  sorry

end longest_side_of_similar_triangle_l221_221626


namespace multiply_difference_of_cubes_l221_221484

def multiply_and_simplify (x : ℝ) : ℝ :=
  (x^4 + 25 * x^2 + 625) * (x^2 - 25)

theorem multiply_difference_of_cubes (x : ℝ) :
  multiply_and_simplify x = x^6 - 15625 :=
by
  sorry

end multiply_difference_of_cubes_l221_221484


namespace phi_eq_pi_div_two_l221_221587

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.cos (x + ϕ)

theorem phi_eq_pi_div_two (ϕ : ℝ) (h1 : 0 ≤ ϕ) (h2 : ϕ ≤ π)
  (h3 : ∀ x : ℝ, f x ϕ = -f (-x) ϕ) : ϕ = π / 2 :=
sorry

end phi_eq_pi_div_two_l221_221587


namespace cricket_team_matches_l221_221844

theorem cricket_team_matches 
  (M : ℕ) (W : ℕ) 
  (h1 : W = 20 * M / 100) 
  (h2 : (W + 80) * 100 = 52 * M) : 
  M = 250 :=
by
  sorry

end cricket_team_matches_l221_221844


namespace field_trip_savings_l221_221378

-- Define the parameters given in the conditions
def num_students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_per_month : ℕ := 4
def num_months : ℕ := 2

-- Define the weekly savings for the class
def weekly_savings : ℕ := num_students * contribution_per_student_per_week

-- Define the total weeks in the given number of months
def total_weeks : ℕ := num_months * weeks_per_month

-- Define the total savings in the given number of months
def total_savings : ℕ := weekly_savings * total_weeks

-- Now, we state the theorem
theorem field_trip_savings : total_savings = 480 :=
by {
  -- calculations are skipped
  sorry
}

end field_trip_savings_l221_221378


namespace coal_extraction_in_four_months_l221_221648

theorem coal_extraction_in_four_months
  (x1 x2 x3 x4 : ℝ)
  (h1 : 4 * x1 + x2 + 2 * x3 + 5 * x4 = 10)
  (h2 : 2 * x1 + 3 * x2 + 2 * x3 + x4 = 7)
  (h3 : 5 * x1 + 2 * x2 + x3 + 4 * x4 = 14) :
  4 * (x1 + x2 + x3 + x4) = 12 :=
by
  sorry

end coal_extraction_in_four_months_l221_221648


namespace smallest_option_l221_221973

-- Define the problem with the given condition
def x : ℕ := 10

-- Define all the options in the problem
def option_a := 6 / x
def option_b := 6 / (x + 1)
def option_c := 6 / (x - 1)
def option_d := x / 6
def option_e := (x + 1) / 6
def option_f := (x - 2) / 6

-- The proof problem statement to show that option_b is the smallest
theorem smallest_option :
  option_b < option_a ∧ option_b < option_c ∧ option_b < option_d ∧ option_b < option_e ∧ option_b < option_f :=
by
  sorry

end smallest_option_l221_221973


namespace find_integer_pairs_l221_221880

theorem find_integer_pairs :
  { (m, n) : ℤ × ℤ | n^3 + m^3 + 231 = n^2 * m^2 + n * m } = {(4, 5), (5, 4)} :=
by
  sorry

end find_integer_pairs_l221_221880


namespace solve_for_x_l221_221238

theorem solve_for_x (x : ℝ) (h : |x - 2| = |x - 3| + 1) : x = 3 :=
by
  sorry

end solve_for_x_l221_221238


namespace eyes_saw_plane_l221_221150

theorem eyes_saw_plane (total_students : ℕ) (fraction_looked_up : ℚ) (students_with_eyepatches : ℕ) :
  total_students = 200 → fraction_looked_up = 3/4 → students_with_eyepatches = 20 →
  ∃ eyes_saw_plane, eyes_saw_plane = 280 :=
by
  intros h1 h2 h3
  sorry

end eyes_saw_plane_l221_221150


namespace evaluate_f_l221_221832

def f (x : ℝ) : ℝ := sorry  -- Placeholder function definition

theorem evaluate_f :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, f (x + 5/2) = -1 / f x) ∧
  (∀ x : ℝ, x ∈ [-5/2, 0] → f x = x * (x + 5/2))
  → f 2016 = 3/2 :=
by
  sorry

end evaluate_f_l221_221832


namespace number_of_common_tangents_l221_221056

def circleM (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0
def circleN (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

theorem number_of_common_tangents : ∃ n : ℕ, n = 2 ∧ 
  (∀ x y : ℝ, circleM x y → circleN x y → false) :=
by
  sorry

end number_of_common_tangents_l221_221056


namespace tracy_two_dogs_food_consumption_l221_221540

theorem tracy_two_dogs_food_consumption
  (cups_per_meal : ℝ)
  (meals_per_day : ℝ)
  (pounds_per_cup : ℝ)
  (num_dogs : ℝ) :
  cups_per_meal = 1.5 →
  meals_per_day = 3 →
  pounds_per_cup = 1 / 2.25 →
  num_dogs = 2 →
  num_dogs * (cups_per_meal * meals_per_day) * pounds_per_cup = 4 := by
  sorry

end tracy_two_dogs_food_consumption_l221_221540


namespace returning_players_l221_221180

-- Definitions of conditions
def num_groups : Nat := 9
def players_per_group : Nat := 6
def new_players : Nat := 48

-- Definition of total number of players
def total_players : Nat := num_groups * players_per_group

-- Theorem: Find the number of returning players
theorem returning_players :
  total_players - new_players = 6 :=
by
  sorry

end returning_players_l221_221180


namespace pentagonal_tiles_count_l221_221434

theorem pentagonal_tiles_count (t s p : ℕ) 
  (h1 : t + s + p = 30) 
  (h2 : 3 * t + 4 * s + 5 * p = 120) : 
  p = 10 := by
  sorry

end pentagonal_tiles_count_l221_221434


namespace expectedAdjacentBlackPairs_l221_221342

noncomputable def numberOfBlackPairsInCircleDeck (totalCards blackCards redCards : ℕ) : ℚ := 
  let probBlackNext := (blackCards - 1) / (totalCards - 1)
  blackCards * probBlackNext

theorem expectedAdjacentBlackPairs (totalCards blackCards redCards expectedPairs : ℕ) : 
  totalCards = 52 → 
  blackCards = 30 → 
  redCards = 22 → 
  expectedPairs = 870 / 51 → 
  numberOfBlackPairsInCircleDeck totalCards blackCards redCards = expectedPairs :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end expectedAdjacentBlackPairs_l221_221342


namespace triangle_area_parallel_line_l221_221085

/-- Given line passing through (8, 2) and parallel to y = -x + 1,
    the area of the triangle formed by this line and the coordinate axes is 50. -/
theorem triangle_area_parallel_line :
  ∃ k b : ℝ, k = -1 ∧ (8 * k + b = 2) ∧ (1/2 * 10 * 10 = 50) :=
sorry

end triangle_area_parallel_line_l221_221085


namespace cylinder_original_radius_l221_221178

theorem cylinder_original_radius
    (r h: ℝ)
    (h₀: h = 4)
    (h₁: π * (r + 8)^2 * 4 = π * r^2 * 12) :
    r = 12 :=
by
  -- Insert your proof here
  sorry

end cylinder_original_radius_l221_221178


namespace probability_of_at_least_one_head_in_three_tosses_is_7_over_8_l221_221485

def probability_of_at_least_one_head (p : ℚ) (n : ℕ) : ℚ := 
  1 - (1 - p)^n

theorem probability_of_at_least_one_head_in_three_tosses_is_7_over_8 :
  probability_of_at_least_one_head (1/2) 3 = 7/8 :=
by 
  sorry

end probability_of_at_least_one_head_in_three_tosses_is_7_over_8_l221_221485


namespace part1_part2_l221_221019

-- Define the function y in Lean
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part (1)
theorem part1 (x : ℝ) : y (1/2) x < 0 ↔ -1 < x ∧ x < 2 :=
  sorry

-- Part (2)
theorem part2 (x m : ℝ) : y m x < (1 - m) * x - 1 ↔ 
  (m = 0 → x > 0) ∧ 
  (m > 0 → 0 < x ∧ x < 1 / m) ∧ 
  (m < 0 → x < 1 / m ∨ x > 0) :=
  sorry

end part1_part2_l221_221019


namespace largest_negative_integer_solution_l221_221138

theorem largest_negative_integer_solution :
  ∃ x : ℤ, x < 0 ∧ 50 * x + 14 % 24 = 10 % 24 ∧ ∀ y : ℤ, (y < 0 ∧ y % 12 = 10 % 12 → y ≤ x) :=
by
  sorry

end largest_negative_integer_solution_l221_221138


namespace algebraic_expr_value_l221_221191

theorem algebraic_expr_value {a b : ℝ} (h: a + b = 1) : a^2 - b^2 + 2 * b + 9 = 10 := 
by
  sorry

end algebraic_expr_value_l221_221191


namespace largest_k_for_sum_of_integers_l221_221020

theorem largest_k_for_sum_of_integers (k : ℕ) (n : ℕ) (h1 : 3^12 = k * n + k * (k + 1) / 2) 
  (h2 : k ∣ 2 * 3^12) (h3 : k < 1031) : k ≤ 486 :=
by 
  sorry -- The proof is skipped here, only the statement is required 

end largest_k_for_sum_of_integers_l221_221020


namespace sqrt_expression_l221_221098

theorem sqrt_expression (h : n < m ∧ m < 0) : 
  (Real.sqrt (m^2 + 2 * m * n + n^2) - Real.sqrt (m^2 - 2 * m * n + n^2)) = -2 * m := 
by {
  sorry
}

end sqrt_expression_l221_221098


namespace pure_imaginary_a_zero_l221_221499

theorem pure_imaginary_a_zero (a : ℝ) (h : ∃ b : ℝ, (i : ℂ) * (1 + (a : ℂ) * i) = (b : ℂ) * i) : a = 0 :=
by
  sorry

end pure_imaginary_a_zero_l221_221499


namespace miles_left_to_reach_E_l221_221219

-- Given conditions as definitions
def total_journey : ℕ := 2500
def miles_driven : ℕ := 642
def miles_B_to_C : ℕ := 400
def miles_C_to_D : ℕ := 550
def detour_D_to_E : ℕ := 200

-- Proof statement
theorem miles_left_to_reach_E : 
  (miles_B_to_C + miles_C_to_D + detour_D_to_E) = 1150 :=
by
  sorry

end miles_left_to_reach_E_l221_221219
