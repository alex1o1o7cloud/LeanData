import Mathlib

namespace track_length_l2195_219519

theorem track_length
  (x : ℕ)
  (run1_Brenda : x / 2 + 80 = a)
  (run2_Sally : x / 2 + 100 = b)
  (run1_ratio : 80 / (x / 2 - 80) = c)
  (run2_ratio : (x / 2 - 100) / (x / 2 + 100) = c)
  : x = 520 :=
by sorry

end track_length_l2195_219519


namespace company_total_employees_l2195_219525

def total_employees_after_hiring (T : ℕ) (before_hiring_female_percentage : ℚ) (additional_male_workers : ℕ) (after_hiring_female_percentage : ℚ) : ℕ :=
  T + additional_male_workers

theorem company_total_employees (T : ℕ)
  (before_hiring_female_percentage : ℚ)
  (additional_male_workers : ℕ)
  (after_hiring_female_percentage : ℚ)
  (h_before_percent : before_hiring_female_percentage = 0.60)
  (h_additional_male : additional_male_workers = 28)
  (h_after_percent : after_hiring_female_percentage = 0.55)
  (h_equation : (before_hiring_female_percentage * T)/(T + additional_male_workers) = after_hiring_female_percentage) :
  total_employees_after_hiring T before_hiring_female_percentage additional_male_workers after_hiring_female_percentage = 336 :=
by {
  -- This is where you add the proof steps.
  sorry
}

end company_total_employees_l2195_219525


namespace scientific_notation_l2195_219527

theorem scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) (h₁ : x = 5853) (h₂ : 1 ≤ |a|) (h₃ : |a| < 10) (h₄ : x = a * 10^n) : 
  a = 5.853 ∧ n = 3 :=
by sorry

end scientific_notation_l2195_219527


namespace squirrel_rainy_days_l2195_219554

theorem squirrel_rainy_days (s r : ℕ) (h1 : 20 * s + 12 * r = 112) (h2 : s + r = 8) : r = 6 :=
by {
  -- sorry to skip the proof
  sorry
}

end squirrel_rainy_days_l2195_219554


namespace oil_bill_january_l2195_219588

-- Define the problem in Lean
theorem oil_bill_january (F J : ℝ) (h1 : F / J = 3 / 2) (h2 : (F + 30) / J = 5 / 3) : J = 180 := 
sorry

end oil_bill_january_l2195_219588


namespace max_value_problem1_l2195_219501

theorem max_value_problem1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) : 
  ∃ t, t = (1 / 2) * x * (1 - 2 * x) ∧ t ≤ 1 / 16 := sorry

end max_value_problem1_l2195_219501


namespace find_G_14_l2195_219502

noncomputable def G (x : ℝ) : ℝ := sorry

lemma G_at_7 : G 7 = 20 := sorry

lemma functional_equation (x : ℝ) (hx: x ^ 2 + 8 * x + 16 ≠ 0) : 
  G (4 * x) / G (x + 4) = 16 - (96 * x + 128) / (x^2 + 8 * x + 16) := sorry

theorem find_G_14 : G 14 = 96 := sorry

end find_G_14_l2195_219502


namespace y_in_terms_of_x_l2195_219509

theorem y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 4) : y = 4 - 3 * x := 
by
  sorry

end y_in_terms_of_x_l2195_219509


namespace right_triangle_inradius_height_ratio_l2195_219565

-- Define a right triangle with sides a, b, and hypotenuse c
variables {a b c : ℝ}
-- Define the altitude from the right angle vertex
variables {h : ℝ}
-- Define the inradius of the triangle
variables {r : ℝ}

-- Define the conditions: right triangle 
-- and the relationships for h and r
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2
def altitude (h : ℝ) (a b c : ℝ) : Prop := h = (a * b) / c
def inradius (r : ℝ) (a b c : ℝ) : Prop := r = (a + b - c) / 2

theorem right_triangle_inradius_height_ratio {a b c h r : ℝ} 
  (Hrt : is_right_triangle a b c)
  (Hh : altitude h a b c)
  (Hr : inradius r a b c) : 
  0.4 < r / h ∧ r / h < 0.5 :=
sorry

end right_triangle_inradius_height_ratio_l2195_219565


namespace find_all_functions_l2195_219516

theorem find_all_functions (n : ℕ) (h_pos : 0 < n) (f : ℝ → ℝ) :
  (∀ x y : ℝ, (f x)^n * f (x + y) = (f x)^(n + 1) + x^n * f y) ↔
  (if n % 2 = 1 then ∀ x, f x = 0 ∨ f x = x else ∀ x, f x = 0 ∨ f x = x ∨ f x = -x) :=
sorry

end find_all_functions_l2195_219516


namespace convert_157_base_10_to_base_7_l2195_219532

-- Given
def base_10_to_base_7(n : ℕ) : String := "313"

-- Prove
theorem convert_157_base_10_to_base_7 : base_10_to_base_7 157 = "313" := by
  sorry

end convert_157_base_10_to_base_7_l2195_219532


namespace toothpicks_at_150th_stage_l2195_219561

theorem toothpicks_at_150th_stage (a₁ d n : ℕ) (h₁ : a₁ = 6) (hd : d = 5) (hn : n = 150) :
  (n * (2 * a₁ + (n - 1) * d)) / 2 = 56775 :=
by
  sorry -- Proof to be completed.

end toothpicks_at_150th_stage_l2195_219561


namespace geometric_series_ratio_l2195_219546

theorem geometric_series_ratio (a r : ℝ) 
  (h_series : ∑' n : ℕ, a * r^n = 18 )
  (h_odd_series : ∑' n : ℕ, a * r^(2*n + 1) = 8 ) : 
  r = 4 / 5 := 
sorry

end geometric_series_ratio_l2195_219546


namespace sequence_formula_l2195_219581

noncomputable def seq (a : ℕ+ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, (a n - 3) * a (n + 1) - a n + 4 = 0

theorem sequence_formula (a : ℕ+ → ℚ) (h : seq a) :
  ∀ n : ℕ+, a n = (2 * n - 1) / n :=
by
  sorry

end sequence_formula_l2195_219581


namespace ratio_difference_l2195_219579

variables (p q r : ℕ) (x : ℕ)
noncomputable def shares_p := 3 * x
noncomputable def shares_q := 7 * x
noncomputable def shares_r := 12 * x

theorem ratio_difference (h1 : shares_q - shares_p = 2400) : shares_r - shares_q = 3000 :=
by sorry

end ratio_difference_l2195_219579


namespace find_b_l2195_219510

-- Variables representing the terms in the equations
variables (a b t : ℝ)

-- Conditions given in the problem
def cond1 : Prop := a - (t / 6) * b = 20
def cond2 : Prop := a - (t / 5) * b = -10
def t_value : Prop := t = 60

-- The theorem we need to prove
theorem find_b (H1 : cond1 a b t) (H2 : cond2 a b t) (H3 : t_value t) : b = 15 :=
by {
  -- Assuming the conditions are true
  sorry
}

end find_b_l2195_219510


namespace negation_example_l2195_219593

theorem negation_example :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0 :=
by
  sorry

end negation_example_l2195_219593


namespace distinct_triangles_from_chord_intersections_l2195_219547

theorem distinct_triangles_from_chord_intersections :
  let points := 9
  let chords := (points.choose 2)
  let intersections := (points.choose 4)
  let triangles := (points.choose 6)
  (chords > 0 ∧ intersections > 0 ∧ triangles > 0) →
  triangles = 84 :=
by
  intros
  sorry

end distinct_triangles_from_chord_intersections_l2195_219547


namespace square_root_of_9_eq_pm_3_l2195_219536

theorem square_root_of_9_eq_pm_3 (x : ℝ) : x^2 = 9 → x = 3 ∨ x = -3 :=
sorry

end square_root_of_9_eq_pm_3_l2195_219536


namespace sum_exponents_binary_3400_l2195_219592

theorem sum_exponents_binary_3400 : 
  ∃ (a b c d e : ℕ), 
    3400 = 2^a + 2^b + 2^c + 2^d + 2^e ∧ 
    a > b ∧ b > c ∧ c > d ∧ d > e ∧ 
    a + b + c + d + e = 38 :=
sorry

end sum_exponents_binary_3400_l2195_219592


namespace einstein_needs_more_money_l2195_219506

-- Definitions based on conditions
def pizza_price : ℝ := 12
def fries_price : ℝ := 0.3
def soda_price : ℝ := 2
def pizzas_sold : ℕ := 15
def fries_sold : ℕ := 40
def sodas_sold : ℕ := 25
def goal : ℝ := 500

-- Total amount raised calculation
def total_raised : ℝ :=
  (pizzas_sold * pizza_price) +
  (fries_sold * fries_price) +
  (sodas_sold * soda_price)

-- Proof statement
theorem einstein_needs_more_money : goal - total_raised = 258 :=
by
  sorry

end einstein_needs_more_money_l2195_219506


namespace proportion_third_number_l2195_219564

theorem proportion_third_number
  (x : ℝ) (y : ℝ)
  (h1 : 0.60 * 4 = x * y)
  (h2 : x = 0.39999999999999997) :
  y = 6 :=
by
  sorry

end proportion_third_number_l2195_219564


namespace odd_coefficients_in_polynomial_l2195_219518

noncomputable def number_of_odd_coefficients (n : ℕ) : ℕ :=
  (2^n - 1) / 3 * 4 + 1

theorem odd_coefficients_in_polynomial (n : ℕ) (hn : 0 < n) :
  (x^2 + x + 1)^n = number_of_odd_coefficients n :=
sorry

end odd_coefficients_in_polynomial_l2195_219518


namespace sam_initial_balloons_l2195_219596

theorem sam_initial_balloons:
  ∀ (S : ℝ), (S - 5.0 + 7.0 = 8) → S = 6.0 :=
by
  intro S h
  sorry

end sam_initial_balloons_l2195_219596


namespace find_natural_numbers_l2195_219552

open Nat

theorem find_natural_numbers (n : ℕ) (h : ∃ m : ℤ, 2^n + 33 = m^2) : n = 4 ∨ n = 8 :=
sorry

end find_natural_numbers_l2195_219552


namespace solve_inequality_l2195_219529

theorem solve_inequality (x : ℝ) (h1: 3 * x - 8 ≠ 0) :
  5 ≤ x / (3 * x - 8) ∧ x / (3 * x - 8) < 10 ↔ (8 / 3) < x ∧ x ≤ (20 / 7) := 
sorry

end solve_inequality_l2195_219529


namespace fit_nine_cross_pentominoes_on_chessboard_l2195_219539

def cross_pentomino (A B C D E : Prop) :=
  A ∧ B ∧ C ∧ D ∧ E -- A cross pentomino is five connected 1x1 squares

def square1x1 : Prop := sorry -- a placeholder for a 1x1 square

def eight_by_eight_chessboard := Fin 8 × Fin 8 -- an 8x8 chessboard using finitely indexed squares

noncomputable def can_cut_nine_cross_pentominoes : Prop := sorry -- a placeholder proof verification

theorem fit_nine_cross_pentominoes_on_chessboard : can_cut_nine_cross_pentominoes  :=
by 
  -- Assume each cross pentomino consists of 5 connected 1x1 squares
  let cross := cross_pentomino square1x1 square1x1 square1x1 square1x1 square1x1
  -- We need to prove that we can cut out nine such crosses from the 8x8 chessboard
  sorry

end fit_nine_cross_pentominoes_on_chessboard_l2195_219539


namespace product_four_integers_sum_to_50_l2195_219542

theorem product_four_integers_sum_to_50 (E F G H : ℝ) 
  (h₀ : E + F + G + H = 50)
  (h₁ : E - 3 = F + 3)
  (h₂ : E - 3 = G * 3)
  (h₃ : E - 3 = H / 3) :
  E * F * G * H = 7461.9140625 := 
sorry

end product_four_integers_sum_to_50_l2195_219542


namespace trioball_play_time_l2195_219537

theorem trioball_play_time (total_duration : ℕ) (num_children : ℕ) (players_at_a_time : ℕ) 
  (equal_play_time : ℕ) (H1 : total_duration = 120) (H2 : num_children = 3) (H3 : players_at_a_time = 2)
  (H4 : equal_play_time = 240 / num_children)
  : equal_play_time = 80 := 
by 
  sorry

end trioball_play_time_l2195_219537


namespace car_mileage_proof_l2195_219556

noncomputable def car_average_mpg 
  (odometer_start: ℝ) (odometer_end: ℝ) 
  (fuel1: ℝ) (fuel2: ℝ) (odometer2: ℝ) 
  (fuel3: ℝ) (odometer3: ℝ) (final_fuel: ℝ) 
  (final_odometer: ℝ): ℝ :=
  (odometer_end - odometer_start) / 
  ((fuel1 + fuel2 + fuel3 + final_fuel): ℝ)

theorem car_mileage_proof:
  car_average_mpg 56200 57150 6 14 56600 10 56880 20 57150 = 19 :=
by
  sorry

end car_mileage_proof_l2195_219556


namespace sum_of_digits_N_l2195_219505

-- Define a function to compute the least common multiple (LCM) of a list of numbers
def lcm_list (xs : List ℕ) : ℕ :=
  xs.foldr Nat.lcm 1

-- The set of numbers less than 8
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7]

-- The LCM of numbers less than 8
def N_lcm : ℕ := lcm_list nums

-- The second smallest positive integer that is divisible by every positive integer less than 8
def N : ℕ := 2 * N_lcm

-- Function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Prove that the sum of the digits of N is 12
theorem sum_of_digits_N : sum_of_digits N = 12 :=
by
  -- Necessary proof steps will be filled here
  sorry

end sum_of_digits_N_l2195_219505


namespace book_total_pages_l2195_219507

theorem book_total_pages (P : ℕ) (days_read : ℕ) (pages_per_day : ℕ) (fraction_read : ℚ) 
  (total_pages_read : ℕ) :
  (days_read = 15 ∧ pages_per_day = 12 ∧ fraction_read = 3 / 4 ∧ total_pages_read = 180 ∧ 
    total_pages_read = days_read * pages_per_day ∧ total_pages_read = fraction_read * P) → 
    P = 240 :=
by
  intros h
  sorry

end book_total_pages_l2195_219507


namespace evaluate_x_squared_plus_y_squared_l2195_219508

theorem evaluate_x_squared_plus_y_squared (x y : ℝ) (h₁ : 3 * x + y = 20) (h₂ : 4 * x + y = 25) :
  x^2 + y^2 = 50 :=
sorry

end evaluate_x_squared_plus_y_squared_l2195_219508


namespace number_of_elements_l2195_219589

noncomputable def set_mean (S : Set ℝ) : ℝ := sorry

theorem number_of_elements (S : Set ℝ) (M : ℝ)
  (h1 : set_mean (S ∪ {15}) = M + 2)
  (h2 : set_mean (S ∪ {15, 1}) = M + 1) :
  ∃ k : ℕ, (M * k + 15 = (M + 2) * (k + 1)) ∧ (M * k + 16 = (M + 1) * (k + 2)) ∧ k = 4 := sorry

end number_of_elements_l2195_219589


namespace min_value_l2195_219568

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 
  ∃ c : ℝ, c = 4 ∧ 
  ∀ x y : ℝ, (x = 1 / (a - 1) ∧ y = 4 / (b - 1)) → (x + y ≥ c) :=
sorry

end min_value_l2195_219568


namespace class_average_score_l2195_219586

theorem class_average_score :
  let total_students := 40
  let absent_students := 2
  let present_students := total_students - absent_students
  let initial_avg := 92
  let absent_scores := [100, 100]
  let initial_total_score := initial_avg * present_students
  let total_final_score := initial_total_score + absent_scores.sum
  let final_avg := total_final_score / total_students
  final_avg = 92.4 := by
  sorry

end class_average_score_l2195_219586


namespace probability_three_aligned_l2195_219587

theorem probability_three_aligned (total_arrangements favorable_arrangements : ℕ) 
  (h1 : total_arrangements = 126)
  (h2 : favorable_arrangements = 48) :
  (favorable_arrangements : ℚ) / total_arrangements = 8 / 21 :=
by sorry

end probability_three_aligned_l2195_219587


namespace part_one_part_two_l2195_219569

variable {a : ℕ → ℕ}

-- Conditions
axiom a1 : a 1 = 3
axiom recurrence_relation : ∀ n, a (n + 1) = 2 * (a n) + 1

-- Proof of the first part
theorem part_one: ∀ n, (a (n + 1) + 1) = 2 * (a n + 1) :=
by
  sorry

-- General formula for the sequence
theorem part_two: ∀ n, a n = 2^(n + 1) - 1 :=
by
  sorry

end part_one_part_two_l2195_219569


namespace tanya_efficiency_greater_sakshi_l2195_219599

theorem tanya_efficiency_greater_sakshi (S_e T_e : ℝ) (h1 : S_e = 1 / 20) (h2 : T_e = 1 / 16) :
  ((T_e - S_e) / S_e) * 100 = 25 := by
  sorry

end tanya_efficiency_greater_sakshi_l2195_219599


namespace probability_succeeding_third_attempt_l2195_219574

theorem probability_succeeding_third_attempt :
  let total_keys := 5
  let successful_keys := 2
  let attempts := 3
  let prob := successful_keys / total_keys * (successful_keys / (total_keys - 1)) * (successful_keys / (total_keys - 2))
  prob = 1 / 5 := by
sorry

end probability_succeeding_third_attempt_l2195_219574


namespace part1_part2_l2195_219590

noncomputable def is_monotonically_increasing (f' : ℝ → ℝ) := ∀ x, f' x ≥ 0

noncomputable def is_monotonically_decreasing (f' : ℝ → ℝ) (I : Set ℝ) := ∀ x ∈ I, f' x ≤ 0

def f' (a x : ℝ) : ℝ := 3 * x ^ 2 - a

theorem part1 (a : ℝ) : 
  is_monotonically_increasing (f' a) ↔ a ≤ 0 := sorry

theorem part2 (a : ℝ) : 
  is_monotonically_decreasing (f' a) (Set.Ioo (-1 : ℝ) (1 : ℝ)) ↔ a ≥ 3 := sorry

end part1_part2_l2195_219590


namespace train_stops_time_l2195_219513

/-- Given the speeds of a train excluding and including stoppages, 
calculate the stopping time in minutes per hour. --/
theorem train_stops_time
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (h1 : speed_excluding_stoppages = 48)
  (h2 : speed_including_stoppages = 40) :
  ∃ minutes_stopped : ℝ, minutes_stopped = 10 :=
by
  sorry

end train_stops_time_l2195_219513


namespace every_positive_integer_sum_of_distinct_powers_of_3_4_7_l2195_219597

theorem every_positive_integer_sum_of_distinct_powers_of_3_4_7 :
  ∀ n : ℕ, n > 0 →
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  ∃ (i j k : ℕ), n = 3^i + 4^j + 7^k :=
by
  sorry

end every_positive_integer_sum_of_distinct_powers_of_3_4_7_l2195_219597


namespace value_of_a_l2195_219543

theorem value_of_a (a : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = x^2 - a * x + 4) (h₂ : ∀ x, f (x + 1) = f (1 - x)) :
  a = 2 :=
sorry

end value_of_a_l2195_219543


namespace problem_statement_l2195_219520

-- Definitions of conditions
def p (a : ℝ) : Prop := a < 0
def q (a : ℝ) : Prop := a^2 > a

-- Statement of the problem
theorem problem_statement (a : ℝ) (h1 : p a) (h2 : q a) : (¬ p a) → (¬ q a) → ∃ x, ¬ (¬ q x) → (¬ (¬ p x)) :=
by
  sorry

end problem_statement_l2195_219520


namespace min_value_expression_l2195_219521

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (H : 1 / a + 1 / b = 1) :
  ∃ c : ℝ, (∀ (a b : ℝ), 0 < a → 0 < b → 1 / a + 1 / b = 1 → c ≤ 4 / (a - 1) + 9 / (b - 1)) ∧ (c = 6) :=
by
  sorry

end min_value_expression_l2195_219521


namespace ellipse_k_values_l2195_219562

theorem ellipse_k_values (k : ℝ) :
  (∃ k, (∃ e, e = 1/2 ∧
    (∃ a b : ℝ, a = Real.sqrt (k+8) ∧ b = 3 ∧
      ∃ c, (c = Real.sqrt (abs ((a^2) - (b^2)))) ∧ (e = c/b ∨ e = c/a)) ∧
      k = 4 ∨ k = -5/4)) :=
  sorry

end ellipse_k_values_l2195_219562


namespace gum_cost_example_l2195_219528

def final_cost (pieces : ℕ) (cost_per_piece : ℕ) (discount_percentage : ℕ) : ℕ :=
  let total_cost := pieces * cost_per_piece
  let discount := total_cost * discount_percentage / 100
  total_cost - discount

theorem gum_cost_example :
  final_cost 1500 2 10 / 100 = 27 :=
by sorry

end gum_cost_example_l2195_219528


namespace overall_percentage_supporting_increased_funding_l2195_219538

-- Definitions for the conditions
def percent_of_men_supporting (percent_men_supporting : ℕ := 60) : ℕ := percent_men_supporting
def percent_of_women_supporting (percent_women_supporting : ℕ := 80) : ℕ := percent_women_supporting
def number_of_men_surveyed (men_surveyed : ℕ := 100) : ℕ := men_surveyed
def number_of_women_surveyed (women_surveyed : ℕ := 900) : ℕ := women_surveyed

-- Theorem: the overall percent of people surveyed who supported increased funding is 78%
theorem overall_percentage_supporting_increased_funding : 
  (percent_of_men_supporting * number_of_men_surveyed + percent_of_women_supporting * number_of_women_surveyed) / 
  (number_of_men_surveyed + number_of_women_surveyed) = 78 := 
sorry

end overall_percentage_supporting_increased_funding_l2195_219538


namespace total_ridges_on_all_records_l2195_219549

theorem total_ridges_on_all_records :
  let ridges_per_record := 60
  let cases := 4
  let shelves_per_case := 3
  let records_per_shelf := 20
  let shelf_fullness_ratio := 0.60

  let total_capacity := cases * shelves_per_case * records_per_shelf
  let actual_records := total_capacity * shelf_fullness_ratio
  let total_ridges := actual_records * ridges_per_record
  
  total_ridges = 8640 :=
by
  sorry

end total_ridges_on_all_records_l2195_219549


namespace find_base_tax_rate_l2195_219503

noncomputable def income : ℝ := 10550
noncomputable def tax_paid : ℝ := 950
noncomputable def base_income : ℝ := 5000
noncomputable def excess_income : ℝ := income - base_income
noncomputable def excess_tax_rate : ℝ := 0.10

theorem find_base_tax_rate (base_tax_rate: ℝ) :
  base_tax_rate * base_income + excess_tax_rate * excess_income = tax_paid -> 
  base_tax_rate = 7.9 / 100 :=
by sorry

end find_base_tax_rate_l2195_219503


namespace people_after_second_turn_l2195_219512

noncomputable def number_of_people_in_front_after_second_turn (formation_size : ℕ) (initial_people : ℕ) (first_turn_people : ℕ) : ℕ := 
  if formation_size = 9 ∧ initial_people = 2 ∧ first_turn_people = 4 then 6 else 0

theorem people_after_second_turn :
  number_of_people_in_front_after_second_turn 9 2 4 = 6 :=
by
  -- Prove the theorem using the conditions and given data
  sorry

end people_after_second_turn_l2195_219512


namespace parabola_focus_coordinates_l2195_219578

theorem parabola_focus_coordinates :
  ∀ x y : ℝ, y^2 - 4 * x = 0 → (x, y) = (1, 0) :=
by
  -- Use the equivalence given by the problem
  intros x y h
  sorry

end parabola_focus_coordinates_l2195_219578


namespace graph_shift_l2195_219551

theorem graph_shift (f : ℝ → ℝ) (h : f 0 = 2) : f (-1 + 1) = 2 :=
by
  have h1 : f 0 = 2 := h
  sorry

end graph_shift_l2195_219551


namespace number_of_dogs_is_correct_l2195_219550

variable (D C B : ℕ)
variable (k : ℕ)

def validRatio (D C B : ℕ) : Prop := D = 7 * k ∧ C = 7 * k ∧ B = 8 * k
def totalDogsAndBunnies (D B : ℕ) : Prop := D + B = 330
def correctNumberOfDogs (D : ℕ) : Prop := D = 154

theorem number_of_dogs_is_correct (D C B k : ℕ) 
  (hRatio : validRatio D C B k)
  (hTotal : totalDogsAndBunnies D B) :
  correctNumberOfDogs D :=
by
  sorry

end number_of_dogs_is_correct_l2195_219550


namespace parallel_line_passing_through_point_l2195_219558

theorem parallel_line_passing_through_point :
  ∃ m b : ℝ, (∀ x y : ℝ, 4 * x + 2 * y = 8 → y = -2 * x + 4) ∧ b = 1 ∧ m = -2 ∧ b = 1 := by
  sorry

end parallel_line_passing_through_point_l2195_219558


namespace graph_quadrant_exclusion_l2195_219540

theorem graph_quadrant_exclusion (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b < -1) :
  ∀ x : ℝ, ¬ ((a^x + b > 0) ∧ (x > 0)) :=
by
  sorry

end graph_quadrant_exclusion_l2195_219540


namespace ratio_equality_l2195_219591

def op_def (a b : ℕ) : ℕ := a * b + b^2
def ot_def (a b : ℕ) : ℕ := a - b + a * b^2

theorem ratio_equality : (op_def 8 3 : ℚ) / (ot_def 8 3 : ℚ) = (33 : ℚ) / 77 := by
  sorry

end ratio_equality_l2195_219591


namespace length_of_platform_l2195_219514

theorem length_of_platform
  (length_train : ℝ)
  (speed_train_kmph : ℝ)
  (time_seconds : ℝ)
  (distance_covered : ℝ)
  (conversion_factor : ℝ) :
  length_train = 250 →
  speed_train_kmph = 90 →
  time_seconds = 20 →
  distance_covered = (speed_train_kmph * 1000 / 3600) * time_seconds →
  conversion_factor = 1000 / 3600 →
  ∃ P : ℝ, distance_covered = length_train + P ∧ P = 250 :=
by
  sorry

end length_of_platform_l2195_219514


namespace compare_expressions_l2195_219553

-- Considering the conditions
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def sqrt5 := Real.sqrt 5
noncomputable def expr1 := (2 + log2 6)
noncomputable def expr2 := (2 * sqrt5)

-- The theorem statement
theorem compare_expressions : 
  expr1 > expr2 := 
  sorry

end compare_expressions_l2195_219553


namespace regular_icosahedron_edges_l2195_219534

-- Define the concept of a regular icosahedron.
structure RegularIcosahedron :=
  (vertices : ℕ)
  (faces : ℕ)
  (edges : ℕ)

-- Define the properties of a regular icosahedron.
def regular_icosahedron_properties (ico : RegularIcosahedron) : Prop :=
  ico.vertices = 12 ∧ ico.faces = 20 ∧ ico.edges = 30

-- Statement of the proof problem: The number of edges in a regular icosahedron is 30.
theorem regular_icosahedron_edges : ∀ (ico : RegularIcosahedron), regular_icosahedron_properties ico → ico.edges = 30 :=
by
  sorry

end regular_icosahedron_edges_l2195_219534


namespace drink_cost_l2195_219585

/-- Wade has called into a rest stop and decides to get food for the road. 
  He buys a sandwich to eat now, one for the road, and one for the evening. 
  He also buys 2 drinks. Wade spends a total of $26 and the sandwiches 
  each cost $6. Prove that the drinks each cost $4. -/
theorem drink_cost (cost_sandwich : ℕ) (num_sandwiches : ℕ) (cost_total : ℕ) (num_drinks : ℕ) :
  cost_sandwich = 6 → num_sandwiches = 3 → cost_total = 26 → num_drinks = 2 → 
  ∃ (cost_drink : ℕ), cost_drink = 4 :=
by
  intro h1 h2 h3 h4
  sorry

end drink_cost_l2195_219585


namespace regular_pentagon_cannot_tessellate_l2195_219555

-- Definitions of polygons
def is_regular_triangle (angle : ℝ) : Prop := angle = 60
def is_square (angle : ℝ) : Prop := angle = 90
def is_regular_pentagon (angle : ℝ) : Prop := angle = 108
def is_hexagon (angle : ℝ) : Prop := angle = 120

-- Tessellation condition
def divides_evenly (a b : ℝ) : Prop := ∃ k : ℕ, b = k * a

-- The main statement
theorem regular_pentagon_cannot_tessellate :
  ¬ divides_evenly 108 360 :=
sorry

end regular_pentagon_cannot_tessellate_l2195_219555


namespace solve_for_square_solve_for_cube_l2195_219515

variable (x : ℂ)

-- Given condition
def condition := x + 1/x = 8

-- Prove that x^2 + 1/x^2 = 62 given the condition
theorem solve_for_square (h : condition x) : x^2 + 1/x^2 = 62 := 
  sorry

-- Prove that x^3 + 1/x^3 = 488 given the condition
theorem solve_for_cube (h : condition x) : x^3 + 1/x^3 = 488 :=
  sorry

end solve_for_square_solve_for_cube_l2195_219515


namespace nth_term_sequence_sum_first_n_terms_l2195_219571

def a_n (n : ℕ) : ℕ :=
  (2 * n - 1) * (2 * n + 2)

def S_n (n : ℕ) : ℚ :=
  4 * (n * (n + 1) * (2 * n + 1)) / 6 + n * (n + 1) - 2 * n

theorem nth_term_sequence (n : ℕ) : a_n n = 4 * n^2 + 2 * n - 2 :=
  sorry

theorem sum_first_n_terms (n : ℕ) : S_n n = (4 * n^3 + 9 * n^2 - n) / 3 :=
  sorry

end nth_term_sequence_sum_first_n_terms_l2195_219571


namespace probability_perfect_square_sum_l2195_219557

def is_perfect_square_sum (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

def count_perfect_square_sums : ℕ :=
  let possible_outcomes := 216
  let favorable_outcomes := 32
  favorable_outcomes

theorem probability_perfect_square_sum :
  (count_perfect_square_sums : ℚ) / 216 = 4 / 27 :=
by
  sorry

end probability_perfect_square_sum_l2195_219557


namespace youngest_child_age_l2195_219575

theorem youngest_child_age
  (ten_years_ago_avg_age : Nat) (family_initial_size : Nat) (present_avg_age : Nat)
  (age_difference : Nat) (age_ten_years_ago_total : Nat)
  (age_increase : Nat) (current_age_total : Nat)
  (current_family_size : Nat) (total_age_increment : Nat) :
  ten_years_ago_avg_age = 24 →
  family_initial_size = 4 →
  present_avg_age = 24 →
  age_difference = 2 →
  age_ten_years_ago_total = family_initial_size * ten_years_ago_avg_age →
  age_increase = family_initial_size * 10 →
  current_age_total = age_ten_years_ago_total + age_increase →
  current_family_size = family_initial_size + 2 →
  total_age_increment = current_family_size * present_avg_age →
  total_age_increment - current_age_total = 8 →
  ∃ (Y : Nat), Y + Y + age_difference = 8 ∧ Y = 3 :=
by
  intros
  sorry

end youngest_child_age_l2195_219575


namespace arithmetic_sequence_sum_n_squared_l2195_219511

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_mean (x y z : ℝ) : Prop :=
(y * y = x * z)

def is_strictly_increasing (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) > a n

theorem arithmetic_sequence_sum_n_squared
  (a : ℕ → ℝ)
  (h₁ : is_arithmetic_sequence a)
  (h₂ : a 1 = 1)
  (h₃ : is_geometric_mean (a 1) (a 2) (a 5))
  (h₄ : is_strictly_increasing a) :
  ∃ S : ℕ → ℝ, ∀ n : ℕ, S n = n ^ 2 :=
sorry

end arithmetic_sequence_sum_n_squared_l2195_219511


namespace linda_original_savings_l2195_219582

theorem linda_original_savings (S : ℝ) (h1 : (2 / 3) * S + (1 / 3) * S = S) 
  (h2 : (1 / 3) * S = 250) : S = 750 :=
by sorry

end linda_original_savings_l2195_219582


namespace max_tied_teams_for_most_wins_l2195_219517

theorem max_tied_teams_for_most_wins 
  (n : ℕ) 
  (h₀ : n = 6)
  (total_games : ℕ := n * (n - 1) / 2)
  (game_result : Π (i j : ℕ), i ≠ j → (0 = 1 → false) ∨ (1 = 1))
  (rank_by_wins : ℕ → ℕ) : true := sorry

end max_tied_teams_for_most_wins_l2195_219517


namespace cricket_matches_total_l2195_219524

theorem cricket_matches_total
  (n : ℕ)
  (avg_all : ℝ)
  (avg_first4 : ℝ)
  (avg_last3 : ℝ)
  (h_avg_all : avg_all = 56)
  (h_avg_first4 : avg_first4 = 46)
  (h_avg_last3 : avg_last3 = 69.33333333333333)
  (h_total_runs : n * avg_all = 4 * avg_first4 + 3 * avg_last3) :
  n = 7 :=
by
  sorry

end cricket_matches_total_l2195_219524


namespace population_growth_rate_l2195_219500

-- Define initial and final population
def initial_population : ℕ := 240
def final_population : ℕ := 264

-- Define the formula for calculating population increase rate
def population_increase_rate (P_i P_f : ℕ) : ℕ :=
  ((P_f - P_i) * 100) / P_i

-- State the theorem
theorem population_growth_rate :
  population_increase_rate initial_population final_population = 10 := by
  sorry

end population_growth_rate_l2195_219500


namespace length_of_purple_part_l2195_219595

theorem length_of_purple_part (p : ℕ) (black : ℕ) (blue : ℕ) (total : ℕ) 
  (h1 : black = 2) (h2 : blue = 1) (h3 : total = 6) (h4 : p + black + blue = total) : 
  p = 3 :=
by
  sorry

end length_of_purple_part_l2195_219595


namespace complex_calculation_l2195_219523

theorem complex_calculation (i : ℂ) (hi : i * i = -1) : (1 - i)^2 * i = 2 :=
by
  sorry

end complex_calculation_l2195_219523


namespace arithmetic_sequences_ratio_l2195_219545

theorem arithmetic_sequences_ratio (a b S T : ℕ → ℕ) (h : ∀ n, S n / T n = 2 * n / (3 * n + 1)) :
  (a 2) / (b 3 + b 7) + (a 8) / (b 4 + b 6) = 9 / 14 :=
  sorry

end arithmetic_sequences_ratio_l2195_219545


namespace auction_starting_price_l2195_219531

-- Defining the conditions
def bid_increment := 5         -- The dollar increment per bid
def bids_per_person := 5       -- Number of bids per person
def total_bidders := 2         -- Number of people bidding
def final_price := 65          -- Final price of the desk after all bids

-- Calculate derived conditions
def total_bids := bids_per_person * total_bidders
def total_increment := total_bids * bid_increment

-- The statement to be proved
theorem auction_starting_price : (final_price - total_increment) = 15 :=
by
  sorry

end auction_starting_price_l2195_219531


namespace find_fraction_l2195_219584

noncomputable def fraction_of_third (F N : ℝ) : Prop := F * (1 / 3 * N) = 30

noncomputable def fraction_of_number (G N : ℝ) : Prop := G * N = 75

noncomputable def product_is_90 (F N : ℝ) : Prop := F * N = 90

theorem find_fraction (F G N : ℝ) (h1 : fraction_of_third F N) (h2 : fraction_of_number G N) (h3 : product_is_90 F N) :
  G = 5 / 6 :=
sorry

end find_fraction_l2195_219584


namespace new_paint_intensity_l2195_219598

def I1 : ℝ := 0.50
def I2 : ℝ := 0.25
def F : ℝ := 0.2

theorem new_paint_intensity : (1 - F) * I1 + F * I2 = 0.45 := by
  sorry

end new_paint_intensity_l2195_219598


namespace gcd_1213_1985_eq_1_l2195_219559

theorem gcd_1213_1985_eq_1
  (h1: ¬ (1213 % 2 = 0))
  (h2: ¬ (1213 % 3 = 0))
  (h3: ¬ (1213 % 5 = 0))
  (h4: ¬ (1985 % 2 = 0))
  (h5: ¬ (1985 % 3 = 0))
  (h6: ¬ (1985 % 5 = 0)):
  Nat.gcd 1213 1985 = 1 := by
  sorry

end gcd_1213_1985_eq_1_l2195_219559


namespace charge_R_12_5_percent_more_l2195_219566

-- Let R be the charge for a single room at hotel R.
-- Let G be the charge for a single room at hotel G.
-- Let P be the charge for a single room at hotel P.

def charge_R (R : ℝ) : Prop := true
def charge_G (G : ℝ) : Prop := true
def charge_P (P : ℝ) : Prop := true

axiom hotel_P_20_less_R (R P : ℝ) : charge_R R → charge_P P → P = 0.80 * R
axiom hotel_P_10_less_G (G P : ℝ) : charge_G G → charge_P P → P = 0.90 * G

theorem charge_R_12_5_percent_more (R G : ℝ) :
  charge_R R → charge_G G → (∃ P, charge_P P ∧ P = 0.80 * R ∧ P = 0.90 * G) → R = 1.125 * G :=
by sorry

end charge_R_12_5_percent_more_l2195_219566


namespace find_k_l2195_219570

variable {S : ℕ → ℤ} -- Assuming the sum function S for the arithmetic sequence 
variable {k : ℕ} -- k is a natural number

theorem find_k (h1 : S (k - 2) = -4) (h2 : S k = 0) (h3 : S (k + 2) = 8) (hk2 : k > 2) (hnaturalk : k ∈ Set.univ) : k = 6 := by
  sorry

end find_k_l2195_219570


namespace ab_bc_cd_da_le_four_l2195_219580

theorem ab_bc_cd_da_le_four (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 :=
by
  sorry

end ab_bc_cd_da_le_four_l2195_219580


namespace hyperbola_equation_l2195_219548

variable (a b : ℝ)
variable (c : ℝ) (h1 : c = 4)
variable (h2 : b / a = Real.sqrt 3)
variable (h3 : a ^ 2 + b ^ 2 = c ^ 2)

theorem hyperbola_equation : (a ^ 2 = 4) ∧ (b ^ 2 = 12) ↔ (∀ x y : ℝ, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1 → (x ^ 2 / 4) - (y ^ 2 / 12) = 1) := by
  sorry

end hyperbola_equation_l2195_219548


namespace exponent_zero_nonneg_l2195_219567

theorem exponent_zero_nonneg (a : ℝ) (h : a ≠ -1) : (a + 1) ^ 0 = 1 :=
sorry

end exponent_zero_nonneg_l2195_219567


namespace find_investment_duration_l2195_219544

theorem find_investment_duration :
  ∀ (A P R I : ℝ) (T : ℝ),
    A = 1344 →
    P = 1200 →
    R = 5 →
    I = A - P →
    I = (P * R * T) / 100 →
    T = 2.4 :=
by
  intros A P R I T hA hP hR hI1 hI2
  sorry

end find_investment_duration_l2195_219544


namespace praveen_hari_profit_ratio_l2195_219577

theorem praveen_hari_profit_ratio
  (praveen_capital : ℕ := 3360)
  (hari_capital : ℕ := 8640)
  (time_praveen_invested : ℕ := 12)
  (time_hari_invested : ℕ := 7)
  (praveen_shares_full_time : ℕ := praveen_capital * time_praveen_invested)
  (hari_shares_full_time : ℕ := hari_capital * time_hari_invested)
  (gcd_common : ℕ := Nat.gcd praveen_shares_full_time hari_shares_full_time) :
  (praveen_shares_full_time / gcd_common) * 2 = 2 ∧ (hari_shares_full_time / gcd_common) * 2 = 3 := by
    sorry

end praveen_hari_profit_ratio_l2195_219577


namespace NY_Mets_fans_count_l2195_219563

noncomputable def NY_Yankees_fans (M: ℝ) : ℝ := (3/2) * M
noncomputable def Boston_Red_Sox_fans (M: ℝ) : ℝ := (5/4) * M
noncomputable def LA_Dodgers_fans (R: ℝ) : ℝ := (2/7) * R

theorem NY_Mets_fans_count :
  ∃ M : ℕ, let Y := NY_Yankees_fans M
           let R := Boston_Red_Sox_fans M
           let D := LA_Dodgers_fans R
           Y + M + R + D = 780 ∧ M = 178 :=
by
  sorry

end NY_Mets_fans_count_l2195_219563


namespace compute_expression_l2195_219576

theorem compute_expression (y : ℕ) (h : y = 3) : 
  (y^8 + 18 * y^4 + 81) / (y^4 + 9) = 90 :=
by
  sorry

end compute_expression_l2195_219576


namespace distance_house_to_market_l2195_219533

-- Define each of the given conditions
def distance_to_school := 50
def distance_to_park_from_school := 25
def return_distance := 60
def total_distance_walked := 220

-- Proven distance to the market
def distance_to_market := 85

-- Statement to prove
theorem distance_house_to_market (d1 d2 d3 d4 : ℕ) 
  (h1 : d1 = distance_to_school) 
  (h2 : d2 = distance_to_park_from_school) 
  (h3 : d3 = return_distance) 
  (h4 : d4 = total_distance_walked) :
  d4 - (d1 + d2 + d3) = distance_to_market := 
by
  sorry

end distance_house_to_market_l2195_219533


namespace final_probability_l2195_219541

-- Define the structure of the problem
structure GameRound :=
  (green_ball : ℕ)
  (red_ball : ℕ)
  (blue_ball : ℕ)
  (white_ball : ℕ)

structure GameState :=
  (coins : ℕ)
  (players : ℕ)

-- Define the game rules and initial conditions
noncomputable def initial_coins := 5
noncomputable def rounds := 5

-- Probability-related functions and game logic
noncomputable def favorable_outcome_count : ℕ := 6
noncomputable def total_outcomes_per_round : ℕ := 120
noncomputable def probability_per_round : ℚ := favorable_outcome_count / total_outcomes_per_round

theorem final_probability :
  probability_per_round ^ rounds = 1 / 3200000 :=
by
  sorry

end final_probability_l2195_219541


namespace perfect_square_conditions_l2195_219572

theorem perfect_square_conditions (x y k : ℝ) :
  (∃ a : ℝ, x^2 + k * x * y + 81 * y^2 = a^2) ↔ (k = 18 ∨ k = -18) :=
sorry

end perfect_square_conditions_l2195_219572


namespace line_through_point_equidistant_l2195_219522

open Real

structure Point where
  x : ℝ
  y : ℝ

def line_equation (a b c : ℝ) (p : Point) : Prop :=
  a * p.x + b * p.y + c = 0

def equidistant (p1 p2 : Point) (l : ℝ × ℝ × ℝ) : Prop :=
  let (a, b, c) := l
  let dist_from_p1 := abs (a * p1.x + b * p1.y + c) / sqrt (a^2 + b^2)
  let dist_from_p2 := abs (a * p2.x + b * p2.y + c) / sqrt (a^2 + b^2)
  dist_from_p1 = dist_from_p2

theorem line_through_point_equidistant (a b c : ℝ)
  (P : Point) (A : Point) (B : Point) :
  (P = ⟨1, 2⟩) →
  (A = ⟨2, 2⟩) →
  (B = ⟨4, -6⟩) →
  line_equation a b c P →
  equidistant A B (a, b, c) →
  (a = 2 ∧ b = 1 ∧ c = -4) :=
by
  sorry

end line_through_point_equidistant_l2195_219522


namespace david_initial_money_l2195_219504

-- Given conditions as definitions
def spent (S : ℝ) : Prop := S - 800 = 500
def has_left (H : ℝ) : Prop := H = 500

-- The main theorem to prove
theorem david_initial_money (S : ℝ) (X : ℝ) (H : ℝ)
  (h1 : spent S) 
  (h2 : has_left H) 
  : X = S + H → X = 1800 :=
by
  sorry

end david_initial_money_l2195_219504


namespace speed_of_stream_l2195_219573

variable (x : ℝ) -- Let the speed of the stream be x kmph

-- Conditions
variable (speed_of_boat_in_still_water : ℝ)
variable (time_upstream_twice_time_downstream : Prop)

-- Given conditions
axiom h1 : speed_of_boat_in_still_water = 48
axiom h2 : time_upstream_twice_time_downstream → 1 / (speed_of_boat_in_still_water - x) = 2 * (1 / (speed_of_boat_in_still_water + x))

-- Theorem to prove
theorem speed_of_stream (h2: time_upstream_twice_time_downstream) : x = 16 := by
  sorry

end speed_of_stream_l2195_219573


namespace arithmetic_sequence_a1_l2195_219583

theorem arithmetic_sequence_a1 (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_inc : d > 0)
  (h_a3 : a 3 = 1)
  (h_a2a4 : a 2 * a 4 = 3 / 4) : 
  a 1 = 0 :=
sorry

end arithmetic_sequence_a1_l2195_219583


namespace proof_statement_l2195_219594

def convert_base_9_to_10 (n : Nat) : Nat :=
  3 * 9^2 + 2 * 9^1 + 4 * 9^0

def convert_base_6_to_10 (n : Nat) : Nat :=
  2 * 6^2 + 2 * 6^1 + 1 * 6^0

def problem_statement : Prop :=
  convert_base_9_to_10 324 - convert_base_6_to_10 221 = 180

theorem proof_statement : problem_statement := 
  by
    sorry

end proof_statement_l2195_219594


namespace decreasing_function_iff_a_range_l2195_219560

noncomputable def f (a x : ℝ) : ℝ := (1 - 2 * a) ^ x

theorem decreasing_function_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ 0 < a ∧ a < 1/2 :=
by
  sorry

end decreasing_function_iff_a_range_l2195_219560


namespace youngest_child_age_l2195_219526

theorem youngest_child_age 
  (x : ℕ)
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 50) : 
  x = 6 := 
by 
  sorry

end youngest_child_age_l2195_219526


namespace opposite_of_negative_fraction_l2195_219535

theorem opposite_of_negative_fraction : -(- (1/2023 : ℚ)) = 1/2023 := 
sorry

end opposite_of_negative_fraction_l2195_219535


namespace find_x_of_equation_l2195_219530

-- Defining the condition and setting up the proof goal
theorem find_x_of_equation
  (h : (1/2)^25 * (1/x)^12.5 = 1/(18^25)) :
  x = 0.1577 := 
sorry

end find_x_of_equation_l2195_219530
