import Mathlib

namespace NUMINAMATH_GPT_trajectory_of_point_A_l571_57111

theorem trajectory_of_point_A (m : ℝ) (A B C : ℝ × ℝ) (hBC : B = (-1, 0) ∧ C = (1, 0)) (hBC_dist : dist B C = 2)
  (hRatio : dist A B / dist A C = m) :
  (m = 1 → ∀ x y : ℝ, A = (x, y) → x = 0) ∧
  (m = 0 → ∀ x y : ℝ, A = (x, y) → x^2 + y^2 - 2 * x + 1 = 0) ∧
  (m ≠ 0 ∧ m ≠ 1 → ∀ x y : ℝ, A = (x, y) → (x + (1 + m^2) / (1 - m^2))^2 + y^2 = (2 * m / (1 - m^2))^2) := 
sorry

end NUMINAMATH_GPT_trajectory_of_point_A_l571_57111


namespace NUMINAMATH_GPT_longer_side_of_rectangle_l571_57122

theorem longer_side_of_rectangle 
  (radius : ℝ) (A_rectangle : ℝ) (shorter_side : ℝ) 
  (h1 : radius = 6)
  (h2 : A_rectangle = 3 * (π * radius^2))
  (h3 : shorter_side = 2 * 2 * radius) :
  (A_rectangle / shorter_side) = 4.5 * π :=
by
  sorry

end NUMINAMATH_GPT_longer_side_of_rectangle_l571_57122


namespace NUMINAMATH_GPT_parity_equiv_l571_57121

open Nat

theorem parity_equiv (p q : ℕ) : (Even (p^3 - q^3) ↔ Even (p + q)) :=
by
  sorry

end NUMINAMATH_GPT_parity_equiv_l571_57121


namespace NUMINAMATH_GPT_geom_seq_problem_l571_57151

variable {a : ℕ → ℝ}  -- positive geometric sequence

-- Conditions
def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n, a n = a 0 * r^n

theorem geom_seq_problem
  (h_geom : geom_seq a)
  (cond : a 0 * a 4 + 2 * a 2 * a 4 + a 2 * a 6 = 25) :
  a 2 + a 4 = 5 :=
sorry

end NUMINAMATH_GPT_geom_seq_problem_l571_57151


namespace NUMINAMATH_GPT_sum_eight_smallest_multiples_of_12_l571_57106

theorem sum_eight_smallest_multiples_of_12 :
  let series := (List.range 8).map (λ k => 12 * (k + 1))
  series.sum = 432 :=
by
  sorry

end NUMINAMATH_GPT_sum_eight_smallest_multiples_of_12_l571_57106


namespace NUMINAMATH_GPT_quadratic_eq_complete_square_l571_57125

theorem quadratic_eq_complete_square (x p q : ℝ) (h : 9 * x^2 - 54 * x + 63 = 0) 
(h_trans : (x + p)^2 = q) : p + q = -1 := sorry

end NUMINAMATH_GPT_quadratic_eq_complete_square_l571_57125


namespace NUMINAMATH_GPT_find_ab_l571_57115

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l571_57115


namespace NUMINAMATH_GPT_order_of_abc_l571_57135

noncomputable def a := Real.log 6 / Real.log 0.7
noncomputable def b := Real.rpow 6 0.7
noncomputable def c := Real.rpow 0.7 0.6

theorem order_of_abc : b > c ∧ c > a := by
  sorry

end NUMINAMATH_GPT_order_of_abc_l571_57135


namespace NUMINAMATH_GPT_at_least_eight_composites_l571_57126

theorem at_least_eight_composites (n : ℕ) (h : n > 1000) :
  ∃ (comps : Finset ℕ), 
    comps.card ≥ 8 ∧ 
    (∀ x ∈ comps, ¬Prime x) ∧ 
    (∀ k, k < 12 → n + k ∈ comps ∨ Prime (n + k)) :=
by
  sorry

end NUMINAMATH_GPT_at_least_eight_composites_l571_57126


namespace NUMINAMATH_GPT_box_volume_l571_57107

def volume_of_box (l w h : ℝ) : ℝ := l * w * h

theorem box_volume (l w h : ℝ) (hlw : l * w = 36) (hwh : w * h = 18) (hlh : l * h = 8) : volume_of_box l w h = 72 :=
by
  sorry

end NUMINAMATH_GPT_box_volume_l571_57107


namespace NUMINAMATH_GPT_transformations_result_l571_57162

theorem transformations_result :
  ∃ (r g : ℕ), r + g = 15 ∧ 
  21 + r - 5 * g = 0 ∧ 
  30 - 2 * r + 2 * g = 24 :=
by
  sorry

end NUMINAMATH_GPT_transformations_result_l571_57162


namespace NUMINAMATH_GPT_sequence_term_general_formula_l571_57101

theorem sequence_term_general_formula (S : ℕ → ℚ) (a : ℕ → ℚ) :
  (∀ n, S n = n^2 + (1/2)*n + 5) →
  (∀ n, (n ≥ 2) → a n = S n - S (n - 1)) →
  a 1 = 13/2 →
  (∀ n, a n = if n = 1 then 13/2 else 2*n - 1/2) :=
by
  intros hS ha h1
  sorry

end NUMINAMATH_GPT_sequence_term_general_formula_l571_57101


namespace NUMINAMATH_GPT_base_10_to_base_7_equiv_base_10_to_base_7_678_l571_57178

theorem base_10_to_base_7_equiv : (678 : ℕ) = 1 * 7^3 + 6 * 7^2 + 5 * 7^1 + 6 * 7^0 := 
by
  -- proof steps would go here
  sorry

theorem base_10_to_base_7_678 : "678 in base-7" = "1656" := 
by
  have h1 := base_10_to_base_7_equiv
  -- additional proof steps to show 1 * 7^3 + 6 * 7^2 + 5 * 7^1 + 6 * 7^0 = 1656 in base-7
  sorry

end NUMINAMATH_GPT_base_10_to_base_7_equiv_base_10_to_base_7_678_l571_57178


namespace NUMINAMATH_GPT_find_y_from_equation_l571_57170

theorem find_y_from_equation :
  ∀ y : ℕ, (12 ^ 3 * 6 ^ 4) / y = 5184 → y = 432 :=
by
  sorry

end NUMINAMATH_GPT_find_y_from_equation_l571_57170


namespace NUMINAMATH_GPT_initially_calculated_average_height_l571_57142

theorem initially_calculated_average_height 
    (students : ℕ) (incorrect_height : ℕ) (correct_height : ℕ) (actual_avg_height : ℝ) 
    (A : ℝ) 
    (h_students : students = 30) 
    (h_incorrect_height : incorrect_height = 151) 
    (h_correct_height : correct_height = 136) 
    (h_actual_avg_height : actual_avg_height = 174.5)
    (h_A_definition : (students : ℝ) * A + (incorrect_height - correct_height) = (students : ℝ) * actual_avg_height) : 
    A = 174 := 
by sorry

end NUMINAMATH_GPT_initially_calculated_average_height_l571_57142


namespace NUMINAMATH_GPT_solve_for_x_l571_57189

theorem solve_for_x : 
  ∃ x : ℚ, x^2 + 145 = (x - 19)^2 ∧ x = 108 / 19 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l571_57189


namespace NUMINAMATH_GPT_no_such_n_exists_l571_57131

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, n * sum_of_digits n = 100200300 :=
by
  sorry

end NUMINAMATH_GPT_no_such_n_exists_l571_57131


namespace NUMINAMATH_GPT_largest_side_of_rectangle_l571_57127

theorem largest_side_of_rectangle (l w : ℕ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 1920) : max l w = 101 := 
by
  sorry

end NUMINAMATH_GPT_largest_side_of_rectangle_l571_57127


namespace NUMINAMATH_GPT_running_track_diameter_l571_57155

theorem running_track_diameter 
  (running_track_width : ℕ) 
  (garden_ring_width : ℕ) 
  (play_area_diameter : ℕ) 
  (h1 : running_track_width = 4) 
  (h2 : garden_ring_width = 6) 
  (h3 : play_area_diameter = 14) :
  (2 * ((play_area_diameter / 2) + garden_ring_width + running_track_width)) = 34 := 
by
  sorry

end NUMINAMATH_GPT_running_track_diameter_l571_57155


namespace NUMINAMATH_GPT_mike_initial_marbles_l571_57161

-- Defining the conditions
def gave_marble (initial_marbles : ℕ) (given_marbles : ℕ) : ℕ := initial_marbles - given_marbles
def marbles_left (initial_marbles : ℕ) (given_marbles : ℕ) : ℕ := initial_marbles - given_marbles

-- Using the given conditions
def initial_mike_marbles : ℕ := 8
def given_marbles : ℕ := 4
def remaining_marbles : ℕ := 4

-- Proving the statement
theorem mike_initial_marbles :
  initial_mike_marbles - given_marbles = remaining_marbles :=
by
  -- The proof
  sorry

end NUMINAMATH_GPT_mike_initial_marbles_l571_57161


namespace NUMINAMATH_GPT_students_left_zoo_l571_57141

theorem students_left_zoo
  (students_first_class students_second_class : ℕ)
  (chaperones teachers : ℕ)
  (initial_individuals remaining_individuals : ℕ)
  (chaperones_left remaining_individuals_after_chaperones_left : ℕ)
  (remaining_students initial_students : ℕ)
  (H1 : students_first_class = 10)
  (H2 : students_second_class = 10)
  (H3 : chaperones = 5)
  (H4 : teachers = 2)
  (H5 : initial_individuals = students_first_class + students_second_class + chaperones + teachers) 
  (H6 : initial_individuals = 27)
  (H7 : remaining_individuals = 15)
  (H8 : chaperones_left = 2)
  (H9 : remaining_individuals_after_chaperones_left = remaining_individuals - chaperones_left)
  (H10 : remaining_individuals_after_chaperones_left = 13)
  (H11 : remaining_students = remaining_individuals_after_chaperones_left - teachers)
  (H12 : remaining_students = 11)
  (H13 : initial_students = students_first_class + students_second_class)
  (H14 : initial_students = 20) :
  20 - 11 = 9 :=
by sorry

end NUMINAMATH_GPT_students_left_zoo_l571_57141


namespace NUMINAMATH_GPT_ajhsme_1989_reappears_at_12_l571_57171

def cycle_length_letters : ℕ := 6
def cycle_length_digits  : ℕ := 4
def target_position : ℕ := Nat.lcm cycle_length_letters cycle_length_digits

theorem ajhsme_1989_reappears_at_12 :
  target_position = 12 :=
by
  -- Proof steps can be filled in here
  sorry

end NUMINAMATH_GPT_ajhsme_1989_reappears_at_12_l571_57171


namespace NUMINAMATH_GPT_jenna_weight_lift_l571_57187

theorem jenna_weight_lift:
  ∀ (n : Nat), (2 * 10 * 25 = 500) ∧ (15 * n >= 500) ∧ (n = Nat.ceil (500 / 15 : ℝ))
  → n = 34 := 
by
  intros n h
  have h₀ : 2 * 10 * 25 = 500 := h.1
  have h₁ : 15 * n >= 500 := h.2.1
  have h₂ : n = Nat.ceil (500 / 15 : ℝ) := h.2.2
  sorry

end NUMINAMATH_GPT_jenna_weight_lift_l571_57187


namespace NUMINAMATH_GPT_number_of_wheels_on_each_bicycle_l571_57129

theorem number_of_wheels_on_each_bicycle 
  (num_bicycles : ℕ)
  (num_tricycles : ℕ)
  (wheels_per_tricycle : ℕ)
  (total_wheels : ℕ)
  (h_bicycles : num_bicycles = 24)
  (h_tricycles : num_tricycles = 14)
  (h_wheels_tricycle : wheels_per_tricycle = 3)
  (h_total_wheels : total_wheels = 90) :
  2 * num_bicycles + 3 * num_tricycles = 90 → 
  num_bicycles = 24 → 
  num_tricycles = 14 → 
  wheels_per_tricycle = 3 → 
  total_wheels = 90 → 
  ∃ b : ℕ, b = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_wheels_on_each_bicycle_l571_57129


namespace NUMINAMATH_GPT_min_value_of_f_l571_57109

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 4*x + 5) * (x^2 + 4*x + 2) + 2*x^2 + 8*x + 1

theorem min_value_of_f : ∃ x : ℝ, f x = -9 :=
  sorry

end NUMINAMATH_GPT_min_value_of_f_l571_57109


namespace NUMINAMATH_GPT_reporter_earnings_per_hour_l571_57144

/-- A reporter's expected earnings per hour if she writes the entire time. -/
theorem reporter_earnings_per_hour :
  ∀ (word_earnings: ℝ) (article_earnings: ℝ) (stories: ℕ) (hours: ℝ) (words_per_minute: ℕ),
  word_earnings = 0.1 →
  article_earnings = 60 →
  stories = 3 →
  hours = 4 →
  words_per_minute = 10 →
  (stories * article_earnings + (hours * 60 * words_per_minute) * word_earnings) / hours = 105 :=
by
  intros word_earnings article_earnings stories hours words_per_minute
  intros h_word_earnings h_article_earnings h_stories h_hours h_words_per_minute
  sorry

end NUMINAMATH_GPT_reporter_earnings_per_hour_l571_57144


namespace NUMINAMATH_GPT_terminating_decimals_count_l571_57137

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end NUMINAMATH_GPT_terminating_decimals_count_l571_57137


namespace NUMINAMATH_GPT_elmer_more_than_penelope_l571_57167

def penelope_food_per_day : ℕ := 20
def greta_food_factor : ℕ := 10
def milton_food_factor : ℤ := 1 / 100
def elmer_food_factor : ℕ := 4000

theorem elmer_more_than_penelope :
  (elmer_food_factor * (milton_food_factor * (penelope_food_per_day / greta_food_factor))) - penelope_food_per_day = 60 := 
sorry

end NUMINAMATH_GPT_elmer_more_than_penelope_l571_57167


namespace NUMINAMATH_GPT_normal_price_of_article_l571_57120

theorem normal_price_of_article (P : ℝ) (h : 0.9 * 0.8 * P = 144) : P = 200 :=
sorry

end NUMINAMATH_GPT_normal_price_of_article_l571_57120


namespace NUMINAMATH_GPT_inequality_proof_l571_57146

theorem inequality_proof 
  (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (sum_eq_one : a + b + c + d = 1) :
  (a^2 / (1 + a)) + (b^2 / (1 + b)) + (c^2 / (1 + c)) + (d^2 / (1 + d)) ≥ 1/5 := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_proof_l571_57146


namespace NUMINAMATH_GPT_orange_sacks_after_95_days_l571_57179

-- Define the conditions as functions or constants
def harvest_per_day : ℕ := 150
def discard_per_day : ℕ := 135
def days_of_harvest : ℕ := 95

-- State the problem formally
theorem orange_sacks_after_95_days :
  (harvest_per_day - discard_per_day) * days_of_harvest = 1425 := 
by 
  sorry

end NUMINAMATH_GPT_orange_sacks_after_95_days_l571_57179


namespace NUMINAMATH_GPT_fraction_beans_remain_l571_57169

theorem fraction_beans_remain (J B B_remain : ℝ) 
  (h1 : J = 0.10 * (J + B)) 
  (h2 : J + B_remain = 0.60 * (J + B)) : 
  B_remain / B = 5 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_beans_remain_l571_57169


namespace NUMINAMATH_GPT_area_of_triangle_tangent_at_pi_div_two_l571_57186

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem area_of_triangle_tangent_at_pi_div_two :
  let x := Real.pi / 2
  let slope := 1 + Real.cos x
  let point := (x, f x)
  let intercept_y := f x - slope * x
  let x_intercept := -intercept_y / slope
  let y_intercept := intercept_y
  (1 / 2) * x_intercept * y_intercept = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_tangent_at_pi_div_two_l571_57186


namespace NUMINAMATH_GPT_inheritance_shares_l571_57124

theorem inheritance_shares (A B : ℝ) (h1: A + B = 100) (h2: (1/4) * B - (1/3) * A = 11) : 
  A = 24 ∧ B = 76 := 
by 
  sorry

end NUMINAMATH_GPT_inheritance_shares_l571_57124


namespace NUMINAMATH_GPT_soccer_players_count_l571_57195

theorem soccer_players_count (total_socks : ℕ) (P : ℕ) 
  (h_total_socks : total_socks = 22)
  (h_each_player_contributes : ∀ p : ℕ, p = P → total_socks = 2 * P) :
  P = 11 :=
by
  sorry

end NUMINAMATH_GPT_soccer_players_count_l571_57195


namespace NUMINAMATH_GPT_sample_size_l571_57100

theorem sample_size (k n : ℕ) (r : 2 * k + 3 * k + 5 * k = 10 * k) (h : 3 * k = 12) : n = 40 :=
by {
    -- here, we will provide a proof to demonstrate that n = 40 given the conditions
    sorry
}

end NUMINAMATH_GPT_sample_size_l571_57100


namespace NUMINAMATH_GPT_group_formations_at_fair_l571_57192

theorem group_formations_at_fair : 
  (Nat.choose 7 3) * (Nat.choose 4 4) = 35 := by
  sorry

end NUMINAMATH_GPT_group_formations_at_fair_l571_57192


namespace NUMINAMATH_GPT_how_many_pairs_of_shoes_l571_57140

theorem how_many_pairs_of_shoes (l k : ℕ) (h_l : l = 52) (h_k : k = 2) : l / k = 26 := by
  sorry

end NUMINAMATH_GPT_how_many_pairs_of_shoes_l571_57140


namespace NUMINAMATH_GPT_sequence_sum_l571_57134

theorem sequence_sum (a : ℕ → ℝ) (h_seq : ∀ n, a (n + 1) = (1/3) * a n) (h_a4a5 : a 4 + a 5 = 4) :
    a 2 + a 3 = 36 :=
    sorry

end NUMINAMATH_GPT_sequence_sum_l571_57134


namespace NUMINAMATH_GPT_minimum_value_of_expression_l571_57130

theorem minimum_value_of_expression (x y : ℝ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : 2 * x + 3 * y = 8) : 
  (∀ a b, a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 8 → (2 / a + 3 / b) ≥ 25 / 8) ∧ 
  (∃ a b, a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 8 ∧ 2 / a + 3 / b = 25 / 8) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l571_57130


namespace NUMINAMATH_GPT_smallest_solution_exists_l571_57116

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end NUMINAMATH_GPT_smallest_solution_exists_l571_57116


namespace NUMINAMATH_GPT_nadine_hosing_time_l571_57160

theorem nadine_hosing_time (shampoos : ℕ) (time_per_shampoo : ℕ) (total_cleaning_time : ℕ) 
  (h1 : shampoos = 3) (h2 : time_per_shampoo = 15) (h3 : total_cleaning_time = 55) : 
  ∃ t : ℕ, t = total_cleaning_time - shampoos * time_per_shampoo ∧ t = 10 := 
by
  sorry

end NUMINAMATH_GPT_nadine_hosing_time_l571_57160


namespace NUMINAMATH_GPT_integrate_differential_eq_l571_57199

theorem integrate_differential_eq {x y C : ℝ} {y' : ℝ → ℝ → ℝ} (h : ∀ x y, (4 * y - 3 * x - 5) * y' x y + 7 * x - 3 * y + 2 = 0) : 
    ∃ C : ℝ, ∀ x y : ℝ, 2 * y^2 - 3 * x * y + (7/2) * x^2 + 2 * x - 5 * y = C :=
by
  sorry

end NUMINAMATH_GPT_integrate_differential_eq_l571_57199


namespace NUMINAMATH_GPT_password_decryption_probability_l571_57118

theorem password_decryption_probability :
  let A := (1:ℚ)/5
  let B := (1:ℚ)/3
  let C := (1:ℚ)/4
  let P_decrypt := 1 - (1 - A) * (1 - B) * (1 - C)
  P_decrypt = 3/5 := 
  by
    -- Calculations and logic will be provided here
    sorry

end NUMINAMATH_GPT_password_decryption_probability_l571_57118


namespace NUMINAMATH_GPT_student_error_difference_l571_57173

theorem student_error_difference (num : ℤ) (num_val : num = 480) : 
  (5 / 6 * num - 5 / 16 * num) = 250 := 
by 
  sorry

end NUMINAMATH_GPT_student_error_difference_l571_57173


namespace NUMINAMATH_GPT_income_increase_is_60_percent_l571_57159

noncomputable def income_percentage_increase 
  (J T M : ℝ) 
  (h1 : T = 0.60 * J) 
  (h2 : M = 0.9599999999999999 * J) : ℝ :=
  (M - T) / T * 100

theorem income_increase_is_60_percent 
  (J T M : ℝ) 
  (h1 : T = 0.60 * J) 
  (h2 : M = 0.9599999999999999 * J) : 
  income_percentage_increase J T M h1 h2 = 60 :=
by
  sorry

end NUMINAMATH_GPT_income_increase_is_60_percent_l571_57159


namespace NUMINAMATH_GPT_domain_of_f_l571_57198

open Set Real

noncomputable def f (x : ℝ) : ℝ := (x + 6) / sqrt (x^2 - 5 * x + 6)

theorem domain_of_f : {x : ℝ | ∃ y, y = f x} = {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l571_57198


namespace NUMINAMATH_GPT_value_of_expression_l571_57148

theorem value_of_expression (y : ℝ) (h : 6 * y^2 + 7 = 2 * y + 10) : (12 * y - 4)^2 = 80 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l571_57148


namespace NUMINAMATH_GPT_students_playing_both_correct_l571_57174

def total_students : ℕ := 36
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def neither_players : ℕ := 7
def students_playing_both : ℕ := 17

theorem students_playing_both_correct :
  total_students - neither_players = (football_players + long_tennis_players) - students_playing_both :=
by 
  sorry

end NUMINAMATH_GPT_students_playing_both_correct_l571_57174


namespace NUMINAMATH_GPT_find_a2_l571_57191

def S (n : Nat) (a1 d : Int) : Int :=
  n * a1 + (n * (n - 1) * d) / 2

theorem find_a2 (a1 : Int) (d : Int) :
  a1 = -2010 ∧
  (S 2010 a1 d) / 2010 - (S 2008 a1 d) / 2008 = 2 →
  a1 + d = -2008 :=
by
  sorry

end NUMINAMATH_GPT_find_a2_l571_57191


namespace NUMINAMATH_GPT_ratio_of_luxury_to_suv_l571_57153

variable (E L S : Nat)

-- Conditions
def condition1 := E * 2 = L * 3
def condition2 := E * 1 = S * 4

-- The statement to prove
theorem ratio_of_luxury_to_suv 
  (h1 : condition1 E L)
  (h2 : condition2 E S) :
  L * 3 = S * 8 :=
by sorry

end NUMINAMATH_GPT_ratio_of_luxury_to_suv_l571_57153


namespace NUMINAMATH_GPT_dividend_calculation_l571_57172

theorem dividend_calculation (divisor quotient remainder : ℕ) (h1 : divisor = 18) (h2 : quotient = 9) (h3 : remainder = 5) : 
  (divisor * quotient + remainder = 167) :=
by
  sorry

end NUMINAMATH_GPT_dividend_calculation_l571_57172


namespace NUMINAMATH_GPT_smaller_number_4582_l571_57143

theorem smaller_number_4582 (a b : ℕ) (ha : 10 ≤ a) (hb : 10 ≤ b) (ha_b : a < 100) (hb_b : b < 100) (h : a * b = 4582) :
  min a b = 21 :=
sorry

end NUMINAMATH_GPT_smaller_number_4582_l571_57143


namespace NUMINAMATH_GPT_parabola_directrix_l571_57180

theorem parabola_directrix (x y : ℝ) : 
  (∀ x: ℝ, y = -4 * x ^ 2 + 4) → (y = 65 / 16) := 
by 
  sorry

end NUMINAMATH_GPT_parabola_directrix_l571_57180


namespace NUMINAMATH_GPT_find_a_l571_57184

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a*x - 1 = 0}

theorem find_a (a : ℝ) (h : B a ⊆ A) : a = 0 ∨ a = -1 ∨ a = (1 / 3) :=
sorry

end NUMINAMATH_GPT_find_a_l571_57184


namespace NUMINAMATH_GPT_sum_of_solutions_l571_57128

theorem sum_of_solutions :
  (∃ S : Finset ℝ, (∀ x ∈ S, x^2 - 8*x + 21 = abs (x - 5) + 4) ∧ S.sum id = 18) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l571_57128


namespace NUMINAMATH_GPT_symmetric_origin_l571_57147

def symmetric_point (p : (Int × Int)) : (Int × Int) :=
  (-p.1, -p.2)

theorem symmetric_origin : symmetric_point (-2, 5) = (2, -5) :=
by
  -- proof goes here
  -- we use sorry to indicate the place where the solution would go
  sorry

end NUMINAMATH_GPT_symmetric_origin_l571_57147


namespace NUMINAMATH_GPT_calculate_mean_score_l571_57158

theorem calculate_mean_score (M SD : ℝ) 
  (h1 : M - 2 * SD = 60)
  (h2 : M + 3 * SD = 100) : 
  M = 76 :=
by
  sorry

end NUMINAMATH_GPT_calculate_mean_score_l571_57158


namespace NUMINAMATH_GPT_range_of_a_l571_57177

noncomputable def f (x a : ℝ) := x^2 + 2 * x - a
noncomputable def g (x : ℝ) := 2 * x + 2 * Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2, (1/e) ≤ x1 ∧ x1 < x2 ∧ x2 ≤ e ∧ f x1 a = g x1 ∧ f x2 a = g x2) ↔ 
  1 < a ∧ a ≤ (1/(e^2)) + 2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l571_57177


namespace NUMINAMATH_GPT_division_of_fractions_l571_57102

theorem division_of_fractions : (1 / 10) / (1 / 5) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_division_of_fractions_l571_57102


namespace NUMINAMATH_GPT_problem1_problem2_l571_57145

theorem problem1 : Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0 :=
by
  sorry

theorem problem2 : Real.sqrt 6 / Real.sqrt 18 * Real.sqrt 27 = 3 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l571_57145


namespace NUMINAMATH_GPT_jims_final_paycheck_l571_57168

noncomputable def final_paycheck (g r t h m b btr : ℝ) := 
  let retirement := g * r
  let gym := m / 2
  let net_before_bonus := g - retirement - t - h - gym
  let after_tax_bonus := b * (1 - btr)
  net_before_bonus + after_tax_bonus

theorem jims_final_paycheck :
  final_paycheck 1120 0.25 100 200 50 500 0.30 = 865 :=
by
  sorry

end NUMINAMATH_GPT_jims_final_paycheck_l571_57168


namespace NUMINAMATH_GPT_difference_highest_lowest_score_l571_57119

-- Declare scores of each player
def Zach_score : ℕ := 42
def Ben_score : ℕ := 21
def Emma_score : ℕ := 35
def Leo_score : ℕ := 28

-- Calculate the highest and lowest scores
def highest_score : ℕ := max (max Zach_score Ben_score) (max Emma_score Leo_score)
def lowest_score : ℕ := min (min Zach_score Ben_score) (min Emma_score Leo_score)

-- Calculate the difference
def score_difference : ℕ := highest_score - lowest_score

theorem difference_highest_lowest_score : score_difference = 21 := 
by
  sorry

end NUMINAMATH_GPT_difference_highest_lowest_score_l571_57119


namespace NUMINAMATH_GPT_arctan_sum_of_roots_l571_57190

theorem arctan_sum_of_roots (u v w : ℝ) (h1 : u + v + w = 0) (h2 : u * v + v * w + w * u = -10) (h3 : u * v * w = -11) :
  Real.arctan u + Real.arctan v + Real.arctan w = π / 4 :=
by
  sorry

end NUMINAMATH_GPT_arctan_sum_of_roots_l571_57190


namespace NUMINAMATH_GPT_find_denominator_l571_57193

theorem find_denominator (y : ℝ) (x : ℝ) (h₀ : y > 0) (h₁ : 9 * y / 20 + 3 * y / x = 0.75 * y) : x = 10 :=
sorry

end NUMINAMATH_GPT_find_denominator_l571_57193


namespace NUMINAMATH_GPT_vector_addition_correct_dot_product_correct_l571_57152

-- Define the two vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

-- Define the expected results
def a_plus_b_expected : ℝ × ℝ := (4, 3)
def a_dot_b_expected : ℝ := 5

-- Prove the sum of vectors a and b
theorem vector_addition_correct : a + b = a_plus_b_expected := by
  sorry

-- Prove the dot product of vectors a and b
theorem dot_product_correct : a.1 * b.1 + a.2 * b.2 = a_dot_b_expected := by
  sorry

end NUMINAMATH_GPT_vector_addition_correct_dot_product_correct_l571_57152


namespace NUMINAMATH_GPT_find_n_l571_57188

theorem find_n (n : ℕ) (h1 : n > 13) (h2 : (12 : ℚ) / (n - 1 : ℚ) = 1 / 3) : n = 37 := by
  sorry

end NUMINAMATH_GPT_find_n_l571_57188


namespace NUMINAMATH_GPT_profit_when_x_is_6_max_profit_l571_57110

noncomputable def design_fee : ℝ := 20000 / 10000
noncomputable def production_cost_per_100 : ℝ := 10000 / 10000

noncomputable def P (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x - 0.8
  else 14.7 - 9 / (x - 3)

noncomputable def cost_of_x_sets (x : ℝ) : ℝ :=
  design_fee + x * production_cost_per_100

noncomputable def profit (x : ℝ) : ℝ :=
  P x - cost_of_x_sets x

theorem profit_when_x_is_6 :
  profit 6 = 3.7 := sorry

theorem max_profit :
  ∀ x : ℝ, profit x ≤ 3.7 := sorry

end NUMINAMATH_GPT_profit_when_x_is_6_max_profit_l571_57110


namespace NUMINAMATH_GPT_irreducible_fraction_unique_l571_57165

theorem irreducible_fraction_unique :
  ∃ (a b : ℕ), a = 5 ∧ b = 2 ∧ gcd a b = 1 ∧ (∃ n : ℕ, 10^n = a * b) :=
by
  sorry

end NUMINAMATH_GPT_irreducible_fraction_unique_l571_57165


namespace NUMINAMATH_GPT_John_next_birthday_age_l571_57163

variable (John Mike Lucas : ℝ)

def John_is_25_percent_older_than_Mike := John = 1.25 * Mike
def Mike_is_30_percent_younger_than_Lucas := Mike = 0.7 * Lucas
def sum_of_ages_is_27_point_3_years := John + Mike + Lucas = 27.3

theorem John_next_birthday_age 
  (h1 : John_is_25_percent_older_than_Mike John Mike) 
  (h2 : Mike_is_30_percent_younger_than_Lucas Mike Lucas) 
  (h3 : sum_of_ages_is_27_point_3_years John Mike Lucas) : 
  John + 1 = 10 := 
sorry

end NUMINAMATH_GPT_John_next_birthday_age_l571_57163


namespace NUMINAMATH_GPT_gcd_divisor_l571_57112

theorem gcd_divisor (p q r s : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) 
  (hpq : Nat.gcd p q = 40) (hqr : Nat.gcd q r = 50) (hrs : Nat.gcd r s = 60) (hsp : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) 
  : 13 ∣ p :=
sorry

end NUMINAMATH_GPT_gcd_divisor_l571_57112


namespace NUMINAMATH_GPT_scientific_notation_l571_57196

theorem scientific_notation (a n : ℝ) (h1 : 100000000 = a * 10^n) (h2 : 1 ≤ a) (h3 : a < 10) : 
  a = 1 ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_l571_57196


namespace NUMINAMATH_GPT_total_apples_eaten_l571_57114

def Apples_Tuesday : ℕ := 4
def Apples_Wednesday : ℕ := 2 * Apples_Tuesday
def Apples_Thursday : ℕ := Apples_Tuesday / 2

theorem total_apples_eaten : Apples_Tuesday + Apples_Wednesday + Apples_Thursday = 14 := by
  sorry

end NUMINAMATH_GPT_total_apples_eaten_l571_57114


namespace NUMINAMATH_GPT_number_of_triangles_in_lattice_l571_57104

-- Define the triangular lattice structure
def triangular_lattice_rows : List ℕ := [1, 2, 3, 4]

-- Define the main theorem to state the number of triangles
theorem number_of_triangles_in_lattice :
  let number_of_triangles := 1 + 2 + 3 + 6 + 10
  number_of_triangles = 22 :=
by
  -- here goes the proof, which we skip with "sorry"
  sorry

end NUMINAMATH_GPT_number_of_triangles_in_lattice_l571_57104


namespace NUMINAMATH_GPT_point_inside_circle_l571_57123

theorem point_inside_circle (a : ℝ) (h : 5 * a^2 - 4 * a - 1 < 0) : -1/5 < a ∧ a < 1 :=
    sorry

end NUMINAMATH_GPT_point_inside_circle_l571_57123


namespace NUMINAMATH_GPT_range_of_a_l571_57139

theorem range_of_a 
  (a : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x < 0, f x = a^x)
  (h2 : ∀ x ≥ 0, f x = (a - 3) * x + 4 * a)
  (h3 : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0) :
  0 < a ∧ a ≤ 1 / 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l571_57139


namespace NUMINAMATH_GPT_proof_problem_l571_57132

def consistent_system (x y : ℕ) : Prop :=
  x + y = 99 ∧ 3 * x + 1 / 3 * y = 97

theorem proof_problem : ∃ (x y : ℕ), consistent_system x y := sorry

end NUMINAMATH_GPT_proof_problem_l571_57132


namespace NUMINAMATH_GPT_range_of_a_l571_57164

theorem range_of_a (a x y : ℝ)
  (h1 : x + 3 * y = 2 + a)
  (h2 : 3 * x + y = -4 * a)
  (hxy : x + y > 2) : a < -2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l571_57164


namespace NUMINAMATH_GPT_least_integer_divisors_l571_57108

theorem least_integer_divisors (n m k : ℕ)
  (h_divisors : 3003 = 3 * 7 * 11 * 13)
  (h_form : n = m * 30 ^ k)
  (h_no_div_30 : ¬(30 ∣ m))
  (h_divisor_count : ∀ (p : ℕ) (h : n = p), (p + 1) * (p + 1) * (p + 1) * (p + 1) = 3003)
  : m + k = 104978 :=
sorry

end NUMINAMATH_GPT_least_integer_divisors_l571_57108


namespace NUMINAMATH_GPT_exists_xy_l571_57156

open Classical

variable (f : ℝ → ℝ)

theorem exists_xy (h : ∃ x₀ y₀ : ℝ, f x₀ ≠ f y₀) : ∃ x y : ℝ, f (x + y) < f (x * y) :=
by
  sorry

end NUMINAMATH_GPT_exists_xy_l571_57156


namespace NUMINAMATH_GPT_cost_price_is_800_l571_57105

theorem cost_price_is_800 (mp sp cp : ℝ) (h1 : mp = 1100) (h2 : sp = 0.8 * mp) (h3 : sp = 1.1 * cp) :
  cp = 800 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_is_800_l571_57105


namespace NUMINAMATH_GPT_amount_made_per_jersey_l571_57154

-- Definitions based on conditions
def total_revenue_from_jerseys : ℕ := 25740
def number_of_jerseys_sold : ℕ := 156

-- Theorem statement
theorem amount_made_per_jersey : 
  total_revenue_from_jerseys / number_of_jerseys_sold = 165 := 
by
  sorry

end NUMINAMATH_GPT_amount_made_per_jersey_l571_57154


namespace NUMINAMATH_GPT_proof_problem_l571_57185

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def condition (f : ℝ → ℝ) : Prop :=
∀ x1 x2, (0 ≤ x1) → (0 ≤ x2) → (x1 ≠ x2) → (x1 - x2) * (f x1 - f x2) > 0

theorem proof_problem (f : ℝ → ℝ) (hf_even : even_function f) (hf_condition : condition f) :
  f 1 < f (-2) ∧ f (-2) < f 3 := sorry

end NUMINAMATH_GPT_proof_problem_l571_57185


namespace NUMINAMATH_GPT_solve_equation_l571_57149

theorem solve_equation :
  ∀ (x : ℝ), (1 / 7 + 7 / x = 15 / x + 1 / 15) → x = 8 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_equation_l571_57149


namespace NUMINAMATH_GPT_fraction_calculation_l571_57138

theorem fraction_calculation : (3/10 : ℚ) + (5/100 : ℚ) - (2/1000 : ℚ) = 348/1000 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_calculation_l571_57138


namespace NUMINAMATH_GPT_line_equation_l571_57166

-- Define the points A and M
structure Point where
  x : ℝ
  y : ℝ

def A := Point.mk 3 1
def M := Point.mk 4 (-3)

def symmetric_point (A M : Point) : Point :=
  Point.mk (2 * M.x - A.x) (2 * M.y - A.y)

def line_through_origin (B : Point) : Prop :=
  7 * B.x + 5 * B.y = 0

theorem line_equation (B : Point) (hB : B = symmetric_point A M) : line_through_origin B :=
  by
  sorry

end NUMINAMATH_GPT_line_equation_l571_57166


namespace NUMINAMATH_GPT_B_elements_l571_57136

def B : Set ℤ := {x | -3 < 2 * x - 1 ∧ 2 * x - 1 < 3}

theorem B_elements : B = {-1, 0, 1} :=
by
  sorry

end NUMINAMATH_GPT_B_elements_l571_57136


namespace NUMINAMATH_GPT_find_a_l571_57197

theorem find_a
  (a b c : ℝ) 
  (h1 : ∀ x : ℝ, x = 1 ∨ x = 2 → a * x * (x + 1) + b * x * (x + 2) + c * (x + 1) * (x + 2) = 0)
  (h2 : a + b + c = 2) : 
  a = 12 := 
sorry

end NUMINAMATH_GPT_find_a_l571_57197


namespace NUMINAMATH_GPT_base9_to_base10_l571_57183

def num_base9 : ℕ := 521 -- Represents 521_9
def base : ℕ := 9

theorem base9_to_base10 : 
  (1 * base^0 + 2 * base^1 + 5 * base^2) = 424 := 
by
  -- Sorry allows us to skip the proof.
  sorry

end NUMINAMATH_GPT_base9_to_base10_l571_57183


namespace NUMINAMATH_GPT_largest_divisor_even_squares_l571_57117

theorem largest_divisor_even_squares (m n : ℕ) (hm : Even m) (hn : Even n) (h : n < m) :
  ∃ k, k = 4 ∧ ∀ a b : ℕ, Even a → Even b → b < a → k ∣ (a^2 - b^2) :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_even_squares_l571_57117


namespace NUMINAMATH_GPT_polygon_number_of_sides_l571_57133

-- Definitions based on conditions
def sum_of_exterior_angles (n : ℕ) : ℕ := 360
def exterior_angle (angle : ℕ) : ℕ := 30

-- The theorem statement
theorem polygon_number_of_sides (n : ℕ) (angle : ℕ) 
  (h1 : sum_of_exterior_angles n = 360)
  (h2 : exterior_angle angle = 30) : 
  n = 12 := 
by
  sorry

end NUMINAMATH_GPT_polygon_number_of_sides_l571_57133


namespace NUMINAMATH_GPT_li_payment_l571_57175

noncomputable def payment_li (daily_payment_per_unit : ℚ) (days_li_worked : ℕ) : ℚ :=
daily_payment_per_unit * days_li_worked

theorem li_payment (work_per_day : ℚ) (days_li_worked : ℕ) (days_extra_work : ℕ) 
  (difference_payment : ℚ) (daily_payment_per_unit : ℚ) (initial_nanual_workdays : ℕ) :
  work_per_day = 1 →
  days_li_worked = 2 →
  days_extra_work = 3 →
  difference_payment = 2700 →
  daily_payment_per_unit = difference_payment / (initial_nanual_workdays + (3 * 3)) → 
  payment_li daily_payment_per_unit days_li_worked = 450 := 
by 
  intros h_work_per_day h_days_li_worked h_days_extra_work h_diff_payment h_daily_payment 
  sorry

end NUMINAMATH_GPT_li_payment_l571_57175


namespace NUMINAMATH_GPT_find_fraction_l571_57103

theorem find_fraction
  (N : ℝ)
  (hN : N = 30)
  (h : 0.5 * N = (x / y) * N + 10):
  x / y = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_l571_57103


namespace NUMINAMATH_GPT_train_length_l571_57150

theorem train_length (L : ℝ) :
  (∀ t₁ t₂ : ℝ, t₁ = t₂ → L = t₁ / 2) →
  (∀ t : ℝ, t = (8 / 3600) * 36 → L * 2 = t) →
  44 - 36 = 8 →
  L = 40 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l571_57150


namespace NUMINAMATH_GPT_boarders_joined_l571_57157

theorem boarders_joined (initial_boarders : ℕ) (initial_day_scholars : ℕ)
  (final_boarders : ℕ) (x : ℕ)
  (ratio_initial : initial_boarders * 16 = initial_day_scholars * 7)
  (ratio_final : final_boarders * 2 = initial_day_scholars)
  (final_boarders_eq : final_boarders = initial_boarders + x)
  (initial_boarders_val : initial_boarders = 560)
  (initial_day_scholars_val : initial_day_scholars = 1280)
  (final_boarders_val : final_boarders = 640) :
  x = 80 :=
by
  sorry

end NUMINAMATH_GPT_boarders_joined_l571_57157


namespace NUMINAMATH_GPT_fraction_picked_l571_57181

/--
An apple tree has three times as many apples as the number of plums on a plum tree.
Damien picks a certain fraction of the fruits from the trees, and there are 96 plums
and apples remaining on the tree. There were 180 apples on the apple tree before 
Damien picked any of the fruits. Prove that Damien picked 3/5 of the fruits from the trees.
-/
theorem fraction_picked (P F : ℝ) (h1 : 3 * P = 180) (h2 : (1 - F) * (180 + P) = 96) :
  F = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_picked_l571_57181


namespace NUMINAMATH_GPT_find_a_l571_57182

theorem find_a :
  ∃ (a : ℤ), (∀ (x y : ℤ),
    (∃ (m n : ℤ), (x - 8 + m * y) * (x + 3 + n * y) = x^2 + 7 * x * y + a * y^2 - 5 * x - 45 * y - 24) ↔ a = 6) := 
sorry

end NUMINAMATH_GPT_find_a_l571_57182


namespace NUMINAMATH_GPT_quadrilateral_area_l571_57113

noncomputable def area_of_quadrilateral (a : ℝ) : ℝ :=
  let sqrt3 := Real.sqrt 3
  let num := a^2 * (9 - 5 * sqrt3)
  let denom := 12
  num / denom

theorem quadrilateral_area (a : ℝ) : area_of_quadrilateral a = (a^2 * (9 - 5 * Real.sqrt 3)) / 12 := by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l571_57113


namespace NUMINAMATH_GPT_forgotten_angle_l571_57176

theorem forgotten_angle {n : ℕ} (h₁ : 2070 = (n - 2) * 180 - angle) : angle = 90 :=
by
  sorry

end NUMINAMATH_GPT_forgotten_angle_l571_57176


namespace NUMINAMATH_GPT_find_specified_time_l571_57194

theorem find_specified_time (distance : ℕ) (slow_time fast_time : ℕ → ℕ) (fast_is_double : ∀ x, fast_time x = 2 * slow_time x)
  (distance_value : distance = 900) (slow_time_eq : ∀ x, slow_time x = x + 1) (fast_time_eq : ∀ x, fast_time x = x - 3) :
  2 * (distance / (slow_time x)) = distance / (fast_time x) :=
by
  intros
  rw [distance_value, slow_time_eq, fast_time_eq]
  sorry

end NUMINAMATH_GPT_find_specified_time_l571_57194
