import Mathlib

namespace find_b_l964_96405

open Real

theorem find_b (b : ℝ) : 
  (∀ x y : ℝ, 4 * y - 3 * x - 2 = 0 -> 6 * y + b * x + 1 = 0 -> 
   exists m₁ m₂ : ℝ, 
   ((y = m₁ * x + _1 / 2) -> m₁ = 3 / 4) ∧ ((y = m₂ * x - 1 / 6) -> m₂ = -b / 6)) -> 
  b = -4.5 :=
by
  sorry

end find_b_l964_96405


namespace exists_triangle_l964_96469

variable (k α m_a : ℝ)

-- Define the main constructibility condition as a noncomputable function.
noncomputable def triangle_constructible (k α m_a : ℝ) : Prop :=
  m_a ≤ (k / 2) * ((1 - Real.sin (α / 2)) / Real.cos (α / 2))

-- Main theorem statement to prove the existence of the triangle
theorem exists_triangle :
  ∃ (k α m_a : ℝ), triangle_constructible k α m_a := 
sorry

end exists_triangle_l964_96469


namespace average_growth_rate_le_half_sum_l964_96447

variable (p q x : ℝ)

theorem average_growth_rate_le_half_sum : 
  (1 + p) * (1 + q) = (1 + x) ^ 2 → x ≤ (p + q) / 2 :=
by
  intro h
  sorry

end average_growth_rate_le_half_sum_l964_96447


namespace tomas_first_month_distance_l964_96442

theorem tomas_first_month_distance 
  (distance_n_5 : ℝ := 26.3)
  (double_distance_each_month : ∀ (n : ℕ), n ≥ 1 → (distance_n : ℝ) = distance_n_5 / (2 ^ (5 - n)))
  : distance_n_5 / (2 ^ (5 - 1)) = 1.64375 :=
by
  sorry

end tomas_first_month_distance_l964_96442


namespace evaluate_expression_l964_96482

variable (a b : ℝ) (h : a > b ∧ b > 0)

theorem evaluate_expression (h : a > b ∧ b > 0) : 
  (a^2 * b^3) / (b^2 * a^3) = (a / b)^(2 - 3) :=
  sorry

end evaluate_expression_l964_96482


namespace frobenius_two_vars_l964_96489

theorem frobenius_two_vars (a b n : ℤ) (ha : 0 < a) (hb : 0 < b) (hgcd : Int.gcd a b = 1) (hn : n > a * b - a - b) :
  ∃ x y : ℕ, n = a * x + b * y :=
by
  sorry

end frobenius_two_vars_l964_96489


namespace find_f_l964_96449

-- Define the function f and its conditions
def f (x : ℝ) : ℝ := sorry

axiom f_0 : f 0 = 0
axiom f_xy (x y : ℝ) : f (x * y) = f ((x^2 + y^2) / 2) + 3 * (x - y)^2

-- Theorem to be proved
theorem find_f (x : ℝ) : f x = -6 * x + 3 :=
by sorry -- proof goes here

end find_f_l964_96449


namespace integer_solutions_l964_96488

-- Define the equation to be solved
def equation (x y : ℤ) : Prop := x * y + 3 * x - 5 * y + 3 = 0

-- Define the solutions
def solution_set : List (ℤ × ℤ) := 
  [(-13,-2), (-4,-1), (-1,0), (2, 3), (3, 6), (4, 15), (6, -21),
   (7, -12), (8, -9), (11, -6), (14, -5), (23, -4)]

-- The theorem stating the solutions are correct
theorem integer_solutions : ∀ (x y : ℤ), (x, y) ∈ solution_set → equation x y :=
by
  sorry

end integer_solutions_l964_96488


namespace num_socks_in_machine_l964_96439

-- Definition of the number of people who played the match
def num_players : ℕ := 11

-- Definition of the number of socks per player
def socks_per_player : ℕ := 2

-- The goal is to prove that the total number of socks in the washing machine is 22
theorem num_socks_in_machine : num_players * socks_per_player = 22 :=
by
  sorry

end num_socks_in_machine_l964_96439


namespace sum_bn_l964_96467

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n : ℕ, a n = a 0 + n * d

def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n-1))) / 2

def geometric_sequence (a : ℕ → ℕ) :=
  ∃ r : ℕ, a 1 = a 0 * r ∧ a 2 = a 1 * r

-- Given S_5 = 35
def S5_property (S : ℕ → ℕ) := S 5 = 35

-- a_1, a_4, a_{13} is a geometric sequence
def a1_a4_a13_geometric_sequence (a : ℕ → ℕ) :=
  ∃ r : ℕ, a 4 = a 1 * r ∧ a 13 = a 4 * r

-- Define the sequence b_n and conditions
def bn_prop (a b : ℕ → ℕ) := ∀ n : ℕ, b n = a n * (2^(n-1))

-- Main theorem
theorem sum_bn {a b : ℕ → ℕ} {S T : ℕ → ℕ} (h_a : arithmetic_sequence a 2) (h_S5 : S5_property S) (h_geo : a1_a4_a13_geometric_sequence a) (h_bn : bn_prop a b)
  : ∀ n : ℕ, T n = 1 + (2 * n - 1) * 2^n := sorry

end sum_bn_l964_96467


namespace circle_range_of_m_l964_96422

theorem circle_range_of_m (m : ℝ) :
  (∃ h k r : ℝ, (∀ x y : ℝ, (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2 ↔ x ^ 2 + y ^ 2 - x + y + m = 0)) ↔ (m < 1/2) :=
by
  sorry

end circle_range_of_m_l964_96422


namespace Janet_initial_crayons_l964_96404

variable (Michelle_initial Janet_initial Michelle_final : ℕ)

theorem Janet_initial_crayons (h1 : Michelle_initial = 2) (h2 : Michelle_final = 4) (h3 : Michelle_final = Michelle_initial + Janet_initial) :
  Janet_initial = 2 :=
by
  sorry

end Janet_initial_crayons_l964_96404


namespace find_higher_interest_rate_l964_96450

-- Definitions and conditions based on the problem
def total_investment : ℕ := 4725
def higher_rate_investment : ℕ := 1925
def lower_rate_investment : ℕ := total_investment - higher_rate_investment
def lower_rate : ℝ := 0.08
def higher_to_lower_interest_ratio : ℝ := 2

-- The main theorem to prove the higher interest rate
theorem find_higher_interest_rate (r : ℝ) (h1 : higher_rate_investment = 1925) (h2 : lower_rate_investment = 2800) :
  1925 * r = 2 * (2800 * 0.08) → r = 448 / 1925 :=
sorry

end find_higher_interest_rate_l964_96450


namespace find_second_number_l964_96462

theorem find_second_number (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 70 + 13) / 3 + 9 → x = 40 :=
by
  sorry

end find_second_number_l964_96462


namespace chris_money_left_l964_96437

def video_game_cost : ℕ := 60
def candy_cost : ℕ := 5
def babysitting_rate : ℕ := 8
def hours_worked : ℕ := 9
def earnings : ℕ := babysitting_rate * hours_worked
def total_cost : ℕ := video_game_cost + candy_cost
def money_left : ℕ := earnings - total_cost

theorem chris_money_left
  (h1 : video_game_cost = 60)
  (h2 : candy_cost = 5)
  (h3 : babysitting_rate = 8)
  (h4 : hours_worked = 9) :
  money_left = 7 :=
by
  -- The detailed proof is omitted.
  sorry

end chris_money_left_l964_96437


namespace find_smallest_n_l964_96476

/-- 
Define the doubling sum function D(a, n)
-/
def doubling_sum (a : ℕ) (n : ℕ) : ℕ := a * (2^n - 1)

/--
Main theorem statement that proves the smallest n for the given conditions
-/
theorem find_smallest_n :
  ∃ (n : ℕ), (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 6 → ∃ (ai : ℕ), doubling_sum ai i = n) ∧ n = 9765 := 
sorry

end find_smallest_n_l964_96476


namespace merchant_profit_percentage_l964_96451

noncomputable def cost_price_of_one_article (C : ℝ) : Prop := ∃ S : ℝ, 20 * C = 16 * S

theorem merchant_profit_percentage (C S : ℝ) (h : cost_price_of_one_article C) : 
  100 * ((S - C) / C) = 25 :=
by 
  sorry

end merchant_profit_percentage_l964_96451


namespace john_reaching_floor_pushups_l964_96406

-- Definitions based on conditions
def john_train_days_per_week : ℕ := 5
def reps_to_progress : ℕ := 20
def variations : ℕ := 3  -- wall, incline, knee

-- Mathematical statement
theorem john_reaching_floor_pushups : 
  (reps_to_progress * variations) / john_train_days_per_week = 12 := 
by
  sorry

end john_reaching_floor_pushups_l964_96406


namespace imaginary_number_condition_fourth_quadrant_condition_l964_96492

-- Part 1: Prove that if \( z \) is purely imaginary, then \( m = 0 \)
theorem imaginary_number_condition (m : ℝ) :
  (m * (m + 2) = 0) ∧ (m^2 + m - 2 ≠ 0) → m = 0 :=
by
  sorry

-- Part 2: Prove that if \( z \) is in the fourth quadrant, then \( 0 < m < 1 \)
theorem fourth_quadrant_condition (m : ℝ) :
  (m * (m + 2) > 0) ∧ (m^2 + m - 2 < 0) → (0 < m ∧ m < 1) :=
by
  sorry

end imaginary_number_condition_fourth_quadrant_condition_l964_96492


namespace peter_total_books_is_20_l964_96430

noncomputable def total_books_peter_has (B : ℝ) : Prop :=
  let Peter_Books_Read := 0.40 * B
  let Brother_Books_Read := 0.10 * B
  Peter_Books_Read = Brother_Books_Read + 6

theorem peter_total_books_is_20 :
  ∃ B : ℝ, total_books_peter_has B ∧ B = 20 := 
by
  sorry

end peter_total_books_is_20_l964_96430


namespace two_pow_n_minus_one_div_by_seven_iff_two_pow_n_plus_one_not_div_by_seven_l964_96420

theorem two_pow_n_minus_one_div_by_seven_iff (n : ℕ) : (7 ∣ 2^n - 1) ↔ ∃ k : ℕ, n = 3 * k :=
by sorry

theorem two_pow_n_plus_one_not_div_by_seven (n : ℕ) : n > 0 → ¬(7 ∣ 2^n + 1) :=
by sorry

end two_pow_n_minus_one_div_by_seven_iff_two_pow_n_plus_one_not_div_by_seven_l964_96420


namespace equal_number_of_boys_and_girls_l964_96424

theorem equal_number_of_boys_and_girls
    (num_classrooms : ℕ) (girls : ℕ) (total_per_classroom : ℕ)
    (equal_boys_and_girls : ∀ (c : ℕ), c ≤ num_classrooms → (girls + boys) = total_per_classroom):
    num_classrooms = 4 → girls = 44 → total_per_classroom = 25 → boys = 44 :=
by
  sorry

end equal_number_of_boys_and_girls_l964_96424


namespace factorize_expression_l964_96410

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l964_96410


namespace circle_passing_given_points_l964_96428

theorem circle_passing_given_points :
  ∃ (D E F : ℚ), (F = 0) ∧ (E = - (9 / 5)) ∧ (D = 19 / 5) ∧
  (∀ (x y : ℚ), x^2 + y^2 + D * x + E * y + F = 0 ↔ (x = 0 ∧ y = 0) ∨ (x = -2 ∧ y = 3) ∨ (x = -4 ∧ y = 1)) :=
by
  sorry

end circle_passing_given_points_l964_96428


namespace sum_of_faces_l964_96441

theorem sum_of_faces (n_side_faces_per_prism : ℕ) (n_non_side_faces_per_prism : ℕ)
  (num_prisms : ℕ) (h1 : n_side_faces_per_prism = 3) (h2 : n_non_side_faces_per_prism = 2) 
  (h3 : num_prisms = 3) : 
  n_side_faces_per_prism * num_prisms + n_non_side_faces_per_prism * num_prisms = 15 :=
by
  sorry

end sum_of_faces_l964_96441


namespace bruces_son_age_l964_96407

variable (Bruce_age : ℕ) (son_age : ℕ)
variable (h1 : Bruce_age = 36)
variable (h2 : Bruce_age + 6 = 3 * (son_age + 6))

theorem bruces_son_age :
  son_age = 8 :=
by {
  sorry
}

end bruces_son_age_l964_96407


namespace different_kinds_of_hamburgers_l964_96425

theorem different_kinds_of_hamburgers 
  (n_condiments : ℕ) 
  (condiment_choices : ℕ)
  (meat_patty_choices : ℕ)
  (h1 : n_condiments = 8)
  (h2 : condiment_choices = 2 ^ n_condiments)
  (h3 : meat_patty_choices = 3)
  : condiment_choices * meat_patty_choices = 768 := 
by
  sorry

end different_kinds_of_hamburgers_l964_96425


namespace complement_of_angle_correct_l964_96403

noncomputable def complement_of_angle (α : ℝ) : ℝ := 90 - α

theorem complement_of_angle_correct (α : ℝ) (h : complement_of_angle α = 125 + 12 / 60) :
  complement_of_angle α = 35 + 12 / 60 :=
by
  sorry

end complement_of_angle_correct_l964_96403


namespace parallel_lines_b_value_l964_96423

-- Define the first line equation in slope-intercept form.
def line1_slope (b : ℝ) : ℝ :=
  3

-- Define the second line equation in slope-intercept form.
def line2_slope (b : ℝ) : ℝ :=
  b + 10

-- Theorem stating that if the lines are parallel, the value of b is -7.
theorem parallel_lines_b_value :
  ∀ b : ℝ, line1_slope b = line2_slope b → b = -7 :=
by
  intro b
  intro h
  sorry

end parallel_lines_b_value_l964_96423


namespace area_of_sector_radius_2_angle_90_l964_96454

-- Given conditions
def radius := 2
def central_angle := 90

-- Required proof: the area of the sector with given conditions equals π.
theorem area_of_sector_radius_2_angle_90 : (90 * Real.pi * (2^2) / 360) = Real.pi := 
by
  sorry

end area_of_sector_radius_2_angle_90_l964_96454


namespace main_problem_l964_96480

def arithmetic_sequence (a : ℕ → ℕ) : Prop := ∃ a₁ d, ∀ n, a (n + 1) = a₁ + n * d

def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop := ∀ n, S n = (n * (a 1 + a n)) / 2

def another_sequence (b : ℕ → ℕ) (a : ℕ → ℕ) : Prop := ∀ n, b n = 1 / (a n * a (n + 1))

theorem main_problem (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) 
  (h1 : a_3 = 5) 
  (h2 : S_3 = 9) 
  (h3 : arithmetic_sequence a)
  (h4 : sequence_sum a S)
  (h5 : another_sequence b a) : 
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, T n = n / (2 * n + 1)) := sorry

end main_problem_l964_96480


namespace tv_cost_l964_96474

theorem tv_cost (savings original_savings furniture_spent : ℝ) (hs : original_savings = 1000) (hf : furniture_spent = (3/4) * original_savings) (remaining_spent : savings = original_savings - furniture_spent) : savings = 250 := 
by
  sorry

end tv_cost_l964_96474


namespace simplify_expression_l964_96408

theorem simplify_expression :
  let a := 7
  let b := 2
  (a^5 + b^8) * (b^3 - (-b)^3)^7 = 0 := by
  let a := 7
  let b := 2
  sorry

end simplify_expression_l964_96408


namespace joan_games_l964_96417

theorem joan_games (last_year_games this_year_games total_games : ℕ)
  (h1 : last_year_games = 9)
  (h2 : total_games = 13)
  : this_year_games = total_games - last_year_games → this_year_games = 4 := 
by
  intros h
  rw [h1, h2] at h
  exact h

end joan_games_l964_96417


namespace solve_quadratic_solve_cubic_l964_96496

theorem solve_quadratic (x : ℝ) (h : 2 * x^2 - 32 = 0) : x = 4 ∨ x = -4 := 
by sorry

theorem solve_cubic (x : ℝ) (h : (x + 4)^3 + 64 = 0) : x = -8 := 
by sorry

end solve_quadratic_solve_cubic_l964_96496


namespace red_balls_unchanged_l964_96468

-- Definitions: 
def initial_red_balls : ℕ := 3
def initial_blue_balls : ℕ := 2
def initial_yellow_balls : ℕ := 5

def remove_blue_ball (blue_balls : ℕ) : ℕ :=
  if blue_balls > 0 then blue_balls - 1 else blue_balls

-- Condition after one blue ball is removed
def blue_balls_after_removal := remove_blue_ball initial_blue_balls

-- Prove that the number of red balls remain unchanged
theorem red_balls_unchanged : initial_red_balls = 3 :=
by
  sorry

end red_balls_unchanged_l964_96468


namespace age_problem_l964_96481

theorem age_problem (a b c d : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : b = 3 * d)
  (h4 : a + b + c + d = 87) : 
  b = 30 :=
by sorry

end age_problem_l964_96481


namespace haley_candy_l964_96471

theorem haley_candy (X : ℕ) (h : X - 17 + 19 = 35) : X = 33 :=
by
  sorry

end haley_candy_l964_96471


namespace prob_xi_ge_2_eq_one_third_l964_96426

noncomputable def pmf (c k : ℝ) : ℝ := c / (k * (k + 1))

theorem prob_xi_ge_2_eq_one_third 
  (c : ℝ) 
  (h₁ : pmf c 1 + pmf c 2 + pmf c 3 = 1) :
  pmf c 2 + pmf c 3 = 1 / 3 :=
by
  sorry

end prob_xi_ge_2_eq_one_third_l964_96426


namespace telescope_visual_range_increase_l964_96401

theorem telescope_visual_range_increase (original_range : ℝ) (increase_percent : ℝ) 
(h1 : original_range = 100) (h2 : increase_percent = 0.50) : 
original_range + (increase_percent * original_range) = 150 := 
sorry

end telescope_visual_range_increase_l964_96401


namespace trajectory_of_midpoint_l964_96486

open Real

theorem trajectory_of_midpoint (A : ℝ × ℝ) (P : ℝ × ℝ) (M : ℝ × ℝ)
    (hA : A = (-2, 0))
    (hP_on_curve : P.1 = 2 * P.2 ^ 2)
    (hM_midpoint : M = ((A.1 + P.1) / 2, (A.2 + P.2) / 2)) :
    M.1 = 4 * M.2 ^ 2 - 1 :=
sorry

end trajectory_of_midpoint_l964_96486


namespace divides_b_n_minus_n_l964_96414

theorem divides_b_n_minus_n (a b : ℕ) (h_a : a > 0) (h_b : b > 0) :
  ∃ n : ℕ, n > 0 ∧ a ∣ (b^n - n) :=
by
  sorry

end divides_b_n_minus_n_l964_96414


namespace ratio_of_sum_to_first_term_l964_96459

-- Definitions and conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 0 * (1 - (2 ^ n)) / (1 - 2)

-- Main statement to be proven
theorem ratio_of_sum_to_first_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geo : geometric_sequence a 2) (h_sum : sum_of_first_n_terms a S) :
  S 3 / a 0 = 7 :=
sorry

end ratio_of_sum_to_first_term_l964_96459


namespace geom_seq_prop_l964_96499

-- Definitions from the conditions
def geom_seq (a : ℕ → ℝ) := ∀ (n : ℕ), (a (n + 1)) / (a n) = (a 1) / (a 0) ∧ a n > 0

def condition (a : ℕ → ℝ) :=
  (1 / (a 2 * a 4)) + (2 / (a 4 ^ 2)) + (1 / (a 4 * a 6)) = 81

-- The statement to prove
theorem geom_seq_prop (a : ℕ → ℝ) (hgeom : geom_seq a) (hcond : condition a) :
  (1 / (a 3) + 1 / (a 5)) = 9 :=
sorry

end geom_seq_prop_l964_96499


namespace base_r_representation_26_eq_32_l964_96438

theorem base_r_representation_26_eq_32 (r : ℕ) : 
  26 = 3 * r + 6 → r = 8 :=
by
  sorry

end base_r_representation_26_eq_32_l964_96438


namespace expression_value_l964_96483

theorem expression_value (a b : ℕ) (ha : a = 45) (hb : b = 15) : (a + b)^2 - (a^2 + b^2) = 1350 := by 
  sorry

end expression_value_l964_96483


namespace tan_A_eq_11_l964_96453

variable (A B C : ℝ)

theorem tan_A_eq_11
  (h1 : Real.sin A = 10 * Real.sin B * Real.sin C)
  (h2 : Real.cos A = 10 * Real.cos B * Real.cos C) :
  Real.tan A = 11 := 
sorry

end tan_A_eq_11_l964_96453


namespace symmetric_point_l964_96421

-- Definitions
def P : ℝ × ℝ := (5, -2)
def line (x y : ℝ) : Prop := x - y + 5 = 0

-- Statement 
theorem symmetric_point (a b : ℝ) 
  (symmetric_condition1 : ∀ x y, line x y → (b + 2)/(a - 5) * 1 = -1)
  (symmetric_condition2 : ∀ x y, line x y → (a + 5)/2 - (b - 2)/2 + 5 = 0) :
  (a, b) = (-7, 10) :=
sorry

end symmetric_point_l964_96421


namespace am_gm_inequality_for_x_l964_96490

theorem am_gm_inequality_for_x (x : ℝ) : 1 + x^2 + x^6 + x^8 ≥ 4 * x^4 := by 
  sorry

end am_gm_inequality_for_x_l964_96490


namespace min_value_abc_l964_96443

open Real

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : 1/a + 1/b + 1/c = 9) :
  a^4 * b^3 * c^2 ≥ 1/10368 :=
sorry

end min_value_abc_l964_96443


namespace distance_between_Jay_and_Sarah_l964_96458

theorem distance_between_Jay_and_Sarah 
  (time_in_hours : ℝ)
  (jay_speed_per_12_minutes : ℝ)
  (sarah_speed_per_36_minutes : ℝ)
  (total_distance : ℝ) :
  time_in_hours = 2 →
  jay_speed_per_12_minutes = 1 →
  sarah_speed_per_36_minutes = 3 →
  total_distance = 20 :=
by
  intros time_in_hours_eq jay_speed_eq sarah_speed_eq
  sorry

end distance_between_Jay_and_Sarah_l964_96458


namespace B_knit_time_l964_96444

theorem B_knit_time (x : ℕ) (hA : 3 > 0) (h_combined_rate : 1/3 + 1/x = 1/2) : x = 6 := sorry

end B_knit_time_l964_96444


namespace ratio_xy_l964_96433

theorem ratio_xy (x y : ℝ) (h : 2*y - 5*x = 0) : x / y = 2 / 5 :=
by sorry

end ratio_xy_l964_96433


namespace remainder_of_n_l964_96436

theorem remainder_of_n (n : ℕ) (h1 : n^2 ≡ 9 [MOD 11]) (h2 : n^3 ≡ 5 [MOD 11]) : n ≡ 3 [MOD 11] :=
sorry

end remainder_of_n_l964_96436


namespace percent_of_z_equals_120_percent_of_y_l964_96466

variable {x y z : ℝ}
variable {p : ℝ}

theorem percent_of_z_equals_120_percent_of_y
  (h1 : (p / 100) * z = 1.2 * y)
  (h2 : y = 0.75 * x)
  (h3 : z = 2 * x) :
  p = 45 :=
by sorry

end percent_of_z_equals_120_percent_of_y_l964_96466


namespace polygon_perimeter_is_35_l964_96418

-- Define the concept of a regular polygon with given side length and exterior angle
def regular_polygon_perimeter (n : ℕ) (side_length : ℕ) : ℕ := 
  n * side_length

theorem polygon_perimeter_is_35 (side_length : ℕ) (exterior_angle : ℕ) (n : ℕ)
  (h1 : side_length = 7) (h2 : exterior_angle = 72) (h3 : 360 / exterior_angle = n) :
  regular_polygon_perimeter n side_length = 35 :=
by
  -- We skip the proof body as only the statement is required
  sorry

end polygon_perimeter_is_35_l964_96418


namespace gerald_jail_time_l964_96452

theorem gerald_jail_time
    (assault_sentence : ℕ := 3) 
    (poisoning_sentence_years : ℕ := 2) 
    (third_offense_extension : ℕ := 1 / 3) 
    (months_in_year : ℕ := 12)
    : (assault_sentence + poisoning_sentence_years * months_in_year) * (1 + third_offense_extension) = 36 :=
by
  sorry

end gerald_jail_time_l964_96452


namespace arithmetic_to_geometric_progression_l964_96446

theorem arithmetic_to_geometric_progression (d : ℝ) (h : ∀ d, (4 + d) * (4 + d) = 7 * (22 + 2 * d)) :
  ∃ d, 7 + 2 * d = 3.752 :=
sorry

end arithmetic_to_geometric_progression_l964_96446


namespace age_ratio_proof_l964_96493

variable (j a x : ℕ)

/-- Given conditions about Jack and Alex's ages. -/
axiom h1 : j - 3 = 2 * (a - 3)
axiom h2 : j - 5 = 3 * (a - 5)

def age_ratio_in_years : Prop :=
  (3 * (a + x) = 2 * (j + x)) → (x = 1)

theorem age_ratio_proof : age_ratio_in_years j a x := by
  sorry

end age_ratio_proof_l964_96493


namespace ending_point_divisible_by_9_l964_96460

theorem ending_point_divisible_by_9 (n : ℕ) (ending_point : ℕ) 
  (h1 : n = 11110) 
  (h2 : ∃ k : ℕ, 10 + 9 * k = ending_point) : 
  ending_point = 99999 := 
  sorry

end ending_point_divisible_by_9_l964_96460


namespace equation_of_tangent_line_l964_96475

theorem equation_of_tangent_line (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 4 * x + a * y - 17 = 0) →
   (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ 4 * x - 3 * y + 11 = 0) :=
sorry

end equation_of_tangent_line_l964_96475


namespace merchant_marked_price_percent_l964_96445

theorem merchant_marked_price_percent (L : ℝ) (hL : L = 100) (purchase_price : ℝ) (h1 : purchase_price = L * 0.70) (x : ℝ)
  (selling_price : ℝ) (h2 : selling_price = x * 0.75) :
  (selling_price - purchase_price) / selling_price = 0.30 → x = 133.33 :=
by
  sorry

end merchant_marked_price_percent_l964_96445


namespace cos_neg_1500_eq_half_l964_96478

theorem cos_neg_1500_eq_half : Real.cos (-1500 * Real.pi / 180) = 1/2 := by
  sorry

end cos_neg_1500_eq_half_l964_96478


namespace prove_inequality_l964_96432

noncomputable def problem_statement (p q r : ℝ) (n : ℕ) (h_pqr : p * q * r = 1) : Prop :=
  (1 / (p^n + q^n + 1)) + (1 / (q^n + r^n + 1)) + (1 / (r^n + p^n + 1)) ≤ 1

theorem prove_inequality (p q r : ℝ) (n : ℕ) (h_pqr : p * q * r = 1) (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r) : 
  problem_statement p q r n h_pqr :=
by
  sorry

end prove_inequality_l964_96432


namespace triangle_BFD_ratio_l964_96484

theorem triangle_BFD_ratio (x : ℝ) : 
  let AF := 3 * x
  let FE := x
  let ED := x
  let DC := 3 * x
  let side_square := AF + FE
  let area_square := side_square^2
  let area_triangle_BFD := area_square - (1/2 * AF * side_square + 1/2 * side_square * FE + 1/2 * ED * DC)
  (area_triangle_BFD / area_square) = 7 / 16 := 
by
  sorry

end triangle_BFD_ratio_l964_96484


namespace percentage_food_given_out_l964_96435

theorem percentage_food_given_out 
  (first_week_donations : ℕ)
  (second_week_donations : ℕ)
  (total_amount_donated : ℕ)
  (remaining_food : ℕ)
  (amount_given_out : ℕ)
  (percentage_given_out : ℕ) : 
  (first_week_donations = 40) →
  (second_week_donations = 2 * first_week_donations) →
  (total_amount_donated = first_week_donations + second_week_donations) →
  (remaining_food = 36) →
  (amount_given_out = total_amount_donated - remaining_food) →
  (percentage_given_out = (amount_given_out * 100) / total_amount_donated) →
  percentage_given_out = 70 :=
by sorry

end percentage_food_given_out_l964_96435


namespace interior_angle_ratio_l964_96497

variables (α β γ : ℝ)

theorem interior_angle_ratio
  (h1 : 2 * α + 3 * β = 4 * γ)
  (h2 : α = 4 * β - γ) :
  ∃ k : ℝ, k ≠ 0 ∧ 
  (α = 2 * k ∧ β = 9 * k ∧ γ = 4 * k) :=
sorry

end interior_angle_ratio_l964_96497


namespace small_frog_reaches_7th_rung_medium_frog_cannot_reach_1st_rung_large_frog_reaches_3rd_rung_l964_96456

-- 1. Prove that the small frog can reach the 7th rung
theorem small_frog_reaches_7th_rung : ∃ (a b : ℕ), 2 * a + 3 * b = 7 :=
by sorry

-- 2. Prove that the medium frog cannot reach the 1st rung
theorem medium_frog_cannot_reach_1st_rung : ¬(∃ (a b : ℕ), 2 * a + 4 * b = 1) :=
by sorry

-- 3. Prove that the large frog can reach the 3rd rung
theorem large_frog_reaches_3rd_rung : ∃ (a b : ℕ), 6 * a + 9 * b = 3 :=
by sorry

end small_frog_reaches_7th_rung_medium_frog_cannot_reach_1st_rung_large_frog_reaches_3rd_rung_l964_96456


namespace constant_term_eq_160_l964_96491

-- Define the binomial coefficients and the binomial theorem
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the general term of (2x + 1/x)^6 expansion
def general_term_expansion (r : ℕ) : ℤ :=
  2^(6 - r) * binom 6 r

-- Define the proof statement for the required constant term
theorem constant_term_eq_160 : general_term_expansion 3 = 160 := 
by
  sorry

end constant_term_eq_160_l964_96491


namespace angles_on_x_axis_eq_l964_96402

open Set

def S1 : Set ℝ := { β | ∃ k : ℤ, β = k * 360 }
def S2 : Set ℝ := { β | ∃ k : ℤ, β = 180 + k * 360 }
def S_total : Set ℝ := S1 ∪ S2
def S_target : Set ℝ := { β | ∃ n : ℤ, β = n * 180 }

theorem angles_on_x_axis_eq : S_total = S_target := 
by 
  sorry

end angles_on_x_axis_eq_l964_96402


namespace vegetables_sold_mass_correct_l964_96429

-- Definitions based on the problem's conditions
def mass_carrots : ℕ := 15
def mass_zucchini : ℕ := 13
def mass_broccoli : ℕ := 8
def total_mass_vegetables := mass_carrots + mass_zucchini + mass_broccoli
def mass_of_vegetables_sold := total_mass_vegetables / 2

-- Theorem to be proved
theorem vegetables_sold_mass_correct : mass_of_vegetables_sold = 18 := by 
  sorry

end vegetables_sold_mass_correct_l964_96429


namespace cost_to_fill_pool_l964_96494

/-- Definition of the pool dimensions and constants --/
def pool_length := 20
def pool_width := 6
def pool_depth := 10
def cubic_feet_to_liters := 25
def liter_cost := 3

/-- Calculating the cost to fill the pool --/
def pool_volume := pool_length * pool_width * pool_depth
def total_liters := pool_volume * cubic_feet_to_liters
def total_cost := total_liters * liter_cost

/-- Theorem stating that the total cost to fill the pool is $90,000 --/
theorem cost_to_fill_pool : total_cost = 90000 := by
  sorry

end cost_to_fill_pool_l964_96494


namespace relationship_of_rationals_l964_96464

theorem relationship_of_rationals (a b c : ℚ) (h1 : a - b > 0) (h2 : b - c > 0) : c < b ∧ b < a :=
by {
  sorry
}

end relationship_of_rationals_l964_96464


namespace smallest_number_divisible_remainders_l964_96416

theorem smallest_number_divisible_remainders :
  ∃ n : ℕ,
    (n % 10 = 9) ∧
    (n % 9 = 8) ∧
    (n % 8 = 7) ∧
    (n % 7 = 6) ∧
    (n % 6 = 5) ∧
    (n % 5 = 4) ∧
    (n % 4 = 3) ∧
    (n % 3 = 2) ∧
    (n % 2 = 1) ∧
    n = 2519 :=
sorry

end smallest_number_divisible_remainders_l964_96416


namespace p_q_work_l964_96470

theorem p_q_work (p_rate q_rate : ℝ) (h1: 1 / p_rate + 1 / q_rate = 1 / 6) (h2: p_rate = 15) : q_rate = 10 :=
by
  sorry

end p_q_work_l964_96470


namespace intersection_M_N_l964_96473

def M : Set ℝ := {x | x < 1/2}
def N : Set ℝ := {y | y ≥ -4}

theorem intersection_M_N :
  (M ∩ N = {x | -4 ≤ x ∧ x < 1/2}) :=
sorry

end intersection_M_N_l964_96473


namespace melissa_games_played_l964_96472

theorem melissa_games_played (total_points : ℕ) (points_per_game : ℕ) (num_games : ℕ) 
  (h1 : total_points = 81) 
  (h2 : points_per_game = 27) 
  (h3 : num_games = total_points / points_per_game) : 
  num_games = 3 :=
by
  -- Proof goes here
  sorry

end melissa_games_played_l964_96472


namespace burn_5_sticks_per_hour_l964_96440

-- Define the number of sticks each type of furniture makes
def sticks_per_chair := 6
def sticks_per_table := 9
def sticks_per_stool := 2

-- Define the number of each furniture Mary chopped up
def chairs_chopped := 18
def tables_chopped := 6
def stools_chopped := 4

-- Define the total number of hours Mary can keep warm
def hours_warm := 34

-- Calculate the total number of sticks of wood from each type of furniture
def total_sticks_chairs := chairs_chopped * sticks_per_chair
def total_sticks_tables := tables_chopped * sticks_per_table
def total_sticks_stools := stools_chopped * sticks_per_stool

-- Calculate the total number of sticks of wood
def total_sticks := total_sticks_chairs + total_sticks_tables + total_sticks_stools

-- The number of sticks of wood Mary needs to burn per hour
def sticks_per_hour := total_sticks / hours_warm

-- Prove that Mary needs to burn 5 sticks per hour to stay warm
theorem burn_5_sticks_per_hour : sticks_per_hour = 5 := sorry

end burn_5_sticks_per_hour_l964_96440


namespace box_volume_correct_l964_96465

-- Define the dimensions of the original sheet
def length_original : ℝ := 48
def width_original : ℝ := 36

-- Define the side length of the squares cut from each corner
def side_length_cut : ℝ := 4

-- Define the new dimensions after cutting the squares
def new_length : ℝ := length_original - 2 * side_length_cut
def new_width : ℝ := width_original - 2 * side_length_cut

-- Define the height of the box
def height_box : ℝ := side_length_cut

-- Define the expected volume of the box
def volume_box_expected : ℝ := 4480

-- Prove that the calculated volume is equal to the expected volume
theorem box_volume_correct :
  new_length * new_width * height_box = volume_box_expected := by
  sorry

end box_volume_correct_l964_96465


namespace length_of_AB_area_of_ΔABF1_l964_96409

theorem length_of_AB (A B : ℝ × ℝ) (x1 x2 y1 y2 : ℝ) :
  (y1 = x1 - 2) ∧ (y2 = x2 - 2) ∧ ((x1 = 0) ∧ (y1 = -2)) ∧ ((x2 = 8/3) ∧ (y2 = 2/3)) →
  |((x1 - x2)^2 + (y1 - y2)^2)^(1/2)| = (8 / 3) * (2)^(1/2) :=
by sorry

theorem area_of_ΔABF1 (A B F1 : ℝ × ℝ) (x1 x2 y1 y2 : ℝ) :
  (F1 = (0, -2)) ∧ ((y1 = x1 - 2) ∧ (y2 = x2 - 2) ∧ ((x1 = 0) ∧ (y1 = -2)) ∧ ((x2 = 8/3) ∧ (y2 = 2/3))) →
  (1/2) * (((x1 - x2)^2 + (y1 - y2)^2)^(1/2)) * (|(-2-2)/((2)^(1/2))|) = 16 / 3 :=
by sorry

end length_of_AB_area_of_ΔABF1_l964_96409


namespace infinite_geometric_series_sum_l964_96477

theorem infinite_geometric_series_sum
  (a : ℚ) (r : ℚ) (h_a : a = 1) (h_r : r = 2 / 3) (h_r_abs_lt_one : |r| < 1) :
  ∑' (n : ℕ), a * r^n = 3 :=
by
  -- Import necessary lemmas and properties for infinite series
  sorry -- Proof is omitted.

end infinite_geometric_series_sum_l964_96477


namespace find_Japanese_students_l964_96448

theorem find_Japanese_students (C K J : ℕ) (hK: K = (6 * C) / 11) (hJ: J = C / 8) (hK_value: K = 48) : J = 11 :=
by
  sorry

end find_Japanese_students_l964_96448


namespace solve_for_y_l964_96434

theorem solve_for_y {y : ℝ} : 
  (2012 + y)^2 = 2 * y^2 ↔ y = 2012 * (Real.sqrt 2 + 1) ∨ y = -2012 * (Real.sqrt 2 - 1) := by
  sorry

end solve_for_y_l964_96434


namespace value_range_a_for_two_positive_solutions_l964_96479

theorem value_range_a_for_two_positive_solutions (a : ℝ) :
  (∃ (x : ℝ), (|2 * x - 1| - a = 0) ∧ x > 0 ∧ (0 < a ∧ a < 1)) :=
by 
  sorry

end value_range_a_for_two_positive_solutions_l964_96479


namespace floor_of_pi_l964_96419

noncomputable def floor_of_pi_eq_three : Prop :=
  ⌊Real.pi⌋ = 3

theorem floor_of_pi : floor_of_pi_eq_three :=
  sorry

end floor_of_pi_l964_96419


namespace fractions_sum_to_decimal_l964_96461

theorem fractions_sum_to_decimal :
  (2 / 10) + (4 / 100) + (6 / 1000) = 0.246 :=
by 
  sorry

end fractions_sum_to_decimal_l964_96461


namespace no_four_points_with_all_odd_distances_l964_96431

theorem no_four_points_with_all_odd_distances :
  ∀ (A B C D : ℝ × ℝ),
    (∃ (x y z p q r : ℕ),
      (x = dist A B ∧ x % 2 = 1) ∧
      (y = dist B C ∧ y % 2 = 1) ∧
      (z = dist C D ∧ z % 2 = 1) ∧
      (p = dist D A ∧ p % 2 = 1) ∧
      (q = dist A C ∧ q % 2 = 1) ∧
      (r = dist B D ∧ r % 2 = 1))
    → false :=
by
  sorry

end no_four_points_with_all_odd_distances_l964_96431


namespace intersection_A_B_l964_96415

def A : Set ℝ := { x | x > -1 }
def B : Set ℝ := { y | (y - 2) * (y + 3) < 0 }

theorem intersection_A_B : A ∩ B = Set.Ioo (-1) 2 :=
by
  sorry

end intersection_A_B_l964_96415


namespace find_x_equals_4_l964_96411

noncomputable def repeatingExpr (x : ℝ) : ℝ :=
2 + 4 / (1 + 4 / (2 + 4 / (1 + 4 / x)))

theorem find_x_equals_4 :
  ∃ x : ℝ, x = repeatingExpr x ∧ x = 4 :=
by
  use 4
  sorry

end find_x_equals_4_l964_96411


namespace rank_matrix_sum_l964_96412

theorem rank_matrix_sum (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (h : ∀ i j, A i j = ↑i + ↑j) : Matrix.rank A = 2 := by
  sorry

end rank_matrix_sum_l964_96412


namespace gas_cost_correct_l964_96457

def cost_to_fill_remaining_quarter (initial_fill : ℚ) (final_fill : ℚ) (added_gas : ℚ) (cost_per_litre : ℚ) : ℚ :=
  let tank_capacity := (added_gas * (1 / (final_fill - initial_fill)))
  let remaining_quarter_cost := (tank_capacity * (1 / 4)) * cost_per_litre
  remaining_quarter_cost

theorem gas_cost_correct :
  cost_to_fill_remaining_quarter (1/8) (3/4) 30 1.38 = 16.56 :=
by
  sorry

end gas_cost_correct_l964_96457


namespace distinct_dragons_count_l964_96485

theorem distinct_dragons_count : 
  {n : ℕ // n = 7} :=
sorry

end distinct_dragons_count_l964_96485


namespace total_cost_is_83_50_l964_96463

-- Definitions according to the conditions
def cost_adult_ticket : ℝ := 5.50
def cost_child_ticket : ℝ := 3.50
def total_tickets : ℝ := 21
def adult_tickets : ℝ := 5
def child_tickets : ℝ := total_tickets - adult_tickets

-- Total cost calculation based on the conditions
def cost_adult_total : ℝ := adult_tickets * cost_adult_ticket
def cost_child_total : ℝ := child_tickets * cost_child_ticket
def total_cost : ℝ := cost_adult_total + cost_child_total

-- The theorem to prove that the total cost is $83.50
theorem total_cost_is_83_50 : total_cost = 83.50 := by
  sorry

end total_cost_is_83_50_l964_96463


namespace solve_xyz_system_l964_96400

theorem solve_xyz_system :
  ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 
    (x * (6 - y) = 9) ∧ 
    (y * (6 - z) = 9) ∧ 
    (z * (6 - x) = 9) ∧ 
    x = 3 ∧ y = 3 ∧ z = 3 :=
by
  sorry

end solve_xyz_system_l964_96400


namespace factorize_perfect_square_l964_96495

variable (a b : ℤ)

theorem factorize_perfect_square :
  a^2 + 6 * a * b + 9 * b^2 = (a + 3 * b)^2 := 
sorry

end factorize_perfect_square_l964_96495


namespace subtract_abs_from_local_value_l964_96498

-- Define the local value of 4 in 564823 as 4000
def local_value_of_4_in_564823 : ℕ := 4000

-- Define the absolute value of 4 as 4
def absolute_value_of_4 : ℕ := 4

-- Theorem statement: Prove that subtracting the absolute value of 4 from the local value of 4 in 564823 equals 3996
theorem subtract_abs_from_local_value : (local_value_of_4_in_564823 - absolute_value_of_4) = 3996 :=
by
  sorry

end subtract_abs_from_local_value_l964_96498


namespace statement_1_statement_2_statement_3_statement_4_main_proof_l964_96487

noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

theorem statement_1 : ¬ ∃ x, x = 2 ∧ ∀ y, f y ≤ f x := sorry

theorem statement_2 : ∃! x, f x - x = 0 := sorry

theorem statement_3 : ¬ ∃ k > 0, ∀ x > 0, f x > k * x := sorry

theorem statement_4 : ∀ x1 x2 : ℝ, x2 > x1 ∧ f x1 = f x2 → x1 + x2 > 4 := sorry

theorem main_proof : (¬ ∃ x, x = 2 ∧ ∀ y, f y ≤ f x) ∧ 
                     (∃! x, f x - x = 0) ∧ 
                     (¬ ∃ k > 0, ∀ x > 0, f x > k * x) ∧ 
                     (∀ x1 x2 : ℝ, x2 > x1 ∧ f x1 = f x2 → x1 + x2 > 4) := 
by
  apply And.intro
  · exact statement_1
  · apply And.intro
    · exact statement_2
    · apply And.intro
      · exact statement_3
      · exact statement_4

end statement_1_statement_2_statement_3_statement_4_main_proof_l964_96487


namespace decagon_adjacent_probability_l964_96455

-- Define the setup of a decagon with 10 vertices and the adjacency relation.
def is_decagon (n : ℕ) : Prop := n = 10

def adjacent_vertices (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v1 = (v2 + 9) % 10)

-- The main theorem statement, proving the probability of two vertices being adjacent.
theorem decagon_adjacent_probability (n : ℕ) (v1 v2 : ℕ) (h : is_decagon n) : 
  ∀ (v1 v2 : ℕ), v1 ≠ v2 → v1 < n → v2 < n →
  (∃ (p : ℚ), p = 2 / 9 ∧ 
  (adjacent_vertices v1 v2) ↔ (p = 2 / (n - 1))) :=
sorry

end decagon_adjacent_probability_l964_96455


namespace find_c_for_min_value_zero_l964_96427

theorem find_c_for_min_value_zero :
  ∃ c : ℝ, c = 1 ∧ (∀ x y : ℝ, 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 6 * x - 6 * y + 9 ≥ 0) ∧
  (∀ x y : ℝ, 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 6 * x - 6 * y + 9 = 0 → c = 1) :=
by
  use 1
  sorry

end find_c_for_min_value_zero_l964_96427


namespace birdhouse_distance_l964_96413

theorem birdhouse_distance (car_distance : ℕ) (lawnchair_distance : ℕ) (birdhouse_distance : ℕ) 
  (h1 : car_distance = 200) 
  (h2 : lawnchair_distance = 2 * car_distance) 
  (h3 : birdhouse_distance = 3 * lawnchair_distance) : 
  birdhouse_distance = 1200 :=
by
  sorry

end birdhouse_distance_l964_96413
