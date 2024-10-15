import Mathlib

namespace NUMINAMATH_GPT_solve_for_x_l1708_170830

theorem solve_for_x :
  ∀ x : ℤ, 3 * x + 36 = 48 → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1708_170830


namespace NUMINAMATH_GPT_green_more_than_blue_l1708_170827

-- Define the conditions
variables (B Y G n : ℕ)
def ratio_condition := 3 * n = B ∧ 7 * n = Y ∧ 8 * n = G
def total_disks_condition := B + Y + G = 72

-- State the theorem
theorem green_more_than_blue (B Y G n : ℕ) 
  (h_ratio : ratio_condition B Y G n) 
  (h_total : total_disks_condition B Y G) 
  : G - B = 20 := 
sorry

end NUMINAMATH_GPT_green_more_than_blue_l1708_170827


namespace NUMINAMATH_GPT_fraction_of_orange_juice_in_large_container_l1708_170835

def total_capacity := 800 -- mL for each pitcher
def orange_juice_first_pitcher := total_capacity / 2 -- 400 mL
def orange_juice_second_pitcher := total_capacity / 4 -- 200 mL
def total_orange_juice := orange_juice_first_pitcher + orange_juice_second_pitcher -- 600 mL
def total_volume := total_capacity + total_capacity -- 1600 mL

theorem fraction_of_orange_juice_in_large_container :
  (total_orange_juice / total_volume) = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_orange_juice_in_large_container_l1708_170835


namespace NUMINAMATH_GPT_polynomial_solution_exists_l1708_170806

open Real

theorem polynomial_solution_exists
    (P : ℝ → ℝ → ℝ)
    (hP : ∃ (f : ℝ → ℝ), ∀ x y : ℝ, P x y = f (x + y) - f x - f y) :
  ∃ (q : ℝ → ℝ), ∀ x y : ℝ, P x y = q (x + y) - q x - q y := sorry

end NUMINAMATH_GPT_polynomial_solution_exists_l1708_170806


namespace NUMINAMATH_GPT_rotten_eggs_prob_l1708_170871

theorem rotten_eggs_prob (T : ℕ) (P : ℝ) (R : ℕ) :
  T = 36 ∧ P = 0.0047619047619047615 ∧ P = (R / T) * ((R - 1) / (T - 1)) → R = 3 :=
by
  sorry

end NUMINAMATH_GPT_rotten_eggs_prob_l1708_170871


namespace NUMINAMATH_GPT_equation_of_latus_rectum_l1708_170875

theorem equation_of_latus_rectum (y x : ℝ) : (x = -1/4) ∧ (y^2 = x) ↔ (2 * (1 / 2) = 1) ∧ (l = - (1 / 2) / 2) := sorry

end NUMINAMATH_GPT_equation_of_latus_rectum_l1708_170875


namespace NUMINAMATH_GPT_math_problem_l1708_170892

variables {x y : ℝ}

theorem math_problem (h1 : x + y = 6) (h2 : x * y = 5) :
  (2 / x + 2 / y = 12 / 5) ∧ ((x - y) ^ 2 = 16) ∧ (x ^ 2 + y ^ 2 = 26) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1708_170892


namespace NUMINAMATH_GPT_point_on_x_axis_l1708_170807

theorem point_on_x_axis : ∃ p, (p = (-2, 0) ∧ p.snd = 0) ∧
  ((p ≠ (0, 2)) ∧ (p ≠ (-2, -3)) ∧ (p ≠ (-1, -2))) :=
by
  sorry

end NUMINAMATH_GPT_point_on_x_axis_l1708_170807


namespace NUMINAMATH_GPT_symmetry_axis_of_f_l1708_170865

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem symmetry_axis_of_f :
  ∃ k : ℤ, ∃ k_π_div_2 : ℝ, (f (k * π / 2 + π / 12) = f ((k * π / 2 + π / 12) + π)) :=
by {
  sorry
}

end NUMINAMATH_GPT_symmetry_axis_of_f_l1708_170865


namespace NUMINAMATH_GPT_cost_of_history_book_l1708_170810

theorem cost_of_history_book (total_books : ℕ) (cost_math_book : ℕ) (total_price : ℕ) (num_math_books : ℕ) (num_history_books : ℕ) (cost_history_book : ℕ) 
    (h_books_total : total_books = 90)
    (h_cost_math : cost_math_book = 4)
    (h_total_price : total_price = 396)
    (h_num_math_books : num_math_books = 54)
    (h_num_total_books : num_math_books + num_history_books = total_books)
    (h_total_cost : num_math_books * cost_math_book + num_history_books * cost_history_book = total_price) : cost_history_book = 5 := by 
  sorry

end NUMINAMATH_GPT_cost_of_history_book_l1708_170810


namespace NUMINAMATH_GPT_sin_transformation_identity_l1708_170808

theorem sin_transformation_identity 
  (θ : ℝ) 
  (h : Real.cos (π / 12 - θ) = 1 / 3) : 
  Real.sin (2 * θ + π / 3) = -7 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_sin_transformation_identity_l1708_170808


namespace NUMINAMATH_GPT_projection_matrix_exists_l1708_170814

noncomputable def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, (20 : ℚ) / 49], ![c, (29 : ℚ) / 49]]

theorem projection_matrix_exists :
  ∃ (a c : ℚ), P a c * P a c = P a c ∧ a = (20 : ℚ) / 49 ∧ c = (29 : ℚ) / 49 := 
by
  use ((20 : ℚ) / 49), ((29 : ℚ) / 49)
  simp [P]
  sorry

end NUMINAMATH_GPT_projection_matrix_exists_l1708_170814


namespace NUMINAMATH_GPT_find_number_of_white_balls_l1708_170859

-- Define the conditions
variables (n k : ℕ)
axiom k_ge_2 : k ≥ 2
axiom prob_white_black : (n * k) / ((n + k) * (n + k - 1)) = n / 100

-- State the theorem
theorem find_number_of_white_balls (n k : ℕ) (k_ge_2 : k ≥ 2) (prob_white_black : (n * k) / ((n + k) * (n + k - 1)) = n / 100) : n = 19 :=
sorry

end NUMINAMATH_GPT_find_number_of_white_balls_l1708_170859


namespace NUMINAMATH_GPT_common_difference_is_two_l1708_170896

-- Define the properties and conditions.
variables {a : ℕ → ℝ} {d : ℝ}

-- An arithmetic sequence definition.
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Problem statement to be proved.
theorem common_difference_is_two (h1 : a 1 + a 5 = 10) (h2 : a 4 = 7) (h3 : arithmetic_sequence a d) : 
  d = 2 :=
sorry

end NUMINAMATH_GPT_common_difference_is_two_l1708_170896


namespace NUMINAMATH_GPT_find_functions_l1708_170891

def satisfies_equation (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))

theorem find_functions (f : ℤ → ℤ) (h : satisfies_equation f) : (∀ x, f x = 2 * x) ∨ (∀ x, f x = 0) :=
sorry

end NUMINAMATH_GPT_find_functions_l1708_170891


namespace NUMINAMATH_GPT_simplify_cos_diff_l1708_170895

theorem simplify_cos_diff :
  let a := Real.cos (36 * Real.pi / 180)
  let b := Real.cos (72 * Real.pi / 180)
  (b = 2 * a^2 - 1) → 
  (a = 1 - 2 * b^2) →
  a - b = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_cos_diff_l1708_170895


namespace NUMINAMATH_GPT_basis_transformation_l1708_170850

variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

theorem basis_transformation (h_basis : ∀ (v : V), ∃ (x y z : ℝ), v = x • a + y • b + z • c) :
  ∀ (v : V), ∃ (x y z : ℝ), v = x • (a + b) + y • (a - c) + z • b :=
by {
  sorry  -- to skip the proof steps for now
}

end NUMINAMATH_GPT_basis_transformation_l1708_170850


namespace NUMINAMATH_GPT_prince_spending_l1708_170898

theorem prince_spending (CDs_total : ℕ) (CDs_10_percent : ℕ) (CDs_10_cost : ℕ) (CDs_5_cost : ℕ) 
  (Prince_10_fraction : ℚ) (Prince_5_fraction : ℚ) 
  (total_10_CDs : ℕ) (total_5_CDs : ℕ) (Prince_10_CDs : ℕ) (Prince_5_CDs : ℕ) (total_cost : ℕ) :
  CDs_total = 200 →
  CDs_10_percent = 40 →
  CDs_10_cost = 10 →
  CDs_5_cost = 5 →
  Prince_10_fraction = 1/2 →
  Prince_5_fraction = 1 →
  total_10_CDs = CDs_total * CDs_10_percent / 100 →
  total_5_CDs = CDs_total - total_10_CDs →
  Prince_10_CDs = total_10_CDs * Prince_10_fraction →
  Prince_5_CDs = total_5_CDs * Prince_5_fraction →
  total_cost = (Prince_10_CDs * CDs_10_cost) + (Prince_5_CDs * CDs_5_cost) →
  total_cost = 1000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end NUMINAMATH_GPT_prince_spending_l1708_170898


namespace NUMINAMATH_GPT_total_green_marbles_l1708_170854

-- Conditions
def Sara_green_marbles : ℕ := 3
def Tom_green_marbles : ℕ := 4

-- Problem statement: proving the total number of green marbles
theorem total_green_marbles : Sara_green_marbles + Tom_green_marbles = 7 := by
  sorry

end NUMINAMATH_GPT_total_green_marbles_l1708_170854


namespace NUMINAMATH_GPT_time_for_one_essay_l1708_170877

-- We need to define the times for questions and paragraphs first.

def time_per_short_answer_question := 3 -- in minutes
def time_per_paragraph := 15 -- in minutes
def total_homework_time := 4 -- in hours
def num_essays := 2
def num_paragraphs := 5
def num_short_answer_questions := 15

-- Now we need to state the total homework time and define the goal
def computed_homework_time :=
  (time_per_short_answer_question * num_short_answer_questions +
   time_per_paragraph * num_paragraphs) / 60 + num_essays * sorry -- time for one essay in hours

theorem time_for_one_essay :
  (total_homework_time = computed_homework_time) → sorry = 1 :=
by
  sorry

end NUMINAMATH_GPT_time_for_one_essay_l1708_170877


namespace NUMINAMATH_GPT_max_cookies_eaten_l1708_170869

def prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem max_cookies_eaten 
  (total_cookies : ℕ)
  (andy_cookies : ℕ)
  (alexa_cookies : ℕ)
  (hx : andy_cookies + alexa_cookies = total_cookies)
  (hp : ∃ p : ℕ, prime p ∧ alexa_cookies = p * andy_cookies)
  (htotal : total_cookies = 30) :
  andy_cookies = 10 :=
  sorry

end NUMINAMATH_GPT_max_cookies_eaten_l1708_170869


namespace NUMINAMATH_GPT_tan_alpha_in_second_quadrant_l1708_170800

theorem tan_alpha_in_second_quadrant (α : ℝ) (h₁ : π / 2 < α ∧ α < π) (hsin : Real.sin α = 5 / 13) :
    Real.tan α = -5 / 12 :=
sorry

end NUMINAMATH_GPT_tan_alpha_in_second_quadrant_l1708_170800


namespace NUMINAMATH_GPT_find_distance_AB_l1708_170805

variable (vA vB : ℝ) -- speeds of Person A and Person B
variable (x : ℝ) -- distance between points A and B
variable (t1 t2 : ℝ) -- time variables

-- Conditions
def startTime := 0
def meetDistanceBC := 240
def returnPointBDistantFromA := 120
def doublingSpeedFactor := 2

-- Main questions and conditions
theorem find_distance_AB
  (h1 : vA > vB)
  (h2 : t1 = x / vB)
  (h3 : t2 = 2 * (x - meetDistanceBC) / vA) 
  (h4 : x = meetDistanceBC + returnPointBDistantFromA + (t1 * (doublingSpeedFactor * vB) - t2 * vA) / (doublingSpeedFactor - 1)) :
  x = 420 :=
sorry

end NUMINAMATH_GPT_find_distance_AB_l1708_170805


namespace NUMINAMATH_GPT_russian_needed_goals_equals_tunisian_scored_goals_l1708_170820

-- Define the total goals required by each team
def russian_goals := 9
def tunisian_goals := 5

-- Statement: there exists a moment where the Russian remaining goals equal the Tunisian scored goals
theorem russian_needed_goals_equals_tunisian_scored_goals :
  ∃ n : ℕ, n ≤ russian_goals ∧ (russian_goals - n) = (tunisian_goals) := by
  sorry

end NUMINAMATH_GPT_russian_needed_goals_equals_tunisian_scored_goals_l1708_170820


namespace NUMINAMATH_GPT_statement2_statement3_l1708_170834

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Conditions for the statements
axiom cond1 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c p = q ∧ f a b c q = p
axiom cond2 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c p = f a b c q
axiom cond3 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c (p + q) = c

-- Statement 2 correctness
theorem statement2 (a b c p q : ℝ) (hpq : p ≠ q) (h : f a b c p = f a b c q) : 
  f a b c (p + q) = c :=
sorry

-- Statement 3 correctness
theorem statement3 (a b c p q : ℝ) (hpq : p ≠ q) (h : f a b c (p + q) = c) : 
  p + q = 0 ∨ f a b c p = f a b c q :=
sorry

end NUMINAMATH_GPT_statement2_statement3_l1708_170834


namespace NUMINAMATH_GPT_simplify_fraction_l1708_170833

def expr1 : ℚ := 3
def expr2 : ℚ := 2
def expr3 : ℚ := 3
def expr4 : ℚ := 4
def expected : ℚ := 12 / 5

theorem simplify_fraction : (expr1 / (expr2 - (expr3 / expr4))) = expected := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1708_170833


namespace NUMINAMATH_GPT_integral_negative_of_negative_function_l1708_170868

theorem integral_negative_of_negative_function {f : ℝ → ℝ} 
  (hf_cont : Continuous f) 
  (hf_neg : ∀ x, f x < 0) 
  {a b : ℝ} 
  (hab : a < b) 
  : ∫ x in a..b, f x < 0 := 
sorry

end NUMINAMATH_GPT_integral_negative_of_negative_function_l1708_170868


namespace NUMINAMATH_GPT_eight_people_lineup_two_windows_l1708_170812

theorem eight_people_lineup_two_windows :
  (2 ^ 8) * (Nat.factorial 8) = 10321920 := by
  sorry

end NUMINAMATH_GPT_eight_people_lineup_two_windows_l1708_170812


namespace NUMINAMATH_GPT_part_a_smallest_number_of_lines_part_b_smallest_number_of_lines_l1708_170822

theorem part_a_smallest_number_of_lines (n : ℕ) : 
  (n * (n - 1)) / 2 ≥ 2022 → ∃ k : ℕ, k = 65 :=
sorry

theorem part_b_smallest_number_of_lines (n : ℕ) : 
  (n * (n - 1)) / 2 ≥ 2023 → ∃ k : ℕ, k = 65 :=
sorry

end NUMINAMATH_GPT_part_a_smallest_number_of_lines_part_b_smallest_number_of_lines_l1708_170822


namespace NUMINAMATH_GPT_total_journey_distance_l1708_170838

theorem total_journey_distance : 
  ∃ D : ℝ, 
    (∀ (T : ℝ), T = 10) →
    ((D/2) / 21 + (D/2) / 24 = 10) →
    D = 224 := 
by
  sorry

end NUMINAMATH_GPT_total_journey_distance_l1708_170838


namespace NUMINAMATH_GPT_total_trees_planted_l1708_170860

/-- A yard is 255 meters long, with a tree at each end and trees planted at intervals of 15 meters. -/
def yard_length : ℤ := 255

def tree_interval : ℤ := 15

def total_trees : ℤ := 18

theorem total_trees_planted (L : ℤ) (d : ℤ) (n : ℤ) : 
  L = yard_length →
  d = tree_interval →
  n = total_trees →
  n = (L / d) + 1 :=
by
  intros hL hd hn
  rw [hL, hd, hn]
  sorry

end NUMINAMATH_GPT_total_trees_planted_l1708_170860


namespace NUMINAMATH_GPT_sin_alpha_plus_pi_over_2_l1708_170811

theorem sin_alpha_plus_pi_over_2 
  (h1 : Real.pi / 2 < α) (h2 : α < Real.pi) (h3 : Real.tan α = -4 / 3) :
  Real.sin (α + Real.pi / 2) = -3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_plus_pi_over_2_l1708_170811


namespace NUMINAMATH_GPT_roots_of_quadratic_eq_l1708_170855

noncomputable def r : ℂ := sorry
noncomputable def s : ℂ := sorry

def roots_eq (h : 3 * r^2 + 4 * r + 2 = 0 ∧ 3 * s^2 + 4 * s + 2 = 0) : Prop :=
  (1 / r^3) + (1 / s^3) = 1

theorem roots_of_quadratic_eq (h:3 * r^2 + 4 * r + 2 = 0 ∧ 3 * s^2 + 4 * s + 2 = 0) : roots_eq h :=
sorry

end NUMINAMATH_GPT_roots_of_quadratic_eq_l1708_170855


namespace NUMINAMATH_GPT_cost_of_adult_ticket_l1708_170881

theorem cost_of_adult_ticket
  (A : ℝ) -- Cost of an adult ticket in dollars
  (x y : ℝ) -- Number of children tickets and number of adult tickets respectively
  (hx : x = 90) -- Condition: number of children tickets sold
  (hSum : x + y = 130) -- Condition: total number of tickets sold
  (hTotal : 4 * x + A * y = 840) -- Condition: total receipts from all tickets
  : A = 12 := 
by
  -- Proof is skipped as per instruction
  sorry

end NUMINAMATH_GPT_cost_of_adult_ticket_l1708_170881


namespace NUMINAMATH_GPT_Sheelas_monthly_income_l1708_170845

theorem Sheelas_monthly_income (I : ℝ) (h : 0.32 * I = 3800) : I = 11875 :=
by
  sorry

end NUMINAMATH_GPT_Sheelas_monthly_income_l1708_170845


namespace NUMINAMATH_GPT_abs_inequality_solution_l1708_170826

theorem abs_inequality_solution {a : ℝ} (h : ∀ x : ℝ, |2 - x| + |x + 1| ≥ a) : a ≤ 3 :=
sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1708_170826


namespace NUMINAMATH_GPT_average_rate_l1708_170804

variable (d_run : ℝ) (d_swim : ℝ) (r_run : ℝ) (r_swim : ℝ)
variable (t_run : ℝ := d_run / r_run) (t_swim : ℝ := d_swim / r_swim)

theorem average_rate (h_dist_run : d_run = 4) (h_dist_swim : d_swim = 4)
                      (h_run_rate : r_run = 10) (h_swim_rate : r_swim = 6) : 
                      ((d_run + d_swim) / (t_run + t_swim)) / 60 = 0.125 :=
by
  -- Properly using all the conditions given
  have := (4 + 4) / (4 / 10 + 4 / 6) / 60 = 0.125
  sorry

end NUMINAMATH_GPT_average_rate_l1708_170804


namespace NUMINAMATH_GPT_monotonicity_and_range_of_a_l1708_170848

noncomputable def f (x a : ℝ) := Real.log x - a * x - 2

theorem monotonicity_and_range_of_a (a : ℝ) (h : a ≠ 0) :
  ((∀ x > 0, (Real.log x - a * x - 2) < (Real.log (x + 1) - a * (x + 1) - 2)) ↔ (a < 0)) ∧
  ((∃ M, M = Real.log (1/a) - a * (1/a) - 2 ∧ M > a - 4) → 0 < a ∧ a < 1) := sorry

end NUMINAMATH_GPT_monotonicity_and_range_of_a_l1708_170848


namespace NUMINAMATH_GPT_angle_ratio_half_l1708_170886

theorem angle_ratio_half (a b c : ℝ) (A B C : ℝ) (h1 : a^2 = b * (b + c))
  (h2 : A = 2 * B ∨ A + 2 * B = Real.pi) 
  (h3 : A + B + C = Real.pi) : 
  (B / A = 1 / 2) :=
sorry

end NUMINAMATH_GPT_angle_ratio_half_l1708_170886


namespace NUMINAMATH_GPT_difference_between_numbers_l1708_170836

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 20000) (h2 : b = 2 * a + 6) (h3 : 9 ∣ a) : b - a = 6670 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_numbers_l1708_170836


namespace NUMINAMATH_GPT_distance_in_scientific_notation_l1708_170861

theorem distance_in_scientific_notation :
  ∃ a n : ℝ, 1 ≤ |a| ∧ |a| < 10 ∧ n = 4 ∧ 38000 = a * 10^n ∧ a = 3.8 :=
by
  sorry

end NUMINAMATH_GPT_distance_in_scientific_notation_l1708_170861


namespace NUMINAMATH_GPT_quadratic_root_shift_l1708_170824

theorem quadratic_root_shift (r s : ℝ)
    (hr : 2 * r^2 - 8 * r + 6 = 0)
    (hs : 2 * s^2 - 8 * s + 6 = 0)
    (h_sum_roots : r + s = 4)
    (h_prod_roots : r * s = 3)
    (b : ℝ) (c : ℝ)
    (h_b : b = - (r - 3 + s - 3))
    (h_c : c = (r - 3) * (s - 3)) : c = 0 :=
  by sorry

end NUMINAMATH_GPT_quadratic_root_shift_l1708_170824


namespace NUMINAMATH_GPT_reflected_line_eq_l1708_170890

noncomputable def point_symmetric_reflection :=
  ∃ (A : ℝ × ℝ) (B : ℝ × ℝ) (A' : ℝ × ℝ),
  A = (-1 / 2, 0) ∧ B = (0, 1) ∧ A' = (1 / 2, 0) ∧ 
  ∀ (x y : ℝ), 2 * x + y = 1 ↔
  (y - 1) / (0 - 1) = x / (1 / 2 - 0)

theorem reflected_line_eq :
  point_symmetric_reflection :=
sorry

end NUMINAMATH_GPT_reflected_line_eq_l1708_170890


namespace NUMINAMATH_GPT_original_number_of_men_l1708_170897

variable (M W : ℕ)

def original_work_condition := M * W / 60 = W
def larger_group_condition := (M + 8) * W / 50 = W

theorem original_number_of_men : original_work_condition M W ∧ larger_group_condition M W → M = 48 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_men_l1708_170897


namespace NUMINAMATH_GPT_extreme_value_a_range_l1708_170832

theorem extreme_value_a_range (a : ℝ) :
  (∃ x : ℝ, (1 / Real.exp 1 < x ∧ x < Real.exp 1 ∧ x + a * Real.log x + 1 + a / x = 0)) →
  -Real.exp 1 < a ∧ a < -1 / Real.exp 1 :=
by sorry

end NUMINAMATH_GPT_extreme_value_a_range_l1708_170832


namespace NUMINAMATH_GPT_find_train_length_l1708_170849

noncomputable def speed_kmh : ℝ := 45
noncomputable def bridge_length : ℝ := 245.03
noncomputable def time_seconds : ℝ := 30
noncomputable def speed_ms : ℝ := (speed_kmh * 1000) / 3600
noncomputable def total_distance : ℝ := speed_ms * time_seconds
noncomputable def train_length : ℝ := total_distance - bridge_length

theorem find_train_length : train_length = 129.97 := 
by
  sorry

end NUMINAMATH_GPT_find_train_length_l1708_170849


namespace NUMINAMATH_GPT_cookie_radius_proof_l1708_170862

-- Define the given equation of the cookie
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 36 = 6 * x + 9 * y

-- Define the radius computation for the circle derived from the given equation
def cookie_radius (r : ℝ) : Prop :=
  r = 3 * Real.sqrt 5 / 2

-- The theorem to prove that the radius of the described cookie is as obtained
theorem cookie_radius_proof :
  ∀ x y : ℝ, cookie_equation x y → cookie_radius (Real.sqrt (45 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_cookie_radius_proof_l1708_170862


namespace NUMINAMATH_GPT_arith_seq_formula_geom_seq_sum_l1708_170851

-- Definitions for condition 1: Arithmetic sequence {a_n}
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  (a 4 = 7) ∧ (a 10 = 19)

-- Definitions for condition 2: Sum of the first n terms of {a_n}
def sum_arith_seq (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Definitions for condition 3: Geometric sequence {b_n}
def geometric_seq (b : ℕ → ℤ) : Prop :=
  (b 1 = 2) ∧ (∀ n, b (n + 1) = b n * 2)

-- Definitions for condition 4: Sum of the first n terms of {b_n}
def sum_geom_seq (T : ℕ → ℤ) (b : ℕ → ℤ) : Prop :=
  ∀ n, T n = (b 1 * (1 - (2 ^ n))) / (1 - 2)

-- Proving the general formula for arithmetic sequence
theorem arith_seq_formula (a : ℕ → ℤ) (S : ℕ → ℤ) :
  arithmetic_seq a ∧ sum_arith_seq S a → 
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, S n = n ^ 2) :=
sorry

-- Proving the sum of the first n terms for geometric sequence
theorem geom_seq_sum (b : ℕ → ℤ) (T : ℕ → ℤ) (S : ℕ → ℤ) :
  geometric_seq b ∧ sum_geom_seq T b ∧ b 4 = S 4 → 
  (∀ n, T n = 2 ^ (n + 1) - 2) :=
sorry

end NUMINAMATH_GPT_arith_seq_formula_geom_seq_sum_l1708_170851


namespace NUMINAMATH_GPT_cos_210_eq_neg_sqrt3_div_2_l1708_170815

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end NUMINAMATH_GPT_cos_210_eq_neg_sqrt3_div_2_l1708_170815


namespace NUMINAMATH_GPT_lowest_fraction_of_job_done_l1708_170866

theorem lowest_fraction_of_job_done :
  ∀ (rateA rateB rateC rateB_plus_C : ℝ),
  (rateA = 1/4) → (rateB = 1/6) → (rateC = 1/8) →
  (rateB_plus_C = rateB + rateC) →
  rateB_plus_C = 7/24 := by
  intros rateA rateB rateC rateB_plus_C hA hB hC hBC
  sorry

end NUMINAMATH_GPT_lowest_fraction_of_job_done_l1708_170866


namespace NUMINAMATH_GPT_find_original_denominator_l1708_170863

theorem find_original_denominator (d : ℕ) 
  (h : (10 : ℚ) / (d + 7) = 1 / 3) : 
  d = 23 :=
by 
  sorry

end NUMINAMATH_GPT_find_original_denominator_l1708_170863


namespace NUMINAMATH_GPT_find_solutions_l1708_170885

theorem find_solutions :
  ∀ x y : Real, 
  (3 / 20) + abs (x - (15 / 40)) < (7 / 20) →
  y = 2 * x + 1 →
  (7 / 20) < x ∧ x < (2 / 5) ∧ (17 / 10) ≤ y ∧ y ≤ (11 / 5) :=
by
  intros x y h₁ h₂
  sorry

end NUMINAMATH_GPT_find_solutions_l1708_170885


namespace NUMINAMATH_GPT_unknown_road_length_l1708_170872

/-
  Given the lengths of four roads and the Triangle Inequality condition, 
  prove the length of the fifth road.
  Given lengths: a = 10 km, b = 5 km, c = 8 km, d = 21 km.
-/

theorem unknown_road_length
  (a b c d : ℕ) (h0 : a = 10) (h1 : b = 5) (h2 : c = 8) (h3 : d = 21)
  (x : ℕ) :
  2 < x ∧ x < 18 ∧ 16 < x ∧ x < 26 → x = 17 :=
by
  intros
  sorry

end NUMINAMATH_GPT_unknown_road_length_l1708_170872


namespace NUMINAMATH_GPT_express_in_standard_form_l1708_170828

theorem express_in_standard_form (x : ℝ) : x^2 - 6 * x = (x - 3)^2 - 9 :=
by
  sorry

end NUMINAMATH_GPT_express_in_standard_form_l1708_170828


namespace NUMINAMATH_GPT_discount_difference_l1708_170825

def single_discount (original: ℝ) (discount: ℝ) : ℝ :=
  original * (1 - discount)

def successive_discount (original: ℝ) (first_discount: ℝ) (second_discount: ℝ) : ℝ :=
  original * (1 - first_discount) * (1 - second_discount)

theorem discount_difference : 
  let original := 12000
  let single_disc := 0.30
  let first_disc := 0.20
  let second_disc := 0.10
  single_discount original single_disc - successive_discount original first_disc second_disc = 240 := 
by sorry

end NUMINAMATH_GPT_discount_difference_l1708_170825


namespace NUMINAMATH_GPT_JamesFlowers_l1708_170867

noncomputable def numberOfFlowersJamesPlantedInADay (F : ℝ) := 0.5 * (F + 0.15 * F)

theorem JamesFlowers (F : ℝ) (H₁ : 6 * F + (F + 0.15 * F) = 315) : numberOfFlowersJamesPlantedInADay F = 25.3:=
by
  sorry

end NUMINAMATH_GPT_JamesFlowers_l1708_170867


namespace NUMINAMATH_GPT_seven_not_spheric_spheric_power_spheric_l1708_170823

def is_spheric (r : ℚ) : Prop := ∃ x y z : ℚ, r = x^2 + y^2 + z^2

theorem seven_not_spheric : ¬ is_spheric 7 := 
sorry

theorem spheric_power_spheric (r : ℚ) (n : ℕ) (h : is_spheric r) (hn : n > 1) : is_spheric (r ^ n) := 
sorry

end NUMINAMATH_GPT_seven_not_spheric_spheric_power_spheric_l1708_170823


namespace NUMINAMATH_GPT_harry_spends_1920_annually_l1708_170847

def geckoCount : Nat := 3
def iguanaCount : Nat := 2
def snakeCount : Nat := 4

def geckoFeedTimesPerMonth : Nat := 2
def iguanaFeedTimesPerMonth : Nat := 3
def snakeFeedTimesPerMonth : Nat := 1 / 2

def geckoFeedCostPerMeal : Nat := 8
def iguanaFeedCostPerMeal : Nat := 12
def snakeFeedCostPerMeal : Nat := 20

def annualCostHarrySpends (geckoCount guCount scCount : Nat) (geckoFeedTimesPerMonth iguanaFeedTimesPerMonth snakeFeedTimesPerMonth : Nat) (geckoFeedCostPerMeal iguanaFeedCostPerMeal snakeFeedCostPerMeal : Nat) : Nat :=
  let geckoAnnualCost := geckoCount * (geckoFeedTimesPerMonth * 12 * geckoFeedCostPerMeal)
  let iguanaAnnualCost := iguanaCount * (iguanaFeedTimesPerMonth * 12 * iguanaFeedCostPerMeal)
  let snakeAnnualCost := snakeCount * ((12 / (2 : Nat)) * snakeFeedCostPerMeal)
  geckoAnnualCost + iguanaAnnualCost + snakeAnnualCost

theorem harry_spends_1920_annually : annualCostHarrySpends geckoCount iguanaCount snakeCount geckoFeedTimesPerMonth iguanaFeedTimesPerMonth snakeFeedTimesPerMonth geckoFeedCostPerMeal iguanaFeedCostPerMeal snakeFeedCostPerMeal = 1920 := 
  sorry

end NUMINAMATH_GPT_harry_spends_1920_annually_l1708_170847


namespace NUMINAMATH_GPT_ratio_rectangle_to_semicircles_area_l1708_170818

theorem ratio_rectangle_to_semicircles_area (AB AD : ℝ) (h1 : AB = 40) (h2 : AD / AB = 3 / 2) : 
  (AB * AD) / (2 * (π * (AB / 2)^2)) = 6 / π :=
by
  -- here we process the proof
  sorry

end NUMINAMATH_GPT_ratio_rectangle_to_semicircles_area_l1708_170818


namespace NUMINAMATH_GPT_defective_percentage_is_correct_l1708_170821

noncomputable def percentage_defective (defective : ℕ) (total : ℝ) : ℝ := 
  (defective / total) * 100

theorem defective_percentage_is_correct : 
  percentage_defective 2 3333.3333333333335 = 0.06000600060006 :=
by
  sorry

end NUMINAMATH_GPT_defective_percentage_is_correct_l1708_170821


namespace NUMINAMATH_GPT_remainder_of_division_l1708_170889

theorem remainder_of_division (x r : ℕ) (h : 23 = 7 * x + r) : r = 2 :=
sorry

end NUMINAMATH_GPT_remainder_of_division_l1708_170889


namespace NUMINAMATH_GPT_find_point_P_l1708_170883

-- Define the function
def f (x : ℝ) := x^4 - 2 * x

-- Define the derivative of the function
def f' (x : ℝ) := 4 * x^3 - 2

theorem find_point_P :
  ∃ (P : ℝ × ℝ), (f' P.1 = 2) ∧ (f P.1 = P.2) ∧ (P = (1, -1)) :=
by
  -- here would go the actual proof
  sorry

end NUMINAMATH_GPT_find_point_P_l1708_170883


namespace NUMINAMATH_GPT_exp_gt_f_n_y_between_0_and_x_l1708_170842

open Real

noncomputable def f_n (x : ℝ) (n : ℕ) : ℝ :=
  (Finset.range (n + 1)).sum (λ k => x^k / k.factorial)

theorem exp_gt_f_n (x : ℝ) (n : ℕ) (h1 : 0 < x) :
  exp x > f_n x n :=
sorry

theorem y_between_0_and_x (x : ℝ) (n : ℕ) (y : ℝ)
  (h1 : 0 < x)
  (h2 : exp x = f_n x n + x^(n+1) / (n + 1).factorial * exp y) :
  0 < y ∧ y < x :=
sorry

end NUMINAMATH_GPT_exp_gt_f_n_y_between_0_and_x_l1708_170842


namespace NUMINAMATH_GPT_pears_seed_avg_l1708_170873

def apple_seed_avg : ℕ := 6
def grape_seed_avg : ℕ := 3
def total_seeds_required : ℕ := 60
def apples_count : ℕ := 4
def pears_count : ℕ := 3
def grapes_count : ℕ := 9
def seeds_short : ℕ := 3
def total_seeds_obtained : ℕ := total_seeds_required - seeds_short

theorem pears_seed_avg :
  (apples_count * apple_seed_avg) + (grapes_count * grape_seed_avg) + (pears_count * P) = total_seeds_obtained → 
  P = 2 :=
by
  sorry

end NUMINAMATH_GPT_pears_seed_avg_l1708_170873


namespace NUMINAMATH_GPT_num_of_valid_numbers_l1708_170853

def is_valid_number (n : ℕ) : Prop :=
  let a := n / 10
  let b := n % 10
  a >= 1 ∧ a <= 9 ∧ b >= 0 ∧ b <= 9 ∧ (9 * a) % 10 = 4

theorem num_of_valid_numbers : ∃ n, n = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_num_of_valid_numbers_l1708_170853


namespace NUMINAMATH_GPT_min_abc_sum_l1708_170801

theorem min_abc_sum (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 2010) : 
  a + b + c ≥ 78 := 
sorry

end NUMINAMATH_GPT_min_abc_sum_l1708_170801


namespace NUMINAMATH_GPT_numWaysElectOfficers_l1708_170856

-- Definitions and conditions from part (a)
def numMembers : Nat := 30
def numPositions : Nat := 5
def members := ["Alice", "Bob", "Carol", "Dave"]
def allOrNoneCondition (S : List String) : Bool := 
  S.all (members.contains)

-- Function to count the number of ways to choose the officers
def countWays (n : Nat) (k : Nat) (allOrNone : Bool) : Nat :=
if allOrNone then
  -- All four members are positioned
  Nat.factorial k * (n - k)
else
  -- None of the four members are positioned
  let remaining := n - members.length
  remaining * (remaining - 1) * (remaining - 2) * (remaining - 3) * (remaining - 4)

theorem numWaysElectOfficers :
  let casesWithNone := countWays numMembers numPositions false
  let casesWithAll := countWays numMembers numPositions true
  (casesWithNone + casesWithAll) = 6378720 :=
by
  sorry

end NUMINAMATH_GPT_numWaysElectOfficers_l1708_170856


namespace NUMINAMATH_GPT_ticket_difference_l1708_170887

-- Definitions representing the number of VIP and general admission tickets
def numTickets (V G : Nat) : Prop :=
  V + G = 320

def totalCost (V G : Nat) : Prop :=
  40 * V + 15 * G = 7500

-- Theorem stating that the difference between general admission and VIP tickets is 104
theorem ticket_difference (V G : Nat) (h1 : numTickets V G) (h2 : totalCost V G) : G - V = 104 := by
  sorry

end NUMINAMATH_GPT_ticket_difference_l1708_170887


namespace NUMINAMATH_GPT_sara_spent_on_hotdog_l1708_170870

def total_cost_of_lunch: ℝ := 10.46
def cost_of_salad: ℝ := 5.10
def cost_of_hotdog: ℝ := total_cost_of_lunch - cost_of_salad

theorem sara_spent_on_hotdog :
  cost_of_hotdog = 5.36 := by
  sorry

end NUMINAMATH_GPT_sara_spent_on_hotdog_l1708_170870


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l1708_170802

variables {a b c q : ℝ}

theorem geometric_sequence_ratio (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sequence : ∃ q : ℝ, (a + b + c) * q = b + c - a ∧
                         (a + b + c) * q^2 = c + a - b ∧
                         (a + b + c) * q^3 = a + b - c) :
  q^3 + q^2 + q = 1 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l1708_170802


namespace NUMINAMATH_GPT_total_players_l1708_170809

theorem total_players 
  (cricket_players : ℕ) (hockey_players : ℕ)
  (football_players : ℕ) (softball_players : ℕ)
  (h_cricket : cricket_players = 12)
  (h_hockey : hockey_players = 17)
  (h_football : football_players = 11)
  (h_softball : softball_players = 10)
  : cricket_players + hockey_players + football_players + softball_players = 50 :=
by sorry

end NUMINAMATH_GPT_total_players_l1708_170809


namespace NUMINAMATH_GPT_total_pebbles_count_l1708_170843

def white_pebbles : ℕ := 20
def red_pebbles : ℕ := white_pebbles / 2
def blue_pebbles : ℕ := red_pebbles / 3
def green_pebbles : ℕ := blue_pebbles + 5

theorem total_pebbles_count : white_pebbles + red_pebbles + blue_pebbles + green_pebbles = 41 := by
  sorry

end NUMINAMATH_GPT_total_pebbles_count_l1708_170843


namespace NUMINAMATH_GPT_find_fraction_l1708_170816

-- Let's define the conditions
variables (F N : ℝ)
axiom condition1 : (1 / 4) * (1 / 3) * F * N = 15
axiom condition2 : 0.4 * N = 180

-- theorem to prove the fraction F
theorem find_fraction : F = 2 / 5 :=
by
  -- proof steps would go here, but we're adding sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_find_fraction_l1708_170816


namespace NUMINAMATH_GPT_students_in_each_group_l1708_170839

theorem students_in_each_group (num_boys : ℕ) (num_girls : ℕ) (num_groups : ℕ) 
  (h_boys : num_boys = 26) (h_girls : num_girls = 46) (h_groups : num_groups = 8) : 
  (num_boys + num_girls) / num_groups = 9 := 
by 
  sorry

end NUMINAMATH_GPT_students_in_each_group_l1708_170839


namespace NUMINAMATH_GPT_solve_congruence_l1708_170829

-- Define the condition and residue modulo 47
def residue_modulo (a b n : ℕ) : Prop := (a ≡ b [MOD n])

-- The main theorem to be proved
theorem solve_congruence (m : ℕ) (h : residue_modulo (13 * m) 9 47) : residue_modulo m 26 47 :=
sorry

end NUMINAMATH_GPT_solve_congruence_l1708_170829


namespace NUMINAMATH_GPT_total_area_of_pyramid_faces_l1708_170894

theorem total_area_of_pyramid_faces (base_edge lateral_edge : ℝ) (h : base_edge = 8) (k : lateral_edge = 5) : 
  4 * (1 / 2 * base_edge * 3) = 48 :=
by
  -- Base edge of the pyramid
  let b := base_edge
  -- Lateral edge of the pyramid
  let l := lateral_edge
  -- Half of the base
  let half_b := 4
  -- Height of the triangular face using Pythagorean theorem
  let h := 3
  -- Total area of four triangular faces
  have triangular_face_area : 1 / 2 * base_edge * h = 12 := sorry
  have total_area_of_faces : 4 * (1 / 2 * base_edge * h) = 48 := sorry
  exact total_area_of_faces

end NUMINAMATH_GPT_total_area_of_pyramid_faces_l1708_170894


namespace NUMINAMATH_GPT_cos_fourth_power_sum_l1708_170879

open Real

theorem cos_fourth_power_sum :
  (cos (0 : ℝ))^4 + (cos (π / 6))^4 + (cos (π / 3))^4 + (cos (π / 2))^4 +
  (cos (2 * π / 3))^4 + (cos (5 * π / 6))^4 + (cos π)^4 = 13 / 4 := 
by
  sorry

end NUMINAMATH_GPT_cos_fourth_power_sum_l1708_170879


namespace NUMINAMATH_GPT_count_three_digit_numbers_between_l1708_170884

theorem count_three_digit_numbers_between 
  (a b : ℕ) 
  (ha : a = 137) 
  (hb : b = 285) : 
  ∃ n, n = (b - a - 1) + 1 := 
sorry

end NUMINAMATH_GPT_count_three_digit_numbers_between_l1708_170884


namespace NUMINAMATH_GPT_min_length_BC_l1708_170831

theorem min_length_BC (A B C D : Type) (AB AC DC BD BC : ℝ) :
  AB = 8 → AC = 15 → DC = 10 → BD = 25 → (BC > AC - AB) ∧ (BC > BD - DC) → BC ≥ 15 :=
by
  intros hAB hAC hDC hBD hIneq
  sorry

end NUMINAMATH_GPT_min_length_BC_l1708_170831


namespace NUMINAMATH_GPT_smallest_enclosing_sphere_radius_l1708_170893

-- Define the radius of each small sphere and the center set
def radius (r : ℝ) : Prop := r = 2

def center_set (C : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ c ∈ C, ∃ x y z : ℝ, 
    (x = 2 ∨ x = -2) ∧ 
    (y = 2 ∨ y = -2) ∧ 
    (z = 2 ∨ z = -2) ∧
    (c = (x, y, z))

-- Prove the radius of the smallest enclosing sphere is 2√3 + 2
theorem smallest_enclosing_sphere_radius (r : ℝ) (C : Set (ℝ × ℝ × ℝ)) 
  (h_radius : radius r) (h_center_set : center_set C) :
  ∃ R : ℝ, R = 2 * Real.sqrt 3 + 2 :=
sorry

end NUMINAMATH_GPT_smallest_enclosing_sphere_radius_l1708_170893


namespace NUMINAMATH_GPT_simplify_expression_l1708_170858

variable (d : ℤ)

theorem simplify_expression :
  (5 + 4 * d) / 9 - 3 + 1 / 3 = (4 * d - 19) / 9 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1708_170858


namespace NUMINAMATH_GPT_complementary_angles_l1708_170882

theorem complementary_angles (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 :=
by 
  sorry

end NUMINAMATH_GPT_complementary_angles_l1708_170882


namespace NUMINAMATH_GPT_three_digit_numbers_square_ends_in_1001_l1708_170841

theorem three_digit_numbers_square_ends_in_1001 (n : ℕ) :
  100 ≤ n ∧ n < 1000 ∧ n^2 % 10000 = 1001 → n = 501 ∨ n = 749 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_three_digit_numbers_square_ends_in_1001_l1708_170841


namespace NUMINAMATH_GPT_two_point_three_five_as_fraction_l1708_170876

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end NUMINAMATH_GPT_two_point_three_five_as_fraction_l1708_170876


namespace NUMINAMATH_GPT_range_of_a_for_root_l1708_170878

noncomputable def has_root_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ (a * x^2 + 2 * x - 1) = 0

theorem range_of_a_for_root :
  { a : ℝ | has_root_in_interval a } = { a : ℝ | -1 ≤ a } :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_for_root_l1708_170878


namespace NUMINAMATH_GPT_seashells_total_l1708_170852

theorem seashells_total {sally tom jessica : ℕ} (h₁ : sally = 9) (h₂ : tom = 7) (h₃ : jessica = 5) : sally + tom + jessica = 21 := by
  sorry

end NUMINAMATH_GPT_seashells_total_l1708_170852


namespace NUMINAMATH_GPT_min_value_inequality_l1708_170844

theorem min_value_inequality (a b : ℝ) (h : a * b = 1) : 4 * a^2 + 9 * b^2 ≥ 12 :=
by sorry

end NUMINAMATH_GPT_min_value_inequality_l1708_170844


namespace NUMINAMATH_GPT_base_b_of_256_has_4_digits_l1708_170888

theorem base_b_of_256_has_4_digits : ∃ (b : ℕ), b^3 ≤ 256 ∧ 256 < b^4 ∧ b = 5 :=
by
  sorry

end NUMINAMATH_GPT_base_b_of_256_has_4_digits_l1708_170888


namespace NUMINAMATH_GPT_time_spent_cutting_hair_l1708_170819

theorem time_spent_cutting_hair :
  let women's_time := 50
  let men's_time := 15
  let children's_time := 25
  let women's_haircuts := 3
  let men's_haircuts := 2
  let children's_haircuts := 3
  women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255 :=
by
  -- Definitions
  let women's_time       := 50
  let men's_time         := 15
  let children's_time    := 25
  let women's_haircuts   := 3
  let men's_haircuts     := 2
  let children's_haircuts := 3
  
  show women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255
  sorry

end NUMINAMATH_GPT_time_spent_cutting_hair_l1708_170819


namespace NUMINAMATH_GPT_terminal_velocity_steady_speed_l1708_170846

variable (g : ℝ) (t₁ t₂ : ℝ) (a₀ a₁ : ℝ) (v_terminal : ℝ)

-- Conditions
def acceleration_due_to_gravity := g = 10 -- m/s²
def initial_time := t₁ = 0 -- s
def intermediate_time := t₂ = 2 -- s
def initial_acceleration := a₀ = 50 -- m/s²
def final_acceleration := a₁ = 10 -- m/s²

-- Question: Prove the terminal velocity
theorem terminal_velocity_steady_speed 
  (h_g : acceleration_due_to_gravity g)
  (h_t1 : initial_time t₁)
  (h_t2 : intermediate_time t₂)
  (h_a0 : initial_acceleration a₀)
  (h_a1 : final_acceleration a₁) :
  v_terminal = 25 :=
  sorry

end NUMINAMATH_GPT_terminal_velocity_steady_speed_l1708_170846


namespace NUMINAMATH_GPT_range_of_a_l1708_170803

/--
Given the parabola \(x^2 = y\), points \(A\) and \(B\) are on the parabola and located on both sides of the y-axis,
and the line \(AB\) intersects the y-axis at point \((0, a)\). If \(\angle AOB\) is an acute angle (where \(O\) is the origin),
then the real number \(a\) is greater than 1.
-/
theorem range_of_a (a : ℝ) (x1 x2 : ℝ) : (x1^2 = x2^2) → (x1 * x2 = -a) → ((-a + a^2) > 0) → (1 < a) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l1708_170803


namespace NUMINAMATH_GPT_find_value_of_f2_sub_f3_l1708_170880

variable (f : ℝ → ℝ)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem find_value_of_f2_sub_f3 (h_odd : is_odd_function f) (h_sum : f (-2) + f 0 + f 3 = 2) :
  f 2 - f 3 = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_f2_sub_f3_l1708_170880


namespace NUMINAMATH_GPT_max_value_of_a_exists_max_value_of_a_l1708_170899

theorem max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  a ≤ (Real.sqrt 6 / 3) :=
sorry

theorem exists_max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  ∃ a_max: ℝ, a_max = (Real.sqrt 6 / 3) ∧ (∀ a', (a' ≤ a_max)) :=
sorry

end NUMINAMATH_GPT_max_value_of_a_exists_max_value_of_a_l1708_170899


namespace NUMINAMATH_GPT_circle_tangent_proof_l1708_170857

noncomputable def circle_tangent_range : Set ℝ :=
  { k : ℝ | k > 0 ∧ ((3 - 2 * k)^2 + (1 - k)^2 > k) }

theorem circle_tangent_proof :
  ∀ k > 0, ((3 - 2 * k)^2 + (1 - k)^2 > k) ↔ (k ∈ (Set.Ioo 0 1 ∪ Set.Ioi 2)) :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_proof_l1708_170857


namespace NUMINAMATH_GPT_option_C_correct_inequality_l1708_170813

theorem option_C_correct_inequality (x : ℝ) : 
  (1 / ((x + 1) * (x - 1)) ≤ 0) ↔ (-1 < x ∧ x < 1) :=
sorry

end NUMINAMATH_GPT_option_C_correct_inequality_l1708_170813


namespace NUMINAMATH_GPT_parity_of_function_parity_neither_odd_nor_even_l1708_170840

def f (x p : ℝ) : ℝ := x * |x| + p * x^2

theorem parity_of_function (p : ℝ) :
  (∀ x : ℝ, f x p = - f (-x) p) ↔ p = 0 :=
by
  sorry

theorem parity_neither_odd_nor_even (p : ℝ) :
  (∀ x : ℝ, f x p ≠ f (-x) p) ∧ (∀ x : ℝ, f x p ≠ - f (-x) p) ↔ p ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_parity_of_function_parity_neither_odd_nor_even_l1708_170840


namespace NUMINAMATH_GPT_proof_problem_l1708_170837

theorem proof_problem (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a > b) (h5 : a^2 - a * c + b * c = 7) :
  a - c = 0 ∨ a - c = 1 :=
 sorry

end NUMINAMATH_GPT_proof_problem_l1708_170837


namespace NUMINAMATH_GPT_sum_proper_divisors_81_l1708_170817

theorem sum_proper_divisors_81 : (3^0 + 3^1 + 3^2 + 3^3) = 40 :=
by
  sorry

end NUMINAMATH_GPT_sum_proper_divisors_81_l1708_170817


namespace NUMINAMATH_GPT_min_period_and_sym_center_l1708_170864

open Real

noncomputable def func (x α β : ℝ) : ℝ :=
  sin (x - α) * cos (x - β)

theorem min_period_and_sym_center (α β : ℝ) :
  (∀ x, func (x + π) α β = func x α β) ∧ (func α 0 β = 0) :=
by
  sorry

end NUMINAMATH_GPT_min_period_and_sym_center_l1708_170864


namespace NUMINAMATH_GPT_largest_digit_divisible_by_6_l1708_170874

def divisibleBy2 (N : ℕ) : Prop :=
  ∃ k, N = 2 * k

def divisibleBy3 (N : ℕ) : Prop :=
  ∃ k, N = 3 * k

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ divisibleBy2 N ∧ divisibleBy3 (26 + N) ∧ (∀ M : ℕ, M ≤ 9 ∧ divisibleBy2 M ∧ divisibleBy3 (26 + M) → M ≤ N) ∧ N = 4 :=
by
  sorry

end NUMINAMATH_GPT_largest_digit_divisible_by_6_l1708_170874
