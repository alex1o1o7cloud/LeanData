import Mathlib

namespace coconut_grove_l738_73822

theorem coconut_grove (x : ℕ) :
  (40 * (x + 2) + 120 * x + 180 * (x - 2) = 100 * 3 * x) → 
  x = 7 := by
  sorry

end coconut_grove_l738_73822


namespace mean_age_of_seven_friends_l738_73854

theorem mean_age_of_seven_friends 
  (mean_age_group1: ℕ)
  (mean_age_group2: ℕ)
  (n1: ℕ)
  (n2: ℕ)
  (total_friends: ℕ) :
  mean_age_group1 = 147 → 
  mean_age_group2 = 161 →
  n1 = 3 → 
  n2 = 4 →
  total_friends = 7 →
  (mean_age_group1 * n1 + mean_age_group2 * n2) / total_friends = 155 := by
  sorry

end mean_age_of_seven_friends_l738_73854


namespace minimum_distance_between_tracks_l738_73867

-- Problem statement as Lean definitions and theorem to prove
noncomputable def rational_man_track (t : ℝ) : ℝ × ℝ :=
  (Real.cos t, Real.sin t)

noncomputable def hyperbolic_man_track (t : ℝ) : ℝ × ℝ :=
  (-1 + 3 * Real.cos (t / 2), 5 * Real.sin (t / 2))

noncomputable def circle_eq := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

noncomputable def ellipse_eq := {p : ℝ × ℝ | (p.1 + 1)^2 / 9 + p.2^2 / 25 = 1}

theorem minimum_distance_between_tracks : 
  ∃ A ∈ circle_eq, ∃ B ∈ ellipse_eq, dist A B = Real.sqrt 14 - 1 := 
sorry

end minimum_distance_between_tracks_l738_73867


namespace max_ab_l738_73849

noncomputable def f (a x : ℝ) : ℝ := -a * Real.log x + (a + 1) * x - (1/2) * x^2

theorem max_ab (a b : ℝ) (h₁ : 0 < a)
  (h₂ : ∀ x, f a x ≥ - (1/2) * x^2 + a * x + b) : 
  ab ≤ ((Real.exp 1) / 2) :=
sorry

end max_ab_l738_73849


namespace rational_solutions_count_l738_73868

theorem rational_solutions_count :
  ∃ (sols : Finset (ℚ × ℚ × ℚ)), 
    (∀ (x y z : ℚ), (x + y + z = 0) ∧ (x * y * z + z = 0) ∧ (x * y + y * z + x * z + y = 0) ↔ (x, y, z) ∈ sols) ∧
    sols.card = 3 :=
by
  sorry

end rational_solutions_count_l738_73868


namespace mean_of_set_l738_73826

theorem mean_of_set (m : ℝ) (h : m + 7 = 12) :
  (m + (m + 6) + (m + 7) + (m + 11) + (m + 18)) / 5 = 13.4 :=
by sorry

end mean_of_set_l738_73826


namespace expression_equivalence_l738_73886

def algebraicExpression : String := "5 - 4a"
def wordExpression : String := "the difference of 5 and 4 times a"

theorem expression_equivalence : algebraicExpression = wordExpression := 
sorry

end expression_equivalence_l738_73886


namespace problem_statement_l738_73883

-- Define that f is an even function and decreasing on (0, +∞)
variables {f : ℝ → ℝ}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f (x)

def is_decreasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f y < f x

-- Main statement: Prove the specific inequality under the given conditions
theorem problem_statement (f_even : is_even_function f) (f_decreasing : is_decreasing_on_pos f) :
  f (1/2) > f (-2/3) ∧ f (-2/3) > f (3/4) :=
by
  sorry

end problem_statement_l738_73883


namespace Mina_digits_l738_73830

theorem Mina_digits (Carlos Sam Mina : ℕ) 
  (h1 : Sam = Carlos + 6) 
  (h2 : Mina = 6 * Carlos) 
  (h3 : Sam = 10) : 
  Mina = 24 := 
sorry

end Mina_digits_l738_73830


namespace parker_shorter_than_daisy_l738_73807

noncomputable def solve_height_difference : Nat :=
  let R := 60
  let D := R + 8
  let avg := 64
  ((3 * avg) - (D + R))

theorem parker_shorter_than_daisy :
  let P := solve_height_difference
  D - P = 4 := by
  sorry

end parker_shorter_than_daisy_l738_73807


namespace smallest_total_students_l738_73893

theorem smallest_total_students (n : ℕ) (h1 : 25 * n % 100 = 0)
  (h2 : 10 * n % 4 = 0)
  (h3 : ∃ (y z : ℕ), y = 3 * z / 2 ∧ (y + z - n / 40 = n / 4)) :
  ∃ k : ℕ, n = 200 * k :=
by
  sorry

end smallest_total_students_l738_73893


namespace rich_avg_time_per_mile_l738_73860

-- Define the total time in minutes and the total distance
def total_minutes : ℕ := 517
def total_miles : ℕ := 50

-- Define a function to calculate the average time per mile
def avg_time_per_mile (total_time : ℕ) (distance : ℕ) : ℚ :=
  total_time / distance

-- Theorem statement
theorem rich_avg_time_per_mile :
  avg_time_per_mile total_minutes total_miles = 10.34 :=
by
  -- Proof steps go here
  sorry

end rich_avg_time_per_mile_l738_73860


namespace least_pos_integer_to_yield_multiple_of_5_l738_73817

theorem least_pos_integer_to_yield_multiple_of_5 (n : ℕ) (h : n > 0) :
  ((567 + n) % 5 = 0) ↔ (n = 3) :=
by {
  sorry
}

end least_pos_integer_to_yield_multiple_of_5_l738_73817


namespace remaining_batches_l738_73804

def flour_per_batch : ℕ := 2
def batches_baked : ℕ := 3
def initial_flour : ℕ := 20

theorem remaining_batches : (initial_flour - flour_per_batch * batches_baked) / flour_per_batch = 7 := by
  sorry

end remaining_batches_l738_73804


namespace cyclist_speed_ratio_l738_73850

variables (k r t v1 v2 : ℝ)
variable (h1 : v1 = 2 * v2) -- Condition 5

-- When traveling in the same direction, relative speed is v1 - v2 and they cover 2k miles in 3r hours
variable (h2 : 2 * k = (v1 - v2) * 3 * r)

-- When traveling in opposite directions, relative speed is v1 + v2 and they pass each other in 2t hours
variable (h3 : 2 * k = (v1 + v2) * 2 * t)

theorem cyclist_speed_ratio (h1 : v1 = 2 * v2) (h2 : 2 * k = (v1 - v2) * 3 * r) (h3 : 2 * k = (v1 + v2) * 2 * t) :
  v1 / v2 = 2 :=
sorry

end cyclist_speed_ratio_l738_73850


namespace map_distance_to_actual_distance_l738_73853

theorem map_distance_to_actual_distance
  (map_distance : ℝ)
  (scale_inches : ℝ)
  (scale_miles : ℝ)
  (actual_distance : ℝ)
  (h_scale : scale_inches = 0.5)
  (h_scale_miles : scale_miles = 10)
  (h_map_distance : map_distance = 20) :
  actual_distance = 400 :=
by
  sorry

end map_distance_to_actual_distance_l738_73853


namespace nearest_integer_power_l738_73818

noncomputable def power_expression := (3 + Real.sqrt 2)^6

theorem nearest_integer_power :
  Int.floor power_expression = 7414 :=
sorry

end nearest_integer_power_l738_73818


namespace trajectory_of_M_l738_73873

theorem trajectory_of_M (x y t : ℝ) (M P F : ℝ × ℝ)
    (hF : F = (1, 0))
    (hP : P = (1/4 * t^2, t))
    (hFP : (P.1 - F.1, P.2 - F.2) = (1/4 * t^2 - 1, t))
    (hFM : (M.1 - F.1, M.2 - F.2) = (x - 1, y))
    (hFP_FM : (P.1 - F.1, P.2 - F.2) = (2 * (M.1 - F.1), 2 * (M.2 - F.2))) :
  y^2 = 2 * x - 1 :=
by
  sorry

end trajectory_of_M_l738_73873


namespace compare_neg_two_cubed_l738_73824

-- Define the expressions
def neg_two_cubed : ℤ := (-2) ^ 3
def neg_two_cubed_alt : ℤ := -(2 ^ 3)

-- Statement of the problem
theorem compare_neg_two_cubed : neg_two_cubed = neg_two_cubed_alt :=
by
  sorry

end compare_neg_two_cubed_l738_73824


namespace find_student_ticket_price_l738_73890

variable (S : ℝ)
variable (student_tickets non_student_tickets total_tickets : ℕ)
variable (non_student_ticket_price total_revenue : ℝ)

theorem find_student_ticket_price 
  (h1 : student_tickets = 90)
  (h2 : non_student_tickets = 60)
  (h3 : total_tickets = student_tickets + non_student_tickets)
  (h4 : non_student_ticket_price = 8)
  (h5 : total_revenue = 930)
  (h6 : 90 * S + 60 * non_student_ticket_price = total_revenue) : 
  S = 5 := 
sorry

end find_student_ticket_price_l738_73890


namespace divide_circle_three_equal_areas_l738_73837

theorem divide_circle_three_equal_areas (OA : ℝ) (r1 r2 : ℝ) 
  (hr1 : r1 = (OA * Real.sqrt 3) / 3) 
  (hr2 : r2 = (OA * Real.sqrt 6) / 3) : 
  ∀ (r : ℝ), r = OA → 
  (∀ (A1 A2 A3 : ℝ), A1 = π * r1 ^ 2 ∧ A2 = π * (r2 ^ 2 - r1 ^ 2) ∧ A3 = π * (r ^ 2 - r2 ^ 2) →
  A1 = A2 ∧ A2 = A3) :=
by
  sorry

end divide_circle_three_equal_areas_l738_73837


namespace find_x_l738_73884

variables {a b : EuclideanSpace ℝ (Fin 2)} {x : ℝ}

theorem find_x (h1 : ‖a + b‖ = 1) (h2 : ‖a - b‖ = x) (h3 : inner a b = -(3 / 8) * x) : x = 2 ∨ x = -(1 / 2) :=
sorry

end find_x_l738_73884


namespace find_matrix_N_l738_73888

-- Define the given matrix equation
def condition (N : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  N ^ 3 - 3 * N ^ 2 + 4 * N = ![![8, 16], ![4, 8]]

-- State the theorem
theorem find_matrix_N (N : Matrix (Fin 2) (Fin 2) ℝ) (h : condition N) :
  N = ![![2, 4], ![1, 2]] :=
sorry

end find_matrix_N_l738_73888


namespace theta_range_l738_73872

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem theta_range (k : ℤ) (θ : ℝ) : 
  (2 * ↑k * π - 5 * π / 6 < θ ∧ θ < 2 * ↑k * π - π / 6) →
  (f (1 / (Real.sin θ)) + f (Real.cos (2 * θ)) < f π - f (1 / π)) :=
by
  intros h
  sorry

end theta_range_l738_73872


namespace sum_of_powers_of_4_l738_73895

theorem sum_of_powers_of_4 : 4^0 + 4^1 + 4^2 + 4^3 = 85 :=
by
  sorry

end sum_of_powers_of_4_l738_73895


namespace weight_cut_percentage_unknown_l738_73838

-- Define the initial conditions
def original_speed : ℝ := 150
def new_speed : ℝ := 205
def increase_supercharge : ℝ := original_speed * 0.3
def speed_after_supercharge : ℝ := original_speed + increase_supercharge
def increase_weight_cut : ℝ := new_speed - speed_after_supercharge

-- Theorem statement
theorem weight_cut_percentage_unknown : 
  (original_speed = 150) →
  (new_speed = 205) →
  (increase_supercharge = 150 * 0.3) →
  (speed_after_supercharge = 150 + increase_supercharge) →
  (increase_weight_cut = 205 - speed_after_supercharge) →
  increase_weight_cut = 10 →
  sorry := 
by
  intros h_orig h_new h_inc_scharge h_speed_scharge h_inc_weight h_inc_10
  sorry

end weight_cut_percentage_unknown_l738_73838


namespace john_total_time_spent_l738_73814

-- Define conditions
def num_pictures : ℕ := 10
def draw_time_per_picture : ℝ := 2
def color_time_reduction : ℝ := 0.3

-- Define the actual color time per picture
def color_time_per_picture : ℝ := draw_time_per_picture * (1 - color_time_reduction)

-- Define the total time per picture
def total_time_per_picture : ℝ := draw_time_per_picture + color_time_per_picture

-- Define the total time for all pictures
def total_time_for_all_pictures : ℝ := total_time_per_picture * num_pictures

-- The theorem we need to prove
theorem john_total_time_spent : total_time_for_all_pictures = 34 :=
by
sorry

end john_total_time_spent_l738_73814


namespace water_breaks_vs_sitting_breaks_l738_73844

theorem water_breaks_vs_sitting_breaks :
  (240 / 20) - (240 / 120) = 10 := by
  sorry

end water_breaks_vs_sitting_breaks_l738_73844


namespace mabel_total_tomatoes_l738_73861

theorem mabel_total_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 :=
by
  sorry

end mabel_total_tomatoes_l738_73861


namespace gcf_45_75_90_l738_73839

-- Definitions as conditions
def number1 : Nat := 45
def number2 : Nat := 75
def number3 : Nat := 90

def factors_45 : Nat × Nat := (3, 2) -- represents 3^2 * 5^1 {prime factor 3, prime factor 5}
def factors_75 : Nat × Nat := (5, 1) -- represents 3^1 * 5^2 {prime factor 3, prime factor 5}
def factors_90 : Nat × Nat := (3, 2) -- represents 2^1 * 3^2 * 5^1 {prime factor 3, prime factor 5}

-- Theorems to be proved
theorem gcf_45_75_90 : Nat.gcd (Nat.gcd number1 number2) number3 = 15 :=
by {
  -- This is here as placeholder for actual proof
  sorry
}

end gcf_45_75_90_l738_73839


namespace arithmetic_mean_of_q_and_r_l738_73855

theorem arithmetic_mean_of_q_and_r (p q r : ℝ) 
  (h₁: (p + q) / 2 = 10) 
  (h₂: r - p = 20) : 
  (q + r) / 2 = 20 :=
sorry

end arithmetic_mean_of_q_and_r_l738_73855


namespace parallel_vectors_implies_value_of_t_l738_73825

theorem parallel_vectors_implies_value_of_t (t : ℝ) :
  let a := (1, t)
  let b := (t, 9)
  (1 * 9 - t^2 = 0) → (t = 3 ∨ t = -3) := 
by sorry

end parallel_vectors_implies_value_of_t_l738_73825


namespace union_sets_l738_73815

open Set

variable {α : Type*}

def A : Set ℝ := {x | -2 < x ∧ x < 2}

def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = 2^x}

theorem union_sets : A ∪ B = {z | -2 < z ∧ z < 4} :=
by sorry

end union_sets_l738_73815


namespace find_A_coordinates_l738_73894

-- Given conditions
variable (B : (ℝ × ℝ)) (hB1 : B = (1, 2))

-- Definitions to translate problem conditions into Lean
def symmetric_y (P B : ℝ × ℝ) : Prop :=
  P.1 = -B.1 ∧ P.2 = B.2

def symmetric_x (A P : ℝ × ℝ) : Prop :=
  A.1 = P.1 ∧ A.2 = -P.2

-- Theorem statement
theorem find_A_coordinates (A P B : ℝ × ℝ) (hB1 : B = (1, 2))
    (h_symm_y: symmetric_y P B) (h_symm_x: symmetric_x A P) : 
    A = (-1, -2) :=
by
  sorry

end find_A_coordinates_l738_73894


namespace remainder_when_divided_by_x_plus_2_l738_73811

def q (x D E F : ℝ) : ℝ := D*x^4 + E*x^2 + F*x - 2

theorem remainder_when_divided_by_x_plus_2 (D E F : ℝ) (h : q 2 D E F = 14) : q (-2) D E F = -18 := 
by 
     sorry

end remainder_when_divided_by_x_plus_2_l738_73811


namespace domain_of_f_monotonicity_of_f_inequality_solution_l738_73870

open Real

noncomputable def f (x : ℝ) : ℝ := log ((1 - x) / (1 + x))

theorem domain_of_f :
  ∀ x, -1 < x ∧ x < 1 → ∃ y, y = f x :=
by
  intro x h
  use log ((1 - x) / (1 + x))
  simp [f]

theorem monotonicity_of_f :
  ∀ x y, -1 < x ∧ x < 1 → -1 < y ∧ y < 1 → x < y → f x > f y :=
sorry

theorem inequality_solution :
  ∀ x, f (2 * x - 1) < 0 ↔ (1 / 2 < x ∧ x < 1) :=
sorry

end domain_of_f_monotonicity_of_f_inequality_solution_l738_73870


namespace find_hypotenuse_of_right_angle_triangle_l738_73810

theorem find_hypotenuse_of_right_angle_triangle
  (PR : ℝ) (angle_QPR : ℝ)
  (h1 : PR = 16)
  (h2 : angle_QPR = Real.pi / 4) :
  ∃ PQ : ℝ, PQ = 16 * Real.sqrt 2 :=
by
  sorry

end find_hypotenuse_of_right_angle_triangle_l738_73810


namespace students_not_in_same_column_or_row_l738_73847

-- Define the positions of student A and student B as conditions
structure Position where
  row : Nat
  col : Nat

-- Student A's position is in the 3rd row and 6th column
def StudentA : Position := {row := 3, col := 6}

-- Student B's position is described in a relative manner in terms of columns and rows
def StudentB : Position := {row := 6, col := 3}

-- Formalize the proof statement
theorem students_not_in_same_column_or_row :
  StudentA.row ≠ StudentB.row ∧ StudentA.col ≠ StudentB.col :=
by {
  sorry
}

end students_not_in_same_column_or_row_l738_73847


namespace Lisa_goal_achievable_l738_73833

open Nat

theorem Lisa_goal_achievable :
  ∀ (total_quizzes quizzes_with_A goal_percentage : ℕ),
  total_quizzes = 60 →
  quizzes_with_A = 25 →
  goal_percentage = 85 →
  (quizzes_with_A < goal_percentage * total_quizzes / 100) →
  (∃ remaining_quizzes, goal_percentage * total_quizzes / 100 - quizzes_with_A > remaining_quizzes) :=
by
  intros total_quizzes quizzes_with_A goal_percentage h_total h_A h_goal h_lack
  let needed_quizzes := goal_percentage * total_quizzes / 100
  let remaining_quizzes := total_quizzes - 35
  have h_needed := needed_quizzes - quizzes_with_A
  use remaining_quizzes
  sorry

end Lisa_goal_achievable_l738_73833


namespace max_value_of_f_l738_73816

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem max_value_of_f : ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 2 := 
by
  sorry

end max_value_of_f_l738_73816


namespace sheila_saving_years_l738_73879

theorem sheila_saving_years 
  (initial_amount : ℝ) 
  (monthly_saving : ℝ) 
  (secret_addition : ℝ) 
  (final_amount : ℝ) 
  (years : ℝ) : 
  initial_amount = 3000 ∧ 
  monthly_saving = 276 ∧ 
  secret_addition = 7000 ∧ 
  final_amount = 23248 → 
  years = 4 := 
sorry

end sheila_saving_years_l738_73879


namespace solve_inequality_l738_73808

theorem solve_inequality (x : ℝ) (h : 1 / (x - 1) < -1) : 0 < x ∧ x < 1 :=
sorry

end solve_inequality_l738_73808


namespace min_value_l738_73899

theorem min_value (x : ℝ) (h : 0 < x) : x + 2 / (2 * x + 1) - 3 / 2 ≥ 0 :=
sorry

end min_value_l738_73899


namespace exotic_meat_original_price_l738_73806

theorem exotic_meat_original_price (y : ℝ) :
  (0.75 * (y / 4) = 4.5) → y = 96 :=
by
  intro h
  sorry

end exotic_meat_original_price_l738_73806


namespace golden_section_point_l738_73897

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem golden_section_point (AB AP PB : ℝ)
  (h1 : AP + PB = AB)
  (h2 : AB = 5)
  (h3 : (AB / AP) = (AP / PB))
  (h4 : AP > PB) :
  AP = (5 * Real.sqrt 5 - 5) / 2 :=
by sorry

end golden_section_point_l738_73897


namespace harvest_duration_l738_73889

theorem harvest_duration (total_earnings earnings_per_week : ℕ) (h1 : total_earnings = 1216) (h2 : earnings_per_week = 16) :
  total_earnings / earnings_per_week = 76 :=
by
  sorry

end harvest_duration_l738_73889


namespace range_of_a_l738_73859

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → x - Real.log x - a > 0) → a < 1 :=
sorry

end range_of_a_l738_73859


namespace product_two_digit_numbers_l738_73887

theorem product_two_digit_numbers (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 777) : (a = 21 ∧ b = 37) ∨ (a = 37 ∧ b = 21) := 
  sorry

end product_two_digit_numbers_l738_73887


namespace find_prime_p_l738_73885

noncomputable def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem find_prime_p (p : ℕ) (hp : p.Prime) (hsquare : isPerfectSquare (5^p + 12^p)) : p = 2 := 
sorry

end find_prime_p_l738_73885


namespace length_of_BC_is_7_l738_73843

noncomputable def triangle_length_BC (a b c : ℝ) (A : ℝ) (S : ℝ) (P : ℝ) : Prop :=
  (P = a + b + c) ∧ (P = 20) ∧ (S = 1 / 2 * b * c * Real.sin A) ∧ (S = 10 * Real.sqrt 3) ∧ (A = Real.pi / 3) ∧ (b * c = 20)

theorem length_of_BC_is_7 : ∃ a b c, triangle_length_BC a b c (Real.pi / 3) (10 * Real.sqrt 3) 20 ∧ a = 7 := 
by
  -- proof omitted
  sorry

end length_of_BC_is_7_l738_73843


namespace four_digit_number_exists_l738_73803

theorem four_digit_number_exists :
  ∃ (x1 x2 y1 y2 : ℕ), (x1 > 0) ∧ (x2 > 0) ∧ (y1 > 0) ∧ (y2 > 0) ∧
                       (x2 * y2 - x1 * y1 = 67) ∧ (x2 > y2) ∧ (x1 < y1) ∧
                       (x1 * 10^3 + x2 * 10^2 + y2 * 10 + y1 = 1985) := sorry

end four_digit_number_exists_l738_73803


namespace clinton_earnings_correct_l738_73877

-- Define the conditions as variables/constants
def num_students_Arlington : ℕ := 8
def days_Arlington : ℕ := 4

def num_students_Bradford : ℕ := 6
def days_Bradford : ℕ := 7

def num_students_Clinton : ℕ := 7
def days_Clinton : ℕ := 8

def total_compensation : ℝ := 1456

noncomputable def total_student_days : ℕ :=
  num_students_Arlington * days_Arlington + num_students_Bradford * days_Bradford + num_students_Clinton * days_Clinton

noncomputable def daily_wage : ℝ :=
  total_compensation / total_student_days

noncomputable def earnings_Clinton : ℝ :=
  daily_wage * (num_students_Clinton * days_Clinton)

theorem clinton_earnings_correct : earnings_Clinton = 627.2 := by 
  sorry

end clinton_earnings_correct_l738_73877


namespace min_value_fraction_l738_73858

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) : 
  ( (x + 1) * (y + 1) / (x * y) ) >= 8 + 4 * Real.sqrt 3 :=
sorry

end min_value_fraction_l738_73858


namespace num_points_common_to_graphs_l738_73896

theorem num_points_common_to_graphs :
  (∃ (x y : ℝ), (2 * x - y + 3 = 0 ∧ x + y - 3 = 0)) ∧
  (∃ (x y : ℝ), (2 * x - y + 3 = 0 ∧ 3 * x - 4 * y + 8 = 0)) ∧
  (∃ (x y : ℝ), (4 * x + y - 5 = 0 ∧ x + y - 3 = 0)) ∧
  (∃ (x y : ℝ), (4 * x + y - 5 = 0 ∧ 3 * x - 4 * y + 8 = 0)) ∧
  ∀ (x y : ℝ), ((2 * x - y + 3 = 0 ∨ 4 * x + y - 5 = 0) ∧ (x + y - 3 = 0 ∨ 3 * x - 4 * y + 8 = 0)) →
  ∃ (p1 p2 p3 p4 : ℝ × ℝ), 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 :=
sorry

end num_points_common_to_graphs_l738_73896


namespace lakeside_fitness_center_ratio_l738_73881

theorem lakeside_fitness_center_ratio (f m c : ℕ)
  (h_avg_age : (35 * f + 30 * m + 10 * c) / (f + m + c) = 25) :
  f = 3 * (m / 6) ∧ f = 3 * (c / 2) :=
by
  sorry

end lakeside_fitness_center_ratio_l738_73881


namespace total_students_class_l738_73820

theorem total_students_class (S R : ℕ) 
  (h1 : 2 + 12 + 10 + R = S)
  (h2 : (0 * 2) + (1 * 12) + (2 * 10) + (3 * R) = 2 * S) :
  S = 40 := by
  sorry

end total_students_class_l738_73820


namespace unique_root_of_quadratic_eq_l738_73834

theorem unique_root_of_quadratic_eq (a b c : ℝ) (d : ℝ) 
  (h_seq : b = a - d ∧ c = a - 2 * d) 
  (h_nonneg : a ≥ b ∧ b ≥ c ∧ c ≥ 0) 
  (h_discriminant : (-(a - d))^2 - 4 * a * (a - 2 * d) = 0) :
  ∃ x : ℝ, (ax^2 - bx + c = 0) ∧ x = 1 / 2 :=
by
  sorry

end unique_root_of_quadratic_eq_l738_73834


namespace x_equals_y_l738_73856

-- Conditions
def x := 2 * 20212021 * 1011 * 202320232023
def y := 43 * 47 * 20232023 * 202220222022

-- Proof statement
theorem x_equals_y : x = y := sorry

end x_equals_y_l738_73856


namespace probability_not_yellow_l738_73846

-- Define the conditions
def red_jelly_beans : Nat := 4
def green_jelly_beans : Nat := 7
def yellow_jelly_beans : Nat := 9
def blue_jelly_beans : Nat := 10

-- Definitions used in the proof problem
def total_jelly_beans : Nat := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans
def non_yellow_jelly_beans : Nat := total_jelly_beans - yellow_jelly_beans

-- Lean statement of the probability problem
theorem probability_not_yellow : 
  (non_yellow_jelly_beans : ℚ) / (total_jelly_beans : ℚ) = 7 / 10 := 
by 
  sorry

end probability_not_yellow_l738_73846


namespace find_range_of_a_l738_73862

def p (a : ℝ) : Prop := 
  a = 0 ∨ (a > 0 ∧ a^2 - 4 * a < 0)

def q (a : ℝ) : Prop := 
  a^2 - 2 * a - 3 < 0

theorem find_range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := 
sorry

end find_range_of_a_l738_73862


namespace exponent_form_l738_73829

theorem exponent_form (y : ℕ) (w : ℕ) (k : ℕ) : w = 3 ^ y → w % 10 = 7 → ∃ (k : ℕ), y = 4 * k + 3 :=
by
  intros h1 h2
  sorry

end exponent_form_l738_73829


namespace log_equality_implies_exp_equality_l738_73800

theorem log_equality_implies_exp_equality (x y z a : ℝ) (h : (x * (y + z - x)) / (Real.log x) = (y * (x + z - y)) / (Real.log y) ∧ (y * (x + z - y)) / (Real.log y) = (z * (x + y - z)) / (Real.log z)) :
  x^y * y^x = z^x * x^z ∧ z^x * x^z = y^z * z^y :=
by
  sorry

end log_equality_implies_exp_equality_l738_73800


namespace solution_set_lg2_l738_73823

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1
axiom f_deriv_lt : ∀ x : ℝ, deriv f x < 1

theorem solution_set_lg2 : { x : ℝ | f (Real.log x ^ 2) < Real.log x ^ 2 } = { x : ℝ | (1/10 : ℝ) < x ∧ x < 10 } :=
by
  sorry

end solution_set_lg2_l738_73823


namespace initial_balloons_correct_l738_73842

-- Define the variables corresponding to the conditions given in the problem
def boy_balloon_count := 3
def girl_balloon_count := 12
def balloons_sold := boy_balloon_count + girl_balloon_count
def balloons_remaining := 21

-- State the theorem asserting the initial number of balloons
theorem initial_balloons_correct :
  balloons_sold + balloons_remaining = 36 := sorry

end initial_balloons_correct_l738_73842


namespace ratio_of_female_to_male_members_l738_73828

theorem ratio_of_female_to_male_members 
  (f m : ℕ) 
  (avg_age_female : ℕ) 
  (avg_age_male : ℕ)
  (avg_age_all : ℕ) 
  (H1 : avg_age_female = 45)
  (H2 : avg_age_male = 25)
  (H3 : avg_age_all = 35)
  (H4 : (f + m) ≠ 0) :
  (45 * f + 25 * m) / (f + m) = 35 → f = m :=
by sorry

end ratio_of_female_to_male_members_l738_73828


namespace sample_size_correct_l738_73851

def sample_size (sum_frequencies : ℕ) (frequency_sum_ratio : ℚ) (S : ℕ) : Prop :=
  sum_frequencies = 20 ∧ frequency_sum_ratio = 0.4 → S = 50

theorem sample_size_correct :
  ∀ (sum_frequencies : ℕ) (frequency_sum_ratio : ℚ),
    sample_size sum_frequencies frequency_sum_ratio 50 :=
by
  intros sum_frequencies frequency_sum_ratio
  sorry

end sample_size_correct_l738_73851


namespace sum_first_n_natural_numbers_l738_73841

theorem sum_first_n_natural_numbers (n : ℕ) (h : (n * (n + 1)) / 2 = 1035) : n = 46 :=
sorry

end sum_first_n_natural_numbers_l738_73841


namespace ratio_monkeys_snakes_l738_73874

def parrots : ℕ := 8
def snakes : ℕ := 3 * parrots
def elephants : ℕ := (parrots + snakes) / 2
def zebras : ℕ := elephants - 3
def monkeys : ℕ := zebras + 35

theorem ratio_monkeys_snakes : (monkeys : ℕ) / (snakes : ℕ) = 2 / 1 :=
by
  sorry

end ratio_monkeys_snakes_l738_73874


namespace erasers_pens_markers_cost_l738_73875

theorem erasers_pens_markers_cost 
  (E P M : ℝ)
  (h₁ : E + 3 * P + 2 * M = 240)
  (h₂ : 2 * E + 4 * M + 5 * P = 440) :
  3 * E + 4 * P + 6 * M = 520 :=
sorry

end erasers_pens_markers_cost_l738_73875


namespace find_b_eq_five_l738_73835

/--
Given points A(4, 2) and B(0, b) in the Cartesian coordinate system,
and the condition that the distances from O (the origin) to B and from B to A are equal,
prove that b = 5.
-/
theorem find_b_eq_five : ∃ b : ℝ, (dist (0, 0) (0, b) = dist (0, b) (4, 2)) ∧ b = 5 :=
by
  sorry

end find_b_eq_five_l738_73835


namespace correct_answer_is_C_l738_73802

def exactly_hits_n_times (n k : ℕ) : Prop :=
  n = k

def hits_no_more_than (n k : ℕ) : Prop :=
  n ≤ k

def hits_at_least (n k : ℕ) : Prop :=
  n ≥ k

def is_mutually_exclusive (P Q : Prop) : Prop :=
  ¬ (P ∧ Q)

def is_non_opposing (P Q : Prop) : Prop :=
  ¬ P ∧ ¬ Q

def events_are_mutually_exclusive_and_non_opposing (n : ℕ) : Prop :=
  let event1 := exactly_hits_n_times 5 3
  let event2 := exactly_hits_n_times 5 4
  is_mutually_exclusive event1 event2 ∧ is_non_opposing event1 event2

theorem correct_answer_is_C : events_are_mutually_exclusive_and_non_opposing 5 :=
by
  sorry

end correct_answer_is_C_l738_73802


namespace final_amount_in_account_l738_73878

noncomputable def initial_deposit : ℝ := 1000
noncomputable def first_year_interest_rate : ℝ := 0.2
noncomputable def first_year_balance : ℝ := initial_deposit * (1 + first_year_interest_rate)
noncomputable def withdrawal_amount : ℝ := first_year_balance / 2
noncomputable def after_withdrawal_balance : ℝ := first_year_balance - withdrawal_amount
noncomputable def second_year_interest_rate : ℝ := 0.15
noncomputable def final_balance : ℝ := after_withdrawal_balance * (1 + second_year_interest_rate)

theorem final_amount_in_account : final_balance = 690 := by
  sorry

end final_amount_in_account_l738_73878


namespace xy_equation_solution_l738_73845

theorem xy_equation_solution (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 11980 / 121 :=
by
  sorry

end xy_equation_solution_l738_73845


namespace rectangle_R2_area_l738_73898

theorem rectangle_R2_area
  (side1_R1 : ℝ) (area_R1 : ℝ) (diag_R2 : ℝ)
  (h_side1_R1 : side1_R1 = 4)
  (h_area_R1 : area_R1 = 32)
  (h_diag_R2 : diag_R2 = 20) :
  ∃ (area_R2 : ℝ), area_R2 = 160 :=
by
  sorry

end rectangle_R2_area_l738_73898


namespace math_problem_l738_73869

theorem math_problem
  (x y : ℝ)
  (h1 : x + y = 5)
  (h2 : x * y = -3)
  : x + (x^3 / y^2) + (y^3 / x^2) + y = 590.5 :=
sorry

end math_problem_l738_73869


namespace largest_divisor_of_five_consecutive_integers_product_correct_l738_73832

noncomputable def largest_divisor_of_five_consecutive_integers_product : ℕ :=
  120

theorem largest_divisor_of_five_consecutive_integers_product_correct :
  ∀ (n : ℕ), (∃ k : ℕ, k = n * (n + 1) * (n + 2) * (n + 3) * (n + 4) ∧ 120 ∣ k) :=
sorry

end largest_divisor_of_five_consecutive_integers_product_correct_l738_73832


namespace area_of_four_triangles_l738_73809

theorem area_of_four_triangles (a b : ℕ) (h1 : 2 * b = 28) (h2 : a + 2 * b = 30) :
    4 * (1 / 2 * a * b) = 56 := by
  sorry

end area_of_four_triangles_l738_73809


namespace price_of_basic_computer_l738_73864

theorem price_of_basic_computer (C P : ℝ) 
    (h1 : C + P = 2500) 
    (h2 : P = (1/8) * (C + 500 + P)) :
    C = 2125 :=
by
  sorry

end price_of_basic_computer_l738_73864


namespace intersection_A_B_union_A_complement_B_subset_C_B_range_l738_73836

def set_A : Set ℝ := { x | 1 ≤ x ∧ x < 6 }
def set_B : Set ℝ := { x | 2 < x ∧ x < 9 }
def set_C (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

theorem intersection_A_B :
  set_A ∩ set_B = { x | 2 < x ∧ x < 6 } :=
sorry

theorem union_A_complement_B :
  set_A ∪ (compl set_B) = { x | x < 6 } ∪ { x | x ≥ 9 } :=
sorry

theorem subset_C_B_range (a : ℝ) :
  (set_C a ⊆ set_B) → (2 ≤ a ∧ a ≤ 8) :=
sorry

end intersection_A_B_union_A_complement_B_subset_C_B_range_l738_73836


namespace erdos_ginzburg_ziv_2047_l738_73840

open Finset

theorem erdos_ginzburg_ziv_2047 (s : Finset ℕ) (h : s.card = 2047) : 
  ∃ t ⊆ s, t.card = 1024 ∧ (t.sum id) % 1024 = 0 :=
sorry

end erdos_ginzburg_ziv_2047_l738_73840


namespace parabola_distance_relation_l738_73880

theorem parabola_distance_relation {n : ℝ} {x₁ x₂ y₁ y₂ : ℝ}
  (h₁ : y₁ = x₁^2 - 4 * x₁ + n)
  (h₂ : y₂ = x₂^2 - 4 * x₂ + n)
  (h : y₁ > y₂) :
  |x₁ - 2| > |x₂ - 2| := 
sorry

end parabola_distance_relation_l738_73880


namespace has_exactly_one_solution_l738_73876

theorem has_exactly_one_solution (a : ℝ) : 
  (∀ x : ℝ, 5^(x^2 + 2 * a * x + a^2) = a * x^2 + 2 * a^2 * x + a^3 + a^2 - 6 * a + 6) ↔ (a = 1) :=
sorry

end has_exactly_one_solution_l738_73876


namespace equality_of_arithmetic_sums_l738_73852

def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem equality_of_arithmetic_sums (n : ℕ) (h : n ≠ 0) :
  sum_arithmetic_sequence 8 4 n = sum_arithmetic_sequence 17 2 n ↔ n = 10 :=
by
  sorry

end equality_of_arithmetic_sums_l738_73852


namespace overall_class_average_proof_l738_73827

noncomputable def group_1_weighted_average := (0.40 * 80) + (0.60 * 80)
noncomputable def group_2_weighted_average := (0.30 * 60) + (0.70 * 60)
noncomputable def group_3_weighted_average := (0.50 * 40) + (0.50 * 40)
noncomputable def group_4_weighted_average := (0.20 * 50) + (0.80 * 50)

noncomputable def overall_class_average := (0.20 * group_1_weighted_average) + 
                                           (0.50 * group_2_weighted_average) + 
                                           (0.25 * group_3_weighted_average) + 
                                           (0.05 * group_4_weighted_average)

theorem overall_class_average_proof : overall_class_average = 58.5 :=
by 
  unfold overall_class_average
  unfold group_1_weighted_average
  unfold group_2_weighted_average
  unfold group_3_weighted_average
  unfold group_4_weighted_average
  -- now perform the arithmetic calculations
  sorry

end overall_class_average_proof_l738_73827


namespace unique_solution_l738_73812

def is_prime (p : ℕ) : Prop := ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem unique_solution (n : ℕ) :
  (0 < n ∧ is_prime (n + 1) ∧ is_prime (n + 3) ∧
   is_prime (n + 7) ∧ is_prime (n + 9) ∧
   is_prime (n + 13) ∧ is_prime (n + 15)) ↔ n = 4 :=
by
  sorry

end unique_solution_l738_73812


namespace exists_positive_integers_abc_l738_73819

theorem exists_positive_integers_abc (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_m_gt_one : 1 < m) (h_n_gt_one : 1 < n) :
  ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ m^a = 1 + n^b * c ∧ Nat.gcd c n = 1 :=
by
  sorry

end exists_positive_integers_abc_l738_73819


namespace distinct_terms_in_expansion_l738_73857

theorem distinct_terms_in_expansion :
  let n1 := 2 -- number of terms in (x + y)
  let n2 := 3 -- number of terms in (a + b + c)
  let n3 := 3 -- number of terms in (d + e + f)
  (n1 * n2 * n3) = 18 :=
by
  sorry

end distinct_terms_in_expansion_l738_73857


namespace functional_equation_solution_l738_73831

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y, f (f (f x)) + f (f y) = f y + x) → (∀ x, f x = x) :=
by
  intros f h x
  -- Proof goes here
  sorry

end functional_equation_solution_l738_73831


namespace triangle_BC_60_l738_73813

theorem triangle_BC_60 {A B C X : Type}
    (AB AC BX CX : ℕ) (h1 : AB = 70) (h2 : AC = 80) 
    (h3 : AB^2 - BX^2 = CX*(CX + BX)) 
    (h4 : BX % 7 = 0)
    (h5 : BX + CX = (BC : ℕ)) 
    (h6 : BC = 60) :
  BC = 60 := 
sorry

end triangle_BC_60_l738_73813


namespace check_true_propositions_l738_73882

open Set

theorem check_true_propositions : 
  ∀ (Prop1 Prop2 Prop3 : Prop),
    (Prop1 ↔ (∀ x : ℝ, x^2 > 0)) →
    (Prop2 ↔ ∃ x : ℝ, x^2 ≤ x) →
    (Prop3 ↔ ∀ (M N : Set ℝ) (x : ℝ), x ∈ (M ∩ N) → x ∈ M ∧ x ∈ N) →
    (¬Prop1 ∧ Prop2 ∧ Prop3) →
    (2 = 2) := sorry

end check_true_propositions_l738_73882


namespace total_members_in_sports_club_l738_73866

-- Definitions as per the conditions
def B : ℕ := 20 -- number of members who play badminton
def T : ℕ := 23 -- number of members who play tennis
def Both : ℕ := 7 -- number of members who play both badminton and tennis
def Neither : ℕ := 6 -- number of members who do not play either sport

-- Theorem statement to prove the correct answer
theorem total_members_in_sports_club : B + T - Both + Neither = 42 :=
by
  sorry

end total_members_in_sports_club_l738_73866


namespace alcohol_water_ratio_l738_73865

theorem alcohol_water_ratio (A W A_new W_new : ℝ) (ha1 : A / W = 4 / 3) (ha2: A = 5) (ha3: W_new = W + 7) : A / W_new = 1 / 2.15 :=
by
  sorry

end alcohol_water_ratio_l738_73865


namespace solve_system_of_inequalities_l738_73805

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x - 2 > 0) ∧ (3 * (x - 1) - 7 < -2 * x) → 1 < x ∧ x < 2 :=
by
  sorry

end solve_system_of_inequalities_l738_73805


namespace two_bishops_placement_l738_73891

theorem two_bishops_placement :
  let squares := 64
  let white_squares := 32
  let black_squares := 32
  let first_bishop_white_positions := 32
  let second_bishop_black_positions := 32 - 8
  first_bishop_white_positions * second_bishop_black_positions = 768 := by
  sorry

end two_bishops_placement_l738_73891


namespace sale_coupon_discount_l738_73863

theorem sale_coupon_discount
  (original_price : ℝ)
  (sale_price : ℝ)
  (price_after_coupon : ℝ)
  (h1 : sale_price = 0.5 * original_price)
  (h2 : price_after_coupon = 0.8 * sale_price) :
  (original_price - price_after_coupon) / original_price * 100 = 60 := by
sorry

end sale_coupon_discount_l738_73863


namespace fry_sausage_time_l738_73821

variable (time_per_sausage : ℕ)

noncomputable def time_for_sausages (sausages : ℕ) (tps : ℕ) : ℕ :=
  sausages * tps

noncomputable def time_for_eggs (eggs : ℕ) (minutes_per_egg : ℕ) : ℕ :=
  eggs * minutes_per_egg

noncomputable def total_time (time_sausages : ℕ) (time_eggs : ℕ) : ℕ :=
  time_sausages + time_eggs

theorem fry_sausage_time :
  let sausages := 3
  let eggs := 6
  let minutes_per_egg := 4
  let total_time_taken := 39
  total_time (time_for_sausages sausages time_per_sausage) (time_for_eggs eggs minutes_per_egg) = total_time_taken
  → time_per_sausage = 5 := by
  sorry

end fry_sausage_time_l738_73821


namespace Jims_apples_fits_into_average_l738_73848

def Jim_apples : Nat := 20
def Jane_apples : Nat := 60
def Jerry_apples : Nat := 40

def total_apples : Nat := Jim_apples + Jane_apples + Jerry_apples
def number_of_people : Nat := 3
def average_apples_per_person : Nat := total_apples / number_of_people

theorem Jims_apples_fits_into_average :
  average_apples_per_person / Jim_apples = 2 := by
  sorry

end Jims_apples_fits_into_average_l738_73848


namespace perimeter_shaded_region_l738_73871

-- Definitions based on conditions
def circle_radius : ℝ := 10
def central_angle : ℝ := 300

-- Statement: Perimeter of the shaded region
theorem perimeter_shaded_region 
  : (10 : ℝ) + (10 : ℝ) + ((5 / 6) * (2 * Real.pi * 10)) = (20 : ℝ) + (50 / 3) * Real.pi :=
by
  sorry

end perimeter_shaded_region_l738_73871


namespace Brandy_energy_drinks_l738_73801

theorem Brandy_energy_drinks 
  (maximum_safe_amount : ℕ)
  (caffeine_per_drink : ℕ)
  (extra_safe_caffeine : ℕ)
  (x : ℕ)
  (h1 : maximum_safe_amount = 500)
  (h2 : caffeine_per_drink = 120)
  (h3 : extra_safe_caffeine = 20)
  (h4 : caffeine_per_drink * x + extra_safe_caffeine = maximum_safe_amount) :
  x = 4 :=
by
  sorry

end Brandy_energy_drinks_l738_73801


namespace average_annual_growth_rate_in_2014_and_2015_l738_73892

noncomputable def average_annual_growth_rate (p2013 p2015 : ℝ) (x : ℝ) : Prop :=
  p2013 * (1 + x)^2 = p2015

theorem average_annual_growth_rate_in_2014_and_2015 :
  average_annual_growth_rate 6.4 10 0.25 :=
by
  unfold average_annual_growth_rate
  sorry

end average_annual_growth_rate_in_2014_and_2015_l738_73892
